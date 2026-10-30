"""WP-11 Phase D: wp11_workcell_v1, compiled tasks on the Isaac config, HOME, lifecycle and return-to-home.

CPU only. Isaac is never launched: the Isaac config/runner plumbing is checked
without SimulationApp, and the lifecycle/return-home tests use explicit fake
backends (not runtime validation). Tests that need the official Franka asset
(the real C2 checker and FK) are skipped when it is absent, and say so.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import episode_lifecycle as lc
from robotic_cutting_sim import goal_planner, task_scenarios as ts
from robotic_cutting_sim.control import ActionLimits
from robotic_cutting_sim.live_env import CuttingTaskEnv
from robotic_cutting_sim.scene_config import (SceneConfigError, compile_task, configured_task_payload,
                                              task_from_payload, workcell_home)
from robotic_cutting_sim.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]
FRANKA_ASSET = ROOT/"artifacts/workcell_asset_cache"
Q_HOME = (-0.0874749943614006, -0.3457081913948059, -0.1642773449420929, -2.275455951690674,
          -0.05930738523602486, 1.9334616661071777, 0.5642676949501038)


@pytest.fixture(scope="module")
def wp11_task():
    return compile_task("trim01_wp11_workcell", "trim_bump_rho10")


def _franka_asset_present():
    from robotic_cutting_sim.workcell import ASSETS
    return (FRANKA_ASSET/ASSETS["franka"]).is_file()


# ---------------------------------------------------------------- scenario and workcell

def test_wp11_scenario_loads_additively_and_v1_v3_scenarios_are_unchanged():
    all_scenarios = ts.scenarios()
    assert "wp11_workcell_v1" in all_scenarios
    v3 = json.loads((ROOT/"configs/wp10_scenarios_v3.json").read_text())["scenarios"]
    for name, config in v3.items():
        assert all_scenarios[name] == config
    blademaster, wp11 = all_scenarios["blademaster_workcell_v1"], all_scenarios["wp11_workcell_v1"]
    assert blademaster["world_translation_m"] == [0., 0., .05] and wp11["world_translation_m"] == [0., 0., .10]
    for key in ("cutting_board", "riser", "table", "fixture_mode", "extend_fixture_bases_to_table"):
        assert {k: v for k, v in wp11[key].items() if k != "scope"} == {k: v for k, v in blademaster[key].items() if k != "scope"} \
            if isinstance(wp11[key], dict) else wp11[key] == blademaster[key]
    assert ts.scenario_home("blademaster_workcell_v1") is None and ts.scenario_home(None) is None
    home = ts.scenario_home("wp11_workcell_v1")
    assert np.allclose(home["joint_positions"], Q_HOME, atol=0, rtol=0)
    assert np.allclose(np.round(home["joint_positions"], 4), [-.0875, -.3457, -.1643, -2.2755, -.0593, 1.9335, .5643])
    assert home["finger_position_m"] == .0112
    # hand pointing straight down (hand +z = world -z)
    assert np.allclose(Rotation.from_quat(home["hand_pose_world_xyzw"][3:]).apply([0, 0, 1]), [0, 0, -1], atol=1e-6)


def test_wp11_scenario_applies_to_v4_tasks_with_raised_board_and_riser():
    task = ts.apply_scenario(make_task("TRIM-02-K2-R10"), "wp11_workcell_v1")
    base = make_task("TRIM-02-K2-R10")
    assert np.allclose(task.world_from_workpiece_xyzw, np.r_[np.asarray(base.world_from_workpiece_xyzw[:3])+[0, 0, .1],
                                                             base.world_from_workpiece_xyzw[3:]])
    assert task.edge_orientation_local_xyzw == base.edge_orientation_local_xyzw  # v4 rake untouched
    board, riser = task.fixture_boxes_local
    assert (board.role, riser.role) == ("board", "riser")
    assert np.isclose(board.center_m[2]+board.half_extents_m[2], 0.)                      # same top surface
    assert np.isclose(riser.center_m[2]-riser.half_extents_m[2], task.table_top_world_z_m-.10)  # riser on the table
    assert np.isclose(riser.center_m[2]+riser.half_extents_m[2], board.center_m[2]-board.half_extents_m[2])
    assert board.static_friction == board.dynamic_friction == riser.static_friction == .3 == task.material.friction
    assert ts.table_material(task, "wp11_workcell_v1") == dict(static_friction=.3, dynamic_friction=.3, restitution=0.)
    for task_id in ("TRIM-01-R", "CUT-01-R"):
        with pytest.raises(ValueError, match="restricted"):
            ts.apply_scenario(make_task(task_id), "wp11_workcell_v1")
    # an existing scenario keeps its own allow list
    with pytest.raises(ValueError, match="restricted"):
        ts.apply_scenario(make_task("TRIM-02-K2-R10"), "blademaster_workcell_v1")


def test_scene_workcell_selects_wp11_with_physx_tlmpm_identical_boxes(wp11_task):
    v4 = ts.apply_scenario(make_task("TRIM-02-K2-R10"), "wp11_workcell_v1")
    assert wp11_task.world_from_workpiece_xyzw == v4.world_from_workpiece_xyzw
    assert wp11_task.fixture_boxes_local == v4.fixture_boxes_local      # identical MaterialFixtureBoxes
    assert wp11_task.table_contact_material == dict(static_friction=.3, dynamic_friction=.3, restitution=0.)
    from robotic_cutting_sim.native_fixture import fixture_authoring_plan
    plan = fixture_authoring_plan(wp11_task, None)
    assert plan["table"]["physics_material"]["static_friction"] == .3
    assert [row["physics_material"]["static_friction"] for row in plan["boxes"]] == [.3, .3]
    # a registered task without a scenario still authors no table material
    assert fixture_authoring_plan(make_task("TRIM-01-R"), None)["table"] is None
    assert wp11_task.home["scenario"] == "wp11_workcell_v1"
    assert (wp11_task.solver_module, wp11_task.cutting_law, wp11_task.max_cuts) == ("tlmpm_v5", 8, 1)


def test_planner_home_is_fk_q_home_with_the_blade_hanging_and_workspace_extended(wp11_task):
    home_edge = wp11_task.reference_edge_pose_world(0.)
    assert np.allclose(home_edge[:3], [0., 0., .288], atol=1e-5)
    edge_line = Rotation.from_quat(home_edge[3:]).apply([0., 1., 0.])
    assert np.allclose(edge_line, [0., 0., -1.], atol=1e-5)     # edge line along hand +z = world down
    expected = goal_planner.workcell_home_edge_pose(wp11_task.scene, wp11_task.tool_profile)
    assert np.allclose(home_edge, expected, atol=1e-9) or np.allclose(home_edge[:3], expected[:3], atol=1e-9)
    limits = wp11_task.action_limits()
    assert limits.workspace_max_local_m == (.14, .12, .22) and limits.workspace_min_local_m == ActionLimits().workspace_min_local_m
    labels = [label.split(":")[0] for label in wp11_task.reference_phase_labels]
    order = [label for i, label in enumerate(labels) if i == 0 or labels[i-1] != label]
    assert order == ["REORIENT", "PRE_CUT", "SKIM", "WITHDRAW", "RETURN_HOME", "DONE"]
    i = labels.index("SKIM")
    metrics = goal_planner.orientation_metrics(wp11_task.reference_edge_pose_world(wp11_task._times()[i])[3:],
                                               motion=[1., 0., 0.])
    assert abs(metrics["rho_deg"]-10.) < 1e-6 and metrics["tau_deg"] < 1e-6
    # the REORIENT end pose is in free space above the material (safe height)
    end = wp11_task.reference_edge_pose_local(wp11_task._times()[labels.index("PRE_CUT")])
    assert end[2] >= .028+goal_planner.DEFAULT_OPTIONS["safe_height_m"]-1e-9


# Compiled ids of every shipped scene x goal, computed at 4aef698/3c8da92 before Phase D:
# scenes without a HOME-declaring workcell must compile bit-identically.
PINNED_IDS = {
    ("bar_160x30x20", "sever_mid"): "CFG-bar_160x30x20-sever_mid-4e2c6f46",
    ("bar_160x30x20", "slice_then_push"): "CFG-bar_160x30x20-slice_then_push-fc8d724b",
    ("bar_160x30x20", "trim_plane_z20"): "CFG-bar_160x30x20-trim_plane_z20-3eeb4df2",
    ("bar_slotted_board_4slots", "divide_4"): "CFG-bar_slotted_board_4slots-divide_4-181adfd1",
    ("bar_slotted_board_4slots", "slice_then_push"): "CFG-bar_slotted_board_4slots-slice_then_push-a9a56659",
    ("bar_three_bumps", "remove_all_protrusions"): "CFG-bar_three_bumps-remove_all_protrusions-32fbd67c",
    ("bar_three_bumps", "divide_4"): "CFG-bar_three_bumps-divide_4-08f2ffcf",
    ("bar_three_bumps", "trim_plane_z20"): "CFG-bar_three_bumps-trim_plane_z20-9736ca43",
}


@pytest.mark.parametrize("scene,goal", sorted(PINNED_IDS))
def test_existing_configured_tasks_compile_unchanged(scene, goal):
    task = compile_task(scene, goal)
    assert task.task_id == PINNED_IDS[(scene, goal)]
    assert task.home is None and task.action_limits() == goal_planner.action_limits(task.controller["planner"])
    assert "home" not in goal_planner.plan_with_report(task.scene, task.goal, task.tool_profile, task.controller["planner"])[1]


def test_configured_payload_round_trip_and_tamper_detection(wp11_task):
    payload = configured_task_payload(wp11_task)
    again = task_from_payload(payload)
    assert again.task_id == wp11_task.task_id and again.config_sha256 == wp11_task.config_sha256
    assert again.edge_position_knots_local_txyz == wp11_task.edge_position_knots_local_txyz
    data = json.loads(payload)
    data["goal"]["planner"]["rake_deg"] = 12.
    with pytest.raises(SceneConfigError, match="differs"):
        task_from_payload(json.dumps(data))
    with pytest.raises(TypeError):
        configured_task_payload(make_task("TRIM-01-R"))


# ---------------------------------------------------------------- Isaac config plumbing (no launch)

def test_isaac_config_accepts_a_compiled_task_and_rejects_overrides(wp11_task):
    from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig
    common = dict(task_id=wp11_task.task_id, configured_task=configured_task_payload(wp11_task),
                  tool_attachment_mode="physical_grasp", max_cuts=1, camera_mode="native_single_view",
                  camera_layout="wrist_d405_global_d455")
    config = IsaacSimCuttingConfig(**common)
    assert config.velocity_feedforward and config.native_noise == "off" and config.surface_profile == "current_volume_v1"
    for change, message in ((dict(task_id="CFG-other"), "CFG"), (dict(tool_attachment_mode="legacy_fixed_tool"), "physical_grasp"),
                            (dict(scenario="wp11_workcell_v1"), "no runtime override"),
                            (dict(particle_spacing_m=.002), "no runtime override"),
                            (dict(initial_joint_positions=Q_HOME), "no runtime override"),
                            (dict(max_cuts=2), "max_cuts"), (dict(cutting_law=5), "cutting_law")):
        with pytest.raises(ValueError, match=message):
            IsaacSimCuttingConfig(**{**common, **change})
    assert IsaacSimCuttingConfig(**{**common, "cutting_law": 8}).cutting_law == 8
    # registered configs are untouched (the new field defaults to None)
    legacy = IsaacSimCuttingConfig(task_id="TRIM-01-R", tool_attachment_mode="physical_grasp")
    assert legacy.configured_task is None
    with pytest.raises(ValueError, match="task_id must be one of"):
        IsaacSimCuttingConfig(task_id=wp11_task.task_id, tool_attachment_mode="physical_grasp")


def test_tlmpm_v5_scene_up_to_its_digit_limit_is_accepted_by_the_isaac_config():
    from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig
    task = compile_task("bar_three_bumps", "remove_all_protrusions")
    config = IsaacSimCuttingConfig(task_id=task.task_id, configured_task=configured_task_payload(task),
                                   tool_attachment_mode="physical_grasp", max_cuts=3)
    assert config.max_cuts == 3


def test_open_cutting_env_passes_compiled_task_to_isaac_sim_and_lab_still_raises(tmp_path, wp11_task, monkeypatch):
    from robotic_cutting_sim import isaac_sim_runner
    from robotic_cutting_sim.live_env import open_cutting_env
    with pytest.raises(NotImplementedError, match="registered task ids"):
        with open_cutting_env(runtime="isaac_lab", output_dir=tmp_path/"lab", task=wp11_task):
            pass
    assert not (tmp_path/"lab").exists()
    seen = {}

    class Sentinel(Exception):
        pass

    def fake_launch(config, output):
        seen["config"] = config
        raise Sentinel()
    monkeypatch.setattr(isaac_sim_runner, "launch_app", fake_launch)
    with pytest.raises(Sentinel):
        with open_cutting_env(runtime="isaac_sim", output_dir=tmp_path/"sim", task=wp11_task, action_limits=wp11_task.action_limits()):
            pass
    config = seen["config"]
    assert config.task_id == wp11_task.task_id and config.max_cuts == 1 and config.tool_attachment_mode == "physical_grasp"
    assert task_from_payload(config.configured_task).task_id == wp11_task.task_id
    manifest = json.loads((tmp_path/"sim/manifest.json").read_text())
    assert manifest["config"]["configured_task"] == config.configured_task if "config" in manifest else True


# ---------------------------------------------------------------- state machine

def test_state_machine_legal_path_and_illegal_transitions():
    machine = lc.EpisodeStateMachine()
    with pytest.raises(lc.IllegalTransition):
        machine.transition("READY")
    for state in ("RESETTING", "READY", "RUNNING"):
        machine.transition(state)
    machine.transition("TASK_COMPLETE", reason="TIME_LIMIT")
    assert machine.snapshot()["termination_reason"] == "TIME_LIMIT"
    for state in ("RETRACTING", "RETURNING_HOME", "FINISHED"):
        machine.transition(state)
    for bad in ("RUNNING", "READY", "RETRACTING", "TASK_COMPLETE"):
        with pytest.raises(lc.IllegalTransition):
            machine.transition(bad)
    machine.transition("RESETTING")
    assert machine.snapshot()["termination_reason"] is None
    machine.transition("READY")
    for bad in ("TASK_COMPLETE", "RETRACTING", "RETURNING_HOME", "FINISHED"):
        with pytest.raises(lc.IllegalTransition):
            machine.transition(bad)
    with pytest.raises(lc.IllegalTransition):
        machine.transition("NOT_A_STATE")
    assert set(lc.EPISODE_STATES) == set(s for s in lc.TRANSITIONS if s) and all(
        set(targets) <= set(lc.EPISODE_STATES) for targets in lc.TRANSITIONS.values())


# ---------------------------------------------------------------- return-home planner (fake kinematics)

BASE = np.array([.3, 0., .1])


def fake_fk(q):
    q = np.asarray(q, float)
    return np.r_[BASE+.1*q[:3], Rotation.from_rotvec([0., 0., q[3]]).as_quat()]


def fake_ik(pose, seed):
    q = np.asarray(seed, float).copy()
    q[:3] = (np.asarray(pose[:3])-BASE)/.1
    q[3] = Rotation.from_quat(pose[3:]).as_rotvec()[2]
    error = fake_fk(q)
    return q, {"position_error_m": float(np.linalg.norm(error[:3]-pose[:3])),
               "rotation_error_rad": float((Rotation.from_quat(error[3:]).inv()*Rotation.from_quat(pose[3:])).magnitude())}


FAKE_HOME = np.array([0., 0., 1.5, .3, 0., 0., 0.])
FAKE_START = np.array([.2, .1, .2, -.5, .1, .1, .1])
LIMITS = dict(joint_lower=-np.full(7, 2.8), joint_upper=np.full(7, 2.8))


def passing_checker(calls):
    def check(qs, modes):
        calls.append((np.asarray(qs).copy(), list(modes)))
        return {"status": "PASS", "failing": []}
    return check


def test_trapezoid_respects_limits_and_ends_on_target():
    s = lc.trapezoid_profile(1., .6, 1.5, .02)
    v = np.diff(np.r_[0., s])/.02
    assert np.isclose(s[-1], 1.) and v.max() <= .6+1e-9 and np.all(np.diff(s) >= -1e-12)
    a = np.diff(np.r_[0., v, 0.])/.02
    assert np.abs(a).max() <= 1.5*(1+1e-6)+1e-9
    tiny = lc.trapezoid_profile(1e-4, .6, 1.5, .02)
    assert np.isclose(tiny[-1], 1e-4)
    with pytest.raises(ValueError):
        lc.trapezoid_profile(1., 0., 1., .02)


def test_return_home_plan_is_lift_then_hand_down_then_home_under_limits():
    calls = []
    config = lc.load_return_home_config()
    plan = lc.plan_return_home(FAKE_START, FAKE_HOME, edge_fk=fake_fk, solve_ik=fake_ik, control_dt=.02,
                               checker=passing_checker(calls), **LIMITS)
    assert plan["status"] == "PLANNED" and plan["reason"] is None and plan["variant"] == "rotate_then_move"
    q, labels = np.asarray(plan["q"]), plan["labels"]
    order = [label for i, label in enumerate(labels) if i == 0 or labels[i-1] != label]
    assert order == ["RETRACTING", "ROTATE_HAND_DOWN", "MOVE_HOME", "SETTLE_HOME"]
    assert np.allclose(q[-1], FAKE_HOME) and labels.count("SETTLE_HOME") == int(np.ceil(config["settle_s"]/.02))
    lift = q[np.array(labels) == "RETRACTING"]
    edges = np.array([fake_fk(x) for x in lift])
    assert np.allclose(edges[:, :2], fake_fk(FAKE_START)[:2]) and np.isclose(edges[-1, 2], fake_fk(FAKE_HOME)[2])
    assert np.allclose(edges[:, 3:], fake_fk(FAKE_START)[3:])           # orientation held during the lift
    rotated = fake_fk(q[np.array(labels) == "ROTATE_HAND_DOWN"][-1])
    assert np.allclose(rotated[3:], fake_fk(FAKE_HOME)[3:]) or np.allclose(rotated[3:], -fake_fk(FAKE_HOME)[3:])
    velocity, _ = lc.trajectory_limits(q, .02, FAKE_START)
    assert velocity <= config["joint_limits"]["max_velocity_rad_s"]+1e-9
    assert plan["attempts"][-1]["max_joint_space_acceleration_rad_s2"] <= config["joint_limits"]["max_acceleration_rad_s2"]*(1+1e-6)
    (checked, modes), = calls
    assert len(checked) == len(q)+1 and np.allclose(checked[0], FAKE_START)
    assert set(modes[:1+len(lift)]) == {"contact_free"} and set(modes[1+len(lift):]) == {"free"}


def test_return_home_preflight_rejection_falls_back_then_refuses():
    calls = []

    def reject_rotate(qs, modes):
        calls.append(len(qs))
        return {"status": "FAIL", "binding": "blade_collision"} if len(calls) == 1 else {"status": "PASS"}
    plan = lc.plan_return_home(FAKE_START, FAKE_HOME, edge_fk=fake_fk, solve_ik=fake_ik, control_dt=.02,
                               checker=reject_rotate, **LIMITS)
    assert plan["status"] == "PLANNED" and plan["variant"] == "direct"
    assert plan["attempts"][0]["reason"] == "PREFLIGHT_FAILED:blade_collision"
    refused = lc.plan_return_home(FAKE_START, FAKE_HOME, edge_fk=fake_fk, solve_ik=fake_ik, control_dt=.02,
                                  checker=lambda qs, modes: {"status": "FAIL", "binding": "franka1_collision"}, **LIMITS)
    assert refused["status"] == "REFUSED" and refused["reason"] == "PREFLIGHT_FAILED:franka1_collision" and not len(refused["q"])
    none = lc.plan_return_home(FAKE_START, FAKE_HOME, edge_fk=fake_fk, solve_ik=fake_ik, control_dt=.02, checker=None, **LIMITS)
    assert none["status"] == "REFUSED" and none["reason"] == "PREFLIGHT_NOT_RUN"


def test_return_home_plan_refuses_ik_failure_and_joint_limits():
    def bad_ik(pose, seed):
        return np.asarray(seed, float), {"position_error_m": .05, "rotation_error_rad": 0.}
    plan = lc.plan_return_home(FAKE_START, FAKE_HOME, edge_fk=fake_fk, solve_ik=bad_ik, control_dt=.02,
                               checker=passing_checker([]), **LIMITS)
    assert plan["status"] == "REFUSED" and plan["reason"] == "IK_FAILURE"
    outside = FAKE_HOME.copy()
    outside[4] = 2.79
    plan = lc.plan_return_home(FAKE_START, outside, edge_fk=fake_fk, solve_ik=fake_ik, control_dt=.02,
                               checker=passing_checker([]), **LIMITS)
    assert plan["status"] == "REFUSED" and plan["reason"] == "JOINT_LIMIT"
    fast = lc.load_return_home_config()
    fast["joint_limits"]["max_velocity_rad_s"] = .05     # the lift alone needs ~0.4 rad/s on joint 3
    plan = lc.plan_return_home(FAKE_START, FAKE_HOME, edge_fk=fake_fk, solve_ik=fake_ik, control_dt=.02,
                               checker=passing_checker([]), config=fast, **LIMITS)
    assert plan["status"] in ("PLANNED", "REFUSED")
    if plan["status"] == "PLANNED":   # retimed slower instead of exceeding the limit
        assert lc.trajectory_limits(plan["q"], .02, FAKE_START)[0] <= .05+1e-9


def test_home_outcome_is_separate_with_explicit_reasons():
    good = dict(plan_status="PLANNED", plan_reason=None, executed=True, aborted_reason=None, grasp_lost=False,
                home_joint_error_max_rad=.002, tool_translation_error_m=.001, tool_rotation_error_rad=.01,
                grasp_slip_max_m=.0005, grasp_rotation_max_rad=.005, label_changes=0, unintended_tool_contacts_max=0,
                measured_path_clearance_status="PASS")
    outcome = lc.evaluate_home_outcome(good)
    assert outcome["status"] == "PASS" and outcome["reason"] is None and "success" not in outcome
    assert lc.evaluate_home_outcome({**good, "label_changes": 3})["reason"] == "NO_LABEL_CHANGES"
    assert lc.evaluate_home_outcome({**good, "home_joint_error_max_rad": .05})["reason"] == "HOME_JOINT_ERROR"
    assert lc.evaluate_home_outcome({**good, "plan_status": "REFUSED", "plan_reason": "IK_FAILURE"})["reason"] == "PLAN_REFUSED:IK_FAILURE"
    assert lc.evaluate_home_outcome({**good, "executed": False, "aborted_reason": "GRASP_LOST",
                                     "grasp_lost": True})["reason"] == "ABORTED:GRASP_LOST"
    assert lc.evaluate_home_outcome({**good, "measured_path_clearance_status": "FAIL"})["status"] == "FAIL"
    assert lc.evaluate_home_outcome({**good, "grasp_slip_max_m": None})["status"] == "FAIL"   # missing evidence is not a pass


# ---------------------------------------------------------------- env integration with a fake robot backend

class FakeRobot:
    """Joint-position robot whose edge follows fake_fk exactly; physics-free, for wrapper contracts only."""
    runtime_source = "fake_robot_cpu_contract"

    def __init__(self, task, *, label_change_at=None, lose_grasp_at=None, plan=True):
        self.task, self.time_s, self.steps = task, 0., 0
        self.config = SimpleNamespace(robot_substeps=1, camera=False)
        self.options = SimpleNamespace(tool_attachment_mode="physical_grasp")
        self.home = {"joint_positions": tuple(FAKE_HOME), "scenario": "fake"}
        self.label_change_at, self.lose_grasp_at = label_change_at, lose_grasp_at
        self.commands, self.grasp_trace = [], []
        array = lambda data: SimpleNamespace(numpy=lambda: np.asarray(data))
        self.codes = np.zeros(4, np.int32)
        self.material = SimpleNamespace(
            masses=np.ones(4), world_points=np.zeros((4, 3)), max_cuts=1,
            history_state=lambda: {"positions": np.zeros((4, 3)), "velocities": np.zeros((4, 3)), "codes": self.codes.copy()},
            bed=SimpleNamespace(nodes=array(np.zeros((4, 8), int)), weights=array(np.zeros((4, 8))),
                                gradients=array(np.zeros((4, 8, 3)))))
        self.surface = (np.zeros((3, 3)), np.array([[0, 1, 2]]))
        self.surface_metadata = {"component_ranges": [{"first_triangle": 0, "triangles": 1, "mechanical_component": 0}],
                                 "partition_rule": "Synthetic CPU unit fixture"}
        self.calibration = {}
        if plan:
            self.plan_return_home = lambda q: lc.plan_return_home(q, FAKE_HOME, edge_fk=fake_fk, solve_ik=fake_ik,
                                                                  control_dt=.02, checker=passing_checker([]), **LIMITS)
            self.check_measured_path = lambda qs, modes: {"status": "PASS", "samples": len(qs)}

    def reset(self, seed):
        self.q, self.time_s, self.steps = FAKE_HOME.copy(), 0., 0   # reset at HOME, as the runner does
        self.codes[:] = 0
        self.last_diagnostics = {}

    def proprioception(self):
        edge = fake_fk(self.q)
        return {"timestamp_s": self.time_s, "q": np.stack([self.q, np.zeros(7)]), "qdot": np.zeros((2, 7)),
                "tool_pose": edge, "tool_body_pose_xyzw": edge}

    def step(self, action):
        self.commands.append(action)
        if action is not None and action["mode"] == "joint_position_rad":
            self.q = np.asarray(action["target"], float).copy()
        self.steps += 1
        self.time_s = round(self.time_s+.02, 9)
        if self.label_change_at is not None and self.steps == self.label_change_at:
            self.codes[0] = 1
        lost = self.lose_grasp_at is not None and self.steps >= self.lose_grasp_at
        self.grasp_trace.append({"contacts": {"unintended_contact_count": 0, "tensor_forces": {"records": []}}})
        self.last_diagnostics = {"grasp_lost": lost, "next_material_wrench_world": np.zeros(6),
                                 "grasp_status": {"status": "LOST" if lost else "HOLDING", "slip_since_settling_m": 1e-4,
                                                  "rotation_since_settling_rad": 1e-3}}


def fake_task(duration=.1):
    real = make_task("CUT-01")
    return SimpleNamespace(task_id="FAKE", seed=3, duration_s=duration, control_dt_s=.02, validate_action=real.validate_action,
                           to_dict=lambda: {"protocol_sha256": "fake"}, T_world_workpiece=np.eye(4),
                           evaluate_oracle=lambda data: {"status": "FAIL_TASK_METRICS", "reason": "Synthetic uncut fixture"})


def test_env_lifecycle_info_and_opt_in_observation():
    backend = FakeRobot(fake_task(.06))
    env = CuttingTaskEnv(backend, observation_profile="state")
    with pytest.raises(RuntimeError, match="reset"):
        env.end_task()                       # nothing before reset
    obs, info = env.reset()
    assert info["episode_state"]["state"] == "READY" and "episode" not in obs
    with pytest.raises(ValueError):
        env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False, "cut_index": 5})
    assert env.episode_state == "READY"      # a rejected action executes nothing and changes no state
    _, _, _, truncated, info = env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    assert info["episode_state"]["state"] == "RUNNING" and not truncated
    env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    _, _, terminated, truncated, info = env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    assert truncated and not terminated and info["episode_state"] == {"state": "TASK_COMPLETE", "termination_reason": "TIME_LIMIT",
                                                                     "home_outcome": None, "transitions": 4}
    with pytest.raises(RuntimeError, match="Episode ended"):
        env.step()
    env.reset()
    exposed = CuttingTaskEnv(FakeRobot(fake_task()), observation_profile="state", expose_episode_state=True)
    obs, _ = exposed.reset()
    assert obs["episode"]["state"] == "READY" and set(obs) == {"timestamp_s", "proprio", "episode"}
    assert "force" not in json.dumps(obs["episode"]) and "codes" not in obs["episode"]


def test_env_end_task_then_return_home_pass_separate_from_task_outcome():
    backend = FakeRobot(fake_task(1.))
    env = CuttingTaskEnv(backend, observation_profile="state")
    env.reset()
    env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    env.end_task("CONTROLLER_DONE")
    times = [row["time_s"] for row in env._history]
    assert times == sorted(set(times))       # end_task never records a duplicate timestamp
    with pytest.raises(lc.IllegalTransition, match="step is not allowed"):
        env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    seen = []
    result = env.return_home(on_step=lambda action, res, before, label: seen.append((label, res[4]["episode_state"]["state"])))
    home = result["home_outcome"]
    assert home["status"] == "PASS", home
    assert env.episode_state == "FINISHED" and env.lifecycle.home_outcome == "PASS"
    states = [row["to"] for row in env.lifecycle.history]
    assert states == ["RESETTING", "READY", "RUNNING", "TASK_COMPLETE", "RETRACTING", "RETURNING_HOME", "FINISHED"]
    assert seen[0] == ("RETRACTING", "RETRACTING") and seen[-1] == ("SETTLE_HOME", "RETURNING_HOME")
    assert all(not command["cut_enabled"] and command["mode"] == "joint_position_rad" for command in backend.commands[1:])
    assert result["measurement"]["home_joint_error_max_rad"] < 1e-12 and result["measurement"]["label_changes"] == 0
    assert result["measurement"]["executed"] and result["measurement"]["steps"] == len(result["labels"])
    task = env.evaluate()        # the task outcome is independent and unchanged by the home outcome
    assert task["status"] == "FAIL_TASK_METRICS" and not task["success"] and "home_outcome" not in task
    with pytest.raises(lc.IllegalTransition):
        env.return_home()
    env.reset()
    assert env.episode_state == "READY" and env.lifecycle.home_outcome is None


def test_end_task_with_history_stride_records_the_final_state_once():
    env = CuttingTaskEnv(FakeRobot(fake_task(1.)), observation_profile="state", history_stride=4)
    env.reset()
    for _ in range(3):
        env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    env.end_task()
    times = [row["time_s"] for row in env._history]
    assert times == [0., .06] and np.all(np.diff(times) > 0)


def test_return_home_failures_label_change_grasp_loss_and_refusal():
    env = CuttingTaskEnv(FakeRobot(fake_task(1.), label_change_at=5), observation_profile="state")
    env.reset()
    env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    result = env.return_home()           # RUNNING -> implicit end_task
    assert result["home_outcome"]["status"] == "FAIL" and result["home_outcome"]["reason"] == "NO_LABEL_CHANGES"
    assert env.lifecycle.termination_reason == "RETURN_HOME_REQUESTED"
    backend = FakeRobot(fake_task(1.), lose_grasp_at=8)
    env = CuttingTaskEnv(backend, observation_profile="state")
    env.reset()
    env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    env.end_task()
    result = env.return_home()
    assert result["home_outcome"]["reason"] == "ABORTED:GRASP_LOST" and backend.steps == 7+1
    backend = FakeRobot(fake_task(1.))
    backend.plan_return_home = lambda q: {"status": "REFUSED", "reason": "IK_FAILURE", "q": [], "labels": []}
    env = CuttingTaskEnv(backend, observation_profile="state")
    env.reset()
    env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    env.end_task()
    result = env.return_home()
    assert result["home_outcome"]["reason"] == "PLAN_REFUSED:IK_FAILURE" and backend.steps == 1   # nothing moved
    no_home = FakeRobot(fake_task(1.), plan=False)
    no_home.home = None
    env = CuttingTaskEnv(no_home, observation_profile="state")
    env.reset()
    env.step({"mode": "joint_position_rad", "target": FAKE_START, "cut_enabled": False})
    env.end_task()
    assert env.return_home()["home_outcome"]["status"] == "NOT_APPLICABLE" and env.episode_state == "FINISHED"


# ---------------------------------------------------------------- the real C2 checker and FK (asset needed)

@pytest.mark.skipif(not _franka_asset_present(), reason="NOT_RUN: official Franka asset cache absent (artifacts/workcell_asset_cache)")
def test_declared_home_hand_pose_matches_newton_fk_and_plan_preflight_passes(wp11_task):
    pytest.importorskip("mujoco")
    from robotic_cutting_sim import motion_preflight as mp
    checker = mp.MotionChecker(wp11_task)
    edge = checker.edge_fk(Q_HOME)
    assert np.allclose(edge[:3], wp11_task.reference_edge_pose_world(0.)[:3], atol=1e-5)
    receipt = mp.preflight_plan(wp11_task, Q_HOME, until_phase="REORIENT", checker=checker)
    assert receipt["status"] == "PASS" and receipt["samples_by_phase"] == {"REORIENT": receipt["samples"]}
    assert receipt["path"]["worst"]["blade_collision"]["value"] > .02   # REORIENT stays well clear of the bump
    plan = mp.plan_return_home_for_task(wp11_task, np.asarray(receipt["q_until_end"]), checker=checker)
    assert plan["status"] == "PLANNED" and np.allclose(plan["q"][-1], Q_HOME)
    # a path into the table is rejected by the same checker
    down = np.asarray(receipt["q_until_end"]).copy()
    bad = [down+(np.array([0., .9, 0., .9, 0., 0., 0.])*t) for t in np.linspace(0, 1, 15)]
    rejected = checker.check_joint_path(bad, ["free"]*len(bad))
    assert rejected["status"] == "FAIL" and rejected["failing"]
