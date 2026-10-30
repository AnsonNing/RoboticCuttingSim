"""WP-11 review fixes R3 (CPU): rejected actions never execute their gripper part, the collector never drops
the ADR 0016 relocation silently, ``slots_per_node`` needs tlmpm_v5, path_v3a stall wiring in the collector, and
the opt-in ``surface_unmeshable`` evaluation key (absent unless the backend reports a summary)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim.control import IKFailure
from robotic_cutting_sim.live_env import CuttingTaskEnv
from robotic_cutting_sim.policy_env import PolicyReadyEnv, ResearchPreset
from robotic_cutting_sim.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]
POSE = np.array([0., 0., .05, 0., 0., 0., 1.])
UNREACHABLE_Z = .5


def _array(data):
    return SimpleNamespace(numpy=lambda: np.asarray(data))


def _material():
    return SimpleNamespace(
        masses=np.ones(2), world_points=np.zeros((2, 3)), lattice=np.zeros((2, 3), int), max_cuts=1,
        world_state=lambda: {"positions": np.zeros((2, 3)), "velocities": np.zeros((2, 3)), "codes": np.zeros(2, np.int32)},
        bed=SimpleNamespace(nodes=_array(np.zeros((2, 8), int)), weights=_array(np.zeros((2, 8))),
                            gradients=_array(np.zeros((2, 8, 3)))))


def _grasp_backend():
    """Native-like fake: physical grasp, a gripper command, IK that fails before any physics for high targets."""
    actual = make_task("CUT-01")
    task = SimpleNamespace(task_id="CUT-01", duration_s=1., control_dt_s=.02, validate_action=actual.validate_action,
                           to_dict=actual.to_dict, hand_from_edge_xyz_xyzw=np.array([0., 0., 0., 0., 0., 0., 1.]),
                           goal=None, goal_kind="cut", scene=None, tool_profile_id=None, T_world_workpiece=np.eye(4))
    b = SimpleNamespace(task=task, time_s=0., material=_material(), gripper_command="hold", steps=[],
                        options=SimpleNamespace(tool_attachment_mode="physical_grasp"),
                        config=SimpleNamespace(camera=False), last_diagnostics={}, runtime_source="live_isaac_sim")

    def proprioception():
        return {"timestamp_s": b.time_s, "q": np.zeros((2, 7)), "qdot": np.zeros((2, 7)), "tool_pose": POSE.copy(),
                "hand_pose": POSE.copy(), "finger_q": np.full((2, 2), .011), "finger_qdot": np.zeros((2, 2)),
                "tool_attachment_mode": "physical_grasp"}

    def step(action):
        if action is not None and action["mode"] == "cartesian_edge_pose_world" and action["target"][2] > UNREACHABLE_Z-1e-9:
            raise IKFailure("IK failed before native articulation step: unreachable", {"position_error_m": 1.})
        b.steps.append(dict(target=np.asarray(action["target"]).copy(), gripper=b.gripper_command))
        b.time_s = round(b.time_s+.02, 9)

    def set_gripper(command):
        b.gripper_command = command

    b.reset = lambda seed: setattr(b, "time_s", 0.)
    b.step, b.set_gripper, b.proprioception = step, set_gripper, proprioception
    b.capture = lambda: ({}, {})
    b.calibration = {}
    b.close = lambda: None
    return b


def test_rejected_action_never_executes_its_gripper_release():
    backend = _grasp_backend()
    inner = CuttingTaskEnv(backend, observation_profile="state")
    env = PolicyReadyEnv(inner, ResearchPreset(gripper_control_mode="policy", collision_check=False))
    env.reset(seed=0)
    unreachable = POSE.copy()
    unreachable[2] = UNREACHABLE_Z
    _, _, _, _, info = env.step({"type": "edge_pose", "target": unreachable, "gripper": "release"})
    assert info["action_status"] == "REJECTED_IK_UNREACHABLE" and info["ik_status"] == "rejected"
    assert info["action_executed"]["kind"] == "HOLD_AFTER_REJECTION" and "gripper" not in info["action_executed"]
    assert backend.gripper_command == "hold"                     # the release part did not execute
    assert backend.steps[-1]["gripper"] == "hold" and np.allclose(backend.steps[-1]["target"], POSE)
    assert env.observe()["gripper"]["command"] == "hold"
    # control: the same release with a reachable target executes, gripper included
    _, _, _, _, info = env.step({"type": "edge_pose", "target": POSE, "gripper": "release"})
    assert info["action_status"] == "ACCEPTED" and info["action_executed"]["gripper"] == "release"
    assert backend.gripper_command == "release" and backend.steps[-1]["gripper"] == "release"


def test_legacy_env_without_preset_is_unchanged_on_ik_failure():
    """The legacy CuttingTaskEnv (no preset) still raises IKFailure with ik_failure='raise' (unchanged)."""
    backend = _grasp_backend()
    env = CuttingTaskEnv(backend, observation_profile="state")
    env.reset(seed=0)
    unreachable = POSE.copy()
    unreachable[2] = UNREACHABLE_Z
    with pytest.raises(IKFailure):
        env.step({"mode": "cartesian_edge_pose_world", "target": unreachable, "cut_enabled": False, "gripper": "release"})
    _, _, _, _, info = env.step({"mode": "cartesian_edge_pose_world", "target": POSE, "cut_enabled": False})
    assert "grasp_status" in info                               # legacy info keeps its keys


# ----------------------------------------------------------------------------- kinematic slots_per_node

def test_kinematic_slots_per_node_requires_tlmpm_v5():
    from robotic_cutting_sim.kinematic_env import KinematicConfig, make_kinematic_env
    with pytest.raises(ValueError, match="slots_per_node applies to solver='tlmpm_v5'"):
        KinematicConfig(slots_per_node=16)
    with pytest.raises(ValueError, match="tlmpm_v5"):
        make_kinematic_env("CUT-01-R", particle_spacing_m=.004, slots_per_node=16)
    assert KinematicConfig(solver="tlmpm_v5", slots_per_node=16, max_cuts=3).slots_per_node == 16
    assert KinematicConfig().slots_per_node is None


# ----------------------------------------------------------------------------- collector

def _collector():
    spec = importlib.util.spec_from_file_location("rcs_collect_dataset_r3", ROOT/"scripts/collect_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_collector(tmp_path, out, *extra):
    command = [sys.executable, str(ROOT/"scripts/collect_dataset.py"), "--runtime", "kinematic",
               "--scene", "trim01_wp11_workcell", "--goal", "trim_bump_rho10", "--spacing", "0.004",
               "--material-dt", "0.0005", "--device", "cpu", "--episodes", "1", "--out", str(out), *extra]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_collector_refuses_the_relocation_option_before_creating_output(tmp_path):
    out = tmp_path/"relocation"
    done = _run_collector(tmp_path, out, "--planner-options",
                          json.dumps({"controller_profile": "path_v2", "perception": "wrist_depth_relocate_v1"}))
    assert done.returncode != 0 and "perception" in done.stderr and "ADR 0016" in done.stderr
    assert not out.exists()                                     # nothing created, nothing launched


def test_collector_refuses_stall_options_without_a_bias_pose_profile(tmp_path):
    plan = {"schema": "rcs_episode_plan_v1", "episodes": [
        {"controller": "scripted", "seed": 1, "stall_timeout_s": 1., "max_steps": 5}]}
    (tmp_path/"plan.json").write_text(json.dumps(plan))
    out = tmp_path/"stall_v1"
    done = _run_collector(tmp_path, out, "--episode-plan", str(tmp_path/"plan.json"))
    assert done.returncode != 0 and "bias_pose" in done.stderr and not out.exists()


def test_scripted_controller_reports_a_stall_as_its_stop_reason():
    from robotic_cutting_sim.scene_config import compile_task
    collector = _collector()
    task = compile_task("trim01_wp11_workcell", "trim_bump_rho10_clear", planner_options={"controller_profile": "path_v3a"})
    from robotic_cutting_sim import goal_planner
    plan = goal_planner.plan(task.scene, task.goal, task.tool_profile, task.controller["planner"])
    stuck = np.asarray(plan[0].pose, float)                     # the measured edge never moves (blocked)
    clock = {"t": 0.}
    backend = SimpleNamespace(measured=lambda: {"timestamp_s": clock["t"], "tool_pose": stuck.copy()})
    env = SimpleNamespace(task=task, backend=backend, time_s=0.)
    entry = {"controller": "scripted", "edge_feedback": "measured", "stall_timeout_s": .5}
    act, done, receipt, privileged = collector.make_controller(env, entry)
    assert receipt["stall"]["stall_timeout_s"] == .5 and privileged
    finished = False
    for k in range(200):
        clock["t"] = env.time_s = .02*k
        act({"proprio": {"tool_pose": stuck.copy()}, "timestamp_s": clock["t"]})
        finished = done()
        if finished:
            break
    assert isinstance(finished, str) and finished.startswith("WAYPOINT_STALL:")
    assert receipt["stall"]["events"] and receipt["stall"]["events"][0]["reason"] == "WAYPOINT_STALL"
    # without stall options the controller is exactly the declared profile (no stall receipt)
    _, done, receipt, _ = collector.make_controller(env, {"controller": "scripted", "edge_feedback": "nominal"})
    assert "stall" not in receipt and done() is False


def test_make_controller_refuses_a_relocation_plan():
    collector = _collector()
    task = SimpleNamespace(controller={"planner": {"perception": "wrist_depth_relocate_v1"}})
    with pytest.raises(ValueError, match="ADR 0016"):
        collector.make_controller(SimpleNamespace(task=task), {"controller": "scripted"})


# ----------------------------------------------------------------------------- surface_unmeshable (R1 finding 2)

def _evaluation_backend(summary, configured):
    actual = make_task("CUT-01")
    task = SimpleNamespace(task_id="CUT-01", duration_s=.04, control_dt_s=.02, validate_action=actual.validate_action,
                           to_dict=actual.to_dict, particle_spacing_m=.004,
                           evaluate_oracle=lambda data: {"status": "FAIL_TASK_METRICS", "reason": "synthetic"})
    if configured:
        task.evaluate_goal = lambda data: {"status": "FAIL", "verdict": "FAIL", "reason": "synthetic goal"}
    b = SimpleNamespace(task=task, time_s=0., material=_material(), closed=False)
    b.proprioception = lambda: {"timestamp_s": b.time_s, "q": np.zeros((2, 7)), "qdot": np.zeros((2, 7)),
                                "tool_pose": POSE.copy()}
    b.reset = lambda seed: setattr(b, "time_s", 0.)
    b.step = lambda action: setattr(b, "time_s", round(b.time_s+.02, 9))
    b.capture, b.calibration, b.close = (lambda: ({}, {})), {}, (lambda: None)
    b.surface_arrays = (np.zeros((3, 3)), np.array([[0, 1, 2]]))
    b.surface_info = {"component_ranges": [{"first_triangle": 0, "triangles": 1, "mechanical_component": 0}],
                      "partition_rule": "synthetic"}
    if summary != "absent":
        b.surface_unmeshable_summary = lambda: summary
    return b


@pytest.mark.parametrize("configured", [False, True], ids=["oracle", "configured_goal"])
@pytest.mark.parametrize("summary", ["absent", None, {"cells": 3, "mode": "record"}], ids=["no_method", "none", "dict"])
def test_surface_unmeshable_is_reported_only_when_the_backend_has_a_summary(summary, configured):
    env = CuttingTaskEnv(_evaluation_backend(summary, configured), observation_profile="state")
    env.reset(seed=0)
    env.step({"mode": "cartesian_edge_pose_world", "target": POSE, "cut_enabled": False})
    result = env.evaluate()
    assert result["status"] == ("FAIL" if configured else "FAIL_TASK_METRICS")
    if isinstance(summary, dict):
        assert result["surface_unmeshable"] == summary
    else:
        assert "surface_unmeshable" not in result
    if configured:   # the configured-goal path (_evaluate_configured_goal) adds it exactly once, too
        assert result["surface_extraction"] == "MEASURED"
