"""WP-11 Phase G: cutting_research_v1 policy-ready env, action validation, privileged separation, dataset round trip.

Uses the real kinematic backend at coarse spacing (Warp CPU) so the wrapper is
exercised against the actual runtime contract; Native/Lab are NOT_RUN here.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from robotic_cutting_sim import dataset as ds
from robotic_cutting_sim.policy_env import (ACTION_STATUSES, DEPLOYABLE_RESET_CHECKS, PRIVILEGED_INFO_KEYS,
                                            PRIVILEGED_KEYS, ResearchPreset, check_tool_collision,
                                            find_privileged_info_keys, find_privileged_keys)
from robotic_cutting_sim.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]
TRIM = dict(scene="trim01_wp11_workcell", goal="trim_bump_rho10",
            planner_options={"max_rotation_deg": 1.0, "max_translation_m": 0.001})


# ----------------------------------------------------------------------------- validate_action (additive fix)

def test_validate_action_keeps_legacy_digit_range_and_accepts_explicit_max_cuts():
    task = make_task("CUT-01-R")
    pose = np.r_[0., 0., .05, 0., 0., 0., 1.]
    base = {"mode": "cartesian_edge_pose_world", "target": pose, "cut_enabled": True}
    assert task.validate_action({**base, "cut_index": 1})["cut_index"] == 1
    with pytest.raises(ValueError, match="0 or 1"):
        task.validate_action({**base, "cut_index": 2})          # legacy tasks: only 0/1 without max_cuts
    assert task.validate_action({**base, "cut_index": 14}, max_cuts=15)["cut_index"] == 14
    for bad in (15, -1, True, 2.0):
        with pytest.raises(ValueError):
            task.validate_action({**base, "cut_index": bad}, max_cuts=15)
    with pytest.raises(ValueError, match="max_cuts"):
        task.validate_action({**base, "cut_index": 0}, max_cuts=16)
    with pytest.raises(ValueError):
        task.validate_action({**base, "cut_index": 1}, max_cuts=1)


def test_configured_task_max_cuts_can_only_narrow():
    from robotic_cutting_sim.scene_config import compile_task
    task = compile_task("bar_three_bumps", "remove_all_protrusions", particle_spacing_m=.004)
    pose = np.r_[0., 0., .05, 0., 0., 0., 1.]
    action = {"mode": "cartesian_edge_pose_world", "target": pose, "cut_enabled": True, "cut_index": task.max_cuts-1}
    assert task.validate_action(action)["cut_index"] == task.max_cuts-1
    with pytest.raises(ValueError):
        task.validate_action(action, max_cuts=task.max_cuts-1)


@pytest.mark.cpu_sim
def test_kinematic_v5_registered_task_accepts_digits_up_to_max_cuts():
    pytest.importorskip("warp")
    from robotic_cutting_sim.kinematic_env import make_kinematic_env
    env = make_kinematic_env("CUT-01-R", particle_spacing_m=.004, material_dt=.0005, solver="tlmpm_v5", max_cuts=5,
                             cutting_law=8)
    observation, _ = env.reset(seed=1)
    pose = np.asarray(observation["proprio"]["tool_pose"], float)
    env.step({"mode": "cartesian_edge_pose_world", "target": pose, "cut_enabled": True, "cut_index": 4})
    with pytest.raises(ValueError):
        env.step({"mode": "cartesian_edge_pose_world", "target": pose, "cut_enabled": True, "cut_index": 5})
    env.close()


# ----------------------------------------------------------------------------- collision check (pure geometry)

def test_tool_collision_check_flags_fixture_and_table_penetration():
    from robotic_cutting_sim.scene_config import compile_task
    task = compile_task("trim01_wp11_workcell", "trim_bump_rho10", particle_spacing_m=.004)
    level = np.r_[0., 0., 0., 0., .08715574274765817, 0., .9961946980917454]
    high = level.copy()
    high[:3] = [-.026, 0., .16]
    assert check_tool_collision(task, high)["status"] == "PASS"
    inside_board = level.copy()
    inside_board[:3] = [-.05, 0., .1-.004]          # edge 4 mm inside the cutting board (world z 0.090..0.100)
    result = check_tool_collision(task, inside_board)
    assert result["status"] == "COLLISION_FIXTURE" and result["fixture"].startswith("support_floor")
    below_table = level.copy()
    below_table[:3] = [.40, 0., task.table_top_world_z_m-.01]   # outside every box, below the table top
    assert check_tool_collision(task, below_table)["status"] == "COLLISION_TABLE"


# ----------------------------------------------------------------------------- the policy-ready env (kinematic)

@pytest.fixture(scope="module")
def trim_env(tmp_path_factory):
    pytest.importorskip("warp")
    from robotic_cutting_sim.live_env import open_cutting_env
    out = tmp_path_factory.mktemp("policy_env")/"run"
    with open_cutting_env(runtime="kinematic", output_dir=out, preset="cutting_research_v1", particle_spacing_m=.004,
                          material_dt=.0005, device="cpu", **TRIM) as env:
        yield env


@pytest.mark.cpu_sim
def test_preset_defaults_and_reset_checks(trim_env):
    env = trim_env
    observation, info = env.reset(seed=42)
    preset = info["preset"]
    assert preset["preset_id"] == "cutting_research_v1" and preset["tool_attachment_mode"] == "physical_grasp"
    assert preset["tool_init_mode"] == "pregrasped" and preset["gripper_control_mode"] == "hold"
    checks = env.last_reset_checks          # the harness's full receipt (privileged fields included)
    for key in ("time_zero", "labels_clean", "oracle_history_single_sample", "lifecycle_ready_after_reset",
                "ik_failures_zero", "not_terminated", "control_steps_zero", "action_log_empty",
                "observation_timestamp_matches"):
        assert checks[key] is True, key
    assert checks["material_at_reference_max_m"] == 0. and info["max_cuts"] == 1
    assert info["reset_checks"] == {key: checks[key] for key in DEPLOYABLE_RESET_CHECKS if key in checks}
    assert env.task.task_id.startswith("CFG-trim01_wp11_workcell-trim_bump_rho10")
    assert observation["task"]["goal"]["goal_id"] == "trim_bump_rho10" and observation["task"]["goal_text"]
    assert observation["episode"]["state"] == "READY" and observation["timestamp_s"] == 0.


@pytest.mark.cpu_sim
def test_actions_get_explicit_status_and_rejections_are_never_executed(trim_env):
    env = trim_env
    observation, _ = env.reset(seed=42)
    start = observation["edge_command_world_xyzw"].copy()
    cases = [
        ({"type": "edge_delta", "target": [0.]*6, "frame": "world"}, "ACCEPTED"),
        ({"type": "edge_delta", "target": [0., 0., -.05, 0., 0., 0.], "frame": "world"}, "CLAMPED"),
        ({"type": "edge_pose", "target": [0, 0, 0, 0, 0, 0, 2]}, "REJECTED_INVALID_TARGET"),
        ({"type": "edge_delta", "target": [0.]*5}, "REJECTED_INVALID_TARGET"),
        ({"type": "edge_delta", "target": [0.]*6, "speed": 1}, "REJECTED_INVALID_SCHEMA"),
        ({"type": "edge_delta", "mode": "cartesian_edge_delta", "target": [0.]*6}, "REJECTED_INVALID_SCHEMA"),
        ({"type": "teleport", "target": [0.]*6}, "REJECTED_INVALID_SCHEMA"),
        ({"type": "edge_delta", "target": [0.]*6, "cut_index": 1}, "REJECTED_CUT_INDEX_OUT_OF_RANGE"),
        ({"type": "edge_delta", "target": [0.]*6, "cut_enabled": 1}, "REJECTED_INVALID_SCHEMA"),
        ({"type": "joint_position", "target": [0.]*7}, "REJECTED_UNSUPPORTED_BY_RUNTIME"),
        ({"gripper": "release"}, "REJECTED_GRIPPER_UNSUPPORTED"),
        ("not a mapping", "REJECTED_INVALID_SCHEMA"),
    ]
    for action, expected in cases:
        commanded = env.inner._commanded.copy()
        time_before = env.time_s
        _, _, _, _, info = env.step(action)
        assert info["action_status"] == expected, (action, info["action_status"], info["action_rejection"])
        assert info["action_status"] in ACTION_STATUSES
        assert abs(info["timestamp_s"]-info["action_timestamp_s"]-env.task.control_dt_s) < 1e-9
        assert info["action_timestamp_s"] == time_before
        if expected.startswith("REJECTED"):
            # Never executed: an explicit cut-disabled HOLD of the last accepted target runs instead.
            assert info["action_executed"]["kind"] == "HOLD_AFTER_REJECTION"
            assert info["action_executed"]["cut_enabled"] is False
            assert np.allclose(env.inner._commanded, commanded) and info["action_rejection"]["detail"]
        else:
            assert info["action_executed"]["kind"] == "REQUESTED"
    assert env.action_log[-1]["status"] == "REJECTED_INVALID_SCHEMA" and len(env.action_log) == len(cases)
    assert not np.allclose(env.inner._commanded, start)   # the CLAMPED step moved 1 mm


@pytest.mark.cpu_sim
def test_collision_rejection_and_no_step_policy(trim_env, tmp_path):
    env = trim_env
    env.reset(seed=42)
    # Drive the commanded edge into the cutting board beside the bar (board |y| <= 50 mm, bar |y| <= 15 mm).
    down = {"type": "edge_pose", "target": [-.05, .10, .09, 0., .08715574274765817, 0., .9961946980917454]}
    statuses = []
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(down)
        statuses.append(info["action_status"])
        if info["action_status"] == "REJECTED_COLLISION_FIXTURE":
            break
    assert statuses[-1] == "REJECTED_COLLISION_FIXTURE", statuses[-5:]
    assert info["collision_check"]["status"] == "COLLISION_FIXTURE"
    assert info["action_executed"]["kind"] == "HOLD_AFTER_REJECTION"
    env.preset = ResearchPreset(rejection_policy="no_step")
    try:
        time_before, steps_before = env.time_s, env.inner._control_steps_count
        _, reward, terminated, truncated, info = env.step({"type": "edge_pose", "target": [0, 0, 0, 0, 0, 0, 2]})
        assert info["action_executed"]["kind"] == "NOT_EXECUTED" and env.time_s == time_before
        assert env.inner._control_steps_count == steps_before and reward == 0.
    finally:
        env.preset = ResearchPreset()


@pytest.mark.cpu_sim
def test_deployable_observation_has_no_privileged_keys_and_privileged_needs_opt_in(trim_env, tmp_path):
    env = trim_env
    observation, _ = env.reset(seed=42)
    observation, _, _, _, info = env.step({"type": "edge_delta", "target": [0.]*6})
    assert find_privileged_keys(observation) == [] and "privileged" not in info
    assert not set(observation) & PRIVILEGED_KEYS and not set(observation["proprio"]) & PRIVILEGED_KEYS
    with pytest.raises(PermissionError):
        env.privileged()
    from robotic_cutting_sim.live_env import open_cutting_env
    with open_cutting_env(runtime="kinematic", output_dir=tmp_path/"priv", preset="cutting_research_v1",
                          particle_spacing_m=.004, material_dt=.0005, device="cpu",
                          observation_profile="privileged_state", **TRIM) as privileged_env:
        observation, _ = privileged_env.reset(seed=42)
        observation, _, _, _, info = privileged_env.step({"type": "edge_delta", "target": [0.]*6})
        assert find_privileged_keys(observation) == []                       # still deployable
        state = info["privileged"]
        assert {"material_state", "cut_codes", "oracle_geometry", "contact_wrench_world", "tool_truth"} <= set(state)
        labels = privileged_env.privileged(components=True)["component_labels"]
        assert labels["count"] == 1 and len(labels["labels"]) == len(state["cut_codes"])


@pytest.mark.cpu_sim
def test_reset_isolates_episodes(trim_env):
    env = trim_env
    _, first = env.reset(seed=42)
    first_full = dict(env.last_reset_checks)   # privileged reset checks live on the harness side, not in info
    for _ in range(5):
        env.step({"type": "edge_delta", "target": [0., 0., -.001, 0., 0., 0.], "frame": "world", "cut_enabled": True})
    _, second = env.reset(seed=42)
    for key in ("time_zero", "action_log_empty", "control_steps_zero", "oracle_history_single_sample"):
        assert second["reset_checks"][key] is True
    assert env.last_reset_checks["labels_clean"] is True
    assert env.last_reset_checks["material_codes_sha256"] == first_full["material_codes_sha256"]
    assert env.last_reset_checks["edge_pose_world_xyzw"] == first_full["edge_pose_world_xyzw"]
    assert second["episode_index"] == first["episode_index"]+1


# ----------------------------------------------------------------------------- info dicts (WP-11 review finding 9)

@pytest.mark.cpu_sim
def test_default_info_dicts_carry_no_privileged_key(trim_env):
    env = trim_env
    observation, info = env.reset(seed=42)
    assert find_privileged_info_keys(info) == [] and "privileged" not in info
    assert set(info["reset_checks"]) <= set(DEPLOYABLE_RESET_CHECKS)
    infos = [env.step(action)[4] for action in (
        {"type": "edge_delta", "target": [0.]*6}, {"type": "edge_delta", "target": [0., 0., -.05, 0., 0., 0.]},
        {"type": "edge_pose", "target": [0, 0, 0, 0, 0, 0, 2]}, {"gripper": "hold"})]
    for step_info in infos:
        assert find_privileged_info_keys(step_info) == [], step_info
        assert "grasp_status" not in step_info and "privileged" not in step_info
    # the walk itself finds privileged names at any depth
    assert find_privileged_info_keys({"a": [{"b": {"edge_pose_world_xyzw": 1}}], "grasp_status": "HOLDING"}) == [
        "/a[0]/b/edge_pose_world_xyzw", "/grasp_status"]
    assert PRIVILEGED_KEYS <= PRIVILEGED_INFO_KEYS


@pytest.mark.cpu_sim
def test_privileged_profile_moves_reset_checks_and_grasp_status_to_the_privileged_channel(tmp_path):
    from robotic_cutting_sim.live_env import open_cutting_env
    with open_cutting_env(runtime="kinematic", output_dir=tmp_path/"priv", preset="cutting_research_v1",
                          particle_spacing_m=.004, material_dt=.0005, device="cpu",
                          observation_profile="privileged_state", **TRIM) as env:
        _, info = env.reset(seed=42)
        assert find_privileged_info_keys(info) == []            # everything privileged is under info["privileged"]
        full = info["privileged"]["reset_checks"]
        assert full == env.last_reset_checks and "edge_pose_world_xyzw" in full and "material_at_reference_max_m" in full
        assert "tool_truth" in info["privileged"] and set(info["reset_checks"]) <= set(DEPLOYABLE_RESET_CHECKS)
        _, _, _, _, step_info = env.step({"type": "edge_delta", "target": [0.]*6})
        assert find_privileged_info_keys(step_info) == [] and "grasp_status" in step_info["privileged"]


# ----------------------------------------------------------------------------- action programs

def test_action_program_expansion_respects_speed_and_time_limit():
    program = json.loads((ROOT/"configs/wp11/action_programs/trim_rho10_handwritten_v1.json").read_text())
    rows = ds.expand_action_program(program, .02, 700)
    assert len(rows) == 700 and rows[-1][0] == "HOLD"
    poses = np.asarray([a["target"] for label, a in rows if a["type"] == "edge_pose"])
    steps = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    assert steps.max() <= .001+1e-9                       # every step within the task's 1 mm limit
    armed = [label for label, a in rows if a["cut_enabled"]]
    assert set(armed) == {"SKIM"} and len(armed) == 130
    with pytest.raises(ValueError, match="allows"):
        ds.expand_action_program(program, .02, 100)
    with pytest.raises(ValueError):
        ds.load_action_program({"schema": "rcs_action_program_v1", "segments": [{"kind": "teleport"}]})


def test_action_programs_do_not_use_the_planner():
    source = (ROOT/"scripts/make_phaseG_action_programs.py").read_text()
    assert "goal_planner" not in source.replace("no ``goal_planner``", "") and "policies" not in source.split('"""', 2)[2]


# ----------------------------------------------------------------------------- collector, reader, reliability

@pytest.mark.cpu_sim
def test_collector_index_reader_round_trip_and_replay(tmp_path):
    pytest.importorskip("warp")
    plan = {"schema": "rcs_episode_plan_v1", "episodes": [
        {"label": "program", "controller": "action_file",
         "action_file": str(ROOT/"configs/wp11/action_programs/sever_mid_handwritten_v1.json"), "seed": 3,
         "return_home": True},
        {"label": "hold", "controller": "policy", "policy": "robotic_cutting_sim.policy_examples:hold_policy",
         "seed": "random", "expect": "FAIL"},
    ]}
    (tmp_path/"plan.json").write_text(json.dumps(plan))
    out = tmp_path/"dataset"
    command = [sys.executable, str(ROOT/"scripts/collect_dataset.py"), "--runtime", "kinematic", "--scene",
               "bar_160x30x20", "--goal", "sever_mid", "--spacing", "0.004", "--material-dt", "0.0005",
               "--device", "cpu", "--episode-plan", str(tmp_path/"plan.json"), "--record-every", "20",
               "--out", str(out)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert completed.returncode in (0, 3), completed.stderr[-3000:]
    index = ds.load_index(out)
    assert [row["status"] for row in index["episodes"]] == ["STORED", "STORED"]
    for row in index["episodes"]:
        assert ds.validate_episode(out, row["episode_id"])["status"] == "PASS"
    first = ds.load_episode(out, 0)
    assert first["meta"]["frequencies"]["control_rate_hz"] == pytest.approx(50.)
    assert first["meta"]["home_outcome"]["status"] == "NOT_APPLICABLE"
    assert first["meta"]["controller_receipt"]["uses_planner"] is False
    assert not [key for key in first["arrays"] if any(part in PRIVILEGED_KEYS for part in key.split("__"))]
    assert ds.load_episode(out, 1)["meta"]["verdict"] == "FAIL"
    replay = ds.replay_kinematic(out, "episode_000")
    assert replay["status"] == "PASS" and replay["final_codes_mismatched"] == 0, replay
    report = json.loads((out/"reliability.json").read_text())
    assert report["cross_episode"]["failed_episode_stored_and_usable"] is True
    assert report["episodes"][1]["checks"]["expected_outcome"] is True
    reader = subprocess.run([sys.executable, str(ROOT/"scripts/read_dataset.py"), str(out)], cwd=ROOT,
                            capture_output=True, text=True, timeout=600)
    assert reader.returncode == 0 and "READ_DATASET PASS" in reader.stdout, reader.stdout[-2000:]


def _meta(i, *, codes="a", verdict="PASS", expect=None, rss=1000., gpu=None, leaks=0, controller="scripted"):
    reset = {"time_zero": True, "labels_clean": True, "material_at_reference_max_m": 0., "material_velocity_max_m_s": 0.,
             "material_codes_sha256": "z", "oracle_history_single_sample": True, "lifecycle_ready_after_reset": True,
             "ik_failures_zero": True, "not_terminated": True, "control_steps_zero": True, "action_log_empty": True,
             "observation_timestamp_matches": True, "tool_pose_world_xyzw": [0, 0, 0, 0, 0, 0, 1],
             "tool_linear_speed_m_s": 0., "tool_angular_speed_rad_s": 0., "grasp_reset_status": None,
             "camera_timestamp_matches": None}
    return {"episode_id": f"episode_{i:03d}", "label": str(i), "controller": controller, "seed": 42,
            "controller_receipt": {"kind": controller}, "verdict": verdict, "expect": expect, "reset_checks": reset,
            "clock_errors": 0, "observation_privileged_leaks": leaks, "recorder_closed": True,
            "final_codes_sha256": codes, "runtime_options": {"device": "cpu"},
            "memory_after": {"host_rss_mib": rss, "gpu_used_mib_total": gpu}, "preset": {"camera_every": 1},
            "control_dt_s": .02, "camera_age_max_s": 0.}


def test_reliability_report_gates_policy_ready():
    config = json.loads((ROOT/"configs/wp11/phaseG_reliability_v1.json").read_text())
    good = [_meta(i, rss=1000.+i) for i in range(5)]+[_meta(5, verdict="FAIL", expect="FAIL", controller="policy")]
    validations = [{"episode_id": m["episode_id"], "status": "PASS"} for m in good]
    assert ds.reliability_report(good, config, "kinematic", validations=validations)["POLICY_READY"] is True
    leak = [_meta(i, rss=1000.+300*i) for i in range(5)]+[_meta(5, verdict="FAIL", controller="policy", rss=2500.)]
    report = ds.reliability_report(leak, config, "kinematic", validations=validations)
    assert not report["POLICY_READY"] and "host_memory_no_leak" in report["failing"]
    drift = [_meta(i, codes=str(i)) for i in range(5)]+[_meta(5, verdict="FAIL", controller="policy")]
    assert "same_seed_same_controller_repeatable" in ds.reliability_report(drift, config, "kinematic",
                                                                           validations=validations)["failing"]
    privileged = good[:-1]+[_meta(5, verdict="FAIL", controller="policy", leaks=1)]
    assert "episode_005:no_privileged_keys_in_observations" in ds.reliability_report(
        privileged, config, "kinematic", validations=validations)["failing"]
    too_few = good[:2]+good[-1:]
    assert "consecutive_episodes_in_one_process" in ds.reliability_report(too_few, config, "kinematic",
                                                                          validations=validations)["failing"]
    no_failure = good[:5]
    assert "failed_episode_stored_and_usable" in ds.reliability_report(no_failure, config, "kinematic",
                                                                       validations=validations)["failing"]


def test_reliability_v1a_tool_velocity_rule_requires_repeat_not_absolute_rest():
    v1a = json.loads((ROOT/"configs/wp11/phaseG_reliability_v1a.json").read_text())
    assert v1a["amends"] == "phaseG_reliability_v1" and v1a["reset"]["tool_velocity_rule"] == "repeat_episode_0"
    metas = [_meta(i) for i in range(5)]+[_meta(5, verdict="FAIL", controller="policy")]
    for meta in metas:   # the Native smoke's clamped-body solver jitter, identical every reset
        meta["reset_checks"].update(tool_linear_speed_m_s=.0036, tool_angular_speed_rad_s=.148)
    validations = [{"episode_id": m["episode_id"], "status": "PASS"} for m in metas]
    v1 = json.loads((ROOT/"configs/wp11/phaseG_reliability_v1.json").read_text())
    assert "episode_000:tool_at_rest_after_reset" in ds.reliability_report(metas, v1, "kinematic",
                                                                           validations=validations)["failing"]
    assert ds.reliability_report(metas, v1a, "kinematic", validations=validations)["POLICY_READY"] is True
    metas[3]["reset_checks"]["tool_angular_speed_rad_s"] = .2
    assert "episode_003:tool_velocity_repeats_episode_0" in ds.reliability_report(
        metas, v1a, "kinematic", validations=validations)["failing"]
