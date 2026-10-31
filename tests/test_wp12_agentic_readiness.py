"""WP-12 Phase 6: agentic-control readiness (spec §6.1-6.5) on the real kinematic backend (CPU).

CPU only (Warp "cpu", 4 mm particles, material_dt 0.5 ms unless noted: development resolution, never an
acceptance result) on blade_master_flat_board_v1 scenes (tlmpm_v5, law 8) through the public
``open_cutting_env(..., preset="cutting_research_v1")`` API. The GPU/Native multi-episode runs are in
docs/reports/WP12_AGENTIC_READINESS.md.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import agentic_readiness as ar

ROOT = Path(__file__).resolve().parents[1]
CUT = dict(scene="configs/wp12/scenes/bar_flat_board.json", goal="sever_mid")
TRIM = dict(scene="configs/wp12/scenes/trim01_flat_board.json", goal="trim_bump_rho10",
            planner_options={"max_rotation_deg": 1.0, "max_translation_m": 0.001})
COARSE = dict(particle_spacing_m=.004, material_dt=.0005, device="cpu")


def _open(tmp, **kwargs):
    from robotic_cutting_sim.live_env import open_cutting_env
    options = dict(runtime="kinematic", output_dir=tmp/"run", preset="cutting_research_v1", seed=42)
    options.update(kwargs)
    return open_cutting_env(**options)


@pytest.fixture(scope="module")
def cut_env(tmp_path_factory):
    pytest.importorskip("warp")
    tmp = tmp_path_factory.mktemp("phase6")
    with _open(tmp, **CUT, **COARSE, surface_profile="cut_aware_surface_v1") as env:
        yield env


def _plan_policy(env):
    from robotic_cutting_sim import goal_planner
    from robotic_cutting_sim.policies import WaypointPolicy
    task = env.task
    options = task.controller["planner"]
    return WaypointPolicy(task, goal_planner.plan(task.scene, task.goal, task.tool_profile, options),
                          task.action_limits(), **goal_planner.policy_options(options))


# ----------------------------------------------------------------------------- 6.1 API

@pytest.mark.cpu_sim
def test_api_reset_step_evaluate_with_versioned_scene_surface_and_material_profiles(tmp_path):
    pytest.importorskip("warp")
    scene = json.loads((ROOT/CUT["scene"]).read_text())
    scene["scene_id"] = "bar_flat_board_soft_material"
    scene["workpiece"]["material_profile"] = "wp12_soft_elastic_v1"   # materials_wp12_v1 (versioned config)
    with _open(tmp_path, scene=scene, goal="sever_mid", particle_spacing_m=.004, material_dt=.0001, device="cpu",
               surface_profile="cut_aware_surface_v1") as env:
        observation, info = env.reset(seed=42)
        assert info["surface_profile"] == "cut_aware_surface_v1" and info["seed"] == 42
        assert env.task.scene["workpiece"]["material_profile"] == "wp12_soft_elastic_v1"
        assert env.backend.material.parameter_receipt["requested"]["young_modulus_pa"] != pytest.approx(5e4)
        assert ar.leak_paths(observation) == [] and ar.leak_paths(info) == []   # no material parameters leak
        spec = env.task_spec()
        assert spec["scene"]["supports"]["kind"] == "flat_board" and spec["goal"]["type"] == "sever"
        for _ in range(3):
            observation, reward, terminated, truncated, step_info = env.step(
                {"type": "edge_delta", "target": [0., 0., -.0005, 0., 0., 0.], "cut_enabled": True})
        assert step_info["action_status"] in ("ACCEPTED", "CLAMPED") and not terminated and not truncated
        assert ar.leak_paths(observation) == [] and ar.leak_paths(step_info) == []
        env.end_task("TEST")
        result = env.evaluate()
        # an episode ended before its time limit is never scored as a task result (integrity: complete_episode)
        checks = result["integrity"]["checks"]
        assert result["verdict"] == "FAIL_INTEGRITY" and checks["complete_episode"] is False
        assert all(ok for name, ok in checks.items() if name != "complete_episode")


def test_full_physics_v1_accepts_the_flat_board_tasks_at_the_formal_resolution():
    from robotic_cutting_sim import runtime_profiles as rp
    from robotic_cutting_sim.policy_env import compile_policy_task
    profile = rp.runtime_profile("full_physics_v1")
    assert profile["acceptance_valid"] is True
    for row in (CUT, TRIM):
        task = compile_policy_task(rp.profile_scene(row["scene"], profile), row["goal"],
                                   planner_options=row.get("planner_options"))
        assert rp.check_task(profile, task, profile["material_dt_s"]) == {
            "particle_spacing_m": .001, "grid_dx_m": .004, "control_dt_s": .02, "material_dt_s": .0001}
    fast = rp.runtime_profile("research_fast_v1")
    assert fast["acceptance_valid"] is False and fast["fidelity_limits"]["scope"].startswith("Policy iteration")


# ----------------------------------------------------------------------------- 6.2 observations

@pytest.mark.cpu_sim
def test_deployable_observation_contents_and_no_wp12_surface_material_geometry_leaks(cut_env):
    env = cut_env
    observation, info = env.reset(seed=42)
    for key in ("timestamp_s", "proprio", "ee_pose_world_xyzw", "edge_command_world_xyzw", "gripper", "cameras",
                "task", "episode"):
        assert key in observation
    assert observation["task"]["goal"]["type"] == "sever" and observation["task"]["goal_text"]
    assert observation["episode"]["state"] == "READY" and observation["episode"]["control_step"] == 0
    assert set(observation["gripper"]) >= {"available", "control_mode", "command", "width_m"}
    assert not env.cameras_available and observation["cameras"] == {}   # kinematic: no cameras (Native: 3 RGB-D)
    assert ar.leak_paths(observation) == [] and ar.leak_paths(info) == []
    policy = _plan_policy(env)
    for _ in range(220):   # into the armed CUT phase: codes, cut events and cut faces now exist
        observation, _, _, _, step_info = env.step(policy(observation))
        assert ar.leak_paths(observation) == [], ar.leak_paths(observation)
        assert ar.leak_paths(step_info) == [], ar.leak_paths(step_info)
        for key in ("action_status", "action_executed", "action_requested", "episode_state", "collision_check"):
            assert key in step_info
    assert env.backend.material.cut_events_metadata()["enabled"]
    assert np.count_nonzero(env.backend.material.history_state()["codes"]) > 0
    # the privileged channel exists only with the explicit opt-in profile
    with pytest.raises(PermissionError):
        env.privileged()
    with pytest.raises(PermissionError):
        env.surface_provenance()


def test_leak_walker_flags_every_wp12_layer_and_skips_the_privileged_channel():
    tree = {"proprio": {"q": [0.]}, "task": {"goal": {"type": "sever"}},
            "x": {"hermite": 1, "material_parameters": {"young_modulus_pa": 1.}, "lattice_report": {}},
            "privileged": {"cut_codes": 1}}
    found = ar.leak_paths(tree)
    assert "/x/hermite" in found and "/x/material_parameters" in found and "/x/material_parameters/young_modulus_pa" in found
    assert "/x/lattice_report" in found and not any(p.startswith("/privileged") for p in found)


# ----------------------------------------------------------------------------- 6.3 actions

@pytest.mark.cpu_sim
def test_action_interface_motion_orientation_cut_flags_index_rejection_reset_and_termination(cut_env):
    env = cut_env
    observation, _ = env.reset(seed=42)
    start = np.asarray(observation["edge_command_world_xyzw"], float)
    # Cartesian motion (absolute edge pose, bounded by the task's action limits)
    target = start.copy()
    target[2] += .0005
    observation, _, _, _, info = env.step({"type": "edge_pose", "target": target.tolist()})
    assert info["action_status"] == "ACCEPTED"
    assert np.allclose(observation["edge_command_world_xyzw"][:3], target[:3], atol=1e-12)
    # orientation control: absolute pose with a 0.5 deg rotation about world z
    rotated = target.copy()
    rotated[3:] = (Rotation.from_rotvec([0., 0., np.deg2rad(.5)])*Rotation.from_quat(target[3:])).as_quat()
    observation, _, _, _, info = env.step({"type": "edge_pose", "target": rotated.tolist()})
    angle = (Rotation.from_quat(observation["edge_command_world_xyzw"][3:])*Rotation.from_quat(target[3:]).inv()).magnitude()
    assert info["action_status"] == "ACCEPTED" and angle == pytest.approx(np.deg2rad(.5), abs=1e-9)
    # tool reorientation: an edge-frame rotation delta larger than one step is clamped, never exceeded
    observation, _, _, _, info = env.step({"type": "edge_delta", "target": [0., 0., 0., 0., .5, 0.], "frame": "edge"})
    assert info["action_status"] == "CLAMPED" and info["action_report"]["rotation_clamped"]
    # cut enable / disable and cut index (bar_flat_board keeps K = 4 digits)
    env.step({"type": "edge_delta", "target": [0.]*6, "cut_enabled": True, "cut_index": 3})
    env.step({"type": "edge_delta", "target": [0.]*6, "cut_enabled": False})
    rows = env.action_log[-2:]
    assert [r["cut_enabled"] for r in rows] == [True, False] and rows[0]["cut_index"] == 3
    # rejected actions are never executed: an explicit cut-disabled HOLD keeps the clock
    for name, action, expected in ar_probe_cases("kinematic"):
        before = np.asarray(env.observe()["edge_command_world_xyzw"], float).copy()
        observation, _, _, _, info = env.step(action)
        assert info["action_status"] == expected, name
        assert info["action_executed"]["kind"] == "HOLD_AFTER_REJECTION" and info["action_executed"]["cut_enabled"] is False
        assert np.array_equal(observation["edge_command_world_xyzw"], before)
    # failed action followed by a valid action
    observation, _, _, _, info = env.step({"type": "edge_delta", "target": [0., 0., .0002, 0., 0., 0.]})
    assert info["action_status"] == "ACCEPTED" and info["action_executed"]["kind"] == "REQUESTED"
    # safe reset in the middle of an episode, then termination
    observation, info = env.reset(seed=42)
    assert info["reset_checks"]["time_zero"] and info["reset_checks"]["action_log_empty"]
    assert np.array_equal(observation["edge_command_world_xyzw"], start)
    env.step({"type": "edge_delta", "target": [0.]*6})
    env.end_task("CONTROLLER_DONE")
    assert env.episode_state == "TASK_COMPLETE"
    with pytest.raises(RuntimeError):
        env.step({"type": "edge_delta", "target": [0.]*6})


def ar_probe_cases(runtime):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rcs_phase6_driver", ROOT/"scripts/run_wp12_phase6.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [(name, copy.deepcopy(action), expected[runtime] if isinstance(expected, dict) else expected)
            for name, action, expected in module.PROBES]


@pytest.mark.cpu_sim
def test_two_controllers_execute_the_same_scene_and_goal_through_the_public_api(cut_env):
    env = cut_env
    results = {}
    for controller in ("scripted", "independent_sequence"):
        observation, _ = env.reset(seed=42)
        if controller == "scripted":
            policy = _plan_policy(env)
            act = policy
        else:
            from robotic_cutting_sim import dataset as ds
            program = ar.independent_through_cut_program(env.task_spec(), observation["edge_command_world_xyzw"])
            rows = ds.expand_action_program(program, env.task.control_dt_s, int(round(env.task.duration_s/env.task.control_dt_s)))
            assert all(not str(label).startswith(("APPROACH", "PRE_CUT")) for label, _ in rows)   # not planner phases
            actions = iter(a for _, a in rows)
            act = (lambda observation: next(actions))
        truncated = False
        while not truncated:
            observation, _, terminated, truncated, info = env.step(act(observation))
            assert not terminated
        result = env.evaluate()
        results[controller] = result
        assert result["integrity"]["status"] == "PASS" and result["verdict"] in ("PASS", "FAIL")
        assert result["outcome"]["metrics"]["mechanical_components"] >= 1
    # both controllers sever the bar into the two declared pieces at this development resolution
    for controller, result in results.items():
        assert result["outcome"]["checks"]["expected_substantial_pieces"], (controller, result["outcome"]["metrics"])


# ----------------------------------------------------------------------------- 6.5 repeated episodes / task switch

@pytest.mark.cpu_sim
def test_task_switch_mixed_tasks_leave_no_stale_labels_topology_contacts_or_recorder(tmp_path):
    pytest.importorskip("warp")
    with _open(tmp_path, **CUT, **COARSE, surface_profile="cut_aware_surface_v1") as env:
        observation, info = env.reset(seed=42)
        assert "task_switch" not in info   # default reset info is unchanged
        first = ar.post_reset_state(env)
        cut_task_id = env.task.task_id
        policy = _plan_policy(env)
        for _ in range(220):
            observation, *_ = env.step(policy(observation))
        assert np.count_nonzero(env.backend.material.history_state()["codes"]) > 0
        receipt = env.switch_task(**TRIM, particle_spacing_m=.004)
        assert receipt["version"] == "task_switch_v1" and receipt["from_task_id"] == cut_task_id
        with pytest.raises(RuntimeError):
            env.step({"type": "edge_delta", "target": [0.]*6})   # a switch requires a reset first
        observation, info = env.reset(seed=42)
        assert info["task_switch"]["to_task_id"] == env.task.task_id != cut_task_id
        assert observation["task"]["goal"]["type"] == "trim_plane" or observation["task"]["goal"]["type"] == "remove_protrusions"
        trim = ar.post_reset_state(env, first)
        assert trim["failing"] == [], trim["checks"]
        assert trim["checks"]["new_material_object"] and trim["checks"]["new_surface_extractor"]
        assert trim["checks"]["surface_topology_fresh"] and trim["checks"]["cut_event_log_empty"]
        policy = _plan_policy(env)
        for _ in range(40):
            observation, *_ = env.step(policy(observation))
        env.switch_task(**CUT, particle_spacing_m=.004)
        observation, info = env.reset(seed=42)
        again = ar.post_reset_state(env, trim)
        assert again["failing"] == [] and env.task.task_id == cut_task_id
        assert again["codes_sha256"] == first["codes_sha256"] and again["particles"] == first["particles"]
        assert env.max_cuts == 4 and observation["task"]["max_cuts"] == 4
        # a reset without a switch carries no task_switch key again
        _, info = env.reset(seed=42)
        assert "task_switch" not in info


def test_switch_task_refuses_incompatible_requests(tmp_path):
    pytest.importorskip("warp")
    with _open(tmp_path, **CUT, **COARSE) as env:
        env.reset(seed=42)
        with pytest.raises(ValueError, match="one of them"):
            env.switch_task()
        goal = json.loads((ROOT/"configs/wp11/goals/sever_mid.json").read_text())
        goal["episode"]["control_dt_s"] = .01
        with pytest.raises(ValueError, match="control dt"):
            env.switch_task(scene=CUT["scene"], goal=goal)


@pytest.mark.cpu_sim
def test_phase6_driver_mixed_plan_stores_valid_episodes_and_replays_bitwise_on_cpu(tmp_path):
    pytest.importorskip("warp")
    import importlib.util
    spec = importlib.util.spec_from_file_location("rcs_phase6_driver_run", ROOT/"scripts/run_wp12_phase6.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    plan = {"schema": "rcs_wp12_phase6_plan_v1", "runtime": "kinematic", "material_dt": .0005, "device": "cpu",
            "surface_profile": "cut_aware_surface_v1", "record_every": 10, "seed": 42,
            "tasks": {"CUT": {"scene": CUT["scene"], "goal": "sever_mid"},
                      "TRIM": {"scene": TRIM["scene"], "goal": TRIM["goal"], "planner_options": TRIM["planner_options"]}},
            "episodes": [{"task": "CUT", "controller": "scripted"},
                         {"task": "TRIM", "controller": "probe_then_scripted"},
                         {"task": "CUT", "controller": "independent_sequence"},
                         {"task": "CUT", "controller": "scripted"}],
            "limits": {"memory_fit_from_episode": 1}}
    # 4 mm particles: the plan has no particle_spacing_m field, so patch the compile through planner-free scenes
    scenes = {}
    for name, row in plan["tasks"].items():
        scene = json.loads((ROOT/row["scene"]).read_text())
        scene["solver"]["particle_spacing_m"] = .004
        path = tmp_path/f"{name}_scene.json"
        path.write_text(json.dumps(scene))
        row["scene"] = str(path)
        scenes[name] = scene
    plan_path = tmp_path/"plan.json"
    plan_path.write_text(json.dumps(plan))
    out = tmp_path/"run"
    status = driver.main(["episodes", "--plan", str(plan_path), "--out", str(out), "--max-steps", "60"])
    report = json.loads((out/"phase6_report.json").read_text())
    assert status in (0, 3)
    for name in ("all_planned_episodes_stored", "every_episode_validates_dataset", "every_episode_validates_spec_6_4",
                 "no_privileged_keys_in_observations_or_info",
                 "no_wp12_surface_material_geometry_keys_in_observations_or_info", "no_clock_errors", "every_reset_clean",
                 "every_episode_evaluated", "probes_rejected_and_valid_action_follows", "repeat_groups"):
        assert report["checks"][name], (name, report["validations"], report["episodes"])
    assert [e["task"] for e in report["episodes"]] == ["CUT", "TRIM", "CUT", "CUT"]
    assert report["repeatability"] and report["repeatability"][0]["final_codes_identical"]
    record = json.loads((out/"episode_001"/"spec_6_4_validation.json").read_text())
    assert record["status"] == "PASS" and record["not_applicable"] == ["camera_observations"]
    assert record["items"]["surface_reconstruction_metadata"]["checks"]["recorder_log_hash_matches_file"]
    side = json.loads((out/"episode_001"/"phase6.json").read_text())
    assert side["task_switch"]["version"] == "task_switch_v1" and side["probe"]["all_rejected_as_declared"]
    assert ar.recorded_leak_walk(out/"episode_001")["status"] == "PASS"
    # CPU deterministic replay of a stored episode (codes bitwise)
    from robotic_cutting_sim import dataset as ds
    replay = ds.replay_kinematic(out, "episode_000")
    assert replay["status"] == "PASS" and replay["final_codes_mismatched"] == 0
    # the validator names a missing item
    (out/"episode_003"/"surface_provenance.json").unlink()
    broken = ar.validate_episode_dir(out/"episode_003")
    assert broken["status"] == "FAIL" and "surface_reconstruction_metadata" in broken["failing"]


@pytest.mark.cpu_sim
def test_kinematic_reset_clears_the_surface_cache_time(tmp_path):
    """Regression (found in Phase 6): an episode whose only extraction was at t = 0 left _surface_time == 0, so
    the first extraction after the next reset returned the cleared cache instead of the new episode's surface."""
    pytest.importorskip("warp")
    with _open(tmp_path, **CUT, **COARSE) as env:
        env.reset(seed=42)
        env.backend.surface   # extraction at t = 0
        env.step({"type": "edge_delta", "target": [0.]*6})   # no extraction at t > 0
        env.reset(seed=42)
        vertices, triangles = env.backend.surface
        assert len(vertices) > 0 and len(triangles) > 0
