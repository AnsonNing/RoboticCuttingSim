"""ADR 0016 wrist-depth relocation hook (``planner_options.perception``) on CPU with synthetic depth.

The synthetic renderer casts the wp11 wrist D405_0 geometry measured on the live_001
SLICE-PUSH s1301 episode (K of the 848x480 depth stream; camera 113 mm above, 174 mm
-y and 45 mm -x of the knife edge, optical axis along the edge line) against boxes
in the workpiece frame: the rigid bar parts, the knife blade and the board plane.
"""
import importlib.util
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import generalization as G
from robotic_cutting_sim import goal_planner
from robotic_cutting_sim import perception_relocate as hook
from robotic_cutting_sim.policies import WaypointPolicy
from robotic_cutting_sim.scene_config import SceneConfigError, compile_task, scene_world_pose

ROOT = Path(__file__).resolve().parents[1]
LIVE = json.loads((ROOT/"configs/wp11/generalization_round1_live.json").read_text())
OVERRIDES = LIVE["controller"]["planner_overrides"]
HOOKED = {**OVERRIDES, "perception": hook.HOOK}
K = np.array([[446.8027594274324, 0., 424.], [0., 446.8027594274324, 240.], [0., 0., 1.]])
T_EDGE_CAMERA = np.eye(4)
T_EDGE_CAMERA[:3, :3] = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
T_EDGE_CAMERA[:3, 3] = [-.112975, -.174387, -.045016]
U0 = 560                     # column crop of the 848-wide image (the bar lies in the last ~150 columns)


@pytest.fixture(scope="module")
def samples():
    config = G.load_round()
    out = {}
    for seed in (1301, 1302):
        _, info = G.compile_sample(config, G.build_sample(config, "SLICE_PUSH", seed))
        out[seed] = (info["scene"], info["goal"])
    return out


def _matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def _slab(origin, rays, low, high):
    """Entry distance of each ray into an axis-aligned box (inf on a miss)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (low-origin)/rays, (high-origin)/rays
    near = np.nanmax(np.minimum(t1, t2), axis=-1)
    far = np.nanmin(np.maximum(t1, t2), axis=-1)
    return np.where((far >= near) & (far > 0), np.maximum(near, 0.), np.inf)


def render(scene, edge_world, parts, *, time_s=0., noise=0., seed=0):
    """Wrist depth frame for an edge pose over rigid bar parts [(x0, x1), ...] (local x, rendered faces)."""
    T_world_local = _matrix(scene_world_pose(scene))
    T_world_camera = _matrix(edge_world)@T_EDGE_CAMERA
    T_local_camera = np.linalg.inv(T_world_local)@T_world_camera
    Kc = K.copy()
    Kc[0, 2] -= U0
    v, u = np.mgrid[:480, :848-U0]
    rays_camera = np.stack([(u-Kc[0, 2])/Kc[0, 0], (v-Kc[1, 2])/Kc[1, 1], np.ones(u.shape)], -1)
    rays = rays_camera@T_local_camera[:3, :3].T            # local direction per unit optical depth
    origin = T_local_camera[:3, 3]
    base = scene["workpiece"]["base_box_m"]
    W, H = base["width_m"], base["height_m"]
    hits = [np.where(rays[..., 2] < 0, -origin[2]/rays[..., 2], np.inf)]   # board top plane z = 0
    for x0, x1 in parts:
        hits.append(_slab(origin, rays, np.array([x0, -W/2, 0.]), np.array([x1, W/2, H])))
    edge_local = np.linalg.inv(T_world_local)@np.r_[edge_world[:3], 1.]
    hits.append(_slab(origin, rays, np.array([edge_local[0]-.00075, -.12, edge_local[2]]),
                      np.array([edge_local[0]+.00075, .12, edge_local[2]+.04])))
    depth = np.min(hits, axis=0)
    if noise:
        depth = depth+np.random.default_rng(seed).normal(0., noise, depth.shape)
    valid = np.isfinite(depth) & (depth >= .07) & (depth <= .5)
    return dict(depth_m=np.where(valid, depth, 0.).astype(np.float32), valid_mask=valid, K=Kc,
                T_world_camera=T_world_camera, physics_timestamp_s=float(time_s), camera_id=hook.CAMERA_ID)


def _plan(scene, goal, overrides):
    return goal_planner.plan_with_report(scene, goal, _tool(), goal_planner.planner_options(goal, overrides))


def _tool():
    from robotic_cutting_sim.tool_profile import PROFILES
    return PROFILES["ruler_knife_v2_y"]


# ------------------------------------------------------------------------------------- default inert

def test_hook_is_inert_when_not_requested(samples):
    scene, goal = samples[1301]
    plan, report = _plan(scene, goal, OVERRIDES)
    assert "perception" not in report and "perception" not in report["options"]
    assert not any(w.label.split(":", 1)[0].startswith("SURVEY") for w in plan)
    # The live_001 task id (declared before the hook existed) is reproduced exactly.
    live_goal = deepcopy(goal)
    live_goal["episode"]["duration_s"] = 29.0
    declared = next(s for s in LIVE["samples"] if s["sample_id"] == "E1_SLICE_PUSH_s1301")["declared_live_task"]
    assert compile_task(scene, live_goal, planner_options=OVERRIDES).task_id == declared["live_task_id_at_declaration_cpu"]
    assert "perception" in goal_planner.OPT_IN_OPTIONS and "perception" not in goal_planner.DEFAULT_OPTIONS


def test_hooked_plan_is_the_default_plan_plus_survey_waypoints(samples):
    for seed, (scene, goal) in samples.items():
        default, _ = _plan(scene, goal, OVERRIDES)
        plan, report = _plan(scene, goal, HOOKED)
        rest = [w for w in plan if not w.label.split(":", 1)[0].startswith("SURVEY")]
        assert rest == default, seed                                 # every other waypoint identical
        labels = [w.label for w in plan]
        assert labels[1:5] == ["SURVEY_APPROACH:survey0"]*2+["SURVEY:survey0", "SURVEY_LEAVE:survey0"]
        withdraw = max(i for i, label in enumerate(labels) if label == "WITHDRAW:cut0")
        assert labels[withdraw+1:withdraw+5] == ["SURVEY_APPROACH:survey1"]*2+["SURVEY:survey1", "SURVEY_LEAVE:survey1"]
        assert labels[withdraw+5] == "APPROACH:cut1"
        block = report["perception"]
        assert block["hook"] == hook.HOOK and block["side"] == 1 and block["surveys"] == [{"op": "survey1",
                                                                                          "next_cut_op": "cut1"}]
        base = scene["workpiece"]["base_box_m"]
        expected = np.array([base["length_m"]/2+.01, 0., base["height_m"]+.008])
        assert np.allclose(block["survey_pose_local_m"], expected)
        T = _matrix(scene_world_pose(scene))
        cut = next(w for w in plan if w.label == "CUT:cut1")
        for label in ("SURVEY:survey0", "SURVEY:survey1"):
            w = next(w for w in plan if w.label == label)
            assert np.allclose(w.pose[:3], (T@np.r_[expected, 1.])[:3]) and np.allclose(w.pose[3:], cut.pose[3:])
            assert w.hold_s == .3 and w.speed_m_s == .02 and not w.cut
        events = hook.events_from_plan(plan, block)
        shifted = {labels[i] for i in events[1]["shifted"]}
        assert "RETURN_HOME" not in shifted and not any(s.startswith("SURVEY") for s in shifted)
        assert {"APPROACH:cut1", "CUT:cut1", "LOWER:move1", "PUSH:move1", "RETRACT:move1"} <= shifted
        assert not shifted & {"CUT:cut0", "HOME"}


def test_eligibility_errors(samples):
    scene, goal = samples[1301]
    sever = deepcopy(goal["steps"][0])
    for bad, match in (({**HOOKED, "perception": "look_at_v0"}, "perception must be one of"),
                       ({**HOOKED, "controller_profile": "path_v3"}, "path_v2 only"),
                       ({**HOOKED, "controller_profile": "v1"}, "path_v2 only")):
        with pytest.raises(SceneConfigError, match=match):
            _plan(scene, goal, bad)
    single = {**sever, "planes": sever["planes"][:1], "expected_pieces": 2}
    reversed_middle = {**sever, "planes": [sever["planes"][0], sever["planes"][1], {**sever["planes"][0],
                       "origin_local_m": [0., 0., 0.]}], "expected_pieces": 4}
    approach = {**sever, "planes": [{**row, "approach_local": [0., 0., -1.]} for row in sever["planes"]]}
    wrap = lambda row, name: {**row, "goal_id": name, "schema": "rcs_goal_v1", "episode": goal["episode"],
                              "planner": goal["planner"]}
    for bad, match in ((single, ">= 2 planes"), (reversed_middle, "monotonic"), (approach, "approach_local")):
        with pytest.raises(SceneConfigError, match=match):
            _plan(scene, wrap(bad, "bad"), HOOKED)
    plan, report = _plan(scene, wrap(sever, "two"), HOOKED)
    assert report["perception"]["calibration_op"] == "survey0"          # a plain two-plane sever goal is eligible
    descending = {**sever, "planes": sever["planes"][::-1]}
    assert _plan(scene, wrap(descending, "down"), HOOKED)[1]["perception"]["side"] == -1


# ------------------------------------------------------------------------------------- measurement

@pytest.mark.parametrize("seed", [1301, 1302])
@pytest.mark.parametrize("shift", [-.0015, 0., .0006, .0012, .0025])
def test_end_face_relative_shift_on_synthetic_depth(samples, seed, shift):
    scene, goal = samples[seed]
    plan, report = _plan(scene, goal, HOOKED)
    geometry = hook.scene_bar_geometry(scene)
    edge = np.asarray(next(w for w in plan if w.label == "SURVEY:survey0").pose, float)
    half = scene["workpiece"]["base_box_m"]["length_m"]/2
    cut0 = goal["steps"][0]["planes"][0]["origin_local_m"][0]
    calibration = hook.measure_end_face(render(scene, edge, [(-half, half)]), geometry, 1, edge_pose_world=edge)
    after = [(-half-.0012, cut0-.0005-.0012), (cut0+.0005+shift, half+shift)]
    survey = hook.measure_end_face(render(scene, edge, after, time_s=9.), geometry, 1, edge_pose_world=edge)
    assert calibration["end_face"]["valid"] and calibration["end_face"]["lines_used"] >= 15
    assert abs(calibration["end_face"]["x_m"]-half) < .0006
    estimate, record = hook.relative_shift(calibration, survey)
    assert record["valid"] and abs(estimate-shift) < 1e-4, (estimate, shift)


def test_depth_noise_and_invalid_estimates(samples):
    scene, goal = samples[1301]
    plan, _ = _plan(scene, goal, HOOKED)
    geometry = hook.scene_bar_geometry(scene)
    edge = np.asarray(next(w for w in plan if w.label == "SURVEY:survey0").pose, float)
    half = scene["workpiece"]["base_box_m"]["length_m"]/2
    calibration = hook.measure_end_face(render(scene, edge, [(-half, half)], noise=1e-4, seed=1), geometry, 1)
    survey = hook.measure_end_face(render(scene, edge, [(-half, half+.0011)], noise=1e-4, seed=2), geometry, 1)
    estimate, record = hook.relative_shift(calibration, survey)
    assert record["valid"] and abs(estimate-.0011) < 2.5e-4
    far = hook.measure_end_face(render(scene, edge, [(-half, half+.0075)]), geometry, 1)
    assert hook.relative_shift(calibration, far)[0] is None             # |shift| > 6 mm: no shift
    empty = hook.measure_end_face(render(scene, edge, []), geometry, 1)  # no bar in view
    value, record = hook.relative_shift(calibration, empty)
    assert value is None and not record["valid"] and record["applied_shift_m"] == 0.


# ------------------------------------------------------------------------------------- policy

def _task(scene, goal, overrides=HOOKED):
    goal = deepcopy(goal)
    goal["episode"]["duration_s"] = 60.
    return compile_task(scene, goal, planner_options=overrides)


def _run(task, plan, report, depth, max_s=60.):
    """Ideal-integrator loop of a RelocatePolicy; ``depth(state)`` renders the current frame."""
    state = {}

    def source(camera_id):
        assert camera_id == hook.CAMERA_ID
        return depth(state)
    policy = hook.relocate_policy(task, plan, report["perception"], depth_source=source, limits=task.action_limits(),
                                  **goal_planner.policy_options(task.controller["planner"]))
    edge = np.asarray(plan[0].pose, float)
    events = []
    for step in range(int(max_s/task.control_dt_s)):
        state.update(edge=edge.copy(), time_s=step*task.control_dt_s, phase=policy.phase)
        action = policy({"proprio": {"tool_pose": edge.copy()}, "timestamp_s": step*task.control_dt_s})
        if policy.last_vision is not None:
            events.append(policy.last_vision)
        if policy.done:
            break
        delta = action["target"]
        edge = np.r_[edge[:3]+delta[:3], (Rotation.from_rotvec(delta[3:])*Rotation.from_quat(edge[3:])).as_quat()]
    return policy, events


def test_policy_surveys_and_relocates_the_second_cut(samples):
    scene, goal = samples[1302]                                          # yaw +18 deg
    task = _task(scene, goal)
    plan, report = goal_planner.plan_with_report(task.scene, task.goal, task.tool_profile, task.controller["planner"])
    half = scene["workpiece"]["base_box_m"]["length_m"]/2
    cut0 = goal["steps"][0]["planes"][0]["origin_local_m"][0]

    def depth(state):
        surveyed = state["phase"].endswith("survey1") or state["phase"].startswith("SURVEY_LEAVE:survey1")
        parts = [(-half-.0012, cut0-.0017), (cut0+.0017, half+.0012)] if surveyed else [(-half, half)]
        return render(scene, state["edge"], parts, time_s=state["time_s"])
    policy, events = _run(task, plan, report, depth)
    assert policy.done and [e["kind"] for e in events] == ["calibration", "survey"]
    estimate = policy.vision["surveys"][0]["estimate"]
    assert estimate["valid"] and abs(estimate["shift_m"]-.0012) < 1e-4
    R = _matrix(scene_world_pose(scene))[:3, :3]
    vector = R@np.array([estimate["shift_m"], 0., 0.])
    assert np.allclose(policy.vision["applied_shift_world_m"], vector)
    shifted = set(hook.events_from_plan(plan, report["perception"])[1]["shifted"])
    for k, (before, after) in enumerate(zip(plan, policy.waypoints)):
        moved = np.asarray(after.pose[:3])-np.asarray(before.pose[:3])
        assert np.allclose(moved, vector if k in shifted else 0., atol=1e-12), before.label
        assert after.pose[3:] == before.pose[3:] and after.label == before.label
    assert policy.waypoints[-1] == plan[-1] and policy.vision["inputs"] == hook.INPUTS
    record = policy.vision["calibration"]
    assert record["valid"] and record["frame_timestamp_s"] == record["observation_timestamp_s"]


@pytest.mark.parametrize("failure", ["stale", "raises", "none"])
def test_policy_without_a_current_frame_applies_no_shift(samples, failure):
    scene, goal = samples[1301]
    task = _task(scene, goal)
    plan, report = goal_planner.plan_with_report(task.scene, task.goal, task.tool_profile, task.controller["planner"])
    half = scene["workpiece"]["base_box_m"]["length_m"]/2

    def depth(state):
        if failure == "raises":
            raise RuntimeError("camera offline")
        frame = render(scene, state["edge"], [(-half, half)], time_s=state["time_s"])
        if failure == "stale":
            frame["physics_timestamp_s"] -= .02
        return frame
    policy, events = _run(task, plan, report, depth if failure != "none" else None)
    assert policy.done and policy.vision["applied_shift_m"] == 0. and policy.waypoints == plan
    estimate = policy.vision["surveys"][0]["estimate"]
    assert not estimate["valid"] and estimate["applied_shift_m"] == 0. and estimate["reasons"]


def test_policy_rejects_profiles_other_than_the_delta_law(samples):
    scene, goal = samples[1301]
    task = _task(scene, goal)
    plan, report = goal_planner.plan_with_report(task.scene, task.goal, task.tool_profile, task.controller["planner"])
    with pytest.raises(ValueError):
        hook.relocate_policy(task, plan, report["perception"], command_mode="bias_pose")
    with pytest.raises(ValueError):
        hook.relocate_policy(task, plan, {"hook": "other"})


# ------------------------------------------------------------------------------------- runner wiring

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_builds_the_hook_only_when_requested(samples, tmp_path):
    module = _load("run_wp10_live_perception", ROOT/"scripts/run_wp10_live.py")
    scene, goal = samples[1301]
    args = Namespace(edge_feedback="nominal", position_tolerance=None)
    for overrides, kind in ((OVERRIDES, WaypointPolicy), (HOOKED, hook.RelocatePolicy)):
        task = _task(scene, goal, overrides)
        env = SimpleNamespace(task=task, backend=SimpleNamespace(), _frames={}, time_s=0.)
        policy, receipt = module.configured_policy(env, args)
        assert type(policy) is kind
        assert ("perception" in receipt) == (kind is hook.RelocatePolicy)
        if kind is WaypointPolicy:
            assert policy.waypoints == goal_planner.plan(task.scene, task.goal, task.tool_profile, task.controller["planner"])
    paths = {}
    for name, value in (("scene", scene), ("goal", {**goal, "episode": {**goal["episode"], "duration_s": 60.}})):
        paths[name] = tmp_path/f"{name}.json"
        paths[name].write_text(json.dumps(value))
    captured = []
    module.run_configured = lambda a, out, task: captured.append(task) or 0
    module.new_output = lambda path: Path(path)
    module.write_json = lambda *a, **k: None
    module.manifest = lambda config, kind: config
    argv = ["--runtime", "isaac_sim", "--scene", str(paths["scene"]), "--goal", str(paths["goal"]), "--controller",
            "scripted", "--planner-options", json.dumps(HOOKED), "--steps", "10", "--out", str(tmp_path/"never")]
    with pytest.raises(SystemExit):
        module.main(argv)                                                 # the hook needs rendered cameras
    assert module.main([*argv, "--record-cameras"]) == 0 and captured[0].controller["planner"]["perception"] == hook.HOOK
    assert module.main([*argv[:-5], json.dumps(OVERRIDES), *argv[-4:]]) == 0   # no hook: no camera requirement
    assert "perception" not in captured[1].controller["planner"]
