"""ADR 0009 wrist-depth survey and controller profile path_v2_vision on CPU (synthetic depth renders).

The synthetic renderer ray-marches the live wrist-D405 geometry (848x480 K, camera
looking straight down, image rows along world +x, 45 mm +x / -85 mm y of the
edge) over a filleted bar on the board. Only a column crop around the bar is
rendered (a valid camera with a shifted principal point).
"""
import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.control import ActionLimits
from robotic_cutting_sim.policies import (CONTROLLER_PROFILES, VisionRetargetPolicy, WaypointPolicy,
                                          controller_profile, ideal_plan_end_s, skim, slice_push, slice_push_vision,
                                          through_cut, vision_policy, vision_survey_options)
from robotic_cutting_sim.task_scenarios import configure_task
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.wrist_depth_survey import (BACKGROUND, EXCLUDED, MATERIAL, BarGeometry, SurveyConfig,
                                                    back_project, estimate_shift, select_bar_top, survey_frame)
from rollout_reference import (check_behaviour, compare_to_reference, ideal_rollout, load_reference, make_case,
                               rollout_tolerance)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "blademaster_workcell_v1"
FX, HEIGHT, U0, U1 = 446.8027594274324, 480, 620, 740
R_WORLD_CAMERA = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])   # camera X = world +y, Y = +x, Z = down
CAMERA_FROM_EDGE = np.array([.045, -.085, .14365])
OFFSET = .0003       # rendered surface beyond the outermost particle centre
TOP = .0198          # rendered bar top (local z)
FILLET = .001


def _task():
    return configure_task(make_task("SLICE-PUSH-01"), scenario=SCENARIO)


TASK = _task()
GEOMETRY = BarGeometry.from_task(TASK)
Z0 = TASK.T_world_workpiece[2, 3]


def bar(shift_right=0., shift_left=-.003, cut=True):
    """Local x intervals of the rendered bar parts (cut 0 at -30 mm: left / right of the kerf)."""
    if not cut:
        return [(-.0795-OFFSET, .0795+OFFSET)]
    return [(-.0795-OFFSET+shift_left, -.0305+OFFSET+shift_left), (-.0295-OFFSET+shift_right, .0795+OFFSET+shift_right)]


def _height(x, y, parts):
    h = np.where((np.abs(x) <= .09) & (np.abs(y) <= .05), 0., -.06)
    h = np.where((np.abs(np.abs(x)-.03) < .004) & (np.abs(y) <= .05), -.03, h)
    inside = np.abs(y) <= .015
    for a, b in parts:
        top = np.full_like(x, TOP)
        for corner, sign in ((a+FILLET, -1.), (b-FILLET, 1.)):
            beyond = sign*(x-corner) > 0
            top = np.where(beyond, TOP-FILLET+np.sqrt(np.clip(FILLET**2-(x-corner)**2, 0., None)), top)
        h = np.where(inside & (x >= a) & (x <= b), np.maximum(h, top), h)
    return h


def render(edge_world, parts, *, knife=True, noise=0., seed=0, time_s=0.):
    """Depth observation dict of the wrist camera for an edge pose over the given bar parts."""
    camera = np.asarray(edge_world[:3], float)+CAMERA_FROM_EDGE
    K = np.array([[FX, 0., 424.-U0], [0., FX, 240.], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_WORLD_CAMERA, camera
    v, u = np.mgrid[:HEIGHT, :U1-U0]
    rays = np.stack([(u-K[0, 2])/FX, (v-K[1, 2])/FX, np.ones(u.shape)], -1)@R_WORLD_CAMERA.T
    c = camera-[0., 0., Z0]                                   # camera in workpiece-local coordinates
    edge_local = np.asarray(edge_world[:3], float)-[0., 0., Z0]

    rx, ry = rays[..., 0].ravel(), rays[..., 1].ravel()

    def occupied(t, index=slice(None)):
        x, y, z = c[0]+rx[index]*t, c[1]+ry[index]*t, c[2]-t
        hit = z <= _height(x, y, parts)
        if knife:
            hit |= (np.abs(x-edge_local[0]) <= .00075) & (np.abs(y) <= .06) & (z >= edge_local[2]) & (z <= edge_local[2]+.04)
        return hit
    step = .0004
    hi = np.full(rx.shape, np.nan)
    for t in np.arange(c[2]-.075, c[2]+.0605, step):            # coarse march: first occupied sample per ray
        open_rays = np.nonzero(np.isnan(hi))[0]
        if not len(open_rays):
            break
        hit = occupied(t, open_rays)
        hi[open_rays[hit]] = t
    found = np.isfinite(hi)
    index = np.nonzero(found)[0]
    lo, high = hi[index]-step, hi[index]
    for _ in range(14):                                          # bisection on the hit rays only
        mid = (lo+high)/2
        inside = occupied(mid, index)
        high, lo = np.where(inside, mid, high), np.where(inside, lo, mid)
    t = np.zeros(rx.shape)
    t[index] = high
    found, depth = found.reshape(u.shape), t.reshape(u.shape)
    if noise:
        depth = np.where(found, depth+np.random.default_rng(seed).normal(0., noise, depth.shape), 0.)
    valid = found & (depth >= .07) & (depth <= .5)
    return dict(depth_m=np.where(valid, depth, 0.).astype(np.float32), valid_mask=valid, K=K, T_world_camera=T,
                physics_timestamp_s=float(time_s), camera_id="wrist_d405_0")


def edge_at(x_local, clearance=.008):
    pose = np.asarray(TASK.reference_edge_pose_world(0.), float)
    return np.r_[TASK.points_to_world([x_local, 0., .02+clearance]), pose[3:]]


CAL_EDGE, SURVEY_EDGE = edge_at(-.03), edge_at(.03)


@pytest.fixture(scope="module")
def calibration():
    return survey_frame(render(CAL_EDGE, bar(cut=False)), GEOMETRY, edge_pose_world=CAL_EDGE)


# ------------------------------------------------------------------------------------------- estimator

def test_back_project_inverts_the_pinhole_projection():
    K = np.array([[FX, 0., 424.], [0., FX, 240.], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_euler("xyz", [170., 5., 80.], degrees=True).as_matrix(), [.1, -.2, .3]
    point = np.array([.12, -.15, .05])
    optical = np.linalg.inv(T)@np.r_[point, 1.]
    u, v = (K@optical[:3])[:2]/optical[2]
    depth = np.zeros((480, 848))
    depth[int(round(v)), int(round(u))] = optical[2]
    cloud = back_project(depth, K, T)
    assert cloud["valid"].sum() == 1
    recovered = cloud["points_world"][int(round(v)), int(round(u))]
    pixel = np.linalg.norm(recovered-point)
    assert pixel < optical[2]/FX                                     # within one pixel footprint (rounded pixel)
    with pytest.raises(ValueError):
        back_project(depth[0], K, T)


def test_selection_classes_and_scan_axis(calibration):
    frame = render(CAL_EDGE, bar(cut=False))
    selection = select_bar_top(back_project(frame["depth_m"], frame["K"], frame["T_world_camera"], frame["valid_mask"]),
                               GEOMETRY, edge_pose_world=CAL_EDGE)
    labels, z = selection["labels"], selection["points_local"][..., 2]
    assert selection["scan_axis"] == 0 and selection["scan_sign"] == 1.      # image rows run along +x
    assert np.all(z[labels == MATERIAL] > GEOMETRY.board_top_local_z+.003)
    assert np.all(z[labels == BACKGROUND] <= GEOMETRY.board_top_local_z+.003)
    assert (labels == EXCLUDED).sum() > 0 and selection["counts"]["material_px"] > 10000
    assert calibration["end_minus_x"]["valid"] is False                       # -80 mm end: out of view
    assert calibration["kerf_right"]["valid"] is False                        # no kerf before cut 0
    assert calibration["end_plus_x"]["valid"] and calibration["end_plus_x"]["lines_used"] >= 60


@pytest.mark.parametrize("shift", [-.003, -.001, 0., .0012, .003])
def test_end_face_shift_recovers_the_rendered_part_shift(calibration, shift):
    survey = survey_frame(render(SURVEY_EDGE, bar(shift)), GEOMETRY, edge_pose_world=SURVEY_EDGE)
    estimate, record = estimate_shift(calibration, survey)
    assert record["valid"] and record["reasons"] == []
    assert abs(estimate-shift) <= .00015, (estimate, shift)
    assert record["applied_shift_m"] == estimate and record["calibration_face"] == "end_plus_x"
    assert survey["kerf_right"]["valid"] is False                             # kerf at -30 mm: out of view at cut 1


@pytest.mark.parametrize("sigma,seed", [(.0001, 0), (.0001, 1), (.0002, 2)])
def test_end_face_shift_with_depth_noise(sigma, seed):
    calibration = survey_frame(render(CAL_EDGE, bar(cut=False), noise=sigma, seed=seed), GEOMETRY,
                               edge_pose_world=CAL_EDGE)
    survey = survey_frame(render(SURVEY_EDGE, bar(.002), noise=sigma, seed=100+seed), GEOMETRY,
                          edge_pose_world=SURVEY_EDGE)
    estimate, record = estimate_shift(calibration, survey)
    assert record["valid"] and abs(estimate-.002) <= .0003, (estimate, record)


@pytest.mark.parametrize("shift", [.0005, .0015, .0025])
def test_kerf_face_formula_where_the_kerf_is_in_view(calibration, shift):
    """ADR formula x_kf + 29.5 mm + r_s, with the camera over cut 0 (end of lift_0), uncrushed synthetic kerf."""
    survey = survey_frame(render(CAL_EDGE, bar(shift)), GEOMETRY, edge_pose_world=CAL_EDGE)
    assert survey["kerf_right"]["valid"]
    estimate, record = estimate_shift(calibration, survey, method="kerf_face")
    assert record["valid"] and abs(estimate-shift) <= .0002, (estimate, record)


def test_kerf_face_under_the_blade_is_reported_invalid(calibration):
    """A kerf face pushed under the blade (-2 mm) is occluded by it: no face, no estimate."""
    survey = survey_frame(render(CAL_EDGE, bar(-.002)), GEOMETRY, edge_pose_world=CAL_EDGE)
    estimate, record = estimate_shift(calibration, survey, method="kerf_face")
    assert estimate is None and "kerf_right_face" in record["reasons"][0]


def test_invalid_measurements_apply_no_shift(calibration):
    empty = survey_frame(render(SURVEY_EDGE, []), GEOMETRY, edge_pose_world=SURVEY_EDGE)
    estimate, record = estimate_shift(calibration, empty)
    assert estimate is None and not record["valid"] and record["applied_shift_m"] == 0.
    assert "bracketed lines" in record["reasons"][0]
    estimate, record = estimate_shift(empty, empty)
    assert estimate is None and record["reasons"][0].startswith("calibration: no valid end face")
    far = survey_frame(render(SURVEY_EDGE, bar(.007)), GEOMETRY, edge_pose_world=SURVEY_EDGE)
    estimate, record = estimate_shift(calibration, far)
    assert estimate is None and record["shift_m"] == pytest.approx(.007, abs=.0003)
    assert "> 0.006" in record["reasons"][0] and record["applied_shift_m"] == 0.
    strict = SurveyConfig(min_lines=10**6)
    assert estimate_shift(calibration, far, config=strict)[1]["valid"] is False
    with pytest.raises(ValueError):
        estimate_shift(calibration, far, method="oracle")


# ------------------------------------------------------------------------------------------- profile + plan

PINNED = {  # computed at 9b6df5f (before path_v2_vision) on the blademaster_workcell_v1 tasks
    # The ".../rollout" digests pinned here before WP-11 Phase A now pin the stored reference trajectory:
    # tests/rollout_reference.py HISTORIC_ROLLOUT_SHA256["v1|path_v2/<task>"].
    "v1/SLICE-PUSH-01/plan": "277be06fa090a5f2dc2f0f0541215b36f5451de48415988e43b1ba97c17082e6",
    "v1/TRIM-01-R/plan": "e2c8b79acf519b86f721910a0ec085f7861be3068731d0bc42a0e4c287d3f813",
    "v1/CUT-01-R/plan": "8caf708576416a8027f2e6f46b5edc9915223ae0b9b67b546078ba18bc47cfe1",
    "path_v2/SLICE-PUSH-01/plan": "d83c42e3724504fed6590af01b56fa8c3778351774db89ab344ec1d3424c72b4",
    "path_v2/TRIM-01-R/plan": "2b330ca36e0a6e0176b278e811f54624404b66d5925decce6bdb0b7af410f36f",
    "path_v2/CUT-01-R/plan": "a3d47471bd64a34d8b41956ab6e933e005ef25b0e48ceb2ff030151fb90578a2",
}
FIELDS = ("pose", "cut", "saw_amplitude_m", "saw_period_s", "speed_m_s", "cut_index", "hold_s", "label", "track_path")


def _plan_sha(plan):
    return hashlib.sha256(json.dumps([[np.asarray(getattr(w, f)).tolist() for f in FIELDS] for w in plan]).encode()).hexdigest()


@pytest.mark.parametrize("name", ["v1", "path_v2"])
@pytest.mark.parametrize("task_id,builder", [("SLICE-PUSH-01", slice_push), ("TRIM-01-R", skim), ("CUT-01-R", through_cut)])
def test_v1_and_path_v2_plans_and_rollouts_are_pinned(name, task_id, builder):
    """Adding path_v2_vision changed neither the v1/path_v2 plans nor their scripted rollouts.

    Plans stay pinned by hash: they are identical across the BLAS kernels tested (OpenBLAS Prescott,
    Nehalem, Sandybridge, Haswell, Zen, SkylakeX, Cooperlake) and on the GitHub runner. Rollouts were
    pinned by the SHA-256 of their raw action bytes until WP-11 Phase A; that digest depends on the BLAS
    kernel (one-ulp differences in 3-vector dot products, max 8.7e-19 m per action), so it failed on the
    GitHub runner with identical Python/NumPy/SciPy. The rollout guarantee is now: the stored reference
    hashes to the original digest, the live rollout matches it (exact cut flags and waypoint sequence,
    continuous actions within steps*16*eps), and its behaviour (bounds, progress, reach, trajectory error,
    termination) is checked directly. See tests/rollout_reference.py and docs/reports/WP11_PHASE_A_CI.md.
    """
    policy_options, plan_options = controller_profile(name)
    task = configure_task(make_task(task_id), scenario=SCENARIO)
    plan = builder(task, **plan_options)
    assert _plan_sha(plan) == PINNED[f"{name}/{task_id}/plan"]
    case = f"{name}/{task_id}"
    reference = load_reference()[case]
    policy, case_task, start, steps = make_case(case)
    assert _plan_sha(policy.waypoints) == PINNED[f"{name}/{task_id}/plan"] and np.array_equal(start, plan[0].pose)
    tol = rollout_tolerance(steps)
    live = ideal_rollout(policy, case_task, start, steps)
    compare_to_reference(live, reference, tol)
    assert check_behaviour(live, policy, case_task, tol) == reference["done_step"]


def test_path_v2_vision_profile_extends_path_v2_only():
    assert CONTROLLER_PROFILES["v1"] == {"name": "v1", "path_tracking": False}
    assert controller_profile("path_v2_vision") == controller_profile("path_v2")
    assert vision_survey_options("v1") is None and vision_survey_options("path_v2") is None
    survey = vision_survey_options("path_v2_vision")
    assert survey["task_ids"] == ("SLICE-PUSH-01",) and survey["camera_id"] == "wrist_d405_0"
    assert survey["method"] == "end_face_plus_x" and survey["diagnostic_methods"] == ("kerf_face",)
    with pytest.raises(ValueError):
        vision_survey_options("v9")
    with pytest.raises(ValueError):
        vision_policy(configure_task(make_task("CUT-01-R"), scenario=SCENARIO))
    with pytest.raises(ValueError):
        vision_policy(TASK, "path_v2")


def test_vision_plan_structure_and_ideal_duration():
    _, plan_options = controller_profile("path_v2")
    base = slice_push(TASK, **plan_options)
    plan, indices = slice_push_vision(TASK, **plan_options)
    labels = [w.label for w in plan]
    assert labels == ["approach_0", "start_0", "calibrate_0", "cut_0", "hold_0", "lift_0", "move_1", "survey_1",
                      "retarget_1", "cut_1", "hold_1", "raise_in_kerf", "push", "push_hold", "release", "retract"]
    assert indices == {"calibration": 2, "survey": 7, "shift_from": 8}
    assert plan[2].hold_s == plan[7].hold_s == plan[8].hold_s == .3 and plan[8].pose == plan[7].pose
    assert plan[5].speed_m_s == .04 and plan[11].speed_m_s == .02 and not plan[8].cut
    unchanged = [i for i in range(16) if i not in (2, 5, 7, 8, 11)]
    for i in unchanged:
        assert plan[i] == base[i if i < 8 else i-1]
    assert [w.pose for w in plan[:8]] == [w.pose for w in base[:8]]
    # Ideal tracking (zero shift): finishes at 14.32 s of the 16 s episode (path_v2: 15.16 s).
    policy = vision_policy(TASK)
    end = ideal_plan_end_s(TASK, policy.waypoints, path_tracking=True, cross_track_gain=.5, position_tolerance_m=.0003)
    assert end == pytest.approx(14.32) and end <= 14.5
    assert ideal_plan_end_s(TASK, base, path_tracking=True, cross_track_gain=.5, position_tolerance_m=.0003) == \
        pytest.approx(15.16)


# ------------------------------------------------------------------------------------------- mocked policy run

class World:
    """Ideal plant plus a synthetic wrist depth source; cut 0 shifts the part right of x = -30 mm."""

    def __init__(self, policy, shift, *, stale=False, broken=False):
        self.policy, self.shift, self.stale, self.broken = policy, shift, stale, broken
        self.edge = np.asarray(policy.waypoints[0].pose, float)
        self.time_s, self.cut0_done, self.frames = 0., False, []

    def depth(self, camera_id):
        assert camera_id == "wrist_d405_0"
        if self.broken:
            raise RuntimeError("camera offline")
        frame = render(self.edge, bar(self.shift) if self.cut0_done else bar(cut=False),
                       time_s=self.time_s-(.02 if self.stale else 0.))
        self.frames.append((self.time_s, self.policy.phase))
        return frame

    def run(self, steps=800):
        rows = []
        for k in range(steps):
            self.time_s = k*TASK.control_dt_s
            action = self.policy({"proprio": {"tool_pose": self.edge.copy()}, "timestamp_s": self.time_s})
            rows.append(dict(step=k, phase=self.policy.phase, edge=self.edge.copy(), action=action,
                             vision=self.policy.last_vision))
            if self.policy.done:
                break
            delta = action["target"]
            self.edge = np.r_[self.edge[:3]+delta[:3],
                              (Rotation.from_rotvec(delta[3:])*Rotation.from_quat(self.edge[3:])).as_quat()]
            self.cut0_done |= self.policy.phase == "hold_0"
        return rows


def _vision(**kwargs):
    policy = vision_policy(TASK, **kwargs)
    return policy, [w.pose for w in policy.waypoints]


@pytest.mark.parametrize("shift", [.0015, -.002])
def test_mocked_policy_run_surveys_and_retargets_cut_1(shift):
    policy, planned = _vision()
    world = World(policy, shift)
    policy.depth_source = world.depth
    rows = world.run()
    assert policy.done
    events = [row for row in rows if row["vision"] is not None]
    assert [row["vision"]["kind"] for row in events] == ["calibration", "survey"]
    calibration, survey = policy.vision["calibration"], policy.vision["survey"]
    # Frames are taken when the settle holds end: at the calibrate_0 / survey_1 poses, current timestamps.
    assert world.frames[0][1] == "cut_0" and world.frames[1][1] == "retarget_1"
    for record, edge in ((calibration, CAL_EDGE), (survey, SURVEY_EDGE)):
        assert record["valid"] and record["frame_timestamp_s"] == record["observation_timestamp_s"]
        step = next(row for row in rows if row["step"]*TASK.control_dt_s == record["observation_timestamp_s"])
        assert np.linalg.norm(step["edge"][:3]-edge[:3]) <= .0003
    estimate = policy.vision["estimate"]
    assert estimate["valid"] and abs(estimate["shift_m"]-shift) <= .00015
    applied = policy.vision["applied_shift_m"]
    assert applied == estimate["shift_m"] and policy.vision["applied_shift_world_m"] == [applied, 0., 0.]
    assert policy.vision["shifted_labels"] == ["retarget_1", "cut_1", "hold_1", "raise_in_kerf", "push", "push_hold",
                                               "release", "retract"]
    for k, (before, after) in enumerate(zip(planned, (w.pose for w in policy.waypoints))):
        offset = np.asarray(after[:3])-np.asarray(before[:3])
        assert np.allclose(offset, [applied, 0., 0.] if k >= 8 else 0., atol=1e-15) and after[3:] == before[3:]
    assert policy.vision["diagnostics"]["kerf_face"]["valid"] is False         # kerf out of view at the cut-1 start
    cut = [row for row in rows if row["action"]["cut_enabled"]]
    assert {row["phase"] for row in cut} == {"cut_0", "cut_1"}
    for row in cut:
        x = TASK.points_to_local(row["edge"][:3])[0]
        expected = -.03 if row["phase"] == "cut_0" else .03+applied
        assert abs(x-expected) < 1e-6
    survey_step = events[1]["step"]
    assert rows[survey_step]["phase"] == "retarget_1"
    assert np.linalg.norm(rows[survey_step]["action"]["target"][:3]) > 0      # the action moves to the shifted start
    assert rows[-1]["step"]*TASK.control_dt_s <= 14.5
    json.dumps(policy.vision)                                                  # JSON-serialisable receipt


@pytest.mark.parametrize("failure,reason", [("stale", "stale depth frame"), ("broken", "depth source failed"),
                                            ("none", "no depth source")])
def test_mocked_policy_without_a_current_frame_applies_no_shift(failure, reason):
    policy, planned = _vision()
    world = World(policy, .0015, stale=failure == "stale", broken=failure == "broken")
    policy.depth_source = None if failure == "none" else world.depth
    world.run()
    assert policy.done and policy.vision["applied_shift_m"] == 0. and policy.vision["shifted_labels"] == []
    assert not policy.vision["estimate"]["valid"] and reason in policy.vision["estimate"]["reasons"][0]
    assert [w.pose for w in policy.waypoints] == planned


def test_vision_policy_validation():
    plan, indices = slice_push_vision(TASK)
    with pytest.raises(ValueError):
        VisionRetargetPolicy(TASK, plan, indices={"calibration": 7, "survey": 2, "shift_from": 8})
    with pytest.raises(ValueError):
        VisionRetargetPolicy(TASK, plan, indices=indices, survey_method="oracle")
    with pytest.raises(ValueError):
        slice_push_vision(TASK, settle_s=0.)


# ------------------------------------------------------------------------------------------- CLI / runner

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(argv):
    module, captured = _load("run_wp10_live_vision", ROOT/"scripts/run_wp10_live.py"), {}

    class Stop(Exception):
        pass

    def stop(task, **options):
        captured.update(task=task, options=options)
        raise Stop

    module.open_cutting_env = stop
    module.new_output = lambda path: Path(path)
    module.write_json = lambda *a, **k: None
    module.manifest = lambda config, kind: captured.setdefault("config", config)
    captured["status"] = module.main(["--runtime", "isaac_sim", "--out", "unused", *argv])
    return captured


def test_cli_accepts_path_v2_vision_only_for_slice_push_with_cameras():
    ok = _parse(["--task", "SLICE-PUSH-01", "--controller", "scripted", "--record-cameras",
                 "--controller-profile", "path_v2_vision"])
    assert ok["status"] == 1 and ok["config"]["controller_profile"] == "path_v2_vision" and ok["options"]["camera"]
    for bad in (["--task", "SLICE-PUSH-01", "--controller", "scripted", "--controller-profile", "path_v2_vision"],
                ["--task", "CUT-01-R", "--controller", "scripted", "--record-cameras",
                 "--controller-profile", "path_v2_vision"],
                ["--task", "SLICE-PUSH-01", "--record-cameras", "--controller-profile", "path_v2_vision"]):
        with pytest.raises(SystemExit):
            _parse(bad)


def _stub_env(time_s=0.):
    frame = render(CAL_EDGE, bar(cut=False), time_s=time_s)
    calls = []

    def capture():
        calls.append("capture")
        return {"wrist_d405_0": {key: value for key, value in frame.items() if key not in ("K", "T_world_camera")}}, {}
    backend = SimpleNamespace(proprioception=lambda: {"timestamp_s": time_s, "tool_pose": CAL_EDGE},
                              capture=capture, calibration={"wrist_d405_0": {"K": frame["K"].tolist(),
                                                                             "T_world_camera": frame["T_world_camera"].tolist()}})
    return SimpleNamespace(task=TASK, action_limits=None, time_s=time_s, backend=backend, _frames={}), frame, calls


def test_scripted_policy_and_depth_source_wiring():
    module = _load("run_wp10_live_vision_wiring", ROOT/"scripts/run_wp10_live.py")
    env, frame, calls = _stub_env(.5)
    args = dict(motion="push", speed=.01, angle=0., offset=0., edge_feedback="measured", position_tolerance=None)
    policy = module.scripted_policy(env, Namespace(**args, controller_profile="path_v2_vision"))
    assert isinstance(policy, VisionRetargetPolicy) and policy.edge_estimator is not None
    assert policy.path_tracking and policy.position_tolerance_m == .0003
    assert [w.pose for w in policy.waypoints] == [w.pose for w in vision_policy(TASK).waypoints]
    latest = policy.depth_source("wrist_d405_0")
    assert calls == ["capture"] and latest["physics_timestamp_s"] == .5
    assert np.array_equal(latest["depth_m"], frame["depth_m"]) and np.allclose(latest["K"], frame["K"])
    assert set(latest) == {"camera_id", "depth_m", "valid_mask", "K", "T_world_camera", "physics_timestamp_s"}
    env.time_s = .52                                                       # a frame from an earlier physics time
    with pytest.raises(ValueError):
        policy.depth_source("wrist_d405_0")
    assert module.policy_plan_name(TASK, policy) == "slice_push_vision"
    assert module.policy_plan_name(TASK, WaypointPolicy(TASK, [])) == "slice_push"
    v2 = module.scripted_policy(env, Namespace(**args, controller_profile="path_v2"))
    assert type(v2) is WaypointPolicy                                       # path_v2 unchanged
    override = module.scripted_policy(env, Namespace(**{**args, "position_tolerance": .0004},
                                                     controller_profile="path_v2_vision"))
    assert override.position_tolerance_m == .0004
    assert module.controller_profile_receipt(Namespace(controller_profile="path_v2_vision", position_tolerance=None)) \
        == CONTROLLER_PROFILES["path_v2_vision"]


@pytest.fixture(scope="module")
def runner():
    return _load("run_blademaster_demos_vision", ROOT/"scripts/run_blademaster_demos.py")


def test_runner_passes_path_v2_vision_for_demo_3_only(runner, tmp_path, capsys):
    spec = runner.DEMOS["3"]
    v1 = runner.run_argv(spec, tmp_path/"demo3", python="PY")
    argv = runner.run_argv(spec, tmp_path/"demo3", python="PY", controller_profile="path_v2_vision")
    assert argv == [*v1[:-8], "--controller-profile", "path_v2_vision", *v1[-8:]]
    out = _parse(argv[4:])                                                  # drop PY, script, --runtime isaac_sim
    assert out["status"] == 1 and out["config"]["controller_profile"] == "path_v2_vision"
    for name, value in runner.expected_config(spec, "path_v2_vision").items():
        assert out["config"][name] == value, name
    root = tmp_path/"demos"
    assert runner.main(["--demo", "3", "--dry-run", "--run-id", "R1", "--root", str(root),
                        "--controller-profile", "path_v2_vision"]) == 0
    assert "--controller-profile path_v2_vision --steps 800" in capsys.readouterr().out and not root.exists()
    for demo in ("2", "all"):
        with pytest.raises(SystemExit):
            runner.main(["--demo", demo, "--dry-run", "--run-id", "R1", "--root", str(root),
                         "--controller-profile", "path_v2_vision"])


def test_runner_summarises_the_vision_receipt(runner):
    assert runner.vision_summary({}) is None
    vision = dict(method="end_face_plus_x", camera_id="wrist_d405_0", applied_shift_m=.0012,
                  applied_shift_world_m=[.0012, 0., 0.], shifted_labels=["retarget_1", "cut_1"], adr="adr",
                  calibration=dict(observation_timestamp_s=.5, frame_timestamp_s=.5, valid=True, reason=None),
                  survey=dict(observation_timestamp_s=6.8, frame_timestamp_s=6.8, valid=True, reason=None),
                  estimate=dict(valid=True, reasons=[], shift_m=.0012, calibration_face_x_m=.0792,
                                survey_face_x_m=.0804),
                  diagnostics={"kerf_face": dict(valid=False, reasons=["out of view"], shift_m=None)})
    summary = runner.vision_summary(dict(policy_vision=vision, policy_plan_ideal_end_s=14.38))
    assert summary["applied_shift_m"] == .0012 and summary["valid"] and summary["plan_ideal_end_s"] == 14.38
    assert summary["frames"]["survey"]["frame_timestamp_s"] == 6.8
    assert summary["diagnostics"] == {"kerf_face": {"valid": False, "reasons": ["out of view"], "shift_m": None}}
