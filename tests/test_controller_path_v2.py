"""Opt-in scripted controller profile path_v2 (path tracking, slow approach, settle) on CPU.

v1 (the default) stays bit-identical (pinned rollouts in tests/test_blademaster_demo.py);
path_v2 is checked on an ideal integrator and on a simulated laggy plant
(first-order lag + one-step delay + a 2 mm z offset that appears at the cut start).
"""
import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.control import ActionLimits
from robotic_cutting_sim.policies import (CONTROLLER_PROFILES, Waypoint, WaypointPolicy, _cap_with_priority,
                                          controller_profile, shape_approaches, skim, slice_push, through_cut)
from robotic_cutting_sim.task_scenarios import configure_task
from robotic_cutting_sim.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "blademaster_workcell_v1"
LIMIT = ActionLimits().max_translation_m
POLICY, PLAN = controller_profile("path_v2")


def _task(task_id):
    return configure_task(make_task(task_id), scenario=SCENARIO)


def _path_v2(task, builder, **kwargs):
    return WaypointPolicy(task, builder(task, **PLAN, **kwargs), **POLICY)


def _segment_error(edge, p0, p1):
    direction = (p1-p0)/np.linalg.norm(p1-p0)
    s = np.clip((edge-p0)@direction, 0., np.linalg.norm(p1-p0))
    return float(np.linalg.norm(edge-(p0+s*direction)))


def _plant(policy, task, *, alpha=1., delay=0, offset_z=0., kind="bias", steps=900, start_offset=(0., 0., .01)):
    """Run ``policy`` on a plant: commanded pose integrates each delta; the actual edge follows the
    commanded pose ``delay`` steps late with first-order gain ``alpha``. ``offset_z`` appears once the
    first cut step is issued: as a measurement bias (``bias``) or an actual displacement (``disp``)."""
    edge = np.asarray(policy.waypoints[0].pose, float)
    edge = np.r_[edge[:3]+start_offset, edge[3:]]
    commanded, actual, bias, started = [edge.copy()]*(delay+1), edge.copy(), 0., False
    log = []
    for k in range(steps):
        measured = np.r_[actual[:3]+[0., 0., bias], actual[3:]]
        index = policy.index
        action = policy({"proprio": {"tool_pose": measured.copy()}, "timestamp_s": k*task.control_dt_s})
        log.append(dict(step=k, index=index, phase=policy.phase, measured=measured, action=action,
                        tracking=policy.last_tracking, waypoint=policy.index))
        if policy.done:
            break
        delta, last = action["target"], commanded[-1]
        commanded.append(np.r_[last[:3]+delta[:3], (Rotation.from_rotvec(delta[3:])*Rotation.from_quat(last[3:])).as_quat()])
        source = commanded[-1-delay]
        actual = np.r_[actual[:3]+alpha*(source[:3]-actual[:3]), source[3:]]
        if not started and policy.waypoints[policy.index].cut:
            started = True
            if kind == "bias":
                bias = offset_z
            else:
                actual[2] += offset_z
    return log


def _cut_rows(log, policy):
    """Rows issued while a cut waypoint was active, with that waypoint's reference segment."""
    rows = []
    for row in log:
        w = row["waypoint"]
        if w < len(policy.waypoints) and policy.waypoints[w].cut:
            rows.append((row, np.asarray(policy.waypoints[w-1].pose[:3]), np.asarray(policy.waypoints[w].pose[:3])))
    return rows


# ------------------------------------------------------------------------------------------- v1 unchanged

def test_profiles_and_v1_defaults_are_unchanged():
    assert CONTROLLER_PROFILES["v1"] == {"name": "v1", "path_tracking": False}
    assert controller_profile("v1") == ({"path_tracking": False}, {})
    assert POLICY == {"path_tracking": True, "cross_track_gain": .5, "position_tolerance_m": .0003}
    assert PLAN == {"approach_speed_m_s": .02, "approach_slow_distance_m": .01, "settle_s": .3}
    with pytest.raises(ValueError):
        controller_profile("v3")
    policy = WaypointPolicy(make_task("CUT-01-R"), [])
    assert policy.path_tracking is False and policy.position_tolerance_m == .0005
    with pytest.raises(ValueError):
        WaypointPolicy(make_task("CUT-01-R"), [], cross_track_gain=0.)


@pytest.mark.parametrize("task_id,builder,kwargs", [
    ("CUT-01-R", through_cut, {}), ("CUT-01-R", through_cut, {"tilt_deg": 20.}),
    ("TRIM-01-R", skim, {}), ("SLICE-PUSH-01", slice_push, {}),
])
def test_plans_are_identical_with_approach_options_off(task_id, builder, kwargs):
    task = _task(task_id)
    off = builder(task, approach_speed_m_s=None, approach_slow_distance_m=.05, settle_s=0., **kwargs)
    assert off == builder(task, **kwargs)
    assert shape_approaches(off) == off


def test_v1_profile_rollout_matches_the_default_policy():
    task = make_task("CUT-01-R")
    plan = through_cut(task, saw_amplitude_m=.003)
    default = _plant(WaypointPolicy(task, plan), task, steps=600)
    explicit = _plant(WaypointPolicy(task, plan, **controller_profile("v1")[0]), task, steps=600)
    assert len(default) == len(explicit)
    for a, b in zip(default, explicit):
        assert np.array_equal(a["action"]["target"], b["action"]["target"]) and b["tracking"] is None


def test_cap_gives_the_cross_track_part_priority():
    cross, forward = np.array([.0015, 0., 0.]), np.array([0., .002, 0.])
    capped = _cap_with_priority(cross, forward, .002)
    assert np.linalg.norm(capped) == pytest.approx(.002) and np.allclose(capped[0], .0015)
    assert np.allclose(_cap_with_priority(np.array([.003, 0., 0.]), forward, .002), [.002, 0., 0.])
    assert np.allclose(_cap_with_priority(np.array([.0005, 0., 0.]), np.array([0., .001, 0.]), .002), [.0005, .001, 0.])


# ------------------------------------------------------------------------------------------- tracking

@pytest.mark.parametrize("task_id,builder,kwargs", [
    ("TRIM-01-R", skim, {}), ("CUT-01-R", through_cut, {}), ("CUT-01-R", through_cut, {"tilt_deg": 20.}),
    ("CUT-01-R", through_cut, {"yaw_deg": 30.}),
])
def test_path_v2_on_an_ideal_loop_stays_on_the_segment_and_ends_at_p1(task_id, builder, kwargs):
    task = _task(task_id)
    policy = _path_v2(task, builder, **kwargs)
    log = _plant(policy, task)
    assert policy.done
    rows = _cut_rows(log, policy)
    assert rows
    for row, p0, p1 in rows:
        assert _segment_error(row["measured"][:3], p0, p1) < 1e-9
    for row in log:
        assert np.linalg.norm(row["action"]["target"][:3]) <= LIMIT+1e-12
    cut = next(i for i, w in enumerate(policy.waypoints) if w.cut)
    completed = next(row for row in log if row["index"] == cut and row["waypoint"] > cut)
    assert np.linalg.norm(completed["measured"][:3]-np.asarray(policy.waypoints[cut].pose[:3])) <= .0003


@pytest.mark.parametrize("kind", ["bias", "disp"])
@pytest.mark.parametrize("offset_z", [.002, -.002])
def test_path_v2_on_a_laggy_plant_converges_to_the_skim_line(kind, offset_z):
    """First-order lag 0.7/step + one-step delay (~1.4 steps total lag; the live Native Demo 2 episode
    lags ~0.75 step) and a 2 mm z offset at the cut start."""
    task = _task("TRIM-01-R")
    policy = _path_v2(task, skim)
    log = _plant(policy, task, alpha=.7, delay=1, offset_z=offset_z, kind=kind)
    assert policy.done
    rows = _cut_rows(log, policy)
    errors = np.array([_segment_error(row["measured"][:3], p0, p1) for row, p0, p1 in rows])
    assert errors[1:3].max() > .0015                                   # the offset really hit the edge
    assert errors[11:].max() <= .0003                                  # <= 10 steps after the offset
    for row in log:
        assert np.linalg.norm(row["action"]["target"][:3]) <= LIMIT+1e-12
    cut = rows[-1][0]["waypoint"]
    completed = next(row for row in log if row["index"] == cut and row["waypoint"] > cut)
    assert np.linalg.norm(completed["measured"][:3]-rows[-1][2]) <= .0003     # cut completed at P1


def test_v1_on_the_same_laggy_plant_reproduces_the_diagnosed_offset_and_path_v2_removes_it():
    """Live Demo 2 diagnosis: v1 reaches the cut start while still moving at 2 mm/step, overshoots,
    then aims at the far end point, so the z error decays only over the whole pass."""
    task = _task("TRIM-01-R")
    errors = {}
    for name, policy in (("v1", WaypointPolicy(task, skim(task))), ("path_v2", _path_v2(task, skim))):
        log = _plant(policy, task, alpha=.7, delay=1)
        assert policy.done
        rows = _cut_rows(log, policy)
        errors[name] = np.array([_segment_error(row["measured"][:3], p0, p1) for row, p0, p1 in rows])
    assert errors["v1"][11:31].min() > .0015                           # v1: still > 1.5 mm 10-30 steps in
    assert errors["path_v2"][11:].max() <= .0003
    # One step from the same 2 mm z error: v1 corrects ~0.01 mm, path_v2 half the error (1 mm).
    plan = skim(task)
    measured = np.asarray(plan[1].pose, float)
    measured = np.r_[measured[:3]+[0., 0., .002], measured[3:]]
    v1 = WaypointPolicy(task, plan, index=2, _start_time=0.)
    assert abs(v1({"proprio": {"tool_pose": measured}, "timestamp_s": 0.})["target"][2]) < 5e-5
    tracked = _path_v2(task, skim)
    tracked.index, tracked._start_time = 3, 0.                         # path_v2 plan: above, slow, start, cut
    assert tracked.waypoints[3].cut
    correction = tracked({"proprio": {"tool_pose": measured}, "timestamp_s": 0.})["target"][:3]
    assert correction[2] == pytest.approx(-.001, abs=1e-9)


def test_path_v2_saw_keeps_the_v1_oscillation_and_the_segment():
    task = _task("CUT-01-R")
    policy = _path_v2(task, through_cut, saw_amplitude_m=.003, saw_period_s=.4)
    log = _plant(policy, task)
    assert policy.done
    rows = _cut_rows(log, policy)
    along = Rotation.from_quat(policy.waypoints[3].pose[3:]).apply([0., 1., 0.])
    lateral = np.array([(row["measured"][:3]-p0)@along for row, p0, p1 in rows])
    assert lateral.max() > .002 and lateral.min() < -.002              # the saw stroke is executed
    for row, p0, p1 in rows:                                           # off-saw error stays on the segment
        offset = row["measured"][:3]-((row["measured"][:3]-p0)@along)*along
        assert _segment_error(offset, p0-(p0@along)*along, p1-(p1@along)*along) < 1e-9
        assert np.linalg.norm(row["action"]["target"][:3]) <= LIMIT+1e-12


def test_track_path_flag_tracks_a_non_cut_waypoint():
    task = _task("CUT-01-R")
    a = tuple(task.reference_edge_pose_world(0.))
    b = tuple(np.r_[np.asarray(a[:3])+[.02, 0., 0.], a[3:]])
    policy = WaypointPolicy(task, [Waypoint(a), Waypoint(b, track_path=True)], path_tracking=True)
    measured = np.r_[np.asarray(a[:3])+[0., 0., .001], a[3:]]
    policy.index, policy._start_time = 1, 0.
    action = policy({"proprio": {"tool_pose": measured}, "timestamp_s": 0.})
    assert not action["cut_enabled"] and action["target"][2] == pytest.approx(-.0005)
    assert policy.last_tracking["cross_track_error_m"] == pytest.approx(.001)


# ------------------------------------------------------------------------------------------- approach shaping

@pytest.mark.parametrize("task_id,builder", [("TRIM-01-R", skim), ("CUT-01-R", through_cut)])
def test_slow_approach_and_settle_hold_timing(task_id, builder):
    task = _task(task_id)
    base, plan = builder(task), builder(task, **PLAN)
    assert len(plan) == len(base)+1
    above, slow, start, cut = (np.asarray(w.pose, float) for w in plan[:4])
    assert np.allclose(above, base[0].pose) and np.allclose(start, base[1].pose) and plan[3] == base[2]
    assert np.linalg.norm(slow[:3]-start[:3]) == pytest.approx(.01) and plan[1].speed_m_s is None
    assert plan[2].speed_m_s == .02 and plan[2].hold_s == .3 and plan[1].hold_s == 0. and not plan[2].cut
    policy = WaypointPolicy(task, plan, **POLICY)
    log = _plant(policy, task)
    slow_steps = [row for row in log if row["waypoint"] == 2]          # actions issued toward the start
    moving = [row for row in slow_steps if np.linalg.norm(row["action"]["target"][:3]) > 1e-9]
    assert all(np.linalg.norm(row["action"]["target"][:3]) <= .02*task.control_dt_s+1e-12 for row in slow_steps)
    assert len(moving) == pytest.approx(.01/(.02*task.control_dt_s), abs=1)        # 25 steps of 0.4 mm
    held = [row for row in slow_steps if np.linalg.norm(row["measured"][:3]-start[:3]) <= .0003]
    assert len(held) == round(.3/task.control_dt_s)                                 # 0.3 s settle at the start
    assert all(not row["action"]["cut_enabled"] for row in slow_steps)
    first_cut = next(row for row in log if row["waypoint"] == 3)
    assert first_cut["action"]["cut_enabled"] and first_cut["step"] == slow_steps[-1]["step"]+1


def test_slice_push_path_v2_shapes_both_cut_starts_and_fits_the_episode():
    task = _task("SLICE-PUSH-01")
    base, plan = slice_push(task), slice_push(task, **PLAN)
    labels = [w.label for w in plan]
    assert labels == ["approach_0", "start_0", "start_0", "cut_0", "hold_0", "lift_0", "move_1", "move_1",
                      "cut_1", "hold_1", "raise_in_kerf", "push", "push_hold", "release", "retract"]
    assert [w.label for w in base] == [label for i, label in enumerate(labels) if i not in (1, 6)]
    for start in (2, 7):
        assert plan[start].speed_m_s == .02 and plan[start].hold_s == .3 and plan[start+1].cut
        assert np.allclose(plan[start].pose, base[start-1 if start == 2 else start-2].pose)
    policy = WaypointPolicy(task, plan, **POLICY)
    log = _plant(policy, task, steps=800, start_offset=(0., 0., 0.))
    # Ideal tracking: done at 15.16 s of the 16 s episode (v1: 13.64 s); the slow approaches and settles add ~1.5 s.
    assert policy.done and log[-1]["step"]*task.control_dt_s < task.duration_s-.5
    for row in log:
        if row["action"]["cut_enabled"]:
            x = task.points_to_local(row["measured"][:3])[0]
            assert row["phase"] in ("cut_0", "cut_1") and min(abs(x+.03), abs(x-.03)) < 1e-6


def test_shape_approaches_validation_and_first_waypoint_start():
    task = _task("CUT-01-R")
    plan = through_cut(task)[1:]                                       # the start is the first waypoint
    shaped = shape_approaches(plan, approach_speed_m_s=.02, settle_s=.3)
    assert len(shaped) == len(plan) and shaped[0].hold_s == .3 and shaped[0].speed_m_s is None
    with pytest.raises(ValueError):
        shape_approaches(plan, approach_speed_m_s=0.)
    with pytest.raises(ValueError):
        shape_approaches(plan, approach_speed_m_s=.02, approach_slow_distance_m=0.)


# ------------------------------------------------------------------------------------------- run_wp10_live CLI

def _script():
    spec = importlib.util.spec_from_file_location("run_wp10_live_path_v2", ROOT/"scripts/run_wp10_live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(argv):
    module, captured = _script(), {}

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


def test_cli_parses_the_profile_and_defaults_to_v1():
    assert _parse(["--task", "CUT-01-R"])["config"]["controller_profile"] == "v1"
    out = _parse(["--task", "TRIM-01-R", "--controller", "scripted", "--controller-profile", "path_v2"])
    assert out["status"] == 1 and out["config"]["controller_profile"] == "path_v2"   # stopped at runtime launch
    for bad in (["--task", "CUT-01-R", "--controller-profile", "path_v2"],           # not scripted
                ["--task", "CUT-01-R", "--controller", "scripted", "--controller-profile", "v9"]):
        with pytest.raises(SystemExit):
            _parse(bad)


def _stub_env(task_id):
    task = _task(task_id)
    env = SimpleNamespace(task=task, action_limits=None, time_s=0.)
    env.backend = SimpleNamespace(proprioception=lambda: {"timestamp_s": 0., "tool_pose": task.reference_edge_pose_world(0.)})
    return env


@pytest.mark.parametrize("task_id,builder", [("CUT-01-R", through_cut), ("TRIM-01-R", skim), ("SLICE-PUSH-01", slice_push)])
def test_scripted_policy_applies_the_profile(task_id, builder):
    module = _script()
    env = _stub_env(task_id)
    base = dict(motion="push", speed=.01, angle=0., offset=0., edge_feedback="measured", position_tolerance=None)
    v1 = module.scripted_policy(env, Namespace(**base, controller_profile="v1"))
    legacy = module.scripted_policy(env, Namespace(**base))                       # namespaces without the key
    expected_v1 = builder(env.task, **({} if builder is skim else {"cut_speed_m_s": .01}))
    assert v1.waypoints == legacy.waypoints == expected_v1
    assert not v1.path_tracking and v1.position_tolerance_m == .0005
    v2 = module.scripted_policy(env, Namespace(**base, controller_profile="path_v2"))
    assert v2.path_tracking and v2.cross_track_gain == .5 and v2.position_tolerance_m == .0003
    assert v2.waypoints == builder(env.task, **PLAN, **({} if builder is skim else {"cut_speed_m_s": .01}))
    assert v2.edge_estimator is not None
    override = module.scripted_policy(env, Namespace(**{**base, "position_tolerance": .0004}, controller_profile="path_v2"))
    assert override.position_tolerance_m == .0004
    receipt = module.controller_profile_receipt(Namespace(controller_profile="path_v2", position_tolerance=.0004))
    assert receipt == {**CONTROLLER_PROFILES["path_v2"], "position_tolerance_m": .0004}
    assert module.controller_profile_receipt(Namespace(controller_profile="v1", position_tolerance=None)) == \
        {"name": "v1", "path_tracking": False}


# ------------------------------------------------------------------------------------------- kinematic preflight

@pytest.mark.cpu_sim
def test_kinematic_trim_episode_with_path_v2_tracks_the_skim_line():
    pytest.importorskip("warp")
    from robotic_cutting_sim.kinematic_env import make_kinematic_env
    env = make_kinematic_env("TRIM-01-R", particle_spacing_m=.004, material_dt=.0005, scenario=SCENARIO,
                             action_limits=ActionLimits())
    policy = WaypointPolicy(env.task, skim(env.task, **PLAN), **POLICY)
    observation, _ = env.reset()
    p0, p1 = (np.asarray(policy.waypoints[i].pose[:3]) for i in (2, 3))
    errors = []
    for _ in range(int(round(env.task.duration_s/env.task.control_dt_s))):
        action = policy(observation)
        if policy.done:
            break
        observation, _, terminated, truncated, info = env.step(action)
        assert info["action_report"]["applied_translation_m"] <= LIMIT+1e-9 and info["ik_status"] == "ok"
        assert not truncated
        if action["cut_enabled"]:
            errors.append(_segment_error(np.asarray(observation["proprio"]["tool_pose"][:3]), p0, p1))
    assert policy.done and errors and max(errors) <= .0003
