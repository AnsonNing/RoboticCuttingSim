"""WP-11 Phase C5b (CPU): controller profile path_v3 and goal trim_bump_rho10_clear.

* Diagnosis: on the plant fitted to the C5 live trace (tests/c5b_plant.py) the path_v2 delta law
  (command += target - measured) has lightly damped / unstable closed-loop poles at ~8.4 steps per
  period (~6 Hz) and reproduces the C5 PRE_CUT dip below the target.
* path_v3 (absolute target = planned reference + slowly adapting bias): real dominant bias-loop pole,
  no limit cycle, settle error <= 0.05 mm, no dip below the target > 0.05 mm during PRE_CUT and SKIM
  cross-track <= 0.1 mm on that plant and on perturbed variants (extra delay, pose-dependent sag).
* v1/path_v2 profiles, plans and compiled task ids are unchanged.
* The chip-clearing sweep is planned from the scene config only.
"""
import inspect
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from c5b_plant import ARX, rollout
from robotic_cutting_sim import goal_planner
from robotic_cutting_sim.goals import validate_goal
from robotic_cutting_sim.policies import (CONTROLLER_PROFILES, Waypoint, WaypointPolicy, controller_profile,
                                          shape_approaches)
from robotic_cutting_sim.scene_config import GOAL_DIR, SceneConfigError, compile_task

P = np.polynomial.polynomial
SCENE = "trim01_wp11_workcell"
GOAL = "trim_bump_rho10_clear"
TARGET_Z = .1202            # world: bar top 0.12 + edge clearance 0.2 mm
SKIM_START = np.array([-.022, 0., TARGET_Z])


@pytest.fixture(scope="module")
def clear_task():
    return compile_task(SCENE, GOAL)


@pytest.fixture(scope="module")
def c5_task():
    return compile_task(SCENE, "trim_bump_rho10", planner_options={"max_rotation_deg": 1., "max_translation_m": .001})


def _plan(task):
    return goal_planner.plan(task.scene, task.goal, task.tool_profile, task.controller["planner"])


def _unit_gain(axis):
    (a1, a2), b, _ = ARX[axis]
    return np.r_[1., -a1, -a2], np.asarray(b)*(1-a1-a2)/sum(b)


# ------------------------------------------------------------------ stability (linear analysis)

@pytest.mark.parametrize("axis", [0, 1, 2])
def test_path_v2_delta_law_rings_at_about_6_hz_on_the_fitted_plant(axis):
    """c_k = c_{k-1} + (T - m_{k-1}): (1 - q) A(q) + q B(q) = 0 (q = z^-1)."""
    A, B = _unit_gain(axis)
    roots = 1/P.polyroots(P.polyadd(P.polymul([1., -1.], A), P.polymul([0., 1.], B)))
    dominant = roots[np.argmax(np.abs(roots))]
    period_steps = 2*np.pi/abs(np.angle(dominant))
    assert abs(dominant) > .85 and abs(dominant.imag) > .1
    assert 7.5 < period_steps < 9.5 and 5. < 1/(period_steps*.02) < 7.  # Hz


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_path_v3_bias_loop_has_a_real_stable_dominant_pole(axis):
    """b_k = b_{k-1} + k F(e), e = -G b - offset: (1 - q) A Fden + k q B Fnum = 0, F = alpha / (1 - (1 - alpha) q)."""
    profile = CONTROLLER_PROFILES["path_v3"]
    k, alpha = profile["bias_gain"], profile["error_filter"]
    A, B = _unit_gain(axis)
    roots = 1/P.polyroots(P.polyadd(P.polymul(P.polymul([1., -1.], A), [1., -(1-alpha)]), k*P.polymul([0., 1.], B)*alpha))
    dominant = roots[np.argmax(np.abs(roots))]
    assert np.abs(roots).max() < .95
    assert abs(dominant.imag) < 1e-9 and dominant.real > 0   # non-oscillatory slowest mode
    # one-step-delay reference case of the docstring: z^2 - z + k = 0 has real roots for k <= 1/4
    assert k <= .25 and np.all(np.isreal(np.roots([1., -1., k])))


# ------------------------------------------------------------------ closed loop on the fitted plant

def _metrics(rows):
    pre = [r for r in rows if r["phase"].startswith("PRE_CUT")]
    skim = [r for r in rows if r["phase"].startswith("SKIM")]
    z = np.array([r["measured"][2]-TARGET_Z for r in pre])
    last = np.array([r["measured"][:3] for r in pre[-10:]])
    hold = np.array([r["measured"][2] for r in pre[-15:]])
    cross = np.array([np.hypot(r["measured"][1], r["measured"][2]-TARGET_Z) for r in skim if r["cut"]])
    return dict(pre_cut_min_dz=float(z.min()), settle_error=float(np.linalg.norm(last.mean(0)-SKIM_START)),
                settle_z_peak_to_peak=float(np.ptp(hold)), skim_cross_max=float(cross.max()), skim_steps=len(cross))


def test_path_v2_reproduces_the_c5_pre_cut_dip(c5_task):
    rows, _ = rollout(c5_task, _plan(c5_task), controller_profile("path_v2")[0], max_s=8., stop_after="SKIM", seed=0)
    m = _metrics(rows)
    assert m["pre_cut_min_dz"] < -1.5e-4          # C5 live: -0.217 mm
    assert m["settle_z_peak_to_peak"] > 1e-4      # still ringing at the end of the settle


PLANTS = [dict(seed=0), dict(seed=1), dict(seed=2), dict(seed=3, extra_delay_steps=1),
          dict(seed=4, sag_per_m=.002), dict(seed=5, sag_per_m=-.002)]


@pytest.mark.parametrize("plant", PLANTS, ids=lambda p: "-".join(f"{k}{v}" for k, v in p.items()))
def test_path_v3_settles_without_dip_or_limit_cycle(clear_task, plant):
    rows, policy = rollout(clear_task, _plan(clear_task), controller_profile("path_v3")[0], max_s=10.,
                           stop_after="SKIM", **plant)
    m = _metrics(rows)
    assert m["skim_steps"] > 80
    assert m["pre_cut_min_dz"] >= -5e-5           # never more than 0.05 mm below the target during PRE_CUT
    assert m["settle_error"] <= 5e-5              # settle error <= 0.05 mm
    assert m["settle_z_peak_to_peak"] <= 6e-5     # no limit cycle: only measurement noise (z rms 9 um) remains
    assert m["skim_cross_max"] <= 1e-4            # SKIM cross-track (y, z) <= 0.1 mm
    assert policy.last_tracking is None or policy.last_tracking["command_mode"] == "bias_pose"


def test_path_v3_commands_absolute_bounded_targets(clear_task):
    rows, _ = rollout(clear_task, _plan(clear_task), controller_profile("path_v3")[0], max_s=4., seed=0)
    commands = np.array([r["command"] for r in rows])
    limits = clear_task.action_limits()
    steps = np.linalg.norm(np.diff(commands[:, :3], axis=0), axis=1)
    angles = [(Rotation.from_quat(b)*Rotation.from_quat(a).inv()).magnitude() for a, b in zip(commands[:-1, 3:], commands[1:, 3:])]
    assert steps.max() <= limits.max_translation_m+1e-12 and max(angles) <= limits.max_rotation_rad+1e-9
    task = clear_task
    plan = _plan(task)
    nominal = np.asarray(plan[0].pose, float)
    measured = nominal+np.r_[0., .002, -.001, 0., 0., 0., 0.]   # grasp offset: measured edge != hand-FK edge
    policy = WaypointPolicy(task, plan, limits, edge_estimator=lambda observation: measured.copy(),
                            **controller_profile("path_v3")[0])
    action = policy({"proprio": {"tool_pose": nominal}, "timestamp_s": 0.})
    assert action["mode"] == "cartesian_edge_pose_world" and set(action) <= {"mode", "target", "cut_enabled", "cut_index"}
    # b_0 = nominal - measured: the first command is one bounded step from the commanded (nominal) pose, no jump
    assert np.linalg.norm(action["target"][:3]-nominal[:3]) <= limits.max_translation_m+1e-12
    assert np.allclose(policy._bias_state["bias_position"], nominal[:3]-measured[:3])


def test_path_v3_ideal_schedule_follows_the_plan_exactly(clear_task):
    """On the ideal integrator the bias stays zero: the reference schedule passes through every waypoint."""
    knots = np.asarray(clear_task.edge_position_knots_local_txyz, float)
    T = clear_task.T_world_workpiece
    world = knots[:, 1:]@T[:3, :3].T+T[:3, 3]
    for w in _plan(clear_task):
        assert np.min(np.linalg.norm(world-np.asarray(w.pose[:3]), axis=1)) < 1e-9, w.label
    assert clear_task.duration_s == 28.


def test_path_v3_rejects_sawing_and_bad_gains(clear_task):
    plan = [Waypoint(_plan(clear_task)[0].pose), Waypoint(_plan(clear_task)[1].pose, saw_amplitude_m=.001)]
    policy = WaypointPolicy(clear_task, plan, clear_task.action_limits(), **controller_profile("path_v3")[0])
    edge = np.asarray(plan[0].pose, float)
    with pytest.raises(ValueError, match="sawing"):
        for k in range(400):
            policy({"proprio": {"tool_pose": edge}, "timestamp_s": .02*k})
    with pytest.raises(ValueError):
        WaypointPolicy(clear_task, plan, command_mode="bias_pose", bias_gain=.5)
    with pytest.raises(ValueError):
        WaypointPolicy(clear_task, plan, command_mode="absolute")


# ------------------------------------------------------------------ v1 / path_v2 unchanged

def test_existing_profiles_and_shaping_unchanged():
    assert controller_profile("v1") == ({"path_tracking": False}, {})
    assert controller_profile("path_v2") == (
        {"path_tracking": True, "cross_track_gain": .5, "position_tolerance_m": .0003},
        {"approach_speed_m_s": .02, "approach_slow_distance_m": .01, "settle_s": .3})
    policy = WaypointPolicy(None, [])
    assert policy.command_mode == "delta" and policy._bias_state is None
    plan = [Waypoint((0., 0., .1, 0., 0., 0., 1.)), Waypoint((0., 0., .05, 0., 0., 0., 1.)),
            Waypoint((.05, 0., .05, 0., 0., 0., 1.), cut=True)]
    assert shape_approaches(plan, approach_speed_m_s=.02, settle_s=.3) == shape_approaches(
        plan, approach_speed_m_s=.02, settle_s=.3, presettle_m=None)
    shaped = shape_approaches(plan, approach_speed_m_s=.02, settle_s=.3, presettle_m=.001, presettle_s=.5)
    assert [w.pose[2] for w in shaped] == pytest.approx([.1, .06, .051, .05, .05])
    assert [w.hold_s for w in shaped] == [0., 0., .5, .3, 0.]


def test_compiled_ids_of_existing_goals_unchanged(c5_task):
    assert c5_task.task_id == "CFG-trim01_wp11_workcell-trim_bump_rho10-570db2f5"
    assert compile_task(SCENE, "trim_bump_rho10").task_id == "CFG-trim01_wp11_workcell-trim_bump_rho10-a33f6613"
    slice_push = validate_goal("slice_then_push")
    assert "push_mode" not in slice_push["steps"][1]


# ------------------------------------------------------------------ goal and sweep plan

def test_goal_declares_trim_thresholds_and_chip_move():
    raw = json.loads((GOAL_DIR/f"{GOAL}.json").read_text())
    goal = validate_goal(GOAL)
    trim, move = goal["steps"]
    reference = validate_goal("trim_bump_rho10")
    for key in ("select", "plane_local_z_m", "edge_clearance_m", "minimum_removed_fraction", "minimum_substrate_share",
                "maximum_detached_below_fraction", "minimum_unselected_retained_fraction"):
        assert trim[key] == reference[key], key
    assert goal["planner"]["controller_profile"] == "path_v3" and goal["planner"]["rake_deg"] == 10.
    assert move["type"] == "move_piece" and move["push_mode"] == "surface_sweep"
    assert move["direction_local"] == [1., 0., 0.] and move["minimum_displacement_m"] == .025
    assert move["maximum_other_displacement_m"] == .002 and move["sweep_height_m"] == .002
    assert raw["steps"][1]["selector"] == {"box_local_m": {"min": [-.012, -.012, .0205], "max": [.012, .012, .029]}}


def test_surface_sweep_plan_is_config_only_and_clears_the_bar_top(clear_task):
    assert "task" not in inspect.signature(goal_planner._surface_sweep).parameters
    plan, report = goal_planner.plan_with_report(clear_task.scene, clear_task.goal, clear_task.tool_profile,
                                                 clear_task.controller["planner"])
    labels = [w.label for w in plan]
    assert labels == ["HOME", "REORIENT:dome", "PRE_CUT:dome", "PRE_CUT:dome", "PRE_CUT:dome", "SKIM:dome",
                      "WITHDRAW:dome", "APPROACH_PUSH:move1", "LOWER:move1", "PUSH:move1", "PUSH_HOLD:move1",
                      "RELEASE:move1", "RETRACT:move1", "RETURN_HOME"]
    sweep = report["operations"][-1]
    assert sweep["kind"] == "surface_sweep_push" and not sweep["warnings"]
    # local frame: dome footprint x in [-12, 12] mm, bar end at x = 80 mm, bar top z = 20 mm
    assert sweep["start_local_m"] == pytest.approx([-.022, 0., .022])
    assert sweep["end_local_m"] == pytest.approx([.085, 0., .022])
    assert sweep["orientation_metrics"]["rho_deg"] == pytest.approx(10.) and sweep["orientation_metrics"]["tau_deg"] < 1e-9
    moves = [w for w in plan if w.label.endswith(":move1")]
    assert not any(w.cut for w in moves)
    skim = next(w for w in plan if w.label.startswith("SKIM"))
    for w in moves:
        assert np.allclose(w.pose[3:], skim.pose[3:]) or np.allclose(w.pose[3:], -np.asarray(skim.pose[3:]))
    from robotic_cutting_sim.scene_config import scene_world_pose
    top = scene_world_pose(clear_task.scene)[2]+.02
    section = clear_task.tool_profile.cross_section_xz()
    for w in moves:
        R = Rotation.from_quat(w.pose[3:]).as_matrix()
        lowest = w.pose[2]+(np.c_[section[:, 0], np.zeros(len(section)), section[:, 1]]@R.T)[:, 2].min()
        assert lowest-top > .0015, w.label        # the blade body stays > 1.5 mm above the bar top
    push = clear_task.phase_intervals_s()
    assert "PUSH:move1" in push and "PUSH_HOLD:move1" in push   # the goal evaluator's move window


def test_surface_sweep_field_validation():
    base = json.loads((GOAL_DIR/f"{GOAL}.json").read_text())
    bad = json.loads(json.dumps(base))
    bad["steps"][1]["push_mode"] = "shovel"
    with pytest.raises(SceneConfigError, match="push_mode"):
        validate_goal(bad)
    bad = json.loads(json.dumps(base))
    del bad["steps"][1]["push_mode"]
    with pytest.raises(SceneConfigError, match="surface_sweep"):
        validate_goal(bad)
    bad = json.loads(json.dumps(base))
    bad["steps"][1]["direction_local"] = [0., 0., 1.]
    with pytest.raises((SceneConfigError, ValueError), match="horizontal"):
        compile_task(SCENE, bad)
