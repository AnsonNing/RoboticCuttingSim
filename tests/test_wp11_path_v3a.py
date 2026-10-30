"""WP-11 review finding 8 (CPU): path_v3 bias anti-windup -> opt-in controller profile path_v3a.

* path_v3 is unchanged: an open-loop replay of the recorded C5b live Native episode
  (artifacts/wp11/phaseC5/live_trim_rho10_clear_002) through the current path_v3 reproduces every
  recorded command and policy_tracking row bitwise (tests/c5b_replay.py).
* Fixing path_v3 in place was not possible without changing that run: the policy's own action-limit
  clamp was active on 254 of 1057 C5b controller calls, so a freeze-while-clamped rule changes the
  command sequence (path_v3a differs from step 2 of the replay). Hence path_v3a, path_v3 kept.
* path_v3a: |bias| <= 3 mm / 15 mrad, no bias integration after a clamped command, and the opt-in
  per-waypoint stall timeout ends the waypoint or the episode with an explicit reason. On a blocked
  blade (fitted C5 plant with a floor) path_v3 winds up to > 30 mm; path_v3a stays at the bound and
  the stall is detected. No false stall over the full nominal plan on the fitted plant variants.
"""
import numpy as np
import pytest

import c5b_replay
from c5b_plant import rollout
from robotic_cutting_sim import goal_planner
from robotic_cutting_sim.policies import CONTROLLER_PROFILES, Waypoint, WaypointPolicy, controller_profile
from robotic_cutting_sim.scene_config import compile_task
from test_wp11_phase_c5b import PLANTS, TARGET_Z, _metrics

SCENE, GOAL = "trim01_wp11_workcell", "trim_bump_rho10_clear"
BLOCK_Z = TARGET_Z+.002     # the blade is stopped 2 mm above the skim height


@pytest.fixture(scope="module")
def task():
    return compile_task(SCENE, GOAL)


@pytest.fixture(scope="module")
def plan(task):
    return goal_planner.plan(task.scene, task.goal, task.tool_profile, task.controller["planner"])


def _bias_norms(rows):
    return np.array([np.linalg.norm(r["tracking"]["bias_position_m"]) for r in rows
                     if r["tracking"] and "bias_position_m" in r["tracking"]])


# ------------------------------------------------------------------ profiles

def test_path_v3_profile_unchanged_and_path_v3a_is_path_v3_plus_anti_windup():
    v3 = controller_profile("path_v3")
    assert v3 == ({"path_tracking": True, "position_tolerance_m": .0003, "command_mode": "bias_pose", "bias_gain": .1,
                   "error_filter": .5, "bias_deadband_m": 2e-5, "bias_deadband_rad": 2e-4, "stop_gain": .15,
                   "stop_min_step_m": 5e-6},
                  {"approach_speed_m_s": .02, "approach_slow_distance_m": .01, "settle_s": .3, "presettle_m": .001,
                   "presettle_s": .5})
    v3a = controller_profile("path_v3a")
    assert v3a[1] == v3[1]
    assert v3a[0] == {**v3[0], "bias_limit_m": .003, "bias_limit_rad": .015, "anti_windup": True}
    assert {k: v for k, v in CONTROLLER_PROFILES["path_v3a"].items() if k not in ("name", "bias_limit_m", "bias_limit_rad",
            "anti_windup")} == {k: v for k, v in CONTROLLER_PROFILES["path_v3"].items() if k != "name"}
    policy = WaypointPolicy(None, [], command_mode="bias_pose")
    assert policy.bias_limit_m is None and policy.bias_limit_rad is None and not policy.anti_windup
    assert policy.stall_timeout_s is None and policy.termination_request is None


def test_anti_windup_and_stall_options_are_validated(task, plan):
    for bad in (dict(bias_limit_m=.003), dict(anti_windup=True), dict(stall_timeout_s=1.)):
        with pytest.raises(ValueError, match="bias_pose"):
            WaypointPolicy(task, plan, **bad)          # delta mode (v1/path_v2) never takes them
    for bad in (dict(bias_limit_m=0.), dict(bias_limit_rad=-1.), dict(stall_timeout_s=float("nan")),
                dict(stall_action="abort"), dict(stall_timeouts_by_label={"PRE_CUT:dome": 0.})):
        with pytest.raises(ValueError):
            WaypointPolicy(task, plan, **{**controller_profile("path_v3a")[0], **bad})
    planner = goal_planner.planner_options(task.goal, {"controller_profile": "path_v3a"})
    assert goal_planner.policy_options(planner)["anti_windup"] is True


# ------------------------------------------------------------------ C5b live replay (path_v3 bitwise)

needs_c5b = pytest.mark.skipif(not c5b_replay.available(), reason="C5b live artifact not present")


@needs_c5b
def test_c5b_live_replay_path_v3_bitwise_and_path_v3a_would_differ():
    task, rows, measured0 = c5b_replay.load()
    commands, phases, trackings, calls, _ = c5b_replay.replay(task, rows, measured0, "path_v3")
    result = c5b_replay.compare_to_record(rows, commands, phases, calls, trackings)
    assert result["steps"] == 1056 and result["bitwise_equal_steps"] == 1056 and result["max_abs_difference"] == 0.
    assert result["phases_equal"] and result["modes_equal"] and result["cut_flags_equal"] and result["tracking_equal"]
    # the anti-windup freeze would have been active in the C5b run -> path_v3 cannot be fixed in place
    commands_a, _, trackings_a, _, _ = c5b_replay.replay(task, rows, measured0, "path_v3a")
    assert sum(bool(t and t.get("command_clamped")) for t in trackings_a) > 200
    assert sum(bool(t and t.get("bias_frozen")) for t in trackings_a) > 200
    assert not all(np.array_equal(a, b) for a, b in zip(commands, commands_a))
    assert not any(t and t.get("bias_saturated") for t in trackings_a)   # the 3 mm bound itself is never reached


# ------------------------------------------------------------------ fitted plant (closed loop)

@pytest.mark.parametrize("plant", PLANTS, ids=lambda p: "-".join(f"{k}{v}" for k, v in p.items()))
def test_path_v3a_keeps_the_c5b_settle_quality(task, plan, plant):
    rows, _ = rollout(task, plan, controller_profile("path_v3a")[0], max_s=10., stop_after="SKIM", **plant)
    m = _metrics(rows)
    assert m["skim_steps"] > 80
    assert m["pre_cut_min_dz"] >= -5e-5 and m["settle_error"] <= 5e-5
    assert m["settle_z_peak_to_peak"] <= 6e-5 and m["skim_cross_max"] <= 1e-4


@pytest.mark.parametrize("plant", [dict(seed=0), dict(seed=3, extra_delay_steps=1), dict(seed=4, sag_per_m=.002)],
                         ids=lambda p: "-".join(f"{k}{v}" for k, v in p.items()))
def test_path_v3a_stall_timeout_has_no_false_positive_on_the_nominal_plan(task, plan, plant):
    rows, policy = rollout(task, plan, {**controller_profile("path_v3a")[0], "stall_timeout_s": 1.}, max_s=28., **plant)
    assert policy.done and policy.stall_events == [] and policy.termination_request is None


def test_blocked_blade_path_v3_winds_up_path_v3a_is_bounded(task, plan):
    rows, policy = rollout(task, plan, controller_profile("path_v3")[0], max_s=12., seed=0, floor_z=BLOCK_Z)
    assert _bias_norms(rows).max() > .02                     # the finding: path_v3 integrates without bound
    assert not policy.done and policy.phase == "PRE_CUT:dome"
    rows, policy = rollout(task, plan, controller_profile("path_v3a")[0], max_s=12., seed=0, floor_z=BLOCK_Z)
    norms = _bias_norms(rows)
    assert norms.max() <= .003*(1+1e-12) and norms.max() > .0029   # bounded, and the bound is reached
    assert any(r["tracking"].get("bias_saturated") for r in rows if r["tracking"])
    assert min(r["command"][2] for r in rows) >= TARGET_Z-.003-1e-9   # pressing depth below the plan <= the bound
    assert policy.stall_events == []                          # stall detection is opt-in


def test_blocked_blade_stall_ends_the_episode_with_an_explicit_reason(task, plan):
    options = {**controller_profile("path_v3a")[0], "stall_timeout_s": 1.}
    rows, policy = rollout(task, plan, options, max_s=12., seed=0, floor_z=BLOCK_Z)
    assert policy.done and policy.termination_request == "WAYPOINT_STALL:PRE_CUT:dome"
    (event,) = policy.stall_events
    assert event["reason"] == "WAYPOINT_STALL" and event["label"] == "PRE_CUT:dome" and event["action"] == "end_episode"
    assert event["position_error_m"] > .0009 and event["timeout_s"] == 1.   # presettle waypoint 1 mm above the target
    assert rows[-1]["time_s"] < 6.                            # detected about 1 s after the blade stopped
    assert _bias_norms(rows).max() <= .003*(1+1e-12)
    observation = {"proprio": {"tool_pose": rows[-1]["measured"]}, "timestamp_s": rows[-1]["time_s"]}
    action = policy(observation)                              # after the stall: a cut-disabled zero-delta hold
    assert action["mode"] == "cartesian_edge_delta" and not action["cut_enabled"] and not np.any(action["target"])


def test_blocked_blade_stall_can_skip_the_waypoint(task, plan):
    options = {**controller_profile("path_v3a")[0], "stall_timeout_s": 1., "stall_action": "skip_waypoint"}
    rows, policy = rollout(task, plan, options, max_s=12., seed=0, floor_z=BLOCK_Z)
    assert policy.termination_request is None and len(policy.stall_events) >= 2
    first = policy.stall_events[0]
    assert first["label"] == "PRE_CUT:dome" and first["action"] == "skip_waypoint"
    indices = [e["waypoint_index"] for e in policy.stall_events]
    assert indices == sorted(indices) and len(set(indices)) == len(indices)   # every stall moved on
    assert any("stalls" in (r["tracking"] or {}) for r in rows)


def test_per_label_stall_timeout_override(task, plan):
    options = {**controller_profile("path_v3a")[0], "stall_timeouts_by_label": {"PRE_CUT:dome": .5}}
    _, policy = rollout(task, plan, options, max_s=12., seed=0, floor_z=BLOCK_Z)
    assert policy.termination_request == "WAYPOINT_STALL:PRE_CUT:dome" and policy.stall_events[0]["timeout_s"] == .5


def test_stall_timer_restarts_inside_the_tolerance():
    """A waypoint held within tolerance (a hold) never stalls, however long the hold."""
    pose = (0., 0., .2, 0., 0., 0., 1.)
    plan = [Waypoint(pose), Waypoint(pose, hold_s=5., label="HOLD")]
    task = type("T", (), {"control_dt_s": .02, "hand_from_edge_xyz_xyzw": None, "T_world_workpiece": np.eye(4)})()
    policy = WaypointPolicy(task, plan, **{**controller_profile("path_v3a")[0], "stall_timeout_s": .2})
    for k in range(300):
        policy({"proprio": {"tool_pose": np.asarray(pose)}, "timestamp_s": .02*k})
    assert policy.done and policy.stall_events == []
