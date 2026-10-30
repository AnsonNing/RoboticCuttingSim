"""Open-loop replay of the recorded WP-11 C5b live Native episode through the path_v3/path_v3a controllers.

The live run (``docs/evidence/wp11/phaseC5b_live_trim_clear_001.json``, attempt 002,
``artifacts/wp11/phaseC5/live_trim_rho10_clear_002``) recorded, for every policy step, the
observation the scripted controller saw (``input_observation_proprio``, ``input_timestamp_s``),
the measured tool edge after the step (``measured.tool_pose``, the ``--edge-feedback measured``
estimator source) and the submitted absolute command. The measured edge at t = 0 is not in
``actions.jsonl``; it is rebuilt from the recorded tool-body pose at t = 0
(``robot_state.npz``) with the same ``physical_tool.measured_edge_pose`` the runtime used (checked
bitwise on step 1). Feeding these recorded inputs to a fresh controller reproduces the controller
outputs of the run; two controllers that produce bitwise identical commands from the same inputs
would also have produced the identical closed-loop run (same command -> same simulator input).
CPU only; skipped by callers when the artifact is absent.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"artifacts/wp11/phaseC5/live_trim_rho10_clear_002"
SCENE, GOAL = "trim01_wp11_workcell", "trim_bump_rho10_clear"
TASK_ID = "CFG-trim01_wp11_workcell-trim_bump_rho10_clear-699d8cee"
NON_POLICY_PHASES = ("HOLD_AFTER_STOP", "RETURN_HOME")


def available():
    return (RUN/"episode_000/actions.jsonl").exists() and (RUN/"episode_000/robot_state.npz").exists()


def load(run=RUN):
    from robotic_cutting_sim.physical_tool import measured_edge_pose
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.tool_profile import tool_spec_for_task
    task = compile_task(SCENE, GOAL)
    if task.task_id != TASK_ID:
        raise AssertionError(f"compiled {task.task_id}, recorded {TASK_ID}")
    rows = [json.loads(line) for line in (run/"episode_000/actions.jsonl").open()]
    policy_rows = [r for r in rows if not str(r.get("policy_phase")).startswith(NON_POLICY_PHASES)]
    state = np.load(run/"episode_000/robot_state.npz", allow_pickle=True)
    tool = [i for i, p in enumerate(state["body_prim_paths"]) if str(p).endswith("CuttingTool_0")][0]
    spec = tool_spec_for_task(task)
    if not np.array_equal(measured_edge_pose(np.asarray(rows[0]["measured"]["tool_body_pose_xyzw"]), spec),
                          np.asarray(rows[0]["measured"]["tool_pose"])):
        raise AssertionError("edge reconstruction from the tool-body pose is not bitwise (step 1)")
    if float(state["time_s"][0]) != 0.:
        raise AssertionError("robot_state.npz does not start at t = 0")
    measured0 = measured_edge_pose(np.asarray(state["body_pose"][0, tool], float), spec)
    return task, policy_rows, measured0


def replay(task, policy_rows, measured0, profile="path_v3", **overrides):
    """(commands, phases, trackings, policy): one controller call per recorded policy step, plus the final
    (unrecorded) call at which the run left the RETRACT phase."""
    from robotic_cutting_sim import goal_planner
    from robotic_cutting_sim.policies import WaypointPolicy, controller_profile
    options = dict(task.controller["planner"])
    waypoints = goal_planner.plan(task.scene, task.goal, task.tool_profile, options)
    policy_options = dict(goal_planner.policy_options(options))
    if profile != options["controller_profile"]:
        policy_options.update(controller_profile(profile)[0])
    policy_options.update(overrides)
    feed = {"edge": np.asarray(measured0, float)}
    policy = WaypointPolicy(task, waypoints, task.action_limits(), edge_estimator=lambda _: feed["edge"].copy(),
                            **policy_options)
    commands, phases, trackings, calls = [], [], [], []
    for k, row in enumerate(policy_rows):
        observation = {"proprio": {key: np.asarray(v, float) for key, v in row["input_observation_proprio"].items()},
                       "timestamp_s": float(row["input_timestamp_s"])}
        action = policy(observation)
        commands.append(np.asarray(action["target"], float).copy())
        calls.append(dict(mode=action["mode"], cut_enabled=action["cut_enabled"], cut_index=action.get("cut_index")))
        phases.append(policy.phase)
        trackings.append(None if policy.last_tracking is None else dict(policy.last_tracking))
        feed["edge"] = np.asarray(row["measured"]["tool_pose"], float)
    last = policy_rows[-1]
    observation = {"proprio": {key: np.asarray(v, float) for key, v in last["observation_proprio"].items()},
                   "timestamp_s": float(last["time_s"])}
    action = policy(observation)
    commands.append(np.asarray(action["target"], float).copy())
    calls.append(dict(mode=action["mode"], cut_enabled=action["cut_enabled"], cut_index=action.get("cut_index")))
    phases.append(policy.phase)
    trackings.append(None if policy.last_tracking is None else dict(policy.last_tracking))
    return commands, phases, trackings, calls, policy


def compare_to_record(policy_rows, commands, phases, calls, trackings=None):
    """Max |replayed - submitted| over the recorded steps, exact-equality count and phase/mode/cut agreement
    (and, with ``trackings``, equality of the recorded ``policy_tracking`` diagnostics)."""
    recorded = np.array([r["submitted"]["target"] for r in policy_rows], float)
    replayed = np.array(commands[:len(policy_rows)])
    return dict(steps=len(policy_rows), max_abs_difference=float(np.abs(recorded-replayed).max()),
                bitwise_equal_steps=int(sum(np.array_equal(a, b) for a, b in zip(recorded, replayed))),
                phases_equal=all(p == r["policy_phase"] for p, r in zip(phases, policy_rows)),
                modes_equal=all(c["mode"] == r["submitted"]["mode"] for c, r in zip(calls, policy_rows)),
                cut_flags_equal=all(c["cut_enabled"] == r["submitted"]["cut_enabled"] for c, r in zip(calls, policy_rows)),
                final_phase=phases[-1],
                **({} if trackings is None else dict(tracking_equal=all(
                    json.loads(json.dumps(t)) == r.get("policy_tracking") for t, r in zip(trackings, policy_rows)))))
