"""``reference_command_policy_v1``: the compiled goal-planner schedule as ``edge_delta_v1`` actions (P1-5 T5.4).

A ``ConfiguredTaskSpec`` carries the dense ideal-tracking reference schedule of ``goal_planner_v1`` (one knot
per control step, ``scene_config.ideal_schedule``), built from the scene, goal, tool profile and planner
options only: no particle, label, component or oracle input. Control step ``k`` (1-based, ending at
``t = k * dt``) moves the edge from knot ``k-1`` to knot ``k`` with the digit armed on that motion:

* target pose ``reference_edge_pose_world(k * dt)``;
* armed digit ``active_cut_index((k - 1) * dt)`` (``-1`` = disarmed).

``reference_command_actions`` turns the targets into open-loop ``edge_delta_v1`` actions with the same frozen
NumPy path as ``recorded_command_policy_v1`` (``policies_batched.recorded_command_actions``): each action is
computed against the reconstructed commanded pose, so float32 rounding does not accumulate; the residual of
the reconstruction against the schedule is reported. Under ideal tracking the material sees the
reconstructed poses (no robot, reaction not fed back).

:class:`ReferenceCommandPolicyV1` is the device table policy (``[E, T, 8]``, gathered by ``step_count``) for a
vector env; envs past the table end get the HOLD row (no motion, disarmed).
"""
import numpy as np

from .policies_batched import RecordedCommandPolicyV1, recorded_command_actions

POLICY_REFERENCE = "reference_command_policy_v1"


def reference_schedule(task, steps=None, start_step=0):
    """Per-control-step reference of a compiled task (CPU, NumPy).

    ``start_step`` > 0 starts at knot ``start_step`` (initial pose = that knot; window tests only).

    Returns ``dict(targets [T,7] WORLD XYZ+XYZW float64, cut_enabled [T] bool, cut_index [T] int32 (digit,
    0 when disarmed, the submitted default), resolved_cut [T] int32 (-1 disarmed), labels [T] str (planner
    phase pursued during the step), initial_pose [7], steps T, control_dt_s)``."""
    dt = float(task.control_dt_s)
    declared = int(round(float(task.duration_s) / dt))
    start_step = int(start_step)
    T = declared - start_step if steps is None else int(steps)
    if not 0 <= start_step < declared or not 1 <= T <= declared - start_step:
        raise ValueError(f"start_step + steps must lie in [1, {declared}]")
    times = dt * (start_step + np.arange(T + 1))
    times[-1] = min(times[-1], float(task.duration_s))
    targets = np.stack([np.asarray(task.reference_edge_pose_world(float(t)), float) for t in times[1:]])
    resolved = np.asarray([task.active_cut_index(float(t)) for t in times[:-1]], np.int32)
    labels = [str(task.reference_phase_labels[min(k, len(task.reference_phase_labels) - 1)])
              for k in range(start_step, start_step + T)]
    return dict(targets=targets, cut_enabled=resolved >= 0, cut_index=np.where(resolved >= 0, resolved, 0).astype(np.int32),
                resolved_cut=resolved, labels=labels,
                initial_pose=np.asarray(task.reference_edge_pose_world(float(times[0])), float), steps=T,
                start_step=start_step, control_dt_s=dt)


def reference_command_actions(task, steps=None, start_step=0):
    """``(actions [T,8] float32, reconstructed [T,7] float64, schedule, report)`` of a compiled task."""
    schedule = reference_schedule(task, steps, start_step)
    actions, reconstructed, report = recorded_command_actions(
        schedule["targets"], schedule["cut_enabled"], schedule["cut_index"], schedule["initial_pose"],
        task.action_limits(), task.T_world_workpiece, int(task.max_cuts))
    armed = actions[:, 6] > 0
    report.update(policy=POLICY_REFERENCE, task_id=task.task_id, config_sha256=task.config_sha256,
                  armed_steps=int(armed.sum()),
                  digit_schedule_equal=bool(np.array_equal(np.where(armed, schedule["cut_index"], -1),
                                                           schedule["resolved_cut"])),
                  inputs="scene, goal, tool profile and planner options only (compiled reference schedule)")
    return actions, reconstructed, schedule, report


class ReferenceCommandPolicyV1(RecordedCommandPolicyV1):
    """Device table of planner-schedule actions; env ``n`` follows table ``table_index[n]`` (see the module)."""

    version = POLICY_REFERENCE

    @classmethod
    def from_tasks(cls, tasks, table_index=None, *, device, steps=None):
        """One table per distinct task (by ``config_sha256``); ``table_index`` defaults to each env's own task."""
        keys, tables, reports, env_table = {}, [], [], []
        for task in tasks:
            key = task.config_sha256
            if key not in keys:
                actions, _, _, report = reference_command_actions(task, steps)
                keys[key] = len(tables)
                tables.append(actions)
                reports.append(report)
            env_table.append(keys[key])
        if len({len(t) for t in tables}) != 1:
            raise ValueError("reference tables must have equal lengths")
        policy = cls(tables, env_table if table_index is None else table_index, device=device)
        policy.reports = reports
        return policy
