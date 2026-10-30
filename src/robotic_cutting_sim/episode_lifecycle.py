"""Episode lifecycle (WP-11 Phase D): state machine, physical return-to-home and the home outcome.

``reset()`` is a simulator state reset (robot, knife pose/velocity, gripper
hold, workpiece geometry, TLMPM state/codes, clock, history, re-stabilized
grasp). ``return_home()`` is a physical robot motion executed through the same
runtime, never a teleport. The two are distinct and an RL episode is never
forced to return home.

States (``EPISODE_STATES``)::

    RESETTING -> READY -> RUNNING -> TASK_COMPLETE | TASK_FAILED
        -> RETRACTING -> RETURNING_HOME -> FINISHED -> RESETTING -> READY

``TASK_COMPLETE`` means the task phase *ended normally* (time limit or the
controller declared it done); it is not task success. ``TASK_FAILED`` means it
was terminated (grasp lost, invalid contact data, repeated IK failures). Task
success is only ever ``env.evaluate()``; the home outcome is a separate
``home_outcome`` (PASS / FAIL / NOT_APPLICABLE with a reason) and the two are
never merged into one boolean.

Return-to-home plan (``plan_return_home``; pure numpy, IK/FK/checker injected):

1. RETRACTING: straight Cartesian lift of the edge along world +z (orientation
   held) to the HOME edge height, IK chained from the measured joints;
2. RETURNING_HOME: joint-space move to the hand-down HOME orientation at the
   lifted position, then to q_home (or, if that rotate pose is infeasible, the
   direct joint-space move to q_home); then a settle hold at q_home.

All segments are time-scaled at the control rate under the declared joint
velocity (hard gate) and acceleration limits, and the whole sampled path must
pass the geometric preflight (C2 collision checker including the held knife)
before anything moves. Cutting stays disabled throughout.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
RETURN_HOME_CONFIG_PATH = ROOT/"configs/wp11/return_home_v1.json"

EPISODE_STATES = ("RESETTING", "READY", "RUNNING", "TASK_COMPLETE", "TASK_FAILED",
                  "RETRACTING", "RETURNING_HOME", "FINISHED")
TRANSITIONS = {
    None: ("RESETTING",),
    "RESETTING": ("READY", "RESETTING"),
    "READY": ("RUNNING", "RESETTING"),
    "RUNNING": ("TASK_COMPLETE", "TASK_FAILED", "RESETTING"),
    "TASK_COMPLETE": ("RETRACTING", "FINISHED", "RESETTING"),
    "TASK_FAILED": ("RETRACTING", "FINISHED", "RESETTING"),
    "RETRACTING": ("RETURNING_HOME", "FINISHED", "RESETTING"),
    "RETURNING_HOME": ("FINISHED", "RESETTING"),
    "FINISHED": ("RESETTING",),
}
TASK_ENDED = ("TASK_COMPLETE", "TASK_FAILED")
HOME_STATUSES = ("PASS", "FAIL", "NOT_APPLICABLE")
RETURN_LABELS = ("RETRACTING", "ROTATE_HAND_DOWN", "MOVE_HOME", "SETTLE_HOME")


class IllegalTransition(RuntimeError):
    """An episode-state transition that the lifecycle does not allow."""


class EpisodeStateMachine:
    """Explicit episode states with a per-episode transition log (restarted at RESETTING); illegal transitions raise."""

    def __init__(self):
        self.state = None
        self.termination_reason = None   # why the task phase ended (TASK_COMPLETE / TASK_FAILED)
        self.home_outcome = None         # PASS / FAIL / NOT_APPLICABLE after a return-home attempt
        self.history = []

    def transition(self, to, *, time_s=None, reason=None):
        if to not in EPISODE_STATES:
            raise IllegalTransition(f"Unknown episode state {to!r}")
        if to not in TRANSITIONS[self.state]:
            raise IllegalTransition(f"Illegal episode transition {self.state} -> {to}")
        if to == "RESETTING":   # the log, termination reason and home outcome are per episode
            self.history, self.termination_reason, self.home_outcome = [], None, None
        self.history.append({"from": self.state, "to": to, "time_s": None if time_s is None else float(time_s),
                             "reason": reason})
        if to in TASK_ENDED:
            self.termination_reason = reason
        self.state = to
        return to

    def require(self, states, action):
        if self.state not in states:
            raise IllegalTransition(f"{action} is not allowed in episode state {self.state} (needs one of {list(states)})")

    def snapshot(self):
        """Non-privileged lifecycle view (no material, force or oracle state)."""
        return {"state": self.state, "termination_reason": self.termination_reason,
                "home_outcome": self.home_outcome, "transitions": len(self.history)}


def load_return_home_config(path=None):
    config = json.loads(Path(path or RETURN_HOME_CONFIG_PATH).read_text())
    if config.get("version") != "return_home_v1":
        raise ValueError("Expected a return_home_v1 config")
    return config


# ---------------------------------------------------------------- time scaling

def trapezoid_profile(distance, v_max, a_max, dt):
    """Arc length samples s(t_k), k = 1..N (s_N = distance), of a rest-to-rest trapezoid at rate dt."""
    distance, v_max, a_max, dt = map(float, (distance, v_max, a_max, dt))
    if min(v_max, a_max, dt) <= 0 or not np.isfinite([distance, v_max, a_max, dt]).all() or distance < 0:
        raise ValueError("A trapezoid needs a finite nonnegative distance and positive limits")
    if distance < 1e-12:
        return np.zeros(0)
    t_acc = v_max/a_max
    if a_max*t_acc**2 >= distance:   # triangular
        t_acc = np.sqrt(distance/a_max)
        v_peak, t_flat = a_max*t_acc, 0.
    else:
        v_peak, t_flat = v_max, (distance-a_max*t_acc**2)/v_max
    total = 2*t_acc+t_flat
    steps = max(1, int(np.ceil(total/dt-1e-9)))
    scale = total/(steps*dt)   # stretch so the profile ends exactly on a control step (only slows it)
    t = np.arange(1, steps+1)*dt*scale

    def s(t):
        if t <= t_acc:
            return .5*a_max*t*t
        if t <= t_acc+t_flat:
            return .5*a_max*t_acc**2+v_peak*(t-t_acc)
        r = max(total-t, 0.)
        return distance-.5*a_max*r*r
    out = np.array([s(x) for x in t])
    out[-1] = distance
    return out


def joint_segment(q_a, q_b, limits, dt):
    """Synchronized trapezoid from q_a to q_b: (N, 7) joint samples (excluding q_a)."""
    q_a, q_b = np.asarray(q_a, float), np.asarray(q_b, float)
    delta = q_b-q_a
    largest = float(np.abs(delta).max())
    if largest < 1e-12:
        return np.zeros((0, 7))
    s = trapezoid_profile(largest, limits["max_velocity_rad_s"], limits["max_acceleration_rad_s2"], dt)
    return q_a+np.outer(s/largest, delta)


def trajectory_limits(q, dt, q_prev=None):
    """Max |velocity| and |acceleration| per joint of a control-rate trajectory starting from rest at q_prev."""
    q = np.asarray(q, float)
    if q_prev is not None:
        q = np.vstack([np.asarray(q_prev, float)[None], q])
    if len(q) < 2:
        return 0., 0.
    v = np.diff(q, axis=0)/dt
    v = np.vstack([np.zeros((1, q.shape[1])), v, np.zeros((1, q.shape[1]))])  # rest at both ends
    a = np.diff(v, axis=0)/dt
    return float(np.abs(v).max()), float(np.abs(a).max())


# ---------------------------------------------------------------- the return-home plan

def _ik_ok(info, config):
    return (info.get("position_error_m", np.inf) <= config["ik"]["max_position_error_m"]
            and info.get("rotation_error_rad", np.inf) <= config["ik"]["max_rotation_error_rad"])


def _cartesian_lift(q_start, edge_start, clear_z, *, solve_ik, config, dt):
    """(qs, poses, receipt) of the straight edge lift; qs is None with a reason on failure."""
    retract = config["retract"]
    lift = float(clear_z-edge_start[2])
    receipt = {"lift_m": max(lift, 0.), "clear_z_world_m": float(clear_z), "speed_m_s": retract["speed_m_s"]}
    if lift <= 1e-6:
        return np.zeros((0, 7)), np.zeros((0, 7)), {**receipt, "status": "NOT_NEEDED"}
    speed = float(retract["speed_m_s"])
    for attempt in range(4):
        s = trapezoid_profile(lift, speed, retract["acceleration_m_s2"], dt)
        poses = np.array([np.r_[edge_start[:3]+[0., 0., x], edge_start[3:]] for x in s])
        qs, seed, worst = [], np.asarray(q_start, float), 0.
        for i, pose in enumerate(poses):
            q, info = solve_ik(pose, seed)
            if not _ik_ok(info, config):
                return None, poses, {**receipt, "status": "IK_FAILURE", "sample": i, "ik_info": _plain(info)}
            worst = max(worst, float(np.abs(np.asarray(q)-seed).max()))
            qs.append(np.asarray(q, float))
            seed = qs[-1]
        qs = np.asarray(qs)
        velocity, _ = trajectory_limits(qs, dt, q_start)
        if worst <= config["joint_limits"]["max_cartesian_sample_joint_step_rad"] and \
                velocity <= config["joint_limits"]["max_velocity_rad_s"]+1e-9:
            return qs, poses, {**receipt, "status": "PLANNED", "speed_m_s": speed, "samples": len(qs),
                               "max_joint_step_rad": worst, "retimed": attempt}
        speed *= .5   # retime (slower) rather than exceed a joint limit
    return None, poses, {**receipt, "status": "JOINT_VELOCITY_LIMIT", "max_joint_step_rad": worst, "retimed": attempt}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def plan_return_home(q_start, q_home, *, edge_fk, solve_ik, control_dt, joint_lower, joint_upper,
                     checker=None, config=None, home_edge_pose=None):
    """Plan WITHDRAW -> clear -> hand-down -> q_home from measured joints; nothing moves here.

    ``edge_fk(q7) -> WORLD edge XYZ+XYZW``; ``solve_ik(pose, seed_q7) -> (q7, info)`` with
    ``position_error_m``/``rotation_error_rad``; ``checker(qs, modes) -> receipt`` with
    ``status`` PASS/FAIL (modes per sample: ``contact_free`` or ``free``). Without a
    checker the plan is REFUSED when the config requires preflight.

    Returns a dict: ``status`` PLANNED / REFUSED, ``reason`` (None when planned;
    IK_FAILURE, JOINT_LIMIT, JOINT_VELOCITY_LIMIT, PREFLIGHT_FAILED:<check>,
    PREFLIGHT_NOT_RUN), ``q`` (N, 7) control-rate targets, ``labels`` per target,
    ``attempts`` and the receipts.
    """
    config = deepcopy(config or load_return_home_config())
    q_start, q_home = np.asarray(q_start, float), np.asarray(q_home, float)
    lower, upper = np.asarray(joint_lower, float)[:7], np.asarray(joint_upper, float)[:7]
    limits = config["joint_limits"]
    dt = float(control_dt)
    base = {"config_version": config["version"], "q_start": q_start.tolist(), "q_home": q_home.tolist(),
            "control_dt_s": dt, "limits": limits}
    if q_start.shape != (7,) or q_home.shape != (7,) or not np.isfinite(np.r_[q_start, q_home]).all():
        return {**base, "status": "REFUSED", "reason": "INVALID_JOINTS", "q": np.zeros((0, 7)), "labels": [], "attempts": []}
    margin = limits["min_position_margin_rad"]
    if np.any(q_home < lower+margin) or np.any(q_home > upper-margin):
        return {**base, "status": "REFUSED", "reason": "JOINT_LIMIT", "detail": "q_home violates the joint-limit margin",
                "q": np.zeros((0, 7)), "labels": [], "attempts": []}
    home_edge = np.asarray(edge_fk(q_home) if home_edge_pose is None else home_edge_pose, float)
    edge_start = np.asarray(edge_fk(q_start), float)
    lift_q, _, lift = _cartesian_lift(q_start, edge_start, max(home_edge[2], edge_start[2]),
                                      solve_ik=solve_ik, config=config, dt=dt)
    base.update(home_edge_pose_world_xyzw=home_edge.tolist(), start_edge_pose_world_xyzw=edge_start.tolist(),
                retract=lift)
    if lift_q is None:
        return {**base, "status": "REFUSED", "reason": lift["status"], "q": np.zeros((0, 7)), "labels": [], "attempts": []}
    q_lift = lift_q[-1] if len(lift_q) else q_start
    lifted = np.asarray(edge_fk(q_lift), float)
    rotate_pose = np.r_[lifted[:2], home_edge[2], home_edge[3:]]
    q_rot, rot_info = solve_ik(rotate_pose, q_home)
    variants = []
    if _ik_ok(rot_info, config):
        variants.append(("rotate_then_move", [("ROTATE_HAND_DOWN", np.asarray(q_rot, float)), ("MOVE_HOME", q_home)]))
    variants.append(("direct", [("MOVE_HOME", q_home)]))
    attempts, settle = [], int(np.ceil(config["settle_s"]/dt-1e-9))
    for name, segments in variants:
        qs, labels, current = [lift_q], ["RETRACTING"]*len(lift_q), q_lift
        for label, target in segments:
            segment = joint_segment(current, target, limits, dt)
            qs.append(segment)
            labels += [label]*len(segment)
            current = target
        qs.append(np.repeat(q_home[None], settle, axis=0))
        labels += ["SETTLE_HOME"]*settle
        qs = np.vstack([q for q in qs if len(q)])
        velocity, acceleration = trajectory_limits(qs, dt, q_start)
        joint_space = np.array([label != "RETRACTING" for label in labels])
        _, joint_acc = trajectory_limits(qs[joint_space], dt, q_lift) if joint_space.any() else (0., 0.)
        row = {"variant": name, "samples": len(qs), "duration_s": len(qs)*dt,
               "max_joint_velocity_rad_s": velocity, "max_joint_acceleration_rad_s2": acceleration,
               "max_joint_space_acceleration_rad_s2": joint_acc,
               "rotate_ik": _plain(rot_info) if name == "rotate_then_move" else None,
               "min_joint_limit_margin_rad": float(np.minimum(qs-lower, upper-qs).min())}
        if row["min_joint_limit_margin_rad"] < margin:
            attempts.append({**row, "status": "REFUSED", "reason": "JOINT_LIMIT"})
            continue
        if velocity > limits["max_velocity_rad_s"]+1e-9:
            attempts.append({**row, "status": "REFUSED", "reason": "JOINT_VELOCITY_LIMIT"})
            continue
        if joint_acc > limits["max_acceleration_rad_s2"]*(1+1e-6)+1e-9:
            attempts.append({**row, "status": "REFUSED", "reason": "JOINT_ACCELERATION_LIMIT"})
            continue
        if checker is None:
            status = "REFUSED" if config["preflight"]["required"] else "PLANNED"
            attempts.append({**row, "status": status, "reason": "PREFLIGHT_NOT_RUN" if status == "REFUSED" else None,
                             "preflight": {"status": "NOT_RUN"}})
            if status == "PLANNED":
                return {**base, "status": "PLANNED", "reason": None, "variant": name, "q": qs, "labels": labels,
                        "attempts": attempts, "preflight": {"status": "NOT_RUN"}}
            continue
        modes = ["contact_free" if label == "RETRACTING" else "free" for label in labels]
        receipt = checker(np.vstack([q_start[None], qs]), ["contact_free"]+modes)
        if receipt.get("status") != "PASS":
            attempts.append({**row, "status": "REFUSED",
                             "reason": f"PREFLIGHT_FAILED:{receipt.get('binding', receipt.get('reason', 'unknown'))}",
                             "preflight": receipt})
            continue
        attempts.append({**row, "status": "PLANNED", "reason": None, "preflight": receipt})
        return {**base, "status": "PLANNED", "reason": None, "variant": name, "q": qs, "labels": labels,
                "attempts": attempts, "preflight": receipt}
    return {**base, "status": "REFUSED", "reason": attempts[-1]["reason"], "q": np.zeros((0, 7)), "labels": [],
            "attempts": attempts}


# ---------------------------------------------------------------- the home outcome

def evaluate_home_outcome(measurement, config=None):
    """PASS/FAIL + reason from return-home measurements; independent of the task outcome.

    ``measurement`` keys: plan_status, plan_reason, executed (bool), aborted_reason,
    grasp_lost, home_joint_error_max_rad, tool_translation_error_m,
    tool_rotation_error_rad, grasp_slip_max_m, grasp_rotation_max_rad,
    label_changes, unintended_tool_contacts_max, measured_path_clearance_status.
    """
    gates = (config or load_return_home_config())["outcome"]
    m = measurement
    checks = {"plan_preflight": m.get("plan_status") == "PLANNED",
              "trajectory_executed": bool(m.get("executed")) and m.get("aborted_reason") is None,
              "grasp_held": not m.get("grasp_lost", True),
              "home_joint_error": _le(m.get("home_joint_error_max_rad"), gates["max_home_joint_error_rad"]),
              "tool_translation_vs_reset": _le(m.get("tool_translation_error_m"), gates["max_tool_translation_error_m"]),
              "tool_rotation_vs_reset": _le(m.get("tool_rotation_error_rad"), gates["max_tool_rotation_error_rad"]),
              "grasp_slip": _le(m.get("grasp_slip_max_m"), gates["max_grasp_slip_m"]),
              "grasp_rotation": _le(m.get("grasp_rotation_max_rad"), gates["max_grasp_rotation_rad"]),
              "no_label_changes": _le(m.get("label_changes"), gates["max_label_changes"]),
              "no_unintended_tool_contacts": _le(m.get("unintended_tool_contacts_max"), gates["max_unintended_tool_contacts"])}
    if gates["require_measured_path_clearance"]:
        checks["measured_path_clearance"] = m.get("measured_path_clearance_status") == "PASS"
    failed = [name for name, ok in checks.items() if not ok]
    if not checks["plan_preflight"]:
        reason = f"PLAN_REFUSED:{m.get('plan_reason')}"
    elif not checks["trajectory_executed"]:
        reason = f"ABORTED:{m.get('aborted_reason')}"
    else:
        reason = None if not failed else failed[0].upper()
    return {"status": "PASS" if not failed else "FAIL", "reason": reason, "checks": checks, "failed_checks": failed,
            "gates": gates, "scope": "Physical return-to-home only; reported separately from the task outcome"}


def _le(value, bound):
    return value is not None and np.isfinite(value) and float(value) <= float(bound)+1e-12


def pose_error(a, b):
    """(translation m, rotation rad) between two XYZ+XYZW poses."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a[:3]-b[:3])), float((Rotation.from_quat(a[3:]).inv()*Rotation.from_quat(b[3:])).magnitude())


def slerp_poses(a, b, n):
    """n poses from a (exclusive) to b (inclusive), linear position, slerp rotation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rotation = Slerp([0., 1.], Rotation.from_quat([a[3:], b[3:]]))
    return np.array([np.r_[a[:3]+(b[:3]-a[:3])*t, rotation(t).as_quat()] for t in np.arange(1, n+1)/n])
