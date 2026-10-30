"""Scripted knife-motion policies that emit the same delta actions as RL.

A WaypointPolicy reads only the observation (measured proprioception) and
outputs bounded ``cartesian_edge_delta`` actions toward edge waypoints, with
optional sawing (oscillation along the edge line) while cutting. The motion
builders express common knife uses — push cut, oblique/yawed cut, slicing/saw
cut and surface skimming — from the task's nominal geometry (initial workpiece
pose and dimensions known before the episode, never oracle material state).

Controller profiles (``CONTROLLER_PROFILES``): ``v1`` is the frozen
point-to-point controller (every waypoint is approached straight from the
measured edge). ``path_v2`` is opt-in: cut waypoints follow the planned
straight segment from the previous waypoint's planned pose with cross-track
feedback (``WaypointPolicy(path_tracking=True)``), and the plan builders slow
the final approach to the cut start and settle there
(``approach_speed_m_s``/``approach_slow_distance_m``/``settle_s``).
``path_v2_vision`` (ADR 0009, SLICE-PUSH-01 only) is path_v2 plus a wrist-D405
depth survey: ``VisionRetargetPolicy`` takes a calibration frame at the end of
the cut-0 start settle and a survey frame at the end of the cut-1 start settle
(``slice_push_vision``), estimates the +x shift of the part right of cut 0
(``wrist_depth_survey``) and shifts cut 1 and every later waypoint by it.
``path_v3`` (WP-11 Phase C5b, opt-in) replaces the integrating delta law by
absolute edge targets: an internal planned reference moves along the plan
(path_v2 slow approach/settle, plus a proportional stop before holds and cut
ends) and the command is ``cartesian_edge_pose_world`` = reference + a slowly
adapting bias estimate (``WaypointPolicy(command_mode="bias_pose")``); see
``_bias_pose_call`` for the law and its stability argument.
``path_v3a`` (WP-11 review finding 8, opt-in) is path_v3 with anti-windup: the
bias magnitude is bounded (``bias_limit_m``/``bias_limit_rad``) and the bias
does not integrate on a step whose previous command was clamped by the action
limits (``anti_windup``). path_v3 itself is unchanged (it was clamped on 254 of
1057 steps of the C5b live run, so a freeze would have changed that run). An
opt-in per-waypoint stall timeout (``stall_timeout_s``, bias_pose only) ends the
waypoint or the episode with an explicit reason.
"""
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from .control import DELTA_MODE, ActionLimits, nominal_edge_pose


@dataclass(frozen=True)
class Waypoint:
    pose: tuple                 # WORLD edge XYZ + XYZW
    cut: bool = False           # arm cutting while moving to this waypoint
    saw_amplitude_m: float = 0. # oscillation along the edge line while moving
    saw_period_s: float = .4
    speed_m_s: float | None = None  # None = the per-step limit
    cut_index: int = 0
    hold_s: float = 0.          # after reaching the pose, keep commanding it (with ``cut``) this long
    label: str = ""             # phase name for recording; never changes the motion
    track_path: bool = False    # path tracking (WaypointPolicy.path_tracking) also for a non-cut waypoint


# Scripted controller profiles. ``v1`` is the frozen default; ``path_v2`` is the
# opt-in path-tracking profile (policy options + plan-builder approach options).
CONTROLLER_PROFILES = {
    "v1": dict(name="v1", path_tracking=False),
    "path_v2": dict(name="path_v2", path_tracking=True, cross_track_gain=.5, approach_speed_m_s=.02,
                    approach_slow_distance_m=.01, settle_s=.3, position_tolerance_m=.0003),
    # ADR 0009: path_v2 plus a wrist-D405 depth survey that retargets cut 1 of
    # SLICE-PUSH-01 by the measured shift of the part right of cut 0.
    "path_v2_vision": dict(name="path_v2_vision", path_tracking=True, cross_track_gain=.5, approach_speed_m_s=.02,
                           approach_slow_distance_m=.01, settle_s=.3, position_tolerance_m=.0003,
                           vision_survey=dict(adr="docs/adr/0009-demo3-wrist-depth-retarget.md",
                                              task_ids=("SLICE-PUSH-01",), camera_id="wrist_d405_0",
                                              method="end_face_plus_x", diagnostic_methods=("kerf_face",),
                                              retarget_settle_s=.3, lift_speed_m_s=.04, raise_speed_m_s=.02)),
    # WP-11 Phase C5b: absolute edge targets = planned reference + slowly adapting bias (no integrating
    # delta on the measured edge, which limit-cycled at ~6 Hz in C5). path_v2's slow approach and settle.
    "path_v3": dict(name="path_v3", path_tracking=True, approach_speed_m_s=.02, approach_slow_distance_m=.01,
                    settle_s=.3, position_tolerance_m=.0003, command_mode="bias_pose", bias_gain=.1,
                    error_filter=.5, bias_deadband_m=2e-5, bias_deadband_rad=2e-4, stop_gain=.15,
                    stop_min_step_m=5e-6, presettle_m=.001, presettle_s=.5),
    # WP-11 review finding 8: path_v3 + anti-windup. The bias is bounded (C5b live max |b| = 1.55 mm /
    # 5.9 mrad, the bounds are about twice that) and frozen while the previous command was clamped by the
    # action limits. path_v3 stays unchanged (the C5b live trajectory; tests/test_wp11_path_v3a.py).
    "path_v3a": dict(name="path_v3a", path_tracking=True, approach_speed_m_s=.02, approach_slow_distance_m=.01,
                     settle_s=.3, position_tolerance_m=.0003, command_mode="bias_pose", bias_gain=.1,
                     error_filter=.5, bias_deadband_m=2e-5, bias_deadband_rad=2e-4, stop_gain=.15,
                     stop_min_step_m=5e-6, presettle_m=.001, presettle_s=.5, bias_limit_m=.003,
                     bias_limit_rad=.015, anti_windup=True),
}
# Policy options a profile may carry (absent keys keep the WaypointPolicy defaults, so v1/path_v2 are unchanged).
_POLICY_KEYS = ("path_tracking", "cross_track_gain", "position_tolerance_m", "command_mode", "bias_gain",
                "error_filter", "bias_deadband_m", "bias_deadband_rad", "stop_gain", "stop_min_step_m",
                "bias_limit_m", "bias_limit_rad", "anti_windup")
STALL_ACTIONS = ("end_episode", "skip_waypoint")
COMMAND_MODES = ("delta", "bias_pose")
ABSOLUTE_MODE = "cartesian_edge_pose_world"


def controller_profile(name):
    """(policy options, plan-builder options) of a named controller profile."""
    if name not in CONTROLLER_PROFILES:
        raise ValueError(f"Unknown controller profile {name!r}; known: {sorted(CONTROLLER_PROFILES)}")
    profile = CONTROLLER_PROFILES[name]
    policy = {key: profile[key] for key in _POLICY_KEYS if key in profile}
    plan = {key: profile[key] for key in ("approach_speed_m_s", "approach_slow_distance_m", "settle_s", "presettle_m",
                                          "presettle_s") if key in profile}
    return policy, plan


def vision_survey_options(name):
    """Depth-survey options of a named controller profile (None for profiles without a survey)."""
    controller_profile(name)
    survey = CONTROLLER_PROFILES[name].get("vision_survey")
    return None if survey is None else dict(survey)


def edge_from_observation(observation, task):
    """Commanded-frame edge pose from deployable proprioception."""
    return nominal_edge_pose(observation["proprio"], task.hand_from_edge_xyz_xyzw)


@dataclass
class WaypointPolicy:
    task: object
    waypoints: list
    limits: ActionLimits = field(default_factory=ActionLimits)
    position_tolerance_m: float = .0005
    rotation_tolerance_rad: float = float(np.deg2rad(.5))
    index: int = 0
    _start_time: float | None = None
    # Optional edge-pose source (observation -> WORLD edge XYZ+XYZW), e.g. a
    # measured tool-edge estimate for a live controller. None keeps the
    # nominal commanded-frame edge from proprioception (edge_from_observation).
    edge_estimator: Callable | None = None
    _hold_start: float | None = None
    # Opt-in path tracking (controller profile path_v2). Off: the frozen v1
    # point-to-point behaviour, bit for bit. On: a waypoint with ``cut`` (or
    # ``track_path``) follows the planned straight segment from the PREVIOUS
    # waypoint's planned pose to its own pose, with cross-track feedback
    # ``cross_track_gain`` (per step, on the measured edge).
    path_tracking: bool = False
    cross_track_gain: float = .5
    last_tracking: dict | None = None   # diagnostics of the latest tracked step (None otherwise)
    # Opt-in absolute-target law (controller profile path_v3, ``_bias_pose_call``). "delta" (default)
    # keeps every behaviour above bit for bit; the fields below are read only in "bias_pose".
    command_mode: str = "delta"
    bias_gain: float = .1             # per-step gain of the bias estimate
    error_filter: float = .5          # first-order low-pass coefficient on the bias error
    bias_deadband_m: float = 2e-5     # soft deadband on the filtered position error
    bias_deadband_rad: float = 2e-4   # soft deadband on the filtered rotation error
    stop_gain: float = .15            # reference step <= stop_gain * remaining before holds, cut ends and turns
    stop_min_step_m: float = 5e-6     # ... but never below this (the stop ends in finite time)
    _bias_state: dict | None = None
    # Anti-windup (controller profile path_v3a; None/False keep path_v3 bit for bit). bias_pose only.
    bias_limit_m: float | None = None     # |bias position| <= this (radial clip)
    bias_limit_rad: float | None = None   # |bias rotation vector| <= this
    anti_windup: bool = False             # no bias integration on a step whose previous command was clamped
    # Opt-in stall timeout (bias_pose only; None = off): a waypoint whose measured edge is outside the
    # tolerance and has not come closer by stall_min_progress_m / _rad for stall_timeout_s is a stall.
    # stall_action "end_episode": the policy is done and ``termination_request`` names the reason;
    # "skip_waypoint": the next waypoint starts. Every stall is listed in ``stall_events``.
    stall_timeout_s: float | None = None
    stall_min_progress_m: float = 2e-4
    stall_min_progress_rad: float = 2e-3
    stall_action: str = "end_episode"
    stall_timeouts_by_label: dict | None = None   # optional per-waypoint-label timeout overrides (s; None = off)
    termination_request: str | None = None
    stall_events: list = field(default_factory=list)
    _stall_track: dict | None = None

    def __post_init__(self):
        if not 0 < self.cross_track_gain <= 1:
            raise ValueError("cross_track_gain must be in (0, 1]")
        if self.command_mode not in COMMAND_MODES:
            raise ValueError(f"command_mode must be one of {COMMAND_MODES}")
        if self.command_mode == "bias_pose" and not (
                0 < self.bias_gain <= .25 and 0 < self.error_filter <= 1 and self.bias_deadband_m >= 0
                and self.bias_deadband_rad >= 0 and 0 < self.stop_gain <= 1 and self.stop_min_step_m > 0):
            raise ValueError("bias_pose needs bias_gain in (0, 0.25], error_filter and stop_gain in (0, 1] "
                             "and non-negative deadbands")
        stall = self.stall_timeout_s is not None or bool(self.stall_timeouts_by_label)
        if (self.bias_limit_m is not None or self.bias_limit_rad is not None or self.anti_windup or stall) \
                and self.command_mode != "bias_pose":
            raise ValueError("bias_limit_m/bias_limit_rad/anti_windup/stall_timeout_s apply to command_mode='bias_pose'")
        for name in ("bias_limit_m", "bias_limit_rad", "stall_timeout_s"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive or None")
        for label, value in (self.stall_timeouts_by_label or {}).items():
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(f"stall timeout of {label!r} must be positive or None")
        if self.stall_action not in STALL_ACTIONS or self.stall_min_progress_m <= 0 or self.stall_min_progress_rad <= 0:
            raise ValueError(f"stall_action must be one of {STALL_ACTIONS} and the stall progress thresholds positive")

    @property
    def done(self):
        return self.index >= len(self.waypoints)

    @property
    def phase(self):
        """Label of the current waypoint ("done" after the last one)."""
        return "done" if self.done else self.waypoints[self.index].label

    def _target(self, waypoint, time_s):
        pose = np.asarray(waypoint.pose, float)
        if waypoint.saw_amplitude_m > 0:
            along = Rotation.from_quat(pose[3:]).apply([0., 1., 0.])
            phase = 2*np.pi*(time_s-self._start_time)/waypoint.saw_period_s
            pose = np.r_[pose[:3]+waypoint.saw_amplitude_m*np.sin(phase)*along, pose[3:]]
        return pose

    def _segment(self, waypoint):
        """(P0, unit direction, length) of a tracked waypoint's reference segment, else None."""
        if not self.path_tracking or not (waypoint.cut or waypoint.track_path) or self.index == 0:
            return None
        start = np.asarray(self.waypoints[self.index-1].pose[:3], float)
        delta = np.asarray(waypoint.pose[:3], float)-start
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            return None  # degenerate segment: plain point target
        return start, delta/length, length

    def _tracking_action(self, waypoint, segment, edge, time_s):
        """Path-tracking translation: cross-track feedback first, then along-track feed."""
        start, direction, length = segment
        final = np.asarray(waypoint.pose, float)
        progress = float((edge[:3]-start)@direction)
        closest = start+direction*min(max(progress, 0.), length)
        cross = self.cross_track_gain*(closest-edge[:3])
        step = self.limits.max_translation_m
        if waypoint.speed_m_s is not None:
            step = min(step, waypoint.speed_m_s*self.task.control_dt_s)
        forward = direction*min(step, max(length-progress, 0.))
        if waypoint.saw_amplitude_m > 0:
            # Sawing as in v1: the full along-edge offset to the oscillating target, not gain-scaled.
            along = Rotation.from_quat(final[3:]).apply([0., 1., 0.])
            phase = 2*np.pi*(time_s-self._start_time)/waypoint.saw_period_s
            lateral = (closest+waypoint.saw_amplitude_m*np.sin(phase)*along-edge[:3])@along
            cross += (np.clip(lateral, -self.limits.max_translation_m, self.limits.max_translation_m)-cross@along)*along
            forward -= (forward@along)*along
        translation = _cap_with_priority(cross, forward, self.limits.max_translation_m)
        self.last_tracking = dict(segment_length_m=length, along_track_m=progress,
                                  cross_track_error_m=float(np.linalg.norm(edge[:3]-(start+direction*progress))),
                                  cross_track_command_m=float(np.linalg.norm(cross)))
        rotation = (Rotation.from_quat(final[3:])*Rotation.from_quat(edge[3:]).inv()).as_rotvec()
        return {"mode": DELTA_MODE, "frame": "world", "target": np.r_[translation, rotation],
                "cut_enabled": bool(waypoint.cut), **({"cut_index": waypoint.cut_index} if waypoint.cut_index else {})}

    def __call__(self, observation):
        if self.command_mode == "bias_pose":
            return self._bias_pose_call(observation)
        if self.edge_estimator is None:
            edge = edge_from_observation(observation, self.task)
        else:
            edge = np.asarray(self.edge_estimator(observation), float)
        time_s = float(observation["timestamp_s"])
        self.last_tracking = None
        while not self.done:
            waypoint = self.waypoints[self.index]
            if self._start_time is None:
                self._start_time = time_s
            final = np.asarray(waypoint.pose, float)
            position_error = np.linalg.norm(final[:3]-edge[:3])
            rotation_error = (Rotation.from_quat(final[3:])*Rotation.from_quat(edge[3:]).inv()).magnitude()
            reached = position_error <= self.position_tolerance_m and rotation_error <= self.rotation_tolerance_rad
            segment = self._segment(waypoint)
            if segment is not None:
                # Tracked: along-track progress reached the segment end, and the end pose is within tolerance.
                start, direction, length = segment
                reached = reached and float((edge[:3]-start)@direction) >= length-self.position_tolerance_m
            if reached and waypoint.hold_s > 0:
                if self._hold_start is None:
                    self._hold_start = time_s
                reached = time_s-self._hold_start >= waypoint.hold_s-1e-9
            if reached:
                self.index, self._start_time, self._hold_start = self.index+1, None, None
                continue
            if segment is not None:
                return self._tracking_action(waypoint, segment, edge, time_s)
            target = self._target(waypoint, time_s)
            step = self.limits.max_translation_m
            if waypoint.speed_m_s is not None:
                step = min(step, waypoint.speed_m_s*self.task.control_dt_s)
            translation = target[:3]-edge[:3]
            norm = np.linalg.norm(translation)
            if norm > step:
                translation *= step/norm
            if waypoint.saw_amplitude_m > 0:
                # Sawing: the along-edge oscillation is not throttled by the descent speed.
                along = Rotation.from_quat(final[3:]).apply([0., 1., 0.])
                lateral = (target[:3]-edge[:3])@along
                translation += np.clip(lateral, -self.limits.max_translation_m, self.limits.max_translation_m)*along \
                    - (translation@along)*along
            rotation = (Rotation.from_quat(target[3:])*Rotation.from_quat(edge[3:]).inv()).as_rotvec()
            return {"mode": DELTA_MODE, "frame": "world", "target": np.r_[translation, rotation],
                    "cut_enabled": bool(waypoint.cut), **({"cut_index": waypoint.cut_index} if waypoint.cut_index else {})}
        return {"mode": DELTA_MODE, "frame": "world", "target": np.zeros(6), "cut_enabled": False}

    # ---------------------------------------------------------------- path_v3 (WP-11 Phase C5b)

    def _decelerates(self, index):
        """Stop the reference geometrically at waypoint ``index``: a hold follows, a cut ends, or the path turns > 30 deg."""
        waypoint = self.waypoints[index]
        if waypoint.hold_s > 0 or waypoint.cut:
            return True
        if index == 0 or index+1 >= len(self.waypoints):
            return index+1 >= len(self.waypoints)
        here = np.asarray(waypoint.pose[:3], float)
        incoming = here-np.asarray(self.waypoints[index-1].pose[:3], float)
        outgoing = np.asarray(self.waypoints[index+1].pose[:3], float)-here
        if np.linalg.norm(incoming) < 1e-9 or np.linalg.norm(outgoing) < 1e-9:
            return False
        cosine = incoming@outgoing/(np.linalg.norm(incoming)*np.linalg.norm(outgoing))
        return bool(cosine < np.cos(np.deg2rad(30.)))

    def _stall_timeout(self, waypoint):
        by_label = self.stall_timeouts_by_label or {}
        return by_label[waypoint.label] if waypoint.label in by_label else self.stall_timeout_s

    def _stall_check(self, waypoint, final, measured, time_s):
        """Stall event of the current (not reached) waypoint, else None (opt-in ``stall_timeout_s``).

        The timer restarts when the waypoint starts, when the measured edge is
        within the waypoint tolerance, and whenever its position (rotation)
        distance to the waypoint drops by ``stall_min_progress_m`` (``_rad``)
        below the best so far. A stall is ``stall_timeout_s`` without either.
        """
        timeout = self._stall_timeout(waypoint)
        if timeout is None:
            return None
        distance = float(np.linalg.norm(final[:3]-measured[:3]))
        angle = float((Rotation.from_quat(final[3:])*Rotation.from_quat(measured[3:]).inv()).magnitude())
        track = self._stall_track
        if track is None or track["index"] != self.index:
            track = self._stall_track = dict(index=self.index, distance=distance, angle=angle, since=time_s)
            return None
        if distance <= self.position_tolerance_m and angle <= self.rotation_tolerance_rad:
            track.update(distance=distance, angle=angle, since=time_s)
            return None
        if distance < track["distance"]-self.stall_min_progress_m:
            track.update(distance=distance, since=time_s)
        if angle < track["angle"]-self.stall_min_progress_rad:
            track.update(angle=angle, since=time_s)
        if time_s-track["since"] < timeout-1e-9:
            return None
        self._stall_track = None
        bias = (self._bias_state or {}).get("bias_position")
        return dict(reason="WAYPOINT_STALL", waypoint_index=int(self.index), label=waypoint.label, time_s=float(time_s),
                    timeout_s=float(timeout), action=self.stall_action, position_error_m=distance, rotation_error_rad=angle,
                    best_position_error_m=float(track["distance"]), bias_position_m=None if bias is None else
                    np.asarray(bias, float).tolist())

    def _bias_pose_call(self, observation):
        """path_v3: absolute edge target = planned reference + slowly adapting bias (WP-11 Phase C5b).

        Per control step k (WORLD frame; the observation carries the edge m_{k-1}
        measured after command c_{k-1}):

        * reference r_k: moves from r_{k-1} straight toward the current waypoint
          at min(step limit, waypoint speed * dt); before a hold, at a cut end or
          at a > 30 deg turn the step is also <= ``stop_gain`` * remaining
          distance, but >= ``stop_min_step_m`` (a geometric stop instead of a
          velocity step: on the fitted plant an abrupt stop undershoots by ~1.2
          steps, a stop with gain 0.15 not measurably); rotation at the rotation
          step limit. It never reads the measured edge.
        * bias error e_k = r_{k-1} - m_{k-1} with its component along the
          reference velocity (r_{k-1} - r_{k-2}) removed, so tracking lag along
          the path is not learned as a bias; low-pass
          f_k = f_{k-1} + ``error_filter`` * (e_k - f_{k-1}); soft deadband
          (``bias_deadband_m``/``_rad``) so noise is not chased.
        * bias b_k = b_{k-1} + ``bias_gain`` * deadband(f_k) (rotation: rotation
          vector, applied on the left); b_0 = nominal hand-FK edge - measured edge,
          so c_0 equals the env's commanded pose.
        * command c_k = r_k + b_k, bounded to the action step limits from c_{k-1}.

        A waypoint completes when r reached it and the measured edge is within
        ``position_tolerance_m``/``rotation_tolerance_rad`` of it (then its hold
        runs). Stability: the measured edge enters only through the bias, so for
        a plant m = G(c) + offset (G stable, unit DC gain) the feedback loop is
        (1 - z^-1) + bias_gain * F(z) z^-1 G(z) = 0 with F the error filter;
        the reference is open loop. For a one-step-delay plant G = z^-1, F = 1
        this is z^2 - z + k = 0, real roots for k <= 1/4 (k = 0.1: 0.89, 0.11).
        The path_v2 delta law integrates the full measured error with unit gain
        (k = 1: z^2 - z + 1 = 0, |z| = 1 at pi/3 rad/step), the C5 ringing; on
        the plant fitted to the C5 trace path_v2 has |z| = 0.92 (z axis) and
        1.07 (y) at ~8.4 steps per period (~6 Hz), path_v3 a real dominant pole
        ~0.90 (tests/test_wp11_phase_c5b.py). Sawing is not implemented here.
        """
        from .control import bound_target
        nominal = edge_from_observation(observation, self.task)
        measured = nominal if self.edge_estimator is None else np.asarray(self.edge_estimator(observation), float)
        time_s = float(observation["timestamp_s"])
        state = self._bias_state
        if state is None:
            state = self._bias_state = dict(
                reference=measured.copy(), velocity=np.zeros(3), angular=np.zeros(3),
                bias_position=nominal[:3]-measured[:3],
                bias_rotation=(Rotation.from_quat(nominal[3:])*Rotation.from_quat(measured[3:]).inv()).as_rotvec(),
                error_position=np.zeros(3), error_rotation=np.zeros(3), command=nominal.copy())
            error = rotation_error_vector = np.zeros(3)
            frozen = False
        else:
            reference = state["reference"]
            error = _without_along(reference[:3]-measured[:3], state["velocity"])
            rotation_error_vector = _without_along(
                (Rotation.from_quat(reference[3:])*Rotation.from_quat(measured[3:]).inv()).as_rotvec(), state["angular"])
            a = self.error_filter
            state["error_position"] = state["error_position"]+a*(error-state["error_position"])
            state["error_rotation"] = state["error_rotation"]+a*(rotation_error_vector-state["error_rotation"])
            # path_v3a anti-windup: the error of a step whose command was clamped is not learned as a bias.
            frozen = bool(self.anti_windup and state.get("clamped", False))
            if not frozen:
                state["bias_position"] = state["bias_position"]+self.bias_gain*_soft_deadband(
                    state["error_position"], self.bias_deadband_m)
                state["bias_rotation"] = (Rotation.from_rotvec(self.bias_gain*_soft_deadband(
                    state["error_rotation"], self.bias_deadband_rad))*Rotation.from_rotvec(state["bias_rotation"])).as_rotvec()
        saturated = False
        if self.bias_limit_m is not None:
            state["bias_position"], clipped = _clip_norm(state["bias_position"], self.bias_limit_m)
            saturated |= clipped
        if self.bias_limit_rad is not None:
            state["bias_rotation"], clipped = _clip_norm(state["bias_rotation"], self.bias_limit_rad)
            saturated |= clipped
        reference = state["reference"]
        self.last_tracking = None
        stalls = []
        while not self.done:
            waypoint = self.waypoints[self.index]
            if waypoint.saw_amplitude_m > 0:
                raise ValueError("Controller profile path_v3 (bias_pose) does not implement sawing")
            if self._start_time is None:
                self._start_time = time_s
            final = np.asarray(waypoint.pose, float)
            at_reference = (np.linalg.norm(final[:3]-reference[:3]) <= 1e-9
                            and (Rotation.from_quat(final[3:])*Rotation.from_quat(reference[3:]).inv()).magnitude() <= 1e-9)
            reached = (at_reference and np.linalg.norm(final[:3]-measured[:3]) <= self.position_tolerance_m
                       and (Rotation.from_quat(final[3:])*Rotation.from_quat(measured[3:]).inv()).magnitude()
                       <= self.rotation_tolerance_rad)
            if reached and waypoint.hold_s > 0:
                if self._hold_start is None:
                    self._hold_start = time_s
                reached = time_s-self._hold_start >= waypoint.hold_s-1e-9
            if not reached:
                stall = self._stall_check(waypoint, final, measured, time_s)
                if stall is None:
                    break
                stalls.append(stall)
                self.stall_events.append(stall)
                if self.stall_action == "end_episode":
                    self.termination_request = f"WAYPOINT_STALL:{waypoint.label or self.index}"
                    self.index = len(self.waypoints)
                    break
            self.index, self._start_time, self._hold_start = self.index+1, None, None
        if self.done:
            if stalls:
                self.last_tracking = dict(command_mode="bias_pose", stalls=stalls, termination_request=self.termination_request)
            return {"mode": DELTA_MODE, "frame": "world", "target": np.zeros(6), "cut_enabled": False}
        step = self.limits.max_translation_m
        if waypoint.speed_m_s is not None:
            step = min(step, waypoint.speed_m_s*self.task.control_dt_s)
        offset = final[:3]-reference[:3]
        remaining = float(np.linalg.norm(offset))
        if self._decelerates(self.index):
            step = min(step, max(self.stop_gain*remaining, self.stop_min_step_m))
        position = final[:3].copy() if remaining <= max(step, 1e-6) else reference[:3]+offset*(step/remaining)
        r_reference, r_final = Rotation.from_quat(reference[3:]), Rotation.from_quat(final[3:])
        relative = (r_final*r_reference.inv()).as_rotvec()
        angle = float(np.linalg.norm(relative))
        limit = self.limits.max_rotation_rad
        rotation = r_final if angle <= max(limit, 1e-9) else Rotation.from_rotvec(relative*(limit/angle))*r_reference
        state["velocity"] = position-reference[:3]
        state["angular"] = (rotation*r_reference.inv()).as_rotvec()
        state["reference"] = np.r_[position, rotation.as_quat()]
        desired = np.r_[position+state["bias_position"], (Rotation.from_rotvec(state["bias_rotation"])*rotation).as_quat()]
        command, report = bound_target(state["command"], desired, self.limits, getattr(self.task, "T_world_workpiece", None))
        state["command"] = command
        state["clamped"] = bool(report["translation_clamped"] or report["rotation_clamped"] or report["workspace_clamped"])
        self.last_tracking = dict(command_mode="bias_pose", reference_world_m=position.tolist(),
                                  measured_minus_previous_reference_m=(measured[:3]-reference[:3]).tolist(),
                                  cross_track_error_m=float(np.linalg.norm(error)),
                                  bias_position_m=state["bias_position"].tolist(),
                                  bias_rotation_rad=float(np.linalg.norm(state["bias_rotation"])),
                                  filtered_error_m=state["error_position"].tolist())
        if self.anti_windup or self.bias_limit_m is not None or self.bias_limit_rad is not None:
            self.last_tracking.update(bias_frozen=frozen, bias_saturated=saturated, command_clamped=state["clamped"])
        if stalls:   # skip_waypoint stalls of this call
            self.last_tracking["stalls"] = stalls
        return {"mode": ABSOLUTE_MODE, "target": command, "cut_enabled": bool(waypoint.cut),
                **({"cut_index": waypoint.cut_index} if waypoint.cut_index else {})}


def _clip_norm(vector, limit):
    """(vector radially clipped to norm <= limit, whether it was clipped)."""
    norm = float(np.linalg.norm(vector))
    if norm <= limit:
        return vector, False
    return np.asarray(vector, float)*(limit/norm), True


def _without_along(vector, direction):
    """``vector`` minus its component along ``direction`` (unchanged when the direction is ~zero)."""
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.asarray(vector, float)
    unit = np.asarray(direction, float)/norm
    return vector-(vector@unit)*unit


def _soft_deadband(vector, width):
    """Shrink a vector by ``width`` (zero inside the deadband): no discontinuity at the band edge."""
    norm = float(np.linalg.norm(vector))
    if norm <= width:
        return np.zeros_like(vector)
    return vector*(1.-width/norm)


def _cap_with_priority(priority, rest, limit):
    """``priority + k*rest`` with the largest k in [0, 1] whose norm stays <= limit (priority alone capped first)."""
    norm = float(np.linalg.norm(priority))
    if norm >= limit:
        return priority*(limit/norm)
    a = float(rest@rest)
    if a == 0.:
        return priority
    b, c = 2.*float(priority@rest), norm*norm-limit*limit
    k = (-b+np.sqrt(b*b-4.*a*c))/(2.*a)
    return priority+min(1., k)*rest


def _pose(position, rotation):
    return tuple(np.r_[position, rotation.as_quat()])


def shape_approaches(plan, *, approach_speed_m_s=None, approach_slow_distance_m=.01, settle_s=0., presettle_m=None,
                     presettle_s=0.):
    """Slow the final approach to every cut start and settle there before cutting (path_v2).

    For each cut waypoint whose predecessor (the cut start) is not cutting: the
    last ``approach_slow_distance_m`` of the straight planned move to the start
    is driven at ``approach_speed_m_s`` (an intermediate waypoint is inserted
    where the slow part begins; it copies the start's flags, speed and label),
    and the start is held for ``settle_s``. ``approach_speed_m_s=None`` and
    ``settle_s=0`` return the plan unchanged. A start that is the first
    waypoint has no planned predecessor segment and only gets the settle hold.
    ``presettle_m`` (opt-in, path_v3; None = off): a further waypoint
    ``presettle_m`` before the start on the same approach line is held
    ``presettle_s`` (the bias estimate converges there, at rest), then the last
    ``presettle_m`` are driven at the approach speed; it must be shorter than
    the slow distance.
    """
    if approach_speed_m_s is None and not settle_s:
        return list(plan)
    if approach_speed_m_s is not None and not approach_speed_m_s > 0:
        raise ValueError("approach_speed_m_s must be positive (or None)")
    if not approach_slow_distance_m > 0 or settle_s < 0:
        raise ValueError("approach_slow_distance_m must be positive and settle_s non-negative")
    if presettle_m is not None and not (0 < presettle_m < approach_slow_distance_m and presettle_s >= 0
                                        and approach_speed_m_s is not None):
        raise ValueError("presettle_m needs an approach speed, 0 < presettle_m < approach_slow_distance_m "
                         "and presettle_s >= 0")
    starts = {i-1 for i, w in enumerate(plan) if w.cut and i > 0 and not plan[i-1].cut}
    shaped = []
    for i, waypoint in enumerate(plan):
        if i not in starts:
            shaped.append(waypoint)
            continue
        changes = {"hold_s": max(waypoint.hold_s, settle_s)}
        if approach_speed_m_s is not None and i > 0:
            here = np.asarray(waypoint.pose[:3], float)
            back = np.asarray(plan[i-1].pose[:3], float)-here
            distance = float(np.linalg.norm(back))
            if distance > approach_slow_distance_m+1e-9:
                slow_from = here+back*(approach_slow_distance_m/distance)
                shaped.append(replace(waypoint, pose=tuple(np.r_[slow_from, waypoint.pose[3:]]), hold_s=0.))
            speed = approach_speed_m_s if waypoint.speed_m_s is None else min(waypoint.speed_m_s, approach_speed_m_s)
            if presettle_m is not None and distance > presettle_m+1e-9:
                shaped.append(replace(waypoint, pose=tuple(np.r_[here+back*(presettle_m/distance), waypoint.pose[3:]]),
                                      hold_s=presettle_s, speed_m_s=speed))
            changes["speed_m_s"] = speed
        shaped.append(replace(waypoint, **changes))
    return shaped


def oriented(task, *, tilt_deg=0., yaw_deg=0., roll_deg=0.):
    """Task edge orientation turned in the WORLD frame.

    tilt: about the task edge line (oblique cut plane); yaw: about world z;
    roll: about the task blade normal (one end of the edge leads).
    """
    base = Rotation.from_quat(task.reference_edge_pose_world(0.)[3:])
    edge_line, normal = base.apply([0., 1., 0.]), base.apply([0., 0., 1.])
    extra = (Rotation.from_rotvec(np.deg2rad(yaw_deg)*np.array([0., 0., 1.]))
             * Rotation.from_rotvec(np.deg2rad(tilt_deg)*edge_line)
             * Rotation.from_rotvec(np.deg2rad(roll_deg)*normal))
    return extra*base


def through_cut(task, *, tilt_deg=0., yaw_deg=0., roll_deg=0., saw_amplitude_m=0., saw_period_s=.4,
                cut_speed_m_s=.01, clearance_m=.008, overtravel_m=.004, cut_index=0, offset_m=0.,
                approach_speed_m_s=None, approach_slow_distance_m=.01, settle_s=0.):
    """Approach above the target plane, cut along the advance line, retract.

    The cut line passes through the workpiece center at mid height; it ends
    below the workpiece bottom by ``overtravel_m`` (vertical component).
    ``offset_m`` shifts the cut line along the workpiece-local target normal
    (0 keeps the line through the task target exactly). ``approach_*``/``settle_s``
    shape the approach to the cut start (``shape_approaches``; defaults: off).
    """
    rotation = oriented(task, tilt_deg=tilt_deg, yaw_deg=yaw_deg, roll_deg=roll_deg)
    advance = rotation.apply([1., 0., 0.])
    if advance[2] > -.2:
        raise ValueError("A through cut needs an advance direction pointing down")
    T = task.T_world_workpiece
    height = task.geometry.base_height_m
    center = T[:3, :3]@np.array(task.target_origin_local_m)+T[:3, 3]+T[:3, 2]*height/2
    if offset_m:
        center = center+T[:3, :3]@(offset_m*np.asarray(task.target_normal_local, float))
    top, bottom = center[2]+height/2, center[2]-height/2
    start = center+advance*((top+clearance_m-center[2])/advance[2])
    end = center+advance*((bottom-overtravel_m-center[2])/advance[2])
    above = start+np.array([0., 0., .03])
    plan = [Waypoint(_pose(above, rotation)), Waypoint(_pose(start, rotation)),
            Waypoint(_pose(end, rotation), cut=True, saw_amplitude_m=saw_amplitude_m, saw_period_s=saw_period_s,
                     speed_m_s=cut_speed_m_s, cut_index=cut_index),
            Waypoint(_pose(end, rotation), cut=False),
            Waypoint(_pose(start+np.array([0., 0., .03]), rotation))]
    return shape_approaches(plan, approach_speed_m_s=approach_speed_m_s,
                            approach_slow_distance_m=approach_slow_distance_m, settle_s=settle_s)


def skim(task, *, clearance_m=None, margin_m=.03, speed_m_s=.025, saw_amplitude_m=0.,
         approach_speed_m_s=None, approach_slow_distance_m=.01, settle_s=0.):
    """Move the edge across the material above the target plane (TRIM).

    The pass spans the bumps (or the whole workpiece without bumps) plus
    ``margin_m`` on each side, at the target plane plus the edge clearance.
    ``approach_*``/``settle_s`` shape the approach to the pass start
    (``shape_approaches``; defaults: off).
    """
    rotation = Rotation.from_quat(task.reference_edge_pose_world(0.)[3:])
    advance = rotation.apply([1., 0., 0.])
    direction = advance-advance[2]*np.array([0., 0., 1.])
    direction /= np.linalg.norm(direction)
    plane = task.target_plane_world()
    clearance = task.edge_clearance_m if clearance_m is None else clearance_m
    T = task.T_world_workpiece
    center = T[:3, 3]
    if task.geometry.bumps:
        along = [(T[:3, :3]@np.array([b.x_m, b.y_m, 0.]))@direction for b in task.geometry.bumps]
        reach = [b.radius_x_m+b.radius_y_m for b in task.geometry.bumps]  # conservative footprint
        low = min(a-r/2 for a, r in zip(along, reach))-margin_m
        high = max(a+r/2 for a, r in zip(along, reach))+margin_m
    else:
        low, high = -task.geometry.length_m/2-margin_m, task.geometry.length_m/2+margin_m
    start = center+direction*low
    end = center+direction*high
    for point in (start, end):  # put the edge at the target plane plus clearance
        point[2] += clearance-(point@plane[:3]+plane[3])/plane[2]
    above = start+np.array([0., 0., .03])
    plan = [Waypoint(_pose(above, rotation)), Waypoint(_pose(start, rotation)),
            Waypoint(_pose(end, rotation), cut=True, speed_m_s=speed_m_s, saw_amplitude_m=saw_amplitude_m),
            Waypoint(_pose(end+np.array([0., 0., .03]), rotation))]
    return shape_approaches(plan, approach_speed_m_s=approach_speed_m_s,
                            approach_slow_distance_m=approach_slow_distance_m, settle_s=settle_s)


def slice_push(task, *, cut_speed_m_s=.01, clearance_m=.008, overtravel_m=.004, hold_s=.5,
               kerf_speed_m_s=.02, push_distance_m=None, push_speed_m_s=None, push_height_m=None,
               release_m=.002, retract_height_m=.03, approach_speed_m_s=None, approach_slow_distance_m=.01,
               settle_s=0.):
    """Two sequential cuts, then a cut-disabled blade-face push (ADR 0007 section 3).

    For a v3 slice-push task (tasks.DemoTaskSpec): cut 0 on the declared
    plane 0 (``cut_index`` 0, armed descent to ``overtravel_m`` below the bar
    bottom), hold, lift inside the kerf and move to plane 1 with cutting
    disabled, cut 1 (``cut_index`` 1), hold, raise the edge to
    ``push_height_m`` above the board inside the kerf, push along the declared
    push direction with cutting disabled, hold, back off and retract. Only
    nominal task geometry is used, never oracle material state.
    ``approach_*``/``settle_s`` shape the approaches to both cut starts
    (``shape_approaches``; defaults: off).
    """
    planes = getattr(task, "multi_cut_planes_local", ())
    if len(planes) != 2:
        raise ValueError("slice_push needs a task that declares two cut planes")
    normal = np.asarray(task.target_normal_local, float)
    offsets = []
    for origin, plane_normal in planes:
        if not np.allclose(plane_normal, normal, atol=1e-9):
            raise ValueError("slice_push cut planes must share the task target normal")
        offsets.append(float((np.asarray(origin, float)-np.asarray(task.target_origin_local_m, float))@normal))
    distance = task.push_distance_m if push_distance_m is None else push_distance_m
    speed = task.push_speed_m_s if push_speed_m_s is None else push_speed_m_s
    height = task.push_edge_height_m if push_height_m is None else push_height_m
    common = dict(cut_speed_m_s=cut_speed_m_s, clearance_m=clearance_m, overtravel_m=overtravel_m)
    first = through_cut(task, cut_index=0, offset_m=offsets[0], **common)
    second = through_cut(task, cut_index=1, offset_m=offsets[1], **common)
    rotation = first[2].pose[3:]
    T = task.T_world_workpiece
    board_top = max(box.center_m[2]+box.half_extents_m[2] for box in task.fixture_boxes_local)
    pushed = planes[task.pushed_piece_cut_index][0]
    kerf = T[:3, :3]@np.array([pushed[0], pushed[1], board_top+height])+T[:3, 3]
    direction = T[:3, :3]@np.asarray(task.push_direction_local, float)
    pushed_to = kerf+distance*direction
    released = pushed_to-release_m*direction
    up = T[:3, 2]
    plan = [replace(first[0], label="approach_0"), replace(first[1], label="start_0"),
            replace(first[2], label="cut_0"),
            replace(first[3], hold_s=hold_s, label="hold_0"),
            Waypoint(first[1].pose, speed_m_s=kerf_speed_m_s, label="lift_0"),
            replace(second[1], label="move_1"),
            replace(second[2], label="cut_1"),
            replace(second[3], hold_s=hold_s, label="hold_1"),
            Waypoint(tuple(np.r_[kerf, rotation]), speed_m_s=speed, label="raise_in_kerf"),
            Waypoint(tuple(np.r_[pushed_to, rotation]), speed_m_s=speed, label="push"),
            Waypoint(tuple(np.r_[pushed_to, rotation]), hold_s=hold_s, label="push_hold"),
            Waypoint(tuple(np.r_[released, rotation]), speed_m_s=speed, label="release"),
            Waypoint(tuple(np.r_[released+up*(np.asarray(first[1].pose[:3])@up-released@up+retract_height_m), rotation]),
                     label="retract")]
    return shape_approaches(plan, approach_speed_m_s=approach_speed_m_s,
                            approach_slow_distance_m=approach_slow_distance_m, settle_s=settle_s)


# ------------------------------------------------------------------ path_v2_vision (ADR 0009)

def slice_push_vision(task, *, retarget_settle_s=.3, lift_speed_m_s=.04, raise_speed_m_s=.02,
                      approach_speed_m_s=.02, approach_slow_distance_m=.01, settle_s=.3, **slice_push_options):
    """The path_v2 slice_push plan with ADR 0009 survey points; returns (plan, indices).

    Relative to ``slice_push(task, approach..., settle...)``: the cut-0 start
    settle hold is labelled ``calibrate_0`` (calibration frame at its end), the
    cut-1 start settle hold ``survey_1`` (survey frame at its end), a
    ``retarget_1`` waypoint (the cut-1 start, shifted at runtime by the
    estimate, approached at ``approach_speed_m_s`` and held ``retarget_settle_s``)
    is inserted before ``cut_1``, the disarmed lift in the kerf runs at
    ``lift_speed_m_s`` and the disarmed raise in the kerf at ``raise_speed_m_s``.
    ``indices``: calibration / survey waypoint and the first waypoint the shift applies to.
    """
    plan = slice_push(task, kerf_speed_m_s=lift_speed_m_s, approach_speed_m_s=approach_speed_m_s,
                      approach_slow_distance_m=approach_slow_distance_m, settle_s=settle_s, **slice_push_options)
    labels = [w.label for w in plan]
    if not settle_s > 0 or labels.count("start_0") != 2 or labels.count("move_1") != 2:
        raise ValueError("slice_push_vision needs the path_v2 settle holds at both cut starts")
    calibration = len(labels)-1-labels[::-1].index("start_0")
    survey = len(labels)-1-labels[::-1].index("move_1")
    plan[calibration] = replace(plan[calibration], label="calibrate_0")
    plan[survey] = replace(plan[survey], label="survey_1")
    raise_index = labels.index("raise_in_kerf")
    plan[raise_index] = replace(plan[raise_index], speed_m_s=raise_speed_m_s)
    retarget = Waypoint(plan[survey].pose, speed_m_s=approach_speed_m_s, hold_s=retarget_settle_s, label="retarget_1")
    plan.insert(survey+1, retarget)
    assert plan[survey+2].label == "cut_1"
    return plan, dict(calibration=calibration, survey=survey, shift_from=survey+1)


def ideal_plan_end_s(task, plan, *, max_s=None, **policy_options):
    """Input time at which ``WaypointPolicy`` finishes ``plan`` on an ideal integrator (None if not within max_s)."""
    policy = WaypointPolicy(task, list(plan), **policy_options)
    edge = np.asarray(plan[0].pose, float)
    steps = int(round((task.duration_s if max_s is None else max_s)/task.control_dt_s))
    for step in range(steps+1):
        action = policy({"proprio": {"tool_pose": edge.copy()}, "timestamp_s": step*task.control_dt_s})
        if policy.done:
            return step*task.control_dt_s
        if action["mode"] == ABSOLUTE_MODE:  # path_v3: the ideal integrator reaches the bounded absolute target
            edge = np.asarray(action["target"], float).copy()
            continue
        delta = action["target"]
        edge = np.r_[edge[:3]+delta[:3], (Rotation.from_rotvec(delta[3:])*Rotation.from_quat(edge[3:])).as_quat()]
    return None


@dataclass
class VisionRetargetPolicy(WaypointPolicy):
    """path_v2 WaypointPolicy plus the ADR 0009 wrist-depth survey and cut-1 retarget.

    ``depth_source(camera_id)`` returns the latest rendered observation of that
    camera (``depth_m``, ``valid_mask``, ``K``, ``T_world_camera``,
    ``physics_timestamp_s``); it must be current at the observation time. When
    the ``calibration`` / ``survey`` waypoint (``indices``) completes, the
    frame is surveyed (``wrist_depth_survey.survey_frame``); after the survey
    the shift estimate (``estimate_shift``, ``survey_method``) moves waypoint
    ``indices["shift_from"]`` and every later one by the estimate along the
    workpiece +x axis (= world +x for SLICE-PUSH-01). An invalid estimate
    applies no shift. Only depth, camera calibration and the measured edge pose
    (to exclude the knife) are read; never material state.
    """
    depth_source: Callable | None = None
    camera_id: str = "wrist_d405_0"
    survey_method: str = "end_face_plus_x"
    diagnostic_methods: tuple = ()
    indices: dict = field(default_factory=dict)
    survey_config: object = None
    vision: dict = field(default_factory=dict)
    last_vision: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        from .wrist_depth_survey import METHODS, SurveyConfig
        if self.survey_method not in METHODS or any(m not in METHODS for m in self.diagnostic_methods):
            raise ValueError(f"survey methods are {METHODS}")
        if set(self.indices) != {"calibration", "survey", "shift_from"} or not (
                0 <= self.indices["calibration"] < self.indices["survey"] < self.indices["shift_from"] < len(self.waypoints)):
            raise ValueError("indices needs ordered calibration < survey < shift_from waypoints")
        if self.survey_config is None:
            self.survey_config = SurveyConfig()
        self.waypoints = list(self.waypoints)
        self.vision = dict(adr="docs/adr/0009-demo3-wrist-depth-retarget.md", camera_id=self.camera_id,
                           method=self.survey_method, indices=dict(self.indices),
                           calibration=None, survey=None, estimate=None, diagnostics={},
                           applied_shift_m=0., applied_shift_world_m=[0., 0., 0.], shifted_labels=[],
                           inputs="rendered wrist depth + K + T_world_camera (robot kinematics) + measured edge pose; "
                                  "no particle/label/component/oracle state")

    def _edge(self, observation):
        if self.edge_estimator is None:
            return edge_from_observation(observation, self.task)
        return np.asarray(self.edge_estimator(observation), float)

    def _measure(self, observation, kind):
        from .wrist_depth_survey import BarGeometry, survey_frame
        stamp = float(observation["timestamp_s"])
        record = dict(kind=kind, waypoint=self.indices[kind], label=self.waypoints[self.indices[kind]].label,
                      observation_timestamp_s=stamp, frame_timestamp_s=None, valid=False, reason=None)
        if self.depth_source is None:
            return dict(record, reason="no depth source")
        try:
            frame = self.depth_source(self.camera_id)
        except Exception as error:  # an unavailable camera invalidates the survey; never the episode
            return dict(record, reason=f"depth source failed: {error!r}")
        record["frame_timestamp_s"] = float(frame["physics_timestamp_s"])
        if not abs(record["frame_timestamp_s"]-stamp) <= 1e-9:
            return dict(record, reason="stale depth frame (timestamp differs from the observation)")
        measurement = survey_frame(frame, BarGeometry.from_task(self.task), self.survey_config,
                                   edge_pose_world=self._edge(observation))
        return dict(record, valid=True, measurement=measurement)

    def _estimate(self):
        from .wrist_depth_survey import estimate_shift
        calibration, survey = self.vision["calibration"], self.vision["survey"]
        reasons = [f"{name}: {'not measured' if record is None else record['reason']}"
                   for name, record in (("calibration", calibration), ("survey", survey))
                   if record is None or not record["valid"]]
        if reasons:
            return None, dict(method=self.survey_method, valid=False, reasons=reasons, shift_m=None, applied_shift_m=0.)
        shift, estimate = estimate_shift(calibration["measurement"], survey["measurement"], method=self.survey_method,
                                         config=self.survey_config)
        for method in self.diagnostic_methods:
            self.vision["diagnostics"][method] = estimate_shift(
                calibration["measurement"], survey["measurement"], method=method, config=self.survey_config)[1]
        return shift, estimate

    def _apply(self, shift):
        vector = self.task.T_world_workpiece[:3, :3]@np.array([shift, 0., 0.])
        for k in range(self.indices["shift_from"], len(self.waypoints)):
            w = self.waypoints[k]
            self.waypoints[k] = replace(w, pose=tuple(np.r_[np.asarray(w.pose[:3], float)+vector, w.pose[3:]]))
        self.vision.update(applied_shift_m=float(shift), applied_shift_world_m=vector.tolist(),
                           shifted_labels=[w.label for w in self.waypoints[self.indices["shift_from"]:]])

    def __call__(self, observation):
        self.last_vision = None
        before = self.index
        action = super().__call__(observation)
        for kind in ("calibration", "survey"):
            k = self.indices[kind]
            if not before <= k < self.index or self.vision[kind] is not None:
                continue
            self.vision[kind] = self._measure(observation, kind)
            event = dict(kind=kind, record=self.vision[kind])
            if kind == "survey":
                shift, estimate = self._estimate()
                self.vision["estimate"] = estimate
                if shift is not None:
                    self._apply(shift)
                event.update(estimate=estimate, applied_shift_m=self.vision["applied_shift_m"],
                             applied_shift_world_m=self.vision["applied_shift_world_m"])
                # Restart from the waypoint after the survey with the (possibly shifted) targets.
                self.index, self._start_time, self._hold_start = k+1, None, None
                action = super().__call__(observation)
            self.last_vision = event
        return action


def vision_policy(task, name="path_v2_vision", *, depth_source=None, limits=None, cut_speed_m_s=.01,
                  **policy_overrides):
    """VisionRetargetPolicy of a profile with a ``vision_survey`` (path_v2_vision) for its declared tasks."""
    survey = vision_survey_options(name)
    if survey is None:
        raise ValueError(f"Controller profile {name!r} has no vision survey")
    if task.task_id not in survey["task_ids"]:
        raise ValueError(f"Controller profile {name!r} is declared for {survey['task_ids']} only, not {task.task_id}")
    policy_options, plan_options = controller_profile(name)
    plan, indices = slice_push_vision(task, cut_speed_m_s=cut_speed_m_s, retarget_settle_s=survey["retarget_settle_s"],
                                      lift_speed_m_s=survey["lift_speed_m_s"], raise_speed_m_s=survey["raise_speed_m_s"],
                                      **plan_options)
    return VisionRetargetPolicy(task, plan, limits or ActionLimits(), **{**policy_options, **policy_overrides},
                                depth_source=depth_source, camera_id=survey["camera_id"], survey_method=survey["method"],
                                diagnostic_methods=tuple(survey["diagnostic_methods"]), indices=indices)
