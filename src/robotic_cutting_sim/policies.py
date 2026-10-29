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
}


def controller_profile(name):
    """(policy options, plan-builder options) of a named controller profile."""
    if name not in CONTROLLER_PROFILES:
        raise ValueError(f"Unknown controller profile {name!r}; known: {sorted(CONTROLLER_PROFILES)}")
    profile = CONTROLLER_PROFILES[name]
    policy = {key: profile[key] for key in ("path_tracking", "cross_track_gain", "position_tolerance_m") if key in profile}
    plan = {key: profile[key] for key in ("approach_speed_m_s", "approach_slow_distance_m", "settle_s") if key in profile}
    return policy, plan


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

    def __post_init__(self):
        if not 0 < self.cross_track_gain <= 1:
            raise ValueError("cross_track_gain must be in (0, 1]")

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


def shape_approaches(plan, *, approach_speed_m_s=None, approach_slow_distance_m=.01, settle_s=0.):
    """Slow the final approach to every cut start and settle there before cutting (path_v2).

    For each cut waypoint whose predecessor (the cut start) is not cutting: the
    last ``approach_slow_distance_m`` of the straight planned move to the start
    is driven at ``approach_speed_m_s`` (an intermediate waypoint is inserted
    where the slow part begins; it copies the start's flags, speed and label),
    and the start is held for ``settle_s``. ``approach_speed_m_s=None`` and
    ``settle_s=0`` return the plan unchanged. A start that is the first
    waypoint has no planned predecessor segment and only gets the settle hold.
    """
    if approach_speed_m_s is None and not settle_s:
        return list(plan)
    if approach_speed_m_s is not None and not approach_speed_m_s > 0:
        raise ValueError("approach_speed_m_s must be positive (or None)")
    if not approach_slow_distance_m > 0 or settle_s < 0:
        raise ValueError("approach_slow_distance_m must be positive and settle_s non-negative")
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
