"""Scripted knife-motion policies that emit the same delta actions as RL.

A WaypointPolicy reads only the observation (measured proprioception) and
outputs bounded ``cartesian_edge_delta`` actions toward edge waypoints, with
optional sawing (oscillation along the edge line) while cutting. The motion
builders express common knife uses — push cut, oblique/yawed cut, slicing/saw
cut and surface skimming — from the task's nominal geometry (initial workpiece
pose and dimensions known before the episode, never oracle material state).
"""
from dataclasses import dataclass, field

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

    @property
    def done(self):
        return self.index >= len(self.waypoints)

    def _target(self, waypoint, time_s):
        pose = np.asarray(waypoint.pose, float)
        if waypoint.saw_amplitude_m > 0:
            along = Rotation.from_quat(pose[3:]).apply([0., 1., 0.])
            phase = 2*np.pi*(time_s-self._start_time)/waypoint.saw_period_s
            pose = np.r_[pose[:3]+waypoint.saw_amplitude_m*np.sin(phase)*along, pose[3:]]
        return pose

    def __call__(self, observation):
        edge = edge_from_observation(observation, self.task)
        time_s = float(observation["timestamp_s"])
        while not self.done:
            waypoint = self.waypoints[self.index]
            if self._start_time is None:
                self._start_time = time_s
            final = np.asarray(waypoint.pose, float)
            position_error = np.linalg.norm(final[:3]-edge[:3])
            rotation_error = (Rotation.from_quat(final[3:])*Rotation.from_quat(edge[3:]).inv()).magnitude()
            if position_error <= self.position_tolerance_m and rotation_error <= self.rotation_tolerance_rad:
                self.index, self._start_time = self.index+1, None
                continue
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


def _pose(position, rotation):
    return tuple(np.r_[position, rotation.as_quat()])


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
                cut_speed_m_s=.01, clearance_m=.008, overtravel_m=.004, cut_index=0):
    """Approach above the target plane, cut along the advance line, retract.

    The cut line passes through the workpiece center at mid height; it ends
    below the workpiece bottom by ``overtravel_m`` (vertical component).
    """
    rotation = oriented(task, tilt_deg=tilt_deg, yaw_deg=yaw_deg, roll_deg=roll_deg)
    advance = rotation.apply([1., 0., 0.])
    if advance[2] > -.2:
        raise ValueError("A through cut needs an advance direction pointing down")
    T = task.T_world_workpiece
    height = task.geometry.base_height_m
    center = T[:3, :3]@np.array(task.target_origin_local_m)+T[:3, 3]+T[:3, 2]*height/2
    top, bottom = center[2]+height/2, center[2]-height/2
    start = center+advance*((top+clearance_m-center[2])/advance[2])
    end = center+advance*((bottom-overtravel_m-center[2])/advance[2])
    above = start+np.array([0., 0., .03])
    return [Waypoint(_pose(above, rotation)), Waypoint(_pose(start, rotation)),
            Waypoint(_pose(end, rotation), cut=True, saw_amplitude_m=saw_amplitude_m, saw_period_s=saw_period_s,
                     speed_m_s=cut_speed_m_s, cut_index=cut_index),
            Waypoint(_pose(end, rotation), cut=False),
            Waypoint(_pose(start+np.array([0., 0., .03]), rotation))]


def skim(task, *, clearance_m=None, margin_m=.03, speed_m_s=.025, saw_amplitude_m=0.):
    """Move the edge across the material above the target plane (TRIM).

    The pass spans the bumps (or the whole workpiece without bumps) plus
    ``margin_m`` on each side, at the target plane plus the edge clearance.
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
    return [Waypoint(_pose(above, rotation)), Waypoint(_pose(start, rotation)),
            Waypoint(_pose(end, rotation), cut=True, speed_m_s=speed_m_s, saw_amplitude_m=saw_amplitude_m),
            Waypoint(_pose(end+np.array([0., 0., .03]), rotation))]
