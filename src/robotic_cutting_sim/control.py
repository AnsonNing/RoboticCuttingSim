"""Policy-facing knife-edge action resolution (action API v2).

Pure CPU helpers shared by both live runtimes. They turn a relative
``cartesian_edge_delta`` action or a bounded absolute edge pose into the
absolute WORLD knife-edge target that the existing runtimes already accept.
Nothing here solves IK, steps physics or reads oracle material state.

Poses are SI XYZ + unit XYZW. A delta is [dx, dy, dz, rx, ry, rz]: metres
and a rotation vector in radians, about the knife-edge center.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .scene_frames import invert_transform, pose_matrix, transform_points, validate_pose_xyzw

DELTA_MODE = "cartesian_edge_delta"
DELTA_FRAMES = ("edge", "world")


class IKFailure(RuntimeError):
    """A runtime could not reach a commanded edge pose before any physics step.

    Both runtimes raise it before writing articulation targets, so a caller may
    reject the action and issue a hold instead. It subclasses RuntimeError so
    legacy callers that caught RuntimeError keep their behavior.
    """

    def __init__(self, message, info=None):
        super().__init__(message)
        self.info = dict(info or {})


@dataclass(frozen=True)
class ActionLimits:
    """Per-control-step motion bounds and an edge-position workspace box.

    Defaults at the 50 Hz task rate allow 0.1 m/s and 100 deg/s. The workspace
    box is in workpiece LOCAL coordinates and bounds the knife-edge center only;
    it is a safety envelope, not collision checking.
    """

    max_translation_m: float = .002
    max_rotation_rad: float = float(np.deg2rad(2.))
    workspace_min_local_m: tuple = (-.14, -.12, -.012)
    workspace_max_local_m: tuple = (.14, .12, .16)
    limit_absolute: bool = True

    def __post_init__(self):
        low, high = np.asarray(self.workspace_min_local_m, float), np.asarray(self.workspace_max_local_m, float)
        if (low.shape != (3,) or high.shape != (3,) or not np.isfinite(np.r_[low, high]).all()
                or np.any(high <= low) or not np.isfinite([self.max_translation_m, self.max_rotation_rad]).all()
                or self.max_translation_m <= 0 or not 0 < self.max_rotation_rad <= np.pi):
            raise ValueError("Action limits need positive finite step bounds and a nonempty workspace box")

    def to_dict(self):
        return {**asdict(self), "version": "action_limits_v1"}


def _clamp_norm(vector, limit):
    norm = float(np.linalg.norm(vector))
    if norm <= limit:
        return vector.copy(), norm, False
    return vector*(limit/norm), norm, True


def compose_delta(reference_pose, delta, frame="edge"):
    """Unbounded composition of a delta with a reference edge pose."""
    pose = validate_pose_xyzw(reference_pose)
    delta = np.asarray(delta, float)
    if pose.shape != (7,) or delta.shape != (6,) or not np.isfinite(delta).all():
        raise ValueError("Expected one reference edge pose and a finite 6-vector delta")
    if frame not in DELTA_FRAMES:
        raise ValueError("Delta frame must be 'edge' or 'world'")
    rotation = Rotation.from_quat(pose[3:])
    step = Rotation.from_rotvec(delta[3:])
    if frame == "edge":
        return np.r_[pose[:3]+rotation.apply(delta[:3]), (rotation*step).as_quat()]
    return np.r_[pose[:3]+delta[:3], (step*rotation).as_quat()]


def bound_target(reference_pose, target_pose, limits, T_world_workpiece=None):
    """Clamp an absolute target to one step from the reference and to the workspace.

    The bounded rotation follows the shortest arc from the reference orientation.
    Returns the bounded pose and a report of requested/applied motion.
    """
    reference, target = validate_pose_xyzw(reference_pose), validate_pose_xyzw(target_pose)
    if reference.shape != (7,) or target.shape != (7,):
        raise ValueError("Expected single edge poses")
    translation, requested_translation, translation_clamped = _clamp_norm(
        target[:3]-reference[:3], limits.max_translation_m)
    r_reference = Rotation.from_quat(reference[3:])
    relative = (Rotation.from_quat(target[3:])*r_reference.inv()).as_rotvec()
    rotvec, requested_rotation, rotation_clamped = _clamp_norm(relative, limits.max_rotation_rad)
    position = reference[:3]+translation
    workspace_clamped = False
    T = np.eye(4) if T_world_workpiece is None else np.asarray(T_world_workpiece, float)
    local = transform_points(invert_transform(T), position)
    bounded_local = np.clip(local, limits.workspace_min_local_m, limits.workspace_max_local_m)
    if not np.allclose(bounded_local, local, atol=0, rtol=0):
        workspace_clamped = True
        position = transform_points(T, bounded_local)
    quaternion = (Rotation.from_rotvec(rotvec)*r_reference).as_quat()
    pose = np.r_[position, quaternion/np.linalg.norm(quaternion)]
    return pose, {"requested_translation_m": requested_translation,
                  "applied_translation_m": float(np.linalg.norm(position-reference[:3])),
                  "requested_rotation_rad": requested_rotation,
                  "applied_rotation_rad": float(np.linalg.norm(rotvec)),
                  "translation_clamped": translation_clamped, "rotation_clamped": rotation_clamped,
                  "workspace_clamped": workspace_clamped}


def resolve_delta(reference_pose, delta, limits, *, frame="edge", T_world_workpiece=None):
    """Delta action -> bounded absolute WORLD edge target and its report."""
    delta = np.asarray(delta, float)
    if delta.shape != (6,) or not np.isfinite(delta).all():
        raise ValueError("A delta action is a finite [dx,dy,dz,rx,ry,rz]")
    # Clamp the commanded components before composing so the edge-frame
    # direction of a saturated request is preserved.
    translation, _, _ = _clamp_norm(delta[:3], limits.max_translation_m)
    rotation, _, _ = _clamp_norm(delta[3:], limits.max_rotation_rad)
    pose, report = bound_target(reference_pose, compose_delta(reference_pose, np.r_[translation, rotation], frame),
                                limits, T_world_workpiece)
    report.update({"requested_translation_m": float(np.linalg.norm(delta[:3])),
                   "requested_rotation_rad": float(np.linalg.norm(delta[3:])),
                   "translation_clamped": bool(np.linalg.norm(delta[:3]) > limits.max_translation_m),
                   "rotation_clamped": bool(np.linalg.norm(delta[3:]) > limits.max_rotation_rad)})
    return pose, report


def normalized_delta(action, limits):
    """Map an RL action in [-1,1]^6 to a metric delta at the step limits."""
    action = np.asarray(action, float)
    if action.shape != (6,) or not np.isfinite(action).all():
        raise ValueError("A normalized delta is a finite 6-vector")
    action = np.clip(action, -1., 1.)
    return np.r_[action[:3]*limits.max_translation_m, action[3:]*limits.max_rotation_rad]


def nominal_edge_pose(proprio, hand_from_edge_xyzw):
    """Commanded-frame edge pose from deployable hand FK when available.

    In physical-grasp mode the measured free-tool pose is privileged; IK targets
    the nominal hand mount, so delta actions accumulate on that frame.
    """
    if "hand_pose" in proprio:
        T = pose_matrix(proprio["hand_pose"])@pose_matrix(hand_from_edge_xyzw)
        return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()]
    return validate_pose_xyzw(proprio["tool_pose"]).copy()
