"""Explicit rigid frame conversions for geometry, runtime and offline validation.

T_destination_source maps source column vectors into destination coordinates.
Public poses are meters, XYZ + unit XYZW. Nothing here moves a simulator body.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def validate_pose_xyzw(poses):
    pose = np.asarray(poses, dtype=float)
    if pose.shape[-1:] != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite (...,7) XYZ+XYZW poses")
    # Float32 articulated FK chains can accumulate sub-ppm norm roundoff.
    # This input representation tolerance is separate from rotation-error gates.
    if not np.allclose(np.linalg.norm(pose[..., 3:], axis=-1), 1., atol=1e-6, rtol=0):
        raise ValueError("Frame quaternions must already be unit XYZW; convention is never guessed")
    return pose


def xyzw_to_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise ValueError("Expected a finite (...,4) quaternion")
    return q[..., [3, 0, 1, 2]].copy()


def wxyz_to_xyzw(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise ValueError("Expected a finite (...,4) quaternion")
    return q[..., [1, 2, 3, 0]].copy()


def pose_matrix(pose_xyzw):
    pose = validate_pose_xyzw(pose_xyzw)
    if pose.shape != (7,):
        raise ValueError("A transform needs exactly one pose")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def validate_transform(transform):
    matrix = np.asarray(transform, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-7, rtol=0)):
        raise ValueError("Expected a proper rigid 4x4 column-vector transform; scale/reflection is not accepted")
    return matrix


def matrix_pose(transform):
    matrix = validate_transform(transform)
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


def invert_transform(transform):
    matrix = validate_transform(transform)
    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T@matrix[:3, 3]
    return inverse


def transform_points(T_destination_source, points):
    matrix = validate_transform(T_destination_source)
    points = np.asarray(points, dtype=float)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Expected finite (...,3) points")
    return points@matrix[:3, :3].T+matrix[:3, 3]


def transform_poses(T_destination_source, poses_xyzw):
    matrix = validate_transform(T_destination_source)
    poses = validate_pose_xyzw(poses_xyzw)
    flat = poses.reshape(-1, 7)
    rotation = Rotation.from_matrix(matrix[:3, :3])*Rotation.from_quat(flat[:, 3:])
    result = np.c_[transform_points(matrix, flat[:, :3]), rotation.as_quat()]
    return result.reshape(poses.shape)


def compose_pose(parent_pose_xyzw, local_pose_xyzw):
    """Compose a fixed local pose with one or multiple parent poses."""
    parent, local = validate_pose_xyzw(parent_pose_xyzw), validate_pose_xyzw(local_pose_xyzw)
    if local.shape != (7,):
        raise ValueError("The fixed local mount must be one XYZ+XYZW pose")
    flat = parent.reshape(-1, 7)
    rotation = Rotation.from_quat(flat[:, 3:])
    return np.c_[flat[:, :3]+rotation.apply(local[:3]),
                  (rotation*Rotation.from_quat(local[3:])).as_quat()].reshape(parent.shape)


def pose_errors(expected_xyzw, actual_xyzw):
    expected, actual = validate_pose_xyzw(expected_xyzw), validate_pose_xyzw(actual_xyzw)
    if expected.shape != actual.shape:
        raise ValueError("Expected and measured poses must have matching shapes/timestamps")
    position = np.linalg.norm(actual[..., :3]-expected[..., :3], axis=-1)
    rotation = (Rotation.from_quat(expected.reshape(-1, 7)[:, 3:]).inv()
                * Rotation.from_quat(actual.reshape(-1, 7)[:, 3:])).magnitude()
    return position, rotation.reshape(position.shape)


def usd_row_to_column(matrix, meters_per_unit=1.):
    """Convert a Gf row-vector matrix into an SI column-vector rigid transform."""
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("USD metersPerUnit must be finite and positive")
    result = np.asarray(matrix, dtype=float).T.copy()
    result[:3, 3] *= meters_per_unit
    return validate_transform(result)


def column_to_usd_row(transform, meters_per_unit=1.):
    matrix = validate_transform(transform).copy()
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("USD metersPerUnit must be finite and positive")
    matrix[:3, 3] /= meters_per_unit
    return matrix.T


def usd_camera_to_cv(T_world_usd_camera):
    """USD camera: +X right,+Y up,-Z forward. CV: +X right,+Y down,+Z forward."""
    return validate_transform(T_world_usd_camera)@np.diag([1., -1., -1., 1.])


def rotate_wrench_at_same_point(T_destination_source, wrench_force_then_torque):
    """Rotate a wrench about the same physical application point; no origin shift.

    Moving a wrench from knife edge to hand COM additionally requires r×F,
    as performed by the robot coupler. A frame translation alone is not that shift.
    """
    matrix = validate_transform(T_destination_source)
    wrench = np.asarray(wrench_force_then_torque, dtype=float)
    if wrench.shape[-1:] != (6,) or not np.isfinite(wrench).all():
        raise ValueError("Expected finite (...,6) force-then-torque wrenches")
    return np.concatenate([wrench[..., :3]@matrix[:3, :3].T,
                           wrench[..., 3:]@matrix[:3, :3].T], axis=-1)


@dataclass(frozen=True)
class WorkpieceTarget:
    """The cutting target belongs to the workpiece, never to absolute world Z."""
    world_from_workpiece_xyzw: tuple = (0., 0., 0., 0., 0., 0., 1.)
    height_m: float = .015
    edge_clearance_m: float = .0005

    def __post_init__(self):
        pose = validate_pose_xyzw(self.world_from_workpiece_xyzw)
        if pose.shape != (7,) or not np.isfinite([self.height_m, self.edge_clearance_m]).all():
            raise ValueError("Target requires one rigid workpiece pose and finite local heights")

    @property
    def T_world_workpiece(self):
        return pose_matrix(self.world_from_workpiece_xyzw)

    @property
    def T_workpiece_world(self):
        return invert_transform(self.T_world_workpiece)

    @property
    def normal_world(self):
        return self.T_world_workpiece[:3, 2].copy()

    @property
    def edge_height_local_m(self):
        return self.height_m+self.edge_clearance_m

    def points_to_world(self, points_local):
        return transform_points(self.T_world_workpiece, points_local)

    def points_to_local(self, points_world):
        return transform_points(self.T_workpiece_world, points_world)

    def poses_to_world(self, poses_local_xyzw):
        return transform_poses(self.T_world_workpiece, poses_local_xyzw)

    def poses_to_local(self, poses_world_xyzw):
        return transform_poses(self.T_workpiece_world, poses_world_xyzw)

    def signed_plane_distance(self, points_world):
        return self.points_to_local(points_world)[..., 2]-self.height_m

    def plane_world(self):
        point = self.points_to_world([0, 0, self.height_m])
        return np.r_[self.normal_world, -np.dot(self.normal_world, point)]

    def solver_gravity(self, gravity_world=(0., 0., -9.81)):
        """Gravity if the backend stores particle state in the workpiece frame."""
        vector = np.asarray(gravity_world, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Gravity must be a finite world vector")
        return self.T_workpiece_world[:3, :3]@vector

    def require_backend_frame_support(self, supports_nonidentity=False):
        """Current TLMPM fixture is authored in world coordinates: reject fake relocation."""
        if not supports_nonidentity and not np.allclose(self.T_world_workpiece, np.eye(4), atol=1e-9, rtol=0):
            raise NotImplementedError("This backend has a world-authored fixture and supports identity workpiece pose only; transforming visuals or targets alone is invalid")
        return True
