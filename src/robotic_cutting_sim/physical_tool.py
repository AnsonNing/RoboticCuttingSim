"""Geometry and frame contracts for an independent, physically grasped tool.

All poses are SI XYZ+XYZW. The TOOL frame has the nominal HAND orientation,
with its origin at the handle center. Nominal mounts initialize/plan a grasp;
none of these helpers constrains or corrects a simulated body's measured pose.
The provisional densities/grip settings are engineering inputs, not calibrated
material properties or evidence that either runtime has achieved a grasp.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .robot import wedge_mesh
from .scene_frames import (compose_pose, invert_transform, matrix_pose,
                           pose_matrix, transform_points, validate_pose_xyzw)


@dataclass(frozen=True)
class ToolGraspConfig:
    """Finite per-finger drive limits and predeclared hold engineering gates."""

    friction: float = .6
    max_finger_force_n: float = 20.
    initial_finger_position_m: float = .014
    closed_target_m: float = .002
    stiffness_n_m: float = 2000.
    damping_n_s_m: float = 20.
    settle_s: float = 2.
    stability_window_s: float = .5
    translation_tolerance_m: float = .002
    rotation_tolerance_rad: float = np.pi / 60
    linear_speed_tolerance_m_s: float = .02
    angular_speed_tolerance_rad_s: float = .2
    min_bilateral_contact_fraction: float = .95
    max_penetration_m: float = .001

    def __post_init__(self):
        values = asdict(self)
        if not np.isfinite(list(values.values())).all():
            raise ValueError("Grasp parameters must be finite")
        if (min(self.friction, self.max_finger_force_n, self.closed_target_m,
                self.damping_n_s_m) < 0 or self.stiffness_n_m <= 0
                or not 0 <= self.closed_target_m <= self.initial_finger_position_m <= .04
                or not 0 < self.stability_window_s <= self.settle_s
                or not 0 < self.min_bilateral_contact_fraction <= 1
                or min(self.translation_tolerance_m, self.rotation_tolerance_rad,
                       self.linear_speed_tolerance_m_s, self.angular_speed_tolerance_rad_s,
                       self.max_penetration_m) <= 0):
            raise ValueError("Invalid finite grip drive, position, or stability settings")

    def to_dict(self):
        return {**asdict(self), "force_limit_scope": "each finger prismatic joint",
                "status": "PROVISIONAL_ENGINEERING_SETTINGS"}


def _mass_properties(vertices, triangles, density):
    """Exact signed-tetrahedron integrals of a closed outward triangle mesh."""
    a, b, c = np.asarray(vertices, float)[np.asarray(triangles, int)].transpose(1, 0, 2)
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    total_volume = float(volume.sum())
    if total_volume <= 1e-15:
        raise ValueError("Closed outward mesh must enclose positive finite volume")
    sums = a + b + c
    com = (volume[:, None] * sums).sum(0) / (4 * total_volume)
    second = sum(np.einsum("i,ij,ik->jk", volume, v, v)
                 for v in (sums, a, b, c)) / 20
    mass = density * total_volume
    inertia_origin = density * (np.trace(second) * np.eye(3) - second)
    inertia_com = inertia_origin - mass * (com @ com * np.eye(3) - np.outer(com, com))
    return total_volume, mass, com, (inertia_com + inertia_com.T) / 2


def _convex_triangles(vertices):
    points = np.asarray(vertices, float)
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    a, b, c = points[faces].transpose(1, 0, 2)
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a - points.mean(0)) < 0
    faces[inward] = faces[inward, ::-1]
    return faces


@dataclass(frozen=True)
class ToolShape:
    """One convex compound collider, with vertices expressed in TOOL coordinates."""

    name: str
    vertices_tool_m: np.ndarray
    triangles: np.ndarray
    density_kg_m3: float
    role: str

    def __post_init__(self):
        vertices = np.array(self.vertices_tool_m, dtype=float, copy=True)
        triangles = np.array(self.triangles, dtype=np.int32, copy=True)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
                or not np.isfinite(vertices).all() or triangles.ndim != 2
                or triangles.shape[1] != 3 or len(triangles) < 4
                or triangles.min() < 0 or triangles.max() >= len(vertices)
                or not np.isfinite(self.density_kg_m3) or self.density_kg_m3 <= 0):
            raise ValueError("A tool shape requires a finite closed convex solid and density")
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                        triangles[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        a, b, c = vertices[triangles].transpose(1, 0, 2)
        outward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a - vertices.mean(0))
        volume = _mass_properties(vertices, triangles, self.density_kg_m3)[0]
        if (not np.all(counts == 2) or np.any(outward <= 0)
                or not np.isclose(volume, ConvexHull(vertices).volume, rtol=1e-8, atol=1e-14)):
            raise ValueError("Tool collider must be a closed outward convex triangle mesh")
        vertices.setflags(write=False)
        triangles.setflags(write=False)
        object.__setattr__(self, "vertices_tool_m", vertices)
        object.__setattr__(self, "triangles", triangles)

    @property
    def mass_properties(self):
        """(volume_m3, mass_kg, COM_tool_m, inertia_at_COM_in_TOOL_kg_m2)."""
        return _mass_properties(self.vertices_tool_m, self.triangles, self.density_kg_m3)

    def to_dict(self):
        volume, mass, com, inertia = self.mass_properties
        return {"name": self.name, "role": self.role, "density_kg_m3": self.density_kg_m3,
                "vertices_tool_m": self.vertices_tool_m.tolist(), "triangles": self.triangles.tolist(),
                "volume_m3": volume, "mass_kg": mass, "com_tool_m": com.tolist(),
                "inertia_com_tool_kg_m2": inertia.tolist(), "collision_approximation": "convexHull"}


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    hand_from_tool_xyz_xyzw: tuple
    tool_from_grasp_xyz_xyzw: tuple
    tool_from_edge_xyz_xyzw: tuple
    handle_half_extents_m: tuple
    blade_length_m: float
    blade_height_m: float
    blade_half_width_m: float
    shapes: tuple[ToolShape, ...]
    source_task_id: str | None = None

    def __post_init__(self):
        for pose in (self.hand_from_tool_xyz_xyzw, self.tool_from_grasp_xyz_xyzw,
                     self.tool_from_edge_xyz_xyzw):
            if validate_pose_xyzw(pose).shape != (7,):
                raise ValueError("Each tool frame requires one XYZ+XYZW pose")
        dimensions = np.r_[self.handle_half_extents_m, self.blade_length_m,
                           self.blade_height_m, self.blade_half_width_m]
        if (not np.isfinite(dimensions).all() or dimensions.shape != (6,) or np.any(dimensions <= 0)
                or not self.shapes or len({shape.name for shape in self.shapes}) != len(self.shapes)):
            raise ValueError("Tool dimensions and named physical shapes must be nonempty and positive")

    @property
    def T_hand_tool(self):
        return pose_matrix(self.hand_from_tool_xyz_xyzw)

    @property
    def T_tool_grasp(self):
        return pose_matrix(self.tool_from_grasp_xyz_xyzw)

    @property
    def T_tool_edge(self):
        return pose_matrix(self.tool_from_edge_xyz_xyzw)

    @property
    def mass_kg(self):
        return float(sum(shape.mass_properties[1] for shape in self.shapes))

    @property
    def com_tool_m(self):
        return sum(shape.mass_properties[1] * shape.mass_properties[2] for shape in self.shapes) / self.mass_kg

    @property
    def inertia_com_tool_kg_m2(self):
        com, inertia = self.com_tool_m, np.zeros((3, 3))
        for shape in self.shapes:
            _, mass, local_com, local_inertia = shape.mass_properties
            delta = local_com - com
            inertia += local_inertia + mass * (delta @ delta * np.eye(3) - np.outer(delta, delta))
        return inertia

    @property
    def principal_moments_kg_m2(self):
        return np.linalg.eigvalsh(self.inertia_com_tool_kg_m2)

    @property
    def principal_axes_xyzw(self):
        _, axes = np.linalg.eigh(self.inertia_com_tool_kg_m2)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        return Rotation.from_matrix(axes).as_quat()

    def to_dict(self):
        return {"tool_id": self.tool_id, "source_task_id": self.source_task_id,
                "frame_convention": "SI XYZ+XYZW; TOOL orientation nominal HAND; origin handle center",
                "hand_from_tool_xyz_xyzw": list(self.hand_from_tool_xyz_xyzw),
                "tool_from_grasp_xyz_xyzw": list(self.tool_from_grasp_xyz_xyzw),
                "tool_from_edge_xyz_xyzw": list(self.tool_from_edge_xyz_xyzw),
                "handle_half_extents_m": list(self.handle_half_extents_m),
                "blade_length_m": self.blade_length_m, "blade_height_m": self.blade_height_m,
                "blade_half_width_m": self.blade_half_width_m,
                "mass_kg": self.mass_kg, "com_tool_m": self.com_tool_m.tolist(),
                "inertia_com_tool_kg_m2": self.inertia_com_tool_kg_m2.tolist(),
                "principal_moments_kg_m2": self.principal_moments_kg_m2.tolist(),
                "principal_axes_xyzw": self.principal_axes_xyzw.tolist(),
                "mass_model": "Exact closed-mesh volumes, nonoverlapping components meeting at faces",
                "material_status": "PROVISIONAL polymer1100/steel7800kg_m3; uncalibrated",
                "contact_scope": "Rigid compound versus robot/fixtures; TLMPM soft contact uses blade only",
                "nominal_mount_usage": "RESET_AND_PLANNING_ONLY; never measured-tool pose correction",
                "shapes": [shape.to_dict() for shape in self.shapes]}


def _shaft_patch(T_tool_edge, start, length, height, half_width):
    """A rectangle inside an actual blade face that separates shaft from blade.

    The shaft's start is on the handle's front face. Both half-space conditions
    make the three component interiors disjoint while retaining shared faces.
    CUT selects the rear blade face; TRIM selects the upper sloping face.
    """
    lower = .0004  # Must match the unchanged TLMPM wedge, not a new blade profile.
    faces = [([-length, 0, (height - lower) / 2], [-1, 0, 0], [0, 0, 1], height + lower),
             ([-length / 2, 0, height / 2], [height, 0, length], [-length, 0, height], np.hypot(length, height)),
             ([-length / 2, 0, -lower / 2], [lower, 0, -length], [-length, 0, -lower], np.hypot(length, lower))]
    candidates = []
    for center, normal, tangent, face_length in faces:
        normal, tangent = np.asarray(normal, float), np.asarray(tangent, float)
        normal /= np.linalg.norm(normal)
        tangent /= np.linalg.norm(tangent)
        center = transform_points(T_tool_edge, center)
        normal = T_tool_edge[:3, :3] @ normal
        tangent = T_tool_edge[:3, :3] @ tangent
        across = T_tool_edge[:3, 1]
        patch = np.array([center + sy * min(.004, .4 * half_width) * across
                          + sx * min(.0015, .4 * face_length) * tangent
                          for sy, sx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]])
        distances = (start - center) @ normal
        if distances.min() > 1e-6 and patch[:, 2].min() > start[:, 2].max() + 1e-6:
            direction = start.mean(0) - center
            candidates.append((normal @ direction / np.linalg.norm(direction), patch))
    # Rotated future heads may present a triangular end face to the handle.
    # A centroid shrink stays inside that triangle and leaves a real mating face.
    triangle = np.array([[0., 0., 0.], [-length, 0., height], [-length, 0., -lower]])
    for side in [-1., 1.]:
        local = triangle.copy()
        local[:, 1] = side * half_width
        center_local = local.mean(0)
        center = transform_points(T_tool_edge, center_local)
        patch = transform_points(T_tool_edge, center_local + .35 * (local - center_local))
        normal = T_tool_edge[:3, 1] * side
        if ((start - center) @ normal).min() > 1e-6 and patch[:, 2].min() > start[:, 2].max() + 1e-6:
            direction = start.mean(0) - center
            candidates.append((normal @ direction / np.linalg.norm(direction), patch))
    if not candidates:
        raise ValueError("Blade mount cannot connect a shaft without penetrating handle/blade; unsupported geometry")
    return max(candidates, key=lambda item: item[0])[1]


def make_tool_spec(task=None, *, hand_from_edge_xyz_xyzw=None, blade_length_m=None,
                   blade_height_m=None, blade_half_width_m=None):
    """Make one coherent tool independently of workpiece/seed/goal semantics.

    Explicit overrides take precedence over TaskSpec. Omitting task uses the
    legacy TRIM head/mount. No simulator, asset download, or physics step occurs.
    """
    def select(explicit, name, fallback):
        return explicit if explicit is not None else getattr(task, name, fallback)

    hand_edge = select(hand_from_edge_xyz_xyzw, "hand_from_edge_xyz_xyzw", (0., 0., .18, 1., 0., 0., 0.))
    hand_edge = validate_pose_xyzw(hand_edge)
    if hand_edge.shape != (7,):
        raise ValueError("One nominal hand-to-edge pose is required")
    length = float(select(blade_length_m, "blade_length_m", .012))
    height = float(select(blade_height_m, "blade_height_m", .004))
    half_width = float(select(blade_half_width_m, "blade_half_width_m", .025))
    if not np.isfinite([length, height, half_width]).all() or min(length, height, half_width) <= 0:
        raise ValueError("Blade dimensions must be positive and finite")
    hand_tool = (0., 0., .10365, 0., 0., 0., 1.)
    tool_edge = matrix_pose(invert_transform(pose_matrix(hand_tool)) @ pose_matrix(hand_edge))
    half = (.008, .011, .020)
    handle_vertices = np.array([[x, y, z] for x in [-half[0], half[0]]
                                for y in [-half[1], half[1]] for z in [-half[2], half[2]]])
    start = np.array([[x, y, half[2]] for x in [-.003, .003] for y in [-.004, .004]])
    T_tool_edge = pose_matrix(tool_edge)
    end = _shaft_patch(T_tool_edge, start, length, height, half_width)
    shaft_vertices = np.r_[start, end]
    blade_vertices, blade_triangles = wedge_mesh(length, height, half_width)
    blade_vertices = transform_points(T_tool_edge, blade_vertices)
    shapes = (ToolShape("Handle", handle_vertices, _convex_triangles(handle_vertices), 1100., "handle"),
              ToolShape("Shaft", shaft_vertices, _convex_triangles(shaft_vertices), 7800., "shaft"),
              ToolShape("Blade", blade_vertices, blade_triangles, 7800., "blade"))
    return ToolSpec("panda_compound_cutting_tool_v1", hand_tool, (0., 0., 0., 0., 0., 0., 1.),
                    tuple(tool_edge), half, length, height, half_width, shapes, getattr(task, "task_id", None))


def measured_edge_pose(world_tool_pose_xyzw, spec):
    """Actual edge from actual free-body state; no hand pose enters this path."""
    return compose_pose(world_tool_pose_xyzw, spec.tool_from_edge_xyz_xyzw)


def nominal_hand_pose(world_edge_target_xyzw, spec):
    """IK target only. Applying this to a measured free tool would fake a grasp."""
    T_world_edge = pose_matrix(world_edge_target_xyzw)
    return matrix_pose(T_world_edge @ invert_transform(spec.T_tool_edge) @ invert_transform(spec.T_hand_tool))


def world_com_position(world_tool_pose_xyzw, spec):
    return transform_points(pose_matrix(world_tool_pose_xyzw), spec.com_tool_m)


def shift_wrench_world(wrench_force_then_torque, from_point_world_m, to_point_world_m):
    """Same world axes, new reference point: tau_to=tau_from+(p_from-p_to)×F."""
    wrench = np.asarray(wrench_force_then_torque, float)
    source, destination = np.asarray(from_point_world_m, float), np.asarray(to_point_world_m, float)
    if (wrench.shape != (6,) or source.shape != (3,) or destination.shape != (3,)
            or not np.isfinite(np.r_[wrench, source, destination]).all()):
        raise ValueError("Expected one finite world force-then-torque and two SI world points")
    return np.r_[wrench[:3], wrench[3:] + np.cross(source - destination, wrench[:3])]


def edge_wrench_to_com(wrench_force_then_torque_world, world_tool_pose_xyzw, spec):
    return shift_wrench_world(wrench_force_then_torque_world,
                              measured_edge_pose(world_tool_pose_xyzw, spec)[:3],
                              world_com_position(world_tool_pose_xyzw, spec))


def edge_twist_from_com(linear_com_world_m_s, angular_world_rad_s, world_tool_pose_xyzw, spec):
    """Return [linear_edge,angular] in world axes, preserving virtual power."""
    linear, angular = np.asarray(linear_com_world_m_s, float), np.asarray(angular_world_rad_s, float)
    if linear.shape != (3,) or angular.shape != (3,) or not np.isfinite(np.r_[linear, angular]).all():
        raise ValueError("Expected finite world COM linear/angular velocities")
    delta = measured_edge_pose(world_tool_pose_xyzw, spec)[:3] - world_com_position(world_tool_pose_xyzw, spec)
    return np.r_[linear + np.cross(angular, delta), angular]


def grasp_relative_state(world_hand_pose_xyzw, world_tool_pose_xyzw, spec):
    """Privileged geometric diagnostic only; bilateral contact remains essential."""
    measured = invert_transform(pose_matrix(world_hand_pose_xyzw)) @ pose_matrix(world_tool_pose_xyzw)
    error = invert_transform(spec.T_hand_tool) @ measured
    return {"hand_from_tool_measured_xyz_xyzw": matrix_pose(measured),
            "nominal_translation_error_m": float(np.linalg.norm(error[:3, 3])),
            "nominal_rotation_error_rad": float(Rotation.from_matrix(error[:3, :3]).magnitude()),
            "physical_grasp_validated": False,
            "limitation": "Relative pose alone cannot establish finger contact or physical support"}
