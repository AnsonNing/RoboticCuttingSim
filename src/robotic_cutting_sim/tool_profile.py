"""Parametric straight-edge cutting tools: one profile, every consumer.

A ToolProfile defines the blade as a convex cross-section in the edge frame's
(x, z) plane, extruded along the edge line (y). The same profile produces the
rigid colliders and mass properties (ToolSpec), the visual mesh, the TLMPM
blade SDF polygon and the nominal hand-to-edge mount, so they cannot diverge.

Edge frame: origin at the edge center, +x the advance direction (away from the
blade body), y along the edge, z the blade-thickness normal. TOOL frame:
nominal HAND orientation with its origin at the handle center (as in
physical_tool). Hand frame: +z approach toward the fingertips, y the finger
closing axis. Dimensions and densities are engineering choices, not a
calibrated real tool.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .physical_tool import ToolShape, ToolSpec, _convex_triangles
from .scene_frames import invert_transform, matrix_pose, pose_matrix, transform_points

LEGACY_LOWER_Z_M = -.0004  # The retained TLMPM wedge's lower rear vertex.
HAND_FROM_TOOL = (0., 0., .10365, 0., 0., 0., 1.)  # Handle center at the finger pads.


@dataclass(frozen=True)
class ToolProfile:
    profile_id: str
    edge_length_m: float
    blade_depth_m: float
    thickness_m: float
    bevel: str = "double"                 # double | single | legacy_wedge
    bevel_included_angle_deg: float = 25.
    handle_half_extents_hand_m: tuple = (.050, .011, .020)
    blade_density_kg_m3: float = 7800.
    handle_density_kg_m3: float = 1100.

    def __post_init__(self):
        values = [self.edge_length_m, self.blade_depth_m, self.thickness_m, self.bevel_included_angle_deg,
                  *self.handle_half_extents_hand_m, self.blade_density_kg_m3, self.handle_density_kg_m3]
        if (len(self.handle_half_extents_hand_m) != 3 or not np.isfinite(values).all() or min(values) <= 0
                or self.bevel not in ("double", "single", "legacy_wedge")
                or not 0 < self.bevel_included_angle_deg < 90):
            raise ValueError("Tool profile needs positive finite dimensions and a known bevel")
        if self.bevel != "legacy_wedge" and self.bevel_length_m > self.blade_depth_m:
            raise ValueError("Bevel is longer than the blade depth; increase the angle or depth")

    @property
    def half_width_m(self):
        return self.edge_length_m/2

    @property
    def bevel_length_m(self):
        rise = self.thickness_m/2 if self.bevel == "double" else self.thickness_m
        half_angle = np.deg2rad(self.bevel_included_angle_deg)/(2 if self.bevel == "double" else 1)
        return rise/np.tan(half_angle)

    def cross_section_xz(self):
        """Counter-clockwise convex blade polygon in edge (x, z), edge vertex first."""
        D, t, b = self.blade_depth_m, self.thickness_m, None
        if self.bevel == "legacy_wedge":
            return np.array([[0., 0.], [-D, t], [-D, LEGACY_LOWER_Z_M]])
        b = self.bevel_length_m
        if self.bevel == "double":
            h = t/2
            return np.array([[0., 0.], [-b, h], [-D, h], [-D, -h], [-b, -h]])
        # Single (chisel) bevel: flat reference face at z=0, bevel on +z.
        return np.array([[0., 0.], [-b, t], [-D, t], [-D, 0.]])

    def blade_mesh_edge(self):
        """Closed outward prism in edge coordinates: (vertices, triangles)."""
        section = self.cross_section_xz()
        y = self.half_width_m
        vertices = np.array([[x, s*y, z] for s in (-1., 1.) for x, z in section])
        return vertices, _convex_triangles(vertices)

    def hand_from_edge(self):
        """Edge at the far handle face plus the blade depth along hand +z.

        Edge +x = hand +z (advance away from the palm), edge z = hand y (the
        fingers squeeze across the blade thickness), edge y = hand x.
        """
        if self.bevel == "legacy_wedge":
            raise ValueError("Legacy wedges keep their task-specific mounts")
        R = np.column_stack([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        z = HAND_FROM_TOOL[2]+self.handle_half_extents_hand_m[2]+self.blade_depth_m
        return (0., 0., float(z), *Rotation.from_matrix(R).as_quat())

    def tlmpm_profile(self):
        """Float32 polygon and half-width consumed by tlmpm_v2."""
        return self.cross_section_xz().astype(np.float32), np.float32(self.half_width_m)

    def to_dict(self):
        return {**asdict(self), "version": "tool_profile_v1", "half_width_m": self.half_width_m,
                "bevel_length_m": None if self.bevel == "legacy_wedge" else self.bevel_length_m,
                "cross_section_xz_m": self.cross_section_xz().tolist(),
                "status": "ENGINEERING_GEOMETRY_UNCALIBRATED"}


RULER_KNIFE_V1 = ToolProfile("ruler_knife_v1", edge_length_m=.120, blade_depth_m=.040, thickness_m=.0015,
                             bevel="double", bevel_included_angle_deg=25.)
PROFILES = {RULER_KNIFE_V1.profile_id: RULER_KNIFE_V1}


def legacy_profile(task):
    """The retained task wedge expressed as a profile (TLMPM-identical polygon)."""
    return ToolProfile("legacy_wedge_"+task.task_id, edge_length_m=2*task.blade_half_width_m,
                       blade_depth_m=task.blade_length_m, thickness_m=task.blade_height_m, bevel="legacy_wedge")


def make_profile_tool_spec(profile):
    """Handle + blade compound for a spine-mounted straight-edge tool.

    Components meet at the spine plane and have disjoint interiors, matching
    physical_tool's compound-mass convention. No simulator is touched.
    """
    hand_edge = profile.hand_from_edge()
    T_tool_edge = invert_transform(pose_matrix(HAND_FROM_TOOL))@pose_matrix(hand_edge)
    half = np.asarray(profile.handle_half_extents_hand_m, float)
    handle = np.array([[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1])
                       for z in (-half[2], half[2])])
    blade_edge, _ = profile.blade_mesh_edge()
    blade = transform_points(T_tool_edge, blade_edge)
    if blade[:, 2].min() < half[2]-1e-9:
        raise ValueError("Blade must start at the handle's far face")
    shapes = (ToolShape("Handle", handle, _convex_triangles(handle), profile.handle_density_kg_m3, "handle"),
              ToolShape("Blade", blade, _convex_triangles(blade), profile.blade_density_kg_m3, "blade"))
    return ToolSpec(profile.profile_id, HAND_FROM_TOOL, (0., 0., 0., 0., 0., 0., 1.), tuple(matrix_pose(T_tool_edge)),
                    tuple(half), profile.blade_depth_m, profile.thickness_m, profile.half_width_m, shapes, None)


def require_profile_consistency(spec, profile):
    """Reject a ToolSpec whose blade collider is not exactly the profile's blade."""
    blades = [shape for shape in spec.shapes if shape.role == "blade"]
    if len(blades) != 1:
        raise NotImplementedError("A profile tool needs exactly one blade collider")
    expected = transform_points(spec.T_tool_edge, profile.blade_mesh_edge()[0])
    actual = np.asarray(blades[0].vertices_tool_m)
    if expected.shape != actual.shape or np.abs(np.sort(expected, 0)-np.sort(actual, 0)).max() > 1e-9:
        raise NotImplementedError("Blade collider differs from the TLMPM profile polygon")
    if not np.allclose([spec.blade_length_m, spec.blade_height_m, spec.blade_half_width_m],
                       [profile.blade_depth_m, profile.thickness_m, profile.half_width_m], atol=1e-12, rtol=0):
        raise NotImplementedError("ToolSpec blade dimensions differ from the profile")
