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


@dataclass(frozen=True)
class EndHeldToolProfile(ToolProfile):
    """tool_profile_v2 (ADR 0013): a slim grip at one END of the blade.

    The handle box is centred at the finger pads (HAND_FROM_TOOL). The blade
    starts at the handle's far +z face and runs ``edge_length_m`` further along
    hand +z, so the edge line is parallel to the approach axis. The blade width
    (``blade_depth_m``) spans from a spine flush with one handle face to the
    edge, across hand x (arrangement "y") or across the finger closing axis
    hand y (arrangement "x"). ``blade_normal_hand_axis`` names the hand axis
    along the blade thickness. The blade cross-section is the same polygon as
    any ToolProfile with the same blade numbers.

    Edge frame in hand coordinates: +y (the edge line) = hand +z, pointing from
    the grip toward the tip; +x (advance, away from the blade body) = hand +x
    ("y") or hand +y ("x"); z = x × y, i.e. -hand y ("y") or +hand x ("x").
    """

    handle_half_extents_hand_m: tuple = (.012, .011, .035)
    blade_normal_hand_axis: str = "y"

    def __post_init__(self):
        super().__post_init__()
        if self.bevel == "legacy_wedge" or self.blade_normal_hand_axis not in ("x", "y"):
            raise ValueError("An end-held profile needs a real bevel and blade_normal_hand_axis 'x' or 'y'")
        half = np.asarray(self.handle_half_extents_hand_m, float)
        spine_face, normal_face = (half[0], half[1]) if self.blade_normal_hand_axis == "y" else (half[1], half[0])
        if self.blade_depth_m <= 2*spine_face or self.thickness_m/2 > normal_face:
            raise ValueError("The blade must overhang the handle width and stay within its thickness")

    def _hand_axes(self):
        """(advance axis, edge-line axis) as hand unit vectors."""
        advance = [1., 0., 0.] if self.blade_normal_hand_axis == "y" else [0., 1., 0.]
        return np.asarray(advance), np.array([0., 0., 1.])

    def hand_from_edge(self):
        """Edge centre half an edge length beyond the handle's far face, on the edge line.

        The spine is flush with the handle's -advance face, so the edge lies
        ``blade_depth_m`` from it along the advance axis.
        """
        advance, along = self._hand_axes()
        half = np.asarray(self.handle_half_extents_hand_m, float)
        spine = half[0] if self.blade_normal_hand_axis == "y" else half[1]
        R = np.column_stack([advance, along, np.cross(advance, along)])
        position = (np.asarray(HAND_FROM_TOOL[:3])+(self.blade_depth_m-spine)*advance
                    + (half[2]+self.half_width_m)*along)
        return (*map(float, position), *Rotation.from_matrix(R).as_quat())

    def to_dict(self):
        return {**super().to_dict(), "version": "tool_profile_v2", "arrangement": self.arrangement,
                "grip": "end_held", "edge_line_hand_axis": "+z",
                "blade_normal_hand_axis": self.blade_normal_hand_axis,
                "finger_contact_faces": "handle +/-y faces (hand finger closing axis)"}

    @property
    def arrangement(self):
        return "A_blade_normal_hand_y" if self.blade_normal_hand_axis == "y" else "B_blade_normal_hand_x"


RULER_KNIFE_V2_Y = EndHeldToolProfile("ruler_knife_v2_y", edge_length_m=.120, blade_depth_m=.040, thickness_m=.0015,
                                      bevel="double", bevel_included_angle_deg=25., blade_normal_hand_axis="y")
RULER_KNIFE_V2_X = EndHeldToolProfile("ruler_knife_v2_x", edge_length_m=.120, blade_depth_m=.040, thickness_m=.0015,
                                      bevel="double", bevel_included_angle_deg=25., blade_normal_hand_axis="x")
PROFILES = {RULER_KNIFE_V1.profile_id: RULER_KNIFE_V1, RULER_KNIFE_V2_Y.profile_id: RULER_KNIFE_V2_Y,
            RULER_KNIFE_V2_X.profile_id: RULER_KNIFE_V2_X}


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


def handle_box_edge(profile):
    """Handle as an edge-frame (center, half extents) box for blunt material contact."""
    spec = make_profile_tool_spec(profile)
    corners = transform_points(invert_transform(spec.T_tool_edge), spec.shapes[0].vertices_tool_m)
    low, high = corners.min(0), corners.max(0)
    return (high+low)/2, (high-low)/2


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


def retarget_task_tool(task, profile_id, hand_offset_world_m=(0., 0., 0.)):
    """A v2 task carrying another profile tool on the SAME hand trajectory.

    Every scripted hand pose is kept (optionally translated by a world offset)
    and the edge knots/orientation are re-derived through the new profile's
    mount, so IK, colliders and TLMPM stay on one profile. Opt-in: used for
    rigid grasp controls and presentation views; the result is not a new
    cutting task and keeps the base task's thresholds and id.
    """
    from dataclasses import replace
    from .tasks import DemoTaskSpec, ProfileTaskSpec
    if not isinstance(task, ProfileTaskSpec) or isinstance(task, DemoTaskSpec):
        raise NotImplementedError("Tool retargeting is defined only for v2 ProfileTaskSpec tasks")
    profile = PROFILES[profile_id]
    offset = np.asarray(hand_offset_world_m, float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Hand offset must be one finite world translation")
    old_mount, new_mount = pose_matrix(task.hand_from_edge_xyz_xyzw), pose_matrix(profile.hand_from_edge())
    T_local_world = invert_transform(task.T_world_workpiece)
    offset_local = T_local_world[:3, :3]@offset
    R_old = Rotation.from_quat(task.edge_orientation_local_xyzw).as_matrix()
    knots, orientation = [], None
    for t, *position in task.edge_position_knots_local_txyz:
        T_edge = np.eye(4)
        T_edge[:3, :3], T_edge[:3, 3] = R_old, position
        T_hand = T_edge@invert_transform(old_mount)
        T_hand[:3, 3] += offset_local
        T_new = T_hand@new_mount
        knots.append((float(t), *map(float, T_new[:3, 3])))
        orientation = tuple(map(float, Rotation.from_matrix(T_new[:3, :3]).as_quat()))
    return replace(task, tool_profile_id=profile.profile_id, hand_from_edge_xyz_xyzw=tuple(profile.hand_from_edge()),
                   blade_length_m=profile.blade_depth_m, blade_height_m=profile.thickness_m,
                   blade_half_width_m=profile.half_width_m, edge_orientation_local_xyzw=orientation,
                   edge_position_knots_local_txyz=tuple(knots))


def tool_spec_for_task(task):
    """The physical tool a runtime authors for a task.

    v2 (profile) tasks get their profile tool; v1 tasks keep make_tool_spec(task)
    exactly, so their colliders, mass and receipts are unchanged.
    """
    from .physical_tool import make_tool_spec
    from .tasks import ProfileTaskSpec
    if isinstance(task, ProfileTaskSpec):
        return make_profile_tool_spec(task.tool_profile)
    return make_tool_spec(task)


def require_task_tool(spec, task):
    """Reject a tool whose blade or mount differs from what the task's TLMPM and IK use."""
    from .tasks import ProfileTaskSpec
    if isinstance(task, ProfileTaskSpec):
        require_profile_consistency(spec, task.tool_profile)
        if not np.allclose(spec.T_hand_tool@spec.T_tool_edge, pose_matrix(task.hand_from_edge_xyz_xyzw), atol=1e-10, rtol=0):
            raise NotImplementedError("Tool hand-to-edge frame differs from the task's IK mount")
        return
    from .tool_compatibility import require_matching_blade
    require_matching_blade(spec, task.hand_from_edge_xyz_xyzw,
                           [task.blade_length_m, task.blade_height_m, task.blade_half_width_m])

