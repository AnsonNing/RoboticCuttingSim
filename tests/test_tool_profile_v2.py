"""ruler_knife_v2 (ADR 0013): end-held blade profiles, v1 pinned unchanged."""
import hashlib
import json

import numpy as np
import pytest

from robotic_cutting_sim.physical_tool import ToolGraspConfig
from robotic_cutting_sim.scene_frames import invert_transform, pose_matrix, transform_points
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.tool_profile import (HAND_FROM_TOOL, PROFILES, RULER_KNIFE_V1, RULER_KNIFE_V2_X,
                                              RULER_KNIFE_V2_Y, EndHeldToolProfile, ToolProfile, handle_box_edge,
                                              make_profile_tool_spec, require_profile_consistency,
                                              require_task_tool, retarget_task_tool)

V2 = (RULER_KNIFE_V2_Y, RULER_KNIFE_V2_X)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_v1_profile_outputs_are_pinned_bitwise():
    # Digests recorded at 9bd8a29, before tool_profile_v2 was added.
    p = RULER_KNIFE_V1
    assert type(p) is ToolProfile and PROFILES["ruler_knife_v1"] is p
    assert _sha(json.dumps(p.to_dict(), sort_keys=True).encode()) == \
        "d1c280791de17718b5755d1c1fed21c96d0d2a771a0dc78c302ab1dc6398eed6"
    spec = make_profile_tool_spec(p)
    arrays = b"".join(np.ascontiguousarray(a).tobytes() for shape in spec.shapes
                      for a in (shape.vertices_tool_m, shape.triangles))
    assert _sha(arrays) == "808967427095beeced71278b05298f024ba9304540ac7800952e9648cf09c213"
    assert _sha(json.dumps(spec.to_dict(), sort_keys=True).encode()) == \
        "85e7241b7cdb99c3ba79b19fb204d827ce5ffb677d3ca163228f79fd137e80f3"
    assert p.hand_from_edge() == (0.0, 0.0, 0.16365000000000002, .5, .5, .5, -.5)
    polygon, half_width = p.tlmpm_profile()
    assert _sha(polygon.tobytes()+half_width.tobytes()) == \
        "8205bb4792bb767a5dbd4d446ce60d9804e014ac470fa2bc27a3faad26927fbd"
    center, half = handle_box_edge(p)
    assert _sha(np.r_[center, half].tobytes()) == "634f1104e007693e97a84514bd8c19a0516df43f7af6c238074cc98a3878cf19"
    assert p.to_dict()["version"] == "tool_profile_v1" and "arrangement" not in p.to_dict()


@pytest.mark.parametrize("profile", V2, ids=lambda p: p.profile_id)
def test_v2_blade_is_v1_blade(profile):
    assert PROFILES[profile.profile_id] is profile
    for name in ("edge_length_m", "blade_depth_m", "thickness_m", "bevel", "bevel_included_angle_deg",
                 "blade_density_kg_m3"):
        assert getattr(profile, name) == getattr(RULER_KNIFE_V1, name)
    a, b = profile.tlmpm_profile(), RULER_KNIFE_V1.tlmpm_profile()
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    record = profile.to_dict()
    assert record["version"] == "tool_profile_v2" and record["cross_section_xz_m"] == RULER_KNIFE_V1.to_dict()["cross_section_xz_m"]
    assert record["arrangement"] == ("A_blade_normal_hand_y" if profile.blade_normal_hand_axis == "y" else "B_blade_normal_hand_x")
    assert record["handle_half_extents_hand_m"] == (.012, .011, .035)


@pytest.mark.parametrize("profile", V2, ids=lambda p: p.profile_id)
def test_v2_single_source_consistency(profile):
    spec = make_profile_tool_spec(profile)
    require_profile_consistency(spec, profile)
    T_hand_edge = pose_matrix(profile.hand_from_edge())
    assert np.allclose(spec.T_hand_tool@spec.T_tool_edge, T_hand_edge, atol=1e-12, rtol=0)
    assert spec.hand_from_tool_xyz_xyzw == HAND_FROM_TOOL
    # The visual mesh is the collider: blade vertices are exactly the profile prism.
    blade = [s for s in spec.shapes if s.role == "blade"][0]
    expected = transform_points(spec.T_tool_edge, profile.blade_mesh_edge()[0])
    assert np.abs(np.sort(blade.vertices_tool_m, 0)-np.sort(expected, 0)).max() < 1e-12
    # TLMPM polygon is the collider cross-section in the edge frame.
    in_edge = transform_points(invert_transform(spec.T_tool_edge), blade.vertices_tool_m)
    polygon, half_width = profile.tlmpm_profile()
    assert np.allclose(np.unique(np.round(in_edge[:, [0, 2]], 12), axis=0),
                       np.unique(np.round(polygon.astype(float), 12), axis=0), atol=1e-7)
    assert np.allclose(np.abs(in_edge[:, 1]), half_width)
    # Blunt handle obstacle box is the handle collider in the edge frame.
    handle = [s for s in spec.shapes if s.role == "handle"][0]
    corners = transform_points(invert_transform(spec.T_tool_edge), handle.vertices_tool_m)
    center, half = handle_box_edge(profile)
    assert np.allclose(center, (corners.max(0)+corners.min(0))/2) and np.allclose(half, (corners.max(0)-corners.min(0))/2)
    assert np.allclose(np.sort(half), np.sort(profile.handle_half_extents_hand_m))
    # Disjoint interiors: the blade lies beyond the handle's far +z face (they share only that face).
    assert blade.vertices_tool_m[:, 2].min() == pytest.approx(profile.handle_half_extents_hand_m[2], abs=1e-12)
    assert handle.vertices_tool_m[:, 2].max() == pytest.approx(profile.handle_half_extents_hand_m[2], abs=1e-12)
    # Tampered tools are rejected.
    other = EndHeldToolProfile("other", .121, .04, .0015, blade_normal_hand_axis=profile.blade_normal_hand_axis)
    with pytest.raises(NotImplementedError):
        require_profile_consistency(make_profile_tool_spec(other), profile)


@pytest.mark.parametrize("profile", V2, ids=lambda p: p.profile_id)
def test_v2_handle_at_blade_end_and_edge_parallel_to_hand_z(profile):
    T = pose_matrix(profile.hand_from_edge())
    assert np.allclose(T[:3, 1], [0, 0, 1], atol=1e-12)  # edge line = hand +z (approach axis)
    normal_axis = {"x": 0, "y": 1}[profile.blade_normal_hand_axis]
    assert np.isclose(abs(T[normal_axis, 2]), 1.)  # blade normal = declared hand axis
    advance_axis = 1-normal_axis
    assert np.allclose(T[:3, 0], np.eye(3)[advance_axis])
    center, half = handle_box_edge(profile)
    # No handle volume within +/- edge_length/2 of the edge centre along the edge line.
    assert center[1]+half[1] <= -profile.half_width_m+1e-12
    # Spine flush with the handle's -advance face; edge blade_depth beyond it.
    spec = make_profile_tool_spec(profile)
    blade = spec.shapes[1].vertices_tool_m
    handle_half = profile.handle_half_extents_hand_m[advance_axis]
    assert blade[:, advance_axis].min() == pytest.approx(-handle_half, abs=1e-12)
    assert blade[:, advance_axis].max() == pytest.approx(profile.blade_depth_m-handle_half, abs=1e-12)
    assert np.isclose(np.ptp(blade[:, normal_axis]), profile.thickness_m)
    assert np.isclose(np.ptp(blade[:, 2]), profile.edge_length_m)
    assert np.isclose(T[2, 3], HAND_FROM_TOOL[2]+profile.handle_half_extents_hand_m[2]+profile.half_width_m)


@pytest.mark.parametrize("profile", V2, ids=lambda p: p.profile_id)
def test_v2_mass_properties(profile):
    spec = make_profile_tool_spec(profile)
    handle, blade = spec.shapes
    hx, hy, hz = profile.handle_half_extents_hand_m
    assert np.isclose(handle.mass_properties[1], 8*hx*hy*hz*profile.handle_density_kg_m3, rtol=1e-12)
    blade_mass = profile.edge_length_m*profile.thickness_m*(profile.blade_depth_m-profile.bevel_length_m/2)*profile.blade_density_kg_m3
    assert np.isclose(blade.mass_properties[1], blade_mass, rtol=1e-9)
    assert .09 < spec.mass_kg < .10  # about 41 g polymer grip + 54 g steel blade
    inertia = spec.inertia_com_tool_kg_m2
    assert np.allclose(inertia, inertia.T) and np.linalg.eigvalsh(inertia).min() > 0
    moments = np.sort(spec.principal_moments_kg_m2)
    assert moments[0]+moments[1] >= moments[2]-1e-15  # triangle inequality of a real body
    com = spec.com_tool_m
    handle_com, blade_com = handle.mass_properties[2], blade.mass_properties[2]
    assert handle_com[2] < com[2] < blade_com[2]  # COM between handle and blade, along the edge line
    assert np.allclose(com, (handle.mass_properties[1]*handle_com+blade.mass_properties[1]*blade_com)/spec.mass_kg)


@pytest.mark.parametrize("profile", V2, ids=lambda p: p.profile_id)
def test_v2_finger_pads_contact_the_handle_y_faces(profile):
    """Both arrangements are squeezed across hand y by the unchanged grasp config."""
    spec = make_profile_tool_spec(profile)
    handle, blade = spec.shapes
    grasp = ToolGraspConfig()
    half_y = profile.handle_half_extents_hand_m[1]
    assert half_y == RULER_KNIFE_V1.handle_half_extents_hand_m[1]
    assert grasp.closed_target_m < half_y < grasp.initial_finger_position_m  # opens wider, closes past the faces
    # Finger pads: 17.5 x 16.5 mm patches centred at HAND_FROM_TOOL (actual USD pad cubes).
    pad_half_x, pad_half_z = .00875, .00825
    for side in (-1., 1.):
        plane = handle.vertices_tool_m[np.isclose(handle.vertices_tool_m[:, 1], side*half_y)]
        assert len(plane) == 4  # a full rectangular handle face at y = +/- half width
        assert plane[:, 0].min() <= -pad_half_x and plane[:, 0].max() >= pad_half_x
        assert plane[:, 2].min() <= -pad_half_z and plane[:, 2].max() >= pad_half_z
    # Nothing of the blade lies in the pad region (|z| <= pad half length) beyond the handle faces.
    assert blade.vertices_tool_m[:, 2].min() > pad_half_z
    # Pad closing direction squeezes the handle, never the blade.
    assert np.abs(handle.vertices_tool_m[:, 1]).max() == pytest.approx(half_y)


def test_v2_rejects_bad_parameters():
    with pytest.raises(ValueError):
        EndHeldToolProfile("bad", .12, .04, .0015, blade_normal_hand_axis="z")
    with pytest.raises(ValueError):
        EndHeldToolProfile("bad", .12, .02, .0015)  # blade narrower than the handle
    with pytest.raises(ValueError):
        EndHeldToolProfile("bad", .12, .04, .0015, bevel="legacy_wedge")


@pytest.mark.parametrize("profile", V2, ids=lambda p: p.profile_id)
def test_retarget_task_keeps_hand_trajectory(profile):
    base = make_task("CUT-01-R")
    lift = np.array([0., 0., .12])
    task = retarget_task_tool(base, profile.profile_id, lift)
    assert task.tool_profile is profile and task.task_id == base.task_id
    assert task.tool_profile.to_dict()["version"] == "tool_profile_v2"
    for t in (0., .5, 1., 1.6):
        a, b = pose_matrix(base.reference_hand_pose_world(t)), pose_matrix(task.reference_hand_pose_world(t))
        assert np.allclose(b[:3, :3], a[:3, :3], atol=1e-12) and np.allclose(b[:3, 3], a[:3, 3]+lift, atol=1e-12)
    # Hand-down start: the blade hangs below the fingers, above the table.
    hand = pose_matrix(task.reference_hand_pose_world(0.))
    assert np.allclose(hand[:3, 2], [0, 0, -1], atol=1e-9)
    spec = make_profile_tool_spec(profile)
    require_task_tool(spec, task)
    tip = transform_points(hand@spec.T_hand_tool, spec.shapes[1].vertices_tool_m)[:, 2].min()
    assert tip > base.table_top_world_z_m+.05
    same = retarget_task_tool(base, "ruler_knife_v1")
    assert np.allclose(same.edge_position_knots_local_txyz, base.edge_position_knots_local_txyz, atol=1e-12)
    assert np.allclose(np.abs(np.dot(same.edge_orientation_local_xyzw, base.edge_orientation_local_xyzw)), 1., atol=1e-12)
    with pytest.raises(NotImplementedError):
        retarget_task_tool(make_task("CUT-01"), profile.profile_id)


def test_native_config_tool_override_is_opt_in_and_guarded():
    from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig
    assert IsaacSimCuttingConfig().tool_profile_id is None
    assert IsaacSimCuttingConfig().tool_profile_hand_offset_world_m == (0., 0., 0.)
    common = dict(tool_attachment_mode="physical_grasp", camera_layout="wrist_d405_global_d455",
                  camera_mode="native_single_view")
    IsaacSimCuttingConfig(task_id="CUT-01-R", tool_profile_id="ruler_knife_v2_y",
                          tool_profile_hand_offset_world_m=(0., 0., .12), **common)
    for bad in (dict(task_id="CUT-01", tool_profile_id="ruler_knife_v2_y"),
                dict(task_id="CUT-01-R", tool_profile_id="unknown", **common),
                dict(task_id="CUT-01-R", tool_profile_hand_offset_world_m=(0., 0., .1), **common)):
        with pytest.raises(ValueError):
            IsaacSimCuttingConfig(**bad)
