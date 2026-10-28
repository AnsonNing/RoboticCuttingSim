"""CPU geometry/frame/inertia tests do not establish a runtime physical grasp."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.physical_tool import (
    ToolGraspConfig, ToolShape, edge_twist_from_com, edge_wrench_to_com,
    grasp_relative_state, make_tool_spec, measured_edge_pose, nominal_hand_pose,
    shift_wrench_world, world_com_position,
)
from robotic_cutting_sim.scene_frames import compose_pose, pose_errors, pose_matrix, transform_points
from robotic_cutting_sim.tasks import make_task


def intersection_inradius(first, second):
    # Independent convex half-space feasibility: positive r means volume overlap,
    # zero means contact, negative r means separated. No sampling can miss overlap.
    equations = np.r_[ConvexHull(first.vertices_tool_m).equations,
                       ConvexHull(second.vertices_tool_m).equations]
    result = linprog([0., 0., 0., -1.], A_ub=np.c_[equations[:, :3], np.ones(len(equations))],
                     b_ub=-equations[:, 3], bounds=[(None, None)] * 4, method="highs")
    assert result.success
    return result.x[3]


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
def test_connected_compound_has_shared_faces_without_double_counted_volume(task_id):
    spec = make_tool_spec(make_task(task_id))
    handle, shaft, blade = spec.shapes
    assert [shape.role for shape in spec.shapes] == ["handle", "shaft", "blade"]
    assert intersection_inradius(handle, shaft) == pytest.approx(0., abs=2e-9)
    assert intersection_inradius(shaft, blade) == pytest.approx(0., abs=2e-9)
    assert intersection_inradius(handle, blade) < -.005
    assert all(shape.mass_properties[0] > 0 for shape in spec.shapes)
    assert .025 < spec.mass_kg < .100  # Finite provisional handheld-tool scale.
    assert np.all(spec.principal_moments_kg_m2 > 0)
    assert spec.principal_moments_kg_m2[-1] <= sum(spec.principal_moments_kg_m2[:2])
    assert json.loads(json.dumps(spec.to_dict()))["mass_kg"] == spec.mass_kg


def test_handle_grip_region_matches_official_pad_center_and_finite_finger_drive():
    spec, grip = make_tool_spec(), ToolGraspConfig()
    np.testing.assert_allclose(spec.hand_from_tool_xyz_xyzw[:3], [0, 0, .0584 + .04525])
    assert spec.handle_half_extents_m == (.008, .011, .020)
    q_contact = spec.handle_half_extents_m[1]
    assert grip.initial_finger_position_m > q_contact > grip.closed_target_m
    force = min(grip.max_finger_force_n, grip.stiffness_n_m * (q_contact - grip.closed_target_m))
    assert force == pytest.approx(18.)
    assert 0 < spec.mass_kg * 9.81 < 2 * force * grip.friction
    assert grip.max_finger_force_n == 20 and grip.friction == .6
    assert (grip.settle_s, grip.stability_window_s, grip.min_bilateral_contact_fraction) == (2., .5, .95)
    assert grip.to_dict()["status"] == "PROVISIONAL_ENGINEERING_SETTINGS"


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
def test_closed_mesh_mass_inertia_match_independent_box_and_triangular_prism_integrals(task_id):
    spec = make_tool_spec(make_task(task_id))
    handle, _, blade = spec.shapes
    half = np.array(spec.handle_half_extents_m)
    mass = 1100 * np.prod(2 * half)
    np.testing.assert_allclose(handle.mass_properties[1], mass, rtol=1e-12)
    np.testing.assert_allclose(handle.mass_properties[2], np.zeros(3), atol=1e-18)
    expected_handle_inertia = mass / 3 * np.diag([half[1]**2 + half[2]**2,
                                                half[0]**2 + half[2]**2,
                                                half[0]**2 + half[1]**2])
    np.testing.assert_allclose(handle.mass_properties[3], expected_handle_inertia, atol=1e-18)
    length, height, width = spec.blade_length_m, spec.blade_height_m, spec.blade_half_width_m
    triangle = np.array([[0., 0., 0.], [-length, 0., height], [-length, 0., -.0004]])
    mean = triangle.mean(0)
    centered = triangle - mean
    covariance = centered.T @ centered / 12
    covariance[1, 1] = width**2 / 3
    blade_mass = 7800 * length * (height + .0004) / 2 * (2 * width)
    inertia_edge = blade_mass * (np.trace(covariance) * np.eye(3) - covariance)
    R = spec.T_tool_edge[:3, :3]
    np.testing.assert_allclose(blade.mass_properties[1], blade_mass, rtol=1e-7)
    np.testing.assert_allclose(blade.mass_properties[2], transform_points(spec.T_tool_edge, mean), atol=1e-9)
    np.testing.assert_allclose(blade.mass_properties[3], R @ inertia_edge @ R.T, rtol=3e-7, atol=1e-13)
    principal_R = Rotation.from_quat(spec.principal_axes_xyzw).as_matrix()
    np.testing.assert_allclose(principal_R @ np.diag(spec.principal_moments_kg_m2) @ principal_R.T,
                               spec.inertia_com_tool_kg_m2, atol=1e-17)


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
def test_nominal_mount_roundtrip_and_actual_slip_are_distinct(task_id):
    task, spec = make_task(task_id), make_tool_spec(make_task(task_id))
    hand = task.reference_hand_pose_world(3.4)
    nominal_tool = compose_pose(hand, spec.hand_from_tool_xyz_xyzw)
    actual_edge = measured_edge_pose(nominal_tool, spec)
    assert max(pose_errors(actual_edge, task.reference_edge_pose_world(3.4))) < 1e-12
    assert max(pose_errors(nominal_hand_pose(actual_edge, spec), hand)) < 1e-12
    slipped = nominal_tool.copy()
    slipped[:3] += [.004, -.003, .002]
    slipped[3:] = (Rotation.from_euler("xyz", [.07, -.1, .04]) * Rotation.from_quat(slipped[3:])).as_quat()
    before = slipped.copy()
    drift = grasp_relative_state(hand, slipped, spec)
    assert drift["nominal_translation_error_m"] == pytest.approx(np.linalg.norm([.004, -.003, .002]))
    assert drift["nominal_rotation_error_rad"] > .1
    assert not drift["physical_grasp_validated"]
    assert max(pose_errors(measured_edge_pose(slipped, spec), actual_edge)) > .003
    np.testing.assert_array_equal(slipped, before)  # Helpers do not overwrite the body pose.


def test_one_tool_interface_supports_randomized_objects_and_arbitrary_edge_rotation():
    for name in ["CUT-01", "TRIM-01"]:
        first, other = make_tool_spec(make_task(name, variant="randomized", seed=1)), make_tool_spec(make_task(name, variant="randomized", seed=12))
        for shape, corresponding in zip(first.shapes, other.shapes):
            np.testing.assert_array_equal(shape.vertices_tool_m, corresponding.vertices_tool_m)
        np.testing.assert_allclose(first.T_tool_edge, other.T_tool_edge)
    rotation = Rotation.from_euler("xyz", [25, 30, 40], degrees=True).as_quat()
    custom = make_tool_spec(hand_from_edge_xyz_xyzw=(0., 0., .18, *rotation))
    assert custom.source_task_id is None
    np.testing.assert_allclose(custom.T_hand_tool @ custom.T_tool_edge,
                               pose_matrix((0., 0., .18, *rotation)), atol=1e-15)
    assert intersection_inradius(custom.shapes[1], custom.shapes[2]) == pytest.approx(0., abs=2e-9)
    sideways = make_tool_spec(hand_from_edge_xyz_xyzw=(0., 0., .18,
                               *Rotation.from_euler("x", 90, degrees=True).as_quat()))
    assert intersection_inradius(sideways.shapes[1], sideways.shapes[2]) == pytest.approx(0., abs=2e-9)


def test_edge_wrench_com_shift_sign_and_virtual_power_under_world_rotation():
    spec = make_tool_spec(make_task("CUT-01"))
    rng = np.random.default_rng(42)
    for _ in range(20):
        world_tool = np.r_[rng.normal(size=3), Rotation.random(random_state=rng).as_quat()]
        wrench = rng.normal(size=6)
        linear, angular = rng.normal(size=3), rng.normal(size=3)
        shifted = edge_wrench_to_com(wrench, world_tool, spec)
        edge_twist = edge_twist_from_com(linear, angular, world_tool, spec)
        assert shifted @ np.r_[linear, angular] == pytest.approx(wrench @ edge_twist, abs=1e-13)
        restored = shift_wrench_world(shifted, world_com_position(world_tool, spec),
                                      measured_edge_pose(world_tool, spec)[:3])
        np.testing.assert_allclose(restored, wrench, atol=1e-14)
    shifted = shift_wrench_world([0., 2., 0., 0., 0., 0.], [.1, 0., 0.], [0., 0., 0.])
    np.testing.assert_allclose(shifted, [0., 2., 0., 0., 0., .2])
    dt = .02
    assert (shifted * dt)[5] == pytest.approx(.004)  # N·m·s, not a force or frame translation.


@pytest.mark.parametrize("kwargs", [{"friction": -.1}, {"max_finger_force_n": np.inf},
                                     {"closed_target_m": .03}, {"stiffness_n_m": 0},
                                     {"stability_window_s": 3.}, {"min_bilateral_contact_fraction": 1.1}])
def test_invalid_grip_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        ToolGraspConfig(**kwargs)


def test_invalid_mount_or_open_mesh_is_rejected_and_low_grip_remains_configurable():
    with pytest.raises(ValueError):
        make_tool_spec(hand_from_edge_xyz_xyzw=(0., 0., .18, 1., 0., 0., 1.))
    with pytest.raises(ValueError):
        make_tool_spec(hand_from_edge_xyz_xyzw=(0., 0., .11, 1., 0., 0., 0.))
    with pytest.raises(ValueError):
        make_tool_spec(blade_height_m=np.nan)
    handle = make_tool_spec().shapes[0]
    with pytest.raises(ValueError):
        ToolShape("open", handle.vertices_tool_m, handle.triangles[:-1], 1100., "handle")
    low = replace(ToolGraspConfig(), friction=.02, max_finger_force_n=.5)
    assert low.max_finger_force_n == .5 and low.friction == .02
