"""Analytic geometry controls, separate from live Isaac/runtime acceptance."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.benchmark import HeightFieldSpec
from robotic_cutting_sim.scene_frames import (
    WorkpieceTarget, column_to_usd_row, compose_pose, invert_transform, matrix_pose,
    pose_errors, pose_matrix, rotate_wrench_at_same_point, transform_points,
    usd_camera_to_cv, usd_row_to_column, validate_pose_xyzw, validate_transform,
    wxyz_to_xyzw, xyzw_to_wxyz,
)
from robotic_cutting_sim.scene_validation import (
    edge_sweep_metrics, flattening_checks, material_task_metrics,
    mesh_heightfield_metrics, trajectory_metrics,
)

PROTOCOL = json.loads((Path(__file__).resolve().parents[1]/"configs/isaac_acceptance.json").read_text())


def test_known_column_transform_and_inverse_move_points_correctly():
    matrix = pose_matrix([1, 2, 3, 0, 0, np.sqrt(.5), np.sqrt(.5)])
    np.testing.assert_allclose(transform_points(matrix, [1, 0, 0]), [1, 3, 3], atol=1e-15)
    np.testing.assert_allclose(transform_points(invert_transform(matrix), [1, 3, 3]), [1, 0, 0], atol=1e-15)


def test_target_plane_rotates_with_workpiece_and_signed_distance_is_local():
    target = WorkpieceTarget((.2, -.3, .4, np.sqrt(.5), 0, 0, np.sqrt(.5)))
    np.testing.assert_allclose(target.normal_world, [0, -1, 0], atol=1e-15)
    point = target.points_to_world([.01, .02, .015])
    assert point[1] == pytest.approx(-.315)
    assert target.signed_plane_distance(point) == pytest.approx(0, abs=1e-15)
    assert target.signed_plane_distance(point+[0, -.001, 0]) == pytest.approx(.001)
    assert np.dot(target.plane_world()[:3], point)+target.plane_world()[3] == pytest.approx(0, abs=1e-15)
    np.testing.assert_allclose(target.solver_gravity(), [0, -9.81, 0], atol=1e-14)


def test_downward_hand_and_rxpi_mount_give_world_identity_edge():
    hand = [.3, .1, .5, 1, 0, 0, 0]
    edge = compose_pose(hand, [0, 0, .18, 1, 0, 0, 0])
    np.testing.assert_allclose(edge[:3], [.3, .1, .32], atol=1e-15)
    _, angle = pose_errors([.3, .1, .32, 0, 0, 0, 1], edge)
    assert angle == pytest.approx(0, abs=1e-15)


def test_quaternion_api_order_is_explicit_and_has_known_physical_rotation():
    vendor_wxyz = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    q = wxyz_to_xyzw(vendor_wxyz)
    np.testing.assert_allclose(Rotation.from_quat(q).apply([1, 0, 0]), [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(xyzw_to_wxyz(q), vendor_wxyz)
    assert not np.allclose(Rotation.from_quat(vendor_wxyz).apply([1, 0, 0]), [0, 1, 0])


def test_usd_row_matrix_and_centimeter_units_are_not_confused():
    usd = np.eye(4)
    usd[3, :3] = [120, -30, 50]
    matrix = usd_row_to_column(usd, meters_per_unit=.01)
    np.testing.assert_allclose(transform_points(matrix, [0, 0, 0]), [1.2, -.3, .5])
    np.testing.assert_allclose(column_to_usd_row(matrix, .01), usd)


def test_optical_frame_conversion_targets_forward_not_housing_axis():
    world_usd = pose_matrix([0, 0, 1, 0, 0, 0, 1])
    cv = usd_camera_to_cv(world_usd)
    np.testing.assert_allclose(transform_points(cv, [0, 0, 1]), [0, 0, 0])
    np.testing.assert_allclose(transform_points(cv, [0, 1, 1]), [0, -1, 0])


def test_same_point_wrench_rotation_does_not_invent_a_lever_arm():
    T = pose_matrix([100, 200, 300, 0, 0, np.sqrt(.5), np.sqrt(.5)])
    np.testing.assert_allclose(rotate_wrench_at_same_point(T, [1, 0, 0, 0, 1, 0]),
                               [0, 1, 0, -1, 0, 0], atol=1e-15)


def test_nonidentity_backend_guard_prevents_visual_only_relocation():
    assert WorkpieceTarget().require_backend_frame_support()
    moved = WorkpieceTarget((.1, 0, 0, 0, 0, 0, 1))
    with pytest.raises(NotImplementedError):
        moved.require_backend_frame_support()
    assert moved.require_backend_frame_support(supports_nonidentity=True)


@pytest.mark.parametrize("transform", [np.diag([2., 1, 1, 1]), np.diag([-1., 1, 1, 1]), np.ones((4, 4))])
def test_scaled_reflected_and_nonhomogeneous_frames_rejected(transform):
    with pytest.raises(ValueError):
        validate_transform(transform)


def test_nonunit_quaternion_is_not_silently_normalized():
    with pytest.raises(ValueError):
        validate_pose_xyzw([0, 0, 0, 0, 0, 0, 2])


def test_float32_fk_norm_roundoff_preserves_known_hand_attachment():
    # A measured FK quaternion can have sub-ppm norm error without a different rotation.
    hand = [.3, .1, .5, 1+7e-7, 0, 0, 0]
    edge = compose_pose(hand, [0, 0, .18, 1, 0, 0, 0])
    np.testing.assert_allclose(edge[:3], [.3, .1, .32], atol=1e-15)
    assert pose_errors([.3, .1, .32, 0, 0, 0, 1], edge)[1] == pytest.approx(0, abs=1e-15)
    with pytest.raises(ValueError):
        validate_pose_xyzw([0, 0, 0, 1.00001, 0, 0, 0])


def straight_cut():
    times = np.arange(401)*.02
    x = np.interp(times, [0, 1.6, 5.6, 8], [-.03, -.03, .045, .045])
    poses = np.c_[x, np.zeros(len(times)), np.full(len(times), .0155),
                   np.tile([0, 0, 0, 1], (len(times), 1))]
    return times, poses


def test_correct_plane_crosses_analytic_dome_but_wrong_plane_or_lateral_path_misses():
    times, poses = straight_cut()
    target, spec = WorkpieceTarget(), HeightFieldSpec()
    report = trajectory_metrics(times, poses, poses, target, spec, PROTOCOL)
    assert report["status"] == "PASS"
    assert report["sweep"]["cross_section_sweep_coverage"] == 1
    for axis, value in [(1, .1), (2, .1)]:
        wrong = poses.copy(); wrong[:, axis] = value
        result = edge_sweep_metrics(wrong, target, spec)
        assert not result["actual_edge_intersects_original_material"]
        assert result["cross_section_sweep_coverage"] == 0


def test_translated_rotated_workpiece_preserves_the_same_local_edge_task():
    times, local = straight_cut()
    q = Rotation.from_euler("xyz", [25, -30, 40], degrees=True).as_quat()
    target = WorkpieceTarget(tuple([.3, -.2, .5, *q]))
    world = target.poses_to_world(local)
    report = trajectory_metrics(times, world, world, target, HeightFieldSpec(), PROTOCOL)
    assert report["status"] == "PASS"
    np.testing.assert_allclose(report["local_actual_poses_xyzw"][:, :3], local[:, :3], atol=3e-16)
    assert report["sweep"]["cross_section_sweep_coverage"] == 1


def test_using_hand_or_flipped_edge_pose_fails_geometry_or_orientation():
    times, expected = straight_cut()
    high = expected.copy(); high[:, 2] += .18
    assert trajectory_metrics(times, expected, high, WorkpieceTarget(), HeightFieldSpec(), PROTOCOL)["status"] == "FAIL"
    flipped = expected.copy(); flipped[:, 3:] = [1, 0, 0, 0]
    report = trajectory_metrics(times, expected, flipped, WorkpieceTarget(), HeightFieldSpec(), PROTOCOL)
    assert not report["checks"]["active_max_edge_rotation_error_deg"]


def plane_mesh(height):
    vertices = np.array([[-.06, -.04, height], [.06, -.04, height],
                         [.06, .04, height], [-.06, .04, height]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return vertices, faces


def test_true_triangle_surface_flattening_and_gouge_are_separate_from_a_cut():
    target, spec = WorkpieceTarget(), HeightFieldSpec()
    vertices, faces = plane_mesh(.015)
    perfect = mesh_heightfield_metrics(vertices, faces, target, spec)
    assert perfect["valid_surface_fraction"] == 1
    assert perfect["surface_rms_plane_error_m"] == pytest.approx(0)
    assert flattening_checks(perfect, PROTOCOL)["status"] == "PASS"
    low, _ = plane_mesh(.013)
    gouged = mesh_heightfield_metrics(low, faces, target, spec)
    assert gouged["maximum_gouge_depth_m"] == pytest.approx(.002)
    assert gouged["overcut_volume_m3"] == pytest.approx(.12*.08*.002)
    assert flattening_checks(gouged, PROTOCOL)["status"] == "FAIL"
    initial = np.array([[0, 0, .001], [0, 0, .020]])
    final = np.array([[0, 0, .001], [.1, 0, .020]])
    cutting = material_task_metrics(initial, final, [1, 1], np.array([0, 1]), "analytic test partition", target, spec)
    assert cutting["detached_initial_above_target_mass_fraction"] == 1
    assert cutting["surface_flattening_status"].startswith("NOT_RUN")


def test_missing_mesh_rays_are_not_filtered_to_improve_flatness():
    vertices, faces = plane_mesh(.015)
    incomplete = mesh_heightfield_metrics(vertices, faces[:1], WorkpieceTarget(), HeightFieldSpec())
    assert incomplete["missing_surface_fraction"] > .45
    assert incomplete["maximum_gouge_depth_m"] == .015
    assert flattening_checks(incomplete, PROTOCOL)["status"] == "FAIL"


def test_mesh_measurement_follows_rotated_workpiece_not_world_height():
    vertices, faces = plane_mesh(.015)
    rotation = Rotation.from_euler("xyz", [70, 20, 25], degrees=True).as_quat()
    target = WorkpieceTarget(tuple([.2, -.4, .6, *rotation]))
    measured = mesh_heightfield_metrics(target.points_to_world(vertices), faces, target, HeightFieldSpec())
    assert measured["maximum_substrate_surface_residual_height_m"] < 1e-15
    assert measured["surface_rms_plane_error_m"] < 1e-15
    assert measured["valid_surface_fraction"] == 1


def test_component_provenance_is_required_for_detached_mass():
    initial = np.array([[0, 0, .001], [0, 0, .020]])
    report = material_task_metrics(initial, initial, [1, 1], np.array([0, 1]), "", WorkpieceTarget(), HeightFieldSpec())
    assert report["status"] == "NOT_RUN"
