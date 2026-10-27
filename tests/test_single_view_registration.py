import numpy as np
import pytest

from robotic_cutting_sim.single_view_registration import analytic_plane_depth, register_native_depth, register_ovrtx_single_view_depth, register_kit_single_view_depth


CAMERA = {"K": [[4., 0, 2.], [0, 4., 1.]], "width": 4, "height": 2, "model": "pinhole"}
CAMERA["K"].append([0., 0., 1.])


def cloud(u, v, geometric_z):
    return [(u-2.)/4*geometric_z, -(v-1.)/4*geometric_z, -geometric_z, 1.]


def test_measured_noise_survives_reprojection_and_geometric_depth_disagreement():
    # One point at an output pixel center; deliberately large measured bias.
    points = np.array([[cloud(1.5, .5, 1.)]], np.float32)
    native = np.array([[1.02]], np.float32)
    result = register_native_depth(points, native, CAMERA)
    assert result["valid_mask"][0, 1]
    assert result["depth_m"][0, 1] == native[0, 0]
    assert result["depth_m"][0, 1] != 1.
    assert result["diagnostics"]["pointcloud_measurement_disagreement_gt_0_1mm"] == 1
    assert not result["diagnostics"]["consistency_filter_applied"]


def test_reprojected_grid_is_not_treated_as_original_rgb_grid():
    points = np.array([[cloud(2.5, 1.5, 1.), cloud(.5, .5, 1.)]], np.float32)
    native = np.array([[1.02, .98]], np.float32)
    result = register_native_depth(points, native, CAMERA)
    assert result["depth_m"][1, 2] == native[0, 0]
    assert result["depth_m"][0, 0] == native[0, 1]
    assert result["source_index"][1, 2] == 0


def test_nearest_measured_z_wins_and_output_has_exact_source_provenance():
    points = np.array([[cloud(1.5, .5, 1.), cloud(1.5, .5, 1.)]], np.float32)
    native = np.array([[1.03, 1.01]], np.float32)
    result = register_native_depth(points, native, CAMERA)
    assert result["depth_m"][0, 1] == native[0, 1]
    assert result["source_index"][0, 1] == 1
    assert result["diagnostics"]["duplicate_or_occluded_samples"] == 1


def test_invalid_alpha_depth_and_behind_camera_samples_stay_invalid():
    points = np.array([[cloud(.5, .5, 1.), cloud(1.5, .5, 1.), cloud(2.5, .5, -1.)]], np.float32)
    points[0, 0, 3] = 0
    native = np.array([[1., 0., 1.]], np.float32)
    result = register_native_depth(points, native, CAMERA)
    assert not result["valid_mask"].any()


def test_huge_native_xyz_is_used_only_as_direction_not_silently_replaced_with_ideal_z():
    points = np.array([[cloud(.5, .5, 1e6)]], np.float32)
    native = np.array([[.8]], np.float32)
    result = register_native_depth(points, native, CAMERA)
    assert result["depth_m"][0, 0] == native[0, 0]
    assert result["diagnostics"]["pointcloud_geometry_depth_outside_native_range"] == 1


def test_slanted_plane_depth_depends_on_x_and_y_ray_direction():
    camera = {**CAMERA, "T_world_camera": np.eye(4).tolist()}
    depth = analytic_plane_depth(camera, [-.25, -.1, 1.], -1.)
    expected = 1 / (1 - .25*(np.arange(4)+.5-2.)/4 - .1*(np.arange(2)[:, None]+.5-1.)/4)
    np.testing.assert_allclose(depth, expected, atol=1e-12)


def test_nonperspective_runtime_model_is_rejected():
    with pytest.raises(ValueError):
        register_native_depth(np.array([[cloud(.5, .5, 1.)]], np.float32), np.ones((1, 1), np.float32),
                              {**CAMERA, "model": "fisheye"})


def test_explicit_source_correspondence_offset_keeps_noisy_native_value():
    points = np.array([[cloud(1.5, .5, 1.)]], np.float32)
    native = np.array([[1.02]], np.float32)
    original = register_native_depth(points, native, CAMERA)
    shifted = register_native_depth(points, native, CAMERA, pointcloud_pixel_offset=(1., 0.))
    assert original["valid_mask"][0, 1] and not shifted["valid_mask"][0, 1]
    assert shifted["depth_m"][0, 2] == native[0, 0]
    assert shifted["source_index"][0, 2] == 0
    assert shifted["diagnostics"]["pointcloud_pixel_offset"] == [1., 0.]


def test_validated_association_profile_rejects_uncontrolled_runtime_or_optics():
    points = np.array([[cloud(1.5, .5, 1.)]], np.float32)
    native = np.array([[1.02]], np.float32)
    with pytest.raises(ValueError, match="renderer version"):
        register_ovrtx_single_view_depth(points, native, CAMERA,
            ovrtx_version="other", ovstage_version="0.3.0.1791986420", native_model_parameters={})
    with pytest.raises(ValueError, match="resolution"):
        register_ovrtx_single_view_depth(points, native, CAMERA,
            ovrtx_version="0.6.0.1791947691", ovstage_version="0.3.0.1791986420", native_model_parameters={})
    camera = {"width": 640, "height": 480, "K": [[317.0431137084961, 0., 320.],
        [0., 317.0431137084961, 240.], [0., 0., 1.]]}
    point = np.array([[[0., 0., -1., 1.]]], np.float32)
    parameters = {"enabled": True, "baselineMM": 95., "focalLengthPixel": 317.04310808172426,
        "sensorSizePixel": 640., "maxDisparityPixel": 150., "confidenceThreshold": 1.,
        "noiseDownscaleFactorPixel": 1., "minDistance": .52, "maxDistance": 6.,
        "outlierRemovalEnabled": True, "rgbDepthOutputMode": 0, "effectiveDistance": 4.,
        "noiseMean": .25, "noiseSigma": .25}
    result = register_ovrtx_single_view_depth(point, native, camera,
        ovrtx_version="0.6.0.1791947691", ovstage_version="0.3.0.1791986420", native_model_parameters=parameters)
    assert result["depth_m"][240, 321] == native[0, 0]
    assert result["diagnostics"]["source_correspondence_validated"]
    assert not result["diagnostics"]["noise_calibration_validated"]
    with pytest.raises(ValueError, match="depth-model parameters"):
        register_ovrtx_single_view_depth(point, native, camera,
            ovrtx_version="0.6.0.1791947691", ovstage_version="0.3.0.1791986420",
            native_model_parameters={**parameters, "baselineMM": 55.})
    kit = register_kit_single_view_depth(point, native, camera, isaacsim_version="6.2.0.0",
        kit_version="110.4.0+feature.386251.187cf21b.gl", camera_extension_version="1.4.0",
        native_model_parameters=parameters)
    assert kit["depth_m"][240, 321] == native[0, 0]
    assert kit["diagnostics"]["source_correspondence_validated"]
    assert not kit["diagnostics"]["live_dynamics_sensor_synchronization_validated"]
    with pytest.raises(ValueError, match="Kit runtime"):
        register_kit_single_view_depth(point, native, camera, isaacsim_version="6.2.0.0",
            kit_version="new_version", camera_extension_version="1.4.0", native_model_parameters=parameters)
