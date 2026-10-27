import numpy as np

from robotic_cutting_sim.single_view_controls import analytic_control, feature_acceptance


def test_box_occludes_analytic_plane_and_depth_feature_missing_rows_fail():
    camera = {"K": [[10., 0., 5.], [0., 10., 5.], [0., 0., 1.]],
              "width": 10, "height": 10, "T_world_camera": np.eye(4)}
    plane, scene, box = analytic_control(camera, {"slope_x": .1, "slope_y": -.1, "distance_m": 1., "box": True},
        {"box_center_xy_m": [0., 0.], "box_center_z_below_plane_origin_m": .2,
         "box_half_extents_m": [.1, .1, .05]})
    assert box.sum() == 4
    np.testing.assert_allclose(scene[box], .75)
    rgb = np.zeros((10, 10, 3), np.uint8)
    rgb[box] = [200, 10, 10]
    rules = {"rgb_red_minimum_8bit": 10, "rgb_red_to_green_ratio": 1.5,
             "rgb_red_to_blue_ratio": 1.5, "depth_below_analytic_plane_minimum_m": .05}
    result = feature_acceptance(rgb, scene, np.ones((10, 10), bool), plane, box, rules)
    assert result["boundary_error_p95_px"] == 0
    result = feature_acceptance(rgb, scene, np.zeros((10, 10), bool), plane, box, rules)
    assert result["boundary_error_p95_px"] == 10
    assert result["missing_feature_scanlines"] == 4
