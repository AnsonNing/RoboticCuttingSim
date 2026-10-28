import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.scene_frames import pose_matrix
from robotic_cutting_sim.wrist_camera_audit import (depth_comparison, effective_fov_deg,
    obb_signed_projection_gap, project_points, trace_metrics)


def synthetic_trace():
    time = np.arange(4)*.02
    hands = np.zeros((4, 2, 7))
    hands[:, :, 6] = 1
    hands[:, 0, 0] = time
    hands[:, 1, :3] = [.2, .1, .3]
    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_euler("x", .4).as_matrix()
    mount[:3, 3] = [.01, .02, .03]
    specs = {name: {"parent_hand_index": i if i < 2 else None,
                   "T_hand_optical_cv": mount.tolist()} for i, name in enumerate(["wrist_d405_0", "wrist_d405_1", "global_d455"])}
    trace = []
    for index, stamp in enumerate(time):
        row = {"timestamp_s": float(stamp), "cameras": {}}
        for name, spec in specs.items():
            arm = spec["parent_hand_index"]
            hand = pose_matrix(hands[index, arm]) if arm is not None else None
            world = hand@mount if arm is not None else np.eye(4)
            row["cameras"][name] = {"T_world_hand": None if hand is None else hand.tolist(),
                "T_hand_camera": mount.tolist() if hand is not None else None,
                "T_world_camera": world.tolist(), "T_camera_world": np.linalg.inv(world).tolist()}
        trace.append(row)
    protocol = json.loads((Path(__file__).parents[1]/"configs/wrist_camera_audit_v1.json").read_text())
    return trace, time, hands, specs, protocol


def test_trace_checks_actual_hand_fk_and_detects_stale_camera_and_bad_inverse():
    args = synthetic_trace()
    assert trace_metrics(*args)["status"] == "PASS"
    bad = copy.deepcopy(args[0])
    bad[2]["cameras"]["wrist_d405_0"]["T_world_camera"] = bad[1]["cameras"]["wrist_d405_0"]["T_world_camera"]
    result = trace_metrics(bad, *args[1:])
    assert not result["checks"]["wrist_d405_0_actual_fk"]
    assert not result["checks"]["wrist_d405_0_inverse"]


def test_global_drift_and_missing_control_step_are_not_hidden_by_good_other_cameras():
    trace, time, hands, specs, protocol = synthetic_trace()
    trace[-1]["cameras"]["global_d455"]["T_world_camera"][0][3] += .001
    result = trace_metrics(trace, time, hands, specs, protocol)
    assert not result["checks"]["global_d455_actual_fk"]
    trace, time, hands, specs, protocol = synthetic_trace()
    assert not trace_metrics(trace[::2]+trace[-1:], time, hands, specs, protocol)["checks"]["every_control_interval_sampled"]


def test_actual_resolution_and_nonsquare_intrinsics_determine_fov():
    fx = 640/(2*np.tan(np.deg2rad(87)/2))
    wrong = {"width": 640, "height": 480, "K": [[fx, 0, 320], [0, fx, 240], [0, 0, 1]]}
    assert effective_fov_deg(wrong)["vertical_deg"] > 70
    correct = {"width": 848, "height": 480, "K": [[fx*848/640, 0, 424], [0, fx*848/640, 240], [0, 0, 1]]}
    actual = effective_fov_deg(correct)
    assert actual["horizontal_deg"] == pytest.approx(87)
    assert abs(actual["vertical_deg"]-58) < 2


def test_frustum_projector_preserves_near_far_and_behind_camera_rejection():
    calibration = {"width": 848, "height": 480, "K": [[400, 0, 424], [0, 400, 240], [0, 0, 1]],
                   "T_world_camera": np.eye(4), "near_m": .07, "far_m": .5}
    points = np.array([[0, 0, .1], [0, 0, .06], [0, 0, -.1], [0, 0, .51], [1, 0, .1]])
    uv, z, valid = project_points(points, calibration)
    np.testing.assert_allclose(uv[0], [424, 240])
    assert valid.tolist() == [True, False, False, False, False]


def test_all_depth_error_tails_and_native_holes_remain_visible():
    expected = np.ones((2, 3))
    actual = expected.copy(); actual[0, 0] += .2; actual[0, 1] = 0
    result = depth_comparison(actual, actual > 0, expected, expected > 0)
    assert result["maximum_absolute_error_m"] == pytest.approx(.2)
    assert result["absolute_error_gt_5mm_pixels"] == 1
    assert result["valid_mask_iou"] == pytest.approx(5/6)


def test_obb_certified_gap_and_rigid_covariance_with_overlap_distinct():
    a, b = np.eye(4), np.eye(4);b[0, 3] = .11
    assert obb_signed_projection_gap(a, [.1]*3, b, [.1]*3) == pytest.approx(.01)
    frame = pose_matrix([.3, -.2, .5, *Rotation.from_euler("xyz", [.3, -.2, .1]).as_quat()])
    assert obb_signed_projection_gap(frame@a, [.1]*3, frame@b, [.1]*3) == pytest.approx(.01)
    b[0, 3] = .09
    assert obb_signed_projection_gap(a, [.1]*3, b, [.1]*3) < 0
