import numpy as np

from scripts.audit_task_object_visibility import object_metrics


def test_roi_keeps_holes_and_valid_large_error_tail_separate():
    reference = np.ones((2, 3), np.float32)
    native = np.array([[1., 1.01, 0.], [1., 1.002, 1.]], np.float32)
    valid = np.array([[True, True, False], [True, True, True]])
    mask = np.array([[True, True, True], [False, True, False]])
    row, errors, _ = object_metrics(mask, reference, native, valid, reference, np.ones_like(valid))
    assert row["reference_visible_pixels"] == 4
    assert row["native_valid_coverage"] == .75
    assert row["native_invalid_pixels"] == 1
    assert row["native_depth_vs_mesh"]["absolute_error_gt_1mm_pixels"] == 2
    assert row["native_depth_vs_mesh"]["absolute_error_gt_5mm_pixels"] == 1
    assert row["native_invalid_or_error_gt_1mm_pixels"] == 3
    assert row["native_invalid_or_error_gt_5mm_pixels"] == 2
    assert len(errors) == 3
    assert row["native_depth_vs_mesh"]["max_absolute_error_m"] > .009


def test_nonfinite_marked_valid_is_recorded_as_malformed_without_finite_error_claim():
    reference = np.ones((1, 2))
    native = np.array([[np.nan, 1.]])
    valid = np.ones((1, 2), bool)
    row, _, _ = object_metrics(valid, reference, native, valid, reference, valid)
    assert row["native_valid_but_nonfinite_or_nonpositive_pixels"] == 1
    assert row["native_invalid_pixels"] == 0
    assert row["native_valid_coverage"] == .5
    assert row["native_depth_vs_mesh"]["pixels"] == 1
    assert row["native_invalid_or_error_gt_1mm_pixels"] == 1
