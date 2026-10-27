"""Versioned full-column represented-solid diagnostics, separate from Task v1.

Empty columns represent no solid above the workpiece base datum. They remain in
all 4,800 differences and all missing/union counts, never a fitted or culled ROI.
These evaluator arrays must never be inserted into policy observations.
"""
import numpy as np


def compare_represented_columns(actual_height, actual_valid, target_height, target_valid,
                                *, spacing_xy_m, initial_base_volume_m3, limits):
    actual, target = np.asarray(actual_height, float), np.asarray(target_height, float)
    av, tv = np.asarray(actual_valid, bool), np.asarray(target_valid, bool)
    if (actual.shape != target.shape or av.shape != actual.shape or tv.shape != target.shape
            or actual.ndim != 2 or not actual.size
            or not np.isfinite(actual[av]).all() or not np.isfinite(target[tv]).all()
            or np.asarray(spacing_xy_m).shape != (2,) or min(spacing_xy_m) <= 0
            or not np.isfinite(initial_base_volume_m3) or initial_base_volume_m3 <= 0):
        raise ValueError("Need aligned height/validity grids, finite intersected columns, positive area/volume")
    effective_actual = np.where(av, actual, 0.)
    effective_target = np.where(tv, target, 0.)
    error = effective_actual-effective_target
    deficit = np.maximum(-error, 0.)
    union_missing = ~av | ~tv
    metrics = {"columns": int(actual.size),
        "maximum_positive_residual_m": float(np.maximum(error, 0.).max()),
        "maximum_gouge_deficit_m": float(deficit.max()),
        "p95_absolute_error_m": float(np.quantile(abs(error), .95)),
        "rms_error_m": float(np.sqrt(np.mean(error**2))),
        "overcut_volume_m3": float(deficit.sum()*np.prod(spacing_xy_m)),
        "overcut_volume_fraction_of_initial_base": float(deficit.sum()*np.prod(spacing_xy_m)/initial_base_volume_m3),
        "actual_missing_columns": int((~av).sum()), "target_missing_columns": int((~tv).sum()),
        "union_missing_columns": int(union_missing.sum()), "both_empty_columns": int((~av & ~tv).sum()),
        "actual_empty_target_filled_columns": int((~av & tv).sum()),
        "actual_filled_target_empty_columns": int((av & ~tv).sum()),
        "actual_missing_fraction": float((~av).mean()), "target_missing_fraction": float((~tv).mean()),
        "union_missing_fraction": float(union_missing.mean()),
        "absolute_error_gt_0_5mm_columns": int(np.sum(abs(error) > .0005)),
        "absolute_error_gt_1mm_columns": int(np.sum(abs(error) > .001)),
        "error_trimming_applied": False}
    checks = {name: bool(metrics[name] <= value) for name, value in limits.items()}
    receipt = {"status": "PASS_REPRESENTED_SHAPE" if all(checks.values()) else "FAIL_REPRESENTED_SHAPE",
               "scope": "Full fixed-footprint represented-solid column geometry only",
               "metrics": metrics, "checks": checks, "empty_column_height_m": 0.,
               "all_columns_retained": True, "physical_cutting_inferred": False,
               "original_v1_result_overwritten": False}
    debug = {"actual_height_m": actual, "actual_valid": av, "target_height_m": target,
             "target_valid": tv, "actual_effective_height_m": effective_actual,
             "target_effective_height_m": effective_target, "signed_error_m": error,
             "deficit_m": deficit, "union_missing_mask": union_missing}
    return receipt, debug


def combine_with_physical_receipt(shape_receipt, physical_receipt, required_checks):
    """Require explicit physical gates; a shape-only result cannot certify a task."""
    physics = {} if physical_receipt is None else physical_receipt.get("checks", {})
    absent = [name for name in required_checks if name not in physics]
    checks = {name: physics.get(name) is True for name in required_checks}
    shape = shape_receipt["status"] == "PASS_REPRESENTED_SHAPE"
    status = ("NOT_RUN_COMBINED_GOAL" if physical_receipt is None or absent else
              "PASS_REPRESENTED_GOAL" if shape and all(checks.values()) else "FAIL_REPRESENTED_GOAL")
    return {"status": status, "represented_shape_passed": shape,
            "required_physical_checks": checks, "missing_physical_checks": absent,
            "full_simulation_acceptance": False, "original_v1_result_overwritten": False,
            "scope": "Separate represented-solid task goal with explicit physical topology/removal/mass/history gates"}
