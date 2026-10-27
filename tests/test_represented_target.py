import numpy as np

from robotic_cutting_sim.represented_target import compare_represented_columns, combine_with_physical_receipt

LIMITS = dict(maximum_positive_residual_m=.001, maximum_gouge_deficit_m=.001,
              p95_absolute_error_m=.0005, rms_error_m=.0005,
              overcut_volume_fraction_of_initial_base=.01, actual_missing_fraction=.01,
              target_missing_fraction=.01, union_missing_fraction=.01)


def compare(actual, av, target, tv):
    return compare_represented_columns(actual, av, target, tv, spacing_xy_m=(.001, .001),
                                      initial_base_volume_m3=.000002, limits=LIMITS)[0]


def test_both_empty_stays_in_missing_denominator_and_can_fail_coverage():
    h, valid = np.full((10, 10), .02), np.ones((10, 10), bool)
    valid[0, 0] = False
    result = compare(h, valid, h, valid)
    assert result["status"] == "PASS_REPRESENTED_SHAPE"
    assert result["metrics"]["columns"] == 100
    assert result["metrics"]["both_empty_columns"] == 1
    assert result["metrics"]["union_missing_fraction"] == .01
    valid[0, 1] = False
    result = compare(h, valid, h, valid)
    assert result["status"] == "FAIL_REPRESENTED_SHAPE"
    assert result["metrics"]["maximum_gouge_deficit_m"] == 0
    assert not result["checks"]["union_missing_fraction"]


def test_one_sided_empty_column_keeps_full_height_error():
    h, av, tv = np.full((10, 10), .02), np.ones((10, 10), bool), np.ones((10, 10), bool)
    av[0, 0] = False
    result = compare(h, av, h, tv)
    assert result["metrics"]["maximum_gouge_deficit_m"] == .02
    assert result["metrics"]["actual_empty_target_filled_columns"] == 1
    assert result["status"] == "FAIL_REPRESENTED_SHAPE"
    result = compare(h, tv, h, av)
    assert result["metrics"]["maximum_positive_residual_m"] == .02
    assert result["metrics"]["actual_filled_target_empty_columns"] == 1


def test_signed_known_offsets_fail_without_a_fitted_bias_correction():
    h, valid = np.full((10, 10), .02), np.ones((10, 10), bool)
    for delta in (-.0015, .0015):
        result = compare(h+delta, valid, h, valid)
        assert result["status"] == "FAIL_REPRESENTED_SHAPE"
        assert np.isclose(result["metrics"]["p95_absolute_error_m"], .0015)
        assert not result["metrics"]["error_trimming_applied"]


def test_shape_alone_or_failed_physical_gate_cannot_promote_combined_goal():
    shape = {"status": "PASS_REPRESENTED_SHAPE"}
    assert combine_with_physical_receipt(shape, None, ["physical_split"])["status"] == "NOT_RUN_COMBINED_GOAL"
    assert combine_with_physical_receipt(shape, {"checks": {"physical_split": False}}, ["physical_split"])["status"] == "FAIL_REPRESENTED_GOAL"
    receipt = combine_with_physical_receipt(shape, {"checks": {"physical_split": True}}, ["physical_split"])
    assert receipt["status"] == "PASS_REPRESENTED_GOAL"
    assert not receipt["full_simulation_acceptance"]
    assert not receipt["original_v1_result_overwritten"]
