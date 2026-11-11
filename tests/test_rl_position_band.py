"""Position band v2 (review fix F2): pre-declared method, derivation, per-run classes and the run-set rule."""
import math

import numpy as np
import pytest

from robotic_cutting_sim.rl import position_band as PB

pytest.importorskip("scipy")


def test_method_is_pre_declared_and_names_the_consumer_api():
    method = PB.load_method()
    assert method["id"] == PB.BAND_ID and method["metric"]["name"] == PB.METRIC
    assert method["tolerance"]["alpha"] == 0.01 and method["reference_sample"]["runs_per_seed_min"] == PB.MIN_RUNS
    assert method["supersedes"]["kept"] is True
    assert set(method["classification_per_run"]) == set(PB.CLASSES)


@pytest.mark.parametrize("n,c", [(1, 0), (5, 0), (6, 1), (35, 1), (36, 2), (40, 2)])
def test_critical_count(n, c):
    assert PB.critical_count(n) == c


def test_derive_lognormal_bound_and_classes():
    values = np.exp(np.random.default_rng(3).normal(math.log(2e-4), .3, 12))
    band = PB.derive(values)
    assert band["status"] == "DERIVED" and band["kind"] == "one_sided_lognormal_prediction_bound"
    y = np.log(values)
    from scipy import stats
    k = stats.t.ppf(.99, 11) * math.sqrt(1 + 1 / 12)
    assert band["upper_m"] == pytest.approx(math.exp(y.mean() + k * y.std(ddof=1)))
    assert band["nominal_false_reject_rate"] == .01 and band["upper_m"] > values.max()
    reference = {"schema": PB.REFERENCE_SCHEMA, "seeds": {"15": band}}
    assert PB.classify(15, band["upper_m"], reference) == PB.WITHIN
    assert PB.classify(15, band["upper_m"] * 1.0001, reference) == PB.OUTSIDE
    assert PB.classify(15, band["lower_m"] * .5, reference) == PB.BELOW
    assert PB.classify(15, float("nan"), reference) == PB.OUTSIDE
    assert PB.classify(15, None, reference) == PB.NO_DATA
    assert PB.classify(15, 1., reference, invalid=True) == PB.INVALID
    assert PB.classify(15, 1., reference, applicable=False) == PB.NOT_APPLICABLE
    with pytest.raises(ValueError):
        PB.classify(512, 1e-5, reference)


def test_non_normal_reference_falls_back_to_the_maximum():
    values = np.r_[np.full(11, 1e-4) * np.linspace(1, 1.01, 11), 5e-2]
    band = PB.derive(values)
    assert band["shapiro_wilk_p"] < PB.NORMALITY_P_MIN
    assert band["kind"] == "fallback_empirical_max" and band["upper_m"] == pytest.approx(5e-2)
    assert band["nominal_false_reject_rate"] == pytest.approx(1 / 13)


def test_undersampled_reference_is_inconclusive():
    band = PB.derive([1e-4, 2e-4, 3e-4, 4e-4])
    assert band["status"] == "INCONCLUSIVE_UNDERSAMPLED" and band["upper_m"] is None
    with pytest.raises(ValueError):
        PB.classify(512, 1e-4, {"schema": PB.REFERENCE_SCHEMA, "seeds": {"512": band}})


def test_aggregate():
    assert PB.aggregate([PB.WITHIN] * 4)["verdict"] == "MET"
    assert PB.aggregate([PB.WITHIN] * 3 + [PB.OUTSIDE])["verdict"] == "NOT_MET"
    many = PB.aggregate([PB.WITHIN] * 9 + [PB.BELOW, PB.OUTSIDE, PB.INVALID])
    assert many["verdict"] == "MET" and many["scored"] == 11 and many["critical_count"] == 1
    assert many["not_scored"] == {PB.INVALID: 1}
    assert PB.aggregate([PB.NO_DATA])["verdict"] == "NO_SCORED_RUNS"
    with pytest.raises(ValueError):
        PB.aggregate(["inside"])


def test_reference_file_matches_the_declared_method():
    if not PB.REFERENCE_PATH.exists():
        pytest.skip("reference values not derived yet (NOT_RUN)")
    reference = PB.load_reference()
    method = PB.load_method()
    assert reference["method_digest"] == PB.method_digest(method)
    for metric in PB.METRICS:
        for seed in ("512", "15"):
            band = reference["statistics"][metric]["seeds"][seed]
            assert band["n"] >= PB.MIN_RUNS and band["status"] == "DERIVED"
            again = PB.derive(band["values_m"], alpha=band["alpha"])   # SciPy versions differ in the last bits
            assert set(again) == set(band) and again["kind"] == band["kind"]
            for key, value in band.items():
                if isinstance(value, float):
                    assert again[key] == pytest.approx(value, rel=1e-6), key


def test_method_file_is_consumable_by_the_acceptance_judge():
    """``position_band_m = {seed: {high_m}}`` of the judge's statistic (final max position difference), as
    ``rl.perf_acceptance.load_position_band`` reads it."""
    if not PB.REFERENCE_PATH.exists():
        pytest.skip("reference values not derived yet (NOT_RUN)")
    method = PB.load_method()
    reference = PB.load_reference()
    assert method["position_band_m_statistic"] == PB.JUDGE_METRIC == "final_max_position_diff_m"
    table = method["position_band_m"]
    assert set(table) == {"512", "15"}
    for seed, entry in table.items():
        assert isinstance(entry["high_m"], float)
        assert entry["high_m"] == reference["statistics"][PB.JUDGE_METRIC]["seeds"][seed]["upper_m"]
    assert method["version"] == PB.BAND_ID and method["derived"]["method_digest"] == PB.method_digest(method)


def test_method_digest_ignores_only_the_derived_keys():
    method = PB.load_method()
    digest = PB.method_digest(method)
    assert PB.method_digest(dict(method, position_band_m={"512": {"high_m": 1.}}, status="X")) == digest
    assert PB.method_digest(dict(method, tolerance=dict(method["tolerance"], alpha=.05))) != digest
