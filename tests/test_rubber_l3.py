"""F3: RUBBER-PHYS-10 L3 rule v4 (rubber_l3): leave-one-out statistics, prediction bound, Bonferroni family rate."""
import numpy as np
import pytest

from robotic_cutting_sim import rubber_l3 as L


def test_loo_statistic_by_hand():
    ref = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 4.0]])
    m = L.loo_statistics(ref)
    assert m.tolist() == [1.5, 1.5, 3.0]          # e.g. run 2 vs mean of runs 0, 1: max(|2-0.5|, |4-1|)


def test_deterministic_reference_requires_bitwise_equality():
    ref = {"a": np.ones((10, 5))}
    ok = L.evaluate(ref, {"same": {"a": np.ones(5)}, "off": {"a": np.r_[np.ones(4), 1+1e-12]}})
    assert ok["bounds"]["a"]["method"] == "deterministic_reference"
    assert ok["runs"]["same"]["verdict"] == "PASS" and ok["runs"]["off"]["verdict"] == "FAIL"


def test_family_false_reject_rate_is_controlled_for_exchangeable_runs():
    """Monte Carlo: correlated Gaussian series (random walk in time), 16 statistics, n = 20 reference repeats; an
    exchangeable candidate is rejected (any statistic OUTSIDE) at a rate near or below alpha_family = 0.05."""
    rng = np.random.default_rng(1)
    trials, rejects = 300, 0
    for _ in range(trials):
        ref, cand = {}, {}
        for s in range(16):
            walk = np.cumsum(rng.normal(size=(21, 30)), axis=1)*(1+s)
            ref[f"s{s}"], cand[f"s{s}"] = walk[:20], walk[20]
        rejects += L.evaluate(ref, {"c": cand})["runs"]["c"]["verdict"] == "FAIL"
    assert rejects/trials <= 0.10


def test_a_shifted_run_is_rejected():
    rng = np.random.default_rng(2)
    walk = np.cumsum(rng.normal(size=(21, 30)), axis=1)
    res = L.evaluate({"a": walk[:20]}, {"shifted": {"a": walk[20]+25.0}})
    assert res["runs"]["shifted"]["verdict"] == "FAIL"


def test_needs_three_repeats():
    with pytest.raises(ValueError):
        L.loo_statistics(np.zeros((2, 3)))
