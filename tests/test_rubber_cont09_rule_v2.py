"""Arithmetic of the RUBBER-CONT-09 (b) v2 rule (protocol rubber_contact_tests_v2, ADR 0023 §22.9) as implemented in
``scripts/rubber/p2_4b/gpu_contact_batched_check_v2.py``: d = max_{i,j} |A_i - B_j|, s = the larger within-side range,
PASS iff d <= lam s + eps_rel max_{i,c} |A_i[c]|. Pure numpy; no GPU, no Warp, no local artifacts."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"/"rubber"/"p2_4b"))
import gpu_contact_batched_check_v2 as g  # noqa: E402

LAM, EPS = 4.0, 1e-6


def test_thresholds_and_law_match_the_pinned_protocol():
    import hashlib
    raw = g.PROTOCOL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == g.PROTOCOL_SHA256
    proto = json.loads(raw)
    assert proto["tests"]["RUBBER-CONT-09"]["thresholds"] == {"k_per_side": 5, "lam": LAM, "eps_rel": EPS}
    assert proto["common"]["contact_law"] == g.LAW == "grid_node_penalty_coulomb_v2"


def test_hand_computed_values():
    A = np.array([[1.0, 10.0], [1.2, 10.0], [0.9, 10.0], [1.1, 10.0], [1.0, 10.0]])
    B = np.array([[1.05, 10.0], [1.3, 10.0], [1.0, 10.0], [0.95, 10.0], [1.1, 10.0]])
    r = g.band_v2(A, B, LAM, EPS)
    # component 0: d = max(max A - min B, max B - min A) = max(1.2 - 0.95, 1.3 - 0.9) = 0.4; s = max(0.3, 0.35)
    assert r["d"][0] == pytest.approx(0.4)
    assert r["s"][0] == pytest.approx(0.35)
    # floor uses the largest |component| of the observable vector over the A episodes: 1e-6 * 10
    assert r["floor"] == pytest.approx(1e-5)
    assert r["limit"][0] == pytest.approx(4*0.35+1e-5)
    # component 1: deterministic and identical -> d = s = 0 -> passes on the floor
    assert r["d"][1] == 0.0 and r["s"][1] == 0.0 and r["pass"].all()


def test_d_equals_max_pairwise_difference():
    rng = np.random.default_rng(3)
    A, B = rng.normal(size=(5, 7)), rng.normal(size=(5, 7))
    r = g.band_v2(A, B, LAM, EPS)
    brute = np.abs(A[:, None, :]-B[None, :, :]).max(axis=(0, 1))
    np.testing.assert_allclose(r["d"], brute, rtol=0, atol=0)
    np.testing.assert_allclose(r["s"], np.maximum(A.max(0)-A.min(0), B.max(0)-B.min(0)), rtol=0, atol=0)


def test_boundary_is_inclusive_and_shift_beyond_lam_plus_one_s_rejects():
    A = np.array([[0.0], [1.0], [0.5], [0.5], [0.5]])          # s_A = 1
    B_edge = A+3.0                                               # d = max(1 - 3, 4 - 0) = 4 = lam * s: PASS
    assert g.band_v2(A, B_edge, LAM, 0.0)["pass"].all()
    B_out = A+3.0+1e-9                                           # d just above lam * s: FAIL
    assert not g.band_v2(A, B_out, LAM, 0.0)["pass"].any()
    B_shift = A+(LAM+1)*1.0+1e-12                                # a shift > (lam + 1) s is always rejected
    assert not g.band_v2(A, B_shift, LAM, 0.0)["pass"].any()


def test_bitwise_deterministic_scalar_with_any_change_fails_unless_within_floor():
    A = np.full((5, 3), 2.0)
    B = A.copy()
    B[:, 0] += 1e-6                                              # floor = 1e-6 * 2 = 2e-6 >= 1e-6: PASS
    B[:, 1] += 3e-6                                              # > floor, s = 0: FAIL
    r = g.band_v2(A, B, LAM, EPS)
    assert r["pass"].tolist() == [True, False, True]


def test_band_v2_all_aggregates_rows_and_scalars():
    def rec(vals):
        return {"obs": [[{"x": np.array(v)} for v in step] for step in vals]}
    # 2 steps, 2 envs, one observable of 2 components; env 1 is excluded from the check
    rng = np.random.default_rng(0)
    runs_a = [rec([[rng.normal(size=2), [0.0, 0.0]], [rng.normal(size=2), [0.0, 0.0]]]) for _ in range(5)]
    runs_b = [rec([[rng.normal(size=2), [9.0, 9.0]], [rng.normal(size=2), [9.0, 9.0]]]) for _ in range(5)]
    ok, rows, n, summ = g.band_v2_all(runs_a, runs_b, [0], LAM, EPS)
    assert n == 4 and summ["rows"] == 2 and ok
    ok1, rows1, n1, summ1 = g.band_v2_all(runs_a, runs_b, [1], LAM, EPS)
    assert not ok1 and summ1["scalars_failing"] == 4 and summ1["scalars_with_s_zero"] == 4
    assert rows1[0]["d_over_limit"] == np.inf


def test_component_count_mismatch_refused():
    with pytest.raises(ValueError):
        g.band_v2(np.zeros((5, 2)), np.zeros((5, 3)), LAM, EPS)
