"""RUBBER-CONT-09 b rule v3 (K2 review K2-3; ``scripts/rubber/p2_4b/gpu_contact_batched_check_rule_v3.py``): per-quantity
floors, count / flag summary columns exact. Pure numpy; no GPU, no Warp, no local artifacts."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"/"rubber"/"p2_4b"))
import gpu_contact_batched_check_rule_v3 as g3  # noqa: E402
import gpu_contact_batched_check_v2 as g2  # noqa: E402

LAM, EPS = 4.0, 1e-6


def test_diag_columns_copy_matches_the_solver():
    src = (ROOT/"src"/"robotic_cutting_sim"/"tlmpm_rubber_v1.py").read_text()
    start = src.index("DIAG_COLUMNS = (")
    ns = {}
    exec(src[start:src.index(")", start)+1], ns)
    assert ns["DIAG_COLUMNS"] == g3.DIAG_COLUMNS
    assert set(g3.EXACT_COLUMNS) <= set(g3.DIAG_COLUMNS)


def _summary(k, energy, noise, rng, particles=41220.0, cells=5175.0):
    s = np.zeros((k, len(g3.DIAG_COLUMNS)))
    s[:, g3.DIAG_COLUMNS.index("particles")] = particles
    s[:, g3.DIAG_COLUMNS.index("cells")] = cells
    s[:, g3.DIAG_COLUMNS.index("J_min")] = 1.0
    s[:, g3.DIAG_COLUMNS.index("elastic")] = energy+noise*rng.normal(size=k)
    return s


def test_v2_floor_was_vacuous_and_v3_rejects_the_same_shift():
    """An out-of-reach env with ~1e-6 J elastic energy and 1e-9 J noise: a 1e-3 J coupling shift passes v2 (floor
    1e-6 x 41,220 particles = 0.041) and is rejected by v3 (floor of the elastic column 1e-12 J)."""
    rng = np.random.default_rng(7)
    A = _summary(5, 1e-6, 1e-9, rng)
    B = _summary(5, 1e-6, 1e-9, rng)
    B[:, g3.DIAG_COLUMNS.index("elastic")] += 1e-3
    v2 = g2.band_v2(A, B, LAM, EPS)
    assert v2["floor"] == pytest.approx(EPS*41220.0) and v2["pass"].all()
    rows = {q["quantity"]: q for q in g3.band_v3("summary", A, B, LAM, EPS)}
    assert not rows["summary.elastic"]["pass"]
    assert rows["summary.elastic"]["floor"] < 1e-11
    assert rows["summary.J_min"]["pass"] and rows["summary.J_min"]["floor"] == pytest.approx(EPS)
    # counts and the nonfinite flag are exact columns, never band-scored
    for c in g3.EXACT_COLUMNS:
        assert rows[f"summary.{c}"]["kind"] == "exact" and rows[f"summary.{c}"]["pass"]


def test_exact_columns_fail_on_any_change():
    rng = np.random.default_rng(1)
    A = _summary(5, 1e-6, 0.0, rng)
    B = _summary(5, 1e-6, 0.0, rng, particles=41221.0)
    rows = {q["quantity"]: q for q in g3.band_v3("summary", A, B, LAM, EPS)}
    assert not rows["summary.particles"]["pass"] and rows["summary.cells"]["pass"]


def test_wrench_force_and_torque_have_their_own_floor():
    """A 1e-7 N.m torque shift under a 1 N force: v2 hides it (floor 1e-6 N), v3 rejects it (torque floor 1e-12)."""
    A = np.zeros((5, 6))
    A[:, 2] = 1.0
    A[:, 5] = 1e-6
    B = A.copy()
    B[:, 5] += 1e-7
    assert g2.band_v2(A, B, LAM, EPS)["pass"].all()
    rows = {q["quantity"]: q for q in g3.band_v3("wrench", A, B, LAM, EPS)}
    assert rows["wrench[0].force"]["pass"] and not rows["wrench[0].torque"]["pass"]
    multi = {q["quantity"] for q in g3.band_v3("box_wrench", np.zeros((5, 2, 1, 6)), np.zeros((5, 2, 1, 6)), LAM, EPS)}
    assert multi == {"box_wrench[0].force", "box_wrench[0].torque", "box_wrench[1].force", "box_wrench[1].torque"}


def test_markers_unchanged_from_v2_and_band_v3_all_aggregates():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(5, 9, 3)), rng.normal(size=(5, 9, 3))
    (q,) = g3.band_v3("markers", A, B, LAM, EPS)
    v2 = g2.band_v2(A, B, LAM, EPS)
    assert q["pass"] == bool(v2["pass"].all()) and q["floor"] == v2["floor"]

    def rec(step_vals):
        return {"obs": [[{"summary": v} for v in env_vals] for env_vals in step_vals]}
    runs_a = [rec([[_summary(1, 1e-6, 0, rng)[0], _summary(1, 1e-6, 0, rng)[0]]]) for _ in range(5)]
    runs_b = [rec([[_summary(1, 1e-6, 0, rng)[0], _summary(1, 2e-6, 0, rng)[0]]]) for _ in range(5)]
    ok, rows, summ = g3.band_v3_all(runs_a, runs_b, [0], LAM, EPS)
    assert ok and summ["band_scalars"] == 15 and summ["exact_scalars"] == 3
    ok1, _, summ1 = g3.band_v3_all(runs_a, runs_b, [1], LAM, EPS)
    assert not ok1 and summ1["band_scalars_failing"] == 1


def test_laws_pin_the_existing_protocols():
    import hashlib
    for proto, sha, law in g3.LAWS.values():
        raw = (ROOT/"configs"/"rubber"/proto).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha
        assert __import__("json").loads(raw)["common"]["contact_law"] == law
