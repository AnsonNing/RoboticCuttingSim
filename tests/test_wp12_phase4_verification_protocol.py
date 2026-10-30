"""configs/wp12/phase4_verification_protocol_v1.json: declared rows are well formed and runnable by the runner."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"configs/wp12/phase4_verification_protocol_v1.json"


def load():
    return json.loads(PROTOCOL.read_text())


def test_rows_are_unique_typed_and_counterfactuals_are_valid():
    from robotic_cutting_sim.tlmpm_v5_counterfactual import CounterfactualOptions
    p = load()
    ids = [r["id"] for r in p["rows"]]
    assert len(ids) == len(set(ids))
    for r in p["rows"]:
        assert r["kind"] in ("res", "e1", "probe") and r["part"] in ("A", "B")
        assert r["family"] in p["hypotheses"] or r["family"] == "baseline"
        if "counterfactual" in r:
            options = CounterfactualOptions(**r["counterfactual"])
            assert not options.is_identity()
            assert "COUNTERFACTUAL" in r["purpose"]
        else:
            assert "COUNTERFACTUAL" not in r["purpose"]
        if r["kind"] == "res":
            assert r["case"] in ("CUT", "TRIM", "OBL20", "MULTI2")
            assert abs(round(.02/r["dt_s"])*r["dt_s"]-.02) < 1e-12
            assert r["grid_dx_m"] == pytest.approx(4*r["spacing_m"])
        if r["kind"] == "e1":
            spec = r["e1_spec"]
            assert spec["scene"] == "bar_flat_board" and spec["edge_end_z_mm"] == 0.
        if r["kind"] == "probe":
            assert r["case"] in ("S7", "G3b") and r.get("counterfactual", {}).get("blade_contact_stiffness") is None


def test_every_hypothesis_row_reference_is_declared():
    p = load()
    ids = {r["id"] for r in p["rows"]}
    for name, h in p["hypotheses"].items():
        for rid in h.get("rows", []):
            assert rid in ids, (name, rid)
        for pair in h.get("pairs", []):
            assert pair[0] in ids, (name, pair)
    assert p["budget"]["cap_gpu_h"] >= p["budget"]["total_estimate_gpu_h"]


def test_runner_reads_the_declared_rows():
    spec = importlib.util.spec_from_file_location("verify_runner", ROOT/"scripts/run_wp12_phase4_verification.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = runner.protocol_rows(PROTOCOL)
    assert set(rows) == {r["id"] for r in load()["rows"]}
    for case in ("S7", "G3b"):
        points, masses, rot, start, phases, geo = runner.probe_case(case)
        assert len(points) == len(masses) > 0 and len(phases) == 2 and "locus" in geo
