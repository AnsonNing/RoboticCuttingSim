"""ADR 0006 Amendment 1 M2 rescore (scripts/rescore_phase1_oblique.py) on synthetic data (CPU, read-only)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPACING = .001


def _module():
    spec = importlib.util.spec_from_file_location("rescore_phase1_oblique", ROOT/"scripts/rescore_phase1_oblique.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bar(label_shift_m=0., stray=False, plane_x_m=0.):
    """1 mm particles in an x-z bar, chain stencils along x (4 mm nodes), labels from a 20 deg plane."""
    x, z = np.meshgrid(np.arange(-.0195, .02, SPACING), np.arange(.0005, .02, SPACING), indexing="ij")
    reference = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
    cell = np.floor((reference[:, 0]+.1)/.004).astype(int)
    nodes = np.zeros((len(reference), 8), np.int32)
    nodes[:, 0], nodes[:, 1] = cell, cell+1
    weights = np.zeros((len(reference), 8))
    weights[:, :2] = .5
    tilt = np.deg2rad(20.)
    origin, normal = np.array([plane_x_m, 0., .01]), np.array([np.cos(tilt), 0., np.sin(tilt)])
    signed = (reference-origin)@normal
    codes = np.where(signed < label_shift_m, 1, 2).astype(np.int32)
    if stray:
        far = np.flatnonzero((signed > 3*SPACING) & (signed < 4*SPACING))[0]
        codes[far] = 1
    data = dict(reference_stencil_nodes=nodes, reference_stencil_weights=weights,
                reference_stencil_gradients=np.zeros((*nodes.shape, 3)), reference_positions=reference,
                mass_kg=np.full(len(reference), 1e-6), codes=np.stack([np.zeros_like(codes), codes]))
    return data, origin, normal


def test_ideal_labels_pass_amended_but_fail_the_original_2mm_statistic():
    m = _module()
    data, origin, normal = _bar()
    report = m.rescore_m2(data, origin, normal, SPACING)
    assert report["verdict"] == "PASS" and all(report["checks"].values())
    metrics = report["metrics"]
    assert metrics["actual_interface_m"] == pytest.approx(metrics["ideal_geometric_label_interface_m"])
    assert metrics["actual_interface_m"] > .002  # oblique plane on a grid-aligned stencil
    assert report["original_2mm"]["verdict"] == "FAIL"
    assert not report["original_2mm"]["checks"]["swept_plane_interface_le_2mm"]
    assert metrics["far_particles_in_opposite_side_component"] == 0


def test_labels_within_one_spacing_still_pass():
    m = _module()
    data, origin, normal = _bar(label_shift_m=.5*SPACING)
    report = m.rescore_m2(data, origin, normal, SPACING)
    assert report["verdict"] == "PASS"
    assert report["metrics"]["actual_interface_m"] > report["metrics"]["ideal_geometric_label_interface_m"]


def test_shifted_or_stray_labels_and_unequal_parts_fail():
    m = _module()
    data, origin, normal = _bar(label_shift_m=3*SPACING)
    shifted = m.rescore_m2(data, origin, normal, SPACING)
    assert shifted["verdict"] == "FAIL"
    assert not shifted["checks"]["interface_le_ideal_plus_one_spacing"]
    assert not shifted["checks"]["no_far_particle_in_opposite_side_component"]
    data, origin, normal = _bar(stray=True)
    stray = m.rescore_m2(data, origin, normal, SPACING)
    assert stray["verdict"] == "FAIL" and not stray["checks"]["no_far_particle_in_opposite_side_component"]
    assert stray["metrics"]["far_particles_in_opposite_side_component"] >= 1
    data, origin, normal = _bar(plane_x_m=.012)
    unequal = m.rescore_m2(data, origin, normal, SPACING)
    assert unequal["verdict"] == "FAIL" and not unequal["checks"]["two_substantial_parts_each_ge_40pct"]


def _row_dir(tmp_path, data, origin, normal, fit=True, row="M2"):
    row_dir = tmp_path/"m2_law5"
    row_dir.mkdir()
    planes = {"0": {"status": "FIT", "fit": {"origin": origin.tolist(), "normal": normal.tolist()}}} if fit else \
        {"0": {"status": "NOT_RUN", "reason": "none"}}
    (row_dir/"result.json").write_text(json.dumps({"row": row, "verdict": "FAIL", "cutting_law": 5,
        "particle_spacing_m": SPACING, "formal_resolution": True, "checks": {}, "context": {"max_cuts": 1},
        "metrics": {"swept_planes": planes}}))
    np.savez_compressed(row_dir/"physics_state.npz", **data)
    return row_dir


def test_cli_is_read_only_and_never_overwrites(tmp_path):
    m = _module()
    data, origin, normal = _bar()
    row_dir = _row_dir(tmp_path, data, origin, normal)
    before = {p.name: p.read_bytes() for p in row_dir.iterdir()}
    out = tmp_path/"rescore.json"
    assert m.main(["--row-dir", str(row_dir), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["verdict"] == "PASS" and report["original_2mm"]["verdict"] == "FAIL"
    assert report["source_row_verdict"] == "FAIL" and report["cutting_law"] == 5 and report["read_only"]
    assert set(report["inputs_sha256"]) == {"result.json", "physics_state.npz"}
    assert {p.name: p.read_bytes() for p in row_dir.iterdir()} == before
    with pytest.raises(SystemExit):
        m.main(["--row-dir", str(row_dir), "--out", str(out)])  # exists
    with pytest.raises(SystemExit):
        m.main(["--row-dir", str(row_dir), "--out", str(row_dir/"rescore.json")])  # inside the row dir
    assert not (row_dir/"rescore.json").exists()


def test_cli_reports_not_run_without_a_fit_and_rejects_other_rows(tmp_path):
    m = _module()
    data, origin, normal = _bar()
    row_dir = _row_dir(tmp_path, data, origin, normal, fit=False)
    out = tmp_path/"rescore.json"
    m.main(["--row-dir", str(row_dir), "--out", str(out)])
    assert json.loads(out.read_text())["verdict"] == "NOT_RUN"
    (tmp_path/"other").mkdir()
    other = _row_dir(tmp_path/"other", data, origin, normal, row="M4")
    with pytest.raises(SystemExit):
        m.main(["--row-dir", str(other), "--out", str(tmp_path/"other.json")])
