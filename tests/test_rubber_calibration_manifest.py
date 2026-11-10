"""R0.2: calibration-dataset manifest validator (ADR 0023 §14.2): CSV headers/units/time, splits, freeze hash."""
import copy
import json

import pytest

from robotic_cutting_sim import rubber_calibration as rc
from robotic_cutting_sim.rubber_material import ROOT


def _manifest():
    return {
        "schema": rc.SCHEMA, "dataset_id": "synthetic_test_v1", "status": "PILOT", "units": "SI",
        "specimens": [{"specimen_id": "S1", "material_label": "synthetic", "supplier": "-", "batch": "-",
                       "dimensions_m": {"length": 0.1, "width": 0.01, "thickness": 0.004},
                       "dimension_uncertainty_m": 1e-4, "mass_kg": 0.0046, "mass_uncertainty_kg": 1e-5,
                       "temperature_c": 21.0, "preconditioning": {"mullins_cycles": 3, "max_stretch": 2.0}}],
        "experiments": [
            {"experiment_id": "T1", "type": "tensile_load_unload", "specimen_id": "S1", "gauge_length_m": 0.05,
             "grip": {"type": "clamped", "clamp_length_m": 0.01}, "crosshead_speed_m_s": 0.001,
             "max_stretch": 2.0, "cycles": 1, "data_file": "T1.csv",
             "columns": {"time_s": "s", "displacement_m": "m", "force_n": "N"},
             "sensor": {"load_cell_range_n": 100.0, "resolution_n": 0.01, "sample_rate_hz": 100.0},
             "temperature_c": 21.0, "split": "fit"},
            {"experiment_id": "F1", "type": "friction_sliding", "specimen_id": "S1", "counterface": "steel",
             "normal_load_n": 1.0, "sliding_speed_m_s": 0.01, "data_file": "F1.csv",
             "columns": {"time_s": "s", "tangential_force_n": "N", "normal_force_n": "N", "displacement_m": "m"},
             "temperature_c": 21.0, "split": "heldout"}],
        "splits": {"fit": ["T1"], "heldout": ["F1"], "frozen_at": None, "sha256": None},
        "boundary_conditions_note": "synthetic test data, not a material"}


def _write(tmp_path):
    (tmp_path/"T1.csv").write_text("time_s,displacement_m,force_n\n0,0,0\n0.1,0.0001,0.05\n0.2,0.0002,0.1\n")
    (tmp_path/"F1.csv").write_text("time_s,tangential_force_n,normal_force_n,displacement_m\n0,0,1,0\n0.1,0.5,1,0.001\n")
    return tmp_path


def test_shipped_manifest_has_no_real_data():
    m = rc.load_manifest(ROOT/"configs"/"rubber"/"calibration"/"none_v1.json")
    assert m["status"] == "NO_REAL_DATA" and m["specimens"] == [] and m["experiments"] == []


def test_valid_pilot_manifest(tmp_path):
    result = rc.validate_manifest(_manifest(), _write(tmp_path))
    assert result["rows"] == {"T1": 3, "F1": 2}


@pytest.mark.parametrize("csv_text,message", [
    ("time_s,force_n,displacement_m\n0,0,0\n", "header"),
    ("time_s,displacement_m,force_n\n0,0,0\n0,1,1\n", "strictly increasing"),
    ("time_s,displacement_m,force_n\n0,0,0\n0.1,nan,1\n", "non-finite"),
    ("time_s,displacement_m,force_n\n", "no data rows"),
    ("time_s,displacement_m,force_n\n0,0\n", "values"),
])
def test_csv_refusals(tmp_path, csv_text, message):
    _write(tmp_path)
    (tmp_path/"T1.csv").write_text(csv_text)
    with pytest.raises(rc.CalibrationManifestError, match=message):
        rc.validate_manifest(_manifest(), tmp_path)


def test_unit_and_boundary_condition_refusals(tmp_path):
    _write(tmp_path)
    m = _manifest()
    m["experiments"][0]["columns"]["force_n"] = "lbf"
    with pytest.raises(rc.CalibrationManifestError, match="unknown unit"):
        rc.validate_manifest(m, tmp_path)
    m = _manifest()
    m["experiments"][0]["grip"] = {}
    with pytest.raises(rc.CalibrationManifestError, match="boundary condition"):
        rc.validate_manifest(m, tmp_path)
    m = _manifest()
    del m["experiments"][1]["counterface"]
    with pytest.raises(rc.CalibrationManifestError, match="missing"):
        rc.validate_manifest(m, tmp_path)


def test_split_disjointness(tmp_path):
    _write(tmp_path)
    m = _manifest()
    m["splits"]["heldout"] = ["F1", "T1"]
    with pytest.raises(rc.CalibrationManifestError, match="disjoint"):
        rc.validate_manifest(m, tmp_path)
    m = _manifest()
    m["experiments"][1]["split"] = "fit"
    with pytest.raises(rc.CalibrationManifestError, match="split"):
        rc.validate_manifest(m, tmp_path)


def test_freeze_hash_detects_changes(tmp_path):
    _write(tmp_path)
    frozen = rc.freeze(_manifest(), tmp_path, "2026-11-10T00:00:00Z")
    assert frozen["status"] == "FROZEN_FOR_HELDOUT" and len(frozen["splits"]["sha256"]) == 64
    rc.validate_manifest(frozen, tmp_path)
    (tmp_path/"manifest.json").write_text(json.dumps(frozen))
    assert rc.load_manifest(tmp_path/"manifest.json")["status"] == "FROZEN_FOR_HELDOUT"
    (tmp_path/"F1.csv").write_text("time_s,tangential_force_n,normal_force_n,displacement_m\n0,0,1,0\n0.1,0.6,1,0.001\n")
    with pytest.raises(rc.CalibrationManifestError, match="freeze hash"):
        rc.validate_manifest(frozen, tmp_path)
    edited = copy.deepcopy(frozen)
    edited["experiments"][0]["temperature_c"] = 25.0
    _write(tmp_path)
    with pytest.raises(rc.CalibrationManifestError, match="freeze hash"):
        rc.validate_manifest(edited, tmp_path)


def test_no_real_data_must_be_empty():
    m = json.loads((ROOT/"configs"/"rubber"/"calibration"/"none_v1.json").read_text())
    m["splits"]["fit"] = ["T1"]
    with pytest.raises(rc.CalibrationManifestError, match="NO_REAL_DATA"):
        rc.validate_manifest(m, check_files=False)
