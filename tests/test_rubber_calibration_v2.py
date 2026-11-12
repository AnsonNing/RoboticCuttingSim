"""P5.1 / CAL-01 (manifest part): calibration manifest v2 (data origin, sensors, sealed SHA, origin-bound statuses),
directory guard, freeze hash, the REAL-gate negative tests (no synthetic manifest can yield REAL_CALIBRATED, in any
directory, status or verdict) and the gate-directory tripwire that closes the v1 hole (plan F10)."""
import copy
import itertools
import json
from pathlib import Path

import pytest

from robotic_cutting_sim import rubber_calibration as rc1
from robotic_cutting_sim import rubber_calibration_v2 as rc2
from robotic_cutting_sim import rubber_material as rm
from robotic_cutting_sim.rubber_material import ROOT

SHA = "a"*64


def _sensor(kind, sigma, res, rate, unit):
    return {"kind": kind, "noise_sigma": sigma, "resolution": res, "sample_rate_hz": rate, "full_scale": 50.0,
            "unit": unit}


def manifest(origin="synthetic_simulator", status=None):
    synthetic = origin == "synthetic_simulator"
    m = {"schema": rc2.SCHEMA, "dataset_id": "test_ds", "data_origin": origin,
         "status": status or ("SYNTHETIC_PILOT" if synthetic else "PILOT"), "units": "SI",
         "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
         "validation_kind": rc2.VALIDATION_SYNTHETIC if synthetic else rc2.VALIDATION_REAL,
         "specimens": [{"specimen_id": "S1", "material_label": "test fixture", "temperature_c": 21.0,
                        "dimensions_m": {"length": 0.06, "width": 0.012, "thickness": 0.004},
                        "nominal_dimensions_m": {"length": 0.06, "width": 0.012, "thickness": 0.004},
                        "dimension_uncertainty_m": 2e-5, "mass_kg": 0.0033, "mass_uncertainty_kg": 5e-7}],
         "experiments": [
             {"experiment_id": "T1", "type": "tensile_load_unload", "specimen_id": "S1", "split": "fit",
              "boundary_condition": "end grips", "loading": {"strain_rate_1_s": 1.0},
              "sensors": {"load_cell": _sensor("load_cell", 0.15, 1e-3, 1000.0, "N")},
              "data_files": {"force": {"file": "T1_force.csv",
                                       "columns": {"time_s": "s", "displacement_m": "m", "force_n": "N"}}}},
             {"experiment_id": "T3", "type": "tensile_load_unload", "specimen_id": "S1", "split": "heldout",
              "boundary_condition": "end grips", "loading": {"strain_rate_1_s": 3.0},
              "sensors": {"load_cell": _sensor("load_cell", 0.15, 1e-3, 1000.0, "N")},
              "data_files": {"force": {"file": "T3_force.csv",
                                       "columns": {"time_s": "s", "displacement_m": "m", "force_n": "N"}}}}],
         "splits": {"fit": ["T1"], "heldout": ["T3"], "frozen_at": None, "sha256": None},
         "boundary_conditions_note": "test fixture, not a material"}
    if synthetic:
        m["synthetic"] = {"generator_commit": "0123456789abcdef", "generator_tree_clean": True,
                          "truth_profile": {"n_t": 6, "cfl": 0.25}, "sensor_model_id": "test_sensors",
                          "sealed_truth_sha256": SHA, "protocol_sha256": SHA, "prior_box": {},
                          "noise_seed_sha256": SHA}
    return m


def write_files(d):
    for e in ("T1", "T3"):
        (Path(d)/f"{e}_force.csv").write_text("time_s,displacement_m,force_n\n0.001,0.00006,0.1\n0.002,0.00012,0.2\n")
    return d


def test_valid_synthetic_and_real_manifests(tmp_path):
    write_files(tmp_path)
    assert rc2.validate_manifest(manifest(), tmp_path)["rows"] == {"T1.force": 2, "T3.force": 2}
    assert rc2.validate_manifest(manifest("real_measurement"), tmp_path)["data_origin"] == "real_measurement"


@pytest.mark.parametrize("origin,status", [("synthetic_simulator", "PILOT"),
                                           ("synthetic_simulator", "FROZEN_FOR_HELDOUT"),
                                           ("synthetic_simulator", "NO_REAL_DATA"),
                                           ("real_measurement", "SYNTHETIC_PILOT"),
                                           ("real_measurement", "SYNTHETIC_FROZEN_FOR_HELDOUT")])
def test_status_is_bound_to_origin(tmp_path, origin, status):
    m = manifest(origin)
    m["status"] = status
    with pytest.raises(rc2.CalibrationManifestV2Error):
        rc2.validate_manifest(m, write_files(tmp_path))


def test_refusals(tmp_path):
    write_files(tmp_path)
    cases = []
    m = manifest(); m["data_origin"] = "simulated"; cases.append((m, "data_origin"))
    m = manifest(); del m["synthetic"]; cases.append((m, "synthetic"))
    m = manifest("real_measurement"); m["synthetic"] = manifest()["synthetic"]; cases.append((m, "synthetic"))
    m = manifest(); m["synthetic"]["sealed_truth_sha256"] = "x"; cases.append((m, "sha256"))
    m = manifest(); m["synthetic"]["generator_tree_clean"] = False; cases.append((m, "clean"))
    m = manifest(); m["validation_kind"] = rc2.VALIDATION_REAL; cases.append((m, "validation_kind"))
    m = manifest(); m["calibration_status"] = "REAL_CALIBRATED"; cases.append((m, "calibration_status"))
    m = manifest(); m["experiments"][0]["sensors"] = {}; cases.append((m, "sensors"))
    m = manifest(); m["experiments"][0]["sensors"]["load_cell"]["noise_sigma"] = -1; cases.append((m, "noise_sigma"))
    m = manifest(); m["experiments"][0]["sensors"]["load_cell"]["kind"] = "laser"; cases.append((m, "kind"))
    m = manifest(); m["splits"]["heldout"] = ["T1", "T3"]; cases.append((m, "disjoint"))
    m = manifest(); m["splits"]["heldout"] = []; cases.append((m, "exactly once"))
    m = manifest(); m["experiments"][1]["split"] = "fit"; cases.append((m, "split"))
    m = manifest(); m["experiments"][0]["data_files"]["force"]["file"] = "../T1_force.csv"; cases.append((m, "plain"))
    m = manifest(); m["schema"] = rc1.SCHEMA; cases.append((m, "schema"))
    for bad, match in cases:
        with pytest.raises(rc2.CalibrationManifestV2Error, match=match):
            rc2.validate_manifest(bad, tmp_path)


def test_freeze_hash_detects_changes(tmp_path):
    write_files(tmp_path)
    frozen = rc2.freeze(manifest(), tmp_path, "2026-11-11T00:00:00Z")
    assert frozen["status"] == "SYNTHETIC_FROZEN_FOR_HELDOUT" and len(frozen["splits"]["sha256"]) == 64
    rc2.validate_manifest(frozen, tmp_path)
    (tmp_path/"T3_force.csv").write_text("time_s,displacement_m,force_n\n0.001,0.00006,0.1\n0.002,0.00012,0.3\n")
    with pytest.raises(rc2.CalibrationManifestV2Error, match="freeze hash"):
        rc2.validate_manifest(frozen, tmp_path)
    write_files(tmp_path)
    edited = copy.deepcopy(frozen)
    edited["experiments"][0]["loading"]["strain_rate_1_s"] = 2.0
    with pytest.raises(rc2.CalibrationManifestV2Error, match="freeze hash"):
        rc2.validate_manifest(edited, tmp_path)
    assert rc2.freeze(manifest("real_measurement"), tmp_path, "t")["status"] == "FROZEN_FOR_HELDOUT"


def test_oracle_file_is_hashed(tmp_path):
    write_files(tmp_path)
    m = manifest()
    m["synthetic"]["truth_profile"]["oracle_file"] = "oracle_heldout.json"
    (tmp_path/"oracle_heldout.json").write_text("{}\n")
    frozen = rc2.freeze(m, tmp_path, "t")
    (tmp_path/"oracle_heldout.json").write_text("{\"T3\": 1}\n")
    with pytest.raises(rc2.CalibrationManifestV2Error, match="freeze hash"):
        rc2.validate_manifest(frozen, tmp_path)


def test_write_manifest_refuses_gate_directory_and_overwrite(tmp_path):
    target = rc2.GATE_DIR/"p2_5_test_must_not_exist.json"
    with pytest.raises(rc2.CalibrationManifestV2Error, match="REAL gate directory"):
        rc2.write_manifest(target, manifest())
    assert not target.exists()
    p = rc2.write_manifest(tmp_path/"m.json", manifest())
    with pytest.raises(rc2.CalibrationManifestV2Error, match="no overwrite"):
        rc2.write_manifest(p, manifest())


def _recipe(manifest_id):
    raw = json.loads((ROOT/"configs"/"materials"/"rubber_generic_visco_v1.json").read_text())
    raw["calibration_status"] = "REAL_CALIBRATED"
    raw["calibration_manifest"] = manifest_id
    raw["heldout_verdict"] = "PASS"
    return raw


@pytest.mark.parametrize("status,verdict,where", list(itertools.product(
    rc2.SYNTHETIC_STATUSES, ("PASS", "FAIL", "NOT_RUN", None), ("gate_copy", "synthetic_dir", "plain"))))
def test_no_synthetic_manifest_can_yield_real_calibrated(tmp_path, status, verdict, where):
    """F10: the v1 gate reads v1 manifests only, so a v2 synthetic manifest is refused in any directory, status or
    verdict; the v2 gate refuses synthetic manifests too."""
    d = tmp_path/where
    d.mkdir()
    write_files(d)
    m = manifest()
    m = rc2.freeze(m, d, "t") if status == "SYNTHETIC_FROZEN_FOR_HELDOUT" else m
    (d/"syn_v2.json").write_text(json.dumps(m))
    raw = _recipe("syn_v2")
    raw["heldout_verdict"] = verdict
    with pytest.raises((rm.RubberRecipeError, rc1.CalibrationManifestError)):     # schema refusal by the v1 loader
        rm.parse_recipe(raw, manifest_dir=d)
    with pytest.raises(rc2.CalibrationManifestV2Error):
        rc2.real_calibrated_gate_v2(d/"syn_v2.json", d/"no_ledger.json", thresholds_sha256="c"*64)


def test_v1_hole_is_closed_by_the_gate_directory_tripwire(tmp_path):
    """The hole the plan found: the same synthetic data rewritten as a v1 manifest (no origin field) passes the
    unchanged v1 gate. The tripwire refuses any such file in the gate directory; the shipped gate directory is clean."""
    write_files(tmp_path)
    v1 = {"schema": rc1.SCHEMA, "dataset_id": "masquerade", "status": "PILOT", "units": "SI",
          "specimens": [{"specimen_id": "S1", "material_label": "x", "supplier": "-", "batch": "-",
                         "dimensions_m": {"length": 0.06, "width": 0.012, "thickness": 0.004},
                         "dimension_uncertainty_m": 2e-5, "mass_kg": 0.0033, "mass_uncertainty_kg": 5e-7,
                         "temperature_c": 21.0, "preconditioning": {}}],
          "experiments": [{"experiment_id": "T1", "type": "stress_relaxation", "specimen_id": "S1",
                           "gauge_length_m": 0.06, "grip": "end", "hold_stretch": 1.3, "hold_time_s": 0.3,
                           "data_file": "T1_force.csv", "columns": {"time_s": "s", "displacement_m": "m",
                                                                    "force_n": "N"},
                           "temperature_c": 21.0, "split": "heldout"}],
          "splits": {"fit": [], "heldout": ["T1"], "frozen_at": None, "sha256": None},
          "boundary_conditions_note": "masquerade"}
    frozen = rc1.freeze(v1, tmp_path, "t")
    (tmp_path/"masquerade.json").write_text(json.dumps(frozen))
    recipe = rm.parse_recipe(_recipe("masquerade"), manifest_dir=tmp_path)     # the hole: v1 gate passes
    assert recipe.calibration_status == "REAL_CALIBRATED"
    problems = rc2.gate_directory_audit(tmp_path)
    assert any("masquerade.json" in p and "FROZEN_FOR_HELDOUT" in p for p in problems)
    (tmp_path/"syn_v2.json").write_text(json.dumps(manifest()))
    assert any("syn_v2.json" in p and "synthetic" in p for p in rc2.gate_directory_audit(tmp_path))
    assert rc2.gate_directory_audit() == []          # the repository's REAL gate directory


def test_no_synthetic_manifest_outside_the_synthetic_directory():
    """Every JSON file in configs/ that carries a synthetic marker and a calibration schema lives under
    configs/rubber/calibration_synthetic/."""
    offenders = []
    for path in (ROOT/"configs").rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and str(data.get("schema", "")).startswith("rcs_rubber_calibration_dataset"):
            if rc2.is_synthetic(data) and rc2.SYNTHETIC_DIR not in path.parents:
                offenders.append(str(path))
    assert offenders == []


def _real_ledger(tmp_path):
    """A real-origin frozen test manifest, a committed fit receipt (temporary git repo) and a ONE_SHOT ledger written
    by evaluate_heldout with an exact predictor (test fixture data, not a material)."""
    import subprocess
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    repo = tmp_path/"repo"
    data = repo/"data"
    data.mkdir(parents=True)
    write_files(data)
    m = rc2.freeze(manifest("real_measurement"), data, "t")
    (data/"real.json").write_text(json.dumps(m))
    rec = {"schema": rcf.FIT_RECEIPT_SCHEMA, "theta_hat": {"mu_eq_pa": 2.6e5, "g1": 0.1, "tau1_s": 0.01, "g2": 0.1,
                                                          "tau2_s": 0.1, "mu_table": 0.5},
           "objective": {"manifest_freeze_sha256": m["splits"]["sha256"], "fit_experiments": ["T1"]}}
    (repo/"fit.json").write_text(json.dumps(rec))
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "fixture"]):
        subprocess.run(cmd, cwd=repo, check=True)
    exact = {e: {r: rce.read_channels(data/f["file"]) for r, f in x["data_files"].items()}
             for x in m["experiments"] for e in [x["experiment_id"]]}
    out = rcf.evaluate_heldout(data/"real.json", [repo/"fit.json"], tmp_path/"ledger",
                               predict=lambda th, exps: {e["id"]: exact[e["id"]] for e in exps},
                               thresholds={"T3": {"M1": 0.01, "M2": 0.01}}, metric_settings={},
                               registry_dir=tmp_path/"registry", thresholds_sha256="c"*64)
    assert out["verdict"] == "PASS"
    return repo, data, tmp_path/"ledger"/f"heldout_{m['splits']['sha256']}.json", m


def test_real_gate_v2_positive_control_and_refusals(tmp_path):
    """F3 (P5.7 review): the gate reads and checks the ledger file itself (schema, ONE_SHOT, verdict, freeze,
    thresholds SHA, the global registry entry, committed fit receipts); a hand-written dict is refused."""
    repo, data, ledger, m = _real_ledger(tmp_path)
    kw = dict(thresholds_sha256="c"*64, registry_dir=tmp_path/"registry", repo_root=repo)
    assert rc2.real_calibrated_gate_v2(data/"real.json", ledger, **kw)
    with pytest.raises(rc2.CalibrationManifestV2Error, match="ledger FILE"):
        rc2.real_calibrated_gate_v2(data/"real.json", {"verdict": "PASS", "manifest_freeze_sha256":
                                                       m["splits"]["sha256"], "validation_kind": rc2.VALIDATION_REAL},
                                    **kw)
    with pytest.raises(rc2.CalibrationManifestV2Error, match="other thresholds"):
        rc2.real_calibrated_gate_v2(data/"real.json", ledger, **{**kw, "thresholds_sha256": "d"*64})
    forged = json.loads(ledger.read_text())
    forged["label"] = "ONE_SHOT"
    copy_path = tmp_path/"forged.json"
    copy_path.write_text(json.dumps(forged))            # same content class, but not the registered ledger file
    with pytest.raises(rc2.CalibrationManifestV2Error, match="registry"):
        rc2.real_calibrated_gate_v2(data/"real.json", copy_path, **kw)
    audit = dict(forged, label="AUDIT_RERUN (not the one-shot result)", audit_rerun="x")
    (tmp_path/"audit.json").write_text(json.dumps(audit))
    with pytest.raises(rc2.CalibrationManifestV2Error, match="ONE_SHOT"):
        rc2.real_calibrated_gate_v2(data/"real.json", tmp_path/"audit.json", **kw)
    with pytest.raises(rc2.CalibrationManifestV2Error, match="registry"):
        rc2.real_calibrated_gate_v2(data/"real.json", ledger, **{**kw, "registry_dir": tmp_path/"empty"})
    import subprocess
    (repo/"fit.json").write_text((repo/"fit.json").read_text()+" ")       # receipt differs from the evaluated one
    with pytest.raises(rc2.CalibrationManifestV2Error, match="not committed"):
        rc2.real_calibrated_gate_v2(data/"real.json", ledger, **kw)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-am", "edit"], cwd=repo,
                   check=True)
    with pytest.raises(rc2.CalibrationManifestV2Error, match="not committed"):
        rc2.real_calibrated_gate_v2(data/"real.json", ledger, **kw)
    pilot = manifest("real_measurement")
    (data/"pilot.json").write_text(json.dumps(pilot))
    with pytest.raises(rc2.CalibrationManifestV2Error, match="FROZEN_FOR_HELDOUT"):
        rc2.real_calibrated_gate_v2(data/"pilot.json", ledger, **kw)
    (data/"v1.json").write_text((ROOT/"configs"/"rubber"/"calibration"/"none_v1.json").read_text())
    with pytest.raises(rc2.CalibrationManifestV2Error, match="v1 manifest has no data origin"):
        rc2.real_calibrated_gate_v2(data/"v1.json", ledger, **kw)


def test_column_units_are_bound_to_names(tmp_path):
    """F10: a column named *_n must be in N, *_m in m, *_s in s."""
    write_files(tmp_path)
    m = manifest()
    m["experiments"][0]["data_files"]["force"]["columns"]["force_n"] = "m"
    with pytest.raises(rc2.CalibrationManifestV2Error, match="must have unit 'N'"):
        rc2.validate_manifest(m, tmp_path)


def test_v1_module_and_gate_unchanged():
    """The v1 loader and the REAL gate code are not modified by P2-5 (pins taken at the P2-5 base 77f5a99: the whole
    v1 module, and the REAL_CALIBRATED block of ``rubber_material.parse_recipe``)."""
    import hashlib
    src = ROOT/"src"/"robotic_cutting_sim"
    assert hashlib.sha256((src/"rubber_calibration.py").read_bytes()).hexdigest() == (
        "790f41d080445fff44373008f2ef069dda44b8771215c8164817f433cc8f2eb4")
    text = (src/"rubber_material.py").read_text()
    i = text.index('    if status == "REAL_CALIBRATED":')
    block = text[i:text.index("    elif verdict is not None", i)]
    assert hashlib.sha256(block.encode()).hexdigest() == "4624225fe3d97ed75e2485f8c0701fbab27e3fa8b420eafae1ab7052a77ad158"
