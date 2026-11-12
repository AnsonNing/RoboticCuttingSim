"""Rubber calibration-dataset manifest ``rcs_rubber_calibration_dataset_v2`` (ADR 0023 §23.2 decision 6; P2-5 task P5.1).

Schema v1 (``rubber_calibration``, unchanged) has no data origin, and the ``REAL_CALIBRATED`` gate of
``rubber_material.parse_recipe`` (unchanged) checks only that a v1 manifest in ``configs/rubber/calibration/`` is
``FROZEN_FOR_HELDOUT`` with a held-out verdict PASS. A synthetic dataset written in the v1 schema could therefore pass
that gate (plan F10). This module closes the hole without touching the v1 code:

- **Schema v2** adds ``data_origin`` (``real_measurement`` | ``synthetic_simulator``), a ``sensors`` block for every
  experiment (noise sigma, resolution, sample rate, full scale), the labels ``calibration_status`` and
  ``validation_kind``, and for synthetic data a ``synthetic`` block (generator commit, truth profile, sensor model id,
  ``sealed_truth_sha256``, protocol SHA, prior box). Statuses are origin-bound: ``PILOT`` / ``FROZEN_FOR_HELDOUT`` only
  for real data, ``SYNTHETIC_PILOT`` / ``SYNTHETIC_FROZEN_FOR_HELDOUT`` only for synthetic data, ``NO_REAL_DATA`` for an
  empty real manifest. A synthetic manifest always carries ``validation_kind: NUMERICAL_VALIDATION_SYNTHETIC``.
- **Directories.** Synthetic manifests are written only under ``configs/rubber/calibration_synthetic/`` or
  ``artifacts/`` and never into the REAL gate's directory ``configs/rubber/calibration/`` (``write_manifest`` refuses).
  The v1 gate loads with the v1 loader, so a v2 manifest is refused there by schema.
- **Tripwire** (``gate_directory_audit``, run by the CPU tests): until a real-data protocol
  (``rubber_calib_tests_real_v1``) is declared, every manifest in the gate directory must be a v1 ``NO_REAL_DATA``
  manifest; anything else (a v1 manifest with data, a v2 manifest, any synthetic marker) fails CI. This is what makes
  the v1 ``REAL_CALIBRATED`` path unreachable in this repository.
- **Gate v2** (``real_calibrated_gate_v2``) is the versioned gate for future real data: it accepts only a v2
  manifest with ``data_origin: real_measurement``, status ``FROZEN_FOR_HELDOUT``, a matching freeze hash and a one-shot
  held-out receipt with verdict PASS for that freeze; it refuses every v1 manifest (origin unknown) and every
  synthetic manifest. No recipe uses it yet (NOT_RUN: no real data on this machine).

Every value derived from a synthetic manifest is ``SIMULATOR_ONLY_UNCALIBRATED`` / ``NUMERICAL_VALIDATION_SYNTHETIC``.
"""
import copy
import hashlib
import json
from pathlib import Path

from . import rubber_calibration as rc1
from .rubber_material import CALIBRATION_DIR, ROOT

SCHEMA = "rcs_rubber_calibration_dataset_v2"
ORIGINS = ("real_measurement", "synthetic_simulator")
REAL_STATUSES = ("NO_REAL_DATA", "PILOT", "FROZEN_FOR_HELDOUT")
SYNTHETIC_STATUSES = ("SYNTHETIC_PILOT", "SYNTHETIC_FROZEN_FOR_HELDOUT")
STATUSES = REAL_STATUSES+SYNTHETIC_STATUSES
FROZEN_STATUSES = ("FROZEN_FOR_HELDOUT", "SYNTHETIC_FROZEN_FOR_HELDOUT")
CALIBRATION_STATUS = "SIMULATOR_ONLY_UNCALIBRATED"
VALIDATION_SYNTHETIC = "NUMERICAL_VALIDATION_SYNTHETIC"
VALIDATION_REAL = "REAL_MEASUREMENT_HELDOUT"
SYNTHETIC_DIR = ROOT/"configs"/"rubber"/"calibration_synthetic"
GATE_DIR = CALIBRATION_DIR
REAL_PROTOCOL_DECLARED = False      # rubber_calib_tests_real_v1 is declared NOT_RUN (no real data); see the tripwire
UNITS = rc1.UNITS
EXPERIMENT_TYPES = ("tensile_load_unload", "stress_relaxation", "cantilever_rebound", "gravity_sag",
                    "friction_sliding")
SENSOR_KINDS = ("load_cell", "crosshead", "video_markers", "force_plate")
_TOP_KEYS = {"schema", "dataset_id", "data_origin", "status", "units", "calibration_status", "validation_kind",
             "specimens", "experiments", "splits", "synthetic", "boundary_conditions_note"}
_SPECIMEN_KEYS = {"specimen_id", "material_label", "dimensions_m", "nominal_dimensions_m", "dimension_uncertainty_m",
                  "mass_kg", "mass_uncertainty_kg", "temperature_c"}
_EXPERIMENT_KEYS = {"experiment_id", "type", "specimen_id", "split", "boundary_condition", "loading", "sensors",
                    "data_files"}
_SENSOR_KEYS = {"kind", "noise_sigma", "resolution", "sample_rate_hz", "full_scale", "unit"}
_SYNTHETIC_KEYS = {"generator_commit", "generator_tree_clean", "truth_profile", "sensor_model_id",
                   "sealed_truth_sha256", "protocol_sha256", "prior_box", "noise_seed_sha256"}
_SPLIT_KEYS = {"fit", "heldout", "frozen_at", "sha256"}
COLUMN_UNIT_SUFFIXES = (("_s", "s"), ("_m", "m"), ("_n", "N"), ("_pa", "Pa"), ("_kg", "kg"), ("_rad", "rad"))


class CalibrationManifestV2Error(ValueError):
    pass


def _refuse(message):
    raise CalibrationManifestV2Error(message)


def _keys(obj, allowed, where, required=None):
    if not isinstance(obj, dict):
        _refuse(f"{where} must be an object")
    unknown = set(obj)-set(allowed)
    if unknown:
        _refuse(f"{where}: unknown keys {sorted(unknown)}")
    missing = set(allowed if required is None else required)-set(obj)
    if missing:
        _refuse(f"{where}: missing keys {sorted(missing)}")


def _hex64(value, where):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _refuse(f"{where} must be a lowercase sha256 hex digest")


def canonical_bytes(manifest):
    data = copy.deepcopy(manifest)
    data["splits"]["sha256"] = None
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def data_files(manifest):
    """Data file names of a v2 manifest (``data_files``: role -> {file, columns})."""
    return [entry["file"] for exp in manifest.get("experiments", []) for entry in exp.get("data_files", {}).values()]


def hashed_files(manifest):
    """Every file covered by the freeze hash: the data files and, for synthetic data, the oracle file."""
    names = list(data_files(manifest))
    oracle = (manifest.get("synthetic") or {}).get("truth_profile", {}).get("oracle_file")
    if oracle:
        names.append(oracle)
    return names


def freeze_hash(manifest, directory):
    """SHA-256 over the canonical manifest (``splits.sha256`` blanked) and every hashed file (sorted by name)."""
    h = hashlib.sha256(canonical_bytes(manifest))
    for name in sorted(hashed_files(manifest)):
        path = Path(directory)/name
        if not path.is_file():
            _refuse(f"data file {name!r} missing")
        h.update(name.encode()+b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def _check_sensor(name, sensor, where):
    _keys(sensor, _SENSOR_KEYS, f"{where}.sensors.{name}")
    if sensor["kind"] not in SENSOR_KINDS:
        _refuse(f"{where}.sensors.{name}.kind {sensor['kind']!r} not in {SENSOR_KINDS}")
    for key in ("noise_sigma", "resolution", "sample_rate_hz"):
        v = sensor[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v >= 0 or v != v:
            _refuse(f"{where}.sensors.{name}.{key} must be a finite number >= 0")
    if sensor["sample_rate_hz"] <= 0:
        _refuse(f"{where}.sensors.{name}.sample_rate_hz must be > 0")
    if sensor["unit"] not in UNITS:
        _refuse(f"{where}.sensors.{name}.unit {sensor['unit']!r} unknown")


def validate_manifest(manifest, directory=None, *, check_files=True):
    """Validate a v2 manifest dict; returns a summary. ``directory`` holds the data files when they are checked."""
    _keys(manifest, _TOP_KEYS, "manifest", required=_TOP_KEYS-{"synthetic"})
    if manifest["schema"] != SCHEMA:
        _refuse(f"schema must be {SCHEMA!r}")
    origin, status = manifest["data_origin"], manifest["status"]
    if origin not in ORIGINS:
        _refuse(f"data_origin {origin!r} not in {ORIGINS}")
    if status not in STATUSES:
        _refuse(f"status {status!r} not in {STATUSES}")
    synthetic = origin == "synthetic_simulator"
    if synthetic and status not in SYNTHETIC_STATUSES:
        _refuse(f"a synthetic manifest must have a SYNTHETIC_* status, not {status!r}")
    if not synthetic and status not in REAL_STATUSES:
        _refuse(f"status {status!r} is reserved for synthetic manifests")
    if manifest["units"] != "SI":
        _refuse("units must be 'SI'")
    if manifest["calibration_status"] != CALIBRATION_STATUS:
        _refuse(f"calibration_status must be {CALIBRATION_STATUS!r} (a manifest never promotes a recipe)")
    expected_kind = VALIDATION_SYNTHETIC if synthetic else VALIDATION_REAL
    if manifest["validation_kind"] != expected_kind:
        _refuse(f"validation_kind must be {expected_kind!r} for data_origin {origin!r}")
    if synthetic != ("synthetic" in manifest):
        _refuse("the 'synthetic' block is required for synthetic data and refused for real data")
    if synthetic:
        syn = manifest["synthetic"]
        _keys(syn, _SYNTHETIC_KEYS, "synthetic")
        for key in ("sealed_truth_sha256", "protocol_sha256", "noise_seed_sha256"):
            _hex64(syn[key], f"synthetic.{key}")
        if not isinstance(syn["generator_commit"], str) or len(syn["generator_commit"]) < 7:
            _refuse("synthetic.generator_commit must be a git commit id")
        if syn["generator_tree_clean"] is not True:
            _refuse("synthetic.generator_tree_clean must be true (data generated from a committed clean tree)")
        if not isinstance(syn["truth_profile"], dict) or not {"n_t", "cfl"} <= set(syn["truth_profile"]):
            _refuse("synthetic.truth_profile must give at least n_t and cfl")
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"]:
        _refuse("dataset_id must be a non-empty string")
    specimens, experiments, splits = manifest["specimens"], manifest["experiments"], manifest["splits"]
    if not isinstance(specimens, list) or not isinstance(experiments, list):
        _refuse("specimens and experiments must be lists")
    _keys(splits, _SPLIT_KEYS, "splits")
    if status == "NO_REAL_DATA":
        if specimens or experiments or splits["fit"] or splits["heldout"]:
            _refuse("NO_REAL_DATA manifests have empty specimens, experiments and splits")
        if splits["frozen_at"] is not None or splits["sha256"] is not None:
            _refuse("NO_REAL_DATA manifests are not frozen")
        return {"status": status, "data_origin": origin, "experiments": 0, "rows": {}}
    ids = set()
    for i, sp in enumerate(specimens):
        _keys(sp, _SPECIMEN_KEYS, f"specimens[{i}]")
        for key in ("dimensions_m", "nominal_dimensions_m"):
            _keys(sp[key], {"length", "width", "thickness"}, f"specimens[{i}].{key}")
            if not all(isinstance(v, (int, float)) and v > 0 for v in sp[key].values()):
                _refuse(f"specimens[{i}].{key} must be positive")
        if not sp["mass_kg"] > 0:
            _refuse(f"specimens[{i}].mass_kg must be positive")
        if sp["specimen_id"] in ids:
            _refuse(f"duplicate specimen_id {sp['specimen_id']!r}")
        ids.add(sp["specimen_id"])
    exp_ids, rows = set(), {}
    for i, exp in enumerate(experiments):
        where = f"experiments[{i}]"
        _keys(exp, _EXPERIMENT_KEYS, where)
        if exp["type"] not in EXPERIMENT_TYPES:
            _refuse(f"{where}.type {exp['type']!r} not in {EXPERIMENT_TYPES}")
        if exp["specimen_id"] not in ids:
            _refuse(f"{where}: unknown specimen_id {exp['specimen_id']!r}")
        if exp["experiment_id"] in exp_ids:
            _refuse(f"duplicate experiment_id {exp['experiment_id']!r}")
        exp_ids.add(exp["experiment_id"])
        if exp["boundary_condition"] in (None, "", {}):
            _refuse(f"{where}: boundary condition missing")
        if exp["split"] not in ("fit", "heldout"):
            _refuse(f"{where}.split must be 'fit' or 'heldout'")
        if not isinstance(exp["sensors"], dict) or not exp["sensors"]:
            _refuse(f"{where}: every experiment needs a non-empty sensors block")
        for name, sensor in exp["sensors"].items():
            _check_sensor(name, sensor, where)
        files = exp["data_files"]
        if not isinstance(files, dict) or not files:
            _refuse(f"{where}.data_files must be a non-empty {{role: {{file, columns}}}} object")
        for role, entry in files.items():
            _keys(entry, {"file", "columns"}, f"{where}.data_files.{role}")
            for col, unit in (entry["columns"] or {}).items():
                want = next((u for suf, u in COLUMN_UNIT_SUFFIXES if col.endswith(suf)), None)
                if want is not None and unit != want:
                    _refuse(f"{where}.data_files.{role}: column {col!r} must have unit {want!r}, not {unit!r}")
            if Path(entry["file"]).name != entry["file"]:
                _refuse(f"{where}.data_files.{role}: data files are plain names next to the manifest")
            if check_files:
                if directory is None:
                    _refuse("a data directory is required to check files")
                try:
                    rows[f"{exp['experiment_id']}.{role}"] = rc1.check_csv(Path(directory)/entry["file"],
                                                                         entry["columns"])
                except rc1.CalibrationManifestError as exc:
                    _refuse(str(exc))
    fit, held = set(splits["fit"]), set(splits["heldout"])
    if fit & held:
        _refuse(f"splits are not disjoint: {sorted(fit & held)}")
    if (fit | held) != exp_ids:
        _refuse(f"splits must list every experiment exactly once (missing {sorted(exp_ids-(fit | held))}, "
                f"unknown {sorted((fit | held)-exp_ids)})")
    if not fit or not held:
        _refuse("both splits must be non-empty")
    for exp in experiments:
        listed = "fit" if exp["experiment_id"] in fit else "heldout"
        if listed != exp["split"]:
            _refuse(f"experiment {exp['experiment_id']!r}: split field {exp['split']!r} != splits listing {listed!r}")
    if status in FROZEN_STATUSES:
        if not splits["frozen_at"] or not splits["sha256"]:
            _refuse(f"{status} requires splits.frozen_at and splits.sha256")
        if check_files and freeze_hash(manifest, directory) != splits["sha256"]:
            _refuse("freeze hash mismatch: manifest or data files changed after freezing")
    elif splits["sha256"] is not None:
        _refuse("an unfrozen manifest has splits.sha256 = null")
    return {"status": status, "data_origin": origin, "experiments": len(experiments), "rows": rows}


def freeze(manifest, directory, frozen_at):
    """Frozen copy: status ``*FROZEN_FOR_HELDOUT`` for the manifest's origin, ``frozen_at`` and the freeze hash."""
    data = copy.deepcopy(manifest)
    synthetic = data.get("data_origin") == "synthetic_simulator"
    data["status"] = "SYNTHETIC_PILOT" if synthetic else "PILOT"
    data["splits"]["frozen_at"], data["splits"]["sha256"] = None, None
    validate_manifest(data, directory)
    data["status"] = "SYNTHETIC_FROZEN_FOR_HELDOUT" if synthetic else "FROZEN_FOR_HELDOUT"
    data["splits"]["frozen_at"] = str(frozen_at)
    data["splits"]["sha256"] = freeze_hash(data, directory)
    return data


def _inside(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


def write_manifest(path, manifest, *, overwrite=False):
    """Write a v2 manifest. Synthetic manifests are refused inside the REAL gate directory; a manifest that does not
    validate (without the files) is refused; an existing file is never overwritten unless ``overwrite``."""
    path = Path(path)
    validate_manifest(manifest, check_files=False)
    if manifest["data_origin"] == "synthetic_simulator" and _inside(path, GATE_DIR):
        _refuse(f"synthetic manifests are never written into the REAL gate directory {GATE_DIR}")
    if path.exists() and not overwrite:
        _refuse(f"{path} exists (no overwrite)")
    path.write_text(json.dumps(manifest, indent=1)+"\n")      # key order kept: CSV column order is declared
    return path


def load_manifest(path, *, check_files=True):
    path = Path(path)
    manifest = json.loads(path.read_text())
    validate_manifest(manifest, path.parent, check_files=check_files)
    return manifest


def is_synthetic(manifest):
    """True if a manifest dict carries any synthetic marker (any schema)."""
    if not isinstance(manifest, dict):
        return False
    text = json.dumps(manifest, sort_keys=True).lower()
    return (manifest.get("data_origin") == "synthetic_simulator" or "synthetic" in manifest
            or str(manifest.get("status", "")).startswith("SYNTHETIC")
            or manifest.get("validation_kind") == VALIDATION_SYNTHETIC or "synthetic_simulator" in text)


def gate_directory_audit(directory=GATE_DIR):
    """Tripwire for the v1 REAL gate: returns a list of problems (empty = clean). Until a real-data protocol is
    declared, the gate directory may hold only v1 ``NO_REAL_DATA`` manifests (and the v1 schema file)."""
    problems = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name.endswith(".schema.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except ValueError as exc:
            problems.append(f"{path.name}: not JSON ({exc})")
            continue
        if is_synthetic(data):
            problems.append(f"{path.name}: synthetic manifest in the REAL gate directory")
        if data.get("schema") != rc1.SCHEMA:
            problems.append(f"{path.name}: schema {data.get('schema')!r} (only v1 manifests are read by the gate)")
        elif data.get("status") != "NO_REAL_DATA" and not REAL_PROTOCOL_DECLARED:
            problems.append(f"{path.name}: v1 status {data.get('status')!r} before a real-data protocol is declared "
                            "(the v1 manifest has no data origin; the v1 REAL gate cannot tell real from synthetic)")
    return problems


HELDOUT_SCHEMA = "rcs_rubber_heldout_receipt_v1"


def _git_committed_sha(repo_root, path):
    """SHA-256 of ``path`` as committed at HEAD of ``repo_root`` (None if untracked)."""
    import subprocess
    rel = str(Path(path).resolve().relative_to(Path(repo_root).resolve()))
    out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=repo_root, capture_output=True)
    return hashlib.sha256(out.stdout).hexdigest() if out.returncode == 0 else None


def real_calibrated_gate_v2(manifest_path, ledger_path, *, thresholds_sha256, registry_dir=None, repo_root=ROOT):
    """Versioned REAL gate for future real data (NOT used by any recipe yet). Returns True or raises.

    Hardened after the P5.7 review (F3): the held-out evidence must be the LEDGER FILE written by
    ``rubber_calibration_fit.evaluate_heldout`` (a dict is refused), and the gate checks it itself:
    - the manifest is v2, ``data_origin: real_measurement``, ``FROZEN_FOR_HELDOUT``, freeze hash valid; every v1 and
      every synthetic manifest is refused;
    - the ledger has the held-out schema, label ``ONE_SHOT``, no audit re-run, verdict PASS, ``validation_kind``
      ``REAL_MEASUREMENT_HELDOUT``, this manifest's freeze and the expected thresholds SHA-256;
    - the global one-shot registry entry for the freeze names exactly this ledger (path and SHA-256);
    - every fit receipt in the ledger is committed at HEAD of ``repo_root`` with the SHA-256 recorded in the ledger."""
    if not isinstance(ledger_path, (str, Path)):
        _refuse("REAL gate v2: the held-out evidence must be the ledger FILE written by evaluate_heldout, "
                "not a caller-supplied object")
    path = Path(manifest_path)
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        _refuse(f"REAL gate v2: schema {data.get('schema')!r} refused (a v1 manifest has no data origin)")
    if is_synthetic(data):
        _refuse("REAL gate v2: a synthetic manifest can never yield REAL_CALIBRATED")
    validate_manifest(data, path.parent)
    if data["data_origin"] != "real_measurement" or data["status"] != "FROZEN_FOR_HELDOUT":
        _refuse("REAL gate v2: needs data_origin real_measurement and status FROZEN_FOR_HELDOUT")
    freeze = data["splits"]["sha256"]
    lpath = Path(ledger_path)
    raw = lpath.read_bytes()
    led = json.loads(raw)
    if (led.get("schema") != HELDOUT_SCHEMA or led.get("label") != "ONE_SHOT" or led.get("audit_rerun")
            or led.get("verdict") != "PASS"):
        _refuse("REAL gate v2: the ledger is not a ONE_SHOT held-out receipt with verdict PASS")
    if led.get("manifest_freeze_sha256") != freeze:
        _refuse("REAL gate v2: the held-out receipt is for another manifest freeze")
    if led.get("validation_kind") != VALIDATION_REAL:
        _refuse("REAL gate v2: the held-out receipt is not a real-data evaluation")
    if led.get("thresholds_sha256") != thresholds_sha256:
        _refuse("REAL gate v2: the held-out receipt used other thresholds")
    from .rubber_calibration_fit import HELDOUT_REGISTRY_DIR
    reg = Path(HELDOUT_REGISTRY_DIR if registry_dir is None else registry_dir)/f"{freeze}.json"
    if not reg.is_file():
        _refuse("REAL gate v2: no one-shot registry entry for this freeze")
    entry = json.loads(reg.read_text())
    if entry.get("ledger_sha256") != hashlib.sha256(raw).hexdigest() or Path(entry["ledger"]) != lpath.resolve():
        _refuse("REAL gate v2: the registry's one-shot ledger for this freeze is a different file")
    for fit in led.get("fits", []):
        work = hashlib.sha256(Path(fit["fit_receipt"]).read_bytes()).hexdigest()
        if _git_committed_sha(repo_root, fit["fit_receipt"]) != fit["fit_receipt_sha256"] or work != fit[
                "fit_receipt_sha256"]:
            _refuse(f"REAL gate v2: fit receipt {fit['fit_receipt']} is not committed with the recorded SHA-256")
    return True
