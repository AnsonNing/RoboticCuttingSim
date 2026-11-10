"""Rubber calibration-dataset manifest ``rcs_rubber_calibration_dataset_v1`` (ADR 0023 §14.2; task R0.2).

Loader and validator only: no fitting happens here (P2-5). A manifest lists specimens and experiments with their
boundary conditions, data files (CSV with declared columns and units) and the fit / held-out split. Validation:
schema and status words, every experiment's boundary conditions, CSV headers, units and strictly increasing time,
disjoint splits, and the freeze hash (SHA-256 over the canonical manifest and every data file) that must be written
before any held-out evaluation. The manifest shipped now is ``none_v1`` with ``NO_REAL_DATA``.
"""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "rcs_rubber_calibration_dataset_v1"
STATUSES = ("NO_REAL_DATA", "PILOT", "FROZEN_FOR_HELDOUT")
UNITS = {"s", "m", "N", "-", "Pa", "kg", "rad", "deg"}
_TOP_KEYS = {"schema", "dataset_id", "status", "units", "specimens", "experiments", "splits",
             "boundary_conditions_note"}
_SPECIMEN_KEYS = {"specimen_id", "material_label", "supplier", "batch", "dimensions_m", "dimension_uncertainty_m",
                  "mass_kg", "mass_uncertainty_kg", "temperature_c", "preconditioning"}
_SPLIT_KEYS = {"fit", "heldout", "frozen_at", "sha256"}
# Required fields per experiment type (beyond experiment_id, type, specimen_id, split). The boundary-condition
# fields (grip/clamp/counterface) are mandatory: ADR §14.2 "each experiment's boundary conditions present".
EXPERIMENT_FIELDS = {
    "tensile_load_unload": {"gauge_length_m", "grip", "crosshead_speed_m_s", "max_stretch", "cycles", "data_file",
                            "columns", "sensor", "temperature_c"},
    "stress_relaxation": {"gauge_length_m", "grip", "hold_stretch", "hold_time_s", "data_file", "columns",
                          "temperature_c"},
    "cantilever_rebound_video": {"clamp", "release", "video_file", "fps", "camera_calibration_file", "markers",
                                 "tracked_file", "columns", "temperature_c"},
    "gravity_sag_static": {"clamp", "free_length_m", "tip_drop_m", "temperature_c"},
    "friction_sliding": {"counterface", "normal_load_n", "sliding_speed_m_s", "data_file", "columns",
                         "temperature_c"},
    "incline_slip_onset": {"counterface", "slip_angle_deg", "temperature_c"},
}
BOUNDARY_FIELDS = {"tensile_load_unload": "grip", "stress_relaxation": "grip", "cantilever_rebound_video": "clamp",
                   "gravity_sag_static": "clamp", "friction_sliding": "counterface",
                   "incline_slip_onset": "counterface"}
DATA_FILE_FIELDS = ("data_file", "tracked_file")


class CalibrationManifestError(ValueError):
    pass


def _refuse(message):
    raise CalibrationManifestError(message)


def _keys(obj, allowed, where, required=None):
    if not isinstance(obj, dict):
        _refuse(f"{where} must be an object")
    unknown = set(obj)-set(allowed)
    if unknown:
        _refuse(f"{where}: unknown keys {sorted(unknown)}")
    missing = set(allowed if required is None else required)-set(obj)
    if missing:
        _refuse(f"{where}: missing keys {sorted(missing)}")


def canonical_bytes(manifest):
    """Canonical JSON of the manifest with ``splits.sha256`` blanked (the hash input)."""
    data = copy.deepcopy(manifest)
    data["splits"]["sha256"] = None
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def freeze_hash(manifest, directory):
    """SHA-256 over the canonical manifest followed by every referenced data file (sorted by name)."""
    h = hashlib.sha256(canonical_bytes(manifest))
    for name in sorted(_data_files(manifest)):
        path = Path(directory)/name
        if not path.is_file():
            _refuse(f"data file {name!r} missing")
        h.update(name.encode()+b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def _data_files(manifest):
    files = []
    for exp in manifest.get("experiments", []):
        for key in DATA_FILE_FIELDS:
            if exp.get(key):
                files.append(exp[key])
    return files


def check_csv(path, columns):
    """Header must equal the declared columns (in order), units must be known, every value finite, and the
    ``time_s`` column (if declared) strictly increasing. Returns the number of data rows."""
    if not isinstance(columns, dict) or not columns:
        _refuse(f"{path}: columns must be a non-empty {{name: unit}} object")
    for name, unit in columns.items():
        if unit not in UNITS:
            _refuse(f"{path}: column {name!r} has unknown unit {unit!r}")
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != list(columns):
        _refuse(f"{path}: header {rows[0] if rows else None} != declared columns {list(columns)}")
    previous = -math.inf
    count = 0
    time_index = rows[0].index("time_s") if "time_s" in rows[0] else None
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(columns):
            _refuse(f"{path}:{line}: {len(row)} values, expected {len(columns)}")
        try:
            values = [float(v) for v in row]
        except ValueError:
            _refuse(f"{path}:{line}: non-numeric value")
        if not all(math.isfinite(v) for v in values):
            _refuse(f"{path}:{line}: non-finite value")
        if time_index is not None:
            if values[time_index] <= previous:
                _refuse(f"{path}:{line}: time is not strictly increasing")
            previous = values[time_index]
        count += 1
    if count == 0:
        _refuse(f"{path}: no data rows")
    return count


def validate_manifest(manifest, directory=None, *, check_files=True):
    """Validate a manifest dict. ``directory`` holds the data files (required when files are checked)."""
    _keys(manifest, _TOP_KEYS, "manifest")
    if manifest["schema"] != SCHEMA:
        _refuse(f"schema must be {SCHEMA!r}")
    if manifest["status"] not in STATUSES:
        _refuse(f"status {manifest['status']!r} not in {STATUSES}")
    if manifest["units"] != "SI":
        _refuse("units must be 'SI'")
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"]:
        _refuse("dataset_id must be a non-empty string")
    specimens = manifest["specimens"]
    experiments = manifest["experiments"]
    if not isinstance(specimens, list) or not isinstance(experiments, list):
        _refuse("specimens and experiments must be lists")
    splits = manifest["splits"]
    _keys(splits, _SPLIT_KEYS, "splits")
    if manifest["status"] == "NO_REAL_DATA":
        if specimens or experiments or splits["fit"] or splits["heldout"]:
            _refuse("NO_REAL_DATA manifests have empty specimens, experiments and splits")
        if splits["frozen_at"] is not None or splits["sha256"] is not None:
            _refuse("NO_REAL_DATA manifests are not frozen")
        return {"status": "NO_REAL_DATA", "experiments": 0, "rows": {}}
    ids = set()
    for i, sp in enumerate(specimens):
        _keys(sp, _SPECIMEN_KEYS, f"specimens[{i}]")
        dims = sp["dimensions_m"]
        _keys(dims, {"length", "width", "thickness"}, f"specimens[{i}].dimensions_m")
        if not all(isinstance(v, (int, float)) and v > 0 for v in dims.values()):
            _refuse(f"specimens[{i}].dimensions_m must be positive")
        if sp["mass_kg"] <= 0:
            _refuse(f"specimens[{i}].mass_kg must be positive")
        if sp["specimen_id"] in ids:
            _refuse(f"duplicate specimen_id {sp['specimen_id']!r}")
        ids.add(sp["specimen_id"])
    exp_ids, rows = set(), {}
    for i, exp in enumerate(experiments):
        where = f"experiments[{i}]"
        kind = exp.get("type")
        if kind not in EXPERIMENT_FIELDS:
            _refuse(f"{where}.type {kind!r} not in {sorted(EXPERIMENT_FIELDS)}")
        base = {"experiment_id", "type", "specimen_id", "split"}
        optional = {"cycles", "crosshead_speed_m_s", "max_stretch", "sensor", "markers"} & EXPERIMENT_FIELDS[kind]
        _keys(exp, base | EXPERIMENT_FIELDS[kind], where, required=base | (EXPERIMENT_FIELDS[kind]-optional))
        if exp["specimen_id"] not in ids:
            _refuse(f"{where}: unknown specimen_id {exp['specimen_id']!r}")
        if exp["experiment_id"] in exp_ids:
            _refuse(f"duplicate experiment_id {exp['experiment_id']!r}")
        exp_ids.add(exp["experiment_id"])
        boundary = exp[BOUNDARY_FIELDS[kind]]
        if boundary in (None, "", {}):
            _refuse(f"{where}: boundary condition {BOUNDARY_FIELDS[kind]!r} missing")
        if exp["split"] not in ("fit", "heldout"):
            _refuse(f"{where}.split must be 'fit' or 'heldout'")
        if check_files:
            if directory is None:
                _refuse("a data directory is required to check files")
            for key in DATA_FILE_FIELDS:
                if exp.get(key):
                    rows[exp["experiment_id"]] = check_csv(Path(directory)/exp[key], exp["columns"])
    fit, held = set(splits["fit"]), set(splits["heldout"])
    if fit & held:
        _refuse(f"splits are not disjoint: {sorted(fit & held)}")
    if (fit | held) - exp_ids:
        _refuse(f"splits name unknown experiments {sorted((fit | held)-exp_ids)}")
    for exp in experiments:
        listed = "fit" if exp["experiment_id"] in fit else "heldout" if exp["experiment_id"] in held else None
        if listed != exp["split"]:
            _refuse(f"experiment {exp['experiment_id']!r}: split field {exp['split']!r} != splits listing {listed!r}")
    if manifest["status"] == "FROZEN_FOR_HELDOUT":
        if not splits["frozen_at"] or not splits["sha256"]:
            _refuse("FROZEN_FOR_HELDOUT requires splits.frozen_at and splits.sha256")
        if check_files and freeze_hash(manifest, directory) != splits["sha256"]:
            _refuse("freeze hash mismatch: manifest or data files changed after freezing")
    return {"status": manifest["status"], "experiments": len(experiments), "rows": rows}


def freeze(manifest, directory, frozen_at):
    """Return a frozen copy (status FROZEN_FOR_HELDOUT, frozen_at, sha256). Validates first."""
    data = copy.deepcopy(manifest)
    data["status"] = "PILOT"
    validate_manifest(data, directory)
    data["status"] = "FROZEN_FOR_HELDOUT"
    data["splits"]["frozen_at"] = str(frozen_at)
    data["splits"]["sha256"] = None
    data["splits"]["sha256"] = freeze_hash(data, directory)
    return data


def load_manifest(path, *, check_files=True):
    path = Path(path)
    manifest = json.loads(path.read_text())
    validate_manifest(manifest, path.parent, check_files=check_files)
    return manifest
