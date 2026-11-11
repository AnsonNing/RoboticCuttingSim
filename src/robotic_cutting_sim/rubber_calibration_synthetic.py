"""Synthetic truth datasets for the P2-5 numerical validation (ADR 0023 §23.2 decision 6; plan §3.2; tasks P5.4-P5.5).

NOT a calibration: the "truth" is the simulator itself (same model family ``neo_hookean_jbar_prony_v1``, M = 2,
beta 20, the protocol's contact law) at a finer resolution (n_t 6, CFL 0.25) than the fit model (n_t 4, CFL 0.5),
sampled at the sensor timestamps and corrupted by seeded sensor noise. Same-family truth cannot reveal model-form
error against real rubber; it tests identifiability, the pipeline and generalisation under discretisation and noise.

Sealing:
- the hidden parameters theta* (and the true density) are drawn from the declared prior box with a SECRET seed
  (``os.urandom``); the noise uses a second secret seed;
- both seeds, theta*, rho* and the noise-free truth channels go to ``<run>/sealed/truth.json`` (exclusive create);
  only its SHA-256 (``sealed_truth_sha256``) and the SHA-256 of the noise seed enter the manifest;
- ``reveal`` checks the SHA before returning the sealed content; the fit / held-out API refuses every ``sealed/``
  path (``rubber_calibration_fit.guard_path``).

Measured quantities: specimen dimensions = nominal + N(0, sigma_dim), mass = rho* V_nominal + N(0, sigma_mass); the fit
model uses the nominal lattice geometry and the measured density m / V_measured. Labels: ``SIMULATOR_ONLY_UNCALIBRATED``,
``NUMERICAL_VALIDATION_SYNTHETIC``.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from . import rubber_calibration_experiments as rce
from . import rubber_calibration_v2 as rc2

SENSORS_OF_TYPE = {"tensile_load_unload": ("load_cell", "crosshead"), "stress_relaxation": ("load_cell", "crosshead"),
                   "cantilever_rebound": ("video",), "gravity_sag": ("video",),
                   "friction_sliding": ("force_plate", "video")}
BOUNDARY = {"tensile_load_unload": "end-face grips (axial; face-centre pins), crosshead on grip 1",
            "stress_relaxation": "end-face grips (axial; face-centre pins), crosshead on grip 1",
            "cantilever_rebound": "clamp: jaw x <= 0 (2 dx) fully held; tip load on the last particle layer",
            "gravity_sag": "clamp: jaw x <= 0 (2 dx) fully held",
            "friction_sliding": "block on a rigid table plane z = 0; push traction on the back particle layer"}


def secret_seed():
    """A fresh 128-bit secret seed (never printed by the library)."""
    return int.from_bytes(os.urandom(16), "little")


def seed_sha256(seed):
    return hashlib.sha256(int(seed).to_bytes(16, "little")).hexdigest()


def draw_truth(prior, seed):
    """theta* and rho* from the declared prior box: log-uniform for the log-scale parameters, uniform otherwise."""
    rng = np.random.default_rng(int(seed))
    theta = {}
    for name in ("mu_eq_pa", "g1", "tau1_s", "g2", "tau2_s", "mu_table"):
        lo, hi = prior[name]
        if name in ("mu_eq_pa", "tau1_s", "tau2_s"):
            theta[name] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        else:
            theta[name] = float(rng.uniform(lo, hi))
    lo, hi = prior["density_kg_m3"]
    return theta, float(rng.uniform(lo, hi))


def protocol_experiments(protocol):
    """Forward-model experiment dicts of the protocol (nominal specimen geometry)."""
    sp = protocol["specimens"]
    out = []
    for e in protocol["experiments"]:
        out.append({"id": e["id"], "type": e["type"], "split": e["split"], "specimen_id": e["specimen_id"],
                    "specimen": dict(sp[e["specimen_id"]]["nominal_dimensions_m"]), "loading": dict(e["loading"])})
    return out


def measured_specimens(protocol, rho, rng):
    out = []
    unc = protocol["measurement"]
    for sid in sorted(protocol["specimens"]):
        nominal = protocol["specimens"][sid]["nominal_dimensions_m"]
        V = nominal["length"]*nominal["width"]*nominal["thickness"]
        dims = {k: float(nominal[k]+rng.normal(0.0, unc["dimension_sigma_m"])) for k in ("length", "width",
                                                                                          "thickness")}
        mass = float(rho*V+rng.normal(0.0, unc["mass_sigma_kg"]))
        out.append({"specimen_id": sid, "material_label": "synthetic (simulator truth; not a material)",
                    "dimensions_m": dims, "nominal_dimensions_m": dict(nominal),
                    "dimension_uncertainty_m": unc["dimension_sigma_m"], "mass_kg": mass,
                    "mass_uncertainty_kg": unc["mass_sigma_kg"], "temperature_c": 21.0})
    return out


def sensors_for(protocol, exp_type):
    return {name: dict(protocol["sensors"][name]) for name in SENSORS_OF_TYPE[exp_type]}


def file_name(exp_id, role, realisation=None):
    return f"{exp_id}_{role}.csv" if realisation is None else f"{exp_id}_{role}_{realisation}.csv"


def noisy_channels(protocol, experiments, clean, rng):
    """One noise realisation of every experiment's clean channels (experiments in protocol order)."""
    return {e["id"]: rce.add_noise({r: c for r, c in clean[e["id"]].items() if r in ("force", "video")},
                                   sensors_for(protocol, e["type"]), rng) for e in experiments}


def write_dataset(out_dir, *, protocol, protocol_sha256, experiments, noisy, specimens, oracle, dataset_id, status,
                  synthetic, note):
    """Write the CSVs, the oracle file and the v2 manifest (status ``SYNTHETIC_PILOT``) into ``out_dir`` (must not
    exist). ``synthetic``: the manifest's synthetic block (generator commit, truth profile, sealed SHA, ...). Returns
    the manifest path. Freezing is a separate step (``rubber_calibration_v2.freeze``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    m_exps = []
    for e in experiments:
        files = {}
        for role, cols in noisy[e["id"]].items():
            name = file_name(e["id"], role)
            rce.write_channels(out_dir/name, cols)
            files[role] = {"file": name, "columns": rce.column_units(cols)}
        m_exps.append({"experiment_id": e["id"], "type": e["type"], "specimen_id": e["specimen_id"],
                       "split": e["split"], "boundary_condition": BOUNDARY[e["type"]], "loading": e["loading"],
                       "sensors": sensors_for(protocol, e["type"]), "data_files": files})
    oracle_name = "oracle_heldout.json"
    (out_dir/oracle_name).write_text(json.dumps(oracle, sort_keys=True)+"\n")
    syn = dict(synthetic)
    syn["truth_profile"] = dict(syn["truth_profile"], oracle_file=oracle_name)
    syn["protocol_sha256"] = protocol_sha256
    manifest = {"schema": rc2.SCHEMA, "dataset_id": dataset_id, "data_origin": "synthetic_simulator",
                "status": status, "units": "SI", "calibration_status": rc2.CALIBRATION_STATUS,
                "validation_kind": rc2.VALIDATION_SYNTHETIC, "specimens": specimens, "experiments": m_exps,
                "splits": {"fit": [e["id"] for e in experiments if e["split"] == "fit"],
                           "heldout": [e["id"] for e in experiments if e["split"] == "heldout"],
                           "frozen_at": None, "sha256": None},
                "synthetic": syn, "boundary_conditions_note": note}
    path = out_dir/"manifest.json"
    rc2.write_manifest(path, manifest)
    rc2.validate_manifest(manifest, out_dir)
    return path


def seal(sealed_dir, payload):
    """Write ``truth.json`` into ``sealed_dir`` (exclusive create, read-only); returns (path, sha256)."""
    sealed_dir = Path(sealed_dir)
    if sealed_dir.name != "sealed":
        raise ValueError("the sealed truth lives in a directory named 'sealed'")
    sealed_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(sealed_dir, 0o700)
    path = sealed_dir/"truth.json"
    text = json.dumps(payload, sort_keys=True)+"\n"
    with open(path, "x") as fh:
        fh.write(text)
    os.chmod(path, 0o400)
    return path, hashlib.sha256(text.encode()).hexdigest()


def reveal(sealed_path, expected_sha256, reason="reveal"):
    """Return the sealed truth after checking its SHA-256 against the manifest's ``sealed_truth_sha256``. Every read
    is appended to ``read_log.txt`` next to the sealed file (UTC, pid, reason). The seal is procedural: the file is
    read-only in an owner-only directory on the same filesystem as the fitting process."""
    import datetime
    log = Path(sealed_path).parent/"read_log.txt"
    with open(log, "a") as fh:
        fh.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} pid {os.getpid()} {reason}\n")
    data = Path(sealed_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != expected_sha256:
        raise ValueError(f"sealed truth SHA mismatch: {sha} != {expected_sha256}")
    return json.loads(data)


def jsonable_channels(ch):
    return {role: {k: np.asarray(v).tolist() for k, v in cols.items()} for role, cols in ch.items()
            if role in ("force", "video")}
