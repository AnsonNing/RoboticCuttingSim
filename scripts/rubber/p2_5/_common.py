"""Shared helpers of the P2-5 run scripts (provenance, GPU sharing samples, protocol pin, no-overwrite outputs)."""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_calib_tests_v1.json"
THRESHOLDS = ROOT/"configs"/"rubber"/"rubber_calib_thresholds_v1.json"
SYNTHETIC_DIR = ROOT/"configs"/"rubber"/"calibration_synthetic"
LABELS = {"calibration_status": "SIMULATOR_ONLY_UNCALIBRATED", "validation_kind": "NUMERICAL_VALIDATION_SYNTHETIC",
          "recipe_promoted": False}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def tree_state():
    return {"commit": git("rev-parse", "HEAD"), "clean": git("status", "--porcelain") == "",
            "branch": git("rev-parse", "--abbrev-ref", "HEAD")}


def load_protocol(expected_sha=None):
    raw = PROTOCOL.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise SystemExit(f"protocol sha {sha} != pinned {expected_sha}")
    return json.loads(raw), sha


def gpu_sample():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
                              "driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout.strip().splitlines()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=20).stdout.strip().splitlines()
    except Exception as exc:                      # recorded, never hidden
        return {"error": repr(exc)}
    return {"utc": utc(), "gpus": out, "compute_apps": apps, "own_pid": os.getpid()}


def gpu_free_mib(index):
    for line in gpu_sample().get("gpus", []):
        parts = [p.strip() for p in line.split(",")]
        if parts[0] == str(index):
            return float(parts[3])-float(parts[2])
    return None


def require_gpu(physical_index, min_free_mib=4096):
    free = gpu_free_mib(physical_index)
    if free is None or free < min_free_mib:
        raise SystemExit(f"BLOCKED: GPU{physical_index} free {free} MiB < {min_free_mib} MiB")
    return free


def environment():
    import warp as wp
    import scipy
    return {"python": platform.python_version(), "warp": wp.config.version, "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(), "argv": sys.argv,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def module_shas():
    src = ROOT/"src"/"robotic_cutting_sim"
    names = ("rubber_calibration_v2.py", "rubber_calibration_experiments.py", "rubber_calibration_fit.py",
             "rubber_calibration_synthetic.py", "tlmpm_rubber_v1.py", "tlmpm_rubber_visco_v1.py",
             "tlmpm_rubber_contact_v1.py", "rubber_contact.py", "live_rubber_material.py",
             "batched/rubber_material_batch.py", "batched/rubber_batch.py", "batched/rubber_contact_batch.py")
    return {n: hashlib.sha256((src/n).read_bytes()).hexdigest() for n in names}


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as fh:
        fh.write(json.dumps(jsonable(obj), indent=1, sort_keys=True)+"\n")
    return path


def setup_warp():
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    return wp


def experiments_by_id(protocol):
    from robotic_cutting_sim import rubber_calibration_synthetic as rcs
    return {e["id"]: e for e in rcs.protocol_experiments(protocol)}


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"{utc()} {msg}"
        print(line, flush=True)
        with open(self.path, "a") as fh:
            fh.write(line+"\n")
