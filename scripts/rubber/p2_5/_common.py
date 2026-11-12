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


class GPULockPerEvaluation:
    """Blocking flock on rcs_rl_runner/gpu<N>.lock held only around each forward evaluation (v2 rule: queued jobs
    interleave between evaluations). ``clock()`` returns the cumulative hold time in seconds (the v2 budget clock).
    After every acquisition the free memory must be >= ``min_free_mib`` (else BLOCKED)."""

    def __init__(self, index, min_free_mib=4096):
        import fcntl
        self.fcntl, self.index, self.min_free = fcntl, int(index), min_free_mib
        self.path = f"/mnt/data/ningan/lab_records/rcs_rl_runner/gpu{self.index}.lock"
        self.held_s, self.acquisitions, self.wait_s, self._t = 0.0, 0, 0.0, None
        self.samples = []

    def __enter__(self):
        import time
        self.fh = open(self.path, "a")
        w0 = time.monotonic()
        self.fcntl.flock(self.fh, self.fcntl.LOCK_EX)
        self.wait_s += time.monotonic()-w0
        self.acquisitions += 1
        self._t = time.monotonic()
        if self.acquisitions <= 3 or self.acquisitions % 10 == 0:
            self.samples.append(gpu_sample())
        free = gpu_free_mib(self.index)
        if free is None or free < self.min_free:
            self.__exit__(None, None, None)
            raise SystemExit(f"BLOCKED: GPU{self.index} free {free} MiB < {self.min_free} MiB after acquiring the lock")
        return self

    def __exit__(self, *exc):
        import time
        if self._t is not None:
            self.held_s += time.monotonic()-self._t
            self._t = None
        self.fcntl.flock(self.fh, self.fcntl.LOCK_UN)
        self.fh.close()
        return False

    def clock(self):
        import time
        return self.held_s+((time.monotonic()-self._t) if self._t is not None else 0.0)

    def receipt(self):
        return {"lock": self.path, "mode": "blocking flock per forward evaluation", "held_s": self.held_s,
                "acquisitions": self.acquisitions, "wait_s": self.wait_s, "samples": self.samples}


class LockedObjective:
    """Objective proxy: every forward population runs under the per-evaluation GPU lock."""

    def __init__(self, objective, lock):
        self.objective, self.lock = objective, lock

    def __getattr__(self, name):
        return getattr(self.objective, name)

    def residuals_population(self, thetas):
        with self.lock:
            return self.objective.residuals_population(thetas)
