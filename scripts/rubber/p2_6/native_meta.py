"""Run metadata and GPU sharing samples for the P6.4 / P6.5 Native rubber scripts (RL-10 fields: commit, dirty tree,
protocol and task config SHA-256, SHA-256 of every rubber module and of the Native host, Warp / Python / driver, GPU
name / index, nvidia-smi sharing samples, command). Stdlib only, plus warp when it is already imported."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT/"src"/"robotic_cutting_sim"
SOURCES = ("native_rubber_host.py", "rubber_native_compare.py", "rubber_l3.py", "live_rubber_material.py",
           "rubber_contact.py", "rubber_contact_envelope.py", "tlmpm_rubber_contact_v1.py", "tlmpm_rubber_v1.py",
           "tlmpm_rubber_visco_v1.py", "rubber_self_penetration.py", "rubber_material.py", "rubber_scene.py",
           "rubber_constitutive.py", "rubber_visco_constitutive.py", "rubber_constraint_work.py",
           "rl/rubber_family.py", "usd_surface_bridge.py", "presentation_camera.py", "workcell.py",
           "isaac_sim_runner.py", "live_material.py")


def sh(*cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources_sha256():
    return {name: (sha256(PKG/name) if (PKG/name).is_file() else None) for name in SOURCES}


def smi(gpu):
    q = sh("nvidia-smi", f"--id={gpu}", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,"
           "utilization.gpu", "--format=csv,noheader,nounits")
    index, name, uuid, driver, total, used, util = [x.strip() for x in q.split(",")]
    apps = [a for a in sh("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                          "--format=csv,noheader,nounits").splitlines() if a.startswith(uuid)]
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "index": int(index), "name": name, "uuid": uuid,
            "driver": driver, "total_mib": float(total), "used_mib": float(used),
            "free_mib": float(total)-float(used), "util_pct": float(util),
            "compute_apps": [dict(zip(("pid", "process_name", "used_mib"), [x.strip() for x in a.split(",")[1:]]))
                             for a in apps]}


def run_meta(gpu, *, protocol, task_config, extra=None):
    status = sh("git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "configs", "tests")
    meta = {"command": [sys.executable, *sys.argv], "cwd": str(ROOT), "pid": os.getpid(),
            "commit": sh("git", "rev-parse", "HEAD"), "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tree": bool(status), "dirty_paths": status.splitlines(),
            "protocol": {"path": str(Path(protocol).relative_to(ROOT)), "sha256": sha256(protocol)},
            "task_config": {"path": str(Path(task_config).relative_to(ROOT)), "sha256": sha256(task_config)},
            "sources_sha256": sources_sha256(), "python": platform.python_version(),
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "TMPDIR",
                                                   "OMNI_KIT_ACCEPT_EULA")},
            "gpu_index": gpu, "gpu_start": smi(gpu), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if "warp" in sys.modules:
        wp = sys.modules["warp"]
        meta["warp"] = {"version": getattr(wp.config, "version", None), "file": getattr(wp, "__file__", None)}
    meta.update(extra or {})
    return meta


class SmiSampler:
    """nvidia-smi samples of one GPU every ``period_s`` in a daemon thread (sharing label, job VRAM peak)."""

    def __init__(self, gpu, period_s=5.0, pid=None):
        self.gpu, self.period, self.pid = gpu, float(period_s), os.getpid() if pid is None else pid
        self.samples, self._stop = [], threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(smi(self.gpu))
            except Exception as exc:          # sampling must never kill the run
                self.samples.append({"error": repr(exc)})
            self._stop.wait(self.period)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=30)
        return self.summary()

    def summary(self):
        ok = [s for s in self.samples if "compute_apps" in s]
        own = [float(a["used_mib"]) for s in ok for a in s["compute_apps"] if str(a["pid"]) == str(self.pid)]
        others = sorted({(a["pid"], a["process_name"]) for s in ok for a in s["compute_apps"]
                         if str(a["pid"]) != str(self.pid)})
        return {"samples": len(ok), "period_s": self.period, "job_pid": self.pid,
                "job_vram_peak_mib": max(own) if own else None,
                "gpu_used_peak_mib": max((s["used_mib"] for s in ok), default=None),
                "gpu_util_max_pct": max((s["util_pct"] for s in ok), default=None),
                "other_compute_processes": [{"pid": p, "process_name": n} for p, n in others],
                "sharing": "SHARED_GPU" if others else "EXCLUSIVE_AT_SAMPLES"}


def write_new(path, payload):
    """Write JSON to a path that must not exist (never overwrite)."""
    with open(path, "x") as f:
        json.dump(payload, f, indent=1, default=_default)


def _default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")
