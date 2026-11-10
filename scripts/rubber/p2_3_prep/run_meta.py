"""GPU run metadata of the P2-3 prep runners (review F10): GPU driver version, dirty-tree flag and protocol SHA-256s,
merged into each run's meta.json. No P2-3 protocol is declared yet (``rubber_visc_tests_v1`` follows the review), so the
SHAs are those of the current protocol ``rubber_phys_tests_v4.json`` and of ``workload_e_v1.json``."""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_FILES = ("configs/rubber/rubber_phys_tests_v4.json", "configs/rubber/workload_e_v1.json")


def run_meta():
    def sh(*cmd):
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = sh("git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests")
    return {"commit": sh("git", "rev-parse", "HEAD"),
            "dirty_tree": bool(status), "dirty_paths": status.splitlines(),
            "gpu_driver_version": sh("nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"),
            "nvidia_smi_header": next((l.strip(" |") for l in sh("nvidia-smi").splitlines() if "Driver Version" in l), None),
            "protocol_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PROTOCOL_FILES}}
