"""Read-only audit; the official stack is assessed independently of standalone MPM."""
import platform
import shutil
import sys

from .recording import ROOT, command, manifest, write_json


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    commands = {"uname": ["uname", "-a"], "os": ["cat", "/etc/os-release"],
                "cpu": ["lscpu"], "ram": ["free", "-b"],
                "gpu": ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap", "--format=csv,noheader,nounits"],
                "nvidia_smi": ["nvidia-smi"], "nvcc": ["/usr/local/cuda-12.8/bin/nvcc", "--version"],
                "cmake": ["cmake", "--version"]}
    report = manifest({}, "audit")
    report["system"] = {name: command(args) for name, args in commands.items()}
    report["disk"] = {str(p): dict(zip(("total", "used", "free"), shutil.disk_usage(p)))
                      for p in [ROOT, ROOT.parent, "/mnt/data"] if __import__("pathlib").Path(p).exists()}
    report["python_executable"] = sys.executable
    report["machine"] = platform.machine()
    report["source_shas"] = {name: command(["git", "rev-parse", "HEAD"], cwd=ROOT/".deps"/name)
                             for name in ["IsaacLab", "newton", "CRESSim-MPM"]}
    gpu_lines = report["system"]["gpu"].get("stdout", "").strip().splitlines()
    driver = gpu_lines[0].split(",")[-2].strip() if gpu_lines else None
    report["compatibility"] = {
        "isaaclab_required": "v3.0.0 / Python 3.12 / torch 2.11 cu130 / Newton 1.6.1 / Warp 1.17",
        "isaacsim_required": "6.2.0", "official_linux_tested_driver": "595.58.03",
        "detected_driver": driver,
        "cuda13_driver_gate": "BLOCKED" if driver is None or int(driver.split('.')[0]) < 580 else "PASS",
        "isaacsim_runtime": "NOT_RUN" if report["packages"]["isaacsim"] else "BLOCKED",
        "sources": ["https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0",
                    "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html"],
    }
    write_json(output/"system_report.json", report)
    return report["compatibility"]
