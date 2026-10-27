import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    def convert(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False)+"\n")


def command(args, timeout=30, cwd=None):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return {"command": args, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": args, "returncode": None, "error": str(e)}


def manifest(config, backend):
    packages = {}
    for name in ["newton", "warp-lang", "numpy", "scipy", "isaaclab", "isaacsim", "torch", "usd-core"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for name in ["newton", "CRESSim-MPM", "IsaacLab"]:
        checkout = ROOT/".deps"/name
        if checkout.is_dir():
            diff = command(["git", "diff", "HEAD"], cwd=checkout)["stdout"]
            sources[name] = {"sha":command(["git","rev-parse","HEAD"],cwd=checkout)["stdout"].strip(),
                             "tracked_patch_sha256":hashlib.sha256(diff.encode()).hexdigest(),
                             "dirty":bool(diff)}
    return {"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "command": sys.argv, "python": sys.version, "backend": backend,
            "git_sha": command(["git", "rev-parse", "HEAD"], cwd=ROOT)["stdout"].strip(),
            "git_dirty": bool(command(["git", "status", "--porcelain"], cwd=ROOT)["stdout"]),
            "config": config, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "packages": packages, "source_checkouts":sources, "physics_proven": False, "vision_proven": False}


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
