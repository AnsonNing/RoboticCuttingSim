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
        if isinstance(value, np.generic):
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
        def captured(name):
            value = getattr(e, name, None) or ""
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        return {"command": args, "returncode": None, "stdout": captured("stdout"),
                "stderr": captured("stderr"), "error": str(e)}


def _successful_output(receipt):
    return receipt["stdout"] if receipt["returncode"] == 0 else None


def manifest(config, backend):
    packages = {}
    for name in ["newton", "warp-lang", "numpy", "scipy", "isaaclab", "isaacsim", "torch", "usd-core",
                 "mujoco", "mujoco-warp", "newton-usd-schemas", "ovrtx", "ovstage"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for name in ["newton", "CRESSim-MPM", "IsaacLab"]:
        checkout = ROOT/".deps"/name
        if checkout.is_dir():
            checks = {"diff": command(["git", "diff", "HEAD"], cwd=checkout),
                      "revision": command(["git", "rev-parse", "HEAD"], cwd=checkout)}
            diff, revision = (_successful_output(checks[key]) for key in ("diff", "revision"))
            sources[name] = {"sha": revision.strip() if revision is not None else None,
                             "tracked_patch_sha256": hashlib.sha256(diff.encode()).hexdigest()
                                                     if diff is not None else None,
                             "dirty": bool(diff) if diff is not None else None}
            failed = {key: value for key, value in checks.items() if value["returncode"] != 0}
            if failed:
                sources[name]["command_errors"] = failed
    git = {"revision": command(["git", "rev-parse", "HEAD"], cwd=ROOT),
           "status": command(["git", "status", "--porcelain"], cwd=ROOT)}
    revision, status = (_successful_output(git[key]) for key in ("revision", "status"))
    git_errors = {key: value for key, value in git.items() if value["returncode"] != 0}
    return {"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "command": sys.argv, "python": sys.version, "backend": backend,
            "git_sha": revision.strip() if revision is not None else None,
            "git_dirty": bool(status) if status is not None else None,
            **({"git_command_errors": git_errors} if git_errors else {}),
            "config": config, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "packages": packages, "source_checkouts":sources,
            "local_code_sha256": {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((ROOT/"src").rglob("*.py"))},
            "physics_proven": False, "vision_proven": False}


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
