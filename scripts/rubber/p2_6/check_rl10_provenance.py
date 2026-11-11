"""RL-10 provenance check of the P2-6 runs (rubber_rl_tests_v2 / v3 / v4; read-only, writes one JSON).

For every formal GPU run directory: the run meta carries commit, dirty flag (must be False), protocol and task-config
SHA-256, the sha256 of each rubber module, Warp / Torch / Python, GPU name + driver + index, command and start time; every
row carries the SHARED / EXCLUSIVE label with nvidia-smi samples; the protocol and task config were committed (their
declaration commit) before the run started and the run commit contains them with the pinned bytes; the module hashes
recorded equal the files at the run commit. The evidence writers refuse to overwrite.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_6/check_rl10_provenance.py --out artifacts/rl_ready/p2_6/<file>.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P26 = ROOT/"artifacts"/"rl_ready"/"p2_6"
DECL = {"configs/rubber/rubber_rl_tests_v2.json": "400f5c5", "configs/rubber/rl/rubber_strip_drag_v2.json": "400f5c5",
        "configs/rubber/rubber_rl_tests_v3.json": "ba41461"}
WRITERS = ("scripts/rubber/p2_6/write_rubber_rl_evidence.py", "scripts/rubber/p2_6/write_contact_envelope_v2.py",
           "scripts/rubber/p2_6/write_contact_envelope_v3.py", "scripts/rubber/p2_6/run_rubber_rl_gpu.py",
           "scripts/rubber/p2_6/run_contact_envelope_v1.py", "scripts/rubber/p2_6/run_contact_envelope_v2.py")
META_KEYS = ("commit", "dirty_tree", "protocol_sha256", "task_config_sha256", "sources_sha256", "warp", "torch",
             "python", "gpu", "physical_gpu", "command", "started")


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def commit_time(c):
    return datetime.fromisoformat(git("show", "-s", "--format=%cI", c))


def blob_sha(commit, path):
    p = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, cwd=ROOT)
    return hashlib.sha256(p.stdout).hexdigest() if p.returncode == 0 else None


def check_run(d, protocol_path, extra_decl=()):
    meta = json.loads((d/"meta.json").read_text())
    out = {"run": d.name, "commit": meta.get("commit", "")[:7], "problems": []}
    out["problems"] += [f"meta missing {k}" for k in META_KEYS if k not in meta]
    if meta.get("dirty_tree") is not False:
        out["problems"].append("dirty tree")
    started = datetime.strptime(meta["started"], "%Y-%m-%dT%H:%M:%S%z")
    for path in (protocol_path, "configs/rubber/rl/rubber_strip_drag_v2.json", *extra_decl):
        decl = DECL[path]
        if commit_time(decl) >= started:
            out["problems"].append(f"{path} declared at {decl} after the run start")
        if subprocess.run(["git", "merge-base", "--is-ancestor", decl, meta["commit"]], cwd=ROOT).returncode != 0:
            out["problems"].append(f"run commit does not contain the declaration {decl} of {path}")
    if blob_sha(meta["commit"], protocol_path) != meta["protocol_sha256"]:
        out["problems"].append("protocol bytes at the run commit != recorded sha")
    if blob_sha(meta["commit"], "configs/rubber/rl/rubber_strip_drag_v2.json") != meta["task_config_sha256"]:
        out["problems"].append("task config bytes at the run commit != recorded sha")
    for name, sha in meta.get("sources_sha256", {}).items():
        if blob_sha(meta["commit"], f"src/robotic_cutting_sim/{name}") != sha:
            out["problems"].append(f"module {name} sha != file at the run commit")
    rows = sorted((d/"rows").glob("*.json"))
    out["rows"] = []
    for p in rows:
        r = json.loads(p.read_text())
        sh = r.get("gpu_sharing", {})
        ok = sh.get("label") in ("SHARED_GPU", "EXCLUSIVE_AT_SAMPLES") and bool(sh.get("samples"))
        ok &= r.get("protocol_sha256") == meta["protocol_sha256"] and r.get("dirty_tree") is False
        out["rows"].append({"row": p.stem, "verdict": r.get("verdict"), "sharing": sh.get("label"),
                            "samples": len(sh.get("samples") or []), "ok": bool(ok)})
        if not ok:
            out["problems"].append(f"row {p.stem}: sharing / protocol / dirty fields")
    out["gpu"], out["started"], out["protocol_sha256"] = meta.get("gpu"), meta["started"], meta["protocol_sha256"][:8]
    out["pass"] = not out["problems"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--v4-run", default=None)
    ap.add_argument("--v4-declaration", default=None, help="commit of the rubber_rl_tests_v4 declaration")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    runs = [(P26/n, "configs/rubber/rubber_rl_tests_v2.json") for n in ("gpu_rl_a_001", "gpu_rl_b_001", "gpu_rl_c_001",
                                                                       "gpu_rl_d_001")]
    runs.append((P26/"gpu_rl05_v3_001", "configs/rubber/rubber_rl_tests_v3.json"))
    if a.v4_run:
        DECL["configs/rubber/rubber_rl_tests_v4.json"] = a.v4_declaration
        runs.append((Path(a.v4_run) if Path(a.v4_run).is_absolute() else ROOT/a.v4_run,
                     "configs/rubber/rubber_rl_tests_v4.json"))
    res = {"schema": "rcs_rl10_provenance_v1", "head": git("rev-parse", "HEAD"),
           "declarations": {p: {"commit": c, "committed": git("show", "-s", "--format=%cI", c),
                                "sha256": blob_sha(c, p)} for p, c in DECL.items()},
           "runs": [check_run(d, p) for d, p in runs]}
    w = []
    for path in WRITERS:
        text = (ROOT/path).read_text()
        guarded = ("refusing to overwrite" in text or "exist_ok=False" in text)
        w.append({"writer": path, "refuses_overwrite": guarded})
    res["writers"] = w
    res["verdict"] = "PASS" if all(r["pass"] for r in res["runs"]) and all(x["refuses_overwrite"] for x in w) \
        else "FAIL"
    out.write_text(json.dumps(res, indent=1)+"\n")
    print(res["verdict"], [(r["run"], r["pass"], r["problems"][:3]) for r in res["runs"]])


if __name__ == "__main__":
    main()
