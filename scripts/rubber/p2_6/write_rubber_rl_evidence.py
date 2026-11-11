"""Write ``docs/evidence/rl_ready/p2_6_rubber_rl_001.json`` from the P6.3 GPU run directories and the CPU test logs
(read-only on the inputs; refuses to overwrite the output). Every GPU row keeps its verdict, its plan columns, the
SHARED_GPU / EXCLUSIVE_AT_SAMPLES label with the co-tenant samples, the driver / GPU, the dirty flag and the protocol
SHA; rows that were not run are NOT_RUN.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_6/write_rubber_rl_evidence.py \
        --gpu-run artifacts/rl_ready/p2_6/<run1> [--gpu-run ...] --cpu-tests artifacts/rl_ready/p2_6/<cpu_tests_dir>
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/"docs"/"evidence"/"rl_ready"/"p2_6_rubber_rl_001.json"
PROTOCOL_SHA = "6db5468172e0cf7c507c51b97757550442715ef80221cf818f6fb69cdd189548"
TASK_SHA = "7395e6104cab16a247dc03afb133f8bd65c96d68a0d01d058bc59e6fbbf2addb"
ROWS = ("RL01", "RL02b", "RL02c", "RL03", "RL04b", "RL05", "RL06", "RL07", "RL08")
DROP = ("receipt", "reports", "rows", "samples")


def _slim(obj, depth=0):
    """Keep the evidence compact: drop bulky per-step lists below the top level, keep scalars and summaries."""
    if isinstance(obj, dict):
        return {k: _slim(v, depth+1) for k, v in obj.items() if not (depth >= 1 and k in ("receipt", "samples"))}
    if isinstance(obj, list) and len(obj) > 64:
        return {"truncated_list_len": len(obj), "head": obj[:8]}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-run", action="append", default=[])
    ap.add_argument("--cpu-tests", action="append", default=[])
    a = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    rows, runs = {}, []
    for d in a.gpu_run:
        d = Path(d) if Path(d).is_absolute() else ROOT/d
        meta = json.loads((d/"meta.json").read_text())
        if meta["protocol_sha256"] != PROTOCOL_SHA or meta["task_config_sha256"] != TASK_SHA:
            raise SystemExit(f"{d}: protocol / task config sha mismatch")
        runs.append({"dir": str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d),
                     "meta": {k: meta.get(k) for k in ("commit", "dirty_tree", "dirty_paths", "command", "device",
                                                         "physical_gpu", "gpu", "warp", "torch", "python",
                                                         "free_mb_at_start", "started", "ended", "rows")}})
        for p in sorted((d/"rows").glob("*.json")):
            r = json.loads(p.read_text())
            share = r.get("gpu_sharing", {})
            r["gpu_sharing"] = {"label": share.get("label"), "co_tenant_pids": share.get("co_tenant_pids"),
                                "samples": share.get("samples")}
            if p.stem in rows:
                raise SystemExit(f"row {p.stem} appears in two runs")
            rows[p.stem] = {"run": runs[-1]["dir"], **_slim(r)}
    for row in ROWS:
        rows.setdefault(row, {"verdict": "NOT_RUN"})
    cpu = []
    for d in a.cpu_tests:
        d = Path(d) if Path(d).is_absolute() else ROOT/d
        for p in sorted(d.glob("*.log")):
            tail = p.read_text().strip().splitlines()[-3:]
            cpu.append({"log": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "tail": tail})
    ev = {"schema": "rcs_rl_ready_evidence_v1", "id": "p2_6_rubber_rl_001",
          "phase": "P2-6 RL-ready rubber: P6.1 / P6.2 CPU tests and P6.3 GPU rows of rubber_rl_tests_v1",
          "adr": "docs/adr/0023-rubber-material-v1.md §23.2 decision 7",
          "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
          "labels": {"coupling": "kinematic_grip_ideal_tracking", "perception": "ideal_marker_tracking",
                     "self_contact": "NOT_IMPLEMENTED_V1 (detector surface_grid_v1)",
                     "robot_feedback_coupling": "NOT_IMPLEMENTED_PHASE2"},
          "protocol": {"path": "configs/rubber/rubber_rl_tests_v2.json", "sha256": PROTOCOL_SHA},
          "task_config": {"path": "configs/rubber/rl/rubber_strip_drag_v2.json", "sha256": TASK_SHA},
          "gpu_runs": runs, "gpu_rows": rows, "cpu_tests": cpu,
          "not_in_this_evidence": {"RL-09": "Native (P6.4 / P6.5, separate task)", "RL-10 / RL-11": "P6.6"}}
    OUT.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(json.dumps({k: v.get("verdict") for k, v in rows.items()}),
          hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
