"""RUBBER-PHYS-11 refinement / GCI study under protocol v4 (R2.2, R2.3, F4), single-env reference.

    TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_2/run_refinement_v4.py \
        --device cpu --workers 24 --out artifacts/rl_ready/p2_2/<run_id>

Jobs from ``configs/rubber/rubber_phys_tests_v4.json`` (``rubber_refinement.jobs``), one process per job (longest first),
raw/<Q>/<label>.json written as soon as a job finishes (an exception is stored as that job's result), then
``rubber_refinement.evaluate`` writes summary.json (never overwritten; ``--evaluate-only DIR`` re-applies the rule).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/"configs"/"rubber"/"rubber_phys_tests_v4.json"


def _cost(job):
    qid, _, run = job
    base = {"Q1": 4.0, "Q2": 1.0, "Q3": 2.0, "Q4": 4.0}[qid]
    return base*run["n_t"]**4/run["cfl"]*(run["ppc"]/2)**3


def _run(job, out, device):
    qid, label, run = job
    import warp as wp
    wp.config.log_level = 0
    from robotic_cutting_sim import rubber_refinement as RF
    cfg = json.loads(CONFIG.read_text())
    st = cfg["tests"]["RUBBER-PHYS-11"]["setup"]
    start = time.perf_counter()
    try:
        res = RF.CASES[qid](st["observables"][qid], run, device=device, options=st["solver_options"].get(qid),
                            guard=st["stable_step_guard"])
        res = {"status": "OK", **res}
    except Exception:
        res = {"status": "ERROR", "traceback": traceback.format_exc()}
    res.update(qid=qid, label=label, run=run, job_wall_s=time.perf_counter()-start, pid=os.getpid(), device=device)
    path = Path(out)/"raw"/qid/f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, default=float))
    return qid, label, res["status"], res["job_wall_s"]


def _run_star(a):
    return _run(*a)


def evaluate(out, name="summary.json"):
    from robotic_cutting_sim import rubber_refinement as RF
    cfg = json.loads(CONFIG.read_text())
    rows = [json.loads(p.read_text()) for p in sorted((Path(out)/"raw").glob("*/*.json"))]
    verdicts, measured = RF.evaluate(rows, cfg)
    values = set(verdicts.values())
    overall = "FAIL" if "FAIL" in values else ("PASS" if values == {"PASS"} else "INCOMPLETE")
    summary = {"test": "RUBBER-PHYS-11 (protocol v4)", "verdict": overall, "criteria": verdicts, "measured": measured,
               "errors": [{"label": r["label"], "traceback": r["traceback"][-1500:]} for r in rows
                          if r["status"] != "OK"], "jobs": len(rows)}
    target = Path(out)/name
    if target.exists():
        sys.exit(f"refusing to overwrite {target}")
    target.write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--evaluate-only")
    ap.add_argument("--summary-name", default="summary.json")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT/"src"))
    if args.evaluate_only:
        s = evaluate(args.evaluate_only, args.summary_name)
        print(json.dumps({"verdict": s["verdict"], "criteria": s["criteria"]}, indent=1))
        return
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite existing output directory {out}")
    out.mkdir(parents=True)
    from robotic_cutting_sim import rubber_refinement as RF
    import warp as wp
    cfg = json.loads(CONFIG.read_text())
    jobs = sorted(RF.jobs(cfg), key=_cost, reverse=True)
    meta = {"command": " ".join(sys.argv), "python": sys.executable,
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "src_or_configs_dirty": bool(subprocess.run(["git", "status", "--porcelain", "src", "configs"], cwd=ROOT,
                                                        capture_output=True, text=True).stdout.strip()),
            "config": str(CONFIG.relative_to(ROOT)), "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "device": args.device, "workers": args.workers, "jobs": len(jobs), "warp": wp.config.version,
            "host": platform.node(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED"}
    (out/"run_meta.json").write_text(json.dumps(meta, indent=1))
    start = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for qid, label, status, wall in pool.imap_unordered(_run_star, [(j, str(out), args.device) for j in jobs],
                                                            chunksize=1):
            print(f"{label} {status} {wall:.0f}s", flush=True)
    meta.update(wall_s=time.perf_counter()-start, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out/"run_meta.json").write_text(json.dumps(meta, indent=1))
    s = evaluate(out)
    print(json.dumps({"verdict": s["verdict"], "criteria": s["criteria"]}, indent=1))


if __name__ == "__main__":
    main()
