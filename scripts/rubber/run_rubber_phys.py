"""RUBBER-PHYS runner (ADR 0023 R1.6): one process per configuration, raw results first, then the frozen criteria.

    PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/run_rubber_phys.py \
        --tests 01,02,03,04,05,06,07,08,09 --device cpu --workers 24 --out artifacts/rl_ready/p2_1/<run_id>

Protocol: configs/rubber/rubber_phys_tests_v2.json (frozen before this run). The output directory must not exist
(never overwritten). Every case writes raw/<test>/<label>.json as soon as it finishes; an exception is stored as
the case result (preserved failure). ``--evaluate-only DIR`` re-applies the criteria to existing raw results.
Material status: SIMULATOR_ONLY_UNCALIBRATED (numerical verification only).
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

ROOT = Path(__file__).resolve().parents[2]


def _jobs(tests, cfg, device):
    from robotic_cutting_sim import rubber_phys as P
    jobs = []
    if "01" in tests:
        for case in cfg["tests"]["RUBBER-PHYS-01"]["setup"]["cases"]:
            jobs.append(("01", case["id"], "case_patch", dict(case=case, device=device)))
    if "08" in tests:
        for g in cfg["tests"]["RUBBER-PHYS-08"]["setup"]["gamma"]:
            jobs.append(("08", f"gamma{g:g}", "case_shear", dict(gamma=g, device=device)))
    for tid, configs in (("02", P.configs_02), ("03", P.configs_03), ("05", P.configs_05)):
        if tid in tests:
            for c in configs(cfg):
                kw = dict(L=c["L"], W=c["W"], t=c["t"], n_t=c["n_t"], mu=c["mu"], g=c["g"], device=device,
                          delta_ref=c.get("delta_ref"), label=c["label"])
                jobs.append((tid, c["label"], "case_static_sag", kw))
    if "04" in tests:
        jobs.append(("04", "dt", "case_vibration", dict(run_id="dt", device=device)))
        jobs.append(("04", "dt_half", "case_vibration_half", dict(device=device)))
    if "06" in tests:
        jobs.append(("06", "uniaxial", "case_uniaxial", dict(device=device)))
    if "07" in tests:
        jobs.append(("07", "stretch_release", "case_stretch_release", dict(device=device)))
    if "09" in tests:
        for c in P.configs_09(cfg):
            jobs.append(("09", f"{c['scenario']}_t{c['t']*1e3:g}mm_Wt{c['Wt']}", "case_records",
                         dict(scenario=c["scenario"], t=c["t"], Wt=c["Wt"], device=device)))
    return jobs


def _run_job(job, out_dir):
    tid, label, func, kw = job
    import warp as wp
    wp.config.log_level = getattr(wp, "LOG_WARNING", None) or 0
    from robotic_cutting_sim import rubber_phys as P
    cfg = P.load_config()
    start = time.perf_counter()
    try:
        if func == "case_vibration_half":
            # dt/2: exactly twice the substeps of the CFL 0.5 run at the same record interval
            from robotic_cutting_sim import rubber_material as rm
            from robotic_cutting_sim import rubber_scene as rs
            st = cfg["tests"]["RUBBER-PHYS-04"]["setup"]
            mat = P.material(st["mu_pa"])
            sc = rs.compile_strip(rs.RubberStripSpec(length=st["L_m"], width=st["W_m"], thickness=st["t_m"],
                                                     n_t=st["n_t"], case="A"))
            rule = rm.step_rule(sc.dx, st["record_interval_s"], mu0_max=mat.mu0, kappa_max=mat.kappa,
                                density_min=mat.density, cfl=0.5)
            result = P.case_vibration("dt_half", cfg, device=kw["device"],
                                      substeps=2*rule["substeps_per_control_step"])
        elif func == "case_vibration":
            result = P.case_vibration(kw["run_id"], cfg, device=kw["device"])
        elif func in ("case_uniaxial", "case_stretch_release"):
            result = getattr(P, func)(cfg, device=kw["device"])
        elif func == "case_records":
            result = P.case_records(kw["scenario"], kw["t"], kw["Wt"], cfg, device=kw["device"])
        else:
            result = getattr(P, func)(**kw)
        result = {"status": "OK", **result}
    except Exception:
        result = {"status": "ERROR", "traceback": traceback.format_exc()}
    result.update({"test": f"RUBBER-PHYS-{tid}", "label": label, "function": func,
                   "job_wall_s": time.perf_counter()-start, "pid": os.getpid()})
    path = Path(out_dir)/"raw"/tid/f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, default=float))
    return tid, label, result["status"], result["job_wall_s"]


def _evaluate(out_dir):
    from robotic_cutting_sim import rubber_phys as P
    cfg = P.load_config()
    raw = Path(out_dir)/"raw"
    summary = {}
    def load(tid):
        if not (raw/tid).is_dir():
            return None
        rows = [json.loads(p.read_text()) for p in sorted((raw/tid).glob("*.json"))]
        return rows
    evaluators = {"01": P.evaluate_01, "02": P.evaluate_02, "03": P.evaluate_03, "05": P.evaluate_05,
                  "08": P.evaluate_08, "09": P.evaluate_09}
    for tid in ("01", "02", "03", "04", "05", "06", "07", "08", "09"):
        rows = load(tid)
        name = f"RUBBER-PHYS-{tid}"
        if rows is None:
            summary[name] = {"verdict": "NOT_RUN"}
            continue
        errors = [r for r in rows if r["status"] != "OK"]
        ok_rows = [r for r in rows if r["status"] == "OK"]
        try:
            if tid in evaluators:
                verdicts, measured = evaluators[tid](ok_rows, cfg)
            elif tid == "04":
                batched = None
                bpath = Path(out_dir)/"batched_04.json"
                if bpath.exists():
                    batched = json.loads(bpath.read_text())
                verdicts, measured = P.evaluate_04(ok_rows, cfg, batched=batched)
            elif tid == "06":
                verdicts, measured = P.evaluate_06(ok_rows[0], cfg) if ok_rows else ({"run": "FAIL"}, {})
            elif tid == "07":
                verdicts, measured = P.evaluate_07(ok_rows[0], cfg) if ok_rows else ({"run": "FAIL"}, {})
        except Exception:
            verdicts, measured = {"evaluation": "FAIL"}, {"traceback": traceback.format_exc()}
        if errors:
            verdicts["all_cases_ran"] = "FAIL"
        values = set(verdicts.values())
        verdict = "FAIL" if "FAIL" in values else ("PASS" if values == {"PASS"} else
                                                  "PASS_EXCEPT_NOT_RUN" if values <= {"PASS", "NOT_RUN"} else "FAIL")
        summary[name] = {"verdict": verdict, "criteria": verdicts, "measured": measured,
                         "cases": len(rows), "errors": [{"label": e["label"], "traceback": e["traceback"][-2000:]}
                                                        for e in errors],
                         "devices": sorted({r.get("device", "?") for r in ok_rows}),
                         "case_wall_s_total": sum(r["job_wall_s"] for r in rows)}
    (Path(out_dir)/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="01,02,03,04,05,06,07,08,09")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--evaluate-only")
    args = ap.parse_args()
    if args.evaluate_only:
        print(json.dumps({k: v["verdict"] for k, v in _evaluate(args.evaluate_only).items()}, indent=1))
        return
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite existing output directory {out}")
    out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT/"src"))
    from robotic_cutting_sim import rubber_phys as P
    cfg = P.load_config()
    tests = [t.strip().zfill(2) for t in args.tests.split(",")]
    jobs = _jobs(tests, cfg, args.device)
    jobs.sort(key=lambda j: 0 if j[0] in ("02", "05", "03", "06", "07") else 1)   # long jobs first
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src", "configs"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    import warp as wp
    import numpy as np
    meta = {"command": " ".join(sys.argv), "python": sys.executable, "commit": commit, "src_or_configs_dirty": bool(dirty),
            "config": str(P.CONFIG.relative_to(ROOT)), "config_sha256": hashlib.sha256(P.CONFIG.read_bytes()).hexdigest(),
            "device": args.device, "workers": args.workers, "jobs": len(jobs), "warp": wp.config.version,
            "numpy": np.__version__, "cpu": platform.processor() or platform.machine(), "host": platform.node(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED"}
    (out/"run_meta.json").write_text(json.dumps(meta, indent=1))
    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for tid, label, status, wall in pool.starmap(_run_job, [(j, str(out)) for j in jobs], chunksize=1):
            print(f"{tid} {label} {status} {wall:.1f}s", flush=True)
    meta["wall_s"] = time.perf_counter()-start
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out/"run_meta.json").write_text(json.dumps(meta, indent=1))
    summary = _evaluate(out)
    print(json.dumps({k: v["verdict"] for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
