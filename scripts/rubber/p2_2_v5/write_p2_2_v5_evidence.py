"""Compact protocol-v5 evidence (docs/evidence/rl_ready/p2_2_v5_001.json) from the raw outputs in
artifacts/rl_ready/p2_2_v5.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_2_v5/write_p2_2_v5_evidence.py [--out docs/evidence/rl_ready/p2_2_v5_001.json]

Reads only; never overwrites an existing evidence file. Missing raw outputs are recorded as NOT_RUN. Also records the
reproducibility of the v5 re-runs of v4 configurations (identical code paths, deterministic Warp CPU).
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT/"artifacts"/"rl_ready"/"p2_2_v5"
A4 = ROOT/"artifacts"/"rl_ready"/"p2_2"
CONFIG = ROOT/"configs"/"rubber"/"rubber_phys_tests_v5.json"


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _meta(m, keys=("command", "commit", "src_or_configs_dirty", "tree_dirty_src_scripts_configs_tests", "config_sha256",
                   "python", "warp", "workers", "jobs", "cuda_visible_devices", "started", "finished", "wall_s",
                   "load_avg_start", "load_avg_end", "cpu_count")):
    return {k: m.get(k) for k in keys if k in m}


def phys(run="rubber_phys_v5_001", summary="summary_final.json"):
    d = A/run
    s = load(d/summary) or load(d/"summary.json")
    if s is None:
        return {"status": "NOT_RUN"}
    out = {"raw": f"artifacts/rl_ready/p2_2_v5/{run}", "summary_file": summary if (d/summary).exists() else
           "summary.json", "meta": _meta(load(d/"run_meta.json") or {}), "tests": {}}
    for tid, t in s.items():
        if t.get("verdict") == "NOT_RUN":
            continue
        out["tests"][tid] = {"verdict": t["verdict"], "criteria": t.get("criteria"), "cases": t.get("cases"),
                             "errors": t.get("errors"), "case_wall_s_total": t.get("case_wall_s_total"),
                             "measured": t.get("measured")}
    raw = {}
    for tid in ("04", "08"):
        rows = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(d/"raw"/tid/"*.json")))]
        keep = {"04": ("run", "n_t", "particles", "dt_s", "substeps_per_record", "cfl", "f1_hz", "f_EB_hz",
                       "energy_drift_rel_peak_psi", "invalid_flags", "cfl_eff_max", "stable_step_guard", "Jbar_range",
                       "job_wall_s"),
                "08": ("gamma", "substeps", "control_dt_s", "dt_s", "deformation_reserve", "cfl_actual",
                       "cfl_eff_max_end", "invalid_flags", "sigma_12_rel_error", "N1_rel_error", "job_wall_s")}[tid]
        raw[tid] = [{k: r.get(k) for k in keep} for r in rows]
    out["rows_04_08"] = raw
    rows09 = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(d/"raw"/"09"/"*.json")))]
    out["rows_09"] = [{"label": r["label"], "invalid_flags": r.get("invalid_flags"), "valid": r.get("valid"),
                       "first_flag_step": next((x["step"] for x in r.get("step_flags", []) if x["flags"]), None),
                       "nonfinite_steps": [x["step"] for x in r.get("step_flags", []) if x["nonfinite"]]}
                      for r in rows09 if r.get("status") == "OK"]
    return out


def batched_04(run="batched_04_v5_001"):
    b, m = load(A/run/"batched_04.json"), load(A/run/"meta.json")
    if b is None:
        return {"status": "NOT_RUN"}
    return {"raw": f"artifacts/rl_ready/p2_2_v5/{run}", "meta": m,
            **{k: b.get(k) for k in ("comparison", "N", "n_t", "steps", "substeps", "dt_s", "stable_step_guard",
                                     "particles", "array_mismatches", "first_mismatch", "bytewise_equal",
                                     "formal_dt_run_tip_equal", "formal_dt_run_dt_equal", "bitwise_equal",
                                     "invalid_reason_all_zero", "wall_s")},
            "per_env_f1_over_fEB": [x.get("f1_hz", float("nan"))/x["f_EB_hz"] for x in b["per_env_vibration_metrics"]]}


def refinement(run="refinement_v5_001"):
    d = A/run
    s = load(d/"summary.json")
    if s is None:
        return {"status": "NOT_RUN"}
    cfg = json.loads(CONFIG.read_text())
    st = cfg["tests"]["RUBBER-PHYS-11"]["setup"]
    table = {}
    for q in st["observables"]:
        m = s["measured"][q]
        f, c = m.get("triplet_finest") or {}, m.get("triplet_coarsest") or {}
        pick = ("status", "p_obs", "p_eff", "fs", "order_verified", "phi_ext", "phi_ext_p_obs", "gci_coarse",
                "gci_fine", "gci_coarse_p_obs", "gci_fine_p_obs", "range_rel", "r21", "r32")
        table[q] = {"values_cfl0.5": m.get("values_cfl0.5"), "triplet_4_6_8": {k: f.get(k) for k in pick},
                    "production_gci_nt4": f.get("gci_coarse"), "triplet_3_4_6": {k: c.get(k) for k in pick},
                    "dt_change": m.get("dt_change_cfl050_to_cfl025"),
                    "rubber_rl_v1_bias_reported_not_scored": m.get("rubber_rl_v1_bias_reported_not_scored"),
                    "invalid_or_unconverged": m.get("invalid_or_unconverged"), "missing": m.get("missing")}
    rows = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(d/"raw"/"*"/"*.json")))]
    runs = [{"label": r["label"], "status": r["status"], "Q": r.get("Q"), "particles": r.get("particles"),
             "substeps": r.get("substeps"), "dt_s": r.get("dt_s"), "invalid": r.get("invalid"),
             "guard_violations": r.get("guard_violations"), "converged": r.get("converged"),
             "init_time_s": r.get("init_time_s"), "job_wall_s": r.get("job_wall_s")} for r in rows]
    return {"raw": f"artifacts/rl_ready/p2_2_v5/{run}", "meta": _meta(load(d/"run_meta.json") or {}),
            "verdict": s["verdict"], "criteria": s["criteria"], "table": table, "runs": runs,
            "Q3": s["measured"].get("Q3")}


def reproducibility():
    """v5 re-runs of configurations identical to v4 ones (same code path): equal values expected (Warp CPU)."""
    out = {}
    r5, r4 = A/"refinement_v5_001"/"raw", A4/"refinement_v4_002"/"raw"
    for p in sorted(glob.glob(str(r5/"*"/"*.json"))):
        a = json.loads(Path(p).read_text())
        b = load(r4/a["qid"]/Path(p).name)
        if b is not None and a.get("status") == b.get("status") == "OK":
            out[f"refinement {a['label']}"] = {"v5": a["Q"], "v4": b["Q"], "equal": a["Q"] == b["Q"]}
    p5, p4 = A/"rubber_phys_v5_001"/"raw"/"02", A4/"rubber_phys_v4_001"/"raw"/"02"
    for p in sorted(glob.glob(str(p5/"*_nt4.json"))):
        a, b = json.loads(Path(p).read_text()), load(p4/Path(p).name)
        if b is not None and a.get("status") == b.get("status") == "OK":
            out[f"-02 {a['label']}"] = {"v5": a["delta_m"], "v4": b["delta_m"], "equal": a["delta_m"] == b["delta_m"]}
    return {"pairs": out, "all_equal": all(v["equal"] for v in out.values()) if out else None, "count": len(out)}


def gpu(run="workload_e_v2_gpu_001"):
    rows, m = load(A/run/"rows.json"), load(A/run/"meta.json")
    if rows is None:
        return {"status": "NOT_RUN"}
    keep = ("N", "particles_per_env", "control_hz", "material_dt_s", "substeps", "n_t", "beta", "resolution_profile",
            "stable_step_guard", "bc_launch", "cameras", "recording", "env_steps_per_s", "aggregate_sim_time_per_wall",
            "ms_per_substep", "particle_substeps_per_s", "vram_peak_mb_process_cumulative", "mempool_used_mb",
            "reset_p95_ms", "physics_verdict", "vram_criterion_4gb", "cold_wall_s", "warm_wall_s_median", "build_s",
            "gpu", "driver", "sharing", "co_tenants_seen", "cfl_eff_max_end", "Jbar_range", "invalid_reason",
            "guard_violations", "audits")
    return {"raw": f"artifacts/rl_ready/p2_2_v5/{run}", "verdict": m.get("verdict"),
            "meta": {k: m.get(k) for k in ("command", "commit", "dirty_tree", "gpu", "gpu_driver_version",
                                           "nvidia_smi_header", "cuda_visible_devices", "cuda_device_order",
                                           "protocol_sha256", "warp", "started", "finished", "gpu_sharing")},
            "co_tenant_samples": [{"time": s["time"], "util_pct": s["util_pct"], "mem_used_mb": s["mem_used_mb"],
                                   "others": [f"pid {o['pid']} {o['used_mb']} MB" for o in s["others"]]}
                                  for s in [m["co_tenant_before"]]+m.get("co_tenant_samples", [])
                                  +[m.get("co_tenant_after", {"time": None, "util_pct": None, "mem_used_mb": None,
                                                              "others": []})]],
            "rows": [{k: r.get(k) for k in keep} for r in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT/"docs"/"evidence"/"rl_ready"/"p2_2_v5_001.json"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    # "section 21" below is the number this protocol had on claude/rl-rubber-v5; it is ADR 0023 section 19A after
    # the K1 merge. The string is kept as written into p2_2_v5_001.json (which carries an additive k1_note).
    ev = {"schema": "rcs_rl_ready_evidence_v1",
          "work_package": "P2-2 protocol v5 (R2.5 review decisions (a), (b), findings F7, F9; ADR 0023 section 21)",
          "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
          "protocol": {"file": "configs/rubber/rubber_phys_tests_v5.json",
                       "sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(), "declaration_commit": "8881ff0"},
          "rubber_phys_v5": phys(), "batched_04_v5": batched_04(), "refinement_v5": refinement(),
          "workload_e_v2_gpu": gpu(), "reproducibility_v4_configurations": reproducibility()}
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
