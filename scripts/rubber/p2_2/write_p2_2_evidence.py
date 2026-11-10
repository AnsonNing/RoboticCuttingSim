"""Compact P2-2 evidence (docs/evidence/rl_ready/p2_2_rubber_001.json) from the raw outputs in artifacts/rl_ready/p2_2.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_2/write_p2_2_evidence.py [--out docs/evidence/rl_ready/p2_2_rubber_001.json]

Reads only; never overwrites an existing evidence file. Missing raw outputs are recorded as NOT_RUN.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT/"artifacts"/"rl_ready"/"p2_2"


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def f2():
    r, m = load(A/"f2_gpu_smoke_guard_001"/"results.json"), load(A/"f2_gpu_smoke_guard_001"/"meta.json")
    if r is None:
        return {"status": "NOT_RUN"}
    rows = [{k: x.get(k) for k in ("N", "stable_step_guard", "particles_per_env", "material_dt_s", "substeps",
                                    "ms_per_substep_warm_median", "env_steps_per_s_warm", "reset_p95_ms",
                                    "vram_peak_mb_process", "finite", "mass_constant", "invalid_reason", "audits",
                                    "guard_violations_after_run_per_env", "cfl_eff_max_end_per_env", "Jbar_range",
                                    "physics_verdict", "pass")} for x in r]
    return {"raw": "artifacts/rl_ready/p2_2/f2_gpu_smoke_guard_001", "commit": m["commit"], "gpu": m["device_name"],
            "cuda_visible_devices": m["cuda_visible_devices"], "sharing": m["gpu_sharing"],
            "co_tenant_before": m["co_tenant_before"]["apps"], "co_tenant_after": m["co_tenant_after"]["apps"],
            "window": [m["started"], m["finished"]], "rows": rows,
            "verdict": "PASS" if all(x["pass"] for x in rows) else "FAIL"}


def f8():
    rows = []
    for f in sorted(glob.glob(str(A/"f8_clamp_dissipation_001"/"*"/"diag.json"))):
        r = load(f)
        rows.append({k: r[k] for k in ("dirichlet", "cfl", "n_t", "mask", "free_free", "max_abs_dE_rel_psi",
                                       "mean_dE_rel_psi_last_quarter", "slope_dE_rel_psi_per_T_ax",
                                       "mean_length_rel_error", "max_lateral_m", "invalid", "dt_s", "particles")})
    return {"raw": "artifacts/rl_ready/p2_2/f8_clamp_dissipation_001", "label": "DIAGNOSTIC", "rows": rows}


def f10():
    rows = []
    for f in sorted(glob.glob(str(A/"f10_static_precision_001"/"*"/"diag.json"))):
        r = load(f)
        rows.append({"run": Path(f).parent.name, **{k: r[k] for k in (
            "precision", "cfl", "ratio_EB_at_3T1", "ratio_EB_end", "drift_ratio_per_T1_after_3T1", "band_after_3T1",
            "psi_rel_change_3T1_to_end", "ratio_EB_per_T1", "wall_s")}})
    return {"raw": "artifacts/rl_ready/p2_2/f10_static_precision_001", "label": "DIAGNOSTIC", "rows": rows}


def phys_v4():
    s, m = load(A/"rubber_phys_v4_001"/"summary.json"), load(A/"rubber_phys_v4_001"/"run_meta.json")
    if s is None:
        return {"status": "NOT_RUN"}
    return {"raw": "artifacts/rl_ready/p2_2/rubber_phys_v4_001", "meta": {k: m.get(k) for k in (
        "command", "commit", "src_or_configs_dirty", "config", "config_sha256", "workers", "jobs", "warp", "wall_s",
        "started", "finished")}, "verdicts": {k: v["verdict"] for k, v in s.items()},
        "criteria": {k: v.get("criteria") for k, v in s.items()},
        "measured": {k: v.get("measured") for k, v in s.items() if k != "RUBBER-PHYS-02"},
        "RUBBER-PHYS-02_rows": (s.get("RUBBER-PHYS-02", {}).get("measured") or {}).get("rows")}


def l3():
    r, m = load(A/"l3_v4_001"/"l3.json"), load(A/"l3_v4_001"/"meta.json")
    if r is None:
        return {"status": "NOT_RUN"}
    ev = r["evaluation"]
    return {"raw": "artifacts/rl_ready/p2_2/l3_v4_001", "verdict": r["verdict"], "rule": r["rule"],
            "commit": m["commit"], "config_sha256": m["config_sha256"], "gpu": m["device_name"],
            "sharing": m["gpu_sharing"], "co_tenant": [m["co_tenant_before"]["apps"], m["co_tenant_after"]["apps"]],
            "window": [m["started"], m["finished"]], "runs": ev["runs"], "alpha_stat": ev["alpha_stat"],
            "bounds": {k: {x: b[x] for x in ("upper", "method", "shapiro_p", "n")} for k, b in ev["bounds"].items()},
            "batched_rows": [x for x in ev["rows"] if x["run"].startswith("batched")],
            "heldout_outside_runs": r["heldout_outside_runs"], "heldout_runs": r["heldout_runs"],
            "receipts": r["batched_receipts"], "wall_s": r["wall_s"], "vram_peak_mb_process": r["vram_peak_mb_process"]}


def throughput():
    rows, prof, m = load(A/"throughput_v4_001"/"rows.json"), load(A/"throughput_v4_001"/"profile.json"), \
        load(A/"throughput_v4_001"/"meta.json")
    if rows is None:
        return {"status": "NOT_RUN"}
    keep = ("N", "particles_per_env", "control_hz", "material_dt_s", "substeps", "n_t", "beta", "stable_step_guard",
            "cameras", "recording", "build_s", "cold_wall_s", "warm_wall_s_median", "env_steps_per_s",
            "aggregate_sim_time_per_wall", "ms_per_substep", "particle_substeps_per_s", "reset_p95_ms",
            "vram_peak_mb_process_cumulative", "mempool_used_mb", "physics_verdict", "cfl_eff_max_end")
    return {"raw": "artifacts/rl_ready/p2_2/throughput_v4_001", "commit": m["commit"], "gpu": m["device_name"],
            "cuda_visible_devices": m["cuda_visible_devices"], "sharing": m.get("gpu_sharing"),
            "co_tenant": [m["co_tenant_before"]["apps"]]+[c["apps"] for c in m.get("co_tenant_samples", [])],
            "window": [m["started"], m.get("finished")], "rows": [{k: r.get(k) for k in keep} for r in rows],
            "profile": None if prof is None else {
                "guard_share_of_gpu_time": prof.get("guard_share_of_gpu_time"),
                **{lab: {k: prof[lab][k] for k in ("N", "substeps", "gpu_ms_per_substep")} | {
                    "top_kernels_ms": dict(list(prof[lab]["kernels_ms"].items())[:12])}
                   for lab in ("guard_on", "guard_off") if lab in prof}}}


def batched_gpu():
    r, m = load(A/"batched_gpu_v4_001"/"results.json"), load(A/"batched_gpu_v4_001"/"meta.json")
    if r is None:
        return {"status": "NOT_RUN"}
    return {"raw": "artifacts/rl_ready/p2_2/batched_gpu_v4_001", "commit": m and m["commit"],
            "sharing": m and m.get("gpu_sharing"), "gpu": m and m["device_name"],
            "cases": [{k: x.get(k) for k in ("case", "N", "verdict", "wall_s", "receipt", "error")} |
                      {"per_env": [{"verdict": p["verdict"], "measured": p["measured"]} for p in x.get("per_env", [])]}
                      for x in r]}


def batched_l0():
    out = {}
    for case, run in (("07_ramped", "001"), ("07_instant", "001"), ("12", "003")):
        r = load(A/f"f4_batched_l0_{case}_{run}"/"l0.json")
        out[case] = {"status": "NOT_RUN"} if r is None else dict(r, raw=f"artifacts/rl_ready/p2_2/f4_batched_l0_{case}_{run}")
    out["earlier_12_runs"] = {}
    for run in ("001", "002"):
        d = A/f"f4_batched_l0_12_{run}"
        r, note = load(d/"l0.json"), d/"NOTE.txt"
        out["earlier_12_runs"][run] = {"first_mismatch": r and r["first_mismatch"],
                                       "note": note.read_text().strip() if note.exists() else None}
    out["note_07"] = ("the 07 runs compared with np.array_equal (signed zero equal, NaN unequal; both states stayed "
                      "finite); their 'commit' field was read at the end of the run, the code imported at the start "
                      "was 44e5b59")
    return out


def refinement():
    aborted = (A/"refinement_v4_001"/"ABORTED.txt")
    s, m = load(A/"refinement_v4_002"/"summary.json"), load(A/"refinement_v4_002"/"run_meta.json")
    first = {"run": "artifacts/rl_ready/p2_2/refinement_v4_001",
             "status": aborted.read_text().strip() if aborted.exists() else None}
    if s is None:
        return {"status": "NOT_RUN", "aborted_first_run": first}
    return {"raw": "artifacts/rl_ready/p2_2/refinement_v4_002", "meta": m, "verdict": s["verdict"],
            "criteria": s["criteria"], "measured": s["measured"], "errors": s["errors"], "aborted_first_run": first}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT/"docs"/"evidence"/"rl_ready"/"p2_2_rubber_001.json"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    ev = {"schema": "rcs_rl_ready_evidence_v1", "work_package": "P2-2 rubber hyperelastic (ADR 0023 §15 R2.1-R2.4, "
                                                                "§18.8 F1-F5, F8-F10, §19)",
          "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED", "protocol": "configs/rubber/rubber_phys_tests_v4.json",
          "F2_gpu_smoke_guard": f2(), "F8_clamp_dissipation": f8(), "F10_static_precision": f10(),
          "rubber_phys_v4": phys_v4(), "L3_v4": l3(), "refinement_v4": refinement(), "throughput_v4": throughput(),
          "batched_gpu_v4": batched_gpu(), "batched_l0_v4": batched_l0()}
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
