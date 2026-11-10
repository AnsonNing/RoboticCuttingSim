"""Compact evidence of the P2-1 follow-up (ADR 0023 §18) from the raw artifacts (no simulation).

    python3 scripts/rubber/p2_1_followup/write_followup_evidence.py --out docs/evidence/rl_ready/p2_1_followup_001.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT/"artifacts"/"rl_ready"/"p2_1"


def load(rel):
    return json.loads((A/rel).read_text())


def spectrum():
    d = load("followup_stability_spectrum_n4_001/stability_spectrum.json")
    keep = ("name", "J", "cfl_crit_musl", "cfl_crit_lumped", "r_exact", "negative_modes", "K_asymmetry_rel",
            "K_min_nontranslational_rel")
    return {"label": d["label"], "n_cells": d["n_cells"], "beta": d["beta"], "boundary": d["boundary"],
            "rows": [{k: r[k] for k in keep} for r in d["rows"]],
            "negative_modes_total": sum(r["negative_modes"] for r in d["rows"]),
            "K_asymmetry_max": max(r["K_asymmetry_rel"] for r in d["rows"])}


def release_runs():
    out = {}
    for name in ("cfl050_001", "cfl025_001", "cfl0125_001", "cfl025_freefree_001", "cfl025_002"):
        d = load(f"followup_07_release_{name}/diag.json")
        rows = [r for r in d["rows"] if r["finite"]]
        pre = [r for r in rows if r["t_ms"] <= 10.0]
        post = [r for r in rows if r["t_ms"] > 10.0]
        out[name] = {
            "label": d["label"], "cfl_nominal": d["cfl_nominal"], "free_free": d["free_free"],
            "first_nonfinite_ms": d["first_nonfinite"] and d["first_nonfinite"]["t_ms"],
            "first_j_domain_violation_ms": d["first_j_domain_violation"] and d["first_j_domain_violation"]["t_ms"],
            "pre_buckling_0_10ms": {"max_abs_energy_change_rel_psi": max(abs(r["mechanical_rel_change"]) for r in pre),
                                    "min_principal_stretch": min(r["stretch_min"] for r in pre),
                                    "max_guard_cfl_bound": max(r["cfl_bound_max"] for r in pre),
                                    "J_range": [min(r["J_min"] for r in pre), max(r["J_max"] for r in pre)],
                                    "Jbar_min": min(r["Jbar_min"] for r in pre),
                                    "min_length_rel": min(r["length_rel"] for r in pre),
                                    "max_speed_m_s": max(r["max_speed"] for r in pre),
                                    "max_KEp_minus_KEg_j": max(r["kinetic_particle"]-r["kinetic_grid"] for r in pre)},
            "lateral_over_1mm_at_ms": next((r["t_ms"] for r in rows if r["lateral_max_m"] > 1e-3), None),
            "post_10ms": {"min_principal_stretch": min(r["stretch_min"] for r in post),
                          "max_guard_cfl_bound": max(r["cfl_bound_max"] for r in post),
                          "J_min": min(r["J_min"] for r in post), "Jbar_min": min(r["Jbar_min"] for r in post),
                          "energy_change_rel_psi_at_end": rows[-1]["mechanical_rel_change"], "end_ms": rows[-1]["t_ms"],
                          "KEp_minus_KEg_end_j": rows[-1]["kinetic_particle"]-rows[-1]["kinetic_grid"],
                          "max_lateral_m": max(r["lateral_max_m"] for r in post)},
            "note": ("bc_ke_removed_lumped of the *_001 runs is invalid (zero-copy view) and is omitted; the *_002 "
                     "value counts re-supplied lumped energy and is not an energy measure")}
    return out


def l103_runs():
    import numpy as np
    out = {}
    for name in ("l103_cfl050_002", "l103_cfl025_002", "l103_cfl050_freefree_002"):
        d = load(f"followup_07_release_{name}/diag.json")
        rows = [r for r in d["rows"] if r["finite"]]
        E = np.array([r["mechanical_rel_change"] for r in rows])
        t = np.array([r["t_ms"] for r in rows])
        out[name] = {"label": d["label"], "cfl_nominal": d["cfl_nominal"], "free_free": d["free_free"],
                     "end_ms": float(t[-1]), "max_abs_energy_change_rel": float(np.abs(E).max()),
                     "mean_energy_change_by_15ms_window": [float(E[(t >= a) & (t < a+15)].mean())
                                                           for a in range(0, 75, 15)],
                     "max_lateral_m": max(r["lateral_max_m"] for r in rows),
                     "mean_length_rel": float(np.mean([r["length_rel"] for r in rows[1:]]))}
    return out


def stop_rule():
    d = load("followup_stop_rule_001/stop_rule.json")
    return {k: d[k] for k in ("config", "ratio_EB_at_stop_x_f32", "ratio_EB_at_stop_u_f64", "ratio_EB_final_x_f32",
                              "ratio_EB_final_u_f64", "elastic_rel_change_after_stop")} | {
        "v2_stop_t_s": d["initializer"]["time_s"], "f1_hz": d["f1_hz"],
        "ratio_EB_u_f64_series": [[r["t_s"], r["delta_u_f64"]/d["delta_EB_m"]] for r in d["after_extra"]]}


def v2_marker_changes():
    import glob
    out = []
    for tid in ("02", "03", "05"):
        for f in sorted(glob.glob(str(A/"rubber_phys_001"/"raw"/tid/"*.json"))):
            d = json.loads(Path(f).read_text())
            i = d["initializer"]
            out.append({"test": tid, "label": d["label"], "t_over_T1": i["time_s"]*d["f1_EB_hz"],
                        "last5_marker_changes_m": [h["marker_change_m"] for h in i["history_tail"]]})
    return {"configurations": len(out), "all_last5_exactly_zero": all(all(c == 0.0 for c in r["last5_marker_changes_m"])
                                                                      for r in out),
            "t_over_T1_range": [min(r["t_over_T1"] for r in out), max(r["t_over_T1"] for r in out)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    ev = {"schema": "rcs_rubber_p2_1_followup_evidence_v1", "adr": "docs/adr/0023-rubber-material-v1.md §18",
          "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
          "stability_spectrum": spectrum(),
          "guard_bound_check": {k: v for k, v in load("followup_bound_check_001/bound_check.json").items()
                                if k != "discrete"} | {"discrete_min_speed_ratio": load(
                                    "followup_bound_check_001/bound_check.json")["discrete"]["min_bound_over_exact"]},
          "release_diagnostics": release_runs(), "release_l103_bc_diagnostics": l103_runs(),
          "release_physics": load("followup_release_physics_001/release_physics.json"),
          "stop_rule_rerun": stop_rule(), "v2_initializer_marker_changes": v2_marker_changes(),
          "pollution_log": (A/"pollution_diag_001"/"pollution_diag.log").read_text().splitlines()}
    v3 = A/"rubber_phys_v3_001"
    if (v3/"summary.json").exists():
        ev["rubber_phys_v3_001"] = {"run_meta": json.loads((v3/"run_meta.json").read_text()),
                                    "summary": json.loads((v3/"summary.json").read_text())}
    smoke = A/"gpu_smoke_001"
    if (smoke/"results.json").exists():
        ev["gpu_smoke_001"] = {"meta": json.loads((smoke/"meta.json").read_text()),
                               "results": json.loads((smoke/"results.json").read_text()),
                               "commit_note": "run from a git-archive snapshot of 6c9477f (meta.commit is empty)"}
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
