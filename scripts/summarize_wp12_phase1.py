#!/usr/bin/env python3
"""Compact evidence and Markdown tables from a WP-12 Phase 1 run folder (read-only on the run).

  .venv/bin/python scripts/summarize_wp12_phase1.py --run artifacts/wp12/phase1_001 \
      --evidence docs/evidence/wp12/phase1_surface_diagnosis_001.json > artifacts/wp12/phase1_001/tables.md
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MM = 1e3


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def f(v, scale=MM, nd=3):
    return "—" if v is None else f"{v*scale:.{nd}f}"


def att_row(case, k, a):
    b = a.get("budget_quadratic_mean_patch_rms_m", {})
    inc = a.get("budget_increments_m", {})
    l6 = a.get("L6_reconstruction", {})
    cv, lg = l6.get("current_volume_v1", {}), l6.get("legacy", {})
    return {
        "case": case, "cut": k, "links": a.get("links"), "core_relaxed": a.get("core_relaxed"),
        "L1_path_p95": g(a, "L1_path_vs_declared_plane", "p95"), "L1_sweep_rough": g(a, "L1_path_sweep_roughness", "quadratic_mean"),
        "L2_assign_p95": g(a, "L2_assignment_vs_sweep", "all", "p95"), "L2_assign_max": g(a, "L2_assignment_vs_sweep", "all", "max"),
        "L3_pre_p95": g(a, "L3_pre_assignment_displacement", "all", "p95"), "L3_pre_max": g(a, "L3_pre_assignment_displacement", "all", "max"),
        "L3_pre_mean": g(a, "L3_pre_assignment_displacement", "all", "mean_signed_m"),
        "L4_stair_p95": g(a, "L4_lattice_staircase_reference", "all", "p95"), "L4_stair_max": g(a, "L4_lattice_staircase_reference", "all", "max"),
        "ref_rough_hermite": g(a, "reference_roughness_hermite", "quadratic_mean"),
        "ref_rough_midpoint": g(a, "reference_roughness_midpoint", "quadratic_mean"),
        "L5_particle_p95": g(a, "L5_post_cut_particle_residual", "normal_residual_core", "p95"),
        "L5_particle_max": g(a, "L5_post_cut_particle_residual", "normal_residual_core", "max"),
        "L5_F_p95": g(a, "L5_post_cut_deformation", "normal_residual_core", "p95"),
        "rough_face_particles": b.get("final_face_particles"), "rough_physics": b.get("final_physics"),
        "rough_lattice": b.get("final_lattice"), "rough_F_locus": b.get("final_hermite_locus"),
        "rough_mesh_cv1": b.get("mesh_current_volume_v1"), "rough_mesh_legacy": b.get("mesh_legacy"),
        "inc": inc,
        "cv1_d_p95": g(cv, "distance_to_side_locus_plane", "p95"), "cv1_d_max": g(cv, "distance_to_side_locus_plane", "max"),
        "cv1_rev_p95": g(cv, "locus_to_mesh_distance", "p95"), "cv1_cov": cv.get("coverage"),
        "cv1_bias": g(cv, "signed_bias", "bias_toward_side2_m"), "cv1_thick": g(cv, "signed_bias", "thickness_bias_m"),
        "cv1_nrm_p50": g(cv, "normal_error_deg_core", "p50"), "cv1_nrm_p95": g(cv, "normal_error_deg_core", "p95"),
        "lg_d_p95": g(lg, "distance_to_side_locus_plane", "p95"), "lg_rev_p95": g(lg, "locus_to_mesh_distance", "p95"),
        "lg_cov": lg.get("coverage"), "lg_thick": g(lg, "signed_bias", "thickness_bias_m"),
        "lg_nrm_p50": g(lg, "normal_error_deg_core", "p50"), "lg_nrm_p95": g(lg, "normal_error_deg_core", "p95"),
    }


def surface_row(case, rep):
    return {"case": case, "volume_err_current": g(rep, "volume", "total_relative_error_current"),
            "max_component_volume_err": g(rep, "volume", "max_abs_component_relative_error_current"),
            "watertight": g(rep, "mesh_validity", "watertight"), "shells": g(rep, "mesh_validity", "shells"),
            "degenerate": g(rep, "mesh_validity", "degenerate_triangles"),
            "non_manifold": g(rep, "mesh_validity", "non_manifold_edges"),
            "self_intersections_sampled": g(rep, "mesh_validity", "self_intersection", "intersecting_pairs_sampled"),
            "shells_equal_components": g(rep, "topology", "shells_equal_components"),
            "rim_p95": g(rep, "rim", "corner_to_mesh_distance", "p95"), "rim_max": g(rep, "rim", "corner_to_mesh_distance", "max"),
            "open_edges_near_junction": g(rep, "rim", "open_edges_near_junction"),
            "region": rep.get("region"), "extract_wall_s": rep.get("extract_wall_s"),
            "skipped": rep.get("unmeshable_skipped")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    a = p.parse_args()
    run = a.run if a.run.is_absolute() else ROOT/a.run
    if a.evidence.exists():
        raise SystemExit(f"Refusing to overwrite {a.evidence}")
    protocol = json.loads((run/"protocol.json").read_text())
    summary = json.loads((run/"summary.json").read_text())
    cases, attribution, surfaces = {}, [], []
    for path in sorted(run.glob("G*.json")):
        r = json.loads(path.read_text())
        cid = r.get("case", path.stem)
        row = {"verdict": r.get("verdict"), "checks": r.get("checks"), "mechanical_components": r.get("mechanical_components"),
               "lattice_components": r.get("lattice_components"), "mass_fractions": (r.get("mass_fractions") or [])[:6],
               "labelled": r.get("labelled"), "physics_wall_s": r.get("physics_wall_s"), "analysis_wall_s": r.get("analysis_wall_s"),
               "recorder_audit": r.get("recorder_audit"), "phases": r.get("phases"), "temporal_fixed_state": None}
        for key in ("partial", "thin_chip", "multi_cut", "static", "rest", "controls", "planes"):
            if key in r:
                row[key] = r[key]
        if "recontact" in r:
            row["recontact_frames"] = [{k: v for k, v in fr.items() if k != "temporal_vs_previous"} |
                                       {"temporal_residual_p95_m": g(fr, "temporal_vs_previous", "residual_after_transport", "p95"),
                                        "temporal_residual_max_m": g(fr, "temporal_vs_previous", "residual_after_transport", "max")}
                                       for fr in r["recontact"]["frames"]]
        if "temporal_fixed_state" in r:
            row["temporal_fixed_state"] = {k: {"same_topology": v["same_topology"],
                                               "residual_max_m": g(v, "residual_after_transport", "max")}
                                           for k, v in r["temporal_fixed_state"].items()}
        cases[cid] = row
        for k, att in (r.get("attribution") or {}).items():
            if att.get("status") == "MEASURED":
                attribution.append(att_row(cid, k, att))
        for profile, rep in (r.get("surface_reports") or {}).items():
            surfaces.append({**surface_row(cid, rep), "profile": profile})
    baselines = {}
    if (run/"baselines.json").exists():
        for sid, b in json.loads((run/"baselines.json").read_text()).items():
            baselines[sid] = {"components": b["components"], "recorded_components": b["recorded_components"],
                              "mass_fractions": b["mass_fractions"][:5],
                              "attribution": [att_row(sid, k, x) | {"total_deformation_p95": g(x, "total_deformation_normal_residual_core", "p95")}
                                              for k, x in b["attribution"].items() if x.get("status") == "MEASURED"],
                              "surfaces": {p: surface_row(sid, rep) for p, rep in b["surface_reports"].items()
                                           if rep.get("status") in ("MEASURED", "NO_CUT_LINKS")},
                              "topology_layer": b["topology_layer"]}
    evidence = {"schema": "wp12_phase1_surface_diagnosis_evidence_v1", "run": str(run.relative_to(ROOT)),
                "protocol": {k: protocol.get(k) for k in ("command", "git", "device", "device_name", "warp", "newton", "python",
                                                          "written_before_run_utc", "criteria", "seeds", "surface_profiles")},
                "summary": summary, "cases": cases, "attribution": attribution, "surfaces": surfaces, "baselines": baselines,
                "units": "metres unless noted; angles in degrees"}
    a.evidence.parent.mkdir(parents=True, exist_ok=True)
    a.evidence.write_text(json.dumps(evidence, indent=1, default=str))
    # ------------------------------------------------------------- Markdown tables (mm)
    print("| Case | cut | links | L1 path P95 | L2 assign P95/max | L3 pre-assign P95/max (mean) | L4 staircase P95/max | "
          "L5 post-cut particle P95/max | rough: face particles / Hermite-tr / midpoint-tr / mesh cv1 / mesh legacy |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in attribution:
        print(f"| {r['case']} | {r['cut']} | {r['links']} | {f(r['L1_path_p95'])} | {f(r['L2_assign_p95'])}/{f(r['L2_assign_max'])} | "
              f"{f(r['L3_pre_p95'])}/{f(r['L3_pre_max'])} ({f(r['L3_pre_mean'])}) | {f(r['L4_stair_p95'])}/{f(r['L4_stair_max'])} | "
              f"{f(r['L5_particle_p95'])}/{f(r['L5_particle_max'])} | {f(r['rough_face_particles'])} / {f(r['rough_physics'])} / "
              f"{f(r['rough_lattice'])} / {f(r['rough_mesh_cv1'])} / {f(r['rough_mesh_legacy'])} |")
    print()
    print("| Case | cut | cv1: mesh→locus P95/max | locus→mesh P95 | coverage | thickness bias | shift to side 2 | normal P50/P95 (deg) | "
          "legacy: mesh→locus P95 | locus→mesh P95 | coverage | thickness bias | normal P50/P95 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in attribution:
        print(f"| {r['case']} | {r['cut']} | {f(r['cv1_d_p95'])}/{f(r['cv1_d_max'])} | {f(r['cv1_rev_p95'])} | "
              f"{f(r['cv1_cov'], 1, 2)} | {f(r['cv1_thick'])} | {f(r['cv1_bias'])} | {f(r['cv1_nrm_p50'], 1, 1)}/{f(r['cv1_nrm_p95'], 1, 1)} | "
              f"{f(r['lg_d_p95'])} | {f(r['lg_rev_p95'])} | {f(r['lg_cov'], 1, 2)} | {f(r['lg_thick'])} | "
              f"{f(r['lg_nrm_p50'], 1, 1)}/{f(r['lg_nrm_p95'], 1, 1)} |")
    print()
    print("| Case | profile | volume err (current) | worst component | closed | shells = components | self-int. (sampled) | rim P95/max | extract (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in surfaces:
        print(f"| {s['case']} | {s['profile']} | {f(s['volume_err_current'], 100, 2)} % | {f(s['max_component_volume_err'], 100, 2)} % | "
              f"{s['watertight']} | {s['shells_equal_components']} | {s['self_intersections_sampled']} | {f(s['rim_p95'])}/{f(s['rim_max'])} | "
              f"{f(s['extract_wall_s'], 1, 2)} |")
    if baselines:
        print()
        print("| Baseline | cut | links | L4 staircase P95 | total deformation P95 | rough: face particles / midpoint-tr / mesh cv1 / mesh legacy | "
              "cv1 mesh→locus P95 | cv1 coverage | cv1 thickness bias | legacy thickness bias | cv1 normal P50/P95 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for sid, b in baselines.items():
            for r in b["attribution"]:
                print(f"| {sid} | {r['cut']} | {r['links']} | {f(r['L4_stair_p95'])} | {f(r['total_deformation_p95'])} | "
                      f"{f(r['rough_face_particles'])} / {f(r['rough_lattice'])} / {f(r['rough_mesh_cv1'])} / {f(r['rough_mesh_legacy'])} | "
                      f"{f(r['cv1_d_p95'])} | {f(r['cv1_cov'], 1, 2)} | {f(r['cv1_thick'])} | {f(r['lg_thick'])} | "
                      f"{f(r['cv1_nrm_p50'], 1, 1)}/{f(r['cv1_nrm_p95'], 1, 1)} |")


if __name__ == "__main__":
    main()
