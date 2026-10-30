#!/usr/bin/env python3
"""WP-12 Phase 3: compact evidence JSON and markdown tables from a ``run_wp12_phase3_surface.py`` run (read-only).

  .venv/bin/python scripts/summarize_wp12_phase3.py --run artifacts/wp12/phase3_surface_001 \\
      --perf perf_001 --diagnostics diagnostics_001.json --evidence docs/evidence/wp12/phase3_surface_001.json

Writes the evidence file (refuses to overwrite) and ``<run>/tables_NNN.md`` (new file).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A, B, B_MID = "current_volume_v1", "cut_aware_surface_v1", "cut_aware_surface_v1_midpoint_fallback"
CASES = ("S1_G3a", "S1_G3b", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
GEOM = (("current_p50_s", "cur P50"), ("current_p95_s", "cur P95"), ("current_max_s", "cur max"),
        ("reference_p95_s", "ref P95"), ("reference_max_s", "ref max"), ("coverage", "coverage"),
        ("abs_bias_toward_side2_s", "abs bias"), ("abs_thickness_bias_s", "abs thick"),
        ("normal_p50_deg", "nrm P50"), ("normal_p95_deg", "nrm P95"), ("rim_corner_p95_s", "rim P95"))
INTEG = (("closed", "closed"), ("shells_equal_components", "shells=comp"), ("self_intersections_sampled", "self-int"),
         ("volume_current_abs_rel", "abs dV cur"), ("volume_reference_abs_rel", "abs dV ref"),
         ("fixed_state_topology_stable", "fixed hash"), ("roughness_mesh_core_qm_s", "rough mesh"),
         ("roughness_locus_core_qm_s", "rough locus"))


def load(path):
    return json.loads(Path(path).read_text())


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        if abs(x) >= 100:
            return f"{x:.0f}"
        return f"{x:.{nd}f}"
    return str(x)


def joined(values, nd=3):
    vals = [fmt(v, nd) for v in values]
    return vals[0] if len(set(vals)) == 1 else " / ".join(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--perf", default=None)
    p.add_argument("--diagnostics", default=None)
    p.add_argument("--evidence", type=Path, required=True)
    a = p.parse_args()
    run = a.run if a.run.is_absolute() else ROOT/a.run
    evidence = a.evidence if a.evidence.is_absolute() else ROOT/a.evidence
    if evidence.exists():
        raise SystemExit(f"Refusing to overwrite {evidence}")
    record = load(run/"protocol_record.json")
    summary = load(run/"summary.json")
    reps = {c: sorted(run.glob(f"{c}_r[0-9].json")) for c in (*CASES, "S9")}
    res = {c: [load(x) for x in paths] for c, paths in reps.items() if paths}
    s0 = load(run/"S0_r0.json") if (run/"S0_r0.json").exists() else None
    g2 = load(run/"G2_r0.json") if (run/"G2_r0.json").exists() else None
    perf = load(run/a.perf/"perf.json") if a.perf else None
    diag = load(run/a.diagnostics) if a.diagnostics else None
    md = []
    ev = {"schema": "wp12_phase3_surface_evidence_v1", "run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
          "protocol": {k: record[k] for k in ("protocol_file", "protocol_sha256", "protocol_last_commit",
                                              "protocol_unchanged_since_freeze", "written_before_run_utc")},
          "command": record["command"], "git_head": record["git_head"], "git_dirty_paths": record["git_dirty_paths"],
          "device": record["device"], "device_name": record["device_name"], "warp": record["warp"],
          "newton": record["newton"], "repeats": record["repeats"], "cases": {}, "wall_s": summary.get("wall_s"),
          "total_wall_s": summary.get("total_wall_s")}
    # ---- verdicts
    md.append("### Verdicts (thresholds_for_candidate_B applied to every method; r0 / r1)\n")
    md.append("| Case | A current_volume_v1 | B cut_aware_surface_v1 | B midpoint fallback | B failing rows (value r0 / r1, threshold) |")
    md.append("|---|---|---|---|---|")
    if s0:
        tc = s0["threshold_checks"]
        md.append(f"| S0 uncut box | {tc[A]['verdict']} | {tc[B]['verdict']} | — | "
                  + "; ".join(f"{k} {fmt(c['value'])}" for k, c in tc[B]["checks"].items() if c["verdict"] != "PASS") + " |")
        ev["cases"]["S0"] = {"threshold_checks": tc, "methods": {m: {"box": r["box"], "validity": {
            k: r["validity"].get(k) for k in ("boundary_edges", "non_manifold_edges", "shells", "watertight", "triangles")}}
            for m, r in s0["methods"].items()}}
    if g2:
        tc = g2["threshold_checks"]
        fails = [f"{pl} {k} {fmt(c['value'])}" for pl, row in tc[B].items() if isinstance(row, dict)
                 for k, c in row["checks"].items() if c["verdict"] != "PASS"]
        md.append(f"| G2 static planes | {tc[A]['verdict']} | {tc[B]['verdict']} | {tc[B_MID]['verdict']} | {'; '.join(fails)} |")
        keep = ("cut_face_vs_analytic", "core_vs_analytic", "non_crease_normal_deg", "non_manifold_edges", "boundary_edges",
                "shells", "clamped_cut_face_vertices", "cut_face_max_excluding_clamped", "clamped_vs_analytic")
        ev["cases"]["G2"] = {"threshold_checks": tc, "planes": {pl: {m: {k: r[k] for k in keep if k in r} for m, r in rows.items()}
                                                                for pl, rows in g2["planes"].items()}}
    for c in CASES:
        if c not in res:
            continue
        rows = res[c]
        tcs = [r["threshold_checks"] for r in rows]
        fails = {}
        for r in tcs:
            for k, chk in r[B]["checks"].items():
                if chk["verdict"] not in ("PASS", "REPORT_ONLY", "NOT_APPLICABLE"):
                    fails.setdefault(k, {"values": [], "threshold": chk.get("threshold"), "op": chk.get("op")})
        for k in fails:
            fails[k]["values"] = [r[B]["checks"].get(k, {}).get("value") for r in tcs]
        ftxt = "; ".join(f"{k} {joined(v['values'])} ({v['op']} {fmt(v['threshold'])})" for k, v in fails.items())
        strict = ""
        if "strict_all_S1_rows" in tcs[0][B]:
            strict = " (strict all-S1: " + joined([r[B]["strict_all_S1_rows"]["verdict"] for r in tcs]) + ")"
        md.append(f"| {c} | {joined([r[A]['verdict'] for r in tcs])} | {joined([r[B]['verdict'] for r in tcs])}{strict} | "
                  f"{joined([r[B_MID]['verdict'] for r in tcs])} | {ftxt} |")
        ev["cases"][c] = {"repeats": len(rows), "physics_wall_s": [r["physics_wall_s"] for r in rows],
                          "analysis_wall_s": [r["analysis_wall_s"] for r in rows], "setup": rows[0]["setup"],
                          "recorder_audit": [r["recorder_audit"] for r in rows],
                          "mechanical_components": [r["frames"][-1]["mechanical_components"] for r in rows],
                          "threshold_checks": tcs,
                          "scalars": [{m: rr["frames"][-1]["reports"][m]["scalars"] for m in (A, B, B_MID)} for rr in rows],
                          "legacy_scalars_r0": rows[0]["frames"][-1]["reports"].get("legacy", {}).get("scalars"),
                          "cut_aware_info_r0": {k: v for k, v in rows[0]["frames"][-1]["reports"][B]["cut_aware_info"].items()
                                                if k != "mesh_validity_reference"}}
        if c == "S4":
            ev["cases"][c]["s4_provenance"] = [r["s4_provenance"] for r in rows]
            ev["cases"][c]["g4_hold_phase"] = [r["g4_hold_phase"] for r in rows]
            md.append(f"| S9 G4 hold | — | {joined([r['g4_hold_phase']['verdict'] for r in rows])} | — | |")
        if c == "S8":
            ev["cases"][c]["recontact_frames"] = [[{k: fr[k] for k in ("frame", "substep", "components")}
                                                   | {"min_gap_m": fr.get("min_particle_distance_between_pieces_m"),
                                                      "meshes": {m: {kk: x.get(kk) for kk in ("mesh_shells", "shell_components",
                                                                                              "inter_shell_intersections_sampled",
                                                                                              "self_intersections_sampled",
                                                                                              "mixed_side_regions", "topology_hash")}
                                                                 for m, x in fr["meshes"].items()},
                                                      "temporal_p95_s": {m: t["residual_after_transport"]["p95_spacings"]
                                                                         for m, t in (fr.get("temporal") or {}).items()}}
                                                   for fr in r["recontact_frames"]] for r in rows[:1]]
        if c == "S6":
            ev["cases"][c]["slice_frames"] = [[{"frame": fr["frame"], "components": fr["mechanical_components"],
                                                "shells": {m: x["mesh_shells"] for m, x in fr["reports"].items()}}
                                               for fr in r["slice_frames"]] for r in rows]
    if "S9" in res:
        rows = res["S9"]
        for name in rows[0]["threshold_checks"]:
            md.append(f"| S9 {name} | {joined([r['threshold_checks'][name][A]['verdict'] for r in rows])} | "
                      f"{joined([r['threshold_checks'][name][B]['verdict'] for r in rows])} | — | |")
        ev["cases"]["S9"] = {"controls": [r["controls"] for r in rows], "threshold_checks": [r["threshold_checks"] for r in rows]}
    # ---- metric matrix
    for title, cols in (("Cut-face geometry (spacings s = 1 mm; degrees)", GEOM), ("Mesh integrity, volume, temporal", INTEG)):
        md.append(f"\n### {title}; A = current_volume_v1, B = cut_aware_surface_v1 (recorder, F); r0 / r1\n")
        md.append("| Case | Method | " + " | ".join(h for _, h in cols) + " |")
        md.append("|---|---|" + "---|"*len(cols))
        for c in CASES:
            if c not in res:
                continue
            for m, label in ((A, "A"), (B, "B"), (B_MID, "B mid")):
                vals = [[r["frames"][-1]["reports"][m]["scalars"].get(k) for r in res[c]] for k, _ in cols]
                md.append(f"| {c} | {label} | " + " | ".join(joined(v) for v in vals) + " |")
    # ---- gate
    gate = summary.get("comparison_gate", {})
    ev["comparison_gate"] = gate
    md.append("\n### Comparison gate (B vs A beyond the measurement uncertainty)\n")
    md.append(f"Verdict **{gate.get('verdict')}**; parts: " + ", ".join(f"{k} = {v}" for k, v in (gate.get("parts") or {}).items()))
    md.append("\n| Case | current P95 | coverage | B worse on |")
    md.append("|---|---|---|---|")
    for c, worse in (gate.get("B_worse") or {}).items():
        pc = gate["per_case"].get(c, {})
        md.append(f"| {c} | {pc.get('current_p95_s', {}).get('decision', '—')} | {pc.get('coverage', {}).get('decision', '—')} | "
                  f"{', '.join(worse) if worse else '—'} |")
    # ---- perf
    if perf:
        ev["performance"] = perf
        md.append("\n### Performance\n")
        md.append("| State | particles | A extract warm (s) | A mesh V/T | A GPU mempool high (MB) | B full (s) | B topology+QEF | "
                  "B embedding | B audit | B Hermite | B unchanged frame (s) | B mesh V/T | B RSS setup/extract (MB) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, s in perf["states"].items():
            if "particles" not in s:
                continue
            pa, pb = s[A], s[B]
            br = pb["full_rebuild_breakdown_median_s"]
            md.append(f"| {k} | {s['particles']} | {pa['extract_s_median_warm']:.3f} | {pa['vertices']}/{pa['triangles']} | "
                      f"{pa['gpu_mempool_used_high_bytes']/1e6:.0f} | {pb['full_rebuild_median_s']:.3f} | {br['topology_and_qef_s']:.3f} | "
                      f"{br['embedding_s']:.3f} | {br['audit_s']:.3f} | {pb['cut_event_processing_hermite_median_s']:.4f} | "
                      f"{pb['unchanged_frame_median_s']:.3f} | {pb['vertices']}/{pb['triangles']} | "
                      f"{pb['rss']['setup_delta_bytes']/1e6:.0f}/{pb['rss']['extraction_delta_bytes']/1e6:.0f} |")
        md.append("\nIncremental vs full rebuild per recorded frame:\n")
        md.append("| State | frame | dirty cells | incremental (s) | full (s) | identical |")
        md.append("|---|---|---|---|---|---|")
        for k, s in perf["states"].items():
            for r in (s.get(B) or {}).get("incremental_sequence", []):
                md.append(f"| {k} | {r['frame']} | {r['update']['dirty_cells']} ({r['update']['mode']}) | "
                          f"{r['incremental_wall_s']:.3f} | {r['full_wall_s']:.3f} | {r['identical']} |")
    if diag:
        ev["diagnostics"] = diag
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(ev, separators=(",", ":")))
    k = 1
    while (run/f"tables_{k:03d}.md").exists():
        k += 1
    (run/f"tables_{k:03d}.md").write_text("\n".join(md)+"\n")
    print("\n".join(md))
    print(f"\nwrote {evidence} and {run/f'tables_{k:03d}.md'}")


if __name__ == "__main__":
    main()
