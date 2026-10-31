#!/usr/bin/env python3
"""Compact evidence + markdown tables of the Phase 3 v2 matrix (artifacts/wp12/phase3_surface_002).

  .venv/bin/python scripts/summarize_wp12_phase3_v2.py --run artifacts/wp12/phase3_surface_002 --summary summary_001.json \
      --evidence docs/evidence/wp12/phase3_surface_002.json

Reads only; writes the evidence JSON (refuses to overwrite) and <run>/tables_NNN.md (next free number).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A, B1, B2 = "current_volume_v1", "cut_aware_surface_v1", "cut_aware_surface_v2"
B2PC, B2NOM4 = "cut_aware_surface_v2_position_consistent", "cut_aware_surface_v2_ablation_no_M4"
SHORT = {A: "A", B1: "B1", B2: "B2", B2PC: "B2pc", B2NOM4: "B2-noM4"}
PHYSICAL = ("S1_G3a", "S1_G3b", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
SETS = ("saved", "H1", "H2")
KEYS = ("current_p95_s", "current_max_unambiguous_s", "reference_p95_s", "matched_normal_p95_deg", "triangle_normal_p95_deg",
        "abs_thickness_bias_s", "coverage", "unexplained_intersections_sampled", "self_intersections_sampled",
        "faces_nearer_other_component_unexplained", "non_manifold_edges", "shells_equal_components",
        "volume_current_abs_rel")


def load(run, tag, case, r):
    p = run/f"eval_{tag}/{case}_r{r}.json"
    if tag != "saved" and case == "S9":
        p = run/f"heldout_{tag}/S9_r{r}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=2):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "T" if x else "F"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    return f"{x:.{nd}f}"


def span(vals, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    if all(isinstance(v, bool) for v in vals):
        return "/".join(fmt(v) for v in vals)
    lo, hi = min(vals), max(vals)
    return fmt(lo, nd) if abs(hi-lo) < 10**(-nd)/2 else f"{fmt(lo, nd)}–{fmt(hi, nd)}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()
    run = args.run if args.run.is_absolute() else ROOT/args.run
    summary = json.loads((run/args.summary).read_text())
    ev = {"schema": "wp12_phase3_surface_002_evidence_v1", "run": str(run.relative_to(ROOT)), "summary_file": args.summary,
          "protocol_record_sha256": summary["protocol_record_sha256"], "method_frozen_sha256": summary["method_frozen_sha256"],
          "phase3_gate": summary["phase3_gate"], "S0": summary.get("S0"), "G2": summary.get("G2"), "sets": {}}
    md = []
    for tag in SETS:
        if tag not in summary["sets"]:
            continue
        entry = summary["sets"][tag]
        es = {"complete": entry["complete"], "verdicts": entry["verdicts"],
              "gates": {k: {"verdict": g.get("verdict"), "parts": g.get("parts"), "B_worse": g.get("B_worse"),
                            "current_p95_decisions": g.get("current_p95_decisions"),
                            "coverage_decisions": g.get("coverage_decisions"), "topology_decisions": g.get("topology_decisions")}
                        for k, g in entry["gates"].items()},
              "cases": {}}
        for case in PHYSICAL:
            reps = [load(run, tag, case, r) for r in range(args.repeats)]
            reps = [r for r in reps if r is not None]
            if not reps:
                continue
            row = {"repeats": len(reps), "scalars_v2": {}, "v2_counts": [], "extract_wall_s": {}, "analysis_wall_s": [],
                   "P4_status": [r["frames"][-1]["v2_state"]["P4_reference_consistency"]["status"] for r in reps],
                   "P2_floor_p95_deg": [r["frames"][-1]["v2_state"]["P2_floor_p95_deg"] for r in reps]}
            for kind in (A, B1, B2, B2PC, B2NOM4):
                if kind not in reps[0]["frames"][-1]["reports_v2"]:
                    continue
                row["scalars_v2"][kind] = {k: [r["frames"][-1]["reports_v2"][kind]["scalars"].get(k) for r in reps] for k in KEYS}
                row["extract_wall_s"][kind] = [r["frames"][-1]["reports"][kind]["extract_wall_s"] for r in reps]
                extra = [r["frames"][-1]["reports_v2"][kind] for r in reps]
                row["scalars_v2"][kind]["intersections_lattice_faces_only"] = [e.get("intersections_lattice_faces_only") for e in extra]
                row["scalars_v2"][kind]["intersections_before_fold_guard"] = [e.get("intersections_before_fold_guard") for e in extra]
                row["scalars_v2"][kind]["P3_worst_excess_s"] = [((e.get("blocks") or {}).get("all") or {}).get("current_p95_P3", {}).get("worst_excess_s") for e in extra]
            for r in reps:
                info = r["frames"][-1]["reports"][B2]["cut_aware_info"]["v2"]
                row["v2_counts"].append({"M1_constrained": info["M1_constrained_qef"]["constrained"],
                                         "M2_moved": info["M2_hinge"]["moved"], "M2_mixed": info["M2_hinge"]["mixed_regions"],
                                         "M2_no_hinge": info["M2_hinge"]["mixed_without_hinge_corner"],
                                         "M3_flipped_before": info["M3_fold_guard"].get("flipped_triangles_before"),
                                         "M3_flipped_after": info["M3_fold_guard"].get("flipped_triangles_after"),
                                         "M3_reps_lsq": info["M3_fold_guard"].get("representatives_lsq"),
                                         "M3_reps_translation": info["M3_fold_guard"].get("representatives_translation_only"),
                                         "M4_flagged_links": info["M4_unlabelled_separation"]["flagged_links"],
                                         "M4_cells": info["M4_unlabelled_separation"]["fallback_particles"]})
                row["analysis_wall_s"].append(r["analysis_wall_s"])
            if case == "S8":
                row["P5"] = [{"eligible_pairs": (r.get("p5") or {}).get("eligible_pairs"),
                              "max_p95_s": (r.get("p5") or {}).get("max_p95_s")} for r in reps]
            if case == "S7":
                row["P7"] = [{k: {"value": v.get("value"), "verdict": v.get("verdict"),
                                  "lattice_faced": v.get("lattice_faced"),
                                  "particle_cell_fallback_by_construction": v.get("particle_cell_fallback_by_construction")}
                              for k, v in (r.get("s7_p7") or {}).items()} for r in reps]
            es["cases"][case] = row
        ev["sets"][tag] = es
        # tables
        md.append(f"\n### State set `{tag}` (complete: {entry['complete']})\n")
        for proto in ("v1", "v2"):
            md.append(f"\nThreshold verdicts, protocol {proto} (r0 / r1):\n")
            md.append("| Case | A | B v1 | B v2 | B v2 pc (label) | B v2 no-M4 (ablation) |\n|---|---|---|---|---|---|")
            for case in PHYSICAL:
                vs = entry["verdicts"].get(case)
                if not vs:
                    continue
                cells = [" / ".join(v[proto].get(k, "–") for v in vs) for k in (A, B1, B2, B2PC, B2NOM4)]
                md.append(f"| {case} | " + " | ".join(cells) + " |")
            if "S9" in entry["verdicts"]:
                s9 = entry["verdicts"]["S9"]
                for k in (A, B1, B2, B2PC):
                    allpass = all(v.get(k) == "PASS" for rr in s9 for v in rr.values())
                    md.append(f"| S9 ({SHORT[k]}) | all five controls x {len(s9)} repeats: {'PASS' if allpass else 'see JSON'} |")
        md.append("\nKey v2-reading scalars (range over repeats):\n")
        md.append("| Case | Method | current P95 s | P3 worst excess s | ref P95 s | matched normal P95° | tri normal P95° | thickness s | unexpl. inters. | inters. lattice-only | non-manifold | shells=comp |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for case, row in es["cases"].items():
            for kind, sc in row["scalars_v2"].items():
                md.append(f"| {case} | {SHORT[kind]} | {span(sc['current_p95_s'])} | {span(sc['P3_worst_excess_s'])} | "
                          f"{span(sc['reference_p95_s'])} | {span(sc['matched_normal_p95_deg'], 1)} | "
                          f"{span(sc['triangle_normal_p95_deg'], 1)} | {span(sc['abs_thickness_bias_s'])} | "
                          f"{span(sc['unexplained_intersections_sampled'], 0)} | {span(sc['intersections_lattice_faces_only'], 0)} | "
                          f"{span(sc['non_manifold_edges'], 0)} | {span(sc['shells_equal_components'])} |")
        md.append("\nB v2 activations per repeat (M1 constrained / M2 moved of mixed / M3 flips before->after (reps LSQ, translation) / M4 flagged links, cells):\n")
        md.append("| Case | r0 | r1 |\n|---|---|---|")
        for case, row in es["cases"].items():
            cells = [f"{c['M1_constrained']} / {c['M2_moved']} of {c['M2_mixed']} / {c['M3_flipped_before']}->{c['M3_flipped_after']} "
                     f"({c['M3_reps_lsq']}, {c['M3_reps_translation']}) / {c['M4_flagged_links']}, {c['M4_cells']}" for c in row["v2_counts"]]
            md.append(f"| {case} | " + " | ".join(cells) + " |")
        md.append("\nExtraction wall time per final frame, s (A on cuda:0 incl. GPU; B CPU NumPy), range over repeats:\n")
        md.append("| Case | A | B v1 | B v2 |\n|---|---|---|---|")
        for case, row in es["cases"].items():
            w = row["extract_wall_s"]
            md.append(f"| {case} | {span(w.get(A))} | {span(w.get(B1))} | {span(w.get(B2))} |")
        md.append("\nComparison gates (parts):\n")
        md.append("| Gate | verdict | not worse | better P95 every case | better coverage | better topology |\n|---|---|---|---|---|---|")
        for k, g in es["gates"].items():
            pa = g.get("parts") or {}
            md.append(f"| {k} | {g.get('verdict')} | {pa.get('not_worse_on_any_metric')} | {pa.get('better_current_p95_every_case')} | "
                      f"{pa.get('better_coverage_partial_and_chip')} | {pa.get('better_topology')} |")
        for k, g in es["gates"].items():
            worse = {c: m for c, m in (g.get("B_worse") or {}).items() if m}
            md.append(f"\n`{k}` B worse: " + ("; ".join(f"{c}: {', '.join(m)}" for c, m in worse.items()) or "none"))
            md.append(f"\n`{k}` current P95: " + ", ".join(f"{c} {d}" for c, d in (g.get("current_p95_decisions") or {}).items()))
    md.insert(0, "## Phase 3 v2 gate\n\n| Gate | saved | H1 | H2 | verdict |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {k} | " + " | ".join(str((v["per_set"].get(t) or ["–"])[0]) for t in SETS) + f" | {v['verdict']} |"
        for k, v in summary["phase3_gate"].items()))
    ev_path = args.evidence if args.evidence.is_absolute() else ROOT/args.evidence
    if ev_path.exists():
        raise SystemExit(f"Refusing to overwrite {ev_path}")
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    ev_path.write_text(json.dumps(ev, indent=1))
    k = 1
    while (run/f"tables_{k:03d}.md").exists():
        k += 1
    (run/f"tables_{k:03d}.md").write_text("\n".join(md)+"\n")
    print("\n".join(md))
    print(f"\nwrote {ev_path} and {run/f'tables_{k:03d}.md'}")


if __name__ == "__main__":
    main()
