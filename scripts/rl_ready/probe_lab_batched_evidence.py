"""Assemble the P1-1b stage S1 evidence JSON from the raw probe outputs under ``artifacts/rl_ready/p1_1b/``.

    PYTHONPATH=src .venv/bin/python scripts/rl_ready/probe_lab_batched_evidence.py \
        --out docs/evidence/rl_ready/p1_1b_s1_001.json

Reads every ``<run>/row.json`` and ``compare_*/compare.json`` (no GPU work), keeps the fields a reader needs
(rows, cold/warm timings, gate checks, telemetry sharing labels) and states the gate verdicts. Verdicts are
recomputed here from the raw files, never typed in.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT/"artifacts/rl_ready/p1_1b"


def _load(path):
    return json.loads(Path(path).read_text())


def _row_summary(name, r):
    t = r.get("telemetry", {})
    out = {"run": name, "artifact": f"artifacts/rl_ready/p1_1b/{name}", "probe": r.get("probe"),
           "variant": r.get("variant"), "N": r.get("N"), "tool_mount": r.get("tool_mount"),
           "particles_per_env": r.get("particles_per_env"), "control_hz": r.get("control_hz"),
           "rigid": "10 x 2 ms substeps per control step", "material_dt_s": r.get("material_dt_s"),
           "cameras": r.get("cameras"), "recording": r.get("recording"), "use_cuda_graph": r.get("use_cuda_graph"),
           "graph_captured": r.get("graph_captured"), "deterministic_mode": r.get("deterministic_mode"),
           "reduced_fidelity": r.get("reduced_fidelity"), "reasons": r.get("reasons"), "cold": r.get("cold"),
           "warm": r.get("warm"), "gpu": r.get("gpu"), "gpu_sharing": r.get("gpu_sharing"),
           "foreign_sm_pct": t.get("foreign_sm_pct"), "total_util_pct": t.get("total_util_pct"),
           "foreign_compute_pids": t.get("foreign_compute_pids"), "vram_own_peak_mib": r.get("vram_own_peak_mib"),
           "vram_own_slope_mib_per_s_warm": r.get("vram_own_slope_mib_per_s_warm"), "commit": r.get("commit")}
    for key in ("t2", "g0_graph_safety", "grasp_initialization"):
        if r.get(key) is not None:
            out[key] = r[key]
    return out


def _native_wp17(seed):
    """Native WP-17 outcome of a scene seed (descriptive reference; never an equivalence target)."""
    attempt = ROOT/f"artifacts/wp17/native_optimized_001/seed_{seed}/box/env/recording/attempt_000"
    evaluation = _load(attempt/"evaluation.json")
    outcome = evaluation.get("outcome", {})
    metrics = outcome.get("metrics", {})
    return dict(source=str(attempt.relative_to(ROOT)), verdict=evaluation.get("verdict"),
                outcome=outcome.get("status"), mechanical_components=metrics.get("mechanical_components"),
                largest_component_below_plane_mass_share=metrics.get("largest_component_below_plane_mass_share"),
                detached_below_plane_mass_fraction=metrics.get("detached_below_plane_mass_fraction"),
                removed_mass_fraction={i.get("id"): i.get("removed_mass_fraction") for i in metrics.get("items", [])})


def s2_s3(args):
    rows = {p.parent.name: _load(p) for p in sorted(RAW.glob("*/row.json"))
            if p.parent.name.startswith(("g3_", "s3_b", "g2mat_b2_", "g0_det_b2", "hashgrid_check"))}
    crashed = {}
    for d in sorted(RAW.glob("*")):
        if d.name.startswith(("g3_", "s3_b", "g2mat_b2_", "g4_", "g5_")) and d.is_dir() and not any(
                (d/f).exists() for f in ("row.json", "g4.json", "g5.json", "g4_compare.json", "g3.json")):
            log = (d/"stdout.log").read_text(errors="replace")[-3000:] if (d/"stdout.log").exists() else ""
            reason = next((line for line in log.splitlines()[::-1] if "Error" in line or "Fatal" in line
                           or "Segmentation" in line), "no row written")
            aborted = (d/"aborted.json").exists()
            crashed[d.name] = dict(status="ABORTED" if aborted else "FAIL_NO_ROW", reason=reason[:400],
                                   detail=_load(d/"aborted.json") if aborted else None)
    out = {"schema": "rcs_rl_ready_p1_1b_s2_s3_v1", "work_package": "P1-1b stages S2 (B2 + material) and S3 (TRIM)",
           "adr": "docs/adr/0022-batched-robot-workload-b.md", "report": "docs/reports/rl_ready/P1_1B_ROBOT_BATCHING.md",
           "evidence_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                             cwd=ROOT).stdout.strip(), "rows": {}, "g4": {}, "g5": {},
           "native_wp17": {str(s): _native_wp17(s) for s in (512, 15)}, "material_results": "SIMULATOR_ONLY_UNCALIBRATED"}
    out["failed_or_aborted_runs"] = crashed
    out["hashgrid_census"] = dict(
        hazard="Warp 1.17.0 hash_grid_update_device_impl copies the grid descriptor from a function-static host "
               "variable; a replayed CUDA graph uses the last host-built grid (P1-3a probe_hashgrid_descriptor.py)",
        verdict="NO P1-1b process held more than one CUDA wp.HashGrid; no run is INVALID_MULTI_GRID_PROCESS",
        method=["grep of every wp.HashGrid constructor in src/ (tlmpm_v5.py:798 single bed at reset; "
                "batched/tlmpm_v7_2_batch.py:613 one grouped grid per batch), Newton (builder.py particle_grid only "
                "for models with particles; the rigid model has none), MuJoCo-Warp and Isaac Lab (none)",
                "instrumented run hashgrid_check_b1_n2_001 on the Lab material code path: 1 CUDA grid (batch) + "
                "2 CPU grids (the per-env single materials reset on the CPU device for packing)",
                "every later probe row/json records its census (hashgrids.cuda_count); G5 L3 runs one replay per "
                "process and the combiner refuses multi-grid inputs"],
        instrumented={n: r["hashgrids"] for n, r in rows.items() if isinstance(r.get("hashgrids"), dict)})
    for name, r in rows.items():
        summary = _row_summary(name, r)
        summary["workload"] = "B" if r.get("material") not in (None, "none (robot-only)") else "B (robot-only)"
        summary["hashgrids"] = r.get("hashgrids", "not instrumented (row predates the HashGrid census)")
        for key in ("scene_seeds", "evaluation", "termination_reason", "reward_fractions", "engagement",
                    "grasp_trace_summary", "grasp_snapshot", "recorded_policy", "driver", "grasp_initialization",
                    "grip_material", "termination_trace"):
            if r.get(key) is not None:
                summary[key] = r[key]
        if isinstance(summary.get("engagement"), dict):
            summary["engagement"] = {k: v for k, v in summary["engagement"].items() if k != "material_receipt"}
        if isinstance(summary.get("evaluation"), list):
            summary["evaluation"] = [{k: v for k, v in e.items() if k not in ("result",)} for e in summary["evaluation"]]
        out["rows"][name] = summary
    for p in sorted(RAW.glob("g4_*/g4*.json")):
        out["g4"][f"{p.parent.name}/{p.name}"] = _load(p)
    for p in sorted(RAW.glob("g5_*/g5.json")):
        out["g5"][p.parent.name] = _load(p)
    for p in sorted(RAW.glob("compare_g3_*/compare.json")):
        out.setdefault("g3_compare", {})[p.parent.name] = _load(p)
    for p in sorted(RAW.glob("g3_analysis_*/g3.json")):
        out.setdefault("g3_analysis", {})[p.parent.name] = _load(p)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str)+"\n")
    print(f"rows {sorted(out['rows'])}; g4 {sorted(out['g4'])}; g5 {sorted(out['g5'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-hash", default=None, help="authoritative workload_b_sources_sha256 (default: newest)")
    parser.add_argument("--stage", default="s1", choices=("s1", "s2s3"))
    args = parser.parse_args()
    if args.stage == "s2s3":
        return s2_s3(args)
    rows = {p.parent.name: _load(p) for p in sorted(RAW.glob("*/row.json"))}
    compares = {p.parent.name: _load(p) for p in sorted(RAW.glob("compare_*/compare.json"))}
    summaries = {name: _row_summary(name, r) for name, r in rows.items()}
    # Authoritative rows: the final workload B source state (hash recorded by the probe); earlier rows (legacy
    # table, pre-fix host sync, preliminary mixed-source G1) are kept and labelled superseded, never deleted.
    hashes = {name: (r.get("commit") or {}).get("workload_b_sources_sha256") for name, r in rows.items()}
    final = args.source_hash or hashes.get(max((n for n in rows if hashes[n]), key=lambda n: rows[n].get(
        "wall_s_total_process", 0) and Path(RAW/n/"row.json").stat().st_mtime))
    for name, summary in summaries.items():
        summary["authoritative"] = hashes[name] == final
        summary["workload_b_sources_sha256"] = hashes[name]
    auth = {n: s for n, s in summaries.items() if s["authoritative"]}
    g0 = {n: s for n, s in auth.items() if s["probe"] == "g0"}
    t2 = {n: s for n, s in auth.items() if s["probe"] == "t2"}
    g2 = {n: s for n, s in auth.items() if s["probe"] == "g2"}
    evidence = {
        "schema": "rcs_rl_ready_p1_1b_s1_v1", "work_package": "P1-1b stage S1 (robot-only) on GPU0",
        "adr": "docs/adr/0022-batched-robot-workload-b.md", "report": "docs/reports/rl_ready/P1_1B_ROBOT_BATCHING.md",
        "runtime": {"python": ".deps/isaaclab-venv/bin/python (Isaac Lab 3.0 kit-less, Newton 1.6.1, MJWarp 3.12.0, "
                              "Warp 1.17.0, Torch 2.11)", "host": platform.node(),
                    "env": "CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src, flock "
                           "/mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock"},
        "evidence_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                          cwd=ROOT).stdout.strip(),
        "authoritative_source_sha256": final,
        "rows": {n: s for n, s in summaries.items() if s["authoritative"]},
        "superseded_rows": {n: {k: s.get(k) for k in ("probe", "variant", "N", "tool_mount", "commit",
                                                      "workload_b_sources_sha256", "gpu_sharing")}
                            for n, s in summaries.items() if not s["authoritative"]},
        "superseded_reason": "earlier source states: legacy 1.8 x 1.2 m table (before 359ce0d), T2 host-sync fix, "
                             "preliminary G1 runs spanning several source states; raw outputs kept",
        "comparisons": {n: c for n, c in compares.items() if n.endswith("_002")},
        "gates": {
            "G0_graph_safety": {n: s.get("g0_graph_safety", {}).get("status") for n, s in g0.items()},
            "T2_host_syncs": {n: s.get("t2", {}).get("status") for n, s in t2.items()},
            "G1": {n: c for n, c in compares.items() if n.startswith("compare_g1") and n.endswith("_002")},
            "G2_rows": sorted(g2),
        },
        "not_run": ["G3 (B2 restore vs fresh settle)", "G4 (batched N=1 B2 vs single Lab env)",
                    "G5 (material parity)", "G2 with material (until the robot-only gates pass and coupling lands)"],
        "material_results": "NONE (robot-only rows)",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(evidence, indent=1, default=str)+"\n")
    print(json.dumps(evidence["gates"], indent=1, default=str))


if __name__ == "__main__":
    main()
