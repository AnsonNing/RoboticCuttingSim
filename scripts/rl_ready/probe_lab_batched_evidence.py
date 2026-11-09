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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-hash", default=None, help="authoritative workload_b_sources_sha256 (default: newest)")
    args = parser.parse_args()
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
        "authoritative_source_sha256": final, "rows": summaries, "comparisons": compares,
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
