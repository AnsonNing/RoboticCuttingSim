#!/usr/bin/env python3
"""Compact P1-3a evidence from the raw ``artifacts/rl_ready/p1_3`` runs (torch-free; ``.venv`` is fine).

Usage::

    PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_p1_3a_evidence.py \\
        --samestate samestate_s512_k33_002 samestate_s15_k43_002 --ab ab_n1_001 ab_n4_001 ab_n8_001 ab_n16_001 \\
        --episode episode_512_001 episode_15_001 episode_512_15_512_15_001 --nsys nsys_001 \\
        --failed samestate_s512_k33_001 samestate_s15_k43_001 --extra extra.json \\
        --out docs/evidence/rl_ready/p1_3a_001.json
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.rl.benchmark import BenchmarkRow, co_tenant_stats, markdown_table  # noqa: E402

P13 = ROOT / "artifacts" / "rl_ready" / "p1_3"
CONTACTS = "rigid tool + handle/2 measured pad boxes + fixtures + particle contacts (material only)"
REDUCED = "workload A: ideal_edge_tracking_no_robot_dynamics, reaction_not_fed_back"
TELEMETRY_KEYS = ("gpu_sharing", "sharing_reasons", "own_vram_peak_mib", "foreign_sm_pct", "foreign_memory_mib",
                  "total_util_pct", "gpu_name", "gpu_index", "gpu_driver", "valid_samples", "foreign_compute_pids",
                  "foreign_process_count_max")


def load(run):
    data = json.loads((P13 / run / "result.json").read_text())
    data["_manifest"] = json.loads((P13 / run / "manifest.json").read_text())
    data["_run"] = str((P13 / run).relative_to(ROOT / "artifacts"))
    return data


def gpu(data):
    t = data.get("telemetry_run_phase") or data["telemetry"]
    return dict(gpu_name=t.get("gpu_name"), gpu_index=t.get("gpu_index"), gpu_driver=t.get("gpu_driver"),
                gpu_sharing=t.get("gpu_sharing", "UNKNOWN"), co_tenant=co_tenant_stats(t),
                vram_peak_mib=data["telemetry"].get("own_vram_peak_mib"))


def short_telemetry(t):
    return {k: t.get(k) for k in TELEMETRY_KEYS} if t else None


def ab_rows(data):
    rows = []
    g = gpu(data)
    N = data["num_envs"]
    particles = data["particles_per_env"]
    for name, w in data["step_wall_s"].items():
        eps = data["env_steps_per_s"][name]
        same = data["contact_counts_identical_to_first"][name]
        rows.append(BenchmarkRow(
            boundary="batched_headless", workload_id=f"A-replay s512/s15 alternating, steps 3-{2 + w['n']} "
            f"[{name}]", num_envs=N, particles_per_env=int(round(sum(particles) / len(particles))), control_hz=50.,
            material_dt_s=1e-4, substeps_per_control_step=200, cameras="none", robot_dof=0, contacts=CONTACTS,
            recording="none", warm_cold="warm", env_steps_per_s=eps, aggregate_sim_time_per_wall=eps * .02,
            vram_peak_mib=g["vram_peak_mib"], reset_mean_s=None, reset_p95_s=None,
            physics_verdict="REPORTED_NOT_SCORED", reduced_fidelity=True, gpu_name=g["gpu_name"],
            gpu_index=g["gpu_index"], gpu_driver=g["gpu_driver"], gpu_sharing=g["gpu_sharing"],
            co_tenant=g["co_tenant"], source_commit=data["_manifest"]["git_head"], source=data["_run"],
            notes=f"{REDUCED}; interleaved A/B in one process ({', '.join(data['variants'])}); median step "
                  f"{w['median']:.4f} s (min {w.get('min', float('nan')):.4f}); per-step contact counts identical "
                  f"to {data['variants'][0]}: {same}; VRAM column = whole process (all variants)"))
    return rows


def episode_rows(data):
    rows = []
    g = gpu(data)
    N = data["num_envs"]
    for name, t in data["timing"].items():
        r = data["results"][name]
        verdict = "PASS" if (r["l2"]["status"] == "PASS" and all(
            e["evaluation_2_frames"]["verdict"] == "PASS" and e["evaluation_121_frames"]["verdict"] == "PASS"
            and e["final"]["codes_equal_native"] for e in r["envs"])) else "FAIL"
        rows.append(BenchmarkRow(
            boundary="batched_headless", workload_id=f"A-replay full episode seeds {data['seeds']} [{name}]",
            num_envs=N, particles_per_env=None, control_hz=50., material_dt_s=1e-4, substeps_per_control_step=200,
            cameras="none", robot_dof=0, contacts=CONTACTS, recording="none", warm_cold="warm",
            env_steps_per_s=t["warm_env_steps_per_s"], aggregate_sim_time_per_wall=t["warm_env_steps_per_s"] * .02,
            vram_peak_mib=g["vram_peak_mib"], reset_mean_s=None, reset_p95_s=None, physics_verdict=verdict,
            reduced_fidelity=True, gpu_name=g["gpu_name"], gpu_index=g["gpu_index"], gpu_driver=g["gpu_driver"],
            gpu_sharing=g["gpu_sharing"], co_tenant=g["co_tenant"], source_commit=data["_manifest"]["git_head"],
            source=data["_run"], notes=f"{REDUCED}; 1200 steps; variants interleaved per step "
                                       f"({', '.join(data['variants'])}); verdict = L2 + frozen evaluator (2 and 121 "
                                       "frames) + final codes equal Native"))
    return rows


def nsys_summary(directory):
    out = {}
    for path in sorted((P13 / directory).glob("*_cuda_gpu_kern_sum.csv")):
        label = path.name.replace("_cuda_gpu_kern_sum.csv", "")
        kernels = {}
        for r in csv.DictReader(path.open()):
            kernels[r["Name"]] = dict(instances=int(r["Instances"]), total_ns=float(r["Total Time (ns)"]),
                                      med_ns=float(r["Med (ns)"]), min_ns=float(r["Min (ns)"]),
                                      avg_ns=float(r["Avg (ns)"]))
        total = sum(k["total_ns"] for k in kernels.values())
        out[label] = dict(kernel_total_ns=total, kernels=kernels)
    return dict(status="MEASURED" if out else "NOT_RUN", run=f"rl_ready/p1_3/{directory}", reports=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samestate", nargs="*", default=[])
    parser.add_argument("--ab", nargs="*", default=[])
    parser.add_argument("--episode", nargs="*", default=[])
    parser.add_argument("--nsys", nargs="*", default=[])
    parser.add_argument("--failed", nargs="*", default=[])
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, evidence = [], dict(schema="rcs_rl_p1_3a_evidence_v1", runs={})
    for run in args.samestate:
        d = load(run)
        evidence["runs"][run] = {k: d.get(k) for k in (
            "status", "error", "seed", "sample", "time_s", "copies", "checks", "frozen_vs_alias_free",
            "frozen_vs_frozen_rerun", "alias_free_vs_alias_free_rerun", "frozen_vs_forced_fallback",
            "after_graph_replays_frozen_vs_alias_free", "after_graph_replays_frozen_vs_forced_fallback",
            "copies_identical_alias_free", "speed_m_s_max", "guard_violations", "forced_fallback_guard_violations",
            "stage_graph_ms", "stage_graph_scope", "stage_eager_activity_ms", "alias_free_receipt", "wall_s")}
        evidence["runs"][run]["telemetry"] = short_telemetry(d.get("telemetry_run_phase") or d["telemetry"])
    for run in args.ab:
        d = load(run)
        evidence["runs"][run] = {k: d.get(k) for k in (
            "status", "error", "num_envs", "seeds", "variants", "steps", "step_wall_s", "env_steps_per_s",
            "speedup_vs_first_median", "contact_counts_identical_to_first", "contact_counts_first_difference_step",
            "contact_counts_total", "final_vs_first", "guard_violations", "receipts", "particles_per_env",
            "memory_bytes", "mempool_high_bytes_all_envs", "activity_ms_per_profiled_step", "activity_profiled_steps",
            "timing_scope", "build_wall_s", "peak_rss_mib", "wall_s")}
        evidence["runs"][run]["telemetry"] = short_telemetry(d.get("telemetry_run_phase") or d["telemetry"])
        if d["status"] == "COMPLETE":
            rows += ab_rows(d)
    for run in args.episode:
        d = load(run)
        evidence["runs"][run] = {k: d.get(k) for k in (
            "status", "error", "num_envs", "seeds", "variants", "complete", "payload_check", "results",
            "cross_variant", "timing", "timing_scope", "mempool_high_bytes", "peak_rss_mib", "wall_s")}
        evidence["runs"][run]["telemetry"] = short_telemetry(d.get("telemetry_run_phase") or d["telemetry"])
        if d["status"] == "COMPLETE":
            rows += episode_rows(d)
    for run in args.failed:
        d = json.loads((P13 / run / "result.json").read_text())
        evidence["runs"][run] = dict(status=d["status"], error=(d.get("error") or "")[-1200:], preserved=True)
    evidence["nsys"] = [nsys_summary(r) for r in args.nsys]
    evidence["rows"] = [r.to_dict() for r in rows]
    evidence["markdown_table"] = markdown_table(rows)
    if args.extra:
        evidence.update(json.loads(args.extra.read_text()))
    args.out.write_text(json.dumps(evidence, indent=1, sort_keys=True, default=str) + "\n")
    print(evidence["markdown_table"])


if __name__ == "__main__":
    main()
