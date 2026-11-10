#!/usr/bin/env python3
"""Compact P1-2 milestone evidence from the raw ``artifacts/rl_ready/p1_2`` runs (torch-free; ``.venv`` is fine).

Usage::

    PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_p1_2_evidence.py \\
        --m0 m0_s512_001 m0_s15_001 --m1 m1_n4_001 --l4 l4_deterministic_001 \\
        --m2 m2_n1_001 m2_n8_001 m2_n1_002 m2_n16_001 --single p1_0/p1_2_single_ref_s512_001 \\
        --nsys nsys_n1_001 nsys_n4_001 --out docs/evidence/rl_ready/p1_2_milestones_001.json
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.rl.benchmark import BenchmarkRow, co_tenant_stats, markdown_table  # noqa: E402

P12 = ROOT / "artifacts" / "rl_ready" / "p1_2"
CONTACTS = "rigid tool + handle/2 measured pad boxes + fixtures + particle contacts (material only)"
REDUCED = "workload A: ideal_edge_tracking_no_robot_dynamics, reaction_not_fed_back"


def load(run, root=P12):
    path = root / run / "result.json"
    data = json.loads(path.read_text())
    data["_manifest"] = json.loads((root / run / "manifest.json").read_text())
    data["_run"] = str((root / run).relative_to(ROOT / "artifacts"))
    return data


def gpu(row_data):
    t = row_data["telemetry"]
    return dict(gpu_name=t.get("gpu_name"), gpu_index=t.get("gpu_index"), gpu_driver=t.get("gpu_driver"),
                gpu_sharing=t.get("gpu_sharing", "UNKNOWN"), co_tenant=co_tenant_stats(t),
                vram_peak_mib=t.get("own_vram_peak_mib"))


def particles(data):
    counts = data.get("particles_per_env") or data.get("receipt_particles")
    return int(round(sum(counts) / len(counts))) if counts else None


def evaluations_pass(episodes):
    return all(e["evaluation"]["verdict"] == "PASS" for e in episodes) if episodes else False


def m2_row(data, workload, throughput_only=False):
    g = gpu(data)
    t = data["timing"]
    verdict = "PASS" if data["l2"]["status"] == "PASS" and evaluations_pass(data["episodes"]) else "FAIL"
    if throughput_only:
        verdict = "REPORTED_NOT_SCORED"
    return BenchmarkRow(
        boundary="batched_headless", workload_id=workload, num_envs=data["num_envs"],
        particles_per_env=particles(data), control_hz=50., material_dt_s=1e-4, substeps_per_control_step=200,
        cameras="none", robot_dof=0, contacts=CONTACTS, recording="none", warm_cold="warm",
        env_steps_per_s=t["env_steps_per_s"], aggregate_sim_time_per_wall=t["env_steps_per_s"] * .02,
        vram_peak_mib=g["vram_peak_mib"], reset_mean_s=t["partial_reset_mean_s"], reset_p95_s=t["partial_reset_p95_s"],
        physics_verdict=verdict, reduced_fidelity=True, gpu_name=g["gpu_name"], gpu_index=g["gpu_index"],
        gpu_driver=g["gpu_driver"], gpu_sharing=g["gpu_sharing"], co_tenant=g["co_tenant"],
        source_commit=data["_manifest"]["git_head"], source=data["_run"],
        notes=f"{REDUCED}; replay drive (recorded measured edge + pads); {t['global_steps']} batch steps, "
              f"{len(data['episodes'])} full episodes evaluated; per-step device sync for timing; reset = masked "
              f"partial reset (host call); L2 every {data['l2'].get('update_every_steps')} step(s)"
              + ("; THROUGHPUT ALTERNATION window (partial episode, no evaluation; audits only)" if throughput_only
                 else ""))


def cold_row(data, workload):
    g = gpu(data)
    t = data["timing"]
    cold = t.get("cold_process_start_to_first_step_done_s")
    return BenchmarkRow(
        boundary="batched_headless", workload_id=workload, num_envs=data["num_envs"],
        particles_per_env=particles(data), control_hz=50., material_dt_s=1e-4, substeps_per_control_step=200,
        cameras="none", robot_dof=0, contacts=CONTACTS, recording="none", warm_cold="cold",
        env_steps_per_s=None, aggregate_sim_time_per_wall=None, vram_peak_mib=g["vram_peak_mib"],
        reset_mean_s=t["first_reset_s"], reset_p95_s=None, physics_verdict="REPORTED_NOT_SCORED",
        reduced_fidelity=True, gpu_name=g["gpu_name"], gpu_index=g["gpu_index"], gpu_driver=g["gpu_driver"],
        gpu_sharing=g["gpu_sharing"], co_tenant=g["co_tenant"], source_commit=data["_manifest"]["git_head"],
        source=data["_run"], notes=f"cold = process start -> first control step done {cold:.2f} s (user Warp "
                                   "kernel cache; build/pack/capture included); reset column = first full reset")


def m0_row(data):
    g = gpu(data)
    t = data["timing"]
    verdict = "PASS" if data["l2"]["status"] == "PASS" and all(
        e["verdict"] == "PASS" for e in data["evaluation_2_frames"]) else "FAIL"
    return BenchmarkRow(
        boundary="batched_headless", workload_id=f"A-replay-s{data['seed']} (M0)", num_envs=1,
        particles_per_env=None, control_hz=50., material_dt_s=1e-4, substeps_per_control_step=200, cameras="none",
        robot_dof=0, contacts=CONTACTS, recording="none", warm_cold="warm",
        env_steps_per_s=t["warm_env_steps_per_s"], aggregate_sim_time_per_wall=t["warm_env_steps_per_s"] * .02,
        vram_peak_mib=g["vram_peak_mib"], reset_mean_s=data["api_check"]["reset_s"], reset_p95_s=None,
        physics_verdict=verdict, reduced_fidelity=True, gpu_name=g["gpu_name"], gpu_index=g["gpu_index"],
        gpu_driver=g["gpu_driver"], gpu_sharing=g["gpu_sharing"], co_tenant=g["co_tenant"],
        source_commit=data["_manifest"]["git_head"], source=data["_run"],
        notes=f"{REDUCED}; M0 replay with 121 host samples taken between steps (outside timing)")


def m1_row(data):
    g = gpu(data)
    t = data["timing"]
    eps = [e for v in data["episodes"].values() for e in v]
    verdict = "PASS" if data["l2"]["status"] == "PASS" else "FAIL"
    return BenchmarkRow(
        boundary="batched_headless", workload_id="A-mixed (M1: 2 replay + 2 recorded_command_policy_v1)", num_envs=4,
        particles_per_env=None, control_hz=50., material_dt_s=1e-4, substeps_per_control_step=200, cameras="none",
        robot_dof=0, contacts=CONTACTS, recording="none", warm_cold="warm", env_steps_per_s=t["env_steps_per_s"],
        aggregate_sim_time_per_wall=t["env_steps_per_s"] * .02, vram_peak_mib=g["vram_peak_mib"],
        reset_mean_s=t["partial_reset_mean_s"], reset_p95_s=t["partial_reset_p95_s"], physics_verdict=verdict,
        reduced_fidelity=True, gpu_name=g["gpu_name"], gpu_index=g["gpu_index"], gpu_driver=g["gpu_driver"],
        gpu_sharing=g["gpu_sharing"], co_tenant=g["co_tenant"], source_commit=data["_manifest"]["git_head"],
        source=data["_run"], notes=f"{REDUCED}; L2 every step; {len(eps)} episodes evaluated; verdict column = L2 "
                                   "(evaluator verdicts per env in the evidence)")


def single_row(data):
    g = gpu(data)
    t = data["timing"]
    return BenchmarkRow(
        boundary="single_env_headless", workload_id="A/D-s512 (frozen LiveCuttingMaterial, rerun in P1-2 window)",
        num_envs=1, particles_per_env=data.get("particles"), control_hz=50., material_dt_s=1e-4,
        substeps_per_control_step=200, cameras="none", robot_dof=0, contacts=CONTACTS, recording="none",
        warm_cold="warm", env_steps_per_s=t["warm_env_steps_per_s"],
        aggregate_sim_time_per_wall=t["warm_sim_time_per_wall"], vram_peak_mib=g["vram_peak_mib"],
        reset_mean_s=t["first_reset_s"], reset_p95_s=None,
        physics_verdict=data["invariants"]["status"], reduced_fidelity=False, gpu_name=g["gpu_name"],
        gpu_index=g["gpu_index"], gpu_driver=g["gpu_driver"], gpu_sharing=g["gpu_sharing"], co_tenant=g["co_tenant"],
        source_commit=data["_manifest"]["git_head"], source=data["_run"],
        notes="P1-0 runner (run_baseline.py --mode replay), same venv and GPU window as the batched rows; reset "
              "column = first reset (solver build + graph capture); never merged with batched rows")


def nsys_summary(run):
    path = P12 / run / "kern_cuda_gpu_kern_sum.csv"
    if not path.exists():
        return dict(status="NOT_RUN", run=run)
    rows = list(csv.DictReader(path.open()))
    kernels = {}
    for r in rows:
        name = r.get("Name", "")
        kernels[name] = dict(instances=int(r["Instances"]), total_ns=float(r["Total Time (ns)"]),
                             avg_ns=float(r["Avg (ns)"]))
    return dict(status="MEASURED", run=run, kernel_count=len(kernels), kernels=kernels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0", nargs="*", default=[])
    parser.add_argument("--m1", nargs="*", default=[])
    parser.add_argument("--l4")
    parser.add_argument("--m2", nargs="*", default=[])
    parser.add_argument("--single")
    parser.add_argument("--alt", nargs="*", default=[], help="m2 throughput-alternation runs (one batch per process)")
    parser.add_argument("--prefix-runs", nargs="*", default=[], help="pre-review-fix runs kept with their commit")
    parser.add_argument("--nsys", nargs="*", default=[])
    parser.add_argument("--extra", type=Path, help="JSON with hand-written statuses/notes merged into the evidence")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, evidence = [], dict(schema="rcs_rl_p1_2_evidence_v1", runs={})
    for run in args.m0:
        d = load(run)
        evidence["runs"][run] = {k: d.get(k) for k in ("status", "error", "seed", "api_check", "payload_check", "l2",
                                                       "final", "evaluation_2_frames", "evaluation_121_frames",
                                                       "evaluation_2_vs_121_equal", "parity_vs_native", "timing",
                                                       "receipt_fidelity", "mempool_high_bytes", "peak_rss_mib",
                                                       "wall_s", "telemetry")}
        if d["status"] == "COMPLETE":
            rows.append(m0_row(d))
    for m1 in args.m1:
        d = load(m1)
        evidence["runs"][m1] = {k: d.get(k) for k in ("status", "error", "seeds", "drive", "payload_check", "l2",
                                                           "episodes", "partial_resets", "recorded_command_policy",
                                                           "timing", "receipt_fidelity", "mempool_high_bytes",
                                                           "peak_rss_mib", "wall_s", "telemetry")}
        if d["status"] == "COMPLETE":
            rows.append(m1_row(d))
    if args.l4:
        d = load(args.l4)
        evidence["runs"][args.l4] = {k: d.get(k) for k in ("status", "error", "l4_status", "l4_error",
                                                           "deterministic_mode", "steps", "copy_vs_copy",
                                                           "run_vs_run", "labelled", "walls_s",
                                                           "wrench_run_vs_run_equal", "traceback", "wall_s",
                                                           "telemetry", "scope")}
    m2 = {}
    for run in args.m2:
        d = load(run)
        m2[run] = d
        evidence["runs"][run] = {k: d.get(k) for k in ("status", "error", "num_envs", "seeds", "episodes_per_env",
                                                       "stagger", "payload_check", "l2", "episodes", "resets",
                                                       "memory", "timing", "receipt_fidelity", "particles_per_env",
                                                       "memory_bytes_per_env", "peak_rss_mib", "wall_s", "telemetry",
                                                       "stopped_at_max_global_steps")}
        if d["status"] == "COMPLETE":
            seeds = sorted(set(d["seeds"]))
            workload = "A-replay-" + "/".join(f"s{s}" for s in seeds)
            rows.append(m2_row(d, workload))
            rows.append(cold_row(d, workload))
    if args.single:
        d = load(args.single, root=ROOT / "artifacts" / "rl_ready")
        evidence["runs"][args.single] = {k: d.get(k) for k in ("status", "error", "timing", "invariants", "parity",
                                                               "wall_s", "telemetry")}
        if d["status"] == "COMPLETE":
            rows.append(single_row(d))
    alt = {}
    for run in args.alt:
        d = load(run)
        alt[run] = d
        evidence["runs"][run] = {k: d.get(k) for k in ("status", "error", "num_envs", "seeds", "l2", "timing",
                                                       "memory", "peak_rss_mib", "wall_s", "telemetry",
                                                       "stopped_at_max_global_steps")}
        if d["status"] == "COMPLETE":
            rows.append(m2_row(d, "A-replay-" + "/".join(f"s{s}" for s in sorted(set(d["seeds"]))), True))
    if alt:
        rounds = {}
        for run, d in alt.items():
            rounds.setdefault(run.split("_n")[0], {})[d["num_envs"]] = dict(
                run=run, env_steps_per_s=d["timing"]["env_steps_per_s"], step_median_s=d["timing"]["step_median_s"],
                foreign_sm_mean=(d["telemetry"].get("foreign_sm_pct") or {}).get("mean"))
        for r in rounds.values():
            base = r.get(1, {}).get("env_steps_per_s")
            for v in r.values():
                v["ratio_vs_n1_same_round"] = None if not base else v["env_steps_per_s"] / base
        evidence["throughput_alternation"] = dict(rounds=rounds, scope=(
            "separate processes (one batch, one device HashGrid each), alternated N=1/8/16 within each round, "
            "same venv/GPU; replay drive; partial windows (300/150/100 steps); SHARED_GPU"))
    evidence["prefix_runs"] = list(args.prefix_runs)
    for run, entry in evidence["runs"].items():
        root = P12 if (P12 / run).exists() else ROOT / "artifacts" / "rl_ready"
        m = json.loads((root / run / "manifest.json").read_text())
        entry["source"] = dict(git_head=m.get("git_head"), git_dirty=m.get("git_dirty"),
                               git_diff_stat=m.get("git_diff_stat"), started_unix_s=m.get("started_unix_s"),
                               pre_review_fix=run in args.prefix_runs)
    n1 = [d for d in m2.values() if d["num_envs"] == 1 and d["status"] == "COMPLETE"]
    if n1:
        base = sum(d["timing"]["env_steps_per_s"] for d in n1) / len(n1)
        evidence["throughput_vs_batched_n1"] = dict(
            n1_env_steps_per_s_mean=base, n1_runs=[d["_run"] for d in n1],
            ratios={d["_run"]: dict(num_envs=d["num_envs"], env_steps_per_s=d["timing"]["env_steps_per_s"],
                                    ratio=d["timing"]["env_steps_per_s"] / base)
                    for d in m2.values() if d["status"] == "COMPLETE"},
            scope="same venv, same GPU, separate processes, SHARED_GPU (co-tenant load varies between runs)")
    for row in rows:
        run = row.source.split("p1_2/")[-1]
        if run in args.prefix_runs:
            row.notes = "PRE-REVIEW-FIX source (dirty tree, commit " + (row.source_commit or "?")[:7] + "); " + row.notes
    evidence["nsys"] = [nsys_summary(r) for r in args.nsys]
    evidence["rows"] = [r.to_dict() for r in rows]
    evidence["markdown_table"] = markdown_table(rows)
    if args.extra:
        evidence.update(json.loads(args.extra.read_text()))
    for run in evidence["runs"].values():
        t = run.get("telemetry")
        if t:
            run["telemetry"] = {k: t.get(k) for k in ("gpu_sharing", "sharing_reasons", "own_vram_peak_mib",
                                                      "foreign_sm_pct", "foreign_memory_mib", "total_util_pct",
                                                      "gpu_name", "gpu_index", "gpu_driver", "valid_samples")}
        mem = run.get("memory")
        if mem and "samples" in mem:
            mem["samples"] = mem["samples"][::max(1, len(mem["samples"]) // 40)]
    args.out.write_text(json.dumps(evidence, indent=1, sort_keys=True, default=str) + "\n")
    print(evidence["markdown_table"])


if __name__ == "__main__":
    main()
