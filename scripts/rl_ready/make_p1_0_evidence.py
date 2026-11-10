#!/usr/bin/env python3
"""Assemble the compact P1-0 baseline evidence JSON (and print its Markdown table) from finished runs.

Inputs: ``artifacts/rl_ready/p1_0/<run>`` directories written by ``run_baseline.py`` and the WP-17 Native
evidence (``docs/evidence/wp17/performance_001.json`` plus its retained GPU verdicts). Native rows are
re-labelled evidence (``source: wp17_evidence``); nothing Native is rerun. Never overwrites its output.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.rl.benchmark import BenchmarkRow, co_tenant_stats, markdown_table  # noqa: E402

ART = ROOT / "artifacts"
WP17_EVIDENCE = ROOT / "docs" / "evidence" / "wp17" / "performance_001.json"
NATIVE_CAMERAS = "global_d455 (RGB) + wrist_d405_0 (RGB-D), surface on capture"
NATIVE_CONTACTS = "PhysX robot/tool/fingers + material tool/pads/fixture, 20 ms reaction feedback"
HEADLESS_CONTACTS = "material tool/handle/pads/fixture (replayed poses, reaction computed but not fed back)"


def load(path):
    return json.loads(Path(path).read_text())


def native_gpu(seed):
    """Sharing evidence of the WP-17 Native run on GPU1 (process list only; utilisation not sampled per PID)."""
    base = ART / "wp17" / "native_optimized_001" / f"seed_{seed}" / "gpu"
    verdict = load(base / "gpu_verdict.json")
    foreign, display = set(), set()
    with (base / "gpu_samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            for proc in row["procs"]:
                if proc["gpu"] == 1 and not proc.get("job"):
                    (display if proc["pid"] == 4060 else foreign).add(proc["pid"])
    return dict(job_peak_mib=verdict["gpu1_job_peak_mib"], others_peak_mib=verdict["gpu1_others_peak_mib"],
                samples=verdict["samples"], foreign_pids=sorted(foreign), display_pids=sorted(display),
                sharing="SHARED_GPU" if foreign else "EXCLUSIVE_GPU")


def native_rows(evidence, commit_note):
    rows = []
    for seed in ("512", "15"):
        episode = evidence["episodes"]["optimized"][seed]
        timing = episode["timing"]
        gpu = native_gpu(seed)
        particles = episode["recording"]["mass"]["particles"]
        co = dict(samples=gpu["samples"], foreign_processes_max=len(gpu["foreign_pids"]),
                  foreign_memory_mib_max=gpu["others_peak_mib"], foreign_sm_pct_max=None,
                  scope="WP-17 runner GPU verdict: process list and memory at 1 Hz; per-process utilisation not "
                        "recorded; GPU1 others peak includes the display server")
        common = dict(boundary="single_env_native", workload_id=f"D-s{seed}", num_envs=1, particles_per_env=particles,
                      control_hz=50., material_dt_s=1e-4, substeps_per_control_step=200, cameras=NATIVE_CAMERAS,
                      robot_dof=18, contacts=NATIVE_CONTACTS,
                      recording="full (actions, robot state, 121 physics samples, camera frames, media)",
                      vram_peak_mib=float(gpu["job_peak_mib"]), reset_mean_s=None, reset_p95_s=None,
                      physics_verdict="PASS", reduced_fidelity=False, gpu_name="NVIDIA GeForce RTX 5090", gpu_index=1,
                      gpu_driver=None, gpu_sharing=gpu["sharing"], co_tenant=co,
                      source_commit=episode["source_commit"], source="wp17_evidence")
        rows.append(BenchmarkRow(**common, warm_cold="warm", env_steps_per_s=1200 / timing["step_call_total_s"],
                                 aggregate_sim_time_per_wall=24. / timing["step_call_total_s"],
                                 notes=f"Action loop only: sum of 1200 step calls {timing['step_call_total_s']:.2f} s "
                                       "(PhysX 10 substeps + IK + material graph + recording per action). Physics "
                                       "verdict = WP-17 integrity/outcome PASS. Driver not recorded in WP-17 "
                                       f"evidence (UNVERIFIED; inventory now shows 570.211.01). {commit_note}"))
        rows.append(BenchmarkRow(**common, warm_cold="cold", env_steps_per_s=1200 / timing["factory_to_finalize_s"],
                                 aggregate_sim_time_per_wall=24. / timing["factory_to_finalize_s"],
                                 notes=f"Factory entry -> finalize {timing['factory_to_finalize_s']:.2f} s incl. Kit "
                                       f"start/scene/reset {timing['startup_to_first_step_s']:.2f} s, evaluation "
                                       f"{timing['internal_evaluation_s']:.2f} s and media {timing['media_s']:.2f} s."))
    return rows


def headless_rows(run_dir, commit):
    result = load(run_dir / "result.json")
    manifest = load(run_dir / "manifest.json")
    if result["status"] != "COMPLETE" or result["mode"] != "replay":
        raise ValueError(f"{run_dir.name} is not a complete replay run")
    t, inv, tel = result["timing"], result["invariants"], result["telemetry"]
    seed = result["seed"]
    steps = result["steps_run"]
    co = co_tenant_stats(tel)
    import numpy as np
    with np.load(run_dir / "steps.npz") as data:
        walls = data["step_wall_s"].astype(float)
    cold_total = t["cold_process_start_to_first_step_done_s"] + float(walls[1:].sum())
    common = dict(boundary="single_env_headless", workload_id=f"A/D-s{seed}", num_envs=1,
                  particles_per_env=result["particles"], control_hz=50., material_dt_s=1e-4,
                  substeps_per_control_step=200, cameras="none", robot_dof=0, contacts=HEADLESS_CONTACTS,
                  recording="none (121 parity samples read outside the timed step calls)",
                  vram_peak_mib=tel.get("own_vram_peak_mib"), physics_verdict=inv["status"], reduced_fidelity=False,
                  gpu_name=tel.get("gpu_name"), gpu_index=tel.get("gpu_index"), gpu_driver=tel.get("gpu_driver"),
                  gpu_sharing=tel.get("gpu_sharing", "UNKNOWN"), co_tenant=co, source_commit=commit or
                  manifest["git_head"], source="measured")
    warm = BenchmarkRow(**common, warm_cold="warm", env_steps_per_s=t["warm_env_steps_per_s"],
                        aggregate_sim_time_per_wall=t["warm_sim_time_per_wall"], reset_mean_s=t["warm_reset_mean_s"],
                        reset_p95_s=t["warm_reset_p95_s"],
                        notes=f"Steps 2..{steps}: update_pad_obstacles + advance (graph launch of 200 substeps + device "
                              f"sync + reaction readback) + device_reduction_v1 diagnostics; mean {t['warm_step_mean_s']*1e3:.1f} "
                              f"ms, p95 {t['warm_step_p95_s']*1e3:.1f} ms. Reset = {len(t['warm_resets_s'])} warm resets "
                              "in the same process (task particles, solver construction, graph capture). Physics verdict "
                              "= material invariants (finite, detF>0, constant mass, no digit rewrites); parity "
                              f"reported separately. Run {run_dir.name}.")
    cold = BenchmarkRow(**common, warm_cold="cold", env_steps_per_s=steps / cold_total,
                        aggregate_sim_time_per_wall=steps * 0.02 / cold_total, reset_mean_s=t["first_reset_s"],
                        reset_p95_s=t["first_reset_s"],
                        notes=f"Process start -> first step done {t['cold_process_start_to_first_step_done_s']:.2f} s "
                              f"(first reset incl. Warp module load and graph capture {t['first_reset_s']:.2f} s, user "
                              f"kernel cache: {t['warp_modules_first_reset']['compiled']} modules compiled, "
                              f"{t['warp_modules_first_reset']['cached']} cached) + steps 2..{steps}. Run {run_dir.name}.")
    return [warm, cold], result


def batched_rows():
    return [BenchmarkRow(boundary="batched_headless", workload_id="A", num_envs=n, particles_per_env=None,
                         control_hz=50., material_dt_s=1e-4, substeps_per_control_step=200, cameras="none",
                         robot_dof=None, contacts="n/a", recording="none", warm_cold="n/a", env_steps_per_s=None,
                         aggregate_sim_time_per_wall=None, vram_peak_mib=None, reset_mean_s=None, reset_p95_s=None,
                         physics_verdict="NOT_RUN", reduced_fidelity=False, gpu_name=None, gpu_index=None,
                         gpu_driver=None, gpu_sharing="UNKNOWN", co_tenant={}, source_commit=None, source="not_run",
                         notes="NOT_RUN in P1-0: the batched runtime does not exist yet (P1-1).")
            for n in (1, 4, 16)]


def kernel_split(run_dir):
    result = load(run_dir / "result.json")
    if result["status"] != "COMPLETE" or result["mode"] != "kernels":
        raise ValueError(f"{run_dir.name} is not a complete kernels run")
    windows = {}
    for name, window in result["windows"].items():
        windows[name] = dict(
            steps=window["steps"], substeps_timed=window["substeps_timed"],
            kernel_gpu_time_per_substep_us=round(window["kernel_gpu_time_per_substep_us"], 2),
            kernel_gpu_time_per_control_step_ms=round(window["kernel_gpu_time_per_control_step_ms"], 3),
            eager_step_wall_mean_s=round(window["eager_step_wall_mean_s"], 4),
            graph_step_wall_after_window_mean_s=round(window["graph_step_wall_after_window_mean_s"], 4),
            reaction_impulse_norm=[round(e["reaction_impulse_norm"], 6) for e in window["eager_steps"]],
            cut_index=[e["cut_index"] for e in window["eager_steps"]],
            groups_per_substep_us={k: round(v["per_substep_us"], 2) for k, v in window["kernel_groups"].items()},
            group_share={k: round(v["share"], 4) for k, v in window["kernel_groups"].items()},
            top_kernels_per_substep_us={k: round(v["per_substep_us"], 2) for k, v in sorted(
                window["kernels"].items(), key=lambda kv: -kv[1]["per_substep_us"])[:12]},
            kernel_launches_per_substep={k: v["calls"] / window["substeps_timed"]
                                         for k, v in window["kernels"].items()})
    return dict(run=run_dir.name, scope=result["scope"], phase_info=result["phase_info"],
                telemetry=co_tenant_stats(result["telemetry"]), gpu_sharing=result["telemetry"].get("gpu_sharing"),
                windows=windows)


def parity(result):
    p = result["parity"]
    comparisons = result["comparisons"]
    over_time = [dict(sample=c["sample"], time_s=c["time_s"], max_position_diff_m=c["max_position_diff_m"],
                      mean_position_diff_m=c["mean_position_diff_m"], different_code_particles=c["different_code_particles"],
                      labelled_count_diff=c["labelled_count_diff"]) for c in comparisons
                 if c["sample"] % 10 == 0 or c["sample"] in (1, 2, 5) or c["different_code_particles"] and
                 c["sample"] == (p["first_sample_with_code_difference"] or {}).get("sample")]
    return dict(samples_compared=p["samples_compared"], reference=p["reference"],
                max_position_diff_m_over_time=p["max_position_diff_m_over_time"],
                max_mean_position_diff_m_over_time=p["max_mean_position_diff_m_over_time"],
                first_sample_with_position_difference=p["first_sample_with_position_difference"],
                first_sample_with_code_difference=p["first_sample_with_code_difference"],
                final=p["final"], selected_samples=over_time, invariants=result["invariants"],
                trajectory_checks=result["trajectory"]["checks"])


def band_entry(run_dir):
    import numpy as np
    result = load(run_dir / "result.json")
    if result["status"] != "COMPLETE" or result["mode"] != "replay":
        return dict(run=run_dir.name, status=result["status"], error=(result.get("error") or "")[-300:])
    p, final = result["parity"], result["parity"]["final"]
    evaluation = result.get("evaluation") or {}
    with np.load(run_dir / "final_state.npz") as data:
        codes = data["codes"]
    return dict(run=run_dir.name, status="COMPLETE", replay_obstacles=result.get("replay_obstacles"),
                final_codes_equal_native=final["codes_equal"], final_different_code_particles=final["different_code_particles"],
                final_labelled_replay={k: v["labelled"] for k, v in final["labelled_replay"].items()},
                final_labelled_count_diff=final["labelled_count_diff"],
                final_max_position_diff_m=final["max_position_diff_m"],
                final_mean_position_diff_m=final["mean_position_diff_m"],
                max_position_diff_m_over_time=p["max_position_diff_m_over_time"],
                max_position_diff_m_by_sample=[c["max_position_diff_m"] for c in result["comparisons"]],
                different_code_particles_by_sample=[c["different_code_particles"] for c in result["comparisons"]],
                first_sample_with_code_difference=p["first_sample_with_code_difference"],
                replay_final_codes_sha256=final["replay_final_codes_sha256"],
                invariants=result["invariants"]["status"], reactions=result.get("reactions"),
                evaluator_replay=evaluation.get("replay"), evaluator_native=evaluation.get("native_same_evaluator"),
                warm_env_steps_per_s=result["timing"]["warm_env_steps_per_s"], gpu_sharing=result["telemetry"]["gpu_sharing"],
                _codes=codes)


def l3_band(base, names_by_seed):
    import numpy as np
    out = {}
    for seed, names in names_by_seed.items():
        entries = [band_entry(base / name) for name in names]
        done = [e for e in entries if e["status"] == "COMPLETE"]
        codes = [e.pop("_codes") for e in done]
        pairwise = [int(np.count_nonzero(codes[i] != codes[j])) for i in range(len(codes))
                    for j in range(i + 1, len(codes))]

        def rng(key, sub=None):
            values = [e[key] if sub is None else (e[key] or {}).get(sub) for e in done]
            values = [v for v in values if v is not None and not isinstance(v, dict)]
            return dict(min=min(values), max=max(values), n=len(values)) if values else None
        evaluated = [e for e in done if e["evaluator_replay"]]
        per_sample = np.asarray([e["max_position_diff_m_by_sample"] for e in done])
        out[f"s{seed}"] = dict(
            runs=len(entries), complete=len(done), runs_detail=[{k: v for k, v in e.items() if not k.endswith("_by_sample")}
                                                                for e in entries],
            final_different_code_particles=rng("final_different_code_particles"),
            final_codes_equal_native_count=sum(bool(e["final_codes_equal_native"]) for e in done),
            final_labelled_digit0=dict(min=min(e["final_labelled_replay"]["0"] for e in done),
                                       max=max(e["final_labelled_replay"]["0"] for e in done)),
            final_labelled_digit1=dict(min=min(e["final_labelled_replay"]["1"] for e in done),
                                       max=max(e["final_labelled_replay"]["1"] for e in done)),
            max_position_diff_m_over_time=rng("max_position_diff_m_over_time"),
            final_max_position_diff_m=rng("final_max_position_diff_m"),
            max_position_diff_m_band_by_sample=dict(sample_every_s=0.2, min=per_sample.min(0).tolist(),
                                                    max=per_sample.max(0).tolist()),
            samples_with_code_difference_any_run=int(np.count_nonzero(
                np.asarray([e["different_code_particles_by_sample"] for e in done]).max(0))),
            replay_vs_replay_final_code_differences=pairwise,
            evaluator=dict(runs=len(evaluated),
                           verdicts=[e["evaluator_replay"]["verdict"] for e in evaluated],
                           removed_mass_fraction=[e["evaluator_replay"]["removed_mass_fraction"] for e in evaluated],
                           largest_component_below_plane_mass_share=[
                               e["evaluator_replay"]["largest_component_below_plane_mass_share"] for e in evaluated],
                           detached_below_plane_mass_fraction=[
                               e["evaluator_replay"]["detached_below_plane_mass_fraction"] for e in evaluated],
                           native_same_evaluator=evaluated[0]["evaluator_native"] if evaluated else None,
                           note="The first replay run of each seed predates the evaluator option (codes/positions only)"))
    return out


def throughput_band(base, names):
    """Warm stepping of every complete full-episode replay run, with its own GPU sharing evidence."""
    out = []
    for name in names:
        result = load(base / name / "result.json")
        if result["status"] != "COMPLETE":
            continue
        t, tel = result["timing"], result["telemetry"]
        out.append(dict(run=name, seed=result["seed"], warm_env_steps_per_s=t["warm_env_steps_per_s"],
                        warm_step_mean_ms=t["warm_step_mean_s"] * 1e3, warm_step_p95_ms=t["warm_step_p95_s"] * 1e3,
                        material_graph_wall_mean_ms=t["material_graph_wall_mean_s"] * 1e3,
                        cold_process_start_to_first_step_done_s=t["cold_process_start_to_first_step_done_s"],
                        first_reset_s=t["first_reset_s"], warm_reset_mean_s=t["warm_reset_mean_s"],
                        own_vram_peak_mib=tel.get("own_vram_peak_mib"), gpu_sharing=tel.get("gpu_sharing"),
                        sharing_reasons=tel.get("sharing_reasons"),
                        foreign_sm_pct_mean=(tel.get("foreign_sm_pct") or {}).get("mean"),
                        foreign_sm_pct_max=(tel.get("foreign_sm_pct") or {}).get("max"),
                        evaluate_option=result.get("evaluation") is not None))
    return out


def sequential_baseline(rows):
    out = {}
    for row in rows:
        if row.boundary == "single_env_headless" and row.warm_cold == "warm":
            step = 1. / row.env_steps_per_s
            out[row.workload_id] = {str(n): dict(wall_per_vector_step_s=n * step, aggregate_env_steps_per_s=1. / step,
                                                 aggregate_sim_time_per_wall=0.02 / step) for n in (1, 4, 8, 16)}
    return dict(status="DERIVED_NOT_RUN", rule="N sequential single-env warm steps: wall = N x single warm step time; "
                "aggregate env-steps/s does not grow with N", by_workload=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", nargs="+", required=True, help="replay run directory names")
    parser.add_argument("--kernels", required=True)
    parser.add_argument("--coldcache")
    parser.add_argument("--extra-failed", nargs="*", default=[], help="failed/aborted run names to list")
    parser.add_argument("--band", nargs="*", default=[], help="seed=run[,run...] for the L3 band (incl. --replay runs)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    base = ART / "rl_ready" / "p1_0"
    evidence = load(WP17_EVIDENCE)
    rows, parity_out, runs = [], {}, {}
    rows += native_rows(evidence, "Native rows are not rerun in P1-0.")
    for name in args.replay:
        new, result = headless_rows(base / name, None)
        rows += new
        parity_out[f"s{result['seed']}:{name}"] = parity(result)
        t = result["timing"]
        runs[name] = dict(seed=result["seed"], wall_s=result["wall_s"], device=result["device"],
                          material_receipt={k: v for k, v in result["material_receipt"].items() if k != "solver_receipt"},
                          timing={k: v for k, v in t.items() if not k.startswith("warp_modules")},
                          warp_modules_first_reset={k: v for k, v in t["warp_modules_first_reset"].items()
                                                    if k != "lines"},
                          telemetry=result["telemetry"], manifest=load(base / name / "manifest.json"))
    rows += batched_rows()
    cold = None
    if args.coldcache:
        result = load(base / args.coldcache / "result.json")
        t = result["timing"]
        cold = dict(run=args.coldcache, status=result["status"], steps_run=result["steps_run"],
                    kernel_cache="fresh empty Warp kernel cache (true compile cold start)",
                    process_start_to_first_step_done_s=t["cold_process_start_to_first_step_done_s"],
                    first_reset_s=t["first_reset_s"], warm_resets_s=t["warm_resets_s"],
                    warp_modules_first_reset={k: v for k, v in t["warp_modules_first_reset"].items() if k != "lines"},
                    module_lines=t["warp_modules_first_reset"]["lines"], gpu_sharing=result["telemetry"]["gpu_sharing"],
                    scope="Truncated to the first steps on purpose; cold-start measurement only, never a throughput row")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    data = dict(schema="rcs_rl_p1_0_baseline_v1", evidence_commit_parent=head,
                boundaries_note="Timings of different boundaries are never merged into one speedup number.",
                rows=[row.to_dict() for row in rows], markdown_table=markdown_table(rows),
                replay_parity=parity_out, kernel_split=kernel_split(base / args.kernels), cold_start_fresh_cache=cold,
                runs=runs, failed_or_aborted_runs=args.extra_failed,
                l3_band=l3_band(base, {int(item.split("=")[0]): item.split("=")[1].split(",") for item in args.band}),
                sequential_n_single_env=sequential_baseline(rows),
                single_env_headless_throughput_band=throughput_band(base, [n for item in args.band
                                                                             for n in item.split("=")[1].split(",")]),
                import_origin=dict(status="PASS", detail=(
                    "Every GPU run used .deps/isaacsim-venv/bin/python with PYTHONPATH=src (chain scripts) and "
                    "run_baseline.py inserts <worktree>/src at sys.path[0] before importing robotic_cutting_sim. The "
                    "Isaac Sim venv has no robotic_cutting_sim installed, and the main checkout has no "
                    "robotic_cutting_sim.rl subpackage, so the rl.replay relative imports (live_material, tlmpm_v7_2, "
                    "goals) can only resolve to this worktree; verified by printing the module files. Manifest "
                    "source_sha256 records the worktree sources; runs after this note also record package_origin.")),
                replay_obstacles=dict(label="handle_and_measured_pads_v1", verification=(
                    "reset pad boxes rebuilt from robot_state.npz row 0 + recorded reset edge are bitwise equal to the "
                    "recorded configure_pad_obstacles input (finger_geometry_00.json, material.json); the knife body "
                    "poses of the same robot_state rows composed with tool_from_edge reproduce the recorded advance() "
                    "edge pose at all 1200 steps (max 2.8e-17 m), so the finger body poses are the same PhysX snapshot "
                    "Native read; Native and replay finger-pad reactions are zero at every step (the pads never touch "
                    "the material in these two episodes, so pad geometry is not exercised beyond that)")),
                native_source=dict(evidence="docs/evidence/wp17/performance_001.json",
                                   gpu_verdicts="wp17/native_optimized_001/seed_<s>/gpu/gpu_verdict.json"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(data, stream, indent=1, sort_keys=True, allow_nan=False)
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
