#!/usr/bin/env python3
"""Assemble the compact P1-5 CUT evidence (``docs/evidence/rl_ready/p1_5_cut_001.json``) from finished runs.

Inputs: ``artifacts/rl_ready/p1_5/<run>`` directories of ``run_cut_single.py`` (C1 v5 Demo 1 replays, C2 v7_2
single references, optional T5.8 replays of the Native v7_2 recording) and, optionally, the T5.8 Native launch
directory (``launch_native_cut.py``). The Native WP-12 Demo 1 row is re-labelled evidence (``source:
wp12_evidence``), never rerun. Failed runs are listed, never dropped. Never overwrites its output.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.rl.benchmark import BenchmarkRow, co_tenant_stats, markdown_table  # noqa: E402

ART = ROOT / "artifacts"
P15 = ART / "rl_ready" / "p1_5"
DEMO1 = ART / "wp12" / "demos_001" / "DEMO1_A"
HEADLESS_REPLAY_CONTACTS = "material tool/handle/two measured pads/fixture (replayed poses; reaction computed, not fed back)"
HEADLESS_REFERENCE_CONTACTS = "material tool/handle/fixture (ideal tracking, handle box only; reaction not fed back)"
NATIVE_CONTACTS = "PhysX robot/tool/fingers + material tool/pads/fixture, 20 ms reaction feedback"
NATIVE_CAMERAS = "global D455 + two wrist D405 RGB-D + overview/closeup (record every 10 steps)"


def load(path):
    return json.loads(Path(path).read_text())


def git(*args):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def run_summary(name):
    run = P15 / name
    result, manifest = load(run / "result.json"), load(run / "manifest.json")
    t = result.get("timing", {})
    key = "replay" if result["mode"] == "replay" else "reference_run"
    ev = (result.get("evaluation") or {}).get(key)
    out = dict(run=name, status=result["status"], error=(result.get("error") or "")[-600:] or None,
               mode=result["mode"], git_head=manifest["git_head"], git_dirty=manifest["git_dirty"],
               wall_s=result["wall_s"], task_id=result.get("task_id"), steps=result.get("steps_run"),
               verdict=None if ev is None else ev["summary"]["verdict"], evaluation=None if ev is None else ev["summary"],
               evaluation_wall_s=None if ev is None else ev["wall_s"], invariants=result.get("invariants"),
               hashgrid_census=result.get("hashgrid_census"), gpu_sharing=result["telemetry"].get("gpu_sharing"),
               own_vram_peak_mib=result["telemetry"].get("own_vram_peak_mib"),
               co_tenant=co_tenant_stats(result["telemetry"]),
               timing={k: t.get(k) for k in ("warm_step_mean_s", "warm_step_median_s", "warm_step_p95_s",
                                             "warm_env_steps_per_s", "warm_sim_time_per_wall", "first_step_s",
                                             "cold_process_start_to_first_step_done_s", "first_reset_s",
                                             "warm_resets_s", "warm_reset_p95_s", "material_graph_wall_mean_s",
                                             "all_steps_wall_s", "peak_rss_mib", "mempool_used_high_bytes_after_episode")},
               shaping_final=(result.get("shaping") or {}).get("samples", [{}])[-1],
               shaping_bonds=(result.get("shaping") or {}).get("bonds"))
    if result["mode"] == "replay" and result.get("parity"):
        p = result["parity"]
        native = (result.get("evaluation") or {}).get("native_same_evaluator")
        out.update(parity=dict(samples_compared=p["samples_compared"],
                               max_position_diff_m_over_time=p["max_position_diff_m_over_time"],
                               max_mean_position_diff_m_over_time=p["max_mean_position_diff_m_over_time"],
                               samples_with_code_difference=p["samples_with_code_difference"],
                               final=p["final"], reference=p["reference"], native_checkpoint=p["native_checkpoint"]),
                   native_same_evaluator=None if native is None else native["summary"],
                   reactions=result.get("reactions"), trajectory_checks=result["trajectory"]["checks"],
                   trajectory_source=result["trajectory"]["source"],
                   source_equivalence_audit=result.get("source_equivalence_audit"),
                   recording=result["trajectory"]["source"].get("episode"))
    if result["mode"] == "reference":
        out.update(final=result.get("final"), policy=result.get("policy"), workload=result.get("workload"),
                   reaction_steps=result.get("reaction_steps"), drive=result.get("drive"))
    out["_result"] = result
    out["_manifest"] = manifest
    return out


def band(names):
    """Pairwise final code / position differences between runs of the same configuration."""
    finals = {n: np.load(P15 / n / "final_state.npz") for n in names}
    rows = []
    for a, b in itertools.combinations(names, 2):
        ca, cb = finals[a]["codes"], finals[b]["codes"]
        pa, pb = finals[a]["positions"].astype(float), finals[b]["positions"].astype(float)
        d = np.linalg.norm(pa - pb, axis=1)
        rows.append(dict(a=a, b=b, different_code_particles=int(np.count_nonzero(ca != cb)),
                         labelled_diff=int(np.count_nonzero(cb)) - int(np.count_nonzero(ca)),
                         max_position_diff_m=float(d.max()), mean_position_diff_m=float(d.mean())))
    labelled = {n: int(np.count_nonzero(finals[n]["codes"])) for n in names}
    shas = {n: hashlib.sha256(np.ascontiguousarray(finals[n]["codes"])).hexdigest() for n in names}
    return dict(runs=list(names), labelled_particles=labelled, final_codes_sha256=shas, pairs=rows,
                max_pairwise_code_diff=max((r["different_code_particles"] for r in rows), default=0),
                max_pairwise_position_diff_m=max((r["max_position_diff_m"] for r in rows), default=0.),
                scope="final state (float32 positions saved), GPU float-atomic band; reported, not scored")


def headless_rows(summary, workload_id, contacts, reduced, commit_note):
    r, t = summary["_result"], summary["timing"]
    tele = r["telemetry"]
    verdict = summary["verdict"] if summary["verdict"] in ("PASS", "FAIL") else ("FAIL" if summary["verdict"] else "NOT_RUN")
    resets = t.get("warm_resets_s") or []
    steps = summary["steps"]
    common = dict(boundary="single_env_headless", workload_id=workload_id, num_envs=1,
                  particles_per_env=r.get("particles"), control_hz=50., material_dt_s=1e-4,
                  substeps_per_control_step=200, cameras="none", robot_dof=0, contacts=contacts,
                  recording="none in timed calls (host snapshots every 10 steps outside timing)",
                  vram_peak_mib=tele.get("own_vram_peak_mib"),
                  reset_mean_s=float(np.mean(resets)) if resets else None,
                  reset_p95_s=t.get("warm_reset_p95_s"), physics_verdict=verdict, reduced_fidelity=reduced,
                  gpu_name=tele.get("gpu_name"), gpu_index=tele.get("gpu_index"), gpu_driver=tele.get("gpu_driver"),
                  gpu_sharing=tele.get("gpu_sharing", "UNKNOWN"), co_tenant=co_tenant_stats(tele),
                  source_commit=summary["git_head"], source=f"measured ({summary['run']})")
    cold_s = t.get("cold_process_start_to_first_step_done_s") or 0.
    cold_total = cold_s + (t.get("all_steps_wall_s") or 0.) - (t.get("first_step_s") or 0.)
    return [BenchmarkRow(**common, warm_cold="warm", env_steps_per_s=t["warm_env_steps_per_s"],
                         aggregate_sim_time_per_wall=t["warm_sim_time_per_wall"],
                         notes=f"Steps 2..{steps}: material calls per control step (graph launch + sync + read-only "
                               f"diagnostics); snapshots/evaluation outside timing. Warm resets: {len(resets)} "
                               f"(solver rebuild + graph capture, previous solver released first). {commit_note}"),
            BenchmarkRow(**common, warm_cold="cold", env_steps_per_s=steps / cold_total,
                         aggregate_sim_time_per_wall=steps * .02 / cold_total,
                         notes=f"Process start -> last step without snapshot time: {cold_total:.1f} s (process start "
                               f"-> first step done {cold_s:.2f} s, user Warp kernel cache).")]


def native_actions_timing(episode):
    """Per-step Native walls from a WP-12-format actions file (task steps carry ``step_wall_s``)."""
    import gzip
    path = episode / "actions.jsonl"
    stream = path.open() if path.exists() else gzip.open(episode / "actions.jsonl.gz", "rt")
    step_wall, material_wall = [], []
    with stream:
        for line in stream:
            row = json.loads(line)
            step_wall.append(row.get("step_wall_s"))
            material_wall.append(row["privileged_diagnostics"].get("material_wall_s"))
    sw = np.asarray([x for x in step_wall if x is not None], float)
    mw = np.asarray([x for x in material_wall if x is not None], float)
    return dict(control_steps=len(step_wall), steps_with_step_wall=int(len(sw)), step_wall_mean_s=float(sw.mean()),
                step_wall_p95_s=float(np.percentile(sw, 95)), step_wall_sum_s=float(sw.sum()),
                material_wall_steps=int(len(mw)), material_wall_mean_s=float(mw.mean()),
                scope="actions rows: step_wall_s on task steps (PhysX 10 substeps + IK + material graph + cameras + "
                      "recording); return-home rows carry no step_wall_s; material_wall_s = material graph per step")


def native_rows(workload_id, episode, verdict, process_wall_s, sharing, co, commit, source, solver):
    t = native_actions_timing(episode)
    common = dict(boundary="single_env_native", workload_id=workload_id, num_envs=1, particles_per_env=96000,
                  control_hz=50., material_dt_s=1e-4, substeps_per_control_step=200, cameras=NATIVE_CAMERAS,
                  robot_dof=18, contacts=NATIVE_CONTACTS,
                  recording="full (actions, robot state, 75 physics samples, camera frames every 10 steps)",
                  vram_peak_mib=None, reset_mean_s=None, reset_p95_s=None, physics_verdict=verdict,
                  reduced_fidelity=False, gpu_name="NVIDIA GeForce RTX 5090", gpu_index=0, gpu_driver=None,
                  gpu_sharing=sharing, co_tenant=co, source_commit=commit, source=source)
    n, warm = t["steps_with_step_wall"], t["step_wall_sum_s"]
    rows = [BenchmarkRow(**common, warm_cold="warm", env_steps_per_s=n / warm, aggregate_sim_time_per_wall=n * .02 / warm,
                         notes=f"{n} task action calls, mean {t['step_wall_mean_s'] * 1e3:.0f} ms (PhysX + IK + material "
                               f"graph mean {t['material_wall_mean_s'] * 1e3:.0f} ms + cameras + recording); {solver}. "
                               "Job VRAM and per-process utilisation not sampled (UNVERIFIED)."),
            BenchmarkRow(**common, warm_cold="cold", env_steps_per_s=t["control_steps"] / process_wall_s,
                         aggregate_sim_time_per_wall=t["control_steps"] * .02 / process_wall_s,
                         notes=f"{t['control_steps']} control steps (700 task + 40 return home) over the process wall "
                               f"{process_wall_s:.0f} s incl. Kit start, grasp settling, evaluation and media.")]
    return rows, t


def native_demo1_rows():
    attempt = load(DEMO1 / "attempt.json")
    analysis = load(DEMO1 / "analysis_001" / "analysis.json")
    co = dict(foreign_processes_max=len({a.split(",")[1] for a in attempt["gpu_before"]["compute_apps"]
                                         if a.startswith("GPU-76f61dc7")}),
              scope="WP-12 launcher: nvidia-smi compute apps on GPU0 before/after only; utilisation not sampled")
    return native_rows("C1-native-demo1", DEMO1 / "run" / "episode_000", analysis["task_outcome"]["verdict"],
                       analysis["performance"]["process_wall_s"], "SHARED_GPU", co, attempt["git_sha"] + " (dirty)",
                       "wp12_evidence", "tlmpm_v5 law 8")


def native_t5_8(name):
    if name is None:
        return dict(status="NOT_RUN")
    base = P15 / name
    launch = load(base / "launch.json") if (base / "launch.json").exists() else load(base / "launch.json.started")
    out = dict(dir=name, status=launch.get("status"), launch={k: launch.get(k) for k in (
        "command", "gpu", "need_free_mib", "gpu_before", "gpu_after", "returncode", "wall_s", "git_head", "git_dirty",
        "reason", "started", "ended")})
    result_path = base / "run" / "episode_000" / "result.json"
    out["rows"] = []
    if result_path.exists() and launch.get("returncode") == 0:
        r = load(result_path)
        foreign = [a for a in launch["gpu_before"]["compute_apps"] + launch["gpu_after"]["compute_apps"]]
        co = dict(foreign_processes_max=len({a.split(",")[1] for a in foreign}),
                  scope="launcher nvidia-smi before/after only (compute apps); utilisation not sampled")
        rows, timing = native_rows("C2-native-v7_2", base / "run" / "episode_000",
                                   (r.get("task_outcome") or {}).get("verdict", "NOT_RUN"), launch["wall_s"],
                                   "SHARED_GPU" if foreign else "UNKNOWN", co, launch["git_head"], f"measured ({name})",
                                   "tlmpm_v7_2 law 8 (v7_2_optimizations)")
        out["rows"], out["timing"] = [row.to_dict() for row in rows], timing
        out["_rows"] = rows
    if result_path.exists():
        r = load(result_path)
        out.update(task_id=r.get("task_id"), task_status=r.get("task_status"), run_status=r.get("status"),
                   task_outcome_verdict=(r.get("task_outcome") or {}).get("verdict"),
                   outcome_metrics=((r.get("task_outcome") or {}).get("outcome") or {}).get("metrics"),
                   integrity=((r.get("task_outcome") or {}).get("integrity") or {}).get("status"),
                   steps=r.get("steps"), wall_s=r.get("wall_s"), home_outcome=(r.get("home_outcome") or {}).get("status"),
                   task_end_reason=r.get("task_end_reason"), episode_state_final=r.get("episode_state_final"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v5-replay", nargs="+", required=True)
    p.add_argument("--v7-2-reference", nargs="+", required=True)
    p.add_argument("--native-t5-8", help="native_cut_v7_2_NNN directory (launch record)")
    p.add_argument("--v7-2-native-replay", nargs="*", default=[])
    p.add_argument("--excluded", nargs="*", default=[], help="run directories listed but not used as rows (smoke...)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    commit = git("rev-parse", "HEAD")
    v5 = [run_summary(n) for n in a.v5_replay]
    v72 = [run_summary(n) for n in a.v7_2_reference]
    v72n = [run_summary(n) for n in a.v7_2_native_replay]
    note = "Frozen single LiveCuttingMaterial, .deps/isaacsim-venv, no Kit."
    rows, demo1_timing = native_demo1_rows()
    t58 = native_t5_8(a.native_t5_8)
    rows += t58.pop("_rows", [])
    for s in v5:
        if s["status"] == "COMPLETE":
            rows += headless_rows(s, "C1-v5-demo1-replay", HEADLESS_REPLAY_CONTACTS, False, note)
    for s in v72:
        if s["status"] == "COMPLETE":
            rows += headless_rows(s, "C2-v7_2-single-reference", HEADLESS_REFERENCE_CONTACTS, True,
                                  note + " reduced_fidelity: ideal tracking, no robot, reaction not fed back, no pads.")
    for s in v72n:
        if s["status"] == "COMPLETE":
            rows += headless_rows(s, "C2-v7_2-native-replay", HEADLESS_REPLAY_CONTACTS, False, note)
    complete = lambda runs: [s["run"] for s in runs if s["status"] == "COMPLETE"]  # noqa: E731
    strip = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}  # noqa: E731
    demo1 = load(DEMO1 / "analysis_001" / "analysis.json")
    evidence = dict(
        schema="rcs_rl_p1_5_cut_evidence_v1", id="p1_5_cut_001", evidence_commit=commit,
        plan="docs/reports/rl_ready/P1_5_PLAN.md (T5.1, T5.3-T5.6, T5.8)",
        declaration=dict(path="configs/rl_benchmark/cut_v7_2_v1.json",
                         sha256=hashlib.sha256((ROOT / "configs/rl_benchmark/cut_v7_2_v1.json").read_bytes()).hexdigest(),
                         declared_in_commit="cb76a33 (before any CUT run)"),
        c1_v5_demo1_replay=dict(
            native_reference=dict(run="wp12/demos_001/DEMO1_A (main checkout, read-only)", git=demo1["analysis_tool"]["git_sha"],
                                  verdict=demo1["task_outcome"]["verdict"], class_=demo1["class"],
                                  outcome_metrics=demo1["task_outcome"].get("outcome_metrics"),
                                  performance=demo1["performance"], actions_timing=demo1_timing),
            runs=[strip(s) for s in v5], band=band(complete(v5)) if len(complete(v5)) > 1 else None),
        c2_v7_2_single_reference=dict(runs=[strip(s) for s in v72],
                                      band=band(complete(v72)) if len(complete(v72)) > 1 else None),
        t5_8_native_v7_2=t58,
        t5_8_v7_2_native_replay=dict(runs=[strip(s) for s in v72n],
                                     band=band(complete(v72n)) if len(complete(v72n)) > 1 else None),
        rows=[r.to_dict() for r in rows], markdown_table=markdown_table(rows), excluded_runs=a.excluded)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (never overwritten)")
    a.out.write_text(json.dumps(evidence, indent=1, sort_keys=True, allow_nan=False, default=str) + "\n")
    print(evidence["markdown_table"])


if __name__ == "__main__":
    main()
