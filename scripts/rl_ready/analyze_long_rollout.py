#!/usr/bin/env python3
"""P1-4 analysis (T4.4 / T4.7): read one supervised run directory and compute episode tallies, canary verdicts,
infrastructure-failure tallies, memory bounds B0-B5 per attempt, co-tenant labels and timing.

    .venv/bin/python scripts/rl_ready/analyze_long_rollout.py --run-dir artifacts/rl_ready/p1_4/lr_a_001 \\
        [--evidence docs/evidence/rl_ready/p1_4_001.json]

Writes ``<run-dir>/analysis_NNN.json`` (never overwritten) and, with ``--evidence``, a compact evidence file
(exclusive create). Bounds come from the config named in the run's manifest; nothing is re-declared here.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from robotic_cutting_sim.rl import supervision as S  # noqa: E402
from robotic_cutting_sim.rl.stability import GATED, evaluate_bounds  # noqa: E402
from robotic_cutting_sim.rl.telemetry import summarize  # noqa: E402


def _episode_row(r):
    ev = r.get("evaluation") or {}
    can = r.get("canary") or {}
    l2 = r.get("l2") or {}
    return dict(env=r["env"], episode_index=r["episode_index"], segment=r["segment"], cause=r["cause"],
                steps=r["steps"], end_class=r["end_class"], drive=r.get("drive"), scene_seed=r.get("scene_seed"),
                seed=r.get("seed"), global_step_end=r.get("global_step_end"), beyond_quota=r.get("beyond_quota"),
                verdict=ev.get("verdict"), labelled=ev.get("labelled"), final_codes_sha256=ev.get("final_codes_sha256"),
                canary=can.get("status"), canary_codes_equal_native=can.get("codes_equal_native"),
                canary_different_code_particles=can.get("different_code_particles"),
                l2=l2.get("status"), termination_names=r.get("termination_names"),
                invalid_names=r.get("invalid_names"), evaluation_wall_s=ev.get("wall_s"))


def analyze(run_dir, config=None):
    run_dir = Path(run_dir)
    sup, sup_report = S.read_ledger(run_dir / "supervisor.jsonl")
    tele, tele_report = S.read_ledger(run_dir / "telemetry.jsonl")
    start = next((r for r in sup if r["type"] == "supervisor_start"), {})
    meta = start.get("meta", {})
    final = next((r for r in sup if r["type"] == "final"), None)
    if config is None:
        config = json.loads((ROOT / meta["config"]).read_text())
    run = config["runs"][meta["run"]]
    N = run["num_envs"]
    bounds = dict(config["bounds"], B0=dict(config["bounds"]["B0"], own_vram_peak_mib=run.get("own_vram_bound_mib")))
    gate = config["bounds"].get("slope_gate_statistic", "theil_sen")
    exits = [r for r in sup if r["type"] == "exit"]
    attempts, episodes, process_classes = [], [], []
    for ex in exits:
        a = ex["attempt"]
        records, report = S.read_ledger(run_dir / ex["attempt_dir"] / "ledger.jsonl")
        tel = [r for r in tele if r.get("type") == "sample" and r.get("attempt") == a]
        eps = [r for r in records if r["type"] == "episode"]
        episodes += [dict(_episode_row(r), attempt=a) for r in eps]
        end = next((r for r in records if r["type"] == "end"), None)
        built = next((r for r in records if r["type"] == "built"), {})
        beats = [r for r in records if r["type"] == "heartbeat" and r.get("walls_s")]
        walls = np.array([w for r in beats for w in r["walls_s"]], float)
        gpu = [dict(r["gpu"], phase="run") for r in tel if isinstance(r.get("gpu"), dict)]
        adapter = next((r for r in records if r["type"] == "adapter"), {})
        audits = [r for r in records if r["type"] == "l2_audit"]
        process_classes.append(ex["classification"]["cls"])
        attempts.append(dict(
            attempt=a, attempt_dir=ex["attempt_dir"], classification=ex["classification"], wall_s=ex["wall_s"],
            ledger_report=report, incomplete_segments=ex["incomplete_segments"],
            global_steps=end["global_steps"] if end else max((r.get("global_step") or 0 for r in records
                                                               if isinstance(r.get("global_step"), int)), default=0),
            reduced=bool(next((r for r in records if r["type"] == "start"), {}).get("reduced")),
            adapter_source=adapter.get("source"), adapter_infra_key=adapter.get("infra_key"),
            receipt=built.get("receipt"),
            bounds=evaluate_bounds(records, tel, bounds, num_envs=N, quota_classes=S.QUOTA_CLASSES, gate=gate,
                                   required=GATED if meta.get("backend") == "trim" else ("B3", "B4")),
            l2_audits=dict(n=len(audits), foreign_candidates_max=max((r.get("grouped_query_foreign") or 0)
                                                                      for r in audits) if audits else None,
                           range_audit_nonzero=[r["step"] for r in audits if r.get("range_audit_nonzero")],
                           mass_constant_all=all(r.get("mass_bitwise_constant", True) for r in audits)),
            step_wall_s=dict(n=int(len(walls)), median=float(np.median(walls)) if len(walls) else None,
                             mean=float(walls.mean()) if len(walls) else None,
                             p95=float(np.percentile(walls, 95)) if len(walls) else None,
                             env_steps_per_s=float(N * len(walls) / walls.sum()) if len(walls) else None,
                             scope="backend.step wall: batched step incl. the per-step host copy of the info "
                                   "(adapter) and the device L2 update; evaluation, resets, samples outside"),
            telemetry=summarize(gpu) if gpu else dict(gpu_sharing="NOT_SAMPLED"),
            child_rss_peak_mib=max(((r.get("child") or {}).get("hwm_mib") or 0.) for r in tel) if tel else None))
    incomplete = [r for r in sup if r["type"] == "episode_incomplete"]
    by_env = defaultdict(Counter)
    for e in episodes:
        by_env[e["env"]][e["end_class"]] += 1
    for r in incomplete:
        by_env[r["env"]][S.INFRA_INCOMPLETE] += 1
    quota = {e: sum(c[k] for k in S.QUOTA_CLASSES) for e, c in by_env.items()}
    full = {e: sum(c[k] for k in S.FULL_LENGTH_CLASSES) for e, c in by_env.items()}
    drives = {spec["env"]: spec["drive"] for spec in run["envs"]} if isinstance(run.get("envs"), list) else {}
    canary_eps = [e for e in episodes if e["canary"] is not None]
    infra_proc = Counter(c for c in process_classes if c != S.COMPLETE)
    env_invalid = sum(c[S.ENV_INVALID_STATE] for c in by_env.values())
    infra_incomplete = sum(c[S.INFRA_INCOMPLETE] for c in by_env.values())
    families = Counter()
    for cls, n in infra_proc.items():
        families[S.INFRA_CLASSES[cls]] += n
    if env_invalid:
        families["NAN_INVALID_STATE (env level)"] += env_invalid
    labelled = defaultdict(list)
    for e in episodes:
        if e["end_class"] in S.FULL_LENGTH_CLASSES:
            labelled[e["env"]].append(e["labelled"])
    task = {e: dict(success=c[S.TASK_SUCCESS], failure=c[S.TASK_FAILURE]) for e, c in by_env.items()}
    out = dict(
        schema="rcs_rl_p1_4_analysis_v1", run_dir=str(run_dir), run=meta.get("run"), meta=meta,
        supervisor_status=final and final.get("status"), supervisor_ledger=sup_report, telemetry_ledger=tele_report,
        attempts=attempts,
        episodes=dict(
            total_records=len(episodes), per_env_classes={e: dict(c) for e, c in sorted(by_env.items())},
            quota_episodes_per_env=dict(sorted(quota.items())), full_length_episodes_per_env=dict(sorted(full.items())),
            task_outcomes_per_env=dict(sorted(task.items())),
            labelled_full_length_per_env={e: dict(n=len(v), distinct=sorted(set(v)), min=min(v), max=max(v))
                                          for e, v in sorted(labelled.items()) if v},
            l2_pass=sum(1 for e in episodes if e["l2"] == "PASS"),
            l2_fail=[dict(env=e["env"], episode_index=e["episode_index"], end_class=e["end_class"])
                     for e in episodes if e["l2"] == "FAIL"],
            rows=episodes),
        canaries=dict(episodes=len(canary_eps), passed=sum(1 for e in canary_eps if e["canary"] == "PASS"),
                      failed=[dict(env=e["env"], episode_index=e["episode_index"], attempt=e["attempt"],
                                   different_code_particles=e["canary_different_code_particles"])
                              for e in canary_eps if e["canary"] != "PASS"]),
        infrastructure=dict(process_classes=process_classes, process_failures=dict(infra_proc),
                            env_level_invalid_state=env_invalid, infra_incomplete_episodes=infra_incomplete,
                            families=dict(families),
                            note="process classes per attempt; env-level ENV_INVALID_STATE and INFRA_INCOMPLETE "
                                 "episodes are infrastructure tallies, never task failures"),
        native_limitations_retained=config.get("native_limitations_retained"))
    out["acceptance"] = acceptance(out, run, drives)
    return out


def acceptance(out, run, drives):
    quota_needed = run["episodes_per_env"]
    q, f = out["episodes"]["quota_episodes_per_env"], out["episodes"]["full_length_episodes_per_env"]
    deterministic = [e for e, d in drives.items() if d in ("replay_canary", "recorded_command_policy_v1")]
    attempts = out["attempts"]
    bounds = [a["bounds"].get("verdict") for a in attempts if a["bounds"].get("warmup_end_batch_step") is not None]
    census = [a["bounds"].get("B4", {}).get("invalid_multi_grid_process") for a in attempts]
    checks = dict(
        supervisor_complete=str(out["supervisor_status"]).startswith("COMPLETE"),
        not_reduced=not any(a["reduced"] for a in attempts),
        quota_every_env=all(q.get(e, 0) >= quota_needed for e in range(run["num_envs"])),
        full_length_deterministic_envs=all(f.get(e, 0) >= quota_needed for e in deterministic),
        canaries_all_pass=out["canaries"]["episodes"] > 0 and not out["canaries"]["failed"],
        bounds_pass_every_warm_attempt=bool(bounds) and all(v == "PASS" for v in bounds),
        hashgrid_census_valid=not any(census))
    return dict(checks=checks, verdict="PASS" if all(checks.values()) else "FAIL",
                label="INVALID_MULTI_GRID_PROCESS" if any(census) else None)


def compact(analysis):
    out = json.loads(json.dumps(analysis, default=str))
    for a in out["attempts"]:
        a.pop("receipt", None)
        a["classification"].pop("stderr_tail", None)
    out["episodes"]["rows"] = [{k: v for k, v in r.items() if k not in ("final_codes_sha256", "seed")}
                               for r in out["episodes"]["rows"]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--evidence")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads(Path(args.config).read_text()) if args.config else None
    result = analyze(run_dir, config)
    for i in range(1, 1000):
        path = run_dir / f"analysis_{i:03d}.json"
        if not path.exists():
            break
    with path.open("x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True, default=str)
    if args.evidence:
        with Path(args.evidence).open("x") as stream:
            json.dump(compact(result), stream, indent=1, sort_keys=True, default=str)
    print(json.dumps(dict(analysis=str(path), status=result["supervisor_status"],
                          acceptance=result["acceptance"], canaries={k: v for k, v in result["canaries"].items()
                                                                     if k != "failed"},
                          infrastructure=result["infrastructure"]["families"],
                          bounds=[a["bounds"].get("gated_statuses") for a in result["attempts"]]), default=str))


if __name__ == "__main__":
    main()
