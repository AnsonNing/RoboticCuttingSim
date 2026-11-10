#!/usr/bin/env python3
"""Review fixes of the 2480dc2 acceptance review: compact evidence and additive corrections.

* recomputes the M2 alternation ratios over matched step windows from the raw ``steps.npz`` (F3) and the inferred
  gain over the frozen single env (F4);
* reads the band-v2 classification (``make_position_band_v2.py``), the F1 GPU probe comparison and the refresh-cost
  runs;
* writes ``docs/evidence/rl_ready/review_fix1_001.json`` (never overwritten) and adds a ``corrections_review_fix1``
  block to ``p1_0_baseline_001.json``, ``p1_2_milestones_001.json`` and ``p1_3a_001.json`` (original values kept;
  the block is replaced if this script is run again).

``PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_review_fix1_evidence.py --classification <json>
--f1-compare <json> --refresh-cost <dir> [<dir> ...] --tests <json> --out docs/evidence/rl_ready/review_fix1_001.json``
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "rl_ready"
EVIDENCE = ROOT / "docs" / "evidence" / "rl_ready"
PARTICLES = {512: 102732, 15: 103898}
MEAN_PARTICLES = 103315


def load(path):
    return json.loads(Path(path).read_text())


def git(*args):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def matched_windows():
    """F3: env-steps/s over equal step windows of the alternation rounds (raw step walls)."""
    def rate(run, n, a, b):
        w = np.load(ART / "p1_2" / run / "steps.npz")["step_wall_s"].astype(float)[a:b]
        return float(n * len(w) / w.sum())
    out = {}
    for rnd in ("r1", "r2"):
        n1, n8, n16 = (f"alt_{rnd}_n{n}_001" for n in (1, 8, 16))
        row = dict(n1_windows={f"[{a}:{b}]": rate(n1, 1, a, b) for a, b in ((0, 100), (0, 150), (100, 200),
                                                                            (200, 300), (0, 300))},
                   n8_0_150=rate(n8, 8, 0, 150), n16_0_100=rate(n16, 16, 0, 100))
        row["ratio_n8_over_n1_0_150"] = row["n8_0_150"] / row["n1_windows"]["[0:150]"]
        row["ratio_n16_over_n1_0_100"] = row["n16_0_100"] / row["n1_windows"]["[0:100]"]
        row["reported_before"] = dict(r1=dict(n8=1.4457016763753847, n16=1.4166372167529087),
                                      r2=dict(n8=1.8497606024665385, n16=2.100133569024508))[rnd]
        row["co_tenant"] = "matched (SM mean 46-47 %)" if rnd == "r1" else "confounded (22-35 % vs 45 %), not claimed"
        out[rnd] = row
    single, batched_n1 = 5.630, 5.091
    p12 = load(EVIDENCE / "p1_2_milestones_001.json")
    for row in p12["rows"]:
        if row["source"] == "rl_ready/p1_0/p1_2_single_ref_s512_001":
            single = row["env_steps_per_s"]
        if row["source"] == "rl_ready/p1_2/m2_n1_001" and row["warm_cold"] == "warm":
            batched_n1 = row["env_steps_per_s"]
    factor = batched_n1 / single
    r1 = out["r1"]
    out["f4_vs_frozen_single"] = dict(
        frozen_single_env_steps_per_s=single, batched_n1_env_steps_per_s=batched_n1, batched_n1_over_single=factor,
        scope="full s512 episodes, separate processes, co-tenant SM mean 24 % in both (p1_2_single_ref_s512_001, "
              "m2_n1_001)",
        inferred_gain_over_frozen_single=[r1["ratio_n16_over_n1_0_100"] * factor,
                                          r1["ratio_n8_over_n1_0_150"] * factor],
        status="INFERRED_NOT_MEASURED (no single_env_headless row inside the alternation round)",
        adr_0021_estimate="1.3-3x over sequential single envs: the inferred gain is below the lower estimate "
                          "(ADR 0021 section 14 risk 6)")
    return out


def corrections_p1_2(windows, classification):
    m0 = {r["run"]: r for r in classification["rows"] if r["run"].startswith("p1_2/m0_s")}
    r1 = windows["r1"]
    return dict(
        date="2026-11-10", source="docs/reports/rl_ready/P1_1_P1_2_P1_3A_ACCEPTANCE_REVIEW.md section 7",
        milestone_status=dict(
            M0_L3_position_band=dict(
                previous="3/4 inside; pre-fix m0_s15_001 outside ...; post-fix s15 inside",
                status="SUPERSEDED", band_v1_strict_tally="1/4 inside",
                band_v1_per_run={k: v["band_v1_status"] for k, v in m0.items()},
                band_v2_per_run={k: v["band_v2_class"] for k, v in m0.items()},
                band_v2_aggregate=classification["aggregates"].get("P1-2 M0 batched N=1 frozen_v1")),
            M1_env_independence="PASS (CPU bitwise L1; GPU outcome-level only; L4 BLOCKED) - review F8",
            M2_throughput_gain_over_N1=dict(
                previous="1.45x (N=8), 1.42x (N=16): unequal windows (N=1 steps 0-300, N=8 0-150, N=16 0-100)",
                corrected=f"{r1['ratio_n8_over_n1_0_150']:.2f}x (N=8, steps 0-150), "
                          f"{r1['ratio_n16_over_n1_0_100']:.2f}x (N=16, steps 0-100) over batched N=1, round 1",
                vs_frozen_single=windows["f4_vs_frozen_single"])),
        rows_particles_per_env={"0": PARTICLES[512], "1": PARTICLES[15], "2": PARTICLES[512], "3": PARTICLES[15],
                                "4": MEAN_PARTICLES, "5": MEAN_PARTICLES},
        rows_particles_per_env_note="rows 0-5 (M0 x4, M1 x2) had null; per-env counts 102,732 (s512) / 103,898 "
                                    "(s15), mean 103,315 for the mixed N=4 batch",
        throughput_alternation_matched_windows=windows,
        multi_hashgrid_audit_addendum="default-path refresh + step-time census added (review fix F1); the "
                                      "construction-time guard alone did not cover a grid built after the batch")


def corrections_p1_3a(classification):
    rows = [r for r in classification["rows"] if r["package"].startswith("P1-3a alias_free_v1 batched")
            and "INVALID" not in r["package"]]
    s15 = [r for r in rows if r["seed"] == 15]
    s512 = [r for r in rows if r["seed"] == 512]
    return dict(
        date="2026-11-10", source="docs/reports/rl_ready/P1_1_P1_2_P1_3A_ACCEPTANCE_REVIEW.md section 7",
        failures=[
            dict(index=1, previous="end-to-end throughput gain not demonstrated on the SHARED GPU (interleaved walls "
                 "within +-2 %)", status="WITHDRAWN",
                 reason="derived from same-process multi-variant runs (ab_n*, smoke_ab, pair_n16_reversed) that are "
                        "INVALID_MULTI_GRID_PROCESS; the valid one-variant-per-process runs (solo_*, hardened_*) "
                        "measure 1.40-1.45x (report section 3.0 and 3.4)"),
            dict(index=2, previous="s15 L3 position band (P1-0 single-env) exceeded by alias_free_v1 in 2/3 envs and "
                 "by the frozen_v1 control", status="SUPERSEDED",
                 reason="the frozen_v1 'control' was the INVALID multi-grid episode_15_001; the s15 envs are 4 "
                        "(episode_solo_15, episode_512_15_512_15 envs 1 and 3, hardened_episode_15) and all 4 are "
                        "outside band v1 (3 above, 1 below); band v1 itself is superseded by band v2",
                 band_v1_strict=dict(s512=f"{sum(r['band_v1_status'] == 'inside' for r in s512)}/{len(s512)} inside",
                                     s15=f"{sum(r['band_v1_status'] == 'inside' for r in s15)}/{len(s15)} inside"),
                 band_v2={f"{r['run']}#env{r['env']}": r["band_v2_class"] for r in rows},
                 band_v2_aggregate=classification["aggregates"].get("P1-3a alias_free_v1 batched"))],
        not_run=[dict(index=0, previous="EXCLUSIVE_GPU timing (both GPUs shared by 3+ foreign compute processes during "
                      "every run)", corrected="EXCLUSIVE_GPU timing NOT_RUN: every run had >= 1 foreign compute "
                      "process (most rows record 1; episode_512_15_512_15_001 records 4)")],
        rows_particles_per_env={"14": PARTICLES[512], "15": PARTICLES[15], "16": MEAN_PARTICLES,
                                "17": PARTICLES[512], "18": PARTICLES[15]},
        report_contradiction="P1_3A_ALIAS_FREE_CONTACTS.md section 8 said 'inside the frozen GPU band on GPU' against "
                             "its own status table; corrected in the report",
        default_path_hazard="frozen_v1 now rebuilds its own grid before every launch (review fix F1)")


def corrections_p1_0(classification):
    rows = [r for r in classification["rows"] if r["run"].startswith("p1_0/")]
    return dict(date="2026-11-10", l3_band=dict(
        status="SUPERSEDED", superseded_by="position_band_v2 (configs/rl_benchmark/position_band_v2.json, values "
        "configs/rl_benchmark/position_band_v2_reference.json)",
        reason="[min, max] of 4 runs rejects a conforming run with probability 2/(n+1) = 40 %",
        band_v2_reclassification={r["run"]: r["band_v2_class"] for r in rows}))


def write_block(path, block, key="corrections_review_fix1"):
    text = Path(path).read_text()
    data = json.loads(text)
    data[key] = block
    out = json.dumps(data, indent=1, ensure_ascii=True)
    Path(path).write_text(out + ("\n" if text.endswith("\n") else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--f1-compare")
    parser.add_argument("--f1-runs", nargs="*", default=[])
    parser.add_argument("--refresh-cost", nargs="*", default=[])
    parser.add_argument("--tests")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    classification = load(args.classification)
    windows = matched_windows()
    refresh = []
    for run in args.refresh_cost:
        d = load(Path(run) / "result.json")
        m = load(Path(run) / "manifest.json")
        tel = d.get("telemetry_run_phase") or d.get("telemetry") or {}
        refresh.append(dict(run=str(Path(run).relative_to(ART)) if str(run).startswith(str(ART)) else str(run),
                            status=d["status"], refresh=d["refresh"], num_envs=d.get("num_envs"),
                            step_wall_s=(d.get("step_wall_s") or {}).get("frozen_v1"),
                            env_steps_per_s=(d.get("env_steps_per_s") or {}).get("frozen_v1"),
                            refresh_host_wall_ms=d.get("refresh_host_wall_ms"),
                            refresh_gpu_event_ms=d.get("refresh_gpu_event_ms"),
                            particles_per_env=d.get("particles_per_env"), gpu_sharing=tel.get("gpu_sharing"),
                            foreign_sm_pct_mean=(tel.get("foreign_sm_pct") or {}).get("mean"),
                            foreign_sm_pct_max=(tel.get("foreign_sm_pct") or {}).get("max"),
                            foreign_process_count_max=tel.get("foreign_process_count_max"),
                            gpu_index=m.get("args", {}).get("gpu_index"),
                            git_dirty=m.get("git_dirty"),
                            cuda_visible_devices=m.get("cuda_visible_devices"), git_head=m.get("git_head")))
    evidence = dict(
        schema="rcs_rl_review_fix1_evidence_v1", review="docs/reports/rl_ready/P1_1_P1_2_P1_3A_ACCEPTANCE_REVIEW.md",
        reviewed_commit="3682786", base_commit="2480dc2", branch="claude/rl-review-fix1",
        evidence_commit_parent=git("rev-parse", "HEAD"),
        f1_default_path_descriptor=dict(
            fix="BatchedCuttingMaterial.advance(): refresh_device_hashgrid() (host rebuild of the batch's own grid, "
                "same build call as the substep) before every launch, graph and eager, both particle-contact modes; "
                "step-time census (Python weak registry + last host build, no gc scan, no device sync); sticky "
                "per-env bit FOREIGN_DEVICE_HASHGRID (128) for a guarded batch; receipt()['device_hashgrid']",
            constants=dict(refresh="host_rebuild_before_every_launch_v1", census="device_hashgrid_census_v1"),
            kernel_sources_changed=False, frozen_kernel_pin="tests/test_batched_particle_contacts.py "
            "FROZEN_BATCH_KERNEL_SHA256 unchanged",
            gpu_probe=None if not args.f1_compare else load(args.f1_compare),
            gpu_probe_runs=args.f1_runs, refresh_cost=refresh),
        f2_position_band_v2=dict(method="configs/rl_benchmark/position_band_v2.json",
                                 reference="configs/rl_benchmark/position_band_v2_reference.json",
                                 consumer="robotic_cutting_sim.rl.position_band",
                                 seeds={k: {kk: vv for kk, vv in v.items() if kk != "values_m"}
                                        for k, v in classification["reference"]["seeds"].items()},
                                 reference_runs=classification["reference"]["reference_runs"],
                                 rejected_reference_runs=classification["reference"]["rejected_runs"],
                                 reclassification=classification["rows"], aggregates=classification["aggregates"]),
        f3_f4_throughput=windows,
        tests=None if not args.tests else load(args.tests))
    with open(args.out, "x") as stream:
        json.dump(evidence, stream, indent=1, ensure_ascii=True)
        stream.write("\n")
    write_block(EVIDENCE / "p1_2_milestones_001.json", corrections_p1_2(windows, classification))
    write_block(EVIDENCE / "p1_3a_001.json", corrections_p1_3a(classification))
    write_block(EVIDENCE / "p1_0_baseline_001.json", corrections_p1_0(classification))
    print(json.dumps(windows, indent=1))


if __name__ == "__main__":
    main()
