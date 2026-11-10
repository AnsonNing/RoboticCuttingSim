# P1-6 R0 — RCS acceptance harness, judge and `acceptance_v1`

Date: 2026-11-10. Branch `claude/rl-p1-6-r0` (from `claude/rl-p1-3a` @ `25c4fc1`). Decision: D-11
(`docs/RL_READY_PLAN.md` §5). Interface: the R0 section of the external agentic optimizer's P1-6 implementation
note (flags, marks, result / verdict / config keys and exit codes copied from there). Compact evidence:
`docs/evidence/rl_ready/p1_6_r0_002.json` (final harness; `p1_6_r0_001.json` is the superseded first version); raw outputs (not committed) under `artifacts/rl_ready/p1_6_r0/`.

## Status

| Item | Status |
| --- | --- |
| R0a harness `scripts/rl_ready/run_perf_acceptance.py` (`perf_acceptance_v1`), 4 modes | DONE |
| R0b judge `scripts/rl_ready/judge_perf_acceptance.py` (`judge_perf_acceptance_v1`) | DONE |
| R0c config `configs/rl_benchmark/acceptance_v1.json` (`rcs_perf_acceptance_v1`), decision D-11 | DONE (frozen from its first commit) |
| Shared pieces `src/robotic_cutting_sim/rl/perf_acceptance.py` (marks, rounds, variant check, codec, config check) | DONE |
| Local runner `scripts/rl_ready/run_perf_acceptance_local.py` (RCS runs an acceptance without the optimizer) | DONE |
| CPU tests `tests/test_perf_acceptance.py` | PASS: `.venv` 39 passed, 1 skipped (Torch-only) in 0.4 s; Torch venv (D-09 overlay) 40 passed in 32.9 s |
| GPU smoke (B = default `frozen_v1`, C = `alias_free_v1`; throughput N=1/2, episode N=1 s512 + s15, samestate s512) + judge | PASS (`smoke_003`: PHYSICS / PERFORMANCE / INTEGRITY PASS, one descriptive band WARN; `SHARED_GPU`) |
| Full acceptance (all plan entries, G9 N=1/8/16 ABBA × 5, held-out s15) | NOT_RUN (the external caller runs it per candidate) |
| `EXCLUSIVE_GPU` rows | NOT_RUN (every smoke row is `SHARED_GPU`) |

## 1. What the pieces do

**Harness** (one process = one side of one run; cwd = the checkout under test, `PYTHONPATH=<checkout>/src`):
`--variant-json` → `open_vector_env(**variant)`; reserved keys (the workload: `num_envs`, `task`, `device`,
`strict`, `cameras`, …) and keys that are not parameters of `open_vector_env` / `VectorCuttingEnv.__init__` raise
`UNSUPPORTED_OPTION` before anything is built. Workload A: the WP-17 Native replays (seed 512 dev, 15 held-out)
through `step_measured_edge` (1 mm / 4 mm / 0.1 ms / 200 substeps, 50 Hz, handle + two measured pads, no cameras,
no recording, 0 robot DOF, `reduced_fidelity`). Seeds run in rounds (`plan_rounds`: cyclic when N ≥ #seeds,
otherwise one round per seed; samestate always per seed). **Every round runs in its own child process** (`5ea4ce4`):
the parent only samples GPU telemetry (its process tree includes the child) and assembles `result.json`, so each
process builds exactly one material and holds one device `wp.HashGrid` (Warp 1.17 static-descriptor hazard; the
child additionally refuses to build while another device grid is alive, `MULTIPLE_DEVICE_HASHGRIDS`). Marks go to
the inherited stdout; the global step is carried across rounds. Marks `RCS_PERF_MARK {"event", "global_step", "num_envs"}` are printed after
`torch.cuda.synchronize()`; `global_step` counts vector steps over the process. The in-process clock spans exactly
the marked window (warm_end → episode_end).

| Mode | What runs | Dumps (npz, no pickles; sha256 in `result.json`) |
| --- | --- | --- |
| `throughput` | 1 untimed + 10 timed full resets, W + K replay steps | final codes per env (work check) |
| `episode` | W + K steps; in-process L2 monitor + 2-frame `evaluate()` (candidate-reported); contact counts per step; Native sample steps | per env: sample codes (exact), sample positions as int8 µm deltas to the Native sample with exact escapes, exact final positions / velocities / codes, contacts, invalid_reason |
| `samestate` | per seed: the Native state of samples 15/33/43/60/90/119 written into the batch, contact stage alone twice (`run_p1_3a` helpers); then a W + K replay from reset | impulses (both reruns), stats, final codes / positions, contacts |
| `profile` | as throughput, the K timed steps inside `cudaProfilerStart/Stop` | final codes |

`result.json` (`rcs_perf_result_v1`): `environment` (`commit`, `package_root` inside the checkout, harness and
shared-module sha256, Python, CUDA_VISIBLE_DEVICES), `variant` (+ sha256), `rows` (the plan columns of the last
round: N, particles/env, control Hz, material dt, cameras, env-steps/s, aggregate sim-time/wall, VRAM peak, reset
p95, physics verdict `REPORTED_NOT_SCORED`, substeps, robot DOF, contacts, recording, warm/cold, GPU, sharing
label and sampled co-tenant statistics), `round_rows`, `rounds` (receipts incl. `particle_contact_receipt`, timing,
resets, HashGrid census, candidate-reported L2 and evaluation), `dumps`, `marks`, telemetry summary.
Exit 0 = completed; 3 = `result.json` with `status: ERROR` (unsupported option, invalid state, missing attribute:
attributable to the code under test); 1 = infrastructure (only `error.json`; the caller retries); 2 = usage or
`--out` exists (nothing written).

**Judge** (CPU, evaluator checkout): validates every run (schema, harness version, commit, package root inside
the side's checkout, harness and `perf_acceptance.py` bytes equal to the evaluator's, mode / N / seeds / steps /
variant as planned, dump sha256) and recomputes:

- episodes (G6, G7), hard gates: final codes = Native, labelled counts = Native, invariants on the
  sampled frames and the final state (finite, digits ≤ 2, no digit rewrites, invalid_reason 0), the frozen TRIM
  evaluator (`goals.evaluate_goal` through the compiled task; references from the WP-17 recording) on 2 frames and
  on the Native sample series, per-env contact totals vs the baseline; candidate-reported L2 vs the recomputation
  (disagreement → INTEGRITY). **Position band: descriptive only.** The final max position difference vs Native
  is compared with the band file named by `physics.position_band.path` (high value per seed, or the reference
  variant's own value for that seed if larger). The P1-0 band (`configs/rl_benchmark/position_band_p1_0.json`,
  min–max of 4 runs, ~40 % false rejects per the acceptance review of `2480dc2`) has `mode: descriptive`: an
  exceedance is a `WARN` in `verdict.warnings` and in the gate values, never a gate result. `mode: gate` makes it a
  PHYSICS check; the coordinator points `path`/`mode` at band v2 (`position_band_v2.json`) when it lands. The band
  file must provide `position_band_m: {"<seed>": [low, high] | {"high_m": x}}`;
- samestate (G4, G8): identical non-zero impulse sets (both reruns) and stats[4] per sample, impulse max relative
  difference ≤ 1e-5, short-replay final codes equal and positions / contact totals within tolerance;
- G1 / G2 from the caller's junit summaries (0 failed; no skip beyond the baseline's);
- G9 per N: candidate / baseline outer-clock env-steps/s per ABBA pair, median and pairs > 1, VRAM, reset p95,
  same work (final labelled counts), inner/outer clock within 5 %, same GPU.

A candidate failure where the baseline fails the same check for the same seed is BLOCKED (the reference does not
meet it in that environment). Integrity problems on the candidate side are INTEGRITY FAIL and FAIL the layer they
invalidate (timing → PERFORMANCE, results / dumps → the gate's layer); on the baseline / caller side they are
BLOCKED. A judge exception writes a BLOCKED verdict (never a FAIL).

**Tests** (`tests/test_perf_acceptance.py`, CPU): config schema and contract values (thresholds, seeds, plan,
reference variant accepted by the variant check, evaluator files / frozen paths match tracked files and leave
`vector_env.py` / `batched/*` free), config problem reporting; mark format and parsing, outer-rate window rule,
rounds, unknown / reserved variant keys → `UNSUPPORTED_OPTION`, position codec bounds and escapes, junit summary;
harness CLI (`--out` exists → exit 2 and nothing written, unknown variant → `result.json` `ERROR` / exit 3,
argument validation) and, in the Torch venv only, one real harness process on the Warp CPU for one control step
(marks, rows, dumps); the judge on synthetic outputs (synthetic Native references injected): PASS, PHYSICS FAIL
(final codes), PERFORMANCE FAIL (slower), INTEGRITY FAIL on clock disagreement (also PERFORMANCE FAIL), INTEGRITY
FAIL on a tampered dump (also PHYSICS FAIL), candidate harness error → FAIL, BLOCKED for missing baseline outputs,
missing runs or a caller-blocked manifest, and the CLI (verdict written, never overwritten; missing WP-17 tree →
BLOCKED, never FAIL).

## 2. Interface: mismatches and how they were resolved

Checked against the external adapter's own validators (read-only import, no bytecode written): its
`core.config_problems(acceptance_v1.json)` returns no problem, `harness.parse_mark` / `inner_rate` /
`environment_problems` accept the harness's marks and `result.json`, `accept.verdict_problems` accepts the
judge's `verdict.json` and `adapter_checks` finds nothing on the local manifest.

| # | Interface point | RCS behaviour | Resolution |
| --- | --- | --- | --- |
| 1 | "Exit 0 when the run completed; non-zero without `result.json` = infrastructure failure" | A failure attributable to the code under test (unsupported option, invalid material state, missing solver attribute) is neither: the harness writes `result.json` with `status: ERROR`, `error_code`, and exits **3**. Infrastructure errors (CUDA, OOM, I/O, missing WP-17 tree) write only `error.json` and exit 1. | Additive. The caller retries only result-less runs; the judge turns a candidate `ERROR` (baseline fine) into FAIL of that gate, and a result-less candidate run whose baseline succeeded (after the caller's retry) into FAIL unless its `error.json` says out-of-memory (then BLOCKED). |
| 2 | Unknown option must raise `UNSUPPORTED_OPTION` | `VectorCuttingEnv` itself raises `TypeError` for unknown keywords. | The harness checks the variant against the signatures of `open_vector_env` / `VectorCuttingEnv.__init__` first and raises `UnsupportedVectorConfiguration("UNSUPPORTED_OPTION")`; keywords that define the workload (`num_envs`, `task`, `device`, `strict`, `cameras`, …, `perf_acceptance.RESERVED_OPTIONS`) are refused the same way. A candidate adds an option by adding a parameter to `VectorCuttingEnv.__init__`. |
| 3 | Judge `--out DIR` | The caller creates `DIR` before calling the judge. | The judge accepts an existing **empty** directory and refuses one that holds anything (exit 2, nothing written). |
| 4 | `rows[*].num_envs == N` for the clock check; several seeds in one call | With several rounds, `rows` holds only the **last** round's row (the one the caller's last `warm_end` → first `episode_end` window measures); every round's row is in `round_rows`. `global_step` is cumulative over the process (differences are what the caller uses). | Consistent with the caller's rule. G9 runs use one seed (one round). |
| 5 | Config keys | The judge also needs `thresholds` (numbers of §10 + `clock_tolerance`) and `physics` (P1-0 band, impulse / contact / replay tolerances). | Both present; `perf_acceptance.config_problems` checks them, the caller's `CONFIG_KEYS` are a subset. |
| 6 | `gpu_mem_mib` example `{"1": 4096, …, "16": 20480}` | Measured process VRAM is 0.83–1.0 GB at N=1–2 (this smoke) and 2.2 GB at N=16 (P1-3a, two variants in one process). The example would require 24.5 GB free on GPU1 for N=16 (with the 4 GB margin), which the shared GPU rarely has. | `{"1": 2048, "4": 3072, "8": 4096, "16": 6144}` (≈ 2.5–3× measured). |
| 7 | Plan `G4_samestate` / `G8_default` `steps: 300` | Particle contacts of the WP-17 replays start at control step 314 (s512) and 321 (s15): a 300-step replay contains no contact at all (smoke_001: contact totals 0 / 0). | `steps: 400` (covers the s512 contact window 314–363 and the s15 onset). The samestate contact-stage samples are `15, 32, 33, 34, 35, 36, 43, 60, 80` (s512 has contacts at 32–36, s15 at 33–85). |
| 8 | VRAM / reset p95 come from the process under test | The harness reports `vram_peak_mib` from nvidia-smi per-process samples (`vram_source: nvidia_smi_process`) and falls back to the allocator peak (Warp mempool high + Torch reserved, `allocator_peak`) when the PID is not visible (PID namespace). | The judge compares only like with like (different sources → `G9:n<N>:vram` BLOCKED). Both values are candidate-reported (limitation in every verdict). |
| 9 | Evaluator file set | The caller's floor includes `tests/*`: a candidate cannot edit an existing test (e.g. a receipt source list); new test files are fine. `rl/vector_env.py` and `batched/*` are deliberately **not** in `evaluator_files` / `frozen_paths`, so a candidate can add opt-in options. | Noted; no RCS change. |
| 10 | Shim note: "`evaluate()` … each env's `success` field" | RCS `VectorCuttingEnv.evaluate()` returns per env `{env, scene_seed, task_id, step_count, time_s, frames, final_codes_sha256, labelled, summary, result, evaluator, wall_s}`; the boolean is `result["success"]` (`summary["verdict"] == "PASS"`), there is no top-level `success`. | For the external side's first real smoke. |
| 11 | Warp kernel cache | Warp writes its kernel cache outside `--out` unless `WARP_CACHE_PATH` is set. | The caller sets it (as specified); the harness writes nothing else outside `--out`. |
| 13 | Position band (coordinator, after the review of `2480dc2`) | Not a hard gate. | `physics.position_band = {path, mode}`; P1-0 values in `position_band_p1_0.json`, `mode: descriptive` (WARN only); a band v2 file needs `position_band_m` per seed (`[low, high]` or `{high_m}`); `mode: gate` then enables it. A missing / unreadable band file is a WARN (descriptive) or BLOCKED (gate). |
| 14 | One material per process (coordinator; the integration branch's `UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS` guard) | The first version built several rounds sequentially in one process (releasing in between). | Since `5ea4ce4` each round is its own child process (G6 N=1 s512 + s15 = two processes); the harness never relies on the guard and never builds a second material in a process. |
| 12 | Reset p95 rule `C ≤ 1.25 × B + 2 ms` | On the shared GPU each process settles in one of two reset-latency modes, ≈ 1 ms or ≈ 7 ms, for all its resets, independently of the variant (smoke_001: B 1.7 / C 7.1 ms at N=1, B 7.4 / C 1.4 ms at N=2), so `G9:n<N>:reset` FAILed on a co-tenant artefact (smoke_001 REJECTED_PERFORMANCE, preserved). | **Amended by coordinator decision (`214ac11`), before any baseline:** per-side medians of reset p95 over all ABBA processes at N; both sides `EXCLUSIVE_GPU` → strict rule (FAIL if C > 1.25 × B + 2 ms); either side `SHARED_GPU` (or `UNKNOWN`) → PASS within the strict limit, WARN `shared_gpu_reset_bimodality` in `verdict.warnings` (not a failure) within 1.25 × B + 10 ms, FAIL beyond. Both slacks and the rule are config `thresholds` (`reset_p95_slack_ms`, `reset_p95_shared_slack_ms`, `reset_p95_rule`). |


## 3. GPU smoke

All runs `SHARED_GPU` (foreign compute processes, foreign SM max 64–98 %); RTX 5090, driver 570.211.01,
`.deps/isaacsim-venv` (Python 3.12.12, Warp 1.17.0, Torch 2.11.0+cu128, NumPy 2.3.1), no Kit; one RCS job per GPU
under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock`. The smoke configs are **not** the contract:
B = `{}` (default `frozen_v1`), C = `{"particle_contact_mode": "alias_free_v1"}` (a known-good, known-faster change),
both from the same checkout. Workload A: s512 (102,732 particles) and, in G6 of smoke_003, s15; 50 Hz, 0.1 ms,
200 substeps, no cameras, no recording, 0 robot DOF, `reduced_fidelity`; throughput warm (W=3, K=20).

```bash
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env TMPDIR=/mnt/data/ningan/tmp \
  .deps/isaacsim-venv/bin/python scripts/rl_ready/run_perf_acceptance_local.py \
  --config artifacts/rl_ready/p1_6_r0/smoke_config_003.json --baseline-checkout . --candidate-checkout . \
  --proposal-json artifacts/rl_ready/p1_6_r0/proposal_alias_free_v1.json --gpu-index 0 \
  --cpu-tests-args "tests/test_repo_naming.py tests/test_rl_benchmark.py tests/test_perf_acceptance.py" \
  --out artifacts/rl_ready/p1_6_r0/smoke_003
```

**`smoke_003`** (final harness `5ea4ce4`, GPU0, 10:29–10:57, 15 MB of outputs; judge 47 s, peak RSS 2.2 GB, `.venv`,
CPU). Evidence `docs/evidence/rl_ready/p1_6_r0_002.json`.

| Gate | Status | Value |
| --- | --- | --- |
| G1 (59 CPU tests, both sides) | PASS | 0 failed, 1 skip (the Torch-only test) on both sides |
| G4 samestate s512: 9 recorded states + 400-step replay | PASS | non-zero impulse sets and stats[4] identical (12–31 contact particles at samples 32–36), impulse rel. diff 0; replay final codes identical, max position diff B–C 7.5e-6 m, contacts 152,546 / 152,550 |
| G6 episode N=1, s512 and s15 (two rounds = two child processes per side) | PASS | codes = Native (0 different) and labelled = Native (37,958 / 85,786) on both sides; evaluator PASS on 2 and 121 frames; invariants PASS; contacts C vs B 2.9e-4 (s512), 5.1e-3 (s15) |
| Position band (descriptive) | WARN | s512 final max diff B 3.89e-5, **C 4.77e-5 m > P1-0 high 3.9e-5 m** (a known-good change; would have been a spurious hard FAIL); s15 B 8.3e-5 / C 8.8e-5 m (high 4.6e-4) |
| G9 N=1 (2 ABBA pairs) | PASS | C/B 1.448, 1.275 (median 1.361); B 3.00 / 3.43, C 4.34 / 4.37 env-steps/s |
| G9 N=2 (primary in the smoke) | PASS | 1.373, 1.365 (median 1.369) |
| G9 VRAM / reset / work / clocks | PASS | 832 → 894 MiB (N=1), 936 → 998 MiB (N=2); reset p95 medians 7.3 vs 7.2 ms, 10.4 vs 6.8 ms; same labelled counts; inner/outer within 1e-4 |

Layers: **PHYSICS PASS, PERFORMANCE PASS, INTEGRITY PASS** (the caller's mapping gives RCS_PR_PENDING); one
descriptive position-band WARN.

Earlier runs (preserved):

- `smoke_002` (`6b71363`, rounds still in one process, the P1-0 band still a hard check; GPU0): PASS / PASS / PASS,
  ratios 1.465 (N=1) / 1.381 (N=2); evidence `p1_6_r0_001.json` (superseded by `p1_6_r0_002.json`).
- `smoke_001` (GPU1, 1 pair, original samestate samples, 300-step G4 replay): PHYSICS PASS, INTEGRITY PASS, ratios
  1.474 / 1.384, **PERFORMANCE FAIL on `G9:n1:reset`** (B 1.71 ms vs C 7.09 ms; at N=2 the other way round) — the
  reset-latency bimodality of item 12 in §2; its G4 replay had zero contacts (item 7).
- `adapter_probe_001` (GPU0 by UUID, `6b71363`): the external adapter's own `harness.command` / `harness_env` /
  `run` driving this harness, throughput N=2, W=3, K=20, B then C: rc 0, inner/outer 1.000004 / 1.000009, no
  environment problems, C/B 1.381.
- `probe_throughput_n2_001` (GPU1) and `cpu_probe/episode_cpu_001` (Warp CPU, 2 steps, 28 s per step): bring-up.

**Re-judge under the amended reset rule** (`214ac11`; CPU only, `.venv`, no GPU rerun; evidence
`docs/evidence/rl_ready/p1_6_r0_003.json`). Each re-judge directory (`artifacts/rl_ready/p1_6_r0/<smoke>_rejudge_001/`)
holds a config copy with the new threshold keys (smoke_001's `position_band_m` mapped to the descriptive band input)
and a manifest copy pointing at it (`rejudge_of` records the original manifest and config sha256); the original
verdicts are untouched. smoke_001's outputs come from the harness before `6b71363` and are accepted with the
judge's re-judge-only `--known-harness` option (recorded in `verdict.rejudge`).

| Smoke | Original verdict | Re-judge | Reset gates (B / C median reset p95, regime, limits) |
| --- | --- | --- | --- |
| smoke_001 | PHYSICS PASS, **PERFORMANCE FAIL** (`G9:n1:reset`), INTEGRITY PASS | **PASS / PASS / PASS**, 1 WARN | n1: 1.705 / 7.094 ms, shared, strict 4.131, shared 12.131 → PASS + WARN `shared_gpu_reset_bimodality`; n2: 7.426 / 1.394 ms → PASS |
| smoke_003 | PASS / PASS / PASS (1 band WARN) | **PASS / PASS / PASS** (same band WARN) | n1: 7.303 / 7.186 ms → PASS; n2: 10.429 / 6.782 ms → PASS (strict limits 11.13 / 15.04 ms) |

After smoke_003 the `contact_total_rel_tol` floor was raised from 0.01 to 0.02: frozen_v1 vs alias_free_v1 s15
contact totals differ by 0.51 %, a 2× margin only, and G7 has one baseline run (no spread term).

## 4. Open

- Position band: `physics.position_band` points at the descriptive P1-0 file; switch `path`/`mode` to band v2
  (`position_band_v2.json`, `mode: gate`) when it lands (the file must provide `position_band_m` per seed).
- The full acceptance (G4/G6/G7/G8 incl. s15 held-out, G9 N=1/8/16 × 5 ABBA pairs) has not been run; the first real
  run is the external caller's T10 GPU rehearsal. `EXCLUSIVE_GPU` rows: NOT_RUN.
- VRAM peak and reset p95 are measured inside the process under test (candidate-reported); only env-steps/s has an
  outer clock.
- G1 runs the whole `tests/` in `.venv`: at `5ea4ce4` (+ the 0.02 contact floor) **2243 passed, 25 skipped,
  6 xfailed, 0 failed in 2576 s (42 min 56 s)** (`artifacts/rl_ready/p1_6_r0/full_cpu_suite_002.{log,xml}`); the
  timeout is 7200 s per suite (the caller runs both sides sequentially, in parallel with the GPU blocks). An earlier
  run (`full_cpu_suite_001`) had 10 failures, all in `test_perf_acceptance.py`, caused by editing the config and
  judge while that run was in progress (module collected before the change); preserved, not a code failure.
