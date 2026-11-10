# P1-4 — memory, stability and long rollout (T4.1, T4.3, T4.4, T4.5)

Date: 2026-11-10. Branch `claude/rl-p1-4` (from integration `30b8a67`; P1-2b integration `5f315ba` merged in
`4727579` after LR-A ended). Plan: `docs/RL_READY_PLAN.md` §4 P1-4,
design `docs/reports/rl_ready/P1_5_PLAN.md` §6 and tasks T4.1–T4.7, ADR 0021. Status words: PASS / FAIL / BLOCKED /
NOT_RUN. Compact evidence: `docs/evidence/rl_ready/p1_4_001.json`. Raw output (not committed):
`artifacts/rl_ready/p1_4/<run>_NNN/`, logs `artifacts/rl_ready/p1_4/logs/`.

## Status

| Item | Status |
| --- | --- |
| T4.1 supervisor, crash-safe ledger, classification, restart, fault-injection CPU tests | PASS (CPU tests below) |
| T4.3 pre-declared `long_rollout_v1.json` (LR-A) and `long_rollout_v2.json` (T4.5 runs) | PASS (pinned by SHA-256 before their runs) |
| **T4.4 LR-A** N=8, 8 serial full TRIM episodes per env, frozen_v1, replay canaries | **PASS** (all acceptance checks; SHARED_GPU) |
| LR-A canaries (codes bitwise = Native, labelled count, evaluator PASS) | **16/16 PASS** |
| LR-A memory bounds B0–B4 | **PASS** (VRAM slope 0.0 MiB / 1000 batch steps, RSS slope −0.28 MiB / 1000 env-steps) |
| LR-A infrastructure failures | **0** (process COMPLETE in 1 attempt; 0 env-level invalid states; 0 incomplete episodes) |
| Adapter switch to `termination_reason_v1` / `classify_process_exit` (P1-2b) with tested fallback | PASS (CPU) |
| T4.5 LR-B reset stress N=16 | **FAIL** on B0 only (own VRAM peak 2,304 MiB > declared 2,253 MiB); reset check 668/668 bitwise PASS, canary rule PASS, B1–B4 PASS |
| T4.5 LR-C rebuild cycles (10 × open/50 steps/close) | **PASS** (C1–C5) |
| T4.5 LR-D GPU fault probes (NaN env, bad pose (T4.2 re-check), SIGSEGV, hang, device OOM) | **PASS** 5/5 classified as declared (first NaN probe exposed a harness bug: preserved as `lr_d_nan_001`, fixed, rerun) |
| T4.5 LR-A2 (`alias_free_v1`, 4 episodes per env) | **NOT_RUN** (coordinator GPU policy: only if LR-A were invalid; LR-A is valid) |
| T4.6 LR-E Lab B1 long run | **NOT_RUN** (declared outline only; needs a Lab child backend, see §8) |
| EXCLUSIVE_GPU rows | NOT_RUN (every run had co-tenants) |
| Native limitations (`reload_in_place` needs cameras; NOT POLICY_READY) | retained (§6) |

## 1. What was built (T4.1)

| Piece | File | What it does |
| --- | --- | --- |
| Ledger | `src/robotic_cutting_sim/rl/supervision.py` `Ledger`, `read_ledger`, `LedgerFollower` | append-only JSONL, created exclusively; one `os.write` + `os.fsync` per record, so a crash loses at most the record being written; readers tolerate a truncated tail |
| Classification | `classify_process_end`, `classify_error_text`, `classify_episode_end` | every process end is `COMPLETE` or exactly one infrastructure class (below); every episode end is a task outcome, a termination, an env-level invalid state or a truncation |
| `info` adapter | `step_status(info, terminated, truncated, num_envs=N)` | reads `info["termination_reason"]` (P1-2b bitfield, bits shared with the Lab env: 1 rigid, 2 material, 4 grasp lost, 8 IK streak) when present, else derives `MATERIAL_INVALID` from today's `invalid_reason`; a per-env infrastructure key (`infra_error` / `infrastructure_error` / `infra_reason` / `infrastructure_failure`) is read when present; the source is written to the ledger (`adapter` record). `rl/vector_env.py` is not edited |
| Supervisor | `Supervisor`, `SupervisorPolicy`; CLI `scripts/rl_ready/supervise_rollout.py` | one child process per attempt (one material = one device HashGrid per process); watchdog on ledger activity (startup limit until the first stepped heartbeat, then max(`hang_min_s`, factor × median step)); wall limit; `py-spy dump` if installed, SIGTERM, SIGKILL; 5 s samples of per-PID VRAM and co-tenants (`rl.telemetry.take_sample`: compute apps, `-q -d PIDS`, pmon) and child VmRSS / VmHWM / threads / fds (`/proc`); free-VRAM check before launch (D-07); classification; `INFRA_INCOMPLETE` for every segment in flight at a crash; at most `max_restarts` restarts into a new `attempt_NNN/` with `resumed_from` |
| Child | `scripts/rl_ready/run_long_rollout.py` | `trim` backend: workload A `VectorCuttingEnv` (`strict=False`), drives per env in one batched call per step (replay canary via `step_measured_edge`, `RecordedCommandPolicyV1`, `RandomPolicyV1`), device L2 every step with per-episode accumulators, grouped-hash audit every 600 steps, frozen evaluator at every episode end, canary comparison with the Native final codes, census hooks (`wp.HashGrid.__init__` with device and creator, `wp.capture_begin` count, Warp user modules, gc census of live device grids). `fake` backend: NumPy stand-in with the same info contract, for CPU tests only. Child-only fault injection `RCS_FAULT_INJECT` |
| Schedule | `src/robotic_cutting_sim/rl/long_rollout.py` `RolloutSchedule` | stagger segments force-reset at `offset × k`, then back-to-back episodes; quota counting; deterministic seeds `episode_seed(base, env, index)` (SplitMix64); resume continues at each env's next schedule index (an episode in flight at a crash is replayed with the same seed) |
| Statistics | `src/robotic_cutting_sim/rl/stability.py` | Theil–Sen slope with Sen's 95 % interval (gate statistic) and OLS with t-based 95 % interval (always reported), warm-up rule, bounds B0–B5, required-bound rule (a missing counter is NOT_EVALUABLE, never PASS) |
| Analysis | `scripts/rl_ready/analyze_long_rollout.py` | per-env episode classes, quota and full-length counts, canary verdicts, infrastructure tallies (process classes, env-level invalid state, incomplete episodes), bounds per attempt, co-tenant label, timing, acceptance checks; compact evidence |

**Infrastructure classes** (process): `INFRA_OOM_HOST`, `INFRA_OOM_DEVICE` (family OOM); `INFRA_CUDA_ERROR` (CUDA /
kernel error; also a signalled child whose stderr shows a CUDA error); `INFRA_CRASH_SIGNAL`, `INFRA_KILLED`
(segfault / signal; SIGKILL cause UNVERIFIED unless kernel OOM evidence is readable); `INFRA_HANG`,
`INFRA_TIMEOUT` (timeout); `INFRA_INVALID_STATE` (NaN / invalid state at process level, e.g. strict mode);
`INFRA_DISK`; `INFRA_PYTHON_EXCEPTION`; `INFRA_UNKNOWN_EXIT`.
**Episode classes:** `TASK_SUCCESS` / `TASK_FAILURE` (frozen evaluator verdict of a full-length episode),
`TERMINATED` (task-level termination reason, e.g. `CONTACT_SPEED_GUARD`), `ENV_INVALID_STATE` (numerical invalid
bit `NONFINITE_STATE` / `SLOT_OVERFLOW` / `NONFINITE_WRENCH`, or a per-env infra flag: an env-level NaN / invalid-state
infrastructure tally, isolated, excluded from success denominators), `TRUNCATED_BY_FORCED_RESET`,
`TRUNCATED_RUN_END`, `INFRA_INCOMPLETE`. This refines the plan's single `ENV_INVALID` into `TERMINATED` and
`ENV_INVALID_STATE` so that numerical failures land in the infrastructure tally.

## 2. Pre-declared configuration (T4.3)

`configs/rl_benchmark/long_rollout_v1.json` (the plan calls it `p1_4_long_rollout_v1.json`; the task brief's name
is used), committed in `475fb62` before any run and pinned by SHA-256
(`4813f96e…de3ef`, `tests/test_long_rollout_config.py`). It declares LR-A (main), LR-A2, LR-B, LR-C, LR-D;
seeds, drives, offsets, cadence, the warm-up rule, bounds B0–B5, the canary rule, classes, timeouts and the
recovery policy. Deviation from the plan text: the slope gate is the Theil–Sen slope (task brief: robust linear
fit) with OLS and its 95 % interval reported next to it, and a flag when the OLS upper bound exceeds the limit.

| Bound | Quantity | Pass rule (after warm-up) |
| --- | --- | --- |
| B0 | own-PID VRAM peak | ≤ run bound (LR-A 1536 MiB) |
| B1 | Warp mempool current, Torch allocated | max − min ≤ 1 MiB |
| B2 | own-PID VRAM vs batch step | Theil–Sen slope ≤ 2 MiB / 1000 batch steps; no single increase > 64 MiB |
| B3 | VmRSS at post-gc episode-end samples vs env-steps | slope ≤ 0.5 MiB / 1000 env-steps; last-quintile median − second-quintile median ≤ max(64 MiB, 3 %) |
| B4 | graph and grid state | graph captures = 1 per process; HashGrid census = 1 at every census point; device `wp.HashGrid` inits = 1; Warp module count constant |
| B5 | reset latency | reported per kind; not a gate |

Warm-up ends when every env has completed its first quota episode and the reset after it.

## 3. CPU tests (T4.1)

Commands (worktree `/home/ningan/RoboticCuttingSim-rl-p14` at the final head):

- `.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_rl_supervision.py tests/test_rl_stability.py
  tests/test_long_rollout_config.py tests/test_rl_termination.py tests/test_vector_env_contract.py tests/test_rl_benchmark.py
  tests/test_repo_naming.py` → **133 passed, 1 skipped** (the skip is the P1-2b contract test's Torch-only case).
- Torch runtime (`PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src CUDA_VISIBLE_DEVICES=
  .deps/isaacsim-venv/bin/python -m pytest -p no:cacheprovider -q tests/test_rl_supervision.py tests/test_rl_stability.py
  tests/test_long_rollout_config.py tests/test_vector_env_p12b.py tests/test_vector_env_api.py`) → **136 passed** (4 min).

Fault-injection tests (`tests/test_rl_supervision.py`) run the real `Supervisor` over the real child script with
its NumPy fake backend (no GPU): SIGSEGV, SIGABRT, SIGKILL (cause UNVERIFIED without kernel evidence),
`MemoryError`, simulated CUDA error string, simulated CUDA OOM string, hang (heartbeat watchdog; startup limit
before the first step, step limit after), wall timeout, `ENOSPC`, strict-mode invalid env, per-env NaN in
non-strict mode (isolated: the other envs continue), restart into `attempt_002` with the schedule resumed (the
in-flight episode index is replayed with the same seed; 4 `INFRA_INCOMPLETE` episodes recorded), the
`termination_reason` path and the legacy fallback, a second device grid (`INVALID_MULTI_GRID_PROCESS`), the
`termination_reason_v1` split (NaN → `TERMINATED`, bit 64 → `ENV_INFRASTRUCTURE`, strict invalid env →
`TASK_STRICT_INVALID_ENV`, VectorEnvInfrastructureError kinds), LR-B (every reset checked; a leaking reset is
detected) and LR-C (census after close, counters per build), and the diagnostic evaluation of a NaN env
(`evaluation_error` instead of a process failure).

## 4. GPU smoke of the trim backend (not evidence)

`artifacts/rl_ready/p1_4/smoke_trim_001` with a derived smoke config (`artifacts/rl_ready/p1_4/smoke_config_001.json`,
LR-A drives, 40-step episodes, offset 5, 2 episodes per env; status `SMOKE_ONLY`). Command:
`flock …/gpu1.lock env CUDA_VISIBLE_DEVICES=1 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python
scripts/rl_ready/supervise_rollout.py --config artifacts/rl_ready/p1_4/smoke_config_001.json --run SMOKE --gpu-index 1
--name smoke_trim`. COMPLETE in 206 s: N=8, 115 batch steps, median step 1.68 s (5.07 env-steps/s), own VRAM peak 1412
MiB, SHARED_GPU (foreign compute PID 599867, foreign SM mean 48 %, max 68 %), B0–B4 PASS (mempool and Torch
allocated exactly flat, 1 capture, census 1), L2 PASS in 16/16 episodes, grouped-hash foreign candidates 0. As
expected for 40-step episodes nothing is cut (labelled 0), the evaluator returns FAIL_INTEGRITY and the canaries
miss; the smoke only validates the code path.

## 5. LR-A main run (T4.4)

Config `long_rollout_v1` LR-A (`475fb62`, SHA-256 `4813f96e…`), code `cc9fb19` (pre-P1-2b API: `info` has no
`termination_reason`, adapter source `legacy_invalid_reason`, split `p1_4_v1`). Command (worktree
`RoboticCuttingSim-rl-p14`):

```
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock env CUDA_VISIBLE_DEVICES=1 TMPDIR=/mnt/data/ningan/tmp \
  PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rl_ready/supervise_rollout.py --run LR-A --gpu-index 1 --name lr_a
```

Raw output `artifacts/rl_ready/p1_4/lr_a_001/`. Queued 10:33, started 11:32 JST (lock), ended 16:35; 1 attempt,
`COMPLETE`, 5.06 h, 10,650 batch steps (= the declared full-length arithmetic), not reduced. Workload A,
N=8, 102,732 (s512) / 103,898 (s15) particles per env, 50 Hz control, material dt 0.1 ms (200 substeps), no
cameras, no recording, warm Warp cache (build 3.2 s), RTX 5090 GPU1, **SHARED_GPU** (foreign compute and graphics
processes throughout; foreign SM mean 35 %, max 98 %; foreign memory mean 10.9 GiB, max 25.7 GiB; total GPU
utilisation mean 99 %).

**Episodes** (64 evaluated, plus 7 stagger segments and 8 run-end segments, none evaluated):

| env | drive | scene | quota / full-length episodes | outcome | labelled final codes (all 8) |
| --- | --- | --- | --- | --- | --- |
| 0 | replay canary | 512 | 8 / 8 | TASK_SUCCESS 8, canary PASS 8 | 37,958 every episode (= Native) |
| 1 | replay canary | 15 | 8 / 8 | TASK_SUCCESS 8, canary PASS 8 | 85,786 every episode (= Native) |
| 2, 4 | recorded commands | 512 | 8 / 8 each | TASK_SUCCESS 16 | 33,964 every episode |
| 3, 5 | recorded commands | 15 | 8 / 8 each | TASK_SUCCESS 16 | 81,210 every episode |
| 6, 7 | random policy | 512 / 15 | 8 / 8 each | TASK_FAILURE 16 (descriptive) | 0 (the random policy never cuts) |

Canaries: **16/16 PASS** (final codes bitwise equal to the Native final codes, 0 differing particles, labelled
count = Native, evaluator PASS). Every episode of the same env and drive ended with the same labelled count across
8 serial episodes with staggered partial resets in between. L2: 71/71 segments PASS (code monotonicity, finite
state, det F > 0, slot overflow 0); 18 grouped-hash audits with 0 foreign candidates, ranges clean, mass bitwise
constant. No env terminated and none went non-finite, so the diagnostic-evaluation bug found later (LR-D,
`968f522`) was **latent and not exercised** in LR-A.

**Memory** (warm-up ends at batch step 2,250 by the fixed rule):

| Bound | Result | Status |
| --- | --- | --- |
| B0 own VRAM peak | 1,412 MiB (bound 1,536) | PASS |
| B1 Warp mempool current / Torch allocated | max − min 0 B / 0 B over 226 samples (437,089,923 B / constant) | PASS |
| B2 own VRAM vs batch step (2,723 samples after warm-up) | Theil–Sen 0.000 MiB / 1000 batch steps (95 % CI 0.000 to 0.000); OLS 0.000 (CI 0.000 to 0.000); max single increase 0 MiB | PASS |
| B3 RSS at post-gc episode-end samples (57) | Theil–Sen −0.276 MiB / 1000 env-steps (CI −0.322 to −0.173); OLS −0.314 (CI −0.354 to −0.275); quintile medians 1820.2 / 1820.0 / 1807.0 / 1805.4 / 1805.1 MiB, last − second = −14.9 MiB (allowed 64) | PASS |
| B4 graph and grid state | 1 graph capture (0 recaptures), live device HashGrids 1 at every census, 1 device `wp.HashGrid` init, 20 Warp modules constant | PASS |
| B5 reset latency | episode-end partial resets n=64: p50 2.7 ms, p95 7.9 ms, max 8.9 ms; staggered n=7: p50 6.6 ms, p95 9.0 ms | reported |

Child RSS peak (VmHWM) 2,156 MiB. Step timing (batched step incl. the per-step host copy of the info and the
device L2 update; evaluation, resets and samples outside): median 1.697 s, mean 1.700 s, p95 2.509 s,
4.71 env-steps/s at N=8 (SHARED_GPU; not comparable to EXCLUSIVE rows).

**Infrastructure:** process class `COMPLETE` (agrees with the env's `classify_process_exit`: `ok`), 0
infrastructure failures, 0 `ENV_INVALID_STATE`, 0 `INFRA_INCOMPLETE`.

**Acceptance (pre-declared, all PASS):** supervisor COMPLETE; not reduced; ≥ 8 quota episodes for every env;
≥ 8 full-length episodes for every replay and recorded-command env; every canary episode PASS; B0–B4 PASS;
HashGrid census 1 throughout.

## 5b. T4.5 on the P1-2b API (config `long_rollout_v2`, split `termination_reason_v1`)

`long_rollout_v2.json` (`2c75c90`, SHA-256 `524c84d6…`) was declared before any of these runs. All ran on GPU0
(`gpu0.lock`, `CUDA_VISIBLE_DEVICES=0`, `--gpu-index 0`) from a temporary worktree of this branch at the commit
named per run, while LR-A held GPU1 (one RCS job per GPU), all **SHARED_GPU**. Command form:
`flock …/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src
.deps/isaacsim-venv/bin/python scripts/rl_ready/supervise_rollout.py --config configs/rl_benchmark/long_rollout_v2.json
--run <RUN> --gpu-index 0 --name <name> [--fault-inject <spec>]`.

**LR-D fault probes** (N=2: env 0 replay s512, env 1 replay s15; 40 batch steps declared, reduced by design;
max_restarts 0; hang limit 60 s; own VRAM peak 954 MiB each):

| Run | Commit | Fault | Supervisor | Process class / env's `classify_process_exit` | Episodes |
| --- | --- | --- | --- | --- | --- |
| `lr_d_nan_001` | `a884c86` | `nan_env:1@20` | FAILED_INFRA_PYTHON_EXCEPTION | harness bug: `VectorCuttingEnv.evaluate()` raised `ValueError: Expected finite (...,3) points` (`transform_points`) on the diagnostic evaluation of the NaN env; preserved | — |
| `lr_d_nan_002` | `968f522` | `nan_env:1@20` | **COMPLETE** | COMPLETE / ok (agree) | env 1 `TERMINATED` (`MATERIAL_INVALID`; invalid `NONFINITE_STATE`, `NONFINITE_WRENCH`; `evaluation_error` recorded), env 1 then ran 19 clean steps after its reset; env 0 ran all 40 steps, L2 PASS |
| `lr_d_pose_001` | `a884c86` | `bad_pose:0@20` (one non-finite measured pose) | **COMPLETE** (no crash) | COMPLETE / ok | env 0 `TERMINATED` with `MATERIAL_INPUT_DOMAIN_GUARD`, L2 PASS; env 1 unaffected (T4.2 re-check under the supervisor) |
| `lr_d_segv_001` | `a884c86` | `sigsegv@20` | FAILED_INFRA_CRASH_SIGNAL (SIGSEGV) | INFRA_CRASH_SIGNAL / infrastructure CRASH_SIGSEGV (agree) | 2 `INFRA_INCOMPLETE` |
| `lr_d_hang_001` | `a884c86` | `hang@20` | FAILED_INFRA_HANG | detected after 60.0 s of ledger silence, SIGTERM (py-spy not installed) / infrastructure CRASH_SIGTERM (agree) | 2 `INFRA_INCOMPLETE` |
| `lr_d_oom_001` | `a884c86` | `device_oom@20` | FAILED_INFRA_OOM_DEVICE | `torch.OutOfMemoryError` (8 GiB after `set_per_process_memory_fraction(0.02)`, this process only), child `classify_exception` infrastructure/OOM / infrastructure OOM (agree) | 2 `INFRA_INCOMPLETE` |

**LR-C rebuild cycles** (`lr_c_001`, `a884c86`; N=4 recorded commands; 10 × build → full reset → 50 steps →
`close()`; 454 s): **PASS**. C1 live device grids 1 while open and 0 after every close; C2 Warp mempool after
close 82,431 B and Torch allocated 8.5 MB in every cycle (max − min 0); C3 own VRAM after close 1,086 MiB in every
cycle (slope 0.0 MiB / cycle); C4 RSS after close 1852.6, 1978.4, 1979.2, 2049.1, 2051.1, then 1986.4–1986.5 MiB
for cycles 5–9: Theil–Sen 1.04 MiB / cycle (95 % CI 0.0 to 19.1), OLS 6.34 MiB / cycle (the OLS interval exceeds
the 8 MiB bound; the growth stops after cycle 4); C5 one capture and one device grid per build (1…10). L2 PASS in
10/10 cycles; build 3.1 s warm (5.8 s first).

**LR-B reset stress** (`lr_b_001`, `968f522`; N=16: envs 0–1 untouched replay canaries s512 / s15, envs 2–15
random policy, each force-reset with probability 0.05 per step (SplitMix64 counter schedule); 1,000 batch steps;
input guard on (env default); 73.3 min; 102,732 / 103,898 particles per env; SHARED_GPU, foreign SM mean 56 %, max
99 %, foreign memory mean 16.9 GiB): supervisor COMPLETE, adapter source `termination_reason`, process class
COMPLETE (agrees with `classify_process_exit`).

| Check | Result | Status |
| --- | --- | --- |
| reset check (every forced reset: per-env 64-bit checksums of 61 arrays the reset kernels write, seeds excluded, vs the initial full reset) | 668 forced resets, 668 checks, **0 mismatches** (declared estimate about 700) | PASS |
| canary rule (untouched canaries vs Native samples every 100 steps, ≤ 1 differing particle) | 20 samples, max 1 differing particle, max position difference 9.5e-5 m (descriptive) | PASS |
| L2 per segment | 668/668 PASS (no env terminated; the input guard never fired) | PASS |
| B0 own VRAM peak (bound 2,253 MiB = 2.2 GiB) | **2,304 MiB** | **FAIL** |
| B1 Warp mempool / Torch allocated (warm-up 200 steps) | max − min 0 B / 0 B (34 samples) | PASS |
| B2 own VRAM vs batch step (705 samples) | Theil–Sen 0.000 MiB / 1000 batch steps (CI 0 to 0), OLS 0.000; max single increase 0 | PASS |
| B3 RSS post-gc (17 samples) | Theil–Sen −2.04 MiB / 1000 env-steps (CI −3.52 to 0.00), OLS −2.50; last − second quintile −19.7 MiB | PASS |
| B4 | 1 capture, census 1, 1 device grid init, modules constant | PASS |
| B5 forced partial reset latency (n=504 reset calls) | p50 8.2 ms, p95 15.9 ms, max 23.3 ms | reported |

Step median 4.31 s, p95 5.16 s (3.71 env-steps/s at N=16, SHARED_GPU). **LR-B verdict: FAIL** (B0). Diagnosis (not a
re-score): own VRAM rose from 1,364 to 2,304 MiB in the window that contains the first full reset and the first
computation of the reset-check reference, then stayed exactly flat for the whole run. Torch's reserved memory
grew from 452 to 680 MiB (+228 MiB) at that point: the checksum temporaries of the harness's own reset check
(int64 copies of node-slot arrays) held by the caching allocator. M2's N=16 run without this check peaked at 2,064
MiB, so the excess matches the measurement overhead, not env growth. A rerun would need a chunked checksum or a
new declaration (v3); neither was done here.

## 6. Retained Native limitations (no new Native long rollout in P1-4)

- `stage_reset="reload_in_place"` requires live cameras; a camera-off in-place reset crashes PhysX (4 of 4 probes,
  [WP11_PHASE_H_PROFILING](../WP11_PHASE_H_PROFILING.md)).
- The Native `reload_in_place` memory probe (8 × 20 steps): host 24.2 MiB / episode, GPU 9.3 MiB / episode
  (`docs/POLICY_READY_API.md`).
- Native stays NOT POLICY_READY (GPU 52 MiB / episode vs threshold 32, unresolved; `docs/PROGRESS.md`).
- Batched results never replace these FULL_NATIVE_REFERENCE limitations.

## 7. P1-4 acceptance mapping

| Plan item | Evidence | Status |
| --- | --- | --- |
| long vectorized rollout with repeated partial resets | LR-A: 10,650 batch steps, N=8, 71 partial resets (64 episode-end + 7 staggered) while other envs were mid-episode; LR-B: forced resets at N=16 | PASS |
| bounded VRAM / RSS (slope reported) | LR-A B0–B4 PASS (VRAM slope 0.0 MiB / 1000 batch steps, RSS slope −0.28 MiB / 1000 env-steps, both with 95 % CIs); LR-C C1–C5 PASS; LR-B B1–B4 PASS (slopes 0.0 and −2.0) but **B0 FAIL** (peak 2,304 > 2,253 MiB, harness check overhead) | PASS on LR-A (the main run); LR-B budget FAIL reported |
| 8+ serial full episodes | LR-A: 8 full-length episodes in each of 8 envs (64) | PASS |
| infrastructure failures reported apart from task failures | process classes and env-level tallies (§1); CPU fault tests; LR-D 5 GPU probes classified as declared, agreeing with the env's `classify_process_exit` | PASS |
| Native limitations retained | §6 | PASS (retained, unchanged) |

P1-4 overall: the plan's five acceptance items are met by LR-A plus the fault tests and probes (`SHARED_GPU`,
workload A `reduced_fidelity`). Not clean: LR-B is FAIL on its declared VRAM budget B0 (all other LR-B checks PASS);
LR-A2 and LR-E are NOT_RUN by decision / scope.

## 8. Open items and what T4.5 needs

- **Env gap (for P1-2b owners):** `VectorCuttingEnv.evaluate()` raises `ValueError` from `transform_points` for an
  env whose state is non-finite instead of returning an integrity failure. The P1-4 child now records
  `evaluation_error` (`968f522`); the env itself is unchanged here.
- **Split note:** under `termination_reason_v1` (the env's own classification) a non-finite material state is a
  task-level `MATERIAL_INVALID` termination, not infrastructure; only bit 64, `VectorEnvInfrastructureError` and
  crashes are infrastructure. LR-A used the earlier split `p1_4_v1` (numerical invalid = env-level
  infrastructure); it had no terminations, so both splits give the same LR-A tallies.
- **LR-A2 (`alias_free_v1`)**: NOT_RUN by the coordinator's GPU policy (full run only if LR-A were invalid). It is
  declared in `long_rollout_v2` (4 episodes per env, 5,850 batch steps, about 2.6–4.1 h shared). Short
  alias-free evidence exists in P1-3a.
- **T4.6 LR-E (Lab B1 long run)**: NOT_RUN. Needs a `lab` child backend over
  `open_vector_env(backend="isaac_lab_newton", tool_mount="rigid_mount_v1")` in `.deps/isaaclab-venv`, a
  recorded-command action table re-encoded from the Lab HOME edge pose (the WP-17 tables start from the Native
  reset edge), per-episode evaluation through the Lab env's `evaluate`, and a v3 run entry with the Lab options;
  about 1.3–2 h on GPU0.
- **LR-B B0 FAIL**: the reset-check checksum temporaries add about 230 MiB of Torch reserved memory at N=16. Options:
  chunk the checksum per array and call `torch.cuda.empty_cache()` after the reference, then rerun under a new
  declaration (v3), or accept the FAIL with the diagnosis. The batched env's own memory was flat.
- **EXCLUSIVE_GPU** rows: NOT_RUN (co-tenants in every window).
- `py-spy` is not installed: hang dumps are unavailable (the watchdog records that).
- LR-B ran no grouped-hash audits (`reset_stress` mode does not call the audit; LR-A ran 18 with 0 foreign
  candidates).
