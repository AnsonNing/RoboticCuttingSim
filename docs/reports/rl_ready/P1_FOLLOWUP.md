# Phase 1 follow-up: `acceptance_v2` and review findings 3, 4, 10, 12

Date 2026-11-10, revised 2026-11-11 after the follow-up review (`P1_FOLLOWUP_REVIEW.md`, verdict
PASS_WITH_LIMITATIONS). Branch `claude/rl-p1-followup`, from `c5e96b2`, whose code is identical to the reviewed
`f03fbff`. Source: `docs/reports/rl_ready/PHASE1_ACCEPTANCE_REVIEW.md` (findings 1, 2, 3, 4, 10 and 12, §7, §8).
Decision: D-14 in `docs/RL_READY_PLAN.md`. Raw outputs, not committed: `artifacts/rl_ready/p1_followup/`. Compact
evidence: `docs/evidence/rl_ready/p1_followup_001.json` (first declaration) and `p1_followup_002.json` (re-declared
window, GPU re-check).

## Status

| Item | Status | Where |
| --- | --- | --- |
| Finding 1: the G9 window times the tool-contact and cutting phase (`acceptance_v2`) | DONE. Window re-declared to steps 229–400 before any candidate was judged (follow-up review finding 1, coordinator decision (b)). GPU re-check PASS (`SHARED_GPU`, foreign SM 0 %). A full acceptance under v2 is NOT_RUN. | `configs/rl_benchmark/acceptance_v2.json`, judge `G9:window` |
| Finding 2: frozen contract, pin tests | DONE. `acceptance_v1` is pinned at `3592d940…`. `acceptance_v2` is pinned at `b66ace29…` (the first declaration `a008dd38…` was replaced, never used) and freezes at its merge into `main`. | `tests/test_acceptance_frozen.py` |
| Harness and judge config path | DONE. The judge and the local runner already took a required `--config`; the harness gains an optional `--config` (provenance only, default unchanged). v1 is judged exactly as before (tested; the review re-judged `smoke_003` with both judges: identical gates). | `scripts/rl_ready/{run,judge}_perf_acceptance.py` |
| Finding 3: workload-B `fidelity()` and the time step | DONE (CPU and Lab Warp-CPU tests) | `rl/lab_vector_env.py` `lab_fidelity` |
| Finding 4: failed B2 settle latched in non-strict mode | DONE (Lab Warp-CPU test with real failing gates) | `rl/lab_batched_env.py` `reset_envs` |
| Finding 10: caller errors and the contact-pool overflow are infrastructure | DONE (CPU tests). No new class. | `rl/supervision.py`, `rl/termination.py` |
| Finding 12: rollout files label privileged fields | DONE as an opt-in `rcs_rl_rollout_v2`; v1 stays readable. The env metadata labels the privileged info keys, including `autoreset` / `terminal_captured` (follow-up review finding 6). | `rl/rollout.py`, `check_rl_rollout.py`, `rl/vector_contract.py` |
| Follow-up review findings 1–3 (wording, window, statistics, figures), 5 (CI on `main`), 6, 7 (README) | DONE on this branch. The progress file's U2 is the coordinator's. | this report, D-14, docstrings, workflow, README |
| D-14 and this report | DONE | |

## 1. `acceptance_v2` (findings 1 and 2)

**Unchanged from v1:** the schema `rcs_perf_acceptance_v1` (so the external adapter's reader works), harness, judge,
evaluator file set, reference variant `alias_free_v1`, seeds, the physics plan (G4, G6, G7, G8), every threshold except
`primary_pairs_above_1`, the physics tolerances, the frozen paths, the dev limits and the timeouts. A CPU test asserts
this key by key. `version` is `acceptance_v2`, and the adapter refuses a file whose name and version disagree.

### 1.1 The window: steps 229–400 of the s512 replay from reset, at every N

**Two contact counters.** The material solver keeps per-env statistics per control step:

| Counter | What it counts |
| --- | --- |
| stats slot 1, **tool–material contact** | particles in contact with the tool obstacles (handle, pads, edge) |
| stats slot 4, **particle–particle contact** | cut faces, or pairs from distant reference positions |

The first declaration (`a008dd38…`, steps 321–400) and my first report called slot 4 "contact" and placed the
window after its onset. That was wrong for the claim "times tool–material contact": the window held 46 of the 101
tool-contact steps and 44 % of the labels (follow-up review finding 1).

**s512 data.** Sources: Native `privileged_diagnostics.material_contact_statistics` in the WP-17 `actions.jsonl`, the
`smoke_003` G6 dumps (slot 4) and the GPU re-check dumps (both slots):

| s512 | Tool contact (slot 1) | Particle contact (slot 4) | Labels |
| --- | --- | --- | --- |
| whole episode | steps 229–366, 101 non-zero steps (gaps 237–256 and 259–273) | steps 315–364, 49 steps | 0 at step 250, 1,978 at 260, 37,958 from 380 |
| first declaration, 321–400 | 46 steps | 44 steps | +16,588 |
| **v2 window, 229–400** | **all 101 steps** | **all 49 steps** | **0 → 37,958 (all)** |
| untimed warm-up, 1–228 | none | none | none |
| v1 window at N=8/16, 21–120 | none | none | none |

**Seed 15** (held out, not timed): tool contact 229–516, particle contact 321–842, labels until step 530.

**Why W=228, K=172:**
- Step 229 is the first Native tool-contact step, so W=228 is the largest warm-up that loses none of the
  tool-contact phase.
- A G9 run is 400 replay steps whatever the W/K split, so this window costs the same as the first declaration.
- Every N times the same steps; v2 has no `steps_by_num_envs`.
- The 34 steps after contact (367–400) remain.
- The timed window now includes the cutting-onset steps 255–258 and 274–275. These are the steps where tool contact
  resumes after a gap, and they show 2.4–3.0× step walls in both GPU checks. Their cause is not established; it is now
  per-episode work inside the window.

**`G9:window`** (judge, `performance.window_check`):
- It reads the G6 and G7 episode dumps of the same variants, which replay s512 from reset. The harness now dumps slot 1
  per step as `tool_contacts`, next to slot 4 as `contacts`, from one host read per step. G9 throughput runs read no
  statistics, so their timing is unchanged.
- In every s512 env, on both sides, it requires at least 80 tool-contact steps and at least 30,000 new labels in the
  window. Both minimums were set from the Native data (101 steps, 37,958 labels) with about 20 % margin, before the GPU
  re-check.
- Particle contacts are reported per env, never gated.
- A dump without `tool_contacts` (an older harness) cannot pass.
- It runs on the episode processes, not inside the G9 throughput processes. That limitation is written into every
  verdict.

**`performance.phases`** (descriptive) reports the median step wall per side, and their ratio, from the harness's
per-round `step_walls_s`, which are in-process and candidate-reported. The phases are:

| Phase | Steps |
| --- | --- |
| approach | 21–228 |
| tool contact before cutting | 229–250 |
| cutting before particle contact | 251–314 |
| particle contact | 315–364 |
| after contact | 365–400 |
| timed window | 229–400 |

### 1.2 Statistics and budget, set before any run

**Kept from v1:**
- ABBA pairs inside one lock hold per N, on the same GPU;
- the caller's outer clock;
- N=16 as the primary N (median ≥ 1.05 and every pair > 1);
- the every-N floor (median ≥ 0.97);
- the 5 % clock tolerance.

**Changed:** pairs go from 5 to 3 (`BCCBBC`), and N goes from {1, 8, 16} to {1, 16}.

**This is a trade-off, not stricter than v1 overall.** The follow-up review's finding 2 corrects my earlier "stricter"
claim:
- Only the sign-test part is stricter: 3 of 3 pairs gives a 1/8 false-accept rate under no speed change, against 3/16
  for 4 of 5.
- With the 1.05 median floor, both false-accept rates are about 0 for a neutral candidate.
- At pair noise σ ≤ 0.02 (log ratio), v2 roughly doubles the near-threshold errors in both directions.
- At σ ≥ 0.03, v2 loses power: a true 1.10 is rejected about 14 % of the time at σ = 0.05, against 5 % for v1.
- The N=1 floor rests on 3 pairs of 172 timed steps (v1: 5 pairs of 300). The 172 steps reduce the noise penalty
  against the first declaration's 80.
- N=8, and every N other than 1 and 16, is not gated.
- Exclusive repeats show σ ≈ 0.006, where these effects are negligible. Run acceptances on an `EXCLUSIVE_GPU` where
  possible.

N=8 is dropped rather than N=1 because N=8 and N=16 are both in the baseline's saturated regime: aggregate env-steps/s
is flat from N=4 on. N=1 is the only unsaturated point; it is 9.6 % of the G9 GPU time and 5.3 % of one acceptance.

**Budget, GPU time only.** Inputs are the EXCLUSIVE workload-A rows of `alias_free_v1`: per-step medians of 0.0978 s
(N=1) and 1.1799 s (N=16), all pre-contact, plus 13.0–14.8 s of overhead per process. A G9 run is 400 steps: 52 s at
N=1 and 487 s at N=16.

| Design | G9 GPU time | Largest G9 lock hold | One acceptance (exclusive) | At ×1.3 / ×1.5 sharing |
| --- | --- | --- | --- | --- |
| v1 statistics with a contact window (5 pairs, N 1/8/16) | 130 min | 81 min (N=16) | 2.9 h | 3.8 / 4.3 h |
| 3 pairs, N 1/8/16 | 78 min | 49 min | 2.04 h | 2.65 / 3.1 h |
| **v2: 3 pairs, N 1/16** | **53.9 min** | **48.7 min** | **1.64 h** | **2.13 / 2.46 h** |

The physics plan, unchanged, adds 44.5 min:

| Gate | Runs | Time |
| --- | --- | --- |
| G6 | 6 × 261 s | 26.1 min |
| G7 | 2 × 362 s | 12.1 min |
| G4 | 4 × 57 s | 3.8 min |
| G8 | 2 × 75 s | 2.5 min |

**Contact-phase revision.** The factor is the mean step over steps 1–400 divided by the approach mean. It was measured
only at N=2, for `alias_free_v1`, in episode mode, in two `SHARED_GPU` runs:

| Run | Foreign SM mean | Factor | Contact-phase median / approach median |
| --- | --- | --- | --- |
| `gpu_window_check_001` | 31.5 % | 1.061 | 1.33 |
| `gpu_window_check_002` | 0 % | 1.033 | about 1.01 |

The 1.33× of the first check was therefore mostly co-tenant noise. With these factors:

| Item | Revised |
| --- | --- |
| One acceptance, exclusive GPU | about 1.69–1.75 GPU-h |
| Largest sharing factor that fits 2.5 h | about ×1.43–1.48 |
| Largest lock hold (G9 N=16), exclusive | about 50–52 min |
| Largest sharing factor for a ≤ 90-min hold | about ×1.73–1.79 |

The factor at N=16 is unmeasured. The config's budget text was written before the re-check and quotes only the first
factor.

**Not in the GPU time:** the G1/G2 CPU suites (`cpu_tests.args ["tests"]`, 50–53 min per side on this host). The
external caller runs them in parallel with the GPU blocks; the local runner runs them one side after the other.

Under heavy co-tenancy, such as `smoke_003` at about ×2.2, one acceptance exceeds 2.5 h, and its absolute rows are
`SHARED_GPU`. The caller's GPU admission should wait instead.

### 1.3 Kept, with reasons in the config

- **VRAM peak and reset p95 stay candidate-reported:** `thresholds.candidate_reported`, plus the judge limitation in
  every verdict.
- **Reset p95:** strict on `EXCLUSIVE_GPU`; on `SHARED_GPU`, a WARN up to +10 ms, then FAIL.
- **`contact_total_rel_tol` stays 0.02.** `contact_total_rel_tol_note` gives the reason: in `smoke_003`, the s15 contact
  totals of `frozen_v1` and `alias_free_v1` differ by 0.51 %. A floor of 0.01 would leave only a 2× margin for a
  known-good change, and G7 has one baseline run, so it has no spread term.
- **The position band:** `position_band_v2.json`, descriptive. The band files are pinned too.
- **`baseline_measure` uses the v2 window:** throughput at N 1/8/16 and a profile at N=8, all with W=228 and K=172.

### 1.4 Freeze (finding 2)

`tests/test_acceptance_frozen.py` pins:
- `acceptance_v1.json`: `3592d9408f314c3d2825171361b8ec640534c60577bb6216a10c4f2b22897bf4`, frozen at the PR #12 merge `40c94c7` (D-11);
- `acceptance_v2.json`: `b66ace2967b4aad8382ca4297032c0e513a4cf3ebe6564dad03494b8e7dd612a`;
- both band files.

It also checks that each config's version equals its file name, and that every `acceptance_v*.json` is pinned.

The pin changed once, from `a008dd38…` to `b66ace29…`, when the window was re-declared. That was before any candidate
was judged and before the merge, the freeze point. No verdict exists under the first declaration. After the merge into
`main`, any change is `acceptance_v3`.

The "frozen from its first commit" wording was corrected in D-11 and in the R0 report on `claude/rl-phase1`
(`0b99e7b`), which is merged into this branch. This branch does not edit either file.

### 1.5 External adapter compatibility (read-only check)

I imported the adapter's own `core.config_problems` read-only, with no bytecode written. It returns no problem for v1 or
the re-declared v2. Its acceptance driver:
- reads a single `performance.warmup_steps`, and `steps_by_num_envs` with an `or {}` fallback. v2 has no
  `steps_by_num_envs`, so every N uses `steps` 172.
- reads `order` and `pairs` for every N;
- passes no `--config` to the harness, which is why the harness flag is optional.

Its dev limits cap only `--steps` (300 ≥ 172), so dev runs of the v2 window are allowed.

## 2. GPU window checks (GPU0, `SHARED_GPU`)

### 2.1 Re-check of the re-declared window (`gpu_window_check_002`)

```bash
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
  WARP_CACHE_PATH=/mnt/data/ningan/tmp/warp_cache_p1fu PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  .deps/isaacsim-venv/bin/python scripts/rl_ready/run_perf_acceptance.py \
  --variant-json artifacts/rl_ready/p1_followup/variants/alias_free_v1.json --mode episode --num-envs 2 --seeds 512 \
  --warmup-steps 228 --steps 192 --device cuda:0 --gpu-index 0 --config configs/rl_benchmark/acceptance_v2.json \
  --out artifacts/rl_ready/p1_followup/gpu_window_check_002
```

**Run setup:**

| Item | Value |
| --- | --- |
| Commit | `4943d75` (the re-declared config, sha256 `b66ace29…`), clean tree |
| Process | one process, one round, wall 78 s (build 1.2 s, warm kernel cache), 01:12–01:13 on 2026-11-11 |
| GPU | RTX 5090, driver 570.211.01 |
| Software | Warp 1.17.0, Torch 2.11.0+cu128 |
| Workload | N=2, 102,732 particles per env, 50 Hz, 0.1 ms material dt, 200 substeps, no cameras, no recording, warm |
| Co-tenants | label `SHARED_GPU`: one foreign compute PID, 642659 (not touched), foreign SM 0 % |
| Own VRAM peak | 1,014 MiB |

**Result: PASS.** Both envs, inside 229–400:

| | e0 | e1 | Native |
| --- | --- | --- | --- |
| tool-contact steps | 101 of 101 (229–366) | 101 of 101 (229–366) | 101, the same non-zero set |
| tool contacts, total | 308,502 | 308,718 | 308,600 |
| particle-contact steps (reported) | 49 of 49 (315–364) | 49 of 49 (315–364) | 49 |
| labels, step 220 → 400 | 0 → 37,958 | 0 → 37,958 | 37,958 |
| `invalid_reason` | 0 | 0 | |

Both envs meet the gate minimums (80 steps, 30,000 labels) with margin.

**In-process per-phase step walls** (N=2, candidate-reported):

| Phase | Steps | Median | Mean |
| --- | --- | --- | --- |
| approach | 21–228 | 0.1597 s | 0.1598 s |
| tool contact before cutting | 229–250 | 0.1595 s | 0.1596 s |
| cutting before particle contact | 251–314 | 0.1614 s | 0.1859 s |
| particle contact | 315–364 | 0.1621 s | 0.1624 s |
| after contact | 365–400 | 0.1611 s | 0.1611 s |
| timed window | 229–400 | 0.1614 s | 0.1705 s |

The mean step over 1–400 is 1.033× the approach mean. The slowest steps are the cutting-onset steps: 257, 256, 274,
255 and 275, at 3.03, 3.01, 2.89, 2.84 and 2.44× the approach median.

### 2.2 First check, of the first declaration (`gpu_window_check_001`, preserved)

**Setup:** commit `cf04c1a`, W=320, K=100, wall 138 s (build 22 s with a fresh kernel cache). Foreign SM mean 31.5 %
(max 71 %), 4 foreign compute PIDs.

**Particle contact (slot 4):** steps 315–364 in both envs. In 321–400, 44 steps and +16,587 / +16,588 labels.

**Per-phase medians:**

| Phase | Median step wall |
| --- | --- |
| approach | 0.245 s |
| particle contact, 321–364 | 0.326 s |
| window, 321–400 | 0.295 s |

The 1.33× contact-phase factor of this run is not reproduced on the quiet GPU (§2.1), so it is mostly co-tenant
interference.

## 3. Finding 3: workload-B `fidelity()`

`LabVectorCuttingEnv.fidelity()` now calls `lab_fidelity` (`lab_fidelity_v2`).

**What it compares:**
- The batched material's packed values are compared with `NATIVE_TRIM_V7_2_REFERENCE`, using the same keys as
  workload A (`vector_env.FIDELITY_REFERENCE_KEYS`, through the shared helper `reference_reasons`): law, max cuts, slots,
  spacing, dx, material dt, substeps, control dt and episode steps.
- A non-default `feedback`, `ik_iterations` or `velocity_feedforward` is also a reason.

**What it returns:** `reference`, `actual` and `non_default_options`. A non-default `material_dt_s` is now reduced
fidelity, with a named reason, and is never silently full. The recorded B2 configuration stays not reduced (tested).

**Tests:**
- `tests/test_rl_lab_fidelity.py` (CPU) covers dt 2e-4 and 5e-4, every reference key, the robot options, B1 and the
  default values.
- The Lab Warp-CPU `open_vector_env` test now runs the review's scenario: B2 with `mjwarp_grip_v1`, non-strict, dt 2e-4.
  It asserts the dt, substeps 100, spacing and episode-step reasons (`run_p1_2b_smoke.py --material-dt`).

## 4. Finding 4: B2 settle failure in non-strict mode

**The fix:**
- `reset_envs` now re-applies `GRASP_SETTLE_FAILED` (rigid_reason bit 16) and `TERM_RIGID_INVALID` to every masked
  world whose `GraspSnapshot.passed` is false. This is `grasp_settle_latch_v1`.
- The caller's first `reset()` is a snapshot restore through `reset_envs(all)`. It therefore no longer clears the flag.
- Such a world terminates on its first step after every reset.

**Receipts:**
- `grasp_initialization` is now in both receipts (the env and the vector env).
- The env receipt also gains `grasp_settle_latch`, which lists the failed worlds, and the bit 16 name.

**Test** (Isaac Lab venv, Warp CPU; `probe_lab_batched_cpu.py --mode grasp_latch`):
1. B2 runs non-strict with an unmeetable translation tolerance (1e-9 m), so both worlds FAIL their real settle gates.
2. World 0's snapshot flag is then overridden to "passed", a test-only injection that makes the per-world masking
   visible.
3. After the first `reset()`, world 1 keeps bit 16 and `TERM_RIGID_INVALID`, and world 0 is cleared.
4. After one step, world 1 is terminated.
5. A masked reset re-latches it.

Strict mode, the default and the mode of every recorded run, still raises at build.

## 5. Finding 10: classification

| Path | Before | After |
| --- | --- | --- |
| Child `infra_error` with env category `caller`, split `termination_reason_v1` | `classify_error_text`; a `FloatingPointError` or non-finite message became `TASK_STRICT_INVALID_ENV`, and `agrees_with_env_exit` hid it | `INFRA_PYTHON_EXCEPTION`, refined only to the OOM, disk, CUDA or pool classes when the text names one; the disagreement with `classify_process_exit` stays visible |
| `classify_process_exit` with "Invalid rigid worlds … CONTACT_POOL_OVERFLOW" | task `STRICT_INVALID_ENV` | infrastructure `CONTACT_POOL_OVERFLOW`, tested before the task signature (as in `classify_exception`) |
| Legacy info path (no `termination_reason`), material bit 256 | `infra` 0, so a task-level `TERMINATED` | `infra` 1 from the derived INFRASTRUCTURE bit: `ENV_INVALID_STATE` under `p1_4_v1`, `ENV_INFRASTRUCTURE` under `termination_reason_v1` |

No class is added. A test asserts that the classes used are in the declared `long_rollout_v1` and `long_rollout_v2`
class sets. One existing assertion in `tests/test_rl_termination.py` encoded the old legacy-path behaviour
(`infra == [0, 0, 0]`); it now expects `[0, 1, 0]`, with a comment.

## 6. Finding 12: privileged fields in rollout files

**Opt-in writer `RolloutWriter(schema="rcs_rl_rollout_v2")`.** It writes the same arrays as v1 and adds to the manifest:

| Manifest key | Content |
| --- | --- |
| `field_roles` (`rollout_field_roles_v1`) | the role of every stored field |
| `privileged_non_deployable` | `invalid_reason`, `termination_reason`, and `obs_critic` / `next_obs_critic` when present |
| `oracle_derived` | the privileged fields plus `rewards` (severed-bond progress) and `terminated` |
| `deployable_policy_inputs` | exactly `obs_policy` / `next_obs_policy` |
| `actions_semantics` | the action the caller stored: `run_rl_smoke.py` stores the policy output clamped to [-1, 1]. This also corrects the docstring's "executed action" claim. |

**Compatibility:**
- v1 stays the default and readable. `load_rollout` reads both versions.
- `check_rl_rollout.py` adds the item `non_deployable_fields_labelled` for v2 rollouts.
- For a v1 rollout it reports `field_roles: UNLABELLED_V1` without changing the v1 result.
- `run_rl_smoke.py --rollout-schema` selects the version.

**Env side:**
- `vector_contract.PRIVILEGED_INFO_KEYS` (the oracle-derived step info keys) is published as
  `env.metadata["privileged_info_keys"]` on both backends.
- Since the follow-up review (its finding 6), the list includes `autoreset` and `terminal_captured`. Under
  `autoreset="same_step"`, these are the done flags derived from the oracle termination.
- The shared contract checks every privileged key that a step's info actually carries, including optional keys.
- A CPU source scan checks that every `info[...]` key written by `VectorCuttingEnv` is either a contract key or a
  labelled privileged key.

**Tests:**
- CPU: the role table.
- Torch venv: the S-CPU rollout is rewritten as v2, with and without critic, and passes the consumer check; a mislabelled
  v2 manifest FAILs.

## 7. Tests

All suites ran at code commit `2195bb2` (the last code commit before the docs). Raw logs and junit files are in
`artifacts/rl_ready/p1_followup/`. No test was run on GPU1.

```bash
CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp .venv/bin/python -m pytest -q -p no:cacheprovider -rs tests
# 2494 passed, 37 skipped, 6 xfailed, 0 failed in 3182 s -> PASS
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
  .venv/bin/python -m pytest -q -p no:cacheprovider -rs tests
# 2494 passed, 37 skipped, 6 xfailed, 0 failed in 3017 s -> PASS (skip set identical to the CUDA-hidden run)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_vector_env_*.py tests/test_rl_*.py tests/test_perf_acceptance.py \
  tests/test_acceptance_frozen.py
# 392 passed, 13 skipped in 808 s -> PASS
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaaclab-venv/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_rl_lab_*.py tests/test_vector_env_contract.py
# 70 passed in 500 s -> PASS (includes the finding 3 and 4 Lab Warp-CPU tests)
```

**After the window re-declaration and the follow-up review fixes.** The runs below cover the changed and dependent
test files only, at docs commit `f91145f`. The code is identical to `259a9e2` plus the docs, and includes the merge of
`d3d9dee`.

```bash
SEL="tests/test_acceptance_frozen.py tests/test_perf_acceptance.py tests/test_rl_lab_fidelity.py \
  tests/test_rl_rollout_roles.py tests/test_rl_smoke.py tests/test_rl_supervision.py tests/test_rl_termination.py \
  tests/test_rl_lab_*.py tests/test_vector_env_*.py tests/test_repo_naming.py tests/test_batched_lab_coupling.py \
  tests/test_long_rollout_config.py tests/test_rl_stability.py tests/test_rl_consolidation_cpu.py \
  tests/test_rl_configured_family.py tests/test_rl_position_band.py"   # as listed in the run logs
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider -rs $SEL
# 290 passed, 10 skipped in 67 s -> PASS
flock .../gpu0.lock env CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -p no:cacheprovider -rs $SEL
# 290 passed, 10 skipped in 68 s -> PASS (identical skip set)
# Torch venv: test_perf_acceptance, test_acceptance_frozen, test_rl_rollout_roles, test_rl_smoke, test_vector_env_*,
#   test_rl_lab_fidelity, test_rl_supervision, test_rl_termination
# 241 passed, 3 skipped in 533 s -> PASS (NOT_RUN: CUDA stream test; workload B contract needs the Lab venv)
# Isaac Lab venv: test_rl_lab_*.py, test_vector_env_contract.py, test_batched_lab_coupling.py
# 72 passed in 301 s -> PASS
```

The full suites above were not re-run after the re-declaration. GitHub CI on the branch head runs the full suite in the
CI environment.

**What was skipped (NOT_RUN):**
- `.venv`, 37 skips: the Torch and Isaac Lab modules, plus WP-12 records that are not linked here. The known WP-12
  failure does not appear, because `artifacts/wp12` is not linked in this worktree.
- Torch venv, 13 skips: Isaac Lab / pxr modules, the WP-12 Demo 1 recording, and the CUDA stream test.

**Simulated CI.** This worktree has no `artifacts/`, CUDA was hidden, and torch, Isaac Lab, MuJoCo and the Newton USD
importer looked absent. The run used the project `.venv` with a pytest plugin that hides those packages. Results are in
§8.

## 8. GitHub CI

**What CI lacks.** `.github/workflows/tests.yml` runs the full suite without `artifacts/`, torch, Isaac Lab, MuJoCo,
the Newton USD importer and CUDA, but with pxr.

**Changes in this branch:**
- `origin/claude/rl-phase1` is merged with a normal merge commit. It brings the module-level NOT_RUN skips of the
  robot-batching USD tests (`cc0e24c`, `63c1cfb`) and the Phase 1 doc fixes.
- The workflow gains a concurrency group per workflow and ref. A newer push to the same ref cancels the older run,
  except on `main`: `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}` (follow-up review finding 5), so every
  merge commit on `main` keeps a completed run.

**My new and changed tests and their local-only inputs:**
- The Torch test is gated by `importorskip("torch")`.
- The Lab tests need Isaac Lab and skip with NOT_RUN without it.
- The judge reads the real WP-17 tree only through its CLI test, which expects BLOCKED when the tree is absent.
- Every other new test needs only committed files.

**CI simulated locally.** The test tree is the merge of `2195bb2` with `claude/rl-phase1`; it has the same code as the
branch head. It was a detached temporary worktree, `/mnt/data/ningan/tmp/rcs_p1fu_ci`, with no `artifacts/`. The
project `.venv` ran with CUDA hidden. A pytest plugin made torch, Isaac Lab, MuJoCo / MuJoCo-Warp and
`newton_usd_schemas` look uninstalled. The plugin is copied to `artifacts/rl_ready/p1_followup/ci_block_imports.py`.

| Run | Result |
| --- | --- |
| new and changed test files, 16 modules | 225 passed, 23 skipped, 0 failed; every skip states NOT_RUN or the missing module |
| full suite | **2465 passed, 58 skipped, 6 xfailed, 0 failed in 2939 s** (`ci_sim_full_002`) |

`ci_sim_full_001` is preserved and is not a code result. It is a collection error caused by my first import blocker:
`find_spec` raised for `mujoco` where an uninstalled package returns None.

## 9. Open

- **A full acceptance under `acceptance_v2` is NOT_RUN.** It needs a candidate. Its real GPU time is also unmeasured; the
  estimate is in §1.2.
- **The adapter does not pass `--config` to the harness.** The harness records the config only when it is given.
  Selecting v2 in the adapter is the other agent's change (read-only here).
- **`G9:window` is measured on the episode processes, not inside the G9 throughput processes.** Per-phase walls,
  VRAM and reset p95 are candidate-reported. The contact-phase cost factor is measured at N=2 only.
- **The statistical trade-off (§1.2) is a PR limitation:** N=8 is not gated, and the N=1 floor rests on 3 pairs of 172
  steps.
- **The cutting-onset step-wall spikes (2.4–3.0×) are unexplained.** They are now timed.
- **`rcs_rl_rollout_v2` is opt-in.** Declared runs (`rl_smoke_v1`, long rollouts) keep v1.
- **Not addressed here:** Phase 1 review findings 5–9 and 11 (not in scope). The progress file's U2 pointer
  (follow-up review finding 7) is the coordinator's.
