# Phase 1 follow-up: `acceptance_v2` and review findings 3, 4, 10, 12

Date 2026-11-10. Branch `claude/rl-p1-followup`, from `c5e96b2`, whose code is identical to the reviewed `f03fbff`.
Source: `docs/reports/rl_ready/PHASE1_ACCEPTANCE_REVIEW.md` (findings 1, 2, 3, 4, 10 and 12, §7, §8). Decision: D-14 in
`docs/RL_READY_PLAN.md`. Raw outputs, not committed: `artifacts/rl_ready/p1_followup/`. Compact evidence:
`docs/evidence/rl_ready/p1_followup_001.json`.

## Status

| Item | Status | Where |
| --- | --- | --- |
| Finding 1: G9 times tool–material contact (`acceptance_v2`) | DONE. Config declared before any run. The GPU contact check PASSed (`SHARED_GPU`). A full acceptance under v2 is NOT_RUN. | `configs/rl_benchmark/acceptance_v2.json`, judge `G9:window` |
| Finding 2: frozen contract, pin tests | DONE. `acceptance_v1` is pinned at `3592d940…`. `acceptance_v2` is pinned at `a008dd38…` and freezes at its merge into `main`. | `tests/test_acceptance_frozen.py` |
| Harness and judge config path | DONE. The judge and the local runner already took a required `--config`; the harness gains an optional `--config` (provenance only, default unchanged). v1 is judged exactly as before (tested). | `scripts/rl_ready/{run,judge}_perf_acceptance.py` |
| Finding 3: workload-B `fidelity()` and the time step | DONE (CPU and Lab Warp-CPU tests) | `rl/lab_vector_env.py` `lab_fidelity` |
| Finding 4: failed B2 settle latched in non-strict mode | DONE (Lab Warp-CPU test with real failing gates) | `rl/lab_batched_env.py` `reset_envs` |
| Finding 10: caller errors and the contact-pool overflow are infrastructure | DONE (CPU tests). No new class. | `rl/supervision.py`, `rl/termination.py` |
| Finding 12: rollout files label privileged fields | DONE as an opt-in `rcs_rl_rollout_v2`; v1 stays readable. The env metadata labels the privileged info keys. | `rl/rollout.py`, `check_rl_rollout.py`, `rl/vector_contract.py` |
| D-14 and this report | DONE | |

## 1. `acceptance_v2` (findings 1 and 2)

**Unchanged from v1:** the schema `rcs_perf_acceptance_v1` (so the external adapter's reader works), harness, judge,
evaluator file set, reference variant `alias_free_v1`, seeds, the physics plan (G4, G6, G7, G8), every threshold except
`primary_pairs_above_1`, the physics tolerances, the frozen paths, the dev limits and the timeouts. A CPU test asserts
this key by key. `version` is `acceptance_v2`, and the adapter refuses a file whose name and version disagree.

### 1.1 The window: steps 321–400 of the s512 replay from reset, at every N

Data are from the WP-17 s512 and s15 replays. The source is the per-step contact counts and Native-sample codes of the
P1-6 R0 `smoke_003` G6 dumps, which are the same for `frozen_v1` and `alias_free_v1`:

| | s512 (timed) | s15 (held out, not timed) |
| --- | --- | --- |
| labelling (cutting) | steps 251–380, 0 → 37,958 labelled (21,370 at step 320) | steps 251–530, 0 → 85,786 |
| particle contacts non-zero | steps 315–364 (49 steps, total 152,455) | steps 322–846 |
| in the v2 window 321–400 | 44 contact steps, 143,022 contacts (93.8 % of the episode total), 60 cutting steps, 16,588 new labels | 79 contact steps |
| in the v1 window at N=8/16 (21–120) | 0 contacts, 0 labels | 0 |

`warmup_steps` 320 and `steps` 80 apply at every N, so every N times the same phase.

Why these steps:
- The first contact steps (315–320) stay in the untimed warm-up, so one-time first-contact work is not timed (for
  example contact-pool growth or lazy allocations).
- The approach (steps 1–250) is untimed.
- A later start would lose contact steps, because s512 contacts end at step 364.
- 80 steps is the minimum length asked for. The last 20 steps (381–400) come after the cut.

`G9:window` (new, `performance.window_check`) makes the claim checkable in every acceptance:
- It reads the G6 and G7 episode dumps of the same variants, which replay s512 from reset and record per-step
  contacts and the Native-sample codes.
- Every s512 env, on both sides, must show at least 40 contact steps and at least 10,000 new labels inside the window.
- Candidate below while the baseline meets it: FAIL. Baseline below: BLOCKED. No dumps: BLOCKED.
- Limitation, written into every verdict: this is measured on the episode-mode processes, not inside the G9 throughput
  processes.

`performance.phases` (new, descriptive) reports the per-phase median step wall per side, and their ratio. The source is
the harness's new per-round `step_walls_s`, which are in-process and candidate-reported. The phases are approach 21–250,
cutting before contact 251–314, onset 315–320, the timed window and its contact and post-contact parts. This answers the
review's "report the per-phase speed".

### 1.2 Statistics and budget, set before any run

**Kept from v1:**
- ABBA pairs inside one lock hold per N, on the same GPU;
- the caller's outer clock;
- N=16 as the primary N;
- the every-N floor (median ≥ 0.97);
- the 5 % clock tolerance.

**Changed:**
- Pairs go from 5 to 3 (`BCCBBC`). The primary rule is now median ≥ 1.05 with 3 of 3 pairs > 1. Under no speed change,
  the one-sided sign-test false-accept rate is 1/8. That is stricter than v1's 4 of 5 (3/16).
- N goes from {1, 8, 16} to {1, 16}. N=8 is dropped because N=8 and N=16 are both saturated: aggregate env-steps/s is
  flat from N=4 on. N=1 is the only unsaturated point and costs 5 % of G9.

A contact window needs at least 314 untimed replay steps per process. That is why the budget forces these cuts.

The estimate was made before any run. It uses the EXCLUSIVE workload-A rows of `alias_free_v1` (the reference variant).
Their per-step medians are 0.098 s at N=1, 0.571 s at N=8 and 1.180 s at N=16, all pre-contact, plus 13–15 s of
overhead per process.

| Design | G9 GPU time | Largest G9 lock hold | One acceptance (exclusive) | At ×1.3 / ×1.5 sharing |
| --- | --- | --- | --- | --- |
| v1 statistics with this window (5 pairs, N 1/8/16) | 130 min | 81 min (N=16) | 2.9 h | 3.8 / 4.3 h |
| 3 pairs, N 1/8/16 | 78 min | 49 min | 2.04 h | 2.65 / 3.1 h |
| **v2: 3 pairs, N 1/16** | **54 min** | **49 min** | **1.64 h** | **2.13 / 2.46 h** |

The physics plan, unchanged, adds 44.5 min:

| Gate | Runs | Time |
| --- | --- | --- |
| G6 | 6 × 261 s | 26 min |
| G7 | 2 × 362 s | 12 min |
| G4 | 4 × 57 s | 4 min |
| G8 | 2 × 75 s | 2.5 min |

**Revision from the GPU check (§2), not a change to the config.** `alias_free_v1`'s contact-phase step is about 1.33×
its approach step. A 400-step G9 run therefore costs about 6 % more than the pre-contact estimate. The s15 episodes have
a long contact phase, so the physics plan costs about 8 % more.

| | Before the GPU check | Revised |
| --- | --- | --- |
| One acceptance, exclusive | 1.64 GPU-h | about 1.75 GPU-h |
| Largest sharing factor that fits 2.5 h | ×1.5 | about ×1.43 |
| Largest lock hold (G9 N=16), exclusive | 49 min | about 52 min |
| Largest sharing factor for a ≤ 90-min hold | ×1.85 | about ×1.75 |

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
- **`baseline_measure` uses the v2 window:** throughput at N 1/8/16 and a profile at N=8, all with W=320 and K=80.

### 1.4 Freeze (finding 2)

`tests/test_acceptance_frozen.py` pins:
- `acceptance_v1.json`: `3592d9408f314c3d2825171361b8ec640534c60577bb6216a10c4f2b22897bf4`;
- `acceptance_v2.json`: `a008dd3811da917190972a67a15fea713e17969fd442172cd68efe4730da5af4`;
- both band files.

It also checks that each config's version equals its file name, and that every `acceptance_v*.json` is pinned.

`acceptance_v2` freezes at the commit that merges it into `main`. Any change after that is `acceptance_v3`.

The "frozen from its first commit" wording was corrected in D-11 and in the R0 report on `claude/rl-phase1`
(`0b99e7b`), which is merged into this branch. This branch does not edit either file.

### 1.5 External adapter compatibility (read-only check)

I imported the adapter's own `core.config_problems` read-only, with no bytecode written. It returns no problem for v1 or
v2. Its acceptance driver:
- reads a single `performance.warmup_steps` and `steps_by_num_envs`, both of which v2 keeps;
- reads `order` and `pairs` for every N;
- passes no `--config` to the harness, which is why the harness flag is optional.

## 2. GPU contact check (GPU0, `SHARED_GPU`)

```bash
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
  WARP_CACHE_PATH=/mnt/data/ningan/tmp/warp_cache_p1fu PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  .deps/isaacsim-venv/bin/python scripts/rl_ready/run_perf_acceptance.py \
  --variant-json artifacts/rl_ready/p1_followup/variants/alias_free_v1.json --mode episode --num-envs 2 --seeds 512 \
  --warmup-steps 320 --steps 100 --device cuda:0 --gpu-index 0 --config configs/rl_benchmark/acceptance_v2.json \
  --out artifacts/rl_ready/p1_followup/gpu_window_check_001
```

**Run setup:**

| Item | Value |
| --- | --- |
| Commit | `cf04c1a`, clean tree |
| Process | one harness process, one child round, wall 138 s (build 22 s with a fresh Warp kernel cache), 21:43–21:46 |
| GPU | RTX 5090, driver 570.211.01 |
| Software | Warp 1.17.0, Torch 2.11.0+cu128 |
| Workload | N=2, 102,732 particles per env, 50 Hz, 0.1 ms material dt, 200 substeps, no cameras, no recording, warm |
| Co-tenants | foreign SM mean 31.5 %, max 71 %; 4 foreign compute PIDs, among them 642659, which was not touched |
| Own VRAM peak | 1,014 MiB |

**Result: PASS.**
- In both envs, particle contacts are non-zero at steps 315–364 (49 steps).
- Inside the window 321–400: 44 contact steps, with 143,063 and 143,110 contacts.
- Labelled particles go from 21,371 / 21,370 at step 320 to 37,958 at step 400, so 16,587 / 16,588 new labels.
- `invalid_reason` is 0 in both envs.
- This is the same profile as the N=1 recordings, and above the `G9:window` thresholds (40 contact steps, 10,000 labels).

**In-process per-phase median step walls** (N=2, `SHARED_GPU`, candidate-reported):

| Phase | Steps | Median step wall |
| --- | --- | --- |
| approach | 21–250 | 0.245 s |
| cutting before contact | 251–314 | 0.246 s |
| onset | 315–320 | 0.274 s |
| timed window | 321–400 | 0.295 s |
| timed contact | 321–364 | 0.326 s |
| after contact | 365–400 | 0.211 s |

For `alias_free_v1`, the contact-phase step costs about 1.33× an approach step. So v1's pre-contact window did not
represent the cutting phase for this variant. That is review finding 1, now measured.

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

**Env side:** `vector_contract.PRIVILEGED_INFO_KEYS` (the oracle-derived step info keys) is published as
`env.metadata["privileged_info_keys"]` on both backends, and the shared contract checks it.

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
- The workflow gains a concurrency group per workflow and ref, with `cancel-in-progress`. A newer push to the same ref
  cancels the older run.

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
  VRAM and reset p95 are candidate-reported.
- **`rcs_rl_rollout_v2` is opt-in.** Declared runs (`rl_smoke_v1`, long rollouts) keep v1.
- **Not addressed here:** findings 5–9 and 11 (not in scope).
