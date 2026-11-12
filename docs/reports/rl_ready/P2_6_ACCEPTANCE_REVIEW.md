# P2-6 — xhigh acceptance review (P6.7, 2026-11-12)

**Question.** Should `claude/rl-rubber-p26` @ **145777d** be merged into `claude/rl-rubber` (@ `2a7cb77`, an
ancestor of 145777d)? p26 contains:

- P2-6 itself;
- the Native host branch `claude/rl-rubber-p26n` (merged at `7404816` and `45cdc66`; head `14435ee`), reviewed here
  for the first time;
- `origin/main` @ `bfe6d5e` (merged at `e589b65`).

**Verdict: GO_WITH_CONDITIONS.**

- **Protocol integrity: PASS.**
  - Every declaration (ENV1, ENV2 v1 / v2, P2-6 v1 / v2 / v3 / v4) has exactly one commit, and that commit holds only
    the declaration and its pin test.
  - Each one was committed before its run. Supersession scope holds: v3 and v4 change RL-05 only, and every other
    test is byte-equal to v2.
  - Formal seeds are fresh, and the development seeds are recorded and disjoint.
  - Every FAIL is preserved: ENV2 v2 at μ 0.4, RL-05 v2, RL-05 v3, the DRAG recording bug and the smoke patch bug.
  - No sign of tuning on formal seeds beyond the declared final-attempt rule was found (§1).
- **Correctness: PASS_WITH_LIMITATIONS.**
  - Batched and single runs agree: byte-wise on the CPU, and within the L3 band on the GPU.
  - Isolation and masked resets are byte-wise on the CPU. Randomisation is per env. The policy observation and the
    scripted controllers are deployable.
  - The dispatch leaves Phase 1 unchanged. The Native host feeds the measured pose and touches no frozen module.
  - The RL-09 statistics are sound.
  - One new defect is MEDIUM (F1): host memory grows without bound on resets. It was measured in this review.
  - The other findings are LOW contract or labelling divergences (§2, §7).
- **Merge of main: PASS.** `vector_env.py` is main's file plus the 13-line dispatch. No other Phase 1 path, config or
  script differs from `bfe6d5e` (§3).
- **Honesty: PASS.**
  - 24 numbers were recomputed from the raw artifacts and all match, except one off-by-one in a FAIL narrative (§4).
  - The limitations are stated: friction 0.3, no robot coupling, throughput, BLAS, SHARED_GPU, uncalibrated.
  - Nothing claims calibration or a trained policy.
- **Tests: PASS.**
  - Fresh worktree, CUDA hidden: `.venv` 121 passed, 7 skipped (Torch-gated NOT_RUN); Isaac Lab Torch venv
    **167 passed, 0 skipped** (§5).
  - GitHub CI on the code-identical `73c624b` is green. CI on 145777d was still in progress at the time of review.

**Material status.** SIMULATOR_ONLY_UNCALIBRATED everywhere: receipts, the family, the evaluator labels, the report and
the ADR.

**How the review was done.**

- **Read-only.** No branch was modified, nothing was merged, and no link target was written.
- **Fresh worktree.** Detached `RoboticCuttingSim-rl-p26review` at 145777d, with symlinks for `.venv`, `.deps` and
  `artifacts/{rl_ready,wp17,workcell_asset_cache}`. It was removed afterwards.
- **Trial merge.** A throwaway worktree at 145777d ran `git merge --no-commit origin/claude/rl-rubber-p25` (0b9a8d8).
  It was aborted and removed.
- **GPU.** One DIAGNOSTIC spot-check on GPU0, under the blocking `rcs_rl_runner/gpu0.lock` with 30.9 GB free (§6).
  The co-tenant PID 642659 (766 MiB, 0 % util) was not touched.

**Inputs read.**

- The plan's P2-6 and "Phase 2 Done" sections, `round_prompt.md` and AGENTS.md.
- `P2_6_RUBBER_RL.md` (all sections) and ADR 0023 §23 / §23.6.
- The declarations `rubber_rl_tests_v1..v4`, `rubber_strip_drag_v1/v2`, `contact_envelope_v1/v3` and
  `rubber_contact_envelope_v1/v2`.
- The code: `rl/rubber_{family,env,contract,scripted}.py`, the dispatch in `rl/vector_env.py`,
  `batched/rubber_material_batch.py` (P2-6 diff), `native_rubber_host.py`, `rubber_native_compare.py`, `rubber_l3.py`,
  `rl/termination.py` / `vector_contract.py` (the Phase 1 contract), all of `scripts/rubber/p2_6/` and the P2-6 tests.
- The evidence `docs/evidence/rl_ready/p2_6_*.json`.
- The raw artifacts under `/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/p2_6/`: every run meta and row,
  `rl09.json`, the replay metas, the development runs and the RL-11 logs.

## 1. Protocol integrity — PASS

### 1.1 Order of events (KST; commit times are committer dates, run times are from the run metas)

| # | Event | Time | Check |
| --- | --- | --- | --- |
| 1 | ENV1 envelope + loader (`887c08a`) | 11-11 19:18:32 | data from formal rows only |
| 2 | ENV2 law-v3 protocol declared (`c208864`) | 19:19:43 | 1 config + its pin test; only commit of the file |
| 3 | `contact_envelope_v1_001` at `a79f7db`, clean | 19:22:16 – 19:38:40 | PASS |
| 4 | P2-6 v1 pair declared (`130ccd4`) | 19:40:14 | superseded by K2-1; never run formally; kept pinned |
| 5 | ENV2 law-v2 protocol declared (`08b9dcd`) | 19:51:10 | only commit of the file |
| 6 | `contact_envelope_v2_001` at `40ad8d4`, clean | 19:53:06 – 20:03:28 | **FAIL at μ 0.4**: STRIP n_t 4 D04 0.948, n_t 6 D04 0.929 / slip 1.43 % (recomputed) |
| 7 | DRAG rerun `contact_envelope_v2_002_drag` (`40f40e2`), labelled | 20:04:37 – 20:07:14 | the formal row's recording bug is kept |
| 8 | `contact_envelope_v3` recorded (`8054c9e`) | 20:07:38 | v2 sliding 0.3 / 0.4 / 0.3 → P2-6 range [0.3, 0.3] |
| 9 | **P2-6 v2 pair declared** (`400f5c5`) | 20:08:16 | 2 configs + pin test; only commit of each file |
| 10 | `gpu_rl_a_001` at `a95da89`, clean (RL-01/02b/03/04b/05 v2) | 20:24:01 – 21:09:43 | RL-05 v2 **FAIL** preserved |
| 11 | `native_rubber_001` at `1615285` (contains `400f5c5`), clean | 21:08:15 | protocol / config bytes at `1615285` = pins |
| 12 | `gpu_rl_c/b/d_001` at `3807912`, clean | 21:09:45 / 21:11:20 / 22:06:42 | |
| 13 | RL-09 replays at `5e64347` (contains `400f5c5`), clean, GPU1 | 23:41:40 | replay path byte-identical to `1615285` (only `KitRubberScene` changed) |
| 14 | RL-09 compare, evidence (`4093f49`) | 11-12 01:18 / 01:19:42 | |
| 15 | v3 development, CPU stand-in, seeds 5000–5011 | 00:56 – 03:00 | `diag_rl05_dev_001..005`; only seeds 5000–5011 appear |
| 16 | **v3 declared** (`ba41461`), runner `ae38c9e` | 03:01:59 / 03:03:57 | |
| 17 | `gpu_rl05_v3_001` at `ae38c9e`, seeds 7000 + e | 03:35:47 | **FAIL** (8 / 8 better, 0 successes) preserved |
| 18 | v4 development, full model, GPU: seeds 5000–5003 (`45cdc66`), 5004–5007 (`d92550d`) | 04:46 – 05:08, 06:17 – 06:57 | metas list exactly these seeds |
| 19 | **v4 declared** (`386cfbc`), runner pin `e0bce4e` | 06:58:07 / 06:58:16 | `final_attempt`: no v5 |
| 20 | `gpu_rl05_v4_001` at `e0bce4e`, clean, seeds 8000–8007 | 06:58:18 – 07:36:35 | **PASS**, 2 s after the pin |
| 21 | RL-06 re-measure at `8b8e905`, `OPENBLAS_NUM_THREADS=1` | 07:37:09 – 09:12:08 | |
| 22 | pins (`73c624b`), report and evidence (`145777d`) | 07:37:34 / 10:12:21 | `73c624b` → 145777d: 3 docs files only |

Further checks:

- **Single commit per declaration.** `git log --format=%h -- <file>` shows exactly one commit for each of
  `rubber_rl_tests_v1..v4`, `rubber_strip_drag_v1/v2`, `rubber_contact_envelope_v2` and `contact_envelope_v3`.
- **Supersession scope.** v3 and v4 have their non-RL-05 tests JSON-equal to v2 (recomputed). The RL-05 `pass` text
  is verbatim across v1–v4, and the success definition (10 mm for 25 steps) never changed. v1 → v2 changed only the
  contact law and friction, as `test_v2_declarations_after_k2_1` asserts.
- **Formal seeds never seen before their run.** A search of every `seeds` / `env_seeds` / `seed` field under
  `p2_6/` finds 7000–7007 only in `gpu_rl05_v3_001` and 8000–8007 only in `gpu_rl05_v4_001`.
  - The v4 GPU budget adds up to the two recorded batches (22 + 40 min ≈ 1.0 GPU-h). That is consistent with no
    unrecorded development run.
  - The dev harness refuses seeds below 5000 but does not refuse 7000+ / 8000+. Nothing was found that used them.
- **Final-attempt rule.** The rule was declared only at v4, after two FAILs. The criterion and the success definition
  stayed unchanged, the seeds are fresh, and the v4 margin is not marginal:
  - 7 / 8 envs better than the zero action;
  - 5 successes, against 2 / 4 in development;
  - returns −0.7 … −30 against −1.7 … −57.

  The pass is not a lucky draw. It shows that a deployable-observation controller can solve the task on fresh seeds,
  which is what the RL-05 sanity check is for. The report says it passed "only with the third declared controller".
- **Errata are additive.** Examples: `p2_6_rubber_rl_001_note_001` for the RL-07 cost-arm label, report §6 1–6, and
  §9.4. Nothing was re-scored.
- **Writer refusal works.** `gpu_rl_d_001.log` is a second queued attempt refused with "refusing to overwrite".

## 2. Correctness — PASS_WITH_LIMITATIONS

### 2.1 Batched vs single parity

- **L0 (CPU).** `test_rl02a_l0_n1` and `test_rl02a_l0_n3_heterogeneous_with_partial_reset` drive single
  `LiveRubberMaterial` references with the batch's own staged q1 / b1 / traction (`drive_single_reference`).
  - Checked byte-wise after every step: every solver / contact / visco / S1 array, markers, marker velocities and the
    invalid bits. This includes after a masked reset of env 1.
  - The CPU geometry is the labelled `cpu_parity_v1` slots, not the formal slots.
- **L3 (GPU).** `gpu_rl_d_001`: N 8 batch, envs 0 and 1 against 20 single GPU repeats each.
  - 0 / 90 statistics outside; max M/U **0.7743** (env0.marker0_z); held-out 0 / 10 outside (recomputed).
  - Only 2 of the 8 envs are scored, as declared.
- **Reset path.** `projection_assembly_v2` (`rubber_material_batch.py:943-994`) is byte-identical to
  `tlmpm_rubber_v1.projection_blocks`:
  - same product order `(m·wa)·wb`;
  - `np.add.at` keeps the loop order;
  - `tests/test_rubber_projection_assembly.py` covers 5 scenes, including L125_t4 and L150_t6, at 3 densities.

### 2.2 Isolation and partial reset

- **CPU.**
  - `test_rl04a…`: envs 0 / 2 are byte-identical when env 1's seed and actions change. A masked reset equals a fresh
    env byte-wise (except the episode counter) and evolves identically.
  - `test_rl03_partial_resample_changes_only_that_env` holds.
  - `reset_masked` resets particles, nodes, groups, traction, env rows, shadows, projection rows, visco registry state,
    contact and S1. The tests above cover every one of them.
- **GPU.**
  - RL-04 b (−09 v2 rule, λ 4) at **N 4** passes: 3,780 scalars, worst d / limit 0.469 (recomputed).
  - RL-06: 53 partial resets at N 8, finite, VRAM growth 0.
  - No GPU row scores partial-reset isolation directly (RL-04 b uses full resets). The masked-reset kernels are the
    ones the CPU byte-wise tests cover, so this is a coverage note (F6), not a defect.

### 2.3 Randomisation per env

- Per reset, each env draws μ0, ρ, Prony g / τ scale, tip traction (magnitude and direction), preload (sweep, yaw)
  and goal from `np.random.default_rng(seed)` in the declared order (`rubber_family.py:268-289`).
- Tests:
  - two envs reset with identical seeds are byte-identical;
  - the KS test passes for the 11 non-degenerate draws;
  - three heterogeneous parameter rows are present;
  - every value is in range and obeys the batch step rule.
- **Not randomised per reset:**
  - friction is fixed at 0.3 (declared limitation);
  - length / thickness are drawn per env at construction (`slot_seed`) and are fixed for that env's lifetime. A slot
    change on reset is refused (`UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET`), and `rebuild()` is the only route.
    `rebuild()` has no test in the suite; my ad-hoc CPU check passed (§5).
  - `reset(seeds=None)` reuses the env's seed, so the next episode repeats the identical draws.

  Report §7 does not state the last two points (F5).

### 2.4 Termination / reward / done against the Phase 1 `open_vector_env` contract

**Equivalent to Phase 1:**

- `termination_reason` bits: rubber bits 1 / 2 / 4 / 8 → `TERM_MATERIAL_INVALID`, sticky; `TERM_TRUNCATED` at the
  horizon.
- `terminated = reason & TERMINAL_BITS`, `truncated = reason & TERM_TRUNCATED`.
- The infrastructure guard and `VectorEnvInfrastructureError`.
- Strict-mode raise on an invalid env.
- `seed=` / `seeds=` reset semantics.
- Success comes only from `evaluate()`; the reward is labelled PROVISIONAL.

**Contract points:**

- `terminal_capture` and the evaluation stream are not needed: only `autoreset=False` is supported, and
  `rubber_tracking_eval_v1` keeps device counters (`within_streak`) every step.
- The reward was recomputed on the host to 1e-12 every step, including out-of-box actions; truncation at the horizon
  is exact (`test_rl05_reward_host_recomputation_and_truncation`).

**Divergences (all LOW, §7):**

- privileged info-key labelling (F2);
- the class of a strict-mode invalid rubber env (F3);
- Phase 1 option kwargs raise `TypeError` instead of `UNSUPPORTED_OPTION` (F4).

### 2.5 Deployable observation and scripted controllers

- **Policy observation.** `_observe` (`rubber_env.py:389-410`) builds `policy` from:
  - the commanded grip pose and twist;
  - time remaining;
  - the previous action;
  - the 9 ideal-tracked markers and their velocities;
  - the goal;
  - the preload flag.

  It holds no force, wrench, material parameter, J, energy or contact term. Those are only in the opt-in `critic`.
- **Labelling.** Markers are cross-section centroids from the material state, labelled `ideal_marker_tracking`. That
  is kinematic, not oracle material state; the label carries the perception gap.
- **Controllers.** All three (`rubber_scripted.py`) read only `policy` slices: grip pose 0:7, markers 20:47, the tip
  velocity 71:73, the goal 74:77 and the preload flag 77. The runners pass `obs["policy"]` only.

### 2.6 Dispatch

- The branch (`vector_env.py:1218-1230`) fires only for `backend == "kinematic_grip_batched"` or a task with
  `is_rubber_task_family`. No Phase 1 task class defines that attribute or a `__getattr__`.
- Phase 1 behaviour is unchanged:
  - `test_default_backend_path_is_unchanged_for_cutting_tasks`;
  - the Phase 1 vector-env suites (api / consolidation / contract for both backends in the Isaac Lab venv / p12b) all
    pass.

### 2.7 Native host (first review of `claude/rl-rubber-p26n`)

**Loop.** `run_bridge_episode` (`native_rubber_host.py:260-308`) runs, per step:

1. kinematic target via `set_kinematic_targets`;
2. 10 PhysX steps;
3. `get_world_poses` (float32 → float64 via `pose_from_wxyz`);
4. strip frame (pure translation);
5. `advance_measured`;
6. surface update.

The material therefore receives the measured pose. With a fake PhysX offset, the test checks that it is the measured
pose and not the command.

**Frozen modules.**

- `isaac_sim_runner.py` and `live_material.py` are byte-identical to `2a7cb77` and to `bfe6d5e`. The Native meta
  hashes them as 7ed74a61… and 58466e5a…, which equal the head.
- The host only imports `launch_app`, `wait_for_workcell_stage`, `numpy_data` and `pose_from_wxyz`; it patches
  nothing.
- The physics modules are equal at the head and in the Native meta: `tlmpm_rubber_v1`, `rubber_contact`,
  `tlmpm_rubber_contact_v1`, `tlmpm_rubber_visco_v1`, `rubber_self_penetration` and the constitutive modules.
- `live_rubber_material.py` differs from the meta only by the K2-2 receipt label.

**Scope.** Feedback is off and the clamp is kinematic (labelled). RL-09 therefore validates that the Kit-hosted and the
headless material are the same physics; it does not validate coupled robot physics. Together with RL-02 (batch ≡
single), this closes "same recipe in RL headless and Full Native".

### 2.8 RL-09 statistics

- **Band.** `rubber_l3_band_v4` (unchanged, from P2-2):
  - LOO statistics of 20 replays;
  - a one-sided lognormal prediction bound with t at 1 − α_stat and an inflation of √(1+1/n);
  - Cantelli fallback.
- **Bonferroni.** α_stat = 0.05 / 46 = 1.087e-3 over the whole family.
- **Deterministic branch** (`14435ee`): statistics whose 20 repeats are bitwise identical get U = 0, and a candidate
  must be bitwise equal. The stochastic rest is called with α_family · n_stoch / n_all, so α_stat stays 0.05 / 46.
  - It is stricter than before, and it is unchanged when no statistic is deterministic.
  - Re-run on `native_rubber_001`, it has no effect: there is no deterministic statistic, and the RL-09 result is
    reproduced.
- **Power.** Inputs are bitwise and the bound sits at the GPU-atomics scale (U ≈ 2.8e-7 m for marker1_z). Any
  physics difference above round-off chaos would be rejected.
- **Calibration of the rule.** Held-out 10 / 10 within (max M/U 0.978) is consistent with the nominal error rate.
- **Verdict:** sound.

### 2.9 New defect: unbounded host memory per reset (F1, MEDIUM)

- **Mechanism.** `BatchedRubberMaterial._proj_inv_cache` (`rubber_material_batch.py:467`, used at `:470-480` and
  `:725-729`) caches the F8 block inverses keyed by `(env, float32 mass bytes)`.
  - A reset that resamples density (every reset of the declared task) adds one entry: 1.56 / 2.25 / 3.06 MB for
    t 4 / 5 / 6 mm, computed from `projection_structure`.
  - Nothing evicts entries, and an exact repeat of a continuous density draw is rare.
  - The cache predates P2-6 (P2-3), but P2-6 is the first path to resample on every reset.
- **Measured (§6, GPU0, declared task, N 8).**
  - 48 partial resets with fresh seeds give +48 cache entries (+2.29 MB each) and RSS 2,197 → 2,302 MB, i.e.
    **+2.19 MB per reset**, linear.
  - The CPU parity slots behave the same way (40 resets → 40 entries).
- **Why RL-06 missed it.** RL-06 scores VRAM growth only, and RL-06 / RL-10 never sample host RSS.

## 3. Merge of main — PASS

- `git diff bfe6d5e origin/claude/rl-rubber-p26 -- src/robotic_cutting_sim/rl/vector_env.py configs/rl_benchmark
  scripts/rl_ready` shows only the 13-line dispatch branch (§2.6). `configs/rl_benchmark` and `scripts/rl_ready` are
  empty diffs.
- Every non-docs path in `bfe6d5e..145777d` is an addition (202 files), except `rl/vector_env.py`.
  - The Phase 1 `batched/{material_batch,particle_contacts,constitutive,errors}.py`, `acceptance_v1/v2` and the
    `rl/*` contract files are equal to main.
  - Docs show no modified Phase 1 record against main.
- **Both-sides check.** Against the merge base `db036a0`, the only file changed on both sides is `rl/vector_env.py`.
  The rubber side's entire change there was the dispatch, which the resolution keeps verbatim. Nothing from either
  side was dropped.
- **p26n integrity.** Every file authored on `claude/rl-rubber-p26n` is byte-equal at the head, except the report,
  whose RL-09 clause table is identical.
- **Current main.** `origin/main` is now `62caecc` (PR #14 / #15: `particle_contacts.py` alias_free_v2). It does not
  overlap any p26 file, and main's `vector_env.py` is unchanged since `bfe6d5e`. A later main merge for the Phase 2 PR
  should be clean.
- **Trial merge of p25.** `origin/claude/rl-rubber-p25` (0b9a8d8) onto 145777d has **one conflict**:
  `docs/adr/0023-rubber-material-v1.md`, where both branches add a "§23.6". Everything else auto-merges. Resolve it
  additively: keep both texts and renumber one (C3).

## 4. Honesty — spot-checks against the raw artifacts

| Claim (report / ADR) | Raw value | Source | |
| --- | --- | --- | --- |
| ENV2 v2 FAIL: D04 0.948 (n_t 4), 0.929 / slip 1.43 % (n_t 6) | 0.94831; 0.92920 / 0.014263 | `contact_envelope_v2_001/summary.json` | ✓ |
| RL-02 c 0 / 90, max M/U 0.774 env0 marker0_z | 0.77427, 90 stats, held-out 0 / 10 | `gpu_rl_d_001/rows/RL02c.json` | ✓ |
| RL-04 b 3,780 scalars, worst 0.469 | 3780, 0 failing, 0.46882 | `gpu_rl_a_001/rows/RL04b.json` | ✓ |
| RL-06 53 resets, p95 5.0 s, median 0.40 s, VRAM growth 0 | 53, 5.0037, 0.4010, 0.0 | `gpu_rl_c_001/rows/RL06.json` | ✓ |
| RL-06 re-measure p95 0.093 s, median 0.081 s, `OPENBLAS_NUM_THREADS=1` | 0.09275, 0.08087, meta `blas_threads_env` = 1 | `gpu_rl06_reset_remeasure_001` | ✓ |
| RL-07 N 8 / N 1 gain 1.46, 1,228 MiB | 1.45884, 1228.0 | `gpu_rl_b_001/rows/RL07.json` | ✓ |
| RL-08 dt ΔO1 / L 8.8e-6, ΔO3 / O3 3.2e-5 | 8.8254e-6, 3.2383e-5 | `RL08.json` | ✓ |
| RL-08 b φ 0.91873 / 0.91478 / 0.91367 mm, bias +0.71 % | φ_ext 0.912249 mm → +0.7106 % | `RL08b.json` | ✓ |
| RL-05 v4 7 / 8 better, 5 successes, returns and final distances as listed | identical | `gpu_rl05_v4_001/rows/RL05v4.json` | ✓ |
| RL-05 v3 8 / 8 better, 0 successes | identical | `gpu_rl05_v3_001` | ✓ |
| RL-05 v3 "tip within 0.7–6.5 mm by step 20 in **6** envs" | **5** envs (0.7 / 0.9 / 1.8 / 6.5 / 4.5 mm, envs 3–7) in the recorded every-10-step samples | `RL05v3.json` | nit (F9) |
| RL-09 rl09 sha `11106fca…2ff7` | `11106fcaf9d8…0be2ff7` | `native_rubber_001_rl09/rl09.json` | ✓ |
| RL-09 receipts 30 / 30; inputs 30 / 30 bitwise; 0 / 46; max 0.654 marker1_z; α_stat 1.087e-3; 45 lognormal / 1 Cantelli; held-out 10 / 10 (max 0.978); bridge 7.45e-9 m, 101 updates | identical | `rl09.json` | ✓ |
| RL-10 PASS on six runs | PASS, 0 problems, head `73c624b` | `rl10_provenance_001.json` | ✓ |
| RL-11 suite counts (2,911 / 2,914 / 385 / 386 / 388) | identical | `rl11_cpu_00{1,2}/*.log` | ✓ |

Limitations stated: friction 0.3 (§1, §7, ADR); robot coupling `NOT_IMPLEMENTED_PHASE2`; ≈ 1 env-step/s and no speedup
claim; BLAS user note (§8, `threadpoolctl` absent); SHARED_GPU labels per row; Q1w bias; self-contact detected only;
SIMULATOR_ONLY_UNCALIBRATED. No calibration claim and no trained policy. **Missing from §7:** F1, F5, and the contract
divergences F2–F4 (condition C2).

## 5. Tests (fresh worktree at 145777d, `CUDA_VISIBLE_DEVICES=''`)

| Command | Result |
| --- | --- |
| `OPENBLAS_NUM_THREADS=4 .venv/bin/python -m pytest -p no:cacheprovider -q -rs tests/test_rubber_rl_env.py tests/test_rubber_rl_dispatch.py tests/test_rubber_rl_p6_0_declarations.py tests/test_rubber_scripted.py tests/test_native_rubber_host.py tests/test_native_rubber_compare.py tests/test_rubber_p2_6_frozen_sources.py tests/test_rubber_contact_envelope.py tests/test_rubber_contact_envelope_check.py tests/test_rubber_device_staging.py tests/test_rubber_projection_assembly.py tests/test_rubber_resolution_label.py tests/test_batched_frozen_sources.py tests/test_acceptance_frozen.py tests/test_native_source_freeze.py tests/test_wp12_phase5_frozen.py tests/test_vector_env_contract.py tests/test_vector_env_api.py tests/test_vector_env_consolidation.py tests/test_vector_env_p12b.py tests/test_repo_naming.py tests/test_rubber_p2_3_protocols.py tests/test_rubber_p2_4_protocols.py` | **121 passed, 7 skipped** (13 s). The 7 skips are Torch-gated modules (`test_rubber_rl_env`, `_dispatch`, `test_rubber_scripted`, `test_vector_env_*`): NOT_RUN, not passes |
| `PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaaclab-venv/bin/python -m pytest -p no:cacheprovider -q -rs` on the same selection minus `test_native_source_freeze` / `test_wp12_phase5_frozen` / `test_rubber_p2_{3,4}_protocols` | **167 passed, 0 skipped** (602 s). Includes RL-01 / 02 a / 03 / 04 a / 05 CPU, the dispatch and refusals, the scripted controllers, Native host and compare (with the P6.5 artifacts, so the evidence recomputation ran), the declaration and P2-6 / Phase 1 / acceptance pins, the Phase 1 vector-env contract for both backends, and naming |
| Ad-hoc, Isaac Lab venv, CPU: `RubberVectorEnv.rebuild()` to new slots, then the contract, then comparison with a fresh env | contract PASS; particle state bitwise equal to the fresh env (DIAGNOSTIC; no suite test covers `rebuild`) |
| Ad-hoc, `.venv`, CPU: 40 masked resets of a `cpu_parity_v1` batch | 40 cache entries (F1) |
| GitHub CI | `73c624b` run 43113339930: **success** (2,884 passed, 62 skipped, 6 xfailed). There is no Torch in CI, so every P2-6 env / dispatch / scripted test is NOT_RUN there. 145777d (+3 docs files) run 43132866105 was **in progress** at review time |

## 6. GPU spot-check (DIAGNOSTIC, not a protocol row)

**Command.**

`flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .deps/isaacsim-venv/bin/python <scratchpad>/gpu_spot.py`

**Setup.** 145777d, RTX 5090 GPU0, driver 570.211.01. Window 10:25:01 – 10:26:44 KST. **SHARED_GPU**: PID 642659 was
present at 0 % util. Declared task, N 8, `slot_seed` 0, strict default.

**Results.**

- `check_rubber_vector_env_contract` (12 steps, partial reset of env 1, evaluate): **PASS** in 96 s. Build 3.3 s.
- 48 partial resets with fresh seeds:
  - reset wall 0.043–0.061 s;
  - cache entries 9 → 57;
  - cache 39.8 → 131.5 MB;
  - RSS 2,197 → 2,302 MB (**+2.19 MB / reset**, linear).

## 7. Findings, ranked

| ID | Severity | Finding | Concrete failure scenario | Recommendation |
| --- | --- | --- | --- | --- |
| F1 | **MEDIUM** | Unbounded host-memory growth per reset: `_proj_inv_cache` (`batched/rubber_material_batch.py:467-480`, `:725-729`), +2.2 MB RSS per resampled reset (measured, §6). RL-06 measured VRAM only | RL training on the declared task on this 62 GB shared host. 60-step episodes at the measured rates give ≈ 0.13 GB / h (full fidelity, N 8) to ≈ 0.45 GB / h (`rubber_rl_v1`). A multi-day run, or any future throughput gain, reaches the OOM killer, which can also take down other projects' processes | Bound or drop the cache: an LRU of a few entries, or no caching of resampled masses. Results stay byte-identical (it only memoises). Add host RSS to a declared long-rollout row. Record it as a limitation now |
| F2 | LOW–MEDIUM | The privileged info-key labelling diverges from the Phase 1 contract now on the same branch. The rubber env has no `metadata["privileged_info_keys"]`, and `termination_reason` / `invalid_penalty_applied` (derived from J-domain / CFL / self-contact oracle bits) are unlabelled (`rl/rubber_env.py:229-230, 344, 587-592`; `rl/rubber_contract.py:70`; contrast `rl/vector_contract.py:25-28, 99-101, 124-125`). The policy observation is clean. The declared task config fixes `info_privileged: ["invalid_reason"]` | A generic wrapper (the external agentic optimizer, a rollout writer) that filters info by `env.metadata["privileged_info_keys"]` raises `KeyError` on the rubber env. With a defensive default it passes oracle-derived termination detail to a learner | Versioned change, since `rubber_env.py` is pinned: a rubber contract v2 / env metadata listing `invalid_reason`, `termination_reason` and `invalid_penalty_applied`. Until then, a stated limitation |
| F3 | LOW | A strict-mode invalid rubber env is misclassified. `InvalidRubberEnvs` (`rubber_material_batch.py:113`) is not an `InvalidBatchEnvs`, so `classify_exception` returns `("caller", "InvalidRubberEnvs")` and `classify_process_exit` returns `unknown` (verified). Phase 1 gives `task` / `STRICT_INVALID_ENV` | A P1-4-style runner with the `open_vector_env` default `strict=True` logs a NaN rubber env as a programming error or unknown exit instead of a task outcome. No P2-6 verdict is affected: every GPU row ran `strict=False` | Recognise `InvalidRubberEnvs` in `termination.py` (not frozen), or document |
| F4 | LOW | Phase 1 options on the rubber backend (`evaluation_stream`, `terminal_capture`, `input_guard`, `tool_mount`, `grip_contact_profile`) raise `TypeError: unexpected keyword` (`rubber_env.py:186-189, 714`; verified), not `UNSUPPORTED_OPTION` | A caller that switches backends by catching `UnsupportedRubberConfiguration` / `.code` gets an untyped error. The behaviour is fail-closed, so nothing runs silently | Map unknown kwargs to `UNSUPPORTED_OPTION` in a versioned change, or list it in the report |
| F5 | LOW | Plan coverage is under-stated. Length / thickness are randomised per env at construction only (fixed per env across episodes; `rebuild()` is untested). `reset(seeds=None)` repeats the identical episode (`rubber_env.py:420`). Friction is fixed (stated) | A user expecting the plan's per-reset geometry randomisation, or resetting done envs without new seeds, trains on a fixed geometry per env or on one repeated episode | Report §7 / §8 additive note; a suite test for `rebuild()` |
| F6 | LOW | Evidence coverage. RL-07 (the Phase 2 throughput row) is SHARED_GPU with only start / end samples over 12 min. GPU isolation (RL-04 b) is at N 4 only. No GPU row scores partial-reset isolation. RL-02 c scores 2 of 8 envs. All of this is declared and labelled | A reader of the Phase 2 performance table takes the SHARED rows for acceptance rows, which round rules reserve for EXCLUSIVE windows | Before the Phase 2 PR: rerun RL-07 EXCLUSIVE with a continuous co-tenant sampler, or keep the SHARED label in the table |
| F7 | LOW | Freeze coverage. `batched/rubber_material_batch.py` (`advance_device`, `projection_assembly_v2`, the P2-6 step / reset path) is not in `tests/test_rubber_p2_6_frozen_sources.py` (the receipts do record its sha). RL-10's checker covers the six GPU runs, not `native_rubber_001` / `_replays` or the ENV2 runs. I verified those manually: clean, the commits contain `400f5c5`, the protocol / config bytes equal the pins | A later edit of the batch module (for example the F1 fix) trips no pin | Pin it after the F1 fix, with a declared reason |
| F8 | LOW | GitHub CI has no Torch, so P2-6 RL tests are NOT_RUN in CI (as in Phase 1). Only the local Torch-venv runs evidence them | A regression in `rubber_env` / the dispatch merges with green CI | Keep the local Torch run as a merge gate (it was done here) |
| F9 | NIT | (a) Report §5 / evidence 002 say "6 envs" where the recorded data show 5. (b) `native_rubber_host.py:4-5` docstring and `:40` (`EPISODE["task_config"]`) still say `rubber_rl_env_v1` / law v3 / `rubber_strip_drag_v1`; the runners override this and the receipts are correct. (c) `run_rubber_rl05_v4.py:11` names the v3 script. (d) `tests/test_rubber_rl_env.py:348` ends in `or True` (vacuous). (e) `rl11_cpu_001/commit.txt` says `d92550d` but the directory also holds the `73c624b` GPU1-visible logs; their trailers and the evidence are correct. (f) The frozen v2 config's `resolution_variants` note "v3 envelope row n_t 3: 0.8" is stale | Misreading only | Additive errata; fix code text in the next versioned change |
| F10 | INFO | ADR 0023 §23.6 collides with p25's §23.6 (the trial merge's only conflict) | A merge-time conflict | Keep both; renumber additively |

## 8. Verdict and conditions

**GO_WITH_CONDITIONS** for merging `claude/rl-rubber-p26` @ 145777d into `claude/rl-rubber`.

### Conditions for this merge

- **C1.** GitHub CI on 145777d is green. A failure only in "Install media test tools" is an apt-mirror outage: rerun
  it. The delta from the green, code-identical `73c624b` is 3 docs files.
- **C2.** Additive erratum, with no re-scoring and no edit of the frozen files, in report §7 / §8 and an ADR §23.6
  note:
  - F1 with the measured +2.2 MB per reset, and that RL-06 / RL-10 do not sample host RSS;
  - F5: geometry per env at construction, and `seeds=None` repeats the episode;
  - F2–F4 as known contract divergences from Phase 1;
  - the F9 corrections (the v3 count, and the stale texts as errata).
- **C3.** When p25 and p26 both land on `claude/rl-rubber`, resolve the ADR §23.6 collision additively: keep both
  texts and renumber one, for example P2-6 → §23.7.

### Before the Phase 2 PR to `main` (Z2); tracked, not blocking this merge

- **Z-a.** Fix F1 in a byte-identical, tested change. Then pin `rubber_material_batch.py` (F7), and measure host RSS in
  a declared long-rollout row (a new small declaration; RL-05's "no v5" rule is unaffected).
- **Z-b.** Resolve F2 / F3 / F4 by a versioned contract or env change, or write them in the PR body as limitations.
- **Z-c.** Rerun RL-07 in an EXCLUSIVE_GPU window with a continuous co-tenant sampler, or keep the SHARED_GPU label in
  the Phase 2 performance table (F6).
- **Z-d.** Merge current `main` (62caecc). No p26 file overlaps it.

**Phase 2 Done boxes this supports** (the PR body must keep these limitations):

| Box | Status |
| --- | --- |
| 3 (RL headless ≥ 8 envs + Full Native) | PASS_WITH_LIMITATIONS: friction fixed at 0.3; geometry per env, not per episode; kinematic grip; ≈ 1 env-step/s; F1 open |
| 4 (status) | SIMULATOR_ONLY_UNCALIBRATED |
| 5 (old baselines) | PASS via RL-11 and this review's Phase 1 selections |
