# Phase 2 — xhigh acceptance review (Z1, 2026-11-12)

**Question.** Should the Phase 2 PR be opened and merged from `claude/rl-ready-rubber-physics` @ **92e671a** into
`main` @ `62caecc`? Main is already merged into the branch.

The PR brings:
- Phase 2: rubber, P2-0..P2-6 and the K2 follow-ups;
- the documentation:
  - `README.md`;
  - the draft `docs/reports/rl_ready/FINAL_REPORT.md`, whose §9 "Phase 2 merge and pin" is PENDING by design;
  - `docs/RL_READY_PROGRESS.md`.

**Verdict: GO_WITH_CONDITIONS.** No BLOCKER and no MAJOR finding. The conditions are documentation errata before
the PR is opened, then before the merge the CI gate and, if an idle GPU exists, an `EXCLUSIVE_GPU` RL-07 row (§8).

- **Integration: PASS.**
  - **Files changed.** Against `main`, only four existing files change: `README.md`, `docs/RL_READY_PLAN.md`,
    `docs/RL_READY_PROGRESS.md` and `src/robotic_cutting_sim/rl/vector_env.py`. The last one gains 13 lines, the
    dispatch branch.
  - **Nothing frozen touched.** No existing source, test, config or script is modified or deleted. No frozen file,
    threshold, evaluator, record, `acceptance_v1`/`acceptance_v2` evaluator file or `frozen_paths` entry changes.
  - **No leakage.** No pre-existing module imports a rubber module, apart from the lazy import in the dispatch branch.
  - **Code reviewed is code merged.** 92e671a takes its `src`/`tests`/`configs`/`scripts` tree byte-for-byte from
    e13339b. That tree is identical to c21ae8d, the P2-6 head whose review conditions were checked.
- **Tests: PASS.** Run on 92e671a in a fresh worktree (§3).
  - `.venv` full suite with CUDA hidden: **3,011 passed, 41 skipped (all NOT_RUN-gated), 6 xfailed, 0 failed**.
  - Isaac Lab Torch venv, CUDA hidden, running the RL-11 Torch selection plus workload B, the pins and the naming
    test: **521 passed, 1 skipped (NOT_RUN: CUDA hidden), 0 failed**.
  - **CI.** The push run on 92e671a was **cancelled** by the concurrency rule when the docs-only 03e62d3 was pushed
    30 s later.
    - The code-identical c21ae8d run is **green**: 2,951 passed, 0 failed.
    - The runs on 03e62d3 and e13339b (same code tree) were still in the pytest step at 15:16 KST, after all setup
      steps, including "Install media test tools", had passed.
  - **GPU (DIAGNOSTIC, `EXCLUSIVE_GPU` on GPU1).**
    - The README rubber example runs as written at N=8.
    - The Phase 1 workload A contract at N=2 reproduces the pre-Phase 2 smoke.
- **Phase 2 Done criteria (§1).**
  - Box 1 PASS.
  - Box 2 PARTIAL.
  - Box 3 PASS_WITH_LIMITATIONS.
  - Box 4 PASS: no real data, status SIM_ONLY.
  - Box 5 PASS.
  - Box 6 PENDING: this PR, then the pin.
  - The P2-5 synthetic FAIL and the μ 0.3 friction limit are represented honestly.
- **Docs honesty: PASS_WITH_LIMITATIONS.**
  - Spot-checks: 30 claims (69 values) checked; 68 values match. One count in a FAIL narrative is wrong: RL-05 v2 was
    "better than zero action in 4/8", but the data show **5/8**. The verdict is unaffected.
  - Every recorded FAIL appears in the final report. Nothing claims calibration or a trained policy.
  - Several summary statements are inaccurate or stale (F1–F7). One is in the README API description.
  - **§4.4 disagreement: 1.40× / 1.37× is correct.** The figures 1.45× / 1.42× came from comparing unequal windows.
    The exclusive table's own 1.43–1.49× has the same window bias (§4.2).
- **Hygiene: PASS.**
  - The naming test passes, and none of the 230 commit messages contains the acronym.
  - Size: 317 files, 6.5 MB. The largest file is 252 KB.
  - The two media PNGs are captioned honestly, including the rendered-clamp defect they show.

## 0. How the review was done

**Read-only.**
- No branch was modified and nothing was merged.
- The only write is this file, committed alone.

**Fresh worktree.**
- Detached `RoboticCuttingSim-rl-z1review` at 92e671a.
- Symlinks for `.venv`, `.deps` and `artifacts/{rl_ready,wp17,workcell_asset_cache}`.
- No link target was written: after the runs, nothing in the shared one-shot held-out registry
  (`artifacts/rl_ready/calibration_heldout_registry`) was newer than the start marker.
- The worktree was removed afterwards.

**GPU.**
- One DIAGNOSTIC chain on GPU1, under the blocking `/mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock`.
- 31.9 GB free at start.
- A 5 s sampler saw no foreign process on GPU1.
- PID 642659 on GPU0 was not touched, and the external optimizer's `locks/gpu1.lock` was not taken.

**Inputs read.**
- The plan's P2-0..P2-6 and "Phase 2 Done 判定", `round_prompt.md` and AGENTS.md.
- The five step reviews (their conditions only).
- The step reports where numbers were checked; ADR 0023 §3, §12, §23.3 and §23.7.
- `FINAL_REPORT.md` (all), the README diff and the `RL_READY_PROGRESS.md` diff.
- Code:
  - the dispatch;
  - `rl/rubber_{env,contract,family}.py`;
  - `rubber_calibration_v2.py`;
  - the contact-law defaults;
  - the pin tests;
  - `configs/rl_benchmark/acceptance_v2.json`.
- Raw artifacts:
  - P1-2 `alt_r1_*`;
  - `exclusive_001`;
  - `p2_6/gpu_rl_{a,b}_001`;
  - `rl11_cpu_003`;
  - `rl11_workload_a_contract_n2_001`.

## 1. Phase 2 Done criteria

| # | Criterion (plan) | Verdict | Evidence and reasoning |
| --- | --- | --- | --- |
| 1 | Rubber recipe and hyperelastic / viscoelastic / contact models with real, reproducible run evidence | **PASS** | **What exists.** Recipe schema `rcs_rubber_material_v1` (`configs/materials/rubber_material_v1.schema.json`); recipes `rubber_generic_{soft,visco}_v1`; models `neo_hookean_jbar_v1` and `neo_hookean_jbar_prony_v1`; contact laws `grid_node_penalty_coulomb_v1/v2/v3`. **Protocols.** Every formal protocol is a committed, SHA-pinned config (`configs/rubber/*_tests_v*.json`), checked by the `test_rubber_*protocol*` / `*_declarations` pin tests. **Evidence.** Compact files are in `docs/evidence/rl_ready/p2_*.json`; each carries the commit, a clean-tree flag and the raw-artifact path. **Reproducibility is measured, not asserted.** (a) CPU L0, single vs batched, is bytewise equal (RL-02 a, CAL-02). (b) 30 headless replays reproduce the Native material inputs bitwise (RL-09 b). (c) The CPU suite recomputes the RL-09 evidence from the artifacts (`test_native_rubber_compare.py`) |
| 2 | Large deformation, gravity, stretch / release, rebound, hysteresis and friction pass physically and numerically | **PARTIAL** (agrees with the final report) | **PASS:** gravity sag; cantilever frequency (−0.44 % against Euler–Bernoulli); large deformation (P2-2 v4); stretch / release after the opt-in F8 / F9 fixes (-07 v4); rebound; the Prony tangent gate; per-env viscoelastic reset. **Open, all recorded:** (a) RUBBER-PHYS-11 v5 Q1w FAIL (GCI 2.16 % > 2 %), accepted as a labelled +0.62 % bias. (b) Hysteresis passes only as "v1 FAIL, v2 PASS (post-hoc measurement correction)" (VISC-03). (c) The RL task's law v2 failed -01 / -04 / -05 in `contact_v2_001`, and its envelope fails at μ 0.4, so for the RL task friction is validated only at μ 0.3. (d) Law v3 holds for translational sliding only (K2-1). (e) The -01 d stick-run energy books FAIL. (f) Self-contact is detected, not resolved. (g) The κ = 20 μ model-form error (ADR 0023 §3; F6). The final report's box-2 row omits (b) and (g) (F4) |
| 3 | The same rubber in RCS multi-env RL headless and in Full Native; reliable batch tests with at least 8 envs | **PASS_WITH_LIMITATIONS** | **N=8 GPU rows:** RL-01 contract; RL-02 b L2 (8 slots, 60 steps); RL-03 randomisation (8 slots); RL-05 v4; RL-06 (400 steps, 53 partial resets, VRAM growth 0); RL-07; RL-12 (host RSS flat). **Native RL-09 PASS:** the same recipe, profile and law v2 run in Kit; 0 / 46 statistics fall outside `rubber_l3_band_v4`; inputs are bitwise equal in 30 / 30 replays. This review reran the README example at N=8 on 92e671a (§3.4). **Limitations:** GPU isolation (RL-04 b) ran at N=4, and RL-02 c scores 2 envs (F4); friction is fixed at 0.3; geometry is per env, not per episode; the grip is kinematic, with no robot coupling; ≈ 1 env-step/s; every throughput and Native row is SHARED_GPU (F2) |
| 4 | REAL_CALIBRATED only after an independent held-out pass on real data; otherwise SIM_ONLY | **PASS** (no real data; status SIM_ONLY) | **Status:** `SIMULATOR_ONLY_UNCALIBRATED` everywhere: every receipt, the family, the evaluator labels, the README, the ADR and the report. **The REAL path is closed:** `real_calibrated_gate_v2` refuses synthetic and v1 manifests, and the tripwire `gate_directory_audit` keeps `configs/rubber/calibration/` at `NO_REAL_DATA`. **The P2-5 synthetic FAIL is represented honestly:** CAL-04 FAIL (χ²_red 2.484 > 1.910); CAL-05 FAIL (21 / 31); CAL-06 FAIL (g1 −51 %, g2 +39 %, τ2 −41 %); v3 NOT_RUN; the withdrawn "basin" wording is not reused. **Is the interface ready for real data?** It is ready for ingestion, gating and one-shot held-out bookkeeping. It is not ready to produce a validated calibration, for three reasons: the fit procedure is unvalidated (the report says so); `rubber_calib_tests_real_v1` exists only as an outline (F4 c); and calibration output does not plug into the RL task without further steps (F1) |
| 5 | Old CUT / TRIM / v5 / v7 baselines unbroken | **PASS** | No frozen path or existing test changed (§2). The full `.venv` suite and the Torch selection pass on 92e671a (§3). On 92e671a, the Phase 1 workload A GPU contract at N=2 gives the same contract result and evaluator verdicts as `p1_2b/smoke_a_n2_001` (before Phase 2) and as RL-11 (§3.4) |
| 6 | Models, tests, scene and docs merged into RCS; external pin updated | **PENDING** | This PR first, then a new versioned descriptor (round rules) |

## 2. Integration correctness

### Diff against `main`

Command: `git diff origin/main...HEAD`. Main is an ancestor of the head, so the two-dot diff is identical.

- **Size.** 317 files, +122,394 / −35.
- **Deletions.** All 35 are in `README.md` (15 lines), `docs/RL_READY_PROGRESS.md` (19) and `docs/RL_READY_PLAN.md`
  (1). The plan line is the D-14 freeze record, which now names the PR #13 merge commit.
- **`src`, `configs`, `scripts`, `tests`.** 265 files, +59,417 / −0. All are new files except `rl/vector_env.py`.
- **New configs** are only under `configs/rubber/` and `configs/materials/rubber_*`. No loader globs
  `configs/materials/`.
- **New scripts** are only under `scripts/rubber/`. `scripts/rl_ready/` is untouched.

### Phase 1 behaviour for the existing backends

The dispatch (`vector_env.py:1218-1230`) is entered only in two cases:
- `backend == "kinematic_grip_batched"`; or
- the task carries the duck-typed marker `is_rubber_task_family`. Only `RubberStripTaskFamily` sets it
  (`rl/rubber_family.py:76`).

So every call on `kinematic_ideal_tracking_batched` or `isaac_lab_newton` with a `None` / int / dict / `TaskFamily` /
`ConfiguredTaskFamily` task falls through to the unchanged body.

- `BACKENDS` still lists only the two Phase 1 backends.
- `test_rubber_rl_dispatch.py::test_default_backend_path_is_unchanged_for_cutting_tasks` covers the fall-through. It
  passed in this review's Torch run.
- The GPU contract at N=2 reproduced the pre-Phase 2 smoke (§3.4).

The only observable change: a call that names the new backend string now raises `UnsupportedRubberConfiguration`
instead of the Phase 1 refusal. That call was refused before as well.

### Frozen files

- **No byte changes** to:
  - `configs/rl_benchmark/*` (`acceptance_v1/v2.json` keep their pinned sha256);
  - `scripts/rl_ready/*`;
  - any listed `evaluator_files` source or `frozen_paths` entry;
  - the v1–v7_2 solvers, `goals.py` and `evaluation.py` (v1 thresholds);
  - the WP-11..17 evidence.
- **Pin tests unchanged and passing:** `test_batched_frozen_sources.py`, `test_acceptance_frozen.py`,
  `test_native_source_freeze.py` and `test_wp12_phase5_frozen.py`.
- **One evaluator glob grows.** Only `tests/*` gains content, and only new files (ADR 0023 §12 notes it). The P1-6
  judge runs from its frozen evaluator checkout, so no acceptance is affected. See F8 for `acceptance_v3`.

### No leakage into the cutting paths

- No pre-existing module references any of the 36 new modules. Checked at `main` and at HEAD.
- Rubber code only reads frozen code: `tlmpm_v6._grid_positions`, `batched.errors`, `material_batch.InvalidBatchEnvs`,
  `rl.termination` and `isaac_sim_runner` helpers.
- The rubber constitutive registry is a separate dict (`batched/constitutive_rubber.py:173`). Nothing registers into
  `batched/constitutive.REGISTRY`.
- The only global setting is `wp.set_module_options({"enable_backward": False})`. It is per module, as in the frozen
  solvers.
- No environment variable, NumPy error state or Torch default is mutated.

### The merge commit

92e671a merges ab839ab (integration branch, docs) with e13339b (rubber).

- `git diff 92e671a^2 92e671a -- src configs scripts tests` is empty.
- `git diff c21ae8d 92e671a` touches 8 docs files only.

So the code checked by the P2-6 review conditions (c21ae8d) is exactly the code proposed for `main`.

## 3. Tests (fresh worktree at 92e671a)

### 3.1 `.venv` full suite

**Command:** `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 TMPDIR=<tmp> .venv/bin/python -m pytest -p no:cacheprovider -q -rs
--durations=25` (all of `tests/`).

**Result:** **3,011 passed, 41 skipped, 6 xfailed, 0 failed** in 2,859 s (exit 0). Python 3.12.12, NumPy 2.5.4, SciPy
1.18.1, Warp 1.17.0.

- **Skips.** All 41 are NOT_RUN gates, not passes:
  - 27 need Torch (absent in `.venv`), including every rubber RL env, dispatch and scripted test;
  - 3 need the Isaac Lab venv;
  - 11 need a machine-local artifact (WP-10 / WP-11 / WP-12 records).
- **What this covers.**
  - Every Torch-free rubber test: physics, visco, contact, envelope, S1, calibration v1 / v2 and gate tripwire,
    protocol and declaration pins, `test_rubber_p2_6_frozen_sources.py`, Native host and compare (the latter
    including the RL-09 recomputation from the artifacts).
  - The Phase 1 batched CPU parity / isolation / partial-reset tests and both particle-contact modes.
  - The frozen pins: `test_batched_frozen_sources.py`, `test_acceptance_frozen.py`, `test_native_source_freeze.py`,
    `test_wp12_phase5_frozen.py`.
  - The WP-11 / WP-12 / v5 / v7 / law-8 regression tests (for example
    `test_tlmpm_v5.py::test_v5_three_bumps_removed_by_three_skims`, `test_cutting_law_v8.py`).
  - `test_repo_naming.py`.
- **Baseline comparison.** RL-11 on the P2-6 head `b2cc6c3` gave 2,932 passed / 41 skipped / 6 xfailed. The 79 extra
  passes are the P2-5 and K2 tests merged since.
- **Link targets.** No file under `artifacts/rl_ready/` (two levels), the held-out registry or the asset cache is
  newer than the run's start marker.

### 3.2 Torch selection (Isaac Lab venv, pytest overlay)

**Command:**

`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONPATH=<overlay>:src .deps/isaaclab-venv/bin/python -m pytest -q
-p no:cacheprovider -rfEs tests/test_vector_env_*.py tests/test_rl_{actions,benchmark,observations,policies_batched,replay,reward_packed,stability,supervision,termination}.py
tests/test_perf_acceptance.py tests/test_rubber_rl_*.py tests/test_rubber_proj_inv_cache_bounded.py
tests/test_rubber_device_staging.py tests/test_rubber_scripted.py tests/test_native_rubber_*.py
tests/test_batched_particle_contacts_v2.py tests/test_rl_lab_*.py tests/test_rubber_p2_6_frozen_sources.py
tests/test_batched_frozen_sources.py tests/test_acceptance_frozen.py tests/test_repo_naming.py`

The selection is the RL-11 Torch selection plus the Isaac Lab workload B tests and the pin and naming tests.

**Result:** **521 passed, 1 skipped, 0 failed** in 968.5 s (exit 0).
- The skip is `test_vector_env_gpu_stream.py` (CUDA hidden): NOT_RUN.
- The selection covers:
  - the Phase 1 vector-env contract for **both** Phase 1 backends (Isaac Lab venv, so workload B ran);
  - the rubber env, dispatch and refusals, contract (F2–F4) and scripted controllers;
  - the Native host and compare, including the RL-09 recomputation from the artifacts;
  - the P2-6 and Phase 1 pins and `acceptance_v2` judge self-tests.
- Environment: Python 3.12.12, NumPy 2.5.2, Torch 2.11.0+cu128, Warp 1.17.0.

### 3.3 GitHub CI

| Commit | Run | Result |
| --- | --- | --- |
| 92e671a (head under review) | 43161309104 (push) | **cancelled** at `setup-python`. The 03e62d3 push 30 s later triggered `cancel-in-progress`, which the workflow sets for refs other than `main`. No test ran |
| 03e62d3 (= 92e671a + 1 line of `RL_READY_PROGRESS.md`) | 43161361779 (push) | **in progress** at 15:16 KST: every step before pytest passed, including "Install media test tools"; pytest running. Pushing this review cancels it (see below) |
| e13339b (`claude/rl-rubber`; same code tree) | 43161271052 (push) | **in progress** at 15:16 KST: every step before pytest passed; pytest running |
| c21ae8d (P2-6 head; same code tree) | 43147390993 (push) | **success**: 2,951 passed, 93 skipped, 6 xfailed, 0 failed (7,382.8 s). The skips are Torch-gated (NOT_RUN), artifact-gated, or behind `main`'s AVX-512 and Franka-asset gates. So the rubber-line CI debt the P2-5 review listed (`rl_lab_*` runner dependencies; `test_batched_pose_interp.py` on EPYC 7763) is cleared |

Pushing this review file cancels run 43161361779 in turn, under the same rule. Runs on `claude/rl-rubber` are not
affected. The merge gate is CI on the PR head (§8.2).

### 3.4 GPU smoke (DIAGNOSTIC, not a protocol row)

**Command.** `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock <chain>`. The chain ran two processes, each
under `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .deps/isaacsim-venv/bin/python`:

- (a) the README "Rubber material (Phase 2)" example, verbatim plus timing and shape checks, with
  `OPENBLAS_NUM_THREADS=1`;
- (b) `scripts/rl_ready/run_p1_2b_smoke.py --backend kinematic_ideal_tracking_batched --gpu-index 1 --tasks 512 15
  --steps 50 --reset-step 25 --evaluation-stream`.

**Setup.**
- Code: 92e671a, clean.
- GPU: RTX 5090 GPU1 (PCI F1:00), driver 570.211.01.
- Libraries: Warp 1.17.0, Torch 2.11.0+cu128.
- Time: 14:34:22–14:35:09 KST.
- Sharing: **EXCLUSIVE_GPU**. The 5 s sampler saw only this review's PIDs on GPU1.
- Warm kernel cache; no cameras; no recording.

**(a) Rubber, N=8.** Declared task `rubber_strip_drag_v2`, `slot_seed=0`. The slots run at n_t 4 / 5 / 6; the declared
step is 5,226 substeps per 20 ms.
- Build 3.06 s; full reset 0.30 s.
- `obs["policy"]` is float32 [8, 78] on `cuda:0`, and `policy` is the only observation key.
- Three zero-action steps took 8.42 / 8.10 / 8.04 s, i.e. ≈ 1.0 env-step/s. This agrees with RL-07's SHARED 0.96.
- Outputs are finite; `termination_reason` is 0.
- A partial reset of env 3 took 0.06 s and changed only row 3 of the policy observation.
- `evaluate()` returns 8 rows from `rubber_tracking_eval_v1`.
- `metadata["privileged_info_keys"]` = (`invalid_reason`, `termination_reason`, `invalid_penalty_applied`), so the
  P2-6 F2 fix is live.
- The receipt says `SIMULATOR_ONLY_UNCALIBRATED`, law `grid_node_penalty_coulomb_v2`.
- Process VRAM 1,196 MiB; max RSS 2.50 GB; exit 0, 30 s wall.

**(b) Phase 1 workload A, N=2.** 102,732 / 103,898 particles/env, 50 Hz, dt 100 µs × 200, warm.
- Contract `vector_env_contract_v1` **PASS** on all clauses: receipt / spaces / metadata, full reset, partial reset,
  steps, evaluate.
- 9.15 env-steps/s (warm median 0.217 s); VRAM 1,620 MiB; one CUDA HashGrid; exit 0, 17 s wall.
- The evaluation-stream verdicts match the pre-Phase 2 smoke `p1_2b/smoke_a_n2_001` and
  `p2_6/rl11_workload_a_contract_n2_001`:
  - FAIL_INTEGRITY (frozen evaluator) on both partial episodes of 50 and 24 steps, 0 labelled;
  - final step counts [50, 24].

## 4. Documentation honesty

### 4.1 Spot-checks (30 claims, 69 values; 68 match)

| Claim (where) | Evidence | Value |
| --- | --- | --- |
| RL-07 0.658 / 1.019 / 0.960 / 1.009 env-steps/s at N 1/4/8/16 (FR §4.1) | `p2_6_rubber_rl_001.json` `/gpu_rows/RL07/rows[0..3]` | 0.6580 / 1.0188 / 0.9599 / 1.0085 ✓ |
| RL-07 N8/N1 gain 1.46; VRAM 1,228 MiB at N=8 (FR, README) | `RL07/scored` | 1.4588; 1228.0 ✓. Note: the N=1 row is the smallest slot only (52,224 particles); N=8 averages 76,848 |
| `rubber_rl_v1` 3.277 env-steps/s at N=8 (FR) | `RL07/rows[9]` | 3.2774 ✓ |
| ≈ 10× slower than workload A's 9.7 env-steps/s (FR §4.1) | raw `p2_6/gpu_rl_b_001/workload_a_n8/result.json` | 9.656 → 10.06× ✓. This number is only in the raw artifact |
| RL-06: 400 steps, 53 partial resets, VRAM growth 0, p95 5.0 s (FR) | `RL06` | 400 / 53 / 0.0 / 5.0037 s ✓ |
| RL-06 p95 0.093 s with `OPENBLAS_NUM_THREADS=1` (FR, README) | `p2_6_rubber_rl_p66_001.json` | 0.09275 ✓ |
| RL-09: max M/U 0.654; 0 / 46; bridge 7.45e-9 m; 1.81 s per warm step (FR, README) | `p2_6_native_rubber_001.json` | 0.65405; 0 / 46; 7.4505e-9; 1.8075 ✓ |
| RL-12: RSS 2,298.0 MiB; growth 0.0; bound 32; cache cap 16 (FR) | `p2_6_rubber_rl_004.json` `/row` | 2297.99; 0.0; 32.0; 16 ✓ |
| RL-11: `.venv` 2,932 passed / 0 failed; Torch 414 / 416 (FR) | `p2_6_rubber_rl_p67_001.json` | ✓ |
| RL-02 L3 max M/U 0.774 (FR) | `RL02c/batch` | 0.77427 ✓ |
| RL-04: 3,780 scalars, worst d/limit 0.469 (FR) | `RL04b/summary` | 3780; 0.46882 ✓ |
| RL-08: ΔO1/L 8.8e-6; ΔO3/O3 3.2e-5; Q1w +0.71 % (FR) | `RL08/dt`, `RL08b/bias_n_t4` | 8.825e-6; 3.238e-5; 0.00710 ✓ |
| **RL-05 v2 "better than zero action in 4/8"** (FR §4.1 l. 423; `P2_6_RUBBER_RL.md` l. 30, 202; ADR l. 2488) | `p2_6_rubber_rl_001.json` `RL05/scripted_better`; raw `gpu_rl_a_001/rows/RL05.json` | **5 / 8** ✗. Envs 2, 4, 5, 6, 7 have a higher scripted return. The verdict (FAIL: ≥ 7 / 8 and ≥ 1 success required; 0 successes) is unaffected |
| CAL-04: χ²_red 2.484 > 1.910; χ²_pilot 1.2732 (FR) | `p2_5_calib_synthetic_001.json` `/cal04`, `/thresholds` | 2.48379 / 1.90974; 1.27316 ✓ |
| CAL-05: 21 / 31 (FR) | `/cal05_rows` | 21 of 31 within ✓ |
| CAL-06: g1 −51 %, g2 +39 % (FR) | `/reveal/fits[0]/rows` | −51.1 %, +39.1 % ✓ |
| True θ: χ²_red 1.416; held-out 31 / 31 (FR) | `/supplementary_diagnostic_not_scored` | 1.41575; 31 × `within_tol` ✓ |
| -09 v2: 1,566 scalars; worst 0.34 (FR) | `p2_4_contact_batched_002.json` | 1566; 0.33899 ✓ |
| Envelope v2: dissipation 0.948 at n_t 4; slip error 1.43 % at n_t 6 (FR) | `p2_6_contact_envelope_002.json` | 0.94831; 0.014263 ✓ |
| S1 cost −0.05 % (N=1), +0.48 % (N=8) (FR) | `p2_s1_selfpen_001.json` `/cost_rows` | −0.048 %; +0.477 % ✓ |
| Workload E 0.64 / 1.35 / 1.47 / 1.52 env-steps/s (FR) | `p2_2_rubber_001.json` `/throughput_v4` | 0.6395 / 1.3454 / 1.4665 / 1.5154 ✓ |
| P2-1 GPU smoke 0.432 / 0.621 / 0.735 env-steps/s (FR) | `p2_1_followup_001.json` `/gpu_smoke_001` | 0.4323 / 0.6209 / 0.7354 ✓ |
| -04: −0.44 % against Euler–Bernoulli (FR) | `p2_1_rubber_001.json` | −0.441 % ✓ |
| Cantilever: 11.54 Hz against 11.57 Hz (FR) | `p2_0_rubber_probes_001.json` `/E3_dynamic` | 11.5413 vs 11.572 ✓ |
| -07b 1.058 %; -02 1.0203 (FR) | `p2_1_followup_001.json` | 0.010581; 1.02031 ✓ |
| Prony tangent max ratio 0.9910; E′ −1.04 / −2.54 %; books 2.58 % (FR) | `p2_3_visco_001.json` | 0.99101; −1.042 / −2.544 %; 2.576 % ✓ |
| -11 v4 GCI 3.98 %; v5 Q1w GCI 2.16 %; bias +0.62 % (FR) | `p2_2_rubber_001.json`, `p2_2_v5_001.json` | 0.03978; 0.021604; 0.006166 ✓ |
| P2-5 GPU-hours: 8.3 for v1; total ≈ 14.5 (FR §7) | the items listed there | 0.4 + 1.07 + 0.16 + 1.07 + 3.3 + 2.3 = 8.30; + 4.93 + 1.3 = 14.53 ✓ |
| P2-6 lock windows ≈ 7.5 h (FR §7) | the windows listed there | 449 min = 7.48 h ✓ |
| P1-2 matched 1.40× / 1.37× (FR §2.7, §4.4) | raw `p1_2/alt_r1_n{1,8,16}_001/steps.npz` | 1.3958 / 1.3699 ✓ (unequal windows give 1.4469 / 1.4178) |

FR = `FINAL_REPORT.md`.

### 4.2 §4.4: which batching gain is right, and the erratum needed

**1.40× (N=8) / 1.37× (N=16) is right.** Recomputed from the raw step walls of P1-2 alternation round 1:
- N=8 over steps [0:150], divided by N=1 over [0:150]: **1.396×**.
- N=16 over [0:100], divided by N=1 over [0:100]: **1.370×**.
- The superseded 1.45× / 1.42× (1.447 / 1.418) divided by a 300-step N=1 window, whose cost rises late (review fix
  F3, `P1_1_P1_2_MILESTONES.md`, commit 0a26b5f).
- `P1_PERFORMANCE_TABLE.md` was written later (ff4133f). It still quotes the superseded pair, at §5.1 line 223 and
  §5.2 line 280.

**The exclusive table has the same window bias.** Its §5.1 frozen_v1 ratios over N=1 (1.49 / 1.43 / 1.46×) also
divide by a 300-step N=1 window: steps 21–320, which end in the late approach and the first contact steps. The N ≥ 4
rows use 100-step windows.
- Mean step / median step is **1.040** (frozen_v1) and **1.050** (`alias_free_v1`) at N=1, and **1.000** at every
  N ≥ 4 (`exclusive_001/a_n*/rounds/round_0.json`). The tail pulls the N=1 mean rate down.
- A reviewer proxy on median steps (not a measured matched window) gives:
  - frozen_v1: 1.43 / 1.37 / 1.40× at N 4 / 8 / 16;
  - `alias_free_v1`: 1.35 / 1.37 / 1.33×.
- These agree with the matched P1-2 values.
- So "material-only batches gain about 1.4–1.5×" is ≈ 4–5 % high. That phrase appears in FR §2.3,
  `P1_PERFORMANCE_TABLE.md` §5.1, `PHASE1_PR_BODY.md` and `PHASE1_ACCEPTANCE_REVIEW.md`.

**Erratum needed** (additive: strike through the old values, keep them):
1. `P1_PERFORMANCE_TABLE.md` §5.1 and §5.2:
   - quote "1.40× (N=8) / 1.37× (N=16), matched windows (review fix F3); superseded: 1.45× / 1.42×";
   - state that the §5.1 N=1 baseline is a 300-step window against 100-step windows at N ≥ 4 (mean / median 1.04–1.05
     at N=1 only), so its ratios carry the same bias.
2. FR §2.3 "Scaling": "about 1.4× (1.37–1.40× in matched windows; the exclusive table's 1.43–1.49× uses unequal
   windows)". FR §4.4 should point to the new erratum.

### 4.3 Host paths

- **No host paths:** `FINAL_REPORT.md`, ADR 0023, `P2_5_RUBBER_CALIBRATION.md`, `RL_READY_PLAN.md`.
- **`README.md`:** one new line, the overlay path in the rubber Torch test command. This follows the README's existing
  test-command style, which is allowed.
- **Step reports and reviews:** they carry `/mnt/data/...` lock and `TMPDIR` paths inside exact-command records. 59
  docs on `main` already do this (AGENTS.md: record exact commands).
- **`RL_READY_PROGRESS.md`:** one line names the external frozen checkout. Acceptable for a progress log.

### 4.4 Completeness of 「待使用者決定」

The list has Data, U1–U4, U-P15-1..3 and U-P2-1..6; FR §6 mirrors it. Two items are missing. Both need the user,
because the agent may not change the shared venvs or the CI cost:
- **BLAS threads.**
  - Default in force: `OPENBLAS_NUM_THREADS=1` is a documented user setting, not enforced.
  - Alternative: approve adding `threadpoolctl` to the venv locks, or an in-code guard.
  - Why it matters: the reset bits depend on the thread count, and the reset p95 is 5 s at the default against 0.09 s
    with one thread.
- **Torch in GitHub CI.**
  - Default in force: the local Torch-venv runs are the merge gate. The P2-6 env, dispatch and scripted tests are
    NOT_RUN in CI.
  - Alternative: a Torch CI job. CI already takes about 2 h.

**Stale text in the list.**
- U2 still says "window being finalised to start at tool contact"; D-14 finalised the window.
- FR line 538 says the U-P2-3 entry "is split by the U-P2-4..6 lines"; 30b4a44 rejoined it.

### 4.5 FAILs, claims and representation

- **Every FAIL verdict** in `docs/evidence/rl_ready/p2_*.json` maps to an FR §4.1 bullet:
  - -07, -07b and -02;
  - the L3 band;
  - -08;
  - -11 v4 and v5;
  - VISC-03 and -05 v1;
  - CONT-01 / 03 / 04 / 05 v1, CONT-01 / 04 / 05 v2, CONT-01 d v3, and the -09 v1 rule;
  - ENV-01 v2;
  - RL-05 v2 and v3;
  - CAL-04 / 05 / 06.

  The `pilot_validation` FAIL from development is covered by the "v2 development" narrative.
- **No overclaiming:** the README and the FR claim no calibration, no real-world validity and no policy training.
- **P2-5** follows erratum E1: a truncated fit, with the basin hypothesis untested.
- **Media** are labelled as reported, not physics. `P2_6_RUBBER_RL.md` §9.4 states the frozen rendered clamp that is
  visible in the close-up; the visual smoke is NOT_RUN.

## 5. Repository hygiene

- **Naming.** `tests/test_repo_naming.py`: 2 passed (it also ran inside both suites). The 230 commit messages in
  `origin/main..92e671a` contain no standalone acronym (case-insensitive word-boundary search).
- **Sizes.** 317 files, 6.52 MB. Largest files:
  - `docs/media/p2_6/native_rubber_001_closeup_steps_0_30_50_100.png`: 252 KB;
  - ADR 0023: 240 KB;
  - `p2_4_contact_x2_001.json`: 237 KB.

  New evidence is 1.48 MB; the synthetic datasets are 0.94 MB. No `artifacts/` file is committed, and the only
  binaries are the two 960 × 540 PNGs.
- **Media.** The close-up (steps 0 / 30 / 50 / 100) and the overview (step 100) come from `native_rubber_001`. They
  show the strip deforming and the rendered jaw left at its reset pose. The report records that jaw as a fixed defect
  (`measured_pose_v1`); its visual smoke is NOT_RUN.
- **Committed oracle data.** `configs/rubber/calibration_synthetic/*/oracle_heldout.json` (145 KB each) holds the
  synthetic held-out truth series. It was committed with the dataset (a0250bb) and accepted by the P2-5 review. The
  fit API refuses held-out experiments.

## 6. Conditions of the step reviews

| Review | Condition | Status at 92e671a |
| --- | --- | --- |
| P2-0..P2-2 | Reviewed batched F8 / F10 before batched VISC runs and P2-6 grips | Met (P2-3 prerequisites review) |
| P2-3 prereq | Cite visc as "v1 FAIL, v2 PASS (post-hoc measurement correction)" (R2) | Met in FR §4.1; missing from the FR box-2 row (F4 b) |
| K2 combined | K2-1 v3 scope; the Q1w conditions (RL-08 b reports the n_t 4 bias against a trigger) | Met: the RL task uses law v2; RL-08 b gives +0.71 % against a 1 % trigger |
| P2-5 (GO after Addendum A) | C1–C4 errata; R1–R3; the rubber-line CI debt, for the Phase 2 PR | Met. CI debt cleared: in the c21ae8d run, `rl_lab_*` and `test_batched_pose_interp.py` are NOT_RUN skips, with 0 failed |
| P2-6 (GO_WITH_CONDITIONS) | C1: CI green | c21ae8d is green (a later, code-identical head) |
| | C2: errata (F1, F5, F2–F4, F9) | Met (`P2_6_RUBBER_RL.md` §12; ADR §23.7). F9 (d): the `or True` is removed. F9 (b): the docstring is fixed, with a declared pin update |
| | C3: the ADR §23.6 collision | Met: P2-6 renumbered to §23.7 |
| | Z-a: fix F1, pin the module, add a host-RSS row | Met: bounded LRU; `rubber_material_batch.py` pinned; RL-12 PASS |
| | Z-b: F2 / F3 / F4 | Met: privileged keys in metadata (seen live in §3.4); `InvalidRubberEnvs(InvalidBatchEnvs)`; unknown kwargs raise `UNSUPPORTED_OPTION` |
| | Z-c: rerun RL-07 EXCLUSIVE, or keep the SHARED label | The SHARED label was kept. The stated reason ("the GPUs always carry other projects' processes") did not hold during this review (F2) |
| | Z-d: merge `main` | Met |

## 7. Findings, ranked

There is no BLOCKER and no MAJOR finding. The physics, protocol and code defects found by the step reviews are
either fixed or recorded. The new findings are:
- documentation accuracy (F1, F3–F7);
- one measurement-rule item (F2);
- forward-looking items (F8, F9).

| ID | Severity | Finding | Concrete failure scenario | Recommendation |
| --- | --- | --- | --- | --- |
| F1 | **LOW–MEDIUM** | **The calibration-to-RL handoff is not closed, and neither the FR nor the README says so.** (1) Friction: the P2-5 friction rows use law v3, so the fitted `μ_table` is a law-v3 translational-sliding coefficient (`P2_5_RUBBER_CALIBRATION.md` §3). The RL task uses law v2 with μ fixed at 0.3. (2) Resolution: θ̂ is `RESOLUTION_SPECIFIC` to n_t 4, but the RL slots run at n_t 4 / 5 / 6. (3) Material source: the RL task draws material from the fixed ranges in `rubber_strip_drag_v2`, not from a recipe | A user with real data runs the P2-5 pipeline and puts θ̂ and `μ_table` into the RL task. The friction law differs, and the t 5 / 6 mm slots carry an unshifted resolution bias. That is the "unknown material-physics difference" plan P2-6 rules out, and the documentation calls the interface "ready" | One paragraph in FR §5 and in the README's "Calibration interface" bullet, listing what a real-data user still needs: a new task-config version; a law-v2 friction fit, or an equivalence check at the target μ; and the bias shift for n_t 5 / 6 |
| F2 | LOW | **Phase 2 has no `EXCLUSIVE_GPU` throughput row.** RL-07 is `SHARED_GPU`, with nvidia-smi samples only at the start and end of a 12-minute row. The only co-tenant was PID 642659, at 0 % util in the start sample. The Native row and every replay are SHARED too. The round rules say phase acceptance tables need a window with no other heavy compute, and that rows are rerun "when a GPU becomes free". GPU1 had no foreign process during this review (§3.4) | The Phase 2 throughput (≈ 1 env-step/s; N8/N1 1.46×) becomes the only reference on `main` and in the next external descriptor. A later cycle or an RL-run budget uses it as an acceptance-grade baseline without knowing whether foreign load was present between the two samples | Before the merge, if an idle GPU exists: rerun RL-07 unchanged (`rubber_rl_tests_v2`) as an additional labelled row, with a continuous sampler. Keep the original row and its verdict. Otherwise the PR body states that every Phase 2 row is SHARED |
| F3 | LOW | **README API description is wrong for the rubber backend.** The README says "**All** backends share one contract (`rl.vector_contract`) … `evaluate(env_ids)` through the frozen `goals.evaluate_goal` … `FAIL_INTEGRITY`". The rubber backend has its own contract (`rl.rubber_contract`, `rubber_vector_contract_v1`, actions [N, 6]) and its own evaluator (`rubber_tracking_eval_v1`, with a top-level `success`). It shares only the reset / step / `termination_reason` / `privileged_info_keys` conventions | A wrapper over `open_vector_env` reads `result["success"]` / `summary` from `evaluate()`, as the README promises. On the rubber backend it raises `KeyError` | README: change to "Both cutting backends share …" and add one line naming the rubber contract and evaluator |
| F4 | LOW | **FR accuracy.** (a) Box 3 says "RL-01..RL-12 PASS at N=8", but GPU isolation (RL-04 b) ran at N=4 and RL-02 c scores 2 envs. (b) The box-2 row omits the post-hoc hysteresis PASS and the κ model-form error. (c) §4.2 calls `rubber_calib_tests_real_v1` "declared". It is only an outline inside `rubber_calib_tests_v1.json` `real_data` ("to be declared when real data exist"; `REAL_PROTOCOL_DECLARED = False`, `rubber_calibration_v2.py:47`), with no thresholds. (d) Stale text: lines 3–6 (the P2-5 / P2-6 reports "are on" the rubber branches) and line 538 (the U-P2-3 split). (e) §8 and the README give the row list as `RL-01,RL-02b`; `run_rubber_rl_gpu.py` accepts only `RL01,RL02b` and exits with "unknown rows". (f) RL-05 v2 "4/8" should be **5/8**: the same miscount sits in FR l. 423, `P2_6_RUBBER_RL.md` l. 30 and l. 202, and ADR l. 2488 | (a) A reader believes GPU isolation was tested at N=8. (c) A user with real data believes thresholds are already frozen and runs the held-out before declaring them, which plan P2-5 forbids. (e) The reproduction command fails | Additive corrections in the FR. For (f), add an additive erratum note in the P2-6 report and the ADR, as F9 did for "6 envs" |
| F5 | LOW | **Batching gain overstated (inherited from Phase 1, §4.2).** `P1_PERFORMANCE_TABLE.md` §5 quotes the superseded 1.45× / 1.42×. The exclusive §5.1 ratios, and FR §2.3's "1.4–1.5×" built on them, compare a 300-step N=1 window with 100-step windows, which makes them ≈ 4–5 % high | Someone sizes N, or checks the ADR 0021 expectation of at least 1.3×, against "1.4–1.5×". The matched value is 1.37–1.40×, and ≈ 1.24–1.26× over the frozen single env | The additive erratum of §4.2, in this PR (docs only) |
| F6 | LOW | **Two limitations are missing from FR §5 and the README.** (1) The κ = 20 μ model-form error. ADR 0023 §3 quantifies it: J − 1 = +1.2 / +2.5 / +5.4 % and nominal stress −2.1 / −2.7 / −4.2 % at λ 1.25 / 1.5 / 2.0; bending stiffness −1.6 %. (2) The default contact law is v1 (`rubber_contact.py:61`, `ContactConfig.law` at `:105`), which has the worst formal record (CONT-01 / 03 / 04 / 05 FAIL). The README says "v1 is the default" without saying that. The friction envelope is enforced only by the RL family (`rl/rubber_family.py:139-156`), while both recipes carry a nominal `friction_table` of 0.8 (range 0.3–1.5). So `LiveRubberMaterial` / `BatchedRubberMaterial` used directly build a v1, μ 0.8 scene with no refusal, outside every established envelope row | A user compares volume change or a confined (pinch) response with real rubber (ν ≈ 0.4995) and takes a +5 % J error for physics. A new scene built from a recipe without an explicit law silently runs v1 at μ 0.8 | List both as limitations, and say that the envelope is checked only in the RL task. Changing the code default, or enforcing the envelope in the material classes, would be a versioned decision, not part of this PR |
| F7 | LOW | **`RL_READY_PROGRESS.md` 「進行中」 contradicts itself and is out of date.** (1) It says "P2-6 uses law v2 with table friction **U[0.3, 0.4]**"; the config fixes [0.3, 0.3], and the same paragraph later says "FIXED at 0.3". (2) "Running: K2 fixes …; P2-6 env …; Native …; P2-5 …" — all of it is done. (3) It calls the pilot bias "a resolution bias …, not an optimiser failure". E1 withdrew that, though later text in the paragraph corrects it. (4) U2 still says "window being finalised". (5) Two pending user decisions are missing (§4.4) | A session that resumes from this file, or any reader of it, believes friction is randomised in [0.3, 0.4], or restarts "running" work | Rewrite 「進行中」 to the current state and move the history below it. Add the two 待使用者決定 entries |
| F8 | INFO | **Effects on the next acceptance version.** After the merge, `acceptance_v2` G1 (`cpu_tests: tests`, timeout 7,200 s) also runs the rubber suites: about 3,000 s locally, while the whole suite took 7,383 s on GitHub runners. G2's `torch_tests` does not list the rubber Torch tests, and G1 runs in `.venv`, which has no Torch. A P1-6 candidate may edit the changeable `rl/vector_env.py` or `batched/*`. Its only rubber guards are the text check `test_vector_env_dispatch_branch_present` and the Torch-free rubber tests | A future candidate keeps the dispatch text but breaks the rubber env at runtime. It is accepted, and the break shows only in a later local Torch run | When `acceptance_v3` is cut: add `tests/test_rubber_rl_*.py`, `test_rubber_scripted.py` and `test_native_rubber_*.py` to `torch_tests`, and re-measure the G1 wall time against its timeout |
| F9 | INFO | **CI cost.** The GitHub CI full suite takes ≈ 2 h 03 min (c21ae8d; +414 s over 73c624b). Torch-gated tests are NOT_RUN in CI | Pressure on the CI queue; no correctness risk | Track it together with the Torch-CI user decision (§4.4) |

## 8. Verdict and conditions

**GO_WITH_CONDITIONS** for opening and merging the Phase 2 PR (`claude/rl-ready-rubber-physics` → `main`).

### 8.1 Before the PR is opened

Documentation only, additive, no re-scoring.

- **D1.** FR corrections:
  - F4 (a)–(f);
  - the §2.3 / §4.4 batching wording (F5);
  - the calibration-handoff paragraph (F1);
  - the two missing limitations (F6).
- **D2.** The additive erratum in `P1_PERFORMANCE_TABLE.md` §5.1 / §5.2 (F5, §4.2), and the RL-05 v2 5/8 erratum note
  in `P2_6_RUBBER_RL.md` and the ADR (F4 f).
- **D3.** README:
  - the contract wording (F3);
  - the handoff sentence (F1);
  - the v1-default and κ = 20 μ limitations (F6);
  - the row names (F4 e).
- **D4.** Rewrite 「進行中」 in `RL_READY_PROGRESS.md` and add the two 待使用者決定 entries (F7, §4.4).
- **D5.** Run `tests/test_repo_naming.py` on the docs commit. If any file under `src/`, `tests/`, `configs/` or
  `scripts/` changes after 92e671a, rerun the `.venv` suite and the Torch selection of §3.

### 8.2 Before the merge

- **M1.** GitHub CI is green on the PR head. If the only failure is "Install media test tools", that is an apt-mirror
  outage: rerun the job.
- **M2.** If an idle GPU window exists, run RL-07 on it as an additional, labelled `EXCLUSIVE_GPU` row (F2). If not,
  the PR body states that every Phase 2 performance row is SHARED_GPU, and why.
- **M3.** The PR body links this review. It carries the limitations of §8.3 and the Done table of §1, with box 6
  PENDING until the new descriptor is SYNCED.

### 8.3 Known limitations for the PR body (not blocking)

- **Material status.** `SIMULATOR_ONLY_UNCALIBRATED`. No real data exists, and no recipe is promoted.
- **Calibration.**
  - The P2-5 synthetic numerical validation **FAILED** (CAL-04 / 05 / 06), and v3 was not run. The fit procedure is
    therefore not validated.
  - θ̂ is resolution-specific to n_t 4.
  - The real-data protocol exists only as an outline.
  - Handing calibration results to the RL task needs a new task config, a law-v2 friction fit and a bias shift (F1).
- **Physics.**
  - The κ = 20 μ model-form error (F6).
  - The n_t 4 bending bias of +0.62 % / +0.71 %; the -11 v5 FAIL is kept.
  - Hysteresis: v1 FAIL, v2 PASS only after a post-hoc measurement correction.
  - The -01 d energy books FAIL.
  - Self-contact is detected only, not resolved.
  - The default law is v1, whose FAILs are recorded. The friction envelope is enforced only in the RL task.
  - Law v3 is valid for translational sliding only (K2-1).
- **RL task.**
  - μ is fixed at 0.3; the v2 envelope fails at 0.4.
  - Geometry varies per env, not per episode, and `reset(seeds=None)` repeats the episode.
  - The grip is kinematic, with no robot coupling; the 20 ms handoff was assessed as significant.
  - RL-05 passed only with the third declared controller.
  - GPU isolation was tested at N=4.
- **Performance.**
  - ≈ 1 env-step/s at N=8–16 at full fidelity.
  - Every Phase 2 row is SHARED_GPU unless M2 adds one.
  - There is no EXCLUSIVE Native row.
- **Operations.**
  - Set `OPENBLAS_NUM_THREADS=1`. It is not enforced, and the reset bits depend on it.
  - The P2-6 Torch tests are NOT_RUN in GitHub CI; the local Torch run is the gate.
  - The Native visual smoke of `measured_pose_v1` is NOT_RUN.
- **Inherited from Phase 1.** The batching gain is 1.37–1.40× in matched windows (F5).

## Addendum A — re-check for PR #16 at head `9ea50a6` (2026-11-12)

**Verdict: GO.** Merge #16 once CI (M1) is green on the PR head. Three small text items remain (A1–A3). None of them
blocks the merge.

**Scope.** PR #16 (https://github.com/AnsonNing/RoboticCuttingSim/pull/16), head `9ea50a6`, base `main` `62caecc`.
The commits since the review head `92e671a` are:

| Commit | Content |
| --- | --- |
| 03e62d3 | one progress line |
| 8b45de7 | this review |
| ee17077 | D4 |
| 756120f | D1/D2 |
| 3a47191 | D3 |
| 79a5e28 | U-P2-7/8 in the final report |
| 016a193 / 9919a6e / fc7dff5 | M2 evidence |
| 9ea50a6 | M2 row in the final report |

**How it was checked.**
- Read-only. Nothing was committed or pushed and no GitHub comment was posted.
- `git diff 92e671a 9ea50a6` was read in full for every document.
- The M2 sampler and raw rows were recomputed from `artifacts/rl_ready/p2_6/` of the p26 worktree.
- The PR body was read with `gh api repos/AnsonNing/RoboticCuttingSim/pulls/16 --jq .body`.

### A.1 No code changed after 92e671a: confirmed

- **Code paths.** `git diff 92e671a 9ea50a6 --stat -- src configs scripts tests` is empty.
- **Whole diff.** 9 files, +1,439 / −34, all under `README.md` and `docs/`.
- **Files added:** this review, `p2_6_rubber_rl07_exclusive_001.json` and its `_note.md`.
- **Existing files changed:** `README.md`, `RL_READY_PROGRESS.md`, `FINAL_REPORT.md`, `P2_6_RUBBER_RL.md`, ADR 0023,
  and `P1_PERFORMANCE_TABLE.md` (the first change to that file in this PR).
- **Consequence.** The test results of §3 (`.venv` 3,011 passed / 0 failed; Torch 521 passed / 0 failed) and the GPU
  smoke still apply to the PR head, so D5 needs no rerun.
- **Naming.** The acronym search over every tracked file and path at `9ea50a6`, the 10 commit messages in `92e671a..9ea50a6` and
  the PR body finds nothing.

### A.2 D1–D4: met

**D1 (`FINAL_REPORT.md`).**
- **Box 2** now records:
  - the post-hoc hysteresis pass;
  - the κ = 20 μ model-form error, with the correct citation ADR 0023 **§6**;
  - that this review grades the box PARTIAL.
- **Box 3** names the N=8 rows, records GPU isolation (RL-04 b) at N=4, and notes that RL-02 c scores 2 envs of an N=8
  batch. The protocol text confirms that RL-02 c runs N=8.
- **§4.2** says the real-data protocol is an outline: no thresholds, NOT_RUN, to be declared before any held-out run.
- **Stale passages:** the header and the U-P2-3 note are replaced.
- **Reproduction:** the §8 row names are now `RL01,RL02b`, with the full list.
- **RL-05 v2** reads 5/8 with the erratum noted, in both §4.1 and §4.4.
- **Batching gain:** §2.3 and §4.4 now give 1.40× / 1.37× and state the 4–5 % window bias.
- **New limitations in §5:**
  - the κ model-form error;
  - the v1 default law;
  - the envelope enforced only in the RL task, with recipe μ 0.8;
  - the calibration-to-RL handoff paragraph (friction law, resolution, material source).
- **§6** adds U-P2-7 and U-P2-8.

**D2 (errata).** All three are additive and keep the old values struck through:
- `P1_PERFORMANCE_TABLE.md`: strikes 1.45× / 1.42× → 1.40× / 1.37× in §5.1 and §5.2, and adds §5.5 erratum E-1 with
  the window-bias numbers of this review §4.2;
- `P2_6_RUBBER_RL.md`: §0 / §5 strike-throughs plus a new §13;
- ADR §23.7: a strike-through plus a closing erratum note.

No frozen evidence JSON was edited.

**D3 (`README.md`).**
- "Both cutting backends share one contract".
- A paragraph naming `rl.rubber_contract`, `rubber_vector_contract_v1`, actions `[N, 6]` and `rubber_tracking_eval_v1`.
- v1 is marked as the default law with the worst record.
- The real-data protocol is described as an outline.
- The handoff paragraph.
- The κ and v1 / envelope limitations.
- Row names `RL01,RL02b`.

**D4 (`RL_READY_PROGRESS.md`).**
- 「進行中」 is rewritten as the current status.
- The old entries, including "U[0.3, 0.4]", moved unchanged to a "Status log" section.
- U-P2-7 (BLAS threads) and U-P2-8 (Torch in CI) are added to 「待使用者決定」 with defaults and alternatives.

**Residual (A1).** Two stale bits remain, both non-blocking:
- the U2 line still says "window being finalised to start at tool contact";
- the new 「進行中」 lists D1–D3 and M2 as "In progress", although 756120f…9ea50a6 completed them.

Correct both in the post-merge docs PR.

### A.3 M2 (supplementary EXCLUSIVE RL-07 row): honest and correctly transcribed

**Provenance.**
- Run `artifacts/rl_ready/p2_6/gpu_rl07_exclusive_001` at commit `c21ae8d`, clean tree. That commit has the same code
  tree as 92e671a / 9ea50a6.
- The declared runner `run_rubber_rl_gpu.py --rows RL07`, with `rubber_rl_tests_v2` pinned.
- RTX 5090 GPU1 (F1:00.0), driver 570.211.01, `OPENBLAS_NUM_THREADS=1`.
- Blocking `rcs_rl_runner/gpu1.lock`, 31,937 MB free at start.
- Window 15:17:38–15:26:45 KST.

**Sharing label: EXCLUSIVE_GPU is justified.** I recomputed the sampler data from
`rl07_exclusive_samples_001.jsonl`:
- 275 samples; maximum gap 2.000 s; span 548 s, which covers the run window.
- 0 foreign entries.
- Only three PIDs appear: the runner 3345701 and its two Phase 1 workload-A children 3350159 / 3350354. The sampler
  script `rl07_exclusive_sampler_001.py` classifies them as the run's own processes by ancestry.

The evidence also records two weaker in-run labels, which is honest:
- the runner's own two-sample label `EXCLUSIVE_AT_SAMPLES` (`runner_label`);
- the children's self-label SHARED, explained by the parent holding GPU memory.

The sampler sees compute processes only, not utilisation. A foreign job shorter than 2 s could be missed; this does
not change the label.

**Transcription.** All 10 raw rows (`rows/RL07.json`) were recomputed as N / median of the 5 warm walls. They match
the evidence JSON, the note, FR §4.1 and the PR body:

| Set | N | env-steps/s | VRAM (MiB) |
| --- | --- | --- | --- |
| full slot set | 1 / 4 / 8 / 16 | 0.648 / 1.002 / 0.942 / 0.996 | 812 / 972 / 1,228 / 1,676 |
| uniform | 1 / 8 / 16 | 0.624 / 1.144 / 1.209 | 876 / 1,196 / 1,548 |
| `rubber_rl_v1` | 1 / 8 | 1.195 / 3.236 | 908 / 1,004 |

- The FR omits the uniform N=4 row (1.029). It is in the JSON, and that is fine.
- **N8/N1 ratios (median):** 1.454 / 1.835 / 2.709 for the three sets.
- **"Within about 2 %"** holds: the largest difference is +2.2 %. The shared figures are slightly *higher* than the
  exclusive ones, so this is run-to-run spread, not a co-tenant penalty.
- **Formal row unchanged.** The formal SHARED row `gpu_rl_b_001` stays the row of record with its PASS, as all three
  documents say.

### A.4 M3 (PR body): met, with one wording fix (A2)

**What the body contains.**
- It cites this review by path (`docs/reports/rl_ready/PHASE2_ACCEPTANCE_REVIEW.md`, GO_WITH_CONDITIONS) and states
  D1–D4 done and M1–M3 open.
- It carries the Done table: boxes 1 / 4 / 5 PASS, box 2 PARTIAL, box 3 PASS_WITH_LIMITATIONS, box 6 PENDING. Its
  verdicts match §1 of this review.
- It carries the full §8.3 limitation list, including:
  - the P2-5 FAIL with its numbers, and that the fit is not validated;
  - the outline status and the handoff;
  - κ, the bending bias, post-hoc hysteresis, -01 d, self-contact, the v1 default and envelope scope, and v3 scope;
  - μ 0.3, geometry per env, the kinematic grip, RL-05 v4 and isolation at N=4;
  - the performance labels, BLAS, CI Torch, the visual smoke and the Phase 1 erratum.
- It lists the user decisions U-P2-1..8.
- It has no host paths and no calibration or training claim.

**A2 (non-blocking; editing the PR body does not trigger CI).**
- "Against `main`, only 4 existing files change" is now **5**: `P1_PERFORMANCE_TABLE.md` gained the additive erratum
  E-1 in this PR.
- "Every formal Phase 2 row is SHARED_GPU" should read "every formal Phase 2 **throughput** row (RL-06, RL-07, Native)
  is SHARED_GPU". RL-01, RL-02 b, RL-03 and RL-04 b were labelled `EXCLUSIVE_AT_SAMPLES`.

### A.5 Erratum to this review (A3)

- §1 (box 2, item g) and F6 cite the κ = 20 μ model-form error as "ADR 0023 §3". It is **ADR 0023 §6** (D3 —
  near-incompressibility; the table and model-form numbers are at the ADR's lines ≈ 318–345). The final report and
  README already cite §6. The numbers are unaffected.

### A.6 CI (M1)

CI on 9ea50a6 was still running at the re-check:
- push run 43168433241;
- `pull_request` run 43168462836.

Merge once the PR check is green. If "Install media test tools" is the only failure, it is an apt-mirror outage:
rerun. The code tree equals c21ae8d, whose CI was green (2,951 passed, 0 failed), so a code failure would be
unexpected. Any failure outside that step needs a look before merging.

### A.7 Remaining items (none blocking)

| ID | Item | When |
| --- | --- | --- |
| A1 | `RL_READY_PROGRESS.md` U2 "window being finalised"; 「進行中」 still says D1–D3 / M2 are in progress | post-merge docs PR (FINAL_REPORT §9 fill-in) |
| A2 | PR body: "4 existing files" → 5; "every formal Phase 2 row" → "every formal throughput row" | PR body edit now, or record in the follow-up |
| A3 | This review's "ADR 0023 §3" → §6 | this addendum |
| — | F8 (`acceptance_v3`: rubber Torch tests in `torch_tests`, G1 timeout) and F9 (CI cost, U-P2-8) | as recorded |
