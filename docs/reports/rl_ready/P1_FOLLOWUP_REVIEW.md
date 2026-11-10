# Phase 1 follow-up — xhigh review (2026-11-11)

**Commit reviewed: `5d05dd6`** (branch `claude/rl-p1-followup`), read-only, in the detached worktree
`/home/ningan/RoboticCuttingSim-rl-fureview`. Every `file:line` below refers to `5d05dd6`.

**Scope.** `git diff c5e96b2..5d05dd6`, without the `claude/rl-phase1` commits merged by `68e7d62` (`0b99e7b`,
`f6fad6e`, `0ded0be`, `7ac820b`, `18ee275`, `bc171ec`, `cc0e24c`, `63c1cfb`), which were reviewed before. The
follow-up commits are `cf04c1a`, `683fe42`, `eb22d92`, `8e583c3`, `2195bb2`, `49a6f42`, the merge `68e7d62`, `52d2d6c`
and `5d05dd6`. They answer findings 1, 2, 3, 4, 10 and 12 of `PHASE1_ACCEPTANCE_REVIEW.md`, and add the CI
concurrency block.

**Method.**
- I read `P1_FOLLOWUP.md`, `docs/evidence/rl_ready/p1_followup_001.json`, D-14, `acceptance_v2.json` and every
  changed source, script and test file.
- I recomputed the window numbers from the raw dumps (`p1_6_r0/smoke_003` G6, `p1_followup/gpu_window_check_001`)
  and from the WP-17 Native per-step material statistics (§3.1).
- I re-judged the preserved `smoke_003` outputs with the judge of `c5e96b2` and with the judge of `5d05dd6`, and ran
  `G9:window` on the real dumps (§3.2). I ran the rollout checker of both commits on two preserved v1 rollouts.
- I ran the external adapter's `config_problems` on both configs, read-only (no bytecode, no file written).
- CPU tests only (§6). No GPU job. All scratch outputs are under my session scratchpad; nothing outside this file was
  edited.

## 1. Verdict

**PASS_WITH_LIMITATIONS. PR #13: GO under the conditions in §8.**

- No BLOCKER, no MAJOR.
- No frozen module changed: `git diff c5e96b2..5d05dd6` over the `frozen_paths` of both configs, `batched/`,
  `rl/replay.py`, `rl/benchmark.py`, `rl/telemetry.py`, `profiling.py`, `physical_tool.py` and `grasp_monitor.py` is
  empty. Under `configs/` the only change is the new `acceptance_v2.json`.
- `acceptance_v1.json` is byte-identical (`3592d940…`). Both pins and the band pins are correct.
- The judge judges v1 exactly as before. A re-judge of `smoke_003` with both judges gives identical gates (§3.2).
- Findings 2, 3, 4, 10 and 12 are fixed and tested. Default physics is unchanged.
- Finding 1 is fixed for the primary N=16: the window now holds tool–material contact (46 steps) and cutting
  (16,588 new labels), against none before. However:
  - the headline "contact" numbers (44 of 49 steps, 93.8 %) count **particle–particle** contacts, not tool contacts;
  - the window holds 46 of the 101 tool-contact steps and 44 % of the labels (finding 1).
- The claim that the 3-pair rule is "stricter than v1" holds only for its sign-test part. The N=1 floor got noisier
  (finding 2).
- GitHub CI run 42836425615 on `5d05dd6`: **success**, 2465 passed, 58 skipped, 6 xfailed, 0 failed. This equals
  the local CI simulation `ci_sim_full_002`.

## 2. Per-item verdicts

| Item | Verdict | Basis |
| --- | --- | --- |
| Finding 1: `acceptance_v2` G9 window | **PASS_WITH_LIMITATIONS** | Reproduced: 44 particle-contact steps, 143,022 (93.81 %), +16,588 labels, 0 in the v1 window 21–120. The GPU check holds (44 / 44 steps, +16,587 / +16,588). Limitations: finding 1 (what "contact" counts, coverage of the tool-contact phase). |
| `G9:window` gate | **PASS_WITH_LIMITATIONS** | Correct on synthetic and real dumps: PASS on `smoke_003` (44 / 16,588 vs 44 / 16,587). Candidate below → FAIL, baseline below → BLOCKED, no dumps → BLOCKED. Limitations: measured on G6/G7 episode processes; counts particle contacts and labels, not tool contacts; margin 40 vs 44 steps. |
| Statistics (3 pairs, N {1, 16}) | **PASS_WITH_LIMITATIONS** | Adequate on an exclusive GPU (σ ≈ 0.006). The "stricter" claim is incomplete. On a shared GPU the N=1 floor rejects more neutral candidates. N=8 is not gated (finding 2). |
| Budget (1.64 → about 1.75 GPU-h) | **PASS_WITH_LIMITATIONS** | Arithmetic reproduces (§3.3). Limitations: GPU time only; the contact factor comes from one N=2 `SHARED_GPU` run; D-14 lacks the revision; the "5 % of G9" figure is wrong (finding 3). |
| Finding 2: freeze and pins | **PASS** | `tests/test_acceptance_frozen.py`: v1, v2 and both band files are pinned. Every `acceptance_v*.json` must be pinned. Version equals file name. |
| Harness and judge: v1 unchanged | **PASS** | Only additive result keys (`step_walls_s`, optional `acceptance_config`), written outside the timed loop. Judge: new paths only under `window_check` / `phases`. Re-judge identical (§3.2). |
| Finding 3: B `fidelity()` | **PASS** | `lab_fidelity_v2` compares workload A's nine keys plus the robot options. Live Lab test at dt 2e-4 reports dt, 100 substeps, spacing and steps. The recorded B2 receipt (dt 1e-4, dx 0.004, 8 slots, 0.02 s) stays full fidelity. |
| Finding 4: B2 settle latch | **PASS** | `reset_envs` re-applies bit 16 and `TERM_RIGID_INVALID` from `snapshot.passed` (`lab_batched_env.py:826-831`). `rigid_validity_kernel` keeps the bit (`r = reason[e] … reason[e] = r`). No Lab autoreset (`_reset_envs_from_buffer` returns None, `:617-618`). Strict mode unchanged. |
| Finding 10: classification | **PASS** | Caller errors → `INFRA_PYTHON_EXCEPTION` or a named infra class (`supervision.py:350-352`). Pool overflow tested before the task signature (`termination.py:213-217`), as in `classify_exception`. Legacy bit 256 → infra on both splits. No new class. |
| Finding 12: rollout v2, metadata | **PASS** (one LOW note) | `rcs_rl_rollout_v2` is opt-in. The checker gives identical v1 results on two preserved rollouts. `privileged_info_keys` is on both backends and contract-checked. LOW: finding 6. |
| Frozen physics, thresholds, records | **PASS** | §1. |
| CI | **PASS_WITH_LIMITATIONS** | The concurrency block is correct and the run succeeded. The branch still lacks `6ddca19` / `c292cf2` / `d3d9dee` (finding 4). It also cancels runs on `main` (finding 5). |
| Naming, no artifacts | **PASS** | No `artifacts/`, `.npz`, `.log` or `.xml` committed. `tests/test_repo_naming.py`: 2 passed at `5d05dd6`. |

## 3. What I re-derived (holds unless a finding says otherwise)

### 3.1 The G9 window, from raw data

Which counter is which:
- `run_perf_acceptance.py` records `P3.stats_counts(env)` per step. That is stats slot 4 (`run_p1_3a.py:120-122`).
- Slot 4 is incremented only by the particle–particle contact kernel (`batched/tlmpm_v7_2_batch.py:318-322`), for
  pairs across a cut or from distant reference positions.
- Tool–material contacts are slot 1 (`_rigid_contacts_b`, `:209-212`). Slot 1 is not dumped.
- Native records both slots per control step: `privileged_diagnostics.material_contact_statistics` in the WP-17
  `actions.jsonl` (`tlmpm_v7_2_contacts.py:144`, `tlmpm_v5.py:495`).

| s512 | Particle contact (slot 4) | Tool contact (slot 1) | Labels |
| --- | --- | --- | --- |
| whole episode | 315–364, 49 steps | 229–366, 101 non-zero steps | 0 at 250, 1,978 at 260, 37,958 from 380 |
| v2 window 321–400 | 44 steps, 93.8 % of the total | **46 steps, 29.3 % of the total** | +16,588 (43.7 %) |
| untimed 229–320 | 5 steps | **55 steps** | 21,370 (56.3 %) |
| v1 window 21–120 (N=8/16) | 0 | 0 | 0 |
| v1 window 21–320 (N=1) | 5 | 55 | 21,370 |

Sources: the slot-4 rows from the `smoke_003` G6 dumps and the GPU-check dumps (identical ranges in all four); the
slot-1 rows from Native `seed_512`.

Per dump:

| Dump | Contact steps in window | Contacts in window (share) | Labels 320 → 400 | `invalid_reason` |
| --- | --- | --- | --- | --- |
| `smoke_003` G6 baseline (`frozen_v1`) | 44 | 143,022 (93.81 %) | 21,370 → 37,958 | 0 |
| `smoke_003` G6 candidate (`alias_free_v1`) | 44 | 143,066 (93.81 %) | 21,371 → 37,958 | 0 |
| `gpu_window_check_001` e0 | 44 | 143,063 (93.82 %) | 21,371 → 37,958 | 0 |
| `gpu_window_check_001` e1 | 44 | 143,110 (93.82 %) | 21,370 → 37,958 | 0 |

Other checks:
- s15 (held out): 79 particle-contact and 80 tool-contact steps in the window.
- The Native samples are every 10 steps, so the label counts at steps 320 and 400 are exact sample values.
- The GPU check ran at `cf04c1a` on a clean tree, after that commit (21:43:41). The config was therefore declared
  before the run.

### 3.2 Judge

**v1 behaviour.** I re-judged `smoke_003_rejudge_001/{config,manifest}.json`, CPU only, into scratch directories:
- with the judge of `c5e96b2` (exported tree);
- with the judge of `5d05dd6`, plus `--known-harness` for the smoke harness pair `cd5450e3…` / `76d0fc98…`.

Result:
- Both: PHYSICS / PERFORMANCE / INTEGRITY PASS, ratios 1.3612 / 1.3692.
- Every gate (id, status, value, threshold, detail) is identical.
- The only differences are the re-judge notes and the re-judge limitation that `--known-harness` adds.
- Without `--known-harness`, the new judge marks every smoke run as an integrity problem. This is by design: the
  harness and `perf_acceptance.py` bytes changed (finding 8).

**`G9:window` on real data.** I used a probe config: the smoke config with the v2 G9 block and `window_check.plans =
["G6_replay"]`.
- `G9:window` PASSes: baseline 44 steps / +16,588, candidate 44 / +16,587.
- The G9 runs timed on the old window are refused as integrity problems.
- `episode_rows` keep only dump rows without problems. `series` is not serialized into the verdict (`combine` copies
  only `checks` / `values`).

### 3.3 Budget arithmetic

Inputs, from the `exclusive_001` walls:
- `alias_free_v1` median step: N=1 0.0978 s, N=8 0.5709 s, N=16 1.1799 s;
- per-process overhead: 13.0 s (N=1) and 14.8 s (N=16).

| Item | Value |
| --- | --- |
| G9 run, N=1 | 52 s |
| G9 run, N=16 | 487 s |
| G9 total | 6 × 539 s = 53.9 min |
| G9 N=16 hold | 48.7 min |
| v1 design, G9 | 130 min |
| 3 pairs with N=8 | 78 min |
| Physics plan | 26.1 + 12.1 + 3.8 + 2.5 = 44.5 min |
| One acceptance | 1.64 GPU-h |

The revision factor:
- GPU check: mean step wall over steps 1–400 / mean approach step = 1.061. The phase medians match the report.
- 1.64 × about 1.06–1.08 ≈ 1.75 GPU-h. 2.5 / 1.75 = ×1.43, and 90 / 52 = ×1.73.

### 3.4 Findings 3, 4, 10 and 12

- **Finding 3.**
  - `lab_fidelity` (`lab_vector_env.py:60-78`) uses `reference_reasons`, the helper workload A now shares
    (`vector_env.py:104-111`; A's behaviour is unchanged).
  - `LAB_PHYSICS_DEFAULTS` equals the `BatchedLabOptions` defaults (tested).
- **Finding 4.**
  - The latch masks per world: `failed = mask & ~passed`.
  - The vector wrapper's full reset calls `env.reset()`, which restores the snapshot through `reset_envs(all)`
    (`lab_vector_env.py:384-387`). This is the path the probe tests.
  - `grasp_initialization` is initialized to None for B1 (`lab_batched_env.py:245`).
- **Finding 10.** The split-specific branch only affects `termination_reason_v1`. Its refinement classes are in
  `long_rollout_v2`'s `process_classes` (tested).
- **Finding 12.**
  - `load_rollout` reads both schemas.
  - The checker of `c5e96b2` and that of `5d05dd6` give identical item results on the preserved v1 rollouts
    `p1_5/smoke_s_trim_n8_002` and `p1_5/smoke_s_cut_n4_001`. Both PASS; the new one adds `schema` and
    `field_roles: UNLABELLED_V1`.

### 3.5 External adapter (read-only)

- `core.config_problems` returns `[]` for v1 and for v2.
- Its acceptance driver reads `steps_by_num_envs` with an `or {}` fallback, so the key's absence in v2 is harmless.
- Its dev limits cap only `--steps` (K), so dev runs with W=320 are allowed.

## 4. Findings (most severe first)

### 1 — MINOR — the v2 "contact" figures count particle–particle contacts; the window holds under half of the tool-contact and cutting phase

**Where.**
- The counted quantity is stats slot 4 (§3.1), described as tool contact in:
  - D-14 (`RL_READY_PLAN.md:197`): "s512 contact starts at step 315", "44 of the 49 contact steps";
  - `P1_FOLLOWUP.md:12` (status: "G9 times tool–material contact"), and the §1.1 rationale at `:33-46`;
  - the judge docstring (`judge_perf_acceptance.py:27-28`: "must hold tool-material contact") and
    `perf_acceptance.py:9`.
- The config's own window text (`acceptance_v2.json:44`) correctly says "particle contacts".
- Its description (`:4`, "times tool-material contact and cutting") is literally true: 46 tool-contact steps.

**What happens.**
- The window 321–400 was placed after the **particle**-contact onset (315).
- Tool contact starts at step 229, and cutting (labels) at 251–260. So the untimed warm-up holds:
  - 55 of the 101 tool-contact steps;
  - 56 % of the labels;
  - the whole cutting onset.
- 34 of the 80 timed steps (367–400) have no tool contact.
- At N=1 the v1 window (21–320) held more tool contact (55 steps) and more labels (21,370) than v2 does (46 and
  16,588). "v1 timed no contact" holds for N=8/16 only. That was the original finding's scope.
- The rationale for leaving 315–320 untimed ("one-time first-contact work, contact-pool growth, lazy allocations") is
  not measured. In training, every episode has its own contact onset after a reset. So onset work is per-episode
  workload, unless it is a per-process allocation.
- The GPU check shows step walls of 2.0–3.7× the approach median at 255–258 and 2.3–2.7× at 274–275 (cutting
  onset, `SHARED_GPU`). Their cause is not established.

**An alternative the design did not consider.**
- A G9 run costs W+K = 400 steps whatever the split.
- For example, W=228, K=172 has the same GPU cost and times:
  - all 101 tool-contact steps;
  - all cutting;
  - all 49 particle-contact steps;
  - with 2.15× the timed steps, which also lowers the noise in finding 2.
- The config's statement that "a window inside the cutting phase needs ≥ 314 replay steps" follows from the chosen
  start, not from the cutting phase.

**Scenario.** A candidate adds work only around the cutting onset, for example first frontier or slot-claim work in
229–320, and saves at least as much in 321–400. It passes `G9:n16`, because G6/G7 check physics, not speed.

**Mitigation.**
- At N=16 (primary), v2 is a strict improvement: from 0 to 46 tool-contact steps, 44 particle-contact steps and
  16,588 labels.
- The tool-contact kernels that run in 229–320 also run in 321–366, so their per-step cost is timed.

**Recommendation.**
- Before the PR, correct the wording in D-14, the report (status row and §1.1) and the two docstrings. The counter is
  particle–particle (cut-face) contact; give the tool-contact figures from Native slot 1 as in §3.1.
- Then choose one, before `acceptance_v2` freezes at its merge:
  - **(a)** keep the window and carry the coverage as a PR limitation (§7 item 2); or
  - **(b)** declare a window that starts at tool contact (for example W=228 / K=172), with a new pin, updated
    D-14 and `G9:window` thresholds, a re-run of the GPU window check, and a short re-review of that config diff.
- No candidate has been judged under v2, so (b) is not score tuning. After the merge, any change is
  `acceptance_v3`. Either choice is acceptable for the merge.

### 2 — MINOR — "stricter than v1" holds only for the sign-test part; near-threshold errors and the N=1 floor got worse

**Where.** `acceptance_v2.json:36,45,49`; `P1_FOLLOWUP.md:71-75`; D-14.

**Model.**
- The pair log-ratios are independent and identically distributed, N(ln r, σ²).
- Primary rule: median ≥ 1.05 and at least m pairs > 1 (v1: 5 pairs, m = 4; v2: 3 pairs, m = 3).
- Measured noise:
  - an exclusive repeat (`exclusive_001` `a_n8_s15_default` vs `_r2`: 9.296 vs 9.352 env-steps/s) gives σ ≈ 0.006;
  - the shared ABBA smokes (`smoke_002` / `smoke_003`, N=1/2) give pair-to-pair |Δ ln| of 0.003, 0.029, 0.127 and
    0.006, so σ ≈ 0.012 without the outlier and ≈ 0.046 with it.

**P(accept), v1 / v2, by true speed ratio r and pair noise σ:**

| σ | r = 1.00 | 1.03 | 1.04 | 1.06 | 1.08 | 1.10 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 0 / 0 | 0.000 / 0.002 | 0.037 / 0.076 | 0.962 / 0.922 | 1.000 / 1.000 | 1 / 1 |
| 0.02 | 0 / 0 | 0.036 / 0.069 | 0.185 / 0.229 | 0.813 / 0.759 | 0.996 / 0.982 | 1 / 1 |
| 0.03 | 0.001 / 0.004 | 0.111 / 0.135 | 0.270 / 0.276 | 0.721 / 0.652 | 0.960 / 0.909 | 0.998 / 0.987 |
| 0.05 | 0.023 / 0.032 | 0.203 / 0.180 | 0.324 / 0.270 | 0.611 / 0.491 | 0.840 / 0.706 | 0.954 / 0.860 |

What the table shows:
- Under no speed change, both false-accept rates are about 0, because the 1.05 median floor binds. The 1/8 vs 3/16
  comparison matters only when σ ≫ 0.05.
- For σ ≤ 0.02, v2 roughly doubles the near-threshold errors in both directions.
- For σ ≥ 0.03, v2 loses power: a true 1.10 is rejected 14 % of the time at σ = 0.05, against 5 % for v1.

**The N=1 floor.**
- v1 timed 300 steps at N=1 (`steps_by_num_envs`); v2 times 80 steps, about 8 s, from 3 pairs instead of 5.
- I assumed the noise scales with √(300/80) = 1.94. Under that assumption:

| σ (v1 window) | P(neutral candidate fails the floor), v1 / v2 | P(5 % regression passes the floor), v1 / v2 |
| --- | --- | --- |
| 0.010 | 0.0000 / 0.010 | 0.000 / 0.054 |
| 0.015 | 0.0001 / 0.059 | 0.005 / 0.141 |
| 0.020 | 0.0024 / 0.120 | 0.026 / 0.210 |

**N=8.** With N {1, 16}, a regression at any N other than 1 and 16 is not gated. For example, launch parameters
tuned for N=16 could regress N=8. The saturation argument is a property of the baseline, not of a candidate.

**Recommendation.**
- Replace "stricter than v1" in D-14 and the report with the trade-off above. The frozen config text stays; D-14
  records the correction.
- Put the summary into the PR limitations.
- Run acceptances on an `EXCLUSIVE_GPU` when possible. At σ ≈ 0.006 every effect above is negligible.
- Option (b) of finding 1 (K=172) would also reduce the N=1 noise.

### 3 — MINOR — figures in the report and D-14

- **"N=1 … costs 5 % of G9"** (`acceptance_v2.json:45`, `P1_FOLLOWUP.md:75`). N=1 costs 6 × 52 s = 312 s of 3,234 s,
  which is 9.6 % of G9 (5.3 % of one acceptance).
- **"reads … `steps_by_num_envs`, both of which v2 keeps"** (`P1_FOLLOWUP.md:141`). v2 has no `steps_by_num_envs`;
  the test asserts its absence (`tests/test_perf_acceptance.py:111`). This has no functional effect (§3.5).
- **D-14 carries only the pre-check budget** (1.64 GPU-h, ×1.5, 49 min, ×1.85). The revision (about 1.75 GPU-h,
  ×1.43, about 52 min, ×1.75) is only in the report.
- **The budget is GPU time only.** The G1/G2 CPU suites (`cpu_tests.args ["tests"]`, 50–53 min per side on this host)
  add wall time. The local runner runs them one side after the other.
- **The 1.33× / 6 % contact factor** comes from one N=2 `SHARED_GPU` episode-mode run. The contact-phase cost at
  N=16 is unmeasured.

**Recommendation.** Correct these in the report and D-14, documentation only.

### 4 — MINOR (merge condition) — the branch does not contain `6ddca19`, `c292cf2`, `d3d9dee`

- `claude/rl-phase1`'s later commits make the live-chain pose-interpolation tests NOT_RUN on runners without
  AVX-512.
- CI on `5d05dd6` passed, but on a runner without AVX-512 these tests would fail.
- **Recommendation:** merge `main` after PR #12 merges, rerun CI, then open PR #13.

### 5 — LOW — the CI concurrency group also cancels in-progress runs on `main`

`.github/workflows/tests.yml:4-6`:
- The group is per workflow and ref, so a branch push and its PR run do not cancel each other: the refs differ,
  `refs/heads/…` vs `refs/pull/N/merge`. That is correct.
- But on `main`, two merges in quick succession leave the first merge commit without a completed CI run.
- **Suggestion:** `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`.

### 6 — LOW — `PRIVILEGED_INFO_KEYS` omits the done flags derived from the oracle termination

- `info["autoreset"]` and `info["terminal_captured"]` (`vector_env.py:939, 944`) equal the
  `termination_reason`-derived done flags. They are not listed in `vector_contract.py:23-25`.
- The contract check (`:98`) only compares against keys in `STEP_INFO`, so it cannot catch a missing optional key.
- **Recommendation:** list them, or state in the metadata that the done flags are training signals, as the rollout's
  `oracle_derived` already does for `terminated`.

### 7 — LOW — stale pointers outside this branch's diff

- `RL_READY_PROGRESS.md:34` (U2) still states the `acceptance_v1` rule (4 of 5 pairs, N 1/8/16).
- `README.md:288-289` says the first acceptance "needs `acceptance_v2`".
- After the merge, both should name `acceptance_v2` and its limitations.

### 8 — INFO — the harness and shared-module bytes changed

- `run_perf_acceptance.py` and `perf_acceptance.py` differ from `c5e96b2`.
- So the judge refuses earlier outputs as integrity problems unless it is given `--known-harness` (re-judge only).
- The first v2 acceptance therefore needs a baseline and evaluator checkout that contain this branch.
- `HARNESS_VERSION` stays `perf_acceptance_v1`. The new result keys are additive.

### 9 — INFO — `acceptance_v1`'s `G9:n1:work` would fail at random; v2's does not

- At step 320 the labelled count differs by one particle between runs, and between env slots of the same run:
  21,370 vs 21,371 in `smoke_003` B vs C and in GPU-check e0 vs e1.
- `acceptance_v1`'s N=1 G9 run ends at step 320, and `G9:n1:work` requires equal final counts.
- v2's runs end at step 400, after the labels saturate at step 380 (37,958 in all four dumps).
- This is another reason never to use v1. No action on this branch.

## 5. Not changed (checked)

- **Thresholds and plan.** Thresholds other than `primary_pairs_above_1`, the physics tolerances, the plan, the seeds,
  the reference variant, the evaluator file set, `frozen_paths`, the dev limits and the timeouts equal v1, key by key
  (test and my semantic diff).
- **Timeout.** `run_timeout_s` 1800 s covers G9 N=16 (≈ 516 s exclusive) up to about ×3.5 co-tenant slowdown.
- **Evidence and records.** No evidence file or record of an earlier run is changed by the follow-up commits. The
  three evidence-file edits in the range are the additive blocks of the merged `7ac820b` / `18ee275`.
- **Experiment outputs.** `ci_sim_full_001` (the failed CI simulation) is preserved, and no output directory was
  overwritten.

## 6. Tests run by this review (CPU only; never GPU1)

```bash
cd /home/ningan/RoboticCuttingSim-rl-fureview   # detached at 5d05dd6
SEL="tests/test_acceptance_frozen.py tests/test_perf_acceptance.py tests/test_rl_lab_fidelity.py \
  tests/test_rl_rollout_roles.py tests/test_rl_smoke.py tests/test_rl_supervision.py tests/test_rl_termination.py \
  tests/test_rl_lab_env.py tests/test_rl_lab_grip_profile.py tests/test_rl_lab_ik.py tests/test_rl_lab_kernels.py \
  tests/test_rl_lab_rigid_isolation.py tests/test_rl_lab_scene_usd.py tests/test_vector_env_contract.py \
  tests/test_repo_naming.py tests/test_batched_lab_coupling.py tests/test_long_rollout_config.py \
  tests/test_rl_stability.py tests/test_rl_consolidation_cpu.py tests/test_rl_configured_family.py \
  tests/test_rl_position_band.py tests/test_vector_env_consolidation.py tests/test_vector_env_p12b.py \
  tests/test_vector_env_api.py"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider -rs $SEL
# 285 passed, 10 skipped in 70.9 s -> PASS (skips: Torch / Isaac Lab modules, NOT_RUN in .venv)
flock -w 900 /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m pytest -q -p no:cacheprovider -rs $SEL
# 285 passed, 10 skipped in 71.0 s -> PASS (identical skip set)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python \
  -m pytest -q -p no:cacheprovider -rs tests/test_rl_smoke.py tests/test_perf_acceptance.py \
  tests/test_acceptance_frozen.py tests/test_rl_rollout_roles.py tests/test_rl_supervision.py \
  tests/test_rl_termination.py tests/test_rl_lab_fidelity.py tests/test_vector_env_contract.py \
  tests/test_vector_env_api.py tests/test_vector_env_p12b.py tests/test_vector_env_consolidation.py \
  tests/test_rl_consolidation_cpu.py tests/test_rl_configured_family.py tests/test_long_rollout_config.py \
  tests/test_rl_stability.py
# 268 passed, 3 skipped in 857.3 s -> PASS (skips NOT_RUN: workload B contract needs the Isaac Lab venv (2);
# WP-12 Demo 1 recording not present (1)); incl. the v2 rollout tests and the CPU harness end-to-end with --config
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaaclab-venv/bin/python \
  -m pytest -q -p no:cacheprovider -rs tests/test_rl_lab_env.py tests/test_rl_lab_grip_profile.py \
  tests/test_rl_lab_fidelity.py tests/test_rl_lab_ik.py tests/test_rl_lab_kernels.py \
  tests/test_rl_lab_rigid_isolation.py tests/test_rl_lab_scene_usd.py tests/test_vector_env_contract.py \
  tests/test_batched_lab_coupling.py
# 72 passed, 0 skipped in 381.4 s -> PASS (incl. the finding-4 latch test and the finding-3 dt 2e-4 Lab test)
```

**CPU re-derivations (no GPU):**
- `judge_perf_acceptance.py` of `c5e96b2` and of `5d05dd6` on `smoke_003`, plus the v2 probe config (§3.2);
- `check_rl_rollout.py` of both commits on two preserved v1 rollouts (§3.4);
- the adapter's `config_problems` (§3.5).

**Not run by this review:**
- the full CPU suite (recorded by the author at `2195bb2`: 2494 passed, 37 skipped, 6 xfailed, 0 failed, with CUDA
  hidden and with GPU0 visible);
- any GPU job.

The naming test was run after `git add` of this file (see the commit).

## 7. Limitations that must appear in the PR #13 body

1. **`acceptance_v2` is declared, not exercised.**
   - No full acceptance under v2 has run.
   - Its GPU time is an estimate: about 1.75 GPU-h exclusive after the contact-phase revision, plus CPU suites.
   - The contact-phase cost at N=16 is unmeasured.
   - v2 freezes at its merge commit into `main`; after that, any change is `acceptance_v3`.
2. **What the G9 window times.**
   - Steps 321–400 at every N.
   - Particle–particle contact: 44 of 49 steps (94 %).
   - Tool–material contact: 46 of 101 steps (29 % of the tool-contact count).
   - Cutting: 60 of about 130 steps (44 % of the labels).
   - 34 timed steps without tool contact.
   - Untimed: the approach (1–228), and the tool-contact and cutting onset (229–320, 55 tool-contact steps, 56 % of
     the labels).
   - (Unless option (b) of finding 1 is taken before the merge.)
3. **What `G9:window` checks.**
   - It runs on the G6/G7 episode-mode processes, not inside the G9 throughput processes.
   - It counts particle–particle contacts and new labels, not tool contacts.
   - Its margin is 40 steps against 44 measured.
4. **Statistics.**
   - 3 pairs, N {1, 16}. This is not stricter than v1 as a whole (finding 2).
   - On a shared GPU, a neutral candidate fails the 80-step N=1 floor with a few-percent to about 12 % probability.
   - N=8, and every N other than 1 and 16, is not gated.
   - Acceptances should run on an `EXCLUSIVE_GPU` where possible.
5. **Candidate-reported values.** VRAM peak, reset p95 and `performance.phases` are measured inside the process
   under test. The phases are descriptive.
6. **`acceptance_v1`** is frozen (`3592d940…`) and must not be used. Besides finding 1, its N=1 work gate would fail
   at random (finding 9).
7. **The external adapter** passes no `--config` to the harness and has not been switched to v2. That is the other
   agent's change.
8. **`rcs_rl_rollout_v2` is opt-in.** Declared runs (`rl_smoke_v1`, the long rollouts) keep the unlabelled v1.
   `info["autoreset"]` / `terminal_captured` are not in `privileged_info_keys` (finding 6).
9. **B2 test coverage.** The settle latch is tested on Warp CPU, with a test-only flag injection for one world. The
   Phase 1 B2 limitations otherwise remain: G1/G3, G4 and G5 with the profile are NOT_RUN.
10. **Not addressed here:** findings 5–9 and 11 of the Phase 1 review.

## 8. Go / no-go

**GO for PR #13**, after PR #12, under these conditions:

1. Merge `main` (with PR #12 and `6ddca19` / `c292cf2` / `d3d9dee`) into the branch, and get CI green on the
   result (finding 4).
2. Correct the documentation of findings 1–3: D-14, `P1_FOLLOWUP.md` and the two docstrings. This needs no re-review
   if the diff is documentation and docstring text only.
3. Decide the window before the merge: (a) keep it, with §7 item 2; or (b) declare a new window, with a new pin, a
   re-run of the GPU window check and a short re-review of the config diff.
4. The PR body carries the §7 list, or a faithful equivalent, and links this report.

**No-go** for a P1-6 acceptance under `acceptance_v2` before it is merged into `main` (its freeze point).

**Follow-ups, not blocking:** findings 5–7.
