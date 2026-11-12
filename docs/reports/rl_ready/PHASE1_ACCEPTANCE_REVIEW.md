# Phase 1 — xhigh acceptance review (2026-11-10)

**Commit reviewed: `f03fbff`** (branch `claude/rl-ready-rubber-physics`), read-only, in the detached worktree
`/home/ningan/RoboticCuttingSim-rl-review`. The branch head moved to `644d9aa` while the review ran. Its two
commits, `9289328` and `644d9aa`, change only `docs/RL_READY_PROGRESS.md`, and the progress-file line numbers below
refer to `644d9aa`. Every other `file:line` refers to `f03fbff`.

**Scope.**
- Everything merged since the earlier review at `3682786` (`git log 3682786..f03fbff`, 97 commits):
  - P1-1b B2 grasp follow-up (`mjwarp_grip_v1`, `material_input_domain_v1`);
  - P1-2b (`isaac_lab_newton` dispatch, `tasks=`, `termination_reason`, evaluation stream, input guard, terminal
    capture);
  - P1-4 (supervisor, long rollout);
  - P1-5 (CUT, RL smoke, parity matrix);
  - R0 (P1-6 harness, judge, `acceptance_v1`);
  - the fixes for the earlier review's findings (its §7);
  - the consolidation pass;
  - the FallbackGrid device fix;
  - the formal `EXCLUSIVE_GPU` performance table.
- Phase 1 as a whole, against plan §4 and the specification's "Phase 1 Done" checklist.

**Method.**
- I read the plan (§1, §2, §4, §5 D-01…D-12), the specification (P1-0…P1-6, Phase 1 Done), the progress file,
  AGENTS.md, the README section, every report and evidence file under `docs/*/rl_ready/`, ADR 0021 and ADR 0022,
  and the code under `src/robotic_cutting_sim/{batched,rl}/`.
- I recomputed the claims that carry the verdict from the raw artifacts in
  `/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/` (§5).
- A second, independent reader reviewed the workload-B backend and the P1-4 supervisor code. I re-read the code
  for each of its findings before accepting it.
- CPU tests only (§6). No GPU job was run, and nothing outside this file was edited.

## 1. Overall verdict

**PASS_WITH_LIMITATIONS. Merge to `main`: GO, under the conditions in §8.**

- No BLOCKER.
- No frozen solver, evaluator, goal, scene config, v1 threshold or record is modified. `git diff --name-status
  origin/main...f03fbff` modifies only `README.md`, `pyproject.toml` (pytest `pythonpath`) and three
  `.claude/agents/*.md` files; every other change is a new file.
- No geometric deletion is presented as cutting.
- No oracle or force state enters the deployable policy observation.
- Failed runs are preserved.
- The `EXCLUSIVE_GPU` labels reproduce exactly from the raw samples.

Four MAJOR findings remain:
- two concern the P1-6 contract `acceptance_v1` (findings 1 and 2);
- two are workload-B defects on opt-in paths that no recorded evidence used (findings 3 and 4).

Phase 1 is **not "done"** after this merge. Checkbox 7 (external optimizer sync) is PENDING, so after the merge
the status is `MERGED_RCS_PENDING_OPTIMIZER_SYNC` until the external descriptor is pinned (SYNCED).

## 2. Per-package verdicts

| Package | Verdict | Basis (limitations by finding number) |
| --- | --- | --- |
| P1-1b B1 (rigid mount), workload B N worlds | **PASS_WITH_LIMITATIONS** | **Evidence:**<br>- Batched articulation plus batched material at N=1/4/8/16 with the delayed wrench. Frozen evaluator PASS in every env of every exclusive B1 run (33/33; 37/37 with B2). Labelled counts are identical across N.<br>- G1 per-world isolation is bitwise (deterministic mode).<br>- One device HashGrid per process.<br>**Limitations:**<br>- `reduced_fidelity: rigid_tool_mount_no_grasp`.<br>- The "Native confirmation" uses frozen single-material replays, not Kit (N2).<br>- No Lab CUT (T5.9 NOT_RUN). |
| P1-1b B2 with `mjwarp_grip_v1`, and `material_input_domain_v1` | **PASS_WITH_LIMITATIONS** | **Evidence:**<br>- S3 TRIM N=2/4 and exclusive N=4: evaluator PASS 4/4.<br>- The grasp holds 1196/1200 steps. Max slip is 1.51–1.59 mm and max rotation 14–15 mrad, against about 1 µm in Native.<br>- The opt-in profile is listed in `fidelity()`.<br>- The guard is device-side and per-env; in-domain envs are bitwise unchanged (CPU tests).<br>**Limitations:**<br>- Finding 4: a failed settle is forgotten in non-strict mode.<br>- Finding 3: dt is not checked by `fidelity()`.<br>- B2 G1/G3 deterministic, G4 and G5 with the profile: NOT_RUN.<br>- The SIGSEGV guard rests on one paired N=4 attempt. |
| P1-2b vector API | **PASS_WITH_LIMITATIONS** | **Holds:**<br>- One contract for both backends (`rl.vector_contract`).<br>- `tasks=` (TRIM seeds and configured scenes; mixed layouts refused with codes).<br>- `termination_reason_v1` mapping, test-iterated over `INVALID_REASONS`.<br>- Evaluation stream (opt-in), terminal capture, `FAIL_INTEGRITY` for non-finite envs.<br>- The input guard is default-on with an `input_guard=False` rollback.<br>- Device-only masks; no per-env loop on the step path.<br>**Limitations:** findings 3, 12, N3. |
| P1-4 supervisor and long rollout | **PASS_WITH_LIMITATIONS** | **LR-A PASS, re-checked from raw:**<br>- 64 full-length episodes;<br>- canaries 16/16 bitwise = Native;<br>- own VRAM peak 1,412 MiB;<br>- foreign SM mean 34.8 %.<br>**Other runs:** LR-C PASS. LR-D 5/5 PASS. **LR-B FAIL** on its own B0 bound (2,304 MiB, re-checked).<br>**Limitations:**<br>- LR-A2 and LR-E NOT_RUN.<br>- Every run is `SHARED_GPU`.<br>- LR-A ran pre-F1 code (finding 5).<br>- Finding 10. |
| P1-5 CUT, RL smoke, parity matrix | **PASS_WITH_LIMITATIONS** | **CUT, batched v7_2 scene:**<br>- 35/35 reference episodes PASS, re-checked from raw (the 4 FAIL_INTEGRITY entries of M1 are truncated random-policy segments).<br>- Exclusive N=1/8/16: 33/33.<br>**CUT row is PARTIAL by design:**<br>- the frozen v5 CUT is not batched (`UNSUPPORTED_SOLVER`);<br>- batched CUT is ideal tracking with no robot;<br>- the batched replay of the Native v7_2 recording is NOT_RUN.<br>**RL smoke:** PASS (no training).<br>**Matrix:** text issues in findings 9 and 12. |
| R0: harness, judge, `acceptance_v1` | **PASS_WITH_LIMITATIONS** for merging the tooling. **The contract is not fit for a first optimizer acceptance** until findings 1 and 2 are resolved (as `acceptance_v2`). | The harness and judge work as specified. `tests/test_perf_acceptance.py` passes in `.venv` and in the Torch venv (§6), and the GPU smoke and re-judge are recorded. Finding 1: the G9 window times no cutting at N=8/16. Finding 2: the contract was amended in place while declared frozen. |
| P1-6 sync (Phase 1 Done checkbox 7) | **PENDING** (correctly unticked) | **Recorded:**<br>- No candidate has been accepted.<br>- No external pin exists.<br>- The docs say PENDING and `MERGED_RCS_PENDING_OPTIMIZER_SYNC` "at best after a merge" (`P1_PARITY_MATRIX.md:118-124`).<br>**Not claimed anywhere:** SYNCED, a PR as a merge, or an accepted candidate. |
| Earlier review fixes (F1–F10, notes) | **PASS** | **F1:**<br>- the per-launch own-grid rebuild and the step-time census are in `material_batch.py:925-934`;<br>- the sticky bit 256 is in the receipt.<br>**F2:** band v2 was pre-declared, with 20 references.<br>**F3/F4/F6/F8/F9:** corrected additively.<br>**F5:** terminal capture and the `NOT_EVALUATED` refusal.<br>**F7:** diagnostics opt-in.<br>**F10:** `importorskip("pxr")`.<br>**N5:** pins added.<br>**Still open:** N8 (finding 12) and N9. |
| Consolidation (3674780) | **PASS_WITH_LIMITATIONS** | **Holds:**<br>- grip profile passthrough with codes;<br>- `FAIL_INTEGRITY` in `evaluate()`;<br>- one material→termination mapping, including bits 128 and 256;<br>- configured `tasks=`.<br>**Limitations:** doc inaccuracies (findings 7, 9) and the supervisor gaps (finding 10). |
| FallbackGrid device fix (db036a0) | **PASS** | Every launch is pinned to its own device, and only `alias_free_v1` uses this path. `test_fallback_grid_reproduces_warp_grouped_query_sequence` passes with GPU0 visible (§6). |
| Formal `EXCLUSIVE_GPU` table (8e503e1) | **PASS_WITH_LIMITATIONS** | **Holds:**<br>- 20 EXCLUSIVE / 3 SHARED, reproduced exactly from the raw telemetry, the chain sampler and the chain logs (§5);<br>- every plan column is present;<br>- the boundaries are never merged;<br>- reduced-fidelity labels are in §6 of the table and on every evidence row.<br>**Limitations:** findings 5 and 11. Workload-A windows are pre-contact at N ≥ 4. |
| **Phase 1 overall** | **PASS_WITH_LIMITATIONS** | Checklist status in §3. Merge conditions in §8. |

## 3. Plan §4 criteria and the "Phase 1 Done" checklist

### 3.1 Plan §4 criteria

| Item | Status found | Evidence checked |
| --- | --- | --- |
| P1-1b | **N=1/4/16 batched articulation with material coupling and the delayed wrench:** MET. B1 runs at N=1/4/8/16 in the exclusive window; N=16 is SHARED (0 % foreign SM).<br>**Per-env wrench isolation:** MET (G1 bitwise, B1).<br>**Native confirmation:** frozen single-material replays reproduce the batched final codes (G5 L3, B1). That is not a Kit run (N2).<br>**B2:** PARTIAL (see §2). | `exclusive_001/b1_n{1,4,4_r2,8,16}/row.json`; `p1_1b_s2_s3_001.json`; `p1_1b_b2_001.json`; raw `s3_b2grip_n{2,4}_001` |
| P1-2 / P1-2b | **API tests:** MET (shapes, dtypes, device, no per-env CPU round trip, deployable observation without oracle terms).<br>**M0/M1/M2:** MET, as found by the earlier review. | §6 tests; `observations.py:30-79` |
| P1-3 | **Opt-in `alias_free_v1`:** 1.38–1.47× in EXCLUSIVE pairs, pre-contact windows (N ≥ 4).<br>**Rollback:** `frozen_v1` is the default.<br>**RUN_TO_RUN:** not done. | `P1_PERFORMANCE_TABLE.md:250-262` |
| P1-4 | **Long rollout with partial resets, bounded VRAM and RSS, 8+ serial episodes, infrastructure classified apart from task failures:** MET on LR-A.<br>**LR-B:** FAIL on its B0 bound.<br>**Native limitations:** retained. | raw `p1_4/lr_a_001`, `lr_b_001` (§5) |
| P1-5 | **Matrix:** 9 PASS / 6 PARTIAL / 1 UNSUPPORTED.<br>**Batched CUT and TRIM with the independent evaluator:** MET for TRIM. CUT is PARTIAL.<br>**RL smoke:** MET. | raw `p1_5/m{0,1,2}_cut_*` |
| P1-6 | **R0 only.** The optimizer cycle is NOT_RUN. | — |

### 3.2 The "Phase 1 Done" checklist

| Phase 1 Done checkbox | Ticked in the matrix | This review |
| --- | --- | --- |
| 1. True GPU batching at N=1/4/8–16 with partial reset; throughput / memory / latency | [x] | **MET with limitations.**<br>- EXCLUSIVE rows for A N=1/4/8/16, CUT N=1/8/16 and B1 N=1/4/8.<br>- B1 N=16 is SHARED only.<br>- No EXCLUSIVE partial-reset latency and no batched cold rows (finding 11).<br>- Measured at `691b571` (finding 5). |
| 2. Representative CUT/TRIM run in RL headless mode with success/failure evaluation; gaps explicit | [x] | **MET under the checkbox's "gaps explicit" wording.**<br>- CUT stays **PARTIAL**: the original frozen-v5 CUT is not batched.<br>- By specification §9.6, "whole-RCS RL-ready" therefore stays PARTIAL for CUT. That must be stated in the PR. |
| 3. Single-env reference vs batched numeric and cut-history regression; frozen-profile regression | [x] | **MET with limitations.**<br>- L0/L1 CPU bitwise for TRIM and CUT.<br>- The GPU position band is descriptive.<br>- L4 RUN_TO_RUN is BLOCKED.<br>- The batched replay of the Native v7_2 recording is NOT_RUN.<br>- Frozen sources are pinned and unchanged. |
| 4. GPU-native action/obs/reward/done, per-env seeds, reproducible scenes, API tests | [x] | **MET with limitations.**<br>- No per-episode geometry or material randomisation inside a live batch (row 9 PARTIAL).<br>- Env seeds do not drive physics. |
| 5. Long rollout, OOM/memory/reset stress, real GPU telemetry | [x] | **MET with a recorded FAIL.** LR-B, the declared reset-stress run, FAILed B0. The tick should say so (finding 9). |
| 6. Headless / vision / Native speeds reported separately; no fake comparison | [x] | **MET.**<br>- The three boundaries are separate rows.<br>- Batched vision is UNSUPPORTED, not estimated.<br>- No B-vs-Native ratio is given. |
| 7. External optimizer results written back and synced; a PR alone is PENDING | [ ] PENDING | **PENDING (correct).** |

## 4. Findings (most severe first)

### 1 — MAJOR — `acceptance_v1`'s G9 speed gate never times tool–material contact at N=8/16

**Where.**
- `configs/rl_benchmark/acceptance_v1.json:36-37` (`performance`: warm-up 20, steps 100, 300 at N=1, seed 512,
  primary N=16) and `:64-65` (`baseline_measure`).
- WP-17 contact starts at control step 314 (s512) and 321 (s15): `P1_6_R0_ACCEPTANCE_HARNESS.md:108`,
  `P1_PERFORMANCE_TABLE.md:156-160`.

**What happens.**
- G9 times replay steps 21–120 at N=8/16. These are entirely pre-contact, with 0 labelled particles at the end of
  the window.
- At N=1 it times steps 21–320, which include only the first ~6 contact steps.
- The same document knew the onset: it moved G4/G8 from 300 to 400 steps for that reason. G9 kept the
  approach-only window.
- Neither D-11 nor the R0 report says that the speed gate excludes cutting.

**Scenario.** A candidate makes the approach phase faster and the cutting phase slower, for example extra
frontier, pending-link or law-8 work after first contact. It passes `G9:n16` (median ≥ 1.05, ≥ 4/5 pairs > 1),
because G6 and G7 check only the physics of full episodes, not their speed. The reverse also happens: a candidate
that speeds up only the contact or cutting work cannot pass.

**Mitigation.**
- No candidate has been judged yet.
- For `frozen_v1` the step cost is nearly phase-independent (P1-0 kernel split), so the baseline numbers stand.

**Recommendation.** Before the first optimizer cycle, declare `acceptance_v2` with a G9 window that covers contact
and cutting, for example warm-up to step 300 and time steps 300–700, or time whole episodes. Report the per-phase
speed. Until then, P1-6 must not accept candidates under v1. This is a PR limitation.

### 2 — MAJOR — `acceptance_v1` was amended in place four times while it is declared frozen from its first commit

**Where.**
- The contract says it is frozen once committed:
  - `acceptance_v1.json:4`: "Frozen once committed; any change is a new acceptance_v2";
  - D-11 in `RL_READY_PLAN.md:194`;
  - `P1_6_R0_ACCEPTANCE_HARNESS.md:14`: "DONE (frozen from its first commit)".
- `git log --follow` shows four changes after the first commit `485241d`:
  - `5ea4ce4`: the position rule became descriptive;
  - `3c2642a`: `contact_total_rel_tol` 0.01 → **0.02**, after smoke_003 showed a 2× margin;
  - `214ac11`: reset p95 on a shared GPU went from FAIL to WARN, after smoke_001 FAILed on reset;
  - `ac3f6c6`: the band pointer moved to band v2.
- Unlike `long_rollout_v1`, no test pins the file's SHA-256.

**Scenario.**
- The external optimizer refers to the contract by name, `acceptance_v1`. Its content depends on the commit.
- A re-judge of smoke_001 turned PERFORMANCE FAIL into PASS under the same name.
- Two of the four changes loosened a gate after the data had been seen.

**Mitigation.**
- All four changes came before any baseline or candidate.
- The reset and band changes are recorded in D-11, and the contact-floor change in the R0 report §3.
- The original verdicts are kept, and the re-judges went into separate directories.
- The progress file says the contract "freezes when it enters the Phase 1 merge on `main`".

**Recommendation.**
- Correct the three "frozen from its first commit" statements.
- Record the freeze point in D-11: the merge commit and the SHA-256 of `acceptance_v1.json`.
- Add a pin test.
- Treat any later change, including the finding-1 fix, as `acceptance_v2`.

### 3 — MAJOR — workload B `fidelity()` ignores the material time step (fidelity can be lowered silently)

**Where.**
- `src/robotic_cutting_sim/rl/lab_vector_env.py:461-472`. It checks episode steps and particle spacing only.
- `:204` passes `task.material_dt_s` to the Lab env.
- `src/robotic_cutting_sim/batched/lab_scene.py:209-221`. Its reasons come from the tool mount only.
- Workload A compares `material_dt_s`, `substeps_per_control_step` and five other keys against the Native
  reference (`vector_env.py:1071-1078`).

**Scenario.** `open_vector_env(4, backend="isaac_lab_newton", tool_mount="physical_grasp_v1",
grip_contact_profile="mjwarp_grip_v1", task=TaskFamily(..., material_dt_s=2e-4))` runs 100 material substeps per
control step instead of 200. Yet `fidelity()` returns `reduced_fidelity: False`, with no reason and with
`computed_not_hand_set: True`. The second reader reproduced this on a B2 env (dt 2e-4 and 5e-4). This violates
plan §1, "never lowered silently".

**Evidence impact.** None. Every recorded B row uses 1e-4 s and 200 substeps.

**Recommendation.** Compute the Lab fidelity against the same reference keys as workload A, or refuse a
non-default `material_dt_s` with an `UNSUPPORTED_*` code. Add a test.

### 4 — MAJOR — B2 in non-strict mode forgets a failed pregrasp settle at the caller's first reset

**Where.**
- `lab_vector_env.py:211`: the constructor already calls `env.reset()`, which runs the settle and takes the
  snapshot.
- `lab_batched_env.py:757`: a failed world gets `GRASP_SETTLE_FAILED`, once.
- `:631-635`: every later `reset()` restores the snapshot through `reset_envs(all)`.
- `:819-822`: `reset_envs` zeroes `rigid_reason`.
- `snapshot.passed` (`:773`) is never read.
- `grasp_initialization` appears in neither receipt.

**Scenario.**
1. A caller opens `open_vector_env(4, backend="isaac_lab_newton", tool_mount="physical_grasp_v1",
   grip_contact_profile="mjwarp_grip_v1", strict=False)`, which is training mode.
2. World 3 fails its settle gates.
3. The caller makes the usual first `reset(seeds=...)`.
4. `termination_reason[3]` is now 0. World 3 runs, and may be scored, on a pregrasp that failed.

**Evidence impact.** None. In strict mode, the default and the mode of every recorded B2 run, a settle failure
raises at construction. The code predates `3682786`; the P1-2b wrapper made it reachable through the public API.

**Recommendation.** Keep a per-world settle-failed latch that `reset_envs` re-applies from `snapshot.passed`. Put
`grasp_initialization` into both receipts. Add a test.

### 5 — MINOR — the formal performance rows measured code without review fix F1's per-step grid refresh

**Where.**
- `P1_PERFORMANCE_TABLE.md:31-33` records the benchmark code as `691b571`.
- That commit predates the review-fix merge `f5178f5` and the consolidation `3674780`. `git merge-base
  --is-ancestor 5503122 691b571` is false.
- The merged `advance()` now runs `_check_foreign_hashgrids()` and a host `refresh_device_hashgrid()` before every
  launch (`material_batch.py:925-934`).
- The same holds for LR-A (code `cc9fb19`) and the P1-5 CUT rows.

**What was measured separately (fix branch, SHARED_GPU).**
- The refresh costs ≤ 0.25 ms of GPU time per step.
- On an idle stream (strict mode) it returns in 0.09 ms on the host.
- Issued behind a running graph (training mode, as in the CUT runner, `run_cut_batched.py:114`), the host waits
  about 14 ms.
- So the rows are probably within run-to-run spread, but the table does not say that the merged hot path differs.

**Recommendation.** Add this note to the table and the PR. Optionally re-run a short EXCLUSIVE spot check at the
merge commit: A `frozen_v1` N=1/16 in strict mode and CUT N=8 in training mode.

### 6 — MINOR — the README "Known limitations" are stale

**Where.** `README.md:274-275` says "Every batched throughput number so far is `SHARED_GPU` … Exclusive-GPU rows are
pending."

**What is wrong.**
- The EXCLUSIVE table has been merged since `8e503e1`.
- The report list (`README.md:278-286`) has no link to `P1_PERFORMANCE_TABLE.md`.
- The section does not say that P1-6 sync is PENDING, or that "whole-RCS RL-ready" is PARTIAL for CUT.

The rest of the section matches the code. The commands and flags exist (`run_p1_2b_smoke.py:50-66`,
`run_rl_smoke.py:251-254`).

### 7 — MINOR — the progress file still carries stale running state (head `644d9aa`)

These lines are out of date:
- `docs/RL_READY_PROGRESS.md:58-59`: B2 "Remaining: `grip_contact_profile` passthrough…" was done in the
  consolidation.
- `:63`: P1-5 "Open: … `tasks=` for configured scenes" was also done.
- `:64-68`: "formal Phase 1 acceptance rows running from a detached checkout…" (the "Chain B running" fragment was
  fixed in `9289328`).
- `:76-86`: "Running now: … Parity matrix today: 4 PASS, 8 PARTIAL, 1 UNSUPPORTED (cameras), 2 NOT_RUN". The
  current tally is 9 / 6 / 1.
- `:117-118`: "rows `SHARED_GPU`; formal rows rerun in an exclusive window" and the old "Next:" order.

### 8 — MINOR — the renumbered invalid bits are still stated with their old values in the docs and evidence

**Code: consistent.**
- `termination.py:46-48` is asserted equal to `material_batch.py:64-68` by `tests/test_rl_termination.py`.
- No 64 / 128 / 256 literal is used for material bits anywhere in `rl/`, `batched/` or the scripts.
- Historical rows are never re-decoded with the new table. Live values only are decoded
  (`probe_hashgrid_descriptor.py:288`), and ledgers store names.

**Docs and evidence: stale.**
- `docs/adr/0021-batched-rl-gpu-headless-runtime.md:893` gives `FOREIGN_DEVICE_HASHGRID` as **128**; it is 256.
- The earlier review's §7.3 (`P1_1_P1_2_P1_3A_ACCEPTANCE_REVIEW.md:672`) says the same, without a merge note.
- `review_fix1_001.json` (raw 128) and `p1_1b_b2_001.json` (raw 64 for INPUT_OUT_OF_DOMAIN) carry no
  machine-readable renumbering note.
- `scripts/rl_ready/make_review_fix1_evidence.py:196` has the same string.

**Scenario.** A reader decodes `p1_1b_b2_001.json`'s `invalid_reason: [0, 64]` with
`termination.decode_material_invalid`. The result is `CONTACT_SPEED_GUARD` instead of `INPUT_OUT_OF_DOMAIN`.

**Recommendation.**
- Add correction blocks to both evidence files: the bit map at branch time, kept additive.
- Add an inline note at ADR 0021:893.
- `P1_1B_ROBOT_BATCHING.md:371-375` already has the right merge note.

### 9 — MINOR — parity matrix and progress text are inaccurate in places

**Where.**
- `P1_PARITY_MATRIX.md:46` (row 10): "PASS (all `SHARED_GPU`)" and the label "`SHARED_GPU` on every batched row"
  contradict the EXCLUSIVE table that the same row links.
- `:153-156`, `p1_consolidation_001.json` ("symptom") and `RL_READY_PROGRESS.md:75` misdescribe the WP-12
  failure:
  - they say that `workcell_live_006` "holds analysis class CUTTING_FAILURE";
  - in fact its `analysis_001/analysis.json` says **TASK_SUCCESS**, which equals the committed evidence;
  - `run_wp12_phase5.native_execution` re-derives **CUTTING_FAILURE** from `run/`. That is the assertion that
    fails at `tests/test_wp12_phase5_native_success.py:201`.
- `:102-109` (checkbox 5) is ticked while LR-B, the declared reset-stress run, FAILed its pre-declared VRAM bound.
  The tick should read "met with a recorded FAIL (LR-B B0)".
- The open WP-16 ↔ WP-17 seed-512 final cut-code difference is not listed in the matrix. It is listed in
  P1-0 and P1-5.

### 10 — MINOR — the supervisor and `classify_process_exit` misclassify two infrastructure cases as task failures

**Supervisor.**
- `src/robotic_cutting_sim/rl/supervision.py:343-346`. A child `infra_error` whose own classification is
  `env_category == "caller"` (a harness or programming error) falls back to `classify_error_text`.
- Under split `termination_reason_v1`, a `FloatingPointError` or "non-finite" message then becomes
  `TASK_STRICT_INVALID_ENV`, and `agrees_with_env_exit` hides the disagreement.
- `long_rollout_v2` says that any other exception is `INFRA_PYTHON_EXCEPTION`.

**`classify_process_exit`.**
- `termination.py:213` labels stderr text "Invalid rigid worlds … CONTACT_POOL_OVERFLOW" as task.
- `classify_exception` (`:173-175`) calls the same failure infrastructure.

**Legacy info path.** `supervision.py:399-404` / `:436-438` classify the bit-256 env as task under `p1_4_v1`. Only
the fake backend reaches this.

**Evidence impact.** None. The LR-A/B/C/D classes are unaffected.

**Recommendation.** Map `env_category == "caller"` to `INFRA_PYTHON_EXCEPTION`. Test for `CONTACT_POOL_OVERFLOW`
before the task signature. Add the missing test cases.

### 11 — MINOR — the EXCLUSIVE window leaves several plan columns or workloads unmeasured

| Item | Measured in the EXCLUSIVE window? |
| --- | --- |
| Workload A reset p95 | Full reset only. No partial reset. |
| CUT rows | `partial_reset_p95_s: null`. |
| B rows | Reset p95 "n/a", an empty plan column. |
| Lab partial reset with material | Never measured, on any GPU. |
| B1 N=16 | SHARED only. |
| CUT N=4, B2 at N ≠ 4, A s15 at N ≠ 8, batched cold rows | NOT_RUN. |

Partial-reset p95 exists only from SHARED runs:
- 7.9 ms (LR-A, N=8);
- 15.9 ms (LR-B, N=16);
- 3.2 ms (CUT M1);
- 12–16 ms (Lab robot-only S1).

### 12 — MINOR — RL rollout files carry oracle-derived fields without a non-deployable label (earlier note N8 still open)

**Where.**
- `src/robotic_cutting_sim/rl/rollout.py:134` marks only `obs_critic` as `PRIVILEGED_NON_DEPLOYABLE`.
- `invalid_reason` and `termination_reason` are saved for every transition without a label.
- The env `info` also carries the oracle-derived severed-bond terms of the shaping reward.
- `scripts/rl_ready/run_rl_smoke.py:182` stores `actions.clamp(-1, 1)`, not the executed action that
  `rollout.py:4` documents.

**Holds.** `obs_policy` itself contains only the deployable `POLICY_TERMS`.

**Recommendation.** Label these fields in the manifest and in the env metadata.

### Notes (no action required for this merge)

- **N1.** `material_input_domain_v1` always runs inside `BatchedCuttingMaterial.advance()`
  (`material_batch.py:927`); it has no opt-out.
  - In-domain envs only record their hold state, and their solver inputs are bitwise unchanged
    (`tests/test_batched_input_domain.py`).
  - The receipt names it.
  - The RL-side guard is default-on with an `input_guard=False` rollback (API revision `p1_2b`).
  - Acceptable as a versioned safety guard. Say so in the PR.
- **N2.** P1-1b's "Native confirmation of material parity" is the frozen single-env material (the class Native
  runs), replayed on the batch's edge and pad sequences (ADR 0022:405-411, 477). It is not a new Kit run.
- **N3.** API edge cases:
  - `evaluate()` refuses an env that was autoreset to step 0 by raising for the whole call
    (`vector_env.py:1025`). The refusal is right, but a per-env `NOT_EVALUATED` entry would let the other envs be
    scored.
  - `evaluate(env_ids=[-1])` silently scores env N−1, on both backends.
- **N4.** Lab `TERM_IK_STREAK` is recomputed each step and so is not sticky (`lab_batched_env.py:491-493`). It is
  off by default. The Sen 95 % interval in `stability.py:75-78` is one order statistic off; it is reported only,
  never used as a gate.
- **N5.** The B2 rows carry `reduced_fidelity: False` with the listed difference `mjwarp_grip_v1`.
  - That profile is a global MuJoCo-Warp option set: the box–box collider and `impratio` 10.
  - Its `physical_calibration` is NOT_VALIDATED.
  - The PR should say this.
- **N6.** The Markdown performance table has no `reduced_fidelity` column. Every evidence row carries the flag, and
  §6 of the table labels each workload.
- **N7.** The pre-existing failure `test_wp12_phase5_native_success.py::test_reproduces_phase2b_records` is
  **unrelated to this branch**:
  - the test, `scripts/run_wp12_phase5.py` and `generalization_wp12.py` are byte-identical to `origin/main`
    (`git diff` is empty);
  - its outcome depends only on `artifacts/wp12`, which here links the main checkout's artifacts;
  - reproduced here: 1 failed, 36 passed in the file, on `workcell_live_006`;
  - it stays OPEN as a WP-12 records question.
- **N8.** Under D-12, Phase 2 steps merge into this same branch. The PR must therefore be cut at the reviewed
  commit plus this report, and anything merged after `f03fbff` other than documentation needs its own review.

## 5. What was verified against raw artifacts (holds)

**Exclusive classification.**
- I recomputed it for all 23 rows. The inputs were each row's own `telemetry_samples.jsonl` (or `row.json`
  telemetry for the B rows), the chain sampler restricted to the row's start/end from `chain_{a,b,c}.log`, and the
  before/after compute-app snapshots.
- Result: 20 EXCLUSIVE and 3 SHARED: `a_n8_s15_default` (graphics PID 2301065), `b1_n4` (2364307) and `b1_n16`
  (2565857/8). Foreign SM was 0 % everywhere. This is identical to `P1_PERFORMANCE_TABLE.md` §4.
- The `a_n1_*` and `a_n4_default` rows had no chain-sampler coverage, and the table records that as "unavailable".
- Script: `scratchpad/p1rev_x7/scripts/excl_check.py`, not committed.

**Table values.** I spot-checked:
- aggregate sim-time/wall of A N=1/4/16 and `a_n8_s15_default_r2` (0.1350 / 0.2011 / 0.1966 / 0.1870);
- B2 N=4 env-steps/s 7.372;
- B1 N=16 8.789;
- CUT `result.json` COMPLETE at `691b571`, with full-reset p95 values as tabled.

**B2 physics.**
- Exclusive `b2_n4`: evaluator PASS 4/4 with labelled 38,220 / 81,034 / 38,220 / 81,033. `termination_reason` is
  16 (truncation) in every env.
- The grasp status is HOLDING at the end. Each env was SLIPPING at 4 steps. Max slip is 1.51 / 1.59 mm and max
  rotation 14.0 / 15.1 mrad.
- `robot_b2grip_n2_001`: 1196/1200 HOLDING.
- `s3_b2grip_n{2,4}_001`: PASS in every env.

**B1.** Evaluator PASS in every env of N=1, 4 (repeat), 8 and 16. Labelled 33,740 / 80,986 per seed in every N.

**Device grids.** The B rows show one CUDA HashGrid. The CPU HashGrids of the frozen packing solvers do not count.

**P1-5 CUT.**
- `m0/m1/m2_cut_*`: 35 reference episodes, all PASS.
- M1's four FAIL_INTEGRITY entries are random-policy segments ended by a forced reset or the run end
  (`complete_episode` false), not reference episodes.

**P1-4 LR-A.** Read from the ledger and the telemetry:
- 72 episode records, of which 64 are full-length;
- canaries 16/16 PASS, `codes_equal_native` 16/16;
- own VRAM max 1,412 MiB;
- foreign SM mean 34.8 % (max 98 %).

**P1-4 LR-B.** Own VRAM max 2,304 MiB. Analysis verdict FAIL (`bounds_pass: false`), with every other check true.

**Integrity.**
- No frozen module, evaluator, goal or scene config was modified (see §1).
- No `artifacts/` file is committed.
- The diff contains no binary file and no file over 1 MB.
- A pattern scan of the diff for API keys, tokens and private keys finds nothing.
- No rubber (Phase 2) source is on this branch.

**Default paths.**
- The `frozen_v1` batched kernel sources are pinned (`FROZEN_BATCH_KERNEL_SHA256`, `test_batched_frozen_sources.py`).
- F1 adds only host-side bookkeeping and a rebuild with identical inputs.
- The input-domain kernel leaves in-domain inputs untouched.
- The single-env frozen path is unchanged, because no frozen file changed.

## 6. Tests run by this review (CPU only; never GPU1)

```bash
cd /home/ningan/RoboticCuttingSim-rl-review   # detached at f03fbff
SEL="tests/test_batched_*.py tests/test_rl_*.py tests/test_vector_env_*.py tests/test_perf_acceptance.py \
     tests/test_long_rollout_config.py tests/test_native_source_freeze.py tests/test_repo_naming.py"
CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp .venv/bin/python -m pytest -q -p no:cacheprovider -rs $SEL
# 511 passed, 26 skipped in 582.5 s -> PASS (26 skips = Torch / Isaac Lab modules, NOT_RUN in .venv)
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
  .venv/bin/python -m pytest -q -p no:cacheprovider -rs $SEL
# 511 passed, 26 skipped -> PASS (identical skip set; incl. the FallbackGrid test with GPU0 visible)
CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_vector_env_*.py tests/test_rl_*.py tests/test_perf_acceptance.py
# 334 passed, 10 skipped in 819.6 s -> PASS (skips NOT_RUN: GPU stream test needs CUDA; Lab/pxr modules)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaaclab-venv/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_rl_lab_*.py tests/test_vector_env_contract.py
# 56 passed in 239.1 s -> PASS (incl. the mjwarp_grip_v1 passthrough tests)
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wp12_phase5_native_success.py
# 1 failed (test_reproduces_phase2b_records, workcell_live_006), 36 passed -> pre-existing, unrelated (N7)
```

Not run by this review:
- the full CPU suite (the consolidation's run at `53b915e` is recorded: 2443 passed, 1 failed (N7), 28 skipped,
  6 xfailed);
- any GPU job.

The naming test was run after `git add` of this file (see the commit).

## 7. Limitations that must appear in the PR body

1. **Phase 1 is not "done".**
   - The external-optimizer sync (checkbox 7) is PENDING: no candidate has been accepted and no descriptor is
     pinned.
   - After the merge the status is `MERGED_RCS_PENDING_OPTIMIZER_SYNC` until the external pin (SYNCED).
2. **P1-6 contract.**
   - `acceptance_v1`'s G9 times only the pre-contact replay window at N=8/16 (steps 21–120; contact starts at step
     314).
   - The contract was amended in place four times before any baseline (contact floor 0.01 → 0.02; reset-p95
     FAIL → WARN on shared GPUs; band rules), although it is described as frozen from its first commit.
   - It freezes at this merge (SHA-256 to be recorded).
   - The first optimizer acceptance needs `acceptance_v2` with a contact/cutting window.
   - VRAM and reset p95 remain candidate-reported.
3. **CUT is PARTIAL, so "whole-RCS RL-ready" is PARTIAL for CUT.**
   - The original frozen `tlmpm_v5` CUT is not batched (`UNSUPPORTED_SOLVER`); it runs single-env headless and in
     Native.
   - Batched CUT is the new opt-in `bar_flat_board_tlmpm_v7_2_v1` scene: ideal tracking, no robot, reaction not
     fed back.
   - Lab CUT and the batched replay of the Native v7_2 recording are NOT_RUN.
4. **Reduced fidelity.**
   - Workload A (replayed or ideal edge, no robot dynamics, reaction not fed back) and B1 (rigid tool mount) are
     `reduced_fidelity`.
   - B2 physical grasp requires the opt-in MuJoCo-Warp profile `mjwarp_grip_v1` (`physical_calibration`
     NOT_VALIDATED). Slip is about 1.5 mm and rotation about 15 mrad, against about 1 µm / 1e-4 rad in Native.
   - B2 G1/G3 deterministic isolation, G4 and G5 with the profile: NOT_RUN.
   - The SIGSEGV input guard rests on one paired N=4 attempt.
5. **Known workload-B defects (opt-in paths, no recorded evidence affected).**
   - `fidelity()` ignores a non-default `material_dt_s` (finding 3).
   - In non-strict B2, a failed pregrasp settle is cleared by the first `reset()` (finding 4).
   - Both are to be fixed before B2 is used in training mode or with a non-default dt.
6. **Performance.**
   - The formal EXCLUSIVE table (GPU1, 20/23 runs exclusive) measured code `691b571`, which predates the per-step
     HashGrid refresh of review fix F1 (≤ 0.25 ms GPU per step, measured separately) and the consolidation.
   - Workload-A throughput windows are pre-contact at N ≥ 4.
   - B1 N=16 is SHARED only.
   - No EXCLUSIVE partial-reset latency, no batched cold rows. CUT N=4 and B2 at N ≠ 4 are NOT_RUN.
   - P1-4 and every partial-reset latency are `SHARED_GPU`.
   - Batching gains about 1.4–1.5× over N=1 for the material-only workloads (A, CUT) and 1.75× for B1. For A
     it saturates at N≈4.
7. **Stability.**
   - LR-B (N=16 reset stress) **FAILed** its pre-declared VRAM bound B0 (2,304 > 2,253 MiB; the excess comes
     from the harness's own reset checksum and was not re-run).
   - LR-A2 (`alias_free_v1`) and LR-E (Lab long run) are NOT_RUN.
8. **Parity limits.**
   - L4 RUN_TO_RUN is BLOCKED (frozen v5 atomics).
   - GPU position bands (band v2) are descriptive.
   - Batched GPU isolation is outcome-level. Bitwise isolation is shown on CPU only.
   - The WP-16 ↔ WP-17 seed-512 final cut-code difference remains an open issue; there is no physics-equivalence
     claim.
9. **Not supported in the batched path.**
   - Cameras, vision profiles and physics recorders (`UNSUPPORTED_SENSOR` / `UNSUPPORTED_RECORDER`).
   - Per-episode geometry or material randomisation inside a live batch.
   - Mixed TRIM + CUT layouts in one batch.
   - Solvers other than `tlmpm_v7_2`.
   - One device HashGrid per process (Warp 1.17).
10. **Classification and data labelling gaps** (findings 10 and 12): two infrastructure cases can be counted as
    task failures; rollout files store `invalid_reason` / `termination_reason` without a non-deployable label.
11. **Pre-existing, unrelated:** `test_wp12_phase5_native_success.py::test_reproduces_phase2b_records` fails when
    `artifacts/wp12` links the main checkout's WP-12 artifacts (OPEN, WP-12).
12. **Materials** are provisional and uncalibrated (`SIMULATOR_ONLY_UNCALIBRATED`).

## 8. Merge go / no-go

**GO** for the Phase 1 PR to `main`, under these conditions:

1. The PR body carries the §7 list verbatim, or a faithful equivalent, and links this report.
2. The PR head is `f03fbff` plus this report and documentation-only commits. Code merged after `f03fbff`, including
   any Phase 2 (D-12) merge into this branch, is out of this review and needs its own.
3. Should be done in the same PR (documentation only, no re-review): findings 6 and 7 (README and progress), the
   finding-9 wording, and the finding-2 wording (record the freeze SHA of `acceptance_v1.json`).

**No-go for using `acceptance_v1` in a P1-6 acceptance** until findings 1 and 2 are resolved as `acceptance_v2`.

**Before B2 training-mode use:** fix findings 3 and 4.

**Follow-ups, not blocking:** findings 5, 8, 10, 11 and 12.
