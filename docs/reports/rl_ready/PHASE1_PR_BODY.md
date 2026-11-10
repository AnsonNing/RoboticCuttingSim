# Phase 1: RL-ready batched GPU headless runtime (PR description)

Branch `claude/rl-phase1` → `main`. The head is the reviewed commit `f03fbff`, plus the Phase 1 acceptance review
(`c5e96b2`) and documentation-only commits that apply its required fixes (findings 2, 6, 7, 8, 9). No source, test,
script or config changes after `f03fbff`. `configs/rl_benchmark/acceptance_v1.json` is byte-identical to the
reviewed version (sha256 `3592d9408f314c3d2825171361b8ec640534c60577bb6216a10c4f2b22897bf4`).

**Acceptance review:** [PHASE1_ACCEPTANCE_REVIEW.md](PHASE1_ACCEPTANCE_REVIEW.md), xhigh, at `f03fbff`:
**PASS_WITH_LIMITATIONS, merge GO under the conditions in its §8.** No BLOCKER. Four MAJOR findings: two concern
the P1-6 contract `acceptance_v1` (findings 1, 2) and two are workload-B defects on opt-in paths that no recorded
evidence used (findings 3, 4).

**Status:** `MERGED_RCS_PENDING_OPTIMIZER_SYNC after merge`, until the external descriptor is pinned in the external
agentic optimizer repository (SYNCED). Phase 1 is not "done" until then.

## Summary: what is RL-ready

- **Batched GPU material runtime** (`robotic_cutting_sim.batched`): N envs of the `tlmpm_v7_2` law-8 material in
  one process, one launch per kernel for all envs, packed from the frozen single solvers; frozen `frozen_v1` default
  with an opt-in `alias_free_v1` contact mode (1.38–1.47× in EXCLUSIVE pairs). One device HashGrid per process
  (Warp 1.17). ADR 0021; [P1-0 baseline](P1_0_BASELINE.md); [P1-1/P1-2 milestones](P1_1_P1_2_MILESTONES.md);
  [P1-3a contacts](P1_3A_ALIAS_FREE_CONTACTS.md).
- **Vector env API** (`robotic_cutting_sim.rl`, `open_vector_env`): one contract (`rl.vector_contract`) for two
  backends — workload A `kinematic_ideal_tracking_batched` (replayed or ideal edge, no robot) and workload B
  `isaac_lab_newton` (N Franka worlds on Newton/MuJoCo-Warp, B1 rigid mount, B2 physical grasp with the opt-in
  `mjwarp_grip_v1`). Torch device tensors, masked partial reset, per-env seeds and `tasks=`, `termination_reason_v1`,
  frozen evaluator through `evaluate()`, `FAIL_INTEGRITY` for non-finite envs, computed `fidelity()` in receipts.
  ADR 0022; [P1-1b robot batching](P1_1B_ROBOT_BATCHING.md).
- **Long rollout and supervision**: crash-safe supervisor and ledger, task vs infrastructure classification
  ([P1-4 long rollout](P1_4_LONG_ROLLOUT.md)).
- **CUT and RL smoke**: batched CUT on the opt-in v7_2 scene, frozen v5 CUT single-env headless, RL pipeline smoke
  without training ([P1-5 CUT](P1_5_CUT.md)).
- **Parity and performance**: [P1 parity matrix](P1_PARITY_MATRIX.md) (9 PASS / 6 PARTIAL / 1 UNSUPPORTED);
  formal [P1 performance table](P1_PERFORMANCE_TABLE.md) (GPU1, 20 of 23 runs `EXCLUSIVE_GPU`).
- **P1-6 tooling (R0)**: acceptance harness, judge and `acceptance_v1` ([P1-6 R0](P1_6_R0_ACCEPTANCE_HARNESS.md)).
  The contract is merged as tooling only; it is not fit for a P1-6 acceptance (see limitation 2).
- Plan, decisions and progress: [RL_READY_PLAN.md](../../RL_READY_PLAN.md) (D-01…D-13),
  [RL_READY_PROGRESS.md](../../RL_READY_PROGRESS.md). README section "RL-ready (Phase 1)".

No frozen solver, evaluator, goal, scene config, v1 threshold or record is modified. No geometric deletion is
presented as cutting. No oracle or force state enters the deployable policy observation. Failed runs are preserved.

Further notes from the review (§4 notes):
- `material_input_domain_v1` always runs inside `BatchedCuttingMaterial.advance()` with no opt-out; in-domain envs
  are bitwise unchanged and the receipt names it. It is a versioned safety guard (N1).
- The B2 rows carry `reduced_fidelity: False` with the listed difference `mjwarp_grip_v1`, a global MuJoCo-Warp option
  set (box–box collider, `impratio` 10) whose `physical_calibration` is NOT_VALIDATED (N5).
- Code merged into the branch after `f03fbff` (for example Phase 2 merges under D-12) is outside this review (N8).

## Limitations (review §7, all twelve)

1. **Phase 1 is not "done".**
   - The external-optimizer sync (checkbox 7) is PENDING: no candidate has been accepted and no descriptor is
     pinned.
   - After the merge the status is `MERGED_RCS_PENDING_OPTIMIZER_SYNC` until the external pin (SYNCED).
2. **P1-6 contract.**
   - `acceptance_v1`'s G9 times only the pre-contact replay window at N=8/16 (steps 21–120; contact starts at step
     314).
   - The contract was amended in place four times before any baseline (contact floor 0.01 → 0.02; reset-p95
     FAIL → WARN on shared GPUs; band rules), although it was described as frozen from its first commit. The
     wording is corrected in D-11 and the R0 report.
   - It freezes at this merge, with sha256 `3592d9408f314c3d2825171361b8ec640534c60577bb6216a10c4f2b22897bf4`
     (the merge commit is recorded in D-11 after the merge).
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

Merge conditions and follow-ups (review §8): no use of `acceptance_v1` in a P1-6 acceptance until findings 1 and 2
are resolved as `acceptance_v2`; findings 3 and 4 fixed before B2 training-mode use; findings 5, 8, 10, 11 and 12
are follow-ups (finding 8's documentation and evidence notes are already in this PR).

## Tests (review §6, at `f03fbff`; CPU only, no GPU job)

| Suite | Result |
| --- | --- |
| `.venv`, RL/batched/vector/acceptance selection, `CUDA_VISIBLE_DEVICES=''` | 511 passed, 26 skipped (Torch / Isaac Lab modules, NOT_RUN in `.venv`) in 582.5 s — PASS |
| Same selection, GPU0 visible (under the GPU0 flock) | 511 passed, 26 skipped (identical skip set, incl. the FallbackGrid test) — PASS |
| `.deps/isaacsim-venv` (Torch, D-09 overlay): `test_vector_env_*`, `test_rl_*`, `test_perf_acceptance` | 334 passed, 10 skipped (GPU stream test needs CUDA; Lab/pxr modules) in 819.6 s — PASS |
| `.deps/isaaclab-venv`: `test_rl_lab_*`, `test_vector_env_contract` | 56 passed in 239.1 s — PASS (incl. the `mjwarp_grip_v1` passthrough tests) |
| `.venv`: `test_wp12_phase5_native_success.py` | 1 failed (`test_reproduces_phase2b_records`, `workcell_live_006`), 36 passed — pre-existing, unrelated (N7) |

Not run by the review: the full CPU suite (the consolidation's run at `53b915e` is recorded: 2443 passed, 1 failed
(N7), 28 skipped, 6 xfailed) and any GPU job. The documentation-only commits after the review ran the repository
naming test before each commit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
