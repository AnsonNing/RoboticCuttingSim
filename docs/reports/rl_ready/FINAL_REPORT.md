# RL-Ready Simulator & Rubber Physics — final report (DRAFT)

**Status: DRAFT, 2026-11-12.** The Phase 2 merge into `main` and the new external descriptor pin are still
pending (§9). Phase 2 (P2-5, P2-6 and the K2 follow-ups) is merged into `claude/rl-ready-rubber-physics` at
`92e671a`, so every report cited here, including [P2_5_RUBBER_CALIBRATION.md](P2_5_RUBBER_CALIBRATION.md) and
[P2_6_RUBBER_RL.md](P2_6_RUBBER_RL.md), is on this branch.

**Revision after the Phase 2 acceptance review (Z1).** The xhigh review
[PHASE2_ACCEPTANCE_REVIEW.md](PHASE2_ACCEPTANCE_REVIEW.md) (at `92e671a`, verdict GO_WITH_CONDITIONS) checked 69
values in this draft. 68 matched; the RL-05 v2 count was wrong (5/8, not 4/8). This revision applies its documentation
conditions D1 (findings F1, F4, F5, F6): box 2 and box 3 wording, the real-data protocol status, stale passages, the
reproduction row names, the batching-gain wording (§2.3, §4.4), two missing limitations and the calibration-to-RL
handoff (§5). No verdict is re-scored.

**Scope.** This report covers the user's plan v2.0 ("RL-Ready Simulator & Rubber Physics", 2026-11-10). Its in-repo
copy is [RL_READY_SPECIFICATION.md](../../RL_READY_SPECIFICATION.md). Work packages and decisions D-01..D-14 are in
[RL_READY_PLAN.md](../../RL_READY_PLAN.md); the running log is [RL_READY_PROGRESS.md](../../RL_READY_PROGRESS.md).

**How to read this report.**

- Status words: PASS / FAIL / PARTIAL / BLOCKED / NOT_RUN / PENDING. A skipped test is NOT_RUN, and exit code 0 is
  not task success.
- Every number cites the report or evidence file it comes from. Where two sources disagree, the report says so (§4.4).
- `EXCLUSIVE_GPU` / `SHARED_GPU` follow decision D-07: a row is SHARED when another process used the GPU during it.
- Material status everywhere is **`SIMULATOR_ONLY_UNCALIBRATED`**. No real rubber data exists on the machine.
- "The external agentic optimizer" is the separate research project that proposed and judged a performance candidate
  (P1-6). Its own records live in the external optimizer repository; this repository holds the RCS side of every
  step.

## Contents

1. [Summary against the plan's Done criteria](#1-summary-against-the-plans-done-criteria)
2. [Phase 1 — RL-ready batched simulator](#2-phase-1--rl-ready-batched-simulator)
3. [P1-6 — the external agentic optimizer cycles](#3-p1-6--the-external-agentic-optimizer-cycles)
4. [Phase 2 — rubber material physics](#4-phase-2--rubber-material-physics)
5. [Known limitations and open items](#5-known-limitations-and-open-items)
6. [User decisions pending](#6-user-decisions-pending)
7. [Resource use](#7-resource-use)
8. [Reproduction](#8-reproduction)
9. [Phase 2 merge and pin (PENDING)](#9-phase-2-merge-and-pin-pending)

## 1. Summary against the plan's Done criteria

**In one paragraph.** RCS now runs N cutting environments in one process with batched GPU kernels:
`open_vector_env`, masked partial reset, Torch device tensors, and the frozen evaluators. The same path runs the
representative TRIM task, a versioned CUT scene and a robot-batched workload. Phase 1 is merged into `main` (PRs
#12–#15). The external agentic optimizer found one accepted speed-up, `alias_free_v2`: 1.279× at N=16. It was
merged and pinned (SYNCED).

Phase 2 adds a TLMPM rubber material: Neo-Hookean with a J̄ volumetric term, Prony viscoelasticity, and grid-node
penalty Coulomb contact. The material runs single-env, batched and in Native Kit, with a batched rubber RL task and a
calibration interface for real data. Every Phase 2 package records its FAILs. The synthetic calibration test (P2-5)
**FAILED**, and the material stays uncalibrated. The Phase 2 PR is not yet open.

### 1.1 Phase 1 Done (specification, seven checkboxes)

| # | Criterion (abridged) | Status | Evidence |
| --- | --- | --- | --- |
| 1 | True GPU batching at N=1, 4, 8–16 with partial reset; throughput / memory / latency | **PASS** (with limitations) | EXCLUSIVE rows exist for workload A at N=1/4/8/16, CUT at N=1/8/16 and B1 at N=1/4/8 ([P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) §3). B1 N=16 is SHARED only. No EXCLUSIVE partial-reset latency and no batched cold rows exist ([PHASE1_ACCEPTANCE_REVIEW.md](PHASE1_ACCEPTANCE_REVIEW.md) §3.2). Milestones M0–M2 PASS ([P1_1_P1_2_MILESTONES.md](P1_1_P1_2_MILESTONES.md)). M3 (N ≥ 32) is NOT_RUN: no report records a run above N=16 |
| 2 | Representative CUT / TRIM in RL headless mode with evaluation; gaps explicit | **PARTIAL** | TRIM: PASS. CUT: PARTIAL by design — the frozen v5 CUT is not batched (`UNSUPPORTED_SOLVER`); batched CUT is the opt-in v7_2 scene with ideal tracking and no robot ([P1_PARITY_MATRIX.md](P1_PARITY_MATRIX.md) rows 2a/2b, U-P15-1). "Whole-RCS RL-ready" is therefore PARTIAL for CUT |
| 3 | Single-env vs batched numeric / cut-history parity; frozen-profile regression | **PASS** (with limitations) | CPU L0/L1 bitwise parity for TRIM and CUT. The GPU position band is descriptive. L4 RUN_TO_RUN is BLOCKED (the frozen `tlmpm_v5._particle_contacts` mixes add/max atomics). The batched replay of the Native v7_2 recording is NOT_RUN ([PHASE1_ACCEPTANCE_REVIEW.md](PHASE1_ACCEPTANCE_REVIEW.md) §3.2) |
| 4 | GPU-native action / obs / reward / done, per-env seeds, reproducible scenes, API tests | **PASS** (with limitations) | Parity matrix rows 7, 8, 14 PASS. Row 9 is PARTIAL: no per-episode geometry or material randomisation inside a live batch (U-P15-3) |
| 5 | Long rollout, OOM / memory / reset stress, real GPU telemetry | **PARTIAL** | LR-A PASS (N=8, 64 full episodes, 5.06 h). LR-C PASS. LR-D 5/5 PASS. **LR-B FAIL** on its own VRAM bound (2,304 > 2,253 MiB). LR-A2 and LR-E NOT_RUN. Every run is SHARED_GPU ([P1_4_LONG_ROLLOUT.md](P1_4_LONG_ROLLOUT.md)). The review calls this "MET with a recorded FAIL" |
| 6 | Headless, vision and Native speeds reported separately; no fake comparison | **PASS** | Three boundaries in separate rows. Batched vision is UNSUPPORTED, not estimated. No B-vs-Native ratio ([P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) §5) |
| 7 | External optimizer results written back to RCS and synced | **PASS** | `p16c0`: descriptor `rcs_rl_vector_v1` pinned at `bfe6d5e`, SYNCED. `p16c1`: `alias_free_v2` accepted, merged in PR #14 (`9ee1e9d`), pinned as `rcs_rl_vector_v2`, SYNCED ([RL_READY_PROGRESS.md](../../RL_READY_PROGRESS.md), "進行中"). The follow-up PR #15 (`62caecc`) has no recorded pin (§5) |

### 1.2 Phase 2 Done (specification, six checkboxes)

| # | Criterion (abridged) | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Rubber recipe and hyperelastic / viscoelastic / contact models with reproducible run evidence | **PASS** (with limitations) | Recipe schema `rcs_rubber_material_v1`; models `neo_hookean_jbar_v1`, `neo_hookean_jbar_prony_v1`; contact laws v1/v2/v3. Each protocol was declared and SHA-pinned before its formal run, with compact evidence under `docs/evidence/rl_ready/p2_*` ([ADR 0023](../../adr/0023-rubber-material-v1.md); K2 review: [P2_K2_COMBINED_REVIEW.md](P2_K2_COMBINED_REVIEW.md) §0) |
| 2 | Large deformation, gravity, stretch / release, rebound, hysteresis and friction tests pass physically and numerically | **PARTIAL** | Most declared tests PASS. Remaining, all recorded: the RUBBER-PHYS-11 v5 refinement FAIL (wide-strip sag GCI 2.16 % > 2 %, accepted as a labelled +0.62 % bias); friction above μ 0.3 unsupported in the RL task (envelope FAIL at 0.4); contact law v3 for translational sliding only (K2-1); the -01 d stick-run energy books FAIL; self-contact detected, not resolved. Hysteresis passes only as "v1 FAIL, v2 PASS (post-hoc measurement correction)" (VISC-03, §4.1). The κ = 20 μ near-incompressibility treatment is a model-form error: volume ratio J − 1 = +5.4 % and nominal stress −4.2 % at stretch 2.0, bending stiffness −1.6 % ([ADR 0023](../../adr/0023-rubber-material-v1.md) §6). ADR 0023 §23.3 calls the same state PASS_WITH_LIMITATIONS; the Phase 2 review says PARTIAL ([PHASE2_ACCEPTANCE_REVIEW.md](PHASE2_ACCEPTANCE_REVIEW.md) §1) |
| 3 | The same rubber in multi-env RL headless and in Full Native; at least 8 envs with reliable results | **PASS** (with limitations) | P2-6 RL-01..RL-12 PASS (RL-05 only on its third declared controller). The N=8 GPU rows are RL-01, RL-02 b, RL-03, RL-05 v4, RL-06, RL-07 and RL-12; GPU isolation (RL-04 b) was tested at **N=4**, and the GPU L3 band (RL-02 c) runs an N=8 batch but scores 2 envs. Native RL-09 PASS. The P2-6 review grades box 3 PASS_WITH_LIMITATIONS: friction fixed at 0.3, geometry per env, kinematic grip, about 1 env-step/s ([P2_6_ACCEPTANCE_REVIEW.md](P2_6_ACCEPTANCE_REVIEW.md) §8; [P2_6_RUBBER_RL.md](P2_6_RUBBER_RL.md) §0) |
| 4 | REAL_CALIBRATED only after an independent held-out pass on real data; otherwise SIM_ONLY | **PASS** (conditional box: no real data) | Status stays `SIMULATOR_ONLY_UNCALIBRATED`. The synthetic numerical test of the calibration pipeline (P2-5, `rubber_calib_tests_v2`) **FAILED** CAL-04 / CAL-05 / CAL-06 ([P2_5_RUBBER_CALIBRATION.md](P2_5_RUBBER_CALIBRATION.md) §10) |
| 5 | Old CUT / TRIM / v5 / v7 baselines not broken | **PASS** | RL-11 on the P2-6 head: `.venv` 2,932 passed / 0 failed; Torch selection 414–416 passed; frozen Phase 1 paths equal to `main` except the 13-line dispatch branch ([P2_6_RUBBER_RL.md](P2_6_RUBBER_RL.md) §12) |
| 6 | Models, tests, scene and docs merged into RCS; external pin updated | **PENDING** | Phase 2 PR, merge and new descriptor are NOT_RUN at the time of this draft (§9) |

### 1.3 Pull requests merged into `main`

From `git log origin/main`:

| PR | Content | Merge commit | Review |
| --- | --- | --- | --- |
| #12 | Phase 1: RL-ready batched GPU headless runtime | `40c94c7` | PASS_WITH_LIMITATIONS ([PHASE1_ACCEPTANCE_REVIEW.md](PHASE1_ACCEPTANCE_REVIEW.md)) |
| #13 | Phase 1 follow-up: `acceptance_v2`, workload-B and labelling fixes | `bfe6d5e` | PASS_WITH_LIMITATIONS, re-review PASS ([P1_FOLLOWUP_REVIEW.md](P1_FOLLOWUP_REVIEW.md)) |
| #14 | P1-6 cycle `p16c1`: opt-in `alias_free_v2`, accepted by the frozen evaluator | `9ee1e9d` | PASS_WITH_LIMITATIONS ([P1_6_P16C1_PR14_REVIEW.md](P1_6_P16C1_PR14_REVIEW.md)) |
| #15 | `p16c1` review follow-ups: slow-contact shrink test, shrink counter, fail-closed r/dt check | `62caecc` | PASS ([P1_6_P16C1_PR15_REVIEW.md](P1_6_P16C1_PR15_REVIEW.md)) |

## 2. Phase 1 — RL-ready batched simulator

### 2.1 Batched physics design ([ADR 0021](../../adr/0021-batched-rl-gpu-headless-runtime.md), [ADR 0022](../../adr/0022-batched-robot-workload-b.md); decisions D-06, D-10)

- **One process, batched kernels.** N independent envs run as batched Warp kernels in one process, packed into one
  flat structure-of-arrays with exact per-env counts and offsets, and one grouped HashGrid (group = env). Python
  loops over single envs, N Kit processes and N graphs were rejected; unsupported configurations raise
  `UNSUPPORTED_*` (ADR 0021 §2).
- **Pack from frozen references.** Each env's initial state comes from the frozen single `tlmpm_v7_2` solver, so CPU
  parity can be bitwise per env (L0/L1). GPU parity is invariants plus outcome bands (L2/L3).
- **Graph and reset.** One CUDA graph per batch shape. Masked partial reset, with no reallocation and no recapture.
  A geometry change needs `rebuild()`.
- **Coupling.** The 20 ms delayed reaction wrench stays on the device as `[N,6]`.
- **Workloads.** A: replayed or ideally tracked edge, real batched cutting, `reduced_fidelity`. B: N Franka worlds
  on Isaac Lab Newton / MuJoCo-Warp coupled to the same batched material. B1 is a rigid knife mount
  (`reduced_fidelity`); B2 is a physical grasp with the opt-in `mjwarp_grip_v1`. C: the CUT scene
  `bar_flat_board_tlmpm_v7_2_v1`. D: the WP-17 TRIM replay.
- **Rubber extension point.** The `ConstitutiveModel` registry (ADR 0021 §11) is what Phase 2 built on.

### 2.2 Contract

`open_vector_env(num_envs, task=, tasks=, device=, backend=)` with backends `kinematic_ideal_tracking_batched` (A/C)
and `isaac_lab_newton` (B), sharing `rl.vector_contract`:

- `reset(seeds=, env_ids=)`; `step(actions[N,A])` returns obs, reward, terminated, truncated, info as Torch device
  tensors;
- `termination_reason` bits: 1 rigid, 2 material, 4 grasp lost, 8 IK streak, 16 truncated, 32 input guard,
  64 infrastructure;
- `evaluate()` through the frozen `goals.evaluate_goal`, with `FAIL_INTEGRITY` for non-finite envs;
- `receipt()` with computed fidelity;
- the deployable observation carries no oracle material state, cut codes or contact truth.

Evidence: [P1_PARITY_MATRIX.md](P1_PARITY_MATRIX.md) rows 7, 8, 12, 14 (9 PASS / 6 PARTIAL / 1 UNSUPPORTED overall).
Cameras are UNSUPPORTED in the batched path (row 6).

### 2.3 Throughput (formal exclusive table)

Source: [P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) §3 and §5, evidence
[`p1_exclusive_001.json`](../../evidence/rl_ready/p1_exclusive_001.json).

Setup:
- GPU1 (RTX 5090, driver 570.211.01), code `691b571` (before review fix F1's per-step HashGrid refresh);
- 50 Hz control, material dt 0.1 ms = 200 substeps;
- no cameras, no recording in the timed calls; warm.

Workload A is s512, 102,732 particles/env, 20 warm-up and 100 timed batch steps (300 at N=1), every row
`EXCLUSIVE_GPU`. Physics is `REPORTED_NOT_SCORED`: throughput mode has no evaluator.

| Workload A, env-steps/s | N=1 | N=4 | N=8 | N=16 |
| --- | --- | --- | --- | --- |
| `frozen_v1` (default) | 6.749 | 10.055 | 9.617 | 9.831 |
| `alias_free_v1` (opt-in) | 9.739 | 13.846 | 14.012 | 13.559 |
| VRAM peak `frozen_v1` / `alias_free_v1` (MiB) | 864 / 926 | 1084 / 1178 | 1356 / 1482 | 2044 / 2202 |

- **Pre-contact windows.** Tool contact starts at replay step ~314 (s512), so the windows at N ≥ 4 are entirely
  pre-contact; N=1 includes the first contact steps (§3 row scopes). The `alias_free_v1` ratios are therefore
  pre-contact numbers.
- **Other rows:**
  - single-env headless replay s512: 7.567 env-steps/s;
  - Native D-s512 (WP-17 evidence, full RGB-D rig): 4.040;
  - batched CUT: 6.849 / 10.350 / 10.223 at N=1/8/16 (33/33 episodes evaluator PASS);
  - B1 rigid mount: 4.955 / 7.655 / 8.666 at N=1/4/8; N=16 8.789 is SHARED;
  - B2 `mjwarp_grip_v1` N=4: 7.372 (4/4 PASS, grasp held).
- **Scaling.** Material-only batches gain about 1.4× over N=1 and saturate from N≈4: 1.37–1.40× in matched windows
  (P1-2, review fix F3). The exclusive table's own 1.43–1.49× (§5.1) divides by a 300-step N=1 window against
  100-step windows at N ≥ 4, which makes it about 4–5 % high
  ([PHASE2_ACCEPTANCE_REVIEW.md](PHASE2_ACCEPTANCE_REVIEW.md) §4.2; erratum in the table). The table's explanation
  for the saturation is that one ~100k-particle env already keeps the SMs about 76 % busy.
- **Contact modes.** `alias_free_v1` / `frozen_v1` is 1.38–1.47× in every pair (§5.3).
- **Boundaries.** Native, single-env headless and batched rows are never merged into one speed-up. The single
  comparisons are in §5.2 of the table.

### 2.4 Parity

- **Replay parity.** The single-env replay of the WP-17 measured trajectory reproduces the Native final cut codes
  exactly, 4/4 runs per seed. Max position difference: 1.4e-5–3.8e-5 m (s512) and 2.0e-4–4.6e-4 m (s15)
  ([P1_0_BASELINE.md](P1_0_BASELINE.md); progress "P1-0 baseline").
- **Batched episodes.** Workload A M0/M1/M2 final codes equal Native, with frozen evaluator PASS (M2 N=8 24/24,
  N=16 16/16) ([P1_1_P1_2_MILESTONES.md](P1_1_P1_2_MILESTONES.md)).
- **CPU parity.** L0/L1 is bitwise for TRIM and CUT. The GPU position band v2 (20 frozen single-env references,
  one-sided 99 % log-normal bound) is descriptive only.
- **Robot batching.** B1 final codes are bitwise across N, and G5 L3 frozen single-material replays reproduce them.
  That is not a Kit run ([P1_1B_ROBOT_BATCHING.md](P1_1B_ROBOT_BATCHING.md)).
- **Open.** The WP-16 ↔ WP-17 Native seed-512 final cut-code difference (4,266 particles) is unresolved.
  Replay parity is against WP-17 only and is no physics-equivalence claim ([P1_PARITY_MATRIX.md](P1_PARITY_MATRIX.md)
  row 3a).

### 2.5 Long rollouts ([P1_4_LONG_ROLLOUT.md](P1_4_LONG_ROLLOUT.md), evidence `p1_4_001.json`)

| Run | Result |
| --- | --- |
| LR-A: N=8, 8 serial full TRIM episodes per env | **PASS**: 5.06 h, 10,650 batch steps; canaries 16/16 bitwise = Native; VRAM peak 1,412 MiB (bound 1,536), VRAM slope 0.0 MiB per 1000 batch steps; 0 infrastructure failures |
| LR-B: N=16 reset stress | **FAIL**, on the declared VRAM bound B0 only (2,304 > 2,253 MiB); 668/668 resets bitwise. Kept as FAIL |
| LR-C: rebuild cycles | PASS |
| LR-D: GPU fault probes | PASS, 5/5 classified as declared |
| LR-A2 (`alias_free_v1`), LR-E (Lab B1 long run), EXCLUSIVE rows | NOT_RUN |

Every run had co-tenants (SHARED_GPU).

### 2.6 CUT and TRIM in the batched path ([P1_5_CUT.md](P1_5_CUT.md), evidence `p1_5_cut_001.json`)

- **Frozen v5 CUT.** WP-12 Demo 1 single-env headless replay ×4: evaluator PASS 4/4. Final codes equal Native in
  3/4; the fourth differs by 2 particles.
- **Batched CUT** (opt-in scene `bar_flat_board_tlmpm_v7_2_v1`): N=1/4/8/16, 35/35 reference episodes evaluator PASS,
  L2 PASS. A fresh Native v7_2 CUT episode PASSed.
- **RL pipeline smoke** (no training): S-CUT N=4 and S-TRIM N=8 consumer checks PASS. The first S-TRIM run FAILed on a
  runner bug; it is preserved and was rerun PASS.
- **NOT_RUN:** the batched replay of the Native v7_2 recording (T5.8 batched) and Lab CUT (T5.9).
- **TRIM in workload B.** B1 S3 frozen evaluator PASS in every env. B2 with `mjwarp_grip_v1`: S3 N=2/4 PASS. The
  default point-contact grip FAILs (GRASP_LOST), and grip slip is about 1.5 mm against about 1 µm in Native
  ([P1_PARITY_MATRIX.md](P1_PARITY_MATRIX.md) rows 3c/3d).

### 2.7 Withdrawn claim and the HashGrid descriptor bug

- **The bug.** Warp 1.17 keeps one static host descriptor per `wp.HashGrid`, so a replayed CUDA graph queries the
  **last-built** grid. A process holding two or more device HashGrids computes contacts on the wrong grid. The probe
  `scripts/rl_ready/probe_hashgrid_descriptor.py` shows 8,804,086 → 0 candidates
  ([P1_3A_ALIAS_FREE_CONTACTS.md](P1_3A_ALIAS_FREE_CONTACTS.md);
  [P1_1_P1_2_P1_3A_ACCEPTANCE_REVIEW.md](P1_1_P1_2_P1_3A_ACCEPTANCE_REVIEW.md)).
- **The rule since then.** One device HashGrid per process; `UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS`; a sticky
  `FOREIGN_DEVICE_HASHGRID` bit with per-launch grid refresh (review fix F1).
- **The withdrawn claim.** P1-1's `ab_timing_001` timed single, N=1 and N=4 in one process. It is classified
  `INVALID_MULTI_GRID_PROCESS`, and its "≈1.9× at N=4" claim is **withdrawn**
  ([P1_1_P1_2_MILESTONES.md](P1_1_P1_2_MILESTONES.md) line 135; ADR 0021 carries the withdrawal inline).
- **The replacement figure.** The matched-load P1-2 gain is 1.40× (N=8) and 1.37× (N=16) over batched N=1,
  `SHARED_GPU`, about 1.24–1.26× over the frozen single env (inferred).

### 2.8 Acceptance contracts (D-11, D-14)

- **`acceptance_v1`** (`3592d940…`, frozen at `40c94c7`). It was amended in place four times before any baseline,
  and two of those amendments loosened a gate after data had been seen. Its G9 speed gate timed only the pre-contact
  approach at N=8/16 (Phase 1 review findings 1 and 2). It is never used for an acceptance.
- **`acceptance_v2`** (`b66ace29…`, frozen at the PR #13 merge `bfe6d5e`, D-14):
  - G9 times replay steps 229–400 of s512, which cover all 101 tool-contact steps and all labelling, at N {1, 16}
    with 3 ABBA pairs;
  - N=16 paired median ≥ 1.05 with 3 of 3 pairs > 1; every N ≥ 0.97;
  - `G9:window` gates tool-contact steps and new labels;
  - it is a trade-off against v1, not stricter overall, and N=8 is not gated.

## 3. P1-6 — the external agentic optimizer cycles

The RCS side is [P1_6_R0_ACCEPTANCE_HARNESS.md](P1_6_R0_ACCEPTANCE_HARNESS.md) (harness, judge, local runner),
[P1_6_p16c1.md](P1_6_p16c1.md) (the accepted candidate and its gates) and the progress entries. The optimizer's own
cycle records (adapter, sandbox, ledger) are in the external optimizer repository.

**Cycle `p16c0` — baseline pin.**
- The optimizer repository added a new versioned descriptor `rcs_rl_vector_v1`, pinned at RCS `bfe6d5e` with a
  frozen checkout. Historical descriptors were untouched.
- Smoke PASS: the CPU contract; the host contract in a verified writable copy (RCS tests 266 passed / 0 failed /
  40 skipped); the GPU contract at N=2 with partial reset on GPU1.
- Its PR merged; the sync receipt chain went MERGED_RCS_PENDING_OPTIMIZER_SYNC → **SYNCED**.
- The adapter's xhigh review was PASS_WITH_LIMITATIONS with a re-check GO
  ([RL_READY_PROGRESS.md](../../RL_READY_PROGRESS.md), "進行中").

**Cycle `p16c1` — `alias_free_v2`.**

1. **Baseline** under `acceptance_v2` at `bfe6d5e`, GPU1: 9.57 / 13.76 / 13.44 env-steps/s at N=1/8/16 (progress).
2. **Research run.** One session, $6.02, 67 min. It profiled RCS: at N=8 in the contact window the particle-contact
   kernel was 35.5 % of kernel time (988 µs per substep), then P2G (24.8 %) and MUSL (10.9 %). It proposed two
   parity-level-A changes to the opt-in contact path ([P1_6_p16c1.md](P1_6_p16c1.md)):
   - env-major hash buckets (a stable radix sort over about 23–24 bits instead of 64-bit grouped keys, with no binary
     search per cell);
   - a per-env query shrink while every particle of the env is slower than 0.25 m/s.

   Dev measurements: N=8 1.216×, N=1 1.117×, identical episode codes.
3. **Acceptance attempts.**
   - **a01, a02: BLOCKED** (no status word, not counted). The evaluation sandbox had runtime bugs: no writable CUDA
     temp dir for NVRTC, an unmounted interpreter symlink chain, missing read-only test artifacts, and an inherited
     kernel cache. The adapter was fixed: private temp and cache directories per run, interpreter-chain mounts,
     read-only test artifacts, a process-based SHARED_GPU label, a counted GPU budget, and a queue-fair GPU lock
     wait.
   - **a03: stopped by the controller, not counted.** It was starved on the GPU lock: the RCS session's queued
     blocking `flock` jobs always took the lock before the adapter's 5 s non-blocking poll could see it free. The fix
     waits in the kernel's flock queue.
   - **a04: ACCEPTED** (counted).
     - Every gate PASS: G1 2545 tests / 0 failed; G2 165 / 165; G4, G6, G7, G8; G9 n1 / n16 / window; integrity.
     - Evaluator file set sha256 `d6153fb8…` (281 files) at frozen evaluator `bfe6d5eaee0d`.
     - Speed-up candidate / baseline (outer clock, interleaved ABBA, SHARED_GPU with 1–5 foreign processes): N=1
       median **1.168** (pairs 1.165 / 1.175 / 1.168); N=16 median **1.279** (pairs 1.279 / 1.280 / 1.142),
       threshold 1.05.
     - VRAM at N=16: 2,202 → 2,276 MiB (limit 2,678) ([P1_6_p16c1.md](P1_6_p16c1.md)).
4. **RCS PR #14** (head `8d6a2b3`, CI green). The xhigh review was **PASS_WITH_LIMITATIONS** with no BLOCKER or
   MAJOR finding ([P1_6_P16C1_PR14_REVIEW.md](P1_6_P16C1_PR14_REVIEW.md)). Its limitations:
   - the shrink is speed-conditional, and the CPU tests barely exercised it;
   - O(N·C) bucket memory and an int32 cap of N ≤ 2495.

   Merged into `main` at **`9ee1e9d`**.
5. **Pin.** `rcs_rl_vector_v2` at `9ee1e9d`. It keeps the default contact mode; `alias_free_v2` is an explicit
   opt-in. Smoke PASS including the GPU stage; the optimizer-side PR merged; **SYNCED** (receipt chain verifies).
6. **Follow-ups, PR #15** (`62caecc`, review PASS):
   - a slow-contact shrink test;
   - a per-env shrink-engaged counter, whose GPU cost head/base is 1.0006;
   - a fail-closed check refusing launches with r/dt other than at construction.

   The frozen `acceptance_v2` harness is unchanged; audit notes are kept for a future `acceptance_v3`
   ([P1_6_P16C1_FOLLOWUPS.md](P1_6_P16C1_FOLLOWUPS.md), [P1_6_P16C1_PR15_REVIEW.md](P1_6_P16C1_PR15_REVIEW.md)).

**What went wrong, in short.** Two acceptance attempts were lost to evaluation-sandbox bugs that only showed with the
real Isaac/Warp runtime inside the sandbox. One more was lost to lock starvation between the optimizer and the RCS
session's queued GPU jobs. None was counted as a verdict.

A note on absolute numbers: the absolute env-steps/s inside a04 (for example 7.46 at N=16 baseline) is far below the
baseline measurement (13.44). The GPU was shared with foreign training processes, so only the interleaved ratios are
meaningful ([P1_6_p16c1.md](P1_6_p16c1.md) run table).

## 4. Phase 2 — rubber material physics

Design: [ADR 0023](../../adr/0023-rubber-material-v1.md). Plan of the later steps:
[P2_REMAINING_PLAN.md](P2_REMAINING_PLAN.md). Phase 2 ran in parallel with Phase 1 by user instruction (D-12) on the
branches `claude/rl-rubber*`. Every step was merged into the integration branch after an xhigh review.

### 4.1 Step by step

**P2-0 — decision and recipe** (ADR 0023 accepted).
- Opt-in TLMPM rubber solver `tlmpm_rubber_v1`, single-env and batched on the Phase 1 layout.
- Newton VBD was rejected: in the probe it was 22–107× too soft, and dt / iteration dependent (ADR 0023 E6).
- Model `neo_hookean_jbar_v1`: isochoric Neo-Hookean plus a J̄ cell-averaged volumetric term. Near-incompressibility
  uses J̄ and a documented reduced κ = 20 μ (ν_eff 0.4754); the real K is recorded separately. Pure FLIP, dt = 0.5 dx/c.
- CPU probes: cantilever frequency 11.54 Hz against 11.57 Hz beam theory; elastica sag within 0.4 %; energy drift
  < 0.05 % (progress, Phase 2 detail).
- Recipe schema `rcs_rubber_material_v1`; calibration manifest `NO_REAL_DATA` (`configs/rubber/calibration/none_v1.json`).

**P2-1 — deformation without a tool** ([P2_0_P2_1_RUBBER.md](P2_0_P2_1_RUBBER.md), evidence `p2_1_rubber_001.json`,
`p2_1_followup_001.json`).
- Formal CPU run, 66 configurations: RUBBER-PHYS-01..06, 08 and 09 PASS. For example, -04 free vibration is −0.44 %
  against Euler–Bernoulli, and batched N=4 is bitwise.
- **-07 stretch & release: FAIL.** It went non-finite 3 steps after an instantaneous release at CFL 0.5. The root
  cause is the protocol: the release buckles and folds the strip at the clamp. It is not J̄ and not FLIP (ADR 0023 §18).
- Protocol v3 added an opt-in stable-step guard. Under v3, -03, -05, -12 and -07a PASS. **-07b FAILED** (energy
  1.058 % vs 1 %), and **-02 FAILED** (1 of 24 configs at 1.0203 vs 1.02, float32 static precision). Both are kept.
- GPU smoke (SHARED): N=1/4/8 L2 PASS at 0.432 / 0.621 / 0.735 env-steps/s (97,280 particles/env, 6,662 substeps).
  The **L3 band with 3 repeats FAILED** (25 of 480 values); a sounder rule was declared for P2-2.

**P2-2 — hyperelasticity** ([P2_2_RUBBER_HYPERELASTIC.md](P2_2_RUBBER_HYPERELASTIC.md), evidence
`p2_2_rubber_001.json`, `p2_2_v5_001.json`). Protocol v4 was declared before its runs.
- Opt-in fixes:
  - F8, an energy-consistent clamp: -07b went from 1.058 % to 0.33 %;
  - F9, grid-consistent traction work: -07a went from 0.96 % to 0.0047 %;
  - F10, float64 static accumulation.
- v4 results: -02, -03, -05, -06 and -07 PASS. **-08 FAILED** on validity: a step of 1.30 vs the guard 1.0 at γ 2,
  with stresses exact to 1.6e-6. **-11 refinement FAILED on Q1** (GCI 3.98 %, p_obs outside the declared window).
- **Protocol v5.** Bending scenes moved to n_t ≥ 4 (`rubber_ref_v1`). The **-11 v5 run FAILED on Q1w**: wide-strip sag
  GCI 2.16 % > 2 %. The K2 review accepted n_t 4 with its measured +0.62 % bias as a labelled limitation, without a
  v6 rule (U-P2-2).
- Throughput, workload E (41,220 particles/env, 4,996 substeps, SHARED): 0.64 / 1.35 / 1.47 / 1.52 env-steps/s at
  N=1/4/8/16.
- Review ([P2_0_P2_2_ACCEPTANCE_REVIEW.md](P2_0_P2_2_ACCEPTANCE_REVIEW.md)): P2-0 PASS, P2-1 and P2-2
  PASS_WITH_LIMITATIONS.

**P2-3 — viscoelasticity** ([P2_3_RUBBER_VISCO.md](P2_3_RUBBER_VISCO.md), evidence `p2_3_visco_001.json`,
`p2_3_visco_002.json`, `p2_3_moving_grip_001.json`).
- Model `neo_hookean_jbar_prony_v1`: Prony overstress with an exact exponential update, float64 internal state, and
  per-env internal variables cleared by a masked reset (VISC-06 PASS).
- Prony tangent gate: max ratio 0.9910 ≤ 1 (PASS).
- Moving-clamp energy books: `kick_average_v1`, RUBBER-GRIP-01..04 PASS.
- **Protocol supersession, visc v1 → v2.**
  - v1 (`visc_v1_001`): 7 PASS / **2 FAIL**.
    - -03 hysteresis: E′ at 16 / 32 Hz is −1.04 / −2.54 % against a 1 % tolerance.
    - -05 energy books: the cantilever books are 2.58 % against 0.5 %.
  - v2 re-scored only -03 and -05 with measurement / quadrature corrections and unchanged thresholds, and both PASS.
  - Cite them as **"v1 FAIL, v2 PASS (post-hoc measurement correction)"**: the diagnostic re-analysis computed the v2
    outcome on v1 data before v2 was declared (review R2). The v1 FAILs are preserved.
- Review ([P2_PREREQ_P2_3_REVIEW.md](P2_PREREQ_P2_3_REVIEW.md)): PASS_WITH_LIMITATIONS.

**P2-4 — contact, friction and coupling** ([P2_4_RUBBER_CONTACT.md](P2_4_RUBBER_CONTACT.md), evidence
`p2_4_contact_*.json`).
- **Protocol supersession, contact law v1 → v2 → v3.**
  - **v1** (`grid_node_penalty_coulomb_v1`, `contact_v1_001`): -02 penetration and -07 delay assessment PASS;
    **-01 incline, -03 impact books, -04 friction dissipation and -05 pinch-and-pull FAIL** (§3).
  - **v2** (task X1: dashpots on the realised node velocity): sliding accurate to μ ≤ 0.6, static to μ ≤ 0.8. In
    `contact_v2_001`, -02 and -03 PASS; **-01, -04 and -05 FAIL** (μ 0.8 sliding, pinch, stick-run books) (§8).
  - **X2:** the friction–normal flutter could not be removed. This is a recorded negative result (§10).
  - **v3** (task X3: patch slip-rate equalisation): `contact_v3_001` PASSes every criterion at μ 0.2–0.8 except the
    declared **-01 d stick-run energy books FAIL** (§11).
- **K2-1 (MAJOR, combined review).** The v3 equalisation has an unbounded, μ-independent patch torque. A spinning
  block reverses within 1 ms, and a sideways-dragged strip translates rigidly. v3 is therefore accepted **for
  straight translational sliding only**, and the RL task uses law v2
  ([P2_K2_COMBINED_REVIEW.md](P2_K2_COMBINED_REVIEW.md) §1).
- **Batched contact.** B1 PASS (CPU L0/L1 bytewise). The GPU per-env isolation check -09 **FAILED** under its first
  rule (1 of 108 rows) and PASSed under a v2 rule that corrects a statistical defect: 1,566 scalars, worst d/limit
  0.34. The K2 review graded this PASS_WITH_LIMITATIONS.
- **Coupling delay.** RUBBER-CONT-07 assessed the 20 ms force handoff as significant at every point. Robot-coupled
  rubber is therefore deferred (U-P2-1).
- **S1 self-penetration detector** `surface_grid_v1`: GPU check PASS (L2, TRIP, NO_FALSE_TRIP). It costs −0.05 %
  (N=1) and +0.48 % (N=8). It detects only, setting the sticky bit 8; no self-contact force exists (§12;
  K2: PASS_WITH_LIMITATIONS).

**Friction envelope for the RL task** ([P2_6_RUBBER_RL.md](P2_6_RUBBER_RL.md) §1).
- The v3 envelope (`rubber_contact_envelope_v1`) PASSed at μ ≤ 0.8. After K2-1, the v2 envelope
  (`rubber_contact_envelope_v2`) **FAILED at the 0.4 target**: dissipation 0.948 < 0.95 on the strip at n_t 4, and
  slip error 1.43 % at n_t 6. μ 0.3 PASSes everywhere.
- The first P2-6 declarations (v3 law, μ up to 0.8) were superseded before any formal run.
- **RL table friction is fixed at 0.3, not randomised** (U-P2-3).

**P2-5 — calibration interface and synthetic numerical validation**
([P2_5_RUBBER_CALIBRATION.md](P2_5_RUBBER_CALIBRATION.md), evidence `p2_5_calib_synthetic_001.json`; review
[P2_5_ACCEPTANCE_REVIEW.md](P2_5_ACCEPTANCE_REVIEW.md): GO_WITH_CONDITIONS, then Addendum A re-review GO).
- **Delivered:**
  - manifest v2 with data origin and sensors;
  - a closed REAL gate (`real_calibrated_gate_v2` refuses synthetic and v1 manifests; a CI tripwire covers the v1
    path);
  - population-batched forward models of five experiment types, bytewise equal to the single references on the CPU;
  - bounded TRF / CMA-ES / ask-tell fit API;
  - a sealed truth and a global one-shot held-out registry.

  The test is synthetic: the "truth" is the simulator at n_t 6 with sensor noise. **It is not a calibration.**
- **v1** (`rubber_calib_tests_v1`). Protocol, pilot (χ²_pilot 1.2732) and thresholds were frozen before the formal
  dataset. The formal TRF fit was **stopped for compute** before any receipt: it projected about 15 more GPU0-hours.
  CAL-04..07 are NOT_RUN for v1 (§5).
- **v2** (`rubber_calib_tests_v2`, declared before any v2 receipt). It changed only the fit method (staged bounded TRF,
  FD step 1e-2) and the budget. Formal result **FAIL** (§10):

  | CAL | Verdict | Detail |
  | --- | --- | --- |
  | 01, 02, 03, 08, 09 | PASS | interface and refusals; batched = single (bytewise); truth receipts; order and sealing; labels |
  | 04 fit quality | **FAIL** | χ²_red 2.484 > 1.910 |
  | 05 held-out | **FAIL** | 21 / 31 |
  | 06 recovery | **FAIL** | g1 −51 %, g2 +39 %, τ2 −41 %; μ_eq, τ1 and μ_table recovered |
  | 07 negative control | PASS | on one viscous row only (erratum E4: discrimination between an adequate and a deficient fit is not shown) |

- **Mechanism, as corrected by the review** (§12.1, erratum E1). Every stage after S1 stopped on its declared
  `max_nfev` cap while the cost still fell 1.6–7.1 % per evaluation. g1 was still rising toward the truth, and
  0.85 h of the 3 h budget was unused. This is non-convergence under the declared caps after a bound-active S2;
  whether a separate basin exists is **untested**. The earlier "worse valley / different basin" wording is withdrawn.
  Post-reveal and unscored, the true θ gives χ²_red 1.416 and passes 31/31 held-out.
- **v3 NOT_RUN.** A final blind re-test (new sealed dataset, stages to convergence) was allowed only within 8 GPU-h.
  Development on the revealed dataset (1.3 GPU-h) raised the estimate to 7.1–10.2 GPU-h (central about 8.6), so no
  v3 was declared or run. **The P2-5 result is the v2 FAIL** (§11.2; U-P2-6).
- **Labels.** Any fitted θ is `RESOLUTION_SPECIFIC` to `rubber_ref_v1` (n_t 4). The pilot's "≈ 10 % Prony bias" came
  from an unconverged fit and is unmeasured (U-P2-5). Guidance for users with real data is in §12.2 of the report.

**P2-6 — RL-ready rubber** ([P2_6_RUBBER_RL.md](P2_6_RUBBER_RL.md), evidence `p2_6_*.json`; review
[P2_6_ACCEPTANCE_REVIEW.md](P2_6_ACCEPTANCE_REVIEW.md): GO_WITH_CONDITIONS, conditions done in §12 of the report).

The env: `open_vector_env(backend="kinematic_grip_batched")` with `RubberStripTaskFamily`, task `rubber_strip_drag_v2`,
profile `rubber_rl_env_v2` (contact law v2, μ 0.3), slots L 100/125/150 × t 4/5/6 mm. It has 6-D grip actions, a
78-dim deployable observation, an opt-in 106-dim critic observation, the shaping reward `rubber_tip_tracking_v1`, and
the evaluator `rubber_tracking_eval_v1`.

| Row | Verdict | Key numbers |
| --- | --- | --- |
| RL-01 contract | PASS | CPU N 1 / 3, GPU N 1 / 8, both profiles |
| RL-02 parity | PASS | CPU L0 bytewise; GPU L2 60 steps; GPU L3 0 / 90 outside the band (max M/U 0.774) |
| RL-03 randomisation, RL-04 isolation | PASS | -09 v2 rule: 3,780 scalars, worst d/limit 0.469 |
| RL-05 scripted sanity v2 / v3 | **FAIL / FAIL** (preserved) | v2: better than zero action in 5/8 (erratum: first written as 4/8), 0 successes. v3: 8/8, 0 successes (the tip reached the goal, then swung out) |
| RL-05 v4 (declared as the last attempt) | PASS | better in 7/8, 5 evaluator successes |
| RL-06 long rollout | PASS | N=8, 400 steps, 53 partial resets, VRAM growth 0. Reset p95 5.0 s, or 0.093 s with `OPENBLAS_NUM_THREADS=1` |
| RL-07 throughput | PASS (SHARED_GPU) | N=8 / N=1 gain 1.46. Full slot set 0.658 / 1.019 / 0.960 / 1.009 env-steps/s at N=1/4/8/16. VRAM 1,228 MiB at N=8 |
| RL-08 dt / spacing | PASS | ΔO1/L 8.8e-6, ΔO3/O3 3.2e-5. RL-08 b Q1w at L/t 37.5: +0.71 % (trigger 1 % not met) |
| **RL-09 Full Native** | **PASS** | `native_rubber_001` (GPU0, SHARED): 100/100 steps, 1.81 s per warm step with cameras and recording. 30 headless replays (GPU1): receipts identical; material inputs bitwise 30/30; 0 / 46 statistics outside `rubber_l3_band_v4` (max M/U 0.654); held-out 10/10 inside; invalid bits 0; bridge error 7.45e-9 m |
| RL-10 provenance | PASS | 7 formal runs (after RL-12) |
| RL-11 Phase 1 non-regression | PASS | final rerun: `.venv` 2,932 passed / 0 failed; Torch 414 / 416 passed; workload A contract N=2 PASS |
| **RL-12 host memory** (`rubber_rl_tests_v5`, added after the review) | **PASS** | Before the fix, host RSS grew 2.19 MiB per reset (the F8 inverse cache never evicted). After the bounded LRU: RSS flat at 2,298.0 MiB over resets 10–80, growth 0.0 MiB (bound 32), cache at its cap 16 |

- **Protocol supersessions in P2-6.**
  - `rubber_rl_tests_v1` + `rubber_strip_drag_v1` (law v3, μ ≤ 0.8) were superseded by K2-1 and never run formally.
  - The v2 pair is the declaration of record.
  - v3 and v4 changed RL-05 only. RL-05 v2 FAIL → v3 FAIL → **v4 PASS**, with v4 declared as the final scripted
    attempt and every FAIL kept.
  - v5 adds RL-12 only.
- **Throughput against Phase 1.** At N=8 the rubber env is about 10× slower per env-step than workload A, measured
  in the same session at 9.7 env-steps/s. The main reason is 5,226 substeps per control step instead of 200 (§4).
  The reduced-fidelity `rubber_rl_v1` (n_t 3) runs 3.277 env-steps/s at N=8.
- **Supplementary `EXCLUSIVE_GPU` RL-07 row.** The Phase 2 review asked for this row (F2, condition M2). It was rerun
  unchanged on an idle GPU with a continuous co-tenant sampler, as an additional labelled row (evidence
  `docs/evidence/rl_ready/p2_6_rubber_rl07_exclusive_001.json` and its note).
  - **Setup:** the exact declared RL-07 runner and settings (`rubber_rl_tests_v2`), commit `c21ae8d` (clean),
    RTX 5090 GPU1, `OPENBLAS_NUM_THREADS=1`.
  - **Sharing:** **EXCLUSIVE_GPU**. 275 samples at 2 s with a largest gap of 2.0 s, and 0 foreign-process samples.
  - **Status:** the formal SHARED_GPU row (`gpu_rl_b_001`) and its PASS stay the row of record; nothing was
    re-scored.
  - **Numbers:** env-steps/s is N / median warm-step wall over 5 warm steps.

  | Set | N | env-steps/s exclusive | env-steps/s shared (formal) | VRAM MiB (process peak) |
  | --- | --- | --- | --- | --- |
  | full slot set | 1 | 0.648 | 0.658 | 812 |
  | full slot set | 4 | 1.002 | 1.019 | 972 |
  | full slot set | 8 | 0.942 | 0.960 | 1,228 |
  | full slot set | 16 | 0.996 | 1.009 | 1,676 |
  | uniform L125_t4 | 1 | 0.624 | 0.624 | 876 |
  | uniform L125_t4 | 8 | 1.144 | 1.160 | 1,196 |
  | uniform L125_t4 | 16 | 1.209 | 1.235 | 1,548 |
  | `rubber_rl_v1` (reduced fidelity) | 1 | 1.195 | 1.221 | 908 |
  | `rubber_rl_v1` (reduced fidelity) | 8 | 3.236 | 3.277 | 1,004 |

  - **N8/N1 (median, matched 5-step windows):** full slot set **1.454×** (shared 1.459×), uniform 1.835× (1.860×),
    reduced fidelity 2.709× (2.684×).
  - **Conclusion:** the shared formal numbers were within about 2 % of exclusive. Every other Phase 2 throughput row
    and the Native row remain SHARED_GPU.

### 4.2 Material status

`SIMULATOR_ONLY_UNCALIBRATED` everywhere. No recipe is promoted. No synthetic manifest can pass the REAL gate. The
real-data protocol `rubber_calib_tests_real_v1` exists only as an **outline** inside `rubber_calib_tests_v1.json`
(`real_data`: "to be declared when real data exist"). It has no thresholds and is NOT_RUN; it must be declared and
frozen before any real held-out evaluation.

### 4.3 Reviews of Phase 2

| Review | Verdict |
| --- | --- |
| [P2_0_P2_2_ACCEPTANCE_REVIEW.md](P2_0_P2_2_ACCEPTANCE_REVIEW.md) | P2-0 PASS; P2-1, P2-2 PASS_WITH_LIMITATIONS; merge GO |
| [P2_PREREQ_P2_3_REVIEW.md](P2_PREREQ_P2_3_REVIEW.md) | prerequisites and P2-3 PASS_WITH_LIMITATIONS; moving-clamp energy PASS |
| [P2_K2_COMBINED_REVIEW.md](P2_K2_COMBINED_REVIEW.md) | merge GO; contact v1 PASS (as a record), v2 PASS_WITH_LIMITATIONS, v3 translational only (K2-1 MAJOR); S1 PASS_WITH_LIMITATIONS; Q1w option (c) |
| [P2_5_ACCEPTANCE_REVIEW.md](P2_5_ACCEPTANCE_REVIEW.md) | GO_WITH_CONDITIONS (C1–C4 errata); Addendum A re-review GO |
| [P2_6_ACCEPTANCE_REVIEW.md](P2_6_ACCEPTANCE_REVIEW.md) | GO_WITH_CONDITIONS; F1 host memory → RL-12; F2–F4 API gaps fixed; Z-c RL-07 stays SHARED |

### 4.4 Where sources disagree

- **P1-2 batching gain.** [P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) §5 (twice) cites "P1-2's matched-load
  result (N8/N1 1.45×, N16/N1 1.42×)". [P1_1_P1_2_MILESTONES.md](P1_1_P1_2_MILESTONES.md) (review fix F3) supersedes
  those figures: they compared unequal windows. The matched-window values are 1.40× and 1.37×. **Resolved:** the
  Phase 2 review recomputed them from the raw step times (1.396× / 1.370×; the unequal windows give 1.447× /
  1.418×), so 1.40× / 1.37× is correct. The exclusive table's §5.1 ratios over N=1 (1.43–1.49×) carry the same window
  bias and are about 4–5 % high ([PHASE2_ACCEPTANCE_REVIEW.md](PHASE2_ACCEPTANCE_REVIEW.md) §4.2). An additive
  erratum now stands in [P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) §5.
- **p16c1 absolute speeds.** The baseline measurement gave 13.44 env-steps/s at N=16, the a04 baseline runs 7.46–7.83
  (shared GPU). Only the interleaved ratios are used.
- **RL-05 v2 count.** [P2_6_RUBBER_RL.md](P2_6_RUBBER_RL.md) §0 / §5 and ADR 0023 §23.7 first said the v2 scripted
  controller beat the zero action in 4/8 envs; the evidence shows 5/8 (envs 2, 4, 5, 6, 7). Additive errata are in
  both files. The verdict (FAIL: 0 successes) is unaffected.
- **Phase 2 Done box 2.** ADR 0023 §23.3 grades it PASS_WITH_LIMITATIONS. This report says PARTIAL because recorded
  FAILs (Q1w, μ 0.4, -01 d) and unimplemented self-contact remain.
- **P2-5 mechanism.** The report's §9–§10 explain the v2 FAIL as a "resolution bias" and a "different basin". Erratum
  E1 (§12.1) withdraws both, and this report follows §12.1.
- **The rubber env docstring** names task `rubber_strip_drag_v1`; the default and the declaration of record are
  `rubber_strip_drag_v2`.

## 5. Known limitations and open items

**Phase 1**
- **CUT** is PARTIAL: the frozen v5 CUT is not batched (U-P15-1).
- **Cameras, vision profiles and physics recorders** are UNSUPPORTED in the batched path (U-P15-2).
- **Randomisation:** no per-episode geometry or material randomisation inside a live batch (U-P15-3).
- **Fidelity:** workload A and B1 are `reduced_fidelity`. B2 grip slip is about 1.5 mm, against about 1 µm in Native.
  B2 G1/G3/G4/G5 with the profile are NOT_RUN.
- **NOT_RUN or BLOCKED:**
  - L4 RUN_TO_RUN: BLOCKED (frozen mixed atomics);
  - M3 (N ≥ 32), LR-A2, LR-E: NOT_RUN;
  - EXCLUSIVE partial-reset latency, batched cold rows, CUT N=4: NOT_RUN;
  - batched replay of the Native v7_2 CUT, Lab CUT: NOT_RUN.
- **Formal table code:** it measured `691b571`, before review fix F1.
- **Acceptance contract:** `acceptance_v2` does not gate N=8, and its reset / VRAM gates are candidate-reported. Audit
  notes wait for an `acceptance_v3` ([P1_6_P16C1_FOLLOWUPS.md](P1_6_P16C1_FOLLOWUPS.md)).
- **`alias_free_v2`:** N ≤ 2495 (int32), O(N·C) memory. The external pin is `rcs_rl_vector_v2` at `9ee1e9d`; no
  record shows a pin of `main` after PR #15 (`62caecc`).
- **Open physics item:** the WP-17 seed-512 final cut-code difference against WP-16 (4,266 particles).
- **Pre-existing test failure:** `tests/test_wp12_phase5_native_success.py::test_reproduces_phase2b_records` fails
  where `artifacts/wp12` links the main checkout's artifacts. It is unrelated, and WP-12 records are untouched
  (progress, consolidation entry).

**Phase 2**
- **Friction:** fixed at μ 0.3 in the RL task; law v3 is translational-only (K2-1); a torque-bounded equalisation is
  future work.
- **Coupling:** robot-coupled rubber (B3 / C1) is deferred; the grip is kinematic and the wrench is reported only
  (U-P2-1).
- **Self-contact:** NOT_IMPLEMENTED_V1, detection only. A pass-through faster than the check interval can be missed.
- **Throughput:** about 1 env-step/s at N=8–16 at full fidelity. RL-07 is SHARED_GPU with only start/end co-tenant
  samples. No EXCLUSIVE Native row exists.
- **Bending bias:** +0.62 % (L/t 20) and +0.71 % (L/t 37.5) at n_t 4; the -11 v5 FAIL is kept.
- **Randomisation:** geometry is per env, not per episode. `reset(seeds=None)` repeats the episode's draws.
- **RL-05:** the scripted sanity passed only with the third declared controller, and its success depends on the goal
  direction.
- **Calibration:** the P2-5 fit procedure is not validated (v2 FAIL; v3 not run). Real data is absent.
- **BLAS threads:** `threadpoolctl` is absent from both venvs, so `OPENBLAS_NUM_THREADS=1` is a documented user
  setting, not enforced. Reset bits depend on the BLAS thread count.
- **CI:** GitHub CI has no Torch, so the P2-6 env tests are NOT_RUN there. Local Torch-venv runs are the gate.
- **κ = 20 μ model-form error:** the reduced bulk ratio (ν_eff 0.4754 instead of about 0.4995 for real rubber) gives
  J − 1 = +1.2 / +2.5 / +5.4 % and nominal stress −2.1 / −2.7 / −4.2 % at uniaxial stretch 1.25 / 1.5 / 2.0, and
  bending stiffness −1.6 % ([ADR 0023](../../adr/0023-rubber-material-v1.md) §6). Volume change and confined
  (pinch) responses must not be compared with real rubber as if they were physics.
- **Default contact law:** the code default is law v1 (`rubber_contact.py`), which has the worst formal record
  (RUBBER-CONT-01 / 03 / 04 / 05 FAIL). The friction envelope is enforced only in the RL task family
  (`rl/rubber_family.py`). Both recipes carry a nominal table friction of 0.8, so a `LiveRubberMaterial` /
  `BatchedRubberMaterial` built directly from a recipe without an explicit law runs v1 at μ 0.8 with no refusal,
  outside every established envelope row. Changing the default or enforcing the envelope in the material classes
  would be a versioned decision.
- **Native visual smoke:** the rendered-clamp visual fix (`measured_pose_v1`) has not run in Kit (visual smoke
  NOT_RUN).
- **Merge and pin:** Phase 2 PR, merge and the new external descriptor are PENDING (§9).

**Calibration results do not plug directly into the RL task** (Phase 2 review F1). The calibration interface is
ready for ingesting, gating and one-shot held-out bookkeeping of real data, not for handing a fit to the RL task:
- **Friction law.** P2-5 fits friction with contact law v3 (translational sliding only), so a fitted `μ_table` is a
  law-v3 coefficient. The RL task uses law v2 with μ fixed at 0.3.
- **Resolution.** Fitted θ are `RESOLUTION_SPECIFIC` to n_t 4 (`rubber_ref_v1`), while the RL slots run at
  n_t 4 / 5 / 6 (t 4 / 5 / 6 mm).
- **Material source.** The RL task draws its material from the fixed ranges in `rubber_strip_drag_v2`, not from a
  recipe.

A user with real data therefore still needs a new task-config version, a law-v2 friction fit (or an equivalence
check at the target μ), and the resolution bias shift for n_t 5 / 6.

## 6. User decisions pending

From 「待使用者決定」 in [RL_READY_PROGRESS.md](../../RL_READY_PROGRESS.md). Each default is in force until the user
decides otherwise. None blocks work.

| Id | Question | Default in force | Alternative |
| --- | --- | --- | --- |
| Data | Real rubber measurements | None on this machine. Calibration status stays `SIMULATOR_ONLY_UNCALIBRATED` | Provide data. The interface and guidance are ready (P2-5 §12.2) |
| U1 | Where optimizer-written candidate code runs | GPU1 only, not GPU0 | — |
| U2 | Speed gate | `acceptance_v2` (D-14): G9 times steps 229–400, N ∈ {1, 16}, 3 ABBA pairs, N=16 median ≥ 1.05 with all 3 pairs faster, every N ≥ 0.97; VRAM ≤ 1.10 × B + 256 MiB; reset p95 ≤ 1.25 × B + 2 ms (shared GPU: WARN up to + 10 ms). `acceptance_v1` frozen, unused | — |
| U3 | Per-cycle budget | Research agent $20 / 6 h plus one xhigh review (about 1 working day, 3–5 GPU-h) | — |
| U4 | Web access for the research agent | None | — |
| U-P15-1 | Batch the frozen `tlmpm_v5` CUT? | No (about 3–4 agent-days deferred). Batched CUT uses the opt-in v7_2 scene; the CUT parity row stays PARTIAL | Option (a): batch v5 |
| U-P15-2 | Batched camera / vision profile | None; cameras UNSUPPORTED in the batched path | — |
| U-P15-3 | Per-episode geometry / material randomisation in a running batch | No; per-env seeds plus rebuild | — |
| U-P2-1 | Robot-coupled rubber (rubber wrenches fed back to the Isaac Lab robot, tasks B3 + C1) | Out of Phase 2; kinematic grips (the 20 ms handoff was assessed significant) | — |
| U-P2-2 | RUBBER-PHYS-11 Q1w (GCI 2.16 % vs 2 % at n_t 4) | Accept n_t 4 with its measured +0.62 % bias as a labelled limitation; the v5 FAIL stays recorded (confirmed by the K2 review) | n_t 6 (about 5× cost per step) |
| U-P2-3 | Rubber friction in the RL task | Fixed at μ 0.3 with contact law v2 (the v2 envelope fails at 0.4; v3 translational-only). Higher μ is a labelled limitation | A torque-bounded equalisation (future work) |
| U-P2-4 | P2-5 fit compute | Declare v2 changing only the fit method, with a hard budget (done; result FAIL) | Fund the unchanged v1 fit (about 10–15 GPU0-h) |
| U-P2-5 | Prony identifiability at n_t 4 | Keep fitted θ labelled resolution-specific; the bias size is unmeasured | Fit with an n_t 6 forward model (several-fold GPU cost) |
| U-P2-6 | P2-5 synthetic recovery FAIL | The v2 FAIL is this round's result | Fund a v3 (about 9–10 GPU-h: branch-modulus parametrisation, joint stage from the start, Broyden Jacobian updates, new sealed dataset) |

Two further items were added after the Phase 2 review (§4.4) as U-P2-7 and U-P2-8:

| Id | Question | Default in force | Alternative |
| --- | --- | --- | --- |
| U-P2-7 | BLAS threads for the rubber reset | `OPENBLAS_NUM_THREADS=1` is a documented user setting (README, P2-6 report §8), not enforced; reset bits are comparable only across processes with the same setting | Install `threadpoolctl` in the venvs, or add an in-code guard |
| U-P2-8 | Torch in GitHub CI | Keep CI as is; every Torch-gated test (all P2-6 env tests, the Phase 1 vector-env tests) is NOT_RUN there and runs locally before merges | A CPU Torch CI job (CI already takes about 2 h) |

## 7. Resource use

GPU time is reported per package where the reports give it; there is no plan-wide total. Hardware: two RTX 5090
32 GB (driver 570.211.01), shared with other projects. One RCS job per GPU ran under a per-GPU flock (D-07).

| Package | Reported GPU time | Label | Source |
| --- | --- | --- | --- |
| P1 formal exclusive table | GPU1 lock 18:18:26–21:00:01, about 2.7 h; 23 runs | 20 EXCLUSIVE / 3 SHARED | [P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) §1 |
| P1-4 LR-A | 5.06 h (one run) | SHARED_GPU | [P1_4_LONG_ROLLOUT.md](P1_4_LONG_ROLLOUT.md) |
| P1-6 acceptance (estimate) | about 1.69–1.75 GPU-h per acceptance on an exclusive GPU | estimate | D-14; [P1_FOLLOWUP.md](P1_FOLLOWUP.md) |
| P1-6 `p16c1` research run | 1 session, $6.02, 67 min. The acceptance wall times are in the external optimizer repository | SHARED_GPU | progress |
| P2-5 v1 phase | about 8.3 GPU-h (design probes 0.4, pilot 1.07 + 0.16, formal data 1.07, stopped fit 3.3, stopped negative control 2.3) | SHARED_GPU | [P2_5_RUBBER_CALIBRATION.md](P2_5_RUBBER_CALIBRATION.md) §7 |
| P2-5 v2 | development 1.75; formal fit 2.15 (lock hold); negative control 0.63; held-out 0.2; diagnostic 0.2 | SHARED_GPU | same, §9–§10 |
| P2-5 v3 development | 1.3 GPU-h | SHARED_GPU | same, §11.2 |
| P2-6 formal GPU rows | run windows 45 / 79 / 55 / 95 / 22 / 38 / 95 / 20 min (a, b, c, d, RL-05 v3, RL-05 v4, RL-06 re-measure, RL-12) | mostly SHARED; a and d EXCLUSIVE_AT_SAMPLES | [P2_6_RUBBER_RL.md](P2_6_RUBBER_RL.md) §4, §10, §12 |
| P2-6 RL-05 v4 development | about 1.0 GPU-h | SHARED | same, §5; review §1 |
| P2-6 Native RL-09 | launcher wall 188.9 s on GPU0, plus 30 replays on GPU1 | SHARED_GPU | same, §9.2 |

- **Sums.** My sums of these items, not stated in the reports, are about 14.5 GPU-h for P2-5 and about 7.5 h of
  lock-window wall time for the P2-6 formal rows.
- **Against the plan.** [P2_REMAINING_PLAN.md](P2_REMAINING_PLAN.md) estimated about 11 GPU-hours for P2-5 and P2-6
  together, so this was exceeded.
- **Not totalled.** Earlier Phase 2 GPU checks (P2-1 smoke, P2-2 L3 and workload E, P2-3 VISC-10, P2-4 -09, S1 231 s)
  appear only per row in their reports.
- **Shared GPUs.** Nearly every Phase 2 GPU row is SHARED_GPU. A long-running process of another project was present
  on GPU0 throughout and was never touched.

## 8. Reproduction

Paths are repository-relative.
- **Raw outputs:** `artifacts/rl_ready/` (git-ignored; a link to a large data disk).
- **Compact evidence:** `docs/evidence/rl_ready/`.
- **Media:** `docs/media/p2_6/`.
- **Placeholders:** `<tmp>` is a large temporary directory, `<gpu lock>` is the per-GPU flock file of the shared
  machine, and `<overlay>` is the pytest overlay built by `scripts/rl_ready/make_pytest_overlay.sh` (D-09; nothing is
  installed into `.deps`).

```bash
# CPU suites (project venv), once with CUDA hidden and once with a GPU visible
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider tests/
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_repo_naming.py
# Torch suites (Phase 1 vector env + rubber RL env), Warp/Torch CPU device
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONPATH=<overlay>:src .deps/isaacsim-venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_vector_env_*.py tests/test_rl_*.py tests/test_perf_acceptance.py \
  tests/test_rubber_rl_*.py tests/test_rubber_device_staging.py tests/test_rubber_scripted.py tests/test_native_rubber_*.py
# Isaac Lab workload B tests (Warp-CPU MJWarp)
PYTHONPATH=<overlay>:src .deps/isaaclab-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_rl_lab_*.py tests/test_vector_env_contract.py

# Phase 1 throughput row (one process per (N, variant); one device HashGrid per process)
flock <gpu lock> env CUDA_VISIBLE_DEVICES=<g> TMPDIR=<tmp> PYTHONPATH=src .deps/isaacsim-venv/bin/python \
  scripts/rl_ready/run_perf_acceptance.py --variant-json <variant.json> --mode throughput --num-envs <N> \
  --seeds 512 --warmup-steps 20 --steps 100 --device cuda:0 --gpu-index <g> --out artifacts/rl_ready/<new_run>
# A full acceptance without the external optimizer (frozen contract acceptance_v2)
flock <gpu lock> env TMPDIR=<tmp> .deps/isaacsim-venv/bin/python scripts/rl_ready/run_perf_acceptance_local.py \
  --config configs/rl_benchmark/acceptance_v2.json --baseline-checkout <B> --candidate-checkout <C> \
  --proposal-json <P.json> --gpu-index <g> --out artifacts/rl_ready/<new_run>
# Batched CUT (P1-5) and the long-rollout supervisor (P1-4)
.deps/isaacsim-venv/bin/python scripts/rl_ready/run_cut_batched.py --num-envs <N> --gpu-index <g> --run-name <name>
.deps/isaacsim-venv/bin/python scripts/rl_ready/supervise_rollout.py --help

# P2-6 rubber GPU rows (sets OPENBLAS_NUM_THREADS=1, blocking per-GPU lock)
# Row names as run_rubber_rl_gpu.py accepts them: RL01 RL02b RL02c RL03 RL04b RL05 RL06 RL07 RL08 RL08b
# (RL-05 v3 / v4, RL-12: scripts/rubber/p2_6/run_rubber_rl05_v{3,4}.py, run_rubber_rl12_hostmem.py)
scripts/rubber/p2_6/queue_gpu_rows.sh <g> <rows, e.g. RL01,RL02b> artifacts/rl_ready/<new_run>
# P2-6 Native host (Kit; refuses unless 14 GiB + 4 GiB are free), replays, comparison: P2_6_RUBBER_RL.md §9.2
.venv/bin/python scripts/rubber/p2_6/launch_native_rubber.py --gpu 0 --task-config rubber_strip_drag_v2 ...
# P2-5 calibration: scripts/rubber/p2_5/{gen_synthetic,run_pilot,run_fit_v2,run_heldout,reveal}.py (P2_5 report §3–§10)
```

The exact command of every formal row is in its report and in the run's `meta.json` under `artifacts/rl_ready/`.
Protocols and declarations are SHA-pinned in `configs/rubber/` and `configs/rl_benchmark/` and checked by the
`tests/test_*_protocol*.py` / `*_declarations.py` pin tests.

## 9. Phase 2 merge and pin (PENDING)

**Status: PENDING.** The orchestrator fills this in after the merge.

| Item | Value |
| --- | --- |
| P2-5 / P2-6 branches merged into `claude/rl-ready-rubber-physics` | PENDING — merge commit `<sha>` |
| xhigh review of the Phase 2 PR | PENDING — verdict `<verdict>`, report `<path>` |
| Phase 2 PR | PENDING — `#<number>` |
| Merge into `main` | PENDING — merge commit `<sha>` |
| GitHub CI on the merged head | PENDING — `<run id, counts>` |
| New external descriptor | PENDING — `<descriptor id>` pinned at `<sha>`; smoke `<result>` |
| Sync status | PENDING — `<RCS_PR_PENDING / MERGED_RCS_PENDING_OPTIMIZER_SYNC / SYNCED>` |
| Phase 2 Done box 6 | PENDING until SYNCED |
