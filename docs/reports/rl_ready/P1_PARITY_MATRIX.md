# Phase 1 — final feature-parity matrix (RL_GPU_HEADLESS vs FULL_NATIVE_REFERENCE)

Date: 2026-11-10. Branch `claude/rl-p1-consolidate`, from integration commit `181ed37`, which contains
P1-0…P1-5, P1-1b with B2, R0 and the review fixes. This matrix is for the Phase 1 xhigh acceptance review. The
rows follow [P1_5_PLAN.md](P1_5_PLAN.md) §3. Row 3 (TRIM) and row 4 (robot) are split by workload and mount, and
row 16 (long rollout) is new.

How the statuses were set:

- Every status comes from merged evidence. Nothing is filled in from intent.
- Evidence files live in `docs/evidence/rl_ready/`. Each is cited as `file` @ the commit that added it.
- Reports live in `docs/reports/rl_ready/`.
- Status words: PASS / PARTIAL / FAIL / UNSUPPORTED / NOT_RUN / N/A.
- A PASS on a reduced workload names that label in the Label column.
- Batched performance numbers in the rows below are `SHARED_GPU` unless stated. The formal `EXCLUSIVE_GPU` rows
  (GPU1, 2026-11-10 18:18–21:00 KST, 20 of 23 runs exclusive) are in [P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) (`p1_exclusive_001.json` @ `ff4133f`, merged 8e503e1): workload A N=1/4/8/16 frozen_v1
  6.75/10.06/9.62/9.83 and alias_free_v1 9.74/13.85/14.01/13.56 env-steps/s; single_env_headless 7.57 (s512) /
  7.21 (s15); CUT N=1/8/16 6.85/10.35/10.22; B1 N=1/4/8 4.96/7.66/8.67 (N=16 8.79 SHARED); B2 `mjwarp_grip_v1` N=4 7.37.

Runtimes:

- `RL_GPU_HEADLESS` is the batched runtime behind `open_vector_env`, with two backends:
  - workload A `kinematic_ideal_tracking_batched`: the edge is replayed or ideally tracked, with no robot;
  - workload B `isaac_lab_newton`: N Franka worlds on Newton/MuJoCo-Warp.
- `FULL_NATIVE_REFERENCE` is the unchanged Native Isaac Sim 6.2 runtime (PhysX Franka, physical grasp, RTX
  cameras).

## Matrix

| # | Capability | RL_GPU_HEADLESS | FULL_NATIVE_REFERENCE | Evidence (commit + file) | Gap | Label |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Scene / goal / material schema | **PARTIAL**. `rcs_scene_v1` / `rcs_goal_v1` go through the same `compile_task`. Seeded TRIM uses `TaskFamily`; configured `tlmpm_v7_2` scenes use `ConfiguredTaskFamily` (P1-5 T5.2). Since this branch, configured scenes can also be selected per env through `tasks=` (`{scene, goal, planner_options, family_id, seed}` or the workload id `cut_v7_2_v1`). TRIM and configured slots in one batch are refused by the compiled layout (`UNSUPPORTED_NONUNIFORM_BATCH`: K 2 vs 4, 1200 vs 700 steps, reward kind) | **PASS**: every configured scene (v4–v7_2) and registered task | `p1_2_milestones_001.json` @ `f0a544b`; `p1_5_cut_001.json` @ `354fc13`; this branch `53b915e` (`tests/test_rl_consolidation_cpu.py`, `tests/test_vector_env_consolidation.py`) | Solvers other than `tlmpm_v7_2` are refused (`UNSUPPORTED_SOLVER`). A batch with mixed layouts is refused | PARTIAL (labelled: other solvers `UNSUPPORTED_SOLVER`; one batched layout per batch) |
| 2a | CUT, batched (versioned v7_2 scene `bar_flat_board_tlmpm_v7_2_v1`, workload C2) | **PASS for the declared v7_2 workload**. N=1/4/8/16: 35/35 reference episodes evaluator PASS (36-frame stream), L2 PASS, final codes within the single band. CPU L0 bitwise vs the frozen single v7_2 (2.5 mm) | **PASS**: Native v7_2 CUT episode, T5.8 (2 pieces 0.5 / 0.5, HOME PASS) | `p1_5_cut_001.json` @ `354fc13` (merged `03a2f72`); [P1_5_CUT.md](P1_5_CUT.md) §5–§6 | Batched replay of the Native v7_2 recording (N=1 / N=4) NOT_RUN. Lab/B1 CUT (T5.9) NOT_RUN. No EXCLUSIVE CUT rows | `reduced_fidelity` (ideal tracking, no robot, reaction not fed back) |
| 2b | CUT, frozen v5 (WP-12 Demo 1, C1) | **single_env_headless only**: Demo 1 replay 4/4 evaluator PASS, final codes = Native in 3/4 (2 particles off in 1/4). Batched v5: **UNSUPPORTED** (`UNSUPPORTED_SOLVER`, by design) | **PASS** (WP-12 Demo 1 Native v5) | `p1_5_cut_001.json` @ `354fc13`; [P1_5_CUT.md](P1_5_CUT.md) §3 | The frozen v5 / v7 / v7_1 physics is not batched | CUT row overall: **PARTIAL** (as declared in P1_5_PLAN §2) |
| 3a | TRIM, workload A (replayed / ideal edge, `tlmpm_v7_2`, law 8, 1 mm / 4 mm / 0.1 ms) | **PASS**. M0 / M1 / M2: final codes = Native, frozen evaluator PASS (M2 N=8 24/24, N=16 16/16). Replay parity: final codes = Native 4/4 per seed; positions within band v2 | **PASS** (WP-17 seeds 512 / 15) | `p1_2_milestones_001.json` @ `f0a544b`; `p1_0_baseline_001.json` @ `9f46fc9`; `review_fix1_001.json` @ `1f7d39f` (band v2) | No EXCLUSIVE rows | `reduced_fidelity` (ideal tracking, no robot dynamics, reaction not fed back) |
| 3b | TRIM, workload D (single_env_headless replay of WP-17) | **PASS**: 3 / 3 per seed | **PASS** | `p1_0_baseline_001.json` @ `9f46fc9`; [P1_0_BASELINE.md](P1_0_BASELINE.md) | — | reference wrapper (not batched) |
| 3c | TRIM, workload B1 (rigid knife mount) | **PASS**: S3 N=2 / 4, seeds 512 / 15, frozen evaluator PASS in every env | **PASS** | `p1_1b_s2_s3_001.json` @ `78a211b`; [P1_1B_ROBOT_BATCHING.md](P1_1B_ROBOT_BATCHING.md) S2/S3 | Lab long run (LR-E) NOT_RUN | `reduced_fidelity: rigid_tool_mount_no_grasp, finger_opening_fixed_nominal` |
| 3d | TRIM, workload B2 (physical grasp) | **PASS with the opt-in `mjwarp_grip_v1`**: S3 N=2 and N=4, seeds 512 / 15, frozen evaluator PASS in every env, no termination. Default point profile: **FAIL** (GRASP_LOST in every env) | **PASS** | `p1_1b_b2_001.json` @ `e1b6634` (merged `52dfcf6`); [P1_1B_ROBOT_BATCHING.md](P1_1B_ROBOT_BATCHING.md) "B2 grasp follow-up" §3 | Slip about 1.5 mm and rotation about 15 mrad, against about 1 µm / 1e-4 rad in Native | listed difference `contact_solver_profile: mjwarp_grip_v1` (MuJoCo-Warp options, `physical_calibration: NOT_VALIDATED`) |
| 4a | Robot articulation / contact, B1 | **PASS (reduced)**: N=1/4/16 batched articulation; delayed-wrench coupling; G1 bitwise isolation (`run_to_run`); G5 L0 bitwise, G5 L3 (frozen single material = batched final codes); T2 host-sync audit; `open_vector_env(backend="isaac_lab_newton")` contract PASS | **PASS** (PhysX physical grasp) | `p1_1b_s1_001.json` @ `660691c`; `p1_1b_s2_s3_001.json` @ `78a211b`; `p1_2b_001.json` @ `c578348` | Second Franka parked | `reduced_fidelity: rigid_tool_mount_no_grasp` |
| 4b | Robot articulation / contact, B2 | **PARTIAL**. With `mjwarp_grip_v1` the grasp holds 1196/1200 steps, settle passes 3/3 and the deterministic settle passes. The material SIGSEGV after a lost grasp is guarded by `material_input_domain_v1` (bit 128); the paired N=4 crash configuration completes. Since this branch, `grip_contact_profile` passes through `open_vector_env` (Warp-CPU Lab test) and is recorded in `receipt()["grip"]` and `fidelity()` | **PASS** | `p1_1b_b2_001.json` @ `e1b6634`; this branch `53b915e` (`tests/test_rl_lab_grip_profile.py::test_open_vector_env_passes_mjwarp_grip_v1_to_the_lab_env`); ADR 0022 implementation note "P1 consolidation" | B2 G1/G3 deterministic isolation, G4 vs the single Lab env and G5 replays with the profile: NOT_RUN. It is not proven that the guard removes every SIGSEGV path. Slip is not at Native level | opt-in profile, listed difference |
| 5 | Cut history / persistent severing | **PASS for the solver contract**. Per-env packed codes, frontier / link and pending state are batched. L2 monotonicity holds every step (0 violations). The 36-frame stream integrity is checked on CUT. A partial reset restores history (CPU bitwise) | **PASS** (oracle history; `cut_events_v1` opt-in, v5 only) | `p1_5_cut_001.json` @ `354fc13`; `p1_2_milestones_001.json` @ `f0a544b`; `p1_2b_001.json` @ `c578348` (evaluation stream) | `cut_events_v1` and cut-aware surfaces: `UNSUPPORTED_RECORDER`. L4 RUN_TO_RUN: BLOCKED (frozen v5 `_particle_contacts` mixes add/max atomics) | recorders UNSUPPORTED (listed) |
| 6 | Cameras RGB-D / vision | **UNSUPPORTED** in batched: `UNSUPPORTED_SENSOR` on both backends; the kit-less Newton path has no RTX renderer | **PASS** (camera rig v5: global D455 + two wrist D405; the global D455 native calibration is NOT_VALIDATED) | No `docs/evidence/rl_ready` file. Refusal tests: `tests/test_vector_env_p12b.py::test_isaac_lab_newton_dispatch_refusals_without_isaac_lab`. Native: `docs/PROGRESS.md` camera sections | No batched vision profile | UNSUPPORTED (out of Phase 1 scope, U-P15-2; vision data comes from the Native reference) |
| 7 | Vectorized reset / step | **PASS** (`SHARED_GPU`). One process; one launch per kernel for all envs (nsys). Masked partial reset: non-reset envs are CPU bitwise; GPU outcome level only. N=1/4/8/16 measured; 1.40× (N=8) / 1.37× (N=16) aggregate over N=1 in matched windows; about 1.24–1.26× vs the frozen single env (inferred). Partial-reset p95 about 7 ms (TRIM) / 3.2 ms (CUT M1) | N/A (N=1 by design; `reload_in_place` reset) | `p1_2_milestones_001.json` @ `f0a544b`; `review_fix1_001.json` @ `1f7d39f` (matched windows); `p1_5_cut_001.json` @ `354fc13` | EXCLUSIVE rows: [P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) (`p1_exclusive_001.json` @ `ff4133f`, merged 8e503e1) (batching plateaus at N≈4 for A). L4 RUN_TO_RUN BLOCKED | workload A `reduced_fidelity` |
| 8 | GPU-native action / obs / reward / done | **PASS**. Batch-first Torch tensors on the device; Warp runs on Torch's stream. One int32 readback per step in strict mode, none in training mode. No per-env host loop. T2 sync audit for Lab B1/B2 | N/A (single env; Gymnasium adapter retained) | `p1_2_milestones_001.json` @ `f0a544b`; `p1_1b_s1_001.json` @ `660691c` | — | — |
| 9 | Seeded randomisation | **PARTIAL**. Per-env RNG seeds (`reset(seeds=)`); SplitMix64 policies. Heterogeneous scene seeds in one batch (`tasks=[512, 15]`, both backends). Since this branch, per-slot seeds also work for configured scenes in `tasks=`. A geometry change on partial reset raises `UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET` | **PASS** (seeded scenes, `task_randomization_v1`) | `p1_2b_001.json` @ `c578348`; this branch `53b915e` | No per-episode geometry or material randomisation inside a live batch (that needs a new env). CUT `task_randomization_v1` unsupported. Domain randomisation is user decision U-P15-3 | PARTIAL (labelled) |
| 10 | Performance profiling | **PASS** (all `SHARED_GPU`): benchmark rows with the plan's columns; telemetry with co-tenant sampling; `StageProfiler` kernel split; nsys graph-node traces; CUT rows | **PASS** (WP-16/17 timings; Native rows from WP-17 evidence) | `p1_0_baseline_001.json` @ `9f46fc9`; `p1_3a_001.json` @ `e144362`; `p1_5_cut_001.json` @ `354fc13` | EXCLUSIVE rows in [P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) (`p1_exclusive_001.json` @ `ff4133f`, merged 8e503e1) | `SHARED_GPU` on every batched row |
| 11 | Rubber material (placeholder) | **PARTIAL** (extension point only): the `batched/constitutive.py` `ConstitutiveModel` registry (per-env params, per-particle state); one model; mixed models `UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS` | **NOT_RUN** (Phase 2) | No evidence file on this branch; [ADR 0021](../../adr/0021-batched-rl-gpu-headless-runtime.md) §11. Phase 2 is in progress on `claude/rl-rubber` (decision D-12, its own acceptance and PR) | Rubber model not in Phase 1 | PARTIAL (placeholder; Phase 2) |
| 12 | Frozen independent evaluators | **PASS**. `goals.evaluate_goal` via `ConfiguredTaskSpec.evaluate_goal` on host snapshots, with source SHA256 in receipts. TRIM verdicts and metrics equal Native on replay. The sever evaluator decides batched CUT episodes. Since this branch, a non-finite env gets a structured `FAIL_INTEGRITY` entry with decoded reasons and no metrics; the evaluator is not called for it, and the other envs are evaluated normally (the P1-4 finding is fixed) | **PASS** | `p1_0_baseline_001.json` @ `9f46fc9`; `p1_2_milestones_001.json` @ `f0a544b`; `p1_5_cut_001.json` @ `354fc13`; this branch `53b915e` (`tests/test_vector_env_consolidation.py`) | v1 `evaluate_oracle` / the precision benchmark stay single-env (`NOT_APPLICABLE` to configured tasks) | v1 benchmark single-env (labelled) |
| 13 | Recording / replay | **PASS for RL data and replay**. Measured-trajectory replay drive; WP-17 replay parity (codes = Native 4/4 per seed; positions in band v2); WP-12 CUT replay adapter; `rcs_rl_rollout_v1` transitions with a consumer check | **PASS** (streamed exact-precision recording, media) | `p1_0_baseline_001.json` @ `9f46fc9`; `review_fix1_001.json` @ `1f7d39f`; `p1_5_cut_001.json` @ `354fc13` | Physics recorders (`cut_events_v1`, surfaces, raw 2 ms traces): `UNSUPPORTED_RECORDER` | physics recorders UNSUPPORTED (listed) |
| 14 | Termination reasons | **PASS**. `termination_reason_v1` on both backends: 1 rigid, 2 material, 4 grasp lost, 8 IK streak, 16 truncated, 32 input guard, 64 infrastructure; `VectorEnvInfrastructureError`; `classify_process_exit`; P1-4 LR-D 5/5 classified as declared. Since this branch (`material_invalid_termination_v1`), every material bit maps to 2, and `FOREIGN_DEVICE_HASHGRID` also maps to 64. Both backends and the supervisor use one function. Receipts and `decode_termination_detail` list every bit. A test iterates `material_batch.INVALID_REASONS` | **PASS** (GRASP_LOST / IK, HOME outcome) | `p1_2b_001.json` @ `c578348`; `p1_4_001.json` @ `7c5d8ca`; this branch `53b915e` (`tests/test_rl_termination.py`) | — | — |
| 15 | RL pipeline smoke | **PASS** (no training). S-CPU; S-CUT N=4 × 1,400 steps; S-TRIM N=8 × 1,500 steps. The consumer check passes, including Torch GAE = NumPy. The first S-TRIM run FAILED on a runner bug; it is preserved, fixed in `9a51a86` and rerun PASS 4/4. The smokes ran with the evidence script uncommitted (`git_dirty`) | N/A | `p1_5_cut_001.json` @ `354fc13` (key `t5_12_smoke`); [P1_5_CUT.md](P1_5_CUT.md) §7 | Smoke numbers are not benchmark rows | workload A `reduced_fidelity` |
| 16 | Long rollout / memory / stability (P1-4) | **PARTIAL**. LR-A **PASS**: N=8, 8 full episodes per env (64), VRAM slope 0.0, canaries 16/16. LR-B reset stress N=16: **FAIL** on B0 only (2,304 > 2,253 MiB, from the harness's checksum temporaries), 668/668 resets bitwise. LR-C rebuild cycles **PASS**. LR-D fault probes 5/5 **PASS** | Native limitations retained (`reload_in_place` needs cameras; NOT POLICY_READY) | `p1_4_001.json` @ `7c5d8ca` (merged `369d5db`); [P1_4_LONG_ROLLOUT.md](P1_4_LONG_ROLLOUT.md) | LR-A2 (`alias_free_v1`) NOT_RUN. LR-E (Lab B1 long run) NOT_RUN. EXCLUSIVE NOT_RUN | workload A `reduced_fidelity`, `SHARED_GPU` |

**Tally.** Each split row is counted by its overall status: row 2 PARTIAL, row 3 PASS, row 4 PARTIAL.

| Status | Count | Rows |
| --- | --- | --- |
| PASS | 9 | 3, 5, 7, 8, 10, 12, 13, 14, 15 |
| PARTIAL | 6 | 1, 2, 4, 9, 11, 16 |
| UNSUPPORTED | 1 | 6 |

This is the P1-5 plan's target tally, plus the new row 16. No row is FAIL overall. The recorded FAILs inside rows
are the B2 default point profile, LR-B B0 and the first S-TRIM smoke; all are preserved.

## Phase 1 Done checklist (specification, seven checkboxes)

Ticked means the checkbox is met by merged evidence, with its limitation stated. Unticked means open.

- [x] **True GPU batched execution at N=1, 4 and at least 8–16, with partial reset; throughput, memory and step
  latency reported.** Met on `SHARED_GPU`:
  - workload A N=1/4/8/16 (`p1_2_milestones_001.json` @ `f0a544b`), matched-window throughput (`review_fix1_001.json`
    @ `1f7d39f`);
  - CUT N=1/4/8/16 (`p1_5_cut_001.json` @ `354fc13`);
  - Lab B1 N=1/4/16 (`p1_1b_s1_001.json` @ `660691c`);
  - reset stress at N=16 (`p1_4_001.json` @ `7c5d8ca`).

  Formal `EXCLUSIVE_GPU` rows: [P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) (`p1_exclusive_001.json` @ `ff4133f`, merged 8e503e1). Limitation: B1 N=16 is `SHARED_GPU` only (a 2–7 MiB foreign graphics context);
  batched cold rows NOT_RUN.
- [x] **Representative CUT / TRIM run in the RL headless physics mode with success/failure evaluation; gaps have
  explicit status.**
  - TRIM PASS: A `reduced_fidelity`, B1 reduced, B2 with opt-in `mjwarp_grip_v1`.
  - CUT PARTIAL as declared: v7_2 batched PASS; frozen v5 single_env_headless and Native only; batched v5
    `UNSUPPORTED_SOLVER`.
  - Rows 2–4.
- [x] **Numerical and cut-history regression of single-env reference vs batched, and regression of the old frozen
  profiles.**
  - TRIM: L0/L1 CPU bitwise; L3 GPU with band v2; G5 L3 (Lab).
  - CUT: L0 CPU bitwise (T5.2); batched final codes within the single band; frozen v5 Demo 1 replay = Native verdict
    4/4.
  - Frozen-source SHA tests run in the CPU suite.

  Limitations: L4 RUN_TO_RUN is BLOCKED (frozen v5 atomics); the batched replay of the Native v7_2 recording is
  NOT_RUN; GPU position bands are descriptive. Evidence: `p1_2_milestones_001.json`, `review_fix1_001.json`,
  `p1_5_cut_001.json`, `p1_1b_s2_s3_001.json`.
- [x] **GPU-native action / observation / reward / done, independent per-env seeds, reproducible scenes, API
  tests.**
  - Rows 8 and 9.
  - The contract checker (`rl.vector_contract`) passes on both backends (`p1_2b_001.json` @ `c578348`).
  - The test counts of this branch are in the consolidation section below.

  Limitation: no per-episode randomisation inside a live batch (row 9 PARTIAL).
- [x] **Long rollout, OOM / memory / reset stress tests, real GPU telemetry.** P1-4 (`p1_4_001.json` @ `7c5d8ca`):
  - LR-A PASS;
  - LR-C PASS;
  - LR-D 5/5 PASS (NaN env, bad pose, SIGSEGV, hang, device OOM);
  - telemetry with per-PID VRAM and co-tenants.

  Limitations: LR-B **FAIL** on its declared VRAM bound B0 (harness overhead, diagnosed, not rerun); LR-A2 and the
  Lab long run LR-E NOT_RUN; all `SHARED_GPU`.
- [x] **Headless, vision / non-vision and full Native workloads report speed separately; no fake performance
  comparison.** Three boundaries are never merged into one speedup (`p1_0_baseline_001.json` @ `9f46fc9`):
  - `single_env_native`, with cameras (WP-17 evidence rows, labelled as such);
  - `single_env_headless`;
  - `batched_headless`, state only.

  Batched vision is UNSUPPORTED, not estimated. The "≈1.9× at N=4" claim from a multi-grid process was withdrawn
  (P1-2 report). The EXCLUSIVE table: [P1_PERFORMANCE_TABLE.md](P1_PERFORMANCE_TABLE.md) (`p1_exclusive_001.json` @ `ff4133f`, merged 8e503e1).
- [ ] **External agentic optimizer: accepted results written back to RCS and the version synced; a PR alone is
  PENDING.** **PENDING.**
  - The R0 acceptance harness, judge and `acceptance_v1` are merged (`87cfa3b`; `p1_6_r0_002.json` @ `3c2642a`,
    `p1_6_r0_003.json` @ `488c034`), and the GPU smoke passed.
  - No candidate has been accepted yet.
  - The P1-6 sync / descriptor pin is still pending (status `MERGED_RCS_PENDING_OPTIMIZER_SYNC` at best after a
    merge).

The plan's "Phase 1 done" also requires these, all still open:

- an xhigh acceptance review that passes, or PASS_WITH_LIMITATIONS written into the PR and docs. The review of
  P1-1 / P1-2 / P1-3a was PASS_WITH_LIMITATIONS ×3; the Phase 1 review that covers P1-4, P1-5, B2 and this
  consolidation is next;
- the merge to `main` through a PR;
- the external descriptor pinned (SYNCED).

## Consolidation (this branch)

| Task | Change | Status |
| --- | --- | --- |
| B2 through the vector env | `grip_contact_profile` (incl. `mjwarp_grip_v1`) is validated before the Isaac Lab import, forwarded to `BatchedLabOptions`, and recorded in `receipt()["grip"]` and `fidelity()`. Codes: `UNSUPPORTED_GRIP_CONTACT_PROFILE` (unknown), `UNSUPPORTED_TOOL_MOUNT` (non-default without B2), `UNSUPPORTED_OPTION` (workload A). ADR 0022 implementation note | PASS (CPU + Isaac Lab Warp-CPU test) |
| `evaluate()` on a non-finite env | A structured `FAIL_INTEGRITY` entry (`nonfinite_env_integrity_failure_v1`): decoded material and termination reasons, no physics metrics, evaluator not called. Covers the current state and `episode="last_completed"`, on both backends. Other envs are evaluated normally. Strict `step()` is unchanged | PASS (Torch CPU tests) |
| Invalid bit → termination | `material_invalid_termination_v1`: one mapping for both backends and the supervisor. The supervisor's stale copy of the material bits, which lacked 128 and 256, is replaced. A strict-mode foreign-grid raise is classified as infrastructure without changing the raise | PASS (CPU tests iterate `INVALID_REASONS`) |
| `tasks=` for configured scenes | Configured entries resolve to a `ConfiguredTaskFamily`. TRIM + CUT mixes are refused by the compiled layout | PASS (CPU + Torch CPU tests) |

Tests of the consolidation commit `53b915e` (evidence `p1_consolidation_001.json`). The machine was shared, with a
load average of about 65 on 64 cores.

| Suite | Passed | Failed | Skipped | Other |
| --- | --- | --- | --- | --- |
| Full CPU suite, `CUDA_VISIBLE_DEVICES=''` | 2443 | 1 | 28 | 6 xfailed |
| Full CPU suite, `CUDA_VISIBLE_DEVICES=0` (under the GPU0 flock) | 2443 | 1 | 28 | 6 xfailed |
| Torch RL / vector suite, CPU device | 343 | 0 | 10 | — |
| Isaac Lab suite, Warp CPU | 56 | 0 | 0 | — |

- The one CPU failure, `test_wp12_phase5_native_success.py::test_reproduces_phase2b_records`, comes from local
  WP-12 artifacts: `workcell_live_006` holds analysis class CUTTING_FAILURE, while the committed evidence says
  TASK_SUCCESS. It fails identically at the base `181ed37` and the test imports no RL module. It is OPEN and
  unrelated to this branch.
- The Torch skips are NOT_RUN: the workload-B modules need Isaac Lab or OpenUSD, and the CUDA stream test needs a
  GPU.
