# P1-5 plan — whole-RCS feature parity, the CUT decision, RL pipeline smoke, and the P1-4 long-rollout design

Date: 2026-11-10. Branch `claude/rl-ready-rubber-physics`. Analysis started at `f0a544b`; P1-3a was merged during
this round (`3682786`, progress `7bfa0b6`). Status: **PLAN**. This round was analysis and decisions only: no GPU job, no
source edit. Two read-only CPU probes were run (§9). Requirements: [RL_READY_PLAN.md](../../RL_READY_PLAN.md)
§2–§4 (P1-4, P1-5, "Phase 1 done") and the specification's P1-5 table and "Phase 1 Done" checklist
([RL_READY_SPECIFICATION.md](../../RL_READY_SPECIFICATION.md)). Status words: PASS / FAIL / BLOCKED / NOT_RUN,
plus PARTIAL and UNSUPPORTED (explicit refusal) in the matrix. A skipped item is not a pass.

## 0. Decisions at a glance

1. **CUT in the RL path = option (b) + option (c); option (a) is costed and deferred to a user decision.**
   - (b) A new versioned, opt-in scene `bar_flat_board_tlmpm_v7_2_v1` for the batched path. Everything is copied
     unchanged from the frozen WP-12 CUT scene `configs/wp12/scenes/bar_flat_board.json`: the CUT-01-R bar
     (160 × 30 × 20 mm), the `cutting_tasks_v1` material, `ruler_knife_v2_y`, the flat board, max_cuts 4, the
     frozen `sever_mid` goal and its thresholds, and the WP-12 workcell planner overrides. Only
     `solver.module` changes, `tlmpm_v5` → `tlmpm_v7_2`.
   - (c) The frozen v5 CUT keeps two homes:
     - `single_env_headless` reference rows, a replay of the recorded Native WP-12 Demo 1 CUT (P1-0 style);
     - the Native runtime.
   - In the batched path, v5 stays `UNSUPPORTED_SOLVER`.
   - The CUT row therefore ends **PARTIAL**, never PASS, unless the user also orders (a).
2. **Feature-parity matrix today (15 rows, RL_GPU_HEADLESS column):**
   - PASS 4: TRIM (labelled `reduced_fidelity`), vectorized reset/step (`SHARED_GPU`), GPU-native tensors,
     profiling.
   - PARTIAL 8: schema, robot (B1 PASS / B2 FAIL), cut history, seeded randomisation, rubber placeholder, frozen
     evaluators, recording/replay, termination reasons.
   - UNSUPPORTED 1: cameras.
   - NOT_RUN 2: CUT, RL pipeline smoke.
   - Details in §3.
3. **RL smoke (§5).** A random plus scripted policy runs over N=8 TRIM and N=4 CUT envs, with `autoreset="same_step"`
   and forced partial resets. Transitions are written as `rcs_rl_rollout_v1` NumPy chunks
   (obs / next_obs / action / reward / terminated / truncated / episode ids). A standard-loop consumer check
   (episode segmentation, returns, GAE, minibatching) reads them. No training.
4. **P1-4 (§6).** One supervised process per run, holding **one material = one device HashGrid**. The main run is
   N=8 × ≥ 8 serial full TRIM episodes, with replay canaries. A reset-stress run, a rebuild-cycle run and
   fault-injection runs follow. Memory bounds and classification rules are declared before any run.
   Infrastructure failures are classified apart from task outcomes, and runs resume into new directories.
5. **Task list (§7).**
   - P1-5: 13 tasks.
     - Owned by this plan: T5.1–T5.9, T5.12, T5.13.
     - Owned by the running **P1-2b** agent (worktree `RoboticCuttingSim-rl-p12b`), consumed and verified here:
       Lab dispatch T5.10 and `termination_reason` T5.11.
   - P1-4: 7 tasks. Pose-bounds invalidation (T4.2) is also P1-2b's.
   - Deferred: option (a).

## 1. Facts this plan rests on (verified in code, records or probes)

### 1.1 What "representative CUT" is in RCS today

| CUT configuration | Solver | Evaluator | Where it ran | Batched today |
| --- | --- | --- | --- | --- |
| CUT-01 v1 (`configs/cutting_tasks_v1.json`, legacy wedge path) | legacy `tlmpm` (v1), ternary one-cut codes | frozen v1 precision benchmark `tasks.evaluate_task_oracle` (`tasks.py:675`; requires codes in {0,1,2}, `:701`) | Lab/Kit WP-07 era | UNSUPPORTED_SOLVER (legacy material without compiled task, `batched/layout.py:130-133`) |
| WP-12 workcell CUT / Phase 7 **Demo 1**: `bar_flat_board` + `configs/wp11/goals/sever_mid.json` + planner overrides `path_v2`, 1 mm, 1° (`configs/wp12/workcell_live_v1.json`) | `tlmpm_v5` law 8, K=4, S=8 | `goals.evaluate_goal` (sever) | Native Isaac Sim 6.2, physical grasp: Demo 1 PASS (96,000 particles, 144 g → 72 g + 72 g) | UNSUPPORTED_SOLVER (probe, §9) |
| WP-14 `standard_cut` (`configs/wp14/scenes/standard_cut.json`, research profile `v7_cut_research_v1`) | `tlmpm_v7` | same | WP-14 Native C00/H01 v5 and v7: 4/4 TASK_SUCCESS | UNSUPPORTED_SOLVER |
| WP-14 follow-up control "v7_2, frozen C00 Straight CUT" (same scene/goal/controller, solver overridden) | `tlmpm_v7_2` (pre-WP-16 source, i.e. no contact/frontier optimizations) | same, unchanged | ideal tracking (kinematic), 700 commands: true two-part sever PASS; x/F P95 0.9992×, roughness 1.0000–1.0020× of v7 ([evidence](../../evidence/wp14_followup/cut_regression_001.json), [report](../WP14_FOLLOWUP_RESULTS.md)) | would pack (probe) |

v7-family CUT has the strongest CUT evidence in RCS. WP-14 reports that, against v5, original v7 improves
near-cut x/F P95 by 5.60–8.49× and locus roughness by 78–86 %. That is why `v7_cut_research_v1` exists. v7_2 keeps
these values within 0.2 % on C00 (`docs/PROGRESS.md` WP-14 sections). v5 with law 8 remains the untagged default.
All v7 variants are `EXPERIMENTAL_NOT_ACCEPTED`, and every material is `SIMULATOR_ONLY_UNCALIBRATED`.

### 1.2 Particle counts and episode size (CPU probe 1, §9)

- `bar_flat_board` + `configs/wp14/goals/standard_cut.json` (the frozen `sever_mid` goal at 14 s) with the
  workcell overrides compiles for both solvers:
  - `tlmpm_v5` → task id `CFG-bar_flat_board-sever_mid-c32f94a4`, exactly the Demo 1 task id;
  - `tlmpm_v7_2` → `CFG-bar_flat_board-sever_mid-46180291`.
- Both compiles: **96,000 particles, 0.144 kg**, 1 mm / 4 mm, **700 control steps (14 s)**, plan DONE at 11.48 s,
  digit 0 only.
- Action limits: 1 mm and 1° per step, the same as seeded TRIM.
- Native Demo 1 advanced the material for 740 steps (700 + 40 return-home steps).

### 1.3 The batched material accepts the v7_2 CUT scene and refuses v5 (CPU probe 2, §9)

- `layout.pack` of the v7_2 CUT task on the Warp CPU device succeeds:
  - 96,000 particles, 5,850 grid nodes;
  - S=8, K=4, 1 obstacle box (handle), 120 edge segments, flat-board fixture;
  - 9.9 s to pack;
  - frozen single-solver memory 45.3 MB, against 48.8–49.3 MB for TRIM.
- The v5 task raises `UnsupportedBatchConfiguration: UNSUPPORTED_SOLVER: solver 'tlmpm_v5'; only tlmpm_v7_2`.
- No batched kernel needs to change for a v7_2 CUT.

### 1.4 The CUT evaluator is independent of the solver and supports v7_2 state

- `goals.py` imports only NumPy and `cut_topology`; it imports no solver.
- `evaluate_goal` (`goals.py:1394`) scores a recording from:
  - positions;
  - packed codes;
  - reference stencils (nodes, weights, gradients);
  - masses and the compiled task.
- `_sever_outcome` (`goals.py:611-631`) uses mechanical components and the frozen midpoint location statistic.
- v7_2 uses the same packed code encoding as v5: `tlmpm_v7_2.py:19` → `tlmpm_v7.py:22` →
  `tlmpm_v5.py:44 CODE_ENCODING = "packed2"`.
- The batched `evaluate()` already passes `code_encoding="packed2"` and `max_cuts` from the task
  (`rl/vector_env.py:636-676`).
- The same unchanged evaluator scored the v7_2 C00 CUT in WP-14.
- The v1 `evaluate_oracle` is `NOT_APPLICABLE` to configured tasks (`scene_config.py:784`). It stays the
  single-env v1 benchmark.

### 1.5 CUT command input without vision

- **Scripted, deterministic.**
  - `ConfiguredTaskSpec.reference_action(t)` / `reference_edge_pose_world(t)` / `active_cut_index(t)`
    (`scene_config.py:728-749`) form a dense per-control-step schedule from `goal_planner_v1`. It uses only the
    scene, goal, tool and options: no particle, label or oracle input.
  - WP-14 verified that "all 700 actual commands independently replay from the compiled planner".
  - The existing `recorded_command_actions` (`rl/policies_batched.py:71`) turns any target sequence into
    `edge_delta_v1` actions open loop.
- **Recorded Native v5 CUT with measured poses**
  (`/home/ningan/RoboticCuttingSim/artifacts/wp12/demos_001/DEMO1_A/run/`, git `a960dc6` dirty, GPU0, 911 s wall):
  - `episode_000/actions.jsonl.gz`: 740 rows with `privileged_diagnostics.measured_tool_pose`, `active_cut`,
    `cut_index` and `resolved_target_world_xyz_xyzw`. The file was gzipped in place after analysis. Read it
    streaming and never rewrite the source.
  - `robot_state.npz`: `body_pose [741, 23, 7]`.
  - `physics_state.npz`: 75 frames of positions, velocities and codes plus reference stencils (160 MB).
  - `runtime/finger_geometry_00.json`, `material.json` (3 obstacle boxes: handle + two pads), `setup.json`,
    `task.json` (`max_cuts` 4, `tlmpm_v5`, 14 s).
  - Native verdict: PASS (integrity PASS over 40 frames; outcome: 2 components, 0.5 / 0.5 mass).
  - Source equivalence:
    - `tlmpm_v5.py` at `a960dc6` has SHA256 `a7a05b2f…`, identical to today.
    - `git diff a960dc6 HEAD -- live_material.py` touches only other solvers, receipts and a scratch release; the
      v5 code path is unchanged in substance. The implementer re-audits this.
    - The Native runner at `a960dc6` already called `update_pad_obstacles(_pad_boxes(actual tool pose))` every
      step (`isaac_sim_runner.py:1502` at that commit), the same semantics P1-0 replayed for TRIM.
  - DEMO1_B is the same physics with the `cut_aware_surface_v1` render profile.
- The WP-13/WP-14 Native C00 `physics_state.npz` files were deleted in the 2026-11-02/04 cleanups
  (`artifacts/wp12/DELETED_NPZ_*.txt` lists). Whether their action logs remain is UNVERIFIED and not needed.

### 1.6 Source drift that matters

`tlmpm_v7_2.py` changed after the WP-14 control (`0e92f1d`, `f55f94a`: the WP-16/17 `optimize_contacts` /
`optimize_frontier` defaults). The batched path requires both optimizations (`layout.py:134-136,149-151`). The
existing v7_2 CUT PASS therefore covers the **unoptimized** v7_2 only. A fresh single-env reference on the current
source is a prerequisite (T5.6). The WP-17 seed-512 open discrepancy (4,266 particles between WP-16 and WP-17
Native) shows that the optimized path is not proven final-code-equivalent. No equivalence is claimed here.

### 1.7 Batched runtime facts used below

- Measured on `SHARED_GPU` ([P1_1_P1_2_MILESTONES.md](P1_1_P1_2_MILESTONES.md),
  [evidence](../../evidence/rl_ready/p1_2_milestones_001.json)):
  - N=1 batched TRIM: 0.29–0.31 s/step at a 45–49 % co-tenant, 0.20 s at 23 %.
  - N=8: 1.6 s/step. N=16: 3.3 s/step.
  - About 1.4× aggregate gain at N=8–16 over N=1 at matched co-tenant load.
  - Own VRAM: 834–866 MiB (N=1), 1,378 MiB (N=8), 2,064 MiB (N=16).
  - Partial reset p95: 6.9–7.5 ms.
- `alias_free_v1` (P1-3a, opt-in) runs 1.40–1.45× faster end to end.
- Lab B1 TRIM: 1.3 s/step at N=4, own VRAM 974–1,108 MiB
  ([P1_1B_ROBOT_BATCHING.md](P1_1B_ROBOT_BATCHING.md)).
- Native Kit with cameras: 5.5–5.7 GiB.
- CUT costs about 0.93× TRIM per env (96,000 vs 103,000 particles), with 700 instead of 1,200 steps per episode.

## 2. CUT decision

### 2.1 Options and cost

| Option | What has to be built | New physics code | Implementation (rcs-impl) | GPU | What it proves / risks |
| --- | --- | --- | --- | --- | --- |
| **(a) batched `tlmpm_v5`** | (1) env-indexed copies of 4 v5 kernels: `_rigid_contacts` (`tlmpm_v5.py:184-297`, 114 lines), `fixture_contacts.fixture_contacts`, `_assign_band` (`:616-643`), `_assign_lead` (`:644-667`); (2) a `BatchedTLMPMv5.step` in the v5 launch order (`:816-890`: FLIP `_g2p` position update, no frontier or link completion); (3) layout/pack/receipt generalization (`validate_bed`, uniform signature without `optimize_*`, a per-solver array set); (4) solver dispatch in `material_batch`; (5) CPU L0 bitwise and L1 tests vs the frozen v5 single solver; (6) an ADR 0021 amendment | about 450–550 kernel and class lines, plus 400–600 test lines | 2.5–3.5 agent-days, 0.5 day xhigh review | about 2 h on GPU1: L2/L3 with the Demo 1 replay at N=1/4/8 | Reproduces the frozen Native CUT physics in the batch: the best fidelity to the Demo 1 record. Costs: a second batched solver kept in parity forever, so every Phase-2 constitutive or contact change is ported twice. Unlocks nothing else (TRIM is v7_2). The frozen v5 `_particle_contacts` also blocks RUN_TO_RUN (P1-2 L4). |
| **(b) versioned v7_2 CUT scene** | (1) a scene file differing only in `solver.module` and description; (2) a pre-declared workload; (3) `ConfiguredTaskFamily` in the vector env; (4) the `sever_bond_progress_packed_v1` shaping reward; (5) the `reference_command_policy_v1` driver; (6) multi-frame `evaluate()`; (7) CPU L0 tests on the CUT geometry | **0** | 1.5–2.5 agent-days | about 3 h on GPU1, plus an optional 0.5 h Native run (Kit) for a reference | The CUT-01 family runs truly batched with the unchanged evaluator, on the solver every other RL row already validates (L0 bitwise, L2/L3, P1-3a, P1-1b coupling). It is a **new physics configuration** of the CUT-01-R task: an opt-in scene, not the Demo 1 physics. Risk: optimized v7_2 has never cut this bar (§1.6). |
| **(c) single-env only, batched PARTIAL** | (1) a replay adapter for the WP-12 recording format; (2) single_env_headless rows: the v5 Demo 1 replay plus a v7_2 ideal-tracking reference | 0 | 0.75 day | about 35 min on GPU1 | The frozen v5 CUT runs headless with the independent evaluator and parity against Native samples (like P1-0 TRIM). Alone it leaves the batched CUT row at NOT_RUN. |
| (d) cross-solver replay (diagnostic) | the Native v5 Demo 1 measured trajectory driven through batched v7_2 | 0 | 0.25 day after (c) | 15 min | Shows whether v7_2 severs under v5-Native's actual open-loop robot motion. Descriptive only (different solver than the recording, no reaction feedback); never an acceptance row. |

### 2.2 Recommendation: (b) + (c), with (d) as an optional diagnostic; defer (a)

Reasons:

1. **The requirement is met without new physics.**
   - The specification asks that representative CUT actually runs in the RL headless physics mode, with success
     or failure decided by the independent evaluator, and that gaps stay PARTIAL.
   - (b) does that with zero new kernels:
     - the batched v7_2 kernels are already L0-bitwise against the frozen single solver
       ([P1_1_CORE_REVIEW.md](P1_1_CORE_REVIEW.md));
     - the pack accepts the CUT scene (§1.3);
     - the evaluator is solver-independent and has already scored a v7_2 CUT (§1.4).
2. **Frozen records and physics stay frozen.**
   - Under AGENTS.md and plan §1, new physics must be versioned and opt-in. (b) is a new file.
   - `bar_flat_board.json`, `sever_mid`, CUT-01 v1, the WP-12/13/14 records, law 8 and v5 are untouched.
   - The v5 CUT stays reachable unchanged through (c) and Native.
3. **Cost and maintenance.**
   - (b) + (c) costs about 2.5–3 agent-days once.
   - (a) costs about 3–4 days plus permanent double maintenance through Phase 2.
   - (a) only adds batched execution of a configuration that RCS's own WP-13/14 CUT studies found less consistent
     (x/F) than the v7 family.
4. **Honesty is cheap to keep.**
   - The batched CUT is labelled with its scene id and `solver: tlmpm_v7_2`, plus
     `differs_from_native_cut_reference: solver tlmpm_v5 → tlmpm_v7_2`.
   - The matrix CUT row stays **PARTIAL**: batched = versioned v7_2 CUT; frozen v5 CUT =
     single_env_headless + Native only; batched v5 = UNSUPPORTED_SOLVER.

What this does **not** claim:

- v7_2 CUT outcomes equal v5 CUT outcomes;
- any Native parity for the v7_2 CUT before T5.8;
- calibration;
- that the CUT-01 v1 precision benchmark (legacy solver, `evaluate_oracle`) runs batched. It never will; it is
  explicitly single-env.

**Escalation rule (pre-declared).**

- If the optimized-v7_2 single reference (T5.6 ii) or the batched M0 fails the frozen sever evaluator, that failure
  is the result and is preserved. There is no tuning, no threshold change and no new scene variant chosen after
  seeing outcomes.
- The CUT row then reads "batched CUT FAIL on v7_2", and (a) becomes the only route to a passing batched CUT.
- That route is a user decision: **U-P15-1** "Is batched execution of the frozen v5 CUT physics required for
  Phase 1?" The default without a decision is no: (a) is deferred and the row stays PARTIAL.

### 2.3 Pre-declared CUT workload (to be committed by T5.1 before any CUT run)

- `configs/rl_benchmark/scenes/bar_flat_board_tlmpm_v7_2_v1.json`:
  - byte-equal to `configs/wp12/scenes/bar_flat_board.json` except `solver.module = "tlmpm_v7_2"` and a
    description naming the source file and the one difference;
  - `scene_id` stays `bar_flat_board`, so particles, stencils and the plan are identical. A test asserts the equality
    of sampled particles, masses, stencils and reference schedule against the v5 compile.
- `configs/rl_benchmark/cut_v7_2_v1.json` (status `PREDECLARED`):
  - scene path and SHA256;
  - goal `configs/wp14/goals/standard_cut.json` and its SHA256 (frozen `sever_mid`, 14 s);
  - planner options `{max_rotation_deg 1, max_translation_m 0.001, controller_profile path_v2}`, the
    `workcell_live_v1` overrides;
  - sim seed 42, 700 steps;
  - driver `reference_command_policy_v1`;
  - success = `goals.evaluate_goal` verdict PASS and nothing else;
  - parity rules (below);
  - an explicit list of what is not claimed.
- `configs/rl_benchmark/workloads_v2.json`. `workloads_v1.json` stays unchanged. It defines:
  - C1 `cut_v5_native_replay`: frozen v5, Demo 1 measured trajectory, single_env_headless;
  - C2 `cut_v7_2_batched`: the new scene, ideal tracking, batched;
  - C3 `cut_v7_2_lab_b1`: optional.
- Parity rules for C2:
  - CPU L0 bitwise: batched N=1 and each env of N=3 copies against the frozen single v7_2 on identical commands.
  - GPU:
    - L2 invariants every step;
    - final verdict equal to the single reference;
    - labelled count and final-code differences reported against the band of ≥ 3 single runs;
    - positions reported, not scored (the GPU float-atomic band).
  - C1 (v5 replay vs Native) uses the P1-0 rules: final codes vs the 75 Native samples, evaluator verdict and
    metrics vs Native, and a position-difference band from ≥ 3 runs.

## 3. Feature-parity matrix (status today, 2026-11-10)

Columns: RL_GPU_HEADLESS and FULL_NATIVE_REFERENCE status today; evidence; gap; the plan to close the gap, or the
PARTIAL / UNSUPPORTED label it keeps. Task ids are in §7.

| # | Capability | RL_GPU_HEADLESS today | FULL_NATIVE_REFERENCE today | Evidence | Gap | Plan → target after P1-5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Scene / goal / material schema | **PARTIAL**: same `rcs_scene_v1`/`rcs_goal_v1` compiled by `compile_task` (the compiled `config_sha256` equals the WP-17 payload in every P1-2 run); vector env accepts only seeded-TRIM `TaskFamily` (`vector_env.py:98-141,224`); material accepts only `tlmpm_v7_2` law 8 with optimizations (`layout.py:124-160`); Lab env accepts any compiled v7_2 task with a uniform rigid scene | **PASS**: every configured scene (v4–v7_2), registered v1–v4 tasks | [P1_1_P1_2_MILESTONES §2](P1_1_P1_2_MILESTONES.md); probes §1.2–§1.3 | no configured-scene family; v5/v7/v7_1 scenes and legacy tasks refused | T5.2 `ConfiguredTaskFamily` (any compiled v7_2 scene + goal) → **PARTIAL** (labelled: other solvers `UNSUPPORTED_SOLVER`) |
| 2 | CUT (representative) | **NOT_RUN**: no v7_2 CUT scene declared; declared CUT scenes are v5/v7, refused `UNSUPPORTED_SOLVER` (probe) | **PASS**: WP-12 Demo 1 Native v5 sever PASS; WP-13/14 Native C00 v5/v7 TASK_SUCCESS; v7_2 C00 ideal tracking PASS (pre-optimization source) | [P1_1B §S2/S3 status](P1_1B_ROBOT_BATCHING.md); `artifacts/wp12/demos_001/DEMO1_A` (main checkout); [cut_regression_001](../../evidence/wp14_followup/cut_regression_001.json) | no batched CUT; no headless CUT row on current source; no Native v7_2 CUT reference | §2: T5.1–T5.9 → **PARTIAL** (batched = versioned v7_2 CUT, evaluator-decided; frozen v5 CUT single_env_headless + Native; batched v5 UNSUPPORTED) |
| 3 | TRIM (representative) | **PASS** (`reduced_fidelity` labelled): workload A M0/M1/M2, replay episodes 40/40 codes = Native and evaluator PASS, recorded-command episodes PASS; workload B B1 N=2/4 both seeds evaluator PASS | **PASS**: WP-17 seeds 512/15 | [P1_1_P1_2_MILESTONES](P1_1_P1_2_MILESTONES.md), [P1_1B §3](P1_1B_ROBOT_BATCHING.md), [p1_2_milestones_001](../../evidence/rl_ready/p1_2_milestones_001.json) | full-fidelity physical-grasp batched TRIM fails (row 4); no EXCLUSIVE_GPU rows | P1-4 long rollout; multi-frame evaluation (T5.2) → **PASS** (`reduced_fidelity`) |
| 4 | Robot articulation / contact | **PARTIAL**: B1 rigid mount (`reduced_fidelity: rigid_tool_mount_no_grasp`) N=1/4/16, bitwise isolation in deterministic mode, delayed-wrench coupling, G5 L3 material parity PASS; **B2 physical grasp FAIL** (tool slips 8–13 mm, up to 0.5 rad, in MJWarp fingers during the HOME→skim reorientation → GRASP_LOST); G4 FAIL vs pre-set tolerance; B2 deterministic BLOCKED; **SIGSEGV** in the material graph after grasp loss; `open_vector_env(backend="isaac_lab_newton")` not wired (`vector_env.py:203`); second Franka parked | **PASS**: PhysX physical grasp, WP-17 grasp stabilization PASS | [P1_1B](P1_1B_ROBOT_BATCHING.md), [p1_1b_s2_s3_001](../../evidence/rl_ready/p1_1b_s2_s3_001.json), [ADR 0022](../../adr/0022-batched-robot-workload-b.md) | B2; vector-env dispatch; crash instead of invalid flag | P1-2b: dispatch (T5.10) and per-env pose-bounds invalidation (T4.2); B2 root cause in worktree `RoboticCuttingSim-rl-b2`; T5.9 B1 CUT (stretch) → **PARTIAL** (B1 only) until B2 passes |
| 5 | Cut history / persistent severing | **PARTIAL**: per-env packed codes, frontier/link and pending state batched; L2 code monotonicity every step (0 violations); replay final codes = Native; partial reset restores history (CPU bitwise); `evaluate()` integrity sees only 2 frames; `cut_events_v1` and cut-aware surfaces `UNSUPPORTED_RECORDER` | **PASS**: oracle history at stride; `cut_events_v1` opt-in (v5 only) | [P1_1_P1_2_MILESTONES §5, §8](P1_1_P1_2_MILESTONES.md); [P1_1_CORE_REVIEW](P1_1_CORE_REVIEW.md) | multi-frame integrity; recorders | P1-2b full evaluation stream, used every 20 steps in every P1-5 / P1-4 episode (T5.2) → **PASS** for the solver contract; recorders stay UNSUPPORTED (listed) |
| 6 | Cameras RGB-D | **UNSUPPORTED**: state only, `UNSUPPORTED_SENSOR` (`vector_env.py:208,220`; ADR 0022 §10); the kit-less Newton path has no RTX renderer | **PASS**: v5 rig (global D455 + two wrist D405 RGB-D, timestamps); global D455 native calibration NOT_VALIDATED (`docs/PROGRESS.md`) | `docs/PROGRESS.md` camera sections | no batched vision profile | Out of Phase 1 unless requested (**U-P15-2**); keep the refusal → **UNSUPPORTED** (labelled; vision data from the Native reference) |
| 7 | Vectorized reset / step | **PASS** (`SHARED_GPU`): one process; one launch per kernel for all envs (nsys 15,434 vs 15,436 kernels at N=1 vs N=4); masked partial reset (non-reset envs bitwise on CPU), p95 6.9–7.5 ms; N=1/4/8/16 measured; about 1.4× aggregate at N=8–16 over N=1 at matched co-tenant load; Lab B1 robot-only 317.8 env-steps/s at N=16 | N/A (N=1 by design; `reload_in_place` reset) | [P1_1_P1_2_MILESTONES §6](P1_1_P1_2_MILESTONES.md) | EXCLUSIVE_GPU rows; long-run stability | P1-4 (§6); EXCLUSIVE window when GPU1 is free → **PASS** |
| 8 | GPU-native action / obs / reward / done | **PASS**: batch-first Torch tensors on the device; Warp on Torch's stream; one int32 readback per step (strict) or 0 (training); T2 sync audit (Lab); no per-env host loop | N/A (single env; Gymnasium adapter retained) | [P1_1_P1_2_MILESTONES §1](P1_1_P1_2_MILESTONES.md), [P1_1B T2](P1_1B_ROBOT_BATCHING.md) | — | keep → **PASS** |
| 9 | Seeded randomisation | **PARTIAL**: per-env RNG seeds (`reset(seeds=)`); counter-based SplitMix64 policies; per-slot seeded TRIM geometry (heterogeneous seeds in one batch); a geometry change on partial reset raises `UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET` (`vector_env.py:456`); per-env material params exist in the batch table (`constitutive.py` `PARAM_NAMES`) but only come from each scene's profile; CUT `task_randomization_v1` unsupported | **PASS**: seeded scenes, `task_randomization_v1` | `vector_env.py:98-133`, `policies_batched.py:141` | per-episode geometry / material randomisation inside a live batch | T5.2 per-slot seeds for configured scenes; per-episode randomisation needs `rebuild()` / a new env → **PARTIAL** (labelled); domain randomisation is a user decision (**U-P15-3**) |
| 10 | Performance profiling | **PASS**: benchmark rows with the plan's columns; telemetry with `SHARED_GPU` / `EXCLUSIVE_GPU`; `StageProfiler` kernel split; nsys graph-node traces; all rows `SHARED_GPU` | **PASS**: WP-16/17 timings; Native rows from WP-17 evidence | [P1_0_BASELINE](P1_0_BASELINE.md), [P1_3A](P1_3A_ALIAS_FREE_CONTACTS.md) | EXCLUSIVE rows; CUT rows | T5.6/T5.7 CUT rows → **PASS** |
| 11 | Rubber material (placeholder) | **PARTIAL** (extension point only): `batched/constitutive.py` `ConstitutiveModel` registry (per-env params, per-particle state, reset-equals-fresh test); one model `log_strain_von_mises_plastic_v5`; mixed models `UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS` | **NOT_RUN** (Phase 2) | [ADR 0021 §11](../../adr/0021-batched-rl-gpu-headless-runtime.md) | rubber model | Phase 2 (P2-0..P2-6) → **PARTIAL** in Phase 1 |
| 12 | Frozen independent evaluators | **PARTIAL**: `goals.evaluate_goal` via `ConfiguredTaskSpec.evaluate_goal` on host snapshots, source SHA256 in receipts; TRIM verdicts and metrics identical to Native on replay; the sever (CUT) evaluator never ran on batched state; v1 `evaluate_oracle` is `NOT_APPLICABLE` to configured tasks and its legacy solver path is refused by design | **PASS** | §1.4; [P1_0_BASELINE](P1_0_BASELINE.md) | CUT verdicts in the batch | T5.7 → **PASS** (v1 benchmark stays single-env, labelled) |
| 13 | Recording / replay | **PARTIAL**: measured-trajectory replay drive (`step_measured_edge`, `MeasuredTrajectoryDriver`) PASS; open/close receipts; host snapshots; no streamed episode recorder; no RL transition writer; physics recorders `UNSUPPORTED_RECORDER` | **PASS**: streamed exact-precision recording (actions, robot state, physics stream, MP4) | [P1_0_BASELINE](P1_0_BASELINE.md), `rl/replay.py` | RL transitions; WP-12 CUT recording format | T5.5 CUT replay adapter; T5.12 `rcs_rl_rollout_v1`; T4.1 ledger → **PASS** for RL data and replay; physics recorders stay UNSUPPORTED |
| 14 | Termination reasons | **PARTIAL**: vector env `terminated = invalid_reason != 0` (6 material bits, plus bit 64 `CONTACT_SPEED_GUARD` with `alias_free_v1`), `truncated` at the declared step count; Lab env `termination_reason` bits 1 rigid / 2 material / 4 grasp lost / 8 IK streak (`lab_batched_env.py:51-55`); no shared field; infrastructure faults crash the process instead of being classified | **PASS**: episode-state termination reasons, GRASP_LOST / IK, HOME outcome | [P1_1B §1](P1_1B_ROBOT_BATCHING.md) | one taxonomy; infra / task separation | P1-2b `termination_reason` (T5.11); T4.1 infra classification; T4.2 (P1-2b) → **PASS** |
| 15 | RL pipeline smoke | **NOT_RUN** | N/A | — | — | T5.12 (§5) → **PASS** |

**Today:**

- PASS 4 (rows 3, 7, 8, 10);
- PARTIAL 8 (rows 1, 4, 5, 9, 11, 12, 13, 14);
- UNSUPPORTED 1 (row 6);
- NOT_RUN 2 (rows 2, 15).

**Target if every P1-5 / P1-4 task passes:**

- PASS 9 (rows 3, 5, 7, 8, 10, 12, 13, 14, 15);
- PARTIAL 5 (rows 1, 2, 4, 9, 11);
- UNSUPPORTED 1 (row 6).

Rows that stay PARTIAL carry their label in every report. The target is not a promise: each cell flips only on
recorded evidence.

## 4. Phase 1 Done checklist map

| Specification checkbox | Status now | Closes with |
| --- | --- | --- |
| N=1, 4, ≥ 8–16 true GPU batched execution and partial reset; throughput, memory, step latency | PASS on SHARED_GPU (P1-2 M0–M2); EXCLUSIVE rows NOT_RUN | P1-4 telemetry; an EXCLUSIVE window if one becomes available |
| Representative CUT/TRIM run in RL headless physics mode with success/failure evaluation; gaps explicitly labelled | TRIM PASS (`reduced_fidelity`); CUT NOT_RUN | T5.1–T5.9 → CUT PARTIAL (labelled per §2) |
| Single↔batched numerical and cut-history regression; old frozen profile regression | TRIM PASS (L0 CPU bitwise, L3 GPU, G5 L3); CUT NOT_RUN; v5 frozen CUT regression NOT_RUN | T5.2 (CPU L0 CUT), T5.6 (v5 replay vs Native, v7_2 reference band), T5.7 |
| GPU-native action/obs/reward/done, independent seeds, reproducible scenes, API tests | PASS (seeded TRIM); configured scenes NOT_RUN | T5.2, T5.11 |
| Long rollout, OOM / memory / reset stress, real GPU telemetry | NOT_RUN (M2 has 1,575–3,950-step memory samples only) | P1-4 (§6) |
| Headless, vision/non-vision and full Native speeds reported separately | PASS for headless / Native; vision UNSUPPORTED in headless | keep labels; Native rows from evidence |
| External agentic optimizer results written back to RCS and synced | design done; PENDING (P1-6 R0) | P1-6 |

## 5. RL pipeline smoke design (T5.12)

**Goal.** Show that a standard RL loop can consume the batched env. The smoke drives a random or scripted policy
over N envs, with repeated partial resets, and saves obs, action, reward and done in a documented format. **No
training and no gradient step.**

**Runtime.**

- One process, one `VectorCuttingEnv`, one material, one device HashGrid.
- `.deps/isaacsim-venv`, `PYTHONPATH=src`, no Kit.
- Settings: `autoreset="same_step"`, `strict=False` (training mode: invalid envs terminate, are reset and are
  counted), `observation_profile="state"`.
- A second run with `privileged_state` writes `obs_critic` in a separate, clearly non-deployable array.

**Runs (each pre-declared in `configs/rl_benchmark/rl_smoke_v1.json`).**

| Run | Task | N | Policies per env | Partial resets | Batch steps | Estimate (shared GPU1) |
| --- | --- | --- | --- | --- | --- | --- |
| S-TRIM | seeded TRIM 512 / 15 (WP-17 configuration) | 8 | envs 0–3 `recorded_command_policy_v1` (s512, s15, s512, s15); envs 4–7 `RandomPolicyV1` (arm probability 0.2) | autoreset at truncation (1,200) or invalid; forced `reset(env_ids)` of random envs on a SplitMix64 schedule every 50–300 steps | 1,500 | about 40 min; own VRAM ≤ 1.5 GiB |
| S-CUT | `cut_v7_2_v1` (§2.3) | 4 | envs 0–1 `reference_command_policy_v1`; envs 2–3 `RandomPolicyV1` | as above; episode 700 steps | 1,400 | about 25 min; ≤ 1.2 GiB |
| S-CPU (pytest, Torch venv) | seeded TRIM at 4 mm particles (`reduced_fidelity`), `parity_mode` CPU | 3 | 1 recorded-command, 2 random | `max_episode_steps=10` (`reduced_fidelity`), forced resets | 40 | < 2 min, CPU |

**Forced-reset convention (makes the data loop-consistent).**

- Before `reset(env_ids)`, the writer marks the last stored transition of those envs `truncated=True` and stores
  their current observation as `next_obs` (a time-limit-style truncation).
- `episodes.jsonl` records the end as `TRUNCATED_BY_FORCED_RESET`.
- Autoreset ends use `info["final_obs"]` as `next_obs`.

**Format `rcs_rl_rollout_v1`** (`artifacts/rl_ready/p1_5/smoke_<task>_n<N>_NNN/`, never overwritten).

- `manifest.json` contains:
  - schema id and version;
  - the env `receipt()` (fidelity, source SHA256s, material receipt);
  - `single_observation_space` / `single_action_space` (Box shape, bounds, dtype);
  - N, T and chunk size;
  - policy ids and seeds;
  - drive per env, autoreset mode and forced-reset schedule seed;
  - git commit, dirty flag and versions;
  - a chunk list with SHA256 per chunk.
- `chunk_NNNNN.npz` holds T_c ≤ 256 steps: `obs_policy [T_c,N,D] f32`, `next_obs_policy [T_c,N,D] f32`,
  `actions [T_c,N,8] f32` (executed action), `rewards [T_c,N] f32`, `terminated [T_c,N] bool`,
  `truncated [T_c,N] bool`, `episode [T_c,N] i64`, `step_count [T_c,N] i64`, `seed [T_c,N] i64`,
  `drive [T_c,N] i8`, `action_status [T_c,N] i32`, `invalid_reason [T_c,N] i32`,
  `termination_reason [T_c,N] i32` (T5.11), `reset_kind [T_c,N] i8` (0 none, 1 autoreset, 2 forced).
  `obs_critic` appears only in privileged runs, and the manifest flags it non-deployable.
- `episodes.jsonl` holds one line per finished episode: env, episode, seed, length, undiscounted return, end class
  (TRUNCATED_TIME_LIMIT / TERMINATED_INVALID + reason bits / TRUNCATED_BY_FORCED_RESET) and, for complete episodes,
  the frozen evaluator verdict summary. Evaluation is on the host, outside timing, and never feeds the reward.
- Device tensors are copied to pinned host buffers once per chunk, not per step, so there is no per-step host sync
  beyond the env's own strict readback, which is off here.

**Consumer check** (`scripts/rl_ready/check_rl_rollout.py`; NumPy, plus a Torch path when available).

- Shapes and dtypes match the declared spaces. Every value is finite. Actions lie in [-1, 1].
- `obs_policy[t+1] == next_obs_policy[t]` bitwise wherever the env is not done at t.
- `done = terminated | truncated` matches the `episode` increments and the `step_count` resets per env.
- Episode returns recomputed from `rewards` equal `episodes.jsonl`.
- A CleanRL-style GAE (γ 0.99, λ 0.95, zero value function, bootstrap on truncation only) gives finite
  advantages. The Torch path repeats it on the device.
- A shuffled minibatch iterator over T × N runs one pass, emulating a PPO data loader without any update.
- No observation term outside `rl.observations.POLICY_TERMS` appears in `obs_policy`.
- The check prints PASS or FAIL per item. Exit code 0 alone is not a pass.

**What the smoke reports.**

- PASS / FAIL per consumer item.
- Episode counts by end class.
- Evaluator verdicts: scripted TRIM is expected to PASS and random envs to FAIL. Whatever happens is reported.
- Reset p50/p95, own VRAM and RSS.
- Throughput, labelled as a smoke and not a benchmark row.
- An Isaac Lab B1 variant (N=4, 1,200 steps, GPU0) is optional after T5.10.

## 6. P1-4 design: memory, stability, long rollout (updated)

### 6.1 Process rules (Warp 1.17 HashGrid hazard)

- **One material per process, so exactly one live device `wp.HashGrid`.**
  - Warp 1.17 replays a captured hash-grid build with the descriptor of the grid built last on the host
    (P1-3a probe: 8,804,086 → 0 candidates).
  - `BatchedCuttingMaterial` refuses a second live CUDA grid (`UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS`,
    `material_batch.py:399-418`).
  - `alias_free_v1` keeps one grid and rebuilds it on the host before each launch. Its `FallbackGrid` is not a
    `wp.HashGrid`.
- **Consequences:**
  - CUT and TRIM run in separate processes (their K also differs: 4 vs 2);
  - frozen single-env references and Native runs get their own processes;
  - no in-process A/B of contact modes;
  - the evaluator stays host-only (it builds no grid).
- **Every run records a HashGrid census:**
  - `live_device_hashgrids()` at start, after every full reset and at the end;
  - every `wp.HashGrid.__init__`, with device and creator, as in P1-1b;
  - the material receipt's `device_hashgrid_guard`.
- A run whose census ever exceeds 1 is `INVALID_MULTI_GRID_PROCESS`. It is preserved, never used as a row.
- Variants (`frozen_v1`, `alias_free_v1`) run one per process.

### 6.2 Runs (pre-declared in `configs/rl_benchmark/p1_4_long_rollout_v1.json` before the first run)

All runs:

- warm Warp cache;
- `strict=False`;
- L2 invariants every step (code monotonicity, finite state, slot overflow, det F > 0, grouped-hash foreign
  candidates 0);
- under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock`, with `CUDA_VISIBLE_DEVICES=<N>` and
  `TMPDIR=/mnt/data/ningan/tmp`;
- a free-VRAM check of the estimate + 4 GiB before start (D-07);
- `SHARED_GPU` / `EXCLUSIVE_GPU` from sampled co-tenant use.

| Run | What | N | Drivers | Length | Estimate (shared) | Own VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| **LR-A** (main) | TRIM s512 / s15, `frozen_v1` (default path) | 8 | envs 0–1 **replay canaries** (`step_measured_edge` of WP-17 s512 / s15); envs 2–5 `recorded_command_policy_v1`; envs 6–7 `RandomPolicyV1` | **≥ 8 serial full 1,200-step episodes per env** (64+ episodes); env k starts at offset 150·k, so resets fall while other envs are mid-episode; about 10,650 batch steps | 4.0–4.7 h | ≤ 1.5 GiB |
| LR-A2 | as LR-A with `alias_free_v1` (after the P1-3a merge lands) | 8 | same | 4 serial episodes per env | about 1.8 h | ≤ 1.6 GiB |
| LR-B | reset stress | 16 | 2 untouched canaries; 14 random envs, each force-reset with probability 0.05 per step (SplitMix64 schedule) | 1,000 batch steps (about 700 resets) | about 55 min | ≤ 2.2 GiB |
| LR-C | rebuild cycle: open → 50 steps → `close()` / `release()`, × 10 in one process (one live grid at a time) | 4 | recorded command | 500 steps | about 15 min | ≤ 1.1 GiB |
| LR-D | fault injection (§6.4) | 2 | recorded command | a few steps each | about 10 min | ≤ 1 GiB |
| LR-E (optional) | Lab B1 TRIM, `.deps/isaaclab-venv`, GPU0 | 4 | recorded command | 3 serial episodes per env via `reset_envs(mask)` | about 1.3 h | ≤ 1.2 GiB |
| LR-F (optional, after T5.2–T5.4) | CUT `cut_v7_2_v1` | 8 | 6 reference-command, 2 random | 8 serial 700-step episodes per env | about 2.5 h | ≤ 1.4 GiB |

**Canaries.**

- Replay through `step_measured_edge` reproduced the Native final codes in 40/40 earlier episodes (P1-0, M2).
- Every canary episode must therefore end with the Native labelled count (37,958 for s512, 85,786 for s15) and
  evaluator PASS. Differing code counts are reported.
- A canary miss marks possible state leakage across resets. It is a FAIL of the canary check, followed by
  investigation. It is never re-scored.
- In LR-B, every forced reset is followed by a device comparison of the reset env with its packed initial state:
  x = reference, v = 0, codes = 0, F = Fpi = I, plastic = 0, plus the tool, stroke, frontier, guard and pending
  arrays. A single reduction kernel counts mismatching words. Reset is a copy, so this check is bitwise even on
  the GPU.

### 6.3 Memory measurements and pre-declared bounds

**Samples.**

- Every 50 batch steps, record:
  - Warp mempool used, current and high (`wp.get_mempool_used_mem_current/_high`);
  - Torch `memory_allocated` / `memory_reserved`;
  - own-PID VRAM from `GpuTelemetry` (1 Hz `nvidia-smi`, the existing `rl/telemetry.py`);
  - `VmRSS` / `VmHWM`, open file descriptors, thread count and the CUDA graph-capture count;
  - Warp modules loaded and the HashGrid census.
- At every episode end, take the same sample after `gc.collect()`.
- `tracemalloc` stays off in measured runs. It is enabled only in a diagnostic rerun if B3 fails.

**Warm-up (fixed rule).** It ends when every env has completed its first episode and its first reset.

| Bound | Quantity | Pass rule (after warm-up) |
| --- | --- | --- |
| B1 | Warp mempool current, Torch allocated | max − min ≤ 1 MiB (buffers are reused; the expected value is 0) |
| B2 | own-PID VRAM (`nvidia-smi`) | OLS slope ≤ 2 MiB per 1,000 batch steps; no single increase > 64 MiB |
| B3 | RSS at post-gc episode-end samples | OLS slope ≤ 0.5 MiB per 1,000 env-steps, and last-quintile median − second-quintile median ≤ max(64 MiB, 3 %) |
| B4 | graph and grid state | graph captures = 1 per batch shape for the whole process (0 recaptures); HashGrid census = 1 throughout; Warp module count constant |
| B5 | reset latency | reported (p50, p95, max per kind); not a gate |

- Slopes are reported with 95 % confidence intervals.
- A bound is never changed after a run. A failed bound is reported as FAIL, with the tracemalloc diagnostic
  rerun attached.
- The M2 data (RSS ±45–55 MiB per 1,000 steps without monotone growth) is why B3 samples after `gc.collect()` at
  fixed points instead of at arbitrary times.

### 6.4 Infrastructure failures vs task outcomes

**Structure.** A thin supervisor, `scripts/rl_ready/supervise_rollout.py`, launches the rollout child,
`scripts/rl_ready/run_long_rollout.py`.

**Child ledger.** The child appends `ledger.jsonl`, flushed and `fsync`-ed per record, with these record types:

- `start`: config, commit, versions, GPU, census;
- `sample`;
- `reset`: ids, kind, wall;
- `episode`: env, index, seed, steps, end class, `invalid_reason`, verdict summary, `final_codes_sha256`,
  labelled count, drive;
- `heartbeat`;
- `infra_error`: written from a top-level handler with the exception type and message;
- `end`.

A crash therefore keeps everything up to the last record.

**Supervisor classification** (from the return code / signal, the last ledger records and the stderr tail):

| Class | Rule |
| --- | --- |
| COMPLETE | exit 0 and an `end` record |
| INFRA_OOM_DEVICE | `torch.cuda.OutOfMemoryError`, CUDA error 2, or a Warp allocation failure recorded by the child |
| INFRA_CUDA_ERROR | other CUDA / Warp runtime errors (illegal address, launch failure) |
| INFRA_CRASH_SIGNAL | return code −11 / −6 / −7 (SIGSEGV, SIGABRT, SIGBUS), e.g. the P1-1b B2 graph SIGSEGV |
| INFRA_KILLED | −9 / 137; the cause stays UNVERIFIED unless kernel OOM evidence is readable |
| INFRA_HANG | the watchdog saw no heartbeat for max(300 s, 20 × median step); SIGTERM, then SIGKILL; a `py-spy dump` first if installed |
| INFRA_PYTHON_EXCEPTION | any other exception (a code bug) |
| INFRA_DISK | `ENOSPC` / IO errors |

**Episode end classes** are never mixed with infrastructure classes:

- TASK_SUCCESS / TASK_FAILURE: the frozen evaluator verdict;
- ENV_INVALID: device validity bits, plus `CONTACT_SPEED_GUARD`. The evaluator still runs for diagnostics, but the
  episode is excluded from success denominators;
- TRUNCATED_BY_FORCED_RESET: not evaluated;
- INFRA_INCOMPLETE: in flight when the process died. It counts toward infrastructure tallies only.

**Recovery.**

- At most one automatic restart, into a new directory `_00{k+1}` with `resumed_from` and the infrastructure event.
- The episode schedule continues deterministically: seed = SplitMix64(seed, episode).
- Nothing is overwritten. Reports carry both runs.

**Fault injection** (child-only `RCS_FAULT_INJECT`, never in library code):

- CPU tests:
  - `exception@k`, `sigsegv@k`, `sigkill@k` and `hang@k` → each class;
  - `nan_env:e@k` → ENV_INVALID for env e only, while the others continue (non-strict isolation).
- One GPU probe (LR-D) repeats the NaN, SIGSEGV and hang cases and adds a **device OOM**:
  `torch.cuda.set_per_process_memory_fraction(0.02)`, then a large allocation. Only this process is affected.
  No co-tenant memory is touched.

### 6.5 Retained Native limitations (no new Native long rollout in P1-4)

- `stage_reset="reload_in_place"` requires live cameras. A camera-off in-place reset crashes PhysX (4 of 4 probes,
  [WP11_PHASE_H_PROFILING](../WP11_PHASE_H_PROFILING.md)).
- The Native `reload_in_place` memory probe (8 × 20 steps) measured host 24.2 MiB/episode and GPU 9.3 MiB/episode
  (`docs/POLICY_READY_API.md:488`).
- Native stays NOT POLICY_READY (GPU 52 MiB/episode vs threshold 32, unresolved; `docs/PROGRESS.md:387`).
- These remain the FULL_NATIVE_REFERENCE limitations in every P1-4 report. Batched results never replace them.

### 6.6 P1-4 acceptance mapping

| Plan item | Evidence |
| --- | --- |
| long vectorized rollout with repeated partial resets | LR-A (+ LR-A2, LR-B) |
| bounded VRAM / RSS (slope reported) | B1–B4 in LR-A, LR-A2, LR-B, LR-C |
| 8+ serial full episodes | LR-A: ≥ 8 per env × 8 envs |
| infrastructure failures reported apart from task failures | §6.4 classes, LR-D plus CPU fault tests |
| Native limitations retained | §6.5 |

## 7. Task list for the rcs-impl agent

Sizes: S ≤ 0.5 agent-day, M 0.5–1.5, L > 1.5. Venvs:

- CPU tests: `.venv` (`.venv/bin/python -m pytest`).
- Torch tests: D-09, `PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python -m pytest -p no:cacheprovider`.
- Headless GPU: `.deps/isaacsim-venv`, no Kit.
- Lab: `.deps/isaaclab-venv`.
- Native: `scripts/run_wp10_live.py --runtime isaac_sim`.

Every GPU job runs under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock`, with an explicit device
and the estimate + 4 GiB free before start (D-07). Only one RCS job runs per GPU. External-optimizer code stays on
GPU1 (U1); these tasks may use either lock.

Commit rules:

- Stage explicit paths only.
- Run `tests/test_repo_naming.py` before every commit.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

### P1-5

| Id | Task | Depends | Where / GPU | VRAM | Wall | Size |
| --- | --- | --- | --- | --- | --- | --- |
| T5.1 | Pre-declare the CUT workload (§2.3): scene file, `cut_v7_2_v1.json`, `workloads_v2.json`; freeze test (SHA pins; particle / stencil / schedule equality with the v5 compile) | — | CPU | — | — | S |
| T5.2 | Vector env `ConfiguredTaskFamily`, a thin constructor over P1-2b's per-env `tasks=` (v7_2 scenes + goal + planner options + per-slot sim seeds; uniform checks; `UNSUPPORTED_SOLVER` for others; `UNSUPPORTED_NONUNIFORM_BATCH` when mixing K); fidelity computed against the declared scene plus a named Native reference (CUT: `native_reference: NOT_RUN` until T5.8); P1-2b's full evaluation stream used with host snapshots every 20 steps; receipts. Tests: API shapes, refusals, **CPU L0 bitwise CUT** (full-resolution substeps with blade contact and link completion; coarse 2 mm full episode `reduced_fidelity`; N=1 and N=3 copies vs frozen single v7_2), partial-reset bitwise on CUT | T5.1; P1-2b merged | CPU (Torch venv for API) | — | — | M |
| T5.3 | `sever_bond_progress_packed_v1` shaping reward (target = lattice bonds crossing a declared sever plane; off-target = severed bonds farther than the frozen 2 mm bound from every plane; status `PROVISIONAL_SHAPING_NOT_SUCCESS`); reward dispatch by goal type in the vector env and the Lab env; unit tests vs a NumPy reference | T5.1 (core); T5.2 (dispatch) | CPU | — | — | S–M |
| T5.4 | `reference_command_policy_v1`: compiled planner schedule → `edge_delta_v1` via `recorded_command_actions`; tests: reconstruction residual ≤ 1e-9 m against `reference_edge_pose_world`, 700 commands, digit schedule equal | T5.1 | CPU | — | — | S |
| T5.5 | Replay adapter for the WP-12 recording format (Demo 1): streaming read of `actions.jsonl.gz` (source untouched), `physics_state.npz` samples, `material.json` / `finger_geometry_00.json` / `setup.json`, per-step pad boxes from `robot_state.npz`; source-equivalence audit for the v5 path (`tlmpm_v5.py` SHA, `live_material.py` diff since `a960dc6`, runner pad semantics); CPU tests on the first 20 steps | — | CPU | — | — | M |
| T5.6 | **single_env_headless CUT rows**, one material per process: (i) frozen v5 Demo 1 replay × 4 processes (final codes vs 75 Native samples, position band, frozen evaluator, plan row columns); (ii) frozen single v7_2 ideal-tracking with T5.4 commands × 3 processes (the single reference band for T5.7) | T5.4, T5.5 | GPU1 headless | ≤ 1 GiB each | about 40 min | S |
| T5.7 | **Batched CUT** (C2): M0 N=1; M1 N=4 (2 reference, 2 random, staggered partial resets); M2 N=8 × 2 episodes and N=16 × 1 episode; L2 every step; evaluator per episode with ≥ 36 frames; compare with the T5.6 (ii) band; benchmark rows; optional (d) cross-solver replay of Demo 1 at N=1 (descriptive) | T5.2–T5.6 | GPU1 headless | ≤ 2.2 GiB | about 3 h | M |
| T5.8 | **FULL_NATIVE_REFERENCE for C2**: one Native scripted episode of the new scene with the `workcell_live_v1` flags (cameras on, physical grasp, 700 + home steps); then measured-trajectory replay through the frozen single v7_2 (× 3 processes) and the batch (N=1, N=4 copies) vs the Native samples, P1-0 style | T5.1, T5.5 | GPU0 (or GPU1) Kit, one Native at a time | about 6 GiB (needs ≥ 10 GiB free) | about 20 min Native + 45 min replays | M |
| T5.9 | (stretch) **Lab B1 CUT** (C3): `open_lab_batched_env(task=C2 task, tool_mount="rigid_mount_v1", material="tlmpm_v7_2")` N=2/4 with the reference driver; evaluator per env; G5-style single replay of the batch's inputs (one per process) | T5.3, T5.4 | GPU0 Lab venv | ≤ 1.2 GiB | about 1 h | M |
| T5.10 | **Owned by P1-2b** (in progress): `open_vector_env(backend="isaac_lab_newton", tool_mount=...)` dispatch (the P1-1b list: `launch_simulation` lifetime, reset / partial-reset mapping, rewards, refusals). P1-5 only verifies it: a GPU smoke N=2 B1, 50 steps, once merged | P1-2b | GPU0 Lab | ≤ 1 GiB | 15 min | (M, P1-2b) |
| T5.11 | **Owned by P1-2b** (in progress): unified `info["termination_reason"]` (bits shared with the Lab env: 1 rigid, 2 material, 4 grasp lost, 8 IK streak; `invalid_reason` kept as detail). P1-5 consumes it in the smoke format | P1-2b | CPU | — | — | (S, P1-2b) |
| T5.12 | RL smoke (§5): rollout writer, `run_rl_smoke.py`, `check_rl_rollout.py`; S-CPU pytest; S-TRIM N=8 and S-CUT N=4 GPU runs | T5.2–T5.4, T5.11 | GPU1 headless | ≤ 1.5 GiB | about 70 min | M |
| T5.13 | Evidence `docs/evidence/rl_ready/p1_5_parity_001.json`, report `P1_5_FEATURE_PARITY.md` (matrix re-filled from evidence only), `RL_READY_PROGRESS.md` entry; then an xhigh review (rcs-review) | all above | CPU | — | — | S + review |
| (T5.14) | **Deferred, user decision U-P15-1:** option (a) batched `tlmpm_v5` (§2.1) with an ADR 0021 amendment | U-P15-1 | CPU, then GPU1 | ≤ 2 GiB | about 2 h GPU | L |

### P1-4

| Id | Task | Depends | Where / GPU | VRAM | Wall | Size |
| --- | --- | --- | --- | --- | --- | --- |
| T4.1 | Supervisor, child, streaming ledger, classification and resume (§6.4); fault-injection CPU tests; census hooks | — | CPU | — | — | M |
| T4.2 | **Owned by P1-2b** ("per-env pose-bounds invalidation", in progress): no crash on out-of-domain inputs. Required properties: the P1-1b SIGSEGV is reproduced with an injected far-away or non-finite edge pose (CPU first, then GPU N=2); the substep inputs of envs whose `pose_status` is invalid are neutralized and grid-index computation is guarded against non-finite positions; the env gets an `INVALID_*` bit; CPU L0 stays bitwise for valid envs. P1-4 LR-D re-checks it under the supervisor | P1-2b | CPU, then GPU1 (5 min) | ≤ 1 GiB | — | (M, P1-2b) |
| T4.3 | Pre-declare `p1_4_long_rollout_v1.json` (runs, seeds, drivers, offsets, cadence, bounds B1–B5, warm-up rule, canary rule) | — | CPU | — | — | S |
| T4.4 | LR-A main run (`frozen_v1`) | T4.1, T4.3 | GPU1 or GPU0 headless | ≤ 1.5 GiB | 4.0–4.7 h | S (execution) |
| T4.5 | LR-B, LR-C, LR-D; LR-A2 after the P1-3a merge | T4.1–T4.3 | same | ≤ 2.2 GiB | about 3.2 h total | S |
| T4.6 | (optional) LR-E Lab B1 on GPU0; LR-F CUT after T5.2–T5.4 | T4.1, T5.x | GPU0 Lab / GPU1 | ≤ 1.4 GiB | 1.3 h / 2.5 h | S |
| T4.7 | Evidence `p1_4_long_rollout_001.json`, report `P1_4_LONG_ROLLOUT.md`, xhigh review | above | CPU | — | — | S + review |

**Order.**

1. Now, in parallel (CPU): T5.1, T5.3 (reward core), T5.4, T5.5, T4.1, T4.3. P1-2b delivers T5.10, T5.11 and T4.2.
2. After P1-2b merges: T5.2, then the GPU queues.
3. GPU1 queue: T5.6 (needs only T5.4 / T5.5, so it can start earlier) → T5.7 → T5.12 → T4.4 → T4.5.
4. GPU0 queue: T5.8 → T5.9 → T4.6 (LR-E).
5. Close with T5.13 and T4.7.

GPU total: about 9–10 h on GPU1 plus about 3 h on GPU0, all `SHARED_GPU` unless a window frees up.

## 8. Risks, open items, user decisions

- **R1, optimized v7_2 has not cut the bar.** The existing v7_2 CUT PASS used the pre-WP-16 source (§1.6). T5.6 (ii)
  runs before any batched CUT. A FAIL is the result (§2.2 escalation rule).
- **R2, no Native v7_2 CUT exists.** Until T5.8, the C2 parity reference is the frozen single v7_2 under identical
  commands, not Native. Native confirmation costs one Kit episode.
- **R3, ideal tracking.** C2 has no robot dynamics, and the reaction is not fed back. Its receipt computes
  `reduced_fidelity` with reasons, exactly like workload A. B1 CUT (T5.9) is a stretch goal, and B2 stays FAIL.
- **R4, WP-12 recording format.** The adapter (T5.5) must reproduce the Native per-step pad-box semantics of
  `a960dc6`. Pad / material contact may again never occur. If so, that is reported as in P1-0.
- **R5, shared GPUs.** Every number is co-tenant dependent. Rows are labelled, and only matched-load alternations
  support ratios (P1-2 §6).
- **R6, the HashGrid hazard.** It is handled by §6.1. A frozen single material built on CUDA after a batch is
  detected only at the next full reset (P1-2 §8). Scripts must not build one in a batch process.
- **R7, parallel ownership.** P1-2b (dispatch, `tasks=`, `termination_reason`, evaluation stream, pose-bounds
  invalidation) and the B2 grasp work run in their own worktrees. T5.2, T5.12 and the P1-4 runs start only on a
  branch head that contains the P1-2b merge, and record that commit in every manifest.
- **U-P15-1.** Batched frozen v5 CUT required in Phase 1? Default: no. (a) is deferred and the CUT row stays PARTIAL.
- **U-P15-2.** Batched vision profile required in Phase 1? Default: no. The camera row stays UNSUPPORTED.
- **U-P15-3.** Per-episode domain randomisation (geometry / material) inside a live batch required? Default: no.
  Per-slot seeds and new-env rebuilds only.

## 9. Read and probed for this plan

**Read:**

- `docs/RL_READY_PLAN.md`, `docs/RL_READY_SPECIFICATION.md` (P1-4, P1-5, Phase 1 Done, brief), `docs/RL_READY_PROGRESS.md`;
- reports P1_0, P1_1_CORE_REVIEW, P1_1B, P1_1_P1_2_MILESTONES (the committed `f0a544b` version), P1_3, P1_3A;
- ADR 0021, ADR 0022;
- `rl/vector_env.py`, `rl/lab_batched_env.py`, `rl/policies_batched.py`, `rl/reward_packed.py`;
- `batched/layout.py`, `batched/material_batch.py` (guard, invalid bits, release), `batched/constitutive.py`,
  `batched/tlmpm_v7_2_batch.py` (kernel list);
- `tlmpm_v5.py` (`step`, kernels), `tlmpm_v7_2.py` (optimization flags), `goals.py`
  (`evaluate_goal`, sever outcome), `tasks.py` (`make_task`, v1 oracle), `scene_config.py` (solvers, compile,
  reference schedule), `seeded_trim.py`, `live_material.py` (+ diff since `a960dc6`);
- `configs/cutting_tasks_v1.json`, `configs/wp12/scenes/bar_flat_board.json`, `configs/wp12/workcell_live_v1.json`,
  `configs/wp12/phase7_demos_v1.json`, `configs/wp11/goals/sever_mid.json`, `configs/wp14/*` (scene, goal,
  `v7_cut_research_v1`), `configs/wp14_followup/*`, `configs/rl_benchmark/workloads_v1.json`;
- `docs/PROGRESS.md` WP-12..WP-17 sections, `docs/reports/WP14_FOLLOWUP_RESULTS.md`,
  `docs/evidence/wp14_followup/cut_regression_001.json`;
- the Demo 1 Native run directory (manifest, task, result, material, robot / physics array shapes).

**CPU probe 1** (no Warp kernels):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -   # compile_task(bar_flat_board with module v5 / v7_2,
#   configs/wp14/goals/standard_cut.json, planner_options {max_rotation_deg 1, max_translation_m 0.001,
#   controller_profile path_v2}, seed 42); sample_particles, schedule, action_limits; plus wp14 standard_cut
```

Result: v5 `CFG-bar_flat_board-sever_mid-c32f94a4`, v7_2 `CFG-bar_flat_board-sever_mid-46180291`, 96,000
particles, 0.144 kg, 700 steps, DONE 11.48 s, digits {−1, 0}, limits 1 mm / 1°. WP-14 `standard_cut`:
`CFG-bar_flat_board-sever_mid-5da99059` (`tlmpm_v7`), 96,000 particles.

**CPU probe 2** (Warp 1.17.0 CPU device, kernel cache in the session scratchpad, no repository writes):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src TMPDIR=/mnt/data/ningan/tmp .venv/bin/python -
#   LiveCuttingMaterial(CouplingConfig(0.02, 1e-4, dx 0.004, device='cpu', feedback=True), task=<v7_2 CUT>,
#   max_cuts 4, cutting_law 8, solver 'tlmpm_v7_2', v7_2_optimizations=True); layout.pack([EnvSpec(...)]);
#   the same for the v5 task
```

Result: v7_2 pack OK in 9.92 s (particles [96000], nodes [5850], S 8, K 4, boxes 1, segments 120, fixtures
True; single memory 45.28 MB). v5 raises `UnsupportedBatchConfiguration: UNSUPPORTED_SOLVER: solver 'tlmpm_v5';
only tlmpm_v7_2`.
