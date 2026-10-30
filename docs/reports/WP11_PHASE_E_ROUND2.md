# WP-11 Phase E — generalization round 2 (goal_eval_v2, Test D2, feasibility sweep)

Date: 2026-10-30. Branch `claude/wp11-cutting-refinement`.
Round file: [`configs/wp11/generalization_round2.json`](../../configs/wp11/generalization_round2.json), predeclared in
`0496562` before any declared sample was built, preflighted or run.
Evidence: [`docs/evidence/wp11/phaseE_round2_001.json`](../evidence/wp11/phaseE_round2_001.json) (round summary).
Diagnostics: [`docs/evidence/wp11/phaseE_round2_diagnostics_001.json`](../evidence/wp11/phaseE_round2_diagnostics_001.json)
(read-only: post-hoc v2 rescoring, the S = 8 / repeat check and per-run label diagnostics).

**Scope.** Every stage-2 verdict is a **kinematic ideal-tracking** result (`runtime_source = kinematic_ideal_tracking`):
the knife edge goes exactly where it is commanded. Robot dynamics, grasp, slip, controller tracking and cameras are not
modelled, and the material is provisional and uncalibrated. These are **not live results**. `GRASP_FAILURE` is
live-only and is never assigned here. Stage 1 is geometry only: Newton IK plus the C2 MuJoCo collision model with
the held `ruler_knife_v2_y`, in `wp11_workcell_v1` (HOME = `q_home`). Round-1 files, verdicts and evidence are
unchanged.

## Result

Stage 1 covers all 46 samples. Stage 2 is kinematic, on the declared subset of 18 samples. Columns:
- **Feasible** = samples − SCENE_INFEASIBLE − IK_FAILURE.
- **NOT_RUN** = feasible sweep samples outside the predeclared stage-2 subset. They stay in every feasible
  denominator.

GRASP_FAILURE is 0 in every row (it is not exercised in the kinematic stage). Nothing is excluded, and each
declared seed is reported once.

| Family / test | Samples | SCENE_INFEASIBLE | IK_FAILURE | Feasible | Stage-2 run | CUTTING_FAILURE | NOT_RUN (not in subset) | Kinematic TASK_SUCCESS | Success among stage-2 | Success among feasible | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TEST_D2 | 6 | 0 | 0 | 6 (100%) | 6 | 3 | 0 | 3 | 50% | 50% | 50% |
| CUT_W | 10 | 0 | 2 | 8 (80%) | 3 | 0 | 5 | 3 | 100% | 38% | 30% |
| OBLIQUE_W | 10 | 0 | 0 | 10 (100%) | 3 | 3 | 7 | 0 | 0% | 0% | 0% |
| TRIM_W | 10 | 3 | 3 | 4 (40%) | 3 | 3 | 1 | 0 | 0% | 0% | 0% |
| TEST_A_W | 10 | 1 | 3 | 6 (60%) | 3 | 1 | 3 | 2 | 67% | 33% | 20% |
| **Sweep (C), 40** | **40** | **4 (10%)** | **8 (20%)** | **28 (70%)** | 12 | 7 | 16 | **5** | 42% | 18% | 12% |
| **All, 46** | **46** | 4 | 8 | 34 (74%) | 18 | 10 | 16 | **8** | 44% | 24% | 17% |

**Reading the result.**

- **A. goal_eval_v2 works as designed.**
  - Under v2, the round-1 Test D s1702 and s1704 rescore to PASS (`POST_HOC_DIAGNOSTIC`). Their round-1
    CUTTING_FAILURE is unchanged.
  - In round 2, every TEST_D2 sever step and every v2 substrate check passed (scoped retention included). The
    three TEST_D2 failures are all physical non-separation of one chip: code-zero bridges. None is an
    evaluator-composition failure.
- **B. S = 16.** No overflow. The maximum occupancy was 8 slots (s2502, the round-1 s1701 pattern: oblique sever
  plus skims). A post-round diagnostic shows the D2 skim failure does not depend on S: S = 8 and an S = 16 repeat
  give the same result with the same bridging particles.
- **C. The sweep now crosses the workcell boundary.**
  - Stage 1 classed 30% of the sweep as infeasible: 8 IK_FAILURE and 4 SCENE_INFEASIBLE. Round 1 had 0 of 41.
  - The kinematic subset found three physics/law boundaries:
    1. ±30–35° oblique severs never separate (the bottom layers are left unlabelled);
    2. ρ 45° skims assign no labels at all;
    3. some D2 skims leave 1–4 interior chip particles unlabelled.
  - These are kinematic results; nothing was relaxed.

## A. Goal evaluator v2 (`goal_eval_v2`, ADR 0015 Amendment 1)

Opt-in and versioned. A goal selects it with the top-level field `"evaluator_version": "goal_eval_v2"`. Without the
field, a goal validates to the same normalized form as before, so compiled task ids, plans and v1 evaluations are
byte-identical. Commits `b624df6` (evaluator, tests, ADR amendment), `ccfdd8a` (scene `slots_per_node`).

Design (full text: [ADR 0015 Amendment 1](../adr/0015-scene-goal-controller-evaluation-separation.md)):

1. **Digit allocation per step.** Steps are taken in order, and digits are numbered across the episode:
   - `sever`: one digit per plane;
   - `divide`: pieces − 1;
   - `remove_protrusions`: one per selected protrusion;
   - `trim_plane`: one;
   - `move_piece`: none.

   This is the goal-planner numbering. With a compiled task, the allocation is cross-checked against the plan
   report (`MATCH` on every round-2 sample and on both post-hoc samples). A `MISMATCH`, or a recording that keeps
   too few digits, makes the sequence `NOT_RUN`. It is never scored.
2. **Scoped components.** The mechanical components of the final codes with every digit outside a set *D* zeroed:
   the same eight-node compatibility graph, packed or ternary.
3. **`sever` / `divide` step.** All sever checks (pieces, main mass, location, far-particle rule) use the step's **own**
   digits. Location therefore comes only from that cut's own separation interfaces.
4. **`remove_protrusions` / `trim_plane` step.**
   - *Prior pieces* are the anchored components of the earlier sever/divide steps' digits.
   - *After* is the components of those digits plus the step's own.
   - Each prior piece must keep one substrate component, which holds ≥ 0.99 of the piece's below-plane mass, with
     ≤ 0.01 of it detached. So a step may not split material below its plane.
   - Removal and retention use the scoped substrate. The digits of other remove/trim steps are out of scope.
5. **Unchanged parts.** `move_piece` steps and non-sequence goals are scored exactly as in v1. **Every threshold is
   the v1 threshold**, and every step must still hold at the end of the episode.

### CPU tests

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest tests/test_goal_eval_v2.py` | 10 passed, rc 0 |
| `… tests/test_scene_slots_per_node.py tests/test_phase2b_wiring.py tests/test_tlmpm_v5.py "tests/test_scene_goal_config.py::test_kinematic_smoke_three_protrusions_tlmpm_v5"` | 67 passed, rc 0 |
| `… tests/test_generalization_round2.py tests/test_generalization_round1.py tests/test_generalization_round1_live.py tests/test_generalization_round1_live_003.py tests/test_goal_eval_v2.py tests/test_scene_slots_per_node.py tests/test_scene_goal_config.py tests/test_wp11_phase_c5b.py` | 120 passed, rc 0 |

`tests/test_goal_eval_v2.py` covers the following. All fixtures use synthetic packed codes on `bar_three_bumps`
(4 mm particles).

- **v1 unchanged.**
  - Three v1 evaluations (sever-then-remove sequence, remove all, sever) are pinned by the SHA-256 of their
    canonical JSON. The pins were computed with the pre-v2 `goals.py` of `c29f02a` (identical digests).
  - The compiled ids are pinned: C5b `CFG-trim01_wp11_workcell-trim_bump_rho10-a33f6613`, and the round-1 stage-1
    ids of Test D s1702/s1704, SLICE-PUSH s1301, Test C s1601 and CUT s1102.
  - v1 reproduces the round-1 failure on the synthetic history: `cut_at_declared_planes`,
    `single_substrate_component` and `substrate_holds_below_plane_mass`.
- **v2 scoped verdicts.**
  - Sever then remove all gives PASS: the sever's own-digit statistic is ≤ 2 mm, there are 2 prior pieces each
    with one substrate, and removal is 1.0 / 1.0 / 1.0. Ternary codes give the same items.
  - Remove p1, then sever, then trim p0's region gives PASS under v2. v1 fails retention and the single-substrate
    rule on the same history.
  - On a single goal, v2 = v1 (same digest).
- **Negative cases.**
  - A remove digit that also cuts straight down through the +x piece: v2 FAIL, with
    `single_substrate_component_per_prior_piece` and `substrate_holds_below_plane_mass_per_prior_piece` false. The
    split piece's share is < 0.99, and the sever step still passes.
  - A sever 8 mm off its plane still FAILS `cut_at_declared_planes`.

### Post-hoc diagnostic: round-1 Test D s1702 / s1704 under v2 (`POST_HOC_DIAGNOSTIC`)

Read-only. The round-1 verdicts (CUTTING_FAILURE), classes and evidence are **unchanged**; this is not a round result.

    cd /mnt/data/ningan/RoboticCuttingSim-wp11-E2-0496562        # clean worktree of 0496562
    TMPDIR=/mnt/data/ningan/tmp /home/ningan/RoboticCuttingSim/.venv/bin/python scripts/rescore_goal_eval_v2.py \
        --stage1 …/artifacts/wp11/phaseE/round1_stage1_001 --material-root …/artifacts/wp11/phaseE/round1_kinematic_001 \
        --samples E1_TEST_D_s1702 E1_TEST_D_s1704 --out …/artifacts/wp11/phaseE/round2_posthoc_v2_001   # rc 0, 6 s

How it works:

- `compile_task` on the recorded stage-1 scene and goal reproduces the recorded task id.
- The preserved `physics_state.npz` is the recording.
- v1 rescoring reproduces the recorded verdict and failed checks exactly (`match = True` for both).

| Sample | Step | v1 (recorded and rescored) | v2 | v2 numbers |
| --- | --- | --- | --- | --- |
| s1702 (sever 0°, remove p0+p1, gaussian ×2) | sever (digit 0) | FAIL `cut_at_declared_planes`: frozen statistic 77.0 mm (156 cross-component nodes incl. chip interfaces) | **PASS** | 2 pieces 0.703 / 0.297; own-digit statistic 1.50 mm (54 nodes) = ideal 1.50 mm ≤ 2 mm |
| | remove_all (digits 1, 2) | FAIL `single_substrate_component` (2 substrate components), `substrate_holds_below_plane_mass` (share 0.700) | **PASS** | prior pieces 0.297 / 0.703: each one substrate, share 1.000, detached 0; removed p0 0.970, p1 0.960 |
| s1704 (sever 0°, remove p0, dome) | sever (digit 0) | FAIL: statistic 48.0 mm (127 nodes) | **PASS** | 2 pieces 0.724 / 0.276; own-digit statistic 1.50 mm (63 nodes) |
| | remove_all (digit 1) | FAIL: 2 substrate components, share 0.722 | **PASS** | prior pieces 0.276 / 0.724: each one substrate, share 1.000, detached 0; removed p0 0.984 |

Both rescore to v2 **PASS**. The digit allocation cross-check is `MATCH` for both. The integrity layer is PASS. The
v2 verdict is a diagnostic of the evaluator design, not a re-classification: round 1 keeps both as CUTTING_FAILURE.
Outputs: `artifacts/wp11/phaseE/round2_posthoc_v2_001/` (`posthoc_summary.json`, per-sample `*_v1.json`,
`*_v2.json`), log `artifacts/wp11/logs/phaseE_E2_posthoc_v2_001.log`.

## B/C. Round 2 declaration

| Family | Seeds | Purpose and ranges (round 1 in brackets) | Goal | Stage 2 |
| --- | --- | --- | --- | --- |
| TEST_D2 | 2501–2506 | Round-1 Test D composition with goal_eval_v2 and S = 16. L 150–170, W 28–34, H 16–22 mm; xy ±20 mm, yaw ±20°; cut at −(0.18–0.30) L; p0 at −0.02…0.06 L, p1 at 0.30…0.36 L (footprints ≥ 14 mm apart); radii 8–11, h 5–9 mm; rake 10/25° | declared patterns: `sever:0,remove:all` · `sever:20,remove:all` · `remove:p0,sever:0` · `sever:0,trim:p1` · `sever:20,remove:p0,trim:p1` · `trim:p0,sever:0,remove:p1` | all feasible |
| CUT_W | 2101–2110 | vertical sever; L 100–200 [130–170], W 20–45 [26–34], H 12–35 [16–24] mm; xy ±100 [±20] mm; yaw ±60° [±20°]; cut ±0.35 L [±0.2 L] | sever | first 3 feasible |
| OBLIQUE_W | 2201–2210 | oblique sever at 35, −35, 30, −30, 25, −25, 20, −20, 15, −15° by seed index [0/20°]; xy ±50 mm, yaw ±40°; cut ±0.25 L | sever | first 3 feasible |
| TRIM_W | 2301–2310 | one protrusion; L 100–200, W 20–45, H 12–35 mm; xy ±100 mm, yaw ±60°; protrusion x ±0.42 L [±0.25 L], y ±6 [±2] mm, radii 6–14 [9–13] mm, h 3–16 [5–9] mm; rake 5/35/45/60° [10/25°] × 3 families | remove all | first 3 feasible |
| TEST_A_W | 2401–2410 | fixed Test A bar on a drawn pose (xy ±100 mm, yaw ±60°) [identity]; vertical, ±35° [±20°], slanted advance 30° to −y / +y [15°], skims ρ 5 / 45 / 60°, −x skim ρ 25°, skew 35° [20°] | sever / remove | first 3 feasible |

Common (as round 1):
- `wp11_workcell_v1` (+100 mm), `ruler_knife_v2_y`, `path_v2`, `plane_section`, law 8, 1 mm particles, 4 mm grid,
  0.1 ms, material seed 42;
- sever/remove rules and thresholds of round 1;
- `tlmpm_v4` for ≤ 2 digits, otherwise `tlmpm_v5`. The TEST_D2 family override is `tlmpm_v5` with
  `slots_per_node = 16` for every digit count;
- duration = plan end + 1 s, rounded up to 0.5 s.

History is max(5, ceil(steps / 20)) frames, half of round 1's 40, declared to protect the /mnt/data 15 GB floor.

**Stage-2 subset rule** (`stages.stage2_subset`): every feasible TEST_D2 sample, plus the first 3 stage-1-feasible
samples per sweep family in seed order. Every other feasible sweep sample is NOT_RUN ("not in the predeclared stage-2
subset") and stays in the feasible denominator.

**Rules added before the declaration commit.** These were found on the non-declared development seeds 90001–90006.
The only other pre-declaration run was one 60-step kinematic wiring smoke, `artifacts/wp11/phaseE/round2_dev/`.

1. **Stage-1 geometry `staged_v2`** (opt-in; round 1 keeps `staged_v1`). In dev seeds 90005/90006, a trim of p0
   or a remove of p1 ran while the other protrusion was still present. The C2 model's single convex hull over both
   protrusions then bridged their ≥ 14 mm gap, and PRE_CUT:p1 "collided" by −7.2 mm. In addition, a trim_plane
   pass was not counted as a removal. `staged_v2` checks blade/robot against the union of per-protrusion hulls,
   one single-protrusion checker per remaining protrusion, and counts a trim_plane pass at the base top as removing
   the protrusions whose centre lies in its region.
2. **TEST_D2 pattern 4 removes p0 first.** On dev seed 90005, removing p1 while p0 was present put the trailing
   blade body over p0 at ρ 10°: −0.75 mm against p0's *own* hull, a real contact in a contact-free phase. Patterns
   only remove or trim a protrusion whose trailing (−x) side is clear.
3. **Scene `solver.slots_per_node`** (opt-in, `tlmpm_v5` only). It is carried to `make_configured_kinematic_env`
   and to `LiveCuttingMaterial` (from the configured task, so the Native runner uses it without a runner change).
   Absent, scenes and ids are unchanged. The dev smoke reported `slots_per_node 16`.
4. **TRIM_W range.** Protrusion y was narrowed to ±6 mm and radii to 6–14 mm before declaring. A simple footprint
   model then keeps about 73% of draws inside the base, against 58% with ±8 mm / 6–16 mm.

## Stage 1 — motion preflight (all 46 samples)

    cd /mnt/data/ningan/RoboticCuttingSim-wp11-E2-0496562        # fresh detached worktree of 0496562, git_dirty = False
    TMPDIR=/mnt/data/ningan/tmp /home/ningan/RoboticCuttingSim/.venv/bin/python scripts/run_generalization.py \
        --config configs/wp11/generalization_round2.json preflight \
        --out /home/ningan/RoboticCuttingSim/artifacts/wp11/phaseE/round2_stage1_001     # 09:12:33-09:20:27 KST, rc 0

Stage 1 ran on CPU with `staged_v2`. Mean wall per sample:

| Family | Mean wall per sample (s) |
| --- | --- |
| TEST_D2 | 20.1 |
| CUT_W | 10.4 |
| OBLIQUE_W | 12.1 |
| TRIM_W | 4.9 |
| TEST_A_W | 7.9 |

The progress lines print an `[E1]` tag. That is a cosmetic print label; sample ids are `E2_*`.

| Class (sweep, 40) | Count | Rate |
| --- | --- | --- |
| Preflight-feasible | 28 | 70% |
| IK_FAILURE | 8 | 20% |
| SCENE_INFEASIBLE | 4 | 10% |

TEST_D2: 6 / 6 feasible. Every one has a planned informational return-home.

Every stage-1 failure, with its diagnosis (draws from `samples[].draws` / `derived`):

| Sample | Class | Where | Diagnosis |
| --- | --- | --- | --- |
| CUT_W s2106 | IK_FAILURE | APPROACH:cut0, pose error 2.2 mm > 1 mm | xy (+81, +78) mm, yaw +41° — reach limit of the chained IK |
| CUT_W s2107 | IK_FAILURE | APPROACH:cut0, 1.5 mm | xy (−71, −69) mm, yaw +45° |
| TRIM_W s2303 | IK_FAILURE | SKIM:p0 (armed), 0.30 mm > 0.2 mm armed tolerance | 198 mm bar, xy (−26, −82) mm, yaw +31°, ρ 5°, protrusion at −61 mm |
| TRIM_W s2305 | IK_FAILURE | REORIENT:p0, 2.4 mm | xy (+63, −17) mm, yaw +44°, ρ 35°, 30 mm bar |
| TRIM_W s2306 | IK_FAILURE | REORIENT:p0, 1.5 mm | xy (+27, −50) mm, yaw +41°, ρ 35°, 32 mm bar |
| TEST_A_W s2402 | IK_FAILURE | APPROACH:cut0, 2.0 mm | +35° oblique; xy (+8, −78) mm, yaw +47.5° |
| TEST_A_W s2405 | IK_FAILURE | APPROACH:cut0, 1.1 mm | slanted advance +y 30°; xy (+68, −3) mm, yaw +12° |
| TEST_A_W s2409 | IK_FAILURE | REORIENT:p0, 1.4 mm | ρ 10° with edge skew 35°; xy (+31, −46) mm, yaw −12° |
| TRIM_W s2301 | SCENE_INFEASIBLE | blade_collision −0.25 mm vs `static_workpiece_base`, PRE_CUT:p0 | ρ 5° (all three ρ 5° samples that reached the collision check fail identically) |
| TRIM_W s2302 | SCENE_INFEASIBLE | same, −0.25 mm | ρ 5° |
| TEST_A_W s2406 | SCENE_INFEASIBLE | same, −0.25 mm | ρ 5° |
| TRIM_W s2304 | SCENE_INFEASIBLE | planner/compile: the reference schedule never reaches RETURN_HOME | The workcell HOME edge pose is at local y = −121.6 mm, outside the wp11 action workspace (local y ≥ −120 mm), for xy (−97, +76) mm, yaw +43°. The "needs 6.30 s" in the message is the unclamped duration estimate. |

Boundaries measured:

- **Rake 5° is infeasible geometrically.** The blade underside sits 0.25 mm below the base top with the
  0.2 mm edge clearance, in a contact-free phase.
- **IK fails toward the reach limit.** IK fails at |xy| ≳ 70–110 mm combined with yaw ≳ 40°. Separately, the
  steep orientations (ρ 35° on tall bars, 35° skew, 35° oblique) fail at REORIENT or APPROACH.
- **Narrow margins among the feasible samples.** TEST_A_W s2408 (−x skim at xy (+3, −98) mm) passed with only
  0.93 mm Franka-1 clearance. ±30–35° obliques bind on `knife_handle_clearance` at 5.6–6.1 mm (vertical: 11.1 mm).
- **Caveat on IK_FAILURE.** It is the declared chained-IK class from q_home. A failure there is not proof that
  no IK solution exists.

## Post-hoc and post-round diagnostics (read-only)

- **v2 rescoring of round-1 s1702/s1704.** See section A.
- **The round-2 TEST_D2 episodes scored with v1** (`POST_HOC_DIAGNOSTIC`). The script and output are
  `artifacts/wp11/phaseE/round2_diag_001/v1_on_test_d2_posthoc_001.{py,txt}`. The goal was scored without
  `evaluator_version` on the same recordings, and the round class is v2's. **v1 FAILs all six**, including the
  three v2 successes:
  - s2503 fails `single_substrate_component` (remove before sever) and `cut_at_declared_planes`;
  - s2504 fails `cut_at_declared_planes` and the trim's single substrate;
  - s2505 fails `main_pieces_retain_mass`, `cut_at_declared_planes`, `unselected_protrusions_retained` (p1 is
    trimmed later) and the single substrate.

  s2505 shows why v2 scopes the sever's *piece* checks as well as its location: the two chips from the later
  steps hold more than 1% of the mass. The v2 verdicts fail only on the physical non-separations.
- **Slot capacity and repeat check (s2501).** This was a post-round diagnostic, not a round sample. The S = 8
  config is the round file minus `slots_per_node`: `artifacts/wp11/phaseE/round2_diag_001/diag_s8_config.json`,
  sha256 in the diagnostics evidence.

  | Run | S | Device | Class | Removed p0 / p1 | Unassigned owned particles | Episode s |
  | --- | --- | --- | --- | --- | --- | --- |
  | round run | 16 | cuda:1 | CUTTING_FAILURE | 0.051 / 0.000 | p0 (16.5, 0.5, 24.5), (16.5, 1.5, 24.5) mm; p1 (57.5, 1.5, 24.5) mm | 498 |
  | S = 8 rerun | 8 | cuda:0 | CUTTING_FAILURE | 0.048 / 0.000 | identical positions | 479 |
  | S = 16 repeat | 16 | cuda:1 | CUTTING_FAILURE | 0.057 / 0.000 | identical positions | 509 |

  The failure is reproducible and independent of S. Its maximum occupancy is 6.
- **Failure diagnostics.** The command is `scripts/diagnose_generalization_round2.py` (committed `39c4e82`, run
  from a clean worktree at `39c4e82`). It wrote
  [`docs/evidence/wp11/phaseE_round2_diagnostics_001.json`](../evidence/wp11/phaseE_round2_diagnostics_001.json)
  (`failure_diagnostics`). The same script run on the round-1 runs gave
  `artifacts/wp11/phaseE/round2_diag_001/diagnostics_round1_runs_001.json`:
  - 29 round-1 skims, 0 unassigned owned particles;
  - five 20° oblique severs, 0 unassigned particles within 8 mm of the plane.

## Stage 2 — kinematic material physics (declared subset, 18 samples)

    cd /mnt/data/ningan/RoboticCuttingSim-wp11-E2-0496562
    TMPDIR=/mnt/data/ningan/tmp /home/ningan/RoboticCuttingSim/.venv/bin/python scripts/run_generalization.py \
        --config configs/wp11/generalization_round2.json queue \
        --stage1 /home/ningan/RoboticCuttingSim/artifacts/wp11/phaseE/round2_stage1_001 \
        --out-root /home/ningan/RoboticCuttingSim/artifacts/wp11/phaseE/round2_kinematic_001 \
        --log-dir /home/ningan/RoboticCuttingSim/artifacts/wp11/logs \
        --queue-log /home/ningan/RoboticCuttingSim/artifacts/wp11/logs/phaseE_E2_queue_001.log \
        --devices cuda:1 cuda:0 --max-concurrent 2 --max-total 2 --max-other 2 --min-free-gb 15
    /home/ningan/RoboticCuttingSim/.venv/bin/python scripts/run_generalization.py --config configs/wp11/generalization_round2.json \
        summarize --stage1 …/round2_stage1_001 --material-root …/round2_kinematic_001 --log-dir …/artifacts/wp11/logs \
        --out /home/ningan/RoboticCuttingSim/docs/evidence/wp11/phaseE_round2_001.json

How the queue ran:

- **Timing.** 09:20:59–10:35:58 KST. All 18 runs had rc 0 and no aborts. 16 ran on cuda:1 and 2 on cuda:0; the
  device is recorded per sample.
- **Concurrency.** At most 2 of this queue's jobs ran at a time in total. Each job was launched only with ≤ 2
  other active jobs on its GPU (logged per START). Non-RCS processes were never touched.
- **Disk.** `/mnt/data` went from 25 GB to 18 GB free; other agents also wrote. Round-2 outputs total 0.88 GB:
  kinematic 787 MB, diagnostics 93 MB, stage 1 0.8 MB, dev 3.6 MB. The 15 GB floor was never reached.

Runtime per episode (wall s; mean [min–max]; wall per simulated second; shared GPUs, so these are throughput
figures):

| Family | n | Episode wall | per sim-s | physics_state MB (mean / max) |
| --- | --- | --- | --- | --- |
| TEST_D2 | 6 | 580 [488–830] | 32.9 | 42.0 / 49.2 |
| CUT_W | 3 | 547 [304–697] | 46.3 | 60.7 / 72.7 |
| OBLIQUE_W | 3 | 317 [288–370] | 29.3 | 33.8 / 37.7 |
| TRIM_W | 3 | 223 [200–268] | 31.2 | 53.7 / 77.9 |
| TEST_A_W | 3 | 389 [294–463] | 30.8 | 40.9 / 41.1 |

### Per sample (stage 2)

Metric definitions are as in round 1:
- **Completion:** substantial pieces / expected pieces, or minimum removed fraction. Sequences show the worst step.
- **Position error:** frozen statistic (vertical), actual − ideal (oblique), or the residual substrate height
  (removal; quality only).

| Sample | Goal | Sampled / derived | Solver | Particles | Class | Completion | Position error mm | Max slots | Episode wall s | Device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E2_TEST_D2_s2501 | sever:0 → remove:all | 166x30x22, xy (+1, +10), yaw +3.6°, ρ 10; ellipsoid_cap / dome | tlmpm_v5 S16 K3 | 111991 | CUTTING_FAILURE | 0.0 (p0 0.051, p1 0.000) | 18.6 (residual) | 6 | 498 | cuda:1 |
| E2_TEST_D2_s2502 | sever:20 → remove:all | 152x30x21, xy (−2, +8), yaw −4.7°, ρ 25; dome / ellipsoid_cap | tlmpm_v5 S16 K3 | 97621 | CUTTING_FAILURE | 0.0 (p0 0.981, p1 0.000) | 21.1 (residual) | **8** | 583 | cuda:0 |
| E2_TEST_D2_s2503 | remove:p0 → sever:0 | 170x32x16, xy (+16, +12), yaw −11.8°, ρ 10 | tlmpm_v5 S16 K2 | 88866 | **TASK_SUCCESS** | 0.959 | 6.0 (residual) | 3 | 488 | cuda:1 |
| E2_TEST_D2_s2504 | sever:0 → trim:p1 | 158x32x19, xy (+14, −12), yaw −0.9°, ρ 25 | tlmpm_v5 S16 K2 | 97148 | **TASK_SUCCESS** | 0.983 | 1.5 (sever) | 3 | 532 | cuda:1 |
| E2_TEST_D2_s2505 | sever:20 → remove:p0 → trim:p1 | 164x28x17, xy (−1, +10), yaw +18.7°, ρ 10 | tlmpm_v5 S16 K3 | 79039 | **TASK_SUCCESS** | 0.952 | 0.77 | 4 | 550 | cuda:1 |
| E2_TEST_D2_s2506 | trim:p0 → sever:0 → remove:p1 | 158x34x21, xy (+2, −13), yaw +9.2°, ρ 25; ellipsoid_cap / gaussian | tlmpm_v5 S16 K3 | 114161 | CUTTING_FAILURE | 0.0 (trim p0 0.000) | 14.2 (residual) | 6 | 830 | cuda:1 |
| E2_CUT_W_s2101 | sever | 158x38x28, xy (−23, −13), yaw +13.8° | tlmpm_v4 | 168112 | **TASK_SUCCESS** | 1 | 1.5 | — | 639 | cuda:1 |
| E2_CUT_W_s2102 | sever | 100x28x31, xy (−16, +64), yaw +38.1° | tlmpm_v4 | 86800 | **TASK_SUCCESS** | 1 | 1.5 | — | 304 | cuda:1 |
| E2_CUT_W_s2103 | sever | 172x30x33, xy (−29, −35), yaw +13.4° | tlmpm_v4 | 170280 | **TASK_SUCCESS** | 1 | 1.5 | — | 697 | cuda:1 |
| E2_OBLIQUE_W_s2201 | sever +35° | 154x34x17, xy (+18, −21), yaw −27.2° | tlmpm_v4 | 89012 | CUTTING_FAILURE | 0.5 (1 component) | — | — | 370 | cuda:1 |
| E2_OBLIQUE_W_s2202 | sever −35° | 142x28x18, xy (−7, −7), yaw +2.7° | tlmpm_v4 | 71568 | CUTTING_FAILURE | 0.5 | — | — | 288 | cuda:1 |
| E2_OBLIQUE_W_s2203 | sever +30° | 164x28x17, xy (+23, −9), yaw −38.1° | tlmpm_v4 | 78064 | CUTTING_FAILURE | 0.5 | — | — | 294 | cuda:1 |
| E2_TRIM_W_s2307 | remove (dome, ρ 45) | 142x38x13, xy (−69, −47), yaw −38.7° | tlmpm_v4 | 70485 | CUTTING_FAILURE | 0.0 | 16.6 (residual) | — | 202 | cuda:1 |
| E2_TRIM_W_s2308 | remove (gaussian, ρ 45) | 118x32x35, xy (+80, +20), yaw −52.3° | tlmpm_v4 | 132733 | CUTTING_FAILURE | 0.0 | 24.3 (residual) | — | 200 | cuda:1 |
| E2_TRIM_W_s2309 | remove (ellipsoid_cap, ρ 45) | 196x30x32, xy (−42, −64), yaw −41.9° | tlmpm_v4 | 189520 | CUTTING_FAILURE | 0.0 | 13.2 (residual) | — | 268 | cuda:0 |
| E2_TEST_A_W_s2401 | sever vertical | Test A bar, xy (+76, +24), yaw −35.3° | tlmpm_v4 | 97044 | **TASK_SUCCESS** | 1 | 1.5 | — | 409 | cuda:1 |
| E2_TEST_A_W_s2403 | sever −35° | xy (−82, −22), yaw −19.0° | tlmpm_v4 | 97044 | CUTTING_FAILURE | 0.5 | — | — | 463 | cuda:1 |
| E2_TEST_A_W_s2404 | sever, slanted advance −y 30° | xy (+6, −66), yaw −13.0° | tlmpm_v4 | 97044 | **TASK_SUCCESS** | 1 | 1.5 | — | 294 | cuda:1 |

Not in the stage-2 subset (preflight-feasible, NOT_RUN by the declared rule):
- CUT_W s2104, s2105, s2108, s2109, s2110;
- OBLIQUE_W s2204–s2210;
- TRIM_W s2310 (ρ 60°);
- TEST_A_W s2407 (ρ 45°), s2408 (−x ρ 25°), s2410 (ρ 60°).

### Maximum slots used (tlmpm_v5)

| Sample | Pattern | Digits K | S | Max occupied slots | Overflow entries |
| --- | --- | --- | --- | --- | --- |
| E2_TEST_D2_s2501 | sever:0, remove:all | 3 | 16 | 6 | 0 |
| E2_TEST_D2_s2502 | sever:20, remove:all (round-1 s1701 pattern) | 3 | 16 | **8** | 0 |
| E2_TEST_D2_s2503 | remove:p0, sever:0 | 2 | 16 | 3 | 0 |
| E2_TEST_D2_s2504 | sever:0, trim:p1 | 2 | 16 | 3 | 0 |
| E2_TEST_D2_s2505 | sever:20, remove:p0, trim:p1 | 3 | 16 | 4 | 0 |
| E2_TEST_D2_s2506 | trim:p0, sever:0, remove:p1 | 3 | 16 | 6 | 0 |
| E2_TEST_D2_s2501 (S = 8 diagnostic) | as s2501 | 3 | 8 | 6 | 0 |

Round 1 for comparison: Test C s1603 reached 8 / 8, and Test D s1701 overflowed at S = 8. In round 2 the oblique
sever-plus-skims pattern peaked at exactly 8, so S = 8 would have had no spare slot. S = 16 leaves 8.

## Failures (every one, with diagnosis)

Stage-1 failures are listed under Stage 1. The stage-2 CUTTING_FAILUREs (10) follow. Diagnostics are read-only and
change no class.

- **E2_TEST_D2_s2501** — `step1:remove_all:selected_protrusions_removed`.
  - The sever step passed: 0.770 / 0.230, own-digit location 1.5 mm, 2 prior pieces each with one substrate.
  - Skim digits 1 and 2 labelled the chips (p0: 1812 of 1817 above-plane particles on side 2). However, 2 (p0) and
    1 (p1) interior chip particles at z = H + 2.5 mm, in the +x part of the footprint, stayed unassigned. They are
    code-zero bridges: a node shared by a code-0 particle joins sides 1 and 2.
  - Relabelling just those particles (diagnostic only) would separate p0 (1814 / 1817).
  - Reproduced with S = 8 and with an S = 16 repeat.
- **E2_TEST_D2_s2502** — `step1:remove_all:selected_protrusions_removed`.
  - p0 (dome) was removed, 0.981. p1 (ellipsoid cap) is 0.000: 4 unassigned owned particles plus 13 side-1
    labels inside the chip.
  - The relabel test alone does not separate it, so the side-1 mislabels also bridge.
  - The sever step (20°) and the v2 substrate checks passed.
- **E2_TEST_D2_s2506** — `step0:trim_p0:material_above_plane_removed`, 0.000.
  - p0 (ellipsoid cap, h 5.1 mm) had 3 unassigned interior particles at z = H + 2.5 mm, x = x_p0 + 5.5 mm, which
    bridge. Relabelling them (diagnostic only) would separate 738 / 750.
  - The later sever and the remove of p1 (0.958) passed.
- **E2_OBLIQUE_W_s2201 (+35°), s2202 (−35°), s2203 (+30°), E2_TEST_A_W_s2403 (−35°)** — no separation (1
  component): `expected_substantial_pieces`, `every_plane_has_an_interface`, `cut_at_declared_planes`,
  `no_far_particle_in_another_region_piece`.
  - Every particle within ±4 mm of the plane is labelled.
  - However, 90–302 particles in the bottom 1–3 layers (z = 0.5–2.5 mm), 4–8 mm from the plane on one side, stay
    unassigned. On the 4 mm stencil a node reaches up to 4 (|n_x| + |n_z|) ≈ 5.6 mm from the plane at 35°, so these
    particles bridge the two sides.
  - At 20° the same plan leaves 0 unassigned (round 1, five runs) or 28 (D2 s2505, which separated).
  - This is the boundary of `plane_section` + 4 mm overtravel with law 8 on the 4 mm stencil. Planner extent or
    overtravel for steep obliques is a design question, not changed here.
- **E2_TRIM_W_s2307, s2308, s2309 (ρ 45°)** — `selected_protrusions_removed`, 0.000.
  - Law 8 never engaged: 0 engaged armed steps and 0 nonzero codes. The blade reaction peaked at 0.18–0.27 N.
  - No label was ever written, so ρ 45° skims cannot remove material in this law.
  - The ρ 45° and ρ 60° TEST_A_W samples (s2407, s2410) and TRIM_W s2310 (ρ 60°) were feasible but outside the
    subset, so they were not run.

## Nominations for later stages (not run here)

| Purpose | Seeds |
| --- | --- |
| Representative v2 successes | TEST_D2 s2505 (sever 20° → remove p0, keep p1 → trim p1; scoped retention), s2503 (remove before sever), s2504 (sever → trim) |
| Representative failures (law / planner boundary) | OBLIQUE_W s2201 (35° bottom-layer bridge), TRIM_W s2307 (ρ 45° no engagement), TEST_D2 s2501 (interior code-zero bridge, reproducible) |
| Workcell boundary | CUT_W s2102 (yaw +38°, xy (−16, +64) mm, PASS), TEST_A_W s2401 (xy (+76, +24) mm, yaw −35°, PASS), TEST_A_W s2408 (Franka-1 slack 0.93 mm, not run) |

## Open items

1. **Interior code-zero bridges in skims** (TEST_D2 s2501/s2502/s2506). Investigate why 1–4 chip particles stay
   unassigned or mislabelled under law 8 on these caps (none in 29 round-1 skims). This is cutting-law behaviour,
   not an evaluator issue; any fix is a new opt-in law or labelling version.
2. **Steep oblique severs (≥ 30°).** Bottom layers stay unassigned 4–8 mm from the plane. Options: oblique-aware
   overtravel or extent (planner, opt-in), or a label band that covers the stencil reach.
3. **ρ ≥ 45° skims never engage law 8.** Either declare the rake domain of law 8 (≤ 35°) or study the engagement
   gate.
4. **ρ 5° is geometrically infeasible** with 0.2 mm edge clearance (blade underside −0.25 mm).
5. **IK reach envelope.** |xy| ≳ 70 mm with yaw ≳ 40° fails chained IK from q_home; HOME can fall outside the
   local action workspace at large offsets (TRIM_W s2304).
6. **Live (Native) runs of TEST_D2 v2 successes.** The scene `slots_per_node` already reaches `LiveCuttingMaterial`
   through the configured task; this is a CPU test, and no Native run is part of round 2.

## Commits

| Commit | Content |
| --- | --- |
| `b624df6` | goal_eval_v2, `tests/test_goal_eval_v2.py`, ADR 0015 Amendment 1 |
| `ccfdd8a` | opt-in scene `solver.slots_per_node` → kinematic + live material, `tests/test_scene_slots_per_node.py` |
| `0496562` | **round-2 declaration** (`configs/wp11/generalization_round2.json`), generalization/run_generalization round-2 support, `staged_v2`, `scripts/rescore_goal_eval_v2.py`, `tests/test_generalization_round2.py` |
| `39c4e82` | `scripts/diagnose_generalization_round2.py` (read-only) |
| this commit | report, evidence |
