# WP-12 Phase 5 — generalization results and retained infrastructure history

Date: 2026-10-31 (KST). Engineering finalization branch `codex/wp12-finalize`.
The benchmark is **FROZEN** at `722468c`; the full 170-sample kinematic benchmark is complete.
ID_validated feasible success is **47/61 = 0.77 [0.65, 0.86]**, original ≥0.9 target **NOT MET**.
The frozen Native stratified sample is complete: **18/27 task success**, including **14/19 ID_validated**
(0.737 [0.512, 0.882]); its original ≥0.80 target is **NOT MET**. Execution success is not task success.
Material parameters remain provisional and uncalibrated. Kinematic results are ideal tracking, never live robot results.

Current results are in [benchmark results](#benchmark-results-kinematic-benchmark-split--2026-10-31)
and the finalization sections below. The early infrastructure/DEV descriptions in §§1–5 are preserved history;
their DRAFT, budget and NOT_RUN statements describe that earlier stage, not the current frozen benchmark.
The [freeze decision](WP12_PHASE5_FREEZE_DECISION.md) and
[frozen benchmark](../../configs/wp12/benchmark_v1.json) supersede those draft declarations.
No solver, law, goal threshold or historical result is changed by this finalization.

## 1. What exists

| Piece | File | Reuses |
| --- | --- | --- |
| Generators, feasibility classifier, execution classes, summarizer, Native sub-sampler, freeze check, quality blocks | `src/robotic_cutting_sim/generalization_wp12.py` | `generalization.py` (WP-11 Phase E): plane helpers, `compile_sample` (solver and duration rules), `staged_preflight` (`staged_v2`), `classify_material`, `episode_metrics`, `history_stride` |
| Benchmark definition (DRAFT) | `configs/wp12/benchmark_v1_DRAFT.json` | WP-11 Phase E dimension, cut-plane, sever and duration rules |
| Envelope (DECLARED, xhigh D2) | `configs/wp12/supported_envelope_v1.json` | `WP12_PHASE4_XHIGH_DECISION.md` D2: admissible bounds per rule; the notes split each range into VALIDATED / PARTIALLY_VALIDATED / NOT_RUN / KNOWN_FAILURE parts |
| Runner | `scripts/run_wp12_phase5.py` | `scripts/run_generalization.py` (episode loop, queue, GPU check), `scripts/run_wp12_workcell_live.py` (Native launch pattern) |
| CPU tests | `tests/test_wp12_generalization.py` (38 tests) | — |

### 1.1 Task families A–H (spec §5.1) on `blade_master_flat_board_v1`

Every sample is an `rcs_scene_v1` flat-board scene plus an `rcs_goal_v1` goal with `evaluator_version = goal_eval_v2a`.
The solver is `tlmpm_v5`, law 8, S = 16 and `max_cuts` = planned digits. The planner is `path_v2` with
`cut_extent = plane_section`. A variant is picked as `variants[index % len]`.

| Family | Builder | Variants (goal types) |
| --- | --- | --- |
| A single straight cut | `single_cut` | `vertical` (sever, node-plane snapped) |
| B oblique and direction-varied | `oblique_cut` | `tilt` (normal tilted about local y), `plane_yaw` (normal yawed about z), `advance_tilt` (slanted advance in a vertical plane); sever. Since D5 the DRAFT uses `tilt_slotted` (tilt on the explicit Option D slotted variant, board fixed by the variant) and `tilt_solid` (tilt on the solid board, own dimension `cut_tilt_solid_deg`) instead of `tilt`, and B has **no ID stratum** (§5) |
| C near-horizontal trim | `trim` | `skim_protrusion` (remove_protrusions), `trim_plane_protrusion` (trim_plane at z = H over the bump), `trim_layer` (trim_plane over the full length at z = H − depth; removed from the DRAFT by D5, a known failure) |
| D multiple sequential slices | `multi_slice` | `multi_plane_sever` (K planes in the declared `cut_order`), `cut_cut` (sequence[sever, sever]) |
| E multiple protrusion removal | `multi_protrusion` | `remove_all`, `remove_selected`, `bumps_sequential` (sequence of remove_protrusions [p_i]), `trim_trim` (sequence of two trim_plane) |
| F mixed sever + trim | `sever_trim` | `sever_then_trim`, `trim_then_sever` (sequences) |
| G cut + push / relocation | `cut_push` | `cut_push` (sequence[sever, move_piece kerf_face]), `trim_clear_chip` (sequence[remove [p0], move_piece surface_sweep over the chip box]) |
| H shape / pose / support | `shape_pose_support` | `sever`, `trim`. This is the only family that randomizes `board_kind` (solid or the explicit 8 mm slotted variant, one groove under the cut plane) and board dimensions. B `tilt_slotted` fixes the slotted variant by its name. |

**Board rule.** `flat_board_v1` has no oriented TLMPM fixture boxes, so the board is co-rotated with the workpiece
(`cutting_board.yaw_deg = workpiece yaw`). Its centre stays at the record value, and the workpiece z is the board top.
At about |yaw| > 3.2° a board corner passes the table edge. Such samples are `scene_geometrically_infeasible`.

### 1.2 Mixed operations (spec §5.8)

| Operation | Expressed as |
| --- | --- |
| cut → cut | D `cut_cut` |
| cut → push | G `cut_push` |
| trim → trim | E `trim_trim` |
| trim → clear chip | G `trim_clear_chip` |
| sever → trim / trim → sever | F `sever_then_trim` / `trim_then_sever` |
| bumps removed sequentially | E `bumps_sequential` |
| cut → reposition → cut | G `cut_reposition_cut` (WP-12 Phase 6, formerly NOT_EXPRESSIBLE): sever at node-snapped x1, `kerf_face` push of the +x piece by d, sever at node-snapped reference x2 = midpoint(x1, L/2) with the opt-in step field `planner_offset_local_m = [d, 0, 0]` (open loop, declared push distance). Added to the DRAFT G variants (G samples with index ≥ 2 change; dev seeds 0, 1 do not) |

The §5.8 checks map to existing machinery:
- cut history persists and earlier cuts stay valid: goal_eval_v2a per-step scopes plus its final all-digit substrate
  check;
- cut indices: the planner's digit allocation, cross-checked by `goals.step_digit_allocation`;
- robot and tool validity: preflight, workspace-clamp counts, and the Native grasp monitor.

"No accidental reconnection" (components never decrease between steps) is declared in the benchmark file, but no code
computes it yet (§4).

**Missing goal features and proposed minimal versioned additions** (`generalization_wp12.MISSING_FEATURES`; items 1
and 2 are now implemented and listed in `generalization_wp12.IMPLEMENTED_FEATURES`):

1. **`planner_offset_local_m`: IMPLEMENTED (WP-12 Phase 6)** as an opt-in field of an inline sequence step of any
   type (finite 3-vector, workpiece-local metres, norm ≤ 0.2 m). Only `goal_planner` reads it: the step is planned
   from its declared geometry and every waypoint of the step is translated rigidly by the offset (a shifted `_Frame`,
   t' = t + R·offset); plan-report rows of the step carry `planner_offset_local_m`. Every evaluator ignores it, so the
   step is scored on its declared reference-space plane. It is open loop (the declared push distance); the measured
   relocation stays ADR 0016 `wrist_depth_relocate_v1` (live only).
2. **Per-step planner options: IMPLEMENTED (WP-12 Phase 6)** as an opt-in step field `planner` holding per-operation
   options only (`goal_planner.STEP_PLANNER_OPTIONS`: rake, skew, skim speed/margin/direction, protrusion order, cut
   speed/clearance, overtravel, hold, kerf speed, saw amplitude/period, safe height, push release, cut extent), merged
   over the goal's options for that step's operations. Goal-wide options (controller profile, home, action limits,
   perception) and unknown keys are `SceneConfigError`; plan-report rows carry `step_planner_options`. The DRAFT does
   not use it: E `trim_trim` keeps one goal-wide rake, so its samples are unchanged.

   Both fields are absent unless declared. The normalized goals, compiled task ids, plan reports and waypoints of 18
   existing scene/goal pairs, the first two ID dev samples of every Phase 5 family and the goals of all 238 declared A–F/H samples are pinned
   against the pre-change tree (`tests/fixtures/wp12_goal_step_fields_pin.json`). Tests:
   `tests/test_goal_step_planner_fields.py` (validation, exact rigid translation of only the offset step at yaw 0° and
   1.5° for v1 and path_v2, a per-step ρ of 10° / 30° in `orientation_metrics`, and `evaluate_goal` output that is
   identical with and without the fields under v1 / v2 / v2a) and
   `tests/test_wp12_generalization.py::test_cut_reposition_cut_is_expressible_with_the_step_offset`.
   Kinematic DEV run (`scripts/run_wp12_cut_reposition_cut.py`; `bar_flat_board`, x1 = 0, d = 20 mm, x2 = 40 mm planned
   at 60 mm; coarse 4 mm particles, material dt 0.5 ms, Warp CPU, seed 42; development resolution only):
   `artifacts/wp12/phase6_goalfeat_001/result.json`. goal_eval_v2a verdict **PASS**, with sever1 PASS (0.50 / 0.50),
   push PASS (19.95 mm in the push window, other piece 3.5e-8 m) and sever2 PASS (0.75 / 0.25, plane statistic about
   0 at x2). The pushed side is +20.61 mm from reference at the start of cut 2; this includes about 0.65 mm of kerf
   opening by cut 1. A sub-millimetre location error is below the 4 mm grid's resolution. The episode took 84 s wall
   for 21.5 s simulated.
3. **Continuous physics overrides.** `rcs_scene_v1` selects a material only by profile id. Friction, density and
   cutting resistance can therefore vary only as the discrete `materials_wp12_v1` profiles. Proposal: an opt-in
   `workpiece.material_overrides`, validated by `materials_wp12.admit`. This is a `scene_config.py` change; the
   Phase 4c agent owns that file. TLMPM contact parameters stay solver config.
4. **Approach position along the edge line** is fixed by the planner. Only the armed start distance (`cut_clearance_m`)
   varies.

### 1.3 Randomization (spec §5.2, §5.4)

The 40 dimensions are declared as data (`dimensions`). Each has a group (scene / cutting / physics), a kind (uniform /
choice / integer) and an optional vector size (protrusion rows, slice jitters).

| Group | Dimensions |
| --- | --- |
| Scene | workpiece x/y/yaw, length/width/height, protrusion family/count/spread/x/y/radii/height, board length/width/thickness/friction, board kind |
| Cutting | cut fraction (target), cut count, cut order (+ keys), slice jitter, trim depth, cut tilt τ, plane yaw, advance tilt, rake ρ, skew ψ, skim/clear direction, cut clearance (approach position), cut/skim speed, push distance, selected protrusion |
| Physics | `material_profile`: `cutting_tasks_v1` (ID); OOD: the ten `materials_wp12_v1` profiles (material, friction, density and cutting resistance as discrete profiles) |

Board friction affects only the PhysX knife/robot–board contact (`friction_scope` of the workcell record). It has no
effect in kinematic runs, which is recorded with the dimension.

- **Independent, composable streams.** Every dimension draws from its own
  `default_rng([benchmark salt, family salt, seed, sha256(name)[:8]])`. Adding, removing or reordering one dimension
  leaves every other value unchanged (tested). A dimension that a variant does not read is still drawn and is recorded
  with `applied = false`.
- **ID / OOD.** The `range_sets.ID` and `range_sets.OOD` specs are per dimension, and families may override them.
  OOD uses `uniform_outside` / `integer_outside` shells or disjoint choice sets; validation rejects any OOD range that
  overlaps its ID range. An OOD sample takes every ID draw and replaces **exactly one** OOD dimension. The dimension is
  chosen among those its variant reads, cycled by seed index, so each OOD outcome is attributable to one dimension
  (`ood_boundary_by_dimension` in the summary).
- **Seeds and splits.** seed = family base + split offset (benchmark 0, held_out 500, dev 900) + range-set offset
  (ID 0, OOD 200) + i. Validation checks that every (family, seed) is unique over splits, range sets and families.
  The DRAFT counts per stratum are: benchmark 10, held_out 5, dev 2.
- **Strata.** There are 16 strata (8 families × ID/OOD) with 10 benchmark configurations each: 160 candidates
  (spec minimum: 6 × 10 = 60), plus 80 held-out and 32 dev.
- **Coverage.** `coverage()` checks that the benchmark seeds reach every variant and every declared OOD dimension.
  `freeze_problems` refuses a file that misses one. To meet that, each family declares at most about five OOD
  dimensions; the others stay ID-randomized (listed in `sampling.ood_coverage_rule`).

### 1.4 Feasibility classifier (spec §5.3)

`classify_feasibility(sample, envelope, compile_fn, preflight_fn)` runs before execution. It never drops a sample and
keeps every check result. Checks run in this order:

1. **Scene geometry** → `scene_geometrically_infeasible`:
   - generator checks: the plane lies inside the workpiece at z = 0 and z = H with a 2-cell margin; plane spacing is at
     least 2 cells; separately removed protrusions do not overlap; a sever misses the protrusions where declared;
   - then `scene_config` validation and compile: `SceneInfeasible`, flat-board record errors (e.g. a board corner past
     the table edge), and planner geometry errors (the WP-11 rule).
2. **Supported envelope** → `physically_unsupported`. Generic data-driven rules (`max`, `min`, `range`, `in_set`,
   `subset`, with optional `when` conditions such as board kind) apply to the sample's `envelope_parameters`:
   - rake min/max; skew, tilt, plane-yaw and advance-tilt maxima;
   - board kind, material, digits, workpiece height, protrusion families, speeds, trim depth, operation pattern.

   A **null bound is NOT_EVALUATED** and marks the sample `envelope_pending`; it is never silently supported. The
   `materials_wp12` CFL admission is also checked here (`wp12_stiff_v1` at 0.1 ms is refused).
3. **Motion preflight** (WP-11 `staged_preflight` `staged_v2`: Newton IK, C2 MuJoCo distances, flat-board blade/board
   rule):
   - an IK pose error or a joint-limit/continuity failure → `ik_unreachable`;
   - any collision/clearance check → `collision_constrained`.

   By default the preflight still runs for unsupported samples, so their reachability is recorded too.

The primary class is the first failing class in the declared precedence (default: scene → physically unsupported →
IK → collision). A config error or a preflight crash is a **classification error**: it is reported as `not_run` with
the reason and counted.

After execution, a `valid_feasible` sample becomes one of:
- `task_success`;
- `execution_failure`, with kind solver_invalid / plan_not_finished / integrity / outcome / crash / grasp;
- `not_run`: feasible but not executed. It stays in the feasible denominator. A partial smoke is always `not_run`,
  never a task result.

### 1.5 Runners, evaluation and summary

`scripts/run_wp12_phase5.py`:

- **`feasibility`** (CPU): resumable with `--resume`; existing per-sample files are kept.
- **`kinematic`**: one episode, using the goal planner's `WaypointPolicy` and goal_eval_v2a. It records:
  - the per-episode config, seed, feasibility class, draws and envelope parameters;
  - the generalization metrics;
  - **physical quality**: substrate retention, material loss, contact stability (finite state, min det F, max speed),
    penetration (particles below the board top, workspace clamps), integrity/validity. Grasp is NOT_APPLICABLE in
    kinematic runs;
  - **geometry quality** on the selected `surface_profile` (`particle_surface_v1` default, `cut_aware_surface_v1`
    selectable): mesh validity, shells vs mechanical components, and volume consistency at `basic` level; `full` adds
    the `surface_metrics_v1.surface_report` cut-face error;
  - **runtime**: reset overhead, episode wall time, sim-s per wall-s, fresh surface extraction time, process GPU memory;
  - **reliability**: reset success and recorded-output validity;
  - the failure class.
- **`queue`**: runs every valid_feasible sample once in declared order. It is resumable: an existing output is
  preserved, never rerun. At most 2 jobs per device, launched only while at most 2 other jobs are active on that GPU
  (WP-11 queue logic).
- **`native-select`**: the declared rule, `first_k_valid_feasible` with k = 3 per stratum in declared seed order, or
  `seeded_shuffle_valid_feasible`. The selection uses only feasibility classes, never outcomes. Shortfalls and every
  unselected sample's reason are recorded.
- **`native-prepare` / `native-launch [--dry-run]`**: the WP-12 Phase 2b path, `run_wp10_live.py --runtime isaac_sim
  --scene/--goal` with the workcell_live controller, live duration rule and attempt receipt. The mapping of the Native
  `result.json` to the execution classes is **provisional**: it must be confirmed against
  `run_wp12_workcell_live_analysis.py` before the freeze.
- **`summarize`**: per stratum, family, range set, overall, and per OOD dimension. It reports:
  - samples, class counts and class distribution;
  - the feasibility rate (valid_feasible / samples);
  - success among feasible (task_success / valid_feasible; not_run stays in the denominator) and success among
    executed;
  - overall success (task_success / samples);
  - execution-failure kinds;
  - Wilson 95 % score intervals with k and n;
  - the §5.7 targets as `met_point` / `met_lower_bound` (a target that is not met is reported as not met).

  The summarizer asserts that every declared sample is counted exactly once: a missing output becomes `not_run`
  ("no record"), and duplicate or undeclared rows raise.

### 1.6 What the DRAFT file declares

Families, strata, dimensions and ranges, seeds, feasibility precedence, Native rule, metric definitions (spec §5.6), the
goal-threshold snapshot (the `goals.py` defaults, checked by a test; plus the WP-11 `sever_rule` for the minimum part
fraction), §5.7 targets (kinematic ID ≥ 0.90 and Native ID ≥ 0.80 success among feasible) and the budget. The budget
estimates are: feasibility about 1.5 CPU-h; kinematic ≤ 40 GPU-h worst case, with a declared cap of 30 GPU-h and 24 h
wall; Native 48 episodes, cap 36 h.

## 2. DEV smoke (development seeds only)

All of these are **DEV** runs. They exercise the infrastructure; they are not results. Python `.venv/bin/python`
(Warp 1.17.0) and `TMPDIR=/mnt/data/ningan/tmp`. The GPUs were shared, with both at 99–100 % utilization from other
agents' jobs.

| Run | Command | Wall | Result |
| --- | --- | --- | --- |
| `artifacts/wp12/phase5_dev_001/feasibility` | `scripts/run_wp12_phase5.py feasibility --split dev --out …` (CPU, full C2 preflight) | 3:40, max RSS 652 MB | 32 / 32 classified: 28 valid_feasible, 2 ik_unreachable (B OOD plane yaw 32.9° at APPROACH; G ID `-x` chip sweep at APPROACH_PUSH), 1 collision_constrained (C ID `trim_plane_protrusion` with a `-x` skim: hand 54.5 mm into the table and pad 15.2 mm into the board at SKIM), 1 scene_geometrically_infeasible (H OOD yaw 5.65°: board corner past the table edge). Kept, but superseded: it was run on the DRAFT before the `-x` change |
| `…/phase5_dev_001/kinematic/P5_A_ID_dev_s51900` | `kinematic --steps 15 --device cuda:1` | 18.1 s | Partial → `not_run`. It exposed a runner bug: geometry quality was `ERROR` (`backend.surface` is a property). The output is kept |
| `…/phase5_dev_001/kinematic_002/P5_A_ID_dev_s51900` | same, after the fix | 19.5 s | Partial → `not_run`. Geometry MEASURED on `particle_surface_v1`: 64,410 vertices, watertight, 1 shell = 1 component, volume error −0.66 %. The 2.9 µs extraction time was a cache hit, so the timing was then changed to force a fresh extraction |
| `artifacts/wp12/phase5_dev_002/feasibility` | as dev_001, on the current DRAFT (`-x` skim/sweep moved to OOD) | 3:41, max RSS 655 MB | 32 / 32: **29 valid_feasible**; B OOD s53101 ik_unreachable (plane yaw 32.9°), G OOD s58101 ik_unreachable (`clear_direction -x`, APPROACH_PUSH), H OOD s59101 scene_geometrically_infeasible (yaw 5.65°). Every ID dev sample is valid_feasible. `envelope_pending` is true for all (TEMPLATE) |
| `…/phase5_dev_002/kinematic/P5_A_ID_dev_s51900` | `kinematic --steps 15 --surface-profile cut_aware_surface_v1 --device cuda:1` | 17.9 s | Partial → `not_run`. Cut-aware geometry MEASURED: 15,894 vertices, watertight, shells = components, volume error −0.034 %, fresh extraction 0.21 s. Throughput 0.022 sim-s per wall-s on the loaded GPU (about 450 s for a 10 s episode), 79,016 particles |
| `…/phase5_dev_002/summary_dev.json` | `summarize --split dev …` | < 1 s | 32 samples counted once; feasibility 29/32 = 0.906 [0.758, 0.968]; success among feasible 0/29 [0, 0.117] (nothing complete was run: the one partial smoke is not_run) |
| `…/phase5_dev_002/native_selection_dev.json` | `native-select --split dev …` | < 1 s | The rule runs. Dev strata have only 2 seeds, so every stratum records a shortfall with its reason (B/G/H OOD: 1) |
| `…/phase5_dev_002/native_dryrun_A` | `native-launch --dry-run --sample-id P5_A_ID_dev_s51900` | < 10 s | Native compile `CFG-P5_A_ID_dev_s51900-…-1ca5598c`: plan end 12.1 s, duration 14.5 s, 725 steps, stride 19. Command recorded; the launch was **NOT_RUN** (dry run: no formal run, both GPUs busy) |

**One DEV finding changed the DRAFT (allowed: dev iteration, before any freeze).** In dev_001 both infeasible ID
samples were `-x` motions: a `-x` skim (C) and a `-x` chip sweep (G). `-x` is therefore not in the confirmed-valid
region of this workcell. `skim_direction` and `clear_direction` are now ID `+x` and OOD `-x`, and they were added to
the C and G OOD dimensions. dev_002 confirms it: `-x` fails again in G OOD; every ID sample is feasible.

Not run: any complete kinematic episode, any Native episode, and any benchmark or held-out seed.

CPU tests: `.venv/bin/python -m pytest tests/test_wp12_generalization.py` → **31 passed** (4.8 s). Together with
`test_generalization_round1/2`, `test_repo_naming` and `test_wp12_flat_board_workcell`: 90 passed. These cover:
- seed reproducibility and id round trips;
- independent dimension streams;
- range adherence for all 272 declared samples;
- ID/OOD separation and overlap rejection;
- held-out / dev / benchmark disjointness and overlap rejection;
- ≥ 10 configurations per stratum and the Native k ≥ 3 rule;
- coverage;
- board co-rotation and on-board z; slotted variant only in H, groove under the plane; node-plane snapping;
- mixed-operation mapping and NotExpressible;
- one real compile per family;
- the classifier, with a synthetic case per class, precedence, envelope pending, the material CFL refusal, a real
  board-off-table case and classification errors;
- envelope rule kinds and freeze problems;
- execution classes, Wilson intervals, summary denominators (not_run in the feasible denominator, missing records
  added, duplicate / unknown / undeclared rows refused), the per-OOD-dimension boundary, and the Native selection rule.

## 3. What waits for the Phase 4 envelope (and before freezing)

1. ~~Fill `supported_envelope_v1.json`~~ **Done** (xhigh decision D2, 2026-10-31).
2. ~~Re-declare the ID ranges inside that envelope~~ **Done** per D5 (§5). `freeze_problems` returns no problem for the
   re-declared DRAFT and the declared envelope.
2a. **Run E2** ([`phase4_e2_protocol_v1.json`](../../configs/wp12/phase4_e2_protocol_v1.json), committed before any
   formal row). Record the outcome as a new envelope version (v1 untouched). Then re-declare: only cells whose E2 rows
   pass may move into ID (B slotted θ ≤ 20°, φ ≤ 15°, β ≤ 15°; speeds 8–12 / 20–30 mm/s; C `trim_plane_protrusion`,
   E `trim_trim`, G `cut_push`).
3. **Confirm the Native result mapping** (`native_execution`) against the Phase 2b analysis, including grasp failures.
4. **Decide the budget options** (`budget.if_over_budget`) before the freeze, never after results.
5. **Freeze**: copy to `configs/wp12_benchmark_v1.json` with status FROZEN and a non-DRAFT id. `freeze_problems` must be
   empty. Commit, then run benchmark → held-out once.

## 4. Known limitations

- **Board friction** has no kinematic effect (PhysX only).
- **Continuous physics, contact parameters and edge-line approach position** are not expressible (§1.2). Cut →
  reposition → cut has been expressible since WP-12 Phase 6, open loop only.
- **Workpiece yaw** is limited to about ±3.2° by the co-rotated board and the table edge. That is a workcell boundary,
  measured as scene-infeasible.
- **"No accidental reconnection" between steps** is declared but not yet computed by the runner. The final-state
  goal_eval_v2a checks cover persistence at the end only.
- **Multi-episode stability** (repeated resets in one process) is Phase 6. The declared reproducibility repeats
  (first valid_feasible ID sample per family, rerun on the other device) need a `--repeat` flag; it is not yet built.
- **Dependency.** The module depends on the Phase 4c files that are still uncommitted (`materials_wp12.py`,
  `configs/materials/materials_wp12_v1.json`, and the `scene_config.py` changes that import `geometry_providers.py`).
  The DRAFT's OOD material ids, and the CFL test, need them. Commit those before or together with these files.

## 5. Re-declaration per the xhigh decision D5 (2026-10-31)

Source: [WP12_PHASE4_XHIGH_DECISION.md](WP12_PHASE4_XHIGH_DECISION.md) D5, with D1 and D2. The envelope is
[`supported_envelope_v1.json`](../../configs/wp12/supported_envelope_v1.json) (DECLARED). The file stays a **DRAFT**.
The change is recorded in its `redeclaration_d5` block.

**Rule.** ID uses only VALIDATED cells of the envelope: flat board, `path_v2`, `cutting_tasks_v1`, 1 mm particles,
4 mm grid, 0.1 ms. "Inside a bound" means admissible for execution, not validated. A family or variant without a
validated cell is handled in one of two ways:
- it is absent from ID (B);
- or it is declared **ID-unconfirmed**: run as declared and reported in full, but outside the §5.7 targets.

### 5.1 Changes

| Item | Before | Now (D5) |
| --- | --- | --- |
| Cut / skim speed (ID) | 8–12 / 20–30 mm/s | fixed at **10 / 25 mm/s**, the only validated speeds. OOD shells 4–30 / 10–60 mm/s around them. 8–12 and 20–30 mm/s are E2 rows |
| Family B | ID + OOD; `tilt` ±20° on the solid board (2 of 10 ID benchmark samples above the 9.9° solid-board bound) | **OOD only.** `tilt_slotted`: explicit Option D slotted variant, OOD θ (20, 34.9]°. `tilt_solid`: new OOD dimension, 0 < \|θ\| ≤ 9.9° on the solid board. `plane_yaw`: OOD (15, 35]°. `advance_tilt`: OOD (15, 30]°. The B ID ranges kept in the file (slotted θ ≤ 20°, φ ≤ 15°, β ≤ 15°) are post-E2 candidates and the base draws of the OOD samples; they are no ID claim |
| Family C | `skim_protrusion`, `trim_plane_protrusion`, `trim_layer`; OOD includes trim depth | `trim_layer` removed (known failure: the continuous chip never separates). The trim-depth dimension is removed from C, and skim speed is added as a C OOD dimension. `trim_plane_protrusion` is ID-unconfirmed until E2 |
| Targets (§5.7) | over all ID samples | over population **ID_validated**. `target_scope.id_unconfirmed`: F (all), G (all), C `trim_plane_protrusion`, E `trim_trim`. These are reported separately (`by_id_population`) with their own Wilson intervals |
| Materials (OOD) | ten profiles | same list. Notes: G_c and friction variants are force-only / weak-effect variants, never counted as generalization of separation. `wp12_stiff_v1` stays and is classified physically_unsupported |
| Demos | — | `phase7_demos`: Demo 2 = slotted 20° with the planner's 4 mm overtravel (E2 + Native first). Demo 6 = cut → push → trim in one episode (E2 + Native first). Demo 7 = one ID sample of a validated stratum, chosen by the Native rule after the freeze |
| Strata | 16 (160 benchmark candidates) | **15** (150; held-out 75, dev 30) |

ID_validated cells are: A `vertical`; C `skim_protrusion`; D `multi_plane_sever`, `cut_cut`; E `remove_all`,
`remove_selected`, `bumps_sequential`; H `sever`, `trim`. B has no ID samples.

### 5.2 Code changes

All in `generalization_wp12.py` and the Phase 5 runner. Physics, goals, thresholds and the planner are unchanged.

- **`freeze_problems` now evaluates rules with a `when` condition.** Finding 2 of the decision: before, it skipped them.
  - A conditioned rule is checked against a family's ID range when at least one declared ID sample of the family,
    whose variant reads that dimension, satisfies the condition. A family without an ID stratum is judged on probe
    builds of its ID seeds.
  - New: every declared ID sample of every split is evaluated against the envelope. An UNSUPPORTED or PENDING sample is
    a freeze problem.
  - On the pre-D5 DRAFT (kept as `tests/fixtures/wp12_benchmark_v1_DRAFT_pre_d5.json`) it now flags:
    - the B tilt range against `sever_tilt_solid_board`;
    - exactly the two B ID benchmark samples above 9.9° (`P5_B_ID_benchmark_s52006`, `s52009`), plus `held_out` /
      `dev` ones;
    - C `trim_depth_m` and the three C `trim_layer` benchmark samples.

    On the re-declared DRAFT it returns no problem.
- **New `oblique_cut` variants `tilt_slotted` and `tilt_solid`** (`VARIANT_BOARD_KIND`). `tilt` is kept unchanged for
  older files.
- **`target_scope` and `id_population()`**; `summarize(..., population=)` adds `by_id_population` and computes a target
  that declares `population` over that population only.
- **The Phase 5 runner's** `declared()` records each declared sample's variant. Its kinematic episode is factored into
  `kinematic_episode()`, which E2 reuses; behaviour is unchanged.
- **`tests/test_goal_step_planner_fields.py`** builds its phase5 pin from the pre-D5 snapshot. The pin keeps testing the
  opt-in step fields, not the DRAFT content.

### 5.3 DEV feasibility on the re-declared DRAFT (`artifacts/wp12/phase5_dev_003`)

```
TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase5.py feasibility --split dev \
    --out artifacts/wp12/phase5_dev_003/feasibility                      # CPU, 3:21 wall, max RSS 654 MB
.venv/bin/python scripts/run_wp12_phase5.py summarize --split dev \
    --feasibility artifacts/wp12/phase5_dev_003/feasibility --out artifacts/wp12/phase5_dev_003/summary_dev.json
```

DEV only, dev seeds, development check of the re-declaration (not a result). 30 / 30 samples classified,
**26 valid_feasible**. Every ID dev sample is valid_feasible and SUPPORTED by the envelope (`envelope_pending` false
everywhere). The four infeasible samples are all OOD:

| Sample | Variant | Class | Reason |
| --- | --- | --- | --- |
| `P5_B_OOD_dev_s53100` | `tilt_slotted`, θ 32.1° | collision_constrained | knife handle 0.68 mm into the board at HOLD (the planner's 4 mm overtravel). This matches E1: the groove-limit overtravel is unreachable at 20–35°. E2 measures the handle slack at 20° and 25° |
| `P5_C_OOD_dev_s54101` | `trim_plane_protrusion`, ρ 47.25° | physically_unsupported | `trim_rake_deg` (ρ > 44°) |
| `P5_G_OOD_dev_s58101` | `trim_clear_chip`, −x sweep | ik_unreachable | APPROACH_PUSH (as in dev_002) |
| `P5_H_OOD_dev_s59101` | `trim`, yaw beyond 3.2° | scene_geometrically_infeasible | board corner past the table edge (as in dev_002) |

The summary splits the 14 ID dev samples into 9 ID_validated and 5 ID_unconfirmed. The kinematic target is computed
over the 9 (0 / 9: nothing was executed, all `not_run`).

CPU tests: `tests/test_wp12_generalization.py` **38 passed**. With `test_goal_step_planner_fields`, `test_repo_naming`,
`test_generalization_round1/2` and `test_wp12_flat_board_workcell`: 139 passed.

## Benchmark results (kinematic, benchmark split) — 2026-10-31

Frozen `configs/wp12/benchmark_v1.json` (`722468c`) with envelope v2; tooling `177faae`; kinematic, law 8 /
`tlmpm_v5`, 1 mm / 4 mm / 0.1 ms, both GPUs shared; 170 samples, 16.28 job-hours, 0 not_run, 0 missing records.
Evidence: [`phase5_benchmark_kinematic_001.json`](../evidence/wp12/phase5_benchmark_kinematic_001.json).
Run outputs: `artifacts/wp12/phase5_001/`. Wilson 95 % intervals; every declared sample counted once.

| Group | Feasible | Success among feasible | 95 % CI |
| --- | --- | --- | --- |
| All 170 | 151/170 | 93/151 (overall 93/170) | [0.54, 0.69] |
| ID (all) | 77/80 | 53/77 | [0.58, 0.78] |
| **ID_validated (spec §5.7 target population)** | 61/64 | **47/61 = 0.77** | **[0.65, 0.86]** |
| ID_unconfirmed | 16/16 | 6/16 | [0.18, 0.61] |
| OOD | 74/90 | 40/74 | [0.43, 0.65] |

Per family (ID / OOD, success among feasible): A 10/10, 9/9; B 10/10, 5/7; BF (OOD only, solid-board tilt
0–9.9°) 2/10; C 8/9, 5/5; **D 0/10, 0/10**; E 5/8, 3/10; F 6/10, 4/7; G 4/10, 4/8; H 10/10, 8/8.
Infeasible classes: 5 scene-infeasible, 3 IK-unreachable, 7 collision-constrained, 4 physically unsupported.
All 58 execution failures are outcome failures (no solver, integrity, crash or plan failures).

- **§5.7 kinematic target (≥ 0.9 among feasible ID): NOT MET** (point 0.77, lower bound 0.65). Not redefined.
- **Family D (multiple sequential slices) fails every sample in ID and OOD.** Read-only finalization diagnosis below confirms later cuts miss their declared reference planes after the
  workpiece slides. All 20 verdicts remain FAIL; separation alone does not satisfy the task.
- Native sub-sample and the minimal held-out/Demo 7 execution are reported below. Reproducibility repeats and
  the remaining held-out matrix are NOT_RUN under the authorized finalization scope.

## Native benchmark sub-sample — completed 2026-10-31

[Selected-cohort evidence](../evidence/wp12/finalize_native_cohort_001.json) ·
[whole-benchmark coverage](../evidence/wp12/phase5_native_001.json) ·
[frozen pre-outcome selection](../evidence/wp12/phase5_native_selection_001.json).
The independent existing driver finished all 27 selected episodes without restart or interruption.
The selection was the first three **feasible** seeds in each of A–H ID plus A OOD, using feasibility alone,
never kinematic outcomes. These are **24 ID + 3 A OOD**, not a random Native test of all 170 configurations.
All 27 use live Isaac Sim 6.2, default `particle_surface_v1`, unchanged law 8 / `tlmpm_v5` and material seed 42.

| Selected Native population | Task success / selected feasible | Wilson 95% interval | Kinematic success, same IDs |
| --- | --- | --- | --- |
| All selected | 18/27 = 0.667 | [0.478, 0.814] | 19/27 |
| All ID | 15/24 = 0.625 | [0.427, 0.788] | 16/24 |
| **ID_validated (original Native target population)** | **14/19 = 0.737** | **[0.512, 0.882]** | 14/19 |
| ID_unconfirmed | 1/5 = 0.200 | [0.036, 0.624] | 2/5 |
| A OOD only | 3/3 = 1.000 | [0.439, 1.000] | 3/3 |

Per ID family A/B/C/D/E/F/G/H: **3/3, 3/3, 2/3, 0/3, 2/3, 1/3, 1/3, 3/3**.
The original Native ≥0.80 target is **NOT MET**, by either point estimate or lower bound. These small,
stratified denominators do not support a full-population Native success estimate or broad OOD claim.
In the whole-benchmark coverage report, **124 unselected feasible rows are NOT_RUN** and 19 rows are infeasible;
its full denominators must not be confused with the selected-cohort rates above.

All nine task failures are outcome failures, with complete finite recorded state; none is a solver, integrity,
grasp, IK, clamp, crash or unfinished-plan failure. Native and kinematic success labels agree on **26/27**:
18 both succeed, eight both fail, one F sample succeeds only kinematically. This is outcome-label agreement,
not equality of trajectories or material states. All 27 return HOME; maximum task slip is 11.93 µm,
rotation 0.522 mrad; no task-phase IK-not-ok or workspace-clamp episode. Sum of attempt wall times is
**10006.43 s (2.78 h)**, range 225.79–675.74 s; the driver log's 2.85 h includes overhead.
Classification used the reviewed complete-receipt guard; it did not change any frozen numerical gate.

Commands (read-only summaries; no new simulation):

```bash
.venv/bin/python scripts/run_wp12_phase5.py summarize --split benchmark --feasibility artifacts/wp12/phase5_001/feasibility_benchmark --kinematic-root artifacts/wp12/phase5_001/kinematic_benchmark --native-root artifacts/wp12/phase5_native_001 --out docs/evidence/wp12/phase5_native_001.json
.venv/bin/python scripts/report_wp12_native.py --out docs/evidence/wp12/finalize_native_cohort_001.json
```

## Finalization: Family D diagnosis (read-only, no new episode)

[Diagnostic evidence](../evidence/wp12/finalize_family_d_001.json), generated by
`scripts/diagnose_wp12_family_d.py --out docs/evidence/wp12/finalize_family_d_001.json`.
The script reads only saved result/manifest/action/particle-history files and records their SHA256 identities.
The census covers all 20 D failures: every leaf failure is `cut_at_declared_planes`; in sequence variants the
parent failure is `step_1_cut1`. Integrity passes. The two representatives are the first ID and first OOD
sample in declared seed order, not selected by their outcome. No diagnostic reproduction was necessary.

| Saved representative | Actual separation | Near-plane x displacement before later armed strokes | Frozen location statistic, limit 2 mm |
| --- | --- | --- | --- |
| ID seed 54000, two planes in descending order | 3/3 expected pieces; mass retained; no debris | second cut: median −1.191 mm, P05/P95 −1.200/−1.182 mm | first 1.5 mm PASS; second 4.0 mm FAIL |
| OOD seed 54200, four ascending cuts | 5/5 expected pieces; mass retained; no debris | cuts 2/3/4: median +1.195/+2.189/+3.344 mm | 1.5/4.0/4.5/5.0 mm; later three FAIL |

The saved kinematic edge commands stay on the declared world-space planes (maximum transformed x error <2e-17 m),
while the material moves between strokes. Final opposing-digit reference-neighbour interfaces shift by +1 mm
on ID cut 2 and −1/−2/−3 mm on OOD cuts 2–4. These link midpoints are a diagnostic, distinct from the evaluator's
full eight-node compatibility graph and its frozen maximum midpoint statistic. Together, the actions, motion
history, cut digits and original evaluation support **open-loop cuts on a sliding workpiece** as the failure cause,
not lack of separation, an IK error, a scene/goal identity mismatch or a solver crash in these representatives.
The material displacement is sampled at the last recorded frame before each armed stroke (41-frame histories),
not measured continuously. The 20-result census supports the shared failed criterion; detailed motion attribution
is scoped to these two representatives.

Future work may use wrist-depth relocation or replanning between strokes. No controller is added, no benchmark
is rescored, and the 0/20 result and ≥90% NOT MET judgment remain unchanged.

## Finalization: minimal held-out / Demo 7 protocol

Only 40 held-out ID configurations were classified (CPU motion preflight, 268.50 s):
38 feasible, 2 collision-constrained; the 45 OOD candidates were NOT_RUN (scope reduced).
The frozen draw requires every ID feasibility row and reads no task outcomes. Its validated feasible pool has
30 candidates. `default_rng([1205, 7])` selected **P5_D_ID_held_out_s54501**, Family D `cut_cut`, seed **54501**.
[Selection receipt](../evidence/wp12/finalize_demo7_selection_001.json),
[exact scene](../../configs/wp12/finalization/demo7_scene.json),
[goal](../../configs/wp12/finalization/demo7_goal.json) and
[execution scope](../../configs/wp12/finalization/demo7_execution_v1.json) were committed at **ab1ec8f**
before any held-out outcome. Material simulation seed remains 42; sample seed is the scene/task draw.
Standard compilation deterministically sets `max_cuts` from raw 1 to the two planned digits and replaces the
raw 120 s feasibility-probe duration with the execution duration (kinematic 14.5 s, Native 20.5 s).
The selected geometry, two declared goal planes and every goal threshold are unchanged; this is the existing
wrapper contract, not an outcome-dependent config edit.

The one selected kinematic episode **FAILS** its outcome (process exit 0, 150.66 s wall; episode loop 142.88 s,
14.5 s simulated, cuda:1). Integrity PASS; cut 1 PASS at 1.5 mm, cut 2 FAIL at **4.0 mm versus 2.0 mm**.
Both scoped sever operations separate; the plan finishes, no material is lost and there are no workspace clamps.
[New result, manifest, exact command and hashes](../evidence/wp12/finalize_demo7_kinematic_001.json).
Run: `artifacts/wp12/finalize_001/kinematic_held_out/P5_D_ID_held_out_s54501`.
No alternate is run: the draw is retained as a failed held-out example, separate from the 170-sample benchmark.
The same frozen sample then ran exactly once in Native Isaac Sim on GPU 0, after the existing driver completed.
It also **FAILS** (`CUTTING_FAILURE`, outcome `step1:cut1:cut_at_declared_planes`): first cut 1.5 mm PASS,
second cut **4.0 mm > 2.0 mm**. Integrity PASS, finite state, complete receipts, controller DONE, HOME PASS,
0 IK-not-ok/workspace clamps; task slip 5.29 µm, rotation 0.493 mrad. Process wall **344.43 s**, 20.5 s
declared task duration (the existing Native wrapper allows physical tracking; geometry/planes/thresholds unchanged).
[Native result, exact command, manifest and hashes](../evidence/wp12/finalize_demo7_native_001.json).
Run: `artifacts/wp12/finalize_001/native_held_out/P5_D_ID_held_out_s54501`.
Camera recording/media was **NOT_RUN** for this benchmark-style attempt; no cosmetic rerun was made.
The second Native surface profile and the full 85-episode held-out matrix are **NOT_RUN (scope reduced)**.
A failure remains evidence; no manual alternative is selected to improve presentation.
