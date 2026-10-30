# WP-12 Phase 5 — generalization benchmark infrastructure (DRAFT)

Date: 2026-10-31 (KST). Branch `claude/wp12-physical-cut-surfaces` (HEAD `a348b25` plus an uncommitted tree shared with
other agents). Spec: [WP12_SPECIFICATION.md](../WP12_SPECIFICATION.md) Phase 5 (§5.1–5.8).

> **Status: DRAFT infrastructure. No formal benchmark result exists.** The benchmark file
> [`configs/wp12/benchmark_v1_DRAFT.json`](../../configs/wp12/benchmark_v1_DRAFT.json) is **not frozen**. The supported
> envelope [`configs/wp12/supported_envelope_v1.json`](../../configs/wp12/supported_envelope_v1.json) is a **TEMPLATE**:
> it has fields and no bounds. The orchestrator fills the bounds after the Phase 4 xhigh envelope decision. Every run
> below is **DEV**: development seeds only, under `artifacts/wp12/phase5_dev_NNN`. No benchmark or held-out seed was
> preflighted or run. Kinematic results are **kinematic ideal tracking**, never live results. Material parameters are
> provisional and uncalibrated.

Nothing in `tlmpm*.py`, law 8, any goal threshold, `goal_eval_v1/v2/v2a`, `blade_master_flat_board_v1`, `final_v4` or
the WP-11 records was changed. The benchmark builds every task from scene and goal configs using the existing goal types.
The solver has no task-specific branch.

## 1. What exists

| Piece | File | Reuses |
| --- | --- | --- |
| Generators, feasibility classifier, execution classes, summarizer, Native sub-sampler, freeze check, quality blocks | `src/robotic_cutting_sim/generalization_wp12.py` | `generalization.py` (WP-11 Phase E): plane helpers, `compile_sample` (solver and duration rules), `staged_preflight` (`staged_v2`), `classify_material`, `episode_metrics`, `history_stride` |
| Benchmark definition (DRAFT) | `configs/wp12/benchmark_v1_DRAFT.json` | WP-11 Phase E dimension, cut-plane, sever and duration rules |
| Envelope (TEMPLATE) | `configs/wp12/supported_envelope_v1.json` | Pending statements of `WP12_CUTTING_PHYSICS.md` C6 and E1, as notes |
| Runner | `scripts/run_wp12_phase5.py` | `scripts/run_generalization.py` (episode loop, queue, GPU check), `scripts/run_wp12_workcell_live.py` (Native launch pattern) |
| CPU tests | `tests/test_wp12_generalization.py` (31 tests) | — |

### 1.1 Task families A–H (spec §5.1) on `blade_master_flat_board_v1`

Every sample is an `rcs_scene_v1` flat-board scene plus an `rcs_goal_v1` goal with `evaluator_version = goal_eval_v2a`.
The solver is `tlmpm_v5`, law 8, S = 16 and `max_cuts` = planned digits. The planner is `path_v2` with
`cut_extent = plane_section`. A variant is picked as `variants[index % len]`.

| Family | Builder | Variants (goal types) |
| --- | --- | --- |
| A single straight cut | `single_cut` | `vertical` (sever, node-plane snapped) |
| B oblique and direction-varied | `oblique_cut` | `tilt` (normal tilted about local y), `plane_yaw` (normal yawed about z), `advance_tilt` (slanted advance in a vertical plane); sever |
| C near-horizontal trim | `trim` | `skim_protrusion` (remove_protrusions), `trim_plane_protrusion` (trim_plane at z = H over the bump), `trim_layer` (trim_plane over the full length at z = H − depth) |
| D multiple sequential slices | `multi_slice` | `multi_plane_sever` (K planes in the declared `cut_order`), `cut_cut` (sequence[sever, sever]) |
| E multiple protrusion removal | `multi_protrusion` | `remove_all`, `remove_selected`, `bumps_sequential` (sequence of remove_protrusions [p_i]), `trim_trim` (sequence of two trim_plane) |
| F mixed sever + trim | `sever_trim` | `sever_then_trim`, `trim_then_sever` (sequences) |
| G cut + push / relocation | `cut_push` | `cut_push` (sequence[sever, move_piece kerf_face]), `trim_clear_chip` (sequence[remove [p0], move_piece surface_sweep over the chip box]) |
| H shape / pose / support | `shape_pose_support` | `sever`, `trim`. This is the only family that declares `board_kind` (solid or the explicit 8 mm slotted variant, one groove under the cut plane) and board dimensions. |

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
| **cut → reposition → cut** | **NOT_EXPRESSIBLE.** The builder raises `NotExpressible`; this is a missing feature, never a sample class |

The §5.8 checks map to existing machinery:
- cut history persists and earlier cuts stay valid: goal_eval_v2a per-step scopes plus its final all-digit substrate
  check;
- cut indices: the planner's digit allocation, cross-checked by `goals.step_digit_allocation`;
- robot and tool validity: preflight, workspace-clamp counts, and the Native grasp monitor.

"No accidental reconnection" (components never decrease between steps) is declared in the benchmark file, but no code
computes it yet (§4).

**Missing goal features and proposed minimal versioned additions** (`generalization_wp12.MISSING_FEATURES`):

1. **`planner_offset_local_m`** (needed for cut → reposition → cut). A sever plane is both the planner target and the
   evaluator's reference-space plane. The second cut must be planned where the pushed piece now is. Proposal: an opt-in
   sever/divide step field `planner_offset_local_m`, which only `goal_planner` reads (added to the plane origin) and
   every evaluator ignores. Alternative: an opt-in planner option `follow_declared_moves`, which offsets later steps by
   the earlier `move_piece` `distance_m × direction_local`. Both are open loop. The measured relocation is ADR 0016
   `wrist_depth_relocate_v1` (live only). This needs `goals.py` (`_OPT_IN_FIELDS`) and `goal_planner.py`; they are not
   edited here.
2. **Per-step planner options.** Rake, skew, skim direction and speeds are goal-wide, so trim → trim cannot use two
   rakes. Proposal: an opt-in step field `planner`, merged over the goal's block for that step only.
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

1. **Fill `supported_envelope_v1.json`** (status `DECLARED`, every bound, evidence references) after the Phase 4 xhigh
   decision. Inputs: the ρ envelope (C6: ρ ≤ 44° declared for law 8), the oblique-on-flat-board E1 decision (solid and
   slotted), material admission and support (Phase 4c), the resolution study (workpiece height), the digit limit, and
   the supported operation patterns (class (c) residual chips in sequential cuts).
2. **Re-declare the ID ranges inside that envelope** (`freeze_problems` flags every ID range that reaches outside a
   mapped rule; e.g. a rake ceiling below 35° flags C/E/F/G/H). Revisit the B tilt ID range (±20°) against the E1
   result.
3. **Confirm the Native result mapping** (`native_execution`) against the Phase 2b analysis, including grasp failures.
4. **Decide the budget options** (`budget.if_over_budget`) before the freeze, never after results.
5. **Freeze**: copy to `configs/wp12_benchmark_v1.json` with status FROZEN and a non-DRAFT id. `freeze_problems` must be
   empty. Commit, then run benchmark → held-out once.

## 4. Known limitations

- **Board friction** has no kinematic effect (PhysX only).
- **Continuous physics, contact parameters, edge-line approach position and cut → reposition → cut** are not
  expressible (§1.2).
- **Workpiece yaw** is limited to about ±3.2° by the co-rotated board and the table edge. That is a workcell boundary,
  measured as scene-infeasible.
- **"No accidental reconnection" between steps** is declared but not yet computed by the runner. The final-state
  goal_eval_v2a checks cover persistence at the end only.
- **Multi-episode stability** (repeated resets in one process) is Phase 6. The declared reproducibility repeats
  (first valid_feasible ID sample per family, rerun on the other device) need a `--repeat` flag; it is not yet built.
- **Dependency.** The module depends on the Phase 4c files that are still uncommitted (`materials_wp12.py`,
  `configs/materials/materials_wp12_v1.json`, and the `scene_config.py` changes that import `geometry_providers.py`).
  The DRAFT's OOD material ids, and the CFL test, need them. Commit those before or together with these files.
