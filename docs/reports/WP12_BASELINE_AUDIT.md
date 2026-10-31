# WP-12 Phase 0.1–0.3 — repository audit and frozen baseline

Date: 2026-10-30 (KST). Branch `claude/wp12-physical-cut-surfaces`. Spec: [WP12_SPECIFICATION.md](../WP12_SPECIFICATION.md)
§Phase 0. Compact evidence: [`docs/evidence/wp12/phase0_baseline_001.json`](../evidence/wp12/phase0_baseline_001.json).
Run outputs (git-ignored): `artifacts/wp12/baseline/`, logs `artifacts/wp12/logs/`.

Scope: every baseline scene below is a **kinematic ideal-tracking** run (no robot, IK, grasp, slip, dynamics or
cameras) of the retained TLMPM material, which is provisional and uncalibrated. Native evidence quoted here is
historical; no Native run was made in this step.

## Result

| Item | Status | Note |
| --- | --- | --- |
| 0.1 file audit | PASS | all 23 listed files exist; differences in §2 |
| 0.2 capability confirmation | PASS (CPU + kinematic); Native NOT_RUN (historical evidence only) | §3 |
| 0.3 freeze: versions, GPU, CI | PASS | §1; CI on this branch was `in_progress` at freeze time |
| 0.3 freeze: full CPU suite | PASS (shared tree: 1196 passed, 5 xfailed, 0 failed, 0 skipped); clean worktree: 1 FAIL, path-dependent (see §1) | §1 |
| Baseline scenes 1–8 (kinematic GPU rerun) | **8/8 reproduce the frozen WP-11 verdict** | §7; frozen Phase E `physics_state.npz` files were found deleted, so scenes 3–8 compare at verdict/metric level |
| Current geometry metrics (scenes 1–3, plus 4–8) | MEASURED (draft method `surface_metrics_v1`, 15 CPU tests PASS) | §8 |
| Existing failure list | PASS | §9, 20 entries |
| Native rerun | NOT_RUN | not in scope of this step |


## 1. Freeze (0.3)

- Commit: audit started at `ce547f90c17d544d34af811d412d1bfcaad960d0`. The baseline runs ran at
  `1c9160546e923e1c782719497eaca52cba4c92ad` (recorded `git_dirty = true` in every manifest because of this step's
  untracked tools). `1c91605` changes only `.claude/agents/rcs-impl.md`, `docs/PROGRESS.md`, `docs/WP12_PLAN.md`
  relative to `ce547f9`, so the physics source is identical. main is `fbe99bd`.
- Working tree: other agents edited the shared worktree during this step (from 23:32 KST: `scene_config.py`,
  `isaac_sim_runner.py`, `motion_preflight.py`, `scripts/preflight_wp11_orientations.py` modified; new
  `flat_board_workcell.py`, `configs/wp12_workcells_v1.json`, …; commits `3f13379`, `dd1538b`). The baseline runs
  imported their modules at 23:25, before those edits, and the kinematic runners do not import
  `isaac_sim_runner.py`. Later (after the runs) other agents also modified `kinematic_env.py`, `live_material.py`
  and `blade_probe.py`; `scripts/wp12_baseline_capture.py` patches `LiveCuttingMaterial.advance` and
  `KinematicCuttingBackend.close`, so it must be rechecked against those changes before reuse. Because of the shared
  tree, the CPU suite was also run in a clean detached worktree (below).
- Runtimes:

  | Runtime | Python | Key packages |
  | --- | --- | --- |
  | `.venv` (CPU tests, kinematic) | 3.12.12 | warp-lang 1.17.0, newton 1.6.1, numpy 2.5.4, scipy 1.18.1, mujoco 3.12.0, usd-core 26.8, pytest 9.1.1 |
  | `.deps/isaacsim-venv` (Native) | 3.12.12 | isaacsim 6.2.0.0, warp-lang 1.17.0, newton 1.6.1, torch 2.11.0+cu128, numpy 2.3.1 |
  | `.deps/isaaclab-venv` (Lab) | 3.12.12 | isaaclab 28.0.0, warp-lang 1.17.0, newton 1.6.1, torch 2.11.0+cu128 |

  Warp 1.17.0 with CUDA Toolkit 12.9, driver 570.211.01 (CUDA 12.8). Ubuntu 22.04.5, kernel 6.8.0-138, 64 CPUs.
- GPUs: 2 × RTX 5090 (32 GB). Scenes 1, 2, 3, 6 ran on cuda:1 and 4, 5, 7, 8 on cuda:0, four concurrent jobs per
  GPU (plus the CPU suite), so wall times are not comparable with the frozen runs (which ran with at most two jobs per GPU). GPU 0 held two idle processes of
  another user; GPU 1 is shared with another project. No foreign process was touched.
- CI (`gh run list -L 5` at 23:37 KST): the three newest runs on this branch (`41071099677`, `41070755704`,
  `41070509279`) were `in_progress`; the last completed runs were green: main `41017393752` (46 min) and
  `claude/wp11-cutting-refinement` `41017385564`.
- CPU suite: 
  - Shared worktree (`.venv/bin/python -m pytest -q`, started 23:20 KST at `ce547f9`/`1c91605`, collected before the
    other agents' edits): **1196 passed, 5 xfailed, 0 failed, 0 skipped, 1508.8 s** (wall 25:12, peak RSS 0.84 GB),
    exit 0. Log `artifacts/wp12/logs/phase0_pytest_full_001.log`.
  - Clean detached worktree `/mnt/data/ningan/RoboticCuttingSim-wp12-p0-3f13379` (removed afterwards) at `3f13379` (only
    `.venv`/`.deps`/`artifacts` symlinked, `PYTHONPATH=<worktree>/src`, `-rfEs`): **1 failed, 1195 passed, 5 xfailed,
    0 skipped, 1515.1 s**, exit 1. The failure is
    `tests/test_blademaster_demo_runner.py::test_final_v1_dry_run_matches_the_recorded_final_runs`
    (`AssertionError: 1a`): the dry-run command starts with the checkout path (`/mnt/data/...`) while the recorded
    final_v1 command starts with `/home/ningan/RoboticCuttingSim`, so the test only passes in the original checkout.
    Environment-dependent, not a code regression; reported as is. Log
    `artifacts/wp12/logs/phase0_pytest_clean_worktree_3f13379_001.log`.
  - New this step: `tests/test_surface_metrics_v1.py` 15 passed (0.4 s); with `tests/test_repo_naming.py` 17 passed.
- Freeze record: `artifacts/wp12/logs/phase0_freeze_001.json` (copied into the evidence JSON).

## 2. File audit (0.1)

All files listed in spec §0.1 exist at `1c91605` (line counts in the evidence JSON). Differences from the spec's
list, by the latest code:

- There is no single "Workcell / Controller / Recording / Evaluation" module. Workcell: `workcell.py` (static USD
  workcell, table at `/World/Table`), `task_scenarios.py` (`_cutting_board_boxes`, scenario files
  `configs/wp10_scenarios_v3.json`, `configs/wp11_scenarios_v1.json`), `native_fixture.py` (USD authoring plan).
  Controller: `control.py` (ActionLimits, delta resolution), `policies.py` (WaypointPolicy, path_v2/path_v3),
  `goal_planner.py` (goal → waypoints), `episode_lifecycle.py` (HOME / Return-to-Home). Recording: `recording.py`,
  `dataset.py`, `scripts/collect_dataset.py`/`read_dataset.py`. Evaluation: `goals.py` (`goal_eval_v1/v2/v2a`),
  `evaluation.py` and `scene_validation.py` (frozen v1 oracle), `demo_evaluation.py`, `multi_cut_evaluation.py`,
  `surface_evaluation.py`.
- The Native runtime is `isaac_sim_runner.py` + `live_env.py` (`open_cutting_env`); the policy-facing wrapper is
  `policy_env.py` (`PolicyReadyEnv`, `open_cutting_env(..., preset="cutting_research_v1")`).
- The live material adapter is `live_material.py` (`LiveCuttingMaterial`), not `live_env.py`; it selects
  `tlmpm_v4` (laws 4–8) or `tlmpm_v5` (opt-in) and owns the fixture/table contact setup.
- `ParticleSurface` reconstruction lives in `cutting_surface.py`; `usd_surface_bridge.py` only stages the mesh.
- Phase E round runners: `scripts/run_generalization.py` (kinematic), `scripts/run_generalization_live.py` (Native);
  there are no separate "round1/round2 kinematic runner" scripts, the round is selected with `--config`.

## 3. WP-11 capabilities (0.2)

"CPU" = covered by the full CPU suite run in §1 (all listed files collected and passed unless noted). Native
and live results are historical evidence, not rerun here.

| Capability | Module | Tests (CPU) | Evidence (historical) | Confirmed now |
| --- | --- | --- | --- | --- |
| `ruler_knife_v2` physical grasp | `tool_profile.py` (`ruler_knife_v2_y/_x`), `physical_grasp*`, `isaac_sim_runner.py` | `test_tool_profile_v2.py`, `test_wp11_f1_controls.py` | `evidence/wp11/phaseB_knife_v2_001.json`, `phaseF1_grasp_001.json`, `phaseF1_negctrl_v9_001.json` (Native) | CPU PASS; Native historical |
| `tlmpm_v5` packed multi-cut | `tlmpm_v5.py`, `cut_topology.py` (packed helpers) | `test_tlmpm_v5.py` | `phaseM_tlmpm_v5_001.json` | CPU PASS; GPU rerun: scenes 1, 3, 4, 5, 8 below |
| Law 8 | `tlmpm_v4.py`/`tlmpm_v5.py` (`_segment_plow`, `_assign_lead`), ADR 0012 | `test_law8_validation.py`, `test_phase_f_v2.py`, `test_summarize_phase_f.py` | `phaseF_material_001.json`, `phaseF_material_v2_001.json` | CPU PASS; GPU rerun: all 8 scenes |
| Near-horizontal TRIM (ρ = 10°) | `skim_orientation.py`, `goal_planner.py`, `policies.py` path_v3 | `test_wp11_phase_c5.py`, `test_wp11_phase_c5b.py`, `test_wp11_path_v3a.py` | `phaseC4_trim_material_001.json`, `phaseC5b_live_trim_clear_001.json` (Native) | CPU PASS; kinematic rerun scene 2 |
| Multiple slices (Test B) | `goals.py` divide, `tlmpm_v5` | `test_generalization_round1.py`, `test_multi_cut_evaluation.py` | `phaseE_round1_001.json`, `phaseE_round1_live_001.json` (s1502 live) | kinematic rerun scene 3 |
| Multiple protrusion removal (Test C) | `goals.py` remove_protrusions, staged preflight | `test_wp11_default_preflight_staged.py`, `test_generalization_round1.py` | `phaseE_round1_live_003.json` (Native) | kinematic rerun scene 4 |
| Scene / goal / controller / evaluation separation | `scene_config.py`, `goals.py`, `goal_planner.py`, ADR 0015 | `test_scene_goal_config.py`, `test_goal_eval_v2.py`, `test_goal_eval_v2a.py`, `test_wp11_phase_d.py` | `phaseD_home_001.json` | CPU PASS |
| Configured perception relocation | `perception_relocate.py`, ADR 0016 | `test_perception_relocate.py` | `phaseE_round1_live_004.json` (Native, live-only) | CPU PASS; live-only, not rerun (scene 5 is the kinematic failure it corrects) |
| PolicyReadyEnv | `policy_env.py`, `docs/POLICY_READY_API.md` | `test_policy_ready_env.py`, `test_policy_interface.py` | `phaseG_policy_ready_001.json` | CPU PASS |
| Dataset collection | `dataset.py`, `scripts/collect_dataset.py`/`read_dataset.py` | `test_collector_recycling.py`, `test_collector_info_leaks.py` | `phaseG_policy_ready_001.json` | CPU PASS |
| Native reliability | `isaac_sim_runner.py` (`stage_reset="reload_in_place"`), `runtime_profiles.py` | `test_phaseH_runtime.py` | `phaseH_native_reliability_v1b_001.json`, `phaseI_final_v4_regression_002.json` | historical only (Native not rerun) |

## 4. How the current workcell is built (for Phase 2 replacement)

`wp11_workcell_v1` (`configs/wp11_scenarios_v1.json`) inherits `blademaster_workcell_v1`, raises the workpiece
and supports by +100 mm (`world_translation_m` z = 0.10) and declares `cutting_board` (10 mm thick, half-width
y 50 mm, wood colour), `riser` (grey, footprint = original support) and `table` materials ("task_material"
friction 0.3, restitution 0).

**One source of fixture boxes.** `scene_config.compile_task` first builds the raw support boxes in workpiece-local
coordinates (`raw_support_boxes`: solid/slotted board segments with top at local z = 0). When
`supports.workcell` is set it calls `task_scenarios._cutting_board_boxes(task, workcell_config(scene), world_z)`,
which turns **each support segment into two `MaterialFixtureBox`es**: `<name>_board` (same top, thickness 10 mm,
y widened to 50 mm) and `<name>_riser` (original x/y footprint, from the board bottom down to the table top
`table_top_world_z_m − workpiece_world_z`). The resulting tuple replaces `ConfiguredTaskSpec.fixture_boxes_local`.
Everything downstream reads that tuple:

- **TLMPM contact (physics).** `LiveCuttingMaterial.reset` (`live_material.py`) passes
  `fixture_boxes=[(center, half)]` from `task.fixture_boxes_local` and `table_plane = (R_wl[2,:], z_wl −
  table_top_world_z)` to the solver constructor (`tlmpm_v4.TLMPM` / `tlmpm_v5.TLMPM`, with `floor=False`,
  `fixture_walls=False`). Each substep launches `fixture_contacts.fixture_contacts` (finite-box SDF penalty
  impulse per box plus the half-space table, task friction; it never constrains positions or labels). The legacy
  tray (`_rigid_contacts` with `floor != 0`, hard-coded boxes) is off for every configured task. The blade and
  blunt tool boxes are handled separately by `_rigid_contacts` (blade polygon SDF + cutting work) and
  `_obstacle_contacts` (handle and finger-pad boxes in the edge frame).
- **Rendering and PhysX.** `isaac_sim_runner._author_task_fixture` deactivates `/World/Tray*`, creates
  `/World/TaskFixture` at the workpiece pose and authors one `UsdGeom.Cube` per box (`Box_{i}`, scale = half
  extents, display colour and role from `native_fixture.fixture_authoring_plan`, `UsdPhysics.CollisionAPI`, and a
  bound UsdPhysics material when declared). The table is the static workcell prim `/World/Table` from
  `workcell.py`; the runner moves it so its top is at `table_top_world_z_m` and binds the table material. The
  visual box equals the PhysX collider and the TLMPM fixture box.
- **Planning/preflight.** The C2 MuJoCo collision model and `goal_planner` read the same compiled task (HOME from
  `home_joint_positions`, action workspace from the scenario).

A Phase 2 `blade_master_flat_board_v1` therefore needs one alternative to `_cutting_board_boxes` (a single board
box resting on the table, no riser) plus the matching scenario entry; the TLMPM, PhysX and USD paths take boxes
generically. Isaac Lab builds its own boxes (`isaac_lab_env.py` `fixed_box`) and does not accept configured tasks.

## 5. How the visible surface is produced

- `cutting_surface.ComponentSurface.extract(bed)` → `CuttingSurface.extract(x, code, F)`:
  `CurrentMechanicalPartition` labels particles by the **current** mechanical components of the actual eight-node
  stencil graph (`cut_topology.mechanical_components[_packed]`, compatible codes only, gradients included). For
  each component a separate Newton `ParticleSurface` (cached per particle set) extracts an isosurface; meshes are
  concatenated with **no weld across components** (`combine_component_meshes`).
- Profile used by every configured/registered task (kinematic and Native): `current_volume_v1`: voxel = spacing/2
  (0.5 mm), kernel radius 3·spacing, threshold 0.5, kernel_scale 0.5, no field or mesh smoothing, per-particle
  radius `(spacing/2)·cbrt(det F)` (`affine_surface_controls.current_volume_radii`). Validated only on closed affine
  boxes (`docs/SURFACE_GEOMETRY_VALIDATION.md`), not on cut faces.
- `unmeshable` policy: `raise` (default) or opt-in `skip_presentation_v1` (configured Native tasks, live_003).
- Native: `isaac_sim_runner._update_surface` transforms the vertices to world and calls
  `usd_surface_bridge.DynamicSurfaceBridge.update` (one `/World/ContinuousSpecimen` mesh, all topology arrays
  rewritten each update, timestamps as custom attributes). `surface_update="every_step"` (default) or
  `"on_capture"` (Phase H opt-in).
- Consequence for WP-12: the cut face is whatever isosurface the kernel field of each component's particles
  gives. Nothing in the surface path knows where the blade went; the only cut information used is the component
  partition (global separation), not the per-link discontinuity. A cut that has not yet separated the component
  produces no face at all.

## 6. Where law 8 writes cut digits (tlmpm_v5; tlmpm_v4 identical for K ≤ 2)

Per substep (`TLMPM.step`, launch order): `_start_cut` / `_remember_cut` → `_stress` → `_segment_motion` (law-4
M_k per edge segment: armed, moving forward or drawing, motion within θ_max = 45° of the blade plane; 120 segments
of 1 mm for the 120 mm v2 edge) → `_segment_plow` (law 8: M_k and out-of-plane motion ≥ θ_plow = 15°) →
`_rigid_contacts` (blade SDF contact; cutting work → `progress`; engagement flag when work is done on a segment
with M_k) → obstacle/fixture/particle contacts → `_claim_slots` → P2G / grid / G2P / MUSL / deformation →
`_update_stroke` (stroke S = cut+1 while engaged; persists for 2Δx = 8 mm of travel) → **`_assign_band`** → (law 8)
**`_assign_lead`** → `_advance` (trajectory index).

- `_assign_band` (law 4): for the active digit, if the stroke is on and the particle's digit is still 0, the
  particle is in the edge's width (|y| ≤ 60 mm), behind the edge (local x ≤ 0, ≥ −8 mm − this substep's relative
  advance), its segment has M_k, and |side| ≤ band h = 24 mm, where `side` is the signed distance to the surface
  swept by the moving edge segment (`_swept_side`: normal = segment motion × edge direction, oriented with the
  blade normal; blade normal z if the segment does not move). Digit = 2 if side ≥ 0 else 1. Written once
  (`code |= digit << 2k`), never rewritten.
- `_assign_lead` (law 5 rule, law 8 gated by `segment_plow`): ahead of the edge (0 < local x ≤ 8 mm), Δx < |side|
  ≤ 24 mm (Δx = 4 mm core), same sign rule.
- The label boundary is therefore the sign of each particle's distance to the swept edge surface, sampled at
  particle positions (1 mm), at the substep the particle passes behind the edge. The digit itself carries no
  geometry: neither the crossing point on a link nor the blade normal is stored.
- **Blade poses available on the host.** `LiveCuttingMaterial.advance` builds the substep trajectory with
  `coupled_robot.interpolate_measured_poses(previous_local, current_local, substeps, control_dt)` (linear position,
  slerp rotation, constant twist), assigns it to `bed.poses`/`bed.twists`/`bed.cuts` and replays the 200 substeps in
  a captured CUDA graph (`bed.index` advances per substep). The host therefore knows every substep pose exactly
  (deterministic from the two control-step poses), but not *which substep* assigned a given particle; the codes
  are read back only after the control step. A Phase 3 Hermite recorder must capture the assignment substep (or
  the pose index) on the device, e.g. a per-particle `assign_index`/`assign_substep` array written in the same
  kernel branch that writes the digit, as an opt-in read-only output.
- This step's capture (`scripts/wp12_baseline_capture.py`) records the per-control-step start/end local poses (from
  which the 200 substep poses follow exactly), solver contact statistics, last-substep `segment_ok`/`segment_plow`
  counts, stroke and labelled count.

## 7. Baseline scenes (0.3)

Every run uses an existing WP-11 runner unchanged, with the original protocol seeds (material seed 42; the Phase E
sample seeds), wrapped by `scripts/wp12_baseline_capture.py`, which only adds read-only hooks (final solver
snapshot incl. F, initial/final ParticleSurface mesh built exactly as the runtimes do, per-step blade poses and
contact/plow diagnostics). The first launch (`*_001`) was stopped by me after ~4 min to add the wrapper (the
runners' `physics_state.npz` has no F, which the live `current_volume_v1` surface needs); those partial
directories are kept with `ABORTED_BY_OPERATOR.txt` and are not results.

Command pattern (per scene, from `artifacts/wp12/baseline/scenes_002.json`):

    TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/wp12_baseline_capture.py -- <runner + original args> --out artifacts/wp12/baseline/<scene>_002
    .venv/bin/python scripts/wp12_baseline_extract.py --run artifacts/wp12/baseline/<scene>_002 --scene-id <id> --frozen <WP-11 run dir>

| # | Scene (runner, protocol) | Particles | Verdict now | Frozen verdict | Key numbers now / frozen | Pieces (mass fractions) | Wall (s) now / frozen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CUT-01-R vertical through-cut (`run_phase1_matrix.py --row M1a --cutting-law 8 --solver v5`) | 96,000 | PASS | PASS (`phaseM/m1a_law8_v5_001`) | labelled 28,528 / 28,528; final codes identical (agreement 1.000) | 2 (0.50/0.50) | 497 / 737 |
| 2 | TRIM ρ = 10°, v2_y double bevel (`run_phase_c4_trim_material.py --task TRIM-02-K2-R10 --control skim`) | 97,044 | PASS | PASS (`phaseC4/k2_r10_skim_law8_001`) | removed 0.9559 / 0.9569; labelled 36,984 / 36,985; code agreement 0.99998 | 2 (0.990/0.010) | 487 / 377 |
| 3 | Test B s1502, 4 slices (`run_generalization.py` round 1) | 87,248 | TASK_SUCCESS | TASK_SUCCESS | 5 pieces; position error 5.43 / 5.43 mm (report-only) | 5 (0.207 ×4, 0.171) | 753 / 967 |
| 4 | Test C s1602, three protrusions (round 1) | 110,286 | TASK_SUCCESS | TASK_SUCCESS | cut completion 0.951 / 0.946 | 4 (0.971, 0.018, 0.007, 0.004) | 922 / 546 |
| 5 | SLICE-PUSH s1301 (round 1), kinematic | 74,256 | CUTTING_FAILURE (`cut_at_declared_planes`) | CUTTING_FAILURE | location statistic 4.0 / 4.0 mm | 3 (0.353, 0.327, 0.321) | 1040 / 500 |
| 6 | Oblique +35° s2201 (round 2) | 89,012 | CUTTING_FAILURE (1 component) | CUTTING_FAILURE | cut completion 0.5 / 0.5; 26,788 labelled | 1 | 512 / 370 |
| 7 | TRIM rake 45° s2307 (round 2) | 70,485 | CUTTING_FAILURE (`selected_protrusions_removed`) | CUTTING_FAILURE | 0 labels; 0 armed steps with law-4 M_k | 1 | 309 / 202 |
| 8 | Test D2 s2505: sever 20° → remove p0 → trim p1 (round 2, `goal_eval_v2`) | 79,039 | TASK_SUCCESS | TASK_SUCCESS | cut completion 0.952 / 0.952 | 4 (0.743, 0.245, 0.006, 0.006) | 901 / 550 |

Notes:

- SLICE-PUSH relocation (ADR 0016) is Native-only (wrist D405 depth); the kinematic scene reproduces the
  uncorrected failure it fixes. The live successes (`phaseE_round1_live_004.json`) are historical.
- `goal_eval_v2a` post-hoc on scenes 3–8 (`scripts/rescore_goal_eval_v2a.py`, output
  `artifacts/wp12/baseline/goal_eval_v2a_001`): recorded verdicts reproduced; v2a = v2 for all six; D2 s2505 is FAIL
  under v1 as already documented. POST_HOC_DIAGNOSTIC only.
- **Frozen recordings missing.** At 23:26:18 KST every `physics_state.npz` under
  `artifacts/wp11/phaseE/round1_kinematic_001/` and `round2_kinematic_001/` disappeared (directory mtimes; `/mnt/data`
  went from 95% to 93% used). This step did not delete them; `result.json`, `manifest.json` and `actions.jsonl` remain.
  Scenes 3–8 are therefore compared at verdict/metric level only. The Phase M and C4 recordings (scenes 1, 2) are
  intact and compare at code level.
- GPU float32 atomic accumulation is nondeterministic (Phase H finding), so code-level identity is not expected on
  GPU; scene 1 happened to be identical, scene 2 differs in 2 of 97,044 codes.
- Solver diagnostics (from the capture): the law-8 plow mask was **never** on at the last substep of any armed step in
  any scene (all motions are within 15° of the blade plane, including the ρ = 10° skims), so every label in these
  baselines comes from `_assign_band`. Maximum blade penetration 0.44–0.51 mm for the in-plane cuts, 1.2 mm in
  scenes 4 and 7. Maximum fixture/table penetration 0.06–0.23 mm except scenes 4 (2.87 mm) and 8 (2.28 mm), where
  detached chips fall about 100 mm from the raised `wp11_workcell_v1` board to the table (penalty contact). The scene 4 view shows the landed chips spread into particle clouds 20–30 mm wide while staying one mechanical component each (relevant to Phase 2's board-on-table workcell).
- Per scene the package is `artifacts/wp12/baseline/<scene>_002/wp12_extract_NNN/` (`baseline_compact.npz` 6–9 MB:
  initial/final positions world+local, reference positions, lattice, final codes, mechanical component labels, F,
  plastic strain, cut progress, initial/final ParticleSurface meshes, per-step blade poses and diagnostics, law-4
  stroke arrays; `baseline_summary.json`; `baseline_views.png` top / side / cut-face close-up / section). The extract
  in `scenes_002.json` is the current one; earlier extracts carry `SUPERSEDED.txt` (draft metric/view changes only).


## 8. Current geometry metrics (Phase 1 starting point)

**Method** (`src/robotic_cutting_sim/surface_metrics_v1.py`, tests `tests/test_surface_metrics_v1.py`, no Newton
import). Measured against the material state, not an ideal plane:

- Cut locus: every 6-neighbour reference-lattice link whose endpoint codes conflict (`cut_topology` semantics,
  ternary or packed). For each such link the *side locus points* `q_a = x_a + F_a (X_b − X_a)/2` and
  `q_b = x_b + F_b (X_a − X_b)/2` (reference link midpoint mapped into current space by each side's F) and the shared
  midpoint `(x_a + x_b)/2` (as specified; it falls into the gap once the pieces separate, so the side locus is the
  primary reference).
- Cut-face region of the mesh: vertices whose nearest particle is an endpoint of a cut link and not on the original
  exterior surface. "Core" additionally excludes particles within 3 lattice steps of the exterior (face edges, where
  the kernel isosurface rounds corners).
- Distance: from each region vertex to the least-squares plane of the same-component side-locus points within 1.5 mm
  (nearest-sample distances are also reported; they are quantised by the 1 mm lattice, ≈ 0.6 mm even for a perfect face).
- Normal deviation: triangle normal vs the mean signed link direction (mapped by F) of the side-locus points within
  3 mm, unsigned.
- Roughness: RMS residual of a plane fit over 3 mm patches (radius 1.5 mm), per mechanical component, for the mesh
  region and for the side-locus points (material-state roughness).
- Mesh validity: degenerate triangles, boundary / non-manifold / inconsistently oriented edges, connected components.
- Method floor (unit test): an exact mesh on an axis-aligned cut scores 0; on a 30° oblique plane through a 1 mm
  lattice the staircase alone gives plane distance P95 ≈ 0.19 mm, normal deviation P50 ≈ 2° (P95 ≈ 7°), side-locus
  roughness P95 ≈ 0.18 mm.
- Limits: the locus is lattice-quantised; F maps are first order; the region is chosen by nearest particle; where the
  pieces have not separated (scene 6) ParticleSurface draws **no cut face at all** (one component), so the numbers there
  describe the mesh near an open kerf, not a face. These are diagnostic numbers for Phase 1, not thresholds.

**Results** (final frame, solver-local frame, `current_volume_v1` mesh; mm unless noted):

| # | Cut links | Plane distance P50 / P95 / max | Core P50 / P95 | Normal dev. P50 / P95 (core) | Mesh roughness P50 / P95 | Side-locus roughness P50 / P95 | Mesh validity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 CUT-01-R | 600 | 0.086 / 0.496 / 0.751 | 0.101 / 0.494 | 13.3° / 46.7° (14.2° / 54.2°) | 0.058 / 0.198 | 0.132 / 0.421 | closed, oriented, 2 mesh components = 2 pieces, 0 degenerate |
| 2 TRIM ρ 10° | 364 | 0.235 / 0.587 / 1.119 | 0.283 / 0.568 | 18.1° / 53.7° (20.3° / 38.7°) | 0.080 / 0.237 | 0.214 / 0.349 | closed, 2 = 2, 0 degenerate |
| 3 Test B | 2128 | 0.156 / 0.583 / 1.199 | 0.179 / 0.600 | 15.6° / 37.3° (18.0° / 41.0°) | 0.068 / 0.167 | 0.014 / 0.338 | closed, 5 = 5, 0 degenerate |
| 4 Test C | 859 | 0.252 / 0.680 / 1.139 | 0.265 / 0.678 | 24.5° / 65.2° (25.8° / 58.3°) | 0.074 / 0.245 | 0.185 / 0.356 | closed, 4 = 4, 0 degenerate |
| 5 SLICE-PUSH | 952 | 0.168 / 0.554 / 1.151 | 0.170 / 0.562 | 12.4° / 40.8° (13.1° / 42.1°) | 0.060 / 0.165 | 0.009 / 0.227 | closed, 3 = 3, 1 degenerate |
| 6 oblique 35° (not separated) | 990 | 0.230 / 0.780 / 1.160 | 0.212 / 0.750 | 16.8° / 73.0° (19.3° / 38.4°) | 0.100 / 0.397 | 0.237 / 0.459 | closed, 1 = 1, 1 degenerate |
| 7 rake 45° (no labels) | 0 | — | — | — | — | — | closed, **2 mesh components for 1 piece**, 1 degenerate |
| 8 Test D2 | 1165 | 0.249 / 0.693 / 1.328 | 0.250 / 0.657 | 19.7° / 65.3° (17.1° / 45.7°) | 0.063 / 0.179 | 0.122 / 0.386 | closed, 4 = 4, 1 degenerate |

Reading (starting points for Phase 1, not conclusions):

- The meshes are topologically sound (closed, consistently oriented, no non-manifold edges; at most one degenerate
  triangle). Mesh components equal mechanical components except scene 7, where one piece of heavily plowed but
  unlabelled material is drawn as two closed shells.
- The cut faces lie on average 0.09–0.28 mm from the material's own side locus (P95 0.5–0.7 mm, max 0.75–1.3 mm),
  i.e. within about one particle radius to one spacing. Their orientation is much noisier: median 12–25° from the
  local link direction, P95 37–65°, against a method floor of 2° / 7°.
- The material locus itself is rough where it was disturbed: side-locus roughness P95 0.23–0.42 mm in every cut
  scene, larger than the mesh roughness (P95 0.17–0.25 mm), which the kernel smooths. In scenes 3 and 5 (vertical cuts
  through supported slices) the median patch is flat (0.009–0.014 mm) and only some regions are rough; in the TRIM
  and Test C skims the median patch is rough (0.19–0.21 mm). The views show why: in scene 1 the particles next to the
  face are displaced ±1–3 mm by the 1.5 mm wedge, and in scene 2 the top 3–4 substrate layers under the skim are
  sheared into tilted columns, which the mesh reproduces as ridges along the blade edge. This points to particle
  motion near the cut (physics) as a main contributor, with the reconstruction adding orientation noise; Phase 1 has
  to separate the two quantitatively.
- Before any cut the same bodies give meshes with 1.4–1.7× more triangles (lattice-aligned particles, e.g. scene 1:
  122,870 vs 76,422 vertices); not a defect, but it shows the extraction depends on the particle arrangement.
- ParticleSurface extraction of the final state took 0.28–0.92 s per call in this (loaded) setting.


## 9. Existing failure cases (with evidence)

Compact list (20 entries with causes and evidence paths): `artifacts/wp12/logs/phase0_existing_failures_001.json`,
copied into the evidence JSON. Summary:

| Case | Class | Cause | Evidence |
| --- | --- | --- | --- |
| SLICE-PUSH s1301–s1305 (kinematic) | CUTTING_FAILURE | bar slides 1.16–1.25 mm; cut 1 one layer short (4.0 mm statistic); fixed live by ADR 0016 relocation | `phaseE_round1_001.json`, `phaseE_round1_live_004.json` |
| Test A s2, −20° oblique | CUTTING_FAILURE | location 1.02 (kinematic) / 1.023 mm (live) > 1.00 mm | `phaseE_round1_001.json`, `phaseE_round1_live_001.json` |
| Test D s1701 | solver limit | v5 slot overflow at S = 8 | `phaseE_round1_001.json` |
| Test D s1702/s1704 | v1 evaluator composition | PASS under v2/v2a (post hoc) | `WP11_PHASE_E_ROUND2.md` |
| Oblique ±30–35° s2201–s2203, Test A W s2403 | CUTTING_FAILURE | bottom 1–3 layers unassigned 4–8 mm from the plane bridge the sides on the 4 mm stencil | `phaseE_round2_001.json`, `phaseE_round2_diagnostics_001.json`; reproduced here (scene 6) |
| Rake 45° s2307–s2309 | CUTTING_FAILURE | motion at θ_max = 45°: M_k never true, no label | `phaseE_round2_001.json`; reproduced (scene 7) |
| Test D2 s2501/s2502/s2506 | CUTTING_FAILURE | 1–4 interior code-zero particles bridge a chip | `phaseE_round2_001.json` |
| Round 2 feasibility | 4 SCENE_INFEASIBLE, 8 IK_FAILURE / 40 | ρ 5° gouge; IK reach with yaw ~40° | `phaseE_round2_001.json` |
| Test C live_001 / live_002 | SCENE_INFEASIBLE / NOT_RUN | convex-hull preflight; unmeshable fragment | `phaseE_round1_live_001.json`, `_003.json` |
| Test C live_003 indentation | reported | 400 top-layer particles depressed > 1 mm (max 3.19 mm) | `phaseE_round1_live_003.json` |
| Phase F F2.9 blunt handle (v1) | FAIL (control design) | v2 PASS | `phaseF_material_001.json`, `_v2_001.json` |
| Phase F3 resolution | not converged / medium ERROR | removal 0.82 / 0.58 at 2 / 1.5 mm; force ×2.6 at dt/2 | `phaseF_material_001.json` |
| Phase F F2.11 | report-only location FAIL | cuts 2, 3 at 4.0 / 4.5 mm | `phaseF_material_001.json` |
| C4 single bevel, ρ 0°, negative-control windows | FAIL rows | gate measures blade-plane motion; 0° presses down | `phaseC4_trim_material_001.json` |
| F1 v8 low-friction negative control | FAIL (design), v9 PASS | | `phaseF1_grasp_001.json`, `phaseF1_negctrl_v9_001.json` |
| C5 live TRIM clearance; C5b attempt 001 | audit FAIL; FAIL_EXECUTION (OOM) | path_v2 oscillation; foreign GPU job | `phaseC5_live_trim_001.json`, `phaseC5b_live_trim_clear_001.json` |
| Native reliability | v1 `tool_at_rest_after_reset` FAIL; recycle FAIL; +49 MiB/episode; GPU nondeterminism | | `phaseH_native_reliability_v1b_001.json` |
| final_v4 frozen strict FAILs | documented, reproduced | Demo 1b location/time; Demo 2 precision 5/8 | `phaseI_final_v4_regression_00{1,2}.json` |
| `research_fast_v1` | not acceptance-valid; SEVER location FAIL | | `phaseH_profiling_001.json` |


## 10. Open items

- The Phase E round 1/2 kinematic `physics_state.npz` recordings were deleted outside this step at 23:26:18 KST
  in an intentional cleanup requested by the user (only npz files not cited by docs; the 12 cited ones were kept;
  each folder has a `DELETED_NPZ_20261030.txt` list; see PROGRESS). Scene-level reproductions with full state now exist under
  `artifacts/wp12/baseline/` for s1502, s1602, s1301, s2201, s2307, s2505.
- Native was not rerun; Native capability rows are historical.
- `surface_metrics_v1` is a draft: Phase 1 should fix the region rule, decide on the core exclusion and add
  reference-space (pre-deformation) variants; its floor on oblique cuts is non-zero by construction.
- For Phase 3 the solver does not record when or where a digit is assigned; a device-side assignment-substep record
  is needed (§6).
- Disk: baseline outputs total 0.8 GB on `/mnt/data` (67 GB free after the deletion above).

