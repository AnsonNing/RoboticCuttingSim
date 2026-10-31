# WP-12 Phase 6 — agentic-control readiness

Date: 2026-10-31 (KST). Branch `claude/wp12-physical-cut-surfaces` (HEAD `48b3326` plus an uncommitted tree shared with
other agents). Spec: [WP12_SPECIFICATION.md](../WP12_SPECIFICATION.md) Phase 6 (§6.1–6.7) and §5.8.
Evidence: [`docs/evidence/wp12/phase6_readiness_001.json`](../evidence/wp12/phase6_readiness_001.json).
Profiling: [WP12_PERFORMANCE.md](WP12_PERFORMANCE.md).

"Agentic-control ready" here means only that the environment has the research interfaces and the checks the spec
asks for. It does **not** mean an autonomous cutting policy exists, and it does not mean the material model is
calibrated (material parameters are provisional and uncalibrated). Nothing in this phase changed physics, a cutting
law, a goal evaluator, a threshold, `tlmpm.py`/`tlmpm_v2.py`, `final_v4`, law 8 or a WP-11 record.

## 0. Verdicts at a glance

| Spec item | Verdict | Evidence |
| --- | --- | --- |
| §5.8 goal features: cut → reposition → cut, per-step planner options | **PASS** (implemented, opt-in, pinned) | `tests/test_goal_step_planner_fields.py`; CPU kinematic cut → push → cut PASS on all three steps (`artifacts/wp12/phase6_goalfeat_001`) |
| §6.1 API (`open_cutting_env(runtime, preset, scene, goal, runtime_profile="full_physics_v1", seed=42)`) | **PASS** | Native phase6_004 and kinematic phase6_005 at `full_physics_v1`, all episodes PASS; CPU tests |
| §6.2 observations; privileged state out (incl. WP-12 surface/material/geometry fields) | **PASS** | 0 leaks in 18 recorded episodes + CPU tests; Native observations carry both wrist D405s and the global D455 |
| §6.3 actions incl. rejection, safe reset, termination, two controllers | **PASS** | CPU tests; probes in kinematic ep 5/7 and Native ep 0 |
| §6.4 recording + validator | **PASS** | §6.4 validator PASS 18/18 recorded episodes (cameras NOT_APPLICABLE on kinematic) |
| §6.5 repeated episodes, kinematic (13 episodes, mixed tasks, failure → reset, failed action → valid action) | **PARTIAL** | every state-isolation, leak, storage and repeatability check PASS; the declared host-RSS gate FAILS (85.9 MiB/episode OLS, a one-time +679 MiB high-water step at the first long episode, 1.6 MiB/episode after it) |
| §6.5 Native multi-episode (CUT → TRIM → CUT, one process) | **PASS** (3 episodes; memory trend not gated) | phase6_004: no stale state, no tool detachment, no camera drift, 3/3 PASS |
| §6.6 profiling | **PARTIAL** | full breakdown from the uncontended Phase 3 / WP-11 H data plus contended Phase 6 measurements; a clean (quiet-GPU) measurement was NOT_RUN; the Phase 6 StageProfiler per-step stages were lost to a runner bug (fixed) |
| §6.7 checklist | 7 PASS, 1 PARTIAL | §8 |


## 1. What was added (all opt-in, versioned)

| Piece | File | Default behaviour |
| --- | --- | --- |
| Goal step field `planner_offset_local_m` (cut → reposition → cut) and per-step `planner` options (e.g. trim → trim with two rakes) | `goals.py`, `goal_planner.py` (`STEP_PLANNER_OPTIONS`, `validate_step_planner`) | absent unless declared; every existing goal's normalized form, task id, plan report and waypoints are pinned byte-identical (`tests/fixtures/wp12_goal_step_fields_pin.json`, 18 scene/goal pairs + 238 Phase 5 goals) |
| Phase 5 generator: G variant `cut_reposition_cut` = sequence[sever x1, move_piece kerf_face +x by d, sever reference x2 with offset [d,0,0]] | `generalization_wp12.py`, `configs/wp12/benchmark_v1_DRAFT.json` (DRAFT, not frozen) | `MIXED_OPERATIONS` has no NOT_EXPRESSIBLE row left; the two features moved to `IMPLEMENTED_FEATURES` |
| `PolicyReadyEnv.switch_task` (`task_switch_v1`): mixed tasks in one process, kinematic and Native | `policy_env.py` | no switch → reset info has no `task_switch` key; nothing changes |
| Kinematic reset clears the surface cache time (bug found here, §5.3) | `kinematic_env.py` (one line, render cache only) | the cache was already cleared; only the stale time is now cleared too |
| Readiness helpers (harness only, never policy inputs): `LEAK_KEYS_WP12`, `leak_paths`, `recorded_leak_walk`, `independent_through_cut_program`, `post_reset_state`, `attachment_snapshot`/`attachment_drift`, `classify_episode`, `validate_episode_dir` (§6.4) | `agentic_readiness.py` | new module |
| Phase 6 runner (`episodes` / `rescore` / `validate`) and read-only summarizer | `scripts/run_wp12_phase6.py`, `scripts/summarize_wp12_phase6.py` | new scripts; episodes are `collect_dataset.run_episode` episodes (`rcs_dataset_v1` layout) |
| Declared plans | `configs/wp12/phase6/*_v1.json` (stopped runs, kept) and `*_v1a.json` (the runs below) | — |

## 2. §6.1 Environment API

The spec call is unchanged and works with WP-12 scenes:

```python
with open_cutting_env(runtime="isaac_sim" | "kinematic", preset="cutting_research_v1",
                      scene="configs/wp12/scenes/bar_flat_board.json", goal=..., planner_options=...,
                      runtime_profile="full_physics_v1", seed=42,
                      surface_profile="cut_aware_surface_v1",            # versioned surface profile (opt-in)
                      recording={"root": ...}, output_dir=...) as env:
    obs, info = env.reset(seed=42); obs, r, term, trunc, info = env.step(action); result = env.evaluate()
```

- **Scenes** are `rcs_scene_v1` configs (flat board `blade_master_flat_board_v1`, slotted variant, geometry providers
  `heightfield_v2` / experimental SDF-CSG); **material profiles** are `workpiece.material_profile`
  (`cutting_tasks_v1` or a `materials_wp12_v1` id, CFL-admitted); **surface profiles** are `surface_profile`
  (`particle_surface_v1` default, `cut_aware_surface_v1`). The policy never touches TLMPM internals: actions in,
  deployable observations out.
- Recorded: Native `full_physics_v1` (phase6_004), kinematic `full_physics_v1` (phase6_005), kinematic
  `research_fast_v1` (phase6_003, development only), and CPU tests (`test_api_reset_step_evaluate_with_versioned_scene_surface_and_material_profiles`
  uses `wp12_soft_elastic_v1` + cut-aware; `test_full_physics_v1_accepts_the_flat_board_tasks_at_the_formal_resolution`).
- The Native compile reproduces the Phase 2b task ids (`CFG-bar_flat_board-sever_mid-c32f94a4`,
  `CFG-trim01_flat_board-trim_bump_rho10-4f7c00e6`) through the policy-env path.

## 3. §6.2 Observations

Deployable observation (`PolicyReadyEnv.observe`): `cameras` (Native: `wrist_d405_0`, `wrist_d405_1`, `global_d455`
RGB-D + valid mask + calibration; the kinematic backend renders none and says so: `cameras_available = false`),
`proprio` (joints q/qdot, measured edge-command pose), `ee_pose_world_xyzw`, `gripper` (available, control mode,
command, width), `task` (goal config, goal text, time remaining, max cuts), `episode` (lifecycle state, control step,
episode index) and per-step `info` (`action_status`, `action_rejection`, `action_requested`, `action_executed`,
`collision_check`, `episode_state`, `termination_reason`).

Privileged state stays privileged: particles, cut codes, components, true tool pose, contact wrenches, grasp status
are only under `observation_profile="privileged_state"`. Leak checks were extended to the WP-12 layers
(`agentic_readiness.LEAK_KEYS_WP12`: surface provenance / Hermite / cut-event log / reference-grid topology / QEF /
embedding / audit; material parameters and receipts; geometry-provider lattice reports):

| Where | Walk | Result |
| --- | --- | --- |
| CPU test, cut-aware flat-board CUT, 220 steps into the armed cut (codes and cut events exist) | every observation and step info, `PRIVILEGED_INFO_KEYS` ∪ `LEAK_KEYS_WP12` | 0 paths (`test_deployable_observation_contents_and_no_wp12_surface_material_geometry_leaks`); `privileged()` / `surface_provenance()` raise `PermissionError` without the opt-in |
| CPU test, `wp12_soft_elastic_v1` material + cut-aware | reset info, task spec | material parameters only in `material.json` / privileged receipts |
| phase6_003 (13 kinematic episodes), phase6_005 (2), phase6_004 (3 Native) | live walk of every reset/step observation and info (runner wrapper), plus the collector's `PRIVILEGED_KEYS` / `PRIVILEGED_INFO_KEYS` counts | 0 leaking observations / infos in every episode (Native included) |
| same runs, stored | `agentic_readiness.recorded_leak_walk`: stored policy-facing infos (minus the documented privileged `grasp_status`), recorded proprioception, `episode_export.npz` keys | PASS for every stored episode |

The leak-walker unit test checks that every WP-12 layer name is flagged and that an explicit `info["privileged"]`
channel is skipped (`test_leak_walker_flags_every_wp12_layer_and_skips_the_privileged_channel`).


## 4. §6.3 Actions

| Requirement | How (tested in `tests/test_wp12_agentic_readiness.py`; recorded in phase6_003/004) |
| --- | --- |
| Cartesian motion | `edge_pose` absolute WORLD edge target (ACCEPTED, commanded = target) |
| Orientation control | `edge_pose` with a rotated quaternion (0.5° about z, exact) |
| Tool reorientation | `edge_delta` rotation in the edge frame; beyond one step it is CLAMPED (`rotation_clamped`), never exceeded |
| Cut enable / disable | `cut_enabled` per action (default not armed), logged per step |
| Cut index | `cut_index` in [0, max_cuts) (bar: K = 4); 99 → `REJECTED_CUT_INDEX_OUT_OF_RANGE` |
| Sequential operations | sequence goals: MULTI (5 slices, 4 digits) and REPOS (sever → push → sever, goal_eval_v2a) |
| Invalid-action rejection | six probes per probe episode: non-unit quaternion, unknown type, unknown key, cut index, NaN delta, gripper release; each gets its declared `REJECTED_*` status and an explicit cut-disabled `HOLD_AFTER_REJECTION`, never the requested motion |
| Failed action followed by valid action | the first action after the probes is `ACCEPTED`/`REQUESTED` (kinematic episodes 5 and 7; Native episode 0) |
| Safe reset | `reset()` mid-episode: time 0, empty action log, same start pose; Native: grasp re-stabilized every reset |
| Episode termination | `end_task` → TASK_COMPLETE, a later step raises; the time limit truncates; grasp loss / IK-failure limits terminate (classified) |
| Two controllers, same scene + goal | scripted goal planner (`WaypointPolicy`) and `independent_sequence` (open-loop program built from `env.task_spec()` and the reset observation only, `uses_planner = false`); both evaluated with integrity PASS (CPU test: both sever the bar into the two declared pieces; phase6_003 episodes 0/1 and 9/10: both 0.50/0.50) |

## 5. §6.5 Repeated episodes

### 5.1 Kinematic, mixed tasks in one process (`artifacts/wp12/phase6_003`, research_fast_v1 = development only)

Command: `TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase6.py episodes --plan
configs/wp12/phase6/kinematic_mixed_v1a.json --device cuda:1 --out artifacts/wp12/phase6_003` (Warp 1.17.0, cuda:1
shared with other agents' jobs at 100 % utilization; seed 42 every episode; `cut_aware_surface_v1`; process wall
28 min 04 s; max RSS 3.77 GB; exit 3 = report FAIL on one gate). Twelve task switches in 13 episodes.

| Ep | Task (controller) | Verdict | Class | Steps | Wall (s) | RSS after (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | CUT (scripted) | FAIL | OUTCOME_FAILURE: pieces 0.50/0.50, only `cut_at_declared_planes` fails (3.0 mm > 2 mm at the 8 mm grid, the declared research_fast limit) | 700 | 111 | 989 |
| 1 | CUT (independent_sequence) | FAIL | same (0.50/0.50, location 3.0 mm) | 700 | 114 | 1015 |
| 2 | TRIM (scripted, path_v3) | PASS | TASK_SUCCESS | 725 | 120 | 1038 |
| 3 | CUT (scripted, stop after PRE_CUT, hold to the time limit) | FAIL | EXPECTED_FAILURE (expect FAIL; one piece) | 700 | 114 | 1046 |
| 4 | CUT (scripted) after the failure | FAIL | location only, as ep 0 (codes identical to ep 0) | 700 | 79 | 1058 |
| 5 | TRIM (probe → scripted) | PASS | 6/6 probes rejected as declared, next action ACCEPTED | 725 | 117 | 1053 |
| 6 | MULTI 5 slices (scripted, 4 digits) | PASS | 5 pieces 0.238/0.200/0.188/0.188/0.188 | 1850 | 312 | 1733 |
| 7 | CUT (probe → scripted) | FAIL | location only; 6/6 probes rejected, next ACCEPTED | 700 | 87 | 1733 |
| 8 | TRIM (scripted) repeat | PASS | codes identical to ep 2 | 725 | 85 | 1734 |
| 9 | CUT (scripted) repeat | FAIL | location only; codes identical to ep 0 / 4 | 700 | 84 | 1734 |
| 10 | CUT (independent_sequence) repeat | FAIL | location only; codes identical to ep 1 | 700 | 84 | 1734 |
| 11 | MULTI (scripted) repeat | PASS | codes identical to ep 6 | 1850 | 233 | 1741 |
| 12 | REPOS cut → push 20 mm → cut (scripted, goal_eval_v2a) | FAIL | sever1 and sever2 fail only `cut_at_declared_planes` (8 mm grid); push PASS; pieces 0.50/0.50 then 0.74/0.26 | 1075 | 136 | 1743 |

The CUT/REPOS FAILs are the known resolution limit of the development profile, not a readiness defect: the
same CUT task passes at `full_physics_v1` on Native (§5.2) and in WP-11/Phase 2b. The same cut → reposition → cut
goal PASSes all three steps at 4 mm particles / 4 mm grid on CPU (`artifacts/wp12/phase6_goalfeat_001`).

Report checks (`phase6_report.json`, rescored with the stop-phase grouping fix as `phase6_report_rescore_001.json`):

| Check | Result |
| --- | --- |
| all 13 planned episodes stored, 0 execution failures | PASS |
| every episode validates (`dataset.validate_episode`) and the §6.4 validator | PASS 13/13 (cameras NOT_APPLICABLE on kinematic) |
| no privileged keys (`PRIVILEGED_INFO_KEYS`) in observations / info; no WP-12 surface/material/geometry keys (`LEAK_KEYS_WP12`, live walk of every reset/step observation and info) | PASS (0 / 0 in every episode); post-hoc walk of the stored infos, proprioception and exports: 13/13 PASS |
| no clock errors | PASS |
| every reset clean (labels 0, material at reference and at rest, oracle history 1 sample, no pending contact wrench, empty cut-event log, fresh surface topology: 1 row, `full` update, 0 cut triangles, audit ok; new material / recorder / extractor objects) | PASS 13/13 |
| every episode evaluated; expected outcome (ep 3 FAIL) | PASS |
| probes rejected as declared and the next valid action executes | PASS (ep 5, 7) |
| repeat groups (verdict rule on GPU) | PASS: CUT scripted {0, 4, 9}, CUT independent {1, 10}, TRIM {2, 8}, MULTI {6, 11}; final codes also bitwise identical within every group |
| GPU memory slope (OLS eps 1–12) ≤ 32 MiB/episode | PASS (0.0; 1102 → 1134 MiB after ep 0, then flat) |
| **host RSS slope (OLS eps 1–12) ≤ 64 MiB/episode** | **FAIL: 85.9 MiB/episode** |

**Host memory diagnosis (reported, gate kept as FAIL).** RSS deltas per episode: +25.6, +23.1, +7.9, +12.4, −4.7,
**+679.4** (ep 6, the first 37 s / 1850-step MULTI episode), then 0.0, +0.7, +0.3, 0.0, +6.8, +2.2. The step is the
high-water mark of the first longer episode (its oracle history and recorder buffers are about 2.6x those of a 14 s
episode, and glibc does not return the freed heap); the second MULTI episode (ep 11) adds only 6.8 MiB. Diagnostic
OLS over episodes 6–12: 1.6 MiB/episode. So there is no unbounded growth in this run, but the declared whole-run
gate fails because the episode mix is not uniform; a fair gate needs per-task-length windows or a fixed task order,
which is a declaration for a future run, not a rescoring of this one.


### 5.2 Native, CUT → TRIM → CUT in one process (`artifacts/wp12/phase6_004`, full_physics_v1)

Command: `OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python scripts/run_wp12_phase6.py
episodes --plan configs/wp12/phase6/native_cut_trim_cut_v1a.json --gpu 0 --profile stages --out
artifacts/wp12/phase6_004`. Isaac Sim 6.2, cuda:0 shared with other agents' jobs (100 % utilization at launch,
26 GB free), seed 42, `full_physics_v1` (1 mm / 4 mm / 0.1 ms, `stage_reset = reload_in_place`), physically
grasped `ruler_knife_v2_y`, wrist D405 x2 + global D455 at `camera_every = 5`, recording every 10 steps,
`cut_aware_surface_v1`, scripted goal planner with **nominal** edge feedback (no privileged controller input), no
return home. C2 plan preflight PASS for both tasks before launch. Process wall 46 min 16 s, max RSS 17.8 GB, exit 0.

| Ep | Task | Verdict | Outcome | Steps | Wall (s) | Grasp at end (slip / rotation since settling) | RSS / GPU after (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | CUT (6 invalid-action probes, then scripted) | **PASS** | 2 pieces 0.500 / 0.500; 6/6 probes `REJECTED_*` as declared (`GRIPPER_LOCKED` on Native), next action ACCEPTED | 700 | 1271 | HOLDING, 4.7 µm / 0.29 mrad | 10578 / 5880 |
| 1 | TRIM (task switch, path_v3) | **PASS** | dome removed 0.979, substrate share 0.99996 | 725 | 940 | HOLDING, 0.4 µm / 0.02 mrad | 11432 / 6040 |
| 2 | CUT again (task switch) | **PASS** | 2 pieces 0.500 / 0.500 | 700 | 530 | HOLDING, 1.6 µm / 0.26 mrad | 11455 / 6056 |

All report checks PASS (`phase6_report.json`): stored and validated (dataset + §6.4, 13/13 items with camera
frames for all three cameras), no privileged / WP-12 keys in observations or infos (live walk 0/0 per episode;
post-hoc walk PASS), no clock errors, every reset clean (labels 0, material at reference and rest, one history
sample, no pending contact wrench, empty cut-event log, fresh surface topology, a new material / cut-event recorder /
surface extractor / tool contact sensor object after each switch, grasp re-stabilized `PASS_STABLE_GRASP`), probes,
**no tool detachment** (grasp HOLDING at every episode end, no termination), **knife-in-hand drift** over the six
reset/end snapshots 4.7 µm / 0.29 mrad (bound 2 mm), **no camera attachment drift** (the hand→optical mounts of
both wrist cameras are identical across all snapshots; the measured-vs-USD camera pose differs by at most 2.2e-8 m).
The task ids equal the Phase 2b records (`…sever_mid-c32f94a4`, `…trim_bump_rho10-4f7c00e6`), and the verdicts
equal Phase 2b's (CUT PASS, TRIM PASS), here with nominal instead of measured edge feedback.

Resources over 3 episodes (reported, not gated: 2 fitted points): host RSS +854 MiB after the first switch
(CUT → TRIM) and +23 MiB after the second; process GPU 5880 → 6040 → 6056 MiB. The WP-11 v1b Native run measured
48.8 MiB/episode host growth for `reload_in_place` over 8 same-task episodes; this short run neither confirms nor
refutes a bound for mixed tasks.


### 5.3 Finding: stale surface-cache time in the kinematic reset (fixed)

`KinematicCuttingBackend.reset` cleared the cached surface but not its timestamp. When the previous episode's last
extraction was at t = 0 (no extraction during its steps), the first extraction after the next reset returned the
cleared cache (`None`) instead of extracting. It surfaced through the Phase 6 task-switch test as an IndexError on
the new episode's first surface row. Fix: the timestamp is cleared with the cache (render cache only; physics and
the default outputs are unchanged). Regression test `test_kinematic_reset_clears_the_surface_cache_time`. The
Native runner re-extracts during reset and did not have the problem.

### 5.4 Determinism categories

| Category | Evidence | Result |
| --- | --- | --- |
| **CPU deterministic replay** | `test_phase6_driver_mixed_plan_stores_valid_episodes_and_replays_bitwise_on_cpu`: a stored CPU episode of a mixed-task run replayed open loop on a fresh kinematic env (`dataset.replay_kinematic`) | final codes mismatched 0 (bitwise); also the WP-11 replay test |
| **GPU kinematic repeatability** (phase6_003) | 4 repeat groups, same task / controller / seed in one process after other tasks | verdicts identical in every group; final codes were also bitwise identical in every group in this run. That is an observation, **not** a claimed capability (GPU float32 atomics can reorder) |
| **Native verdict-level repeatability** (phase6_004) | CUT ep 0 vs ep 2 (ep 0 had six rejected-action HOLD steps at HOME first) | verdict PASS both, pieces 0.500 / 0.500 both; the final codes happened to be identical (sha256 `7dd406ed…`). Native bitwise repeatability is **not** claimed: WP-11 v1b measured cross-reset code differences (dome 0.9789 vs 0.9770) for the same controller |
| **Numerical trajectory variation** | per-step wall times and tracking errors in every `actions.jsonl`; Native CUT ep 0 vs ep 2 differ in timing only (material step 985 vs 306 ms mean under different GPU contention) | reported, never gated |


## 6. §6.4 Recording

`agentic_readiness.validate_episode_dir` (CLI: `scripts/run_wp12_phase6.py validate --root DIR`) checks one stored
episode for every spec §6.4 item:

| Item | Where it is stored |
| --- | --- |
| Full configuration | `manifest.json` (runtime config), `task.json`, `material.json`, `episode.json` (scene, goal, configured-task payload, tool profile, preset) |
| Runtime profile | `episode.json` `runtime_profile` + `acceptance_valid` (or the explicit numerics in `physics_version`) |
| Action sequence | `actions.jsonl` (submitted action, info, measured state per step) + `action_log.jsonl` (status, executed kind, cut flags, wall time) |
| Camera observations | `observations/<camera>_####.npz` + calibration in `frame_manifest.json` (NOT_APPLICABLE on kinematic) |
| Physical state | `physics_state.npz` (positions, velocities, codes at the recorded frames, reference lattice and stencil), `robot_state.json/.npz` |
| Cut events | code history in `physics_state.npz`, per-step cut flags; with cut-aware: `cut_events_log.npy` |
| Contact diagnostics | per-step material contact statistics and reaction wrenches (`actions.jsonl`), Native grasp status per step |
| Surface reconstruction metadata | `surface_provenance.json`: profile, reference-grid topology (sites, links, dual cells, hashes), source hashes, recorder log SHA256 (checked against `cut_events_log.npy`), per-extraction Hermite counts by source, QEF, embedding, regions, audit, frame→extraction map |
| Evaluation | `result.json`, `episode.json` `task_result` (integrity / outcome / quality) |
| Failure classification | `phase6.json` `failure_classification` (`agentic_readiness.FAILURE_CLASSES`), checked against the stored record |
| Randomization seed | `episode.json` `seed` + `seed_source` |
| Code commit | `episode.json` `git` (sha, dirty flag, packages) |
| Timing | episode / evaluation / reset wall, per-step wall; optional `profile.json` (StageProfiler) |

Every stored Phase 6 episode passes both `dataset.validate_episode` (`rcs_dataset_v1` integrity) and the §6.4
validator: phase6_003 13/13, phase6_005 2/2, phase6_004 3/3 (Native: all 13 items incl. camera
frames for both wrist D405s and the global D455). `test_phase6_driver_mixed_plan_stores_valid_episodes_and_replays_bitwise_on_cpu`
shows the validator naming the missing item when `surface_provenance.json` is deleted. The surface provenance of
every cut-aware episode carries the recorder log SHA256, and it equals the hash of the stored
`cut_events_log.npy` (checked by the validator).


## 7. §6.6 Profiling

See [WP12_PERFORMANCE.md](WP12_PERFORMANCE.md). `full_physics_v1` stays the acceptance profile; `research_fast_v1` and
the coarse CPU numerics are labelled development only everywhere.

## 8. §6.7 checklist

| §6.7 requirement | Verdict | Evidence |
| --- | --- | --- |
| Tasks can be generated from configuration | **PASS** | every Phase 6 task is an `rcs_scene_v1` + `rcs_goal_v1` compile (`scene_config.compile_task`); the Phase 5 generator builds families A–H, now including cut → reposition → cut, from declared dimensions (`generalization_wp12.py`, DRAFT benchmark); task switching recompiles from config |
| Task completion can be evaluated objectively | **PASS** | goal evaluators with separate integrity / outcome / quality layers (`goals.py`, goal_eval_v1/v2/v2a) read only the recorded oracle; every Phase 6 episode is evaluated; an early-ended episode is never scored as a task result (`complete_episode`) |
| The agent can repeatedly Reset / Execute / Evaluate | **PASS** (kinematic: PARTIAL on the declared host-memory gate) | 13 + 2 kinematic and 3 Native episodes in single processes with task switches; every reset clean |
| The agent can get the failure reason | **PASS** | `evaluate()` outcome checks / `fail_reasons`, `termination_reason` and lifecycle state in every info, `action_status` + `action_rejection` per step, `agentic_readiness.classify_episode` (EXPECTED_FAILURE / OUTCOME_FAILURE / INTEGRITY_FAILURE / TERMINATED_*) stored per episode |
| Different research methods can share the environment | **PASS** | goal planner + WaypointPolicy, an open-loop program from `task_spec()` only, and the collector's `module:factory` policy hook all run on the same env and task ids |
| The dataset can collect success and failure cases | **PASS** | `rcs_dataset_v1` stores and validates every episode: success, the deliberate failure (ep 3), resolution-limited failures and probe episodes |
| Replacing the policy needs no physics-engine change | **PASS** | the controllers live outside the env; the same task ids (physics config hashes) serve every controller; Phase 6 changed no physics file |
| Formal tasks need no privileged state in the policy | **PASS** for the scripted planner and the independent sequence | Native CUT/TRIM/CUT PASS at `full_physics_v1` with **nominal** edge feedback (scene config + deployable command only); `measured` edge feedback stays a declared privileged controller input (`PRIVILEGED_CONTROLLER_INPUTS`) and was not used |

Readiness only: no autonomous cutting policy was trained or evaluated, and the material model is not calibrated.


## 9. Commands, tests, artifacts

| Run | Command (all `TMPDIR=/mnt/data/ningan/tmp`, seed 42) | Wall | Result | Output |
| --- | --- | --- | --- | --- |
| goal feature | `.venv/bin/python scripts/run_wp12_cut_reposition_cut.py --out artifacts/wp12/phase6_goalfeat_001` (CPU, 4 mm / 0.5 ms, development) | 87 s | PASS 3/3 steps; push 19.95 mm in the window vs 20 mm declared | `artifacts/wp12/phase6_goalfeat_001` |
| kinematic v1 (stopped) | `run_wp12_phase6.py episodes --plan configs/wp12/phase6/kinematic_mixed_v1.json --device cuda:1 --out artifacts/wp12/phase6_001` | stopped after 2 episodes | plan error: no `hold_until_time_limit`, episodes FAIL_INTEGRITY by design; kept with `ABORTED.txt` | `artifacts/wp12/phase6_001` |
| Native v1 (stopped) | `… --plan configs/wp12/phase6/native_cut_trim_cut_v1.json --gpu 0 --profile stages --out artifacts/wp12/phase6_002` | stopped in episode 0 | same plan error; kept with `ABORTED.txt` | `artifacts/wp12/phase6_002` |
| kinematic mixed v1a | `.venv/bin/python scripts/run_wp12_phase6.py episodes --plan configs/wp12/phase6/kinematic_mixed_v1a.json --device cuda:1 --out artifacts/wp12/phase6_003`; rescore `… rescore --root artifacts/wp12/phase6_003 --tag 001` | 28 min 04 s | report FAIL on `host_memory_bounded` only (§5.1) | `artifacts/wp12/phase6_003`, log `artifacts/wp12/logs/phase6_003.log` |
| Native v1a | `OMNI_KIT_ACCEPT_EULA=YES .deps/isaacsim-venv/bin/python scripts/run_wp12_phase6.py episodes --plan configs/wp12/phase6/native_cut_trim_cut_v1a.json --gpu 0 --profile stages --out artifacts/wp12/phase6_004` | 46 min 16 s | PASS (3/3 PASS) | `artifacts/wp12/phase6_004`, log `…/logs/phase6_004.log` |
| kinematic full physics v1a | `.venv/bin/python scripts/run_wp12_phase6.py episodes --plan configs/wp12/phase6/kinematic_full_physics_v1a.json --device cuda:1 --profile stages --out artifacts/wp12/phase6_005` | 11 min 58 s | PASS (CUT, TRIM PASS) | `artifacts/wp12/phase6_005` |
| validator | `.venv/bin/python scripts/run_wp12_phase6.py validate --root artifacts/wp12/phase6_003` | < 10 s | PASS 13/13 | `artifacts/wp12/phase6_003/spec_6_4_validate_cli_001.json` |
| evidence | `.venv/bin/python scripts/summarize_wp12_phase6.py --run … --extra … --tests artifacts/wp12/phase6_tests/tests_phase6.json --out docs/evidence/wp12/phase6_readiness_001.json` | < 1 min | — | evidence JSON |

CPU tests (`.venv/bin/python -m pytest`, Warp 1.17.0 CPU): new `tests/test_wp12_agentic_readiness.py` (10 tests) and
`tests/test_goal_step_planner_fields.py` (42 incl. parametrized), plus the policy / dataset / recording / leak /
goal / generalization / workcell suites, 27 files: **385 passed, 3 failed** (555 s, log
`artifacts/wp12/logs/phase6_tests_002.log`). The 3 failures are pre-existing at HEAD `48b3326` in files this phase
did not touch: `test_phaseH_runtime.py::test_simulator_modules_do_not_import_the_profiler` (the committed
`isaac_sim_runner.py` contains the word "profiling") and two `test_closeup_camera.py` recorder tests (their test
double lacks `surface_selection`, since `9fde661`).


## 10. Open issues

1. **Kinematic host-memory gate FAIL** (§5.1): a one-time high-water step at the first long episode; the gate needs
   a declared per-length window or a fixed task order in a new run. Native memory over > 3 mixed-task episodes is
   not measured (WP-11 v1b: ~49 MiB/episode for same-task `reload_in_place`; process recycling exists).
2. **Profiling**: no quiet-GPU Phase 6 measurement (NOT_RUN: both GPUs busy all session); the StageProfiler per-step
   stages of phase6_004/005 were lost to the runner's leak-watcher wrapper (fixed afterwards, not re-measured).
3. **research_fast_v1 CUT verdicts** fail on location by construction; they are development-only and are not used as
   task results.
4. **`switch_task` scope**: same control dt; Native also the same grid dx and tool profile. Changing the held knife,
   the workcell USD or the surface profile between episodes needs a new process. Isaac Lab: NOT_RUN.
5. **Open-loop offset**: `planner_offset_local_m` uses the declared push distance, not the measured piece pose
   (ADR 0016 is the measured route, live only). No `full_physics_v1` or Native cut → reposition → cut run yet.
6. **Native bitwise repeatability is not claimed** even though CUT ep 0 and ep 2 ended with identical codes here.
7. The 3 pre-existing CPU test failures (§9) belong to other work packages' files.


