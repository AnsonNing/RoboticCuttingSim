# Policy-ready API and dataset format (WP-11 Phase G)

The `cutting_research_v1` preset turns the shared cutting environment into a
policy-facing research API: validated actions with explicit status, deployable
observations kept apart from privileged simulator state, a dataset collector
with a reader, and multi-episode reliability checks (`POLICY_READY`). It is
opt-in. Without `preset=` every existing default, legacy preset
(`policy_config.policy_preset`) and frozen benchmark behaves as before.

Results and evidence: [phaseG_policy_ready_001](evidence/wp11/phaseG_policy_ready_001.json).
Older references: [POLICY_CONTROL.md](POLICY_CONTROL.md) (action API v2, reward,
Gym, agent tools) and [SHARED_TASK_API.md](SHARED_TASK_API.md).

## Opening an environment

```python
from robotic_cutting_sim.live_env import open_cutting_env

with open_cutting_env(runtime="kinematic",            # or "isaac_sim" (Native); Isaac Lab: NOT_RUN
                      output_dir="artifacts/wp11/phaseG/my_run",   # fresh directory, never reused
                      preset="cutting_research_v1",
                      scene="trim01_wp11_workcell",   # rcs_scene_v1 name/path/dict
                      goal="trim_bump_rho10",         # rcs_goal_v1 name/path/dict
                      planner_options={"max_rotation_deg": 1.0, "max_translation_m": 0.001},
                      tool_profile=None,              # optional tool profile id override (recorded in the task id)
                      particle_spacing_m=0.004,       # optional numerical override (coarse runs; recorded)
                      seed=42,
                      recording={"root": "artifacts/wp11/phaseG/my_run_data", "every": 10},   # optional
                      observation_profile="deployable",    # or "privileged_state" (explicit opt-in)
                      preset_overrides={"camera_every": 5},
                      material_dt=0.0005, device="cpu") as env:   # runtime options pass through
    observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(
        {"type": "edge_delta", "target": [0, 0, -0.001, 0, 0, 0], "frame": "world", "cut_enabled": False})
    result = env.evaluate()          # the goal's own evaluator (integrity / outcome / quality)
    home = env.return_home()         # optional physical return (demonstrations); RL just calls reset()
```

A compiled `scene_config.ConfiguredTaskSpec` may be passed as `task=` instead of
`scene=`/`goal=`.

### Preset `cutting_research_v1` (`policy_env.ResearchPreset`)

| Field | Default | Meaning |
| --- | --- | --- |
| `tool_attachment_mode` | `physical_grasp` | Native: the knife is a free rigid body held by the fingers |
| `tool_init_mode` | `pregrasped` | Native: the unchanged physical pregrasp settle at every reset |
| `gripper_control_mode` | `hold` | `hold`: only `gripper: "hold"` is accepted; `policy`: `release` too |
| `home_reset` | `when_declared` | Native resets the arm to the scene HOME when the scene declares one |
| `delta_frame` | `world` | default frame of `edge_delta` actions |
| `rejection_policy` | `hold` | rejected action: explicit cut-disabled HOLD step (`no_step`: nothing runs, the clock stops) |
| `collision_check` | `True` | sampled blade/handle vs declared fixture boxes and the table (see below) |
| `collision_tolerance_m` | 0.0005 | allowed penetration before `REJECTED_COLLISION_*` |
| `include_global_camera` | `True` | Native: the global D455 joins both wrist D405s |
| `camera_every` | 1 | Native: control steps between rendered policy camera observations |
| `default_cut_enabled` | `False` | an action without `cut_enabled` is not armed |

Action limits are the compiled task's own (`task.action_limits()`: the planner
step limits and the workcell workspace box), else `ActionLimits()`.

### Lifecycle

`reset(seed)` is the direct simulator reset (robot, knife pose/velocity, grasp
re-stabilization, workpiece, TLMPM state and cut codes, clocks, oracle history,
action log); it never moves the robot physically. `return_home()` is the
separate preflighted physical motion with its own `home_outcome`
(`PASS`/`FAIL`/`NOT_APPLICABLE`); RL episodes are never forced to return home.
Every `info` carries `episode_state` (`episode_lifecycle.EpisodeStateMachine`:
`RESETTING -> READY -> RUNNING -> TASK_COMPLETE | TASK_FAILED -> RETRACTING ->
RETURNING_HOME -> FINISHED`) and `termination_reason`. `TASK_COMPLETE` is not
task success; success is only `evaluate()`. When the backend reports surface cells it could not mesh
(`backend.surface_unmeshable_summary()` returns a dict), the `evaluate()` result
also carries `surface_unmeshable`; without such a summary (kinematic, or the raising
surface mode) the key is absent and every result is unchanged.

## Actions

One mapping per 20 ms control step (50 Hz):

```python
{"type": "edge_pose",      "target": [x, y, z, qx, qy, qz, qw]}        # absolute WORLD knife-edge pose
{"type": "edge_delta",     "target": [dx, dy, dz, rx, ry, rz], "frame": "world" | "edge"}  # m, rotation vector
{"type": "joint_position", "target": q7}                               # live runtimes only (radians)
# optional on every action:
#   "cut_enabled": bool           (default False)
#   "cut_index":   int in [0, max_cuts)   (max_cuts = the solver's retained digits; up to 15 with tlmpm_v5)
#   "gripper":     "hold" | "release"     ("release" needs gripper_control_mode="policy" and physical_grasp)
{"gripper": "hold"}                                                    # gripper-only action
```

The legacy `mode` names (`cartesian_edge_pose_world`, `cartesian_edge_delta`,
`joint_position_rad`) are accepted in place of `type`, so the existing scripted
policies (`policies.WaypointPolicy`) drive the env unchanged. Edge frame: +x is
the cutting (advance) direction, y runs along the edge, z is the blade normal.

Every action passes, in order: schema validation, finite/unit-quaternion target
checks, `cut_index` range, gripper policy, action limits (per-step translation
and rotation bounds plus the workspace box, which may **clamp**), joint limits,
the geometric collision check, and finally the runtime's IK. `info` reports:

| Key | Content |
| --- | --- |
| `action_status` | `ACCEPTED`, `CLAMPED` or `REJECTED_<reason>` |
| `action_rejection` | `{reason, detail}` or `None` |
| `action_requested` | the submitted action (JSON) |
| `action_executed` | what actually ran: `kind` = `REQUESTED`, `HOLD_AFTER_REJECTION`, `JOINT_HOLD_AFTER_REJECTION`, `HOLD_GRIPPER_NOOP` or `NOT_EXECUTED`, plus mode/target/cut flags |
| `action_report` | requested vs applied motion and which limit clamped |
| `collision_check` | status, fixture penetration, table clearance |
| `action_timestamp_s`, `timestamp_s` | observation time the action was computed from, and the time after the step |
| `ik_status`, `ik_info` | `rejected` with the runtime's IK report on `REJECTED_IK_UNREACHABLE` |

Default step info carries no privileged value: the inner env's `grasp_status` (the
contact-based grasp monitor, simulator truth) is removed from the returned info
(WP-11 review finding 9, 2026-10-30) and is available as
`info["privileged"]["grasp_status"]` with `observation_profile="privileged_state"`.
A grasp loss still ends the episode and is named in `termination_reason` /
`episode_state`. The recorder keeps the full inner info in `actions.jsonl`
(privileged storage). The legacy `open_cutting_env` without `preset=` is unchanged
and still returns `grasp_status` in its info.

Rejection reasons: `INVALID_SCHEMA`, `INVALID_TARGET`, `CUT_INDEX_OUT_OF_RANGE`,
`JOINT_LIMIT`, `GRIPPER_LOCKED`, `GRIPPER_UNSUPPORTED`, `UNSUPPORTED_BY_RUNTIME`
(joint targets on the kinematic backend), `COLLISION_FIXTURE`,
`COLLISION_TABLE`, `IK_UNREACHABLE`. A rejected action is **never executed and
never replaced silently**: with `rejection_policy="hold"` the env runs one
explicit cut-disabled hold of the last accepted target (reported as
`HOLD_AFTER_REJECTION`) so the 50 Hz clock keeps running; with `no_step` nothing
runs. This includes the gripper part: an action with `"gripper": "release"` whose
target the runtime IK rejects leaves the gripper command unchanged (the inner env
sets it before the IK fails, before any physics step; the wrapper restores it), so
the HOLD runs with the previous gripper command. `env.action_log` keeps one row per step (status, executed kind, cut flags,
timestamps, wall time) and the recorder stores it.

The collision check is geometry only: about 9,000 sampled points on the blade
faces/edge and the handle box at the commanded edge pose against the scene's
declared rigid fixture boxes (board, riser, pedestals) and the table top. The
workpiece is not an obstacle (cutting it is the task). Robot links and cameras
are not part of this per-step check; they are covered by the C2 MuJoCo
preflight that `scripts/collect_dataset.py` runs before any Native launch
(scripted plans and action programs) and by the runtime IK.

`validate_action(action, *, max_cuts=None)` (`tasks.CuttingTaskSpec`) keeps the
legacy digits 0/1 for every existing caller; only an explicit `max_cuts`
(the runtime material history, 1–15) widens the range. `ConfiguredTaskSpec`
uses its scene `max_cuts` and an explicit `max_cuts` can only narrow it.
`CuttingTaskEnv` and the kinematic backend pass the material's `max_cuts` when
it exceeds 2, so `tlmpm_v5` digits 2–14 are now reachable through the API.

## Observations

`env.observe()` (also returned by `reset`/`step`) is the **deployable**
observation:

| Key | Content |
| --- | --- |
| `timestamp_s` | physics time of the measurement |
| `proprio` | measured `q`, `qdot` (2 x 7), `hand_pose`, `finger_q`, `finger_qdot` (physical grasp); kinematic: `q`, `qdot`, `tool_pose` (the ideal commanded edge) |
| `ee_pose_world_xyzw` | measured hand (end-effector) pose (kinematic: hand from the ideal edge) |
| `edge_command_world_xyzw` | the last accepted commanded edge (controller state, not a measurement) |
| `gripper` | `available`, `control_mode`, `command`, `width_m` (sum of finger encoders) |
| `cameras` | Native: `rgb`, `depth_m`, `valid_mask`, `calibration` for `wrist_d405_0`, `wrist_d405_1`, `global_d455` |
| `camera_timestamp_s`, `camera_age_s` | physics time of the latest rendered frames and their age (0 when `camera_every == 1`) |
| `task` | `task_id`, `goal` (the rcs_goal_v1 struct), `goal_text`, `scene_id`, `tool_profile_id`, `duration_s`, `control_dt_s`, `max_cuts`, `home_declared`, `time_remaining_s` |
| `episode` | lifecycle `state`, `termination_reason`, `home_outcome`, `control_step`, `episode_index` |

`env.task_spec()` returns the full a-priori description (scene config, goal,
action types, action limits, preset). The kinematic backend renders no cameras
(`cameras == {}`).

### Privileged state (debug only)

`observation_profile="privileged_state"` (explicit opt-in) adds `info["privileged"]`
to every step and to the reset info (there also `info["privileged"]["reset_checks"]`,
the full post-reset receipt) and enables `env.privileged(components=...)`; otherwise
`privileged()` raises `PermissionError`. The privileged dict is never merged into
the observation: `material_state` (particle positions, velocities, codes),
`cut_codes`, `code_encoding`, `oracle_geometry` (reference particles, lattice,
masses, workpiece transform), `contact_wrench_world` (the material reaction
wrench), `tool_truth` (simulator pose and velocity of the held knife body and
edge), `grasp_status`, and optionally `component_labels` (actual MPM stencil
graph). `policy_env.PRIVILEGED_KEYS` lists the names that must never appear in a
deployable observation; the tests and the collector walk every observation for
them. `policy_env.PRIVILEGED_INFO_KEYS` (a superset: also the privileged reset-check
names) must never appear in a default reset or step info outside `info["privileged"]`;
`find_privileged_info_keys(info)` walks an info dict recursively
(`tests/test_policy_ready_env.py`).

**Reset checks.** The default reset info's `reset_checks` holds only bookkeeping and
deployable fields (`policy_env.DEPLOYABLE_RESET_CHECKS`: clock at zero, history /
lifecycle / IK counter / termination / step counter / action log reset, observation
and camera timestamps, `q_active`). The privileged fields (`labels_clean`,
`material_at_reference_max_m`, `material_velocity_max_m_s`, `material_codes_sha256`,
`particles`, simulator `tool_pose_world_xyzw` / `edge_pose_world_xyzw`, tool speeds,
`grasp_reset_status`) are in `env.last_reset_checks` for the harness (the collector
stores the full receipt in `episode.json` `reset_checks`, which the reliability checks
read, unchanged) and in `info["privileged"]["reset_checks"]` with
`privileged_state`. Before 2026-10-30 (review finding 9) the default reset info
carried the full receipt; `episode.json` `reset_info` of datasets collected before
then therefore contains it too.

**Privileged controller inputs** (`policy_env.PRIVILEGED_CONTROLLER_INPUTS`):

| Input | Why it is privileged |
| --- | --- |
| `edge_feedback="measured"` (path_v2/path_v3/path_v3a measured-edge feedback, used by the C5 scripted run) | simulator tool-edge pose of the physically held knife (tool truth) |
| `reward.CuttingReward` | reference particles, cut codes, measured tool edge (shaping only) |
| `env.evaluate()` | oracle particle history, codes, components (scoring only) |

The goal planner itself reads only the scene and goal configs (a priori), never
material, label or component state. The collector records the privileged inputs
of every episode's controller in `episode.json` (`privileged_controller_inputs`).

## Dataset collector and format (`rcs_dataset_v1`)

```bash
# kinematic (CPU or GPU), several controllers in one process from an episode plan
TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/collect_dataset.py --runtime kinematic \
    --scene trim01_wp11_workcell --goal trim_bump_rho10 \
    --planner-options '{"max_rotation_deg": 1.0, "max_translation_m": 0.001}' \
    --episode-plan configs/wp11/phaseG/kinematic_reliability_trim_v1.json \
    --spacing 0.004 --material-dt 0.0005 --device cpu --out artifacts/wp11/phaseG/<new_dir>
# Native Isaac Sim (C2 preflight of every plan/program first; refuses to launch on FAIL)
OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python scripts/collect_dataset.py \
    --runtime isaac_sim --gpu 0 --scene ... --goal ... --episode-plan configs/wp11/phaseG/native_reliability_trim_v1.json \
    --camera-every 5 --record-every 10 --out artifacts/wp11/phaseG/<new_dir>
# read, validate, analyze, replay (kinematic episodes)
.venv/bin/python scripts/read_dataset.py artifacts/wp11/phaseG/<dir> --replay-kinematic
```

Controllers: `scripted` (the goal planner through `WaypointPolicy`, optional
`edge_feedback: measured`, `stop_after_phase`, `hold_until_time_limit`, and with a
`bias_pose` profile (path_v3/path_v3a) the opt-in stall options `stall_timeout_s`,
`stall_action` (`end_episode` | `skip_waypoint`), `stall_timeouts_by_label`; see
"Controller profile path_v3a" below),
`action_file` (an open-loop `rcs_action_program_v1` file) and `policy` (an
external `module:factory`, `factory(task_spec, seed) -> policy(observation) ->
action`; examples in `policy_examples.py`). An episode plan
(`rcs_episode_plan_v1`) mixes them per episode with fixed or `"random"` seeds
(drawn from `--seed-base`, recorded), optional `return_home` and an optional
declared `expect` verdict. Every episode is stored, successful or failed, and the
env is reset automatically between episodes.

Layout:

```
<root>/dataset_manifest.json   version, command, git commit/dirty, packages, collector config, reliability config
<root>/index.json              one row per episode: id, dir, label, controller, seed, verdict, expect,
                               termination, home outcome, steps, sha256 of the export
<root>/reliability.json        Test F checks per episode and across episodes, POLICY_READY
<root>/preflight.json          Native only: C2 preflight receipts
<root>/runtime/                runtime receipts (Native kit log, grasp resets, camera calibration, ...)
<root>/episode_###/
    episode.json               scene / workpiece / supports / tool profile, seed + source, task id and config sha256,
                               recompilable task payload, controller receipt, privileged controller inputs,
                               frequencies, task result (goal evaluator), home outcome + measurement,
                               termination and task-end reasons, lifecycle transitions, action status counts,
                               physical diagnostics (tracking, grasp), reset checks, memory, git, physics version
    action_log.jsonl           per-step action status log of the policy env
    actions.jsonl              per-step submitted/executed action, info, measured robot state, diagnostics
                               (EpisodeRecorder; `privileged_diagnostics` and `material_diagnostics` are privileged)
    frame_manifest.json        recorded frame times and camera calibration
    observations/<cam>_####.npz  RGB, depth, valid mask, K, T_world_camera, physics timestamp (Native)
    physics_state.npz          PRIVILEGED material history (positions, velocities, codes) at the frame times
    robot_state.json/.npz      measured robot state and rigid-body poses
    task.json, material.json, result.json
    episode_export.npz         consolidated DEPLOYABLE arrays (below)
```

`episode_export.npz` keys: `step_time_s`, `action_timestamp_s`,
`action_status`, `executed_kind`, `executed_mode`,
`executed_edge_target_world_xyzw`, `executed_joint_target_rad`, `cut_enabled`,
`cut_index`, `requested_action_json`, `policy_phase`, `termination_reason` (return-home rows have `action_status` `RETURN_HOME_PLAN`; datasets written before 2026-10-30 06:50 KST label them `LEGACY_UNVALIDATED`),
`reward`, `proprio__<q|qdot|hand_pose|finger_q|finger_qdot|tool_pose>`,
`frame_time_s`, and per camera `camera__<name>__<rgb|depth_m|valid_mask|time_s|K|T_world_camera>`.
h5py is not installed in either runtime, so the export is NumPy `.npz`.

Frequencies are recorded, never assumed: `frequencies.control_rate_hz` is the
task rate (50 Hz); `recorded_camera_rate_hz` is the stored camera rate
(50 Hz / `--record-every`); `policy_camera_rate_hz` is the rate at which the
policy saw fresh frames (50 Hz / `--camera-every`); `measured_camera` gives the
measured frame intervals per camera. Camera content is never 50 Hz unless both
are 1.

`scripts/read_dataset.py` validates every stored episode (required files,
export sha256 against the index, uniform 50 Hz task clock, action timestamps
one control step before their results, frame and physics-state timestamps at
step times, no privileged key in the export, evaluation and termination
present, final-codes hash) and analyzes it (status counts, phases, armed steps,
camera frames, verdict). `--replay-kinematic` re-executes a kinematic
episode's executed absolute edge targets open loop on a fresh kinematic env
and compares the final cut codes (exact on CPU); Native episodes are analyzed
only.

The collector refuses, before it creates the output directory or launches a runtime,
a scripted plan with `planner_options.perception` (the ADR 0016 wrist-depth survey +
relocation): its scripted controller is a plain `WaypointPolicy` without the relocation
hook or a depth source, and it used to drop the option silently. Run relocation with
`scripts/run_wp10_live.py`. Stall options with a non-`bias_pose` profile are refused
the same way.

### Controller profile path_v3a (WP-11 review finding 8, opt-in)

`path_v3` integrates its bias without bound, also while the command is clamped: with a
blocked blade the bias wound up (> 30 mm on the fitted C5 plant with a floor;
-12 mm in the review) and the arm kept pressing until the time limit. `path_v3a`
(`policies.CONTROLLER_PROFILES`, planner option `"controller_profile": "path_v3a"`) is
path_v3 plus anti-windup:

- the bias is clipped to `bias_limit_m = 3 mm` and `bias_limit_rad = 15 mrad` (the C5b
  live run peaked at 1.55 mm / 5.9 mrad);
- `anti_windup`: no bias integration on a step whose previous command was clamped by
  the action limits (translation, rotation or workspace);
- opt-in stall timeout (`WaypointPolicy(stall_timeout_s=..., stall_action=...,
  stall_timeouts_by_label=...)`, any `bias_pose` profile; default off): a waypoint whose
  measured edge is outside the tolerance and has not come `stall_min_progress_m`
  (0.2 mm) / `stall_min_progress_rad` (2 mrad) closer within the timeout is a stall.
  `end_episode` makes the policy done with `termination_request =
  "WAYPOINT_STALL:<label>"` (the collector ends the task with that reason and does not
  run `hold_until_time_limit`); `skip_waypoint` starts the next waypoint. Every stall is
  in `policy.stall_events` and the collector receipt (`controller_receipt.stall`).

path_v3 is unchanged, because a fix in place would have changed the C5b live run: an
open-loop replay of the recorded C5b episode
(`artifacts/wp11/phaseC5/live_trim_rho10_clear_002`, `tests/c5b_replay.py`) through the
current path_v3 reproduces all 1056 recorded commands and `policy_tracking` rows
bitwise, and the controller's own action-limit clamp was active on 254 of its 1057
calls (REORIENT, WITHDRAW, APPROACH_PUSH, RETRACT, PRE_CUT), so the freeze would have
been active (`tests/test_wp11_path_v3a.py`). On the fitted C5 plant path_v3a keeps the
C5b settle / SKIM quality, raises no false stall over the full nominal plan, stays at
the 3 mm bound on a blocked blade and detects the stall about 1 s after the blade
stops. path_v3a has not been run on a live runtime (NOT_RUN).

## Action programs (`rcs_action_program_v1`)

An open-loop, file-provided action sequence (Test E). Segment kinds:
`edge_pose` (repeat one absolute target `steps` times; the env's limits bound
each step), `edge_line` (absolute targets on the straight line `from` -> `to` at
`speed_m_s`, orientation slerped), `edge_delta`, `hold` and
`hold_until_time_limit`; each segment has a `label`, `cut_enabled` and optional
`cut_index`. `dataset.expand_action_program(program, dt, steps)` expands it.
`scripts/make_phaseG_action_programs.py` writes the two Phase G programs from
declared scene/tool/goal numbers only (no planner code);
`scripts/preflight_action_program.py` reproduces the bounded commanded path and
runs the C2 IK + collision preflight.

## Multi-episode reliability (`POLICY_READY`)

`dataset.reliability_report` applies the predeclared
[`phaseG_reliability_v1`](../configs/wp11/phaseG_reliability_v1.json) checks.
Per episode, after the reset: clock at 0, labels clean, material at its
reference positions and at rest, start codes identical to episode 0, oracle
history / lifecycle / IK counter / termination latch / step counters / action
log reset, tool pose equal to episode 0 within 0.1 mm / 1 mrad and at rest;
live only: grasp re-stabilized (`PASS_STABLE_GRASP`) and current camera frames.
During the episode: action and observation clocks consistent, no privileged key
in any observation, bounded camera age (live). After it: evaluated, declared
expectation met, recorder closed, stored episode validates. Across episodes: the
declared number of consecutive episodes in one process without an execution
failure (kinematic 5, Native 3), no host-RSS or process-GPU-memory growth
(least-squares slope from episode 1 at most 64 / 32 MiB per episode), same seed
and controller repeatable (exact final codes on CPU; verdict otherwise), and at
least one failed episode stored and usable. `POLICY_READY` is set only if every
check passes. It is an API/lifecycle/storage reliability statement, not a claim
about task success or policy quality.

## Runtime profiles (WP-11 Phase H4, opt-in)

`open_cutting_env(..., preset="cutting_research_v1", runtime_profile=<id>)` and
`scripts/collect_dataset.py --runtime-profile <id>` select a declared profile of
[`configs/wp11/runtime_profiles_v1.json`](../configs/wp11/runtime_profiles_v1.json)
(`runtime_profiles.py`). Without `runtime_profile` nothing changes: legacy calls and
the plain `cutting_research_v1` preset keep their defaults (Native `stage_reset="reopen"`).

| Profile | `acceptance_valid` | Material numerics | Native | Cameras / recording |
| --- | --- | --- | --- | --- |
| `full_physics_v1` | **true** (the only acceptance profile) | scene solver at the formal resolution, **required**: 1 mm particles, 4 mm grid, 0.1 ms material dt, 20 ms control | `stage_reset="reload_in_place"` | as declared by the preset and caller (not reduced); recording every 10 steps; oracle history every step |
| `research_fast_v1` | **false** on every episode | scene solver override 2 mm particles, 8 mm grid (recorded in the task id), 0.2 ms material dt | `stage_reset="reload_in_place"` | wrist D405s only (`include_global_camera=False`), policy frames every 5 steps (10 Hz), recording every 25 steps (2 Hz) |

- A profile fixes its numerics: passing `particle_spacing_m`, `material_dt`, `dx`
  (and for `full_physics_v1` `history_stride`) as well raises `ValueError`;
  `full_physics_v1` also refuses a scene whose solver is not 1 mm / 4 mm.
- Every reset info, `task_spec()`, `evaluate()` result, `episode.json` and
  `index.json` row carries `runtime_profile` and `acceptance_valid`.
- **`research_fast_v1` fidelity limits** (F3 resolution study,
  [WP11_PHASE_F_LAW8.md](reports/WP11_PHASE_F_LAW8.md)): TRIM removal 0.816
  (coarse) vs 0.956 (formal) on TRIM-02-K2-R10, so the >= 0.9 criterion can fail;
  CUT interface location 3.0 mm (its ideal-partition floor) vs 1.5 mm, so the frozen
  2 mm location bound cannot pass; contact forces are neither dt- nor
  resolution-converged. Use it for policy iteration and pipeline debugging only,
  never for acceptance, benchmarks or calibration.
- Measured speedups and per-stage timings: [WP11_PHASE_H_PROFILING.md](reports/WP11_PHASE_H_PROFILING.md).

### Native stage reset (`IsaacSimCuttingConfig.stage_reset`, WP-11 Phase H1)

`"reopen"` (default) exports and opens the workcell at every reset; Kit retains
about 55-66 MiB of host memory per `open_stage` of this workcell (not returned by
`malloc_trim`, no Python objects or USD layers retained), which made Native fail the
memory gate. `"reload_in_place"` keeps the open stage after the first reset: it
releases the previous cameras (annotators and render products), tensor views and
PhysX objects (`reset_simulation`, detach, attach on the SimulationManager stage-update
interface), clears the session layer, reloads the root layer from disk
(`renew_stage_layers`, unit-tested against a fresh open) and then authors the reset
exactly as after an open. It requires `camera=True` and `physical_grasp` (with
`camera=False` the second reset segfaults inside PhysX `start_simulation` in
Isaac Sim 6.2). Every post-reset state is bitwise identical to reset 0 and to a
Phase G reopen reset. The residual growth is not zero (see the report).

### Profiling (`robotic_cutting_sim.profiling`, opt-in)

`scripts/collect_dataset.py --profile stages` writes `episode_###/profile.json`
(per-stage mean / max per control step, share of the control-step time, episode-level
stages). `--profile kernels` additionally times every Warp kernel eagerly (graph
disabled, a device sync around each launch; perturbing, use for the split only).
Nothing in the simulator imports the profiler; with no profiler installed the code
paths are unchanged (tested bitwise on CPU).

### Process recycling (`collect_dataset.py --recycle-every N`, opt-in mitigation)

Without the flag every episode runs in the collector's one process (unchanged). With
`--recycle-every N` the collector process becomes a parent: it creates the dataset
directory, writes the manifest (`config.process_recycling = {every, blocks}`), runs the
Native C2 preflight once, then runs every block of N consecutive plan episodes in a
fresh child process (`recycle/block_###.json` spec, `runtime_block_###/` runtime
receipts, `recycle/block_###_result.json`). Episode ids, seeds (fixed and the
`--seed-base` random stream, drawn once over the whole plan) and the index stay
continuous; each child opens the env with `recording["episode_index_start"]` (opt-in
`PolicyReadyEnv` field, default 0) so the env's own episode index matches. Each
`episode.json` carries `recycle = {block, every, block_episode, pid, runtime_dir}`. A child
that dies leaves its unfinished episodes as `FAIL_EXECUTION` rows; the parent stops after
two consecutive failed blocks. The parent validates and scores the whole dataset; the
reliability report then also contains `memory.per_process` (per-block series, slopes and
peak RSS). A reliability config may opt in to `memory.trend_scope = "per_process"` (gate
on the largest within-process slope); every existing config keeps the whole-run fit.
Recycling bounds per-process growth by restarting the process; it is a mitigation, not
evidence that the runtime does not leak. Tested on the kinematic CPU backend
(`tests/test_collector_recycling.py`: 5 episodes in 3 processes are bitwise equal to the
one-process collection, `read_dataset.py` PASS). Each child writes its block result after every
episode inside the env context, because on Native the env close (Kit `SimulationApp.close`) ends
the process. **Native status: not demonstrated.** The declared Native run
`native_recycle_mitigation_001` (at `4925c60`, before that fix) stored all 4 episodes but lost both
block results, so they were indexed as `FAIL_EXECUTION` (MITIGATION_NOT_DEMONSTRATED); the fixed
collector (`16f0ffa`) has not been run on Native (NOT_RUN).

## Phase G results (2026-10-30)

Full receipts: [phaseG_policy_ready_001](evidence/wp11/phaseG_policy_ready_001.json).

| Run | Episodes | Result |
| --- | --- | --- |
| Kinematic Test F, 4 mm, CPU (`kin_reliability_trim_4mm_001`) | 7 in one process: scripted (fixed and random seed), action program, deliberate failure, random policy with invalid actions, two exact repeats | **POLICY_READY** (v1 and v1a); 7/7 validate; 7/7 kinematic replays reproduce the final codes exactly; repeats bitwise identical; RSS slope 1.9 MiB/episode |
| Kinematic Test E, 1 mm, `trim_bump_rho10` | scripted vs hand-written program | both PASS: dome 97.89% vs 97.51%, substrate 0.99996 |
| Kinematic Test E, 1 mm, `sever_mid` | scripted vs hand-written program | both PASS: two 50/50 pieces |
| Native Tests E/F, 1 mm, GPU 0 (`native_reliability_trim_001`) | 4 in one process, each with physical return home | scripted (measured-edge feedback) PASS 97.89%; hand-written open-loop program PASS 98.47% (no privileged input); deliberate failure FAIL as declared (dome 0%), stored and valid; repeat PASS 97.70%. Reset bitwise repeatable, grasp re-stabilized every reset. **Not POLICY_READY**: host RSS grows 140 MiB/episode (gate 64) |
| Native memory diagnostic (`native_memory_probe_001`) | 8 x 20 steps | host RSS +82 MiB per reset with no plateau, GPU memory flat: the growth is in the Native runtime reset, not the wrapper |

`phaseG_reliability_v1a` amends only the post-reset tool-velocity rule
(declared after the Native smoke, before the formal run): PhysX reports an
instantaneous velocity of 3.6 mm/s and 0.148 rad/s for the clamped tool while
the arm is at rest and the pose-derived settle-window speeds are 6e-5 m/s; v1a
requires that velocity to repeat episode 0. Under v1, Native also fails
`tool_at_rest_after_reset` on every episode; both are reported.

## Phase H Native status (2026-10-30)

Full receipts: [phaseH_profiling_001](evidence/wp11/phaseH_profiling_001.json), report
[WP11_PHASE_H_PROFILING.md](reports/WP11_PHASE_H_PROFILING.md).

| Run | Result |
| --- | --- |
| Native Test F rerun, clean worktree `ecc77de`, `full_physics_v1` (`native_reliability_trim_full_physics_001`) | same 4-episode plan: PASS / PASS / FAIL (as declared) / PASS (repeat with identical final codes), home PASS x4, every reset bitwise equal to episode 0. Host RSS slope **56.5 MiB/episode** (PASS, gate 64; Phase G 140). **Not POLICY_READY**: GPU slope 52 MiB/episode over 3 non-monotonic points (gate 32). v1 additionally fails the known `tool_at_rest_after_reset`. |
| Native memory diagnostic with `reload_in_place` (`native_memory_probe_inplace_001`, 8 x 20 steps) | host 24.2 MiB/episode (Phase G 82.0), GPU 9.3; reset checks bitwise equal to episode 0 and to a Phase G reopen reset |

Native therefore stayed **NOT POLICY_READY** after the Phase H rerun (superseded by v1b below); the
kinematic backend remains POLICY_READY. Host growth is reduced, not eliminated.

## Native status: POLICY_READY under v1b (2026-10-30)

Full receipts: [phaseH_native_reliability_v1b_001](evidence/wp11/phaseH_native_reliability_v1b_001.json),
report section "Native reliability v1b" in [WP11_PHASE_H_PROFILING.md](reports/WP11_PHASE_H_PROFILING.md).

[`phaseG_reliability_v1b`](../configs/wp11/phaseG_reliability_v1b.json) keeps every v1a gate and
threshold and declares (before the run) 8 consecutive full Native episodes in one process
(the Test F plan cycled twice, `full_physics_v1` with `reload_in_place`) and the memory estimator
(unweighted least-squares slope over episodes 1..7).

| Run | Result |
| --- | --- |
| `native_reliability_v1b_001` (clean worktree `a65e71d`, GPU 0, 8 full episodes with physical return home, 5600 s) | PASS / PASS / FAIL (as declared) / PASS, twice; 8/8 stored episodes validate (`read_dataset.py` PASS); every reset bitwise equal to episode 0 and `PASS_STABLE_GRASP`. Host RSS slope **48.8 MiB/episode** (gate 64), process GPU slope **2.0 MiB/episode** (gate 32). **v1b POLICY_READY = true**, v1a true (identical scoring), v1 false (`tool_at_rest_after_reset` only, the known clamp jitter) |

**Native (Isaac Sim) is POLICY_READY** under v1b for this task and configuration
(`cutting_research_v1`, `full_physics_v1` + `reload_in_place`, cameras on, physical grasp). It is
an API/lifecycle/storage reliability statement, not "no leak": the host still grows about 50 MiB per
full episode, so long single-process collections should use `--recycle-every N` (mitigation, above).
With the default `stage_reset="reopen"` (no runtime profile) Native was not shown POLICY_READY
(Phase G: 140 MiB/episode).

## Scope and limits

- Runtimes: kinematic and Isaac Sim Native. Isaac Lab with the preset raises
  `NotImplementedError` (NOT_RUN).
- The per-step collision check excludes robot links and cameras; the C2
  preflight covers them for planned paths and action programs only, not for
  arbitrary online policy actions (a live policy relies on the runtime IK and the
  physics).
- One compiled task per process on Native (the runtime config carries the
  configured task); different goals need separate processes.
- `stage_reset="reload_in_place"` requires live cameras (camera-off in-place resets crash PhysX).
- The kinematic backend raises `ValueError` for `slots_per_node` unless `solver="tlmpm_v5"`
  (it used to ignore it silently).
- Material parameters are provisional and uncalibrated.
