# Shared live cutting API

`CUT-01` and `TRIM-01` use the same `CuttingTaskEnv` API over actual Isaac Lab or
Isaac Sim backends. The wrapper implements no cutting physics or robot dynamics.
Task definitions, fixtures, material and workpiece-local goals are shared.
The second Franka holds its normal safe park in camera rig v5; both wrist
cameras are axis-aligned with their own hand. The second camera has no
material-visibility requirement.
No model training is included.

The factory now defaults to `camera_layout="wrist_d405_global_d455"`:
`wrist_d405_0` and `wrist_d405_1` are articulated hand-mounted streams;
`global_d455` is World-fixed. Both wrists are `ideal_d405_prototype`, with
no validated stereo/noise/hardware calibration. The Kit Global uses the
official D455 asset and native SingleView pipeline; the Lab route is ideal
Warp. Legacy fixed cameras remain available with
`camera_layout="legacy_fixed_d455"`. See [camera details](WRIST_CAMERAS.md).

The final named-camera API smokes use two actual control steps in each runtime,
explicit Cartesian and joint actions, three synchronized saved observations,
Global rendering, independent evaluation and clean close. Lab/CUT tests the
default two-wrist observation; native Sim/TRIM tests
`include_global_camera=True`. Both retain `NOT_RUN` task success for the
unfinished episode. Sources and measured payload checks are in
[the wrist API receipt](evidence/wrist_task_api_smoke.json).

The following fixed-camera smokes are preserved historical evidence.

Actual API validation on2026-10-28: Lab/CUT-01 and native Kit/TRIM-01 each ran
reset, explicit Cartesian and joint actions, two synchronized RGB-D observations,
Global rendering, independent evaluation and close. Both passed the2-step smoke;
the unfinished cutting goal correctly remains `NOT_RUN`. CPU wrapper contracts
separately check observation isolation, reset seeds and stale-camera rejection.
Receipts and source hashes: [shared_task_api_smoke.json](evidence/shared_task_api_smoke.json).

Final launcher checks also pass actual two-step Kit/TRIM with explicit2 mm
particles and Lab/CUT with the final reset behavior. They exercise both action
modes and retain process/source receipts in
[task_regression.json](evidence/task_regression.json). These are API smokes,
separate from the400-step physical recordings.

Use the dedicated runtime environment, and a fresh output directory:

```bash
# Full reference episode, actual Kit/PhysX and native SingleView RGB-D.
.deps/isaacsim-venv/bin/python scripts/run_cutting_task.py \
  --runtime isaac_sim --task CUT-01 --output-dir artifacts/my_cut_api

# Same API, actual Lab/InteractiveScene; explicitly ideal Warp cameras.
.deps/isaaclab-venv/bin/python scripts/run_cutting_task.py \
  --runtime isaac_lab --task TRIM-01 --particle-spacing 0.002 \
  --output-dir artifacts/my_trim_api

# Two-step actual Cartesian + joint control smoke, not a cutting-success test.
.deps/isaaclab-venv/bin/python scripts/run_cutting_task.py \
  --runtime isaac_lab --task CUT-01 --steps 2 --exercise-actions \
  --output-dir artifacts/my_api_control_smoke
```

For complete raw physics, surfaces, camera streams, videos and independent
audits, use the recording commands in [CUTTING_TASKS](CUTTING_TASKS.md).
The API CLI preserves actual observation arrays, Global images, measured tool
trajectory and an independent task score. It checks worker receipts because Kit
can return process code zero after an exception.

The original task factory retains 1 mm particles. The successful live TRIM
reference episodes explicitly use `--particle-spacing 0.002` (Python:
`particle_spacing_m=0.002` in `open_cutting_env`). This changes initial quadrature,
not particle deletion during cutting. The retained 1 mm TRIM baseline does not
fully separate. With 2 mm particles, both Lab and native Kit detach a 1.608 g chip
and remove 98.5294% of above-plane reference material. The original strict
full-surface goal still returns `FAIL_TASK_METRICS` / `success=False`;
this is independent of physical separation and episode completion. See the
[actual episode audit](reports/CUTTING_TASK_EPISODES.md).

## Python

Run one runtime per process, using its dedicated Python. For direct Kit use,
set `OMNI_KIT_ACCEPT_EULA=YES`, as the existing recording launcher does.
Use `PYTHONPATH=src` when the project is not installed in that isolated environment.

```python
from robotic_cutting_sim.live_env import open_cutting_env

with open_cutting_env("CUT-01", runtime="isaac_sim",
                      include_global_camera=False,
                      output_dir="artifacts/my_reference") as env:
    observation, info = env.reset(seed=42)
    for _ in range(400):
        observation, reward, terminated, truncated, info = env.step()
    result = env.evaluate()
    print(result["status"], result["success"])
    global_rgb = env.render("global")
```

For randomized tasks, pass `task_variant="randomized"` through the same factory's
backend options. `reset()` preserves the constructed task seed. Native Kit can
rebuild the stage for another `reset(seed)`; Lab's geometry/fixtures are fixed at
construction, so a changed randomized seed explicitly requires a new environment.
The fixed variant stays geometrically identical across seeds. Distribution and
one-seed actual validation are recorded in [TASK_RANDOMIZATION](TASK_RANDOMIZATION.md).

`step(None)` executes the reference trajectory. An external controller supplies:

```python
env.step({"mode": "cartesian_edge_pose_world",
          "target": [x, y, z, qx, qy, qz, qw], "cut_enabled": True})
env.step({"mode": "joint_position_rad",
          "target": joint_positions_7, "cut_enabled": True})
```

Cartesian poses describe the **knife edge in world coordinates**, in metres and
unit XYZW. Joint positions are radians and must respect the actual articulation
limits. A bare seven-vector uses the configured `action_mode`. Mapping actions
can explicitly select either mode. External actions arm cutting independently
of the reference time window; the retained single-cut history stays permanent.
This version supports one cut plane per episode; arbitrary repeated cut planes
and moving workpiece frames need further solver work.

`observe()` defaults to two wrist RGB arrays, optical-Z depth in metres, valid
masks, camera calibration and time, keyed by `wrist_d405_0` / `wrist_d405_1`.
Each wrist image is 848×480; Global is 640×480. The default `vision_proprio`
profile additionally provides measured `q`, `qdot` and the knife-edge pose.
`vision_only` omits those. Set `include_global_camera=True` to add Global RGB-D
and calibration under `global_d455`. Force, material particles, cut labels and
target-plane oracle always remain outside formal observations.
Global and individual depth views are also available through
`render(camera, mode="rgb_array" | "depth_array")`; `"global"` aliases
`"global_d455"` in the new layout.

Each step checks measured hand / fixed-mount composition. Camera, calibration,
surface and robot timestamps must agree; stale or nonfinite camera/calibration
timestamps are rejected. Raw recordings preserve all three sensors regardless
of formal observation inclusion. Full measured transforms, native runtime
intrinsics and prototype/physical model labels remain distinct.

`evaluate()` independently recomputes task metrics from actual reference MPM
stencils, persistent codes, immutable masses and attributed current surfaces.
Episode completion does not imply task success. The placeholder step reward is
zero; future learning code should add an explicit reward adapter. The evaluator
is privileged and separate from policy observations. Reset rebuilds particles,
stress, plasticity and cut history, resets articulated state and clears evaluator
history; strict camera/numerical repeatability has separately recorded limits.

Current material parameters and fracture behavior remain provisional. The
reaction is applied with an explicit20ms coupling delay. Lab kit-less cameras
are ideal; Kit wrists use ideal geometric RTX and Global uses native SingleView.
Camera and complete-physics acceptance
are audited separately from the task's `PASS_TASK_METRICS` result.
# Evaluation v2 and parked wrist camera update

The current camera rig is v5: Franka 1 holds its normal park posture and both
D405s have axis-aligned fixed hand-local mounts. Secondary material/knife visibility is not
a task success or execution condition. Historical camera protocols remain
versioned separately; see [v5 validation](WRIST_CAMERA_V5.md).

`env.evaluate()` preserves the original v1 score. The optional
`env.evaluate(profile="diagnostic" | "research" | "precision")` returns
Evaluation v2's separate episode, recorded-state integrity, outcome and geometry
results. Live values have explicit incomplete source-audit status; the offline
recording evaluator verifies archived bytes and physics/mesh provenance.
Evaluation never stops physics or prevents saving a failed episode. See
[the new API and profiles](EVALUATION_V2.md).

# Configured tasks, HOME and physical return-to-home (WP-11 Phase D)

Isaac Sim accepts a compiled scene/goal task (ADR 0015). The runner rebuilds it
from its recorded scene, goal and planner config and requires the same
`CFG-...` id. Fixtures, tool, material solver, cutting law and number of cut
digits all come from the scene; `tlmpm_v5` is allowed up to its digit limit.
Isaac Lab still raises `NotImplementedError` for compiled tasks.

```python
from robotic_cutting_sim.scene_config import compile_task
task = compile_task("trim01_wp11_workcell", "trim_bump_rho10")
with open_cutting_env(runtime="isaac_sim", task=task, action_limits=task.action_limits(),
                      output_dir="artifacts/my_configured_run") as env:
    observation, info = env.reset(seed=42)        # READY, Franka 0 at HOME
    ...                                           # env.step(...) while RUNNING
    env.end_task("CONTROLLER_DONE")               # TASK_COMPLETE (not a success claim)
    task_outcome = env.evaluate()                 # the goal evaluator
    home = env.return_home()                      # physical motion, separate home_outcome
```

`scripts/run_wp10_live.py --scene <scene> --goal <goal> --controller scripted`
drives the goal planner's waypoints with the planner's controller profile and,
with `--edge-feedback measured`, the measured tool-edge feedback. Before Isaac
starts, the plan is preflighted in the CPU `.venv` with the C2 collision checker
(`scripts/wp11_motion_preflight.py`). If the plan fails, the run is refused with
`SCENE_INFEASIBLE` or `IK_FAILURE`. `--stop-after-phase`, `--hold-after-stop-s`
and `--return-home` are opt-in.

**HOME reset.** Some scenarios declare `home_joint_positions`, for example
`wp11_workcell_v1` in `configs/wp11_scenarios_v1.json`, or a scene can name one
with `supports.workcell.from_scenario`. For these, reset puts Franka 0 at
q_home and authors the knife at FK(q_home)·hand_from_tool, with the blade
hanging below the fingers. It then runs the unchanged physical pregrasp settle.
Franka 1 stays at the shared park. `setup.json` records the result under
`home_reset`.

**Lifecycle.** `info["episode_state"]` is always present. Pass
`expose_episode_state=True` to also add it to the observation under `"episode"`.
It contains only the state, the termination reason and the home outcome. The
states run in this order:

`RESETTING → READY → RUNNING → TASK_COMPLETE | TASK_FAILED → RETRACTING →
RETURNING_HOME → FINISHED`

Illegal transitions raise. `reset()` is the simulator state reset; it is not
`return_home()`.

**Return to home.** `return_home()` runs these segments, in order:

1. a straight Cartesian lift to the HOME edge height;
2. a joint-space rotation to the hand-down orientation;
3. a joint-space move to q_home;
4. a settle.

The rules (`configs/wp11/return_home_v1.json`):

- The whole path is checked before anything moves: joint velocity and
  acceleration limits, the C2 geometric preflight including the held knife,
  and cutting disabled throughout.
- `home_outcome` (PASS / FAIL / NOT_APPLICABLE, with a reason) measures:
  - joint error to home;
  - tool pose error against the post-reset tool pose;
  - grasp slip and rotation;
  - native unintended tool contacts;
  - cut-label changes, which must be 0;
  - clearance of the measured joint path.
- The home outcome is never merged with the task outcome.
