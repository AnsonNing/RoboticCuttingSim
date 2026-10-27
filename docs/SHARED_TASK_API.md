# Shared live cutting API

`CUT-01` and `TRIM-01` use the same `CuttingTaskEnv` API over actual Isaac Lab or
Isaac Sim backends. The wrapper implements no cutting physics or robot dynamics.
Task definitions, fixtures, material and workpiece-local goals are shared.
The second Franka remains parked. No model training is included.

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

`observe()` includes only two RGB arrays, optical-Z depth in metres, valid masks,
camera calibration and time. The default `vision_proprio` profile additionally
provides measured `q`, `qdot` and the knife-edge pose. `vision_only` omits those.
Force, material particles, cut labels, target-plane oracle and Global images
remain outside formal observations. Global and depth debug views are available
through `render(camera, mode="rgb_array" | "depth_array")`.

`evaluate()` independently recomputes task metrics from actual reference MPM
stencils, persistent codes, immutable masses and attributed current surfaces.
Episode completion does not imply task success. The placeholder step reward is
zero; future learning code should add an explicit reward adapter. The evaluator
is privileged and separate from policy observations. Reset rebuilds particles,
stress, plasticity and cut history, resets articulated state and clears evaluator
history; strict camera/numerical repeatability has separately recorded limits.

Current material parameters and fracture behavior remain provisional. The
reaction is applied with an explicit20ms coupling delay. Lab kit-less cameras
are ideal; Kit uses actual native sensors. Camera and complete-physics acceptance
are audited separately from the task's `PASS_TASK_METRICS` result.
