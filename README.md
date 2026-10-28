# RoboticCuttingSim

A physical cutting research environment with two Franka Panda arms: continuous soft material, a fixed tool and fixtures, persistent TLMPM material separation, two simulated RealSense D455 RGB-D streams, a Global Camera, and shared control and task-scoring APIs. The second Franka is currently parked; no model is trained in this phase.

**CUT-01 and TRIM-01 have each completed actual 400-step / 8-second runs in both Isaac Lab 3.0 GA and Isaac Sim 6.2, producing persistent material separation.** Actual Franka joint and end-effector control moves the tool; the measured blade drives the material, and material reaction feeds back to the arm. No particles are deleted and no bumps are presegmented.

| Fixed task | Actual physical result, one episode each in Lab and Sim | Independent task score |
| --- | --- | --- |
| CUT-01: completely sever a 160×30×20 mm bar | Continuous 144 g bar → two 72 g parts; cut-location error upper bound 1.5 mm | **PASS_TASK_METRICS** |
| TRIM-01: trim a continuous single-bump surface | Explicit 2 mm particle configuration; 1.608 g chip; 98.53% of initial bump material removed | Persistent separation passes; strict full-surface accuracy **FAIL_TASK_METRICS** |

Both fixed tasks preserve mass and particle count, with separation persisting for 2.4 seconds after withdrawal. The original 1 mm TRIM did not fully disconnect; its failure and the extended-path and second-cut failures are preserved. Material and fracture parameters are not calibrated against real samples, and resolution or coupling-energy convergence has not been established. Full-benchmark surface accuracy, coupling work, native-depth tails and camera reset still include FAIL results; completing an episode does not establish complete acceptance.

![TRIM-01: live Isaac Sim 6.2, dual-Franka bar-cutting scene, t=4.8s](docs/evidence/trim01_live_kit.png)

The videos below are encoded directly from **native camera arrays captured during live Isaac Sim execution**, with 41 frames per episode and accompanying source and SHA256 records. They are not USD replay.

| Task | Global | Dual RGB | Dual measured depth | Synchronized views / trajectory / provenance |
| --- | --- | --- | --- | --- |
| CUT-01 | [Video](docs/media/cut01/global.mp4) | [D455 0](docs/media/cut01/d455_0_rgb.mp4) / [D455 1](docs/media/cut01/d455_1_rgb.mp4) | [D455 0](docs/media/cut01/d455_0_depth.mp4) / [D455 1](docs/media/cut01/d455_1_depth.mp4) | [Synchronized](docs/media/cut01/synchronized_multiview.mp4) / [Commanded vs Actual](docs/media/cut01/trajectory.png) / [Provenance](docs/media/cut01/provenance.json) |
| TRIM-01 | [Video](docs/media/trim01/global.mp4) | [D455 0](docs/media/trim01/d455_0_rgb.mp4) / [D455 1](docs/media/trim01/d455_1_rgb.mp4) | [D455 0](docs/media/trim01/d455_0_depth.mp4) / [D455 1](docs/media/trim01/d455_1_depth.mp4) | [Synchronized](docs/media/trim01/synchronized_multiview.mp4) / [Commanded vs Actual](docs/media/trim01/trajectory.png) / [Provenance](docs/media/trim01/provenance.json) |

## Run both scenes

Use the isolated environments already installed locally and a fresh output directory; installation and versions are documented in [SETUP](docs/SETUP.md), [Isaac Lab](docs/ISAAC_LAB_INTEGRATION.md) and [Isaac Sim](docs/ISAAC_SIM_VALIDATION.md). These commands actually advance the robot and material, saving measured trajectories, physics traces, RGB-D, Global views and independent scores.

```bash
# Isaac Sim 6.2 / PhysX / native SingleView RGB-D
.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id CUT-01 --steps 400 --gpu 1 --device cuda:1 \
  --camera-mode native_single_view --output-dir artifacts/cut01_fresh

.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id TRIM-01 --particle-spacing 0.002 --steps 400 --gpu 1 --device cuda:1 \
  --camera-mode native_single_view --output-dir artifacts/trim01_fresh

# Run the same Task in Isaac Lab 3.0 / NewtonManager; Kit-less cameras use ideal Warp
.deps/isaaclab-venv/bin/python scripts/run_isaac_lab_cutting.py \
  --task-id TRIM-01 --particle-spacing 0.002 --steps 400 --reset-check \
  --output-dir artifacts/lab_trim01_fresh

# Run fixed and randomized scenes for both tasks sequentially; no training
.venv/bin/python scripts/run_cutting_task_benchmark.py \
  --runtime isaac_sim --variants fixed randomized --seeds 42 --gpu 1 \
  --output-dir artifacts/task_matrix_fresh
```

CUT uses 1 mm particles, and successful TRIM uses explicit 2 mm initial sampling; both material grids are 4 mm. The original API/Task default remains 1 mm; specify `--particle-spacing 0.002` to reproduce successful TRIM. Current episode times are approximately 86–253 seconds in Sim and 197–458 seconds in Lab, excluding first-start costs and hardware-load effects; real-time performance is not guaranteed.

Randomization supports different lengths, local cut positions, static workpiece positions/yaw, and 1–3 continuous dome/Gaussian/ridge/irregular bumps. Native seed42 has been executed: CUT produces 66.6/59.4 g parts; TRIM's three bumps produce three chips and remove 96.28% of target material. This is a single-seed engineering validation, not a randomized-task success rate. See [Task benchmark](docs/CUTTING_BENCHMARK.md) and [randomization settings](docs/TASK_RANDOMIZATION.md).

## Shared research API

Both tasks share `reset()`, `step(action)`, `observe()`, `render()`, `close()` and `evaluate()`, supporting Cartesian blade-edge poses and joint-position actions. Official observations contain only dual RGB-D, calibration, timestamps and optional robot proprioception; force, material oracle and Global views remain separate debug/scoring data.

```python
from robotic_cutting_sim.live_env import open_cutting_env

# Use the selected runtime's Python; full settings are in SHARED_TASK_API.md
with open_cutting_env("TRIM-01", runtime="isaac_sim",
                      particle_spacing_m=0.002,
                      output_dir="artifacts/my_trim_api") as env:
    obs, info = env.reset(seed=42)
    for _ in range(400):
        obs, reward, terminated, truncated, info = env.step()  # scripted reference
    result = env.evaluate()  # Independent scoring; complete surface criteria may still FAIL
```

Object, Material, Fixture, Robot, Camera, Physics and Task Goal are separate; targets are defined in the workpiece-local frame. Future RL/IL/VLA/Agent work can use the corresponding control and scoring adapters with the existing cutting physics. Dynamic object holding, arbitrary new cut planes and coordinated dual-arm operation still require extensions.

Details: [shared API](docs/SHARED_TASK_API.md), [task definitions and success criteria](docs/CUTTING_TASKS.md), [actual episodes and negative results](docs/reports/CUTTING_TASK_EPISODES.md), [camera validation](docs/TASK_CAMERA_VALIDATION.md), [surface representation errors](docs/REPRESENTED_TARGET_VALIDATION.md), and [full progress](docs/PROGRESS.md).

## Regression checks and existing baselines

```bash
.venv/bin/python -m pytest -q
```

205 CPU tests pass, counted separately from actual GPU/Isaac episodes. Fixed-seed D0–D3 specimen-sampling hashes remain unchanged; an actual 5-step Lab D0 physics-prefix regression also completed, without claiming a full rerun. All historical standalone, D0–D3, Newton implicit MPM and CRESSim evidence is preserved; see the [original benchmark](docs/BENCHMARK_REPORT.md), [TLMPM validation](docs/reports/TLMPM_VALIDATION.md), [D0 live acceptance](docs/reports/ISAAC_LIVE_ACCEPTANCE.md) and [D0 video](docs/media/live_isaac_kit_d0/synchronized_multiview.mp4). Large raw data and official assets remain in local ignored caches.
