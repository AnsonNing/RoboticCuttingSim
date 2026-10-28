# Live CUT-01 / TRIM-01 benchmark

Both tasks now execute physical cutting in actual Isaac Lab 3.0 GA and Isaac
Sim 6.2. A sequential matrix runner, common environment API, workpiece-local
targets, seeded variants and independent evaluators are implemented. This is an
executable research environment; the complete scientific acceptance is **not
all PASS**. No policy training is included.

The current camera layout is dual moving D405 prototypes plus a closer fixed
Global D455. Both tasks have also completed four full final-rig live episodes
(one per Task/runtime), preserving CUT Task PASS and coarse TRIM physical
separation / strict surface FAIL. FK/time/optics pass; secondary knife occlusion,
flange mounting overlap and untrimmed native depth tails have separate statuses.
See [current camera configuration](WRIST_CAMERAS.md),
[independent final-rig results](WRIST_CAMERA_VALIDATION.md) and
[current captured videos](../README.md). The table below preserves the earlier
fixed-camera results; those recordings are not relabelled as wrist-camera runs.

## Recorded full episodes

All entries below contain400 control intervals /8 simulated seconds. Masses are
immutable recorded reference masses with unchanged particle counts. Connectivity
is recomputed from actual8-node MPM supports, positive weights, stress gradients
and permanent cut history, including code-zero bridges.

| Recording under `artifacts/` | Runtime | Physical separation | Task v1 | Recorded loop wall time |
| --- | --- | --- | --- | --- |
| `isaac_lab_cut01_001` | Lab | PASS:72 +72 g | PASS | 458.14 s |
| `isaacsim_cut01_001` | Sim /native RGB-D | PASS:72 +72 g | PASS | 253.08 s |
| `isaac_lab_trim01_001` | Lab /1 mm particles | FAIL:one connected body | FAIL | 472.22 s |
| `isaacsim_trim01_001` | Sim /1 mm particles | FAIL:one connected body | FAIL | 271.05 s |
| `isaac_lab_trim01_coarse_001` | Lab /2 mm particles | PASS:1.608 g chip,98.5294% target removal | FAIL:surface precision | 197.31 s |
| `isaacsim_trim01_coarse_001` | Sim /2 mm particles | PASS:1.608 g chip,98.5294% target removal | FAIL:surface precision | 86.33 s |
| `isaac_native_randomized_benchmark_001/cut-01_randomized_seed42` | Sim /1 mm particles | PASS:66.6 +59.4 g | PASS | See recorded process receipt |
| `isaac_native_randomized_benchmark_001/trim-01_randomized_seed42` | Sim /2 mm particles | PASS:three chips,96.2791% target removal | FAIL:surface precision | See recorded process receipt |

The randomized matrix took316.08 s including both subprocesses and startup.
These are observed timings on the shared machine, not exclusive performance
benchmarks. Each recording retains its exact versions, source bytes/hashes,
seed, task configuration, numerical override and raw results.

Physical separation persists for2.4 s at the retained postcut sample times.
The randomized TRIM has0.672 /0.768 /1.044 g detached components and126.096 g
substrate. Its three reference bumps start continuous with the same base.
Contacts continue after the reference withdrawal interval, so separation does
not imply every chip is in blade-clear free flight. The one tested randomized
seed is not a success-rate estimate.

Independent full audits and negative experiments:

- [Fixed and coarse live episodes](reports/CUTTING_TASK_EPISODES.md).
- [Randomized native episodes](reports/RANDOMIZED_TASK_EPISODES.md).
- [Native cameras](TASK_CAMERA_VALIDATION.md), including strict failures.
- [Represented target](REPRESENTED_TARGET_VALIDATION.md): both analytic and
  represented-column goals FAIL; whole3D distances are descriptive only.
- [Original18-gate D0 acceptance](reports/ISAAC_LIVE_ACCEPTANCE.md).

## Reproduce

Use the dedicated runtime installations documented in
[Lab integration](ISAAC_LAB_INTEGRATION.md) and
[Sim validation](ISAAC_SIM_VALIDATION.md). Output directories must be fresh;
existing artifacts are never overwritten.

The benchmark launcher now defaults to `--camera-layout wrist_d405_global_d455`.
Pass `--camera-layout legacy_fixed_d455` to reproduce the earlier camera rig.
The physical material/task recipes are shared. The standalone recording
launchers retain their old defaults, so their wrist examples in
[WRIST_CAMERAS](WRIST_CAMERAS.md) select the new layout explicitly.

```bash
# Both fixed tasks, actual native Sim/PhysX/SingleView
.venv/bin/python scripts/run_cutting_task_benchmark.py \
  --runtime isaac_sim --variants fixed --seeds 42 --gpu 1 \
  --output-dir artifacts/fixed_matrix_NEW

# Both seeded variants, actual native Sim
.venv/bin/python scripts/run_cutting_task_benchmark.py \
  --runtime isaac_sim --variants randomized --seeds 42 --gpu 1 \
  --output-dir artifacts/random_matrix_NEW

# Same fixed matrix, actual Lab/NewtonManager; ideal Warp cameras
.venv/bin/python scripts/run_cutting_task_benchmark.py \
  --runtime isaac_lab --variants fixed --seeds 42 --gpu 0 \
  --output-dir artifacts/lab_matrix_NEW
```

The launcher explicitly selects CUT1 mm /TRIM2 mm particles, while retaining the
factory's original1 mm defaults. `--particle-spacing` explicitly overrides both
cases. Only `fixed` and `randomized` variants are accepted. Each case runs in its
own runtime process and retains stdout, a manifest, measured trajectory,
physical state/surfaces, cameras and evaluator output. The matrix reports
`episode_execution`, `physical_cutting`, `task_goal_v1` and `camera_acceptance`
separately. `COMPLETE` means the requested full episodes executed, not that all
task, camera or physics gates passed. Short `--steps` runs are smokes and cannot
pass final physical/task success.

For a controller adapter, use [the same environment API](SHARED_TASK_API.md):

```bash
.deps/isaacsim-venv/bin/python scripts/run_cutting_task.py \
  --runtime isaac_sim --task TRIM-01 --particle-spacing 0.002 \
  --output-dir artifacts/trim_api_NEW
```

Formal observations default to two wrist RGB-D views and optional measured
proprioception. Material state, force and cut labels remain outside observations;
Global RGB-D can be included explicitly with `include_global_camera=True` in the
API or `--include-global-camera` in its CLI, and is always recorded for review.
Current actions are Cartesian knife-edge poses or active-arm joint positions;
the second Franka holds an observation posture. The knife is a rigid
fixed tool transform, with simplified mounting geometry; grasp acquisition is
not modelled. Dynamic workpiece holding and arbitrary additional cut planes are
not implemented in this baseline.

## Quantitative scope and remaining failures

CUT v1 requires two substantial components, mass/history preservation, persistent
separation and a2 mm reference-support cut-location bound. TRIM additionally
requires90% target-material removal and frozen full-footprint surface limits.
Those evaluators use oracle data independently of observations. A completed
trajectory never establishes cutting success.

The preserved fixed-camera recordings produce2×640×480 RGB-D plus1280×960
Global. Their41-frame Task camera audits pass current geometry/time synchronization. Native material
coverage is approximately96.5–99.1%, with invalid samples and error tails retained.
The small TRIM chips remain below the frozen25-pixel visibility criterion in
the tested views; full CAM-04 and strict reset are not universally passing.
Lab's kit-less cameras are ideal and are never labelled native D455.

The final wrist layout instead records2×848×480 ideal D405 RGB-D plus640×480
native Global in Sim, with ideal Warp views in Lab. Its separate four-episode
audit and hardware-model limitations are in
[WRIST_CAMERA_VALIDATION](WRIST_CAMERA_VALIDATION.md).

TRIM's strict heightfield score includes boundary/missing rays. The nominal
20 mm worst deficit is a conservative missing-ray result, not evidence that the
knife gouged20 mm into the bar. Analytic/represented surface controls expose this
representation limitation without deleting bins or changing the failed gate.
Fine1 mm TRIM's two uncut bridge particles, the longer sweep, second pass and
2 mm-grid negative controls are retained. The successful coarser sampling does
not establish resolution convergence. Material calibration, coupling-work
convergence, native rigid-contact/joint-limit audit coverage and broad repeated
task statistics remain separate open work.

## Regression evidence

The current CPU suite has235 passing tests, distinct from the live runs. Meaningful
contracts cover seeded local frames/fixtures, original D0–D3 sampler hashes,
connectivity bridges, conservation, wrong-position/partial cuts, surface failures,
observation isolation, actions and receipt classification.

An actual five-control-step Lab D0 regression against the prior FF002 prefix
retains exactly the reference positions, masses and cut codes. Maximum material
position difference is52.15 nm; joint difference is5.04e−12 rad. Cameras were
disabled in this small physics check. Existing D0 camera checks and all original
D0–D3 artifacts remain; this is not a new full D0–D3 dynamics rerun. See
[the regression receipt](evidence/task_regression.json).
