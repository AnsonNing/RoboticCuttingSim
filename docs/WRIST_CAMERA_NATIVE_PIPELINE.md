# Native wrist RGB-D pipeline

The explicit `wrist_d405_global_d455` layout runs two hand-mounted geometric
D405 prototypes and one world-fixed official D455 asset in live Isaac Sim 6.2.
Native PhysX supplies both actual hand poses. The retained TLMPM material and
dynamic surface are updated from the actual cutting edge; no robot or material
replay is used. The native CLI retains `legacy_fixed_d455` as its default.

| Recorded camera ID | Parent | Actual capture | Resolution | Public optical-depth range |
|---|---|---|---|---|
| `wrist_d405_0` | Actual cutting hand | `CameraSensor` RGB + geometric RTX depth; `ideal_d405_prototype` | 848×480 | 0.07–0.50 m |
| `wrist_d405_1` | Actual secondary hand | Same prototype capability | 848×480 | 0.07–0.50 m |
| `global_d455` | World | Official Isaac 6.2 D455 USD + native `SingleViewDepthCameraSensor`; provisional hardware model | 640×480 | 0.60–6.00 m |

The prototype does not simulate D405 disparity, stereo matching, noise, ISP,
physical calibration or interference. The right optical reference records the
18 mm baseline, without rendering a second imager. RGB and depth use the same
ideal optical render product. The Global sensor uses the separately controlled
native SingleView recipe, with exact-version registration and unmodified native
measured values. Its RGB annotator shares the pseudo-depth render product;
physical D455 color/depth hardware calibration remains unvalidated.

The shared rig is defined in
[`camera_rig.py`](../src/robotic_cutting_sim/camera_rig.py) and
the active [`wrist_camera_rig_v2.json`](../configs/wrist_camera_rig_v2.json).
Version1 remains available as the original farther-Global baseline. Housing and
brackets are visual geometry with no additional mass, inertia or contact model.
The native runner rejects active rigid-body or collision APIs under these camera
roots. Both wrist roots must be actual children of their corresponding Panda
hand, and the Global optical transform must remain fixed. The secondary arm
holds the declared observation posture, approximately hand position
`[0.20, 0.18, 0.34] m` with a downward hand orientation. Legacy layouts retain
their original parked secondary posture.

## Run and observation contract

```bash
.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id CUT-01 --steps 400 --record-every 10 \
  --camera-layout wrist_d405_global_d455 --camera-mode native_single_view \
  --gpu 1 --device cuda:1 --output-dir artifacts/native_wrist_cut_fresh

.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id TRIM-01 --particle-spacing .002 --steps 400 --record-every 10 \
  --camera-layout wrist_d405_global_d455 --camera-mode native_single_view \
  --gpu 1 --device cuda:1 --output-dir artifacts/native_wrist_trim_fresh
```

The new layout requires `camera_mode=native_single_view`, because Global must
use the native model. This does not make either D405 a native hardware model;
each frame and result retains its per-camera capability label. The explicit
2 mm TRIM sampling override changes the entire initial specimen and is recorded
in the task receipt; it does not change the default 1 mm task.

Raw recording saves all three sensors in `observations/`. The shared API's
default policy camera IDs are only `wrist_d405_0` and `wrist_d405_1`; Global is
an explicit `include_global_camera` opt-in. Saving its raw arrays does not imply
default policy inclusion. Force, cut codes, component labels and evaluator masks
are absent from observations. The old `global` camera remains debug-only in
the legacy layout.

## Current geometry, calibration and occlusion

After each 20 ms native control interval, the runner composes each **measured**
hand transform with its immutable hand-to-optical transform. It checks this
against the current USD optical prim, rejecting translation errors above
10 µm or rotation errors above 0.1 mrad. The check covers both arms even when
capture cadence is only one frame every ten control intervals. Capture then
checks Replicator's runtime camera view against both current USD and measured
composition. Rendering holds the native physics clock, step count, joints,
velocities and both measured hand poses unchanged.

Transforms use column vectors, meters, XYZW hand quaternions and OpenCV optical
axes: +X right, +Y down, +Z forward. Runtime `K` comes from the actual render
product, including its aspect fit. The canonical 848×480 pilot measures
`fx=fy=446.802759`, `cx=424`, `cy=240`, giving approximately 87°×56.4849°.
The Global native product measures `fx=fy=317.043120` at 640×480.

All new optical prims render from 1 mm to 20 m. The public sensor range is
applied **after the nearest first hit**. A gripper closer than 70 mm therefore
occludes material behind it and produces invalid prototype depth; it is not
clipped away to reveal that material. RGB retains the nearest visible geometry.
Native D455 processing retains the validated 0.52–6 m input recipe, followed
by the declared 0.60–6 m public range. Raw measurements and rejected first hits
are preserved separately; no ideal depth fills native holes.

## Recorded evidence

- `robot_state.npz`: every control-step state, including
  `hand_pose_both_xyzw` with shape `(steps+1, 2, 7)`.
- `camera_extrinsics.jsonl`: initial state and every control step, independent
  of image cadence. Terminal-reset samples use a separate reset file.
- `camera_pose_trace.json`: the same completed episode samples, saved before
  terminal reset. Each camera records `T_world_hand`, `T_hand_camera`,
  `T_world_camera`, `T_camera_world`, the current USD optical transform,
  discrepancy metrics and episode/native timestamps. World-camera hand fields
  are null.
- `frame_manifest.json`: each capture's full current runtime calibration,
  measured rig sample and runtime-versus-current-USD checks.
- `observations/<camera>_<frame>.npz`: RGB, optical-Z meters, validity, `K`,
  current extrinsics and timestamps. Wrist frames also retain hand/mount
  transforms. Prototype and unvalidated-physical-calibration labels are explicit.
- `camera_debug/`: raw RTX first-hit depth, reference validity and range-rejected
  masks. Global also retains native distance/point-cloud AOVs and registered
  values before public range filtering.
- `final_live_scene.usda`: the fully composed actual native scene before reset,
  including current hand-mounted cameras, fixtures and material. Its receipt
  records timestamp and SHA256 for independent whole-scene first-hit audits.

`calibration_runtime.json` is the latest calibration and may describe the
terminal reset. Historical frames must use their own manifest or observation
matrices. `calibration_authored.json` is not a substitute for runtime projection.

## Validation boundary

The source-frozen canonical pilot
`artifacts/isaacsim_wrist_rig_smoke_002` passes two live control steps, twenty
PhysX substeps and three captures. It records all three camera IDs, three
20 ms pose samples and the composed final scene. Measured-versus-USD wrist
translation errors are at most 63.7 nm and 4.4 nm; Global drift is zero in
that short recording. This is `PASS_SMOKE`, not full task or sensor acceptance.

The subsequent source-frozen v2 pilot
`artifacts/isaacsim_wrist_rig_smoke_003` uses corrected D405 housing dimensions
and moves Global to `[1.10, -1.20, 1.05] m`, looking at `[0, 0, 0.25] m`.
Its three frames retain 15.908–15.913% valid native Global depth, compared with
approximately 6.36% in the farther-view pilot. Independent whole-scene CPU
audit `artifacts/wrist_camera_native_pilot003_audit_001` passes measured
kinematics, the unchanged engineering visibility/optics checks and conservative
visual clearance. The first Global frame contains 122 depth-consistent material
pixels and 12 knife pixels, above the existing 25/8 thresholds. All three frames
pass those checks. A rare approximately 0.85 m valid native error remains in the
receipt; no native tail is filtered and full sensor acceptance is not scored.

The native worker verifies every frozen source and snapshot before application
startup and again before live reset. A mismatch writes an explicit integrity
failure and stops. An earlier VGA pixel-aspect trial is preserved as
`NONCANONICAL_SOURCE_RACE` in
`artifacts/isaacsim_wrist_pixel_aspect_smoke_001`; it cannot establish canonical
camera validation. The final 848×480 pilot uses matching source bytes.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_native_camera_geometry.py tests/test_native_source_freeze.py \
  tests/test_isaac_sim_runner.py
```

These thirteen CPU tests include rejection of frozen moving-camera extrinsics,
both-hand frame composition, inverse transforms, active camera contacts and
source/snapshot startup races. Full CUT/TRIM task metrics, whole-trajectory
clearance, native depth holes, reset repeatability and the separate actual
secondary-arm motion probe require their own episode/audit receipts.

## Completed v2 native episodes and secondary-arm motion

Both final v2 native recordings complete 400 control intervals, 4,000 PhysX
substeps, 8 simulated seconds, 401 actual dual-hand/camera pose samples and 41
three-camera capture sets. Runtime and rig sources stay frozen through both
episodes; native source-integrity checks pass.

| Native recording | Physical outcome | Shared task metrics |
|---|---|---|
| `artifacts/isaacsim_wrist_cut01_002` | Two 72 g parts; graph 1→2; 2.4 s post-cut persistence; zero history rewrites | `PASS_TASK_METRICS` |
| `artifacts/isaacsim_wrist_trim01_001` | 1.608 g detached chip and 144.024 g substrate; 98.5294% reference cap mass disconnected; 2.4 s persistence; zero history rewrites | `FAIL_TASK_METRICS`: surface p95/RMS, gouge and overcut criteria remain failed |

The loop/evaluation/reset times are 256.89 s and 91.81 s respectively, while the
other GPU also runs Lab work; these are not isolated performance measurements.
Task metrics do not promote native hardware sensor calibration or complete
benchmark acceptance. The first farther-Global CUT recording remains preserved
as `artifacts/isaacsim_wrist_cut01_001`.

The independent motion recording `artifacts/isaacsim_wrist_motion_probe_001`
then runs 100 actual controls/1,000 PhysX substeps over 2 s, producing 101 per-step
pose samples and 21 three-camera captures. The declared secondary observation
posture receives only a joint 1 sinusoid of 0.08 rad amplitude and 2 s period; it
is disabled in the complete task baselines. Actual arm1 joint excursion is
0.081217 rad, actual hand translation 21.74 mm, and wrist1 optical translation
28.11 mm with 0.081440 rad rotation. The Global optical transform has exactly
zero translation and rotation drift in this recording.

For all 21 captures, native runtime wrist1 extrinsics match the separately
recorded actual PhysX hand composition within 15.7 nm translation. Every 20 ms
USD optical sample agrees within 17.2 nm. Both source-integrity checks pass.
This proves actual articulation-driven wrist-camera movement without writing
commanded camera world poses. The recording remains `PASS_SMOKE`, with task
metrics `NOT_RUN` for its incomplete 2 s task. The compact input hashes,
source verification and per-camera numerical evidence are in
[`native_wrist_motion_proof.json`](evidence/native_wrist_motion_proof.json).

```bash
.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id TRIM-01 --particle-spacing .002 --steps 100 --record-every 5 \
  --camera-layout wrist_d405_global_d455 --camera-mode native_single_view \
  --observation-motion-probe --gpu 1 --device cuda:1 \
  --output-dir artifacts/native_wrist_motion_fresh
```
