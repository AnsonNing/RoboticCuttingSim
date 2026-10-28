# Dual wrist D405 and fixed Global D455

**Current: camera rig v5.** Both cameras use an axis-aligned fixed hand-local
mount. Franka 1 uses the normal safe park. No wrist-target aiming or observation
posture is used; secondary object visibility is advisory. See [v5 configuration
and validation](WRIST_CAMERA_V5.md) and the preserved [v4 checkpoint](WRIST_CAMERA_V4.md).
The v3 configuration and observation-arm experiments below remain historical.

The new layout is `wrist_d405_global_d455`. The old fixed cameras remain available
as `legacy_fixed_d455`; all prior recordings and scores are preserved. No policy
training is included.

The preserved v3 layout uses a **target-centred close Global view**, at
`(0.28,-0.42,0.46)` m looking toward `(0,0,0.02)` m, about0.67 m away.
This supersedes the earlier whole-workcell composition below. Two complete
new native episodes and two separate Lab smokes are reported in
[GLOBAL_CLOSE_VIEW](GLOBAL_CLOSE_VIEW.md); the latest videos are first in
[README](../README.md). The wrist mounts, optics and depth ranges are unchanged.

| Camera ID | Mount | Live Isaac Lab | Live Isaac Sim |
| --- | --- | --- | --- |
| `wrist_d405_0` | Actual Franka0 `panda_hand` child | Ideal Warp RGB-D | Actual RTX geometric RGB-D |
| `wrist_d405_1` | Actual Franka1 `panda_hand` child | Ideal Warp RGB-D | Actual RTX geometric RGB-D |
| `global_d455` | Fixed World transform, official D455 asset | Ideal Warp RGB-D from USD optics | Native SingleView D455 pipeline |

Both wrist backends are explicitly **`ideal_d405_prototype`**. Actual RTX rendering
does not validate D405 stereo disparity, noise, interference, ISP/colorimetry or
physical calibration. Global native D455 also retains its earlier physical
calibration/noise limitations. The backends never silently substitute ideal
depth for missing native measurements.

## Optics and asset provenance

The official Sim6.2 sensor catalog has no D405 entry; the checked conventional
D405 asset URLs returned404. The custom visual asset uses the manufacturer's
42 mm front width,42 mm front height,23 mm depth and18 mm stereo baseline. The left depth/color
imager supplies co-located ideal Depth/Color optical frames; a right optical
reference frame is18 mm along camera X, without a second sensor or stereo solver.
Official references and the exact asset checks are in
[the final geometry receipt](evidence/wrist_camera_rig_geometry_v2.json) and
[the shared configuration](../configs/wrist_camera_rig_v3.json). The original
marketing-axis interpretation is preserved with the first rig version; the
final axes follow the manufacturer's mechanical drawing/table.

The selected wrist profile is848×480, with square-pixel pinhole intrinsics:

```text
fx = fy = 446.802773 px; cx = 424; cy = 240
effective FOV = 87.000000 × 56.484916 degrees
logical depth range = 0.07 to0.50 m; invalid depth =0
RGB render clipping = 0.001 to20 m
```

This approximates the nominal87×58 degree product FOV. The manufacturer's70 mm
Min-Z table entry applies to848×480; it does not validate this ideal renderer's
physical depth performance. The retained640×480 prototype exposed a native RTX
aspect-fit change to about87×71 degrees. A pixel-aspect trial did not correct
the runtime projection, so that profile was not promoted. Actual render-product
K is recorded; authored USD lens values are never substituted for runtime K.

Global uses the official
`Isaac/Sensors/RealSense/D455/rsd455.usd`, a640×480 native registered profile and
public range0.6–6 m. The underlying native recipe retains its recorded0.52–6 m
range; public filtering does not alter or fill the retained raw measurements.
Kit's effective projection and the Lab ideal authored-USD projection have
separate calibration receipts. See [native pipeline details](WRIST_CAMERA_NATIVE_PIPELINE.md).

The renderer retains close grippers/brackets as occluders and RGB beyond the
depth range. Only the public depth validity is filtered after the first hit.
Five actual GPU plane/foreground controls check the near/far boundaries and a
6 cm foreground occluding a20 cm target. These are ideal optical controls,
separate from stereo/hardware validation.

## Attachment and second-arm pose

The fixed transforms are defined in each measured hand frame, using metres and
XYZW quaternions. Full-precision matrices are in the shared configuration and
each episode's rig receipt. Franka0 observes from the knife's side/rear; Franka1
uses a safe observation posture rather than the old distant park. The original
park put the second wrist about0.9 m from the material, beyond D405 range.
The new hand target is approximately `(0.20,0.18,0.34)` m with downward orientation.
Its actual official-model FK, joint margins and initial contact check are saved.

The current Global position is `(0.28,-0.42,0.46)` m, looking toward
`(0,0,0.02)` m. This prioritizes the workpiece and knife, intentionally cropping
the workstation. The v2 overview at `(1.10,-1.20,1.05)` m and the v1 placement
`(1.65,-1.75,1.50)` m remain in their versioned configurations and recordings.

The camera roots are actual children of both articulated `panda_hand` prims.
Each20 ms environment step checks:

```text
T_world_camera(t) = T_world_hand_measured(t) @ T_hand_camera
T_camera_world(t) = inverse(T_world_camera(t))
```

The mount is constant across steps/resets; there is no independent camera
animation or look-at controller. Native capture additionally compares current
USD and Replicator optical poses against the measured-hand composition while
holding physics. The physics engine advances10 rigid substeps per environment
step; images are sampled at their explicitly recorded camera cadence.

Housings and mounting rods are dimensioned visual geometry. They add no contact
colliders, mass or inertia to the robots. Static exact triangle/OBB checks retain
the initial rejected housing collision and the subsequent clear configuration.
Whole-trajectory geometric clearances and native robot contacts are separate
checks; a visual-clearance PASS is not a camera-contact physics validation.

## Execute and observe

Use a fresh output directory. The native launcher requires
`--camera-mode native_single_view` for this mixed layout; only Global uses the
native D455 model, while both wrist streams remain ideal prototypes.

```bash
.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id CUT-01 --camera-layout wrist_d405_global_d455 \
  --camera-mode native_single_view --steps 400 --gpu 1 --device cuda:1 \
  --output-dir artifacts/wrist_cut_NEW

.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id TRIM-01 --particle-spacing 0.002 \
  --camera-layout wrist_d405_global_d455 --camera-mode native_single_view \
  --steps 400 --gpu 1 --device cuda:1 --output-dir artifacts/wrist_trim_NEW

.deps/isaaclab-venv/bin/python scripts/run_isaac_lab_cutting.py \
  --task-id CUT-01 --camera-layout wrist_d405_global_d455 \
  --steps 400 --reset-check --output-dir artifacts/lab_wrist_cut_NEW

.deps/isaaclab-venv/bin/python scripts/run_isaac_lab_cutting.py \
  --task-id TRIM-01 --particle-spacing 0.002 \
  --camera-layout wrist_d405_global_d455 --steps 400 --reset-check \
  --output-dir artifacts/lab_wrist_trim_NEW
```

The shared API defaults to this new layout:

```python
from robotic_cutting_sim.live_env import open_cutting_env

with open_cutting_env("TRIM-01", runtime="isaac_sim",
                      particle_spacing_m=0.002,
                      include_global_camera=False,
                      output_dir="artifacts/wrist_api_NEW") as env:
    obs, info = env.reset(seed=42)
    assert set(obs["rgb"]) == {"wrist_d405_0", "wrist_d405_1"}
    obs, reward, terminated, truncated, info = env.step()
    overview_rgb = env.render("global_d455")
    overview_depth = env.render("global_d455", mode="depth_array")
```

Set `include_global_camera=True` to include Global RGB-D and its calibration in
formal observations. Force, oracle material state and task evaluation remain
outside the observation allowlist. `vision_only` and `vision_proprio` still work;
Cartesian knife-edge poses and joint-position actions are unchanged.

The public API CLI also supports `--include-global-camera`. To reproduce the old
camera version explicitly pass `--camera-layout legacy_fixed_d455`. A static
preview can be authored with `robotic-cutting-sim scene --preset workcell
--camera-layout wrist_d405_global_d455 --output-dir artifacts/wrist_preview_NEW`;
that preview does not run cutting.

## Recording and independent validation

All three cameras save synchronized RGB, optical-Z depth and validity masks.
Native records per-frame calibration inside `frame_manifest.json` and NPZ
payloads; Lab saves `rgbd/calibration_*.json`. Both retain every20 ms measured
hand/camera transform in `camera_pose_trace.json`; native also keeps JSONL.
The current surfaces, actual robot poses, fixed mount, source bytes/hashes,
seed and independent task evaluation are preserved separately.

```bash
.venv/bin/python scripts/check_d405_ideal_optics.py \
  --output-dir artifacts/d405_optics_NEW --device cuda:0
.venv/bin/python scripts/audit_wrist_camera_rig.py \
  --episode artifacts/wrist_cut_NEW --output-dir artifacts/wrist_audit_NEW
.venv/bin/python scripts/render_wrist_camera_evidence.py \
  artifacts/wrist_cut_NEW artifacts/wrist_media_NEW
```

The audit criteria were committed before final full episodes. They separately
score FK/time, optics, visibility, ranges and conservative mount clearances.
Incomplete runs remain partial. All errors and invalid samples are retained;
prototype optics, task physics and native hardware-model validation have
independent statuses.

## Preserved v2 episodes and evidence

Both CUT-01 and TRIM-01 completed 400 control intervals / 8 seconds in both
Isaac runtimes with the final rig. All 1,604 pose samples pass FK, inverse,
fixed-Global and time checks. All 164 captured three-camera timestamps use the
recorded runtime optics. CUT preserves two 72 g parts and Task PASS; coarse
TRIM preserves the 1.608 g detached chip and strict surface-goal FAIL.

Native Global minimum valid-depth fractions are 15.8337% for CUT and 15.8490%
for TRIM. CUT passes the frozen engineering view screens. TRIM retains a
secondary-wrist knife visibility FAIL at 6.4–8.0 s: the actual first arm's palm
occludes the withdrawn knife. During active cutting (1.6–5.6 s), all three
views see the knife. Material and each current mechanical component remain
visible in at least one view throughout the saved post-split frames.

The primary mounting rod intersects its adjacent own wrist flange at the
anchor region, failing the literal sampled robot-surface clearance screen.
All housings and other robot-body pairs are clear at the sampled poses.
There is no validated mount contact/mass model. Rare native depth tails,
strict TRIM surface errors and delayed coupling-work failures are retained.
See [the full independent report](WRIST_CAMERA_VALIDATION.md) and
[machine-readable receipt](evidence/wrist_camera_audit.json) for every status.

Public captured media: [CUT RGB-D](media/wrist_cut01/synchronized_multiview.mp4),
[TRIM RGB-D](media/wrist_trim01/synchronized_multiview.mp4), and
[actual second-arm motion](media/wrist_motion_probe/synchronized_multiview.mp4).
The [motion plot](media/wrist_motion_probe/camera_pose_tracking.png) compares
measured-hand composition with recorded optical movement. Each media directory
includes `camera_transforms.npz` with every measured transform and timestamp,
`camera_pose_trace.csv`, and `provenance.json` with input/output hashes.
Individual RGB and depth videos are linked from [README](../README.md).
