# Camera rig v5 — two hand-aligned wrist cameras

Both wrist D405s now use the same fixed optical mount on their actual
`panda_hand`: translation `[-0.085, 0.045, 0.020]` m and identity rotation.
The camera looks along hand +Z, with camera X/Y parallel to hand X/Y. Neither
wrist mount uses a look-at target, knife aiming, a gimbal or horizon tracking.
Franka 1 retains its normal safe park.

The old slant came from Franka 0's earlier knife-looking mount. Camera rig v4
removed the second arm's special observation pose but retained that primary
camera angle. The user's later request removes the remaining slant as well.
[V5 configuration](../configs/wrist_camera_rig_v5.json) links the unchanged v4
hash; every v1-v4 configuration, recording, video and README image remains
preserved. Global's close view and every physical/task threshold are unchanged.

The sensor follows the actual hand rigidly. An actual wrist rotation can still
rotate the image relative to world gravity; the sensor is not a stabilized
world camera. The side offset leaves the gripper/tool volume unobstructed and
does not imply optical aiming. The bracket's box orientation is only a visual
strut between its mounting points, not a camera orientation command.

## Validation

The [source-linked receipt](evidence/wrist_camera_v5_validation.json) records
new actual Native CUT/TRIM episodes, Lab smoke, camera/FK/timestamp checks,
sampled robot-mesh clearance and decoded media. Original CUT/TRIM task scores
remain independent of camera composition. A view of a robot, an occluded view
or a view without the workpiece does not become evidence of failed cutting.

Both housings and brackets have zero intersections with the actual hand/finger
triangles in the new static geometry control. All new recordings retain RGB-D,
calibration, timestamps, actual hand poses and hand-to-camera/world transforms.
The second camera's v5 local mount equals its v4 mount exactly; the preserved
v4 actual secondary-motion probe independently measured 24.947 mm camera motion
with a constant local mount and zero Global drift.

The complete Native CUT visual clearance audit also passes all 401 measured
robot poses with zero unexcepted camera/robot triangle intersections. Native
CUT/TRIM each complete 400 steps; original CUT is PASS and TRIM precision
remains FAIL. Lab's separate two-step smoke passes RGB-D/FK/clock contracts.
The old full-footprint-in-wrist-frusta screen fails in both Native v5 episodes;
its exact false checks and raw coverage remain recorded. That view-composition
screen is separate from valid calibration, mounting and cutting, and no wrist
is re-aimed to satisfy it. Global remains available as a separate optional view.

Actual local Chrome passes streamed camera/step/reset/Global controls. The first
v5 browser attempt preserved a 10-second reset acknowledgment timeout; the
reset completed after that deadline. A separate attempt with a declared
60-second UI-reset wait passes every state-change assertion. External Mac
connectivity remains NOT_RUN.

Camera housings/brackets remain visual-only prototypes without mass or physical
colliders. Sampled visual clearance is reported separately from native rigid
contacts and continuous swept/closed-volume clearance. D405 hardware noise,
stereo reconstruction and physical calibration remain unvalidated.

Reproduce with a fresh directory:

```bash
.venv/bin/python scripts/audit_camera_mount_geometry.py artifacts/v5_mount_NEW
.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id CUT-01 --steps 400 --gpu 0 --device cuda:0 \
  --tool-attachment-mode physical_grasp --grip-initial-position 0.0112 \
  --camera-layout wrist_d405_global_d455 --camera-mode native_single_view \
  --output-dir artifacts/v5_cut_NEW
```

The current [README image](media/scene_current_v5.png),
[CUT video](media/wrist_v5_cut01/synchronized_multiview.mp4) and
[TRIM video](media/wrist_v5_trim01/synchronized_multiview.mp4) come from new
actual v5 captures. The v4 equivalents remain under their original paths.
