# Camera rig v4 — parked Franka 1 with a wrist D405

**Preserved v4 checkpoint.** The later [v5 update](WRIST_CAMERA_V5.md) also
removes the primary wrist's historical angled mount. All v4 records below
remain unchanged; v5 is now the active camera rig.

Franka 0 continues to hold the independent knife and execute cutting. Franka 1
uses the original safe park: `[-pi/2, -0.5, 0, -2, 0, 1.5, pi/4]` rad. Camera
layout no longer overrides either runtime's secondary-arm initialization.

`SECONDARY_OBSERVATION_JOINTS` and `SECONDARY_OBSERVATION_HAND_XYZW` were removed
from active code. The secondary optical mount is a fixed hand-local translation
`[-0.085, 0.045, 0.020]` m with identity rotation: it looks along the gripper's
local +Z without a world target, optimization, tracking or gimbal. It is an
actual child of the measured Franka 1 `panda_hand`. Runtime calibration composes
the measured hand transform with the fixed local optical transform.

[Rig v4](../configs/wrist_camera_rig_v4.json) references the unchanged v3 hash.
[Rig v3](../configs/wrist_camera_rig_v3.json), its original observation posture,
the previous README image and all older recordings remain preserved. Primary
wrist/Global mounts, intrinsics, ranges, material/physics and task thresholds
are unchanged. Both D405s remain ideal prototypes with visual-only housings
and brackets; camera mass and native camera colliders are not modeled.

## Actual validation

The [compact receipt](evidence/wrist_camera_v4_validation.json) links commands,
source/input hashes and independent results.

| Check | Actual result |
| --- | --- |
| Native and Lab full CUT | Both execute 400 steps / 8 s; original task score PASS, two 72 g parts |
| Native full TRIM | New 400-step / 8 s capture; chip separation and original strict surface FAIL retained |
| Separate Lab CUT/TRIM smokes | Both execute and preserve all three RGB-D streams, calibration, clock and measured transforms |
| Measured wrist/optical FK and clocks | PASS for full Native CUT and Lab smokes; Global stays world-fixed |
| Explicit secondary-joint motion calibration probe | Actual Native arm motion carries the hand-mounted camera by about 24.947 mm; fixed hand-local mount retained; separate probe never counted as a cut |
| Static official convex-collider task starts | CUT and TRIM PASS the recorded no-penetrating-robot-contact screen at the original 10 micrometre diagnostic tolerance |
| Both housings/brackets against actual hand/finger triangles | Zero intersections |
| Secondary camera versus every recorded robot visual body | Zero unexcepted triangle intersections at all saved motion-probe and Lab-smoke samples |
| Existing primary bracket versus its own link7 flange | FAIL retained: up to 74 intersecting triangles; this v4 change does not repair or hide the earlier primary-camera limitation |
| Secondary sees material/knife | False in new recordings; advisory only, while RGB-D/FK/optics contracts and cutting still pass |
| Remote viewer | Relaunched on the v4 scene; actual local Chrome decodes the streamed GUI. External Mac connection remains separate |

The first deployment-driver attempt preserves a Native metadata precheck
BLOCKED result because that existing venv lacks an installed project import
under Python isolated mode. Separate direct actual Native runs import this
checkout explicitly and complete successfully; no dependency or driver change
was used to bypass the precheck. The previous independent clean-bootstrap
physical smokes also completed and are reported in the resumption note.

The first new static diagnostic had an incorrectly mismatched IK mount and is
retained as FAIL; the corrected diagnostic constructs a controller with each
task's actual mount before solving the primary arm. No secondary IK or target
aiming is used. All contact records and failed attempts remain preserved.

Visual-surface intersection tests are sampled geometric checks. They do not
establish swept clearance, closed-volume containment, sensor calibration or
physical camera mass/contact behavior. The
[official camera documentation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_camera.html)
and the project's pinned runtime/source receipts remain the API references.

## Visibility and task separation

[Camera audit v2](../configs/wrist_camera_audit_v2.json) retains historical
v1-v3 observation-camera requirements. For v4's explicitly marked parked
secondary wrist, material pixels, knife pixels and nonempty in-range depth are
advisory. Invalid payloads, range/mask errors, bad calibration, moving local
mounts or timestamp mismatch still fail their respective camera contracts.
An empty or occluded view can be recorded without preventing CUT/TRIM execution
or changing their physics/task scores. Sensor selection in the viewer changes
only the preview, never the wrist mount or parked arm.

## Reproduce and view

```bash
.venv/bin/python scripts/audit_camera_mount_geometry.py artifacts/v4_mount_NEW
.deps/isaaclab-venv/bin/python scripts/validate_parked_wrist.py artifacts/v4_park_NEW
.venv/bin/python scripts/audit_wrist_camera_rig.py \
  --episode artifacts/wrist_v4_native_cut_001 --output-dir artifacts/v4_camera_NEW
.venv/bin/python scripts/audit_wrist_camera_clearance.py \
  --episode artifacts/wrist_v4_native_motion_001 --output-dir artifacts/v4_clearance_NEW
```

The full Native CUT command is the physical-grasp command in
[physical tool use](PHYSICAL_TOOL.md#reproduce); the current rig is v4 by
default. `--observation-motion-probe --steps 100` is an explicit calibration
experiment only, never an automatic task behavior or observation optimizer.

[Actual v4 CUT video](media/wrist_v4_cut01/synchronized_multiview.mp4),
[v4 TRIM video](media/wrist_v4_trim01/synchronized_multiview.mp4),
[CUT provenance](media/wrist_v4_cut01/provenance.json),
[TRIM provenance](media/wrist_v4_trim01/provenance.json),
[updated README image](media/scene_current_v4.png) and
[image provenance](media/scene_current_v4.json). Fourteen newly encoded videos
fully decode, use all 41 captured times per task and preserve original arrays.
