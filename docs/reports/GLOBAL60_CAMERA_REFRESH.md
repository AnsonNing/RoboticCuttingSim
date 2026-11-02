# Physical Global camera refresh — work in progress

The requested view uses an explicit `wrist_camera_rig_v9`: the optical origin is
0.60 m from the initial workpiece base-plane centre, with a higher frontward
azimuth chosen to keep the active hand beside the working region. The fixed
world offset from that target is `(0.18, -0.36, 0.4449719092)` m. Scene creation
sets that target once; the camera does not track the robot or optimize a wrist.
Both hand-local v5 D405 mounts and Franka 1's normal park remain unchanged in
new recordings. Default camera selection remains v5.

Global retains the controlled native v5 640×480 optics and SingleView sensor
settings. Rig9's logical valid-depth cutoff explicitly matches the existing
native 0.52 m cutoff. This is a provisional simulated camera, with the existing
native correspondence/outlier limitations; it is not a new hardware calibration
or depth-accuracy claim. Wrist RGB/depth and source timestamps are preserved.

Only thirteen historical sources have exact per-capture surface meshes. Their
513 new Global captures reproduce every original live surface hash, and every
rendered robot/tool matrix matches its measured state. They are labelled camera
replays; their original wrist images remain byte-identical. Historical wrist
calibrations/park states remain those actually recorded, including pre-v5 rigs.
No old wrist is re-aimed or relabelled v5.

Other saved episodes contain particle positions without per-frame F, or lack
material states altogether. Current-volume/cut-aware surfaces cannot be
reconstructed exactly from those positions. An early legacy-radius rendering
proposal was rejected and retained as an angle diagnostic; none of its images
may enter this publication. These 31 source cases require new Native recordings.
Five exact-input aliases share a new recording, leaving 26 unique engineering
recordings. This avoids repeating the historical scientific study: the original
results, denominators, failures, media and source evidence remain unchanged.
New task and HOME outcomes will be reported from the new recordings themselves.

Optional `--save-render-surfaces` writes the exact staged mesh at every capture,
with a hash checked against the camera bridge, timestamp and any face-varying
normals. It does not change particles, F, solver code, contacts, surfaces,
observations, thresholds or evaluation. Full movies retain the complete timeline
and original camera data. A labelled fixed digital crop enlarges the work area;
a full Global RGB inset retains the complete captured view and its occluders.

The maintained scene entry accepts an explicit camera selection:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_wp14_scene.py prepare \
  --scene standard_cut --camera-rig-profile wrist_camera_rig_v9 \
  --out artifacts/global60_prepare_NEW
# Use launch in the pinned Native environment for a new actual recording.
```

Normal Native `open_cutting_env(..., camera_rig_profile="wrist_camera_rig_v9")`
accepts the same Scene/Goal inputs. Omitting that option retains v5. Existing
scene catalog, frozen CUT research profile and physics configurations are intact.

Current state: exact saved-mesh replays complete; new recordings, full movies,
visibility review, release review, final exact-commit CI and remote media checks
are in progress. Publication verification and the new DONE marker are not written
until those checks pass. Runtime commands, versions, seeds, failed attempts,
protected-history baseline and individual timing are retained under
`artifacts/wp14/global60_001/`.
