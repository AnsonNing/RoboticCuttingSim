# WP-03 — Dynamic surface to programmable RGB-D

State: **ideal array-output feasibility PASS; full G3 / D455 RTX BLOCKED**.
The later live environment also produces RGB-D at every control step and includes
the current tool, tray and table in its ray scene. A120-step episode was executed
and official observation boundaries tested. Native dual-D455 USD hierarchies are
composed in the separate workcell; they have not yet supplied RTX array outputs.
The offline probe described below renders material only; this limitation does
not apply to live tool/tray occlusion. See `ENVIRONMENT_API.md` and `WORKCELL.md`.
The tested path is recorded corrected Newton MPM state → Newton1.6.1
`ParticleSurface` → triangle vertices/indices/normals → a small Warp ray camera →
three RGB/depth/valid-mask NumPy arrays. This is an actual numeric pipeline, not a
GUI screenshot. It does not use Isaac Sim RTX or official D455 sensor annotators.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python -m robotic_cutting_sim.cli camera-check --input artifacts/newton_indent_002 --samples 9 --output-dir artifacts/camera_indent_002
.venv/bin/python scripts/export_replay.py artifacts/scene_d0_001/physics_testbed.usda artifacts/newton_indent_002 artifacts/camera_indent_002 artifacts/indent_replay_002.usda
```

Use fresh output paths. Physical and surface timestamps match by construction
in offline replay; no claim of live Kit synchronization/latency is made.
Input physics-state SHA256 is stored in the camera manifest. Particle radii come
from recorded spacing, independent of material density. Source state is corrected
post-buffer-fix run `newton_indent_002`.

Three cameras output640×480 arrays: two ideal pinholes with provisional87° HFOV,
and a global view with65° HFOV. Positions are respectively (0,-.5,.55),
(.5,-.4,.25),(.8,-.8,.8)m. Optical frames, intrinsics, transforms, range and invalid
encoding are in `calibration.json`. They are clearly labeled ideal cameras;
vendor distortion, stereo baselines/noise and actual D455 calibration are NOT_RUN.

Nine extracted surfaces span0–2s. Meshes have about12,240–12,346 vertices and
24,476–24,688 triangles. After a .240s first extraction, later extractions take
about .110–.115s including array export/compression. Three camera capture/save
passes take .023–.036s total per snapshot; these include disk encoding and are not
pure render-kernel latency. Sampled total GPU0 VRAM peaked at1031MiB including
display/baseline. No real-time claim is made.

Analytic finite-plane QA compares optical-Z depth and the complete expected
in-frustum valid mask:252,160 valid pixels, zero mask mismatches, max absolute
error8.41e-7m and RMSE1.17e-7m. This is numerical ideal-camera verification at
1e-5m tolerance, not a RealSense accuracy claim or comparison against a cut surface.

Across the recorded deformation, consecutive-snapshot common-valid pixels with
depth change>10µm total2,851 /2,489 /1,642 for ideal0 /ideal1 /global. RGB is uint8,
depth_m float32, masks bool. These changes are real changes in extracted MPM state;
their agreement with exact physical material boundaries has not been validated.
The current camera meshes contain material only, so knife/tray occlusion is not
included. Nearest mesh intersection handles self-occlusion and depth boundaries.

Local outputs:

- `artifacts/camera_indent_002/rgbd/`: RGB PNG and RGB-D NPZ, camera-separated.
- `artifacts/camera_indent_002/surface/`: oracle triangle meshes and timestamps.
- `artifacts/indent_replay_002.usda`: self-contained table/tray/blade/material replay,
  with9 surface topology samples and121 blade poses. USD topology/time range checked.
- `artifacts/indent_replay.usda`: retained failed first export due NumPy→Gf scalar
  conversion; do not use it. Corrected exporter converts scalars explicitly.

Remaining G3 work: compatible Isaac driver/runtime, live mesh staging into its RTX
render product, actual D455 camera API, source/surface/depth alignment including
tool occlusion, camera-clock synchronization, modeled stereo and a genuinely cut
surface. `VISION_PROVEN=false` remains appropriate even though the ideal camera
feasibility result is positive. Surface extraction itself may bridge small gaps;
its visual topology is never treated as solver fracture truth.
