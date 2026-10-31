# WP-03 — Actual RTX arrays and dynamic surface bridge (2026-10-28)

The existing Newton/Warp baseline is preserved. Standalone NVIDIA OVRTX now
renders the complete official workcell into programmable RGB-D arrays on the
unchanged host driver. This is a separate runtime from Isaac Sim6.2 / Isaac Lab3.0.
Actual Isaac `SingleViewDepthCameraSensor` remains BLOCKED because Isaac is absent.

| Check | Actual result / evidence |
|---|---|
| CPU bridge contracts | 6 tests PASS: mode separation, stale-frame rejection, topology replacement, units, camera transforms, CameraParams decoding |
| Measured ParticleSurface → OpenUSD | PASS:9 snapshots,9 different topology hashes; vertices/indices/timestamps roundtrip exactly; `artifacts/surface_bridge_002` |
| Complete USD → ideal Warp RGB-D | PASS:3×640×480,3 measured frames at0/1/2s;448k triangles,144 Mesh/Cube prims; `artifacts/workcell_rgbd_001` |
| Complete USD → actual RTX RGB-D | PASS: `artifacts/ovrtx_rgbd_003`; actual CPU-readable uint8 RGB480×640×3 and float32 depth480×640 from OVRTX DLPack mappings |
| RTX SI plane / calibration QA | PASS:192314/156593/29192 independently visible table pixels; max errors0.40/1.15/0.99µm, tolerance10µm; `artifacts/ovrtx_depth_qa_001` |
| Moving articulated robots → RTX | PASS:robot-only measured replay over base workcell,3frames across5.917s; changed depth pixels54070/34548/8021; `artifacts/ovrtx_robot_002` |
| Native stereo post-process AOVs | PASS for array production: `artifacts/ovrtx_d455_002`; real `OmniSensorDepthSensorSingleViewAPI` / `DepthSensorDistance`, not Python-added noise |
| D455 registration/calibration | NOT_VALIDATED:post-process reprojection changes pixel correspondence; raw point-cloud output also needs validity QA. Do not use original RGB K/pose as registered modeled-depth calibration |
| Isaac6.2 native D455 wrapper | BLOCKED:actual command returned exit2 and `result.json`; `artifacts/isaac_d455_001`, no ideal fallback |
| Live physics/camera scheduling / true cut surface | NOT_RUN / NOT_ASSESSED:these tests replay measured deformation or robot state; they do not establish cutting |

The full ideal geometry includes both native Panda mesh hierarchies, D455 bodies,
table, tray, blade and changing material. Warp omits four authored Capsule/Sphere
prims, reported in `geometry.json`; Mesh/Cube visibility and transforms are used.
RTX uses the native USD geometry, with simple USD PreviewSurface material overrides
because the geometry-only asset cache does not contain external MDL packages.

## Reproduce

From the repository root, using a fresh output directory for every run:

```bash
.venv/bin/python -m pytest -q tests/test_sensor_bridge.py
.venv/bin/python scripts/check_dynamic_surface_bridge.py \
  artifacts/research_handoff_001/live_env/surface artifacts/surface_bridge_fresh
.venv/bin/python scripts/capture_workcell_rgbd.py \
  artifacts/research_handoff_001/research_workcell_replay.usda artifacts/workcell_rgbd_fresh --device cpu
```

OVRTX is isolated under `.deps/rtx-venv` (ignored). Its wheel was1,348,231,444bytes,
OVStage66,832,291bytes; installed renderer environment occupies about2.5GiB.
The project `.venv`, driver, OS and other projects were not changed. This host's optional
environment reads project site-packages through an uncommitted `.pth` file.
For an independent installation, create a Python3.12 virtual environment and
install this project with its `tools` extra plus `requirements-rtx.lock` there.
Warp1.17 is additionally needed only for the independent ray-camera QA.

```bash
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py \
  artifacts/research_handoff_001/research_workcell_replay.usda artifacts/rtx_fresh \
  --samples 3 --warmup 5 --gpu 1
.venv/bin/python scripts/check_ovrtx_depth.py artifacts/rtx_fresh artifacts/rtx_depth_fresh
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py \
  artifacts/research_handoff_001/research_workcell_replay.usda artifacts/d455_fresh \
  --camera-mode simulated_d455 --confidence 1 --ir-effective-distance 4 --gpu 1
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py \
  artifacts/dual_franka_cpu_006/dual_franka_replay.usda artifacts/rtx_robot_fresh \
  --base-scene artifacts/research_handoff_001/workcell/workcell.usda --gpu 1
```

Use `--gpu 1` with all GPUs visible:OVRTX's Vulkan/CUDA matching warns when
`CUDA_VISIBLE_DEVICES` remaps devices. Renderer/device pinning runs camera work on GPU1, with a small context
allocation also observed on GPU0. Logs retain R570 DLSS quality fallback and
transform-history warnings; no long-exposure or rolling-shutter claim is made.

On a compatible machine with actual Isaac6.2 installed:

```bash
PYTHONPATH="$PWD/src" /path/to/isaac-sim/python.sh scripts/capture_isaac_d455.py \
  artifacts/research_handoff_001/workcell/workcell.usda \
  artifacts/research_handoff_001/live_env/surface artifacts/isaac_d455_fresh
```

The Isaac-specific launch path has not executed beyond its missing-runtime gate
here. Its native color and depth camera extrinsics remain separate and unaligned;
annotator warmup is bounded. Existing userland renderer success must not be
reported as an Isaac6.2 or Lab3.0 launch.

## Numerical and provenance details

`DynamicSurfaceBridge` validates finite SI vertices, integer triangle bounds and
strictly increasing matching physics/surface timestamps. Each update replaces
points, indices, counts and extents in one OpenUSD change block and blocks stale
normals from weaker layers. No geometry is generated or removed to simulate cuts.

An actual first OVRTX attempt found a100× unit error:root stage metadata is not
inherited from sublayers. `ovrtx_rgbd_001` and its original output are preserved,
with `post_run_audit.json` marking FAIL. The corrected code explicitly retains
meters/up-axis/timecodes and checks runtime `CameraParams` scale and extrinsics.
The known-plane QA computes its visible-pixel mask from a separate CPU ray camera
and the known table, never by filtering against the RTX depths being tested.

RTX render-product aspect fitting changes the effective native D455 intrinsics:
at640×480, actual fx=fy317.0431,cx320,cy240. The authored vertical aperture alone
would suggest fy377.6600. `runtime_calibration.json` and raw CameraParams record
the measured renderer projection; `calibration.json` records authored optics.

In the corrected ideal replay, changed depth pixels above10µm were1638/1374/246.
Three camera captures per sampled geometry took0.519/0.350/0.318s in the warm-cache
probe; initialization, shader compilation and a live solver are excluded. Source
physics timestamps and actual renderer capture-window timestamps are both saved.
Five warmup steps put the renderer timestamp83.333ms after the held source state;
this is explicit replay convergence, not validated live synchronization.

The optional modeled mode requests NVIDIA's actual SingleView input/output AOVs.
D455 nominal baseline95mm is sourced from the manufacturer, while noise(.25pixel
mean/sigma), confidence and IR-distance heuristic are provisional. Default
confidence.7 with IR disabled produced only1905/1138valid depth pixels in the flat
workcell. Explicit confidence1 and IR effectiveDistance4m increased raw validity
to122279/42949pixels atframe0. This proves the native model executes, not that its
noise or D455 accuracy is physically calibrated. RGB is rendered from the depth
optical frame, not the separate native color camera; modeled-depth registration
is deliberately marked false. Do not interpret all changes in modeled depth as
material motion, since the model can change visibility/noise as well.

The runner exposes `--noise-mean` and `--noise-sigma` (both default.25). A paired
static slanted-checker-plane control actually ran with exact0/0 and.25/.25 in
`artifacts/single_view_plane_noise_off_001` and `single_view_plane_noise_on_001`.
Each has three frozen source times, confidence1, IR4m and ten warmup steps. Exact
zero parameters executed successfully without an epsilon fallback. OFF had zero
native depth pixels changing above10µm across frozen times; ON had43/41 such
pixels in the two views. The raw scalar/XYZ outputs remain intact for the separate
registration and analytic plane-error audit; temporal variation alone does not
calibrate a physical D455.

## Primary sources

- [Isaac6.2 depth sensor API and asset wrapping](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_camera_depth.html)
- [Isaac6.2 experimental RTX API](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/py/source/extensions/isaacsim.sensors.experimental.rtx/docs/index.html)
- [Isaac6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
- [OVRTX supported drivers, including R570 Blackwell GeForce](https://nvidia-omniverse.github.io/ovrtx/driver_requirements.html)
- [OVRTX initialization and two clocks](https://nvidia-omniverse.github.io/ovrtx/python_api/getting_started.html)
- [OVRTX time-sampled scene population](https://nvidia-omniverse.github.io/ovrtx/scene/loading_usd.html)
- [OVRTX image/depth outputs](https://nvidia-omniverse.github.io/ovrtx/sensors/cameras/outputs.html)
- [OVRTX CameraParams layout](https://nvidia-omniverse.github.io/ovrtx/sensors/cameras/decoding_outputs.html)
- [NVIDIA SingleView renderer schema and output AOVs](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.camera/latest/camera_extension.html)
- [RealSense D400 manufacturer datasheet, D45595mm baseline](https://www.intelrealsense.com/wp-content/uploads/2024/10/Intel-RealSense-D400-Series-Datasheet-October-2024.pdf)
- [OpenUSD Mesh topology](https://openusd.org/release/api/class_usd_geom_mesh.html)

OVRTX0.6.0.1791947691 and OVStage0.3.0.1791986420 are pre-release NVIDIA proprietary
runtimes under the NVIDIA Software License Agreement and Product Specific Terms
for NVIDIA AI Products. Third-party binaries/assets remain ignored and are not
redistributed in this repository.
