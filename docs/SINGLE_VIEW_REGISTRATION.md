# Native SingleView depth registration audit

**Status: prototype implemented; promotion to validated simulated D455 remains
BLOCKED.** This CPU-only adapter registers actual NVIDIA SingleView samples to
the original virtual RGB camera. It does not substitute ideal depth or add
Python noise. A fresh noise-OFF slanted-plane control misses the unchanged
1 mm p95 registration criterion: 1.309 mm and 1.344 mm for the two cameras.
The remaining sampling bias and noise behavior are recorded, rather than hidden.

## Convention and implementation

NVIDIA describes SingleView as one camera view reprojected to a stereo pair,
with disparity quantization/randomization and confidence filtering. Its
`DepthSensorPointCloudPosition` AOV is relative to the left camera; its scalar
distance AOV is in the reprojected grid. See the primary
[Camera Extension documentation](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.camera/latest/camera_extension.html)
and [post-render SPG changelog](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.camera/1.4.0/CHANGELOG.html).

Actual captures establish the native point-cloud axes as X right, Y up, Z
backward. Conversion to a CV direction is therefore `[X, -Y, -Z]`. Projecting
that direction with the runtime `CameraParams` intrinsics associates the sample
with the original RGB grid. No extra 95 mm baseline translation is needed;
the point cloud already supplies the left-frame direction. That conclusion is
checked against measured workcell imagery and the independent analytic plane.

The helper uses XYZ **only for direction/pixel association**. Output optical-Z
values are copied exactly from native `DepthSensorDistance`, including its native
biases and outliers. A nearest-measured-depth z-buffer selects an actual sample
when several project into one pixel. Every output pixel records its source AOV
index in separate debug data. There is no averaging, hole filling, depth
correction, ideal-depth input, random sampling, or point-cloud/depth consistency
rejection. Very large finite XYZ magnitudes are used only as directions and are
reported separately; they are never mistaken for measured geometric positions.

An initial candidate consistency filter would have dropped genuine depth
differences and created a misleading exact match to ideal depth. That candidate
was rejected. The delivered helper explicitly tests that biased native samples
survive, and that point-cloud Z is never substituted for measured Z.

The target is the renderer's RGB from the original **virtual depth-camera view**.
This does not register the separate D455 color imager, launch the Isaac sensor
wrapper, or calibrate physical D455 hardware. Output files retain
`registration_validated=false`, `rgb_depth_aligned=false` and
`simulated_d455_validated=false`; their same-grid reprojection is a prototype.

## Actual experiments

The raw workcell input is `artifacts/ovrtx_d455_002`: three native frames from
each D455 view. The fresh independent control is
`artifacts/single_view_plane_fixture_001/slanted_plane.usda`, a checker-colored
plane with world equation `Z = 1 + 0.25 X + 0.10 Y`. Its X and Y slopes expose
pixel-association errors that a plane with no horizontal depth gradient can
miss. Three frozen scene times, 0/1/2 s, are rendered with unchanged geometry.

The RTX agent captured exact native controls on GPU1, using OVRTX
`0.6.0.1791947691` and ovstage `0.3.0.1791986420`, baseline 95 mm, confidence 1,
IR effective distance 4 m and ten warmup frames:

- `single_view_plane_noise_off_001`: native noise mean 0, sigma 0. Exact zeros
  executed successfully; no epsilon fallback was used.
- `single_view_plane_noise_on_001`: native noise mean 0.25, sigma 0.25.

No raw capture is modified. Canonical offline audits are
`single_view_registration_workcell_002` and `single_view_registration_plane_002`.
Compact full receipts and input hashes are in
[single_view_registration_audit.json](evidence/single_view_registration_audit.json).
[Control plot](evidence/single_view_registration_control.png).

| Actual first-frame measurement | Camera0 | Camera1 |
|---|---:|---:|
| Workcell raw-grid mean absolute error vs original ideal image | 21.833 mm | 24.556 mm |
| Registered native-Z mean absolute error | 0.437 mm | 0.489 mm |
| Registered native-Z RMSE, including hard outliers | 6.899 mm | 5.360 mm |
| Largest retained workcell native depth error | 504.851 mm | 315.413 mm |
| Native valid workcell samples | 122,279 | 42,949 |
| Registered workcell pixels after z-buffer collisions | 121,380 | 42,732 |
| Noise-OFF plane signed mean error | +0.826 mm | +0.847 mm |
| Noise-OFF plane p95 absolute error | 1.309 mm | 1.344 mm |
| Noise-OFF plane maximum absolute error | 1.659 mm | 1.700 mm |

The huge improvement in workcell alignment is real, but the retained hard tails
remain visible. Native workcell samples beyond 1 cm error are not erased to make
the result look clean. Source-index checks establish bit-for-bit preservation
of selected native measurements across every audited frame.

The native geometric XYZ lies on the fresh plane to roughly 0.5 micrometers,
while its paired scalar depth has the bias above. Neighbor comparisons suggest
about one left-image pixel of scalar/XYZ sampling association difference on this
slope. This is an **inference**, not a confirmed NVIDIA implementation defect.
No empirically fitted pixel shift or calibration correction is applied.

Native noise ON primarily changes validity in this fixture. Registered coverage
drops from approximately 94% to 47%. Between consecutive static frames, about
137,000 validity pixels change in each camera; OFF has no temporal changes.
Across all three frames, common raw-grid ON/OFF depths differ by more than
1 micrometer in only 57/54 samples (RMS about 10 micrometers). After registration,
only one common sample per camera exceeds 1 micrometer, with a retained maximum
change of 0.530/0.544 mm. The shader settings therefore demonstrably affect native
output, but these data do not establish realistic Gaussian metric-depth noise.

## Reproduce

The renderer captures use the separate existing `.deps/rtx-venv` environment described
in the camera setup documentation. CPU registration uses the normal project venv.

```bash
.venv/bin/python scripts/make_single_view_registration_fixture.py artifacts/plane_fresh
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py artifacts/plane_fresh/slanted_plane.usda artifacts/plane_off_fresh --samples 3 --warmup 10 --gpu 1 --camera-mode simulated_d455 --confidence 1 --ir-effective-distance 4 --noise-mean 0 --noise-sigma 0
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py artifacts/plane_fresh/slanted_plane.usda artifacts/plane_on_fresh --samples 3 --warmup 10 --gpu 1 --camera-mode simulated_d455 --confidence 1 --ir-effective-distance 4
.venv/bin/python scripts/audit_single_view_registration.py --run off=artifacts/plane_off_fresh --run on=artifacts/plane_on_fresh --plane-fixture artifacts/plane_fresh/fixture.json --output-dir artifacts/registration_fresh
.venv/bin/python -m pytest tests/test_single_view_registration.py -q
```

The plane audit intentionally exits 1 with a **BLOCKED** receipt when its p95
exceeds 1 mm. Seven CPU tests pass for coordinate conversion, exact noisy-sample
preservation, z-buffer provenance, invalid samples and analytic plane geometry;
they do not change the failed native registration control into a pass.

Next work requires confirming the scalar/XYZ sample association in NVIDIA's
native model or obtaining explicit per-sample imager coordinates, then rerunning
this unchanged plane and noise-control protocol. The working ideal RGB-D and
robot/material research environment remain independently usable.
