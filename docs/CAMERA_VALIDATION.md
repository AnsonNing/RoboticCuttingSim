# Camera validation for Isaac integration

The native SingleView source-association problem is resolved independently in
standalone OVRTX and actual Isaac Sim 6.2 / Kit native sensor controls. Both
cameras passed all preregistered development and held-out plane/foreground-box
controls while retaining native measured depth and noise. **A live cutting
episode still needs its own camera/state synchronization and visibility audit;
these controls do not calibrate physical D455 hardware.** The earlier failed audit remains unchanged
in [SINGLE_VIEW_REGISTRATION.md](SINGLE_VIEW_REGISTRATION.md).

## Frozen protocol and actual source correspondence

Commit `a83d1e9` froze
[`single_view_correspondence_v1.json`](../configs/single_view_correspondence_v1.json)
before new captures. Its SHA256 is
`3dc0ed00e4d5be88fd187459492b823caceb2501019d613a990085535f0486c2`.
Twelve unique geometries produce fourteen frames per camera: positive/negative
X and Y slopes, distances 0.85–1.8 m, five held-out geometries, a foreground
red box in development and held-out scenes, and three repeated static frames.
The independent analytic geometry uses runtime intrinsics and extrinsics.

The old one-slope result suggested an X sample-association offset. H0 retains the
old projection. The single preregistered H1 adds exactly one RGB pixel in +X after
projecting native XYZ direction, then copies **raw `DepthSensorDistance`** to the
associated pixel. This is an empirically validated version-specific AOV
association; it is not a confirmed NVIDIA defect or a fitted depth correction.
No offset search or threshold adjustment followed held-out results.

Every OFF frame had to satisfy p95 absolute plane error below 1 mm and at least
90% valid known-geometry coverage. All analytic feature scanlines count in the
RGB/depth boundary comparison, including missing data; p95 error must be at most
0.5 pixels. RGB-versus-analytic feature IoU must be at least 0.95. Samples are
never rejected because of ideal/analytic depth disagreement. Native range,
finite forward directions, image bounds and nearest-measured-Z collisions are
the only registration selection rules. Native shader outlier removal remains
explicitly enabled in both captures.

NVIDIA documents stereo reprojection, disparity quantization/randomization and
left-camera-relative cloud coordinates in its
[Camera Extension](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.camera/latest/camera_extension.html).
The [camera extension changelog](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.camera/1.4.0/CHANGELOG.html)
distinguishes the current post-render SPG AOVs from deprecated post-processing
AOVs. Installed USD headers expose parameter declarations; the packaged shader
algorithm is not available as inspectable source. The actual Isaac 6.2 Python
sensor API describes noise mean/sigma as disparity quantization parameters in
pixels, not a physical metric Gaussian calibration.

## OVRTX results

Actual captures use OVRTX `0.6.0.1791947691`, ovstage `0.3.0.1791986420`, native
SingleView at 640×480, baseline 95 mm, confidence 1, IR effective distance 4 m,
ten warmup frames, and native noise OFF `(0,0)` versus ON `(0.25,0.25)`.
These are renderer controls, not robot dynamics or a live Isaac episode.

| Fixed control result | Camera0 | Camera1 |
|---|---:|---:|
| H0 worst OFF plane p95 | 2.1122 mm | 2.0481 mm |
| H1 worst OFF plane p95 | 0.0005015 mm | 0.0004587 mm |
| H1 OFF coverage across all cases | 92.81–96.73% | 92.81–96.73% |
| H1 foreground feature boundary p95 | 0 pixels | 0 pixels |
| Worst foreground RGB/analytic IoU | 0.99919 | 0.99973 |
| H1 maximum OFF full-scene error, including tails | 1.543 mm | 1.559 mm |
| H1 maximum ON full-scene p95 | 1.693 mm | 1.638 mm |
| H1 maximum retained ON full-scene error | 4.469 mm | 3.330 mm |
| H1 ON coverage across all cases | 45.99–74.04% | 45.99–74.02% |

All twenty-eight OFF frame/camera tests pass H1; H0 failures are preserved in the
same audit. The maximum error is reported independently of p95. The few OFF
boundary tails are retained, not erased to match the plane. All chosen measured
values have exact native source indices and compare bit-for-bit with raw Z.

Among common OFF/ON raw samples differing by more than 1 μm, H1 retains
729,499/729,502 samples for camera 0 and 728,536/728,536 for camera 1. Only three
camera 0 samples are unselected by geometric projection/z-buffer selection;
there are **zero depth-error or consistency rejections**. The source-value noise
retention ledger and output-grid comparison are distinct, because noise changes
validity and sample association. Near static frames show mostly changing holes;
negative-X and farther held-out cases show extensive native metric changes.
No Python noise, XYZ-Z replacement, smoothing or hole filling is used. Physical
noise realism remains uncalibrated.

[Full audit and hashes](evidence/single_view_heldout_audit.json),
[control figure](evidence/single_view_heldout_controls.png).

Canonical local outputs:

- `artifacts/single_view_holdout_fixture_002`: frozen USD, cases and protocol.
- `artifacts/single_view_holdout_off_001`, `...on_001`: untouched native AOVs.
- `artifacts/single_view_holdout_audit_001`: initial unchanged gate results.
- `artifacts/single_view_holdout_audit_002`: same gate results with extra noise
  retention accounting; no new capture or threshold change.
- `artifacts/single_view_holdout_audit_003`: canonical repeat with exact
  pre-analysis source byte snapshots; unchanged raw captures and results.
- `artifacts/native_registered_workcell_002`: six programmatically readable
  registered real native workcell frames with exact raw-Z provenance. The first
  attempt `_001` rejected authored focal-length rounding of 0.000012 pixels;
  `_002` uses the same 0.0001-pixel tolerance already allowed for runtime K.

`register_ovrtx_single_view_depth` rejects other renderer versions, resolutions,
intrinsics and native recipes. The generic default H0 helper remains unchanged.
The explicit profile reports validated **source correspondence**, while physical
D455/noise calibration flags remain false. It targets the original virtual
depth-camera RGB view, not a separately calibrated D455 color imager.

## Workcell FOV and occlusion

An independent CPU closest-visual-triangle audit uses actual runtime camera
calibrations at 0, 4, 8 s in the frozen coupled replay. Guide-purpose collision
geometry is excluded. This is an offline geometry diagnostic and cannot pass
live camera acceptance. Full receipt:
[camera visibility](evidence/camera_visibility_audit.json).

The native D455 views have HFOV 90.532°/VFOV74.251°. Visible material optical Z
is 0.6147–0.7561 m, within the configured 0.52–6 m range. They see 714–1370 material
pixels and 26–91 knife pixels at these three times. The material is small in this
wide view, so native validity holes and the blade can limit surface detail.

The existing 640×480 global view sees both Frankas, material and table, but only
8–15 knife pixels, below the frozen CAM-03 requirement of 25. This failure was
reported before live capture. The live runner is being configured with a
1280×960 global product while preserving native D455 640×480; its actual live
visibility must be measured independently.

## Real Isaac camera controls

Commit `e9096fd` froze a separate
[`single_view_kit_correspondence_v1.json`](../configs/single_view_kit_correspondence_v1.json)
before the first real Kit capture. SHA256:
`2107bf4101fcfe798a9b1958af15bf02dbab182ee80ebcdecaba241c0af4e851`.
It requires IsaacSim 6.2.0.0/camera extension1.4.0 and preserves every geometry,
H0/H1 candidate and acceptance threshold from the OVRTX protocol. The OVRTX
profile is never silently applied to Kit. This separate camera-only runner does
not claim robot stepping or cutting.

**PASS_SOURCE_CORRESPONDENCE_CONTROLS** for both real Kit native cameras. Actual
versions are Isaac Sim `6.2.0.0`, Kit
`110.4.0+feature.386251.187cf21b.gl`, kernel
`210.4.11+release.15522.174283e4.gl`, camera extension `1.4.0`, native RTX sensor
API `2.0.1`, Warp `1.17.0`, Torch `2.11.0+cu128`, Python 3.12, RTX 5090 and
driver `570.211.01`. This is an actual `SingleViewDepthCameraSensor` capture
through Kit/Replicator, using held geometry rather than a dynamics episode.
The wrapper returns native XYZ3; raw measured Z is retained independently.

| Same prefrozen Kit control | Camera0 | Camera1 |
|---|---:|---:|
| H0 worst OFF plane p95 | 2.1124 mm | 2.0482 mm |
| H1 worst OFF plane p95 | 0.0002310 mm | 0.0002282 mm |
| H1 OFF coverage | 92.81–96.73% | 92.81–96.73% |
| H1 foreground boundary p95 | 0 pixels | 0 pixels |
| H1 maximum retained OFF scene error | 1.542 mm | 1.559 mm |
| H1 maximum ON scene p95 | 1.693 mm | 1.638 mm |
| H1 maximum retained ON scene error | 3.856 mm | 3.546 mm |
| H1 ON coverage | 46.07–73.94% | 45.96–73.92% |
| Native nonzero-noise samples retained | 730,040/730,044 | 728,520/728,524 |

All twenty-eight OFF frame/camera tests pass. The eight unselected nonzero-noise
samples arise from projection/image bounds/z-buffer selection, with no
depth-error or consistency rejection. Complete records:
[Kit audit and hashes](evidence/single_view_kit_audit.json),
[actual Kit control figure](evidence/single_view_kit_controls.png).

Canonical captures are `artifacts/single_view_kit_off_005` and
`artifacts/single_view_kit_on_002`; the audit is
`artifacts/single_view_kit_audit_003` with pre-analysis implementation byte
snapshots. Audits001/002 remain preserved. After startup, fourteen three-camera
captures take 12.08 s OFF and 11.55 s ON in this nonexclusive GPU0 run; shader
compilation and application launch are excluded. This is not a live cutting
frame-rate benchmark. Twenty-eight registered native ON frames are readable at
`artifacts/native_registered_kit_controls_001`.

The app starts on R570/GPU0. Early launch/import/EULA-environment failures and
the SingleView sensor's rejection of an `rgb` constructor annotator are preserved
in `single_view_kit_off_001/002/003.launch.log` and
`single_view_kit_off_003/failure.log`. RGB is obtained from a direct Replicator
annotator on the same native render product. OFF004/ON001 completed captures but
misnamed the project app version `0.1.0` as the Kit version; both are preserved,
and the clean repeat records actual `get_kit_version()` and kernel versions.
No geometry, threshold or candidate changed for the repeat.

`register_kit_single_view_depth` has a separate exact-runtime guard and the
same controlled optics/native recipe guard. It is validated by actual Kit
controls, independently of the OVRTX result. It explicitly leaves physical/noise
calibration and a particular dynamics episode's synchronization unvalidated.

## Actual live D0 camera/mesh synchronization

The independent audit of all 41 frames in
`artifacts/isaacsim_live_native_d0_ff_001` uses actual recorded PhysX poses for
all 22 rigid links, the measured hand-fixed wedge, and each frame's actual
current-component ParticleSurface mesh. Runtime camera intrinsics/extrinsics
come from actual Kit camera parameters. Each source mesh hash is checked
against the live capture receipt. CPU rays are an independent geometry
reference, not new sensor imagery or a replacement live episode.

The reference excludes guide-purpose collision hulls including instance proxies
and honors mesh sidedness/clipping. Early CPU diagnostics001–003 incorrectly
included instanced guide hulls and then investigated sidedness; those failed
diagnostics remain preserved. Explicit instance-proxy purpose filtering resolves
the discrepancy. Two additional CPU tests cover that failure and sidedness/near
clipping. The canonical full audit stores pre-analysis source snapshots and input
hashes at `artifacts/live_camera_mesh_ff_002`.

| All 41 frames, worst case | D455 0 | D455 1 | Global |
|---|---:|---:|---:|
| Measured-state mesh vs actual RTX p95 | 0.179 μm | 0.179 μm | 0.715 μm |
| Mesh/RTX valid-mask IoU minimum | 0.999988 | 0.999962 | 0.999976 |
| Largest retained mesh/RTX silhouette tail | 579.8 mm | 353.0 mm | 710.2 mm |
| Native depth vs mesh p95 | 0.238 μm | 0.358 μm | RTX only |
| Native/mesh valid-mask IoU minimum | 0.44510 | 0.20547 | RTX only |
| Largest retained native error | 620.7 mm | 1003.6 mm | RTX only |
| Maximum native-valid chip pixels | 36 | 34 | 31 RTX pixels |

The large silhouette/hard-tail maxima remain included in RMSE and maximum-error
records; p95 alone does not describe their severity. No valid sample is trimmed.
All physical/surface/camera timestamp differences are below 9e−16 s after
explicitly subtracting the native physics clock's 6 ms startup origin.

The full 1280×960 global view passes the frozen visibility count in every frame.
Minimum depth-consistent first-hit counts are Franka0=18,610, Franka1=27,791,
knife=26, material=608 and table=161,311 pixels. These are geometry attributions
checked at actual captured RTX pixels; native semantic segmentation was not
captured and is not claimed. The best separated-chip native-depth coverage
exceeds the frozen 25-pixel visibility criterion.

**CAM-03 geometric visibility PASS; dynamic geometry bridge and timestamps PASS;
strict native CAM-04 FAIL.** Native validity holes yield mask IoU far below the
unchanged 0.99 gate. Successful deterministic mesh/RTX agreement does not replace
that native failure. Source registration, actual native sensor execution,
physical noise realism, and full dynamic camera acceptance remain distinct.
[Full actual-live camera audit](evidence/live_camera_mesh_audit.json).

## Reproduce

Use fresh output names; existing experiment directories are never overwritten.

```bash
.venv/bin/python scripts/make_single_view_holdout_fixture.py artifacts/controls_fresh
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py artifacts/controls_fresh/holdout_controls.usda artifacts/controls_off_fresh --samples 14 --warmup 10 --gpu 0 --camera-mode simulated_d455 --confidence 1 --ir-effective-distance 4 --noise-mean 0 --noise-sigma 0
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py artifacts/controls_fresh/holdout_controls.usda artifacts/controls_on_fresh --samples 14 --warmup 10 --gpu 0 --camera-mode simulated_d455 --confidence 1 --ir-effective-distance 4 --noise-mean .25 --noise-sigma .25
.venv/bin/python scripts/audit_single_view_holdout.py artifacts/controls_fresh artifacts/controls_off_fresh artifacts/controls_on_fresh artifacts/audit_fresh
.venv/bin/python scripts/register_native_capture.py artifacts/ovrtx_d455_002 artifacts/registered_fresh
.venv/bin/python scripts/audit_camera_visibility.py artifacts/coupled_surface_002/cutting_workcell_replay.usda artifacts/ovrtx_coupled_002/runtime_calibration.json artifacts/visibility_fresh
.venv/bin/python -m pytest tests/test_single_view_registration.py tests/test_single_view_controls.py -q
```

The camera commands in the recorded manifests are canonical. Ten focused CPU tests
pass for raw-noise preservation, source-index provenance, profile rejection,
analytic geometry and missing-feature penalties. Real Kit command:

```bash
.venv/bin/python scripts/make_single_view_holdout_fixture.py artifacts/kit_controls_fresh --protocol configs/single_view_kit_correspondence_v1.json
OMNI_KIT_ACCEPT_EULA=YES .deps/isaacsim-venv/bin/python scripts/capture_kit_single_view_controls.py artifacts/kit_controls_fresh artifacts/kit_off_fresh --gpu 0 --noise off
OMNI_KIT_ACCEPT_EULA=YES .deps/isaacsim-venv/bin/python scripts/capture_kit_single_view_controls.py artifacts/kit_controls_fresh artifacts/kit_on_fresh --gpu 0 --noise on
.venv/bin/python scripts/audit_single_view_holdout.py artifacts/kit_controls_fresh artifacts/kit_off_fresh artifacts/kit_on_fresh artifacts/kit_audit_fresh
.venv/bin/python scripts/register_native_capture.py artifacts/kit_on_fresh artifacts/kit_registered_fresh --runtime kit
.venv/bin/python scripts/audit_live_camera_mesh.py artifacts/isaacsim_live_native_d0_ff_001 artifacts/live_camera_audit_fresh
.venv/bin/python -m pytest tests/test_camera_mesh_audit.py -q
```
