# Live Isaac Sim6.2 validation

Actual runtime initialization now succeeds on the unchanged RTX5090 host with
driver570.211.01. Isaac Sim6.2.0.0 uses Kit110.4.0, bundled USD25.11, native
PhysX110.4.4 and `omni.sensors.nv.camera`1.4.0. The dedicated environment is
`.deps/isaacsim-venv`; it does not inherit the standalone or Lab environment.
Torch2.11.0+cu128 actually allocated a CUDA1 tensor and returned its square.
Newton1.6.1 and Warp1.17.0 are retained for the existing material and IK code.

The official6.2 [Python installation instructions](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html)
explicitly offer Torch2.11 with CUDA12/cu128. The
[requirements table](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
labels595.58.03 as a tested Linux driver; it is not evidence that every lower
driver fails. The [Omniverse driver table](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)
also lists Blackwell GeForce R570. No driver, OS or other project changed.

## Runtime receipts

| Actual attempt | Result |
|---|---|
| `isaacsim_startup_001` |BLOCKED:default base experience lacked optional robot-schema package; exit55 |
| `isaacsim_startup_002` |BLOCKED:minimal experience needed the upstream docstring-test dependency; exit55 |
| `isaacsim_startup_003` |Interrupted after299s:registry stalled fetching the1.494GB Vulkan shadercache |
| `isaacsim_startup_004` |PASS_BOOTSTRAP:real SimulationApp initialization, five updates, exit0; zero physics steps |
| `isaacsim_live_smoke_001/002` |FAIL:actual6.2 API/schema integration errors, retained with logs |
| `isaacsim_live_smoke_003` |Physics executed; independent camera audit FAIL_CAMERA_ALIAS. Asset products silently adopted2048×1080 and aliased both views |
| `isaacsim_live_smoke_004` |FAIL:strict runtime camera-pose check caught the unfixed housing dynamics |
| `isaacsim_live_smoke_005` |PASS_EXECUTION:2 control steps,20 native PhysX substeps, distinct640×480 RTX views and1280×960 Global |
| `isaacsim_live_native_smoke_001` |PASS_EXECUTION:same real loop with registered native SingleView measured depth, plus measured robot/material reset |
| `isaacsim_live_native_d0_001` |PASS_EXECUTION:400 control steps/4,000 actual PhysX steps/8s;41 native D455/Global captures; mechanical graph1→2 |
| `isaacsim_live_native_d0_ff_001` |PASS_EXECUTION:same400-step live recipe with explicit joint-velocity feedforward; tracking meets ROB-02 numeric limits |
| `isaacsim_live_native_d0_ff_no_feedback_001` |PASS_EXECUTION:matched feedforward controller with material wrench disabled at the robot; actual edge motion changes |
| `isaacsim_live_native_d0_ff_repeat_001` |PASS_EXECUTION:independent same-seed full repeat; ENV-01 reset/repeat tests FAIL, retained below |

The Vulkan shadercache was then fetched from the same official CDN directly,
SHA256 `ce1230814a7e3dbc32926adb5802c0c79ab09665036d54dfcbe74b57db459549`,
and installed under `.deps/isaacsim-extension-cache`. This is an installation
workaround, without a renderer/driver version substitution. Extension downloads
also use Kit's user cache under `~/.local/share/ov/data/exts/v2`.

## Live execution

```bash
.venv/bin/python -m pytest -q tests/test_isaac_sim_runner.py
.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --steps 400 --record-every 10 --camera-mode native_single_view \
  --gpu 1 --device cuda:1 --output-dir artifacts/isaacsim_native_fresh

# Explicit controller variant; retains identical gains, geometry and material.
.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --steps 400 --record-every 10 --camera-mode native_single_view \
  --velocity-feedforward --gpu 1 --device cuda:1 \
  --output-dir artifacts/isaacsim_native_ff_fresh

# Review actual captured arrays, without a USD replay or another renderer.
.venv/bin/python scripts/render_live_isaac_evidence.py \
  artifacts/isaacsim_native_ff_fresh artifacts/isaacsim_native_ff_media_fresh
```

`configs/isaac_sim_cutting.kit` selects the actual PhysX/RTX extensions rather
than loading unrelated ROS, policy or GUI workflows. Only the cached official
robot geometry is reused. In every control interval:

1. Newton computes joint **targets** from the desired edge pose; its robot
   dynamics never step.
2. Native PhysX advances both Panda articulations and applies the previous
   interval's measured material wrench at the actual hand-attached edge.
3. PhysX tensors supply the measured hand/edge pose to the retained TLMPM solver.
4. Current mechanical components produce the new USD surface. Native PhysX
   writes actual link transforms to USD; the wedge remains a fixed hand child.
5. Native RTX camera captures run while physics is held. Camera projection,
   optical pose and resolution are read back and checked; physics counters and
   measured robot state must remain unchanged during rendering.

Native camera housing rigid bodies are explicitly kinematic. Pre-authored asset
render products are deactivated in the session layer; fresh unique products
prevent the observed adoption/aliasing failure. Cached asset files are unchanged.
The initial joint-position write occurs only during reset. `step()` uses targets
and actual dynamics, with no robot/tool teleport or recorded trajectory input.

## Cameras and acceptance boundary

Native D455 optical views remain640×480; Global uses1280×960 to preserve visibility
of the small blade while showing both arms. Native SingleView uses the independently
validated [exact Kit profile](CAMERA_VALIDATION.md):95mm baseline,317.0431px focal,
confidence1, IR4m,150px maximum disparity, range.52–6m and noise.25/.25.
The runner reads the actual schema values and exact Kit/camera versions before
applying that profile. Registered depth values are bitwise retained native
measurements. Raw depth/XYZ and RTX reference depth are separate camera debug
artifacts; ideal reference depth never fills native holes.

The native smoke retained115,163 /46,829 valid initial pixels in the two views.
Its measured same-seed reset errors were zero for joints and material positions,
with zero residual cut codes. Camera reset repeatability and independent live
geometry/timestamp acceptance remain separate tests. `PASS_EXECUTION` does not
promote physical cutting, surface flattening or complete Isaac acceptance.
The400-step D0 episode is recorded in `artifacts/isaacsim_live_native_d0_001`.
Its actual detached mass is1.656g, final minimum detF.939, and maximum accumulated
plastic strain.498. It took82.74s including final reset, excluding application
initialization. Baseline position tracking fails the frozen gates:active maximum
1.256mm versus1mm, and all-phase5.715mm versus5mm. Active height error.222mm and
orientation error.0271° meet their respective.5mm/1° gates. Complete acceptance
is therefore not promoted. These values are measured diagnostics, not calibrated
physical-material validation.

Completed actual live videos are `live_global.mp4`, `live_d455_0.mp4` and
`live_d455_1.mp4` in that episode directory. They contain41 directly captured
frames at5fps and explicitly identify the Live Isaac Sim/Kit source. Remaining
surface, contact, camera synchronization and repeatability gates use the separate
independent acceptance audit.

## Controller comparison and reproducibility

The explicit `--velocity-feedforward` variant sends the finite difference of
successive commanded IK joint targets to native PhysX velocity targets. Baseline
PD remains available and its failed tracking measurements remain above. Robot
gains, desired edge path, material, cameras and transforms are unchanged.

| Measured trajectory quantity | Baseline PD | Velocity feedforward | Frozen limit |
|---|---:|---:|---:|
| Active maximum edge position error |1.25565mm|.48042mm|1mm|
| All-phase maximum position error |5.71478mm|2.16346mm|5mm|
| Active maximum height error |.22245mm|.25161mm|.5mm|
| Active maximum rotation error |.02706°|.03559°|1°|

The feedforward run still produces two actual mechanical components and a1.656g
chip. It applies a peak1.99723N material reaction. The next interval's applied
wrench equals the prior computed wrench exactly in recorded precision; current
impulse divided by20ms equals the computed next wrench exactly. In the matched
`--no-feedback` run, the solver still computes reactions up to2.06695N but the
robot receives zero material wrench. The maximum measured edge difference is
.62647mm, versus5.79µm between two feedback-enabled repeats. This establishes
measurable coupling without asserting converged interface energy.

The full same-seed repeat has a1.505mm maximum material-position difference and
.02089m/s maximum velocity difference. Final mechanical membership is identical,
but recorded cut-code histories differ. These exceed ENV-01's.1mm/.01m/s limits.
After the repeat, a real same-process reset restores q and material positions
exactly, clears every cut code, and resets the clock to zero. Native camera
reset repeatability fails: masks change at1543/580 pixels; common-valid maximum
depth differences are6.44/22.67mm. Global common-valid maximum difference is
14.313mm. For all three views the99th percentile is zero, but the frozen maximum
and exact-mask requirements are still violated. Native invalid/outlier values
are retained, without filtering to change that result. The independent gate
audit must keep this failure separate from registration-profile acceptance.
The diagnostic `isaacsim_reset_probe_002` additionally confirms all nine DOFs
and all22 measured rigid-link poses reset exactly. Initial/reset reconstructed
surfaces differ by at most3.73nm in symmetric nearest-vertex distance. The
remaining render/reset discrepancy is unresolved; native noise alone does not
explain the ideal Global difference.

Evidence is in `isaacsim_live_native_d0_ff_repeat_001/{comparison.json,
feedback_comparison.json,reset_receipt.json,camera_reset_quantiles.json}`.
The directly captured repeat videos and trajectory plot are under
`artifacts/isaacsim_live_native_d0_ff_repeat_media_002`: `live_global.mp4`,
`live_d455_0.mp4`, `live_d455_1.mp4`, `live_measured_depth.mp4`,
`live_synchronized_multiview.mp4`, and `planned_vs_actual_trajectory.png`.
Full acceptance remains false; execution, tracking and registration evidence
do not override the failed reproducibility or unfinished surface/task gates.

Artifacts include frozen source copies/hashes, native process exit/log/phase,
actual robot joints and22 link poses, commanded/measured edge poses, raw and
registered cameras, per-frame current surface provenance, full reference
stencils, final constitutive state, material-reaction diagnostics and reset
receipts. Force, material labels and Global images stay outside observations.
Native MDL packages remain uncached; explicit PreviewSurface overrides preserve
geometry/occlusion. Initial asset-loading logs retain the unresolved-MDL warnings.
The installed package freeze is `artifacts/isaacsim_runtime_freeze_001.txt`.
Two unused optional import converters require external `usd-exchange`; they are
not enabled, since injecting its standalone USD into the native Kit environment
would change the validated bundled-USD runtime.
