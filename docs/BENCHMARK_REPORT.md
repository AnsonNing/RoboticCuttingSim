# Standalone simulation benchmark

The benchmark now executes D0–D3 articulated cutting episodes, reconstructs
current mechanical surfaces, composes measured robot/tool motion and captures
full-scene RGB-D. It is an executable engineering baseline with preserved results.

```bash
.venv/bin/python -m robotic_cutting_sim.cli benchmark \
  --variants D0 D1 D2 D3 --seeds 42 --output-dir artifacts/fresh_benchmark
```

Every case starts from one continuous base plus bumps and zero cut history.
The same fixed controller runs8 s at50 Hz with measured Panda poses and20 ms
delayed reaction feedback. Five sampled material surfaces plus all401 robot
states produce USD replay and two ideal RGB-D observations with Global debug.
Replay capture is distinct from the400-step live API and41-frame native RTX test.

## Actual seed42 baseline

`artifacts/simulation_benchmark_001` completed all four physics and camera cases.
Wall time170.49 s includes setup,reconstruction,diagnostics and IO; concurrent
development may affect timing. It is not a performance ranking.

| Variant | Recipe | Final components | Detached mass | Residual height proxy |
|---|---|---:|---:|---:|
| D0 | Fixed dome |2|1.656 g|1.266 mm|
| D1 | Shape/location/size |2|2.376 g|1.286 mm|
| D2 | Three continuous bumps |3|0.972 g|21.713 mm|
| D3 | Multiple bumps,varied supported material |2|0.393 g|16.013 mm|

`PASS_EXECUTION` means actual numerical guards and camera checks completed.
The fixed sweep does not flatten D2/D3; the large residuals remain negative
baseline results. No policy was trained/selected. One seed per variant cannot
establish population success or generalization.

Explicit benchmark D0 uses friction0.4; canonical legacy coupled D0 used0.3.
These are separate protocols. D3 density,E,Poisson,yield and friction actually
apply; hardening,rectangular blade-thickness mapping and camera perturbations
remain explicitly unsupported/deferred in parameter receipts.

## Metrics and acceptance limits

Oracle metrics stay outside observations: immutable mass ledger,source-reported
mechanical component mass/momentum,kinetic/gravitational energy,outside-footprint
mass,remaining-height and overcut proxies. Height uses particle-bin maximum Z
plus1 mm support radius; it is not a mesh/RGB-D reconstruction or exact volume.
Component provenance and bottom-reference-layer anchors are explicit. Missing
provenance never silently becomes measured chip mass. New recordings include
frame-aligned measured tool poses.

Formal observations contain RGB,depth,validity,calibration,time and measured
proprioception. Force,material labels/particles/stress and Global are excluded.
No rendering mask creates a cut.

The frozen matrix supplies noncontact,indentation,uncut scraping,withdrawal,
recontact,repeat and resolution controls. A later `supported_cross_cuts` test uses
flat rigid support **without sidewalls** and sweeps the full second cross section:
1→2→3 components,unchanged existing digits,detF≥0.9027,peak speed0.696 m/s over4.5 s.
The original finite partial second-cut FAIL remains unchanged. Unsuccessful
free-specimen controls remain preserved. This is not a two-direction Franka test.

The fixture-wall option defaults on. A fresh default regression retains final
labels/components; all-frame position difference28.4 µm and velocity0.00545 m/s
are within the prior frozen repeat limits. Seven actual GPU analytic controls
were rerun and passed. Spatial/time refinement remains exploratory; no convergence
is claimed. Material and force calibration remain open.

Receipts: [benchmark](evidence/complete_simulation_benchmark.json),
[supported cross cuts](evidence/supported_cross_cuts.json),
[frozen matrix](reports/TLMPM_VALIDATION.md),
[coupling](COUPLED_ROBOT_VALIDATION.md),[actual RTX](reports/WP-03-cutting.md).

`benchmark_executable=true`; formal `BENCHMARK_READY=false` remains accurate for
the original full scientific criteria: calibration,convergence,broader seeds,
complete D3 recipe application and D455 modeling remain open. Full Lab3.0/Sim6.2
execution is BLOCKED on the installed stack; this runnable backend is standalone.
The separate [native SingleView audit](SINGLE_VIEW_REGISTRATION.md) preserves
actual noisy samples and the failed slanted-plane promotion criterion.
