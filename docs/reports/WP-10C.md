# WP-10C — live ruler, profile tasks and actual finger pads

Status: IN PROGRESS. Full task episodes and the frozen ruler grasp controls
are being measured; a startup smoke is not a completed cutting task.

Both live runtimes complete a two-step ruler smoke with a real independent
`ruler_knife_v1`, `tlmpm_v2` cutting law 2, and three blunt boxes (handle plus
two actual USD finger pads). Raw measurements:
`artifacts/wp10/c_native_ruler_smoke_001` and
`artifacts/wp10/c_lab_ruler_smoke_001`. Neither smoke has an IK rejection.
The Native and Lab actual edge-frame pad half extents are approximately
8.25 x 8.75 x 7.60 mm; pad centers are about 60.24 mm behind the edge and
18.56–18.60 mm to each side. These are measured geometry, not assumed pad size.
The compound ruler mass is 0.150585112 kg (1.47724 N weight). At 20 N per
finger and friction 0.6 the nominal sliding budget is 24 N; the finite-pad
torsional budget is 0.156141 N m. These are analytic budgets, not grasp results.

CPU full regression after the integration: **461 PASS in 92.45 s**,
`artifacts/wp10/cpu_full_002`; all tests, including `cpu_sim`, run.
Scaled-pad/current-body-motion and equal-and-opposite separate-pad impulse
tests pass. The frozen `tlmpm.py` is unchanged; law-1 bitwise regression passes.

The full Native/Lab CUT-01-R, TRIM-01-R and CUT-02 runs use protocol 1 mm
particles, 0.1 ms material steps, physical grasp, unchanged v1 thresholds,
camera rig v5, and actual three-camera RGB-D. Their declared plans are
`artifacts/wp10/c_{native,lab}_plan_001.json`. Every raw command, resolved and
accepted target, actual hand/tool/body pose, IK result, reaction, finger
diagnostic, timestamp and source snapshot is retained. See
[ADR 0004](../adr/0004-live-ruler-pads-and-validation.md) for contact scope.
