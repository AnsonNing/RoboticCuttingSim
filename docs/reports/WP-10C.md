# WP-10C — live ruler, profile tasks and actual finger pads

Completion: incomplete; stopped at the user's request for Claude handoff on
2026-10-29. The results below are actual GPU/runtime measurements. A successful
command or startup smoke is separate from a successful cutting task.

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

## Full reference episodes

All use seed 42 and the original v2 task placement/rake. Wall time is the
runtime harness total, excluding its outer recording process.

| Runtime | Task | Execution | Strict task | Steps / IK rejects | Wall s | Artifact under `artifacts/wp10/` |
| --- | --- | --- | --- | --- | --- | --- |
| Native | CUT-01-R | PASS | PASS | 400 / 0 | 365.06 | `c_native_cut01r_reference_001` |
| Lab | CUT-01-R | PASS | FAIL | 400 / 0 | 599.43 | `c_lab_cut01r_reference_001` |
| Native | TRIM-01-R | PASS | FAIL | 400 / 394 | 247.92 | `c_native_trim01r_reference_001` |
| Lab | TRIM-01-R | FAIL pregrasp | NOT_RUN | 0 / n/a | 18.28 | `c_lab_trim01r_reference_001` |
| Native | CUT-02 | PASS | PASS | 400 / 0 | 834.08 | `c_native_cut02_reference_001` |
| Lab | CUT-02 | PASS | FAIL | 400 / 0 | 959.84 | `c_lab_cut02_reference_001` |

Native CUT-01-R separates two 72 g components; CUT-02 separates two 180 g
components. Both Lab cuts retain one connected component. Bottom-layer
unassigned material/compatibility bridges remain in the first Lab cut; this
is not an IK rejection. Native TRIM removes 0% of the reference above-plane
mass. Lab TRIM fails the frozen pregrasp angular-speed stability check before
formal material time advances. No tolerance was changed to admit it.

The early reference-controller tracking statistic compared measured tool pose
with the held/accepted target after rejection. It does not measure the
requested reference-path error during those 394 rejected TRIM steps. The
handoff recorder now includes the requested reference target separately;
do not interpret the old small held-target error as successful TRIM tracking.

Both seven-case frozen ruler rigid-grasp control suites report PASS:
`c_native_grasp_controls_001/result.json` and
`c_lab_grasp_controls_001/result.json`. These engineering comparisons include
the expected weakening/release responses; they do not prove stable holding
under every load or at the TRIM rake. Both complete 401-pose CUT-01-R sampled
visual camera/tool clearance audits report PASS in
`c_{native,lab}_cut01r_clearance_001`. Triangle-intersection screening is not
a complete collision/contact certification.

## TRIM diagnosis and versioned candidate scene

Actual pad corners at 25 degrees extend about 1.8 mm below the target plane.
Alternate IK branches can follow the path but penetrate the original table by
about 15.97 mm. The explicit `raised_workpiece_50mm_v1` scenario retains the
25-degree rake and raises the workpiece/supports by 50 mm. Its Native two-step
smoke passes; Lab still fails angular-speed pregrasp stability
(`c_{native,lab}_raised_trim_smoke_002`). Neither is a completed TRIM task.

The new `ruler_workcell_v2` candidate adds a 35-degree TRIM rake. Its Newton
geometry-only preflight passes all 401 IK samples with zero penetrating
table/self contacts, minimum joint margin 0.09146 rad and maximum per-step
joint change 0.00567 rad. Minimum target-plane clearance is 8.27 mm for the
lower pad, 14.13 mm for the handle and 85.42 mm for the wrist bracket.
`c_workcell_v2_preflight_001/result.json` contains the measured primary-arm seed
now stored in `configs/wp10_scenarios_v2.json`. Live execution of this candidate
is **NOT_RUN**. Fixture/camera/robot intersections and dynamic holding need
separate validation. Franka 1 and the fixed hand-local v5 cameras are unchanged.

A separate ideal-tracking, full-size 1 mm TRIM probe also fails with 0% removal
(`c_trim_kinematic_formal_001`, 136.53 s). Thus fixing reachability alone does
not establish skimming: law-2 label classification/compatibility bridges need
investigation. The new `tlmpm_v3.py` is an explicit experimental hypothesis;
only syntax and CPU kernel compilation pass, with its physics **NOT_RUN**.

## Provenance and remaining delivery

[Compact C/D evidence](../evidence/wp10/wp10cd_handoff_v1.json) records exact
commands, seeds, commits, source hashes, raw receipt paths and wall times.
Most runs started from `02ceca6` with preserved dirty-source snapshots; that
commit alone is not their complete source identity. Raw three-camera RGB-D,
calibration/transforms, command traces and measured robot/tool poses remain in
the completed live episode directories. Final annotated WP-10 robot videos
and new ruler README screenshots are **NOT_RUN**; existing v5 CUT/TRIM images
and historical videos remain preserved. Continue from
[the Claude handoff](../HANDOFF_WP10_CLAUDE_20261029.md).
