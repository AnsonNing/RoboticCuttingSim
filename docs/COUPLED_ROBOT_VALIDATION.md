# Measured robot/material coupling — 2026-10-28

`CoupledCutting` adds a small executable coupling testbed between the real
MuJoCo-controlled dual Panda articulation and the experimental persistent-cut
TLMPM material solver. It retains every material particle and uses measured
joint-derived knife poses. The second arm stays parked. This work does not start
Isaac Lab, add a policy, or change the existing blade-only environment.

The robot now optionally carries a closed convex triangular wedge. Its six
vertices and dimensions exactly match TLMPM's extruded triangular contact SDF:
12 mm length, 4 mm upper height, rear lower Z=-0.4 mm, and 25 mm half width.
A fixed hand-to-wedge Rx(pi) mount lets the downward Panda hand advance the
edge along world +X. The knife pose denotes its cutting edge, not box center.
The default box knife and its standalone results remain available unchanged.

## Coupling algorithm

Every 20 ms, the rigid controller advances ten substeps under the previous
material reaction wrench. The new measured knife pose and old measured pose
define a linear/SLERP path for 200 TLMPM steps of 0.1 ms each. The material
solver receives those actual poses and their world linear/angular velocities.
Its equal-opposite contact impulses, divided by 20 ms, become the wrench applied
to the robot during the next interval. Force and torque are shifted from the
knife edge to the hand COM before the articulated solve.

This is explicit staggered coupling with a **20 ms delay**, not a monolithic
contact solve. The material and robot clocks agree after each interval. Captured
material graphs retain stable trajectory buffers, which are filled from measured
robot state each tick. A slow 8 s approach/hold/sweep/withdrawal path reduces
tracking error at the cut depth. No particle is moved according to a fragment
script; no particle or mass is deleted.

Proprioception contains only `q`, `qdot`, `tool_pose` and `timestamp_s`. Material
state, reactions, contact impulses and work live in explicit diagnostic outputs.
Cameras are supplied by a separate integration work package.

## Reproduce the frozen D0 experiment

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/run_coupled_cutting.py --output-dir artifacts/coupled_fresh --compare-feedback
.venv/bin/python -m pytest tests/test_robot_contract.py tests/test_coupled_robot_contract.py -q
```

Canonical artifacts: `artifacts/coupled_robot_cut_002`. Its `implementation/`
directory preserves exact executed modules/drivers, with manifest SHA256s and a
supplemental executed-driver receipt. The prior `coupled_robot_cut_001` is an
intermediate run and remains untouched. Run 002 follows the final contact-work,
new-cut progress-reset and gradient-support corrections to the material solver.

Files include 401 measured robot states and all link/tool poses, 81 material
snapshots, constitutive/failure state, immutable mass and particle IDs, impulses,
applied wrenches, timestamps, trace CSV, setup/source receipts and exact knife
geometry. `dual_franka_replay.usda` contains complete official Panda geometry at
every measured timestamp, plus the actual wedge. Numerical USD read-back position
discrepancy is zero. This USD is measured offline replay, not a new solver run.

Public compact evidence: [coupled_robot_validation.json](evidence/coupled_robot_validation.json).
Measured plots: [coupled_robot_trace.png](evidence/coupled_robot_trace.png).

| Actual test | Result |
|---|---|
| Continuous initial material | One conservative mechanical transfer component, 19,340 particles |
| Coupled cutting after withdrawal | Two components: 138-particle 1.656 g chip and 19,202-particle 230.424 g substrate |
| Existing cut digits | Never change after assignment; no re-welding by continuum transfer |
| Chip movement after withdrawal | 1.492 mm; chip rests near the tray wall, no claim of dramatic free flight |
| Finite material dynamics | PASS: minimum det(F)=0.90604, peak speed 0.20245 m/s |
| Actual active cutting depth | 15.6445–15.7204 mm for the nominal 15.5 mm command |
| Tracking over complete episode | Maximum 5.707 mm during approach/withdrawal, rather than at the settled cut depth |
| Actual contact reaction | Peak material force 1.0413 N |
| Wrench returned to robot | Previous-interval impulse carry error exactly zero |
| Matched feedback-disabled control | Same commands/seed; feedback changes joints by up to 3.6588e-4 rad and tool position by 0.34481 mm |
| Default standalone box regression | PASS; q, qdot and body poses remain exactly identical; tool quaternion round-off is at most 2.22e-16 |
| CPU configuration/frame tests | PASS: 16 tests, separate from the actual GPU/robot physics runs |

Both feedback ON and OFF create two components in this single D0 episode. That
is not a performance comparison or evidence about population success. The force
toggle demonstrates that material reactions influence the articulated robot,
instead of merely being logged beside an independent animation.

The interface work ledger measures +14.0945 mJ of material boundary work and
-13.7689 mJ of work from material wrenches applied to the articulated tool. The
residual is +0.32566 mJ, about 2.31% of material boundary work for this episode.
This finite-delay ledger is an actual diagnostic, not a claim of converged total
energy conservation. Torque impulses are accumulated about moving knife points,
and their transfer between intervals also uses a lever-arm approximation.

The feedback-enabled 8 s run took 35.348 s including setup, state export and
replay construction. Mean interval costs were 7.40 ms IK, 4.05 ms rigid dynamics,
and 58.85 ms material physics. These are host-specific isolated probe timings.

## Remaining validation boundary

This establishes an experimental two-way, delayed articulated cutting path and
a persistent mechanical split in one uncalibrated D0 example. It does not establish
Isaac Lab execution, calibrated Plasticine, converged contact/cutting physics,
arbitrary robust trajectories, or a complete research benchmark. The separate
material validation work package evaluates recontact, refinement and reset.
The body/knife collision mesh agrees with the material wedge, while hand/arm
contact against deformable material remains outside this minimal adapter.

Parameter changes and larger force/time-step regimes can destabilize explicit
partitioned coupling. No reaction clipping or artificial fragment motion is used
to hide that limitation. Formal `physics_proven` remains false.
