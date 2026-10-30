# End-to-end cutting validation checkpoint

Both requested actual runtimes now execute an 8 s D0 task with actual dual
Franka articulations, a hand-mounted wedge, retained Warp TLMPM, persistent
mechanical separation and reaction feedback. Lab uses the actual GA task and
scene APIs; Sim6.2 uses actual Kit/PhysX and native camera sensors. Neither
runner consumes offline robot/material animation. No driver upgrade, other-project
modification or policy training was performed.

The independent [18-gate report](reports/ISAAC_LIVE_ACCEPTANCE.md) contains exact
versions, commands, per-runtime tables, source provenance, comparisons and
negative results. [Lab](ISAAC_LAB_INTEGRATION.md), [Kit](ISAAC_SIM_VALIDATION.md),
[coordinates](SCENE_COORDINATE_VALIDATION.md),
[trajectory](ROBOT_TRAJECTORY_VALIDATION.md),
[camera](CAMERA_VALIDATION.md) and
[surface geometry](SURFACE_GEOMETRY_VALIDATION.md) have separate evidence.

Execution and permanent separation pass. The frozen complete integration
protocol does not pass: surface reconstruction/quadrature, 20 ms delayed-work
balance, selected post-withdrawal contact events and strict numerical/camera
repeatability remain unresolved. No thresholds were relaxed. Canonical Lab
native-contact and repeat receipts are appended as they complete, preserving
earlier runs and failed infrastructure probes.

Current real Kit video delivery is [here](media/live_isaac_kit_d0/provenance.json),
with Global, both D455 RGB views, measured depth and synchronized multiview files.
Those files encode arrays captured during live integration, rather than replay
rendering. Oracle data and reactions remain outside policy observations.

The next implementation checkpoint adds CUT-01 and TRIM-01 through shared
workpiece-local geometry, fixture, tool mounting, controller and evaluation
contracts. Each task requires its own live episodes and success receipt; the
existing D0–D3 results cannot pass these new tasks.
