# Shared CUT-01 and TRIM-01 task definitions

[`tasks.py`](../src/robotic_cutting_sim/tasks.py) supplies CPU geometry, fixture data, local targets, scripted reference motion, action validation and an offline task evaluator for the live Lab/Kit adapters. It creates no simulator, robot, camera, cut labels or policy. Existing D0 geometry and physics defaults are unchanged.

The measurable goal thresholds were committed in `02e427f` before the initial task smokes. The same configuration now explicitly records the predeclared material, fixture, blade mount and motion knots, before full-duration task episodes. [`cutting_tasks_v1.json`](../configs/cutting_tasks_v1.json) is the shared protocol; later runs must record its exact SHA256. No threshold was selected using a CUT/TRIM result.

| Task | Continuous workpiece | Fixture | Local goal | Knife |
| --- | --- | --- | --- | --- |
| CUT-01 | 160×30×20 mm bar, no bumps | Two supports centred at X=±55 mm, top Z=0; open central slot X∈(−20,+20) mm | Full severing at local X=0 into two substantial physical parts | Edge `Ry(+π/2)`, downwards advance, 35 mm blade length, 4 mm height, 50 mm full edge width |
| TRIM-01 | Same bar plus a continuous central cosine dome, radius12 mm/height8 mm | One 180×50 mm floor, top Z=0 | Remove the bump and measure the substrate at local Z=20 mm | Edge identity, +X advancement, 12 mm blade length, 4 mm height, 50 mm full edge width |

The original definitions use 1 mm particle spacing, a 4 mm material grid, an 8 s episode and 20 ms control intervals. Material parameters remain provisional: density1500 kg/m³, Young50 kPa, Poisson0.3, yield1000 Pa, friction0.3, zero hardening and retained edge-work energy0.05 J/m². The complete cell-centred bar has 96,000 particles and mass144 g. TRIM has 97,044 particles/mass145.566 g. Its curved dome still has discretization error; cell-centred20 mm base dimensions avoid the previous odd-layer15 mm/2 mm base overshoot.

The successful live TRIM follow-up explicitly selects `particle_spacing_m=.002`
(`--particle-spacing 0.002`), with 12,136 particles /145.632 g. This initial
quadrature override is recorded in `task.json`; it does not change the original
1 mm default or remove particles during an episode. Both actual Lab and Sim
follow-ups detach 1.608 g, while their strict surface goal remains FAIL.
Original negative experiments and [all live results](reports/CUTTING_TASK_EPISODES.md)
remain separately available.

## Fixed frames, mounts and paths

All targets belong to the workpiece. `sample_particles()` returns **local** points/masses/lattice. `points_to_world/local()`, `target_plane_world()` and world-reference pose methods explicitly apply `world_from_workpiece_xyzw`; quaternion order is XYZW. Pure geometry supports arbitrary proper rigid workpiece poses. Each live adapter must transform the actual material, tool motion, wrenches, gravity/fixtures and surface consistently, or reject a pose it cannot support. Moving only the visual mesh is invalid.

CUT uses `T_hand_edge = [0,0,.18, √.5,0,√.5,0]`. Combining that diagonal-axis180° mount with the edge's `Ry(+π/2)` leaves the Panda hand at `Rx(π)` and **180 mm above** the edge throughout the plunge. Reusing the horizontal TRIM mount for a vertical edge would put the hand near material/table height. At the full CUT plunge the edge is Z=−3 mm, the hand is Z=177 mm, and the blade's upper end remains above the20 mm material. The knife stays inside the40 mm support slot and has7 mm clearance to the world table top at Z=−10 mm.

TRIM retains `[0,0,.18,1,0,0,0]`, with an explicit0.2 mm edge clearance above its20 mm target. No plane fitting or hidden target offset is applied.

| Time, s | CUT edge local XYZ, mm | TRIM edge local XYZ, mm |
| ---: | --- | --- |
| 0 | `(0,0,50)` | `(−45,0,60)` |
| 1 / 1.6 | `(0,0,30)` | `(−45,0,20.2)` |
| 5.6 | `(0,0,−3)` | `(45,0,20.2)` |
| 6.4 / 8 | `(0,0,50)` | `(80,0,60)` |

The baseline arms cut digit0 during `[1.6,5.6)` s; outside it `active_cut_index()` is−1. This is a scripted reference scheduler. Future callers can explicitly arm cutting through actions, rather than being forced to follow the reference clock.

## Minimal API

```python
from robotic_cutting_sim.tasks import make_task

task = make_task("CUT-01", seed=42, variant="fixed")
points_local, masses_kg, lattice = task.sample_particles()
action = task.reference_action(3.0)
action = task.validate_action(action)
edge_world = task.reference_edge_pose_world(3.0)
hand_world = task.reference_hand_pose_world(3.0)
```

`CuttingTaskSpec` exposes `geometry`, `particle_spacing_m`, `grid_dx_m`, `fixture_boxes_local`, `world_from_workpiece_xyzw`, `table_top_world_z_m`, `hand_from_edge_xyz_xyzw`, `duration_s`, `control_dt_s` (also readable as `control_dt`), `material`, `goal_kind`, target origin/normal and blade dimensions. `FixtureBox` stores a local centre and positive half extents.

The action mapping has `mode`, `target`, and optional Boolean `cut_enabled`. Supported modes are `cartesian_edge_pose_world` (one7D XYZ+XYZW **edge** pose) and `joint_position_rad` (active-arm7D, leaving the other arm parked, or both-arm2×7D). `validate_action()` validates data; the live adapter owns IK, dynamics, joint limits and sensor observations. There are no force/oracle fields in this action contract or any observation generated by this module.

`variant="fixed"` preserves identical geometry across seeds. The implemented
`variant="randomized"` uses a separately frozen distribution for bar length,
local cutting position, static XY/yaw and continuous bump families/counts.
Its goals and supports remain in the workpiece frame; see
[TASK_RANDOMIZATION](TASK_RANDOMIZATION.md) for seed and distribution evidence.

## Offline goal evaluator

`task.evaluate_oracle(recording)` consumes recorded **world** particle/reference positions, immutable masses, cut-code history, timestamps and actual eight-node stencil weights/**stress gradients**. It recomputes mechanical connectivity, including code-zero bridges. Component IDs inferred from visual gaps, side codes alone, missing gradients or renderer masks cannot replace this graph. A short smoke returns `NOT_RUN` for task completion, while retaining its numeric diagnostics.

CUT requires an initial single component/zero codes, unchanged reference mass/count, persistent nonzero digits and at least0.5 s disconnected post-cut topology. Two substantial parts must each contain at least40% of initial mass and together at least99%; any minor components remain counted and mass-reported. Actual severed particle-pair/node midpoint distances must remain within2 mm of the workpiece-local target plane. This statistic uses actual reference support, not a six-neighbour shortcut.

TRIM separately requires at least90% of **reference particle-centre** mass initially above target to detach. Final flatness requires an actual current mechanically attributed substrate triangle mesh and matching physical timestamp. Full-footprint1 mm rays retain missing regions and all residual errors. Frozen limits are1 mm peak/gouge,0.5 mm p95/RMS,≤1% overcut and≤1% missing surface. Particle-centre heights or nominal analytic bump volume do not substitute for those mesh measurements.

Evaluator outcomes are `PASS_TASK_METRICS`, `FAIL_TASK_METRICS` or `NOT_RUN`. Even a pass sets `full_simulation_acceptance=False`; it certifies neither native execution nor cameras, controller quality, complete finite/contact screens, material calibration or training success. These checks run on oracle recordings outside future policy observations.

## CPU evidence and current boundary

```bash
.venv/bin/python -m pytest -q tests/test_cutting_tasks.py
.venv/bin/python scripts/audit_task_geometry.py --output-dir artifacts/task_geometry_fresh
```

18 meaningful CPU tests pass: exact full-cell bar mass/bounds, a continuous base/dome union, mount direction/holder clearance, fixed-seed provenance, rigid frame invariance, both action modes, complete/partial/wrong-location reference cuts, persistent history/mass loss, gradient requirements and an attributed flat-versus-raised triangle-surface control. These analytic tests do not claim a solver rollout.

The CPU generator receipt is [`cutting_task_geometry.json`](evidence/cutting_task_geometry.json). Its single six-neighbour geometric component is deliberately labelled **geometric**; only live actual stencils can establish mechanical cutting. Generated reference poses are planned/derived, never presented as measurements. Initial two-step Lab/Kit smokes preserved a task-evaluation JSON scalar serialization failure after real physics/capture; the portable evaluator now returns native Python Boolean checks. Full-duration physical task, native camera and finite/contact acceptance are recorded independently by the live adapters.
