# Task material frame and fixture controls

`PASS_FRAME_AND_FIXTURE_CONTROLS`, seed 42, actual CUDA0 run
`artifacts/task_frame_fixture_001`. The protocol and tolerances in
[`probe_task_frames.py`](../scripts/probe_task_frames.py) were committed as
`2a7b1eb` before numerical execution. Protocol SHA256:
`b8019654b617c2132f4ce8377fab5cbeb1e0896fb4445a87f6f72e10a2c111c2`.
The [public receipt](evidence/task_frame_fixture_controls.json) includes exact
protocol, source/package hashes, command, timings, measurements and raw artifact
hashes. No shared solver, adapter, fixture or task source was changed by this audit.

```bash
.venv/bin/python scripts/probe_task_frames.py artifacts/task_frame_fixture_001 --device cuda:0
```

The actual CUT-01 material is 96,000 particles, 144 g, 1 mm spacing on a 4 mm
reference grid. Two actual material adapters receive the same five-interval
LOCAL measured blade segment, from `(-0.8,0,21)` to `(0.8,0,18)`mm. One workpiece
uses the identity frame; the other uses a 37° WORLD yaw and translation
`(170,-110,0)`mm. LOCAL positions, velocities, total F, codes, reference particles,
masses, lattice and measured poses are retained in both NPZ files. Cutting is
disarmed to isolate coordinate/contact covariance; this is a prescribed blade
contact control, not an articulated robot or complete severing episode.

| Measurement | Actual | Frozen limit |
|---|---:|---:|
| Maximum LOCAL position difference | 0.0959 µm | 20 µm |
| Maximum LOCAL velocity difference | 0.0140 mm/s | 2 mm/s |
| Maximum total-F Frobenius difference | 6.31e-6 | 2e-3 |
| Maximum LOCAL force-impulse difference | 5.90e-9 Ns | 1e-5 Ns |
| Maximum LOCAL torque-impulse difference | 1.07e-10 Nms | 1e-6 Nms |
| Peak contact force | 0.14833 N | greater than 0.0001 N |
| Horizontal WORLD force rotation effect | 0.0921 N | greater than 0.00001 N |

The initial samples/masses/lattice are identical, all codes remain zero, all
states are finite, WORLD position/velocity round trips pass, and the returned
WORLD force and torque match rotation of the actual LOCAL reaction. The wrench
is rotated at the same physical blade edge; no origin shift or translation
moment is appropriate here. The subsequent articulation adapter separately
handles the lever arm from the edge to its force application point.

The direct GPU controls launch the actual `fixture_contacts` Warp kernel using
the actual adapter's uploaded finite boxes/table plane and contact parameters.
Expected impulses use known flat-face gaps and a separate scalar normal/friction
ledger. Pedestal top and WORLD table contacts produce positive physical impulses;
maximum analytic discrepancy is 5.22e-13 Ns. A point below the pedestal's top but
outside its finite bounds produces exactly zero impulse. A point at the plunge
slot center, 3 mm below the workpiece base and 7 mm above the WORLD table, also
produces exactly zero impulse. There is no implicit tray floor across that slot.

A separate tilted translated adapter uses Euler XYZ `(17,-11,29)`° and WORLD
translation `(120,-80,35)`mm. Its actual configured LOCAL gravity is
`(-1.871836,-2.815470,-9.208988)`m/s², matching `Rᵀ g_world`. Its uploaded LOCAL
table plane represents WORLD `z=-10 mm`: signed-distance residual is at most
2.21 nm on three independent points. Gravity is an **analytic configuration
receipt**, without a tilted material episode. The transformed table kernel is
actually executed with a known WORLD particle/velocity, and its returned WORLD
impulse differs from the independent analytic value by 5.34e-13 Ns.

All probes completed in 6.46 s wall time while other jobs used the GPUs; this is
not an exclusive performance benchmark. This validates static frame plumbing
and finite contact behavior. It does not validate moving workpiece frames,
tilted material dynamics, fracture convergence, robot tracking, cameras or full
CUT/TRIM task success. Full native task episodes are audited separately.
