# Seeded CUT/TRIM task variants

`make_task(..., variant="randomized", seed=...)` now generates bounded variations of the shared CUT-01 and TRIM-01 definitions. Fixed tasks and their default 1 mm sampling are unchanged. The randomization protocol was defined before randomized live episodes; [the CPU generation receipt](evidence/task_randomization.json) records complete actual TaskSpecs, seeds, parameter draws, numerical overrides and source hashes. Runtime success is scored in separate episode receipts.

```python
from robotic_cutting_sim.tasks import make_task

task = make_task("TRIM-01", seed=42, variant="randomized",
                 particle_spacing_m=.002)  # Explicit numerical override.
local_points, reference_mass_kg, lattice = task.sample_particles()
action = task.reference_action(3.0)
recorded_configuration = task.to_dict()
```

The new [protocol](../configs/task_randomization_v1.json) uses `numpy.random.Generator(PCG64(seed))`, with an explicit draw order. It does not mutate global random state. Material, blade/mount definitions, control interval, duration and caller-selected particle spacing are inherited from the fixed task. No component labels, deleted particles, force observations or trained policy are generated.

| Parameter | Randomized range |
| --- | --- |
| Bar length | 140, 160 or 180 mm; width 30 mm, base height 20 mm |
| Workpiece frame offset | X/Y ±10 mm and yaw ±10°, Z offset zero |
| CUT plane | Local X in −8 to +8 mm, at 2 mm increments |
| CUT supports | From each outer bar edge +10 mm margin to a slot extending ±20 mm around the cut |
| TRIM bump count | 1–3, over one continuous base |
| TRIM bump X | Distinct anchors −30 / 0 / +30 mm, each with ±2 mm jitter |
| TRIM bump Y | ±1 mm |
| TRIM bump X/Y radii | 8–12 / 6–11 mm |
| TRIM bump height | 6–10 mm |
| TRIM shapes | Dome, Gaussian, ridge or compact asymmetric `irregular` cap |
| TRIM reference sweep | Covers all bump bounds plus 30 mm margins, clipped 2 mm inside bar ends |

The seeded transform is composed with the caller's input workpiece frame: `T_world_actual = T_world_input @ T_input_offset`. Geometry, fixture corners, target planes and edge/hand reference poses transform together. Offsets are expressed in the input frame; an arbitrary tilted input frame therefore rotates their world translation too. The current live randomized examples use a horizontal input frame. Pure coordinate covariance does not certify dynamics for arbitrary workpiece orientations.

For the Lab backend, randomized geometry and fixtures are constructed with the environment. Recreate the environment to use a different randomized seed; changing that seed on the same Lab environment's reset is explicitly rejected. An ordinary reset retains the constructed seed. The native Kit backend can rebuild its stage and resample on a changed seed. These CPU generation tests do not certify either backend's reseeding or reset repeatability.

CUT's smallest possible analytic half contains more than 44% of bar mass, above the original 40% substantial-piece threshold. Its full-plunge height, 35 mm blade, diagonal hand mount and 40 mm open support slot are preserved. TRIM retains local target Z=20 mm and commanded edge Z=20.2 mm, with the original 8 s knot times. Variable sweep lengths change speed, so the reference path is a baseline to test, not a guarantee of removal across all shapes or seeds.

`irregular` adds no Bump fields. Its compact cosine cap has a smoothly varying exponent, producing asymmetric height while remaining nonnegative, bounded by the supplied height, and zero outside the declared elliptical support. The legacy `FAMILIES` draw sequence remains exactly `(dome, gaussian, ellipsoid_cap, ridge)`. Existing D0–D3 choices and their original family formulas are unchanged. Eight specimen/specification/mass/lattice hash controls captured from commit `51e5bfe` before this change verify that boundary.

The CPU receipt uses explicit 2 mm particles and contains seeds 42, 43 and 2026 for each task. Seed 42 generates a 140 mm bar with frame translation (−1.2224, +7.1720) mm and yaw +3.94736°. CUT uses local X=+4 mm and 10,500 particles / 126 g. TRIM generates dome / irregular / dome bumps, sweeps X≈−67.468 to +68 mm, and has 10,715 particles / 128.580 g. Each initial CPU specimen has one geometric six-neighbour component, unique lattice cells and an unchanged single-base union mass; actual MPM continuity still requires the runtime's recorded eight-node stress-gradient graph.

```bash
.venv/bin/python -m pytest -q \
  tests/test_task_randomization.py tests/test_cutting_tasks.py \
  tests/test_task_episode_audit.py tests/test_benchmark_foundation.py
```

All 69 tests pass. Controls cover reproducibility/different seeds; complete base quadrature and unique mass; overlapping-bump union; target/action/fixture covariance; cut-slot mass fractions; bump containment and sweep coverage; irregular support/asymmetry; legacy D0–D3 exact outputs; invalid seeds; and fixed-task preservation. The standalone CPU artifacts are under `artifacts/task_randomization_001` and contain initial particle/mass/lattice NPZs and reference-action NPZs. They do not establish actual robot execution, physical cutting, registered RGB-D, or randomized task success.

## Actual native seed-42 episodes

Subsequent actual Isaac Sim 6.2 runs are now independently audited in [the episode report](reports/RANDOMIZED_TASK_EPISODES.md) and [public receipt](evidence/randomized_native_task_episode_audit.json), with raw recordings under `artifacts/isaac_native_randomized_benchmark_001`. Each executes 400 actual control steps / 4,000 PhysX steps over 8 s and records 41 native RGB-D frames. Task check, metric and status dictionaries exactly recompute from actual saved stencils, reference masses, history, measured poses and current surfaces; all source hashes resolve.

| Actual one-seed case | Numerical sampling | Physical outcome | Original strict task score |
| --- | --- | --- | --- |
| CUT-01 randomized seed42 | 1 mm particles / 4 mm grid | 1 → 2 components, 66.6 + 59.4 g, unchanged 126 g mass | PASS_TASK_METRICS |
| TRIM-01 randomized seed42 | Explicit 2 mm particles / 4 mm grid | 1 → 4 components; three chips; 96.2791% above-plane mass removed | FAIL_TASK_METRICS: surface-quality errors retained |

Both start continuous with zero labels, preserve nonzero cut history and retain their final disconnected topology throughout the saved 2.4 s postcut window. Native TRIM's positive residual is 0.1837 mm, but full-footprint p95/RMS error is 1.6039 / 0.5504 mm and overcut is 1.9541%; no tails or missing rays are discarded. Its detached chips can still contact the tool after reference withdrawal. CUT's interface-work residual remains 15.91%, above the earlier 10% screen; TRIM's 2.36% residual is only one coupling diagnostic.

These are one observed example per task, with different declared particle spacings. They do not establish a success rate, generalization, repeatability, convergence, full native-camera acceptance or completion of the 18-gate benchmark. Fixed and earlier failed recordings remain unchanged.
