# Single finer-reference-grid TRIM diagnostic

**FAIL_PHYSICS_ONLY_CANDIDATE.** The single frozen 2 mm reference-grid diagnostic
completed the original 400-step, 8 s TRIM trajectory but remained one mechanical
component. Detached reference-above-plane mass fraction is zero. The negative
result is preserved, and this grid branch is stopped without further tuning.

Commit `f1841ef` froze the hypothesis, physical criteria, unchanged v1 surface
limits and implementation before numerical output. Protocol SHA256:
`f90ad10a916bec9db5a4918cb3b83275605fb872d68165b72656757972b9eaa5`.
The [public receipt](evidence/trim_grid_2mm_diagnostic.json) retains protocol,
setup, source/package hashes, full contact/reaction trace and negative checks.

```bash
.venv/bin/python scripts/probe_trim_grid.py artifacts/trim_grid_2mm_001 --device cuda:0
```

This physics-only `LiveCuttingMaterial` run changes the exposed reference-grid
input from 4 mm to 2 mm using `dataclasses.replace` on the original TRIM task.
It retains all 97,044 particles at 1 mm spacing, 145.566 g mass, original material,
fixtures/table, 20.2 mm cutting plane, blade dimensions, 0.1 ms material timestep,
20 ms control interval, 8 s reference trajectory and H2 surface parameters.
Nominal bulk quadrature changes from 64 to 8 particles per reference cell. No
robot, Isaac runtime, sensor, training or actual articulation feedback is present.

The existing source also derives assignment normal band `6*dx`, assignment rear
window `2*dx` and particle-contact reference exclusion `3*dx` from this input.
Those become 12/4/6 mm, respectively. These automatic changes are retained and
recorded; this is a single-input grid diagnostic, not a transfer-grid convergence
proof with every physical window fixed. No shared core code was changed.

| Frozen physical measurement/check | Actual |
|---|---:|
| Initial → final actual stencil components | 1 → 1: FAIL |
| Detached initial above-plane mass fraction | 0% versus required 90%: FAIL |
| Fully withdrawn interval, 6.4–8 s | 1 component for 1.6 s: FAIL |
| Existing nonzero digit rewrites | 0 |
| Particle count/mass vector | Unchanged; uploaded mass vector exactly equal |
| All recorded control-boundary states/reactions | Finite |
| Minimum recorded total-F Jacobian | 0.881632 |
| Maximum recorded particle speed | 0.429409 m/s |
| Maximum material reaction force | 0.097659 N |
| Tool-contact pairs after full withdrawal | 0 |
| Abort guard fired | No |

Components are recomputed from the actual reference eight-node stencils,
including positive shape weights, nonzero stress gradients and code-zero
bridges. Final history counts `[62,170, 33,860, 1,014]` for codes `[0,1,2]` do
not establish a separate cap. No deletion, label repair, presegmentation or
forced fragment motion is applied. Generic adapter previous-wrench fields are
proposals for a caller; no robot exists here to receive those forces.

The final H2 surface is scored with the original v1 whole fixed footprint.
Maximum positive residual is 7.696835 mm, p95 absolute error 1.104259 mm and
RMS 0.943639 mm. Two of 4,800 footprint rays are missing. Their unchanged
conservative accounting gives a maximum target-plane deficit of 20 mm;
**this is not a measured 20 mm knife gouge**. The overcut-volume proxy is
0.854347% of initial base volume. Full v1 status remains `FAIL_TASK_METRICS`;
both physical separation and surface criteria fail. No ROI or idealized surface
replaces that result.

The local artifact retains all 401 control-boundary all-particle positions,
velocities and codes as `.npy` files, immutable reference/mass/lattice/stencils,
final F/Fpi/plasticity/contact work, actual fixture/tool geometry, full reaction
and contact trace, final H2 mesh and explicit current-component provenance.
Finite/Jacobian checks are at the saved 20 ms control boundaries; they do not
establish every-substep or energy convergence. Wall time was 110.36 s,
nonexclusive. The runner exits nonzero for this recorded physical failure.
Any future candidate requires separate actual Isaac Lab/Sim episodes before
runtime/default promotion; this negative branch introduces neither.
