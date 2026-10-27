# Benchmark foundation: continuous D0–D3 recipes and recorded-run diagnostics

2026-10-28, Asia/Seoul. **Scenario generation and offline metrics work;
the complete simulation benchmark is not ready.** This module does not launch
a solver, change an observation, implement a policy, or certify cutting.

New files:

- `src/robotic_cutting_sim/benchmark.py`: continuous material recipes, sampling,
  provenance, independent metrics and explicit recording comparison.
- `scripts/run_benchmark_foundation.py`: standalone preparation/analysis commands;
  refuses to overwrite previous output directories.
- `tests/test_benchmark_foundation.py`: 24 analytic CPU cases, all executed/PASS.
- `docs/reports/benchmark_foundation_evidence.json`: compact executed receipts.

Existing geometry, environment, command-line interface and observations were not
modified. A backend must explicitly apply each recipe parameter and declare
unsupported parameters before a prepared scenario can count as a simulation run.

## Scenarios and sampler

`BenchmarkSpec.generate(seed=42, variant="D0")` contains `HeightFieldSpec`,
`MaterialParameters`, and tool/camera condition metadata.
`sample_specimen(spec, spacing_m)` returns positions, mass, lattice indices and
a provenance receipt. There are no initial chip IDs or preassigned cut labels.

| Variant | Generated conditions |
|---|---|
| D0 | Fixed centered dome and fixed provisional material; geometry independent of seed |
| D1 | One seeded bump with random family, position, radii and height |
| D2 | Three seeded bumps with different locations, families and sizes |
| D3 | Two–five seeded bumps; density, stiffness, Poisson ratio, yield stress, hardening and friction variation; blade thickness and camera pose perturbation recipes |

Supported bump families are dome, compact Gaussian, ellipsoid cap and ridge.
Every bump forms part of one continuous base; overlapping height functions use
their maximum rather than generating overlapping independent bodies. The cell
sampler evaluates that one height function once. Its receipt hashes positions,
masses, lattice indices and the full scenario, and records spacing, count and
mass. D0 at2 mm spacing is byte-identical to the existing generator's positions,
mass and lattice, preserving the working baseline.

All materials remain `provisional_uncalibrated`. Optional failure-model name,
critical stress and fracture-energy fields are configuration slots; defaults
are `not_configured` / null. Generating those fields does not implement a failure
law. Likewise D3 camera transforms and tool thickness are recipes until consumed
by the appropriate backend.

Executed:

```bash
.venv/bin/python -m pytest tests/test_benchmark_foundation.py -q
# 24 passed

.venv/bin/python scripts/run_benchmark_foundation.py prepare \
  --variants D0 D1 D2 D3 --seeds 42 43 \
  --output-dir artifacts/benchmark_foundation_001/scenarios
```

| Variant | Seed42 particles / mass | Seed43 particles / mass |
|---|---:|---:|
| D0 | 19,340 /0.232080 kg | 19,340 /0.232080 kg |
| D1 | 19,401 /0.232812 kg | 19,308 /0.231696 kg |
| D2 | 19,386 /0.232632 kg | 19,336 /0.232032 kg |
| D3 | 19,303 /0.244966 kg | 19,469 /0.275015 kg |

All eight saved initial states were read back and had exactly one six-neighbor
sampled component. This checks initial connected geometry only. It does not prove
solver mechanical connectivity after cutting, and these eight scenarios have
not been simulated by this runner.

## NPZ contract and metric meaning

`analyze_recording(path, MetricConfig(...), run_metadata=None)` accepts:

| Field | Meaning |
|---|---|
| `positions`, `time_s` | Required finite arrays `(T,N,3)`, `(T,)`; timestamps strictly increasing |
| `mass_kg` | Required `(N,)` or `(T,N)`, finite and nonnegative |
| `velocities` | Optional finite `(T,N,3)` |
| `tool_pose_xyzw` | Optional frame-aligned `(T,7)`, with unit XYZW quaternions |
| `tool_half_extents` | Optional positive3-vector for oriented-box proxy diagnostics |
| `component_ids` | Optional integer `(N,)` or `(T,N)` grouping; −1 means unclassified |
| `component_provenance` | Required scalar string before component masses are reported |
| `substrate_anchor_mask` | Optional Boolean `(N,)` reference-substrate mask |

Without an explicit anchor mask, the initial lowest particle layer identifies
reference-substrate components. This is not proof of attachment to the fixture.
Components containing those reference anchors form the substrate group; other
source-reported components form the detached group. Unknown labels remain in the
surface estimate and in an explicit unclassified mass budget. Component masses,
centers of mass and linear momenta are measured from the recorded data. Raw cut
side codes never substitute for component IDs. Text provenance does not certify
the source's component algorithm: persistence and recontact remain null here.

Mass reporting separates measured per-frame masses from a static mass array.
For static arrays, a zero computed temporal mass change simply reuses the same
input masses and is not an independent mass-conservation measurement. Mass outside
the original XY footprint is reported as displaced mass/reference volume, not
as physically detached chip mass. Missing component evidence leaves detached
mass unavailable rather than inferring a cut from motion or a visual gap.

Height, roughness and overcut are explicitly **particle-bin proxies**. Each XY bin
uses maximum particle-center Z plus a configured support radius; empty bins have
zero height. Surface metrics use source-reported reference-substrate components
plus unknown particles when component labels are available, and all particles
otherwise. This selection affects only offline statistics, never simulation
states or visibility. Detached chips may occlude the substrate when labels are
missing. Coarse sampling, bin holes, support-radius assumptions and initial
quadrature can bias these values. Mesh/RGB-D quality metrics and refinement remain
separate work.

Tool penetration uses the recorded oriented bounding box and particle centers.
It is not an exact wedge/SDF contact certificate, nor particle-radius-aware.
Kinetic and gravitational energies can be computed, but total energy balance
remains `NOT_EVALUATED` without strain energy, tool/fixture work and dissipation.
Reported timing uses only an explicitly supplied positive `wall_s` value.
No force field is used or added to policy observations; these diagnostics are
an offline oracle interface.

## Executed comparison of existing real solver recordings

```bash
.venv/bin/python scripts/run_benchmark_foundation.py compare \
  --run newton=artifacts/newton_settle_002 \
  --run cressim_corrected=artifacts/CRESSim_candidate/verification_002/settle \
  --metadata newton=artifacts/newton_settle_002/result.json \
  --metadata cressim_corrected=artifacts/CRESSim_candidate/verification_002/settle/result.json \
  --particle-support-radius-m .001 \
  --output-dir artifacts/benchmark_foundation_001/settle_comparison
```

Both existing 121-frame recordings were processed successfully. Initial positions,
initial masses and timestamps match; recorded tool poses differ because the
adapters use different tool representations. Material/contact semantics are not
controlled, so the comparison does not rank solvers or report a speedup.

| Offline settling diagnostic | Newton | Corrected CRESSim |
|---|---:|---:|
| Residual height proxy | 6.8989 mm | 6.9484 mm |
| Roughness standard-deviation proxy | 0.7643 mm | 0.7671 mm |
| Overcut proxy | 0 m³ | 0 m³ |
| Occupied XY bins | 100% | 100% |
| Component/chip mass | NOT_AVAILABLE | NOT_AVAILABLE |

These are measurements of existing settling recordings, not cutting successes.
The supplied Newton receipt lacks the specific `wall_s` key, so its normalized
timing is unavailable. The code does not silently substitute a different timing
scope. Corrected CRESSim's receipt explicitly supplies1.93644 s.

`compare_recordings` records input hashes and a match check; a matching set does
not establish a frozen paired protocol. `solver_ranking` and `formal_cutting_success`
remain null and `benchmark_ready` remains false.

## Validation and remaining work

Analytic tests check deterministic seeds, connected multi-bump sampling, overlapping
unions, parameter rejection, surface-bin residual/void volume, mass loss detection,
component mass/COM/momentum accounting, unknown-label preservation, refusal to infer
chips from side codes or missing provenance, rotated tool-box geometry, invalid
recording rejection and comparison boundaries. The toy fixtures are labeled
analytic controls, never physical trials.

Pending integration: run D0–D3 through a declared backend, freeze training/selection
versus held-out seeds if future research needs them, measure actual material/camera
condition application, validate persistent mechanical separation and chip contact,
freeze convergence/quality thresholds, add measured mesh/RGB-D geometry checks,
and align robot, solver and sensor timestamps. This foundation cannot promote
any unfinished stage to a formal benchmark pass.

The TLMPM topology review was read-only. A side compatibility graph can transmit
momentum indirectly through an unassigned code0 field, so reference six-neighbor
bonds alone cannot prove disconnection. The backend's component audit should use
its actual stencils and account for nonzero-gradient supports, activity cutoffs
and any remaining unassigned-field bridges. This module consumes explicit source
component provenance and does not independently certify those internals.

Raw preparation states, manifests and recorded-run reports are retained locally
under `artifacts/benchmark_foundation_001`; the compact receipt contains hashes.
No existing failed experiment or solver output was overwritten.
