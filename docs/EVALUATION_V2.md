# Evaluation v2 — integrity, cutting outcome and geometric quality

Evaluation v2 is a read-only research tool. It never decides whether a scene can
load, a tool can contact material, a policy can act or a failed demonstration can
be saved. It consumes recorded particle/history/mesh state and reports three
independent layers. The supplied [WP-09 specification](WP09_SPECIFICATION.md)
and versioned [protocol](../configs/evaluation_v2.json) define the scope.

## Diagnosis of the original surface score

`scene_validation.mesh_heightfield_metrics()` samples vertical intersections
over a rectangular workpiece footprint. Its historical conservative deficit
uses height zero for a missing intersection. A missing ray against a 20 mm
target therefore contributes a 20 mm deficit. This is an explicit legacy
penalty, not a measured knife gouge. That implementation and
`cutting_tasks_v1.json` remain unchanged.

The independently built, uncut 2 mm flatbar has 4,788 measured intersections
out of 4,800 rays. Relative to the analytic plane, paired p95 error is
1.526435 mm and maximum measured mesh deficit is 1.855862 mm. Twelve rays
remain missing. Its exact represented-self comparison has zero paired error,
while retaining every missing mask. No knife or material dynamics produced
these errors. The [control receipt](evidence/evaluation_v2_validation.json)
links raw arrays, target hashes and the exact reproduction commands below.

V2 separates the following quantities without replacing missing values:

| Quantity | Meaning and limits |
| --- | --- |
| `measured_maximum_surface_deficit_m` | Largest measured vertical mesh deficit at an actual paired intersection; reconstruction and physical deformation are still confounded |
| `paired_p95_absolute_error_m`, `paired_rms_error_m` | Conditional statistics over explicitly counted paired rays; never a full-region accuracy claim when coverage is incomplete |
| Missing actual / unknown target columns | Counts, fractions, locations and separate validity masks; heights remain NaN in the raw NPZ |
| `unknown_missing_column_volume_budget_m3` | Conservative geometric budget over missing actual columns with a known target height, measured from the explicit bottom; unknown target columns remain separate |
| `whole_region_deficit_volume_m3` | `null` whenever any required actual or target intersection is missing |
| Protected reference material detached mass | Actual particle reference mass below/outside the requested removal region that is no longer attached to bottom-anchored substrate components |
| `true_material_gouge` | `NOT_MEASURABLE_FROM_RECONSTRUCTED_SURFACE_ALONE`; a mesh deficit alone cannot establish physical knife damage |

Particle quadrature, reconstruction, missing boundary intersections and actual
deformation are not uniquely identifiable from one final mesh. V2 reports
that uncertainty. Analytic and independently frozen represented targets have
separate names, source/version/hash and results. No target, bias or registration
is fitted to a cut result. The represented target is diagnostic; it never
replaces the original analytic precision benchmark.

## Three independent layers and profiles

**Physical integrity** checks aligned finite recorded states, positive masses,
time order, irreversible history, archived source verification and traceability.
It recomputes the actual MPM graph from current codes, node weights and stress
gradients. Separation is reported, not required by integrity. A valid uncut or
unsuccessful demonstration can therefore remain useful. Incomplete provenance
returns `NOT_MEASURABLE_INCOMPLETE_PROVENANCE`; invalid numeric/history state
returns `FAIL`. Reference mass/count checks are distinguished from independently
recorded per-frame masses. `PASS_RECORDED_STATE_INTEGRITY` does not certify a
calibrated material model, converged coupling energy or full WP-08 physics.

**Cutting outcome** reports component counts/masses, sampled separation
persistence, cut position, target reference-mass removal, protected-material
detachment and each bump's removal. Overlapping bumps use initial analytic
height argmax ownership, with stable index ties, so mass is never counted twice.
No future component labels enter earlier frames. Reference particle-centre
measurements retain their resolution limits and units.

**Surface quality** reports geometry and coverage independently of outcome.
Full-object, interior, boundary and initial bump ROIs are defined from the goal
before evaluating the result. The predeclared 3 mm boundary diagnostic does
not remove boundary samples from the full-object metric. Capsule domains keep
every bounding-grid ray and additionally record the initial domain mask and
outside-domain counts. Explicit box/ellipse/capsule ROIs support local goals.

| Profile | Goal assessment | Surface assessment |
| --- | --- | --- |
| Diagnostic | Measurements only; `success=null` | Descriptive metrics and measurability |
| Research | Existing task outcome conditions, or explicit versioned caller thresholds | Measured independently; no universal flatness gate |
| Precision | Same outcome reported separately | Original v1 precision score, checks and metrics retained under `legacy_precision_benchmark` |

Research adds **no new universal threshold**. Existing CUT uses the frozen two
substantial parts, mass, location and persistence criteria. Existing TRIM uses
its original 90% reference-region removal plus topology/mass/history conditions.
Original precision still requires its unchanged 0.5 mm p95/RMS, 1 mm peak/gouge,
1% overcut/missing and other checks. Custom thresholds are explicit metric-path
min/max rules in a versioned goal; missing metrics never pass. Custom goals do
not silently borrow a legacy task's successful outcome. Incomplete geometry
cannot pass a requested surface-quality threshold.

`episode_completed`, `physical_integrity.status`,
`profile_assessment.goal_status`, `profile_assessment.surface_quality_status`,
`legacy_precision_benchmark.status` and `success` are distinct fields.
Research success means the declared research outcome passed with sufficient
recorded-state evidence; it does not assert precision or full physical fidelity.

## API and target representations

The implementation is three small modules:
[evaluation.py](../src/robotic_cutting_sim/evaluation.py),
[surface_evaluation.py](../src/robotic_cutting_sim/surface_evaluation.py) and
[surface_distance.py](../src/robotic_cutting_sim/surface_distance.py).
There is no new simulator, registry or policy pipeline.

```python
# Existing behavior and historical benchmark remain available.
old_score = task.evaluate_oracle(recorded_data)
old_live_score = env.evaluate()

diagnostic = task.evaluate(recorded_data, profile="diagnostic")
research = task.evaluate(recorded_data, profile="research")
precision = task.evaluate(recorded_data, profile="precision")
live_metrics = env.evaluate(profile="diagnostic")

from robotic_cutting_sim.evaluation import evaluate_recording
result, raw_arrays = evaluate_recording(recorded_data, task=task,
                                       profile="research")
```

Live in-memory measurements explicitly lack the offline archive/contact audit;
their numeric metrics remain usable, while missing source validation cannot
become an overall research PASS. V2 data is privileged evaluation/debug output
and is never inserted into vision observations. Evaluation never changes
particle state, masses, forces, actions, cutting history or meshes.

`GoalSpec` specifies goal type, target/frame transform, version/source, region,
applicable metric names, optional thresholds, duration and sampling resolution.
Supported goal semantics are `sever`, `trim_surface`, `remove_region`,
`shape_to_target` and `diagnostic_only`. A removal domain is fixed before
evaluation and selects the initial reference region; detached material outside
that target remains in the protected-material ledger.

`TargetGeometry` supports analytic planes, sampled nonflat heightfields,
triangle meshes and a runtime signed-distance callable. Height comparisons
require a nonvertical target surface. Mesh targets instead use bidirectional
nearest-triangle queries at every referenced vertex and triangle centroid,
with full area-weighted p95/RMS quadrature. Sampled maxima are not continuous
Hausdorff bounds. Open meshes have no claimed enclosed-volume difference.
Implicit targets provide directed area statistics and explicitly lack reverse
coverage without target-surface samples. A callable's source/version is saved;
its executable implementation must be supplied separately for reproduction.
Coordinate-frame mismatch is `NOT_MEASURABLE`; registration is never fitted.

These geometry extensions do not add arbitrary-CAD or universal-material
cutting to TLMPM. Current actual cutting remains the independently supported
continuous procedural heightfield/capsule material pipeline.

## Actual recorded results

All historical status/check/metric values match their original recorded scores
exactly. Both new full camera-v4 CUT episodes use the actual independent tool,
400 control steps, 4,000 formal rigid substeps and three RGB-D streams.

| Recording | Outcome | Research | Original precision |
| --- | --- | --- | --- |
| Fixed 2 mm Lab TRIM | 98.5294% target mass detached | PASS declared outcome | FAIL |
| Fixed 2 mm Native TRIM | 98.5294% target mass detached | PASS declared outcome | FAIL |
| Native randomized TRIM, seed 42 | 96.2791% overall; 94.9153%, 96.9697%, 96.6667% per initial bump | PASS declared outcome | FAIL |
| New camera-v4 Native CUT | Two 72 g parts; original location/persistence conditions | PASS | PASS |
| New camera-v4 Lab CUT | Two 72 g parts; original location/persistence conditions | PASS | PASS |
| New camera-v4 Native TRIM | Physical chip separation and measured target removal | PASS declared outcome | FAIL |
| New hand-aligned camera-v5 Native CUT | Two 72 g parts; unchanged task conditions | PASS | PASS |
| New hand-aligned camera-v5 Native TRIM | 98.5294% target mass detached | PASS declared outcome | FAIL |

The three-bump recording has actual separated topology; its geometry quality
remains independently reported. Fixed Lab/Native analytic paired p95 values
are approximately 1.673/1.675 mm. Against the independent represented target,
paired p95 is approximately 0.421/0.368 mm, with missing target/actual coverage
still explicit. These conditional values are not replacement precision PASS
claims. The fixed Lab example has a measured mesh deficit up to 14.468 mm at
boundary geometry, distinct from the legacy 20 mm missing-ray penalty and
from physical below-target detached mass, which is zero.

Known-geometry tests cover uncut, partial removal, no removal, obvious overcut,
missing mesh, boundary gaps, mismatched frames, missing physics provenance,
nonfinite state, mass loss, rewritten history, sloped targets, curved implicit
targets and bidirectional mesh distance. No-data cases cannot pass. CPU tests
are analysis-code evidence, separate from actual simulator recordings.

## Reproduce and export

Use fresh output directories. The evaluator returns exit zero for a completed
analysis even when a task fails; inspect the separate result fields.

```bash
.venv/bin/python scripts/evaluate_episode_v2.py \
  artifacts/isaac_lab_trim01_coarse_001 artifacts/evaluation_lab_NEW \
  --represented-target artifacts/represented_flatbar_target_002
.venv/bin/python scripts/evaluate_episode_v2.py \
  artifacts/isaacsim_trim01_coarse_001 artifacts/evaluation_native_NEW \
  --represented-target artifacts/represented_flatbar_target_002
.venv/bin/python scripts/evaluate_episode_v2.py \
  artifacts/isaac_native_randomized_benchmark_001/trim-01_randomized_seed42 \
  artifacts/evaluation_randomized_NEW
.venv/bin/python scripts/validate_surface_evaluation_v2.py \
  artifacts/represented_flatbar_target_002 artifacts/evaluation_controls_NEW
.venv/bin/python -m pytest -q
```

Each output saves three profile JSON files, all raw ray/ROI/validity arrays,
optional bidirectional 3D samples, source snapshots/hashes, input hashes and
input/source immutability checks. `episode_manifest_v2.json` joins the task,
seed, initial geometry, goal/profile/version, time and evaluation files to
original observation/action/trajectory assets. It references original captured
assets rather than regenerating them. Partial/failed recordings can still be
analyzed and exported; absent physics remains explicitly unmeasurable.

The public [validation receipt](evidence/evaluation_v2_validation.json) contains
exact artifact paths and hashes. Raw recordings and failed iterations remain
local under ignored `artifacts/`; existing published evidence is unchanged.
