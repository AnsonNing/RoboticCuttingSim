# Fixed represented-target geometry diagnostic

Both actual coarse TRIM episodes retain **FAIL** under the original analytic
Task v1 and the separately versioned represented-solid column goal. The latter
uses an independent, fixed, uncut flatbar target and all 4,800 footprint columns.
Descriptive whole-3D distances are also reported, with no new acceptance gate.
No source episode, original scorer, physics or observation is modified.

The frozen contract is
[`represented_flatbar_target_v1.json`](../configs/represented_flatbar_target_v1.json),
protocol `represented_flatbar_2mm_whole_columns_v1_20261028`, SHA256
`6e1b4dc3f54d0d26fc386d986a903d48e0c423b2640c897535704adad3c735a4`.
Source commits are `94c8aa0`, `daefcaa` and `a7dfd25`.
The protocol file was written before unsolicited coarse-completion messages;
its Git commit follows those notifications. The commit is therefore not a
pre-runtime-completion freeze. No actual coarse arrays were opened until the
independent target was generated, controls checked and target file sealed.
Target parameters and thresholds were unchanged after those notices.

The independent no-bump 160 × 30 × 20 mm box uses 12,000 full-cell particles,
2 mm spacing, 144 g mass and identity total F. H2 parameters remain voxel 1 mm,
kernel 6 mm, density threshold 0.5, kernel scale 0.5, isotropic with all smoothing
disabled. No cut output determines target particles, mesh, alignment or radius.
The target file at `artifacts/represented_flatbar_target_002/fixed_target.npz`
has SHA256
`782d959f4dfb74f8658dc4b7017b68e0d7b7408f162d3181f34c45556016ec01`.
Target extraction took 23.8 ms and the control run 1.90 s, nonexclusive.

All 160 × 30 cell-centered rays use 1 mm spacing over the original full
footprint. Empty represented-solid columns have height zero. Both-empty columns
have zero represented-height difference and still count in actual, target and
union missing fractions. One-sided gaps keep the full target-column deficit;
target-empty, actual-filled columns keep their positive residual. No column,
edge or error tail is dropped. Limits remain 1 mm positive/gouge maximum,
0.5 mm p95/RMS and 1% overcut/missing. Overcut uses the original analytic base
volume as denominator. This is a distinct goal; it never replaces analytic v1.

The independent target has 12 empty columns, or 0.25%. Its exact-self control
has zero errors with all 4,800 columns retained. Both ±1.5 mm offset controls
fail as expected. A single additional missing actual column at the predeclared
center fails with a 19.999 mm deficit. The target itself still fails the
analytic-plane criteria: p95 1.526 mm, conservative maximum deficit 20 mm and
overcut-volume proxy 1.106%. Target build `001` is preserved as a constructor
failure: an explicit `bumps=()` was required for the no-bump measurement domain;
the independent box particle recipe and frozen parameters were unchanged.

Actual substrate triangles are selected using recomputed conservative MPM
stencils, stress gradients and reference-bottom anchored components. Recorded
labels are checked against those actual stencils. Every triangle of the
attributed substrate is retained; detached chip triangles are accounted for by
physical provenance. Each runtime has a 1.608 g detached component and
144.024 g substrate, 98.5294% above-plane mass detached and all required
topology/removal/mass/history/provenance checks true. Shape alone cannot promote
the combined goal, and these checks do not certify full numerical/camera/reset
acceptance.

| New full-column represented-target metric | Actual Lab | Actual Kit |
|---|---:|---:|
| Columns retained | 4,800 | 4,800 |
| Actual / target / union empty columns | 6 / 12 / 12 | 8 / 12 / 12 |
| Target-empty, actual-filled columns | 6 | 4 |
| Maximum positive residual | 17.537 mm | 17.513 mm |
| Maximum deficit | 0.556 mm | 0.467 mm |
| Absolute error p95 | 0.422 mm | 0.369 mm |
| RMS error | 0.570 mm | 0.543 mm |
| Overcut-volume proxy / initial base volume | 1.0300% | 0.9924% |
| >0.5 mm / >1 mm error columns | 81 / 6 | 4 / 4 |
| Represented shape and combined goal | FAIL | FAIL |

The retained one-sided outer-column changes produce large height errors even
where the surfaces are close in 3D. Their discontinuity is measured and kept;
no fitted alignment, empty-column masking, threshold relaxation or alternate
success label repairs the failed result.

| Original analytic-plane v1 metric, retained separately | Actual Lab | Actual Kit |
|---|---:|---:|
| Maximum positive residual | 0.109 mm | 0.106 mm |
| Absolute error p95 | 1.673 mm | 1.675 mm |
| RMS error | 0.615 mm | 0.552 mm |
| Conservative maximum deficit | 20 mm | 20 mm |
| Overcut-volume proxy / initial base volume | 2.041% | 2.019% |
| Original status | FAIL_TASK_METRICS | FAIL_TASK_METRICS |

The 20 mm values include missing-column conservative penalties; they are not
measured 20 mm knife gouges.

The additional descriptive 3D audit queries the exact nearest triangle at every
referenced substrate/target vertex and every triangle centroid, in both
directions. Area-weighted p95 uses all triangle centroids and their full areas.
Maxima include every vertex and centroid; they are sampled maxima, not
continuous Hausdorff bounds. The exact recorded workpiece transform is used,
without fitted alignment. All meshes are closed and consistently oriented.

| Whole-3D descriptive measurement | Actual Lab | Actual Kit |
|---|---:|---:|
| Actual → target centroid area-weighted p95 | 0.2916 mm | 0.2851 mm |
| Target → actual centroid area-weighted p95 | 0.2919 mm | 0.2852 mm |
| Actual → target sampled maximum | 0.5559 mm | 0.4749 mm |
| Target → actual sampled maximum | 0.5581 mm | 0.4730 mm |
| Closed volume difference / fixed target volume | 0.15496% | 0.15539% |

The fixed represented target volume is 94.0437 cm³ versus the analytic initial
96 cm³; actual substrate mesh volumes are 93.8980/93.8976 cm³. These renderer
volume comparisons are separate from the conserved particle mass ledger. The
3D audit has no pass/fail thresholds and cannot override either heightmap FAIL.

The [public receipt](evidence/represented_target_geometry_audit.json) includes
all original/new metrics, physical checks, target controls, source hashes and
artifact hashes. Five CPU tests pass, including one-sided missing columns,
known offsets, physical-gate blocking, exact-mesh self distance and known
1.5 mm translation. Reproduce with fresh output paths:

```bash
.venv/bin/python scripts/build_represented_flatbar_target.py artifacts/represented_flatbar_target_002 --device cuda:0
.venv/bin/python scripts/evaluate_represented_target.py artifacts/isaac_lab_trim01_coarse_001 artifacts/represented_flatbar_target_002 artifacts/represented_goal_lab_coarse_001
.venv/bin/python scripts/evaluate_represented_target.py artifacts/isaacsim_trim01_coarse_001 artifacts/represented_flatbar_target_002 artifacts/represented_goal_kit_coarse_001
.venv/bin/python scripts/describe_represented_mesh_distance.py artifacts/represented_goal_lab_coarse_001 artifacts/represented_flatbar_target_002 artifacts/represented_mesh_distance_lab_coarse_001
.venv/bin/python scripts/describe_represented_mesh_distance.py artifacts/represented_goal_kit_coarse_001 artifacts/represented_flatbar_target_002 artifacts/represented_mesh_distance_kit_coarse_001
.venv/bin/python -m pytest -q tests/test_represented_target.py tests/test_represented_mesh_distance.py
```

Both actual evaluators intentionally exit nonzero for the retained represented
goal failures. No further target tuning or runtime/default promotion follows
from this diagnostic.
