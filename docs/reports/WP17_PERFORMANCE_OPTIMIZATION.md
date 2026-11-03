# WP-17 — accelerate the active cutting frontier

**Full Episode Wall Time:** pooled 1034.211 → 765.959 s, **25.94% less time**
(1.350×). **Episodes/GPU-hour:** 6.96183 → 9.39999. Both new complete Native
episodes PASS original independent integrity/outcome and complete MP4 verification.

| Scene | WP16 full wall, s | WP17 full wall, s | Speedup | Episodes/GPU-hour, WP16 → WP17 |
| --- | ---: | ---: | ---: | ---: |
| 512, one protrusion | 621.606924 | 375.653021 | 1.655× | 5.79144 → 9.58331 |
| 15, two protrusions | 412.604198 | 390.305633 | 1.057× | 8.72507 → 9.22354 |

These are two known-scene engineering regressions, not a uniform 1.5× gain.
Implementation: `f55f94afe8c0155c0b70094e41b4c0dcb45da3d7`.
[Compact evidence](../evidence/wp17/performance_001.json) records exact values,
source/controller identities, failures, hashes and relative artifact locators;
[WP16](WP16_PERFORMANCE_OPTIMIZATION.md) retains the previous work.

## Equal timing boundary and retained execution

Both WP16 baselines already enable `v7_2_optimizations=True` and
`material_diagnostics="device_reduction_v1"`; both remain enabled in WP17.
The frozen `trim_skim_c3` controller commit is `87e2a5642d4642ad3b09384e176530cc2603e2d0`,
SHA256 `10e128c57627a6f304471c7d9c9ddeecf7b2206a10add218776ce039f8efa506`.
Scene seeds 512/15 and simulation seed42 retain TLMPM v7_2, 1 mm particles,
4 mm grid, 0.1 ms material timestep, 50 Hz control, ten PhysX and 200 material
substeps/action, 1200 actions / 24 s, physical grasp and bidirectional reaction.
Material, rig14 Global RGB, wrist D405 v5 RGB-D, normal arm1 park, API,
observation isolation, evaluator and >=90% target / >=99% substrate rules stay fixed.
Existing CUDA Graphs, surfaces on capture and streamed recording are reused.

Full wall runs from factory entry through `finalize_end`, including reset, all
cold compilation, actions, physical evaluation and complete success MP4 export.
Startup is not amortized; first-action IK/diagnostic compilation remains inside
the action-call timing. Factory exit, outer process and external audits are separate.

| Measured seconds, WP16 → WP17 | Seed512 | Seed15 |
| --- | ---: | ---: |
| Startup/reset before first action | 70.667 → 65.974 | 70.462 → 67.022 |
| Sum of action calls | 537.871 → 297.003 | 328.692 → 310.430 |
| Material graph, within action calls | 388.715 → 154.618 | 173.846 → 164.786 |
| Finalization, including evaluation/media | 12.021 → 11.681 | 12.404 → 11.871 |
| Outer Native process | 626.946 → 380.926 | 417.821 → 395.074 |
| External wrapper | 630.633 → 384.532 | 421.558 → 398.668 |

Nested timings must not be summed. Additional audit-command walls: seed512
verification 1.6203 s, comparison 0.4590 s; seed15 verification/comparison together
2.6149 s. These are outside Native timing. Action-loop samples show no competing
GPU1 compute PID and 47 MiB other GPU1 memory; host contention is not causally isolated.

## Actual active-frontier attribution

A real Native seed512 run retains full internal frontier history through four
phases. At action300, all five prelaunch samples (substeps 0/49/99/149/199) have
4202 eligible particles and 14,607,360 support-neighbor occurrences/substep.

| Actual phase/action | Frontier, ms / 200 substeps | Particle query/contact, ms / 200 substeps |
| --- | ---: | ---: |
| No contact / 20 | 0.849 | 60.727 |
| Approach / 250 | 0.849 | 59.617 |
| Active cutting / 300 | 1695.375 | 63.494 |
| Post-cut motion / 410 | 0.835 | 60.629 |

Active frontier consumes 92.27% of 1.837379 s measured kernel time. Its interval
adds **[0, 0]** new link events/completions: this measures eligible search/rejection,
not positive completion. Hash construction and blade SDF internals are unresolved.
Eager instrumentation perturbs execution; it is separate from production graph
and full timing. The deliberately stopped 410-action profile is **FAIL_INTEGRITY**,
not a success episode. Raw evidence: `artifacts/active_profile_003/`.

## Scoped change and focused verification

The v7_2 helper reuses p-specific transforms, local positions and swept sides;
it applies original positive current-interface predicates before evaluating the
old interface, then exits both loops after the original first selected neighbor.
Stable node/neighbor order, exact `sa*sb < 0` and interpolation, current twist for
both interfaces, pending lifecycle, endpoint work, gradient-only support and
code-zero bridges remain. There is no support deduplication, reordered traversal,
new buffer, contact modification or cutting-law change. The original
`tlmpm_v7_2._complete_links` is byte-identical and callable as rollback.

**58 CPU frontier checks and 10 related checks PASS**, including real dispatch,
original contact rollback and the three v6 default/frozen-source checks. The
initial brittle filename CI failure is preserved; defaults/physics were not changed.
On the actual action300/substep0 GPU checkpoint, all four integer outputs match;
23 shared checkpoint arrays, including four restoration sources, are unchanged.
Three alternating warm 20-launch graphs restore outputs before every launch:
median 0.167702585 → 0.029516273 s, **5.681699×**, including copy costs.
A separate **12-case GPU probe PASS** exercises seven positive-completion and two
new-pending cases; all four outputs match and 19 read-only arrays/path remain unchanged.
These are component checks, not full-history identity. Probe artifacts are
`artifacts/frontier_same_state_001/` and `artifacts/frontier_positive_gpu_001/`.
Native and the same-state GPU probe use Python 3.12.12, Warp 1.17.0 and NumPy 2.3.1.
Native uses Isaac Sim 6.2.0.0 and Torch 2.11/cu128. Only the positive-fixture GPU
probe uses NumPy 2.5.4.

## Physical, history, surface and media comparison

Both new 1200-action runs have 121 physical samples, finite numerical/contact
statistics, positive detF, bilateral loaded grasp and no rejection/grasp loss or
arm1 command change. Particle counts and mass arrays match within pairs:
102732 / 154.098 g and 103898 / 155.847 g. Every target has 100% removed mass
(0.498 g; 0.7425/1.5045 g), substrate share is 0.9999999999991656 with zero detached
below-plane mass, and components remain 2 / 3. Codes are monotone with no rewrites.

**Seed512 final codes differ:** 33692 → 37958 labelled particles. All 4266 extra
labels are cut0 digit0→1 in substrate at local z0.5–19.5 mm below the 20 mm plane:
6.399 g, zero target particles; cut1 is unchanged. Reference arrays and initial
x/v match. Position drift precedes code changes (45.3 nm at 0.2 s, 6.895 µm at
5 s); the first sampled code difference is 5.2 s. Final maximum position difference
is 39.071 µm. Seed15 final codes match, but intermediate codes differ by up to ten
particles and final maximum position difference is 113.176 µm. The cause is not
isolated and cannot be assigned solely to atomics. State-dependent eligibility
and histories can alter cost; fixed-state kernel gain and observed whole-run
speedups must remain separate evidence.

| Descriptive surface metric, WP16 → WP17 | Seed512 | Seed15 |
| --- | ---: | ---: |
| Plane-error p95, mm | 0.483370 → 0.483402 | 0.486061 → 0.486086 |
| Plane-error RMS, mm | 0.173539 → 0.173558 | 0.188272 → 0.188221 |
| Missing surface fraction | 0 → 0 | 0.000390625 → 0.000390625 |
| Maximum gouge proxy, mm | 1.402096 → 1.403033 | 20 → 20 |
| Overcut/base volume fraction | 0.007054816 → 0.007055320 | 0.007737343 → 0.007736837 |

Actual Global captures at 6 s and 24 s were reviewed for both scenes, with no
obvious visual regression. Quality stays **REPORTED_NOT_SCORED**, including seed15
missing cells/20 mm gouge proxy; **HOME NOT_RUN**, materials provisional/uncalibrated.
Both complete original
MP4s PASS full decode, hashes, timestamps, raw/camera/source identity. The four
prior WP16 MP4 hashes are unchanged against preserved full-decode receipts.
No successful physics or media was rerun. Raw pair: `artifacts/native_optimized_001/`.

## Minimal reproduction, rollback and limits

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_wp17_frontier.py
```

Use the existing frozen-controller Native launcher with `PYTHONPATH=src` for the
intended checkout. Pass `v7_2_optimizations=True, material_diagnostics="device_reduction_v1"`
to `open_cutting_env(..., runtime="isaac_sim", ...)` or `agentic_trim_env.open_env(...)`.
Before reset/graph capture, `v7_2_optimizations=False` restores contact/frontier;
retain device diagnostics. Low-level `optimize_frontier=False` keeps the WP16
contact path independently. Receipts identify actual flags/kernel/helper hash.
An agentic control robot, VLA or policy keeps the same actions and vision/proprioception
fields, excluding force/oracle state. Two zero-action startup failures and the
incomplete profile remain preserved. Particle-contact/hash/SDF/process-reuse work
was omitted after this scoped measured gain. No held-out, convergence or training
claim follows from these two scenes. Publication and CI status are tracked separately
in the repository pull request and [CPU workflow](https://github.com/AnsonNing/RoboticCuttingSim/actions/workflows/tests.yml).
