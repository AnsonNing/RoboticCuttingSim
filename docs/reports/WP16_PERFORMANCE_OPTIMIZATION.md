# WP-16 — faster complete Native cutting episodes

Implementation: `0e92f1d917b8117faf2426b168ff25e151ae6d79`; baseline:
`c8a573b6337ee6bc0ea0257badb3413f08b59a23`.
Two complete paired Native episodes reduce total wall time by **27.21%**
(1.374× pooled). Physical integrity, original task outcome and complete MP4
verification PASS for all four runs. [Compact evidence](../evidence/wp16/performance_001.json).

## Changes and measured bottleneck

The complete baseline seed512 executes 1200 actions / 24 seconds. Its material
CUDA Graph takes 577.065 s, 78.51% of control time. The 141 cut-enabled actions
average 1967.48 ms in the graph; the other 1059 average 282.96 ms. This is an
actual Native phase split, rather than a kernel-level attribution.

The existing eager kernel profiler, using recorded positions, velocities and
codes with incomplete internal history, measures about 154 ms blade contact
and 64 ms particle contact per 200 material substeps. Those represent about
54.4% and 22.6% of its kernel time. It does not reproduce seed512's expensive
active frontier history, and its eager wall time is not production graph time.

Adopted changes:

- A v7_2-only contact kernel skips the six shared reaction atomics and the
  particle impulse addition when the blade impulse is exactly zero. Component
  comparisons retain tiny nonzero impulses whose squared norm underflows.
  SDF evaluation, work, engagement, friction and all real contacts still run.
- An explicit `material_diagnostics="device_reduction_v1"` reads every original
  safety field on the device, returning finite-state status and three extrema
  in 16 bytes. It preserves finite checks, the reaction handoff, slot checks
  and synchronization. The default `"snapshot"` path remains available.

The small scoped helper retains the original shared step order; it avoids
changing the source-pinned v7 CUT profile. v5, v6, v7, v7_1, the v7_2 link
kernel, material parameters, tool, evaluators and frozen profile pins retain
their original content. The existing CUDA Graph, on-capture surface updates
and streamed recording are reused.

## Complete episodes

Both pairs use the frozen `trim_skim_c3` controller SHA256
`10e128c57627a6f304471c7d9c9ddeecf7b2206a10add218776ce039f8efa506`,
controller commit `87e2a5642d4642ad3b09384e176530cc2603e2d0`, scene seeds
512 and 15, and simulation seed42. They retain 1 mm particles, 4 mm grid,
0.1 ms material timestep, 200 material substeps per 20 ms control interval,
physical grasp, bidirectional reaction, rig14 Global RGB and wrist_0 RGB-D.
All 1200 actions and original physical/quality evaluation rules remain.

The full Native boundary is factory entry through final recording, independent
evaluation and complete success-video export. Process startup/teardown and the
additional external verification are reported separately in the receipt.

| Scene | Baseline full Native wall | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| 512, one protrusion | 813.352 s | 621.607 s | 1.308× |
| 15, two protrusions | 607.435 s | 412.604 s | 1.472× |

The pooled total is 1420.787 → 1034.211 s, saving 386.576 s. The aspirational
1.5–2× full-episode target was not uniformly reached.

| Measured phase, seconds | 512 baseline → optimized | 15 baseline → optimized |
| --- | ---: | ---: |
| Startup/reset/first compilation | 64.146 → 70.667 | 66.195 → 70.462 |
| Sum of 1200 action calls | 736.048 → 537.871 | 526.964 → 328.692 |
| Material graph, within action calls | 577.065 → 388.715 | 361.788 → 173.846 |
| Finalization, including evaluation/media | 12.109 → 12.021 | 13.210 → 12.404 |
| Outer Native process | 818.884 → 626.946 | 612.709 → 417.821 |

Action calls average 613.373 → 448.226 ms for seed512 and 439.137 → 273.910 ms
for seed15. Whole-episode throughput is 0.02951 → 0.03861 simulated s/wall s
for seed512 and 0.03951 → 0.05817 for seed15. Startup includes any first-use compilation;
it is not amortized out of the primary result. Nested phases must not be summed.

Seed15 baseline includes one read-only internal-state snapshot costing
14.822 ms. The raw total is retained. GPU1 has no competing compute process;
an unrelated job runs on GPU0. Host/GPU contention is logged for each episode.
These are two engineering regressions on known scenes, not fresh held-out
success-rate or policy-training measurements.

Observed action-loop memory ranges, in MiB:

| Scene | Host RSS baseline → optimized | GPU process baseline → optimized |
| --- | ---: | ---: |
| 512 | 6054–6561 → 6090–6562 | 5030–5350 → 5052–5350 |
| 15 | 6083–6638 → 6030–6506 | 5308–5926 → 5052–5638 |

These are sampled ranges, not proven peak bounds or a causal memory reduction.
The diagnostic scratch allocation for 102732 particles is 25,696 bytes plus a
16-byte summary. Streamed recording remains enabled.

## Physical and surface comparison

All four runs have 1200 actions, 121 physical samples, real bilateral loaded
grasp throughout, zero rejected actions/grasp losses and no arm1 command change.
Every selected protrusion has 100% removed mass; below-plane substrate mass
share is 0.9999999999991656 and detached below-plane fraction is 0. Components
are 2 for seed512 and 3 for seed15. All 102732/103898 particles and their mass arrays remain.
Packed codes are monotone with no digit rewrite; final code arrays match within
each pair. Intermediate code histories and floating-point states are not
bitwise identical. All original numerical checks pass; minimum detF remains
positive (512: 0.910024 → 0.910193; 15: 0.843745 → 0.843825).

| Descriptive surface metric | 512 baseline → optimized | 15 baseline → optimized |
| --- | ---: | ---: |
| Plane error p95, mm | 0.483355 → 0.483370 | 0.485947 → 0.486061 |
| Plane error RMS, mm | 0.173535 → 0.173539 | 0.188245 → 0.188272 |
| Missing surface, fraction | 0 → 0 | 0.000390625 → 0.000390625 |
| Maximum gouge proxy, mm | 1.400125 → 1.402096 | 20 → 20 |
| Overcut/base volume, fraction | 0.007054700 → 0.007054816 | 0.007737204 → 0.007737343 |

Paired actual 6 s cutting and 24 s final images were reviewed: no obvious visual
regression. The small metric changes coexist with nondeterministic GPU atomics;
they do not establish calibrated physical equivalence. Seed15 retains its
missing/alignment cells and 20 mm gouge proxy limitation. Quality is
REPORTED_NOT_SCORED; HOME is NOT_RUN. All four complete original MP4s pass
full decode, hashes, frame times and raw recording/source identity checks.
Raw artifacts remain local; the compact evidence records relative artifact
locators, source/controller/scene identities and all four movie hashes.

## Focused verification

- CPU contact tests: 8 PASS, including real contact/friction/work, rotation,
  positive-gap predicted impact, tiny impulse, support contact and rollback.
  Original/current direct kernel comparison: 56 arrays exactly equal.
- CPU diagnostics, finite rejection and configuration checks PASS; existing
  observation/action/API/streamed-recording compatibility: 15 PASS.
- Same-state GPU blade test: 200 launches decrease from about 153.53 ms to
  1.012–1.255 ms. Particle impulse/progress/work match exactly; reaction
  accumulation differs by about 2e-10 from permitted atomic ordering.
- Same-state GPU diagnostics: 11.100 ms to 0.201 ms per warm call, with matching
  scalars, unchanged state hashes and original nonfinite rejection. Its first
  reduction compilation costs 113.9 ms separately. This is a component result.
- Native pairs: original independent integrity/outcome PASS 4/4; media PASS 4/4.
  Surface metrics are reported above without changing any acceptance threshold.

Native uses Python 3.12.12, Warp 1.17.0, NumPy 2.3.1, Newton 1.6.1,
Isaac Sim 6.2.0.0 and Torch 2.11/cu128 on an RTX 5090. The isolated probes use
NumPy 2.5.4; their diagnostic timing is not substituted for Native component time.
The determinant retains a float64 intermediate and float32 output; its closed
form can differ from NumPy LU in the last output bit without changing physics.

Focused CPU commands, from this checkout:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_wp16_contact_atomics.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_material_diagnostics.py tests/test_isaac_sim_runner.py
```

## Runtime selection and rollback

Keep the existing GPU-isolated Native launcher and controller. Set
`PYTHONPATH=src` from the intended checkout, since a shared editable virtual
environment can otherwise import another checkout. Pass these ordinary backend
options through `open_cutting_env(..., runtime="isaac_sim", ...)`:

```python
options = {
    "v7_2_optimizations": True,
    "material_diagnostics": "device_reduction_v1",
}
# Pass **options to the existing environment factory.
```

The existing `agentic_trim_env.open_env(...)` accepts the same two options,
alongside `stream_physics=True`. Its reset, explicit step, observation,
calibration, recording and close contracts are unchanged. An existing agentic
control robot, VLA or policy supplies the same actions and receives the same
vision/proprioception fields; no force or material oracle enters observations.
Frozen C3 execution is the direct consumer compatibility check. No new VLA
training throughput or long-run training reliability is claimed.

Set `v7_2_optimizations=False, material_diagnostics="snapshot"` to restore the
original contact execution and diagnostic path before reset/graph capture.
The low-level solver also accepts `optimize_contacts=False`. For exact original
source bytes, use the baseline commit above. Runtime setup and solver receipts
record the selected path and implementation hashes.

## Rejected work, failures and limits

A derived tool-projection cache was implemented and CPU-checked, then removed:
the actual seed15 snapshot had zero eligible link searches, so its extra
buffers had no demonstrated benefit. The prototype and failed compilation
attempt remain retained locally. Conservative SDF broadphase and particle-pair
changes were not needed for this bounded change.

A missing asset cache caused a zero-step FAIL_EXECUTION; identical hashed
assets fixed a new attempt. An external evaluator sandbox initially hid its
frozen source location (FAIL); read-only verification of the same completed
recording repaired it (PASS). Successful physics and media were not rerun.
The paused earlier evaluation, historical outcomes and media remain preserved.

The optimized partial-state diagnostic has 129–131 ms GPU kernel time per
control interval: particle contacts take about 49%, blade contact about 1.2%.
It still omits the expensive active frontier history seen in Native seed512.
Remaining costs include that frontier/link work, particle contacts, robot
coupling, surfaces and rendering. Surface quality stays descriptive: missing
surface and gouge proxies are reported without declaring quality acceptance.
Material parameters remain provisional and uncalibrated; HOME is NOT_RUN.
