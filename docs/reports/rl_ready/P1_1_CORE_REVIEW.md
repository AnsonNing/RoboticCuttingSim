# P1-1 batched material core — xhigh review (2026-11-10)

Scope: commits 220bcde, 8560e42, 1c3306f, 3f8d0d5, c2bfe1c (`src/robotic_cutting_sim/batched/`, batched
tests, `scripts/probe_batched_material_gpu.py`) against ADR 0021 and the frozen tlmpm_v7_2 chain.
Read-only reviewer at xhigh; CPU tests and CPU probes only (GPUs shared).

**Verdict: PASS_WITH_LIMITATIONS** — no CRITICAL physics or isolation defect; one MAJOR bug.

## Verified

- 113 batched CPU tests PASS; frozen modules unchanged, SHA256 pins match.
- Every ENV kernel diffed structurally against its frozen counterpart: differences are env indexing,
  per-env parameters (same float32 bits as the single path) and per-env reduction rows, plus dead branches
  removed for configurations that raise UNSUPPORTED. Launch order identical. One launch per operation over
  all envs (obstacle boxes one launch per box, as in the frozen code).
- Guarded slot-occupancy `atomic_max` (c2bfe1c) equivalent (monotone integer max, atomics-only writes).
- Grouped HashGrid: 64-bit (cell, group) keys, reserved before capture, persistent groups array, queries
  pass the env id; NaN positions stay isolated by the group filter.
- Extra CPU probes: heterogeneous 3-env scenes with non-zero particle contacts, conflicting codes and handle +
  2 pads touching material — bitwise equal for 6 substeps on every array; wrench conversion for rotated
  workpieces ≤4.7e-16 relative to the live path; device pose path 0 ulp over 668,928 components for
  float32-rounded FK quaternions; poison-reset probe: reset env equals a fresh pack (seed kept when none
  given), others byte-identical.

## Findings (all assigned to a fixer, tracked in ADR 0021 "Review fixes")

| # | Severity | Finding |
| --- | --- | --- |
| 1 | MAJOR | `reset()` sorts `env_ids` but pairs `seeds`/`scene_keys` by position → wrong seeds; geometry-change guard fails open for unsorted ids |
| 2 | MINOR | shipped parity tests leave particle contacts, box reactions, pad recipients, link completions, rotations and per-env digit differences at zero |
| 3 | MINOR | reset coverage test checks naming, not restoration (no poison test) |
| 4 | MINOR | out-of-range device cut digits are flagged only after 200 substeps ran |
| 5 | MINOR | `pose_interp` docstring misdescribes the frozen normalisation; device path stricter |
| 6 | MINOR | ADR §7 Torch/Warp stream rule not implemented (assigned to the vector env) |
| 7 | MINOR | no per-step pad-box setter |
| 8 | MINOR | receipts do not hash the coupling modules; `pose_interpolation` field fixed |
| — | NIT | category of `reset_mask`/`reset_seed`; ADR wording on capture allocation and co-tenant; non-atomic `rebuild()`; `solver.cuts` unvalidated in preset mode; host copy kept for the batch lifetime |

| Requirement | Result |
| --- | --- |
| True batching | PASS |
| Per-env isolation | PASS (CPU bitwise, NaN poison, GPU grouped-query audit) |
| Partial reset | PASS except finding 1 |
| No silent physics change | PASS on CPU; GPU L2 smoke only, L3 pending (milestones) |
| Frozen modules untouched | PASS |
| Explicit UNSUPPORTED | PASS except finding 1 |
| Delayed wrench (20 ms) | PASS |
