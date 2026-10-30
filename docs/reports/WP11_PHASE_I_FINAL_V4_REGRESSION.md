# WP-11 Phase I: final_v4 BladeMaster regression (2026-10-30)

**Verdict: PASS.** The frozen BladeMaster set `--demo-set final_v4` (cutting law 8, seed 42) reproduces at
WP-11 HEAD `d57d0f2`. All four demos have the same runner verdict, the same execution and outcome layers,
and the same frozen strict status and failed-check set as the reference run `live_20261029_law8_final`.
Task, protocol and plan identity are also the same. Metrics agree within GPU nondeterminism. The documented
frozen-strict FAILs happen again unchanged: Demo 1b (location and window) and Demo 2 (precision 5/8).

Evidence: [phaseI_final_v4_regression_001.json](../evidence/wp11/phaseI_final_v4_regression_001.json).

## Run

| Item | Value |
| --- | --- |
| Code | clean detached worktree `/mnt/data/ningan/RoboticCuttingSim-wp11-I-d57d0f2` at `d57d0f2` (HEAD moved from `a24cbd4` to `d57d0f2` before the worktree was created; that commit only touches `docs/PROGRESS.md`). Symlinks: `.venv`, `.deps/*` and `artifacts/{assets_audit_002,workcell_asset_cache}` point to the main checkout; `artifacts/wp10_claude` points to `/mnt/data/ningan/RoboticCuttingSim_artifacts/wp10_claude` |
| `git_dirty` | False in every run receipt, demo summary (evaluator), demo and episode `manifest.json`, and `index.json` |
| Command | `cd /mnt/data/ningan/RoboticCuttingSim-wp11-I-d57d0f2 && TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_blademaster_demos.py --demo-set final_v4 --gpu 0 --run-id live_20261030_wp11_regression_001` |
| Per-demo argv | identical to the reference except `--gpu 0` (reference: `--gpu 1`) and `--out` |
| Device | GPU 0, RTX 5090. At launch it had 2 other processes (flospo, 1.5 GB, 0% util) and 30.8 GB free. GPU 1 had an 18.6 GB job at 88-99% |
| Wall time | runner 32:11, exit 0. Physics per demo: 1a 363 s, 1b 446 s, 2 359 s, 3 553 s (reference: 350, 440, 352, 528 s) |
| Attempts | 1 per demo, no relaunch |
| Output | `artifacts/wp10_claude/demos/live_20261030_wp11_regression_001/` (2.2 GiB; stages run, evaluate, render; no publish) |
| Disk | `/mnt/data`: 24.4 GiB free at launch, 18.5 GiB lowest sample; the 15 GB floor was never reached |

During Demos 2-3, another RCS agent started two Phase R kinematic jobs (`--device cuda:1`). They hold
idle contexts of about 0.5 GB on GPU 0, where utilization stayed at 0%. Demo 3 launched inside the same
runner with 29.8 GB free on GPU 0. No other process was touched.

## Comparison (reference `live_20261029_law8_final` → regression `live_20261030_wp11_regression_001`)

Identity is identical for all four demos: task id; `task.json` sha256; task, strict and research
`protocol_sha256`; outcome protocol; `policy_plan`; controller-profile receipt; `policy_motion`; demo set;
law 8; steps. final_v4 artifacts do not record a plan hash field, so plan identity is the set of those
fields.

| Demo | Runner verdict | Frozen strict | Key metrics (ref → new) | Max grasp slip | Same? |
| --- | --- | --- | --- | --- | --- |
| 1a CUT-01-R | PASS → PASS | PASS_TASK_METRICS → same | parts 50/50% → 50/50%; location 1.5 mm → 1.5 mm; persistence after withdrawal 3.0 → 3.0 s; edge x error 0.046 → 0.046 mm (in contact 0.022 → 0.025 mm) | 0.722 → 0.727 µm | yes |
| 1b CUT-01-R 20° | PASS → PASS (swept-plane) | FAIL_TASK_METRICS → same; fails `cut_at_workpiece_local_target_plane` (5.5 mm → 5.5 mm) and `postcut_disconnected_topology_persists` | parts 50.46/49.54% → same; swept interface 4.784 → 4.783 mm (allowed 5.444 mm); label-geometry disagreement 1.067 → 1.067 mm; persistence 1.82 → 1.82 s | 0.786 → 0.786 µm | yes |
| 2 TRIM-01-R | PASS → PASS (separation) | FAIL_TASK_METRICS → same; precision 5/8 → 5/8, same 3 failed checks | removal 97.99% → 97.89%; substrate 99.992% → 99.992%; detached 0.008% → 0.008%; p95 0.583 → 0.581 mm; overcut 1.658% → 1.660%; gouge 20 → 20 mm; edge z error 0.026 → 0.026 mm (in contact 0.0094 → 0.0093 mm) | 0.716 → 0.695 µm | yes |
| 3 SLICE-PUSH-01 | PASS → PASS (PASS_SLICE_PUSH) | NOT_APPLICABLE → same | parts 37.5/31.25/31.25% → same; declared-plane error 2.6e-10 m (both cuts) → same; push 19.680 → 19.687 mm; others ≤ 0.9 µm → ≤ 0.06 µm; 0 label changes during push; wrist-depth shift 1.2009 → 1.2005 mm | 0.806 → 0.906 µm | yes |

Execution is identical across the board: 400/400/400/800 steps, 0 IK rejections, HOLDING throughout, and
the same cut-enabled step counts (158/168/167/317). Tracking-error p95/max is bit-identical in 1a, 1b
and 2, and the 1b slip is bit-identical too. Action-clamp counts differ: 1a 10 → 5, 2 2 → 3, 3 4 → 9.
They affect no layer.

The comparison is read-only:
`/mnt/data/ningan/RoboticCuttingSim_artifacts/wp10_claude/wp11_phaseI/compare_phaseI_001.py`
(its sha256 is in the evidence), output `compare_phaseI_001.json` next to it. The edge-error statistic is
the max |edge − target| over cut-enabled steps; for Demo 2 the target is the plane + `edge_clearance_m`.
On the reference run it reproduces the README values of 0.046 mm and 0.026 mm.

## Code-path changes since `3e26d6b`

The final_v4 path changed in `run_wp10_live.py`, `policies.py`, `isaac_sim_runner.py`, `live_env.py`,
`live_material.py`, `multi_cut_evaluation.py`, `tasks.py`, `task_scenarios.py`, `tool_profile.py`,
`cutting_surface.py`, `cut_topology.py`, `blade_probe.py` and `native_fixture.py` (the evidence has the
diffstat). None of these changes altered a verdict or the identity. The following are unchanged:
`tlmpm_v4.py`, `tlmpm_v3.py`, `run_blademaster_demos.py`, `render_demo_episode.py`,
`demo_evaluation.py`, `evaluation.py` and `cutting_tasks_v2/v3.json`.

## Status

- final_v4 regression, all four demos: **PASS**.
- Lab parity and the other P1/P2 items from `BLADEMASTER_DEMOS.md`: **NOT_RUN** (out of scope).
- Integrity: no source edits, no threshold, physics or record changes, no rescoring. The reference run is
  untouched and nothing was overwritten. Material parameters remain provisional and uncalibrated.
