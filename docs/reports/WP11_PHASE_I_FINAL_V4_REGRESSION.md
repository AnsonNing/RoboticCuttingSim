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

## Scope: commits after the pin

The result covers `d57d0f2` only. While this run was in progress, eight commits landed on the branch
(`ce8c810`..`2e790ff`, review R1-R3). Four of them touch final_v4-path files: `cutting_surface.py` (d009053:
versioned unmeshable-surface policy, default `raise` as on main), `isaac_sim_runner.py`, `live_env.py` and
`policies.py` (3a75b83: new opt-in profile `path_v3a`; `path_v2` and `path_v2_vision` are described as
unchanged). This run recorded no unmeshable-surface skips in any demo, and d009053 states that all-meshable
outputs are byte-identical. That makes an effect on final_v4 unlikely, but it is **NOT_RUN** at `2e790ff`;
a rerun at the new tip would be needed to cover those commits.

## Status

- final_v4 regression, all four demos: **PASS**.
- Lab parity and the other P1/P2 items from `BLADEMASTER_DEMOS.md`: **NOT_RUN** (out of scope).
- Integrity: no source edits, no threshold, physics or record changes, no rescoring. The reference run is
  untouched and nothing was overwritten. Material parameters remain provisional and uncalibrated.

## Run 002 at the merge HEAD 6c36e39

**Verdict: PASS.** The final_v4 set reproduces at the merge HEAD `6c36e39`, which covers the review-fix commits
that the scope note above left NOT_RUN. Each of the four demos was compared against the reference
`live_20261029_law8_final` and against run 001. All of them have the same runner verdict, the same execution
and outcome layers, the same frozen strict status and failed-check set, and the same task, protocol and plan
identity. Metrics agree within GPU nondeterminism. The documented frozen-strict FAILs repeat unchanged: Demo 1b
fails on location and window, and Demo 2 scores precision 5/8.

Evidence: [phaseI_final_v4_regression_002.json](../evidence/wp11/phaseI_final_v4_regression_002.json).

| Item | Value |
| --- | --- |
| Code | clean detached worktree `/mnt/data/ningan/RoboticCuttingSim-wp11-I2-6c36e39` at `6c36e39`, with the same symlinks as run 001 |
| `git_dirty` | False in every run receipt, demo summary (run and evaluator), demo and episode `manifest.json`, and `index.json`; every one records sha `6c36e391` |
| Command | `cd /mnt/data/ningan/RoboticCuttingSim-wp11-I2-6c36e39 && TMPDIR=/mnt/data/ningan/tmp /usr/bin/time -v .venv/bin/python scripts/run_blademaster_demos.py --demo-set final_v4 --gpu 1 --run-id live_20261030_wp11_regression_002` |
| Per-demo argv | identical to the reference except `--out` (same `--gpu 1`) |
| Wall time | 12:38:28 to 13:14:39 JST; runner 36:11, exit 0, max RSS 15.6 GB. Physics per demo: 1a 364 s, 1b 627 s, 2 405 s, 3 567 s (reference: 350, 440, 352, 528 s) |
| Attempts | 1 per demo, no relaunch |
| Output | `artifacts/wp10_claude/demos/live_20261030_wp11_regression_002/` (2.2 GiB; stages run, evaluate and render; no publish) |
| Disk | `/mnt/data`: 23.8 GiB free at launch and 20.4 GiB at the lowest sample; the 15 GB floor was never reached |

**GPU.** GPU 1 was idle at launch: no compute process and 71 MiB used. During Demo 1a a non-RCS job
started on GPU 1 (`aw0 infer`, pid 4004311). It used 20-24 GB at 86-99% utilization. The runner keeps one
`--gpu` for all demos. It started Demo 1b with 12.1 GiB free on GPU 1 and Demo 2 with only 8.4 GiB free.
Demo 2 therefore started below the 12 GB launch floor, with one other active job. I did not interrupt the
runner and did not touch that job. Demo 3 started with GPU 1 idle again. The contention explains the longer
walls of 1b and 2. There was no OOM and no timeout, and no layer changed.

**Unmeshable-surface policy (d009053).** In every demo, the `config` in `runtime/setup.json` has
`surface_unmeshable: "raise"` (`configured_task` None), which is the expected resolution for these legacy
tasks. No surface skip happened:

- there is no top-level `surface_unmeshable` key in setup.json;
- there is no `surface_unmeshable_*.jsonl` file;
- there is no `surface_unmeshable` key in result.json;
- no `actions.jsonl` row mentions it;
- the logs contain no `unmeshable` or `SurfaceNotScorable` text.

The reference and run 001 do not have the key, because they predate d009053. Their only behaviour was `raise`.

### Comparison (reference → run 001 → run 002)

Identity is the same in all three runs, for all four demos: task id; `task.json` sha256; task, strict and
research `protocol_sha256`; outcome protocol; `policy_plan`; controller-profile receipt; `policy_motion`;
demo set; law 8; steps.

| Demo | Runner verdict | Frozen strict | Key metrics (ref → 001 → 002) | Max grasp slip | Same? |
| --- | --- | --- | --- | --- | --- |
| 1a CUT-01-R | PASS ×3 | PASS_TASK_METRICS ×3 | parts 50/50% ×3; location 1.5 mm ×3; persistence after withdrawal 3.0 s ×3; edge x error 0.046 mm ×3 (in contact 0.022 → 0.025 → 0.024 mm) | 0.722 → 0.727 → 0.829 µm | yes |
| 1b CUT-01-R 20° | PASS ×3 (swept-plane) | FAIL_TASK_METRICS ×3; fails `cut_at_workpiece_local_target_plane` (5.5 mm ×3) and `postcut_disconnected_topology_persists` | parts 50.46/49.54% ×3 (002: 50.465/49.535%); swept interface 4.784 → 4.783 → 4.783 mm (allowed 5.444 mm); label-geometry disagreement 1.067 mm ×3; persistence 1.82 s ×3 | 0.786 µm ×3 | yes |
| 2 TRIM-01-R | PASS ×3 (separation) | FAIL_TASK_METRICS ×3; precision 5/8 ×3, same 3 failed checks (gouge, overcut, surface p95) | removal 97.99 → 97.89 → 97.99%; substrate 99.992% ×3; detached 0.008% ×3; p95 0.583 → 0.581 → 0.579 mm; overcut 1.658 → 1.660 → 1.658%; gouge 20 mm ×3; edge z error 0.026 mm ×3 (in contact 0.0094 → 0.0093 → 0.0090 mm) | 0.716 → 0.695 → 0.692 µm | yes |
| 3 SLICE-PUSH-01 | PASS ×3 (PASS_SLICE_PUSH) | NOT_APPLICABLE ×3 | 3 parts 37.5/31.25/31.25% ×3; declared-plane error 2.6e-10 m (both cuts) ×3; push 19.680 → 19.687 → 19.685 mm; others ≤ 0.9 / 0.06 / 0.62 µm; 0 label changes during push ×3; wrist-depth shift 1.2009 → 1.2005 → 1.1955 mm | 0.806 → 0.906 → 0.841 µm | yes |

Execution is identical in all three runs: 400/400/400/800 steps, 0 IK rejections, HOLDING throughout, grasp
never lost, and cut-enabled steps 158/168/167/317. As in run 001, tracking-error p95/max in 1a, 1b and 2 is
bit-identical to the reference. The Demo 2 removal fraction is bit-identical to the reference this time.
Action-clamp counts vary (ref/001/002: 1a 10/5/6, 1b 17/17/13, 2 2/3/3, 3 4/9/5). They affect no layer.

The comparison is read-only:
`/mnt/data/ningan/RoboticCuttingSim_artifacts/wp10_claude/wp11_phaseI/compare_phaseI_002.py`. It is the run-001
script plus a `surface_unmeshable_receipt` that is not part of identity; its sha256 is in the evidence. Its
outputs are `compare_phaseI_002_ref.json` and `compare_phaseI_002_vs001.json` next to the script.

### Code-path changes since run 001 (`d57d0f2..6c36e39`)

These final_v4-path files changed:

- `cutting_surface.py` and `isaac_sim_runner.py` (d009053: unmeshable-surface policy, default `raise`);
- `isaac_sim_runner.py`, `live_env.py` and `policies.py` (3a75b83 and 2e790ff: opt-in `path_v3a`, privileged
  info split, `evaluate()` adds `surface_unmeshable` only when the backend reports a summary);
- `live_material.py` and `blade_probe.py` (46be189: reject out-of-range cut digits).

None of them changed a verdict, the identity or a failed-check set. The following are unchanged in this
range: `tlmpm_v4.py`, `tlmpm_v3.py`, `run_wp10_live.py`, `run_blademaster_demos.py`, `render_demo_episode.py`,
`demo_evaluation.py`, `evaluation.py`, `tasks.py`, `multi_cut_evaluation.py`, `task_scenarios.py`,
`tool_profile.py`, `cut_topology.py` and `native_fixture.py`.

### Scope: commits after the 6c36e39 pin

While run 002 was being written up, three commits landed on the branch (`cbd80d1`, `b90658d`, `495d639`). Only
`b90658d` touches a final_v4-path file, `isaac_sim_runner.py`, and it makes two changes:

1. `surface_unmeshable='skip_presentation_v1'` now raises without a `configured_task`.
2. `capture()` with `surface_update='on_capture'` refreshes the surface without the scorability check.

Neither path applies to final_v4. Run 002 recorded `raise` and `surface_update: every_step`, and the commit
states that the every_step path is byte-for-byte unchanged. Run 002 covers `6c36e39` only; final_v4 at
`495d639` is **NOT_RUN**.

### Status (run 002)

- final_v4 regression at `6c36e39`, all four demos: **PASS**. At `495d639` (b90658d runner change): **NOT_RUN**.
- Unmeshable-surface policy on final_v4: **PASS**. It resolves to `raise` and no surface skip happened.
- Lab parity and the other P1/P2 items: **NOT_RUN** (out of scope).
- Integrity: no source edits, no threshold, physics or record changes, no rescoring. The reference run and
  run 001 are untouched and nothing was overwritten. Material parameters remain provisional and uncalibrated.
