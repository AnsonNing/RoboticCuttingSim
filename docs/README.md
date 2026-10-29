# Documentation

## Start here

| Need | Read |
| --- | --- |
| Install, select a GPU and reproduce | [Portable deployment](PORTABLE_DEPLOYMENT.md) · [Setup](SETUP.md) |
| Run physical-grasp CUT/TRIM | [Physical tool use](PHYSICAL_TOOL.md) |
| Control or connect a research adapter | [Shared environment API](SHARED_TASK_API.md) |
| Measure cutting outcome and geometry independently | [Evaluation v2](EVALUATION_V2.md) |
| Check both hand-aligned wrist cameras | [Camera rig v5](WRIST_CAMERA_V5.md) |
| Use the same-instance remote GUI | [Remote interactive viewer](REMOTE_INTERACTIVE_VIEWER.md) |
| Run or reproduce the three BladeMaster-style demos | [Demo report](reports/BLADEMASTER_DEMOS.md) · [Plan](BLADEMASTER_DEMO_PLAN.md) |
| Check current status and unresolved work | [Progress](PROGRESS.md) |

## BladeMaster demos (2026-10-29)

- [Demo report](reports/BLADEMASTER_DEMOS.md): live Native Demo 1 (complete
  and 20° oblique cut), Demo 2 (35° skim) and Demo 3 (two cuts, three parts,
  blade-face push), with every attempt in
  [demo evidence](evidence/phase3/demos_live_004.json) and media under
  [media/demos](media/demos/).
- Cutting laws: final [ADR 0012 law 8](adr/0012-cutting-law-8-plow-gated-lead.md)
  (gate matrices [law 8 001](evidence/phase1/matrix_law8_001.json) and
  [clean-tree 002](evidence/phase1/matrix_law8_002.json): M1–M9 PASS; smokes and quick checks
  [law 6](evidence/phase1/law6_smokes_001.json), [law 7](evidence/phase1/law7_quickcheck_001.json),
  [law 8](evidence/phase1/law8_quickcheck_001.json));
  history [ADR 0006 law 4](adr/0006-swept-band-cutting-law-4.md) (with Amendment 1),
  [ADR 0008 law 5](adr/0008-cutting-law-5-lead-preassignment.md),
  [ADR 0010 law 6](adr/0010-cutting-law-6-deferred-lead-commit.md),
  [ADR 0011 law 7](adr/0011-cutting-law-7-eq8-coupled-flip.md);
  1 mm matrix [law 5 computed](evidence/phase1/matrix_law5_002.json),
  [review](evidence/phase1/matrix_law5_adjudication_001.json),
  [law 4](evidence/phase1/matrix_law4_001.json); handoff diagnosis
  [WP-10 Claude](reports/WP-10_CLAUDE_DIAGNOSIS_20261029.md).
- Workcell and tasks: [ADR 0007](adr/0007-blademaster-workcell-and-demo-tasks.md)
  (`blademaster_workcell_v1`, `SLICE-PUSH-01`, evaluation, runner) and
  [ADR 0009](adr/0009-demo3-wrist-depth-retarget.md) (Demo 3 wrist-depth retarget);
  [Native smokes](evidence/phase2/native_smokes_001.json).

## Current scene and validation

- [Evaluation v2](EVALUATION_V2.md): separate integrity, cutting outcome and
  geometry; Diagnostic/Research/Precision profiles and preserved strict scores.
- [Hand-aligned wrist-camera v5](WRIST_CAMERA_V5.md): both cameras use fixed
  hand-local rotations, with Franka 1 parked and no wrist-target aiming.
  The [v4 checkpoint](WRIST_CAMERA_V4.md) preserves the earlier primary slant.
- [Interrupted-work resumption](RESUMPTION_20261029.md): recovered WP-08
  bootstrap/randomized/capsule outcomes and explicit remaining failures.
- [Independent physical-tool validation](PHYSICAL_TOOL_INDEPENDENT_VALIDATION.md):
  task, contact, reaction, camera and numerical results, with acceptance limits.
- [Grip perturbation controls](PHYSICAL_TOOL_CONTROLS.md) and
  [native contact sensor](PHYSICAL_CONTACTS.md): actual loads, slip and release.
- [Camera rig](WRIST_CAMERAS.md) and [close Global view](GLOBAL_CLOSE_VIEW.md):
  measured wrist mounts, sensor models and known occlusion limitations.
- [Task definitions](CUTTING_TASKS.md) and [benchmark](CUTTING_BENCHMARK.md):
  workpiece-local goals and independent success criteria.
- [Current v5 CUT RGB-D](media/wrist_v5_cut01/synchronized_multiview.mp4),
  [v5 TRIM RGB-D](media/wrist_v5_trim01/synchronized_multiview.mp4) and
  [preserved v3 physical-grasp recordings](media/wp08_physical_grasp_v2/README.md).

## Runtime and reproduction references

[Compatibility](COMPATIBILITY.md) · [Native Sim integration](ISAAC_SIM_VALIDATION.md) ·
[Lab integration](ISAAC_LAB_INTEGRATION.md) ·
[WP-07 clean-environment results](PORTABLE_DEPLOYMENT_VALIDATION.md)

Current reproducibility checks distinguish new isolated environments on the
original host from independent cross-host tests. A smoke is never full cutting
success; executing an episode is never complete physics acceptance.

## Preserved research history

Earlier reports and media remain in their original paths. They are retained
as evidence and regression baselines, rather than repeated on the front page.

- [Original D0–D3 benchmark](BENCHMARK_REPORT.md).
- [TLMPM validation](reports/TLMPM_VALIDATION.md).
- [Earlier wrist-camera validation](WRIST_CAMERA_VALIDATION.md).
- [Original live-Isaac acceptance](reports/ISAAC_LIVE_ACCEPTANCE.md).
- [WP-08 specification](WP08_SPECIFICATION.md) and
  [frozen baseline](evidence/wp08_baseline.json).

Machine-readable summaries are in [evidence](evidence/); captured media and
their provenance are in [media](media/). Large raw experiments, environments
and official assets remain outside Git in ignored local directories.
