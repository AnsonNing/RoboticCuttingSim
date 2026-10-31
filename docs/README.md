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
| Use the reusable WP-11 environment (scene + goal configs, policy API, datasets) | [Policy-ready API](POLICY_READY_API.md) · [WP-11 plan](WP11_PLAN.md) |

## WP-13: integrated candidate and honest rejection (2026-11-01)

- [Results, physical gains and failures](reports/WP13_RESULTS.md), [live phase checklist](WP13_PLAN.md).
- `tlmpm_v7` remains an explicit experimental opt-in. Primary TRIM fails to detach;
  v5 / Law 8 and `particle_surface_v1` remain the production defaults.
- [Frozen 32-episode paired cohort and six Native episodes](../configs/wp13/focused_v1.json).
  Material parameters remain provisional and uncalibrated; publication is still in progress.

## WP-12: retained flat-board workcell (2026-10-31)

- [Final validation, Native demos and acceptance matrix](reports/WP12_FINAL_VALIDATION.md).
- [Generalization: 170 saved kinematic runs, Native coverage and Family D diagnosis](reports/WP12_GENERALIZATION.md).
- [Workcell](reports/WP12_WORKCELL.md), [ADR 0017](adr/0017-blade-master-flat-board-workcell.md):
  one board directly on the table, physical knife grasp, explicit slotted-board variant.
- [Surface decision](reports/WP12_PHASE3_XHIGH_DECISION.md): `particle_surface_v1` default;
  `cut_aware_surface_v1` opt-in, Surface PARTIAL, both frozen comparison gates FAIL.
- [Cutting physics](reports/WP12_CUTTING_PHYSICS.md), [resolution study](reports/WP12_RESOLUTION_STUDY.md)
  and [supported envelope v2](../configs/wp12/supported_envelope_v2.json): law 8 / TLMPM v5 unchanged;
  contact force NOT CONVERGED, material uncalibrated, solid-board oblique limits retained.
- [API readiness](reports/WP12_AGENTIC_READINESS.md), [performance](reports/WP12_PERFORMANCE.md)
  and [environment API](POLICY_READY_API.md); kinematic host-memory gate FAIL remains disclosed.
- [Plan and phase status](WP12_PLAN.md), [specification](WP12_SPECIFICATION.md),
  [historical takeover](HANDOFF_WP12_20261031.md), [current progress](PROGRESS.md).
- [Native gallery](media/wp12/demos/) and [evidence](evidence/wp12/).

## Repository map

| Path | Purpose |
| --- | --- |
| `src/robotic_cutting_sim/` | Physics, scene/goal configuration, controllers, environments, rendering and evaluation |
| `configs/` | Versioned scenes, goals, profiles and frozen protocols |
| `scripts/` | Reproduction runners and read-only diagnostics/reports |
| `tests/` | CPU unit/integration checks; GPU probes invoked separately |
| `docs/`, `docs/evidence/`, `docs/media/` | Reports, compact receipts and published Native media |
| `artifacts/`, `.deps/`, `.venv/` | Ignored local experiments, assets/dependencies and environment; never bundled with Git |

## WP-11 history: reusable cutting environment (2026-10-30)

- Plan and acceptance: [WP11_PLAN.md](WP11_PLAN.md). Decisions:
  - [ADR 0013](adr/0013-ruler-knife-v2-and-trim-orientation-metrics.md): `ruler_knife_v2` and the τ/ψ/ρ/μ orientation metrics;
  - [ADR 0014](adr/0014-scalable-cut-history-sparse-logical-fields.md): `tlmpm_v5` multi-cut history;
  - [ADR 0015](adr/0015-scene-goal-controller-evaluation-separation.md): scene, goal, controller and evaluation as separate layers, with `goal_eval_v2`;
  - [ADR 0016](adr/0016-configured-perception-relocation.md): wrist-depth relocation for configured multi-cut plans.
- Reports:
  - [Phase A: CI](reports/WP11_PHASE_A_CI.md)
  - [Phase C: near-horizontal TRIM](reports/WP11_PHASE_C_TRIM.md)
  - [Phase E round 1](reports/WP11_PHASE_E_ROUND1.md) and [round 2](reports/WP11_PHASE_E_ROUND2.md) (generalization)
  - [Phase F1: grasp stress](reports/WP11_PHASE_F1_GRASP.md)
  - [Phase F2/F3: law 8 boundaries and resolution](reports/WP11_PHASE_F_LAW8.md)
  - [Phase G/H: policy API, profiling, Native reliability](reports/WP11_PHASE_H_PROFILING.md)
  - [Review negative controls](reports/WP11_REVIEW_NEGATIVE_CONTROLS.md)
  - [Phase I: final_v4 regression](reports/WP11_PHASE_I_FINAL_V4_REGRESSION.md)
- Evidence lives under [evidence/wp11/](evidence/wp11/) and media under [media/wp11/](media/wp11/).

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

## Earlier scene and validation references

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
