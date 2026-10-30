# WP-11 — Cutting environment refinement, horizontal trimming, robustness and policy readiness

**Source:** the user's WP-11 specification (2026-10-30), the reference image
`/home/ningan/a.png` ("我的刀具要像是 a.png 這樣"), and the user's
supplementary "Project Goal & Generalization Requirements" (2026-10-30).

**Guiding goal (supplement).** Build a reusable, physically interactive
cutting environment, not a fixed set of demos. That means:

- the same robot, tool, material solver and evaluation pipeline serve any
  scene configured by the user;
- the task goal is defined separately from the scene;
- any controller can run the task: scripted, a learned policy or an agent;
- a single episode can contain multiple cuts and operations.
**Branch:** `claude/wp11-cutting-refinement`, cut from main `3e26d6b`.
**Baseline:** the BladeMaster demos with `--demo-set final_v4` and cutting
law 8 (see the [demo report](reports/BLADEMASTER_DEMOS.md)). That baseline
and its thresholds are frozen; every new feature is versioned and opt-in.

## Baseline audit (2026-10-30)

**main commit:** `3e26d6b`.

**Tests**

| Run | Command | Result |
| --- | --- | --- |
| Local, `cc881f7` | `.venv/bin/python -m pytest -q` | 780 passed, 5 strict xfail, 0 failed (815 s) |
| CI (GitHub Actions `tests.yml`, run 40915870753) | — | 5 failed, 748 passed, 27 skipped, 5 xfailed |

The five CI failures:
- 4 pinned scripted-rollout/plan hashes (`test_blademaster_demo.py:98`,
  `test_wrist_depth_survey.py:251`, three cases);
- `ModuleNotFoundError: matplotlib` in
  `test_tool_profile.py::test_profile_sdf_matches_ruler_polygon_geometry`.

**Demos** (`live_20261029_law8_final`, Native): 1a, 1b, 2 (35° TRIM) and 3
(SLICE-PUSH) all PASS. Known limits are listed in the report:
- provisional material;
- law 8 validated only for in-plane cuts and 25°/35° rakes, with the 15°
  gate untested;
- no Lab parity;
- TRIM precision 5/8;
- the live runtime does not log stroke or plow state;
- the swept normal is ill-conditioned for sawing;
- stroke persistence is centre-speed only.

**Reusable components:**
- `ToolProfile` / `make_profile_tool_spec` / `require_profile_consistency`;
- physical grasp (`ToolGraspConfig`, `native_tool`, `grasp_monitor`);
- `WaypointPolicy` with `path_v2`, `path_v2_vision` and measured-edge
  feedback;
- `open_cutting_env` (reset/step/observe/render/evaluate/close);
- the scenario and task versioning (`wp10_scenarios_v3`,
  `cutting_tasks_v3`);
- TLMPM `tlmpm_v4` (laws 1–8);
- `run_wp10_live.EpisodeRecorder`, `demo_evaluation`,
  `run_blademaster_demos`, `render_demo_episode`, `run_phase1_matrix` and
  `rescore_phase1_matrix`.

**Disk:** `/mnt/data` had 30 GB free after removing stale pytest temp files;
the root disk has 12 GB free. A live 400-step episode recorded at 10 Hz is
about 0.46 GB.

## Phases and acceptance

| Phase | Priority | Scope | Done when |
| --- | --- | --- | --- |
| A | P0 | Fix CI and reproducibility | Clean GitHub Actions passes. Hash tests check real controller behaviour with a justified tolerance, not regenerated hashes. matplotlib is handled explicitly. |
| B | P0 | `ruler_knife_v2`, [ADR 0013](adr/0013-ruler-knife-v2-and-trim-orientation-metrics.md) | One profile drives visual, collider, mass, grasp frame, edge frame and TLMPM. A Native physical grasp is stable. It looks like the reference. |
| C | P0 | Near-horizontal TRIM | Preflight over rakes 0/5/10/15/25/35° (Newton IK and real colliders). Material-level 1 mm kinematic test with the full negative controls. Live Native `HOME → REORIENT → PRE-CUT → SKIM → WITHDRAW → RETURN_HOME`: ≥ 90% bump separated, substrate intact, grasp held. |
| D | P0 | Return-to-home and episode lifecycle | Physical return trajectory; separate task and home outcomes; episode state machine. |
| E | P0 | Generalization | Seeded scene randomization with feasibility classes. Round 1: ≥ 5 seeds per core demo, with total, feasible and overall success rates. |
| F | P0/P1 | Grasp stability and law 8 validation | Grasp stress suite on v2. Law 8 boundary cases (15° gate, near-horizontal, rotation, sawing, finite edge). Resolution/dt study. |
| G | P0 | Policy-ready API and dataset collector | Research preset; validated actions with explicit rejection; observations separate from privileged data; collector; multi-episode reliability (`POLICY_READY`). |
| H | P1 | Profiling and safe optimization | Per-stage timing; low-risk overheads reduced; Research Fast and Full Physics profiles. |
| S | P0 | Scene / task-goal / controller / evaluation separation (supplement §2, §4) | A config-only path from scene (workpiece geometry, pose, protrusions, supports) plus goal (sever at planes, divide into N pieces, remove selected protrusions or all in a region, trim to a plane or shape, sequential cuts, move a piece) to an evaluator. The same scene works with different goals, and the same goal with different controllers. Success criteria follow the goal. |
| M | P0 | Scalable multi-cut history, [ADR 0014](adr/0014-scalable-cut-history-sparse-logical-fields.md) | `tlmpm_v5`: packed 2-bit digits (K ≤ 15) and sparse per-node logical-field slots. Equal to v4 law 8 for K ≤ 2. Scaling measured. Test B (≥ 3 slices) and Test C (3 protrusions removed one by one) at 1 mm. All negative controls pass. |
| I | P0 | Final validation and delivery | `final_v4` regression, horizontal TRIM video, generalization, policy-ready validation, the supplement's Tests A–F, CI, docs. Merge after review. |

### Generalization acceptance (supplement §8)

| Test | Requirement |
| --- | --- |
| A | One workpiece, several feasible knife orientations and cutting directions. Real cutting, reachability, grasp stability and persistent separation are all checked. |
| B | Three or more configurable slices in one episode, giving the correct number of independently moving pieces with persistent cut history. |
| C | A workpiece with three protrusions at different positions, removed one at a time by successive operations. Removal is scored per protrusion, together with substrate integrity and physical chip separation. |
| D | Predeclared random seeds varying pose, dimensions, protrusion layout and goal. Feasibility, execution, task success and failure reasons are reported separately. |
| E | The same task run by the scripted controller and by an independently supplied action sequence through the public API, with no hidden task-specific simulator changes. |
| F | Consecutive episodes with reset, state isolation, recording, termination and evaluation. Failed episodes remain usable. |

Environment reset (a direct physics-state reset, suitable for RL) stays
separate from the optional physical return-to-home used for demonstrations.
RL episodes are not forced to return home physically.

Execution order is:

1. A ∥ B ∥ M, then C2 preflight ∥ C4 material ∥ S.
2. C3/C5 live.
3. D, then E/F (with Tests A–D), then G (Tests E/F), then H, then I. Every phase records its baseline, hypothesis, minimal
change, tests, preserved failures, review, and a commit/push.

## Status

The **進行中** section of `docs/PROGRESS.md` holds the running jobs and the
next step.
