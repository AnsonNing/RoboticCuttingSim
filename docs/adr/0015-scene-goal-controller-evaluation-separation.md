# ADR 0015 — Configurable scenes and task goals, independent controllers and goal-driven evaluation

Date: 2026-10-30. Status: accepted for implementation. This is WP-11 Phase S,
from the supplement's §2, §4 and §8.

## Context

Today a task is a hand-written JSON entry in v1/v2/v3. That entry fixes the
workpiece geometry, fixtures, tool, reference edge path, goal kind
(sever/flatten), thresholds and duration. Scenarios only translate the
workpiece or change the rake, and they require identity orientation.

As a result, every new experiment needs a new task id, and often code
changes. The goal is inseparable from the scene, and the scripted plans
(`through_cut`, `skim`, `slice_push`) are tied to task ids.

The supplement asks for separate layers. Each layer works with any choice in
the others: the same scene with different goals, the same goal with
different controllers, and new tasks written as configuration only:

| Layer | What it is |
| --- | --- |
| Scene | What exists |
| Task goal | What should be achieved |
| Controller | How the robot tries |
| Physics | What happens |
| Evaluation | How the outcome is measured |

## Decision

All of the following is additive. v1–v3 tasks, scenarios, plans and
thresholds keep their exact behaviour.

1. **Scene config** `rcs_scene_v1`, in JSON under `configs/wp11/scenes/`:
   - **Workpiece:** base box (length, width, height); a list of protrusions
     (id, family dome/gaussian/ellipsoid_cap/ridge, x, y, radius_x,
     radius_y, height); world pose (translation and yaw); material profile.
   - **Supports:** a cutting board, solid or slotted at declared x positions
     with a slot width, plus a riser height. These are compiled to the same
     PhysX + TLMPM boxes (scenario v3 semantics).
   - **Tool:** profile id and grasp preset.
   - **Solver:** `tlmpm_v4` or `tlmpm_v5`, cutting law, and K.
   - **Randomization:** optional; seed and ranges. The sampled values are
     written into the compiled scene.

   Protrusions are part of the one continuous heightfield sample. They are
   never separate bodies.
2. **Goal config** `rcs_goal_v1`, under `configs/wp11/goals/`. Each goal type
   has its own evaluator, which reports PASS/FAIL plus continuous and
   per-item metrics. Success follows the goal, not a universal flat-surface
   rule.

   | Goal | Fields | Notes |
   | --- | --- | --- |
   | `sever` | planes, expected pieces, minimum part fraction | Location: the frozen 2 mm statistic for grid-aligned planes, otherwise the ADR 0006 Amendment 1 swept-plane rule |
   | `divide` | N pieces, minimum fraction | |
   | `remove_protrusions` | ids, `all`, or a local region; per-item removal ≥ 0.9; substrate ≥ 0.99; detached below-plane ≤ 1% | Precision is reported separately |
   | `trim_plane` | plane, region | Separation sets the verdict; precision is reported |
   | `move_piece` | reference-region selector, direction, minimum displacement; others ≤ 2 mm; no label change while disarmed | |
   | `sequence` | ordered sub-goals | All must hold at the end; per-step results are reported |
3. **Compiled task.** `compile_task(scene, goal)` returns a
   `ConfiguredTaskSpec`, a subclass of `ProfileTaskSpec`. Its id is
   `CFG-<scene>-<goal>-<hash8>`, and its protocol document is the merged
   configuration with its sha256, so runtimes, recorders and receipts work
   unchanged. Duration and control dt are declared in the goal (default
   control dt 0.02 s).
   - **Equivalence test:** a config equal to TRIM-01-R or CUT-01-R must
     produce identical particles, fixtures and tool mount to the frozen
     task.
   - **Workpiece yaw:** supported wherever the runtimes already use
     `world_from_workpiece`. Any remaining identity-only assumption becomes an
     explicit error, never a silent wrong result.
4. **Controllers.**
   - **Planner:** `goal_planner.plan(scene, goal, profile, options)` builds
     `Waypoint` sequences for the scripted controller from scene
     configuration only. Sever planes produce generalized through-cuts with
     any plane, edge line and approach. Protrusions produce `surface_trim`
     passes (HOME → REORIENT → PRE-CUT → SKIM → WITHDRAW per protrusion, then
     RETURN_HOME). Piece moves produce a blade-face push.
   - **Same interface for everyone:** external policies and agents use the
     same `open_cutting_env(task=compiled)` reset/step/observe/evaluate API.
   - **No oracle input:** the planner never reads particle, label or
     component state. Perception hooks (for example the ADR 0009 wrist-depth
     survey) are optional inputs.
5. **Evaluation.** A read-only goal evaluator runs over recorded episodes.
   It supports packed (`tlmpm_v5`) and ternary codes, and keeps the three
   layers separate: integrity, outcome and quality.

## Acceptance

- Configs reproduce TRIM-01-R and CUT-01-R exactly; no existing task changes.
- One scene with three goals gives three different, correct verdicts on the
  same recorded or kinematic episode.
- Test E: one goal is run by the scripted planner and by an independent action
  sequence through the public API, with no hidden task-specific changes.
- Tests B, C and D are expressed as scene + goal configs (multi-slice,
  three protrusions, seeded randomization with feasibility classes).

## Amendment 1 (2026-10-30): `goal_eval_v2`, step-scoped sequence evaluation (WP-11 Phase E round 2)

**Context.** Round 1 (`docs/reports/WP11_PHASE_E_ROUND1.md`) found that a `sequence` of sever then
remove_protrusions cannot pass under the v1 evaluator, even when the cut and the removals worked (Test D
s1702, s1704):

- v1 scores every step on the final components of *all* cut digits.
- `remove_protrusions` (and `trim_plane`) require a single substrate component holding ≥ 99% of the
  below-plane mass. After a sever there are always two.
- The sever's location statistic assigns every cross-component reference node to the nearest declared plane,
  so the chip/substrate interfaces of later skims (tens of mm away) count against the sever plane (77 mm and
  48 mm).

**Decision.** A versioned, opt-in evaluator, `goal_eval_v2`, selected by the top-level goal field
`"evaluator_version": "goal_eval_v2"`. Rules:

1. **v1 is the default and is unchanged.** A goal without the field validates to the same normalized form, so
   compiled task ids, plans and v1 evaluations stay byte-identical. This is pinned by
   `tests/test_goal_eval_v2.py`: v1 evaluation digests computed with the pre-v2 `goals.py` of `c29f02a`,
   the C5b compiled ids, and round-1 stage-1 task ids. A declared version is part of the goal, so it gives
   new ids. Steps never carry their own version.
2. **Digit allocation.** Cut digits are allocated to steps by the declared convention, which is the
   goal-planner numbering:
   - steps are taken in order, and digits are numbered across the episode;
   - `sever`: one digit per plane;
   - `divide`: pieces − 1;
   - `remove_protrusions`: one per selected protrusion;
   - `trim_plane`: one;
   - `move_piece`: none.

   When a compiled task is given, the allocation is cross-checked against the planner's plan report
   (`MATCH` / `MISMATCH` / `NOT_AVAILABLE`). A mismatch, or a recording that keeps fewer digits than needed,
   makes the sequence `NOT_RUN`. It is never scored.
3. **Scoped components.** For a set of digits *D*, the scoped components are the mechanical components
   (the same eight-node compatibility graph, packed or ternary) of the final codes with every digit outside
   *D* set to 0. This is exactly the history those digits alone would have written.
4. **`sever` / `divide` step.** Every sever check (expected substantial pieces, main-piece mass, plane
   location, far-particle rule) is computed on the components of the step's **own** digits. Location
   therefore comes only from the separation interfaces of its own cut digits.
5. **`remove_protrusions` / `trim_plane` step.**
   - *Prior pieces* are the anchored (bottom-layer) components of the digits of the **earlier sever/divide
     steps** of the sequence.
   - *After* is the components of those digits plus the step's own.
   - Substrate integrity is judged **per prior piece**: within each one, the largest after-component holds
     ≥ `minimum_substrate_share` of its below-plane mass, ≤ `maximum_detached_below_fraction` of it is
     detached from the substrate, and exactly one substrate component remains. So a step may not split
     material below its plane.
   - Removal, unselected retention, residual height and surface quality use the after-substrate.
   - Digits of earlier remove/trim steps are not in scope: their chips are those steps' results.
   - `trim_plane` steps are scoped only for horizontal planes; otherwise the step is `NOT_RUN`.
6. **Unscoped steps and goals.** `move_piece` steps, which read the state of their own move window, and
   non-sequence goals are scored exactly as in v1.
7. **Thresholds are unchanged.** Every threshold is the v1 threshold. Sequence success still requires every
   step to hold at the end of the episode. The integrity layer is unchanged.

**Consequences.**

- A remove step that really splits a sever piece below its plane still fails. A sever that is misplaced
  still fails its location check. Both are covered by negative tests.
- v2 assumes the declared digit convention. An external controller that numbers its digits differently
  must be scored with v1, or the convention must be declared.
- Round-1 verdicts and evidence are not changed. Their v2 rescoring is a read-only `POST_HOC_DIAGNOSTIC`
  (round-2 report).
