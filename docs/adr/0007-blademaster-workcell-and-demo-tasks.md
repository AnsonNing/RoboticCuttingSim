# ADR 0007 — BladeMaster workcell v1, demo tasks and demo runner

Date: 2026-10-29. Status: accepted for implementation; Phase 2–5 of the
[demo plan](../BLADEMASTER_DEMO_PLAN.md). Live demo acceptance still depends
on Phase 1 ([ADR 0006](0006-swept-band-cutting-law-4.md)).

## Context (survey of the current tree)

- Task fixtures are USD cubes with `CollisionAPI` and default PhysX
  material (`isaac_sim_runner._author_task_fixture`). TLMPM uses the same
  post-scenario boxes plus an infinite table plane, with the task material
  friction 0.3. No PhysX friction material is bound to the table, fixtures
  or workpiece supports. The only physics material is the tool grip material
  (0.6/0.6).
- Scenarios only translate the workpiece, extend fixture bases to the table
  and set the TRIM rake. Ruler TRIM needs the raised placement because the
  palm otherwise collides with the table. It also needs a steeper rake, since
  the finger pads cross the target plane at 25°. `ruler_workcell_v2` is
  still NOT_RUN live.
- Recording runs at 5 Hz (`--record-every 10`). No renderer reads
  `run_wp10_live` episodes, and no GIF writer, live particle-debug view or
  overview camera exists.
- `through_cut` always passes through the task target and cannot place an
  offset second cut. With `max_cuts=2`, evaluation returns NOT_RUN unless a
  `MultiCutGoal` is supplied. No post-cut manipulation evaluator exists.
  Episodes are capped at 400 steps (8 s, from the v1 protocol).

## Decision

1. **Scenario `blademaster_workcell_v1`** goes in a new file,
   `configs/wp10_scenarios_v3.json`; the v1/v2 scenario files stay
   unchanged. It inherits the `ruler_workcell_v2` placement: +50 mm, 35°
   TRIM rake and the measured TRIM IK seed. Each task support box becomes a
   **cutting board** of the same top surface:
   - a 10 mm wood-coloured board box (local z ∈ [−10, 0] mm), widened in y
     to a board half-width of 50 mm;
   - a separate grey riser from the table to the board.
   The same boxes go to both PhysX (`CollisionAPI` + `MaterialAPI`) and
   TLMPM fixture contacts. The PhysX static/dynamic friction equals the
   task material friction (0.3); restitution is 0. The table gets the same
   material, so no visual fixture lacks a matching collider and no collider
   is invisible.
   - For CUT-01-R/CUT-02, the board keeps the frozen pedestal gap (|x| <
     20 mm) as a relief channel, so the edge can overtravel below the bar.
   - For TRIM-01-R, the board is the support floor.
   - The y widening is beyond every material particle (|y| ≤ 15/30 mm), so
     the bar's support contacts are unchanged.
   Franka 1 stays at the shared safe park, the wrist D405s stay on rig v5,
   and the Global D455 keeps its close-view mount.
   - If the raised workpiece leaves the D455 frame, a scenario-specific
     look-at height is recorded explicitly, never silently.
   - A presentation-only **overview camera** (RGB, not part of any
     observation) is authored for the videos.
2. **Versioned demo task protocol `configs/cutting_tasks_v3.json`** with
   its own spec class, duration and active interval. v1/v2 files, IDs and
   thresholds stay unchanged.
   - Demo 1 and Demo 2 reuse the v2 tasks CUT-01-R and TRIM-01-R with
     their inherited thresholds.
   - Demo 3 adds **`SLICE-PUSH-01`**: the CUT-01-R bar (160×30×20 mm,
     same provisional material) on a board with two 8 mm relief slots at
     x = ±30 mm. Board segments cover [−90, −34], [−26, 26] and [34, 90] mm,
     giving pieces of about 31/37.5/31% mass. The episode is 16 s with
     max_cuts 2.
3. **Plan builders** (`policies.py`, additive):
   - `through_cut` gains an optional offset of the cut line; the default
     keeps today's behaviour exactly;
   - `slice_push`: cut 0 at x = −30 mm → lift and move with cut disabled
     → cut 1 at x = +30 mm → raise the edge to 3 mm above the board inside
     the kerf → push +x by 20 mm with the blade face, `cut_enabled=False`
     → hold → retract.
   The scripted controller corrects the commanded edge from the
   **measured edge pose** (tool proprioception, not oracle material state),
   so the edge stays within ±0.3 mm of each target plane.
4. **Evaluation** (read-only, separate from the frozen oracle):
   - Demo 1 uses the frozen strict oracle plus a research report. The
     oblique variant is scored against the actual swept plane (ADR 0006
     statistic); its frozen location check is reported as is.
   - Demo 2 reports separation (≥ 90% above-plane removal, substrate
     intact, detached below-plane mass ≤ 1%) separately from the precision
     score, which keeps the 0.5 mm surface thresholds.
   - Demo 3 uses `evaluate_multi_cut` with the two declared planes, plus a
     new `post_cut_manipulation` evaluator:
     - ≥ 3 mechanical components after cut 1;
     - no new labels while cutting is disabled;
     - the pushed piece's centroid moves ≥ 10 mm along the push direction;
     - every other piece moves ≤ 2 mm;
     - components never merge.
5. **Recording and media**:
   - demos record cameras at 10 Hz (`--record-every 5`) and keep every raw
     episode;
   - a new `scripts/render_demo_episode.py` renders, from one live episode:
     overview and Global D455 RGB, both wrist D405 RGB and depth, a
     particle debug view coloured by mechanical component (stencil graph),
     and diagnostics (cutting reaction, finger contact, tool pose, labelled
     count and component history);
   - each frame is annotated with controller, time, cut on/off, cut index
     and the final evaluation;
   - output is MP4 plus a small GIF under `docs/media/demos/`.
6. **Demo runner**: `scripts/run_blademaster_demos.py --demo {1,2,3,all}`
   runs Native (`open_cutting_env(..., runtime="isaac_sim",
   tool_attachment_mode="physical_grasp", tool_init_mode="pregrasped")`)
   with law 4, scripted controllers and fresh output directories. It then
   evaluates, renders, and writes one summary JSON per demo plus an index.
   It never overwrites a previous run.

## Consequences

- None of this changes frozen physics, thresholds or records. Scenario v3
  and task v3 are opt-in, and all old scenarios and tasks keep working.
- Acceptance videos come only from live Native episodes. Kinematic runs
  are used only as preflight and are labelled as such.
- Lab parity (P1) and the cut-aware surface (P1) remain separate. If the
  component ParticleSurface shows Demo 3 pieces visually merged, that is
  reported as a rendering limitation, not hidden.
