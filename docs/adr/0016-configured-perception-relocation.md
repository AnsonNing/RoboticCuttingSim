# ADR 0016 — Wrist-depth relocation as an opt-in planner perception hook for configured multi-cut sever goals

Date: 2026-10-30. Status: accepted, declared before any live run that uses the hook.

## Context

Round 1 of WP-11 Phase E failed every SLICE-PUSH sample on cut 1: the 5 kinematic seeds and live s1301 under
`live_001` (`docs/reports/WP11_PHASE_E_ROUND1.md`).

- After cut 0 the +x part slides +1.16 to +1.25 mm.
- Cut 1 is then commanded at the planned world plane, so it lands 1 mm short in the reference frame.
- The frozen location statistic reads 4.0 mm against the 2 mm bound.

ADR 0009 (`path_v2_vision`) fixed the same error for SLICE-PUSH-01 in `blademaster_workcell_v1` with `ruler_knife_v1`.
That code is tied to the task id SLICE-PUSH-01 and to its legacy plan labels, and ADR 0015 §4 allows perception hooks
only as optional planner inputs.

**Measured on the recorded `live_001` s1301 episode (wp11_workcell_v1, ruler_knife_v2_y, v5 camera rig):**

- **Camera pose.** The wrist D405_0 sits about 113 mm above the knife edge, 174 mm to its −y side and 45 mm to its −x
  side. Its optical axis is local +y, so image rows run along the bar's local x.
- **The ADR 0009 settle poses cannot see the right features.**
  - At the cut-1 start settle, the +x end face projects to row −14, outside the image.
  - Lines on the near side of the bar are hidden beyond x ≈ edge x by the blade.
  - At the cut-0 start settle the visible bar top ends at x ≈ +28 mm.
  - So the ADR 0009 `end_face_plus_x` method has no face in either frame: 0 bracketed lines.
- **The −x end face is visible in both frames, but it belongs to the wrong piece.** It is the end of the piece left of
  cut 0, which moved −1.23 mm. Measured from two different poses it gave −1.227 mm, so the estimator is nearly
  independent of the viewpoint.
- **The kerf face of the middle piece is visible at the cut-1 settle, but it is biased.** It measures the
  blade-deformed kerf layer: +1.66 mm, against +1.14 mm for the rigid part that cut 1 must split. That is an error
  of +0.53 mm, about half a particle layer.

## Decision

Add the opt-in planner option `planner_options.perception = "wrist_depth_relocate_v1"`. When the option is absent,
nothing changes: every plan, report, compiled task id, controller receipt and recording is byte-identical.

1. **Eligibility.** The option applies to a `sever` goal with two or more planes, either the whole goal or one step of
   a `sequence`. The plan is rejected with a `SceneConfigError` (never silently inert) in any of these cases:
   - the planes do not share one normal along local ±x;
   - a plane declares `approach_local`;
   - the plane offsets along the normal are not strictly monotonic;
   - the controller profile is not `path_v2`;
   - the goal has no such sever.

   Monotonic offsets mean that each next cut acts on the outer piece on the progression side. That piece carries the
   bar's end face on that side, and the face is far from every cut.
2. **Planned survey pose.** The pose is derived from the scene config only, with no look-at aiming, no observation-pose
   optimization and no visibility gate.
   - **Position:** the knife edge at local (s·(L/2 + 10 mm), 0, top + `cut_clearance_m`), where s is the progression
     side.
   - **Orientation:** the edge keeps the cuts' own orientation, so the camera keeps its fixed v5 mount relative to the
     hand.
   - **Why this pose:** the edge is at the PRE_CUT height, 10 mm beyond the end face and in free space. The blade plane
     lies outside every ray to the face, and the camera views the face from the same height and at about the same
     angle as the PRE_CUT frames in which the estimator worked. This was checked by projecting the end face through
     the recorded camera-from-edge transform: rows 144–156 and columns 716–802 of 480 × 848, depth 0.16–0.19 m.
   - **Waypoints:**
     - `SURVEY_APPROACH:surveyJ`, 30 mm above the pose (`safe_height_m`), then 10 mm above it;
     - `SURVEY:surveyJ`, reached at the path_v2 approach speed (0.02 m/s) and held for the path_v2 settle (0.3 s);
     - `SURVEY_LEAVE:surveyJ`, 30 mm above the pose.
   - **When:** `survey0` (calibration) runs after HOME and before cut 0. `survey{k+1}` runs after cut k's WITHDRAW and
     before cut k+1.
   - The C2 plan preflight checks these waypoints like any other free-space waypoint.
3. **Measurement** (policy `perception_relocate.RelocatePolicy`, a `WaypointPolicy` subclass in the delta law).
   - **When:** at the end of each `SURVEY` hold, the policy takes the current wrist D405_0 depth frame. Its timestamp
     must equal the observation time.
   - **How:** the frame is back-projected with its own K and `T_world_camera`. Bar-top points are selected in the scene's
     nominal height and |y| band, and the end-face feature on side s is measured with the unchanged
     `wrist_depth_survey.end_face_x`: the profile crossing 0.2 mm below the local top, median over at least 15 lines,
     line spread at most 0.6 mm.
   - **Estimate:** Δ̂_J = x_face(survey J) − x_face(survey 0). This is a relative measurement from the same planned
     pose, so the rendered-surface offset cancels.
   - **Invalid estimate:** if a face is invalid, or |Δ̂| > 6 mm, no shift is applied and the reason is recorded.
4. **Relocation.** Each valid Δ̂_J is the end piece's total displacement along local x since calibration. Every
   waypoint from cut J's first waypoint up to RETURN_HOME is moved by R_world_local·(Δ̂_J, 0, 0), with two exceptions:
   - `SURVEY*` waypoints, which are fixed observation poses;
   - RETURN_HOME, the fixed workcell HOME.

   This covers the later cuts, holds and withdrawals, and the push of a following `move_piece` step, which lowers into
   cut J's kerf. A later survey replaces the offset; it is not added to it. After a survey the policy restarts at the
   next waypoint, as `VisionRetargetPolicy` does.
5. **Inputs, deployable or privileged** (recorded in the plan report, the policy receipt and `result.json`):

   | Input | Used for | Class |
   | --- | --- | --- |
   | Wrist D405_0 depth, valid mask, K | the measurement | deployable sensor |
   | `T_world_camera` of the frame | back-projection | robot kinematics plus the fixed hand-eye mount (in a real cell, a hand-eye calibration) |
   | Nominal edge from joint-state proprioception (`edge_from_observation`) | excluding the blade slab from the depth points | deployable (joint state plus the declared grasp) |
   | Scene config (L, W, H, pose, supports) and goal planes | survey pose, selection bands, side | configuration, known before the episode |
   | Measured tool-edge pose (`--edge-feedback measured`) | waypoint tracking only, as in `live_001` | **privileged** simulator tool truth (already disclosed for path_v2/v3). The hook never reads it. |
   | Particle, label, component and oracle state | not used | **never** a controller input. It is read only by the evaluator and by the report-only truth comparison. |

   As in ADR 0009 Amendment 2, the rendered surface is rebuilt per mechanical component, so depth is not fully
   independent of the cut topology. The measured end face is far from every cut.
6. **Runner wiring.** For `--scene/--goal` runs whose planner options request the hook, `scripts/run_wp10_live.py`
   builds `RelocatePolicy` with the existing `wrist_depth_source`. Such a run must record cameras
   (`--record-cameras` or a vision observation). The per-step `policy_vision` events and the final `policy_vision`
   record are written to the episode rows and `result.json`; for any other run these rows are unchanged. The
   kinematic environment has no cameras and does not apply the hook.
   `isaac_sim_runner.py` and `cutting_surface.py` are not changed.

Nothing else changes: `wrist_depth_survey.py` defaults, `path_v2_vision`, SLICE-PUSH-01 and final_v4, the goal
evaluators and their thresholds, physics, the cutting laws, and v1–v4 records.

## Validation and acceptance

- **CPU tests** (`tests/test_perception_relocate.py`):
  - the hook is inert when not requested;
  - the eligibility errors are raised;
  - the plan structure with the hook;
  - synthetic depth frames from the measured camera geometry recover a known shift;
  - an invalid frame applies no shift;
  - the shift is applied to the right waypoints only.

  The existing pin tests (`test_wrist_depth_survey`, `test_controller_path_v2`, `test_scene_goal_config`,
  `test_generalization_round1*`, `test_wp11_phase_d`) must pass unchanged.
- **Live:** a supplementary stage `configs/wp11/generalization_round1_live_004.json` (a "controller variant"), with
  SLICE-PUSH s1301, s1302 and s1303. The task builder, goal evaluator, thresholds, classes and controller are those of
  `live_001`; only the perception hook is added, and with it the duration that the round's duration rule gives for the
  longer plan.
  - A seed counts as TASK_SUCCESS only under the inherited criteria, including the frozen `cut_at_declared_planes`.
  - The round-1 kinematic failures and `live_001` s1301 stay as recorded.
  - Each seed is reported with the measured Δ̂ against the oracle displacement of the cut-J layers. The oracle value is
    report-only.
