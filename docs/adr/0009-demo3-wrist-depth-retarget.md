# ADR 0009 — Demo 3: retarget cut 1 from wrist D405 depth (controller profile `path_v2_vision`)

Date: 2026-10-29. Status: accepted, declared before any live Demo 3 episode
under this profile is counted.

## Context

The first live Demo 3 run used law 5 and controller v1
(`artifacts/wp10_claude/demos/live_20261029_b/demo3`). Its outcome by the
plan §4 definition holds:

- three mechanically independent parts (35.6/32.5/31.9%);
- the pushed piece moved 19.3 mm;
- the other pieces stayed stationary;
- no labels were added during the push.

The SLICE-PUSH-01 v3 multi-cut report still fails `declared_cut_planes`. The
digit-1 severed-bond midpoints lie 2.0000000670 mm from the declared plane
x = +30 mm. The frozen bound is 2 mm, and the statistic is quantised in 1 mm
particle layers.

Kinematic M8b shows why (Phase 1 review, 2026-10-29): cut 0 pushes the
remaining bar about +2.3 mm (law 4 pushes it about +1.2 mm). Cut 1 is then
commanded at the world plane, so it lands at about reference x = 27.7 mm.
The reviewer ruled out several fixes as illegitimate: switching laws,
widening the frozen bound, changing board friction, feeding oracle part
positions to the controller, and re-declaring the planes.

## Decision

Add opt-in scripted controller profile `path_v2_vision`, which is path_v2
plus one perception step. Everything else stays unchanged: the task
SLICE-PUSH-01, its v3 thresholds and declared planes, law 5, and all
evaluators.

1. **Calibration frame.** Before cut 0 (start pose, cutting disabled), take
   the latest wrist D405_0 depth frame. Back-project it with that frame's K
   and `T_world_camera`; the extrinsics come from robot kinematics, not
   material state. Select bar-top points above the board top, within the
   bar's nominal |y| band. Measure the bar's −x end face \(x_\text{end,0}\).
   The rendered-surface outward offset is then \(r_s = (-79.5\ \text{mm}) -
   x_\text{end,0}\): the outermost particle centre minus the face.
2. **Survey frame.** After cut 0, lift and move above cut 1, cutting still
   disabled, and settle for path_v2's 0.3 s. Then take the latest wrist depth
   frame and find the kerf of cut 0: the first material face at
   \(x > -30\ \text{mm}\) within the expected kerf band, scanning along +x.
   That face is the left face \(x_\text{kf}\) of the middle+right part. Its
   rigid shift is \(\hat\Delta = x_\text{kf} + 29.5\ \text{mm} + r_s\).
3. **Retarget.** Shift cut 1 and every later waypoint (raise in kerf, push,
   hold, release, retract) by \(\hat\Delta\) along world x. If any
   measurement is invalid (too few points, no face found, or
   \(|\hat\Delta| > 6\ \text{mm}\)), apply no shift and record why. Record the
   measurements, the inputs' timestamps and \(\hat\Delta\) in `result.json`
   and the action rows.

Only rendered RGB-D observations and robot kinematics are used. No particle
state, labels, component IDs or oracle positions reach the controller.

## Acceptance

A live Demo 3 episode under `path_v2_vision` counts only if both hold:

- the plan §4 outcome holds;
- the frozen v3 slice-push report passes, including `declared_cut_planes`.

The v1 run's FAIL stays recorded. If `path_v2_vision` also fails the
declared-plane check, Demo 3 is reported as its outcome PASS plus a
cut-location quality FAIL, with this ADR's measurements attached.

## Amendment 1 (2026-10-29, before any live episode under `path_v2_vision`)

On this rig the wrist D405_0 sits 45 mm in +x of the hand. In the recorded
v1 episode it sees the bar top over about x = −33…+80 mm at the cut-0 start
and about +28…+81 mm at the cut-1 start. Two features the decision relies on
are therefore out of view: the −x end face at calibration, and the cut-0
kerf at the survey pose. The kerf face is visible only at the end of the
lift, and there its estimate is biased: the face top is crushed, and cut 0
severed one particle layer further right than the formula assumes.
Measured there, Δ̂ ≈ 2.6 mm against a true shift of about 1.19 mm.

The applied method is therefore `end_face_plus_x`. It measures the +x end
face of the middle+right part in both frames. That face carries the same
rigid shift as the cut-1 target, so Δ̂ = x_end,+(survey) − x_end,+(calibration),
a relative measurement in which the rendered-surface offset cancels. The
original kerf-face estimate is still computed and recorded, as a diagnostic
only.

Offline validation is in
[evidence](../evidence/phase3/demo3_wrist_survey_offline_001.json), using
the v1 live episode; particle state scores the estimate only:

- Δ̂ = 1.177 mm against the shift of the layers at x = +29.5/+30.5 mm
  (where cut 1 must land): 1.189 mm, error −0.012 mm.
- All 30 calibration × survey frame pairs: worst error 0.065 mm.
- With added depth noise of up to 0.2 mm: worst error 0.098 mm.

Inputs are still rendered depth plus kinematics only. The acceptance rule
above is unchanged.
