# WP-11 Phase F1: long-duration physical grasp of `ruler_knife_v2_y`

**Verdict:** `FAIL_OR_INCOMPLETE_F1_GATED_CASES`. All 31 predeclared cases ran,
and 30 of 31 met their predeclared expectation. The one gated failure is a
negative control, `low_friction_trim_advance_0p1n`. Every hold, rotation,
cutting-load and torque case passed the inherited holding gates, as did both
release cases and three of the four negative controls. The gates were not
changed after the results.

- **Evidence:** [`docs/evidence/wp11/phaseF1_grasp_001.json`](../evidence/wp11/phaseF1_grasp_001.json)
- **Run:** `artifacts/wp11/phaseF1/f1_grasp_003`
- **Protocol:** [`configs/physical_tool_controls_v8_ruler_v2_wp11.json`](../../configs/physical_tool_controls_v8_ruler_v2_wp11.json),
  declared and committed (`0c39b23`) before any run.
- **Runner:** `scripts/run_physical_tool_controls.py`, through an additive opt-in
  path, `start_profile: wp11_home_v1`. Runs with earlier protocols are unchanged.
- **Code commit of the run:** `9a4cb5f`.
- **Full physics acceptance:** false.

## Setup

- **Scene and task:** the scene is `trim01_wp11_workcell` and the goal is `trim_bump_rho10`.
  This gives the compiled task `CFG-trim01_wp11_workcell-trim_bump_rho10-a33f6613`,
  the same task as the Phase D smoke run.
- **Runtime:** Native Isaac Sim 6.2, PhysX CPU articulation. The rigid step is 2 ms and the control step 20 ms.
- **Seed and GPU:** seed 42 on GPU 0.
- **Process per case:** each case runs in a fresh process.
- **HOME start:** HOME is the declared `q_home`: hand pointing down, blade hanging. The
  physical pregrasp settle must report `PASS_STABLE_GRASP`.
- **Path to the TRIM pose:** the planner's own ideal REORIENT schedule (66 knots,
  ≤ 2 mm and ≤ 2° per step). It ends in free space 30 mm above the workpiece.
  The measured rake there is ρ = 10.03°, with an error of 0.39 mm and 0.57 mrad.
- **Grip:** inherited from v7. The finger drive limit is 20 N (the configured Franka
  limit), μ = 0.6, and the initial opening is 11.2 mm. The measured handle contact
  force is about 18 N per finger.
- **Loads:** wrenches are applied at the measured edge centre. Their axes are taken from
  the measured edge frame at the case start: x is the advance, y is the edge line and
  z is the blade normal. For v2_y the blade normal is the finger closing axis (hand y).
- **Material:** never advanced. There is no material contact.

The gates are copied unchanged from v7/v5:

- relative translation ≤ 2 mm and relative rotation ≤ π/60 rad;
- relative speeds in the last 0.5 s ≤ 0.02 m/s and 0.2 rad/s;
- bilateral loaded-handle fraction ≥ 0.95;
- penetration ≤ 1 mm;
- no unintended support;
- the grasp monitor never reports LOST.

## Results

Notes on the columns:

- Slip is the tool-to-hand relative motion over the case.
- The finger force is the per-finger handle contact force, mean over the last 0.5 s
  (left/right).
- Torque_y is the applied torque about the finger axis, taken about the grip centre.
- Gravity adds +0.0083 N·m about the finger axis in the TRIM pose.

| Case | Gate | Status | Slip (mm / mrad) | Finger force L/R (N) | Torque_y (N·m) | Monitor |
| --- | --- | --- | --- | --- | --- | --- |
| home_hold_30s | hold | PASS | 0.0003 / 0.004 | 18.0 / 18.0 | 0 | HOLDING |
| trim_hold_30s | hold | PASS | 0.0003 / 0.023 | 18.5 / 17.6 | 0 | HOLDING |
| reorient_round_trip_1x (peak hand speed 1.86 rad/s) | hold | PASS | 0.0015 / 0.12 | 18.0 / 18.0 | – | HOLDING |
| reorient_round_trip_2x (peak hand speed 3.70 rad/s) | hold | PASS | 0.0048 / 0.14 | 18.0 / 18.0 | – | HOLDING |
| force −advance 0.1 / 0.5 N | hold | PASS | ≤ 0.0008 / ≤ 0.13 | 18.5 / 17.6 | −0.009 / −0.045 | HOLDING |
| force −normal 0.1 / 0.5 N | hold | PASS | ≤ 0.0007 / ≤ 0.13 | 18.7 / 17.4 at 0.5 N | 0 | HOLDING |
| force +edge line 0.1 / 0.5 N | hold | PASS | ≤ 0.0005 / ≤ 0.07 | 18.5 / 17.6 | −0.003 / −0.014 | HOLDING |
| torque about each edge axis, 0.01 / 0.05 N·m | hold | PASS | ≤ 0.0007 / ≤ 0.13 | 18.5 / 17.6 | up to −0.05 | HOLDING |
| cutting_replay_c4_x1 (400 steps, peak 0.081 N) | hold | PASS | 0.0004 / 0.04 | 18.5 / 17.6 | −0.0002 | HOLDING |
| cutting_replay_c4_x10 (peak 0.81 N) | reported | (hold gates pass) | 0.0008 / 0.13 | 18.5 / 17.6 | −0.002 | HOLDING |
| force −advance 2 N | reported | (hold gates fail) | 0.60 / 218 | 18.5 / 17.6 | −0.18 | SLIPPING from 0.52 s |
| force −advance 5 N | reported | (hold gates fail) | 15.5 / 2299 | – | −0.45 | LOST at 0.12 s |
| force −normal 2 N | reported | (hold gates fail: last-window speed) | 0.46 / 6.9 | 19.5 / 16.7 | 0 | HOLDING |
| force −normal 5 N | reported | (hold gates fail) | 4.4 / 322 | 23.7 / 18.1 | 0 | SLIPPING |
| force +edge line 2 N | reported | (hold gates pass) | 0.001 / 0.12 | 18.5 / 17.6 | −0.056 | HOLDING |
| force +edge line 5 N | reported | (hold gates fail) | 0.63 / 104 | 18.5 / 17.6 | −0.14 | SLIPPING |
| weak_grip_home (0.5 N) | negative control | PASS | 432 / 2258 | 0 | 0 | LOST at 0.12 s |
| low_friction_home (μ = 0.02) | negative control | PASS | 300 / 2163 | 0 | 0 | LOST at 0.16 s |
| weak_grip_trim (0.5 N) | negative control | PASS | 9.3 / 450 | 0.7 / 0.2 | 0 | LOST at 0.22 s |
| low_friction_trim (μ = 0.02) | diagnostic | (reported) | 3.6 / 450 | 18.4 / 17.6 | 0 | SLIPPING |
| **low_friction_trim_advance_0p1n** | negative control | **FAIL** | 0.056 / 6.0 | 18.5 / 17.6 | −0.009 | HOLDING |
| release_home | release | PASS | 134 mm drop, free flight 156 ms at −9.81 m/s² | 0 | – | released |
| release_trim | release | PASS | 29 mm drop, free flight 72 ms at −9.81 m/s² | 0 | – | released |

Penetration across all hold cases is ≤ 9.5 µm. The 30 s holds show no creep:
the relative speed in the last window is 0.6 µm/s at HOME and 45 µm/s in the TRIM pose.
After each round trip the tool returns to within 17 µm of its HOME world pose. The
joints differ from `q_home` by up to 0.056 rad, which is redundancy (null-space) drift
of the IK, not slip.

## Loads the end-held v2 grip tolerates

The grip holds with micrometre slip under all of these:

- the 30 s stationary holds at HOME and in the TRIM pose;
- wrist reorientation at twice the planner REORIENT rate;
- every cutting-load force (≤ 0.5 N);
- every torque (≤ 0.05 N·m);
- the recorded C4 reaction, and the same reaction at 10×.

**Margin over the recorded cutting reaction.** The C4 ρ = 10° peak is 0.081 N, and
most of it is along −blade normal, pushing into one finger. Against that peak:

- the gated 0.5 N cases are a 6× margin;
- the 10× replay still holds;
- along −advance, slip begins between 0.5 N and 2 N, a factor of about 6 to 25.

## Risk for cutting: torque about the finger axis

The weakest direction is rotation about the finger closing axis, which is the
blade normal. Only torsional pad friction resists it, and the edge sits 90 mm from
the grip centre along the edge line.

- **Held:** 0.045 N·m (0.5 N along −advance) and 0.05 N·m of pure torque about the
  blade normal.
- **Slips:** 0.18 N·m (2 N along −advance) rotates the tool 12.5° while the
  bilateral contact stays at 1.0. At 0.45 N·m (5 N) the grasp is LOST.
- **Slip onset:** between 0.05 and 0.18 N·m.

The −normal and edge-line stress cases also rotate the tool, through the 28 mm offset
of the edge from the grip axis, and through torque about hand x for loads along the
closing axis.

**Gravity in the TRIM pose.** At ρ = 10° gravity gives 0.0083 N·m about the finger
axis and −0.047 N·m about hand x. The pad couple resists the hand-x part. At μ = 0.6
the finger-axis torque is harmless. At μ = 0.02 that 0.0083 N·m alone rotates the tool
by 450 mrad (the low_friction_trim case).

**What this means for cutting:**

- Recorded cutting loads are about 7× or more below the torsional slip onset.
- Plowing, sawing or a stalled advance would leave that margin quickly.
  Advance resistance above about 1 N at the edge is the first expected failure: the
  blade turns in the fingers, with no drop.
- Finger-axis torque should be monitored, using the grasp monitor's rotation
  warning at π/60, in C5 and any higher-load operation.

## Failure and preserved runs

**`low_friction_trim_advance_0p1n` (FAIL, kept).**

- The 0.1 N load along −advance gives −0.0090 N·m about the finger axis. This
  almost cancels the +0.0083 N·m gravity torque, leaving about −0.0007 N·m. So at
  μ = 0.02 there was no larger response: +0.06 mm and +6 mrad, against the required
  5 mm or 87 mrad.
- The load sign was fixed before running, from the C4 reaction direction.
- This is a design miss in the negative control. It is not evidence that friction is
  ineffective. The gravity-only low-friction case did slip.
- A sign-reversed or larger-load variant would be a new protocol version. It was not
  run.

**`low_friction_trim` (diagnostic).** Before the run I predicted it might hold,
because the weight acts mostly along the closing axis. Translation did stay small
(3.6 mm). But the tool rotated 450 mrad about the finger axis, so the prediction was
wrong for rotation.

**Preserved runs:**

- **`f1_grasp_001` (failure).**
  - `home_hold_30s` failed as FAIL_SOURCE_CHANGED. A concurrent agent edited
    `goal_planner.py` (opt-in `cut_extent`) during the case, and the frozen
    source-integrity check rejected it. Its holding gates otherwise passed.
  - `force_advance_2n` and `force_advance_5n` failed because of a runner bug.
    `wp11_extra_metrics` read a non-existent `time_s` from the grasp-monitor reports,
    and Kit exited Python inside `app.close()` before the error was recorded.
  - The bug is fixed in `9a4cb5f`, with a regression test.
  - The run was stopped after 8 cases.
- **`f1_grasp_002` (partial).** Pinned worktree at the same buggy commit; stopped
  after 3 cases.
- **`smoke_001` and `dryrun_001`.** Pre-checks.
- **Pinned worktree.** The final run used a git worktree pinned at `9a4cb5f`
  (`/mnt/data/ningan/rcs_f1_worktree`). Other agents' uncommitted edits therefore
  could not change the hashed sources.

## Commands

```
cd /mnt/data/ningan/rcs_f1_worktree   # detached at 9a4cb5f; .deps/.venv/artifacts symlinked to the main checkout
TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_physical_tool_controls.py --runtime isaac_sim \
  --protocol configs/physical_tool_controls_v8_ruler_v2_wp11.json --seed 42 --gpu 0 \
  --tool-profile ruler_knife_v2_y --output-dir artifacts/wp11/phaseF1/f1_grasp_003
TMPDIR=/mnt/data/ningan/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wp11_f1_controls.py tests/test_physical_tool_controls.py   # 24 passed
```

The matrix took 1161 s of wall time (19:37:56Z to 19:57:17Z). Artifacts total 2.8 GB.

## Limits

- **Rigid probes only.** The material is never advanced, and the workpiece is not a
  PhysX collider in these runs.
- **No velocity feedforward.** `rigid_probe_step` issues zero joint-velocity targets;
  the live runner adds feedforward. The knot rate is the planner's rate, and the hand
  rate is measured.
- **Force-only replay.** The C4 replay contains force only, because C4 did not record
  torque. No C5 live reaction existed when the protocol was frozen.
- **Fixed wrench directions.** Each wrench direction is fixed at the case start.
- **Narrow coverage.** One seed, Native only (Isaac Lab NOT_RUN), and provisional,
  uncalibrated friction, finger and tool parameters.
- **Grip opening differs from live.** The live runner and C5 use the default 14 mm
  initial finger opening, not 11.2 mm. Both settle to about 18 N.
