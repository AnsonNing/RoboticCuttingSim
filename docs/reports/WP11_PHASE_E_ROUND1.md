# WP-11 Phase E — generalization round 1 (preflight + kinematic material)

Date: 2026-10-30. Branch `claude/wp11-cutting-refinement`.
Round file: [`configs/wp11/generalization_round1.json`](../../configs/wp11/generalization_round1.json), predeclared in
`4b2fe08` before any declared sample was built, preflighted or run.
Evidence: [`docs/evidence/wp11/phaseE_round1_001.json`](../evidence/wp11/phaseE_round1_001.json).

**Scope.** Every stage-2 verdict is a **kinematic ideal-tracking** result (`runtime_source = kinematic_ideal_tracking`):
the knife edge goes exactly where it is commanded. Robot dynamics, grasp, slip, controller tracking and cameras are not
modelled, and the material is provisional and uncalibrated. These are **not live results**. `GRASP_FAILURE` is a
live-only class; it is not exercised in this round and is never assigned. Stage 1 is geometry only: Newton IK plus
the C2 MuJoCo collision model with the held `ruler_knife_v2_y`, in `wp11_workcell_v1` (HOME = `q_home`).

## Result

| Family / test | Samples | SCENE_INFEASIBLE | IK_FAILURE | Preflight-feasible | CUTTING_FAILURE | NOT_RUN | Kinematic TASK_SUCCESS | Success among feasible | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUT | 6 | 0 | 0 | 6 (100%) | 0 | 0 | 6 | 100% | 100% |
| TRIM | 6 | 0 | 0 | 6 (100%) | 0 | 0 | 6 | 100% | 100% |
| SLICE_PUSH | 5 | 0 | 0 | 5 (100%) | 5 | 0 | 0 | 0% | 0% |
| TEST_A | 8 | 0 | 0 | 8 (100%) | 1 | 0 | 7 | 88% | 88% |
| TEST_B | 5 | 0 | 0 | 5 (100%) | 0 | 0 | 5 | 100% | 100% |
| TEST_C | 5 | 0 | 0 | 5 (100%) | 0 | 0 | 5 | 100% | 100% |
| TEST_D | 6 | 0 | 0 | 6 (100%) | 3 | 0 | 3 | 50% | 50% |
| **All** | **41** | 0 | 0 | 41 (100%) | 9 | 0 | **32** | 78% | 78% |

GRASP_FAILURE = 0 in every row (not exercised: kinematic). Nothing is excluded. Each declared seed is reported
exactly once. Feasible = samples − SCENE_INFEASIBLE − IK_FAILURE.

**Reading the failures.** All 9 CUTTING_FAILUREs are reported as such under the predeclared rules. Their causes differ
(details under "Failures"):

- **5 × SLICE-PUSH: real kinematic location error.** All pieces separate and every push succeeds. After cut 0, the +x
  part slides +1.16 to +1.25 mm (wedge displacement, as in Phase M). Cut 1 therefore lands one particle layer short
  (−1.00 mm in the reference frame), and the frozen 2 mm statistic reads 4.0 mm. The ADR 0009 wrist-depth
  relocation (live only, not exercised) exists to correct exactly this.
- **1 × Test A s2 (−20° oblique): marginal location error.** The swept-plane rule fails by 0.02 mm: actual − ideal is
  1.02 mm against the 1.00 mm limit (one particle spacing).
- **1 × Test D s1701: solver limit.** tlmpm_v5 slot overflow at S = 8 (318 entries) during the second skim after an
  oblique sever. The run aborted as declared ("never silently merged").
- **2 × Test D s1702 / s1704 (sever then remove): evaluator-composition limits, not cutting failures.** The sever
  produced two pieces (0.69/0.30 and 0.72/0.28) and the protrusions were removed (0.970/0.960 and 0.984). However,
  (a) the remove step requires a single substrate component holding ≥ 99% of the below-plane mass, which can never
  hold after a sever; and (b) the sever step's frozen location statistic assigns every cross-component node,
  including the chip/substrate interfaces at z = H tens of mm away, to the only declared plane, giving 77 / 48 mm
  against an ideal 1.5 mm. The `sever_then_remove_all` goal kind is therefore unsatisfiable as declared under the
  current goals.py semantics. That is a round-design finding, and the class is left unchanged.

## Families and rules (predeclared)

| Family | Seeds | Randomized (all physical workpiece geometry or pose) | Goal |
| --- | --- | --- | --- |
| CUT | 1101–1106 | L 130–170, W 26–34, H 16–24 mm; xy ±20 mm, yaw ±20°; cut x = ±0.2 L; angle 0° / 20° oblique by seed index | sever (30 mm relief slot) |
| TRIM | 1201–1206 | dims, pose; one protrusion: x ±0.25 L, y ±2 mm, radii 9–13 / 9–11.5 mm, h 5–9 mm; dome / gaussian / ellipsoid_cap × rake 10° / 25° (all six combinations) | remove_protrusions |
| SLICE-PUSH | 1301–1305 | dims, pose; cuts at −(24–36) / +(24–36) mm; 8 mm slots | sever ×2, then push of the +x piece 20 mm (≥ 10 mm, others ≤ 2 mm) |
| Test A | 0–7 | fixed 160×30×20 bar, dome at +45 mm; vertical, +20°, −20°, slanted-advance (15°) cuts at −32 mm; skims +x ρ10, +x ρ25, −x ρ10, +x ρ10 ψ20 | sever / remove |
| Test B | 1501–1505 | dims, pose; 3 / 4 slices (by seed index), equal spacing ±3 mm; tlmpm_v5 | divide into K+1 |
| Test C | 1601–1605 | dims, pose; three protrusions at −0.3125 L / 0 / +0.3125 L (±4 mm), sizes and families cycled; ρ = 10°; tlmpm_v5 | remove all, one pass and digit each |
| Test D | 1701–1706 | dims, pose, cut location, 1–2 protrusions with families, cut angle, rake and goal kind, all drawn | sever / remove all / remove selected / sever then remove all |

Common: `wp11_workcell_v1` (+100 mm), `ruler_knife_v2_y`, controller `path_v2`, law 8, 1 mm particles, 4 mm grid,
0.1 ms, material seed 42, `tlmpm_v4` for ≤ 2 digits else `tlmpm_v5`. The episode length is the ideal plan end + 1 s,
rounded up to 0.5 s. The RNG is `numpy.random.default_rng([salt, seed])`, with draws in declared order. Raw draws and
applied values are in the evidence (`samples[].draws`, `derived`) and in each compiled scene's `randomization.sampled`.

**Rules added before the declaration commit** (found on non-declared development seeds 90001–90004; recorded in the
round file and the commit message):

1. `goal_planner` opt-in `cut_extent = "plane_section"` (additive; plans and ids without it are unchanged; pinned-id
   tests pass). The v1 bounding-box extent ends a 20° oblique cut 20–30 mm below the bar, outside the wp11 action
   workspace (z ≥ −12 mm), so the reference schedule never finishes.
2. Staged stage-1 geometry for scenes with ≥ 2 protrusions. The C2 model's single convex hull over all protrusions
   bridges the gaps between them and keeps protrusions already removed. Without this, the dev Test C/D samples
   "collided" by −8 mm at a later PRE_CUT.
3. Vertical sever planes are snapped to solver node planes (multiples of 4 mm). Lengths and widths are rounded to even
   millimetres, heights to whole millimetres. On the real 4 mm stencil, the frozen 2 mm midpoint statistic scores an
   *ideal* cut 1.5 mm on a node plane but 2.0–3.5 mm between node planes. The bound itself is unchanged.

## Stage 1 — preflight (all 41 samples)

Command (clean detached worktree `4b2fe08`, `git_dirty = False`, CPU):

    cd /mnt/data/ningan/RoboticCuttingSim-wp11-E-4b2fe08
    TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_generalization.py preflight \
        --out artifacts/wp11/phaseE/round1_stage1_001      # 10 min 05 s wall, rc 0

- 41 / 41 PASS, covering the full HOME → … → RETURN_HOME schedule (IK chain from `q_home`, dense ≤ 2 mm / ≤ 1°
  samples).
- The binding checks are `blade_collision` (blade vs workpiece at 0.0 mm in the contact-free PRE_CUT, the edge 0.2 mm
  above the plane) and `knife_handle_clearance` (7.5–11 mm). The worst joint-limit margin is ≥ 0.57 rad. The
  informational physical return-to-home plan (Phase D) was PLANNED for all 41.
- Mean wall per sample: CUT 11.8 s, TRIM 8.1 s, SLICE-PUSH 22.6 s, Test A 10.7 s, Test B 26.3 s, Test C 13.9 s,
  Test D 13.9 s.
- No sample was SCENE_INFEASIBLE or IK_FAILURE. The ranges were chosen to be physically reasonable inside the
  C2-validated workcell; a wider round is needed to measure the feasibility boundary.

## Stage 2 — kinematic material physics (all 41 feasible samples)

    cd /mnt/data/ningan/RoboticCuttingSim-wp11-E-2a93b99      # clean detached worktree, git_dirty = False
    TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_generalization.py queue \
        --stage1 artifacts/wp11/phaseE/round1_stage1_001 --out-root artifacts/wp11/phaseE/round1_kinematic_001 \
        --log-dir artifacts/wp11/logs --queue-log artifacts/wp11/logs/phaseE_queue_003.log \
        --devices cuda:1 cuda:0 --max-concurrent 2 --max-other 2
    .venv/bin/python scripts/run_generalization.py summarize --stage1 artifacts/wp11/phaseE/round1_stage1_001 \
        --material-root artifacts/wp11/phaseE/round1_kinematic_001 --out docs/evidence/wp11/phaseE_round1_001.json

- The queue ran 05:03–06:38 KST. 22 samples ran on cuda:0 and 19 on cuda:1; the device is recorded per sample.
- Each sample used at most 2 of this queue's jobs per GPU, and launched only while at most 2 other active jobs were on
  that GPU.
- Output size: 2.9 GB in total, 74–79 MB mean per run (max 93 MB). History stride = max(5, ceil(steps / 40)), stored
  as float32.

**Queue history (all recorded in the queue logs).**

| Queue | Commit | What happened |
| --- | --- | --- |
| `queue_001` | `4b2fe08` | cuda:0 only; waited (3 active Phase F rows); stopped before any launch when the coordinator released GPU 1. |
| `queue_002` | `8d9d520` | Device list added. Two launches failed in argparse (`--config` placed after the subcommand, rc 2) before any output directory was created; the queue was stopped. |
| `queue_003` | `2a93b99` | Fix, plus abort on repeated fast launch failures; ran every sample once. |

Log-derived class: Test D s1701 raised `SlotOverflowError` inside `LiveCuttingMaterial.advance` (step 678 of 900),
so it has no `result.json`. `summarize` reads its preserved log and assigns the round file's "solver state invalid
(slot overflow)" class, CUTTING_FAILURE. Its directory is not modified.

Runtime per episode (wall s; mean [min–max]; wall per simulated second):

| Family | Episode wall | per sim-s |
| --- | --- | --- |
| CUT | 255 [197–349] | 25.0 |
| TRIM | 161 [138–199] | 23.1 |
| SLICE-PUSH | 1009 [500–1456] | 44.8 |
| Test A | 341 [233–533] | 35.6 |
| Test B | 902 [770–986] | 34.6 |
| Test C | 469 [394–546] | 33.8 |
| Test D | 348 [236–510] | 28.6 |

These timings come from two GPUs shared with other jobs (Phase F rows, other agents), so they are throughput figures,
not solver benchmarks.

### Per sample

Metric definitions:

| Metric | sever / divide | remove |
| --- | --- | --- |
| Completion | substantial pieces / expected pieces | minimum removed mass fraction over the selected protrusions |
| Separation | expected substantial pieces present | selected protrusions removed and a single substrate |
| Position error | frozen 2 mm midpoint statistic (grid-aligned planes), or actual − ideal swept-plane statistic (oblique planes) | residual substrate particle-centre height above the plane in the protrusion footprints (quality only, includes plowed or deformed substrate) |
| Overcut | debris mass outside the main pieces | below-plane mass detached from the substrate |

Sequences report the worst step. Test B (divide) location is quality only: about 5 mm, from bar sliding between
cuts; the location bound does not apply.

| Sample | Goal | Sampled / derived | Solver | Particles | Episode s | Class | Completion | Separation | Position error mm | Overcut | Episode wall s | Device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_CUT_s1101 | sever | 144x34x18 mm; yaw +13.1°, xy (+12, +17) mm; cut_angle_deg 0.0; cut x 20.0 mm | tlmpm_v4 | 88128 | 10.0 | TASK_SUCCESS | 1 | True | 1.5 | 0 | 212 | cuda:1 |
| E1_CUT_s1102 | sever | 164x32x17 mm; yaw -9.9°, xy (+10, +9) mm; cut_angle_deg 20.0; cut x 4.4 mm | tlmpm_v4 | 89216 | 10.0 | TASK_SUCCESS | 1 | True | 0.342 | 0 | 224 | cuda:1 |
| E1_CUT_s1103 | sever | 150x32x20 mm; yaw -11.2°, xy (+8, +18) mm; cut_angle_deg 0.0; cut x -8.0 mm | tlmpm_v4 | 96000 | 10.5 | TASK_SUCCESS | 1 | True | 1.5 | 0 | 236 | cuda:0 |
| E1_CUT_s1104 | sever | 138x30x16 mm; yaw -16.6°, xy (-0, +4) mm; cut_angle_deg 20.0; cut x -12.1 mm | tlmpm_v4 | 66240 | 10.0 | TASK_SUCCESS | 1 | True | 0.171 | 0 | 197 | cuda:0 |
| E1_CUT_s1105 | sever | 168x30x23 mm; yaw +5.7°, xy (+5, +2) mm; cut_angle_deg 0.0; cut x -24.0 mm | tlmpm_v4 | 115920 | 10.5 | TASK_SUCCESS | 1 | True | 1.5 | 0 | 349 | cuda:1 |
| E1_CUT_s1106 | sever | 168x30x18 mm; yaw +10.2°, xy (-11, -9) mm; cut_angle_deg 20.0; cut x -24.0 mm | tlmpm_v4 | 90720 | 10.0 | TASK_SUCCESS | 1 | True | 0.482 | 0 | 310 | cuda:1 |
| E1_TRIM_s1201 | remove_protrusions | 164x30x16 mm; yaw +18.6°, xy (+15, -19) mm; protrusion_family dome; rake_deg 10.0 | tlmpm_v4 | 79329 | 7.0 | TASK_SUCCESS | 0.936 | True | 4.78 | 0.000102 | 138 | cuda:0 |
| E1_TRIM_s1202 | remove_protrusions | 162x28x20 mm; yaw -7.3°, xy (+19, -12) mm; protrusion_family gaussian; rake_deg 10.0 | tlmpm_v4 | 91221 | 7.0 | TASK_SUCCESS | 0.948 | True | 3.98 | 0 | 151 | cuda:0 |
| E1_TRIM_s1203 | remove_protrusions | 160x34x18 mm; yaw -15.8°, xy (+18, +0) mm; protrusion_family ellipsoid_cap; rake_deg 10.0 | tlmpm_v4 | 99323 | 7.0 | TASK_SUCCESS | 0.993 | True | 0.202 | 0 | 175 | cuda:0 |
| E1_TRIM_s1204 | remove_protrusions | 142x30x23 mm; yaw +19.6°, xy (-10, +18) mm; protrusion_family dome; rake_deg 25.0 | tlmpm_v4 | 98548 | 7.0 | TASK_SUCCESS | 0.977 | True | 0.0529 | 4.08e-05 | 164 | cuda:1 |
| E1_TRIM_s1205 | remove_protrusions | 154x30x24 mm; yaw -12.6°, xy (+1, -6) mm; protrusion_family gaussian; rake_deg 25.0 | tlmpm_v4 | 111424 | 7.0 | TASK_SUCCESS | 0.934 | True | 6.8 | 0 | 199 | cuda:0 |
| E1_TRIM_s1206 | remove_protrusions | 140x32x16 mm; yaw +13.5°, xy (-16, -1) mm; protrusion_family ellipsoid_cap; rake_deg 25.0 | tlmpm_v4 | 73171 | 7.0 | TASK_SUCCESS | 0.997 | True | 0.69 | 0 | 142 | cuda:1 |
| E1_SLICE_PUSH_s1301 | sequence | 156x28x17 mm; yaw -6.3°, xy (+14, +9) mm; cut x -28.0/24.0 mm | tlmpm_v4 | 74256 | 21.5 | CUTTING_FAILURE | 1 | True | 4 | 0 | 500 | cuda:0 |
| E1_SLICE_PUSH_s1302 | sequence | 150x32x21 mm; yaw +18.1°, xy (-8, -4) mm; cut x -24.0/28.0 mm | tlmpm_v4 | 100800 | 23.0 | CUTTING_FAILURE | 1 | True | 4 | 0 | 1456 | cuda:1 |
| E1_SLICE_PUSH_s1303 | sequence | 168x30x18 mm; yaw -14.2°, xy (+17, +12) mm; cut x -28.0/36.0 mm | tlmpm_v4 | 90720 | 22.0 | CUTTING_FAILURE | 1 | True | 4 | 0 | 1383 | cuda:1 |
| E1_SLICE_PUSH_s1304 | sequence | 170x32x19 mm; yaw -6.7°, xy (-13, -6) mm; cut x -28.0/32.0 mm | tlmpm_v4 | 103360 | 22.5 | CUTTING_FAILURE | 1 | True | 4 | 0 | 736 | cuda:0 |
| E1_SLICE_PUSH_s1305 | sequence | 156x34x20 mm; yaw -12.0°, xy (+2, +14) mm; cut x -32.0/28.0 mm | tlmpm_v4 | 106080 | 23.0 | CUTTING_FAILURE | 1 | True | 4 | 0 | 968 | cuda:0 |
| E1_TEST_A_s0 | sever | Vertical through-cut at local x = -32 mm (advance -z, edge l | tlmpm_v4 | 97044 | 11.5 | TASK_SUCCESS | 1 | True | 1.5 | 0 | 533 | cuda:0 |
| E1_TEST_A_s1 | sever | 20 deg oblique through-cut at x = -32 mm (normal turned +20  | tlmpm_v4 | 97044 | 11.5 | TASK_SUCCESS | 1 | True | 0.342 | 0 | 379 | cuda:0 |
| E1_TEST_A_s2 | sever | 20 deg oblique through-cut at x = -32 mm, opposite inclinati | tlmpm_v4 | 97044 | 11.5 | CUTTING_FAILURE | 1 | True | 1.02 | 0 | 384 | cuda:0 |
| E1_TEST_A_s3 | sever | Vertical plane at x = -32 mm cut with the advance tilted 15  | tlmpm_v4 | 97044 | 12.5 | TASK_SUCCESS | 1 | True | 1.5 | 0 | 437 | cuda:0 |
| E1_TEST_A_s4 | remove_protrusions | Remove the dome with a skim along +x at rake 10 deg. | tlmpm_v4 | 97044 | 7.0 | TASK_SUCCESS | 0.969 | True | 2.45 | 0 | 244 | cuda:1 |
| E1_TEST_A_s5 | remove_protrusions | Remove the dome with a skim along +x at rake 25 deg. | tlmpm_v4 | 97044 | 7.0 | TASK_SUCCESS | 0.973 | True | 6.35 | 0.000104 | 242 | cuda:1 |
| E1_TEST_A_s6 | remove_protrusions | Remove the dome with a skim along -x at rake 10 deg (edge li | tlmpm_v4 | 97044 | 8.0 | TASK_SUCCESS | 0.967 | True | 3.91 | 2.08e-05 | 277 | cuda:0 |
| E1_TEST_A_s7 | remove_protrusions | Remove the dome with a skim along +x at rake 10 deg and edge | tlmpm_v4 | 97044 | 7.0 | TASK_SUCCESS | 0.971 | True | 3.25 | 3.13e-05 | 233 | cuda:1 |
| E1_TEST_B_s1501 | divide | 152x32x20 mm; yaw +2.1°, xy (+17, -5) mm; slices 3; cut x -38.0/-1.7/35.6 mm | tlmpm_v5 | 97280 | 24.0 | TASK_SUCCESS | 1 | True | 4.62 | 0 | 882 | cuda:1 |
| E1_TEST_B_s1502 | divide | 164x28x19 mm; yaw -11.9°, xy (-8, +13) mm; slices 4; cut x -48.4/-18.9/16.7/51.9 mm | tlmpm_v5 | 87248 | 30.0 | TASK_SUCCESS | 1 | True | 5.43 | 0 | 967 | cuda:0 |
| E1_TEST_B_s1503 | divide | 154x32x20 mm; yaw +9.8°, xy (+8, +11) mm; slices 3; cut x -39.5/1.1/36.4 mm | tlmpm_v5 | 98560 | 24.0 | TASK_SUCCESS | 1 | True | 5.44 | 0 | 904 | cuda:0 |
| E1_TEST_B_s1504 | divide | 152x30x19 mm; yaw +14.9°, xy (-18, -0) mm; slices 4; cut x -42.8/-15.9/14.8/47.6 mm | tlmpm_v5 | 86640 | 30.0 | TASK_SUCCESS | 1 | True | 5.14 | 0 | 986 | cuda:1 |
| E1_TEST_B_s1505 | divide | 166x30x18 mm; yaw -2.5°, xy (+20, -16) mm; slices 3; cut x -42.7/2.4/41.9 mm | tlmpm_v5 | 89640 | 23.0 | TASK_SUCCESS | 1 | True | 5.86 | 0 | 770 | cuda:1 |
| E1_TEST_C_s1601 | remove_protrusions | 168x32x16 mm; yaw -10.9°, xy (-16, +16) mm; families dome/gaussian/ellipsoid_cap | tlmpm_v5 | 89172 | 14.0 | TASK_SUCCESS | 0.909 | True | 7.4 | 0.000116 | 485 | cuda:0 |
| E1_TEST_C_s1602 | remove_protrusions | 152x32x22 mm; yaw +5.5°, xy (+17, +9) mm; families gaussian/ellipsoid_cap/dome | tlmpm_v5 | 110286 | 14.0 | TASK_SUCCESS | 0.946 | True | 0.135 | 0 | 546 | cuda:0 |
| E1_TEST_C_s1603 | remove_protrusions | 152x32x19 mm; yaw +9.1°, xy (+8, +6) mm; families ellipsoid_cap/dome/gaussian | tlmpm_v5 | 94927 | 13.5 | TASK_SUCCESS | 0.968 | True | 0.0929 | 0.00013 | 457 | cuda:1 |
| E1_TEST_C_s1604 | remove_protrusions | 152x30x17 mm; yaw +2.2°, xy (+1, -10) mm; families dome/gaussian/ellipsoid_cap | tlmpm_v5 | 80924 | 14.0 | TASK_SUCCESS | 0.965 | True | 0.676 | 0 | 394 | cuda:0 |
| E1_TEST_C_s1605 | remove_protrusions | 166x30x17 mm; yaw +20.0°, xy (-14, -9) mm; families gaussian/ellipsoid_cap/dome | tlmpm_v5 | 87018 | 14.0 | TASK_SUCCESS | 0.944 | True | 0.735 | 0 | 463 | cuda:1 |
| E1_TEST_D_s1701 | sequence | 142x28x18 mm; yaw +0.3°, xy (+5, +1) mm; cut_angle_deg 20.0; rake_deg 25.0; goal_kind sever_then_remove_all; protrusions 2; families ellipsoid_cap/ellipsoid_cap; cut x -33.4 mm | tlmpm_v5 | 74799 | 18.0 | CUTTING_FAILURE | — | — | — | — | — | cuda:0 |
| E1_TEST_D_s1702 | sequence | 140x30x19 mm; yaw +13.3°, xy (+4, +17) mm; cut_angle_deg 0.0; rake_deg 10.0; goal_kind sever_then_remove_all; protrusions 2; families gaussian/gaussian; cut x -28.0 mm | tlmpm_v5 | 80600 | 18.0 | CUTTING_FAILURE | 0.96 | False | 77 | 0.00958 | 510 | cuda:1 |
| E1_TEST_D_s1703 | remove_protrusions | 164x28x20 mm; yaw +11.9°, xy (+18, +3) mm; cut_angle_deg 20.0; rake_deg 25.0; goal_kind remove_all; protrusions 1; families ellipsoid_cap; cut x -47.7 mm | tlmpm_v4 | 92706 | 7.0 | TASK_SUCCESS | 0.992 | True | 4.93 | 0.000142 | 236 | cuda:0 |
| E1_TEST_D_s1704 | sequence | 144x30x22 mm; yaw -5.9°, xy (+8, +15) mm; cut_angle_deg 0.0; rake_deg 10.0; goal_kind sever_then_remove_all; protrusions 1; families dome; cut x -32.0 mm | tlmpm_v4 | 95726 | 15.5 | CUTTING_FAILURE | 0.984 | False | 48 | 0.00705 | 367 | cuda:0 |
| E1_TEST_D_s1705 | sever | 158x32x17 mm; yaw -6.9°, xy (-6, +6) mm; cut_angle_deg 0.0; rake_deg 10.0; goal_kind sever; protrusions 2; families gaussian/gaussian; cut x -36.0 mm | tlmpm_v4 | 86768 | 11.0 | TASK_SUCCESS | 1 | True | 1.5 | 0 | 317 | cuda:1 |
| E1_TEST_D_s1706 | sever | 150x32x18 mm; yaw -14.1°, xy (-15, -12) mm; cut_angle_deg 0.0; rake_deg 10.0; goal_kind sever; protrusions 1; families gaussian; cut x -28.0 mm | tlmpm_v4 | 86679 | 11.0 | TASK_SUCCESS | 1 | True | 1.5 | 0 | 310 | cuda:1 |

Per-protrusion removal, Test C (p0 / p1 / p2):

| Sample | p0 | p1 | p2 | Max slots used (S = 8) |
| --- | --- | --- | --- | --- |
| E1_TEST_C_s1601 | 0.972 | 0.909 | 0.998 | 6 |
| E1_TEST_C_s1602 | 0.960 | 0.997 | 0.946 | 4 |
| E1_TEST_C_s1603 | 1.000 | 0.968 | 0.982 | 8 |
| E1_TEST_C_s1604 | 0.966 | 0.965 | 0.997 | 5 |
| E1_TEST_C_s1605 | 0.957 | 0.994 | 0.944 | 4 |

Test B pieces and slot use:

| Sample | Slices | Mechanical components | Max slots used (S = 8) |
| --- | --- | --- | --- |
| E1_TEST_B_s1501 | 3 | 4 | 4 |
| E1_TEST_B_s1502 | 4 | 5 | 4 |
| E1_TEST_B_s1503 | 3 | 4 | 4 |
| E1_TEST_B_s1504 | 4 | 5 | 5 |
| E1_TEST_B_s1505 | 3 | 4 | 4 |

The Test B samples with 4 slices give 5 components and those with 3 slices give 4. Every sample has persistent
history (integrity PASS), and no slot overflow occurs. Test C s1603 used all 8 slots without overflowing, and
Test D s1701 overflowed: S = 8 is at its limit for multi-skim and sever + skim sequences.

## Failures (every one, with reason)

- E1_SLICE_PUSH_s1301: CUTTING_FAILURE — outcome: step0:slice_two:cut_at_declared_planes
- E1_SLICE_PUSH_s1302: CUTTING_FAILURE — outcome: step0:slice_two:cut_at_declared_planes
- E1_SLICE_PUSH_s1303: CUTTING_FAILURE — outcome: step0:slice_two:cut_at_declared_planes
- E1_SLICE_PUSH_s1304: CUTTING_FAILURE — outcome: step0:slice_two:cut_at_declared_planes
- E1_SLICE_PUSH_s1305: CUTTING_FAILURE — outcome: step0:slice_two:cut_at_declared_planes
- E1_TEST_A_s2: CUTTING_FAILURE — outcome: cut_at_declared_planes
- E1_TEST_D_s1701: CUTTING_FAILURE — solver state invalid: slot overflow (run aborted at step 678, phase SKIM:p1): 318 (particle, corner) entries found all 8 slots of their node held other codes; the affected substeps are invalid (fields are never merged). Rerun with a larger slots_per_node.
- E1_TEST_D_s1702: CUTTING_FAILURE — outcome: step0:sever:cut_at_declared_planes, step1:remove_all:substrate_holds_below_plane_mass, step1:remove_all:single_substrate_component
- E1_TEST_D_s1704: CUTTING_FAILURE — outcome: step0:sever:cut_at_declared_planes, step1:remove_all:substrate_holds_below_plane_mass, step1:remove_all:single_substrate_component

SLICE-PUSH diagnosis (`artifacts/wp11/phaseE/round1_diagnostics_slice_push_001.txt`, script alongside it):

| Sample | Cut 0 label boundary | Cut 1 label boundary | Material shift before cut 1 | Frozen statistic (cut 0 / cut 1) |
| --- | --- | --- | --- | --- |
| s1301 | −28.00 (exact) | +23.00 vs +24 | +1.16 mm | 1.5 / 4.0 mm |
| s1302 | −24.00 | +27.00 vs +28 | +1.24 mm | 1.5 / 4.0 mm |
| s1303 | −28.00 | +35.00 vs +36 | +1.19 mm | 1.5 / 4.0 mm |
| s1304 | −28.00 | +31.00 vs +32 | +1.20 mm | 1.5 / 4.0 mm |
| s1305 | −32.00 | +27.00 vs +28 | +1.25 mm | 1.5 / 4.0 mm |

## Nominated seeds for the later live stage (not run here)

| Purpose | Seeds |
| --- | --- |
| Representative successes | CUT s1101 (vertical, yaw +13°, cut at +20 mm); CUT s1102 (20° oblique); TRIM s1203 (ellipsoid cap, ρ 10°, 0.993); TRIM s1201 (dome, ρ 10°, 0.936, lowest TRIM margin); Test A s6 (−x skim, edge −y); Test B s1502 (4 slices, 5 pieces); Test C s1602 (three protrusions, all ≥ 0.946) |
| Representative failures | SLICE-PUSH s1301 (second-cut 1 mm shift; the live run should exercise the ADR 0009 relocation, which targets this error); Test A s2 (−20° oblique, marginal location) |
| Not for live until fixed | Test D s1701 (slot overflow at S = 8 needs a larger `slots_per_node`); Test D s1702 / s1704 (the sequence goal is unsatisfiable as declared) |

## Open items for round 2 (new round file; nothing here is changed retroactively)

- Widen the pose, size and rake ranges until SCENE_INFEASIBLE and IK_FAILURE actually occur; round 1 measured no
  feasibility boundary.
- `sever_then_remove` needs a per-step evaluation that does not rely on a single substrate or on a single-plane
  location statistic for the whole history (goals.py change, versioned).
- tlmpm_v5 with sever + skims: evaluate S = 16 for mixed sequences.
- The SLICE-PUSH second-cut wedge shift is planner/physics behaviour; a kinematic round with measured-shift
  re-targeting would separate it from the live-only ADR 0009 feature.
- Live (Native) runs of the nominated seeds, with grasp monitoring, to exercise GRASP_FAILURE.

## Tests

- `tests/test_generalization_round1.py`: 24 passed.
- Full CPU suite at `4b2fe08`: 969 passed, 1 skipped, 5 xfailed in 1277 s
  (`artifacts/wp11/logs/phaseE_cpu_full_4b2fe08_001.log`).

## Live stage (Native, 2026-10-30)

Declaration: [`configs/wp11/generalization_round1_live.json`](../../configs/wp11/generalization_round1_live.json)
(`1ecfe5e`, committed and pushed before any live run). The round file declares no live criteria, so this file does;
the round file, the kinematic evidence and every threshold are unchanged. Supplementary declaration for Test C only:
[`configs/wp11/generalization_round1_live_002.json`](../../configs/wp11/generalization_round1_live_002.json)
(`4b2eab6`, before its run). Evidence:
[`docs/evidence/wp11/phaseE_round1_live_001.json`](../evidence/wp11/phaseE_round1_live_001.json).

**Scope.** These are live Native results (`runtime_source = live_isaac_sim`). They come from Isaac Sim 6.2 with Kit/PhysX
driving the Franka, the physically grasped `ruler_knife_v2_y` (`physical_grasp`, pregrasped at HOME, never fixed to the
hand), the retained TLMPM material (law 8, 1 mm particles, 4 mm grid, 0.1 ms) and RTX cameras. There is one seed per
sample, so there are no repeat statistics. The material is provisional and uncalibrated, and there is no Lab parity.

**What is the same as the kinematic stage.** The scene, goal and solver come from `build_sample` and `compile_sample`,
unchanged; the kinematic task id is recompiled and checked against stage 1. The goal evaluator is `goals.evaluate_goal`
with the same goal JSON, checked to differ only in `episode.duration_s`. The planner is the same: `goal_planner_v1`,
`path_v2`, `plane_section`, the per-sample rake/skew/direction, and no oracle input. The schedule is the same: the full
plan, HOME → operations → planner RETURN_HOME.

**What is live-specific (declared).**
- The C5 live controller limits are 1° and 1 mm per 20 ms step. Phase D measured 11 mm of REORIENT lag at 2°/step;
  cutting speeds are unchanged.
- Edge feedback is measured (the privileged tool-edge pose, as in C5).
- Duration = ceil((1.1 × live plan end + 1 s) / 0.5 s) × 0.5 s.
- History stride follows the round rule.
- After the controller's DONE, the knife holds with cutting disabled until the time limit. Then the Phase D physical
  return-to-home runs and is scored as `home_outcome`, separately.
- Live task ids therefore differ from the kinematic ids.
- Classes: SCENE_INFEASIBLE / IK_FAILURE (live preflight) > NOT_RUN > IK_FAILURE (any task-phase step with
  `ik_status ≠ ok` or a workspace clamp) > GRASP_FAILURE (monitor LOST, or slip > 2 mm / rotation > π/60 rad) >
  CUTTING_FAILURE > TASK_SUCCESS.

### Result

| Sample | Purpose | Kinematic (round 1 stage 2) | **Live class** | Live goal verdict | Live `home_outcome` |
| --- | --- | --- | --- | --- | --- |
| TRIM s1203 | ellipsoid cap, ρ 10° | TASK_SUCCESS (0.993) | **TASK_SUCCESS** | PASS | PASS |
| CUT s1101 | vertical sever, yaw +13° | TASK_SUCCESS | **TASK_SUCCESS** | PASS | PASS |
| Test A s6 | skim along −x, ρ 10° | TASK_SUCCESS (0.967) | **TASK_SUCCESS** | PASS | PASS |
| Test C s1602 (live_001) | three protrusions, tlmpm_v5 K = 3 | TASK_SUCCESS | **SCENE_INFEASIBLE** (refused before launch) | — | — |
| Test C s1602 (live_002) | same, round-1 staged preflight | — | **NOT_RUN** (runtime error in SKIM:p2) | — | — |
| Test C s1602 (live_003) | same, after the surface fix; staged runner default preflight | — | **TASK_SUCCESS** (see [live_003](#test-c-s1602-live_003-supplementary-after-the-surface-fix)) | PASS | PASS |
| Test B s1502 | four slices, tlmpm_v5 K = 4 | TASK_SUCCESS | **TASK_SUCCESS** | PASS | PASS |
| CUT s1102 | 20° oblique sever | TASK_SUCCESS | **TASK_SUCCESS** | PASS | PASS |
| TRIM s1201 | dome, ρ 10°, lowest margin | TASK_SUCCESS (0.936) | **TASK_SUCCESS** | PASS | PASS |
| SLICE-PUSH s1301 | cut-1 shift, plain planner | CUTTING_FAILURE | **CUTTING_FAILURE** | FAIL (`cut_at_declared_planes`) | PASS |
| Test A s2 | −20° oblique, marginal | CUTTING_FAILURE | **CUTTING_FAILURE** | FAIL (`cut_at_declared_planes`) | PASS |

**live_001 totals (9 declared samples):** TASK_SUCCESS 6, CUTTING_FAILURE 2, SCENE_INFEASIBLE 1; GRASP_FAILURE 0,
IK_FAILURE 0, NOT_RUN 0. **live_002 (Test C only):** NOT_RUN 1. The commit message of `fef0e1e` says "7 TASK_SUCCESS";
that is a typo. The evidence file says 6, and so does this table.

Every live verdict that could be scored matches its kinematic class. Both kinematic failures recur live, from the same
cause and with nearly the same numbers. Test D s1701/s1702/s1704 are held (slot overflow; unsatisfiable sequence goal).
The other round-1 seeds were not nominated and not run.

### Goal metrics (live)

| Sample | Pieces / removal | Substrate share | Location (goal statistic) | Other |
| --- | --- | --- | --- | --- |
| TRIM s1203 | p0 0.994 | 1.00000 | — (residual height quality only) | 2 components |
| CUT s1101 | 2 / 2 | — | 1.50 mm (frozen 2 mm statistic, node plane) | edge within ±0.09 mm of the plane on 147 armed steps |
| Test A s6 | p0 0.967 | 0.99998 | — | detached below-plane 2.1e-5 |
| Test B s1502 | 5 / 5 | — | 4.9 mm (divide: quality only, as kinematic 5.4 mm) | packed K = 4, 5 components |
| CUT s1102 | 2 / 2 | — | 0.342 mm (swept-plane actual − ideal, limit 1 mm) | — |
| TRIM s1201 | p0 0.962 (kinematic 0.936) | 0.99992 | — | detached 7.6e-5 |
| SLICE-PUSH s1301 | 3 / 3, push 19.9 mm (others ≤ 0.04 mm) | — | cut 0 1.5 mm, **cut 1 4.0 mm > 2 mm** | live bar shift before cut 1 +1.16 mm, cut-1 label boundary +23.00 vs +24 mm (kinematic: +1.16 mm, +23.00) |
| Test A s2 | 2 / 2 | — | **1.023 mm > 1.00 mm** (kinematic 1.02) | — |

SLICE-PUSH diagnostic: `artifacts/wp11/phaseE/round1_live_001/diagnostics_slice_push_live_001.{py,txt}`. The measured
edge stayed within ±0.09 mm of each declared plane during both cuts. The error is the bar sliding after cut 0, not
tracking.

### Grasp, IK, contacts, skim orientation

- **Grasp** (task phase, GraspMonitor): HOLDING on every step of every run. Max slip 0.64–1.29 µm, max rotation
  0.059–0.113 mrad, against gates of 2 mm / 52.4 mrad.
- **IK:** `ik_status = ok` on every task step (575–1950 per run), with 0 workspace clamps. The declared per-step rate
  limits clamped 20–53 translation steps and 178–344 rotation steps per run; these are reported, not failures.
- **Penetration:** 0 unintended native tool contacts in every run. Finger-pad tensor penetration was at most 6.8–7.3 µm.
- **Measured-path C2 audit** (report only; declared uncut workpiece, nominal grasp):
  - PASS for every sever, divide and push run.
  - The three ρ 10° skims FAIL `blade_collision` in the contact-free PRE_CUT settle: −0.20 mm (TRIM s1203), −0.19 mm
    (Test A s6) and −0.24 mm (TRIM s1201). This is the C5 open issue: measured-edge ringing under the integrating
    delta command, with zero material reaction and no labels while disarmed. C5b's `path_v3` targets it; it was not
    used here.
  - Fingers, handle, D405s, table, board and Franka 1 PASS in every run.
- **τ/ψ/ρ/μ of the measured edge on armed skim steps** (mean / max |·|, degrees):

| Sample | τ | ψ | ρ (declared 10) | μ per step | μ chord | Edge z − (target + 0.2 mm) | Lowest blade point − bar top |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRIM s1203 | 0.001 / 0.005 | 0.002 / 0.017 | 10.000 / 10.017 | −0.14 / 18.5 | −0.13 | −0.18 … +0.12 mm | ≥ −0.13 mm |
| Test A s6 (−x) | 0.001 / 0.005 | 0.002 / 0.010 | 10.000 / 10.016 | −0.12 / 11.1 | −0.20 | −0.14 … +0.15 mm | ≥ −0.10 mm |
| TRIM s1201 | 0.006 / 0.035 | 0.007 / 0.036 | 10.000 / 10.014 | −0.65 / 20.2 | −0.56 | −0.29 … +0.39 mm | ≥ −0.24 mm |

  Per-step μ is dominated by ±0.2 mm vertical ringing over 0.5 mm steps, as in C5. The chord gives the motion
  elevation.
- **Plan lag** (live controller DONE minus the ideal live plan end): +0.06 to +0.44 s. Every plan reached DONE inside
  the episode.
- **home_outcome:** PASS on all 8 run episodes. Home joint error 0.39–0.50 mrad, tool pose versus reset ≤ 0.09 mm /
  0.17 mrad, 0 label changes. The planner's RETURN_HOME already brings the edge back to the HOME pose, so the physical
  return-to-home starts near HOME. It lasts 0.7–1.3 s and corrects the arm's null-space joint drift. Unlike C5, it does
  not return from the workpiece.

### Test C s1602

1. **live_001: SCENE_INFEASIBLE, before any physics.** The runner's default C2 plan preflight models all three
   protrusions as one convex hull for the whole plan. It refused with blade vs bump −8.4 mm at PRE_CUT:p2
   (contact_free). The round file's own geometry rule for scenes with two or more protrusions
   (`stages.stage1_geometry`, `generalization.staged_preflight`) was not carried into the live_001 declaration. That
   was an omission in my declaration, not a property of the scene. The class is kept.
2. **live_002 (declared separately): NOT_RUN.**
   - The staged preflight on the live task passed (`inputs/staged_preflight.json`; blade slack +0.05 mm, same
     thresholds).
   - The Native episode then ran 691 control steps (13.8 s of 20.5 s). It finished SKIM:p0, WITHDRAW:p0 and
     SKIM:p1, and was in the middle of SKIM:p2 with cutting armed.
   - Grasp was HOLDING (slip 0.77 µm) and IK was ok on every step.
   - It then aborted in the live visual-surface update: `isaac_sim_runner._update_surface` →
     `cutting_surface.extract` raised `RuntimeError('ParticleSurface returned no mesh for 11 physical particles')`.
     This is the presentation mesh for an 11-particle fragment, not the material solver or a goal check.
   - No `physics_state.npz` or episode result was written, so the goal cannot be scored.
   - Physics had started, so under the failure policy it was not relaunched.
   - **Test C live is open.** It needs a runtime fix in which a component too small to mesh is skipped or merged in
     the visual surface instead of aborting the episode. That is a change to shared `cutting_surface.py` /
     `isaac_sim_runner.py`, and it is not made here.

### SLICE-PUSH and the ADR 0009 hook

The ADR 0009 wrist-depth relocation could not be applied to a configured plan without substantial new code:
- `policies.vision_policy` refuses any task id other than SLICE-PUSH-01.
- `goal_planner.planner_options` rejects `path_v2_vision`.
- `run_wp10_live.py` refuses `--controller-profile` for `--scene/--goal`.
- The survey was validated only in `blademaster_workcell_v1` with `ruler_knife_v1`.

So s1301 ran with the plain planner, as declared, and failed exactly as predicted in the declaration. What is missing:
1. A goal-planner perception hook that marks the cut-0 / cut-1 settle holds as calibration / survey points and inserts
   the retarget waypoint.
2. Runner wiring of the depth source for configured tasks.
3. A new validation of the `end_face_plus_x` survey in `wp11_workcell_v1` with `ruler_knife_v2_y` and a yawed bar.

The kinematic stage cannot provide that validation, because it has no cameras.

### Runs, commands, timing, disk

All runs used seed 42 and `TMPDIR=/mnt/data/ningan/tmp`, from clean detached worktrees:
- live_001: `/mnt/data/ningan/RoboticCuttingSim-wp11-Elive-1ecfe5e`;
- live_002: `…-Elive2-4b2eab6`;
- analysis and rendering: `…-Eanalysis-*` at `40bef78` → `fc65b59` → `0421050`, each clean.

    .venv/bin/python scripts/run_generalization_live.py queue --root artifacts/wp11/phaseE/round1_live_001 \
        --queue-log artifacts/wp11/logs/phaseE_live_queue_001.log                 # 06:55–07:48 KST
    .venv/bin/python scripts/run_generalization_live.py --live-config configs/wp11/generalization_round1_live_002.json \
        queue --root artifacts/wp11/phaseE/round1_live_002 --queue-log artifacts/wp11/logs/phaseE_live002_queue_001.log
    # per sample (exact argv in <sample>/attempt_001.json):
    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 .deps/isaacsim-venv/bin/python scripts/run_wp10_live.py --runtime isaac_sim \
        --scene <sample>/inputs/scene.json --goal <sample>/inputs/goal.json \
        --planner-options '{"max_rotation_deg": 1.0, "max_translation_m": 0.001}' --controller scripted \
        --edge-feedback measured --hold-until-time-limit --return-home --record-cameras --overview-camera \
        --record-every {5|10} --oracle-stride <stride> --steps <steps> --gpu <g> --seed 42 --out <sample>/run_001
    .venv/bin/python scripts/run_generalization_live.py analyze --sample-dir <sample> --run run_001 --out <sample>/analysis_00N
    .venv/bin/python scripts/run_generalization_live.py summarize --root artifacts/wp11/phaseE/round1_live_001 \
        --out docs/evidence/wp11/phaseE_round1_live_001.json \
        --supplementary configs/wp11/generalization_round1_live_002.json:artifacts/wp11/phaseE/round1_live_002

| Sample | Device | Live steps (task + home) | Episode wall s | Process wall s | Run dir |
| --- | --- | --- | --- | --- | --- |
| TRIM s1203 | cuda:0 | 575 + 66 | 443 | 481 | 0.59 GB |
| CUT s1101 | cuda:0 | 725 + 43 | 529 | 575 | 0.65 GB |
| Test A s6 | cuda:0 | 650 + 57 | 518 | 582 | 0.80 GB |
| Test C s1602 live_001 | cuda:0 | 0 (refused) | — | 30 | 0.04 MB |
| Test B s1502 | cuda:1 | 1950 + 56 | 1506 | 1577 | 1.55 GB |
| CUT s1102 | cuda:0 | 725 + 42 | 442 | 480 | 0.65 GB |
| TRIM s1201 | cuda:0 | 575 + 62 | 345 | 390 | 0.54 GB |
| SLICE-PUSH s1301 | cuda:0 | 1450 + 38 | 838 | 906 | 1.08 GB |
| Test A s2 | cuda:0 | 800 + 36 | 686 | 750 | 0.60 GB |
| Test C s1602 live_002 | cuda:0 | 691 (aborted) | — | 660 | 0.42 GB |

- **Device policy:** at most 2 of these runs at once. A GPU was used only with ≤ 2 other active jobs and ≥ 12 GiB free.
  GPU 1 qualified once, for Test B. Non-RCS processes were never touched.
- **Disk:** 6.5 GB in total under `artifacts/wp11/phaseE/round1_live_00{1,2}`. `/mnt/data` went from 22–26 GB free to
  17 GB (other agents also write), so the 15 GB floor was never reached.
- **Analyzer correction:** `analysis_001` of TRIM s1203 (tool at `1ecfe5e`) wrongly counted the planner RETURN_HOME
  phase as return-home rows. It is kept; `analysis_002` (`40bef78`) is the one used. Its class is unchanged.

### Media

Rendered with `scripts/render_demo_episode.py`, the C5 renderer. For packed K > 2 episodes I used
`scripts/render_packed_episode.py`, which swaps only the particle-panel component cache for
`mechanical_components_packed`; the stock renderer fails on K = 4.

| Name | Files | Source |
| --- | --- | --- |
| Test A s6 (−x skim) | [demo.gif](../media/wp11/test_a_s6/demo.gif) (2.4 MB), [demo.mp4](../media/wp11/test_a_s6/demo.mp4), [poster.png](../media/wp11/test_a_s6/poster.png) | `round1_live_001/E1_TEST_A_s6/run_001`, render `render_E1_TEST_A_s6_001` |
| Test B s1502 (four slices) | [demo.gif](../media/wp11/test_b_s1502/demo.gif) (3.3 MB), [demo.mp4](../media/wp11/test_b_s1502/demo.mp4), [poster.png](../media/wp11/test_b_s1502/poster.png) | `round1_live_001/E1_TEST_B_s1502/run_001`, render `render_E1_TEST_B_s1502_004` |

- In the Test B particle panel, three pieces are coloured and two are drawn grey. The renderer has three colour slots;
  this is presentation only, and the goal counts 5 substantial pieces.
- Test C has no media, because no live episode finished.
- SLICE-PUSH has no media, because it did not reach TASK_SUCCESS (as declared).
- Earlier renders `render_E1_TEST_B_s1502_001/002` (stock renderer IndexError, no output) and `_003` (title
  truncated) are kept.

### Open items

1. Fix the visual-surface abort on tiny fragments (`cutting_surface.extract`), then run Test C live under a new
   declaration.
2. Carry the staged multi-protrusion geometry into the runner's default plan preflight.
3. Build the configured-plan ADR 0009 perception hook (SLICE-PUSH).
4. Rerun the skims with `path_v3` against the PRE_CUT blade-dip audit.
5. Add repeated seeds, needed for any live success rate.

### Test C s1602 live_003 (supplementary, after the surface fix)

Declaration: [`configs/wp11/generalization_round1_live_003.json`](../../configs/wp11/generalization_round1_live_003.json)
(`002c95e`, committed and pushed before the run). Evidence:
[`docs/evidence/wp11/phaseE_round1_live_003.json`](../evidence/wp11/phaseE_round1_live_003.json). The declaration
inherits `generalization_round1_live.json` with its sha256 pinned: same scene, same goal JSON, same duration rule,
controller, evaluator, classes and failure policy. Only the code commit and the preflight wording differ. The stated
reason is that both earlier failures were infrastructure: live_001 failed on the preflight geometry and live_002 on
the presentation mesh. **live_001 stays SCENE_INFEASIBLE and live_002 stays NOT_RUN; both records are unchanged.**
live_003 is a separate, labelled result. No threshold was changed and nothing was rescored.

**Code changes (before the declaration).**
- `d3f94e9`, `cutting_surface.CuttingSurface.extract`: a current mechanical component for which ParticleSurface
  returns no triangles is left out of the presentation mesh only. It is recorded in the returned diagnostics as
  `unmeshable_skipped` (fragments, particles, component rows), a key that appears only when something was skipped.
  - The extractor raises only if no component meshes at all.
  - Physics, cut labels, goal evaluation and recording are unchanged. The physical component count is still
    reported.
  - CPU tests in `tests/test_cutting_surface.py` (`current_volume_v1`, the live profile): an 11-particle sparse
    fragment plus a single loose particle are skipped and recorded, and the meshable components are unchanged. For
    all-meshable input, the output (vertices, faces and the diagnostics dict) is byte-identical to the pre-fix
    sha256 for both profiles.
  - **Review R1 erratum (after the run).** `d3f94e9` changed the extractor's default for every caller, including
    the frozen v1 TRIM oracle path, where main raised. The skip is now a versioned option:
    `CuttingSurface/ComponentSurface(unmeshable="raise")` is the default and raises exactly as on main, and
    `unmeshable="skip_presentation_v1"` is the `d3f94e9` behaviour. The Isaac Sim runner selects it through
    `IsaacSimCuttingConfig.surface_unmeshable`. Its default (`None`) resolves to `skip_presentation_v1` only for
    configured (WP-11) tasks (`configured_task` set), whose goal evaluators use the surface as quality evidence
    only. Legacy D0, the registered CUT/TRIM tasks and the final_v4 tasks keep `raise`. The resolved value is in
    the runtime `setup.json` (`config.surface_unmeshable`, plus a `surface_unmeshable` receipt in skip mode).
  - Skip mode is now recorded. Every extraction that skips something is written with its reset index, step
    index and time to `runtime/surface_unmeshable_NN.jsonl`. Every step's `privileged_diagnostics.surface_unmeshable`
    carries the cumulative counts and the skips since the previous step. `backend.surface_unmeshable_summary()` is
    the episode summary, which evaluation results carry as `surface_unmeshable` (the `live_env.py` hook is a
    separate review-R3 change). A skipped component that holds a lowest-layer anchor particle is a substrate
    component; then `refresh_surface_for_evaluation()` raises `SurfaceNotScorable`, so surface-dependent metrics
    are NOT_RUN (configured goals) or unscored (v1 oracle), never PASS.
  - **Reproducing live_003 at the new HEAD:** the same declaration and runner argv
    (`run_wp10_live.py --runtime isaac_sim --scene … --goal …`) compile a configured task, so the runner resolves
    `surface_unmeshable="skip_presentation_v1"` without any new flag. To get main's behaviour instead (which aborts
    as live_002 did if a fragment cannot be meshed), pass `surface_unmeshable="raise"` through `open_cutting_env`
    backend options. live_003 evidence is unchanged; it ran at `002c95e`, where the skip was unconditional.
- `2b4f0a7`, runner default plan preflight (`scripts/wp11_motion_preflight.py plan`, which `run_wp10_live.py` calls
  before any Isaac launch):
  - Scenes with two or more protrusions use `generalization.staged_preflight`, the round-1 helper, with an additive
    `until_phase`. The receipt gains `preflight_geometry = staged_multi_protrusion_v1`.
  - Scenes with zero or one protrusion keep exactly the previous calls and receipt keys.
    `tests/test_wp11_default_preflight_staged.py` covers this, plus a real CLI run on the Test C live task.
  - The analyzer also gained a report-only staged measured-path check.
  - `isaac_sim_runner.py`, `policies.py` and `goal_planner.py` were not touched.
- `74aafb0` (after the run, presentation only): `render_packed_episode.py --wide-piece-palette`.

**Run.** The run used the runner's own default preflight (not `--skip-plan-preflight`). It returned PASS, staged,
with binding `blade_collision` slack +0.049 mm at PRE_CUT:p1, the same as the live_002 queue-side gate. Then one
Native episode ran on cuda:0. Of 1025 task steps plus 63 home steps (1088 recorded), the plan reached DONE at 18.22 s
against an ideal 17.48 s (+0.74 s lag). The episode then held to the 20.5 s limit and returned home. It passed step
691, where live_002 had aborted, with no runtime error. Wall times: episode 947 s, process 1020 s. The run directory
is 1.63 GB, and `/mnt/data` went from 26.0 to 22.8 GB free.

| Quantity | live_003 (live Native) | Kinematic stage 2 (same seed) |
| --- | --- | --- |
| Class / goal verdict | **TASK_SUCCESS** / PASS (integrity 7/7, outcome 6/6) | TASK_SUCCESS |
| Removed mass p0 / p1 / p2 (≥ 0.9) | 0.962 / 0.997 / 0.972 | 0.960 / 0.997 / 0.946 |
| Substrate share (≥ 0.99) | 0.9999999999988 (single substrate component) | 0.9999999999988 |
| Detached below-plane mass (≤ 0.01) | 0.0 | 0.0 |
| Mechanical components (final) | 4: 107054 / 2040 / 737 / 455 particles | 4 |
| Residual substrate height p0 / p1 / p2 (quality) | 0.094 / 0.180 / 0.187 mm | 0.078 / 0.068 / 0.135 mm |
| Grasp (task phase) | HOLDING on 1025/1025 steps; max slip 0.83 µm, max rotation 0.095 mrad | — |
| IK | `ok` on 1025/1025; 0 workspace clamps; 35 translation / 178 rotation rate-limit clamps | — |
| Native tool contacts | 0 unintended; finger-pad tensor penetration ≤ 7.2 µm | — |
| `home_outcome` | PASS: 1.26 s, joint error ≤ 0.50 mrad, tool 0.088 mm / 0.16 mrad, 0 label changes, home path C2 PASS | — |

**Chip independence.** This was measured with a read-only oracle diagnostic,
`round1_live_003/E1_TEST_C_s1602/diagnostics_001/chip_independence.{py,json}`. It maps each protrusion's goal-owned
mass to the current packed mechanical components at all 219 history frames. Its final removed fractions equal the
evaluator's to all digits.
- All three chips are independent at the end. Each removed chip is a single component holding 100 % of that
  protrusion's removed mass, with 0 below-plane particles, distinct from the substrate and from the other chips.
- They separated one at a time, each during its own skim: p0 at 5.60 s (SKIM:p0), p1 at 10.40 s (SKIM:p1) and p2
  at 14.30 s (SKIM:p2).
- By 21.76 s all three chips had left the bar, ending 85–122 mm below its top beyond the +x end.
  *Correction (2026-10-30, main, read-only recheck of the final frame):* this statement is based on chip centroids
  and is too strong. The final mechanical components are 455 / 737 / 2040 / 107054 particles. 405 of the p1 chip's
  2040 particles (component 2) lie on the bar top at world x ≈ 0.031–0.046 m, 3–13 mm above the substrate top, while
  the rest of that chip is on the table beyond the +x end. They are in the chip component, not the substrate, so the
  separation verdict and TASK_SUCCESS are unchanged. The README scene table shows this in the Test C end frame
  (`docs/media/wp11/scenes/test_c_s1602_end.png`).
- No recorded frame has a component under 100 particles (`diagnostics_001/component_sizes.json`). The runner does
  not persist `surface_info`, so whether the extractor skipped a transient fragment between recorded frames is
  **not measured**. At the final, evaluated frame nothing was skipped. (After review R1 the runner records every
  skip in skip mode, as described under the code changes above; live_003 predates that.)

**Skim orientation of the measured edge on armed steps** (mean / max |·|, degrees; ρ declared 10):

| Skim | Armed steps | τ | ψ | ρ | μ per step | μ chord | Edge z − (target + 0.2 mm) | Lowest blade point − bar top |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p0 | 83 (4.26–5.90 s) | 0.001 / 0.011 | 0.003 / 0.025 | 10.000 / 10.015 | −0.26 / 18.9 | −0.17 | −0.16 … +0.12 mm | ≥ −0.11 mm |
| p1 | 87 (8.78–10.50 s) | 0.002 / 0.012 | 0.003 / 0.023 | 10.000 / 10.017 | −0.30 / 22.9 | −0.16 | −0.23 … +0.17 mm | ≥ −0.18 mm |
| p2 | 78 (12.86–14.40 s) | 0.003 / 0.013 | 0.004 / 0.020 | 10.001 / 10.016 | −0.38 / 21.6 | −0.20 | −0.26 … +0.19 mm | ≥ −0.21 mm |

As in live_001, per-step μ is dominated by about ±0.2 mm of vertical ringing over 0.5 mm steps; the chord gives the
motion elevation. The peak material reaction on the blade was 0.04, 0.10 and 0.15 N.

**Measured-path C2 clearance (report only; nominal grasp).**
- Declared uncut geometry (the live_001 check): FAIL `blade_collision` −8.67 mm against `static_workpiece_bump`. This
  is the single-hull artefact: the convex hull over all three protrusions stays present after they are removed.
- Staged geometry (round-1 rule, chunks without the protrusions already removed): FAIL `blade_collision` −0.23 mm at
  PRE_CUT:p2 against `static_workpiece_base`.
  - Chunks with removed sets {}, {p0} and {p0, p1} FAIL `blade_collision`. The final chunk, {p0, p1, p2} (planner
    RETURN_HOME and hold), PASSes.
  - This is the same contact-free PRE_CUT dip as the three live_001 ρ 10° skims (−0.19 to −0.24 mm; C5 open issue,
    `path_v3` not used).
- Fingers, handle, D405s, table, board and Franka 1 PASS.

**Surface quality (reported, not scored).** 0.66 % of the substrate heightfield cells have no surface, and
`maximum_gouge_depth_m` reads 0.022 (the full bar height, so those cells have no substrate surface). The close-up RGB
shows a round patch on the bar top near p1's footprint. The three live_001 skims had 0 % missing and ≤ 1.9 mm. This
does not come from the unmeshable skip: at the final frame there are 4 components and none was skipped.

*Diagnosis (review R1; read-only, final recorded frame of `run_001/episode_000/physics_state.npz`).* The 22 mm
maximum gouge and the 0.66 % missing are a heightfield-registration artefact, not material loss:
- The bar slid during the episode. The substrate's median displacement is +0.767 mm in x (mean +0.773 mm), and its
  yaw change is 0.0067°. The frozen heightfield (`scene_validation.mesh_heightfield_metrics`) is fixed to the
  initial workpiece pose. Its 152 × 32 = 4864 one-millimetre cells therefore no longer cover the bar's current
  footprint.
- Exactly 32 of the 4864 cells (32/4864 = 0.658 %) contain no substrate particle centre. They form the whole −x end
  column (ix = 0, all 32 iy): the bar's −x end moved from −75.5 to −74.83 mm. The metric charges a missing cell the
  full target height (`deficit = height − 0` for invalid cells), so each one reads 22 mm.
- Behind the artefact there is a smaller real dent. After the median displacement is removed, 400 of the 14638
  top-layer substrate particles (reference z ≥ 19.5 mm) moved by more than 1 mm. All of them lie in the three skim
  footprints (reference x −70…−25, −25…15 and 15…60 mm: 74, 223 and 103 particles). They were pressed down by
  1.42–1.47 mm on average (mean relative z per footprint), with a maximum of 2.01 / 2.63 / 3.19 mm.
- Two one-cell-deep spots stand out in the current-pose cell grid. At the p1 exit (reference x ≈ +11.5 mm, cells
  (iy, ix) = (14, 88) and (17, 88)) the top substrate particle centre is at 18.47 / 18.48 mm, 3.0 mm below the
  21.5 mm top-layer centre. At the p2 exit (x ≈ +54.5 mm, cells (15, 131) and (16, 131)) it is at 19.41 mm, 2.1 mm
  below. Only 6 covered cells have a top particle centre below 20 mm.
- No below-plane particle detached (0 below-plane particles outside the substrate component; mass fraction 0.0).
  Each chip holds 0 below-plane particles (`chip_independence.json`). TASK_SUCCESS therefore stands under the
  declared, separation-based goal. The dent is a quality observation; no threshold or score changed.
- Commands (read-only, main-repo `.venv`): the reviewer's script
  `/mnt/data/ningan/tmp/claude-1004/-home-ningan-RoboticCuttingSim/e538d80f-78df-4b03-8e24-f5e0ea11b086/scratchpad/surface_check.py`,
  run as `.venv/bin/python -I <script>` (2.2 s); it is not committed. A heightfield registered to the current
  substrate pose (a new, quality-only metric) was **NOT_RUN**; the frozen v1 metric is unchanged.

**Media** (rendered with `render_packed_episode.py --wide-piece-palette` at `74aafb0`, clean worktree):
[demo.gif](../media/wp11/test_c_s1602/demo.gif) (3.6 MB), demo.mp4 (local only; `*.mp4` is gitignored) and
[poster.png](../media/wp11/test_c_s1602/poster.png), from `render_E1_TEST_C_s1602_002`.
- The stock palette has three hues and treats components under 1 % of the particles as grey. Here p0 (0.4 %) and p2
  (0.7 %) would both be grey.
- The opt-in wide palette uses five hues and a 0.2 % piece threshold, with the largest component (substrate) pinned
  to the first hue. The substrate stays blue (major 4, minor 0). Only the substrate hue is pinned; chips are ordered
  by position along the bar, so a chip keeps its hue only while chips are removed left to right. That order held in
  this run, but it was not checked frame by frame.
- The earlier render `render_E1_TEST_C_s1602_001`, at `002c95e` and without the pinning (the substrate changed hue as
  chips split off), is kept.

**Commands** (seed 42, `TMPDIR=/mnt/data/ningan/tmp`; queue, analyze and summarize in the run worktree
`/mnt/data/ningan/RoboticCuttingSim-wp11-Elive3-002c95e`, render in `…-Eanalysis3-74aafb0`, both clean; the two
oracle diagnostics with the main-repo `.venv`, read-only):

    .venv/bin/python scripts/run_generalization_live.py --live-config configs/wp11/generalization_round1_live_003.json \
        queue --root artifacts/wp11/phaseE/round1_live_003 --queue-log artifacts/wp11/logs/phaseE_live003_queue_001.log
    # exact runner argv in round1_live_003/E1_TEST_C_s1602/attempt_001.json (as live_001, --record-every 5 --oracle-stride 26 --steps 1025 --gpu 0)
    .venv/bin/python scripts/run_generalization_live.py --live-config configs/wp11/generalization_round1_live_003.json \
        analyze --sample-dir artifacts/wp11/phaseE/round1_live_003/E1_TEST_C_s1602 --run run_001 \
        --out artifacts/wp11/phaseE/round1_live_003/E1_TEST_C_s1602/analysis_001          # 14 s, at 002c95e
    .venv/bin/python scripts/run_generalization_live.py --live-config configs/wp11/generalization_round1_live_003.json \
        summarize --root artifacts/wp11/phaseE/round1_live_003 --out docs/evidence/wp11/phaseE_round1_live_003.json
    .venv/bin/python scripts/render_packed_episode.py artifacts/wp11/phaseE/round1_live_003/E1_TEST_C_s1602/run_001/episode_000 \
        --out artifacts/wp11/phaseE/round1_live_003/render_E1_TEST_C_s1602_002 --title … --label … --wide-piece-palette   # 91 s

**live_003 totals (Test C only):** TASK_SUCCESS 1. Open items 1 (surface abort) and 2 (staged default preflight) from
the list above are addressed by `d3f94e9` and `2b4f0a7`. Remaining:
- the PRE_CUT blade dip (item 4);
- the Test C missing-surface quality anomaly (diagnosed in review R1 as a registration artefact over a smaller
  real dent; a current-pose heightfield metric is not implemented);
- persisting `surface_info` skip counts in the recording (done in review R1 for skip mode; the evaluation-result
  summary depends on the R3 `live_env.py` hook);
- repeated seeds.

### SLICE-PUSH live_004 (supplementary controller variant, ADR 0016 wrist-depth relocation)

**Declarations.**
- Stage: [`configs/wp11/generalization_round1_live_004.json`](../../configs/wp11/generalization_round1_live_004.json)
  (`c29f02a`, committed and pushed before any live_004 run).
- Design: [ADR 0016](../adr/0016-configured-perception-relocation.md) (`0e96f07`).
- Evidence: [`docs/evidence/wp11/phaseE_round1_live_004.json`](../evidence/wp11/phaseE_round1_live_004.json).

**What changes, and what does not.**
- Inherited from live_001: the builder, the goal JSON (only the duration differs), `goals.evaluate_goal` with every
  threshold, the classes, the measured edge feedback and the default C2 preflight.
- The only new element is `planner_options.perception = wrist_depth_relocate_v1`, plus the longer plan it gives under
  the duration rule.
- The round-1 kinematic CUTTING_FAILUREs (5/5) and the live_001 s1301 CUTTING_FAILURE stay as recorded above. live_004
  is a separate, labelled controller-variant result, and nothing is rescored.

**Why the ADR 0009 poses were not reused.** On the live_001 s1301 recording, the v5 wrist D405_0 sits about 113 mm
above, 174 mm −y and 45 mm −x of the edge, looking along local +y. From the cut-start settles:
- the +x end face is out of the image (row −14);
- the −x end belongs to the left piece (−1.23 mm);
- the kerf face reads the blade-deformed layer (+0.53 mm error).

ADR 0016 therefore adds a planned survey pose: the edge in the cut orientation at local (L/2 + 10 mm, 0, top + 8 mm).
It is visited once before cut 0 (calibration) and once after cut 0. The shift is the difference of the +x end-face
positions measured with the unchanged `wrist_depth_survey.end_face_x`. It is applied to cut 1 and to the push; HOME and
the survey poses are not shifted.

**Inputs.** The hook reads only deployable inputs:
- wrist depth with K and T_world_camera;
- the nominal edge from joint state, used for blade exclusion;
- the scene config.

The measured tool-edge pose stays the privileged tracking feedback, as in live_001, and the hook never reads it. The
oracle values below are report-only.

| Seed | Measured shift (wrist depth) | Actual shift (oracle, report-only) | Cut 0 / cut 1 label boundary | Frozen statistic | Pieces | Push / others | Grasp (slip, rotation) | `home_outcome` | **Class** | Wall (episode / process) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1301 | **+1.158 mm** (40 / 39 lines, spread 9 / 29 µm) | +1.149 mm, cut-1 layers (error **+0.009 mm**); +1.150 mm, end layer | −28.00 / **+24.00 mm** (declared −28 / +24) | 1.5 mm ≤ 2 mm | 3 / 3 | 16.7 mm (≥ 10) / 0.04 mm (≤ 2) | HOLDING, 0.95 µm, 0.095 mrad | PASS | **TASK_SUCCESS** | 1257 s / 1320 s |
| s1302 | **+0.876 mm** (47 / 47 lines, spread 11 / 20 µm) | +1.160 mm, cut-1 layers (error **−0.283 mm**); +0.884 mm, end layer (error −0.007 mm) | −24.00 / **+28.00 mm** (declared −24 / +28) | 1.5 mm ≤ 2 mm | 3 / 3 | 18.1 mm (≥ 10) / 0.04 mm (≤ 2) | HOLDING, 0.94 µm, 0.103 mrad | PASS | **TASK_SUCCESS** | 1742 s / 1830 s |
| s1303 | **+1.177 mm** (43 / 43 lines, spread 8 / 28 µm) | +1.167 mm, cut-1 layers (error **+0.010 mm**); +1.170 mm, end layer | −28.00 / **+36.00 mm** (declared −28 / +36) | 1.5 mm ≤ 2 mm | 3 / 3 | 14.0 mm (≥ 10) / 0.01 mm (≤ 2) | HOLDING, 1.33 µm, 0.140 mrad | PASS | **TASK_SUCCESS** | 1418 s / 1500 s |

**live_004 totals (3 declared seeds): TASK_SUCCESS 3**; CUTTING_FAILURE, GRASP_FAILURE, IK_FAILURE, SCENE_INFEASIBLE and NOT_RUN 0.
For the same seeds, the round-1 kinematic results (plain planner) are 3 CUTTING_FAILURE and live_001 s1301 is CUTTING_FAILURE.
Those records are unchanged.

**s1301 details** (`artifacts/wp11/phaseE/round1_live_004/E1_SLICE_PUSH_s1301/`: `run_001`, `analysis_001`,
`perception_001.json`):
- **Measurement.** The calibration frame was taken at t = 4.70 s and the survey frame at t = 17.74 s. Each frame
  timestamp equals its observation timestamp.
- **Applied shift.** The world shift was (+1.151, −0.127, 0) mm, the bar's local +x at a yaw of −6.3°. It was applied to
  the 13 waypoints from APPROACH:cut1 to RETRACT:move1.
- **Cut 1 position.** The measured edge was +1.156 mm (mean) from the declared world plane during cut 1, as intended:
  the relocation itself. In the reference frame cut 1 landed on the declared plane. The material had shifted +1.165 mm
  before cut 1; in live_001 the shift was +1.16 mm and the cut landed at +23.00.
- **Plan end.** The live plan ended at 35.06 s against 34.44 s ideal, a lag of 0.62 s.
- **IK.** `ik_status` was ok on all 1950 task steps, with 0 workspace clamps.
- **Contacts and clearance.** There were 0 unintended contacts, and the measured-path C2 audit PASSed.
- **Run.** Device cuda:1, run directory 1.43 GB, git `c29f02a` (clean detached worktree
  `/mnt/data/ningan/RoboticCuttingSim-wp11-Elive4-c29f02a`), seed 42.

**s1302 and s1303: first launch, not a retry.**
- **Why they were not in the first queue.** queue_001 did not launch them: free space on `/mnt/data` fell from 23.0 GB
  to 15.4 GB while s1301 ran (s1301 wrote 1.43 GB, other agents the rest), and the declared policy refused the next
  launch (15.4 GB minus the 1.6 GB estimate is below the 15 GB floor). That first evidence commit `fa0eabb` recorded
  them as NOT_RUN.
- **The launch.** Main then freed `/mnt/data` (26.8 GB free). Both seeds were launched for the first time with the
  recorded command, from the same clean worktree (`c29f02a`), under the same declaration and the same GPU/disk policy.
  Neither seed had started before, so this is a first launch, not a retry, as the NOTE line at the top of
  `artifacts/wp11/logs/phaseE_live004_queue_002.log` records:

      cd /mnt/data/ningan/RoboticCuttingSim-wp11-Elive4-c29f02a && TMPDIR=/mnt/data/ningan/tmp .venv/bin/python \
          scripts/run_generalization_live.py --live-config configs/wp11/generalization_round1_live_004.json queue \
          --root artifacts/wp11/phaseE/round1_live_004 --queue-log artifacts/wp11/logs/phaseE_live004_queue_002.log \
          --only E1_SLICE_PUSH_s1302 E1_SLICE_PUSH_s1303

- **s1302** ran 09:07:51–09:38:21 on cuda:0. The GPU had 1 other active job, and disk went from 28.1 to 24.4 GB.
  - The world shift was (+0.833, +0.273, 0) mm at a yaw of +18.1°. The measured edge during cut 1 was +0.875 mm (mean)
    from the declared world plane.
  - `ik_status` was ok on all 1975 task steps. The plan lagged 0.44 s, and the run directory is 1.68 GB.
  - The +x piece did not move rigidly here. The wrist estimate matches the end layer it measures, but the layers at cut
    1 had moved 0.28 mm more.
  - The relocation still put cut 1 on the declared reference plane (the label boundary is exactly +28.00 mm), because
    a 0.28 mm residual is inside the half-layer (0.5 mm) margin of the 1 mm lattice.
  - This is the first live evidence of a limit of the ADR 0016 rigid-piece assumption. A larger non-rigid slide would
    not be corrected fully.
- **s1303** waited 37 s for a GPU: both GPUs had 3 other active jobs, more than the ≤ 2 the policy allows. It then ran
  09:39:02–10:04:02 on cuda:0, with 2 other active jobs; disk went from 24.4 to 21.5 GB.
  - The world shift was (+1.141, −0.289, 0) mm at a yaw of −14.2°. The measured edge during cut 1 was +1.176 mm (mean)
    from the declared world plane.
  - `ik_status` was ok on all 2000 task steps. The plan lagged 1.04 s, and the run directory is 1.60 GB.
- **Both seeds:**
  - 0 workspace clamps and 0 unintended contacts;
  - the measured-path C2 audit PASSed;
  - the runner's plan preflight PASSed before each launch;
  - the recorded task id equals the CPU recompile;
  - analysis (`analysis_001`, `perception_001.json`) used the same clean worktree.

**Pre-run checks.**
- **CPU C2 preflight.** The full hooked live plans PASSed for all three seeds before the declaration. The runner
  repeated the check for s1301 before launch, and it PASSed.
- **CPU tests.**
  - The new `tests/test_perception_relocate.py` has 20 tests. They cover:
    - the hook is inert when not requested, and the live_001 task id is reproduced;
    - with the hook, the plan is the default plan plus the survey waypoints;
    - eligibility errors;
    - synthetic-depth shift recovery within 0.1 mm, including a +18° yaw;
    - invalid, stale or missing frames apply no shift;
    - runner wiring.
  - The live_004 declaration test passes.
  - The existing pin suites pass unchanged (`.venv/bin/python -m pytest` exit 0, 305 passed):
    `test_wrist_depth_survey`, `test_controller_path_v2`, `test_scene_goal_config`, `test_generalization_round1*`,
    `test_wp11_phase_d`, `test_wp11_phase_c5b`, `test_blademaster_demo*`, `test_policy_ready_env`, and
    `test_wp11_default_preflight_staged`.

**Commands.**

    .venv/bin/python scripts/run_generalization_live.py --live-config configs/wp11/generalization_round1_live_004.json \
        queue --root artifacts/wp11/phaseE/round1_live_004 --queue-log artifacts/wp11/logs/phaseE_live004_queue_001.log
    # s1302 / s1303: the queue_002 command above (first launch, --steps 1975 / 2000, --oracle-stride 50, --gpu 0)
    # runner argv (attempt_001.json): run_wp10_live.py --runtime isaac_sim --scene/--goal <inputs> --planner-options
    #   '{"max_rotation_deg": 1.0, "max_translation_m": 0.001, "perception": "wrist_depth_relocate_v1"}' --controller scripted
    #   --edge-feedback measured --hold-until-time-limit --return-home --record-cameras --overview-camera --record-every 10
    #   --oracle-stride 49 --steps 1950 --gpu 1 --seed 42
    .venv/bin/python scripts/run_generalization_live.py --live-config configs/wp11/generalization_round1_live_004.json \
        analyze --sample-dir artifacts/wp11/phaseE/round1_live_004/E1_SLICE_PUSH_s1301 --run run_001 \
        --out artifacts/wp11/phaseE/round1_live_004/E1_SLICE_PUSH_s1301/analysis_001
    .venv/bin/python scripts/analyze_perception_relocate.py --sample-dir artifacts/wp11/phaseE/round1_live_004/E1_SLICE_PUSH_s1301 \
        --planner-options '{"max_rotation_deg": 1.0, "max_translation_m": 0.001, "perception": "wrist_depth_relocate_v1"}' \
        --out artifacts/wp11/phaseE/round1_live_004/E1_SLICE_PUSH_s1301/perception_001.json

**Media.** These are from s1301, the first TASK_SUCCESS (no render of s1302/s1303 was made). They were rendered with `scripts/render_demo_episode.py`
(K = 2), and the render directory is `round1_live_004/render_E1_SLICE_PUSH_s1301_001`:
- [demo.gif](../media/wp11/slice_push_relocate/demo.gif) (3.5 MB);
- [demo.mp4](../media/wp11/slice_push_relocate/demo.mp4);
- [poster.png](../media/wp11/slice_push_relocate/poster.png).

The temporary frames were rendered on the root disk to keep `/mnt/data` usage minimal.

**Scope.**
- These are three live seeds, one run each, so there are no repeat statistics.
- The relocation is validated live at yaws of −6.3°, +18.1° and −14.2°, and bar lengths of 156, 150 and 168 mm.
- The hook assumes the outer piece moves rigidly. s1302 shows a 0.28 mm non-rigid residual at the cut, inside the
  layer margin.
- The material is provisional and uncalibrated.
