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
