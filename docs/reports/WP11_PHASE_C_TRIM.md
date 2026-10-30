# WP-11 Phase C — near-horizontal TRIM (C2 preflight, C4 material, C5 / C5b live)

Concise report of the Phase C runs. The evidence files are the record; this page links them, summarizes them and
adds a disclosure that no earlier report made (the particle residual, below). Material parameters are provisional
and uncalibrated. Every run used seed 42 and is a single run unless stated otherwise. No result, threshold or record
was changed for this report.

| Step | Evidence | What it is | Result |
| --- | --- | --- | --- |
| C2 | [`phaseC2_preflight_001.json`](../evidence/wp11/phaseC2_preflight_001.json) | Geometry-only feasibility preflight | ρ = 10° is the smallest passing rake. Configuration chosen in [ADR 0013](../adr/0013-ruler-knife-v2-and-trim-orientation-metrics.md) |
| C4 | [`phaseC4_trim_material_001.json`](../evidence/wp11/phaseC4_trim_material_001.json) | Kinematic law-8 material runs, ρ = 0–35° | The double bevel separates at ρ = 5–15° and fails at 0°. The chisel fails at ρ ≥ 5° |
| C5 | [`phaseC5_live_trim_001.json`](../evidence/wp11/phaseC5_live_trim_001.json) | Live Native TRIM, ρ = 10° | Goal PASS (97.89 % removed), home PASS. Measured-path clearance FAIL (PRE_CUT dip) |
| C5b | [`phaseC5b_live_trim_clear_001.json`](../evidence/wp11/phaseC5b_live_trim_clear_001.json) | Live Native TRIM plus chip clear | Goal PASS (97.89 % removed, chip pushed +109 mm off the bar), PRE_CUT clearance PASS |

## C2 — geometric preflight

- Commit `20101c1`. Command: `scripts/preflight_wp11_orientations.py --out artifacts/wp11/phaseC2/c2_preflight_003
  --workers 16 …`. Wall time 262 s.
- Scope: Newton IK plus MuJoCo signed distances on the Franka colliders, the finger hulls, the v2 tool hulls, the
  D405 boxes and the workcell. No dynamics, material, grasp or rendering.
- Below ρ = 12.5° the double bevel's lowest point drops below the edge: by 0.75 mm at 0°, 0.45 mm at 5° and
  0.15 mm at 10°. With the frozen 0.2 mm edge clearance, ρ = 0° and 5° gouge the substrate. **10° is the
  smallest passing rake.**
- The shared configuration for all tasks is `ruler_knife_v2_y`, edge line +y, workpiece raised +100 mm and
  TRIM ρ = 10°. It is recorded in ADR 0013.

## C4 — TRIM material (kinematic, law 8)

- Kinematic ideal tracking, 1 mm particles, 0.1 ms material dt, 4 mm grid. No robot, IK, grasp or camera.
- Double bevel (`ruler_knife_v2_y`): separation PASS at ρ = 5° and 10° (also 12.5° and 15°). FAIL at 0°: 70.3 %
  removed (default controller) and 87.0 % (exact 0.2 mm clearance). At 0° the lower bevel face presses the bar top,
  so the lowest above-plane layer receives the substrate-side digit.
- Chisel (`ruler_knife_v2_y_chisel`): FAIL at ρ ≥ 5° under law 8. The failures are 1–3 above-plane code-0
  particles thrown ahead of the edge (direct code-0 bridges).
- Precision is reported separately and is not part of the outcome. The p95 plane error is 0.51–0.57 mm in every run
  (limit 0.5 mm), so no run passes the full frozen precision score. The frozen `maximum_gouge_depth_m` reads
  16–20 mm in several runs, including the passing ones. It comes from individual surface columns, not from a
  physical gouge.
- Recommendation recorded in the evidence: the double bevel at ρ = 5–10°, with the edge at target + 0.2 mm,
  preferring 10°.

## C5 — live Native TRIM, ρ = 10°

- Run `artifacts/wp11/phaseC5/live_trim_rho10_001` at `0c39b23`, cuda:1. Wall time 697 s (episode 637 s).
  Command in the evidence (`run_wp10_live.py --scene trim01_wp11_workcell --goal trim_bump_rho10 …`).
- Goal `trim_bump_rho10` PASS: 97.89 % of the dome removed, below-plane substrate share 0.99996, 2 mechanical
  components, detached below-plane fraction 4.2e-5. HOLDING on all 1035 steps (max slip 0.75 µm). Home PASS.
- Measured-path geometric clearance FAIL: the blade came −0.16 mm into the declared uncut workpiece box during
  PRE_CUT, with zero material reaction. This is an open issue, addressed in C5b.

## C5b — live Native TRIM plus chip clear

- Run `artifacts/wp11/phaseC5/live_trim_rho10_clear_002` at `03dfd1d`, from a clean worktree, on cuda:1.
- Attempt 001 is FAIL_EXECUTION. It was infrastructure: an unrelated job on GPU 1 caused a CUDA OOM. The output is
  preserved.
- Goal `trim_bump_rho10_clear` PASS. The trim step removed 97.89 % of the dome. The clear step pushed the chip
  +109 mm off the bar, with 2 components throughout and 0 label changes after the skim. PRE_CUT clearance PASS
  (blade lowest point ≥ +0.076 mm above the bar top). HOLDING 1400/1400. Home PASS.

## Disclosure: substrate rim particles left above the plane

The C5 and C5b evidence records `residual_substrate_height_m` of about 11–12 mm. That is the highest final
substrate particle centre above the 20 mm target plane in the dome footprint. No report mentioned it until now.
The goal reports it as quality, not as a scored check (`REPORTED_NOT_SCORED`), so every verdict above stands. It
is still a real particle-level residual, and the surface mesh hides it:

| Run (frame) | Recorded particle residual | Substrate particles > 1 mm above plane (height range) | Recorded mesh residual |
| --- | --- | --- | --- |
| C5 live (t = 14.00 s, evaluated) | 11.52 mm | 6 (2.46–11.52 mm) | 0.181 mm, 0 % missing |
| C5 live (t = 20.70 s, last recorded frame, after return home) | — | 6 (max 14.53 mm) | — |
| C5 material preflight used for live (`preflight_kin_slow_law8_001`) | 11.03 mm | 6 (2.48–11.03 mm) | 0.182 mm |
| C5b formal material preflight (`preflight_kin_clear_law8_001`) | 11.16 mm | 6 (6.95–11.16 mm) | 0.110 mm |
| C5b live (t = 28.00 s, evaluated) | 12.02 mm (episode `result.json` quality) | 7 (3.41–12.02 mm) | 0.105 mm, 0.04 % missing |
| C5b live (t = 35.16 s, last recorded frame) | 12.13 mm (evidence `live.trim_step`) | 7 (3.59–12.13 mm) | — |

- In C5b live these are 7 rim particles of the dome footprint. All have reference z = 20.5 mm, the lowest layer
  above the plane, at reference x 1.5–9.5 mm and |y| 2.5–9.5 mm. Each carries the substrate-side digit (code 1)
  and is still in the substrate mechanical component, so the outcome counts it as substrate. They end
  3.6–12.1 mm above the plane at the last frame (3.4–12.0 mm at evaluation), lifted and pulled inward behind the
  departing chip. C5 shows the same particles, with 6 of them above 1 mm.
- The mesh residual (0.105–0.18 mm) does not show them. The likely reason is that a few isolated particle centres
  standing above the bulk produce no isosurface under the `current_volume_v1` profile. This is consistent with the
  CPU test in which a single loose particle yields no triangles. That reason was not checked separately on these
  frames.
- So the particle-level and mesh-level residuals disagree by two orders of magnitude, and only the mesh value
  appeared in reports. A reader should treat "dome removed to the plane" as true for the mass and the surface
  mesh, but not for these 6–7 strung-up rim particles.
- Verification (read-only, main-repo `.venv`, `-I`): the reviewer's
  `/mnt/data/ningan/tmp/claude-1004/-home-ningan-RoboticCuttingSim/e538d80f-78df-4b03-8e24-f5e0ea11b086/scratchpad/rescore_c5b.py`
  (10 s; rescored verdict PASS, equal to the recorded one). Also
  `/mnt/data/ningan/tmp/claude-1004/-home-ningan-RoboticCuttingSim/e538d80f-78df-4b03-8e24-f5e0ea11b086/scratchpad/r1/c5_particles.py`
  and `…/r1/c5_particles_eval.py` (last and evaluated frames) and `…/r1/c5_time.py`. None of these scripts is
  committed.

## Correction: Phase M Test C chip merge (bumps3, `artifacts/wp11/phaseM/bumps3_law8_v5_001`)

The Phase M evidence ([`phaseM_tlmpm_v5_001.json`](../evidence/wp11/phaseM_tlmpm_v5_001.json)) and `docs/PROGRESS.md`
say this: about 49 substrate top-layer particles (reference z 11.5 mm, x 8.5–42.5 mm), labelled positive by skims 1
and 2 "where skim 2's start overlaps skim 1's end", join chips 1 and 2. That implies stationary material in an
overlap zone. **It is not.** At the recorded final frame (verified read-only with
`/mnt/data/ningan/tmp/claude-1004/-home-ningan-RoboticCuttingSim/e538d80f-78df-4b03-8e24-f5e0ea11b086/scratchpad/r1/phaseM_bridge.py`):
- The chips-1+2 body (1545 particles) holds 746 bump-1 and 750 bump-2 particles plus exactly 49 non-bump
  particles. All 49 have reference z = 11.5 mm (the bar's top layer) and reference x 8.5–42.5 mm.
- Those 49 particles were carried with the chips: median |displacement| 61.0 mm, z raised by +7.8 to +40.9 mm
  (mean +25.7 mm), final x −139 to +82 mm. The rest of the substrate's top layer moved 5.1 mm on average (the bar
  slide).
- Their packed codes are 40 (16 particles: positive on both the skim-1 and skim-2 digits), 32 (25: positive on
  skim 2, skim 1 unassigned) and 36 (8: skim-1 negative, skim-2 positive). The evidence's "codes 40/42" is
  inexact for this frame.

They are top-layer material torn off and carried about 61 mm by the blade, forming a multi-positive bridge between
the two chips. They are not stationary overlap material. The Phase M verdicts and evidence file are unchanged; this
paragraph corrects the explanation only.
