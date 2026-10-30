# WP-12 Phase 5 freeze decision (xhigh): envelope v2, benchmark v1, Phase 7 demos

Date 2026-10-31 (KST). Decider `rcs-xhigh` (user instruction 2026-10-31). Branch `claude/wp12-physical-cut-surfaces`.
Evidence and decisions were read at `1f9d441`. The tree is now at `c279a4d`: the orchestrator meanwhile committed the
opt-in `tlmpm_v6` candidate and its guard test; none of it is used here.

**Inputs:** [D1–D5](WP12_PHASE4_XHIGH_DECISION.md); [E2](WP12_PHASE4_E2.md) with `docs/evidence/wp12/phase4_e2_001.json`
(row values read from the evidence, not the summary); envelope v1 and the PROPOSED v2; the
[prototype](WP12_PHASE4_PROTOTYPE_DECISION.md) and [Phase 3](WP12_PHASE3_XHIGH_DECISION.md) §4 decisions; the DRAFT;
[GENERALIZATION](WP12_GENERALIZATION.md); `generalization_wp12.py`, `run_wp12_phase5.py`, `workcell_live_v1.json`,
`phase2b_native_001.json`; spec §5.1–5.8 and §7.2.

**Written:** `configs/wp12/supported_envelope_v2.json` (DECLARED), `configs/wp12/benchmark_v1.json` (FROZEN on commit)
and this file. **Unchanged:** envelope v1, the PROPOSED file, the DRAFT, and every solver, law, threshold, goal,
evaluator, runner, test and record. Nothing was committed.

**Checks run** (CPU, no physics): `freeze_problems(benchmark_v1, envelope_v2) == []`; all 289 declared samples build,
validate and round-trip by id; the threshold snapshot equals the `goals.py` defaults; the first sample of each of the 17
strata compiles; `pytest tests/test_wp12_generalization.py tests/test_tlmpm_v6.py` gives 52 passed. Scripts:
scratchpad `freeze/` (`make_envelope_v2.py`, `make_benchmark_v1.py`, `check_freeze.py`, `check_build.py`).

## Decisions

| | Decision |
| --- | --- |
| E-1 | **Envelope v2 is DECLARED.** The E2 reclassification is accepted. Bounds stay those of v1 except one widening: the pattern `sequence:sever>move_piece>remove_protrusions`, PARTIALLY_VALIDATED, for Demo 6. The **solid-board tilt bound stays at 9.9°** as a declared boundary-study bound, although 5° is a known failure inside it. The slanted chop is kinematically infeasible with this planner and is excluded. The validated cells are written as `phase5_id_rules` |
| E-2 | **Benchmark v1 is frozen on commit.** 17 strata × 10 = 170 samples, plus 85 held-out and 34 dev. ID uses validated cells only: 64 ID_validated and 16 ID_unconfirmed benchmark samples. A new OOD-only family **BF** measures the solid-board tilt boundary. `advance_tilt` is dropped from B |
| E-3 | **Native:** k = 3 for the 8 ID strata plus A_OOD (27 episodes). The other 8 OOD strata are NOT_RUN by declared budget. Native success is the Phase 2b definition; the PROVISIONAL runner mapping must be replaced before any Native benchmark episode |
| E-4 | **Phase 7:** Demos 1–7 exactly as in §3. Each runs `particle_surface_v1` (the regression record), then `cut_aware_surface_v1` on identical inputs. Demo 7 is a seeded draw from the held-out ID_validated samples, made before any held-out episode |

## 1. Envelope v2 (E-1)

**E2 facts, checked in the evidence JSON.**
- 18 rows exited 0 with label `E2_FORMAL`; 17 are CONFIRMED. SOLID_p05 left 1 part with 26 direct bridges (armed end
  0.275 mm); SOLID_p02 gave 2 parts (0.212 mm). Solid-board armed ends lie in 0.16–0.28 mm.
- Slotted ±20 / +25: lowest edge point −3.49 / −3.49 / −3.38 mm; handle slack 2.516 / 2.515 / 1.228 mm.
- Removal: skims 0.975 / 0.961 (20 / 30 mm/s), 0.952 (ρ 35), 0.973 (ψ ±10); TRIM_PLANE 0.959; TRIM_TRIM 0.970.
  CUT_PUSH pushed 19.90 mm; CUT_PUSH_TRIM pushed 19.91 mm, then trimmed 0.968 (3 parts). ADV ±15: blade 15.53 mm inside
  the board.
- **Not in the E2 report:** SKIM_r35's reported-quality maximum residual substrate height above the plane is
  **11.1 mm** (ρ 10 skims 1.0–1.4 mm, ψ ±10 3.2–3.3 mm, the CUT_PUSH_TRIM trim 4.1 mm). At ρ 35 a still-attached chip
  root stands up while the ≥ 0.9 removal criterion passes.

| Dimension (law 8, path_v2) | Phase 5 ID cell (VALIDATED) | Known failure / infeasible | Admissible, NOT_RUN |
| --- | --- | --- | --- |
| Slotted tilt θ (Option D, 4 mm overtravel) | \|θ\| ≤ 20 (E2 ±20; vertical D8_M400) | 35 (scripted); 32.1 handle collision (DEV) | (20, 34.9], except one +25 point |
| Solid tilt θ | 0 only | 5 through the planner (E2); ≥ 10 at h = 0 (E1) | (0, 9.9]: BF study (2° CONFIRMED once) |
| Plane yaw φ (solid) | \|φ\| ≤ 15 (E2 ±15) | 32.9 IK-unreachable (DEV) | (15, 35] |
| Advance tilt β | 0 | every β ≠ 0 with this planner: low edge end 60 mm·sin β inside the board | excluded |
| Rake ρ / skew ψ | ρ 10–35, \|ψ\| ≤ 10 (E2, both ends) | ρ 45 (tie); ρ < 9.2 infeasible | ρ (35, 44], ψ (10, 30] |
| Cut / skim speed | 8–12 / 20–30 mm/s (E2, both ends) | none | 4–8, 12–30 / 10–20, 30–60 |
| Patterns | sever, multi-sever, 1–3 removals, region trim, trim→trim, cut→push | trim_layer (D5) | sever↔trim, cut→push→trim (partial), clear-chip, cut→push→cut |

**Why 9.9° is kept.** The envelope separates samples that are never executed from samples that are executed and
measured. A bound below 5° would make the 5–9.9° samples physically_unsupported, so the 2–5° transition and the 5–10°
gap would stay NOT_RUN in the §7.4 matrix for good. OOD has no target, so a known failure inside the bound cannot move
any target, and `ood_boundary_by_dimension` attributes each failure to θ. The cost is about 10 kinematic episodes.
Guards: the scope sentence names this as the single exception, BF is OOD-only, and the rule note forbids ID, target and
demo use.

**Rejected:**
- widening any numeric bound (no evidence goes beyond v1);
- an ID cell for solid θ ∈ [0, 2]: 2° is one run at 0.212 mm, while the armed end ranges 0.16–0.28 mm and 5° fails at
  0.275 mm;
- a slanted-chop OOD stratum: geometry makes every sample collision_constrained, so it measures no boundary.

The pattern widening does not touch the benchmark: all 289 declared samples classify identically under v1 and v2.

## 2. Benchmark v1 (E-2, E-3)

| Family | Variants | ID (validated cells) | OOD dimensions (one at a time) | §5.7 target |
| --- | --- | --- | --- | --- |
| A | vertical | pose, size, cut 8–12 mm/s, clearance 6–10 mm | x, y, yaw, L, H, cut fraction, speed, clearance, material | inside |
| B | tilt_slotted (Option D), plane_yaw | slotted \|θ\| ≤ 20, \|φ\| ≤ 15 | slotted θ (20, 34.9], φ (15, 35] | inside |
| BF | tilt_solid | none (OOD only) | solid θ with 0 < \|θ\| ≤ 9.9 | outside (boundary study) |
| C | skim, trim_plane | ρ 10–35, ψ ±10, 20–30 mm/s, +x | ρ, −x skim, ψ, bump height, skim speed | inside |
| D | multi_plane_sever, cut_cut | 2–3 cuts in any order, ±3 mm jitter | 4–5 cuts, L, x, speed | inside |
| E | remove_all, remove_selected, bumps_sequential, trim_trim | 2–3 bumps | 4 bumps, height, ρ | inside |
| F | sever_then_trim, trim_then_sever | as in D5 | cut fraction, ρ, height, count | **outside** (class (c) failed 3/6) |
| G | cut_push; trim_clear_chip, cut_reposition_cut | push 20 mm, cut fraction 0–0.2 | push distance, −x sweep, cut fraction, x, ρ | cut_push inside; the other two **outside** |
| H | sever, trim | board 300–360 × 170–190 × 15–25 mm, μ 0.25–0.45, solid or slotted | yaw, board thickness and width, L, y radius | inside |

**ID changes from the D5 DRAFT.** Speeds become 8–12 and 20–30 mm/s. **L is 140–160 mm** (was 170): the flat-board
evidence covers 50–160 mm. The **push is fixed at 20 mm**, the only validated distance (E2 and the five WP-11
SLICE-PUSH runs). **G's cut fraction is 0–0.2**, which keeps the pushed piece at the validated 0.3–0.5 L (0.7 L was never
tested). B gets an ID stratum, BF is new, and `advance_tilt` is dropped.

**ID justification.** A machine check confirms that every ID_validated sample satisfies `phase5_id_rules` and every
ID_unconfirmed sample violates at least one. A range counts as ID only if both ends were confirmed through the planner
(or by an equivalent validated record) and a stated mechanism covers the interior: the solver works in the workpiece
frame on a co-rotated board and TLMPM sees only the board-top plane (board friction acts in PhysX only); a vertical
node-plane sever is location-invariant; arming in air does not touch the material; the armed end is set by the 0.3 mm
tolerance, not the speed.

**Uncertainty.** Each validated end is **one** kinematic run: seed 42, the 160 × 30 × 20 mm bar, one lattice phase.
- One success in one run bounds the per-sample success rate only to [0.21, 1] (Wilson 95 %).
- Dimensions were confirmed one at a time; their interactions are untested, and measuring them is this benchmark's job
  (for example slotted 20° at 12 mm/s on a 24 mm bar at a random phase, or ρ 35 on three bumps at 30 mm/s).
- Watch list: about 0.1 mm armed-end margin; ρ 35 removes 0.952 and leaves an 11.1 mm residual; the 30 mm/s skim
  removes 0.961; slotted handle slack 2.52 mm.
- Per stratum (n ≤ 10), 10/10 gives only [0.72, 1]. Pooled over about 60 feasible ID_validated samples, the lower bound
  reaches 0.90 only at ≥ 59/60.
- An ID failure stays an ID failure: never moved to OOD or reclassified. A cell change means envelope v3 and benchmark v2.

**BF draws** are fixed by the seeds; they were seen at the freeze check and **not** adjusted. |θ| is 0.45, 2.68, 3.14,
and the other seven lie in 5.6–8.9°; 8 of 10 are negative. BF is therefore a coarse boundary study that mainly tests
monotonicity beyond 5° and the sign. More samples at 2–5° would need per-family counts (benchmark v2).

**Seeds.** Rule unchanged: base + split {0, 500, 900} + set {0, 200} + i; per stratum 10 benchmark, 5 held-out and 2 dev.
BF starts at 60000, because 59000 would collide with H's dev OOD seeds 59100–59101. Order of use: the benchmark split;
then held-out, once and never for tuning; then Native on benchmark seeds only, plus Demo 7 from held-out.

**Native (E-3).**
- **Selection:** `first_k_valid_feasible`, k = 3, on feasibility only. The 8 ID strata give 24 episodes (about 19
  ID_validated). A_OOD adds 3 pose and size episodes, where Native-only reach, tracking and grasp effects are expected
  to concentrate.
- **Success:** the `workcell_live_v1` `task_success_requires` (goal PASS, plan DONE, finite state, slip ≤ 2 mm and
  rotation ≤ π/60, no IK failure or clamp), classified by `classify_live`. A launched attempt with no result is a crash,
  never not_run.
- **Budget:** 6–17 min per episode; estimate 5 h, cap 10 h, one episode at a time on GPU 0.

**Metrics and targets.** The DRAFT blocks are kept verbatim: each goal's own declared thresholds (the `goals.py`
defaults, checked) plus the WP-11 minimum-part rule. Targets: kinematic ≥ 0.90 and Native ≥ 0.80 success among feasible,
over the ID_validated samples of the benchmark split; the point estimate and the lower bound are both reported, and
held-out is reported separately. "No reconnection" and `--repeat` are declared but not built: reported NOT_RUN if still
missing, and neither gates success.

**Kinematic budget:** about 220 episodes; estimate 15–35 job-hours, cap 45, at most 2 jobs per device. Over budget the
run pauses and resumes from the same committed files, with no re-selection. `validate_benchmark` accepts only DRAFT or
FROZEN, so the file carries `status: FROZEN` and `freeze.state: FROZEN_ON_COMMIT`.

## 3. Phase 7 demos (E-4)

Every demo: `tlmpm_v5` law 8 at 1 mm / 4 mm / 0.1 ms, seed 42, `cutting_tasks_v1`; Native Isaac Sim 6.2 only, never a
kinematic replay; the Phase 2b flags plus `--record-cameras` and `--overview-camera`; planner overrides 1°, 1 mm,
`path_v2`. `particle_surface_v1` runs first, then `cut_aware_surface_v1` on identical inputs with cut close-ups (never
`cut_aware_surface_v2`). Every run is kept.

| Demo | Exact configuration | Evidence now | Phase 7 |
| --- | --- | --- | --- |
| 1 Clean cut | `scenes/bar_flat_board.json` + `wp11/goals/sever_mid.json` | `workcell_live_001` PASS | rerun + pair |
| 2 Oblique | E2 `SLOT_p20` inputs (`build_row_sample`): the **explicit Option D 8 mm groove**, θ = +20°, plane through x = 0 at the board top, 4 mm overtravel, 10 mm/s; labelled Option D | kinematic CONFIRMED; Native NOT_RUN | Native + pair; optional still of E2 `SOLID_p05` labelled KNOWN FAILURE |
| 3 Trim | `scenes/trim01_flat_board.json` + `wp11/goals/trim_bump_rho10.json` (path_v2, ρ 10) | `workcell_live_003` PASS 0.977; disclose the report-only PRE_CUT dip | rerun + pair (spikes shown) |
| 4 Four slices | `scenes/bar_flat_board.json` + `wp12/goals/divide_5_slices.json` | `workcell_live_004` PASS | rerun + pair |
| 5 Three bumps | `scenes/bar_three_bumps_flat_board.json` + `wp12/goals/remove_all_protrusions_rho10.json` | `workcell_live_005` PASS; 2 chip particles 0.6 mm below the board top (disclosed) | rerun + pair |
| 6 Mixed | E2 `CUT_PUSH_TRIM` inputs: sever at x = 0, push the +x piece 20 mm, then trim p0 (12 × 10 × 8 mm at x = −44.8 mm) on the **stationary** piece, one episode | kinematic CONFIRMED (3 parts, 0.968); Native NOT_RUN | Native + pair; describe it as "the other piece was moved" |
| 7 Generalized | `phase7_demos.demo_7`: from the held-out ID_validated valid_feasible pool in declared order, the first sample after a `default_rng([1205, 7])` permutation. Drawn right after held-out feasibility and before any held-out episode, recorded in `artifacts/wp12/phase7_demo7_selection_001.json`, and committed | after the freeze | Native + pair; never redrawn |

## Findings ranked by severity

1. **High.** The Native mapping is PROVISIONAL and weaker than the Phase 2b definition (`scripts/run_wp12_phase5.py:488`):
   no grasp gates, no IK or clamp counts, and a preflight refusal mapped to not_run. Native successes could therefore be
   false positives. The frozen file declares the stricter definition, which must be implemented first.
2. **High (resolved by exclusion).** No flat-board slanted chop can be executed. `through_cut_rotation`
   (`goal_planner.py:212`) and the edge-centre stop in `_relief_end` (`:297`) guarantee a board collision, so every B
   `advance_tilt` sample would have been collision_constrained.
3. **Medium.** Provenance. The runner defaults to the DRAFT and envelope v1 (`run_wp12_phase5.py:553–554`), and the
   queue does not forward `--envelope` (`:366`). Classes are identical under v1 and v2, so a wrong envelope only changes
   the recorded hash; a DRAFT config would be a different benchmark. E2 also ran before the v6 plumbing was committed
   (`28c8379`): disclosed, not hash-verified.
4. **Medium.** The E2 report omits SKIM_r35's 11.1 mm residual: ρ 35 passes the criterion with a standing chip root.
   Watch-listed; the criterion is unchanged.
5. **Medium.** The DRAFT ID went beyond the evidence in three places (L 170 mm, 15–25 mm pushes, 0.7 L pushed pieces),
   all fixed here. Every validated end is still a single run.
6. **Low.** The v1 scope says `freeze_problems` skips `when` rules (fixed in v2). The DRAFT notice names the path
   `configs/wp12_benchmark_v1.json`; the actual path is `configs/wp12/benchmark_v1.json`. Reconnection and `--repeat`
   are not built.

## Next steps (in order)

1. **Orchestrator.** Run
   `.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from robotic_cutting_sim import generalization_wp12 as W; print(W.freeze_problems(W.load_benchmark('configs/wp12/benchmark_v1.json'), W.load_envelope('configs/wp12/supported_envelope_v2.json')))"`.
   It must print `[]`. Then commit the two configs and this file **unedited**, in one commit with no other files.
   Expected sha256: `915a2a8f…` (benchmark), `040e7b96…` (envelope).
2. **rcs-impl** (CPU, in a separate commit; the frozen files are never edited):
   - `native_execution` implements `native.success_definition`, with a test that reproduces `workcell_live_001–007`;
   - for a FROZEN config, default to `config['envelope']` and refuse a mismatch; the queue forwards `--envelope`;
   - a test pins the sha256, `freeze_problems == []` and the `phase5_id_rules` split;
   - a Demo 7 selector; optionally reconnection and `--repeat`;
   - `configs/wp12/phase7_demos_v1.json` (PREDECLARED) from §3, with the Demo 2 and 6 files generated from the E2 rows.
3. **Kinematic.** Pass `--config configs/wp12/benchmark_v1.json --envelope configs/wp12/supported_envelope_v2.json`,
   writing to `artifacts/wp12/phase5_bench_001`. Benchmark: `feasibility --split benchmark`, then
   `queue --devices cuda:1 cuda:0`, then `summarize`. Held-out: feasibility, then the committed Demo 7 selection, then
   the held-out kinematic run (once).
4. **Native** (GPU 0, one run at a time). First Demo 2 and Demo 6 with `particle_surface_v1`: the first Native slotted
   and push runs, whose outcomes never change the frozen files. Then `native-select` and the 27 episodes. Then Demos 1,
   3, 4, 5 and 7, and every `cut_aware_surface_v1` pair.
5. **Report** in `WP12_GENERALIZATION.md`: the matrix and failure classes; a BF table per sample (θ, armed end, parts,
   bridges); the declared exclusions; and the §7.4 Oblique row, which becomes PASS ("range declared and evidenced") only
   after Demo 2 Native.
