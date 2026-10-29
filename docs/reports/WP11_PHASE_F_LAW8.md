# WP-11 Phase F2/F3: law-8 boundary validation and resolution study

**Verdicts at a glance**

- **F2 boundary rows:** 15 of 16 PASS. **F2.9_blunt_handle FAILS.** The negative control
  was not loaded enough: the grip contact-window mean force was 0.077 N, and the declared
  level is > 0.1 N. The FAIL record stands; see the [diagnosis](#f29_blunt_handle-fail-diagnosis).
- **F3 formal rows:** CUT and TRIM both PASS, so the formal PASS is reproduced.
- **F3 variants:** `coarse` and `dt_half` are CHARACTERIZATION rows, as declared.
- **F3 medium (1.5 mm, 6 mm, 0.15 ms):** ERROR / NOT_RUN for both CUT and TRIM, because
  0.02 / 0.00015 is not an integer number of substeps. The runner's labelled substitute,
  `medium_dt133_supplementary`, is reported separately as **non-protocol**. It is never
  the declared medium row.

| Item | Value |
| --- | --- |
| Evidence | [`docs/evidence/wp11/phaseF_material_001.json`](../evidence/wp11/phaseF_material_001.json), written by `scripts/summarize_phase_f.py` (read-only; it copies verdicts and never overwrites) |
| Protocol | [`configs/wp11/phaseF_protocol_v1.json`](../../configs/wp11/phaseF_protocol_v1.json), sha256 `0efcf885…`, frozen in `11b9c99` and `f2168b6` before any run. Every row records this digest. |
| Runner | `scripts/run_phase_f_material.py` at `f2168b6` (sha256 `5aea75d7…`, equal to `git show f2168b6:…`). Helpers are in `src/robotic_cutting_sim/law8_validation.py`. |
| Runs | `artifacts/wp11/phaseF/<run>/` (`result.json`, `manifest.json`, `actions.jsonl`, `physics_state.npz`). Logs are `artifacts/wp11/logs/phaseF_<run>.log`; the queue log is `artifacts/wp11/phaseF/queue.log`. |
| Execution | Clean worktree at `f2168b6`, `git_dirty = false` on every row. Run from `/mnt/data/ningan/RoboticCuttingSim-wp11-F-f2168b6` on `cuda:0`, seed 42, `kinematic_ideal_tracking`. |
| Common settings | Cutting law 8 on `tlmpm_v4` (`tlmpm_v5` for F2.11). Knife `ruler_knife_v2_y`, double bevel. 1 mm particles, 4 mm grid, 0.1 ms material dt, 20 ms control dt. |
| Material | Provisional and uncalibrated: E = 50 kPa, ν = 0.3, yield 1 kPa, no hardening, μ = 0.3 |
| Scope | No robot, IK, grasp, Isaac Sim/Lab or cameras. Full physics acceptance: false. |

Every row passes the mass check: the recorded mass equals the sampler mass within the
frozen tolerance.

## Per-row results (F2 and F3 formal)

Notes on the table:

- **Force** is the material reaction on the tool: the impulse divided by the control dt.
  The value shown is the mean / max over the armed phase.
- **Wall** is the row's `wall_s`. Three rows shared GPU 0 at a time.

| Row | Setup | Criterion | Verdict | Key numbers | Wall (s) |
| --- | --- | --- | --- | --- | --- |
| F2.1_gate_14 | TRIM-02-K2, ρ = 14° skim | removal ≥ 0.9, substrate ≥ 0.99 | PASS | removal 0.941, substrate 0.99998. **Plowing on 0/168 cutting steps, segment fraction 0.0** | 360 |
| F2.1_gate_15 | same, ρ = 15° | same | PASS | removal 0.933, substrate 0.99998. **Plowing on 0/168, fraction 0.0** (the float32 tie at exactly 15° falls on the non-plowing side, as in C4) | 308 |
| F2.1_gate_16 | same, ρ = 16° | same | PASS | removal 0.933, substrate 0.99987. **Plowing on 168/168 steps, segment fraction 1.0.** The lead rule labels 41124 particles (37044 at 14°) but removes the same mass. | 334 |
| F2.3_inplane_v2 | CUT-03-K2 vertical cut | frozen strict oracle | PASS | 2 parts of 0.072 kg each; interface 1.5 mm (≤ 2 mm); persists 2.4 s; force 0.012 / 0.057 N | 373 |
| F2.4_rocking | CUT-03-K2, edge rocks ±10° about the blade normal | two parts ≥ 40% each and ≥ 99% together; no far particle on the wrong side | PASS | parts 0.50 / 0.50. Interface 1.5 mm, equal to the ideal geometric label value. 0 far particles in the opposite component. Strict oracle (report-only) PASS. | 334 |
| F2.5_pure_rotation | half-depth cut, then 0.5 s armed ±8° rotation about the edge centre, then armed withdrawal | one component; labels within 1 cell of the swept region | PASS | 1 component, 15238 labels, 0 outside the swept region (final and at assignment). 1742 labels were added while rotating. The stroke was active on 25/25 rotate steps, and the centre segment speed was 0. | 316 |
| F2.6_saw_cut | CUT-03-K2 with ±6 mm, 0.4 s sawing while descending | frozen strict oracle | PASS | 2 parts, interface 1.5 mm, force 0.021 / 0.085 N. Swept-normal error 0.0°, but min \|m×e\|² = 8.4e-8 (m/s)² ([not validated](#what-law-8-is-not-validated-for)). | 336 |
| F2.6_pure_draw | half-depth cut, then 0.8 s armed draw in place | one component; no label more than 1 cell below the deepest edge | PASS | 1 component, 14400 labels, 904 of them added during the draw. Max excess below the deepest edge: +0.015 mm at assignment, −0.5 mm final (tolerance 4 mm). m×e = 0 on 40/40 draw steps, so the blade-normal fallback applied. | 331 |
| F2.7_withdrawal_v2 | half-depth cut, then armed withdrawal | one component; core band below the stop all code 0; labels unchanged during withdrawal | PASS | 2400 core particles below the stop, 100% code 0. 13496 labels before and after 90 withdraw steps. Stroke active on 0 withdraw steps. | 319 |
| F2.8_finite_edge | CUT-04-K2-W60; half the bar width under the edge | one component; no label with \|l\| > L; severed bonds only within the edge span | PASS | 1 component. 35714 labels, max \|l\| at assignment 59.59 mm (L = 60 mm). 750 severed bonds, all at l = 30.5–59.5 mm; 0 outside the span. | 763 |
| F2.9_flat_press | blade face down, armed, pressed −z | 0 labels; 1 component; contact-window mean > 1 N | PASS | 0 labels, 1 component, mean 1.49 N, max 1.89 N over 25 contact steps | 330 |
| **F2.9_blunt_handle** | armed grip pressed 3 mm into the bar side | 0 labels; 1 component; grip contact-window mean > 0.1 N | **FAIL** | 0 labels, 1 component; **mean 0.077 N**, max 0.131 N over 30 contact steps | 328 |
| F2.9_side_push | armed blade-face push into the bar end | 0 labels; 1 component; mean > 0.1 N | PASS | 0 labels, 1 component, mean 0.318 N, max 0.359 N over 64 contact steps | 329 |
| F2.9_no_contact | armed skim 3 mm above the dome | 0 contacts, 0 labels | PASS | 0 blade contacts and 0 grip force on 167 armed steps; 0 labels | 329 |
| F2.10_recontact | after a full cut, the disarmed blade pushes one piece back for 0.5 s | stays 2 components; nonzero piece–piece contact; no label change while disarmed | PASS | 2 components on all 45 post-cut frames. 3.47e6 contact events and 0.117 N·s impulse during the push. Up to 1575 cross-piece pairs within contact distance. 0 label changes on 221 disarmed steps. | 343 |
| F2.11_multicut_history | `tlmpm_v5`, 3 cuts on a slotted bar, then a disarmed push | 4 parts; earlier digits unchanged; no slot overflow | PASS | 4 parts (0.413 / 0.200 / 0.194 / 0.194). 0 changes of any digit after its cut ended, and 0 nonzero-digit rewrites. Max 4 of 8 slots used, 0 overflow; 0 labels during 216 push steps. | 1204 |
| F3_cut formal | CUT-03-K2, formal resolution | frozen strict oracle | PASS | identical to F2.3 (28528 labels, interface 1.5 mm) | 338 |
| F3_trim formal | TRIM-02-K2-R10, formal resolution | removal ≥ 0.9, substrate ≥ 0.99 | PASS | removal 0.956 (998/1044), substrate 1.0. The C4 record `k2_r10_skim_law8_001` had 0.957 (999/1044): reproduced within GPU nondeterminism. | 249 |

**F2.11, report-only.** The configured-goal evaluation, which is not a Phase F
criterion, fails `cut_at_declared_planes` for planes 1 and 2. The midpoint statistic is
4.0 and 4.5 mm against the 2 mm bound, while ideal labels give 1.5 mm. The earlier cuts
move the bar before the later ones, as in M8b. It is reported, not rescored.

## F3 resolution study

Notes on the table:

- **Separation** for CUT is the frozen strict oracle. For TRIM it is `trim_separation`
  (removal / substrate share).
- **Location** is the frozen midpoint statistic (bound 2 mm). The ideal-partition floor in
  brackets is the smallest value any labelling can reach at that spacing.
- **Force** is the armed-phase mean / max |F|.

| Variant (spacing, grid, dt) | Status | CUT separation; masses (kg); location | CUT force (N) | TRIM removal / substrate; masses (kg) | TRIM force (N) | min det F (CUT / TRIM) | Wall (s) (CUT / TRIM) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| formal (1 mm, 4 mm, 0.1 ms) | protocol, PASS / PASS | PASS; 0.0720 / 0.0720; 1.5 mm [1.5] | 0.012 / 0.057 | 0.956 / 1.0; 0.1441 / 0.0015 | 0.013 / 0.081 | 0.938 / 0.918 | 338 / 249 |
| coarse (2 mm, 8 mm, 0.2 ms) | characterization, criterion FAIL / FAIL | 2 parts but FAIL on location; 0.0720 / 0.0720; 3.0 mm [3.0] | 0.001 / 0.015 | **0.816** / 0.9994; 0.1442 / 0.0014 | 0.014 / 0.168 | 0.950 / 0.890 | 37 / 51 |
| dt_half (1 mm, 4 mm, 0.05 ms) | characterization, criterion PASS / PASS | PASS; 0.0720 / 0.0720; 1.5 mm [1.5] | 0.031 / 0.130 | 0.973 / 1.0; 0.1440 / 0.0015 | 0.023 / 0.104 | 0.946 / 0.872 | 689 / 689 |
| medium (1.5 mm, 6 mm, 0.15 ms), as declared | **ERROR / NOT_RUN** | `ValueError: control_dt must be an integer multiple of material_dt` | – | same error | – | – | 0.02 / 0.02 |
| medium_dt133_supplementary (1.5 mm, 6 mm, 0.02/133 = 0.150376 ms), **non-protocol** | supplementary, criterion FAIL / FAIL | 2 parts but FAIL on location; 0.0711 / 0.0698; 2.75 mm [2.75] | 0.013 / 0.080 | **0.577** / 1.0; 0.1416 / 0.0011 | 0.025 / 0.142 | 0.966 / 0.965 | 98 / 88 |

**Stability.** Every run is stable: no NaN, no inverted step, and min det F ≥ 0.872.

**Medium error.** CouplingConfig rejected the declared dt during `env.reset`
(`coupled_robot.py:37`), because 0.02 / 0.00015 = 133.3 substeps. The process exited
with status 1, as GNU time records in the row log. The queue log still prints `rc=0` for
these two rows, because it records the wrapper's exit code; the evidence flags the
mismatch in `integrity.queue_rc0_but_log_exit_nonzero`. The supplementary variant was
defined in the committed runner before any run.

The reading of the resolution study (characterization; nothing is rescored):

- **The formal rows reproduce their PASS,** and so does halving dt.
- **The CUT location check cannot pass on a coarser lattice.** At 2 mm and 1.5 mm the
  frozen 2 mm statistic sits below the ideal-partition floor (3.0 and 2.75 mm). Separation
  and mass are otherwise intact.
- **TRIM separation is not resolution-converged.** It depends on where the lowest
  above-plane particle layer falls relative to the edge, which runs 0.2 mm above the
  target:
  - At 1 mm that layer sits at z = 20.5 mm, 0.3 mm above the edge, and 46 of its 316
    particles stay attached.
  - At 1.5 mm it sits at 20.25 mm, only 0.05 mm above the edge, and all 160 of its 160
    particles get the substrate digit.
  - At 2 mm, 18 of 70 particles in the lowest layer and 7 of 38 in the next stay attached.
- **Contact forces are not converged.** Halving dt raises the CUT mean force 2.6× and
  the max 2.3×. The coarse lattice gives a force 12× lower. The forces here are
  provisional diagnostics, not calibrated loads.

## F2.9_blunt_handle FAIL diagnosis

The numbers come from `actions.jsonl` and `physics_state.npz` in
`artifacts/wp11/phaseF/f2_9_blunt_handle_law8_001`. They are read on the CPU; nothing
was rerun.

**Contact and force history.**

- **Phases:** `handle_press` runs 1.38–1.96 s (30 steps at 10 mm/s) and `handle_hold`
  runs 1.98–2.26 s (15 steps).
- **First contact:** the grip −y face starts at y = 17.8 mm, against the bar side face at
  15 mm. It first touches at y = 14.8 mm, at t = 1.68 s. From then on there is grip
  contact on 30 of the 45 steps.
- **During the press:** |F| rises 0.028 → 0.072 → 0.100 → 0.131 N. The 0.131 N is
  reached at the 6 mm stop, with the face at y = 12 mm.
- **During the hold:** |F| decays steadily from 0.105 to 0.018 N.
- **Window statistics:** mean 0.0767 N and max 0.1308 N over the 30 contact steps. Over
  all 45 steps the mean is 0.051 N.
- **No cutting activity:** 0 blade contacts, the stroke never active, and min det F 0.975.

**Contact patch.** The grip barely overlaps the bar.

- The handle box lies 60–120 mm along the edge line from the edge centre. Turned to −x,
  that is world x = 60–120 mm, but the bar ends at x = 80 mm.
- In z the grip spans 10–34 mm, while the bar is 20 mm high.
- The patch is therefore about 20 × 10 mm (2e-4 m²), at the free +x end corner. The peak
  mean pressure is 0.65 kPa.

**Why 3 mm into a 50 kPa bar gives so little force.** The 3 mm went into rigid motion
of the bar, not into compression.

- CUT-03-K2 has no fixture walls. The bar rests on two pedestals (x = −90…−20 and
  20…90 mm) with μ = 0.3 friction only.
- At t = 2.0 s, the end of the bar (x > 60 mm) has moved −3.09 mm on the grip face,
  −3.03 mm in the core and −3.00 mm on the far face. The across-width compression is at
  most 0.13 mm (0.45% strain).
- The core band fits a yaw of 1.29° to 1.50° (t = 2.0 to 2.3 s) about x ≈ −43 mm. The
  yaw stays at 1.41° until t = 8 s, because friction holds the bar where it was pushed.
- The force is therefore limited by friction against yaw and by bending, not by the
  bar's stiffness:
  - Pure sliding is bounded by μW = 0.3 × 1.413 N = 0.42 N.
  - The textbook minimum for yawing a rod pushed at one end is (√2 − 1)μW ≈ 0.18 N,
    about a pivot L/√2 = 113 mm from the pushed end (x ≈ −33 mm).
  - That matches the measured 0.131 N peak and the x ≈ −43 mm pivot.
- When the grip stops, the bar comes to rest against it and the contact force decays.
- **If the bar were restrained**, 3 mm across 30 mm on 2e-4 m² would give
  E·ε·A ≈ 1.0 N elastically, capped by the 1 kPa yield at about 0.2–0.6 N. Either value
  is above 0.1 N.

The FAIL is a control-design flaw: the workpiece was unrestrained and the grip sat
mostly beyond the bar end. It is not a labelling event, since there were zero labels.
It means blunt contact at > 0.1 N remains unvalidated.

**Proposed v2 row (not run).** It would be declared as a new protocol version before any
run, with the v1 FAIL record unchanged:

1. **Put the grip over material and support.** Shift the plan by −40 mm along x, so that
   the 60 mm grip spans x = 20–80 mm over the right pedestal. Lower it to cover bar
   z ≈ 2–20 mm, keeping the blade ≥ 8 mm outside the side face and above the pedestal.
2. **React the press.** Either add a declared −y backstop fixture beside the bar on that
   pedestal, as a CUT-03-K2 variant, or press the grip −z onto the bar top over the
   pedestal. The flat-press row shows a blade-face −z press gives more than 1 N.
3. **Keep the criterion, add a check.** Keep the > 0.1 N contact-window criterion. Add a
   report-only rigid-body displacement and across-width compression of the workpiece, so
   that an unrestrained workpiece cannot absorb the press unnoticed.

## What law 8 is validated for

This holds only for kinematic ideal tracking at 1 mm / 4 mm / 0.1 ms (and 0.05 ms),
seed 42, with `ruler_knife_v2_y` (double bevel) on CUT-03-K2, CUT-04-K2-W60 and
TRIM-02-K2.

- **The plow-gate boundary.** ADR 0012 left θ_plow untested between 0° and 25°. It is now
  measured:
  - ρ = 14° and 15° never plow.
  - ρ = 16° plows on every cutting step and segment.
  - All three separate (removal 0.941 / 0.933 / 0.933). At these rakes, the switch at 15°
    does not change the separation outcome.
- **Cutting motions:** in-plane cut, ±10° rocking, sawing while descending, pure rotation,
  pure draw, armed withdrawal, and a finite edge with half the bar width under it.
- **Labels stay where the edge went.** Pure rotation adds no label outside the swept
  region, pure draw none below the deepest edge, and withdrawal none at all. The finite
  edge severs no bond outside its span.
- **Negative controls:** flat press (1.49 N), side push (0.32 N) and no contact.
- **Pieces after the cut:** re-contact without re-merge, and three-cut history persistence
  with no slot overflow on `tlmpm_v5`.

## What law 8 is not validated for

- **Chisel profile.** The C4 finding in
  [ADR 0013](../adr/0013-ruler-knife-v2-and-trim-orientation-metrics.md) (section "Decision record — C4 material level")
  stands. The gate measures motion against the blade plane, not against the pushing face.
  The chisel's top face pushes the chip at ρ + 25° while the gate sees only 5–10°, so
  ejected chip particles stay code 0 and bridge (ρ = 10° FAIL). Phase F did not test the
  chisel.
- **Loaded blunt-handle contact (> 0.1 N).** See the F2.9 FAIL above.
- **Calibrated material.** All parameters are provisional. Reaction forces are neither
  dt- nor spacing-converged (F3).
- **Isaac Lab or live tracking.** Every row is kinematic ideal tracking, and no Lab row
  exists. The ADR 0012 live-noise statements are not covered here.
- **Swept normal while sawing.** The swept normal is ill-conditioned. In F2.6_saw_cut the
  host-side |m×e|² drops to 8.4e-8 (m/s)²: |m×e| = 0.29 mm/s against segment speeds up to
  93 mm/s, about 0.3%. In F2.6_pure_draw, m×e = 0 on all 40 draw steps and the normal
  falls back to the blade normal. Ideal tracking gives 0.0° of normal error. But an
  out-of-plane velocity error of about 0.3 mm/s would turn the swept normal by about 45°.
  The solver uses the substep twist, and this report uses the control-step view.
- **Stroke persistence under rotation.** The 2 dx persistence budget counts edge-centre
  travel only: `tlmpm_v4._update_stroke` integrates the linear twist. In F2.5 the centre
  speed was 0 on all 25 rotate steps (`max_travel_since_engaged = 0`). The stroke stayed
  active only because the outer segments were cutting. A rotation about the edge centre
  that does no work would never use up the budget. This is untested.
- **Other resolutions.** TRIM separation fails at 2 mm and at the supplementary 1.5 mm
  lattice. The frozen cut-location statistic is below the ideal floor there. The declared
  medium row never ran.

## Commands, device and wall times

All rows ran from the clean worktree `/mnt/data/ningan/RoboticCuttingSim-wp11-F-f2168b6`
at `f2168b6`, three at a time on `cuda:0`. The queue window was
2026-10-30T04:16:37+09:00 to 05:06:47+09:00. The command template:

```
/home/ningan/RoboticCuttingSim/.venv/bin/python scripts/run_phase_f_material.py \
    --row <ROW> --variant <formal|coarse|dt_half|medium|medium_dt133_supplementary> \
    --device cuda:0 --out /home/ningan/RoboticCuttingSim/artifacts/wp11/phaseF/<run>
```

The exact command of each row is in `rows[].command` and `rows[].log.command_timed` of
the evidence.

**Wall times (s):**

| Group | Wall times (s) |
| --- | --- |
| F2.1 gates 14 / 15 / 16 | 360 / 308 / 334 |
| F2.3 to F2.7 | F2.3 373, F2.4 334, F2.5 316, F2.6 saw 336, F2.6 draw 331, F2.7 319 |
| F2.8 to F2.11 | F2.8 763, F2.9 flat 330, blunt 328, side 329, none 329, F2.10 343, F2.11 1204 |
| F3_cut | formal 338, coarse 37, dt_half 689, medium 0.02 (ERROR), dt133 98 |
| F3_trim | formal 249, coarse 51, dt_half 689, medium 0.02 (ERROR), dt133 88 |

The evidence was generated with:

```
.venv/bin/python scripts/summarize_phase_f.py --runs artifacts/wp11/phaseF/f*_law8*_001 \
    --notes <notes.json> --out docs/evidence/wp11/phaseF_material_001.json
```

The notes file holds the scope, diagnosis and limitation text, which is copied into the
evidence. The evidence's `integrity` block confirms the following:

- 26 rows, one run sha (`f2168b6`), no dirty run, a single device and seed;
- every protocol digest matches the protocol file;
- no queue/result verdict mismatch;
- the two `rc=0` / exit-1 ERROR rows are flagged.

The summarizer CPU test is
`.venv/bin/python -m pytest -q tests/test_summarize_phase_f.py`: 2 passed, exit 0.
