# ADR 0006 — Cutting law 4: work-gated stroke, geometric swept band (BladeMaster III-C)

Date: 2026-10-29. Status: accepted for implementation; scientific acceptance
is the Phase 1 matrix below ([plan](../BLADEMASTER_DEMO_PLAN.md) §2).

## Context

- Law 2 writes a side digit only where a 1 mm edge segment did cutting work
  in that substep. Unlabelled (code 0) particles then bridge the cut through
  the 4 mm reference stencil. Ideal-tracking 1 mm TRIM-01-R removes 0%, and
  Lab CUT-01-R stays one component
  ([diagnosis](../reports/WP-10_CLAUDE_DIAGNOSIS_20261029.md)).
- Law 3 added three grid-scale closures (lateral, persistence, lead). TRIM
  still fails at 0%. Its remaining code-0 bridges are substrate columns beside
  the dome (for example local (−11.5, ±14.5, 16.5–19.5) mm). Those columns
  passed behind the edge while only the dome's narrow front segments were
  energized, so the ±2-cell lateral closure did not reach them.
- The 4.0 mm cut-location failure in the law-3 Lab replays comes from the
  lead closure. The measured Lab edge sits at x ≈ −0.51 to −0.55 mm. The lead
  closure labels the −0.5 mm particle layer by its pre-contact position,
  which is +0.04 mm on the positive side. Law 2 labels particles only after
  they pass the edge, by which point edge contact has moved them back to the
  negative side. Native law 2 recorded the whole −0.5 mm layer on side 1 and
  the +0.5 mm layer on side 2, so that run passes.
- [BladeMaster](https://arxiv.org/abs/2609.27342) Section III-C keeps two
  states separate. Cutting progress \(D_p\) (Eq. 5) changes only the
  directional contact impulse. The side digit (Eq. 6) goes to every
  unassigned particle inside a finite band behind the edge (Eq. 4: \(|s|\le
  h\), \(|\ell|\le L\), \(-b_\text{back}-\max(u_c\Delta t,0)\le f\le 0\),
  with \(h = 6\Delta x\)), independent of each particle's own work. Ordinary
  collision changes neither \(D_p\) nor \(c_p\).

## Decision

Add opt-in cutting law 4 in a new module, `tlmpm_v4.py`. It is derived from
`tlmpm_v3.py`, and its laws 1/2/3 must equal v3 bit for bit. `tlmpm.py`,
`tlmpm_v2.py`, `tlmpm_v3.py`, v1/v2 task thresholds and every record stay
unchanged. Runtime presets keep their current defaults; law 4 is selected
only explicitly.

Per substep, with edge frame axes d (local x, toward the cutting front), e
(local y, along the edge) and n (local z, blade normal):

1. **Contact and progress \(D_p\)**: identical to law 2. Directional work is
   \(\Delta W=(J^+_d u_c+\eta J^+_\text{slice}|u_s|)\cdot\cos^2\) (the
   in-plane alignment). It releases only the directional impulse component
   and never touches stress or side codes.
2. **Segment cutting motion \(M_k\)** (edge segments at particle spacing):
   the segment velocity is \(v_k=v+\omega\times(e\,\ell_k)\). Then \(M_k\)
   holds when the armed digit \(\ge 0\), the segment moves forward
   (\(v\cdot d>u_\min\)) or draws (\(|v\cdot e|>u_\min\) with
   \(v\cdot d\ge-u_\min\)), and the motion is within
   \(\theta_\max = 45^\circ\) of the blade plane:
   \((v\cdot n)^2\le\tan^2\theta_\max\,[(v\cdot d)^2+(v\cdot e)^2]\).
   Here \(u_\min = 10^{-5}\) m/s, the law-2/3 value.
3. **Engagement \(W\)**: \(W = 1\) when, in this substep, some particle
   receives directional cutting work \(\Delta W>0\) at a segment with
   \(M_k\). Blunt-obstacle (handle, finger-pad) contacts never contribute.
4. **Stroke state \(S\)**, one per blade (not per segment):
   - If the armed digit \(c\ge0\) and some \(M_k\) holds:
     - with \(W\), set \(S=c+1\) and reset the travel \(\tau=0\);
     - otherwise, if \(S=c+1\), add the edge-centre path length,
       \(\tau \mathrel{+}= |v|\Delta t\), and set \(S=0\) once
       \(\tau>b_\text{persist}=2\Delta x\);
     - otherwise set \(S=0\).
   - Otherwise set \(S=0\) and \(\tau=0\).
   An edge that leaves the material's free surface keeps separating for at
   most two reference cells of travel. Stopping, withdrawing, a blade-normal
   motion, disarming or a changed digit ends the stroke immediately.
5. **Side assignment** (Eq. 4/6 on the actual swept surface): while
   \(S=c+1\), every particle whose digit for \(c\) is still 0 and that lies
   inside the finite band receives a digit. The band is \(|s|\le h=6\Delta
   x\), \(|\ell|\le L\) (the full finite edge half-length),
   \(-b_\text{back}-\max(u_c\Delta t,0)\le f\le 0\) with \(b_\text{back} =
   2\Delta x\), and \(M_{k(\ell)}\). The digit is 1 when \(s<0\) and 2
   otherwise. \(s\) is the signed distance to the surface swept by that edge
   segment: normal \(\propto v_k\times e\), oriented with \(n\), falling back
   to \(n\) for a pure draw (law 3's `_swept_side`). With a raked skim, the
   cut follows the edge's horizontal path, not the tilted blade face.
6. Law 4 runs no lateral, persistence-per-segment or lead closure. Labels
   still change only P2G/MUSL compatibility and contact selection, starting
   with the next substep. No particle is deleted, moved or duplicated.

Why this separates ordinary collision from cutting:
- A side push or flat press moves more than 45° out of the blade plane, so
  \(M_k\) never holds.
- Stopping and withdrawal have no forward or draw motion.
- An edge that never touches material never sets \(W\).
- Blunt parts do no directional work.
- A partial cut followed by withdrawal never labels anything ahead of the
  edge (\(f>0\)), so the ligament stays code 0 and connected.

## Phase 1 acceptance matrix

All rows use GPU at the formal resolution (1 mm particles, 0.1 ms, 4 mm
grid), kinematic ideal tracking unless noted, seed 42, `ruler_knife_v1`. They
are scored with the actual 8-node compatibility graph (`mechanical_components`
with weights and gradients). Every row also reports the label-to-swept-surface
distance distribution, which is bounded by \(h\) by construction. Each sever
row also reports an interface statistic: the frozen midpoint statistic, but
measured against the swept surface the engaged edge actually traced.

| # | Test | PASS when |
| --- | --- | --- |
| M1 | CUT-01-R and CUT-02 reference vertical cut | frozen strict oracle PASS (two parts, mass, location ≤ 2 mm, persistence) |
| M2 | 20° oblique cut: blade tilted 20° about its edge line, moving along d through the CUT-01-R bar | two substantial parts (≥ 40% each, ≥ 99% together); swept-surface interface statistic ≤ 2 mm |
| M3 | TRIM-01-R 25° reference skim (also report the 35° `ruler_workcell_v2` scenario) | frozen `reference_above_plane_material_removed` (≥ 90%); the largest component holds ≥ 99% of the below-plane reference mass. Surface precision is reported separately |
| M4 | Armed vertical cut to half depth (10 mm), then armed withdrawal | one component, with code-0 ligament below the stopping depth |
| M5 | Armed side push and armed flat press | contact force present (above the law-3 test levels); zero new labels; one component |
| M6 | Armed stop (0.5 s at mid-depth) and armed withdrawal | label count unchanged during the stop or withdrawal |
| M7 | Full cut, then withdrawal and hold | separated topology persists ≥ 0.5 s (the frozen persistence check) |
| M8 | Two sequential cuts (`cut_index` 0 and 1) with a lift and move between them | three mechanically independent parts (2-digit graph); lift/move adds no labels |
| M9 | Measured-trajectory replays of Native CUT-01-R and CUT-02 | frozen strict PASS. Lab replays are reported (parity is P1) |

The CPU suite gets small-block law-4 tests for the same behaviours, plus
v4-vs-v3 bitwise equality of laws 1/2/3.

## Consequences

- Law 4 is a process-zone approximation with provisional, uncalibrated
  parameters (\(h, b_\text{back}, b_\text{persist}, \theta_\max\)), not a
  converged fracture model. Any parameter change gets a new law version or
  an explicitly recorded variant, never a silent edit.
- The frozen cut-location check requires the label boundary to coincide with
  the node plane at x = 0, which means an edge within about ±0.5 mm of the
  target. Live demos therefore need the measured edge held near the target
  plane by the controller. The physics and thresholds stay unchanged.
- If a matrix row fails, the failure is preserved and diagnosed with
  `scripts/analyze_cut_bridges.py`, now using actual stencil
  weights/gradients. Any closure added afterwards becomes a new, separately
  tested variant.

## Amendment 1 (2026-10-29, after the law-4 matrix, before any law-5 matrix row)

The law-4 matrix showed that two row definitions in the table above cannot
discriminate the cutting law. Both original verdicts stay recorded as FAIL in
`docs/evidence/phase1/matrix_law4_001.json`. For law 5 and later laws, the
amended criteria below are declared before any of their rows run:

- **M2 (oblique cut).** The frozen midpoint statistic is ≤ 2 mm only for
  planes aligned with the grid. Applying the exact geometric sign of the
  fitted 20° swept plane to the same particles scores 4.44 mm; the law-4
  labels scored 4.78 mm. Every label/geometry disagreement lies within
  1.07 mm of the plane.
  - **Amended criterion:** two substantial parts (unchanged), and an
    interface statistic ≤ the ideal-geometric-label statistic plus one
    particle spacing.
  - In addition, no particle more than two particle spacings from the swept
    plane may sit in the component of the opposite side.
  - The original 2 mm value is still reported.
- **M8 (two sequential cuts).** M8a cut at x = ∓10 mm inside the 40 mm
  pedestal gap. After cut 0, the remaining bar hangs as a cantilever and
  sags 5–8 mm before cut 1 starts. The scripted path, planned for the
  undeformed bar, ends 4 mm below the undeformed bottom, so it stops above
  the sagged bottom layer and leaves it connected. That is a test-geometry
  limitation, not a labelling fault: both digits label correctly wherever the
  edge passes.
  - **Acceptance row:** M8b, two cuts with each piece supported (SLICE-PUSH-01
    on the slotted board). M8a stays a reported diagnostic.
