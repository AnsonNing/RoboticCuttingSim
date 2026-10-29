# ADR 0008 — Cutting law 5: law 4 plus lead pre-assignment outside a one-cell core

Date: 2026-10-29. Status: accepted for implementation; acceptance is the
[ADR 0006](0006-swept-band-cutting-law-4.md) Phase 1 matrix, rerun with law 5.

## Context

Law 4 is preserved unchanged, together with its failures:

- 1 mm CUT-01-R: strict PASS (72/72 g, location 1.5 mm).
- 1 mm TRIM-01-R (25°): FAIL, 0% removal
  (`artifacts/wp10_claude/phase1_law4/k_trim01r_law4_001`,
  [evidence](../evidence/phase1/law4_smokes_001.json)).
- Law 4 removes the substrate-side bridges that law 3 left. Only two code-0
  particles remain. Their reference positions are (4.5, ±2.5, 24.5) mm,
  4.3 mm above the swept plane z = 20.2 mm.
- Before the edge reached them, the bump front pushed these two particles
  ahead of the edge, and they finished near x = 74 mm. They were never behind
  the edge, so Eq. (6) never assigned them.
- The raked edge advances forward-down and pushes 34 bump-bottom particles
  (reference z = 20.5 mm) below the swept plane before passing. Those
  particles correctly join side 1.
- Through the shared reference node at z = 24 mm, the two code-0 particles
  couple the side-2 chip to these side-1 particles, and through them to the
  substrate.
- At 35° on the CPU test block, carried bump material leaves the whole
  bump connected.

BladeMaster's Eq. (4) band already includes a lead extent \(b_\text{lead}\),
but Eq. (6) assigns only behind the edge. Carried or ejected chip material is
exactly what that leaves unassigned.

## Decision

Add `cutting_law=5` to `tlmpm_v4.py`. Laws 1–4 stay bitwise unchanged
(no extra launches). Law 5 is law 4 with one additional assignment rule.

**Lead pre-assignment.** While the law-4 stroke is active (\(S=c+1\)), a
particle is assigned as soon as the edge comes within the lead extent of it,
under these conditions:

- the particle's digit for \(c\) is still 0;
- it lies in the lead region \(0 < f \le b_\text{lead} = 2\Delta x\) of a
  segment with \(M_k\), with \(|\ell| \le L\);
- it is clearly off the swept surface: \(s_\text{core} < |s| \le h\), where
  the core half-width \(s_\text{core} = \Delta x\) (one reference cell).

It receives the digit of its side, using the same swept-surface \(s\) as
law 4. Particles in the core band \(|s| \le \Delta x\) are still assigned
only behind the edge, exactly as in law 4.

Why this cannot sever material the edge has not reached:

- Before the edge arrives, the only new labels are at distance
  \(> \Delta x\) from the swept surface, one side each.
- Opposite-side pre-assigned particles are then more than \(2\Delta x\) apart
  along the normal, so for planes aligned with the grid axes they share no
  stencil node.
- The core band of code-0 particles, compatible with both sides, continues
  to couple them at every node in between.
- A partial-cut ligament ahead of the edge therefore stays mechanically
  connected.
- The label boundary at the cut (and so the frozen cut-location statistic)
  is still set only by core particles after the edge has actually passed.
- Negative controls are unchanged: without an active stroke nothing is
  assigned, so side push, flat press, stopping, withdrawal and uncontacted
  edges add no labels.

What it fixes: material more than one cell off the swept surface is
labelled when the edge comes within \(2\Delta x\), before it can be carried
away. That covers the bump material above the plane and the substrate below.

## Parameters and versioning

The parameters are \(h = 6\Delta x\), \(b_\text{back} = 2\Delta x\),
\(b_\text{persist} = 2\Delta x\), \(\theta_\max = 45^\circ\),
\(b_\text{lead} = 2\Delta x\) and \(s_\text{core} = \Delta x\). They are
provisional and uncalibrated, and they are recorded in every receipt. If law 5
fails a matrix row, the failure is preserved, and any change becomes another
explicitly versioned variant.

## Acceptance

The full ADR 0006 matrix (M1–M9) is rerun with law 5. Law 4 results remain
preserved as their own evidence.

CPU tests:
- v4 laws 1–4 are bitwise unchanged;
- the law-4 negative controls pass for law 5 too;
- raked skims at 25° and 35° separate the bump by the stencil graph
  (≥ 90% of above-plane particles off the base component, ≥ 99% of
  below-plane particles in it);
- a half-depth cut keeps its ligament;
- pre-assigned particles always satisfy \(|s| > \Delta x\) at assignment.
