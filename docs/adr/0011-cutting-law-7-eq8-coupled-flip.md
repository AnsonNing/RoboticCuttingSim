# ADR 0011 — Cutting law 7: law 5 labels with BladeMaster Eq. (8) coupling of pⁿ for FLIP

Date: 2026-10-29. Status: accepted for implementation, declared before any
law-7 row or episode.

## Context

The plan's §2 gate requires every matrix row to pass before live demos. Two
recent laws each fail a different row:

| Law | Fails | Mechanism |
| --- | --- | --- |
| 5 | M8b (computed `adr0006a1_adr0008`) | Labels assigned ahead of the edge increase how far cut 0 pushes the remaining bar (~2.3 mm vs ~1.2 mm under law 4), so the second declared plane is missed by one particle layer |
| 6 ([ADR 0010](0010-cutting-law-6-deferred-lead-commit.md)) | 1 mm TRIM-01-R at 35° (`ruler_workcell_v2`): one component, 0% removed (`docs/evidence/phase1/law6_smokes_001.json`) | Without labels during the cut, a carried bump particle (reference (8.5, 0.5, 21.5) mm) is pushed below the plane and bridges |

The Phase 1 review traced law 5's extra push to transfer dynamics rather
than topology. Nodes that hold both digits are identical under laws 4 and 5.
What changes is that compatible fields at one node carry different FLIP
reference velocities.

In `tlmpm_v4.py` `_grid_update`, the new nodal velocity is computed from the
coupled \(p+\Delta p\), but the FLIP reference is the uncoupled
`old = p_c/m_c`. BladeMaster Eq. (8) couples the whole state
\(S = (m, p^n, \Delta p)\) between compatible logical grids: \(\bar S^a = S^a +
\frac{1}{M}\sum_{b\ne a, \lnot\text{conflict}} (m^a S^b - m^b S^a)\). The
consistent FLIP reference is therefore \(\bar p^{n,a}/m^a\). Our solver
departs from Eq. (8) exactly where labels split material into fields that
are still compatible.

## Decision

Add `cutting_law=7` to `tlmpm_v4.py`:

- **Labels:** identical to law 5 (law 4 stroke and band, plus law 5 lead
  pre-assignment).
- **Grid update:** a separate grid-update kernel, used only by law 7, also
  couples \(p^n\) as in Eq. (8). The kernel computes `old[c,i]` as
  \(\bar p^{n,c}_i/m^c_i\), with \(\bar p^n\) formed by the same pairwise
  antisymmetric exchange that already produces the coupled
  \(p^n+\Delta p\).
- **What follows:**
  - Two mutually compatible fields at a node then behave exactly like one
    merged field, for both the PIC and FLIP parts.
  - Incompatible pairs (an actual cut) stay decoupled.
  - Momentum is conserved as before, because the exchange is antisymmetric.
- **Scope of the change:** the MUSL remap and deformation update are
  unchanged; they already use the coupled velocity. Laws 1–6 keep the
  existing kernel and stay bitwise unchanged.
- **Parameters:** identical to law 5. It remains a provisional,
  uncalibrated process-zone model with no deletion.

## Acceptance (gate for live demos)

1. **Phase 1 matrix:** the full ADR 0006 matrix (with Amendment 1) is run
   with law 7 and scored by `scripts/rescore_phase1_matrix.py` under the
   computed criteria `adr0006a1_adr0008`. The M8 verdict is the M8b
   topology check AND the runner's slice-push criterion. Every row must
   pass before any law-7 live episode counts. M8a is a diagnostic and the
   Lab replays (M9c/d) are report-only.
2. **Quick check first:** before the full matrix, run three rows: M8b, M3b
   (35°) and M3a. If any fails, stop: preserve and diagnose the failure, and
   use no law-7 run live.
3. **CPU tests:** laws 1–6 bitwise unchanged; for two compatible fields at
   a node, law 7's grid update equals the merged single-field result to
   rounding; the law-5 negative controls hold under law 7.
