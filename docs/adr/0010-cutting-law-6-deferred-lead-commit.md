# ADR 0010 — Cutting law 6: law 5 lead decisions, committed only when the stroke ends

Date: 2026-10-29. Status: accepted for implementation. Declared before any
law-6 row or episode.

## Context

The plan (§2) requires every row of the 1 mm matrix to pass, negative
controls included, before live demos. Under the computed criteria
`adr0006a1_adr0008` (`docs/evidence/phase1/matrix_law5_002.json`), law 5
passes M1–M7 and M9 but **fails M8**:

- The acceptance row M8b (SLICE-PUSH-01, ideal tracking) scores 2.0000000670 mm
  on the second-cut declared plane, against a frozen 2 mm bound.
- Cause: cut 0 pushes the remaining bar about +2.3 mm. Law 4 pushes it only
  about +1.2 mm and passes M8b.

The live demos were run under the review's conditional acceptance, so the
§2 gate was not met. Those runs stay recorded as law-5 evidence; they do not
count as the gated acceptance run.

The Phase 1 review shows the extra push is dynamic, not topological. The
number of nodes holding both digits is the same under laws 4 and 5. The
difference is that law 5 assigns labels ahead of the edge during the cut,
which splits ahead-of-edge material into separate logical grid fields:

- Each field keeps its own `old` velocity, `old = p_c/m_c`
  (`tlmpm_v4.py` `_grid_update`), while the coupled `new` velocity is the
  merged one.
- With FLIP 0.95, the field split therefore changes the transfer.

Labels ahead of the edge exist only to stop carried or ejected chip material
from staying code 0 and bridging after the cut. They have no reason to alter
the mechanics while the edge is still cutting.

## Decision

Add `cutting_law=6` to `tlmpm_v4.py`. Laws 1–5 stay bitwise unchanged and
launch nothing new. Law 6 is law 5 with the lead rule deferred:

1. **Pending decision.** Under exactly law 5's lead condition (active stroke,
   digit 0 for the armed cut, segment with \(M_k\), \(0 < f \le 2\Delta x\),
   \(\Delta x < |s| \le 6\Delta x\)), record a pending side for the particle
   instead of writing its code. The pending value is the cut digit and the
   side from \(s\). A later qualifying substep may overwrite it.
2. **Commit at stroke end.** When the stroke state goes from \(c+1\) to 0
   (stopping, withdrawal, a blade-normal motion, disarming, a changed digit,
   or travel without work exceeding \(2\Delta x\)), every particle with a
   pending side for \(c\) whose digit for \(c\) is still 0 gets the pending
   digit. All pending values for \(c\) are then cleared.
3. **Everything else is law 4.** Engagement, stroke and behind-the-edge band
   assignment are unchanged, so the band rule assigns particles the edge
   actually passes as usual. While a stroke is active, law 6 creates no label
   ahead of the edge. Its mechanics during the cut equal law 4's.
4. **No other changes.** The parameters are law 5's: \(h = 6\Delta x\),
   \(b_\text{back} = b_\text{persist} = b_\text{lead} = 2\Delta x\),
   \(s_\text{core} = \Delta x\), \(\theta_\max = 45^\circ\). No particle is
   deleted, moved or duplicated, and labels change only compatibility and
   contact selection from the next substep.

Expected consequences, verified by the matrix rather than assumed:

- **Through cuts** behave as under law 4: nothing pending survives behind the
  edge.
- **TRIM:** chip particles carried ahead keep their pending side 2, which is
  committed when the skim stroke ends.
- **Partial cuts** (M4/M6): pending off-core labels below the stopping depth
  are committed at withdrawal. The core band stays code 0, so the ligament
  stays connected (ADR 0008 definition).

## Acceptance (gate for live demos)

Run the full ADR 0006 matrix (Amendment 1) with law 6 under the computed
criteria `adr0006a1_adr0008`, i.e. `scripts/rescore_phase1_matrix.py`. That
means M1–M9 rows, with M8 scored on M8b, its verdict being ADR 0006 topology
AND the runner's slice-push criterion.

Every row must PASS before any law-6 live episode counts. M8a stays a
reported diagnostic, and Lab replays are report-only.

The CPU tests must show that laws 1–5 are bitwise unchanged, that law 6 never
labels ahead of an active edge, and that the law-5 negative controls hold.

If any row fails, law 6 is not used live. The failure is preserved and
diagnosed.
