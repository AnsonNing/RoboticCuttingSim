# ADR 0012 — Cutting law 8: law-5 lead pre-assignment only for plowing (out-of-plane) strokes

Date: 2026-10-29. Status: accepted for implementation. Declared before any
law-8 row or episode.

## Context

The plan's §2 gate requires the whole matrix to pass before live demos.
Each law so far fails a different row (computed criteria
`adr0006a1_adr0008`):

| Law | Fails | Source |
| --- | --- | --- |
| 4 | M3: TRIM 0% removed (a chip carried ahead of the raked edge stays code 0) | `matrix_law4_002.json` |
| 5 | M8b: cut 0 pushes the bar about 2.3 mm (law 4: 1.2 mm), so the second declared plane misses by 2.0000000670 mm against a 2 mm bound | `matrix_law5_002.json` |
| 6 ([ADR 0010](0010-cutting-law-6-deferred-lead-commit.md)) | TRIM at 35°: one component | `law6_smokes_001.json` |
| 7 ([ADR 0011](0011-cutting-law-7-eq8-coupled-flip.md)) | M8b (3 mm), M3a (89.8%), M3b (0%) | `law7_quickcheck_001.json` |

The two kinds of row need different things:

- **In-plane cuts** (M1, M2, M4–M9) pass under law 4. Law 5's lead
  pre-assignment is unnecessary there, and in practice it changes the cut
  mechanics.
- **The raked skims** (M3) need the lead rule. There the edge moves 25–35°
  out of its own plane, so the rake face lifts and plows material ahead of
  the edge. That carried material is exactly what the lead rule labels
  ([ADR 0008](0008-cutting-law-5-lead-preassignment.md)).

An edge that moves within its plane displaces material only by its wedge
thickness and carries nothing ahead.

## Decision

Add `cutting_law=8` to `tlmpm_v4.py`. Laws 1–7 stay bitwise unchanged and
launch nothing new. Law 8 is law 4 plus law 5's lead rule, with the lead rule
applied only on plowing segments:

- **Plowing segment.** In addition to \(M_k\) (cutting motion within 45° of
  the blade plane), segment \(k\) is plowing when its motion leaves the blade
  plane by at least \(\theta_\text{plow} = 15^\circ\):
  \((v\cdot n)^2 \ge \tan^2\theta_\text{plow}\,[(v\cdot d)^2+(v\cdot e)^2]\).
- **Lead pre-assignment** (law 5, [ADR 0008](0008-cutting-law-5-lead-preassignment.md))
  runs only for particles whose segment is cutting and plowing. It has the
  same condition and parameters as law 5: \(0<f\le 2\Delta x\) and
  \(\Delta x<|s|\le 6\Delta x\).
- **Everything else is law 4:** contact, progress, engagement, stroke and
  band assignment.

Choice of \(\theta_\text{plow}\):

- It separates the two motion classes that occur: in-plane cuts at 0°, plus
  measurement noise in live runs, and rake skims at 25° and 35°.
- It was fixed before any law-8 result. It is not fitted to a statistic.
- It is a provisional, uncalibrated process-zone parameter like the others.
  Any change would be another explicitly versioned variant.

Expected (to be verified, not assumed):

- Under ideal tracking, in-plane rows run exactly as under law 4, and the
  constant-rake TRIM rows run exactly as under law 5.
- Under live tracking noise, an in-plane cut can briefly read as plowing. It
  then degrades toward law 5 behaviour, which the law-5 live runs show is
  benign.

## Acceptance (gate for live demos)

1. **Full matrix.** Run the full ADR 0006 matrix (Amendment 1) with law 8,
   scored by `scripts/rescore_phase1_matrix.py` under the computed
   `adr0006a1_adr0008` criteria. The M8 verdict is the M8b topology AND the
   runner's slice-push criterion. M8a is a diagnostic; M9c/d are report-only.
2. **Gate.** Every row must PASS before any law-8 live episode counts.
3. **Final demos.** Only then rerun the final live demo set with law 8.
4. **CPU tests.**
   - Laws 1–7 stay bitwise unchanged.
   - Law 8 equals law 4 bitwise on an in-plane through cut.
   - Law 8 equals law 5 bitwise on a constant-rake skim (rake ≥ 15°).
   - All negative controls hold.
