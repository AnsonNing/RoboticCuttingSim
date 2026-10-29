# ADR 0012 — Cutting law 8: law-5 lead pre-assignment only for plowing (out-of-plane) strokes

Date: 2026-10-29. Status: accepted. The gate is met: computed M1–M9 PASS in
`matrix_law8_001` / `matrix_law8_002`, and the final live set ran afterwards.
Declared before any law-8 row or episode.

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

## Review note (2026-10-29, rcs-review PASS with conditions; no verdict changes)

- **Equivalence wording.** On CPU, law 8 is bitwise equal to law 4 for
  in-plane cuts and to law 5 for constant-rake skims
  (`tests/test_cutting_law_v8.py`). On GPU the rows match only within GPU
  nondeterminism, not bitwise. Repeated GPU runs differ from about 0.1 s
  onward, with equal final codes and |dx| ≤ 0.06 mm. In-plane rows match the
  law-4 rows to 0–8 final codes and 0.03–0.47 mm in position; TRIM rows match
  the law-5 rows to 5–16 codes, with identical or near-identical removal.
- **Validation scope.** Law 8 is validated only for in-plane cuts (0°, plus
  live noise below 9° in contact) and for 25°/35° constant-rake skims. The
  matrix has no motion in the 15–25° range, so θ_plow itself is untested:
  any threshold between about 0° and 25° would give the same verdicts. The
  law switches discontinuously at 15°.
- **Reported failures and limits.**
  - Still reported: M2's original 2 mm FAIL, the M8a diagnostic FAIL, the
    M9c/d Lab report-only rows, and the failures of laws 4–7.
  - Provenance: the M1a/M3a/M3b/M8b inputs were first simulated on an
    uncommitted tree before e19c2b6. They are rerun cleanly as `_002` and
    rescored in `docs/evidence/phase1/matrix_law8_002.json`.
- Commit 38e92a9 says the reruns were "from f7ab36e". Precisely, the two
  probes started at f7ab36e. M3b/M8b recorded 9f89575, a docs-only commit
  made while they ran. The code is identical to e19c2b6, and every rerun was
  clean.

## WP-11 Phase F validation record (2026-10-30)

Phase F2/F3 ran under protocol `configs/wp11/phaseF_protocol_v1.json`, frozen before any
run. All rows used kinematic ideal tracking at `f2168b6` on `cuda:0`, seed 42. Report:
[WP11_PHASE_F_LAW8.md](../reports/WP11_PHASE_F_LAW8.md). Evidence:
[phaseF_material_001.json](../evidence/wp11/phaseF_material_001.json).

- **θ_plow is now measured at the boundary.** ρ = 14° and 15° never plow, and ρ = 16°
  plows on 100% of steps and segments. All three separate.
- **F2:** 15 of 16 boundary rows PASS. F2.9_blunt_handle FAILS: the grip mean force was
  0.077 N against the declared > 0.1 N, because the unrestrained bar yawed away.
- **F3:** the formal CUT and TRIM rows reproduce their PASS. The declared medium row is
  ERROR / NOT_RUN, because the dt is not an integer substep.
- **Not validated:** the chisel (ADR 0013 C4), live or Lab tracking, the sawing swept
  normal under noise, and rotation-only stroke persistence.
