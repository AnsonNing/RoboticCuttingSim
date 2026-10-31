# ADR 0020 — Candidate solver `tlmpm_v6`: grid-consistent position update

Date: 2026-10-31. **Status: proposed (candidate, not adopted); S1 kill stage FAILED on 2026-10-31 (flat-board CUT did not separate on v6, 1 vs 2 components on v5), so the protocol outcome is REJECTED — see [`WP12_TLMPM_V6_CANDIDATE.md`](../reports/WP12_TLMPM_V6_CANDIDATE.md); the xhigh reviewer records the decision.** Source: WP-12 Phase 4 prototype decision P-A
([`WP12_PHASE4_PROTOTYPE_DECISION.md`](../reports/WP12_PHASE4_PROTOTYPE_DECISION.md), commit `b05c7f0`). ADR numbers
0018 and 0019 stay reserved for the cut-aware surface and for any exit law. Acceptance or rejection is an xhigh
decision on the staged protocol `configs/wp12/phase4_gridpos_protocol_v1.json`; until then nothing uses this solver.

## Context

`tlmpm_v5` (ADR 0014) advances particle positions with the FLIP/PIC-blended particle velocity,
\(x \mathrel{+}= \Delta t\, v_p\), while the deformation gradient is updated from the MUSL grid velocity,
\(F \mathrel{+}= \Delta t \sum_I v_I^{MUSL} \otimes \nabla N_I\). The two fields differ by the FLIP null-space
component of \(v_p\), which the grid cannot see. The WP-12 V1–V4 root-cause verification
([`WP12_CUTTING_PHYSICS.md`](../reports/WP12_CUTTING_PHYSICS.md), COUNTERFACTUAL rows, never acceptance) found that
this strain-free drift explains 0.958–0.996 of the first-band \(|dx - \bar F dX|^2\) in 8 of 8 rows, and that a
counterfactual grid-velocity position update removes most of the link scatter while leaving the component counts of
10 of 10 cases unchanged. The same counterfactual also changes forces, tool work, overlap, labels and kinetic energy
(decision findings 1 and 3), so it is a physics change, not a visual one.

## Decision

Build `src/robotic_cutting_sim/tlmpm_v6.py` as an **opt-in candidate**:

- `class TLMPM(tlmpm_v5.TLMPM)`; only `step()` is overridden. Its body is the v5 step launch for launch, plus:
  1. before `_start_cut` and every contact kernel: `x_start <- copy(x)`;
  2. after `_deformation`, before `_update_stroke` / `_assign_band` / `_assign_lead` (law 8 reads the new positions):
     \(x_p \leftarrow x_{start,p} + \Delta t \sum_{c<8,\ slot[p,c]\ge 0} w_{pc}\, v^{MUSL}_{node(p,c)\cdot S + slot(p,c)}\),
     the same field, slots and weights as `_deformation`.
- `v_p` (the FLIP/PIC blend, \(\alpha\) = `config.pic`) is kept unchanged. `_assign_band` keeps `v_p` in its rear
  tolerance; the mismatch is \(\le \Delta t\,|v_p - S v^{MUSL}|\) (about 10 µm per substep) against b_back = 8 mm.
- Option `position_update`: `"grid_musl_v1"` (default of the module) or `"flip_v5"` (identity, tests only; bitwise
  tlmpm_v5 on CPU).
- Receipts record `solver = "tlmpm_v6"`, the option and the sha256 of `tlmpm_v5.py` and `tlmpm_v6.py`.
- Selection is explicit only: `LiveCuttingMaterial(solver="tlmpm_v6")`, `KinematicConfig(solver="tlmpm_v6")`,
  `blade_probe.drive_blade(solver_version=6)`. `packed_solvers.py` names the packed solvers {v5, v6}. Scene configs,
  presets, tasks, the envelope, the benchmark, E2, Phase 5 and every demo keep `tlmpm_v5`; the cut-aware surface stays
  tlmpm_v5-only.

## What stays frozen

`tlmpm_v4.py`, `tlmpm_v5.py` and the V1–V4 counterfactual module byte for byte; laws 4, 5 and 8 and their parameters;
the material config (α 0.05, k 0.1, zone = s, G_c, μ); defaults, presets, `final_v4` and task identities; all records,
protocols and verdicts; envelope v1; benchmark v1; the E2 protocol.

## Classification and wording

A time-integration change that removes strain-free FLIP null-space drift from the positions. It is not a cutting law,
not fracture, not calibration and not "the standard TLMPM update" (Karamelo advects with \(v^+\); this rule uses
\(v^{MUSL}\) so that x and F share one field). If accepted, report only: "particle positions consistent with the
deformation gradient; separation outcomes unchanged; forces changed and still unconverged". Never claim better or
cleaner cutting, a fixed law-8 failure region, force convergence, full x/F consistency (label timing and the CUT
0.5 mm P50 remain) or a surface gain before a Phase 3 v3 protocol. v6 is never selected, accepted or tuned on a
surface metric.

## Acceptance (protocol `phase4_gridpos_protocol_v1`)

- **S0 (CPU, `tests/test_tlmpm_v6.py`).** flip_v5 bitwise v5; grid_musl_v1 bitwise the counterfactual grid positions;
  \(|x - X - u_{grid}| < 10^{-6}\) m; CPU negative controls; free-block momentum; full CPU suite.
- **S1 (kill set).** M1a, M3b, M4, M5a, M5b, M8b, F2.7_withdrawal_v2, F2.9_blunt_handle_v2, flat-board CUT and
  TRIM ρ 10°, each against a V5 arm at the same commit. Any FAIL: REJECTED, kept, no tuning.
- **S2 / S3.** The full law-8 acceptance (matrix, Phase F, Phase R (b), new post-cut controls) and the WP-12 rows.
- **S4 (Native).** Only before any Native use.

## Consequences

- If rejected, the module stays as a recorded negative result; nothing else changes.
- If accepted, adoption into any default is a separate decision with its own envelope version; Completion Criterion 4
  may then be re-examined through the Phase 3 reopen measure, which is "eligible", not "reopened".
- A CPU finding made while building the S0 checks: the 4 mm CPU test block (`blade_probe.block`, s = dx) places
  particles on node planes; zero-mass corners receive the stress-gradient term and the grid update drops it, so total
  momentum is not conserved on that lattice in v4/v5 alike. It is not a v6 effect, and formal 1 mm lattices (node
  fractions 1/8, 3/8, …) have no zero-weight corners. The momentum check uses a 2 mm lattice.
