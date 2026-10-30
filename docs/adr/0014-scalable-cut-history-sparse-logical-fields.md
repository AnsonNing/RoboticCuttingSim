# ADR 0014 — Scalable persistent cut history: packed digits and sparse per-node logical fields (`tlmpm_v5`)

Date: 2026-10-30. Status: accepted for implementation. This is WP-11 P0 from
the user's supplementary requirements (multiple cuts per episode, sequential
removal of protrusions).

## Context

`tlmpm_v4` keeps one ternary digit per cut: 0 means not reached, 1 and 2 are
the two sides. A particle code is \(c=\sum_k d_k 3^k\). Each grid node holds
\(3^K\) logical fields, and the grid update loops over every compatible pair,
so a node costs \(O(3^K)\) memory and \(O(9^K)\) work. The evaluators build a
\(3^K\times3^K\) compatibility table. `LiveCuttingMaterial` allows
`max_cuts` ∈ {1, 2}. The required behaviour does not fit that:

- three or more slices give four or more pieces;
- three protrusions must be removed one after another, one cut each;
- mixed tasks combine several of these.

| K | Fields per node | Pairs per node |
| --- | --- | --- |
| 2 | 9 | 81 |
| 3 | 27 | 729 |
| 4 | 81 | 6561 |

Only a handful of distinct codes ever meet at one reference node, because a
node's support touches at most the 8 surrounding cells. Most of the \(3^K\)
fields are therefore empty.

## Decision

Add a new solver module, `tlmpm_v5.py`. `tlmpm.py`, `tlmpm_v2.py`,
`tlmpm_v3.py` and `tlmpm_v4.py` stay unchanged, so every historical law and
result remains reproducible. `tlmpm_v5` implements law 8 (and law 4/5 for
reference) with two representational changes. None of the cutting rules
change.

1. **Packed digits.** Each particle's code is one int32 holding 2 bits per
   cut: \(d_k = (c \gg 2k) \,\&\, 3\), with 0 = unassigned, 1 = negative side,
   2 = positive side, and 3 invalid. This allows \(K_\max = 15\) cuts.
   - Conflict test: two codes conflict when they lie on opposite sides of
     some cut. With \(A_1\) the mask of cuts where \(a\) has digit 1 and
     \(A_2\) where it has digit 2 (likewise \(B_1, B_2\) for \(b\)), the codes
     conflict iff \((A_1 \,\&\, B_2) \,|\, (A_2 \,\&\, B_1) \ne 0\). This is
     O(1) and needs no table.
   - The semantics are exactly v4's ternary-digit semantics (BladeMaster
     Eq. (7)). Only the encoding differs.
2. **Sparse per-node logical fields.** Each reference node has \(S\) slots
   (default \(S = 8\)). Every substep, a slot-claim kernel assigns each
   (particle, stencil-corner) pair to the slot holding its code at that node,
   using `wp.atomic_cas`. The pair is stored as `slot[p, corner]`.
   - P2G, G2P, MUSL and the deformation update address `node·S + slot`
     instead of `code·N + node`.
   - The grid update loops over the occupied slots of a node only and tests
     compatibility with the bitmask test. That is \(O(S^2)\) per node,
     independent of \(K\).
   - Overflow (more than \(S\) distinct codes at one node) sets a device flag
     and is reported as an error. It is never silently merged.
   - Memory is \(O(S\cdot N)\) for any \(K \le 15\).
3. **Evaluators** (`cut_topology.mechanical_components`, multi-cut
   evaluation, bridge analysis) gain a packed-code path. It uses the same
   per-node grouping and union-find as today, with the bitmask conflict test
   replacing the \(3^K\) table. The ternary v1–v4 paths stay unchanged.
4. **Cut indices.** Each cut gets its own digit index `cut_index` ∈
   [0, K). A digit can only go from 0 to 1 or 2. Earlier cuts persist,
   including partial ones. The runtime and policy API accept
   `cut_index < max_cuts` with `max_cuts ≤ 15`.

## Validation (before any multi-cut live run)

1. **Equivalence with v4 for K ≤ 2, law 8, same inputs.**
   - CPU: positions, velocities, codes (decoded to ternary), F, progress and
     reactions equal v4 to float rounding (the summation order changes, so
     the result is not bitwise). The tolerance is stated in the test.
     *Correction (2026-10-30, WP-11 review finding 11):* the expectation in
     parentheses was wrong. The observed CPU result is bitwise equal to v4
     (Phase M evidence `phaseM_tlmpm_v5_001.json`, `cpu_tests`), and
     `tests/test_tlmpm_v5.py::test_v5_bitwise_equals_v4_on_aligned_push`
     asserts bitwise equality of positions, velocities, F and forces for the
     aligned push. The requirement stays the stated tolerance; bitwise
     equality is an observed, stronger property on CPU, not a GPU guarantee
     (GPU float32 atomics are nondeterministic).
   - GPU 1 mm: final codes and components match v4 to within GPU
     nondeterminism on the CUT-01-R and M8b rows.
2. **Scaling.** Memory and substep time are measured for K = 1…8 on the 1 mm
   bar.
3. **Multi-cut physics** (1 mm, kinematic).
   - Test B: 3 and 5 sequential slices give 4 and 6 mechanically independent
     pieces, with every cut keeping its history.
   - Test C: three protrusions are removed by three successive skims
     (3 digits), checked per protrusion; the substrate stays intact.
   - Overflow never fires; the maximum occupied slots per node is reported.
4. **Negative controls.** All of them (side push, flat press, no contact,
   stop, armed withdrawal, half-depth cut, blunt handle) hold under `tlmpm_v5`
   law 8.
   *Status (2026-10-30, WP-11 review, Phase R):* shown at formal resolution
   (1 mm, 4 mm, 0.1 ms; kinematic, seed 42) for single-digit histories at
   K = 5 only: the top digit (4) armed, digits 0–3 forced to zero. 7 / 7 rows
   PASS on the tasks, plans and criteria of their tlmpm_v4 counterparts
   (`docs/reports/WP11_REVIEW_NEGATIVE_CONTROLS.md`,
   `docs/evidence/wp11/phaseR_negative_controls_001.json`). Negative controls
   after earlier real cuts (nonzero lower digits) are NOT_RUN. Before this only
   the CPU unit tests on a coarse block covered the item.

## Consequences

- Live runtimes select `tlmpm_v5` only explicitly, through a solver/law
  option recorded in receipts. `final_v4` keeps `tlmpm_v4`.
- The practical cut capacity is \(K \le 15\) and \(S\) distinct codes per
  node. Both limits are validated and documented, not assumed.
