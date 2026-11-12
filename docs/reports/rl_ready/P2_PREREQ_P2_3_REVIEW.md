# P2-3 prerequisites, moving-clamp energy and P2-3 viscoelasticity — xhigh acceptance review (2026-11-11)

**Scope.** `git log 3a43d1b..025ecaf` on `claude/rl-rubber-p23` (3a43d1b = the last accepted rubber commit, already in
`claude/rl-ready-rubber-physics`; 025ecaf = the head of `claude/rl-rubber-p23`). Three work packages:

1. P2-3 prerequisites in the batched path (`claude/rl-rubber` b922665 … 1b21801, ADR 0023 §20): batched F8 / F9 / F10,
   byte-wise L0, the `motion_table_v1` default BC launch, the F8 cross-group refusal, GPU run metadata.
2. Moving-clamp energy (ADR §21.1): root cause, `constraint_work = kick_average_v1`, protocol
   `rubber_moving_grip_tests_v1` (ea32f2a), RUBBER-GRIP-01..04.
3. P2-3 viscoelasticity `neo_hookean_jbar_prony_v1` (ADR §21.2–§21.7): implementation, `rubber_visc_tests_v1`
   (`95ad1f94…`, 7 PASS / 2 FAIL) and `rubber_visc_tests_v2` (`212fe7af…`, VISC-03 / -05 re-scored, PASS).

**Method.** Read-only review in the detached worktree `RoboticCuttingSim-rl-rreview` at 025ecaf (no code edited).
Inputs: AGENTS.md, ADR 0023 §20–§21, `P2_3_RUBBER_VISCO.md`, `P2_2_RUBBER_HYPERELASTIC.md` §11, the previous review
`P2_0_P2_2_ACCEPTANCE_REVIEW.md`, the three protocols, the evidence `p2_3_moving_grip_001.json`,
`p2_3_visco_001.json`, `p2_3_visco_002.json`, every changed module and test, and the raw outputs under
`/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/{p2_3_prep,p2_3}/`. Independent checks: recomputation of
RUBBER-VISC-03 / -05 v2 from the raw series with my own estimator and rod transfer, a byte comparison of every v2 raw
column with v1, numpy probes of the Prony update, the Maxwell energy identity and the branch-work quadratures, three
CPU probes of claims the committed tests do not cover, a trial merge into the integration head, and the CPU suites in
both CUDA modes. No GPU job (GPU0 was only made visible to the CPU suites, under its lock; GPU1 untouched).

## 1. Verdicts

| Package | Verdict | Basis |
|---|---|---|
| P2-3 prerequisites in the batched path (§20) | **PASS_WITH_LIMITATIONS** | Batched F8 / F9 / F10 byte-wise equal to the single reference on CPU (N 1 / N 3 heterogeneous, case A / C / block, resets); cross-group refusal correct; GPU metadata complete. Limitations: R1 (the new default `motion_table_v1` keeps a stale table when substeps run after `begin_control_step()` or a full `reset()` without `set_group_motions()`), R7 (GPU equivalence of the BC launches not shown), R8 |
| Moving-clamp energy (§21.1) | **PASS** | Root cause verified (the end-velocity quadrature of a projected constraint is not the conjugate of the discrete kinetic energy); `kick_average_v1` is bookkeeping only (byte-identical state, tested); protocol committed before its run; GRIP-01..04 PASS with first-order convergence. Notes: R11 |
| P2-3 viscoelasticity (§21.2–§21.7) | **PASS_WITH_LIMITATIONS** | Formulation, exact exponential update, float64 state, `params64`, masked reset and energy books correct; v1 7 PASS / 2 FAIL preserved; v2 changes are genuine measurement / quadrature corrections with unchanged thresholds and fit / held-out split, and reproduce from the raw data. Limitations: R2 (v2 is a post-hoc re-scoring whose outcome was computed on v1 data before declaration), R3, R4, R5, R7 |

**Merge: GO** for 025ecaf (plus this review) into `claude/rl-ready-rubber-physics`. The trial merge onto 36b8947 is
clean (no file changed on both sides except the identical P2-0..P2-2 review). On the merged tree the rubber and batched
CPU suites pass (§6). No Phase 1 frozen file, `configs/rl_benchmark/*`, `scripts/rl_ready/*` or rubber frozen file
changes. R1 is not a merge blocker (every caller in the repository steps through `advance()` or calls
`set_group_motions()` first), but it must be fixed before P2-6 builds a vector env on the batched rubber API.

## 2. Findings (most severe first)

**R1 — MEDIUM. `motion_table_v1` (the new batched default) replays the previous control step's motion when substeps
run after `begin_control_step()` or a full `reset()` without `set_group_motions()`.**
- Where: `src/robotic_cutting_sim/batched/rubber_material_batch.py:463-468` (`begin_control_step` copies q1 → q0 but
  does not refresh the table), `:640-687` (`reset_masked` resets the group arrays, not the table), `:470-486`
  (`set_group_motions` is the only refresh after construction); `src/robotic_cutting_sim/batched/rubber_batch.py:449-455`
  (`prepare_motion_table`), `:194-232` (`_bc_table_b` reads the table whenever `index + 1 <= substeps`).
- Scenario (CPU probe, N = 1 case A, 39 substeps): `advance(translation = +0.2 mm z)`, then `begin_control_step()` and
  39 direct `solver.step()` calls. In `all_nodes_v1`, and in the single reference whose `begin_control_step` "makes
  every group hold", this is a hold. With `motion_table_v1` the grip moves another 0.2 mm: particle positions differ
  by max |Δx| = 2.0e-4 m. Same result (2.0e-4 m, N = 2) for a full `reset()` (which zeroes the shared substep index)
  followed by direct substeps. A partial reset followed by direct substeps happens to be safe: the shared index is past
  the table and the kernel falls back to `group_target`.
- Impact: `advance()` always calls `set_group_motions()` and is unaffected; every committed test and formal run steps
  through it (the BC micro-benchmark calls `set_group_motions()` itself), so the recorded results stand. The ADR §20.2 statement "every variant is
  byte-identical to `all_nodes_v1`" holds for the `advance()` path only. The default path has a silent wrong-BC mode for
  a public method sequence that is valid for the previous default.
- Recommendation: call `prepare_motion_table()` at the end of `begin_control_step()` and of `reset_masked()` (one small
  launch), or keep a dirty flag that `step()` checks; add tests for "begin + direct steps" and "full reset + direct
  steps" against `all_nodes_v1`. Fix before P2-6.

**R2 — LOW-MEDIUM (protocol). `rubber_visc_tests_v2` is a post-hoc re-scoring; its outcome was known before it was
declared.**
- The two corrections are genuine (§4.3, §4.4): the v1 last-cycle estimator and the grip-1-reaction-vs-gauge-strain
  comparison bias the exact solution as much as the simulation, and the left-point branch work is an O(dt) quadrature
  of the exact branch history. Thresholds, rates, the fit / held-out split, the Prony sets, the scenes, the step rule,
  `control_dt` and every solver option except `visco_energy` are unchanged (asserted by
  `tests/test_rubber_p2_3_protocols.py:68-88`); the de-embedding uses only ρ, L, the marker positions and ω.
- But the DIAGNOSTIC re-analysis `visc_diag_v1_reanalysis_001` (b9baa57, 00:14 KST) computed the v2 result on the v1
  data — de-embedded E′ −0.40…−0.41 %, identification passing, held-out 32 Hz E′ −0.08 % / E″ +1.83 % — before v2 was
  declared (dca4848, 00:16:54). The held-out rates are held out from the identification, not from the protocol design.
  v2 records this (`provenance.seen_before_declaration`), and the v1 FAILs stand.
- Recommendation: cite the result as "VISC-03 / -05: FAIL under v1; PASS under v2 (post-hoc measurement correction)",
  never as a blind pass. The next blind evidence for the rate dependence should come from rates or scenes not seen
  (P2-5, or a v3 with a fresh held-out rate and the corrected time stamp of R3).

**R3 — LOW. The declared wrench time stamp is dt/2 late; it biases E″ by up to −0.45 % at 32 Hz.**
- The control-step wrench is the mean of the kicks at t_{k−1} + j·dt, j = 0…n−1, centred at t_k − Δt_c/2 − dt/2, not
  at the declared t_k − Δt_c/2. My recomputation from the `visc_v2_001` raw series: de-embedded E″ −0.435, −0.443,
  −0.467, −0.511, −0.625, −0.867 % (1…32 Hz) with the declared stamp; −0.411, −0.411, −0.412, −0.395, −0.404,
  −0.419 % re-stamped, equal to E′ and the energy-based loss. The predicted shift (ω dt/2)/tan δ is +0.02…+0.46 %.
- Correctly not acted on (found after the run; E″ passes its 3 % / 4 % limits). Recommendation: adopt the corrected
  stamp in the next protocol that scores E″.

**R4 — LOW. The committed RUBBER-VISC-09 tests cover only the full option set.**
- `tests/test_rubber_visco_solver.py:23-24, 84-113` and `tests/test_rubber_visco_batched.py:192-212` run F8 +
  `kick_average_v1` + F9 + F10 only. `TLMPMRubberVisco.step()` (`tlmpm_rubber_visco_v1.py:293-363`) duplicates the
  frozen launch sequence including the lumped-clamp, float32-state and particle-velocity-work branches; those branches
  are not compared with the frozen `TLMPMRubber` in any committed test. A `LiveRubberMaterial` with Prony terms and the
  default options builds exactly that path.
- My CPU probe (default options, guard on, table plane, moving grip with rotation and traction, 5 control steps): M = 2
  with every g = 0 and M = 0 are byte-identical to the frozen `TLMPMRubber` in every device array and the wrench, for
  case A, case C and the block. Recommendation: add the default-option parametrisation to the VISC-09 tests.

**R5 — LOW. The visco guard holds by 0.9 % and its stated premise is false.**
- `visc_tangent_check_001`: max ratio 0.9910 over 2400 states; the isochoric material part is not PSD (min eigenvalue
  −1.26e5 Pa), so the Weyl argument of ADR §21.2.6 does not apply; the bound holds through the margin of the elastic W
  bound, with the worst states at J ≈ 2.6–2.9 (far outside the physical range of β = 20 rubber). The dynamic runs are
  well inside (CFL ≤ 0.5 step rule, 0 guard violations). Recommendation: a guard v2 with an explicit bound on the
  material part before visco scenes with large strain (P2-4 fold, P2-6), as the P2-3 report §6 already anticipates.

**R6 — INFO. What the v2 books test.** With the kick-paired W_b, B_N closes for any Prony update: it tests the
equilibrium energy, the transfers and that the booked branch PK1 is the one applied (the declared layer N). The branch
itself is checked by -01 / -03 / -04 against references, by (d) ΔD / ΔW_b, and by the reported identity
W_b − (D + Ψ_v) (cantilever ≤ 1.0e-5 of S_N, strips +0.07…+0.32 %, the O(ε) term). Scoring that identity in a later
protocol would make the layer-M check explicit.

**R7 — INFO. GPU.** The byte identity of `motion_table_v1` with `all_nodes_v1`, of the batched options with the single
reference and of the visco path with the elastic path is shown on the Warp CPU device only. On the GPU only L2
invariants were run (`gpu_smoke_003`, `visc_cost_001`); FMA contraction can differ between the table kernel and the
inline `group_target`, and the float64 reaction / work / dissipation atomics are order-nondeterministic. No GPU
strip / cantilever run exists (NOT_RUN).

**R8 — INFO. Batched wrench signed zero.** `_wrench_b` (`rubber_material_batch.py:118-127`) now writes −0.0 for a zero
reaction (+0.0 before), matching the single reference. The default batched output changes in signed-zero bits only.

**R9 — INFO. Motion-table memory.** (substeps + 1)·N·G·96 B: 1.9 MB for workload E at N = 4, about 0.98 GB at
N = 1024, G = 2 (4,997 substeps per control step). Not in the reported VRAM rows (N ≤ 8); report it in the P2-6 cost
rows.

**R10 — INFO.** `test_hereditary_integral_step_limit_and_linear_ramp` raises two SciPy `IntegrationWarning`s
(round-off, 200 subdivisions) from `rubber_visco_reference.py:188` on its step-limit case; the test passes and the
formal -01 errors (≤ 3.0e-6) are far below the 1e-4 limit.

**R11 — INFO. Moving-grip thresholds.** M ≤ 1 % and |H| ≤ 0.2 % were set after the diagnostic of S1 (0.38 % / 0.23 %),
which the protocol states (`seen_before_declaration: ["S1"]`). The unseen S2 / S3 give 0.06–0.09 %, and -02 (ratio
0.49–0.52 at half the step) is the stronger evidence that the residual is the O(dt) integrator error.

## 3. P2-3 prerequisites in the batched path (§20)

- **F8 / F9 / F10 batched** (`rubber_batch.py`, `rubber_layout.py:261-318`): the single solver's `_proj_rhs`,
  `_proj_solve`, `_proj_particles`, `_deformation_f64`, `_positions_f64` are launched unchanged over the concatenated
  rows; `_proj_reaction_b` and `_traction_work_grid_b` differ from the single kernels only in the per-env target. Block
  inverses are the single solver's bits; a reset with another mass rebuilds them with the single solver's
  `projection_blocks` (host, cached per (env, mass)). Per-env state isolation: rows, blocks and inverses are disjoint
  per env; option arrays are in the masked reset and the vars()-complete classification. Byte-wise L0 tests pass.
- **Byte-wise L0** (`_same_bytes`: dtype, shape, raw bytes): correct; the P2-2 erratum (value-exact, signed zero not
  distinguished) is accurate and the records are unchanged.
- **`motion_table_v1` default:** `_motion_table_b` (`rubber_batch.py:177-191`) executes the same statements as
  `tlmpm_rubber_v1.group_target` before M X + b, so on the CPU device the BC values are the same bits; the tests compare
  every snapshot array, wrench and summary against `all_nodes_v1` with two grips, rotation, partial masks, an inactive
  grip, the affine block and past-table steps (`tests/test_rubber_batched_options.py:264-321`). Not covered: R1. The
  GPU cost claims (BC pair 0.0806 → 0.0185 ms; end-to-end ratio median 0.804) are interleaved A/B rows with the
  workload stated (workload E, N = 4, 41,220 particles/env, dt 4.0e-6 s, 4,996 substeps, graph chunks, no cameras,
  SHARED_GPU).
- **F8 cross-group refusal** (`rubber_layout.py:261-274, 286-295`): counts particles whose stencil holds constrained
  nodes of two groups in the same component with nonzero weight; the batch refuses them. Correct and conservative
  (inactive groups are counted too). The frozen single-env solver still accepts such scenes (no current scene has one).
- **GPU metadata** (`scripts/rubber/p2_3_prep/run_meta.py`): commit, dirty flag, driver, protocol SHAs and co-tenant
  samples are recorded (`gpu_smoke_003/meta.json`, `visc_cost_001/meta.json`).

## 4. Physics checks

### 4.1 Moving-clamp energy and the kick-average work (§21.1)

The argument is the discrete identity of symplectic Euler: with KE quadratic in the momentum, KE_{n+1} − KE_n =
(dt f + Λ_n)·(g_n + g_{n+1})/2, so the constraint work conjugate to the discrete KE is Λ_n·(g_n,C + v_presc,n)/2.
`proj_work_kick_average` (`rubber_constraint_work.py:62-76`) reads g_n from `old` (the P2G-pass p/m) after the
projection and before the MUSL `_grid_update` overwrites it — the right place in the launch order
(`tlmpm_rubber_v1.py:819-866`). The diagnostic reproduces the ADR table (`moving_clamp_diag_001`: F8 linear
end −19.24 %, kick −0.006 % (max 0.384 %), halved at 24 substeps; lumped 4.24 % with any quadrature). The v1 end-velocity
error is dt-independent; the kick-average residual halves with dt. `TLMPMRubberCW` re-launches the frozen kernels in
the frozen order and only diverts the work target; state, reaction and wrench are byte-identical (tested). Formal run
`moving_grip_v1_001` (ea32f2a, clean): GRIP-01..04 PASS, values equal to ADR §21.1.

### 4.2 Viscoelastic formulation (§21.2)

- S⁰ = μ₀J^(−2/3)(I − (I₁/3)C⁻¹) with C⁻¹ = cofᵀcof/J² (`rubber_visco_constitutive.py:83-95`); F·S⁰ =
  aF − (aI₁/3J) cof F, so `nh_jbar_pk1` = F S⁰ + p(J̄) cof F and P = P_NH + F Σ(Q_i − g_i S⁰) = F(g∞S⁰ + ΣQ_i) + p cof F:
  the overstress form is exact. Q lives on the reference-configuration PK2 (objective).
- Update Q ← aQ + b ΔS⁰, a = e^(−dt/τ), b = g(τ/dt)(1 − e^(−dt/τ)) via `expm1` in float64 on the host: the exact
  solution of Q̇ = −Q/τ + g Ṡ⁰ for S⁰ linear in the step. Probe: a unit-rate ramp gives |ΔQ/Q| ≤ 3.5e-14 at dt from
  2.5e-6 to 1e-3 s (dt-independent); a 16 Hz sinusoid converges as O(dt²) (1.1e-7, 2.8e-8, 6.9e-9 at dt 20, 10, 5 µs),
  i.e. only the piecewise-linear interpolation of S⁰ depends on dt. VISC-04a: τ fits within 8e-8 across dt.
- Energy: Ψ_v = |Q|²/(4gμ₀) and the trapezoid D = dt(|Q^n|² + |Q^{n+1}|²)/(4gμ₀τ) are the stored energy and dissipation
  of a Maxwell branch with spring 2gμ₀ and dashpot 2gμ₀τ in the strain-like S⁰/(2μ₀); `sym6_ddot` counts off-diagonals
  twice (Frobenius). Probe: W − D − Ψ_v = −3.5e-9 of W over a 16 Hz history.
- With every g = 0, `prony_stress_particle` returns `p_nh` without an addition (`:156-157`) and the guard factor is
  exactly 1 with a zero overstress: byte identity holds by construction; tested for the full option set, probed by me
  for the default options (R4).
- float64 `visco_q` / `visco_s_prev` / `params64`, masked reset of the registry state, `params64` staging and the
  Prony scale resampling at the batch dt are implemented as designed and tested (VISC-06 / -07).

### 4.3 RUBBER-VISC-03 v2: estimator and the 1-D rod transfer

- Geometry: strip 30 × 6 × 3 mm, case C with `end_mask "x"` (axial grips at the end faces, lateral free), markers at
  the nearest particle layers a = 7.25 mm, b = 22.25 mm (`marker_ref_x` in the raw metadata), ε_g =
  (u(b) − u(a))/(b − a), σ from the grip-1 (driven) reaction. For ρu_tt = E*u_xx, u(0) = 0, u(L) = Ue^{iωt}:
  u = U sin kx / sin kL, σ(L) = E*kU cos kL / sin kL, so σ(L)/ε_g = E*k(b − a)cos kL/(sin kb − sin ka) and
  σ(0)/ε_g = E*k(b − a)/(sin kb − sin ka), k = ω(ρ/E*)^½ — the formulas in `rubber_visco_reference.py:288-308`. The
  lateral (Love) term (~4e-5 at 32 Hz) and the end-node half-cell inertia (part of the rod's own mass) are negligible
  or included. The grip-0 / grip-1 check validates the model on the data, including L: a 3 % error in L would move
  the predicted 32 Hz difference by about 0.12 %, against an observed agreement of 0.003 %.
- The fixed-point de-embedding starts from E_meas and uses no reference value. The whole-period window with σ and ε
  mean-removed is exact for steady sinusoids; the v1 last-cycle window (313 / 157 samples for 312.5 / 156.25 per
  period, σ mean kept) leaks.
- Independent recomputation (my estimator, rod transfer and fixed point, from `visc_v2_001/raw`): de-embedded E′
  −0.402, −0.402, −0.403, −0.404, −0.401, −0.414 %; E″ −0.435 … −0.867 %; loss −0.397 … −0.434 %; grip 0 vs grip 1
  ≤ 0.003 %; not de-embedded 32 Hz E′ −1.867 %. Identical to the report.

### 4.4 RUBBER-VISC-05 v2: kick-paired branch work

- The kick of substep n applies P_n against the velocity average (g_n + g_{n+1})/2, i.e. against (ΔF_{n−1} + ΔF_n)/2;
  `visco_branch_work_kick_average` (`tlmpm_rubber_visco_v1.py:137-153`) books V pb_n:(ΔF_{n−1} + ΔF_n)/2 with ΔF from
  the MUSL grid velocity after the BC (the increment F actually receives) and keeps the left-point sum with the v1
  arithmetic. It runs after the position update and writes only its own arrays.
- Probe (1-DOF oscillator with one Maxwell branch, symplectic Euler, ω²τ dt/2 ≈ 5): left-point books −3.26 / −1.63 /
  −0.81 of the initial energy at dt, dt/2, dt/4 (first order, as in the cantilever 2.58 / 1.05 / 0.47 %); kick-paired
  −2.4e-3 / −3.0e-4 / −2.8e-5. A left-point Riemann sum of an exact smooth history has the same O(dt) bias: a quadrature
  defect, not a dynamics defect.
- Independent recomputation from `visc_v2_001/raw`: max |B_N|/S_N 0.0021–0.0072 % (strips), 0.0353 % (creep),
  0.0121 / 0.0062 / 0.0031 % (cantilever ×1 / ×2 / ×4); left-point on the same runs 2.5758 / 1.0539 / 0.4736 %; g = 0
  control 0.0198 %; 0 negative D increments. Identical to the report.

## 5. Protocol honesty and records

| Protocol | Declared (commit, KST) | Implementation | Formal run (commit, clean, KST) |
|---|---|---|---|
| `rubber_moving_grip_tests_v1` `ff14f39d…` | ea32f2a 22:08:13 (with runner and pin) | 59fca99 22:05 | smoke 22:08:19, `moving_grip_v1_001` 22:08:38–22:09:34 at ea32f2a, clean |
| `rubber_visc_tests_v1` `95ad1f94…` | edfd555 22:25:39 | efff788 23:06, fc3d23e 23:08 (gate 23:08:18), 555d6fb 23:24:23 | `visc_v1_001` 23:24:30–23:44:50 at 555d6fb, clean |
| `rubber_visc_tests_v2` `212fe7af…` | dca4848 00:16:54 (protocol + pin only) | bc52a2d 00:20:39, runner 889a782 00:23:53 | smoke 00:23:59, `visc_v2_001` 00:24:25–00:44:57 at 889a782, clean |

- Every protocol was committed and SHA-pinned before its formal run; the run metadata record the pinned SHA, the
  commit and a clean tree. Commits made while `visc_v1_001` ran (546d5a2, 0555131) touch only diagnostic / evidence
  scripts. No protocol file changed after its declaration commit.
- Thresholds: every v1 threshold of -03 / -05 is in v2 unchanged; the only addition is the stricter validity bound
  `deembed_consistency_rel` = e1_rel / 5 (`tests/test_rubber_p2_3_protocols.py:68-88`). The fit uses 1 / 4 / 16 Hz only
  (`scripts/rubber/p2_3/run_visc_v2.py:386`); the held-out predictions are compared with the simulated 2 / 8 / 32 Hz.
- v1 → v2: each change corrects a defect that biases the exact solution (estimator leakage −0.25 / −0.69 % in E′ on
  exact sinusoids; drive-end reaction vs gauge strain −1.45 % at 32 Hz for the exact rod; left-point quadrature O(dt)).
  The longer 16 / 32 Hz runs keep v1's intent (window after 6 τ_max). Not loosening; but post hoc (R2).
- FAILs preserved: `visc_v1_001` raw and summary unchanged; `p2_3_visco_001.json` written once (203aa97) with -03 / -05
  FAIL; report §3 and ADR §21.6 unchanged; `p2_3_visco_002.json` lists the v1 verdicts as preserved.
- v2 validity `dynamics_identity_with_v1`, checked independently: for all 14 jobs every v1 raw column except
  `branch_work` is byte-identical to the v2 run over the common rows, and v2's `branch_work_left_v1` equals v1's
  `branch_work` byte for byte.
- Material status `SIMULATOR_ONLY_UNCALIBRATED` in the recipe, the protocols, the receipts and the evidence. The
  default recipe `rubber_generic_soft_v1.json` has no Prony terms, so the default `LiveRubberMaterial` still builds the
  frozen `TLMPMRubber`.

## 6. Frozen files, CI safety, tests

- Unchanged in `3a43d1b..025ecaf`: `tlmpm_rubber_v1.py` (`28fc5189…c8d5`), `rubber_constitutive.py`
  (`c12cc77e…b869`), `batched/constitutive.py` (`3c4db62f…`), `batched/errors.py` (`6b2b8589…`), every other Phase 1
  batched module, `configs/rl_benchmark/*`, `scripts/rl_ready/*`, the rubber protocols v1–v4 and every earlier evidence
  file. Changed source: Phase 2 batched rubber modules, `rubber_material.py`, `live_rubber_material.py`, and new modules.
- CI safety: no new test reads `artifacts/`, `/mnt/` or needs CUDA or torch; SciPy is a declared dependency. The formal
  v2 identity check records NOT_RUN when v1 raw data are absent (protocol `validity`).
- CPU suites (`tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py
  tests/test_batched_*.py tests/test_rl_policies_batched.py tests/test_repo_naming.py`, 410 collected):
  - at 025ecaf, `CUDA_VISIBLE_DEVICES=''`: 406 passed, 4 skipped (torch absent: NOT_RUN), 533 s;
  - at 025ecaf, `CUDA_VISIBLE_DEVICES=0` (GPU0 lock; co-tenant PID 642659 present, not touched): 406 passed,
    4 skipped (torch: NOT_RUN), 538 s;
  - trial merge of 025ecaf into 36b8947 (uncommitted, temporary worktree, removed afterwards),
    `CUDA_VISIBLE_DEVICES=''`: 423 passed, 4 skipped (torch: NOT_RUN), 548 s (the integration side adds Phase 1
    batched tests).
  The implementer's 397 are the first three globs only (397 + 13 = 410).

## 7. Recommendations before P2-4 / P2-6

1. Fix R1 (refresh the motion table in `begin_control_step()` and `reset_masked()`), with tests.
2. Add the default-option VISC-09 tests (R4).
3. Cite VISC-03 / -05 as "v1 FAIL, v2 PASS (post-hoc measurement correction)" (R2); adopt the corrected wrench stamp
   (R3) and a fresh held-out rate in the next E″-scoring protocol.
4. Guard v2 before large-strain visco scenes (R5).
5. GPU: a declared GPU equivalence check of the batched visco path at the value level (not bitwise), and the
   motion-table memory in the P2-6 cost rows (R7, R9).

## 8. Commands

Review worktree at 025ecaf; `.venv` Python 3.12.12, Warp 1.17.0, NumPy 2.5.4, SciPy 1.18.1, pytest 9.1.1.

```
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider -rs tests/test_rubber_*.py \
  tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_batched_*.py \
  tests/test_rl_policies_batched.py tests/test_repo_naming.py
flock -w 3000 /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  .venv/bin/python -m pytest … (same selection)
git merge-tree 3a43d1b 36b8947 025ecaf      # 36 added, 10 merged, 0 changed in both
git worktree add --detach <tmp> 36b8947 && git -C <tmp> merge --no-commit --no-ff 025ecaf   # then the CPU suites
```

Probes (scratch scripts, not committed; CPU only): byte comparison of every `visc_v2_001` raw column with
`visc_v1_001`; recomputation of VISC-03 v2 (estimator, rod transfer, fixed point, re-stamped E″) and VISC-05 v2 books
from the raw series; numpy Prony / Maxwell / 1-DOF branch-work probes (§4.2, §4.4); VISC-09 with default options (R4);
the stale-table probe of R1:

```python
b = BatchedRubberMaterial([live], device="cpu", parity_mode=True, bc_launch=launch)   # launch: all_nodes_v1 / motion_table_v1
b.advance(translation=tr)          # grip +0.2 mm
b.begin_control_step()             # hold: no set_group_motions (same result with b.reset() here)
for _ in range(b.solver.substeps):
    b.solver.step()
# x differs between the two launches by max 2.0e-4 m (b.reset([1]) instead: equal, past-table fallback)
```
