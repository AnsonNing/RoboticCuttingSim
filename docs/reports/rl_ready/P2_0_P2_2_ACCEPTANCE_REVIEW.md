# P2-0 / P2-1 / P2-2 — xhigh acceptance review (R2.5 of ADR 0023, 2026-11-10)

**Scope.** This review covers rubber steps P2-0, P2-1 and P2-2 of Phase 2:
- the recipe and the calibration manifest;
- the `tlmpm_rubber_v1` single-env reference and the batched rubber path;
- RUBBER-PHYS-01..12 under protocols v1–v4;
- the refinement study, throughput and guard profiling.

It also decides the two questions the implementer escalated:
- (a) RUBBER-PHYS-11 Q1;
- (b) RUBBER-PHYS-08.

**Commit.** The review is fixed at **3a43d1b** on `claude/rl-rubber` and was read-only in the detached worktree
`RoboticCuttingSim-rl-rreview`. While it ran, five "P2-3 prep" commits landed on the branch: b922665, b55aefc, 88f0352,
89ccb82 and 670ad1b. They add batched F8/F9/F10 and change the **default** batched BC launch, first to
`constrained_list_v1` and then to `motion_table_v1`. These commits, any later ones, and an uncommitted ADR edit in the
rubber worktree are **not covered** (§6).

**Inputs read:**
- the specification's Phase 2 part (P2-0..P2-2, "Phase 2 Done");
- plan §1 and D-12;
- AGENTS.md;
- ADR 0023 in full, including §15, §18 and §19;
- the reports `P2_0_P2_1_RUBBER.md` and `P2_2_RUBBER_HYPERELASTIC.md`;
- the evidence files `p2_0_rubber_probes_001.json`, `p2_1_rubber_001.json`, `p2_1_followup_001.json` and
  `p2_2_rubber_001.json`;
- `configs/rubber/*` and `configs/materials/rubber_*`;
- every rubber module, single and batched;
- the rubber tests and runners;
- the raw outputs under `/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/{p2_0,p2_1,p2_2}` (run metadata, summaries,
  raw -02/-07/-08/-11/L0/L3/throughput rows).

**Method.**
- CPU tests and three small CPU probes; no GPU job.
- Every number quoted as "recomputed" was derived again from the raw artifacts or the code.

## 1. Verdicts

| Package | Verdict | Basis |
| --- | --- | --- |
| **P2-0** recipe, manifest, references, frozen v1 tests | **PASS** | The deliverables are complete and validated. The recipe is `SIMULATOR_ONLY_UNCALIBRATED`, and `REAL_CALIBRATED` is refused without a passed held-out manifest. The `none_v1` manifest is `NO_REAL_DATA`. The analytic references reproduce E3–E5. The v1 test set was frozen before any implementation and is SHA-pinned. The ADR resolution claim that P2-2 refuted is a design-document error carried in F1, not a P2-0 deliverable defect. |
| **P2-1** solver, scene, runtime, initializer, batched path, RUBBER-PHYS-01..10/12 | **PASS_WITH_LIMITATIONS** | Correct NH J-bar physics; pure FLIP; true batching; L0/L1 CPU parity and L2/L3 (v4 rule) GPU parity hold. Every earlier FAIL is preserved. Limitations: -07 passes only with single-env opt-in options; at 3a43d1b the batched path still has the lumped clamp, which fails -07b energy (F3). The guarded -08 fails validity (F2). The batched path has float32 static precision (F3). The L0 wording overstates "bitwise" (F5). |
| **P2-2** hyperelastic verification, refinement, β sensitivity, profiling | **PASS_WITH_LIMITATIONS** | Stresses, J/J̄ tracking, large deformation, load/unload and elastic recovery are verified. The F8/F9/F10 fixes are sound and were declared before their formal runs. Throughput and VRAM rows are honest. **Test-level FAILs stand:** -11 v4 (Q1 GCI 3.98 % at the production grid n_t 3) and -08 v4 (guard validity). Both are resolved by the protocol declarations in §3 and §4. Neither needs a solver change. |

**Merge: GO for 3a43d1b** (plus this review) into `claude/rl-ready-rubber-physics`. This is **not** a GO for the branch
tip (§6).

No BLOCKER was found:
- **No Phase 1 or frozen file changed.** `git diff --name-status db036a0 3a43d1b` (db036a0 is the merge base with the
  integration branch) lists 77 files, all `A`, none `M` or `D`. Every non-rubber change since the D-12 fork point
  4c8bd03 comes from integration commits (db036a0 and the P1-4 commits), all ancestors of d423b1b. A trial merge
  (`git merge-tree db036a0 d423b1b 3a43d1b`) has no conflict.
- **No protocol file was edited after commit.** Each of `rubber_phys_tests_v1..v4.json` was committed once (2bed4f9,
  63ecfd1, 849e10b, 3d12969), and the SHA pins pass. Every formal run started after the protocol it used (§5.2).
- **No threshold was loosened.** The test asserts thresholds are unchanged between versions. The -11 rule added an
  Fs 3 branch, which is stricter than ADR §14.4, and it was declared before the run.
- **Every failure is preserved and visible**, verified in the summaries:
  - v2 -07 FAIL (`rubber_phys_001`);
  - v3 -07b and -02 FAILs (`rubber_phys_v3_001`);
  - v2 L3 FAIL (`gpu_smoke_001`);
  - v4 -08 and -11 FAILs (`rubber_phys_v4_001`, `refinement_v4_002`);
  - the aborted `refinement_v4_001` (`ABORTED.txt`);
  - the -12 L0 attempts `_001` and `_002`.
- **The material status is `SIMULATOR_ONLY_UNCALIBRATED` everywhere** (the recipe, every evidence file and every run
  meta). No rubber cutting.

## 2. Findings (most severe first)

**F1 — MEDIUM. RUBBER-PHYS-11 v4 FAIL at the production grid. The documented n_t 3 accuracy claim is refuted.**
- **Scenario:** `refinement_v4_002`, Q1 (static sag, L/t 20, t 4 mm, W = t).
  - p_obs = 3.264 (recomputed, Celik iteration).
  - The declared rule gives p_eff 2 and Fs 3, so GCI(n_t 3) = 3.98 % > 2 %.
- **The ADR claim was wrong in the ADR's own data.**
  - ADR 0023 §2 E1 "Findings", §4.2 and the §8 D5 label `rubber_rl_v1` "full (≤ 0.5 % bending error vs converged, E1)"
    all state ≤ 0.5 %. The code label is `rubber_material.py:34`.
  - That figure generalises the L/t 10 rows. The L/t 20 J-bar row of the same E1 table (0.999 / 0.993 / 0.990 at
    n_t 3 / 4 / 6) already showed n_t 3 0.9 % above n_t 6.
  - -11 measures +1.08 % for n_t 3 against the p_eff = 2 extrapolation.
- **The wide strips are worse.** For the geometry of workload E (W/t ≥ 4), the v4 -02 pairs give the n_t 3 bias
  against the two-grid extrapolation (Lt20 t4 W4t: δ3/δ4 − 1 = 0.77 %):
  - **+1.26 % to +1.75 %** (p 3.26 to 2);
  - under the declared rule's *favourable* branch (Fs 1.25, p_eff 2), a GCI of 2.19 %, which also FAILs.
- **The workload-E rows are mislabelled.** They are labelled `reduced_fidelity: false` (`workload_e_v1.json`) on the
  strength of the refuted claim.
- **Recommendation:** decision (a), §3.

**F2 — MEDIUM. RUBBER-PHYS-08 v4 validity FAIL.**
- **Scenario:** `rubber_phys_v4_001/raw/08/gamma2.json`. The invalid bit is 4 (`RUBBER_CFL_EXCEEDED`). The stress
  errors are 7.9e-7 (σ12) and 1.6e-6 (N1).
- **Recomputed guard value:** CFL_eff = 0.5 · c_bound(F_γ=2)/c_p = **1.297** (`rubber_constitutive.py:130-146`).
- **The trip is warranted.** The block is fully prescribed (every node in the affine group), so no free mode can go
  unstable. For a general scene, though, the exact discrete ratio 2.36 at γ 2 (ADR §18.1 F1) puts the step near 1.18,
  inside the measured free-surface critical range 1.2–1.5 (E2).
- **Recommendation:** decision (b), §4.

**F3 — MEDIUM. At 3a43d1b, the energy-consistent constraint and float64 state are single-env only.**
- **Refusal is correct.** The batched path refuses the F8/F9/F10 options (`batched/rubber_layout.py:85-89`, test
  `test_rubber_solver_options.py:130-134`).
- **Consequence 1 — the batched path keeps the lumped clamp.** That clamp has been measured to dissipate numerically:
  - −2.6e-4 … −1.4e-3 Ψ per axial period after an axial release (`f8_clamp_dissipation_001`);
  - batched GPU -07b gives 1.058 % of Ψ, a FAIL of the v3 criterion, declared "reported" in `batched_gpu_v4`
    (`batched_gpu_v4_001`).
- **Consequence 2 — float32 static precision in the batched path.** Batched -02 would reproduce the v3 scatter.
- **Where the PASSes hold.** The v4 PASSes of -02/-03/-05/-07 are single-env reference results. ADR §19.3 summarises
  the batched GPU row as "PASS (07b … reported)", which reads better than the report's own "-07b FAIL (reported)".
- **Recommendation:**
  - State "single-env" next to those PASSes.
  - Treat a reviewed batched F8/F10 as a precondition for batched P2-3 VISC runs and for P2-6 grips. That work started
    in b922665, which is unreviewed.

**F4 — LOW-MEDIUM. The escalation framing of the -11 counterfactual is inaccurate.**
- `P2_2_RUBBER_HYPERELASTIC.md` §10 and the escalation say that with "p_obs and Fs 1.25 the same data give 1.19 %".
- No branch of the declared rule uses the unclipped p_obs. The rule's verified branch (Fs 1.25) still uses
  p_eff = clip(p_obs, 1, 2) = 2, which gives **1.66 %** (recomputed).
- "~1 % bending error at n_t 3" is the narrow-strip number. The strip actually used for RL is wide (F1).
- **Recommendation:** quote 1.66 % and the wide-strip range wherever the counterfactual is cited.

**F5 — LOW. L0 is value equality, not bitwise.**
- **What the tests do.** The tests (`test_rubber_batched.py:46-55`) and the F4 runner (`run_batched_l0_v3.py:49-60`)
  compare with `np.array_equal`, so +0.0 equals −0.0. The runner also uses `equal_nan` for -12.
- **The -12 result.** `f4_batched_l0_12_003` is value-equal at every step. Its first byte-level difference is at
  step 54, in the derived `wrench`: a signed zero (single `end_control_step` vs batched `_wrench_b`). The evidence field
  is still named `bitwise_equal_every_step: true`.
- **The comparator changed after results.** It changed twice for -12 (`_001` with plain `array_equal` → `_002`
  byte-wise → `_003` with `equal_nan`). All three runs are preserved and disclosed. The final definition is the natural
  extension of the test's L0 and changes no state-array verdict.
- **Recommendation:** either call L0 "value-exact (`array_equal`; NaN positions equal; signed zero not distinguished)",
  or make the wrench sign-consistent and compare state arrays byte-wise.

**F6 — LOW. The batched L0 runs predate their v4 section, and their evidence records the wrong commit.**
- The `batched_l0_v4` runs `f4_batched_l0_07_ramped_001`, `_07_instant_001` and `_12_001` started at 18:21:10 on
  44e5b59. The v4 file that declares `batched_l0_v4` was committed at 18:31:22.
- No criterion was set after data: the criterion ("bitwise every control step") is the v2 -10 L0 criterion, and the
  task was declared in ADR §18.8 F4 at 17:51.
- The evidence records `commit` 295bdc4, read at the end of the run. The report states the code at start was 44e5b59.
- **Recommendation:** correct the evidence field.

**F7 — LOW. The guard was never on for some formal runs.**
- §18.2 says the guard is "mandatory for P2-2 runs", but -04 (free vibration) and -09 were never run formally with it
  on. Their v2 results stand.
- The 31 L3-v4 runs of the -04 environments (guard on, no flag) cover only 30 ms (about 0.6 T1).
- **Recommendation:** include -04 in v5 with the guard on.

**F8 — LOW (latent). The F8 projection solves per (group, component) blocks independently.**
- Location: `tlmpm_rubber_v1.py:335-360, 240-257`.
- Cross-group coupling through particles that touch constrained nodes of two groups (same component) is ignored.
- The projection is exact for every current scene: case A has one group, case C's faces lie L apart, and the pins sit
  within one group. It would be silently inexact for P2-4/P2-6 multi-grip or pad scenes.
- **Recommendation:**
  - Refuse, or merge, blocks with a nonzero cross term.
  - Add an energy-balance test with a **moving** projected grip. The formal -07 energy windows begin after the release,
    and -06 uses the lumped clamp, so moving-constraint work under the projection is not yet scored.

**F9 — LOW. ppc 3 is accepted for any scene.**
- `rubber_scene.compile_strip` accepts ppc ∈ {2, 3} (`rubber_scene.py:157`) for any scene. Neither the live runtime nor
  the batch packer checks ppc against the recipe's profile (`rubber_material.py:345` refuses ppc ≠ 2 only in the
  recipe).
- The test `test_rubber_scene.py::test_refusals` changed its refused value from ppc 3 to ppc 4. This is disclosed and
  matches ADR §14.4.
- **Recommendation:** refuse ppc 3 outside an explicit `quadrature_check` flag, or label it in every receipt.

**F10 — LOW. The GPU run metadata is incomplete.**
- Plan §2 requires the GPU driver version. `throughput_v4_001/meta.json`, `f2_gpu_smoke_guard_001` and `l3_v4_001`
  record the GPU name, index and co-tenant samples, but no driver.
- The throughput, batched-GPU and F2 metas have no dirty-tree flag, and `batched_gpu_v4_001` also has no protocol SHA.
- **Recommendation:** add these fields to the GPU runners.

**Informational (no action required for this merge):**
- **I1.** The v4 -02 and -07 PASSes are confirmations of diagnostics seen before the declaration. Example: 07b, the F8
  diagnostic at n_t 4, mask x and CFL 0.5, gave 0.34 %, and the formal run gave 0.33 %. Thresholds are unchanged, the
  method change is mechanistically justified, and the FAILs are preserved, so this is acceptable. These are not blind
  tests.
- **I2.** The reset resamples materials on the host (`rubber_material_batch.py:387-430`, `.numpy()` syncs plus a loop
  over the reset envs) before one masked kernel. This is fine for P2-2; ADR D10 asks for device-side resampling in P2-6.
- **I3.** `gpu_smoke_001/meta.json` has an empty `commit`. That run used a `git archive` snapshot of 6c9477f, as
  documented.
- **I4.** The sections of `P2_2_RUBBER_HYPERELASTIC.md` are out of order (§9, §12–§14, §10, §11).

## 3. Decision (a) — RUBBER-PHYS-11 Q1: **move bending-dominated scenes to n_t ≥ 4**

**Recommendation.** Make `rubber_ref_v1` (n_t ≥ 4) the full-fidelity production grid for bending-dominated scenes. ADR
D5 already names it "refinement anchor, acceptance". It covers acceptance tests, P2-5 calibration fits and the Full
Native reference. `rubber_rl_v1` (n_t 3) stays selectable for RL throughput, but only as an explicitly labelled
reduced-resolution profile that carries its measured bias. Do not accept a "~1 %" error as full fidelity.

**Reasons:**
1. **The declared criterion fails at n_t 3, and the bias is larger than "~1 %" for the geometry that matters.**
   - Narrow strips: +1.08 %.
   - Wide strips: +1.26 % to +1.75 %.
   - The wide strips would FAIL the declared rule even on its Fs 1.25 branch (2.19 %).
   - Option 2 can never PASS without changing the criterion after the data. It would leave the production profile
     permanently failing its own verification.
2. **The ADR basis for n_t 3 is refuted (F1).** Keeping the "full" label requires a new justification, not a
   reinterpretation.
3. **n_t 4 is expected to pass the unchanged rule.**
   - From the existing (4, 6) pair: GCI 0.62 % at Fs 1.25 and 1.48 % at Fs 3 (both p_eff 2).
   - n_t 4 bias: Q1 +0.50 %, Q2 −0.45 %, Q4 +0.02 %; wide strips +0.50 % … +0.99 %.
   - These numbers are indicative only. They come from data already seen, which is why v5 must add a new, unseen grid
     (below).
4. **A resolution-dependent bias in calibration is avoidable.** P2-5 would otherwise absorb about 1–1.75 % into the
   fitted μ.

**Cost.** Particles/env scale as n_t³, substeps as n_t, so GPU work per env-step scales as n_t⁴.

| Workload E (150 × 20 × 4 mm, range maxima μ 1.3 MPa / ρ 1000) | n_t 3 (`rubber_rl_v1`) | n_t 4 (`rubber_ref_v1`) | Ratio |
| --- | --- | --- | --- |
| particles/env | 41,220 | 97,280 | 2.36 |
| material dt / substeps per 20 ms | 4.003 µs / 4,996 | 3.00 µs / 6,662 | 1.33 |
| particle-substeps per env-step | 2.06e8 | 6.48e8 | 3.15 |
| env-steps/s, N 8 (SHARED_GPU, different sessions) | 1.467 (`throughput_v4_001`) | 0.468 guard on / 0.473 off (`f2_gpu_smoke_guard_001`) | ≈ 3.1 |
| env-steps/s, N 16 | 1.515 | ≈ 0.48, projected from the saturated 3.1e8 particle-substeps/s (not measured) | ≈ 3.15 |
| VRAM peak, N 8 | 824 MB | 922 MB | not binding (≤ 4 GB) |

The throughput cost (about 3×) lands only on scenes that claim bending accuracy. RL can keep n_t 3 under its label. The
R2.4 profile already identified a cheaper `_bc_b` launch (14–17 % of GPU time) as a mitigation.

**Protocol changes to declare in `rubber_phys_tests_v5.json`, SHA-pinned before any run:**
- **RUBBER-PHYS-11 v5 setup:**
  - `production_n_t`: **4** for Q1, Q2 and Q4, and for a new observable **Q1w**. Q1w is the -02 wide configuration:
    L/t 20, t 4 mm, W = 4t, μ 0.4 MPa, α 0.05, `f64_shadow_v1`, static_stop_v3.
  - Q3 keeps its v4 PASS. It is homogeneous and grid-independent.
- **RUBBER-PHYS-11 v5 grids:**
  - `grids_n_t` [3, 4, 6, **8**].
  - `triplet_finest` (4, 6, 8). The production GCI is GCI_coarse of n_t 4, with h = (1/8, 1/6, 1/4), r21 = 4/3 and
    r32 = 3/2.
  - n_t 8 is new data. Its single-env runs may use a GPU (recorded), because Warp CPU needs hours at n_t 8.
- **The `rule` block stays byte-identical to v4.** A test should assert this. The block contains:
  - Celik's iteration;
  - `p_bounds` [1, 2] and `p_rule` clip;
  - Fs 1.25 / 3.0 with the verified window [1, 3];
  - the grid-independence, oscillatory and inconclusive rules.

  The thresholds `gci` 0.02 and `dt_rel` 0.001, the CFL set {0.5, 0.25}, the guard 1.0 and the solver options are also
  unchanged.
- **n_t 3 is reported, not scored.** For each Q, report its value, |φ(3)/φ_ext − 1| and its GCI by the same rule, as the
  measured bias of `rubber_rl_v1`. The v4 -11 FAIL stays the v4 result.
- **ADR 0023 amendment (not a protocol change, required):**
  - Correct the E1 "Findings" line and §4.2 (n_t 3: ≤ 0.5 % only at L/t 10; about 1 % at L/t 20; 1.3–1.75 % for
    W = 4t).
  - §8 D5: `rubber_ref_v1` becomes the full-fidelity profile for bending-dominated scenes.
  - `rubber_rl_v1` becomes `reduced_fidelity: thickness_resolution_n_t3`, and its receipts print the measured biases.
  - The code label in `rubber_material.RESOLUTION_PROFILES` changes in the same versioned commit.
  - Add a new `workload_e_v2.json` with a full row at n_t 4 and a labelled n_t 3 row. `workload_e_v1.json` stays
    unchanged.

## 4. Decision (b) — RUBBER-PHYS-08: **yes, declare a v5 -08, with r = 1.5 rather than 1.3**

- **Why r = 1.3 is wrong.** At γ 2, r = 1.3 gives CFL_eff = 1.297/1.3 = **0.9974**, 0.26 % under the limit (recomputed).
  - That margin is smaller than the float32 evaluation of the bound.
  - It is the smallest value read off the failing run, so it would look like tuning even though dt only shrinks.
- **The reserve is physically motivated, not cosmetic.** The trip is warranted for any free-boundary scene at γ 2
  (F2).
- **Changes to declare in `rubber_phys_tests_v5.json`:**
  - **`setup.deformation_reserve`: 1.5.** It is derived by a rule written into the protocol and evaluated analytically
    before the run: r = ⌈2 · CFL_eff,max(r = 1) / (0.9 · limit)⌉ / 2, with CFL_eff,max from
    `rubber_constitutive.wave_modulus_bound_np` at the declared γ_max = 2.
    - CFL_eff,max = 1.297 → r = 1.5.
    - CFL_eff at γ 2 then becomes 0.864, a 13 % margin.
    - Use the same rule for every large-strain test in P2-3/P2-4.
  - **`setup.imposed`:** γ is ramped linearly over one control interval of the v4 duration (control_dt = 200 ·
    dt_max(r = 1)), now 300 substeps. The block is fully prescribed, so the ramp rate does not affect the stress.
  - **Unchanged:** guard limit 1.0, the validity rule (any invalid flag FAILs), the criteria (|σ12/(μγ) − 1| ≤ 0.005,
    |N1/(μγ²) − 1| ≤ 0.005) and the measurement.
  - **History stays:** the v4 -08 FAIL and the v2 (unguarded) PASS remain recorded.
- **Not loosening.** The step only shrinks, and no threshold moves.

## 5. Verification record

### 5.1 Physics (code and probes)

- **Neo-Hookean J-bar energy and PK1** (`rubber_constitutive.py:65-97`).
  - Code: P = aF + (p(J̄) − aI₁/(3J)) cof F, p = κ/2 (J − 1/J), J̄ the cell mean of det F (deterministic CSR gather,
    `:151-162`). Stress is evaluated in float64 and rounded once.
  - Probe 1: the PK1 equals the derivative of the *cell* energy Σ V μ/2 (J^(−2/3) I₁ − 3) + V_c U(J̄), with eight
    particles sharing J̄. Over 200 random F, the relative error against central differences was **6.9e-9**.
- **Validity window.** `_stress` counts a violation when det F or J̄ lies outside [0.2, 5], NaN included
  (`in_domain`). The floor 0.05 enters the evaluation only. No SVD or clamp hides inversion (`tlmpm_rubber_v1.py:136-145`).
- **Pure FLIP.** pic ≠ 0 is refused (`tlmpm_rubber_v1.py:611-612`); the batched path is fixed at pic 0. There is no
  artificial damping outside the labelled initializer (`:773-780`).
- **Stable-step guard.**
  - The reference-frame acoustic tensor Q(N) = aI + (p′ + 5aI₁/(9J²)) ggᵀ − (2a/(3J))(fgᵀ + gfᵀ) was re-derived (the
    cofactor second derivative vanishes in N⊗N). The Weyl bound in `wave_modulus_bound` is correct.
  - Probe 2: the bound is ≥ the exact acoustic maximum at γ 0.5, 1 and 2. CFL_eff at γ 2 is 1.297 / 0.997 / 0.864 /
    0.648 for r = 1 / 1.3 / 1.5 / 2.
  - The guard is read-only and graph-safe. Its per-env counters are cleared by the masked reset
    (`rubber_material_batch.py:176-196`).
- **F8 consistent-mass projection** (`tlmpm_rubber_v1.py:213-295, 843-853`).
  - It is the KE-metric orthogonal projection onto (W M_p v)_C = m_C v_presc: μ = A⁻¹(p_C − m_C v_presc), with
    A = m Σ w_I w_J constant in TLMPM. This is a RATTLE-type velocity projection.
  - Repeating the MUSL scatter makes the constrained grid velocity exact. The reaction −m_I μ_I and the work
    impulse · v_presc are consistent.
  - Probe 3 (independent; case C, L 12 mm, W 4 mm, t 2 mm, n_t 2, 768 particles, λ 1.03 cosine ramp 60 T_ax plus a
    10 T_ax hold, instantaneous release, 20 T_ax, guard on, Warp CPU) gave the following energy drift:

    | Constraint | Slope (Ψ/T_ax) | Mean dE/Ψ, last quarter | max \|dE\|/Ψ |
    | --- | --- | --- | --- |
    | `lumped_v1` | −1.7e-3 | −3.0 % | – |
    | projection | +3.2e-5 | +0.08 % | 1.56 % (bounded; coarse grid) |

    This confirms the claim qualitatively. Latent limit: F8 of §2.
- **F9 traction work** (`:297-306, 863-865`). f·Δx uses the displacement driven by the same MUSL grid velocity as the
  F update. That is the work conjugate of the discrete scheme. The v1 accumulation is diverted to a scratch array, so
  nothing is counted twice.
- **F10 float64 shadows** (`:309-332, 855-858, 927-929`). F and x are rounded once per substep. No external state
  assignment bypasses `sync_shadow` (grep). The reset restores both shadows.
- **The solver sources are the ones the formal runs used.** `tlmpm_rubber_v1.py` (sha256 `28fc5189…c8d5`) and
  `rubber_constitutive.py` (`c12cc77e…b869`) are unchanged since ba2dec5 / bdf2838, before every v4 formal run, and
  still unchanged at 670ad1b.

### 5.2 Protocol honesty (commit order against run starts, KST)

| Protocol | Commit (time) | Formal runs and their start |
| --- | --- | --- |
| v1 | 2bed4f9 (15:06:56) | none: the stop rule was found unreachable in pre-run smoke (`protocol_diagnostics_001`, 15:22) |
| v2 | 63ecfd1 (15:27:17) | `rubber_phys_001` 15:28:22 |
| v3 | 849e10b (16:42:07) | `rubber_phys_v3_001` 16:42:18; `summary_with_validity.json` re-evaluation without overwrite |
| v4 | 3d12969 (18:31:22) | `rubber_phys_v4_001` and `refinement_v4_001` 18:31:37; `l3_v4_001` 18:31:46; `throughput_v4_001` 18:34:47; `batched_gpu_v4_001` 18:37:14; `refinement_v4_002` 18:52:04 at a3a079a (one-line measurement fix in `case_q4`, protocol unchanged) |
| §18.8 F2 / F4 | ADR 5b0a44d (17:51) | F2 smoke 18:03:55 (v2 L2 criterion); F4 L0 18:21:10, which is before v4 (F6) |

Run metadata:
- The CPU formal metas (`rubber_phys_*`, `refinement_*`) record `src_or_configs_dirty: false` and the protocol SHA.
- `l3_v4_001` records `dirty: false`.
- `throughput_v4_001`, `batched_gpu_v4_001` and `f2_gpu_smoke_guard_001` record the commit but no dirty flag, and
  `batched_gpu_v4_001` also has no protocol SHA (F10).

The ADR §19.1/§19.2 rationale is unchanged since 3d12969. The diagnostic numbers it cites (F8 at n_t 3/4, CFL 0.5; F10 for the W = t triple) existed before
18:31:22, per the file times in `f8_clamp_dissipation_001` and `f10_static_precision_001`.

### 5.3 Single vs batched parity and isolation

- **True batching.** One launch sequence over the flat particle and node arrays of all envs (`batched/rubber_batch.py:143-170`):
  - per-env μ, κ and ρ via `env_of_particle`;
  - per-env violation, CFL and work rows;
  - masked partial-reset kernels (`rubber_material_batch.py:116-196, 441-458`);
  - no Python loop over envs in stepping.
- **CPU L0/L1.** The tests pass: N 1, N 3 heterogeneous, N 3 copies, perturbation isolation, partial reset equal to a
  fresh env with the other envs unchanged, vars()-complete reset classification, resampled materials, ownership
  audits, refusals, and the guard tripping in one env.
- **Opt-in options.** The batched path refuses them with `UNSUPPORTED_SOLVER`, and the guard is supported.
- **L3 rule v4** (`rubber_l3.py`).
  - Design: one max-deviation statistic per (env, observable); leave-one-out reference of 20 GPU repeats; lognormal
    prediction bound with Bonferroni α 0.05/16 and a Cantelli fallback. It was declared before its run.
  - Result: 0 of 16 outside, max M/U 0.877 (Shapiro p 0.076–0.996, lognormal for all 16); held-out repeats 10 of
    10 WITHIN.
  - It replaces a demonstrably weak rule (a fourth sample leaves the range of three with probability ½). The v2 FAIL is
    preserved.
  - Limitation: the horizon is 30 ms.

### 5.4 Phase 1 safety

- **Phase 1 files untouched** (§1). `tests/test_batched_frozen_sources.py` passes. `batched/constitutive.py`, the
  Phase 1 batched modules, `configs/rl_benchmark/*` and `scripts/rl_ready/*` are byte-identical to the merge base.
- **Repo-wide guard tests pass:** `test_phaseH_runtime.py::test_simulator_modules_do_not_import_the_profiler`,
  `test_root_cause_recorders.py::test_counterfactual_module_is_not_reachable_from_production_code` and
  `test_tlmpm_v6.py` (16 passed).
- **The naming test passes.**
- **Effect on the P1-6 acceptance.** `acceptance_v1.cpu_tests` runs the whole `tests/` directory. The rubber tests add
  about 30 s to it.

### 5.5 Performance honesty

- **Columns.** Every rubber row reports the plan columns: N, particles/env, control Hz, material dt,
  substeps/control step, n_t, β, cameras, env-steps/s, aggregate sim-time/wall, VRAM peak, reset p95, physics verdict,
  warm/cold, GPU and `SHARED_GPU`.
- **Co-tenants.** The run meta samples them before and during the run.
- **Comparison with v7_2.** It is explicitly called not a matched comparison, and the substep count is printed (D11).
- **Recomputed values:**
  - dt 4.003 µs and 4,996 substeps at n_t 3; 6,662 substeps at n_t 4;
  - 3.12e8 particle-substeps/s at N 16.
- **Missing:** the driver version (F10).

## 6. Merge and freeze

- **Merge 3a43d1b, not the branch tip.** The commits after 3a43d1b (b922665, b55aefc, 88f0352, 89ccb82, 670ad1b, …)
  are unreviewed. They change:
  - the **default** batched BC launch (now `motion_table_v1`) in the code path that produced the P2-2 GPU rows;
  - `tests/test_rubber_batched.py` and the batched refusal test.

  The rubber worktree also holds an uncommitted ADR edit by another agent. Steps:
  1. `git merge 3a43d1b` into `claude/rl-ready-rubber-physics`.
  2. Bring this review file over separately: cherry-pick its commit, which only adds this file.
  3. Run the full CPU suite on the merged tree.
  4. Review the P2-3 prep commits as their own step. The single-env solver sources are unchanged at 670ad1b.
- **Freeze (R2.5).**
  - Pin `tlmpm_rubber_v1.py` and `rubber_constitutive.py` at the SHA-256 values in §5.1 in a frozen-source test. Later
    behaviour goes into new opt-in options or versions.
  - Record the acceptance status in ADR 0023 rather than editing the module's `STATUS` constant. Editing the constant
    would change the source hash recorded in every P2-1/P2-2 receipt. If it is edited, use one documented commit
    changing only that line, and pin the new hash.
- **Before P2-3 formal runs:**
  1. Declare and run `rubber_phys_tests_v5` (§3, §4, and F7: -04 with the guard).
  2. Make the ADR 0023 corrections (F1, F3 wording, F5 wording).
  3. Relabel `rubber_rl_v1` and add `workload_e_v2`.
  4. Review the batched F8/F10 work, including a cross-group refusal and a moving-grip energy test (F8). It is a
     precondition for batched VISC runs and P2-6 grips.

## 7. Commands (review worktree at 3a43d1b; `.venv` Python 3.12.12, Warp 1.17.0, NumPy 2.5.4, SciPy 1.18.1, pytest 9.1.1)

| Command | Result |
| --- | --- |
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -p no:cacheprovider tests/test_rubber_*.py` | 113 passed, 24.0 s (21:19:45) |
| `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -p no:cacheprovider tests/test_rubber_*.py` | 113 passed, 24.6 s (21:20:09) |
| `CUDA_VISIBLE_DEVICES='' … pytest tests/test_tlmpm_rubber_v1.py tests/test_live_rubber_material.py tests/test_batched_frozen_sources.py tests/test_repo_naming.py` | 33 passed |
| repo-wide guard tests (§5.4) | 16 passed |
| `git diff --name-status db036a0 3a43d1b`; `git merge-tree db036a0 d423b1b 3a43d1b` | 77 × `A`; no conflict |

The probes were temporary scripts (not committed), run with CPU only:
- Probe 1, the PK1 of the cell energy: NumPy finite differences, 200 random F.
- Probe 2, the guard bound against the exact acoustic tensor: NumPy, 4,000 directions per F.
- Probe 3, F8: `LiveRubberMaterial` on Warp CPU with the parameters in §5.1.
- The GCI was recomputed by an independent Celik iteration from the `refinement_v4_002` values. It matches
  `rubber_gci.study` to the last digit.

The first CUDA-hidden test log was overwritten by a concurrent writer in a shared scratch directory, so the run was
repeated into a fresh directory. The results above are from that repeat; the overwritten log is not used.
