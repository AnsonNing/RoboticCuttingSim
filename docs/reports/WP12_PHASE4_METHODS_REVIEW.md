# WP-12 Phase 4 supplementary methods review: MPM transfer, fracture, contact and surface methods against P1–P5

Date 2026-10-31 (KST). Reviewer `rcs-xhigh` (user instruction 2026-10-31). Branch `claude/wp12-physical-cut-surfaces`,
HEAD `b0a3fb8`; `tlmpm_v5.py` sha256 `a7a05b2f…` (the hash recorded by every E1 row). Nothing was committed. No solver,
law, threshold, protocol, frozen record, WP-11 evidence or run directory was changed. `resolution_002` and
`phase4_e1_001` were read while running and have since completed (110/110 and 47/47 rows). Their uncommitted
summaries (`docs/evidence/wp12/phase4_resolution_001.json`, `phase4_e1_001.json`) agree with §2: tool force
NOT_CONVERGED, law-8 flat-board onset at 10°.

**Labels.** REVIEWER = read-only analysis of saved states. STATIC = relabelling of a saved final state with no
dynamics after it. PROBE = a reviewer CPU run of the unchanged `tlmpm_v5`; its COUNTERFACTUAL variants are host-side.
None is an acceptance result, a law or a profile (scripts: §8). **Scope:** methods not in
[WP12_SURFACE_TECH_REVIEW.md](WP12_SURFACE_TECH_REVIEW.md), judged against **P1** x vs F at cut faces, **P2** chips
scattering without codes, **P3** connected oblique bottom layers on the flat board, **P4** non-converging contact
force, **P5** wrong cut-aware topology.

## Verdict

1. **P1 is an identity of the update, not only a hypothesis.** With a fixed reference stencil, F is exactly the
   gradient of the grid displacement field. The particle positions also integrate the FLIP velocity's null-space part,
   which F never sees. The PROBE explains 95 % (first layer) to 100 % (> 6 s) of \|dx − F̄dX\|² by that drift. Moving
   x with the MUSL grid velocity cuts the first-layer P99 from 1.88 s to 0.38 s (COUNTERFACTUAL) (§1.1, §2.2).
2. **P2 is P1 on sub-cell chips. It is not fracture.** TRIM chip links stretched 2–5 s have endpoint F stretches of
   at most 1.11. TLMPM v5 has no damage state, and same-code material always stays coupled through the stencil. Under
   the provisional J2 material, a Griffith estimate rules out tearing for chips thinner than about 6 mm (§1.4, §2.3).
3. **P3 has two mechanisms on the flat board.** P3a: code-0 particles of the uphill wedge share nodes with both
   digits (A_p15/20/25/35, every OBL20 dx = 4s row). P3b: material displaced across the plane ahead of the edge gets the
   opposite digit (all 96 bridges of A_p10, 228/380 of A_p30). The L9-EL shadow labels unswept ligaments: R_p20 at
   h = 0.75 and 1.0 mm is predicted SEPARATED, with 668 and 638 labels below the edge stop. A STATIC CRAMP-style
   visibility test separates the pure-P3a rows and keeps every h ≥ 0.75 mm row connected (§2.4).
4. **P4 has two resolution confounds.** The penalty is a series spring K_p = k·m/Δt². A two-parameter series fit
   reproduces the CUT and OBL20 Δt series: the formal 0.1 ms gives about 22–30 % of the stiff-contact limit. Second,
   the cutting zone is |z| ≤ s while the blade is fixed in metres. For s ≥ 1.36 mm every flank contact lies inside the
   zone, which is why the CUT force collapses at s = 1.5 and 2 mm (§1.2, §2.5).
5. **P5 is downstream of P1/P2.** No surface method (MDC, divided voxels, CPDI corners) may reopen Phase 3 before the
   solver state is consistent. Applied to today's state, any of them would be a visual-only fix.
6. **Modules, in order.** (i) The grid-velocity position update, the standard MPM form and what the TLMPM reference
   code does (S). (ii) A Δt-independent contact law: physical penalty stiffness (S) or velocity projection (M).
   (iii) CRAMP/CPIC particle-node visibility as a shadow law-9 input for P3a (M). (iv) XPIC or TL-APIC only if (i)
   leaves velocity noise (M). (v) Damage or cohesive fracture: not now, since no confirmed problem is fracture (L).
   (vi) Manifold DC: later, for the surface (M). BFEMP (*barrier* FEM–MPM coupling) and IPC need an implicit
   integrator, i.e. a solver replacement: rejected.

## 1. Facts of `tlmpm_v5` that decide applicability (verified in code)

| # | Fact | Code | Consequence for the methods |
| --- | --- | --- | --- |
| 1.1 | Ḟ = Σ_I v_I^M ⊗ ∇_X N_I(X_p) with fixed reference gradients and the **MUSL-remapped** grid velocity; positions move with the **FLIP-blended particle velocity** x += Δt v_p, α = pic = 0.05 | `tlmpm_v5.py:588–590`, `:603–613`, `:866–877`; `material_config.py:13` | F = I + Σ U_I ⊗ ∇N_I exactly. x̃ = X + Σ N_I U_I would be exactly F-consistent: lattice links are exact for trilinear N when nodes sit midway between particle columns, as in every s = dx/4 run. v_p = S v⁺ + (1−α)(I − SS⁺)V (Hammerquist & Nairn 2017, Eq. 13–17), so x − x̃ = ∫(1−α)(I−SS⁺)V dt + O(slot history). The standard MPM position update is X ← X + Δt S v⁺ (their Eq. 14). The TLMPM reference code moves particles with the grid velocity (Karamelo `solid.cpp:576–615`); snow MPM uses the FLIP velocity (Stomakhin 2013, step 10), as v5 does. In UL the stencil is rebuilt every step; in TL the drift is never fed back |
| 1.2 | Penalty impulse jn = k m/Δt · max(−gap − Δt·min(v_n,0), 0); friction capped at m\|v_t\| | `:154–165` | Spring K_p = k m_p/Δt² (15 N/m per particle at 1 mm, 0.1 ms) plus a dashpot k m_p/Δt. Per area k ρ s/Δt² = 1.5×10⁷ Pa/m, the same order as E/ℓ for ℓ = 1–4 mm, so the compliance is in series with the material |
| 1.3 | Contact impulses enter the **grid** through P2G, not the particle | `:513` | The per-substep "fraction of overlap removed" (resolution protocol, `stability_limit.other_rates`) is nominal; the particle feels a grid-smoothed increment |
| 1.4 | No damage or fracture state: J2 log-strain plasticity, perfect (no hardening), yield 1 kPa | `:77–99` | The stress is bounded. The elastic energy density ψ ≤ σ_y²/6μ ≈ 8.7 J/m³ (deviatoric, unconfined), so a Griffith tear with G_c = 0.05 J/m² needs t ≳ G_c/ψ ≈ 6 mm. Same-code material shares fields at every node it touches, so it stays one mechanical body: TL has no numerical fracture (de Vaucorbeil 2020 abstract) and v5 has no physical one |
| 1.5 | Cutting zone \|local z\| ≤ s and −2dx ≤ local x ≤ 2r; after progress = 1 the advance and drag parts of the contact impulse are released | `:255`, `:277`, `:840–841`; r = 0.45 s at `:718` | Flank contact needs \|z\| < 0.75 mm + 0.45 s. It is entirely inside the zone when s ≥ 1.36 mm, so coarse rows carry almost no advance force |
| 1.6 | Cross-cut and self-contact between particles only for conflicting codes or a reference distance > 3dx | `:484` | Code-0 particles never contact either side; they couple through grid fields (the bridge mechanism of P3) |

## 2. Evidence read for this review (REVIEWER / STATIC / PROBE; not acceptance)

### 2.1 P1 on `resolution_002` final states (non-conflicting lattice links, \|dx − F̄dX\|/s by band from the plane)

| Rows | ≤ 1 s: P50 / P99 | 1–3 s P99 | > 6 s P99 | Reading |
| --- | --- | --- | --- | --- |
| CUT s 1, dx 4 mm, Δt 0.2 / 0.1 / 0.05 / 0.025 ms | 0.108/2.69, 0.080/1.83, 0.069/1.43, 0.098/0.90 | 1.31, 1.04, 0.81, 0.60 | 0.14, 0.12, 0.09, 0.07 | concentrated at the face; P99 falls sub-linearly with Δt; P50 does not |
| CUT g4s, Δt 0.05: s 2 / 1.5 / 1 / 0.75 / 0.5 mm | P99 0.79, 0.97, 1.43, 1.51, 1.85 s (1.6, 1.5, 1.4, 1.1, 0.9 mm) | — | ≤ 0.09 | not convergent in units of s; slowly falling in metres |
| CUT dx 4 mm fixed: s 1 / 0.75 / 0.5, 0.1 ms | P99 1.83, 3.22, 3.11 s | 1.04, 1.97, 2.12 | 0.12, 0.23, 0.33 | more particles per cell, more drift (larger null space) |
| TRIM s 1, Δt 0.2 → 0.025 | P99 2.74 → 1.19 (max 2.8–3.9) | 2.58 → 1.14 | ≤ 0.09 | same pattern at the skim face |
| MULTI2 / OBL20 s 1, 0.1 → 0.025 | P99 3.53 → 2.19 / 6.20 → 2.46 | — | ≤ 0.17 | same |

The F6 repeats are identical (CUT 1.83 / 1.83), so the band is not GPU noise. No existing configuration meets the
Phase 3 §3.3 reopen measure (≥ 2× on first-layer P50 **and** P99): at 0.025 ms the P99 halves but the P50 does not.

### 2.2 P1 identity PROBE (CPU, unchanged `tlmpm_v5`, law 8; 24 × 12 × 12 mm bar, 1 mm / 4 mm / 0.1 ms, vertical cut at 20 mm/s, `ruler_knife_v2`-like 1.5 mm double-bevel profile; 2 components in every variant)

| Variant | ≤ 1 s P50 / P99 | 1–3 s P99 | 3–6 s P99 | > 6 s P99 | Explained by ∫(v_p − S v^M)dt (same-code links, by band) |
| --- | --- | --- | --- | --- | --- |
| base (α 0.05) | 0.173 / 1.881 | 0.964 | 0.486 | 0.191 | 0.950 / 0.965 / 0.998 / 1.000 (residual P99 > 6 s: 1e-4 s) |
| COUNTERFACTUAL α = 1 | 0.071 / 0.446 | 0.334 | 0.165 | 0.048 | 0.846 / 0.915 / 0.990 / 1.000 |
| COUNTERFACTUAL x += Δt S v^M | 0.059 / 0.377 | 0.230 | 0.032 | 0.000 | — (no drift by construction) |

The xgrid remainder of 0.38 s is label timing: vertical links keep P99 0.39 s against 0.06 s for horizontal ones,
because neighbours labelled at different substeps use different fields in between. The override runs after each
substep, so law 8 still saw the FLIP position (V1.4 removes this). α = 1 keeps the MUSL mismatch (I − SS⁺)S: 0.45 s.

### 2.3 P2: TRIM dome (chip side z_ref > 18 mm), uncut links

| Row | Links > 2 s long | Their length P50 / max | Endpoint F max principal stretch P50 / max | det F P5 |
| --- | --- | --- | --- | --- |
| TRIM s 1, 0.1 ms | 269 / 6 466 | 2.33 / 5.1 s | 1.065 / 1.112 | 0.970 |
| TRIM s 1, 0.025 ms | 87 / 6 470 | 2.36 / 4.2 s | 1.150 / 1.219 | 0.972 |
| TRIM s 0.5, 0.05 ms | 228 / 54 394 | 2.44 / 6.6 s | 1.151 / 1.318 | 0.921 |
| TRIM s 2, 0.05 ms | 242 / 683 | 2.73 / 8.2 s | 1.331 / 1.529 | 0.952 |

Positions stretch the chip 2–8× while F stretches it by 11–53 %: the separation is not in the constitutive state.
It is drift of a body the solver still treats as one (§1.4; Phase 3 F7). Link length only locates the drift; per the
user's rule it is not fracture evidence.

### 2.4 P3: `phase4_e1_001` (law 8 unchanged; shadow candidates from `result.json`; STATIC rules on `snapshots.npz`)

| Rows | Law 8 | Bridges needing a mis-sided label (P3b) | L9-EL shadow (labels below the edge stop) | STATIC all-pairs visibility (labels) |
| --- | --- | --- | --- | --- |
| A, τ = 0…±5°, h = 0 | SEPARATED | — | SEPARATED | unchanged (0) |
| A_p10 / A_m10 | CONNECTED (96) | **96 / 96** | SEPARATED (2 521) | CONNECTED (0) |
| A_p15, p20, p25, p35 (± mirror) | CONNECTED (6–180) | 0 | SEPARATED (3 542–5 490) | **SEPARATED (6–180)** |
| A_p30 | CONNECTED (380) | 228 / 380 | SEPARATED (5 480) | CONNECTED (152) |
| R_p05 h 0.25 / 0.375 / 0.5 | CONNECTED | 0 | SEP / SEP / CONN | SEPARATED (20–60) |
| R_p05 h 0.75 / 1.0 | CONNECTED | 0 | CONNECTED (1 436; 618 below h at 1.0) | CONNECTED (0 / 4) |
| R_p20 h 0.25 … 1.0 | CONNECTED (56–188) | 0 | **SEPARATED at every h (668 / 638 below h at 0.75 / 1.0)** | SEP at h ≤ 0.5; **CONNECTED at 0.75, 1.0 (0)** |
| R_p35 h 0.25 … 1.0 | CONNECTED (272–424) | 0 | SEP at h ≤ 0.75 (367 below h at 0.75); CONN at 1.0 | SEP at h ≤ 0.5; CONNECTED at 0.75, 1.0 |
| D (slotted): groove limit 7.0–8.5 mm (reach FAIL at 20–35°); reach limit 6.7 / 5.4 / 3.9 / 2.2 mm at 20 / 25 / 30 / 35° | SEPARATED, except D_p35_reach CONNECTED (90) | 0 | SEPARATED | D_p35_reach SEPARATED (90) |
| `resolution_002` OBL20, dx = 4s, all s and Δt | CONNECTED (38–214) | 0 | — | SEPARATED at every s and Δt (labels = bridges) |
| OBL20 s 0.75, dx 4 mm | CONNECTED (650) | 450 / 650 | — | CONNECTED |

- **Visibility rule.** A code-0 particle within 2dx of the plane takes its reference side only if every
  stencil-sharing particle on the other side is reached across the swept patch (the plane above the final edge height
  h). A ligament under the edge therefore always blocks it.
- **Lattice phase.** Every E1 A/R row has the plane through the node line x = 0, z = 0, where a segment-to-node test
  labels nothing; the 10° onset may be phase-specific (V3.2). The rebuilt stencil reproduces every recorded OBL20
  bridge count exactly.

### 2.5 P4: steady tool force on `resolution_002` (N; s 1 mm, dx 4 mm unless stated)

| Case | Δt 0.2 / 0.1 / 0.05 / 0.025 ms | Series fit F∞/(1+(Δt/Δt₀)²) on the two finest | Fit check at 0.1 / 0.2 ms |
| --- | --- | --- | --- |
| CUT | 0.0068 / 0.0145 / 0.0327 / 0.0521 | Δt₀ 0.050 ms, F∞ 0.065 | 0.0132 / 0.0039 (obs. 0.0145 / 0.0068) |
| OBL20 | 0.0128 / 0.0275 / 0.0593 / 0.0801 | Δt₀ 0.069 ms, F∞ 0.091 | 0.0293 / 0.0096 (obs. 0.0275 / 0.0128) |
| TRIM | 0.0602 / 0.0905 / 0.1053 / 0.1391 | does not fit (successive ratios 1.50, 1.16, 1.32) | second mechanism (V4.6) |
| CUT at 0.05 ms, s 2 / 1.5 / 1 / 0.75 / 0.5 (dx 4s) | 0.0006 / 0.0005 / 0.0327 / 0.0478 / 0.0508 | the jump between 1.5 and 1 mm is §1.5 (zone ≥ flank) | — |

Δt₀ ≈ 0.05 ms means K_p/s² = E/ℓ with ℓ ≈ 0.8 mm, about one particle at the edge. Blade overlap (1.200 / 0.430 /
0.119 / 0.035 mm) counts particles passing through the released zone, so it is not a compliance measure. This is
hypothesis H4 with predictions (V4.2–V4.4), not a convergence verdict.

## 3. Methods

**Per method:** what it is and its physical assumptions · how it defines **fracture or separation** · which P it
addresses and how · fit to TLMPM v5 · the reusable module · cost (S < 1 day, M 1–3 days, L > 1 week, each plus the
acceptance rerun) · risks · the verifying experiment (§5). Source status is in §7.

**3.1 MLS-MPM and CPIC (Hu et al. 2018).**
- **What it is.** Updated-Lagrangian MLS-MPM keeps APIC transfers. F ← (I + Δt C) F, with the APIC matrix C reused as
  the velocity gradient, and particles move with the grid-interpolated velocity, not a FLIP velocity.
- **CPIC.** Each step, rigid-surface samples splat a narrow-band distance, a per-surface affinity and a side tag to
  the nodes; particle tags come from the nodes. P2G goes only to compatible nodes (tags agree on every shared
  surface). Incompatible nodes return a projected ghost velocity (sticky, slip or separate, with friction), and the
  impulse goes to the rigid body. A weak penalty acts on penetrated particles.
- **Separation.** Purely geometric, from the *current* blade. There is no stress, energy or damage criterion, and no
  lasting crack (a particle's colour lasts only while its kernel nodes keep affinity), as the repository's CRESSim
  check found (`PERSISTENT_CUTTING.md:24–26`).
- **Addresses.** P1 indirectly (position from the grid), P3a (pair compatibility), P4 (projection, not a spring).
  **TL fit:** MLS adds little to precomputed reference gradients. CPIC splats in current space and TL nodes have no
  current position, so the swept blade must be pulled back to the reference, where a per-(particle, corner) tag can
  persist. **Modules:** (a) pair visibility for exit assignment (M; §2.4; V3.3); (b) velocity-projection contact (M–L).
- **Risks.** Per-particle codes only approximate pair coupling ("any pair" cut every h = 1.0 mm ligament, "all pairs"
  kept them). Projection changes the law-4 work input D_p: a new law version.

**3.2 APIC (Jiang et al. 2015), XPIC (Hammerquist & Nairn 2017) and the TLMPM reference code.**
- **APIC** conserves angular momentum, keeps PIC's stability without its dissipation, and moves positions with the
  grid. **XPIC(m)** filters the null space of the extrapolation; its Eq. 14 is the general position update.
- **de Vaucorbeil's TLMPM code (Karamelo)** moves positions with the grid velocity (`solid.cpp:576–615`, read). It
  blends velocities with FLIP fraction 0.99, updates F from Σ v_I ⊗ ∇₀w (double mapping optional) and has a TL-APIC
  sub-method. **In the transfer, v5 differs from it in the position update** (and FLIP fraction 0.95).
- **Fracture:** none. **Addresses** P1 and P2 (the same drift), P5 indirectly.
- **Modules.**
  - **3.2a grid-consistent position update.** x += Δt Σ w v^MUSL, one kernel after `_deformation` (S). Exact
    consistency needs x and Ḟ from the *same* grid velocity (v^MUSL under MUSL, v⁺ under USL). PROBE: first-layer
    P99 1.88 → 0.38 s, far field at the float floor. The S v⁺ form with MUSL F (Karamelo with double mapping, ≈ our
    α = 1) keeps the (I − SS⁺)S mismatch: 0.45 s.
  - **3.2b XPIC(2)** on the slot fields (M).
  - **3.2c TL-APIC** ported from Karamelo (M–L). Its TL conservation properties are undocumented in what was read.
- **Risks.** Contact and the law-8 assignment read x. A sub-cell face then cannot conform to the 1.5 mm blade on its
  own, so overlap, force and the TRIM residue may change. The label-timing remainder stays: in the PROBE, vertical
  links keep P99 0.39 s against 0.06 s for horizontal ones.
- **Experiments.** V1.2–V1.4, V2.3.

**3.3 CPDI / CPDI2 (Sadeghirad, Brannon et al. 2011, 2013).**
- **What it is.** CPDI convects each particle domain by F (r_i = F r_i⁰); CPDI2 tracks the corners and enriches weak
  material interfaces. Both cure UL extension instability and cell-crossing noise.
- **Separation.** They *prevent* numerical separation. Separation needs "domain freezing", which is grid-dependent
  and physically justified only together with a damage model (Homel 2013).
- **TL fit.** v5 has no cell crossing and no numerical fracture (the TLMPM abstract says the same), so CPDI addresses
  neither P1 nor P2.
- **P5.** Corners interpolated from the grid field are an F-consistent embedding: a **visual-only fix** on today's
  FLIP positions, the solver's own field after 3.2a (S; only after reopening).

**3.4 MPM fracture: explicit cracks, cohesive zones, damage fields.**
- **CRAMP (Nairn 2003).** Up to 3 velocity fields per node, chosen by a particle-to-node **line-crossing** test, with
  Bardenhagen crack-face contact. **No propagation criterion** ("future work").
- **Daphalapurkar et al. 2007.** A cohesive traction-separation law on a predefined weak interface.
- **Nairn 2015/2016 orthogonal cutting**, the closest analogue to our setup. An initial crack runs along the whole
  prescribed path with a trilinear cohesive law (toughness from fracture tests), and grows when the root opening
  reaches δc ("limited simulations to straight crack growth"). Rigid tool, Coulomb multimaterial contact, PIC damping.
- **Homel & Herbold 2017 (DFG).** The gradient of a kernel damage field (strain-softening Rankine, secondary source)
  re-partitions each node into two contact fields.
- **CD-MPM (Wolper et al. 2019).** UL single-field phase field driven by the history maximum of tensile energy (G, l₀),
  with degraded stress and no partitioning. AnisoMPM (2020) is anisotropic non-local damage. Johnson–Cook and
  phase-field TLMPM papers exist (2022; UNVERIFIED beyond abstracts).
- **Fracture definitions.** G_c, a traction-separation law or δc; or a damage or phase variable from stress or
  tensile energy. None uses particle distance. **Addresses** P2 only if chips really break (excluded by §1.4, §2.3),
  P3 through CRAMP visibility (= 3.1a), P4 partly (Nairn's cutting force is G_c plus plasticity, the analogue of D_p).
- **TL fit.** DFG maps onto v5's node fields (one damage-side bit per digit). A cohesive interface between
  conflicting fields matches Nairn's imperfect-interface law.
- **Risks.** A second, stress-driven separation source conflicts with law 8's labels unless it replaces them, which is
  a law change. Damage also needs gating against blunt contact.
- **Cost.** L, plus toughness and failure data the project lacks. V2.2 decides whether it is needed at all.

**3.5 MPM contact.**
- **Grid contact.** Bardenhagen et al. 2000/2001 use fields per body, mass-gradient normals, contact when
  (v_a − v_cm)·n > 0 and Coulomb friction; Huang et al. 2011 and Nairn et al. 2018 refine it. The resolution is about
  one cell (4 mm): usable for the board, not for the 1.5 mm blade.
- **TLMPM contact.** de Vaucorbeil & Nguyen 2021 is particle-to-particle. Karamelo's min-penetration fix uses
  f = m₁m₂/((m₁+m₂)Δt²)(1 − R/r)Δx, the same Δt-tied stiffness as v5 with k = 1 (agent-read; not confirmed as the
  paper's algorithm). The LS-DYNA soft constraint uses nodal mass/Δt² (secondary source).
- **Others.** Velocity projection (Stomakhin et al. 2013 §8, read): inelastic Coulomb, on grid velocities and again on
  particle velocities before the position update. IPC (Li et al. 2020): log barrier, implicit Newton, CCD line search.
  No study of MPM contact-force convergence under refinement was found (UNVERIFIED).
- **Fracture.** None. **Addresses** P4, and P3b if it is driven by compliance.
- **Modules.** (a) Physical penalty stiffness `penalty_area_v1` (S). (b) Particle velocity projection in `_g2p` before
  the x update (M; new law-4 work input). (c) Grid-field contact for the board (M). (d) IPC (L): replaces the
  integrator, **rejected**. Experiments V4.2–V4.5.

**3.6 BFEMP (Li, Fang, Li & Jiang, CMAME 390:114350, 2022).**
- **What the acronym means.** BFEMP is **"barrier FEMP": barrier-based FEM–MPM coupling**, not bonded or boundary
  FEM–MPM. No other meaning was found (search not exhaustive).
- **Method.** An IPC barrier b = −κ(d/d̂ − 1)² ln(d/d̂) acts between MPM particles and FEM boundary elements, with grid
  displacements as unknowns. It is solved fully implicitly by projected Newton with a CCD- and det F-safe line search,
  and is interpenetration-free "regardless of the time step size or the mesh resolutions". Cutting is future work.
- **Fracture:** none. **Addresses** P4 in principle (a prescribed knife is a kinematic barrier). **TL fit: rejected.** It needs an optimization-based implicit integrator, which would replace the solver. An explicit
  barrier gains nothing, since its stiffness at small d again bounds Δt. The transferable lesson is a contact distance
  declared in metres, not in s.

**3.7 Manifold Dual Contouring (Schaefer, Ju & Warren 2007) and divided voxels (Qi, Milef & De 2021).**
- **MDC.** Octree vertex clustering with a manifold test yields a 2-manifold of the original genus. It may still
  self-intersect and has no sub-cell features. Input is signed Hermite data, which `cut_events_v1` already gives.
- **Divided voxels.** A "divide" operator splits a voxel into two full-size voxels with the connectivity kept, and
  draws the cut from them; partial and progressive cuts, no collision response, deformation model UNVERIFIED. That is
  what `cut_aware_surface_v1`'s per-side duplication already does.
- **Fracture:** none (topological, along a given path). **Addresses** only the non-manifold part of P5. The folds and the faces nearer another piece are F3-layer
  embedding errors and stay. MDC is M, inside `cut_aware_surface`, after Phase 3 reopens.

**3.8 Cutting simulators.** DiSECt (Heiden et al. 2021) uses FEM, a prescribed cut mesh and virtual nodes. Its springs
lose stiffness with knife force (k′ = k − γ‖f_knife‖, fitted): damage driven by force. RoboCook is learned; chip
separation in Ambati et al. 2012 and TopoCut is UNVERIFIED. Every physics-based cutting simulator reviewed prescribes
the path, as law 8 does; its physics lies in the separation *resistance* (G_c, spring damage, D_p).

## 4. Comparison across P1–P5

**Legend.** **D** = removes the mechanism; **p** = partial or indirect; – = no effect; ✗ = would hide or worsen it.
Cost: S < 1 day, M 1–3 days, L > 1 week; each also needs the acceptance rerun.

| Method → module | P1 x vs F | P2 chips | P3 oblique exit | P4 force | P5 topology | Fracture defined by | Fits TL v5 | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.2a grid-consistent position update | **D** (identity, PROBE) | **D** (same drift) | p (P3b?) | p (contact geometry) | p (removes the cause) | none | ideal | S |
| 3.2b XPIC(m) | **D** for null-space noise | **D** | – | – | p | none | good (extra passes) | M |
| 3.2c TL-APIC (Karamelo has one) | **D** | **D** | – | – | p | none | port; TL conservation undocumented | M–L |
| 3.1a CPIC / CRAMP pair visibility | – | – | **D** for P3a (STATIC) | – | – | prescribed surface | via reference pull-back | M |
| 3.1b / 3.5b velocity projection | – | – | p | **D** (rigid limit) | – | – | needs node positions → particle-level variant | M–L |
| 3.3 CPDI / CPDI2 | – (TL has no cell crossing) | – | – | – | ✗ now / p after 3.3 (corner embedding) | none (prevents separation) | redundant for transfer | S (surface) |
| 3.4 CRAMP / cohesive / Nairn cutting | – | p (if chips break) | p (visibility only) | p (cutting force from G_c) | – | G_c, traction-separation | multi-field fits; law conflict | L |
| 3.4 DFG / CD-MPM / phase field | – | p (if chips break) | p (crush ligament; needs ε_f) | – | – | damage from stress or energy | DFG fits node fields | L |
| 3.5a physical penalty stiffness | – | – | p | **D** for the Δt confound | – | – | ideal | S |
| 3.5c grid multi-field contact | – | – | – | p (board only; dx-resolved) | – | – | good | M |
| 3.5d IPC / barrier | – | – | – | **D** | – | – | ✗ implicit | L (rejected) |
| 3.6 BFEMP (barrier FEM–MPM) | – | – | – | p | – | — | ✗ implicit, FEM coupling | L (rejected) |
| 3.7 manifold DC | – | – | – | – | p (manifoldness only) | — | surface only | M |
| 3.8 divided voxels | – | – | – | – | p (cut-cell topology) | prescribed | surface or embedding | M–L |

## 5. Root-cause verification plan (current solver; before any method is adopted)

Rules for every row: law 8 and `tlmpm_v5` unchanged unless the row is labelled COUNTERFACTUAL; a COUNTERFACTUAL row
lives in a scratch variant (never `tlmpm_v5`, never a profile), is labelled `COUNTERFACTUAL_NOT_A_GATE` and never
enters an acceptance verdict. Commit the protocol (`configs/wp12/phase4_rootcause_protocol_v1.json`) before the first
formal row; keep failures; never overwrite `resolution_002`, `phase4_e1_001` or any record. 1 mm / 4 mm / 0.1 ms,
seed 42, `cuda:1`, kinematic ideal tracking, unless a row says otherwise. Budgets use the measured walls of
`resolution_002` (CUT/TRIM/OBL20 at s 1 mm: 98–134 s at 0.1 ms, 381 s at 0.025 ms) and E1 (404 s per row).

| ID | Hypothesis tested | Cases and parameters | Metrics | Prediction (basis) | Confirms / refutes | Budget |
| --- | --- | --- | --- | --- | --- | --- |
| V1.1 | P1 is concentrated at cut faces and scales with Δt and particles per cell | read-only: every `resolution_002` row, FINAL and END_CUT states | per-link \|dx − F̄dX\|/s, P50/P99 in bands ≤1, 1–3, 3–6, >6 s from the declared plane; same for TRIM (z = 18 mm) | §2.1 values reproduce (REVIEWER) | confirm: first-band/far-field P99 ratio ≥ 5 everywhere; refute: ratio < 2 | CPU, 10 min |
| V1.2 | P1 = integrated FLIP null-space drift (identity of §1.1) | new physics-neutral recorder `flip_drift_v1`: after `_deformation`, d_p += Δt(v_p − Σ w v^MUSL) (reads `v`, `new_v`, `slot`, `weights`); CPU test: positions, F, codes bitwise equal with and without it; rows CUT, TRIM, OBL20, MULTI2 at 0.1 and 0.05 ms | fraction of Σ\|dx − F̄dX\|² explained by \|d_q − d_p\| on same-code links, per band; residual P99 | ≥ 0.95 in the first band, ≥ 0.999 beyond 6 s (CPU probe §2.2: 0.950 / 0.9976 / 1.0000) | confirm: as predicted; refute: < 0.8 in the first band (then slot-history or light-node mechanisms dominate; measure them next) | 8 rows, ≈ 0.4 GPU-h |
| V1.3 | the drift has a Δt·(1−α)/α part and an α-independent part | COUNTERFACTUAL pic α ∈ {0.25, 0.5, 1.0} on CUT, TRIM, OBL20 and the Phase 3 S7 dome; plus α per unit time held fixed, α(Δt) = 1 − 0.95^(Δt/0.1 ms), at 0.05 and 0.025 ms | V1.1 bands; chip scatter (V2.1); forces; components; energy | α = 1: first-band P99 2–5× lower but not at the float floor (probe 1.88 → 0.45 s); fixed-α-per-time rows lose most of the Δt trend | refute the decomposition if α = 1 leaves ≥ 0.8× the α = 0.05 value | 18 rows, ≈ 1 GPU-h |
| V1.4 | positions advected with the F velocity remove P1 | COUNTERFACTUAL kernel variant: x += Δt Σ w v^MUSL after `_deformation` (FLIP velocity kept); CUT, TRIM, OBL20, MULTI2, S7 dome, G3b; 1 mm and 0.5 mm (dx 4 s) | V1.1 bands; residual on links with different code-assignment substeps; components, bridges, TRIM removal; tool force; blade and board overlap; min det F; momentum/energy | same-history links at the float floor beyond 1 s; first band ≥ 3× lower (probe: P99 1.88 → 0.38 s, P50 0.173 → 0.059 s); CUT/MULTI2 separation unchanged | confirm: ≥ 2× on first-band P50 and P99 (the Phase 3 §3.3 reopen measure) with no lost separation; refute: < 2× | 12 rows, ≈ 2.3 GPU-h (0.5 mm rows ≈ 20 min) |
| V2.1 | chip "tears" are kinematic, not constitutive | read-only: TRIM rows of `resolution_002`, `phase3_surface_00{1,2}` S7 | per long link: \|dx\|/\|F̄dX\|; endpoint principal stretches and det F; components by the solver's own stencil graph | ≥ 95 % of links with \|dx\| > 2 s have \|F̄dX\| < 1.25 s (§2.3: F stretch ≤ 1.11) | refute if a majority of long links are F-consistent (then the chip is really deformed and V2.2 decides) | CPU, 10 min |
| V2.2 | no fracture criterion is reached in chips under the provisional material | physics-neutral final-state export of `Fpi`, `plastic`, stress for TRIM and S7 | elastic energy density ψ, von Mises, pressure, ε_p in chip vs substrate; Griffith check t·ψ vs G_c = 0.05 J/m² (provisional); report ε_p distribution (no ε_f is calibrated) | ψ ≤ ≈ 9 J/m³ in unconfined chips (J2 bound σ_y²/6μ), so a Griffith tear needs t ≥ G_c/ψ ≈ 6 mm | if ψ·t ≥ G_c for chips < 6 mm, a fracture model becomes a physics question (§6 step 6) | 4 rows, ≈ 0.2 GPU-h |
| V2.3 | the P1 fix removes chip scatter | V1.4 rows TRIM and S7 | V2.1 metrics | long-link fraction falls ≥ 5× | refute if scatter stays (then sub-cell chip motion needs its own study) | in V1.4 |
| V3.1 | flat-board oblique bridges have two mechanisms (P3a in-core wedge, P3b mis-sided labels) | read-only: all completed E1 rows and `resolution_002` OBL20 | bridges that need a digit opposite to the particle's reference side; shares per row | §2.4 (A_p10 96/96 P3b; A_p15–25/35 0; A_p30 228/380; OBL20 g4s 0) | — (classification) | CPU, 10 min |
| V3.2 | onset and P3a/P3b split depend on lattice phase | new E1b rows: plane_x0 ∈ {0.5, 1.0, 2.0} mm at τ ∈ {10, 20, 30}°, h = 0 (E1 has only x0 = 0, where the plane contains the node line x = 0, z = 0) | E1 metrics + V3.1 split | onset moves by ≥ 5° with phase; P3b share changes | refute if every phase gives the E1 onset (10°) and split | 9 rows, ≈ 1 GPU-h |
| V3.3 | a CRAMP-style visibility test separates swept wedges but not ligaments | shadow evaluator `exit_candidates_v2`: per substep, a code-0 particle takes its reference side only if every stencil-sharing opposite-side particle is reached across the swept patch (recorded edge history, pulled back to reference); E1, E1b, N1–N9 of the review §6 | predicted components and bridges; labels below the edge stop; labels per row | STATIC on E1/OBL20 (§2.4): separates ±15/20/25/35°, connected at h ≥ 0.75 mm, ≤ 300 labels; fails A_p10 and A_p30 (P3b) | refute if any N-control or h ≥ 0.75 mm row separates, or > 1 000 labels per exit | CPU on saved E1 + 9 E1b rows |
| V3.4 | P3b is caused by displacement ahead of the edge (contact compliance, P1) | A_p10, A_p30 at 0.05 ms; COUNTERFACTUAL k = 0.4; V1.4 variant | mis-sided labels, their s_ref and displacement at assignment; bridges | direction not predicted; measured | P3b is a law-input problem if it persists under stiff contact and grid-consistent x | 6 rows, ≈ 1 GPU-h |
| V4.1 | the Δt dependence of force is series penalty compliance K_p = k m/Δt² | read-only fit on CUT, OBL20, MULTI2 Δt series (s 1 mm) | F(Δt) = F∞/(1+(Δt/Δt₀)²) | CUT Δt₀ ≈ 0.050 ms, F∞ ≈ 0.065 N; OBL20 0.069 ms, 0.091 N (§2.5) | — (fit; predicts V4.2–V4.4) | CPU |
| V4.2 | same K_p gives the same force at any Δt | COUNTERFACTUAL k ∈ {0.2, 0.4} at 0.1 ms; CUT, OBL20, TRIM | steady \|F\|; overlap outside the cutting zone; momentum | F(k 0.4, 0.1 ms) = F(k 0.1, 0.05 ms) ± 10 %: CUT 0.0327, OBL20 0.0593 N | refute if off by > 25 % (another Δt effect: α per substep, work integration); the dashpot k m/Δt differs by 2× (ζ = √k/2: 0.32 vs 0.16), reported | 6 rows, ≈ 0.25 GPU-h |
| V4.3 | fixed physical stiffness makes force Δt-invariant | COUNTERFACTUAL k = 0.1·(Δt/0.1 ms)² at 0.05 and 0.025 ms | as V4.2 | CUT 0.0145 N, OBL20 0.0275 N ± 10 % | refute if > 25 % drift | 6 rows, ≈ 0.6 GPU-h |
| V4.4 | the frozen formula is pre-asymptotic, not divergent | formal law, Δt 0.0125 ms (1 600 substeps per control step), CUT and OBL20 | as V4.2 | CUT ≈ 0.061 N, OBL20 ≈ 0.087 N | refute if < 0.055 / 0.080 N or above F∞ | 2 rows, ≈ 0.4 GPU-h |
| V4.5 | the coarse-spacing force collapse is the cutting zone \|z\| ≤ s | recorder split of the tool reaction into in-zone and out-of-zone contacts; CUT at s ∈ {0.75, 1, 1.5, 2} mm, 0.05 ms; COUNTERFACTUAL zone = 1 mm in metres | force by zone and by direction (advance, drag, normal) | for s ≥ 1.36 mm all flank contacts are in the zone (blade half-thickness 0.75 mm + 0.45 s ≤ s) and carry no advance force | refute if out-of-zone contacts exist and carry force at s ≥ 1.5 mm | 8 rows, ≈ 0.3 GPU-h |
| V4.6 | TRIM force has a second Δt mechanism | force decomposition of V4.5 on TRIM at 0.2–0.025 ms; α per unit time fixed | as V4.5 | TRIM does not fit V4.1 (ratios 1.50 / 1.16 / 1.32) | — (attribution) | 4 rows |

**Reading order of the outcomes.** V1.2 + V1.4 decide P1 and, through V2.1–V2.3, P2. V4.2–V4.5 decide P4. V3.1–V3.4
decide what an exit law (if any) must do. P5 has no row of its own: it is re-measured with the
Phase 3 diagnostic script on V1.4 states, and reopens only under the Phase 3 §3.3 criterion with a supported (not
counterfactual) configuration that passed the full law-8 acceptance.

## 6. Recommendation and order of work

1. **Now: report-only, about 5 GPU-hours plus CPU.**
   - Fold the §8 scripts into a repository diagnostic.
   - Run V1.1 and V3.1.
   - Add the `flip_drift_v1` recorder with its bitwise-neutrality CPU test, then run V1.2.
   - Run V4.1–V4.4 (cheap; they decide whether P4 is compliance only), then V1.4 and V1.3.
2. **With the completed E1.** Run V3.2 and V3.3 (with N1–N9); decision D1 stays at xhigh. **Do not implement L9-EL as
   designed**: its shadow already cuts unswept ligaments at τ = 20–35°, h = 0.75–1.0 mm (§2.4), which fails the
   review's own control N3.
3. **First prototype, only if V1.2 confirms and V1.4 passes: grid-consistent position update.**
   - x += Δt Σ w v^MUSL, FLIP velocity kept. Opt-in, either as `tlmpm_v6` or as a v5 option that defaults off and
     stays bitwise.
   - This is the standard MPM / TLMPM-reference update (§1.1, §3.2), not a new law. It still changes contact and
     labels, because both read x.
   - Acceptance: CPU bitwise identity with the option off; the full ADR 0006 matrix and Phase F rows; the controls
     (side push, flat press, stop, withdrawal, partial cut, blunt handle); CUT/TRIM non-regression; energy, reaction
     and momentum against law 8; the `resolution_002` subset at 1 / 0.75 / 0.5 mm.
   - Phase 3 reopens only if first-layer P50 and P99 both fall ≥ 2× under this *supported* configuration.
4. **Second prototype, only if V4.2/V4.3 confirm H4: Δt-independent contact stiffness.**
   - `penalty_area_v1`: K_A declared in Pa/m, so k = K_A s²Δt²/m, guarded by K_p ≤ 0.5 m/Δt²; opt-in, K_A provisional.
   - Velocity projection (§3.5b) is the follow-up if a rigid limit is wanted. It needs a new law-4 work input.
   - Same acceptance as above. Rerun force convergence with the zone confound (V4.5) removed from the s-path.
5. **P3a law-9 input, only if D1 needs a law: CRAMP-style visibility.** Run it in shadow (`exit_candidates_v2`) on
   E1, E1b and N1–N9. It must keep h ≥ 0.75 mm connected, label at most a few hundred particles and never fire in a
   control. P3b waits for V3.4; it may vanish under prototypes 1 and 2.
6. **Not now.**
   - Cohesive or damage fracture: no confirmed problem is fracture, and there is no calibrated toughness. Revisit if
     V2.2 finds ψ·t ≥ G_c in chips thinner than 6 mm, or if the user wants chip breakage as a capability (new ADR,
     data, controls).
   - XPIC / TL-APIC: only if prototype 1 leaves velocity-noise effects.
   - MDC and divided voxels: after Phase 3 reopens.
   - IPC and BFEMP: rejected.

## 7. Sources and verification status

**R** = read by this reviewer (full text or source). **A** = read in primary text or source by the literature agent,
DOIs checked against Crossref. **Ab** = abstract or secondary only. **U** = UNVERIFIED.

| Group | Sources |
| --- | --- |
| Transfers, read here (R) | Hammerquist & Nairn, CMAME 318:724 (2017), XPIC, [pdf](https://www.cof.orst.edu/cof/wse/faculty/Nairn/papers/XPICPaper.pdf). Stomakhin et al., TOG 32(4):102 (2013), [pdf](https://www.math.ucdavis.edu/~jteran/papers/SSCTS13.pdf). Karamelo TLMPM source, [github](https://github.com/adevaucorbeil/karamelo) (`src/solid.cpp`, `src/tlmpm.cpp`) |
| Transfers (A) | Hu et al., TOG 37(4) (2018), MLS-MPM/CPIC, doi:10.1145/3197517.3201293, [pdf](https://par.nsf.gov/servlets/purl/10088895). Jiang et al., TOG 34(4) (2015), APIC, doi:10.1145/2766996. Sadeghirad et al., IJNME 86:1435 (2011), CPDI, doi:10.1002/nme.3110. Sadeghirad et al., IJNME 95:928 (2013), CPDI2, doi:10.1002/nme.4526. Homel, CPDI domain-freezing slides (2013), [pdf](https://csm.mech.utah.edu/content/wp-content/uploads/2011/02/130313_MPMworkshop_cpdiDomainFreezing_Homel.pdf) |
| TLMPM papers (Ab) | de Vaucorbeil, Nguyen & Hutchinson, CMAME 360:112783 (2020), doi:10.1016/j.cma.2019.112783. de Vaucorbeil & Nguyen, CMAME 373:113503 (2021), doi:10.1016/j.cma.2020.113503. Bui et al., IJNME (2025), doi:10.1002/nme.70105. Damage and phase-field TLMPM (2022): doi:10.1016/j.cma.2021.114388 and doi:10.1016/j.cma.2022.115234 |
| Fracture (A) | Nairn, CMES 4:649 (2003), CRAMP, [pdf](https://www.cof.orst.edu/cof/wse/faculty/Nairn/papers/MPMCracks.pdf). Nairn, EFM 149:262 (2015), doi:10.1016/j.engfracmech.2015.07.014, and Interface Focus 6:20150110 (2016), [pdf](https://www.cof.orst.edu/cof/wse/faculty/Nairn/papers/WoodCutting.pdf). Daphalapurkar et al., IJF 143:79 (2007), doi:10.1007/s10704-007-9051-z. Homel & Herbold, IJNME 109:1013 (2017), doi:10.1002/nme.5317. Wolper et al., TOG 38(4) (2019), CD-MPM, doi:10.1145/3306346.3322949 |
| Fracture (Ab) | AnisoMPM (2020), doi:10.1145/3386569.3392428. Moutsanidis, Long & Zhang, CMAME 345:584 (2019), doi:10.1016/j.cma.2018.11.005. Kakouris & Triantafyllou, IJNME 112:1750 (2017), doi:10.1002/nme.5580 |
| Contact | Bardenhagen et al., CMAME 187:529 (2000), doi:10.1016/S0045-7825(99)00338-2, and CMES 2:509 (2001) (A). Li et al., IPC, TOG 39(4) (2020), doi:10.1145/3386569.3392425 (A). Huang et al., IJNME 85:498 (2011), doi:10.1002/nme.2981 (Ab). Nairn et al., CPM 5:285 (2018), doi:10.1007/s40571-017-0168-1 (Ab). Guilkey et al. (2021), doi:10.1016/j.cma.2021.113739 (Ab) |
| BFEMP, surfaces | Li, Fang, Li & Jiang, CMAME 390:114350 (2022), BFEMP, doi:10.1016/j.cma.2021.114350, [arXiv](https://arxiv.org/abs/2108.03349) (A). Schaefer, Ju & Warren, TVCG 13:610 (2007), MDC, doi:10.1109/TVCG.2007.1012 (A). Qi, Milef & De, Vis. Comput. 37:1113 (2021), divided voxels, doi:10.1007/s00371-020-01856-y (Ab; its deformation model U) |
| Cutting | Heiden et al., RSS (2021), DiSECt, [arXiv](https://arxiv.org/abs/2105.12244) (A). Chip separation in Ambati et al. (2012), doi:10.1016/j.commatsci.2011.06.018, in TopoCut and in RoboCook is U |

## 8. Reviewer scripts and commands

Scripts live in `/mnt/data/ningan/tmp/claude-1004/-home-ningan-RoboticCuttingSim/2dbee09e-127d-4db1-a9d9-f2a32de8d08d/scratchpad/methods/`.
They run from the repository root with `.venv/bin/python -I` (Warp 1.17.0, NumPy 2.5.4), CPU only, and write nothing
to the repository or any run directory. Fold them into a repository diagnostic before citing their numbers.

| Scripts | Section and input |
| --- | --- |
| `p1_links.py`, `p2_chip.py` `<rows>` | §2.1, §2.3 (`resolution_002/<row>/state.npz`) |
| `p1_identity_probe.py {base,pic1,xgrid} <out.npz>`, `p1_identity_eval.py <npz…>` | §2.2 (self-built bar on `cpu`, about 39 s each) |
| `p3_mis_sided.py`, `p3_visibility_all.py`, `p3_visibility.py` (any-pair), `p3_crossing*.py` `<E1 rows>` | §2.4 (E1 `snapshots.npz`, `result.json`) |
| `p3_obl20.py`, `p3_obl20_vis.py` `<rows>` | §2.4 OBL20 (stencil rebuilt from the reference) |
