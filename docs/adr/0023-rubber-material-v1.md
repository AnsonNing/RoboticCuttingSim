# ADR 0023 — Rubber material v1 (Phase 2, P2-0): backend, constitutive model, near-incompressibility, viscoelasticity, contact, scene/API

Date: 2026-11-10. **Status: accepted (rcs-xhigh, 2026-11-10)** for the binding decisions D1–D6 and D9–D11 (backend,
constitutive model, near-incompressibility, time integration, resolution, boundary conditions, code organisation,
cost rules, P2-0 deliverables). D7 (viscoelasticity, P2-3) and D8 (contact and coupling, P2-4) are accepted as
**designs**: their formulas, layouts and test protocols are binding, but each needs its own probe evidence before it
is implemented as accepted physics. Plan decision D-12 routes this decision here and the implementation to rcs-impl.
**Amended 2026-11-10 by §18 (P2-1 follow-up):** stable-step guard and deformation reserve (§7 step rule),
static_stop_v3 (§7 initializer), RUBBER-PHYS-07 v3 / -12 (`rubber_phys_tests_v3.json`); no self-contact in v1.

Nothing here changes a frozen solver, profile, threshold, evaluator or record. v1–v7_2 (including
`tlmpm_v7_2_contacts.py`, `tlmpm_v7_2_frontier.py`), law 8, the v1 evaluator, evaluation v2, final_v4, camera rigs
v1–v5, WP-11..WP-17 evidence and every Phase 1 batched module stay byte-identical — **including
`batched/constitutive.py`, `batched/layout.py`, `batched/material_batch.py`, `batched/tlmpm_v7_2_batch.py`,
`configs/rl_benchmark/*` and `scripts/rl_ready/*`** (the batched modules' SHA-256 values are in the v7_2 batch
receipts; the two directories belong to the P1-6 `acceptance_v1` evaluator file set). Everything new is versioned and opt-in. Material parameters are provisional:
`calibration_status = SIMULATOR_ONLY_UNCALIBRATED`; there is no real rubber data on this machine. Geometric
deletion is not cutting; this ADR covers **no cutting** of rubber at all.

Path abbreviations: `RCS/` = `src/robotic_cutting_sim/`, `NW/` = `.deps/newton/newton/_src/` (1.6.1),
`IL/` = `.deps/IsaacLab/source/`, `PR/` = `scripts/rubber/p2_0_probes/` (the probes of this ADR; deliberately not under
`scripts/rl_ready/`, whose files belong to the P1-6 `acceptance_v1` evaluator set).

## 1. Context (verified facts)

| Fact | Evidence |
|---|---|
| The TLMPM grid is the **reference** grid: stencil `nodes/weights/gradients` (trilinear, one cell, 8 corners) is built once from the reference particle positions and never changes | `RCS/tlmpm_v5.py:734-750` |
| Internal force uses the PK1 stress and the reference gradients: `gj += w(impulse + m g dt) − dt V (P ∇N)` | `RCS/tlmpm_v5.py:501-513` |
| `F += dt Σ v_I ⊗ ∇N_I` from the MUSL grid velocity; v6 moves `x` with the same grid field | `RCS/tlmpm_v5.py:604-613`; `RCS/tlmpm_v6.py:58-71` |
| The frozen stress is Hencky log strain + perfect J2 plasticity with a singular-value clamp at 0.05 | `RCS/tlmpm_v5.py:77-99`; batched copy `RCS/batched/constitutive.py:56-80` |
| No grid Dirichlet condition exists; all boundaries (blade, handle/pads, fixtures, table, particle–particle) are particle impulses from the projection/Coulomb helper `_contact_impulse` (normal: fraction `stiffness` of the predicted penetration per substep; friction capped at `m·|v_t|`) | `RCS/tlmpm_v5.py:154-165, 298-327, 472-497`; `RCS/fixture_contacts.py:11-28` |
| Particle self-contact excludes pairs whose **reference** distance is ≤ 3·dx unless their cut codes conflict; with all codes 0 it is a pure self-contact | `RCS/tlmpm_v5.py:477-497` (call `:856`) |
| Default FLIP/PIC blend `pic = 0.05` | `RCS/material_config.py:13` |
| Existing stability rule: longitudinal-wave grid CFL `dt·c/dx ≤ 0.2` on the 4 mm grid; clay `E = 50 kPa` gives 0.17 at 0.1 ms | `configs/materials/materials_wp12_v1.json` (`stability_rule`) |
| Registry (ADR 0021 §11): model id, per-env params, per-particle state (initial `identity` or `zero` only), stress launch at substep step 6; the only model is `log_strain_von_mises_plastic_v5` with the v7_2 parameter table `params[N,9]` | `RCS/batched/constitutive.py:20-31, 85-121`; reset `RCS/batched/material_batch.py:962-964` |
| `configs/materials/elastic_control.yaml` is an uncalibrated clay-like control (E 50 kPa, ν 0.3, ρ 1500), **not** rubber | file contents |
| One device `wp.HashGrid` per process is enforced | `RCS/batched/material_batch.py:498-511` |

**Newton 1.6.1 deformables** (read from source; summary of a source review, spot-checked):

| Fact | Evidence |
|---|---|
| `SolverVBD` supports tetrahedra with a stable-Neo-Hookean-type energy `P = V0[μF + λ'(J−α) cof F]`, `λ' = λ+μ`; per-tet `[k_mu, k_lambda, k_damp]`; stateless ("tet/cloth elasticity is stateless"), so no viscoelastic internal variables | `NW/solvers/vbd/particle_vbd_kernels.py:210-247`; `NW/solvers/vbd/solver_vbd.py:2396-2400` |
| VBD is one implicit-Euler step solved by a **fixed** number of colour-wise Gauss–Seidel sweeps (default 10), no convergence test | `NW/solvers/vbd/solver_vbd.py:300-307` |
| `add_soft_grid` gives **every node a full cell mass** (`cell_x·cell_y·cell_z·density`) | `NW/sim/builder.py:10755-10777` |
| Particle–shape soft contact is filtered by world; VBD self-contact is world-grouped; SemiImplicit/XPBD particle contacts are not | review: `NW/sim/collide.py:980-993`, `NW/geometry/kernels.py:1144-1150, 1336-1367`, `NW/solvers/semi_implicit/kernels_contact.py:36-85` |
| With an external rigid solver VBD coupling is one-way; two-way coupling to MuJoCo-Warp goes through the experimental `solvers/coupled` proxy (lagged impulses) | review: `NW/solvers/vbd/solver_vbd.py:365-366, 558-561`; `NW/solvers/coupled/__init__.py:4-7`, `solver_coupled_proxy.py:1369-1420` |
| Isaac Lab ships `Isaac-Lift-Soft-Franka` (MJWarp Franka + VBD tet beam 0.3×0.04×0.04 m, E 2e5, ν 0.3, proxy coupling, 2 substeps) | review: `IL/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/franka_soft_env_cfg.py:88-173` |
| No analytic tet-cantilever test exists in Newton (only rod/cable Euler–Bernoulli tests and a qualitative soft-body hang) | review: `.deps/newton/newton/tests/test_cable.py:6602`, `test_softbody_simulation.py:92` |

## 2. Probe evidence (P2-0, this ADR)

All probes are small and throwaway in spirit but kept for provenance: scripts in `PR/`, raw outputs in
`artifacts/rl_ready/p2_0/rubber_probes_001/`, a compact summary in `docs/evidence/rl_ready/p2_0_rubber_probes_001.json`.
CPU: AMD Threadripper PRO 9975WX, `.venv` Python 3.12, Warp 1.17.0, NumPy 2.5.4, SciPy 1.18.1; Newton probe in
`.deps/isaaclab-venv` (Newton 1.6.1, Warp 1.17.0), `CUDA_VISIBLE_DEVICES=''`. Material in the TLMPM probes: the
proposed `neo_hookean_jbar_v1` (§5) implemented as probe kernels next to the **unchanged** frozen kernels
`tlmpm_v5._p2g/_grid_update/_g2p/_musl/_deformation` and `tlmpm_v6._grid_positions`, S = 1 slot, codes 0, 2 particles
per cell per axis (spacing dx/2), nodal clamp at X ≤ 0, no contact, no damping unless stated.

### E1 — locking of the TLMPM operator in bending (linearised, `PR/probe_linear_cantilever.py`, `probe_linear_variants.py`)

The frozen stencil is a Q1 element with particle quadrature, so the small-deflection static response is
`K u = f` with `K = Σ_p V_p B_pᵀ D B_p`. The J-bar variant integrates the volumetric part with the dilatation
averaged over each reference cell (§5). Clamped square-section cantilever under gravity, tip deflection / Euler–Bernoulli
(E from κ, μ):

| L/t | β = κ/μ (ν) | J-bar | n_t = 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| 10 | 2.17 (0.30) | no | 0.836 | 0.941 | 0.968 | 0.980 | 0.989 | 0.993 |
| 10 | 20 (0.4754) | no | 0.595 | 0.752 | 0.837 | 0.884 | 0.929 | 0.950 |
| 10 | 20 (0.4754) | **yes** | 1.767 | 1.004 | 0.987 | 0.983 | 0.983 | 0.983 |
| 10 | 1000 (0.4995) | no | 0.217 | 0.358 | 0.454 | 0.520 | 0.605 | 0.662 |
| 10 | 1000 (0.4995) | **yes** | 1.796 | 0.998 | 0.983 | 0.980 | 0.980 | 0.980 |
| 20 | 20 | **yes** | – | 1.020 | 0.999 | 0.993 | 0.990 | – |
| 20 | 20 | no | – | 0.791 | – | 0.908 | 0.946 | – |

Other rows (L/t 10, β 20, J-bar): 1 particle per cell gives 1.299 / 1.098 / 1.039 at n_t 2/3/4 (under-integrated,
too soft) — **rejected**; 3 per cell gives 0.972 / 0.973 / 0.976 (2 per cell is adequate). Wide strip W = 4t:
0.961 / 0.933 / 0.926 (converging between the narrow-beam E and the plate E/(1−ν²) brackets). The converged 3-D value
sits 1–2 % below beam theory because the clamp suppresses lateral contraction at the root.
**Findings:** without J-bar the operator locks volumetrically even at β = 20 (−12 % at n_t = 4); with J-bar the error to
the converged value is ≤ 0.5 % at n_t ≥ 3 and 2–4 % at n_t = 2; n_t = 1 is unusable.

### E2 — explicit stability (`PR/probe_tlmpm_dynamic.py stability`)

Cantilever L 100 mm, W = t = 10 mm, μ 2 MPa, β 20, ρ 1150, n_t 3, J-bar, pic 0. With
`c_p = sqrt((κ+4μ/3)/ρ)` and `CFL = dt·c_p/dx`: CFL 0.5–1.2 finite and bounded (0.05 s, up to 2,408 steps);
CFL 1.5, 1.8, 2.2, 3.0 non-finite within 50–100 steps. **Measured critical CFL ∈ (1.2, 1.5)**.

### E3 — dynamic cantilever, energy, frequency and the PIC trap (`PR/probe_tlmpm_dynamic.py dynamic cpu 3 0.3`)

Same beam, released straight under gravity (step load), 0.3 s ≈ 3.5 periods, n_t 3 (2,232 particles):

| pic | CFL (dt) | mid-swing deflection | f₁ | half-amplitude first → last quarter | energy (J) first → last |
|---|---|---|---|---|---|
| **0** | 0.5 (8.65 µs) | −2.847 mm | **11.54 Hz** | 2.868 → 2.801 mm | 1.8e-8 → −8e-12 |
| **0** | 0.25 (4.33 µs) | −2.847 mm | 11.54 Hz | 2.868 → 2.801 mm | 1.2e-8 → −8e-8 |
| 0.05 | 0.5 | −2.465 mm | – | 2.465 → 0.475 mm | −2.6e-5 → −6.4e-5 |
| 0.05 | 0.25 | −2.171 mm | – | 2.171 → 0.100 mm | −4.1e-5 → −6.3e-5 |

References: linearised TLMPM (E1, n_t 3) −2.830 mm; Euler–Bernoulli static −2.867 mm, f₁ = 11.57 Hz. Peak elastic
energy ≈ 2e-4 J (weight × mean drop at maximum deflection), so with pic 0 the total-energy drift (≤ 1e-7 J) is
< 0.05 % and the response is dt-independent to four digits; the 2 % difference between first- and last-quarter
half-amplitudes is higher-mode beating over non-integer periods, not energy loss.
**pic = 0.05 is a numerical damper whose strength grows as dt shrinks** (more PIC applications per second): it removed
≈ 6e-5 J (≈ 30 % of the peak elastic energy) in 0.3 s, and halving dt cut the remaining amplitude from 19 % to 5 %. Under the specification this would be numerical damping mislabelled as
material behaviour, so rubber runs with pic = 0 (D4).

### E4 — large free bending vs the elastica (`PR/probe_large_sag.py`, `PR/diag_large.py`)

L 100 mm, W = t = 5 mm (L/t 20), μ 0.2 MPa, β 20, ρ 1150: load parameter `α = ρgAL³/(EI) = 9.18`. Reference: the
inextensible Euler elastica (BVP, `θ'' = α(1−s)cosθ`), tip drop 67.82 mm (0.678 L); linear theory would give 114.7 mm.
Static equilibrium by **probe-only** kinetic damping (particle velocities zeroed at kinetic-energy peaks; an
equilibrium initializer, not dynamics):

| n_t | CFL | tip drop | / elastica | J (min, max) | J̄ (min, max) |
|---|---|---|---|---|---|
| 2 | 0.25 | 68.10 mm | 1.004 | 0.965, 1.035 | 0.996, 1.004 |
| 3 | 0.5 | 67.69 mm | 0.998 | 0.963, 1.037 | 0.995, 1.005 |
| 4 | 0.25 | 67.59 mm | 0.997 | 0.962, 1.038 | 0.994, 1.006 |

Undamped large swings (no initializer) are stable: n_t 2, CFL 0.5, 2.0 s (≈ 97k steps, 5 swings, J ∈ [0.94, 1.06]);
n_t 3, CFL 0.5, 1.5 s. **Preserved failure:** the kinetic-damping initializer at n_t 2 and CFL 0.5 became non-finite
(also with J-bar off), while the same initializer at CFL 0.25 and the undamped run at CFL 0.5 are stable (n_t 3 at
CFL 0.5 with the initializer happened to converge). Zeroing
velocities of a leapfrog-type scheme near its stability limit can raise the shadow energy of high-frequency modes; the
initializer must therefore run at CFL ≤ 0.25 or use mass-proportional viscous damping (D4). The physics step itself
is not implicated. Logs: `large_sag_23_cfl05_kineticdamping.log`, `diag_damped_nojbar.log`.

### E5 — analytic checks of the energy (`PR/probe_analytic.py`)

Finite-difference check of `P = ∂W/∂F` (200 random F, J > 0.2): max relative error 6.2e-10. Uniaxial stress of the
compressible model vs the incompressible Neo-Hookean `P = μ(λ − λ⁻²)`:

| β (ν) | J at λ = 1.25 / 1.5 / 2 / 3 | P/P_inc at λ = 1.5 / 2 |
|---|---|---|
| 10 (0.4516) | 1.024 / 1.049 / 1.103 / 1.225 | 0.949 / 0.923 |
| **20 (0.4754)** | **1.012 / 1.025 / 1.054 / 1.125** | **0.973 / 0.958** |
| 50 (0.4901) | 1.005 / 1.010 / 1.023 / 1.054 | 0.989 / 0.982 |
| 3000 (0.4998) | 1.000 / 1.000 / 1.000 / 1.001 | 1.000 / 1.000 |

### E6 — Newton VBD cantilever (`PR/probe_vbd_cantilever.py`, Newton 1.6.1, Warp CPU)

Same beam as E2/E3 (100×10×10 mm, μ 2 MPa, λ = 19.33 μ, ρ 1150, Euler–Bernoulli static tip 2.87 mm, f₁ 11.57 Hz):
`add_soft_grid` with 2 mm cells (1,836 nodes, 6,250 tets), `fix_left`, `k_damp = 0`, no contacts, masses re-lumped to
ρV/8 per cell corner because of the builder mass rule; 0.6 s ≈ 7 periods.

| dt | iterations | mean tip, last period | / EB static | half-amplitude first → last period |
|---|---|---|---|---|
| 4.17 ms | 10 | −306 mm | 107 | 19.6 → 263 mm (growing) |
| 1 ms | 10 | −77.0 mm | 26.8 | 17.9 → 7.0 mm |
| 0.25 ms | 10 | −77.4 mm | 27.0 | 24.5 → 20.0 mm |
| 1 ms | 100 | −63.3 mm | 22.1 | 24.1 → 25.7 mm |
| 4.17 ms | 100 | −110.5 mm | 38.5 | 19.5 → 17.3 mm |

The response is 22–107× softer than the static beam solution and depends on dt and the iteration count (more
iterations → stiffer), which is consistent with non-converged Gauss–Seidel: for this stiff slender beam the inertia
diagonal `m/dt² ≈ 9 N/m` (dt 1 ms) is negligible against the elastic node stiffness `≈ μh ≈ 4 kN/m`, so 10–100 sweeps
cannot transmit bending stiffness along 50 nodes. **Caveat (recorded honestly):** an independent cross-check of the
Newton model setup with `SolverSemiImplicit` was inconclusive (its default particle/triangle contacts were active and the
run left the domain within 2 ms), so the setup itself is not independently verified. The backend decision (§4.3) does
not rest on E6 alone. Isaac Lab's soft-lift template works in a far softer regime (E 0.2 MPa, 40 mm section, coarse mesh).

### E7 — cost of the lean rubber substep (`PR/probe_tlmpm_dynamic.py timing`, `PR/cpu_cost_ratio.py`)

**GPU (RTX 5090, GPU0 under `gpu0.lock`, SHARED_GPU: 98 % co-tenant utilisation before and after).** One CUDA graph of
100 lean substeps (J-bar, NH stress, P2G, grid update, clamp, G2P, MUSL, grid update, clamp, deformation, positions;
**no contacts**), 5 replays:

| strip | particles | nodes (S = 1) | ms per substep (median, range) |
|---|---|---|---|
| 200×20×4 mm, dx 1 mm (n_t 4) | 128,640 | 46,350 | **0.41** (0.33–0.53) |
| 200×20×4 mm, dx 2 mm (n_t 2) | 16,160 | 11,130 | 0.125 (0.12–0.17), launch-latency floor |

A third row meant as a large point doubled L, W, t and dx together and so repeated 128,640 particles (0.41 ms); a
follow-up ~1M-particle run found the lock busy and was not queued. Comparison under similar sharing: the batched v7_2
N = 1 TRIM substep (103k particles) measured 1.35 ms (`ab_timing_001`, ADR 0021 notes); unshared Native v7_2 0.65 ms.
**CPU work proxy** (Warp CPU, serial): v7_2 1,390 ns vs lean rubber 326 ns per particle-substep (4.3×).
**Consequence:** the lean substep is ≈ 3–4× cheaper than v7_2 per particle, but a 20 ms control step needs ≈ 4,000 of
them (vs 200): ≈ 1.6 s per control step at N = 1 for 128k particles on a shared GPU (≈ 6× the shared v7_2 TRIM step),
≈ 0.6 s for a 40k-particle `rubber_rl_v1` strip (dx 1.33 mm, ≈ 2,900 substeps). Contacts, BC reactions, Prony terms and batching (R2.4) are not yet in
these numbers.

## 3. Decisions (summary)

- **D1 Backend:** option (a). A new opt-in TLMPM rubber solver `tlmpm_rubber_v1` (single-env reference) and its batched
  execution `tlmpm_rubber_v1_batch` in the same flat per-env layout, registry, masked reset, CUDA-graph, delayed-wrench
  and vector-env contract as Phase 1. Newton VBD/FEM is **not** adopted as a material backend (option b rejected
  for now; §4.3). No "both".
- **D2 Model:** `neo_hookean_jbar_v1`: decoupled compressible Neo-Hookean, isochoric part per particle, volumetric part
  `U(J) = κ/4 (J² − 1 − 2 ln J)` on the reference-cell-averaged dilatation J̄ (mean dilatation / F-bar), energy-consistent
  PK1 (§5). Mooney–Rivlin / Ogden later as separate model ids only when data asks for them.
- **D3 Near-incompressibility:** J-bar **plus** an explicitly reduced numerical bulk modulus `κ = β μ₀`, **β = 20**
  (ν_eff = 0.4754), recorded next to the physical bulk modulus estimate; declared volume-error budget and J/J̄
  monitoring (§6). ν = 0.499 with plain Neo-Hookean is refused.
- **D4 Time integration:** pure FLIP (`pic = 0`), MUSL, v6 grid-consistent positions, `dt = 0.5·dx/c_p,max` (measured
  critical ≈ 1.2–1.5), CUDA graph in chunks; no artificial damping in dynamics; equilibrium initializer only at CFL ≤ 0.25.
- **D5 Resolution:** 2 particles per cell per axis; ≥ 3 cells through the thickness for `rubber_ref_v1`/`rubber_rl_v1`
  (n_t = 2 only as `reduced_fidelity: coarse_thickness_resolution`); geometry aligned to the particle lattice.
- **D6 Boundary conditions:** nodal Dirichlet groups (clamp / kinematic grip, per-component mask, exact rigid motion,
  measured reaction wrench), traction groups, gravity, table plane and fixture boxes.
- **D7 Viscoelasticity (P2-3 design):** finite-strain Prony series on the isochoric PK2 (Simo/Holzapfel internal
  stress-like variables), exact exponential integrator, `[P·M]` sym-tensor state + previous isochoric stress, cleared or
  snapshot-restored by the masked reset.
- **D8 Contact (P2-4 design):** frozen projection/Coulomb helper with per-env, per-pair friction (table, pads, self);
  pads as kinematic boxes in grip frames; opt-in self-contact through the existing reference-exclusion path; 20 ms
  delayed-wrench assessment protocol (§11).
- **D9 Code organisation:** new modules only; a separate rubber registry; L0 CPU bitwise single↔batched parity.
- **D10 Scene/API:** `rubber_strip_v1` scene family and `RubberStripTaskFamily` behind `open_vector_env` (dispatch
  added in P2-6); material/friction/force/preload randomised per reset in-batch; geometry fixed per slot (rebuild).
- **D11 Cost rule:** every rubber row reports substeps/control step, n_t, β, dt and particles; reduced variants are
  labelled; no comparison with v7_2 rows hides the substep count.

## 4. D1 — backend

### 4.1 TLMPM can represent the strip

Total Lagrangian MPM with a reference stencil is an explicit Q1 finite-element method with particle quadrature and
lumped mass: there is no cell crossing and no numerical fracture, rotations of any size are handled because the
model is objective, and the frozen kernels already do exactly this. E3 reproduces the first bending frequency to
0.25 % and conserves energy; E4 reproduces the large-deflection elastica (α ≈ 9, tip drop 0.68 L) to 0.2–0.4 % with
2–3 cells through the thickness, with J̄ within ±0.5 %. Uniaxial stretch/release is a homogeneous deformation that the
same operator represents exactly (patch test RUBBER-PHYS-01). The only defects found are numerical choices of the
cutting profile (pic 0.05, no J-bar, 4 mm grid), which the rubber profile changes in its own versioned solver.

### 4.2 Grid resolution for a strip a few millimetres thick

E1: with J-bar and 2 particles per cell, bending error vs the converged value is 2–4 % at 2 cells through the
thickness and ≤ 0.5 % at ≥ 3. The 4 mm cutting grid is therefore unusable for a 2–8 mm strip (≤ 2 cells); the rubber
grid is `dx = t_min / n_t` with n_t ≥ 3 (t 3 mm → dx 1 mm; t 4.5 mm → 1.5 mm). Particles per env:
`8 L W t / dx³` (2 per cell per axis). Examples: 150×20×4 mm at dx 1 mm → 96k particles (≈ TRIM's 103k);
at dx 1.33 mm (n_t 3) → 40.5k.

### 4.3 Why not Newton VBD/FEM (option b)

1. **Dynamic fidelity is not established and the probe contradicts it.** VBD solves each step with a fixed number of
   Gauss–Seidel sweeps without a convergence test; the result depends on dt and the iteration count (E6). Rebound and
   vibration (the core of P2-1/P2-3) need converged dynamics; the specification forbids calling an animation-quality
   displacement solver complete material physics.
2. **No internal state:** tet elasticity is stateless; P2-3 Prony variables would have to be added inside Newton.
3. **Linear tets lock** near incompressibility (5-tet hex split); there is no F-bar/J-bar path.
4. **Integration cost:** none of the RCS batched layout, registry, masked reset, delayed-wrench coupling, Native
   (Isaac Sim 6.2 PhysX) hosting or vector-env contract applies; robot coupling would be the experimental proxy
   coupler (lagged) or one-way.
5. **Builder defects** to work around (`add_soft_grid` mass; no tetrahedraliser outside USD/TetWild).

What VBD offers — implicit steps far above the explicit CFL, world-filtered contacts, an existing MJWarp Franka + VBD
template — remains a possible **later, separate** opt-in benchmark backend if RL throughput of D1 proves
insufficient. That would need its own ADR with a converged-iteration study (iterations until f₁, static sag and energy
stop changing), and it could never share the `rubber_material_v1` calibration status without passing the same
RUBBER-PHYS tests.

### 4.4 Rejected inside option (a): running rubber through the v7_2 batch with the tool disarmed

It would keep ~31 operations per substep of which most are cutting/frontier work, S = 8 slots, `pic = 0.05` and no
Dirichlet condition, and it would tie rubber receipts to the v7_2 batch sources. A rubber constitutive model inside the
cutting solver ("cutting rubber") is a later, separate decision.

## 5. D2 — constitutive model `neo_hookean_jbar_v1`

Notation per particle p: `F`, `J = det F`, `C = FᵀF`, `I₁ = tr C`, `cof F = J F⁻ᵀ` (computed from cross products of the
columns of F, no division). Reference volume `V_p = s³` (uniform). Each particle lies in exactly one reference cell c
(fixed in TLMPM); `V_c = Σ_{p∈c} V_p`.

**Discrete energy** (SI, J):

```
Ψ = Σ_p V_p · μ/2 · (J_p^(−2/3) I₁,p − 3)  +  Σ_c V_c · U(J̄_c)
U(J) = κ/4 · (J² − 1 − 2 ln J)                      (Simo–Taylor / Ogden volumetric function)
J̄_c = Σ_{p∈c} V_p J_p / V_c                        (current/reference volume of the cell's quadrature)
```

**PK1 used by the frozen `_p2g`** (exact derivative of Ψ with respect to F_p, including the J̄ dependence):

```
P_p = μ J_p^(−2/3) (F_p − (I₁,p/3) F_p⁻ᵀ)  +  p(J̄_c) · cof F_p,      p(J) = U'(J) = κ/2 (J − 1/J)
    = a F + (p̄ − a I₁ /(3J)) cof F,   a = μ J^(−2/3)
```

Small-strain limit: Lamé `λ = κ − 2μ/3`, `μ`; `E = 9κμ/(3κ+μ)`, `ν = (3κ−2μ)/(2(3κ+μ))`. Cauchy stress for reports:
`σ = J⁻¹ P Fᵀ`. Verified: E5 finite differences (6e-10).

**Warp implementation (shared `@wp.func`s in a new `RCS/rubber_constitutive.py`, used by the single and the batched
kernels so that L0 parity is structural):**

1. *J-bar pass* (`_cell_jbar`, one thread per occupied cell): `J̄_c = Σ det(F_q) / n_c` over the cell's particles from a
   **constant per-env CSR** (`cell_offsets`, `cell_particles`, built at construction in particle order). No atomics:
   deterministic on CPU and GPU, unlike an atomic scatter. The cell id of a particle is its corner-0 node id
   (`nodes[p,0]`), so J̄ is stored in a node-sized array.
2. *Stress pass* (`_stress_nh_jbar`, one thread per particle): reads `F_p`, `J̄_{cell(p)}`, per-env μ, κ; writes
   `stress[p] = P_p` (float32, as every frozen kernel).

**Numerics for J → 0, J < 0, inverted particles.** Valid domain `J_p ∈ [0.2, 5]` and `J̄_c ∈ [0.2, 5]` (rubber in the
declared tests stays within `[0.9, 1.15]`). Outside it:
- the env gets the sticky invalid bit `RUBBER_J_DOMAIN` (new bit in the rubber material's `invalid_reason`), counted per
  substep (`j_domain_violations[N]`); strict mode raises, training mode terminates — the state is never "repaired";
- only to keep the same substep finite, the evaluation uses `J_eval = max(J_p, 0.05)` in `a` and `I₁/(3J)` and
  `J̄_eval = max(J̄, 0.05)` in `p̄`. `cof F` needs no guard and stays finite for J ≤ 0;
- an inverted particle (J ≤ 0) is the same violation. There is no SVD (unlike `tlmpm_v5._stress`), so no
  singular-value clamp hides inversion.

## 6. D3 — near-incompressibility

**Choice:** mean-dilatation J-bar (D2) **and** a reduced numerical bulk modulus `κ = β μ₀` with **β = 20**
(ν_eff = 0.4754), where μ₀ is the instantaneous shear modulus (μ₀ = μ for the elastic model).

**Why both.** J-bar removes volumetric locking at any β (E1: β 1000 converges like β 20). The reduced κ is needed for the
explicit time step: `c_p = sqrt((κ + 4μ₀/3)/ρ)`. For μ 0.5 MPa, ρ 1150 kg/m³ (c_s = 20.9 m/s):

| β | c_p (m/s) | dx/c_p at dx 1 mm | dt (CFL 0.5) | substeps per 20 ms |
|---|---|---|---|---|
| 5 | 52.5 | 19.1 µs | 9.5 µs | 2,100 |
| 10 | 70.2 | 14.2 µs | 7.1 µs | 2,800 |
| **20** | **96.3** | **10.4 µs** | **5.2 µs** | **3,850** |
| 50 | 149.4 | 6.7 µs | 3.3 µs | 6,000 |
| 3000 (real rubber, K ≈ 1.5 GPa) | 1,142 | 0.88 µs | 0.44 µs | 45,700 |

Real bulk moduli (1–2 GPa) are infeasible for explicit RL physics (45k substeps per control step); β = 20 costs a
factor ≈ 12 less than K_real and a factor 1.37 more than β = 10.

**Model-form error accepted and documented (not hidden):** relative to incompressible rubber, β = 20 gives
`J − 1 = +1.2 % / +2.5 % / +5.4 %` and nominal stress −2.1 % / −2.7 % / −4.2 % at uniaxial stretch 1.25 / 1.5 / 2.0
(E5), and bending stiffness −1.6 % (E_eff = 2.951 μ instead of 3 μ). The recipe stores `bulk_modulus_physical_pa`
(literature estimate, `UNMEASURED`) and `bulk_ratio_numerical = 20` separately; every receipt prints ν_eff. A β = 50 run
is part of the P2-2 refinement study as a model-form sensitivity, never as a hidden default change.

**Volume tracking and tolerances.**
- *Tracked:* per particle `J_p = det F_p` (from the integrated F), per cell `J̄_c`; per env and control step the
  segmented diagnostics report `J_min, J_max, J̄_min, J̄_max`, the volume ratio `Σ V_p J_p / Σ V_p`, and the fraction of
  cells with `|J̄_c − 1| > 0.05`. They are privileged (critic/receipt), never in the policy block.
- *Numerical tolerance (verification, pre-declared in RUBBER-PHYS-01/06):* homogeneous tests `|J̄ − J_analytic| ≤ 1e-5`
  (patch) and `≤ 5e-3` (quasi-static uniaxial, mean over the gauge section); global volume ratio drift ≤ 0.5 % in
  bending/vibration tests.
- *Physical tolerance (model-form, reported):* `|J − 1| ≤ 3 %` for stretches ≤ 1.5 (β = 20 analytic: 2.5 %);
  larger values are reported with the stretch that caused them.
- *Hard domain:* `[0.2, 5]` → invalid (§5).

**Refused:** a recipe with `poisson_ratio` near 0.5 and no J-bar; any rubber model without J̄ reporting; ν entered
directly (the recipe takes μ and β; ν is derived).

## 7. D4 — time integration, transfer and step rule

- **Transfer:** pure FLIP, `pic = 0` (E3); MUSL as frozen; positions by `tlmpm_v6._grid_positions` (x and F share the
  same grid field). The rubber solver refuses `pic ≠ 0`. Kinetic energy is reported on the grid (`½ Σ m_I v_I²`);
  particle-velocity null-space components do not enter positions or F.
- **No artificial damping in dynamics.** No mass/stiffness-proportional damping, no PIC blending. Losses come only from
  the declared viscoelastic model (P2-3), friction and the declared contact law (P2-4).
- **Step rule:** `dt ≤ CFL · dx / c_p,max` with **CFL = 0.5** (measured critical 1.2–1.5, E2; margin covers the stiffening
  of NH in compression and contact), c_p,max over the **declared randomisation range** of the batch (μ₀ max, ρ min),
  rounded down so that `control_dt / dt` is an integer. Receipts record CFL, c_p,max and the margin. The old 0.2 rule
  stays the cutting profiles' rule; rubber uses its measured rule, re-measured in P2-2 under large deformation and
  contact. **Amended (§18.2):** optional `deformation_reserve` r ≥ 1 divides dt; the opt-in
  `stable_step_guard_v1` flags `RUBBER_CFL_EXCEEDED` when the deformed-state bound dt·c_bound(F)/dx exceeds 1.0.
- **CUDA graph:** substeps per control step are thousands (§13), so the graph holds a chunk of `K_g` substeps
  (default 500) replayed `substeps/K_g` times; the device substep index carries across replays (`_advance`), so pose
  interpolation is unchanged. No host work between chunks.
- **Static-equilibrium initializer** (used by RUBBER-PHYS-02/05 and by preload snapshots): dynamic relaxation with
  either kinetic damping at **CFL ≤ 0.25** or mass-proportional damping at the solver CFL, stop on
  `max |v| < 1e-5 m/s` and BC-reaction/weight balance within 1e-3 (**amended, §18.5:** the velocity stop is unreachable;
  the stop rule is `static_stop_v3` — float64 grid-displacement markers, ≥ 3 T1, reaction balance). It is labelled `initializer`, never used inside an
  episode, and its result is the starting state, not a dynamic observable (E4 preserved failure).

## 8. D5 — resolution profiles

| Profile | ppc/axis | n_t (min over batch) | β | CFL | pic | Use | Label |
|---|---|---|---|---|---|---|---|
| `rubber_ref_v1` | 2 | ≥ 4 | 20 | 0.5 | 0 | refinement anchor, acceptance | full |
| `rubber_rl_v1` (default) | 2 | ≥ 3 | 20 | 0.5 | 0 | P2-6 vector env | full (≤ 0.5 % bending error vs converged, E1) |
| `rubber_coarse_v1` | 2 | 2 | 20 | 0.5 | 0 | fast curricula | `reduced_fidelity: coarse_thickness_resolution` |

Geometry is placed on the particle lattice (spacing s = dx/2; L, W, t multiples of s, minimum corner on a grid line), so
the body boundary coincides with particle-lattice planes; partially filled boundary cells (t not a multiple of dx) are
allowed and measured in the refinement study. dx, s, dt, substeps, ppc and β are uniform per batch (kernel scalars).

## 9. D6 — boundary conditions and loads (new, opt-in in the rubber solver only)

- **Nodal Dirichlet groups.** Each grid node may belong to one group `g` (per env `G` groups, uniform G; −1 free) with a
  3-bit component mask. Membership: nodes with mass support whose reference position lies in the group's region
  (clamp jaw volume, grip jaw volume, end face). Each group follows a rigid pose `T_g(t)` interpolated per substep by
  the existing pose interpolation (`[N·G, T]` poses). The prescribed node velocity is position-exact:
  `v_I = (T_g(i+1)·X̂_I − T_g(i)·X̂_I)/dt`, `X̂_I = T_g(0)⁻¹·X_I` (no drift of the rigid motion).
  Applied after **both** grid updates (P2G and MUSL), before G2P / deformation.
- **Reaction:** on the P2G pass, `j_I = m_I (v_I − ṽ_I)` on masked components (`ṽ_I` the unconstrained grid velocity);
  the group wrench `W_g = −Σ j_I / dt` with torque about the group origin (lever = current node position) is the measured
  grip/clamp force (tensile tests, robot wrench). Per-env accumulation; float atomics only into the env's own row.
- **Traction groups:** per-env force `f_g(t)` (N) distributed by mass fraction over the group's particles
  (`impulse_p += f_g dt m_p / M_g`), schedule- or action-driven (end-force tests).
- **Gravity** uniform per batch; **table plane and fixture boxes** per env (frozen contact helper, §11).

## 10. D7 — viscoelasticity (P2-3 design)

**Model `neo_hookean_jbar_prony_v1`.** Relaxation function on the isochoric part
`g(t) = g_∞ + Σ_{i=1..M} g_i e^{−t/τ_i}`, `g_∞ = 1 − Σ g_i > 0`, instantaneous shear modulus μ₀, equilibrium
`μ_∞ = g_∞ μ₀`. Instantaneous isochoric PK2 `S⁰(C) = μ₀ J^(−2/3) (I − (I₁/3) C⁻¹)` (zero in the reference state).
Internal stress-like variables (Simo 1987; Holzapfel 1996), symmetric, material frame:

```
S = g_∞ S⁰(C) + Σ_i Q_i          (isochoric PK2);   P = F S + p(J̄) cof F
Q_i(t) = g_i ∫₀ᵗ e^{−(t−s)/τ_i} dS⁰/ds ds
Q_i^{n+1} = a_i Q_i^n + b_i (S⁰_{n+1} − S⁰_n),   a_i = e^{−Δt/τ_i},   b_i = −g_i (τ_i/Δt) expm1(−Δt/τ_i)
```

The update is exact for S⁰ varying linearly over a substep and unconditionally stable; τ_i and Δt enter only through
`a_i, b_i`, so the relaxation time scale is dt-independent (τ → ∞ gives elasticity with μ₀, τ → 0 gives μ_∞). `a_i, b_i`
are computed per env in float64 on the host (expm1 avoids float32 cancellation for τ ≫ Δt) and stored float32 in the
params table. Volumetric viscosity: none (documented). The wave speed for D4 uses μ₀.

**State layout (registry):** `visco_q[P·M]` sym-6 (`xx,yy,zz,yz,xz,xy`, float32, particle-major `p·M + i`) and
`visco_s_prev[P]` sym-6. M is uniform per batch (≤ 4, unused terms g = 0). 24·(M+1) B per particle: 96 B at M = 3
(9.6 MB per 100k particles).

**Reset:** both arrays are registry state with initial value `zero` (stress-free reference start) or `snapshot`
(per-slot preload state, D10). The masked reset kernel writes them for masked envs only; the vars()-complete L1 test
(ADR 0021 §5) covers them, so an env never inherits material memory from its previous episode.

**Analytic references (P2-3 tests):** homogeneous step stretch at fixed C: `S_iso(t) = S⁰(C)·g(t)` exactly;
small-amplitude oscillation: `G'(ω) = μ₀[g_∞ + Σ g_i ω²τ_i²/(1+ω²τ_i²)]`, `G''(ω) = μ₀ Σ g_i ωτ_i/(1+ω²τ_i²)`, loss per
cycle `π γ₀² G''`; several loading rates (one-rate fits are not generalisation); fitted τ at dt and dt/2 equal within
0.5 %; energy balance `E_kin + Ψ + E_grav − W_ext ≤ 0` and equal to the integrated dissipation.

## 11. D8 — contact, friction and coupling (P2-4 design)

- **Law:** the frozen `_contact_impulse` (projection fraction per substep + Coulomb with stick cap), called from new
  rubber kernels with per-env parameters: `friction_table`, `friction_pad`, `friction_self`, `contact_stiffness_fraction`,
  `self_contact_stiffness_fraction`. The projection fraction is numerical; penetration and slip are part of the
  refinement study because the law's effective stiffness scales with 1/dt².
- **Table / fixtures:** per-env plane and box list (flat arrays + offsets, as the v7_2 batch).
- **Gripper pads / robot:** B boxes per grip group in the grip frame, moving with the interpolated grip pose (kinematic),
  reactions into `pad_reaction[N·G·B]` → per-env wrenches `[N, G, B, 6]` mapped to robot bodies (recipients as in
  `configure_pad_obstacles`). Robot links other than pads do not touch the strip in v1 (declared limitation).
  `grip_model = "kinematic_clamp_v1"` (Dirichlet group, no slip) or `"friction_pads_v1"` (pads only).
- **Self-contact:** opt-in `self_contact=True`; pairs closer than `2r` (r = 0.9·s/2) whose reference distance exceeds
  3·dx (the frozen exclusion) get the pair impulse; grouped HashGrid (one per process). Off by default (cost: a hash build
  per substep); on in fold scenes.
- **20 ms delayed wrench assessment.** (1) Ideal-tracking / clamped runs (all P2-1..P2-3 tests and every calibration
  experiment) have no feedback, so no coupling error enters any material parameter. (2) For robot-coupled runs:
  (a) A/B in the single-env reference with coupling intervals 20 ms vs 2 ms (material dt unchanged) on the same scripted
  hand trajectories over a declared envelope (hand speed, strip mass, first-mode frequency vs 50 Hz) — differences in
  tip trajectory, rebound time and decay rate must stay inside the P2-4 tolerance, else the envelope is restricted and
  rows carry `coupling_delay_significant`; (b) delay power error `Σ_k |(w_k − w_{k−1})·ξ_k| Δt` relative to the strip's
  mechanical energy; (c) the energy balance of the robot↔strip exchange. A coupling difference is never fitted as
  viscoelastic damping. (3) Per-env wrench isolation: CPU bitwise perturbation test; GPU instrumented row ownership.

## 12. D9 — code organisation, registry and parity

| New module | Content |
|---|---|
| `RCS/rubber_material.py` | `rubber_material_v1` recipe dataclasses, loader/validator, derived quantities (μ₀, κ, ν_eff, E_eff, c_s, c_p, dt_max), Prony coefficients |
| `RCS/rubber_calibration.py` | calibration-dataset manifest loader/validator (no fitting) |
| `RCS/rubber_reference.py` | analytic references (EB, Timoshenko, f₁, elastica BVP, NH uniaxial/shear, Prony relaxation/G′/G″) |
| `RCS/rubber_constitutive.py` | shared `@wp.func`s (PK1, energies, Prony update, sym-6 helpers) |
| `RCS/tlmpm_rubber_v1.py` | single-env reference solver (status `EXPERIMENTAL_NOT_ACCEPTED`; frozen at P2-2 acceptance) |
| `RCS/rubber_scene.py` | `rubber_strip_v1` scene compiler (particles, BC/traction/pad groups, markers, cell CSR) |
| `RCS/live_rubber_material.py` | single-env runtime wrapper (graph chunks, reset, advance(grip poses) → wrenches, records); Native host in P2-6 |
| `RCS/batched/constitutive_rubber.py` | rubber registry: model id/version, param columns, state variables (`identity`/`zero`/`snapshot`, dtype mat33/float/sym-6, per-particle multiplicity M), stress launcher (J-bar + stress), energy launcher, wave-speed function |
| `RCS/batched/rubber_layout.py`, `rubber_batch.py`, `rubber_material_batch.py` | pack from single solvers built on CPU; ENV kernels; graph, masked reset (incl. per-reset material resampling), segmented diagnostics, wrench |
| `RCS/rl/rubber_family.py`, `rl/rubber_env.py` | task family, `grip_delta_v1` actions, observations, reward, contract checker |
| `configs/materials/rubber_material_v1.schema.json`, `configs/materials/rubber_generic_soft_v1.json`, `configs/rubber/*` | schema, default recipe, calibration manifest, pre-declared tests, workload E definition |

- The rubber path **does not edit** `batched/constitutive.py` (its hash is in v7_2 receipts and its param table is
  v7_2-specific). The rubber registry follows the same protocol (model id, version, params, per-particle state, stress
  launch, receipt) so that both registries can be unified later under a separate decision.
- Frozen kernels launched unchanged: `tlmpm_v5._p2g, _grid_update, _g2p, _musl, _deformation, _contact_impulse,
  _box_sdf` and `tlmpm_v6._grid_positions`. S = 1 with constant `slot = 0`, `slot_code = 0` tables (no `_claim_slots`).
- **Parity (ADR 0021 §5 levels):** L0 CPU bitwise single↔batched (N = 1, N = 3 heterogeneous in geometry, μ, ρ, friction,
  copies), after every control step over every array; L1 isolation and reset (vars()-complete, including registry
  state and resampled parameters); L2 GPU invariants (mass constant between resets, finite, J-domain, env-ownership audits); L3 GPU
  outcome bands from ≥ 3 single-env GPU repeats. J̄ uses a deterministic gather, so the only GPU nondeterminism is the
  frozen P2G/MUSL atomics.
- `open_vector_env` gets an isinstance dispatch to the rubber env only in P2-6, after P1-5 has merged (that file is
  under active Phase 1 change); P2-1..P2-3 use `open_rubber_vector_env` directly. Workload E is defined in
  `configs/rubber/workload_e_v1.json`; `configs/rl_benchmark/workloads_v1.json` stays unchanged. New tests
  `tests/test_rubber_*.py` fall under the `tests/*` glob of `acceptance_v1.evaluator_files`; the P1-6 judge runs from
  its frozen evaluator checkout, so a running acceptance is unaffected, but the glob's content grows (orchestrator
  to note when the next acceptance version is cut).

## 13. D10/D11 — scene family, vector-env API (P2-6 design) and GPU cost

**Scene `rubber_strip_v1`.** Strip L × W × t in a local frame (x along the length, z through the thickness), on the
particle lattice. Default family ranges: L 80–240 mm, W 10–30 mm, t 3–8 mm (with dx from t_min and n_t ≥ 3). Boundary
cases: (A) cantilever, clamp jaw ≥ 2·dx at x ∈ [−ℓ_c, 0]; (B) one kinematic grip at one end, free end;
(C) two grips (stretch / release); (D) strip on the table held by pads (P2-4). Loads: gravity, grip motion, end traction.
Markers: K = 9 cross-section centroids at equally spaced reference x (incl. tip), per-env CSR of particle ids;
surface particles flagged at compile time for viewers and Native.

**`RubberStripTaskFamily`** (per slot): scene geometry and BC case, `material_recipe`, `resolution_profile`,
randomisation ranges (μ_∞, Prony scale, ρ, frictions, end force, preload offset), `env_seeds`, `max_episode_steps`,
goal generator. **Heterogeneity:** particle counts differ per slot (flat layout, no padding). Per reset, in-batch on the
device without allocation or recapture: material parameters (params row, particle masses for ρ, Prony `a_i, b_i`),
frictions, traction schedule, target, preload. Ranges are fixed at build because dt follows c_p,max. Geometry is fixed per
slot; a change needs `rebuild()` (`UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET`, as ADR 0021 §6). Initial deformation:
either a physically simulated preload phase of `settle_steps` (grip moves along a seeded scripted path; actions
ignored; excluded from reward) or a per-slot snapshot computed by the initializer (§7), restored by the masked reset
(registry initial value `snapshot`).

**API.** `open_vector_env(num_envs, runtime="rl_gpu_headless", task=RubberStripTaskFamily(...),
material="rubber_generic_soft_v1", observation_profile="state", backend="kinematic_grip_batched",
action_spec="grip_delta_v1", reward="rubber_tip_tracking_v1", device="cuda")`. Torch device tensors as in ADR 0021 §8.
- *Actions `grip_delta_v1` (A = 6):* normalised delta of the grip pose (≤ 5 mm and ≤ 0.05 rad per 20 ms step), workspace
  clamp, HOLD rows on rejection; the grip is the Dirichlet group (B) or pad frame.
- *Observation `policy` (deployable):* grip pose (7), commanded grip twist (6), time remaining (1), previous action (6),
  marker positions (3K) and marker velocities (3K) as **idealised perception** (`ideal_marker_tracking`, flagged in the
  receipt), goal (3). No force, wrench, stress, J, energy or material parameter.
- *Observation `critic` (privileged, opt-in):* material parameters, grip/clamp wrench, energies, J̄ statistics, contact
  statistics.
- *Reward `rubber_tip_tracking_v1` (`PROVISIONAL_SHAPING_NOT_SUCCESS`):* `−‖x_tip − x_goal‖/L − w_a‖a‖² − invalid_penalty`.
  Success comes from a host evaluator `rubber_tracking_eval_v1` at episode end (tip within 10 mm for the last 25 steps).
  `terminated` on invalid state (nonfinite, J domain, input domain); `truncated` at `max_episode_steps` (default 250).
- Full Native (P2-6): `LiveRubberMaterial` exposes the coupling interface of the cutting material (reset(measured
  pose), advance(pose, t) → wrench), so the Native runner can host the same recipe; surface rendering may differ,
  material physics may not.

**GPU cost (D11).** Lean rubber substep ≈ 17 launches (J-bar, stress, zero impulse, fixture/table, pad boxes ×B,
traction, P2G, grid update, BC, G2P, MUSL, grid update, BC, deformation, positions, advance; + hash build/query with
self-contact) vs ≈ 31 + B for v7_2. Per-particle memory ≈ 470 B (+96 B with M = 3 Prony terms, +≈170 B with snapshot
reset) vs ≈ 480 B for v7_2 TRIM; node memory with S = 1 is 1/8 of v7_2's per node. The decisive difference is the
**substep count**: rubber needs `20 ms / dt` ≈ 2,000–7,000 substeps per control step (µ₀ 0.3–1.3 MPa,
ρ 1000–1150 kg/m³, dx 1–1.5 mm, β 20) against 200 for v7_2 TRIM. Per-substep cost: §2 E7 (0.41 ms at 128k particles,
shared GPU, no contacts). Rows always report substeps/control step, particles, n_t, β and dt; a rubber env-step rate
is never compared with a v7_2 rate without them.

## 14. P2-0 deliverables (specifications for rcs-impl)

### 14.1 `rubber_material_v1` recipe (JSON, SI units)

```json
{
  "schema": "rcs_rubber_material_v1",
  "id": "rubber_generic_soft_v1",
  "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
  "material_identity": {"family": "unknown_elastomer", "candidates": ["NR", "EPDM", "silicone", "TPE"],
                        "evidence": "photograph only; composition unknown"},
  "density_kg_m3": {"value": 1150.0, "range": [900.0, 1400.0]},
  "model": "neo_hookean_jbar_v1",
  "elastic": {
    "shear_modulus_eq_pa": {"value": 4.0e5, "range": [1.0e5, 2.0e6]},
    "bulk_ratio_numerical": {"value": 20.0, "range": [10.0, 100.0]},
    "bulk_modulus_physical_pa": {"value": 2.0e9, "status": "LITERATURE_TYPICAL_UNMEASURED"}
  },
  "viscoelastic": {"applies_to": "isochoric", "max_terms": 4, "prony": []},
  "contact": {"friction_table": {"value": 0.8, "range": [0.3, 1.5]},
              "friction_pad": {"value": 1.0, "range": [0.3, 1.5]},
              "friction_self": {"value": 0.9, "range": [0.3, 1.5]}},
  "numerics": {"resolution_profile": "rubber_rl_v1", "particles_per_cell_axis": 2, "min_cells_through_thickness": 3,
               "cfl": 0.5, "pic": 0.0, "contact_stiffness_fraction": 0.1, "self_contact_stiffness_fraction": 0.05,
               "contact_radius_scale": 0.9, "j_domain": [0.2, 5.0], "j_physical_tolerance": 0.03},
  "derived": "computed by the loader, never input: mu0, kappa, nu_effective, young_effective_pa, c_s, c_p, dt_max(dx)",
  "provenance": {"defined": "ADR 0023", "values": "provisional literature-typical magnitudes, not measured"}
}
```

Validation rules: every quantity finite, positive and inside its range; `Σ g_i < 1`, `g_i ≥ 0`, `τ_i > 0`, at most 4 terms;
`model` must match the presence of Prony terms (`neo_hookean_jbar_prony_v1` iff `prony` non-empty); unknown keys are
refused; `poisson_ratio` is refused as an input key; `calibration_status ∈ {SIMULATOR_ONLY_UNCALIBRATED,
PILOT_FIT_NOT_HELDOUT, REAL_CALIBRATED}` and `REAL_CALIBRATED` requires a manifest id whose held-out split passed
(P2-5); `pic` must be 0. A P2-3 provisional recipe `rubber_generic_visco_v1` adds e.g.
`prony: [{"g": 0.15, "tau_s": 0.01}, {"g": 0.10, "tau_s": 0.1}]` with the same status.

### 14.2 Calibration-dataset manifest (`rcs_rubber_calibration_dataset_v1`, JSON + CSV)

```json
{
  "schema": "rcs_rubber_calibration_dataset_v1",
  "dataset_id": "none_v1",
  "status": "NO_REAL_DATA",
  "units": "SI",
  "specimens": [{"specimen_id": "S1", "material_label": "", "supplier": "", "batch": "",
                 "dimensions_m": {"length": 0.0, "width": 0.0, "thickness": 0.0}, "dimension_uncertainty_m": 0.0,
                 "mass_kg": 0.0, "mass_uncertainty_kg": 0.0, "temperature_c": 0.0,
                 "preconditioning": {"mullins_cycles": 0, "max_stretch": 0.0}}],
  "experiments": [
    {"experiment_id": "T1", "type": "tensile_load_unload", "specimen_id": "S1", "gauge_length_m": 0.0,
     "grip": {"type": "clamped", "clamp_length_m": 0.0}, "crosshead_speed_m_s": 0.0, "max_stretch": 0.0, "cycles": 0,
     "data_file": "T1.csv", "columns": {"time_s": "s", "displacement_m": "m", "force_n": "N"},
     "sensor": {"load_cell_range_n": 0.0, "resolution_n": 0.0, "sample_rate_hz": 0.0}, "split": "fit"},
    {"experiment_id": "R1", "type": "stress_relaxation", "hold_stretch": 0.0, "hold_time_s": 0.0, "data_file": "R1.csv"},
    {"experiment_id": "C1", "type": "cantilever_rebound_video",
     "clamp": {"free_length_m": 0.0, "clamp_length_m": 0.0, "orientation": "horizontal"},
     "release": {"method": "tip_displacement_release", "tip_displacement_m": 0.0},
     "video_file": "C1.mp4", "fps": 0.0, "camera_calibration_file": "", "markers": [{"id": 0, "s_m": 0.0}],
     "tracked_file": "C1_tracks.csv", "columns": {"time_s": "s", "marker_id": "-", "x_m": "m", "z_m": "m"}},
    {"experiment_id": "G1", "type": "gravity_sag_static", "free_length_m": 0.0, "tip_drop_m": 0.0},
    {"experiment_id": "F1", "type": "friction_sliding", "counterface": "steel|wood|pad_material", "normal_load_n": 0.0,
     "sliding_speed_m_s": 0.0, "data_file": "F1.csv",
     "columns": {"time_s": "s", "tangential_force_n": "N", "normal_force_n": "N", "displacement_m": "m"}},
    {"experiment_id": "I1", "type": "incline_slip_onset", "counterface": "", "slip_angle_deg": 0.0}
  ],
  "splits": {"fit": [], "heldout": [], "frozen_at": null, "sha256": null},
  "boundary_conditions_note": "every experiment states its grips/clamps, contact counterfaces and temperature"
}
```

Validator: units, column headers and monotone time of every CSV; each experiment's boundary conditions present; splits
disjoint; `frozen_at` + SHA-256 over manifest and files written before any held-out evaluation (P2-5); status words
`NO_REAL_DATA`, `PILOT`, `FROZEN_FOR_HELDOUT`. The block above shows the field layout (placeholder values); one
experiment entry per loading rate (tensile and relaxation at ≥ 3 rates; the specification's "different rates"). The
manifest shipped now is `none_v1` with `NO_REAL_DATA` and empty `specimens` / `experiments`.

### 14.3 Pre-declared tests (P2-1/P2-2; `configs/rubber/rubber_phys_tests_v1.json`, frozen at commit, a change is v2)

Every test runs in the single-env reference **and** batched (N = 1 and N ≥ 4 heterogeneous); CPU where small, GPU
otherwise. β = 20, pic = 0, CFL 0.5, elastic model unless stated. Thresholds are set from the E1–E5 probes with margin.

| Id | Test | Reference | Pass criterion |
|---|---|---|---|
| RUBBER-PHYS-01 | Patch: homogeneous F imposed by Dirichlet on all nodes of a 6³-cell block (uniaxial λ ∈ {0.7, 1.25, 1.5, 2, 3}, simple shear γ ∈ {0.3, 1}, 90° rotation × stretch), one substep, no gravity | §5 PK1 in float64; objectivity | per particle `‖P−P_ref‖ ≤ 1e-5‖P_ref‖ + 1e-3 Pa`; `|J̄ − J| ≤ 1e-6`; interior-node internal force ≤ 1e-5 of boundary force; `P(RF) = R P(F)` within float32 tolerance |
| RUBBER-PHYS-02 | Small-deflection cantilever sag: clamp all components at x ≤ 0, gravity, L/t ∈ {10, 20}, t ∈ {3, 4, 6} mm, W = t and 4t, n_t ∈ {3, 4}; static state by the §7 initializer | Euler–Bernoulli with E_eff (W ≤ 2t); bracket [EB(1−ν²), EB] (W > 2t); refinement-extrapolated value | narrow: `|δ/δ_EB − 1| ≤ 4 %`; all: `|δ/δ_extrap − 1| ≤ 2 %` at n_t ≥ 3; BC reaction = weight within 0.1 % |
| RUBBER-PHYS-03 | Geometry scaling: L ∈ {60, 90, 120} mm (t 4), t ∈ {3, 4, 6} mm (L 100), small deflection | `δ ∝ L⁴ t⁻²` | fitted exponents `4 ± 0.15` and `−2 ± 0.1` |
| RUBBER-PHYS-04 | Free vibration: straight strip released under gravity, ≥ 5 periods, no damping | EB `f₁ = (1.8751²/2π)√(E_eff I/(ρAL⁴))` | `|f₁/f_EB − 1| ≤ 3 %`; total energy drift ≤ 0.5 % of peak elastic energy; amplitude over the last whole period ≥ 0.95 of the first; f₁ at dt and dt/2 within 0.2 % |
| RUBBER-PHYS-05 | Large-deflection sag: L/t = 20, α ∈ {3, 9, 20}, §7 initializer | inextensible elastica (BVP) | tip drop within 3 % of elastica; J̄ ∈ [0.97, 1.03] |
| RUBBER-PHYS-06 | Uniaxial quasi-static load–unload: strip 60×10×4 mm, axial-only Dirichlet at both end faces (lateral free, rigid modes pinned), λ 1 → 2 → 1, ramp ≥ 50 axial periods, no gravity | compressible-model `P₁₁(λ)`, `J(λ)`, lateral stretch (E5) | nominal stress from the grip reaction within 1 % at λ ∈ {1.25, 1.5, 2}; mean gauge J̄ within 5e-3; unload = load within 1 %; residual length ≤ 0.1 % |
| RUBBER-PHYS-07 | Stretch and release: two grips to λ 1.5, hold, release grip 2 (group removed) | elastic recovery; energy | mean length after release within 0.5 % of L₀; energy drift ≤ 1 % over 10 axial periods; no permanent set |
| RUBBER-PHYS-08 | Simple shear (homogeneous, J ≡ 1) γ ∈ {0.5, 1, 2} | `σ₁₂ = μγ`, `σ₁₁ − σ₂₂ = μγ²` | within 0.5 % |
| RUBBER-PHYS-09 | Observables: surface particles, 9 markers, J/J̄ stats, energies, BC wrench at every control step for -02/-04/-07 at t ∈ {3, 4, 6} mm, W ∈ {t, 4t} | schema `rcs_rubber_record_v1` | complete and schema-valid |
| RUBBER-PHYS-10 | Single ↔ batched | ADR 0021 L0–L3 | CPU bitwise every array every control step (N 1, N 3 heterogeneous, copies); L1 isolation and partial reset; GPU per-env observables inside the band of 3 single-env GPU repeats (band floor 1e-5 relative) |
| RUBBER-PHYS-11 | Refinement (§14.4) | GCI | §14.4 |

P2-3 (`RUBBER-VISC-01..05`: exact relaxation, three loading rates, hysteresis loss vs `πγ₀²G''`, dt-independence of
fitted τ, reset clears internal variables) and P2-4 (`RUBBER-CONT-01..06`: incline slip onset `tan θ = μ`, sliding
acceleration `g(sinθ − μ cosθ)`, penetration ≤ 0.25·s (particle spacing), pad hold vs friction limit, fold self-contact, coupling-delay A/B,
per-env wrench isolation) are declared in their own config versions before their implementation.

### 14.4 dt/spacing refinement study (P2-2)

- Grids n_t ∈ {2, 3, 4, 6} at 2 particles per cell (plus 3 per cell at n_t 3 as a quadrature check); dt at CFL 0.5 and
  0.25 on each grid; model-form sensitivity β ∈ {20, 50} (reported, not a refinement).
- Observables: Q1 static sag (-02), Q2 f₁ (-04), Q3 nominal stress at λ 1.5 (-06), Q4 large-deflection tip (-05),
  Q5 J̄ extremes (-06/-07), Q6 penetration and slip distance (P2-4).
- Method: observed order from the three finest grids, Richardson extrapolation, grid convergence index with
  safety factor 1.25 (Roache).
- Pass: GCI ≤ 2 % for Q1–Q4 at the production profile (`rubber_rl_v1`); dt change CFL 0.5 → 0.25 alters Q1–Q4 by
  ≤ 0.1 %; non-monotone convergence is reported with its range, never smoothed.
- Runs: single env (CPU small, GPU large) plus one batched N = 4 set equal to the single runs (L0 on CPU, L3 on GPU).

## 15. Implementation task list (rcs-impl)

Sizes: S ≈ ½ day, M ≈ 1 day, L ≈ 2–3 days. CPU tests: `PYTHONPATH=src .venv/bin/python -m pytest -p no:cacheprovider
tests/...`; GPU probes under `flock …/gpu<N>.lock`, one per GPU, ≥ 4 GB left free, new `artifacts/rl_ready/p2_*/`
directories.

| Task | Deliverable | Tests (CPU unless stated) | GPU | Size |
|---|---|---|---|---|
| **P2-0** | | | | |
| R0.1 | `rubber_material.py`, schema, `rubber_generic_soft_v1.json` | `test_rubber_material.py`: schema, ranges, refusals (ν input, pic ≠ 0, Σg ≥ 1, unknown keys), derived values vs hand calculation, dt rule | – | M |
| R0.2 | `rubber_calibration.py`, manifest schema, `none_v1` manifest | `test_rubber_calibration_manifest.py`: headers/units/monotone time, split disjointness, freeze hash | – | S |
| R0.3 | `rubber_reference.py` | `test_rubber_reference.py`: EB/f₁ closed forms, elastica α → 0 limit `α/8`, NH uniaxial vs E5 table, simple shear, Prony G′/G″ limits | – | M |
| R0.4 | `configs/rubber/rubber_phys_tests_v1.json` (§14.3), `workload_e_v1.json` | SHA pin test (frozen once committed) | – | S |
| **P2-1** | | | | |
| R1.1 | `rubber_constitutive.py` | PK1 vs float64 FD; objectivity; J-domain flags; cof for J ≤ 0 finite | – | M |
| R1.2 | `tlmpm_rubber_v1.py` (stencil via frozen formula, cell CSR, S = 1 tables, frozen grid kernels, J-bar, stress, BC groups + reaction, traction, table/fixture contact, pic = 0, diagnostics, receipt) | patch test RUBBER-PHYS-01; static BC reaction = weight; 2k-step energy conservation; refusals; frozen-source hashes unchanged (`test_batched_frozen_sources.py` still passes) | – | L |
| R1.3 | `rubber_scene.py` + `rcs_rubber_scene_v1` schema | particle counts, lattice alignment, BC node sets, marker CSR, cell CSR | – | M |
| R1.4 | `live_rubber_material.py` (graph chunks, reset, advance → wrench, records `rcs_rubber_record_v1`) | CPU path without graph; chunk/index continuity | smoke N = 1, 100k particles, ≤ 1 GB, ≤ 2 min | M |
| R1.5 | static initializer (§7) | converges on a small cantilever; refuses CFL > 0.25 with kinetic damping | – | S |
| R1.6 | `scripts/rubber/run_rubber_phys.py` + evidence writer | -01..-03 small on CPU | -04..-09 single env, ≤ 2 GB, ≤ 30 min total | M |
| R1.7 | `batched/constitutive_rubber.py`, `rubber_layout.py`, `rubber_batch.py`, `rubber_material_batch.py` | L0 (N 1, N 3 heterogeneous, copies), L1 (perturbation, partial reset vars()-complete, resampled params), audits | L2 smoke N 1/4/8 (≤ 1.5 GB at 100k particles/env, ≤ 10 min) | L+L |
| **P2-2** | | | | |
| R2.1 | -05/-06/-07/-08 full runs + evidence | – | single + N 4, ≤ 2 GB, ≤ 1 h | M |
| R2.2 | refinement runner + GCI report (§14.4) | GCI arithmetic unit test | ≈ 40 runs, ≤ 3 GB, ≤ 3 h, queued behind Phase 1 | M |
| R2.3 | β-sensitivity / model-form report (β 20 vs 50 vs incompressible analytic) | – | in R2.2 | S |
| R2.4 | profiling (`profiling.py` stage split) and N = 1/4/8/16 throughput + VRAM vs the v7_2 rows | – | ≤ 4 GB, ≤ 30 min | M |
| R2.5 | xhigh review → acceptance; freeze `tlmpm_rubber_v1` | – | – | – |

## 16. Rejected alternatives

| Alternative | Reason |
|---|---|
| Plain compressible NH with ν ≈ 0.499 | E1 (ν 0.4995): tip deflection 0.52 of Euler–Bernoulli (0.53 of the converged J-bar value) at n_t 4 — locking; ν 0.499 means β ≈ 500, so dt 4.8× smaller than β 20 |
| Real bulk modulus, explicit | 45k substeps per 20 ms at dx 1 mm |
| Mixed u–p / implicit pressure now | removes κ from the CFL but adds a global solve per substep per env; revisit only if β = 20 model-form error is unacceptable for real data |
| Mass scaling | changes inertia; allowed later only as `reduced_fidelity: inertia_scaled` for quasi-static tasks |
| Newton VBD/FEM backend now | §4.3 |
| Rod/shell reduced model | cheaper for long strips but a different, reduced-dimensional physics; possible later as its own labelled backend |
| PIC 0.05 (cutting default) | dt-dependent numerical damping (E3) |
| Atomic J-bar scatter | GPU nondeterminism for no benefit; the cell CSR gather is exact and deterministic |
| Editing `batched/constitutive.py` / `workloads_v1.json` | changes Phase 1 receipts and the acceptance file set |

## 17. Risks

| # | Risk | Detection / mitigation |
|---|---|---|
| 1 | Throughput: thousands of substeps per control step | measured rows (R2.4); profiles `rubber_coarse_v1`, narrower stiffness ranges; later opt-in accelerations each labelled |
| 2 | Q1/P0 pressure checkerboard under strong compression | J̄ field statistics, refinement Q5; β moderate |
| 3 | Initializer instability (E4) | CFL ≤ 0.25 for kinetic damping, refusal test |
| 4 | Contact law stiffness ∝ 1/dt² | penetration/slip in the refinement study |
| 5 | Partially filled boundary cells | refinement over non-multiple thicknesses |
| 6 | One device HashGrid per process | self-contact rubber batches never share a process with a cutting batch |
| 7 | GPU atomics nondeterminism | L3 bands; deterministic J̄ gather |
| 8 | Calibration over-claim | status field validator; REAL_CALIBRATED only after frozen held-out pass |

## 18. P2-1 follow-up decisions (amendment, 2026-11-10, rcs-xhigh)

Scope: the RUBBER-PHYS-07 FAIL of the P2-1 formal run, the initializer stop rule of protocol v2, and the reported
test-order dependence of a Phase 1 test. Nothing frozen changed (v1–v7_2, law 8, evaluators, Phase 1 batched modules,
`rubber_phys_tests_v1/v2.json`, the P2-1 records); every failure stays preserved. Scripts:
`scripts/rubber/p2_1_followup/`; raw outputs: `artifacts/rl_ready/p2_1/followup_*`, `pollution_diag_001`, `_002`,
`rubber_phys_v3_001`, `gpu_smoke_001`; compact evidence: `docs/evidence/rl_ready/p2_1_followup_001.json`.
Material status unchanged: `SIMULATOR_ONLY_UNCALIBRATED`.

### 18.1 Diagnosis of RUBBER-PHYS-07 (stretch and release, λ 1.5, v2)

| # | Finding | Evidence |
|---|---|---|
| F1 | **The discrete operator is stable and energy-consistent in compression; its stable step scales with the reference-frame wave speed.** Pure FLIP + MUSL with fixed TLMPM weights is algebraically symplectic Euler for (u, g), g the MUSL grid velocity, with inverse mass B = M_l⁻¹M_cM_l⁻¹ (M_c = W M_p Wᵀ); the particle kinetic energy equals ½gᵀM_lM_c⁻¹M_l g. On a periodic 4³-cell block (exact kernels' stencil, J-bar, finite-difference stiffness) the critical CFL is 1.79 at F = I (lumped FEM: 1.00) and 1.79/r(F) when deformed: r = 1.11 / 1.43 / 2.02 / 2.54 / 3.43 at uniaxial stretch 0.9 / 0.7 / 0.5 / 0.4 / 0.3, 1.45 at 2.0, 2.36 at shear γ 2. Zero negative stiffness modes in 35 states (uniaxial 0.3–3, uniaxial strain, isotropic J 0.5–1.25, shear, 12 random); K symmetric to 1e-8. In TLMPM the reference grid does not shrink with the material, so the speed that matters grows like 1/λ in compression on top of the NH/volumetric tangent. | `followup_stability_spectrum_n4_001` |
| F2 | **The release wave itself is resolved and conserves energy.** From a checkpoint at the v2 release: the strip contracts to 0.671 L₀ at 3.8 ms (1-D shock theory for the reflected compression: 0.668), minimum principal stretch 0.40 near the clamp, J ∈ [0.81, 1.15], J̄ ≥ 0.95, particle speeds 21–35 m/s; total energy within 0.54 / 0.32 / 0.17 % of Ψ at CFL 0.5 / 0.25 / 0.125 over 0–10 ms (O(dt) fluctuation, no drift). TLMPM forces depend on F only, so the transfer is speed-independent; FLIP/grid transfer at high speed is not implicated. | `followup_07_release_cfl{050,025,0125}_001`, `followup_release_physics_001` |
| F3 | **The declared scene buckles and folds — physically.** The post-release compression (≈ 0.33) is ≈ 360× the Euler strain of this 60 × 10 × 4 mm strip (9.1e-4); beam theory gives a dynamic-buckling e-folding time of 0.22 ms at that compression. Lateral marker excursion: 5e-9 m at the release (round-off seed), 8e-5 m at 5 ms, 1 mm at 11.0 ms, 9–15 mm at 14–20 ms; the hinge forms at the clamped end. | same, analytic §18 script `release_physics.py` |
| F4 | **The fold, not the 1-D wave, exceeds the stable step.** In the hinge the minimum principal stretch reaches 0.05–0.06 and the guard bound (§18.2) 13 / 5.0 / 3.3 at CFL 0.5 / 0.25 / 0.125: non-finite at 14.0 ms at CFL 0.5; finite at 0.25 / 0.125 but particle J down to 0.06 / 0.23 while J̄ stays ≥ 0.77 / 0.84. During the 1-D wave the bound peaked at 1.30 (CFL 0.5) without growth. The implementer's hypothesis (compressive stiffening consumes the CFL margin) is right as the mechanism of the blow-up, but the stiffening comes from the fold. | same |
| F5 | **J-bar:** no instability (F1). It does let particle J leave the domain in the fold through intra-cell dilatation patterns that keep J̄ (only μ resists them): the J-domain flag fires on particle J (0.06) while J̄ ≥ 0.77. | same |
| F6 | **Energy loss (≈ 34 % over 75 ms at CFL 0.25) is numerical: the Dirichlet projection at the clamped face, amplified by the fold whose hinge sits there.** Without any Dirichlet group (free-free release) the energy shows no secular change: ±0.25 % for 20 ms at λ 1.5 while buckling, and a bounded ±0.55 % symplectic fluctuation with a mean of +0.04 % over 75 ms at λ 1.03 (CFL 0.5). With the clamp the same λ 1.03 release loses energy secularly, about 0.1 % of Ψ per axial period (mean −0.72 % at 60–75 ms at CFL 0.5, −0.49 % at CFL 0.25), and at λ 1.5 the loss accelerates once the fold forms at the clamp (−4.0 % / −2.1 % at 20 ms for CFL 0.25 / 0.125), with energy moving into grid-scale modes (KE_particle − KE_grid 0.01 J → 0.03–0.07 J). Mechanism: zeroing the constrained grid velocity after P2G and feeding the change back through the FLIP increment acts as an inelastic impact on the grid-scale part of the motion at the wall (the effective MUSL mass M_lM_c⁻¹M_l couples constrained and free nodes). Not physical, not the J-bar, not high-speed transfer. (Correction: an earlier per-substep "lumped KE removed by the BC" measure read 0 because CPU `wp.array.numpy()` is a zero-copy view; with copies it counts ½mṽ² that is mostly re-supplied every substep, so it is not an energy measure and is not used.) | `followup_07_release_*_002`, `followup_07_release_l103_*_002` |
| F7 | **The v2 criteria were not satisfiable by the exact solution.** (a) The time-averaged length of an undamped large-amplitude NH oscillation is not L₀ (P ≈ 3μ(ε − ε²) gives ⟨ε⟩ ≈ ⟨ε²⟩ ≈ 0.1 at λ 1.5), and the strip folds. (b) The v2 denominator E_total(release) = KE + Ψ − W_ext is ≈ 7.6e-6 J, a numerical residual (0 in exact arithmetic): "1 %" meant 3e-7 of the stored energy. | `rubber_phys.evaluate_07`, `diag_07_cfl_factor1` |

**Root cause.** The protocol: an instantaneous release from λ 1.5 drives the declared slender strip into a physical
dynamic buckling and fold. A fold is outside v1 (no self-contact, extreme local compression); at CFL 0.5 the
reference-frame stiffening in the fold exceeds the stable step. The solver behaves as designed inside its domain (F1,
F2); what v1 lacked was a guard that reports leaving the stable-step domain before the state becomes non-finite.

### 18.2 Decision (a) — deformation-aware stable step (solver option, versioned here)

- The step stays **uniform and fixed per batch** (CUDA graph chunks, batching), §7, with two additions:
  1. **`deformation_reserve` r ≥ 1** in the step rule: `dt ≤ CFL · dx / (c_p,max · r)` (r = 1 is the v1 rule and the
     default; a scene declared for strong compression or folds sets r, e.g. 2 for P2-4 fold scenes).
  2. **`stable_step_guard_v1`** (opt-in solver option, single and batched): every substep, every particle,
     `CFL_eff = dt · c_bound(F, J̄) / dx` with
     `ρ c_bound² = a + (p'(J̄) + 5aI₁/(9J²)) g² + (2a/(3J)) (s_max g − J)`, `g = J/s_min = √λ_max(cof Fᵀ cof F)`,
     `s_max = √λ_max(FᵀF)`, `a = μJ^(−2/3)`, `p'(J) = κ/2 (1 + J⁻²)` — Weyl's bound on the largest eigenvalue of the
     reference-frame acoustic tensor `Q(N) = aI + (p' + 5aI₁/(9J²)) g gᵀ − (2a/(3J))(f gᵀ + g fᵀ)` (f = FN, g = cof F N,
     f·g = J). It is ≥ the exact acoustic maximum (299 random F, stretches 0.3–3: modulus ratio 1.0005–1.11) and ≥ the
     exact discrete-operator ratio r(F) of F1 in all 35 states (speed ratio 1.009–1.28; 1.031 at F = I)
     (`followup_bound_check_001`). `CFL_eff > limit` (default **1.0**) counts a
     violation; the env gets the sticky bit **`RUBBER_CFL_EXCEEDED`** (4); strict mode raises, training terminates;
     the state is never touched (bitwise-identical state with the guard on), no host sync, graph-safe, per-env
     counters cleared by the masked reset; diagnostics column `cfl_eff_max` (end of control step).
- **Why limit 1.0:** measured critical CFL at F ≈ I is 1.2–1.5 with free surfaces (E2) and 1.79 without (F1); the
  bound over-estimates the speed by 1–28 %; the λ 1.5 wave survived a transient bound of 1.30, and the blow-up came
  only after the fold drove it to 2.5–13. At CFL 0.5 an undeformed strip sits at 0.515.
- **Status:** off by default in `tlmpm_rubber_v1` (the v1 launch sequence, so every v2 result stands); **on in every
  v3 run; mandatory for P2-2 runs, the `rubber_rl_v1` profile and the P2-6 environments.** Cost: one particle pass per
  substep (one F read, ~150 float32 flops, an atomic only on a violation); to be profiled in R2.4 (fusing into the
  stress pass is allowed if bitwise equal).
- **Rejected:** a fixed dt from the maximum stiffening over the J domain (J alone does not bound c: uniaxial λ 0.3 at
  J ≈ 0.95 already gives r = 3.4; covering the domain means r ≈ 4–19, i.e. 4–19× the substeps); a per-control-step
  adaptive dt (host decision, breaks uniform-dt batching and graph reuse); per-env dt (breaks the uniform batch);
  any silent clamp.

### 18.3 Decision (b) — no self-contact in v1; RUBBER-PHYS-07 v3

- **v1 gets no self-contact.** A fold is a P2-4 scene: opt-in self-contact through the D8 path (the particle-contact
  path with all cut codes 0 is a pure self-contact behind the reference-distance exclusion), `friction_self`, the
  guard and r ≥ 2. The λ 1.5 instantaneous release is the natural RUBBER-CONT-05 (fold) candidate there; until then
  any fold must end in a flag (guard / J domain), which RUBBER-PHYS-12 checks.
- **`rubber_phys_tests_v3.json`** (declared and SHA-pinned at commit `849e10b`, before any v3 formal run; v1/v2
  unchanged). RUBBER-PHYS-07 v3, strip and grips as -06, 0.1 ms control intervals, guard on:
  - **07a `ramped`:** cosine stretch to λ 1.5 (60 T_ax), hold 10 T_ax, then grip 1 is replaced in the same control
    step by a traction on the end particle layer carrying the measured hold force, cosine-ramped to zero over
    60 T_ax; 10 fixed-free axial periods after. Large-strain elastic recovery and energy accounting with external work.
  - **07b `instant`:** as v2 but from λ 1.03: compression wave resolved (dispersive-bore width ≈ 8 mm = 8 dx),
    mean-length bias ≈ 4.5e-4, dynamic buckling over 10 periods below ≈ 8 e-folds from round-off (analysis, not a
    trial run).
  - Criteria per variant: |mean length / L₀ − 1| ≤ 0.005 (v2 threshold), energy drift ≤ 0.01 **of Ψ(release)**
    (the v2 threshold with the ill-posed denominator of F7(b) corrected), no invalid flag, lateral marker excursion
    ≤ 0.1 t (stays in the v1 domain).
- **RUBBER-PHYS-12** (software regression of the guard, designed after the diagnosis and labelled so): the exact v2
  -07 path at CFL 0.5 with the guard on must flag (CFL or J) at a control step strictly before the first
  NONFINITE_STATE step.
- The v2 RUBBER-PHYS-07 FAIL (`rubber_phys_001`) stays the v2 result.

### 18.4 (c) — energy loss

Explained in F6: numerical dissipation of the Dirichlet projection at the clamped face (no secular loss without a
Dirichlet group), small for smooth motion (RUBBER-PHYS-04: ≤ 5.8e-4 of peak Ψ) but about 0.1 % of Ψ per axial period
for a step-released axial wave hitting the clamp, and much larger once a fold forms at the clamp; dt-dependent (not
proportional). Never attributed to the material. P2-2 task F8 (§18.8): an energy-consistent Dirichlet constraint.

### 18.5 Initializer stop rule (amends §7)

- The §7 / v1 stop `max |v| < 1e-5 m/s` is unreachable (confirmed). Correction of the P2-1 mechanism statement: there
  is no grid-invisible null space under pure FLIP (v_p = Wᵀa stays in the range of Wᵀ); for grid-scale modes the
  particle velocity a exceeds the grid image M_l⁻¹M_c a, and float32 force noise keeps both at a floor.
- **static_stop_v2 is not sound as implemented.** Its reaction test (b) is sound; its marker test (a) is defeated by
  float32 position rounding: x = x_start + dt·v drops the increment once dt|v| < ulp(x)/2 (≈ 2e-5 m/s for z ≈ 3 mm at
  dt 5.8 µs). All 33 formal v2 initializers stopped with the last five marker changes exactly 0.0, at 0.9–1.45 T1.
  Re-run of -02 Lt10_t3mm_Wt1_nt3: δ/δ_EB 0.98399 at the v2 stop, 0.98546 at 2.45 T1 (float64 displacement),
  Ψ +0.36 % after the stop.
- **Decision: `static_stop_v3`** (§7 stop replaced): float64 grid displacement u_I += dt v_I after every substep
  (initializer only), markers x64 = x(start) + Σ w u; stop when t ≥ 3 T1 (all modes decayed by e^(−6π) under
  c = 2ω₁), the float64 markers move ≤ 1e-4 of their displacement on 3 consecutive checks, and |R_z + Mg|/(Mg) < 1e-3.
  The static deflection is read from the float64 markers (float32 value reported). Limits: the float64 markers carry
  a float32-force noise floor of ~1e-9 m per check (so δ must exceed ~1e-5 m; every formal δ is ≥ 1.7e-4 m), and the
  float32 static state itself keeps wandering inside a ~6e-4 band after convergence (δ/δ_EB 0.98546–0.98603 from 2.45
  to 11.5 T1 in the re-run) and creeps by ~5e-4 of δ per T1 in Lt20_t6mm_Wt4_nt3 (§18.9); both are reported, P2-2
  (R2.2, F10) quantifies them.
- v3 results: see §18.7.

### 18.6 Reported test-order dependence (Phase 1 test)

- **Not rubber state.** `tests/test_batched_particle_contacts.py::test_fallback_grid_reproduces_warp_grouped_query_sequence`
  fails **alone** whenever a GPU is visible (3/3, five repeats, both worktrees), passes with `CUDA_VISIBLE_DEVICES=''`
  (alone and after every rubber test) and passes after `wp.set_device("cpu")`. `FallbackGrid.build` launches without a
  device, i.e. on Warp's default `cuda:0`; Warp 1.17's RELAXED launch-array access lets that kernel write the test's
  pageable CPU arrays (verified), asynchronously, while `radix_sort_pairs` (dispatched on the arrays' CPU device) reads
  them on the host inside the same `build`: a race whose outcome depends on GPU contention (which explains the earlier
  "passes alone"). A `wp.synchronize()` after the build does not help; pinning the build to the CPU makes all three
  parameters pass (`pollution_diag_002`). Rubber code changes no Warp global.
- **Fix belongs to Phase 1** (the test and `batched/particle_contacts.py` are Phase 1 files; the module is SHA-frozen):
  pin the device in the test (`with wp.ScopedDevice("cpu"):` around the build — the device of every array it already
  creates; strengthens, does not weaken) or, in a versioned module, `with wp.ScopedDevice(s.device)` inside
  `FallbackGrid.build`. Not changed in the rubber branch.
- Regression check (rubber side): `tests/test_rubber_stable_step.py::test_rubber_runtimes_leave_warp_global_state_unchanged`.

### 18.7 Results (formal v3 run and GPU smoke)

**Formal v3 run `rubber_phys_v3_001`** (commit `849e10b`, clean; `rubber_phys_tests_v3.json` sha256 `c7fdfc95…dbd9`;
`.deps/isaacsim-venv` Python, Warp 1.17.0 CPU, 24 workers on the shared Threadripper PRO 9975WX (load 25–36);
36 configurations, 2,839 s wall). Command:
`PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/run_rubber_phys.py --protocol v3 --tests 02,03,05,07,12
--device cpu --workers 24 --out artifacts/rl_ready/p2_1/rubber_phys_v3_001`. The run's `summary.json` was written by
the evaluator loaded at its start; `summary_with_validity.json` re-applies the same criteria plus the declared v3
validity condition (commit `68b2497`, `--evaluate-only`, no overwrite): identical verdicts, no invalid configuration.

| Test | Measured (v3) | Verdict |
|---|---|---|
| -02 small sag (24 configurations) | narrow δ/δ_EB 0.9813–1.0001; wide 0.9245–0.9674 (inside the bracket); reaction ≤ 5.5e-4; converged at 3.00–3.35 T1; no flag; **δ/δ_extrap 1.0039–1.0203: Lt20_t6mm_Wt4_nt3 = 1.0203 > 1.02** | **FAIL** (extrapolated, 1 of 24) |
| -03 scaling | exponents 4.011, −2.011 | PASS |
| -05 large sag | δ/δ_elastica 0.9991 / 0.9984 / 0.9981 (α 20 / 3 / 9); J̄ ∈ [0.9918, 1.0088] | PASS |
| -07a ramped (λ 1.5) | mean length +2.9e-5; energy drift 0.96 % of Ψ; no flag; lateral 2.8e-7 m; CFL_eff ≤ 0.64; J ∈ [0.967, 1.070] | PASS |
| -07b instant (λ 1.03) | mean length +4.4e-4; no flag; lateral 5.9e-7 m; CFL_eff ≤ 0.52; **energy drift 1.058 % of Ψ** | **FAIL** (energy) |
| -07 overall | | **FAIL** (07b energy) |
| -12 guard regression | RUBBER_CFL_EXCEEDED at control step 54 (the first after the release), RUBBER_J_DOMAIN and NONFINITE_STATE at step 56 | PASS |

Reading of the two failures (diagnostics after the formal run, not formal results):
- **-07b energy.** The loss is secular (mean −0.1 % of Ψ per axial period on top of a bounded ±0.3 % fluctuation)
  and comes from the clamp: from a λ 1.03 checkpoint (5 ms-interval ramp, dt 5.80 µs) the same release loses −0.72 %
  (mean, 60–75 ms) at CFL 0.5 and
  −0.49 % at CFL 0.25, while the free-free release has no secular change (+0.04 % mean, ±0.55 % fluctuation)
  (`followup_07_release_l103_*_002`; F6). The criterion is the declared one; the remedy is F8 (energy-consistent
  Dirichlet constraint) and a v4 -07b declared before its run.
- **-07a energy margin.** The declared accounting uses the traction work j·v_p(start of substep); the grid-consistent
  work f·Δx of the loaded end layer differs by 1.4 % of Ψ over the ramp (drift −0.48 % with it instead of +0.96 %).
  The PASS stands as declared, but at the 1 % level the traction channel is not resolved (F9).
- **-02 extrapolated.** v3 moved every static deflection by −0.02 % … +0.30 % (median +0.13 %) relative to the v2
  stop, as expected from §18.5, and the n_t 3 wide strips sit at 1.4–2.0 % from the two-grid extrapolation (v2:
  1.5–1.8 %). Geometrically similar configurations (same L/t, W/t, n_t, α) should give identical δ/δ_EB but scatter
  by up to 0.3 % (Lt20 W4t n_t 3: 0.96527 / 0.96556 / 0.96738; Lt20 W t n_t 3: 0.99700 / 1.00009 / 0.99797), and the
  extrapolation amplifies that scatter 2.3×: the float32 static state, not the stop rule, limits this criterion at
  its margin (F10). The v2 -02 PASS was obtained with the premature v2 stop; the v3 FAIL is the result under the
  v3 stop rule. A continuation of Lt20_t6mm_Wt4_nt3 to 6 T1 shows the float32 static state still creeping (§18.9),
  so the stop rule converges the dynamics but cannot make the float32 static state more precise than ~1e-3.

**GPU smoke `gpu_smoke_001`** (R1.4 / R1.7 L2 / RUBBER-PHYS-10 L3), queued once behind
`flock …/rcs_rl_runner/gpu1.lock` (acquired 16:35:46 with 29.6 GB free; run 16:35:47–16:36:45 KST), RTX 5090 GPU1,
code = `git archive` snapshot of `6c9477f` (guard not yet present), **SHARED_GPU** (co-tenant PID 2128623 762 MB at
start, PID 2144853 712 MB at the end; GPU1 utilisation 65 → 94 %). Workload E smoke variant: 97,280 particles/env, 50 Hz
control, no cameras, no recording.

| Part | N | material dt | substeps | env-steps/s (warm) | sim-time/wall | ms/substep | VRAM peak (process) | reset p95 | warm / cold wall | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A single (R1.4) | 1 | 5.80 µs (nominal μ) | 3,446 | 0.835 | 0.0167 | 0.347 | 742 MB | – | 1.197 / 1.252 s | PASS |
| B batched L2 | 1 | 3.00 µs (range maxima) | 6,662 | 0.432 | 0.0086 | 0.347 | 744 MB | 1.94 ms | 2.313 / 2.419 s | PASS |
| B batched L2 | 4 | 3.00 µs | 6,662 | 0.621 | 0.0124 | 0.967 | 840 MB | 0.86 ms | 6.442 / 6.495 s | PASS |
| B batched L2 | 8 | 3.00 µs | 6,662 | 0.735 | 0.0147 | 1.633 | 968 MB | 0.40 ms | 10.879 / 12.672 s | PASS |
| C L3 band | 4 (RUBBER-PHYS-04 envs) | – | – | – | – | – | – | – | – | **FAIL**: 25 of 480 values outside the band, worst excess 2.1e-5 relative, single-repeat spread up to 3.3e-3 |

L2: finite, mass constant between resets, J̄ ∈ [0.998, 1.002], env-ownership audits 0, every invalid bit 0, ≤ 1.5 GB.
L3 FAIL preserved: the batched values lie within 2.1e-5 of a band whose own repeat spread is 3.3e-3, i.e. consistent
with the frozen P2G/MUSL atomics nondeterminism, but the declared band (min/max of only three repeats) is statistically
weak; a new rule must be declared before a new run (F3). RUBBER-PHYS-10 is therefore L0/L1 PASS (CPU), L2 PASS, L3 FAIL.

### 18.8 Remaining tasks (rcs-impl)

| Task | Owner | Content | Size |
|---|---|---|---|
| F1 | Phase 1 (not rubber) | Pin the device in `test_fallback_grid_reproduces_warp_grouped_query_sequence` (`with wp.ScopedDevice("cpu"):` around `fallback.build`) or pin `s.device` in a versioned `FallbackGrid.build`; re-run with a visible GPU | S |
| F2 | rcs-impl, GPU | Re-run `gpu_smoke_rubber.py` part B with `stable_step_guard=1.0` (N 1/4/8, ≤ 1.5 GB): L2 invariants incl. zero guard violations, ms/substep with and without the guard | S |
| F3 | rcs-impl + review | RUBBER-PHYS-10 L3: the v2 band (min/max of 3 GPU repeats widened by 1e-5) failed (§18.7) and is statistically weak (a fourth i.i.d. sample leaves the range of three with probability 1/2); declare a new L3 rule in a v4 protocol **before** its run (e.g. ≥ 10 repeats and a band from their spread), keep the v2 FAIL | M |
| F4 | P2-2 (R2.2, R2.3) | Refinement / GCI with static_stop_v3 and the guard on; quantify the float32 static floor (~6e-4 band) and the float64-marker noise (~1e-9 m per check); batched (L0) runs of -07 v3 and -12 | M |
| F5 | P2-2 (R2.4) | Profile the guard on GPU; fuse it into the stress pass only if bitwise equal | S |
| F6 | P2-4 | Fold scene RUBBER-CONT-05 from the λ 1.5 instantaneous release: opt-in self-contact, `friction_self`, r = 2, guard on, energy balance with contact work; declared before implementation | L |
| F7 | P2-6 | `RUBBER_CFL_EXCEEDED` → `terminated`; guard on by default in `rubber_rl_v1` envs; receipts print `cfl_eff_max` | S |
| F8 | P2-2 (before a v4 -07) | Energy-consistent Dirichlet constraint: quantify the clamp dissipation (fixed vs free-free, dt, grid, mask x vs xyz) and try a constraint applied consistently to the MUSL mass (e.g. constrained DOFs removed from the M_lM_c⁻¹M_l system, or the particle-side FLIP increment of constrained nodes excluded); bitwise-off by default; then a v4 RUBBER-PHYS-07b declared before its run | M |
| F9 | P2-2 | External-work accounting of traction groups from the grid-consistent particle displacement (Δx_p · j/dt) instead of j · v_p(start): in 07a the two measures differ by 1.4 % of Ψ over the force ramp (§18.7) | S |
| F10 | P2-2 (R2.2) | Float32 static-state precision: geometrically similar -02 configurations scatter by up to 0.3 % in δ/δ_EB; quantify (repeat, longer relaxation, dt) and, if needed, an opt-in compensated (float64 / Kahan) F and position accumulation for the initializer and quasi-static tests, versioned, bitwise-off by default | M |

### 18.9 Continuation check of the -02 outlier

`followup_stop_rule_v3_t6w4nt3_001` (diagnostic; `diag_stop_rule.py --label Lt20_t6mm_Wt4_nt3 --extra 3
--stop-rule static_stop_v3`): static_stop_v3 reproduces the formal value bitwise in its deterministic CPU run
(δ/δ_EB 0.96738 at 3.05 T1); continuing the same damped relaxation, the float64 deflection keeps moving: 0.96727 /
0.96783 / 0.96830 at 4.05 / 5.05 / 6.05 T1, with Ψ +0.38 %, while every physical mode has decayed by more than
e^(−6π). The float32 static state therefore creeps at about 5e-4 of δ per T1 in this configuration, away from its
geometrically similar siblings (0.9653, 0.9656): the -02 result depends on the stopping time at the 1e-3 level. Mechanism
not established (candidates: float32 rounding of F near 1, where the ulp below 1 is half the ulp above 1, under the
noise-driven increments; to be resolved in F10). This does not change the formal verdict (FAIL), which stays as run.

## 19. P2-2 decisions (amendment, 2026-11-10, rcs-impl; declared with protocol v4 before its formal runs)

Scope: ADR §18.8 F2–F5, F8–F10 and §15 R2.1–R2.4. Nothing frozen changed (v1–v7_2, law 8, evaluators, Phase 1 batched
modules, `batched/constitutive.py`, `configs/rl_benchmark/*`, `scripts/rl_ready/*`, `rubber_phys_tests_v1/v2/v3.json`,
the P2-1 records). Raw outputs: `artifacts/rl_ready/p2_2/`; evidence `docs/evidence/rl_ready/p2_2_rubber_001.json`;
report `docs/reports/rl_ready/P2_2_RUBBER_HYPERELASTIC.md`. Material status unchanged: `SIMULATOR_ONLY_UNCALIBRATED`.

### 19.1 Opt-in solver options of `tlmpm_rubber_v1` (single-env; defaults are the v1 launch sequence, bitwise)

| Option | Id | What it changes | Why (diagnostic, before protocol v4) |
|---|---|---|---|
| `dirichlet` | `consistent_mass_projection_v1` (F8) | The pass-1 BC is skipped; after G2P the particle velocities are projected onto the constraint set in the kinetic-energy metric: A = (W M_p Wᵀ) on the constrained (node, component) pairs of each active group (constant in TLMPM, inverted once in float64), μ = A⁻¹(p_C − m_C v_presc), v_p −= Σ_{I∈C} w_pI μ_I; the MUSL pass is repeated, so the MUSL grid velocity meets the constraint exactly. Reaction −m_I μ_I; work = impulse · v_presc. | The lumped projection zeroes the constrained grid velocity but feeds a lumped-mass (Jacobi-like) correction back through the FLIP increment, so the particle state leaves the constraint subspace and the scheme stops being symplectic there. With v = Wᵀa, KE = ½aᵀM_c a and g = M_l⁻¹M_c a, the M_c-orthogonal projection onto {(M_c a)_C = m_C v_presc} is exactly a correction supported on the constrained nodes with the CC block of M_c, and the projected step is symplectic Euler on the constraint subspace. Measured (`f8_clamp_dissipation_001`, λ 1.03 release, 20 T_ax): lumped slope −4.3e-4 Ψ/T_ax (n_t 4, mask x, CFL 0.5) to −1.4e-3 (n_t 3, xyz); projection +0.5e-5 … +1.3e-5 /T_ax, same as free-free (+1.0e-5 … +1.3e-5). |
| `traction_work` | `grid_displacement_v1` (F9) | Traction work accumulated after the position update as f_t (m_p/M_t)·(x_end − x_start) | The grid-consistent work of a nodal force in the symplectic scheme is f_I·Δu_I = Σ_p j_p·Δx_p/dt; the v1 j·v_p(start) differed by 1.4 % of Ψ over the -07a force ramp (§18.7). |
| `state_precision` | `f64_shadow_v1` (F10) | F and x accumulated in float64 shadows (the frozen `_deformation` / `_grid_positions` formulas in float64) and rounded once per substep to the float32 state | `f10_static_precision_001` (-02 Lt20 W = t n_t 3, t 3 / 4 / 6 mm, relaxation continued to 10 T1): float32 δ/δ_EB wanders in a 1.9–2.9e-3 band after 3 T1 with Ψ +0.8–1.0 %, and the three geometrically similar configurations differ by 2.8e-3 at 3 T1 (CPU repeat bitwise identical: deterministic); with the shadow the band is ≤ 4.1e-7 and the three agree to 1e-5 (0.99860). Mechanism: near F ≈ I the per-substep increments dt·∇v fall below half an ulp of F (and of x) once the motion has decayed, so float32 F changes only by noise-driven ulp jumps — the §18.9 creep. |

The batched path refuses lives built with any non-default option (`UNSUPPORTED_SOLVER`); a batched projection /
shadow is future work (P2-6 grips). `rubber_strip_v1` also admits ppc = 3, only as the refinement study's quadrature
check (§14.4).

### 19.2 Protocol v4 (`configs/rubber/rubber_phys_tests_v4.json`, SHA-pinned, declared before its formal runs)

- -02/-03/-05: `state_precision f64_shadow_v1` (method change justified by 19.1 F10; initializer, static_stop_v3,
  guard, criteria and thresholds unchanged). The v3 -02 FAIL stays the v3 result.
- -07 (both variants): `dirichlet consistent_mass_projection_v1`, `traction_work grid_displacement_v1`; criteria and
  thresholds unchanged. The v2 and v3 -07 FAILs stay recorded.
- -06 / -08 re-run with the guard on (R2.1 full runs; protocol otherwise unchanged since v2).
- -10 L3: rule `rubber_l3_band_v4` (`robotic_cutting_sim.rubber_l3`): per (env, observable) the statistic
  M = max_k |y(k) − ȳ_ref(k)|; leave-one-out M_i from 20 single-env GPU repeats; one-sided lognormal prediction bound
  at α = 0.05/16 (Bonferroni over 16 statistics; Cantelli fallback when Shapiro–Wilk p < 0.01 or an M_i = 0); nominal
  false-reject ≤ 5 % per batched run (Monte Carlo of the rule on random-walk series: 2.1–2.3 %); 10 held-out single
  repeats classified and reported. The v2 L3 FAIL stays the v2 result.
- -11: Q1–Q4 on n_t ∈ {2, 3, 4, 6} × CFL {0.5, 0.25}, β 50 at n_t 3/4, ppc 3; GCI of the production grid (n_t 3) from
  the triplet (3, 4, 6) with Celik's observed order, p_eff = clip(p_obs, 1, 2), Fs 1.25 (3.0 if p_obs ∉ [1, 3] or
  oscillatory), GRID_INDEPENDENT when the triplet range ≤ 1e-4; pass GCI ≤ 2 % and CFL 0.5 → 0.25 change ≤ 0.1 %.
- New sections: `throughput_v4` (R2.4/F5), `batched_gpu_v4` (R2.1 GPU N = 4 of -07 v3 / -12), `batched_l0_v4` (F4).

### 19.3 Results (P2-2 formal runs; details: `docs/reports/rl_ready/P2_2_RUBBER_HYPERELASTIC.md`)

| Item | Result |
|---|---|
| F1 (Phase 1 fix `db036a0`) | PASS with a GPU visible (fallback-grid test 3 × 3 alone, file 15 / 15) |
| F2 guard-on GPU smoke N 1/4/8 | PASS: L2 invariants, 0 guard violations, ≤ 922 MB; guard +1.2–1.9 % per substep (SHARED_GPU) |
| -10 L3 (rule v4) | PASS: batched max M/U 0.877, held-out singles 10 / 10 within (v2 L3 FAIL preserved) |
| -02 / -03 / -05 (v4, f64 shadow) | PASS (δ/δ_extrap ≤ 1.0178; similar configurations equal to 1e-6; v3 -02 FAIL preserved) |
| -06 (v4, guard) | PASS |
| -07 (v4, projection + grid traction work) | PASS: 07a energy 0.0047 % of Ψ, 07b 0.33 % (v2 / v3 FAILs preserved) |
| -08 (v4, guard) | FAIL (validity): stresses exact to 1.6e-6, but γ = 2 reaches CFL_eff 1.30 > 1.0 (bound ratio 2.52 at r = 1); a v5 -08 needs r ≥ 1.3 declared before its run |
| -11 refinement | FAIL: production GCI (n_t 3) Q1 3.98 % (p_obs 3.26 outside [1, 3] → Fs 3), Q2 1.28 %, Q3 0.01 %, Q4 0.92 %; dt criterion PASS for all |
| β 50 vs β 20 | reported: simulated differences equal the analytic compressible ones to ≤ 0.16 %; β 20 vs incompressible −2.7 % stress at λ 1.5 |
| R2.4 workload E N 1/4/8/16 | 0.64 / 1.35 / 1.47 / 1.52 env-steps/s, ≤ 952 MB, guard 2.1 % of GPU time (no fusion) |
| Batched GPU N = 4 (-07 v3, -12) and CPU L0 (-07 v3, -12) | PASS (07b v3 batched energy 1.058 % reported, lumped) |

Open decisions for R2.5: production n_t for bending observables (Q1), -08 with an explicit deformation reserve,
batched projection / shadow before P2-6.

## 20. P2-3 prerequisites in the batched path (amendment, 2026-11-10, rcs-impl)

Scope: the P2-3 prerequisites that hold whatever the P2-0..P2-2 acceptance review decides (P2_2 report §14 items 2,
3, 7). Viscoelasticity itself is NOT implemented: `rubber_visc_tests_v1` is declared first, after the review. Nothing
frozen changed (v1–v7_2, law 8, evaluators, Phase 1 batched modules, `batched/constitutive.py`,
`configs/rl_benchmark/*`, `scripts/rl_ready/*`, `rubber_phys_tests_v1..v4.json`, the single-env reference
`tlmpm_rubber_v1.py`). Changed: `batched/rubber_batch.py`, `batched/rubber_layout.py`,
`batched/rubber_material_batch.py` (Phase 2 files). Raw outputs: `artifacts/rl_ready/p2_3_prep/`.

### 20.1 Batched execution of the §19.1 options (opt-in, versioned by the single-env ids)

A batch runs the options its single references (`LiveRubberMaterial(..., dirichlet=, traction_work=,
state_precision=)`) were built with; the option values are part of the uniform batch signature (mixed options:
`UNSUPPORTED_NONUNIFORM_BATCH`); the §19.1 `UNSUPPORTED_SOLVER` refusal is lifted for exactly these ids. With the
defaults the batch allocates no option array and launches no option kernel (bitwise-off).

| Option | Batched execution |
|---|---|
| F8 `consistent_mass_projection_v1` | rows of all envs concatenated (global node / group / block ids, CSR offsets per env); the float64 block inverses copied bit-for-bit from each single reference; `_proj_rhs`, `_proj_solve`, `_proj_particles` of `tlmpm_rubber_v1` launched unchanged, `_proj_reaction_b` with the group's env work; lumped pass-1 BC skipped and MUSL repeated as in the single solver. A reset with new material values rebuilds that env's inverses on the host with the single solver's `projection_blocks` (the particle mass enters the consistent-mass blocks; cached per (env, mass)) and writes them into the preallocated array (no device allocation, no recapture). |
| F9 `grid_displacement_v1` | `_traction_work_grid_b` (work of the particle's env) after the position update; the v1 accumulation goes to a discarded per-env `work_scratch`, as in the single solver. |
| F10 `f64_shadow_v1` | flat `F64` / `x64` shadows; `_deformation_f64` / `_positions_f64` of `tlmpm_rubber_v1` launched unchanged; `reference64` (the single solver's exact float64 X) restores `x64` on reset. |

The masked reset restores the option state of the reset envs (shadows to identity / `reference64`, projection scratch
and `work_scratch` to zero, inverses to the env's material); the vars()-complete classification covers the option
arrays (`OPTION_RESET_ARRAYS`, `OPTION_CONSTANT_ARRAYS`).

### 20.2 BC launch: per-control-step motion table (`motion_table_v1`, new default)

Diagnosis first: the P2-2 profile attributed 14–17 % of GPU time to the `_bc_b` "node loop". An interleaved GPU0
micro-benchmark (`bc_bench_001`, workload E N = 4, 400 rounds, SHARED_GPU) shows the loop over unconstrained nodes is
not the cost: the BC pair of one substep costs 0.0806 ms (median; p10–p90 0.0796–0.0820) over every node and 0.0835 ms
over the packed list of the 768 constrained nodes (`constrained_list_v1`), against a 0.0037 ms one-thread launch floor.
Without the reaction (2 `group_target` evaluations per thread) a launch costs about 34 µs, with it (3 evaluations and
the float64 atomics) about 50 µs, holding or rotating alike: the cost is the per-thread float64 evaluation of the
group map (slerp, quaternion to matrix).

`bc_launch` (constructor of `BatchedTLMPMRubber` / `BatchedRubberMaterial`):

| Value | BC pass |
|---|---|
| `all_nodes_v1` | the P2-1/P2-2 launch: `_bc_b` over every node, maps evaluated per node |
| `constrained_list_v1` | the same body over the packed constrained-node list (no measured gain; kept for the A/B) |
| `motion_table_v1` (default) | `_motion_table_b` evaluates M(s), b(s) of every group at every substep boundary s = k / substeps once per control step (one thread per (k, group), launched by `set_group_motions` / `prepare_motion_table`), and `_bc_table_b` over the constrained-node list computes M X + b from the table; past the table (direct `step()` calls beyond the control step) it evaluates `group_target` as before |

The table holds exactly the M, b that `group_target` computes before M X + b, so on the Warp CPU device every variant is
byte-identical to `all_nodes_v1` and to the single reference (tests: case A heterogeneous, case C two grips with partial
masks, rotation and an inactive grip, affine block; v1 options and F8 + F9 + F10; past-table steps). GPU:
`bc_bench_002` BC pair 0.0185 ms with the table (4.3× less) plus 0.019 ms per control step for the table (≈ 4 ns per
substep); end-to-end interleaved A/B (`bc_e2e_none_001`, workload E N = 4, guard 1.0, graph chunks, 10 rounds,
SHARED_GPU): 0.404 → 0.331 ms per substep, 1.98 → 2.42 env-steps/s, paired ratio median 0.804 (0.74–0.84, all rounds
faster); with F8 + F9 + F10 (`bc_e2e_all_001`) 0.562 → 0.532 ms, ratio median 0.949 (0.81–1.14). The sequential smoke
rows (`gpu_smoke_002/003`, rows run one after another under changing co-tenant load) do not resolve the difference
(0.363 vs 0.386 ms in `_003`); their per-kernel splits are inflated by co-tenant scheduling (a one-thread kernel shows
26 µs per substep in `gpu_smoke_001`) and are not used for the decision. On a GPU the BC float64 reaction atomics were
and remain order-nondeterministic.

### 20.3 F8 block structure in the batched path (review F8)

The batched projection keeps the single solver's structure: one dense block per (env, group, component), solved
independently. Envs cannot couple (disjoint node ranges). Within an env, a particle whose stencil holds constrained
nodes of two groups in the same component would couple two independently solved blocks; `pack_rubber` refuses such
lives with `UNSUPPORTED_SOLVER: PROJECTION_CROSS_GROUP_COUPLING` (`rubber_layout.projection_cross_group_particles`;
the code is `UNSUPPORTED_SOLVER` with a fixed tag because `batched/errors.py` is a frozen Phase 1 module). Every scene
used so far has none (case A one group, case C faces ≥ 2 cells apart, pins inside a group); a one-cell case C strip has
32 coupled particles and is refused (test). Merging coupled blocks is future work (P2-4 / P2-6 multi-grip scenes).

### 20.4 L0 is byte-wise from here on; erratum on earlier "bitwise" wording (review F5)

The batched rubber L0 / L1 / A-B tests now compare dtype, shape and raw bytes (`_same_bytes`), not `np.array_equal`
(which equates +0.0 and −0.0 and needs `equal_nan` for NaN). The only byte difference this found was a signed zero in
the derived wrench of a zero reaction (an inactive grip): Warp's unary minus is `0 - x` (+0.0 stays +0.0), NumPy's flips
the sign bit. `_wrench_b` now computes `(r * -1) / control_dt`, byte-identical to the single reference's
`-reaction / control_dt` for every finite value and both zeros (a NaN keeps its sign bit; a NaN wrench sets
`NONFINITE_STATE`). This changes no state array and no nonzero wrench value.

Erratum (wording only; records unchanged): the P2-1/P2-2 batched L0 results, including §19.3 "CPU L0 (-07 v3, -12)"
and the evidence field `bitwise_equal_every_step: true` of `f4_batched_l0_12_003`, are **value-exact**
(`np.array_equal`, NaN positions equal, signed zero not distinguished). For -12 the first byte-level difference is at
step 54 in the derived wrench (the signed zero above); state arrays were not compared byte-wise in those runs.

### 20.5 Results

CPU (Warp 1.17.0, `.venv`): new `tests/test_rubber_batched_options.py` (23 tests) — L0 byte-wise vs the single
reference after every control step for F8, F9, F10 and all three, at N = 1 and N = 3 heterogeneous (geometry, μ, ρ,
friction, motion; different constrained-row counts per env), case C two grips with rotation, partial masks and an
inactive grip, the guard on, the affine block; L1 perturbation isolation, masked partial reset (vars()-complete with
the option arrays) equal to a fresh env and evolving like one, reset with resampled material equal to the single
reference built with it (F8 inverses rebuilt) and restored on the next plain reset; refusals (mixed options, unknown
launch, cross-group coupling). A mutation check (one ulp in a block inverse, 1e-12 in x64, the v1 traction work into
`work`) is detected by the comparison. Full suites (`cpu_tests_002`, commit c820c59, clean tree; `tests/test_rubber_*.py`, `test_live_rubber_material.py`, `test_tlmpm_rubber_v1.py`, `test_batched_*.py`, `test_rl_policies_batched.py`): `CUDA_VISIBLE_DEVICES=''` 319 passed, 4 skipped (528 s); `CUDA_VISIBLE_DEVICES=0` 319 passed, 4 skipped (534 s); the 4 skips are the torch / WP-17-recording tests of `test_rl_policies_batched.py` (NOT_RUN in `.venv`).

GPU0 smoke `gpu_smoke_003` (commit c820c59, clean tree, driver 570.211.01, workload E N = 4, 41,220 particles/env,
50 Hz, dt 4.0e-6 s, 4,996 substeps, guard 1.0, graph chunks 500, no cameras / recording, SHARED_GPU): rows
`default_all_nodes`, `default_motion_table`, `options_motion_table` (F8 + F9 + F10) all **L2_INVARIANTS_PASS** (finite,
mass constant between resets, invalid bits 0 before and after 4 masked resets with resampled material, guard violations
0, audits 0, max CFL_eff 0.498); options row: F = float32(F64) and x = float32(x64) byte-equal, F8 MUSL momentum on the
2,304 constrained rows minus m v_presc 1.0e-8 (1.7e-8 after the resets) of max |p|; VRAM peak 760 / 762 / 854 MB.
Warm: 2.20 / 2.07 / 1.71 env-steps/s (sequential rows, see §20.2). `gpu_smoke_001` (list launch) and `_002` (table,
before the F10 metadata) are preserved, both L2 PASS; their metas lack the driver version.

Costs of the options (workload E, N = 4, SHARED_GPU): in `gpu_smoke_003` (sequential rows) the table launch with v1
options runs 0.386 ms per substep and with F8 + F9 + F10 0.469 ms (+21 %); the two end-to-end A/B runs give 0.331
(v1 options) and 0.532 ms (all options) with the table, but ran at different times and are not a matched comparison.
The projection adds four row kernels, a particle kernel and a second MUSL scatter per substep; device arrays
65.9 → 91.2 MB. A reset with resampled material rebuilds that env's F8 inverses on the host with the single solver's
`projection_blocks` (0.35–0.87 s per env on workload E; cached per (env, mass)); a reset to the packed material copies
the packed inverses.

Moving-grip energy (review F8 recommendation, diagnostic probe, not a protocol test; `probe_moving_grip_energy.py`,
case A n_t 2, grip lifted 1 mm and rolled 0.15–0.3 rad over 6 ms, then held): residual Ψ + KE + E_grav − W relative to
max(Ψ + KE) is −22 % (linear-in-time motion: velocity jumps at start / stop are inelastic for any constraint) and
−1.6 … −2.0 % with a C2 smootherstep, **the same for the lumped clamp and the projection**. The F8 projection removes the
clamp's dissipation at rest (§19.1) but does not close the energy books of a moving grip to better than ≈ 2 % here; the
cause (constraint-work quadrature j · v_presc at the end of the substep, the jaw particles' kinetic energy, or the
discrete motion) is not yet isolated. Open before batched VISC hysteresis runs with moving grips and P2-6.

### 20.6 Open

- Unreviewed: everything after 3a43d1b on `claude/rl-rubber` (b922665 … this section), including the default
  `motion_table_v1`.
- Moving-grip energy residual (§20.5).
- Host-side F8 inverse rebuild on resampled reset (0.35–0.87 s per env): a vectorised assembly with the same
  accumulation order would keep the bits; not done.
- The F8 row kernels `_proj_rhs` / `_proj_reaction_b` still evaluate `group_target` per row (the same float64 cost as
  §20.2); they could read the motion table.
- Viscoelasticity (`neo_hookean_jbar_prony_v1`) and `rubber_visc_tests_v1` are not started (declared after the review).

## 21. P2-3: moving-grip energy books and the viscoelastic implementation design (amendment, 2026-11-10, rcs-xhigh)

Scope: (1) the moving-clamp energy gap left open in §20.5/§20.6; (2) the binding implementation design of D7
(`neo_hookean_jbar_prony_v1`, refining §10), its protocol `rubber_visc_tests_v1` (declared and SHA-pinned before any
implementation run) and the rcs-impl task list. Nothing frozen changed: v1–v7_2, law 8, evaluators, every Phase 1
module (including `batched/constitutive.py`, `batched/errors.py`), `configs/rl_benchmark/*`, `scripts/rl_ready/*`,
`rubber_phys_tests_v1..v4.json`, and the single-env reference **`tlmpm_rubber_v1.py` (sha256 `28fc5189…c8d5`) and
`rubber_constitutive.py` (`c12cc77e…b869`) stay byte-identical** (the R2.5 freeze recommendation). Branch
`claude/rl-rubber-p23` (from `1b21801`). Raw outputs `artifacts/rl_ready/p2_3/`. Material status unchanged:
`SIMULATOR_ONLY_UNCALIBRATED`.

### 21.1 Moving Dirichlet grip: why the books did not close, and the fix

**Diagnosis** (`scripts/rubber/p2_3/diag_moving_clamp_energy.py`, run `moving_clamp_diag_001` at `87c3696`, Warp CPU;
the §20.5 probe scene: case A 12 × 3 × 3 mm, n_t 2, μ 0.4 MPa, β 20, ρ 1150, gravity; the grip lifts 1 mm and rolls
0.15 rad over 6 ms, then holds 6 ms; 320 particles). A subclass, local to the script, reads after each projection / BC
launch the per-row constraint impulse Λ_n (on the material) and evaluates three quadratures of its work:
*end* Λ_n·v_presc,n (what `tlmpm_rubber_v1` accumulates), *start* Λ_n·g_n,C and *kick average* Λ_n·(g_n,C + v_presc,n)/2,
where g_n is the P2G-pass grid velocity `old` = p/m. Residual (Ψ + KE_particle + E_grav − W)/max(Ψ + KE) at the end of
the hold (max over the run in brackets):

| Profile, constraint | control dt, substeps | end (v1) | kick average | start |
|---|---|---|---|---|
| linear, lumped | 100 µs, 12 | −22.1 % | −4.24 % (4.24 %) | +13.7 % |
| linear, F8 (± F10) | 100 µs, 12 | −19.24 % | −0.006 % (0.38 %) | +19.2 % |
| linear, F8 + F10 | 100 µs, **24** | −19.18 % | −0.004 % (**0.19 %**) | +19.2 % |
| linear, F8 + F10 | **25 µs**, 3 | −19.24 % | −0.006 % (0.39 %) | +19.2 % |
| smootherstep, lumped | 100 µs, 12 | −1.59 % | −0.33 % (0.67 %) | +0.93 % |
| smootherstep, F8 (± F10) | 100 µs, 12 | −1.62 % | +0.19 % (0.23 %) | +2.00 % |
| smootherstep, F8 + F10 | 100 µs, **24** | −1.66 % | +0.09 % (**0.11 %**) | +1.85 % |
| smootherstep, F8 + F10 | **25 µs**, 3 | **−0.35 %** | +0.19 % (0.23 %) | +0.73 % |

Candidates, one by one:

| Candidate | Verdict | Evidence |
|---|---|---|
| Which velocity the constraint work is evaluated at | **Root cause** (projected constraint) | With the kick average the F8 books close to 6e-5 (linear) / 1.9e-3 (smooth) at the end; the residual is a bounded oscillation that halves with dt (0.38 → 0.19 %, 0.23 → 0.11 %), i.e. the O(dt) energy error of symplectic Euler. The end-velocity error is dt-independent (−19.2 % at 12 and 24 substeps) and is exactly ½ Σ Λ_n·(v_presc,n − g_n,C) |
| Discontinuous acceleration (linear profile, start/stop jerk) | Where the quadrature error concentrates, not a loss | The end − kick difference is a staircase: +9.77 % of S at the start jump, +9.46 % at the stop (= ½ Δv·m_eff Δv per jump). With the kick average a linear profile closes as well as a smooth one (6e-5). The jumps inject real, correctly accounted energy into vibration |
| Time integration of the motion table vs the substep | Explains the "smooth" 1.6 % | The pose is linear (slerp) within a control step, so v_presc is piecewise constant and jumps at every control-step boundary even for a C2 control-level profile: the v1 error scales with the control interval (−1.62 % at 100 µs, −0.35 % at 25 µs) and not with dt. The motion table itself is exact (§20.2, bitwise) |
| FLIP increment of constrained nodes (lumped clamp) | Real numerical dissipation, removed by F8 | Lumped with the kick average still loses 4.2 % (linear) / 0.33 % (smooth): its FLIP increment is a lumped-mass correction and the MUSL-pass overwrite never reaches the particles, so a velocity jump is partly inelastic (§18.1 F6). No quadrature closes the lumped books |
| Missing kinetic energy of the moving clamped (jaw) particles | Not missing | `kinetic_particle` sums every particle; the jaw particles carry up to 7.5 % (linear) / 15 % (smooth) of S. Under F8, v_p = Wᵀa always, so the particle KE equals ½πᵀBπ exactly |

**Root cause.** Pure FLIP + MUSL with fixed weights is symplectic Euler in (π, u) with KE = ½πᵀBπ (B = M_l⁻¹M_cM_l⁻¹,
g = Bπ; §18.1 F1). The F8 projection is the standard constrained symplectic Euler for the linear holonomic
constraint u_C = c(t): π_{n+1} = π_n + dt f(u_n) + Λ_n, Λ supported on the constrained rows, (Bπ_{n+1})_C = v_presc,n
(the projection of §19.1 solves exactly B_CC Λ_C = v_presc − (Bπ*)_C). KE is quadratic, so
KE_{n+1} − KE_n = (dt f + Λ_n)·(g_n + g_{n+1})/2 *exactly*: the work conjugate to the discrete kinetic energy is
Λ_n·(g_n,C + v_presc,n)/2. `tlmpm_rubber_v1` books Λ_n·v_presc,n, i.e. it attributes the kick, which happens between
two drifts, entirely to the following drift. The error per substep is ½Λ_n·(v_presc,n − g_n,C); summed, it is
½ Δv·m_eff Δv at every discontinuity of the prescribed velocity (m_eff = m_C A⁻¹ m_C ≥ m_C) plus an O(dt) force term.
For a fixed grip (v_presc = 0, §19.1 runs) both quadratures give ≈ 0, which is why F8 closed the books at rest.
Physically the kick average is also the right limit: a rigid grip that changes its velocity quickly but continuously
from v_old to v_new does work ½(v_old + v_new)·J, not v_new·J.

**Decision.** The constraint is right; the bookkeeping is versioned: option `constraint_work`
(`rubber_constraint_work.py`), `end_velocity_v1` (default, the §19.1 accounting, unchanged) or **`kick_average_v1`**
(W += Λ·(g_n,C + v_presc)/2 with g_n = `old` read before the MUSL grid update overwrites it; equal to v_presc,n−1 on an
active row up to float32, the free velocity on a freshly activated one). Bookkeeping only: every state array, the
reaction and the wrench are byte-identical (test); the v1 end-velocity work is reported in `work_end_velocity`.
`kick_average_v1` requires `dirichlet = consistent_mass_projection_v1` (with the lumped clamp it is refused: that
constraint is not symplectic and has no conjugate work). Every P2-3 test with a moving grip uses F8 + `kick_average_v1`;
a lumped grip is never used for energy, hysteresis or dissipation measurements.

**Implementation** (`59fca99`). Single env: `TLMPMRubberCW` subclasses the frozen `TLMPMRubber` and re-launches the
parent's F8 kernels in the parent's order, diverting `_proj_reaction`'s work to `work_end_velocity` and adding
`proj_work_kick_average` (one thread per constrained row; target `env_of_group[g]`, a one-env zero array in the single
solver). `LiveRubberMaterial(constraint_work=…)` builds the subclass only for the non-default value. Batched:
`BATCHED_OPTIONS["constraint_work"]`, part of the uniform signature (mixed values refused), the same kernel launched
unchanged after `_proj_reaction_b`, per-env `work_end_velocity` cleared by the masked reset (`OPTION_RESET_ARRAYS`).
Tests (`tests/test_rubber_constraint_work.py`, 13): default builds the frozen class; refusals; state byte-identical and
`work_end_velocity` = the default run's `work`; kernel = host float64 every substep; moving-grip closure (linear and
smooth); L0 byte-wise N = 1 / N = 3 heterogeneous / case C two grips with rotation and an inactive grip; masked reset;
vars()-complete classification; mixed refusal. `test_rubber_batched_options.py::test_reset_classification_is_vars_complete_with_options`
now builds its all-options batch with `kick_average_v1` too (otherwise the declared table names an array the batch
does not have). CPU suites: rubber tests 182 passed with `CUDA_VISIBLE_DEVICES=''` and with `=0`.

**Formal result: `rubber_moving_grip_tests_v1`** (declared and pinned at `ea32f2a`, sha256 `ff14f39d…78bc`, before the
run; S1 = the diagnostic scene, S2/S3 unseen). Run `moving_grip_v1_001` (commit `ea32f2a`, clean tree, Warp 1.17.0 CPU,
15 workers, 22:08:38–22:09:34 KST; the runner smoke `moving_grip_v1_smoke_001`, three control steps, is preserved and
labelled SMOKE). Options F8 + `kick_average_v1` + F10, guard 1.0, CFL 0.5 and 0.25 (2× substeps).

| Scene | Profile | Particles, substeps | M (CFL 0.5) | H | M (CFL 0.25) | ratio | v1 end-velocity M (reported) |
|---|---|---|---|---|---|---|---|
| S1 cantilever lift + roll, n_t 2 | linear | 320, 12 | 0.384 % | +3.3e-4 | 0.189 % | 0.49 | 19.5 % |
| S1 | smootherstep | 320, 12 | 0.227 % | +3.7e-4 | 0.113 % | 0.50 | 2.03 % |
| S2 wide cantilever lift + yaw, n_t 3 | linear | 3,744, 44 | 0.094 % | +6.4e-5 | 0.049 % | 0.52 | 4.19 % |
| S2 | smootherstep | 3,744, 44 | 0.057 % | +1.9e-5 | 0.028 % | 0.50 | 0.78 % |
| S3 two-grip load–unload λ 1.2, n_t 3 | linear | 4,320, 35 | 0.070 % | +6.2e-5 | 0.034 % | 0.49 | 0.055 % |
| S3 | smootherstep | 4,320, 35 | 0.088 % | +4.6e-5 | 0.043 % | 0.49 | 0.154 % |

M = max |r|, H = mean of r over the final hold, r = (Ψ + KE + E_grav − W)/max(Ψ + KE). **RUBBER-GRIP-01 PASS**
(M ≤ 1 %, |H| ≤ 0.2 %), **-02 PASS** (ratio 0.49–0.52 ≤ 0.65: first order, i.e. numerical), **-03 PASS** (no flag;
CFL_eff ≤ 0.565, J ∈ [0.970, 1.028]), **-04 PASS** (N = 2 heterogeneous batch byte-identical to two single
references after every control step: S1 120, S2 80, S3 110 steps, every array incl. `work_end_velocity`, F64/x64 and
the projection arrays, diagnostics, markers, wrenches). In S3 the v1 error is small because Ψ dominates S and the
velocity jumps are small; its dt-ratio (2.0 / 0.41) is erratic because the end-velocity bias and the O(dt) error are
of similar size there (reported).

**Consequences and limits.** The v4 -07 PASSes are unaffected (no moving grip in their energy windows) and so is -06
(stress criteria). The velocity jumps at control-step boundaries are part of the prescribed motion (the pose is
linear within a control step) and now cost exactly the work they do; a C¹ pose interpolation would be a separate,
versioned motion option and is not needed for the books. The batched F8 cross-group refusal (§20.3) still applies.
Only CPU runs here; on a GPU the float64 work atomics stay order-nondeterministic (as the reaction since P2-1).

### 21.2 `neo_hookean_jbar_prony_v1`: binding implementation design (refines §10)

**21.2.1 Stress.** With S⁰(C) = μ₀J^(−2/3)(I − (I₁/3)C⁻¹) (C⁻¹ = cof Fᵀ cof F / J², J floored at 0.05 for evaluation as in
§5), the §10 stress S = g_∞S⁰ + ΣQ_i, P = FS + p(J̄) cof F is evaluated in the **overstress form**

```
P = P_NH(F, J̄; μ₀, κ) + F · Σ_i (Q_i − g_i S⁰)        (P_NH = rubber_constitutive.nh_jbar_pk1, unchanged)
```

(algebraically identical, since P_NH = F S⁰ + p cof F). The overstress Σ(Q_i − g_iS⁰) is summed in float64,
multiplied by F in float64, rounded once and added to the float32 P_NH. Consequences: the elastic part is the frozen
function (structural parity with v1); with every g_i = 0 the overstress is an exact zero and P is byte-identical to the
elastic solver (P_NH has no −0.0 entries, so adding ±0.0 leaves its bytes unchanged; RUBBER-VISC-09 tests it); the
instantaneous response (Q_i ≈ g_iS⁰ at fast loading) is μ₀ and the equilibrium response μ_∞ = g_∞μ₀. Volumetric
viscosity: none (§10).

**21.2.2 Update order (per particle, inside the stress pass of substep n, F = F_n at the start of the substep).**
(1) S⁰_n from F_n; (2) Q_i ← a_iQ_i + b_i(S⁰_n − S⁰_prev) for every term; (3) S⁰_prev ← S⁰_n; (4) P from the
overstress form; (5) with `visco_energy=branch_work_v1`, the branch PK1 `visco_pb` = F_nΣQ_i and the dissipation
increment (21.2.7). The Prony state after a control step therefore belongs to t_k − dt (the last stress pass), as the
stress array does. Reset state (F = I, S⁰_prev = 0, Q = 0) is the stress-free reference; the first stress pass gives
S⁰ = 0 and Q = 0 again. a_i, b_i exactly as §10 (exact for S⁰ piecewise linear between stress passes; τ → ∞ elastic
with μ₀, τ → 0 equilibrium μ_∞).

**21.2.3 Precision: float64 internal state and coefficients (amends §10's float32 storage).** For τ ≫ dt,
1 − a_i = dt/τ_i is below float32 resolution near 1: at dt 5 µs, τ 0.1 s, 1 − a = 5e-5 carries a relative error
≈ 6e-4 in float32 (a 0.06 % error of the relaxation rate), at τ 10 s ≈ 6 %; the per-substep increments b_iΔS⁰ are
≈ 1e-4 of Q and would be lost exactly like the F increments of §19.1 F10. Hence `visco_q` and `visco_s_prev` are
float64, and a_i, b_i (and g_i, τ_i, g_∞) live in a **float64 per-env table**; S⁰ is evaluated in float64 from the
float32 F the elastic part reads (the same input; with F10 the float32 F is the rounded shadow). A float32 storage
variant (`visco_precision = f32_v1`) is allowed later only as an opt-in for throughput with its own declared
equivalence test; it is not part of v1.

**21.2.4 Storage, registry, parameters.**

| Item | Layout | Notes |
|---|---|---|
| `visco_q` | `[P·M]` sym-6 float64 (`vec6d`: xx, yy, zz, yz, xz, xy), particle-major `p·M + i` | registry state, initial `zero` or `snapshot`, multiplicity M |
| `visco_s_prev` | `[P]` sym-6 float64 | registry state, initial `zero` or `snapshot` |
| `params` (float32, existing) | `[N, 6]` μ₀, κ, ρ, frictions | unchanged columns (μ₀ = instantaneous modulus) |
| `params64` (new, float64) | `[N, 1 + 4M]`: g_∞, then (g_i, τ_i, a_i, b_i) per term | declared by the model (`param64_names`); a_i, b_i computed on the host in float64 with `expm1` from the batch dt (`RubberRecipe.prony_coefficients`) |
| per-env energy accumulators (option) | `visco_branch_work[N]`, `visco_dissipation[N]` float64; `visco_pb[P]` mat33d | only with `visco_energy = branch_work_v1` |

M is uniform per batch, 1 ≤ M ≤ 4; an env with fewer terms is padded with g = 0 (Q stays an exact zero; the
dissipation and stored-energy terms skip g = 0). Memory: 48(M + 1) B per particle (144 B at M = 2, 240 B at M = 4;
+72 B with the energy option). The registry (`batched/constitutive_rubber.py`) gets the model
`NEO_HOOKEAN_JBAR_PRONY_V1` (id, version 1, `param_names` (float32), `param64_names`, `per_particle_state =
(visco_q ×M, visco_s_prev)`, `launch_stress` = J-bar pass + `_stress_prony_b`, `launch_guard` = 21.2.6, energy
description, receipt with M and the Prony table); `DECLARED_LATER` loses the entry. The batch takes its model from its
lives (uniform; a mixed batch is `UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS`, a different M `UNSUPPORTED_NONUNIFORM_BATCH`).
`MaterialValues` gains an optional `prony` tuple ((g, τ), …) (default empty: the elastic model); the recipe's
`rubber_generic_visco_v1.json` (provisional, §14.1 example values, `SIMULATOR_ONLY_UNCALIBRATED`) is added.
Randomisation per reset (D10): declared ranges `prony_g_scale` (multiplies every g_i, Σg < 1 enforced) and
`prony_tau_scale` (multiplies every τ_i); the host stages params64 (expm1 in float64) like `stage_params`, without
device allocation or recapture.

**21.2.5 Masked partial reset.** A generic registry-state reset kernel per state variable (zero, or copy from a
per-slot snapshot) runs for the masked envs, plus params64 from its stage; `SOLVER_RESET_ARRAYS` / the
vars()-complete classification list `visco_q`, `visco_s_prev`, `params64` and the option accumulators, so an env
cannot inherit material memory (RUBBER-VISC-06). The single-env `reset()` clears both arrays.

**21.2.6 Stable step: μ₀ plus the overstress (guard `stable_step_guard_visco_v1`).** The step rule uses μ₀ (already so:
`RubberRecipe.mu0 = μ_eq/g_∞`; tested in RUBBER-VISC-08a). The algorithmic tangent of the update is
∂S/∂C = (g_∞ + Σb_i)∂S⁰/∂C with g_∞ + Σb_i ≤ 1, but the geometric term carries the current S = S⁰ + Σ(Q_i − g_iS⁰),
and after a load–unload history the overstress can be positive in a direction where S⁰ is not. Hence, per particle
(float64 inputs, float32 bound as the v1 guard):

```
ρ c_v² = (1 + Σ_i (g_i − b_i)) · W(F, J̄; μ₀, κ) + max(0, λ_max^G(Σ_i (Q_i − g_i S⁰)))
```

W = `rubber_constitutive.wave_modulus_bound` (§18.2), λ_max^G the Gershgorin bound of the symmetric overstress
(max over rows of the diagonal plus the absolute off-diagonals; ≥ the largest eigenvalue). By Weyl, this bounds the
largest eigenvalue of the algorithmic acoustic tensor A_el + (N·ΣΔN)I − Σ(g_i − b_i)M⁰ if the isochoric material part
M⁰ is positive semidefinite; that premise and the bound are verified, not assumed, by the **Prony tangent check**
(RUBBER-VISC-08b: ≥ 2,000 states with history-generated Q, exact algorithmic acoustic tensor by float64 central
differences of the discrete update with Q_n frozen, 512 directions; ratio ≤ 1 everywhere, else a declared v2 of the
guard before any further visco run). Limit 1.0, sticky `RUBBER_CFL_EXCEEDED`, read-only, as §18.2.

**21.2.7 Energy accounting (`visco_energy`: `off_v1` default — no array, no launch — or `branch_work_v1`).**
- Ψ_eq = Σ V g_∞μ₀/2(J^(−2/3)I₁ − 3) + Σ V_c U(J̄): the equilibrium free energy. (The frozen diagnostics' `elastic` uses
  μ₀; a separate fixed-order visco diagnostics kernel reports `psi_eq`, `psi_visc_stored`, the accumulators.)
- **Branch work** W_b = Σ_n Σ_p V (F_nΣQ_i^n):(F_{n+1} − F_n), the discrete work of the internal force on the viscous
  branches (stored `visco_pb` × the ΔF = dt Σ v_I ⊗ ∇N recomputed after the MUSL BC, float64, per env). Since
  P = ∂Ψ_eq/∂F + FΣQ_i, the books **B_N = E_kin + Ψ_eq + E_grav + W_b − W_ext** close at any strain up to the O(dt)
  symplectic error (layer N: the numerics lose nothing).
- **Explicit dissipation** D = Σ_n Σ_p V Σ_i dt(|Q_i^n|² + |Q_i^{n+1}|²)/(4g_iμ₀τ_i) and stored branch energy
  Ψ_v = Σ V Σ|Q_i|²/(4g_iμ₀): each branch is an exact linear Maxwell element in the strain-like variable S⁰/(2μ₀),
  for which these are the stored energy and the dissipation rate; they equal the mechanical branch split at small
  strain (O(ε) otherwise). Over a closed steady cycle E_kin, Ψ_eq and Ψ_v are periodic, so the loss per cycle is
  ΔW_ext = ΔW_b (layer N) = ΔD (layer M, small strain) = πε₀²E″(ω)V (analytic): the material, not the numerics,
  accounts for the measured loss (RUBBER-VISC-05).

**21.2.8 Code organisation.** New: `rubber_visco_constitutive.py` (shared `@wp.func`s: sym-6 helpers, S⁰, Prony
update, overstress PK1, dissipation/stored densities, visco guard; numpy references), `tlmpm_rubber_visco_v1.py`
(`TLMPMRubberVisco(TLMPMRubberCW)`: `step()` = the `tlmpm_rubber_v1` launch sequence with the stress pass replaced by
the J-bar pass + `_stress_prony` when M ≥ 1 and the frozen `_stress` when M = 0, plus the opt-in energy kernel after the
position update; its own `reset`, receipts, `nodal_internal_force` refusal), `rubber_visco_reference.py` (G*, E*,
relaxation/creep by partial fractions, hereditary integral by quadrature, beam root), `configs/materials/rubber_generic_visco_v1.json`.
Changed Phase 2 modules only: `rubber_material.py` (MaterialValues.prony, params64 builder), `live_rubber_material.py`
(model selection), `batched/constitutive_rubber.py`, `rubber_layout.py`, `rubber_batch.py`, `rubber_material_batch.py`.
The visco solver's `step()` duplicates the frozen launch order; RUBBER-VISC-09 (byte identity with M = 0 and with
g = 0) guards the copy.

**21.2.9 Cost (estimate, to be measured in RUBBER-VISC-10).** The stress pass adds ≈ 150 float64 flops (S⁰, C⁻¹) +
≈ 20 per term and 96(M + 1) B of traffic per particle-substep; on the RTX 5090 (float64 at 1/64 rate) this is expected
to cost ≈ +15–30 % per substep at M = 2 (the NH stress is already float64). Rows report M, the energy option and
substeps per control step (D11).

### 21.3 Protocol `rubber_visc_tests_v1` (declared before any implementation; sha256 `95ad1f94…f080`, pinned in `tests/test_rubber_p2_3_protocols.py`)

Common: P2 set μ₀ 0.4 MPa, (g, τ) = (0.2, 5 ms), (0.15, 50 ms), g_∞ 0.65; SLS set (0.3, 10 ms); β 20 on μ₀, ρ 1150,
CFL 0.5, F8 + `kick_average_v1` + F9 + F10, guard 1.0 with 21.2.6, no gravity. Analytic values used for the design:
E′/E″ at 1 / 4 / 8 / 16 / 32 Hz = 0.788/0.058, 0.883/0.115, 0.937/0.116, 0.988/0.127, 1.063/0.134 MPa; shear
retardation times 6.2 and 61.9 ms; cantilever 40 × 4 × 4 mm root ζ 4.743 1/s, f_d 11.71 Hz.

| Id | Test | Reference | Pass |
|---|---|---|---|
| -01 | Relaxation after a one-control-step ramp, block (uniaxial isochoric λ 1.5, shear γ 1), SLS and P2, hold 6τ_max | finite-strain hereditary integral (quadrature, independent of the update); g(t)S⁰ reported | ‖P − P_ref‖/‖P_ref(t_r)‖ ≤ 1e-4 |
| -02 | Creep (σ₀ = 0.002E_∞, cosine ramp 20 ms, hold to 0.4 s) and recovery (to 0.8 s), strip 30 × 6 × 3 mm with traction | Boltzmann superposition of the uniaxial creep compliance (partial fractions) on the actual staircase load | curve within 2 % of max ε; creep ratio within 1 %; recovery within 2 % |
| -03 | Load–unload u = ε₀L(1 − cos 2πft), ε₀ 0.002, fit rates 1 / 4 / 16 Hz, held-out 2 / 8 / 32 Hz; block shear γ₀ 0.15 at 4 / 16 Hz | E*(ω), πε₀²E″V; block hereditary integral | E′ 1 %, E″ 3 %, loss 3 % per rate; 4-parameter fit on the fit rates recovers τ within 10 %, g within 5 %; held-out predictions E′ 1 %, E″ 4 %; single-rate fit reported; block 1e-4 |
| -04 | dt, dt/2, dt/4: block relaxation τ fits; cantilever tap and free decay (n_t 4) | declared τ; complex EB root | τ within 0.5 % across dt and of the declared; ζ 0.5 %, f_d 0.2 % across dt; vs root f_d 3 %, ζ 5 % |
| -05 | Energy balance on every strip/cantilever run; 16 Hz at 2× substeps | layer N / layer M (21.2.7) | \|B_N\| ≤ 0.5 % of S_N; first order (ratio ≤ 0.65); per cycle \|ΔW_ext/ΔW_b − 1\| ≤ 0.5 %, \|ΔD/ΔW_b − 1\| ≤ 2 %, D increments ≥ 0 |
| -06 | Partial reset memory: N = 3 batch, env 1 loaded then reset | fresh batch / fresh single | byte-identical after the reset and for 20 further steps; others unchanged; resampled Prony reset = single built with it; single reset clears |
| -07 | Single vs batched L0 / L1 (N 1, N 3 heterogeneous g, τ, μ₀, ρ, geometry; cases A, C, block) | single references | byte-wise every array every control step; isolation; refusals |
| -08 | Step rule on μ₀; Prony tangent check; guard validity | exact algorithmic acoustic tensor (FD) | ratio ≤ 1 on ≥ 2,000 states; zero violations |
| -09 | Elastic limit | `tlmpm_rubber_v1` | byte-identical with M = 0 and with g = 0 |
| -10 | Cost (reported) | – | D11 columns, SHARED_GPU labels |

Time stamps are part of the protocol (a wrench is a control-step average stamped t_k − Δt_c/2 with the sinc factor;
the stress and Prony state belong to t_k − dt; tractions are constant per control step at the midpoint value), so
that a phase lag of Δt_c/2 (0.02 rad at 32 Hz, 16 % of the loss angle) cannot pass as material behaviour. The
identification in -03 is a synthetic dry run of the P2-5 fit / held-out split (no calibration claim); a pre-declaration
check with 0.3 % synthetic noise gave τ errors ≤ 6 % and held-out errors ≤ 2.8 %, and a one-term single-rate fit
mispredicts E″ by 25–43 % at the held-out rates.

### 21.4 Implementation task list (rcs-impl; S ≈ ½ day, M ≈ 1 day, L ≈ 2–3 days)

CPU tests twice (`CUDA_VISIBLE_DEVICES=''` and `=0`); GPU only under `flock …/rcs_rl_runner/gpu<N>.lock` with ≥ 4 GB
free; raw outputs `artifacts/rl_ready/p2_3/visc_*`; no protocol run before its code is committed clean.

| Task | Deliverable | Tests / evidence | GPU | Size |
|---|---|---|---|---|
| R3.0 | **Done here**: `constraint_work kick_average_v1`, `rubber_moving_grip_tests_v1` PASS | §21.1 | – | – |
| R3.1 | `rubber_visco_constitutive.py`: vec6d helpers, S⁰ (float64), Prony update, overstress PK1, dissipation / stored densities, visco guard, numpy twins | S⁰ = 2∂Ψ_iso/∂C by float64 FD; objectivity (Q in the reference frame); update exact for linear S⁰(t) vs closed form; a, b from expm1 byte-equal to the host; g = 0 → P byte-equal to `nh_jbar_pk1` (random F) | – | M |
| R3.2 | `rubber_material.py`: `MaterialValues.prony`, params64 builder, validation (M ≤ 4, Σg < 1), μ₀ step rule; `rubber_generic_visco_v1.json` | recipe refusals, μ₀ step rule (VISC-08a), params64 values | – | S |
| R3.3 | `rubber_visco_reference.py` | SLS closed forms; E(t) ∗ J_E(t) = t interconversion; creep/relaxation limits; hereditary integral vs closed form for a linear ramp; beam root → elastic limit at g = 0 | – | M |
| R3.4 | `tlmpm_rubber_visco_v1.py` (`TLMPMRubberVisco`), `live_rubber_material` model selection, energy option + visco diagnostics, visco guard | VISC-09 (M = 0 and g = 0 byte identity, single); energy accumulators vs host recompute; single reset clears; guard counts on a forced overstress | – | L |
| R3.5 | `scripts/rubber/p2_3/visco_tangent_check.py` (VISC-08b) | runs before any dynamic visco run; a failure stops and goes to a declared guard v2 | – | S–M |
| R3.6 | Registry model + batched path (`constitutive_rubber`, `rubber_layout` params64/state/model from lives, `rubber_batch` allocation and energy kernel, `rubber_material_batch` generic state reset, params64 staging, Prony-scale resampling) | VISC-06, VISC-07 CPU (byte-wise), vars()-complete with the model state; existing batched suites unchanged | L2 smoke N 1/4/8 (≤ 1.5 GB) | L |
| R3.7 | `scripts/rubber/p2_3/run_visc_v1.py` + evidence writer: -01, -03 block, -04a, -06, -07, -09 (CPU, minutes); -02, -03 strip, -04b, -05 (CPU hours or single-env GPU, recorded) | formal run of `rubber_visc_tests_v1` at a clean commit; summary with every verdict, failures preserved | strip/cantilever optional, ≤ 2 GB | M (+ runs L) |
| R3.8 | VISC-10 cost rows (workload E n_t 4 and the labelled n_t 3 row; elastic vs M = 2, energy off/on) | ms/substep, env-steps/s, VRAM, stage split | ≤ 4 GB, ≤ 30 min | S |
| R3.9 | Report `P2_3_RUBBER_VISCO.md`, ADR results section, evidence JSON; then an xhigh review | – | – | S |

Order: R3.1 → R3.2/R3.3 → R3.4 → R3.5 (gate) → R3.6 → R3.7 → R3.8 → R3.9. `float32` Prony storage, C¹ grip-pose
interpolation, device-side resampling (I2), snapshot (preload) reset of the Prony state and the P2-6 dispatch are
not in this list.

### 21.5 Open items and risks

- **ADR numbering:** `claude/rl-rubber-v5` (protocol v5) may also amend this ADR; whichever merges second renumbers.
- The guard's PSD premise (21.2.6) is checked, not proven; RUBBER-VISC-08b decides before any dynamic visco run.
- Finite-strain energy split: Ψ_v and D are exact for the Maxwell branches in S⁰-strain space; at finite strain they
  differ from the mechanical branch work W_b by O(ε) (reported in -05; layer N is exact).
- The strip tests are long on CPU (≈ 0.7 M substeps for -03); a single-env GPU run is allowed and recorded; GPU L3 for
  visco follows the v4 L3 rule if declared later.
- Float64 throughput on consumer GPUs (21.2.9) may motivate `visco_precision f32_v1`, only with its own declared test.
- `rubber_generic_visco_v1` values are provisional literature-typical magnitudes; no calibration claim.

### 21.6 Results of `rubber_visc_tests_v1` (amendment, 2026-11-10, rcs-impl; details: `docs/reports/rl_ready/P2_3_RUBBER_VISCO.md`)

Implementation R3.1–R3.6 as designed in §21.2 (`efff788`, `fc3d23e`, `555d6fb`); `tlmpm_rubber_v1.py` and
`rubber_constitutive.py` unchanged. Gate R3.5 (`visc_tangent_check_001`, `fc3d23e`): 2400 states, max ratio **0.9910**
→ RUBBER-VISC-08b **PASS**; the PSD premise of the isochoric material part is false (min eigenvalue −1.26e5 Pa),
the bound holds through the elastic W bound's margin (0.9 %). Formal CPU run `visc_v1_001` (`555d6fb`, clean, Warp
1.17.0 CPU, 20 min): **-01, -02, -04, -06, -07, -08, -09 PASS; -03 FAIL; -05 FAIL**; every configuration valid.

- **-03 FAIL** (E′ −1.04 % at 16 Hz, −2.54 % at 32 Hz > 1 %; held-out 32 Hz E′ +1.82 % > 1 %; identification (b),
  loss, E″, block (e) pass). Root cause is the declared error budget, not the material. The finite-strain model form
  is −0.40 % at every rate. Drive-end inertia of the grip-1 reaction is −(3/8)ρω²L²/E = −1.48 % at 32 Hz, not the
  budgeted (f/f_ax)² 0.4 %; it is confirmed by the grip-0 / grip-1 difference, −1.96 % observed vs −1.97 % predicted.
  Fourier leakage of the declared estimator at non-integer samples per period is −0.25 / −0.69 % (16 / 32 Hz).
  Their sum reproduces the observed bias within 0.03 %.
- **-05 FAIL** (cantilever |B_N| 2.58 / 1.05 / 0.47 % of S_N at ×1/×2/×4 > 0.5 %; 1 Hz cycle books +1.17 % > 0.5 %;
  strip/creep books ≤ 0.21 %, (b) and (d) pass). Root cause: the declared left-point branch work
  W_b = Σ pb_n:ΔF_n is not conjugate to the discrete kinetic energy. The kick pairs P_n with (ΔF_{n−1} + ΔF_n)/2, and
  the difference −½Σ ΔQ:ΔF is secular for a Maxwell branch, of relative size ≈ dt·ω²·τ/2 for high-frequency content.
  DIAGNOSTIC `visc_diag_cantilever_books_002` (`0555131`): the kick-paired W_kp = Σ pb_n:(ΔF_{n−1} + ΔF_n)/2 closes the
  same runs to ≤ 0.021 % without drift (elastic control 0.04 %). The claim of §21.2.7 that the left-point W_b closes
  "at any strain up to the O(dt) symplectic error" is withdrawn.
- Thresholds and the v1 protocol are unchanged and the v1 FAILs stay the v1 results. A v2 (estimator / inertia budget
  for -03; `visco_energy = branch_work_kick_average_v2` for -05) must be declared before its run; that decision belongs
  to rcs-xhigh. VISC-10 GPU cost rows: report §4.

### 21.7 Decision on the v1 FAILs of RUBBER-VISC-03 and -05: protocol `rubber_visc_tests_v2` (amendment, 2026-11-11, rcs-xhigh)

Scope: whether a v2 is justified for the two v1 FAILs of §21.6, and what it may change. Rule applied: a v2 may only
correct a measurement or reference defect that would also bias the exact solution; thresholds stay unless the exact
solution provably cannot meet them; the held-out rates stay held out; the v1 FAILs remain the v1 results
(`visc_v1_001`, `p2_3_visco_001.json`). Nothing frozen changes (§21 list; `rubber_visc_tests_v1.json` `95ad1f94…`).
Material status unchanged: `SIMULATOR_ONLY_UNCALIBRATED`.

**21.7.1 Independent verification** (DIAGNOSTIC `scripts/rubber/p2_3/diag_visc_v1_reanalysis.py`, run
`visc_diag_v1_reanalysis_001` at `b9baa57`, clean tree; it re-reads the `visc_v1_001` raw series and the
`visc_diag_cantilever_books_002` summary and runs no simulation). E′ relative errors (grip-1 reaction, P2 strip):

| Rate | v1 declared | estimator alone (exact sinusoids) | exact 1-D inertia, grip 1 | grip 1 − grip 0: observed / predicted | whole periods + de-embedded: E′ / E″ |
|---|---|---|---|---|---|
| 1 Hz | −0.404 % | 0 | −0.002 % | −0.003 / −0.003 % | −0.402 / −0.435 % |
| 2 Hz | −0.410 % | 0 | −0.007 % | −0.010 / −0.010 % | −0.402 / −0.443 % |
| 4 Hz | −0.430 % | 0 | −0.027 % | −0.037 / −0.037 % | −0.403 / −0.467 % |
| 8 Hz | −0.507 % | 0 | −0.103 % | −0.139 / −0.140 % | −0.404 / −0.511 % |
| 16 Hz | −1.042 % | −0.249 % | −0.391 % | −0.528 / −0.530 % | −0.405 / −0.508 % |
| 32 Hz | −2.544 % | −0.687 % | −1.453 % | −1.968 / −1.971 % | −0.414 / −0.750 % |

- *Estimator leakage: confirmed.* With 312.5 / 156.25 samples per period the last-"full"-cycle window has 313 / 157
  samples and the σ mean is not removed. On exact steady-state samples on the protocol grid the declared estimator alone
  gives E′ −0.249 / −0.687 %, E″ −0.303 / −0.823 % (16 / 32 Hz). A window of K whole periods with an integer sample count
  (K = 2 at 16 Hz, 4 at 32 Hz: 625 samples) with σ and ε mean-removed gives ≤ 7e-16. Including the start-up
  transient (exact linear response from rest), a whole-period window that starts after 6 τ_max gives ≤ 1e-5 (E′),
  ≤ 1.9e-4 (E″) at every rate. With v1's run lengths that window would start at 5 τ_max / 4.4 τ_max and carry +0.13 % in E″.
- *Drive-end inertia: confirmed, with a corrected factor.* The exact steady state of the 1-D linear viscoelastic rod
  (u(0) = 0, u(L) = U e^{iωt}) with the actual marker layers gives σ(L)/ε_g = E*·k(b − a)cos kL/(sin kb − sin ka),
  k = ω(ρ/E*)^½. The markers are the nearest particle layers a = 7.25 mm, b = 22.25 mm, not L/4 and 3L/4.
  At 32 Hz this is −1.453 %. The implementer's −(3/8)ρω²L²/E = −1.476 % is the mid-span value. The grip-1 − grip-0
  difference, −1.968 % observed vs −1.971 % predicted (−0.528 / −0.530 % at 16 Hz), validates the rod model on the
  measured data. Grip 0 de-embedded with σ(0)/ε_g = E*·k(b − a)/(sin kb − sin ka) agrees with grip 1 within 0.004 %.
- *After both corrections* E′ is −0.402 … −0.414 % at every rate: the finite-strain model form the v1 rationale budgets
  ("about 0.4 %"); the implementer's three-part sum holds (residual ≤ 0.014 % with the exact factor). The protocol's
  identification on the corrected data (DIAGNOSTIC) gives τ₁ −3.2 %, g₁ +1.2 %, τ₂ −0.24 %, g₂ −0.02 % and held-out 2 / 8 /
  32 Hz E′ +0.03 / +0.02 / −0.08 %, E″ +0.03 / −0.20 / +1.83 %: the held-out 32 Hz FAIL has the same cause.
- *Not explained (reported, not acted on):* the de-embedded E″ at 32 Hz (−0.75 %) is lower than at 1–16 Hz
  (−0.44 … −0.51 %), while the energy-based loss is −0.36 … −0.41 % at every rate. The forces of a control step are
  sampled at t_{k−1} + j·dt, j = 0 … n − 1, so the average is centred dt/2 before the declared stamp t_k − Δt_c/2.
  That would shift E″ by +(ω dt/2)/tan δ = +0.02 … +0.46 %, which matches 1–8 Hz but over-corrects 16/32 Hz. The stamp
  is not changed (E″ limit 3 %; no failure depends on it).
- *Energy books: confirmed.* Symplectic Euler gives KE_{n+1} − KE_n = f_n·(Δu_{n−1} + Δu_n)/2 exactly, so the work
  of the branch force conjugate to the discrete KE is W_kp = Σ pb_n:(ΔF_{n−1} + ΔF_n)/2. Reindexed, this is
  Σ ½(pb_n + pb_{n+1}):ΔF_n, the trapezoid over the interval each Prony update spans. The declared left-point W_b
  differs by −½Σ(pb_{n+1} − pb_n):ΔF_n plus boundary terms. For Ψ_eq the same term averages out against Ψ's
  second-order term (a bounded shadow-energy oscillation). For a Maxwell branch at ωτ ≫ 1, Δpb ≈ g_i·(tangent)·ΔF,
  so the term is one-signed, ≈ ½dt·k_b⟨v²⟩ per substep, a ratio ≈ ½ω²τ·dt to the branch dissipation. The drift
  therefore grows with τ and halves with dt (2.58 → 1.05 → 0.47 %).
  *Independent check:* in every diag_002 variant, W_kp equals the branch's own energy D + Ψ_v (computed from Q only)
  to ≤ 1e-5 of S_N. The left-point W_b misses it by exactly its book drift (P2 −2.09 % vs B_N 2.08 %, τ 50 ms −5.70 %
  vs 5.69 %). The left-point sum is therefore a first-order quadrature that would bias an exact branch history too.
  The v1 budget "O(ω dt) ~ 1e-3 of the loss" holds only for the drive frequency, not for the high-frequency content
  the tap and the per-control-step velocity jumps excite. The explicit dissipation D and the dynamics are correct.

**21.7.2 Decision per test.**

- **RUBBER-VISC-03: v2 justified.** Two measurement defects bias the exact solution as much as the simulation:
  (i) the estimator's non-whole window (fixed by K whole periods with an integer number of samples, σ and ε
  mean-removed, the window starting after 6 τ_max; the 16 / 32 Hz runs become 2500 control steps, the 1–8 Hz runs keep
  their v1 length); (ii) the drive-end reaction is not the gauge stress (fixed by de-embedding the measured modulus with
  the exact 1-D rod transfer above).
  The de-embedding uses only ρ, L, the scene's marker reference positions and ω: no Prony parameter and no reference
  value. It is applied identically to every rate, so the fit (1 / 4 / 16 Hz) and the held-out rates (2 / 8 / 32 Hz)
  stay separate. A new *validity* condition guards it: grip 0 and grip 1 de-embedded must agree in E′ within 0.2 %
  (one fifth of e1_rel; observed ≤ 0.004 % on v1 data). The reference E*(ω), the time stamps, the finite-strain
  model-form budget (−0.40 %, declared in v1 and attainable) and every threshold are unchanged.
  Rejected: widening e1_rel (forbidden, and unnecessary: the corrected bias is 0.41 % of a 1 % limit); the
  approximate (3/8)π² budget (approximate, and the wrong lever: a budget does not remove a bias); scoring grip 0
  (still +0.51 % biased at 32 Hz); new drive rates or a different gauge (changes the test).
- **RUBBER-VISC-05: v2 justified.** The declared branch-work quadrature is a measurement defect; the method is fixed,
  not the limit. New opt-in option `visco_energy = branch_work_kick_average_v2`: W_b = Σ_n Σ_p V pb_n:(ΔF_{n−1} + ΔF_n)/2,
  ΔF_{−1} = 0 after a reset, one float64 `visco_df_prev[P]` (mat33d) per particle. The v1 left-point sum is accumulated
  alongside in `visco_branch_work_left` (reported). It is bookkeeping only: the stress pass, D, `visco_pb` and every
  state array are byte-identical to `branch_work_v1` and `off_v1`, and `off_v1` / `branch_work_v1` are unchanged
  (no array, no launch added). The cycle criteria (c), (d) use the v2 whole-period window: over a non-whole window
  Ψ_v, Ψ_eq and E_kin are not periodic (ΔΨ_v was 0.5 % of the 32 Hz loss in v1). Thresholds unchanged.
- **No threshold changes.** Neither failure shows a threshold that the exact solution cannot meet.

**21.7.3 Protocol `configs/rubber/rubber_visc_tests_v2.json`** (declared and SHA-pinned in
`tests/test_rubber_p2_3_protocols.py` in its own commit, before the option is implemented and before any v2 run).
Scope: re-scores RUBBER-VISC-03 and -05 only. -01, -02, -04, -06 … -10 keep their v1 definitions and `visc_v1_001`
results. The -02 creep and -04b cantilever runs are re-run because -05 books them; their -02 / -04b numbers are
reported as a consistency check. Every strip, creep, cantilever and block run whose schedule (or prefix) equals a v1
run must reproduce every recorded v1 column except `branch_work` byte-identically, and its left-point W_b must equal
the v1 `branch_work` column. Otherwise the v2 run is INVALID: an energy option must not move the dynamics.

**21.7.4 Tasks (rcs-xhigh, this branch).** (1) The option in `TLMPMRubberVisco` and the batched path (allocation,
launch, snapshot, masked reset, vars()-complete tables, uniform-signature refusal of mixed v1/v2), default bitwise-off.
(2) Tests: dynamics byte identity (off_v1 / branch_work_v1 / v2, single and batched); left-point accumulator = v1 W_b
bytes; kick-paired accumulator = host float64 recompute every substep; L0 N = 1 / N = 3; reset; refusal.
(3) Declared estimator and de-embedding in `rubber_visco_reference.py` (tests: exact synthetic signals, the
transfer's small-kL expansion and E* round trip). (4) Formal runner `run_visc_v2.py` (CPU), smoke labelled SMOKE.
(5) Formal run, evidence `p2_3_visco_002.json`, report section, results below.

**21.7.5 What v2 does not establish.** No calibration (synthetic Prony sets). The finite-strain model form stays
−0.4 % against the small-strain E*. The 32 Hz E″ phase residual (21.7.1) stays open. No GPU strip/cantilever run.

**21.7.6 Results of `rubber_visc_tests_v2`** (formal run `visc_v2_001`, commit `889a782`, clean tree, Warp 1.17.0 CPU,
16 workers, 00:24:25–00:44:57 KST; evidence `docs/evidence/rl_ready/p2_3_visco_002.json`; details: report §8).

- **Validity `dynamics_identity_with_v1`: PASS.** All 14 jobs reproduce every recorded `visc_v1_001` column except
  `branch_work` byte-for-byte (full length, or the v1 prefix of the longer 16 / 32 Hz runs). Their left-point W_b equals
  the v1 `branch_work` byte-for-byte: the energy option moves no dynamics. 0 invalid flags, 0 guard violations, 0
  negative D increments.
- **RUBBER-VISC-03: PASS.** De-embedded E′ is −0.402, −0.402, −0.403, −0.404, −0.401, −0.414 % (1 … 32 Hz; limit 1 %).
  E″ is −0.435 … −0.867 % (3 %), the loss per cycle −0.397 … −0.434 % (3 %), and grip 0 vs grip 1 ≤ 0.003 % (0.2 %).
  The fit on 1 / 4 / 16 Hz gives τ₁ −3.4 %, τ₂ −0.29 %, g₁ +1.1 %, g₂ +0.07 %. Held-out 2 / 8 / 32 Hz: E′ +0.04 /
  +0.03 / −0.10 % (1 %), E″ +0.04 / −0.26 / +1.90 % (4 %). Block 1.3e-7 / 1.4e-7. On the same series, without
  de-embedding the 32 Hz E′ is −1.87 %, and with the v1 estimator −2.54 % (= v1).
- **RUBBER-VISC-05: PASS.** max |B_N|/S_N is 0.002 … 0.007 % on the strips, 0.035 % on the creep run and
  0.012 / 0.006 / 0.003 % on the cantilever at ×1 / ×2 / ×4. The v1 left-point books on the same runs are
  2.576 / 1.054 / 0.474 % (= v1). 16 Hz ×2 / ×1 = 0.48. Cycle books are within ±0.008 % (left-point at 1 Hz: +1.166 %
  = v1). ΔD/ΔW_b − 1 is −0.39 … −0.41 %. The g = 0 control is 0.020 %.
- **Observation (DIAGNOSTIC, after the run, not scored).** E″ falls with rate while E′ and the loss do not. The
  control-step wrench average is centred dt/2 before the declared stamp; re-stamped, the same data give E″ −0.39 …
  −0.42 % at every rate, equal to E′ and the loss. The stamp is kept (no change after a result); a later protocol may
  adopt it.
- CPU suites (rubber + batched, commit `bc52a2d`): 397 passed, 0 skipped with `CUDA_VISIBLE_DEVICES=''` and with `=0`.
  Material status: `SIMULATOR_ONLY_UNCALIBRATED`. The v1 FAILs remain the v1 results. Independent rcs-review: NOT_RUN.

## 22. P2-4: contact, friction and dynamic coupling (amendment, 2026-11-11, rcs-xhigh)

Scope: the five P2-4 decisions (rubber–table / fixture contact, gripper pads, self-contact, the 20 ms delayed handoff,
GPU per-env isolation), the protocol `rubber_contact_tests_v1` (declared and SHA-pinned before any formal run) and the
implementation of the single-env contact law. Branch `claude/rl-rubber-p24` from `025ecaf`. Nothing frozen changed:
v1–v7_2, law 8, evaluators, every Phase 1 module (`batched/constitutive.py`, `batched/errors.py`,
`batched/particle_contacts.py`, `batched/material_batch.py`, …), `configs/rl_benchmark/*`, `scripts/rl_ready/*`, the
rubber protocols v1–v4 / moving-grip v1 / visc v1–v2, `tlmpm_rubber_v1.py`, `rubber_constitutive.py`,
`tlmpm_rubber_visco_v1.py`, `rubber_visco_constitutive.py`, `rubber_constraint_work.py` (byte-identical). Changed
Phase 2 file: `live_rubber_material.py` (opt-in `contact=` dispatch, `advance(frames=…)`, `last_contact`; without a
ContactSpec it builds exactly the previous classes). New: `rubber_contact.py`, `tlmpm_rubber_contact_v1.py`. Raw outputs
`artifacts/rl_ready/p2_4/`; report `docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md`; evidence
`docs/evidence/rl_ready/p2_4_contact_001.json`. Material status: `SIMULATOR_ONLY_UNCALIBRATED`.

### 22.1 Decision 1 — rubber–table / fixture contact: grid-node penalty with a Coulomb spring–dashpot–slider (`grid_node_penalty_coulomb_v1`), work booked kick-paired

**Verified facts.** (1) The frozen helper (`tlmpm_v5.py:155-165`, used by `tlmpm_rubber_v1._loads` and the Phase 1
pads) is a predictive penalty with an approach-only damper — normal impulse `s m/dt · max(−gap − dt·min(v_n,0), 0)`, i.e.
spring `s m/dt²` plus damper `s m/dt` while approaching — and a friction impulse capped at `m|v_t|` of the particle
velocity at the START of the substep, applied before P2G; its impulses are in no work term. At rest it applies no
friction, so a body at rest on a slope is never a fixed point. (2) Pure FLIP + MUSL with fixed weights is symplectic
Euler in (π, u) with KE = ½πᵀBπ, B = M_l⁻¹M_cM_l⁻¹ (§18.1 F1, §21.1): the drift velocity is Bπ, not the lumped P2G
velocity M_l⁻¹π. A lumped node projection therefore under-corrects the drift (a flat face keeps the B-row fraction it
does not see: the face rows of B sum to 0.625 of the lumped inverse mass at 2 particles per cell), and an overwrite
after the MUSL pass is not symplectic (§18.1 F6, §21.1). (3) Any impulse J that enters the kick (added to the P2G grid
momentum, never overwritten after MUSL) satisfies KE_{n+1} − KE_n = (dt f + J)·(g_n + g_{n+1})/2 exactly.

**Probe evidence** (DIAGNOSTIC; `scripts/rubber/p2_4/probe_contact_laws.py`, run `probe_contact_laws_004` at the clean
commit `259f2b6`; runs `_001`–`_003` and the development sweeps `contact_sweeps_dev_001`
(`diag_contact_sweep.py`) preserved). Block 12 × 6 × 3 mm, n_t 3, μ_table 0.5, gravity ramped per substep over 20 ms,
displacement of the centroid along the slope over the 100 ms hold; slip error = (a_fit − g(sinθ − μcosθ))/(g cosθ);
frictionless rod 30 × 4 × 4 mm, n_t 4, 0.2 m/s end impact:

| Law | stick 0.8 μ / 0.9 μ / 0.95 μ | slip error 1.05 / 1.15 / 1.25 | impact E_end/E0 (CFL 0.5 / 0.25) | verdict |
|---|---|---|---|---|
| frozen per-particle helper (D8 design) | 0.50 / 1.24 / 1.59 mm | +2.0 / +0.8 / +0.4 % | 0.966 / 0.950 (loss grows as dt shrinks) | rejected |
| node velocity projection `grid_node_coulomb_v1` | 0.34 / 0.75 / 1.15 mm | +1.2 / +0.3 / +0.1 % | 0.993 / 0.972 (dt-dependent) | rejected |
| node penalty, no dashpots (sweeps, `_003`) | creeps at 0.8 μ in most settings (4e-5 … 7e-4 m; one κ 0.05 setting held) | +1.4 … +3.3 % | 1.0015 / 1.0008 (conservative) | rejected |
| node penalty, normal dashpot only (`_004` "undamped") | 2.7e-9 / 7.3e-9 m / 1.3 mm | +4.2 / +4.3 / +4.3 % | — | rejected |
| node penalty, tangential dashpot only (sweep) | 1.3e-4 m at 0.8 μ | +1.3 … +9 % | — | rejected |
| **node penalty + both dashpots ζ 0.5 (adopted)** | **1.0e-6 / 8.3e-7 m** / 0.71 mm | **+0.53 / +0.15 / +0.10 %** | **0.962 / 0.960 (dt-independent, booked)** | **adopted** |

Reading. Every explicit Coulomb law tried shows a friction-induced flutter of the elastic block: the contact layer
rattles (only 8–20 % of the bottom nodes in contact per substep), stick creeps and slip loses friction. It is not
removed by refinement (n_t 4 creeps more than n_t 3; CFL 0.25 still creeps) and appears for all three formulations, so
it is a property of explicit Coulomb friction on an elastic model with almost no high-frequency damping (the coupling
∂F_t/∂u_n = μ k_n makes the contact stiffness non-symmetric), not of one implementation. The declared contact dashpots,
acting on the node's predicted (post-force) relative velocity, suppress it; both are needed. Real rubber has strong
high-frequency loss, which the illustrative Prony sets lack (the P2 block holds at 0.95 μ, the elastic block does not).

**Law.** Per reference-grid node with mass (TLMPM nodes are the Q1 material points; on the particle lattice the outer
nodes lie on the faces), x_I = X_I + u_I (float64, advanced with the drift velocity): every obstacle the node penetrates
(gap < 0 at the start of the substep) gives F_n = max(k_n(−gap) − c_n v_n, 0) n, k_n = κ_n m_I/dt², c_n = 2ζ_n√(k_n m_I);
one Coulomb spring–dashpot–slider on the deepest obstacle, F_t = −k_t ξ_t − c_t v_t capped at μF_n (slip: the anchor
slides, ξ_t = −F_t/k_t; the Coulomb dissipation μF_n|Δanchor| is booked), k_t = κ_t m_I/dt², c_t = 2ζ_t√(k_t m_I);
ξ_t += dt(v_I − v_obs) after the drift; a node leaving contact or switching obstacle releases its anchor; nodes of an
active Dirichlet group are skipped. Declared numerical parameters κ_n = κ_t = 0.25, ζ_n = ζ_t = 0.5
(2ζ√κ = 0.5 ≤ 1). Obstacles: P static half-spaces (friction_table) and F rigid kinematic frames with B oriented boxes
each (kind fixture → friction_table, pad → friction_pad), moving x = R(s)X + b(s) with the BC-group interpolation;
the obstacle velocity at the node is position-exact. Applied in the kick after the P2G grid update.

**Work (`kick_average_v1`).** W_n, W_t = Σ J·(g_n + g_{n+1})/2 with g_n = `old` after the P2G grid update and
g_{n+1} = `old` after the final MUSL grid update (two half launches, no per-particle state); obstacle power
P = Σ J·v_obs; contact dissipation D = (P_n − W_n) + (P_t − W_t). Books (layer N):
B = E_kin + Ψ_eq + E_grav + W_b − W_ext − (W_n + W_t). Contact numerics or dashpots are never booked as material
damping.

**Consequences, declared.** (a) Impacts lose energy to the contact dashpots (3.8–4.0 % of E0 for the 30 mm rod at n_t 4),
booked as contact dissipation and dt-independent; the conservative (undamped) normal law returns it but flutters in
friction. (b) At a static fixed point the dashpot sees v* = −dt F_I/m_I and carries the share 2ζ_n√κ_n = 0.5 of the
load without dissipating: penetration δ_I = (1 − 2ζ_n√κ_n) F_I/k_n ∝ dt²/dx (→ 0 under refinement at fixed CFL).
(c) Near-threshold stick (0.95 μ) of the purely elastic model still creeps (domain limit, reported; the P2 material
holds). (d) GPU cost +40–46 % per substep on the workload E single env (§22.7). (e) Coulomb friction is velocity- and
pressure-independent here; real rubber friction is not (calibration is P2-5).

### 22.2 Decision 2 — gripper pads / robot fingers: moving kinematic box frames, per-pad wrench [N, G, B, 6]

- B oriented boxes per kinematic contact frame of kind `pad` (friction_pad), rigid motion interpolated per substep
  exactly like the BC groups (slerp + linear); each box's reaction (impulse on the material and its torque about the
  frame's current origin) is accumulated per control step; the wrench the material exerts on box b of frame f is
  `−reaction/control_dt`: single env [F, B, 6] (`TLMPMRubberContact.contact_wrenches`, `LiveRubberMaterial.last_contact`),
  batched [N, G, B, 6] with G = pad frames per env (uniform), B boxes per frame (uniform).
- Coupling without changing Phase 1 files: the rubber batched material will expose the `MaterialCoupling` surface of
  `rl/lab_batched_env.py` (`pending_wrench [N,6]`, `pad_wrench [N,B',6]` world wrenches, B' = G·B rows, recipients
  mapped to the finger bodies as in `configure_pad_obstacles`), written by a NEW rubber wrench kernel at the end of
  `advance` (strip-local → world with a per-env strip pose, float64, as `_wrench_b`), and the Lab env applies them in
  every substep of the next interval exactly as for the cutting material (`lab_batched_env.py:15, 65-67`). Nothing in
  `material_batch.py` or `lab_batched_env.py` changes.
- `kinematic_clamp_v1` (Dirichlet group, F8 + kick-average work) stays for clamped grips; `friction_pads_v1` = pads only.
  Robot links other than the pads do not touch the strip in v1 (declared, as D8).

### 22.3 Decision 3 — self-contact: not part of the P2-4 acceptance; `NOT_IMPLEMENTED_V1` (declared limitation)

- The P2-4 scenes (table, fixtures, pads, pinch, coupling) do not fold. The only fold scene so far, the λ 1.5
  instantaneous release, reached hinge stretches of 0.05 and a guard bound of 13 at CFL 0.5 before any self-contact
  (§18.1 F4): a fold needs r ≥ 2 … 13, self-contact and probably HF damping — a research item, not a P2-4 gate.
  RUBBER-PHYS-12 already flags compressive folds (guard / J domain).
- Gap that remains: two distant parts of a strip can pass through each other without compression (no flag). Before P2-6
  a per-control-step self-penetration detector is required (flag `RUBBER_SELF_CONTACT_UNRESOLVED`, terminate in
  training; task S1 below).
- Options and cost when self-contact is needed: (a) **particle-pair contact in a new rubber module reusing the grouped
  HashGrid approach** (Phase 1 `particle_contacts.FallbackGrid` pattern, graph-safe, one device HashGrid per process;
  pairs with reference distance > 3 dx, no cut codes; a dashpot-capped pair law since a per-pair slider history is not
  feasible): Phase 1 measured ≈ 73 µs hash build + ≈ 110 µs alias-free contact kernel per substep at 103k particles
  (N = 1, P1-3a) → ≈ +0.18 ms on a 0.22–0.32 ms rubber substep (+55–80 %), ≈ +25–40 % with surface particles only
  (≈ 30 % at workload E); (b) **grid multi-velocity field**: impossible on the TLMPM reference grid (contacting regions
  are far apart in the reference configuration); it needs a second current-configuration sparse grid, region labels
  and an extra P2G/G2P per substep (≈ +80–120 %) — rejected; (c) **detector only, once per control step**: one hash
  build + query per 2,600 substeps (negligible) — recommended now.

### 22.4 Decision 4 — the 20 ms delayed force handoff

- Semantics (`lab_batched_env.py:15`): material step k follows the pose measured at the end of rigid interval k
  (linear within it); its interval-averaged wrench is applied in every substep of interval k + 1 — an effective delay
  of 1–1.5 intervals on a zero-order hold.
- Ideal-tracking / kinematic grips (every P2-1 … P2-4 test, every calibration experiment, the P2-6
  `kinematic_grip_batched` backend) have no wrench feedback, hence no coupling error by construction.
- With feedback the delay is assessed by RUBBER-CONT-07 (1-DOF hand surrogate m_h z'' = k_h(z_c − z) + c_h(ż_c − ż) +
  F_strip(delayed); 20 ms vs 1 ms intervals at the same material dt; envelope strip length 30 / 60 mm, stiff
  1 kg / 10 kN/m and light compliant 0.05 kg / 100 N/m hands, 40 / 200 ms moves). Rows outside the acceptable envelope
  carry `coupling_delay_significant`. The delay energy error E_delay = Σ_k (F_applied,k − F_material,k)·Δz_k is computed
  on the hand side; the material books close independently of it; no material parameter is ever fitted to it.

### 22.5 Decision 5 — GPU per-env wrench and state isolation

- Kernel design: one thread per node, env e = `env_of_node[i]`; all writes to the node's own velocity, impulses and
  slider state; atomics only into rows owned by env e (`e·P + k`, `(e·F + f)·B + b`, `work[e, ·]`, `pen_max[e]`,
  `counts[e, ·]`); obstacle data read from env e's rows only. The contact kernel itself is deterministic (no atomics on
  velocities); its reaction / work atomics and the frozen P2G/MUSL atomics are order-nondeterministic on a GPU.
- Tests: RUBBER-CONT-08 (CPU L0 byte-wise, L1 perturbation and masked reset with the contact arrays) and -09 (GPU:
  exact-zero rows for envs out of reach, perturbation band vs repeat band, audits). Single-env GPU evidence now:
  out-of-reach rows exactly 0.0 (§22.7). The batched contact path is task B1 below.

### 22.6 Protocol `rubber_contact_tests_v1`

`configs/rubber/rubber_contact_tests_v1.json`, sha256 `4d3dc68fc50aeeb247c2b544c65312fd2a651a90a99eb394b11f3fa478b46e4c`,
pinned in `tests/test_rubber_p2_4_protocols.py`, committed at `51023d6` before any formal run (the law at `259f2b6`).
RUBBER-CONT-01 incline stick / slip / onset (rigid Coulomb), -02 penetration vs dt / dx (the law's fixed-point value),
-03 frictionless rod impact (2L/c0, e = 1 elastic; books with contact work, elastic energy and Prony branch work
separated; dissipation attribution and convergence), -04 interface dissipation = μN·slip, -05 pinch-and-pull (per-pad
wrenches, slip ratio vs grip force), -06 self-contact NOT_IMPLEMENTED_V1, -07 coupling-delay assessment, -08 batched
L0/L1, -09 GPU isolation, -10 contact-off identity and law checks (unit tests), -11 cost (reported). The scenes seen in
the probes are listed in its provenance; every formal scene is different (geometry, μ, speed, rod length).

### 22.7 Results (formal CPU run `contact_v1_001`, GPU check `gpu_contact_002`; details: `docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md`)

Formal run: commit `436f91c` (clean tree), protocol sha256 `4d3dc68f…6e4c`, Warp 1.17.0 CPU, 32 workers on the shared
Threadripper, 41 jobs. Every FAIL below is the v1 result and stays preserved; no threshold moves.

| Test | Verdict | Measured |
|---|---|---|
| -01 incline | **FAIL** | elastic stick rows PASS at μ 0.4 / 0.8 and ratios 0.6 / 0.85 (hold displacement ≤ 0.0044 s, a ≈ 0); μ 0.4 slip PASS (+0.20 / +0.06 / +0.03 % of g cosθ), μ_eff(0.4) −0.67 % PASS; **μ 0.8 slip +1.65 / +1.88 / +1.06 % > 1 %, μ_eff(0.8) −2.75 % > 2 %, the P2 stick row at 0.95 μ, μ 0.8 slides (4.8 s, +1.8 % g cosθ), books of the μ 0.4 / 0.6 stick run 1.03 % > 1 %** |
| -02 penetration | **PASS** | max penetration ≤ 0.049 s; dt ratio 0.256 (law 0.25); dx ratio 0.490 (law 0.5); undamped spring sum / reaction 1.000002, interior node ratio 1.050; balance ≤ 4e-5 |
| -03 impact | **FAIL** (books, one run) | n_t 4 books 0.59 / 0.25 % (CFL 0.5 / 0.25, ratio 0.42), P2 0.54 / 0.20 %; **elastic n_t 2 1.024 % > 1 %**; contact time 2.58 ms vs 2L/c0 2.50 ms (+3.3 %); contact dissipation 2.86 % (n_t 4) / 4.71 % (n_t 2) of E0, ratio 0.61; P2 branch dissipation 10.2 % of E0; rebound v/V 0.94 (elastic), 0.90 (P2) |
| -04 dissipation | **FAIL** | μ 0.4: 0.988 / 0.993 / 0.995 of μ∫N dx (PASS); **μ 0.8: 0.932 / 0.930 / 0.943 < 0.95** |
| -05 pinch-and-pull | **FAIL** | N = 0.67 / 1.40 / 2.15 N; **slip ratio 0.972 / 0.964 / 0.956 (band 0.97–1.02)**; cone ≤ 0.9995 PASS; grip scaling spread 1.6 % PASS; **clamp balance 9.8 / 6.0 / 6.3 % > 5 %**; penetration PASS |
| -06 self-contact | NOT_RUN | NOT_IMPLEMENTED_V1 (declared) |
| -07 coupling delay | **PASS** (assessment complete) | all 8 envelope points **coupling_delay_significant**: tip error 0.3–13 %, hand-side delay energy 4–93 % of the strip's mechanical energy; log-decrement tip damping with the 20 ms handoff negative at 6 of 8 points (−0.0006 … −0.030), the 1 ms arm resolved at 4 points (+0.0003, +0.0003, +0.0017, −0.030; the elastic strip has no damping, so ~1e-3 is the estimator's noise level); material books ≤ 0.10 % in every arm |
| -08 / -09 batched, GPU isolation | NOT_RUN | batched contact path not implemented (task B1) |
| -10 unit tests | PASS | `tests/test_rubber_contact.py` 12 / 12; CPU suites (rubber + batched + naming, `045ce6c`): 421 passed with `CUDA_VISIBLE_DEVICES=''` and with `=0` |
| -11 cost (GPU0, SHARED_GPU, driver 570.211.01) | reported | workload E single env (41,220 particles, n_t 3, 2,585 substeps): 0.218 → 0.320 ms per substep with a table and two pads (+46 %; +40 % out of reach); VRAM 686 → 744 MB; contact kernel 0.076 ms (22 % of kernel time: per-node float64 slerp / quat-to-matrix of the frames); out-of-reach contact rows exactly 0.0; GPU vs CPU pad wrench 3.3e-6 relative |

Reading (diagnostic, after the run). The μ 0.8 failures share one cause with the decision probe's near-threshold
creep: the friction-induced flutter grows with μ (its coupling is μ k_n) and the declared ζ = 0.5 suppresses it at
μ ≤ 0.5 but not fully at 0.8 (friction ≈ 2 % low, the extra energy goes into contact vibration). The pinch ratio
(0.96–0.97) and the 6–10 % clamp imbalance come from the same contact vibration under the pads plus lateral slip
components on the bulging strip; the books FAILs are 1.02–1.03 % on the two runs with the coarsest dt / smallest energy
scale. -07 is the decisive coupling finding: with wrench feedback the 20 ms handoff's energy error is comparable to the
strip's mechanical energy across the whole declared envelope (and fast moves have 3–13 % tip errors), so
feedback-coupled rubber rows must carry `coupling_delay_significant` (or run a shorter handoff), while ideal-tracking
grips remain exact.

### 22.8 Remaining tasks (rcs-impl unless stated)

| Task | Content | Size |
|---|---|---|
| X1 (rcs-xhigh) | Decision on the -01 / -04 / -05 FAILs at μ 0.8 / pinch: a v2 law (e.g. dashpot ratio scaled with μ, a frame-table contact kernel, or HF damping in the material set) or a declared μ envelope; any v2 is declared before its run; the v1 FAILs stay | S |
| B1 | Batched contact: `BatchedTLMPMRubberContact` (contact arrays packed from the single references, global node / obstacle rows, the same kernels), `BatchedRubberMaterial` contact subclass (frame motions per env, per-env frictions from `params` columns 3/4, masked reset of `ct_node_u / ct_xi / ct_obst / ct_vobs / ct_work / reactions`, vars()-complete tables, wrench kernel → [N, G, B, 6]), `rubber_layout._validate` accepting `TLMPMRubberContact` with the contact spec in the uniform signature; tests RUBBER-CONT-08 (L0 byte-wise N 1 / N 3 heterogeneous, L1, refusals) | L |
| B2 | RUBBER-CONT-09 on GPU0 (exact-zero rows, perturbation band, audits) after B1 | S |
| B3 | MaterialCoupling adapter for the Lab env: rubber wrench kernel → `pending_wrench [N,6]`, `pad_wrench [N,G·B,6]` world wrenches with recipients (no Phase 1 file changed) | M |
| P1 | Contact cost: per-substep frame table (as `motion_table_v1`) and a bounding-volume reject in `node_penalty_contact` (bytes unchanged on CPU, tested) | M |
| S1 | Self-penetration detector once per control step (HashGrid / FallbackGrid over surface particles, reference distance > 3 dx), sticky flag `RUBBER_SELF_CONTACT_UNRESOLVED`, P2-6 terminates | M |
| C1 | Feedback coupling: a shorter handoff (≤ 1–2 ms) or an iterated / implicit material–robot exchange for Lab-coupled rubber; re-run RUBBER-CONT-07 envelope under a new declared protocol | M–L |
| R1 | Independent rcs-review (xhigh) of §22 and `contact_v1_001` | – |

### 22.9 Task X1 — decision on the v1 contact FAILs: law `grid_node_penalty_coulomb_v2` + declared μ envelope (amendment, 2026-11-11, rcs-xhigh)

Scope: root cause of the μ-dependent FAILs of `contact_v1_001` (-01 slip / onset / P2 stick / books, -03 n_t 2 books,
-04 dissipation, -05 slip ratio / clamp balance) and of the -09 b FAIL of `gpu_cont09_001` (B1), the decision, the
protocol `rubber_contact_tests_v2` (sha256 `89bc7680bb9c2ed6322279b198899cef0faf31c4751b6bd849b22ecd24a339d0`, declared
alone at `8842573` before any run of it), and its formal CPU run `contact_v2_001`. Every v1 result and FAIL stays.
Nothing frozen changed: the v1 kernels are byte-identical (source hashes pinned in `tests/test_rubber_contact_v2.py`;
four `contact_v1_001` jobs rerun after the change are byte-identical, `x1_v1_identity_002`), v1 stays the default law,
`rubber_contact_tests_v1` and every frozen module / protocol / Phase 1 file are untouched. Material status:
`SIMULATOR_ONLY_UNCALIBRATED`.

**Root cause (DIAGNOSTIC: `scripts/rubber/p2_4/diag_contact_x1.py`, `x1_dev_laws.py`, `diag_x1_scenes.py`; raw
`artifacts/rl_ready/p2_4/x1_diag_001`).** The per-substep instrumentation of the INC block (per-node impulses) shows that
the friction actually applied is 97.9 % of μ ΣJ_n at μ 0.8 (stick nodes cost 0.2 % of μΣJ_n, slip misalignment
1.9 %: lateral / axial friction 0.15) and that only 3 % of the bottom nodes are in contact per substep (normal
impulse std/mean 5.8): the face rattles. Two defects:

1. *The v1 dashpots act on the wrong velocity.* They use the predicted lumped velocity `new_v` = M_l⁻¹π_n + dt f/m_I,
   while the node moves with the drift velocity Bπ (§18.1 F1, §21.1). For the face's grid-scale modes this dashpot is not
   dissipative: a **frictionless** elastic block **at rest** keeps a self-excited rattle (27 % of the bottom nodes in
   contact, vibration energy ≈ 100× the static elastic energy, sustained for 0.3 s) with **positive** kick-paired normal
   contact work (+1.1e-8 J) while the nominal dashpot dissipation sums to 3.0e-6 J. This is a modelling defect.
2. *Friction–normal mode-coupling flutter of the grid-scale penalty interface.* The cap μF_n follows the per-node penalty
   force instantaneously (∂F_t/∂u_n = μk_n, k_n = κm_I/dt² tied to the grid). The flutter grows with μ (bottom contact
   fraction 0.28 / 0.19 / 0.08 / 0.03 / 0.02 at μ 0.2 / 0.4 / 0.6 / 0.8 / 1.0), makes friction bursty and misaligned
   and its work correlated with braked node velocities: friction deficit and vibration energy growth (4 % of μN·slip over
   the -04 window at μ 0.8). Not removed by cfl 0.25 / 0.125, n_t 4 / 6 (worse), any κ ∈ [0.0005, 1] or ζ up to the
   2ζ√κ ≤ 1 limit; stiffer material reduces it (rigid limit). It is not continuum physics below μ = 1: rigid-flat
   Coulomb sliding of an elastic half-space is linearly stable for μ < 1.000 and ill-posed above (surface-wave growth
   ∝ |k|; `halfspace_sliding_stability.py`, ν 0.3–0.49, the Renardy / Martins class of results).

Ruled out as causes (the four candidates of the task): the dashpot energy missing from μN·slip (D_contact contains every
dashpot; the deficit has the opposite sign); stick–slip hysteresis (stick share < 0.2 % of the load in sliding); the slip
direction (−v_t instead of the trial direction: ≤ 0.4 % of g cosθ); the timing / velocity of the normal force in the
cone (dashpot on the predicted v*, the realised g_n or their mean: the flutter persists in all; the spring part only:
worse, 8–12 % of g cosθ at μ 0.8).

**Decision: route (a) + (b).**

- (a) `grid_node_penalty_coulomb_v2` (opt-in; `ContactSpec(law="grid_node_penalty_coulomb_v2")`; kernel
  `rubber_contact.node_penalty_contact_v2`, same arguments as v1, selected by the scalar `ct_law`, no new state — the B1
  batched path runs it unchanged and is L0 byte-identical, `test_v2_batched_l0_heterogeneous`): both dashpots and the
  tangential trial act on the node's realised velocity g_n = `old` after the P2G grid update minus the obstacle velocity.
  Corrects defect 1: the frictionless block at rest settles (all bottom nodes in contact, W_n < 0), the elastic block
  holds at 0.95 μ (v1: crept), the static law becomes δ_I = F_I/k_n (springs carry the whole reaction, -02 f), the impact
  books improve (n_t 2: 1.024 → 0.71 %). Parameters unchanged (κ 0.25, ζ 0.5; not re-tuned).
- (b) Defect 2 is **not** corrected; v2 declares a μ envelope with a measured error model (INC, DIAGNOSTIC,
  `sweepK_v2_errormodel.jsonl`): sliding friction accurate to the protocol limits for **μ ≤ 0.6** (slip error ≤ 0.7 %
  of g cosθ, dissipation ≥ 0.96), static friction (hold at 0.95 μ, elastic and P2) for **μ ≤ 0.8**. Outside: μ 0.8 slip
  +1.3 … +2.5 %, dissipation 0.87–0.94; μ 1.0 +2.7 … +5.8 %, 0.85–0.88, the hold at 0.95 μ is lost (slides 7 mm); μ 1.2 /
  1.5 up to +15 % / +43 %. Labels: `contact_sliding_friction_deficit_v2` (μ > 0.6 with sliding),
  `contact_friction_flutter` (μ > 0.8), `contact_coulomb_ill_posed` (μ ≥ 1.0). **This is a real limitation**: rubber–steel
  / rubber–table friction is often 0.6–1.0+, and the generic rubber recipes randomise friction over [0.3, 1.5]
  (nominal 0.8 / 1.0): scored rubber contact rows must stay inside the envelope or carry the labels (task X2).
- Not adopted: a normal-force memory of the cap (Prakash–Clifton type, first-order lag of n_τ substeps, consistent as
  dt → 0). With the v2 dashpots it removes defect 2 in steady sliding (exact to 0.1 % of g cosθ up to μ 1.0 / 1.2 / 1.5 at
  n_τ 100 / 200 / 400), but near threshold under rising load it lets edge micro-slip cascade (0.04–0.5 mm at 0.95 μ for
  every μ, also when the lag acts in slip only) and it collapses (friction 35–75 % of μN) where it does not stabilise.
  It would trade the regime v2 gets right (grasp-and-hold) for sliding accuracy (task X2).

**n_t 2 impact books.** n_t 2 is the declared `rubber_coarse_v1` profile (`reduced_fidelity: coarse_thickness_resolution`),
not a production grid; the v1 1.024 % was the first-order symplectic books error of the coarsest dt (n_t 4: 0.59 / 0.25 %,
ratio 0.42). The 1 % limit is kept for every run; v2 gives 0.71 % (n_t 4: 0.36 / 0.21 %).

**Pinch clamp balance.** Not a moving-clamp / contact work coupling: the window-mean imbalance |⟨F_x(clamp) + T⟩| is
0.002–0.65 % (v1) and 0.45–1.35 % (v2); the v1 per-step metric ⟨|F_x + T|⟩ is the strip's axial vibration (dominant
229–286 Hz = the clamped / sliding-end quarter-wave mode c/4L = 267 Hz of the 30 mm strip). With sustained unidirectional
rate-independent Coulomb slip the sliding end carries a constant force, so the undamped strip keeps the slip-onset
vibration (estimate k_ax v_pull/ω ≈ 0.047 N vs 5 % of T = 0.017 N at 20 µm): the per-step 5 % is unattainable for the
exact solution of the declared model. v2 scores the window-mean (momentum) balance at the same 5 % and reports the
per-step value.

**RUBBER-CONT-09 b (B1, `gpu_cont09_001`).** The v1 band max|A − B| ≤ max|A − A′| + 1e-6 max|A| was statistically unsound:
with one repeat, P(|A − B| > |A − A′|) = 1/2 under H0 by exchangeability (simulated 0.500), so any scalar whose GPU
run-to-run noise exceeds the 1e-6 floor is a coin flip; ~10² scalars reject almost surely. The 1-of-108 FAIL (a clamp
wrench, 1.23e-7 N vs 9.2e-8 N) stays preserved and is not evidence of coupling (criteria a and c PASS). v2 rule: k = 5 A
and 5 B episodes interleaved in one batch, per scalar d = max|A_i − B_j| ≤ 4 s + 1e-6 max|A_i| with s the larger within-side
range; per-scalar false reject ≤ 5.5e-6 under Gaussian / Laplace / t₃ / uniform noise (`cont09_band_calibration.py`,
2e6 samples), family-wise ≤ 0.4 % for ≤ 648 scalars (union bound); a shift > 5 s is always rejected. (a) and (c) stay
strict. The GPU rerun under v2 is task B2-v2.

**Results (formal CPU run `contact_v2_001`, commit `ffb099f`, clean tree, Warp 1.17.0 CPU, 32 workers, 34 jobs, 35 s).**

| Test | v1 (`contact_v1_001`) | v2 (`contact_v2_001`) |
|---|---|---|
| -01 a stick | FAIL (P2 0.95 slides 4.8 s) | **PASS** (10 rows incl. elastic + P2 at 0.95, μ 0.6 / 0.8; ≤ 7e-5 s) |
| -01 b slip | FAIL (μ 0.8 +1.65 / +1.88 / +1.06 %) | **FAIL** (μ 0.8 +1.83 / +1.26 / +2.46 %, outside the envelope, declared expected); μ 0.4 ≤ 0.09 %, μ 0.6 (unseen INC2) +0.78 / +0.42 / +0.38 % PASS |
| -01 c onset | FAIL (μ 0.8 −2.75 %) | **PASS** (−0.31 / −1.61 / −1.47 % at μ 0.4 / 0.6 / 0.8) |
| -01 d books | FAIL (1.03 %, one stick run) | **FAIL** (stick runs 1.67 / 1.28 / 1.19 % of S ≈ 2.5e-10 J; absolute ≤ 4e-12 J; slip runs ≤ 0.09 %) |
| -02 | PASS | **PASS** (+ f: Σk_nδ / reaction 1.000 for every damped run) |
| -03 | FAIL (n_t 2 books 1.024 %) | **PASS** (0.71 % n_t 2; 0.36 / 0.21 % n_t 4, order 0.59; t_c +3.3 %) |
| -04 | FAIL (μ 0.8 0.93–0.94) | **FAIL** (μ 0.8 0.87–0.94, declared expected); μ 0.4 0.986–0.997, μ 0.6 (INC2) 0.954–0.972 PASS |
| -05 | FAIL (ratio 0.972 / 0.964 / 0.956; per-step balance 6–10 %) | **FAIL** (ratio 0.978 / 0.982 / **0.964**); cone, scaling, penetration PASS; window-mean balance 0.78 / 1.35 / 0.45 % PASS (per-step 9.8 / 7.0 / 6.9 % reported) |
| -06 | NOT_RUN | NOT_RUN (NOT_IMPLEMENTED_V1) |
| -07 | PASS | CARRIED_OVER (contact-free) |
| -08 | (B1) PASS | v2 L0 N = 3 incline / pinch / pinch P2 PASS (pytest); mixed v1 / v2 refused |
| -09 | (B1) FAIL b, a / c PASS | NOT_RUN (task B2-v2, rule declared) |

Reading. The law defect (1) is fixed and the static / impact / penetration behaviour is now correct (-01 a, c, -02, -03
pass); the sliding friction above μ 0.6 and the two-sided pinch at μ_pad 0.5 still carry the friction–normal flutter of
defect (2), as declared; the stick-run books are a float-noise-level energy (≤ 4e-12 J) measured against a static
block's energy scale. Status stays EXPERIMENTAL_NOT_ACCEPTED.

**Tasks (rcs-impl unless stated).**

| Task | Content | Size |
|---|---|---|
| X2 (rcs-xhigh) | Defect 2: a contact formulation without the grid-scale friction–normal flutter (e.g. a B-consistent iterated normal constraint on the face, or a loading-safe normal-force memory — lag only in sustained macro-slip with a stick-preserving rule), under a new declared protocol; until then the μ envelope and labels apply and RL friction ranges are clipped or labelled | M–L |
| B2-v2 | RUBBER-CONT-09 on GPU0 under the v2 rule (k 5 per side, interleaved), v2 law | S |
| R1 | Independent rcs-review (xhigh) of §22, §22.9, `contact_v1_001`, `contact_v2_001` | – |

### 22.10 Task X2 — defect 2 (friction–normal flutter): negative result, no v3; the μ envelope is resolution-qualified (amendment, 2026-11-11, rcs-xhigh)

Scope: remove defect 2 of §22.9 so that sliding at μ 0.8 and 1.0 closes within the v2 thresholds, robustly (dt / dx
refinement, other geometry, batched), without regressing hold, onset, impact, penetration or pinch. Branch
`claude/rl-rubber-p24` from `2b2221b`; DIAGNOSTIC scripts `scripts/rubber/p2_4/x2_dev_laws.py`, `diag_contact_x2.py`,
`diag_x2_pattern.py` (`b019684`, modes 11–13 `331550c`); raw `artifacts/rl_ready/p2_4/x2_diag_001` (README; 14 + 8 rows
rerun at the commits identical in every field except wall time); evidence `docs/evidence/rl_ready/p2_4_contact_x2_001.json`
(writer `write_contact_x2_evidence.py`). Nothing in `src`, `tests` or `configs` changed: v1 / v2 kernels byte-identical,
v2 stays the opt-in law, `rubber_contact_tests_v1` / `_v2` unchanged, no protocol v3, no formal run. CPU suites (rubber +
batched + naming) 455 passed with `CUDA_VISIBLE_DEVICES=''` and with `=0` (GPU0 lock), logs `cpu_tests_x2_001`. Material
status `SIMULATOR_ONLY_UNCALIBRATED`.

**What the flutter is (refines §22.9 defect 2; v2 law, INC block, DIAGNOSTIC).**

- Body-scale, not grid-scale: at μ 0.8 about 2 of the 91 bottom nodes are in contact at any substep (4.4 %), v_z rms
  16 mm/s (13 % of the sliding speed), dominant 1.5–3 kHz, wavenumber spectrum broad with its peak at 1–3 waves over the
  16 mm face. Frictionless sliding settles (100 % contact, v_z 2 µm/s); μ 0.2: 60 %, 0.7 mm/s.
- Converged in dt, not in dx. μ 0.8, ratio 1.2: slip error 1.26 / 1.12 / 1.15 % of g cosθ at cfl 0.5 / 0.25 / 0.125, but
  1.26 / 2.0 / 1.7 / 2.8 / 4.6 % at n_t 3 / 4 / 5 / 6 / 8; μ 1.0: 3.8 / 6.2 / 9.9 % at n_t 3 / 4 / 6; even μ 0.4 grows
  (0.04 → 0.18 % at n_t 6). Consequence: a change of the contact's time integration alone (implicit or return-mapped
  tangential update, per-node local solve, Moreau projection, augmented-Lagrangian or stiffer / decoupled normal) cannot
  remove it — the deficit is a property of the semi-discrete model (κ_n 0.0005–1 in X1; κ 1 and ζ 1 here do not remove
  it). The growth under dx refinement at every μ is the signature of short-wavelength growth; the §22.9 half-space
  analysis (bilateral contact kept, stable below μ 1.000) does not cover the separating solutions the model produces, so
  the §22.9 statement that defect 2 "is not continuum physics below μ = 1" is not supported and is withdrawn as unproven.
- A realistic high-frequency loss in the material (one Prony term g 0.15, τ 100 µs on the elastic μ0) removes most of the
  μ 0.8 deficit (≤ 0.28 % at n_t 3 / 4; INC2 ratio 1.1: 1.24 %) and restores the μ 0.8 hold at n_t 4; it does not fix
  μ ≥ 1.0 (1.6–4.5 %, hold at μ 1.0 lost). Real rubber has such loss; the illustrative Prony sets do not.

**Candidates (DIAGNOSTIC; every row in the evidence).**

| Class (dev mode) | Best result | Failure | Verdict |
|---|---|---|---|
| Nonlocal cap, lattice radius 1–3 (Oden–Pires type, conservative) (1) | μ 0.8: 0.6–0.7 % | μ ≥ 1.0: 2.6–8 %; radius in nodes vanishes under refinement | rejected |
| Patch-level cap μ ΣF_n · m_i/Σm (6) | μ 0.8 at n_t 3: 0.12–0.91 % | cfl 0.25: 1.16 %; INC2 D04 0.947; μ 0.9: 2.4 %; μ 1.0: 2.5 %; hold μ 1.2 lost | rejected |
| Patch cap on a lagged patch total, τ 0.1–1 ms (10) | exact to μ 0.95 at n_t 3 and on INC2 | n_t 4: μ 0.8 +1.7 %, μ 0.9 +5.2 %; no recovery from a developed flutter; μ ≥ 1.0 4–44 % | rejected |
| Per-node normal-force lag (Prakash–Clifton type), τ 0.25–3 ms (4) | τ 0.5 ms: μ 0.8 exact incl. n_t 4 / INC2; τ 3 ms exact to μ 1.5 | slips under proportional loading: hold ramp-phase displacement 10.5 µm (τ 0.5 ms) vs 0.19 µm (v2), 52 µm (1 ms), 0.2–0.5 mm after the ramp (3 ms); no recovery after a developed flutter (−12 %); τ 0.25 ms −7 … −20 % (sharp threshold) | rejected (X1 finding quantified) |
| Memory kept across bounces, gated variant (8, 9) | from rest exact to μ 1.5 (τ 3 ms) | friction collapse once nodes bounce (+75 %), gated μ 1.5 +99 %, holds lost | rejected |
| Loading-safe memories: max(F_n, F_lag), slip-distance gate, patch-slip gate (2, 3, 5, 7) | — | flutter not suppressed or −15 … +8 % | rejected |
| **Patch slip-rate equalisation** (13, ζ_d 1): in slip only, outside the cap, a dashpot −2ζ_d√(k_t m_i)(v_t,i − v̄) on the deviation of the realised tangential velocity from the same-substep mean over the patch's slipping nodes (zero net force over the slipping set, patch total ≤ μΣF_n) | **μ 0.6 / 0.8 exact (≤ 0.01 %, D04 ≥ 0.993) at n_t 3 / 4 / 6, cfl 0.25, INC2; recovers from a developed flutter; hold ramp-phase 0.16 µm (v2 0.19); holds to μ 1.5 incl. μ 0.8 at n_t 4 (v2 slides 3.9 mm)** | μ 1.0 exact at n_t 3 / cfl 0.25 / INC2 but n_t 4: +2.4 %, n_t 6: +4.2 % (ratio 1.1); μ ≥ 1.2 sliding fails | promising, **not adopted** (requirement at μ 1.0 not met; src / batched / pad-box implementation and a declared protocol needed) |

**Decision.** Negative result for the X2 requirement (sliding closed at μ 0.8 and 1.0, robust under refinement, supported
range about 0.6–1.5): no `grid_node_penalty_coulomb_v3`, no `rubber_contact_tests_v3`, no formal run. v2 and the
declared labels stay. Two qualifications are recorded (DIAGNOSTIC, not re-scoring any formal row):

1. The v2 envelope (sliding μ ≤ 0.6, static μ ≤ 0.8; §22.9) was measured at n_t 3 (`rubber_rl_v1`) on INC / INC2 and holds
   there. It does not hold at the acceptance profile `rubber_ref_v1` (n_t ≥ 4): the μ 0.8 hold at 0.95 slides 3.9 mm at
   n_t 4; μ 0.6 sliding is +1.19 / +1.72 % on INC2 at n_t 4 and +1.06 % (D04 0.946) on INC at n_t 6 (ratio 1.1). On the
   larger INC2 block at n_t 3, μ 0.8 slides with +2.5 / +5.7 / +8.3 % (the §22.9 error model, +1.3 … +2.5 %, was INC only).
   Rows outside n_t 3 with sliding μ > 0.4 or static μ > 0.6 are therefore outside the established envelope and carry
   `contact_friction_flutter`; RL friction ranges stay clipped or labelled as in §22.9.
2. RUBBER-CONT-01 d (books of the stick runs). Max |B| is the end value, 2.3–4.3e-12 J against S 2.2–4.5e-10 J; B rises
   steadily through the static hold (1.7–3.0e-11 W) and falls only ~40 % from cfl 0.5 to 0.125. The static block keeps a
   sustained jitter (v_rms 5–12 µm/s, growing as dt shrinks) that the contact dashpots dissipate; the drift is the part of
   that dissipation not matched by a booked energy change. This is a genuine residual of the discrete dynamics, more than
   10 orders of magnitude above the float64 resolution of the books sums; an absolute floor tied to float64 resolution
   would not change any verdict, and the relative criterion is not measuring float64 noise. Its weakness for static rows
   is structural (the residual accumulates with the hold duration at fixed S). No rule is declared here (no v3 run); the
   `contact_v2_001` FAIL stands; a future protocol may add an a-priori drift-rate criterion after the jitter source is
   identified (task X5), never a floor fitted to seen data.

**Tasks (rcs-impl unless stated).**

| Task | Content | Size |
|---|---|---|
| X3 | `grid_node_penalty_coulomb_v3` = v2 + patch slip-rate equalisation (dev mode 13, ζ_d 1): normal pass (best obstacle, F_n, trial state, per-patch sums of m v_t over the slipping set, rows e·(P + F·B) + q), tangential pass; new node / patch arrays declared in `rubber_layout.CONTACT_ARRAY_KINDS`, masked reset, batched L0 N = 3; pads and moving boxes (pinch -05), then `rubber_contact_tests_v3` declared and SHA-pinned before its run with scored sliding μ ≤ 0.8 at n_t 3 **and** n_t 4 plus an unseen geometry; μ ≥ 1.0 stays labelled | L |
| X4 (rcs-xhigh) | Stability of steady Coulomb sliding of the declared continuum with unilateral contact (separation allowed) for μ < 1, to decide whether the dx-growing deficit is a discretisation artefact or continuum physics | M |
| X5 | Source of the static stick jitter (5–12 µm/s) and of the -01 d books drift; an a-priori drift-rate books criterion | S–M |
| P2-5 | Calibrated rubber sets must include high-frequency loss (DIAGNOSTIC: one Prony term at τ 100 µs removes most of the μ 0.8 deficit) | – |

## 23. P2-4 completion, P2-5 and P2-6: decisions and plan (amendment, 2026-11-11, rcs-xhigh)

Scope: the remaining P2-4 integration items (B3, S1, P1, the μ envelope), the P2-5 calibration interface and its
synthetic numerical validation, the P2-6 RL-ready rubber environment and its Full Native verification, the Q1w
decision of protocol v5, the Phase 2 Done mapping and the task list. Planning only: no `src/`, `tests/` or `configs/`
change, no GPU job; two CPU probes. Branch `claude/rl-rubber-p24b` at `6b36b65` (B2-v2 + X1 / X2). The full plan,
including the protocol tables, the probe records and the sized task list, is
`docs/reports/rl_ready/P2_REMAINING_PLAN.md`; this section holds the binding decisions. Nothing frozen changes (§22
list; Phase 1 `frozen_paths` / `evaluator_files` of `acceptance_v1` / `acceptance_v2`; rubber protocols v1–v5,
moving-grip v1, visc v1–v2, contact v1–v2). Material status: `SIMULATOR_ONLY_UNCALIBRATED`.

Numbering: X3's results belong in §22.11. When `claude/rl-rubber-v5` is consolidated, its §21 (protocol v5) becomes
**§19A** (task K1), because this lineage's §21 is P2-3.

### 23.1 Verified facts the decisions rest on

- **Batch packing.** A rubber batch packs heterogeneous slots when dx and dt are uniform: t 4 / 5 / 6 mm at dx 1 mm,
  i.e. n_t 4 / 5 / 6, with M = 2 Prony, a table plane, a tip traction region and the protocol options (probe 2;
  `batched/rubber_layout.py:12-14`).
- **Step rule.** μ0 ≤ 0.8 MPa and ρ ≥ 1000 kg/m³ give dt 3.827 µs, i.e. 5,226 substeps per 20 ms (probe 1).
- **Resampled reset.** Host-staged. A density change rebuilds the F8 inverses on the host: 0.27 s per env at
  65k particles (`rubber_material_batch.py:420-426, 580-627`).
- **Grip motion.** Host-staged (`:489-505`).
- **Contact cost.** With no frames the contact kernel skips the per-node slerp that dominated the pad-scene cost
  (`rubber_contact.py:416-470`).
- **Dispatch.** `open_vector_env` on `main` dispatches on `backend` only. Its file is outside `frozen_paths` and
  `evaluator_files`; `tests/test_perf_acceptance.py:152` asserts it stays changeable.
- **Native.** Native constructs the frozen `LiveCuttingMaterial` with cutting task, tool and pads
  (`isaac_sim_runner.py:690-712`), and advances it with the measured pose after PhysX (`:1568`).
- **REAL gate.** The calibration manifest v1 has no data origin, and the `REAL_CALIBRATED` gate checks only status and
  verdict (`rubber_material.py:431-444`).
- **Optimisers.** scipy only; no CMA-ES or BO package.

### 23.2 Decisions

1. **B3 deferred, not required for P2-6.** P2-6 uses kinematic grips (ideal tracking; no wrench feedback, hence no
   coupling error, §22.4).
   - RUBBER-CONT-07 found the 20 ms handoff significant at every envelope point, so B3 without C1 could only produce
     labelled rows.
   - The Lab `MaterialCoupling` surface is defined over an edge pose and a cut digit, so a rubber Lab task would be
     needed as well.
   - B3 + C1 + P1 form a post-Phase-2 robot-coupled rubber package, subject to a user decision. Receipt label:
     `robot_feedback_coupling: NOT_IMPLEMENTED_PHASE2`.
2. **S1 required before P2-6.** Option `self_penetration_check = surface_grid_v1` (default `off_v1`: no array, no
   launch).
   - Test: surface pairs of one env with |x_p − x_q| < s and |X_p − X_q| > 3·dx.
   - Grid: an env-grouped sort grid on the Phase 1 `FallbackGrid` pattern, re-implemented in a Phase 2 module; no
     `wp.HashGrid`.
   - Cadence: inside the graph every `check_substeps` (declared 256, ≈ 1 ms), plus a reported blind-interval counter.
   - Output: sticky bit `RUBBER_SELF_CONTACT_UNRESOLVED = 8`, which terminates the env in training.
   - It is a detector: self-contact physics stays `NOT_IMPLEMENTED_V1`.
3. **P1 deferred.** The P2-6 scene has no contact frames; the plane-only contact cost is measured in RL-07. A
   bounding-volume early-out, byte-identical on CPU, is allowed before the P2-6 GPU runs if the plane-only cost exceeds
   +25 %.
4. **μ envelope.** `configs/rubber/contact_envelope_v1.json` (task ENV1) maps each (law, n_t) to the formally scored
   sliding and static limits; DIAGNOSTIC rows may only narrow them.
   - **P2-6 range.** P2-6 randomises `friction_table ~ U[0.3, min over its slots' n_t of μ_slide,max]`. Pad and self
     frictions are fixed and declared unused.
   - **ENV2.** P2-6 slots have n_t 4–6, so ENV2 declares and runs a formal envelope check at n_t 5 / 6 (and n_t 4 if v2
     is the law) before P2-6 is declared.
   - **Expected bounds.** 0.8 with law v3 if `rubber_contact_tests_v3` passes; 0.4 with v2. If nothing is scored, the
     friction axis is BLOCKED.
   - **Refusal.** Values ≥ 1.0 or above the bound raise `UNSUPPORTED_FRICTION_OUT_OF_ENVELOPE`; μ ≥ 1.0 remains the
     labelled limitation `contact_coulomb_ill_posed`.
5. **Q1w (RUBBER-PHYS-11 v5 FAIL): recommendation to the reviewer (task K2) is (c),** accept n_t 4 with its measured
   bias as a labelled limitation.
   - The v5 FAIL (2.16 % under the unchanged rule; p_obs 3.12 ⇒ Fs 3) stays recorded.
   - `rubber_ref_v1` gains a `measured_bias` block (Q1 +0.39 %, Q1w +0.62 %, Q2 −0.39 %, Q4 −0.05 %) and
     `known_limitation: rubber_phys_11_v5_q1w_gci_2.16pct` in every receipt. Phase 2 Done box 2 is then
     PASS_WITH_LIMITATIONS for this item.
   - (b), a v6 rule that passes the existing data, is rejected as evaluator loosening.
   - (a), n_t 6 for wide strips, costs ≈ 5× per env-step on every wide-strip scene and needs an n_t 12 verification
     run, to remove a bias ≈ 4× below the declared β = 20 model-form error. It is the fallback if (c) is rejected,
     declared as v6 (6 / 8 / 12) before any run.
6. **P2-5: calibration interface + synthetic numerical validation `rubber_calib_tests_v1`.**
   - **Manifest.** `rcs_rubber_calibration_dataset_v2` adds `data_origin`, per-experiment sensor models and a sealed
     truth SHA. Synthetic manifests use `SYNTHETIC_*` statuses and live outside `configs/rubber/calibration/`. A test
     asserts that no synthetic manifest can satisfy `REAL_CALIBRATED`. The v1 module and the gate code stay unchanged.
   - **Forward models.** Population-batched (candidate × experiment → env).
   - **Fit API.** Formal method scipy `least_squares` TRF with a batched FD Jacobian and 3 starts; an in-repo CMA-ES
     (reported); an ask/tell hook for the external agentic optimizer or BO. The objective uses fit-split data only and
     no RL term. The held-out evaluation is one-shot, and sealed paths are refused.
   - **Synthetic truth.** Same model family at n_t 6 / CFL 0.25, sensor sampling and noise, and hidden parameters drawn
     with a secret seed whose SHA alone enters the manifest.
   - **Splits.** Fit: tensile at 2 rates, relaxation, cantilever rebound, gravity sag, friction sliding. Held out:
     other rates (incl. extrapolation), length, thickness, initial deformation and normal load.
   - **Metrics.** Force–displacement nRMSE, hysteresis energy, relaxation, tip trajectory, rebound time, decay, slip
     distance, volume, energy.
   - **Tolerances.** tol = max(2·(noise floor + discretisation floor), declared floor), evaluated on a separate pilot
     dataset and committed before the formal dataset exists.
   - **Power and order.** A negative control (single-rate, single-geometry fit) must fail the held-out split. An order
     audit runs: protocol < pilot < thresholds < dataset freeze < fit receipt < held-out < reveal.
   - **Labels.** `NUMERICAL_VALIDATION_SYNTHETIC`; never a calibration. The real-data protocol is declared but
     NOT_RUN; it requires a high-frequency Prony term (§22.10).
7. **P2-6: `RubberStripTaskFamily` behind `open_vector_env(backend="kinematic_grip_batched")`** (`rl/rubber_family.py`,
   `rl/rubber_env.py`, `rl/rubber_contract.py`).
   - **Dispatch.** The only Phase 1 file touched is a lazy-import branch at the top of `open_vector_env` in the
     non-frozen `rl/vector_env.py`, made last on the consolidated tree.
   - **Task `rubber_strip_drag_v1`.**
     - Scene: a case-A strip on a table, a tip traction and no frames.
     - Slots: L 100 / 125 / 150 mm × W 16 mm × t 4 / 5 / 6 mm at dx 1 mm, `rubber_ref_v1`.
     - Per reset: μ0 [0.3, 0.8] MPa, ρ [1000, 1300], Prony g-scale [0.5, 1.5] and τ-scale [0.5, 2], friction per
       decision 4, and a tip traction.
     - Initial deformation: a 10-step seeded simulated preload with actions ignored.
     - Goal: a tip goal at ≥ 30 mm initial distance.
   - **Actions.** `grip_delta_v1` (6: ≤ 5 mm, ≤ 0.05 rad per step, workspace clamp keeping the jaw above the table).
   - **Observations.** A 78-dim deployable `policy` (grip pose, twist, time, previous action, ideal-tracked markers and
     velocities, goal, preload flag; no force or material state) and an opt-in privileged `critic`.
   - **Reward and success.** `rubber_tip_tracking_v1` (provisional shaping); success only from `rubber_tracking_eval_v1`.
   - **Termination.** Through `rl.termination` (rubber bits → `TERM_MATERIAL_INVALID`).
   - **Not in v1.** `autoreset="same_step"`, cameras.
   - **Profile `rubber_rl_env_v1`.** F8 + kick-average + F9 + F10 + guard 1.0 + S1 + the ENV1 law. Step staging is
     device-side (new opt-in methods, byte-identical to the host path); resets stay host-staged with reported p95.
   - **Acceptance protocol.** `rubber_rl_tests_v1`, RL-01…RL-11 (plan §4.8):
     - contract, parity (L0 / L2 / L3) and randomisation;
     - isolation (L1 + the -09 v2 rule) and episode semantics incl. S1;
     - long rollout, throughput / VRAM vs Phase 1 (scored: N 8 gain over N 1, VRAM ≤ 4 GB) and dt / spacing
       (dt scored, spacing reported with the -11 rule);
     - Native, provenance and Phase 1 non-regression.
8. **Full Native verification.** A new `native_rubber_host.py` hosts `LiveRubberMaterial` (the same recipe, slot,
   profile and law) in the Kit process. It reuses `launch_app`, the static workcell (Frankas at the safe park, camera
   rig v5), `DynamicSurfaceBridge` and the overview RGB-D. `isaac_sim_runner.py` and `live_material.py` stay unchanged.
   - **Loop** (the TLMPM bridge's shape): kinematic clamp target, PhysX step, measured clamp pose,
     `advance(motions)`, surface update, record. No feedback.
   - **Run.** One 100-step episode: settle, lift + yaw, hold with tip contact.
   - **Comparison** with 20 headless GPU replays of the recorded poses (+ 10 held-out):
     - receipts and inputs bitwise equal;
     - every physics statistic inside `rubber_l3_band_v4`;
     - bridge integrity scored;
     - render and camera consistency reported, not scored.

### 23.3 Phase 2 Done (summary; mapping in the plan §6)

| Box | Expected at close |
|---|---|
| 1. models with reproducible evidence | PASS after X3, K1, K2 |
| 2. physics and numerics tests pass | PASS_WITH_LIMITATIONS: Q1w bias; μ ≥ 1.0; -01 d unless X5; self-contact not implemented, detected |
| 3. RL headless (≥ 8 envs) and Full Native | PASS if `rubber_rl_tests_v1` passes |
| 4. REAL_CALIBRATED only after a real held-out pass | PASS: status stays SIMULATOR_ONLY_UNCALIBRATED; P2-5 is synthetic numerical validation only |
| 5. old baselines unbroken | PASS via RL-11 |
| 6. merged, external pin updated | PASS after Z2 (PR to `main`) and Z3 (new versioned descriptor, SYNCED) |

### 23.4 Tasks

The plan §7 has the full table: X3 (running); K1 consolidation; K2 combined review; S1; ENV1 / ENV2; P5.1–P5.7;
P6.0–P6.7; Z1–Z3. Deferred: B3, C1, P1, X4, X5, Q1W-A, and the real-data protocol.

Estimate: ≈ 20 nominal agent-days (≈ 9-day critical path), ≈ 3–5 calendar days at the velocity observed in P2-3 /
P2-4, ≈ 11 GPU-hours (mostly GPU0; the Native run needs ≥ 14 GiB free). Add 0.5–1 day per protocol v2 cycle.
