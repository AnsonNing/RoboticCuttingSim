# ADR 0023 — Rubber material v1 (Phase 2, P2-0): backend, constitutive model, near-incompressibility, viscoelasticity, contact, scene/API

Date: 2026-11-10. **Status: accepted (rcs-xhigh, 2026-11-10)** for the binding decisions D1–D6 and D9–D11 (backend,
constitutive model, near-incompressibility, time integration, resolution, boundary conditions, code organisation,
cost rules, P2-0 deliverables). D7 (viscoelasticity, P2-3) and D8 (contact and coupling, P2-4) are accepted as
**designs**: their formulas, layouts and test protocols are binding, but each needs its own probe evidence before it
is implemented as accepted physics. Plan decision D-12 routes this decision here and the implementation to rcs-impl.

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
  contact.
- **CUDA graph:** substeps per control step are thousands (§13), so the graph holds a chunk of `K_g` substeps
  (default 500) replayed `substeps/K_g` times; the device substep index carries across replays (`_advance`), so pose
  interpolation is unchanged. No host work between chunks.
- **Static-equilibrium initializer** (used by RUBBER-PHYS-02/05 and by preload snapshots): dynamic relaxation with
  either kinetic damping at **CFL ≤ 0.25** or mass-proportional damping at the solver CFL, stop on
  `max |v| < 1e-5 m/s` and BC-reaction/weight balance within 1e-3. It is labelled `initializer`, never used inside an
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
