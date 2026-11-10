# P2-2 — Rubber material v1: hyperelastic verification, refinement, throughput (ADR 0023 §15 R2.1–R2.5, §18.8, §19)

Branch `claude/rl-rubber` (worktree `RoboticCuttingSim-rl-rubber`). Material status: **SIMULATOR_ONLY_UNCALIBRATED**
(numerical verification only; no real rubber data). Protocol: `configs/rubber/rubber_phys_tests_v4.json` (declared
and SHA-pinned in `3d12969` before any v4 formal run; v1/v2/v3 unchanged and their results/FAILs preserved).
Decisions: ADR 0023 §19. Compact evidence: `docs/evidence/rl_ready/p2_2_rubber_001.json`. Raw outputs:
`artifacts/rl_ready/p2_2/` (not committed).

Nothing frozen changed: v1–v7_2 solvers, law 8, evaluators, `tlmpm.py`/`tlmpm_v2.py`, Phase 1 batched modules
(`batched/constitutive.py`, `errors.py`, `particle_contacts.py`, …), `configs/rl_benchmark/*`, `scripts/rl_ready/*`,
`rubber_phys_tests_v1/v2/v3.json`. `tests/test_batched_frozen_sources.py` passes.

## 0. Status per task

<!-- STATUS_TABLE -->

## 1. Integration merge and F1 (Phase 1 device pin)

`5854f34` merges `origin/claude/rl-ready-rubber-physics` (`db036a0`: FallbackGrid pins its launches to its device; P1-4,
B2, R0) into `claude/rl-rubber` (normal merge commit, no rebase). F1 re-check with a GPU visible
(`CUDA_VISIBLE_DEVICES=1`, before the GPU1 exclusive window): `test_fallback_grid_reproduces_warp_grouped_query_sequence`
alone 3 × 3 passed; `tests/test_batched_particle_contacts.py` 15 passed (138 s) with GPU1 visible and 15 passed (134 s)
with CUDA hidden. **PASS** (the §18.6 race is gone).

## 2. F2 — guard-on GPU smoke (R1.7 L2 with `stable_step_guard=1.0`)

`92e1b7d` added `--guard-modes` / `--control-steps` to `scripts/rubber/gpu_smoke_rubber.py` (part B runs guard off and
on back to back per N, same GPU conditions). Command (blocking `flock …/rcs_rl_runner/gpu1.lock`, acquired 18:03:54
with 18.9 GB free; GPU1 was released to the coordinator's exclusive window afterwards):

```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/gpu_smoke_rubber.py \
    --out artifacts/rl_ready/p2_2/f2_gpu_smoke_guard_001 --device cuda:0 --parts B --N 1,4,8 --guard-modes off,1.0 --control-steps 5
```

RTX 5090 GPU1, 18:03:55–18:08:43 KST, **SHARED_GPU** (co-tenant PID 2186023, 13.1 GB, GPU1 utilisation 95–97 %;
PID 2240909 634 MB at the end). Workload E smoke variant (97,280 particles/env, n_t 4, 50 Hz, dt 3.00 µs from the
range maxima μ 1.3 MPa / ρ 1000, 6,662 substeps), per-env material resampled in range, 5 control steps (warm = median
of steps 2–5), no cameras, no recording.

| N | guard | ms/substep (warm median) | env-steps/s | reset p95 | VRAM peak (process, cumulative) | guard violations | max CFL_eff (end) | L2 invariants |
|---|---|---|---|---|---|---|---|---|
| 1 | off | 0.467 | 0.322 | 2.27 ms | 690 MB | – | 0.472 | PASS |
| 1 | 1.0 | 0.476 (+1.9 %) | 0.314 | 2.32 ms | 698 MB | 0 | 0.472 | PASS |
| 4 | off | 1.290 | 0.484 | 2.29 ms | 794 MB | – | 0.305–0.501 | PASS |
| 4 | 1.0 | 1.311 (+1.7 %) | 0.458 | 2.28 ms | 794 MB | 0, 0, 0, 0 | 0.305–0.501 | PASS |
| 8 | off | 2.539 | 0.473 | 2.28 ms | 922 MB | – | 0.305–0.521 | PASS |
| 8 | 1.0 | 2.570 (+1.2 %) | 0.468 | 2.32 ms | 922 MB | 0 × 8 | 0.305–0.521 | PASS |

L2 (every row): finite, per-env mass constant between resets, every invalid bit 0 (incl. `RUBBER_CFL_EXCEEDED`),
env-ownership audits 0, J̄ ∈ [0.9954, 1.0046], VRAM ≤ 1.5 GB. **F2 PASS.** The guard costs 1.2–1.9 % of wall time per
substep here (shared GPU; the co-tenant load explains the slower absolute numbers than `gpu_smoke_001`).

## 3. F3 — RUBBER-PHYS-10 L3 under rule v4

Rule `rubber_l3_band_v4` (`src/robotic_cutting_sim/rubber_l3.py`, declared in v4 §RUBBER-PHYS-10.setup.L3_v4 before
the run; modelled on Phase 1 `position_band_v2`): one statistic per (env, observable) — M = max over the 30 control
steps of |y(k) − ȳ_ref(k)| — so the time correlation of a series does not enter the false-reject rate; the reference
distribution comes from leave-one-out M_i of n = 20 single-env GPU repeats; one-sided lognormal prediction bound
exp(mean ln M + t_{1−a,n−1}·√(1+1/n)·sd ln M) with a = 0.05/16 (Bonferroni over 4 envs × 4 observables), Cantelli
fallback when Shapiro–Wilk p < 0.01 or an M_i = 0. Nominal false-reject ≤ 5 % per batched run for any correlation
(Monte Carlo of the rule on random-walk series: 2.3 % at n = 10, 2.1 % at n = 20). 10 further single repeats
(held-out) are classified by the same rule and reported. The v2 band FAIL (`gpu_smoke_001`) stays the v2 result.

Run `l3_v4_001`: commit `3d12969` (clean), v4 sha256 recorded in `meta.json`; `flock …/gpu0.lock`, RTX 5090 GPU0,
18:31:46–18:34:44 KST, **SHARED_GPU** (co-tenants PID 642659 766 MB, PID 2211019 12.0 GB), 176 s, process VRAM peak
572 MB. RUBBER-PHYS-04 heterogeneous envs (2,304 / 1,872 / 4,608 / 2,304 particles), 1 ms control, dt 5.18 µs
(193 substeps), guard on, no invalid flag in any of the 31 runs.

| run | statistics outside | max M / U | verdict |
|---|---|---|---|
| batched N = 4 (scored) | 0 / 16 | 0.877 | **PASS** |
| held-out singles (10, reported) | 0 / 16 each | 0.53–0.80 | 10 / 10 WITHIN |

All 16 bounds used the lognormal method (Shapiro–Wilk p 0.076–0.996). **RUBBER-PHYS-10: L0/L1 PASS (CPU), L2 PASS
(F2, guard on), L3 PASS under rule v4** (the v2 L3 FAIL remains recorded).

## 4. R2.4 / F5 — throughput, VRAM and guard profile (workload E, `throughput_v4`)

Run `throughput_v4_001`: commit `3d12969`, `flock …/gpu0.lock`, RTX 5090 GPU0, 18:34:47–18:37:12 KST,
**SHARED_GPU** (co-tenant PID 2211019 12.0 GB and PID 642659 766 MB; GPU0 utilisation samples 92 / 93 / 50 / 99 %).
Workload E (`configs/rubber/workload_e_v1.json`): 150 × 20 × 4 mm strip, profile `rubber_rl_v1` (n_t 3, β 20, CFL 0.5
at the range maxima μ 1.3 MPa / ρ 1000), case A clamp, gravity, per-env material resampled inside the ranges at
reset, **stable-step guard on**, CUDA graph chunks of 500 substeps (9 replays + a 496-substep remainder), no cameras,
no recording, no robot. Warm = median of control steps 2–6; cold = step 1 (after build and capture).

| N | particles/env | control Hz | material dt | substeps / control step | n_t | β | cameras | env-steps/s | aggregate sim-time / wall | ms / substep | VRAM peak (process) | reset p95 | physics verdict | warm / cold wall | GPU | sharing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 41,220 | 50 | 4.003 µs | 4,996 | 3 | 20 | none | 0.640 | 0.0128 | 0.313 | 728 MB | 6.0 ms | L2 PASS | 1.56 / 1.91 s | RTX 5090 GPU0 | SHARED_GPU |
| 4 | 41,220 | 50 | 4.003 µs | 4,996 | 3 | 20 | none | 1.345 | 0.0269 | 0.595 | 760 MB | 5.2 ms | L2 PASS | 2.97 / 3.16 s | RTX 5090 GPU0 | SHARED_GPU |
| 8 | 41,220 | 50 | 4.003 µs | 4,996 | 3 | 20 | none | 1.467 | 0.0293 | 1.092 | 824 MB | 8.3 ms | L2 PASS | 5.46 / 3.94 s | RTX 5090 GPU0 | SHARED_GPU |
| 16 | 41,220 | 50 | 4.003 µs | 4,996 | 3 | 20 | none | 1.515 | 0.0303 | 2.113 | 952 MB | 2.6 ms | L2 PASS | 10.56 / 10.08 s | RTX 5090 GPU0 | SHARED_GPU |

L2 for every N: finite, mass constant between resets, invalid bits 0, guard violations 0 (max CFL_eff at the end
0.48–0.53), audits 0. VRAM ≤ 0.95 GB at N = 16 (criterion ≤ 4 GB: **PASS**). Mempool 15 / 61 / 122 / 244 MB (arrays
grow linearly; the rest is the CUDA context). The N = 8 cold step was faster than its warm steps: co-tenant load
changed during the row (the 50 % utilisation sample). Throughput saturates at N ≈ 8: 1.3e8 / 2.8e8 / 3.0e8 / 3.1e8
particle-substeps/s at N = 1 / 4 / 8 / 16.

Against the Phase 1 v7_2 rows (`m2_n16_001`, workload A, N = 16, 103,315 particles/env, 200 substeps of 1e-4 s per
control step, 4.82 env-steps/s, 2,064 MiB, SHARED_GPU with a 45 % co-tenant; P1_1_P1_2_MILESTONES.md §6): the rubber
kernels process 3.1e8 particle-substeps/s against 1.0e8 for v7_2 at N = 16, but a rubber control step needs 4,996
substeps (25× v7_2's 200, set by the β = 20 wave speed, D11), so workload E delivers 1.5 env-steps/s against 4.8 with
2.5× fewer particles per env. The two rows ran on different co-tenant loads and are not a matched comparison.

**F5 guard profile** (`profile.json`): 200 eager substeps of the N = 8 batch under `wp.timing_begin` (CUDA activity
timing, GPU execution time per kernel): 0.826 ms GPU per substep with the guard, 0.734 without (separate builds, shared
GPU: individual kernels vary ±30 % between the two samples, e.g. `_stress_b` 25.2 vs 16.7 ms). The guard kernel
(`_cfl_guard_b`) is **2.1 %** of the GPU time with the guard on; the wall-time delta in F2 was 1.2–1.9 %. The
declared fusion rule (only above 5 % and only if bitwise equal) is not triggered: **no fusion**. Largest kernels:
`_p2g` 25 %, `_bc_b` 14–17 %, `_stress_b` 11–15 %, `_musl` 10 %, `_g2p` 8–10 %, `_deformation` 7–11 %,
`_grid_positions` 6–7 %. `_bc_b` runs over every node of every env twice per substep although only clamp nodes do
work: a group-node index list would remove most of it (P2-6 optimisation candidate, not done here).

## 5. R2.1 (GPU part) — batched RUBBER-PHYS-07 v3 / -12 on a GPU (`batched_gpu_v4`)

Run `batched_gpu_v4_001` (`scripts/rubber/p2_2/run_batched_gpu_07_12.py`; script logic checked first on the CPU in
parity mode against the v3 formal -12: flag steps 54 / 56 / non-finite 56 reproduced): commit `3d12969`,
`flock …/gpu0.lock`, RTX 5090 GPU0, 18:37–18:38 KST, SHARED_GPU. N = 4 copies of the v3 env (19,200 particles/env,
v1 solver options — the batched path has no opt-in option —, guard on, strict off), each env evaluated by the frozen
v3 criteria from its own per-control-step rows.

| case | control steps × substeps | wall | per-env result | verdict |
|---|---|---|---|---|
| -07a ramped (λ 1.5) | 5,621 × 18 | 48 s | mean length +3.0e-5, energy 0.960 % of Ψ, lateral ≤ 1e-6 m, no flag (all 4 envs) | **PASS** (declared criterion) |
| -07b instant (λ 1.03) | 3,373 × 18 | 25 s | mean length +4.4e-4, **energy 1.058 % of Ψ**, no flag (all 4) | FAIL (reported, as declared: same lumped-clamp loss as the v3 single run) |
| -12 guard regression | 68 × 862 | 12 s | CFL flag at step 54, J flag at 56 in all 4 envs; the GPU state stayed finite to the end (the CPU reference became non-finite at 56) | **PASS** |

The batched GPU outcomes reproduce the single-env v3 values to the printed digits (07a 0.96 %, 07b 1.058 %).

## 6. F8 — energy-consistent Dirichlet constraint

**Quantification first** (diagnostic, not a formal result): `f8_clamp_dissipation_001`
(`scripts/rubber/p2_2/diag_clamp_dissipation.py`, commit `ba2dec5`, 20 single-env CPU runs, 18:13–19:25 KST). The -07
strip (60 × 10 × 4 mm, μ 0.4 MPa) is stretched to λ 1.03 (cosine ramp 60 T_ax, 5 ms control interval, hold 10 T_ax),
grip 1 is removed instantaneously and the strip runs 20 T_ax; E_total = KE_particle + Ψ − W_ext relative to the
release, sampled every 0.1 ms; guard on (no flag in any run).

| constraint | CFL | n_t | end mask | particles | max abs dE / Ψ | mean dE / Ψ (last quarter) | slope dE / Ψ per T_ax |
|---|---|---|---|---|---|---|---|
| lumped_v1 | 0.5 | 3 | x | 8,100 | 1.49 % | −1.04 % | −6.4e-4 |
| lumped_v1 | 0.5 | 3 | xyz | 8,100 | 2.94 % | −2.40 % | −1.4e-3 |
| lumped_v1 | 0.5 | 4 | x | 19,200 | 1.09 % | −0.71 % | −4.2e-4 |
| lumped_v1 | 0.5 | 4 | xyz | 19,200 | 1.71 % | −1.33 % | −7.9e-4 |
| lumped_v1 | 0.5 | 6 | x | 64,800 | 0.71 % | −0.46 % | −2.6e-4 |
| lumped_v1 | 0.25 | 3 | x | 8,100 | 0.96 % | −0.71 % | −4.4e-4 |
| lumped_v1 | 0.25 | 3 | xyz | 8,100 | 2.23 % | −1.84 % | −1.1e-3 |
| lumped_v1 | 0.25 | 4 | x | 19,200 | 0.66 % | −0.48 % | −2.8e-4 |
| lumped_v1 | 0.25 | 4 | xyz | 19,200 | 1.15 % | −0.92 % | −5.5e-4 |
| free-free (no group after the release) | 0.5 | 3 | x | 8,100 | 0.85 % | +0.05 % | +1.3e-5 |
| free-free | 0.5 | 4 | x | 19,200 | 0.67 % | +0.04 % | +0.9e-5 |
| **consistent_mass_projection_v1** | 0.5 | 3 | x | 8,100 | 0.43 % | +0.02 % | +1.3e-5 |
| projection | 0.5 | 3 | xyz | 8,100 | 0.43 % | +0.03 % | +1.0e-5 |
| projection | 0.5 | 4 | x | 19,200 | 0.34 % | +0.02 % | +1.0e-5 |
| projection | 0.5 | 4 | xyz | 19,200 | 0.37 % | +0.02 % | +0.8e-5 |
| projection | 0.5 | 6 | x | 64,800 | 0.24 % | +0.01 % | +0.7e-5 |
| projection | 0.25 | 3 | x | 8,100 | 0.21 % | +0.01 % | +0.7e-5 |
| projection | 0.25 | 3 | xyz | 8,100 | 0.22 % | +0.01 % | +0.5e-5 |
| projection | 0.25 | 4 | x | 19,200 | 0.17 % | +0.00 % | +0.5e-5 |
| projection | 0.25 | 4 | xyz | 19,200 | 0.18 % | +0.01 % | +0.3e-5 |

Reading: the lumped clamp loses energy secularly, more for the full clamp (xyz) than for the axial mask, more on
coarser grids, and less at the smaller step but not in proportion to dt (CFL 0.5 → 0.25: × 0.65–0.77) — a constraint
effect, not a time-integration error. The projection removes the secular loss to the free-free level (slope
+0.3…+1.3e-5 per T_ax in every configuration) and its bounded fluctuation halves with dt (0.43 → 0.21 %, O(dt)).

**Method** (`tlmpm_rubber_v1`, option `dirichlet="consistent_mass_projection_v1"`, ADR §19.1): pure FLIP keeps
v = Wᵀa, so KE = ½aᵀM_c a and the MUSL grid velocity is g = M_l⁻¹M_c a. The lumped v1 projection zeroes g_C but feeds
back a lumped-mass correction through the FLIP increment, so a leaves the constraint subspace and the symplectic
structure is lost at the wall. The M_c-orthogonal projection onto {(M_c a)_C = m_C v_presc} is a correction supported
on the constrained nodes only, with the constant CC block of M_c (TLMPM weights never change): μ = A⁻¹(p_C − m_C v_presc),
v_p −= Σ_I w_pI μ_I, then the MUSL scatter is repeated. Dense float64 inverses per (group, component) block (≤ 4,096
nodes; -07: 55 face nodes per grip; workload E clamp ≈ 300 per component), fixed-order gathers (deterministic, graph
safe). Reaction −m_I μ_I (torque about the group origin), work = impulse · v_presc. Tests
(`tests/test_rubber_solver_options.py`): the inverse is exact; after a substep the MUSL momentum on every clamped
(node, component) is < 1e-5 of the largest nodal momentum (lumped: > 100× larger); the static clamp reaction of a
relaxed cantilever balances the weight (< 1e-3). Default `lumped_v1` = v1 launches; every v1 L0 test unchanged. The
batched path refuses the option (`UNSUPPORTED_SOLVER`): **a batched projection is future work** (P2-6 grips).

**v4 -07 (declared before its run, both variants with the projection and F9):** see §9.

## 7. F9 — grid-consistent traction work

Option `traction_work="grid_displacement_v1"`: the traction work is accumulated after the position update as
f_t (m_p / M_t)·(x_end − x_start), the work of the nodal force over the grid-consistent displacement (the energy
pairing of the symplectic scheme), instead of j·v_p(start of substep). Test: the accumulated work equals the numpy
sum over every substep to 1e-6. Used in v4 -07a (§8); the v3 -07a value (0.96 % with the v1 measure, −0.48 % with the
grid-consistent one in the P2-1 diagnostic) stays the v3 result.

## 8. F10 — float32 static-state precision

**Quantification first** (diagnostic): `f10_static_precision_001` (`scripts/rubber/p2_2/diag_static_precision.py`,
commit `ba2dec5`, CPU, 17 runs, 18:12–20:xx KST). Protocol-v3 -02 configurations (relax_mass_proportional_v1,
static_stop_v3 float64 markers, guard 1.0), but the damped relaxation continues to 10 T1 and the float64 tip deflection is
recorded every T1/20. Columns: δ/δ_EB at 3 T1 (the v3 stop time) and at 10 T1, least-squares drift per T1 and band
(max − min) over [3, 10] T1, change of Ψ from 3 to 10 T1.

| configuration (geometrically similar triples) | state precision | CFL | δ/δ_EB at 3 T1 | at 10 T1 | drift / T1 | band [3, 10] T1 | Ψ change |
|---|---|---|---|---|---|---|---|
| Lt20 t3 W = t n_t 3 | float32_v1 (v3) | 0.5 | 0.99733 | 0.99844 | +2.0e-4 | 2.9e-3 | +0.97 % |
| Lt20 t3 W = t n_t 3 (repeat) | float32_v1 | 0.5 | 0.99733 | 0.99844 | +2.0e-4 | 2.9e-3 | +0.97 % (bitwise identical: deterministic CPU) |
| Lt20 t4 W = t n_t 3 | float32_v1 | 0.5 | 1.00009 | 1.00252 | +4.0e-4 | 2.9e-3 | +0.81 % |
| Lt20 t4 W = t n_t 3 | float32_v1 | 0.25 | 0.99692 | 1.00059 | +4.1e-4 | 4.2e-3 | +2.25 % |
| Lt20 t6 W = t n_t 3 | float32_v1 | 0.5 | 0.99797 | 0.99813 | +4.9e-5 | 1.9e-3 | +0.97 % |
| Lt20 t3 W = 4t n_t 3 | float32_v1 | 0.5 | 0.96527 | 0.96544 | +5.6e-5 | 8.8e-4 | +1.04 % |
| Lt20 t4 W = 4t n_t 3 | float32_v1 | 0.5 | 0.96556 | 0.96541 | +1.1e-5 | 8.6e-4 | +0.97 % |
| Lt20 t6 W = 4t n_t 3 | float32_v1 | 0.5 | 0.96738 | 0.96634 | −1.9e-4 | 1.7e-3 | +1.02 % |
| Lt20 t3 W = t n_t 3 | **f64_shadow_v1** | 0.5 | 0.99860 | 0.99860 | +1.2e-8 | 4.1e-7 | −0.00 % |
| Lt20 t4 W = t n_t 3 | f64_shadow_v1 | 0.5 | 0.99860 | 0.99860 | −1.6e-9 | 4.1e-7 | +0.00 % |
| Lt20 t4 W = t n_t 3 | f64_shadow_v1 | 0.25 | 0.99860 | 0.99860 | +4.2e-9 | 3.2e-7 | −0.00 % |
| Lt20 t6 W = t n_t 3 | f64_shadow_v1 | 0.5 | 0.99860 | 0.99860 | −1.4e-9 | 3.4e-7 | +0.00 % |
| Lt20 t3 W = 4t n_t 3 | f64_shadow_v1 | 0.5 | 0.96683 | 0.96683 | +8.9e-11 | 1.6e-7 | +0.00 % |
| Lt20 t4 W = 4t n_t 3 | f64_shadow_v1 | 0.5 | 0.96683 | 0.96683 | +2.4e-9 | 1.6e-7 | +0.00 % |
| Lt20 t6 W = 4t n_t 3 | f64_shadow_v1 | 0.5 | 0.96683 | 0.96683 | −9.0e-10 | 1.9e-7 | −0.00 % |

<!-- F10_LATE -->

Reading: the float32 values at 3 T1 reproduce the v3 formal results exactly (0.96527 / 0.96556 / 0.96738 for the
W = 4t triple, the -02 FAIL). After the stop the float32 state keeps moving in a 0.9–4.2e-3 band while Ψ grows by
≈ 1 % — and more at the smaller step (CFL 0.25: band 4.2e-3, Ψ +2.25 %), which identifies the mechanism: the increments
dt·Σ v_I ⊗ ∇N_I of F (and dt·Σ w v_I of x) fall below half an ulp of F ≈ 1 (6e-8 / 1.2e-7) once the motion has decayed,
so F changes only by noise-driven ulp jumps — a biased random walk, not a slow physical mode. With the float64 shadows
the state is stationary to ≤ 4.1e-7 from 3 to 10 T1, independent of the step (CFL 0.5 / 0.25 identical to 1e-8), and
each geometrically similar triple agrees to 1e-7 (0.998603 / 0.966832): the static precision is no longer limited by
float32 accumulation. The -02 v3 FAIL stays recorded.

**Method** (option `state_precision="f64_shadow_v1"`, ADR §19.1): F64 += dt Σ v_I ⊗ ∇N_I and x64 += dt Σ w_pI v_I in
float64 (the frozen `_deformation` / `_grid_positions` formulas), F = float32(F64), x = float32(x64) every substep;
everything else (stress, transfers, atomics) unchanged. Tests: x and F are exactly the float32 roundings of the
shadows; the path equals the float32 path to ≤ 4 ulp after three control steps; reset restores the shadows. Cost: two
extra per-particle float64 arrays (96 B/particle). Single-env only (batched refuses it). Declared for -02/-03/-05 v4
and the static observables Q1/Q4 of -11 before their runs.

## 9. R2.1 — formal protocol-v4 run of the hyperelastic tests (`rubber_phys_v4_001`)

Command (commit `3d12969`, src/configs clean, v4 sha256 `f1cecee6…38f0`, Warp 1.17.0 CPU, 16 workers on the shared
Threadripper PRO 9975WX, load 32–74; 39 configurations, 4,225 s wall, 18:31–19:42 KST):

```
TMPDIR=/mnt/data/ningan/tmp CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/run_rubber_phys.py \
    --protocol v4 --tests 02,03,05,06,07,08 --device cpu --workers 16 --out artifacts/rl_ready/p2_2/rubber_phys_v4_001
```

Guard on in every configuration; -02/-03/-05 with `f64_shadow_v1`; -07 with the projection and grid traction work.

| Test (v4) | Measured | Verdict |
|---|---|---|
| -02 small sag (24) | narrow δ/δ_EB 0.9834–0.9986; wide 0.9262–0.9668 (inside the bracket); **δ/δ_extrap 1.0050–1.0178** (Lt20 W4t n_t 3: 1.0178); reaction ≤ 5.8e-6; converged; no flag; geometrically similar configurations identical to ≤ 1e-6 (e.g. Lt20 W4t n_t 3: 0.966832 / 0.966832 / 0.966832; v3: 0.96527 / 0.96556 / 0.96738) | **PASS** |
| -03 scaling | exponents 4.0095, −2.0085 | PASS |
| -05 large sag | δ/δ_elastica 0.9991 / 0.9984 / 0.9981 (α 20 / 3 / 9); J̄ ∈ [0.9918, 1.0088] | PASS |
| -06 uniaxial load–unload | stress errors −4.0e-4 … +9.3e-4; gauge J̄ error 5.8e-5; unload/load 7.1e-4; residual length 1.1e-5; no flag | PASS |
| -07a ramped (λ 1.5) | mean length +4.1e-5; **energy drift 0.0047 % of Ψ** (v3: 0.96 %); lateral 7.3e-8 m; CFL_eff ≤ 0.64; no flag | PASS |
| -07b instant (λ 1.03) | mean length +4.4e-4; **energy drift 0.33 % of Ψ** (v3: 1.058 % FAIL); lateral 1.3e-6 m; no flag | PASS |
| -08 simple shear | σ12 and N1 errors ≤ 1.6e-6 (PASS); **γ = 2 sets RUBBER_CFL_EXCEEDED** (validity) | **FAIL** (validity) |

The -08 FAIL is the declared v4 validity rule ("any invalid flag of a configuration FAILs it", the guard newly on in
-08): the homogeneous block at γ = 2 reaches dt·c_bound/dx = 0.515 × 2.52 = 1.30 > 1.0 (bound ratio 1.28 / 1.63 / 2.52
at γ 0.5 / 1 / 2). The run is stable and the stresses are exact (the exact discrete-operator ratio at γ 2 is 2.36, F1 of
§18.1, i.e. 1.21 against a measured critical 1.79 for a periodic block), but the step is outside the guard-certified
domain; a large-shear test needs `deformation_reserve` r ≥ 1.3. Not re-declared after seeing the result: a v5 -08 with
r declared before its run is an open item. -01, -04, -09 (v2) and -12 (v3) were not re-run (protocol and dt unchanged).

**Status per test after P2-2:** -01, -04, -09 PASS (v2); -02, -03, -05, -06, -07 PASS (v4); -08 FAIL (v4 validity;
physics PASS; v2 PASS under the v2 protocol without guard); -10 L0/L1/L2/L3 PASS (L3 under v4); -11 see §10; -12 PASS
(v3; batched GPU and L0 see §5, §11). Earlier FAILs stay recorded: -07 v2 and v3, -02 v3, -10 L3 v2.

<!-- BODY -->
