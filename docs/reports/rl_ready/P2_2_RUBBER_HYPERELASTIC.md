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

| Task | Status | Key numbers |
|---|---|---|
| Merge `origin/claude/rl-ready-rubber-physics` | DONE (`5854f34`) | normal merge commit |
| F1 Phase 1 device pin re-check (GPU visible) | **PASS** | fallback-grid test 3 × 3 alone; file 15 / 15 GPU visible and CUDA hidden |
| F2 guard-on GPU smoke (N 1/4/8) | **PASS** | L2 invariants, 0 guard violations, VRAM ≤ 922 MB; guard +1.2–1.9 % ms/substep |
| F3 L3 rule v4 (declared before run) | **PASS** | batched max M/U 0.877 (0 / 16 outside); held-out 10 / 10 within |
| F8 energy-consistent Dirichlet | **DONE**; v4 -07b **PASS** | lumped −2.6e-4 … −1.4e-3 Ψ/T_ax → projection ≤ +1.3e-5 (= free-free); -07b 1.058 % → 0.33 % |
| F9 grid-consistent traction work | **DONE**; v4 -07a **PASS** | -07a energy 0.96 % (v3) → 0.0047 % (v4) |
| F10 float32 static precision | **DONE**; v4 -02 **PASS** | float32 band 0.9–4.2e-3, Ψ +1–3 % after the stop; f64 shadow ≤ 4.1e-7; δ/δ_extrap max 1.0178 (v3: 1.0203 FAIL) |
| R2.1 hyperelastic runs (v4, CPU) | -02/-03/-05/-06/-07 **PASS**, -08 **FAIL** (validity) | -08 physics 1.6e-6; γ 2 trips the guard (CFL_eff 1.30) |
| R2.1 batched GPU N = 4 | -07a **PASS**, -12 **PASS**, -07b FAIL (reported, lumped v1) | values = single v3 |
| F4 batched L0 -07 v3 / -12 | **PASS** | bitwise (L0 definition) every control step |
| R2.2 refinement / GCI | **FAIL** (Q1) | GCI n_t 3: Q1 3.98 % (Fs 3, p_obs 3.26), Q2 1.28 %, Q3 0.01 %, Q4 0.92 %; dt ≤ 1e-5 |
| R2.3 β 20 vs 50 | DONE (reported) | Q3 −2.7 % vs incompressible; β 50 − β 20 = analytic ±0.16 % |
| R2.4 throughput / VRAM N 1/4/8/16 | DONE; VRAM criterion **PASS** | 0.64 / 1.35 / 1.47 / 1.52 env-steps/s, ≤ 952 MB, SHARED_GPU |
| F5 guard profiling | DONE | guard 2.1 % of GPU time → no fusion |
| R2.5 review / freeze | **NOT_RUN** | reviewer decision |

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
commit `ba2dec5`, CPU, 17 runs, 18:12–20:18 KST). Protocol-v3 -02 configurations (relax_mass_proportional_v1,
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

| Lt20 t6 W = 4t n_t 3 | float32_v1 | 0.25 | 0.96604 | 0.96795 | +2.5e-4 | 2.6e-3 | +2.90 % |
| Lt20 t6 W = 4t n_t 3 | f64_shadow_v1 | 0.25 | 0.96683 | 0.96683 | +2.0e-9 | 2.0e-7 | −0.00 % |

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

## 12. R2.5 — review and freeze: NOT_RUN

R2.5 (xhigh review → acceptance; freeze `tlmpm_rubber_v1`) is not part of this implementation run: the solver
gained three opt-in options in P2-2, the -08 v4 validity FAIL and the open items below are unresolved, and an
acceptance decision belongs to the reviewer. `tlmpm_rubber_v1` stays `EXPERIMENTAL_NOT_ACCEPTED`.

## 13. Tests

| Command | Result |
|---|---|
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rubber_*.py tests/test_tlmpm_rubber_v1.py tests/test_live_rubber_material.py tests/test_batched_*.py tests/test_wp12_phase5_frozen.py tests/test_repo_naming.py` (at `3d12969`) | 315 passed (1,307 s, loaded host) |
| rubber files only (`tests/test_rubber_*.py tests/test_tlmpm_rubber_v1.py tests/test_live_rubber_material.py tests/test_batched_frozen_sources.py`) at the final commit, CUDA hidden / GPU0 visible | 144 / 144 passed |
| same with `CUDA_VISIBLE_DEVICES=0` (GPU visible) | 315 passed (1,083 s) |
| `tests/test_batched_particle_contacts.py`, GPU1 visible / CUDA hidden (F1) | 15 / 15 passed |

New tests: `test_rubber_solver_options.py` (6: defaults, projection constraint and inverse, static reaction, grid
traction work, float64 shadows, batch refusal), `test_rubber_gci.py` (7), `test_rubber_l3.py` (5, incl. a Monte Carlo of
the family false-reject rate), `test_rubber_refinement.py` (2), v4 pin + v4-vs-v3 diff in `test_rubber_phys_config.py`,
ppc 3 lattice in `test_rubber_scene.py` (the old "ppc 3 refused" assertion now refuses ppc 4). No torch code changed;
the torch suite was not run.

## 14. What P2-3 (viscoelasticity, ADR §10) needs

1. **Declare `rubber_visc_tests_v1` (RUBBER-VISC-01..05) before implementation**, with the v4 machinery: guard on,
   L3 rule v4 for GPU parity, and the energy balance written as E_kin + Ψ_eq + Σ Ψ_visc − W_ext + D_visc with the
   integrated dissipation D_visc = Σ ∫ Q_i : Ḋ (reported per step), so "loss per cycle vs πγ₀²G″" is measured on the
   material, not on numerics.
2. **Energy-consistent constraints in every hysteresis / relaxation test**: the lumped clamp loses 0.03–0.14 % of Ψ
   per axial period (§6), the same order as small-g_i Prony losses. Use `consistent_mass_projection_v1` (single env)
   and **implement it in the batched path** (per-env block inverses, CSR of constrained rows; the batch currently
   refuses the option) before batched VISC runs or P2-6 grips.
3. **Float64 state for slow/quasi-static tests** (`f64_shadow_v1`): relaxation tests at fixed C and τ ≫ dt accumulate
   tiny increments exactly as the static cantilever did (§8); the Prony internal variables Q_i (sym-6, per particle,
   M ≤ 4) should be updated in float64 for the exponential integrator (a_i, b_i in float64 from expm1) and stored
   float32 only if a test shows it suffices; batched shadows are also missing.
4. **Registry/state**: `visco_q[P·M]` and `visco_s_prev[P]` with initial value zero/snapshot in
   `batched/constitutive_rubber.py` (Phase 2 file), cleared by the masked reset (vars()-complete L1 test), per-env
   a_i, b_i columns in the params table; L0 parity single ↔ batched as for the elastic model.
5. **Stable step**: the guard uses μ₀ (instantaneous modulus), as now; add the Prony tangent bound check
   (`bound_check.py` analogue) and keep `deformation_reserve` explicit for large-strain cyclic tests (the -08 γ 2
   result shows the guard trips in large homogeneous shear at r = 1).
6. **Tests to carry over**: dt-independence of fitted τ (CFL 0.5 vs 0.25, as -11's dt criterion), three loading rates,
   reset clears internal variables; GCI-style refinement only for Q1/Q2-type observables.
7. **Cost**: guard 2 % (no fusion needed); `_bc_b` (14–17 % of GPU time) iterates all nodes twice per substep — a
   constrained-node list would cut it before P2-6 throughput work.

## 10. R2.2 / R2.3 / F4 — refinement study RUBBER-PHYS-11 (GCI) and β sensitivity

Run `refinement_v4_002` (commit `a3a079a`, src/configs clean, v4 sha256 `f1cecee6…38f0`, Warp 1.17.0 CPU, 22 workers,
44 runs, 7,541 s wall, 18:52–20:57 KST; single-env reference, guard on, static_stop_v3, `f64_shadow_v1` for Q1/Q4):

```
TMPDIR=/mnt/data/ningan/tmp CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .deps/isaacsim-venv/bin/python \
    scripts/rubber/p2_2/run_refinement_v4.py --device cpu --workers 22 --out artifacts/rl_ready/p2_2/refinement_v4_002
```

The first attempt `refinement_v4_001` was aborted by me after 20 min: `case_q4` read `live.last_diagnostics`, which does
not exist before a control step (AttributeError after the physics); preserved with `ABORTED.txt`; the fix (`a3a079a`)
touches measurement code only, the protocol is unchanged. No run is invalid, unconverged or missing in `_002`; no
guard or J flag. Q values at CFL 0.5, β 20, 2 particles per cell (h = 1/n_t):

| Q | n_t 2 | n_t 3 | n_t 4 | n_t 6 | convergence (3, 4, 6) | p_obs | p_eff / Fs | φ_ext | **GCI n_t 3** (criterion ≤ 2 %) | GCI n_t 6 | CFL 0.5 → 0.25 at n_t 3 (≤ 0.1 %) | triplet (2, 3, 4): GCI n_t 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 sag δ (m) | 5.09915e-4 | 4.99302e-4 | 4.96405e-4 | 4.95041e-4 | monotone | 3.26 | 2.0 / **3.0** | 4.93949e-4 | **3.98 % FAIL** | 0.66 % | +8.7e-8 PASS | 4.68 % (p 2.69) |
| Q2 f1 (Hz) | 18.9453 | 19.2017 | 19.2877 | 19.3356 | monotone | 2.75 | 2.0 / 1.25 | 19.374 | **1.28 % PASS** | 0.25 % | −1.0e-5 PASS | 3.05 % (p 2.12) |
| Q3 P11 at λ 1.5 (Pa) | 410,733 | 410,766 | 410,762 | 410,804 | oscillatory (range 1.03e-4) | 4.87 | 2.0 / 3.0 | 410,838 | **0.01 % PASS** | 0.025 % | +1.6e-5 PASS | grid independent |
| Q4 large sag δ (m) | 0.0663048 | 0.0658945 | 0.0658064 | 0.0657977 | monotone | 8.25 | 2.0 / 3.0 | 0.0657907 | **0.92 % PASS** | 0.032 % | −2.4e-9 PASS | 3.34 % (p 3.34) |

**RUBBER-PHYS-11: FAIL** (Q1 production GCI 3.98 % > 2 %; Q2–Q4 PASS; every dt criterion PASS; validity PASS). The
Q1 verdict follows the declared rule: p_obs = 3.26 lies outside the verified window [1, 3], so Fs = 3.0 with p_eff = 2.
With p_obs and Fs 1.25 the same data give 1.19 %, and the actual distance of n_t 3 from the extrapolated value is
+1.08 % — reported, not a re-evaluation; the declared verdict stands. The n_t 3 production grid carries a 1.1 % (Q1),
0.9 % (Q2), 0.2 % (Q4) discretisation error relative to the p_eff = 2 extrapolation; n_t 4 halves it.

The CFL 0.25 triplets give the same picture (GCI n_t 3: Q1 3.98 %, Q2 1.29 %, Q3 grid independent, Q4 0.92 %); dt changes
on every grid ≤ 1.2e-4 (largest: Q2 at n_t 2), i.e. time-step error is negligible against grid error.

**Model form and quadrature (R2.3, reported, not scored)** at CFL 0.5:

| Q | β 50 / β 20 − 1 (n_t 3 / n_t 4) | analytic ratio β 50 / β 20 | β 20 vs incompressible reference | ppc 3 / ppc 2 − 1 (n_t 3; Q3 n_t 4) |
|---|---|---|---|---|
| Q1 | −1.14 % / −1.12 % | δ_EB: −0.98 % | δ_EB(E_eff) / δ_EB(3μ) = 1.0167 | −1.56 % |
| Q2 | +0.62 % / +0.61 % | f_EB: +0.50 % | f_EB(E_eff) / f_EB(3μ) = 0.9918 | +0.90 % |
| Q3 | +1.62 % / +1.63 % | compressible P11: +1.62 % | 410,766 vs 422,222 Pa (incompressible NH): −2.71 % | +0.00 % |
| Q4 | −0.47 % / −0.46 % | elastica(E_eff): −0.39 % | elastica(E_eff) / elastica(3μ) = 1.0066 | −0.53 % |

β 20 is a documented model-form error (§6): −2.7 % nominal stress at λ 1.5 against incompressible NH (as predicted),
+1.7 % static sag, −0.8 % frequency; the simulated β 50 − β 20 differences match the analytic compressible-model
differences to ≤ 0.16 % (the discretisation itself is β-insensitive: β 50 / β 20 the same on n_t 3 and 4). The ppc 3
quadrature check moves the bending observables by −1.6 % (Q1), +0.9 % (Q2), −0.5 % (Q4) at n_t 3 — the same order as the
n_t 3 grid error, and for Q2 ppc 3 at n_t 3 (19.3754 Hz) equals the n_t → ∞ extrapolation of ppc 2 (19.374): the 2-ppc
quadrature is part of the n_t 3 error; uniaxial stress (homogeneous) is insensitive. Q5 (−06 type, load half): gauge J̄
error ≤ 4.1e-5 (β 20), J̄ ∈ [1.000, 1.055] at λ ≤ 2.

**F4 float32 static floor and float64-marker noise:** with `f64_shadow_v1` the static floor is ≤ 4.1e-7 of δ (§8); the
float64-marker noise of static_stop_v3 is below that (geometrically similar configurations agree to 1e-7 in v4 -02).

## 11. F4 — batched L0 (CPU bitwise) of RUBBER-PHYS-07 v3 and -12

`scripts/rubber/p2_2/run_batched_l0_v3.py`: the single-env reference and a `BatchedRubberMaterial` of 2 copies
(Warp CPU, parity mode) get the v3 inputs; after every control step every particle / node / group / traction array,
work, J and CFL violation counters, diagnostics, markers, wrench and invalid bits of both batched envs are compared
with the single reference (the L0 definition of `tests/test_rubber_batched.py`: `np.array_equal`, plus NaN at the same
places). CPU, 19,200 particles/env, guard on, v1 options.

| case | control steps | result | run |
|---|---|---|---|
| -07a ramped (λ 1.5, traction ramp) | 5,621 × 18 substeps | **equal at every step** (both envs), no flag | `f4_batched_l0_07_ramped_001` (code `44e5b59`, 3,974 s) |
| -07b instant (λ 1.03) | 3,373 × 18 | **equal at every step**, no flag | `f4_batched_l0_07_instant_001` (code `44e5b59`, 2,552 s) |
| -12 guard regression | 68 × 862 | **equal at every step** incl. the non-finite steps 56–68; invalid bits 7 = 7 = 7 | `f4_batched_l0_12_003` (`6b26534`, 1,422 s) |

**F4 batched L0: PASS.** Preserved earlier -12 attempts: `_001` compared with `np.array_equal` without `equal_nan`
and stopped matching exactly at step 56, where single and batched states both became non-finite (NaN ≠ NaN); `_002`
compared byte-wise and reported a first byte difference at step 54 in the derived `wrench` (the release step), where
`array_equal` finds equality — a signed-zero difference (+0.0 / −0.0) in a [G, 6] average, not a value difference.
`_003` records both: L0-equal at every step; first byte-wise difference at step 54 (wrench). The 07 runs' `commit`
field was read at the end of the run; the code they imported at their start (18:22 KST) is `44e5b59`.

**Erratum (2026-11-10, ADR 0023 §20.4; review F5):** "bitwise" / "equal at every step" in this section and the
evidence field `bitwise_equal_every_step: true` mean **value-exact** (`np.array_equal`; NaN positions equal; signed zero
not distinguished). The first byte-level difference of -12 (step 54, derived wrench) is the signed zero of a zero
reaction, made sign-consistent in the batched `_wrench_b` after this report; the recorded runs and evidence are
unchanged. From §20 on, batched L0 tests compare raw bytes.

## 15. Artifacts

`artifacts/rl_ready/p2_2/`: `f2_gpu_smoke_guard_001`, `l3_v4_001`, `throughput_v4_001`, `batched_gpu_v4_001`,
`f8_clamp_dissipation_001`, `f10_static_precision_001`, `rubber_phys_v4_001`, `refinement_v4_001` (aborted, preserved),
`refinement_v4_002`, `f4_batched_l0_07_ramped_001`, `f4_batched_l0_07_instant_001`, `f4_batched_l0_12_001` / `_002`
(preserved, comparison notes) / `_003`. Compact: `docs/evidence/rl_ready/p2_2_rubber_001.json`.

## 16. Open items

- -11 Q1 GCI FAIL at the production grid n_t 3 (3.98 % by the declared rule): either `rubber_rl_v1` moves to n_t ≥ 4
  for bending-dominated observables or the acceptance accepts a documented ~1 % bending error at n_t 3 — a decision
  for R2.5; a new -11 version must be declared before any re-run.
- -08 v4 validity FAIL (guard at γ 2): a v5 -08 with an explicit `deformation_reserve` (≥ 1.3) declared before its run.
- Batched execution of `consistent_mass_projection_v1` and `f64_shadow_v1` (the batch refuses both).
- `_bc_b` node loop (14–17 % of GPU time) → constrained-node list.
- R2.5 xhigh review and freeze decision.

## v5 — protocol `rubber_phys_tests_v5` (R2.5 review decisions (a), (b), F7, F9; ADR 0023 §19A)

> K1 note (2026-11-11): this protocol's ADR section was §21 on `claude/rl-rubber-v5`; after the merge into
> `claude/rl-rubber` it is **ADR 0023 §19A** (§21 there is P2-3 visco). Commit messages and the frozen v5 configs keep
> the old number.

Branch `claude/rl-rubber-v5` (from `claude/rl-rubber` @ `1b21801`). Material status: **SIMULATOR_ONLY_UNCALIBRATED**.
Evidence: `docs/evidence/rl_ready/p2_2_v5_001.json`. Raw outputs: `artifacts/rl_ready/p2_2_v5/` (not committed).

### v5.1 Declaration and order

| Commit | Content | Time (KST) |
|---|---|---|
| `8881ff0` | `configs/rubber/rubber_phys_tests_v5.json` (sha256 `7e02f57b…67f0`) + SHA pin and v5-vs-v4 test, committed alone | 2026-11-10 21:58:10 |
| `687f4fe` | review fixes: `rubber_rl_v1` → `reduced_fidelity: thickness_resolution_n_t3` with a measured bias, `workload_e_v2.json`, ADR 0023 E1 / §3 D5 / §4.2 / §8 errata and §21 (now §19A), ppc 3 refused without `quadrature_check` (F9) | before the runs |
| `1bdef79` | v5 runners | before the runs |
| first v5 run | `refinement_v5_001` | 22:06:45 |

Nothing was re-declared after any result. What the test asserts: the -11 rule block, thresholds, CFL set, guard and
verdict CFL are byte-identical to v4, and every threshold of every test is unchanged.

### v5.2 Results

| Test (v5) | Setup change | Measured | Verdict |
|---|---|---|---|
| -02 small sag (24) | n_t 4 / 6, extrapolation pair (4, 6) | narrow δ/δ_EB 0.9828–0.9928; wide 0.9235–0.9594 (in the bracket); **δ/δ_extrap 1.0005–1.0073** (max: Lt20 W4t n_t 4); reaction ≤ 3.4e-6; converged; no flag | **PASS** |
| -03 scaling | n_t 4; series_t L 0.096 m | exponents 4.0077, −2.0082 | **PASS** |
| -04 free vibration | n_t 4, guard on | f1/f_EB − 1 = +4.8e-5; f1(dt)/f1(dt/2) − 1 = −5.3e-5; energy drift ≤ 5.0e-4 of peak Ψ; amplitude ratio 1.0017; CFL_eff ≤ 0.518; no flag | **PASS** |
| -04 batched N = 4 CPU | byte-wise L0, guard on | 312 steps × 257 substeps, 0 byte mismatches (every array, diagnostics, markers, wrench, work, counters); env 0 tip series byte-equal to the formal dt run; invalid bits 0 | **PASS** |
| -05 large sag | n_t 4 | δ/δ_elastica 0.9995 / 0.9939 / 0.9968 (α 20 / 3 / 9); J̄ ∈ [0.9904, 1.0101] | **PASS** |
| -08 simple shear | r = 1.5 (declared rule) | 300 substeps over the v4 interval, CFL 0.333; CFL_eff at end 0.440 / 0.560 / **0.864** (γ 0.5 / 1 / 2; the declared 0.864381); σ12 error −8.6e-7, N1 error −1.7e-6 (every γ); no flag | **PASS** |
| -09 records | guard on | 18 / 18 records valid; cantilever and vibration runs flag-free; stretch_release (λ 1.5, instantaneous release): CFL flag in 6 / 6 runs at the first control step after the release (step 4), plus J_DOMAIN in the two W = t runs; no run became non-finite | **PASS** |
| -11 refinement | see v5.3 | Q1 / Q2 / Q4 GCI PASS; **Q1w GCI 2.16 % > 2 %**; dt PASS; valid PASS | **FAIL** (Q1w) |
| workload E v2, GPU | n_t 4, N 1 / 4 / 8 | L2 invariants and VRAM criterion met for every N (v5.4) | **PASS** |

Not re-run, by the declaration: -01 (v2 PASS), -06 and -07 (v4 PASS; uniaxial, n_t 4), -10 (v4), -11 Q3 (v4 PASS) and
-12 (v3). The earlier FAILs stay recorded: v4 -08 (validity), v4 -11 (Q1 at n_t 3), v3 -02 / -07, v2 -07 and v2 L3.

### v5.3 RUBBER-PHYS-11 v5 (`refinement_v5_001`, CPU)

Run: commit `1bdef79`, clean tree, Warp 1.17.0 CPU, `.deps/isaacsim-venv`. 24 jobs on 24 workers; host load about 45
falling to about 1. Window 2026-11-10 22:06:45 to 2026-11-11 08:47:34 (38,450 s; the last job is Q1w n_t 8 with
331,776 particles). Guard on; `f64_shadow_v1` for Q1 / Q1w / Q4. No invalid flag, no guard violation, every static
initializer converged.

```
TMPDIR=/mnt/data/ningan/tmp CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 .deps/isaacsim-venv/bin/python \
    scripts/rubber/p2_2_v5/run_refinement_v5.py --device cpu --workers 24 --out artifacts/rl_ready/p2_2_v5/refinement_v5_001
```

Values at CFL 0.5, β 20, 2 particles per cell (h = 1/n_t). The production GCI is GCI_coarse of n_t 4 from the triplet
(4, 6, 8), under the unchanged rule.

| Q | n_t 3 | n_t 4 | n_t 6 | n_t 8 (new) | (4, 6, 8) | p_obs | p_eff / Fs | φ_ext (p_eff) | **GCI n_t 4** (≤ 2 %) | GCI_fine n_t 8 | dt CFL 0.5 → 0.25 at n_t 4 (≤ 0.1 %) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 sag, W = t (m) | 4.99302e-4 | 4.96405e-4 | 4.95041e-4 | 4.94799e-4 | monotone | 3.86 | 2 / 3.0 | 4.94487e-4 | **1.48 % PASS** | 0.19 % | −1.1e-7 PASS |
| **Q1w sag, W = 4t (m)** | 4.83413e-4 | 4.79708e-4 | 4.77789e-4 | 4.77343e-4 | monotone | 3.12 | 2 / 3.0 | 4.76769e-4 | **2.16 % FAIL** | 0.36 % | −3.5e-8 PASS |
| Q2 f1 (Hz) | 19.2017 | 19.2877 | 19.3356 | 19.3478 | monotone | 2.90 | 2 / 1.25 | 19.3633 | **0.56 % PASS** | 0.10 % | +4.1e-5 PASS |
| Q4 large sag (m) | 0.0658945 | 0.0658064 | 0.0657977 | 0.0658160 | oscillatory | 3.79 | 2 / 3.0 | 0.0658395 | **0.07 % PASS** | 0.11 % | +4.1e-7 PASS |
| Q3 P11 at λ 1.5 | – | – | – | – | – | – | – | – | v4 PASS (0.01 %), not re-run | – | – |

**RUBBER-PHYS-11 v5: FAIL.** Q1w (the wide strip, the workload E geometry class) has a production GCI of 2.16 %. Under
the declared rule, p_obs 3.12 lies outside the verified window [1, 3], so Fs = 3.0 with p_eff = 2. Q1, Q2 and Q4 PASS.
Every dt criterion and every validity criterion PASS.

The verdict stands as declared. Reported for reading only, not as a re-evaluation:
- n_t 4 lies +0.62 % from the p_eff extrapolation for Q1w (+0.56 % with p_obs);
- the rule's Fs 1.25 branch would give 0.90 % with p_eff 2, but that branch does not apply here.

A further -11 version would have to be declared as v6 before any run.

**Measured bias of `rubber_rl_v1` (n_t 3), reported, not scored.**

| Q | φ(3)/φ_ext − 1 (p_eff 2; p_obs) | GCI_coarse n_t 3 from (3, 4, 6), same rule | φ(4)/φ_ext − 1 (for comparison) |
|---|---|---|---|
| Q1 sag, L/t 20, W = t | **+0.97 %** (+0.93 %) | 3.98 % (Fs 3) | +0.39 % |
| Q1w sag, L/t 20, W = 4t | **+1.39 %** (+1.34 %) | 2.19 % (Fs 1.25, p_obs 2.97) | +0.62 % |
| Q2 first bending frequency | **−0.83 %** (−0.80 %) | 1.28 % | −0.39 % |
| Q4 large-deflection tip drop | **+0.08 %** (+0.11 %) | 0.92 % (Fs 3) | −0.05 % |

The `rubber_material.RESOLUTION_PROFILES["rubber_rl_v1"]["measured_bias"]` block now prints these v5 values.
`workload_e_v2.json` is SHA-pinned and keeps its v4-based text, which points to this measurement.

**Reproducibility.** 27 configurations were rerun in v5 with code paths identical to v4: Q1 / Q2 / Q4 at n_t 3 and 4
(both CFLs) and at n_t 6, plus the twelve -02 n_t 4 configurations. All 27 equal their v4 values exactly (Warp CPU is
deterministic).

### v5.4 Workload E v2 (n_t 4, full fidelity), batched on GPU (`workload_e_v2_gpu_001`)

**Run conditions:**
- **Command:**
  ```
  flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python \
      scripts/rubber/p2_2_v5/run_workload_e_v2.py --out artifacts/rl_ready/p2_2_v5/workload_e_v2_gpu_001
  ```
- **Lock and memory:** blocking lock acquired 22:07:56 with 30,959 MiB free.
- **Code:** commit `1bdef79`, clean tree.
- **GPU:** RTX 5090 GPU0 (uuid GPU-76f61dc7…), driver 570.211.01 (CUDA 12.8). Window 22:07:57–22:09:30.
- **Sharing: SHARED_GPU.** Co-tenant PID 642659 held 766 MB in every sample. GPU utilisation was 0 % before the run and
  99–100 % during it, mostly this run.
- **Workload:** `workload_e_v2.json` row E_full: 150 × 20 × 4 mm, `rubber_ref_v1`, n_t 4, 97,280 particles per env,
  50 Hz, dt 3.002 µs from the range maxima, 6,662 substeps per control step, per-reset material randomisation.
- **Solver settings:** guard 1.0; v1 solver options; BC launch `motion_table_v1` (the code default; part of the P2-3
  prep commits after 3a43d1b, which the R2.5 review did not cover); graph chunks of 500.
- **Not used:** cameras, recording, robot.
- **Timing:** warm = median of control steps 2–6; cold = control step 1 after build and capture; 20 masked resets with
  resampled material.

| N | particles/env | control Hz | material dt | substeps | cameras | env-steps/s (warm) | aggregate sim-time/wall | ms/substep | VRAM peak (process) | reset p95 | physics verdict | cold / warm wall per step | GPU / driver | sharing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 97,280 | 50 | 3.002 µs | 6,662 | none | 0.865 | 0.0173 | 0.173 | 698 MB | 1.03 ms | L2 PASS | 1.255 / 1.156 s | RTX 5090 GPU0 / 570.211.01 | SHARED_GPU |
| 4 | 97,280 | 50 | 3.002 µs | 6,662 | none | 1.057 | 0.0211 | 0.568 | 794 MB | 0.58 ms | L2 PASS | 3.857 / 3.785 s | RTX 5090 GPU0 / 570.211.01 | SHARED_GPU |
| 8 | 97,280 | 50 | 3.002 µs | 6,662 | none | 1.033 | 0.0207 | 1.163 | 922 MB | 0.49 ms | L2 PASS | 7.830 / 7.746 s | RTX 5090 GPU0 / 570.211.01 | SHARED_GPU |

**L2 for every N:**
- finite before and after the resets;
- per-env mass constant between resets;
- invalid bits 0, also after one step past the resets;
- guard violations 0 (max CFL_eff 0.48–0.53);
- env-ownership audits 0;
- J̄ ∈ [0.9950, 1.0049].

The VRAM peak is ≤ 0.92 GB, which meets the ≤ 4 GB criterion. **Verdict: PASS.**

**Throughput.** It saturates by N = 4, at 5.6e8 / 6.8e8 / 6.7e8 particle-substeps/s. These rows are not comparable with
the P2-2 n_t 4 rows of `f2_gpu_smoke_guard_001` (0.47 env-steps/s at N 8): those ran with the `all_nodes_v1` BC launch
on GPU1 beside a 13 GB co-tenant at 95–97 % utilisation.

### v5.5 Tests

Code at `1bdef79`; `.venv` Python 3.12, Warp 1.17.0. Command:

```
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rubber_*.py tests/test_tlmpm_rubber_v1.py \
    tests/test_live_rubber_material.py tests/test_batched_*.py tests/test_rl_policies_batched.py tests/test_repo_naming.py
```

| Run | Result |
|---|---|
| `CUDA_VISIBLE_DEVICES=''` | 331 passed, 4 skipped, 804 s |
| `CUDA_VISIBLE_DEVICES=0`, under the GPU0 lock | 331 passed, 4 skipped, 745 s |

- The 4 skips are the torch / WP-17 recording tests of `test_rl_policies_batched.py`, which are NOT_RUN in `.venv`.
- Logs: `artifacts/rl_ready/p2_2_v5/cpu_tests_001/`.
- New or changed tests:
  - `test_rubber_phys_config.py`: v5 pin, v5-vs-v4 diff, -08 reserve rule, `workload_e_v1` / `v2` pins and labels;
  - `test_rubber_phys_v5_cases.py` (5 tests);
  - `test_rubber_refinement.py`: v5 jobs and evaluation, including a cross-check against the v4 evaluation;
  - `test_rubber_scene.py`: ppc 3 refusal;
  - `test_rubber_material.py`: the relabel.

### v5.6 Other runs, commands and disclosures

- **`rubber_phys_v5_001`:** commit `1bdef79`, clean, 56 cases, 16 workers, 22:06:47–01:56:17 (13,771 s). Command:
  ```
  run_rubber_phys.py --protocol v5 --tests 02,03,04,05,08,09 --device cpu --workers 16 \
      --out artifacts/rl_ready/p2_2_v5/rubber_phys_v5_001
  ```
  - Its own `summary.json` has -04 as PASS_EXCEPT_NOT_RUN, because the batched part was not yet available.
  - `summary_final.json` is the same evaluation with `--batched-04` (evaluate-only, new file, no overwrite).
- **`batched_04_v5_001`:** started after the formal dt run, 1,834 s. Command:
  ```
  run_batched_04_v5.py --formal-dt .../raw/04/dt.json
  ```
  - Its meta records `tree_dirty: true`. The only difference from `1bdef79` was the then-untracked evidence writer
    `write_p2_2_v5_evidence.py`, which the run does not import; it was committed in `dc26f03`.
- **Device:** the -11 declaration pins the Warp CPU device. Every refinement job ran on the CPU; none was moved to a GPU.

### v5.7 Open items

- **-11 v5 FAIL (Q1w at n_t 4, GCI 2.16 % by the declared rule).** The wide strip at n_t 4 does not pass the unchanged
  rule. A reviewer decides the next step. Options:
  - n_t 6 as the full-fidelity grid for wide bending-dominated scenes;
  - a v6 rule;
  - accepting n_t 4 with its documented bias.

  Any of these must be declared before a further run. This bears directly on whether `workload_e_v2` (n_t 4) can carry
  the "full" label for wide strips.
- GPU equivalence of the static refinement observables (float32 atomics in P2G / MUSL) is not established.
- The `motion_table_v1` BC launch used by the GPU rows is still unreviewed (ADR 0023 §20.6).
