# P2-3 — Rubber viscoelasticity `neo_hookean_jbar_prony_v1` (ADR 0023 §21.2–21.6, R3.1–R3.9)

Branch `claude/rl-rubber-p23` (worktree `RoboticCuttingSim-rl-rubber-p23`). Material status:
**SIMULATOR_ONLY_UNCALIBRATED**. The Prony sets (P2: g 0.2 / 0.15, τ 5 / 50 ms; SLS: g 0.3, τ 10 ms) and the recipe
`configs/materials/rubber_generic_visco_v1.json` (the §14.1 example values) are **illustrative, not fitted to any real
rubber data**. RUBBER-VISC-03 (b/c) is a synthetic dry run of the P2-5 fit / held-out split, not a calibration.
Protocol: `configs/rubber/rubber_visc_tests_v1.json` (sha256 `95ad1f94…f080`, pinned in
`tests/test_rubber_p2_3_protocols.py`, unchanged). Evidence: `docs/evidence/rl_ready/p2_3_visco_001.json`. Raw
outputs: `artifacts/rl_ready/p2_3/` (not committed).
**Protocol v2** (`rubber_visc_tests_v2.json`, sha256 `212fe7af…4663`, ADR 0023 §21.7) re-scores RUBBER-VISC-03 and -05
after the v1 FAILs: formal run `visc_v2_001`, evidence `docs/evidence/rl_ready/p2_3_visco_002.json`, results in §8. The
v1 results of §3 (including both FAILs) are preserved unchanged. Cite RUBBER-VISC-03 and -05 as
"v1 FAIL, v2 PASS (post-hoc measurement correction: the diagnostic re-analysis b9baa57 computed the v2 outcome on v1 data before v2 was declared in dca4848)" (review R2, `c9d771f`; ADR §21.7.7).

Nothing frozen changed: `tlmpm_rubber_v1.py` (`28fc5189…`) and `rubber_constitutive.py` (`c12cc77e…`) are byte-identical,
as are the Phase 1 modules (`batched/constitutive.py`, `batched/errors.py`, …), `configs/rl_benchmark/*`,
`scripts/rl_ready/*`, and the rubber protocols v1–v4 / moving-grip v1 / visc v1.

## 0. Status per task

| Task | Status | Key numbers |
|---|---|---|
| R3.1 shared visco Warp functions | **DONE** | S⁰ = 2∂Ψ/∂C (FD), objectivity, exact update for linear S⁰, g = 0 byte-equal to `nh_jbar_pk1` |
| R3.2 recipe / MaterialValues.prony / params64 | **DONE** | expm1 coefficients byte-equal to the host; μ₀ step rule (VISC-08a) |
| R3.3 analytic references | **DONE** | reproduce every declared design value (E′/E″, retardation 6.2 / 61.9 ms, ζ 4.743 1/s, f_d 11.706 Hz) |
| R3.4 `TLMPMRubberVisco` + energy option + guard | **DONE** | VISC-09 single byte-identical; accumulators = host recompute every substep |
| R3.5 Prony tangent check (gate) | **PASS** | 2400 states, max ratio **0.9910** ≤ 1 (`visc_tangent_check_001`) |
| R3.6 registry model + batched path | **DONE** | L0 byte-wise N 1/3, reset memory, vars()-complete, refusals |
| R3.7 formal CPU run `visc_v1_001` | 7 PASS / **2 FAIL** | -01 -02 -04 -06 -07 -08 -09 **PASS**; **-03 FAIL**, **-05 FAIL** (root causes below) |
| R3.8 GPU cost rows (VISC-10) | **DONE (reported)** | P2 stress pass +9…+19 % ms/substep; energy option +65…+123 %; SHARED_GPU (§4) |
| R3.9 report / evidence / ADR §21.6 | **DONE** | review by rcs-review (xhigh) **NOT_RUN** |
| v2 decision (ADR §21.7, rcs-xhigh) | **DONE** | diagnosis verified on v1 raw data; v2 justified for -03 and -05; no threshold changed |
| v2 protocol, option, formal run | -03, -05: **v1 FAIL, v2 PASS (post-hoc measurement correction: the diagnostic re-analysis b9baa57 computed the v2 outcome on v1 data before v2 was declared in dca4848)** | `visc_v2_001`: de-embedded E′ −0.40…−0.41 %; books ≤ 0.035 %; dynamics byte-identical to v1 (§8) |

## 1. Implementation (commits on `claude/rl-rubber-p23`)

- `efff788` R3.1–R3.4. `rubber_visco_constitutive.py`: `vec6d` helpers (xx yy zz yz xz xy), float64 S⁰ (C⁻¹ = cofᵀcof/J²),
  Prony update, overstress PK1 `P = P_NH + float32(F·Σ(Q_i − g_iS⁰))` (terms with g = 0 skipped; all g = 0 → P is
  the frozen `nh_jbar_pk1` result, no addition), branch stored / dissipation densities, `stable_step_guard_visco_v1`
  (`(1 + Σ(g_i − b_i))·W + max(0, Gershgorin(overstress))`), numpy twins. `rubber_material.py`: `MaterialValues.prony`
  (validated: ≤ 4 terms, g ≥ 0, τ > 0, Σg < 1; `to_dict` unchanged for elastic materials), `params64` row
  `[g∞, (g, τ, a, b)×M]` (a = exp(−dt/τ), b = −g(τ/dt)·expm1(−dt/τ), host float64), `pad_prony`. Recipe
  `rubber_generic_visco_v1.json`. `rubber_visco_reference.py`: E*(ω), E(t) / J_E(t) by partial fractions, Boltzmann
  staircase, finite-strain hereditary integral (quad, epsrel 1e-12, integration by parts per interval, exact
  propagation between knots), EB beam root, the protocol's Fourier estimator. `tlmpm_rubber_visco_v1.py`:
  `TLMPMRubberVisco(TLMPMRubberCW)` (the v1 launch sequence; Prony stress pass / guard when M ≥ 1, frozen `_stress`
  / `_cfl_guard` when M = 0; opt-in `visco_energy = branch_work_v1`; visco diagnostics; reset; receipts);
  `LiveRubberMaterial` builds it for a material with Prony terms.
- `fc3d23e` R3.5 tangent-check script. `555d6fb` R3.6 + runner + cost script: registry `prony_model(M)`, batched
  arrays `params64[N,1+4M]`, `visco_q[P·M]`, `visco_s_prev[P]` (+ energy arrays), model from the lives (mixed →
  `UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS`, different M / `visco_energy` → `UNSUPPORTED_NONUNIFORM_BATCH`), generic
  registry-state reset kernel, `stage_params64`, Prony g/τ-scale resampling (`prony_g_scale`, `prony_tau_scale`),
  visco diagnostics; `rubber_visco_protocol.py` (shared builders and the byte-wise L0 comparison).
- `546d5a2`, `0555131`: evidence writer, DIAGNOSTIC cantilever-books script (left-point vs kick-paired branch work).
- Numerical note (found in R3.1): Warp emits float literals in kernels as float32, so `wp.float64(-2.0/3.0)` is
  −0.66666669 in the frozen `nh_jbar_pk1` and in `iso_pk2_d` alike (the overstress form stays consistent); the numpy
  twins use the same exponent, the analytic references the exact −2/3 (difference ~3e-8·|ln J|).

## 2. Gate — RUBBER-VISC-08b Prony tangent check (`visc_tangent_check_001`, commit `fc3d23e`, clean, CPU, 2.4 s)

`CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/visco_tangent_check.py --out
artifacts/rl_ready/p2_3/visc_tangent_check_001` (seed 2311). 2400 states (P2 and SLS × dt, dt/2, dt/4 of the strip
step 5.714 µs; F with stretches 0.6–1.6, |γ| ≤ 1, random rotations; Q from load–hold–unload histories), exact
algorithmic acoustic tensor by float64 central differences of the stress-pass map, 512 directions.
**max ratio 0.9910** (float64 twin 0.9910), median 0.957, p99 0.987 → **PASS**. Reported: the premise "isochoric
material part PSD" does **not** hold (min eigenvalue −1.26e5 Pa over 200 states); the bound holds through the margin of
the elastic W bound (worst cases at J ≈ 2.6–2.9). The (1 + Σ(g−b)) factor adds only ~1e-4 at τ ≫ dt. Margin 0.9 %.

## 3. Formal run `visc_v1_001` (commit `555d6fb`, clean tree, Warp 1.17.0 CPU, 16 workers, 23:24:30–23:44:50 KST)

`CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/run_visc_v1.py
--workers 16 --out artifacts/rl_ready/p2_3/visc_v1_001`. Every configuration valid (no invalid flag, 0 guard
violations, 0 negative dissipation increments). Particles: block 512 (35 substeps / 0.2 ms; ×2, ×4 for -04a), strip
4320 (35), cantilever 5376 (173 / 1 ms; ×2, ×4).

| Test | Verdict | Numbers (threshold) |
|---|---|---|
| -01 relaxation, block, hereditary integral | **PASS** | max ‖P−P_ref‖/‖P_ref(t_r)‖ over the hold: SLS uniaxial 2.9e-6, SLS shear 7.7e-7, P2 uniaxial 3.0e-6, P2 shear 1.1e-6 (1e-4); instantaneous-step closed form differs by 0.26–0.41 % (reported) |
| -02 creep / recovery, strip, traction | **PASS** | curve dev 0.28 % / 0.11 % of max ε (2 %); creep ratio 1.09772 vs 1.09858 (−0.079 %, 1 %); recovery 0.004 % (2 %) |
| -03 hysteresis, 6 rates, identification, block | **FAIL** | (a) E′ rel: 1 Hz −0.40 %, 2 −0.41, 4 −0.43, 8 −0.51, **16 −1.04 %**, **32 −2.54 %** (1 %); E″ −0.44…−1.64 % (3 %); loss −0.36…−0.41 % (3 %). (b) **PASS**: τ₁ −5.1 %, τ₂ −0.4 % (10 %), g₁ +1.7 %, g₂ +0.3 % (5 %). (c) held-out: 2 Hz −0.18 / +0.04 %, 8 Hz −0.07 / −0.54 %, **32 Hz E′ +1.82 %** (1 %), E″ +3.2 % (4 %). (d) one-term 4 Hz fit mispredicts E″ by −33 / +44 / −22 % (reported). (e) block γ₀ 0.15: 1.3e-7 / 1.4e-7 (1e-4) **PASS** |
| -04 dt independence | **PASS** | (a) τ₁, τ₂ fits at ×1/×2/×4 within 8e-8 of each other and of the declared values (0.5 %); (b) ζ ×2/×4 +2.1e-6 / +2.0e-6 (0.5 %), f_d −7e-8 / −8e-8 (0.2 %); vs EB root: f_d 11.7148 vs 11.7059 Hz (+0.076 %, 3 %), ζ 4.7297 vs 4.7434 1/s (−0.29 %, 5 %) |
| -05 energy books | **FAIL** | (a) max \|B_N\|/S_N: strips 0.031–0.21 %, creep 0.16 %, **cantilever 2.58 % / 1.05 % / 0.47 %** (×1/×2/×4; 0.5 %). (b) **PASS** 16 Hz ×2/×1 ratio 0.19 (0.65). (c) ΔW_ext/ΔW_b − 1: **1 Hz +1.17 %**, 2 Hz +0.27, 4 +0.12, 8 +0.08, 16 +0.12, 32 +0.27 % (0.5 %). (d) **PASS** ΔD/ΔW_b − 1: +0.77, −0.13, −0.28, −0.34, −0.21, +0.38 % (2 %); 0 negative increments. (e) g = 0 control 16 Hz: 0.020 %, trend −8.6e-5 /s (reported) |
| -06 partial reset memory | **PASS** | env 1 byte-identical to a fresh batch after the reset and for 20 steps, and to a fresh single; envs 0/2 unchanged; resampled (g ×0.8, τ ×1.5) = single built with it; params64 = host expm1 bytes; plain reset restores; single reset clears, two episodes identical |
| -07 single vs batched L0 / L1 | **PASS** | 0 mismatches, every array every control step: A N1 (2816) / N3 (2816, 2304, 3328), C N1 (4320) / N3 (4320, 3456, 2592, inactive grip step 1), block N1/N3; L1 isolation; vars()-complete; 4 refusals |
| -08 step rule, tangent, validity | **PASS** | (a) dt of the visco recipe = elastic rule at μ₀; (b) max ratio 0.9910 (rerun, same seed); (c) 0 guard violations in every configuration |
| -09 elastic limit | **PASS** | prony = [] vs `tlmpm_rubber_v1` (CW, same options) and g = 0 vs elastic, single and batched, cases A, C, block: 0 mismatches |
| -10 cost | reported | §4 |

### 3.1 RUBBER-VISC-03 FAIL — root cause: the protocol's error budget, not the material

The simulated E′ is low by a rate-dependent amount. Three effects add up and each one is quantified:

1. **Finite-strain model form** ≈ −0.40 % at every rate: the neo-Hookean uniaxial tangent is 3μ(1 − 2ε) at the mean
   strain ε = 0.2 % (observed −0.40 % at 1–4 Hz, where 2 and 3 are negligible). The protocol budgeted "about 0.4 %".
2. **Drive-end inertia** of the grip-1 reaction: quasi-statically σ(L) − σ̄_gauge = −(3/8)ρω²L²ε, i.e. E′ biased by
   −(3/8)ρω²L²/E = −0.10 / −0.40 / −1.48 % at 8 / 16 / 32 Hz. The protocol budgeted (f/f_ax)² ≤ 0.4 % at 32 Hz,
   about 4× too small: the factor is (3/8)π²(f/f₁)², not (f/f₁)². Check: the grip-0 reaction carries +(1/8)ρω²L²/E. The
   E′(grip 1) − E′(grip 0) difference observed is −0.14 / −0.53 / −1.96 %, predicted −0.13 / −0.53 / −1.97 %
   (DIAGNOSTIC, same raw data).
3. **Estimator leakage**: at 16 and 32 Hz a period is 312.5 / 156.25 control steps, so "the last full cycle" is not
   a whole period, and the declared estimator does not remove the σ mean. With exact synthetic steady-state signals the
   declared estimator alone gives E′ −0.25 / −0.69 % and E″ −0.30 / −0.82 % (16 / 32 Hz; 0 at 1–8 Hz).

Sum: 8 Hz −0.50 (obs −0.51), 16 Hz −1.05 (obs −1.04), 32 Hz −2.57 % (obs −2.54 %). The material and solver reproduce
E*(ω) to ≈ 0.1 %. The held-out 32 Hz failure in (c) has the same cause: the simulated 32 Hz value carries the −2.5 %
bias while the fit (on 1/4/16 Hz) does not. The loss per cycle (energy, insensitive to the force location) is within
0.36–0.41 % at every rate, and the block at finite amplitude matches the exact hereditary integral to 1.4e-7.
**Not changed here** (thresholds never move after a result). A v2 must be declared before its run; options: a
mean-removed least-squares sinusoid fit, a control_dt with an integer number of samples per period, an inertia
correction or the mid-span stress, and an error budget with the (3/8)π² factor. Decision for rcs-xhigh.

### 3.2 RUBBER-VISC-05 FAIL — root cause: the declared branch-work quadrature (`branch_work_v1`)

The strip runs close (≤ 0.21 %). In the cantilever, B_N drifts secularly after the tap: −1.0 % at the grip stop and
−2.5 % at 0.55 s. The drift is first order (0.47 % at ×4) and continues after the oscillation has decayed. D − W_b
equals −B_N. DIAGNOSTIC `visc_diag_cantilever_books_001/_002` (CPU, 0.25 s record, commits `546d5a2` / `0555131`):

| Variant | B_N max, left-point W_b (declared) | drift after stop | B_N max, kick-paired W_b (diagnostic) |
|---|---|---|---|
| elastic (all g = 0) | 0.04 % | +0.03 %/s | 0.037 % |
| P2 | 2.08 % | −3.5 %/s | **0.013 %** |
| τ 5 ms only | 1.31 % | −2.3 %/s | 0.017 % |
| τ 50 ms only | 5.69 % | −24 %/s | 0.021 % |
| τ 5 ms only, ×2 substeps | 0.61 % | −1.0 %/s | 0.009 % |

Cause: the KE change of a substep is the internal force times the kick-average velocity, i.e. it pairs P_n with
(ΔF_{n−1} + ΔF_n)/2 (the conjugate of the discrete kinetic energy, as for the constraint work in §21.1). The declared
W_b = Σ pb_n:ΔF_n (left point) differs by ½Σ pb_n:(ΔF_n − ΔF_{n−1}) = −½Σ ΔQ:ΔF + boundary terms. For the elastic part
the same term is a bounded oscillation (symplectic shadow energy). For a Maxwell branch, ΔQ ≈ bΔS⁰ is stiff and
in phase with ΔF, so −½Σ b ΔS⁰:ΔF accumulates. Its ratio to the true dissipation is ≈ dt·ω²·τ/2. That is negligible for
the 11.7 Hz mode, but O(1) for the high-frequency content the tap and the per-control-step velocity jumps excite
(largely volumetric waves, which the isochoric Prony terms damp only slowly). The drift scales with τ (τ 50 ms > τ 5 ms)
and halves with dt, as observed. The 1 Hz cycle-books miss (+1.17 %) is the same bias accumulated over a long, low-loss
cycle. The kick-paired W_kp = Σ pb_n:(ΔF_{n−1} + ΔF_n)/2 telescopes against the KE change and closes the books to ≤ 0.021 %
with no drift in every variant. The ADR §21.2.7 statement that the left-point W_b closes "at any strain up to the O(dt)
symplectic error" does not hold uniformly. The explicit dissipation D (layer M) and the dynamics are correct; only the
book is biased. **Not changed here.** Proposed: `visco_energy = branch_work_kick_average_v2` (bookkeeping only: store
ΔF_n per particle, add ½pb_n:(ΔF_{n−1} + ΔF_n)), declared with a protocol v2 before its run. Decision for rcs-xhigh.

## 4. GPU cost — RUBBER-VISC-10 (reported, not scored)

`flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp
PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/bench_visc_cost.py --out artifacts/rl_ready/p2_3/visc_cost_001`
(commit `0555131`, clean tree, protocol sha `95ad1f94…`). RTX 5090 GPU0, driver 570.211.01, Warp 1.17.0,
23:48:16–23:55:08 KST. **SHARED_GPU**: co-tenant PID 642659 (766 MB) present in every sample (not touched).
Workload E geometry (case A clamp 150 × 20 × 4 mm, 50 Hz control, gravity −z), P2 set nominal (μ₀ 0.4 MPa, β 20,
ρ 1150; elastic rows the same μ₀, κ, ρ), dt from the nominal material (same substeps for every variant of a
resolution), protocol options (F8 + kick-average + F9 + F10), guard 1.0, motion_table_v1 BC launch, CUDA graph chunks of
500 substeps, no cameras, no recording, 1 cold + 3 warm control steps (warm = median). The VRAM column is the process
peak, cumulative over the run (the n_t 3 rows ran after the n_t 4 rows, so their own peak is masked). Every row has
L2 invariants PASS (finite, invalid bits 0, guard violations 0). The registry state is 144 B per particle at M = 2,
plus 72 B with the energy option.

| variant | n_t | N | particles/env | substeps/control step | material dt | cold / warm wall per control step (s) | ms/substep (vs elastic) | env-steps/s | VRAM peak MB | visco state B/particle | L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| elastic | 4 | 1 | 97280 | 3446 | 5.80 µs | 1.133 / 1.060 | 0.307 | 0.944 | 818 | 0 B | PASS |
| P2_energy_off | 4 | 1 | 97280 | 3446 | 5.80 µs | 1.264 / 1.209 | 0.351 (+14 %) | 0.827 | 882 | 144 B | PASS |
| P2_energy_on | 4 | 1 | 97280 | 3446 | 5.80 µs | 2.064 / 2.001 | 0.581 (+89 %) | 0.500 | 924 | 216 B | PASS |
| elastic | 4 | 4 | 97280 | 3446 | 5.80 µs | 2.892 / 2.845 | 0.826 | 1.406 | 1052 | 0 B | PASS |
| P2_energy_off | 4 | 4 | 97280 | 3446 | 5.80 µs | 3.335 / 3.257 | 0.945 (+14 %) | 1.228 | 1244 | 144 B | PASS |
| P2_energy_on | 4 | 4 | 97280 | 3446 | 5.80 µs | 6.222 / 6.154 | 1.786 (+116 %) | 0.650 | 1308 | 216 B | PASS |
| elastic | 4 | 8 | 97280 | 3446 | 5.80 µs | 5.348 / 5.281 | 1.533 | 1.515 | 1468 | 0 B | PASS |
| P2_energy_off | 4 | 8 | 97280 | 3446 | 5.80 µs | 6.330 / 6.261 | 1.817 (+19 %) | 1.278 | 1692 | 144 B | PASS |
| P2_energy_on | 4 | 8 | 97280 | 3446 | 5.80 µs | 11.813 / 11.762 | 3.413 (+123 %) | 0.680 | 1852 | 216 B | PASS |
| elastic | 3 | 1 | 41220 | 2585 | 7.74 µs | 0.548 / 0.509 | 0.197 | 1.965 | ≤ 1852 (cum.) | 0 B | PASS |
| P2_energy_off | 3 | 1 | 41220 | 2585 | 7.74 µs | 0.598 / 0.555 | 0.215 (+9 %) | 1.802 | ≤ 1852 (cum.) | 144 B | PASS |
| P2_energy_on | 3 | 1 | 41220 | 2585 | 7.74 µs | 0.882 / 0.839 | 0.325 (+65 %) | 1.191 | ≤ 1852 (cum.) | 216 B | PASS |
| elastic | 3 | 4 | 41220 | 2585 | 7.74 µs | 1.067 / 1.006 | 0.389 | 3.977 | ≤ 1852 (cum.) | 0 B | PASS |
| P2_energy_off | 3 | 4 | 41220 | 2585 | 7.74 µs | 1.239 / 1.180 | 0.456 (+17 %) | 3.391 | ≤ 1852 (cum.) | 144 B | PASS |
| P2_energy_on | 3 | 4 | 41220 | 2585 | 7.74 µs | 2.216 / 2.158 | 0.835 (+115 %) | 1.853 | ≤ 1852 (cum.) | 216 B | PASS |
| elastic | 3 | 8 | 41220 | 2585 | 7.74 µs | 1.821 / 1.761 | 0.681 | 4.544 | ≤ 1852 (cum.) | 0 B | PASS |
| P2_energy_off | 3 | 8 | 41220 | 2585 | 7.74 µs | 2.105 / 2.050 | 0.793 (+16 %) | 3.902 | ≤ 1852 (cum.) | 144 B | PASS |
| P2_energy_on | 3 | 8 | 41220 | 2585 | 7.74 µs | 3.865 / 3.802 | 1.471 (+116 %) | 2.104 | ≤ 1852 (cum.) | 216 B | PASS |

Stage split (N 4, n_t 4, 50 eager substeps under CUDA activity timing; GPU kernel time per substep): elastic 0.910 ms
(stress pass incl. J-bar 0.108 ms), P2 energy off 1.038 ms (stress pass 0.209 ms: `stress_prony` 0.186 ms), P2 energy on
1.877 ms (`stress_prony` 0.518 ms with the per-particle float64 dissipation atomic, `visco_branch_work` 0.502 ms).
Reading: the Prony stress pass costs **+9 to +19 % per substep** at M = 2, matching the §21.2.9 estimate of +15–30 %.
`branch_work_v1` costs **+65 to +123 %**: the per-particle float64 atomics into one per-env accumulator and the extra
gradient pass. It is a verification option and not for training throughput; a fixed-order per-env reduction is the
obvious fix if it is ever needed at scale.

## 5. CPU test suites

`tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_batched_*.py` (rubber + batched, incl. the 4 new
visco files, 51 new tests): **379 passed, 0 skipped with `CUDA_VISIBLE_DEVICES=''`** (653.7 s, at `0555131`) and **379 passed, 0 skipped with `CUDA_VISIBLE_DEVICES=0`** (615.5 s, under the GPU0 lock, at `555d6fb`; src identical). New files: `test_rubber_visco_constitutive.py` (25),
`test_rubber_visco_solver.py` (11), `test_rubber_visco_batched.py` (13), `test_rubber_visco_tangent_check.py` (2). None
of them needs artifacts, torch or CUDA. GPU-only paths are exercised by the cost script, not by pytest.

## 6. Open items

- -03 and -05 v2 protocol decisions (§3.1, §3.2), and the `branch_work_kick_average_v2` option. rcs-xhigh decides; the v1
  FAILs stay the v1 results. **Decided and run: §8** (-03 / -05: v1 FAIL, v2 PASS (post-hoc measurement correction: the diagnostic re-analysis b9baa57 computed the v2 outcome on v1 data before v2 was declared in dca4848); the v1 FAILs remain the v1 results).
- The visco guard bound holds with 0.9 % margin, and its PSD premise is false. A guard v2 is needed only if a future
  state set exceeds 1.
- `visco_energy = branch_work_v1` uses one float64 atomic per particle-substep (dissipation) plus a branch-work pass
  (§4 cost). A fixed-order reduction would make it deterministic on GPU and cheaper.
- GPU runs of the strip / cantilever tests (allowed by the protocol) were not made: CPU only, NOT_RUN on GPU.
- The ADR numbering may collide with `claude/rl-rubber-v5` (§21.5).
- Independent xhigh review: NOT_RUN.

## 7. What P2-4 (contact / friction, ADR §11 D8) needs from here

- **Contact work in the books.** The table / fixture contact impulses (`_loads`, frozen `_contact_impulse`) are not
  accumulated into any work term. P2-4 needs a contact-work bookkeeping option, kick-paired like §21.1 and §3.2 (a
  projection-type contact is a constraint impulse). Otherwise no energy-based contact or friction test can close
  books, especially with Prony damping present.
- **Moving pads** (`friction_pads_v1`): kinematic boxes in the grip frame with the motion-table interpolation, reaction
  per pad `[N, G, B, 6]`, and per-env frictions (already in `params` columns 3–5, resampled per reset).
- **Declared protocol** (RUBBER-CONT): sliding block / Coulomb stick–slip vs analytic, normal penetration vs the 1/dt²
  projection stiffness (dt and dx refinement), and visco + contact (rolling / indentation hysteresis) separated from
  the material loss through the books.
- **Self-contact** stays off in v1 (§18.3). The fold scene (RUBBER-CONT-05) needs the grouped HashGrid path and
  its cost row.
- **The 20 ms coupling assessment** (§11) and per-env wrench isolation on GPU.
- The visco state reset and params64 staging are already generic. A contact model with per-particle state would reuse
  the registry-state reset.

## 8. Protocol v2 (ADR 0023 §21.7, rcs-xhigh): decision, implementation, formal run `visc_v2_001`

### 8.1 v1 results preserved

`visc_v1_001` (commit `555d6fb`) and `p2_3_visco_001.json` are unchanged. Their verdicts stand: RUBBER-VISC-03 **FAIL**
and RUBBER-VISC-05 **FAIL** under `rubber_visc_tests_v1`. v2 adds results under a separately declared protocol; it does
not replace them. -01, -02, -04, -06 … -10 are not re-scored (their v1 results stand).

### 8.2 Verification of the diagnosis (DIAGNOSTIC `visc_diag_v1_reanalysis_001`, commit `b9baa57`, reads v1 data only)

`PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/diag_visc_v1_reanalysis.py --out artifacts/rl_ready/p2_3/visc_diag_v1_reanalysis_001`

- **Estimator leakage: confirmed.** On exact steady sinusoids sampled on the protocol grid, the declared estimator
  alone gives E′ −0.249 / −0.687 % and E″ −0.303 / −0.823 % (16 / 32 Hz), and 0 at 1–8 Hz. K whole periods with an
  integer sample count (K = 2 at 16 Hz, 4 at 32 Hz, 625 samples), σ and ε mean-removed: ≤ 7e-16.
- **Inertia: confirmed; the factor is corrected.** The exact 1-D linear viscoelastic rod uses the real marker layers
  a = 7.25 mm, b = 22.25 mm (the nearest particle layers, not L/4 and 3L/4). It gives σ(L)/ε_g = E*·k(b − a)cos kL/(sin kb − sin ka),
  i.e. −1.453 % at 32 Hz; §3.1's −(3/8)ρω²L²/E = −1.476 % is the mid-span value. The rod model is checked on data by
  the grip-1 − grip-0 difference: −1.968 % observed vs −1.971 % predicted at 32 Hz, −0.528 / −0.530 % at 16 Hz.
- After both corrections, E′ is −0.402 … −0.414 % at every rate (the budgeted finite-strain model form). The
  identification on the corrected v1 data passes, and the held-out 32 Hz E′ is −0.08 %.
- **Energy books: confirmed.** The kick-paired W_kp is the conjugate of the discrete kinetic energy and equals the
  trapezoid Σ ½(pb_n + pb_{n+1}):ΔF_n of the branch's own work. In diag_002 it equals D + Ψ_v (computed from Q alone)
  to ≤ 1e-5 of S_N, while the left-point W_b misses D + Ψ_v by exactly its book drift (P2 −2.09 % vs B_N 2.08 %).

### 8.3 Decision per test (ADR 0023 §21.7.2)

| Test | v2? | What v2 corrects (a defect that biases the exact solution too) | Thresholds |
|---|---|---|---|
| RUBBER-VISC-03 | **justified** | (i) whole-period, mean-removed estimator over a window starting after 6 τ_max (16 / 32 Hz runs 2500 steps; 1–8 Hz keep their v1 length); (ii) de-embedding of the measured modulus with the exact 1-D rod transfer (ρ, L, marker layers, ω only; same procedure for fit and held-out rates), guarded by a grip-0 / grip-1 consistency validity bound e1_rel / 5 | all v1 values kept |
| RUBBER-VISC-05 | **justified** | `visco_energy = branch_work_kick_average_v2` (kick-paired branch work; v1 left-point sum reported alongside); cycle criteria over the whole-period window | all v1 values kept |

Rejected: widening e1_rel; an approximate (3/8)π² inertia budget; scoring grip 0 (still +0.51 % biased at 32 Hz).
No threshold was shown unattainable for the exact solution, so none moved (the protocol test asserts this).

### 8.4 Implementation

- `912b228` ADR §21.7. `dca4848` declares `rubber_visc_tests_v2.json` alone, SHA-pinned (`212fe7af…4663`), before any
  implementation or v2 run.
- `bc52a2d` option `branch_work_kick_average_v2` in `TLMPMRubberVisco` and the batched path. It adds
  `visco_df_prev[P]` (mat33d) and `visco_branch_work_left[N]`, the kernel `visco_branch_work_kick_average`, masked-reset
  kernels, and layout / snapshot / vars()-complete tables. `off_v1` and `branch_work_v1` add nothing.
  `rubber_visco_reference` gains `steady_window_v2`, `fourier_modulus_window_v2`, `rod_transfer` and `deembed_rod`.
  Tests (`tests/test_rubber_visco_energy_v2.py`, 17):
  - bitwise-off by default;
  - with moving grips, every array is byte-identical between off_v1, branch_work_v1 and v2;
  - the left-point sum equals v1's W_b byte-for-byte;
  - the kick-paired accumulator equals a host recompute every substep;
  - reset; batched L0 N 1 / N 3; masked reset; vars()-complete; mixed v1/v2 refused;
  - a closure property on a small tap;
  - estimator, rod transfer (PDE and small-kL expansion) and de-embedding round trip.
- `889a782` runner `run_visc_v2.py`. Smoke `visc_v2_smoke_001` (3 control steps, label SMOKE, identity with v1 on 3
  rows PASS) is preserved.

### 8.5 Formal run `visc_v2_001` (commit `889a782`, clean tree, Warp 1.17.0 CPU, 16 workers, 00:24:25–00:44:57 KST, 1232 s)

`CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/run_visc_v2.py
--workers 16 --out artifacts/rl_ready/p2_3/visc_v2_001`, protocol sha `212fe7af…`. Every configuration valid (no invalid
flag, 0 guard violations, 0 negative D increments). **Validity `dynamics_identity_with_v1`: PASS.** All 14 jobs
reproduce every recorded v1 column except `branch_work` byte-for-byte: the full v1 length for the 1/2/4/8 Hz strips,
creep, cantilevers and blocks, and the v1 prefix of 1875 / 1719 rows for the longer 16 / 32 Hz runs, the ×2 run and
the g = 0 run. Their left-point W_b equals the v1 `branch_work` column byte-for-byte.

**RUBBER-VISC-03: v1 FAIL, v2 PASS (post-hoc measurement correction: the diagnostic re-analysis b9baa57 computed the v2 outcome on v1 data before v2 was declared in dca4848)** (v2 criteria a, a-validity, b, c, e).

| Rate | window (periods / samples) | E′ de-embedded (1 %) | E″ (3 %) | loss / cycle (3 %) | grip 0 vs 1 E′ (0.2 %) | not de-embedded E′ | v1 estimator on the v2 series |
|---|---|---|---|---|---|---|---|
| 1 Hz (fit) | 1 / 5000 | −0.402 % | −0.435 % | −0.413 % | −0.0000 % | −0.404 % | −0.404 % |
| 2 Hz (held-out) | 1 / 2500 | −0.402 % | −0.443 % | −0.412 % | −0.0000 % | −0.410 % | −0.410 % |
| 4 Hz (fit) | 1 / 1250 | −0.403 % | −0.467 % | −0.413 % | −0.0000 % | −0.430 % | −0.430 % |
| 8 Hz (held-out) | 1 / 625 | −0.404 % | −0.511 % | −0.397 % | −0.0002 % | −0.507 % | −0.507 % |
| 16 Hz (fit) | 2 / 625 | −0.401 % | −0.625 % | −0.410 % | −0.0030 % | −0.792 % | −1.039 % |
| 32 Hz (held-out) | 4 / 625 | −0.414 % | −0.867 % | −0.434 % | −0.0020 % | −1.867 % | −2.543 % |

(b) Fit on 1 / 4 / 16 Hz: τ₁ −3.42 %, τ₂ −0.29 % (limit 10 %); g₁ +1.14 %, g₂ +0.07 % (limit 5 %). (c) Held-out E′ / E″:
2 Hz +0.036 / +0.043 %, 8 Hz +0.027 / −0.259 %, 32 Hz −0.097 / +1.90 % (limits 1 % / 4 %). (d) The single-rate 4 Hz
fit (reported) mispredicts E″ by −33 / +44 / −23 %. (e) Block 1.33e-7 / 1.39e-7 (limit 1e-4). De-embedding converged
in 3–8 iterations.

**RUBBER-VISC-05: v1 FAIL, v2 PASS (post-hoc measurement correction: the diagnostic re-analysis b9baa57 computed the v2 outcome on v1 data before v2 was declared in dca4848)** (v2 criteria a–d).

| Run | max \|B_N\|/S_N, v2 kick-paired (0.5 %) | v1 left-point on the same run (reported) | W_b − (D + Ψ_v) at end / S_N (reported) |
|---|---|---|---|
| strip 1 / 2 / 4 / 8 Hz | 0.0021 / 0.0022 / 0.0031 / 0.0049 % | 0.172 / 0.053 / 0.032 / 0.031 % | +7.5e-4 … +2.0e-3 |
| strip 16 / 32 Hz, 16 Hz ×2 | 0.0064 / 0.0072 / 0.0030 % | 0.117 / 0.224 / 0.020 % | +2.7e-3 / +3.2e-3 / +2.7e-3 |
| creep (-02 run) | 0.035 % | 0.157 % | +6.8e-4 |
| cantilever ×1 / ×2 / ×4 | **0.0121 / 0.0062 / 0.0031 %** | **2.576 / 1.054 / 0.474 %** (= v1) | −1.0e-5 / −3.1e-6 / −1.5e-6 |

(b) 16 Hz ×2 / ×1 = 0.476 (limit 0.65): first order. (c) Cycle books ΔW_ext/ΔW_b − 1 per window: −0.0077, −0.0022,
−0.0008, +0.0008, +0.0010, −0.0006 % (1 … 32 Hz; limit 0.5 %); with the v1 left-point W_b on the same windows it would be
**+1.166** (= v1), +0.265, +0.116, +0.075, +0.230, +0.251 %. (d) ΔD/ΔW_b − 1: −0.39 … −0.41 % (limit 2 %); 0 negative
increments in every run. (e) The g = 0 control at 16 Hz is 0.020 %, trend −4.4e-5 /s (reported). The branch identity is
exact in the small-strain cantilever (≤ 1e-5). In the strips it is +0.07 … +0.3 % of S_N, the O(ε) S⁰-strain vs
mechanical difference that §21.2.7 anticipates.

-02 / -04b recomputed on the v2 runs (reported): creep curve 0.28 / 0.11 %, ratio −0.079 %, recovery 0.004 %;
ζ ×2/×4 +2.1e-6 / +2.0e-6, f_d −7e-8 / −8e-8; vs the root f_d +0.076 %, ζ −0.29 %. These equal the v1 values, as the
byte identity implies.

### 8.6 CPU suites (commit `bc52a2d`; `artifacts/rl_ready/p2_3/cpu_tests_002`)

`tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_batched_*.py`: **397 passed, 0 skipped** with
`CUDA_VISIBLE_DEVICES=''` (619.7 s) and **397 passed, 0 skipped** with `CUDA_VISIBLE_DEVICES=0` under the GPU0 lock
(616.4 s).

### 8.7 Observations (reported, not scored) and open items

- **E″ and the wrench time stamp (DIAGNOSTIC, found after the run).** The de-embedded E″ falls from −0.435 % (1 Hz) to
  −0.867 % (32 Hz), while E′ and the energy-based loss stay at −0.40 … −0.43 %. The control-step wrench is the mean of
  the substep forces at t_{k−1} + j·dt, j = 0 … n − 1, centred dt/2 before the declared stamp t_k − Δt_c/2. Re-stamped
  by dt/2, the same `visc_v2_001` data give E″ −0.411, −0.411, −0.412, −0.395, −0.404, −0.419 %: flat, and equal to E′
  and the loss. v2 keeps the v1 stamps (stamps and thresholds never move after a result), and E″ passes with them. The
  next visco protocol that scores E″ adopts the dt/2-corrected stamp (review R3, ADR §21.7.7).
- The early window of the v1-data re-analysis carried +0.12 % in E″ at 16 / 32 Hz (−0.508 / −0.750 there vs
  −0.625 / −0.867 % here), as the exact transient analysis predicted (+0.13 %).
- The finite-strain model form (−0.40 % against the small-strain E*) remains inside the budget, not removed.
- Not run: GPU strip / cantilever runs. The independent rcs-review (xhigh) of the v2 decision and run has since run
  (`c9d771f`, PASS_WITH_LIMITATIONS); its follow-ups R1–R4 are in ADR §21.7.7.
- `branch_work_kick_average_v2` adds one mat33d per particle (72 B) to `branch_work_v1`'s arrays and is still a
  verification option (per-particle float64 atomics); it is not for training throughput.
- Material status unchanged: **SIMULATOR_ONLY_UNCALIBRATED**.
