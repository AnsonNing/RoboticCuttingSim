# P2-5 — Rubber calibration interface and synthetic numerical validation (ADR 0023 §23.2 decision 6, §23.6)

Branch `claude/rl-rubber-p25` (worktree `RoboticCuttingSim-rl-rubber-p25`, from `claude/rl-rubber` @ `77f5a99`).
Material status everywhere: **SIMULATOR_ONLY_UNCALIBRATED**. Every P2-5 output is **NUMERICAL_VALIDATION_SYNTHETIC**:
the "truth" is the simulator itself (same model family) at a finer resolution with sensor noise. **This is not a
calibration.** No real rubber data exists on this machine, no recipe is promoted, and no synthetic manifest can satisfy
the `REAL_CALIBRATED` gate. Same-family truth cannot reveal model-form error against real rubber; it tests
identifiability, the pipeline and the generalisation of the declared protocol under discretisation and noise.

- Protocol `configs/rubber/rubber_calib_tests_v1.json`, sha256 `1f6816411d141555973537995300ba87927a8162e8eba5f942d51faf40b2ff2b`,
  declared alone in `c5054b9` (before any pilot or formal run), pinned in `tests/test_rubber_p2_5_protocol.py` and the
  run scripts in `0e43c67`.
- Thresholds `configs/rubber/rubber_calib_thresholds_v1.json`, sha256 `40be2f8bb7053199ef6b29776e0a7f125dff350e224124c10b07e965182caa36`,
  committed alone in `4d7d2c4` before the formal dataset existed, pinned in `44dcc9c`.
- **Status: P5.1–P5.6 DONE.** The v1 formal fit was aborted for compute with no receipt (§5). The formal run under
  `rubber_calib_tests_v2` (§9–§10) gives CAL-04 / -05 / -06 FAIL and CAL-07 PASS. Evidence
  `docs/evidence/rl_ready/p2_5_calib_synthetic_001.json`. Raw outputs
  `artifacts/rl_ready/p2_5/` (not committed, never overwritten).
- Nothing frozen changed: `rubber_calibration.py` (v1) and the `REAL_CALIBRATED` block of `rubber_material.parse_recipe`
  are byte-pinned at the P2-5 base (`tests/test_rubber_calibration_v2.py`); no Phase 1 module, `configs/rl_benchmark/*`,
  `scripts/rl_ready/*`, rubber solver module or earlier rubber protocol was edited.
- K2 review conditions (claude/rl-rubber @ `60e9119`) folded into v1 **before** its declaration: friction scoped to
  translational sliding with law v3 (K2-1); every fitted θ labelled `RESOLUTION_SPECIFIC` to `rubber_ref_v1` with the
  Q1w bias (Q1w condition 4); the seal is procedural (stated, reads logged).

## 1. Interface (tasks P5.1–P5.3)

**`rubber_calibration_v2.py`** — schema `rcs_rubber_calibration_dataset_v2`:
- `data_origin` ∈ {`real_measurement`, `synthetic_simulator`}; per-experiment `sensors` (kind, σ, resolution, rate,
  full scale, unit); specimens with nominal and measured dimensions and the measured mass; labels
  `calibration_status` (always `SIMULATOR_ONLY_UNCALIBRATED`) and `validation_kind`; for synthetic data a `synthetic`
  block (generator commit + clean tree, truth profile incl. the oracle file, sensor model id, `sealed_truth_sha256`,
  `noise_seed_sha256`, protocol SHA, prior box).
- Origin-bound statuses: `PILOT` / `FROZEN_FOR_HELDOUT` real only; `SYNTHETIC_PILOT` / `SYNTHETIC_FROZEN_FOR_HELDOUT`
  synthetic only; `NO_REAL_DATA` empty real manifests.
- Freeze hash over the canonical manifest, every CSV and the oracle file; `write_manifest` refuses synthetic manifests
  in the REAL gate directory and never overwrites.
- **Closing the F10 hole without touching the v1 code.** The v1 gate reads only v1 manifests, so a v2 manifest is
  refused by schema in any directory, status or verdict (48 parametrised negative tests). The remaining path — the same
  synthetic data rewritten as a v1 manifest, which has no origin field — does pass the unchanged v1 gate (the test shows
  it) and is closed by a CI tripwire, `gate_directory_audit`: until a real-data protocol is declared, the gate
  directory may hold only v1 `NO_REAL_DATA` manifests. A second test asserts that no synthetic manifest exists in
  `configs/` outside `configs/rubber/calibration_synthetic/`.
- `real_calibrated_gate_v2`: the versioned gate for future real data (v2, `real_measurement`, `FROZEN_FOR_HELDOUT`,
  freeze hash checked, a first one-shot held-out PASS for that freeze); refuses every v1 and every synthetic manifest.
  No recipe uses it (real path NOT_RUN).

**`rubber_calibration_experiments.py`** — forward models of the five experiment types (scene, schedule, contact,
duration, observables, sensor transform, residuals, metrics M1–M9); population batching: (candidate θ_j, experiment k)
→ one env of a `BatchedRubberMaterial`, one batch per experiment, built once and re-used through the masked reset
(`materials=`). A mixin replaces only the batch's randomisation-range check so that every Prony term can be set
independently (Prony arity and CFL still checked). Candidates with the same physics-relevant material share one env
(μ_table acts only with contact).

**`rubber_calibration_fit.py`** — `ParameterSpace` (log μ_eq, stick-breaking logit g with budget G = 0.5 so μ0 ≤
0.8 MPa = the step material, log τ, linear μ_table); `Objective` (fit split only; no RL term: a `reward…` argument
raises `REWARD_TERM_REFUSED`; held-out ids `HELDOUT_EXPERIMENT_REFUSED`; sealed paths `SEALED_PATH_REFUSED`; v1
manifests refused); `fit_trf` / `fit_trf_lockstep` (scipy TRF; speculative forward-difference Jacobian = one population
of p + 1 per evaluation; starts in lockstep, bitwise equal to sequential starts in a CPU test); `covariance` (eigen-
decomposition; unidentifiable directions get infinite variance, never a pseudo-inverse's zero); `CMAES` and
`run_ask_tell` (the plug-in point of the external agentic optimizer or BO); `freeze_fit` (receipt with the
`RESOLUTION_SPECIFIC` label); `evaluate_heldout` (one-shot ledger per manifest freeze, `HELDOUT_ALREADY_EVALUATED`,
named and labelled audit re-runs, exclusive create).

**`rubber_calibration_synthetic.py`** — secret seeds (`os.urandom`), the prior draw, measured specimens, noise
realisations, the dataset writer, `seal` (owner-only directory, read-only file, exclusive create) and `reveal` (SHA
check, every read logged).

## 2. Protocol `rubber_calib_tests_v1` (task P5.4)

**Models.** `neo_hookean_jbar_prony_v1`, M = 2, β 20; options F8 + kick-average + F9 + F10, `visco_energy`
`branch_work_kick_average_v2`, guard 1.0; contact law `grid_node_penalty_coulomb_v3` (friction rows only). Fit model
`rubber_ref_v1` n_t 4, CFL 0.5; truth n_t 6, CFL 0.25. dt from the step material (μ0 ≤ 0.8 MPa, ρ ≥ 1000 kg/m³) for
every candidate, so the objective is smooth in θ: 262 substeps per 1 ms control step at dx 1 mm (fit, t 4 mm), 349 at
dx 0.75 mm (fit, t 3 mm), 784 / 1046 for the truth. Gravity acts as a ramped body traction (solver gravity 0).

**Parameters.** θ = (μ_eq, g1, τ1, g2, τ2, μ_table); fit box μ_eq [1.5, 4]·10⁵ Pa, g1 [0.02, 0.4], g2 = (0.5 − g1)·
[0.02, 0.98], τ1 [2, 30] ms, τ2 [30, 500] ms, μ_table [0.3, 0.8]. Hidden-truth prior box μ_eq [2, 3.5]·10⁵ Pa, g1
[0.08, 0.2], g2 [0.06, 0.15], τ1 [5, 15] ms, τ2 [60, 200] ms, μ_table [0.4, 0.6], ρ [1100, 1200] kg/m³. Density is
measured (mass / measured volume), β is numerical.

**Experiments** (control step 1 ms):

| Id | Type | Specimen (mm) | Loading | Split |
|---|---|---|---|---|
| T1 / T2 | tensile load-unload, case C | 60 × 12 × 4 | λ 1 → 1.5 → 1.1 at ε̇ 1 /s and 8 /s | fit |
| R1 | stress relaxation | 60 × 12 × 4 | ramp to λ 1.3 in 20 ms, hold 0.3 s | fit |
| C1 | cantilever rebound, case A | 42 × 15 × 4 | weight + tip load 13.5 mN ramped 0.3 s, released at 0.4 s, 0.4 s of video | fit |
| G1 | gravity sag | 42 × 15 × 4 | weight ramp 0.3 s, hold to 0.4 s | fit |
| F1 | friction sliding (free block on the table) | 30 × 16 × 4 | weight; back-face push ramp to 0.85 × weight over 0.2 s, released | fit |
| T3 / T4 | tensile | 60 × 12 × 4 | ε̇ 3 /s (interpolation), 20 /s (extrapolation) | held-out rate |
| T5 | tensile | 60 × 12 × **3** | ε̇ 3 /s | held-out thickness |
| C2 | cantilever | **51** × 15 × 4 | as C1 | held-out length |
| C3 | cantilever | 42 × 15 × **3** | as C1 | held-out thickness |
| C4 | cantilever | 42 × 15 × 4 | **2 ×** tip load | held-out initial deformation |
| F2 | friction sliding | 30 × 16 × 4 | top load 1 × weight (**2 × normal load**), push ramp **0.15 s** | held-out load / rate |

**Sensors** (`rubber_calib_sensors_v1`): load cell 50 N FS, σ 0.15 N (0.3 % FS), 1 mN resolution, 1 kHz (control-step
mean of the grip reaction); crosshead σ 1 µm, 1 µm resolution; video 240 fps, σ 0.1 mm (the 9 cross-section centroids
by linear interpolation between control steps); force plate 0.1 N FS, σ 0.3 mN, 1 mN resolution, 1 kHz; specimen
dimensions σ 0.02 mm, mass σ 0.5 mg.

**Tests.** CAL-01 interface and refusals (CPU), CAL-02 forward-model parity (CPU, byte-wise), CAL-03 truth receipts,
CAL-04 χ²_red(θ̂) ≤ 1.5 χ²_pilot, CAL-05 held-out (primary): every (experiment, metric) ≤ tol_mk, CAL-06 recovery
after the reveal, CAL-07 negative control (fit on T2 + C1 must FAIL CAL-05 on a viscous metric), CAL-08 order and
sealing, CAL-09 labels. Tolerance rule tol_mk = max(2 (a_mk + b_mk), floor_m) from a separate pilot (a: two noise
realisations; b: noise-free truth vs the fit model at θ_pilot).

**Deviations from the plan's matrix (declared in the protocol, with reasons).** Tensile width 12 mm and cantilever
42 × 15 mm / held-out 51 mm (particle lattice of n_t 4 / 6 at t 3 and 4 mm; a flat 80 × 4 mm strip has f1 ≈ 3 Hz and a
≈ 49 mm linear sag, so 0.4 s of video would not hold three peaks); the initial deformation and the gravity sag by a
simulated load phase (the static initializer is single-env and has no relaxed Prony state); a force-controlled push
instead of a speed-controlled pusher (slip distance and onset depend on μ; no contact frame); tensile unloading to
λ 1.1 (the strip stays in tension, minimum ≥ 9 % of the peak over the prior box in a 1D Prony check); M5 midpoint
definition, M7 onset threshold 0.5 mm with a 2 ms floor, M8 as RMS; CMA-ES budget 4 × 7.

**Design probes (GPU0 / GPU1, fit model at three prior-box points, not results).** T2 force 12–18 N with the unloading
minimum 1.8–4.3 N; F1 slip 1.9–15 mm and onset 215–263 ms over μ 0.6–0.4, friction force = μ N; C1 rebound with three
peaks (log decrement 0.39–0.55); a 63 mm cantilever with a 0.15 s ramp was still swinging at release (first extremum
downwards), which set the 51 mm length and the 0.30 s ramp before the declaration; no invalid bit anywhere.

**K2-1 (friction) and Q1w (resolution).** Law v2 has no formal row at n_t 4 / 6 (only n_t 3), so P2-5 keeps law v3
and scopes the friction experiments to straight-line translational sliding: a free, symmetric block, one contact region
(the table plane), a uniform back-face push through its symmetry plane. A declared scope check (lateral centroid drift
and change of the markers' y spread ≤ 0.1 mm) is reported for every truth friction row. The fitted μ_table is a
translational-sliding coefficient of law v3 only and must not be used for spin, sideways drag or multi-region contact.
Every fitted θ is `RESOLUTION_SPECIFIC` to `rubber_ref_v1` (n_t 4): n_t 4 carries the measured wide-strip sag bias Q1w
+0.62 % (+0.50 … +0.78 %), and the n_t 6 truth is itself +0.21 %, so the discretisation floor b_mk captures only about
0.4 % of that bias. θ̂ must never be carried to n_t 3 or 6 without the bias shift.

## 3. Pilot (P5.4) — `pilot_gen_001` + `pilot_eval_001` (GPU0 RTX 5090, SHARED_GPU)

- Generation `gen_synthetic.py --kind pilot` at `54a8f06` (clean), 2026-11-11 11:04–12:08 UTC, 3,839 s; 13 truth
  experiments at n_t 6 / CFL 0.25 (51,840–138,240 particles, 784 / 1,046 substeps per 1 ms; T1 371 s … C3 986 s);
  every truth env valid (invalid bits 0). Dataset `configs/rubber/calibration_synthetic/p2_5_pilot_001/`
  (`SYNTHETIC_PILOT`, committed `f042d34`), sealed truth sha `019e8441…`, realisation B in the run directory.
- Evaluation `run_pilot.py` at `f042d34` (clean), 583 s for the fit model at θ_pilot (13 experiments, P = 1).
  **χ²_pilot (reduced, fit split) = 1.2732** (per experiment mean square: C1 1.06, F1 1.66, G1 1.12,
  R1 1.47, T1 1.05, T2 1.77) → CAL-04 limit 1.5 × χ²_pilot = **1.9097**. The fit model at the true
  parameters reproduces the finer truth to near the noise level; the largest discretisation share is in T2 / F1.
- Tolerances tol = max(2 (a + b), floor):

| Held-out | Metric | a (noise) | b (discretisation) | floor | tol |
|---|---|---|---|---|---|
| C2 | M4 | 0.01406 | 0.01172 | 0.01 | **0.05158** (2(a+b)) |
| C2 | M5 | 0.08561 | 0.1679 | 2 | **2** (floor) |
| C2 | M6 | 0.03034 | 0.001289 | 0.005 | **0.06326** (2(a+b)) |
| C2 | M8 | 0 | 2.43e-07 | 0.001 | **0.001** (floor) |
| C2 | M9 | 0 | 0.001004 | 0.02 | **0.02** (floor) |
| C3 | M4 | 0.01007 | 0.001405 | 0.01 | **0.02295** (2(a+b)) |
| C3 | M5 | 0.09293 | 0.01362 | 2 | **2** (floor) |
| C3 | M6 | 0.01338 | 0.0004509 | 0.005 | **0.02766** (2(a+b)) |
| C3 | M8 | 0 | 1.705e-07 | 0.001 | **0.001** (floor) |
| C3 | M9 | 0 | 0.0002839 | 0.02 | **0.02** (floor) |
| C4 | M4 | 0.0123 | 0.001817 | 0.01 | **0.02824** (2(a+b)) |
| C4 | M5 | 0.0698 | 0.008658 | 2 | **2** (floor) |
| C4 | M6 | 0.01477 | 0.0004237 | 0.005 | **0.03039** (2(a+b)) |
| C4 | M8 | 0 | 5.339e-07 | 0.001 | **0.001** (floor) |
| C4 | M9 | 0 | 0.00138 | 0.02 | **0.02** (floor) |
| F2 | M7_onset | 4.007 | 0.1688 | 2 | **8.351** (2(a+b)) |
| F2 | M7_slip | 0.0166 | 0.009964 | 0.02 | **0.05314** (2(a+b)) |
| F2 | M8 | 0 | 1.501e-07 | 0.001 | **0.001** (floor) |
| F2 | M9 | 0 | 0.00865 | 0.02 | **0.02** (floor) |
| T3 | M1 | 0.0122 | 0.003378 | 0.005 | **0.03116** (2(a+b)) |
| T3 | M2 | 0.023 | 0.0001855 | 0.01 | **0.04637** (2(a+b)) |
| T3 | M8 | 0 | 2.158e-05 | 0.001 | **0.001** (floor) |
| T3 | M9 | 0 | 0.0002986 | 0.02 | **0.02** (floor) |
| T4 | M1 | 0.01084 | 0.01104 | 0.005 | **0.04376** (2(a+b)) |
| T4 | M2 | 0.02171 | 0.01619 | 0.01 | **0.07579** (2(a+b)) |
| T4 | M8 | 0 | 0.0002122 | 0.001 | **0.001** (floor) |
| T4 | M9 | 0 | 0.003008 | 0.02 | **0.02** (floor) |
| T5 | M1 | 0.01606 | 0.002246 | 0.005 | **0.03661** (2(a+b)) |
| T5 | M2 | 0.01335 | 0.0001036 | 0.01 | **0.02691** (2(a+b)) |
| T5 | M8 | 0 | 1.84e-05 | 0.001 | **0.001** (floor) |
| T5 | M9 | 0 | 0.0001807 | 0.02 | **0.02** (floor) |

## 4. Formal dataset (P5.5) — `formal_gen_001` (GPU0, SHARED_GPU)

> **[Erratum E3, P5.7 review C3]** Sections 4–9 are the **pre-v2 snapshot** (written before the formal v2 run). The current state is section 10 and the errata in section 12.

`gen_synthetic.py --kind formal` at `44dcc9c` (clean, after the thresholds commit), 2026-11-11 14:14:55–15:19:09 UTC,
3,854 s; every truth env valid; translational-scope check of F1 / F2: lateral centroid drift ≤ 2.9e-10 m, change of
the markers' y spread ≤ 6.7e-9 m (≤ 1e-4 m declared). Manifest `configs/rubber/calibration_synthetic/p2_5_formal_001/
manifest.json`, **`SYNTHETIC_FROZEN_FOR_HELDOUT`**, freeze sha256 `3d37a95070c07776f56cef18dffe54c7a94f4d69c0ecad67f5106211a3f689c9`,
sealed truth sha256 `e5b5c7183199493223ea52089ee4c6688b71fd57b623ba462ae1de980fcd9899` (committed `a0250bb`). The
sealed truth (`artifacts/rl_ready/p2_5/formal_gen_001/sealed/`, mode 0700 / 0400) has **never been read**
(no `read_log.txt`); the one-shot held-out ledger has **not** been consumed. The dataset stays valid for a resumed fit.
> **[Erratum E3]** Superseded: the sealed truth was read once, at the reveal on 2026-11-12T00:47:30Z, after the one-shot held-out (§10). The ledger is consumed. The dataset is **no longer valid for any blind fit**; a re-fit on it would be post-hoc.

## 5. Formal fit (P5.6) — stopped, no receipt

> **[Erratum E3, P5.7 review C3]** Sections 4–9 are the **pre-v2 snapshot** (written before the formal v2 run). The current state is section 10 and the errata in section 12.

`run_fit.py` at `a0250bb` (clean), `--kind formal` on GPU0 and `--kind negative_control` on GPU1, both TRF with the 3
declared Latin-hypercube starts in lockstep (population 21; 18 envs per non-contact experiment after the dedup).

| Run | Rounds done | Round time | Best cost per start (eval) | Trend | Stopped |
|---|---|---|---|---|---|
| formal (T1, T2, R1, C1, G1, F1; ≈ 2,380 residuals) | 7 | ≈ 28 min (GPU0) | 3.24e4 / 4.58e4 / 3.79e4 (7) | −2 … −13 % per round; expected floor ≈ 1.5e3 (χ²_pilot) | 18:30 UTC, 3.3 GPU-h |
| negative control (T2 + C1; ≈ 500 residuals) | 12 | ≈ 11 min (GPU1) | 3.47e2 / 1.87e3 / 4.19e2 (12) | still −5 % per round | 18:36 UTC, 2.3 GPU-h |

Why stopped (agent decision, recorded in `artifacts/rl_ready/p2_5/fit_001/*/ABORTED.json` with every evaluation):
TRF needs many more evaluations than estimated. Extrapolated, the formal fit would hit the declared budget
(max_nfev 40) after about 15 more GPU0-hours. That is about 6× the plan's 2.5 GPU-h for P5.6, and P2-6 jobs were
queued on GPU0 the whole time. The P1-6 runner (priority on GPU1 per the plan) had waited 2 h 16 min for GPU1. No
receipt was written, nothing was revealed and no held-out evaluation ran, so CAL-04…CAL-07 are **NOT_RUN** (not
FAIL). Both logs are preserved.

Observations (diagnostic, not verdicts):
- **The negative control converges.** By round 12 its best start is at a reduced χ² of about 1.4 (cost 347, ≈ 500 residuals). A slow but monotone
  TRF is therefore expected on this problem class (oscillating cantilever records, stick–slip friction onset).
- **The formal fit drifts.** Starts 0 / 1 move toward τ2 ≈ 0.44 s and g1 ≈ 0.02–0.04, outside the prior box, with small
  gains per round. This fits a poorly conditioned or locally non-convex landscape (phase of a 4-period cantilever
  record, the friction onset) and/or a noisy forward-difference Jacobian. GPU float atomics make repeated GPU forward
  calls non-bitwise, and the FD step 1e-3 in u is small against that noise; this is not yet measured.

**Recommended next step (decision for the coordinator / K-review, not taken here).** Option (a): resume the formal fit
unchanged as `fit_002` (same protocol; the held-out ledger and the seal are intact) with an explicit allowance of about
10–15 GPU0-hours, then the negative control (about 3 GPU-h) and the held-out / reveal (about 0.3 h). Option (b): declare
`rubber_calib_tests_v2` (v1 kept pinned) before any fit receipt, changing only the fit method. Candidates: measure GPU
repeatability first, a larger or central FD step, a staged start (T / R / G, then C / F), or `x_scale="jac"`. The
thresholds and the data stay frozen. Under v2 the order audit must record that the v1 fits were stopped unseen.

## 6. CAL status

> **[Erratum E3, P5.7 review C3]** Sections 4–9 are the **pre-v2 snapshot** (written before the formal v2 run). The current state is section 10 and the errata in section 12.

| Test | Status | Evidence |
|---|---|---|
| CAL-01 interface and refusals | **PASS** (CPU, `CUDA_VISIBLE_DEVICES=''`); `=0` run NOT_RUN | `tests/test_rubber_calibration_v2.py` (40), `tests/test_rubber_calibration_fit.py` (13), `tests/test_rubber_p2_5_protocol.py` (7) |
| CAL-02 forward-model parity | **PASS** (CPU, byte-wise for T, R, C, G, F; re-used batch; repeated objective calls); `=0` run NOT_RUN | `tests/test_rubber_calibration_experiments.py` (11) |
| CAL-03 truth receipts | **PASS** (pilot and formal: truth profile, generator commit + clean tree, secret seeds only in `sealed/`, sealed SHA and noise-seed SHA in the manifest, freeze hash) | `artifacts/rl_ready/p2_5/{pilot,formal}_gen_001/gen_receipt.json` |
| CAL-04 fit quality | **NOT_RUN** (fit stopped before termination) | §5 |
| CAL-05 held-out (primary) | **NOT_RUN** | — |
| CAL-06 recovery | **NOT_RUN** (truth sealed, unread) | — |
| CAL-07 negative control | **NOT_RUN** (fit stopped; no held-out) | §5 |
| CAL-08 order and sealing | so far **in order**: protocol `c5054b9` 10:57:19 UTC < pilot run 11:04–12:08 < pilot evaluation end 14:14:21 < thresholds `4d7d2c4` 14:14:35 < formal generation 14:14:55–15:19:09 < manifest commit `a0250bb` 15:19:30; fit receipt, held-out and reveal still to come | git log, receipts |
| CAL-09 labels | **PASS** so far (every manifest, receipt and abort record) | — |

CPU suites (`tests/test_rubber_*.py tests/test_batched_*.py tests/test_repo_naming.py tests/test_wp12_phase5_frozen.py`) at
`40d3c29` with `CUDA_VISIBLE_DEVICES=''`: **593 passed**, 2 warnings (scipy IntegrationWarning, as before), 798 s;
log `artifacts/rl_ready/p2_5/cpu_tests_001/cuda_none.log`. The `=0` repetition is **NOT_RUN**: GPU0 was held by queued P2-6 jobs.

## 7. GPU use (SHARED_GPU throughout; co-tenant PID 642659 present and untouched)

> **[Erratum E3, P5.7 review C3]** Sections 4–9 are the **pre-v2 snapshot** (written before the formal v2 run). The current state is section 10 and the errata in section 12.

Design probes ≈ 0.4 h (GPU0 / GPU1), pilot generation 1.07 h, pilot evaluation 0.16 h, formal generation 1.07 h, formal
fit 3.3 h (GPU0, stopped), negative control 2.3 h (GPU1, stopped): ≈ 8.3 GPU-h, above the plan's ≈ 5 h for P5.4–P5.6.
Measured forward costs (fit model, GPU, M = 2, F8 + F9 + F10, 1 ms control step): P = 7, 23,040 particles per env,
262 substeps: 229 ms per control step (1.8e8 particle-substeps/s); P = 21: 569 ms (2.2e8); case-A P = 14, 20,160
particles: 513 ms (1.4e8); truth single env 77,760 particles, 784 substeps: 412 ms.

## 8. What P2-5 does and does not show

> **[Erratum E3, P5.7 review C3]** Sections 4–9 are the **pre-v2 snapshot** (written before the formal v2 run). The current state is section 10 and the errata in section 12.

It delivers a tested calibration interface: manifest v2 with data origin, a closed REAL gate, population-batched
forward models that are byte-identical to the single references on CPU, TRF / CMA-ES / ask-tell fit API, one-shot
held-out and sealing. It delivers a declared protocol, pilot tolerances and a frozen, sealed formal dataset. It does
**not** yet deliver the formal fit, the held-out verdict, the recovery or the negative control. Even when completed, a
same-family synthetic truth says nothing about model-form error against real rubber; the real-data protocol
`rubber_calib_tests_real_v1` remains declared NOT_RUN. Material status: SIMULATOR_ONLY_UNCALIBRATED.

## 9. v2 method development (orchestrator decision option (b), 2026-11-12; PILOT data only)

> **[Erratum E3, P5.7 review C3]** Sections 4–9 are the **pre-v2 snapshot** (written before the formal v2 run). The current state is section 10 and the errata in section 12.

`fit_001` is recorded as **"v1 formal fit aborted for compute, no receipt"**; logs and abort records preserved, nothing
deleted. Per the decision, `rubber_calib_tests_v2` may change only the fit method and add a compute budget, and is
declared only after the method is validated end-to-end on the pilot. Development ran on GPU0 only under the blocking
lock (SHARED_GPU), pilot data only; the formal dataset and its sealed truth were not read or used. Development GPU
time: repeatability 1,527 s + pilot validation 4,369 s + pilot held-out diagnostic ≈ 420 s ≈ **1.75 GPU-h** (cap 2).

**9.1 GPU repeatability** (`v2_repeatability.py`, θ_pilot, pilot measured densities, 3 identical envs × 2 runs per
fit experiment; `artifacts/rl_ready/p2_5/v2_dev/repeat_default/repeatability.json`).
- Warp 1.17 `DeterministicMode.RUN_TO_RUN` is **not usable**: global mode refuses to compile the frozen Phase 1 kernel
  `tlmpm_v5._particle_contacts` (mixed add / max atomics on one array). No cheap deterministic mode exists without
  touching frozen code.
- Default mode, rms spread of the weighted residuals across copies (σ units) vs the rms forward-difference signal at
  h = 1e-3 in u:

| Exp | spread rms | weakest relevant FD signal (param) | strongest |
|---|---|---|---|
| T1 | 5.3e-6 | 3.7e-4 (τ1) | 7.4e-2 (μ_eq) |
| T2 | 2.3e-6 | 2.4e-3 (τ1) | 1.0e-1 (μ_eq) |
| R1 | 3.1e-6 | 1.2e-3 (τ1) | 1.1e-1 (μ_eq) |
| C1 | 8.3e-7 | 1.5e-3 (τ2) | 4.9e-2 (μ_eq) |
| G1 | 1.8e-7 | 1.3e-4 (g1, τ1) | 1.6e-2 (μ_eq) |
| F1 | **3.9e-4** | **7.4e-4 (τ1), 8.2e-4 (τ2)** | 6.9e-1 (μ_table) |

  Only the contact experiment F1 is noisy (stick/slip of contact nodes under float atomics). At h = 1e-3 its τ columns
  have a signal-to-noise ratio of about 2. **FD rule chosen: h = 1e-2 in u**, which gives ≥ 19 for F1's τ columns and
  ≥ 100 everywhere else (the signals scale linearly with h, the noise does not).

**9.2 Why v1 stalled (and why "the pilot converged" is not the comparison).** No pilot *fit* was ever run in v1.
χ²_pilot = 1.27 is the fit model evaluated *at* the true pilot θ, not a converged fit. The v1 formal TRF stalled for
two measured reasons:
- `x_scale = 1` with Jacobian columns spanning about 200× (μ_eq vs τ1): the spherical trust region in u is limited by
  the stiff μ_eq direction.
- Starts 20–400× above the floor, with F1's τ columns noise-dominated at h = 1e-3.

GPU noise in T / R / C / G is negligible.

**9.3 Staged-fit validation on the pilot** (`run_fit_v2.py --kind pilot_validation`, commit `7f23ea0`, run
`artifacts/rl_ready/p2_5/v2_dev/pilot_validation_001/`).
- Method: S1 μ_eq from T1 → S2 Prony (4) from R1 + T2, 3 lockstep starts → S3 μ_table from F1 → S4 joint polish.
  TRF with prior-box bounds, `x_scale="jac"`, h 1e-2, start at the prior-box centre, budget 4,800 s.
- Stage costs:

| Stage | Best cost | Wall |
|---|---|---|
| S1 (8 evaluations) | 743.5 | 663 s |
| S2 (15 rounds; all 3 starts converge to the same basin) | 233.1 | 2,752 s |
| S3 | 294.8 | 248 s |
| S4 (budget: 1 evaluation) | 1,388.6 on the whole split | 706 s |

  Total 4,369 s; reduced χ² **1.170**, below χ²_pilot 1.273 at the true θ.
- **Recovery against θ_pilot (v1 CAL-06 rule): FAIL.**
  - μ_eq +0.47 % and μ_table +0.27 %: PASS.
  - g1 −15.4 %, τ1 −10.8 %, g2 +8.4 %, τ2 −14.5 % with σ̂ ≈ 1 %: FAIL.
  - Because χ²(θ̂) < χ²(θ_pilot), the fit-model (n_t 4) optimum is itself ≈ 10 % away from the n_t 6 truth in the
    Prony parameters. This is resolution bias absorbed by the viscous parameters (the Q1w condition-4 effect, but in
    g / τ), not a convergence failure; more polishing lowers χ² further and does not move toward θ_pilot.
> **[Erratum E1, P5.7 review C1]** "Not a convergence failure" and "more polishing … does not move toward θ_pilot" are **UNMEASURED**. The pilot S4 ran 1 evaluation, and its S2 stopped at the 15-evaluation cap with two starts still falling 10–15 % per evaluation. That a resolution bias exists is supported (χ²(θ̂) 1.170 < χ²(θ_pilot) 1.273). Its size ("≈ 10 %") comes from an **unconverged** fit and is unmeasured.
- **Held-out diagnostic on the pilot's own data** (circular: the thresholds come from this pilot), at θ̂:
  - 27 / 31 PASS.
  - FAIL: T4 M2 0.104 > 0.076 and T4 M9 0.072 > 0.02 (20 /s rate extrapolation); C2 M9 0.0201 and C3 M9 0.0231 > 0.02
    (dissipated energy).

**9.4 Status.** The staged optimiser is validated as an optimiser: χ² below the truth's in 73 min. The coordinator's
> **[Erratum E3]** Superseded: the staged procedure is **not validated**. Its formal v2 run FAILed CAL-04 (§10). The pilot run reached χ² below the truth's only in its truncated form (S4 had 1 evaluation).
validation condition ("recovers pilot θ within the v1 recovery criteria") is **not met**, for a reason no fit method
can remove under the frozen CAL definitions. `rubber_calib_tests_v2` was therefore **not declared**; a draft is
kept at `artifacts/rl_ready/p2_5/drafts/rubber_calib_tests_v2.json`. No formal v2 fit ran; the formal one-shot ledger and the seal are intact.
Decision needed:
- **(a)** Declare v2 as drafted (h 1e-2, staged, budget 3 h / 1.5 h) and run. Expected: CAL-04 PASS; CAL-06 FAIL on the
  Prony parameters; CAL-05 likely FAIL on T4 (extrapolation) and M9 (an honest negative result about
  resolution-specific identifiability).
- **(b)** A reviewer-level change of the CAL-05 / CAL-06 definitions, outside the scope allowed for v2. For example:
  recovery against the fit model's own noise-free optimum instead of θ*, or tolerances whose b term uses a pilot *fit*
  rather than θ_pilot.

## 10. Formal run under `rubber_calib_tests_v2` (2026-11-12)

**Headline: the pipeline ran end-to-end as pre-registered, but the formal fit did not reach the fit model's optimum
within the declared method and budget (CAL-04 FAIL, χ²_red 2.484 > 1.910). Held-out prediction FAILs (21 / 31) and
θ recovery FAILs for g1, g2 and τ2. The negative control has power (CAL-07 PASS).** A supplementary diagnostic, run
> **[Erratum E4, P5.7 review C4]** "The negative control has power" is too strong: CAL-07 PASSes on **one** viscous row (T5 M2). Discrimination between an adequate and a deficient fit is **not shown** in this run, because both fits score 21 / 31.
after the reveal and never scored, shows that on the formal data the fit model at θ* has χ²_red 1.416 and passes all
31 held-out tolerances. The formal failure is therefore an **optimisation shortfall** (the staged fit stopped in a
worse valley of the Prony parameters), not the resolution bias the pilot showed and the declaration predicted. The
pilot result (§9: the n_t 4 optimum lies ≈ 10 % from the n_t 6 truth in the Prony parameters) still stands. Every
> **[Erratum E1, P5.7 review C1]** The mechanism words in this section ("worse valley", "wrong / different basin", "did not leave that basin", "the S2 data choice and starts") are **withdrawn**. The measured facts are in §12.1. The pilot's "≈ 10 %" is unmeasured (from an unconverged fit).
fitted θ is resolution-specific. Labels unchanged: SIMULATOR_ONLY_UNCALIBRATED, NUMERICAL_VALIDATION_SYNTHETIC.

- **Declaration.** `rubber_calib_tests_v2` was committed alone in `4218dd6` (sha256 `1606457f…01ff`) and pinned in
  `dfd2059`, before any v2 receipt.
  - It changes only the fit procedure (staged bounded TRF, x_scale jac, FD step 1e-2) and the budget: 3 h for the
    formal fit and 1.5 h for the negative control, counted on the run's own GPU0-lock hold time with the lock taken
    per evaluation.
  - It records the corrected run order and the pre-registered predictions: CAL-04 PASS; CAL-06 FAIL on the Prony
    parameters; CAL-05 at risk on T4 / M9.
  - v1, the thresholds, the manifest freeze and the sealed truth are unchanged.
- **Formal fit** (`fit_v2_001/formal`, `run_fit_v2.py` at `dfd2059` clean; receipt sha256 `617d65f3…24d7`, commit
  `ab4faf7`).
  - GPU0: 36 lock acquisitions, 7,737 s hold (budget 10,800 s, not hit), 858 s waiting, 8,595 s wall.
  - Stage best costs: S1 1,665, S2 1,943, S3 296, S4 2,948 (S4 ended at max_nfev 6, status 0).
  - θ̂: μ_eq 3.230e5 Pa, g1 0.0942, τ1 14.3 ms, g2 0.1859, τ2 72.3 ms, μ_table 0.4501.
- **Negative control** (T2 + C1 only; receipt `3446a6d2…137f`, commit `ad4c5aa`): χ²_red 1.314 on its own data;
  2,285 s hold, budget 5,400 s not hit.
- **One-shot held-out** (`run_heldout.py`, ledger `fit_v2_001/heldout/heldout_3d37a950….json`, label ONE_SHOT, both
  receipts in one call).
- **Reveal** at 2026-11-12T00:47:30Z: SHA verified, noise-seed SHA verified, the first and only read of the sealed
  truth (`read_log.txt`).

**CAL results**

| Test | Verdict | Prediction | Basis |
|---|---|---|---|
| CAL-01 | **PASS** | – | 599 passed in both CUDA modes at `dfd2059` (the P2-5 suites included) |
| CAL-02 | **PASS** | – | byte-wise population = single for T / R / C / G / F, repeated calls |
| CAL-03 | **PASS** | – | truth receipts complete, every truth env valid |
| CAL-04 | **FAIL** | PASS (missed) | χ²_red 2.4838 > 1.9097 (the S4 termination clause was met: max_nfev) |
| CAL-05 | **FAIL** | at risk on T4 / M9 | 21 / 31; failures below |
| CAL-06 | **FAIL** | FAIL on g1 / τ1 / g2 / τ2 | μ_eq, τ1, μ_table PASS; g1, g2, τ2 FAIL |
| CAL-07 | **PASS** | – | the negative control fails T5 M2 (viscous), plus M9 / F2 rows |
| CAL-08 | **PASS** (in order) | – | protocol 10:57:19 < pilot ≤ 12:08 < thresholds 14:14:35 < formal data 15:19:09 / commit 15:19:30 < v2 21:32:42 < receipts 23:56:44 / 00:35:08 < held-out 00:35:14 < reveal 00:47:30 UTC |
| CAL-09 | **PASS** | – | labels on every output, no recipe promoted |

**CAL-05 rows of the formal fit** (FAIL rows: T4 M2 / M9 (20 /s extrapolation), C2 M6 / M9, C3 M4 / M6 / M9, C4 M4 / M6 / M9):

| Exp | Metric | Value | Tol | Verdict |
|---|---|---|---|---|
| T3 | M1 | 0.01362 | 0.03116 | PASS |
| T3 | M2 | 0.02241 | 0.04637 | PASS |
| T3 | M8 | 0.0005427 | 0.001 | PASS |
| T3 | M9 | 0.005928 | 0.02 | PASS |
| T4 | M1 | 0.03231 | 0.04376 | PASS |
| T4 | M2 | 0.2447 | 0.07579 | FAIL |
| T4 | M8 | 0.0008697 | 0.001 | PASS |
| T4 | M9 | 0.2087 | 0.02 | FAIL |
| T5 | M1 | 0.01533 | 0.03661 | PASS |
| T5 | M2 | 0.0163 | 0.02691 | PASS |
| T5 | M8 | 0.0005448 | 0.001 | PASS |
| T5 | M9 | 0.00586 | 0.02 | PASS |
| C2 | M4 | 0.03069 | 0.05158 | PASS |
| C2 | M5 | 0.3286 | 2 | PASS |
| C2 | M6 | 0.06523 | 0.06326 | FAIL |
| C2 | M8 | 8.941e-07 | 0.001 | PASS |
| C2 | M9 | 0.0418 | 0.02 | FAIL |
| C3 | M4 | 0.04004 | 0.02295 | FAIL |
| C3 | M5 | 0.3039 | 2 | PASS |
| C3 | M6 | 0.07226 | 0.02766 | FAIL |
| C3 | M8 | 9.654e-07 | 0.001 | PASS |
| C3 | M9 | 0.04319 | 0.02 | FAIL |
| C4 | M4 | 0.04623 | 0.02824 | FAIL |
| C4 | M5 | 0.004568 | 2 | PASS |
| C4 | M6 | 0.11 | 0.03039 | FAIL |
| C4 | M8 | 4.057e-06 | 0.001 | PASS |
| C4 | M9 | 0.03456 | 0.02 | FAIL |
| F2 | M7_onset | 0.01104 | 8.351 | PASS |
| F2 | M7_slip | 0.008367 | 0.05314 | PASS |
| F2 | M8 | 3.249e-07 | 0.001 | PASS |
| F2 | M9 | 0.004327 | 0.02 | PASS |

Negative control: 21 / 31 (FAIL T3 / T4 / T5 / C2 / C3 / C4 / F2 M9, T5 M2, F2 M7_slip / M7_onset).

**CAL-06 recovery (formal fit)**

| Param | θ̂ | θ* | σ̂ | rel. error | Verdict |
|---|---|---|---|---|---|
| mu_eq_pa | 3.23e+05 | 3.1726e+05 | 189 | 1.8% | PASS |
| g1 | 0.094162 | 0.19257 | 0.00179 | 51.1% | FAIL |
| tau1_s | 0.014278 | 0.014349 | 0.000557 | 0.5% | PASS |
| g2 | 0.18593 | 0.13367 | 0.00184 | 39.1% | FAIL |
| tau2_s | 0.072254 | 0.12332 | 0.00115 | 41.4% | FAIL |
| mu_table | 0.45007 | 0.44689 | 0.000267 | 0.7% | PASS |

Negative control (reported):

| Param | θ̂ | θ* | σ̂ | rel. error | Verdict |
|---|---|---|---|---|---|
| mu_eq_pa | 3.2439e+05 | 3.1726e+05 | 711 | 2.2% | FAIL |
| g1 | 0.19634 | 0.19257 | 0.0062 | 2.0% | PASS |
| tau1_s | 0.013721 | 0.014349 | 0.000565 | 4.4% | PASS |
| g2 | 0.11801 | 0.13367 | 0.00622 | 11.7% | PASS |
| tau2_s | 0.10399 | 0.12332 | 0.00881 | 15.7% | PASS |
| mu_table | 0.5 | 0.44689 | inf | 11.9% | NON_IDENTIFIABLE_REPORTED |

**Supplementary diagnostic (never scored; after the reveal; 705 s of the formal fit's remaining budget).** The fit model
was evaluated at θ̂ and at θ* in one population on the formal fit split. Mean squared weighted residual per experiment:

| | T1 | T2 | R1 | C1 | G1 | F1 | χ²_red |
|---|---|---|---|---|---|---|---|
| θ̂ | 1.859 | 5.934 | 4.630 | 3.397 | 0.911 | 1.257 | 2.484 |
| θ* | 1.017 | 1.163 | 2.110 | 0.975 | 0.918 | 2.311 | 1.416 |

At θ* the fit model passes **31 / 31** held-out tolerances.
- The S1 / S2 stages placed the viscous parameters in a different basin, g1 / g2 traded with τ2: T2 / R1 / C1 are
  4–6× worse at θ̂ than at θ*.
- The joint polish (6 evaluations) did not leave that basin.
- θ* lies near the prior-box corner (g1 0.193 of a 0.20 upper bound, τ1 14.3 ms of 15 ms).
- The negative control, which fitted T2 alone in S2, landed much closer.

This explains the miss: the CAL-04 prediction was missed because of the S2 data choice and starts, not the GPU noise
> **[Erratum E1]** Withdrawn. See §12.1: every stage after S1 ended on its `max_nfev` cap with the cost still falling, and whether a separate basin exists was not tested.
or the budget (1 h of the 3 h was unused because S4's max_nfev ended the fit).

**GPU** (all GPU0, SHARED_GPU, PID 642659 untouched, GPU1 not used):
- v2 development: 1.75 h.
- Formal fit: 2.15 h hold.
- Negative control: 0.63 h hold.
- Held-out: 0.2 h.
- Supplementary diagnostic: 0.2 h.

CMA-ES comparison: NOT_RUN (optional in v2, not within the budget).

**What this means.**
- The interface works end-to-end with honest, pre-registered scoring: declaration → pilot → frozen data → fit →
  one-shot held-out → reveal.
- The negative control has power.
> **[Erratum E4]** CAL-07 PASS on one viscous row; discrimination is not shown (both fits score 21 / 31).
- The formal outcome is a FAIL of the declared fit procedure on this dataset; the diagnostic locates it in the
  optimiser's staging, not in the forward model or in identifiability at the truth.
- Recovery at n_t 4 against an n_t 6 truth remains biased for the Prony parameters (pilot).

Any further fit attempt on this dataset would be a new, separately declared protocol, and its result would be
post-reveal (the truth is now known). It could not count as a blind validation. A blind re-test needs a new sealed
formal dataset.

## 12. Errata from the P5.7 review (2026-11-12; `docs/reports/rl_ready/P2_5_ACCEPTANCE_REVIEW.md`, GO_WITH_CONDITIONS)

Additive. Nothing in the evidence, receipts, protocols, thresholds or datasets changes. The markers [Erratum E1–E4]
above point here.

### 12.1 E1 (C1): the measured facts of the v2 formal fit, replacing the "basin" mechanism

Recomputed from `docs/evidence/rl_ready/p2_5/fit_receipt_formal.json`:
- **Caps, not convergence.** Every stage after S1 ended on its `max_nfev` cap (scipy status 0) with the cost still
  falling:
  - S2, all 3 starts at 15 evaluations: −7.1 %, −2.5 % and −1.6 % in the last evaluation;
  - S3 at 8 evaluations: −6.8 %;
  - S4 at 6 evaluations: −5.4 %.
- **Bound-active S2.** All three S2 starts ended at g1's lower bound (0.0800–0.0840 vs the bound 0.08) while g2 rose
  and τ2 fell. μ_eq was frozen at S1's T1-only value (+1.2 % off).
- **S4 still moving.** g1 rose at every evaluation: 0.0840 → 0.0846 → 0.0855 → 0.0875 → 0.0900 → 0.0942, toward θ*
  0.193. τ1 had reached the truth (14.3 ms) before the cap ended the stage.
- **Budget unused.** 3,063 s (0.85 h) of the 3 h budget was left when S4 stopped on its cap.
- **No basin test.** Whether θ̂ lies in a separate local minimum was **not tested**.

The supported statement is: **non-convergence under the declared per-stage evaluation caps, after a bound-active S2;
whether a distinct basin exists is untested.** The post-reveal diagnostic (χ²_red 1.416 at θ* < 2.484 at θ̂, 31 / 31
held-out within tolerance at θ*) is consistent with this. The pilot (§9.3) has the same truncation: the existence of
a resolution bias is supported, but its size (≈ 10 %) and "more polishing does not move toward θ_pilot" are
**UNMEASURED**.

### 12.2 E2 (C2): what a user with real data should do differently

1. **Stop stages on convergence or on the time budget, not on small fixed `max_nfev` caps.**
   - Run the joint polish until xtol / ftol or the budget, and report every stage's termination status.
   - Treat a stage that hits its cap while the cost still falls by more than about 1 % per evaluation as
     **non-converged**: flag it, never score it silently.
2. **Do not freeze μ_eq from one slow tensile test before the viscous stage.**
   - Fit μ_eq together with the Prony parameters, or take it from the relaxation plateau.
   - Flag any parameter that is bound-active at the end of a stage.
3. **Add a global stage on the joint problem:** several starts, or a short CMA-ES / BO through `run_ask_tell`.
4. **Validate the exact declared procedure end-to-end on a pilot** (same caps and budget) before predicting CAL-04.
   The v2 pilot ran S4 for one evaluation only.
5. **Take bounds from prior knowledge, not from a generating prior,** and report parameters that sit at a bound.
   - In v2 the bounds were the u-image of the prior box the truth was drawn from. That is optimistic for a synthetic
     test, and it is wider than the θ prior for g2 (θ̂ g2 0.186 > 0.15).
   - The v1 fit-split trajectories (not the held-out split or the truth) fed the v2 design.
6. **Drive the forward model with the measured fixture or crosshead history** where it exists. The current forward
   models use the declared nominal schedule.
7. **Set tolerances from measured repeatability (a) and a pilot *fit* (b)**, as `rubber_calib_tests_real_v1` states.
   Declare the real protocol, including its extra short-τ Prony term, before any data exists.
8. **θ̂ is RESOLUTION_SPECIFIC** to `rubber_ref_v1` (n_t 4). Never carry it to another resolution without the bias
   shift.
9. **Until F3 is fixed,** use one canonical ledger directory per freeze, and commit the ledger before the reveal. F3 is
   fixed from commit `F3_COMMIT` onward: global one-shot registry, and the gate loads and checks the ledger itself
   (§12.5).

### 12.3 E3 (C3): pre-v2 snapshot

Sections 4–9 are marked as the pre-v2 snapshot. The stale sentences are corrected inline:
- "never been read" / "valid for a resumed fit" (§4);
- "validated as an optimiser" (§9.4).

In the CAL table of §6, CAL-04..07 NOT_RUN is superseded by §10. §8's "does not yet deliver the formal fit" is
superseded by §10.

### 12.4 E4 (C4): the CAL-07 headline

CAL-07 PASSes on one viscous row (T5 M2 0.0516 > 0.0269). Otherwise the T2 + C1 control passed T4 M2 and every
cantilever M6 row that the primary fit failed, and it scored the same 21 / 31. Held-out discrimination between an
adequate and a deficient fit is therefore not shown by this run; only the unscored post-reveal θ* evaluation (31 / 31)
shows it.

### 12.5 Smaller corrections

- `test_rubber_calibration_v2.py` has 38 tests, not 40.
- The REAL-gate negative tests are 24 parametrised cases × 2 gates, not 48 cases.
- C2 M4 PASSed; the CAL-05 failures are T4 M2 / M9, C2 M6 / M9, C3 M4 / M6 / M9 and C4 M4 / M6 / M9.
- The reveal ran on a tree dirty only by the then uncommitted supplementary script, and the diagnostic ran on a tree
  dirty only by `write_evidence.py` edits (F8). `reveal.py` now refuses a dirty tree and pins the protocol.

