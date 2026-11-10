# P2-0 / P2-1 — Rubber material v1 (recipe, references, solver, single-env and batched paths)

Date: 2026-11-10. Branch `claude/rl-rubber` (worktree `/home/ningan/RoboticCuttingSim-rl-rubber`). Binding design:
`docs/adr/0023-rubber-material-v1.md` (§15 task list R0.1–R1.7). Evidence:
`docs/evidence/rl_ready/p2_1_rubber_001.json`. Raw outputs: `artifacts/rl_ready/p2_1/` (not committed).

**Material status: `SIMULATOR_ONLY_UNCALIBRATED`.** There is no real rubber data on this machine; every result below is
numerical verification against analytic references, never a real-world calibration claim. No cutting of rubber.

**Outcome.** P2-0 (R0.1–R0.4) and P2-1 (R1.1–R1.7) are implemented, tested and committed. In the formal run
(protocol `rubber_phys_tests_v2`, single-env reference, CPU), RUBBER-PHYS-01, -02, -03, -04, -05, -06, -08 and -09
**PASS**. **RUBBER-PHYS-07 (stretch & release) FAILs**: the release snap becomes non-finite at CFL 0.5; at CFL 0.25 it
stays finite but leaves the J domain and loses ≈ 34 % of the stored energy. RUBBER-PHYS-10 is **PASS on CPU**:
L0 bitwise single↔batched, L1 isolation and partial reset, and the heterogeneous N = 4 batched RUBBER-PHYS-04. Its GPU
parts (L2 smoke N = 1/4/8, L3 band) and the R1.4 GPU smoke are **BLOCKED / NOT_RUN**: both RCS GPU locks were held by
Phase 1 jobs for the whole session. RUBBER-PHYS-11 (refinement) is P2-2, NOT_RUN. One deviation from the ADR is
declared: the static-initializer stop rule (§2).

## 1. What was built

| Task | Deliverable | Status |
|---|---|---|
| R0.1 | `src/robotic_cutting_sim/rubber_material.py`, `configs/materials/rubber_material_v1.schema.json`, `configs/materials/rubber_generic_soft_v1.json` | done; 21 tests |
| R0.2 | `rubber_calibration.py`, `configs/rubber/calibration/manifest_v1.schema.json`, `configs/rubber/calibration/none_v1.json` (`NO_REAL_DATA`) | done; 11 tests |
| R0.3 | `rubber_reference.py` (NH J-bar PK1/energy, uniaxial / simple shear, EB / Timoshenko sag, f1, elastica BVP, Prony G′/G″) | done; 12 tests (reproduces the ADR E3/E4/E5 values) |
| R0.4 | `configs/rubber/rubber_phys_tests_v1.json` (frozen, SHA-pinned), `rubber_record_v1.schema.json`, `configs/rubber/workload_e_v1.json` | done; superseded in one item by v2 (§2) |
| R1.1 | `rubber_constitutive.py` (shared `@wp.func`s, cell J-bar gather, domain test) | done; 7 tests |
| R1.2 | `tlmpm_rubber_v1.py` (frozen v5/v6 grid kernels unchanged, S = 1, J-bar, stress, Dirichlet groups + float64 reaction wrench, traction, table/fixture contact via the frozen helper, pic = 0 enforced, diagnostics, receipt) | done; 7 tests |
| R1.3 | `rubber_scene.py` (`rubber_strip_v1`, schema `rcs_rubber_scene_v1`) | done; 9 tests |
| R1.4 | `live_rubber_material.py` (step rule, CUDA graph chunks + remainder, sticky validity, records `rcs_rubber_record_v1`) | done; 6 tests; GPU smoke §5 |
| R1.5 | `rubber_initializer.py` (`relax_mass_proportional_v1`, kinetic damping refused above CFL 0.25, `static_stop_v2`) | done; 3 tests |
| R1.6 | `rubber_phys.py` + `scripts/rubber/run_rubber_phys.py` + `write_p2_1_evidence.py` | done; formal run §3 |
| R1.7 | `batched/constitutive_rubber.py`, `rubber_layout.py`, `rubber_batch.py`, `rubber_material_batch.py` | done; 10 tests (L0/L1); GPU L2/L3 §5 |

Frozen files: `tlmpm_v5.py`, `tlmpm_v6.py`, every Phase 1 batched module (`batched/constitutive.py`, `layout.py`,
`material_batch.py`, `tlmpm_v7_2_batch.py`, `errors.py`, `diagnostics.py`, `particle_contacts.py`, `pose_interp.py`),
`configs/rl_benchmark/*` and `scripts/rl_ready/*` have no diff against the branch base `2f316c1`;
`tests/test_batched_frozen_sources.py` passes. New Warp kernels call the frozen kernels/functions unchanged
(`_p2g`, `_grid_update`, `_g2p`, `_musl`, `_deformation`, `_advance`, `_contact_impulse`, `_box_sdf`,
`tlmpm_v6._grid_positions`). The rubber registry is separate from `batched/constitutive.py`; the existing
`UNSUPPORTED_*` codes are reused (no edit of `errors.py`). The `open_vector_env` dispatch is deferred to P2-6 (P1-5);
`BatchedRubberMaterial` is the API the dispatch will wrap (reset/advance/wrench/invalid_reason/markers on the device).

Implementation choices beyond the ADR text (declared, not hidden):
- the PK1, determinants and the J-bar cell mean are evaluated in float64 and stored in float32 (F and the stress stay
  float32 as in every frozen kernel); J-bar is stored float64 (node-sized);
- group motions are interpolated in the kernel from start/end maps per control step (rigid: slerp + linear translation;
  affine: linear, patch / homogeneous tests) instead of `[N·G, T]` pose tables; prescribed node velocity
  `(x(i+1) − x(i))/dt` from float64 reference node positions (position-exact);
- constraint work = BC impulse × prescribed velocity (zero for a fixed clamp), so numerical constraint dissipation stays
  visible in the energy balance;
- diagnostics reduce with a fixed width (64 groups per env) so single and batched reductions sum in the same order.

## 2. Protocol: v1 frozen, v2 declared before the formal run

`rubber_phys_tests_v1.json` was committed (`2bed4f9`) before any implementation. During implementation smoke runs
(before any formal run) its static-initializer stop rule turned out to be unreachable
(`artifacts/rl_ready/p2_1/protocol_diagnostics_001/`): on a 2,304-particle cantilever under critical mass-proportional
damping the tip was stationary (δ/δ_EB = 0.98556) but (a) the max **particle** speed stayed at 1.3–1.8e-4 m/s — the
pure-FLIP particle velocity keeps a grid null-space component that never reaches the grid (moves neither x nor F) and
cannot be damped through the grid — and (b) the **grid** speed settled within 0.09 s at a float32 noise floor (rms
2.4e-5 m/s, max 7–9e-5 m/s at low-mass surface nodes) where a physical mode would decay like e^(−108 t). The ADR §7
stop "max |v| < 1e-5 m/s" is therefore not a usable convergence test. `rubber_phys_tests_v2.json` (`63ecfd1`) changes
**only** that stop rule (`static_stop_v2`: marker stationarity ≤ 1e-4 of the marker displacement on 3 consecutive checks
**and** reaction/weight balance < 1e-3; speeds reported); every test, criterion and threshold is identical (tested by
`tests/test_rubber_phys_config.py`). v1 stays byte-identical. This is a deviation from ADR §7 to be confirmed in the
P2-2 review.

## 3. RUBBER-PHYS results (formal run `rubber_phys_001`)

Command (CPU, single-env reference, one process per configuration, 24 workers; both GPU locks were held by Phase 1):

```
TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/run_rubber_phys.py \
    --tests 01,02,03,04,05,06,07,08,09 --device cpu --workers 24 --out artifacts/rl_ready/p2_1/rubber_phys_001
```

Commit `63ecfd1` (src/configs clean), protocol `rubber_phys_tests_v2.json` (sha256 `e4ff8502…3060`), Warp 1.17.0 CPU,
AMD Threadripper PRO 9975WX (shared host, load 8–30), 66 configurations, 876 s wall (sum of case walls 8,300 s).
Deterministic inputs (no random seeds in these tests). β = 20, pic = 0, CFL 0.5, 2 particles/cell/axis, ρ 1150 unless
stated.

| Test | Reference | Criterion (pre-declared) | Measured | Verdict |
|---|---|---|---|---|
| RUBBER-PHYS-01 patch (8 homogeneous F, 6³ cells, 1 substep) | §5 PK1 float64, objectivity | ‖P−P_ref‖ ≤ 1e-5‖P_ref‖+1e-3 Pa; \|J̄−J_p\| ≤ 1e-6; \|J̄−det F\| ≤ 1e-5; interior/boundary force ≤ 1e-5; objectivity ≤ 1e-5 | worst error / allowance 5.9e-3; 3.2e-7; 2.8e-7; 6.7e-6 (rot90×stretch); 3.7e-6 | **PASS** |
| RUBBER-PHYS-02 small sag (24 configs: L/t 10, 20 × t 3, 4, 6 mm × W t, 4t × n_t 3, 4; α = 0.05) | EB (W = t), plate bracket (W = 4t), 2-grid extrapolation (order 2) | narrow \|δ/δ_EB−1\| ≤ 4 %; wide in [0.774, 1]·δ_EB; all \|δ/δ_ex−1\| ≤ 2 %; \|R_z+Mg\|/Mg ≤ 1e-3; converged ≤ 15 T1 | narrow 0.9803–0.9971; wide 0.9234–0.9650; δ/δ_ex 1.0038–1.0184 (n_t 3 up to 1.84 %); reaction ≤ 1.0e-3 (stop rule); all converged | **PASS** |
| RUBBER-PHYS-03 scaling (L 60/90/120 mm at t 4; t 3/4/6 mm at L 100; g 0.4) | δ ∝ L⁴ t⁻² | exponents 4 ± 0.15, −2 ± 0.1 | 4.012, −2.011 (δ/δ_EB 0.991–1.000) | **PASS** |
| RUBBER-PHYS-04 free vibration (L 60, W = t = 6 mm, μ 2 MPa, 6 T1, 1 ms records) | f_EB = 19.287 Hz | \|f/f_EB−1\| ≤ 3 %; energy ≤ 0.5 % peak Ψ; last/first amplitude ≥ 0.95; f(dt)/f(dt/2) within 0.2 %; batched bitwise | 19.202 Hz (−0.44 %); 1.6e-4 / 5.8e-4 (dt, dt/2); 1.0018; 1.0e-5; batched N = 4 CPU bitwise equal (0 mismatches, 312 steps) | **PASS** |
| RUBBER-PHYS-05 large sag (L/t 20, α 3 / 9 / 20, μ 0.2 MPa) | inextensible elastica | \|δ/δ_el−1\| ≤ 3 %; J̄ ∈ [0.97, 1.03] | 0.9974 / 0.9983 / 0.9988; J̄ ∈ [0.9918, 1.0088] | **PASS** |
| RUBBER-PHYS-06 uniaxial load–unload (60×10×4 mm, n_t 4, λ 1→2→1 over 2×60 T_ax) | compressible P11(λ), J(λ) | stress ≤ 1 % at λ 1.25/1.5/2 (load), 1.25/1.5 (unload); gauge J̄ ≤ 5e-3; unload/load ≤ 1 %; residual length ≤ 0.1 % | max \|err\| 0.093 %; 5.8e-5; 0.071 %; 1.1e-5 | **PASS** |
| RUBBER-PHYS-07 stretch & release (λ 1.5, release grip 2) | elastic recovery, energy conservation | mean length within 0.5 %; energy drift ≤ 1 % over 10 axial periods | run aborted: NONFINITE_STATE + RUBBER_J_DOMAIN 3 control steps (15 ms) after the release (strict mode) | **FAIL** |
| RUBBER-PHYS-08 simple shear γ 0.5 / 1 / 2 | σ12 = μγ, N1 = μγ² | ≤ 0.5 % | 7.9e-7, 1.6e-6 | **PASS** |
| RUBBER-PHYS-09 records (18 runs: 3 scenarios × t 3/4/6 × W t/4t) | `rcs_rubber_record_v1` | complete and schema-valid | 18/18 valid (header + 5 step records each); note: 2 of 6 stretch-release runs (W = t, t 3 and 4 mm) carry RUBBER_J_DOMAIN after the release — the record contract holds, the physics is the -07 failure | **PASS** (records) |
| RUBBER-PHYS-10 single ↔ batched | ADR 0021 L0–L3 | L0, L1 bitwise (CPU); L2 invariants + ≤ 1.5 GB; L3 band | L0, L1: PASS (§4); L2/L3: NOT_RUN (no GPU slot) | **PASS (CPU L0/L1); L2/L3 BLOCKED → NOT_RUN** |
| RUBBER-PHYS-11 refinement / GCI | Roache GCI | GCI ≤ 2 % | not in P2-1 scope | **NOT_RUN** (P2-2 R2.2) |

Provenance note: the formal run used `63ecfd1`, where the diagnostics reduced with an env-size-dependent width;
`c0eec4d` fixed the width at 64 (single and batch then sum in the same order). This changes only the last bits of the
diagnostic energies, never the state: the batched RUBBER-PHYS-04 run on `c0eec4d` reproduces the formal run's tip
series bitwise.

Notes. -01/-08 run with CUDA-independent CPU kernels (the result files do not record a device field; they ran in the
CPU job). -02/-03 use a reduced gravity load so that the sag stays linear (declared in v1/v2: α = 0.05 per
configuration for -02, g = 0.4 m/s² for -03); the elastica/linear ratio is ≥ 0.9969 at the largest α. The wide-strip
values sit inside the plate bracket, 3.5–7.7 % below the narrow-beam EB value (plate stiffening, ADR E1). The -02
two-grid extrapolation assumes order 2; three grids and the observed order are P2-2.

### RUBBER-PHYS-07 failure (preserved) and diagnostics

Formal result: `artifacts/rl_ready/p2_1/rubber_phys_001/raw/07/stretch_release.json` (traceback, strict mode). Two
labelled diagnostic re-runs (strict off, records each 5 ms; NOT formal results):
`artifacts/rl_ready/p2_1/diag_07_cfl_factor1` (CFL 0.50, 862 substeps / 5 ms) and `diag_07_cfl_factor2` (CFL 0.25).

| | CFL 0.50 (protocol) | CFL 0.25 (diagnostic) |
|---|---|---|
| ramp + hold (53 steps) | finite, J 1.025–1.026, energy residual 7.6e-6 J | same |
| first 2 steps after release | KE 0.22 J of ≈ 0.28 J stored, max speed 25–28 m/s, length 0.94 L₀, J ∈ [0.84, 1.16], energy residual −5.7e-4 J | same (−4.6e-4 J) |
| 3rd step after release | non-finite state (all later steps NaN) | J_min 0.188 → RUBBER_J_DOMAIN; finite |
| post-release window | — | length 0.61–0.90 L₀ (buckling / folding), J_min 0.31–0.49, total energy falls monotonically to −0.094 J (≈ 34 % of the stored energy) |

Reading: the release turns the stored strain into a compressive wave of order 50 % strain. The NH axial tangent
μ(1 + 2λ⁻³) is 5.7× its reference value at λ = 0.5, i.e. the wave speed 2.4× — the whole margin of CFL 0.5 against the
measured critical 1.2–1.5 (ADR E2, measured at small strain). Halving dt removes the blow-up, but the strip then folds
onto itself (v1 has no self-contact) with J below the domain and a large, unexplained energy loss. Elastic recovery at
λ = 1.5 is therefore **not** demonstrated in P2-1.

## 4. Batched path: parity, isolation, reset (RUBBER-PHYS-10 L0/L1, CPU)

`tests/test_rubber_batched.py` (10 tests, all pass; Warp CPU, `parity_mode=True`):
- **L0** bitwise after every control step for every particle / node / group / traction array, the cell CSR, the
  constraint work and violation counters, the diagnostics row (17 columns), the 9 markers and the group wrench:
  N = 1; N = 3 heterogeneous (L 9/12/15 mm, W 3/6 mm, μ 0.3/0.4/0.7 MPa, ρ 1050/1150/1200, table friction 0.4/0.8/1.2
  with an active table contact, per-env jaw motion and tip traction); N = 3 copies.
- **RUBBER-PHYS-04 batch**: N = 4 heterogeneous (L 48/60 mm, W 6/12 mm, μ 1/2 MPa, ρ 1000/1150; 11,088 particles),
  312 control steps × 193 substeps: 0 mismatches over every array of every env; env 0's tip series is bitwise equal to
  the formal single run (`artifacts/rl_ready/p2_1/batched_04_001/batched_04.json`, 453 s CPU). Per-env f/f_EB:
  −0.44 %, −0.58 %, +0.45 % (W = 2t), −0.41 %; energy drift ≤ 4.2e-4 of peak Ψ.
- **L1**: perturbing env 1 (motion and traction) leaves envs 0 and 2 bitwise unchanged; `reset([1])` makes env 1
  bitwise equal to a freshly packed env and leaves envs 0/2 bitwise unchanged, and env 1 then evolves exactly like a
  fresh env; the solver/material array classification equals `vars()` (no unclassified device array); reset with a
  resampled material (μ 0.8 MPa, ρ 1010, friction 1.1) then equals a single reference built with those values
  (bitwise); out-of-range materials refused; seeded sampling inside declared ranges; env-ownership audits (nodes, cell
  CSR, markers, groups, traction) all 0; refusals: empty batch, CPU without parity mode, non-uniform dx, stepped single
  env, geometry change on partial reset, the P2-3 Prony model (declared, not registered); strict mode raises with env
  ids and reasons.

## 5. GPU smoke (R1.4, R1.7 L2, RUBBER-PHYS-10 L3) — BLOCKED, NOT_RUN

`scripts/rubber/gpu_smoke_rubber.py` is written. It contains: part A, a single LiveRubberMaterial on CUDA for the
workload-E smoke variant (97,280 particles, graph chunks of 500); part B, BatchedRubberMaterial N = 1/4/8 with per-env
μ and ρ resampled inside [0.3, 1.3] MPa × [1000, 1150] kg/m³, finite / mass-constant / J-domain / audit invariants,
VRAM peak, warm/cold step wall, env-steps/s, sim-time/wall and reset p95; part C, the L3 band (batched N = 4 vs 3
single-env GPU repeats). It was queued with a non-blocking poll on both RCS GPU locks
(`/mnt/data/ningan/lab_records/rcs_rl_runner/gpu{0,1}.lock`, ≥ 6 GB free required) from 15:37 to 15:59 KST.
Both locks stayed held by Phase 1. GPU1 ran `scripts/rl_ready/supervise_rollout.py` (lock age about 5 h) and GPU0 had a
holder plus further blocking waiters; GPU utilisation was 91–100 %. Phase 1 has priority, so no row is reported and no
throughput number exists. Command to run it later:

```
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> TMPDIR=/mnt/data/ningan/tmp \
    PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/gpu_smoke_rubber.py --out artifacts/rl_ready/p2_1/gpu_smoke_001
```

Substitute evidence, **not a VRAM measurement**: the device-array bytes of the same batch built on CPU without
stepping (`artifacts/rl_ready/p2_1/array_memory_cpu_001/array_memory.json`) are 378 B/particle: 35 / 140 / 280 MB at
N = 1/4/8 × 97,280 particles. The CUDA context, graphs and mempool slack come on top. The step rule at the workload-E
range maxima gives dt = 3.0 µs, i.e. 6,662 substeps per 20 ms control step.

| N | particles/env | control Hz | material dt | substeps | cameras | env-steps/s | sim-time/wall | VRAM peak | reset p95 | physics verdict | warm/cold | GPU | sharing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1, 4, 8 | 97,280 | 50 | 3.0 µs | 6,662 | none | NOT_RUN | NOT_RUN | NOT_RUN (arrays 35/140/280 MB) | NOT_RUN | NOT_RUN | – | RTX 5090 (none obtained) | – |

## 6. Tests

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rubber_*.py tests/test_tlmpm_rubber_v1.py tests/test_live_rubber_material.py tests/test_batched_frozen_sources.py tests/test_repo_naming.py` | 110 passed (90 rubber + 18 frozen pins + 2 naming), 7.9 s |
| same rubber files + `tests/test_batched_*.py tests/test_wp12_phase5_frozen.py` | 276 passed, 3 failed (pre-existing, §7.6), 546 s |
| full `tests/` (2,383 collected) | NOT_COMPLETED: stopped after about 20 min at about 4 % on the shared host |
| torch-runtime tests | NOT_RUN (no torch dependency in the rubber modules) |

Rubber tests per file: material 21, calibration manifest 11, reference 12, phys config 4, constitutive 7, scene 9,
solver 7, runtime 6, initializer 3, batched 10.

## 7. Open items and what P2-2 needs

1. **RUBBER-PHYS-07 FAIL** (stretch/release at λ 1.5). Needs (a) a step rule that bounds the wave speed in the deformed
   state (tangent-stiffness CFL or adaptive dt per control step; the ADR's "re-measured in P2-2 under large deformation
   and contact" is now required, CFL 0.5 is not safe beyond moderate compression), (b) self-contact for folding (P2-4
   design, opt-in) or a release protocol that does not fold, and (c) an explanation of the ≈ 34 % energy loss at CFL
   0.25 (candidate: interpenetration without self-contact, grid-position / F inconsistency under folding). A protocol or
   threshold change is a v3 of the frozen tests, never an edit.
2. **Initializer stop rule** (ADR §7) replaced in v2 by marker stationarity + reaction balance: to be confirmed by the
   P2-2 review.
3. **GPU**: R1.4 GPU smoke, L2 N = 1/4/8 (≤ 1.5 GB) and the L3 band are NOT_RUN (BLOCKED by Phase 1 GPU occupancy); run `scripts/rubber/gpu_smoke_rubber.py` when a slot frees. No rubber throughput number exists yet.
4. **RUBBER-PHYS-11** refinement / GCI (R2.2), β-sensitivity (R2.3), profiling and N = 1/4/8/16 throughput (R2.4),
   batched runs of -02/-03/-05..-09 (R2.1; the batched path is bitwise equal on CPU, so CPU repeats add nothing; GPU L3
   bands are the open part).
5. **open_vector_env dispatch** for rubber (P2-6, after P1-5); robot coupling / 20 ms delayed wrench (P2-4).
6. Pre-existing, unrelated: `tests/test_batched_particle_contacts.py::test_fallback_grid_reproduces_warp_grouped_query_sequence`
   (3 params) fails in this environment; the test and every module it imports are byte-identical to `2f316c1`.

## 8. Artifacts

- `artifacts/rl_ready/p2_1/rubber_phys_001/` (raw per configuration, `summary.json`, `run_meta.json`)
- `artifacts/rl_ready/p2_1/batched_04_001/batched_04.json`
- `artifacts/rl_ready/p2_1/diag_07_cfl_factor1/`, `diag_07_cfl_factor2/` (diagnostics, not formal)
- `artifacts/rl_ready/p2_1/protocol_diagnostics_001/` (v1 stop-rule finding)
- `artifacts/rl_ready/p2_1/gpu_smoke_001/` (not created: no GPU slot)
- `artifacts/rl_ready/p2_1/array_memory_cpu_001/array_memory.json` (array bytes, not VRAM)

