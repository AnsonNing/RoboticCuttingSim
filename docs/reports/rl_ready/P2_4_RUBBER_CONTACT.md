# P2-4 — Rubber contact, friction and dynamic coupling (ADR 0023 §22)

Branch `claude/rl-rubber-p24` (worktree `RoboticCuttingSim-rl-rubber-p24`, from `025ecaf`). Material status:
**SIMULATOR_ONLY_UNCALIBRATED** — friction coefficients, contact stiffness and dashpot ratios are declared numerical /
illustrative values, not measured. Protocol: `configs/rubber/rubber_contact_tests_v1.json` (sha256
`4d3dc68fc50aeeb247c2b544c65312fd2a651a90a99eb394b11f3fa478b46e4c`, pinned in `tests/test_rubber_p2_4_protocols.py`,
declared at `51023d6` before any formal run). Evidence: `docs/evidence/rl_ready/p2_4_contact_001.json`. Raw outputs:
`artifacts/rl_ready/p2_4/` (not committed).

Nothing frozen changed: v1–v7_2, law 8, evaluators, every Phase 1 module (`batched/constitutive.py`,
`batched/errors.py`, `batched/particle_contacts.py`, `batched/material_batch.py`, …), `configs/rl_benchmark/*`,
`scripts/rl_ready/*`, the rubber protocols v1–v4 / moving-grip v1 / visc v1–v2, `tlmpm_rubber_v1.py`,
`rubber_constitutive.py`, `tlmpm_rubber_visco_v1.py`, `rubber_visco_constitutive.py`, `rubber_constraint_work.py`.
`live_rubber_material.py` gained an opt-in `contact=` (default: exactly the previous classes; tested byte-wise).

## 0. Status

| Item | Status | Key numbers |
|---|---|---|
| Decisions 1–5 (ADR §22.1–22.5) | **DONE** | law `grid_node_penalty_coulomb_v1`; pads as kinematic box frames, [N, G, B, 6]; self-contact NOT_IMPLEMENTED_V1; delay assessed; isolation by row ownership |
| Protocol `rubber_contact_tests_v1` | **DECLARED** (`51023d6`) | 11 tests; probe scenes listed as seen |
| Contact law + kick-paired work, single env | **DONE** (`259f2b6`) | `rubber_contact.py`, `tlmpm_rubber_contact_v1.py`; off = byte-identical |
| Formal CPU run `contact_v1_001` (`436f91c`, clean) | -02 **PASS**, -07 **PASS**; -01, -03, -04, -05 **FAIL**; -06 / -08 / -09 **NOT_RUN** | §3 |
| GPU check `gpu_contact_002` (GPU0, SHARED_GPU) | single-env L2 **PASS**, out-of-reach rows exactly 0, cost reported | §4 |
| Batched contact (L0 / L1, GPU isolation) | **NOT_RUN** — task B1 / B2 | ADR §22.8 |
| CPU suites | §5 | |

## 1. Decision probe (DIAGNOSTIC, before the protocol)

Script `scripts/rubber/p2_4/probe_contact_laws.py`; the run used for the decision is `probe_contact_laws_004` (clean
`259f2b6`); `_001` (step gravity, μ 0.5 friction in the impact), `_002` (1 ms gravity steps), `_003` (dirty tree, single-ζ
kernel) and the development sweeps `contact_sweeps_dev_001` (`scripts/rubber/p2_4/diag_contact_sweep.py`, uncommitted
tree; the code state of each sweep is in its README) are preserved. Two design errors of the first probes were found
and fixed before any declaration: a gravity *step* on an undamped elastic block overshoots by 2 and periodically unloads
the contact (so even a correct law slips at 0.8 μ), and a rod impact with friction dissipates the face's Poisson
expansion (the 1-D reference is frictionless).

Block 12 × 6 × 3 mm (n_t 3, μ 0.5), gravity ramped per substep over 20 ms, hold displacement and slip error
(a_fit − g(sinθ − μcosθ))/(g cosθ); rod 30 × 4 × 4 mm, n_t 4, 0.2 m/s onto a frictionless wall:

| Law | stick 0.8 / 0.9 / 0.95 μ | slip 1.05 / 1.15 / 1.25 μ | impact E_end/E0, CFL 0.5 / 0.25 |
|---|---|---|---|
| frozen per-particle helper (`_contact_impulse`, D8 design) | 0.50 / 1.24 / 1.59 mm | +2.0 / +0.8 / +0.4 % | 0.966 / 0.950 |
| node velocity projection `grid_node_coulomb_v1` | 0.34 / 0.75 / 1.15 mm | +1.2 / +0.3 / +0.1 % | 0.993 / 0.972 |
| node penalty, no dashpots (`_003`, sweeps) | creeps at 0.8 μ in most settings (4e-5 … 7e-4 m; one κ 0.05 setting held) | +1.4 … +3.3 % | 1.0015 / 1.0008 |
| node penalty, normal dashpot only (`_004` "undamped" — see note) | 2.7e-9 / 7.3e-9 m / 1.3 mm | +4.2 / +4.3 / +4.3 % | 0.962 / 0.960 |
| node penalty, tangential dashpot only (sweep) | 1.3e-4 m at 0.8 μ | +1.3 … +9 % | – |
| **node penalty + both dashpots, ζ 0.5 (adopted)** | **1.0e-6 / 8.3e-7 m** / 0.71 mm | **+0.53 / +0.15 / +0.10 %** | **0.962 / 0.960** |

Note (labelling): at `259f2b6` the probe's `grid_penalty_undamped` variant switched off only the tangential dashpot; it
is a normal-dashpot-only variant, and its impact rows equal the adopted law's. The fully undamped data come from
`_003` (where one ζ drove both dashpots) and the sweeps. The script now switches both off.

What the probe shows: (i) the frozen helper is never at rest on a slope (friction uses the particle's start-of-substep
velocity: at rest it applies none) and its approach damper loses more energy as dt shrinks; (ii) a lumped node projection
under-corrects the MUSL drift velocity Bπ (B = M_l⁻¹M_cM_l⁻¹), so the face chatters (8–20 % of bottom nodes in contact per
substep); (iii) every explicit Coulomb law shows a friction-induced flutter of the elastic block, not removed by n_t 4 or
CFL 0.25; declared contact dashpots on the node's predicted relative velocity suppress it, and both are needed.

## 2. Implementation (single env)

- `src/robotic_cutting_sim/rubber_contact.py`: `ContactSpec` (planes; rigid kinematic frames with oriented boxes, kind
  fixture / pad), the adopted kernel `node_penalty_contact` (one thread per node, obstacles in fixed order), `slider_update`,
  `contact_work_end`, `node_displacement`, `contact_energy`, the float64 box SDF twin, the rejected `node_contact`
  (projection; kept for the A/B record) and host references.
- `src/robotic_cutting_sim/tlmpm_rubber_contact_v1.py`: `TLMPMRubberContact(TLMPMRubberVisco)` — the elastic / visco launch
  sequence plus contact after the P2G grid update, the second work half after the final MUSL update, the node
  displacement and slider update after the drift; reactions per obstacle, wrenches planes [P, 6] / boxes [F, B, 6];
  refuses the frozen `table_plane` / `fixture_boxes` path together with a spec.
- `live_rubber_material.py`: `contact=` (opt-in), `advance(frames=…)`, `last_contact`, contact block in the step record.
- Work (kick-paired, exact conjugate of the discrete KE): W_n, W_t = Σ J·(g_n + g_{n+1})/2; obstacle power P; contact
  dissipation D = (P_n − W_n) + (P_t − W_t); Coulomb-slider and dashpot dissipation reported separately.
- State per node: u (float64), J_n, J_t, slider ξ / obstacle id / obstacle velocity; all cleared by `reset()`.

## 3. Formal CPU run `contact_v1_001`

`CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/run_contact_v1.py --out
artifacts/rl_ready/p2_4/contact_v1_001 --jobs 32` — commit `436f91c`, clean tree, Warp 1.17.0 CPU, Python 3.12,
41 jobs, shared Threadripper PRO 9975WX. The smoke `contact_v1_smoke_001` (5–40 control steps per scene) is preserved
and labelled SMOKE.

### 3.1 RUBBER-CONT-01 incline — FAIL

| Row | measured | criterion | |
|---|---|---|---|
| stick μ 0.4, ratio 0.6 / 0.85 | hold displacement 1e-4 s / 1e-4 s, a ≈ 0 | ≤ 0.01 s, ≤ 0.001 g cosθ | PASS |
| stick μ 0.8, ratio 0.6 / 0.85 | 9e-4 s / 4.4e-3 s, a ≤ 4e-5 g cosθ | | PASS |
| **stick P2, μ 0.8, ratio 0.95** | **4.82 s, a = +0.018 g cosθ (slides)** | | **FAIL** |
| slip μ 0.4, ratio 1.1 / 1.2 / 1.3 | +0.20 / +0.06 / +0.03 % g cosθ | ≤ 1 % | PASS |
| **slip μ 0.8, ratio 1.1 / 1.2 / 1.3** | **+1.65 / +1.88 / +1.06 %** | ≤ 1 % | **FAIL** |
| onset μ 0.4 / **0.8** | μ_eff −0.67 % / **−2.75 %** | ≤ 2 % | PASS / **FAIL** |
| books (max over 12 runs) | 1.03 % (μ 0.4 / 0.6 stick, small energy scale); others ≤ 0.69 %; order ratio 0.29 | ≤ 1 % | **FAIL** (1 run) |
| validity | no flag | | PASS |

### 3.2 RUBBER-CONT-02 penetration vs refinement — PASS

Max penetration ≤ 0.049 s at every resolution (limit 0.25 s); δ(CFL 0.25)/δ(CFL 0.5) = 0.256 (band 0.2–0.3, law 0.25);
δ(n_t 4)/δ(n_t 2) = 0.490 (band 0.4–0.6, law 0.5); undamped law: Σ k_n δ_I / reaction = 1.000002, interior δ_I /
(pA_I/k_n) = 1.050 (≤ 5 %); reaction = weight + load within 4e-5.

### 3.3 RUBBER-CONT-03 frictionless rod impact — FAIL (books of one run)

| Run | max \|B\|/E0 | end | contact dissipation / E0 | branch D_b / E0 | rebound v/V |
|---|---|---|---|---|---|
| elastic n_t 4, CFL 0.5 / 0.25 | 0.59 / 0.25 % | 0.09 / 0.05 % | 2.86 / 3.01 % | – | 0.940 / 0.938 |
| **elastic n_t 2, CFL 0.5** | **1.024 %** | 0.09 % | 4.71 % | – | 0.915 |
| P2 n_t 4, CFL 0.5 / 0.25 | 0.54 / 0.20 % | 0.04 / 0.02 % | 2.86 / 3.01 % | 10.2 % | 0.898 / 0.896 |
| P2 n_t 2, CFL 0.5 | 0.94 % | 0.04 % | 4.71 % | 10.6 % | 0.874 |

Contact time 2.58 ms vs 2L/c0 = 2.50 ms (+3.3 %, PASS); dissipation attribution PASS (separated, D ≥ 0, ≤ 8 %);
convergence D(n_t 4)/D(n_t 2) = 0.61 ≤ 0.75 PASS; first order (0.42 / 0.38 ≤ 0.65) PASS; penetration PASS. The elastic
n_t 2 run exceeds the 1 % books limit by 0.024 %: **FAIL** as declared.

### 3.4 RUBBER-CONT-04 friction dissipation = μN·slip — FAIL

D_contact / (μ Σ N_k Δx_cm,k): μ 0.4: 0.988 / 0.993 / 0.995 (PASS); **μ 0.8: 0.932 / 0.930 / 0.943 (< 0.95, FAIL)**.
Split (μ 0.4 / ratio 1.2): Coulomb slider 0.90, dashpots 0.06, normal work 0.03, tangential work 0.96 of μN·slip.

### 3.5 RUBBER-CONT-05 pinch-and-pull — FAIL

Grip normal force (sum of both pads) 0.67 / 1.40 / 2.15 N at 20 / 40 / 60 µm; slip onset at 4.0 / 7.5 / 11.5 ms of the
pull; sustained slip (tip speed ≈ 0 while the pads move at 0.1 m/s). |T|/(μ_pad N) = **0.972 / 0.964 / 0.956** (band
0.97–1.02: FAIL for two depths); cone max 0.9995 (PASS); slip ratios agree within 1.6 % across a 3.2× grip range (PASS);
**clamp + pad balance 9.8 / 6.0 / 6.3 % (> 5 %, FAIL)**; penetration PASS. Per-pad wrenches [F = 2, B = 1, 6] are
recorded every 0.5 ms.

### 3.6 RUBBER-CONT-07 20 ms delayed handoff — PASS (assessment); every point coupling_delay_significant

| Strip L | hand | move | tip error | hand error | ζ_tip 20 ms / 1 ms | delay energy (20 ms) / E_mech | label |
|---|---|---|---|---|---|---|---|
| 30 mm | stiff 1 kg, 10 kN/m | 40 ms | 13.2 % | 1.8 % | −0.0006 / +0.0003 | 4.1 % | significant |
| 30 mm | stiff | 200 ms | 0.35 % | 0.23 % | +0.0003 / +0.0003 | 87 % | significant |
| 30 mm | light 0.05 kg, 100 N/m | 40 ms | 7.2 % | 1.1 % | −0.0028 / +0.0017 | 33 % | significant |
| 30 mm | light | 200 ms | 0.32 % | 0.28 % | −0.0017 / −0.030 | 93 % | significant |
| 60 mm | stiff | 40 ms | 5.6 % | 1.8 % | −0.030 / n.r. | 70 % | significant |
| 60 mm | stiff | 200 ms | 1.1 % | 0.23 % | −0.0012 / n.r. | 44 % | significant |
| 60 mm | light | 40 ms | 3.4 % | 1.3 % | −0.0067 / n.r. | 56 % | significant |
| 60 mm | light | 200 ms | 1.9 % | 0.43 % | +0.0009 / n.r. | 50 % | significant |

n.r. = fewer than three tip peaks in the decay window. Material books ≤ 0.10 % in all 16 runs: the delay never enters the
material's energy. Reading: with wrench feedback the 20 ms handoff's energy error is comparable to the strip's mechanical
energy at every point, and the fast moves have 3–13 % tip errors; the log-decrement tip damping of the 20 ms arm is
negative at 6 of 8 points (−0.0006 … −0.030), but the elastic strip has no damping and values of ~1e-3 are at the
estimator's noise level (the 1 ms arm also gives −0.030 at one point), so the damping column supports, not proves, energy
injection; the delay energy column is the robust measure. For slow moves the trajectory error is small (0.3–1.9 %). Ideal-tracking grips (all calibration tests, P2-6
`kinematic_grip_batched`) have no feedback and no delay error.

### 3.7 Not run

-06 self-contact (NOT_IMPLEMENTED_V1, declared); -08 batched L0 / L1 and -09 GPU isolation (batched contact path not
implemented, task B1 / B2).

## 4. GPU check `gpu_contact_002` (RUBBER-CONT-11, reported)

`flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp
PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/gpu_contact_check.py --out artifacts/rl_ready/p2_4/gpu_contact_002`
— commit `045ce6c`, clean tree, protocol sha `4d3dc68f…`, RTX 5090 GPU0, driver 570.211.01, Warp 1.17.0, **SHARED_GPU**
(co-tenant PID 642659, 766 MB, before and after; not touched), 31 GB free at start. The first attempt `gpu_contact_001`
failed after two rows (a NumPy bool in a JSON print) and is preserved.

| Variant (workload E, case A 150 × 20 × 4 mm, n_t 3, 41,220 particles, 2,585 substeps, F8 + kick-average + F9 + F10, guard 1.0, graph chunks 500) | ms / substep | env-steps / s | VRAM peak (cumulative) | L2 |
|---|---|---|---|---|
| no contact | 0.218 | 1.77 | 686 MB | PASS |
| table + 2 pads (pinching) | 0.320 (+46 %) | 1.21 | 744 MB | PASS |
| the same spec out of reach | 0.305 (+40 %) | 1.27 | 776 MB | PASS, every contact row exactly 0.0 |

Stage split (eager, 50 substeps, kernel time 0.341 ms): `node_penalty_contact` 0.076 ms (22 %; the per-node float64
slerp / quat-to-matrix of the two frames), F8 projection kernels 0.084 ms, `_bc` 0.030, `_musl` 0.026, `_p2g` 0.025.
GPU vs CPU (small pinch, 10 control steps): pad wrench 3.3e-6 relative, markers 1.5e-9 m (float atomics; reported).

## 5. CPU suites

`tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_batched_*.py
tests/test_repo_naming.py` (new: `test_rubber_contact.py` 12 = RUBBER-CONT-10, `test_rubber_p2_4_protocols.py` 3) at
`045ce6c` (src and tests unchanged since `51023d6`): **421 passed, 0 failed, 0 skipped** with `CUDA_VISIBLE_DEVICES=''`
(575 s) and **421 passed** with `CUDA_VISIBLE_DEVICES=0` under the GPU0 lock (545 s); logs in
`artifacts/rl_ready/p2_4/cpu_tests_001/`. None of the new tests needs local artifacts, torch or CUDA.

## 6. Open items

ADR §22.8: X1 (decision on the μ 0.8 / pinch FAILs, rcs-xhigh), B1–B3 (batched contact, GPU isolation, Lab coupling
adapter), P1 (contact kernel cost), S1 (self-penetration detector before P2-6), C1 (feedback coupling with a shorter or
iterated handoff), R1 (independent review). Independent rcs-review: NOT_RUN.
