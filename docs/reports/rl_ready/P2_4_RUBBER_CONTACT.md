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
| Task X1: law `grid_node_penalty_coulomb_v2` + μ envelope (ADR §22.9) | **DONE** (`0b33246`, protocol v2 `8842573`) | dashpots on the realised node velocity; sliding accurate μ ≤ 0.6, static μ ≤ 0.8; §8 |
| Formal CPU run `contact_v2_001` (`ffb099f`, clean) | -02, -03 **PASS**; -01, -04, -05 **FAIL** (μ 0.8 sliding / pinch 60 µm outside the envelope, stick-run books); -07 CARRIED_OVER; -09 NOT_RUN (B2-v2) | §8 |
| Task B2-v2: RUBBER-CONT-09 under the v2 rule, v2 law, GPU0 `gpu_cont09_v2_001` (`2b8cb9c`, clean, SHARED_GPU) | **PASS** (a, b, c) | 1,566 scalars, 0 failing, worst d / limit 0.34; v2-law batched cost rows reported; §9A |

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

ADR §22.8 / §22.9: X1 done (§8; X2 = defect 2, rcs-xhigh), B1–B3 (batched contact, GPU isolation, Lab coupling
adapter), P1 (contact kernel cost), S1 (self-penetration detector before P2-6), C1 (feedback coupling with a shorter or
iterated handoff), R1 (independent review). Independent rcs-review: NOT_RUN.

## 7. Batched contact path (tasks B1 / B2; branch `claude/rl-rubber-p24b`)

Branch `claude/rl-rubber-p24b` from `33f5fe3`. Material status **SIMULATOR_ONLY_UNCALIBRATED**. Evidence:
`docs/evidence/rl_ready/p2_4_contact_batched_001.json`. Raw outputs: `artifacts/rl_ready/p2_4b/` (not committed).
Nothing frozen changed (Phase 1 modules incl. `batched/constitutive.py`, `errors.py`, `particle_contacts.py`,
`material_batch.py`, `rl/lab_batched_env.py`; `configs/rl_benchmark/*`, `scripts/rl_ready/*`; the rubber protocols and
the frozen rubber modules). The single-env law files `rubber_contact.py` / `tlmpm_rubber_contact_v1.py` are NOT
modified (task X1 may change them on `claude/rl-rubber-p24`).

### 7.1 Implementation

- `batched/rubber_contact_batch.py` (new): `BatchedTLMPMRubberContact(BatchedTLMPMRubber)`. The contact kick is the
  single solver's own `TLMPMRubberContact._launch_contact`, bound unchanged (batched arrays under the single solver's
  `ct_*` names, `node_count` = all nodes), so the shared `rubber_contact` kernels run over all envs;
  `contact_work_end`, `node_displacement`, `slider_update` are launched at the single solver's points through three
  hooks of `BatchedTLMPMRubber.step` (`_after_p2g_grid_update`, `_after_final_grid_update`, `_after_position_update`;
  no-ops without contact). Scalar `ct_*` attributes are copied from the single reference (uniform signature); a v2 law
  that keeps to this pattern plugs in with no batched physics. A single-solver contact array without a declared batched
  row kind (`rubber_layout.CONTACT_ARRAY_KINDS`) is refused (`UNSUPPORTED_SOLVER`), so a new law's state cannot be
  silently left unbatched.
- Per-env frictions: `ct_fric` is the strided view `params[:, 3:5]` (friction_table, friction_pad); a masked reset with
  resampled frictions changes them with no extra array.
- Rows: node arrays at the env's node range; per-env `ct_work[N, 6]`, `ct_pen_max[N]`, `ct_counts[N, 3]`,
  `ct_energy[N, 2]`; planes `e*P + k`, frames `e*F + f`, boxes `(e*F + f)*B + b` (heterogeneous plane values, box
  geometry, frame origins; P, F, B, frame kinds, law and law parameters uniform, else `UNSUPPORTED_NONUNIFORM_BATCH`).
- `BatchedRubberMaterial` with contact: `advance(..., frame_rotation=, frame_translation=, frame_active=)` ([N, F, ...]
  END poses), contact begin-of-step (frames hold, per-step reactions / penetration / counts zeroed; `ct_work`
  cumulative), a wrench kernel → `plane_wrench [N, P, 6]`, `box_wrench [N, F, B, 6]`, the `MaterialCoupling` arrays
  `pad_wrench [N, G·B, 6]` (per-pad wrenches [N, G, B, 6] flattened, G = frames of kind pad; `pad_wrenches()` returns
  the 4-D view) and `pending_wrench [N, 6]` (zero: no edge / tool contact in rubber v1), a NONFINITE check of the contact
  wrenches, the contact energy every control step, `contact_read()` (the `last_contact` keys), masked reset of every
  contact array, `contact_audit`. Values are strip-local (= world for an identity strip pose); the strip-to-world
  transform and the finger recipients are **task B3, not done**.
- Contact off: lives without a ContactSpec build the unchanged `BatchedTLMPMRubber` (no contact array, no launch).

### 7.2 RUBBER-CONT-08 (CPU L0 / L1 / refusals) — **PASS**

`CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -p no:cacheprovider -v tests/test_rubber_contact_batched.py
--junitxml artifacts/rl_ready/p2_4b/cont08_cpu_001/junit.xml` — commit `742858a`, clean tree, protocol sha256
`4d3dc68f…6e4c`, Warp 1.17.0 CPU: **25 / 25 passed** (11 s).

- L0 byte-wise (dtype, shape, raw bytes) against the single `TLMPMRubberContact` after every control step, N = 1 and
  N = 3 heterogeneous (particle counts, mu0, rho, mu_table, mu_pad, plane offsets, frame geometry and motions), with the
  protocol options (F8 + kick-average + F9 + F10, guard 1.0): incline (tilted gravity, table + moving fixture frame,
  elastic), impact (P2-type rods, heterogeneous Prony terms padded to M = 2, per-env initial velocities), pinch (case A
  clamp, table + two pad frames of two boxes, squeeze / pull / roll, one pad released for a step; elastic and P2); also
  the v1 options and the projection law. Compared: every base, option and visco array, every contact array (incl.
  `ct_node_u`, `ct_xi`, `ct_obst`, `ct_vobs`, `ct_work`, the reactions), diagnostics, visco diagnostics, markers, group
  wrenches, the contact block and the per-pad wrenches. Two single-reference placeholders with no batched counterpart are
  excluded and asserted unused: `params64 = [[1.0]]` of the elastic visco class, and the F8 placeholder arrays of free
  scenes with no constrained node (the batch has never allocated either).
- L1: perturbing env 1's pad motion, friction or table plane changes no byte of envs 0 / 2; masked partial reset equals
  a fresh env (all reset arrays incl. contact and material outputs) and evolves like one; a reset with resampled
  frictions equals the single reference built with them; episodes after a full reset repeat byte-wise; vars()-complete
  classification with the contact arrays; contact ownership audit 0 (and a mutation is detected).
- Refusals: mixed contact / no contact, law, zeta_t, zeta_n, kappa_n, plane count, frame count, box count, frame kind;
  an undeclared single contact array.
- An empty-ContactSpec batch is byte-identical to the batch without contact (elastic and P2). Mutation checks (outside
  the suite): shifted friction columns, a dropped slider update, a dropped frame-hold copy are each detected by L0.

### 7.3 RUBBER-CONT-09 (GPU per-env isolation) — **FAIL** (criterion b, 1 of 108 rows); a and c PASS

`flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp
PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4b/gpu_contact_batched_check.py --out
artifacts/rl_ready/p2_4b/gpu_cont09_001` — commit `742858a`, clean tree, protocol sha `4d3dc68f…`, RTX 5090 GPU0,
driver 570.211.01, Warp 1.17.0, **SHARED_GPU** (co-tenant PID 642659, 766 MB, before and after, not touched; a second
co-tenant PID 3093155, 712 MB, at the end), 30,959 MB free at start, wall 74 s. Workload E (case A 150 × 20 × 4 mm,
n_t 3, 41,220 particles/env, 50 Hz, dt 7.74e-6 s, 2,585 substeps, the protocol options, guard 1.0), N = 4: env 1
pinched (squeeze 20 µm ramped over 3 steps, then held; ≈ 4.8 N per pad) and on the table z = 0, envs 0 / 2 / 3 with pads
+1 m and table z = -1 m; one batch, episodes A, A' (repeat), B (env 1 squeeze 30 µm, ≈ 7.1 N), 6 control steps each.

| Criterion | Verdict | Measured |
|---|---|---|
| a zero rows | **PASS** | every contact row of envs 0 / 2 / 3 (plane / box / pad / pending wrenches, ct_work, ct_pen_max, ct_counts, ct_energy, reactions, per-node impulses, slider state, obstacle ids = -1) exactly 0.0 after every step of A, A', B (and of the diagnostic episodes); env 1's rows nonzero every step |
| b perturbation band | **FAIL** | 107 of 108 (step, env, observable) rows pass; failing: step 3, env 2, group (clamp) wrench: max\|A − B\| 1.23e-7 N > max\|A − A'\| 2.09e-8 N + 1e-6 · 0.0708 N = 9.2e-8 N. Markers, energies (summary) and contact wrenches pass everywhere |
| c audits | **PASS** | node / cell / marker / group / traction / contact-row ownership audits 0; finite; invalid bits 0; guard violations 0 |

The reading of b (per control step, per env, per observable vector: max |A − B| ≤ max |A − A'| + 1e-6 · max |A|) was
implemented before any run and is the verdict. Diagnostic only (added after the SMOKE runs `gpu_cont09_smoke_001/002`,
reduced geometry, showed the same effect; never part of the verdict): with three A and two B episodes, 105 of 108 rows
have every A–B difference inside the A–A envelope (+1e-6 relative); the 3 outside are all the clamp wrench of an
out-of-reach env (A–B up to 1.2e-7 N vs A–A up to 8.2e-8 N); aggregated over all steps per observable, every row passes.
Reading (not a verdict): the out-of-reach envs' clamp wrench (F8 reaction, float64 atomics fed by the frozen float32
P2G / MUSL atomics) varies run to run on the GPU by up to ~1.7e-6 of its value, and a single repeat sample A vs A'
underestimates that spread; the A–B differences of envs 0 / 2 / 3 are of the same size as the repeat differences, the
contact rows are exactly zero, and the CPU L1 tests show byte isolation. No coupling is shown, but the declared
criterion is not met; a v2 criterion (e.g. a multi-repeat envelope) would have to be declared (rcs-xhigh) before a new
run. The v1 FAIL stays.

### 7.4 RUBBER-CONT-11 batched cost rows (reported, not scored)

Same run, rows sequential (not interleaved), GPU0 RTX 5090, driver 570.211.01, **SHARED_GPU** (co-tenant utilisation
samples 44 / 9 / 6 / 3 % at the row starts), commit `742858a`, clean tree, protocol sha `4d3dc68f…`. Workload E,
`batched_headless` (material only, BatchedRubberMaterial), 41,220 particles/env, 50 Hz, dt 7.74e-6 s, 2,585 substeps per
control step, cameras none, recording none, robot DOF 0, CUDA graph chunks of 500, 1 cold + 3 warm control steps
(median), reset = masked partial reset of one env (20 resets).

| Row | N | contacts | cold s | warm s | ms / substep | env-steps / s | sim-time / wall | reset p95 | VRAM peak (process, cumulative) | arrays | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no contact | 4 | none | 1.038 | 1.006 | 0.389 | 3.98 | 0.080 | 0.41 ms | 776 MB | 86 MB | L2 PASS |
| -09 configuration | 4 | table + 2 pads per env, env 1 in reach | 1.492 | 1.436 | 0.556 (+43 %) | 2.78 | 0.056 | 0.49 ms | 776 MB | 95 MB | L2 PASS |
| all pinched | 4 | table + 2 pads per env, all in reach | 1.577 | 1.495 | 0.579 (+49 %) | 2.68 | 0.054 | 0.48 ms | 776 MB | 95 MB | L2 PASS |
| N = 1 pinched | 1 | table + 2 pads | 0.847 | 0.786 | 0.304 | 1.27 | 0.025 | 0.39 ms | 776 MB | 24 MB | L2 PASS |

N = 4 with contact gives 2.1× the env-steps/s of N = 1 (2.68 vs 1.27). The contact cost (+43–49 %) matches the
single-env +40–46 % of `gpu_contact_002`; the per-node float64 frame slerp / quat-to-matrix is the known cost (task P1,
not done). The VRAM column is the process peak over the whole run (CUDA context included), not per row.

### 7.5 CPU suites

`tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_batched_*.py
tests/test_repo_naming.py` at `742858a` (tracked tree clean at start; only this report changed during the run):
**446 passed, 0 failed, 0 skipped** with `CUDA_VISIBLE_DEVICES=''` (579 s) and **446 passed** with `=0` under the GPU0
lock (585 s); 2 warnings (scipy IntegrationWarning in `rubber_visco_reference.py`, pre-existing). Logs in
`artifacts/rl_ready/p2_4b/cpu_tests_001/`. New: `tests/test_rubber_contact_batched.py`
(25). None of the new tests needs local artifacts, torch or CUDA.

### 7.6 Open

B3 (MaterialCoupling adapter: strip-to-world wrenches, finger recipients, `box_centers` / `set_measured_poses` /
`reset_masked(mask)` of the coupling surface), a declared decision on the RUBBER-CONT-09 b FAIL (criterion v2 or
otherwise), P1 (contact kernel cost), S1, C1, merge with the X1 law decision, independent review: NOT_RUN.

## 8. Task X1 — decision on the v1 contact FAILs (ADR 0023 §22.9)

Branch `claude/rl-rubber-p24` (B1 merged at `77cdd10`). Material status **SIMULATOR_ONLY_UNCALIBRATED**. Evidence:
`docs/evidence/rl_ready/p2_4_contact_x1_001.json`. Protocol `configs/rubber/rubber_contact_tests_v2.json`, sha256
`89bc7680bb9c2ed6322279b198899cef0faf31c4751b6bd849b22ecd24a339d0` (pinned in `tests/test_rubber_p2_4_protocols.py`,
declared alone at `8842573` before any run of it). Every v1 result and FAIL (`contact_v1_001`, `gpu_cont09_001`) stays.

### 8.1 Root cause (DIAGNOSTIC)

Scripts `scripts/rubber/p2_4/diag_contact_x1.py` (per-substep per-node impulses on the INC block), `x1_dev_laws.py`
(dev law switches), `diag_x1_scenes.py` (v1 scenes with a substituted law), `halfspace_sliding_stability.py` (analytic);
raw outputs `artifacts/rl_ready/p2_4/x1_diag_001` (README with the code state of each sweep), `x1_v2c1_scenes_001`,
`x1_v2soft_scenes_001`, `x1_v1_identity_002`, `x1_halfspace_stability_001.txt`.

| Observation (v1 law, INC block 16 × 8 × 4 mm, ratio 1.2) | μ 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 |
|---|---|---|---|---|---|---|
| slip error / (g cosθ) | +0.02 % | +0.06 % | +0.26 % | +1.88 % | +6.8 % | +11.9 % |
| friction applied / μΣJ_n | 0.999 | 0.998 | 0.996 | 0.979 | 0.947 | 0.881 |
| bottom nodes in contact per substep | 0.28 | 0.19 | 0.08 | 0.03 | 0.02 | 0.009 |

- **Defect 1 (law):** the v1 dashpots act on the predicted lumped velocity, not on the realised drift velocity Bπ. A
  frictionless block at rest rattles for 0.3 s (27 % of the bottom nodes in contact, vibration energy ~100× the static
  elastic energy) with positive kick-paired normal work (+1.1e-8 J) while the nominal dashpot dissipation is 3.0e-6 J.
  With the dashpots on the realised velocity it settles (100 % in contact, W_n < 0).
- **Defect 2 (formulation):** friction–normal mode-coupling flutter of the grid-scale penalty interface (the cap follows
  the per-node penalty force instantaneously); grows with μ; not removed by cfl 0.25 / 0.125, n_t 4 / 6, κ 0.0005–1 or
  ζ to the limit; continuum rigid-flat Coulomb sliding is stable for μ < 1.000 (ill-posed above), so below μ = 1 it is a
  discretisation defect.
- Ruled out: dashpot energy missing from μN·slip (wrong sign), stick–slip hysteresis (0.2 % of μN), slip direction
  (≤ 0.4 %), cone force timing / content (flutter persists; spring-only cone worse).
- A normal-force memory of the cap (lag n_τ dt) removes defect 2 in steady sliding (to μ 1.0 / 1.2 / 1.5 at n_τ 100 /
  200 / 400) but cascades edge micro-slip near threshold under rising load (0.04–0.5 mm at 0.95 μ) and collapses where it
  does not stabilise: **not adopted** (task X2).

### 8.2 Decision

Route (a) + (b): law **`grid_node_penalty_coulomb_v2`** (`0b33246`; opt-in; parameters unchanged; no new state, so the B1
batched path runs it unchanged — L0 byte-identical N = 3, `tests/test_rubber_contact_v2.py`) corrects defect 1; defect 2
is declared as a **μ envelope** with a measured error model: sliding friction to the protocol limits for μ ≤ 0.6, static
friction (hold at 0.95 μ) for μ ≤ 0.8; μ 0.8 sliding +1.3 … +2.5 % of g cosθ, dissipation 0.87–0.94; μ 1.0 +2.7 … +5.8 %,
hold lost; μ ≥ 1.0 continuum ill-posed. Labels `contact_sliding_friction_deficit_v2`, `contact_friction_flutter`,
`contact_coulomb_ill_posed`. **Limitation:** real rubber friction (0.6–1.0+) and the recipes' range [0.3, 1.5] (nominal
0.8 / 1.0) exceed the envelope; until task X2, scored rubber contact rows stay inside it or carry the labels.

n_t 2 impact books: n_t 2 is the reduced_fidelity `rubber_coarse_v1` profile; the v1 1.024 % is the coarsest-dt O(dt)
books error (n_t 4: 0.59 / 0.25 %); the 1 % limit is kept (v2: 0.71 %). Pinch clamp balance: the v1 per-step metric is
the strip's axial vibration (229–286 Hz ≈ c/4L = 267 Hz), unattainable for the exact undamped solution; the window-mean
balance (0.002–1.35 %) shows no clamp / contact work coupling; v2 scores the window mean at 5 %. RUBBER-CONT-09 b: the
v1 single-repeat band rejects any noisy scalar with probability 1/2 under H0 (unsound); v2 rule k = 5 per side, λ = 4,
FWER ≤ 0.4 % (`cont09_band_calibration.py`); a, c strict; the GPU rerun is task B2-v2.

### 8.3 Formal CPU run `contact_v2_001`

`CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/run_contact_v2.py --out
artifacts/rl_ready/p2_4/contact_v2_001 --jobs 32` — commit `ffb099f`, clean tree, protocol sha256 `89bc7680…a339d0`,
Warp 1.17.0 CPU, Python 3.12, 34 jobs, 35 s, shared Threadripper PRO 9975WX; every job records the law it built
(`grid_node_penalty_coulomb_v2` only). Smoke `contact_v2_smoke_001` preserved (SMOKE).

| Test | v1 `contact_v1_001` | v2 `contact_v2_001` |
|---|---|---|
| -01 a stick | FAIL (P2 at 0.95: 4.8 s) | **PASS** (10 rows incl. elastic and P2 at 0.95 for μ 0.6 / 0.8; ≤ 7e-5 s) |
| -01 b slip | FAIL (μ 0.8: +1.65 / +1.88 / +1.06 %) | **FAIL** (μ 0.8: +1.83 / +1.26 / +2.46 %, outside the envelope, declared expected); μ 0.4 ≤ +0.09 %; μ 0.6 (unseen INC2) +0.78 / +0.42 / +0.38 % |
| -01 c onset | FAIL (μ 0.8: −2.75 %) | **PASS** (μ 0.4 / 0.6 / 0.8: −0.31 / −1.61 / −1.47 %) |
| -01 d books | FAIL (1.03 %) | **FAIL** (3 stick runs 1.67 / 1.28 / 1.19 % of S ≈ 2.5e-10 J, absolute ≤ 4e-12 J; slip runs ≤ 0.09 %; order 0.57) |
| -01 e validity | PASS | PASS |
| -02 | PASS | **PASS**; f: Σ k_n δ_I / reaction = 1.000 for every damped run (v1 0.49–0.50 by design) |
| -03 | FAIL (n_t 2 books 1.024 %) | **PASS** (books 0.36 / 0.21 % n_t 4, 0.71 % n_t 2, P2 0.30 / 0.15 / 0.52 %; t_c +3.3 %; D_contact 2.8 / 3.1 / 4.7 % of E0, convergence 0.60) |
| -04 | FAIL (μ 0.8: 0.932 / 0.930 / 0.943) | **FAIL** (μ 0.8: 0.931 / 0.943 / 0.869, declared expected); μ 0.4 0.986–0.997; μ 0.6 (INC2) 0.954 / 0.969 / 0.972 |
| -05 | FAIL (ratio 0.972 / 0.964 / 0.956; per-step balance 9.8 / 6.0 / 6.3 %) | **FAIL** (ratio 0.978 / 0.982 / **0.964**); cone ≤ 0.9999, scaling 1.9 %, penetration ≤ 0.043 s PASS; window-mean balance 0.78 / 1.35 / 0.45 % PASS (per-step 9.8 / 7.0 / 6.9 % reported) |
| -06 | NOT_RUN | NOT_RUN (NOT_IMPLEMENTED_V1) |
| -07 | PASS | CARRIED_OVER (contact-free; v1 result applies) |
| -08 | PASS (B1, v1 law) | v2: L0 N = 3 incline / pinch / pinch P2 byte-identical, v1 / v2 mix refused (pytest) |
| -09 | FAIL b (1 / 108), a / c PASS (B1) | NOT_RUN (task B2-v2; rule declared) |

### 8.4 CPU suites

`tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_batched_*.py
tests/test_repo_naming.py` at `ffb099f` (src and tests unchanged since; B1 merged): **455 passed, 0 failed, 0 skipped**
with `CUDA_VISIBLE_DEVICES=''` (579 s) and **455 passed** with `=0` under the GPU0 lock (578 s); 2 pre-existing scipy
warnings; logs `artifacts/rl_ready/p2_4/cpu_tests_x1_001/`. New: `tests/test_rubber_contact_v2.py` (8), the v2 protocol pin.
None of the tests needs local artifacts, torch or CUDA.

## 9. Task B2-v2 — RUBBER-CONT-09 under the v2 rule (branch `claude/rl-rubber-p24b`)

> Numbering (K1 consolidation, 2026-11-11): on `claude/rl-rubber-p24b` (`c234aee`) this section was 9A and the
> X2 section 9B (both had been 9 on their own branches). The X3 numbering of `claude/rl-rubber-p24` is adopted: 9
> B2-v2, 10 X2, 11 X3. "section 9" in `p2_4_contact_batched_002.json` means this section 9; the `report` field of
> `p2_4_contact_x2_001.json` names section 10; `p2_4_contact_x3_001.json` names section 11. References to 9A / 9B
> written on `claude/rl-rubber-p24b` mean 9 / 10.

Protocol `configs/rubber/rubber_contact_tests_v2.json` (sha256 `89bc7680…a339d0`, checked by the script before it runs;
thresholds k_per_side 5, λ 4, eps_rel 1e-6 read from it), law **`grid_node_penalty_coulomb_v2`** as the protocol
declares for -09 ("the -09 v1 configuration with the v2 law"; the batch's `ct_law` and options receipt record it).
Script `scripts/rubber/p2_4b/gpu_contact_batched_check_v2.py` (new; the v1 script and `gpu_cont09_001` are unchanged and
preserved); rule arithmetic unit-tested on CPU in `tests/test_rubber_cont09_rule_v2.py` (7 tests: hand values, d =
brute-force max pairwise |A_i − B_j|, inclusive boundary, shift > (λ + 1) s rejected, floor on the vector's largest
|A| component, aggregation, protocol pin). Evidence `docs/evidence/rl_ready/p2_4_contact_batched_002.json` (writer
`write_contact_batched_evidence_v2.py`). Raw `artifacts/rl_ready/p2_4b/gpu_cont09_v2_001` (smoke, reduced geometry
30 × 10 × 4 mm n_t 2, script check only: `gpu_cont09_v2_smoke_001`).

`flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp
PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4b/gpu_contact_batched_check_v2.py --out
artifacts/rl_ready/p2_4b/gpu_cont09_v2_001` — commit `2b8cb9c`, clean tree, RTX 5090 GPU0, driver 570.211.01, Warp
1.17.0, Python 3.12, **SHARED_GPU** (co-tenant PID 642659, 766 MB, in every sample, not touched), 30,959 MB free at
start, wall 117 s. Configuration as §7.3: workload E (case A 150 × 20 × 4 mm, n_t 3, 41,220 particles/env, 19,200
nodes/env, 50 Hz, dt 7.74e-6 s, 2,585 substeps, F8 + kick-average + F9 + F10, guard 1.0), N = 4, env 1 pinched (A:
20 µm, B: 30 µm, ramp over 3 steps) on the table, envs 0 / 2 / 3 out of reach; **10 episodes in one batch, interleaved
A1, B1, …, A5, B5, each from a full reset**, 6 control steps each. Observables per the v2 protocol: markers, energies
summary, group (clamp) wrench, plane / box / pad / pending wrenches (the v1 script omitted the pending wrench, which is
zero by construction).

| Criterion | Verdict | Measured |
|---|---|---|
| a zero rows | **PASS** | every contact row of envs 0 / 2 / 3 (plane / box / pad / pending wrenches, ct_work, ct_pen_max, ct_counts, ct_energy, reactions, per-node impulses, slider state, obstacle ids −1) exactly 0.0 after every step of all 10 episodes |
| b perturbation band (v2) | **PASS** | 126 (step, env, observable) rows, **1,566 scalars, 0 failing**; worst d / (λ s + floor) = **0.34** (step 1, env 0, clamp wrench F_x: d 2.21e-7 N, s 1.50e-7 N, floor 5.2e-8 N, limit 6.53e-7 N); every row with d / limit > 0.2 is the clamp wrench of an out-of-reach env; 720 scalars bitwise deterministic (s = 0, d = 0) |
| c audits | **PASS** | node / cell / marker / group / traction / contact-row ownership audits 0; finite every step; invalid bits 0; guard violations 0 |
| validity | met | env 1's contact rows nonzero every step; env 1's pad wrench differs for every A–B pair (last-step top-pad F_z A 3.870–3.875 N, B 5.827–5.829 N) |

**Verdict RUBBER-CONT-09 (v2): PASS.** The v1 FAIL of `gpu_cont09_001` (§7.3) stays preserved.

Diagnostic (not part of the verdict): the same rule applied to env 1 (the perturbed env) rejects 95 of its 522 scalars
(worst d / limit 1,447, the pad wrench at step 0), so the rule has power at the size of the A–B change; no A episode is
bitwise identical to another for any observable except the pending wrench (GPU float atomics: within-side range of
the out-of-reach clamp wrench up to 2.0e-7 N on a reaction of 0.03–0.15 N); the largest d / limit of any other
observable is 0.06 (markers).

Finding (calibration text, not the rule): the protocol's calibration statement assumed ≤ 648 scalars; the declared
observables have 1,566 (846 with s > 0), so its family-wise H0 bound is 0.86 % over all scalars (0.47 % over those with
s > 0) by the union bound at 5.5e-6 per scalar, not 0.4 %. The rule and its thresholds are unchanged by this; noted for
the R1 review.

Erratum §7.3 / §7.4: the "second co-tenant PID 3093155, 712 MB, at the end" of `gpu_cont09_001` was that run's own
process (its `meta.pid`); the SHARED_GPU label stands because of PID 642659. The utilisation samples in §7.4 and below
are GPU-wide and include this process.

### 9.1 RUBBER-CONT-11 batched cost rows, v2 law (reported, not scored)

Same run, rows sequential after -09, GPU0 RTX 5090, driver 570.211.01, **SHARED_GPU** (PID 642659 in every sample;
GPU-wide utilisation before / after each row 47 / 36, 0 / 4, 0 / 72, 3 / 69 %), commit `2b8cb9c`, clean tree, protocol
sha `89bc7680…`. Workload E, `batched_headless` (BatchedRubberMaterial, material only), 41,220 particles/env, 50 Hz,
dt 7.74e-6 s, 2,585 substeps per control step, cameras none, recording none, robot DOF 0, CUDA graph chunks of 500,
1 cold + 3 warm control steps (median), reset = masked partial reset of one env (20 resets). The protocol's `note_v2`
says the v2 kernel has the v1 cost and need not be re-measured; these rows measure it.

| Row | N | contacts | law | cold s | warm s | ms / substep | (v1 law, §7.4) | env-steps / s | sim-time / wall | reset p95 | VRAM peak (process, cumulative) | arrays | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no contact | 4 | none | – | 1.066 | 1.026 | 0.397 | 0.389 | 3.90 | 0.078 | 0.44 ms | 776 MB | 90 MB | L2 PASS |
| -09 configuration | 4 | table + 2 pads per env, env 1 in reach | v2 | 1.493 | 1.453 | 0.562 (+42 %) | 0.556 | 2.75 | 0.055 | 0.48 ms | 776 MB | 100 MB | L2 PASS |
| all pinched | 4 | table + 2 pads per env, all in reach | v2 | 1.585 | 1.543 | 0.597 (+50 %) | 0.578 | 2.59 | 0.052 | 0.64 ms | 776 MB | 100 MB | L2 PASS |
| N = 1 pinched | 1 | table + 2 pads | v2 | 0.843 | 0.800 | 0.309 | 0.304 | 1.25 | 0.025 | 0.47 ms | 776 MB | 25 MB | L2 PASS |

The v2 law costs the same as v1 within the shared-GPU noise of these 3-step medians (+1–3 %, the no-contact row moved
+2 % too). N = 4 with contact: 2.1× the env-steps/s of N = 1. Arrays in MB = 10^6 bytes; the byte counts are identical
to the v1-law rows (§7.4 listed them in MiB as 86 / 95 / 24 MB).

### 9.2 CPU tests

`tests/test_rubber_cont09_rule_v2.py tests/test_rubber_contact_v2.py tests/test_rubber_contact_batched.py
tests/test_rubber_p2_4_protocols.py tests/test_repo_naming.py` at `2b8cb9c` (clean): **46 passed** with
`CUDA_VISIBLE_DEVICES=''` (27 s) and **46 passed** with `=0` under the GPU0 lock (28 s); logs
`artifacts/rl_ready/p2_4b/cpu_tests_v2_001/`. No src change in B2-v2; the full suite was not rerun (last: 455 passed at
`ffb099f`, §8.4). None of the tests needs local artifacts, torch or CUDA.

## 10. Task X2 — defect 2 (friction–normal flutter): negative result (ADR 0023 §22.10)

Branch `claude/rl-rubber-p24` from `2b2221b`. Material status **SIMULATOR_ONLY_UNCALIBRATED**. Label **DIAGNOSTIC**: no
law adopted, no protocol declared, no formal run. Evidence `docs/evidence/rl_ready/p2_4_contact_x2_001.json`; raw
`artifacts/rl_ready/p2_4/x2_diag_001` (README); scripts `x2_dev_laws.py`, `diag_contact_x2.py`, `diag_x2_pattern.py`
(`b019684`, `331550c`); identity reruns at both commits identical in every field except wall time. Nothing in `src`,
`tests` or `configs` changed. Command per row: `CUDA_VISIBLE_DEVICES='' .venv/bin/python
scripts/rubber/p2_4/diag_contact_x2.py <args>` (Warp 1.17.0 CPU, Python 3.12.12).

### 10.1 Character of the flutter (v2, INC)

| Observation | Value |
|---|---|
| bottom nodes in contact per substep, μ 0 / 0.2 / 0.8 | 100 % / 60 % / 4.4 % |
| v_z rms at μ 0.8 (sliding speed 0.12 m/s) | 16 mm/s; dominant 1.5–3 kHz; peak 1–3 waves over the face |
| μ 0.8 slip error, cfl 0.5 / 0.25 / 0.125 | +1.26 / +1.12 / +1.15 % (dt-converged) |
| μ 0.8 slip error, n_t 3 / 4 / 5 / 6 / 8 | +1.26 / +2.0 / +1.7 / +2.8 / +4.6 % (grows) |
| μ 1.0 slip error, n_t 3 / 4 / 6 | +3.8 / +6.2 / +9.9 % |
| HF loss probe (Prony g 0.15, τ 100 µs), μ 0.8 n_t 3 / 4 | ≤ 0.18 / ≤ 0.28 %; μ 1.0: +4.5 % |

Because the deficit is converged in dt, time-integration changes of the contact (implicit / return-mapped tangential
update, Moreau projection, augmented-Lagrangian or decoupled normal stiffness) cannot remove it; κ 1 and ζ 1 do not
either. It grows under dx refinement at every μ tested, which the half-space analysis of §22.9 (contact kept) does not
explain; that statement is withdrawn as unproven (task X4).

### 10.2 Candidates

| Candidate | μ 0.8 sliding | μ 1.0 sliding | Hold 0.95 μ | Robustness | Verdict |
|---|---|---|---|---|---|
| nonlocal cap, radius 2–3 nodes | 0.6–0.7 % | 2.6–8 % | – | radius shrinks with dx | rejected |
| patch-level cap | 0.12–0.91 % (n_t 3) | 2.5 % | to μ 1.0 | cfl 0.25: 1.16 %; INC2 D04 0.947; μ 0.9 2.4 % | rejected |
| patch cap + lagged total τ 0.25 ms | exact (n_t 3) | 4–8 % | to μ 1.0 | n_t 4: +1.7 % (μ 0.8); no recovery | rejected |
| per-node lag τ 0.5 / 1 / 3 ms | exact | −1.9 % / exact / exact | ramp-phase slip 10.5 / 52 µm / 0.2–0.5 mm (v2 0.19 µm) | no recovery (−12 %); τ 0.25 ms −7 … −20 % | rejected |
| memory across bounces (± gate) | exact from rest | exact from rest | lost at τ ≥ 1 ms | friction collapse after bouncing (+75 %, +99 %) | rejected |
| max / slip-distance / patch-slip-gated memories | −1.4 … −15 % | −14 … +0.4 % | – | flutter not suppressed | rejected |
| **patch slip-rate equalisation (mode 13, ζ_d 1)** | **exact at n_t 3 / 4 / 6, cfl 0.25, INC2; recovers** | exact at n_t 3 / cfl 0.25 / INC2; **n_t 4 +2.4 %, n_t 6 +4.2 %** | to μ 1.5; ramp-phase 0.16 µm; μ 0.8 at n_t 4 holds | μ ≥ 1.2 fails | promising, not adopted |

### 10.3 Decision

Negative result for the X2 requirement: no v3, no protocol v3, no formal run. v2, `rubber_contact_tests_v2` and the
labels stay. The mode-13 design is the recommended v3 (task X3: src + batched + pads, then a declared protocol scoring
μ ≤ 0.8 at n_t 3 and 4). Qualifications (ADR §22.10): the v2 envelope holds at n_t 3 on INC / INC2 but not at
`rubber_ref_v1` (μ 0.8 hold slides 3.9 mm at n_t 4; μ 0.6 sliding +1.2 / +1.7 % on INC2 at n_t 4, +1.06 % on INC at
n_t 6; INC2 at μ 0.8: +2.5 … +8.3 %). -01 d: the stick-run books residual (2.3–4.3e-12 J) is a real, slowly accumulating
drift (static jitter 5–12 µm/s dissipated by the dashpots), far above float64 resolution; no rule change, the v2 FAIL
stands (task X5).

### 10.4 CPU suites

`tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_batched_*.py
tests/test_repo_naming.py` at `b019684` (src and tests unchanged since `2b2221b`): **455 passed** with
`CUDA_VISIBLE_DEVICES=''` (608 s) and **455 passed** with `=0` under the GPU0 lock (606 s); logs
`artifacts/rl_ready/p2_4/cpu_tests_x2_001/`.

## 11. Task X3 — law `grid_node_penalty_coulomb_v3`, the μ ≤ 0.8 envelope at the acceptance profile (ADR 0023 §22.11)

Material status **SIMULATOR_ONLY_UNCALIBRATED**. Law `b668bdc` (opt-in; v1 / v2 kernels byte-identical, v1 default).
Protocol `configs/rubber/rubber_contact_tests_v3.json`, sha256 `36d70bea…a856`, declared and pinned alone at `e19f488`
before any formal run (every v2 threshold copied verbatim; new -01 f load-ramp slip at 0.01 s; scored μ 0.2–0.8 on INC
and INC2 at n_t 3 and n_t 4; PINCH at n_t 3 and 4; μ 1.0 rows reported with labels only; -01 d unchanged). Evidence
`docs/evidence/rl_ready/p2_4_contact_x3_001.json`.

### 11.1 Law

v2 plus the patch slip-rate equalisation (X2 dev mode 13, ζ_d = 1): per env and obstacle face (each plane, each face
of each box; rows e·Q + q, Q = P + 6FB), the slipping nodes of a patch with μ > 0 get −dt c_d (v_t − v̄) outside the
Coulomb cap and the slider, c_d = 2ζ_d√(k_t m_I), v̄ their mass-weighted mean realised tangential velocity in the same
substep. The impulses sum to zero per patch (patch total ≤ μΣF_n); stick nodes, slip onset and static friction are v2's.
Two passes (`node_penalty_contact_v3_normal` / `_tangential`), one source for the single and the batched solver.

### 11.2 Formal CPU run `contact_v3_001`

`CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/run_contact_v3.py --out
artifacts/rl_ready/p2_4/contact_v3_001 --jobs 48` — commit `c631cca`, clean tree, 137 jobs, 133 s; smoke
`contact_v3_smoke_001` preserved (SMOKE).

| Group | slip runs: max \|err\| / g cosθ | D04 range | stick runs: max \|disp\| / s | load-ramp max / s | books max (stick) |
|---|---|---|---|---|---|
| INC n_t 3 | 9.1e-8 | 0.99994–0.99997 | 6.2e-5 | 2.9e-4 | 1.64 % (9 FAIL) |
| INC n_t 4 | 1.7e-7 | 0.99995–0.99997 | 8.9e-5 | 4.0e-4 | 2.05 % (11 FAIL) |
| INC2 n_t 3 | 1.8e-7 | 0.99989–0.99995 | 9.1e-5 | 4.4e-4 | 0.51 % |
| INC2 n_t 4 | 1.2e-7 | 0.99991–0.99996 | 1.3e-4 | 5.9e-4 | 0.62 % |

| Test | v2 (`contact_v2_001`) | v3 (`contact_v3_001`) |
|---|---|---|
| -01 a / b / c / e | b FAIL at μ 0.8 | **PASS** (n_t 3 and 4, INC and INC2, μ 0.2–0.8, P2 holds) |
| -01 d books | FAIL (stick runs) | **FAIL** (20 INC stick runs, 1.03–2.05 % of S, ≤ 5.2e-12 J; declared expected, not redefined) |
| -01 f load-ramp slip | – | **PASS** (≤ 5.9e-4 s) |
| -02 / -03 | PASS | **PASS** (frictionless: v3 = v2) |
| -04 | FAIL (μ 0.8: 0.87–0.94) | **PASS** (≥ 0.9999) |
| -05 (n_t 3 / n_t 4) | FAIL a (0.964) | **PASS**: ratios 1.000 / 0.999 / 0.989 and 1.000 / 0.988 / 0.986; cone ≤ 1.000; window balance ≤ 1.4 % |
| μ 1.0 (reported) | – | n_t 3 exact; n_t 4 INC r1.1 +2.4 %, INC2 r1.1 +0.68 % (D04 0.91–0.95): `contact_coulomb_ill_posed` |

### 11.3 Batched path and GPU

- RUBBER-CONT-08 (`tests/test_rubber_contact_v3.py`, 16 tests): **PASS** — L0 byte-wise N = 1 and N = 3 heterogeneous
  (incline, pinch, pinch P2, impact; patch rows and per-pad wrenches included), equalisation exercised in the L0 run,
  L1 isolation (pad motion / friction / plane), masked reset = fresh env, vars()-complete, v2 / v3 mix refused; kernel
  level net-zero (residual 2e-23 against 1e-7 impulses) and env-indexed patch sums; ζ_d = 0 reproduces v2 byte-wise.
- RUBBER-CONT-09 (`gpu_cont09_v3_001`, `scripts/rubber/p2_4/gpu_contact_batched_check_v3.py` = the B2-v2 script with the
  v3 protocol and law; GPU0 lock, SHARED_GPU, co-tenants 3468960 / 642659 untouched): **PASS** — a exact zeros; b 1,566
  scalars, 0 failing, worst d / limit 0.35; c audits 0 incl. the patch-row audit. Smoke `gpu_cont09_v3_smoke_001`
  preserved (SMOKE).
- RUBBER-CONT-11 (reported): batched v3 vs v2 (B2-v2 rows) 0.579 vs 0.562 ms per substep (N 4, -09 configuration),
  0.620 vs 0.597 (N 4 all pinched), 0.323 vs 0.309 (N 1), 0.398 vs 0.397 (no contact); single env workload E
  (`scripts/rubber/p2_4/gpu_contact_cost_v3.py`, `gpu_contact_cost_x3_v2_001` / `_v3_001`): contact 0.322 → 0.332 ms
  per substep (+3.1 %), out of reach 0.306 → 0.313, VRAM 712 → 718 MB.

### 11.4 Decision

v3 is the law for scored rubber contact with μ ≤ 0.8 at n_t 3 and 4 (opt-in; v1 stays the frozen default). Every
criterion of `rubber_contact_tests_v3` passes except -01 d (stick-run books, declared expected, not redefined; X5).
μ > 0.8 sliding rows carry `contact_friction_flutter`, μ ≥ 1.0 also `contact_coulomb_ill_posed`. Declared limitations:
slip-rate waves inside a slipping patch are suppressed; disjoint contact regions on one obstacle face share one patch; a
net patch torque remains. All earlier FAILs (`contact_v1_001`, `contact_v2_001`, `gpu_cont09_001`) and the X2 negative
result stay.

### 11.5 CPU suites

`tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_batched_*.py
tests/test_repo_naming.py` at `eab4ef4` (src unchanged since `b668bdc`): **478 passed** with `CUDA_VISIBLE_DEVICES=''`
(592 s) and **478 passed** with `=0` under the GPU0 lock (585 s); logs `artifacts/rl_ready/p2_4/cpu_tests_x3_001/`. New:
`tests/test_rubber_contact_v3.py` (16), the v3 protocol pin. None of the tests needs local artifacts.
