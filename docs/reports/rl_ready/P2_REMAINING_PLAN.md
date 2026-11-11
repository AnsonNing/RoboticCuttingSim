# Phase 2 remaining plan — P2-4 completion, P2-5 calibration (synthetic), P2-6 RL-ready rubber, Phase 2 Done

Date: 2026-11-11. Branch `claude/rl-rubber-p24b` at the merge `6b36b65` (= `ff15cd2` B2-v2 + `b8b2b5e` X1/X2 of
`claude/rl-rubber-p24`). Author: rcs-xhigh. Status: **PLAN** (decisions binding once ADR 0023 §23 is accepted; nothing
here is a result). This round ran two read-only CPU probes (§9); no GPU job, no `src/`, `tests/` or `configs/` edit.
Material status everywhere: **`SIMULATOR_ONLY_UNCALIBRATED`**. Requirements: [RL_READY_SPECIFICATION.md](../../RL_READY_SPECIFICATION.md)
Phase 2 (P2-4…P2-6, "Phase 2 Done"); [RL_READY_PLAN.md](../../RL_READY_PLAN.md) §1, D-03, D-07, D-12; ADR 0023 §11–§13,
§18–§22. Status words: PASS / FAIL / BLOCKED / NOT_RUN (+ PASS_WITH_LIMITATIONS for reviews). A skipped test is not a
pass. Sizes: S ≈ ½ day, M ≈ 1 day, L ≈ 2–3 days nominal (the ADR 0023 §15 scale); §7.3 converts to expected time.

## 0. Decisions at a glance

1. **B3 (robot-feedback coupling adapter) is not needed for P2-6 and is deferred**, together with C1 (shorter /
   iterated handoff), to a post-Phase-2 "robot-coupled rubber" package. P2-6 uses kinematic grips (ideal tracking,
   no wrench feedback, so no coupling error by construction, ADR §22.4). RUBBER-CONT-07 already assessed the 20 ms
   handoff: every envelope point is `coupling_delay_significant` (delay energy 4–93 % of the strip's mechanical
   energy, tip errors up to 13 %), so a B3 adapter without C1 could only produce labelled rows, and the cutting
   `MaterialCoupling` surface (edge pose + cut digit) would also need a new rubber Lab task. Declared limitation:
   `robot_feedback_coupling: NOT_IMPLEMENTED_PHASE2`.
2. **S1 (self-penetration detector) is required before P2-6**, as ADR §22.3 states. It is a check, not contact physics:
   surface particles of one env closer than the particle spacing s whose reference distance exceeds 3·dx. It runs
   **inside the CUDA graph every `check_substeps` (≈ 1 ms)**, not once per 20 ms, so pass-through between checks
   needs relative speeds above ≈ 1.7 m/s. It uses the env-grouped sort-grid pattern of Phase 1 `FallbackGrid`, not a
   `wp.HashGrid`. A hit sets the sticky bit `RUBBER_SELF_CONTACT_UNRESOLVED` (8); training mode terminates the env.
3. **P1 (contact-kernel cost) is deferred.** The P2-6 scene has a table half-space and no contact frames. With F = 0
   the kernel skips the per-node frame slerp (`rubber_contact.py:416-470`) that caused the +40–50 % of §22.7 / §9A.1.
   Plane-only cost is measured in RL-07. P1 returns only with pad scenes (B3 / C1).
4. **μ envelope → P2-6 friction range.** P2-6 randomises only `friction_table`. Pads and self-contact are absent; their
   frictions are fixed at the recipe value and declared unused. The range is uniform on **[0.3, μ_slide,max]**, the
   minimum over the slot set's n_t of the *formally scored* sliding limit of the chosen law. Expected values:
   - law v3 (if `rubber_contact_tests_v3` passes): μ_slide,max = **0.8**;
   - law v2 at n_t ≥ 4: **0.4**, but only after a formal envelope check (ENV2). Today the n_t ≥ 4 v2 rows are
     DIAGNOSTIC (§22.10).

   μ ≥ 1.0 is refused (`UNSUPPORTED_FRICTION_OUT_OF_ENVELOPE`). It stays the labelled limitation
   `contact_coulomb_ill_posed`. Diagnostic evidence may narrow the envelope, never widen it.
5. **Q1w (RUBBER-PHYS-11 v5 FAIL, GCI 2.16 % vs 2 % at n_t 4): recommend option (c), accept n_t 4 with its measured
   +0.62 % bias as a labelled limitation.** The v5 FAIL stays recorded. No v6 rule, because relaxing Fs after seeing
   p_obs 3.12 would loosen the evaluator. Do not raise wide strips to n_t 6: that costs ≈ 5× per env-step everywhere
   and needs an n_t 12 run to verify. The bias is ≈ 4× smaller than the declared β = 20 model-form error. Details in §5.
   This is a reviewer decision (task K2).
6. **P2-5 = calibration interface + synthetic numerical validation `rubber_calib_tests_v1`.**
   - **Interface:** manifest schema v2 with `data_origin`, a population-batched forward model, a fit API (scipy TRF as
     the formal method; an in-repo CMA-ES; an ask/tell hook for the external agentic optimizer or BO), and a one-shot
     held-out guard.
   - **Synthetic truth:** generated at a finer resolution (n_t 6, CFL 0.25) with sensor noise and hidden, sealed
     parameters.
   - **Splits:** fit on some rates and geometries; held out on other rates, lengths, thicknesses and initial
     deformations.
   - **Tolerances:** a declared formula evaluated on a separate pilot dataset, frozen before the formal dataset exists.
   - **Label:** `NUMERICAL_VALIDATION_SYNTHETIC`. It is never a calibration, and synthetic manifests can never satisfy
     the `REAL_CALIBRATED` gate.
7. **P2-6 = `RubberStripTaskFamily` behind `open_vector_env(backend="kinematic_grip_batched")`.**
   - **Scene:** task `rubber_strip_drag_v1`, a case-A strip on a table, dragged by a kinematic grip toward a tip goal.
   - **Randomisation:** per slot, length and thickness (rebuild); per reset, μ0, ρ, Prony g/τ scales, table friction,
     tip traction and a simulated preload phase (initial deformation).
   - **Spaces:** actions `grip_delta_v1` (A = 6); a deployable 78-dim `policy` observation and an opt-in privileged
     `critic`; reward `rubber_tip_tracking_v1` (provisional shaping); evaluator `rubber_tracking_eval_v1`.
   - **Termination:** through `rl.termination`.
   - **Code:** the only edit to a Phase 1 file is a lazy-import dispatch branch in the non-frozen `rl/vector_env.py`.
   - **Acceptance protocol:** `rubber_rl_tests_v1` (§4.8), tests RL-01…RL-11.
8. **Full Native = a new Kit host that keeps the TLMPM bridge's shape.** It runs `LiveRubberMaterial` in the Kit
   process, fed each control step by the clamp pose measured from PhysX.
   - Scene: static workcell, Frankas at the safe park.
   - Strip surface: `DynamicSurfaceBridge`.
   - Cameras: the v5 overview RGB-D.
   - Run: one 100-step lift/hold/contact episode.

   Comparison: physics against 20 headless GPU replays of the recorded poses (rule `rubber_l3_band_v4`). Inputs and
   receipts must be bitwise equal; the rendered surface may differ. `isaac_sim_runner.py` and the frozen
   `live_material.py` stay unchanged.
9. **Plan size and time.** 23 tasks, plus X3 (running) and the deferred items of §7.1.
   - Nominal: ≈ 20 agent-days with a ≈ 9-day critical path.
   - Expected at the velocity seen in P2-3 / P2-4: **≈ 3–5 calendar days**, if X3 and the reviews do not force a
     protocol v2. Budget **+0.5–1 day per v2 cycle**.
   - GPU: **≈ 11 GPU-hours**, mostly GPU0. The Native run needs ≥ 14 GiB free.

## 1. Facts this plan rests on (verified in code, records or probes)

| # | Fact | Evidence |
|---|---|---|
| F1 | A rubber batch requires a uniform dt, dx, spacing, substeps, gravity, G, Gt, box count and contact law. Geometry, material and frictions may differ per env. Strips of t 4/5/6 mm at dx 1 mm (n_t 4/5/6, rubber_ref_v1) with M = 2 Prony terms, a table plane, a tip traction region and the protocol options (F8 + kick-average + F9 + F10 + guard) pack into **one** batch | `batched/rubber_layout.py:12-14`; probe 2 (§9) |
| F2 | Step rule for μ0 ≤ 0.8 MPa, ρ ≥ 1000 kg/m³, β 20, dx 1 mm, CFL 0.5: c_p,max 130.6 m/s, dt 3.827 µs, **5,226 substeps** per 20 ms. The recipe-wide range (μ_eq ≤ 2 MPa, g∞ 0.75) would need 10,057 | probe 1 (§9) |
| F3 | Per-reset resampling is host-staged: `reset()` reads device arrays and, on a density change, rebuilds the F8 block inverses on the host. That takes **0.27 s per env** for a 65k-particle slot (3 blocks of 255 nodes), and 0.23 s for a full reset with resampling of a small slot | `batched/rubber_material_batch.py:420-426, 580-627`; probe 2 |
| F4 | Group (grip) motions are host-staged (`set_group_motions`: numpy, scipy `Rotation`, `group_q1.numpy()` read) | `batched/rubber_material_batch.py:489-505` |
| F5 | The contact kernel loops over half-spaces, then over frames only if F > 0. The per-node float64 frame slerp, the §22.7 cost driver, is absent in a plane-only scene | `rubber_contact.py:416-470` |
| F6 | `open_vector_env` on `main` (`bfe6d5e`) dispatches on `backend` only. `resolve_tasks` refuses any task that is not a `TaskFamily` or `ConfiguredTaskFamily`. `rl/vector_env.py` is in neither the `acceptance_v2` `frozen_paths` nor its `evaluator_files`. `test_perf_acceptance.py:152` asserts it stays changeable, because it is the optimizer's target | `origin/main:src/robotic_cutting_sim/rl/vector_env.py:1207`, `configs/rl_benchmark/acceptance_v2.json` |
| F7 | The cutting contract is cutting-specific: `POLICY_DIM` 18, `edge_delta_v1` actions, cut info keys. The termination bitfield is reusable | `origin/main:rl/vector_contract.py:12-40`, `rl/termination.py` |
| F8 | The rubber branches lag `main`. This branch's `rl/vector_env.py` is the pre-P1-5 version (1,045 vs 1,243 lines). `rubber_rl_v1` is still labelled "full" here; the relabel exists only on `claude/rl-rubber-v5` (`687f4fe`). That branch's ADR section is also numbered §21, which collides with the visco §21 of this lineage | `git show origin/claude/rl-rubber-v5:docs/adr/0023-rubber-material-v1.md` |
| F9 | Native hosts its material inside `isaac_sim_runner.LiveIsaacSimCutting`. It constructs `LiveCuttingMaterial` with cutting task, tool and pads (`isaac_sim_runner.py:690-712`) and advances it with the measured tool pose after PhysX (`:1534`, `:1568`). `live_material.py` is SHA-frozen. `LiveRubberMaterial` already has `reset()` / `advance(motions)` (`live_rubber_material.py:164, 175`). T5.8 ran Native via `scripts/rl_ready/launch_native_cut.py` (≥ 10 GiB + 4 GiB free, `--gpu 0`) | code; `P1_5_CUT.md` |
| F10 | Calibration manifest v1 has no data-origin field. The `REAL_CALIBRATED` gate checks only `status == FROZEN_FOR_HELDOUT` and `heldout_verdict == PASS`, so a synthetic dataset written in the v1 schema could pass it | `rubber_calibration.py:17-36`, `rubber_material.py:431-444` |
| F11 | Only scipy is installed in `.venv` and in `.deps/isaacsim-venv`: no `cma`, BO or `optuna`. No new dependency is needed | `importlib.util.find_spec` check |
| F12 | `MaterialCoupling` (Lab env) is defined over edge poses and cut digits | `rl/lab_batched_env.py:60-75` |
| F13 | Throughput references, all SHARED_GPU on GPU0: workload E v2 (n_t 4, 97,280 particles, 6,662 substeps) gives 0.87 / 1.06 / 1.03 env-steps/s at N 1 / 4 / 8. VISC-10 at n_t 4, N 8, 3,446 substeps: elastic 1.515, P2 1.278. Pads cost +40–50 % (§9A.1). Phase 1 workload A at N 8: 4.85 env-steps/s (103k particles, 200 substeps) | P2_2 v5.4, P2_3 §4, P2_4 §9A.1, P1-2 |
| F14 | The v2 contact envelope was scored at n_t 3 only (INC / INC2). At n_t ≥ 4 it is DIAGNOSTIC-only and narrower: the μ 0.8 hold slides at n_t 4, μ 0.6 sliding is +1.2–1.7 %, and μ 0.4 sliding grows from 0.04 % to 0.18 % at n_t 6 | ADR §22.9, §22.10 |

## 2. P2-4 completion

### 2.1 B3 — robot-feedback coupling adapter: deferred (not required for P2-6)

Why it is not needed for P2-6:
- **The spec's P2-6 asks for no robot feedback.** It asks for batched reset/step with material, geometry, friction,
  force and preload randomisation, a state benchmark, throughput and convergence, Native usability and provenance.
  The P2-4 robot-gripper item is covered physically by kinematic pad frames (per-pad wrench [N, G, B, 6], pinch
  -05). The 20 ms handoff assessment it requires is done (RUBBER-CONT-07: PASS as an assessment; every point
  significant).
- **B3 alone would only produce labelled rows.** C1 (≤ 1–2 ms or an iterated exchange) is undone, so every coupled
  row would carry `coupling_delay_significant`.
- **B3 alone is not a usable robot-coupled rubber env.** The Lab env's `MaterialCoupling` is defined over an edge
  pose and a cut digit (F12); a rubber Lab task (grip scene, IK on the grip, rubber action/observation) would also be
  needed.

Declared in the P2-6 receipts: `coupling: kinematic_grip_ideal_tracking (no wrench feedback)` and
`robot_feedback_coupling: NOT_IMPLEMENTED_PHASE2`. B3 + C1 + P1 form the follow-up package, which goes to the user
as a scope question (§8).

### 2.2 S1 — self-penetration detector (required before P2-6)

Specification for rcs-impl (task S1):
- **Scope.** A detector only; no self-contact force. It is opt-in per batch (`self_penetration_check="surface_grid_v1"`),
  on by default in the P2-6 env and in the Native host. With the default `off_v1` there are no arrays and no launches
  (bitwise-off).
- **Data.**
  - Surface particles: compile-time flags (`rubber_scene.py:273`), packed per env.
  - Current positions: `x64` when F10 is on, else `x`.
  - Reference positions `X`.
- **Grid.** The Phase 1 `FallbackGrid` pattern (`batched/particle_contacts.py:349-388`), re-implemented in a new
  Phase 2 module (the Phase 1 file stays untouched): keys `env·C + hash(cell)`, cell width d, a stable radix sort and
  ranges. It is graph-safe and never creates a `wp.HashGrid` (this avoids the Warp 1.17 one-descriptor hazard and the
  Phase 1 HashGrid guard). Envs cannot mix by construction.
- **Test.** For surface pairs (p, q) of the same env: |x_p − x_q| < d with d = s (particle volumes overlap), **and**
  |X_p − X_q| > 3·dx (the frozen reference exclusion). Through-thickness pairs (reference t − s ≥ 3.5 mm at n_t 4)
  would need a through-thickness stretch < 0.14, far outside the J domain. Legal bends cannot trigger it either: an
  inner-surface pair at reference 3·dx reaches s only at stretch 1/6.
- **Cadence.** Every `check_substeps` substeps (declared 256 ≈ 1 ms at dt 3.8 µs), inside the graph. A blind-interval
  counter `selfcontact_blind_steps` counts control steps with 2·v_max·Δt_check ≥ t_min − d (info only, reported as a
  rate). Estimated cost: one sort plus a 27-cell query over ≈ 20k surface particles per env, about 20 times per control
  step, i.e. ≤ 0.5 % of a control step (measured in RL-07).
- **Output.**
  - Sticky invalid bit `RUBBER_SELF_CONTACT_UNRESOLVED = 8` (single and batched; `decode_reason` extended).
  - Per-env hit count and first-hit step.
  - Strict mode raises; training mode records, and the P2-6 env maps the bit to `TERM_MATERIAL_INVALID`.
  - The masked reset clears it.
- **Tests (CPU; part of RL-05):**
  - (a) a case-C strip whose far grip is driven back through its own body flags at or before the first other invalid
    bit, and within one check interval of the geometric overlap time;
  - (b) a legal U-bend of radius 3t (prescribed) never flags;
  - (c) L0 byte-wise single vs batched N = 1 / N = 3;
  - (d) isolation: a crossing in env 1 changes no byte of envs 0 / 2;
  - (e) bitwise-off default;
  - (f) masked reset clears.
- **GPU:** a cost row in the S1 smoke (≤ 10 min).

### 2.3 P1 — contact-kernel cost: deferred

P2-6 uses only the table half-space (F = 0), so the slerp cost (22 % of kernel time in `gpu_contact_002`) does not
occur. RL-07 reports the plane-only contact cost. If it exceeds +25 % of the no-contact substep, a bounding-volume
early-out (byte-identical on CPU, tested) may be added before the P2-6 GPU runs. A frame motion table returns with
pad scenes.

### 2.4 μ envelope and the P2-6 friction range

- **Envelope file.** Task ENV1 declares `configs/rubber/contact_envelope_v1.json` (SHA-pinned) after X3's formal run.
  Rows (law, n_t) → `sliding_mu_max`, `static_mu_max`, the scored test ids, run id and protocol SHA.
- **What counts.** Only formally scored PASS rows. DIAGNOSTIC rows may narrow an entry, never widen it.

| Law | n_t 3 | n_t 4 | n_t 5–6 | Source |
|---|---|---|---|---|
| v2 `grid_node_penalty_coulomb_v2` | sliding 0.6, static 0.8 (formal) | not established (DIAGNOSTIC: sliding ≤ 0.4, static ≤ 0.6) | not established (DIAGNOSTIC: μ 0.4 sliding 0.18 % at n_t 6) | `contact_v2_001`; §22.10 |
| v3 (X3, running) | from `rubber_contact_tests_v3` (target sliding 0.8) | from v3 (target 0.8) | **ENV2** | X3, ENV2 |

- **P2-6 range.** `friction_table ~ U[0.3, min_slots μ_slide,max(law, n_t)]`. The lower bound 0.3 is the recipe's.
  The P2-6 slot set has n_t 4–6 (F1), so the n_t 5–6 rows are needed.
- **ENV2.** A declared formal envelope check (`rubber_contact_envelope_v1`, SHA-pinned before its run) for the chosen
  law, CPU only:
  - n_t 5 and 6;
  - sliding at μ 0.3 and at the target μ_max, on INC2 and on an unseen strip-on-table geometry;
  - hold at 0.95 μ_max;
  - thresholds identical to `rubber_contact_tests_v2` -01 b / a and -04.
- **Contingency.**
  - X3 fails: P2-6 uses v2 with a target of 0.4. ENV2 then also scores n_t 4.
  - ENV2 fails at a target: μ_max drops to the largest scored PASS value.
  - Nothing passes: the friction axis of P2-6 is BLOCKED, never silently widened.
- **Out of range.** μ ≥ 1.0 and any value above μ_slide,max raise `UNSUPPORTED_FRICTION_OUT_OF_ENVELOPE` in P2-6 v1.
  There is no labelled override in v1 (smallest path).

### 2.5 Other P2-4 items

| Item | Disposition |
|---|---|
| X3 (v3 law + `rubber_contact_tests_v3`, running on `claude/rl-rubber-p24`) | Dependency of ENV1 / ENV2 / P6.0. Its ADR text should be §22.11 so that this §23 keeps its number |
| -01 d stick-run books FAIL (v2) | Kept. X5 (jitter source + an a-priori drift-rate criterion) is optional. Without it, Phase 2 Done box 2 lists `contact_v2_001` -01 d as a limitation |
| -05 pinch slip ratio FAIL (v2, μ_pad 0.5) | Scored again by X3's v3 protocol. P2-6 has no pads |
| X4 (continuum stability of unilateral Coulomb sliding) | Optional analysis; not on the critical path |
| R1 independent review of §22 / `contact_v1_001` / `contact_v2_001` / B1 / B2 / B2-v2 / X1 / X2 (/ X3) | Part of K2 |
| -06 self-contact | Stays NOT_IMPLEMENTED_V1. S1 turns a crossing into a flag |

## 3. P2-5 — calibration interface and synthetic numerical validation

### 3.1 Interface (new Phase 2 modules; `rubber_calibration.py` v1 and the REAL gate code unchanged)

- **`rubber_calibration_v2.py`: schema `rcs_rubber_calibration_dataset_v2`** = v1 plus:
  - `data_origin ∈ {real_measurement, synthetic_simulator}`;
  - a per-experiment `sensor` block (noise σ, resolution, sample rate) for every type;
  - for synthetic data a `synthetic` block: generator commit, truth profile (n_t, CFL, options), sensor-model id,
    `sealed_truth_sha256`;
  - statuses `NO_REAL_DATA`, `PILOT`, `FROZEN_FOR_HELDOUT`, `SYNTHETIC_PILOT`, `SYNTHETIC_FROZEN_FOR_HELDOUT`.

  Synthetic manifests live under `configs/rubber/calibration_synthetic/`, never in `configs/rubber/calibration/` (the
  REAL gate's directory). The gate loads with the v1 loader, so a v2 synthetic manifest is refused by schema. A new
  negative test asserts that no synthetic manifest, in any directory, status or verdict, can yield `REAL_CALIBRATED`
  (F10).
- **`rubber_calibration_experiments.py` (forward models).** Each manifest experiment type maps to:
  - a `RubberStripSpec` and BC case;
  - a schedule (group motion table or traction ramp; control_dt 1 ms for the rate tests);
  - a contact spec;
  - a duration;
  - an observable extractor and a sensor transform (sampling at sensor timestamps, declared stamps as VISC v2).

  Population evaluation: (candidate θ_j, experiment k) → one env of a `BatchedRubberMaterial`, grouped by uniform dx.
  CPU L0 tests check batched evaluation against single references.
- **`rubber_calibration_fit.py`.**
  - `ParameterSpace`: θ = (μ_eq, g1, τ1, g2, τ2, μ_table). Bounds lie inside the recipe ranges and the batch step rule;
    transforms are log for moduli and τ, and a stick-breaking logit for g with Σg < 1. Density is measured (specimen
    mass) and not fitted; β is numerical (20) and not fitted.
  - `Objective`: weighted residuals (sensor σ) built from **fit-split experiments only**. It has no RL term, and it
    cannot receive a reward callable.
  - Adapters:
    - `least_squares_trf` (scipy; forward-difference Jacobian evaluated as one batch of p + 1 envs per group;
      3-start Latin hypercube), the **formal method**;
    - `cma_es` (in-repo numpy reference, seeded; reported, not formal);
    - `ask_tell` (`ask() → θ[λ, p]`, `tell(loss[λ])`): the hook through which the external agentic optimizer or a BO
      method plugs in. No BO library is added.
  - `freeze_fit(result)` writes a fit receipt (θ̂, covariance, sha256, commit).
  - `evaluate_heldout(manifest, fit_receipt)` refuses unless the manifest is `*FROZEN_FOR_HELDOUT` and the fit
    receipt names only fit-split experiments. It is **one-shot** per manifest freeze id (`HELDOUT_ALREADY_EVALUATED`;
    an audit re-run must be named and labelled).
  - Any path under a `sealed/` directory is refused as input.
  - Outputs carry `calibration_status: SIMULATOR_ONLY_UNCALIBRATED` and `validation_kind:
    NUMERICAL_VALIDATION_SYNTHETIC`. No recipe is promoted.

### 3.2 Synthetic truth (inverse-crime mitigation and sealing)

- **Truth = the same model family** (`neo_hookean_jbar_prony_v1`, M = 2, β 20, contact law of ENV1). It differs from
  the fit model in three ways:
  - resolution (**n_t 6, CFL 0.25** vs the fit model's rubber_ref_v1 n_t 4, CFL 0.5);
  - sensor sampling (force 1 kHz, video markers 240 fps, at sensor timestamps);
  - seeded sensor noise: load cell 0.3 % FS + 1 mN resolution, displacement 1 µm, markers 0.1 mm, specimen
    dimensions ±0.02 mm, mass ±0.5 mg.
- **Hidden θ\*.** Drawn from a declared prior box with a secret seed (`os.urandom`) and written to
  `artifacts/rl_ready/p2_5/<run>/sealed/truth.json`. Only its SHA-256 enters the manifest and its freeze commit. It is
  revealed only after the held-out evaluation.
- **What the result can and cannot show.** Same-family truth cannot reveal model-form error against real rubber. It
  tests identifiability, the pipeline and the generalisation of the declared protocol under discretisation and noise.
  The report says exactly this.

### 3.3 Experiment matrix (fit / held-out; specimens on the lattice, dx uniform per batch)

| Id | Type | Specimen | Loading | Split |
|---|---|---|---|---|
| T1 / T2 | tensile load–unload (case C) | 60 × 10 × 4 mm | λ 1 → 1.5 → 1, ε̇ 1 /s and 8 /s | fit |
| R1 | stress relaxation | 60 × 10 × 4 mm | ramp to λ 1.3 in 20 ms, hold 0.3 s | fit |
| C1 | cantilever rebound (case A, tip traction released) | 80 × 16 × 4 mm | static tip deflection δ₁ (initializer), instant release, 0.4 s of "video" | fit |
| G1 | gravity sag (static) | 80 × 16 × 4 mm | initializer | fit |
| F1 | friction sliding (block pushed by a fixture frame on the table) | 30 × 16 × 4 mm | 50 mm/s, weight only | fit |
| T3 / T4 | tensile | 60 × 10 × 4 mm | ε̇ 3 /s (interpolation), 20 /s (extrapolation) | **held-out rate** |
| T5 | tensile | 60 × 10 × **3** mm | ε̇ 3 /s | **held-out thickness** |
| C2 | cantilever rebound | **120** × 16 × 4 mm | δ₂ | **held-out length** |
| C3 | cantilever rebound | 80 × 16 × **3** mm | δ₁ | **held-out thickness** |
| C4 | cantilever rebound | 80 × 16 × 4 mm | **2 δ₁** | **held-out initial deformation** |
| F2 | friction sliding | 30 × 16 × 4 mm | 20 mm/s, + top traction = 1 × weight | **held-out load / speed** |

Simulated time: ≤ 1 s per experiment; ≈ 2.3 s for the fit split, ≈ 1.8 s held out.

### 3.4 Metrics (spec P2-5 list)

| Metric | Applies to | Definition |
|---|---|---|
| M1 force–displacement | T, R | RMSE of F over the record / max \|F_obs\| |
| M2 hysteresis energy | T | relative error of the loop area ∮F dU |
| M3 relaxation | R | relative error of the force at the end of the hold |
| M4 end (tip) trajectory | C | RMSE of the tip position / release amplitude, 0–0.4 s |
| M5 rebound time | C | first crossing of the static equilibrium after release (absolute error, ms) |
| M6 vibration decay | C | log decrement of the first 3 tip peaks (absolute error) |
| M7 slip distance | F | relative error of the slip distance (and onset time) under the declared push |
| M8 volume | all (synthetic only) | \|mean J̄_pred − mean J̄_truth\| over the record |
| M9 energy | all (synthetic only) | relative error of the total dissipated energy (W_ext − ΔE) from the books |

### 3.5 Protocol `rubber_calib_tests_v1` (declared by P5.4 from this section, SHA-pinned before any pilot or formal run)

| Id | Test | Pass |
|---|---|---|
| CAL-01 | Interface and refusals (CPU): v2 manifest validation (origin, sensor, sealed SHA), split disjointness, path guard, one-shot held-out guard, REAL gate refuses every synthetic manifest, θ bounds vs the step rule | every unit test passes |
| CAL-02 | Forward-model parity (CPU): population batch = single references byte-wise (L0) for every experiment type; repeated objective calls byte-identical on CPU | byte-wise |
| CAL-03 | Truth generation receipts: truth profile, noise seed hash, sealed SHA in the manifest, manifest + data frozen (sha256) | complete; ordering per CAL-08 |
| CAL-04 | Fit quality on the fit split (formal method TRF, 3 starts) | reduced χ² ≤ 1.5 × χ²_pilot(θ_pilot); converged (‖Δθ‖ < 1e-4 relative, or the declared budget of 40 iterations) |
| CAL-05 | **Held-out prediction** (primary) | every (experiment, metric) ≤ its frozen tolerance tol_mk |
| CAL-06 | Parameter recovery (after reveal) | \|θ̂_i − θ\*_i\| ≤ max(3 σ̂_i, 2 % θ\*_i) for each parameter with σ̂_i / θ̂_i < 0.5; the others are reported as non-identifiable |
| CAL-07 | Negative control (power): a fit on T2 + C1 only (one rate, one geometry) | must **FAIL** CAL-05 on at least one held-out viscous metric (M1, M2, M5 or M6). If it passes, the protocol lacks power and CAL-07 FAILs |
| CAL-08 | Order and sealing: protocol commit < pilot < thresholds commit < formal dataset + manifest freeze < fit receipt commit < one-shot held-out < reveal | verified from git and receipts; any violation makes the run INVALID |
| CAL-09 | Labels | every output carries `SIMULATOR_ONLY_UNCALIBRATED` and `NUMERICAL_VALIDATION_SYNTHETIC`; no recipe promoted |

**Tolerance rule (declared now; numbers computed by the pilot).** A pilot dataset uses its own secret seed, and its θ
is revealed after the pilot, which is used only for the tolerances. For each (experiment k, metric m):
- a_mk, the noise floor: the metric between two independent noise realisations of the same truth;
- b_mk, the discretisation floor: the metric between the noise-free truth (n_t 6, CFL 0.25) and the fit model at
  θ_pilot (n_t 4, CFL 0.5);
- **tol_mk = max(2 · (a_mk + b_mk), floor_m)** with floors M1 0.5 %, M2 1 %, M3 0.5 %, M4 1 %, M5 2 ms, M6 0.005,
  M7 2 %, M8 1e-3, M9 2 %.

The thresholds go to `configs/rubber/rubber_calib_thresholds_v1.json` (SHA-pinned) **before** the formal dataset is
generated. The χ²_pilot reference comes from the same pilot.

**Real data (declared, NOT_RUN).** When real data exist:
- the same pipeline runs with `data_origin: real_measurement` under a new `rubber_calib_tests_real_v1`;
- tolerances come from the measurement error plus a pilot on fit data only;
- the manifest is `FROZEN_FOR_HELDOUT` before any held-out evaluation;
- only a held-out PASS can make a recipe `REAL_CALIBRATED`.

The ADR §22.10 note that calibrated sets must include high-frequency loss becomes a requirement of that real protocol:
an additional Prony term with a declared short τ, fitted or reported as non-identifiable.

## 4. P2-6 — RL-ready rubber acceptance

### 4.1 Task family and scene `rubber_strip_drag_v1`

- **Scene.** `rubber_strip_v1` case A (grip group 0 = jaw nodes x ≤ 0, Dirichlet, F8 + kick-average). The strip lies
  on a table half-space z = 0 (strip-local = world for the identity strip pose; the receipt states it). Gravity is −z.
  A tip traction region covers x ∈ [L − 2dx, L]. There is no pad or fixture frame (F = 0).
- **Slot set (geometry per slot; seeded selection at build / `rebuild()`):**
  - L ∈ {100, 125, 150} mm, W 16 mm, t ∈ {4, 5, 6} mm, dx 1 mm;
  - so n_t 4 / 5 / 6, `rubber_ref_v1` (full);
  - 52,224–116,736 particles (probe 1).

  A partial reset with a geometry change raises `UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET`, as in Phase 1 U-P15-3.
- **Material.**
  - Recipe `rubber_generic_visco_v1` (M = 2, illustrative values).
  - Per-reset ranges: μ0 ∈ [0.3, 0.8] MPa (instantaneous), ρ ∈ [1000, 1300] kg/m³, `prony_g_scale` ∈ [0.5, 1.5],
    `prony_tau_scale` ∈ [0.5, 2.0], `friction_table` per §2.4.
  - Batch step rule from the range maxima: 5,226 substeps (F2).
- **External force.** A constant tip traction per episode: horizontal magnitude U[0, 0.5] × the slot's weight in a
  uniform random direction, plus vertical U[−0.25, 0.25] × weight.
- **Initial deformation.** A **simulated preload phase** (ADR D10 option 1): the first `preload_steps` (10) of every
  episode follow a seeded smootherstep grip path (lateral sweep A_y ∈ [−30, 30] mm, yaw ∈ [−0.4, 0.4] rad). During it,
  actions are ignored (`info["drive"] = PRELOAD`), the reward is 0 and the evaluator window excludes it. A per-slot
  snapshot reset is not used: the registry has no `snapshot` initial value yet, and it would cost ≈ 70 MB per snapshot.
- **Goal.** A tip target on the table in the slot's reachable annulus, with an initial tip distance ≥ 30 mm (seeded
  per reset).

### 4.2 API and dispatch

```python
with open_vector_env(num_envs=8, task=RubberStripTaskFamily(slots=..., config="rubber_strip_drag_v1"),
                     backend="kinematic_grip_batched", device="cuda") as env:   # runtime rl_gpu_headless
    obs, info = env.reset(seeds=seeds)                       # obs["policy"]: float32 [N, 78] on the device
    obs, reward, terminated, truncated, info = env.step(a)   # a: float32 [N, 6]
    obs, info = env.reset(env_ids=ids, seeds=s)              # masked partial reset with resampling
    results = env.evaluate(env_ids)                          # rubber_tracking_eval_v1 (host)
```

- **New modules (Phase 2 files):**
  - `rl/rubber_family.py`: `RubberStripTaskFamily`, the slot set, ranges validated against the recipe, the envelope
    and the step rule.
  - `rl/rubber_env.py`: `RubberVectorEnv`, `ENV_VERSION = "rl_rubber_vector_env_v1"`.
  - `rl/rubber_contract.py`: `rubber_vector_contract_v1`.
  - `configs/rubber/rl/rubber_strip_drag_v1.json`.
- **Dispatch.** The only edit to a Phase 1 file is a lazy-import branch at the top of `open_vector_env` in
  `rl/vector_env.py`, which is non-frozen (F6). `backend == "kinematic_grip_batched"` goes to `RubberVectorEnv`. A
  `RubberStripTaskFamily` with another backend raises `UNSUPPORTED_BACKEND`; the rubber backend with a cutting task
  raises `UNSUPPORTED_TASK`. Cutting defaults are mapped explicitly (`action_spec`, `material`, `reward` defaults →
  rubber defaults); any explicit cutting value is refused. The default-backend code path stays byte-for-byte as on
  `main`. The edit is made last, on the post-consolidation tree, to keep the merge with P1-6 optimizer candidates
  trivial.
- **Device-only step.** New opt-in batched methods `set_group_motions_device(q1, b1)` (hemisphere-aligned in a
  kernel) and `set_traction_device(f)`, byte-identical to the host path on CPU (test). The step reads nothing back
  except the strict-mode validity count (as Phase 1). Resets stay host-staged (F3) and their p95 is reported. If the
  measured reset p95 exceeds 1 s, an optional vectorised F8 block assembly can replace the Python loop; it must keep
  the same accumulation order (bit-identical by construction).
- **Other API rules.**
  - Cameras / vision: refused (`UNSUPPORTED_SENSOR`; state profiles only).
  - `autoreset`: `False` only in v1. `"same_step"` raises `UNSUPPORTED_OPTION`.
  - Stream policy: as ADR 0021 §7 (Torch current stream).

### 4.3 Action, observation, reward, done, evaluator

- **Action `grip_delta_v1` (A = 6).** A normalised delta of the commanded grip pose: ≤ 5 mm translation and ≤ 0.05 rad
  axis-angle per 20 ms. The workspace clamp is declared in the task config; it keeps the jaw's lowest node ≥ 1 mm above
  the table, because Dirichlet nodes skip contact. An out-of-box or clamped action is held, with `action_status` and
  `action_clamped` reported; it never terminates the episode. Grip motion is ideal tracking: END pose = commanded
  pose, interpolated by the motion table.
- **`policy` observation (deployable, 78 dims, float32).**
  - Grip pose (7, xyz + quaternion XYZW).
  - Commanded grip twist (6).
  - Time remaining (1).
  - Previous action (6).
  - 9 marker positions (27) and velocities (27), labelled `ideal_marker_tracking`.
  - Goal (3).
  - Preload flag (1).

  It contains no force, wrench, stress, J, energy, contact or material parameter, and the contract checks the names.
- **`critic` (opt-in `privileged_state`).** μ0, ρ, μ_table, Prony g/τ scales, tip traction, grip wrench, plane wrench,
  Ψ_eq, KE, E_grav, J̄ min / max, CFL_eff max, contact count / maximum penetration.
- **Reward `rubber_tip_tracking_v1` (`PROVISIONAL_SHAPING_NOT_SUCCESS`):** −‖x_tip − x_goal‖ / L − 0.01‖a‖² − 1.0 ×
  [invalid]; 0 during preload.
- **Done.**
  - `terminated` = `rl.termination` `TERMINAL_BITS`. Every rubber material bit maps to `TERM_MATERIAL_INVALID`:
    NONFINITE 1, J_DOMAIN 2, CFL 4, SELF_CONTACT_UNRESOLVED 8. The detail is in `info["invalid_reason"]`, listed as
    privileged.
  - `truncated` at `max_episode_steps` (default 250 including preload).
  - Infrastructure failures follow Phase 1 (`TERM_INFRASTRUCTURE`, `VectorEnvInfrastructureError`).
- **Success** comes only from the host evaluator `rubber_tracking_eval_v1`: tip within 10 mm of the goal for the last
  25 steps of the episode.

### 4.4 Solver profile and labels

- **Profile.** `rubber_rl_env_v1` = F8 `consistent_mass_projection_v1` + `kick_average_v1` + F9 + F10 + guard 1.0 + S1
  + the ENV1 contact law; `visco_energy off_v1`; CFL 0.5; r = 1. The Native host uses exactly this profile.
- **Receipts** carry the recipe SHA, slot geometry, n_t, β, dt, substeps, M, contact law and envelope row, and these
  labels:
  - `calibration_status: SIMULATOR_ONLY_UNCALIBRATED`;
  - `coupling: kinematic_grip_ideal_tracking`;
  - `perception: ideal_marker_tracking`;
  - `self_contact: NOT_IMPLEMENTED_V1 (detector surface_grid_v1)`;
  - the `rubber_ref_v1` measured bias (§5).
- **Reduced-fidelity variant.** One explicitly labelled variant may be offered for throughput: `rubber_rl_v1` (n_t 3,
  dx 4/3 mm, t 4 mm only, `reduced_fidelity: thickness_resolution_n_t3` with its v5 measured bias). It appears in its
  own RL-07 rows only.

### 4.5 Throughput expectation (estimate, to be measured; not a claim)

At N = 8 with a mean of ≈ 81k particles/env (the 9-slot set) and 5,226 substeps, an env-step needs ≈ 4.2e8
particle-substeps. At the
measured 3.3–4.3e8 particle-substeps/s (F13, M = 2 with the options), that is **≈ 0.8–1 env-steps/s in aggregate**,
i.e. ≈ 9 s per batch step. The `rubber_rl_v1` variant would be ≈ 3× faster. This is ≈ 5× below Phase 1 workload A
(4.85 env-steps/s at N 8), almost entirely because rubber needs 26× more substeps per control step (5,226 vs 200).
Per particle-substep, rubber is cheaper (≈ 4e8/s vs ≈ 1e8/s for v7_2 TRIM). RL-07 reports these quantities, never a
single speedup.

### 4.6 Native path (FULL_NATIVE_REFERENCE usage of the same recipe)

- **New module** `native_rubber_host.py`, with Kit imports inside functions. It reuses, unchanged:
  - `isaac_sim_runner.launch_app` (Kit 6.2, `--gpu` sets render, physics and material device);
  - `workcell.export` (variant D0, static workcell; Frankas held at the shared safe park, no robot control, camera
    rig v5 untouched);
  - `usd_surface_bridge.DynamicSurfaceBridge`.

  `isaac_sim_runner.py` and `live_material.py` stay unchanged.
- **Bridge (the TLMPM loop's shape).**
  1. Set the kinematic clamp target.
  2. Step PhysX one control step.
  3. Read the clamp's measured world pose from PhysX.
  4. `LiveRubberMaterial.advance(motions={0: measured pose})` on CUDA in the Kit process, with the same recipe, slot,
     profile and contact law as P2-6.
  5. Update the surface mesh from the surface particles.
  6. Capture the overview RGB-D every 10 steps.
  7. Record `rcs_rubber_record_v1`: markers, wrenches, energies, J̄, CFL_eff, invalid bits, and the measured clamp
     pose in float64.

  Feedback is off: the clamp is kinematic, and its wrench is reported, not applied.
- **Run `native_rubber_001`** (scripts `scripts/rubber/p2_6/run_native_rubber.py` and a launcher with the T5.8 memory
  rule).
  - Setup: nominal slot L 125 × W 16 × t 4 mm, nominal material, μ_table at the midpoint of the P2-6 range.
  - Steps 0–24: hold (gravity settle on the table).
  - Steps 25–49: lift 20 mm + yaw 0.3 rad (smootherstep).
  - Steps 50–99: hold (free decay, tip contact).
- **Headless reference.** `replay_native_rubber.py` feeds the recorded measured clamp poses to `LiveRubberMaterial`
  without Kit on the same GPU model: 20 repeats for the band and 10 held-out repeats, which are reported.
- **Comparison.** `compare_native_rubber.py` (CPU) applies RL-09 (§4.8). The rendered surface and the images are not
  physics: only bridge integrity is scored on them.

### 4.7 Why this is the smallest honest path

- **No change to Phase 1 runtimes.** The only Phase 1 file touched is the dispatch in a non-frozen file.
- **No new physics in P2-6.** S1 is a check.
- **One task family covers every randomisation axis the spec lists.**
- **Native verification exercises the measured-pose bridge with contact and dynamics** in one short episode.
- **No robot coupling.** It is deferred with an explicit label.

### 4.8 Protocol `rubber_rl_tests_v1` (declared by P6.0 from this section, SHA-pinned with the task config before any implementation run)

Common:
- Profile `rubber_rl_env_v1`.
- Task config `rubber_strip_drag_v1`.
- Seeds declared.
- GPU rows on RTX 5090 under `rcs_rl_runner/gpu<N>.lock` with ≥ 4 GB left free, labelled SHARED_GPU / EXCLUSIVE_GPU.
- CPU tests run twice (`CUDA_VISIBLE_DEVICES=''` and `=0`). Torch tests run in the Torch venv (D-09).

| Id | Test | Pass |
|---|---|---|
| RL-01 | Contract `rubber_vector_contract_v1` at N = 1 and N = 8 (GPU) and N = 1 / 3 (CPU parity mode): obs keys / shapes / dtypes / device, no privileged names in `policy`, critic opt-in, action space, info keys and dtypes, `termination_reason` ↔ `terminated` / `truncated`, partial-reset bookkeeping, receipt keys (incl. calibration status, fidelity, coupling, perception, envelope row), evaluation keys | every clause holds |
| RL-02 | Parity: (a) CPU L0 byte-wise, env-level batch (N 1, N 3 with heterogeneous slots, materials, frictions, tractions and preloads) vs single `LiveRubberMaterial` references driven by the same grip poses, every control step, every state array and observation source; (b) GPU L2 every step (finite, mass constant between resets, audits 0, invalid bits 0 under the declared scripted actions); (c) GPU L3 `rubber_l3_band_v4` for 2 slots of an N = 8 batch vs 20 single GPU repeats, 50 steps, α_family 0.05; 10 held-out singles reported | (a) byte-wise; (b) every step; (c) M ≤ U for every statistic |
| RL-03 | Randomisation: identical seeds give byte-identical parameters and states; every value in its declared range, in the step rule and in the envelope; resampling env e changes only e; 2,000 host samples per parameter, KS p ≥ 0.001 vs the declared distribution; seeded slot selection; geometry change on partial reset refused | all hold |
| RL-04 | Isolation: CPU L1 byte-wise (perturb env 1's actions, traction, material and preload; envs 0 / 2 unchanged; masked reset = fresh env incl. visco, contact, S1 and preload state); GPU env-level `rubber_contact_tests_v2` -09 rule (k 5 A / 5 B interleaved, λ 4, ε_rel 1e-6) at N = 4 with env 1 perturbed | byte-wise; rule satisfied |
| RL-05 | Episode semantics: every rubber invalid bit forced on CPU maps to `TERM_MATERIAL_INVALID` with its detail; truncation at the horizon; reward = host float64 recomputation every step; evaluator unit tests; S1 tests (a)–(f) of §2.2; GPU N = 8 sanity: scripted policy mean return > zero-action mean return in ≥ 7 / 8 envs and ≥ 1 evaluator success | all hold |
| RL-06 | Long rollout, N = 8 GPU: ≥ 400 batch steps, 60-step episodes (10 preload), staggered partial resets (≥ 40), random + scripted actions | finite; no infrastructure error; VRAM peak growth after warm-up ≤ 64 MiB; reset p95 and invalid-termination rate reported |
| RL-07 | Throughput / VRAM: N 1 / 4 / 8 / 16 for the full slot set and a uniform n_t 4 slot set; labelled `rubber_rl_v1` rows; Phase 1 workload A N 1 / 8 re-measured in the same GPU session with the frozen harness (run, not modified); D11 columns + `N \| particles/env \| control Hz \| material dt \| cameras \| env-steps/s \| aggregate sim-time/wall \| VRAM peak \| reset p95 \| physics verdict`, substeps, n_t, β, M, contact law, S1 cost, plane-only contact cost | scored: N = 8 env-steps/s > N = 1 (measured gain, M2) and VRAM ≤ 4 GB at N = 8; the rest reported |
| RL-08 | dt / spacing on the RL scene (single env, GPU, nominal slot, declared scripted lift-hold-drag): O1 tip position after the hold, O2 peak grip force, O3 tip slip on the table | dt (CFL 0.5 → 0.25 at n_t 4): \|ΔO1\| / L ≤ 1e-3 and \|ΔO3\| / O3 ≤ 1 % scored; spacing (t 4 mm, n_t 4 / 6 / 8) reported with the -11 rule block byte-identical (GCI, observed order, Phase 1 comparison text). The material's scored spacing test remains RUBBER-PHYS-11 |
| RL-09 | Native: (a) receipts identical to the headless reference except device / timing; (b) material inputs = recorded measured clamp poses, bitwise float64; (c) physics: every statistic (markers 27, clamp wrench 6, plane wrench 6, Ψ_eq, KE, E_grav, contact work, J̄ min / max, CFL_eff max) inside `rubber_l3_band_v4` from the 20 replays, α_family 0.05; (d) invalid bits 0; (e) bridge integrity: mesh vertices = surface particles under the strip pose (≤ 1e-6 m); camera consistency (marker reprojection vs rendered depth) reported, not scored | (a)–(e) hold |
| RL-10 | Provenance: protocol and task config SHA-pinned before any run; every run meta carries commit, clean tree, protocol SHA, SHA-256 of each rubber module, Warp / Python / driver, GPU name / index, sharing samples and the command; evidence writers are read-only and never overwrite | complete |
| RL-11 | Phase 1 non-regression on the final head: Phase 1 CPU and Torch suites without new failures vs `main`; frozen-source pins pass; `git diff main -- configs/rl_benchmark scripts/rl_ready` empty; `open_vector_env` default-backend workload A contract GPU N = 2 PASS; `acceptance_v2` judge self-test (CPU) PASS | all hold |

## 5. Q1w — RUBBER-PHYS-11 v5 FAIL: recommendation

**Data** (`refinement_v5_001`; P2_2 v5.3 on `claude/rl-rubber-v5`):
- Q1w = 4.79708e-4 / 4.77789e-4 / 4.77343e-4 m at n_t 4 / 6 / 8, monotone, p_obs 3.12.
- The declared rule sets Fs 3 (p_obs outside [1, 3]), which gives GCI(n_t 4) **2.16 %** > 2 %.
- n_t 4 lies **+0.62 %** from the extrapolation and +0.50 % from n_t 8.
- dt and validity PASS.
- Q1 1.48 %, Q2 0.56 %, Q4 0.07 %: PASS.

| Option | Assessment |
|---|---|
| (a) n_t 6 for wide bending-dominated scenes | Cost ≈ (6/4)⁴ ≈ 5× per env-step for every wide-strip scene: P2-6 (workload E class, W/t ≥ 3), Native and calibration. Its own GCI needs an n_t 12 wide-strip run (≈ 1.1M particles, single env; the -11 declaration pins the CPU, where n_t 8 alone took most of 10.7 h). It buys ≈ 0.4 % of bias |
| (b) a v6 rule | Any v6 that turns this FAIL into a PASS on existing data (Fs 1.25 above p_obs 3, or a least-squares GCI chosen now) is a criterion change after seeing the result, i.e. evaluator loosening. **Rejected** |
| **(c) accept n_t 4 with the documented bias (recommended)** | The FAIL stays the v5 record. `rubber_ref_v1` keeps its label and gains a `measured_bias` block (Q1 +0.39 %, **Q1w +0.62 %**, Q2 −0.39 %, Q4 −0.05 %) plus `known_limitation: rubber_phys_11_v5_q1w_gci_2.16pct`, printed by every recipe and resolution receipt (a label change in `rubber_material.py`, tested). Reasons: the convergence is monotone with a dt-independent observed order ≈ 3; the bias is ≈ 4× below the declared β = 20 model-form error (−2.7 % stress at λ 1.5) and below the P2-5 tolerance floors; it is signed and stable, so a later calibration can de-embed it. Phase 2 Done box 2 then reads PASS_WITH_LIMITATIONS for this item |

If the reviewer rejects (c), option (a) becomes task Q1W-A: declare `rubber_phys_tests_v6` -11 with the triplet
(6, 8, 12) and a GPU allowance, before any run. Its cost: ≈ 1 GPU-day of queue for the n_t 12 static sag, plus ≈ 5×
on every P2-6 and Native row.

## 6. Phase 2 Done — checklist mapping

| Spec box | Evidence now | Remaining work | Expected at close |
|---|---|---|---|
| 1. Rubber recipe + hyperelastic / viscoelastic / contact models with reproducible run evidence | Recipe and manifest (P2-0, review PASS); `rubber_phys` v1–v5 (`p2_1_*`, `p2_2_rubber_001`, `p2_2_v5_001`); `visc_v1_001` / `visc_v2_001` (`p2_3_visco_001/002`); `contact_v1_001` / `contact_v2_001`, `gpu_contact_002`, CONT-08, `gpu_cont09_v2_001` (`p2_4_contact_*`) | X3 formal run; K1 consolidation; K2 combined review (P2-3 v2, v5, P2-4); merge into integration | PASS |
| 2. Large deformation, gravity, stretch / release, rebound, hysteresis and friction pass at the physical and numerical level | -05, -02 / -03, -04, -08 (v5) PASS; -07 (v4) PASS; VISC-01..09 (v1) and -03 / -05 (v2) PASS; CONT-01 a / c, -02, -03 (v2) PASS; -11 v5 Q1 / Q2 / Q4 PASS. **Preserved FAILs:** -11 v5 Q1w; CONT-01 b and -04 at μ 0.8, -01 d, -05 (v2) | X3 (v3: sliding μ ≤ 0.8 at n_t 3 / 4, pinch); ENV2 (n_t 5 / 6); Q1w decision (§5); optional X5 | PASS_WITH_LIMITATIONS (Q1w bias labelled; μ ≥ 1.0 `contact_coulomb_ill_posed`; -01 d unless X5; self-contact NOT_IMPLEMENTED_V1 with the S1 detector) |
| 3. Same rubber material in RCS multi-env RL headless and in Full Native; ≥ 8-env rubber batch with reliable results | Batched L0 / L1 (CPU), L2 at N 1 / 4 / 8 (GPU), L3 v4 PASS at N 4, CONT-08 / -09 v2 PASS. No RL env, no Native rubber | S1; P6.0–P6.7 (`rubber_rl_tests_v1` RL-01..11 incl. N = 8 / 16 and RL-09 Native) | PASS if `rubber_rl_tests_v1` passes |
| 4. REAL_CALIBRATED only after a held-out pass with real data; otherwise SIM_ONLY | `SIMULATOR_ONLY_UNCALIBRATED` in every recipe, record and evidence file; `none_v1` = `NO_REAL_DATA`; gate refuses without a frozen held-out manifest | P5.1–P5.7: interface, gate hardening (F10), synthetic numerical validation `rubber_calib_tests_v1` | PASS (status stays SIM_ONLY; the real held-out path is NOT_RUN for lack of data, stated) |
| 5. Old CUT / TRIM / V5 / V7 baselines unbroken | Frozen-source pins pass in every rubber suite; R2.5 review: Phase 1 files untouched (77 files, all added) | RL-11 on the Phase 2 PR head (suites, pins, `git diff main` of `configs/rl_benchmark` and `scripts/rl_ready` empty, workload A contract after the dispatch edit, judge self-test) | PASS |
| 6. Models, tests, scenes and docs merged into RCS; external optimizer pin updated | P2-0..P2-3 merged into `claude/rl-ready-rubber-physics` (`232e65b`, `86c5681`); nothing on `main` | K1, Z1 acceptance review, Z2 PR to `main` (CI green) + merge, Z3 new versioned descriptor pinned at the merge SHA in the external repository + smoke (incl. the rubber contract N = 2) → SYNCED receipt | PASS after Z3 |

## 7. Task list (rcs-impl unless stated)

### 7.1 Ordered tasks

Rules for every task:
- Branches: one per track from the consolidated branch; merge back through review.
- No protocol run before its declaration and code are committed clean.
- CPU tests run twice.
- GPU under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock`, explicit device, ≥ 4 GB left free, new
  `artifacts/rl_ready/p2_{5,6}/…` directories, evidence JSON and report section per task.
- Naming test before every commit.

| Id | Task | Depends on | GPU | Size |
|---|---|---|---|---|
| X3 | (running, other agent) v3 law + `rubber_contact_tests_v3` formal run; ADR §22.11 | – | small | (L) |
| K1 | Consolidation: merge `claude/rl-rubber-p23` (R1 fix `92e97aa`), `claude/rl-rubber-v5` (`d814567`; ADR v5 §21 renumbered **§19A** with internal references and the v5 report / evidence pointers noted), `claude/rl-rubber-p24` (X1 / X2, later X3), this branch, then `claude/rl-ready-rubber-physics` (`main` with P1-5 `vector_env`) → `claude/rl-rubber-p2c`; full CPU + Torch suites; frozen pins | – (X3 merged when ready) | – | M |
| K2 | rcs-review (xhigh): combined review of P2-3 v2, v5 (incl. the Q1w decision of §5), P2-4 v1 / v2 / B1 / B2 / B2-v2 / X1 / X2 / X3 and this plan; then merge into integration | K1, X3 | – | (M) |
| S1 | Self-penetration detector (§2.2), single + batched, tests (a)–(f) | K1 | smoke 0.1 h | M |
| ENV1 | `configs/rubber/contact_envelope_v1.json` from formal records (§2.4) + loader + test | X3 | – | S |
| ENV2 | Declare `rubber_contact_envelope_v1` (n_t 5 / 6, and n_t 4 if v2), formal CPU run, evidence | ENV1 | – | S–M |
| P5.1 | `rubber_calibration_v2.py` manifest v2 + REAL-gate negative tests (CAL-01 part) | K1 | – | S |
| P5.2 | `rubber_calibration_experiments.py`: forward models for T / R / C / G / F, sensor transforms, population batching; CAL-02 | P5.1 | – | M–L |
| P5.3 | `rubber_calibration_fit.py`: parameter space, objective, TRF / CMA-ES / ask-tell, freeze, one-shot held-out, path guard; CAL-01 | P5.1 | – | M |
| P5.4 | Declare `rubber_calib_tests_v1.json` (§3.5) alone, SHA-pinned; pilot dataset + pilot run; commit `rubber_calib_thresholds_v1.json` | P5.2, P5.3, ENV1 | 1.5 h | S (+run) |
| P5.5 | Formal synthetic dataset (secret seed, sealed truth, manifest v2 frozen) | P5.4 | 1 h | S |
| P5.6 | Formal fit → fit receipt → one-shot held-out → reveal → recovery → negative control; evidence `p2_5_calib_synthetic_001.json`; report `P2_5_RUBBER_CALIBRATION.md`; ADR results | P5.5 | 2.5 h | M |
| P5.7 | rcs-review (xhigh) of P2-5 | P5.6 | – | (S) |
| P6.0 | Declare `rubber_rl_tests_v1.json` (§4.8) + `configs/rubber/rl/rubber_strip_drag_v1.json` (friction range from ENV1 / ENV2), SHA-pinned, alone | ENV2 (friction bound), K1 | – | S |
| P6.1 | `rl/rubber_family.py`, `rl/rubber_env.py`, `rl/rubber_contract.py`; device-side grip / traction staging; preload drive; evaluator; CPU RL-01 / -02a / -03 / -04 L1 / -05 | S1 (code may start on a provisional friction bound); P6.0 before any protocol run | – | L |
| P6.2 | `open_vector_env` dispatch branch in `rl/vector_env.py` (§4.2) + dispatch / refusal tests; Phase 1 suites unchanged | P6.1 | – | S |
| P6.3 | GPU runs RL-01, -02 b / c, -03, -04 GPU, -05 sanity, -06, -07, -08; evidence `p2_6_rubber_rl_001.json` | P6.2 | ≈ 4 h | M |
| P6.4 | Native host `native_rubber_host.py` + run / replay / compare scripts + CPU tests of the compare logic and the bridge (no Kit) | P6.0 (profile fixed), S1 | – | L |
| P6.5 | Native run `native_rubber_001` (GPU0 ≥ 14 GiB free) + 30 headless replays + RL-09 | P6.4 | ≈ 1.5 h | S |
| P6.6 | RL-10 / RL-11 on the final head; report `P2_6_RUBBER_RL.md`; ADR results section; frozen-source pins for the accepted Phase 2 modules | P6.3, P6.5 | 0.2 h (RL-11 workload A contract) | S |
| P6.7 | rcs-review (xhigh) of P2-6 | P6.6 | – | (S) |
| Z1 | rcs-review (xhigh): Phase 2 acceptance against §6 | K2, P5.7, P6.7 | – | (M) |
| Z2 | Phase 2 PR to `main` (D-12), CI green, merge | Z1 | – | S |
| Z3 | External repository (orchestrator / its agent): new versioned descriptor pinned at the Phase 2 merge SHA, smoke incl. `rubber_vector_contract_v1` N = 2 on GPU1, sync receipt → SYNCED | Z2 | 0.3 h | S |
| deferred | B3 + C1 + P1 (robot-coupled rubber), X4, X5, Q1W-A (only if K2 rejects §5 (c)), `rubber_calib_tests_real_v1` (when real data exist) | – | – | – |

### 7.2 Parallel tracks and critical path

```
X3 ─┬─ ENV1 ─ ENV2 ─ P6.0 ─ P6.1 ─ P6.2 ─ P6.3 ─┐
    │                     └── P6.4 ─── P6.5 ─────┼─ P6.6 ─ P6.7 ─┐
K1 ─┼─ S1 ───────────────────────────────────────┘               ├─ Z1 ─ Z2 ─ Z3
    ├─ P5.1 ─ P5.2/P5.3 ─ P5.4 ─ P5.5 ─ P5.6 ─ P5.7 ─────────────┤
    └─ K2 (after X3) ────────────────────────────────────────────┘
```

Three implementation agents can work at once after K1 (P2-5 track, P2-6 env track, Native track). S1, P5.1–P5.3 and
P6.4 are CPU-only and can start right after K1, before X3 lands. P6.0 waits for the friction bound (ENV2), but the
env code (P6.1) may be written against a provisional bound; only runs wait for the declaration.

### 7.3 Time and GPU estimate

| | Nominal (ADR scale) | Expected (observed velocity) |
|---|---|---|
| rcs-impl work | ≈ 17 agent-days | ≈ 45–70 agent-hours |
| reviews (K2, P5.7, P6.7, Z1) | ≈ 3 agent-days | ≈ 6–10 agent-hours |
| critical path (K1 → S1 → P6.1 → P6.2 → P6.3 → P6.6 → P6.7 → Z1 → Z2 → Z3; the ENV1 / ENV2 / P6.0 chain after X3 runs alongside) | ≈ 9 working days | **≈ 2–3 calendar days** |
| total to Phase 2 Done, incl. waiting for X3 and review latency | ≈ 9–10 working days | **≈ 3–5 calendar days** |
| GPU | – | ≈ 11 GPU-hours |

- **Observed velocity.** P2-3 (R3.1–R3.9, nominal ≈ 10–13 days) and P2-4 v1 + B1 / B2 + X1 + B2-v2 (nominal ≈ 6 days)
  each finished within about one working session.
- **Contingency.** Add **0.5–1 day for every protocol v2 cycle** (a FAIL whose root cause is a declared-criterion or
  measurement defect).
- **GPU breakdown.** S1 0.1 h; P5.4–P5.6 ≈ 5 h; P6.3 ≈ 4 h; P6.5 ≈ 1.5 h; P6.6 / Z3 ≈ 0.5 h.
- **GPU placement.** Mostly GPU0. The P2-5 track may use GPU1 when its lock is free, but P1-6 optimizer candidates
  have priority there (U1). The Native run needs GPU0 with ≥ 14 GiB free (T5.8 rule).

## 8. Risks, open items, user decisions

| # | Risk / item | Mitigation |
|---|---|---|
| 1 | X3 fails or scores n_t 4 only | Contingency of §2.4. P2-6 can run with v2 and μ ≤ 0.4 after ENV2; it is BLOCKED on the friction axis only if no row passes |
| 2 | A friction range of [0.3, 0.4] under v2 is narrow against real rubber (0.6–1.0+) | Declared limitation; the real fix is X3 or a later law |
| 3 | Rubber RL throughput ≈ 1 env-step/s aggregate (§4.5) | Honest rows. The `rubber_rl_v1` labelled variant is ≈ 3× faster. No hidden reduction |
| 4 | Host-staged reset (0.27 s per env per density change) | Reported reset p95. Optional bit-identical vectorised F8 assembly if > 1 s |
| 5 | The S1 detector misses pass-throughs faster than ≈ 1.7 m/s between 1 ms checks | Blind-interval counter reported. `check_substeps` is declared and can be lowered |
| 6 | Same-family synthetic truth cannot show model-form error | Stated in every P2-5 output. The real-data protocol is declared but NOT_RUN |
| 7 | The `rl/vector_env.py` dispatch edit conflicts with P1-6 optimizer candidates | A small lazy branch at the top of `open_vector_env`, made last (P6.2) and rebased on the current `main` before Z2 |
| 8 | Native needs ≥ 14 GiB on GPU0 next to co-tenants | Launcher refuses with BLOCKED and the `nvidia-smi` readings (T5.8 pattern) |
| 9 | ADR section collisions (v5 §21, X3) | K1 renumbers v5 to §19A; X3 writes §22.11; this plan is §23 |
| U1 (user) | Robot-coupled rubber (B3 + C1 + P1 + a rubber Lab task) is outside Phase 2 by this plan | Confirm deferral or order it as a follow-up package (≈ 4–6 nominal days) |
| U2 (user) | Q1w: §5 (c) is the recommendation; (a) costs ≈ 5× on wide strips | Reviewer decides in K2; the user may override |

## 9. Probes run for this plan (CPU, read-only; scripts kept in the session scratchpad, text reproduced)

Tree: `claude/rl-rubber-p24b` at `6b36b65` (src identical to `ff15cd2` plus X2's diagnostic scripts). Warp 1.17.0 CPU,
`.venv` Python 3.12. `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python <probe>`.

**Probe 1 (step rule and slot sizes).** It called `rubber_material.step_rule(dx, 0.02, mu0_max=mu_eq_max/g_inf,
kappa_max=20·mu0_max, density_min=rho_min, cfl=0.5)` and `rubber_scene.compile_strip(RubberStripSpec(L, W, t,
n_t=t/1 mm, dx=1e-3, case="A", resolution_profile="rubber_ref_v1", traction_regions=(("tip", ...),)))`. Results:

| Range | c_p,max (m/s) | dt (µs) | substeps / 20 ms |
|---|---|---|---|
| recipe-wide visco (μ_eq ≤ 2 MPa, g∞ 0.75, ρ ≥ 900) | 251.4 | 1.989 | 10,057 |
| **P2-6 (μ0 ≤ 0.8 MPa, ρ ≥ 1000)** | 130.6 | 3.827 | **5,226** |
| workload E v2 (μ0 ≤ 1.3 MPa, ρ ≥ 1000) | 166.5 | 3.002 | 6,662 |

Particles (surface particles) for L × 16 mm × t at dx 1 mm:

| L \ t | 4 mm | 5 mm | 6 mm |
|---|---|---|---|
| 100 mm | 52,224 (15,864) | 65,280 (16,800) | 78,336 (17,736) |
| 125 mm | 65,024 (19,664) | 81,280 (20,800) | 97,536 (21,936) |
| 150 mm | 77,824 (23,464) | 97,280 (24,800) | 116,736 (26,136) |

**Probe 2 (heterogeneous batch, reset cost).**
- Setup: three `LiveRubberMaterial` lives, case A with L 20 / 24 / 16 mm, W 6 mm, t 4 / 5 / 6 mm (n_t 4 / 5 / 6),
  dx 1 mm; M = 2 Prony (g 0.15 / 0.10, τ 10 / 100 ms); μ0 4e5 / 6e5 / 3e5 Pa; ρ 1150 / 1250 / 1050; table plane
  `ContactSpec(planes=((0, 0, 1, 0),))`; tip traction region; options F8 + kick_average_v1 + F9 + F10 + guard 1.0;
  `step_material` μ0 0.8 MPa, ρ 1000. Batch `BatchedRubberMaterial(device="cpu", parity_mode=True,
  ranges={mu0, density, friction_table})`.
- Results:
  - The batch packs: N 3; particles 4,224 / 6,240 / 5,184; 5,226 substeps; dt 3.827 µs; G 1; Gt 1; contact on; M 2.
  - Build: lives 0.37 s, batch 0.045 s.
  - `reset(env_ids=[1], rng_ranges=True)`: 0.23 s.
  - Full-size slot 125 × 16 × 4 mm: 65,024 particles, build 0.49 s.
  - `tlmpm_rubber_v1.projection_blocks` for a new particle mass: **0.27 s**, 3 blocks of 255 nodes.
