# RL-Ready Simulator & Rubber Physics — progress

Specification: [RL_READY_SPECIFICATION.md](RL_READY_SPECIFICATION.md) ·
plan, acceptance and decisions: [RL_READY_PLAN.md](RL_READY_PLAN.md) ·
reports: `docs/reports/rl_ready/` · compact evidence: `docs/evidence/rl_ready/` ·
raw artifacts (git-ignored): `artifacts/rl_ready/`.

## 進行中

- **Phase 1 merged into `main` (PR #12, merge `40c94c7`, 2026-11-11).** xhigh acceptance review PASS_WITH_LIMITATIONS
  ([review](reports/rl_ready/PHASE1_ACCEPTANCE_REVIEW.md), D-13); limitations in the PR body and
  [P1_PARITY_MATRIX.md](reports/rl_ready/P1_PARITY_MATRIX.md). GitHub CI green on the merged head (CI fixes: local-only
  Franka USD tests and OpenBLAS-kernel-specific pose-interpolation comparisons are NOT_RUN where their inputs are
  absent; narrow xhigh review PASS). `acceptance_v1` frozen at `40c94c7` (D-11). Status
  **`MERGED_RCS_PENDING_OPTIMIZER_SYNC`** until the first descriptor pin in the external agentic optimizer repository.
- **PR #13 merged into `main` (merge `bfe6d5e`, 2026-11-11):** `acceptance_v2` (G9 times steps 229–400 = all s512 tool contact; `G9:window` gates tool contacts and labels; frozen at `bfe6d5e`, D-14), findings 3/4/10/12, CI concurrency; xhigh review PASS_WITH_LIMITATIONS + re-review PASS; CI green on both runner types.
- **SYNCED (2026-11-11):** the external agentic optimizer repository pinned a NEW versioned descriptor `rcs_rl_vector_v1` at RCS `bfe6d5e` (frozen checkout `/mnt/data/ningan/lab_records/external/rcs_rl_bfe6d5eaee0d`; historical descriptors untouched); smoke PASS (CPU contract; host contract in a verified writable copy: RCS tests 266 passed / 0 failed / 40 skipped; GPU contract N=2 with partial reset PASS on GPU1 under `rcs_rl_runner/gpu1.lock`); its PR merged; sync receipt chain `baseline_pin` MERGED_RCS_PENDING_OPTIMIZER_SYNC → **SYNCED** (receipt verify ok). Adapter: xhigh review PASS_WITH_LIMITATIONS + re-check GO; fixes incl. evidence path sanitising, baseline-side faults BLOCKED not REJECTED, early proposal text checks, protected pytest configs / hook refusal, `--no-renames`, writable verified test copies.
- **P1-6 cycle `p16c1` (2026-11-11):** baseline measurement under `acceptance_v2` PASS (B0 `bfe6d5e`, GPU1: N=1 9.57, N=8 13.76, N=16 13.44 env-steps/s); the external agentic optimizer's research run (1 session, $6.02, 67 min) profiled RCS (particle contacts 35.5 % of GPU time, then P2G, MUSL) and produced candidate `alias_free_v2` (opt-in; regrouped contact candidates + per-env search radius from a speed bound; its own dev runs: N=8 1.22×, identical episode codes). Both acceptance attempts **BLOCKED** (no status word, not counted) by evaluation-sandbox runtime bugs in the adapter (CUDA temp dir, interpreter mount, test artifacts, kernel cache); fixes + a real-sandbox integration test in progress, then a rerun of the acceptance. RCS untouched.
- **Phase 2 (rubber, D-12):** rubber line merged here (aa62bb5) after the **K2 combined xhigh review** ([P2_K2_COMBINED_REVIEW.md](reports/rl_ready/P2_K2_COMBINED_REVIEW.md), merge GO): v5 PASS_WITH_LIMITATIONS (Q1w: n_t 4 kept with its +0.62 % wide-strip bias labelled; v5 FAIL recorded), contact v1 PASS / v2 PASS_WITH_LIMITATIONS / B1 PASS / B2 PASS_WITH_LIMITATIONS, S1 PASS_WITH_LIMITATIONS; **K2-1 (MAJOR): contact law v3 (patch slip-rate equalisation) has unbounded patch torque — a spinning block reverses within 1 ms, a sideways-dragged strip translates rigidly — so v3 is accepted for straight-line sliding only; P2-6 uses law v2 with table friction U[0.3, 0.4] (labelled limitation)**; the first P6.0 declarations (130ccd4, v3, μ ≤ 0.8) are superseded before any P6 formal run by the v2 pair (400f5c5: `rubber_strip_drag_v2` 7395e610…, `rubber_rl_tests_v2` 6db54681…; profile `rubber_rl_env_v2` = contact law v2), after a formal v2 friction-envelope check (`rubber_contact_envelope_v2`, declared 08b9dcd: μ 0.3 PASS everywhere, μ 0.4 FAIL on the strip scene at n_t 4/6; K2-1 spin-down PASS under v2) — **RL rubber table friction FIXED at 0.3 (labelled limitation, not randomised)**; RL-08 measures the Q1w bias on the P2-6 geometry with a declared fallback. **P2-6 through P6.3 done** (`claude/rl-rubber-p26` aeb3f64): `open_vector_env(backend="kinematic_grip_batched")` rubber strip-drag task (6-D grip actions, 78-dim deployable obs, opt-in privileged), RL-01 contract, RL-02 L2/L3, RL-03 seeds, RL-04 isolation, RL-06 long rollout (53 partial resets, VRAM growth 0), RL-07 throughput (N 8 / N 1 1.46×, 0.96 env-steps/s at N 8 full fidelity, 3.28 reduced), RL-08 convergence PASS; RL-08 b Q1w bias on L/t 37.5 +0.71 % (trigger not met); **RL-05 sanity FAIL** (declared scripted controller too naive); **RL-05 v3 FAIL** (`rubber_rl_tests_v3`, pivot controller, fresh seeds: beats zero action 8/8 but 0 successes — the tip reaches the goal and then swings out) → v4 for RL-05 only, declared as the final scripted attempt (developed at full resolution; if it fails RL-05 stays FAIL as a known limitation). Reset cost: the 5 s p95 traced to the OpenBLAS all-core thread pool on a loaded host (1 thread: 0.06 s). **Native RL-09 PASS** (`claude/rl-rubber-p26n` 4093f49): the same rubber recipe (profile `rubber_rl_env_v2`) runs in a new Kit host fed by PhysX-measured clamp poses (`native_rubber_001`, GPU0, 100/100 steps, 1.81 s/step, SHARED_GPU); 30 headless replays (GPU1): receipts identical, material inputs bitwise equal, 0/46 physics statistics outside the declared band (max 0.65 of limit), held-out 10/10 inside, 0 invalid bits, bridge error 7.45e-9 m. Frozen `isaac_sim_runner.py` / `live_material.py` untouched. **P2-5** (`claude/rl-rubber-p25` fd2699e): calibration interface, protocol v1, pilot, thresholds and frozen sealed formal dataset done (CAL-01/02/03 PASS on CPU); formal fit stopped for compute before a receipt (CAL-04..07 NOT_RUN) → v2 fit method (U-P2-4). Running: K2 fixes (Q1w label in receipts, -09 sub-check, wording); P2-6 env (P6.0 v2 → P6.1–P6.3) on `claude/rl-rubber-p26`; Native rubber host (P6.4/P6.5) on `claude/rl-rubber-p26n`; P2-5 calibration on `claude/rl-rubber-p25`. Status `SIMULATOR_ONLY_UNCALIBRATED`.
- **2026-11-11 usage-limit pause** (weekly API limit, ~06:30–18:00 KST): agents resumed afterwards; no work lost.

## 待使用者決定

- Real rubber measurements are not available on this machine; Phase 2
  calibration status will stay `SIMULATOR_ONLY_UNCALIBRATED` unless data is provided.
- P1-6 defaults in force until the user decides otherwise (none blocks work):
  U1 optimizer-written candidate code runs on GPU1 only (not GPU0);
  U2 speed gate (first optimizer acceptance uses `acceptance_v2`, D-14; `acceptance_v1` is frozen and not used):
  v2 times the cutting phase (window being finalised to start at tool contact), N ∈ {1, 16}, 3 ABBA pairs, N=16
  paired median ≥ 1.05 with all 3 pairs faster, every N ≥ 0.97 (v1 was: N=16 median ≥ 1.05 with ≥ 4/5 pairs
  faster); both: VRAM peak ≤ 1.10 × baseline + 256 MiB, reset p95 ≤ 1.25 × baseline + 2 ms (shared GPU: WARN up
  to + 10 ms);
  U3 per-cycle budget: research agent $20 / 6 h plus one xhigh review (~1 working day, 3–5 GPU-h);
  U4 the research agent gets no web access.
- Phase 2 defaults in force until the user decides otherwise (none blocks work; plan `docs/reports/rl_ready/P2_REMAINING_PLAN.md` / ADR 0023 §23 on the rubber branches):
  U-P2-1 robot-coupled rubber (rubber pad wrenches fed back to the Isaac Lab robot, tasks B3 + C1) stays out of
  Phase 2; P2-6 uses kinematic grips (no coupling error; the 20 ms handoff was assessed significant, CONT-07);
  U-P2-2 RUBBER-PHYS-11 Q1w (wide-strip sag GCI 2.16 % vs 2 % at n_t 4): accept n_t 4 with its measured +0.62 %
  bias as a labelled limitation (the v5 FAIL stays recorded; a v6 rule passing the existing data would be loosening
  after seeing results; n_t 6 costs ~5× per step); final call in the combined rubber review;
  U-P2-3 rubber friction in the RL rubber task is FIXED at μ = 0.3 with contact law v2 (formal v2 envelope check:
- **U-P2-4 (P2-5 fit compute)**: the v1 formal synthetic fit (`rubber_calib_tests_v1`) converged too slowly (projected ~15 GPU0-h vs the plan's 2.5 GPU-h) and was stopped before any fit receipt; truth still sealed. **Default:** declare `rubber_calib_tests_v2` changing only the fit method (staged, bounded, FD step from a measured repeatability) with a hard budget (fit ≤ 3 GPU-h, negative control ≤ 1.5 GPU-h; on exhaustion the best iterate is scored normally). **Alternative:** fund the unchanged v1 fit (~10–15 GPU0-h).
  0.4 fails on the strip scene; K2-1: the v3 law is
  valid for straight-line sliding only); higher μ (real rubber is often 0.6–1.0+) is a labelled limitation
  (friction–normal flutter, X2 negative result; a torque-bounded equalisation is future work).
- P1-5 defaults (none blocks work; [P1_5_PLAN.md](reports/rl_ready/P1_5_PLAN.md)):
  U-P15-1 the frozen `tlmpm_v5` CUT is **not** batched in Phase 1 (option (a), ~3–4 agent-days, deferred): the batched
  path runs CUT on a new opt-in `bar_flat_board_tlmpm_v7_2_v1` scene, the frozen v5 CUT runs single-env headless
  (replay of Native Demo 1); the CUT parity row therefore stays PARTIAL;
  U-P15-2 no batched camera/vision profile (cameras stay UNSUPPORTED in the batched path);
  U-P15-3 no per-episode geometry/material randomisation inside a running batch (per-env seeds + rebuild only).

## Phase 1 results (summary)

Parity matrix ([P1_PARITY_MATRIX.md](reports/rl_ready/P1_PARITY_MATRIX.md)): **9 PASS / 6 PARTIAL / 1 UNSUPPORTED**
(cameras in the batched path). CUT stays PARTIAL, so "whole-RCS RL-ready" is PARTIAL for CUT. Checkbox 7
(external optimizer sync) is PENDING. Throughput rows: the formal `EXCLUSIVE_GPU` table on GPU1
([P1_PERFORMANCE_TABLE.md](reports/rl_ready/P1_PERFORMANCE_TABLE.md), 20 of 23 runs exclusive, code `691b571`);
P1-4 and every partial-reset latency are `SHARED_GPU`.

- **P1-1 core + review fixes** (204 CPU tests at the time; `isaac_lab_newton` coupling + error codes).
- **P1-2 DONE** ([report](reports/rl_ready/P1_1_P1_2_MILESTONES.md), evidence `p1_2_milestones_001.json`, f0a544b):
  `open_vector_env` (workload A, reduced fidelity: replayed edge, no robot). M0 s512/s15, M1 N=4 (staggered partial
  resets, policy + replay envs), M2 N=8 ×3 episodes (24/24) and N=16 (16/16): final codes = Native, frozen evaluator
  PASS, L2 every step, no leak. SHARED_GPU matched-window throughput N=1 → N=8 / N=16: 1.40× / 1.37× (vs the frozen
  single env ~1.24–1.26×); VRAM 834 / 1378 / 2064 MiB; partial reset 4–9 ms. L4 RUN_TO_RUN BLOCKED (frozen
  `tlmpm_v5._particle_contacts` mixes add/max atomics). P1-1 `ab_timing_001` is INVALID_MULTI_GRID_PROCESS (its
  "≈1.9× at N=4" claim is withdrawn); guard `UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS` (a24ac46).
- **P1-3a DONE and merged (3682786)**: opt-in `particle_contact_mode="alias_free_v1"` (C2 alias-free hash dims + C1 +
  speed guard with conditional-node fallback); default `frozen_v1` kernel source byte-identical. Contact stage
  −49…−51 %; same-state contacts/impulses equal; full episodes N=1 (s512, s15) and N=4: final codes = Native,
  frozen evaluator PASS. EXCLUSIVE pairs: 1.38–1.47× (pre-contact windows at N ≥ 4). RUN_TO_RUN and C3 not done.
  **Warp 1.17 finding:** one static host hash-grid descriptor — a replayed graph uses the *last-built* grid, so a
  process holding 2+ `wp.HashGrid`s computes contacts on the wrong grid (probe
  `scripts/rl_ready/probe_hashgrid_descriptor.py`: 8,804,086 → 0 candidates). Rule: one device HashGrid per process;
  multi-variant in-process A/B runs are INVALID (`INVALID_MULTI_GRID_PROCESS`).
- **xhigh review of P1-1/P1-2/P1-3a at 3682786: PASS_WITH_LIMITATIONS ×3**
  ([review](reports/rl_ready/P1_1_P1_2_P1_3A_ACCEPTANCE_REVIEW.md)). **Review fixes merged (f5178f5)** (evidence
  `review_fix1_001.json`): F1 per-launch grid refresh + sticky `FOREIGN_DEVICE_HASHGRID` (bit 256 on the integration
  branch, 128 on the fix branch; refresh ≤ 0.25 ms GPU per step); F2 pre-declared position band v2 (20 frozen
  single-env reference runs, one-sided 99 % log-normal bound; still descriptive, read by `acceptance_v1` since
  ac3f6c6); F3 matched windows 1.40× / 1.37×; F4 ~1.24–1.26× vs the frozen single env (inferred); F5 terminal capture;
  F6/F8/F9/F10 corrections; FallbackGrid device pin db036a0.
- **P1-1b S0–S3** ([report](reports/rl_ready/P1_1B_ROBOT_BATCHING.md), evidence `p1_1b_s2_s3_001.json`, 78a211b):
  G3 snapshot restore PASS; G5 L0 bitwise PASS; G5 L3 (frozen single-material replays reproduce the batch final codes,
  both seeds) PASS — not a Kit run; **S3 TRIM B1 (rigid mount, reduced fidelity): frozen evaluator PASS**, codes
  bitwise across N. The first B2 attempt FAILed (knife slip 8–13 mm / 0.5 rad in MuJoCo-Warp fingers), kept.
- **B2 grasp follow-up merged (52dfcf6)** (evidence `p1_1b_b2_001.json`): root cause MuJoCo-Warp 3.12 GJK/EPA
  box–box multicontact collapsing to 1–2 contacts; fix opt-in `mjwarp_grip_v1` (`box_box` collider, impratio 10;
  `physical_calibration` NOT_VALIDATED). Robot-only N=2 holding 1196/1200 steps, slip 1.5 mm / 15 mrad (Native
  ~1 µm / 1e-4 rad); **S3 TRIM B2 N=2/4: frozen evaluator PASS in every env**. Material SIGSEGV after a lost grasp
  fixed with `material_input_domain_v1` (sticky `INPUT_OUT_OF_DOMAIN`, bit 128 on the integration branch, 64 on the
  B2 branch). B2 G1/G3 deterministic, G4, G5 with the profile: NOT_RUN.
- **P1-2b merged (6e1e544)** (evidence `p1_2b_001.json`): `open_vector_env(backend="isaac_lab_newton")` with the
  same contract (`rl/vector_contract.py`), per-env `tasks=`, `termination_reason` bits (1 rigid, 2 material, 4 grasp
  lost, 8 IK streak, 16 truncated, 32 input guard, 64 infrastructure), `classify_process_exit`, opt-in device
  evaluation stream, input guard (default-on, `input_guard=False` rollback), terminal capture, diagnostics-only
  `allow_multiple_device_hashgrids`.
- **P1-4 merged** ([report](reports/rl_ready/P1_4_LONG_ROLLOUT.md), evidence `p1_4_001.json`): supervisor (child
  process per rollout, fsync'd JSONL ledger, task vs infrastructure classification, restart policy). **LR-A PASS**
  (N=8 workload A, 5.06 h, 64 full episodes, canaries 16/16 bitwise = Native, VRAM peak 1,412 MiB, foreign SM mean
  35 %, SHARED_GPU). **LR-B FAIL** on its pre-declared VRAM bound B0 (2,304 vs 2,253 MiB; 668 resets bitwise-checked,
  0 mismatches; kept as FAIL). LR-C PASS. LR-D 5/5 PASS. LR-A2 and LR-E NOT_RUN.
- **P1-5 CUT + RL smoke merged (03a2f72)** ([report](reports/rl_ready/P1_5_CUT.md), evidence `p1_5_cut_001.json`):
  frozen v5 CUT Demo 1 single-env headless replay ×4 PASS; opt-in `bar_flat_board_tlmpm_v7_2_v1`; fresh Native v7_2
  CUT PASS; **batched CUT N=1/4/8×2/16: 35/35 scripted episodes PASS**. RL pipeline smoke (`rcs_rl_rollout_v1`,
  no training): S-CUT N=4 and S-TRIM N=8 PASS (first S-TRIM run FAILed on a runner bug, kept; fixed 9a51a86).
  CUT parity row PARTIAL by design (U-P15-1). Lab CUT (T5.9) and the batched replay of the Native v7_2 recording
  (T5.8 on GPU) NOT_RUN.
- **P1-6 R0 merged (87cfa3b)** ([report](reports/rl_ready/P1_6_R0_ACCEPTANCE_HARNESS.md)): harness, judge,
  `acceptance_v1.json`, local runner; GPU smoke PHYSICS/PERFORMANCE/INTEGRITY PASS (SHARED_GPU). `acceptance_v1` was
  amended in place four times before any baseline (two loosened a gate after the data had been seen); it freezes at
  the Phase 1 merge (D-11). Its G9 window is pre-contact at N=8/16, so a P1-6 acceptance needs `acceptance_v2`. The
  optimizer-side design and adapter live in the external agentic optimizer repository (CPU rehearsal PASS; live GPU
  rehearsal and first pin NOT_RUN until the merge). P1-6 GPU jobs use the `rcs_rl_runner/gpu<N>.lock` locks.
- **Consolidation merged (3674780)**: `grip_contact_profile` passthrough (`UNSUPPORTED_GRIP_CONTACT_PROFILE`),
  `FAIL_INTEGRITY` in `evaluate()`, one `material_termination_bits` mapping, configured-scene `tasks=`, parity
  matrix, README section. Tests at 53b915e: full CPU 2443 passed / 1 failed / 28 skipped / 6 xfailed; Torch 343 /
  10 skipped; Isaac Lab 56. The 1 failure (`test_wp12_phase5_native_success.py::test_reproduces_phase2b_records`) is
  pre-existing and unrelated (test and scripts byte-identical to `origin/main`): `workcell_live_006`'s
  `analysis_001/analysis.json` says TASK_SUCCESS, matching the committed evidence, but the test's
  `run_wp12_phase5.native_execution` re-derives CUTTING_FAILURE from `run/` and the assertion at
  `tests/test_wp12_phase5_native_success.py:201` fails. It fails only when `artifacts/wp12` links the main
  checkout's artifacts. WP-12 records are out of scope and untouched (open item).
- **Formal EXCLUSIVE_GPU table merged (8e503e1)**: 20 of 23 runs EXCLUSIVE_GPU (3 SHARED only for a < 2 s, 2–7 MiB
  foreign graphics context; 2 of them rerun EXCLUSIVE). Env-steps/s: single_env_headless s512/s15 7.57/7.21;
  workload A `frozen_v1` / `alias_free_v1` N=1 6.75 / 9.74, N=16 9.83 / 13.56 (saturates at N≈4); CUT v7_2
  N=1/8/16 6.85/10.35/10.22 (33/33 episodes PASS); B1 N=1/4/8 4.96/7.66/8.67 (N=4 exclusive repeat; N=16 8.79 SHARED only); B2
  `mjwarp_grip_v1` N=4 7.37 (4/4 PASS, grasp held).
- **Phase 1 xhigh acceptance review at f03fbff: PASS_WITH_LIMITATIONS**, merge GO with conditions
  ([review](reports/rl_ready/PHASE1_ACCEPTANCE_REVIEW.md)). Tests in the review: CPU selection 511 passed / 26
  skipped (CUDA hidden and GPU0 visible); Torch venv 334 passed / 10 skipped; Isaac Lab venv 56 passed.

## Phase 2 (rubber) detail

**Phase 2 (rubber) — parallel since 2026-11-10 14:05 (user, D-12):** worktree `/home/ningan/RoboticCuttingSim-rl-rubber`, branch `claude/rl-rubber` (from 18afce3). **P2-0 decision DONE — ADR 0023 accepted** (2f316c1 on `claude/rl-rubber`): opt-in TLMPM rubber solver `tlmpm_rubber_v1` (single-env reference + batched on the Phase 1 layout; Newton VBD rejected: 22–107× too soft and dt/iteration-dependent in the probe); `neo_hookean_jbar_v1` (isochoric Neo-Hookean + J̄ cell-averaged volumetric term, exact PK1, sticky invalid outside J, J̄ ∈ [0.2, 5]); near-incompressibility = J-bar + documented reduced κ = 20 μ (ν_eff 0.4754; real K recorded separately; J−1 = 1.2/2.5/5.4 % at stretch 1.25/1.5/2.0); pure FLIP, dt = 0.5·dx/c (~5 µs, ~3,850 substeps per 20 ms at μ 0.5 MPa, dx 1 mm); ≥3 cells through thickness; Prony-series viscoelasticity (exact exponential update) designed for P2-3. CPU probes: cantilever frequency 11.54 vs 11.57 Hz beam theory, elastica sag within 0.4 %, energy drift < 0.05 %. **P2-0/P2-1 implemented** (6c9477f on `claude/rl-rubber`, [report](https://github.com/AnsonNing/RoboticCuttingSim/blob/claude/rl-rubber/docs/reports/rl_ready/P2_0_P2_1_RUBBER.md)): recipe `rubber_material_v1`, calibration manifest `NO_REAL_DATA`, frozen RUBBER-PHYS protocol v1 (+ v2 changing only the initializer stop rule, declared before the formal run), `tlmpm_rubber_v1` single-env + batched. Formal CPU run (66 configs): **01 patch, 02 small sag (24 configs), 03 L⁴/t⁻² scaling, 04 free vibration (−0.44 % vs Euler–Bernoulli, batched N=4 bitwise), 05 large sag (elastica ≤ 0.26 %), 06 uniaxial load–unload, 08 simple shear, 09 records: PASS; 07 stretch & release: FAIL** (non-finite 3 steps after release at CFL 0.5 — compression stiffening ~5.7× eats the stability margin; at CFL 0.25 the strip folds onto itself, J 0.19, no self-contact); 10 single vs batched: L0/L1 bitwise PASS on CPU, GPU L2/L3 NOT_RUN (no GPU slot); 11 refinement NOT_RUN (P2-2). **P2-1 follow-up (xhigh, ADR 0023 §18, 5b0a44d):** 07 root cause = the protocol (instant release from λ 1.5 makes the 60×10×4 mm strip buckle and fold at the clamp, ~360× Euler strain; the fold hinge exceeds the stable step) — not J-bar (no negative modes down to stretch 0.3) and not FLIP at speed; opt-in `stable_step_guard_v1` (sticky `RUBBER_CFL_EXCEEDED`, no state change) + `deformation_reserve`; no self-contact in v1 (fold scene moves to P2-4); `static_stop_v3` replaces the unsound v2 stop rule. Formal v3 (declared 849e10b before the run): -03, -05, -12, -07a (λ 1.5 force-ramped release) PASS; **-07b (λ 1.03 instant) FAIL** (energy 1.058 % vs 1 %, clamp dissipation) and **-02 FAIL** (1 of 24 configs 1.0203 vs 1.02, float32 static precision) — both kept. GPU smoke (SHARED_GPU, 97,280 particles/env, dt 3 µs, 6,662 substeps per 20 ms): N=1/4/8 L2 PASS, 0.43/0.62/0.74 env-steps/s, VRAM 744/840/968 MB; L3 band (3 repeats) FAIL → sound rule to be declared. The "pollution" was a Phase 1 device-default bug (FallbackGrid launched on cuda:0 for CPU arrays) — fixed on this branch db036a0. **P2-2 DONE on `claude/rl-rubber` (3a43d1b**, [report](https://github.com/AnsonNing/RoboticCuttingSim/blob/claude/rl-rubber/docs/reports/rl_ready/P2_2_RUBBER_HYPERELASTIC.md)): protocol v4 declared and SHA-pinned (3d12969) before its runs; opt-in energy-consistent clamp (F8: -07b energy 1.058 % → 0.33 %), grid-consistent traction work (F9: -07a 0.96 % → 0.0047 %), float64 static accumulation (F10: float32 drift 0.9–4.2e-3 → ≤ 4.1e-7); v4 formal: -02, -03, -05, -06, -07 PASS; **-08 FAIL** (validity: step 1.30 vs guard 1.0 at γ = 2; stresses exact to 1.6e-6); refinement **-11 FAIL on Q1** (GCI 3.98 % — p_obs 3.26 outside the declared [1, 3] window; Q2 1.28 %, Q3 0.01 %, Q4 0.92 % PASS); F3 L3 v4 rule (20 GPU repeats, prediction bound) PASS; F4 batched L0 bitwise; guard +1.2–1.9 % GPU time. Throughput workload E (41,220 particles/env, dt 4 µs, 4,996 substeps per 20 ms, SHARED_GPU GPU0): N=1/4/8/16 = 0.64 / 1.35 / 1.47 / 1.52 env-steps/s, VRAM 728–952 MB. **xhigh P2-0..P2-2 review (ae6939d on `claude/rl-rubber`): P2-0 PASS, P2-1 PASS_WITH_LIMITATIONS, P2-2 PASS_WITH_LIMITATIONS; merge GO for 3a43d1b** (held until the Phase 1 PR branch is cut, so Phase 2 code does not ride into the Phase 1 PR). Decisions: (a) bending-dominated scenes move to n_t ≥ 4 (`rubber_ref_v1` = full fidelity; workload E n_t 3 becomes an RL profile labelled reduced fidelity with its measured bias; ≈3.1× slower per env-step), declared as `rubber_phys_tests_v5` (grids 3/4/6/8, production GCI from 4/6/8, rule and thresholds byte-identical to v4) before any run; (b) v5 -08 with a declared derivation rule giving r = 1.5. Medium findings: ADR's ≤ 0.5 % bending claim at n_t 3 is wrong; batched path still lacks the energy-consistent clamp / float64 state (being added); "bitwise" L0 is value equality. Now: batched P2-3 prerequisites (energy-consistent clamp, F9/F10, `motion_table_v1` BC launch — to be reviewed). Status `SIMULATOR_ONLY_UNCALIBRATED`. Phase 1 keeps GPU priority.

## P1-0 baseline — 2026-11-10 — PASS

Report [P1_0_BASELINE.md](reports/rl_ready/P1_0_BASELINE.md), evidence
[p1_0_baseline_001.json](evidence/rl_ready/p1_0_baseline_001.json),
[inventory](evidence/rl_ready/p1_0_inventory_001.json). Commits 588c17a … 9f46fc9.

| boundary | workload | warm/cold | particles/env | env-steps/s | sim-time/wall | VRAM peak | reset p95 | physics | GPU |
|---|---|---|---|---:|---:|---:|---:|---|---|
| single_env_native (WP-17 evidence) | D s512 | warm | 102,732 | 4.040 | 0.0808 | 5542 MiB | n/a | PASS | EXCLUSIVE (process list) |
| single_env_native (WP-17 evidence) | D s15 | warm | 103,898 | 3.866 | 0.0773 | 5638 MiB | n/a | PASS | EXCLUSIVE (process list) |
| single_env_headless | A/D s15 replay | warm | 103,898 | **7.419** | 0.1484 | 758 MiB | 0.149 s | PASS | EXCLUSIVE |
| single_env_headless | A/D s512 replay | warm | 102,732 | 5.928 | 0.1186 | 758 MiB | 0.185 s | PASS | SHARED_GPU |
| batched_headless | A | — | — | NOT_RUN | — | — | — | — | — |

All rows: N=1, 50 Hz control, material dt 1e-4 s, 200 substeps, no cameras, 1 mm / 4 mm, tlmpm_v7_2 law 8 K=2,
RTX 5090 #1, driver 570.211.01. Native rows include Kit, RTX cameras and recording; headless rows run the same
material without Kit/robot (replayed measured edge + pad boxes). Not one speedup number.

- Replay of the recorded measured edge/pad trajectory reproduces the Native **final cut codes exactly**
  (4/4 runs per seed); frozen TRIM evaluator PASS 3/3 per seed with Native's metrics. Max position difference
  1.4e-5–3.8e-5 m (s512), 2.0e-4–4.6e-4 m (s15); 1–3 transient mid-episode code differences vanish by the end.
  Pads never touch the material in these episodes. The WP-17 seed-512 WP-16↔WP-17 difference stays open.
- Kernel split (eager, shared GPU): `_particle_contacts` ≈47 % of ~0.83 ms/substep, `_p2g` ≈15 %, phase-independent
  → next bottleneck for P1-3. Co-tenant graphics load on GPU1 changed warm throughput 4.40–7.42 env-steps/s.
- Cold start 5.6–6.4 s with a warm Warp kernel cache, 44.3 s with an empty cache.

## Log

- 2026-11-10 — P1-1b S1 PASS (robot-only gates; batched robots cut with material attached); S2/S3 started.
- 2026-11-10 — P1-1 review fixes done; P1-3 analysis committed (hash aliasing: ~58 % spurious candidates; C1/C2 bitwise); P1-3a started in worktree `RoboticCuttingSim-rl-p13`.
- 2026-11-10 — P1-1b S0 PASS (CPU bitwise); P1-1 review fixes started; S1 robot-only GPU on GPU0.
- 2026-11-10 — P1-1 core done (bitwise CPU parity/isolation/reset; GPU smoke PASS); vector env + milestones started; P1-1 xhigh review started.
- 2026-11-10 — P1-0 PASS; ADR 0022 accepted (D-10); pytest pythonpath fix (45ecf92).
- 2026-11-10 — ADR 0021 accepted with amendments; D-09 pytest overlay for the Torch venv; P1-1/P1-2 implementation started.
- 2026-11-10 — `RL_READY_PLAN.md` committed (decisions D-01..D-07); D-08 headless runtime venv.
- 2026-11-10 — specification copy committed (plan + brief, naming-rule substitution only).
