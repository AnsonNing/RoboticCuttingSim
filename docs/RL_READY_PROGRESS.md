# RL-Ready Simulator & Rubber Physics — progress

Specification: [RL_READY_SPECIFICATION.md](RL_READY_SPECIFICATION.md) ·
plan, acceptance and decisions: [RL_READY_PLAN.md](RL_READY_PLAN.md) ·
reports: `docs/reports/rl_ready/` · compact evidence: `docs/evidence/rl_ready/` ·
raw artifacts (git-ignored): `artifacts/rl_ready/`.

## 進行中

- P1-1 core DONE; review fixes DONE (204 CPU tests PASS; `isaac_lab_newton` coupling + error codes added).
- Running agents / state (2026-11-10 ~10:00 KST):
  1. P1-2 DONE ([report](reports/rl_ready/P1_1_P1_2_MILESTONES.md), evidence `p1_2_milestones_001.json`, f0a544b):
     `open_vector_env` (workload A, reduced fidelity: replayed edge, no robot). M0 s512/s15, M1 N=4 (staggered
     partial resets, policy + replay envs), M2 N=8 ×3 episodes (24/24) and N=16 (16/16): final codes = Native,
     frozen evaluator PASS, L2 every step, no leak. Matched-load throughput (co-tenant ~46 %, SHARED_GPU):
     N=1 3.354 → N=8 4.849 → N=16 4.752 env-steps/s as first reported (1.45× / 1.42×, but over mismatched step
     windows; matched windows give **1.40× / 1.37×** per the acceptance review, being corrected in the report;
     vs the frozen single env ~1.24–1.26×); VRAM 834 / 1378 / 2064 MiB; partial reset
     4–9 ms. Kernel time ~linear in N (one 103k-particle env already fills most of the GPU). L4 RUN_TO_RUN BLOCKED
     (frozen `tlmpm_v5._particle_contacts` mixes add/max atomics). HashGrid audit: no P1-2 evidence run held >1
     device grid; P1-1 `ab_timing_001` is INVALID_MULTI_GRID_PROCESS → its "≈1.9× at N=4" claim is withdrawn.
     Guard `UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS` added (a24ac46).
  - P1-3a merged into this branch (3682786; release() also frees the alias_free_v1 grid, new CPU test).
  - **xhigh acceptance review of P1-1/P1-2/P1-3a at 3682786: PASS_WITH_LIMITATIONS ×3, no BLOCKER**
    ([review](reports/rl_ready/P1_1_P1_2_P1_3A_ACCEPTANCE_REVIEW.md), 2480dc2). MAJOR: frozen_v1 grid-after-build hazard
    is only checked at build/full reset; the 4-run L3 position band is not a valid pass/fail test and some statuses were
    misreported. Fixes: findings 1–4, 6, 8–10 in worktree `/home/ningan/RoboticCuttingSim-rl-fix1` (grid refresh
    before each launch, pre-declared position band v2 with ≥10 replays/seed, matched-window throughput, stale text);
    findings 5 (autoreset → evaluate) and 7 (diagnostics opt-out) folded into P1-2b.
  - Running now: xhigh acceptance review of P1-1/P1-2/P1-3a at 3682786; P1-2b (isaac_lab_newton dispatch, `tasks=`,
    `termination_reason`, full evaluation stream, per-env pose-bounds invalidation) in worktree
    `/home/ningan/RoboticCuttingSim-rl-p12b`; B2 grasp root cause in `/home/ningan/RoboticCuttingSim-rl-b2`;
    P1-6 R0 in `/home/ningan/RoboticCuttingSim-rl-r0`; P1-5 plan DONE (30b8a67) → CUT groundwork (T5.1, T5.3–T5.6,
    T5.8) in `/home/ningan/RoboticCuttingSim-rl-p15` and P1-4 supervisor/ledger + N=8 long run (T4.1, T4.3, T4.4) in
    `/home/ningan/RoboticCuttingSim-rl-p14`. Parity matrix today: 4 PASS, 8 PARTIAL, 1 UNSUPPORTED (cameras),
    2 NOT_RUN (CUT, RL smoke).
  2. P1-1b S0–S3 DONE ([report](reports/rl_ready/P1_1B_ROBOT_BATCHING.md), evidence `p1_1b_s2_s3_001.json`, 78a211b):
     reward/evaluator wiring in the Lab batched env; G3 snapshot restore PASS; G5 L0 bitwise PASS, G5 L3 (frozen single
     replays reproduce batch final codes, both seeds) PASS; **S3 TRIM B1 (rigid mount, reduced fidelity) N=2/4 both
     seeds: frozen evaluator PASS**, codes bitwise across N; **B2 physical grasp FAIL** (knife slips 8–13 mm / 0.5 rad
     in MuJoCo-Warp fingers during HOME→skim reorientation, before material contact → GRASP_LOST; Native PhysX holds);
     B2 under RUN_TO_RUN BLOCKED; G4 B2 arm offset 3.5e-4 rad FAIL vs pre-set tolerance; material graph SIGSEGV after a
     lost grasp (edge leaves domain) → must become an invalid flag; CUT on `tlmpm_v7_2` NOT_RUN (CUT scenes use
     `tlmpm_v5`). HashGrid census: no P1-1b process held >1 device grid.
  3. P1-3a DONE and merged (3682786; branch `claude/rl-p1-3a`, trial-merged tree: 2204 CPU + 113 Torch tests PASS). Opt-in
     `particle_contact_mode="alias_free_v1"` (C2 alias-free hash dims + C1 + speed guard with conditional-node
     fallback); default `frozen_v1` kernel source byte-identical. Contact stage −49…−51 %, full control step
     1.40–1.45× at N=1/4/8/16 (all SHARED_GPU); same-state contacts/impulses equal; full episodes N=1 (s512, s15) and
     N=4: final codes = Native, frozen evaluator PASS. Not done: EXCLUSIVE_GPU timing, RUN_TO_RUN, C3.
     **Warp 1.17 finding:** one static host hash-grid descriptor — a replayed graph uses the *last-built* grid, so any
     process holding 2+ `wp.HashGrid`s computes contacts on the wrong grid for all but the last-built one (probe
     `scripts/rl_ready/probe_hashgrid_descriptor.py`: 8,804,086 → 0 candidates). `alias_free_v1` is protected;
     `frozen_v1` is now guarded at batch build. Multi-variant in-process A/B runs are INVALID
     (`INVALID_MULTI_GRID_PROCESS`); P1-1b and P1-2 evidence audited clean.
  4. P1-6: design DONE in the external agentic optimizer repository (design doc + ADR + CPU scaffold, 35 tests);
     its thin adapter (worktree/proposal/bench/accept/PR/pin/receipt) is implemented there (CPU rehearsal against a
     fake RCS repo PASS, full CPU suite 781 passed; live GPU rehearsal and first pin NOT_RUN until R0 + Phase 1 merge).
     R0 is being implemented in worktree `/home/ningan/RoboticCuttingSim-rl-r0` (branch `claude/rl-p1-6-r0` from
     `claude/rl-p1-3a`).
     Cycle 1 baseline = the Phase 1 merge on RCS `main`. RCS-side prerequisite R0 (after P1-2 and P1-3a land):
     `scripts/rl_ready/run_perf_acceptance.py` (variant JSON → `open_vector_env` kwargs, `--out` outside the
     checkout, stdout timing markers, physics dumps), `scripts/rl_ready/judge_perf_acceptance.py` (runs from the
     frozen evaluator checkout on dumped outputs), frozen `configs/rl_benchmark/acceptance_v1.json`.
     P1-6 GPU jobs use the `rcs_rl_runner/gpu<N>.lock` locks, never the other project's `locks/gpu1.lock`.
- Both GPUs shared with other projects: rows `SHARED_GPU`; formal rows rerun in an exclusive window.
- Next: review fixes → merge P1-2b / B2 follow-up → P1-4 long rollout → P1-5 parity + CUT → R0 → Phase 1 review/PR/merge → first pin → P1-6 cycle.

## 待使用者決定

- Real rubber measurements are not available on this machine; Phase 2
  calibration status will stay `SIMULATOR_ONLY_UNCALIBRATED` unless data is provided.
- P1-6 defaults in force until the user decides otherwise (none blocks work):
  U1 optimizer-written candidate code runs on GPU1 only (not GPU0);
  U2 `acceptance_v1` speed gate: at N=16 paired median ≥ 1.05 with ≥ 4/5 pairs faster, every N ≥ 0.97,
  VRAM peak ≤ 1.10 × baseline + 256 MiB, reset p95 ≤ 1.25 × baseline + 2 ms;
  U3 per-cycle budget: research agent $20 / 6 h plus one xhigh review (~1 working day, 3–5 GPU-h);
  U4 the research agent gets no web access.
- P1-5 defaults (none blocks work; [P1_5_PLAN.md](reports/rl_ready/P1_5_PLAN.md)):
  U-P15-1 the frozen `tlmpm_v5` CUT is **not** batched in Phase 1 (option (a), ~3–4 agent-days, deferred): the batched
  path runs CUT on a new opt-in `bar_flat_board_tlmpm_v7_2_v1` scene, the frozen v5 CUT runs single-env headless
  (replay of Native Demo 1); the CUT parity row therefore stays PARTIAL;
  U-P15-2 no batched camera/vision profile (cameras stay UNSUPPORTED in the batched path);
  U-P15-3 no per-episode geometry/material randomisation inside a running batch (per-env seeds + rebuild only).

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
