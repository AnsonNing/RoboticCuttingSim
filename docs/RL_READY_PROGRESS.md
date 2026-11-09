# RL-Ready Simulator & Rubber Physics — progress

Specification: [RL_READY_SPECIFICATION.md](RL_READY_SPECIFICATION.md) ·
plan, acceptance and decisions: [RL_READY_PLAN.md](RL_READY_PLAN.md) ·
reports: `docs/reports/rl_ready/` · compact evidence: `docs/evidence/rl_ready/` ·
raw artifacts (git-ignored): `artifacts/rl_ready/`.

## 進行中

- P1-1 core DONE; review fixes DONE (204 CPU tests PASS; `isaac_lab_newton` coupling + error codes added).
- Running agents:
  1. P1-2 integration + milestones (GPU1, `rcs_rl_runner/gpu1.lock`): M0/M1 done, M2 N=8/16 running on post-fix
     HEAD → `artifacts/rl_ready/p1_2/`, report `docs/reports/rl_ready/P1_1_P1_2_MILESTONES.md`.
  2. P1-1b S1 robot-only GPU probes (GPU0, `rcs_rl_runner/gpu0.lock`) → `artifacts/rl_ready/p1_1b/`.
  3. P1-3a `alias_free_v1` contact mode in a separate worktree `/home/ningan/RoboticCuttingSim-rl-p13`
     (branch `claude/rl-p1-3a`, merged back after M2) — design: [P1_3_BOTTLENECK_ANALYSIS.md](reports/rl_ready/P1_3_BOTTLENECK_ANALYSIS.md).
- Both GPUs shared with other projects: rows `SHARED_GPU`; formal rows rerun in an exclusive window.
- Next: P1-1/P1-2 acceptance review → merge P1-3a → P1-1b with material → P1-4 long rollout → P1-5 parity matrix → P1-6.

## 待使用者決定

- (none yet) Real rubber measurements are not available on this machine; Phase 2
  calibration status will stay `SIMULATOR_ONLY_UNCALIBRATED` unless data is provided.

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

- 2026-11-10 — P1-1 review fixes done; P1-3 analysis committed (hash aliasing: ~58 % spurious candidates; C1/C2 bitwise); P1-3a started in worktree `RoboticCuttingSim-rl-p13`.
- 2026-11-10 — P1-1b S0 PASS (CPU bitwise); P1-1 review fixes started; S1 robot-only GPU on GPU0.
- 2026-11-10 — P1-1 core done (bitwise CPU parity/isolation/reset; GPU smoke PASS); vector env + milestones started; P1-1 xhigh review started.
- 2026-11-10 — P1-0 PASS; ADR 0022 accepted (D-10); pytest pythonpath fix (45ecf92).
- 2026-11-10 — ADR 0021 accepted with amendments; D-09 pytest overlay for the Torch venv; P1-1/P1-2 implementation started.
- 2026-11-10 — `RL_READY_PLAN.md` committed (decisions D-01..D-07); D-08 headless runtime venv.
- 2026-11-10 — specification copy committed (plan + brief, naming-rule substitution only).
