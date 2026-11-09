# RL-Ready Simulator & Rubber Physics — progress

Specification: [RL_READY_SPECIFICATION.md](RL_READY_SPECIFICATION.md) ·
plan, acceptance and decisions: [RL_READY_PLAN.md](RL_READY_PLAN.md) ·
reports: `docs/reports/rl_ready/` · compact evidence: `docs/evidence/rl_ready/` ·
raw artifacts (git-ignored): `artifacts/rl_ready/`.

## 進行中

- Work packages: P1-0 (baseline, running), P1-1 core + P1-2 components (running). Branch
  `claude/rl-ready-rubber-physics` from origin/main `0c2b17e`.
- Design: ADR 0021 accepted with amendments (`docs/adr/0021-batched-rl-gpu-headless-runtime.md`).
- Running agents (2026-11-10):
  1. P1-0 baseline (high): WP-17 measured-trajectory replay, single_env_headless rows, kernel split,
     ≥3 replays/seed for the L3 band — GPU1 under `/mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock`,
     raw `artifacts/rl_ready/p1_0/`, logs `artifacts/rl_ready/logs/`.
  2. P1-1 core (high): `robotic_cutting_sim.batched` layout / `tlmpm_v7_2_batch` / `material_batch`,
     CPU bitwise L0/L1 tests, then a short GPU smoke.
  3. P1-2 components (high): `batched/pose_interp.py`, `rl/{actions,observations,reward_packed}.py` + tests.
- Next: integrate `rl/vector_env.py` (`open_vector_env`) → M0 (N=1) / M1 (N=4) / M2 (N=8–16) GPU runs →
  xhigh review of P1-1/P1-2 → P1-1b robot batching ADR.

## 待使用者決定

- (none yet) Real rubber measurements are not available on this machine; Phase 2
  calibration status will stay `SIMULATOR_ONLY_UNCALIBRATED` unless data is provided.

## Log

- 2026-11-10 — ADR 0021 accepted with amendments; D-09 pytest overlay for the Torch venv; P1-1/P1-2 implementation started.
- 2026-11-10 — `RL_READY_PLAN.md` committed (decisions D-01..D-07); D-08 headless runtime venv.
- 2026-11-10 — specification copy committed (plan + brief, naming-rule substitution only).
