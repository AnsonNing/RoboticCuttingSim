# RL-Ready Simulator & Rubber Physics — progress

Specification: [RL_READY_SPECIFICATION.md](RL_READY_SPECIFICATION.md) ·
plan, acceptance and decisions: [RL_READY_PLAN.md](RL_READY_PLAN.md) ·
reports: `docs/reports/rl_ready/` · compact evidence: `docs/evidence/rl_ready/` ·
raw artifacts (git-ignored): `artifacts/rl_ready/`.

## 進行中

- Work package: P1-0 (baseline) / P1-1 design. Started 2026-11-10 from origin/main
  `0c2b17e` (PR #11, WP-17) on branch `claude/rl-ready-rubber-physics`.
- Running (2026-11-10): (1) xhigh architecture study → `docs/adr/0021-batched-rl-gpu-headless-runtime.md`
  (read-only, no GPU); (2) high implementation of P1-0: inventory, `robotic_cutting_sim.rl`
  telemetry/benchmark rows, `configs/rl_benchmark/workloads_v1.json`, WP-17 measured-trajectory
  replay + single_env_headless baseline (GPU1 under `rcs_rl_runner/gpu1.lock`, raw output
  `artifacts/rl_ready/p1_0/`, logs `artifacts/rl_ready/logs/`).
- Next: review ADR 0021 → P1-1 batched TLMPM v7_2 (CPU parity first) → P1-2 vector env (M0/M1/M2).

## 待使用者決定

- (none yet) Real rubber measurements are not available on this machine; Phase 2
  calibration status will stay `SIMULATOR_ONLY_UNCALIBRATED` unless data is provided.

## Log

- 2026-11-10 — `RL_READY_PLAN.md` committed (decisions D-01..D-07); D-08 headless runtime venv.
- 2026-11-10 — specification copy committed (plan + brief, naming-rule substitution only).
