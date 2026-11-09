# RL-Ready Simulator & Rubber Physics — progress

Specification: [RL_READY_SPECIFICATION.md](RL_READY_SPECIFICATION.md) ·
plan, acceptance and decisions: [RL_READY_PLAN.md](RL_READY_PLAN.md) ·
reports: `docs/reports/rl_ready/` · compact evidence: `docs/evidence/rl_ready/` ·
raw artifacts (git-ignored): `artifacts/rl_ready/`.

## 進行中

- Work package: P1-0 (baseline) / P1-1 design. Started 2026-11-10 from origin/main
  `0c2b17e` (PR #11, WP-17) on branch `claude/rl-ready-rubber-physics`.
- Running: xhigh architecture study → `docs/adr/0021-batched-rl-gpu-headless-runtime.md`
  (read-only analysis, no GPU).
- Next: `docs/RL_READY_PLAN.md`; P1-0 environment inventory + benchmark runner;
  then P1-1 batched TLMPM v7_2 (CPU parity first), P1-2 vector env.

## 待使用者決定

- (none yet) Real rubber measurements are not available on this machine; Phase 2
  calibration status will stay `SIMULATOR_ONLY_UNCALIBRATED` unless data is provided.

## Log

- 2026-11-10 — specification copy committed (plan + brief, naming-rule substitution only).
