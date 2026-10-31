# WP-13 — Integrated cutting physics

Status: IN_PROGRESS, setup only (2026-11-01 Asia/Seoul).
Branch: `codex/wp13-integrated-cutting`; refreshed starting `origin/main`:
`11895016b34ed821be2bdecbc5351667fb62d47d`.

The supplied WP-13 plan governs this work. Couple material motion, finite causal
cut front and knife/board contact within the same substep. Preserve v5 + Law 8,
v6's rejected results, frozen protocols and the scene/goal/API/camera contracts.
No plane smoothing, hidden fracture, goal changes or historical evidence edits.

| Phase | Status | Next decisive evidence |
| --- | --- | --- |
| A Baseline/instrumentation | IN_PROGRESS | Read existing diagnostics; trace the v6 FB_CUT 12-corner bridge |
| B Coupled kinematics/front | NOT_RUN | CUT/TRIM separate; PARTIAL and negative controls remain connected |
| C Contact | NOT_RUN | Same-geometry dt pair, normal/flank/board impulse, work and penetration |
| D Honest surfaces | NOT_RUN | Unchanged surface profiles on improved state before any renderer change |
| E Focused generalization | NOT_RUN | Precommitted seeds/configs, 24–36 kinematic and 6–8 Native episodes |
| F Publication | NOT_RUN | Compact results, animated real Native gallery, PR merge and exact-main CI/media |

Planning/complex analysis/review may use xhigh; implementation/testing use high.
Execution is durable in tmux `rcs-wp13-codex`; logs and the full supplied plan are
local under ignored `artifacts/wp13/runner_20261101/`. Commands/versions and
measurements will be recorded in `reports/WP13_RESULTS.md` and compact evidence.
An unaccepted solver remains opt-in, with PARTIAL/REJECTED reported explicitly.
