# WP-13 — Integrated cutting physics

Status: IN_PROGRESS, candidate REJECTED FOR PROMOTION after primary TRIM failure;
continue the authorized evidence/publication path (2026-11-01 Asia/Seoul).
Branch: `codex/wp13-integrated-cutting`; refreshed starting `origin/main`:
`11895016b34ed821be2bdecbc5351667fb62d47d`.

The supplied WP-13 plan governs this work. Couple material motion, finite causal
cut front and knife/board contact within the same substep. Preserve v5 + Law 8,
v6's rejected results, frozen protocols and the scene/goal/API/camera contracts.
No plane smoothing, hidden fracture, goal changes or historical evidence edits.

| Phase | Status | Next decisive evidence |
| --- | --- | --- |
| A Baseline/instrumentation | PASS (diagnosis) | v6 failure reproduced; traced corners genuinely remain ahead of the finite edge by 8–32 micrometres |
| B Coupled kinematics/front | PARTIAL / promotion REJECTED | Final CUT and PARTIAL correct; primary TRIM remains one component while v5 separates |
| C Contact | PARTIAL | Velocity-level law and 1 mm recontact measured; dt pair pending, no convergence claim |
| D Honest surfaces | MEASURED / PARTIAL | 15 saved states × two unchanged profiles; one during-state NOT_RUN; no renderer promotion |
| E Focused generalization | IN_PROGRESS / protocol deviation | `configs/wp13/focused_v1.json`: 32 paired kinematic and 6 Native episodes |
| F Publication | IN_PROGRESS | Honest rejection report + functioning v5 gallery; reviewed PR and exact-main CI/media pending |

Planning/complex analysis/review may use xhigh; implementation/testing use high.
Execution is durable in tmux `rcs-wp13-codex`; logs and the full supplied plan are
local under ignored `artifacts/wp13/runner_20261101/`. Commands/versions and
measurements will be recorded in `reports/WP13_RESULTS.md` and compact evidence.
An unaccepted solver remains opt-in, with PARTIAL/REJECTED reported explicitly.

Independent review confirmed dropped compiled planner overrides in the kinematic
driver; P01 duplicates P00 depth. Actual paired paths are retained and reported;
no claimed partial-depth diversity. Finish the frozen batch before repairing the
driver, to preserve dt comparisons. Candidate remains rejected.
