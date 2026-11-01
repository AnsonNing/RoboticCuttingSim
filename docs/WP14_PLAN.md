# WP-14 — v7 refinement and scoped adoption

IN_PROGRESS: setup only, 2026-11-01 Asia/Seoul.
Branch `codex/wp14-v7-scoped-adoption`, refreshed starting `origin/main`
`98c42c272f02c4438de6867663448f222aaf5fdc`.

Use the corrected compiled Planner before changing physics. Keep v7 grid-consistent
kinematics and v5 + Law 8 as the default/fallback. Preserve WP-12/WP-13 sources,
frozen protocols, raw failures, scene/goal/API/camera contracts and Native media.
Acceptance is independent per task family; verified CUT research adoption does
not require TRIM adoption. No smoothing, deletion, permissive labels or hidden fallback.

| Phase | Status | Decisive next work |
| --- | --- | --- |
| A Correct Planner revalidation | NOT_RUN | Commit six paired cases + a different-depth Partial pair; verify actual compiled paths |
| B Coupled front/contact | NOT_RUN | Trace residual bridges first; at most one candidate and one diagnosed correction |
| C Scoped profile validation | NOT_RUN | Separate physical integrity, surface quality and numerical/contact limits |
| D Held-out / Native | NOT_RUN | Freeze envelope and 4–6 held-out pairs; 2–3 Native pairs for proposed supported families |
| E Documentation / release | NOT_RUN | Explicit selectable profiles, capability table, real media, review, PR/main CI and remote verification |

No new physics result or adoption is claimed yet. User-provided full plan and
durable interactive logs: ignored `artifacts/wp14/runner_20261101/`.
Tmux `rcs-wp14-codex` accepts direct user input. Implementation/testing use high;
planning/complex analysis/review may use xhigh. There is no inherited WP-13 deadline.
