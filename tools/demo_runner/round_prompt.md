# Unattended demo session rules (RoboticCuttingSim, 2026-10-29)

You are the long-running Claude Code CLI session in tmux `rcs-demos:claude`. The
user is not watching continuously. Work through
[`docs/BLADEMASTER_DEMO_PLAN.md`](../../docs/BLADEMASTER_DEMO_PLAN.md) until it
is complete. A supervisor (`tools/demo_runner/runner.sh`) resumes you if the
CLI exits, and sends「繼續」when you are idle or after a usage limit.

## Starting or resuming

1. Read `AGENTS.md`, `docs/BLADEMASTER_DEMO_PLAN.md`,
   `docs/reports/WP-10_CLAUDE_DIAGNOSIS_20261029.md` and the top of
   `docs/PROGRESS.md`. Read `docs/HANDOFF_WP10_CLAUDE_20261029.md` and
   the WP-10 reports when you need history.
2. Check `git status`, `git log -5`, `nvidia-smi`, `df -h / /mnt/data` and any
   of your background jobs. Do not redo completed work, and do not rerun an
   experiment that already has a complete result.
3. Keep a short「進行中」section at the top of `docs/PROGRESS.md`: current
   phase, the running jobs with their artifact paths, the next step. Update it
   before you start long jobs, so a resumed session knows where things stand.

## How to work

- Effort: planning, physics-law design, acceptance decisions and reviews at
  xhigh (you, or the `rcs-review` subagent). Implementation such as code,
  tests, probes and Isaac runs goes to the `rcs-impl` subagent at high, with a
  precise brief. Run independent subagents in parallel when they don't share
  GPUs or files.
- Don't ask the user questions. Make the decision, record it in
  `docs/adr/` (next number 0006) or in the plan, and continue.
- Commit and push to `claude/lucid-mendel-vt7n82` after every meaningful step
  (a passing test, a probe result, a fixed bug, a finished report). Commit
  messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Never leave large uncommitted work. Never commit `artifacts/` (it is
  git-ignored); compact evidence JSON goes under `docs/evidence/`, and media
  goes under `docs/media/demos/` (keep each file small: GIF/MP4 a few MB).
- Long GPU or Isaac jobs: launch them in the background with logs under
  `artifacts/wp10_claude/logs/`, then wait with a background until-loop
  (Monitor / run_in_background). Don't poll in a tight loop. While waiting,
  do other independent work, or end the turn; the job notification wakes you.
- Apply the stop condition in the plan: Phase 1 (1 mm cutting matrix including
  the negative controls) must pass before repeated live robot demo attempts.
- When something fails, diagnose it with data (`scripts/analyze_cut_bridges.py`,
  `scripts/replay_measured_edge.py`, receipts), record the failure in a report,
  and try the next hypothesis. Keep every failed output.
- If you are truly blocked on something only the user can decide, write it in
  `docs/PROGRESS.md` under「待使用者決定」, then keep working on everything
  else.

## Integrity (non-negotiable)

- Follow `AGENTS.md`. Frozen v1 physics, thresholds and records stay
  unchanged, and so do `tlmpm.py`, `tlmpm_v2.py`, `cutting_tasks_v1.json` and
  the v2 task thresholds. New laws, tasks and scenarios are versioned and
  opt-in.
- Never present geometric deletion, presegmented parts, renderer masks,
  direct particle-velocity or component-pose edits, or kinematic/replay
  results as live robot cutting.
- Report PASS / FAIL / BLOCKED / NOT_RUN separately. Demo acceptance videos
  must come from live Native Isaac Sim.
- GPU: don't stop processes you didn't start. Prefer GPU 1. The root disk is
  nearly full: outputs go to `artifacts/wp10_claude/` (on /mnt/data), and
  `TMPDIR=/mnt/data/ningan/tmp`.

## Finishing

When all P0 work packages in the plan are done and reviewed (`rcs-review`):

1. Update the README with the three demo videos/GIFs and new scene images,
   plus provenance.
2. Update `docs/PROGRESS.md` and the reports, and tidy the docs index.
3. Run the full CPU suite.
4. Merge the branch into `main` and push, as the user explicitly requested.
5. Then `touch /mnt/data/ningan/lab_records/rcs_demo_runner/DONE`.

P1/P2 items may continue after that only if the plan says so. Otherwise stop
after DONE.
