# Unattended session rules (RoboticCuttingSim, updated 2026-10-30 for WP-12)

You are the long-running Claude Code CLI session in tmux `rcs-demos:claude`. The
user is not watching continuously. Work through the current plan named in
`tools/demo_runner/goal.txt` (since 2026-10-30: WP-12, specification
[`docs/WP12_SPECIFICATION.md`](../../docs/WP12_SPECIFICATION.md), plan
`docs/WP12_PLAN.md` once written) until it is complete. The BladeMaster demos
(final_v4) and WP-11 are finished and frozen as baselines. A supervisor
(`tools/demo_runner/runner.sh`) resumes you if the CLI exits, and sends「繼續」
when you are idle or after a usage limit.

## Starting or resuming

1. Read `AGENTS.md`, `docs/WP12_SPECIFICATION.md`, `docs/WP12_PLAN.md` (if it
   exists) and the top of `docs/PROGRESS.md`. Read the WP-11 plan, reports and
   ADRs listed in the specification §0.1 when you need history.
2. Check `git status`, `git log -5`, `nvidia-smi`, `df -h / /mnt/data` and any
   of your background jobs. Do not redo completed work, and do not rerun an
   experiment that already has a complete result.
3. Keep a short「進行中」section at the top of `docs/PROGRESS.md`: current
   phase, the running jobs with their artifact paths, the next step. Update it
   before you start long jobs, so a resumed session knows where things stand.

## How to work

- Effort: everything runs at high, per the user's 2026-10-30 instruction
  (usage budget): you, `rcs-impl` (implementation, tests, probes, Isaac runs)
  and `rcs-review` (design review, acceptance decisions, final review). Give
  subagents a precise brief. Run independent subagents in parallel when they
  don't share GPUs or files.
- Don't ask the user questions. Make the decision, record it in
  `docs/adr/` (next free number) or in the plan, and continue.
- Commit and push to `claude/wp12-physical-cut-surfaces` after every
  meaningful step (a passing test, a probe result, a fixed bug, a finished
  report). Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never leave large
  uncommitted work. Never commit `artifacts/` (it is git-ignored); compact
  evidence JSON goes under `docs/evidence/`, and media goes under
  `docs/media/` (keep each file small: GIF/MP4 a few MB).
- Long GPU or Isaac jobs: launch them in the background with logs under
  `artifacts/wp12/logs/`, then wait with a background until-loop
  (Monitor / run_in_background). Don't poll in a tight loop. While waiting,
  do other independent work, or end the turn; the job notification wakes you.
- When something fails, diagnose it with data, record the failure in a report,
  and try the next hypothesis. Keep every failed output.
- If you are truly blocked on something only the user can decide, write it in
  `docs/PROGRESS.md` under「待使用者決定」, then keep working on everything
  else.

## Naming rule

The repository (code, docs, configs, commit messages, README) must not contain
the acronym of the user's separate research project (guarded by
`tests/test_repo_naming.py`). When a document needs to mention future users of the
environment, call them agentic control (or policies / learning systems). Phase 0
removes existing mentions from tracked text files and adds a CPU test that
guards against new ones.

## Integrity (non-negotiable)

- Follow `AGENTS.md` and the Engineering Principles in the WP-12
  specification. Frozen v1 physics, thresholds and records stay unchanged, and
  so do `tlmpm.py`, `tlmpm_v2.py`, `cutting_tasks_v1.json`, the v2 task
  thresholds, `final_v4`, law 8 and the WP-11 frozen results. New laws,
  surface profiles, workcells and evaluation protocols are versioned and
  opt-in.
- Never present geometric deletion, presegmented parts, renderer masks, ideal
  target planes, visual-only particle edits, direct particle-velocity or
  component-pose edits, or kinematic/replay results as live robot cutting or
  as a physics improvement.
- Report PASS / FAIL / PARTIAL / NOT_RUN / NOT_APPLICABLE separately. Demo
  acceptance videos must come from live Native Isaac Sim. Predeclare
  benchmark seeds, ranges and thresholds before running them.
- GPU: don't stop processes you didn't start. Prefer GPU 1. Disk is tight
  (root ~97%, /mnt/data ~95%): outputs go to `artifacts/wp12/` (on /mnt/data),
  `TMPDIR=/mnt/data/ningan/tmp`; remove git worktrees and scratch output you
  created once their results are recorded.

## Finishing

When every phase of the current plan is done and reviewed (`rcs-review`):

1. Update the README with the new workcell images, Demo 1–7 videos/GIFs and
   the ParticleSurface vs cut-aware comparison, plus provenance.
2. Update `docs/PROGRESS.md` and the reports, and tidy the docs index.
3. Run the full CPU suite and confirm CI is green.
4. Merge the branch into `main` and push, as the user explicitly requested.
5. Then `touch /mnt/data/ningan/lab_records/rcs_demo_runner/DONE`.

Stop after DONE.
