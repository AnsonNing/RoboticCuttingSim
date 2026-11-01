# WP-14 — v7 refinement and scoped adoption

IN_PROGRESS: Phase A frozen; no physics tuning, 2026-11-01 Asia/Seoul.
Branch `codex/wp14-v7-scoped-adoption`, refreshed starting `origin/main`
`98c42c272f02c4438de6867663448f222aaf5fdc`.

Use the corrected compiled Planner before changing physics. Keep v7 grid-consistent
kinematics and v5 + Law 8 as the default/fallback. Preserve WP-12/WP-13 sources,
frozen protocols, raw failures, scene/goal/API/camera contracts and Native media.
Acceptance is independent per task family; verified CUT research adoption does
not require TRIM adoption. No smoothing, deletion, permissive labels or hidden fallback.

| Phase | Status | Decisive next work |
| --- | --- | --- |
| A Correct Planner revalidation | IN_PROGRESS | `configs/wp14/core_v1.json`: 14 episodes frozen before outcomes; real-driver compiled override tests 4 PASS |
| B Coupled front/contact | DIAGNOSIS_RUNNING | New T00 v7 remains one body; read-only selected-substep trace on idle cuda:0, original core remains cuda:1 |
| C Scoped profile validation | NOT_RUN | Separate physical integrity, surface quality and numerical/contact limits |
| D Held-out / Native | NOT_RUN | Freeze envelope and 4–6 held-out pairs; 2–3 Native pairs for proposed supported families |
| E Documentation / release | NOT_RUN | Explicit selectable profiles, capability table, real media, review, PR/main CI and remote verification |

No new physics result or adoption is claimed yet. User-provided full plan and
durable interactive logs: ignored `artifacts/wp14/runner_20261101/`.
Tmux `rcs-wp14-codex` accepts direct user input. Implementation/testing use high;
planning/complex analysis/review may use xhigh. There is no inherited WP-13 deadline.

Core exact command (new output only):

```bash
PYTHONPATH="$PWD/src" .venv/bin/python scripts/run_wp14_queue.py \
  --protocol configs/wp14/core_v1.json --root artifacts/wp14/core_001
```

Both RTX 5090 GPUs idle at preflight; driver 570.211.01; 30 GiB home /
25 GiB data free. Kinematic pairs use cuda:1 sequentially. Historical protocols
and solver sources are unchanged. Partial uses -12 mm overtravel versus the
recorded prior -10 mm; execution audit will compare actual trajectories.

Held-out declaration before outcomes: `configs/wp14/heldout_v1.json` fixes four
new CUT pairs (8 kinematic episodes) and two Native pairs (4 episodes). Proposed
support is single straight severing, discrete reference/firm material recipes,
120–160 x 24–32 x 16–20 mm continuous bars, ±6 mm cut locations and ±8° workpiece
yaw on a matching solid board. Both blade-line directions are tested. This is
a proposal subject to all failures, not an adopted envelope. TRIM and Multi-cut
are assessed separately; they are not required for CUT-only adoption. The new
held-out configurations have only compiled; physical outcomes are NOT_RUN.

First three completed core pairs: C00 and C01 physically PASS both arms with
identical actual commands and compiled replay PASS. Candidate x/F P95 gains
6.05x / 8.49x; local locus roughness reductions 85.94% / 78.48%. New T00 v5
PASS / v7 FAIL (one mechanical body, fixture overlap 0.467 mm). Quality gains
do not change that failed separation. Remaining eight core episodes pending.
Six current-source CPU physical negatives PASS (11.20 s); six replay/trace
checks PASS (5.32 s), including finite moving-link sweep and bitwise unchanged
solver input arrays after observation. No physical solver source was modified.
