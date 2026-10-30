# WP-12 — Physically Consistent Cut Surfaces, BladeMaster Workcell & Generalized Cutting

**Source:** the user's WP-12 specification (2026-10-30),
[`WP12_SPECIFICATION.md`](WP12_SPECIFICATION.md). Reference images (local, not
in the repository): `/home/ningan/b.png` (target scene: one whole cutting board
lying on the table, workpiece on the board, Franka holding a long straight
knife) and `/home/ningan/a.png` (knife appearance: flat light blade gripped at
one end, wooden board).

**Branch:** `claude/wp12-physical-cut-surfaces`, cut from main `fbe99bd`.
**Frozen baselines:** `final_v4`, law 8 (ADR 0012) and its acceptance records,
WP-11 frozen results, task identities and all historical evidence. Every new
law, surface profile, workcell profile and evaluation protocol is versioned
and opt-in. Material parameters remain provisional and uncalibrated.

**Effort:** all work at high (orchestrator, `rcs-impl`, `rcs-review`), except Phase 4 decisions, root-cause analysis, law-9 design, acceptance verdicts, orientation envelope and resolution-study conclusions, which run at xhigh (`rcs-xhigh`, or `rcs-review` launched with effort xhigh) per the user's 2026-10-31 instruction; Phase 4 implementation and runs stay with `rcs-impl` at high. The Phase 4a root-cause draft and its "law 9 for class (a) only" recommendation are reviewed at xhigh before any law-9 implementation.
**Naming:** the repository does not name the user's separate project
(`tests/test_repo_naming.py`); future users are called agentic control.

## Guiding rules (from the specification's Engineering Principles)

1. Physics first. A visible cut face may exist only where the solver has
   assigned conflicting side digits. No target plane, no mesh mask, no
   visual-only particle edit, no evaluator change, no seed-specific patch.
2. Keep three things apart in every report: physical discontinuity (conflicting
   digits on a lattice link), global separation (mechanical components), and
   surface geometry (what is drawn).
3. Reuse `rcs_scene_v1`, `rcs_goal_v1`, `ConfiguredTaskSpec`, `ToolProfile`,
   `tlmpm_v5`, `cut_topology`, `CuttingSurface`, `PolicyReadyEnv`,
   `goal_eval_v2a` and the Native runtime. No rewrite of TLMPM.
4. Every important change answers: failure cause, candidates, choice and why,
   does it change physics, improvement vs baseline, new failure modes.
5. Thresholds and benchmark seeds are declared (committed) before the run that
   uses them.

## Phases, deliverables and gates

| Phase | Work | Deliverables | Gate |
| --- | --- | --- | --- |
| 0 | Remove the other project's acronym + guard test; audit; baseline freeze (8 scenes); surface technology review | `tests/test_repo_naming.py`, this plan, `reports/WP12_BASELINE_AUDIT.md`, `reports/WP12_SURFACE_TECH_REVIEW.md`, `evidence/wp12/phase0_*` | Research conclusion, architecture and metric method fixed |
| 1 | Cut-surface diagnosis: attribute unevenness to particle physics / labels / fragmentation / reconstruction / blade path; multi-layer debug viewer; G1–G8; metrics of spec §1.4 in reference and current space | `surface_metrics_v1`, `cut_debug_viewer`, `reports/WP12_SURFACE_DIAGNOSIS.md` (feeds the surface report), `evidence/wp12/phase1_*` | The five questions of spec §1.4 Phase 1 Gate answered with numbers |
| 2 | `blade_master_flat_board_v1` workcell: one wooden board directly on the table, no risers; collision/contact consistency; through-cut vs solid board study (options A–D); layout, HOME, cameras; workcell acceptance | `scene_config` profile, Native scene, `reports/WP12_WORKCELL.md`, ADR 0017 | Board contact real, no hidden supports, no knife–board penetration, acceptance list of spec §2.6 reported |
| 3 | `cut_aware_surface_v1`: Hermite cut-event recording (read-only, no physics change), reference-lattice linked volume derived from side codes, local dual contouring with per-side duplicated vertices, cut-side-aware embedding, USD path; S1–S9; profiling | `cut_events_v1`, `cut_aware_surface.py`, `reports/WP12_SURFACE_RECONSTRUCTION.md`, ADR 0018 | Not worse than `particle_surface_v1` on every frozen metric and better on cut-face error; no physics change (bitwise CPU check) |
| 4 | Law 8 failure regions (30–35° oblique, 45° rake, residual bridges, thin-chip residue): per-segment diagnostics, root causes, versioned law 9 only if the mechanism requires it with full law-8 acceptance rerun; orientation envelope; workpiece and material generalization; resolution study | `reports/WP12_CUTTING_PHYSICS.md`, `reports/WP12_RESOLUTION_STUDY.md`, ADR 0019 if law 9 | Each failure class has a root cause; envelope declared; convergence evidence or explicit non-convergence |
| 5 | Predeclared generalization benchmark: families A–H, ID/OOD, feasibility classes, held-out seeds, Native stratified sample | `configs/wp12_benchmark_v1.json` (committed before running), `reports/WP12_GENERALIZATION.md` | Full matrix with denominators, failure classes, OOD boundary |
| 6 | Agentic-control readiness: API, observations (no privileged leaks), actions, recording incl. surface provenance, repeated episodes, profiling | `reports/WP12_PERFORMANCE.md`, tests | Spec §6.7 checklist |
| 7 | Full regression, Native Demos 1–7 with videos/close-ups, visual comparisons, acceptance matrix; review; README/PROGRESS; merge | `reports/WP12_FINAL_VALIDATION.md`, `docs/media/wp12/` | Acceptance matrix with PASS/FAIL/PARTIAL/NOT_RUN/NOT_APPLICABLE per row |

## Architecture decisions (initial; refined by the reports and ADRs)

- **Surface (Phase 3).** Primary candidate: Approach B/C. The reference grid is
  the solver's own regular reference particle lattice (1 mm). A lattice link is
  cut iff its endpoint codes conflict (`cut_topology` semantics), so the
  visual topology is derived from TLMPM side codes and can never be a second
  source of truth. Hermite data (intersection parameter along the link, blade
  normal, cut id) come from the blade pose at the moment the solver assigned
  the digit, captured by a read-only recorder in an opt-in solver variant;
  physics must stay bitwise identical on CPU. Faces are produced by local dual
  contouring around cut links with one representative per material side, and
  vertices are embedded in the same-side particle motion field. The original
  outer surface keeps ParticleSurface or is contoured on the same lattice
  (hybrid) — decided by Phase 1 data. Details: tech review + ADR 0018.
- **Workcell (Phase 2).** New `blade_master_flat_board_v1` profile next to the
  kept `wp11_workcell_v1`; the TLMPM fixture plane, PhysX colliders, visual
  mesh, IK and preflight are generated from the same board record. The knife
  meets the board through real PhysX contact; whether a through-cut is
  achieved with the edge at the board surface (option A/B) is measured, not
  assumed. A slotted board stays an explicit option, never a hidden default.
- **Physics (Phase 4).** Law 8 stays frozen. A law 9 is introduced only if a
  diagnosed mechanism (not a threshold) blocks valid cuts, and must pass the
  complete law-8 acceptance matrix and blunt-contact negative controls.

## Execution notes

- GPU: prefer `cuda:1`; Native/Isaac runs one at a time. Outputs in
  `artifacts/wp12/` (git-ignored), compact evidence in `docs/evidence/wp12/`,
  media in `docs/media/wp12/`.
- Parallel agents share the git index: only the orchestrator commits, using
  `git commit -- <paths>`.

## Status

| Phase | State | Commits / evidence |
| --- | --- | --- |
| 0 | done: acronym removal + guard (`ce547f9`), plan (`1c91605`), technology review (`3f13379`), baseline audit: 8/8 scenes reproduce frozen verdicts, CPU 1196 passed (`2df1e13`) | [audit](reports/WP12_BASELINE_AUDIT.md), [review](reports/WP12_SURFACE_TECH_REVIEW.md) |
| 1 | done (`b31553d`): attribution, G1–G8, debug viewer; surface protocol v1 frozen | [diagnosis](reports/WP12_SURFACE_DIAGNOSIS.md) |
| 2 | 2a done (`ec3e8a7`): profile, single-source geometry, through-cut study (A+B), Native settle and knife-on-board; 2b in progress: live Native cutting acceptance on the board, old-vs-new comparison | [workcell report](reports/WP12_WORKCELL.md), [ADR 0017](adr/0017-blade-master-flat-board-workcell.md) |
| 3 | recorder (`403870f`) and Layers 3–6 core (`89b289d`) done; S0–S9 measurement running; live/Native integration next | |
| 4 | 4a root-cause diagnosis running | |
| 5–7 | not started | |

**Phase 0 gate decision (2026-10-31).** Architecture: Approach B on the 1 mm particle
lattice (review §6), links cut iff endpoint codes conflict, Hermite data from
`cut_events_v1`, per-side representatives, F-corrected same-side embedding;
`particle_surface_v1` kept as baseline and fallback; hybrid only if Phase 1/3 data
require it. Metric method: review §7 (reference, assignment-instant and current
space; never the target plane alone). Thresholds frozen in
`configs/wp12/surface_protocol_v1.json` before any Phase 3 measurement.
