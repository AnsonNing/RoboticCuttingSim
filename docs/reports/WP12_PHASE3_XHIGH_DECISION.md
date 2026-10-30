# WP-12 Phase 3 decision (xhigh): close with `cut_aware_surface_v1` as the opt-in profile

**Date** 2026-10-31. **Branch** `claude/wp12-physical-cut-surfaces`, HEAD `dc210ff`. **Scope** Phase 3 disposition after
`phase3_surface_001` (protocol v1) and `phase3_surface_002` (B v2, protocol v2, saved/H1/H2). Judged against spec Phase 3
(§3.1–3.7, gate), the Completion Criteria, AGENTS.md, [`WP12_PLAN.md`](../WP12_PLAN.md), the tech review §6–7 and the
earlier [decision review](WP12_PHASE3_DECISION_REVIEW.md). Nothing was committed; no code, protocol, threshold or run
directory was changed. New numbers come from read-only scripts outside the repository (listed at the end); fold them
into a repository diagnostic before citing them as evidence (§6 step 1).

## Decision

**CLOSE Phase 3. Do not run a v3 now.** The Phase 3 comparison gate is **FAIL** under the frozen `surface_protocol_v1`
and `surface_protocol_v2`; both FAILs are kept forever. `cut_aware_surface_v1` (B v1) is the shipped **opt-in** profile;
`particle_surface_v1` stays the default; `cut_aware_surface_v2` stays a labelled opt-in experiment, never used in demos;
neither M2 nor any M-subset is promoted to a method.

Why: the remaining failures share one upstream cause, a solver-state inconsistency at the cut face (F3/F7), which belongs
to Phase 4; the v2 gate has a defect that fails any method by construction (D1, §5); a v3 now costs 1–2 days and is
predicted to FAIL on every set (§3). Reopen only under §3.3. This is the Completion Criteria route: keep the working
improvement, record failure causes and follow-up, do not lower the standard.

## 1. Re-check of the earlier review (F1–F4) and of the M4 attribution

| # | Earlier conclusion | Status after the v2 run and this check |
|---|---|---|
| F1 | S6 `B_WORSE` was a region-rule artefact | **Holds.** Under P1, B v1 S6 current P95 0.381 vs A 0.424 (NOT_DIFFERENT) on saved; B_BETTER on H1 (0.38 / 0.45) and H2 (0.356 / 0.55). The review's 0.345 excluded vertices; the P1 value reassigns them and is authoritative |
| F2 | S4 tip has zero hinge weight; M2 should fix it | **Holds where measured.** No-M4 ablation: M2 moves 25/25 regions; current P95 0.96 → 0.44 (saved; P3 excess exactly 0.00) and 0.88 → 0.45 (H1). **Not uniformly better:** H2 (M2 on, M4 off) matched normal 9.9° → 15.7°. **Untested:** the ablation's G4 hold is NOT_RUN on all sets (`eval_*/S4_r*.json` → `g4_hold_phase`) |
| F3 | det F / position disagreement localised at the cut; solver state | **Holds, sharpened.** Uncut links, \|dx − F̄dX\|: P50 0.07–0.71 s and P99 1.3–2.5 s (S7 7.5–11.4) in the first layer (≤ 1 s from a Hermite point); P50 0.001–0.026 s beyond 6 s. det F at flagged endpoints 0.95–1.07. P4 INCONSISTENT on every state except G3a and S4, all three sets. Read "collapse" as **particle-scale disorder**: links stretch in the face plane while cells shrink |
| F4 | v1 triangle-normal rows unmeetable by a faithful mesh | **Holds.** Under P2, B v1 beats A in 7/7 comparable saved cases: G3a 30 vs 49°, G3b 8.4 vs 14.7, S2 5.9 vs 15.7, S3 6.0 vs 12.3, S5 6.9 vs 10.1, S6 5.8 vs 8.2, S8 5.9 vs 6.3. Still unexplained: G3a ≈ 30° |

**M4 attribution (R.5 a).** That M4 causes the B v2 regression is **right**: the no-M4 ablation reproduces B v1 everywhere
except S4 (M2) and some extra M3 folds (S3 27 → 46, G3b 85 → 90). The stated *location* is incomplete. Final frames, r0;
flagged = length > 2 s and \|dx − F̄dX\| > s, as `cut_aware_surface_v2.py:229`:

| State | Flagged | ≤ 1 s from the cut locus | Link direction | Endpoint codes | Pieces split off when flagged links are removed |
|---|---|---|---|---|---|
| saved G3a / G3b / S6 | 140 / 27 / 2 | all (0.50–0.71 s: the face layer) | in the face plane (140 / 26 / 2) | identical code | 0 |
| saved S4 | 48 | all | in plane, along the crack front | code 0 vs digit, all at z = 6 mm (hinge row) | 0 |
| saved S3 / S5 | 29 / 97 | 22 / 67 | mostly in plane | S5: 61 same, 36 mixed | 2 single particles each |
| H2 S5 (75° crossing) | 621 | 232; 238 at 3–6 s | 271 along N | 393 same, 228 mixed | 0 |
| H2 S6 (20 mm pitch) | 595 | 8; **581 at 3–6 s** | 593 along N; 301 with an exterior endpoint | 444 same, 151 mixed | 0 |
| saved / H2 S7 chip | 134 / 167 | 106 / 92 | mixed | all code 2 | **48 / 35 pieces (80 / 88 particles)** |

Corrections to R.5 (a): "every through-cut" is overstated (S2 and S8 flag 0 on every set); in S4, M4 removes exactly the
hinge links M2 needs, so the two fixes conflict by construction; the largest regressions (H2 S5 and S6, 1,107 and 1,140
cells) come mostly from links 3–6 s *inside* the slices, along N, not from the face layer. The common property is
position/F inconsistency *without* topological separation. No per-link (length, inconsistency) pair separates it from a
tear, because the S7 chip links overlap in the low mode (P50 2.9–4.8 s); only topology does. Removing the flagged links
splits the chip into 35–48 pieces and splits off nothing in any through-cut.

## 2. F3 and F7 are solver state for Phase 4, not a measurement artefact

- **Not a metric artefact.** The per-link reading uses no region rule, locus, plane fit or mesh; it agrees with P4's cell
  V/J; it is ~0 in the far field and in the Phase 1 affine controls. F7 is seen directly: same-code neighbours 1 mm apart
  in reference sit up to 19.8 mm apart with F ≈ I. Native TRIM shows the same for both profiles' physics
  (`WP12_SURFACE_LIVE.md` §3: 83–84 uncut links > 3 mm, up to 20.5 mm).
- **Physical consequence.** Contact uses particle positions; stress uses F. A visibly deformed face layer or a scattered
  chip with F ≈ I carries no matching stress, and the torn chip still acts as one body through the stencil. The state
  contradicts itself, so no render can be honest to both the codes and the positions there.
- **Folds sit in this layer.** Every triangle of a B v1 self-intersecting pair (read-only, saved packages): 83/83 (G3b),
  22/22 (S3), 19/19 (S5) and 145/163 (S4) have a particle among their 8 nearest whose incident uncut link is > 0.5 s
  inconsistent with F, against 16 %, 29 %, 43 % and 5 % of all triangles. Median inconsistency 2.0–3.1 s at the folds,
  0.006–0.35 s overall. The dominant remaining B v1 gate failure lies in the F3 layer.
- **Mechanism (hypothesis, untested).** Positions advance with the particle velocity, 95 % FLIP (`material_config.py:13`,
  pic 0.05; `tlmpm_v5.py:588–590`). F advances with the gradient of the MUSL-remapped 4 mm grid velocity
  (`tlmpm_v5.py:870–877`, `:613`). Particle-scale velocity that a 4 mm grid with 64 particles per cell cannot hold
  therefore moves x but not F. A one-particle face layer, a crack-front hinge row and a 1–2-particle chip are such
  sub-grid features. Alternatives to rule out: light side-field nodes beyond a face; cross-cut particle-contact impulses
  (`tlmpm_v5.py:854`).
- **Scope: Phase 4.** Spec §4.1 item 4 (thin-chip residue) and item 5 (resolution), §4.2 "numerical resolution problem",
  §4.7, and the plan's resolution study. Not a law-8 failure region (the cuts separate), so not a reason for a law 9. Any
  solver change (pic, grid ratio) would be a new versioned opt-in solver/material profile. Before adoption it needs the
  complete law-8 acceptance rerun and the negative controls (side push, flat press, stop, withdrawal, partial cut, blunt
  handle). Phase 3 makes no such change.

## 3. Disposition

### 3.1 What a v3 = v1 + M1 + M2 + fragment-only tear rule would change (counterfactual, never a gate)

`gate_v2` (`run_wp12_phase3_surface_v2.py:1221`), recomputed read-only for the labelled no-M4 ablation (M1+M2+M3), is
**FAIL on saved, H1 and H2**: *not worse* False on every set; *better P95 every case* False on every set; *topology*
False on H2.

| Failing row (B v1 under v2, worse than A) | Sets | Cause | v3 expected | Basis |
|---|---|---|---|---|
| S4 current P95 0.96 / 0.88 vs A 0.77 / 0.69 | saved, H1 | tip in notch (F2) | → 0.44 / 0.45, B_BETTER | ablation |
| S4 provenance rows, G4 hold | — | v2: M4 | no M4 → v1 behaviour expected; M2 on the hold **untested** | ablation G4 NOT_RUN |
| Unexplained intersections (G3b, S3–S7) | all | F-extrapolated embedding in the F3 layer | **unchanged**; M3 adds folds | ablation, §2 fold check |
| Faces nearer another component (S5–S8) | all | faces offset outward (F12) in the same layer | **unchanged** | ablation |
| Thickness bias (S3, S5 cut 2, S8; H1 S4) | saved, H1 | F12 (unattributed) | **unchanged** | ablation |
| S8 current P95 0.51 vs A 0.43–0.44 | H1, H2 | recontact layer (unattributed) | **unchanged** | ablation |
| S7 maxima, reference P95 | all | membrane across F7 tear | partly (fragment rule acts on 35–48 pieces); P7 stays INCONCLUSIVE (D6) | M4 on S7: P7 4.71 → 0.24 |
| G2 n123 / n213 0.281 / 0.265 vs 0.25 | static | crease line misses the cell | **unchanged FAIL** | M1 |
| "Better current P95 in every case" | all | reference floor (D1) | **FAIL by construction** | §5 |

**Predicted v3 gate: FAIL on every set.** Cost: method and tests, a protocol v3, about 36 fresh physics runs × 2 repeats
(CUT-01-R 381–542 s each on the shared GPU; about 3–5 GPU hours), evaluation and review. That is roughly 1–2 working days
to learn what the ablation already shows.

### 3.2 Hybrid (spec: "若 Cut-aware Reconstruction 過於昂貴，應比較 Hybrid 方法")

**Not triggered on either ground.** *Cost:* B v1 is not too expensive. Kinematic extraction median is 0.12 s against A's
0.19–0.27 s (R.7). Native is +175 ms per step on the host CPU, of which 98 ms is the per-extraction reference-mesh
validity check and 38 ms the audit. A Warp port or validity only at recorded frames is cheaper than a hybrid, and a
hybrid still needs B's partition, cut faces and audit. *Quality:* the tech-review §6.8 trigger is not met, because B's
outer boundary beats A (S0 exact, where A FAILs; rim P95 0.18–0.38 s against 0.80–1.76 s). The failing rows are all at
cut faces, which a hybrid would still draw with B. Do not build `hybrid_surface_v1`.

### 3.3 Reopen criteria and the deferred v3 plan

**Reopen only if** Phase 4 shows that the first-layer inconsistency (P50 and P99 of \|dx − F̄dX\| at ≤ 1 s) falls by ≥ 2×
under a supported solver configuration that passed the full law-8 acceptance, or the user asks.

1. **Commit `surface_protocol_v3.json` first.** It fixes D1, D3 and D4 with B-independent reasons and declares every
   method parameter; the v1/v2 verdicts stay; G4 hold for every method; dense S8. Reserved H3 seeds:
   - **H3a (v1 geometries):** seeds 3301–3309; CUT-01-R seed 45.
   - **H3b (new geometries):** seeds 3401–3409; CUT-01-R seed 46. G3b yaw 15°; S3 −35°/s; S4 8 of 12 mm; S5 parallel
     cuts 28 mm apart + a 60° crossing; S6 22 mm pitch; S7 dome 4.5 mm, ρ 15°; S8 4 mm push. Offsets by the v2 rule.
2. **Method `cut_aware_surface_v3` = v1 + M1 + M2.** No M3 (it adds folds) and no v2 M4. M4′ instead: flag as v2, but
   act only on lattice pieces that removing the flagged links disconnects from their uncut-graph component; draw those
   as labelled `unlabelled_separation_v1` cells, never cut faces. Zero activations where removal disconnects nothing
   (every through-cut above). The G4 hold topology hash must stay constant. Freeze record before any physical state.
3. **H3 physics, then {A, B v1, B v3} × {v1, v2, v3}.** Decide under v3; print the others beside.

## 4. Profiles for Phase 7, README and acceptance-matrix wording

- **Defaults.** `particle_surface_v1` stays the default everywhere (env, Native, Demos 1–7, frozen records): the opt-in
  rule, plus B's +175 ms per step.
- **Phase 7.** Run the regression of Demos 1–7 with the default. For spec §7.3 "ParticleSurface vs Cut-aware Surface",
  pair each demo with a `cut_aware_surface_v1` Native run on the same inputs (as `surface_live_002/003/005/007`), with cut
  close-ups, **TRIM included with its spikes**, Native runtime only (no kinematic replay), never B v2.
- **README may say:** "opt-in `cut_aware_surface_v1` vs default `particle_surface_v1`; Phase 3 gate FAIL under the frozen
  protocols v1 and v2"; "render geometry, not the collider".
  - *Quantified improvements:* reference-space cut-face P95 ≤ 0.09 s vs 0.53–1.17 s (S7 excepted); coverage 1.00 vs
    0.26 (partial cut) and 0.37 (chip); no dropped fragment; rim P95 0.18–0.38 vs 0.80–1.76 s; S0 exact (2e-6 s); S9
    controls bit-identical to the uncut mesh; topology constant at unchanged codes in 3,715/3,715 Native extractions (A
    re-meshes in 338/702); matched-scale normal better than A in 7/7 comparable saved cases; about ¼ of A's triangles.
  - *Known failures:* folds where the solver's face-layer positions disagree with F; faces drawn outward by 0.10–0.15 s;
    crack-tip vertex inside an open notch (S4 P95 0.96 s); crease clamping (G2 0.40 vs 0.25 s); stretched triangles
    across chips the solver tears without codes (S7; TRIM spikes up to 16 mm).
- **README must not** say "physically accurate", "clean/smooth cut" or "Phase 3 passed"; cite ablation or M2 numbers as
  a method result; show B v2 as an improvement; or hide TRIM.

**Acceptance-matrix rows (exact wording):**
- **Surface: PARTIAL.** `cut_aware_surface_v1` (opt-in) is consistent with the cutting events: faces exactly on
  conflicting links with recorder Hermite data (S4 144/144; Native 600/600 and 2,400/2,400), 0 faces in every negative
  control, stable topology at unchanged codes. It follows the material through total-F embedding, **except** where the
  solver's positions disagree with its own F (cut-face layer, crack front) or tear without codes (thin chips, TRIM):
  there it folds, offsets faces outward and spans tears. Phase 3 comparison gate FAIL (protocols v1 and v2). The default
  `particle_surface_v1` misses partial-cut faces (coverage 0.26) and drops thin chips. Material uncalibrated; the mesh
  is not the collider.
- **Deformation: PASS (scoped).** No target plane and no smoothing: B v1 keeps the pressed zone (S2 reference P95
  0.013 s) and the curved cut (S3 single-plane residual 0.75 s); the default profile smooths sub-3 mm roughness
  (reported) but does not flatten mm-scale deformation. Scope: kinematic S1–S8 and four Native episodes. This row does
  not certify that the near-face deformation is physical (Phase 4).
- **Completion criterion 4 ("切割物理和切面幾何沒有重大矛盾") stays OPEN:** the surface exposes the F7 contradiction,
  which Phase 4 must resolve or bound.

## 5. Open metric and protocol defects (none to be fixed by loosening)

| # | Defect | Consequence | Blocks a v3? |
|---|---|---|---|
| D1 | Gate part "better current P95 in every case" (`surface_protocol_v2.json:132`; `run_wp12_phase3_surface_v2.py:1251`). In 7 of 9 cases on every set (all but G3a and S4), **both** A and B v1 lie below the locus's own leave-one-out floor: G3b floor 0.654, A 0.441, B 0.487; S2 0.878 / 0.58 / 0.54; S8 0.578 / 0.36 / 0.39 | The metric cannot resolve "better" there. P3 fixed the threshold row; the gate kept the v1 requirement. The spec asks only "優於或至少不劣於" | **Yes**: FAIL by construction |
| D2 | The current-space reference (side locus q = x_p + F_p(P − X_p)) is built from the F3 layer it judges; P4 INCONCLUSIVE in 7/9 cases | Embedding changes get tuned against an inconsistent state, and the folds sit there (§2) | **Yes**, until Phase 4 |
| D3 | "Better topology in ≥ 1 case" depends on A failing somewhere. On H2, A drops no fragment, so B v1 FAILs the part while never worse | Spurious FAIL | No (fix in v3) |
| D4 | M4 thresholds frozen in code, not in the protocol (v2 declares none) | Topology-changing parameters escape the protocol | No (process) |
| D5 | M2 (ablation) never run on the G4 hold | "Must not move a tip on a closed state" unverified | Yes, for shipping M2 |
| D6 | S7 chip row INCONCLUSIVE on every set (P7 with the P4 precondition) | No tear-handling method can be validated on S7 until F7 is fixed | **Yes**, for any tear rule |

## 6. Next steps for the implementer (high effort)

1. **Fold the read-only checks into the repository** as a new read-only script, `scripts/diagnose_wp12_phase3_v2.py`:
   M4 location and codes, fragmentation, fold-layer correlation, near-face inconsistency bands, and the counterfactual
   ablation gate labelled `COUNTERFACTUAL_NOT_A_GATE`. Run it on saved, H1 and H2, r0 and r1, CPU only. Write
   `artifacts/wp12/phase3_surface_002/diagnostics_001.json` (new file, never overwrite) and
   `docs/evidence/wp12/phase3_surface_002_diagnostics.json`.
2. **Add a dated "Addendum 2 (xhigh decision)"** to [`WP12_SURFACE_RECONSTRUCTION.md`](WP12_SURFACE_RECONSTRUCTION.md):
   the decision, the §1 corrections to R.5 (a), the fold/F3 correlation, D1–D6, and the counterfactual (labelled).
   R.1–R.8 and every v1/v2 number stay as recorded.
3. **Update the Phase 3 row of `WP12_PLAN.md` and `PROGRESS.md`:** "closed: gate FAIL (v1, v2) kept"; B v1 opt-in
   profile; B v2 labelled experiment; v3 deferred under §3.3; completion criterion 4 open (F3/F7 → Phase 4).
4. **Hand F3/F7 to the Phase 4 xhigh owner as solver-state items, not law items.** Proposed, all report-only:
   (a) a near-face consistency diagnostic over the saved `resolution_002` states, declared as a report-only addendum
   before the summarizer reads them, with no change to `resolution_protocol_v1` rows: the §1 \|dx − F̄dX\| bands, the
   flagged-link count and chip fragmentation, contrasting the `g4s` rows (dx = 4 s) with the `F3_fixed_grid` rows (dx
   4 mm, s 0.75 / 0.5); (b) a labelled COUNTERFACTUAL with pic ∈ {0.5, 1.0} on CUT-01-R, G3b and the S7 dome, to test the
   FLIP hypothesis; (c) a spec §4.2 classification of F7 (numerical resolution / contact / missing side assignment).
5. **Phase 7:** follow §4. Optional performance work (Warp port; validity/audit only at recorded frames) would be a
   versioned option and is not needed for closure.
6. **Do not:** run v3 or protocol v3 now; build a hybrid; promote the ablation or M2; change any default; touch
   `phase3_surface_00{1,2}`, law 8, `final_v4` or the WP-11 evidence.

## Read-only analyses (outside the repository; nothing written to run directories)

Scripts in `/mnt/data/ningan/tmp/claude-1004/-home-ningan-RoboticCuttingSim/2dbee09e-127d-4db1-a9d9-f2a32de8d08d/scratchpad/`,
run with `/home/ningan/RoboticCuttingSim/.venv/bin/python -I`, CPU only, on the r0 final frames of
`artifacts/wp12/phase3_surface_001/*_states.npz`, `phase3_surface_002/heldout_H2/*_states.npz` and
`phase3_surface_002/eval_saved/*_package.npz`. Outputs: `m4_where_*.json` in the same folder.

| Script | What it computes |
|---|---|
| `m4_where.py` | M4 location, direction, fragmentation and distance bands |
| `m4_codes.py` | endpoint codes |
| `fold_where.py` | fold/F3 correlation via `surface_metrics_v2.intersecting_pairs` |
| `abl_gate.py` | `gate_v2` of `cut_aware_surface_v2_ablation_no_M4` from the stored evaluations |
