# WP-12 Phase 3 decision review — `cut_aware_surface_v1` (B) vs `particle_surface_v1` (A)

**Scope.** This review covers the Phase 3 comparison FAIL recorded at HEAD `bdc5da0`
([`WP12_SURFACE_RECONSTRUCTION.md`](WP12_SURFACE_RECONSTRUCTION.md), run `artifacts/wp12/phase3_surface_001`,
protocol [`surface_protocol_v1.json`](../../configs/wp12/surface_protocol_v1.json) frozen at `b31553d`). It judges the
result against the spec (Phase 3 §3.1–3.7 and the gate), AGENTS.md and [`WP12_PLAN.md`](../WP12_PLAN.md).

**What was checked.**
- The protocol is unchanged since the freeze: SHA `afc30c75…` and its only commit is `b31553d`.
- `protocol_record.json` (01:28:14 local) was written before S0 (01:28:15) and G2 (01:28:20).
- `cut_aware_surface.py` and `surface_metrics_v1.py` have not changed since `89b289d`.

**Re-measured.** Read-only re-extractions of the saved r0 final states (cuda not used; B is bit-identical to the run):
- the S4 tip cells;
- S5, S2 volume localisation;
- S5 reference-max provenance;
- S6/S8/S5 component assignment;
- the locus noise floor on G3b, S2, S6, S8.

The ad-hoc scripts are outside the repository: `/mnt/data/ningan/tmp/claude-1004/-home-ningan-RoboticCuttingSim/2dbee09e-127d-4db1-a9d9-f2a32de8d08d/scratchpad/spot.py`, `side.py`,
`comp.py`, `s5.py`. Nothing was written to the run directory. Before any of these numbers is cited as evidence, fold
them into `diagnose_wp12_phase3_surface.py` and write `diagnostics_003.json` (a new file).

## Verdict

- **The Phase 3 FAIL is real and stays FAIL forever under protocol v1.**
- **No B code bug was found.** I agree with the report on this.
- **The report misses one metric defect.** `region_rule_v1` takes a vertex's mechanical component from its nearest
  particle. In S6 this measures 22 % of B's cut-face vertices against the neighbouring slice. It turns S6's
  "B_WORSE current P95" into an artefact.
- **One B design limit is sharper than reported.** In S4 the tip representative carries zero weight on the joined
  (code-0) hinge material.
- **One material finding is broader than reported.** The det F / position disagreement sits within 3 mm of the cut
  face. The same disagreement is behind the "particle-scale roughness" and the S7 tear. It is a solver-state
  property for Phase 4, not proven physical roughness.
- **Recommendation:** commit `surface_protocol_v2` (reasons below, B-independent) **before** implementing and
  measuring a versioned `cut_aware_surface_v2` on fresh states. Do not build `hybrid_surface_v1`.

## 1. Are the FAILs real and correctly attributed?

Ranked by consequence for the decision. *Spot* = re-measured in this review; *data* = consistent with
`diagnostics_002.json` and the run JSON, not re-run.

| # | Failure (report §) | Finding | Status |
|---|---|---|---|
| F1 | S6 current P95 B_WORSE (0.530 vs A 0.442); S6/S8 "faces nearer another component" (§6.5, §8) | **Metric defect, not reported.** `classify_mesh_vertices` takes the component of the nearest particle (`surface_metrics_v1.py:536-537`). B draws each face ½ s outside its piece, and in S6 the slices are within 1 s of each other (opening P50 0.56 mm, 197 crossed links). So **492 of 2,194** cut-face vertices (S6), 18/577 (S8) and 2/3,386 (S5) are assigned to the *other* component. They are then measured against the opposite face's locus. Without them, S6 B current P95 is **0.345 s** (A 0.442). Both A and B carry extractor component provenance (`component_ranges`), so the rule did not need to guess | **Spot: CORRECTED** (report attributes this to physics) |
| F2 | S4 current P95 0.96, normal P95 81°, tip in the open notch (§6.3) | **Confirmed, mechanism sharpened.** Every one of the 25 tip regions has 1–2 side-1 corners, 1–2 side-2 corners and 2–4 code-0 corners. The code-0 corners sit one row below, still joined: their link length along N is 0.99 mm. The QEF point lies on the cell face of the deepest cut links, because the Hermite points exist only there; 14/25 are clamped (`cut_aware_surface.py:794-795`). The restricted trilinear weights (`:858`) therefore give the code-0 hinge **zero weight in 20/25** regions. The vertex is the side-1/side-2 average, 2.40 mm from side 1 in a 4.6 mm centre-to-centre opening. The physical notch closes within one link. This is a B placement limit, not a bug | **Spot: CONFIRMED** |
| F3 | Current-volume rows fail for both methods; det F vs positions disagree by 1–5 % (§6.6) | **Confirmed and localised.** The tet volumes are sane: the reference cell mean is 1.000000. The deficit V/J sits at the cut. In S5, cells < 1.5 mm from the Hermite locus have mean V/J 0.77 with P5 **0.15**; 87 % of the deficit is within 3 mm and the far field is 1.006. In S2, cells < 1.5 mm have mean 0.52; 65 % of the deficit is within 3 mm and the rest is in the pressed zone. Near the cut, particle positions collapse while the solver's own F stays at det F ≈ 0.999. The 4 mm grid cannot resolve this sub-grid motion, so the reference really is inconsistent. **Corollary:** the 0.22–0.39 s face "roughness" that B follows lives in the same layer. The report's wording "real material roughness" (§2, §6.4) is too strong. It is position roughness of the solver state, which F does not see | **Spot: CONFIRMED + refined** |
| F4 | Normal P95 25–43° vs 10°; current P95 ≈ 0.5 s (§6.4) | **Real, and the v1 rows are mutually unsatisfiable on these states (B-independent).** The protocol's own current-space reference (side locus samples) has these residuals against its own 2 mm plane (core): P95 **0.512** (G3b), 0.402 (S2), 0.437 (S6), 0.315 (S8). G3b already exceeds the 0.5 s limit with the sample itself inside the fit. A Delaunay mesh **exactly through the locus samples** has triangle-normal error against the 3 mm locus plane of P50 17–28° and P95 **32–61°**. So any 1 mm mesh that keeps the distance rows fails the 10° row, and only a smoothed mesh can pass it. That contradicts protocol `smoothness_is_never_a_success_criterion` and spec §3.6. B's own 3 mm planes agree with the locus planes at P50 2.5–5.4°. Two outliers are not explained in the report: G3a P95 30.2° and S4 27.8° | **Spot: CONFIRMED (metric inconsistency)** |
| F5 | G2 (1,2,3)/√14 max 0.401 vs 0.25 (§6.1) | **Confirmed** from `G2_r0.json`. 55 clamped cut-face vertices: P50 0.147, max 0.401 s. The other 2,834 have max 7.5e-7 s. QEF rank counts are 6274/610/20. This is a real B limit on a zero-physics test | **Data: CONFIRMED** |
| F6 | Reference max G3b 1.70, S3 0.90, S5 1.32 (§6.2) | **Confirmed; the S5 part is now verified** (the report said "not separately verified"). All 14 S5 vertices with reference error > 0.5 s are region-rule misassignments: 7 are exterior representatives, and 7 are cut vertices assigned to the wrong cut digit at a junction. Measured against B's own face digit, those 7 are ≤ 0.018 s. In G3b, the 24 exterior vertices carry 0.90–1.70 s and the true cut-face max is 0.17 s (crease). Metric property | **Spot (S5) / data (G3b, S3): CONFIRMED** |
| F7 | S7 chip volume 5.7× (membrane) (§6.7) | **Confirmed, but the cause sits upstream of B.** All 86 chip particles carry code 2, so every link inside the chip is uncut by definition. Yet uncut links are 2.4 mm at P50 and up to 19.8 mm long, with \|dx − F dX\| up to 19 mm. Particles that share reference-grid nodes moved apart without F or codes recording it: the same position/F inconsistency as F3, in its extreme form. B is honest to the codes and dishonest to the positions. This is a Phase 4 thin-chip item | **Spot + data: CONFIRMED + refined** |
| F8 | Intra-shell folds G3b 85, S3 27 (§6.5) | **Consistent.** 0 pairs in the reference mesh within each component. 59/85 touch crease vertices. det F > 0 everywhere (0 translation-only representatives), so the material does not invert: this is an embedding artefact of F-extrapolation under non-affine strain | **Data: CONFIRMED** |
| F9 | Inter-piece overlap S6 1,198, S8 46; S4 slit crossings 198 (§6.5) | **Consistent with physics.** The protocol places the material boundary at ½ s: the exterior samples are x_p + F_p(s/2 e), `surface_metrics_v1.py:503`. Pieces closer than 1 s therefore overlap in the protocol's own convention. A hides this by shrinking. **Correction:** S4 also has **95 pairs in the reference mesh**. The report's "0 in the pre-embedding mesh" holds only for G3b and S3. In S4 they come from the zero-thickness slit: both sides' faces are coincident in reference space | **Data: CONFIRMED + note** |
| F10 | S8 slow-motion residual 0.072/0.052 vs 0.05 (§6.8) | **Real number, but the metric's premise is unmet.** The frames are not slow: material motion is P95 0.46–0.48 s per 40 ms frame. Nearest-particle transport is a piecewise-constant field, so its own error scales with the strain increment. The 0.05 s limit sits below the protocol's declared uncertainty band (0.06 s), and the B repeat spread is 0.02 s | **Data: CONFIRMED (metric premise)** |
| F11 | B_mid S8 "PASS" | **Reporting defect (low).** `s8_rows` returns `{}` for B_MID (`run_wp12_phase3_surface.py:972-973`), so *no* S8 row was evaluated, not only the slow-motion row. Report the S8 rows as NOT_RUN for B_mid; never cite this PASS | **CORRECTED** |
| F12 | Thickness bias G3b 0.152, S2 0.120, S3 0.124, S5 cut 2 0.149 | **Unattributed in the report.** *Hypothesis (untested):* the same-side corners one layer inside the face are extrapolated with F across the collapsed near-cut layer of F3, which places the face outward. Attribute it before the v2 method is designed against it | **OPEN** |

**Physics and negative controls.** Phase 3 changed no physics: `tlmpm_v5`, law 8 and `cut_events` are untouched. S9
(armed approach, flat press, side push, disarmed slide and plunge, G4 hold) gives 0 events, 0 digits and 0 cut faces,
with B bit-identical to the uncut extraction. "Hermite points outside the swept region" is 0. The S4 provenance rows
show faces exactly on the 144 conflicting links. These rows are sound and must not change.

## 2. Method fixes for a versioned `cut_aware_surface_v2` (opt-in; v1 kept byte-identical)

Common to every fix:
- render geometry only;
- no target plane, no blade pose beyond the recorder Hermite data, no particle edit, no cut face on a
  non-conflicting link;
- bitwise CPU physics check and S9 rerun required.

**Not allowed:** smoothing vertex positions on cut faces, inward offsets to hide piece overlap, snapping to the swept
plane, and choosing an estimator because it scores better.

| Fix | Mechanism | Expected effect on failing rows | New risks | Physics change |
|---|---|---|---|---|
| **M1 Constrained crease QEF** | Replace "unconstrained QEF then clip" (`:793-795`) with the box-constrained minimiser: an exact active-set over the cell faces, 3 variables. A clip is not the constrained optimum. Count clamps and record the residual | G2 n123 max should drop. If the cut/boundary crease line truly misses the cell, a residual remains: report it, do not loosen G2. Some G3b/S3/S7 crease error | Stays inside the cell, so no new folds; slightly slower | No |
| **M2 Crack-front hinge placement** (preferred over pure per-side representatives) | For regions that hold both side digits, move X_r along the QEF null space, in-plane, toward the region's code-0 hinge corners. The embedding then weights the joined material. One shared tip vertex is kept, so the v1 S4 rows `tip_representatives_shared` and 0 boundary edges keep their meaning. Per-side representatives (side k ∪ code 0) *alone* open the mesh along the front. They need closure faces from a shared front vertex, which amounts to M2 plus extra vertices. Use them only if M2 is insufficient | Removes the membrane across the S4 notch and the tip-driven cut/cut crossings. Crossings from the 24 physically crossed links remain. **S4 normal and current rows may still fail:** the real notch closes within one link (3.6 mm over 1 mm), so the V walls are about 60° from the locus plane. That is faithful, not an error | Hinge depth uncertain by ≤ 1 link; must not move any tip on a closed (G4 hold) state | No |
| **M3 Fold guard (report first)** | Per representative, detect orientation flips of incident triangles against the reference orientation mapped by cof(F). Where one occurs, re-embed that representative with the existing `lsq_affine` estimator (`:589`; region plus 2 uncut links), then translation-only. Count activations and report pairs before and after. Never trigger where det F ≤ 0: report material inversion instead | G3b 85 and S3 27 intra-shell pairs; part of S7 | Could mask a real fold. Mitigated because det F > 0 everywhere today and every activation is logged | No |
| **M4 Unlabelled-separation detector + labelled particle fallback** | Flag uncut links with current length > 2 s **and** \|dx − F dX\| > s. The second test measures inconsistency with F, not stretch alone, so rubbery stretch does not trigger it. Draw the affected lattice pieces as per-particle closed cells (side s·∛det F), labelled `unlabelled_separation_v1`: never cut faces, never in cut-face metrics. Report each activation as a solver-state anomaly for Phase 4. **Do not fall back to A**: A drops the S7 chip, which violates "no dropped fragment" | Removes the S7 membrane. **The chip volume then matches Σ(m/ρ)det F by construction.** Report it as such, never as reconstruction quality | Blocky chip appearance; a false trigger in high-strain regions must be counted. Not a fix for the physics (F7) | No |
| **M5 Position-consistent embedding (labelled variant, optional)** | Use `lsq_affine` instead of total F where the local cell V/J departs from 1 beyond a predeclared tolerance (F3). Report both estimators side by side | May reduce the thickness bias if F12's hypothesis holds; not expected to fix normals | Follows the collapsed layer more closely (rougher) | No |

**Hybrid.** `hybrid_surface_v1` (A for the outer surface) fails the tech review's own criterion (§6.8): B's outer
boundary beats A (S0 exact against A's 0.30 s P95; rim 0.18–0.38 s against 0.80–1.76 s). Do not build it. M4 is a
narrow tear fallback inside B v2, not a hybrid with A.

## 3. Protocol: `surface_protocol_v2`

**Rules.**
- v1 verdicts (B FAIL, gate FAIL) are kept and reported beside every v2 result forever.
- The v2 file is committed, with these reasons, **before** any v2-method measurement on physical states. v2 method
  development before the freeze may use only G2, zero-physics and affine controls.
- v2 thresholds come only from four sources: the spec, review §7.2, Phase 1 A controls and G2, and per-state
  *material-floor* procedures. Never from B's v1 numbers.
- Every v2 metric is applied identically to A, to B v1 and to B v2. Report the full matrix {A, B v1, B v2} ×
  {protocol v1, v2}. This separates metric relief from method improvement.
- The saved v1 states have been inspected while designing v2. v2 therefore also runs on **fresh held-out states**,
  declared in v2:
  - new seeds;
  - one new yaw and one new S5/S6 geometry inside the law-8 envelope;
  - S8 with dense frames.

| Change | Definition | Justification (independent of B) |
|---|---|---|
| **P1 `region_rule_v2`** | Vertex component from extractor provenance, which both A and B provide. Cut row and side from the nearest same-component sample, as in v1. **Max** rows over vertices with a classification margin ≥ 0.5 s between the chosen sample set and the next set (exterior or another cut digit). Ambiguous vertices are counted; **P95 rows use all vertices** | Inferring the component from the nearest particle is ill-posed when pieces are closer than the drawing offset, for any method that draws at the protocol's own ½ s boundary (F1). A nearest-sample classifier is undefined within ½ s of a rim or a junction, and rims are measured by the rim rows (F6) |
| **P2 Matched-scale normal** | Decision row: the angle between the mesh's own 3 mm patch plane (same component, cut and side) and the 3 mm locus plane. Threshold: the per-state P95 angle between 3 mm planes fitted to two disjoint random halves of the locus samples (declared seed), plus the 1° band. The v1 triangle-normal row stays reported as a shading/roughness diagnostic | The reference is rough at 0.3 s. A triangle-scale row is unmeetable by a mesh that interpolates the reference (F4: 32–61°). The comparison must use the same scale on both sides |
| **P3 Current-distance floor** | P95 limit = the locus leave-one-out residual P95 against its own 2 mm plane, per state and side group, + 0.10 s (2× §7.2's B expectation of 0.05 s). Max limits unchanged (1.5 s; S2 2.0 s) | The reference's own floor reaches 0.51 s (F4), so a fixed 0.5 s P95 tests the material, not the mesh |
| **P4 Current volume with a reference-consistency precondition** | Keep the 2 % row. When a component's mean cell V/J departs from 1 by > 1 %, or > 5 % of its cells are < 0.9 (declared), the verdict is **INCONCLUSIVE_REFERENCE**, never PASS. The exact reference-volume rows stay decisive | The reference contradicts the positions near the cut (F3). A judge that contradicts the state cannot pass or fail a mesh that follows the state |
| **P5 Slow motion made slow** | Frames with material displacement P95 ≤ 0.10 s (dense snapshots, new run). Transport by trilinear/IDW interpolation of same-component particle displacement at the vertex (declared). **Threshold 0.05 s unchanged.** Topology-hash constancy stays the jump detector | The v1 premise ("slow motion") was not met (0.47 s per frame), and the nearest-particle transport error is not bounded below the threshold (F10) |
| **P6 Intersection attribution** | Split pairs into *physical*: different pieces, or opposite sides of one cut, where the particle gap < 1 s or the sides are crossed. Everything else is *unexplained*, and every intra-shell pair counts. Only the unexplained pairs and "faces nearer another component" without physical proximity enter the gate; physical pairs are reported as Phase 4 contact data | Drawing a physical overlap of represented volumes is fidelity, not a defect (F9) |
| **P7 S7 accounting** | Chip volume row on the lattice-faced part only. Particle-cell fallback (M4) volume reported separately as "by construction". `skipped_fragments` 0 unchanged | Prevents a tautological PASS |

**Rows that stay exactly as in v1:**
- **S0:** all rows.
- **G2:** Hermite P95 0.05 / max 0.25 / normal 2° / 0 non-manifold, midpoint rows. n123 must be passed by M1, not
  by the threshold.
- **Reference space:** P95 0.1 / max 0.5, with only the P1 margin applied to max.
- **Current-space rows:**
  - |bias toward side 2| ≤ 0.1 s and |thickness bias| ≤ 0.1 s. These are means over hundreds of vertices, so they
    are not noise-limited;
  - coverage ≥ 0.99;
  - rim P95 0.5;
  - closed and shells = components.
- **S4:** all provenance rows, including `tip_representatives_shared`.
- **S5:** `slots_per_node`.
- **S6:** component and per-component reference-volume rows.
- **S7:** `skipped_fragments`.
- **S8:** shared vertices 0 and shell set unchanged.
- **S9:** all rows, including Hermite points inside the finite swept region.
- **Volume and temporal:** the "all" reference-volume row and fixed-state temporal rows.
- **Comparison:** the bands (0.06 s, 1°, 0.5 %), "worse in any repeat / better in every repeat", and "better on
  current P95 in every case, coverage and topology".
- **Never criteria:** smoothness and roughness stay non-criteria.

## 4. Recommendation for completing Phase 3

1. **Record, don't rewrite.** Add a dated addendum to the reconstruction report. It corrects four attributions
   and leaves every v1 number and verdict as is:
   - F1: S6 is a metric artefact;
   - F2: the tip mechanism;
   - F3 and F7: the localisation and the shared position/F cause;
   - F11: B_mid S8 is NOT_RUN.

   Then hand F3/F7 (near-cut particle collapse against F, unlabelled chip tearing) and the F12 hypothesis to Phase 4
   at xhigh. They are solver-state questions, possibly resolution (4 mm grid against 1 mm particles), and belong in
   the resolution study.
2. **Commit `configs/wp12/surface_protocol_v2.json`** with P1–P7, the unchanged-row list and the held-out state
   list.
3. **Implement `cut_aware_surface_v2`** (M1–M4; M5 as a labelled variant). Run unit tests on G2, the affine and
   zero-physics controls, S0 exactness and the S9 bit-identity. B v1 stays the default for existing records.
4. **Run `phase3_surface_002`** (2 repeats, new directory): the saved v1 states plus the fresh held-out states, with
   the full method × protocol matrix. Decide the gate under v2 and print v1 verdicts beside it.
5. **No hybrid** unless v2 shows B's *outer* boundary failing a v2 row that A passes.

**Honest Phase 3 claim now** (for PROGRESS and the acceptance matrix):

> Phase 3 gate: **FAIL** under frozen `surface_protocol_v1` (kept).
>
> `cut_aware_surface_v1` is better than `particle_surface_v1` on:
> - reference-space fidelity (P95 ≤ 0.09 s against 0.53–1.17 s; S7 excepted);
> - coverage (1.00 against 0.26 partial cut and 0.37 chip);
> - topology (no dropped fragment; exact per-component reference volume);
> - rim continuity and frame-stable topology;
> - exactness in S0 and every S9 negative control.
>
> It fails the frozen triangle-scale normal rows and is worse than A there.
>
> Its own design limits:
> - crease clamping;
> - a crack-tip vertex drawn inside an open notch;
> - post-embedding folds;
> - a membrane across an unlabelled tear.
>
> Two current-space rows (volume and normals) are shown to be limited by the reference itself. Particle positions
> near the cut disagree with det F, and these near-cut positions also cause the roughness. This is a solver-state
> finding handed to Phase 4.
>
> B is render geometry from solver state, not the contact surface.

**README may show:**
- A/B comparison figures labelled "`cut_aware_surface_v1` vs `particle_surface_v1`, Phase 3 gate FAIL (frozen v1
  protocol)";
- the S0/S9 exactness and the S4 coverage, with the tip-in-notch limitation visible;
- the S7 chip with "membrane across an unlabelled MPM tear" stated.

**README must not:**
- say "physically accurate cut surface" or "Phase 3 passed";
- use v2 numbers before the v2 freeze and run;
- present B as collision geometry.
