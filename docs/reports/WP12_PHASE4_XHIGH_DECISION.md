# WP-12 Phase 4 xhigh decision: oblique cuts, orientation envelope, resolution, 4c, Phase 5/7 scope (D1–D5)

Date 2026-10-31 (KST). Decider: `rcs-xhigh` (user instruction 2026-10-31). Branch `claude/wp12-physical-cut-surfaces`,
HEAD `ecfa8c6`. Inputs: [CUTTING_PHYSICS](WP12_CUTTING_PHYSICS.md) (4a, C1–C7, E1), [xhigh review](WP12_PHASE4_XHIGH_REVIEW.md),
[methods review](WP12_PHASE4_METHODS_REVIEW.md), [resolution study](WP12_RESOLUTION_STUDY.md),
[4c](WP12_GENERALIZATION_GEOMETRY_MATERIAL.md), [workcell](WP12_WORKCELL.md) §5/§9, [Phase 5](WP12_GENERALIZATION.md),
[readiness](WP12_AGENTIC_READINESS.md), WP-11 E/F reports, the evidence JSONs and the code paths cited below.
**Unchanged:** no solver, law, threshold, protocol, frozen record, WP-11 evidence or run directory. Written: this file
and `configs/wp12/supported_envelope_v1.json` (DECLARED). Nothing committed.
**Checks run (read-only):** an independent recomputation of every resolution verdict from the protocol text
(scratchpad `decision/reconv.py`, `dtsurf.py`); the declared envelope evaluated on all 160 DRAFT benchmark samples;
`pytest tests/test_wp12_generalization.py` (below). The V1–V4 verification (`phase4_verify_001`) is still running and
was not read for any decision; where its outcome could change an answer, that is stated.

## Summary of decisions

| | Decision |
| --- | --- |
| D1 | **No new law now.** Obliques with a sever-plane tilt go to the explicit slotted variant (Option D). Flat-board obliques: \|θ\| ≥ 10° KNOWN FAILURE, 5–10° NOT_RUN, \|θ\| ≤ 5° only PARTIALLY validated (scripted armed end exactly at h = 0) and **not usable in the planner or Native pipeline**. No 6–9° row set before the freeze; a small planner-driven confirmation set (E2) instead. |
| D2 | Envelope declared (table D2, JSON). Inside a JSON bound = admissible for execution, not validated; ID uses validated cells only. |
| D3 | All 66 headline verdicts reproduce (513 of 528 path verdicts). Two reported defects confirmed, one new defect found (F7 probe inside a fit path), three verdicts need an all-path qualification. Claim "outcome-stable", never "converged". |
| D4 | v1 and the existing goals stay frozen. The "0.4 mm stop" is the v1 controller's 0.5 mm waypoint tolerance, not a planned depth; WP-12 flat-board work uses `path_v2`. Planned depth stays h = 0 (never below the board). An opt-in armed-end option is deferred. Material profiles relabelled. |
| D5 | Phase 5 ID = validated cells listed below (B, F and parts of C/E/G have no validated cell today). Demo 2 on the slotted variant at 20°, never on the flat board ≤ 5°. |

**Angle names (spec §4.4).** τ, ψ, ρ, μ are ADR 0013's edge tilt, edge skew, blade rake, motion elevation. E1 and Phase 5
call the oblique angle "τ", but it is the **sever-plane tilt θ about the edge line**; the edge stays parallel to the
board, so ADR τ = 0 in every E1 row. This report writes θ (plane tilt), φ (plane yaw about z) and β (advance tilt inside
the blade plane) for through-cuts and keeps τ, ψ, ρ, μ for their ADR meaning.

## D1. Oblique cuts on `blade_master_flat_board_v1`

**Facts (verified in the evidence).**
- E1 (law 8, scripted edge driven exactly to h = 0): separated at \|θ\| ≤ 5°, connected at every \|θ\| ≥ 10° (both signs,
  6–380 bridges); 5° at h = 0.25 mm already connected (R_p05_h0250, 20 bridges); bridges grow with h at 20° and 35°.
- The low-angle margin is bar- and phase-specific: the 40 × 12 × 12 mm DEV bar stayed connected at 5°; every E1 A row has
  the plane through a node line (methods review §2.4, V3.2 pending).
- Slotted 8 mm variant at the reach-limited overtravel 6.67 / 5.35 / 3.87 mm: separated at 20 / 25 / 30°; 35° at 2.19 mm
  connected (90 bridges). The groove-limit rows are unreachable (handle −0.73 … −5.13 mm); the reach-limit rows keep only
  2.7–7.4 µm of handle-clearance slack (`phase4_e1_001.json` reachability).
- **New, decisive for the pipeline:** the planner never arms the edge down to h = 0. The CUT waypoint is armed
  (`goal_planner.py:368`), the HOLD at the same pose is disarmed (`:370`), and `WaypointPolicy` completes a waypoint within
  `position_tolerance_m` (`policies.py:262–267`): 0.5 mm for v1 (`:124`), 0.3 mm for `path_v2`, at 0.2 mm per 20 ms step.
  The armed end is therefore 0.3–0.5 mm (v1) or 0.1–0.3 mm (`path_v2`) above the board: measured 0.4 vs 0.2 mm in 4c
  (G07 78 vs 79 armed steps) and 0.16–0.28 mm live (Phase 2b §9.2). At 5° that is already outside the window E1 measured.
- Candidates: the proposed law 9 is rejected (`81a9626`); L9-EL fails control N3 in its own shadow (R_p20 h = 0.75 / 1.0 mm
  separated with 668 / 638 labels below the edge stop) and writes 1.2–7.1 k labels up to 24 mm from the plane with
  untested dynamics; the visibility rule is STATIC only and fails the P3b rows (A_p10 96/96 bridges, A_p30 228/380 need a
  mis-sided label).

**Decision.**
1. **No new law now.** The envelope can be met with law 8 on the explicit slotted variant, no candidate passes its own
   controls, and P3b may be an upstream contact/drift effect (V3.4) that a labelling law would only mask.
2. **Route** every θ ≠ 0 sever to the explicit Option D slotted variant (ADR 0017), labelled as such in every sample,
   report and video. Flat board: \|θ\| ≥ 10° **KNOWN FAILURE** (law 8, h = 0); θ = 5° at h ≥ 0.25 mm **KNOWN FAILURE**;
   5° < \|θ\| < 10° **NOT_RUN**; \|θ\| ≤ 5° **PARTIALLY VALIDATED** (scripted h = 0 only) and not validated for the planner or
   Native.
3. **6–9° rows before the freeze: not worth it.** They would refine an onset under a scripted armed end that no
   controller produces, and they cannot move the decision (the pipeline is at 0.1–0.3 mm, where 5° already fails). Phase 5
   OOD measures the flat-board boundary under the real pipeline at no extra cost (D5). Worth running instead: two
   planner-driven flat-board rows (θ = 2°, 5°) inside E2, as a falsification test of the armed-end argument above.

**Where V1–V4 could change this.** V3.2 (E1b phases at 10/20/30°) can only make the ≥ 10° failure phase-dependent; the
bound stays (a failing phase exists). If V3.3 (visibility shadow) passes N1–N9 with a few hundred labels **and** V3.4 shows
P3b vanishes under stiff contact or grid-consistent positions, an opt-in "visibility exit" variant (with prototypes 1–2)
becomes a candidate for flat-board obliques in the later decision: ADR 0019, CPU tests, full law-8 acceptance, N1–N9,
E1/E1b families, and a new envelope version. If P3b persists under both, no candidate exists and flat-board obliques stay
a WP-12 known failure.

## D2. Supported Cutting Orientation Envelope (law 8, 1 mm / 4 mm / 0.1 ms, `ruler_knife_v2_y`)

Kinematic ideal tracking unless "Native". Reach = this workcell (HOME, IK + C2 preflight). Abbreviations: val. =
validated, part. = partially validated.

| Motion | Physically feasible | Kinematically reachable | Validated cutting | Partially validated | Unsupported | Known failure |
| --- | --- | --- | --- | --- | --- | --- |
| Vertical sever (θ = φ = β = 0, τ = 0) | yes | yes (xy ±20/±10 mm, yaw ±3.2° co-rotated board) | flat board, armed end within [0, 0.375 s]: E1 A_p00, resolution CUT/MULTI2 at every s, 4c G01–G22 except thin and tunnel cells, Native CUT and Test B with `path_v2` | H 4–10 mm with `path_v2` (4 post-hoc diagnostic runs) | v1 controller on the flat board | v1 armed end 0.4 mm for H ≤ 10 mm (G07/G09/G10/G21); h ≥ 0.5 mm (S_P050, S_P100); h = 0.375 mm at s = 0.5 mm |
| Oblique, plane tilt θ, solid board | yes | yes to ±35° (A_p35 tie at contact) | none in the planner/Native pipeline | \|θ\| ≤ 5°, scripted h = 0 | — | \|θ\| ≥ 10° (h = 0); 5° at h ≥ 0.25 mm; 20° at every resolution. NOT_RUN: 5–10°, planner-driven 1–5° |
| Oblique θ, explicit slotted (8 mm groove) | yes | to the reach-limited overtravel (µm slack) | θ = 0, 20, 25, 30° (scripted, reach-limit overtravel) | — | groove-limit overtravel at 20–35° (handle) | 35° at 2.19 mm. NOT_RUN: planner 4 mm, −θ, 30–35°, Native |
| Plane yaw φ (vertical plane) | yes (no tilted exit front) | 13° yes, 32.9° IK-unreachable (one DEV sample each) | none | none | — | none; NOT_RUN everywhere |
| Advance tilt β (slanted chop) | yes | not checked on this workcell | none on the flat board | 15°, 30° (WP-11 relief slot) | — | none; NOT_RUN on the flat board |
| Near-horizontal trim, rake ρ (μ = τ = 0) | ρ ≥ 9.2° at 0.2 mm clearance | +x skims yes | ρ 10° (Native flat board, Test C), 14–16°, 25°, 35° (law-8 acceptance) | 10–44° (dome sweep) | ρ > 45° (gate closed by law) | ρ = 45° (float tie, 0 labels); ρ < 9.2° infeasible: the bevel heel sits 0.20–0.45 mm below the edge |
| Edge skew ψ | yes; ill-conditioned near a pure draw | ψ 35° IK-unreachable (WP-11) | 0–20° (WP-11 s7) | — | ψ → 90° at generic yaw (latent defect (e)) | none. NOT_RUN: 20–30°, ψ ≠ 0 on the flat board |
| Motion elevation μ, edge tilt τ | yes | — | μ = 0, τ = 0 | τ ≠ 0: rocking/rotation F2.4/F2.5 (old support) | — | NOT_RUN: μ ≠ 0 (the gate tests the motion's angle to the blade plane, = ρ only at μ = 0) |
| Stroke / approach direction | ±x, −z | −x skim and −x sweep collide or are unreachable here (DEV) | +x skim and push, −z sever | −x skim (WP-11 s6, old workcell) | lateral approach offset (not expressible) | — |
| Multi-step reorientation, one episode | yes | yes | Native Test B (4 cuts), Test C (3 skims) | sever ↔ trim (3/6 WP-11 TEST_D2) | — | class (c) core-band carry (s2501/2502/2506) |
| Sawing, repeated strokes | yes | — | F2.6 saw, F2.10 re-contact, three-cut history (old support) | — | pure draw at generic yaw until fallback v2 ships | NOT_RUN on the flat board and in Native |
| Full-length layer trim | yes | yes | — | — | — | continuous chip never separates (4a §7.2 slab rows) |

The JSON encodes the admissible bounds (θ_solid ≤ 9.9°, θ_slotted ≤ 34.9°, 9.2° ≤ ρ ≤ 44°, ψ ≤ 30°, φ ≤ 35°, β ≤ 30°,
digits ≤ 5, H 4–35 mm, speeds to the DRAFT OOD outer range, trim_layer excluded, `wp12_stiff_v1` excluded) and, per rule,
which part is validated. Envelope-wide: only with `path_v2` (v1 is a known failure for thin pieces); forces are not
converged; Native-validated cells are only the Phase 2b episodes (CUT, partial cut, TRIM ρ 10° with `path_v2`/`path_v3`, Test B, Test C)
and the Phase 6 multi-episode CUT/TRIM/CUT run.

## D3. Resolution study

**Reproduction.** An independent implementation of the protocol rules reproduces all 66 headline verdicts and 513 of the
528 path verdicts; the 15 others are new defect 3.

| Item | Decision |
| --- | --- |
| CUT, MULTI2 discrete outcomes CONVERGED | **Confirmed** on every declared path with three valid points (P_dt_s050 has two without the probe): 2 / 3 pieces, 0 bridges, all lines severed, s 2–0.5 mm, dx 4s and 4 mm, dt 0.2–0.025 ms |
| OBL20 discrete CONVERGED | **Confirmed only as "fails identically"**: a resolution-stable known failure; bridge count oscillates (38–292; 650/944 on the 4 mm grid) |
| TRIM components / criterion / bridges / removal CONVERGED | **Qualified**: true on the two headline paths; the formal-dt path flips at 0.5 mm / 0.1 ms (gap 0.1 s, one part), so the all-path reading is NOT_CONVERGED. Removal passes only through the insensitivity band (0.9975 → 0.9968 → 0.9838). The criterion holds in 22/23 rows; it is controlled by the edge-to-lowest-layer gap, not by refinement |
| Force (mean, steady, peak) NOT_CONVERGED, all cases | **Confirmed**. The s-path also carries the cutting-zone confound (\|z\| ≤ s covers the flank for s ≥ 1.36 mm; V4.5) |
| Near-cut deformation, TRIM depression, det F, surfaces | **Confirmed** NOT_CONVERGED / INCONCLUSIVE. The CUT spacing "CONVERGED" has p = 2.88 against an expected 1: pre-asymptotic, not citable. CUT det F is NOT_CONVERGED on the formal-dt and fixed-grid paths |
| Board contact | **Confirmed with qualification**: no centre below the board on the headline paths, but the fixed-grid path fails for OBL20/MULTI2 and MULTI2 at 0.2 ms reaches 2.8 mm overlap. On spacing paths the zero-limit rule is mis-specified (per-particle penalty ∝ s³/Δt²): overlap stays 14–65 µm at 0.05 ms, bounded, not vanishing |
| Defect 1: zero-limit rule on dt paths | **Confirmed**; re-read with the continuous rule, no headline changes |
| Defect 2: headline ignores the formal-dt and fixed-grid paths | **Confirmed**; it hides the TRIM flip, the OBL20 board metrics and CUT det F |
| Defect 3 (new): the F7 stability probe sits in a fit path | `summarize_wp12_resolution.py:220` builds P_dt at s 0.5 mm from dt 0.2/0.1/0.05 and picks up `CUT_s050_g4s_dt0200`, which the protocol says is "never in a fit". 15 CUT P_dt_s050 verdicts would be INCONCLUSIVE (2 points); no headline is affected |

**What may be claimed (README, acceptance §7.4).** "For a vertical through-cut and a two-cut sequence on the flat board,
with the armed edge ending on the board top, the separation outcome did not change between 2 and 0.5 mm particles,
dt 0.2–0.025 ms and a fixed 4 mm grid. TRIM removal met the WP-11 criterion at every row except 0.5 mm / 0.1 ms, where the
edge was 0.05 mm below the lowest dome layer; the edge clearance must be declared in lattice units. The 20° flat-board
oblique fails at every resolution. Contact forces, near-cut deformation, det F and surface errors are not
resolution-converged; every force number is a dt-dependent diagnostic." Never "numerically converged". Correct
`docs/PROGRESS.md:15` ("分離結果收斂") accordingly. The v1 verdicts stay as frozen; the all-path reading and defect 3 go into
a dated erratum labelled POST_HOC_DIAGNOSTIC; a future protocol v2 excludes probes, uses the continuous rule for surfaces
on dt paths and for board overlap on spacing paths, and states the headline over every declared path.

## D4. Phase 4c: geometry, materials, the default sever plan

**Geometry capability matrix (kinematic, one run per cell).**

| Status | Cells |
| --- | --- |
| Validated | bars L 50–150, W 12–48, H 12–32 mm; block 50×40×30; dome, gaussian, elliptical, three protrusions (CUT and TRIM); heightfield_v2 asymmetric and irregular and surface noise (CUT and TRIM); translation; co-rotated board yaw; SDF arch (CUT) |
| Partial | H ≤ 10 mm pieces and the SDF tunnel: FAIL with v1, PASS with `path_v2` (post-hoc diagnostic); overlapping bumps: CUT only, region `trim_plane` one diagnostic PASS; cut-aware surface flags real bridges in G09/G14/G15b/G17/G18 (correct behaviour, not a pass) |
| Unsupported | per-protrusion TRIM of overlapping bumps (planner mid-gap clipping); SDF TRIM; imported meshes, arbitrary SDF, several bodies, non-rectangular footprints, workpiece yaw relative to the board, sub-cell boundaries |

**Default sever plan.** Decision: **no change to `v1`, `sever_mid` or any existing goal** (frozen identities). The plan
already ends at h = 0 (`_relief_end`, `goal_planner.py:297–310`); the 0.4 mm is v1's armed-end tolerance (D1). One extra
armed step adds 8 labels to G07 (9160 → 9168) and removes all 8 bridges. With a continuous cut clearance v1 ends at
up to 0.5 mm, where even the 20 mm bar fails (S_P050), so v1 is a known failure on the flat board, not only for thin
pieces. WP-12 flat-board work uses `path_v2` (DRAFT common planner, Phase 2b Native): armed end 0.1–0.3 mm, inside
[0, 0.375] mm at 1 mm.
- **Rejected:** a planned stop "at a fraction of s above the board". Raising the target only consumes the window; a target
  below the board top breaks the never-penetrate rule (the kinematic blade does not see the board, WORKCELL §5) and gains
  nothing (S_M025_DIAG).
- **Deferred, opt-in:** an `armed_end` controller option (the armed waypoint completes within κ·s, κ = 0.1, target
  unchanged at h = 0). It is needed only for s < 0.8 mm (0.375 s < 0.3 mm), for flat-board obliques under a future exit law,
  or if Native margins shrink. Acceptance: byte-identical plans when absent, thin pieces, partial/stop/withdraw controls,
  one Native CUT without stall.
- **Thin-piece mechanism:** unresolved and not decision-relevant. The hypothesis is less wedge lift of the bottom layer
  (0.80–0.92 mm on the 20 mm bar); a cheap read-only check reruns G07 and G01 with `cut_diagnostics_v1`.

**Material profiles (all PROVISIONAL_UNCALIBRATED, one CUT and one TRIM run each).**

| Label | Profiles | Honest reading |
| --- | --- | --- |
| Distinct mechanics | soft_elastic, clay_plastic, firm, light, dense | deformation, sag, rebound and force differ; separation identical in kind (all PASS) |
| Force-only variants | low/high_cut_resistance (G_c) | codes and positions identical to the reference under law 8; not a separation property; exclude from any "material generalization of cutting" claim |
| Weak-effect variants | low/high_friction | μ 0.8: 0 codes differ; μ 0.05: 4 / 24 differ; friction-dominated tasks NOT_RUN |
| Outside the formal resolution | stiff | refused at 0.1 ms (CFL 0.290); its "4× TRIM force" is partly dt: the same material's force rises ×2.9 (CUT) and ×1.17 (TRIM) from 0.1 to 0.05 ms |

## D5. Phase 5 distribution and Phase 7 demos

**ID (flat board, `path_v2`, `cutting_tasks_v1`, 1 mm; validated cells only).**
- **A:** vertical sever, H 16–24 mm, L 140–170, W 26–34, xy ±20/±10 mm, yaw ±2°, cut fraction ±0.2, clearance 6–10 mm,
  **cut speed 10 mm/s fixed** (8–12 only after E2).
- **B:** **no validated cell today.** After E2: θ ≤ 20° on the slotted variant with the planner's 4 mm, φ ≤ 15° and β ≤ 15° on
  the solid board, each only if its rows pass. Until then, B has no ID stratum.
- **C:** `skim_protrusion`, ρ 10–35°, ψ ±10°, +x, skim speed 25 mm/s fixed, dome/gaussian/ellipsoid_cap, height 5–9 mm.
  `trim_plane_protrusion` after E2. `trim_layer` is removed (known failure).
- **D:** `multi_plane_sever` and `cut_cut`, 2–3 cuts.
- **E:** `remove_all`, `remove_selected`, `bumps_sequential` (2–3 separate footprints); `trim_trim` after E2.
- **F:** no validated cell (class (c) failed 3/6 in TEST_D2). Run as declared, report fully, but **outside the §5.7
  target** (an "ID-unconfirmed" stratum declared before the freeze).
- **G:** `cut_push` after E2. `trim_clear_chip` and `cut_reposition_cut` are not validated: outside the target.
- **H:** vertical sever and skim on solid or slotted boards; board size, thickness and friction (PhysX only).

**OOD boundary dimensions.**
- **Size and pose:** H 10–16 and 24–35 mm; L, W, xy beyond ID; yaw beyond 3.2° is scene-infeasible.
- **Orientation:** ρ (35, 44] (and < 9.2 / > 44 classified unsupported); ψ (10, 30]; φ (15, 35]; β (15, 30]; slotted θ
  (20, 34.9] (30° preflight-limited).
- **New OOD dimension:** B tilt on the solid board with 0 < \|θ\| ≤ 9.9°. It measures the flat-board boundary under the real
  pipeline and fills the 5–10° gap.
- **Speeds and counts:** cut 4–30 and skim 10–60 mm/s; 4–5 cuts; 4 bumps; −x strokes (reachability); ridge.
- **Materials:** the ten admitted profiles; G_c and friction variants reported as force-only or weak, never counted as
  generalization of separation.

**Phase 7 demos.**

| Demo | Configuration | Status |
| --- | --- | --- |
| 1 | flat board, vertical CUT, `path_v2` | Phase 2b Native PASS |
| 2 | **explicit slotted variant, θ = 20°, planner 4 mm overtravel** (estimated handle slack ≈ 2.5 mm, to be measured), labelled Option D; optional flat-board 20° known-failure comparison (A_p20) | needs E2 kinematic then Native (NOT_RUN). **Not** the flat board ≤ 5°: live armed end 0.16–0.28 mm, 5° fails at 0.25 mm, 2–3° is not visibly oblique |
| 3 | flat board, ρ = 10° dome | Phase 2b Native PASS |
| 4 | Test B, 4 cuts, 5 pieces | Phase 2b Native PASS |
| 5 | Test C, 3 bumps; disclose the two chip particles 0.6 mm below the board top | Phase 2b Native PASS |
| 6 | Cut → reposition (push) → trim in **one** episode, flat board | needs E2 kinematic then Native; class (c) residuals reported if they occur. Not the Phase 6 multi-episode CUT/TRIM/CUT run (separate episodes); clear-chip is NOT_RUN |
| 7 | one Phase 5 benchmark ID sample from a validated stratum (A/C/D/E/H), chosen by the declared Native selection rule | after the freeze |

## Findings ranked by severity

1. **High.** Flat-board obliques ≤ 5° are not usable in the planner or Native pipeline. The armed end (`policies.py:262–267`,
   `goal_planner.py:368–370`) lies outside the 5° window E1 measured (R_p05_h0250). This changes the Demo 2 premise.
2. **High.** `freeze_problems` skips rules with a `when` condition (`generalization_wp12.py:1325`). With the declared
   envelope it flags only C `trim_depth`; 2 of 10 B ID benchmark samples (solid θ > 9.9°) are physically_unsupported
   without a freeze error, and every other solid-board B tilt is admitted but unvalidated.
3. **Medium.** Declaring the envelope fails 3 CPU tests that pin the TEMPLATE state (`test_envelope_template_has_fields_only`,
   `test_classifier_valid_feasible_and_pending_envelope`, `test_freeze_problems`; 29 others pass). These tests encode the
   pre-declaration state by design.
4. **Medium.** Resolution: TRIM headline hides the formal-dt flip; the F7 probe sits in a fit path; PROGRESS overstates
   ("收斂").
5. **Medium.** Speeds (8–12, 20–30 mm/s), φ, β, the slotted planner overtravel, `trim_plane` and push have no flat-board
   validation, yet sit in the DRAFT ID.
6. **Low.** v1 on the flat board fails thin pieces and, with continuous clearances, can end at 0.5 mm. The stiff profile's
   force ratio is confounded by dt.

## Next steps (in order)

1. **Orchestrator.** Commit this file and the DECLARED envelope. In the same commit, change the three TEMPLATE-pinned tests
   to build a template in-test and to check the declared file's invariants; rerun
   `.venv/bin/python -m pytest -q tests/test_wp12_generalization.py` (now 3 failed / 29 passed).
2. **rcs-impl.** Make `freeze_problems` evaluate `when` rules for families that fix `board_kind`. Re-declare the DRAFT per D5:
   B tilt on slotted, trim_layer out, speeds fixed, F/G-partial strata outside the target, solid-board B tilt as an OOD
   dimension.
3. **rcs-impl.** Commit `configs/wp12/phase4_e2_envelope_confirmation_protocol_v1.json` **before any row**: kinematic,
   planner `path_v2`, law 8, 1 mm / 4 mm / 0.1 ms, seed 42, `cut_diagnostics_v1` on, declared predictions, ≤ 2 GPU-h,
   failures kept. Rows:
   - slotted θ +20, −20, +25 (4 mm);
   - solid θ 2, 5 (predict 5° CONNECTED);
   - φ ±15; β ±15;
   - CUT at 8 and 12 mm/s; dome skim at 20 and 30 mm/s;
   - ρ 35°; ψ ±10°;
   - `trim_plane_protrusion`, `trim_trim`, `cut_push`;
   - cut → push → trim (Demo 6 candidate).

   Record the outcome as envelope v2 (new version, v1 untouched). Then freeze Phase 5.
4. **Reports.** Add the dated resolution erratum (D3). Fix `PROGRESS.md:15` and the README WP-12 oblique and resolution
   lines. Merge 4c into CUTTING_PHYSICS with the D4 labels.
5. **After V1–V4.** A separate xhigh decision on prototypes. Any adopted law or profile needs the full law-8 acceptance,
   the controls (side push, flat press, stop, withdrawal, partial cut, blunt handle, N1–N9) and a new envelope version.
6. **Phase 7.** Demos 2 and 6 only after their E2 rows and a Native run. Every slotted demo is labelled Option D.
