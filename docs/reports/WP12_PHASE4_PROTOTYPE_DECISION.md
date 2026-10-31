# WP-12 Phase 4 prototype decision (xhigh): grid-consistent positions, contact, chip fracture, obliques, criterion 6

Date 2026-10-31 (KST). Decider `rcs-xhigh` (user instruction 2026-10-31). Branch `claude/wp12-physical-cut-surfaces`,
HEAD `71a0822`. **Inputs:** [CUTTING_PHYSICS](WP12_CUTTING_PHYSICS.md) "Root-cause verification (V1–V4)",
`docs/evidence/wp12/phase4_verification_001.json`, the row files `artifacts/wp12/phase4_verify_001/*/result.json`, the
[methods review](WP12_PHASE4_METHODS_REVIEW.md) §5–6, [D1–D5](WP12_PHASE4_XHIGH_DECISION.md), the
[Phase 3 decision](WP12_PHASE3_XHIGH_DECISION.md) §3.3, the [xhigh review](WP12_PHASE4_XHIGH_REVIEW.md) §6,
[Phase F](WP11_PHASE_F_LAW8.md), ADRs 0012/0014, spec §4.1–4.3, §4.7, §7.4, the constraints and the completion criteria.
**Unchanged:** no code, law, threshold, protocol, record or run directory; only this file was written; nothing was
committed. **Checks (read-only):** per-case values of V1.4, V2.2, V3.2, V3.4 and V4.2–V4.6; work, energy, momentum and
overlap from the row files (absent from the evidence JSON); `pytest tests/test_root_cause_recorders.py -k
"counterfactual or grid_position"` (7 passed); `tlmpm_v5.py` sha256 `a7a05b2f…`, unchanged.

## Decisions

| | Decision |
| --- | --- |
| P-A | **Build `tlmpm_v6` (`position_update="grid_musl_v1"`) now as an opt-in candidate, off the critical path, under a staged protocol with a kill stage.** Building is not adoption: nothing uses it before Stages 0–3 PASS (Native use also needs Stage 4). It changes the solver numerics; it is not a law and not a cutting improvement |
| P-B | **No contact change in WP-12.** Neither `penalty_area_v1` nor a zone in metres: V4.2–V4.4 refute pure compliance, and a fixed 1 mm zone is not s-independent (0.082 → 0.015 N, s 2 → 0.75 mm). Force stays a characterized NOT_CONVERGED |
| P-C | **Chip fracture is future work, outside WP-12** (exact wording below) |
| P-D | **D1 stands.** The in-core wedge (P3a) keeps every tested \|θ\| ≥ 10° flat-board row connected at every phase; P3b is a node-line lattice-phase artefact. The visibility rule is a future candidate only |
| P-E | **Criterion 6** is met as "studied and explicitly bounded" once E2 is recorded. Oblique cut **PARTIAL**, Negative controls **PARTIAL**, Regression **NOT_RUN** |

## P-A. Grid-consistent position update

**Evidence** (COUNTERFACTUAL_NOT_A_GATE, never acceptance).
- **V1.2:** the drift explains 0.958–0.996 of the first-band \|dx − F̄dX\|² in 8/8 rows, and the refute criterion
  (< 0.8) fires nowhere. The verdict is INCONCLUSIVE only through float32-limited far-field and identity items.
- **V1.4:** in 9/10 cases the first-band P50 falls 3.9–159× and P99 4.6–34×, with components unchanged in 10/10.
  Exception, CUT 0.5 mm: P50 0.056 → 0.068 s (q50 0.83), P99 ÷5.4. The remainder is label timing (P99 0.26–0.69 s
  on different-history links).
- **V2.3:** chip long links → 0. **V3.4:** P3b → 0, but the rows stay CONNECTED.

**Omitted from the V1–V4 Reading** (row files, 1 mm unless stated). Steady force: CUT +66 %, MULTI2 +49 %, TRIM
+24 %, OBL20 −21 %. Tool work: CUT ×2.04, MULTI2 ×1.82, TRIM ×1.43, OBL20 ×0.94. Blade overlap 0.37–0.70 → 0.24–0.42
mm. OBL20: bridges 54 → 110, and 0 → 38 unassigned within 3 mm. TRIM: labels 29 112 → 26 562, final KE 1.7e-9 →
5.9e-8 J, min det F 0.856 → 0.837, 0.5 mm removal 0.984 → 0.952 (≥ 0.9 holds). Wall −4 to +15 %.

**Why build now.** It is the only identified route that resolves, rather than bounds, Completion Criterion 4 (open per
Phase 3 §4), and it addresses the scatter part of §4.1 item 4. The rule is CPU-tested: bitwise v5 when off;
\|x − X − u_grid\| < 1e-6 m when on. **Cost:** about 1.5–2 days of code and plumbing, about 16 GPU-h, and about 1 day
of review. Half the GPU time is a v5 arm that also fills the open v5 gap (M3–M6 and M9 never rerun on v5). **Caveat:**
methods review §6 step 3 asked for "V1.2 confirms and V1.4 passes", and both are INCONCLUSIVE. This decision is a
declared judgement over the per-case data; the verdicts stand.

**Why off the critical path.** It fixes no separation failure (OBL20 gets worse), and forces and labels change. E2,
Phase 5 (benchmark v1) and Demos 1–7 run on `tlmpm_v5` law 8 regardless. If S3 is not done when the Phase 7 final
regression starts, report v6 NOT_RUN or PARTIAL and criterion 4 as bounded; WP-12 does not wait.

**Rule (exact).** New `src/robotic_cutting_sim/tlmpm_v6.py`, `class TLMPM(tlmpm_v5.TLMPM)`; only `step()` overridden:
```
step(advance):            # tlmpm_v5.TLMPM.step launch for launch (tlmpm_v5.py:820-891) plus (1) and (2)
  if position_update == "grid_musl_v1": (1) x_start <- copy(x)      # before _start_cut and every contact kernel
  _start_cut .. _rigid_contacts(k = c.contact_stiffness, zone = s) .. contacts, _claim_slots, _p2g, _grid_update,
  _g2p (v_p = a v_PIC + (1-a)(v_p + delta), a = c.pic; its x += dt v_p, tlmpm_v5.py:590, is overwritten by (2)),
  _musl, _grid_update (new_v = v^MUSL), _deformation (F += dt sum v^MUSL (x) gradN)
  if position_update == "grid_musl_v1":
      (2) x_p <- x_start_p + dt * sum_{c<8, slot[p,c]>=0} weights[p,c] * new_v[nodes[p,c]*S + slot[p,c]]
                                       # same field, slots and weights as _deformation
  _update_stroke, _assign_band, _assign_lead (law 8), _advance       # these read the new x
```
Options: `"grid_musl_v1"` (the v6 default) or `"flip_v5"` (identity, tests only). Receipts record
`solver="tlmpm_v6"`, the option, and the sha256 of `tlmpm_v5.py` and `tlmpm_v6.py`. Only an explicit
`solver="tlmpm_v6"` selects it; one "packed solver ∈ {v5, v6}" helper replaces about 20 `"tlmpm_v5"` checks in `src/`
(e.g. `live_material.py:52–63`, `kinematic_env.py:45–57`). ADR: proposed 0020 (0018 and 0019 stay reserved for the
surface and for any exit law). No import of the counterfactual module. Declared, not changed: `_assign_band` keeps v_p
in its rear tolerance (`tlmpm_v5.py:634`); the mismatch is ≤ dt·\|v_p − S v^MUSL\| ≈ 10 µm per substep, against b_back
8 mm.

**Frozen:** `tlmpm_v4.py`, `tlmpm_v5.py` and the counterfactual module, byte-for-byte; laws 4/5/8 and their
parameters; the material config (α 0.05, k 0.1, zone = s, G_c, μ); defaults, presets, `final_v4` and task identities;
`matrix_law8_00{1,2}`, Phase F/R and WP-11 records; the resolution, E1 and verification runs and protocols; surface
protocols v1/v2 and verdicts; envelope v1; benchmark v1; the E2 protocol. Frozen runners are wrapped read-only (the
Phase F v1 runner `5aea75d7…` is CPU-pinned).

**Classification.** Not visual-only: x feeds blade, board and particle contact and the law-8 assignment, and forces,
work and labels change. It is a time-integration change that removes strain-free FLIP null-space drift. If accepted,
report: "particle positions consistent with the deformation gradient; separation outcomes unchanged; forces changed
and still unconverged".

**Must not be claimed:** better or cleaner cutting; a fixed law-8 failure region or an oblique gain; force
convergence; fracture; calibration; "the standard TLMPM update" (Karamelo uses v⁺; this rule uses v^MUSL so that x and
F share one field); full x/F consistency (label timing and the CUT 0.5 mm P50 remain); a surface gain or "Phase 3
passed" before a v3 protocol; any V1.4 number as v6 evidence. v6 is never selected, accepted or tuned on a surface
metric.

**Acceptance** (`configs/wp12/phase4_gridpos_protocol_v1.json`, committed before any GPU row; skeleton below). Every
GPU row runs in a v6 arm and a v5 law-8 arm at one commit. An existing v5 record stands in only with identical solver,
driver and config sha256. Two repeats each of M1a, M3b and the flat-board CUT set the GPU band.
- **S0, CPU.** `flip_v5` is bitwise v5 (x, v, F, Fpi, codes, reactions, stats); \|x − X − u_grid\| < 1e-6 m; CPU
  controls (flat press, side push, stop, withdrawal, half cut, blunt) give 0 labels and 1 component; free-block
  ΔΣmv = impulse (1e-6); v4/v5 law tests and the full CPU suite pass.
- **S1, kill set (≈ 2.5 GPU-h).** M1a, M3b, M4, M5a, M5b, M8b, F2.7_withdrawal_v2, F2.9_blunt_handle_v2, and
  flat-board CUT and TRIM ρ 10° (E2 pipeline, `path_v2`). Any FAIL means **REJECTED**: stop, keep it, no tuning.
- **S2, full law-8 acceptance (≈ 8 GPU-h).** Matrix M1b, M2, M3a, M3c, M5c, M6, M7, M8a (diagnostic), M9a, M9b (M9c/d
  report-only). Phase F F2.1 at 14/15/16°, F2.3–F2.6, F2.8, F2.9 flat/side/no contact, F2.10, F2.11, F3 formal (dt_half
  characterization); F2.9 blunt v1 reported (its FAIL record stands); Phase R (b) ×7. **New post-cut controls:** after
  one full cut, arm digit 1 and apply a flat press, a side push, and a stop with withdrawal. Criterion: 0 digit-1
  labels, digit 0 and components unchanged.
- **S3, WP-12 (≈ 5 GPU-h).** Flat board with `path_v2`: CUT, partial cut, TRIM ρ 10°, Test B, Test C. 4c G01, G07.
  E1 A_p00, A_p05, A_p10, A_p20, R_p05_h0250, D_p20/25/30 reach. TEST_D2 s2501; TRIM_W s2307. **Effect set** CUT,
  TRIM, OBL20, MULTI2, S7, G3b at 1 / 4 mm, 0.1 ms. Characterization: CUT/TRIM/MULTI2 at s 0.75 and 0.5 mm (0.05 ms),
  and a force dt pair (0.05 ms).
- **S4, Native (v6 only, ≈ 1 h).** Flat-board CUT and TRIM ρ 10° pass the Phase 2b criteria and return HOME at
  ≤ 1.2× the Phase 2b per-step wall. Needed only before Native use.

**Gates:** frozen criteria PASS; no verdict worse than the v5 arm on criterion rows and on the validated cells of
envelope v1; components equal to the v5 arm on separating rows; TRIM removal ≥ 0.9 and substrate ≥ 0.99; integrity
(finite, det F > 0, min det F > 0.5, slot overflow 0, no centre below the board top). **Reported, not gated:** force, work, overlap, particle/grid/null-space KE, elastic energy, piece velocities at FINAL,
label deltas, and known-failure and partially validated rows. A change there is a question for a new envelope version.
**ACCEPTED** iff S0–S3 pass, otherwise **REJECTED** (kept). Cap 18 GPU-h; rows beyond it are NOT_RUN.

**Predictions:** acceptance rows and controls PASS (V1.4: components 10/10, TRIM removal ≥ 0.95; the gates read only
the tool twist). q50 ≥ 2 and q99 ≥ 2 on effect rows CUT, TRIM, MULTI2, S7, G3b; at 0.5 mm, CUT q50 < 2. CUT and
MULTI2 force and work above v5, OBL20 below; overlap lower; known-failure rows unchanged. **Risk rows, no confident
prediction:** M8b (law 5 failed its slice push; CUT work ×2), M2 (OBL20 bridges ×2), A_p05. **dt pair:** no
prediction (V1.3(b) left the CUT force at 0.0328 vs 0.0327 N).

**Phase 3 reopen feed.** The §3.3 measure is unchanged: P50 and P99 at ≤ 1 s fall ≥ 2× under a supported
configuration that passed the full law-8 acceptance. It is computed with `root_cause_diagnostics` on the effect rows,
v6 vs v5 arm, at the formal resolution. **Eligible** iff v6 is ACCEPTED and both factors are ≥ 2 on each of CUT, TRIM,
MULTI2, S7 and G3b (OBL20 and 0.5 mm reported). Only the row set is named here, so this is not post hoc.
**Eligible ≠ reopened:** v3 needs its own xhigh decision and a committed `surface_protocol_v3` (1–2 days, 3–5 GPU-h).

## P-B. Contact

**`penalty_area_v1`: not built.** The methods-review §6 step 4 condition fails: equal K_p is −18 to −27 % (V4.2),
fixed physical stiffness is +23 to +106 % (V4.3), and CUT at 0.0125 ms reaches 0.088 N > F∞ 0.065 N (V4.4).

**Zone in metres: not built.** V4.5 confirms that \|z\| ≤ s swallows the flank for s ≥ 1.36 mm. The omitted row
`V45_CUT_s075_dt0050_zone1mm` shows that a fixed 1 mm zone still depends on s: 0.082 / 0.056 / 0.033 / 0.015 N at s
2 / 1.5 / 1 / 0.75 mm (0.05 ms). The force counts flank contacts (half-thickness 0.75 mm + radius 0.45 s) outside the
zone: geometry, not material. At s = 1 mm the option equals v5, so it changes no supported configuration.

**Record:** "Tool contact force is NOT_CONVERGED in dt and s; every force number is a resolution-specific diagnostic.
(i) Series penalty compliance fits the dt series (V4.1) but is not the whole effect (V4.2–V4.4). (ii) Along s, the
release zone sets the force (V4.5). (iii) TRIM's blade-normal reaction has its own dt trend (V4.6). Only the frozen
minimum-force checks of the negative controls use force." **Board:** no centre below the top; overlap 14–65 µm.
**Future, outside WP-12:** velocity projection, or a release zone from the edge geometry; that is a law-4 contact
version with an ADR and the full acceptance.

## P-C. Chip fracture

**Out of scope for WP-12.** No confirmed problem is fracture: the scatter is kinematic (V2.1), and grid positions
remove all long links (V2.3). V2.2 refutes only the "no tearing below 6 mm" bound. Its excess is dilatational: J2 has
no tension limit, and a 5 mm chip reaches G_c at about 2 % volume expansion (E 50 kPa, ν 0.3). A fracture model would
need calibrated toughness and failure data (none exist), a stress-driven separation source coordinated with law 8's
labels (a new law), gating so that blunt contact, flat press and side push never separate material (§4.3 items 5–6),
and the full acceptance. Coherent chips under v6 are consistent with a solver that has no fracture state.
**Exact wording:**
> "Chip fracture is not modelled in WP-12. TLMPM v5 (and v6) has no damage or fracture state, so same-code material
> stays one mechanical body. Under the provisional, uncalibrated material, a Griffith energy estimate (history-maximum
> tensile energy × chip thickness, G_c = 0.05 J/m² provisional) exceeds G_c in thin chip material (0.26–0.63 J/m²),
> so the material model does not rule out tearing; this is not evidence that chips tear. Chip breakage is future work:
> it needs calibrated toughness and failure data, a new versioned law, and the full law-8 acceptance including
> blunt-contact controls."

## P-D. Obliques

**D1 stands.** Read "grid alignment" as **lattice phase**: the E1 plane through the node line x = 0, z = 0. V3.2: at
x0 0.5 / 1 / 2 mm, every 10/20/30° row is CONNECTED (2–338 bridges, all P3a; x0 1 mm at 10° is a 2-bridge margin
case). V3.4: P3b vanishes at 0.05 ms, k 0.4 or grid positions, and every row stays CONNECTED; OBL20 also stays
connected under grid positions (V1.4). No tested variant rescues \|θ\| ≥ 10° on the flat board. The bound is the law-4
front (f ≤ 0 at a tilted exit), and obliques stay on Option D.

**Visibility rule: future only.** V3.3 stays REFUTED under its declared rule (the trigger fired on uncut timing rows;
STATIC matches 19/19 cut rows; N1–N9 and the per-substep evaluator NOT_RUN). **Revisit only if** a flat-board oblique
capability is required and a per-substep shadow evaluator with edge history passes N1–N9, keeps h ≥ 0.75 mm
connected and labels at most a few hundred particles. Then: an ADR, the full law-8 acceptance and a new envelope.

## P-E. Criterion 6 and acceptance-matrix rows

**Criterion 6** (已知 Law 8 失敗區域經研究、改善或明確界定) is met as **"studied and explicitly bounded"** once E2 is
recorded. No class was fixed by a law change.

| Class | Root cause | Bound |
| --- | --- | --- |
| Obliques | in-core wedge (law-4 front) | flat board \|θ\| ≥ 10° KNOWN FAILURE; slotted Option D 20–30° validated (scripted) |
| 45° rake | float32 tie | ρ ≤ 44° |
| Sequential-cut residuals | core-band carry | KNOWN FAILURE; family F outside the target |
| Thin-chip scatter | FLIP drift; depression is wedge mechanics | "improved" only if v6 is accepted |
| Resolution | — | CUT/MULTI2 outcome-stable; TRIM qualified; force NOT_CONVERGED |
| Support | — | flat vs slotted characterized |
| (e) swept-normal fallback | fix specified, not shipped | pure draw at generic yaw outside the envelope |

Never write "law-8 failure regions fixed".

**Oblique cut: PARTIAL.** Envelope v1 (law 8). Flat board: \|θ\| ≥ 10° KNOWN FAILURE at every tested phase (20° at
every resolution); 5° at h ≥ 0.25 mm KNOWN FAILURE; 5–10° NOT_RUN; ≤ 5° partially validated (scripted h = 0) and
unusable in the planner or Native. Slotted Option D: 20 / 25 / 30° SEPARATED (scripted); 35° connected. NOT_RUN: E2
planner rows and Demo 2 Native. PASS needs both, and means "range declared and evidenced".

**Negative controls: PARTIAL.** Every run control gives 0 false labels: law 8 on v4 (M4–M6, F2.7, F2.9,
blunt_handle_v2; the v1 FAIL kept), v5 Phase R 7/7 (K = 5, single digit), WP-12 G8 5/5 and S9. NOT_RUN: post-cut
controls, M3–M6/M9 on v5, and v6. PASS once the post-cut controls are recorded.

**Regression: NOT_RUN** (§7.1, Phase 7). v6 must leave v4/v5 bitwise unchanged and change no default. Its v5 arm does
not replace the §7.1 rerun of the v4 law-8 acceptance.

## Findings ranked by severity

1. **High.** The V1–V4 Reading omits the grid-position physics changes (work ×2.04, OBL20 bridges ×2, TRIM 0.5 mm
   removal −3 %, TRIM final KE ×36). v6 is a physics change.
2. **High.** The prototype-1 precondition (V1.2 confirms, V1.4 passes) is not literally met; the build is a judgement.
3. **Medium.** V4.5 omits the s 0.75 mm, 1 mm-zone row (0.048 → 0.015 N), which refutes the zone remedy.
4. **Medium.** Law-8 acceptance on v5 is incomplete: M3–M6/M9 never rerun; v5 runs episode rows only
   (`run_phase1_matrix.py:622`); the ADR 0014 post-cut controls are NOT_RUN.
5. **Medium.** About 20 hard-coded `"tlmpm_v5"` checks in `src/` plus a CPU-pinned Phase F runner: plumbing risk.
6. **Low.** `tlmpm_v5_counterfactual.py:7` names a missing test file (the tests are
   `tests/test_root_cause_recorders.py:141–200`).

## Next steps

1. **Orchestrator:** commit this file. The critical path is unchanged: E2 → envelope v2 → Phase 5 on v5 → Phase 7 on v5.
2. **rcs-impl (CPU, parallel):** S0 (`tlmpm_v6.py`, the ADR, the solver helper, read-only wrappers, tests). Then
   commit the protocol from the skeleton and run S1 in GPU slack; S2–S3 only after S1 passes; S4 only before Native use.
3. **xhigh:** accept or reject v6. If accepted and the reopen measure holds, decide separately on Phase 3 v3 vs a
   bounded criterion 4.
4. **Separately:** ship fallback v2 (class (e)), not bundled with v6. Add a dated CUTTING_PHYSICS note for findings 1
   and 3.

## Appendix: protocol skeleton (`configs/wp12/phase4_gridpos_protocol_v1.json`)

```json
{"protocol_id": "wp12_phase4_gridpos_protocol_v1", "spec": "PROTOTYPE_DECISION P-A; spec 4.3 items 1-9; Phase 3 3.3",
 "declared": "before any GPU row; runner refuses an uncommitted or modified protocol; failures kept, never rescored",
 "candidate": {"module": "src/robotic_cutting_sim/tlmpm_v6.py", "solver": "tlmpm_v6",
   "position_update": "grid_musl_v1", "identity_option": "flip_v5", "adr": "0020 (proposed)"},
 "frozen_and_unchanged": ["tlmpm_v4/v5 (v5 a7a05b2f...)", "tlmpm_v5_counterfactual", "laws 4/5/8", "material config",
   "defaults, presets, final_v4", "law-8, Phase F/R, WP-11 records", "resolution_002, phase4_e1_001, phase4_verify_001",
   "surface v1/v2, envelope v1, benchmark v1, E2", "runners read-only"],
 "arms": {"V6": "tlmpm_v6 grid_musl_v1, law 8", "V5": "tlmpm_v5 law 8, same commit or identical-sha256 record",
   "REPEAT": ["M1a", "M3b", "FB_CUT"]},
 "setup": "1 mm / 4 mm / 0.1 ms unless stated; seed 42; kinematic ideal tracking; cut_diagnostics_v1 on",
 "stages": {"S0_cpu": ["bitwise flip_v5 == v5", "|x - X - u_grid| < 1e-6 m", "CPU controls", "momentum", "CPU suite"],
  "S1_kill": ["M1a", "M3b", "M4", "M5a", "M5b", "M8b", "F2.7_v2", "F2.9_blunt_v2", "FB_CUT", "FB_TRIM_r10"],
  "S2_law8": ["M1b", "M2", "M3a", "M3c", "M5c", "M6", "M7", "M8a(diag)", "M9a", "M9b", "M9c/d(report)", "F2.1_14/15/16",
    "F2.3", "F2.4", "F2.5", "F2.6_saw/draw", "F2.8", "F2.9_flat/side/no_contact", "F2.9_blunt_v1(report)", "F2.10",
    "F2.11", "F3_cut/trim", "F3_dt_half(char)", "PhaseR_b x7", "POSTCUT_press", "POSTCUT_push", "POSTCUT_stop_withdraw"],
  "S3_wp12": ["FB_partial", "TestB", "TestC", "G01", "G07", "A_p00", "A_p05", "A_p10", "A_p20", "R_p05_h0250",
    "D_p20/25/30_reach", "TEST_D2_s2501", "TRIM_W_s2307", "EFF_{CUT,TRIM,OBL20,MULTI2,S7,G3b}",
    "RES_{CUT,TRIM,MULTI2}_s075/s050_dt0050", "DT_{CUT,TRIM}_dt0050"],
  "S4_native": ["NAT_FB_CUT", "NAT_FB_TRIM_r10"]},
 "gates": {"S1": "any FAIL -> REJECTED, stop", "S2_S3": ["frozen criteria PASS", "no verdict worse than V5",
   "equal components on separating rows", "TRIM removal >= 0.9, substrate >= 0.99",
   "finite, det F > 0, min det F > 0.5, slot overflow 0, no centre below board top"],
   "S4": "Phase 2b criteria, HOME, per-step wall <= 1.2x"},
 "reopen_measure": {"rows": ["EFF_CUT", "EFF_TRIM", "EFF_MULTI2", "EFF_S7", "EFF_G3b"], "band": "<= 1 s",
   "rule": "q50 >= 2 and q99 >= 2 each, V6 vs V5, only if ACCEPTED", "reported": ["EFF_OBL20", "RES_*_s050"]},
 "budget": {"estimate_gpu_h": {"S1": 2.5, "S2": 8, "S3": 5}, "cap_gpu_h": 18, "native_wall_h": 1},
 "outputs": {"runs": "artifacts/wp12/phase4_gridpos_NNN/<arm>/<row>",
   "evidence": "docs/evidence/wp12/phase4_gridpos_001.json", "report": "docs/reports/WP12_PHASE4_GRIDPOS.md"}}
```
