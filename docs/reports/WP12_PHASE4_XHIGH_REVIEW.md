# WP-12 Phase 4 xhigh review: Phase 4a root causes and the proposed law 9

Date 2026-10-31 (KST). Reviewer: `rcs-xhigh`, per the user's 2026-10-31 instruction. Branch
`claude/wp12-physical-cut-surfaces`; the reviewed files are unchanged from `b012902` to `9157725`.

**Reviewed.** [WP12_CUTTING_PHYSICS.md](WP12_CUTTING_PHYSICS.md), `docs/evidence/wp12/phase4a_law8_rootcause_001.json`,
`artifacts/wp12/phase4a_00{2..5}`, `tlmpm_v5.py`, `cut_diagnostics.py`, ADRs 0006/0008/0010/0011/0012/0017,
WP11_PHASE_F_LAW8.md and WP12_WORKCELL.md.

**Unchanged.** `src/`, the law-8 records, `final_v4` and the WP-11 evidence were not touched. Nothing was committed.

**Reviewer checks.** Read-only analyses of saved runs, plus a small **reviewer CPU probe** (unchanged law 8, Warp
1.17.0 CPU, NumPy 2.5.4). Scripts and outputs: `artifacts/wp12/phase4_xhigh_review_001/`. Results marked REVIEWER,
STATIC or GEOMETRIC are diagnostics. None is an acceptance result or a law-9 simulation.

## Verdict

**REJECT law 9 as proposed** (breakthrough-gated lead rule plus the relative fallback).

- **The 4a diagnosis stands, with the corrections in §2.**
- **The rule misses the case it is meant for.**
  - It fixes the WP-11 30 mm relief slot, where law 8 already separates unchanged with ≥ 6 mm of overtravel or a
    12 mm slot.
  - On the solid `blade_master_flat_board_v1` board it does not help. There (Option A: the edge stops on the board
    top) its breakthrough mask blocks itself.
- **Next.** Run the decisive flat-board experiment E1 (§6). Only if E1 shows a law is needed, evaluate a
  **two-sided exit-ligament rule** (§3.3) offline.
- **(e)** is correct. Ship it as its own versioned numerical variant.
- **(b)** Declare ρ ≤ 44° now. No change to the motion gate.

## 1. Spot-checks on saved data (`check_a.py`, `check_a_time.py`, `check_b.py`)

| Draft claim | Reviewer recomputation | Status |
| --- | --- | --- |
| (a) s2201/2202/2203: 1 component, direct bridges 2/84/6 | 1/1/1; bridges at s = −4.7, +5.2…+6.0, −6.5 mm; f_final +1.0/+1.3/+0.9 mm; f_reference −1.41/−0.92/−1.21 mm | confirmed |
| 684/616/314 unassigned within 12 mm; one side; \|s\| > 4 mm; only failing f > 0; lead rule blocked by the plow mask | same counts; \|s\| 4.68–12.0 / 5.18–11.96 / 6.47–11.89 mm; `only_fail` = ahead_of_edge for all; no other failing reason | confirmed |
| Proposed B_k holds at the final pose | STATIC: B_k = 1 on 38/38, 34/34, 32/32 material segments; would label 3,706 / 3,134 / 2,440 particles (f 0.12–7.96 mm, \|s\| 4.7–24 mm); 2 components | confirmed |
| §8: "unchanged until breakthrough" | STATIC on snapshots: mid-cut, B_k = 1 on 2–4 material segments with an empty core (they would label 0, by chance). At t = 5.60 s, 0.86 mm past the exit, B_k = 1 on only 4/36 (s2201) and 28/30 (s2203) | **refuted as stated**: B_k fires mid-cut and lags the exit by about 1–3 mm |
| (b) s2307: gate closed on 120/120 segments for 17,200 substeps; 0 labels | recorder: seg_ok 0.0, 0 labels. A host float32 recomputation from the recorded poses **passes** on 86/86 moving steps (tan²45° = 1.0f) | confirmed: arithmetic tie |
| (b) COUNTERFACTUAL θ_max 60°: 24,563 labels, 1 component, one carried apex particle | identical; single bridge at reference (7.5, 5.5, 16.5) mm, displaced 29.8 mm; relabelling it gives 2 components | confirmed |
| (e) 0 digits flipped in every recorded run | every `flipped_by_*` field is 0 | confirmed |

## 2. Root causes (a)–(e)

### (a) Oblique exit

**Mechanism.** The law-4 front f = (x−o)·d ≤ 0 (`tlmpm_v5.py:630`) is the plane through the edge ⊥ d. At a tilted
exit face, material on the obtuse ("uphill") side beside the swept surface is never behind the edge when the edge
leaves the bar. With overtravel t along d, a bottom-layer particle at height z_b is passed only if
|s| ≤ z_b/sinθ + t/tanθ.

| Candidate cause | Role |
| --- | --- |
| f ≤ 0 at oblique incidence | **the root cause**, on every support |
| Overtravel | sets where the wedge starts. WP-11's 4 mm came from the action workspace (`cut_extent = plane_section`), not physics |
| Sag over the 30 mm relief slot | decided the reproduced bridges (f_reference < 0); this support does not exist on the flat board |
| Law-8 plow gate | **not a root cause.** It matters only once overtravel moves the wedge outside the core (\|s\| > Δx), so "law 5 would label them" holds only on the slotted board |

**Does (a) occur on `blade_master_flat_board_v1`?** The draft ran no flat-board oblique. The REVIEWER probe
(`cpu_flat_oblique.py`) says it does, in a form the proposed rule cannot reach.

Probe setup: a 40 × 12 × 12 mm bar of 1 mm particles on a solid board box; WP-11 material; `ruler_knife_v2_y`;
10 mm/s along d; law 8; the edge ends on the board top (h = 0).

| τ (deg) | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Law 8: components (direct bridges) | 2 (0) | 1 (2) | 1 (2) | 1 (16) | 1 (14) | 1 (174) | 1 (62) | 1 (234) |
| Proposed B_k = 1 at the end (material segments) | 12/12 | 14/14 | 14/16 | 14/16 | 8/16 | 10/18 | 4/16 | 4/16 |
| STATIC proposed rule → components | 2 | **2** | 1 | 1 | 1 | 1 | 1 | 1 |
| STATIC exit-ligament rule (§3.3) → components | 2 | 2 | 2 | 2 | 2 | 1* | 2 | 2 |

\*Two particles at f = −0.00/−0.01 mm after the stop: a static-test margin that a continued stroke's band would take.

**Reading the probe.**
- At 5° and 10° the cut fails by two bottom-layer particles 0.01–0.05 mm ahead of the edge, a margin case.
- At 15–35° it fails robustly. At 35°: f up to 3.8 mm, layers z 0.4–4.4 mm, unassigned at s −0.8…−11.9 mm, 48
  of them inside the core.
- A GEOMETRIC rigid-lattice model agrees (`geom_flat.py`, `geom_ot.py`, 0.4 mm lift). At h = 0, law 8 fails from
  15–20° depending on lattice phase. The proposed rule changes 1 of 18 (θ, phase) cells. Without sag, 1–2 mm of
  overtravel separates under law 8.

**Draft errors.**
- "±30–35° obliques already bind on handle clearance" misreads WP11_PHASE_E_ROUND2.md:237. That line names the
  active constraint, with 5.6–6.1 mm of margin. Flat-board reachability is unmeasured.
- ADR 0017:121 and WP12_WORKCELL.md:193 keep oblique planes on the explicit slotted variant (Option D). No oblique
  has been run on its 8 mm grooves.

**Verdict (a).**
- **Slotted reproduction:** the cause is correct.
- **WP-12 workcell:** the cause is not demonstrated. The probe points to the in-core wedge.
- **Before any law design**, E1 must decide.
- **The draft's §8** must not claim "obliques ending on a solid board" as a law-9 effect.

### (b) to (e)

**(b) Correct and complete.** The gate is an arithmetic tie at θ_max (`:365`, `:712`), with class (c) behind it.
TRIM_W sampled ρ ∈ {5, 35, 45, 60}°, so 45° is an ill-posed sample of a `≤` test. The frozen FAIL stands.

**(c) Correct by code.** The band needs f ≤ 0 (`:630`). The lead rule needs |s| > Δx (`:661`) and, in law 8, a
plowing segment (`:885`). The bumps3 merge is a representation/evaluation issue.

**(d) Correct.** "Planner clearance relative to the lattice" can at most be a declared, resolution-specific planner
rule. It is never a physics fix.

**(e) Correct and latent.** Add one limit: it repairs float noise (`ill_float`) only. Near-pure draws under tracking
noise (`ill_geom`; Phase F: 0.3 mm/s turns n_s by about 45°) need their own study.

## 3. Law 9 as proposed

### 3.1 Physics and minimality

Breakthrough is a sound physical idea: once the ligament along a segment is consumed, the nearby particles belong to
their sides. The detector is wrong.

- **Wrong test.** "No digit-0 particle in {0 < f ≤ 2Δx, |s| ≤ Δx}" asks whether there is *material* in the core
  ahead. It should ask whether a *ligament straddles* the swept surface.
- **Self-blocking at a supported oblique exit.** The one-sided uphill wedge, the very particles needing labels, sits
  in the core strip. The reused law-5 rule excludes |s| ≤ Δx anyway.
- **Overtravel still needed.** The proposed rule needs about Δx·tanθ − z_b/cosθ (≈ 1.3–2.2 mm at 25–35°). Option A
  provides none.
- **Its only proven gain** is cutting the slotted-board overtravel from about 6–10 mm to about 2–4 mm.
- **No press-versus-directional distinction** is added: the rule inherits law 4's S/M_k gate.
- **Small code, large effect.** It is one counting launch plus one OR, but it can label in empty space and it splits
  fields at the exit.

### 3.2 Safety

**Safe**, because B_k is evaluated only while S = c+1 and M_k:
- flat press and side push: M_k = 0 (`:365`);
- stop: S = 0;
- withdrawal: not moving;
- blunt handle and no contact: W = 0 (`:282`);
- partial cut: the ligament fills the core.

**Not safe:**
1. **Empty space beside material.** S is per blade and persists for 2Δx (`:412`). Any moving segment with an empty
   core gets B_k = 1, even if it never cut. Examples: between bumps more than Δx above the substrate; beside a step,
   pocket, groove wall or second workpiece 4–24 mm away; after the exit. The lead rule then labels material the edge
   never reached, and ADR 0008's safety argument (a code-0 core couples the sides) is void by construction.
2. **Digit-0 counting.** A re-entered kerf with the same digit, labelled carried chips and the rule's own labels all
   count as "empty".
3. **Dynamics.** New labels split fields and change FLIP transfer (ADR 0010): law 5 failed M8b this way (2.3 vs
   1.2 mm of push). Here 2.4–3.7 thousand particles would be labelled at once at the exit, unmeasured.
4. **Sampling.** The mask reads a 1 × 8 × 8 mm strip: one stray particle blocks it and one void opens it.
5. **Multi-cut and intersecting cuts.** Digit-c labels can land on separate bodies (removed chips, the far piece past
   an open kerf) and change their compatibility (the bumps3 merge).

### 3.3 Better candidate: exit-ligament rule L9-EL (evaluate offline on E1 states; not approved)

L9-EL is law 8 plus one assignment rule. Laws 4, 5 and 8 are unchanged, and it adds no parameter (b_lead = 2Δx,
s_core = Δx, h = 6Δx). Per armed digit c, segment k and substep, it runs after `_update_stroke` and before
`_assign_band`:

```
# _ligament_count (particles): ALL codes, l in segment k, 0 < f <= b_lead, |s| <= s_core (s: _swept_side, fallback e)
#     s < 0 -> atomic_add(Nneg[k]);   s >= 0 -> atomic_add(Npos[k])
# _exit_mask (segments):
if S != c+1:  seen[k] = 0; X[k] = 0                  # also cleared when the armed digit changes
else:
    lig = Nneg[k] > 0 and Npos[k] > 0                # material straddles the swept surface ahead
    if lig: seen[k] = 1                              # segment k had a real ligament in this stroke
    X[k] = seg_ok[k] and seen[k] and not lig         # ligament consumed: exit along segment k
# after _assign_band (law 4) and _assign_lead (law 8 plow mask), _assign_exit:
#   digit_c(p) == 0, |l| <= L, 0 < f <= b_lead, X[seg(l)], |s| <= h  ->  code |= (s >= 0 ? 2 : 1) << 2c
```

**By construction:**
- Empty-space segments never fire: they were never `seen`.
- Ligaments never fire: partial cuts, pockets, a stop short of the exit, h ≥ 0.5 mm above the board, M4, F2.7.
- Vertical exits fire but label nothing.
- Counting all codes keeps the rule's own labels from making a ligament "one-sided".
- The core wedge of a supported oblique exit is labelled.

**Must be measured:** substrate labels between bumps; carried chips at piece ends (may help s2501; not assumed); the
one-layer skin at h = 0.5 mm; field-split dynamics.

**Rejected alternatives:**
- A fixture-aware "edge on the board" trigger: it couples the law to the scene and misses free-surface exits.
- Raising h, b_back or θ_max: no effect, or a threshold relaxation.
- Law 5 everywhere: it fails M8b.

## 4. Fallback (e)

**|m×e|² ≤ max(1e-20, (1e-5|m|)²) is correct.**
- 1e-5 is about 100× the float32 noise of nearly parallel unit vectors, and 0.00057° of real geometry.
- It flips 0 recorded digits.
- It fixes generic-yaw pure draws (15 of 28 in the CPU case), which agentic control will use.

**Ship it as its own opt-in variant, "law 8 + swept-normal fallback v2".** Its acceptance:
- CPU bitwise equality with law 8 wherever no `ill_float` digit exists;
- the generic-yaw draw test;
- a matrix rerun with the recorder, predicted equal within GPU nondeterminism.

Do not bundle it into an (a) law: that would make a justified fix wait on a contested rule and confound
attribution. A later exit law builds on it with an ablation row.

## 5. (b): the rake envelope

Declaring ρ ≤ 44° is the honest choice now.

A bevel-derived gate (the edge leads the profile: |α| < 77.5° for the 25° double bevel, asymmetric for the chisel)
is better grounded than a fixed 45°, but it is only a necessary condition. Up to 77.5° it admits plowing, press-like
motion, and any dW > 0 starts a stroke. It would not rescue s2307, which is blocked by (c). Study it after (c), with
face-first controls at α = 80/85/90° and a 70° wedge press without draw.

**Envelope wording** (law 8, `ruler_knife_v2_y`, kinematic ideal tracking, 1 mm):
- **Validated cutting:** ρ = 10° (F3/C4); 14, 15, 16° (F2.1); 25° (M3a); 35° (M3b/M3c).
- **Partially validated:** ρ = 10–44° (dome sweep, diagnostic).
- **Known failure / unsupported:** ρ = 45°. The motion lies on θ_max and float32 decides the gate (s2307, 0 labels;
  the record stands).
- **Unsupported by law:** ρ > 45°. The gate is closed by design: contact and directional work, no labels.
- **Physically infeasible:** ρ ≈ 5° at 0.2 mm clearance.
- **Physically feasible, not validated:** edge-first contact up to 77.5°.
- **Sampling rule:** generators and planners sample ρ ≤ 44°. Live tracking needs an extra margin equal to the
  measured out-of-plane motion noise (NOT_RUN).

## 6. Protocol (commit as `configs/wp12/phase4_law9_protocol_v1.json` before any run)

**Common to every row.** 1 mm / 4 mm / 0.1 ms; seed 42; kinematic ideal tracking; `cuda:1`; clean worktree;
cut_diagnostics_v1 on; failures preserved. Each law row runs under law 8 and under the new law at one commit. Two
law-8 repeats set the GPU-nondeterminism band.

**E1, the decisive experiment.** Law 8 only, no law code. A read-only candidate evaluator in `cut_diagnostics`
records, per segment and substep, the proposed B_k, N⁻/N⁺, `seen`, X and both candidates' would-be labels.

- **E1.1 Option A, `bar_flat_board`.** h = 0 at τ ∈ {0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, −5, −20, −35}°;
  h ∈ {0.25, 0.375, 0.5, 1.0} mm at τ ∈ {5, 20, 35}°.
  REVIEWER prediction: law 8 fails from 5–10° (margin) and robustly from 15–20°; the proposed B_k stays 0 on ≥ 40 %
  of material segments at the end for τ ≥ 20° (≥ 75 % at 30–35°).
- **E1.2 Option D, `bar_flat_board_slotted`.** τ ∈ {20, 25, 30, 35}° at the largest overtravel the 8 mm groove
  allows. Prediction: it separates.
- **E1.3 Reachability.** Preflight/IK for every row, reported as *reachable*, never as validated.

**D1 (decided at xhigh).**
- If Option D separates every reachable oblique, (a) needs no law: route obliques to D and declare Option A a known
  failure from the measured onset.
- Only if Option A must support obliques, or the onset is ≤ 3° (which threatens live vertical cuts):
  1. run L9-EL offline on the E1 states;
  2. write ADR 0019 and the CPU tests;
  3. run the acceptance below.

**Acceptance (spec §4.3, items 1–9).**

1. **CPU tests.**
   - Laws 4, 5 and 8 are bitwise unchanged in v4 and v5, with no new launches.
   - The new law equals law 8 bitwise for: a vertical through-cut into air, a half-depth cut plus withdrawal, a skim
     inside a bar, an axis-aligned draw, a flat press, a side push and a stop.
   - It differs only by exit labels on an oblique exit.
   - The recorder stays neutral.
   - A free block across an exit gives ΔΣmv = tool impulse to 1e-6 relative.
2. **ADR 0006 matrix** (`adr0006a1_adr0008`, `rescore_phase1_matrix.py`): M1a, M1b, M2, M3a, M3b, M3c, M4, M5a, M5b,
   M5c, M6, M7, M8b, M9a, M9b; M8a is diagnostic, M9c/d report-only. All must PASS. Prediction: the M8b push stays
   within 0.1 mm of law 8.
3. **Phase F v1/v2 rows:**
   - F2.1 at 14/15/16°, F2.3, F2.4 rocking, F2.5 rotation, F2.6 saw and draw, F2.7 withdrawal_v2, F2.8 finite edge;
   - F2.9 flat press, side push, no contact and blunt_handle_v2 (the v1 FAIL stands and is reported);
   - F2.10 re-contact, F2.11 multicut, F3 CUT/TRIM formal and dt_half (characterization).
4. **New controls** (0 exit labels unless stated):
   - **N1** Engage a 3 mm fin, then pass a block 5 or 10 mm from the plane.
   - **N2** A pocket straddling the plane, stopped mid-depth: 1 component, core below the stop 100 % code 0.
   - **N3** A stop near the board at τ ∈ {0, 10, 20, 30}°: at h = 1.0 mm it stays 1 component; h = 0.5 mm is reported.
   - **N4** An oblique stroke stopped 1 or 3 mm before the exit, then an armed stop and withdrawal: 1 component.
   - **N5** Cut 1 across an open cut-0 kerf with gaps of 0, 2 and 10 mm: 3 parts; far-piece labels reported against law 8.
   - **N6** A bumps3-type skim at ρ = 10° with the edge ≥ 5 mm above the substrate between bumps: every bump-core
     particle assigned at f ≤ 0.
   - **N7** Rocking and rotation through the exit: no labels outside the swept region beyond law 8's.
   - **N8** An armed stop and an armed withdrawal at the exit: labels unchanged.
   - **N9** Flat press, side push and blunt_handle_v2 on the flat-board scene: 0 labels, 1 component.
5. **Oblique families.**
   - Rows: E1.1 and E1.2; the 4a slot sweep (obl_00 … obl_45, obl_m35, ot06/08/10, slot12, nograv,
     phase025/050/075); new reproductions of s2201/2202/2203, Test A W s2403 and M2 (frozen records untouched).
   - Predictions: Option A separates for |τ| ≤ 35° at h ∈ [0, 0.375]; the slot rows separate at 25–45°; τ = 0 equals
     law 8.
6. **CUT and TRIM non-regression.**
   - Rows: CUT-01-R, CUT-02, CUT-03-K2, CUT-04-K2-W60; TRIM-01-R at 25 and 35°, TRIM-02-K2-R10; the dome rake sweep
     10–44° and the §7.3 phase sweep; Test B (solid and slotted) and Test C; TEST_D2 s2501, bumps3, TRIM_W s2307
     (0 labels expected); the WP-12 through-cut window rows S_P100 … D40_000 (S_P050 and S_P100 stay connected).
   - Criteria: frozen criteria unchanged, no verdict worse, metrics within the law-8 band.
7. **Energy, reaction, momentum** against law 8:
   - kinetic + elastic energy and cumulative cutting work on M1a, M2, M8b and E1 at τ = 20/35°, with no step at the
     exit beyond the band;
   - tool reaction within the band up to the first exit label, ΔF reported after it;
   - piece COM velocities within the band;
   - min det F ≥ 0.85, and tool penetration unchanged.

## 7. Next steps

1. **Amend the 4a draft** (implementer, high).
   - Add the §2 corrections. Mark (a) on the flat board NOT_RUN, with this probe cited as a prediction.
   - Withdraw the §8 law-9 recommendation and the "impossible / bind on handle clearance" wording.
   - Add the §5 envelope and the (e) scope limit.
2. **Commit the §6 protocol, then run E1** with the read-only evaluator (rcs-impl). No law code until D1 is decided
   at xhigh.
3. **Ship (e)** as a separate versioned numerical variant (§4).
4. **Study (c) and (d) separately**, with the §4.7 resolution study. The (b) bevel gate comes only after (c).

**Reviewer commands** (repository root; scripts and outputs in `artifacts/wp12/phase4_xhigh_review_001/`):
- `.venv/bin/python -I check_a.py | check_a_time.py | check_b.py <runs>`
- `geom_flat.py`, `geom_ot.py`
- `cpu_flat_oblique.py <tau_deg> <overtravel_mm> <out.npz>` (10 rows in parallel on CPU, 82–114 s each)
- `relabel_cpu.py cpu_out/*.npz`
