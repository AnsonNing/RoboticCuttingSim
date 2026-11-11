# K2 — combined xhigh review of the Phase 2 rubber line (`025ecaf` + `c9d771f` … `77f5a99`)

Date: 2026-11-11. Reviewer: rcs-review (xhigh), task K2 of ADR 0023 §23.4 / `P2_REMAINING_PLAN.md` §7. Reviewed tree:
`claude/rl-rubber` @ `77f5a99` (detached review worktree, no code edited). Scope: `git log 025ecaf..77f5a99` minus the
already-reviewed `c9d771f`, i.e. the R1/R4 fix and R2/R3 wording, protocol v5, P2-4 (laws v1 / v2 / v3, protocols
`rubber_contact_tests_v1..v3`, B1, B2 / B2-v2 incl. the -09 v2 rule, X1, X2, X3), the K1 merges with the §19A
renumbering, S1 (`rubber_selfpen_check_v1`) and the plan (ADR 0023 §23, `P2_REMAINING_PLAN.md`). Material status
everywhere: **SIMULATOR_ONLY_UNCALIBRATED** (unchanged; no recipe or calibration file touched).

Method: code and config reading with file:line evidence; timestamps of every declaration commit against the
`meta.json` / `run_meta.json` start times of the formal runs; recomputation from raw outputs under
`/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/{p2_2_v5,p2_4,p2_4b,p2_s1,p2_k1}`; kernel-source hash pins
recomputed at every relevant commit; replay of every K1 merge; a trial merge into the integration branch; the rubber and
batched CPU suites with `CUDA_VISIBLE_DEVICES=''` and `=0`; CPU physics probes (DIAGNOSTIC, scripts in the review
scratchpad, §7). No GPU job was run; GPU1 was not used. Three sub-reviews (B1/B2, protocol v5, S1) ran in parallel at
xhigh and their findings were checked against the code before being adopted here.

## 0. Verdicts

| Package | Verdict |
|---|---|
| R1 fix (`22982df`), R4 test, R2 / R3 wording (`92e97aa`) | **PASS** |
| Protocol v5 (`8881ff0` … `d814567`) | **PASS_WITH_LIMITATIONS**: declared before its runs, run and evaluated as declared, every verdict recomputed from raw; -11 v5 FAIL (Q1w) honest and preserved; the Q1w label is not implemented yet (K2-2), small record gaps (K2-9…K2-11) |
| **Q1w decision** | **Accept option (c)**: n_t 4 stays the production grid with its measured +0.62 % wide-strip sag bias as a labelled limitation; the v5 -11 FAIL stays recorded; no v6 rule. Conditions in §3 |
| P2-4 law v1 + `contact_v1_001` | **PASS** (as a record: declared before the run, every FAIL preserved) |
| P2-4 law v2 + `rubber_contact_tests_v2` (X1) | **PASS_WITH_LIMITATIONS** (real defect corrected; -05 d metric changed post hoc, K2-5; -01 b / -04 at μ 0.8, -01 d, -05 a FAIL preserved) |
| B1 batched contact path | **PASS** |
| B2 / B2-v2, RUBBER-CONT-09 and its v2 rule | **PASS_WITH_LIMITATIONS** (the v2 rule corrects a real statistical defect; its "energies summary" observable is vacuous, K2-3) |
| X2 negative result, withdrawal of the X1 claim | **PASS_WITH_LIMITATIONS** (honest; the withdrawn statement still stands uncorrected in §22.9 and in the X1 evidence, K2-6) |
| X3: law v3 + `rubber_contact_tests_v3` | **PASS_WITH_LIMITATIONS for translational sliding only.** The formal run is honest and passes as declared (except the declared -01 d FAIL), but the equalisation is physically wrong for any non-translational slip (rotation about the contact normal, in-plane bending of a dragged strip): K2-1 (MAJOR). v3 is **not accepted** as the P2-6 / ENV1 law in its current form |
| K1 merges and the §19A renumbering | **PASS** |
| S1 self-penetration detector | **PASS_WITH_LIMITATIONS**: correct, opt-in, isolated, declared before its run; documentation errata and weak GPU TRIP margin (K2-12…K2-15) |
| Plan (ADR §23, `P2_REMAINING_PLAN.md`) | **PASS_WITH_LIMITATIONS**: decisions B3/C1 deferred, P1 deferred, S1 required, P2-5 and P2-6 designs and the Native host are sound; decision 4 (μ envelope = v3 → 0.8 for P2-6) must be amended per K2-1, and the Done map must carry it |
| Frozen files, byte-identity claims, CI safety, test suites | **PASS** (§6) |

**Merge: GO** for `claude/rl-rubber` @ `77f5a99` into `claude/rl-ready-rubber-physics` (trial merge onto its tip
`502dd8f` = `1539788` + one progress-doc commit: clean, no conflict, 78 files, suites green, §5). Everything new is
opt-in, nothing frozen changes, and every FAIL is preserved. The merge carries conditions that bind the next tasks
(ENV1 / ENV2 / P6.0), not the merge itself (§5).

## 1. Findings (ranked)

### MAJOR

**K2-1 — law v3: the patch slip-rate equalisation is a full per-substep projection onto the patch-mean slip velocity;
it exerts an unbounded, μ-independent torque and suppresses every non-translational slip mode.**
`src/robotic_cutting_sim/rubber_contact.py:1046-1062` (pass 2), `:983-989` (patch sums), `:64` (ζ_d = 1.0), `:70-73`
(κ_t = 0.25).

- Mechanism. c_d = 2ζ_d√(k_t m_I) with k_t = κ_t m_I/dt², so c_d·dt/m_I = 2ζ_d√κ_t = 1 at the declared parameters (the
  code comment at `:64` states the product). The impulse −dt c_d (v_t,i − v̄) = −m_I (v_t,i − v̄) therefore removes, in one
  substep, each slipping node's whole deviation of realised tangential velocity from the patch mean. Its sum over the
  slipping set is zero, so the patch force stays ≤ μΣF_n (verified), but its moment is not bounded by Coulomb and does
  not scale with μ (it acts for any μ > 0). A patch is a whole obstacle face (the entire table plane of an env).
- Spin probe (DIAGNOSTIC; free block 16 × 16 × 4 mm resting on the table after a 40 ms settle, then spun at
  ω0 = 10 rad/s about the table normal; rigid-Coulomb reference with uniform pressure):

  | μ, n_t | ω after 1 ms: v2 / v3 / rigid Coulomb (rad/s) | mean table torque in that ms: v2 / v3 / Coulomb (N·m) |
  |---|---|---|
  | 0.05, 3 | 9.911 / **−0.389** / 9.930 | 3.45e-6 / **5.28e-4** / 3.5e-6 |
  | 0.5, 3 | 9.297 / **−0.552** / 9.296 | 3.48e-5 / **5.37e-4** / 3.5e-5 |
  | 0.8, 4 | 8.867 / **−0.579** / 8.874 | 5.72e-5 / **5.36e-4** / 5.8e-5 |

  v2 follows the rigid Coulomb torsional decay (within ≈ 1 % over the first 10 ms at μ 0.5 and ≤ 0.2 % over 11 ms at
  μ 0.05; at μ 0.5 it stops at ≈ 17 ms vs 14.2 ms rigid, the elastic block's tail). v3 stops the
  spin inside the first control step at every μ and **reverses it** (friction may not reverse the sliding direction), with
  a torque 9× (μ 0.8), 15× (μ 0.5) and 150× (μ 0.05) the Coulomb bound; the normal load jumps to 1.5× the weight and then
  drops to 0.15× (the block bounces). No invalid flag is raised.
- Drag probe (DIAGNOSTIC; the P2-6 motion class: case-A strip 60 × 8 × 4 mm, n_t 3, lying on the table, μ 0.4 — inside
  the formally scored v2 envelope at n_t 3 — jaw dragged sideways at 0.1 m/s for 80 ms): with v2 the free end lags and
  swings (tip 1.6 mm behind the jaw at 11 ms with the tip moving backwards, 2.75 mm at 26 ms, then overtaking by 0.3 mm:
  in-plane bending and yaw). With v3 the strip translates almost rigidly from the first milliseconds (tip lag 0.49–0.56 mm,
  constant; the tip never moves backwards).
- Why the protocol did not see it. `rubber_contact_tests_v3` scores only translational slip (INC / INC2 incline, PINCH
  pull-out). The "exact" results (-01 b ≤ 1.8e-7 of g cosθ, -04 0.9999–1.0000) are the rigid translation the projection
  imposes. ADR §22.11 declares (i) suppressed slip waves and (iii) "a net torque remains (reported in the wrench)"
  (`docs/adr/0023-rubber-material-v1.md:1991, 2047-2051`); neither says that rotation and in-plane bending in slip are
  removed, nor that the torque is unbounded, μ-independent and can reverse the motion.
- Consequence. The P2-6 task (`P2_REMAINING_PLAN.md` §4.1, §4.3) drags a strip on the table with yaw actions
  (≤ 0.05 rad per step), a preload phase with lateral sweeps ±30 mm and yaw ±0.4 rad (`:352-353`) and a tip goal; the
  Native episode lifts with a 0.3 rad yaw and holds with tip contact (`:473`). This is exactly the non-translational
  slip regime; with v3 its tip dynamics would be unphysical. ADR §22.11 "v3 is the contact law for scored rubber
  contact rows with μ ≤ 0.8 … P2-6 / RL rubber contact scenes should build v3 explicitly" (`:2036-2045`), report §11.4
  (`P2_4_RUBBER_CONTACT.md:584`), ADR §23.2 decision 4 (`:2122`) and the progress default U-P2-3 on the integration branch
  overstate the established envelope.
- Required (before ENV1 / ENV2 / P6.0 are declared):
  1. Record that v3's μ ≤ 0.8 envelope holds for **translational sliding only**; v3 rows with rotation about the normal,
     in-plane bending in slip, or several contact regions on one face are outside it (label, e.g.
     `contact_slip_rate_equalised`).
  2. Do not map v3 → μ_slide,max 0.8 for P2-6 in `contact_envelope_v1.json` until a declared protocol scores
     non-translational slip: (a) torsional spin-down of a block against the rigid Coulomb torque ∫μ p r dA (and the v2
     value at low μ); (b) a lateral strip drag against a converged / independent reference; (c) two separated contact
     regions on one plane sliding at different rates. The plan's contingency (v2, μ ≤ 0.4 after ENV2 at n_t 4–6, else
     the friction axis BLOCKED, `P2_REMAINING_PLAN.md:189-192`) applies until then; ENV2 must include these scenes for
     whichever law is chosen.
  3. A v4 candidate should at least equalise only the deviation from the patch's rigid in-plane motion (translation plus
     spin about the normal, which also removes the net torque) and bound the strength (2ζ_d√κ_t = 1 is a full projection);
     it would still damp in-plane bending in slip, so (b) remains necessary.

**K2-2 — Q1w requires a label that is not implemented yet.** `src/robotic_cutting_sim/rubber_material.py:35-37`:
`rubber_ref_v1` has no `measured_bias` / `known_limitation` block (only `rubber_rl_v1` quotes the n_t 4 numbers "for
comparison", `:50-51`). The accepted option (c) depends on it (§3). Implement and test before any P2-5 / P2-6 declaration.

**K2-3 — RUBBER-CONT-09 b: the "energies summary" observable cannot fail.** The v2 band floor is
`eps_rel · max|A|` over the whole 18-column summary vector (`scripts/rubber/p2_4b/gpu_contact_batched_check_v2.py:70`,
observables `:164`), which contains the integer columns `particles` (41,220) and `cells` (5,175)
(`tlmpm_rubber_v1.py:60-62`): the floor is ≈ 0.04 in mixed units, 10²–10³ × the elastic and kinetic energies of the
out-of-reach env (sub-review probe). 324 of the 1,566 "passing" scalars are therefore uninformative in
`gpu_cont09_v2_001` and `gpu_cont09_v3_001` (and in v1). The PASS stands — the markers (floor ≈ 1.4e-7 m), the clamp
wrench (≤ 9e-7 N), criterion a (exact zero rows) and c (audits), and the CPU L1 byte tests carry the isolation claim —
but report §7.3 / §9 ("energies … pass everywhere", `P2_4_RUBBER_CONTACT.md:262, 425`) overstate it. Add an additive note;
the next declared -09 version uses a per-quantity floor without count columns.

### MINOR

**K2-4 — X3 task requirement "plus an unseen geometry" not met.** ADR §22.10 task X3 (`:1964`) asked for scored
sliding μ ≤ 0.8 "plus an unseen geometry". INC and INC2 were both used in the X2 mode-13 development; the v3 protocol's
`provenance` lists this honestly (unseen: μ 0.2 / 0.4 rows, INC2 n_t 4 at μ 0.2 / 0.4 / 0.8, PINCH, PEN / IMP). Add an
unseen strip-on-table geometry to ENV2.

**K2-5 — RUBBER-CONT-05 d changed after the v1 FAIL.** v1 scored the per-step |F_x(clamp) + T| ≤ 5 % |T|; v2 / v3 score
the window-mean (momentum) balance at the same 5 % and report the v1 metric (`configs/rubber/rubber_contact_tests_v2.json`,
-05 `d_balance`, `d_reported_v1_per_step`). The analysis (axial quarter-wave vibration of the undamped strip, unattainable
for the exact solution) is plausible and the old metric is reported (v3: 8.4–13.8 %), but it is the same class as
VISC-03 / -05 of the c9d771f review. Cite -05 as "v1 FAIL, v3 PASS (post-hoc metric correction of d; the v1 per-step
metric 8.4–13.8 % reported)" in the ADR §22.11 table, report §11 and the evidence (additive).

**K2-6 — the withdrawn X1 claim is uncorrected at its source.** ADR §22.10 withdraws "defect 2 is not continuum physics
below μ = 1" (`:1922-1923`), but §22.9 (`:1815-1817`) and `docs/evidence/rl_ready/p2_4_contact_x1_001.json`
(`root_cause`) still assert it with no pointer. Add an erratum line in §22.9 (original kept) and an additive
`review_corrections` block in the X1 evidence.

**K2-7 — -09 records.** (a) The v3 protocol (`rubber_contact_tests_v3.json:441`) copies the stale "≤ 648 scalars,
family-wise ≤ 0.4 %" after the FWER finding of report §9 (1,566 scalars, 0.86 %); ADR §22.9 (`:1860-1867`) has no erratum.
(b) The ≤ 5.5e-6 per-scalar false-reject rate holds for the four continuous noise families simulated; two-point
(atomic-order) noise gives ≈ 1.9e-3 (sub-review Monte Carlo); "a shift > 5 s is always rejected" omits the floor term.
(c) `gpu_label` counts compute apps on every GPU (`gpu_contact_batched_check_v2.py:197-198`): PID 3468960 of
`gpu_cont09_v3_001` ran on GPU1 but is cited as a GPU0 co-tenant (`P2_4_RUBBER_CONTACT.md:574`, ADR `:2025`); SHARED_GPU
still holds (PID 642659 on GPU0). (d) Report §7.3 has no forward pointer to the §9 co-tenant erratum. Additive errata.

**K2-8 — determinism statements wrong under v3.** `batched/rubber_contact_batch.py:24-25` and ADR §22.5 (`:1728-1730`)
say per-node contact state / the contact kernel is deterministic; under v3 node velocities depend on float64 atomic
patch sums (`rubber_contact.py:986-989, 1056-1058`). ADR §22.11 (v) says these are "inside the -09 band", but the -09
configuration never scores an env that has patch sums (envs 0 / 2 / 3 are out of reach). Correct the wording; a later
-09 variant should perturb one env while every env is in contact.

**K2-9 — batched -04 v5 ran on a dirty tree.** `batched_04_v5_001/meta.json`: `tree_dirty: true`, `dirty_paths` not
recorded (`scripts/rubber/p2_2_v5/run_batched_04_v5.py:64`). The report attributes it to the untracked evidence writer;
the sub-review confirmed that all 14 Warp kernel-module hashes of a clean `git archive 1bdef79` tree equal the run log and
that env 0's tip series is byte-equal to the clean formal dt run. Host Python cannot be proven unchanged. Record dirty
paths (as the workload E runner does) or refuse dirty trees.

**K2-10 — v5 provenance gaps.** (a) The command that produced `rubber_phys_v5_001/summary_final.json` is not recorded
(report v5.6 describes it); it is reproduced exactly by re-evaluation at HEAD and differs from `summary.json` only in -04
(NOT_RUN → PASS for `valid` / `batched_equal`). (b) The workload E v2 GPU rows (22:07:57–22:09:30) were measured while the
40 CPU workers of the formal runs were active (started 22:06:45); not disclosed (rows are reported, not scored).

**K2-11 — v5 label details.** `rubber_rl_v1.measured_bias` quotes "static_sag_L_over_t_10: ≤ 0.5 %" without the width
(the -02 data put n_t 3 at L/t 10, W = 4t at ≈ +1.1–1.3 %) and gives Q4 signed values although the (4, 6, 8) triplet is
OSCILLATORY (`rubber_material.py:45, 49-52`). The -09 `guard_before_nonfinite` PASS is vacuous (no run became non-finite
in its 5 steps); the real evidence remains -12 v3. -08's reserve r = 1.5 is not a production default (r stays 1.0):
large-shear scenes must declare r by the rule or they will fail loudly with the CFL flag.

**K2-12 — S1 test (a) numbers quoted in the ADR / report are not what the test produces.** ADR §23.5
(`docs/adr/0023-rubber-material-v1.md:2246-2249`) and report §12 (`P2_4_RUBBER_CONTACT.md:607`) say: brute-force overlap
at control step 137, substep 42; flag at substep 48. Re-running the test body at `77f5a99` gives overlap (137, 70) and
first hit (137, 72); the CPU row of `gpu_selfpen_001` agrees (`single_cpu.detector.first_hit_substep` 72). The conclusion
(hit at the next check) holds. Additive erratum.

**K2-13 — S1 GPU TRIP has almost no margin and no geometric reference.** The single-env GPU hit is at control step 139
of `FOLD_STEPS = 140` (`tests/test_rubber_self_penetration.py:42`, used by the GPU script); CPU perturbations of 1e-7 in μ
or 1e-9 m in the grip move the overlap between steps 137 and 139 (the detector always flags within one interval of it),
so the GPU spread (139 single, 136 batched, 137 CPU) is trajectory sensitivity, and a re-run could miss step 140 for a
reason unrelated to the detector. Next check version (not an edit of v1): feed identical position sets to the CPU and
GPU detectors and require equal integer hit counts, and lengthen the fold run to ≈ 160–180 steps.

**K2-14 — S1 cost row covers a flat strip only.** The hash dims come from the reference extent
(`rubber_self_penetration.py:222-224`, [257, 35, 10] for the P2-6 strip); a model of the hash gives ≈ 9 candidates per
query flat, ≈ 18 folded flat and ≈ 236 (max 389) for a lifted vertical strip (modulo aliasing in z). The +0.48 % at N 8
vs −0.05 % at N 1 (≈ 1.7 ms per check at N 8) is not explained by sort + query. RL-07 must measure the S1 cost with lifted
/ folded strips and at N 16; consider sizing the hash from the maximum extent per axis.

**K2-15 — S1 tests (b) and the λ 1.5 release trip.** Test (b) uses R = 3t and 5t (inner gaps 21 s and 41 s), far from the
decision boundary; a probe confirms the "legal bends cannot trigger" claim down to R ≈ 0.55t at n_t 4 (flags at
R ≤ 0.49t, closed U), so add R = 0.6t and R = t cases. The λ 1.5 release "trip" (`test_rubber_self_penetration.py:199`)
happens at control step 4 together with J_DOMAIN and CFL at v_max 25–33 m/s: it is a trip in an invalid state, not
fold-detection evidence; relabel it (the t 4 mm CPU trip is also unreported; the GPU non-trip is reported honestly).

### NOTE

- The GPU -09 evidence ran at `2b8cb9c` / `405bf54`, before the R1 refresh and the S1 hooks; by code reading the -09
  launch path is unchanged (S1 off launches nothing; R1's refresh is idempotent before `advance`), and CPU L0 passes at
  `77f5a99`. No GPU -09 run exists at the integrated head (RL-04 will provide one).
- `229f070` rewrote the `report` pointer field of `p2_4_contact_x2_001.json` in place (section renumbering; no result
  changed). Prefer additive notes for evidence files.
- The v3 GPU wrapper rewrites `result.json` after the run, keeping the old label as `label_inherited`
  (`scripts/rubber/p2_4/gpu_contact_batched_check_v3.py:45`); the dirty-tree flag ignores untracked files
  (`gpu_contact_batched_check_v2.py:319`; only the smoke `gpu_cont09_v3_smoke_001` was affected).
- `node_penalty_contact_v2`'s docstring still calls the normal force "the conservative penalty force" although it
  carries the dashpot (`rubber_contact.py:575-577`); the code is right.

## 2. Package notes

### 2.1 R1 / R4 / R2 / R3 (`22982df`, `92e97aa`) — PASS

- R1: `begin_control_step()` and `reset_masked()` now call `prepare_motion_table()`
  (`src/robotic_cutting_sim/batched/rubber_material_batch.py:510-515, 766-768` at `77f5a99`); `advance()` calls the internal
  `_begin_control_step()` and then `set_group_motions()`, which refreshes the table, so the launch sequence of the
  `advance()` path is unchanged. The table is a pure function of each group's q0 / q1 / b / A
  (`batched/rubber_batch.py:463-469`), so refreshing it on a partial reset gives the envs that were not reset the same
  values. `tests/test_rubber_motion_table_refresh.py` (11) covers begin / full / partial / mid-table resets with direct
  steps and the byte equality with `all_nodes_v1`.
- R4: `tests/test_rubber_visco09_default_options.py` compares every frozen device array, the wrench and the summary of
  `TLMPMRubberVisco` (M = 0; M = 2 with g = 0) against the frozen `TLMPMRubber` with the default options, guard on, for
  cases A / C / block — meaningful (it asserts the option tuple and that the scene moved).
- R2 / R3: additive (`p2_3_visco_002.json` gains a `review_corrections_c9d771f` block; no value rewritten); ADR §21.7.7
  and the P2-3 report use the required wording.

### 2.2 Protocol v5 — PASS_WITH_LIMITATIONS

Order: declaration `8881ff0` (2026-11-10 21:58:10, config alone, sha256 `7e02f57b…67f0`) → fixes and runners
`687f4fe` / `1bdef79` (22:06:30) → runs `refinement_v5_001` 22:06:45 and `rubber_phys_v5_001` 22:06:47 at `1bdef79`,
clean, config SHA recorded (reflog of the v5 worktree shows no rebase or amend). A structural diff of v4 and v5 shows
only the declared changes; every threshold and the -11 rule block are identical. Every verdict was recomputed from the
raw outputs with independent code (sub-review): -11 production GCI Q1 1.4845 %, **Q1w 2.1604 %** (p_obs 3.1236 → Fs 3;
FAIL), Q2 0.5591 %, Q4 0.0716 % (oscillatory); n_t 3 biases +0.97 / +1.39 / −0.83 / +0.08 %, n_t 4 +0.39 / +0.62 /
−0.39 / −0.05 %; -08 CFL_eff,max(r = 1) 1.296571 → r 1.5 → CFL_eff 0.86438, no flag (a step-size fix declared before the
run, not a threshold change); -02 ≤ 0.725 %; -03 4.0077 / −2.0082; -05 0.9939–0.9995; -04 f1 +4.8e-5 with the batched
N = 4 CPU L0 compared byte-wise (dtype, shape, bytes of every array, 312 control steps). `summary_final.json`
(03:30) differs from `summary.json` (01:56) only by merging the batched -04 rows (NOT_RUN → PASS) and is reproduced by
re-evaluation at HEAD. v1–v4 evaluations still reproduce at HEAD; defaults unchanged (guard None, r 1.0). F9: only
`rubber_refinement._spec` sets `quadrature_check` (`rubber_refinement.py:43-45`); no production path uses ppc 3.

Limitations: K2-2 (Q1w label), K2-9 (dirty batched -04 run), K2-10, K2-11.

### 2.3 P2-4 contact

**Order (all declarations before their runs, clean trees, SHAs recorded):**

| Protocol | Law commit | Declaration | Runner | Formal run start (meta) |
|---|---|---|---|---|
| `rubber_contact_tests_v1` (`4d3dc68f…`) | `259f2b6` 04:14:32 | `51023d6` 04:17:33 | `436f91c` 04:22:22 | `contact_v1_001` 04:22:24 (commit `436f91c`, dirty false) |
| `rubber_contact_tests_v2` (`89bc7680…`) | `0b33246` 05:30:14 | `8842573` 05:32:28 | `ffb099f` 05:33:31 | `contact_v2_001` 05:33:44 (`ffb099f`, dirty false) |
| -09 v2 rule (in v2) | – | `8842573` 05:32:28 | `2b8cb9c` 05:51:47 | `gpu_cont09_v2_001` 05:52:31 |
| `rubber_contact_tests_v3` (`36d70bea…`) | `b668bdc` 18:14:18 | `e19f488` 18:16:09 | `c631cca` 18:17:52 | `contact_v3_001` 18:17:56 (`c631cca`, dirty false) |
| `rubber_selfpen_check_v1` (`cc5cf333…`) | `a1bdeac` 18:50:44 | `5effd81` 18:51:44 | `bca572d` 18:53:27 | `gpu_selfpen_001` 18:53:32 (`bca572d`, dirty false) |

**Did each change correct a real defect?**
- v1 → v2 law: yes. Defect 1 (dashpots on the predicted lumped velocity while the node drifts with Bπ) is a real
  modelling defect, shown by a frictionless elastic block at rest that rattles with positive normal contact work; v2
  settles it. Parameters were not re-tuned (κ 0.25, ζ 0.5).
- v1 → v2 criteria: thresholds verbatim (recomputed: -01, -02 … -05 identical; v2 adds -02 f and the -09 rule). -05 d
  changed from the per-step to the window-mean balance after the v1 FAIL (K2-5: plausible, old metric reported, needs
  the post-hoc wording). The μ envelope was declared from diagnostics and the μ 0.8 rows stayed scored (and FAILed) —
  narrowing by labels, not by removing rows.
- -09 v1 → v2 rule: yes. With one repeat the v1 band is a coin flip per noisy scalar (P = 1/2 under H0 by
  exchangeability); the v1 FAIL (1 of 108 rows: clamp wrench 1.23e-7 vs 9.2e-8 N) is inside the run-to-run range the v2
  run measured on the same scalar (up to 2.05e-7 N), criteria a / c passed, and CPU L1 is byte-exact. λ 4 / k 5 come
  from a synthetic Monte Carlo committed before the v2 declaration (output 05:23, script `e79b995` 05:27) and are read
  from the pinned protocol. Limits in K2-3, K2-7.
- v2 → v3: thresholds verbatim, -01 f added (a tightening), scope widened (μ 0.2–0.8 × n_t 3 / 4 × INC / INC2, P2 holds
  at 0.6 / 0.8, pinch at n_t 3 / 4); μ 1.0 was never scored and stays reported. The provenance block lists honestly
  what was seen. The v3 PASS is honest **for the scenes it scores** — see K2-1 for what those scenes cannot see.
- X2: an honest negative result; it withdrew the X1 continuum claim (K2-6 for the missing pointers) and qualified the v2
  envelope as resolution-dependent (n_t 3 only).

**v3 physical soundness (asked explicitly):**
- Net zero per patch: **yes** — c_d ∝ m_I and v̄ is the mass-weighted mean of the same slipping set (pass-1 flag and the
  same realised v_t are reused in pass 2; plane / box-face patches have one normal), so Σ impulses = 0 (the kernel test
  measures 2e-23 of 1e-7).
- Friction ≤ μΣF_n: **yes, for the patch force.** Not for the patch moment: K2-1.
- Net-torque limitation: **understated** — unbounded, μ-independent, reverses spin (K2-1).
- Suppressed slip waves: declared, and true; the effect is not confined to the substep scale — the projection acts
  every substep and removes macroscopic rotation and in-plane bending in slip (K2-1).
- v1 / v2 kernels: byte-identical — recomputed source hashes of the 10 v1 kernels and of the v2 kernel at `259f2b6`,
  `436f91c`, `0b33246`, `ffb099f`, `b668bdc`, `77f5a99` all equal the pins of `tests/test_rubber_contact_v2.py` /
  `_v3.py`; the v1 / v2 dispatch in `tlmpm_rubber_contact_v1.py:184-204` is unchanged, v3 is an added branch
  (`:161-183`). v1 stays the default (`rubber_contact.py:61`).

**μ envelope at the acceptance profile:** formally established for v3 at n_t 3 and 4, μ ≤ 0.8, **translational sliding
and static holds** (INC, INC2, PINCH). Not established at n_t 5 / 6 (ENV2, as the plan says) and not established for
non-translational slip (K2-1). For v2 the formal envelope remains n_t 3 only (sliding 0.6, static 0.8).

### 2.4 B1 / B2 — B1 PASS; B2 / B2-v2 PASS_WITH_LIMITATIONS

One process, batched kernels: the contact kick is the single solver's own `_launch_contact` bound to the batch
(`batched/rubber_contact_batch.py:212`), one launch over all N envs' nodes; no Python loop over envs inside a substep
(host loops only in `read()`, `snapshot()`, `set_initial_velocity`). Per-env isolation: node rows; env rows
`ct_work[N,6]`, `ct_pen_max[N]`, `ct_counts[N,3]`, `ct_energy[N,2]`; obstacle rows `e·P+k`, `(e·F+f)·B+b`, patch `e·Q+q`;
frictions are a strided view of per-env `params[:, 3:5]` written in place on reset (graph-safe); every atomic targets
a row of `env_of_node[i]`. Masked reset touches only reset envs (`rubber_contact_batch.py:378-396`; byte tests under v1
and v3). L0 is byte-wise (dtype + shape + bytes of every base, option, visco and contact array, diagnostics, markers,
wrenches): v1 / v3 N 1 and N 3 heterogeneous for incline, impact, pinch, pinch P2; v2 N 3 (declared). The v1 FAIL of
`gpu_cont09_001` is preserved (report §7.3, ADR §22.9, evidence `p2_4_contact_batched_001.json`, unchanged since
`f1ece86`); the co-tenant erratum is honest (`meta.pid` 3093155 was the run itself). Spot-checked numbers match the raw
outputs (v2: 1,566 scalars, 0 failing, 720 with s = 0, worst d / limit 0.339; v3: 1,566, 0, 0.345; env 1 under the same
rule: 95 of 522 rejected — the perturbation is detected). Cost rows state N, 41,220 particles / env, dt 7.74e-6 s, 2,585
substeps, no camera, 1 cold + 3 warm, graph chunk 500, SHARED_GPU, commit, dirty flag and protocol SHA. Limits: K2-3,
K2-7, K2-8; the GPU -09 scene never perturbs an env while the receiving env is itself in contact (CPU L1 pinch tests
cover that).

### 2.5 K1 merges and §19A — PASS

Each K1 merge (`efe3833`, `d7af304`, `d6ebed8`, `2d8da80`) was replayed (`git merge --no-commit` of the second parent
onto the first) and compared with the committed result. Conflicts were only in the ADR, the contact report and the
plan; the resolutions keep every line of both parents except (a) "§21" → "§19A" in the v5 text, errata and comments
(present in the merged ADR at `:86, 208, 242, 383`), (b) "X3 running" → "done" status lines, (c) the P2-3 "PASS" lines
replaced by the R2 wording that `92e97aa` introduced, (d) the 9A / 9B report headings renumbered 9 / 10. Non-conflicting
edits made inside `d7af304` are comments, an additive `k1_note` in `p2_2_v5_001.json` and a note in the v5 report
section (`rubber_material.py`, `rubber_phys.py`, `run_batched_04_v5.py`, `write_p2_2_v5_evidence.py`,
`test_rubber_phys_config.py` docstring). §19A equals the old §21 apart from the heading, the numbering note and the
subsection numbers; the SHA-pinned v5 configs keep "section 21" (noted in §19A). Regenerating the v5 evidence with its
writer reproduces the committed file except the `k1_note` (sub-review).

### 2.6 S1 — PASS_WITH_LIMITATIONS

Code `a1bdeac` (`src/robotic_cutting_sim/rubber_self_penetration.py`, hooks in `live_rubber_material.py` and
`batched/rubber_material_batch.py`), declaration `5effd81` (alone, sha256 `cc5cf333…8b8bc0`), script `bca572d`, run
`gpu_selfpen_001` 18:53:32–18:57:21 on a clean tree; evidence equals the raw `result.json` field for field; the cost row
states N, 65,024 / 19,664 particles, dt, 5,226 substeps, no camera, cold / warm and SHARED_GPU.
- Correctness: the detector matched an independent brute force (cKDTree with the float32 reference exclusion) in 52
  cases — x and x64, N 1 and N 3 envs overlapping in space, random clouds inside and far outside the reference extent,
  negative coordinates, pairs planted at cell corners (0.9999 s found, 1.0001 s not). Envs never mix (env in the key);
  pairs counted once except below ≈ −2^20 cells (≈ −524 m), where neighbouring cells collapse and a pair is counted 3×
  (nothing missed; the "once" docstring at `:94-96` is not strictly true). `hits` is int32 summed over checks and can
  overflow in long runs with massive overlap (bit 8 stays set).
- Opt-in and byte identity: `off_v1` builds no detector, array or launch; with it on, only the detector arrays and the
  invalid-reason array are written (solver state byte-identical, tested); the graph chunk alignment (500 → 256) only adds
  replays — check positions equal the eager positions in all 9,360 (substeps, interval, chunk) combinations, and the
  substep index lives on the device; the warm-up check before capture is cleared by the following reset.
- Isolation / reset: bit 8 is ORed before `_validity_b` in the batched owner (`rubber_material_batch.py:584-588`), so
  `invalid_count` and strict mode see it; the masked reset clears only reset envs (`:762`); mixed options are refused
  (`:426-432`). Tests (c), (d), (f) are not vacuous (envs overlap in space; x64 in the snapshot).
- The blind-interval counter uses the end-of-step v_max and each env's own thickness (the plan says t_min) and ignores
  W < t; a stacked-leg probe found no partially overlapping configuration that is invisible, so the real blind mode is a
  full pass-through between checks, for which t − s is conservative.
- The COST arms run in ABBA order while the declaration says "on, off, on, off" (`gpu_selfpen_check.py:144` vs
  `rubber_selfpen_check_v1.json` line 26) — harmless, undeclared.
- Limits: K2-12…K2-15. Not verified (no GPU job in this review): CUB sort inside the captured graph and the GPU on / off
  behaviour beyond the recorded run.

## 3. Q1w decision: accept option (c), with conditions

**Decision.** `rubber_ref_v1` (n_t 4) stays the production grid for bending-dominated scenes, with its measured wide-strip
sag bias as a labelled limitation. RUBBER-PHYS-11 v5 stays **FAIL** in every record and summary; no v6 rule is declared;
Phase 2 Done box 2 reads PASS_WITH_LIMITATIONS and names this item. This confirms the plan recommendation (ADR §23.2
decision 5, plan §5) and the provisional default U-P2-2 of the integration branch.

Why. The FAIL comes from Roache's safety factor, not from a large error: p_obs 3.12 lies just outside the verified
window [1, 3], so Fs jumps from 1.25 to 3 (with Fs 1.25 the GCI would be 0.90 %). Convergence is monotone over four grids
with a consistent order ≈ 3 in both triplets (3.12 for (4, 6, 8), 2.97 for (3, 4, 6)); the n_t 4 bias is bracketed by
+0.50 % (the n_t 8 value itself) and +0.78 % (if the true order were 1), point estimates +0.62 % (p_eff 2) and +0.56 %
(p_obs); the dt change at n_t 4 is −3.5e-8 relative. Option (a) costs ≈ 5× per env-step on every wide-strip scene and
still leaves n_t 6 at +0.21 %. Option (b) — any rule that turns this FAIL into a PASS on the same data — would be
evaluator loosening and is rejected.

Conditions (binding; before any P2-5 or P2-6 declaration unless stated):
1. **Implement the label** (K2-2): `rubber_ref_v1` gains `measured_bias` — Q1w "+0.62 % (+0.50 … +0.78 % for p ≥ 1),
   GCI 2.16 % (Fs 3, p_obs 3.12)", Q1 +0.39 %, Q2 −0.39 %, Q4 oscillatory (|bias| ≤ 0.05 %, not a signed value) — with its
   scope (L/t 20, W/t 4, t 4 mm, α 0.05, β 20, CFL 0.5, 2 particles per cell), and
   `known_limitation: rubber_phys_11_v5_q1w_gci_2.16pct`, tested. Do not edit the frozen `workload_e_v2.json`
   ("E_full … full"); add an additive qualifier in the v5 report and the test at `tests/test_rubber_phys_config.py:195`
   follows the new label in a versioned commit.
2. **Make the label reach every receipt.** Today the scene receipt carries a free-text `resolution_profile` that defaults
   to `rubber_rl_v1` whatever n_t is and is not validated (`rubber_scene.py:49, 141`; a probe compiled n_t 3 labelled
   `rubber_ref_v1` and n_t 4 labelled `rubber_rl_v1` without refusal). Validate the profile against n_t, refuse unknown
   names, and print label, bias and limitation in the scene, `LiveRubberMaterial`, batch, P2-6 env and Native receipts.
3. **The P2-6 geometry is outside the measured class.** The n_t 4 deviation grows with L/t (-02 data: W = 4t from
   +0.38–0.53 % at L/t 10 to +0.57–0.73 % at L/t 20); the P2-6 slots reach L/t 37.5 (W/t 2.7–4). RL-08 must report the
   n_t 4 bias on the real RL geometry, and P6.0 must declare now a fallback trigger (e.g. bias > 1 % → that scene class
   moves to option (a), declared as v6 (6 / 8 / 12) before any run).
4. **P2-5.** The n_t 6 / CFL 0.25 synthetic truth is correct, but n_t 6 is itself +0.21 % for Q1w, so the discretisation
   floor b_mk captures only ≈ 0.4 % of the bias. Fitted θ is resolution-specific (`rubber_ref_v1`): label it and never
   carry it to n_t 3 or 6 without the bias shift.
5. **Correct the plan's §5 (c) rationale** (`P2_REMAINING_PLAN.md:524`): the "≈ 4× below the β = 20 model-form error"
   compares a sag bias with a uniaxial stress error at λ 1.5 (for small-strain sag the β 20 effect is +1.67 %, a ≈ 2.7×
   margin); "below the P2-5 tolerance floors" is false for the 0.5 % M1 / M3 floors (the tolerance formula absorbs the bias
   through b_mk instead).
6. Scenes where the bias matters (sub-percent small-deflection stiffness, sag or f1 of wide or slender strips; Native
   analytic comparisons; bending-dominated calibration fits) cite it; large-deflection observables are barely affected
   (Q4 ≤ 0.05 %).

## 4. The plan (ADR §23, `P2_REMAINING_PLAN.md`) — PASS_WITH_LIMITATIONS

Sound and accepted:
- **B3 / C1 deferred** (decision 1): consistent with the spec's P2-6 (no robot feedback required; kinematic grips have no
  coupling error by construction) and with RUBBER-CONT-07 (every envelope point `coupling_delay_significant`). The spec's
  P2-4 "rubber–robot gripper" item is covered physically by the kinematic pad frames and the pinch test; the receipt label
  `robot_feedback_coupling: NOT_IMPLEMENTED_PHASE2` must appear in every P2-6 receipt. A user scope decision (U1 /
  U-P2-1).
- **P1 deferred** (decision 3): correct for a plane-only P2-6 scene (F = 0 skips the frame slerp); the +25 % trigger for a
  CPU-byte-identical early-out is a reasonable guard.
- **S1 required** (decision 2): correct; implemented (§2.6).
- **P2-5 design** (decision 6): the manifest v2 with `data_origin`, synthetic statuses outside the REAL gate's directory
  and a negative test that no synthetic manifest can yield `REAL_CALIBRATED` close F10; the TRF formal method with an
  ask/tell hook for an external optimizer satisfies the spec (optimizer free, no RL term); tolerances from a separate
  pilot frozen before the formal dataset, a power test (negative control must fail), and the CAL-08 order audit are the
  right structure; the label `NUMERICAL_VALIDATION_SYNTHETIC` is honest. Additions: Q1w condition 4; the sealed truth lives
  on the same filesystem as the fitting agent (`artifacts/rl_ready/p2_5/<run>/sealed/`), so the seal is procedural —
  CAL-08 detects order violations, not peeking. Have the truth generated and revealed by a separate process / agent, keep
  `sealed/` outside the worktree with restrictive permissions, and log every read of it (or encrypt it with a key the
  fitting agent does not hold).
- **P2-6 design** (decision 7): one task family covering every randomisation axis of the spec; a deployable 78-dim
  `policy` without force, wrench, stress, J, energy, contact or material parameters; privileged `critic` opt-in;
  termination through `rl.termination` with bit 8; the only Phase 1 edit a lazy dispatch branch in the non-frozen
  `rl/vector_env.py`, made last. RL-01…RL-11 cover contract, parity (L0 / L2 / L3), randomisation, isolation (incl. the
  -09 v2 rule, see K2-3), episode semantics incl. S1, long rollout, throughput / VRAM vs Phase 1 with the full workload
  columns, dt / spacing, Native, provenance and Phase 1 non-regression.
- **Native host** (decision 8): a new `native_rubber_host.py` that keeps the TLMPM bridge's shape (kinematic clamp,
  PhysX step, measured pose, `advance`, surface update, record), with `isaac_sim_runner.py` and the frozen
  `live_material.py` untouched, feedback off, and a comparison against 20 headless GPU replays of the recorded poses
  with bitwise inputs and receipts — the smallest honest path for "same recipe in Full Native".

Required amendments:
1. **Decision 4 (μ envelope)**: per K2-1, v3 → 0.8 holds for translational sliding only; ENV1 must not map v3 to the P2-6
   friction range until a declared protocol scores spin-down, lateral drag and multi-region slip; until then the
   contingency applies (v2, μ ≤ 0.4 after ENV2 at n_t 4–6, else the friction axis BLOCKED). The P2-6 preload (lateral
   ±30 mm, yaw ±0.4 rad), the yaw actions and the Native lift + yaw are in exactly the regime v3 gets wrong.
2. **Done map, box 2**: add "law v3: translational sliding only (K2-1)" and "-05 d post-hoc metric correction (K2-5)"
   to the limitations, and the -09 summary-floor note (K2-3) to box 3's evidence.
3. **ENV2** adds an unseen strip-on-table geometry (K2-4) and the non-translational scenes above.
4. **RL-08** reports the n_t 4 bias on the P2-6 geometry with a pre-declared fallback trigger (Q1w condition 3).
5. Q1w rationale numbers (Q1w condition 5).

## 5. Merge decision: GO

- Trial merge of `77f5a99` onto `claude/rl-ready-rubber-physics` @ `502dd8f` (`1539788` plus one progress-doc commit,
  "Phase 2 user-decision defaults"): automatic, **no conflict**, 78 files (the integration branch already holds
  `c9d771f`; merge base `c9d771f`). No path of `acceptance_v1/v2` `frozen_paths` is touched; the only `evaluator_files`
  entries touched are new or changed rubber tests (`tests/*`), as in every earlier rubber merge;
  `configs/rl_benchmark/*`, `scripts/rl_ready/*` and `src/robotic_cutting_sim/rl/*` unchanged. Suites on the merged tree
  (with `tests/test_perf_acceptance.py`): **613 passed, 5 skipped** (`CUDA_VISIBLE_DEVICES=''`, 1,202 s; skips: 3 torch tests — torch is not in `.venv` — and 2 NOT_RUN in `test_rl_policies_batched.py` because the git-ignored WP-17 recordings are absent in the temporary merge worktree; a skip is not a pass).
- GO because: everything new is opt-in (contact laws v1 default, v2 / v3 by `law=`; S1 `off_v1` default; motion-table
  refresh launch-identical on the `advance()` path), nothing frozen changed, every FAIL is preserved, and the
  findings concern labels, records and the use of v3 in later tasks — not the correctness of what the merge enables by
  default.
- Conditions (they bind ENV1 / ENV2 / P5.4 / P6.0, not the merge commit): K2-1 (v3 scope; ENV2 scenes), K2-2 and the
  Q1w conditions, the additive record corrections K2-3 / 5 / 6 / 7 / 8, and the S1 errata / next-version items K2-12…K2-15. This review document records
  the v3 restriction on the branch that is merged, so the integration branch carries it.

## 6. Frozen files, byte identity, CI, tests

- Frozen (verified by `git diff 025ecaf..77f5a99` on the paths): Phase 1 `batched/{constitutive,errors,material_batch,
  particle_contacts}.py`, `configs/rl_benchmark/*`, `scripts/rl_ready/*` — no change. Rubber: `tlmpm_rubber_v1.py`
  (sha256 `28fc5189…`), `rubber_constitutive.py` (`c12cc77e…`), `tlmpm_rubber_visco_v1.py`,
  `rubber_visco_constitutive.py`, `rubber_constraint_work.py` — unchanged; protocols `rubber_phys_tests_v1..v4`,
  `rubber_moving_grip_tests_v1`, `rubber_visc_tests_v1/v2`, `workload_e_v1` unchanged; new protocols added once each
  and SHA-pinned. Material status `SIMULATOR_ONLY_UNCALIBRATED` everywhere; no recipe or calibration file changed. The
  only modified evidence files gained additive blocks (`p2_3_visco_002.json`, `p2_2_v5_001.json`) or a pointer field
  (`p2_4_contact_x2_001.json`, NOTE).
- Byte identity: v1 / v2 contact kernels (recomputed pins, §2.3); `motion_table_v1` `advance()` path (§2.1); S1 off
  (§2.6); L0 byte-wise for v1 / v2 / v3 batched contact and v5 -04 (§2.2, §2.4).
- CI safety: no new test reads artifacts, needs CUDA or uses a skip that could hide a failure; the CI job runs the full
  CPU suite (`.github/workflows/tests.yml`); the 4 skips in the selection below are the torch tests of
  `test_rl_policies_batched.py` (torch is not in `.venv`; they run in the Torch venv per D-09).
- Suites (rubber + batched + `test_rl_policies_batched.py` + `test_repo_naming.py`, as in the `c9d771f` review), review
  worktree at `77f5a99`, `.venv` Python 3.12, Warp 1.17.0:
  - `CUDA_VISIBLE_DEVICES=''`: **521 passed, 4 skipped** (torch), 2 warnings (scipy IntegrationWarning in the visco
    reference), 1,040 s;
  - `CUDA_VISIBLE_DEVICES=0` under the GPU0 lock: **521 passed, 4 skipped** (torch), 2 warnings, 1,130 s;
  - merged tree, `''`, plus `tests/test_perf_acceptance.py`: **613 passed, 5 skipped** (`CUDA_VISIBLE_DEVICES=''`, 1,202 s; skips: 3 torch tests — torch is not in `.venv` — and 2 NOT_RUN in `test_rl_policies_batched.py` because the git-ignored WP-17 recordings are absent in the temporary merge worktree; a skip is not a pass).

## 7. Commands and probes

```
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider -rs tests/test_rubber_*.py \
  tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_batched_*.py \
  tests/test_rl_policies_batched.py tests/test_repo_naming.py
flock -w 3000 /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  .venv/bin/python -m pytest … (same selection)
git worktree add --detach <tmp> 502dd8f && git -C <tmp> merge --no-commit --no-ff 77f5a99   # then the CPU suites
for m in efe3833 d7af304 d6ebed8 2d8da80: checkout m^1, merge --no-commit m^2, git diff --cached m   # K1 replay
```

Probes (DIAGNOSTIC, CPU, scripts in the review session scratchpad, not committed; they reuse the formal runner's
`_live` / `Stepper` / `_material` of `scripts/rubber/p2_4/run_contact_v1.py` with the v3 protocol's `common.options`):
- `spin_probe.py <v2|v3> <μ> 10 [n_t]`: free block 16 × 16 × 4 mm on the plane z = 0, elastic material of the contact
  protocols, gravity ramped over 20 ms, held to 40 ms, then particle velocities set to ω × (x − x_cm) with ω = 10 rad/s
  about z; 40 control steps of 1 ms; ω_z from Σ r × v / Σ r²; the plane wrench torque.
- `drag_probe.py <v2|v3> 0.4`: case-A strip 60 × 8 × 4 mm, n_t 3, on the plane, 40 ms settle, jaw (group 0) translated
  in +y at 0.1 m/s for 80 ms; mid / tip mean y of the float64 positions.
- Kernel-source hash pins recomputed at six commits; v3 protocol thresholds diffed against v1 / v2; run meta start times
  against commit times; the sub-reviews' recomputation scripts (GCI, -02 … -09, -09 band power / floor) as listed in
  their reports.

