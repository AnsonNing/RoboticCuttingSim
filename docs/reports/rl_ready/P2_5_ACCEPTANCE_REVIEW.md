# P2-5 — xhigh acceptance review (P5.7, 2026-11-12)

**Question.** Should `claude/rl-rubber-p25` @ **bc08564** be merged into `claude/rl-rubber` (@ `2a7cb77`, merge base
`77f5a99`)? The merged branch would deliver P2-5 as a complete, honest result that records its FAILs: a calibration
interface for real data plus synthetic numerical validation.

**Verdict: GO_WITH_CONDITIONS.**
- **Protocol integrity: PASS.** It was checked from git, the receipts, the sealed read log and the filesystem atime.
- **Recorded FAILs.** CAL-04, CAL-05 and CAL-06 FAILed and are recorded as FAILs; the verdicts are not softened.
- **Labels.** Every output carries the labels.
- **Frozen material.** Nothing frozen was modified.
- **Tests.** 612 / 612 pass on the head. The trial merge is clean.
- **Conditions.** Four documentation conditions (§8, C1–C4) must land as additive errata. They correct an
  over-interpreted failure mechanism, add the missing guidance for real data, mark stale sections and qualify one
  headline claim.
- **Real data.** The gate and ledger must be hardened (F3) before any real data is used. This is tracked; it does not
  block the merge.

**Material status.** It stays `SIMULATOR_ONLY_UNCALIBRATED` / SIM_ONLY. The plan's requirements are met:
- no real data exists;
- every test is labelled as numerical validation;
- the tolerances were frozen from the pilot before the formal held-out;
- the objective has no RL term.

**How the review was done.**
- **Read-only.** No branch was modified and no merge was made.
- **Fresh worktree.** Detached `RoboticCuttingSim-rl-p25accept` at bc08564. `.venv` / `.deps` / `artifacts/*` are
  symlinked; no link target was written. The worktree was removed afterwards.
- **Trial merge.** A throwaway worktree, detached at 2a7cb77, ran `git merge --no-commit bc08564`. No ref was updated,
  and the worktree was removed afterwards.
- **No GPU** was used.

**Inputs read:**
- the plan's P2-5 section and "Phase 2 Done";
- `round_prompt.md` and AGENTS.md;
- the report `P2_5_RUBBER_CALIBRATION.md` §1–§10 and ADR 0023 §23.6;
- `rubber_calib_tests_v1.json`, `rubber_calib_tests_v2.json` and the `drafts/` copy;
- `rubber_calib_thresholds_v1.json`;
- the evidence `p2_5_calib_synthetic_001.json` and both fit receipts;
- the four new modules and every P2-5 script and test;
- the raw outputs under `/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/p2_5/`: run receipts, logs, the held-out
  ledger, the reveal receipt, both `sealed/read_log.txt` files, the `v2_dev/` runs and the `fit_001` abort records.

"Recomputed" below means the review derived the number again from those files.

## 1. Protocol integrity — PASS

### 1.1 Ordering

All times are UTC. Commit times are committer dates; run times come from the run receipts.

| # | Event | Time | Source / check |
|---|---|---|---|
| 1 | v1 protocol declared alone (`c5054b9`) | 11-11 10:57:19 | 1 file in the commit; sha `1f681641…ff2b` = pin |
| 2 | pilot generation, `54a8f06` clean | 11:04:11 – 12:08:10 | `pilot_gen_001/gen_receipt.json` |
| 3 | pilot θ revealed (allowed: the pilot is development data) | 14:04:38 | pilot `read_log.txt`; `truth.json` atime 14:04:38.81 |
| 4 | pilot evaluation, `f042d34` clean | end 14:14:21 | `pilot_eval_receipt.json` |
| 5 | thresholds committed alone (`4d7d2c4`) | 14:14:35 | 1 file; sha `40be2f8b…aa36` = pin |
| 6 | formal generation, `44dcc9c` clean | 14:14:55 – 15:19:09 | `formal_gen_001/gen_receipt.json` |
| 7 | formal dataset + manifest (`a0250bb`) | 15:19:30 | freeze `3d37a950…89c9` re-validated at the head |
| 8 | v1 formal fit and negative control (`fit_001`), aborted with no receipt | 15:48 – 18:36 | `ABORTED.json` ×2, logs kept |
| 9 | v2 development, pilot only | 19:20 – 21:29 | pilot read log 19:20 / 21:21; formal read log has no entry |
| 10 | v2 declared alone (`4218dd6`), pinned (`dfd2059`) | 21:32:42 / 21:32:55 | 1 file; sha `1606457f…01ff` = pin |
| 11 | formal v2 fit, `dfd2059` clean | 21:33:01 – 23:56:16 | `fit_run_receipt.json` |
| 12 | formal receipt committed alone (`ab4faf7`) | 23:56:44 | byte-equal to the artifact (sha `617d65f3…24d7`) |
| 13 | negative-control fit, `ab4faf7` clean | 23:56:48 – 00:34:53 | starts 4 s after #12 (declared run order kept) |
| 14 | negative-control receipt committed alone (`ad4c5aa`) | 00:35:08 | byte-equal (sha `3446a6d2…137f`) |
| 15 | one-shot held-out of both receipts, `ad4c5aa` clean | 00:35:14 – 00:47:19 | ledger label `ONE_SHOT`; written at 00:47:19 |
| 16 | reveal | 00:47:30.002 | `read_log.txt` has **one** line; `truth.json` atime 00:47:30.016 |
| 17 | supplementary diagnostic (unscored) | 00:59:25 | script committed at 00:47:38 (`1923fc0`), after the reveal |

**Sealing.** The formal `truth.json` (mode 0400, directory 0700) has an atime 13 ms after the logged reveal. The
filesystem is mounted `relatime`, so that atime is the **first read after creation**. This is independent evidence
that nothing read the sealed truth before the reveal: the procedural log and the filesystem agree.

**One-shot held-out.**
- A filesystem search found exactly one `heldout_3d37a950…` ledger (`fit_v2_001/heldout/`).
- It scores both committed receipts in one call, on a clean tree, with no audit re-run.

**Frozen files.** Each frozen file has exactly one commit on the branch: v1, v2, the thresholds, the pilot and formal
dataset directories, and both receipts. Their SHA-256 values equal the pins. The formal manifest still validates
(freeze hash over the manifest, every CSV and the oracle file).

**Code frozen at the v1 declaration.** After `c5054b9`, nothing changed in `rubber_calibration_experiments.py`,
`rubber_calibration_v2.py`, `rubber_calibration_synthetic.py` or `reveal.py`. `rubber_calibration_fit.py` received only
additions (`fit_staged`, `SubSpace`, the budget clock). `fit_trf` gained an `x_scale` argument that defaults to the old
1.0.

### 1.2 Was v2 limited to the fit method and budget?

Yes, as declared. Compared with v1:
- **Fit procedure:** staged S1–S4, `x_scale` "jac", FD step 1e-2, start at the prior-box centre, Latin-hypercube
  starts only in S2.
- **Bounds:** narrowed from the v1 fit box to the u-image of the prior box (F9).
- **Budget:** 3 h / 1.5 h of the run's own GPU-lock hold time.
- **Negative control:** its fit procedure is staged as well.
- **CAL-04:** a new termination clause, vacuous in effect (F6).
- **CAL-08:** an extra ordering step.

Unchanged and verified: the dataset, thresholds, models, metrics, sensors, the CAL-05 / CAL-06 definitions and the labels.

**Draft vs declared v2.** The draft (`artifacts/.../drafts/rubber_calib_tests_v2.json`, 04:20 JST, before the pilot
validation finished) already had the stages, the `max_nfev` caps and the budgets. Only the budget clock changed (wall
clock → GPU-hold time), together with the prediction / method-development text. No tuning toward the formal data is
visible.

**Information flow.** v2 was designed after the v1 formal fit's **fit-split** trajectories had been seen (the drift out
of the prior box in report §5). That is legitimate: the fit split is not sealed, and neither the held-out split nor the
truth was used. Still, "method development used the pilot only" should mention it (F9).

### 1.3 Pre-registered predictions vs outcomes

| Test | Prediction (v2) | Outcome | Reported honestly? |
|---|---|---|---|
| CAL-04 | PASS | **FAIL** (χ²_red 2.4838 > 1.9097, recomputed) | yes, "PASS (missed)" in the table, the headline and the progress file |
| CAL-06 | FAIL on g1 / τ1 / g2 / τ2 (resolution bias 8–15 %) | **FAIL** on g1 −51 %, g2 +39 %, τ2 −41 %; τ1 PASS (−0.5 %) | the verdict matched, the mechanism did not; the text says so ("not the resolution bias … predicted") |
| CAL-05 | at risk on T4 and on M9 | **FAIL** 21 / 31 | the predicted T4 and M9 failures appear, plus unpredicted C3 / C4 M4 and C2–C4 M6 failures; the text lists every failing row |
| CAL-07 | none | **PASS** (T5 M2 0.0516 > 0.0269) | yes; see F5 for the headline wording |

**Recomputed.** The review recomputed CAL-06 for both receipts from θ̂, σ̂ and the revealed θ* using the declared rule
max(3σ̂, 0.02 θ*); every row matches the evidence. Every CAL-05 tolerance equals the frozen threshold, with 21 / 31 PASS
for each fit.

## 2. Correctness of the code

### 2.1 Manifest v2, labels and REAL gate (`rubber_calibration_v2.py`)

**Readiness for real data.**
- The schema is closed (unknown or missing keys are refused).
- `data_origin` is mandatory, and statuses are bound to the origin.
- Every experiment needs sensors with σ, resolution, rate, full scale and an SI unit; specimens need measured dimensions
  and mass with their uncertainties.
- The splits are disjoint and complete.
- The freeze hash covers the manifest, the CSVs and the oracle file.
- Data files must be plain names: no path traversal.
- An unlabelled manifest is refused, because `calibration_status` / `validation_kind` are required keys.
- Real and synthetic data cannot mix inside one manifest: one origin per manifest, and the `synthetic` block is required
  for synthetic data and refused for real data.

**Label propagation.** Labels reach every output. The review checked all 23 result JSONs under `p2_5/`, both committed
receipts, both manifests, the thresholds file and the evidence. The only files without labels are a CPU-test summary and
a smoke oracle data file.

**The F10 hole.**
- The v1 gate reads v1 manifests only, so it refuses a v2 synthetic manifest whatever its directory, status or verdict
  (24 cases × 2 gates).
- The v1 masquerade still passes the unchanged v1 gate, as the test shows. The tripwire `gate_directory_audit` catches
  it, and the gate directory audits clean at the head.

**Pins.** The v1 module (`790f41d0…`) and the `REAL_CALIBRATED` block (`4624225f…`) are byte-identical at the base, at
the p25 head and at `2a7cb77`.

**Weak point (F3).** `real_calibrated_gate_v2` takes the held-out receipt as a caller-supplied dict. The test's own
positive control passes a hand-written dict, so the gate's "first one-shot PASS for that freeze" is not enforced.

### 2.2 Forward-model parity (CAL-02) — real

`run_single` drives a fresh **`LiveRubberMaterial`**, the single-env path. `ExperimentBatch` drives a
**`BatchedRubberMaterial`**. The test compares every raw field byte-wise for T / R / C / G / F, with two heterogeneous
candidates, a re-used batch, swapped env order and repeated objective calls. The comparison is therefore single vs
batched, not batched vs batched.

**Limits.** Parity is shown on Warp CPU at a reduced size: n_t 2, mm-scale specimens, a few control steps. The GPU
formal runs at n_t 4 / 6 are not bitwise, by design: the default Warp float atomics are non-deterministic, and
RUN_TO_RUN mode is refused by a frozen Phase 1 kernel. This is declared, and the GPU noise was measured (report §9.1).

### 2.3 Fit, budget, GPU lock and evidence writer

**`fit_staged` budget clock — correct.**
- The deadline is set on the supplied clock.
- `BudgetedObjective` measures each population on that clock and carries the previous duration across stages.
- The final stage always gets one population.
- The best evaluated point of the final stage is θ̂, and its Jacobian comes from the same population.
- The CPU tests cover the budget rule and the custom clock.

**GPU lock per evaluation — correct.**
- `LockedObjective` takes the blocking flock on `rcs_rl_runner/gpu0.lock` around every `residuals_population`. That is
  the correct lock, not the research repository's `locks/gpu1.lock`.
- The population batch is built lazily inside the first locked call.
- Formal run: 36 acquisitions = 35 evaluations + 1 start check; 7,737 s held, 858 s waiting. The budget clock used
  7,736.5 s.
- Between evaluations the process keeps its CUDA context and batch memory without holding the lock. This is declared,
  and it is acceptable under the ≥ 4 GB-free check.

**Evidence writer — the verdicts it writes are correct.** Checked:
- CAL-04 limit and χ²;
- the CAL-05 / CAL-07 rows (from the ledger);
- CAL-06 (from the reveal);
- the CAL-08 order from first-add commit times and receipt times.

It has gaps (F6, F7). This review closed them by hand: every gap was checked and found in order.

### 2.4 Do the tests test the claims?

Mostly yes. The P2-5 files have 74 tests:

| File | Tests |
|---|---|
| `test_rubber_calibration_v2.py` | 38 |
| `test_rubber_calibration_experiments.py` | 11 |
| `test_rubber_calibration_fit.py` | 17 |
| `test_rubber_p2_5_protocol.py` | 8 |

Gaps are listed in F10.

## 3. Honesty of the report, ADR and evidence

**What is right:**
- The §10 headline states the CAL-04 / -05 / -06 FAILs without softening.
- The diagnostic is labelled `SUPPLEMENTARY_DIAGNOSTIC_NOT_SCORED (after the one-shot held-out and the reveal)`, and no
  verdict uses it.
- The labels are everywhere, and no recipe is promoted.
- "Any re-fit on this dataset is post-hoc; a blind re-test needs a new sealed dataset" is stated.

**What needs correction:**
- The **mechanism** given for the FAIL is over-interpreted (F1).
- The **stale pre-v2 sections** contradict §10 (F4).
- One **headline** claim needs a qualifier (F5).
- The report has **no "what a user with real data should do differently" section** (F2). The integration branch's
  progress entry U-P2-6 points to "diagnosis and fixes in the P2-5 report", but those fixes do not exist.

## 4. Collateral damage — none

**Diff.** `git diff 77f5a99 origin/claude/rl-rubber-p25 --stat` shows 61 files, +22,705 / −0. There are 60 added files
and 1 modified file: ADR 0023, which only appends §23.6.

**Nothing frozen touched.** No change to:
- the Phase 1 modules, `configs/rl_benchmark/*` or `scripts/rl_ready/*`;
- the rubber solver modules or the earlier rubber protocols;
- `configs/materials/*`;
- the thresholds or records;
- the CUT / TRIM / v5 / v7 sources or baselines.

The frozen-source pin tests (`test_batched_frozen_sources.py`, `test_wp12_phase5_frozen.py`) and the v1 calibration pins
pass.

**Merge into `claude/rl-rubber`.** `2a7cb77` changed `rubber_material.py`, `rubber_scene.py`,
`batched/rubber_material_batch.py`, `batched/rubber_contact_batch.py` and `live_rubber_material.py`. Each change is a
receipt or label change (the K2-2 resolution label). The scene compiler now validates the profile's n_t band, and P2-5
uses `rubber_ref_v1` (band [4, ∞)) at n_t 4 and 6, plus `rubber_coarse_v1` at n_t 2 in the CAL-02 test, so every case is
inside its band. The `REAL_CALIBRATED` block pin is unchanged on `2a7cb77`. The trial merge is textually clean (ADR
auto-merged), and the merged tree's tests pass (§5).

**Caveat on reproducing the P2-5 results.** The merged tree reproduces them only up to the added receipt fields. The
formal results were produced at `dfd2059` on top of `77f5a99`.

## 5. Tests and CI

| Run | Command | Result |
|---|---|---|
| p25 head bc08564, fresh worktree | `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rubber_*.py tests/test_batched_*.py tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_repo_naming.py tests/test_wp12_phase5_frozen.py` | **612 passed**, 2 warnings (scipy IntegrationWarning, as before), 750 s |
| trial merge 2a7cb77 + bc08564 (throwaway) | `CUDA_VISIBLE_DEVICES='' … pytest tests/test_rubber_*.py tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_batched_frozen_sources.py tests/test_wp12_phase5_frozen.py` | **477 passed**, 2 warnings, 273 s; includes the K2 tests (`test_rubber_resolution_label.py`, `test_rubber_cont09_rule_v3.py`) |

**Environment.** Python 3.12.12, Warp 1.17.0, SciPy 1.18.1, NumPy 2.5.4. The `CUDA_VISIBLE_DEVICES=0` repetition is
NOT_RUN here: there was no GPU need, and the implementer recorded 599 passed in both modes at `dfd2059`.

**GitHub CI.**
- **Head bc08564:** run 43131450332 is **in progress** at the time of writing.
- **Pre-existing failures.** Every earlier P2-5 run and the base runs (`77f5a99` 43001584345, `2a7cb77` 43011951420)
  conclude **failure** with the **same pre-existing 18 failed + 6 errors**. They are in
  `tests/test_rl_lab_{ik,kernels,rigid_isolation}.py`; the causes are a missing `newton_usd_schemas` and a missing Franka
  asset on the runner. These failures are not on `claude/rl-ready-rubber-physics`, whose CI is green, so they are a
  rubber-line CI debt.
- **Runner-dependent failures.** Runs scheduled on AMD EPYC 7763 runners add 10 failures in
  `tests/test_batched_pose_interp.py` (bitwise / ULP comparisons). Example: `dfd2059` run 43104017170, 28 failed. The same
  file passes on EPYC 9V74 / 9V45 runners and locally.
- **No rubber or P2-5 test fails in any CI log inspected.**
- None of this is caused by P2-5. It is recorded for the Phase 2 PR to `main`.

## 6. Ranked findings

### F1 (MEDIUM): the failure mechanism is over-interpreted

**Where the claim appears:**
- Report §10 and ADR §23.6: "a worse valley", "a wrong Prony basin after S1 / S2", "the joint polish (6 evaluations)
  did not leave that basin", "missed because of the S2 data choice and starts".
- The integration progress file repeats it ("stopped in a worse basin").

**Recomputed from the formal receipt:**
- **Every stage after S1 ended on its `max_nfev` cap (status 0) with the cost still falling:**
  - S2, all 3 starts at 15 evaluations: −7.1 %, −2.5 % and −1.6 % in the last evaluation;
  - S3 at 8 evaluations: −6.8 %;
  - S4 at 6 evaluations: −5.4 %.
- **3,063 s (0.85 h) of the 3 h budget was unused.**
- **S2 is bound-active, not a basin found by its starts.** All three starts drove g1 to the lower edge of the box
  (0.0800–0.0840 against the bound 0.08) while g2 rose and τ2 fell. Three starts landing on the same bound point to the
  stage's sub-problem (μ_eq frozen from S1's T1-only fit, +1.2 %), not to bad starts.
- **In S4, g1 rose at every evaluation (0.0840 → 0.0846 → 0.0855 → 0.0875 → 0.0900 → 0.0942, accelerating toward
  θ* 0.193).** τ1 reached the truth (14.3 ms) before the cap ended the stage.
- **Nobody tested whether θ̂ lies in a separate local minimum.**

The supported statement is: *non-convergence under the declared per-stage evaluation caps, after a bound-active S2;
whether a distinct basin exists is untested.*

**The pilot (§9.3) has the same problem.**
- "More polishing lowers χ² further and does not move toward θ_pilot" was never measured. The pilot S4 had one
  evaluation, and its S2 was truncated at 15 evaluations with two starts still falling 10–15 % per evaluation.
- That a resolution bias exists is supported: χ²(θ̂) 1.170 < χ²(θ_pilot) 1.273, about 245 χ² units over about 2,380
  residuals.
- Its **size ("≈ 10 %") comes from an unconverged fit.**

**Why it matters (failure scenario).** A user with real data follows the advertised remedy (a multi-start joint stage),
keeps the 6-evaluation polish and the frozen μ_eq in S2, and gets the same truncated, bound-active fit. The U-P2-6 note
"no v3: recovery would likely still fail from the ~10 % bias" also rests partly on that unmeasured size.

The no-v3 decision itself is acceptable: the plan requires only the interface plus numerical validation.

### F2 (MEDIUM): the report has no guidance for a user with real data

The orchestrator asked for this section explicitly, and the progress file claims the fixes exist. Condition C2 lists the
required content.

### F3 (MEDIUM; real-data readiness, not exercised): the REAL gate v2 and the one-shot ledger can be bypassed

- `real_calibrated_gate_v2(manifest_path, heldout_receipt)` accepts any dict with `verdict` "PASS", the right
  `manifest_freeze_sha256`, `validation_kind` `REAL_MEASUREMENT_HELDOUT` and no `audit_rerun`. It does not:
  - load the ledger file;
  - check `schema` / `label == "ONE_SHOT"`;
  - check the thresholds SHA, the fit-receipt SHAs or that the receipts are committed.
- `evaluate_heldout` is one-shot **per caller-supplied `ledger_dir`**, not per manifest freeze, and `reveal.py` checks
  only that directory.

**Failure scenario.** With real data, `run_heldout.py` is run twice with two `--ledger` directories. Both results are
labelled `ONE_SHOT`, and the better one, or a hand-written dict, opens the gate.

**Current impact: none.** No recipe uses the gate; the real path is NOT_RUN, and only one ledger exists for this freeze.

### F4 (LOW–MEDIUM): stale pre-v2 statements contradict §10

**In the report:**
- §4 says the sealed truth "has **never been read** (no `read_log.txt`)" and "the dataset stays valid for a resumed fit".
- §6 lists CAL-04..07 as NOT_RUN and CAL-08 as "so far".
- §8 says it "does **not yet** deliver the formal fit …".
- §9.4 says "the staged optimiser is **validated as an optimiser**".

**In ADR §23.6:** "sealed truth … (never read)" and "a resumed fit … can still complete P5.6".

**Failure scenario.** A reader or agent consulting §4 / §6 resumes a "blind" fit on a revealed dataset, or cites a
validated fit procedure. The formal run showed the opposite: the procedure FAILed.

### F5 (LOW): "the negative control has power" is too strong as a headline

CAL-07 PASSes, as defined, on **one** viscous row (T5 M2). Otherwise the T2 + C1 control:
- passed T4 M2 and every cantilever M6 row that the primary fit failed;
- scored the same 21 / 31.

So this run does not show the held-out discriminating an adequate fit from a deficient one. Only the unscored
post-reveal θ* evaluation (31 / 31) does that.

### F6 (LOW): the termination fields cannot show non-convergence

- **CAL-04 clause is vacuous.** The v2 clause accepts any termination, `max_nfev` included, and `write_evidence.py`
  `terminated` is always true for a finished fit.
- **`converged_xtol` is mislabelled.** It is computed as `status in (2, 4)`
  (`rubber_calibration_fit.py:379, :922`). In SciPy, status 2 is ftol, 3 is xtol and 4 is both, so the field reports
  ftol and misses xtol-only stops.

Neither changes a verdict here. Both would let a real-data user score a capped, non-converged fit without a visible
flag.

### F7 (LOW): evidence-writer gaps

- CAL-09 is hard-coded `"PASS"`, not computed.
- The CAL-08 audit does not read `sealed/read_log.txt`.
- It also does not check that the negative-control fit started after the formal receipt commit, or that the v1 fits
  were aborted before the v2 declaration.

This review verified every one of these by hand, and all are in order (§1.1).

### F8 (LOW): reveal and diagnostic on dirty trees

- The reveal ran at `ad4c5aa` with a dirty tree (the uncommitted supplementary script).
- The diagnostic ran at `1923fc0` with a dirty tree (`write_evidence.py` edits).
- `reveal.py` neither requires a clean tree nor pins the protocol SHA (`load_protocol()` without an expected value).

**Impact here: none.** `reveal.py` has not changed since `221c337`, and the recovery numbers were recomputed
independently.

### F9 (LOW): the optimism of the bounds is not stated

- **The bounds are the generating prior box.** The v2 fit bounds are the u-image of the prior box the truth was drawn
  from, which makes the synthetic test easier than the real-data case.
- **The u-box is wider than the θ prior.** It admits g2 outside the θ prior box: θ̂ g2 0.186 > 0.15, so "prior-box
  bounds" is imprecise.
- **The v1 fit-split observation fed the v2 design** (§1.2).

All three deserve one sentence in the report.

### F10 (LOW): test and count gaps

- **v2 pin test.** `test_protocol_v2_is_pinned_and_consistent` zips only `params` / `starts` / `max_nfev`. It does not
  compare stage experiments or list lengths, nor the negative-control stages hard-coded in `run_fit_v2.py`. The review
  compared them by hand, and they match.
- **No test of `GPULockPerEvaluation` / `LockedObjective`.**
- **No binding of CSV column names to units.** For example, a force column declared "N" but holding mm values
  validates.
- **Counts.** `test_rubber_calibration_v2.py` has 38 tests; the report says 40 and "48 parametrised negative tests",
  which are 24 cases × 2 gates.
- **Small slip.** The ADR says "C2–C4 M4 / M6 / M9", but C2 M4 PASSed.

## 7. What is accepted

- **Interface (P5.1–P5.3): PASS_WITH_LIMITATIONS.** It is ready for real data except for F3 (gate and ledger hardening
  before first use) and the F10 unit binding.
- **Protocol v1 / pilot / thresholds / frozen sealed dataset (P5.4–P5.5): PASS.**
- **Formal synthetic validation under v2 (P5.6): FAIL, recorded honestly.**
  - CAL-01 / -02 / -03 / -07 / -08 / -09 PASS.
  - CAL-04 / -05 / -06 FAIL.
  - The fit procedure is **not validated**.
  - Fitted Prony parameters at n_t 4 are resolution-specific (a bias exists; its size is unmeasured).
- **Material status:** `SIMULATOR_ONLY_UNCALIBRATED` / SIM_ONLY. No calibration claim is made anywhere.

## 8. Conditions

Apply C1–C4 as **additive errata** (do not rewrite history; do not touch the evidence, receipts, protocols, thresholds
or datasets), either on `claude/rl-rubber-p25` before the merge or on `claude/rl-rubber` right after it. In both cases
they must land before the Phase 2 PR to `main`.

### C1. Correct the failure mechanism

**Where:** report §10, ADR §23.6 and the progress U-P2-6 / Phase 2 line on `claude/rl-ready-rubber-physics`.

**Replace:**
- "worse valley";
- "wrong Prony basin";
- "did not leave that basin";
- "S2 data choice and starts".

**With the measured facts:**
- S2, S3 and S4 ended on `max_nfev` with the cost still falling;
- all three S2 starts were bound-active at g1 = 0.08;
- g1 was still rising in S4;
- 0.85 h of budget was unused;
- whether a separate basin exists was not tested.

**In §9.3:** delete "more polishing … does not move toward θ_pilot", or mark it unmeasured, and state that the pilot's
"≈ 10 %" bias size comes from an unconverged fit.

### C2. Add a section "What a user with real data should do differently"

It must say, at least:
1. **Stop stages on convergence or on the time budget, not on small fixed `max_nfev` caps.** Run the joint polish until
   xtol / ftol or the budget. Report every stage's termination status. Treat a stage that hits its cap while the cost
   still falls by more than about 1 % per evaluation as non-converged: flag it, never score it silently.
2. **Do not freeze μ_eq from one slow tensile test before the viscous stage.** Fit μ_eq together with the Prony
   parameters, or take it from the relaxation plateau. Flag any bound-active parameter at the end of a stage.
3. **Add a global stage on the joint problem.** Use several starts, or a short CMA-ES / BO through `run_ask_tell`.
4. **Validate the exact declared procedure end-to-end on a pilot** (same caps and budget) before predicting CAL-04. The
   v2 pilot ran S4 for one evaluation only.
5. **Take bounds from prior knowledge, not from a generating prior,** and report parameters that sit at a bound.
6. **Drive the forward model with the measured fixture or crosshead history** where it exists. The current forward
   models use the declared nominal schedule.
7. **Set tolerances from measured repeatability (a) and a pilot *fit* (b)**, as `rubber_calib_tests_real_v1` states.
   Declare the real protocol, including its extra short-τ Prony term, before any data exists.
8. **θ̂ is RESOLUTION_SPECIFIC** to n_t 4. Never carry it to another resolution without the bias shift.
9. **Until F3 is fixed,** use one canonical ledger directory per freeze, and commit the ledger before the reveal.

### C3. Mark the pre-v2 sections as a snapshot

Mark report §4–§9 and the ADR §23.6 bullets before the "Formal run under v2" bullet as the pre-v2 snapshot, with a
pointer to §10. Correct the sentences in F4 ("never been read", "valid for a resumed fit", "validated as an optimiser").

### C4. Qualify the CAL-07 headline

Change "negative control with power" to say: CAL-07 PASS on one viscous row; discrimination is not shown in this run,
because both fits score 21 / 31.

### Tracked, not merge-blocking

**Required before any real-data use:**
- **F3:** the gate loads and checks the ledger file itself (schema, `ONE_SHOT`, thresholds SHA, fit-receipt SHAs,
  committed). The one-shot ledger becomes global per freeze.

**Version fixes, without changing recorded receipts:**
- **F6:** a per-stage termination column; the corrected `converged_xtol` mapping.
- **F7:** CAL-09 computed; CAL-08 reads the read log.
- **F8:** `reveal.py` refuses a dirty tree and pins the protocol.
- **F10:** the test gaps and the unit binding.

**Rubber-line CI debt, for the Phase 2 PR (§5):**
- the `rl_lab_*` runner dependencies;
- the CPU-dependent `test_batched_pose_interp.py`.

## Addendum A — re-review at `07cdb01` (2026-11-12)

**Verdict: GO** for merging `claude/rl-rubber-p25` @ **07cdb01** into `claude/rl-rubber`.
- C1–C4 are met as additive errata.
- F3 and the cheap fixes are correct and tested.
- The v3 decision is recorded honestly.
- The test suite passes at the new head.
- Nothing frozen changed.
- Three small follow-ups (R1–R3) are tracked. None blocks the merge.

**Scope.** Commits `837abdb` (errata), `0b9a8d8` (F3, F6–F8, F10), `b79d377` (v3 estimate), `e868a58` / `340a385`
(v3 code) and `07cdb01` (v3 decision).

**How the re-review was done.**
- **Read-only, fresh worktree.** Detached `RoboticCuttingSim-rl-p25accept2` at 07cdb01, symlinked as before and removed
  afterwards.
- **No GPU** was used.
- **Probes.** Two CPU probes were run in scratch temporary directories.

### A.1 C1–C4: met, additive

**Nothing frozen changed.** `git diff bc08564 07cdb01` is empty for:
- `configs/` (protocols v1 / v2, thresholds, both datasets);
- the evidence `p2_5_calib_synthetic_001.json` and both fit receipts;
- `tests/test_rubber_p2_5_protocol.py`, `run_fit.py`, `run_fit_v2.py`, `run_pilot.py` and `_common.py`;
- the experiments and synthetic modules;
- `rubber_calibration.py` and `rubber_material.py`.

**Branch shape.** Against the merge base the branch is still additions plus one ADR append (63 A, 1 M).

**How each condition was met:**
- **C1:** report §12.1 and the inline E1 markers state the measured facts; the "basin" wording is withdrawn; the pilot's
  "≈ 10 %" is marked UNMEASURED.
- **C2:** report §12.2 adds the nine points.
- **C3:** §4–§9 are marked as the pre-v2 snapshot, with the stale sentences superseded inline.
- **C4:** E4 qualifies the CAL-07 headline.
- **ADR §23.6** carries the same errata.

**Cosmetic.**
- Some erratum blockquotes sit in the middle of a sentence: §9.4, the §10 headline and "This explains the miss …".
- The `## 11.` heading and §11.1 come after §12, while §11.2 comes before it.

These affect readability only.

### A.2 F3 and the cheap fixes: correct and tested

**One-shot registry (F3).**
- `evaluate_heldout` exclusively creates `<registry>/<freeze>.claim` before it evaluates anything. Afterwards it writes
  `<freeze>.json` with the ledger path and SHA-256, and it refuses a freeze that already has a ledger, a claim or an
  entry.
- Every test passes a temporary `registry_dir`. The shared registry under `artifacts/rl_ready/` hashed identically
  before and after the full suite.

**REAL gate v2 (F3).**
- It refuses a dict. It checks the schema, `ONE_SHOT`, no audit, PASS, the freeze, `validation_kind` and the thresholds
  SHA.
- It checks that the registry entry names this exact ledger (path and SHA).
- It checks that each fit receipt is committed at HEAD with the recorded SHA.
- A positive control and each refusal are tested, in a temporary git repo.

**Retroactive registration of the formal_001 ledger: honest.**
- The registry entry's `ledger_sha256` `1dd37498…3c60` equals the ledger file's.
- The ledger is unchanged: its mtime is still the original write, 09:47:19 JST.
- The claim and the entry are labelled "registered retroactively after the P5.7 review F3 fix" and
  "retroactively (F3 fix)".
- The registration is disclosed in the `0b9a8d8` commit message, but **not in the report or the ADR** (R2).

**F6.**
- `converged_xtol` now means SciPy status 3 or 4, and a new `converged` field means status 1–4.
- `termination()` flags a capped stage that is still falling by more than 1 % per evaluation as `NON_CONVERGED`.
- No recorded receipt is affected: every final-stage status recorded so far (0 or `BUDGET_EXHAUSTED`) maps to False
  under both the old and the new mapping.

**F7.**
- The writer now computes CAL-09 and requires exactly one sealed read, after the held-out.
- The committed evidence was correctly not regenerated.

**F8.** `reveal.py` pins the v1 SHA and refuses a dirty tree.

**F10.**
- Column units are now bound to the column-name suffixes.
- Both frozen manifests still validate, with unchanged freeze hashes, and the gate-directory audit is clean.

**R1 (medium for real data; not merge-blocking): an audit re-run can come before the one-shot.**
- `audit_rerun` skips the registry entirely, so an `AUDIT_RERUN` can run before any one-shot and leave no trace.
- A later ONE_SHOT is then claimed and registered, and would pass the gate.
- A CPU probe in temporary directories showed it: an audit first (PASS), then a ONE_SHOT, registered.
- **Failure scenario with real data:** peek at the held-out through an "audit", adjust, then run the "one-shot".
- The behaviour already existed at bc08564; the F3 fix did not close it.
- **Fix before the first real data:**
  - refuse `audit_rerun` unless the registry entry for that freeze exists; or
  - record audits in the registry and make the gate refuse a one-shot that an audit preceded.

### A.3 The v3 decision: recorded honestly

**Ordering.** The cost estimate (`b79d377`, 10:26:37 JST) was committed before the v3 code (`e868a58`, 10:27:57) and
before the first development run, which started at 01:28:03 UTC at `e868a58` with a clean tree.

**What does not exist:**
- `configs/rubber/rubber_calib_tests_v3.json`. `PROTOCOL_V3_SHA` is `None`, so the `formal` and `negative_control`
  kinds and `gen_synthetic --v3` refuse to run.
- A new dataset directory or a new sealed truth.
- Any registry entry other than formal_001's.
- Any new read of the formal truth: `read_log.txt` still has one line, and the atime is unchanged.

**Development runs.**
- They are labelled "DEVELOPMENT (revealed formal_001; never scored)", with both P2-5 labels and no recipe promoted.
- Report §11.2 reports them as development only, and the recovery against the revealed θ* is labelled as development.
- The revised estimate matches the development receipts: 3,803 s of hold time, about 1.3 GPU-h in total.

**R3 (low).** `run_fit_v3.py` labels the `dev2_formal001_stageA` kind "FORMAL" in its run metadata; only
`dev_formal001` gets the DEVELOPMENT label. That run was interrupted before it wrote any receipt, so no artifact is
mislabelled. Fix it before any reuse.

**`fit_staged` defaults are unchanged.**
- The signature, the defaults and `V2_STAGES` are unchanged.
- The new `current_plus_lhs` start design applies only when a stage declares it.
- `ModuliParameterSpace` is a separate class.
- **Probe.** The toy staged fit (`TOY_STAGES`, which includes a two-start Latin-hypercube stage) was run with the
  bc08564 code and with the 07cdb01 code. Both give an identical SHA-256 digest (`d704c3f1…`) over θ̂, every
  evaluation's cost, the starts and the statuses, and the same χ²_red 0.95634478.

### A.4 Tests and CI

| Run | Command | Result |
|---|---|---|
| p25 head 07cdb01, fresh worktree | `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rubber_*.py tests/test_batched_*.py tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_repo_naming.py tests/test_wp12_phase5_frozen.py` | **616 passed**, 2 warnings (scipy IntegrationWarning), 802 s, exit 0. That is the 612 from bc08564 plus 4 new tests (termination flags, the current-plus-LHS start design, the moduli space, the unit binding). The shared registry hashed identically before and after. |

**GitHub CI.**
- **bc08564** (run 43131450332) and **0b9a8d8** (run 43134454559) conclude failure with only the pre-existing 18 failed
  + 6 errors in `tests/test_rl_lab_{ik,kernels,rigid_isolation}.py`. No rubber or P2-5 test fails.
- **07cdb01** (run 43142119241) is in progress at the time of writing.

### A.5 Remaining follow-ups (tracked, not merge-blocking)

- **R1:** close the audit-before-one-shot path before any real-data use.
- **R2:** add one line to report §12.5 / ADR §23.6 recording the retroactive registry entry (path, ledger SHA, label).
- **R3:** the label of the `dev2` kind in `run_fit_v3.py`.
- **Cosmetic:** the section order and the mid-sentence errata markers.
- **Earlier CI debt (§5):** unchanged.
