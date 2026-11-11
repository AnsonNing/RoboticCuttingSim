# P1-6 p16c1 follow-ups (PR #15) — xhigh code review (2026-11-12)

**Commits reviewed:** PR #15, branch `claude/p16c1-followups`, head `27554d21`. Base `main` is `9ee1e9d` (the PR #14
merge).
- `737e6c93`: code (`src/robotic_cutting_sim/batched/particle_contacts.py`, +38 / −7).
- `524bafa8`: tests (`tests/test_batched_particle_contacts_v2.py`, +179 / −8).
- `27554d21`: the PR's report, `docs/reports/rl_ready/P1_6_P16C1_FOLLOWUPS.md`.

The review answers the follow-ups to findings 1, 3 and 4 of `P1_6_P16C1_PR14_REVIEW.md`.

**Where I worked:**
- Tests and probes ran in a fresh detached worktree, `/home/ningan/RoboticCuttingSim-rl-pr15`, at `27554d21`. Like
  the other rl worktrees it had `.venv` and `.deps` symlinks and two read-only input links, `artifacts/wp17` and
  `artifacts/workcell_asset_cache`. The worktree was removed afterwards.
- Mutants ran in scratch copies outside any git worktree. Nothing was committed.
- I read the author's worktree `/home/ningan/RoboticCuttingSim-rl-p16fu` and evidence folder and did not modify them.
- I pushed nothing to the PR branch and posted no GitHub comment.
- Every `file:line` refers to `27554d21`.

**Logs and probes:** all are preserved in
`/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/p1_6_p16c1_pr15_review/` (§8).

## 1. Verdict

**PASS. Merge GO once GitHub CI on `27554d21` is green (§6; still running when this was written).**

- **No BLOCKER, MAJOR or MEDIUM finding.** Findings 1 and 2 are LOW, findings 3 to 6 INFO (§7).
- **The counter is correct, race-free and deterministic.**
  - It is written by exactly one thread per env (the env's first particle), only in envs queried with the
    shrunken reach. No atomics are needed.
  - On CUDA at N=16, the counter equals the number of launches in every slow env and stays 0 in the envs flagged
    fast (§3.1).
  - Partial reset leaves it untouched, as documented (§3.1).
- **The counter costs nothing measurable on the GPU (§4, `SHARED_GPU`).**
  - Kernel A/B, base against head, interleaved in one process, N=16, s512 full resolution (1,643,712 particles),
    60 rotations of 20 launches each.
  - At the Native contact-phase state (s512 step 340) with the shrink engaged, head/base = **1.0006** (p10 0.9985,
    p90 1.0023). At the reset state it is 0.9991.
  - That is about +0.6 µs per substep, about 0.12 ms per 1.66 s control step, or 0.007 %. The accepted N=16 ratio
    of 1.279 is unaffected.
  - Registers rise from 66 to 68 per thread, with no local memory.
- **The fail-closed check fires before any launch of the contact stage.** Every current caller passes the
  construction values exactly, as Python floats (§3.2).
- **The new tests catch what they claim.**
  - The two of the author's mutants I re-ran both fail (A, E).
  - Three new mutants: an unsafe 1.6 r reach, a reach of exactly 2 r, and a dropped dt check. The first and the
    third are caught; the 2 r mutant only by the unit inequality, as expected (§3.3).
  - The unsafe-reach mutant now fails the new test's own bitwise check. Before this PR it was caught only by the
    necessity test's control batch and by the receipt test's unit inequality, not by any checked batch.
  - The slow-contact scene does what the report says (§3.4).
- **Freeze compliance: PASS.**
  - No frozen path, config, script, threshold or evidence record changed.
  - Both acceptance pins hold.
  - No consumer compares the v2 receipt or this file's hash with a stored value (§5).
- **The PR body is accurate and has no host paths (§6).**

## 2. Per-item verdicts

| Item | Verdict | Basis |
| --- | --- | --- |
| Counter: correctness, race freedom, determinism | **PASS** | §3.1. One writer per env under the packed contiguous layout. On CUDA, N=16: counter = launches for slow envs, 0 for flagged envs, for both the module kernel and the verbatim copy. |
| Counter: partial reset | **PASS (as documented)** | §3.1. Cumulative since construction. Probe: [200,200,200] → [400,400,400], unchanged by `reset([1])` and still counting. Semantics: finding 1. |
| Fail-closed r/dt check | **PASS** | §3.2. It is the first statement of `_launch_v2` (`particle_contacts.py:744-749`). Nothing in `launch()` mutates state before it (`:709-716`). Callers: `tlmpm_v7_2_batch.py:639,672` and `run_p1_3a.py:173`, all `solver.radius`/`solver.dt`. Nothing assigns those after construction. |
| New tests fail under the claimed mutations | **PASS** | §3.3. A and E re-run: both fail. Plus 3 new mutants. |
| GPU cost of the counter | **PASS** (`SHARED_GPU`) | §4. Kernel ratio 1.0006 at the contact state; about 0.007 % of a control step. |
| GPU parity of the counter change | **PASS** | §4. Base and head on CUDA, N=16 Native state: identical `stats[4]` and non-zero impulse sets. `stats[5]` relative difference 1.1e-7 (float atomics, as in PR #14). |
| Byte-identical claim for existing callers | **PASS** | §3.5. The author's hash files recompute: 26/26 identical, 5891 steps, 2585 in the v2 file. 12 of the 26 tests take no solver step (finding 5). |
| Freeze: `acceptance_v2` files, thresholds, records, harness | **PASS** | §5 |
| Receipt change against frozen comparisons | **PASS** | §5. The receipt is recorded (`run_perf_acceptance.py:276`), never compared. |
| Tests run here (CPU) | **PASS** | §8: 28 passed; 127 passed + 1 skipped; 130 passed + 5 skipped; torch venv 186 passed + 1 skipped. Every skip is torch or CUDA absence (NOT_RUN there), the same as the PR. |
| GitHub CI on `27554d21` | **PENDING** | §6 |
| PR body and report | **PASS** | §6. Accurate; no host paths in the body. Report notes: finding 5. |

## 3. Kernel and test correctness

### 3.1 The `shrink_engaged` counter

**Where it is written** (`particle_contacts.py:536-544`). Inside `if shrink != 0: if fast_flag[e] == 0:`, the thread
of particle p increments `shrink_engaged[e]` only if `p == 0` or `env_of_particle[p-1] != e`.
- Every thread of env e takes the same branches before that point: `guard_flag[e]` at `:530`, `shrink`, and
  `fast_flag[e]`. All are per env.
- So the env's first particle reaches the increment exactly when the env is queried with the shrunken reach.
- A guard-flagged env returns at `:530`, and its fast flag is also set (guard bound > V, NaN sets both). It is not
  counted, which is consistent: the frozen fallback queries it.
- With contiguous env ranges, which `layout.pack_from_beds` guarantees through `particle_offset`, there is exactly
  one writer per env and launch. The non-atomic read-modify-write cannot race.
- Launches on one stream are ordered, so graph replays accumulate deterministically.
- The array is allocated once (`:692`) and passed by pointer, so it is safe under graph capture.

**Measured on CUDA** (`probes/gpu_counter_ab.py`, N=16, GPU0):
- 7 launches with envs 1, 6 and 11 flagged fast.
- Counter: `[7,0,7,7,7,7,0,7,7,7,7,0,7,7,7,7]`, for the module kernel and for the verbatim copy.
- The same on CPU (N=2, `cpu_smoke_n2.json`).

**Partial reset** (`probes/reset_counter.py`, CPU, coarse N=3 TRIM, material level).
- Per control step the counter reads `[200,200,200]`, `[400,400,400]` (unchanged by `reset([1])`), `[600,…]`,
  `[800,…]`. The receipt shows the same values.
- This matches the docstring (`:60-61`, "cumulative since construction … partial reset does not clear").
- The counter sits on the contact object, so `test_reset_coverage_v2_is_vars_complete` does not classify it. That
  test enumerates arrays of the material, the solver and the diagnostics only (`tests/…_v2.py:414-425`).
- It is not observation data and nothing in the solver reads it.
- Semantics: finding 1.

### 3.2 The r/dt fail-closed check

`_launch_v2` (`:744-749`) raises `UnsupportedBatchConfiguration("UNSUPPORTED_PARTICLE_CONTACT_MODE")` when `r` or
`dt` differs from `construction_radius` / `construction_dt` (`:685`). The check is its first statement.

**Before it:** `launch()` (`:709-716`) only reads config and picks arguments. It mutates nothing.

**Callers:**
- `BatchedTLMPMv7_2.step` passes `dt, r = self.dt, self.radius` (`tlmpm_v7_2_batch.py:639, 672`).
- `run_p1_3a.contact_stage` passes `s.radius, s.dt` (`scripts/rl_ready/run_p1_3a.py:173`).
- Both are the same Python floats set once at construction (`tlmpm_v7_2_batch.py:585-586`). The `!=` comparison is
  exact, so it cannot refuse a current caller.
- A grep of `src/` and `scripts/` finds no later assignment to a batched solver's `dt` or `radius`.

**The test** (`tests/…_v2.py:314-330`):
- Covers 2·dt, 0.5·dt and 1.5·r.
- Asserts the impulse is unchanged and the counter stays `[0,0,0]`, which shows the contact kernel did not run.
- Then a launch with the construction values counts `[1,1,1]`.

**Scope of the guarantee:** it is atomic for the contact stage, not for the substep (finding 4).

### 3.3 Mutation spot-check (scratch copies, never committed)

Script: `mutations/mutate.py`; runner: `mutations/mutate_run.sh`. Each mutant runs in a copy of `src/`, `tests/`,
`configs/`, `assets/` and `pyproject.toml`; pytest's `pythonpath = ["src"]` makes it test the copy.

| Mutant | What it does | Tests | Result |
| --- | --- | --- | --- |
| A (author's) | kernel reads `fast_flag[p]` instead of `fast_flag[e]` | head, 5 selected | **4 failed**: both block scenes, the new test (bitwise, `:291`), the r/dt test (`:330`). The author's run failed the necessity test instead of the block scenes. The read is out of bounds on CPU, so the set varies (finding 3). |
| E (author's) | `_clear_flags_b` launched before the contact kernel | head, 5 selected | **4 failed**: both block scenes, the necessity test, the new test (`:291`). Same as the author's. |
| R (new) | the check compares `r` only | head, refusal test | **failed** (2·dt not refused) |
| U1 (new) | unsafe reach `1.6 r` | head, full v1+v2 files | **3 failed**: the new test's checked batch, bitwise (`:291`); the necessity test (`:167`, its control batch, envs 0/2); the receipt test (`:451`, `2r < reach_slow`). 25 passed. |
| U1 against the base tests | same mutant with the `9ee1e9d` test file | base, full v1+v2 files | **2 failed**: the necessity test's control batch and the receipt inequality. No bitwise test of a checked batch caught it. |
| U2 (new) | reach exactly `2 r` (no `2 dt V`, no slack) | head, 5 selected; base, full v2 file | head selection **passed**. Base full file: only the receipt test failed, at the strict `2r < reach_slow` (base `:280`; float32 2r is just below float64 2r). Its next line, `reach_slow >= 2r + 2dtV`, would fail too. |

**Reading:**
- The new scene is the first test whose checked batch fails bitwise under an unsafe shrunken reach in slow envs
  with contacts. That is the gap of PR #14 finding 1.
- U2 drops only pairs with d in [2r, 2r + dt·|v_rel|). The scene's relative speeds are about 0.1 m/s, which gives a
  band of about 10 µm. No bitwise scene can see it. The direct unit inequalities in the receipt test are the guard
  there, and they fail.
- `mutations/first_attempt_missing_configs/` keeps my first U1 runs. They failed the TRIM tests for a missing
  `configs/` in the copy: a set-up error of mine, not a result.

### 3.4 The slow-contact scene

`probes/slow_scene_table.py` replays the new test's checked batch and prints every substep.

**Scene:** 240, 196 and 225 particles; r = 0.9 mm; dt = 1e-4 s; reach 1.850 mm shrunken, 2.16 mm frozen.

**What each env does:**
- **Envs 0 and 2:** the counter rises by 1 in all 24 substeps. Contacts per substep: env 0 210 every time; env 2
  200 falling to 150. Maximum speeds 0.028–0.078 m/s.
- **Env 1:**
  - Substeps 0–5: engaged, 168 contacts each.
  - Substeps 6–8: not engaged; speeds 20.0, 19.3 and 18.6 m/s; 249, 189 and 183 contacts.
  - Substeps 9–23: engaged again, 116–168 contacts in **every** one of the 15 substeps. The test asks for at
    least 5 (`:301`; finding 5).
- Every env is bitwise in every substep.
- Receipt: `[24, 21, 24]`.

These match the report's description (it gives totals 5040, 3777 and 4211 for the unboosted run).

### 3.5 Byte-identical evidence for current callers

From the author's raw files (`/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/p1_6_p16c1_followups/`):
- `before.jsonl` at `9ee1e9d` and `after.jsonl`: 26/26 existing tests have identical per-step hash sequences.
- 5891 hashed steps, 2585 of them in the v2 file. `before_commit.txt` is `9ee1e9db…`.
- 14 of the 26 tests take solver steps. The other 12 (grid sequence, receipt, resolve and coverage tests) are
  trivially identical (finding 5).
- The plugin hashes every Warp array attribute of the solver after `step()`. That excludes the contact object's
  arrays (`fast_flag`, the buckets, the new counter), as the report states.

## 4. GPU cost of the counter (A/B, `SHARED_GPU`)

**Analysis.** Extra work per thread in a slow env:
- one load of `env_of_particle[p-1]`, the same cache line as `env_of_particle[p]` for 31 of 32 lanes;
- a compare and a branch;
- for one thread per env (16 of 1.64 M threads at N=16), a global read-modify-write.

In fast envs the branch is not taken. Against that stands the candidate walk, about 25 candidates per particle with
several dependent loads each. The expected cost is well below 1 %.

**Probe.** `probes/gpu_counter_ab.py` under `probes/run_gpu_ab_locked.sh`.

*Set-up:*
- Blocking `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock`, `-w 3600`.
- GPU pinned by UUID `GPU-76f61dc7…` (nvidia-smi index 0), with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- 28,516 MiB free at acquisition.
- Lock held 23:43:59–23:44:49 UTC. The P2-5 fit job takes this lock per forward evaluation, so queued jobs
  interleave between its evaluations by design.
- Co-tenant: PID 642659, not lock-governed. GPU utilisation was 96 % at acquisition and 0 % at the end. Label:
  **SHARED_GPU**.
- Workload: N=16, s512 at 1 mm spacing (102,732 particles per env, 1,643,712 in total), dt 1e-4, alias-free dims
  189×69×65, reach 0.950 mm shrunken and 1.080 mm frozen. These match the RL env defaults (`vector_env.py:94`).

*Method:*
- `_particle_contacts_env_major_b` is extracted verbatim from `git show` at `9ee1e9d` (base) and `27554d21` (head)
  into one scratch module.
- The head module's own kernel is timed too. The script asserts that the extracted head text occurs verbatim in the
  imported module.
- Variants run in rotating order (ABC, CBA, …). Each takes 20 launches between CUDA events. 60 rotations.

*State:*
- WP-17 Native s512 at sample 34 (control step 340, in the particle-contact window; 27,718 non-zero codes; maximum
  |v| 0.060 m/s), written into all 16 envs in solver-local float32.
- And at sample 0 (reset).

| State, scenario | base ms/launch | head ms/launch | head (module) | head/base ratio, median [p10, p90], rotations > 1 |
| --- | --- | --- | --- | --- |
| s512 step 340, all slow (shrink engaged, counter runs) | 1.0192 | 1.0197 | 1.0198 | **1.0006** [0.9985, 1.0023], 37/60 (module 1.0005, 35/60) |
| s512 step 340, all fast (frozen reach, counter skipped) | 1.2431 | 1.2437 | 1.2441 | 1.0009 [0.9984, 1.0030], 37/60 (module 1.0009, 40/60) |
| s512 reset, all slow | 1.0838 | 1.0829 | 1.0829 | 0.9991 [0.9975, 1.0016], 18/60 (module 0.9991, 18/60) |
| s512 reset, all fast | 1.2939 | 1.2941 | 1.2942 | 1.0001 [0.9984, 1.0021], 31/60 (module 1.0003, 31/60) |

**Registers:** 66 (base) → 68 (head) per thread, 0 bytes of local memory (`cuFuncGetAttribute`). The timing shows no
occupancy change. With per-warp register allocation in units of 256, both values likely round to the same
allocation; this is an expectation, not a measurement.

**Effect on the step.**
- At N=16 the kernel takes about 1.02 ms per substep, 0.204 s per 200-substep control step. That is about 12 % of
  the accepted v2 step of 1.66 s (`p1_6_p16c1_acceptance.json`, `performance.phases`, timed window 1.6596 s).
- The median delta of 0.06 % on the kernel is about 0.12 ms per control step, **0.007 %**. Even the p90 ratio
  (1.0023) gives 0.03 %.
- The accepted N=16 speed-up of 1.279 cannot be eroded measurably. The SHARED_GPU acceptance noise is on the order
  of several percent (its lowest pair was 1.142).

**Parity on the device, base against head** at the contact-state launch, with envs 1, 6 and 11 flagged:
- `stats[4]` is identical (14 non-zero pair impulses per env);
- the non-zero impulse sets are identical;
- `stats[5]` differs by at most 1.1e-7 relative (float atomic order, as for every GPU v2 run).

The reset state has 0 particle contacts.

**Limits:**
- This is a kernel-level A/B, not an end-to-end acceptance. The other kernels of the stage and the graph are
  unchanged.
- The `r`/`dt` check is host Python. Inside the material's CUDA graph it runs only at capture, so it costs nothing
  per replay.

## 5. Freeze compliance

**Files:**
- `git diff --name-only 9ee1e9d..27554d21` lists only the three files above.
- It is empty for:
  - `configs/`, `scripts/`, `.github/`, `pyproject.toml`, `docs/evidence/`;
  - every `frozen_paths` glob of `acceptance_v2.json` (`tlmpm*.py`, `fixture_contacts.py`,
    `material_diagnostics.py`, `live_material.py`, `coupled_robot.py`, `goals.py`, `seeded_trim.py`,
    `scene_config.py`, `control.py`, `evaluation.py`, `recording.py`);
  - the non-test `evaluator_files` (`src/robotic_cutting_sim/rl/{perf_acceptance,replay,benchmark,telemetry}.py`,
    `profiling.py`).
- The harness `scripts/rl_ready/run_p1_3a.py` is unchanged.

**Pins:**
- `acceptance_v2.json` sha256 `b66ace29…` and `acceptance_v1.json` `3592d940…` equal their pins.
- `tests/test_acceptance_frozen.py`, `tests/test_batched_frozen_sources.py` (frozen solver modules) and
  `tests/test_perf_acceptance.py` pass here (§8).
- `particle_contacts.py` is deliberately outside the evaluator and frozen sets: `test_perf_acceptance.py:152`
  asserts it is not covered.

**Receipts:**
- The v2 receipt gains `shrink_engaged_substeps` (`:807-808`). `run_perf_acceptance.py:276` records the receipt in
  the run results; the judge, the evidence makers and the tests never compare it with a stored value.
- The v1 receipt is unchanged: `test_v2_receipt_resolve_and_v1_unchanged` passes, and v2-only keys are filtered.
- `source_sha256()` (`tlmpm_v7_2_batch.py:536-540`) changes for every mode, because the file changed. It is a
  fingerprint: `test_batched_coupling.py:110-111` recomputes it from the live files, and nothing compares it with a
  record.
- `docs/evidence/rl_ready/p1_6_p16c1_acceptance.json` is untouched and judged `7053c965`. This PR does not claim to
  be accepted under `acceptance_v2`. It is a human follow-up with no speed claim, and §4 shows that none is needed.

**Test files under the evaluator set:**
- `tests/*` is in the `acceptance_v2` evaluator set, the same glob the PR cites for leaving `run_p1_3a.py` frozen.
- Editing this opt-in test file is appropriate. The judge runs from the evaluator checkout, i.e. `main` at
  acceptance time, which every merged PR moves (PR #14 added this file). The edit strengthens assertions and changes
  no threshold, harness, plan or record.
- After merge, G1 holds later candidates that touch v2 to the counter and receipt semantics. That is intended.
- Leaving `run_p1_3a.py` untouched is the right call: `run_perf_acceptance.py` imports it for G4, G6 and G7, so an
  edit would change the measurement of `acceptance_v2` itself (finding 6).

## 6. CI, PR body and report

**CI** (`gh api repos/AnsonNing/RoboticCuttingSim/actions/runs?head_sha=27554d21…`):
- push run 43120401944 and pull_request run 43120447021, workflow "CPU tests".
- Both were `in_progress` at 23:46 UTC.
- Steps 1–7 succeeded, including "Install media test tools"; step 8 (`pytest -q -rsx`) was running.
- PR #14's runs took about 6450 s. **PENDING**; see the merge condition in §9.

**PR body:**
- No host paths (regex check for `/home/`, `/mnt/`, `~/` and the user name: none).
- Every claim checked holds:
  - the counter's one writer;
  - the scene, its assertions and the control batch;
  - the five mutants: A and E re-run here; B, C and D taken from the author's `mutations.log` and
    `mutations_reasons.log`, whose `mutate.py` replacements I read;
  - the vacuous assertions replaced;
  - the r/dt refusal and its test;
  - finding 3 documented only;
  - 26/26 tests and 5891 steps identical;
  - GPU marked NOT_RUN;
  - 28 contact tests passed.
- The open items are listed (finding 2, the v2 audit, findings 6 and 7).

**Report** `P1_6_P16C1_FOLLOWUPS.md`: accurate. Notes in finding 5.

## 7. Findings, ranked

1. **LOW — the counter's semantics differ from its receipt neighbour.**
   - `shrink_engaged_substeps` is cumulative since construction, and partial reset does not clear it. This is
     documented (`:60-61`, `:691`).
   - `guard_violation_substeps`, next to it in the same receipt, is cleared per env on reset.
   - The receipt carries no denominator, such as the number of v2 contact stages.
   - The int32 wraps after 2³¹ substeps, about 10.7 M control steps per material instance. That is unreachable in
     practice.

   *Failure scenario:* an RL monitoring script reports the per-episode shrink engagement as
   `shrink_engaged_substeps[e] / (episode_steps × 200)`. After `reset([e])` the value is inflated, possibly above 1,
   and silently wrong for that env.

   *Recommendation (not blocking):* make the scope explicit in the receipt, for example a sibling key
   `shrink_engaged_scope: "since_construction"`, or a stage counter as denominator. Alternatively, clear it in the
   masked reset like the guard arrays.

2. **LOW — one writer per env relies on contiguous env particle ranges.**
   - `env_of_particle[p-1] != e` marks the first thread of every run of env e.
   - This holds for `layout.pack_from_beds`.

   *Failure scenario:* a later optimizer candidate reorders particles, for example a spatial sort for locality.
   Several threads per env then increment non-atomically. On CUDA the counter overcounts or loses updates and becomes
   nondeterministic. Physics is unaffected.

   *Mitigation:* the new test's `engaged == 1` per substep (`:296`, `:300`) fails on CPU for any such layout, so the
   regression would be caught in G1.

   *Recommendation:* none needed now. A one-line comment at `:539` naming the contiguity assumption would help.

3. **INFO — mutant A is undefined behaviour on CPU.**
   - `fast_flag[p]` reads beyond the N-element array.
   - Which tests fail differs between runs: the author's run failed the necessity test; mine failed both block-scene
     tests.
   - The new test and the r/dt test fail in both. The report's table should not be read as deterministic for A.

4. **INFO — the refusal is atomic for the contact stage, not for the substep.**
   - Through `BatchedTLMPMv7_2.step()` a mismatch is possible only if `solver.radius` or `solver.dt` were assigned
     after construction. Nothing does that today.
   - If it happened, the earlier kernels of that substep (stress, rigid, obstacle and fixture contacts) would already
     have run before the exception, and that solver would have to be discarded.
   - Inside the material's graph, `r` and `dt` are fixed at capture.
   - The PR's wording, "before anything runs", is accurate for `_launch_v2`.

5. **INFO — evidence wording.**
   - "All 26 existing tests … identical": 12 of them take no solver step. The step evidence rests on 14 tests and 5891
     steps. The numbers themselves are right.
   - The report gives the evidence location as "`artifacts/rl_ready/p1_6_p16c1_followups/` (on /mnt/data)". The files
     are at `/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/p1_6_p16c1_followups/`.
   - `:301` asks for at least 5 post-transition substeps with contacts, where the scene has 15/15. `.all()` would be
     a tighter pin.

6. **INFO — the freeze rationale could name the distinction it relies on.**
   - The PR's reason for not touching `run_p1_3a.py` is "in the evaluator file set". That covers `tests/*` too, which
     the PR edits.
   - The real distinction is between measurement code imported by the acceptance harness and opt-in unit tests (§5).
   - The decision is correct. Only the stated reason is incomplete.

Still open from the PR #14 review, not regressed: finding 2 (O(N·C) memory, N ≤ 2495), finding 3 (the v2 harness
audit and the dead host rebuild, for `acceptance_v3`), findings 6 and 7.

## 8. Tests and probes run for this review

All CPU runs in `/home/ningan/RoboticCuttingSim-rl-pr15` at `27554d21`, `CUDA_VISIBLE_DEVICES=''`, Warp 1.17.0.
Logs in `/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/p1_6_p16c1_pr15_review/`.

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -p no:cacheprovider tests/test_batched_particle_contacts_v2.py tests/test_batched_particle_contacts.py -rA -q --durations=15` | **28 passed**, 240.6 s (`contacts.log`). Neither new test is among the 15 slowest (each under 0.28 s). |
| `.venv/bin/python -m pytest -p no:cacheprovider tests/test_batched_hashgrid_guard.py tests/test_repo_naming.py tests/test_perf_acceptance.py tests/test_batched_frozen_sources.py tests/test_acceptance_frozen.py -q -rs` | **127 passed, 1 skipped** (`test_perf_acceptance.py:344`, torch not in `.venv`; run below), 15.2 s (`guard_naming_perf.log`). My first attempt never ran: a shell variable error of mine sent its log to `/`; rerun as above. |
| `.venv/bin/python -m pytest -p no:cacheprovider` on every other `tests/test_batched_*.py` plus `tests/test_vector_env_*.py` | **130 passed, 5 skipped** (torch not in `.venv`: NOT_RUN there, run below), 394.6 s (`batched_vec.log`) |
| `PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaaclab-venv/bin/python -m pytest -p no:cacheprovider` on `test_vector_env_{api,consolidation,contract,p12b,gpu_stream}.py` plus the `acceptance_v2` `torch_tests` (`test_rl_policies_batched`, `test_rl_replay`, `test_rl_actions`, `test_rl_observations`, `test_rl_reward_packed`, `test_perf_acceptance`) | **186 passed, 1 skipped** (`test_vector_env_gpu_stream.py`: CUDA hidden, NOT_RUN), 347.8 s (`torch_tests.log`) |

**Mutants:** §3.3, logs in `mutations/`.

**Probes** (`probes/`):

| Probe | Result |
| --- | --- |
| `gpu_counter_ab.py` via `run_gpu_ab_locked.sh` (GPU0, `gpu0.lock`, 50 s) | §4: `gpu_ab_n16_s34.json`, `gpu_ab_n16_s0.json`, `gpu_ab_locked.log` |
| `gpu_counter_ab.py cpu 2 34 2 1` (CPU smoke) | Native load, counter and parity checks pass (`cpu_smoke_n2.json`) |
| `slow_scene_table.py` (CPU) | §3.4 (`slow_scene_table.out`) |
| `reset_counter.py` (CPU, material partial reset) | §3.1 (`reset_counter.out`) |

## 9. Conditions and recommendations

1. **Merge condition:** GitHub CI "CPU tests" on `27554d21` green in both runs. If the head changes, this review
   covers `27554d21` only.
2. **Follow-ups, none blocking:**
   - finding 1, the receipt scope or a denominator;
   - finding 2, a contiguity comment;
   - finding 5, the evidence path in the report;
   - the open PR #14 findings 2, 3, 6 and 7 as listed.
