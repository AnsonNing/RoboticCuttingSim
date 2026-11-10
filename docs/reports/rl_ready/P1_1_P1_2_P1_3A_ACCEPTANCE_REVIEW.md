# P1-1 / P1-2 / P1-3a — xhigh acceptance review (2026-11-10)

**Scope.** This review covers three work packages:
- P1-1, the true GPU batched `tlmpm_v7_2` material;
- P1-2, the vector env API with partial reset and milestones M0/M1/M2;
- P1-3a, the opt-in `alias_free_v1` particle contacts.

**Commit.** The review is fixed at **3682786** (merge of `claude/rl-p1-3a` into `claude/rl-ready-rubber-physics`). It
was done read-only in a detached worktree checked out at 3682786. When this file was written, the branch head was
d029138. The three commits after 3682786 change only `docs/RL_READY_PROGRESS.md` and add `P1_5_PLAN.md`; they are
mentioned only where they touch a finding.

**Inputs read:**
- plan §1, §2, §4 and the specification;
- ADR 0021;
- the progress file;
- the reports P1_0, P1_1_CORE_REVIEW, P1_1_P1_2_MILESTONES, P1_3_BOTTLENECK_ANALYSIS and P1_3A;
- the evidence files `p1_2_milestones_001.json` and `p1_3a_001.json`;
- the code under `batched/` and `rl/`, the tests, and the three scripts;
- the raw artifacts under `artifacts/rl_ready/{p1_1,p1_2,p1_3}`, recomputed where a claim depends on them.

**Method.** CPU tests and two CPU probes only. No GPU job was run; the GPU jobs this review would ask for are listed
in §6.

## 1. Verdicts

| Package | Verdict | Basis |
| --- | --- | --- |
| **P1-1** true GPU batched material | **PASS_WITH_LIMITATIONS** | True batching, isolation and CPU bitwise parity verified; GPU invariants and Native final codes hold in every replayed episode. Limitations: the default-path HashGrid guard gap (F1), the mis-specified L3 position band (F2), L4 BLOCKED, no EXCLUSIVE_GPU timing. |
| **P1-2** vector env, partial reset, M0/M1/M2 | **PASS_WITH_LIMITATIONS** | API, partial reset, autoreset, observation split and host-sync contract verified in code, tests and probes. M0/M1/M2 physics outcome claims match the raw results. Limitations: the M2 gain is overstated by mismatched windows (F3) and compared against the wrong baseline (F4); autoreset makes the frozen evaluator unusable (F5); M1 GPU isolation is outcome-level only (F8). |
| **P1-3a** opt-in `alias_free_v1` | **PASS_WITH_LIMITATIONS** | Opt-in; default `frozen_v1` device work unchanged; CPU L0 bitwise; GPU same-state PASS; Native final codes and frozen evaluator PASS in every episode; 1.40–1.45× measured one variant per process. Limitations: SHARED_GPU and approach-only window (N1); stale and contradictory evidence/report text (F6, F2); recorded multi-variant commands no longer reproducible at 3682786 (F7). |

No BLOCKER was found.
- No frozen module changed. Since the RL-ready start (`git diff --name-status f242881^ 3682786`), the only
  pre-existing file modified is `pyproject.toml`, which gained the pytest `pythonpath = ["src"]`.
- No geometric deletion is presented as cutting.
- No v1 threshold or record was touched.
- No claim rests on a run labelled `INVALID_MULTI_GRID_PROCESS`.

## 2. Findings (most severe first)

### F1 — MAJOR — default `frozen_v1` path silently uses a later-built device grid; detection only on a full `reset()` (P1-1, P1-2)

**Where:**
- `src/robotic_cutting_sim/batched/material_batch.py:505-507`: the guard runs once, at construction.
- `material_batch.py:755-758`: the descriptor is restored before the launch only when `s.particle_contact` is set,
  that is, only for `alias_free_v1`.
- `src/robotic_cutting_sim/rl/vector_env.py:466-469`: the re-check runs only for `reset()` with `env_ids=None`.
- The limitation is stated in the docstring at `material_batch.py:415-416` and in the milestones report at
  `P1_1_P1_2_MILESTONES.md:291-292`.

**Mechanism (confirmed in the Warp source).** `warp/native/hashgrid.cpp:319-350` writes the function-static
descriptor only in `hash_grid_update_device_impl`, i.e. on `build`. Any CUDA `wp.HashGrid` built on the host after
the batch's capture therefore redirects the batch graph's captured `memcpy_h2d`.

**Failure scenario.**
1. A user opens `VectorCuttingEnv(16, autoreset="same_step", strict=False)` and calls `reset()`.
2. Later, in the same process, they build a frozen single `LiveCuttingMaterial` on CUDA, for example for a
   single-vs-batch comparison or a diagnostic. Its `tlmpm_v5` constructor builds `HashGrid(64,64,32)`.
3. Every following `step()` replays the batch graph with the single material's descriptor: its dims, its
   cell-start/end and point pointers, and its `num_points`.
4. `_particle_contacts_b` then queries the wrong grid for all 16 envs. In the P1-3a probe the candidates dropped from
   8,804,086 to 0. With a larger foreign `num_points` the reads can go out of bounds.
5. Nothing flags it:
   - with autoreset the env never calls a full `reset()` again;
   - `strict=False` reads nothing back;
   - L2 monitoring is not part of `step()`;
   - when the full-reset check finally fires, it raises after the fact and leaves no mark in `invalid_reason` or in
     the receipts.

**What is not affected.**
- A second batch is refused at construction, so the common case is covered.
- `alias_free_v1` protects its own graph, though not the other grid's graph.
- The evidence processes are one grid each (§3.2).
- The gap is documented.

So this is not a blocker for the evidence. It is, however, a silent-wrong-physics hazard on the default runtime
path, which P1-4 long rollouts and the P1-5 RL smoke will exercise.

**Recommendation.** Before P1-4/P1-5 rows are scored, add a versioned default-path guard. Two options:
- a host rebuild of the own grid before each `capture_launch`, as `alias_free_v1` already does; the hardened P1-3a
  runs measured no cost (§3.4 of the P1-3a report);
- a process-wide build counter checked in `advance()`.

On detection, set a sticky `invalid_reason` bit and mark the receipt; do not only raise at the next full reset.

### F2 — MAJOR — the L3 position band cannot serve as an acceptance criterion, and its status is misreported (P1-1, P1-2, P1-3a)

**Where:**
- the criterion: ADR 0021 §5 L3 (`docs/adr/0021-batched-rl-gpu-headless-runtime.md:263-265`);
- the band: `p1_0_baseline_001.json`;
- the claims: `P1_1_P1_2_MILESTONES.md:18,151,153`, `p1_2_milestones_001.json:9` (`M0_L3_position_band`),
  `P1_3A_ALIAS_FREE_CONTACTS.md:19,254,266,351`.

**The band is statistically unusable.** The band is [min, max] of 4 single-env GPU replays. For runs that come from
the same process, a new run falls outside the range of n earlier runs with probability 2/(n+1) = **40 %**. The
criterion therefore rejects conforming runs at a high rate. P1-0 itself marked it "reported, not scored"
(`P1_0_BASELINE.md:21`).

**The status does not match the raw data.**
- **s512 band.** The maximum over time is 1.39105e-5 to **3.84649e-5** m.
  - `m0_s512_postfix_001` measured **3.85346e-5** m, which is above the band maximum. It is reported as "inside (at
    the band max)".
  - P1-3a `episode_512_15_512_15_001` env 0 measured 3.85294e-5 m, also above the maximum. It is reported as
    "inside" / "edge of band".
- **s15 band.** The minimum is **1.95695e-4** m.
  - `m0_s15_postfix_001` measured 1.86602e-4 m, which is below the band. The status table says "post-fix s15 inside".
  - P1-3a's s15 envs measured 8.34e-4, 8.20e-4 and 5.98e-4 m (above) and 1.73e-4 m (below). That is 4 of 4 outside;
    the report says "3 of 4".
- **Strict tally.** M0 is 1 of 4 inside. P1-3a is 3 of 4 inside for s512 and 0 of 4 for s15.
- **Contradiction.** `P1_3A_ALIAS_FREE_CONTACTS.md:351` ("inside the frozen GPU band on GPU") contradicts its own
  :19 and :266.

**Why the verdicts stand anyway.** The physics outcome evidence is strong and independent of the band: final codes,
labelled counts and the frozen evaluator agree with Native in every replayed episode: M0 4, M1 replay envs 4,
M2 40, P1-3a 8.

**Recommendation.**
- Report the L3 position criterion as **NOT_MET / INCONCLUSIVE (band under-sampled)**, not as "3/4 inside".
- Fix the wording at the places listed.
- Before scoring positions again, derive a tolerance from at least 10 runs per seed on one boundary, or state a
  physical tolerance.

**Note on alias_free_v1.** Its s15 maxima average 6.1e-4 m against about 3.5e-4 m for the frozen runs. With 4
against 6 samples a rank test gives p ≈ 0.3, so there is no detectable shift, but also little statistical power.

### F3 — MINOR — the M2 batching gain compares mismatched episode windows (P1-2)

**Where:** `P1_1_P1_2_MILESTONES.md:24,78,250`; `p1_2_milestones_001.json:22`.

**What happened.** The alternation timed different step windows:
- N=1: steps 0–300;
- N=8: steps 0–150;
- N=16: steps 0–100 (`--max-global-steps {300,150,100}`).

The N=1 per-step cost rises in steps 200–300:
- `alt_r1_n1_001` `steps.npz`: 3.130 env-steps/s in [200:300] against 3.469 in [0:100];
- the same pattern appears in `m2_n1_001` and `m2_n1_002`.

**Recomputed with matched windows** (raw `step_wall_s`):

| Ratio | Recomputed | Reported |
| --- | --- | --- |
| N=8 / N=1, steps [0:150] | 4.850 / 3.475 = **1.40×** | 1.45× |
| N=16 / N=1, steps [0:100] | 4.752 / 3.469 = **1.37×** | 1.42× |

**Effect.** M2 "measured gain over N=1" still holds. The progress update at 7bfa0b6 repeats 1.45×/1.42×.

**Recommendation.** Report 1.37–1.40× and use equal windows in future alternations.

### F4 — MINOR — "inside the ADR 0021 estimate of 1.3–3×" uses a different baseline (P1-2)

**Where:** `P1_1_P1_2_MILESTONES.md:254` against ADR 0021 `:524-525`.

**The mismatch.**
- The ADR estimate is relative to *sequential single envs*.
- The measured ratio is relative to *batched N=1*.
- The same report measures batched N=1 about 10 % slower than the frozen single env at matched load (5.091 against
  5.630 env-steps/s at 24 %; `:241-243`).

**Inferred gain.** Against the frozen single env the gain is about 1.37–1.40 × 0.904 ≈ **1.24–1.26×**. This is
inferred, not measured. It falls below the ADR's lower estimate (risk #6 of ADR §14).

**Recommendation.**
- Drop the "inside the estimate" sentence, or measure a `single_env_headless` row inside the same alternation round.
- Keep the two boundaries as separate rows, as plan §2 requires.

### F5 — MINOR — `autoreset="same_step"` makes the frozen evaluator unusable for finished episodes (P1-2)

**Where:**
- the autoreset: `vector_env.py:604-609`;
- `evaluate()`: `vector_env.py:640-674`;
- the ADR promise: ADR §8 "Success comes only from the frozen independent evaluator … at episode end".

**What happens.** The done envs are reset inside `step()` before the caller can snapshot them.

**Probe** (CPU, coarse, `scratchpad/probe/autoreset_evaluate_probe.py`):
- after a truncating step with autoreset, `evaluate([0])` scores the fresh state;
- the result has 1 frame, `step_count` 0, verdict **`FAIL_INTEGRITY`**, and no error or warning;
- an RL loop with autoreset therefore records every finished episode as a failure, or has no success signal at all.

**Effect on evidence.** None: the M2 runs used explicit resets after the evaluation.

**Recommendation.**
- Refuse `evaluate()` (or return `NOT_EVALUATED`) for an env whose episode was just autoreset or whose `step_count`
  is 0.
- Offer an opt-in "snapshot done envs before autoreset" path, outside the default hot path.
- Document the limitation in ADR §8.

### F6 — MINOR — stale or contradictory evidence and document text (P1-1, P1-2, P1-3a)

**`docs/evidence/rl_ready/p1_3a_001.json:71-75`.**
- `failures` still lists "end-to-end throughput gain not demonstrated on the SHARED GPU (interleaved walls within
  ±2 %)". That is the conclusion from the INVALID multi-grid runs, which the report withdraws
  (`P1_3A_ALIAS_FREE_CONTACTS.md:144`).
- It attributes the s15 band failure "to the frozen_v1 control", which comes from the INVALID `episode_15_001`.
- It says "2/3 envs" where the report says 3 of 4 and the raw data say 4 of 4.
- `not_run` says "3+ foreign compute processes" while most rows record 1.

**ADR 0021.**
- `docs/adr/0021-batched-rl-gpu-headless-runtime.md:733-735` still states "about 1.9× aggregate over sequential
  singles at N=4", with no inline marker.
- The withdrawal sits at `:866-870`, under the *P1-1b coupling* heading (`:847`), where a reader of the P1-1 notes
  will not find it.
- The report (`P1_1_P1_2_MILESTONES.md:135`) and the evidence (`p1_2_milestones_001.json` audit) do withdraw it. The
  progress file has no 1.9× claim.

**`docs/RL_READY_PROGRESS.md:11-37` at 3682786** still says that:
- the P1-2 single-ref and nsys runs are "in progress";
- P1-3a will "merge here after the P1-2 chain ends";
- agents are "auditing".

This is fixed at 7bfa0b6, after the reviewed commit.

**Recommendation.**
- Add a dated correction block to `p1_3a_001.json`, or regenerate it as `_002`, keeping `_001`.
- Annotate ADR line 734 inline ("withdrawn, see …").

### F7 — MINOR — the recorded P1-3a multi-variant commands cannot be reproduced at 3682786 (P1-3a)

**Where:** `scripts/rl_ready/run_p1_3a.py:94-99` (`open_env`) and `vector_env.py:192-195`, which has no
`allow_multiple_device_hashgrids` pass-through.

**What fails.**
- `--mode samestate` and `--mode stagebench` (`P1_3A_ALIAS_FREE_CONTACTS.md:326-329`) keep several
  `VectorCuttingEnv`s open in one process.
- After the merge the guard refuses the second one on CUDA with `UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS`.
- The evidence came from the pre-merge commits a18e8f5 and 39d96f3, whose hashes are recorded in the manifests.
  The −51 % / −49 % contact-stage numbers therefore cannot be regenerated from the merged tree.

**Recommendation.**
- State in the report that these modes need the pre-merge commit, or add an explicit diagnostic opt-in through the
  vector env, labelled in the receipt.
- Run each stage graph after `restore_descriptor`, as the runner already does.

### F8 — MINOR — M1 "env independence … bit-identical" holds on the CPU device only (P1-2)

**The criterion.** Plan `:122` requires non-reset envs to stay bit-identical to an un-reset run on the same device.

**Where it is shown.**
- Bitwise, it is shown only on the Warp CPU device: `tests/test_vector_env_api.py:151-176`, coarse 2.5 mm, N=3, plus
  the batched L1 tests.
- On GPU the partial-reset isolation evidence is outcome-level:
  - every replay env keeps Native final codes while other envs are reset (M1 resets at 300/400/500/600; M2 staggered);
  - the grouped query finds 0 foreign candidates;
  - L4 (RUN_TO_RUN), which would make it bitwise, is BLOCKED.

**Status wording.** `P1_1_P1_2_MILESTONES.md:19` reports M1 PASS without that qualifier.

**Recommendation.** "PASS (CPU bitwise L1; GPU outcome-level; L4 BLOCKED)".

### F9 — MINOR — performance rows leave required columns empty (P1-2, P1-3a)

**What is missing.** Plan §2 (`RL_READY_PLAN.md:39-46`) requires `particles/env` in every row. It is `n/a` in:
- 6 P1-2 rows (M0 ×4, M1 ×2);
- 5 P1-3a episode rows.

The counts are known: 102,732 / 103,898, mean 103,315.

**Other gaps.**
- Rows do not carry the `reset p95` that was measured (M0 full reset of 1.2 ms).
- There is no `single_env_headless` row at matched load in the alternation (see F4).

### F10 — MINOR — the prescribed Torch test command fails on out-of-scope P1-1b files (test hygiene)

**What happens.** The prescribed command collects every `tests/test_rl_*.py`.
- It stops at collection on `tests/test_rl_lab_scene_usd.py:14`, a module-level `from pxr import …` with no
  `importorskip`.
- Without that file it gives 15 failed and 6 errors in `test_rl_lab_{rigid_isolation,ik,kernels}.py`, all
  `ModuleNotFoundError: pxr` via `batched/lab_scene.py:623`.

**Why it matters.** These are P1-1b tests that need the Isaac Lab venv. They should skip as NOT_RUN, as
`tests/test_rl_lab_env.py:27` does, instead of failing. The in-scope result is in §4.

### Notes (no action required for acceptance)

**N1 — SHARED_GPU timings are quantised by co-tenant time-slicing.**
- In the N=1 solo runs, the `alias_free_v1` step walls cluster at 0.2017–0.2020 s.
- One `frozen_v1` step ran at 0.1407 s (`solo_n1_frozen_v1_002`), below every alias step (minimum 0.1758 s).
- The 1.40–1.45× is a SHARED number measured over steps 3–14, which is the approach phase with no particle contacts.
- The nsys kernel ratio at N=1 is about 1.5× (672 → 435–442 µs per substep, medians).
- At N=16 both repeats had matched load and both give 1.40×.
- EXCLUSIVE_GPU and RUN_TO_RUN are still NOT_RUN, so no ratio here is a hardware number. The report says so.

**N2 — The pre-hardening `alias_free_v1` processes held two live CUDA grids.**
- At 39d96f3 the `solo_*_alias_free_v1_*` and `episode_solo_*` processes kept the frozen-dims grid built at
  construction in `solver.hash`; d18b119 released it.
- They are valid by build order: the alias-free grid was the last one built on the host and the only grid rebuilt
  inside the graph (`run_p1_3a.py` builds only the active grid).
- Hardened reruns confirm them: N=1 gives 5.069 against 5.053 / 5.054; N=16 gives 6.552 against 6.561 / 6.556.
  Hardened episodes for s512 and s15 PASS.
- The report's "one variant per process = valid" wording hides the two-grid status, and the N=4/N=8 alias numbers
  rest only on those runs.
- `stagebench_*_002` are multi-grid processes with a host rebuild before each replay. They are valid by the same
  mechanism and correctly described.

**N3 — `restore_descriptor` adds a host build per control step.**
- In `alias_free_v1`, `restore_descriptor` (`particle_contacts.py:452-456`) runs a host `HashGrid.build` every
  control step, including an H2D copy from pageable static memory.
- In training mode (`strict=False`) this may bring back a per-step stream sync.
- The "0 host reads per step" test (`test_vector_env_api.py:124-148`) covers only `frozen_v1`.

**N4 — Sequencing against plan `:130` ("only after M2").**
- P1-3a CPU work (9988d8d, 05:16) began while `m2_n8_001` was running (started 04:40) and before `m2_n16_001`
  (06:51).
- The P1-3a GPU runs came after M2.
- It is opt-in and does not affect the M2 evidence.

**N5 — Frozen-source pins are incomplete.**
- The pin test (`tests/test_batched_frozen_sources.py:15-32`) does not cover `goals.py`, `policy_env.py`,
  `control.py`, `scene_config.py` or `tlmpm.py`…`tlmpm_v4.py`.
- None of them changed (git diff above).
- Recommend pinning them, since `evaluate()` runs `goals.evaluate_goal`.

**N6 — Guard implementation details.**
- The guard's `gc` scan raises a harmless torch `FutureWarning` (`torch.distributed.reduce_op`) on `isinstance`.
- `UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS`, when raised inside `VectorCuttingEnv`, is an
  `UnsupportedBatchConfiguration`, not the `UnsupportedVectorConfiguration` that plan §1 names. An `except` on the
  subclass misses it.

**N7 — `evaluate()` integrity is weaker than on a Native recording.**
- `evaluate()` sees 2 frames, so `persistent_history_no_digit_rewrites` checks only t=0 against the end.
- The M0 comparison of 2 and 121 frames was identical, and the milestone runs had the per-step L2 monitor.
- An RL caller who uses `evaluate()` alone gets this weaker integrity check. This is documented at
  `P1_1_P1_2_MILESTONES.md:289-291`.

**N8 — `info` carries oracle-derived terms.**
- It holds the severed bond counts and fractions, and `invalid_reason`.
- It is not an observation, but wrappers that fold `info` into the observation would leak privileged state.
- Mark these terms as privileged in the metadata.

**N9 — Strict mode is not transactional.**
- If `InvalidBatchEnvs` is caught and stepping continues, the commanded pose has already advanced
  (`vector_env.py:588-591`), but `previous_action`, `armed` and the reward baseline were not updated for that step.
- Document that stepping after the exception is unsupported, or make the step transactional.

**N10 — Env seeds do not drive physics or the task.**
- The material is deterministic and the geometry is fixed per slot.
- Seeds are bookkeeping for counter-based policies (`RandomPolicyV1`).
- This is consistent with the report. Seeded randomisation is P1-5.

## 3. What was verified (holds)

### 3.1 Physics correctness and parity (check 1)

**Batching is real.**
- One process, a flat structure-of-arrays layout, env-indexed copies of the frozen kernels and a grouped HashGrid.
- The nsys graph-node trace shows 15,434 against 15,436 kernel instances for N=1 against N=4 over 2 control steps:
  one launch per operation covers all envs.
- `step()` has no per-env Python loop. The host loops are only in validation, receipts and `evaluate()`.

**L0 (CPU bitwise).** The `.venv` suite passes (§4). The comparison is vars()-complete: `compare_env` checks every
Warp array of the frozen single solver after every control step. It covers:
- N=1;
- N=3 heterogeneous;
- identical copies;
- one full-size TRIM substep;
- the 47 + 8 WP-17 frontier fixtures;
- the P1-3a variants, including heavy aliasing and the speed-guard fallback.

**L1 (CPU isolation and reset).** The tests cover the perturbation case, partial and full reset, poison reset and the
caller-order seeds.

My probe (`scratchpad/probe/unsorted_reset_probe.py`, CPU, `VectorCuttingEnv`, N=3):
- `reset(env_ids=[2,0], seeds=[200,100])` is bitwise equal over all arrays to `reset(env_ids=[0,2], seeds=[100,200])`;
- the seeds come out as [100, s1, 200];
- duplicate ids are refused;
- a geometry change is refused (`UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET`) before any state changes.

**L2 (GPU invariants).** Recomputed from the raw `result.json`, L2 is PASS in every step of M1 (1,800 × 4) and M2
(3,950 × 8; 1,575 × 16):
- code monotonicity violations 0;
- nonfinite steps 0;
- slot overflow 0;
- `min detF` 0.84–0.91;
- grouped query foreign 0 (of 64–134 M visited);
- mass bitwise constant.

**L3 (outcome).** Raw counts:

| Run | Final codes vs Native | Frozen evaluator | Note |
| --- | --- | --- | --- |
| M2 N=8 | 24/24 equal, 0 particles differ | 24/24 PASS | |
| M2 N=16 | 16/16 equal | 16/16 PASS | |
| M1 replay envs | equal | PASS | |
| M1 policy envs | differ by 3,994 / 4,576 particles | PASS | expected: ideal tracking of the commanded targets |
| M0 ×4 | equal | PASS on 2 and 121 frames, identical | |

The position band is covered by F2.

**L4.** BLOCKED is confirmed by `l4_deterministic_002`: a `WarpCodegenError` in the frozen
`tlmpm_v5._particle_contacts`, which mixes add and max reductions on `stats`.

**Frozen sources untouched.**
- `git diff --name-status f242881^ 3682786` shows additions plus `pyproject.toml` only.
- The SHA256 pins pass, and the v7_2 pins match the WP-17 receipts.

**Default `frozen_v1` path.** `git diff 4fc431e 3682786 -- batched/tlmpm_v7_2_batch.py` adds only:
- the opt-in dispatch;
- the receipts;
- `particle_contact=None` for the default.

The default substep runs the same two statements as before, and the `_particle_contacts_b` source hash is pinned in
the test. `advance()` adds no device work in the default mode (`speed_guard="off"`, no restore). The graph is
therefore identical.

### 3.2 HashGrid static-descriptor hazard (check 2)

**What the guard does right.**
- It counts live CUDA `wp.HashGrid` objects. Since Warp writes the static only on `build`, counting live grids is
  conservative.
- It also covers grids on two different CUDA devices, which would otherwise copy device-0 descriptors to device 1.
- `rebuild()` exempts its own grid only while the new batch is built.

**Merge resolution.** `release()` now drops both `solver.hash` and `particle_contact.grid`, which are the same
object in `alias_free_v1`. Without this the grid would stay alive and a later batch would be refused. The new
parametrised test passes for both guard modes.

**Known gap.** See F1.

**Evidence processes.**
- **P1-2.** Every milestone mode in `run_vector_milestones.py` opens exactly one env (`mode_m0/m1/m2/profile/l4`),
  and the pack-time frozen solvers are CPU. The audit is correct.
- **P1-3a.** The rows use only `solo_*`, `hardened_*` and episode runs. Every multi-variant run (`ab_n*`,
  `smoke_ab`, `pair_n16_reversed`, `hardened_pair`, `episode_{512,15}_001`) carries an `invalid` label in the
  evidence and feeds no row. The one exception is the stale failure text (F6).
- **Withdrawn claim.** The P1-1 `ab_timing_001` "≈1.9×" is withdrawn in the report and the evidence. The ADR still
  has it inline (F6).

**Other code paths.** The frozen evaluator (`goals.evaluate_goal`) and the L2 audit build no other device grid. The
only other `wp.HashGrid` users in `src/` are frozen single solvers and `live_acceptance.py`, neither of which is used
by these paths. Newton creates its `particle_grid` only for models with more than one particle
(`newton/_src/sim/builder.py:12590-12593`), and the P1-1b Lab env attaches the batch after the Newton model is built.

### 3.3 Vector env API (check 3)

**Step.**
- Torch tensors only, with no implicit copies (`_check_tensor`).
- In strict mode there is exactly one int32 readback per step; in training mode there is none. The test counts
  `wp.array.numpy` and `Tensor.item/tolist/cpu/numpy`, and code reading finds no implicit sync (no boolean-mask
  indexing, `nonzero` or Python `if` on tensors) in `actions.apply`, `reward.step` or `observation.compute`.
- Returned `obs`, `reward` and reward terms are fresh tensors. `info["final_obs"]` is not overwritten by the
  autoreset.

**Reset.**
- Unsorted ids and seeds are paired by sorting them together, and the scene check runs in the caller's order (probe
  above).
- Under autoreset, `_reset_device` re-baselines the reward counts. The next seed is `splitmix64(seed + episode + 1)`
  (tested).

**Observations.**
- The policy block is 18-dimensional: pose, time remaining, previous action, arm flag and step fraction.
- Policy terms are checked by name tokens and by `find_privileged_keys`.
- The critic block exists only with `privileged_state`.
- No force or oracle material state reaches `obs["policy"]` (AGENTS.md); see N8 for `info`.

**Errors.** Every unsupported configuration raises with a code that starts with `UNSUPPORTED`
(`test_every_unsupported_code_is_raised`, `test_unsupported_configurations_raise`).

**Evaluator wiring.** `evaluate()` calls the frozen `goals.evaluate_goal` with 2 frames (N7). Autoreset is covered
by F5.

### 3.4 Performance honesty (check 4)

**Columns.**
- Every evidence row carries N, control Hz, dt, cameras, env-steps/s, sim-time/wall, VRAM, reset p95, verdict,
  substeps, DOF, contacts, recording, warm/cold, GPU name, index and driver, `SHARED_GPU`, co-tenant statistics,
  source and commit.
- Exceptions are in F9.

**Boundaries.**
- The three boundaries stay in separate rows.
- No Native↔headless speedup is formed.
- The one cross-boundary sentence is F4.

**Co-tenant load.** It is reported per run, and the confounded comparisons are explicitly not used.

**The 1.40–1.45×.**
- It is measured one variant per process over the same window (steps 3–14) at every N.
- The co-tenant load is matched at N=16 in both repeats.
- See N1 and N2.

**Still NOT_RUN** (stated in both reports):
- EXCLUSIVE_GPU rows;
- RUN_TO_RUN / L4;
- C3;
- Native confirmation of the policy-driven episodes.

### 3.5 AGENTS.md and naming rules (check 5)

- No geometric deletion, presegmented part or renderer mask is presented as cutting. The cut codes come from the
  TLMPM solver.
- Unsuccessful runs are preserved:
  - `p1_1/gpu_smoke_001` and `ab_timing_001`;
  - `p1_2/smoke_*` and `l4_deterministic_001`;
  - `p1_3/samestate_*_001`, `stagebench_*_001`, `ab_*` and `pair_*`.
- v1 thresholds, physics and records are unmodified.
- The materials stay `SIMULATOR_ONLY_UNCALIBRATED`.
- `tests/test_repo_naming.py` passes.

## 4. Tests run by this review (CPU only, detached worktree at 3682786)

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_batched_*.py tests/test_repo_naming.py
# 168 passed in 493.46 s   -> PASS
CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python -m pytest \
    -q -p no:cacheprovider tests/test_vector_env_api.py tests/test_rl_*.py tests/test_batched_pose_interp.py
# interrupted: collection error tests/test_rl_lab_scene_usd.py (No module named 'pxr')   -> F10
# same without that file: 143 passed, 15 failed + 6 errors (all pxr, P1-1b test_rl_lab_*), 1 skipped
CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python -m pytest \
    -q -rs -p no:cacheprovider tests/test_vector_env_api.py tests/test_vector_env_gpu_stream.py tests/test_rl_actions.py \
    tests/test_rl_observations.py tests/test_rl_reward_packed.py tests/test_rl_replay.py tests/test_rl_benchmark.py \
    tests/test_rl_policies_batched.py tests/test_batched_pose_interp.py tests/test_batched_hashgrid_guard.py \
    tests/test_repo_naming.py
# 151 passed, 1 skipped (test_vector_env_gpu_stream.py: CUDA not available = NOT_RUN)   -> PASS (in scope)
```

**Probes.** Two CPU probes were written to the review session's scratchpad (not committed):
- `unsorted_reset_probe.py`;
- `autoreset_evaluate_probe.py`.

Results are quoted in §3.1 and F5.

**Raw-artifact recomputations.**
- the alternation windows (F3);
- the exact band membership (F2);
- the M1/M2 episode outcomes and L2 (§3.1);
- the source hashes of the P1-3a runs against their recorded commits;
- the one-variant-per-process status of the solo runs (N2).

## 5. Required before Phase 1 "done"

1. **F1.** Add a versioned default-path descriptor guard, plus a sticky invalid bit on detection, before P1-4/P1-5
   rows are scored.
2. **F2.** Change the L3 position criterion and correct the status wording. Re-derive the band from at least 10 runs
   per seed, or state a physical tolerance.
3. **F3–F6, F8, F9.** Correct the numbers and wording, keeping the corrections additive. Do not overwrite `_001`
   evidence; add a corrections block or a `_002`.
4. **F5.** Fix the autoreset/`evaluate()` contract before the P1-5 RL smoke relies on success metrics.
5. **EXCLUSIVE_GPU rows** for M2 and P1-3a, because every throughput number so far is co-tenant dependent.

## 6. GPU jobs this review would ask for (described, not run)

1. **EXCLUSIVE_GPU alternation.** Run batched N=1/8/16 and `single_env_headless` s512/s15 one process each, in the
   same window, over equal step windows that include the cutting phase (for example steps 0–600). Run
   `frozen_v1` and `alias_free_v1`.
2. **L3 tolerance.** Run at least 10 single-env and at least 10 batched N=1 replays per seed, one process each, to set
   a position tolerance from data.
3. **F1 demonstration.** Under `allow_multiple_device_hashgrids=False`:
   - build a batch, then a frozen single `LiveCuttingMaterial` on CUDA;
   - count the batch's contact candidates and `stats[4]` before and after;
   - repeat with the proposed default-path guard, to show the hazard is closed.

## 7. Review fixes (2480dc2)

Branch `claude/rl-review-fix1`, from 2480dc2. Each finding, its fix and the commit are listed in §7.9 when the work is
complete. Raw output: `artifacts/rl_ready/review_fix1/` (not committed). Evidence:
`docs/evidence/rl_ready/review_fix1_001.json`.

### 7.1 F2 — position band v2, pre-declared before any reference run

Declared in `configs/rl_benchmark/position_band_v2.json` (schema `rcs_rl_position_band_method_v2`) and committed
before the first band-v2 reference run. The old band (`position_band_v1_p1_0`, [min, max] of 4 runs) and every
status derived from it are kept and marked superseded; nothing is deleted.

- **Metric (unchanged):** `max_position_diff_m_over_time`, the max over the 121 Native samples of the max particle
  distance between the replay and Native WORLD positions, full 1200-step WP-17 replay.
- **Reference sample:** at least 10 frozen single-env (`LiveCuttingMaterial`) headless replays per seed (512, 15), one
  process each (`run_baseline.py --mode replay --warm-resets 0`), two chains on GPU 0 and GPU 1, each alternating the
  seeds. Every COMPLETE run counts; none is dropped for its value. P1-0, `p1_2_single_ref_s512_001` and all batched
  runs are held out and only re-classified.
- **Tolerance:** one-sided log-normal prediction bound `U = exp(mean(ln m) + k·sd(ln m))`,
  `k = t(0.99, n−1)·sqrt(1 + 1/n)`, nominal **false-reject rate 1 % per run** (assumptions: exchangeable runs,
  ln m normal). Shapiro–Wilk on ln m is reported; if p < 0.01 the bound falls back to the reference maximum (nominal
  false-reject rate 1/(n+1)), labelled `fallback_empirical_max`. The lower bound `exp(mean − k·sd)` is descriptive
  only: a run below it is `BELOW_REFERENCE` (closer to Native than the frozen spread), never a failure.
- **Per-run classes:** `WITHIN` (≤ U), `OUTSIDE` (> U), `INVALID_PROCESS`, `NO_POSITION_DATA`, `NOT_APPLICABLE`.
- **Per run set:** `MET` if the number of OUTSIDE runs r ≤ c(n), the smallest c with P[Bin(n, 0.01) > c] ≤ 0.05
  (c = 0 for n ≤ 5, c = 1 for 6 ≤ n ≤ 35); else `NOT_MET`.
- **Scope:** a statistical regression check of GPU float-atomic spread against Native, not a physical tolerance.
  Final codes, labelled counts and the frozen evaluator stay independent criteria.
- **Amendment 1** (committed 4d39747, after the reference processes had started but before any reference
  `final_max_position_diff_m` value was read; only one `max_position_diff_m_over_time` value had been printed): the
  P1-6 acceptance judge compares the **final** max position difference at step 1200 with
  `position_band_m[<seed>].high`, so `final_max_position_diff_m` becomes a second scored statistic with the same
  reference runs and the same rule. After derivation the method file carries `position_band_m =
  {"512": {"high_m": U}, "15": {"high_m": U}}` for that statistic (the judge's format) plus the band of both
  statistics; the method digest excludes these derived keys. `acceptance_v1.json` is not changed here.
