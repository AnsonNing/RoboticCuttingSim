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

Branch `claude/rl-review-fix1`, from 2480dc2. Each finding, its fix and the commits are listed in §7.8. Raw output: `artifacts/rl_ready/review_fix1/` (not committed). Evidence:
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

### 7.2 F2 — band v2 values and re-classification

**Reference runs** (`artifacts/rl_ready/review_fix1/band_v2_s{512,15}_g{0,1}_00{1..5}`): 20 processes,
`run_baseline.py --mode replay --warm-resets 0 --out-root artifacts/rl_ready/review_fix1`, frozen single
`LiveCuttingMaterial` (tlmpm_v7_2, law 8, one 200-substep CUDA graph), 102,732 / 103,898 particles, 50 Hz, 0.1 ms,
no cameras, no recording. GPU 1 chain 10:39–11:32, GPU 0 chain 11:03–11:57 (RTX 5090, driver 570.211.01), each
under its `rcs_rl_runner` lock, wall 224–343 s per process. Every run is COMPLETE, 1200 steps, invariants PASS and
final codes equal Native; none was rejected. All are **SHARED_GPU** (≥ 1 foreign compute process in every run; that
does not affect the positions, only time). Chain scripts: `artifacts/rl_ready/review_fix1/band_v2_chain.sh`.

| statistic | seed | n | reference range (m) | geometric mean (m) | sd(ln m) | k | Shapiro–Wilk p | **upper U (m)** | lower (descriptive) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max over time (primary) | 512 | 10 | 9.55e-6 – 4.79e-5 | 2.49e-5 | 0.547 | 2.959 | 0.48 | **1.254e-4** | 4.94e-6 |
| max over time (primary) | 15 | 10 | 6.90e-5 – 8.57e-4 | 1.76e-4 | 0.911 | 2.959 | 0.09 | **2.601e-3** | 1.19e-5 |
| final max (judge, amendment 1) | 512 | 10 | 9.55e-6 – 4.79e-5 | 2.49e-5 | 0.547 | 2.959 | 0.48 | **1.254e-4** | 4.94e-6 |
| final max (judge, amendment 1) | 15 | 10 | 4.52e-5 – 1.69e-4 | 7.54e-5 | 0.404 | 2.959 | 0.61 | **2.493e-4** | 2.28e-5 |

Both statistics use the parametric bound (no Shapiro fallback); nominal false-reject rate 1 % per run. For seed 512
the max over time is reached at the final sample in every run, so the two statistics coincide. The 20 frozen single
runs themselves span 5× (s512) and 12× (s15): **the v1 range of 4 runs covered only part of the frozen spread** (of the
20 new frozen single runs, which are by construction conforming, 4/10 (s512) and 9/10 (s15) fall outside v1 on the
max over time, 4/10 and 5/10 on the final max). The s15 max-over-time
band is wide (U ≈ 15× the geometric mean), so it detects only gross deviations; the final-max band is 3.3× the
geometric mean. Files: `configs/rl_benchmark/position_band_v2.json` (method + `position_band_m` for the judge, read
successfully by the integration branch's `rl.perf_acceptance.load_position_band`),
`configs/rl_benchmark/position_band_v2_reference.json` (values, runs, method digest), consumer
`robotic_cutting_sim.rl.position_band`.

**Re-classification of every earlier run** (`artifacts/rl_ready/review_fix1/band_v2_classification_001.json`; v1 =
the superseded P1-0 [min, max]; v2 classes as declared):

| run | env | seed | max over time (m) | v1 | **v2** | final max (m) | v1 (final) | **v2 (final)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1-0 `replay_s512_001` | 0 | 512 | 3.84e-5 | inside | WITHIN | 3.84e-5 | inside | WITHIN |
| P1-0 `band_replay_s512_001` | 0 | 512 | 2.41e-5 | inside | WITHIN | 2.41e-5 | inside | WITHIN |
| P1-0 `band_replay_s512_002` | 0 | 512 | 3.85e-5 | inside | WITHIN | 3.85e-5 | inside | WITHIN |
| P1-0 `band_replay_s512_003` | 0 | 512 | 1.39e-5 | inside | WITHIN | 1.39e-5 | inside | WITHIN |
| P1-0 `p1_2_single_ref_s512_001` | 0 | 512 | 2.39e-5 | inside | WITHIN | 2.39e-5 | inside | WITHIN |
| P1-0 `replay_s15_001` | 0 | 15 | 4.62e-4 | inside | WITHIN | 1.18e-4 | inside | WITHIN |
| P1-0 `band_replay_s15_001` | 0 | 15 | 2.23e-4 | inside | WITHIN | 6.27e-5 | inside | WITHIN |
| P1-0 `band_replay_s15_002` | 0 | 15 | 3.90e-4 | inside | WITHIN | 9.13e-5 | inside | WITHIN |
| P1-0 `band_replay_s15_003` | 0 | 15 | 1.96e-4 | inside | WITHIN | 1.09e-4 | inside | WITHIN |
| P1-2 `m0_s512_001` (pre-fix) | 0 | 512 | 2.24e-5 | inside | WITHIN | 2.24e-5 | inside | WITHIN |
| P1-2 `m0_s512_postfix_001` | 0 | 512 | 3.853e-5 | **above** | WITHIN | 3.853e-5 | above | WITHIN |
| P1-2 `m0_s15_001` (pre-fix) | 0 | 15 | 6.50e-4 | **above** | WITHIN | 1.37e-4 | above | WITHIN |
| P1-2 `m0_s15_postfix_001` | 0 | 15 | 1.87e-4 | **below** | WITHIN | 7.05e-5 | inside | WITHIN |
| P1-3a `episode_solo_512_001` | 0 | 512 | 2.49e-5 | inside | WITHIN | 2.49e-5 | inside | WITHIN |
| P1-3a `episode_solo_15_001` | 0 | 15 | 8.34e-4 | **above** | WITHIN | 1.74e-4 | above | WITHIN |
| P1-3a `episode_512_15_512_15_001` | 0 | 512 | 3.853e-5 | **above** | WITHIN | 3.853e-5 | above | WITHIN |
| P1-3a `episode_512_15_512_15_001` | 1 | 15 | 1.73e-4 | **below** | WITHIN | 7.93e-5 | inside | WITHIN |
| P1-3a `episode_512_15_512_15_001` | 2 | 512 | 2.31e-5 | inside | WITHIN | 2.31e-5 | inside | WITHIN |
| P1-3a `episode_512_15_512_15_001` | 3 | 15 | 8.20e-4 | **above** | WITHIN | 1.80e-4 | above | WITHIN |
| P1-3a `hardened_episode_512_001` | 0 | 512 | 1.58e-5 | inside | WITHIN | 1.58e-5 | inside | WITHIN |
| P1-3a `hardened_episode_15_001` | 0 | 15 | 5.98e-4 | **above** | WITHIN | 1.33e-4 | above | WITHIN |
| P1-3a `episode_512_001` (alias / frozen halves) | 0 | 512 | 3.68e-5 / 2.45e-5 | inside | INVALID_PROCESS | same | inside | INVALID_PROCESS |
| P1-3a `episode_15_001` (alias / frozen halves) | 0 | 15 | 1.56e-3 / 1.99e-2 | above | INVALID_PROCESS | 5.85e-4 / 1.99e-2 | above | INVALID_PROCESS |

Not scored: P1-0 `smoke_replay_s512_001` and `coldcache_replay_s512_001` (20 steps, NOT_APPLICABLE), P1-2
`smoke_m0_001` (NOT_APPLICABLE); M1 replay envs (4) and all 42 M2 episodes recorded no Native position comparison
(NO_POSITION_DATA; their final codes equal Native); the M1 policy-driven envs (4) are not Native replays
(NOT_APPLICABLE). The two INVALID multi-grid processes are not scored (their frozen half, 1.99e-2 m, would be
OUTSIDE both statistics).

**Run-set verdicts (both statistics identical):** P1-0 frozen single 9/9 WITHIN → MET (held-out check of the band);
**P1-2 M0 4/4 WITHIN → MET**; **P1-3a alias_free_v1 8/8 WITHIN → MET**; P1-2 M1/M2 NO_SCORED_RUNS. Corrected
statuses replace "3 of 4 inside" (M0) and "s15 outside in 3 of 4" (P1-3a): the strict v1 tallies were 1/4 (M0), s512
3/4 and s15 0/4 (P1-3a), and v1 is superseded. Corrections are additive in the reports and in the
`corrections_review_fix1` blocks of `p1_0_baseline_001.json`, `p1_2_milestones_001.json` and `p1_3a_001.json`.

### 7.3 F1 — default-path descriptor refresh and step-time census

**Fix** (`batched/material_batch.py`; no kernel source, no graph content changed; the `_particle_contacts_b` pin
passes):
- `advance()` calls `refresh_device_hashgrid()` right before the graph launch **and** before the eager substeps, in
  both particle-contact modes. `frozen_v1` rebuilds `solver.hash` on the host with the identical call of its substep
  (`build(x, 2.4·r, groups=env_of_particle)`); `alias_free_v1` keeps `restore_descriptor`. Label
  `DEVICE_HASHGRID_REFRESH = host_rebuild_before_every_launch_v1`, counted in `receipt()["device_hashgrid"]`.
- Step-time census `device_hashgrid_census_v1`: `wp.HashGrid.__init__`/`build` are wrapped once per process with pure
  bookkeeping (weak registry of grids, weak reference to the grid of the last host `build`). `advance()` checks, with
  no gc scan and no device read, whether another counted grid is alive or was the last one built on the host. Grids
  created before the module import are added by the construction-time gc scan.
- On detection: a per-batch record (`receipt()["device_hashgrid"]["foreign"]`: first and last control step, steps
  flagged, max foreign live grids, foreign last builds), sticky for the batch's life and across `rebuild()`; for a
  guarded batch also the sticky per-env bit `FOREIGN_DEVICE_HASHGRID = 128` (value on the fix branch; **merge note:** 256 on the integration branch since the merge `f5178f5`, because the B2 follow-up's `INPUT_OUT_OF_DOMAIN` already took 128 — raw rows of the fix branch record 128) in `invalid_reason` (so it appears in
  `info["invalid_reason"]`, terminates the envs in training mode and raises `InvalidBatchEnvs` in strict mode). With
  `allow_multiple_device_hashgrids=True` only the record is written.
- Limits: the refresh protects the batch's own graph. Another grid's graph (for example a frozen single
  `LiveCuttingMaterial` built later) now replays the batch's descriptor, so one device grid per process stays the
  rule; the census marks such a process. `vector_env`'s full-reset re-check is unchanged (not this branch's file).

**GPU measurements** (`artifacts/rl_ready/review_fix1/`, `probe_hashgrid_descriptor.py`, `.deps/isaacsim-venv`, Warp
1.17.0, RTX 5090 GPU 0, driver 570.211.01, **SHARED_GPU**: foreign SM mean 50–71 %, 2–3 foreign compute processes,
one process per variant, commits da34452 / a7521d2 (identical library code), clean tree):

- *Cost of the refresh* (`refresh_cost_n{1,16}_{on,off}_00{1,2}`, protocol of the P1-3a solo runs: `frozen_v1`,
  replay steps 3–14 after 2 warm-up steps, env k seed 512 / 15 alternating, 102,732 / 103,898 particles per env,
  50 Hz, 0.1 ms / 200 substeps, no cameras, no recording, warm; `off` = refresh replaced by a no-op, i.e. the pre-fix
  path):

  | N | on: step median (s) | off: step median (s) | refresh host wall per call | refresh GPU time per call (CUDA events) |
  | --- | --- | --- | --- | --- |
  | 1 | 0.4337 / 0.3368 | 0.3353 / 0.3298 | 0.093 ms (max 0.15) | 0.12 ms |
  | 16 | 3.8133 / 3.8324 | 3.8208 / 3.8350 | 0.090–0.094 ms (max 0.12) | 0.25 ms |

  The refresh costs 0.12 ms (N=1) / 0.25 ms (N=16) of GPU time per control step, 0.04 % / 0.007 % of the step; the
  step medians do not resolve it (N=16: on −0.1 % vs off; N=1: `on_002` +1.3 %, `on_001` is a co-tenant outlier with
  the same minimum, 0.303 s vs 0.300–0.304 s). Matches P1-3a's "no measurable cost". Host blocking (review note N3):
  with an idle stream the call returns in 0.09 ms; issued while the batch graph still runs (`f1_*_002`
  `refresh_while_graph_runs`, N=4) it blocks the host for 13.9–14.2 ms (one outlier 326–361 ms under a 98 % SM
  co-tenant) while the graph still had ≥ 560 ms to run: a partial host wait (the build's pageable host-to-device
  copy / scratch allocation), not a device synchronisation. In training mode (no per-step readback) the host can
  therefore wait ~14 ms at the next `advance()` while the GPU stays busy; not measured end to end in training mode.
- *When a replay reads the descriptor* (`replay_race_001.json`): a graph [64 ms spin kernel, A.build, query on A]
  was launched and grid B built on the host 0.05 ms later while the spin kernel still ran: the replay still used
  A's descriptor (8,804,086 candidates = eager). The static is read at **launch**, so a foreign host build during an
  asynchronous replay does not corrupt it; the refresh right before `capture_launch` is sufficient.
- *Batch + foreign grid* (`f1_clean_002`, `f1_foreign_002`, `f1_compare_002.json`; GPU 0, 18:18–18:20, SHARED_GPU,
  foreign SM mean 50–61 %, 3 foreign compute processes; `frozen_v1`, `open_vector_env(4, strict=False)`, 4 copies of
  seed 15 with the recorded Native state of sample 43 (t = 8.6 s), 4 × 103,898 particles, one control step = one
  200-substep graph replay). Two processes: *clean* (one device grid) and *foreign* (after the batch, a
  `HashGrid(64, 64, 32)` — a frozen single material's dims — is built on 20,000 points 5 m away, and built again
  right before the step, so it is the last host build):

  | comparison | candidates per env (mirrored graph) | contacts `stats[4]` per env | non-zero impulse particles | impulse max rel. diff | codes | invalid bit |
  | --- | --- | --- | --- | --- | --- | --- |
  | clean vs clean-equivalent first step of the foreign process (no foreign grid yet) | 8,633,748 = 8,633,748 | 18,807 = 18,807 | 20,748 = 20,748, same set | 5.12e-5 | equal | 0 |
  | clean vs **foreign grid present, refresh on (fixed path)** | **8,633,748 = 8,633,748** | **18,807 = 18,807** | **20,748 = 20,748, same set** | **5.15e-5** | **equal** | **128 (FOREIGN_DEVICE_HASHGRID) in all 4 envs** |
  | clean vs foreign grid present, refresh disabled (pre-fix path) | 8,633,748 → **483,678** | 18,807 → **0** | 20,748 → 20,480 | 1.0 | equal (one step) | 128 |

  With the refresh the batch's contacts match the clean process: identical candidate counts, contact counts and
  non-zero impulse sets, codes equal; the impulse values differ by 5.15e-5 relative, the same as between the two
  clean steps (5.12e-5, GPU float-atomic order over 200 substeps), and `stats[6]` (max overlap) differs in both
  comparisons for the same reason. The compare script's automatic verdict `protected_matches_clean` reads **False**
  only because its impulse threshold (1e-6) was set for a single contact stage, not for a 200-substep step; judged
  against the measured clean-vs-clean spread the result is **PASS**. Without the refresh the hazard is reproduced
  (0 contacts, 483,678 candidates from the foreign descriptor), so the probe is not vacuous. The receipt records
  `FOREIGN_DEVICE_HASHGRID_OBSERVED` (first control step 12, 1 foreign live grid, foreign last host build).

**CPU tests** (`tests/test_batched_hashgrid_guard.py`, monkeypatched predicates, 9 new): refresh precedes the launch
on the graph path (fake `capture_launch`) and the eager path, for `frozen_v1` and `alias_free_v1`; after a foreign
grid is built later, the grid last built on the host at launch time is still the batch's own; the bit, the strict
raise, reset clearing and the sticky record; no gc scan and no array read during the census; opt-out records
without the bit; sticky across `rebuild()`; host grids without the predicates are not censused (refresh still
counted). The full CPU parity suites pass with the refresh in place (§7.6).

### 7.4 F3 / F4 — M2 throughput over matched windows and the right baseline

Recomputed from the raw `steps.npz` (`make_review_fix1_evidence.py`):

| round | N=1 [0:150] | N=8 [0:150] | N=8 / N=1 | N=1 [0:100] | N=16 [0:100] | N=16 / N=1 | reported before |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (matched co-tenant 46–47 %) | 3.475 | 4.850 | **1.40×** | 3.469 | 4.752 | **1.37×** | 1.45× / 1.42× |
| 2 (confounded, not claimed) | 3.473 | 6.199 | 1.79× | 3.469 | 7.038 | 2.03× | 1.85× / 2.10× |

Against the ADR's baseline (sequential frozen single envs): batched N=1 / frozen single = 5.091 / 5.630 = 0.904
(full s512 episodes, co-tenant 24 %), so the gain over the frozen single env is ≈ **1.24–1.26×**, inferred, not
measured in one alternation round, and below the ADR's 1.3× lower estimate (risk #6). Corrected in the milestones
report (status table, §6, §7), `p1_2_milestones_001.json` (`corrections_review_fix1`) and the ADR notes.

### 7.5 F6, F8, F9, F10 and notes

- **F6.** `p1_3a_001.json` gains `corrections_review_fix1`: failure 1 ("no end-to-end gain") WITHDRAWN (from
  INVALID multi-grid runs; valid solo runs give 1.40–1.45×), failure 2 SUPERSEDED (the frozen "control" was the
  INVALID `episode_15_001`; s15 is 4 of 4 outside v1), `not_run` corrected (most rows 1 foreign compute process).
  ADR 0021 carries the withdrawal inline at the "≈1.9× at N=4" sentence.
- **F8.** Milestones report: M1 "PASS (CPU bitwise L1; GPU outcome-level only; L4 BLOCKED)", status table, §8 and the
  evidence correction block.
- **F9.** particles/env filled in the P1-2 rows (M0 102,732 / 103,898; M1 103,315 mean) and the P1-3a episode rows
  (102,732 / 103,898 / 103,315 mean for the mixed N=4), in the reports and the evidence correction blocks.
- **F10.** `tests/test_rl_lab_{scene_usd,ik,kernels,rigid_isolation}.py` call `pytest.importorskip("pxr")` at module
  level; the Torch-venv command now collects and reports them as skipped (NOT_RUN). They still run in `.venv` (40
  passed) and the Isaac Lab venv.
- **N5.** `tests/test_batched_frozen_sources.py` pins `goals.py`, `policy_env.py`, `control.py`, `scene_config.py`,
  `tlmpm.py`…`tlmpm_v4.py` at their f242881^ content (the diff since then is empty).
- **N6.** The construction-time gc scan suppresses the harmless torch `FutureWarning`. The exception-class point
  (`UnsupportedBatchConfiguration` vs `UnsupportedVectorConfiguration`) is in `vector_env.py`, not changed here.
- **N3.** Measured in §7.3 (refresh while the graph runs).
- Not changed here (other owners): F5 and F7 (`rl/vector_env.py`, `run_p1_3a.py`, P1-2b agent); N8, N9 (vector env).

### 7.6 Tests run for these fixes (CPU)

```bash
CUDA_VISIBLE_DEVICES= TMPDIR=/mnt/data/ningan/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_batched_*.py \
    tests/test_wp17_frontier.py tests/test_rl_position_band.py tests/test_rl_lab_ik.py tests/test_rl_lab_kernels.py \
    tests/test_rl_lab_rigid_isolation.py tests/test_rl_lab_scene_usd.py tests/test_repo_naming.py
# 297 passed in 561.02 s   -> PASS (incl. 9 new guard tests, 8 new frozen-source pins, 14 band-v2 tests, 40 Lab tests)
CUDA_VISIBLE_DEVICES= TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src \
    .deps/isaacsim-venv/bin/python -m pytest -q -rs -p no:cacheprovider tests/test_vector_env_api.py tests/test_rl_*.py \
    tests/test_batched_pose_interp.py tests/test_batched_hashgrid_guard.py tests/test_repo_naming.py
# 174 passed, 5 skipped in 117.13 s   -> PASS; the 5 skips are NOT_RUN (test_rl_lab_env needs Isaac Lab; the four
# test_rl_lab_* files need pxr) - the F10 command now completes instead of stopping at collection
```

The first Torch-venv run of the final tree had 1 failure (`test_reference_file_matches_the_declared_method`: SciPy
1.17 in the Torch venv vs 1.18 in `.venv` differ in the last bits of the re-derived band); fixed by a relative
tolerance of 1e-6 (511545f), rerun above. The earlier `.venv` run before the band reference existed: 254 passed,
1 skipped (NOT_RUN: reference not derived yet).

### 7.7 Failed or preserved runs

- The first rerun waiter of `f1_chain2.sh` died on a shell syntax error in the chain script before running anything;
  fixed and requeued (blocking waiter on `gpu0.lock`).
- `f1_clean_001`, `f1_foreign_001`: FAIL, the probe captured its count graph on the Torch stream ("Warp error:
  unknown stream"); fixed in a7521d2, preserved, rerun as `_002`.
- A first F1 chain start on GPU 0 at 12:12 skipped every job (4.9 GB free < 7 GB): no process ran.
- The band-chain and first F1-chain logs print `rc` of the `date` substitution, not of the job; job status was
  checked from each `result.json` (all band runs COMPLETE).

### 7.8 Per-finding status

| Finding | Status | Fix | Commits |
| --- | --- | --- | --- |
| F1 MAJOR default-path grid hazard | **FIXED** (CPU tests PASS; GPU cost and launch-time semantics measured; batch + foreign-grid probe PASS: contacts equal to a clean process, hazard reproduced without the refresh) | host rebuild of the own grid before every launch + step-time census, sticky bit 128, receipt | 5503122, a7521d2, 1f7d39f |
| F2 MAJOR position band | **FIXED** (band v2 pre-declared, derived from 20 runs; every earlier run re-classified; statuses corrected, v1 kept as superseded) | `position_band_v2.json` (+ judge view), `position_band_v2_reference.json`, `rl.position_band` | 1b3eaa1, 03394d2, 4d39747, 5beb8df, da34452, 511545f, 0a26b5f |
| F3 MINOR mismatched windows | **FIXED** (1.40× / 1.37×) | report, ADR, evidence corrections | 0a26b5f, 1f7d39f |
| F4 MINOR wrong baseline | **FIXED** (≈1.24–1.26× over the frozen single env, inferred, below the ADR estimate) | report, ADR, evidence corrections | 0a26b5f, 1f7d39f |
| F5 MINOR autoreset + evaluate | NOT_MINE (P1-2b agent, `rl/vector_env.py`) | — | — |
| F6 MINOR stale text | **FIXED** | `p1_3a_001.json` corrections block; ADR inline withdrawal | 0a26b5f, 1f7d39f |
| F7 MINOR multi-variant commands | NOT_MINE (P1-2b agent, `run_p1_3a.py`) | — | — |
| F8 MINOR M1 bit-identity CPU-only | **RECORDED** (note) | milestones report status, §8; evidence correction | 0a26b5f, 1f7d39f |
| F9 MINOR particles/env | **FIXED** | reports and evidence correction blocks | 0a26b5f, 1f7d39f |
| F10 MINOR Torch-venv collection | **FIXED** | module-level `importorskip("pxr")` | c35375e |
| N3 refresh sync | measured: 0.09 ms host on an idle stream; a 14 ms partial host wait when issued behind a running graph (no device sync) | — | 1f7d39f |
| N5 frozen pins | **FIXED** | 8 more pins | c35375e |
| N6 FutureWarning | **FIXED** (warning); exception class NOT_MINE (vector env) | `live_device_hashgrids` | 5503122 |
| N1, N2, N4, N7, N8, N9, N10 | not acted on (no cheap change in these files; N8/N9 belong to the vector env) | — | — |

Still open: EXCLUSIVE_GPU rows (every number here is SHARED_GPU), L4 RUN_TO_RUN (BLOCKED), a measured
`single_env_headless` row inside an alternation round (F4 is inferred), and Native confirmation of the policy-driven
episodes.
