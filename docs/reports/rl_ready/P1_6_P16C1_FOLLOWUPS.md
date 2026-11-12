# P1-6 cycle p16c1 (`alias_free_v2`): follow-ups to the PR #14 review (2026-11-12)

**Scope.** Findings 1, 3 and 4 of the xhigh review of PR #14
(`docs/reports/rl_ready/P1_6_P16C1_PR14_REVIEW.md` on the `claude/rl-ready-rubber-physics` branch). Findings 2,
5, 6 and 7 are out of scope.

**Base and branch.** Base `main` at `9ee1e9d`, the PR #14 merge. Branch `claude/p16c1-followups`.

**What stays unchanged:**
- `frozen_v1`, the default, and `alias_free_v1`;
- every frozen module, threshold, record and acceptance file;
- `scripts/rl_ready/run_p1_3a.py`.

Every change is inside the opt-in `alias_free_v2` path of `src/robotic_cutting_sim/batched/particle_contacts.py`,
plus its test file.

## 1. What changed

### Finding 1: the CPU tests now exercise the query shrink under contact

**Instrumentation.** No output array shows whether the shrink is used: the shrink is exact, so results are bitwise
equal either way. A test therefore cannot tell from outputs whether the shrink ran. To make it observable:
- `_particle_contacts_env_major_b` gets a per-env counter, `shrink_engaged[e]`.
- The counter is written by the thread of the env's first particle, at the point where the kernel selects
  `reach_slow`. There is one writer per env and no atomics.
- It is a diagnostic: nothing in the solver reads it, it is cumulative since construction, and partial reset does
  not clear it.
- It is reported as `shrink_engaged_substeps` in the v2 receipt only. The v1 receipt keys are unchanged.
- It lives on the contact object, not on the solver, so the reset-coverage classification does not change.

**New test: `test_shrink_engaged_under_slow_contacts_with_fast_slow_transitions`.**

*Scene:*
- Three small, very soft lattice blocks (Young's modulus 140 to 300 Pa), lifted 6 mm off the table plane and the
  fixture, with the tool parked 0.3 m away.
- The lattice is compressed to 0.88, so the spacing is 1.76 mm against 2r = 1.8 mm.
- Codes alternate per x layer, which makes every x-neighbour pair a conflicting contact pair.
- An inward velocity field keeps the layers pressed together.
- 240, 196 and 225 particles; 24 substeps.
- Without the boost there are 5040, 3777 and 4211 non-zero particle contacts. Every env has contacts in every
  substep (at least 210, 65 and 150 per substep).

*What it asserts:*
- **Bitwise:** every env of the v2 batch equals its frozen single after every substep (`compare_env`, every array).
- **Envs 0 and 2:** always below V = 0.25 m/s. They have non-zero contacts in all 24 substeps, and the kernel's
  counter rises by exactly 1 in every substep.
- **Env 1, fast to slow:**
  - Slow in substeps 0 to 5.
  - Before substep 6, 30 particles are boosted to 20 m/s. That is above V and below the certified bound of about
    150 m/s.
  - Before substep 9, its x and v are restored to the pre-boost values. This is a teleport, applied identically to
    the single and to every batch.
  - Per substep, the counter pattern must be exactly 1 1 1 1 1 1 0 0 0 1 … 1: shrunken reach before, frozen reach
    while fast, shrunken reach again after the transition. Env 1 must also have contacts after the transition.
- **Control batch:** its speed check never fires (`shrink_speed_m_s` set to 1e3 after construction, so the shrunken
  reach is kept). It must differ in env 1 during the boost, and envs 0 and 2 must stay bitwise. This shows the scene
  can detect a shrunken reach applied to a fast env.

**Mutation check.** Each mutation was applied locally to `particle_contacts.py`, run, and then removed. No mutation
is committed. Script: `mutate.py`; logs: `mutations.log`, `mutations_reasons.log`.

| Mutation | New test | Other v2 tests failing |
| --- | --- | --- |
| A: kernel reads `fast_flag[p]` instead of `fast_flag[e]` (out-of-bounds read on CPU) | FAIL: env 1 not bitwise | necessity test, r/dt test |
| B: shrink skipped after the first launch | FAIL: counter for envs 0/2 | necessity test |
| C: guard writes the flag by particle id (`fast_flag[p % N]`) | FAIL: counter for envs 0/2 | none |
| D: fast flag never cleared | FAIL: env 1 never re-engages after the boost | none |
| E: flag cleared before the contact kernel | FAIL | both block-scene tests, necessity test |

Without the counter, B, C and D change no output bit, so only the counter catches them.

**Vacuous assertions.**
- The two `fast_flag.numpy().tolist() == [0, 0, 0]` assertions are replaced; they could never fail, because the stage
  clears the flag.
- Block scenes, replacement: the shrink is engaged in at least 1 and fewer than 120 env-substeps, and the no-shrink
  variant reports `[0, 0, 0]`.
- Necessity test, replacement: the checked batch never uses the shrunken reach for env 1, and the unchecked batch
  uses it in all 12 substeps.

**Docstrings.** The module docstring and the block-scene docstring no longer claim the shrink is active "all envs
slow". They now say it is engaged only in the first substep there, and point to the new test.

### Finding 4: the shrunken reach and the launch's r and dt

**The problem.** `reach_slow` and the guard's certified bound are derived from the construction `solver.radius` and
`solver.dt`. A larger launch `dt` or `r` would make the shrunken reach unsafe. The derived cell width and alias
dims also depend on the construction radius, so deriving only `reach_slow` per launch would not make the stage
safe.

**The fix: fail closed.**
- `AliasFreeContacts` records `construction_radius` and `construction_dt`.
- `_launch_v2` raises `UnsupportedBatchConfiguration("UNSUPPORTED_PARTICLE_CONTACT_MODE", "... differs from the
  construction ...")` when the launch values differ, before any launch.
- Every current caller passes exactly `(solver.radius, solver.dt)`: `BatchedTLMPMv7_2.step` and
  `run_p1_3a.contact_stage`.

**New test: `test_v2_launch_refuses_radius_or_dt_other_than_construction`.**
- Launches with 2·dt, 0.5·dt and 1.5·r are refused.
- No impulse is written and the counter is not touched.
- A launch with the construction values then runs.

**Not changed:** `alias_free_v1` has the same latent assumption for its certified bound. Its path was kept untouched
on purpose.

### Finding 3: the harness isolation audit — the harness is frozen, so it is not modified

**Why it counts as frozen:**
- `scripts/rl_ready/run_p1_3a.py` is in the evaluator file set of `acceptance_v2`: `"scripts/rl_ready/*"` in
  `configs/rl_benchmark/acceptance_v2.json`, which is sha-pinned by `tests/test_acceptance_frozen.py`.
- `tests/test_perf_acceptance.py::test_evaluator_files_and_frozen_paths_cover_the_evaluator` explicitly requires
  `scripts/rl_ready/run_p1_3a.py` to be covered by `evaluator_files`.
- The acceptance harness `run_perf_acceptance.py` imports it (G4 samestate helpers, and the G6/G7
  `L2MonitorActiveGrid` audit).
- D-14 (`docs/RL_READY_PLAN.md`) keeps the evaluator file set unchanged and says any change after the freeze is
  `acceptance_v3`.
- The file itself has no sha pin. Changing it would still change the evaluator of every later acceptance under the
  name `acceptance_v2`. It is not modified.

**Notes for a future `acceptance_v3` / harness version:**
1. **The audit checks the wrong grid.** `L2MonitorActiveGrid.audit` builds and audits `pc.grid`, the grouped
   alias-free `wp.HashGrid`. Under `alias_free_v2` the contact kernel queries `EnvMajorGrid` instead. A v2 audit
   should walk `env_grid.keys/starts/ends/ids` with the kernel's own cell walk and count foreign-env candidates. This
   needs a key-validated lookup (`keys[starts[k]] == k`) and the base `e*C`, and could use either the frozen reach or
   the shrunken one gated by the fast flag. The audit should also report `shrink_engaged_substeps` from the receipt.
2. **The audit's grid build** happens once per audited step on the harness side. It is harmless for v2, which never
   queries that grid. It would become unnecessary once (1) lands.
3. **The host-side `restore_descriptor` rebuild** (`particle_contacts.py`, called by
   `BatchedCuttingMaterial.refresh_device_hashgrid` before every launch) is dead work for v2: about one grouped build
   per control step. It is kept for now. Dropping it changes what the single-device-HashGrid guard and the Warp 1.17
   descriptor argument rely on (the review says "once the guard semantics are re-argued"). That argument is not
   made here.

**Mitigation until then (unchanged from the review):**
- isolation by construction (the kernel reads only buckets `e*C … e*C + C − 1`);
- the per-env bitwise CPU tests, including the new one with a fast env next to slow envs;
- the review's CUDA probe, with 0 foreign candidates.

## 2. Byte-identical results for current callers

**Method.**
- A pytest plugin (`hashplug.py`) wraps `BatchedTLMPMv7_2.step`.
- After every call it hashes, with SHA-256, every Warp array attribute of the solver, in name order.
- It records the hash sequence per test node id.
- Both runs used the same command on the full v1 and v2 particle-contact test files, on CPU
  (`CUDA_VISIBLE_DEVICES=''`), Warp 1.17.0:

```
HASHPLUG_OUT=<out>.jsonl PYTHONPATH=<plugin dir> .venv/bin/python -m pytest -p no:cacheprovider -p hashplug \
    tests/test_batched_particle_contacts_v2.py tests/test_batched_particle_contacts.py -q
```

**Runs:**
- **Before:** base `9ee1e9d`. 26 passed in 281 s.
- **After:** the working tree with the code and tests of this branch's first two commits. 28 passed in 290 s. The
  2 new tests are the difference.

**Result (`compare.py`):**
- All 26 existing tests have identical per-step hash sequences, over 5891 hashed solver steps. 2585 of them are in
  the v2 file: block scenes, necessity test, full-resolution TRIM, N=3 TRIM sequence, partial reset at material
  level.
- 0 tests differ.
- This covers `frozen_v1`, `alias_free_v1` and `alias_free_v2` with and without the shrink.

**Not covered:**
- The new counter array lives on the contact object, so it is not in these hashes. It is a diagnostic only, and no
  physics array reads it.
- GPU parity and GPU timing of the extra per-thread branch (one load of `env_of_particle[p-1]` for slow envs) were
  NOT_RUN: GPU0 and GPU1 are busy with rubber work, and CPU was preferred.

## 3. Tests

All CPU runs used `CUDA_VISIBLE_DEVICES=''`.

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest tests/test_batched_particle_contacts_v2.py tests/test_batched_particle_contacts.py` (with the hash plugin) | **28 passed**, 290 s |
| `.venv/bin/python -m pytest tests/test_batched_hashgrid_guard.py tests/test_repo_naming.py tests/test_perf_acceptance.py tests/test_batched_frozen_sources.py tests/test_acceptance_frozen.py -rs` | **127 passed, 1 skipped** (torch not in `.venv`; run below), 15 s |
| `.venv/bin/python -m pytest` on every other `tests/test_batched_*.py` plus `tests/test_vector_env_*.py` | **130 passed, 5 skipped** (torch, NOT_RUN in `.venv`; run below), 403 s |
| `PYTHONPATH=<pytest overlay py312>:src .deps/isaaclab-venv/bin/python -m pytest` on `test_vector_env_{api,consolidation,contract,p12b,gpu_stream}.py` plus the `acceptance_v2` `torch_tests` (`test_rl_policies_batched`, `test_rl_replay`, `test_rl_actions`, `test_rl_observations`, `test_rl_reward_packed`, `test_perf_acceptance`) | **186 passed, 1 skipped** (`test_vector_env_gpu_stream.py`: CUDA hidden, NOT_RUN), 355 s |

**Evidence location.** Logs, hash files, the plugin, the comparison and the mutation script are in
`artifacts/rl_ready/p1_6_p16c1_followups/` (on /mnt/data).

## 4. Still open

- Finding 2: O(N·C) bucket bounds and the int32 cap N ≤ 2495.
- Finding 3: the v2 harness audit and the dead host rebuild (notes in §1, for `acceptance_v3`).
- Finding 6: the label inconsistencies in the evaluator stack.
- Finding 7: wasted `fast_flag` stores when the shrink is off.
- GPU confirmation of the counter: parity and cost.
