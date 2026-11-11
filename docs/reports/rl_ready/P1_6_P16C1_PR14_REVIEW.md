# P1-6 cycle p16c1 (PR #14, `alias_free_v2`) — xhigh code and acceptance review (2026-11-12)

**Commits reviewed:** PR #14 `perf/p1-6-p16c1` at head `8d6a2b3b`. Base `main` is `bfe6d5e`.
- `51034000` and `7053c965` hold the code. The external agentic optimizer wrote them, and the frozen evaluator accepted
  `7053c965`.
- `8d6a2b3b` adds only `docs/evidence/rl_ready/p1_6_p16c1_acceptance.json` and `docs/reports/rl_ready/P1_6_p16c1.md`
  (`git diff --stat 7053c96 8d6a2b3b`).

**Where I worked:**
- The review was read-only, in the detached worktree `/home/ningan/RoboticCuttingSim-rl-pr14`. It has `.venv` and `.deps`
  symlinks like the other rl worktrees, plus two read-only input links: `artifacts/wp17` and
  `artifacts/workcell_asset_cache`.
- Every `file:line` below refers to `8d6a2b3b`.
- I pushed nothing to the PR branch and posted no GitHub comment.

**Probes and logs:** all are preserved in
`/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/p1_6_p16c1_pr14_review/` (§6).

**Scope.** The full diff `bfe6d5e..8d6a2b3b` touches four files:
- `src/robotic_cutting_sim/batched/particle_contacts.py`: +262 / −3;
- `tests/test_batched_particle_contacts_v2.py`: new;
- the two docs files.

The following are unchanged (empty `git diff`):
- the `frozen_paths` of `acceptance_v2.json`;
- `configs/`, `scripts/`, every other test;
- `pyproject.toml`; there is no `conftest.py`.

The authorship trailer of the two optimizer commits is known and accepted, so it is not a finding.

## 1. Verdict

**PASS_WITH_LIMITATIONS. Merge GO: GitHub CI on `8d6a2b3b` is green (§7); conditions in §8.**

- **No BLOCKER and no MAJOR finding.**
- **Default and v1 unchanged:**
  - `frozen_v1` (the default) and `alias_free_v1` keep their code paths and outputs.
  - The diff removes only three lines: the import, the `MODES` tuple and the receipt line (§3.1).
- **Speed-bound query shrink is sound:**
  - The search radius does not drop contacts silently.
  - An env with any particle faster than V = 0.25 m/s uses the frozen 2.4 r reach in the same substep. There is no state
    between substeps.
  - Above the certified bound (150.5 m/s on TRIM), the unchanged v1 guard applies: fallback, sticky bit 64.
  - The float32 slack holds at any coordinate size (§3.2).
- **Regrouping matches Warp's grouped query exactly:**
  - Env-major buckets give the same candidate sequence as Warp's grouped `HashGrid` query. This is shown on CPU by the
    PR's test, and on CUDA by my own probe: 310,227 particles over 3 envs, dims 189×69×66, 4 position sets.
  - Count and order-sensitive hash are equal for every particle.
  - No candidate comes from a foreign env (§3.3, §3.4).
- **Acceptance evidence is consistent and reproduces from the rows.**
  - The ratio medians recompute: N=1 1.168, N=16 1.279.
  - SHARED_GPU is disclosed.
  - The real workload exercised the shrink: the WP-17 Native particle speeds peak at 0.061 m/s (s512) and 0.122 m/s
    (s15), below V (§4).
- **Limitations (§5):**
  - The CPU tests show the shrink under non-zero particle contact far less than they claim (finding 1).
  - The v2 memory is O(N·C) and lowers the int32 N limit to 2495 (finding 2). The PR body does not list this.
  - The harness isolation audit inspects a grid v2 no longer queries (finding 3).

## 2. Per-item verdicts

| Item | Verdict | Basis |
| --- | --- | --- |
| Default / `frozen_v1` / `alias_free_v1` byte-for-byte | **PASS** | §3.1. `tlmpm_v7_2_batch.py` and `material_batch.py` are unchanged. The `frozen_v1` receipt is hard-coded (`tlmpm_v7_2_batch.py:775-778`). v1 skips the new dispatch: `env_grid is None` (`particle_contacts.py:697`), asserted in the test at `:263`. The v1 receipt keeps its keys and their order. `FROZEN_BATCH_KERNEL_SHA256` pin: PASS. G8 (default plan): PASS. |
| Search radius (speed-bound shrink) | **PASS** | §3.2. The check runs per env and per substep, on the same `x` and `v` the kernel reads (`:719-731`). An env above V falls back to the frozen reach; nothing is dropped. Above the certified bound, guard bit 64 applies (unchanged). Reset, teleport and the first step after a partial reset are covered because the stage keeps no state between substeps. |
| Regrouping and ordering, parity | **PASS** | §3.3. The radix sort is stable over ascending ids. Each bound is written exactly once. The key check makes stale bounds unusable. Candidate sequences equal Warp's grouped query on CPU (test) and CUDA (my probe). Accepted episodes: codes and labels equal Native (G6, G7). |
| Multi-env isolation, partial reset | **PASS** | §3.4. The key carries the env prefix `e*C`. My GPU probe found 0 foreign candidates. The per-env shrink flag is isolated, shown at substep 1 of the necessity test. Partial reset: CPU v2 equals `frozen_v1` (no particle contacts in that scene, finding 1). GPU material probe, partial reset under the CUDA graph: v2 final codes equal v1 per env (§7). |
| HashGrid guard, descriptor hazard, device pinning | **PASS_WITH_LIMITATIONS** | §3.5. v2 adds no `wp.HashGrid` and puts no `HashGrid.build` in its graph. The guard tests pass. On CUDA a second v2 material is refused with `UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS` (§7). Limitation: finding 3, an audit of a grid v2 does not query, and a dead host rebuild. |
| Memory | **PASS_WITH_LIMITATIONS** | §3.6. Fixed allocation at construction; no growth per step. The cost is O(N·C), about 8.1 MiB/env allocated and 4.3 MiB/env measured. int32 limit N ≤ 2495 (finding 2). |
| New tests test the claims | **PASS_WITH_LIMITATIONS** | §3.7. The regrouping tests are strong. The CPU evidence for the shrink under contact is 2 env-substeps per block scene. The TRIM tests have zero particle contacts. Two assertions are vacuous (finding 1). |
| PR body against evidence | **PASS_WITH_LIMITATIONS** | §4. The gate table, ratios, rows and SHARED_GPU label are correct. Limitations specific to the PR are missing: findings 1 and 2, plus that the speed-up depends on slow particles. The wording "samestate reruns bitwise" holds for dev only (finding 5). |
| PR checklist (hooks, tests, pytest config, opt-in, no harness or record reads, parity and rollback) | **PASS** | §4.1 |
| Tests run here (CPU) | **PASS** | §6: 26 + 119 + 130 + 179 / 91 passed. The 3 initial torch failures were missing `artifacts/` links in my worktree; all 91 pass on rerun with the links. |
| GitHub CI | **PASS** | §7. Both runs succeeded with 0 failed: push run 43102214049 (2481 passed, 58 skipped, 6 xfailed) and pull_request run 43102223144 (2451 passed, 88 skipped, 6 xfailed). |

## 3. Correctness of the diff

### 3.1 Default and existing opt-ins

**Default path:**
- `resolve()` returns `ParticleContactConfig()` for the default. It gains two fields with defaults
  (`env_major_grid=False`, `shrink_speed_m_s=None`, `:106-107`) that the default path never reads.
- `BatchedTLMPMv7_2` builds no `AliasFreeContacts` for `is_default` and keeps its inline `_particle_contacts_b` launch
  (`tlmpm_v7_2_batch.py:671-677`, unchanged).

**`alias_free_v1`:**
- In `__init__` (`:668-679`) the v2 block runs only if `config.env_major_grid`. That flag can be True only through
  `_resolve_v2`.
- `launch()` (`:697`) dispatches to `_launch_v2` only when `env_grid` is set. The v1 lines that follow are unchanged.
- `receipt()` (`:744-747`) filters out the two new keys for v1. The remaining keys keep the old order; the test checks
  this at `:262`.

**Kernels:** no v1 kernel or `FallbackGrid` line changed. The only removed lines are:
- the `dataclasses` import;
- `MODES`;
- the receipt dict line.

Adding kernels to the module changes its Warp hash, so v1 compiles once more. This is not a behaviour change.

**Receipt hash:** `source_sha256()` (`tlmpm_v7_2_batch.py:536-540`) hashes `particle_contacts.py`, so every mode's
receipt reports a new value. This is a fingerprint, not a gate; nothing compares it with a stored value.

### 3.2 Speed-bound query shrink

**Certificate.** `v5._contact_impulse` (frozen, `tlmpm_v5.py:155-165`):
- Gives `jn = k m / dt · max(−(gap + dt·min(vn, 0)), 0)`.
- With `jn = 0`, `jt = −min(m·|t|, μ·0)·t/|t|` is ±0. C1 then skips the atomics and the stats (`length_sq(ji) > 0` is
  false).

So a candidate contributes only if `d < 2r + dt·|v_rel|`. Both particles are in env e, whose speeds are all ≤ V, so
`|v_rel| ≤ 2V`.

**Float32.** The worry is that `fl(xp − R)` could round above a contacting `xq`. That cannot happen:
- `xq` is itself a float32 value.
- `xp − R < xq` holds exactly, because `xp − xq ≤ d < 2r + 2dtV < R`.
- Rounding is monotone with fixed points, so `fl(xp − R) ≤ xq`.
- Multiplication by the same `inv` followed by `floor` is also monotone.

The 1e-5 relative and 1e-7 m absolute slack (`:85-86`, `shrunken_reach :194-199`) therefore only has to cover the
distance and speed rounding, about 1e-10 m. My float32 emulation of the kernel bound confirms this
(`probes/slack_domain.py`):
- set-up: p and q on one axis, both speeds exactly V and approaching (`vn = −2V`), q the farthest point with `jn > 0`;
- result: 0 misses in 1.75 M pairs, for |x| from 0 to 8 m.

The TRIM solver-local coordinates are |x| ≤ 0.08 m.

**Sub-box.** `reach_slow = min(frozen float32 2.4 r, …)`, and the frozen value is computed exactly as the kernel's
`2.4*radius`. So `x0_slow ≥ x0` and `x1_slow ≤ x1`. The clamp `x0 + D − 1` never engages for a box of about 3 cells
when D ≥ 64.

**When V is exceeded.**
- `_speed_guard_v2_b` (`:468-483`) sets `fast_flag[e]` when any particle of env e has `!(|v|² ≤ V²)` (NaN included) or
  a non-finite `x`.
- The kernel then uses `2.4 r` (`:528-531`) in the same substep.
- Contacts are not missed, so no new flag is needed.
- Above the certified bound (150.5 m/s for the TRIM pack, `certified_speed_bound`), the unchanged path applies:
  `guard_flag` → frozen fallback in `wp.capture_if` → `contact_guard_violations` → sticky
  `INVALID_CONTACT_SPEED_GUARD` = 64 (`material_batch.py:54, 942-944`).

**Reset, teleport, first step after a partial reset.**
- `fast_flag` is recomputed from the current `v` before the kernel and cleared at the end of every stage (`:731`).
- The buckets are rebuilt from the current `x` every substep (`:727`), and stale bounds are rejected by the key check.
- Earlier kernels in the substep (rigid, obstacle and fixture contacts) write `impulse`, not `v` or `x`. The guard and
  the kernel therefore see the same state.
- A pose jump changes `x` or `v` only through later substeps, which are re-checked.
- Partial reset clears `contact_guard_flag` and `violations` (`material_batch.py:1110-1112`). The v2 arrays need no
  reset.

**Residual assumptions:**
- Finite, positive `mass`, stiffness and friction. A NaN parameter would make a dropped candidate NaN in the frozen
  path. v1's C1/C2 certificates make the same assumption, accepted in Phase 1.
- `reach_slow` is fixed from `s.radius`/`s.dt` at construction (finding 4).

### 3.3 Regrouping and ordering

**Determinism.**
- `EnvMajorGrid.build` (`:592-597`) writes `keys[p] = env*C + cell` with Warp's `hash_grid_index` arithmetic and
  `values[p] = p`.
- It sorts with the stable `wp.utils.radix_sort_pairs` over `end_bit = bitlen(N·C − 1)`: 24 bits at N=16 with
  C = 860,706.
- `_env_major_ranges_b` (`:491-503`) writes `starts[k]` and `ends[k]` for each present key from exactly one thread.

The result is deterministic, and ids within a bucket are ascending.

**Memset-free validation is sound.**
- A present key gets fresh bounds.
- An absent key k can never satisfy `keys[starts[k]] == k`, because no sorted position holds k.
- `starts` starts at zeros, so `first < n` always holds (`:543-545`).

**Sequence parity.**
- The PR's test (`:76-115`) compares id sequences with a grouped `wp.HashGrid` on CPU, for dims (5,5,5), (64,64,32)
  and (7,3,4) and a rebuild on moved points. It passes here.
- My CUDA probe `probes/gpu_seq_probe.py` compares, per particle, the candidate count and an order-sensitive hash.
  Workload: TRIM seeds 512 / 15 / 7, 310,227 particles, alias-free dims (189, 69, 66), positions at reference plus
  noise of 0, 0.2, 0.6 and 3 mm. Result for all 4 sets:
  - The env-major walk and Warp's grouped `hash_grid_query` on CUDA are equal.
  - Mean candidates per particle fall from 32.6 to 25.5 with the shrink (−22 %; the PR expects 21/27 cells).
  - The shrunken walk contains every same-env pair with `|xp − xq| < 2r + 2dtV`. The brute-force count matches
    exactly: 682,836 / 966,116 / 808,626 pairs.

**Physics parity on GPU.** The candidate's G4 samestate on the Native states, at 9 samples × 2 seeds:
- identical non-zero impulse sets and `stats[4]`;
- maximum relative impulse difference 4.9e-8 (float atomics; tolerance 1e-5).

G6 and G7 episodes: final codes and labelled counts equal Native for every env (s512: 37,958 labelled, 332 positive).
Contact totals are within 0.39 % of the baseline.

### 3.4 Multi-env isolation and partial reset

**Isolation:**
- The kernel reads only buckets `base = e*C … e*C + C − 1` (`:538-546`).
- The GPU probe found 0 foreign candidates.
- `fast_flag` and `guard_flag` are per env.
- `test_shrink_speed_check_is_necessary_and_exact` (`:143-159`) runs env 1 at 20 m/s. It shows that env 1's flag
  leaves envs 0 and 2 bitwise, and that env 1 differs without the check.

**Partial reset:** `test_partial_reset_v2_equals_default_mode` (`:218-239`) is bitwise equal to `frozen_v1` before and
after `reset([1])`. That scene has no particle contacts (finding 1). The v2 stage is stateless between substeps (§3.2).

### 3.5 HashGrid guard, descriptor hazard, device pinning

**Guard:**
- v2 creates no `wp.HashGrid`. It keeps v1's single alias-free grid as `s.hash` / `contacts.grid` (`:652-660`), so the
  census and `check_single_device_hashgrid` see exactly what they see for v1.
- `tests/test_batched_hashgrid_guard.py` passes here.

**Descriptor hazard:**
- The v2 graph contains no `HashGrid.build`, so the Warp 1.17 static-descriptor hazard cannot affect its own contact
  stage. The fallback's `FallbackGrid` is built from kernels, as in v1.
- `refresh_device_hashgrid` → `restore_descriptor` (`:683-687`) still rebuilds the unused grid on the host before every
  launch. That is dead work, about one grouped build per control step, under 0.1 % (finding 3).

**Device pinning:**
- `EnvMajorGrid` launches pass `device=self.device` (`:594-603`).
- The radix sort runs under `ScopedDevice`.
- `_speed_guard_v2_b` and `_clear_flags_b` (`:725, :731`) are unpinned, like v1's `_speed_guard_b` and
  `_guard_consume_b`. They run inside the solver step's `wp.ScopedDevice(self.device)` (`tlmpm_v7_2_batch.py:640`) and
  inside the harness's `contact_stage` (`ScopedDevice`).

### 3.6 Memory

**What is allocated.** All allocations happen once, in `EnvMajorGrid.__init__` (`:572-586`):
- `keys` and `ids`: 2n int32 each;
- `starts` and `ends`: N·C int32 each;
- `fast_flag`: N.

The first `build` runs outside any capture (`:678-679`) and reserves the sort's temporary buffer. Nothing is allocated
per step. With C = 860,706 (TRIM s512/s15, 20 mm margin), each env costs:
- 6.57 MiB for the bounds;
- 1.57 MiB for keys and ids (n = 102,732);
- 8.1 MiB allocated in total.

**Measured** (candidate-reported, nvidia-smi): +74 MiB at N=16, i.e. +4.3 MiB/env over N=1. This is within
`C ≤ 1.10 × B + 256 MiB`.

**int32 guard.** `N·C < 2³¹` (`:576-577`) caps v2 at N ≤ 2495 for this workpiece. v1's `FallbackGrid` allowed
N ≤ 16,383. Above the cap, v2 refuses at construction with `UNSUPPORTED_INT32_OVERFLOW`, so it fails closed. Finding 2.

### 3.7 What the new tests test

**Strong:**
- `test_env_major_grid_reproduces_warp_grouped_query_sequence`: exact sequences, rebuild with stale bounds, and the
  ordered-subsequence property.
- `test_block_scenes_v2_bitwise`, for the regrouping: 40 substeps with contacts on the frozen reach.
- The necessity test: the speed check is required and the per-env flag is isolated.

**Weaker than stated (measured with `probes/shrink_active.py`, `slow_contact.py`, `trim_contacts.py`).** Finding 1.
- *Block scenes.* The shrink is engaged in only 3 of 120 env-substeps, all in substep 1. Particle speeds then reach
  5–11 m/s (table plane at z = 1.2 mm under the block, compressed lattice).
  - Non-zero contacts on the shrunken reach occur in 2 env-substeps, 114 contacts.
  - Gentler compressions (0.88–0.92) behave the same: 3–4 engaged env-substeps, all bitwise; 756 contacts in total over
    six scenes.
- *TRIM tests.* `test_full_resolution_trim_substep_bitwise_v2`, `test_n3_heterogeneous_trim_sequence_bitwise_v2` and
  `test_partial_reset_v2_equals_default_mode` have **zero** particle–particle contacts in all 1, 800 and 800 stages.
  The shrink is engaged throughout, but no contribution is computed.
- *Vacuous assertions.* `fast_flag.numpy().tolist() == [0, 0, 0]` (`:140`, `:159`) can never fail, because the stage
  clears the flag (`:731`).
- *Inaccurate docstring.* The block-scene docstring (`:123-124`, "the shrink is active (all envs slow)") is wrong for
  39 of 40 substeps.

**Outcome.** The shrink's claim does hold:
- by the argument in §3.2;
- on the real workload (G4, G6, G7 with Native speeds ≤ 0.122 m/s);
- on CUDA geometry (my probe).

The CPU suite alone, however, would miss an error that appears only after substep 1, or only in slow envs with
contacts.

## 4. Acceptance evidence and PR body

**Recomputed from the rows:**
- N=1 pairs: 6.0414/5.1865 = 1.1648, 6.0634/5.1600 = 1.1751, 6.0362/5.1673 = 1.1682 → median **1.1682** (floor 0.97).
- N=16 pairs: 9.5433/7.4615 = 1.2790, 9.5448/7.4594 = 1.2796, 8.9432/7.8282 = 1.1424 → median **1.2790**, 3 of 3 above 1
  (threshold 1.05).
- Even the lowest pair, 1.142, clears 1.05 under SHARED_GPU noise.
- The dev exclusive figures agree: N=8 1.216, N=1 1.117.
- Work gates: same labelled counts. VRAM: 936 / 926 and 2276 / 2202 MiB, both within limits. Reset: within the stricter
  exclusive rule.
- The gate table in the PR body and in `P1_6_p16c1.md` matches `p1_6_p16c1_acceptance.json`.

**SHARED_GPU** is disclosed: GPU index 1, up to 5 foreign processes, utilisation p50 99 %. D-14 asks for EXCLUSIVE_GPU
"where possible". It was not, but the margin makes the result robust.

**Label inconsistency (frozen evaluator stack, not this PR).**
- Every row carries the in-process `gpu_sharing: EXCLUSIVE_GPU` next to `caller_gpu_label: SHARED_GPU`.
- Rows carry `nvidia_smi_index: 0` for UUID `GPU-a278df44…`, which is physical index 1 on this host.
- The reset gate therefore chose the stricter "exclusive" regime. Nothing was inflated. This is for `acceptance_v3`
  (finding 6).

**The real workload exercises the shrink.** WP-17 Native recorded speeds (`probes/native_speeds.py`, 121 samples per
seed, all finite):
- s512 maximum 0.061 m/s;
- s15 maximum 0.122 m/s;
- 0 samples above V = 0.25.

This confirms the PR's "below 0.13 m/s". The G4 samestate states and the G6/G7 episodes therefore ran with the shrink
engaged.

**Limitations missing from the PR body:**
- finding 1 (thin CPU evidence for the shrink);
- finding 2 (O(N·C) memory, N ≤ 2495);
- the speed-up applies while particle speeds stay below V. Faster RL actions or separating parts disengage the shrink
  per env, and the env-major part still applies. No receipt counter shows how often it is engaged.

**Wording:**
- "Files changed" omits the two docs files.
- "GPU dev: samestate reruns bitwise" holds for the dev runs. The acceptance's G4 shows a 4.9e-8 relative difference at
  one sample, from float atomics (finding 5).

### 4.1 PR checklist

- **A-pytest-hooks: PASS.** `git diff bfe6d5e..8d6a2b3b -- src tests` adds no `pytest_plugins`, `def pytest_*`,
  `hookimpl`, `importlib`, `pluginmanager`, `sys.modules`, `_pytest`, `sys.argv`, `environ` / `getenv`, monkeypatch,
  `open(` or `Path(`.
- **A-tests: PASS.** No test file other than the new one changes. Counts: baseline 2534 → candidate 2545; the +11 are
  exactly the new file's 11 items. G1 reports `new_skips: []`.
- **Pytest configuration and evaluator file set: PASS (unchanged).**
- **Opt-in only through `particle_contact_mode='alias_free_v2'`: PASS** (§3.1).
- **No reads of harness mode, argv, WP-17 records or output directories: PASS.**
- **Parity level and rollback: PASS.**
  - Level A given the certificate (§3.2, §3.3).
  - `alias_free_v1` and `frozen_v1` are untouched.
  - `{'shrink_speed_m_s': None}` turns the shrink off; tested as the `v2_no_shrink` variant.
  - `env_major_grid=False` is refused, so v2 always means env-major.

## 5. Findings, ranked

1. **MEDIUM — the CPU tests overstate how much they exercise the query shrink.** Details and numbers in §3.7: 3 of 120
   engaged env-substeps, 114 contacts, zero particle contacts in the TRIM tests, two vacuous assertions, a wrong
   docstring.

   *Failure scenario:* a later edit breaks the shrunken walk only after a fast→slow flag transition, or only in slow
   envs with contacts. Examples: indexing `fast_flag` by `p` instead of `e`; clearing the flag before the kernel.
   `tests/test_batched_particle_contacts_v2.py` would still pass, and only a GPU acceptance would catch it.

   *Recommendation (follow-up, not blocking):*
   - Add a per-env counter of shrink-engaged substeps to the receipt, or a test spy as in `probes/shrink_active.py`, and
     assert that it is > 0 together with non-zero contacts.
   - Add a slow-contact scene, for example the block scene without table penetration, or a recorded Native state on CPU.
   - Replace the two `fast_flag` assertions and correct the docstring.

2. **LOW — O(N·C) bucket bounds; new int32 cap at N ≤ 2495; neither is in the PR's limitations.** Details in §3.6.

   *Failure scenarios:*
   - A configuration with `alias_free_v2` and N ≥ 2496 for this workpiece is refused at construction.
   - Under a 32 GB VRAM limit, about 5–9 % fewer envs fit than with v1, from the measured 4.3 versus the allocated
     8.1 MiB/env against about 85 MiB/env.

   *Recommendation:*
   - State both in the PR body, the docs and the receipt.
   - Later: use per-env bounding-box-local cells, so C becomes the env's own span and not the global alias-free dims, or
     use int64 keys.

3. **LOW — `s.hash` no longer describes what v2 queries.**
   - `L2MonitorActiveGrid.audit` (`scripts/rl_ready/run_p1_3a.py:130-147`) and `_grouped_query_audit` audit the grouped
     grid that v2 no longer queries.
   - `restore_descriptor` still rebuilds that grid on the host every launch.

   *Failure scenario:* an env-isolation bug in the env-major buckets, for example a wrong `base`, would still give
   `grouped_query_foreign = 0` and `l2_reported_clean` in G6/G7.

   *Mitigation today:* isolation by construction, the per-env bitwise CPU tests and my GPU probe (0 foreign candidates).

   *Recommendation:* an env-major audit for v2 in a later harness version (the harness is frozen), or document it.
   Optionally drop the dead host rebuild for v2 once the guard semantics are re-argued.

4. **LOW — `reach_slow` is fixed from `s.radius` / `s.dt` at construction (`:676`), while the kernel takes `r` and `dt`
   per launch (`:599`, `:719`).**
   - Every current caller passes `(s.radius, s.dt)`: `tlmpm_v7_2_batch.py:639, 672` and `run_p1_3a.contact_stage`.
   - *Failure scenario:* a future caller that sub-cycles with a larger `dt` would use an unsafe reach.
   - *Recommendation:* `assert dt == s.dt and r == s.radius` in `_launch_v2`, or derive the reach from the arguments.

5. **INFO — wording and evidence notes.**
   - Level A on GPU means the same contribution set. Sums differ in the last bit because of atomic order, as for v1:
     G4 relative difference 4.9e-8.
   - "Samestate reruns bitwise" refers to dev runs.
   - `source_sha256()` changes for every mode (§3.1).

6. **INFO — frozen evaluator stack: contradictory sharing labels and GPU index per row** (§4). No effect on this verdict,
   since the stricter regime was applied. Record it for `acceptance_v3` or the harness.

7. **INFO — the `shrink_speed_m_s=None` sub-variant does wasted work.** It still stores `fast_flag[e] = 1` from every
   particle (`shrink_sq = −1`, `:481-482`, `:724`): n same-address stores per substep. This affects only the non-default
   sub-variant, which was not accepted.

## 6. Tests and probes run for this review

All tests ran in `/home/ningan/RoboticCuttingSim-rl-pr14` at `8d6a2b3b` on CPU (`CUDA_VISIBLE_DEVICES=''`), with Warp
1.17.0. Logs are in `/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/p1_6_p16c1_pr14_review/`.

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -p no:cacheprovider tests/test_batched_particle_contacts_v2.py tests/test_batched_particle_contacts.py -rA -q` | **26 passed**, 246.9 s (`contacts.log`) |
| `.venv/bin/python -m pytest -p no:cacheprovider tests/test_batched_hashgrid_guard.py tests/test_repo_naming.py tests/test_perf_acceptance.py tests/test_batched_frozen_sources.py -rA -q` | **119 passed, 1 skipped** (torch not in `.venv`; NOT_RUN there, run below), 16.5 s (`guard_naming_perf.log`) |
| `.venv/bin/python -m pytest -p no:cacheprovider` on all other `tests/test_batched_*.py` plus `tests/test_vector_env_*.py` | **130 passed, 5 skipped** (torch, NOT_RUN here), 393 s (`batched_vec.log`) |
| `PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaaclab-venv/bin/python -m pytest -p no:cacheprovider` on `test_vector_env_{api,consolidation,contract,p12b,gpu_stream}.py` plus `acceptance_v2` `torch_tests` | 179 passed, **3 failed**, 5 skipped (`torch_tests.log`). The 3 failures were `FileNotFoundError` on `artifacts/wp17` and `artifacts/workcell_asset_cache`, which my new worktree lacked. |
| Same interpreter, after adding the two read-only artifact links: `test_vector_env_contract.py test_perf_acceptance.py test_rl_policies_batched.py test_rl_replay.py` | **91 passed**, 66.8 s (`torch_rerun.log`). This includes the 3 earlier failures and the WP-17 tests that skipped before. |

Still NOT_RUN: `test_vector_env_gpu_stream.py` (CUDA hidden).

**Probes** (scripts and outputs under `probes/` in the same folder):

| Probe | Result |
| --- | --- |
| `gpu_seq_probe.py` on GPU0 under `flock …/rcs_rl_runner/gpu0.lock`, about 4 s | Env-major equals Warp's grouped query on CUDA (§3.3) |
| `shrink_active.py`, `slow_contact.py`, `trim_contacts.py` (CPU) | Shrink engagement and contact counts (§3.7); every scene bitwise |
| `slack_domain.py` | float32 bound: 0 misses (§3.2) |
| `native_speeds.py` (WP-17 records, read-only) | §4 |
| `coords.py` | TRIM local \|x\| ≤ 0.0795 m; dims (189, 69, 66); certified bound 150.5 m/s |
| `gpu_material_002.py` (material-level CUDA run of v2 and v1 with partial reset, GPU0 under the lock) | See §7. `gpu_material_001_failed_host_reference_needs_parity_mode.log` is preserved: a probe error, not a PR failure. |

## 7. GitHub CI and GPU material probe

**CI: PASS.** Workflow "CPU tests" on `8d6a2b3b` succeeded in both runs (checked 2026-11-11 23:10 UTC):
- push run 43102214049, job 130796285432: 2481 passed, 58 skipped, 6 xfailed, 0 failed, 6447.85 s.
- pull_request run 43102223144, job 130796318987: 2451 passed, 88 skipped, 6 xfailed, 0 failed, 6471.09 s.
- Every step succeeded, including "Install media test tools".
- The pull_request run skips 30 more tests than the push run, but neither run has a failure.

The summary lines come from `gh api repos/AnsonNing/RoboticCuttingSim/actions/jobs/<id>/logs`.

**GPU material probe: PASS** (optional; `probes/gpu_material_002*`, GPU0 under `gpu0.lock`, 21:54 UTC, 10 s).

*Set-up:*
- `BatchedCuttingMaterial` on `cuda:0`, with one CUDA graph per control step (200 substeps).
- N=3 coarse TRIM fixture (`T1.CONFIGS`, alias-free dims 77×64×32), device interpolation.
- 6 control steps with cut digits 0, 0, 1, 1, 0, 0; `reset([1])` after step 2.
- `alias_free_v2` and `alias_free_v1` run in separate processes.

*Both modes:*
- `invalid_reason` [0, 0, 0] and guard violations [0, 0, 0];
- 6 descriptor refreshes; `device_hashgrid` status OK;
- the same 6,655 B of net mempool growth during capture;
- a second material in the same process is refused with `UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS`.

*v2 against v1:*
- Final codes are equal for every env (272, 81 and 260 non-zero codes).
- Final positions agree within 9.7e-8 m (float atomic order).
- Per-step tool-contact counts (slot 1) are identical.

*Limitation:* this short scene has 0 particle–particle contacts (finding 1), so it checks graph capture, partial
reset, the guard and isolation under v2, not the contact sums.

The first attempt (`gpu_material.py`) waited 9.7 min for the lock, held by the P2-5 fit job, and then failed in my probe:
`host_reference` interpolation requires `parity_mode`. Its log is preserved.

## 8. Conditions and recommendations

1. **Merge condition (met):** GitHub CI on `8d6a2b3b` succeeded in both runs (§7). If the head changes, the review and
   CI apply only to `8d6a2b3b`.
2. **Add to the PR body or merge note:**
   - findings 1 and 2;
   - that the speed-up depends on per-env particle speeds below V;
   - that the acceptance ran on SHARED_GPU, with the lowest pair at 1.142.
3. **Follow-ups (do not block this merge):**
   - findings 1, 3 and 4 in the code and tests;
   - finding 6 for `acceptance_v3`;
   - optional: drop the dead host rebuild for v2, and the `fast_flag` stores when the shrink is off.
