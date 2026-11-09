# P1-3a — opt-in `alias_free_v1` particle contacts for the batched material

Date: 2026-11-10. Branch `claude/rl-p1-3a` (from `4fc431e`). Design source:
[P1_3_BOTTLENECK_ANALYSIS.md](P1_3_BOTTLENECK_ANALYSIS.md) (candidates C1, C2, certificate (ii), §4 protocol).
Evidence: [`docs/evidence/rl_ready/p1_3a_001.json`](../../evidence/rl_ready/p1_3a_001.json); raw runs under
`artifacts/rl_ready/p1_3/` (git-ignored, on `/mnt/data`).

| Item | Status |
| --- | --- |
| Opt-in `alias_free_v1` (C1, C2, per-env speed guard with exact frozen fallback), default `frozen_v1` unchanged, receipts | DONE |
| CPU L0 bitwise vs frozen singles (15 new tests; batched + WP-17 + naming suite __CPU_COUNT__) | **PASS** |
| GPU same-state check s512 k=33 / s15 k=43, incl. forced fallback (eager) | **PASS** |
| First GPU fallback design (`wp.HashGrid` rebuilt inside a conditional graph node) | FAIL (preserved) → replaced by `FallbackGrid`, PASS |
| **Warp 1.17.0 HashGrid static-descriptor hazard** (found here; the mechanism behind the P1-1 rule "never build a batch after a single LiveCuttingMaterial in one process") | CONFIRMED by probe; every multi-grid same-process A/B of this package is INVALID and was re-run one variant per process; `alias_free_v1` hardened |
| Contact stage alone (graph replay, own descriptor restored), 1 / 16 copies | −51 % / −49 % (SHARED) |
| Kernel split, nsys graph-node trace, N=1/16, all sub-flags | contact kernel −68 % / −50 %; all kernels −35 % / −29 % (SHARED) |
| End-to-end control-step wall, one variant per process, N=1/4/8/16 | **1.40–1.45× faster** (SHARED, 2 reps at N=1/16) |
| Full episodes, alias_free_v1: N=1 s512, N=1 s15, N=4 (s512, s15, s512, s15)__HARDENED_EP_STATUS__ | final codes = Native, labels = Native, frozen evaluator PASS (2 and 121 frames), L2 PASS, 0 guard violations: **PASS** |
| L3 position band vs Native (P1-0 single-env band) | s512 inside everywhere; s15 **outside** in __S15_OUT__ — the unchanged frozen batched path (P1-2 `m0_s15_001`) is outside too |
| EXCLUSIVE_GPU timing, RUN_TO_RUN determinism, C3 | NOT_RUN |

## 1. What changed

New module `src/robotic_cutting_sim/batched/particle_contacts.py`, opt-in through
`BatchedCuttingMaterial(..., particle_contact_mode=, particle_contact_options=)`,
`BatchedTLMPMv7_2(packed, device, particle_contact=)` and `VectorCuttingEnv(..., particle_contact_mode=,
particle_contact_options=)`.

- `frozen_v1` (default and rollback): `tlmpm_v7_2_batch.step` runs the same two statements as before
  (`self.hash.build(...)`, `wp.launch(_particle_contacts_b, ...)`). The kernel source is byte-identical to `4fc431e`
  (sha256 of its source pinned in `test_default_mode_is_unchanged_and_receipts_name_the_mode`). No frozen module
  changed (`tests/test_batched_frozen_sources.py` PASS).
- `alias_free_v1` with independent sub-flags (defaults in brackets):
  - `skip_zero_impulse_atomics` [on] — C1: the two impulse atomics of a pair are skipped when the pair impulse is
    exactly zero component-wise (`!= 0`; NaN still propagates). Level A unconditionally (analysis fact (i)).
  - `alias_free_hash_dims` [on] — C2: uniform grid dims from the packed reference bounding box of every env plus
    `margin_m` [0.02 m], never below 64×64×32. TRIM 1 mm (s512/s15 spans 186×66×62–63 cells): **189×69×66**
    (1.31·10⁵ → 8.61·10⁵ cells; cell start/end arrays 1.0 → 6.9 MB, shared by all envs). Level A given
    certificate (ii).
  - `speed_guard` [`fallback`] — every substep, a per-particle kernel flags env e when `|v_p|² > bound²` or `x_p` is
    not finite (plain stores, no atomics). `fallback`: flagged envs are skipped by the alias-free kernel and run, in
    the same substep, through the frozen kernel body over the reference-dims (64×64×32) candidate sequence inside a
    conditional graph node (`wp.capture_if`); the substep is counted in `contact_guard_violations[e]` and the material
    sets the sticky invalid bit `INVALID_CONTACT_SPEED_GUARD = 64` (`"CONTACT_SPEED_GUARD"`). `flag`: count and bit
    only, no fallback (no parity claim for that env). `off`: unchecked (A/B timing only; receipt
    `certificate: unchecked`). `speed_bound_m_s` may lower (never raise) the certified bound.
  - Refused (`UNSUPPORTED_PARTICLE_CONTACT_MODE`, new code): unknown mode/option, options on `frozen_v1`, a guard
    without C2, all sub-flags off, a bound above the certified one.
- **HashGrid descriptor guard** (alias_free_v1 only, §3.0): one device `HashGrid` per material (with C2 the unused
  64×64×32 grid is released and `solver.hash` is the alias-free grid) and `advance()` rebuilds that grid on the host
  right before every graph launch, so the graph never replays another grid's descriptor. Measured cost: none
  (§3.4, `hardened_*` runs).
- Receipts (`material.receipt()["particle_contact"]`, also in the solver receipt): mode, every sub-flag, reference
  and chosen dims, per-env cell spans, cell width, alias distance, certified and used guard bound, certificate
  status, conditional-graph support, fallback grid bytes, descriptor guard, violation counts; `particle_contacts.py`
  is in `batched_source_sha256` and the solver `source_sha256`.
- Partial reset clears the reset envs' guard counters and flags; reset coverage stays vars()-complete
  (`CONTACT_GUARD_RESET_ARRAYS`, `CONTACT_GUARD_SCRATCH`).

### Deviations from the analysis text (all conservative)

1. **Alias distance `(D_min − 3)·h`, dims `≥ span + 3`.** The analysis used `(D_min − 2)·h` for the nominal 3×3×3
   query box. Warp computes the box as `floor((x ∓ R)·inv)` in float32, which can reach `c_p ± 2` when `R·inv`
   rounds above 1, so the rounding-safe alias distance is `(D_min − 3)·h` (TRIM: 29·1.08 mm = 31.3 mm, certified
   bound 150.5 m/s with a 1 % distance safety; recorded maximum particle speed 0.12 m/s).
2. **The guard also flags non-finite positions** (the fallback then reproduces the frozen sequence for that env).
3. **Fallback grid.** `wp.HashGrid.build` cannot run inside a conditional graph body: instantiation fails with
   `invalid argument` (probe on the RTX 5090, driver 570.211.01, Warp 1.17.0: kernels, memsets and
   `radix_sort_pairs` work in a conditional body, `HashGrid.build` grouped or ungrouped does not — it copies the
   grid descriptor from pageable host memory). The first GPU runs (`samestate_s512_k33_001`,
   `samestate_s15_k43_001`) failed on this and are preserved. `FallbackGrid` rebuilds the reference-dims buckets
   from a key kernel (`env·C + cell` with Warp's `hash_grid_index` arithmetic), the stable `radix_sort_pairs` and a
   range kernel, and the fallback kernel walks the virtual cells in Warp's x-fastest order. A CPU test asserts the
   identical candidate sequence (ids and order) to a grouped `wp.HashGrid` query for 3 dims sets incl. heavy
   aliasing; the forced-fallback GPU same-state check matches the frozen kernel (§3.1).

## 2. CPU tests (Warp 1.17.0 CPU, `.venv`, Python 3.12.12, NumPy 2.5.4)

```bash
CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp .venv/bin/python -m pytest -p no:cacheprovider -q \
    tests/test_batched_*.py tests/test_wp17_frontier.py tests/test_repo_naming.py
# __CPU_COUNT__   (baseline at 4fc431e: 204 passed in 347.46 s)
CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src \
    .deps/isaacsim-venv/bin/python -m pytest -p no:cacheprovider -q tests/test_vector_env_api.py \
    tests/test_rl_policies_batched.py tests/test_rl_replay.py
# 24 passed in 87.86 s (Torch venv, CPU; vector-env pass-through)
```

New file `tests/test_batched_particle_contacts.py` (15 tests). Every comparison is `np.array_equal` over every Warp
array of the frozen single solver (`compare_env`), after every substep (block scenes) or control step (TRIM):

| Test | Scene | Variants | Non-vacuity asserted |
| --- | --- | --- | --- |
| `test_block_scenes_every_variant_bitwise[frozen_dims]` | 3 heterogeneous compressed blocks, conflicting codes, handle + pads, per-env digits, 40 substeps | frozen_v1 batch, c1, c2_fallback, c1_c2_fallback, c1_c2_flag, c1_c2_unguarded | contacts in envs 1–2; a particle with ≥ 2 non-zero contributions in one substep (max 8) |
| `…[heavy_alias_5x5x5]` | same, singles' grid forced to `HashGrid(5,5,5)` (instance attribute of the test object) | same, `reference_hash_dims=(5,5,5)` | > 10⁵ aliased candidates, ≥ 2 contributions |
| `test_speed_guard_fallback_reproduces_aliased_reference` | 6 env-1 particles at 80 m/s, reference 5×5×5 (bound 12.4 m/s) | fallback vs flag | aliased pairs with non-zero impulse > 0; fallback bitwise for all envs; flag differs in env 1 only; counts only in env 1 |
| `test_full_resolution_trim_substep_bitwise[c1_c2_fallback / c1]` | full-resolution TRIM (1 mm, > 10⁵ particles), blade contact | 2 | > 10⁶ aliased candidates at 64×64×32 |
| `test_n3_heterogeneous_trim_sequence_bitwise` | heterogeneous N=3 TRIM (seeds 512/15/7, different materials and edge speeds), 4 control steps, digits 0 and 1, singles forced to 8×8×8 | alias_free_v1 | codes written in every env, no guard violation |
| `test_partial_reset_alias_free_equals_default_mode` | N=3 material, reset([1]) after step 2, 4 control steps | alias_free_v1 vs frozen_v1 | stale guard count cleared |
| `test_material_guard_sets_sticky_invalid_bit_and_reset_clears_it` | injected 1 m/s particle, bound 0.3 m/s | alias_free_v1 vs frozen_v1 | bit 64 only in env 1, strict raises, reset clears |
| `test_fallback_grid_reproduces_warp_grouped_query_sequence[3 dims]` | 3 groups, random + cell-boundary lattice points | FallbackGrid vs `wp.HashGrid` | sequences untruncated |
| receipts / layout / single-grid guard / coverage / refusals / default-unchanged | — | — | — |

Existing tests changed only where the surface grew: `test_batched_layout.py` (the new UNSUPPORTED code is raised),
`test_batched_coupling.py` (receipt source list includes `particle_contacts.py`). The CPU path never uses the Warp
device descriptor copy of §3.0 (host builds have no static buffer, no graphs), so the CPU tests are unaffected by it.

## 3. GPU runs (RTX 5090 ×2, driver 570.211.01, `.deps/isaacsim-venv`, Warp 1.17.0, no Kit)

Scripts: `scripts/rl_ready/run_p1_3a.py`, `scripts/rl_ready/probe_hashgrid_descriptor.py`, `/usr/local/bin/nsys`
2024.6.2. Each job under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock`, `CUDA_VISIBLE_DEVICES=<N>
TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src`; GPU 0 for `samestate_*`, `ab_*`, `episode_512_001`, `episode_15_001`,
`episode_512_15_512_15_001`, `nsys_001`, `profile_*_001` (except the sub-flag ones); GPU 1 for everything else.
**Every GPU row is `SHARED_GPU`** (≥ 1 foreign compute process always present; foreign SM mean 23–62 %, max 45–99 %
depending on the window). Workload A: seeded TRIM s512/s15 (WP-17 configuration), 102,732 / 103,898 particles per
env, 1 mm / 4 mm / 0.1 ms, 200 substeps per 50 Hz control step, tlmpm_v7_2 law 8, handle + two measured pads, no
cameras, no recording, `reduced_fidelity` (ideal edge tracking), warm.

### 3.0 Warp HashGrid static-descriptor hazard (found during this work)

`warp/native/hashgrid.cpp::hash_grid_update_device_impl` keeps the grid descriptor in a function-static host
variable "so that we can perform host->device memcpy from this variable and have it safely recorded inside CUDA
graphs". The captured memcpy node therefore copies **whatever that static holds at replay time** — the descriptor
of the last grid built on the host in the process — into the replaying grid's id, while the captured rebuild
kernels still write the grid's own buffers. `probe_hashgrid_descriptor.py` (`hashgrid_descriptor_probe_001`, GPU 1):
grid A (points near the origin) and grid B (points 1 m away); a graph rebuilds A and counts query candidates on A.

| replay of A's graph | candidates (eager reference 8,804,086) |
| --- | --- |
| right after capture | 8,804,086 (equal) |
| after `B.build` on the host | **0** (A's query read B's descriptor) |
| after `A.build` on the host again | 8,804,086 (equal) |

Consequences:
- Any process with two device grids replays all graphs with the last-built grid's dims, cell arrays and point
  buffers. This is the mechanism behind the P1-1 rule "never build a batch after a single LiveCuttingMaterial in one
  process". It affects `frozen_v1` too (unchanged here; a versioned default-path guard is open, §6).
- **All multi-variant same-process runs of this package are INVALID** and are preserved, not used:
  `ab_n{1,4,8,16}_001`, `smoke_ab_n1_001`, `pair_n16_reversed_001` (the variant whose grid was built last on the host
  sets the speed of every variant — 2.42 s per N=16 step for all five variants in `ab_n16_001`, an alias-free grid
  last; 3.21 / 3.42 s for alias-free / frozen in `pair_n16_reversed_001`, frozen last), the alias_free_v1 half of
  `episode_512_001` / `episode_15_001` (its graph replayed the frozen env's grid and particle buffers), and the
  stage-graph parts of `samestate_*_002`. Eager comparisons are valid (an eager build copies the static
  immediately). The first conclusion drawn from these runs ("no end-to-end gain") is withdrawn.
- Valid GPU evidence = one variant per process (`solo_*`, `profile_*`, `nsys_*`, `episode_solo_*`,
  `episode_512_15_512_15_001`, `hardened_*`), the eager same-state checks and `stagebench_*_002` (each replay is
  preceded by a host rebuild of that env's own grid, outside the clocks).
- alias_free_v1 is hardened (single grid + host rebuild before each graph launch, §1). `hardened_pair_n16_001` (alias
  built first, frozen last — the order that slowed the alias env to frozen speed before): __HARDENED_PAIR__

### 3.1 Same-state check (contact stage alone, eager, identical recorded Native states)

`samestate_s512_k33_002` (t = 6.6 s) and `samestate_s15_k43_002` (t = 8.6 s), GPU 0: 4 copies of the recorded
state (positions/velocities/codes, world → local) per batch; frozen_v1 twice (GPU float-atomic band), alias_free_v1,
and a forced-fallback batch (`speed_bound_m_s = 0`: every env takes the conditional `FallbackGrid` path).

| check | s512 k=33 | s15 k=43 |
| --- | --- | --- |
| particles with non-zero contact impulse (frozen / alias-free / forced fallback) | 124 / 124 / 124, sets identical | 272 / 272 / 272, sets identical |
| contact count `stats[4]` per env | 21 in all 4 envs, identical | 90 in all 4 envs, identical |
| `stats[6]` (max overlap) | identical | identical |
| impulse alias-free vs frozen | **bitwise equal** | max abs diff 7.1e-15 (rel 4.9e-8) |
| impulse frozen vs frozen rerun (band) | bitwise equal | 1.4e-14 (rel 9.8e-8) |
| impulse forced fallback vs frozen | bitwise equal | 1.4e-14 (rel 9.8e-8) |
| guard violations (alias-free / forced fallback) | 0 / 2 per env (fallback ran) | 0 / 2 per env |
| max particle speed of the state | 0.061 m/s | 0.104 m/s |

`stagebench_s15_c{1,16}_002` (GPU 1) repeat the comparison after the timed replays: identical non-zero sets and
counts for every variant (68 / 1088 particles), impulse within 1.4e-14. Status: **PASS**.

### 3.2 Contact stage alone — three clocks (`stagebench_s15_c{1,16}_002`, GPU 1, s15 k=43)

One graph per variant holding impulse/stats memsets + hash build (incl. CUB sort) + contact kernel(s) [+ guard and
conditional fallback node]; 40 (1 copy) / 20 (16 copies) interleaved repeats; median (min) ms.

| copies | clock | frozen_v1 | c1_only | c1_c2_unguarded | alias_free_v1 | alias-free vs frozen |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | graph replay, CUDA events | 0.487 (0.483) | 0.462 (0.459) | 0.233 (0.230) | 0.238 (0.232) | **−51 %** |
| 1 | graph replay, host wall + sync | 0.493 | 0.467 | 0.237 | 0.243 | −51 % |
| 1 | eager, host wall + sync | 0.535 | 0.510 | 0.279 | 0.313 | −41 % |
| 16 | graph replay, CUDA events | 4.894 (4.866) | 4.303 (4.287) | 2.496 (2.483) | 2.515 (2.497) | **−49 %** |
| 16 | graph replay, host wall + sync | 4.905 | 4.310 | 2.501 | 2.516 | −49 % |
| 16 | eager, host wall + sync | 4.947 | 4.361 | 2.545 | 2.576 | −48 % |

C2 carries the gain; C1 alone saves 5 % (1 copy) / 12 % (16 copies) and nothing measurable on top of C2; the guard
(kernel + conditional node) costs 5–19 µs per stage.

### 3.3 Kernel split, CUDA-graph node trace (nsys `--cuda-graph-trace=node`)

One process per (N, variant), 3 warm-up + 3 profiled control steps (600 substeps), s512/s15 replay (alternating).
µs per substep = Σ instances × median (min) duration / 600. Durations on a shared GPU include time-slice
preemption, so medians of long kernels at N=16 are inflated up to 2× (min is the better estimate of uncontended
execution).

__NSYS_TABLE__

At N=1 the frozen contact kernel is 52 % of kernel time (P1-0 eager split 47–54 %), after C1+C2 ≈ 26 %; p2g is then
the largest kernel at N=16 (≈ 36 % of median time) — the analysis' next target (C7, Level C, separately versioned).
The grouped-build CUB sort is 10–16 % at N=1, 1–2 % at N=16.

### 3.4 End-to-end control step, one variant per process (no profiler)

`solo_n{N}_{variant}_{rep}` (GPU 1, back-to-back processes, rep 1 then rep 2 at N=1/16): replay from reset, steps
3–14, `env.step_measured_edge` = one 50 Hz control step (200-substep graph + coupling + strict int32 readback),
device-synchronised; build/capture excluded (warm).

__SOLO_TABLE__

alias_free_v1 is **1.40–1.45× faster per control step at every N** (≈ 87 ms less at N=1, 0.96 s less at N=16). The
nsys kernel reduction accounts for ≈ 46 ms (N=1, medians); the rest is co-tenant time-slice waiting, which scales
with our GPU time on this shared GPU. Hardened code (`hardened_solo_n{1,16}_alias_free_v1_001`): 0.2019 s / 2.4399 s
median — the descriptor guard costs nothing measurable. The window covers the approach (no particle contacts yet),
which the analysis predicted to be cost-representative (the frozen kernel visits ~8.3 M candidates in every phase);
the full episodes below cover all phases at N=1 and N=4.

### 3.5 Full-episode replays (frozen evaluator, Native comparison)

Drive: recorded measured edge poses, resolved digits and measured pad boxes (`MeasuredTrajectoryDriver` →
`step_measured_edge`), strict mode, L2 monitor every step, grouped-query audits on the grid actually queried
(steps 0/300/600/900/1200), 121 Native samples, frozen evaluator on 2 frames and on the 121-frame series.
P1-0 bands (4 single-env runs): max position difference over time s512 1.39–3.85e-5 m, s15 1.96–4.62e-4 m; final
max s512 1.4–3.8e-5 m, s15 6.3e-5–1.2e-4 m.

| run (GPU) | env / seed | final codes = Native | labelled digit 0 (Native) | evaluator 2 / 121 | L2 | guard | max pos. diff over time | final max | warm env-steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_solo_512_001 (1) | 0 / 512 | yes (0) | 37958 (37958) | PASS / PASS | PASS | 0 | 2.49e-5: inside | 2.49e-5: inside | 7.385 |
| episode_solo_15_001 (1) | 0 / 15 | yes (0) | 85786 (85786) | PASS / PASS | PASS | 0 | 8.34e-4: **outside** | 1.74e-4: **outside** | 7.467 |
| episode_512_15_512_15_001 (0), N=4 | 0 / 512 | yes (0) | 37958 | PASS / PASS | PASS | 0 | 3.85e-5: inside | 3.85e-5: edge of band | 5.839 (batch) |
| | 1 / 15 | yes (0) | 85786 | PASS / PASS | PASS | 0 | 1.73e-4: below band | 7.9e-5: inside | |
| | 2 / 512 | yes (0) | 37958 | PASS / PASS | PASS | 0 | 2.31e-5: inside | 2.31e-5: inside | |
| | 3 / 15 | yes (0) | 85786 | PASS / PASS | PASS | 0 | 8.20e-4: **outside** | 1.80e-4: **outside** | |
__HARDENED_EP_ROWS__| reference: P1-2 `m0_s512_001` (frozen_v1, solo) | 0 / 512 | yes | 37958 | PASS | PASS | — | 2.24e-5 | 2.24e-5 | 3.508 (other window) |
| reference: P1-2 `m0_s15_001` (frozen_v1, solo) | 0 / 15 | yes | 85786 | PASS | PASS | — | 6.50e-4: **outside** | 1.36e-4: **outside** | 3.288 (other window) |

Final codes equal Native with 0 differing particles in every alias-free episode; evaluator PASS on 2 and 121 frames
everywhere; L2 PASS (no digit rewrite, finite, min detF > 0, no slot overflow, no invalid flag, grouped-query
foreign = 0, mass bitwise constant); no guard violation (certified bound 150.5 m/s). Position band: s512 inside
everywhere; **s15 outside the P1-0 single-env band** in __S15_OUT__ — like the unchanged frozen batched path in P1-2
(`m0_s15_001`), so the band is not met by the batched GPU path at s15 in general; it is reported, not attributed to
alias_free_v1. Transient code differences (1–2 particles at t = 5.4 s or 9.6–10.0 s for s15, 6.4 s for s512, as in
P1-0) occur, none at the end. (`episode_512_001` / `episode_15_001` also contain a frozen_v1 control interleaved in
the same process; by the §3.0 mechanism its own grid was the last built on the host, so that half is plausibly
valid — s512 2.45e-5 m inside, s15 1.99e-2 m outside — but it is not used as evidence.)

## 4. Rows (plan format, valid runs only)

__ROWS__

## 5. Rollback

`particle_contact_mode="frozen_v1"` (the default) — or omit the argument — restores the exact pre-P1-3a execution:
the same two statements in `BatchedTLMPMv7_2.step`, the byte-identical `_particle_contacts_b` (source sha256 pinned in
the tests), no extra arrays, kernels, graph nodes or host rebuilds (`advance()` only adds the host rebuild when
`solver.particle_contact` is set). Removing the feature entirely = reverting the P1-3a commits; nothing else depends
on it.

## 6. Failures and open items

- **FAIL, preserved:** `samestate_s512_k33_001`, `samestate_s15_k43_001` — the first fallback design rebuilt a
  `wp.HashGrid` inside the conditional node; graph instantiation failed (`Warp CUDA error 1: invalid argument`,
  `wp_cuda_graph_create_exec`). Fixed by `FallbackGrid` (commit a18e8f5).
- **FAIL, preserved:** `stagebench_s15_c{1,16}_001` — runner bug (`wp.record_event` has no `stream` argument);
  re-run as `_002`.
- **INVALID, preserved:** every multi-variant same-process run listed in §3.0 (Warp static-descriptor hazard).
- **Open (not P1-3a):** the hazard affects `frozen_v1` in any process with a second device grid (e.g. a single
  `LiveCuttingMaterial` or a second batch). A versioned default-path guard (host rebuild before launch, or one grid
  per process asserted) needs its own work package; until then keep one material per process and run A/B variants
  in separate processes.
- **s15 L3 position band** exceeded by alias-free and by the frozen batched path (P1-2); final codes, labels and the
  frozen evaluator are unaffected. Needs a band re-derived for the batched GPU path (≥ 3 runs per seed).
- Not run: EXCLUSIVE_GPU timing (both GPUs shared in every window), `RUN_TO_RUN` comparison (P1-2 L4 pending), C3
  (P1-3b).

## 7. Commands

```bash
PY=.deps/isaacsim-venv/bin/python R=scripts/rl_ready/run_p1_3a.py     # each under the GPU lock (header of R)
$PY $R --mode samestate --seed 512 --sample 33 --copies 4 --gpu-index 0
$PY $R --mode samestate --seed 15 --sample 43 --copies 4 --gpu-index 0
$PY $R --mode stagebench --seed 15 --sample 43 --copies 1 --repeats 40 \
    --variants frozen_v1,alias_free_v1,c1_only,c1_c2_unguarded --gpu-index 1          # and --copies 16 --repeats 20
$PY $R --mode ab --num-envs $N --steps 12 --warmup 2 --kernel-every 0 --variants $V --run-name solo_n${N}_$V \
    --gpu-index 1          # N in 1 4 8 16, V in frozen_v1 alias_free_v1 (2 reps at N=1/16)
$PY $R --mode episode --seeds 512 --variants alias_free_v1 --run-name episode_solo_512 --gpu-index 1
$PY $R --mode episode --seeds 15 --variants alias_free_v1 --run-name episode_solo_15 --gpu-index 1
$PY $R --mode episode --seeds 512,15,512,15 --variants alias_free_v1 --gpu-index 0
nsys profile -o artifacts/rl_ready/p1_3/nsys_00X/n${N}_$V --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop -t cuda --sample=none --cpuctxsw=none $PY $R --mode profile --num-envs $N --variants $V \
    --steps 3 --warmup 3 --gpu-index <gpu>
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv -o <prefix> <prefix>.nsys-rep
CUDA_VISIBLE_DEVICES=1 $PY scripts/rl_ready/probe_hashgrid_descriptor.py artifacts/rl_ready/p1_3/hashgrid_descriptor_probe_001/result.json
$PY $R --mode ab --num-envs $N --steps 12 --warmup 2 --kernel-every 0 --variants alias_free_v1 \
    --run-name hardened_solo_n${N}_alias_free_v1 --gpu-index 1                              # N in 1 16
$PY $R --mode ab --num-envs 16 --steps 12 --warmup 2 --kernel-every 0 --variants alias_free_v1,frozen_v1 \
    --run-name hardened_pair_n16 --gpu-index 1
$PY $R --mode episode --seeds 512 --variants alias_free_v1 --run-name hardened_episode_512 --gpu-index 1   # and 15
PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_p1_3a_evidence.py ... --out docs/evidence/rl_ready/p1_3a_001.json
```

## 8. Merge readiness

Ready to merge into `claude/rl-ready-rubber-physics` as an **opt-in** mode: the default path is unchanged and
byte-identical, physics parity is bitwise on CPU and inside the frozen GPU band on GPU, every alias-free full
episode reproduces the Native final codes and passes the frozen evaluator, and the measured end-to-end gain is
1.40–1.45× (SHARED_GPU, one variant per process). Do not switch the default before an EXCLUSIVE_GPU confirmation;
fix the Warp descriptor hazard for `frozen_v1` in its own work package, and run any later A/B one variant per process
(or restore the descriptor before each launch).
