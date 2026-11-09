# P1-3a — opt-in `alias_free_v1` particle contacts for the batched material

Date: 2026-11-10. Branch `claude/rl-p1-3a` (from `4fc431e`). Design source:
[P1_3_BOTTLENECK_ANALYSIS.md](P1_3_BOTTLENECK_ANALYSIS.md) (candidates C1, C2, certificate (ii), §4 protocol).
Evidence: [`docs/evidence/rl_ready/p1_3a_001.json`](../../evidence/rl_ready/p1_3a_001.json); raw runs under
`artifacts/rl_ready/p1_3/` (git-ignored, on `/mnt/data`).

__STATUS_TABLE__

## 1. What changed

New module `src/robotic_cutting_sim/batched/particle_contacts.py`, opt-in through
`BatchedCuttingMaterial(..., particle_contact_mode=, particle_contact_options=)`,
`BatchedTLMPMv7_2(packed, device, particle_contact=)` and `VectorCuttingEnv(..., particle_contact_mode=,
particle_contact_options=)`.

- `frozen_v1` (default and rollback): `tlmpm_v7_2_batch.step` runs the same two lines as before
  (`self.hash.build(...)`, `wp.launch(_particle_contacts_b, ...)`). The kernel source is byte-identical to `4fc431e`
  (sha256 of its source pinned in `test_default_mode_is_unchanged_and_receipts_name_the_mode`). No frozen module
  changed (`tests/test_batched_frozen_sources.py` PASS).
- `alias_free_v1` with independent sub-flags (defaults in brackets):
  - `skip_zero_impulse_atomics` [on] — C1: the two impulse atomics of a pair are skipped when the pair impulse is
    exactly zero component-wise (`!= 0`; NaN still propagates). Level A unconditionally (analysis fact (i)).
  - `alias_free_hash_dims` [on] — C2: uniform grid dims from the packed reference bounding box of every env plus
    `margin_m` [0.02 m], never below 64×64×32. TRIM 1 mm (s512/s15 spans 186×66×62–63 cells): **189×69×66**
    (1.31·10⁵ → 8.61·10⁵ cells; cell start/end arrays 1.0 → 6.9 MB per build, shared by all envs). Level A given
    certificate (ii).
  - `speed_guard` [`fallback`] — every substep, a per-particle kernel flags env e when `|v_p|² > bound²` or `x_p` is
    not finite (plain stores, no atomics). `fallback`: flagged envs are skipped by the alias-free kernel and run, in
    the same substep, through the frozen kernel body over the reference-dims (64×64×32) candidate sequence inside a
    conditional graph node (`wp.capture_if`); the substep is counted in `contact_guard_violations[e]`, and the
    material sets the sticky invalid bit `INVALID_CONTACT_SPEED_GUARD = 64` (`"CONTACT_SPEED_GUARD"`). `flag`: count
    and bit only, no fallback (no parity claim for that env). `off`: unchecked (A/B timing only; receipt
    `certificate: unchecked`). `speed_bound_m_s` may lower (never raise) the certified bound.
  - Refused (`UNSUPPORTED_PARTICLE_CONTACT_MODE`, new code): unknown mode/option, options on `frozen_v1`, a guard
    without C2, all sub-flags off, a bound above the certified one.
- Receipts (`material.receipt()["particle_contact"]`, also in the solver receipt): mode, every sub-flag, reference
  and chosen dims, per-env cell spans, cell width, alias distance, certified and used guard bound, certificate
  status, conditional-graph support, fallback grid bytes, violation counts; `particle_contacts.py` is in
  `batched_source_sha256` and the solver `source_sha256`.
- Partial reset clears the reset envs' guard counters and flags; reset coverage stays vars()-complete
  (`CONTACT_GUARD_RESET_ARRAYS`, `CONTACT_GUARD_SCRATCH`).

### Deviations from the analysis text (all conservative)

1. **Alias distance `(D_min − 3)·h`, dims `≥ span + 3`.** The analysis used `(D_min − 2)·h` for the nominal
   3×3×3 query box. Warp computes the box as `floor((x ∓ R)·inv)` in float32, which can reach `c_p ± 2` when
   `R·inv` rounds above 1, so the rounding-safe alias distance is `(D_min − 3)·h` (TRIM: 29·1.08 mm = 31.3 mm,
   certified bound 150.5 m/s with a 1 % distance safety; recorded maximum particle speed 0.12 m/s).
2. **The guard also flags non-finite positions** (the fallback then reproduces the frozen sequence for that env).
3. **Fallback grid.** `wp.HashGrid.build` cannot run inside a conditional graph body: instantiation fails with
   `invalid argument` (probe on the RTX 5090, driver 570.211.01, Warp 1.17.0: kernels, memsets and
   `radix_sort_pairs` work in a conditional body, `HashGrid.build` grouped or ungrouped does not; the build copies
   the grid descriptor from pageable host memory). The first GPU runs (`samestate_s512_k33_001`,
   `samestate_s15_k43_001`) failed on this and are preserved. `FallbackGrid` rebuilds the reference-dims buckets
   from a key kernel (`env·C + cell` with Warp's `hash_grid_index` arithmetic), the stable `radix_sort_pairs` and a
   range kernel, and the fallback kernel walks the virtual cells in Warp's x-fastest order. A CPU test asserts the
   identical candidate sequence (ids and order) to a grouped `wp.HashGrid` query for 3 dims sets incl. heavy
   aliasing; the forced-fallback GPU same-state check below matches the frozen kernel.

## 2. CPU tests (Warp 1.17.0 CPU, `.venv`, Python 3.12.12, NumPy 2.5.4)

```bash
CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp .venv/bin/python -m pytest -p no:cacheprovider -q \
    tests/test_batched_*.py tests/test_wp17_frontier.py tests/test_repo_naming.py
# 219 passed in 456.79 s  (baseline at 4fc431e: 204 passed in 347.46 s)
CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src \
    .deps/isaacsim-venv/bin/python -m pytest -p no:cacheprovider -q tests/test_vector_env_api.py \
    tests/test_rl_policies_batched.py tests/test_rl_replay.py
# 24 passed in 87.86 s (Torch venv, CPU; vector-env pass-through)
```

New file `tests/test_batched_particle_contacts.py` (15 tests). Every comparison is `np.array_equal` over every Warp
array of the frozen single solver (`compare_env`), after every substep (block scenes) or control step (TRIM):

| Test | Scene | Variants | Non-vacuity asserted |
| --- | --- | --- | --- |
| `test_block_scenes_every_variant_bitwise[frozen_dims]` | 3 heterogeneous compressed blocks, conflicting codes, handle + pads, per-env digits, 40 substeps | frozen_v1 batch, c1, c2_fallback, c1_c2_fallback, c1_c2_flag, c1_c2_unguarded | contacts in envs 1–2, a particle with ≥ 2 non-zero contributions in one substep (max 8) |
| `…[heavy_alias_5x5x5]` | same, singles' grid forced to `HashGrid(5,5,5)` (instance attribute of the test object) | same, `reference_hash_dims=(5,5,5)` | > 10⁵ aliased candidates, ≥ 2 contributions |
| `test_speed_guard_fallback_reproduces_aliased_reference` | 6 env-1 particles at 80 m/s, reference 5×5×5 (bound 12.4 m/s) | fallback vs flag | aliased pairs with non-zero impulse > 0; fallback bitwise for all envs; flag differs in env 1 only; counts only in env 1 |
| `test_full_resolution_trim_substep_bitwise[c1_c2_fallback / c1]` | full-resolution TRIM (1 mm, > 10⁵ particles), blade contact | 2 | > 10⁶ aliased candidates at 64×64×32 |
| `test_n3_heterogeneous_trim_sequence_bitwise` | heterogeneous N=3 TRIM (seeds 512/15/7, different materials and edge speeds), 4 control steps, digits 0 and 1, singles forced to 8×8×8 | alias_free_v1 | codes written in every env, no guard violation |
| `test_partial_reset_alias_free_equals_default_mode` | N=3 material, reset([1]) at step 2, 4 control steps | alias_free_v1 vs frozen_v1 | stale guard count cleared |
| `test_material_guard_sets_sticky_invalid_bit_and_reset_clears_it` | injected 1 m/s particle, bound 0.3 m/s | alias_free_v1 vs frozen_v1 | bit 64 only in env 1, strict raises, reset clears |
| `test_fallback_grid_reproduces_warp_grouped_query_sequence[3 dims]` | 3 groups, random + boundary lattice points | FallbackGrid vs `wp.HashGrid` | sequences untruncated |
| receipts / coverage / refusals / default-unchanged | — | — | — |

Existing tests updated only where the surface grew: `test_batched_layout.py` (the new UNSUPPORTED code is raised),
`test_batched_coupling.py` (receipt source list includes `particle_contacts.py`).

## 3. GPU runs (RTX 5090, driver 570.211.01, `.deps/isaacsim-venv`, Warp 1.17.0, no Kit)

All GPU runs: GPU index 0 under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock`,
`CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src`, scripts `scripts/rl_ready/run_p1_3a.py` and
`/usr/local/bin/nsys` 2024.6.2. **Every GPU row is `SHARED_GPU`**: 3 foreign compute processes were always present
(foreign SM utilisation mean 45–56 %, max 79–99 % during the runs; the same-state runs saw a mean of 96 %). Workload
A: seeded TRIM s512/s15 (WP-17 configuration), 102,732 / 103,898 particles per env, 1 mm / 4 mm / 0.1 ms, 200
substeps per 50 Hz control step, tlmpm_v7_2 law 8, handle + two measured pads, no cameras, no recording,
`reduced_fidelity` (ideal edge tracking). Variants: `frozen_v1`, `alias_free_v1` (C1 + C2 + fallback guard) and,
for timing, `c1_only`, `c2_fallback_no_c1`, `c1_c2_unguarded`.

### 3.1 Same-state check (contact stage alone, identical recorded Native states)

`samestate_s512_k33_002` (t = 6.6 s) and `samestate_s15_k43_002` (t = 8.6 s): 4 copies of the recorded state per
batch; frozen_v1 and alias_free_v1 eagerly, frozen twice (GPU float-atomic band), plus a forced-fallback batch
(`speed_bound_m_s = 0`: every env takes the conditional `FallbackGrid` path), then the same comparison after 40
replays of a CUDA graph of each stage (alias_free_v1 with its conditional node).

| check | s512 k=33 | s15 k=43 |
| --- | --- | --- |
| particles with non-zero contact impulse (frozen / alias-free / forced fallback) | 124 / 124 / 124, sets identical | 272 / 272 / 272, sets identical |
| contact count `stats[4]` per env | 21 in all 4 envs, identical | 90 in all 4 envs, identical |
| `stats[6]` (max overlap) | identical | identical |
| impulse alias-free vs frozen | **bitwise equal** | max abs diff 7.1e-15 (rel 4.9e-8) |
| impulse frozen vs frozen rerun (band) | bitwise equal | 1.4e-14 (rel 9.8e-8) |
| impulse forced fallback vs frozen | bitwise equal | 1.4e-14 (rel 9.8e-8) |
| after graph replays (alias-free, fallback) | identical sets and counts | identical sets and counts, diffs within the band |
| guard violations (alias-free / forced fallback) | 0 / 2 per env (fallback ran) | 0 / 2 per env |
| max particle speed of the state | 0.061 m/s | 0.104 m/s |

Status: **PASS** (identical contacting sets and counts; values bitwise or inside the frozen run-to-run band).

### 3.2 Kernel split, CUDA-graph node trace (nsys `--cuda-graph-trace=node`)

`nsys_001`: one process per (N, variant), 3 warm-up + 3 profiled control steps (600 substeps), replay of s512/s15
(alternating). Per-substep GPU time = Σ instances × median (min) kernel duration / 600; durations on a shared GPU
include time-slice preemption, so medians at N=16 are inflated (median ≈ 2× min for the long kernels); the min
column is the better estimate of uncontended execution.

| N | stage (µs per substep, median / min) | frozen_v1 | alias_free_v1 | change |
| --- | --- | --- | --- | --- |
| 1 | particle-contact kernel | 348.7 / 344.1 | 115.0 / 113.9 | **−67 % (3.0×)** |
| 1 | hash build: CUB radix sort + Warp kernels | 71.8 / 58.5 | 73.1 / 58.4 | 0 |
| 1 | speed guard | — | 1.6 / 1.5 | new |
| 1 | p2g | 101.1 / 97.7 | 101.2 / 97.8 | 0 |
| 1 | **all kernels** | **672 / 641** | **442 / 412** | **−34 % / −36 %** |
| 16 | particle-contact kernel | 9401 / 4557 | 4642 / 2275 | **−51 % / −50 % (2.0×)** |
| 16 | p2g | 3836 / 1392 | 3735 / 1389 | 0 |
| 16 | hash build | 190 / 160 | 193 / 162 | 0 |
| 16 | speed guard | — | 7.1 / 7.0 | new |
| 16 | **all kernels** | **15359 / 7996** | **10510 / 5721** | **−32 % / −28 %** |

At N=1 the frozen contact kernel is 52 % of kernel time (P1-0 eager split: 47–54 %), after C1+C2 26 %; p2g becomes the
largest kernel at N=16 (36 % of median time) — the analysis' next target (C7, Level C, separately versioned). The
CUB sort of the grouped build (8 kernels × 600) is 10–16 % at N=1, 1–2 % at N=16. Profile-mode control-step walls in
the same processes (nsys attached, separate processes, different minutes): N=1 0.32–0.35 s (frozen) vs 0.24–0.31 s
(alias-free); N=16 3.90 s vs 2.51 s.

### 3.3 Interleaved end-to-end A/B (same process, rotating order, no profiler)

`ab_n{1,4,8,16}_001`: replay from reset, steps 3–32 (N=1/4) or 3–22 (N=8/16) — the approach phase without
particle contacts (contact counts 0 in every variant), and full 1200-step episodes at N=1 below. One control step =
`env.step_measured_edge` (200-substep graph + coupling + strict int32 readback), device-synchronised.

__AB_TABLE__

**The kernel-level reduction does not show up in these walls.** All variants agree within ±2 % at every N
(alias-free 1–2 % slower; over the full s512 episode 0.2016 vs 0.1932 s median, s15 0.2026 vs 0.1988 s), while
nsys shows 230 µs (N=1) to 2.3–4.8 ms (N=16) less kernel time per substep, i.e. an expected 46 ms (N=1) to
0.45–0.95 s (N=16) less per control step. The N=1 wall (0.193–0.198 s) is also 1.45× the frozen kernel sum
(0.134 s per step). __PHASE3__

### 3.4 Full-episode replays (frozen evaluator, Native comparison)

`episode_512_001`, `episode_15_001`: N=1, alias_free_v1 and frozen_v1 interleaved per step in one process (1200
steps each). `episode_512_15_512_15_001`: N=4 (s512, s15, s512, s15), alias_free_v1 only. Drive: recorded
measured edge poses, resolved digits and measured pad boxes (`MeasuredTrajectoryDriver` → `step_measured_edge`),
strict mode, L2 monitor every step, grouped-query audits on the grid actually queried, 121 Native samples.

| run / env | seed | variant | final codes = Native | labelled digit 0 (Native) | evaluator 2 / 121 frames | L2 | guard | max position diff over time / final (P1-0 band over time) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_512_001 | 512 | alias_free_v1 | yes (0) | 37958 (37958) | PASS / PASS | PASS | 0 | 3.68e-5 / 3.68e-5 m (1.39–3.85e-5): inside |
| episode_512_001 | 512 | frozen_v1 | yes (0) | 37958 (37958) | PASS / PASS | PASS | — | 2.45e-5 / 2.45e-5 m: inside |
| episode_15_001 | 15 | alias_free_v1 | yes (0) | 85786 (85786) | PASS / PASS | PASS | 0 | 1.56e-3 / 5.85e-4 m (1.96–4.62e-4): **outside** |
| episode_15_001 | 15 | frozen_v1 | yes (0) | 85786 (85786) | PASS / PASS | PASS | — | 1.99e-2 / 1.99e-2 m: **outside** |
| N=4 env 0 | 512 | alias_free_v1 | yes (0) | 37958 | PASS / PASS | PASS | 0 | 3.85e-5 / 3.85e-5 m: inside |
| N=4 env 1 | 15 | alias_free_v1 | yes (0) | 85786 | PASS / PASS | PASS | 0 | 1.73e-4 / 7.9e-5 m: inside |
| N=4 env 2 | 512 | alias_free_v1 | yes (0) | 37958 | PASS / PASS | PASS | 0 | 2.31e-5 / 2.31e-5 m: inside |
| N=4 env 3 | 15 | alias_free_v1 | yes (0) | 85786 | PASS / PASS | PASS | 0 | 8.20e-4 / 1.80e-4 m: **outside** |

Final codes: 6/6 alias-free episodes (and 2/2 frozen controls) equal Native with 0 differing particles; evaluator
PASS on 2 and on 121 frames everywhere; L2 PASS (no digit rewrite, finite, min detF > 0, no slot overflow, no
invalid flag, grouped-query foreign = 0, mass bitwise constant); no guard violation (certified bound 150.5 m/s).
Position band: s512 inside in 3/3; **s15 outside the P1-0 single-env band in 2/3 alias-free envs and in the frozen_v1
control of the same process** (frozen 2 cm on a transiently mislabelled particle; P1-2 frozen batched `m0_s15_001`
had 6.5e-4 m over time, also outside). The s15 band is therefore not met by the unchanged batched path either; it
is reported, not attributed to alias_free_v1. Transient code differences (1–4 particles) occur in both variants —
s512 at t = 6.4 s (as in P1-0), s15 at t = 5.4 s (one alias-free run) and 9.6–10.4 s — none at the end. The per-step contact count `stats[4]` of alias-free equals frozen in 1169/1200
steps (s512) and 749/1200 (s15) of the interleaved runs, totals 152,465 vs 152,530 and 1,590,748 vs 1,365,930 —
the same divergence order as the positions (float-atomic chaos in a contact-rich seed), not a systematic offset
(same-state counts are identical, 3.1).

## 4. Rows (plan format)

__ROWS__

## 5. Rollback

`particle_contact_mode="frozen_v1"` (the default) — or omit the argument — restores the exact pre-P1-3a execution: the
same two statements in `BatchedTLMPMv7_2.step`, the byte-identical `_particle_contacts_b` (source sha256 pinned in the
tests), no extra arrays, kernels or graph nodes. `particle_contacts.py` is only imported (not executed) in that mode.
Removing the feature entirely = reverting the P1-3a commits; nothing else depends on it.

## 6. Failures and open items

- **FAIL, preserved:** `samestate_s512_k33_001`, `samestate_s15_k43_001` — the first fallback design rebuilt a
  `wp.HashGrid` inside the conditional node; graph instantiation failed (`Warp CUDA error 1: invalid argument`,
  `wp_cuda_graph_create_exec`). Probe: kernels, memsets and `radix_sort_pairs` work in a conditional body,
  `HashGrid.build` does not. Fixed by `FallbackGrid` (commit a18e8f5); the `_002` runs pass.
- **Performance not demonstrated end-to-end.** nsys shows the kernel gain (contact kernel 3.0× at N=1, 2.0× at N=16;
  all-kernel time −34 % / −28 %), but interleaved same-process walls on this shared GPU show ±2 %. __PHASE3_OPEN__
- **s15 L3 position band** exceeded by both alias-free and the frozen control (above); final codes, labels and the
  frozen evaluator are unaffected. Needs a re-derived band for the batched GPU path (≥ 3 runs per seed) — not a
  P1-3a change.
- Not run: `RUN_TO_RUN` deterministic comparison (P1-2 L4 is still pending); C3 (P1-3b); an EXCLUSIVE_GPU window.
- Sub-flag isolation timing exists only as walls (`ab_n1_001`, `ab_n16_001`: c1_only, c2_fallback_no_c1,
  c1_c2_unguarded all within ±1 % of frozen) and as nsys in __NSYS2__.

## 7. Commands

```bash
# CPU (see section 2); GPU jobs (each under the GPU lock, see the header of scripts/rl_ready/run_p1_3a.py)
PY=.deps/isaacsim-venv/bin/python R=scripts/rl_ready/run_p1_3a.py
$PY $R --mode samestate --seed 512 --sample 33 --copies 4 --gpu-index 0
$PY $R --mode samestate --seed 15 --sample 43 --copies 4 --gpu-index 0
$PY $R --mode ab --num-envs 1 --steps 30 --warmup 2 --kernel-every 10 \
    --variants frozen_v1,alias_free_v1,c1_only,c2_fallback_no_c1,c1_c2_unguarded --gpu-index 0
$PY $R --mode ab --num-envs 4 --steps 30 --warmup 2 --kernel-every 10 --variants frozen_v1,alias_free_v1 --gpu-index 0
$PY $R --mode ab --num-envs 8 --steps 20 --warmup 2 --kernel-every 10 --variants frozen_v1,alias_free_v1 --gpu-index 0
$PY $R --mode ab --num-envs 16 --steps 20 --warmup 2 --kernel-every 10 \
    --variants frozen_v1,alias_free_v1,c1_only,c2_fallback_no_c1,c1_c2_unguarded --gpu-index 0
$PY $R --mode episode --seeds 512 --variants alias_free_v1,frozen_v1 --gpu-index 0
$PY $R --mode episode --seeds 15 --variants alias_free_v1,frozen_v1 --gpu-index 0
$PY $R --mode episode --seeds 512,15,512,15 --variants alias_free_v1 --gpu-index 0
nsys profile -o artifacts/rl_ready/p1_3/nsys_001/n${N}_$V --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=stop -t cuda --sample=none --cpuctxsw=none $PY $R --mode profile --num-envs $N --variants $V \
    --steps 3 --warmup 3 --gpu-index 0          # N in 1 16, V in frozen_v1 alias_free_v1; then nsys stats
__PHASE3_CMDS__
PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_p1_3a_evidence.py ... --out docs/evidence/rl_ready/p1_3a_001.json
```

