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

__GPU_SECTIONS__
