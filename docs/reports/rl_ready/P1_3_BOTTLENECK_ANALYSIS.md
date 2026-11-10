# P1-3 — bottleneck analysis: particle–particle contact and the batched substep

Date: 2026-11-10. Branch `claude/rl-ready-rubber-physics` (HEAD at analysis `0b9e695`). Read-only analysis:
no source edited, no GPU job. Evidence: source reading (file:line below), the P1-0/P1-1 records, and a new
**CPU probe on the 242 recorded WP-17 Native states** (both seeds, all 121 samples; Warp 1.17.0 CPU, `.venv`).
Probe scripts and raw JSON: `artifacts/rl_ready/p1_3/cpu_contact_probe_001/` (`contact_probe.py` sha256
`c7ae196b…`, `contact_probe2.py` `04271d8f…`). Frozen `tlmpm_v5.py` sha256 `a7a05b2f…` (unchanged since
`be1d859`); batched `tlmpm_v7_2_batch.py` `da963a79…`.

| Item | Status |
| --- | --- |
| Q1 what `_particle_contacts` computes, candidate vs contributing counts | PASS (source + CPU probe on real states) |
| Q2 candidate changes, CPU bitwise against the frozen kernel on real states | PASS for C1, C2, C3 (242 + 12 states); others argued, NOT_RUN |
| Any GPU timing of a candidate | NOT_RUN (GPUs busy; every GPU gain below is an ESTIMATE) |
| Q3 batch picture N=8–16 | ESTIMATE only; existing N>1 timings are co-tenant-confounded |

## 1. What `_particle_contacts` computes

**Code path.** `tlmpm_v7_2_contacts.step` (`tlmpm_v7_2_contacts.py:189-193`) rebuilds `HashGrid(64,64,32)`
(`tlmpm_v5.py:798`) every substep with cell width `2.4·r` and launches the frozen kernel `tlmpm_v5.py:472-497`
over all n particles with `reference_exclusion = 3·dx` (`:852-856`). The batched copy is
`batched/tlmpm_v7_2_batch.py:294-324` (grouped query `:306`, build `:664-668`).

**Numbers for TRIM** (`material.json` of the WP-17 run; `material_config.py:7,16,17`): spacing 1 mm,
`r = 0.9·0.5·1 mm = 0.45 mm`, contact distance `2r = 0.90 mm`, query radius = cell width `h = 2.4r = 1.08 mm`,
exclusion 12 mm, `k_pc = 0.05`, μ = 0.3, dt = 0.1 ms.

**Per particle p** (`:481-497`):
1. `hash_grid_query(grid, x_p, h)` visits every point of the cells `floor((x_p ± h)/h)` per axis — always
   exactly **3×3×3 = 27 cells** — with **no distance filter** (Warp `native/hashgrid.h:205-212, 236-268`;
   order: cells x fastest, then y, then z; inside a cell, `point_ids` in stable-sorted order = ascending id).
2. Cells wrap modulo the dims (`hashgrid.h:77-106`): 64·1.08 = 69.1 mm in x, but the TRIM base is 159 mm long
   (local x −79.5…79.5 mm; the chip reaches +94 mm), so each physical cell holds ≈2.3 x-aliases.
3. For `q > p`: pass the filter if `_conflict_packed(code_p, code_q)` (opposite digits 1/2 of one cut,
   `tlmpm_v5.py:56-67`) **or** `|X_p − X_q| > 12 mm` (reference distance) (`:484`).
4. Passing pairs evaluate `d = |x_p − x_q|`, the penalty/Coulomb impulse `_contact_impulse(d − 2r, …)`
   (`:155-165`; non-zero iff `d − 2r + dt·min(v_rel·n, 0) < 0`), then **unconditionally** `atomic_add(impulse,p)`
   and `atomic_sub(impulse,q)` (`:492-493`); stats 4/5/6 only if `|j|² > 0` (`:494-497`).

**Why ~8.3M candidates.** 27 cells × 1.08³ mm³ × 1 particle/mm³ ≈ 34 true candidates per query, plus ≈1.4×
that from x-aliasing ⇒ ≈81 per query × 1.03e5 particles. The P1-1/P1-2 GPU audits counted 8.05–8.3M per env
(`artifacts/rl_ready/p1_2/smoke_m2_002/result.json`: 16.1M for 2 envs).

**CPU probe on the recorded states** (frozen kernel and a classifying copy on identical float32 inputs:
recorded positions converted to solver-local coordinates, recorded velocities and codes; counts per substep):

| per substep, 121 states each | seed 512 | seed 15 |
| --- | --- | --- |
| candidates visited | 8.01–8.36M | 8.09–8.43M |
| of which hash aliases (|Δcell|>1) | 58.3–58.9 % | 58.2–59.2 % |
| ordered pairs `q>p` (median) | 4.13M | 4.16M |
| pass the code/reference filter (median) | 2.46M | 2.49M |
| … of which aliases (far, zero impulse, but 2 vec3 atomics each) | ≥ 99.974 % | ≥ 99.946 % |
| non-aliased filter-passing pairs (max) | 644 | 1,295 |
| **pairs with non-zero impulse** (max per state; states with any) | **21** (5/121) | **90** (50/121) |
| particles receiving ≥ 2 non-zero contributions (max) | 6 | 30 |
| ordered pairs closer than R_cut ≈ 0.90–0.92 mm (max) | 66 | 611 |
| cells of the 27 that intersect the R_cut sphere (mean) | 15.1–15.4 | 15.1–15.8 |
| same-side pairs closer than 2r (excluded by the code rule) | ≤ 36 | ≤ 398 |
| max particle speed | 0.061 m/s | 0.122 m/s |

The classifier's non-zero count equals the frozen kernel's `stats[4]` in all 242 states.

**Why the cost is phase independent.** The work is set by particle density and hash geometry: ~8.3M candidate
visits, ~4.1M filter reads (`code[q]`, `reference[q]`) and ~2.46M full evaluations of aliased pairs 69–138 mm
apart, each ending in two zero-valued vec3 atomics (≈14.8M float atomics per substep). The physics (≤ 90
contributing pairs, about 1 per 10⁵ candidates) is noise in that budget. This matches the flat 386–405 µs in
all four P1-0 phases.

**Measurement gap to close first.** The P1-0 split used `cuda_filter = TIMING_KERNEL | TIMING_MEMSET`
(`profiling.py:268`). So the hash build kernels (`compute_point_keys`/`compute_cell_indices`,
`compute_cell_offsets`, launched as `WP_TIMING_KERNEL_BUILTIN`, `native/cuda_util.h:29-43`) and all memcpys
are missing from it. The CUB radix sort (`native/sort.cu:291`) has no Warp timing range, so `TIMING_ALL`
cannot see it either. The P1-0 graph wall minus kernel sum is ≈85 µs/substep (182 vs 165 ms per step,
SHARED). That gap is the upper bound for hash build + copies + graph gaps. WP-17 already listed hash
construction as unresolved. Use nsys (`run_vector_milestones.py --mode profile` with
`--cuda-graph-trace=node`).

## 2. Physics-preserving candidates, ranked

Parity levels used below:
- **A (bitwise)**: every non-zero float contribution reaches every destination in the same order. CPU
  `np.array_equal` holds against the frozen kernel. On GPU the result has the same distribution as the frozen
  kernel (its atomic order is unspecified anyway), and under Warp `RUN_TO_RUN` (sort by `(dest, thread)`,
  `warp/_src/deterministic.py:1-21`; the P1-2 L4 run is still pending) it should be GPU-bitwise against the
  same-mode frozen-batched kernel.
- **B (admissible reorder)**: same multiset of individually applied contributions, different order. CPU is not
  bitwise. The GPU result is still a possible execution of the frozen GPU kernel. Accept by a CPU
  contribution-multiset test plus the L3 band.
- **C (reassociation/approximation)**: rounding that no frozen execution order can produce. Needs a versioned
  id, its own profile and a deviation report. It is `reduced_fidelity` only if the physics model changes;
  none of the candidates below does.

Two facts carry the Level A arguments:
- **(i) A ±0 add is the identity on `impulse`.** `impulse` starts at +0 (`tlmpm_v7_2_contacts.py:30`, batched `tlmpm_v7_2_batch.py:150`) and
  only receives sums. In round-to-nearest, x + (±0) = x for every x ≠ −0, and a sum that starts at +0 can
  never reach −0. Skipping a zero contribution therefore changes no bit and no rounding of the others. NaN
  passes a `!= 0` test, so it still propagates.
- **(ii) Speed certificate.** A non-zero impulse needs `d < 2r + dt·|v_rel| ≤ 2r + 2·dt·v_max`. An aliased
  candidate is at least (D_min − 2)·h = 30 × 1.08 = 32.4 mm away (z dim 32), so it can only contribute when
  v_max > 157 m/s. For comparison, the frozen kernel itself misses closing speeds above about 1.8 m/s at the
  edge of its box, and the recorded maximum is 0.12 m/s. Production code needs a per-substep, per-env device
  `v_max`. That reduction must be guarded or segmented, as in the `_claim_slots` guard
  (`tlmpm_v7_2_batch.py:350-354`), or env-indexed atomics serialize. A violated certificate, or a non-finite
  `v_max`, uses the frozen radius and sets a sticky `invalid_reason` bit. Parity claims cover finite states
  only; a NaN env is invalid either way and may differ.

| # | Change | Level / argument | CPU probe | Expected gain (ESTIMATE) | Risk |
| --- | --- | --- | --- | --- | --- |
| **C1** | Skip the two impulse atomics when `ji` is exactly zero (component-wise `!= 0`, same idiom as v7_2 `_rigid_contacts`, `tlmpm_v7_2_contacts.py:139`). Optionally hoist `reference_exclusion**2.0` (same value). | **A**, unconditional, by fact (i); no certificate needed. A `length_sq` test would wrongly drop underflowing non-zero `ji`. | bitwise 242/242 | removes ≈14.8M float atomics per env-substep; −15…−40 % of `_particle_contacts` | very low |
| **C2** | Alias-free hash dims: a uniform per-batch dims from the packed reference bounding box plus a chip margin, e.g. `(192, 64, 48)` (TRIM needs ≥ 151–181 × 64 × 32–35). | **A** given the certificate (ii): the non-aliased candidate sequence is identical (same virtual cells in the same order, ascending ids per cell); only zero-contributing aliases disappear. | bitwise 242/242; forced heavy aliasing `(8,8,8)` also bitwise (12/12) | candidates −58 %, aliased evaluations −99.9 %; with C1 −55…−70 % of the kernel. Cell-array memsets 1 → ≈4.7 MB per substep, shared by all envs. Also shrinks the grouped binary-search ranges. | low |
| **C3** | Shrink the query radius to `min(2.4r, R_cut)` with `R_cut = 2r(1+1e-5) + 2·dt·v_max(1+1e-5) + 1e-7 m`, read from a device scalar per env (graph-safe). | **A** given (ii): the sub-box is contained in the frozen box (floor and float subtraction are monotone), and the candidates form a subsequence of the frozen order. The 1e-7 m margin exceeds float32 coordinate rounding (ulp(0.08 m) ≈ 7.5e-9 m). | bitwise 12/12 states with the most contacts (both dims). Visits: 2.21–2.28M (with C2) vs 5.4–5.6M (frozen dims) vs 8.0–8.4M | a further −33 % of candidates; with C2 −73 % in total | low; needs a v_max kernel (2 graph nodes/substep) |
| **C4** | Env-major custom hash for the batch, replacing Warp's grouped HashGrid: int32 key `e·C + cell` from the same float32 floor/`%` arithmetic, `wp.utils.radix_sort_pairs(..., end_bit=⌈log2 N·C⌉)` (stable LSD), scatter-clear of last substep's cells instead of an N·C memset. | **A** given (ii): same per-cell ascending-id order and z-y-x cell order as the frozen single, and no foreign env by construction. | not probed (needs the code) | removes 2×⌈log2(1.26N)⌉ dependent key loads per visited cell (≈270 per particle at N=16, vs 54 bound loads in the single) and the 64-bit 8-pass sort (≈3 passes instead) | medium: the Warp sort temp buffer is shared (P1-1 hazard), so reserve before capture |
| C5 | Visit only the cells whose box meets the R_cut sphere (custom query on C4) | A given (ii) (cell subsequence) | counted: 15.1–15.8 of 27 cells; 1.8M vs 2.25M candidates | a further −20 % after C3 | medium |
| C6 | Active set M = {code ≠ 0} ∪ {|x−X| > τ}, with τ = (12 mm − R_cut)/2 − ε; threads outside M query a second hash over M only. This is the provable form of (a)/(c). | A given (ii) (triangle inequality; a conflict needs two coded particles) | non-contributing pairs outside M: 100 % before the cut, 42 % (s512) and 6.9 % (s15) at the end | phase dependent; needs a second hash build | reject for now |
| C7 | P2G/MUSL atomics aggregated per node in the warp (≈8.2 distinct nodes per 32-thread warp, measured from the stencil) or privatized per block | **C** (pre-summed contributions are not a sequential order of the frozen kernel) | n/a | P2G −50…−70 %; matters only after C1–C3 | needs a versioned id, tolerance + L3 + deviation report |
| C8 | Fuse the 4 `wp.copy` and ~10 tiny per-env kernels into existing batched launches | A (exact copies, same op order) | n/a | ≤ ~5 % at N=1, → 0 at large N | low, low value |
| C9 | Finer cells (h/2, 125-cell box) | B (sub-cells interleave the id order; up to 30 particles have ≥ 2 contributions) | n/a | likely negative (more cell visits) | reject |
| C10 | Rebuild the hash less often (Verlet skin) | B (stale cells reorder candidates) + a larger box | n/a | ≤ hash-build share (unmeasured, est. ≤ 5 %) | reject |
| C11 | Sort particles by cell / Morton order | B (`ji(q,p) = −ji(p,q)` bitwise, but ownership and accumulation order change everywhere) | n/a | small: particles are already in z-fastest lattice order | reject |

Answers to the specific questions:
- **(a)** Same-side pairs within 12 mm never contribute, whatever their distance; the code/reference rule
  excludes them, not a bound on F. An F-based distance argument would even be false: up to 398 same-side pairs
  per state sit closer than 2r. Their remaining cost is the visit itself, and only C6 can avoid it.
- **(b)** Use dims and radius (C2, C3), not a finer cell (C9).
- **(c)** C6.
- **(d)** C10, rejected.
- **(e)** C4.
- **(f)** C11.
- **(g)** C7.
- **(h)** C8.

**Tests for C1–C3.** Extend `tests/test_batched_cpu_parity.py` / `test_batched_paths_parity.py` with the new
flags, bitwise against the frozen single after every substep. The current fixtures use 2.5 mm spacing; their
cell is 2.7 mm, so 64 cells span 173 mm and nothing aliases. Therefore:
1. Add a dims parameter, including forced `(8,8,8)` aliasing.
2. Add a full-resolution TRIM case.
3. Assert, as non-vacuity checks, that `stats[4] > 0`, that some particle receives ≥ 2 non-zero contributions
   (order sensitivity), and that the aliased candidate count is > 0.
4. Add a certificate test: an injected high-speed state must fall back to the frozen radius and set the bit.
5. Keep `artifacts/.../cpu_contact_probe_001` as the real-state evidence.

## 3. Batch picture at N=8–16 (ESTIMATES)

- **Linear in N, throughput-bound.**
  - P2G (7 float atomics × 8 corners per particle into ≤ 512 contributions per node; 122–139 µs at N=1).
  - MUSL and `_claim_slots`.
  - Grid-array memsets (≈4 %).
  - The grouped 64-bit sort (8 passes over N·1e5 keys; ≈0.25 ms/substep at N=16, est.).
  - These kernels gain little from batching, so their share grows.
- **Linear, latency-hidden by batching.**
  - The query and per-particle contact, label and frontier kernels; one env fills ≈0.39 of a wave (ADR 0021
    §12, UNVERIFIED for sm_120).
  - `_particle_contacts_b` itself grows by more than linear work. Each cell visit adds two binary searches over
    ≈2.9·N keys today, ≈1.26·N after C2.
- **Constant in N.**
  - About 10 tiny kernels per substep, graph-node gaps, and the shared hash cell arrays. Their share falls
    as 1/N.
- **Expected shares.**
  - Today (contacts dominate and get worse with N): contacts ≈ 45–55 %, P2G + MUSL ≈ 20–25 %.
  - After C1–C3: contacts ≈ 10–20 %, P2G + MUSL ≈ 30–40 % (the next target, C7 = Level C), then
    `_claim_slots` and the sort (C4).
  - Single-env substep after C1–C3: 0.83 → ≈0.55–0.65 ms (≈1.3–1.5× material throughput).
- **Existing evidence cannot settle any of this.**
  - `ab_timing_001` ran at 95–98 % co-tenant GPU utilisation. There, even the atomic-free `_deformation` grew
    13× from N=1 to N=4 (0.014 → 0.183 ms), so the "P2G/MUSL atomics grow with N" reading (0.098 → 0.817;
    0.046 → 0.505 ms) is confounded. The single's own `_particle_contacts` was not printed (top-14 sorted by
    N=4); derived from the totals, the grouped N=1 query is ≈ +15 % over the single.
  - P1-2 `m1_n4_001` (3.65 env-steps/s) and `m2_n1_001` (5.09) are both SHARED (foreign SM mean 46 % vs 24 %).
  - Re-measure on an EXCLUSIVE GPU.

## 4. Recommended P1-3 sequence and protocol

**First change, P1-3a.** In the batched material only, add a versioned opt-in
`particle_contact_execution="alias_free_v1"` (default stays `grouped_hashgrid_v1`). It combines:
- **C1**: zero-atomic skip;
- **C2**: alias-free uniform dims from the packed bounding box;
- the per-env device `v_max` certificate with a new sticky `invalid_reason` bit.

Each part has an independent sub-flag, so A/B rows can isolate it. Frozen modules (`tlmpm_v5`,
`tlmpm_v7_2*`, `LiveCuttingMaterial`) and the single-env/Native runtimes are not touched. Adopting this there
later would need its own versioned option, as WP-17 did.

**Second change, P1-3b.** C3 (query radius `min(2.4r, R_cut)` from the same `v_max`). Then re-profile. C4 only
if the grouped query or sort shows ≥ 10 % at N=8–16. C7 only as a separately versioned Level C mode if P2G
leads.

**Protocol (each change; reject or revert on any FAIL).**
1. CPU L0/L1: the full batched suite plus the new aliasing, multi-contribution and certificate tests,
   bitwise. `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -p no:cacheprovider tests/test_batched_*.py
   tests/test_wp17_frontier.py`.
2. GPU same-state A/B: on identical checkpoints (s512 sample 33, s15 sample 43), compare frozen-batched and
   variant kernels:
   - the set of particles with non-zero contact impulse and `stats[4]` must be identical;
   - values must agree to atomic-order rounding;
   - under `RUN_TO_RUN` (if L4 PASSes), they must be bitwise equal.
3. Kernel split before and after:
   - same GPU index and process;
   - interleaved A/B, ≥ 5 repeats, median and spread;
   - `TIMING_ALL` plus nsys `--cuda-graph-trace=node` (CUB sort visible);
   - the four P1-0 phases for seed 512 and the missing seed 15;
   - N = 1/4/8/16;
   - µs per env-substep per kernel and env-steps/s;
   - sampled co-tenant use with `EXCLUSIVE_GPU`/`SHARED_GPU`;
   - never merged across boundaries.
4. Full-episode replay: both seeds ≥ 3 runs, at N=1 and in a batch. All must hold:
   - final codes equal Native (P1-0: 0 particles, 4/4);
   - labelled counts are equal;
   - max position difference stays inside the P1-0 band (s512 1.4–3.9e-5 m, s15 2.0–4.6e-4 m);
   - the frozen evaluator gives PASS 3/3;
   - L2 invariants hold every step;
   - per-step contact count `stats[4]` stays inside the run-to-run band.
5. Receipts: execution id and sub-flags, dims, certificate status and violation count, source sha256.
   Rollback = the default flag; the old kernel stays byte-identical and callable.
6. Labels: a Level A change adds no `reduced_fidelity` reason. Workload A keeps its existing ideal-tracking
   label. A Level B or C change gets a new execution id, its own profile and a deviation report against the L3
   band. `reduced_fidelity` is used only if the physics model changes, which none of C1–C11 does.

## 5. Commands and limits

```bash
# CPU only, Warp 1.17.0 (.venv), recorded WP-17 Native states, no GPU
D=artifacts/rl_ready/p1_3/cpu_contact_probe_001
A=artifacts/wp17/native_optimized_001/seed_{512,15}/box/env/recording/attempt_000
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python $D/contact_probe.py $A $(seq -s, 0 120) probe_s{512,15}_all.json
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python $D/contact_probe2.py $A 0,33,34,35,36,120 probe2_s512.json $D   # s15: 0,43,48,49,50,120
```

Limits:
- **Inputs.** The recorded samples are 0.2 s apart, so the intermediate substeps were not classified.
  Positions come from float64 recordings round-tripped to float32 (exact at t=0). Each state was tested as
  one substep, with a random non-zero incoming impulse on 30 % of particles. That is not a full replay.
- **Timings.** CPU walls (frozen 47–49 ms; C2 ≈15 ms; C2+C3 ≈10–12 ms per substep) are informative only and
  are not GPU estimates.
- **Not determined.** The GPU time split inside `_particle_contacts` (iteration vs aliased evaluation vs
  atomics) is unknown, so every GPU gain above is an estimate to be replaced by the §4 measurement.
