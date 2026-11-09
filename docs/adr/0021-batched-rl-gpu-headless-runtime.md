# ADR 0021 — Batched RL runtime `rl_gpu_headless`: batched `tlmpm_v7_2` material and vector env API

Date: 2026-11-10. **Status: accepted with amendments (orchestrator, 2026-11-10).**

**Amendments at acceptance.**
1. *Packages:* physics batching (`layout`, `tlmpm_v7_2_batch`, `constitutive`, `pose_interp`, `material_batch`,
   `errors`) lives in `robotic_cutting_sim.batched`; the RL-facing API (`vector_env`, `reward_packed`, benchmark rows,
   telemetry, replay) lives in `robotic_cutting_sim.rl`, which exports `open_vector_env` (plan decision D-04).
2. *Runtime venv:* workload A, the vector env and all Torch tests run in `.deps/isaacsim-venv` (same Python, Warp,
   NumPy 2.3.1 and Torch as the WP-17 Native runs; plan D-08). `.deps/isaaclab-venv` is used only for workload B.
   Single-env and batched rows that are compared must use the same venv.
3. *Numbering:* robot batching (workload B, §9) is work package **P1-1b** in `RL_READY_PLAN.md`; P1-3 keeps the
   specification's meaning (headless/kernel/data-movement optimisation). It starts after M2 of workload A.

Original status line: **proposed.** It is the binding design for work packages P1-1 (true GPU batched physics)
and P1-2 (vectorized env API and partial reset). Acceptance needs the M0–M2 evidence in §13.

Nothing here changes a frozen solver, profile, threshold, evaluator or record. v5, v6, v7, v7_1 and v7_2 (including
`tlmpm_v7_2_contacts.py` and `tlmpm_v7_2_frontier.py`), law 8, the v1 evaluator, evaluation v2, final_v4 and the
WP-11..WP-17 evidence stay byte-identical. Everything new is versioned and opt-in. Material parameters remain
provisional; results are `SIMULATOR_ONLY_UNCALIBRATED`. v7_2 keeps its `EXPERIMENTAL_NOT_ACCEPTED` status
(`tlmpm_v7.py:18`). Batching changes how v7_2 executes, not its physics.

## 1. Context (verified facts)

- **Solver step.** `tlmpm_v7_2.TLMPM.step` (`tlmpm_v7_2.py:122-139`) runs the helper step `tlmpm_v7_2_contacts.step`
  (`:149-224`, the v5/v7 launch order), then the hoisted frontier kernel (`tlmpm_v7_2_frontier.py:27-120`).
- **Tool and stroke state is scalar** (size 1 or per blade segment):
  - `index`, `stroke`, `stroke_travel`, `previous_cut`, `reaction`, `stats[7]` (`tlmpm_v5.py:756-797`);
  - `previous_pose`, `contact_totals[12]` (`tlmpm_v7.py:250-255`);
  - `link_previous_pose`, `link_diagnostics[2]` (`tlmpm_v7_1.py:161-169`).
- **Fixed geometry.** The stencil (`nodes/weights/gradients`) and the link CSR are built once from the reference
  configuration and never mutated (`tlmpm_v5.py:734-783`, `tlmpm_v7_1.py:152-163`).
- **Material coupling.** `LiveCuttingMaterial` captures 200 `step()` calls into one CUDA graph
  (`live_material.py:289-297`). Each control step it:
  1. interpolates poses on the host with SciPy (`coupled_robot.py:105-114`);
  2. calls `assign` for poses, twists and cuts (`live_material.py:336-343`);
  3. launches the graph, synchronizes and runs `check_slots()`;
  4. reads `reaction.numpy()` (`:353-359`);
  5. returns the wrench applied in the *next* 20 ms interval (`:372-374`).
- **Reset rebuilds everything.** It constructs a new solver and recaptures the graph (`kinematic_env.py:123-126`,
  `live_material.py:209-297`).
- **TRIM scene.** Base 160×32×20 mm with 1–2 protrusions; `tlmpm_v7_2`, law 8, K=2; 1 mm particles, 4 mm grid
  (`seeded_trim.py:60-68`). Seeds 512/15 have 102,732 / 103,898 particles (WP17 report :101).
- **Obstacle boxes differ by runtime.** Kinematic and configured tasks get 1 obstacle box, the handle
  (`live_material.py:61-65`; `ConfiguredTaskSpec` is a `ProfileTaskSpec`, `scene_config.py:617`). Native has 3:
  the handle plus two measured pads (`isaac_sim_runner.py:1203-1207`).
- **Memory probe (this ADR; CPU only, Warp 1.17.0).**
  - Command: `CUDA_VISIBLE_DEVICES='' .venv/bin/python`. It built `LiveCuttingMaterial(solver="tlmpm_v7_2",
    device="cpu")` for `compile_task(generate_scene(s), goal_config(), seed=42)` and read `bed.memory_bytes()`.
  - 7,020 grid nodes (45×13×12), S=8, 120 edge segments.
  - **48.80 MB (seed 512) / 49.32 MB (seed 15)** per env, excluding the HashGrid.
  - CPU construction takes 0.09–0.16 s.
- **Measured single-env cost (Native, WP17).**
  - Material graph: 154.6 / 164.8 s per 1200 actions = **128.8 / 137.3 ms per control step**, about 0.65 ms per
    substep (WP17 :40).
  - Particle hash and contact: about 60 ms per 200 substeps (eager, :58-60).
  - The active-frontier cost is state dependent.
- **GPU runs are not bitwise reproducible.** Reaction sums differ by about 2e-10 from atomic order
  (WP16 :128-129). Seed-512 final codes differ, 33,692 vs 37,958 (WP17 :105-112).
- **API base.**
  - Actions are normalized edge deltas: `[-1,1]^6` scaled by `ActionLimits` (`control.py:142-148`, `gym_env.py`).
  - Deployable and privileged observations are separated (`policy_env.py:56-86`).
  - `cutting_reward_v1` rejects packed codes with K≥2 (`reward.py:32-39`), so TRIM has no usable batched reward.
- **Environments.**
  - The RCS `.venv` has **no torch**.
  - `.deps/isaaclab-venv` has torch 2.11+cu128, Warp 1.17.0, Newton 1.6.1, MuJoCo-Warp 3.12.0 and NumPy 2.5.2.
  - `.deps` is a symlink into the main checkout and is treated as read-only.

## 2. Decision summary

1. **One process, batched kernels.** One process runs batched Warp kernels over N independent envs. Rejected: a
   Python loop of single envs, N Kit processes, and N graphs. New package: `src/robotic_cutting_sim/batched/`,
   holding the execution variant **`tlmpm_v7_2_batch`** and runtime **`rl_gpu_headless`**.
2. **Layout.**
   - Flat structure-of-arrays concatenation with exact per-env counts (no padding), `env_of_particle` and offsets.
   - Global node ids are baked in at pack time, and per-env link CSRs are concatenated.
   - One **grouped Warp HashGrid** (group = env) serves all envs.
   - Small per-env state is `[N,…]`.
3. **Pack from frozen references.** Each env's initial state comes from the frozen single solver built on CPU, so it
   is identical by construction (§3.4).
4. **Parity.**
   - CPU parity is bitwise for N=1 and for every env of N, each against its own single solver.
   - GPU parity is structural invariants plus outcome bands.
   - Bitwise GPU checks are only an optional deterministic verification (§5).
5. **Graph and reset.** One graph per batch shape, with fixed device input arrays. Masked partial reset: no
   reallocation, no recapture. Changing geometry needs an explicit `rebuild()` (§6).
6. **Coupling.** The 20 ms delayed wrench handoff stays on the device as `[N,6]` tensors, with no per-env host sync
   (§7).
7. **Route.** Workload **A** first: batched ideal tracking with real batched cutting, P1-1/P1-2. Robot batching on
   Isaac Lab Newton N worlds is workload **B**, P1-3, under its own ADR (§9).
8. **Fidelity and unsupported cases.** Fidelity differences from the Native TRIM reference are computed into
   `receipt.fidelity.reduced_fidelity` with reasons. Everything else raises `UnsupportedBatchConfiguration` (§10).

## 3. Material batch layout

### 3.1 Per-particle arrays (flat, P = Σ nₑ)

- **Arrays carried over.** `x, v, reference, mass, code, F, Fpi, stress, plastic, progress, contact_work, impulse,
  nodes/weights/gradients/slot [P,8], x_start, front_crossings, front_reached, link_velocity_start, link_codes,
  pending_q, pending_cut`.
- **New.** `env_of_particle[P]` (int32), which also serves as the HashGrid group array.
- **Offsets.** `particle_offset[N+1]` and `node_offset[N+1]`.
- **Nodes.** `nodes[p,c]` stores **global** node ids (`local + node_offset[e]`).
- **Link CSR.**
  - `link_offsets` concatenates each env's CSR, shifted by `node_offset[e]`.
  - `link_neighbours` holds global particle ids.
  - Each env is built with the frozen stable argsort (`tlmpm_v7_1.py:154-160`), so the neighbour order equals the
    single-env order shifted by a constant.
- **Launch sizes.** Variable nₑ and per-env node counts cost nothing extra. Launch dims are `P`, `(P,8)` and
  `(S, nodes_total)`.
- **int32 guards** (checked at pack time): `P·8 < 2³¹` and `nodes_total·S < 2³¹`. At about 1e5 particles per env
  this allows N ≈ 2,600.

### 3.2 Per-env state (`[N,…]`, flattened where a frozen kernel needs 1-D)

| State | Single solver | Batched |
|---|---|---|
| per-substep tool pose / twist / cut digit | `[T]`, T=substeps+1 (`tlmpm_v5.py:803-814`) | `[N,T]`, written in place each control step |
| substep index | `index[1]` | **shared `index[1]`** (lockstep; dt and substeps uniform) |
| stroke, travel, engaged, any_motion, previous_cut | `[1]` | `[N]` |
| segment_ok, segment_plow, edge_flags | `[segments]` | `[N·segments]` |
| previous_pose, link_previous_pose, reaction | `[1]` | `[N]` |
| box_centers/halves, box_reactions (handle + pads) | `[B]` | `[N·B]`, B uniform |
| fixture boxes, table plane | Python tuple | `fixture_offset[N+1]`, flat boxes, `plane[N]` |
| stats(7), contact_totals(12, f64), link_diagnostics(2) | `[k]` | `[N·k]`; sum and max families in separate arrays (§5 L4) |
| slot_overflow, slot_max_occupied | `[1]`, cumulative since construction | `[N]`, cleared by reset |
| λ, μ, σ_y, friction, contact/particle stiffness, G_c, sharpness, slice efficiency | launch scalars | `params[N,k]` float32, computed per env in float64 exactly as the single path, then cast |
| clock, seed/RNG, pending wrench, last measured pose, `T_world_local` | Python scalars | `step_count[N]` int, `seed[N]` + counter-based Philox (seedₑ, step, stream), `pending_wrench[N,6]`, `last_pose[N,7]`, `T[N]` |

**Uniform per batch** (kept as kernel scalars, checked at pack time): dt, dx and particle spacing (which set volume, radius and hash cell); substeps, S, K and law 8; blade and tool profile, and B; `material_contact` and `pic`; workpiece-local gravity, because the unchanged `_p2g` takes it as a scalar; `floor=False`.

### 3.3 Kernel plan per substep (production path: optimized contacts and frontier)

Thread counts are for N=1 TRIM (n≈1.03e5; S·nodes = 56,160). Labels:
- **UNCHANGED**: the frozen kernel, launched on batched arrays.
- **ENV**: a new kernel whose body is the frozen body with env-indexed loads and stores. It calls the same frozen
  `@wp.func` helpers: `_profile_sdf, _box_sdf, V._contact_impulse, _swept_side, _conflict_packed, _edge_segment,
  _interface, _endpoint_work, _interface_from_p`.
- **RED**: a per-env reduction.

| # | Operation (source) | Threads | Batched |
|---|---|---|---|
| 1–3 | copy link_previous_pose, link_velocity_start, x_start (`tlmpm_v7_2.py:124-125`, `contacts:157`) | 1, n, n | unchanged copies |
| 4 | `v5._start_cut` (`contacts:158`) | n | ENV: `cuts[e,i]`, `previous_cut[e]` |
| 5 | `v5._remember_cut` (`:159`) | 1 | ENV, N threads |
| 6 | `v5._stress` (`:160`; `tlmpm_v5.py:77-99`) | n | ENV: per-env λ, μ, σ_y. **Constitutive extension point** (§11) |
| 7 | zero edge_flags, engaged, any_motion (`:161-162`) | 120/1/1 | unchanged ops on `[N·…]` |
| 8–9 | `v5._segment_motion`, `_segment_plow` (`:163,168`; `tlmpm_v5.py:346-395`) | 120 | ENV, launched (N,120); RED any_motion |
| 10 | `H._rigid_contacts` (`contacts:16-145`) | n | ENV; RED reaction, stats, totals, engaged |
| 11 | `v5._obstacle_contacts` × B (`:181-184`; `tlmpm_v5.py:298-327`) | n each | ENV; RED reaction, box_reactions. **Keep one launch per box**: fusing would reorder the `reaction` float sum |
| 12 | `V.fixture_contacts` (`:187`; `tlmpm_v7.py:170-202`) | n | ENV: per-env box range and plane; RED stats, totals |
| 13 | `hash.build` (`:190`) | n + sort | one **grouped** build, `groups=env_of_particle` |
| 14 | `v5._particle_contacts` (`:191`; `tlmpm_v5.py:472-497`) | n | ENV: `hash_grid_query(grid, x, r, e)`; `q>p` on global ids unchanged; RED stats |
| 15 | `slot_code.fill_(-1)` (`:194`) | 56k | unchanged |
| 16 | `v5._claim_slots` (`:195`; `tlmpm_v5.py:442-469`) | 8n | ENV only for the integer overflow/max counters; indexing unchanged |
| 17–25 | zero gm/gp/gj; `_p2g`; `_grid_update`; `_g2p`; zero gp/gj; `_musl`; `_grid_update`; `_deformation`; `v6._grid_positions` (`:197-214`) | 8n / 56k / n | **UNCHANGED kernels** (global node ids; uniform volume, dt, gravity, pic) |
| 26 | `v5._update_stroke` (`:216`) | 1 | ENV, N threads |
| 27 | `V._assign_front` (`:218`; `tlmpm_v7.py:204-235`) | n | ENV: `previous_pose[e]`, `poses[e,i]`, `stroke[e]`, `seg_ok[e]` |
| 28 | `V._remember_pose` (`:222`) | 1 | ENV, N threads |
| 29 | copy link_codes (`tlmpm_v7_2.py:130`) | n | unchanged |
| 30 | `F._complete_links` (`tlmpm_v7_2.py:131-138`) | n | ENV: tool, stroke, seg_ok, diagnostics per env; the CSR is same-env by construction |
| 31 | `v5._advance` (`:139`) | 1 | unchanged (shared index) |

Each substep has about 31 operations, plus B−1 obstacle launches, plus hash-build internals. That is about 8–9k
graph nodes per control step, as today and **independent of N**.

Operations outside the substep graph, once per control step: pose interpolation and cut-digit fill; zeroing `reaction`, `box_reactions` and `stats`; the per-env diagnostics reduction; wrench, reward and observation kernels.

The frozen `material_diagnostics._state_partials` strides its groups over *all* particles
(`material_diagnostics.py:22`), so its partial reductions would mix envs. A segmented per-env variant is required.

### 3.4 Packing

`BatchLayout.pack(tasks, material)`:
1. Builds each env's frozen single solver on the **CPU device** through the frozen `LiveCuttingMaterial.reset` path,
   with no stepping.
2. Copies its arrays into the device batch.
3. Records each env's `parameter_receipt` and `solver_receipt`.

Geometry, masses, stencil, CSR, derived material config and initial pose are therefore frozen-identical by
construction. Batched code never re-derives them.

## 4. Isolation of cross-particle operations

| Path | Leak if naive | Guarantee and audit |
|---|---|---|
| P2G/G2P/MUSL/F/x via node ids (`tlmpm_v5.py:500-613`) | none if node ranges are disjoint | Global node offsets. Audit: `nodes[p,c] ∈ [node_offset[e], node_offset[e+1])` |
| Particle–particle contact: `HashGrid(64,64,32)` (`tlmpm_v5.py:798`), distance test (`:481-490`) | **Real cross-env impulses**: envs share local coordinates | Grouped HashGrid. Warp documents groups for "independent environments or worlds … even when their coordinates overlap", valid in CPU/CUDA graph replay (`warp/_src/types.py:7955-7975`; `native/hashgrid.h:180,226`) |
| Support-link CSR | none if per env | Concatenated per-env CSR. Audit: each neighbour is in its own env's range |
| Atomics on `[1]` arrays (reaction, stats, totals, flags, counters) | sums across envs | `[N·k]` targets indexed by env. A static test rejects any env-state `[0]` indexing other than `index[0]` |
| Diagnostics reduction | strided groups mix envs | segmented per-env reduction |
| `pending_q` (a particle id) | none | global id inside the env's range; cleared on reset |

**Rejected: isolation by spatial offsets.**
- **Physical offsets** change the float32 rounding of every SDF, contact and interpolation, which breaks parity.
- **Hash-coordinate offsets** are not enough. Warp cells wrap modulo the grid dims (`hashgrid.h:77-106`), and a
  160 mm workpiece already exceeds the 69 mm cell domain in x. Aliased particles of another env at the same local
  position would still collide unless kernels filter by env.
- **Newton is not a model here.** Its XPBD and SemiImplicit particle contacts do not filter worlds; they rely on
  spatial separation (`newton/_src/solvers/xpbd/kernels.py:278-289`). Yet Newton recommends keeping worlds at the
  origin (`newton/_src/sim/builder.py:3150-3153`).

## 5. Parity strategy

### L0 — CPU bitwise against the frozen solver (required, M0/M1)

**Setup.**
- The batched solver runs on Warp `cpu` against `tlmpm_v7_2.TLMPM` with default optimizations.
- Inputs are identical: per-substep pose and twist arrays from the same `interpolate_measured_poses` output, the same
  cut digits and the same pad boxes.
- After every control step, every array of §3.1–3.2 is compared with `np.array_equal`. Global ids and the
  stats/totals layout are mapped back to single-env form first.

**Cases.**
- N=1.
- N=3 heterogeneous (geometry, particle count and material parameters differ), each env against its own single
  solver.
- N identical copies, which must be mutually identical.
- The WP17 positive-completion and pending fixtures (`tests/test_wp17_frontier.py`), so active link completion is
  covered.
- One full-size TRIM substep as a smoke test.

**Why bitwise identity is achievable (verified).**
- **CPU kernels are serial loops in task order** (`warp/_src/codegen.py:7018-7020`; "CPU kernels are sequential",
  `native/deterministic.h:75`).
- **Float atomics keep their order.** Every float atomic destination belongs to one env, and that env's particles
  are contiguous and in the same order. So each destination receives the same contributions in the same order: p2g
  slots, contact `impulse`, `reaction`, `stats`, `totals`.
- **Slot assignment is identical.** The `_claim_slots` CAS also runs serially.
- **The grouped HashGrid keeps neighbour order.** It sorts `(cell, group)` keys with the stable LSD radix sort
  (`native/hashgrid.cpp:170-178`).
- **Probe result: PASS.** An ad-hoc CPU probe (Warp 1.17.0, not retained) compared grouped against per-env HashGrids.
  It used three overlapping envs of 7.7k–8.4k jittered lattice points, 160 mm long so x wraps the 64-cell grid, with
  the solver's cell width and dims. Full candidate lists of up to 140 per query were identical in content and order.
- **No FMA contraction on CPU.** The CPU target is `generic` unless `-march=native` is set
  (`native/clang/clang.cpp:591-607`).

Both runs must use the same venv, Warp and NumPy. An L0 failure is a defect to fix, **never** a reason to switch to
a tolerance.

### L1 — CPU isolation and reset (bitwise, required)

- **Perturbation.** Perturbing env k's actions leaves every other env bitwise unchanged.
- **Partial reset.** Resetting env k at step s makes it bitwise equal to a fresh run. The other envs stay bitwise
  equal to the no-reset run.
- **Full reset.** Resetting all envs gives a batch bitwise equal to a freshly packed one.
- **Coverage.** The test enumerates `vars()` of the batched objects, so any array the reset misses fails it.

### L2 — GPU structural invariants (every GPU run)

- Mass is exactly constant.
- Codes are monotone per digit, with no rewrites.
- The §4 range audits pass.
- State is finite and `slot_overflow = 0`, both per env.
- An instrumented grouped-query kernel confirms every returned neighbour belongs to the querying env.

### L3 — GPU outcome parity (M0/M2, tolerance-based)

- **Replay source.** WP17 Native recordings hold the measured edge pose and cut digit for every step
  (`actions.jsonl` → `privileged_diagnostics.measured_tool_pose` and `cut_index`), plus 23 body poses
  (`robot_state.npz`). Location: `native_optimized_001/seed_{512,15}/…/recording/attempt_000/`.
- **Band.** Replay them into the single v7_2 material on GPU at least 3 times. That defines the run-to-run band of
  removed target fraction, substrate share, components, labelled count and maximum position difference.
- **Check.** Batched N=1 and N=4 copies must fall within the band and pass the frozen evaluator's thresholds.
- **Pad boxes.** Rebuilding them from recorded finger poses is **UNVERIFIED**. Until it is verified, replays use the
  handle box only and carry that label.
- **Native outcomes** are compared descriptively, never bitwise.

### L4 — optional GPU deterministic verification

- Warp 1.17 offers `DeterministicMode.RUN_TO_RUN` (`warp/config.py:460-475`).
- The frozen kernels cannot use it. They mix `atomic_add` and `atomic_max` on `stats` and `totals` inside one kernel,
  which deterministic mode rejects (`warp/_src/deterministic.py:761-767`).
- The batched kernels therefore keep sum and max families in separate arrays. A verification build can then make the
  GPU N-copy and perturbation tests bitwise.
- **UNVERIFIED:** support for spatial_vector and float64 atomics in hash-query kernels. A GPU probe in M1 settles it.
- This mode is never used for production timing.

## 6. CUDA graph and partial reset

### Graph

- Capture the 200 substeps once per batch shape (N, P, nodes, B, flags, device).
- Build the grouped HashGrid once **before** capture so its 64-bit key buffers are reserved. Group values may change
  during replay.
- Inputs are written in place into the captured arrays: `poses/twists[N,T]` by the interpolation kernel,
  `cuts[N,T]` from actions, `box_*` from pad geometry.
- `index`, `reaction`, `box_reactions` and `stats` are zeroed before each launch.
- Changing N, P or any uniform parameter requires `rebuild()`, which repacks and recaptures.

### Partial reset: `reset(env_ids)`

Masked kernels run on the same stream between control steps, driven by a device mask `[N]`. There is no allocation
and no recapture.

- **Particle kernel** (only where `mask[env_of_particle[p]]`): `x = x_start = reference`; `v = impulse = stress = link_velocity_start = 0`; `F = Fpi = I`; `plastic = progress = contact_work = code = link_codes = front_* = 0`; `pending_q = pending_cut = -1`; every registered internal variable set to its declared initial value (§11).
- **Env kernel:**
  - `previous_cut = -1`;
  - stroke, travel, engaged, any_motion, segment and edge arrays, reaction, box reactions, stats, totals, link
    diagnostics and slot counters all set to 0;
  - pose, twist and cut rows set to the initial tile (pose, zero twist, −1), as in `live_material.py:276-277`;
  - `previous_pose` and `link_previous_pose` set to the initial local edge pose;
  - initial pad boxes restored;
  - `pending_wrench = 0`, `step_count = 0`, `last_pose` = initial measured pose;
  - seed and RNG counter taken from `seeds`.
- **Not reset**, because every substep recomputes them: grid arrays, `slot`, `slot_code`, `old`, `new_v` and the hash
  grid.
- **No initial-state copy is needed today**: the initial state is constants plus `reference`. An initial-value
  registry still allows a device copy for future models.

### Geometry change

- Each env slot keeps its scene (`scene_seeds`) for the batch lifetime.
- `reset(env_ids, scene_seeds=…)` with a different geometry raises
  `UnsupportedBatchConfiguration("UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET")`.
- `rebuild(scene_seeds)` is explicit and resets every env.
- Capacity slots with inactive particles would need an active mask in every kernel, which is a parity risk. They are
  deferred to a separate ADR, even though TRIM particle counts vary by only about 1%.

## 7. Coupling and the 20 ms delayed wrench

1. **Pose interpolation.** `pose_interp` is a Warp kernel that computes in float64 and casts to float32. It
   reproduces `interpolate_measured_poses`: linear xyz and SLERP at `k/substeps`; constant twist `(Δp/dt, rotvec(R₁R₀⁻¹)/dt)`; transform with per-env `T_local_world[e]`; inputs `last_pose[N,7]` and `measured_pose[N,7]` on device.

   A component test against SciPy (sign flips, near-zero and large rotations) requires ≤1 float32 ulp after the
   cast. Receipts record `pose_interpolation: device_f64_slerp_v1`. L0 feeds the SciPy arrays directly.
2. **Graph launch.**
3. **Wrench conversion.** A kernel converts `reaction[N]` and `box_reactions[N·B]` into world material wrenches
   `[N,6]` and pad wrenches `[N,2,6]`, as in `live_material.py:359-372`: rotate at the same point, subtract the
   recipients' box impulses, divide by `control_dt`. The result becomes the next interval's `pending_wrench`; the
   previous value stays as `applied_previous`. Torch reads zero-copy views (`wp.to_torch`).
4. **Validity.**
   - Nonfinite state and slot overflow are detected on the device, giving `invalid_reason[N]`.
   - **Strict mode** (the default for acceptance runs) reads one int32 count per step and raises with the env ids.
   - **Training mode** terminates the affected envs with their reason. They never continue silently and never count
     as successes.

**Workload A.** The wrench is privileged info only (no robot), as in `kinematic_env`.

**Workload B.** The wrench is written each rigid substep in a graph-safe force callback, extending the frozen
`_edge_wrench` / `apply_tool_wrench` / `_pad_wrenches` patterns (`isaac_lab_env.py:388-404`) to `[N]` with per-world
body-index arrays.

**Streams.** All Warp work runs on Torch's current stream (`wp.stream_from_torch`) or uses explicit events.
Unsynchronized mixing is a defect.

## 8. Vector env API (P1-2)

```python
with open_vector_env(num_envs=N, runtime="rl_gpu_headless", task=TaskFamily("agentic_trim_id", scene_seeds=[...]),
                     material="scene_declared", observation_profile="state", device="cuda",
                     backend="kinematic_ideal_tracking_batched", action_spec="edge_delta_v1",
                     reward="trim_bond_progress_packed_v1", autoreset=False, output_dir=...) as env:
    obs, info = env.reset(seeds=seeds_int64_N)
    obs, reward, terminated, truncated, info = env.step(actions)      # torch.float32 [N, 8] on cuda
    obs, info = env.reset(env_ids=ids, seeds=seeds_for_ids)            # others bit-identical (L1)
    results = env.evaluate(env_ids)                                    # frozen evaluator, episode end, host
```

### Returns and runtime

- All returns are device Torch tensors: `obs["policy"] [N,D]`; `reward [N]` float32; `terminated` and `truncated` `[N]` bool.
  - `info` as a dict of `[N]` tensors (step count, `action_status`, `invalid_reason`, reward terms).
- `autoreset=False` is the default. `"same_step"` is opt-in Isaac Lab semantics with `info["final_obs"]`.
- The API needs torch, so it runs in `.deps/isaaclab-venv`.
- The Warp and NumPy core is tested on CPU in `.venv`. A torch test skipped there counts as NOT_RUN, not PASS.

### Actions (`edge_delta_v1`, A=8)

- `a[0:6]`: normalized edge-frame delta. It is a device port of `normalized_delta`, `resolve_delta(frame="edge")`
  and the `bound_target` workspace clamp (`control.py:92-148`), tested against NumPy within float32 tolerance.
- `a[6]`: arm flag (> 0 arms).
- `a[7]`: cut-digit selector mapped to `{0..K−1}`.
- Statuses reuse `policy_env.ACTION_STATUSES`.

### Observations

- **`state` (deployable)** contains only the commanded edge pose (equal to the measured pose under ideal tracking),
  time remaining, previous action, arm flag and step fraction.
- It contains no material state, codes, components, contact truth, wrench, force or generator truth
  (`PRIVILEGED_KEYS`, `policy_env.py:57-63`).
- **Consequence.** TRIM hides the bump count (`seeded_trim.py:71-72`), so a state-only policy cannot see the
  geometry. Geometry-aware deployable observations need a separate camera or height-map capability.
- **`privileged_state`** adds `obs["critic"]` (oracle, explicit opt-in). It never enters `"policy"`.

### Reward

- `trim_bond_progress_packed_v1` is new, with status `PROVISIONAL_SHAPING_NOT_SUCCESS`. It reads privileged codes.
- **Progress** is the Δ fraction of target-plane bond columns severed, judged by `_conflict_packed` on
  reference-lattice bonds that cross the scoring plane inside the protrusion footprints.
- **Penalties:** off-target severing, time, and invalid termination.
- Bonds are a flat per-env CSR built at pack time; the reduction is segmented.

### Termination and success

- `truncated` when `step_count = duration/control_dt` (1200 for TRIM).
- `terminated` on nonfinite state or slot overflow, each with a reason code. Workload B adds grasp loss and IK-failure
  streaks.
- **Success** comes only from the frozen independent evaluator, run on a host snapshot of the selected envs at
  episode end (about 3 MB per env). It never runs inside `step()`.

### Receipt

- Records versions, frozen-source SHA256s, the layout, per-env task and material receipts, uniform parameters, the
  action/reward/observation versions and `pose_interpolation`.
- `fidelity` is computed against `native_trim_v7_2_reference`, not hand-set.
- **Workload A:** `reduced_fidelity=true`, with reasons `ideal_edge_tracking_no_robot_dynamics`,
  `no_finger_pad_obstacles` (1 box vs 3) and `reaction_not_fed_back`.
- Material resolution (1 mm / 4 mm / 0.1 ms / 200 substeps, S=8, K=2) is unchanged. Any change to it sets
  `reduced_fidelity`.

## 9. Robot batching (workload B): facts, route, order

### Upstream facts (read from source)

- **One Newton model holds N worlds.** `replicate_builder_mapping` uses `begin_world` / `add_builder` / `end_world` or
  `builder.replicate`
  (`.deps/IsaacLab/source/isaaclab_newton/isaaclab_newton/cloner/newton_clone_utils.py:383-401`).
- **One MuJoCo-Warp step covers all worlds.** `separate_worlds` defaults on when `world_count > 1`
  (`newton/_src/solvers/mujoco/solver_mujoco.py:4021-4022`).
- **Cross-world collisions are filtered** (`newton/_src/geometry/broad_phase_common.py:256-282`).
- **Wrench composers** hold `[num_envs, num_bodies]` vec3 arrays (`isaaclab/utils/wrench_composer.py:80-88`).
- **Body poses** come back as zero-copy `ProxyArray.torch [N,B,7]`, quaternions XYZW.
- **Auto-reset syncs with the host.** `DirectRLEnv` uses `nonzero()` for auto-reset. This can be overridden when
  `compute_final_obs` is off (`isaaclab/envs/direct_rl_env.py:420-428, 660-682`).

### No upstream batched TLMPM exists

Neither Isaac Lab nor Newton has TLMPM, a total-Lagrangian reference configuration, cut/fracture state or per-node
multi-field slots:
- Newton's only MPM, `SolverImplicitMPM`, is implicit and updated-Lagrangian, with Drucker–Prager plasticity on a
  NanoVDB grid. Per-world isolation is opt-in and off by default
  (`newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:928-942`).
- PhysX deformables are FEM, with no cut API.

The batched Warp TLMPM adapter this ADR specifies is therefore required.

### The current Lab env is not batchable as is

- `num_envs=1`, with fixed `/World/...` prim paths rather than `{ENV_REGEX_NS}` (`isaac_lab_env.py:250-260`).
- It reads only `actions[0]` (`:733`).
- IK runs over the whole model and syncs with the host (`:446-470`).
- Body indices are global (`:403`).
- The CUDA graph is off (`:297`).
- There is no solver selection, so v1–v4 by law with K ∈ {1,2} (`:181, 324-326`).
- Compiled `ConfiguredTaskSpec`s are refused on Lab (`live_env.py:660-662`). **TRIM/v7_2 cannot run on Lab today.**

### Native vs Lab

| | Native (`isaac_sim_runner.py`) | Lab (`isaac_lab_env.py`) |
|---|---|---|
| Rigid engine | Isaac Sim 6.2 PhysX, **CPU dynamics** (`:603-606`), 10 × 2 ms | Newton + MuJoCo-Warp on GPU: implicitfast, elliptic, 50 iterations, 2 ms × decimation 10 (`:293-299`) |
| Tool | Physical grasp of a free knife. Material wrench on the knife, pad wrenches on the fingers | Option 1, `legacy_fixed_tool`: wedge collider rigid under `panda_hand` at the task mount or (0,0,0.18, Rx π) (`:88-106`); wrench applied to the hand with a **hard-coded 0.18 m lever** (`:44`). Option 2, `physical_grasp` with `NewtonTool` (`:110-120`) |
| Tasks / solver | Compiled TRIM scenes, v5–v7_2 (`:705`) | Registered ids only, v1–v4 |
| Cameras | Native RTX | Ideal Warp RGB-D |

**What parity can be claimed.**
- *Claimable:* material parity given an identical measured edge sequence (L0/L3).
- *Not equivalent between Native and Lab:* robot trajectories, grasp slip, the closed reaction→motion loop and
  success rates.
- Lab results carry `runtime_source` and need Native confirmation.

### Order

1. **P1-1 / P1-2 — workload A.** Batched ideal tracking drives the batched material with real cutting, using
   scripted, replayed or policy actions. Milestones M0 → M1 → M2.
2. **P1-3 — workload B, under a new ADR.**
   - `IsaacLabBatchedCuttingEnv(DirectRLEnv)` with an `{ENV_REGEX_NS}` scene and per-world body-index kernels.
   - Batched IK through `isaaclab.controllers.differential_ik` on `body_com_jacobian_w`
     (`articulation_data.py:888`), or joint-delta actions. **UNVERIFIED** with a free tool.
   - **B1:** rigid ruler-knife mount, labelled `reduced_fidelity: rigid_tool_mount_no_grasp`.
   - **B2:** physical grasp. Partial reset restores a recorded settled-grasp state, because settling one world steps
     every world. Its equivalence to a fresh settle is **UNVERIFIED**.
3. **P1-4.** Representative batched CUT/TRIM episodes, then Native single-env confirmation.

## 10. Explicit UNSUPPORTED (`UnsupportedBatchConfiguration(code)`, never a fallback)

- **Solvers:** any other than `tlmpm_v7_2`; also `v7_2_optimizations=False`, whose rollbacks stay single-env features.
- **Non-uniform batches:** dt, dx, spacing, substeps, S, K, law, tool or blade profile, B, `material_contact`, `pic`,
  local gravity; also the legacy `floor=True` tray.
- **Recorders and sensors:** `cut_events`, `cut_aware_surface`, per-step surfaces, camera or vision profiles, RTX
  sensors.
- **Geometry:** a geometry change on partial reset; workpiece motion beyond what the single solver supports.
- **Not yet built:** `backend="isaac_lab_newton"` before P1-3; mixed constitutive models in one batch; per-env tool
  profiles.
- **CPU device:** `device="cpu"` only with `parity_mode=True`, for tests.

## 11. Phase-2 extension point (hyperelastic and viscoelastic rubber)

A batched `ConstitutiveModel` declares:
- `model_id` and its version;
- `per_env_params` → `params[N,k]`;
- `per_particle_state` (name → dtype, shape, initial value or initializer) → registered flat `[P,…]` arrays;
- a stress kernel at step 6, with signature `(F, state…, params[env], env_of_particle) → stress`.

The current model `log_strain_von_mises_plastic_v5` owns `Fpi` and `plastic`.

The registry drives partial reset, snapshots, receipts and the L1 reset-equals-fresh test.

A rubber model must be energy-based: for example compressible neo-Hookean or Mooney–Rivlin, with finite-strain
Prony-series internal variables. It ships as a new solver id; v7_2 is never edited. It stays
`SIMULATOR_ONLY_UNCALIBRATED` until held-out real data pass. Each internal mat33 costs about 3.7 MB per 100k
particles.

## 12. Cost model (ESTIMATES unless marked measured)

### Memory per env

- 48.8–49.3 MB measured (`memory_bytes()`).
- About 3.3 MB of grouped-hash keys and ids, about 2.4 MB of reward bonds, and under 0.2 MB of tables.
- Total about **55 MB/env**: N=16 ≈ 0.9 GB, N=64 ≈ 3.5 GB.
- The GPUs are shared (about 20 GiB was in use on GPU0 at writing). So set an explicit batch budget, check free
  memory before allocating, and fail explicitly when it is short.
- Time, not memory, is the binding limit.

### Time

- N=1 costs 129–137 ms per control step (measured).
- About 10 operations per substep are latency-bound at ≤120 threads: remember_cut, two segment kernels,
  update_stroke, remember_pose, advance, and small memsets and copies.
- Particle kernels launch 1.03e5 threads, about 0.39 of one resident wave on an RTX 5090. That assumes 170 SMs ×
  1,536 threads, which is UNVERIFIED for sm_120.
- Batching keeps the graph node count constant while filling the waves. Expected aggregate gain at N=8–16:
  **about 1.3–3×** over sequential single envs, i.e. about 35k–85k full-fidelity env-steps per GPU-hour.
- A measured gain above 4× must be audited before it is reported.
- Active-frontier work adds up across envs with no straggler wait, but warp divergence remains.
- **UNVERIFIED:** the cost of the grouped 64-bit hash sort at N×1e5 points. The fallback is an array of per-env
  HashGrids: still L0-exact, but linear in N.

## 13. Staged plan

### P1-0 baseline (GPU, scheduled by the orchestrator)

Single-env v7_2 TRIM, workload A, seeds 512/15 with Native-replayed edges: graph time per phase; nsys kernel times for one step per phase; memory; a sequential N=4/8 single-env comparison (a baseline, not a runtime); at least 3 repeats to define the L3 band.

Record commands, versions, seeds, failures and paths.

### Files (`src/robotic_cutting_sim/batched/`)

- `layout.py`: pack-from-frozen, offsets, uniformity checks, guards.
- `tlmpm_v7_2_batch.py`: ENV kernels and `BatchedTLMPMv7_2`.
- `constitutive.py`.
- `pose_interp.py`.
- `material_batch.py`: graph, coupling, reset, diagnostics, wrench.
- `reward_packed.py`.
- `vector_env.py`.
- `errors.py`.
- GPU probe: `scripts/probe_batched_material_gpu.py`, writing to new `artifacts/rl_ready/` directories.

### Tests

CPU tests run with `.venv/bin/python -m pytest`; torch tests run in `.deps/isaaclab-venv`.
- `test_batched_layout.py`: offsets, audits, uniformity, guards, UNSUPPORTED codes.
- `test_batched_cpu_parity.py`: L0.
- `test_batched_isolation.py` and `test_batched_partial_reset.py`: L1, with the `vars()`-complete reset check.
- `test_batched_pose_interp.py`.
- `test_vector_env_api.py`: shapes, dtypes and device; no privileged keys; partial reset; action parity with
  `control.py`.
- `test_batched_frozen_sources.py`: SHA256 pins for the frozen modules named in the Status, `live_material.py` and
  the evaluators.
- The repository naming guard.

### Milestones

- **M0 (N=1):**
  - L0 PASS;
  - GPU N=1 TRIM replay within the L3 band;
  - allocation-free capture;
  - receipts.
- **M1 (N=4):**
  - CPU: heterogeneous L0 and L1 PASS.
  - GPU: L2; the L4 probe (PASS, FAIL or BLOCKED, reported as is); the API over 4 full 1200-step episodes (2 replayed, 2 scripted); staggered partial resets; `evaluate()` through the frozen evaluator.
- **M2 (N=8–16):**
  - throughput and memory against P1-0 (report the measured ratio);
  - L3 for seeds 512/15 across copies;
  - at least 3 consecutive batches with staggered resets and no leak growth.

## 14. Risks and detection

| # | Risk | Detection or mitigation |
|---|---|---|
| 1 | GPU isolation evidence is weak because atomics are nondeterministic | CPU bitwise L0/L1, L2 audits, L4 if Warp supports it |
| 2 | ENV kernels drift from frozen semantics | L0 including active-link fixtures, reused frozen helpers, hash pins |
| 3 | Reset misses state, including future internal variables | Registry plus the `vars()`-complete L1 test |
| 4 | Torch/Warp stream races | Single-stream policy and a sync-every-step debug mode |
| 5 | Grouped hash sort cost, or allocation during capture | M2 timing; capture with allocation checks; per-env grid fallback |
| 6 | Gain below the 1.3–3× estimate | Honest M2 report. Faster curricula must be labelled `reduced_fidelity` |
| 7 | Invalid envs continue silently | Device validity flags, strict mode, exclusion from success statistics |
| 8 | Pad-box reconstruction for replay is UNVERIFIED | Explicit replay label until verified |
| 9 | Workload B unknowns: env-regex scene, batched IK, grasp restore, the hard-coded 0.18 m lever | Resolved in the P1-3 ADR before any code |
| 10 | State-only TRIM policies are blind to geometry | Documented scope. Privileged critic only through an explicit profile |
| 11 | Venv split: no torch in `.venv` | Torch-free core. API tests run in the torch venv; elsewhere they are NOT_RUN |
| 12 | `.deps` symlinks into the main checkout | Treat as read-only; record versions in receipts |

## Implementation notes

Appended during implementation; earlier sections are unchanged. Each subsection names its work package and
records where the code deviates from or refines the text above.

### P1-1/P1-2 device components: `pose_interp`, `edge_delta_v1`, `state` observation, `trim_bond_progress_packed_v1` (2026-11-10)

Files: `src/robotic_cutting_sim/batched/pose_interp.py`, `src/robotic_cutting_sim/rl/{actions,observations,reward_packed}.py`;
tests `tests/test_batched_pose_interp.py`, `tests/test_rl_{actions,observations,reward_packed}.py`.

- **Test invocation.** The `.venv` editable install resolves `robotic_cutting_sim` to the main checkout
  (`/home/ningan/RoboticCuttingSim/src`), not to this worktree. Worktree tests therefore run with `PYTHONPATH=src`
  in `.venv` as well (`PYTHONPATH=src .venv/bin/python -m pytest ...`); without it they test the wrong source or fail
  to import new modules.
- **§7.1 pose interpolation (refinement, stronger than required).**
  - Inputs are `float64 [N,7]` world poses on the device; per-env `T_local_world` is a `wp.mat44d[N]` plus a
    `wp.vec4d[N]` quaternion computed once on the host at pack time with `Rotation.from_matrix` (the call
    `transform_poses` repeats every control step, so it is identical by construction).
  - The kernel follows SciPy's numpy (Cython) operation order exactly, including two host BLAS FMA chains that the
    frozen path uses implicitly: `np.linalg.norm` in `robot.validate_pose` (OpenBLAS ddot = sequential FMA chain) and
    the `(1,3)@(3,3).T` matmul in `transform_points` (`fma(p2,R2, fma(p0,R0, p1*R1))`). All other products and sums
    are single IEEE operations that the compiler cannot contract (`__dmul_rn/__dadd_rn/__dsub_rn` on CUDA via
    `wp.func_native`).
  - Measured: **0 float32 ulp** (bitwise) over 2,341,248 pose/twist components (angles 0 to π, including 1e-14 rad
    cancellation cases, sign flips, heterogeneous transforms) on the Warp CPU device in `.venv` (NumPy 2.5.4,
    SciPy 1.18.1) and `.deps/isaacsim-venv` (NumPy 2.3.1, SciPy 1.17.0), and on CUDA (RTX 5090, GPU1, shared,
    Warp 1.17.0). The gate in the test stays ≤ 1 ulp. A dedicated test pins the two host BLAS orders on the
    machine that runs it (AMD Threadripper PRO 9975WX here); on a host whose BLAS selects another kernel that test
    fails first, and near-zero-rotation twists could then differ by more than 1 ulp (float64 cancellation), at
    absolute errors near 1e-16.
  - The kernel cannot raise: it writes `status[N]` (`OK`, `NONFINITE_POSE`, `NON_UNIT_QUATERNION` with the 1e-6
    tolerance of `validate_pose_xyzw`); outputs of a non-OK env are unspecified and the caller must treat the env as
    invalid. `fill_cut_rows` writes `cuts[N,T]` from a per-env digit.
- **§8 actions `edge_delta_v1`.**
  - The commanded edge pose stays `float64` state on the device (it accumulates like the frozen `inner._commanded`).
    Measured against the frozen NumPy path: max |Δ| < 1e-12 over 5,120 random env-actions (default and TRIM limits)
    and over a 300-step accumulated rollout; the test gate is float32 tolerance.
  - Rejections follow `PolicyReadyEnv` (validation order target → `cut_enabled` → `cut_index`; HOLD keeps the clock):
    nonfinite `a[0:6]` → `REJECTED_INVALID_TARGET`; nonfinite `a[6]` → `REJECTED_INVALID_SCHEMA` (the frozen
    "cut_enabled must be a Boolean"); nonfinite `a[7]` → `REJECTED_CUT_INDEX_OUT_OF_RANGE`. The ADR did not specify
    the last two.
  - Finite out-of-range components are clipped silently, exactly as `normalized_delta` clips; `out_of_box` reports
    it without changing the status. Selector mapping: `digit = min(floor((clip(a7,-1,1)+1)/2·K), K−1)`; `a6 > 0` arms.
  - **Not ported:** the `PolicyReadyEnv` sampled tool-vs-fixture/table collision check and IK rejection, so
    `REJECTED_COLLISION_*` and `REJECTED_IK_UNREACHABLE` are never emitted (recorded in `spec()["not_implemented"]`).
    The delta frame is `edge` per this ADR; the `cutting_research_v1` preset default is `world`.
- **§8 observations (`rl_state_obs_v1`).** Policy block is 18-dim in the fixed order pose(7), time remaining(1),
  previous action(8), arm flag(1), step fraction(1). The pose quaternion is canonicalized to w ≥ 0 in the observation
  only. "Previous action" is the executed action (the HOLD row `[0]*6+[-1,0]` after a rejection). Default critic
  terms: target-bond progress, off-target fraction, pending material wrench, invalid flag. The guard rejects any
  `PRIVILEGED_KEYS` name and material/code/contact/wrench/force/generator tokens in policy term names.
- **§8 reward `trim_bond_progress_packed_v1`.**
  - Module path `rl/reward_packed.py` follows amendment 1 (§13's file list still names `batched/`).
  - Definitions: target bond = vertical reference-lattice bond with lower endpoint `z ≤ h` and upper `z > h` (the
    `compile_seed` "above" test) at a column where `max_b bump.height(x,y) > 0`; this is one bond per footprint
    column. Substrate bonds have both endpoints at `z ≤ h`, all depths by default (optional depth band). Measured:
    seed 512 has 129 target / 298,240 substrate bonds; seed 15 has 452 / 298,240.
  - Off-target is normalized by the env's target-bond count, not by the substrate count, so both terms have the same
    scale. Default weights `(progress 1, off-target 1, time 1e-4/step, invalid 1)` are provisional. Invalid envs get
    only `−invalid_penalty`.
  - Counting is exact int64 cumulative-sum segmentation in Torch (deterministic on GPU), not a Warp kernel. The
    conflict-rule ports (NumPy, Torch) are bit-identical to the frozen `tlmpm_v5._conflict_packed` for K = 1, 2, 15.
  - The NumPy pack/reference path is torch-free; the device reducer needs Torch.

### P1-1 batched material core: `layout`, `tlmpm_v7_2_batch`, `constitutive`, `material_batch`, `diagnostics` (2026-11-10)

Files: `src/robotic_cutting_sim/batched/{errors,layout,constitutive,tlmpm_v7_2_batch,material_batch,diagnostics}.py`;
tests `tests/test_batched_{layout,cpu_parity,isolation,partial_reset,frozen_sources}.py` with helpers in
`tests/batched_fixtures.py`; GPU probe `scripts/probe_batched_material_gpu.py`.

- **L0 result.** Bitwise (`np.array_equal`) on the Warp CPU device over every Warp array of the single solver
  (vars()-complete) after every control step: N=1, N=3 heterogeneous (seeds 512/15/7, different particle and node
  counts, Young's modulus, friction, yield stress, fracture energy), N identical copies, one full-size TRIM substep
  (102,732 particles), and all 47 WP17 frontier fixtures plus the 8 pending-lifecycle fixtures as envs of
  heterogeneous batches. No tolerance was needed anywhere.
- **Test scenes.** CPU tests use seeded TRIM scenes at 2.5 mm particle spacing (`compile_task(...,
  particle_spacing_m=.0025)`; grid, dt, substeps, law, K and S unchanged), with a scripted edge motion that starts
  2 mm before the first protrusion at the reference cutting height. Full resolution is covered by the one-substep smoke.
- **Dead branches dropped (refined §3.3).** The ENV copy of `_rigid_contacts` keeps only the law-4 contact path that
  the frozen step always passes (laws 5/8 use it) and has no tray loop. `floor=True` and laws other than 8 raise
  UNSUPPORTED, so the removed code is unreachable.
- **Kernel scalars kept uniform.** Uniform per batch, checked by `layout.uniform_signature`: dt, dx, spacing, radius,
  volume, trajectory length, S, K, law, segments, B, contact flags, pic, float32 local gravity, contact-radius
  scale, law-4/8 parameters, optimisation flags, control dt and the blade cross-section. Tool-profile differences
  raise `UNSUPPORTED_PER_ENV_TOOL_PROFILE`, including a handle-only difference with an identical cross-section,
  compared via `ToolProfile.to_dict()`. Per-env `params[N,9]` holds λ, μ, σ_y, contact stiffness, friction, G_c,
  sharpness, slice efficiency and particle-contact stiffness.
- **`_claim_slots` guard (performance, result-neutral).** The ENV kernel skips `atomic_max(max_occupied, e, ·)` when
  the stored value is already ≥ the candidate. Integer max is idempotent, so the result is unchanged on CPU and GPU.
  Reason: with an env-indexed address the CUDA compiler no longer warp-aggregates that atomic. The unguarded kernel
  measured 0.355 ms per substep against 0.027 ms for the frozen kernel at N=1 (RTX 5090, shared).
- **Reset restores everything (deviation from §6).**
  - `reset` also restores `slot`, `slot_code` and the grid arrays (`gm/gp/gj/old/new_v`), although every substep
    recomputes them, so that a full reset is bitwise equal to a freshly packed batch over *every* array.
  - The fresh values are those of the frozen constructors, not the §6 text: `x_start`, `link_velocity_start` and
    `link_previous_pose` are **zero** (not `reference` or the initial pose); `previous_pose` is the initial tile pose.
  - The only array a partial reset leaves alone is the shared substep `index`; it is zeroed before every control step.
  - Coverage lists are `material_batch.SOLVER_RESET_ARRAYS / SOLVER_CONSTANT_ARRAYS / SOLVER_SHARED_ARRAYS /
    MATERIAL_*_ARRAYS / RECOMPUTED_EVERY_CONTROL_STEP / DIAGNOSTICS_SCRATCH`, enforced over `vars()`.
  - Constitutive internal variables are reset through the registry (`per_particle_state`, initial value
    `identity` or `zero`).
- **Coupling.**
  - `pad_wrench` is `[N,B,6]`; rows of non-recipient boxes, such as the handle, stay zero. This refines `[N,2,6]`.
  - `invalid_reason[N]` is a sticky bit mask until reset: 1 nonfinite state, 2 slot overflow, 4 invalid measured
    pose (the `pose_interp` status), 8 cut digit out of range, 16 nonfinite wrench.
  - Wrench rotation is a plain sequential float64 sum. It is exactly equal for the TRIM identity workpiece rotation
    (tested). For a non-identity rotation, NumPy's BLAS order may differ by 1 ulp in float64; this is untested.
  - `advance(interpolation="host_reference")` (parity mode only) stores `validate_pose(measured)` as `last_pose`,
    as `LiveCuttingMaterial` does. The device path stores the raw measured pose, and `pose_interp` normalises it
    again; whether that equals the frozen path bitwise for poses that are not exactly unit is **UNVERIFIED** (open).
- **Segmented diagnostics** (`batched/diagnostics.py`, `fuse_fp=False` as the frozen module). Each env's summary
  equals the frozen `MaterialDiagnosticsReduction.read()` of that env exactly; this is tested.
- **Graph capture.** `HashGrid.reserve(P, with_groups=True)` runs before capture. Measured mempool use before and
  after capture is equal: 0 bytes allocated, N=1 and N=4.
- **Upstream hazard (Warp 1.17.0, not caused by batching).**
  - A frozen single `LiveCuttingMaterial` graph, captured without `HashGrid.reserve`, segfaults in `capture_launch`
    after any later, larger non-captured `HashGrid.build` in the same process. 400k points reproduce it, grouped
    or not, with no batched code involved.
  - Batched graphs, which reserve first, survive a later larger batch build.
  - Rule: in one process, build batches before any frozen single material, or use separate processes.
  - Evidence: `artifacts/rl_ready/p1_1/sort_temp_hazard_001/`.
- **GPU smoke (L2)** in `artifacts/rl_ready/p1_1/gpu_smoke_002`.
  - Setup: N=1 and N=4 copies of seed 512 at full resolution (102,732 particles per env), 5 control steps, real
    graph capture, device `pose_interp`.
  - Result: L2 PASS for every env and step. Mass is bitwise constant; codes are monotone; state is finite; minimum
    det F ≥ 0.897; slot overflow is 0; range audits are 0; the instrumented grouped query found 0 foreign
    neighbours in about 8.3M candidates per env.
  - All 4 GPU copies end with identical codes (3,220 labelled), and positions differ by ≤ 7.1e-8 m from atomic
    order. The frozen single on the same GPU also labels 3,220.
  - Memory: 49.26 MB per env (`memory_bytes`, excluding the hash grid).
- **Smoke timing.** These are not the M2 benchmark rows. All runs were SHARED_GPU: GPU1 had a co-tenant at about
  96–100% utilisation; state only, no cameras, no recording, warm.
  - Interleaved A/B in one process (`artifacts/rl_ready/p1_1/ab_timing_001`), per 20 ms control step (200 substeps):
    single graph 270 ms; batched N=1 270 ms; batched N=4 558 ms. That is about 1.9× aggregate over sequential
    singles at N=4.
  - The first smoke, `gpu_smoke_001`, ran before the `_claim_slots` guard and measured N=1 at 0.43 s against the
    single's 0.13 s (then about 1% co-tenant load). It is kept as a failed-performance record.

### P1-2 vector env `open_vector_env` and workload-A policies (2026-11-10)

Files: `src/robotic_cutting_sim/rl/{vector_env,policies_batched}.py` (`open_vector_env`, `TaskFamily`,
`VectorCuttingEnv`, `UnsupportedVectorConfiguration` exported lazily from `robotic_cutting_sim.rl`); tests
`tests/test_vector_env_api.py`, `tests/test_vector_env_gpu_stream.py`, `tests/test_rl_policies_batched.py`;
milestone runner `scripts/rl_ready/run_vector_milestones.py`. Nothing under `batched/` was changed for P1-2.

- **Composition.** The env owns one `BatchedCuttingMaterial` and drives it only through its device buffers:
  zero-copy Torch views (`wp.to_torch`) of `measured_pose`, `cut_input`, `step_count`, `invalid_reason`, `seed`,
  `reset_mask`, `reset_seed`, `pending_wrench`, `code` and the box arrays, then `advance(interpolation="device")`.
  Ideal tracking: the measured pose written to the material is the float64 commanded pose of `edge_delta_v1`.
- **Task family (refines §8 `TaskFamily("agentic_trim_id", scene_seeds)`).** `TaskFamily(scene_seeds,
  family_id="agentic_trim_id"|"agentic_trim_ood", sim_seed=42, particle_spacing_m=None, max_episode_steps=None,
  initial_edge_poses=None, pad_boxes_edge=None, pad_recipients=None)`. Tasks are `compile_task(generate_scene(seed),
  goal_config(), seed=42)`, whose `config_sha256` equals the WP-17 payloads (checked by the milestone runner).
  Default initial edge = `task.reference_edge_pose_world(0)` (as `kinematic_env`); the WP-17 replays use the
  recorded reset edge. `pad_boxes_edge` + `pad_recipients` configure the Native handle + two-pad obstacle set
  (`configure_pad_obstacles`), otherwise the handle box only.
- **Replay drive (new, privileged).** `step_measured_edge(edge_pose[N,7] f64, cut_index[N] i32, pad_boxes=None,
  env_mask=None, actions=None)`: masked envs are driven by recorded measured poses (and their pad boxes are
  written first, as `update_pad_obstacles`), the others apply `actions` through `edge_delta_v1` in the same
  batched control step. `info["drive"]` is 1 for replay and `info["action_status"]` is -1 (`STATUS_REPLAY`).
- **Seeds and autoreset.** Env seeds are the material `seed[N]`. Explicit resets keep or replace them;
  `autoreset="same_step"` resets done envs with the masked device kernels (no host sync) and sets the next seed
  to `splitmix64(seed + episode + 1)`. `RandomPolicyV1` draws from SplitMix64(seed, episode, step, component)
  (counter based; independent of N and of the other envs).
- **Strict and training modes.** `strict=True` (default) keeps the material's single int32 readback per step
  and raises `InvalidBatchEnvs`; `strict=False` performs no host read in `step` (tested by counting
  `wp.array.numpy` and `Tensor.item/tolist/cpu/numpy` calls during a step: 1 and 0 respectively) and reports
  `terminated = invalid_reason != 0`.
- **Streams (§7, implemented here).** Every public call enters `wp.ScopedStream(wp.stream_from_torch(
  torch.cuda.current_stream()))`; the graph is captured on Warp's own stream at construction (the legacy default
  stream cannot be captured) and replayed on the Torch stream. `debug_sync=True` synchronizes after every call.
  GPU test: the recorded Warp stream equals the Torch current stream for the default and a side stream.
- **Evaluation.** `evaluate(env_ids)` hands `task.evaluate_goal` (frozen `goals.evaluate_goal`) a recording of
  two frames: the reset state (t = 0, reference positions, zero codes) and the host snapshot at
  `t = step_count·dt`; reference arrays are the packed frozen ones. `frames=` adds intermediate snapshots
  (M0 compares 2 vs 121 frames). The integrity layer's history check sees only those frames.
- **Fidelity** is computed in `receipt()["fidelity"]` against `NATIVE_TRIM_V7_2_REFERENCE`: always
  `ideal_edge_tracking_no_robot_dynamics` and `reaction_not_fed_back`; `no_finger_pad_obstacles` with the handle
  box only; any resolution or episode-length difference by value; per env
  `finger_pad_boxes_static_in_edge_frame_on_non_measured_steps` when configured pads were not updated from a
  measurement (action-driven steps).
- **Refusals.** `UnsupportedVectorConfiguration` (subclass of `UnsupportedBatchConfiguration`, adds
  `UNSUPPORTED_RUNTIME/OPTION/TASK/DEVICE`) for other runtimes, backends, materials, observation profiles,
  action specs, rewards, autoreset modes, cameras, recorders, reset options, pad boxes without configured pads
  and a scene-seed change on partial reset (`UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET`).
- **`recorded_command_policy_v1`.** Open-loop float32 actions computed against the reconstructed commanded
  pose (frozen NumPy control path), so rounding does not accumulate; the material then sees the commanded
  targets, not the Native measured poses (no tracking error): a different material input than WP-17.
