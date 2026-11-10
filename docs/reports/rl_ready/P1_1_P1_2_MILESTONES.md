# P1-1 / P1-2 — vector env `open_vector_env` (workload A) and milestones M0 / M1 / M2

Date: 2026-11-10. Branch `claude/rl-ready-rubber-physics`. Plan: `docs/RL_READY_PLAN.md` (P1-1, P1-2, D-03..D-10),
design: ADR 0021 (§5 L2–L4, §6–§8, §13 and the implementation notes at its end).
Compact evidence: `docs/evidence/rl_ready/p1_2_milestones_001.json` (rows, per-run results, sources, audit).
Raw outputs (not committed): `artifacts/rl_ready/p1_2/<run>_NNN/`, logs `artifacts/rl_ready/logs/p1_2_*.log`.

## Status

| Item | Status |
| --- | --- |
| `open_vector_env` API (reset / partial reset / step / step_measured_edge / evaluate / receipt / spaces) | PASS (CPU tests, GPU runs) |
| CPU API tests `tests/test_vector_env_api.py` (bitwise partial reset, isolation, seeds, autoreset, action parity, truncation, UNSUPPORTED, frozen evaluator) | PASS 12/12 |
| Stream policy (Warp on Torch's current stream) + `debug_sync` | PASS (GPU test 2/2) |
| `recorded_command_policy_v1` reconstruction of the WP-17 commanded targets | PASS (≤ 3.3e-11 m, ≤ 2.5e-10 quaternion component; host frozen path and device Torch path) |
| **M0** N=1 reset/step + WP-17 replay s512 / s15, frozen evaluator | PASS (pre- and post-review-fix) |
| M0 L3 band: final codes, labelled counts, evaluator | PASS (4/4 runs: codes = Native, 0 particles) |
| M0 L3 band: max position difference vs Native over time | 3 of 4 runs inside; **pre-fix s15 outside** (6.50e-4 m > band max 4.62e-4 m); post-fix s15 inside |
| **M1** N=4, 2 replay + 2 policy-driven full episodes, staggered partial resets, L2 every step, evaluate() | PASS (pre-fix run; post-fix rerun see §5) |
| M1 L4 `DeterministicMode.RUN_TO_RUN` | **BLOCKED** (frozen `tlmpm_v5._particle_contacts` mixes add/max atomics; module does not compile) |
| **M2** N=8, 3 consecutive full episodes per env with staggered partial resets (24 episodes) | PASS physics (L2, codes = Native 24/24, evaluator PASS 24/24) |
| M2 N=16, 1 full episode per env (16 episodes) | PASS physics (codes = Native 16/16, evaluator PASS 16/16) |
| M2 memory: VRAM / mempool / RSS per 100 steps | PASS (Warp mempool and Torch allocation flat; RSS bounded) |
| **M2 throughput gain over N=1** | PASS at matched co-tenant load: **1.45× (N=8), 1.42× (N=16)** over batched N=1 (alternation round 1, separate processes, co-tenant SM mean 46–47 % in all three); see §6 for the confounded comparisons |
| One kernel launch covers all envs (nsys graph-node trace N=1 vs N=4) | PASS |
| Multi-HashGrid audit (Warp 1.17 static-descriptor hazard) | PASS — no P1-2 evidence run held two device grids; guard added |
| EXCLUSIVE_GPU rows | NOT_RUN (GPU1 had a co-tenant at up to 99 % SM during every run) |

## 1. What was built

- `src/robotic_cutting_sim/rl/vector_env.py`: `open_vector_env(num_envs, runtime="rl_gpu_headless",
  task=TaskFamily(scene_seeds=...), material="scene_declared", observation_profile="state"|"privileged_state",
  device="cuda", backend="kinematic_ideal_tracking_batched", action_spec="edge_delta_v1",
  reward="trim_bond_progress_packed_v1", autoreset=False|"same_step", strict=True, output_dir=None,
  debug_sync=False)` → `VectorCuttingEnv`. Exported lazily from `robotic_cutting_sim.rl` (torch-free import).
  - `reset(seeds=)`, `reset(env_ids=, seeds=, scene_seeds=)` (masked, no reallocation/recapture; a scene change
    raises `UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET`), `step(actions[N,8])` → `obs` (`"policy"` [N,18],
    `"critic"` with `privileged_state`), `reward[N]`, `terminated[N]`, `truncated[N]` (1200 for TRIM), `info` of
    `[N]` tensors (step_count, action_status, invalid_reason, cut_digit, drive, seed, episode, reward terms).
    No per-env loop; in strict mode one int32 readback per step (counted in a test: 1 strict, 0 training).
  - `step_measured_edge(edge_pose[N,7], cut_index[N], pad_boxes=None, env_mask=None, actions=None)`: privileged
    replay drive (`drive: measured_trajectory_replay`), mixable with action-driven envs in the same batched step.
  - `evaluate(env_ids)`: frozen `goals.evaluate_goal` (via `ConfiguredTaskSpec.evaluate_goal`) on a host snapshot
    (2 frames: reset state + end state; optional intermediate frames).
  - `receipt()`: versions, frozen and batched source SHA256s, layout, per-env receipts, computed `fidelity`
    against `NATIVE_TRIM_V7_2_REFERENCE` (workload A: `reduced_fidelity: true`, reasons
    `ideal_edge_tracking_no_robot_dynamics`, `reaction_not_fed_back`, plus `no_finger_pad_obstacles` with the
    handle box only, resolution/episode differences by value, per env
    `finger_pad_boxes_static_in_edge_frame_on_non_measured_steps`).
  - Gymnasium 1.4.0 spaces (`single_*`/batched `Box`, `Dict` observation), `metadata`; Stream policy ADR §7.
- `src/robotic_cutting_sim/rl/policies_batched.py`: `recorded_command_policy_v1`, `RandomPolicyV1` (SplitMix64 of
  seed/episode/step/component), `MeasuredTrajectoryDriver` (device tables gathered by step count).
- `scripts/rl_ready/run_vector_milestones.py` (modes m0, m1, m2, l4, profile), `scripts/rl_ready/make_p1_2_evidence.py`.
- Batched core change (requested by the orchestrator after the P1-3a finding, no kernel source changed):
  `BatchedCuttingMaterial` refuses a second live CUDA `wp.HashGrid` in the process
  (`UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS`), exempts its own grid on `rebuild()`, gains `release()` and records
  `device_hashgrid_guard`; opt-out `allow_multiple_device_hashgrids=True` for diagnostics. The vector env
  re-checks on full reset and releases its grid on close. Tests: `tests/test_batched_hashgrid_guard.py`,
  `tests/test_batched_layout.py::test_every_unsupported_code_is_raised`.

## 2. Runtime, versions, commands

`.deps/isaacsim-venv/bin/python` (Python 3.12.12, Warp 1.17.0, NumPy 2.3.1, Torch 2.11.0+cu128, Gymnasium 1.4.0),
`PYTHONPATH=src`, no Kit. GPU: NVIDIA GeForce RTX 5090, physical index 1 (`CUDA_VISIBLE_DEVICES=1`, `cuda:0`
inside), driver 570.211.01, 32,607 MiB. Every job under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock`
with `TMPDIR=/mnt/data/ningan/tmp` and a free-VRAM check (> 6–7 GB) before start. Chains:
`/mnt/data/ningan/tmp/p1_2_chain.sh`, `p1_2_chain2.sh` (post-fix reruns), `p1_2_chain3b.sh` (throughput
alternation); each line is

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock \
  .deps/isaacsim-venv/bin/python scripts/rl_ready/run_vector_milestones.py --gpu-index 1 <mode args>
# m0: --mode m0 --seed {512,15} [--run-name m0_s{512,15}_postfix]
# m1: --mode m1 [--run-name m1_n4_postfix]          (staggered resets at steps 300/400/500/600)
# l4: --mode l4 --steps 260
# m2: --mode m2 --num-envs 1 --seed {512,15} --episodes 1 | --num-envs 8 --episodes 3 --stagger 50
#     | --num-envs 16 --episodes 1 --stagger 25
# alternation: --mode m2 --episodes 1 --stagger 100000 --l2-every 0 --num-envs {1,8,16} --max-global-steps {300,150,100}
# single-env reference: scripts/rl_ready/run_baseline.py --mode replay --seed 512 --gpu-index 1 --warm-resets 0 \
#     --run-name p1_2_single_ref_s512
# nsys: nsys profile --cuda-graph-trace=node -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
#     -o artifacts/rl_ready/p1_2/nsys_n{1,4}_001/report <python> ... --mode profile --num-envs {1,4} --steps 2
PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_p1_2_evidence.py ... --out docs/evidence/rl_ready/p1_2_milestones_001.json
```

Workload A (full fidelity material): seeded TRIM seeds 512 / 15 (WP-17 configuration; compiled
`config_sha256` checked equal to the WP-17 payloads in every run), 102,732 / 103,898 particles, 1 mm / 4 mm /
0.1 ms, 200 substeps per 50 Hz control step, tlmpm_v7_2 law 8, K=2, S=8, handle + two finger-pad obstacle boxes
(replays update them from the recorded measurements each step), no cameras, no recording, robot DOF 0. Material
seed 42; env seeds 42. Every row is `reduced_fidelity` (workload A).

Tests (CPU): `.deps/isaacsim-venv` (Torch, D-09) `tests/test_rl_{actions,observations,reward_packed,replay,benchmark,
policies_batched}.py tests/test_vector_env_api.py tests/test_vector_env_gpu_stream.py tests/test_batched_hashgrid_guard.py
tests/test_repo_naming.py`: 113 passed, 1 skipped (the GPU stream test without CUDA = NOT_RUN on CPU; on GPU1:
2 passed). `.venv` (no Torch): `tests/test_batched_*.py tests/test_repo_naming.py` 151 passed after the guard
(one layout test failed until the new code was added to it — fixed); full `.venv` suite `-m "not gpu"` before the
guard: 2163 passed, 24 skipped, 6 xfailed (torch tests skip there = NOT_RUN).

## 3. Source commits per run (review fixes landed during the chain)

The P1-1 review fixes (a673e32 … 8535ef4) landed while chain 1 ran; every manifest records `git_head`,
`git_dirty`, `git_diff_stat` (from d765855 on) and SHA256 of every `batched/` and `rl/` source at start.

| run | commit | tree | code state |
| --- | --- | --- | --- |
| `m0_s512_001` | 6ba47b7 | dirty (concurrent review-fix edits) | **pre-review-fix** |
| `m0_s15_001` | 7e827a7 | dirty | **pre-review-fix** |
| `m1_n4_001` | 9bf4d21 | dirty | **pre-review-fix** (after a673e32, d210b23; before 0d143c2, 61c4174, 9551a0e) |
| `l4_deterministic_002` | fe971fb | clean | post-fix |
| `m2_n1_001` (s512) / `m2_n1_002` (s15) | ddfe2c7 / fe971fb | clean | post-fix |
| `m2_n8_001` | 359ce0d | dirty: only `scripts/rl_ready/run_vector_milestones.py` | post-fix |
| `m2_n16_001` | 35f01d5 | dirty: only `scripts/rl_ready/make_p1_2_evidence.py` | post-fix |
| `p1_0/p1_2_single_ref_s512_001` | 35f01d5+ | scripts only | frozen single material |
| `profile_n{1,4}_001` (nsys) | 138c937 | docs/scripts only | post-fix |
| `m0_s512_postfix_001` | 138c937 | docs/scripts only | post-fix |
| `m0_s15_postfix_001` | d0a09ff | dirty: HashGrid guard edits in progress (hashes in manifest) | post-fix + guard |
| `m1_n4_postfix_001`, `alt_r*_n*` | see evidence `runs.*.source` | | post-fix + guard |

Between ddfe2c7 and the last run, the only change under `batched/` used by workload A is the HashGrid guard
(construction-time check, no kernel change); `batched/lab_scene.py` (workload B) changed and is not used here.

## 4. Multi-HashGrid audit (Warp 1.17 static-descriptor hazard, P1-3a)

Warp 1.17.0 replays every CUDA graph with the device-grid descriptor of the grid built last on the host. Host (CPU)
grids are not affected. Audit of every P1-2 process (device grids = CUDA `wp.HashGrid` objects alive while a graph
was replayed):

| run(s) | device grids | verdict |
| --- | --- | --- |
| M0 ×4, M1 ×2, M2 N=1/8/16, alternation runs, profile/nsys ×2, smoke_m0/m1/m2 | 1 (the batch's own; pack-time frozen solvers are CPU) | VALID |
| `p1_2_single_ref_s512_001` (frozen single, `--warm-resets 0`) | 1 | VALID |
| `l4_deterministic_002`, `smoke_l4_001` | 0 (failed at CPU module compile) | n/a (BLOCKED / crashed) |
| `tests/test_vector_env_gpu_stream.py` | 2 sequential envs, the first never replayed after the second was built (now released on close) | VALID |
| in-process multi-batch A/B mode (`--mode ab`, added at 35f01d5, never committed to evidence) | would hold 4 | **never run** (chain killed before start); mode removed |
| P1-1 `artifacts/rl_ready/p1_1/ab_timing_001` (single + N=1 + N=4 interleaved in one process; not a P1-2 run) | 3 | **INVALID_MULTI_GRID_PROCESS** — its "≈1.9× at N=4" timing (ADR 0021 notes) is withdrawn by this audit |

No parity or throughput claim of this report rests on a multi-grid process, so nothing had to be rerun for that
reason. Throughput comparisons alternate separate processes instead (§6, `alt_*`).

## 5. Milestones

### M0 (N=1): reset/step and WP-17 replay through `step_measured_edge`

API check per run: reset → one HOLD `edge_delta_v1` step (status ACCEPTED, step_count 1) → full reset
(step_count 0, 1.2 ms). Then 1200 replay steps with the recorded measured edge, resolved digits and measured pad
boxes; 120 host samples (every 10 steps, outside timing) compared with the Native samples; L2 every step.

| run | code | final codes vs Native | labelled d0 (Native) | max pos. diff over time (P1-0 band) | final max / mean diff | transient code diffs | evaluator 2 / 121 frames | L2 (min det F) | warm step median | env-steps/s | co-tenant SM mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m0_s512_001 | pre-fix | equal (0) | 37958 (37958) | 2.24e-5 m (1.39e-5–3.85e-5) inside | 2.24e-5 / 5.2e-7 | step 320, 1 particle | PASS / PASS, identical | PASS (0.910) | 0.293 s | 3.51 | 45 % |
| m0_s512_postfix_001 | post-fix | equal (0) | 37958 | 3.85e-5 m inside (at the band max) | 3.85e-5 / 1.0e-6 | none | PASS / PASS, identical | PASS (0.910) | 0.293 s | 3.50 | 47 % |
| m0_s15_001 | pre-fix | equal (0) | 85786 (85786) | **6.50e-4 m — outside** (1.96e-4–4.62e-4) | **1.36e-4** (band 6.3e-5–1.18e-4) / 1.5e-6 | step 500, 1 particle | PASS / PASS, identical | PASS (0.844) | 0.309 s | 3.29 | 46 % |
| m0_s15_postfix_001 | post-fix | equal (0) | 85786 | 1.87e-4 m (below the band minimum) | 7.0e-5 / 8.6e-7 | step 500, 1 particle | PASS / PASS, identical | PASS (0.844) | 0.309 s | 3.27 | 49 % |

Evaluator (all four): verdict PASS, integrity PASS, removed fraction 1.0 for every protrusion, largest
below-plane component share 1.0 (P1-0/Native: 0.9999999999992), detached 0.0. Pre- vs post-fix: identical final
codes and evaluator outcomes, identical timing (0.293 / 0.309 s median); position differences vary within the GPU
float-atomic run-to-run spread, which the 4-run single-env band (P1-0) under-samples — the pre-fix s15 run is the
only one outside it (P1-3a also reports s15 outside the band in 3 of 4 envs of its own runs). Cold start (process
start → first step done) 6.3–10.5 s; first reset 0.05–0.06 s; graph capture allocation 0 bytes (mempool before =
after); VRAM peak 866 MiB own process.

### M1 (N=4): mixed drive, staggered partial resets, L2 every step, evaluation

`m1_n4_001` (pre-fix): envs (s512 replay, s15 replay, s512 `recorded_command_policy_v1`, s15
`recorded_command_policy_v1`), one batched call per step (`step_measured_edge(..., env_mask, actions)`); partial
resets of env k at steps 300/400/500/600 (each env mid-episode, after cutting started), then each env ran a full
1200-step episode (1800 batch steps).

- L2 every step, 4 envs: code monotonicity 0 violations, nonfinite 0, slot overflow 0, invalid 0, min det F
  0.910/0.844/0.899/0.844, range audits 0, grouped-hash query 0 foreign of 33.6 M candidates, mass bitwise
  constant: **PASS**.
- Replay envs: final codes equal Native (0 particles), labelled 37958 / 85786; evaluator PASS.
- Policy envs (real `edge_delta_v1` actions, 1800/1800 `ACCEPTED`): commanded pose vs recorded targets max
  3.2e-11 m / quaternion component 2.5e-10 (device), host frozen path identical; evaluator **PASS** (removed 1.0,
  share 1.0, detached 0.0) with codes different from Native by 3,994 / 4,576 particles (labelled 33,964 / 81,210
  vs 37,958 / 85,786) — expected: ideal tracking feeds the commanded targets, not the measured Native poses.
- Partial reset 9.5 ms mean / 14.3 ms p95 (host call incl. one seed readback); evaluation 0.57–0.69 s per env.
- Throughput 3.65 env-steps/s (1.10 s per batch step; co-tenant SM mean 46 %).
- Post-fix rerun `m1_n4_postfix_001`: identical outcome (§8).

**L4** (`l4_deterministic_002`, post-fix): **BLOCKED**. With `wp.config.deterministic = RUN_TO_RUN` set before any
module is created, the frozen `tlmpm_v5` module fails to compile (CPU too): `_particle_contacts` mixes
`atomic_add` and `atomic_max` on `stats` ("Deterministic mode does not support mixing 'add' and 'max' reductions
on array 'stats' in the same function or kernel"). The batched step launches unchanged kernels from that module
(`_p2g`, `_grid_update`, `_g2p`, `_musl`, `_deformation`) and the pack path builds frozen single solvers, so the
batch cannot be built in this mode. Unblocking needs batched copies of those kernels in a separate module (a
`batched/` change; not done).

### M2 (N=8 / N=16): continuous rollout, partial resets, memory, L3 across copies

Replay drive; env k replays seed 512 (k even) or 15 (k odd); env k is partially reset at step k·stagger, then runs
full episodes back to back; each completed episode is evaluated on the host and the env is reset.

| run | N | batch steps | full episodes | codes = Native | evaluator PASS | L2 | partial reset mean / p95 | VRAM peak (own) | Warp mempool | RSS slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m2_n1_001 (s512) | 1 | 1200 | 1 | 1/1 | 1/1 | PASS | — | 834 MiB | 55.9 MB | n/a |
| m2_n1_002 (s15) | 1 | 1200 | 1 | 1/1 | 1/1 | PASS | — | 834 MiB | 55.9 MB | +15 MiB/1000 steps (12 samples) |
| m2_n8_001 | 8 | 3950 | 24 (3 per env) | 24/24 | 24/24 | PASS (every step) | 4.3 / 7.3 ms | 1378 MiB | 437.1 MB, flat | −45 MiB/1000 steps |
| m2_n16_001 | 16 | 1575 | 16 | 16/16 | 16/16 | PASS (every step) | 5.9 / 7.5 ms | 2064 MiB | 873.1 MB, flat | +55 MiB/1000 steps (2,650 → 2,288 MiB over the run) |

Labelled counts in every M2 episode: 37,958 (s512) / 85,786 (s15) = Native and the P1-0 band. Memory per env
49.5 MB (`memory_bytes`, excl. hash grid). Warp mempool and Torch allocated bytes are constant between samples
(slopes < 1e-6 B per 1000 steps): no device leak; RSS moves by evaluation buffers, without monotone growth.

## 6. Throughput rows

Main columns (full columns incl. contacts, recording, co-tenant statistics, commits in the evidence
`markdown_table`). All SHARED_GPU (a foreign compute process on GPU1 at up to 99 % SM in every window); warm;
timing = wall of each `env.step*` call with a device synchronize before and after, strict int32 readback inside;
L2 monitor, host samples, resets and evaluation outside.

| run | boundary | N | particles/env | control Hz | material dt | cameras | env-steps/s | aggregate sim-time/wall | VRAM peak | reset p95 | physics verdict | warm/cold | co-tenant SM mean | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m0_s512_001 (pre-fix) | batched_headless | 1 | n/a | 50 | 1e-4 s (200 substeps) | none | 3.508 | 0.0702 | 866 MiB | n/a | PASS | warm | 45 % | 6ba47b7 |
| m0_s15_001 (pre-fix) | batched_headless | 1 | n/a | 50 | 1e-4 s (200 substeps) | none | 3.288 | 0.0658 | 866 MiB | n/a | PASS | warm | 47 % | 7e827a7 |
| m0_s512_postfix_001 | batched_headless | 1 | n/a | 50 | 1e-4 s (200 substeps) | none | 3.499 | 0.0700 | 866 MiB | n/a | PASS | warm | 47 % | 138c937 |
| m0_s15_postfix_001 | batched_headless | 1 | n/a | 50 | 1e-4 s (200 substeps) | none | 3.267 | 0.0653 | 866 MiB | n/a | PASS | warm | 49 % | d0a09ff |
| m1_n4_001 (pre-fix) | batched_headless | 4 | n/a | 50 | 1e-4 s (200 substeps) | none | 3.647 | 0.0729 | 1086 MiB | 0.0143 s | PASS | warm | 46 % | 9bf4d21 |
| m1_n4_postfix_001 | batched_headless | 4 | n/a | 50 | 1e-4 s (200 substeps) | none | 3.637 | 0.0727 | 1086 MiB | 0.0069 s | PASS | warm | 47 % | a24ac46 |
| m2_n1_001 | batched_headless | 1 | 102732 | 50 | 1e-4 s (200 substeps) | none | 5.091 | 0.1018 | 834 MiB | n/a | PASS | warm | 24 % | ddfe2c7 |
| m2_n1_001 | batched_headless | 1 | 102732 | 50 | 1e-4 s (200 substeps) | none | n/a | n/a | 834 MiB | n/a | REPORTED_NOT_SCORED | cold | 24 % | ddfe2c7 |
| m2_n1_002 | batched_headless | 1 | 103898 | 50 | 1e-4 s (200 substeps) | none | 4.803 | 0.0961 | 834 MiB | n/a | PASS | warm | 23 % | fe971fb |
| m2_n1_002 | batched_headless | 1 | 103898 | 50 | 1e-4 s (200 substeps) | none | n/a | n/a | 834 MiB | n/a | REPORTED_NOT_SCORED | cold | 23 % | fe971fb |
| m2_n8_001 | batched_headless | 8 | 103315 | 50 | 1e-4 s (200 substeps) | none | 4.961 | 0.0992 | 1378 MiB | 0.0073 s | PASS | warm | 44 % | 359ce0d |
| m2_n8_001 | batched_headless | 8 | 103315 | 50 | 1e-4 s (200 substeps) | none | n/a | n/a | 1378 MiB | n/a | REPORTED_NOT_SCORED | cold | 44 % | 359ce0d |
| m2_n16_001 | batched_headless | 16 | 103315 | 50 | 1e-4 s (200 substeps) | none | 4.819 | 0.0964 | 2064 MiB | 0.0075 s | PASS | warm | 45 % | 35f01d5 |
| m2_n16_001 | batched_headless | 16 | 103315 | 50 | 1e-4 s (200 substeps) | none | n/a | n/a | 2064 MiB | n/a | REPORTED_NOT_SCORED | cold | 45 % | 35f01d5 |
| p1_2_single_ref_s512_001 | single_env_headless | 1 | 102732 | 50 | 1e-4 s (200 substeps) | none | 5.630 | 0.1126 | 694 MiB | n/a | PASS | warm | 24 % | c696378 |
| alt_r1_n1_001 | batched_headless | 1 | 102732 | 50 | 1e-4 s (200 substeps) | none | 3.354 | 0.0671 | 832 MiB | n/a | REPORTED_NOT_SCORED | warm | 46 % | 638a0f3 |
| alt_r1_n8_001 | batched_headless | 8 | 103315 | 50 | 1e-4 s (200 substeps) | none | 4.849 | 0.0970 | 1376 MiB | n/a | REPORTED_NOT_SCORED | warm | 47 % | 638a0f3 |
| alt_r1_n16_001 | batched_headless | 16 | 103315 | 50 | 1e-4 s (200 substeps) | none | 4.752 | 0.0950 | 2064 MiB | n/a | REPORTED_NOT_SCORED | warm | 46 % | 638a0f3 |
| alt_r2_n1_001 | batched_headless | 1 | 102732 | 50 | 1e-4 s (200 substeps) | none | 3.352 | 0.0670 | 832 MiB | n/a | REPORTED_NOT_SCORED | warm | 45 % | b1673df |
| alt_r2_n8_001 | batched_headless | 8 | 103315 | 50 | 1e-4 s (200 substeps) | none | 6.201 | 0.1240 | 1376 MiB | n/a | REPORTED_NOT_SCORED | warm | 35 % | b1673df |
| alt_r2_n16_001 | batched_headless | 16 | 103315 | 50 | 1e-4 s (200 substeps) | none | 7.040 | 0.1408 | 2064 MiB | n/a | REPORTED_NOT_SCORED | warm | 22 % | b1673df |

The co-tenant's SM share decides these numbers: batched N=1 runs at 5.09 / 4.80 env-steps/s with a 23–24 %
co-tenant (`m2_n1_*`) but at 3.35–3.51 with a 45–49 % co-tenant (M0, `alt_r*_n1`). Comparing `m2_n8_001` (4.96,
co-tenant 44 %) or `m2_n16_001` (4.82, 45 %) with `m2_n1_*` (23–24 %) would suggest "no gain" (0.95–1.03×); that
comparison is confounded and is not used. The frozen single material (`p1_2_single_ref_s512_001`, co-tenant 24 %)
gives 5.63 env-steps/s (0.176 s median step) against 5.09 for batched N=1 under similar load: batched N=1 is
~10 % slower than the frozen single env (the extra validity, diagnostics, pad-box and pose-check kernels and the
Torch action/observation/reward work per step).

**Alternation (one batch per process, alternated within a round; `alt_r{1,2}_n{1,8,16}_001`, partial windows of
300/150/100 steps, no evaluation, L2 audits only):**

| round | N=1 env-steps/s (co-tenant SM mean) | N=8 (co-tenant) | N=16 (co-tenant) | N=8 / N=1 | N=16 / N=1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 3.354 (46 %) | 4.849 (47 %) | 4.752 (46 %) | **1.45×** | **1.42×** |
| 2 | 3.352 (45 %) | 6.201 (35 %) | 7.040 (22 %) | 1.85× (confounded: lighter co-tenant) | 2.10× (confounded) |

Round 1 is the only comparison at matched co-tenant load: batching gives **~1.4×** aggregate env-steps/s at N=8–16
over N=1 (inside the ADR 0021 estimate of 1.3–3×), with no further gain from N=8 to N=16. The same ratio follows
from M0/M1/M2 at similar loads (N=1 3.27–3.51 at 45–49 % vs N=8 4.96 at 44 %, N=16 4.82 at 45 %). Round 2's larger
ratios come from the co-tenant easing off and are not claimed.

### 6.1 Profile: one launch per kernel covers all envs

nsys 2024.6.2 graph-node trace (`--cuda-graph-trace=node`), 2 control steps each, `profile_n1_001` /
`profile_n4_001` (s512 copies): **15,434 vs 15,436 kernel instances** for N=1 vs N=4 (each batched kernel 400 =
2 × 200 substeps, `_obstacle_contacts_b` 1200 = 3 boxes, `_grid_update` 800) — one launch covers all envs; the
grid scales instead (102,732 vs 410,928 particles). GPU kernel time per control step (sum of medians):
136 ms (N=1) vs 420 ms (N=4) = 3.08× for 4× the work (`_particle_contacts_b` median 361 µs → 1,190 µs, `_p2g`
97 → 336 µs). A single 103k-particle env already keeps most of the GPU's share busy, so batching reduces kernel time
per env by ~1.3× at N=4, consistent with the ~1.4× end-to-end gain at N=8–16 (§6).

## 7. Next bottleneck

1. **Batching gives only ~1.4× at N=8–16** (matched co-tenant load): the per-env work (≈1e5 particles,
   200 substeps) already fills most of our time slice; kernel time grows nearly linearly with N (3.1× at N=4),
   and N=16 is no faster per env than N=8. Further gain must come from cheaper kernels: `_particle_contacts_b` is 55–56 % of GPU time at N=1 and N=4 (P1-3a's
   `alias_free_v1` measured 1.40–1.45× end to end), then `_p2g` (15–18 %), the hash-grid radix sort (≈10 % at N=1)
   and `_musl`/`_claim_slots`.
2. An EXCLUSIVE_GPU measurement is still missing; every number here is co-tenant dependent.
3. L4 needs batched copies of the frozen v5 grid kernels (separate module) before determinism can be tested.
4. L3 band: the 4-run single-env band under-samples GPU run-to-run variation for seed 15; define the band from
   more runs (or from the batched copies of M2, which agree on codes in 40/40 episodes) before scoring positions.

## 8. Failures, preserved runs, open items

- Preserved, not used as rows: `smoke_m0_001`, `smoke_m1_001` (crashed in JSON output: mixed int/str keys,
  fixed), `smoke_m1_002`, `smoke_m2_001/002`, `smoke_l4_001` and `l4_deterministic_001` (result-key clash, no
  `result.json`; the rerun `l4_deterministic_002` records the BLOCKED reason).
- `m1_n4_postfix_001` (post-fix + guard, commit a24ac46, dirty only in workload-B docs/scripts): identical outcome
  to the pre-fix run — L2 PASS every step, replay envs equal Native (0 particles), policy envs differ by the same
  3,994 / 4,576 particles, 4/4 evaluator PASS, reconstruction 3.2e-11 m; partial reset 4.0 / 6.9 ms (mean / p95),
  3.64 env-steps/s (co-tenant 47 %).
- Open: EXCLUSIVE_GPU rows; L4; Native confirmation of the policy-driven (ideal tracking) episodes; the
  `evaluate()` integrity layer sees only 2 frames (digit-rewrite history between them is checked on the device by
  the L2 monitor in the milestone runs, not by `evaluate()` itself); a device grid created after a batch (e.g. a
  frozen single material on CUDA built later) is detected only at the env's next full reset.
