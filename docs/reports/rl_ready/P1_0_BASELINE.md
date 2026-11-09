# P1-0 — baseline and benchmark runner

Date: 2026-11-10. Branch `claude/rl-ready-rubber-physics`. Plan: `docs/RL_READY_PLAN.md` §2–§4
(decisions D-03..D-08) and ADR 0021 (L3 band, obstacle label, derived sequential-N baseline).
Compact evidence: `docs/evidence/rl_ready/p1_0_baseline_001.json` (rows, parity, L3 band, kernel split,
cold start, telemetry) and `docs/evidence/rl_ready/p1_0_inventory_001.json`. Raw outputs (not committed):
`artifacts/rl_ready/p1_0/<run>_NNN/` and logs `artifacts/rl_ready/logs/p1_0_*.log`.

## Status

| Check | Status |
| --- | --- |
| Inventory committed | PASS |
| `configs/rl_benchmark/workloads_v1.json` A–E (B, C, E declared NOT_RUN) | PASS |
| Benchmark row schema, Markdown table, GPU telemetry sampler | PASS (CPU tests) |
| Measured-trajectory extraction (both WP-17 seeds, 1200 steps) | PASS |
| single_env_headless A/D rows, warm and cold, both seeds | PASS (measured; mostly `SHARED_GPU`) |
| Kernel split (4 representative phases, seed 512) | PASS (measured, `SHARED_GPU`) |
| single_env_native rows | PASS (taken from WP-17 evidence, `source: wp17_evidence`, not rerun) |
| batched_headless rows | NOT_RUN (no batched runtime in P1-0) |
| Replay parity vs the 121 Native samples, L3 band (4 runs per seed) | PASS — reported, not scored against a tolerance |
| Frozen seeded_trim evaluator on the replay end state | PASS 3/3 per seed (same metrics as Native) |
| Sequential-N single-env baseline | DERIVED_NOT_RUN |

## Runtime and versions (decision D-08)

GPU material runs: `.deps/isaacsim-venv/bin/python` (Python 3.12.12, Warp 1.17.0, NumPy 2.3.1, Torch
2.11.0+cu128), `PYTHONPATH=src`, **no Kit application started**, `CUDA_VISIBLE_DEVICES=1` (device `cuda:0`
inside the process), `TMPDIR=/mnt/data/ningan/tmp`, each job under
`flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock`. GPU: NVIDIA GeForce RTX 5090 (index 1, 32607
MiB), driver 570.211.01, CUDA 12.8. The Isaac Sim venv has no `robotic_cutting_sim` installed and
`run_baseline.py` puts `<worktree>/src` first on `sys.path`; the main checkout has no `rl` subpackage, so
every run imported this worktree (verified by printing the module files; recorded as `import_origin` in
the evidence). CPU tests: `.venv` (Warp 1.17.0, NumPy 2.5.4, no Torch) with `PYTHONPATH=src`.
The tlmpm_v7_2 / contacts / frontier / live_material / material_diagnostics / isaac_sim_runner sources are
byte-identical (sha256) to the WP-17 run commit f55f94a. Inventory: three runtimes probed, all import Warp
1.17.0 and Newton 1.6.1; Isaac Lab 28.0.0 in the Lab venv; Isaac Sim 6.2.0.0 in the Sim venv (metadata
only, not imported).

## Commands

```bash
.venv/bin/python scripts/rl_ready/inventory.py --out docs/evidence/rl_ready/p1_0_inventory_001.json
# chain 1 (/mnt/data/ningan/tmp/p1_0_chain.sh) and chain 2 (p1_0_chain2.sh), each line under the GPU1 flock with
# CUDA_VISIBLE_DEVICES=1 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python ...
scripts/rl_ready/run_baseline.py --mode replay  --seed 512 --gpu-index 1 --warm-resets 5            # replay_s512_001
scripts/rl_ready/run_baseline.py --mode replay  --seed 15  --gpu-index 1 --warm-resets 5            # replay_s15_001
scripts/rl_ready/run_baseline.py --mode kernels --seed 512 --gpu-index 1 --kernel-repeat 3          # kernels_s512_001
scripts/rl_ready/run_baseline.py --mode replay  --seed 512 --gpu-index 1 --kernel-cache fresh \
    --max-steps 20 --warm-resets 2 --run-name coldcache_replay_s512                                   # coldcache_replay_s512_001
scripts/rl_ready/run_baseline.py --mode replay  --seed {512,15} --gpu-index 1 --warm-resets 2 \
    --evaluate --run-name band_replay_s{512,15}                                                        # band_replay_s*_00{1,2,3}
PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_p1_0_evidence.py --replay replay_s512_001 replay_s15_001 \
    band_replay_s15_002 --kernels kernels_s512_001 --coldcache coldcache_replay_s512_001 \
    --band 512=replay_s512_001,band_replay_s512_001,band_replay_s512_002,band_replay_s512_003 \
           15=replay_s15_001,band_replay_s15_001,band_replay_s15_002,band_replay_s15_003 \
    --out docs/evidence/rl_ready/p1_0_baseline_001.json
```

Seeds: scene seeds 512 and 15 (WP-17 `rcs_trim_random_sim_v1`, controller `trim_skim_c3`), material seed 42.
Wall: each full replay process 179–294 s; kernels run 147 s; cold-cache run 49 s. Smoke runs
`smoke_replay_s512_001` (15 steps) and `smoke_kernels_s512_001` (1 step) are kept and not used as rows.
No run failed.

## What the replay feeds the material (`robotic_cutting_sim.rl.replay`)

Native (`isaac_sim_runner.LiveIsaacSimCutting.step`, configured-task path) per 50 Hz control step, after 10
PhysX substeps: `update_pad_obstacles([handle_box_edge(profile), *pad_boxes_edge(pads, pad body poses, edge)])`,
then `advance(edge, step*0.02, cutting=command.cut_enabled, cut_index=command.cut_index)` with
`edge = compose_pose(tool body pose, tool_from_edge)` (world, XYZ+XYZW), then
`numerical_diagnostics("device_reduction_v1")`. Reset: `configure_pad_obstacles(initial boxes, [None, left,
right])` then `reset(initial edge, seed=42)`. Material: `LiveCuttingMaterial(CouplingConfig(0.02, 1e-4, dx
0.004, feedback=True), task=task_from_payload(compiled_task), max_cuts=2, cutting_law=8, solver="tlmpm_v7_2",
v7_2_optimizations=True)`, slots_per_node 8, spacing 0.001 m, one CUDA graph of 200 substeps, no cut-event
recorder (`surface_profile=current_volume_v1`).

- RECORDED and used directly: edge pose (`privileged_diagnostics.measured_tool_pose`), resolved cut flag and
  digit (`active_cut`/`cut_index`, equal to `submitted.cut_enabled/cut_index` at all 1200 steps), reset edge
  (`setup.json home_reset.measured_after_pregrasp.tool_edge_pose_xyzw`), initial boxes
  (`finger_geometry_00.json boxes_edge`), all options (`setup.json` config, `compiled_task.json`, `material.json`).
- RECONSTRUCTED with the identical function: per-step finger pad boxes from the finger body poses in
  `robot_state.npz` (same control step) and the recorded edge. Label `replay_obstacles:
  handle_and_measured_pads_v1`. Verification: reset boxes rebuilt from row 0 are bitwise equal to the
  recorded `configure_pad_obstacles` input; the knife body poses of the same rows reproduce the recorded
  `advance` edge at all 1200 steps (max 2.8e-17 m, 3e-8 rad), i.e. the rows are the PhysX snapshot Native
  read; per-step contact on/off equals Native at all 1200 steps (101 contact steps for seed 512) and both
  Native and replay finger-pad reactions are zero at every step. Limitation: the pads never touch the
  material in these two episodes, so pad geometry is not exercised beyond "no contact".
- The reaction is computed but drives nothing (no robot): the replay is open loop on the measured poses.

## Performance (single env; boundaries never merged)

Main columns (full 24-column table with commit, contacts, recording, co-tenant statistics in the evidence
`markdown_table`):

| boundary | workload | warm/cold | N | particles/env | control Hz | material dt | cameras | env-steps/s | sim-time/wall | VRAM peak | reset p95 | physics verdict | GPU sharing | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_env_native | D-s512 | warm (action loop) | 1 | 102732 | 50 | 1e-4 s (200 substeps) | global D455 + wrist D405 | 4.040 | 0.0808 | 5542 MiB | n/a | PASS | EXCLUSIVE_GPU | wp17_evidence |
| single_env_native | D-s512 | cold (factory→finalize) | 1 | 102732 | 50 | 1e-4 s | same | 3.194 | 0.0639 | 5542 MiB | n/a | PASS | EXCLUSIVE_GPU | wp17_evidence |
| single_env_native | D-s15 | warm | 1 | 103898 | 50 | 1e-4 s | same | 3.866 | 0.0773 | 5638 MiB | n/a | PASS | EXCLUSIVE_GPU | wp17_evidence |
| single_env_native | D-s15 | cold | 1 | 103898 | 50 | 1e-4 s | same | 3.075 | 0.0615 | 5638 MiB | n/a | PASS | EXCLUSIVE_GPU | wp17_evidence |
| single_env_headless | A/D-s512 | warm | 1 | 102732 | 50 | 1e-4 s | none | 5.928 | 0.1186 | 758 MiB | 0.185 s | PASS | SHARED_GPU | replay_s512_001 |
| single_env_headless | A/D-s512 | cold | 1 | 102732 | 50 | 1e-4 s | none | 5.773 | 0.1155 | 758 MiB | 0.352 s | PASS | SHARED_GPU | replay_s512_001 |
| single_env_headless | A/D-s15 | warm | 1 | 103898 | 50 | 1e-4 s | none | 5.629 | 0.1126 | 758 MiB | 0.304 s | PASS | SHARED_GPU | replay_s15_001 |
| single_env_headless | A/D-s15 | cold | 1 | 103898 | 50 | 1e-4 s | none | 5.481 | 0.1096 | 758 MiB | 0.347 s | PASS | SHARED_GPU | replay_s15_001 |
| single_env_headless | A/D-s15 | warm | 1 | 103898 | 50 | 1e-4 s | none | **7.419** | **0.1484** | 758 MiB | 0.149 s (2 resets) | PASS | **EXCLUSIVE_GPU** | band_replay_s15_002 |
| single_env_headless | A/D-s15 | cold | 1 | 103898 | 50 | 1e-4 s | none | 7.177 | 0.1435 | 758 MiB | 0.289 s | PASS | EXCLUSIVE_GPU | band_replay_s15_002 |
| batched_headless | A | — | 1 / 4 / 16 | — | 50 | 1e-4 s | none | NOT_RUN | NOT_RUN | — | — | NOT_RUN | — | — |

Row definitions:

- Native warm = 1200 / sum of the 1200 action calls (PhysX + IK + material graph + recording + cameras on
  capture); Native cold = 1200 / factory→finalize wall (incl. Kit start 66–67 s, evaluation, media). VRAM =
  job peak on GPU1 from the WP-17 GPU verdict; sharing from its process list (only the display server
  besides the job; per-process utilisation was not sampled then). Driver not recorded in WP-17 (UNVERIFIED).
- Headless warm = steps 2..1200: `update_pad_obstacles` + `advance` (graph launch, device sync, reaction
  readback) + `device_reduction_v1` diagnostics; no cameras, no robot, no recording. The 121 parity
  readbacks (`history_state`, 2.6 s per episode) are outside the timed calls. Headless cold = process start
  → last step without the parity readbacks (process start → first step done 5.6–6.4 s with the user Warp
  kernel cache). Reset = warm resets in the same process (task particle sampling, solver construction,
  graph capture); cold reset = the first one. VRAM = own-PID memory from `nvidia-smi` (incl. CUDA context);
  Warp mempool high water 52 MB after an episode, 105 MB after the warm resets. Robot DOF 0.
- Telemetry: `nvidia-smi --query-gpu`, `--query-compute-apps`, `-q -d PIDS` and `pmon` per-PID SM at ~1 Hz;
  `SHARED_GPU` if any foreign compute or non-display graphics process or foreign SM / baseline utilisation
  > 5 %. During most runs another user's graphics processes on GPU1 used up to 3.6–5.7 GiB and 14–99 % SM.

Run-to-run throughput spread of the 8 full replays (same code and inputs): seed 512 5.90–6.45 env-steps/s
(all SHARED_GPU, foreign SM mean 14–21 %); seed 15 4.40 (SHARED, foreign SM mean 40 %, max 99 %, foreign
compute process present), 5.63, 7.38 (SHARED by a graphics process, foreign SM mean 0.3 %) and 7.42
(EXCLUSIVE). The headless warm step on an uncontended GPU is ~134 ms (seed 15), close to the Native WP-17
material graph wall (seed 15: graph_total 164.8 s / 1200 = 137 ms; seed 512: 128.8 ms). Co-tenancy alone
moves the headless number by up to 1.7x, so only the EXCLUSIVE row is a clean single-env material number;
the SHARED rows are kept and labelled.

Sequential-N single env (DERIVED_NOT_RUN, ADR 0021): N sequential steps cost N × the warm single step
(e.g. 4 × 169 ms = 0.67 s, 16 × 169 ms = 2.70 s for seed 512 shared); aggregate env-steps/s stays at the
single-env value (5.9 shared, 7.4 exclusive). This is the reference a batched runtime must beat.

Cold start with an empty Warp kernel cache (`coldcache_replay_s512_001`, 20 steps only, never a row): process
start → first step 44.3 s, of which the first reset 38.7 s; 8 modules compiled, 37.6 s reported, including
`tlmpm_v7_1` 14.7 s and `tlmpm_v5` 5.1 s that a tlmpm_v7_2 run loads only as dependencies of the module
chain. With the user cache the first reset is 0.29–0.35 s; the very first run of this worktree compiled two
modules (`tlmpm_v7_2_contacts`, `_frontier`, 4.4 s) whose hash differed from the cache.

## Kernel split (seed 512, `kernels_s512_001`, SHARED_GPU, foreign SM mean 21 %, max 70 %)

`profiling.StageProfiler(kernels=True)` (CUDA activity GPU time; eager launches of the identical kernel
sequence; graph restored afterwards) over 3 steps (600 substeps) per phase. Phases from the recording:
first Native contact step 229, cutting 255–395.

| phase (steps) | GPU kernel time / substep | / control step | `_particle_contacts` | `_p2g` | `_musl` | `_claim_slots` | memsets (10/substep) | `fixture_contacts` | `_obstacle_contacts` (3/substep) | `_grid_update` (2) | label/stroke group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no contact (20–22) | 812 µs | 162.4 ms | 386 µs (48 %) | 139 | 52 | 36 | 33 | 25 | 17 | 18 | 37 |
| approach, between contact and cut (242–244) | 831 µs | 166.1 ms | 387 µs (47 %) | 130 | 59 | 41 | 34 | 27 | 24 | 22 | 35 |
| active cutting (325–327) | 824 µs | 164.9 ms | 388 µs (47 %) | 122 | 58 | 45 | 33 | 29 | 17 | 23 | 37 |
| post cut (797–799) | 860 µs | 172.0 ms | 405 µs (47 %) | 126 | 59 | 49 | 36 | 27 | 19 | 31 | 33 |

Eager wall 0.21–0.22 s per control step vs 0.17–0.19 s for the neighbouring graph steps (launch overhead
of the eager mode). The kernel time hardly depends on the phase: the cost is the per-substep sweep over all
~103k particles, not tool contact or the cut frontier (the WP-17 frontier optimisation removed that).

## Replay parity vs Native (L3 band, 4 processes per seed)

Each run compares all 121 Native samples (every 0.2 s) with the replay state. Reference arrays (mass,
reference positions, stencils) equal Native exactly; total mass 0.154098 kg (s512) / 0.155847 kg (s15)
constant; invariants PASS in every run (finite, min detF 0.910 / 0.844 > 0, no digit rewrites).

| seed | runs | final codes = Native | final labelled digit 0 (Native) | digit 1 | max position diff over time | final max / mean position diff | samples with any code difference | replay↔replay final code diffs | evaluator (3 runs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 4 | 4/4 (0 particles) | 37958 (37958) | 0 | 1.39e-5 – 3.85e-5 m | 1.4e-5 – 3.8e-5 m / 8.8e-7 m | 1 sample (t = 6.4 s, 1 particle) in 2 runs, then identical | 0 | PASS 3/3; removed p0 1.0; substrate share 0.9999999999992; detached 0.0 |
| 15 | 4 | 4/4 (0 particles) | 85786 (85786) | 0 | 1.96e-4 – 4.62e-4 m | 6.3e-5 – 1.2e-4 m / 1.4e-6 m | transient: samples 45–50 (t = 9.0–10.0 s), 1–3 particles, in all 4 runs; none at the end | 0 | PASS 3/3; removed p0, p1 1.0; substrate share 0.9999999999992; detached 0.0 |

Native with the same evaluator call gives the identical verdict and metrics (PASS, share 0.9999999999991656,
removed 1.0), matching the recorded WP-17 `independent_evaluation.json`. Per-step reaction: contact on/off
identical at all 1200 steps; total impulse max |diff| ≤ 0.22 % of the episode peak (s15), ≤ 0.015 % (s512).
The position difference starts at 4.5–5.2e-8 m at t = 0.2 s (float atomics; the same order as the
WP-16/WP-17 Native difference) and grows over the episode; it never changed a final label here.

The open WP-17 issue (seed 512 final codes differ between WP-16 and WP-17 Native, 4266 particles) is not
resolved by this: the replay reproduces the **WP-17** Native final codes exactly in 4/4 runs, which says the
WP-17 material input is reproducible from the recording; it says nothing about the WP-16 difference and is
not a physics-equivalence claim (the replay has no reaction feedback into the robot).

## Next bottleneck

1. `_particle_contacts` (tlmpm_v5 kernel used by tlmpm_v7_2): ~47 % of material GPU time in every phase,
   a hash-grid query of radius 2.4 r for every particle every substep, although contact can only act
   across a cut or between reference-distant particles. Candidate for P1-3 (hotspot evidence now exists);
   any change needs replay parity and invariants as above.
2. `_p2g` (15–17 %) and `_musl` + `_claim_slots` (~12 %); ten memsets per substep (~4 %).
3. For RL throughput the structural fact is that N sequential envs scale linearly in wall time (derived
   above). `pmon` reports our process SM-busy 75–88 % of the time on the uncontended runs, but that metric
   counts time with any kernel resident, not occupancy; whether one 103k-particle env leaves enough of a
   5090 idle for batching to pay off is UNVERIFIED until the P1-1 batched N=1/4/16 measurement. Batching
   (P1-1) comes before kernel tuning, as the plan says.
4. Cold start: an empty kernel cache costs ~38 s mostly for modules (v5, v7_1) that a v7_2 run does not
   execute; worth a lazy-load check later (not a physics change).

## Files

- `src/robotic_cutting_sim/rl/{__init__,benchmark,telemetry,replay}.py`
- `configs/rl_benchmark/workloads_v1.json`
- `scripts/rl_ready/{inventory,run_baseline,make_p1_0_evidence}.py`
- `tests/test_rl_benchmark.py`, `tests/test_rl_replay.py`
- `docs/evidence/rl_ready/p1_0_{inventory,baseline}_001.json`, this report.

## Open / not determined

- WP-17 Native GPU driver: not recorded in WP-17 evidence (UNVERIFIED; the machine shows 570.211.01 now).
- Per-process utilisation of the WP-17 Native runs was not sampled; their EXCLUSIVE label rests on the
  process list only.
- Kernel split ran on a shared GPU; absolute kernel times are inflated by co-tenancy, shares are more robust.
  No exclusive kernel-split run.
- Seed 15 kernel split NOT_RUN.
- Finger-pad box geometry is verified by construction and by zero pad reactions, not by a pad-material
  contact (none occurs in these episodes).
- The evaluator was applied in-process after the replay (same function and inputs as
  `seeded_trim.evaluate_recording`, without `surface_state`, which feeds quality only), not in a fresh process.
