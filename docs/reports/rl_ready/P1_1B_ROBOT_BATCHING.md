# P1-1b — batched robot workload B (Isaac Lab Newton, N worlds): stages S0 and S1

Date: 2026-11-10. Branch `claude/rl-ready-rubber-physics`. Design: ADR 0022 (accepted, decision D-10) with its
"Implementation notes" (S0, S1). Compact evidence: `docs/evidence/rl_ready/p1_1b_s1_001.json`. Raw outputs (not
committed): `artifacts/rl_ready/p1_1b/<run>_NNN/` (`row.json`, `states.npz` / `trace.npz`, `stdout.log`).

## Status

| Check | Status |
| --- | --- |
| S0 C1-C4 (Warp CPU, `.venv` and Isaac Lab venv) | PASS |
| S0 T1 model receipt and reset-mask audit, T3 coupling contract (Lab on Warp CPU) | PASS |
| IK-rejection hold in the env (unreachable commanded pose, CPU) | PASS |
| T3 with the real `BatchedCuttingMaterial(backend="isaac_lab_newton")` (CPU, coarse task) | PASS (bitwise) |
| G0 smoke, graph capture safety (B1 and B2, N=1; B1 N=4) | PASS |
| T2 host syncs (B1 and B2, N=4, 20 steps) | PASS (one strict-mode int32 read per step, no Torch sync) |
| G1 N=4, `deterministic_mode="run_to_run"`: identical copies, run-to-run, perturb env 2, reset env 1 | PASS (bitwise) |
| G1 sub-claim "reset env equals a fresh run of that env" | NOT_RUN (scripted actions depend on the absolute step) |
| G1 without deterministic mode | band reported (not bitwise, see below) |
| G2 robot-only N=1/4/16 (1200 steps, staggered resets) | MEASURED, `SHARED_GPU`, `reduced_fidelity` |
| G2 with material N=1/4 (1200 steps, reference driver) | MEASURED, `SHARED_GPU`, `reduced_fidelity` (B1) |
| G3, G4, G5 | NOT_RUN |

All GPU rows: NVIDIA GeForce RTX 5090, physical index 0, driver 570.211.01, `CUDA_VISIBLE_DEVICES=0`, under
`flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock`, `TMPDIR=/mnt/data/ningan/tmp`, `PYTHONPATH=src`,
`.deps/isaaclab-venv/bin/python` (Isaac Lab 3.0 kit-less, Newton 1.6.1, MJWarp 3.12.0, Warp 1.17.0, Torch 2.11).
GPU0 was shared the whole time (another project's training/eval processes, 65-90 % foreign SM utilisation,
PID 642659 untouched); every row is labelled `SHARED_GPU` by `robotic_cutting_sim.rl.telemetry`. VRAM headroom was
checked before each job (>= 4 GB left free after the job estimate). Authoritative rows share the workload B source
hash recorded in each row (`workload_b_sources_sha256`); earlier rows are kept and marked superseded in the evidence.

CPU test results at the final source state: `.venv` 40 passed, 1 skipped (the T module, NOT_RUN there), 12 s;
Isaac Lab venv with the D-09 overlay 50 passed, 155 s (C1-C4, T1, T3 synthetic and real material, Lab-level
isolation, B2 settle/restore, IK hold). These CPU suites ran on the host while the N=1 material row was running
on GPU0 (CPU contention on a 64-core host; noted, not corrected for).

## Commands

```
# S0 / CPU tests
CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python -m pytest -p no:cacheprovider tests/test_rl_lab_*.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaaclab-venv/bin/python -m pytest -p no:cacheprovider tests/test_rl_lab_*.py
# S1 GPU probes (one per process), e.g.
CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock \
  .deps/isaaclab-venv/bin/python scripts/rl_ready/probe_lab_batched_gpu.py --probe g0|t2|g1|g2 [--num-envs N] \
  [--tool-mount physical_grasp_v1] [--deterministic --variant base|perturb|reset] [--no-graph] \
  [--material tlmpm_v7_2 --driver reference] --steps S --gpu-index 0 --out artifacts/rl_ready/p1_1b/<run>
PYTHONPATH=src .venv/bin/python scripts/rl_ready/probe_lab_batched_gpu.py --probe compare --inputs A B ... --out <dir>
PYTHONPATH=src .venv/bin/python scripts/rl_ready/probe_lab_batched_evidence.py --out docs/evidence/rl_ready/p1_1b_s1_001.json
```

## G0 — smoke and graph capture safety

A constant 0.5 N external edge force in every env; per step the applied tool impulse must be
`10 x 2 ms x 0.5 N = 0.01 N s` (the wrench kernel ran in all 10 graph-replayed substeps) and, in B2, the contact
integral must count 10 substeps (the post-step contact callback ran inside the graph every substep).

| Run | Graph | Safety | Max impulse deviation | Warm ms/step | Cold: reset / first step |
| --- | --- | --- | --- | --- | --- |
| g0_b1_n1_002 | captured | PASS | 7.1e-10 | 66.4 | 0.15 s / 1.85 s |
| g0_b1_n1_nograph_002 | eager | PASS | 7.1e-10 | 235.0 | 0.12 s / 1.39 s |
| g0_b2_n1_002 | captured (during the settle) | PASS, settle `PASS_PHYSICAL_PREGRASP` | 7.1e-10 | 64.6 | 3.90 s (2 s settle) / 0.46 s |
| g0_b2_n1_nograph_002 | eager | PASS, settle PASS | 7.1e-10 | 190.0 | 19.74 s / 0.56 s |
| g0_b1_n4_001 | captured | PASS | 7.1e-10 | 36.3 | 0.07 s / 1.47 s |

Graph against eager (same steps, separate processes, N=1, default non-deterministic mode): not bitwise; max joint
difference 1.6e-11 (B1) and 5.0e-8 (B2) over 20 steps (`compare_g0_b*_graph_vs_eager_002`). The first build of
the session compiled the Warp/MJWarp kernels (28 s first step, `g0_b1_n1_001`); later rows use the warm cache.

## T2 — host syncs

`torch.cuda.set_sync_debug_mode("warn")` plus counted `wp.array.numpy` / `wp.synchronize*` over 20 warm steps at
N=4: the only host read is `check_valid` (strict mode, one int32 per step: 20 of 20), no Torch synchronization, in
B1 and B2 (`t2_b1_n4_003`, `t2_b2_n4_003`). The first T2 run found one Torch sync source in `_pre_physics_step`
(index assignment with a device index tensor); fixed with masks (`t2_*_001` FAIL kept, `_002`/`_003` PASS).
T2 does not see syncs inside Isaac Lab or Newton that bypass these two hooks.

## G1 — isolation on the GPU

N=4, 40 steps, B1, scripted `edge_delta_v1` actions, per-step joint and body state saved.

| Comparison (authoritative `_002`) | Deterministic `run_to_run` | Default mode |
| --- | --- | --- |
| Identical copies (worlds 1-3 vs world 0, same run) | bitwise | not bitwise, max 9.5e-7 |
| Run to run (two base runs) | bitwise (all worlds) | not bitwise, max 9.5e-7 |
| Env 2 perturbed: worlds 0, 1, 3 vs base | bitwise; world 2 differs (8.0e-2) | not run |
| Env 1 reset at step 30: worlds 0, 2, 3 vs base | bitwise; world 1 differs from step 30 | not run |

So batching isolation holds bitwise on the GPU in deterministic mode. Without it, MJWarp's contact atomics make
even identical worlds and repeated runs differ at the 1e-7 level after 40 steps; that is the run-to-run band, not
an isolation failure. The bitwise claim is made only for deterministic mode.

## G2 — robot-only timing (B1, reduced fidelity)

Labels: `reduced_fidelity: true`, reasons `rigid_tool_mount_no_grasp`, `finger_opening_fixed_nominal`,
`robot_only_no_material`. 1200 control steps (24 s simulated per env), a staggered masked reset of one env every
20 steps (60 resets), strict mode (one int32 read per step, so each step ends synchronized), scripted actions,
CUDA graph of one 2 ms substep replayed 10 times per control step, 0 particles/env, 50 Hz control, no material dt,
cameras none, recording none, warm (cold start reported separately in the rows, about 9 s process-to-first-step).

| N | ms/step mean (p50 / p95) | env-steps/s | aggregate sim-time / wall | reset p95 | own VRAM peak | VRAM slope | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64.0 (62.9 / 87.4) | 15.6 | 0.31 | 15.9 ms | 714 MiB | ~0 | SHARED_GPU (foreign SM 79 % mean) |
| 4 | 45.0 (42.3 / 71.8) | 88.8 | 1.78 | 11.6 ms | 714 MiB | ~0 | SHARED_GPU (77 %) |
| 16 | 50.3 (49.5 / 74.6) | 317.8 | 6.36 | 11.8 ms | 714 MiB | ~0 | SHARED_GPU (76 %) |

The step time hardly depends on N: the robot step is latency-bound (10 Python-driven graph launches, about 250
IK launches, per-step kernels) and the GPU was time-shared with a 65-90 % foreign load, so absolute numbers are
pessimistic and noisy (an earlier repeat at N=1 gave 55.8-61.8 ms). Own VRAM is the CUDA context plus Warp's
mempool reservation and did not grow over 1200 steps with 60 resets (nvidia-smi samples at 1 s).

## G2 with material

Run after the robot-only gates passed, with the fixer's `BatchedCuttingMaterial(backend="isaac_lab_newton")`
(`8535ef4`). B1 (labels `reduced_fidelity: true`, `rigid_tool_mount_no_grasp`, `finger_opening_fixed_nominal`), seeded
TRIM seed 512 compiled at the default 1 mm particle spacing (102,732 particles/env), `tlmpm_v7_2`, law 8, 4 mm grid,
material dt 0.1 ms, 200 material substeps per 50 Hz control step, material CUDA graph captured, feedback on (the
material's reaction wrench of step k is applied in every substep of interval k+1), strict mode (two int32 reads per
step: material and rigid validity), 1200 control steps without resets, cameras none, recording none, warm.
Driver: device reference tracking of `task.reference_action` through `edge_delta_v1` (not a policy and not the
WP-17 controller). Rewards are not computed.

| N | ms/step mean (p50 / p95) | env-steps/s | aggregate sim-time / wall | own VRAM peak | VRAM slope | Foreign SM mean | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 693 (707 / 965) | 1.44 | 0.029 | 866 MiB | 0.003 MiB/s | 70 % | SHARED_GPU |
| 4 | 611 (430 / 857) | 6.54 | 0.131 | 1026 MiB | 0.004 MiB/s | 29 % | SHARED_GPU |

- Engagement: the edge tracked the reference within 7.2 mm (max over the episode, every env), entered the
  material in the cut phase and the reaction reached 0.45 N (78 steps with |F| > 0.01 N per env, 72 armed
  steps); no material or rigid validity flag was set.
- The two rows ran under different co-tenant load (70 % against 29 % foreign SM), so the 4.5x aggregate gain from
  N=1 to N=4 is not a clean scaling number. The N=1 material step (693 ms) is about five times the P1-0
  single-env headless figure (129-137 ms) because the GPU was time-shared; an exclusive-GPU row is still needed.
- In the default (non-deterministic) mode the four identical worlds of the N=4 run diverge slightly
  (reaction difference up to 4.4e-4 N, world 0 of N=4 against N=1 up to 3.1e-4 N, edge 1.8e-6 m): the MJWarp
  run-to-run band of G1 feeds back through the contact-loaded material. These rows make no isolation or parity
  claim; G5 (material parity) is NOT_RUN.

## Findings and fixes during S1

- **Table geometry.** The scene used the legacy 1.8 x 1.2 m Lab table; Native places the table of flat-board tasks
  from `task.table_box_world`. With the legacy table the hand of the TRIM skim pose hit the table and the edge
  stalled 6 cm above the cut plane (no material contact). Fixed (`359ce0d`); robot-only gates were rerun on the
  corrected scene (`_002`/`_003` rows); the earlier rows are superseded, not deleted.
- **Captured callbacks.** No Python-side branching after binding in callbacks captured into the substep graph;
  the Newton graph is recaptured when the material is attached after a capture (B2).
- **Host sync** in the finger-target assignment (T2), fixed with masks.

## Not done / open

- G3 (B2 snapshot restore against a fresh settle; warm-start effect), G4 (batched N=1 B2 against the single Lab
  env), G5 (material parity on identical edge/pad sequences, L0/L3): NOT_RUN.
- Rewards and the frozen evaluator are not wired into the batched env (`reward_available` false); the vector-env
  dispatch is not wired (below).
- G1 "reset env equals a fresh run of that env" needs an action schedule relative to the env's episode step.
- B2 rows other than G0/T2 (G1/G2 for B2) are stage S2.

## Dispatch `rl/vector_env.open_vector_env(backend="isaac_lab_newton")` needs (not edited here)

1. Accept `backend="isaac_lab_newton"` instead of raising `UNSUPPORTED_BACKEND`; plumb `tool_mount`
   (`rigid_mount_v1` | `physical_grasp_v1`) and map `material` to `BatchedLabOptions(material="tlmpm_v7_2" | None)`.
2. Call `rl.lab_batched_env.open_lab_batched_env(num_envs, task=..., tool_mount=..., device=..., material=...)`,
   which returns `(cfg, factory)`; enter `isaaclab.app.sim_launcher.launch_simulation(cfg, {"headless": True,
   "visualizer": []})` for the vector env's lifetime (exit in `close()`), then `env = factory()`.
3. Map `reset()` to `env.reset()` and `reset(env_ids=...)` to `env.reset_envs(mask)` with a device bool mask
   (per-env `seeds` only reach the material; the rigid HOME is deterministic, B2 restores its snapshot);
   `step(action[N,8])` to `env.step(action)`; observations are already `rl_state_obs_v1`.
4. Rewards: compute `trim_bond_progress_packed_v1` from `env.coupling` (material) as for workload A; until then
   the env returns zeros with `extras["reward_available"] = False`.
5. Refuse cameras/vision profiles (`UNSUPPORTED_SENSOR`), `autoreset="same_step"` with final obs, and
   `device="cpu"` (CPU is `cpu_test_mode` only).
