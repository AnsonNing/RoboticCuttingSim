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

---

# Stages S2 and S3 (2026-11-10, second session)

Evidence: `docs/evidence/rl_ready/p1_1b_s2_s3_001.json` (assembled by `probe_lab_batched_evidence.py --stage s2s3`
from the raw outputs in `artifacts/rl_ready/p1_1b/`). All GPU rows on GPU0 under its RCS flock, `SHARED_GPU`
(foreign SM 50-57 % mean during the S3 rows). The lock was also held for about 45 minutes by another RCS job
(P1-3a), which my queue waited for.

## Status

| Check | Status |
| --- | --- |
| Reward/termination/evaluator wiring in the Lab env | PASS (CPU tests; used by every S3 row) |
| G3 B2 restore vs fresh settle, default mode (3 builds) | PASS within grasp tolerance; warm-start effect not separable (see G3) |
| G3/G1 in deterministic mode for B2 | BLOCKED: under `run_to_run` (with `disable_sensors`) the B2 settle fails, the free tool is ejected |
| G4 batched N=1 B2 vs single Lab env | FAIL against the fixed tolerance for arm joints and hand position; tool pose and fingers within (not an acceptance gate) |
| G5 L0 (CPU, 20 steps, B1 S3 input sequence) | PASS bitwise (no contact in the first 20 steps) |
| G5 L3 (GPU, 1200 steps, 3 single replays per env, one replay per process) | PASS: final codes of every single replay bitwise equal to the batched Lab run (B1 S3, both seeds) |
| S3 TRIM B1 N=2 and N=4 (seeds 512, 15) | PASS: frozen evaluator PASS in every env, outcome metrics equal to Native WP-17 (descriptive) |
| S3 TRIM B2 N=2 and N=4 (point grip profile) | FAIL: grasp lost in every env during the free-space reorientation, evaluator FAIL |
| S3 TRIM B2 with the opt-in `finite_pad_patch_v3` grip | FAIL: env 1 non-finite (strict raise); the non-strict rerun was aborted (step rate collapsed); one GPU build crashed (SIGSEGV) |
| G2 with material, B2 (reference driver) N=1 / N=4 | FAIL (infrastructure): SIGSEGV in the material graph replay after the grasp was lost |
| Configured CUT scene for `tlmpm_v7_2` | NOT_RUN: no declared CUT scene uses `tlmpm_v7_2` (the WP-12 CUT demos declare `tlmpm_v5`); compiling one would be a new, undeclared scene without a Native reference |

## 1. Reward, termination and evaluator wiring

- With material attached the env computes `trim_bond_progress_packed_v1` exactly as workload A
  (`TrimBondProgressReward` over `bonds_from_task` per env, device codes view, invalid penalty), and
  `extras["reward_available"] = True`; masked resets re-baseline the counts of the reset envs.
- `termination_reason` bits: 1 rigid invalid, 2 material invalid (`invalid_reason`), 4 grasp lost (B2 monitor),
  8 IK-failure streak (`max_consecutive_ik_failures`, `live_env` semantics, default off). Truncation at the
  task's step count (1200 for seeded TRIM).
- `evaluate(env_ids)` runs the frozen `goals.evaluate_goal` (via `ConfiguredTaskSpec.evaluate_goal`) on a host
  snapshot of each env, built the same way as `rl.vector_env.VectorCuttingEnv.evaluate`.
- Per-env material tasks: one batch may hold different seeded TRIM scenes (512 and 15) when their rigid scenes
  are uniform (`check_uniform_rigid_scene`).

## 2. S2 — B2 physical grasp with material

### G3: snapshot restore against a fresh settle

N=2, B2, hold actions; env 1 is restored to its settled snapshot at step 10 and compared over 25 steps (0.5 s)
with env 0's first 25 steps after the build settle (both start from the same snapshot state; the restore zeroes
env 1's MuJoCo warm start). Three separately built processes, default mode:

| Run | max hand-relative tool translation diff | rotation diff | drift from snapshot (fresh / restored) |
| --- | --- | --- | --- |
| g3_b2_n2_001 | 4.0e-7 m | 1.05e-6 rad | 37.2 / 37.1 um, 1.08e-3 / 1.08e-3 rad |
| g3_b2_n2_002 | 4.1e-7 m | 1.14e-6 rad | same |
| g3_b2_n2_003 | 4.2e-7 m | 1.14e-6 rad | same |

Settled relative pose spread over the three rebuilds: 3.0e-8 m, 9.8e-8 rad. Everything is far inside the grasp
tolerances (2 mm, pi/60), so G3 passes. The restored-vs-fresh difference (about 4e-7 m) is of the same size as
the default-mode run-to-run band (G1), so the warm-start effect itself is bounded by it but not separated from
it. The deterministic runs that would separate it are BLOCKED (next item).

### Deterministic mode and B2

With `deterministic_mode="run_to_run"` (which requires `MJWarpSolverCfg.disable_sensors=True`) the B2 settle fails
in every world: over the stability window the tool moves 1.43 m relative to the hand at up to 4.7 m/s
(`g0_det_b2_n1_diag_001`, non-strict, accumulators recorded). The same build passes the settle in default mode.
B1 (no free tool) is bitwise deterministic in this mode (S1 G1). Not investigated further here; deterministic-mode
isolation evidence therefore exists for B1 only.

### G4

Task `TRIM-01-R` (the registered ruler task the single env supports, ruler knife v1), physical grasp, `feedback=False`
(robot only), 100 steps of the identical `joint_position_rad` sequence: the first successful single run's settled
arm q plus a sine of amplitude min(0.05, half the joint-limit margin) on joints 2, 4 and 6
(`probe_lab_batched_g4.py`; the batched env gets the sequence through its G4-only `step_joint_targets` harness, its
HOME is the single run's settled arm q). The tolerance was fixed before running: 3 x the single env's own spread.

- The frozen single env's settle is itself marginal on GPU: 3 of 7 single builds raised `FAIL_PHYSICAL_PREGRASP`
  (angular speed gate; `g4_single_002/006/009`; `g4_single_001` hit a joint limit with the first, unbounded
  sequence). The batched settle of this tool failed the same gate in 5 of 6 strict builds (`g4_batched_003..007`).
  The three non-strict batched builds (`g4_batched_ns_001..003`) record the accumulators: translation/rotation
  drift 0.1 mm / 2 mrad (inside the gates), max tool angular speed 0.70 / 0.03 / 1.16 rad/s against the 0.2 rad/s
  gate; ns_002 passed.
- Comparison (`g4_compare_002`, singles 005/007/008/010, batched ns_001..003), max over the 100 steps:

| Quantity | Single 4-run spread | Tolerance (3 x) | Batched vs single | Verdict |
| --- | --- | --- | --- | --- |
| Arm joints | 7.1e-5 rad | 2.1e-4 rad | 3.5e-4 rad | outside |
| Hand position | 2.5e-5 m | 7.5e-5 m | 2.7e-4 m | outside |
| Tool position | 1.6e-3 m | 4.9e-3 m | 1.9e-3 m | within |
| Tool rotation | 5.7e-2 rad | 1.7e-1 rad | 4.2e-2 rad | within |
| Finger q | 1.5e-3 m | 4.6e-3 m | 1.5e-3 m | within |

  The batched arm differs from the single env by about 0.35 mrad (0.27 mm at the hand), more than the single env
  differs from itself. Known differences that can explain it: the batched HOME starts from the single's
  post-settle q (the single env starts from its IK pose), MuJoCo model composition (per-world `{ENV_REGEX_NS}`
  paths and `separate_worlds` against global prims) and the tool's initial pose (hand FK x hand_from_tool against
  reference edge x inv(T_tool_edge)). Not separated further. The free tool's pose agrees within the single env's
  (large) spread.

### G5: material parity on identical edge and pad sequences

The Lab run records, per env and step, the material's input (measured edge, cut digit, handle + two measured pad
boxes) and its final codes and local positions. `probe_lab_batched_g5.py` replays that sequence through the
frozen single `LiveCuttingMaterial` with the Native call order (`update_pad_obstacles` then `advance`).

- L0 (CPU, `g5_l0_s3_b1_n2_001`): `BatchedCuttingMaterial(backend="isaac_lab_newton")` fed the recorded sequence
  through the coupling surface against the single material, 20 steps, both envs: codes and positions bitwise
  equal. The first 20 steps of the TRIM episode do not touch the workpiece, so this L0 covers the input path,
  not contact.
- L3 (GPU, `g5_l3_s3_b1_n2_e{0,1}_r{1,2,3}`, combined in `g5_l3_combined_s3_b1_n2_001`): the full 1200-step
  input sequences of both envs of `s3_b1_n2_001` (seeds 512 and 15, real cutting) replayed through the frozen
  single material, three replays per env, one replay per process. Every replay's final codes equal the batched
  run's final codes (0 differing particles; labelled 33,740 and 80,986, band width 0); max position difference
  2.8e-5 m (seed 512) and 2.1e-4 m (seed 15). Native side (descriptive, from P1-0): the P1-0 headless replay of
  the WP-17 Native measured inputs reproduces the Native final codes exactly (labelled 37,958 / 85,786); the Lab
  batch's inputs are its own closed-loop robot's, so its counts differ from Native's (section 3), while for the
  same inputs the batched material equals the material Native runs. This is the "Native confirmation of
  material parity" of P1-1b: the material class Native runs, on the batch's identical edge and pad sequence.
  B2 inputs were not replayed (no B2 episode completed with a held grasp).

### G2 with material, B2

With the reference driver the B2 grasp is lost (as in S3 B2), and both rows (N=1, N=4) ended in a SIGSEGV inside
the material's CUDA graph replay (`material_batch.advance` -> `wp.capture_launch`, Python stack in
`g2mat_b2_n4_001/stdout.log`) instead of an invalid-env flag. B2 material timing is available from the S3 B2
rows below (recorded driver, 1200 steps completed): 629 ms/step at N=2 and 1025 ms/step at N=4.

## 3. S3 — representative TRIM (WP-17 seeds 512 and 15)

Driver: `recorded_command_policy_v1` (imported from `rl/policies_batched.py`, unchanged): the WP-17 Native
commanded edge targets converted to `edge_delta_v1` actions open loop from the env's own initial commanded pose
(reconstruction residual against the recorded targets 0.10 mm for seed 512, 0.27 mm for seed 15; 1 and 3 clamped
steps). Full 1200 steps, material `tlmpm_v7_2` law 8 at 1 mm / 4 mm / 0.1 ms, 102,732 / 103,898 particles (seeds 512 / 15), feedback
on, strict mode, frozen evaluator per env at the end. Closed-loop robot dynamics (MJWarp) differ from Native
PhysX; the comparison below is descriptive, never an equivalence claim.

| Run | env (seed) | Labelled | Verdict | Components | Largest substrate share | Detached | Removed | Residual height |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native WP-17 | 512 | 37,958 | PASS | 2 | 1.0 | 0.0 | 1.0 | 0.0 |
| Native WP-17 | 15 | 85,786 | PASS | 3 | 1.0 | 0.0 | 1.0 / 1.0 | 0.0 / 0.0 |
| s3_b1_n2_001 | 0 (512) | 33,740 | PASS | 2 | 1.0 | 0.0 | 1.0 | 0.0 |
| s3_b1_n2_001 | 1 (15) | 80,986 | PASS | 3 | 1.0 | 0.0 | 1.0 / 1.0 | 0.0 / 0.0 |
| s3_b1_n4_001 | 0, 2 (512) | 33,740 each | PASS | 2 | 1.0 | 0.0 | 1.0 | 0.0 |
| s3_b1_n4_001 | 1, 3 (15) | 80,986 each | PASS | 3 | 1.0 | 0.0 | 1.0 / 1.0 | 0.0 / 0.0 |
| s3_b2_n2_001 | 0 (512) | 0 | FAIL | 1 | 1.0 | 0.0 | 0.0 | 5.4 mm |
| s3_b2_n2_001 | 1 (15) | 36,760 | FAIL | 1 | 1.0 | 0.0 | 0.0 / 0.0 | 5.0 / 7.3 mm |
| s3_b2_n4_001 | 0..3 | 12,032 / 43,959 / 0 / 39,388 | FAIL | 1 | 1.0 | 0.0 | 0.0 | 0-37 mm |

- **B1** (rigid mount, `reduced_fidelity`): every env passes the frozen evaluator with the same component count,
  removed fraction and residual height as Native; 11 % (512) and 6 % (seed 15) fewer labelled particles than
  Native. The final codes of a seed are bitwise identical across the N=2 and N=4 batches and across duplicate
  envs (`final_codes_sha256` 108cc6fd... / ced88127...), even in default mode. Edge tracking of the recorded
  targets: max 5.8 / 5.3 mm (tracking lag under the 1 mm/step limits), reaction up to 0.19 / 0.14 N.
- **B2** (physical grasp, point contact profile, not reduced): the tool slips in the fingers during the free-space
  reorientation from HOME to the skim pose (8-13 mm slip and 0.05-0.5 rad rotation at steps 20-160 with full
  bilateral contact and no material force), the monitor latches GRASP_LOST at steps 177/181 (termination reason
  4 in every env), and the episodes then cut with a misplaced edge: no protrusion removed, evaluator FAIL. The
  N=4 envs with the same seed diverge (default-mode run-to-run differences amplified by the slipping grasp).
  Native PhysX holds the same grasp configuration (WP-17 `grasp_stabilization` PASS, then PASS outcome).
- **B2 with `finite_pad_patch_v3`** (the single Lab env's existing opt-in finite-pad profile, now accepted by the
  batch): `s3_b2patch_n2_001` crashed with SIGSEGV during the build; `_002` (strict) raised `InvalidBatchEnvs` for env 1
  (seed 15: NONFINITE_STATE, INVALID_MEASURED_POSE, NONFINITE_WRENCH, INVALID_OBSTACLE_BOXES); the non-strict
  `_003` was aborted after 43 min (the B2 point-profile row took 13 min). No outcome for this profile.
- Throughput of the S3 rows (warm, strict, recorded driver, SHARED_GPU): B1 686 ms/step at N=2 (2.9 env-steps/s),
  1299 ms/step at N=4 (3.1 env-steps/s); B2 629 / 1025 ms/step (3.2 / 3.9 env-steps/s). Own VRAM peak 974-1108 MiB.

## 4. Needed changes outside my files

- `batched/material_batch.py`: an env whose measured edge / tool trajectory leaves the valid domain (here after a
  lost grasp) must be flagged invalid, not crash the graph replay (SIGSEGV in `capture_launch`). Suggested: use
  `pose_status` (known before the substeps) to neutralize the substep inputs of invalid envs and guard grid index
  computations against non-finite particle positions.
- `batched/errors.py`: nothing new (the eight codes are merged).
- `rl/vector_env.py`: the dispatch list of the S1 section; plus `tasks=` (per-env seeds) and the
  `termination_reason` / `evaluate(env_ids)` surface of the Lab env.

## 5. Warp 1.17 HashGrid static-descriptor hazard (P1-3a finding): census of P1-1b processes

Warp 1.17.0 replays a CUDA graph's HashGrid build with the descriptor of the last grid built on the host, so a
process holding two or more CUDA `wp.HashGrid`s runs particle contacts on the wrong grid. CPU builds have no such
static copy (P1-3a report §2-§3.0). Grid constructors in the code these processes import: `tlmpm_v5.py:798`
(single material solver bed, created at `LiveCuttingMaterial.reset`), `batched/tlmpm_v7_2_batch.py:613` (one
grouped grid per batch), and in Newton only `ModelBuilder.finalize` for models with particles (`particle_grid`,
`builder.py:12590`; the rigid workload B model has none), plus `particle_surface.py` / `remesh.py` (not used).
MuJoCo-Warp and Isaac Lab construct none (grep of `.deps/isaaclab-venv/.../mujoco_warp` and
`.deps/IsaacLab/source`).

| Process type | CUDA HashGrids | How checked | Verdict |
| --- | --- | --- | --- |
| Lab env with material (S1 G2 material rows, S3 B1/B2, G2 B2, patch rows) | 1 (the batch); the per-env single materials used for packing are reset on the CPU device (2 CPU grids for N=2) | instrumented run `hashgrid_check_b1_n2_001` (every `wp.HashGrid.__init__` recorded with device and creator) on the same code path | not affected |
| Lab env robot-only (S1 G0/T2/G1/G2, G3, G4 batched) | 0 | no material code path; `g4_batched_ns_*` rows record 0 | not affected |
| G4 single Lab env (frozen `IsaacLabCuttingEnv` with its `LiveCuttingMaterial`) | 1 | `g4_single_*/g4.json` `hashgrids` | not affected |
| G5 L0 (CPU) | 0 CUDA (all CPU) | CPU device only | not affected |
| G5 L3 single replays | 1 per process | rerun one replay per process (`g5_l3_*_e{0,1}_r{1,2,3}`), each records `hashgrids.cuda_count = 1` | not affected (the multi-replay design was replaced before it ran) |

So no P1-1b process held two CUDA grids and no run is labelled `INVALID_MULTI_GRID_PROCESS`. Every probe now
records its grid census in its row, and the G5 band combiner refuses a multi-grid input.

---

# B2 grasp follow-up (2026-11-10, branch `claude/rl-b2-grasp`)

Evidence: `docs/evidence/rl_ready/p1_1b_b2_001.json` (assembled by `scripts/rl_ready/make_p1_1b_b2_evidence.py`
from `artifacts/rl_ready/p1_1b_b2/`; raw outputs not committed). New probes: `scripts/rl_ready/probe_lab_b2_grasp.py`
(robot-only B2 diagnosis: every MuJoCo-Warp tool contact per control step with its geoms, depth, tool-frame
position, contact-frame force and elliptic-cone usage, hand-frame slip/rotation, finger q, tool/hand velocity;
eager stepping, host reads every step, labelled DIAGNOSTIC, not a timing row) and `scripts/rl_ready/probe_material_ood.py`
(material only, out-of-domain inputs on top of recorded S3 inputs). GPU rows: physical GPU0 under the RCS flock,
`CUDA_VISIBLE_DEVICES=0`, all `SHARED_GPU` (other projects' processes and other RCS agents' jobs). CPU rows: Warp-CPU
MJWarp (`cpu_test_mode`) in the Isaac Lab venv. Same software as above (Isaac Lab 3.0 kit-less, Newton 1.6.1,
MJWarp 3.12.0, Warp 1.17.0).

## Status

| Check | Status |
| --- | --- |
| Root cause of the B2 grasp loss | FOUND, measured on CPU and GPU: MJWarp box-box contact manifold collapse between the finger pads and the handle |
| Opt-in versioned profile `mjwarp_grip_v1` (MuJoCo `box_box` collider + `impratio` 10) | IMPLEMENTED; default (B1, point profile) and existing profiles unchanged |
| Robot-only recorded TRIM sequence, N=2 (seeds 512, 15), GPU, graph, strict, `mjwarp_grip_v1` | PASS: grasp held (HOLDING 1196/1200 steps per env, max slip 1.51 / 1.59 mm, max rotation 14.1 / 15.2 mrad) |
| S3 TRIM B2 N=2 (seeds 512, 15) with material, frozen evaluator, `mjwarp_grip_v1` | PASS in both envs, no termination |
| S3 TRIM B2 N=4 (seeds 512, 15, 512, 15), `mjwarp_grip_v1` | PASS in all four envs, no termination |
| B2 settle gate, G4 ruler-knife setting, strict GPU builds, `mjwarp_grip_v1` | PASS 3 of 3 (point profile: 5 of 6 strict builds FAIL on angular speed; same root cause) |
| B2 settle under deterministic `run_to_run`, `mjwarp_grip_v1` | PASS (point profile: tool ejected 1.43 m); B2 G1/G3 deterministic isolation not run |
| Slip comparable to Native | NO: Native PhysX <= 0.93 um / 9.6e-5 rad; MJWarp with the profile 1.6 mm / 15 mrad (inside the monitor's 2 mm / pi/60 warning gates, three orders of magnitude above Native) |
| Material crash after a lost grasp | Crash site recorded (graph replay); the out-of-domain input that precedes it reproduced and FIXED with an env-local input-domain guard (`INPUT_OUT_OF_DOMAIN` + device-side hold); CPU tests incl. bitwise isolation PASS; GPU see section 4 |

## 1. Root cause

**Native reference** (WP-17 recordings of seeds 512 and 15, the S3 command source): the PhysX contact sensor
reports 8 tool contacts (4 per finger pad, at the corners of the pad overlap) and the grasp monitor is HOLDING in
all 1200 steps, slip <= 0.72 / 0.93 um, rotation <= 7.4e-5 / 9.6e-5 rad.

**MJWarp, default point profile** (`diag_cpu_point_001`, `diag_gpu_point_001`: N=1, seed 512, recorded commands,
robot only, 250 steps). Both finger pads and the tool handle are boxes (pad half extents 8.8 x 7.6 x 8.2 mm, handle
12 x 11 x 30 mm); condim 3, mu 0.6, solref (0.01, 1), solimp (0.99, 0.995, 0.01, 0.5, 2); tool 88.6 g with its
centre of mass 55 mm down the blade; elliptic cones, impratio 1, 50 iterations, 2 ms substeps. Per control step:

- Steps 1-21: 4 contacts per pad at the pad corners, 18 N normal force per pad, cone usage 0.03-0.06, depth 0.09 mm,
  finger q 10.97 mm (handle half thickness 11 mm). The friction capacity (2 x 0.6 x 18 N = 21.6 N) is 25 x the tool
  weight (0.87 N): neither the friction coefficient nor the 20 N finger force limit is the problem.
- From step 22 (0.33 rad into the HOME -> skim reorientation, commanded 0.785 rad/s and 40 mm/s, the Native
  commands) the manifold drops to 2, from step 26 to 1-2 contacts per pad, on diagonally opposite pad edges (left
  pad at tool x = +9 mm, right pad at x = -8 mm). Every remaining contact is on the friction-cone boundary (usage 1.0),
  the reported depth jumps from 0.09 to 2.25 mm, the fingers close 1.9 mm (GPU: down to 6.3 mm, the pads 4.7 mm
  inside the handle), and the tool is kicked (16.7 rad/s at step 26 while the hand turns at 0.7 rad/s). It then
  slides 13.9 mm (GPU 14.7 mm) along the handle axis and twists 50-70 mrad (GPU up to 0.38 rad) about the pad normal,
  and keeps chattering after the motion stops (7-41 rad/s with the hand at rest). The pads had 4 contacts in only
  9 % of the 250 steps. CPU and GPU agree.

**Why.** MuJoCo-Warp 3.12.0 sends BOX-BOX pairs to the general convex narrowphase (GJK/EPA plus multicontact
polygon clipping) unless `DisableBit.NATIVECCD` is set (`mujoco_warp/_src/collision_driver.py:78`, `_narrowphase`
lines 866-870). That path writes ONE EPA depth and ONE EPA normal frame to every manifold point
(`collision_convex.py`: `frame = make_frame(witness1[0] - witness2[0])`, then `for i in range(ncollision):
write_contact(..., dist, ..., frame, ...)`). For two near-parallel faces under a slight roll it yields 1-2 points on
one edge of each pad, all with the deepest depth; the squeeze becomes a couple about the handle axis, the points
flip between edges and the solver sees deep, saturated, alternating contacts. MuJoCo's dedicated `box_box` collider
(`collision_primitive_core.py:589`, selected when NATIVECCD is disabled) returns up to 8 points with per-point depth
and normal; with it the same run keeps 4 contacts per pad in 99.8 % of the steps, cone usage 0.07-0.4, finger q
10.96 mm and tool angular speed <= 0.75 rad/s (`diag_cpu_noccd_001`).

**Remaining creep -> impratio.** With the `box_box` collider alone the tool still turns slowly about the pad normal
at rest (+1 mrad at step 100 to +33 mrad at step 250, about 10 mrad/s, at cone usage 0.07-0.1): creep of MuJoCo's
regularized (soft) friction under the 16 mN m gravity torque. `impratio` 10 (MuJoCo's documented elliptic-cone
setting that makes friction constraints harder than normal ones "without increasing the actual friction
coefficient") removes it: rotation <= 12.5 mrad over 400 steps, recovering to 1-4 mrad at rest
(`diag_cpu_noccd_imp10_001`); impratio 100 gives no further gain (`diag_cpu_noccd_imp100_001`).

**Candidates measured** (CPU, seed 512, robot only):

| Run | Contacts per pad (mean, min) | Max slip | Max rotation | Monitor |
| --- | --- | --- | --- | --- |
| `diag_cpu_point_001` point profile (default) | 2.08, 1 | 13.9 mm | 71 mrad (GPU 379 mrad) | SLIPPING from step 26 |
| `diag_cpu_patch_001` `finite_pad_patch_v3` (condim 4, torsional mu r) | 1.75, 1 | 23.6 mm | 425 mrad | LOST at step 77 |
| `diag_cpu_impratio10_001` impratio 10 only | 3.07, 1 | 12.7 mm | 508 mrad | SLIPPING |
| `diag_cpu_noccd_001` NATIVECCD off only | 4.0, 4 | 1.0 mm | 33 mrad, creeping | HOLDING (2 one-step flags) |
| `diag_cpu_noccd_imp10_001` NATIVECCD off + impratio 10 (= `mjwarp_grip_v1`) | 3.99, 2 | 1.04 mm | 12.5 mrad | HOLDING (2 one-step flags) |
| `diag_cpu_noccd_imp100_001` NATIVECCD off + impratio 100 | 3.93, 2 | 1.05 mm | 12.8 mrad | HOLDING (2 one-step flags) |

Torsional friction (condim 4, the patch profile) does not help: it acts on the collapsed manifold. Pyramidal cones
were not tried (the configured elliptic cone was kept). Not varied, so not claimed either way: solref/solimp, finger
drive stiffness and force limit, contact margin/gap (Newton zeroes box margins for NATIVECCD compatibility; gap
1 mm), tool mass/inertia, solver iterations, substep length. The reorientation uses the Native command sequence; no
slower motion was needed.

## 2. Profile `mjwarp_grip_v1` (opt-in, versioned)

`lab_scene.CONTACT_SOLVER_PROFILES["mjwarp_grip_v1"]`, selected with
`BatchedLabOptions(tool_mount="physical_grasp_v1", grip_contact_profile="mjwarp_grip_v1")` (probes:
`--grip-profile mjwarp_grip_v1`; B1 refuses it with `UNSUPPORTED_TOOL_MOUNT`). Authoring is exactly the point profile
(same USD, geometry, mass, friction 0.6, condim 3, solref/solimp, finger drive 2000 N/m / 20 N). Two MuJoCo-Warp
contact-solver options change:

- `DisableBit.NATIVECCD` on the built MJWarp model: box-box pairs use MuJoCo's `box_box` collider. Newton exposes no
  option for it, so `IsaacLabBatchedCuttingEnv._apply_contact_solver_profile` sets it during bind, before the first
  step and before any CUDA graph capture (it refuses if a graph exists), and `reset()` re-checks it on the live model;
- `impratio = 10` in the solver cfg.

Both are global options of the B2 MuJoCo model: they also apply to the pads or handle against the static table and
fixture boxes and to every elliptic contact. `env.receipt()["contact_solver"]` reads the flags back from the live
model; `fidelity()["listed_differences"]["contact_solver_profile"]` lists the profile. It is not a weld, does not
raise friction and does not slow the motion; it selects a per-point contact model and harder friction
regularization, and is not a calibrated grip (`physical_calibration: NOT_VALIDATED`).

## 3. Results with the profile

**Robot only** (`robot_b2grip_n2_001`, GPU, graph captured, strict, N=2, seeds 512 / 15, recorded WP-17 commands,
1200 steps): settle PASS in both worlds; HOLDING in 1196 of 1200 steps per env, the four single-step SLIPPING flags
being step 1, step 103 (end of the reorientation) and two cut-phase velocity transitions, each a one-substep loss of
bilateral contact; max slip 1.51 / 1.59 mm, max rotation 14.1 / 15.2 mrad; 33 ms/step. The CPU run of the same
sequence (`diag_cpu_noccd_imp10_n2_full_001`) agrees to three digits. The slip comes from impulsive changes of the
commanded velocity (0.67 mm transient at the motion start, 0.3-0.5 mm at the cut-phase transitions) and slow creep
at rest (about 0.01 mm/s); Native shows neither.

**S3 TRIM B2 with material** (`s3_b2grip_n2_001`, GPU, strict, N=2, seeds 512 / 15, `tlmpm_v7_2` law 8 at
1 mm / 4 mm / 0.1 ms, 102,732 / 103,898 particles, feedback on, recorded commands, frozen evaluator per env at the end):

| Run | env (seed) | Labelled | Verdict | Components | Largest substrate share | Detached | Removed | Residual height |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native WP-17 | 512 | 37,958 | PASS | 2 | 1.0 | 0.0 | 1.0 | 0.0 |
| Native WP-17 | 15 | 85,786 | PASS | 3 | 1.0 | 0.0 | 1.0 / 1.0 | 0.0 / 0.0 |
| s3_b1_n2_001 (B1, reduced fidelity) | 512 / 15 | 33,740 / 80,986 | PASS / PASS | 2 / 3 | 1.0 | 0.0 | 1.0 | 0.0 |
| s3_b2_n2_001 (B2 point profile, earlier) | 512 / 15 | 0 / 36,760 | FAIL / FAIL | 1 / 1 | 1.0 | 0.0 | 0.0 | 5.4 / 5.0-7.3 mm |
| **s3_b2grip_n2_001 (B2 `mjwarp_grip_v1`)** | 0 (512) | 38,220 | **PASS** | 2 | 1.0 | 0.0 | 1.0 | 0.0 |
| **s3_b2grip_n2_001 (B2 `mjwarp_grip_v1`)** | 1 (15) | 81,034 | **PASS** | 3 | 1.0 | 0.0 | 1.0 / 1.0 | 0.0 / 0.0 |

- Grasp during the cut: HOLDING in 1196/1200 steps per env, max slip 1.51 / 1.59 mm, max rotation 14.0 / 15.1 mrad;
  termination reason 0 in both envs; no material or rigid flag; reward progress 1.0 / 1.0, off-target 0 / 0.
- Edge tracking of the recorded targets max 6.7 / 6.3 mm; material reaction up to 0.16 / 0.06 N.
- 511 ms/step mean (p50 353 ms, 3.9 env-steps/s), `SHARED_GPU` (foreign SM 32 % mean), own VRAM peak 976 MiB, one
  CUDA HashGrid. `graph_captured` reads false because the Newton graph is dropped when the material attaches and
  recaptured on the next step (as in the earlier B2 rows).
- Descriptive only: the closed-loop MJWarp robot differs from Native PhysX; B2 labels 0.7 % more (seed 512) and 5.5 %
  fewer (seed 15) particles than Native; no equivalence claim. G5 replays of these B2 inputs were not run.
- Provenance: this row ran in the main worktree after the material guard of section 4 was written but before it was
  committed (row `commit.tracked_src_or_scripts_modified: true`); its `workload_b_sources_sha256` (660af37c...) equals
  the digest of the committed sources at `a930a8c`. The guard set no flag (the recorded inputs peak at 0.19 m/s and
  1.17 rad/s).

**S3 N=4 with the profile** (`s3_b2grip_n4_001`, seeds 512 / 15 / 512 / 15, same settings, strict, clean commit
`4bf1663`): frozen evaluator PASS in all four envs, termination reason 0 everywhere, HOLDING 1196/1200 steps per env,
max slip 1.51-1.59 mm, max rotation 14.0-15.1 mrad. Labelled 38,220 / 81,033 / 38,220 / 81,034; the seed-512 final
codes are bitwise identical in envs 0 and 2 and equal to `s3_b2grip_n2_001` env 0 (`final_codes_sha256` 243a3f8b...);
seed 15 env 3 equals `s3_b2grip_n2_001` env 1 (0316c347...), env 1 differs by one labelled particle (default-mode
MJWarp run-to-run band through the contact-loaded material; B1 duplicates were bitwise equal). 1099 ms/step (3.6
env-steps/s), `SHARED_GPU` (foreign SM 49 % mean), own VRAM peak 1108 MiB.

**Settle reliability and deterministic mode with the profile** (the S2 findings "settle fails the angular-speed gate
in 5/6 strict batched builds" and "B2 under `run_to_run` ejects the tool"). The G4 setting (registered `TRIM-01-R`,
ruler knife, HOME = the settled q of `g4_single_005`, hold driver; `probe_lab_b2_grasp.py --registered-task`) on CPU:
the point profile FAILS the settle gate on the tool's angular speed, 1.16 rad/s against 0.2 rad/s
(`settle_cpu_trim01r_point_001`, the same value as the GPU row `g4_batched_ns_003`), with the pad manifold dropping to
2 points; with `mjwarp_grip_v1` the settle PASSES with 5e-5 rad/s, rotation drift 0.36 mrad and 4 points per pad in
every step (`settle_cpu_trim01r_grip_001`). On the seeded TRIM tool the settle accumulators drop from 2.2e-3 to
1.8e-4 rad/s and from 4.4 to 0.38 mrad (`diag_cpu_point_001` against `prof_cpu_mjwarp_grip_v1_001`). So the S2 settle
flakiness has the same root cause. GPU, strict, the G4 batched setting with the profile (`probe_lab_batched_g4.py --grip-profile mjwarp_grip_v1`):
`g4_settle_b2grip_001`, `_002` and `_003` PASS (3 of 3 strict builds) with max tool angular speed 4.6e-5 rad/s and
rotation drift 0.36 mrad, identical accumulators in all three builds (the point-profile rows `g4_batched_ns_001..003`: 0.70 / 0.026 / 1.16 rad/s,
two of three failing; strict builds: 5 of 6 failing, `g4_batched_003..007`).

Deterministic mode (`g0_det_b2grip_n1_001`, `run_to_run`, N=1, graph, 20 steps with a 0.5 N edge force): settle PASS
(angular speed 1.8e-4 rad/s, drift 0.085 mm / 0.38 mrad), slip 0.14 mm and 0.85 mrad over the 20 steps, graph safety
PASS. With the point profile the same build ejected the tool (1.43 m, `g0_det_b2_n1_diag_001`). Deterministic B2 is
therefore no longer blocked by the grasp; the B2 G1/G3 deterministic isolation runs themselves were not done here.

## 4. Material crash after a lost grasp

**Recorded crash site.** `g2mat_b2_n4_001` (reference driver, point profile, grasp lost) died with SIGSEGV in
`BatchedCuttingMaterial.advance -> wp.capture_launch(self.graph)` (`material_batch.py:683` at `b1673df`), i.e. in the
replay of the 200-substep material graph; `g2mat_b2_n1_001` and the crashed patch build left no stack. Between the
robot and that graph only finiteness of the pose and positivity of the boxes was checked.

**What reaches the material after a lost grasp.** The recorded S3 B2 point-profile inputs (`s3_b2_n2_001`) reach
6.7 / 8.7 rad/s and 0.59 m/s edge speed after the loss (B1: 0.9 rad/s, 0.17 m/s). Nothing bounded the edge: a 1e7 m
edge jump of one env on top of real S3 inputs (`probe_material_ood.py --scenario huge`) flings that env's particles
to ~1e7 m and its pending wrench to 9e8 N (fed back to the robot), sets no invalid bit, and the step time grows
without bound: CPU 3.2 -> 6 -> 21 -> 31 s per step (`ood_cpu_huge_001`), GPU 0.6 -> 174 s per step at the default
1 mm spacing (`ood_gpu_huge_preguard_001`, killed by the operator after that step). By code reading (Warp
`native/hashgrid.h`: `hash_grid_index` clamps out-of-range cells) far-flung particles share hash cells, so the
particle-contact query degenerates towards all-pairs; this matches the earlier aborted `s3_b2patch_n2_003` whose
step rate collapsed. These pre-guard probe rows started from the recorded initial edge, so their first step also
jumped to the step-150 edge (harmless there: the edge was above the material); later rows start from step 149.

**The crash configuration with the guard.** `g2mat_b2point_n1_postguard_001` repeats the crashed `g2mat_b2_n1_001`
configuration (B2 point profile, reference driver, material, N=1, 1200 steps; GPU, graph) with the guard, in training
mode (`--non-strict`) so the run continues after the flag: the edge leaves the domain at about step 19 (the
kicked tool, edge angular speed up to 6.7 rad/s), the env gets `INPUT_OUT_OF_DOMAIN` (material invalid reason 64,
termination reason 6 = MATERIAL_INVALID | GRASP_LOST from step 100 on, grasp LOST at step 36), its material inputs
are held (material reaction 0 N throughout), and all 1200 steps complete at 325 ms/step (p95 370 ms), `SHARED_GPU`,
no crash. The same configuration without the guard (`g2mat_b2point_n1_preguard_001`, pinned worktree at `41d9405`,
strict as the original rows) is queued to check whether the SIGSEGV reproduces; NOT_RUN at the time of writing.

**Fix (`material_input_domain_v1`, `material_batch.py`).** A device kernel after the pose interpolation and before
the substep graph checks per env: edge within 1.5 m of the workpiece frame (Franka reach 0.855 m plus a base 0.47 m
from the workcell origin), per-control-step edge speed <= 3.4 m/s and <= 5 rad/s (twice the Franka Panda Cartesian
limits of 1.7 m/s and 2.5 rad/s), pad-box centres within 0.5 m of the edge and half extents in (0, 0.25] m (measured:
0.096 m, 0.03 m). A violation sets `INPUT_OUT_OF_DOMAIN` (bit 64, sticky until reset; strict mode raises
`InvalidBatchEnvs`, the Lab env terminates with `MATERIAL_INVALID`) and, as for an invalid measured pose or invalid
boxes, HOLDS that env's substep inputs from then on: blade at its last in-domain substep pose with zero twist, cut
disarmed, pads at the last in-domain boxes. Everything runs on the device (no host sync, no recapture, graph-safe);
an in-domain env only records its hold state, so its solver inputs and results are bitwise unchanged; `reset_masked`
restores the hold state. Offline check of the recorded inputs: B1 S3 and B2 `mjwarp_grip_v1` S3 never leave the
domain; the B2 point-profile S3 envs would have been flagged at steps 113 and 177 (grasp lost at 177/181).

- CPU tests `tests/test_batched_input_domain.py` (4 PASS): a 1e7 m teleport and a 5 m/s swing of env 1 set only env
  1's bit, hold its inputs (particles finite, |x| < 0.2 m), env 0 stays bitwise equal to a batch that never saw the
  bad input in every step, the hold persists when the input returns to range, a masked reset clears it; an oversized
  pad box is replaced by the last in-domain boxes; strict mode raises with env 1 / `INPUT_OUT_OF_DOMAIN`.
- Probe rows: `ood_cpu_{none,huge,sweep,spin}_postguard_002` (CPU, real S3 inputs, isolation against a clean batch):
  only the bad env is flagged, from the bad step on, particles unmoved, step time unchanged, env 0 bitwise equal to
  the clean batch in every step. `ood_gpu_huge_postguard_001` (GPU, graph, 1 mm): env 1 flagged at the bad step,
  particles unmoved, 0.25-0.28 s/step (pre-guard 174 s). (`ood_cpu_*_postguard_001` are superseded: they started from
  the recorded initial edge, whose jump to step 150 the guard correctly flagged in both envs.)
- Existing suites with the guard: batched material (parity, isolation, partial-reset coverage over `vars()`,
  coupling, layout, HashGrid guard, frozen sources) 147 PASS; workload A vector env (Isaac Sim venv, CPU) 16 PASS,
  1 skipped.

## 5. Tests and commands

- `.venv`: `tests/test_batched_input_domain.py` 4 PASS (81 s); batched material suites 147 PASS (326 s);
  `tests/test_rl_lab_grip_profile.py` 1 PASS + 1 skipped (Lab part NOT_RUN there); lab scene/kernels/IK/rigid
  isolation 42 PASS; `tests/test_repo_naming.py` PASS.
- Isaac Lab venv (D-09 overlay): `tests/test_rl_lab_grip_profile.py` 2 PASS (58 s: the profile reaches the model,
  4-point pads in >= 95 % of 150 reorientation steps, slip < 2 mm and rotation < pi/60, and the default point
  profile collapses to <= 2 points and slides > 5 mm); workload B Lab suites 52 PASS (183 s).
- Isaac Sim venv: `tests/test_vector_env_api.py tests/test_vector_env_gpu_stream.py tests/test_rl_policies_batched.py`
  16 PASS, 1 skipped (CPU).

```
# diagnosis (CPU; GPU: --device cuda --gpu-index 0 under the flock)
CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaaclab-venv/bin/python \
  scripts/rl_ready/probe_lab_b2_grasp.py --device cpu --scene-seeds 512 --steps 250 --non-strict \
  [--grip-profile mjwarp_grip_v1] [--solver-set impratio=10] [--mjw-disable NATIVECCD] --out artifacts/rl_ready/p1_1b_b2/<run>
# S3 / robot-only rows
CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock \
  .deps/isaaclab-venv/bin/python scripts/rl_ready/probe_lab_batched_gpu.py --probe g2 --num-envs 2 \
  --tool-mount physical_grasp_v1 --grip-profile mjwarp_grip_v1 [--material tlmpm_v7_2 --evaluate --record-inputs] \
  --driver recorded --scene-seeds 512 15 --steps 1200 --gpu-index 0 --out artifacts/rl_ready/p1_1b_b2/<run>
# material input domain
PYTHONPATH=src .venv/bin/python scripts/rl_ready/probe_material_ood.py --device cpu --scenario huge --steps 5 \
  --isolation --out artifacts/rl_ready/p1_1b_b2/<run>
```

## 6. Open

- Native-level grasp stiffness is not reached: 1.5 mm / 15 mrad against Native's 1 um / 1e-4 rad. The remaining
  slip comes from impulsive commanded-velocity changes and slow soft-friction creep. Not tried: solref/solimp of the
  pad-handle pair, finger drive gains, smoothing the velocity feed-forward (that would change the action contract),
  MuJoCo's noslip solver (not implemented in MJWarp 3.12).
- `mjwarp_grip_v1` changes two global MJWarp options of the B2 model. Its effect on contacts other than the grip
  (Franka self-collision, pads/handle against table and fixtures) was only observed through the TRIM rows (no other
  tool contact occurred), not tested separately. B1 is unaffected (the profile is refused there).
- The SIGSEGV itself was never reproduced on demand; the crash site is the material graph replay. The guard removes
  the out-of-domain inputs that preceded it and the step-time collapse, and the crash configuration now completes;
  that the guard removes every SIGSEGV path is not proven.
- Not done with the profile: G1/G3 isolation in deterministic mode for B2 (now possible: the deterministic settle
  passes), G4 against the single Lab env (the frozen
  single env keeps the point profile), G5 replays of B2 inputs, an exclusive-GPU timing row, wiring
  `grip_contact_profile` through `rl.vector_env.open_vector_env`. ADR 0022 is not updated here (§5.1 table would list
  the profile as a B2 option).
