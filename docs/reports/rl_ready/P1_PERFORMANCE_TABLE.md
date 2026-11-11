# P1 performance table — exclusive-window rows on GPU1

Date: 2026-11-10 (18:18–21:00 KST). Branch `claude/rl-exclusive` (from integration commit `691b571`). Plan:
`docs/RL_READY_PLAN.md` (Phase 1 acceptance performance table, decision D-03 boundaries, D-07 telemetry). Compact
evidence: [`docs/evidence/rl_ready/p1_exclusive_001.json`](../../evidence/rl_ready/p1_exclusive_001.json) (rows,
per-row classification with the samples it rests on, comparisons, the 24-column `markdown_table`). Raw outputs (not
committed): `/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/exclusive_001/`.

## Status

| Item | Rows | Status | GPU label |
| --- | --- | --- | --- |
| single_env_native (WP-17 Native, from P1-0 evidence, not rerun) | D-s512, D-s15 warm/cold | PASS (copied) | EXCLUSIVE_GPU (process list only, WP-17) |
| single_env_headless, frozen single `LiveCuttingMaterial` replay (P1-0 command) | s512, s15 warm/cold | PASS — invariants PASS, final codes = Native (0 particles), labelled 37,958 / 85,786 = Native | EXCLUSIVE_GPU |
| batched workload A s512 (chain A), `frozen_v1` and `alias_free_v1` | N = 1/4/8/16 × 2 | PASS (measured; physics `REPORTED_NOT_SCORED`, every env `invalid_reason` 0) | EXCLUSIVE_GPU (8/8) |
| batched workload A s15 N=8 | `frozen_v1` (2 runs), `alias_free_v1` | PASS (measured, `REPORTED_NOT_SCORED`) | `alias_free_v1` EXCLUSIVE; `frozen_v1` first run SHARED_GPU (transient foreign graphics process, 0 % SM), repeat EXCLUSIVE |
| batched CUT `bar_flat_board_tlmpm_v7_2_v1` (P1-5 runner) | N = 1/8/16 | PASS — frozen sever evaluator PASS 1/1, 16/16 (2 episodes × 8), 16/16; L2 PASS | EXCLUSIVE_GPU (3/3) |
| single_env_headless CUT C2 reference (added so batched CUT has an exclusive single-env reference) | N=1 warm/cold | PASS — evaluator PASS, 2 pieces | EXCLUSIVE_GPU |
| workload B1 rigid mount, recorded commands (S3 command) | N = 1/4/8/16 (+ N=4 repeat) | PASS — frozen evaluator PASS in every env (1, 4, 4, 8, 16), no terminal bit | N=1, N=8, N=4 repeat EXCLUSIVE; N=4 first run and N=16 SHARED_GPU (transient foreign graphics processes, 0 % SM) |
| workload B2 `mjwarp_grip_v1`, physical grasp | N=4 | PASS — evaluator PASS 4/4, final grasp monitor status HOLDING in every env (max slip 1.51–1.59 mm, max rotation 14.0–15.1 mrad), no terminal bit | EXCLUSIVE_GPU |
| Chain A rows classified from their telemetry | 8 | done (§4) | — |
| NOT_RUN | B1 N=16 exclusive repeat; CUT N=4; B2 at N ≠ 4; workload A s15 at N ≠ 8; batched cold rows | see §7 | — |

20 of the 23 measured runs (one process each) are EXCLUSIVE_GPU under the strict rule; the 3 SHARED_GPU runs (a_n8_s15_default, b1_n4, b1_n16) had
**no foreign compute process and 0 % foreign SM** in every sample — each saw one or two short-lived (< 2 s) 2–7 MiB foreign
*graphics* contexts, identified for b1_n16 as another project's EGL device probe (`egl_probe/build/test_device`). The
first two were repeated in the window (both repeats EXCLUSIVE); b1_n16 (37 min) was not.

## 1. Setup

- **Code:** every benchmark process ran from the detached worktree `/home/ningan/RoboticCuttingSim-rl-review` at
  `691b57157c89dd876ebdd1b377fd25aeba4a55ce` with no tracked change (`git status --porcelain` empty at chain start; every
  run records commit 691b571 with `git_dirty: false` / `dirty: False` / `tracked_src_or_scripts_modified: false`). Nothing was edited there. Scripts that only orchestrate,
  sample or assemble live in this branch: `scripts/rl_ready/p1_exclusive_chain_{a,b,c}.sh` (chain A copied verbatim from
  the raw tree), `exclusive_gpu_sampler.py`, `gpu_process_catcher.sh`, `make_p1_exclusive_evidence.py`.
- **GPU:** NVIDIA GeForce RTX 5090, nvidia-smi index 1 (UUID `GPU-a278df44-…`), 32,607 MiB, driver 570.211.01, CUDA 12.8
  (driver API). `CUDA_VISIBLE_DEVICES=1` (device `cuda:0` inside the process), `TMPDIR=/mnt/data/ningan/tmp`. Every row
  ran under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock`, held for each whole chain; one process per row,
  one device `wp.HashGrid` per process (the harness/runners record the census: 1 after build, 0 after release; Lab rows
  `SINGLE_DEVICE_GRID_OR_NONE`). GPU1 memory used peaked at 3,598 MiB (≥ 28 GiB always free).
- **Runtimes:** material rows `.deps/isaacsim-venv/bin/python` (Python 3.12.12, Warp 1.17.0, Torch 2.11.0+cu128, NumPy
  2.3.1, no Kit); workload B `.deps/isaaclab-venv/bin/python` (Isaac Lab Newton/MJWarp, kit-less). Warp user kernel cache (warm cache) everywhere.
- **Host:** 64 cores; the host was CPU-loaded by other users' jobs (1-min load average ~55–69 until ~19:25, then 7–35;
  logged every 10 s from 19:00:39 in `host_loadavg_chain_{b,c}.log`). GPU exclusivity says nothing about CPU contention; the latency-bound workload B rows
  (Python-driven launches) are the most exposed to it.
- **Chains** (`chain_a.log`, `chain_b.log`, `chain_c.log`): chain A 18:18:26–18:31:20 (workload A s512, 8 rows); chain B
  took the lock at 18:31:20, the same second chain A finished; chain C (three follow-up rows, 20:45–21:00) waited for chain B. GPU1 lock released at 21:00:01.

Commands (each in `/home/ningan/RoboticCuttingSim-rl-review`, prefixed by
`CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/data/ningan/tmp timeout <s>`):

```bash
# chain A (and chain B/C workload A s15): P1-6 R0 harness, one process per (N, variant); variant files {} / {"particle_contact_mode": "alias_free_v1"}
.deps/isaacsim-venv/bin/python scripts/rl_ready/run_perf_acceptance.py --variant-json <D>/variant_<default|alias_free_v1>.json \
  --mode throughput --num-envs <1|4|8|16> --seeds <512|15> --warmup-steps 20 --steps <300 at N=1 | 100> --device cuda:0 \
  --gpu-index 1 --out <D>/a_n<N>[_s15]_<variant>
# single_env_headless (P1-0 command)
.deps/isaacsim-venv/bin/python scripts/rl_ready/run_baseline.py --mode replay --seed <15|512> --gpu-index 1 --warm-resets 5 \
  --run-name excl001_single_s<seed>
# batched CUT (P1-5 commands; N=8 with 2 episodes as m2_cut_n8)
.deps/isaacsim-venv/bin/python scripts/rl_ready/run_cut_batched.py --num-envs <1|8|16> [--episodes 2] --gpu-index 1 --run-name excl001_cut_n<N>
# single_env_headless C2 CUT reference (P1-5 command)
.deps/isaacsim-venv/bin/python scripts/rl_ready/run_cut_single.py --mode reference --gpu-index 1 --run-name excl001_single_cut_ref
# workload B (P1-1b S3 / B2 follow-up commands)
.deps/isaaclab-venv/bin/python scripts/rl_ready/probe_lab_batched_gpu.py --probe g2 --material tlmpm_v7_2 --evaluate \
  --driver recorded --scene-seeds 512 15 --steps 1200 --gpu-index 1 --num-envs <N> \
  [--tool-mount physical_grasp_v1 --grip-profile mjwarp_grip_v1] --out <D>/b<1|2>_n<N>
# evidence (CPU)
PYTHONPATH=/home/ningan/RoboticCuttingSim-rl-review/src .venv/bin/python scripts/rl_ready/make_p1_exclusive_evidence.py \
  --out docs/evidence/rl_ready/p1_exclusive_001.json
```

The frozen runners `run_baseline.py`, `run_cut_batched.py` and `run_cut_single.py` write to fixed directories
(`artifacts/rl_ready/p1_0|p1_5/<run-name>_NNN`); the chain moved each finished run directory (`mv -n`, same filesystem)
into `<D>` right after the row, so nothing of this window stays under `p1_0/` or `p1_5/`.

Seeds: workload A WP-17 scene seeds 512 (chain A) and 15 (chain B), material seed 42; CUT sim/material seed 42, env
seeds 1000+e (+100 per episode); workload B scene seeds cycled 512, 15, 512, 15, …

## 2. Classification rule (decided before the data, implemented in the assembler)

A row is **EXCLUSIVE_GPU** only if all three hold:

1. the row's own in-process telemetry (`rl.telemetry.GpuTelemetry`, ~1 Hz over the whole process incl. a pre-work
   baseline; per-PID SM from `nvidia-smi pmon`, compute *and* graphics processes from `nvidia-smi -q -d PIDS`) labels it
   EXCLUSIVE_GPU: **no foreign compute PID and no foreign non-display graphics PID in any sample**, foreign per-PID SM
   max ≤ 5 %, baseline total utilisation max ≤ 5 %;
2. the chain-level sampler (`exclusive_gpu_sampler.py`, every 2 s, own = the chain shell's process tree) shows the same
   inside the row's start–end window (where the sampler was not yet running — chain A rows `a_n1_*`, `a_n4_default` —
   this check is recorded as unavailable, not as passed);
3. the nvidia-smi compute-app list taken just before (chain B/C also just after) the row is empty.

Anything else is **SHARED_GPU**, with the measured foreign load. The display server (Xorg, PID 4060, 47 MiB) never
counts. No row was relabelled after the fact; the strict rule is applied even where the foreign process was a
transient 2–7 MiB graphics context with 0 % SM (three rows, §4).

## 3. The table

Legend — contacts: **A** = workload A replay: tool edge + handle + two measured finger pads (reaction computed, not fed
back); **H** = single_env_headless replay: tool/handle/pads/fixture, same inputs; **C** = CUT: tool/handle box/fixture,
ideal tracking; **B** = MJWarp robot/tool/fingers/table + material tool/handle/pads/fixture with the reaction wrench fed
back; **N** = Native PhysX robot/tool/fingers + material, 20 ms reaction feedback. Recording: none = nothing in the timed
calls; eval stream = P1-2b device evaluation stream every 20 steps (36 frames/episode), evaluation outside timing.
Cameras: Native rows global D455 RGB + wrist D405 RGB-D (surface on capture); all headless rows none. Robot DOF: the
Lab receipt's `dofs_per_world` for B (18 = two Franka arms with fingers; 24 in B2, where the grasped tool is a free
body); 18 for Native; 0 without a robot. Particles/env of the mixed-seed B rows (seeds 512 / 15 alternating: 102,732 /
103,898) is the mean; B1 N=1 runs seed 512 only. Material dt 0.1 ms
= 200 substeps per 50 Hz control step everywhere; workload B adds 10 rigid substeps of 2 ms per control step.
Warm = steady state after warm-up in a warm process; cold = process start to the last step (single_env rows) or the
WP-17 factory→finalize wall (Native).

| boundary | workload | N | particles/env | control Hz | material dt | cameras | env-steps/s | aggregate sim-time/wall | VRAM peak | reset p95 | physics verdict | substeps | robot DOF | contacts | recording | warm/cold | GPU | sharing (foreign load) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_env_native | D-s512 | 1 | 102732 | 50 | 0.0001 s | Native RGB-D rig | 4.040 | 0.0808 | 5542 MiB | n/a | PASS | 200 | 18 | N | full Native | warm | RTX 5090 #1, driver UNVERIFIED | EXCLUSIVE_GPU (process list only) | wp17_evidence |
| single_env_native | D-s512 | 1 | 102732 | 50 | 0.0001 s | Native RGB-D rig | 3.194 | 0.0639 | 5542 MiB | n/a | PASS | 200 | 18 | N | full Native | cold | RTX 5090 #1, driver UNVERIFIED | EXCLUSIVE_GPU (process list only) | wp17_evidence |
| single_env_native | D-s15 | 1 | 103898 | 50 | 0.0001 s | Native RGB-D rig | 3.866 | 0.0773 | 5638 MiB | n/a | PASS | 200 | 18 | N | full Native | warm | RTX 5090 #1, driver UNVERIFIED | EXCLUSIVE_GPU (process list only) | wp17_evidence |
| single_env_native | D-s15 | 1 | 103898 | 50 | 0.0001 s | Native RGB-D rig | 3.075 | 0.0615 | 5638 MiB | n/a | PASS | 200 | 18 | N | full Native | cold | RTX 5090 #1, driver UNVERIFIED | EXCLUSIVE_GPU (process list only) | wp17_evidence |
| single_env_headless | A/D-s15 | 1 | 103898 | 50 | 0.0001 s | none | 7.213 | 0.1443 | 758 MiB | 0.647 s | PASS | 200 | 0 | H | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | excl001_single_s15_001 |
| single_env_headless | A/D-s15 | 1 | 103898 | 50 | 0.0001 s | none | 6.954 | 0.1391 | 758 MiB | 0.397 s | PASS | 200 | 0 | H | none | cold | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | excl001_single_s15_001 |
| single_env_headless | A/D-s512 | 1 | 102732 | 50 | 0.0001 s | none | 7.567 | 0.1513 | 758 MiB | 0.612 s | PASS | 200 | 0 | H | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | excl001_single_s512_001 |
| single_env_headless | A/D-s512 | 1 | 102732 | 50 | 0.0001 s | none | 7.093 | 0.1419 | 758 MiB | 1.428 s | PASS | 200 | 0 | H | none | cold | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | excl001_single_s512_001 |
| batched_headless | A-s512-frozen_v1 | 1 | 102732 | 50 | 0.0001 s | none | 6.749 | 0.1350 | 864 MiB | 0.264 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n1_default |
| batched_headless | A-s512-alias_free_v1 | 1 | 102732 | 50 | 0.0001 s | none | 9.739 | 0.1948 | 926 MiB | 0.247 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n1_alias_free_v1 |
| batched_headless | A-s512-frozen_v1 | 4 | 102732 | 50 | 0.0001 s | none | 10.055 | 0.2011 | 1084 MiB | 0.242 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n4_default |
| batched_headless | A-s512-alias_free_v1 | 4 | 102732 | 50 | 0.0001 s | none | 13.846 | 0.2769 | 1178 MiB | 0.280 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n4_alias_free_v1 |
| batched_headless | A-s512-frozen_v1 | 8 | 102732 | 50 | 0.0001 s | none | 9.617 | 0.1923 | 1356 MiB | 0.216 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n8_default |
| batched_headless | A-s512-alias_free_v1 | 8 | 102732 | 50 | 0.0001 s | none | 14.012 | 0.2802 | 1482 MiB | 0.263 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n8_alias_free_v1 |
| batched_headless | A-s512-frozen_v1 | 16 | 102732 | 50 | 0.0001 s | none | 9.831 | 0.1966 | 2044 MiB | 0.256 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n16_default |
| batched_headless | A-s512-alias_free_v1 | 16 | 102732 | 50 | 0.0001 s | none | 13.559 | 0.2712 | 2202 MiB | 0.251 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n16_alias_free_v1 |
| batched_headless | A-s15-frozen_v1 | 8 | 103898 | 50 | 0.0001 s | none | 9.296 | 0.1859 | 1388 MiB | 0.500 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | SHARED_GPU (SM mean 0 %/max 0 %; graphics PID [2301065] max 7 MiB) | a_n8_s15_default |
| batched_headless | A-s15-alias_free_v1 | 8 | 103898 | 50 | 0.0001 s | none | 13.746 | 0.2749 | 1482 MiB | 0.510 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n8_s15_alias_free_v1 |
| batched_headless | A-s15-frozen_v1 | 8 | 103898 | 50 | 0.0001 s | none | 9.352 | 0.1870 | 1388 MiB | 0.197 s | REPORTED_NOT_SCORED | 200 | 0 | A | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | a_n8_s15_default_r2 |
| batched_headless | C2-CUT-v7_2-N1 | 1 | 96000 | 50 | 0.0001 s | none | 6.849 | 0.1370 | 944 MiB | 0.262 s | PASS | 200 | 0 | C | eval stream | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | excl001_cut_n1_001 |
| batched_headless | C2-CUT-v7_2-N8 | 8 | 96000 | 50 | 0.0001 s | none | 10.350 | 0.2070 | 2170 MiB | 0.690 s | PASS | 200 | 0 | C | eval stream | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | excl001_cut_n8_001 |
| batched_headless | C2-CUT-v7_2-N16 | 16 | 96000 | 50 | 0.0001 s | none | 10.223 | 0.2045 | 3522 MiB | 0.223 s | PASS | 200 | 0 | C | eval stream | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | excl001_cut_n16_001 |
| single_env_headless | C2-v7_2-single-reference | 1 | 96000 | 50 | 0.0001 s | none | 8.011 | 0.1602 | 688 MiB | 0.133 s | PASS | 200 | 0 | H | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | excl001_single_cut_ref_001 |
| single_env_headless | C2-v7_2-single-reference | 1 | 96000 | 50 | 0.0001 s | none | 7.405 | 0.1481 | 688 MiB | 0.133 s | PASS | 200 | 0 | H | none | cold | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | excl001_single_cut_ref_001 |
| batched_headless | B1-rigid-mount-N1 | 1 | 102732 | 50 | 0.0001 s | none | 4.955 | 0.0991 | 886 MiB | n/a | PASS | 200 | 18 | B | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | b1_n1 |
| batched_headless | B1-rigid-mount-N4 | 4 | 103315 | 50 | 0.0001 s | none | 7.402 | 0.1480 | 1106 MiB | n/a | PASS | 200 | 18 | B | none | warm | RTX 5090 #1, 570.211.01 | SHARED_GPU (SM mean 0 %/max 0 %; graphics PID [2364307] max 7 MiB) | b1_n4 |
| batched_headless | B1-rigid-mount-N8 | 8 | 103315 | 50 | 0.0001 s | none | 8.666 | 0.1733 | 1410 MiB | n/a | PASS | 200 | 18 | B | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | b1_n8 |
| batched_headless | B1-rigid-mount-N16 | 16 | 103315 | 50 | 0.0001 s | none | 8.789 | 0.1758 | 2102 MiB | n/a | PASS | 200 | 18 | B | none | warm | RTX 5090 #1, 570.211.01 | SHARED_GPU (SM mean 0 %/max 0 %; graphics PID [2565857, 2565858] max 14 MiB) | b1_n16 |
| batched_headless | B2-physical-grasp-mjwarp_grip_v1-N4 | 4 | 103315 | 50 | 0.0001 s | none | 7.372 | 0.1474 | 1108 MiB | n/a | PASS | 200 | 24 | B | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | b2_n4 |
| batched_headless | B1-rigid-mount-N4 | 4 | 103315 | 50 | 0.0001 s | none | 7.655 | 0.1531 | 1106 MiB | n/a | PASS | 200 | 18 | B | none | warm | RTX 5090 #1, 570.211.01 | EXCLUSIVE_GPU | b1_n4_r2 |

Row scopes (read the row with its scope):

- **single_env_native** (4 rows): copied from the P1-0 evidence (`source: wp17_evidence`; WP-17 Native, Kit, cameras,
  PhysX robot, full recording), not rerun. Their EXCLUSIVE label rests on the WP-17 process list only (per-PID
  utilisation was not sampled then), driver not recorded (UNVERIFIED).
- **single_env_headless A/D** (frozen single `LiveCuttingMaterial`, tlmpm_v7_2, law 8, 1 mm / 4 mm, replay of the
  WP-17 recording): warm = steps 2..1200 (pad update + graph launch of 200 substeps + device sync + reaction readback +
  `device_reduction_v1` diagnostics); reset p95 = the 5 warm resets in the same process (task particles, solver
  construction, graph capture) — a different operation from the batched full reset; cold reset = the first reset.
  Physics verdict = material invariants; replay parity vs the 121 Native samples is in the evidence (both seeds: final
  codes = Native, labelled 37,958 / 85,786 = Native).
- **batched_headless A** (P1-6 R0 harness, throughput mode, `VectorCuttingEnv` replay via `step_measured_edge`): 20
  warm-up + 100 timed batch steps (300 at N=1), i.e. WP-17 replay steps 21–120 (21–320 at N=1); 1 untimed + 10 timed
  full resets (reset p95 = full reset). **Tool–material contact starts at step ~314 (s512) / ~321 (s15)**, so the N ≥ 4
  windows are entirely pre-contact (0 labelled particles at the end) and the N=1 window contains the first contact
  steps (21,370 labelled). P1-0's kernel split shows the frozen_v1 step cost is nearly phase-independent (812–860 µs per
  substep), so this mostly matters for `alias_free_v1`, whose particle-contact work may depend on the phase (not
  measured here). Physics verdict `REPORTED_NOT_SCORED` (throughput mode has no evaluator; every env `invalid_reason`
  0); the physics of both variants is decided by the P1-6 judge on episode runs, not here.
- **batched_headless CUT** (`bar_flat_board_tlmpm_v7_2_v1`, `reference_command_policy_v1`, `frozen_v1`): warm = all
  synchronized `env.step` calls of all episodes but the first; reset p95 = full resets; verdict = every reference
  episode frozen sever evaluator PASS and L2 PASS.
- **batched_headless B1/B2** (Isaac Lab Newton/MJWarp kit-less, `probe_lab_batched_gpu.py` g2): 1,200 recorded-command
  steps, strict mode, warm = steps 2..1200 with a device sync per step; no resets inside the episode (reset p95 n/a; the
  cold build/reset is in the evidence `runs.<row>.cold`); verdict = frozen evaluator PASS in every env, no terminal
  `termination_reason` bit (16 = `TRUNCATED_HORIZON` at the 1,200-step horizon is not a termination) and one device
  HashGrid.

## 4. Classification of every row

| row | chain | window (KST) | own telemetry: samples / foreign PIDs / foreign SM max | chain sampler: samples / foreign PIDs / foreign SM max | before / after compute apps | label |
|---|---|---|---|---|---|---|
| a_n1_default | A | 18:18:26–18:19:25 | 60 / none / 0.0 % | not running (unavailable) | empty / n/a | **EXCLUSIVE_GPU** |
| a_n1_alias_free_v1 | A | 18:19:25–18:20:10 | 45 / none / 0.0 % | not running (unavailable) | empty / n/a | **EXCLUSIVE_GPU** |
| a_n4_default | A | 18:20:10–18:21:10 | 60 / none / 0.0 % | not running (unavailable) | empty / n/a | **EXCLUSIVE_GPU** |
| a_n4_alias_free_v1 | A | 18:21:10–18:21:58 | 48 / none / 0.0 % | 21 / none / 0 % | empty / n/a | **EXCLUSIVE_GPU** |
| a_n8_default | A | 18:21:58–18:23:50 | 112 / none / 0.0 % | 56 / none / 0 % | empty / n/a | **EXCLUSIVE_GPU** |
| a_n8_alias_free_v1 | A | 18:23:50–18:25:12 | 82 / none / 0.0 % | 41 / none / 0 % | empty / n/a | **EXCLUSIVE_GPU** |
| a_n16_default | A | 18:25:12–18:28:43 | 212 / none / 0.0 % | 105 / none / 0 % | empty / n/a | **EXCLUSIVE_GPU** |
| a_n16_alias_free_v1 | A | 18:28:43–18:31:20 | 157 / none / 0.0 % | 79 / none / 0 % | empty / n/a | **EXCLUSIVE_GPU** |
| single_s15 | B | 18:31:23–18:34:30 | 184 / none / 0.0 % | 93 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| single_s512 | B | 18:34:37–18:37:41 | 180 / none / 0.0 % | 92 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| a_n8_s15_default | B | 18:37:48–18:39:54 | 126 / [2301065] / 0.0 % | 63 / [2300243] / 0 % | empty / empty | **SHARED_GPU** |
| a_n8_s15_alias_free_v1 | B | 18:40:00–18:41:34 | 93 / none / 0.0 % | 47 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| cut_n8 | B | 18:41:40–19:00:39 | 1128 / none / 0.0 % | 569 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| b1_n4 | B | 19:00:45–19:12:14 | 679 / [2364307] / 0.0 % | 345 / none / 0 % | empty / empty | **SHARED_GPU** |
| b2_n4 | B | 19:12:21–19:23:48 | 680 / none / 0.0 % | 344 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| b1_n8 | B | 19:23:55–19:42:53 | 1135 / none / 0.0 % | 569 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| cut_n1 | B | 19:42:59–19:44:53 | 112 / none / 0.0 % | 57 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| b1_n1 | B | 19:45:00–19:49:17 | 255 / none / 0.0 % | 128 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| cut_n16 | B | 19:49:23–20:08:14 | 1128 / none / 0.0 % | 566 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| b1_n16 | B | 20:08:20–20:45:20 | 2216 / [2565857, 2565858] / 0.0 % | 1109 / [2523670, 2523671, 2524017] / 0 % | empty / empty | **SHARED_GPU** |
| single_cut_ref | C | 20:45:29–20:47:07 | 96 / none / 0.0 % | 49 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| a_n8_s15_default_r2 | C | 20:47:13–20:49:06 | 113 / none / 0.0 % | 57 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |
| b1_n4_r2 | C | 20:49:12–20:59:57 | 644 / none / 0.0 % | 322 / none / 0 % | empty / empty | **EXCLUSIVE_GPU** |

Chain A rows `a_n1_default`, `a_n1_alias_free_v1` and `a_n4_default` ran before the chain sampler started (18:21); their
label rests on checks 1 and 3 (own telemetry 45–60 samples with no foreign process and 0 % foreign SM, empty
compute-app list before the row). The SHARED rows' foreign load: graphics contexts of 2–7 MiB living < 2 s, 0 % foreign
SM, no foreign compute process (§7).

## 5. Comparisons (boundaries never merged)

All ratios below divide two separately measured rows; a ratio across boundaries is a comparison of two numbers, never a
merged speedup, and the timed windows differ as stated in §3.

### 5.1 Batched vs batched N=1 (same workload, same runner, separate processes)

| workload (variant, seed) | N=1 env-steps/s | N=4 | N=8 | N=16 | rows |
| --- | --- | --- | --- | --- | --- |
| A `frozen_v1` s512 | 6.749 | 10.055 (1.49×) | 9.617 (1.43×) | 9.831 (1.46×) | all EXCLUSIVE |
| A `alias_free_v1` s512 | 9.739 | 13.846 (1.42×) | 14.012 (1.44×) | 13.559 (1.39×) | all EXCLUSIVE |
| CUT C2 (`frozen_v1`) | 6.849 | NOT_RUN | 10.350 (1.51×) | 10.223 (1.49×) | all EXCLUSIVE |
| B1 rigid mount (robot + material) | 4.955 (s512 only) | 7.655 (1.54×; first run 7.402, SHARED) | 8.666 (1.75×) | 8.789 (1.77×, SHARED, 0 % foreign SM) | see labels |

- Material-only batches gain ~1.4–1.5× over N=1 (A already at N=4, CUT at N=8) and nothing beyond: the batch step grows linearly
  with N after N=4 (CUT: 0.146 → 0.773 → 1.565 s per batch step at N=1/8/16). Our own `pmon` SM-busy share is already 76 % (mean) at
  A N=1, so one ~100k-particle env leaves only ~1/3 of the 5090 idle, and that is what batching recovers. This
  confirms P1-2's matched-load result (N8/N1 1.45×, N16/N1 1.42×) on an exclusive GPU.
- B1 gains more (1.75× at N=8/16) because its N=1 step carries ~50 ms of latency-bound robot work (rigid substeps,
  IK, Python-driven launches: 0.202 s per step vs 0.146 s for CUT N=1 and 0.148 s for A N=1) that the batch amortises; at
  N=16 robot + material costs 1.82 s per batch step against 1.57 s for the material-only CUT batch.
- Host CPU contention moved the latency-bound B rows by a few per cent: B1 N=4 ran at 7.402 env-steps/s with a host
  load average of 55–69 (64 cores) and at 7.655 with 8–11 (§7).

### 5.2 Batched vs the single env (single_env_headless, same inputs; different boundary)

| workload | single env warm (EXCLUSIVE) | batched N=1 | N=4 | N=8 | N=16 |
| --- | --- | --- | --- | --- | --- |
| A s512 vs `frozen_v1` | 7.567 | 0.89× | 1.33× | 1.27× | 1.30× |
| A s512 vs `alias_free_v1` | 7.567 | 1.29× | 1.83× | 1.85× | 1.79× |
| A s15 vs `frozen_v1` / `alias_free_v1` | 7.213 | — | — | 1.30× / 1.91× | — |
| CUT vs C2 single reference | 8.011 | 0.85× | — | 1.29× | 1.28× |

- With the frozen contact path the batched runtime beats one single env by ~1.3× in aggregate env-steps/s at N ≥ 4;
  batched N=1 is 11–15 % slower than the single env (vector-env layer: action/observation tensors, per-env
  reductions, evaluation stream for CUT). The sequential-N single-env baseline (P1-0, DERIVED_NOT_RUN) stays at the
  single-env rate for every N, so these ratios are also the batched gain over sequential N.
- Windows differ: the single env times steps 2..1200 (A) / 2..700 (CUT) including contact and cutting; batched A times
  the pre-contact window 21..120. P1-0's kernel split (frozen_v1 cost nearly phase-independent) makes the A frozen_v1
  comparison fair to within a few per cent; the `alias_free_v1` ratios are pre-contact numbers only.
- Workload B has no single-env headless counterpart (a robot needs Native or the Lab batch); comparing it with
  single_env_native would merge boundaries, so it is not done. For orientation only: Native warm (cameras, PhysX,
  recording) is 3.9–4.0 env-steps/s.

### 5.3 `alias_free_v1` vs `frozen_v1` (one variant per process, both EXCLUSIVE, same window)

| seed, N | frozen_v1 | alias_free_v1 | ratio | VRAM frozen → alias_free |
| --- | --- | --- | --- | --- |
| s512, 1 | 6.749 | 9.739 | 1.443 | 864 → 926 MiB |
| s512, 4 | 10.055 | 13.846 | 1.377 | 1084 → 1178 MiB |
| s512, 8 | 9.617 | 14.012 | 1.457 | 1356 → 1482 MiB |
| s512, 16 | 9.831 | 13.559 | 1.379 | 2044 → 2202 MiB |
| s15, 8 | 9.352 (repeat; first run 9.296, SHARED) | 13.746 | 1.470 | 1388 → 1482 MiB |

1.38–1.47× in every pair (P1-6 smoke on a shared GPU: 1.36–1.37×), +60–160 MiB. One process per row, no ABBA
repetition: this is a descriptive pair per N, not the P1-6 acceptance (G9 needs 5 ABBA pairs, the judge and the physics
gates); the windows are pre-contact (N ≥ 4) where `alias_free_v1`'s contact work may differ from the cutting phase.

### 5.4 Against the earlier SHARED rows

| earlier row (SHARED_GPU, foreign SM mean) | env-steps/s then | now (EXCLUSIVE) | now / then |
| --- | --- | --- | --- |
| P1-0 single_env_headless s512 `replay_s512_001` (14–21 %) | 5.928 | 7.567 | 1.28× |
| P1-0 single_env_headless s15 `replay_s15_001` | 5.629 | 7.213 | 1.28× |
| P1-0 single_env_headless s15 `band_replay_s15_002` (**EXCLUSIVE** then) | 7.419 | 7.213 | 0.97× (run-to-run) |
| P1-2 batched A N=1 `m2_n1_001` s512 (24 %) | 5.091 | 6.749 | 1.33× |
| P1-2 batched A N=8 `m2_n8_001` (44 %, seeds 512/15 mixed) | 4.961 | 9.617 | 1.94× |
| P1-2 batched A N=16 `m2_n16_001` (45 %) | 4.819 | 9.831 | 2.04× |
| P1-2 alternation round 1 N=1/8/16 (46–47 %) | 3.354 / 4.849 / 4.752 | 6.749 / 9.617 / 9.831 | 2.01× / 1.98× / 2.07× |
| P1-5 batched CUT N=1 / N=8 / N=16 (45 / 10 / 43 %) | 3.247 / 9.049 / 5.442 | 6.849 / 10.350 / 10.223 | 2.11× / 1.14× / 1.88× |
| P1-5 C2 single reference (30–48 %) | 3.82–5.26 | 8.011 | 1.52–2.10× |
| P1-1b S3 B1 N=4 `s3_b1_n4_001` (SHARED) | 3.079 | 7.655 | 2.49× |
| P1-1b B2 `mjwarp_grip_v1` N=4 `s3_b2grip_n4_001` (49 %) | 3.639 | 7.372 | 2.03× |

The co-tenant halved most earlier batched numbers; the earlier P1-2 *ratios at matched load* (1.45× / 1.42×) and
the P1-0 EXCLUSIVE single-env row agree with the exclusive window. The P1-2 rows use longer windows with contact and
mixed seeds, so their ratio to the s512 harness rows is indicative, not exact. Physics in the exclusive rows matches
the earlier evidence: workload B labelled counts are identical to P1-1b S3 (B1 33,740 / 80,986 per seed in every N;
B2 38,220 / 81,034 / 38,220 / 81,033), single-env replays reproduce the Native final codes, and the batched CUT final
codes are within 0–60 particles of the three P1-5 single references (28 of 33 reference episodes bitwise equal to one
of them, the others 4–30 particles from the nearest).

## 6. Fidelity labels

- **Workload A**: reduced fidelity — replayed WP-17 edge and measured pads, no robot, the reaction is computed but drives
  nothing. Throughput rows are pre-contact windows (§3).
- **CUT (C2)**: reduced fidelity — versioned v7_2 CUT scene, ideal tracking, no robot, handle box only.
- **B1**: rigid tool mount (`rigid_tool_mount_no_grasp`, `finger_opening_fixed_nominal`), MJWarp robot + material with
  reaction feedback; reduced fidelity.
- **B2**: physical grasp (`physical_grasp_v1`) with the **opt-in** `mjwarp_grip_v1` contact profile; not reduced, but the
  closed-loop MJWarp robot differs from Native PhysX (descriptive comparison only, no equivalence claim).

## 7. Failures, NOT_RUN and open points

- **No row failed.** Every process exited 0; every physics check that applies passed (single-env invariants and Native
  parity, CUT evaluator 33/33 reference episodes + L2, workload B evaluator 37/37 envs over 6 runs, one device HashGrid per
  process). The 1,200-step workload B episodes end with `termination_reason` 16 = `TRUNCATED_HORIZON` in every env
  (the horizon, not a termination; P1-1b's S3 rows predate the P1-2b bitfield and recorded 0).
- **SHARED_GPU rows kept as measured** (not relabelled): `a_n8_s15_default` (one 7 MiB graphics PID in 1 of 126
  samples + one 6 MiB PID in the chain sampler), `b1_n4` (one 7 MiB graphics PID in 1 of 679 samples), `b1_n16` (two
  PIDs, 14 MiB together, in 1 of 2,216 samples; in its window the chain sampler saw three more and the process
  catcher three more, the latter named `egl_probe/build/test_device` from another project's environment). Foreign SM 0 % in every
  sample of all three. The first two were repeated EXCLUSIVE (`a_n8_s15_default_r2`, `b1_n4_r2`); **B1 N=16 has no
  EXCLUSIVE row** (a repeat would have held GPU1 for another ~40 min; NOT_RUN).
- **Budget:** chain B took 134 min and chain C 14 min (window 18:18–21:00 including chain A, 2 h 42 min), above the
  ~2.5 h guide; the chain ran the priority rows first and gated the optional rows on a 180-min budget, which all of
  them met. GPU1 had no foreign compute process at any point of the window.
- **Host CPU load:** another user's CPU jobs kept the 64-core host at a load average of ~60 until ~19:25, then 7–35
  (logged from 19:00:39; per-row median in `runs.<row>.host_loadavg_1min`). Rows before 19:00 have no host-load log
  (a `top` reading at 19:00:35 showed 61–64). GPU exclusivity does not cover this; the latency-bound B rows are the
  most exposed (B1 N=4: 7.402 at load ~58 vs 7.655 at ~10).
- **Not measured here:** CUT N=4; B2 at N ≠ 4; workload A seed 15 at N ≠ 8; batched cold rows (only warm; the B rows'
  build/reset times are in `runs.<row>.cold`); NVIDIA kernel-level split on the exclusive GPU; ABBA repetitions (each
  row is one process, so run-to-run spread is only visible in the two repeats: A s15 N=8 9.296 / 9.352, B1 N=4
  7.402 / 7.655, and P1-0's EXCLUSIVE s15 7.419 vs 7.213 now).
- **Scope limits:** workload A throughput windows are pre-contact (N ≥ 4) or start of contact (N=1); `alias_free_v1`
  physics is not judged here; the native rows are WP-17 evidence (driver UNVERIFIED, process-list-only exclusivity);
  VRAM peak is per-process `nvidia-smi` memory (CUDA context + Warp mempool + Torch), sampled at ~1 Hz.
- The frozen runners write into `artifacts/rl_ready/p1_0|p1_5`; the chains moved this window's run directories into
  `exclusive_001/` right after each row (logged as `moved` in the chain logs). Raw outputs: 77 MB.
