"""Workload B stage S1 GPU probes (ADR 0022 §9): G0 smoke, T2 no-host-sync, G1 deterministic isolation, G2 throughput.

NOT RUN in stage S0 (CPU only). S1 starts after workload A M2 and runs under the RCS GPU rules (decision D-07):
one job per GPU under the flock, explicit device, >= 4 GB left free, rows labelled SHARED_GPU / EXCLUSIVE_GPU from
sampled co-tenant utilisation. Example (physical GPU 1, CUDA ordinal 0 inside the process)::

    CUDA_VISIBLE_DEVICES=1 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src \
        flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock \
        .deps/isaaclab-venv/bin/python scripts/rl_ready/probe_lab_batched_gpu.py --probe g0 --num-envs 1 \
        --gpu-index 1 --out artifacts/rl_ready/workload_b/g0_n1_001

Probes (one Isaac Lab build per process; the output directory must not exist):

* ``g0``  N worlds, ``use_cuda_graph=True``: build/reset/first-step timing (cold), graph capture observed, model
          receipt, 20 warm steps with per-stage wall (IK, 10 substeps, edge/pad, dones, observation).
* ``t2``  20 warm steps under ``torch.cuda.set_sync_debug_mode("error")`` with ``wp.array.numpy`` and
          ``wp.synchronize*`` patched to count (strict-mode reads are whitelisted and counted).
* ``g1``  N=4 with ``deterministic_mode="run_to_run"``: (a) identical copies, (b) env 2 perturbed, (c) env 1 reset at
          step 30; writes per-step joint/body state for the bitwise comparison (``--variant base|perturb|reset``).
* ``g2``  ``--steps`` steps at N with staggered masked resets every ``--reset-every`` steps; env-steps/s, reset p95,
          VRAM peak/slope. Robot-only until the batched material accepts ``backend="isaac_lab_newton"``:
          every row is labelled ``reduced_fidelity`` with the reasons from ``BatchedLabOptions.fidelity()``.

Every row records N, particles/env (0: robot-only), control Hz 50, material dt (none), substeps 10 x 2 ms rigid,
cameras none, recording none, warm/cold, GPU name/index/driver, sharing label, source commit.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

MIN_FREE_MB = 4096


def _gpu_row(index):
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=name,driver_version,memory.total,memory.used,"
                          "utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    name, driver, total, used, util = [x.strip() for x in out.stdout.strip().split(",")]
    return {"name": name, "index": int(index), "driver": driver, "memory_total_mb": float(total),
            "memory_used_mb": float(used), "utilization_pct": float(util)}


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _task(seed):
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    return compile_task(generate_scene(seed), goal_config(), seed=42)


def _actions(n, step, perturb=None):
    import torch
    a = torch.zeros((n, 8), device="cuda")
    a[:, 0] = .6*np.sin(.35*step)
    a[:, 1] = .4*np.cos(.245*step)
    a[:, 2] = -.5 if step < 6 else .3
    a[:, 6] = 1.
    if perturb is not None:
        a[perturb, :3] = torch.tensor([-1., 1., -1.], device="cuda")
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", required=True, choices=("g0", "t2", "g1", "g2"))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--tool-mount", default="rigid_mount_v1")
    parser.add_argument("--gpu-index", type=int, required=True, help="physical nvidia-smi index (for telemetry)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--reset-every", type=int, default=0)
    parser.add_argument("--variant", default="base", choices=("base", "perturb", "reset"))
    parser.add_argument("--scene-seed", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    gpu = _gpu_row(args.gpu_index)
    if gpu["memory_total_mb"]-gpu["memory_used_mb"] < MIN_FREE_MB+2048:
        raise SystemExit(f"BLOCKED: GPU {args.gpu_index} has {gpu['memory_total_mb']-gpu['memory_used_mb']:.0f} MB free; "
                         f"D-07 needs >= {MIN_FREE_MB} MB left after this job")
    import torch
    import warp as wp
    from isaaclab.app.sim_launcher import launch_simulation
    from robotic_cutting_sim.batched import lab_scene
    from robotic_cutting_sim.rl.lab_batched_env import IsaacLabBatchedCuttingEnv
    from robotic_cutting_sim.rl.telemetry import GpuTelemetry
    task = _task(args.scene_seed)
    options = lab_scene.BatchedLabOptions(num_envs=args.num_envs, tool_mount=args.tool_mount, device="cuda:0",
                                          deterministic_mode="run_to_run" if args.probe == "g1" else "not_guaranteed",
                                          strict=True)
    cfg = lab_scene.make_batched_lab_cfg(options, task)
    row = {"probe": args.probe, "N": args.num_envs, "tool_mount": args.tool_mount, "particles_per_env": 0,
           "control_hz": 50, "rigid_substeps": "10 x 2 ms", "material_dt": None, "cameras": "none", "recording": "none",
           "gpu": gpu, "commit": _commit(), "workload": "B", **options.fidelity()}
    telemetry = GpuTelemetry(args.gpu_index).start()
    t_start = time.perf_counter()
    with launch_simulation(cfg, {"headless": True, "visualizer": [], "device": "cuda:0"}):
        telemetry.set_phase("build")
        env = IsaacLabBatchedCuttingEnv(cfg, options, task)
        t_build = time.perf_counter()
        env.reset()
        torch.cuda.synchronize()
        t_reset = time.perf_counter()
        env.step(_actions(args.num_envs, 0))
        torch.cuda.synchronize()
        row.update(cold={"build_s": t_build-t_start, "reset_s": t_reset-t_build,
                         "first_step_s": time.perf_counter()-t_reset}, receipt=env.receipt())
        telemetry.set_phase("warm")
        if args.probe == "t2":
            counts = {"numpy": 0, "synchronize": 0}
            original_numpy = wp.array.numpy

            def counted(self):
                counts["numpy"] += 1
                return original_numpy(self)
            wp.array.numpy = counted
            torch.cuda.set_sync_debug_mode("error")
            try:
                for step in range(1, args.steps+1):
                    env.step(_actions(args.num_envs, step))
            finally:
                torch.cuda.set_sync_debug_mode(0)
                wp.array.numpy = original_numpy
            row["t2"] = {"steps": args.steps, "warp_numpy_reads": counts["numpy"],
                         "expected_strict_reads_per_step": 1, "status": "PASS" if counts["numpy"] <= args.steps else "FAIL"}
        else:
            walls, records, resets = [], [], []
            for step in range(1, args.steps+1):
                if args.variant == "reset" and step == 30:
                    mask = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda")
                    mask[1] = True
                    env.reset_envs(mask)
                if args.reset_every and step % args.reset_every == 0:
                    mask = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda")
                    mask[(step//args.reset_every) % args.num_envs] = True
                    t0 = time.perf_counter()
                    env.reset_envs(mask)
                    torch.cuda.synchronize()
                    resets.append(time.perf_counter()-t0)
                t0 = time.perf_counter()
                env.step(_actions(args.num_envs, step, 2 if args.variant == "perturb" else None))
                torch.cuda.synchronize()
                walls.append(time.perf_counter()-t0)
                if args.probe == "g1":
                    from isaaclab_newton.physics import NewtonManager
                    state = NewtonManager.get_state_0()
                    records.append(np.concatenate([state.joint_q.numpy(), state.body_q.numpy().ravel()]))
            row["warm"] = {"steps": args.steps, "step_wall_s_mean": float(np.mean(walls)),
                           "step_wall_s_p95": float(np.percentile(walls, 95)),
                           "env_steps_per_s": args.num_envs/float(np.mean(walls)),
                           "aggregate_sim_time_per_wall": args.num_envs*.02/float(np.mean(walls)),
                           "reset_p95_s": float(np.percentile(resets, 95)) if resets else None,
                           "vram_peak_bytes": int(torch.cuda.max_memory_allocated())}
            if records:
                np.savez_compressed(out/f"g1_{args.variant}.npz", state=np.asarray(records),
                                    body_world=env.model.body_world.numpy(), coord_start=env.coord_start.numpy())
    row["telemetry"] = telemetry.stop()
    row["gpu_sharing"] = row["telemetry"].get("gpu_sharing")
    (out/"row.json").write_text(json.dumps(row, indent=1, default=str))
    print(json.dumps({k: row.get(k) for k in ("probe", "N", "cold", "warm", "t2", "gpu_sharing")}, default=str))


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise SystemExit("GPU probe: set CUDA_VISIBLE_DEVICES to the one GPU this job holds the flock for")
    main()
