"""P1-2b vector-env contract worker and GPU smoke (one env per process).

Opens ``open_vector_env`` for one backend, runs ``rl.vector_contract.check_vector_env_contract`` (reset, steps,
partial reset, evaluate) and writes ``contract.json`` and ``row.json`` (timing, VRAM/co-tenant telemetry on CUDA,
HashGrid census, termination bits, evaluation modes). The CPU form is the contract test's worker
(``tests/test_vector_env_contract.py``); the CUDA form is the P1-2b GPU smoke.

    # workload A, GPU smoke (Isaac Sim venv, no Kit)
    CUDA_VISIBLE_DEVICES=<i> TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<i>.lock \
      .deps/isaacsim-venv/bin/python scripts/rl_ready/run_p1_2b_smoke.py --backend kinematic_ideal_tracking_batched \
      --device cuda --gpu-index <i> --tasks 512 15 --steps 50 --reset-step 25 --evaluation-stream \
      --out artifacts/rl_ready/p1_2b/smoke_a_n2_001
    # workload B (B1 rigid mount), GPU smoke (Isaac Lab venv)
    ... .deps/isaaclab-venv/bin/python scripts/rl_ready/run_p1_2b_smoke.py --backend isaac_lab_newton --device cuda \
      --gpu-index <i> --tasks 512 15 --steps 50 --reset-step 25 --evaluation-stream --out artifacts/rl_ready/p1_2b/smoke_b1_n2_001
    # CPU contract worker (tests): --device cpu with a coarse task (--spacing 0.0025 --max-episode-steps 3)
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

MIN_FREE_MB = 4096
JOB_ESTIMATE_MB = {"kinematic_ideal_tracking_batched": 3072, "isaac_lab_newton": 3584}


def _gpu_row(index):
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=name,driver_version,memory.total,memory.used,"
                          "utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    name, driver, total, used, util = [x.strip() for x in out.stdout.strip().split(",")]
    return {"name": name, "index": int(index), "driver": driver, "memory_total_mb": float(total),
            "memory_used_mb": float(used), "utilization_pct": float(util)}


def _commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts"],
                               capture_output=True, text=True).stdout.strip()
        return {"sha": sha, "tracked_src_or_scripts_modified": bool(dirty)}
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, choices=("kinematic_ideal_tracking_batched", "isaac_lab_newton"))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--gpu-index", type=int, default=None, help="physical nvidia-smi index (telemetry)")
    parser.add_argument("--tasks", type=int, nargs="+", default=[512, 15], help="per-env scene seeds (tasks=)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--reset-step", type=int, default=25)
    parser.add_argument("--reset-envs", type=int, nargs="*", default=[1])
    parser.add_argument("--evaluation-stream", action="store_true")
    parser.add_argument("--stream-every", type=int, default=10)
    parser.add_argument("--observation-profile", default="state", choices=("state", "privileged_state"))
    parser.add_argument("--tool-mount", default="rigid_mount_v1")
    parser.add_argument("--grip-profile", default=None,
                        help="isaac_lab_newton grip_contact_profile (e.g. mjwarp_grip_v1 with physical_grasp_v1)")
    parser.add_argument("--spacing", type=float, default=None, help="coarse particle spacing (tests only)")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cpu = args.device == "cpu"
    if cpu and os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit("CPU worker: run with CUDA_VISIBLE_DEVICES=''")
    if not cpu and os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise SystemExit("GPU smoke: set CUDA_VISIBLE_DEVICES to the one GPU this job holds the flock for")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    row = {"probe": "p1_2b_smoke" if not cpu else "p1_2b_contract_cpu", "backend": args.backend, "N": len(args.tasks),
           "tasks": args.tasks, "steps": args.steps, "reset_step": args.reset_step, "reset_envs": args.reset_envs,
           "evaluation_stream": args.evaluation_stream, "observation_profile": args.observation_profile,
           "tool_mount": args.tool_mount if args.backend == "isaac_lab_newton" else None,
           "grip_profile": args.grip_profile, "device": args.device,
           "particle_spacing_m": args.spacing or .001, "max_episode_steps": args.max_episode_steps,
           "control_hz": 50, "material_dt_s": 1e-4, "material_substeps_per_control_step": 200, "cameras": "none",
           "recording": "none (evaluation stream on the device when enabled)", "commit": _commit(),
           "strict_mode": not args.non_strict, "status": "NOT_RUN"}
    telemetry = None
    if not cpu:
        gpu = _gpu_row(args.gpu_index)
        row["gpu"] = gpu
        free = gpu["memory_total_mb"] - gpu["memory_used_mb"]
        if free - JOB_ESTIMATE_MB[args.backend] < MIN_FREE_MB:
            row.update(status="BLOCKED", reason=f"{free:.0f} MB free; D-07 needs >= {MIN_FREE_MB} MB left")
            (out / "row.json").write_text(json.dumps(row, indent=1))
            raise SystemExit(row["reason"])
        from robotic_cutting_sim.rl.telemetry import GpuTelemetry
        telemetry = GpuTelemetry(args.gpu_index)
        telemetry.set_phase("baseline")
        for _ in range(3):
            telemetry.sample()
            time.sleep(.3)
        telemetry.set_phase("build")
        telemetry.start()
    t_start = time.perf_counter()
    import torch
    import warp as wp
    hashgrids = []
    _init = wp.HashGrid.__init__

    def _counted(self, *a, **k):
        _init(self, *a, **k)
        hashgrids.append(str(self.device))
    wp.HashGrid.__init__ = _counted
    from robotic_cutting_sim.rl import TaskFamily, open_vector_env
    from robotic_cutting_sim.rl.termination import decode_termination
    from robotic_cutting_sim.rl.vector_contract import check_vector_env_contract
    family_kwargs = {}
    if args.spacing:
        family_kwargs["particle_spacing_m"] = args.spacing
    if args.max_episode_steps:
        family_kwargs["max_episode_steps"] = args.max_episode_steps
    base = TaskFamily(scene_seeds=tuple(args.tasks), **family_kwargs) if family_kwargs else None
    options = dict(evaluation_stream={"every": args.stream_every} if args.evaluation_stream else None,
                   observation_profile=args.observation_profile, strict=not args.non_strict)
    if args.backend == "isaac_lab_newton":
        options.update(tool_mount=args.tool_mount)
        if args.grip_profile is not None:
            options.update(grip_contact_profile=args.grip_profile)
        if cpu:
            options.update(cpu_test_mode=True)
    elif cpu:
        options.update(parity_mode=True)
    sync = (lambda: None) if cpu else torch.cuda.synchronize
    walls, reset_walls = [], []
    try:
        t_open = time.perf_counter()
        with open_vector_env(len(args.tasks), task=base, tasks=list(args.tasks), device=args.device,
                             backend=args.backend, **options) as env:
            sync()
            row["cold"] = {"imports_s": t_open - t_start, "open_s": time.perf_counter() - t_open,
                           "warp_kernel_cache": "user cache (not cleared)"}
            if telemetry:
                telemetry.set_phase("warm")
            step, reset = env.step, env.reset

            def timed_step(actions):
                sync()
                t = time.perf_counter()
                result = step(actions)
                sync()
                walls.append(time.perf_counter() - t)
                return result

            def timed_reset(**kwargs):
                sync()
                t = time.perf_counter()
                result = reset(**kwargs)
                sync()
                reset_walls.append(dict(partial=kwargs.get("env_ids") is not None, wall_s=time.perf_counter() - t))
                return result
            env.step, env.reset = timed_step, timed_reset
            report = check_vector_env_contract(env, steps=args.steps, reset_ids=tuple(args.reset_envs),
                                               reset_step=args.reset_step)
            receipt = env.receipt()
            row["particles_per_env"] = [int(x) for x in env.material.packed.particle_counts]
            row["termination_reason_final"] = report["termination_reason"]
            row["input_guard_reason_final"] = None if env.input_guard is None else env.input_guard.reason.tolist()
            row["evaluation"] = report["evaluation"]
            row["fidelity"] = receipt["fidelity"]
            if "grip" in receipt:
                row["grip"] = receipt["grip"]
                row["contact_solver"] = receipt["lab"].get("contact_solver")
            row["evaluation_mode"] = receipt["evaluation_mode"]
            row["evaluation_stream_device_bytes"] = (None if receipt["evaluation_stream"] is None
                                                     else receipt["evaluation_stream"]["device_bytes"])
            (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n")
        row["contract"] = report
        row["status"] = report["status"]
    except Exception as error:
        from robotic_cutting_sim.rl.termination import classify_exception
        row.update(status="FAIL", error=f"{type(error).__name__}: {error}"[:4000],
                   error_class=classify_exception(error), traceback=traceback.format_exc()[-6000:])
    finally:
        warm = walls[1:] if len(walls) > 1 else walls
        row["timing"] = {"first_step_s": walls[0] if walls else None, "warm_steps": len(warm),
                         "warm_step_median_s": sorted(warm)[len(warm) // 2] if warm else None,
                         "warm_step_mean_s": sum(warm) / len(warm) if warm else None,
                         "env_steps_per_s": (len(args.tasks) * len(warm) / sum(warm)) if warm else None,
                         "resets": reset_walls,
                         "timing_scope": "env.step wall incl. strict-mode readback; device synchronized around each "
                                         "call by this harness only"}
        row["hashgrids"] = {"count": len(hashgrids), "cuda_count": sum(d.startswith("cuda") for d in hashgrids),
                            "devices": hashgrids}
        row["wall_s"] = time.perf_counter() - t_start
        if telemetry:
            telemetry.stop()
            row["telemetry"] = telemetry.summary()
            row["telemetry_warm"] = telemetry.summary(phases=("warm",))
            row["gpu_sharing"] = row["telemetry"].get("gpu_sharing")
            row["vram_peak_own_mib"] = row["telemetry"].get("own_vram_peak_mib")
        (out / "row.json").write_text(json.dumps(row, indent=1, default=str) + "\n")
        if row.get("contract"):
            (out / "contract.json").write_text(json.dumps(row["contract"], indent=1, default=str) + "\n")
    print(json.dumps({k: row.get(k) for k in ("status", "backend", "N", "particles_per_env", "evaluation",
                                               "termination_reason_final", "timing", "gpu_sharing",
                                               "vram_peak_own_mib", "hashgrids", "error")}, indent=1, default=str))
    if row["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
