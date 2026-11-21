"""Run a complete Native TRIM episode with the supplied frozen controller in a GPU1-only namespace."""
import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
C3_SHA256 = "10e128c57627a6f304471c7d9c9ddeecf7b2206a10add218776ce039f8efa506"


def load_controller(path, sha256):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != sha256:
        raise ValueError(f"controller SHA-256 {actual} differs from {sha256}")
    spec = spec_from_file_location("native_trim_controller", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def main():
    from agentic_robot.lab.sim.envs.rcs_gpu1 import _require_gpu1, launch_app_gpu1
    from robotic_cutting_sim import isaac_sim_runner
    from robotic_cutting_sim.agentic_trim_env import open_env, write_json
    from robotic_cutting_sim.profiling import StageProfiler

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(512, 15), required=True)
    p.add_argument("--controller", type=Path, required=True)
    p.add_argument("--controller-sha256", default=C3_SHA256)
    p.add_argument("--runtime-profile")
    p.add_argument("--reuse-grasp-state", action="store_true")
    p.add_argument("--robot-substeps", type=int, default=10)
    args = p.parse_args()
    _require_gpu1()
    isaac_sim_runner.launch_app = launch_app_gpu1
    run = load_controller(args.controller, args.controller_sha256)

    args.out.mkdir(parents=True, exist_ok=False)
    control_dir = args.out / "controller"
    control_dir.mkdir()
    ctx = {"seed": args.seed, "split": "high_speed_engineering", "params": {},
           "out_dir": str(control_dir), "candidate_id": args.controller.stem, "policy_seed": 0}
    write_json(args.out / "spec.json", {"command": [sys.executable, *sys.argv], "ctx": ctx,
               "control": {"path": str(args.controller), "sha256": args.controller_sha256},
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
               "gpu_uuid": os.environ["RSI_GPU1_UUID"], "runtime_profile": args.runtime_profile,
               "robot_substeps": args.robot_substeps, "reuse_grasp_state": args.reuse_grasp_state})
    start = time.perf_counter()
    with open_env(output_dir=args.out / "env", seed=args.seed, stream_physics=True, memory_receipts=True,
                  native_sensor_runtime="isaac62_physx_offline_v1", material_diagnostics="device_reduction_v1",
                  runtime_profile=args.runtime_profile, reuse_grasp_state=args.reuse_grasp_state,
                  robot_substeps=args.robot_substeps) as env:
        factory_s = time.perf_counter() - start
        env.reset()
        reset_s = time.perf_counter() - start - factory_s
        profiler = StageProfiler(sync_device=None, kernels=False).install()
        t0 = time.perf_counter()
        try:
            control = run(env, ctx)
            play_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            env.finalize("HIGH_SPEED_BENCHMARK")
            finalize_s = time.perf_counter() - t0
        finally:
            profiler.uninstall()
            write_json(args.out / "stage_summary.json", profiler.summary())
        ev = json.loads((env.current_recording / "evaluation.json").read_text())
        boundary = env.total_steps == 1200 and abs(env.time_s - 24.0) <= 1e-9
        total = time.perf_counter() - start
        result = {"status": "COMPLETE" if boundary else "INCOMPLETE",
                  "physics_status": ev.get("status"), "task_success": bool(ev.get("success")),
                  "control_steps": env.total_steps, "simulated_s": env.time_s, "seed": args.seed,
                  "factory_s": factory_s, "reset_s": reset_s, "controller_wall_s": play_s,
                  "finalize_s": finalize_s, "full_episode_wall_s": total,
                  "episodes_per_gpu_hour": 3600 / total if boundary else None,
                  "control": control, "evaluation": ev,
                  "recording": str(env.current_recording.relative_to(args.out))}
        write_json(args.out / "benchmark.json", result)
        print(json.dumps({k: result[k] for k in ("status", "physics_status", "control_steps",
                                                "full_episode_wall_s")}), flush=True)
    return 0 if boundary else 1


if __name__ == "__main__":
    sys.exit(main())
