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


def run_episode(env, out, run, ctx, start, factory_s, *, full_cadence):
    from robotic_cutting_sim.agentic_trim_env import write_json
    from robotic_cutting_sim.policy_env import process_memory
    from robotic_cutting_sim.profiling import StageProfiler

    env.reset()
    reset_s = time.perf_counter() - start - factory_s
    with StageProfiler() as profiler:
        t0 = time.perf_counter()
        try:
            control = run(env, ctx)
            play_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            env.finalize("HIGH_SPEED_BENCHMARK")
            finalize_s = time.perf_counter() - t0
        finally:
            write_json(out / "stage_summary.json", profiler.summary())
            if full_cadence:
                write_json(out / "native_timing.json", env.env.backend.runtime_timing_trace())
    rec = env.current_recording
    ev = json.loads((rec / "evaluation.json").read_text())
    frames = json.loads((rec / "frame_manifest.json").read_text())
    physics_stream = json.loads((rec / "physics_stream.json").read_text())
    boundary = env.total_steps == 1200 and abs(env.time_s - 24.0) <= 1e-9
    if full_cadence:
        boundary = boundary and len(frames) == 1201
    total = time.perf_counter() - start
    result = {"status": "COMPLETE" if boundary else "INCOMPLETE",
              "physics_status": ev.get("status"), "task_success": bool(ev.get("success")),
              "control_steps": env.total_steps, "simulated_s": env.time_s, "seed": ctx["seed"],
              "factory_s": factory_s, "reset_s": reset_s, "controller_wall_s": play_s,
              "finalize_s": finalize_s, "full_episode_wall_s": total,
              "episodes_per_gpu_hour": 3600 / total if boundary else None,
              "control": control, "evaluation": ev, "memory": process_memory(),
              "recording": str(rec.relative_to(out)), "episode_dir": str(out),
              "frame_manifest_entries": len(frames), "physics_stream": physics_stream}
    write_json(out / "benchmark.json", result)
    print(json.dumps({k: result[k] for k in ("status", "physics_status", "control_steps",
                                             "full_episode_wall_s")}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(512, 15), required=True)
    p.add_argument("--controller", type=Path, required=True)
    p.add_argument("--controller-sha256", default=C3_SHA256)
    p.add_argument("--runtime-profile", help=argparse.SUPPRESS)
    p.add_argument("--fidelity", choices=("reference", "turbo"))
    p.add_argument("--reuse-grasp-state", action="store_true")
    p.add_argument("--robot-substeps", type=int, default=10)
    p.add_argument("--camera-every", type=int)
    p.add_argument("--record-every", type=int)
    p.add_argument("--camera-snapshot", action="store_true")
    p.add_argument("--steady-render-passes", type=int)
    p.add_argument("--async-camera-write", action="store_true")
    p.add_argument("--buffer-grasp-trace", action="store_true")
    p.add_argument("--stage-reset", choices=("reopen", "reload_in_place"))
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--ik-iterations", type=int, default=48)
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if args.fidelity is not None:
        from robotic_cutting_sim.runtime_profiles import fidelity_profile, native_50hz_cadence
        if args.runtime_profile is not None:
            p.error("Choose --fidelity or --runtime-profile")
        args.runtime_profile = fidelity_profile(args.fidelity, "isaac_sim")
        try:
            args.camera_every, args.record_every = native_50hz_cadence(args.camera_every, args.record_every)
        except ValueError as error:
            p.error(str(error))
    if args.camera_every is None:
        args.camera_every = 10
    if args.record_every is None:
        args.record_every = 10
    from agentic_robot.lab.sim.envs.rcs_gpu1 import _require_gpu1, launch_app_gpu1
    from robotic_cutting_sim import isaac_sim_runner
    from robotic_cutting_sim.agentic_trim_env import AgenticTrimEnv, open_env, write_json

    _require_gpu1()
    isaac_sim_runner.launch_app = launch_app_gpu1
    run = load_controller(args.controller, args.controller_sha256)

    args.out.mkdir(parents=True, exist_ok=False)
    control_dir = args.out / "controller"
    control_dir.mkdir()
    ctx = {"seed": args.seed, "split": "high_speed_engineering", "params": {},
           "out_dir": str(control_dir), "candidate_id": args.controller.stem, "policy_seed": 0}
    spec = {"command": [sys.executable, *sys.argv], "ctx": ctx,
               "control": {"path": str(args.controller), "sha256": args.controller_sha256},
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
               "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "runtime_profile": args.runtime_profile,
               "fidelity": args.fidelity,
               "robot_substeps": args.robot_substeps, "reuse_grasp_state": args.reuse_grasp_state,
               "camera_every": args.camera_every, "record_every": args.record_every,
               "camera_snapshot": args.camera_snapshot, "steady_render_passes": args.steady_render_passes,
               "async_camera_write": args.async_camera_write, "buffer_grasp_trace": args.buffer_grasp_trace,
               "stage_reset": args.stage_reset, "episodes": args.episodes, "ik_iterations": args.ik_iterations}
    write_json(args.out / "spec.json", spec)
    start = time.perf_counter()
    with open_env(output_dir=args.out / "env", seed=args.seed, stream_physics=True, memory_receipts=True,
                  native_sensor_runtime="isaac62_physx_offline_v1", material_diagnostics="device_reduction_v1",
                  runtime_profile=args.runtime_profile, reuse_grasp_state=args.reuse_grasp_state,
                  robot_substeps=args.robot_substeps, camera_every=args.camera_every,
                  record_every=args.record_every, camera_snapshot=args.camera_snapshot,
                  steady_render_passes=args.steady_render_passes, async_camera_write=args.async_camera_write,
                  buffer_grasp_trace=args.buffer_grasp_trace, stage_reset=args.stage_reset,
                  ik_iterations=args.ik_iterations) as env:
        factory_s = time.perf_counter() - start
        rows = [run_episode(env, args.out, run, ctx, start, factory_s, full_cadence=args.record_every == 1)]
        for index in range(1, args.episodes):
            out = args.out / f"warm_{index:03d}"
            control_dir = out / "controller"
            control_dir.mkdir(parents=True)
            (out / "env").mkdir()
            context = {**ctx, "out_dir": str(control_dir)}
            write_json(out / "spec.json", {**spec, "ctx": context, "resident_episode": index})
            memory_log = out / "env/runtime_memory.jsonl"
            os.environ["RCS_RUNTIME_MEMORY_LOG"] = str(memory_log)
            warm = AgenticTrimEnv(env.env, out / "env", env.task, env.manifest, seed=args.seed,
                                 stream_physics=True, memory_log=memory_log, record_every=args.record_every)
            try:
                rows.append(run_episode(warm, out, run, context, time.perf_counter(), 0.,
                                        full_cadence=args.record_every == 1))
            finally:
                warm.finalize("WORKER_EXIT")
            wall = time.perf_counter() - start
            complete = sum(row["status"] == "COMPLETE" and row["task_success"] for row in rows)
            write_json(args.out / "resident_episodes.json", {"episodes": rows, "wall_s": wall,
                       "complete_physical_episodes": complete, "episodes_per_gpu_hour": 3600 * complete / wall})
    return int(any(row["status"] != "COMPLETE" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
