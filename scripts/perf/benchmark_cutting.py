"""Measure contact windows and complete headless CUT or measured-motion TRIM episodes."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/rl_ready"))


def batch_task(args):
    from robotic_cutting_sim.rl import ConfiguredTaskFamily, TaskFamily
    from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
    from robotic_cutting_sim.rl.reference_policy import ReferenceCommandPolicyV1, reference_schedule

    n = args.num_envs
    numerical_overrides = {
        key: value for key, value in (
            ("particle_spacing_m", args.particle_spacing_m),
            ("material_dt_s", args.material_dt_s),
        ) if value is not None
    }
    if args.task == "cut":
        family = ConfiguredTaskFamily.from_cut_workload(
            n, runtime_profile=args.runtime_profile, **numerical_overrides)

        def make_step(env):
            schedule = reference_schedule(env.tasks[0])
            first_armed = int(np.flatnonzero(schedule["cut_enabled"])[0]) + 1
            drive = ReferenceCommandPolicyV1.from_tasks(env.tasks, device=env.torch_device)
            return lambda obs, info: env.step(drive(obs, info)), first_armed, {}

        return family, make_step

    import run_vector_milestones as replay
    from run_perf_acceptance import wp17_record
    if args.wp17 is None:
        raise ValueError("--wp17 is required for TRIM")
    scene_seeds = [args.scene_seeds[e % len(args.scene_seeds)] for e in range(n)]
    order = sorted(set(scene_seeds))
    replay.WP17 = wp17_record(args.wp17, order)
    inputs = replay.load_inputs(scene_seeds)
    family = replay.family_for(scene_seeds, inputs)
    if args.runtime_profile is not None or numerical_overrides:
        family = replace(family, runtime_profile=args.runtime_profile, **numerical_overrides)

    def make_step(env):
        if args.runtime_profile is None and not numerical_overrides:
            replay.check_payloads(env, inputs)
        source = [inputs[x]["trajectory"] for x in order]
        first_armed = min(int(np.flatnonzero(t.cutting)[0]) + 1 for t in source)
        drive = MeasuredTrajectoryDriver(source, [order.index(x) for x in scene_seeds], device=env.torch_device)

        def step(obs, info):
            pose, cut, pads = drive.inputs(info["step_count"])
            return env.step_measured_edge(pose, cut, pad_boxes=pads)

        return step, first_armed, {str(seed): inputs[seed]["trajectory"].checks for seed in order}

    return family, make_step


def execute(args, telemetry):
    import torch
    import warp as wp
    from robotic_cutting_sim.rl import open_vector_env
    import run_vector_milestones as replay
    from run_perf_acceptance import stats_slots

    start = time.perf_counter()
    family, make_step = batch_task(args)
    n = args.num_envs
    kwargs = {"evaluation_stream": {"every": 20}} if args.full else {}
    with open_vector_env(n, task=family, device=args.device, strict=False,
                         particle_contact_mode=args.contact_mode, **kwargs) as env:
        sync = lambda: torch.cuda.synchronize(env.torch_device)
        sync()
        build_s = time.perf_counter() - start
        t0 = time.perf_counter()
        obs, info = env.reset()
        sync()
        reset_s = time.perf_counter() - t0
        drive, first_armed, replay_checks = make_step(env)
        # The CUT controller arms during approach. The first Reference run found
        # tool contact starting at step 234 (armed at 191) in this fixed workload.
        offset = 43 if args.task == "cut" else 15
        first = args.window_start if args.window_start is not None else first_armed + offset
        last = first + args.window_steps - 1
        if not 2 <= first <= last <= env.max_episode_steps:
            raise ValueError(f"invalid measured window [{first}, {last}] for {env.max_episode_steps} steps")
        steps = env.max_episode_steps if args.full else last
        replay._grouped_query_audit = replay._define_kernels()
        monitor = replay.L2Monitor(env)
        rows, contacts, tool_contacts = [], [], []
        telemetry.set_phase("run")
        begin = time.perf_counter()
        for k in range(1, steps + 1):
            t0 = time.perf_counter()
            obs, reward, term, trunc, info = drive(obs, info)
            sync()
            rows.append(time.perf_counter() - t0)
            monitor.update()
            if first <= k <= last:
                pair, tool = stats_slots(env)
                contacts.append(pair.tolist())
                tool_contacts.append(tool.tolist())
            if k in (last, steps):
                monitor.audit(k)
            if k % 50 == 0 or k in (first, last, steps):
                print(f"step {k}/{steps} wall {rows[-1]:.3f}s", flush=True)
        sync()
        episode_s = time.perf_counter() - begin
        telemetry.set_phase("evaluate")
        t0 = time.perf_counter()
        l2 = monitor.result()
        evaluation = env.evaluate() if args.full else None
        eval_s = time.perf_counter() - t0
        wall = sum(rows[first - 1:last])
        contact_count = np.count_nonzero(tool_contacts, axis=0).tolist()
        steps_actual = info["step_count"].tolist()
        complete = all(k == env.max_episode_steps for k in steps_actual) and all(trunc.tolist())
        mass = env.material.solver.mass.numpy()
        offsets = env.material.packed.particle_offset
        res = {
            "status": ("COMPLETE" if complete else "INCOMPLETE") if args.full else "WINDOW_COMPLETE",
            "task": args.task,
            "runtime_profile": args.runtime_profile,
            "contact_mode": args.contact_mode,
            "drive": "reference_command_policy_v1" if args.task == "cut" else "measured_native_trajectory",
            "num_envs": n,
            "steps": steps,
            "control_dt_s": env.control_dt,
            "numerics": [{
                "particle_spacing_m": task.particle_spacing_m, "grid_dx_m": task.grid_dx_m,
                "material_dt_s": family.material_dt_s,
            } for task in env.tasks],
            "build_s": build_s,
            "reset_s": reset_s,
            "episode_s": episode_s,
            "first_step_s": rows[0],
            "eval_s": eval_s,
            "terminal": term.tolist(),
            "truncated": trunc.tolist(),
            "episode_ends": steps_actual, "final_mass_kg_per_env": [
                float(row.sum(dtype=np.float64)) for row in np.split(mass, offsets[1:-1])],
            "window_steps": args.window_steps,
            "window_first_last": [first, last],
            "window_s": wall,
            "batch_step_s": wall / args.window_steps,
            "aggregate_env_steps_s": n * args.window_steps / wall,
            "single_env_sim_s_per_wall_s": env.control_dt * args.window_steps / wall,
            "aggregate_sim_s_per_wall_s": n * env.control_dt * args.window_steps / wall,
            "contact_window_steps_per_env": contact_count,
            "contact_window_verdict": "PASS" if min(contact_count) else "FAIL",
            "step_walls_s": rows,
            "tool_contact_counts": tool_contacts,
            "cross_component_contact_counts": contacts,
            "l2": l2,
            "replay_checks": replay_checks,
            "receipt": env.receipt(),
            "versions": {"warp": wp.config.version, "torch": torch.__version__},
        }
        if evaluation is not None:
            res["evaluation"] = evaluation
            t0 = time.perf_counter()
            names = []
            for e in range(n):
                name = f"final_env{e:02d}.npz"
                np.savez_compressed(args.out / name, **env.snapshot_world(e), **env.evaluation_reference(e))
                names.append(name)
            res["terminal_exports"] = names
            res["terminal_export_wall_s"] = time.perf_counter() - t0
    res["total_wall_s"] = time.perf_counter() - start
    res["episodes_per_gpu_hour"] = 3600.0 * n / res["total_wall_s"] if res["status"] == "COMPLETE" else None
    return res


def main():
    from robotic_cutting_sim.rl.telemetry import GpuTelemetry
    from run_perf_acceptance import write_json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("cut", "trim"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--window-start", type=int)
    p.add_argument("--window-steps", type=int, default=60)
    p.add_argument("--full", action="store_true")
    p.add_argument("--wp17", type=Path)
    p.add_argument("--scene-seeds", type=int, nargs="+", default=(512, 15))
    p.add_argument("--runtime-profile")
    p.add_argument("--particle-spacing-m", type=float)
    p.add_argument("--material-dt-s", type=float)
    p.add_argument("--contact-mode", default="frozen_v1")
    args = p.parse_args()
    if args.num_envs < 1 or args.window_steps < 1:
        p.error("num-envs and window-steps must be positive")
    if args.runtime_profile is not None and (args.particle_spacing_m is not None or args.material_dt_s is not None):
        p.error("--runtime-profile fixes resolution and material dt")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "meta.json", {
        "command": [sys.executable, *sys.argv],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_paths": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "physical_gpu": args.gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "blas": os.environ.get("OPENBLAS_NUM_THREADS"),
    })
    telemetry = GpuTelemetry(args.gpu_index, interval_s=1.0)
    telemetry.set_phase("baseline")
    telemetry.start()
    try:
        result = execute(args, telemetry)
    except Exception:
        import traceback
        (args.out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        sharing = telemetry.stop()
        with (args.out / "telemetry.jsonl").open("x") as stream:
            for row in telemetry.samples:
                stream.write(json.dumps(row, default=str) + "\n")
    result["gpu_sharing"] = sharing
    write_json(args.out / "benchmark.json", result)
    print(json.dumps({k: result[k] for k in ("status", "window_first_last", "aggregate_env_steps_s",
                                            "contact_window_verdict", "total_wall_s")}), flush=True)
    return 0 if (result["contact_window_verdict"] == "PASS" and result["l2"]["status"] == "PASS"
                 and result["status"] != "INCOMPLETE") else 1


if __name__ == "__main__":
    sys.exit(main())
