"""Time the real headless Rubber vector API, with paired slots and seeds."""
import argparse
import json
import os
import platform
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RL07 = runpy.run_path(str(ROOT / "scripts/rubber/p2_6/run_rubber_rl_gpu.py"))
PAIR_SLOTS = ("L100_t4", "L150_t4")


def eager_sample(env, substeps):
    """Separate disposable kernel sample after the episode, with contact and no CUDA Graph replay."""
    from collections import defaultdict

    wp, mat = env.wp, env.material
    solver = mat.solver
    with env._stream():
        mat._begin_control_step()
        solver.prepare_motion_table()
        wp.timing_begin(wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN)
        try:
            for _ in range(min(substeps, solver.substeps)):
                solver.step()
        finally:
            events = wp.timing_end()
    kernels = defaultdict(lambda: {"wall_ms": 0.0, "launches": 0})
    for event in events:
        kernels[event.name]["wall_ms"] += event.elapsed
        kernels[event.name]["launches"] += 1
    return {"substeps": min(substeps, solver.substeps), "kernels": dict(sorted(
                kernels.items(), key=lambda pair: -pair[1]["wall_ms"])),
            "contact_events": solver.ct_counts.numpy().tolist(),
            "diagnostic_boundary": "eager solver substeps only, after the episode; excludes S1 and policy API"}


def execute(args, options, slots, gpu):
    import torch
    import warp as wp

    from robotic_cutting_sim.rl.rubber_contract import contract_actions
    from robotic_cutting_sim.rl.rubber_env import TIP_MARKER
    from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily
    from robotic_cutting_sim.rl.vector_env import open_vector_env
    from robotic_cutting_sim.tlmpm_rubber_v1 import DIAG_COLUMNS
    from robotic_cutting_sim.rubber_material import p_wave_speed

    start = time.perf_counter()
    n = len(slots)
    seeds = list(range(args.seed, args.seed + n))
    family = RubberStripTaskFamily(slots, max_episode_steps=args.warmup + args.steps + args.release, **options)
    gpu.sample("before_build")
    with open_vector_env(n, task=family, backend="kinematic_grip_batched", device=args.device, strict=False,
                         capture_graph=args.mode == "graph") as env:
        wp.synchronize_device(args.device)
        build_s = time.perf_counter() - start
        t0 = time.perf_counter()
        obs, _ = env.reset(seeds=seeds)
        wp.synchronize_device(args.device)
        reset_s = time.perf_counter() - t0
        mat, solver = env.material, env.material.solver
        full_steps = args.warmup + args.steps + args.release
        actions = [contract_actions(n, i, env.torch_device) for i in range(2)]
        zero = torch.zeros_like(actions[0])
        columns = [DIAG_COLUMNS.index(name) for name in ("J_min", "J_max", "max_speed_m_s", "cfl_eff_max")]
        records = []
        memory = [gpu.own_mib()]
        episode_start = time.perf_counter()
        for k in range(full_steps):
            release = k >= args.warmup + args.steps
            if k == args.warmup + args.steps:
                # Diagnostic intervention only. The policy API models an ideal grip without an open action.
                # Drop the Dirichlet clamp and the separate tip load, leaving gravity and table friction active.
                with env._stream():
                    solver.group_active.zero_()
                    env.traction_t.zero_()
            if release:
                action = zero
            elif args.drive == "goal":
                action = torch.zeros_like(zero)
                tip = obs["policy"][:, 20+3*TIP_MARKER:23+3*TIP_MARKER]
                action[:, :2] = ((env.goal[:, :2] - tip[:, :2]) / family.max_translation).clamp(-1, 1)
            else:
                action = actions[k % 2]
            t0 = time.perf_counter()
            obs, reward, term, trunc, info = env.step(action)
            torch.cuda.synchronize(env.torch_device)
            wall = time.perf_counter() - t0
            x = mat.marker_x.numpy().reshape(n, mat.K, 3)
            summary = mat.summary.numpy()[:, columns]
            plane_force = mat.contact.plane_wrench.numpy()[:, 0, :3]
            events = solver.ct_counts.numpy()[:, 0]
            pen = solver.ct_pen_max.numpy()
            finite = np.isfinite(x).all() and np.isfinite(summary).all() and np.isfinite(plane_force).all()
            finite = finite and bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(reward).all())
            records.append({
                "step": k + 1, "phase": "release" if release else "warmup" if k < args.warmup else "drag",
                "wall_s": wall, "markers_m": x.tolist(), "plane_force_n": plane_force.tolist(),
                "tip_speed_m_s": np.linalg.norm(mat.marker_v.numpy().reshape(n, mat.K, 3)[:, -1], axis=1).tolist(),
                "contact_events": events.tolist(), "penetration_m": pen.tolist(),
                "J_min": summary[:, 0].tolist(), "J_max": summary[:, 1].tolist(),
                "speed_max_m_s": summary[:, 2].tolist(), "end_step_elastic_cfl": summary[:, 3].tolist(),
                "cfl_guard_violations_cumulative": solver.cfl_violations.numpy().tolist(),
                "j_domain_violations_cumulative": solver.violations.numpy().tolist(),
                "terminated": term.tolist(), "truncated": trunc.tolist(), "finite": bool(finite),
                "invalid_reason": info["invalid_reason"].tolist(), "action_status": info["action_status"].tolist(),
                "grip_force_n": mat.wrench.numpy()[:, 0, :3].tolist(),
                "tip_to_grip_m": np.linalg.norm(x[:, -1] - x[:, 0], axis=1).tolist(),
                "sag_m": (x[:, :, 2] - x[:, :1, 2]).min(axis=1).tolist(),
                "tip_goal_distance_m": info["tip_distance_m"].tolist(),
            })
            if k == 0 or k == args.warmup - 1 or (k + 1) % 10 == 0 or k + 1 == full_steps:
                positions = solver.x.numpy()
                records[-1]["min_particle_z_m"] = [float(seg[:, 2].min()) for seg in
                                                   np.split(positions, mat.packed.particle_offset[1:-1])]
                gpu.sample(f"step_{k + 1}")
                memory.append(gpu.own_mib())
                print(f"step {k+1}/{full_steps} wall {wall:.3f}s invalid {records[-1]['invalid_reason']}", flush=True)

        episode_s = time.perf_counter() - episode_start
        evaluate = env.evaluate() if not args.release else None
        if args.release:
            release_active = solver.group_active.numpy().tolist()
        else:
            release_active = None
        receipt = env.receipt()
        drag_rows = records[args.warmup:args.warmup + args.steps]
        wall = sum(r["wall_s"] for r in drag_rows)
        bad = any(not r["finite"] or any(r["terminated"]) or any(r["invalid_reason"]) for r in records)
        contact_steps = np.count_nonzero(np.asarray([r["contact_events"] for r in drag_rows]), axis=0).tolist()
        forces = np.asarray([r["plane_force_n"] for r in drag_rows])
        loaded_steps = np.count_nonzero(np.abs(forces[:, :, 2]) > 1e-8, axis=0).tolist()
        result = {
            "status": "PASS" if not bad and min(loaded_steps) > 0 and all(records[-1]["truncated"]) else "FAIL",
            "slots": slots, "seeds": seeds, "num_envs": n, "profile": receipt["profile"]["id"],
            "resolution": family.resolution, "control_dt_s": family.control_dt, "dt_s": solver.dt,
            "substeps": solver.substeps, "particles": np.diff(mat.packed.particle_offset).tolist(),
            "dx_m": [family.spec(s).grid_dx() for s in slots], "build_s": build_s, "reset_s": reset_s,
            "first_step_s": records[0]["wall_s"], "episode_steps": full_steps, "episode_s": episode_s,
            "warmup_steps": args.warmup, "measured_drag_steps": args.steps, "release_steps": args.release,
            "measured_contact_steps_per_env": contact_steps, "loaded_table_steps_per_env": loaded_steps,
            "nominal_cfl": solver.dt / solver.dx * p_wave_speed(
                family.step_material.mu0, family.step_material.kappa, family.step_material.density),
            "actual_guard": {"id": solver.model.receipt().get("guard"), "limit": solver.cfl_guard_limit,
                             "cumulative_violations": records[-1]["cfl_guard_violations_cumulative"]},
            "batch_step_s": wall / args.steps,
            "aggregate_env_steps_s": n * args.steps / wall, "single_env_sim_s_per_wall_s": family.control_dt * args.steps / wall,
            "aggregate_sim_s_per_wall_s": n * family.control_dt * args.steps / wall,
            "episode_wall_episodes_per_gpu_hour": 3600 * n / (build_s + reset_s + episode_s),
            "steady_projection_250_step_episodes_per_gpu_hour": 3600 * n * args.steps / (250 * wall),
            "max_sampled_process_mib": max((v for v in memory if np.isfinite(v)), default=None),
            "gpu_sharing": gpu.sharing(), "mode": args.mode, "drive": args.drive,
            "graph": mat.capture_receipt, "evaluation": evaluate, "released_group_active": release_active,
            "beta": family.bulk_ratio, "cfl": family.profile["cfl"], "prony_terms": solver.M,
        }
        if args.sample_eager_substeps:
            result["eager_kernel_sample"] = eager_sample(env, args.sample_eager_substeps)
    result["total_wall_s"] = time.perf_counter() - start
    return result, receipt, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", default=None)
    p.add_argument("--speed-profile")
    p.add_argument("--mode", choices=("graph", "eager"), default="graph")
    p.add_argument("--sample-eager-substeps", type=int, default=0)
    p.add_argument("--drive", choices=("fixed", "goal"), default="fixed")
    p.add_argument("--slots", choices=("paired", "full", "short"), default="paired")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--release", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu-index", type=int, required=True)
    args = p.parse_args()
    if args.num_envs < 1 or args.warmup < 10 or args.steps < 1 or args.release < 0 or args.sample_eager_substeps < 0:
        p.error("need at least ten warm-up steps, one measured step and a nonnegative release length")
    if args.release and args.sample_eager_substeps:
        p.error("sample the eager contact kernels with the grip active")
    templates = {"short": PAIR_SLOTS[:1], "paired": PAIR_SLOTS, "full": RL07["FULL_ORDER"]}[args.slots]
    slots = [templates[i % len(templates)] for i in range(args.num_envs)]
    args.out.mkdir(parents=True, exist_ok=False)
    gpu = RL07["Gpu"](args.gpu_index)
    meta = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_paths": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "python": platform.python_version(), "command": [sys.executable, *sys.argv],
        "gpu_index": args.gpu_index, "gpu": RL07["sh"]("nvidia-smi", f"--id={args.gpu_index}",
                                                     "--query-gpu=name,driver_version", "--format=csv,noheader"),
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OPENBLAS_NUM_THREADS")},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    try:
        if gpu.free_mb() < RL07["MIN_FREE_MB"] or gpu.apps():
            raise RuntimeError(f"GPU admission failed: {gpu.free_mb():.0f} MiB free, compute apps {gpu.apps()}")
        result, receipt, trace = execute(args, {"resolution": args.resolution, "speed_profile": args.speed_profile},
                                         slots, gpu)
    except Exception:
        import traceback
        failure = traceback.format_exc()
        (args.out / "failure.txt").write_text(failure)
        raise
    for name, payload in (("benchmark", result), ("receipt", receipt), ("trace", trace)):
        (args.out / f"{name}.json").write_text(json.dumps(payload, indent=2, default=RL07["np"].float64) + "\n")
    print(json.dumps(result, default=RL07["np"].float64), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
