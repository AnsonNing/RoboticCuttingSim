#!/usr/bin/env python3
"""Record the live articulated cutting API, including full-scene ideal RGB-D."""
import argparse
import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from robotic_cutting_sim.recording import manifest, new_output, write_json
from robotic_cutting_sim.research_env import ResearchCuttingEnv, ResearchEnvConfig


def check_observation(obs, time_s):
    assert set(obs) == {"rgb", "depth_m", "valid_mask", "calibration", "timestamp_s", "proprio"}
    assert set(obs["proprio"]) == {"q", "qdot", "tool_pose"}
    assert abs(obs["timestamp_s"]-time_s) < 1e-8
    assert obs["proprio"]["q"].shape == (2, 7)
    for name in ("d455_0", "d455_1"):
        rgb, depth, mask = (obs[key][name] for key in ("rgb", "depth_m", "valid_mask"))
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        assert depth.shape == mask.shape == (480, 640) and depth.dtype == np.float32
        assert np.array_equal(mask, depth > 0) and np.isfinite(depth).all() and mask.any()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--record-every", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--scene", type=Path)
    args = p.parse_args()
    output = new_output(args.output_dir)
    for directory in ("observations", "global_debug", "oracle_debug"):
        (output/directory).mkdir()
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    receipt = manifest(parameters, "live_articulated_cutting_env")
    root = Path(__file__).resolve().parents[1]
    sources = [root/"src/robotic_cutting_sim"/name for name in
               ("research_env.py", "coupled_robot.py", "robot.py", "tlmpm.py", "cut_topology.py", "cutting_surface.py")]
    sources.append(Path(__file__))
    (output/"implementation").mkdir()
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for source in sources:
        shutil.copyfile(source, output/"implementation"/source.name)
    receipt["implementation_sha256"] = source_hashes
    write_json(output/"manifest.json", receipt)
    config = ResearchEnvConfig(device=args.device, max_steps=args.steps,
                              scene_path=str(args.scene.resolve()) if args.scene else None)
    records, tools, joints, rows = [], [], [], []
    changed = {name: 0 for name in ("d455_0", "d455_1")}
    tick = time.perf_counter()
    with ResearchCuttingEnv(config) as env:
        obs, setup = env.reset(seed=42)
        write_json(output/"setup.json", setup)
        write_json(output/"calibration.json", obs["calibration"])
        original = {name: obs["depth_m"][name].copy() for name in changed}
        original_q = obs["proprio"]["q"].copy()
        def record(index):
            for name in changed:
                np.savez_compressed(output/"observations"/f"{name}_{index:04d}.npz",
                    rgb=obs["rgb"][name], depth_m=obs["depth_m"][name],
                    valid_mask=obs["valid_mask"][name], timestamp_s=obs["timestamp_s"])
                Image.fromarray(obs["rgb"][name]).save(output/"observations"/f"{name}_{index:04d}.png")
            Image.fromarray(env.render()).save(output/"global_debug"/f"frame_{index:04d}.png")
            state = env.debug_material_state()
            records.append({k: state[k] for k in ("positions", "velocities", "codes", "timestamp_s")})
        check_observation(obs, 0.)
        record(0)
        for step in range(args.steps):
            obs, reward, terminated, truncated, info = env.step()
            check_observation(obs, (step+1)*config.control_dt)
            assert reward == 0 and not terminated and truncated == (step == args.steps-1)
            assert env.observe() is obs  # same state, no stale/cross-time reconstruction
            tools.append(obs["proprio"]["tool_pose"].copy())
            joints.append(obs["proprio"]["q"].copy())
            rows.append({"time_s": obs["timestamp_s"], **info})
            if (step+1) % args.record_every == 0 or step == args.steps-1:
                record((step+1)//args.record_every+int((step+1) % args.record_every != 0))
                for name in changed:
                    depth = obs["depth_m"][name]
                    common = (depth > 0) & (original[name] > 0)
                    changed[name] += int((common & (np.abs(depth-original[name]) > 1e-5)).sum())
            if (step+1) % 25 == 0:
                print(f"step={step+1}/{args.steps} t={obs['timestamp_s']:.2f}s "
                      f"camera={info['camera_wall_s']:.3f}s", flush=True)
        end_components = env.debug_diagnostics()["surface"]["mechanical_components"]
        mass = env.core.masses.copy()
        np.savez_compressed(output/"oracle_debug/physics_state.npz",
            positions=np.array([r["positions"] for r in records]), velocities=np.array([r["velocities"] for r in records]),
            codes=np.array([r["codes"] for r in records]), time_s=[r["timestamp_s"] for r in records], mass_kg=mass)
        np.savez_compressed(output/"robot_proprioception.npz", q=joints, tool_pose_xyzw=tools,
                            time_s=np.arange(1, len(tools)+1)*config.control_dt)
        try:
            env.step()
        except RuntimeError:
            truncation_guard = True
        else:
            truncation_guard = False
        again, _ = env.reset(seed=42)
        reset_error = float(np.abs(original_q-again["proprio"]["q"]).max())
        reset_depth_error = max(float(np.abs(original[n]-again["depth_m"][n]).max()) for n in changed)
    result = {"status": "PASS" if truncation_guard and reset_error < 1e-6 and reset_depth_error < 1e-6 else "FAIL",
        "scope": "live measured Panda + persistent-cut MPM + full-scene ideal RGB-D API",
        "steps": args.steps, "simulation_s": args.steps*config.control_dt, "wall_s": time.perf_counter()-tick,
        "final_mechanical_components": end_components,
        "changed_depth_pixels_10um_accumulated": changed, "same_seed_reset_q_error_rad": reset_error,
        "same_seed_reset_depth_error_m": reset_depth_error, "truncation_guard": truncation_guard,
        "observation_contains_force_oracle_global": False,
        "mean_camera_wall_s": float(np.mean([r["camera_wall_s"] for r in rows])),
        "max_camera_wall_s": max(r["camera_wall_s"] for r in rows),
        "simulated_d455_validated": False, "physics_calibrated": False,
        "capabilities": ResearchCuttingEnv.capabilities()}
    write_json(output/"step_timings.json", rows)
    result["source_sha256"] = source_hashes
    write_json(output/"result.json", result)
    print(result, flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
