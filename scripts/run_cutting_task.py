#!/usr/bin/env python3
"""Exercise the same public CUT/TRIM API in a real Isaac Lab or Sim runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import hashlib
import shutil

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from robotic_cutting_sim.live_env import open_cutting_env
from robotic_cutting_sim.recording import write_json


def worker(args):
    options = {"device": args.device} if args.device else {}
    if args.particle_spacing is not None:
        options["particle_spacing_m"] = args.particle_spacing
    if args.runtime == "isaac_sim":
        options.update(gpu=args.gpu, device=args.device or f"cuda:{args.gpu}", steps=args.steps)
    elif args.device:
        options["material_device"] = args.device
    try:
        with open_cutting_env(args.task, runtime=args.runtime, output_dir=args.output_dir,
                seed=args.seed, action_mode=args.action_mode, camera_layout=args.camera_layout,
                include_global_camera=args.include_global_camera, task_variant=args.task_variant, **options) as env:
            implementation = args.output_dir/"implementation"
            implementation.mkdir()
            source = [Path(__file__).resolve(), Path(__file__).resolve().parents[1]/"src/robotic_cutting_sim/live_env.py"]
            for path in source:
                shutil.copyfile(path, implementation/path.name)
            write_json(implementation/"sha256.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source})
            observation, setup = env.reset(seed=args.seed)
            write_json(args.output_dir/"api_setup.json", setup)
            write_json(args.output_dir/"task.json", env.task.to_dict())
            (args.output_dir/"api_observations").mkdir()
            trace = []
            for index in range(args.steps+1):
                if index:
                    action = None
                    if args.exercise_actions and index <= 2:
                        if index == 1:
                            action = {"mode": "cartesian_edge_pose_world",
                                "target": observation["proprio"]["tool_pose"], "cut_enabled": False}
                        else:
                            target = observation["proprio"]["q"][0].copy()
                            target[0] += .0001
                            action = {"mode": "joint_position_rad", "target": target, "cut_enabled": False}
                    observation, reward, terminated, truncated, info = env.step(action)
                    trace.append({"step": index, "timestamp_s": observation["timestamp_s"],
                        "action_mode": action["mode"] if action else "scripted_reference",
                        "actual_tool_pose_world_xyzw": observation["proprio"]["tool_pose"],
                        "commanded_tool_pose_world_xyzw": (env.task.reference_edge_pose_world(env.time_s)
                            if action is None else action["target"] if action["mode"] == "cartesian_edge_pose_world" else None),
                        "terminated": terminated, "truncated": truncated})
                if index % args.record_every == 0 or index == args.steps:
                    from PIL import Image
                    payload = {name+"_"+field: observation[field][name]
                        for name in observation["rgb"] for field in ("rgb", "depth_m", "valid_mask")}
                    np.savez_compressed(args.output_dir/"api_observations"/f"frame_{index:05d}.npz",
                        timestamp_s=observation["timestamp_s"], q=observation["proprio"]["q"],
                        qdot=observation["proprio"]["qdot"], **payload)
                    Image.fromarray(env.render("global")).save(args.output_dir/"api_observations"/f"global_{index:05d}.png")
                    write_json(args.output_dir/"api_observations"/f"calibration_{index:05d}.json", observation["calibration"])
            evaluation = env.evaluate()
            write_json(args.output_dir/"task_evaluation.json", evaluation)
            write_json(args.output_dir/"api_trace.json", trace)
            result = {"status": "PASS_EXECUTION" if evaluation["episode_complete"] else "PASS_SMOKE",
                "runtime_source": env.runtime_source, "task_id": env.task.task_id,
                "steps": args.steps, "simulation_s": env.time_s,
                "task_metric_status": evaluation["status"], "task_success": evaluation["success"],
                "api_exercised": ["reset", "step", "observe", "render", "evaluate", "close"],
                "explicit_actions_exercised": args.exercise_actions,
                "policy_observation_keys": list(env.observe()),
                "full_physics_camera_acceptance": "SEPARATE_AUDIT_REQUIRED"}
            # Save before Kit teardown, which can swallow Python exit status.
            write_json(args.output_dir/"result.json", result)
            print(json.dumps(result), flush=True)
        return 0
    except Exception:
        if args.output_dir.exists():
            write_json(args.output_dir/"result.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("CUT-01", "TRIM-01"), default="CUT-01")
    parser.add_argument("--runtime", choices=("isaac_sim", "isaac_lab"), default="isaac_sim")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-variant", choices=("fixed", "randomized"), default="fixed")
    parser.add_argument("--camera-layout", choices=("legacy_fixed_d455", "wrist_d405_global_d455"), default="wrist_d405_global_d455")
    parser.add_argument("--include-global-camera", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--particle-spacing", type=float, help="Explicit initial task quadrature in metres; frozen defaults remain available")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--action-mode", choices=("cartesian_edge_pose_world", "joint_position_rad"), default="cartesian_edge_pose_world")
    parser.add_argument("--exercise-actions", action="store_true", help="Short control smoke: explicit Cartesian then joint action")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.steps <= 400 or args.record_every < 1 or (args.exercise_actions and args.steps < 2):
        parser.error("steps must be1..400; action smoke needs at least2; recording interval must be positive")
    if args.worker:
        return worker(args)
    if args.output_dir.exists():
        parser.error("Output already exists; preserve runs and use a new directory")
    environment = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1"}
    start = time.perf_counter()
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], env=environment)
    result = args.output_dir/"result.json"
    receipt = json.loads(result.read_text()) if result.exists() else {"status": "FAIL", "reason": "No worker receipt"}
    if process.returncode and receipt["status"].startswith("PASS"):
        receipt = {**receipt, "status": "FAIL", "reason": "Runtime failed after writing a worker receipt"}
    if args.output_dir.exists():
        write_json(args.output_dir/"process_receipt.json", {"returncode": process.returncode, "wall_s": time.perf_counter()-start})
        write_json(result, receipt)
    return 0 if process.returncode == 0 and receipt["status"] in ("PASS_EXECUTION", "PASS_SMOKE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
