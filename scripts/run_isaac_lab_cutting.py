#!/usr/bin/env python3
"""Execute and record actual Isaac Lab articulations + live retained TLMPM.

The kit-less cameras are explicit full-scene ideal Warp RGB-D. They never pass
native D455 or full Isaac Sim gates. No replay file is read by this runner.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import time

import numpy as np

from robotic_cutting_sim.recording import ROOT, manifest, new_output, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--material-device", default="cuda:0")
    parser.add_argument("--variant", choices=("D0", "D1"), default="D0")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--record-every", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.steps <= 400 or args.record_every < 1:
        parser.error("steps must be 1..400 and record-every positive")
    output = new_output(args.output_dir)
    write_json(output/"phase.json", {"phase": "IMPORTING_ISAAC_LAB", "source": "live_isaac_lab"})
    from isaaclab.app import launch_simulation
    from robotic_cutting_sim.isaac_lab_env import IsaacLabCuttingEnv, LabCuttingConfig, make_lab_cfg
    from robotic_cutting_sim.cut_topology import mechanical_components
    scene_path = args.scene
    if scene_path is None:
        from argparse import Namespace
        from robotic_cutting_sim.workcell import export
        scene_dir = output/"workcell"
        scene_dir.mkdir()
        export(Namespace(variant=args.variant, family="dome", seed=args.seed, spacing=.002,
            asset_root=None), scene_dir)
        scene_path = scene_dir/"workcell.usda"
    options = LabCuttingConfig(device=args.device, material_device=args.material_device,
        seed=args.seed, variant=args.variant, camera=not args.no_camera, feedback=not args.no_feedback,
        static_scene=str(scene_path.resolve()))
    cfg = make_lab_cfg(options)
    record = manifest(asdict(options), "live_isaac_lab")
    record["runtime"] = {"source": "live_isaac_lab", "kitless": True,
        "isaac_lab_release": "3.0.0", "isaac_lab_source_sha": subprocess.check_output(
            ["git", "-C", str(ROOT/".deps/IsaacLab"), "rev-parse", "HEAD"], text=True).strip(),
        "distribution_versions": {name: importlib.metadata.version(name) for name in
            ("isaaclab", "isaaclab-newton", "torch", "warp-lang", "newton", "usd-exchange")},
        "articulation_owner": "isaaclab.scene.InteractiveScene/NewtonManager",
        "task_api": "isaaclab.envs.DirectRLEnv", "physics_dt_s": .002, "decimation": 10,
        "material_dt_s": .0001, "camera": "ideal_warp_rgbd" if options.camera else "disabled",
        "native_d455_validated": False, "isaac_sim_kit_validated": False}
    record["acceptance_protocol_sha256"] = hashlib.sha256((ROOT/"configs/isaac_acceptance.json").read_bytes()).hexdigest()
    write_json(output/"manifest.json", record)
    (output/"rgbd").mkdir()
    writers = {}
    states, material, material_times, diagnostics, rows = [], [], [], [], []
    start = time.perf_counter()
    with launch_simulation(cfg, {"headless": True, "visualizer": [], "device": args.device}):
        write_json(output/"phase.json", {"phase": "CREATING_ACTUAL_LAB_ENV"})
        env = IsaacLabCuttingEnv(cfg, options)
        try:
            obs, info = env.reset(seed=args.seed)
            write_json(output/"setup.json", {**env.setup_receipt, "model_bodies": env.model.body_count,
                "model_joint_coordinates": env.model.joint_coord_count,
                "cutting_hand_body_index": env.hand, "cutting_hand_body_label": env.model.body_label[env.hand],
                "knife_shape_index": env.knife_shape, "knife_shape_label": env.model.shape_label[env.knife_shape],
                "knife_shape_body_index": int(env.model.shape_body.numpy()[env.knife_shape]),
                "T_hand_edge_xyz_xyzw": [0., 0., .18, 1., 0., 0., 0.],
                "body_labels": list(env.model.body_label),
                "articulations": [{"type": type(robot).__module__+"."+type(robot).__name__,
                    "joint_names": robot.joint_names, "body_names": robot.body_names} for robot in env.robots],
                "calibration": env.calibration, "observation_fields": list(obs["policy"])})
            write_json(output/"phase.json", {"phase": "LIVE_STEPPING", "initial_time_s": env.time_s})
            if options.camera:
                import imageio.v2 as imageio
                for name in ("global", "d455_0", "d455_1", "depth", "multiview"):
                    writers[name] = imageio.get_writer(str(output/(name+".mp4")), fps=50,
                        codec="libx264", macro_block_size=1, quality=7)

            def capture(step):
                state = env.measured()
                states.append(state)
                if step % 5 == 0 or step == args.steps:
                    material.append(env.material.debug_state())
                    material_times.append(env.time_s)
                if not options.camera:
                    return
                from PIL import Image, ImageDraw
                labeled = {}
                for name, frame in env.frames.items():
                    image = Image.fromarray(frame["rgb"])
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 640, 27), fill=(0, 0, 0))
                    draw.text((6, 8), f"LIVE ISAAC LAB 3.0 | ideal Warp RGB-D | {name} | t={env.time_s:.2f}s", fill="white")
                    labeled[name] = np.asarray(image)
                    writers[name].append_data(labeled[name])
                z = env.frames["d455_0"]["depth_m"]
                import matplotlib
                depth = (matplotlib.colormaps["turbo"](np.clip((z-.52)/.32, 0, 1))[..., :3]*255).astype(np.uint8)
                depth[z <= 0] = 0
                image = Image.fromarray(depth)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 640, 27), fill=(0, 0, 0))
                draw.text((6, 8), f"LIVE ISAAC LAB | ideal optical Z | range .52-.84 m | t={env.time_s:.2f}s", fill="white")
                depth = np.asarray(image)
                writers["depth"].append_data(depth)
                writers["multiview"].append_data(np.vstack([np.hstack([labeled["global"], labeled["d455_0"]]),
                    np.hstack([labeled["d455_1"], depth])]))
                if step % args.record_every == 0 or step == args.steps:
                    data = {name+"_"+key: frame[key] for name, frame in env.frames.items()
                            for key in ("rgb", "depth_m", "valid_mask")}
                    np.savez_compressed(output/"rgbd"/f"frame_{step:05d}.npz", timestamp_s=env.time_s,
                        surface_timestamp_s=env.time_s, **data)
                    vertices, faces = env.surface
                    np.savez_compressed(output/"rgbd"/f"surface_{step:05d}.npz", timestamp_s=env.time_s,
                        vertices=vertices, triangles=faces)
                    write_json(output/"rgbd"/f"surface_{step:05d}.json", env.surface_metadata)

            capture(0)
            for step in range(1, args.steps+1):
                tick = time.perf_counter()
                env.step(env.reference_action())
                state = env.measured()
                data = env.material.debug_state()
                diagnostic = env.last_diagnostics.copy()
                diagnostics.append(diagnostic)
                finite = all(np.isfinite(data[key]).all() for key in ("positions", "velocities", "F"))
                row = {"step": step, "time_s": env.time_s, "runtime_time_s": diagnostic["runtime_timestamp_s"],
                    "finite": finite, "min_detF": float(np.linalg.det(data["F"]).min()),
                    "max_speed_m_s": float(np.linalg.norm(data["velocities"], axis=1).max()),
                    "tracking_error_m": float(np.linalg.norm(state["tool_pose"][:3]-diagnostic["desired_tool_pose"][:3])),
                    "material_reaction_n": float(np.linalg.norm(diagnostic["next_material_wrench_world"][:3])),
                    "tool_penetration_m": float(diagnostic["material_contact_statistics"][3]),
                    "wall_s": time.perf_counter()-tick}
                rows.append(row)
                capture(step)
                if step % 25 == 0:
                    print(f"LIVE_LAB {step}/{args.steps} t={env.time_s:.2f} runtime={row['runtime_time_s']:.3f} "
                          f"track={row['tracking_error_m']*1000:.3f}mm detF={row['min_detF']:.3f}", flush=True)
                    write_json(output/"phase.json", {"phase": "LIVE_STEPPING", **row})
                if not finite:
                    raise FloatingPointError("Nonfinite live material state")
            bed = env.material.bed
            count, component_ids = mechanical_components(bed.nodes.numpy(), bed.weights.numpy(), data["codes"],
                1, gradients=bed.gradients.numpy())
            np.savez_compressed(output/"robot_state.npz", time_s=[s["timestamp_s"] for s in states],
                q=[s["q"] for s in states], qdot=[s["qdot"] for s in states],
                body_pose=[s["body_pose"] for s in states], hand_pose_xyzw=[s["hand_pose"] for s in states],
                tool_pose_xyzw=[s["tool_pose"] for s in states],
                desired_tool_pose_xyzw=np.vstack([states[0]["tool_pose"], [d["desired_tool_pose"] for d in diagnostics]]),
                next_material_wrench_world=[d["next_material_wrench_world"] for d in diagnostics],
                applied_previous_material_wrench_world=[d["applied_previous_material_wrench_world"] for d in diagnostics],
                material_reaction_impulse_world=[d["current_material_reaction_impulse_world"] for d in diagnostics])
            np.savez_compressed(output/"physics_state.npz", time_s=material_times,
                positions=[s["positions"] for s in material], velocities=[s["velocities"] for s in material],
                codes=[s["codes"] for s in material], mass_kg=env.material.masses,
                reference_positions=env.material.points, reference_indices=env.material.lattice,
                final_F=data["F"], final_Fpi=data["Fpi"], final_plastic_strain=data["plastic_strain"],
                component_ids=component_ids, reference_stencil_nodes=bed.nodes.numpy(),
                reference_stencil_weights=bed.weights.numpy(), reference_stencil_gradients=bed.gradients.numpy())
            env.sim.stage.Export(str(output/"final_live_scene.usda"))
            write_json(output/"diagnostics.json", diagnostics)
            write_json(output/"trace.json", rows)
            result = {"source": "live_isaac_lab", "status": "PASS_EXECUTION" if args.steps == 400 else "PASS_SMOKE",
                "actual_scene_environment_task_api": True, "steps": args.steps, "simulation_s": env.time_s,
                "runtime_s": env.sim.physics_manager.get_simulation_time(), "wall_s": time.perf_counter()-start,
                "final_mechanical_components": count,
                "component_masses_kg": [float(env.material.masses[component_ids == i].sum()) for i in np.unique(component_ids)],
                "native_d455_validated": False, "isaac_sim_kit_validated": False,
                "full_acceptance": "PENDING_SEPARATE_QUANTITATIVE_AUDIT"}
            write_json(output/"result.json", result)
            write_json(output/"phase.json", {"phase": "COMPLETE", **result})
            print(result, flush=True)
        finally:
            for writer in writers.values():
                writer.close()
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
