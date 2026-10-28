#!/usr/bin/env python3
"""Execute and record actual Isaac Lab articulations + live retained TLMPM.

The kit-less cameras are explicit full-scene ideal Warp RGB-D. They never pass
native D455 or full Isaac Sim gates. No replay file is read by this runner.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np

from robotic_cutting_sim.recording import ROOT, manifest, new_output, write_json
from robotic_cutting_sim.coupled_robot import reference_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--material-device", default="cuda:0")
    parser.add_argument("--variant", choices=("D0", "D1"), default="D0")
    parser.add_argument("--task-id", choices=("CUT-01", "TRIM-01"))
    parser.add_argument("--task-variant", default="fixed")
    parser.add_argument("--workpiece-kind", choices=("rectangle", "capsule"), default="rectangle")
    parser.add_argument("--particle-spacing", type=float, help="Explicit task-only initial quadrature override in metres")
    parser.add_argument("--action-mode", choices=("cartesian_edge_pose_world", "joint_position_rad"), default="cartesian_edge_pose_world")
    parser.add_argument("--camera-layout", choices=("legacy_fixed_d455", "wrist_d405_global_d455"), default="legacy_fixed_d455")
    parser.add_argument("--include-global-camera", action="store_true")
    parser.add_argument("--tool-attachment-mode", choices=("legacy_fixed_tool", "physical_grasp"), default="legacy_fixed_tool")
    parser.add_argument("--tool-init-mode", choices=("pregrasped", "free", "pickup"), default="pregrasped")
    parser.add_argument("--grip-friction", type=float)
    parser.add_argument("--grip-force", type=float, help="Finite maximum effort per actual finger in N")
    parser.add_argument("--grip-target", type=float, help="Finger closing target in metres")
    parser.add_argument("--grip-initial-position", type=float,
        help="Explicit initial per-finger opening in metres; physical settling remains required")
    parser.add_argument("--grip-contact-profile", choices=("legacy_point_contact", "finite_pad_patch_v3"),
        default="legacy_point_contact", help="Optional finite soft-pad torsional contact model")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--velocity-feedforward", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--experiment", choices=("cut", "settle"), default="cut")
    parser.add_argument("--global-width", type=int, default=1280)
    parser.add_argument("--global-height", type=int, default=960)
    parser.add_argument("--reset-check", action="store_true")
    parser.add_argument("--record-every", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.steps <= 400 or min(args.record_every, args.global_width, args.global_height) < 1:
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
        task_id=args.task_id, task_variant=args.task_variant, action_mode=args.action_mode,
        workpiece_kind=args.workpiece_kind,
        particle_spacing_m=args.particle_spacing,
        camera_layout=args.camera_layout, include_global_camera=args.include_global_camera,
        velocity_feedforward=args.velocity_feedforward, global_width=args.global_width, global_height=args.global_height,
        experiment=args.experiment,
        tool_attachment_mode=args.tool_attachment_mode, tool_init_mode=args.tool_init_mode,
        grip_contact_profile=args.grip_contact_profile,
        static_scene=str(scene_path.resolve()))
    overrides = {name: value for name, value in {"friction": args.grip_friction,
        "max_finger_force_n": args.grip_force, "closed_target_m": args.grip_target,
        "initial_finger_position_m": args.grip_initial_position}.items() if value is not None}
    if overrides:options = replace(options, grasp=replace(options.grasp, **overrides))
    cfg = make_lab_cfg(options)
    record = manifest(asdict(options), "live_isaac_lab")
    lab_module = Path(importlib.util.find_spec("isaaclab").origin).resolve()
    lab_source_root = subprocess.check_output(
        ["git", "-C", str(lab_module.parent), "rev-parse", "--show-toplevel"], text=True).strip()
    record["runtime"] = {"source": "live_isaac_lab", "kitless": True,
        "isaac_lab_release": "3.0.0", "isaac_lab_source_sha": subprocess.check_output(
            ["git", "-C", lab_source_root, "rev-parse", "HEAD"], text=True).strip(),
        "isaac_lab_source_root": lab_source_root,
        "distribution_versions": {name: importlib.metadata.version(name) for name in
            ("isaaclab", "isaaclab-newton", "torch", "warp-lang", "newton", "usd-exchange")},
        "articulation_owner": "isaaclab.scene.InteractiveScene/NewtonManager",
        "task_api": "isaaclab.envs.DirectRLEnv", "physics_dt_s": .002, "decimation": 10,
        "material_dt_s": .0001, "camera": "ideal_warp_rgbd" if options.camera else "disabled",
        "native_d455_validated": False, "isaac_sim_kit_validated": False}
    record["acceptance_protocol_sha256"] = hashlib.sha256((ROOT/"configs/isaac_acceptance.json").read_bytes()).hexdigest()
    if args.task_id:
        from robotic_cutting_sim.tasks import make_task
        task = make_task(args.task_id, seed=args.seed, variant=args.task_variant,
            particle_spacing_m=args.particle_spacing, workpiece_kind=args.workpiece_kind)
        write_json(output/"task.json", task.to_dict())
        record["task_protocol_sha256"] = task.to_dict()["protocol_sha256"]
    implementation = output/"implementation"
    implementation.mkdir()
    loaded = [Path(__file__), ROOT/"src/robotic_cutting_sim/isaac_lab_env.py",
        ROOT/"src/robotic_cutting_sim/live_material.py", ROOT/"src/robotic_cutting_sim/tlmpm.py",
        ROOT/"src/robotic_cutting_sim/coupled_robot.py", ROOT/"src/robotic_cutting_sim/cutting_surface.py",
        ROOT/"src/robotic_cutting_sim/material_config.py", ROOT/"src/robotic_cutting_sim/scene_frames.py"]
    if options.tool_attachment_mode == "physical_grasp":
        loaded += [ROOT/"src/robotic_cutting_sim/physical_tool.py", ROOT/"src/robotic_cutting_sim/newton_tool.py",
                   ROOT/"src/robotic_cutting_sim/pad_contact.py", ROOT/"src/robotic_cutting_sim/grasp_monitor.py"]
        if options.grip_contact_profile == "finite_pad_patch_v3":loaded += [ROOT/"configs/lab_grip_contact_v3.json"]
    if args.camera_layout == "wrist_d405_global_d455":
        from robotic_cutting_sim.camera_rig import CONFIG_PATH
        loaded += [ROOT/"src/robotic_cutting_sim/camera_rig.py", CONFIG_PATH,
                   ROOT/"src/robotic_cutting_sim/camera.py", ROOT/"src/robotic_cutting_sim/usd_surface_bridge.py"]
    if args.task_id:
        loaded += [ROOT/"src/robotic_cutting_sim/tasks.py", ROOT/"src/robotic_cutting_sim/fixture_contacts.py",
                   ROOT/"configs/cutting_tasks_v1.json"]
        if args.workpiece_kind != "rectangle":loaded += [ROOT/"src/robotic_cutting_sim/workpieces.py"]
        if args.task_variant == "randomized":
            loaded += [ROOT/"src/robotic_cutting_sim/task_randomization.py", ROOT/"configs/task_randomization_v1.json",
                       ROOT/"src/robotic_cutting_sim/benchmark.py"]
    record["implementation_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in loaded}
    for path in loaded:
        shutil.copyfile(path, implementation/path.name)
    write_json(output/"manifest.json", record)
    def verify_sources(phase):
        changed = [str(path.relative_to(ROOT)) for path in loaded
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["implementation_sha256"][str(path.relative_to(ROOT))]]
        write_json(output/"source_integrity.json", {"phase": phase, "status": "FAIL" if changed else "PASS", "changed": changed})
        if changed:
            raise RuntimeError(f"Implementation changed during the recorded run: {changed}")
    (output/"rgbd").mkdir()
    writers = {}
    states, material, material_times, diagnostics, rows = [], [], [], [], []
    start = time.perf_counter()
    with launch_simulation(cfg, {"headless": True, "visualizer": [], "device": args.device}):
        write_json(output/"phase.json", {"phase": "CREATING_ACTUAL_LAB_ENV"})
        env = IsaacLabCuttingEnv(cfg, options)
        try:
            obs, info = env.reset(seed=args.seed)
            if env.physical_tool:
                write_json(output/"grasp_settling_trace.json", env.grasp_settling_trace)
                write_json(output/"grasp_initialization.json", env.grasp_initialization)
            verify_sources("AFTER_ACTUAL_RESET")
            write_json(output/"setup.json", {**env.setup_receipt, "model_bodies": env.model.body_count,
                "model_joint_coordinates": env.model.joint_coord_count,
                "cutting_hand_body_index": env.hand, "cutting_hand_body_label": env.model.body_label[env.hand],
                "knife_shape_index": env.knife_shape, "knife_shape_label": env.model.shape_label[env.knife_shape],
                "knife_shape_body_index": int(env.model.shape_body.numpy()[env.knife_shape]),
                "T_hand_edge_xyz_xyzw": env.mount,
                "body_labels": list(env.model.body_label),
                "articulations": [{"type": type(robot).__module__+"."+type(robot).__name__,
                    "joint_names": robot.joint_names, "body_names": robot.body_names} for robot in env.robots],
                "calibration": env.calibration, "observation_fields": list(obs["policy"])})
            wrist = args.camera_layout == "wrist_d405_global_d455"
            camera_ids = tuple(env.frames) if options.camera else ()
            if wrist and options.camera:
                write_json(output/"camera_rig.json", {"specs": env.camera_rig.specs,
                    "paths": env.camera_rig_paths, "hand_body_indices": env.hands,
                    "mount_collision_model": "visual geometry only; no camera mass/colliders added to robot physics",
                    "d405_model": "ideal_d405_prototype", "global_model": "ideal_rgbd_from_official_d455_asset"})
            write_json(output/"phase.json", {"phase": "LIVE_STEPPING", "initial_time_s": env.time_s})
            if options.camera:
                import imageio.v2 as imageio
                video_ids = (*camera_ids, *(name+"_depth" for name in camera_ids), "multiview") if wrist else ("global", "d455_0", "d455_1", "depth", "multiview")
                for name in video_ids:
                    writers[name] = imageio.get_writer(str(output/(name+".mp4")), fps=50,
                        codec="libx264", macro_block_size=1, quality=7)

            def capture(step):
                state = env.measured()
                states.append(state)
                world_state = env.material.world_state()
                material.append({key: world_state[key] for key in ("positions", "velocities", "codes")})
                material_times.append(env.time_s)
                if not options.camera:
                    return
                from PIL import Image, ImageDraw
                labeled = {}
                for name, frame in env.frames.items():
                    image = Image.fromarray(frame["rgb"])
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 27), fill=(0, 0, 0))
                    draw.text((6, 8), f"LIVE ISAAC LAB 3.0 | ideal Warp RGB-D | {name} | t={env.time_s:.2f}s", fill="white")
                    labeled[name] = np.asarray(image)
                    writers[name].append_data(labeled[name])
                import matplotlib
                if wrist:
                    depth_tiles = []
                    for name in camera_ids:
                        z = env.frames[name]["depth_m"]
                        near, far = env.calibration[name]["near_m"], env.calibration[name]["far_m"]
                        depth = (matplotlib.colormaps["turbo"](np.clip((z-near)/(far-near), 0, 1))[..., :3]*255).astype(np.uint8)
                        depth[z <= 0] = 0
                        image = Image.fromarray(depth)
                        ImageDraw.Draw(image).text((6, 8), f"{name} | ideal Z {near:.2f}-{far:.2f}m | t={env.time_s:.2f}s", fill="white")
                        depth = np.asarray(image)
                        writers[name+"_depth"].append_data(depth)
                        depth_tiles.append(depth)
                    writers["multiview"].append_data(np.vstack([np.hstack([labeled[name] for name in camera_ids]), np.hstack(depth_tiles)]))
                else:
                    z = env.frames["d455_0"]["depth_m"]
                    depth = (matplotlib.colormaps["turbo"](np.clip((z-.52)/.32, 0, 1))[..., :3]*255).astype(np.uint8)
                    depth[z <= 0] = 0
                    image = Image.fromarray(depth)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 640, 27), fill=(0, 0, 0))
                    draw.text((6, 8), f"LIVE ISAAC LAB | ideal optical Z | range .52-.84 m | t={env.time_s:.2f}s", fill="white")
                    depth = np.asarray(image)
                    writers["depth"].append_data(depth)
                    global_tile = np.asarray(Image.fromarray(labeled["global"]).resize((640, 480)))
                    writers["multiview"].append_data(np.vstack([np.hstack([global_tile, labeled["d455_0"]]),
                        np.hstack([labeled["d455_1"], depth])]))
                if step % args.record_every == 0 or step == args.steps:
                    data = {name+"_"+key: frame[key] for name, frame in env.frames.items()
                            for key in ("rgb", "depth_m", "valid_mask")}
                    np.savez_compressed(output/"rgbd"/f"frame_{step:05d}.npz", timestamp_s=env.time_s,
                        surface_timestamp_s=env.time_s, **data)
                    write_json(output/"rgbd"/f"calibration_{step:05d}.json", env.calibration)
                    vertices, faces = env.surface
                    np.savez_compressed(output/"rgbd"/f"surface_{step:05d}.npz", timestamp_s=env.time_s,
                        vertices=vertices, triangles=faces)
                    write_json(output/"rgbd"/f"surface_{step:05d}.json", env.surface_metadata)

            capture(0)
            initial_depth = {name: frame["depth_m"].copy() for name, frame in env.frames.items()}
            for step in range(1, args.steps+1):
                tick = time.perf_counter()
                env.step()
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
            verify_sources("AFTER_LIVE_EPISODE")
            bed = env.material.bed
            count, component_ids = mechanical_components(bed.nodes.numpy(), bed.weights.numpy(), data["codes"],
                1, gradients=bed.gradients.numpy())
            from robotic_cutting_sim.scene_frames import compose_pose, matrix_pose, invert_transform, pose_matrix
            command = np.vstack([env.reference_pose(0),
                                 [d["desired_tool_pose"] for d in diagnostics]])
            edge_from_hand = matrix_pose(invert_transform(pose_matrix(env.mount)))
            np.savez_compressed(output/"robot_state.npz", time_s=[s["timestamp_s"] for s in states],
                q=[s["q"] for s in states], qdot=[s["qdot"] for s in states],
                finger_q=[s["finger_q"] for s in states], finger_qdot=[s["finger_qdot"] for s in states],
                body_pose=[s["body_pose"] for s in states], hand_pose_xyzw=[s["hand_pose"] for s in states],
                hand_poses_world_xyzw=[s["hand_poses"] for s in states],
                tool_pose_xyzw=[s["tool_pose"] for s in states],
                desired_tool_pose_xyzw=command, commanded_hand_pose_xyzw=compose_pose(command, edge_from_hand),
                next_material_wrench_world=[d["next_material_wrench_world"] for d in diagnostics],
                applied_previous_material_wrench_world=[d["applied_previous_material_wrench_world"] for d in diagnostics],
                material_reaction_impulse_world=[d["current_material_reaction_impulse_world"] for d in diagnostics])
            if env.physical_tool:
                env.dynamic_tool.save_raw_trace(output/"tool_substeps.npz")
                np.savez_compressed(output/"physical_tool_state.npz", time_s=[s["timestamp_s"] for s in states],
                    tool_body_pose_xyzw=[s["tool_body_pose_xyzw"] for s in states],
                    tool_edge_pose_xyzw=[s["tool_pose"] for s in states],
                    tool_linear_velocity_world_m_s=[s["tool_linear_velocity_world_m_s"] for s in states],
                    tool_angular_velocity_world_rad_s=[s["tool_angular_velocity_world_rad_s"] for s in states],
                    applied_wrench_edge_world=[d["native_float32_wrench_payload_world"] for d in diagnostics],
                    contact_impulse_on_tool_com_world=[d["physical_tool"]["contact_impulse_on_tool_com_world"] for d in diagnostics],
                    applied_impulse_to_tool_com_world=[d["physical_tool"]["applied_impulse_to_tool_com_world"] for d in diagnostics])
            physics_record = dict(time_s=material_times,
                positions=[s["positions"] for s in material], velocities=[s["velocities"] for s in material],
                codes=[s["codes"] for s in material], mass_kg=env.material.masses,
                reference_positions=env.material.world_points, reference_positions_local=env.material.points,
                world_from_workpiece_xyzw=env.material.task.world_from_workpiece_xyzw if env.task else (0, 0, 0, 0, 0, 0, 1),
                reference_indices=env.material.lattice,
                final_F=data["F"], final_Fpi=data["Fpi"], final_plastic_strain=data["plastic_strain"],
                component_ids=component_ids, reference_stencil_nodes=bed.nodes.numpy(),
                reference_stencil_weights=bed.weights.numpy(), reference_stencil_gradients=bed.gradients.numpy())
            np.savez_compressed(output/"physics_state.npz", **physics_record)
            if env.task:
                vertices, faces = env.surface
                face_labels = np.full(len(faces), -1, np.int32)
                for row in env.surface_metadata["component_ranges"]:
                    face_labels[row["first_triangle"]:row["first_triangle"]+row["triangles"]] = row["mechanical_component"]
                evaluation = env.task.evaluate_oracle({**physics_record, "surface_vertices_world": vertices,
                    "surface_triangles": faces, "surface_triangle_component_ids": face_labels,
                    "surface_component_provenance": env.surface_metadata["partition_rule"],
                    "surface_timestamp_s": env.time_s})
                write_json(output/"task_evaluation.json", evaluation)
            env.sim.stage.Export(str(output/"final_live_scene.usda"))
            if wrist:
                write_json(output/"camera_pose_trace.json", env.camera_pose_trace)
            write_json(output/"diagnostics.json", diagnostics)
            write_json(output/"trace.json", rows)
            result = {"source": "live_isaac_lab", "status": "PASS_EXECUTION" if args.steps == 400 else "PASS_SMOKE",
                "actual_scene_environment_task_api": True, "steps": args.steps, "simulation_s": env.time_s,
                "runtime_s": env.sim.physics_manager.get_simulation_time()-env._runtime_time_origin,
                "native_runtime_s": env.sim.physics_manager.get_simulation_time(), "wall_s": time.perf_counter()-start,
                "final_mechanical_components": count,
                "component_masses_kg": [float(env.material.masses[component_ids == i].sum()) for i in np.unique(component_ids)],
                "native_d455_validated": False, "isaac_sim_kit_validated": False,
                "full_acceptance": "PENDING_SEPARATE_QUANTITATIVE_AUDIT"}
            result["tool_attachment_mode"], result["tool_init_mode"] = options.tool_attachment_mode, options.tool_init_mode
            if env.physical_tool:
                result.update(grasp_initialization=env.grasp_initialization,
                    grasp_status=env.grasp_monitor.last["status"], grasp_lost=env.grasp_monitor.last["grasp_lost"],
                    grip_contact_profile=options.grip_contact_profile,
                    measured_tool_pose_source="independent Newton rigid body",
                    material_wrench_receiver="independent tool COM with actual edge lever arm",
                    direct_hand_material_wrench=False,
                    bilateral_contact_fraction=float(np.mean([d["physical_tool"]["bilateral_substep_fraction"] for d in diagnostics])))
            if env.task:
                result["task_id"], result["task_success"] = env.task.task_id, evaluation["status"]
            if args.reset_check:
                env.reset(seed=args.seed)
                reset_state = env.measured()
                reset_material = env.material.world_state()
                reset = {"scope": "same-process reset to original seed; full repeat assessed separately",
                    "joint_max_abs_error_rad": float(np.max(np.abs(reset_state["q"]-states[0]["q"]))),
                    "material_max_abs_error_m": float(np.max(np.abs(reset_material["positions"]-material[0]["positions"]))),
                    "nonzero_cut_codes": int(np.count_nonzero(reset_material["codes"])),
                    "depth_max_abs_error_m": {name: float(np.max(np.abs(env.frames[name]["depth_m"]-depth)))
                        for name, depth in initial_depth.items()},
                    "depth_valid_mask_exact": {name: bool(np.array_equal(env.frames[name]["depth_m"] > 0, depth > 0))
                        for name, depth in initial_depth.items()}}
                write_json(output/"reset_check.json", reset)
                result["reset_check"] = reset
            write_json(output/"result.json", result)
            write_json(output/"phase.json", {"phase": "COMPLETE", **result})
            print(result, flush=True)
        except Exception as error:
            if env.physical_tool:
                write_json(output/"grasp_settling_trace.json", env.grasp_settling_trace)
                write_json(output/"grasp_initialization.json", getattr(env, "grasp_initialization", {}))
                if env.dynamic_tool.raw_trace:env.dynamic_tool.save_raw_trace(output/"tool_substeps.npz")
            write_json(output/"failure.json", {"status": "FAIL_LIVE_LAB", "error": str(error),
                "tool_attachment_mode": options.tool_attachment_mode, "standalone_fallback": False})
            raise
        finally:
            for writer in writers.values():
                writer.close()
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
