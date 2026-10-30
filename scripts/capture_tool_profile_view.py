#!/usr/bin/env python3
"""Presentation-only RGB view of a physically pregrasped profile tool (WP-11B).

The tool is an independent PhysX rigid body squeezed by the two actual finger
drives (no joint, constraint or pose write). After the normal pregrasp settle
and a short hold of unchanged arm targets, one world-fixed RGB camera aimed at
the measured tool renders a frame. Rendering must not advance physics. The
image is a visual comparison aid only, never an observation or evidence of
cutting.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
PROTOCOL = ROOT/"configs/physical_tool_controls_v7_ruler_v2_native.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-profile", required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--task-id", default="CUT-01-R")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hold-s", type=float, default=1.)
    parser.add_argument("--distance-m", type=float, default=.50)
    parser.add_argument("--eye-direction", type=float, nargs=3, default=(.55, -.70, .45))
    parser.add_argument("--hfov-deg", type=float, default=34.)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rt-subframes", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    if args.tool_profile not in protocol.get("allowed_tool_profiles", ()):
        parser.error("--tool-profile must be one of the protocol's allowed_tool_profiles")
    result = {"status": "RUNNING", "tool_profile": args.tool_profile, "task_id": args.task_id, "seed": args.seed,
              "gpu": args.gpu, "protocol": str(args.protocol), "protocol_sha256": sha256(args.protocol),
              "argv": sys.argv, "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
                  Path(__file__), ROOT/"src/robotic_cutting_sim/tool_profile.py",
                  ROOT/"src/robotic_cutting_sim/isaac_sim_runner.py", ROOT/"src/robotic_cutting_sim/native_tool.py")},
              "scope": "Presentation-only RGB of a physically held tool; never an observation or cutting evidence"}
    write_json(output/"result.json", result)
    start = time.monotonic()
    env = None
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app, numpy_data
        from robotic_cutting_sim.physical_tool import ToolGraspConfig
        grip = replace(ToolGraspConfig(), **protocol["normal_grip"])
        config = IsaacSimCuttingConfig(task_id=args.task_id, seed=args.seed, task_variant="fixed", gpu=args.gpu,
            device=f"cuda:{args.gpu}", particle_spacing_m=.001, camera=False, camera_layout="wrist_d405_global_d455",
            camera_mode="native_single_view", native_noise="off", tool_attachment_mode="physical_grasp",
            tool_init_mode="pregrasped", grasp=grip, tool_profile_id=args.tool_profile,
            tool_profile_hand_offset_world_m=tuple(protocol["tool_profile_retarget"]["hand_offset_world_m"]))
        write_json(output/"runtime_config.json", asdict(config))
        app = launch_app(config, output)
        env = LiveIsaacSimCutting(app, config, output)
        initial = env.reset(seed=args.seed)
        result["pregrasp"] = env.grasp_reset_receipt
        result["tool_spec"] = env.tool_spec.to_dict()
        env.set_gripper("hold")
        q = np.asarray(initial["q"])[0].copy()
        from robotic_cutting_sim.scene_frames import pose_matrix
        relative0 = np.linalg.inv(pose_matrix(initial["hand_pose"]))@pose_matrix(initial["tool_body_pose_xyzw"])
        drift = []
        for _ in range(round(args.hold_s/config.control_dt)):
            state, _ = env.rigid_probe_step(q)
            relative = np.linalg.inv(pose_matrix(state["hand_pose"]))@pose_matrix(state["tool_body_pose_xyzw"])
            error = np.linalg.inv(relative0)@relative
            from scipy.spatial.transform import Rotation
            drift.append((float(np.linalg.norm(error[:3, 3])), float(Rotation.from_matrix(error[:3, :3]).magnitude())))
        state = env.proprioception()
        drift = np.asarray(drift)
        result["hold"] = {"duration_s": args.hold_s, "max_relative_translation_m": float(drift[:, 0].max()),
                          "max_relative_rotation_rad": float(drift[:, 1].max()),
                          "final_finger_q_m": np.asarray(state["finger_q"]).tolist(),
                          "contacts_last": {k: env.grasp_trace[-1]["contacts"].get(k) for k in
                                            ("bilateral_loaded_handle_contact", "raw_contact_count")},
                          "hand_pose_xyzw": state["hand_pose"], "tool_body_pose_xyzw": state["tool_body_pose_xyzw"],
                          "measured_edge_pose_xyzw": state["tool_pose"]}
        # Camera aimed between the fingers and the blade tip of the MEASURED tool.
        from robotic_cutting_sim.presentation_camera import _look_at_cv, author_overview_camera, intrinsics
        from robotic_cutting_sim.scene_frames import transform_points
        T_tool = pose_matrix(state["tool_body_pose_xyzw"])
        points = np.vstack([transform_points(T_tool, shape.vertices_tool_m) for shape in env.tool_spec.shapes])
        target = (points.min(0)+points.max(0))/2
        direction = np.asarray(args.eye_direction, float)/np.linalg.norm(args.eye_direction)
        eye = target+args.distance_m*direction
        spec = {"prim_path": "/World/PresentationCamera/KnifeView", "T_world_camera": _look_at_cv(eye, target).tolist(),
                "hfov_deg": args.hfov_deg, "width": args.width, "height": args.height,
                "K": intrinsics(args.width, args.height, args.hfov_deg).tolist(), "eye_world_m": eye.tolist(),
                "look_at_target_world_m": target.tolist(), "role": "presentation_only", "observation": False}
        author_overview_camera(env.stage, spec)
        import omni.replicator.core as rep
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        sensor = CameraSensor(RtxCamera(spec["prim_path"], tick_rate=0.), resolution=(args.height, args.width),
                              annotators=["rgb"])
        manager = env.SimulationManager
        before = (manager.get_simulation_time(), manager.get_num_physics_steps())
        rgb = None
        for index in range(40):
            rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.)
            data = sensor.get_data("rgb")[0]
            if data is not None and index >= 4:
                rgb = numpy_data(data)
                break
        after = (manager.get_simulation_time(), manager.get_num_physics_steps())
        if rgb is None:
            raise TimeoutError("Presentation camera produced no RGB array")
        if before != after:
            raise RuntimeError(f"Rendering advanced physics: {before}->{after}")
        from PIL import Image
        image = Image.fromarray(np.asarray(rgb)[:, :, :3].astype(np.uint8))
        image.save(output/"knife_view.png")
        result.update(camera=spec, rendering_physics_unchanged=True, image="knife_view.png",
                      image_sha256=sha256(output/"knife_view.png"), status="CAPTURED")
    except BaseException as error:
        result.update(status="FAIL_CAPTURE", error=str(error))
        (output/"failure.txt").write_text(traceback.format_exc())
    finally:
        result["wall_s"] = time.monotonic()-start
        write_json(output/"result.json", result)
        if env is not None:
            env.close()
    print(result["status"], output/"result.json", flush=True)
    return 0 if result["status"] == "CAPTURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
