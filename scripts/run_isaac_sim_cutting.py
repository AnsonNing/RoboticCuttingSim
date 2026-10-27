#!/usr/bin/env python3
"""Run genuine live Isaac Sim6.2 PhysX+TLMPM+RTX; preserve native startup failures.

Run inside the dedicated Isaac Sim environment. --runtime-only tests application
initialization only and can never report a cutting or physics pass.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app, runtime_versions
from robotic_cutting_sim.recording import command, manifest, new_output, write_json


def worker(args, config):
    import numpy as np
    from PIL import Image
    out = args.output_dir
    app = None
    phase = "before_SimulationApp"
    write_json(out/"phase.json",{"phase":phase,"versions":runtime_versions()})
    try:
        app = launch_app(config,out)
        phase = "SimulationApp_initialized"
        write_json(out/"phase.json",{"phase":phase})
        if args.runtime_only:
            for _ in range(5):
                app.update()
            result = {"status":"PASS_BOOTSTRAP","scope":"Isaac Sim6.2 SimulationApp initialization only",
                      "app_updates":5,"physical_steps":0,"live_cutting":"NOT_RUN","full_isaac_integration":False}
        else:
            env = LiveIsaacSimCutting(app,config,out)
            phase = "live_reset"
            write_json(out/"phase.json",{"phase":phase})
            initial = env.reset()
            phase = "live_steps"
            write_json(out/"phase.json",{"phase":phase})
            for directory in ("observations","global_debug","camera_debug","surface_debug","oracle_debug"):
                (out/directory).mkdir()
            proprio, diagnostics, materials,body_poses = [initial], [], [],[env.body_poses()]
            frame_rows = []
            first_capture={}
            def material_state():
                return env.material.world_state() if env.task else env.material.debug_state()
            previous = {}
            changed = {name:0 for name in ("d455_0","d455_1","global")}
            def record():
                frames, sensor_info = env.capture()
                index = len(frame_rows)
                for name,frame in frames.items():
                    if index==0:
                        first_capture[name]={key:frame[key].copy() for key in ("depth_m","valid_mask")}
                    target = out/("global_debug" if name == "global" else "observations")
                    debug_keys={key for key in frame if key.startswith("rtx_reference_") or key.startswith("depth_sensor_")
                                or key in ("native_source_index_debug","native_raw_valid_mask")}
                    public = {key:value for key,value in frame.items() if key not in debug_keys}
                    np.savez_compressed(target/f"{name}_{index:04d}.npz",**public)
                    Image.fromarray(frame["rgb"]).save(target/f"{name}_{index:04d}.png")
                    reference = {key:value for key,value in frame.items() if key in debug_keys}
                    if reference:
                        np.savez_compressed(out/"camera_debug"/f"{name}_{index:04d}.npz",**reference)
                    depth = frame.get("depth_m",frame.get("depth_sensor_distance_raw")).squeeze()
                    if not np.isfinite(depth).all():
                        # Native raw nonfinite pixels are preserved, never replaced by ideal depth.
                        if config.camera_mode != "native_single_view":
                            raise FloatingPointError("Nonfinite RTX optical depth")
                    if name in previous:
                        common = np.isfinite(depth)&np.isfinite(previous[name])&(depth>0)&(previous[name]>0)
                        changed[name] += int(np.sum(common&(np.abs(depth-previous[name])>1e-5)))
                    previous[name] = depth.copy()
                state = material_state()
                materials.append({"positions":state["positions"].copy(),"velocities":state["velocities"].copy(),
                                  "codes":state["codes"].copy(),"time_s":env.time_s})
                vertices,triangles = env.surface_arrays
                np.savez_compressed(out/"surface_debug"/f"surface_{index:04d}.npz",vertices_m=vertices,
                    triangles=triangles,physics_timestamp_s=env.time_s)
                frame_rows.append({"frame":index,"time_s":env.time_s,"native_physics_time_s":env.SimulationManager.get_simulation_time(),
                    "surface":env.surface_receipt,"current_component_surface":env.surface_info,"native_sensor_info":sensor_info,
                    "native_registration":{name:dict(value) for name,value in env.native_registration.items()}})
                write_json(out/"frame_manifest.json",frame_rows)
            record()
            initial_surface_vertices=env.surface_arrays[0].copy()
            initial_surface_triangles=env.surface_arrays[1].copy()
            initial_all_dofs=env.arms.get_dof_positions().numpy().copy()
            start = time.perf_counter()
            for step in range(config.steps):
                actual, diagnostic = env.step()
                proprio.append(actual)
                body_poses.append(env.body_poses())
                diagnostics.append(diagnostic)
                if (step+1)%config.record_every == 0 or step == config.steps-1:
                    record()
                if (step+1)%25 == 0:
                    print(f"LIVE Kit step={step+1}/{config.steps} t={env.time_s:.2f}s "
                          f"components={env.surface_info['mechanical_components']}",flush=True)
            np.savez_compressed(out/"robot_state.npz",time_s=[p["timestamp_s"] for p in proprio],
                q=[p["q"] for p in proprio],qdot=[p["qdot"] for p in proprio],
                hand_pose_xyzw=[p["hand_pose"] for p in proprio],tool_pose_xyzw=[p["tool_pose"] for p in proprio],
                body_pose=body_poses,body_prim_paths=env.rigid_link_paths,
                commanded_tool_pose_xyzw=[env.reference_pose(p["timestamp_s"]) for p in proprio],
                desired_tool_pose_xyzw=[env.reference_pose(p["timestamp_s"]) for p in proprio])
            final=material_state()
            sample_indices=np.rint(np.asarray([s["time_s"] for s in materials])/config.control_dt).astype(int)
            _,component_ids,_=env.surface.surface.partition.partition(final["codes"])
            np.savez_compressed(out/"oracle_debug/physics_state.npz",time_s=[s["time_s"] for s in materials],
                positions=[s["positions"] for s in materials],velocities=[s["velocities"] for s in materials],
                codes=[s["codes"] for s in materials],mass_kg=env.material.masses,
                tool_pose_xyzw=np.asarray([p["tool_pose"] for p in proprio])[sample_indices],
                reference_positions=env.material.world_points if env.task else env.material.points,
                reference_positions_local=env.material.points,reference_indices=env.material.lattice,
                world_from_workpiece_xyzw=env.task.world_from_workpiece_xyzw if env.task else env.material.target.world_from_workpiece_xyzw,
                reference_stencil_nodes=env.material.bed.nodes.numpy(),
                reference_stencil_weights=env.material.bed.weights.numpy(),
                reference_stencil_gradients=env.material.bed.gradients.numpy(),component_ids=component_ids,
                component_provenance="final conservative actual reference-stencil graph including stress gradients; debug only",
                final_F=final["F"],final_Fpi=final["Fpi"],final_cut_progress=final["cut_progress"],
                final_plastic_strain=final["plastic_strain"])
            write_json(out/"oracle_debug/diagnostics.json",diagnostics)
            task_evaluation=None
            if env.task:
                payload=dict(np.load(out/"oracle_debug/physics_state.npz"))
                vertices,triangles=env.surface_arrays
                face_component=np.full(len(triangles),-1,np.int32)
                for row in env.surface_info["component_ranges"]:
                    first=row["first_triangle"]
                    face_component[first:first+row["triangles"]]=row["mechanical_component"]
                if np.any(face_component<0):
                    raise AssertionError("Current mechanical surface ranges do not cover every triangle")
                payload.update(surface_vertices_world=vertices,surface_triangles=triangles,
                    surface_triangle_component_ids=face_component,surface_component_provenance=json.dumps(env.surface_info),
                    surface_timestamp_s=env.time_s)
                task_evaluation=env.task.evaluate_oracle(payload)
                write_json(out/"oracle_debug/task_evaluation.json",task_evaluation)
            qmotion = float(max(np.max(np.abs(state["q"]-initial["q"])) for state in proprio))
            final_q_displacement=float(np.max(np.abs(proprio[-1]["q"]-initial["q"])))
            first_material=materials[0]["positions"].copy()
            again=env.reset()
            reset_material=material_state()
            reset_error=float(np.abs(again["q"]-initial["q"]).max())
            reset_material_error=float(np.abs(reset_material["positions"]-first_material).max())
            reset_codes=int(np.count_nonzero(reset_material["codes"]))
            reset_body=env.body_poses()
            reset_all_dofs=env.arms.get_dof_positions().numpy().copy()
            reset_vertices,reset_triangles=env.surface_arrays
            reset_surface_equal=(np.array_equal(initial_surface_vertices,reset_vertices)
                                 and np.array_equal(initial_surface_triangles,reset_triangles))
            np.savez_compressed(out/"oracle_debug/reset_geometry.npz",initial_body_pose=body_poses[0],
                reset_body_pose=reset_body,initial_all_dofs=initial_all_dofs,reset_all_dofs=reset_all_dofs,
                initial_surface_vertices=initial_surface_vertices,initial_surface_triangles=initial_surface_triangles,
                reset_surface_vertices=reset_vertices,reset_surface_triangles=reset_triangles)
            reset_frames,reset_sensor_info=env.capture()
            (out/"reset_observations").mkdir()
            camera_reset={}
            for name,frame in reset_frames.items():
                np.savez_compressed(out/"reset_observations"/f"{name}.npz",rgb=frame["rgb"],depth_m=frame["depth_m"],
                    valid_mask=frame["valid_mask"],physics_timestamp_s=frame["physics_timestamp_s"])
                camera_reset[name]={"max_depth_error_m":float(np.abs(first_capture[name]["depth_m"]-frame["depth_m"]).max()),
                    "valid_mask_changed_pixels":int(np.sum(first_capture[name]["valid_mask"]!=frame["valid_mask"]))}
            write_json(out/"reset_receipt.json",{"q_max_error_rad":reset_error,"material_max_error_m":reset_material_error,
                "nonzero_cut_codes":reset_codes,"time_s":env.time_s,"dynamics_reset_measured":True,
                "all_nine_dofs_max_error_rad":float(np.abs(initial_all_dofs-reset_all_dofs).max()),
                "all_body_pose_max_abs_error":float(np.abs(np.asarray(body_poses[0])-reset_body).max()),
                "surface_arrays_bitwise_equal":reset_surface_equal,
                "camera_reset_repeatability":camera_reset,"sensor_info":reset_sensor_info})
            result = {"status":("PASS_EXECUTION" if config.steps>=400 else "PASS_SMOKE") if qmotion>1e-5 else "FAIL",
                "scope":"LIVE Isaac Sim6.2 native PhysX articulation + retained TLMPM + native RTX",
                "steps":config.steps,"simulation_s":config.steps*config.control_dt,"wall_s":time.perf_counter()-start,
                "native_physics_steps":diagnostics[-1]["relative_native_physics_steps"],
                "robot_dynamics_source":"Isaac PhysX","standalone_robot_dynamics_steps":0,"replay_inputs":False,
                "camera_mode":config.camera_mode,"camera_frames":len(frame_rows),"changed_depth_pixels_10um":changed,
                "task_id":config.task_id,"task_goal_status":task_evaluation["status"] if task_evaluation else "NOT_RUN",
                "joint_motion_max_rad":qmotion,"final_mechanical_components":frame_rows[-1]["current_component_surface"]["mechanical_components"],
                "final_joint_displacement_rad":final_q_displacement,
                "full_isaac_integration":False,"physical_cutting":"NOT_SCORED","surface_flattening":"NOT_SCORED",
                "native_single_view_calibration":"NOT_VALIDATED","reset_repeatability":"MEASURED_DYNAMICS_MATERIAL_CAMERAS",
                "native_single_view_registration":"VALIDATED_EXACT_PROFILE_APPLIED" if config.camera_mode=="native_single_view" else "NOT_RUN",
                "observation_force_oracle_global":False,
                "limitations":["Native PhysX trajectory differs from standalone MuJoCo and requires comparison",
                    "20ms explicit reaction delay; no monolithic/converged energy claim",
                    "Native raw SingleView depth is never replaced by RTX reference depth",
                    "PASS_EXECUTION requires later independent task/camera acceptance"]}
        write_json(out/"worker_result.json",result)
        return 0 if result["status"].startswith("PASS") else 1
    except Exception as error:
        (out/"failure.log").write_text(traceback.format_exc())
        write_json(out/"worker_result.json",{"status":"BLOCKED" if phase == "before_SimulationApp" else "FAIL",
            "phase":phase,"error":str(error),"full_isaac_integration":False,"standalone_fallback":False})
        raise
    finally:
        if app is not None:
            app.close(wait_for_replicator=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--steps",type=int,default=400)
    p.add_argument("--record-every",type=int,default=10)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--device",default="cuda:1")
    p.add_argument("--camera-mode",choices=["rtx_rgbd","native_single_view"],default="rtx_rgbd")
    p.add_argument("--scene",type=Path)
    p.add_argument("--runtime-only",action="store_true")
    p.add_argument("--no-feedback",action="store_true")
    p.add_argument("--velocity-feedforward",action=argparse.BooleanOptionalAction,default=None,
                   help="Set native drive velocity targets from successive IK commands (new tasks default on)")
    p.add_argument("--task-id",choices=["CUT-01","TRIM-01"])
    p.add_argument("--task-variant",default="fixed")
    p.add_argument("--action-mode",choices=["cartesian","joint"],default="cartesian")
    p.add_argument("--native-noise",choices=["off","on"],default=None,
                   help="Native sensor control recipe; new tasks default off for geometry checks")
    p.add_argument("--surface-profile",choices=["legacy","current_volume_v1"],default=None)
    p.add_argument("--particle-spacing",type=float,default=None,help="Explicit whole-initial-task sampling override in meters")
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    config=IsaacSimCuttingConfig(seed=args.seed,steps=args.steps,record_every=args.record_every,gpu=args.gpu,
                device=args.device,camera_mode=args.camera_mode,feedback=not args.no_feedback,
                velocity_feedforward=args.velocity_feedforward,
                task_id=args.task_id,task_variant=args.task_variant,action_mode=args.action_mode,native_noise=args.native_noise,
                surface_profile=args.surface_profile,
                particle_spacing_m=args.particle_spacing,
                scene=str(args.scene.resolve()) if args.scene else None)
    if args.worker:
        return worker(args,config)
    out=new_output(args.output_dir)
    files=[ROOT/"src/robotic_cutting_sim"/name for name in ("isaac_sim_runner.py","live_material.py","robot.py",
           "tlmpm.py","material_config.py","cut_topology.py","cutting_surface.py","usd_surface_bridge.py","single_view_registration.py")]
    files += [Path(__file__).resolve(),ROOT/"configs/isaac_sim_cutting.kit"]
    if args.task_id:
        files += [ROOT/"src/robotic_cutting_sim"/name for name in ("tasks.py","geometry.py","scene_frames.py","scene_validation.py",
                  "benchmark.py","workcell.py","coupled_robot.py","fixture_contacts.py","affine_surface_controls.py")]
        files += [ROOT/"configs/cutting_tasks_v1.json",ROOT/"configs/particle_surface_affine_v1.json"]
    (out/"implementation").mkdir()
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for f in files:
        shutil.copyfile(f,out/"implementation"/f.name)
    write_json(out/"manifest.json",{**manifest(asdict(config),"LIVE_IsaacSim6.2_cutting"),"versions":runtime_versions(),
        "implementation_sha256":hashes,"nvidia_smi":command(["nvidia-smi","--query-gpu=name,driver_version","--format=csv"]),
        "runtime_only":args.runtime_only})
    argv=[sys.executable,str(Path(__file__).resolve()),"--output-dir",str(out.resolve()),
          "--steps",str(args.steps),"--record-every",str(args.record_every),"--seed",str(args.seed),
          "--gpu",str(args.gpu),"--device",args.device,"--camera-mode",args.camera_mode,"--worker"]
    argv += ["--action-mode",args.action_mode,"--native-noise",config.native_noise,"--surface-profile",config.surface_profile]
    if args.task_id:
        argv += ["--task-id",args.task_id,"--task-variant",args.task_variant]
    if args.particle_spacing is not None:
        argv += ["--particle-spacing",str(args.particle_spacing)]
    if args.scene:
        argv.extend(["--scene",str(args.scene.resolve())])
    if args.runtime_only:
        argv.append("--runtime-only")
    if args.no_feedback:
        argv.append("--no-feedback")
    argv.append("--velocity-feedforward" if config.velocity_feedforward else "--no-velocity-feedforward")
    env=os.environ.copy();env["OMNI_KIT_ACCEPT_EULA"]="YES";env["PYTHONUNBUFFERED"]="1"
    start=time.perf_counter()
    with (out/"stdout.log").open("w") as log:
        process=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt={"command":argv,"returncode":process.returncode,"wall_s":time.perf_counter()-start}
    write_json(out/"process_receipt.json",receipt)
    if (out/"worker_result.json").exists():
        result=json.loads((out/"worker_result.json").read_text())
    else:
        result={"status":"BLOCKED","reason":"Native runtime exited without a completion receipt",
                "full_isaac_integration":False,"standalone_fallback":False}
    if process.returncode != 0 and result["status"].startswith("PASS"):
        result={**result,"status":"FAIL","reason":"Runtime process failed after writing its worker result"}
    result["process"]=receipt
    write_json(out/"result.json",result)
    print(json.dumps(result,indent=2))
    return 2 if result["status"] == "BLOCKED" else 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
