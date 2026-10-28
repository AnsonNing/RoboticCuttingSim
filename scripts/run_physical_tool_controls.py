#!/usr/bin/env python3
"""Run frozen rigid-only physical-tool controls in the existing live backends.

No tool pose is written. External load is not relabelled material feedback.
CPU helpers below support engineering classification; independent raw-data
auditing remains a separate script and complete physics acceptance is false.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
PROTOCOL = ROOT/"configs/physical_tool_controls_v1.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False)+"\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def smooth_joint_targets(start, target, transfer, dt):
    """Quintic actuator targets with bounded increments and continuous speed."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    if start.shape != (7,) or target.shape != (7,) or not np.isfinite(np.r_[start, target]).all():
        raise ValueError("Expected two finite measured/IK arm joint vectors")
    distance = float(np.max(np.abs(target-start)))
    duration = max(transfer["minimum_waypoint_duration_s"],
        1.875*distance/transfer["maximum_joint_speed_rad_s"],
        1.875*distance*dt/transfer["maximum_joint_increment_rad"])
    steps = max(1, int(math.ceil(duration/dt)))
    u = np.arange(1, steps+1)/steps
    a = 10*u**3-15*u**4+6*u**5
    targets = start+a[:, None]*(target-start)
    increments = np.diff(np.vstack([start, targets]), axis=0)
    if np.max(np.abs(increments)) > transfer["maximum_joint_increment_rad"]+1e-10:
        raise ValueError("Quintic targets exceed the frozen joint-increment bound")
    return targets


def relative_poses(hand, tool):
    hand, tool = np.asarray(hand, float), np.asarray(tool, float)
    h = Rotation.from_quat(hand[:, 3:])
    return np.c_[h.inv().apply(tool[:, :3]-hand[:, :3]),
                  (h.inv()*Rotation.from_quat(tool[:, 3:])).as_quat()]


def pose_error(reference, poses):
    reference, poses = np.asarray(reference, float), np.asarray(poses, float)
    return np.linalg.norm(poses[:, :3]-reference[:3], axis=1), (
        Rotation.from_quat(poses[:, 3:])*Rotation.from_quat(reference[3:]).inv()).magnitude()


def native_samples(path, start, end, spec, minimum_load, geometry_profile=None, rigid_dt_s=.002):
    """Derive bilateral evidence from actual decoded raw tool/finger pairs."""
    half = np.asarray(spec["handle_half_extents_m"], float)
    samples = []
    for line in Path(path).open():
        row = json.loads(line); t = row["native_time_s"]
        if not start+1e-7 < t <= end+1e-7: continue
        sensor = row["contacts"]
        tool = np.asarray(row["tool_body_pose_xyzw"], float)
        # Native contact reports describe collision detection before the
        # integrated POST-step actor pose. Use only an explicitly recorded,
        # cross-checked PRE-step pose; old recordings retain their old profile.
        frame = sensor.get("contact_geometry_frame", "LEGACY_POST_STEP_TOOL_POSE")
        if geometry_profile is not None and frame != geometry_profile:
            raise ValueError("Recorded Native contact geometry does not match the required frozen profile")
        if frame == "PRE_STEP_MEASURED_TOOL_POSE":
            geometry = np.asarray(sensor.get("contact_geometry_tool_pose_xyzw"), float)
            before = np.asarray(row.get("tool_body_pose_before_step_xyzw"), float)
            if (geometry.shape != (7,) or before.shape != (7,) or not np.isfinite(geometry).all()
                    or not np.array_equal(geometry, before)
                    or abs(float(sensor.get("contact_geometry_time_s", float("inf")))+rigid_dt_s-t) > 1e-7):
                raise ValueError("PRE-step Native contact geometry lacks matching actual pose/time evidence")
        elif frame == "LEGACY_POST_STEP_TOOL_POSE":
            geometry = tool
        else: raise ValueError("Unknown recorded Native contact geometry frame: "+str(frame))
        rotation = Rotation.from_quat(geometry[3:])
        loads = {"left_finger": 0., "right_finger": 0.}
        unexpected = 0; outside_handle = 0; stale_finger = 0; outside_extent = 0.
        for c in sensor["contacts"]:
            bodies = [c["body0"], c["body1"]]
            matches = [b == "/World/Tools/CuttingTool_0" or b.startswith("/World/Tools/CuttingTool_0/") for b in bodies]
            if sum(matches) != 1: continue
            other = bodies[1 if matches[0] else 0]
            role = "left_finger" if other.endswith("/panda_leftfinger") else "right_finger" if other.endswith("/panda_rightfinger") else "other"
            force = float(np.linalg.norm(c["raw_impulse_world_ns"])/c["dt_s"])
            local = rotation.inv().apply(np.asarray(c["position_world_m"])-geometry[:3])
            handle = bool(np.all(np.abs(local) <= half+1e-6))
            current = abs(t-c["time_s"]) <= max(1e-7, 4*np.finfo(np.float32).eps*max(1., abs(t)))
            if role in loads:
                if handle and current: loads[role] += force
                # A failed handle-region or timestamp test is not evidence of
                # a different supporting body. Retain it separately and never
                # include that point in the loaded-handle fraction.
                if force > minimum_load:
                    if not handle:
                        outside_handle += 1
                        outside_extent = max(outside_extent, float(np.max(np.abs(local)-half)))
                    if not current: stale_finger += 1
            elif force > minimum_load: unexpected += 1
        tensor = sensor.get("tensor_forces", {})
        points = tensor.get("records", [])
        penetration = max([max(0., -c["separation_m"]) for c in points if c.get("separation_m") is not None] or [0.])
        samples.append({"time_s": t, "hand": row["hand_pose_xyzw"], "tool": tool,
            "linear": row["tool_linear_velocity_world_m_s"], "angular": row["tool_angular_velocity_world_rad_s"],
            "left_load_n": loads["left_finger"], "right_load_n": loads["right_finger"],
            "valid": bool(sensor["sensor_valid"] and sensor["sensor_fresh"] and sensor["new_physics_step"]
                and tensor.get("status") == "VALID_NATIVE_NORMAL_AND_FRICTION_DATA"),
            "unexpected": unexpected, "loaded_finger_points_outside_handle": outside_handle,
            "contact_geometry_frame": frame,
            "loaded_finger_points_stale": stale_finger, "maximum_loaded_finger_point_outside_handle_m": outside_extent,
            "contact_count": sensor["raw_contact_count"], "penetration_m": penetration,
            "contact_force": tensor.get("total_filtered_force_world_on_tool_n", [0., 0., 0.])})
    return samples


def lab_samples(path, start, end, spec, minimum_load):
    """Read the actual MJWarp per-substep ledger, with explicit shape ordering."""
    metadata = read_json(Path(path).with_suffix(".json"))
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    shape_body = np.asarray(metadata["shape_body"])
    tool_body = metadata["tool_body_index"]
    fingers = [metadata["left_finger_body_index"], metadata["right_finger_body_index"]]
    handles = set(metadata["handle_shape_indices"])
    samples = []
    for j, t in enumerate(data["rigid_time_s"]):
        if not start+1e-7 < t <= end+1e-7: continue
        loads = [0., 0.]; unexpected = 0; force_sum = np.zeros(3); pen = 0.; count = 0
        for k in range(data["contact_offsets"][j], data["contact_offsets"][j+1]):
            a, b = int(data["shape0"][k]), int(data["shape1"][k])
            if min(a, b) < 0 or tool_body not in (shape_body[a], shape_body[b]): continue
            on0 = shape_body[a] == tool_body
            tool_shape, other = (a, shape_body[b]) if on0 else (b, shape_body[a])
            f = data["contact_wrench_on_shape0_world"][k, :3]*(1 if on0 else -1)
            pressure = float(abs(np.dot(f, data["normal_world"][k])))
            if other in fingers and tool_shape in handles:
                loads[fingers.index(other)] += pressure
            elif np.linalg.norm(f) > minimum_load: unexpected += 1
            force_sum += f; count += 1
            pen = max(pen, -float(data["distance_m"][k]))
        samples.append({"time_s": float(t), "hand": data["hand_pose_xyzw"][j], "tool": data["tool_body_pose_xyzw"][j],
            "linear": data["tool_velocity_com_world"][j, :3], "angular": data["tool_velocity_com_world"][j, 3:],
            "left_load_n": loads[0], "right_load_n": loads[1], "valid": True,
            "unexpected": unexpected, "contact_count": count, "penetration_m": max(0., pen), "contact_force": force_sum})
    return samples


def measure_case(samples, control_rows, initial_state, spec, protocol, duration):
    """Measured relative motion and unsupported COM gravity; no body correction."""
    if len(samples) < 2: return {"evidence_available": False, "reason": "Missing actual rigid-substep ledger"}
    times = np.array([r["time_s"] for r in samples]); hand = np.array([r["hand"] for r in samples]); tool = np.array([r["tool"] for r in samples])
    limits = protocol["measurement"]; dt = protocol["rigid_dt_s"]
    initial = relative_poses(np.asarray(initial_state["hand_pose"])[None], np.asarray(initial_state["tool_body_pose_xyzw"])[None])[0]
    relative = relative_poses(hand, tool); translation, rotation = pose_error(initial, relative)
    velocities = np.diff(relative[:, :3], axis=0)/np.diff(times)[:, None]
    angular = (Rotation.from_quat(relative[1:, 3:])*Rotation.from_quat(relative[:-1, 3:]).inv()).magnitude()/np.diff(times)
    window = times >= times[-1]-limits["stable_window_s"]-1e-7
    finite = all(np.isfinite(np.asarray([r[k] for r in samples], float)).all() for k in ("hand", "tool", "linear", "angular", "contact_force"))
    finite &= np.isfinite(times).all() and all(np.isfinite(r["left_load_n"]+r["right_load_n"]+r["penetration_m"]) for r in samples)
    bilateral = np.array([r["left_load_n"] > limits["minimum_loaded_finger_force_n"] and r["right_load_n"] > limits["minimum_loaded_finger_force_n"] for r in samples])
    q0 = np.asarray(initial_state["q"], float)[0]
    q = np.asarray([r["state"]["q"][0] for r in control_rows if r["phase"] == "perturbation"], float)
    com = tool[:, :3]+Rotation.from_quat(tool[:, 3:]).apply(np.asarray(spec["com_tool_m"]))
    free = np.array([r["contact_count"] == 0 for r in samples])
    blocks=[]; block=[]
    for index, is_free in enumerate(free):
        if is_free: block.append(index)
        elif block: blocks.append(block);block=[]
    if block:blocks.append(block)
    longest = max(blocks, key=len) if blocks else []
    free_duration = float(times[longest[-1]]-times[longest[0]]) if len(longest)>1 else 0.
    acceleration = None
    if free_duration >= limits["freeflight_minimum_duration_s"]:
        acceleration = float(2*np.polyfit(times[longest]-times[longest[0]], com[longest, 2], 2)[0])
    qdelta = q-q0 if len(q) else np.empty((0, 7))
    return {"evidence_available": True, "finite": bool(finite and np.isfinite(q).all()),
        "arm_samples_complete": bool(len(q) == round(duration/protocol["control_dt_s"])),
        "rigid_samples": len(samples), "raw_sampling_complete": bool(len(samples) == round(duration/dt) and np.all(np.diff(times)>0) and np.max(np.diff(times)) <= dt*1.01),
        "all_contact_samples_valid": bool(all(r["valid"] for r in samples)),
        "contact_geometry_frames": {name:sum(r.get("contact_geometry_frame", "ACTUAL_CONTACT_SHAPE_IDS") == name for r in samples)
            for name in sorted({r.get("contact_geometry_frame", "ACTUAL_CONTACT_SHAPE_IDS") for r in samples})},
        "max_relative_translation_m": float(translation.max()), "max_relative_rotation_rad": float(rotation.max()),
        "maximum_downward_tool_displacement_m": float(max(0., np.asarray(initial_state["tool_body_pose_xyzw"])[2]-tool[:, 2].min())),
        "maximum_relative_linear_speed_last_window_m_s": float(np.linalg.norm(velocities[window[1:]], axis=1).max()),
        "maximum_relative_angular_speed_last_window_rad_s": float(angular[window[1:]].max()),
        "bilateral_loaded_handle_fraction": float(bilateral.mean()),
        "maximum_contact_penetration_m": float(max(r["penetration_m"] for r in samples)),
        "unintended_loaded_contact_steps": sum(r["unexpected"] > 0 for r in samples),
        "loaded_finger_outside_handle_steps": sum(r.get("loaded_finger_points_outside_handle", 0) > 0 for r in samples),
        "loaded_finger_outside_handle_points": sum(r.get("loaded_finger_points_outside_handle", 0) for r in samples),
        "loaded_finger_stale_points": sum(r.get("loaded_finger_points_stale", 0) for r in samples),
        "maximum_loaded_finger_point_outside_handle_m": max(r.get("maximum_loaded_finger_point_outside_handle_m", 0.) for r in samples),
        "mean_contact_force_last_window_world_n": np.mean([r["contact_force"] for r, selected in zip(samples, window) if selected],axis=0).tolist(),
        "maximum_arm_deflection_rad": float(np.max(np.abs(qdelta))) if len(qdelta) else None,
        "final_arm_deflection_rad": qdelta[-1].tolist() if len(qdelta) else None,
        "freeflight_longest_duration_s": free_duration, "freeflight_vertical_acceleration_m_s2": acceleration}


def classify_case(name, metrics, protocol):
    limits=protocol["measurement"]
    if not metrics.get("evidence_available"):
        return {"status":"NOT_RUN_MISSING_RIGID_EVIDENCE", "checks": {}}
    checks={k:metrics[k] is True for k in ("finite","raw_sampling_complete","all_contact_samples_valid","arm_samples_complete")}
    stable={"relative_translation":metrics["max_relative_translation_m"] <= limits["maximum_relative_translation_m"],
        "relative_rotation":metrics["max_relative_rotation_rad"] <= limits["maximum_relative_rotation_rad"],
        "relative_linear_speed":metrics["maximum_relative_linear_speed_last_window_m_s"] <= limits["maximum_relative_linear_speed_m_s"],
        "relative_angular_speed":metrics["maximum_relative_angular_speed_last_window_rad_s"] <= limits["maximum_relative_angular_speed_rad_s"],
        "loaded_handle_contacts":metrics["bilateral_loaded_handle_fraction"] >= limits["minimum_bilateral_loaded_handle_fraction"],
        "bounded_penetration":metrics["maximum_contact_penetration_m"] <= limits["maximum_contact_penetration_m"],
        "no_unintended_support":metrics["unintended_loaded_contact_steps"] == 0}
    if name in ("baseline_hold","nominal_load"): checks.update(stable)
    if name == "release":
        acceleration=metrics["freeflight_vertical_acceleration_m_s2"]
        checks.update(independent_relative_motion=metrics["max_relative_translation_m"] > limits["release_minimum_relative_translation_m"],
            actual_downward_motion=metrics["maximum_downward_tool_displacement_m"] > limits["release_minimum_relative_translation_m"],
            unsupported_gravity_interval=acceleration is not None and abs(acceleration+9.81)/9.81 <= limits["freeflight_gravity_relative_error_max"])
    return {"status":"PASS_CASE_ENGINEERING_SCREENS" if all(checks.values()) else "FAIL_CASE_ENGINEERING_SCREENS",
        "checks":{k:bool(v) for k,v in checks.items()}, "stable_hold_checks":{k:bool(v) for k,v in stable.items()},
        "matched_response_comparison":"PENDING_MATRIX_COMPARISON" if name not in ("baseline_hold","release") else "NOT_REQUIRED",
        "full_physics_acceptance":False}


def compare_cases(reference, perturbed, protocol):
    """Reject unmatched starts, then compare measured strong/weak responses."""
    limits=protocol["measurement"]
    a,b=reference["initial_state"],perturbed["initial_state"]
    distance,angle=pose_error(a["tool_body_pose_xyzw"],np.asarray(b["tool_body_pose_xyzw"])[None])
    matched={"same_runtime":reference["runtime"] == perturbed["runtime"],
        "same_source_bytes":reference["source_sha256"] == perturbed["source_sha256"],
        "same_external_edge_load": reference["case_config"]["edge_wrench_world"] == perturbed["case_config"]["edge_wrench_world"],
        "initial_tool_translation":distance[0] <= limits["maximum_matched_start_tool_translation_m"],
        "initial_tool_rotation":angle[0] <= limits["maximum_matched_start_tool_rotation_rad"],
        "initial_arm_joints":np.max(np.abs(np.asarray(a["q"])[0]-np.asarray(b["q"])[0])) <= limits["maximum_matched_start_arm_joint_error_rad"]}
    ma,mb=reference["metrics"],perturbed["metrics"]
    translation=mb["max_relative_translation_m"]-ma["max_relative_translation_m"]
    rotation=mb["max_relative_rotation_rad"]-ma["max_relative_rotation_rad"]
    weakened=translation >= limits["minimum_weak_response_translation_increase_m"] or rotation >= limits["minimum_weak_response_rotation_increase_rad"]
    return {"matched_start_checks":{k:bool(v) for k,v in matched.items()},
        "matched":bool(all(matched.values())),"translation_increase_m":float(translation),"rotation_increase_rad":float(rotation),
        "larger_weak_response":bool(all(matched.values()) and weakened)}


@contextmanager
def open_backend(args, output, protocol):
    from robotic_cutting_sim.physical_tool import ToolGraspConfig
    grip=replace(ToolGraspConfig(), **protocol["normal_grip"])
    spacing=.002 if args.task_id == "TRIM-01" else .001
    if args.runtime == "isaac_sim":
        # Same process-local acknowledgement as the installed native launcher;
        # the existing runtime was explicitly installed with accepted terms.
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
        from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app
        config=IsaacSimCuttingConfig(task_id=args.task_id, seed=args.seed, task_variant="fixed", gpu=args.gpu,
            device=f"cuda:{args.gpu}", particle_spacing_m=spacing, camera=False,
            camera_layout="wrist_d405_global_d455", camera_mode="native_single_view", native_noise="off",
            tool_attachment_mode="physical_grasp", tool_init_mode="pregrasped", grasp=grip)
        write_json(output/"runtime_config.json",asdict(config))
        app=launch_app(config,output)
        env=LiveIsaacSimCutting(app,config,output)
        try: yield env
        finally: env.close()
    else:
        from argparse import Namespace
        from isaaclab.app import launch_simulation
        from robotic_cutting_sim.isaac_lab_env import IsaacLabCuttingEnv, LabCuttingConfig, make_lab_cfg
        from robotic_cutting_sim.workcell import export
        static=output/"workcell";static.mkdir()
        export(Namespace(variant="D0",family="dome",seed=args.seed,spacing=.002,asset_root=None),static)
        config=LabCuttingConfig(task_id=args.task_id,seed=args.seed,device=f"cuda:{args.gpu}",material_device=f"cuda:{args.gpu}",
            particle_spacing_m=spacing,camera=False,camera_layout="wrist_d405_global_d455",grasp=grip,
            tool_attachment_mode="physical_grasp",tool_init_mode="pregrasped",static_scene=str(static/"workcell.usda"))
        write_json(output/"runtime_config.json",asdict(config))
        cfg=make_lab_cfg(config)
        with launch_simulation(cfg,{"headless":True,"visualizer":[],"device":config.device}):
            env=IsaacLabCuttingEnv(cfg,config)
            try:yield env
            finally:env.close()


def measured(env):
    state=env.measured() if hasattr(env,"measured") else env.proprioception()
    keys=("q","qdot","hand_pose","tool_pose","tool_body_pose_xyzw","tool_linear_velocity_world_m_s",
          "tool_angular_velocity_world_rad_s","finger_q","finger_qdot")
    return {key:np.asarray(state[key]).copy() for key in keys if key in state}


def clocks(env, runtime):
    if runtime == "isaac_sim":
        native=float(env.SimulationManager.get_simulation_time())
        return native,native
    return float(env.sim.physics_manager.get_simulation_time()),float(env.dynamic_tool.rigid_time_s)


def material_fingerprint(env):
    state=env.material.debug_state()
    return {"time_s":float(env.material.time_s),"arrays":{
        name:hashlib.sha256(np.ascontiguousarray(state[name]).tobytes()).hexdigest()
        for name in ("positions","velocities","codes","F") if name in state}}


def worker(args, protocol):
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    implementation=output/"implementation";implementation.mkdir()
    paths=[Path(__file__),ROOT/"src/robotic_cutting_sim/physical_tool.py",ROOT/"src/robotic_cutting_sim/scene_frames.py",
        ROOT/"src/robotic_cutting_sim/robot.py",ROOT/"src/robotic_cutting_sim/live_material.py",ROOT/"src/robotic_cutting_sim/tlmpm.py",
        ROOT/"src/robotic_cutting_sim/tasks.py",ROOT/"configs/cutting_tasks_v1.json"]
    paths += [ROOT/"src/robotic_cutting_sim"/name for name in
        (("isaac_sim_runner.py","native_tool.py","physical_contacts.py") if args.runtime == "isaac_sim" else ("isaac_lab_env.py","newton_tool.py"))]
    if args.runtime == "isaac_sim":paths.append(ROOT/"configs/isaac_sim_cutting.kit")
    hashes={str(path.relative_to(ROOT)):sha256(path) for path in paths}
    for path in paths:(implementation/path.name).write_bytes(path.read_bytes())
    from robotic_cutting_sim.recording import manifest
    record=manifest({"case":args.case,"task_id":args.task_id,"seed":args.seed,"gpu":args.gpu,
        "protocol_sha256":sha256(args.protocol),"case_config":protocol["cases"][args.case]},"wp08_rigid_control_"+args.runtime)
    record["implementation_sha256"]=hashes
    write_json(output/"manifest.json",record)
    result={"status":"RUNNING","case":args.case,"runtime":args.runtime,"source_sha256":hashes,
        "case_config":protocol["cases"][args.case],"full_physics_acceptance":False,
        "camera_scope":"NOT_RUN; control cameras explicitly disabled","material_feedback_scope":"NOT_RUN; rigid external-load control only"}
    write_json(output/"result.json",result)
    env=None;control_rows=[];raw_path=output/("physical_grasp_trace_00.jsonl" if args.runtime=="isaac_sim" else "tool_substeps.npz")
    start_wall=time.monotonic()
    try:
        with open_backend(args,output,protocol) as env:
            try:
                env.reset(seed=args.seed)
                result["pregrasp"]=getattr(env,"grasp_reset_receipt",getattr(env,"grasp_initialization",None))
                result["tool_spec"]=env.tool_spec.to_dict()
                result["material_initial"]=material_fingerprint(env)
                if result["pregrasp"] is None:raise RuntimeError("Physical pregrasp evidence is missing")
                required=("rigid_probe_step","solve_edge_ik","set_gripper","set_grip_force_limit","set_grip_friction","set_external_tool_wrench")
                for name in required:
                    if not callable(getattr(env,name,None)):raise RuntimeError("Missing explicit physical backend API: "+name)
                env.set_gripper("hold");env.set_grip_force_limit(protocol["normal_grip"]["max_finger_force_n"])
                env.set_grip_friction(protocol["normal_grip"]["friction"]);env.set_external_tool_wrench(np.zeros(6))
                trace=output/"control_trace.jsonl"
                last_target=measured(env)["q"][0].copy()

                def step(target,phase,wrench):
                    nonlocal last_target
                    if np.max(np.abs(np.asarray(target)-last_target)) > protocol["transfer"]["maximum_joint_increment_rad"]+1e-9:
                        raise RuntimeError("Actual issued joint targets exceed the frozen per-control increment")
                    env.set_external_tool_wrench(wrench)
                    state,diagnostics=env.rigid_probe_step(np.asarray(target,float))
                    native,raw=clocks(env,args.runtime)
                    row={"phase":phase,"native_time_s":native,"raw_trace_time_s":raw,
                        "state":measured(env),"commanded_joint_positions_rad":np.asarray(target).copy(),
                        "commanded_external_edge_wrench_world":np.asarray(wrench).copy(),"diagnostics":diagnostics}
                    control_rows.append(row)
                    last_target=np.asarray(target).copy()
                    with trace.open("a") as stream:stream.write(json.dumps(row,default=json_default,allow_nan=False)+"\n")
                    return row

                state=measured(env);initial_edge=state["tool_pose"].copy()
                goal=initial_edge.copy();goal[:3]=protocol["transfer"]["world_edge_position_m"]
                lift=initial_edge.copy();lift[2]=max(lift[2],goal[2])
                result["transfer_waypoints"]=[]
                for edge in (lift,goal):
                    q,info=env.solve_edge_ik(edge)
                    if (info.get("position_error_m",float("inf")) > protocol["transfer"]["maximum_ik_position_error_m"]
                            or info.get("rotation_error_rad",float("inf")) > protocol["transfer"]["maximum_ik_rotation_error_rad"]):
                        raise RuntimeError("Clearance IK failed its frozen position/orientation bound: "+str(info))
                    start=measured(env)["q"][0]
                    targets=smooth_joint_targets(start,q,protocol["transfer"],protocol["control_dt_s"])
                    result["transfer_waypoints"].append({"desired_edge_pose_xyzw":edge.copy(),"ik":info,"steps":len(targets)})
                    for target in targets:step(target,"clearance_transfer",np.zeros(6))
                hold_target=measured(env)["q"][0]
                for _ in range(round(protocol["transfer"]["post_transfer_stabilization_s"]/protocol["control_dt_s"])):
                    step(hold_target,"post_transfer_stabilization",np.zeros(6))
                state=measured(env)
                position_error,rotation_error=pose_error(goal,np.asarray(state["tool_pose"])[None])
                result["actual_clearance_error"]={"position_m":float(position_error[0]),"rotation_rad":float(rotation_error[0])}
                if (position_error[0] > protocol["transfer"]["maximum_actual_edge_position_error_m"]
                        or rotation_error[0] > protocol["transfer"]["maximum_actual_edge_rotation_error_rad"]):
                    raise RuntimeError("Measured tool did not reach the frozen clearance region")
                result["initial_state"]=state
                hold_target=state["q"][0].copy()
                cfg=protocol["cases"][args.case]
                env.set_grip_force_limit(cfg["force_limit_n"]);env.set_grip_friction(cfg["friction"]);env.set_gripper(cfg["gripper"])
                native_start,raw_start=clocks(env,args.runtime)
                result["evaluation_interval"]={"native_start_s":native_start,"raw_trace_start_s":raw_start,"duration_s":cfg["duration_s"]}
                for i in range(round(cfg["duration_s"]/protocol["control_dt_s"])):
                    u=min(1.,(i+1)*protocol["control_dt_s"]/protocol["measurement"]["load_ramp_s"])
                    ramp=10*u**3-15*u**4+6*u**5
                    step(hold_target,"perturbation",np.asarray(cfg["edge_wrench_world"])*ramp)
                native_end,raw_end=clocks(env,args.runtime)
                result["evaluation_interval"].update(native_end_s=native_end,raw_trace_end_s=raw_end)
                result["material_final"]=material_fingerprint(env)
                result["material_unchanged"]=result["material_initial"] == result["material_final"]
                if args.runtime=="isaac_lab":env.dynamic_tool.save_raw_trace(raw_path)
                arguments=(raw_path,raw_start,raw_end,result["tool_spec"],protocol["measurement"]["minimum_loaded_finger_force_n"])
                samples=(native_samples(*arguments,geometry_profile=protocol.get("native_contact_geometry_profile"),rigid_dt_s=protocol["rigid_dt_s"])
                    if args.runtime=="isaac_sim" else lab_samples(*arguments))
                result["metrics"]=measure_case(samples,control_rows,state,result["tool_spec"],protocol,cfg["duration_s"])
                result["classification"]=classify_case(args.case,result["metrics"],protocol)
                result["status"]="EXECUTED_RIGID_CONTROL"
            finally:
                if args.runtime=="isaac_lab" and hasattr(env,"dynamic_tool") and env.dynamic_tool.raw_trace:
                    if not raw_path.exists():env.dynamic_tool.save_raw_trace(raw_path)
                if hasattr(env,"sim") and getattr(env.sim,"stage",None):env.sim.stage.Export(str(output/"final_control_scene.usda"))
                elif getattr(env,"stage",None):env.stage.Export(str(output/"final_control_scene.usda"))
                result["source_bytes_unchanged"]={path:sha256(ROOT/path)==digest for path,digest in hashes.items()}
                if not all(result["source_bytes_unchanged"].values()):result["status"]="FAIL_SOURCE_CHANGED"
                result["wall_s"]=time.monotonic()-start_wall
                result["raw_trace"]=raw_path.name if raw_path.exists() else None
                result["raw_trace_sha256"]=sha256(raw_path) if raw_path.exists() else None
                write_json(output/"result.json",result)
    except BaseException as error:
        result.update(status="FAIL_RIGID_CONTROL",error=str(error),wall_s=time.monotonic()-start_wall)
        (output/"failure.txt").write_text(traceback.format_exc());write_json(output/"result.json",result)
    print(result["status"],output/"result.json",flush=True)
    return 0 if result["status"]=="EXECUTED_RIGID_CONTROL" and result.get("material_unchanged") else 1


def summarize_matrix(index, output, protocol):
    available={}
    for key,row in index["cases"].items():
        path=output/row["directory"]/"result.json"
        if path.is_file():
            result=read_json(path)
            if (row.get("status")=="EXECUTED_RIGID_CONTROL" and row.get("returncode")==0
                    and row.get("native_contact_view_invalidations",0)==0
                    and result.get("status")=="EXECUTED_RIGID_CONTROL" and result.get("metrics",{}).get("evidence_available")):
                available[key]=result
    comparisons={}
    for runtime in sorted({row["runtime"] for row in index["cases"].values()}):
        baseline=available.get(runtime+"/baseline_hold");strong=available.get(runtime+"/nominal_load")
        for name in ("weak_grip","low_friction"):
            weakened=available.get(runtime+"/"+name)
            if strong and weakened:
                comparison=compare_cases(strong,weakened,protocol)
                comparison["strong_stable"]=strong["classification"]["status"]=="PASS_CASE_ENGINEERING_SCREENS"
                comparison["status"]="PASS_MATCHED_WEAK_RESPONSE" if comparison["larger_weak_response"] and comparison["strong_stable"] else "FAIL_MATCHED_WEAK_RESPONSE"
            else:comparison={"status":"NOT_RUN_MISSING_MATCHED_CASE"}
            comparisons[runtime+"/"+name]=comparison
        if baseline:
            for name in ("nominal_load","edge_torque","stress_load"):
                row=available.get(runtime+"/"+name)
                if not row:continue
                arm=float(np.max(np.abs(np.asarray(row["metrics"]["final_arm_deflection_rad"])-np.asarray(baseline["metrics"]["final_arm_deflection_rad"]))))
                contact=float(np.linalg.norm(np.asarray(row["metrics"]["mean_contact_force_last_window_world_n"])-np.asarray(baseline["metrics"]["mean_contact_force_last_window_world_n"])))
                comparisons[runtime+"/"+name]={"status":"PASS_MEASURED_ARM_RESPONSE" if arm >= protocol["measurement"]["minimum_detectable_arm_response_rad"] else "FAIL_NO_DETECTABLE_ARM_RESPONSE",
                    "final_arm_deflection_difference_rad":arm,"steady_contact_force_change_n":contact,
                    "contact_force_change_detected":contact >= protocol["measurement"]["minimum_detectable_contact_force_change_n"],
                    "scope":"Controlled external wrench; not material-feedback ON/OFF"}
    all_cases=bool(index["cases"]) and len(available)==len(index["cases"]) and all(
        row["classification"]["status"]=="PASS_CASE_ENGINEERING_SCREENS" and row.get("material_unchanged")
        and all(row.get("source_bytes_unchanged",{}).values()) for row in available.values())
    complete=all(all(runtime+"/"+name in available for name in protocol["cases"]) for runtime in {row["runtime"] for row in index["cases"].values()})
    comparisons_ok=bool(comparisons) and all(row["status"].startswith("PASS_") for row in comparisons.values())
    return {"status":"PASS_FROZEN_RIGID_CONTROLS" if all_cases and complete and comparisons_ok else "FAIL_OR_INCOMPLETE_RIGID_CONTROLS",
        "comparisons":comparisons,"all_requested_cases_executed":len(available)==len(index["cases"]),
        "complete_case_set":complete,"independent_physical_audit":"NOT_RUN_BY_THIS_DRIVER",
        "full_physics_acceptance":False,"material_two_way_coupling_validated":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--runtime",choices=("isaac_sim","isaac_lab","both"),default="both")
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--task-id",choices=("CUT-01","TRIM-01"),default="CUT-01")
    parser.add_argument("--cases",nargs="+")
    parser.add_argument("--timeout",type=float,default=900.)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--case",help=argparse.SUPPRESS)
    args=parser.parse_args();args.protocol=args.protocol.resolve()
    if args.gpu < 0 or args.timeout <= 0:parser.error("GPU index must be nonnegative and timeout positive")
    protocol=read_json(args.protocol)
    if args.worker:
        if args.runtime=="both" or args.case not in protocol["cases"]:parser.error("Worker requires one explicit runtime and frozen case")
        return worker(args,protocol)
    selected=args.cases or list(protocol["cases"])
    if any(name not in protocol["cases"] for name in selected):parser.error("Unknown case; use names in the frozen protocol")
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    (output/"protocol.json").write_bytes(args.protocol.read_bytes())
    index={"schema_version":"wp08_physical_controls_index_v1","protocol":{"path":"protocol.json","sha256":sha256(args.protocol)},
        "task_id":args.task_id,"seed":args.seed,"gpu":args.gpu,"cases":{},"status":"PLAN_ONLY_NOT_RUN" if args.dry_run else "RUNNING",
        "scope":"Independent fresh-process physical pregrasp/rigid controls; no material advance or task completion"}
    runtimes=("isaac_sim","isaac_lab") if args.runtime=="both" else (args.runtime,)
    for runtime in runtimes:
        executable=ROOT/(".deps/isaacsim-venv/bin/python" if runtime=="isaac_sim" else ".deps/isaaclab-venv/bin/python")
        for name in selected:
            directory=Path(runtime)/name;key=runtime+"/"+name
            command=[str(executable),str(Path(__file__).resolve()),"--worker","--runtime",runtime,"--case",name,
                "--protocol",str(output/"protocol.json"),"--task-id",args.task_id,"--seed",str(args.seed),"--gpu",str(args.gpu),"--output-dir",str(output/directory)]
            index["cases"][key]={"directory":str(directory),"runtime":runtime,"config":protocol["cases"][name],"command":command,
                "control_trace":"control_trace.jsonl","raw_trace":"physical_grasp_trace_00.jsonl" if runtime=="isaac_sim" else "tool_substeps.npz", "status":"NOT_RUN"}
    write_json(output/"index.json",index)
    if args.dry_run:
        print("PLAN_ONLY_NOT_RUN",output/"index.json");return 0
    for key,row in index["cases"].items():
        print("RUN",key,flush=True);start=time.monotonic()
        log=output/(key.replace("/","_")+".log")
        with log.open("w") as stream:
            process=subprocess.Popen(row["command"],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
            try:code=process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
                row["bounded_termination"]=True
        result_path=output/row["directory"]/"result.json"
        result=read_json(result_path) if result_path.is_file() else {}
        status=result.get("status","FAIL_MISSING_RESULT")
        if status=="RUNNING":status="FAIL_INCOMPLETE_WORKER_RESULT"
        log_paths=[log,output/row["directory"]/"kit.log"]
        invalidations=[{"log":str(path.relative_to(output)),"line":line} for path in log_paths if path.is_file()
            for line in path.read_text(errors="replace").splitlines()
            if re.search(r"(?:simulation\s*view|tensor\s*view).*invalidat",line,re.I)]
        row["native_contact_view_invalidations"]=len(invalidations)
        row["native_contact_view_invalidation_records"]=invalidations
        if invalidations:status="FAIL_NATIVE_CONTACT_VIEW_INVALIDATION"
        row.update(returncode=code,wall_s=time.monotonic()-start,status=status,source_sha256=result.get("source_sha256",{}),log=log.name)
        write_json(output/"index.json",index)
        print("DONE",key,row["status"],flush=True)
    summary=summarize_matrix(index,output,protocol)
    index["status"]=summary["status"];write_json(output/"index.json",index);write_json(output/"result.json",summary)
    print(summary["status"],output/"index.json",flush=True)
    return 0 if summary["status"]=="PASS_FROZEN_RIGID_CONTROLS" else 1


if __name__=="__main__":
    raise SystemExit(main())
