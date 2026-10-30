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
from scipy.spatial.transform import Rotation, Slerp

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


def hand_planning_edge_pose(hand_pose, spec):
    """Planner-only nominal mount on the measured HAND; never a tool update."""
    from robotic_cutting_sim.scene_frames import compose_pose
    mount=compose_pose(spec["hand_from_tool_xyz_xyzw"],spec["tool_from_edge_xyz_xyzw"])
    return compose_pose(hand_pose,mount)


def cartesian_reference(start, goal, u):
    start,goal=np.asarray(start,float),np.asarray(goal,float)
    if start.shape != (7,) or goal.shape != (7,) or not np.isfinite(np.r_[start,goal]).all() or not 0. <= u <= 1.:
        raise ValueError("Cartesian reference requires finite XYZ+XYZW endpoints and progress in[0,1]")
    a=float(np.clip(10*u**3-15*u**4+6*u**5,0.,1.))
    rotations=Slerp([0.,1.],Rotation.from_quat([start[3:],goal[3:]]))
    return np.r_[start[:3]+a*(goal[:3]-start[:3]),rotations(a).as_quat()]


def cartesian_ik_step(start,goal,u,count,previous,solve,transfer):
    """Slow reference progress when needed; never clip or jump joint targets."""
    next_u=min(1.,u+1/count)
    for attempt in range(transfer["maximum_adaptive_subdivisions"]):
        desired=cartesian_reference(start,goal,next_u)
        q,info=solve(desired);q=np.asarray(q,float)
        if q.shape != (7,) or not np.isfinite(q).all():raise ValueError("IK returned invalid arm targets")
        if np.max(np.abs(q-previous)) <= transfer["maximum_joint_increment_rad"]+1e-10:
            if (info.get("position_error_m",float("inf")) > transfer["maximum_ik_position_error_m"]
                    or info.get("rotation_error_rad",float("inf")) > transfer["maximum_ik_rotation_error_rad"]):
                raise ValueError("Cartesian IK failed the unchanged frozen residual bounds")
            return next_u,desired,q,info,attempt
        next_u=(u+next_u)/2
    raise ValueError("Cartesian IK cannot respect the frozen joint increment; no target issued")


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
            active = other.startswith("/World/Franka_0/")
            role = "left_finger" if active and other.endswith("/panda_leftfinger") else "right_finger" if active and other.endswith("/panda_rightfinger") else "other"
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
    tool_override={}
    if getattr(args,"tool_profile",None):
        # Opt-in (protocol v7+): a declared profile tool on the declared lifted hand trajectory.
        if args.runtime != "isaac_sim":raise NotImplementedError("--tool-profile is implemented for isaac_sim only")
        tool_override=dict(tool_profile_id=args.tool_profile,
            tool_profile_hand_offset_world_m=tuple(protocol["tool_profile_retarget"]["hand_offset_world_m"]))
    if args.runtime == "isaac_sim":
        # Same process-local acknowledgement as the installed native launcher;
        # the existing runtime was explicitly installed with accepted terms.
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
        from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app
        config=IsaacSimCuttingConfig(task_id=args.task_id, seed=args.seed, task_variant="fixed", gpu=args.gpu,
            device=f"cuda:{args.gpu}", particle_spacing_m=spacing, camera=False,
            camera_layout="wrist_d405_global_d455", camera_mode="native_single_view", native_noise="off",
            tool_attachment_mode="physical_grasp", tool_init_mode="pregrasped", grasp=grip, **tool_override)
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
        lab_overrides={}
        if "lab_contact_profile" in protocol:
            lab_overrides["grip_contact_profile"]=protocol["lab_contact_profile"]
        config=LabCuttingConfig(task_id=args.task_id,seed=args.seed,device=f"cuda:{args.gpu}",material_device=f"cuda:{args.gpu}",
            particle_spacing_m=spacing,camera=False,camera_layout="wrist_d405_global_d455",grasp=grip,
            tool_attachment_mode="physical_grasp",tool_init_mode="pregrasped",static_scene=str(static/"workcell.usda"),**lab_overrides)
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
    elif (ROOT/"src/robotic_cutting_sim/pad_contact.py").is_file():
        paths.append(ROOT/"src/robotic_cutting_sim/pad_contact.py")
        if "lab_contact_profile" in protocol:paths.append(ROOT/"configs/lab_grip_contact_v3.json")
    if args.task_id in ('CUT-01-R', 'TRIM-01-R', 'CUT-02'):
        paths += [ROOT/'src/robotic_cutting_sim'/name for name in
                  ('tool_profile.py', 'finger_geometry.py', 'tlmpm_v2.py')]
        paths.append(ROOT/'configs/cutting_tasks_v2.json')
    hashes={str(path.relative_to(ROOT)):sha256(path) for path in paths}
    for path in paths:(implementation/path.name).write_bytes(path.read_bytes())
    from robotic_cutting_sim.recording import manifest
    record=manifest({"case":args.case,"task_id":args.task_id,"seed":args.seed,"gpu":args.gpu,"tool_profile":args.tool_profile,
        "protocol_sha256":sha256(args.protocol),"case_config":protocol["cases"][args.case]},"wp08_rigid_control_"+args.runtime)
    record["implementation_sha256"]=hashes
    write_json(output/"manifest.json",record)
    result={"status":"RUNNING","case":args.case,"runtime":args.runtime,"source_sha256":hashes,"tool_profile":args.tool_profile,
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

                def step(target,phase,wrench,desired_edge=None):
                    nonlocal last_target
                    if np.max(np.abs(np.asarray(target)-last_target)) > protocol["transfer"]["maximum_joint_increment_rad"]+1e-9:
                        raise RuntimeError("Actual issued joint targets exceed the frozen per-control increment")
                    env.set_external_tool_wrench(wrench)
                    state,diagnostics=env.rigid_probe_step(np.asarray(target,float))
                    native,raw=clocks(env,args.runtime)
                    row={"phase":phase,"native_time_s":native,"raw_trace_time_s":raw,
                        "state":measured(env),"commanded_joint_positions_rad":np.asarray(target).copy(),
                        "commanded_external_edge_wrench_world":np.asarray(wrench).copy(),"diagnostics":diagnostics}
                    if desired_edge is not None:row["commanded_edge_pose_xyzw"]=np.asarray(desired_edge).copy()
                    control_rows.append(row)
                    last_target=np.asarray(target).copy()
                    with trace.open("a") as stream:stream.write(json.dumps(row,default=json_default,allow_nan=False)+"\n")
                    return row

                state=measured(env)
                cartesian=protocol["transfer"].get("reference_profile")=="cartesian_hand_mount_v4"
                initial_edge=(hand_planning_edge_pose(state["hand_pose"],result["tool_spec"]) if cartesian else state["tool_pose"].copy())
                result["transfer_reference_profile"]=protocol["transfer"].get("reference_profile","legacy_joint_chord")
                result["transfer_planning_start_pose_xyzw"]=initial_edge.copy()
                result["actual_free_edge_before_transfer_xyzw"]=state["tool_pose"].copy()
                goal=initial_edge.copy();goal[:3]=protocol["transfer"]["world_edge_position_m"]
                lift=initial_edge.copy();lift[2]=max(lift[2],goal[2])
                result["transfer_waypoints"]=[]
                for edge in (lift,goal):
                    if cartesian:
                        start=hand_planning_edge_pose(measured(env)["hand_pose"],result["tool_spec"])
                        duration=max(protocol["transfer"]["minimum_waypoint_duration_s"],
                            1.875*np.linalg.norm(edge[:3]-start[:3])/protocol["transfer"]["maximum_cartesian_speed_m_s"])
                        count=max(1,int(math.ceil(duration/protocol["control_dt_s"])));u=0.;issued=0
                        waypoint={"desired_edge_pose_xyzw":edge.copy(),"planning_start_pose_xyzw":start.copy(),
                            "nominal_duration_s":count*protocol["control_dt_s"],"steps":0,"adaptive_subdivisions":0}
                        result["transfer_waypoints"].append(waypoint)
                        while u < 1.-1e-12:
                            if issued >= math.ceil(count*protocol["transfer"]["maximum_time_scale"]):
                                raise RuntimeError("Cartesian transfer exceeded the prefrozen time-scale cap")
                            u,desired,q,info,subdivisions=cartesian_ik_step(start,edge,u,count,last_target,env.solve_edge_ik,protocol["transfer"])
                            step(q,"clearance_transfer",np.zeros(6),desired)
                            issued+=1;waypoint.update(steps=issued,ik=info)
                            waypoint["adaptive_subdivisions"]+=subdivisions
                        continue
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


# --------------------------------------------------------------------------
# Opt-in WP-11 Phase F1 path (protocol "start_profile": "wp11_home_v1", v8+).
# The configured wp11_workcell_v1 scene starts at its declared HOME; the
# near-horizontal TRIM pose is reached with the goal planner's own REORIENT
# schedule. Earlier protocols never reach this code; their worker, gates and
# summary above are unchanged.
# --------------------------------------------------------------------------
WP11_START_PROFILE = "wp11_home_v1"


def wp11_task(protocol):
    """The compiled configured task declared by the protocol (same id and sha256 required)."""
    from robotic_cutting_sim.scene_config import compile_task
    spec = protocol["configured_task"]
    task = compile_task(str(ROOT/spec["scene"]), str(ROOT/spec["goal"]), seed=spec["seed"])
    if task.task_id != spec["task_id"] or task.config_sha256 != spec["config_sha256"]:
        raise ValueError(f"Configured task {task.task_id} ({task.config_sha256[:12]}) differs from the frozen protocol")
    return task


def wp11_reorient_schedule(task, label_prefix="REORIENT"):
    """(start pose, per-control-step WORLD edge targets) of the planner's ideal REORIENT schedule."""
    labels = task.reference_phase_labels
    times = [k[0] for k in task.edge_position_knots_local_txyz]
    index = [i for i, label in enumerate(labels) if label.startswith(label_prefix)]
    if not index or index != list(range(index[0], index[-1]+1)):
        raise ValueError("The configured plan has no single contiguous REORIENT segment")
    start = np.asarray(task.reference_edge_pose_world(times[index[0]]), float)
    targets = np.asarray([task.reference_edge_pose_world(times[i+1]) for i in index], float)
    return start, targets


def wp11_rate_targets(targets, rate):
    """Every rate-th target of the schedule (the last target always kept): rate x the planner's speed."""
    rate = int(rate)
    if rate < 1:
        raise ValueError("Rate multiplier must be a positive integer")
    chosen = list(targets[rate-1::rate])
    if not len(chosen) or not np.array_equal(chosen[-1], targets[-1]):
        chosen.append(targets[-1])
    return np.asarray(chosen, float)


def wp11_edge_wrench(cfg, edge_pose):
    """Constant WORLD wrench at the edge origin from edge-frame axes (x advance, y edge line, z blade normal)."""
    rotation = Rotation.from_quat(np.asarray(edge_pose, float)[3:])
    force = np.asarray(cfg.get("force_edge_n", (0., 0., 0.)), float)
    torque = np.asarray(cfg.get("torque_edge_n_m", (0., 0., 0.)), float)
    return np.r_[rotation.apply(force), rotation.apply(torque)]


def wp11_replay_wrenches(spec, scale, edge_pose):
    """Recorded material reaction forces re-expressed in this run's measured edge frame (torque not recorded)."""
    path = ROOT/spec["path"]
    if sha256(path) != spec["sha256"]:
        raise ValueError("Cutting-load replay source bytes differ from the frozen protocol")
    target = Rotation.from_quat(np.asarray(edge_pose, float)[3:])
    rows = [json.loads(line) for line in path.open()]
    wrenches = []
    for row in rows:
        source = Rotation.from_quat(np.asarray(row[spec["edge_pose_field"]], float)[3:])
        force = np.asarray(row[spec["force_field"]], float)
        wrenches.append(np.r_[scale*target.apply(source.inv().apply(force)), 0., 0., 0.])
    wrenches = np.asarray(wrenches, float)
    if wrenches.shape != (spec["steps"], 6) or not np.isfinite(wrenches).all():
        raise ValueError("Replay source has an unexpected length or non-finite forces")
    return wrenches


def wp11_grip_frame_loads(state, spec, wrench_world):
    """Gravity and an edge wrench about the grip centre (TOOL origin), resolved in HAND axes (y = finger axis)."""
    hand = Rotation.from_quat(np.asarray(state["hand_pose"], float)[3:])
    tool = np.asarray(state["tool_body_pose_xyzw"], float)
    edge = np.asarray(state["tool_pose"], float)
    com = Rotation.from_quat(tool[3:]).apply(np.asarray(spec["com_tool_m"], float))
    weight = np.array([0., 0., -9.81*float(spec["mass_kg"])])
    wrench = np.asarray(wrench_world, float)
    torque = wrench[3:]+np.cross(edge[:3]-tool[:3], wrench[:3])
    return {"gravity_force_hand_n": hand.inv().apply(weight).tolist(),
            "gravity_torque_about_grip_hand_n_m": hand.inv().apply(np.cross(com, weight)).tolist(),
            "com_from_grip_centre_hand_m": hand.inv().apply(com).tolist(),
            "edge_from_grip_centre_hand_m": hand.inv().apply(edge[:3]-tool[:3]).tolist(),
            "applied_force_hand_n": hand.inv().apply(wrench[:3]).tolist(),
            "applied_torque_about_grip_hand_n_m": hand.inv().apply(torque).tolist(),
            "axes": "HAND frame: y = finger closing axis (= v2_y blade normal), z = hand approach (= edge line), x = third axis"}


def wp11_extra_metrics(samples, initial_state, monitor_rows, protocol):
    """Per-finger forces, contact reaction, world tool motion, hand rates and grasp-monitor events."""
    if len(samples) < 2:
        return {"evidence_available": False}
    limits = protocol["measurement"]
    times = np.array([r["time_s"] for r in samples])
    window = times >= times[-1]-limits["stable_window_s"]-1e-7
    left = np.array([r["left_load_n"] for r in samples]); right = np.array([r["right_load_n"] for r in samples])
    force = np.asarray([r["contact_force"] for r in samples], float)
    tool = np.asarray([r["tool"] for r in samples], float); hand = np.asarray([r["hand"] for r in samples], float)
    drift, turn = pose_error(np.asarray(initial_state["tool_body_pose_xyzw"], float), tool)
    dt = np.diff(times)
    hand_rotation = Rotation.from_quat(hand[:, 3:])
    hand_rate = (hand_rotation[1:]*hand_rotation[:-1].inv()).magnitude()/dt
    initial = relative_poses(np.asarray(initial_state["hand_pose"])[None], np.asarray(initial_state["tool_body_pose_xyzw"])[None])[0]
    relative = relative_poses(hand, tool)
    linear = np.linalg.norm(np.diff(relative[:, :3], axis=0), axis=1)/dt
    angular = (Rotation.from_quat(relative[1:, 3:])*Rotation.from_quat(relative[:-1, 3:]).inv()).magnitude()/dt
    final_translation, final_rotation = pose_error(initial, relative[-1:])

    def stats(values):
        return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
                "mean_last_window": float(values[window].mean()), "final": float(values[-1])}
    statuses = [r["status"] for r in monitor_rows]
    # Monitor reports carry no clock: row i is the (i+1)-th control step of the case.
    control_dt = protocol["control_dt_s"]
    first = lambda name: next(((i+1)*control_dt for i, r in enumerate(monitor_rows) if r["status"] == name), None)
    return {"evidence_available": True,
        "left_finger_force_n": stats(left), "right_finger_force_n": stats(right),
        "finger_force_source": "sum of actual Native contact-impulse norms on the handle per finger / dt (normal + friction)",
        "contact_force_on_tool_max_n": float(np.linalg.norm(force, axis=1).max()),
        "contact_force_on_tool_mean_last_window_world_n": force[window].mean(0).tolist(),
        "tool_world_translation_max_m": float(drift.max()), "tool_world_rotation_max_rad": float(turn.max()),
        "hand_angular_speed_max_rad_s": float(hand_rate.max()),
        "relative_linear_speed_max_m_s": float(linear.max()), "relative_angular_speed_max_rad_s": float(angular.max()),
        "final_relative_translation_m": float(final_translation[0]), "final_relative_rotation_rad": float(final_rotation[0]),
        "grasp_monitor": {"status_counts": {name: statuses.count(name) for name in sorted(set(statuses))},
            "ever_lost": any(r["grasp_lost"] for r in monitor_rows), "first_lost_time_s": first("LOST"),
            "first_slipping_time_s": first("SLIPPING"), "final_status": statuses[-1] if statuses else None,
            "max_slip_m": max((r["slip_since_settling_m"] for r in monitor_rows), default=None),
            "max_rotation_rad": max((r["rotation_since_settling_rad"] for r in monitor_rows), default=None),
            "reference": "measured hand/tool at the case start; GraspLossConfig defaults (warning 2 mm / pi/60, loss 20 mm / pi/6)"}}


def classify_wp11(cfg, metrics, extra, protocol, material_unchanged, grip_limit_ok):
    """Predeclared gate group per case: hold, reported, negative_control, diagnostic or release."""
    limits = protocol["measurement"]
    if not metrics.get("evidence_available"):
        return {"status": "NOT_RUN_MISSING_RIGID_EVIDENCE", "checks": {}}
    integrity = {k: metrics[k] is True for k in ("finite", "raw_sampling_complete", "all_contact_samples_valid", "arm_samples_complete")}
    integrity.update(material_unchanged=bool(material_unchanged), grip_within_configured_franka_limit=bool(grip_limit_ok))
    stable = {"relative_translation": metrics["max_relative_translation_m"] <= limits["maximum_relative_translation_m"],
        "relative_rotation": metrics["max_relative_rotation_rad"] <= limits["maximum_relative_rotation_rad"],
        "relative_linear_speed": metrics["maximum_relative_linear_speed_last_window_m_s"] <= limits["maximum_relative_linear_speed_m_s"],
        "relative_angular_speed": metrics["maximum_relative_angular_speed_last_window_rad_s"] <= limits["maximum_relative_angular_speed_rad_s"],
        "loaded_handle_contacts": metrics["bilateral_loaded_handle_fraction"] >= limits["minimum_bilateral_loaded_handle_fraction"],
        "bounded_penetration": metrics["maximum_contact_penetration_m"] <= limits["maximum_contact_penetration_m"],
        "no_unintended_support": metrics["unintended_loaded_contact_steps"] == 0,
        "grasp_monitor_never_lost": not extra["grasp_monitor"]["ever_lost"]}
    gate = cfg["gate"]
    checks = dict(integrity)
    if gate == "hold":
        checks.update(stable)
    elif gate == "release":
        acceleration = metrics["freeflight_vertical_acceleration_m_s2"]
        checks.update(independent_relative_motion=metrics["max_relative_translation_m"] > limits["release_minimum_relative_translation_m"],
            actual_downward_motion=metrics["maximum_downward_tool_displacement_m"] > limits["release_minimum_relative_translation_m"],
            unsupported_gravity_interval=acceleration is not None and abs(acceleration+9.81)/9.81 <= limits["freeflight_gravity_relative_error_max"])
    elif gate not in ("reported", "negative_control", "diagnostic"):
        raise ValueError("Unknown predeclared gate group: "+str(gate))
    passed = all(checks.values())
    return {"gate": gate, "status": "PASS_CASE_GATES" if passed else "FAIL_CASE_GATES",
        "checks": {k: bool(v) for k, v in checks.items()},
        "hold_gate_evaluation": {k: bool(v) for k, v in stable.items()},
        "hold_gates_all_pass": bool(all(stable.values())),
        "gated": gate in ("hold", "release"),
        "matched_comparison": "PENDING_SUMMARY" if gate in ("negative_control", "diagnostic") else "NOT_REQUIRED",
        "full_physics_acceptance": False}


def compare_wp11(reference, perturbed, protocol):
    """Matched start, then a larger measured response (inherited v7 increase thresholds) and grip loss."""
    limits = protocol["measurement"]
    a, b = reference["initial_state"], perturbed["initial_state"]
    distance, angle = pose_error(a["tool_body_pose_xyzw"], np.asarray(b["tool_body_pose_xyzw"])[None])
    matched = {"same_runtime": reference["runtime"] == perturbed["runtime"],
        "same_source_bytes": reference["source_sha256"] == perturbed["source_sha256"],
        "same_start_pose": reference["case_config"]["start"] == perturbed["case_config"]["start"],
        "same_external_load": all(reference["case_config"].get(k) == perturbed["case_config"].get(k)
                                  for k in ("force_edge_n", "torque_edge_n_m", "replay")),
        "initial_tool_translation": distance[0] <= limits["maximum_matched_start_tool_translation_m"],
        "initial_tool_rotation": angle[0] <= limits["maximum_matched_start_tool_rotation_rad"],
        "initial_arm_joints": np.max(np.abs(np.asarray(a["q"])[0]-np.asarray(b["q"])[0])) <= limits["maximum_matched_start_arm_joint_error_rad"]}
    ma, mb = reference["metrics"], perturbed["metrics"]
    translation = mb["max_relative_translation_m"]-ma["max_relative_translation_m"]
    rotation = mb["max_relative_rotation_rad"]-ma["max_relative_rotation_rad"]
    larger = translation >= limits["minimum_weak_response_translation_increase_m"] or rotation >= limits["minimum_weak_response_rotation_increase_rad"]
    loss = (perturbed["extra_metrics"]["grasp_monitor"]["ever_lost"]
            or mb["bilateral_loaded_handle_fraction"] < limits["minimum_bilateral_loaded_handle_fraction"])
    return {"reference_case": reference["case"], "matched_start_checks": {k: bool(v) for k, v in matched.items()},
        "matched": bool(all(matched.values())), "translation_increase_m": float(translation), "rotation_increase_rad": float(rotation),
        "larger_response": bool(all(matched.values()) and larger), "grip_loss_event": bool(loss),
        "reference_holds": reference["classification"]["hold_gates_all_pass"]}


@contextmanager
def open_backend_wp11(args, output, protocol, task):
    from robotic_cutting_sim.physical_tool import ToolGraspConfig
    from robotic_cutting_sim.scene_config import configured_task_payload
    grip = replace(ToolGraspConfig(), **protocol["normal_grip"])
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app
    config = IsaacSimCuttingConfig(task_id=task.task_id, seed=args.seed, gpu=args.gpu, device=f"cuda:{args.gpu}",
        camera=False, camera_layout="wrist_d405_global_d455", camera_mode="native_single_view", native_noise="off",
        tool_attachment_mode="physical_grasp", tool_init_mode="pregrasped", grasp=grip, max_cuts=task.max_cuts,
        configured_task=configured_task_payload(task))
    write_json(output/"runtime_config.json", asdict(config))
    app = launch_app(config, output)
    env = LiveIsaacSimCutting(app, config, output)
    try: yield env
    finally: env.close()


def worker_wp11(args, protocol):
    """One fresh-process F1 case: HOME reset, optional planner REORIENT, then the declared perturbation."""
    from robotic_cutting_sim.grasp_monitor import GraspMonitor
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    implementation = output/"implementation"; implementation.mkdir()
    cfg = protocol["cases"][args.case]; motion = protocol["wp11_motion"]
    paths = [Path(__file__)]+[ROOT/"src/robotic_cutting_sim"/name for name in (
        "physical_tool.py", "scene_frames.py", "robot.py", "live_material.py", "isaac_sim_runner.py", "native_tool.py",
        "physical_contacts.py", "grasp_monitor.py", "scene_config.py", "goal_planner.py", "goals.py", "policies.py",
        "control.py", "tool_profile.py", "finger_geometry.py", "tasks.py", "task_scenarios.py", "tlmpm_v5.py")]
    paths += [ROOT/"configs/isaac_sim_cutting.kit", ROOT/"configs/wp11_scenarios_v1.json",
              ROOT/protocol["configured_task"]["scene"], ROOT/protocol["configured_task"]["goal"]]
    if "replay" in cfg: paths.append(ROOT/protocol["cutting_load_replay_sources"][cfg["replay"]["source"]]["path"])
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    for path in paths: (implementation/path.name).write_bytes(path.read_bytes())
    from robotic_cutting_sim.recording import manifest
    record = manifest({"case": args.case, "seed": args.seed, "gpu": args.gpu, "tool_profile": args.tool_profile,
        "configured_task": protocol["configured_task"], "protocol_sha256": sha256(args.protocol), "case_config": cfg},
        "wp11_f1_rigid_control_isaac_sim")
    record["implementation_sha256"] = hashes
    write_json(output/"manifest.json", record)
    result = {"status": "RUNNING", "case": args.case, "runtime": args.runtime, "source_sha256": hashes, "tool_profile": args.tool_profile,
        "case_config": cfg, "start_profile": WP11_START_PROFILE, "full_physics_acceptance": False,
        "camera_scope": "NOT_RUN; control cameras explicitly disabled",
        "material_feedback_scope": "NOT_RUN; rigid external-load control only (material never advanced)"}
    write_json(output/"result.json", result)
    control_rows = []; raw_path = output/"physical_grasp_trace_00.jsonl"; start_wall = time.monotonic()
    dt = protocol["control_dt_s"]
    try:
        task = wp11_task(protocol)
        if task.tool_profile_id != args.tool_profile:
            raise ValueError("The configured scene tool differs from --tool-profile")
        result["task_id"] = task.task_id
        with open_backend_wp11(args, output, protocol, task) as env:
            try:
                env.reset(seed=args.seed)
                result["pregrasp"] = getattr(env, "grasp_reset_receipt", None)
                result["tool_spec"] = env.tool_spec.to_dict()
                result["material_initial"] = material_fingerprint(env)
                if result["pregrasp"] is None or result["pregrasp"].get("status") != "PASS_STABLE_GRASP":
                    raise RuntimeError("Physical pregrasp evidence is missing or failed")
                for name in ("rigid_probe_step", "solve_edge_ik", "set_gripper", "set_grip_force_limit", "set_grip_friction", "set_external_tool_wrench"):
                    if not callable(getattr(env, name, None)): raise RuntimeError("Missing explicit physical backend API: "+name)
                env.set_gripper("hold"); env.set_grip_force_limit(protocol["normal_grip"]["max_finger_force_n"])
                env.set_grip_friction(protocol["normal_grip"]["friction"]); env.set_external_tool_wrench(np.zeros(6))
                trace = output/"control_trace.jsonl"; monitor_trace = output/"grasp_monitor_trace.jsonl"
                reset_state = measured(env)
                episode_monitor = GraspMonitor(reset_state["hand_pose"], reset_state["tool_body_pose_xyzw"], env.tool_spec)
                last_target = np.asarray(task.home["joint_positions"], float)
                state_box = {"monitor": None, "steps": 0, "case_steps": 0}

                def step(target, phase, wrench, desired_edge=None, subphase=None):
                    nonlocal last_target
                    target = np.asarray(target, float)
                    if np.max(np.abs(target-last_target)) > motion["maximum_joint_increment_rad"]+1e-9:
                        raise RuntimeError("Issued joint targets exceed the predeclared per-control increment (IK branch change?)")
                    env.set_external_tool_wrench(wrench)
                    state, diagnostics = env.rigid_probe_step(target)
                    native, raw = clocks(env, "isaac_sim")
                    rows = env.grasp_trace[-env.config.robot_substeps:]
                    fraction = float(np.mean([r["contacts"]["bilateral_loaded_tool_contact"] for r in rows]))
                    valid = all(r["contacts"]["sensor_valid"] and r["contacts"]["sensor_fresh"] for r in rows)
                    state_box["steps"] += 1
                    episode = episode_monitor.update(state["hand_pose"], state["tool_body_pose_xyzw"], time_s=state_box["steps"]*dt,
                        bilateral_loaded_tool_fraction=fraction, contacts_valid=valid, gripper_command=env.gripper_command)
                    case = None
                    if state_box["monitor"] is not None:
                        state_box["case_steps"] += 1
                        case = state_box["monitor"].update(state["hand_pose"], state["tool_body_pose_xyzw"], time_s=state_box["case_steps"]*dt,
                            bilateral_loaded_tool_fraction=fraction, contacts_valid=valid, gripper_command=env.gripper_command)
                    row = {"phase": phase, "subphase": subphase, "native_time_s": native, "raw_trace_time_s": raw,
                        "state": measured(env), "commanded_joint_positions_rad": target.copy(),
                        "commanded_external_edge_wrench_world": np.asarray(wrench).copy(), "diagnostics": diagnostics}
                    if desired_edge is not None: row["commanded_edge_pose_xyzw"] = np.asarray(desired_edge).copy()
                    control_rows.append(row); last_target = target.copy()
                    with trace.open("a") as stream: stream.write(json.dumps(row, default=json_default, allow_nan=False)+"\n")
                    small = {"phase": phase, "subphase": subphase, "native_time_s": native, "bilateral_loaded_tool_fraction": fraction,
                        "episode_monitor": {k: v for k, v in episode.items() if k != "reference_hand_from_tool_xyz_xyzw"},
                        "case_monitor": None if case is None else {k: v for k, v in case.items() if k != "reference_hand_from_tool_xyz_xyzw"}}
                    with monitor_trace.open("a") as stream: stream.write(json.dumps(small, default=json_default, allow_nan=False)+"\n")
                    return row, case

                def drive(poses, phase, subphase):
                    worst = {"position_error_m": 0., "rotation_error_rad": 0., "steps": 0}
                    rows = []
                    for pose in poses:
                        q, info = env.solve_edge_ik(pose)
                        if (info.get("position_error_m", float("inf")) > motion["maximum_ik_position_error_m"]
                                or info.get("rotation_error_rad", float("inf")) > motion["maximum_ik_rotation_error_rad"]):
                            raise RuntimeError("REORIENT IK failed its predeclared residual bound: "+str(info))
                        rows.append(step(q, phase, np.zeros(6), pose, subphase))
                        worst = {"position_error_m": max(worst["position_error_m"], float(info["position_error_m"])),
                                 "rotation_error_rad": max(worst["rotation_error_rad"], float(info["rotation_error_rad"])),
                                 "steps": worst["steps"]+1}
                    return worst, rows

                home_target = np.asarray(task.home["joint_positions"], float)
                for _ in range(round(motion["home_stabilization_s"]/dt)): step(home_target, "home_stabilization", np.zeros(6))
                home_state = measured(env)
                result["home_state"] = home_state
                schedule_start, schedule = wp11_reorient_schedule(task)
                result["reorient_schedule"] = {"source": motion["reorient_source"], "steps_at_planner_rate": len(schedule),
                    "start_edge_pose_world_xyzw": schedule_start, "trim_edge_pose_world_xyzw": schedule[-1]}
                if cfg["start"] == "trim":
                    targets = wp11_rate_targets(schedule, motion["approach_rate_multiplier"])
                    result["approach_ik_worst"], _ = drive(targets, "approach_reorient", "REORIENT_1x")
                    for _ in range(round(motion["post_motion_stabilization_s"]/dt)): step(last_target, "post_motion_stabilization", np.zeros(6))
                    state = measured(env)
                    position_error, rotation_error = pose_error(schedule[-1], np.asarray(state["tool_pose"])[None])
                    result["actual_trim_pose_error"] = {"position_m": float(position_error[0]), "rotation_rad": float(rotation_error[0])}
                    from robotic_cutting_sim.goal_planner import orientation_metrics
                    result["actual_trim_orientation_metrics"] = orientation_metrics(state["tool_pose"][3:])
                    if (position_error[0] > motion["maximum_actual_edge_position_error_m"]
                            or rotation_error[0] > motion["maximum_actual_edge_rotation_error_rad"]):
                        raise RuntimeError("Measured tool did not reach the predeclared TRIM pose region")
                    approach = relative_poses(np.asarray([home_state["hand_pose"], state["hand_pose"]]),
                        np.asarray([home_state["tool_body_pose_xyzw"], state["tool_body_pose_xyzw"]]))
                    t_err, r_err = pose_error(approach[0], approach[1:])
                    result["approach_slip_vs_home"] = {"relative_translation_m": float(t_err[0]), "relative_rotation_rad": float(r_err[0])}
                elif cfg["start"] != "home":
                    raise ValueError("Unknown start pose: "+str(cfg["start"]))
                state = measured(env)
                result["initial_state"] = state
                result["episode_monitor_before_case"] = {k: v for k, v in episode_monitor.last.items() if k != "reference_hand_from_tool_xyz_xyzw"}
                hold_target = last_target.copy()
                env.set_grip_force_limit(cfg["force_limit_n"]); env.set_grip_friction(cfg["friction"]); env.set_gripper(cfg["gripper"])
                grip_limit_ok = cfg["force_limit_n"] <= protocol["normal_grip"]["max_finger_force_n"]+1e-12
                state_box["monitor"] = GraspMonitor(state["hand_pose"], state["tool_body_pose_xyzw"], env.tool_spec)
                base = wp11_edge_wrench(cfg, state["tool_pose"])
                replay = None
                if "replay" in cfg:
                    source = protocol["cutting_load_replay_sources"][cfg["replay"]["source"]]
                    replay = wp11_replay_wrenches(source, cfg["replay"]["scale"], state["tool_pose"])
                    result["replay_summary"] = {"steps": len(replay), "max_force_n": float(np.linalg.norm(replay[:, :3], axis=1).max()),
                        "mean_force_n": float(np.linalg.norm(replay[:, :3], axis=1).mean()),
                        "mean_force_edge_frame_n": Rotation.from_quat(state["tool_pose"][3:]).inv().apply(replay[:, :3].mean(0)).tolist()}
                    peak = replay[int(np.argmax(np.linalg.norm(replay[:, :3], axis=1)))]
                    result["grip_frame_loads"] = wp11_grip_frame_loads(state, result["tool_spec"], peak)
                else:
                    result["grip_frame_loads"] = wp11_grip_frame_loads(state, result["tool_spec"], base)
                result["applied_constant_wrench_world"] = base
                monitor_rows = []
                native_start, raw_start = clocks(env, "isaac_sim")
                result["evaluation_interval"] = {"native_start_s": native_start, "raw_trace_start_s": raw_start}
                if cfg.get("motion") == "round_trip":
                    forward = wp11_rate_targets(schedule, cfg["rate_multiplier"])
                    backward = wp11_rate_targets(np.vstack([schedule[::-1][1:], schedule_start[None]]), cfg["rate_multiplier"])
                    result["round_trip"] = {"forward_steps": len(forward), "backward_steps": len(backward),
                        "planner_rotation_limit_rad_per_step": float(np.deg2rad(task.controller["planner"]["max_rotation_deg"]))*cfg["rate_multiplier"],
                        "planner_translation_limit_m_per_step": float(task.controller["planner"]["max_translation_m"])*cfg["rate_multiplier"]}
                    for poses, label in ((forward, "HOME_TO_TRIM"), (backward, "TRIM_TO_HOME")):
                        for pose in poses:
                            q, info = env.solve_edge_ik(pose)
                            if (info.get("position_error_m", float("inf")) > motion["maximum_ik_position_error_m"]
                                    or info.get("rotation_error_rad", float("inf")) > motion["maximum_ik_rotation_error_rad"]):
                                raise RuntimeError("Round-trip IK failed its predeclared residual bound: "+str(info))
                            _, case = step(q, "perturbation", np.zeros(6), pose, label); monitor_rows.append(case)
                    hold_target = last_target.copy()
                    for _ in range(round(cfg["hold_after_s"]/dt)):
                        _, case = step(hold_target, "perturbation", np.zeros(6), None, "HOLD_AT_HOME"); monitor_rows.append(case)
                    final = measured(env)
                    result["round_trip"]["final_joint_error_vs_home_max_rad"] = float(np.max(np.abs(final["q"][0]-home_target)))
                    result["round_trip"]["final_tool_world_vs_home_translation_m"] = float(pose_error(home_state["tool_body_pose_xyzw"], np.asarray(final["tool_body_pose_xyzw"])[None])[0][0])
                elif replay is not None:
                    for wrench in replay:
                        _, case = step(hold_target, "perturbation", wrench, None, "REPLAY"); monitor_rows.append(case)
                    for _ in range(round(cfg.get("hold_after_s", 0.)/dt)):
                        _, case = step(hold_target, "perturbation", np.zeros(6), None, "HOLD_AFTER_REPLAY"); monitor_rows.append(case)
                else:
                    for i in range(round(cfg["duration_s"]/dt)):
                        u = min(1., (i+1)*dt/protocol["measurement"]["load_ramp_s"])
                        ramp = 10*u**3-15*u**4+6*u**5
                        _, case = step(hold_target, "perturbation", base*ramp, None, "LOAD"); monitor_rows.append(case)
                duration = state_box["case_steps"]*dt
                native_end, raw_end = clocks(env, "isaac_sim")
                result["evaluation_interval"].update(native_end_s=native_end, raw_trace_end_s=raw_end, duration_s=duration)
                result["material_final"] = material_fingerprint(env)
                result["material_unchanged"] = result["material_initial"] == result["material_final"]
                samples = native_samples(raw_path, raw_start, raw_end, result["tool_spec"], protocol["measurement"]["minimum_loaded_finger_force_n"],
                    geometry_profile=protocol.get("native_contact_geometry_profile"), rigid_dt_s=protocol["rigid_dt_s"])
                result["metrics"] = measure_case(samples, control_rows, state, result["tool_spec"], protocol, duration)
                result["extra_metrics"] = wp11_extra_metrics(samples, state, monitor_rows, protocol)
                result["episode_monitor_final"] = {k: v for k, v in episode_monitor.last.items() if k != "reference_hand_from_tool_xyz_xyzw"}
                result["classification"] = classify_wp11(cfg, result["metrics"], result["extra_metrics"], protocol,
                                                         result["material_unchanged"], grip_limit_ok)
                result["status"] = "EXECUTED_RIGID_CONTROL"
            except BaseException as error:
                # Kit may exit Python inside env.close(); record the failure before leaving the context.
                result.update(status="FAIL_RIGID_CONTROL", error=repr(error))
                (output/"failure.txt").write_text(traceback.format_exc())
                raise
            finally:
                if protocol.get("export_final_control_scene", True) and getattr(env, "stage", None):
                    env.stage.Export(str(output/"final_control_scene.usda"))
                result["source_bytes_unchanged"] = {path: sha256(ROOT/path) == digest for path, digest in hashes.items()}
                if not all(result["source_bytes_unchanged"].values()): result["status"] = "FAIL_SOURCE_CHANGED"
                result["wall_s"] = time.monotonic()-start_wall
                result["raw_trace"] = raw_path.name if raw_path.exists() else None
                result["raw_trace_sha256"] = sha256(raw_path) if raw_path.exists() else None
                write_json(output/"result.json", result)
    except BaseException as error:
        result.update(status="FAIL_RIGID_CONTROL", error=result.get("error") or repr(error), wall_s=time.monotonic()-start_wall)
        if not (output/"failure.txt").exists(): (output/"failure.txt").write_text(traceback.format_exc())
        write_json(output/"result.json", result)
    print(result["status"], output/"result.json", flush=True)
    return 0 if result["status"] == "EXECUTED_RIGID_CONTROL" and result.get("material_unchanged") else 1


def summarize_wp11(index, output, protocol):
    """Per-case gate status, matched negative controls and the gated-suite verdict."""
    available, cases = {}, {}
    for key, row in index["cases"].items():
        path = output/row["directory"]/"result.json"
        result = read_json(path) if path.is_file() else {}
        name = key.split("/", 1)[1]
        ok = (row.get("status") == "EXECUTED_RIGID_CONTROL" and row.get("native_contact_view_invalidations", 0) == 0
              and result.get("status") == "EXECUTED_RIGID_CONTROL" and result.get("metrics", {}).get("evidence_available"))
        if ok: available[name] = result
        cases[name] = {"gate": protocol["cases"][name]["gate"], "run_status": row.get("status"),
                       "case_status": result.get("classification", {}).get("status") if ok else "NOT_EVALUATED"}
    comparisons = {}
    for name, cfg in protocol["cases"].items():
        if "comparison" not in cfg: continue
        reference, perturbed = available.get(cfg["comparison"]), available.get(name)
        if reference is None or perturbed is None:
            comparisons[name] = {"status": "NOT_RUN_MISSING_MATCHED_CASE"}; continue
        comparison = compare_wp11(reference, perturbed, protocol)
        comparison["status"] = "PASS_LARGER_RESPONSE" if comparison["larger_response"] else "FAIL_NO_LARGER_RESPONSE"
        comparison["gated"] = cfg["gate"] == "negative_control"
        comparisons[name] = comparison
    gated = [n for n, c in protocol["cases"].items() if c["gate"] in ("hold", "release", "negative_control")]
    failures = [n for n in gated if n not in available
                or available[n]["classification"]["status"] != "PASS_CASE_GATES"
                or (protocol["cases"][n]["gate"] == "negative_control" and comparisons.get(n, {}).get("status") != "PASS_LARGER_RESPONSE")]
    complete = all(name in available for name in protocol["cases"])
    return {"status": "PASS_F1_GATED_CASES" if complete and not failures else "FAIL_OR_INCOMPLETE_F1_GATED_CASES",
        "cases": cases, "comparisons": comparisons, "gated_cases": gated, "gated_failures": failures,
        "complete_case_set": complete, "independent_physical_audit": "NOT_RUN_BY_THIS_DRIVER",
        "full_physics_acceptance": False, "material_two_way_coupling_validated": False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--runtime",choices=("isaac_sim","isaac_lab","both"),default="both")
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--seed",type=int,default=42)
    from robotic_cutting_sim.tasks import TASK_IDS
    parser.add_argument("--task-id",choices=TASK_IDS,default="CUT-01")
    parser.add_argument("--cases",nargs="+")
    from robotic_cutting_sim.tool_profile import PROFILES
    parser.add_argument("--tool-profile",choices=sorted(PROFILES),default=None,
        help="Opt-in profile tool override; only for protocols declaring allowed_tool_profiles (default: the task's own tool)")
    parser.add_argument("--timeout",type=float,default=900.)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--case",help=argparse.SUPPRESS)
    args=parser.parse_args();args.protocol=args.protocol.resolve()
    if args.gpu < 0 or args.timeout <= 0:parser.error("GPU index must be nonnegative and timeout positive")
    protocol=read_json(args.protocol)
    if args.task_id not in protocol.get('allowed_task_ids', (args.task_id,)):
        parser.error('The selected task is outside this frozen control protocol')
    if (args.tool_profile is not None or "allowed_tool_profiles" in protocol) and \
            args.tool_profile not in protocol.get("allowed_tool_profiles", ()):
        parser.error("--tool-profile must be one of the protocol's allowed_tool_profiles (and is required by such protocols)")
    runtimes=("isaac_sim","isaac_lab") if args.runtime=="both" else (args.runtime,)
    if any(runtime not in protocol.get("allowed_runtimes",("isaac_sim","isaac_lab")) for runtime in runtimes):
        parser.error("The selected runtime is outside the frozen protocol's allowed_runtimes")
    if args.worker:
        if args.runtime=="both" or args.case not in protocol["cases"]:parser.error("Worker requires one explicit runtime and frozen case")
        if protocol.get("start_profile")==WP11_START_PROFILE:return worker_wp11(args,protocol)
        return worker(args,protocol)
    selected=args.cases or list(protocol["cases"])
    if any(name not in protocol["cases"] for name in selected):parser.error("Unknown case; use names in the frozen protocol")
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    (output/"protocol.json").write_bytes(args.protocol.read_bytes())
    index={"schema_version":"wp08_physical_controls_index_v1","protocol":{"path":"protocol.json","sha256":sha256(args.protocol)},
        "task_id":args.task_id,"seed":args.seed,"gpu":args.gpu,"tool_profile":args.tool_profile,"cases":{},"status":"PLAN_ONLY_NOT_RUN" if args.dry_run else "RUNNING",
        "scope":"Independent fresh-process physical pregrasp/rigid controls; no material advance or task completion"}
    for runtime in runtimes:
        executable=ROOT/(".deps/isaacsim-venv/bin/python" if runtime=="isaac_sim" else ".deps/isaaclab-venv/bin/python")
        for name in selected:
            directory=Path(runtime)/name;key=runtime+"/"+name
            command=[str(executable),str(Path(__file__).resolve()),"--worker","--runtime",runtime,"--case",name,
                "--protocol",str(output/"protocol.json"),"--task-id",args.task_id,"--seed",str(args.seed),"--gpu",str(args.gpu),"--output-dir",str(output/directory)]
            if args.tool_profile is not None:command+=["--tool-profile",args.tool_profile]
            index["cases"][key]={"directory":str(directory),"runtime":runtime,"config":protocol["cases"][name],"command":command,
                "control_trace":"control_trace.jsonl","raw_trace":"physical_grasp_trace_00.jsonl" if runtime=="isaac_sim" else "tool_substeps.npz", "status":"NOT_RUN"}
    if protocol.get("start_profile")==WP11_START_PROFILE:
        # Opt-in v8+: the task is the protocol's configured scene/goal (--task-id is not used by the worker).
        index.update(task_id=protocol["configured_task"]["task_id"],configured_task=protocol["configured_task"],start_profile=WP11_START_PROFILE)
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
    summary=(summarize_wp11 if protocol.get("start_profile")==WP11_START_PROFILE else summarize_matrix)(index,output,protocol)
    index["status"]=summary["status"];write_json(output/"index.json",index);write_json(output/"result.json",summary)
    print(summary["status"],output/"index.json",flush=True)
    return 0 if summary["status"] in ("PASS_FROZEN_RIGID_CONTROLS","PASS_F1_GATED_CASES") else 1


if __name__=="__main__":
    raise SystemExit(main())
