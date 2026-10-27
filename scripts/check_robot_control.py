#!/usr/bin/env python3
"""Run real standalone dual-Panda dynamics + IK; never claims physical cutting.

OMP_NUM_THREADS=4 .venv/bin/python scripts/check_robot_control.py --output-dir artifacts/robot_fresh
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.recording import command, manifest, new_output, write_json
from robotic_cutting_sim.robot import DualFrankaController, RobotConfig, compose_tool_pose


def reference_pose(step, count):
    t = step / max(count - 1, 1)
    # A smooth approach, 110 mm surface sweep, then withdrawal; no material here.
    smooth = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
    x = -.055 + .11 * smooth((t - .25) / .5)
    z = .07 - .04 * smooth(t / .25) + .04 * smooth((t - .75) / .25)
    return np.array([x, 0, z, 1, 0, 0, 0])


def plot_trace(output, times, tools, desired, q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 6))
    for axis in range(3):
        axes[0, 0].plot(times, tools[:, axis] * 1000, label="xyz"[axis] + " measured")
        axes[0, 0].plot(times, desired[:, axis] * 1000, "--", alpha=.5)
    axes[0, 0].set(ylabel="knife center [mm]", xlabel="time [s]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(desired[:, 0] * 1000, desired[:, 2] * 1000, "--", label="command")
    axes[0, 1].plot(tools[:, 0] * 1000, tools[:, 2] * 1000, label="measured")
    axes[0, 1].set(xlabel="world x [mm]", ylabel="world z [mm]")
    axes[0, 1].legend()
    axes[1, 0].plot(times, q[:, 0])
    axes[1, 0].set(xlabel="time [s]", ylabel="cutting arm joints [rad]")
    axes[1, 1].plot(times, np.linalg.norm(tools[:, :3] - desired[:, :3], axis=1) * 1000)
    axes[1, 1].set(xlabel="time [s]", ylabel="tool tracking error [mm]")
    figure.suptitle("Measured standalone articulated Panda + fixed knife; no material solver")
    figure.tight_layout()
    figure.savefig(output / "tracking.png", dpi=140)
    plt.close(figure)


def export_replay(robot, times, bodies, tools, path):
    """Author measured link transforms over the official geometry hierarchy."""
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateNew(str(path.resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(float(times[-1] * 60))
    source_paths = [label.split("/", 1)[1] for label in robot.model.body_label]
    operations = []
    for arm in range(2):
        root = UsdGeom.Xform.Define(stage, f"/World/Franka_{arm}").GetPrim()
        root.GetReferences().AddReference(str(robot.asset_path), "/panda")
        root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("none")
        for index in range(robot.bodies_per_arm):
            body_index = arm * robot.bodies_per_arm + index
            source_path = source_paths[body_index]
            destination = f"/World/Franka_{arm}" + source_path[len("/panda"):]
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(destination))
            xf.ClearXformOpOrder()
            # Reset stack: each saved link transform is already in world space.
            op = xf.AddTransformOp()
            xf.SetResetXformStack(True)
            operations.append((body_index, op))
    blade = UsdGeom.Cube.Define(stage, "/World/Knife")
    blade.CreateSizeAttr(2.)
    blade_xf = UsdGeom.Xformable(blade)
    blade_pose_op = blade_xf.AddTransformOp()
    blade_xf.AddScaleOp().Set(Gf.Vec3f(*robot.config.tool_half_extents_m))
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(2.)
    UsdGeom.Xformable(table).AddTranslateOp().Set(Gf.Vec3d(0, 0, -.035))
    UsdGeom.Xformable(table).AddScaleOp().Set(Gf.Vec3f(.9, .6, .025))
    def matrix(pose):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        transform[:3, 3] = pose[:3]
        return Gf.Matrix4d(transform.T.tolist())
    for time_s, state, tool in zip(times, bodies, tools):
        code = Usd.TimeCode(float(time_s * 60))
        for body_index, op in operations:
            op.Set(matrix(state[body_index]), code)
        blade_pose_op.Set(matrix(tool), code)
    for body_index, op in operations:
        op.Set(matrix(bodies[0, body_index]))
    blade_pose_op.Set(matrix(tools[0]))
    stage.GetRootLayer().Save()
    # Read back numerical transforms, not merely successful file creation.
    check = Usd.Stage.Open(str(path.resolve()))
    worst = 0.
    for frame in (0, len(times) - 1):
        cache = UsdGeom.XformCache(Usd.TimeCode(float(times[frame] * 60)))
        for body_index, _ in operations:
            arm = body_index // robot.bodies_per_arm
            p = f"/World/Franka_{arm}" + source_paths[body_index][len("/panda"):]
            read = np.asarray(cache.GetLocalToWorldTransform(check.GetPrimAtPath(p))).T
            worst = max(worst, float(np.linalg.norm(read[:3, 3] - bodies[frame, body_index, :3])))
    return {"status": "PASS" if worst < 1e-6 else "FAIL", "max_link_translation_error_m": worst,
            "scope": "measured dynamic articulation replay; offline USD, no RTX or MPM"}


def collision_checks(robot, q0, output):
    """Negative static collision queries plus actual articulated table response."""
    if robot.config.backend != "mujoco_cpu":
        return {"tests": {"self_collision_detection": "NOT_RUN", "robot_table_detection": "NOT_RUN",
                           "tool_table_dynamic_response": "NOT_RUN"},
                "reason": "Native diagnostic MjData queries are checked on the CPU backend"}
    import mujoco
    robot.reset(q0)
    robot.step_joints(q0)
    native_park = robot.solver.mj_data.qpos.copy()
    data = mujoco.MjData(robot.solver.mj_model)
    examples = {}
    rng = np.random.default_rng(42)
    for attempt in range(200):
        candidate = rng.uniform(robot.lower[:7], robot.upper[:7])
        data.qpos[:] = native_park
        data.qpos[:7] = candidate
        mujoco.mj_forward(robot.solver.mj_model, data)
        for contact in data.contact:
            names = [mujoco.mj_id2name(robot.solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                     for g in contact.geom]
            if contact.dist >= -.005:
                continue
            category = ("self_collision_detection" if all("arm_0" in n and "knife" not in n for n in names)
                        else "robot_table_detection" if any("table" in n for n in names)
                        and any("arm_0" in n and "knife" not in n for n in names) else None)
            if category and category not in examples:
                examples[category] = {"seed": 42, "attempt": attempt, "q_rad": candidate.tolist(),
                                      "geoms": names, "distance_m": float(contact.dist)}
        if len(examples) == 2:
            break
    robot.reset(q0)
    states, targets, contacts, forces, min_distance = [], [], [], [], []
    for step in range(300):
        t = min(step / 180, 1.)
        t = t * t * (3 - 2 * t)
        target = np.array([-.055, 0, .07 - .068 * t, 1, 0, 0, 0])
        state, _ = robot.step_cartesian(target)
        report = robot.diagnostic_contacts()
        knife = [r for r in report["contacts"] if any("knife" in n for n in r["geoms"])
                 and any("table" in n for n in r["geoms"])]
        states.append(state)
        targets.append(target)
        contacts.append(knife)
        forces.append(sum(r["normal_force_N"] for r in knife))
        min_distance.append(min([0.] + [r["distance_m"] for r in knife]))
    tool = np.array([s["tool_pose"] for s in states])
    np.savez_compressed(output / "collision_response.npz", q=np.array([s["q"] for s in states]),
        tool_pose=tool, desired_tool_pose=targets, knife_table_normal_force_N=forces,
        knife_table_min_distance_m=min_distance)
    write_json(output / "collision_contacts.json", contacts)
    response_ok = max(forces) > 1. and min(min_distance) > -.001 and tool[-1, 2] > .004
    result = {"tests": {"self_collision_detection": "PASS" if "self_collision_detection" in examples else "FAIL",
                        "robot_table_detection": "PASS" if "robot_table_detection" in examples else "FAIL",
                        "tool_table_dynamic_response": "PASS" if response_ok else "FAIL"},
              "negative_static_queries": examples, "max_knife_table_normal_force_N": max(forces),
              "maximum_knife_table_penetration_m": -min(min_distance),
              "final_requested_tool_center_z_m": targets[-1][2], "final_measured_tool_center_z_m": tool[-1, 2],
              "scope": "rigid collisions and articulated table response; no material cutting force"}
    write_json(output / "collision_checks.json", result)
    return result


def wrench_check(robot, q0, output):
    robot.reset(q0)
    for _ in range(120):
        state, _ = robot.step_joints(q0)
    baseline = state["tool_pose"].copy()
    tools = []
    for step in range(240):
        state, _ = robot.step_joints(q0, tool_wrench_world=[20, 0, 0, 0, 0, 0] if step < 120 else None)
        tools.append(state["tool_pose"])
    displacement = np.asarray(tools)[119, :3] - baseline[:3]
    return_error = float(np.linalg.norm(np.asarray(tools)[-1, :3] - baseline[:3]))
    np.savez_compressed(output / "diagnostic_wrench_response.npz", tool_pose=tools, baseline=baseline,
                        world_tool_wrench=np.r_[np.tile([[20, 0, 0, 0, 0, 0]], (120, 1)), np.zeros((120, 6))])
    return {"status": "PASS" if displacement[0] > .001 and return_error < .001 else "FAIL",
            "tool_displacement_after_20N_world_x_m": displacement.tolist(), "unload_return_error_m": return_error,
            "scope": "diagnostic force injection validates articulated response; no material-force source"}


def memory_sample():
    result = command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    gpu = []
    for row in result.get("stdout", "").splitlines():
        try:
            pid, mib = row.split(",")
            if int(pid.strip()) == os.getpid():
                gpu.append(float(mib.strip()))
        except ValueError:
            continue
    return {"process_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
            "process_gpu_memory_MiB_per_device": gpu,
            "scope": "RSS high-water mark and one owned-process GPU memory sample; not a GPU peak guarantee"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["mujoco_cpu", "mujoco_warp"], default="mujoco_cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=360)
    args = parser.parse_args()
    if args.steps < 60:
        raise SystemExit("At least 60 steps are required to exercise the path")
    output = new_output(args.output_dir)
    config = RobotConfig(backend=args.backend, device=args.device)
    record = manifest(vars(args), "newton_dual_franka_" + args.backend)
    record["packages"].update({name: importlib.metadata.version(name)
                               for name in ("mujoco", "mujoco-warp", "newton-usd-schemas")})
    record["implementation_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).resolve().parents[1] / "src/robotic_cutting_sim/robot.py")}
    write_json(output / "manifest.json", record)
    start = time.perf_counter()
    robot = DualFrankaController(config)
    write_json(output / "robot.json", robot.description())
    q0, initial_ik = robot.inverse_kinematics(reference_pose(0, args.steps))
    if not initial_ik["converged"]:
        write_json(output / "result.json", {"status": "FAIL", "initial_ik": initial_ik})
        return 1
    robot.reset(q0)
    desired, state_log, body_log, timing, contact_log = [], [], [], [], []
    for step in range(args.steps):
        target = reference_pose(step, args.steps)
        state, info = robot.step_cartesian(target)
        desired.append(target)
        state_log.append(state)
        # Warp CPU .numpy() may share the two reusable state buffers.
        body_log.append(robot.state.body_q.numpy().copy())
        timing.append(info)
        contact_log.append(robot.diagnostic_contacts())
    times = np.array([s["timestamp_s"] for s in state_log])
    q = np.array([s["q"] for s in state_log])
    qdot = np.array([s["qdot"] for s in state_log])
    tools = np.array([s["tool_pose"] for s in state_log])
    desired, bodies = np.array(desired), np.array(body_log)
    errors = np.linalg.norm(tools[:, :3] - desired[:, :3], axis=1)
    park_drift = float(np.max(np.abs(q[:, 1] - robot.initial_q[9:16])))
    np.savez_compressed(output / "trajectory.npz", time_s=times, q=q, qdot=qdot,
                        body_pose=bodies, tool_pose=tools, desired_tool_pose=desired)
    write_json(output / "contacts.json", contact_log)
    write_json(output / "timing.json", timing)
    plot_trace(output, times, tools, desired, q)
    replay = export_replay(robot, times[::6], bodies[::6], tools[::6], output / "dual_franka_replay.usda")
    # Fresh reset must also clear native warm-start/actuator state.
    repeated = []
    for _ in range(2):
        robot.reset(q0)
        run = []
        for step in range(60):
            run.append(robot.step_cartesian(reference_pose(step, args.steps))[0]["q"])
        repeated.append(np.array(run))
    reset_difference = float(np.max(np.abs(repeated[0] - repeated[1])))
    np.savez_compressed(output / "reset_repeat.npz", q_first=repeated[0], q_second=repeated[1])
    # A separate small joint-position command actually changes the articulation.
    robot.reset(q0)
    joint_target = q0.copy()
    joint_target[0] += .05
    for _ in range(120):
        joint_state, _ = robot.step_joints(joint_target)
    joint_error = float(np.max(np.abs(joint_state["q"][0] - joint_target)))
    joint_displacement = float(np.max(np.abs(joint_state["q"][0] - q0)))
    collisions = collision_checks(robot, q0, output)
    wrench = wrench_check(robot, q0, output)
    contact_count = (sum(c["count"] for c in contact_log) if args.backend == "mujoco_cpu" else None)
    tests = {
        "articulation_joint_control": "PASS" if joint_error < .001 and joint_displacement > .04 else "FAIL",
        "cartesian_fk_ik_path": "PASS" if errors.max() < .01 else "FAIL",
        "parked_second_arm": "PASS" if park_drift < .001 else "FAIL",
        "fixed_knife_attachment": "PASS" if all(np.linalg.norm(compose_tool_pose(b[robot.hand_local], config.tool_offset_m)[:3] - t[:3]) < 1e-8 for b, t in zip(bodies, tools)) else "FAIL",
        "collision_free_reference": "PASS" if contact_count == 0 else ("NOT_RUN" if contact_count is None else "FAIL"),
        "same_reset_repeat": "PASS" if reset_difference < 1e-6 else "FAIL",
        "finite_dynamics": "PASS" if np.isfinite(q).all() and np.isfinite(qdot).all() else "FAIL",
        "measured_link_replay": replay["status"],
        "material_contact_coupling": "NOT_RUN", "bidirectional_material_force": "NOT_RUN",
        "isaac_lab_control": "BLOCKED",
        "diagnostic_tool_wrench_response": wrench["status"],
    }
    tests.update(collisions["tests"])
    result = {"status": "FAIL" if "FAIL" in tests.values() else "PASS_STANDALONE_ROBOT",
        "tests": tests, "steps": args.steps, "simulated_duration_s": float(times[-1]),
        "wall_s_including_construction_and_extra_checks": time.perf_counter() - start,
        "construction_wall_s": robot.construction_wall_s,
        "mean_physics_wall_s": float(np.mean([t["physics_wall_s"] for t in timing])),
        "mean_ik_wall_s": float(np.mean([t["ik_wall_s"] for t in timing])),
        "max_tracking_error_m": float(errors.max()), "rms_tracking_error_m": float(np.sqrt(np.mean(errors ** 2))),
        "max_park_drift_rad": park_drift, "reset_q_max_difference_rad": reset_difference,
        "joint_control_final_error_rad": joint_error, "joint_control_displacement_rad": joint_displacement,
        "rigid_reference_contact_count": contact_count, "replay": replay,
        "collision_validation": collisions,
        "diagnostic_wrench_validation": wrench, "memory": memory_sample(),
        "physical_cutting_validated": False, "material_force_coupling_validated": False,
        "rtx_validated": False, "limitations": ["standalone rigid robot test, no MPM material",
            "collider support validated separately from physical cutting", "unvalidated Isaac Lab runtime"]}
    result["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in output.iterdir() if p.suffix in (".npz", ".usda")}
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2))
    return int(result["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
