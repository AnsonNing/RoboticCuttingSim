#!/usr/bin/env python3
"""Frozen, independent LOCAL/WORLD task-adapter and finite-fixture GPU probes.

The blade input is prescribed measured-edge data, not a robot-controller rollout.
Cutting is deliberately disarmed: this isolates frame/contact covariance from
fracture thresholds. Full CUT/TRIM task acceptance belongs to separate episodes.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.coupled_robot import CouplingConfig
from robotic_cutting_sim.live_material import LiveCuttingMaterial
from robotic_cutting_sim.recording import command, manifest, new_output, write_json
from robotic_cutting_sim.scene_frames import transform_points, transform_poses
from robotic_cutting_sim.tasks import make_task


# Frozen before any numerical output of these controls. Do not change tolerances
# or trajectories after seeing results under this protocol id.
PROTOCOL = {
    "protocol_id": "task_frame_fixture_v1_20261028",
    "seed": 42,
    "task": "CUT-01",
    "frame_yaw_degrees": 37.0,
    "frame_translation_m": [0.17, -0.11, 0.0],
    "tilt_xyz_degrees": [17.0, -11.0, 29.0],
    "tilt_translation_m": [0.12, -0.08, 0.035],
    "contact_segment": {
        "control_steps": 5,
        "start_xyz_local_m": [-0.0008, 0.0, 0.021],
        "end_xyz_local_m": [0.0008, 0.0, 0.018],
        "orientation": "actual CUT-01 edge_orientation_local_xyzw",
        "cutting_enabled": False,
        "scope": "prescribed measured blade contact only, not a robot or severing episode",
    },
    "criteria": {
        "pose_roundtrip_position_max_m": 1e-12,
        "pose_roundtrip_angle_max_rad": 1e-10,
        "local_position_max_m": 2e-5,
        "local_velocity_max_m_s": 2e-3,
        "local_F_frobenius_max": 2e-3,
        "local_impulse_force_difference_max_Ns": 1e-5,
        "local_impulse_torque_difference_max_Nms": 1e-6,
        "world_force_rotation_residual_max_Ns": 1e-9,
        "world_torque_rotation_residual_max_Nms": 1e-9,
        "world_state_roundtrip_position_max_m": 1e-10,
        "world_state_roundtrip_velocity_max_m_s": 1e-10,
        "minimum_peak_force_N": 1e-4,
        "minimum_nonzero_horizontal_rotation_effect_N": 1e-5,
        "gravity_analytic_residual_max_m_s2": 1e-12,
        "table_plane_analytic_residual_max_m": 1e-7,
        "fixture_impulse_absolute_tolerance_Ns": 1e-10,
        "fixture_impulse_relative_tolerance": 1e-4,
        "fixture_zero_impulse_max_Ns": 1e-14,
        "minimum_positive_fixture_impulse_Ns": 1e-9,
        "initial_particles_masses_and_lattice_identical": True,
        "all_cut_codes_zero": True,
        "finite_states_and_reaction": True,
    },
    "fixture_controls": [
        {"id": "left_pedestal_top", "point_local_m": [-0.055, 0.0, 0.0002],
         "velocity_local_m_s": [0.02, 0.0, -0.03], "contact": "box_top"},
        {"id": "finite_box_outside", "point_local_m": [-0.095, 0.0, -0.003],
         "velocity_local_m_s": [0.0, 0.0, -0.03], "contact": "none"},
        {"id": "open_plunge_slot", "point_local_m": [0.0, 0.0, -0.003],
         "velocity_local_m_s": [0.0, 0.0, -0.03], "contact": "none"},
        {"id": "world_table_in_slot", "point_local_m": [0.0, 0.0, -0.0098],
         "velocity_local_m_s": [0.0, 0.01, -0.02], "contact": "table"},
    ],
    "tilted_table_control": {
        "point_world_m": [0.03, -0.02, -0.0098],
        "velocity_world_m_s": [0.01, 0.02, -0.03],
        "boxes_enabled": False,
        "scope": "actual transformed table kernel, separate from any tilted material episode",
    },
}
ROOT = Path(__file__).resolve().parents[1]


def norm_max(values):
    return float(np.max(np.linalg.norm(values, axis=-1)))


def state_history(states):
    return {key: np.stack([state[key] for state in states])
            for key in ("positions", "velocities", "F", "codes")}


def run_segment(task, device, output, name):
    import warp as wp

    c = CouplingConfig(device=device, dx=task.grid_dx_m, control_dt=task.control_dt_s)
    adapter = LiveCuttingMaterial(c, task=task)
    segment = PROTOCOL["contact_segment"]
    xyz = np.linspace(segment["start_xyz_local_m"], segment["end_xyz_local_m"],
                      segment["control_steps"]+1)
    local = np.column_stack([xyz, np.tile(task.edge_orientation_local_xyzw, (len(xyz), 1))])
    world = transform_poses(adapter.T_world_local, local)
    recovered = transform_poses(adapter.T_local_world, world)
    angle = (Rotation.from_quat(recovered[:, 3:]) * Rotation.from_quat(local[:, 3:]).inv()).magnitude()
    start = time.perf_counter()
    reset = adapter.reset(world[0], PROTOCOL["seed"])
    states, local_impulses, world_impulses, diagnostics = [adapter.debug_state()], [], [], []
    for index, pose in enumerate(world[1:], 1):
        info = adapter.advance(pose, index*c.control_dt, cutting=False)
        diagnostics.append(info)
        local_impulses.append(adapter.bed.reaction.numpy()[0].astype(float))
        world_impulses.append(info["current_material_reaction_impulse_world"])
        states.append(adapter.debug_state())
    world_state = adapter.world_state()
    position_roundtrip = transform_points(adapter.T_local_world, world_state["positions"])-states[-1]["positions"]
    velocity_roundtrip = world_state["velocities"]@adapter.T_world_local[:3, :3]-states[-1]["velocities"]
    kernel, centers, halves, plane = adapter.bed.fixture_data
    fixture = {"centers_local_m": centers.numpy(), "half_extents_local_m": halves.numpy(),
               "table_plane_local": np.asarray(list(plane)), "radius_m": adapter.bed.radius,
               "dt_s": adapter.material_config.dt, "stiffness": adapter.material_config.contact_stiffness,
               "friction": adapter.material_config.contact_friction,
               "particle_mass_kg": float(adapter.masses[0])}
    arrays = {**state_history(states), "reference_positions_local_m": adapter.points,
              "mass_kg": adapter.masses, "lattice": adapter.lattice,
              "measured_edge_pose_local": local, "measured_edge_pose_world": world,
              "local_reaction_impulse": np.asarray(local_impulses),
              "world_reaction_impulse": np.asarray(world_impulses),
              "time_s": np.arange(len(local))*c.control_dt}
    np.savez_compressed(output/(name+".npz"), **arrays)
    rotation = adapter.T_world_local[:3, :3]
    expected_world = np.column_stack([arrays["local_reaction_impulse"][:, :3]@rotation.T,
                                      arrays["local_reaction_impulse"][:, 3:]@rotation.T])
    receipt = {"task": task.to_dict(), "reset": reset, "fixture": fixture,
               "prescribed_measured_edge_scope": segment["scope"], "diagnostics": diagnostics,
               "pose_roundtrip_position_max_m": norm_max(recovered[:, :3]-local[:, :3]),
               "pose_roundtrip_angle_max_rad": float(angle.max()),
               "world_state_roundtrip_position_max_m": norm_max(position_roundtrip),
               "world_state_roundtrip_velocity_max_m_s": norm_max(velocity_roundtrip),
               "world_force_rotation_residual_max_Ns": norm_max(arrays["world_reaction_impulse"][:, :3]-expected_world[:, :3]),
               "world_torque_rotation_residual_max_Nms": norm_max(arrays["world_reaction_impulse"][:, 3:]-expected_world[:, 3:]),
               "peak_force_N": norm_max(arrays["local_reaction_impulse"][:, :3])/c.control_dt,
               "finite": all(np.isfinite(q).all() for q in arrays.values()),
               "all_cut_codes_zero": bool(np.all(arrays["codes"] == 0)),
               "wall_s_including_recording": time.perf_counter()-start,
               "timing_exclusive": False}
    write_json(output/(name+".json"), receipt)
    del adapter
    gc.collect()
    wp.synchronize_device(device)
    return arrays, receipt


def expected_surface_impulse(gap, normal, velocity, mass, dt, stiffness, friction):
    """Independent scalar normal/friction ledger for known flat control faces."""
    normal, velocity = np.asarray(normal), np.asarray(velocity)
    normal_speed = float(velocity@normal)
    penetration_next = max(0., -gap-dt*min(0., normal_speed))
    normal_impulse = stiffness*mass*penetration_next/dt
    tangent_momentum = mass*(velocity-normal_speed*normal)
    tangent_norm = np.linalg.norm(tangent_momentum)
    friction_impulse = (np.zeros(3) if tangent_norm == 0 else
                        -tangent_momentum*min(1., friction*normal_impulse/tangent_norm))
    return normal_impulse*normal+friction_impulse


def launch_fixture(device, fixture, points, velocities, *, omit_boxes=False):
    import warp as wp
    from robotic_cutting_sim.fixture_contacts import fixture_contacts

    count = len(points)
    centers = np.empty((0, 3), np.float32) if omit_boxes else fixture["centers_local_m"]
    halves = np.empty((0, 3), np.float32) if omit_boxes else fixture["half_extents_local_m"]
    impulse, stats = wp.zeros(count, dtype=wp.vec3, device=device), wp.zeros(8, device=device)
    wp.launch(fixture_contacts, count, [
        wp.array(np.asarray(points, np.float32), dtype=wp.vec3, device=device),
        wp.array(np.asarray(velocities, np.float32), dtype=wp.vec3, device=device),
        wp.full(count, fixture["particle_mass_kg"], dtype=float, device=device), impulse,
        wp.array(np.asarray(centers, np.float32), dtype=wp.vec3, device=device),
        wp.array(np.asarray(halves, np.float32), dtype=wp.vec3, device=device),
        wp.vec4(*fixture["table_plane_local"]), fixture["radius_m"], fixture["dt_s"],
        fixture["stiffness"], fixture["friction"], stats], device=device)
    wp.synchronize_device(device)
    return impulse.numpy().astype(float), stats.numpy()


def fixture_controls(task, fixture, device):
    cases = PROTOCOL["fixture_controls"]
    actual, stats = launch_fixture(device, fixture, [q["point_local_m"] for q in cases],
                                   [q["velocity_local_m_s"] for q in cases])
    rows = []
    criteria = PROTOCOL["criteria"]
    for index, case in enumerate(cases):
        if case["contact"] == "none":
            expected = np.zeros(3)
        else:
            surface_z = (task.fixture_boxes_local[0].center_m[2]+task.fixture_boxes_local[0].half_extents_m[2]
                         if case["contact"] == "box_top" else task.table_top_world_z_m)
            gap = case["point_local_m"][2]-surface_z-fixture["radius_m"]
            expected = expected_surface_impulse(gap, [0., 0., 1.], case["velocity_local_m_s"],
                fixture["particle_mass_kg"], fixture["dt_s"], fixture["stiffness"], fixture["friction"])
        error = float(np.linalg.norm(actual[index]-expected))
        tolerance = criteria["fixture_impulse_absolute_tolerance_Ns"]+criteria["fixture_impulse_relative_tolerance"]*np.linalg.norm(expected)
        passed = error <= tolerance
        if case["contact"] == "none":
            passed &= np.linalg.norm(actual[index]) <= criteria["fixture_zero_impulse_max_Ns"]
        else:
            passed &= np.linalg.norm(actual[index]) >= criteria["minimum_positive_fixture_impulse_Ns"]
        rows.append({**case, "actual_impulse_local_Ns": actual[index], "expected_impulse_local_Ns": expected,
                     "error_Ns": error, "tolerance_Ns": tolerance, "passed": bool(passed)})
    return {"actual_warp_kernel": "robotic_cutting_sim.fixture_contacts.fixture_contacts",
            "actual_fixture_inputs": fixture, "rows": rows, "kernel_statistics": stats,
            "passed": all(q["passed"] for q in rows)}


def tilted_receipt(task, device, output):
    import warp as wp

    adapter = LiveCuttingMaterial(CouplingConfig(device=device, dx=task.grid_dx_m, control_dt=task.control_dt_s), task=task)
    reset = adapter.reset(task.reference_edge_pose_world(0.), PROTOCOL["seed"])
    _, centers, halves, plane = adapter.bed.fixture_data
    actual_plane = np.asarray(list(plane), float)
    rotation, translation = adapter.T_world_local[:3, :3], adapter.T_world_local[:3, 3]
    expected_gravity = rotation.T@np.array([0., 0., -9.81])
    actual_gravity = np.asarray(adapter.material_config.gravity)
    expected_plane = np.r_[rotation[2], translation[2]-task.table_top_world_z_m]
    local_test = np.array([[-.08, -.015, 0.], [.08, .015, .02], [0., 0., -.003]])
    world_test = transform_points(adapter.T_world_local, local_test)
    plane_residual = local_test@actual_plane[:3]+actual_plane[3]-(world_test[:, 2]-task.table_top_world_z_m)
    controls = PROTOCOL["tilted_table_control"]
    local_point = transform_points(adapter.T_local_world, controls["point_world_m"])
    local_velocity = np.asarray(controls["velocity_world_m_s"])@rotation
    fixture = {"centers_local_m": centers.numpy(), "half_extents_local_m": halves.numpy(),
               "table_plane_local": actual_plane, "radius_m": adapter.bed.radius,
               "dt_s": adapter.material_config.dt, "stiffness": adapter.material_config.contact_stiffness,
               "friction": adapter.material_config.contact_friction, "particle_mass_kg": float(adapter.masses[0])}
    actual_impulse, stats = launch_fixture(device, fixture, [local_point], [local_velocity], omit_boxes=True)
    actual_world = actual_impulse[0]@rotation.T
    expected_world = expected_surface_impulse(controls["point_world_m"][2]-task.table_top_world_z_m-fixture["radius_m"],
        [0., 0., 1.], controls["velocity_world_m_s"], fixture["particle_mass_kg"], fixture["dt_s"], fixture["stiffness"], fixture["friction"])
    error = float(np.linalg.norm(actual_world-expected_world))
    c = PROTOCOL["criteria"]
    checks = {
        "gravity": norm_max(actual_gravity-expected_gravity) <= c["gravity_analytic_residual_max_m_s2"],
        "table_coefficients": norm_max(actual_plane-expected_plane) <= c["table_plane_analytic_residual_max_m"],
        "table_signed_distance": float(np.abs(plane_residual).max()) <= c["table_plane_analytic_residual_max_m"],
        "table_impulse": error <= c["fixture_impulse_absolute_tolerance_Ns"]+c["fixture_impulse_relative_tolerance"]*np.linalg.norm(expected_world),
        "table_impulse_nonzero": np.linalg.norm(actual_world) >= c["minimum_positive_fixture_impulse_Ns"],
        "legacy_floor_disabled": not adapter.material_config.floor and not adapter.material_config.fixture_walls,
    }
    receipt = {"scope": "actual adapter configuration plus direct transformed-table kernel; no tilted material episode",
               "task": task.to_dict(), "reset": reset, "gravity_analytic_only": True,
               "actual_gravity_local_m_s2": actual_gravity, "expected_gravity_local_m_s2": expected_gravity,
               "actual_table_plane_local": actual_plane, "expected_table_plane_local": expected_plane,
               "table_signed_distance_residual_m": plane_residual, "point_local_m": local_point,
               "actual_table_impulse_world_Ns": actual_world, "expected_table_impulse_world_Ns": expected_world,
               "table_impulse_error_Ns": error, "kernel_statistics": stats,
               "checks": checks, "passed": all(checks.values())}
    write_json(output/"tilted_frame.json", receipt)
    del adapter
    gc.collect()
    wp.synchronize_device(device)
    return receipt


def compare(identity, rotated, rotated_receipt, identity_receipt):
    c = PROTOCOL["criteria"]
    R = Rotation.from_euler("z", PROTOCOL["frame_yaw_degrees"], degrees=True).as_matrix()
    impulse_difference = rotated["local_reaction_impulse"]-identity["local_reaction_impulse"]
    measurements = {
        "local_position_max_m": norm_max(rotated["positions"]-identity["positions"]),
        "local_velocity_max_m_s": norm_max(rotated["velocities"]-identity["velocities"]),
        "local_F_frobenius_max": float(np.max(np.linalg.norm(rotated["F"]-identity["F"], axis=(-2, -1)))),
        "local_impulse_force_difference_max_Ns": norm_max(impulse_difference[:, :3]),
        "local_impulse_torque_difference_max_Nms": norm_max(impulse_difference[:, 3:]),
        "horizontal_world_force_rotation_effect_N": norm_max((identity["local_reaction_impulse"][:, :3]@R.T-identity["local_reaction_impulse"][:, :3])[:, :2])/float(identity["time_s"][1]),
    }
    checks = {key: measurements[key] <= c[key] for key in measurements if key in c}
    checks["nonzero_horizontal_rotation"] = measurements["horizontal_world_force_rotation_effect_N"] >= c["minimum_nonzero_horizontal_rotation_effect_N"]
    checks["identical_initial_points_masses_lattice"] = all(np.array_equal(identity[q], rotated[q])
        for q in ("reference_positions_local_m", "mass_kg", "lattice"))
    checks["identical_codes"] = np.array_equal(identity["codes"], rotated["codes"])
    for name, receipt in (("identity", identity_receipt), ("yaw_translated", rotated_receipt)):
        for key in ("pose_roundtrip_position_max_m", "pose_roundtrip_angle_max_rad", "world_state_roundtrip_position_max_m",
                    "world_state_roundtrip_velocity_max_m_s", "world_force_rotation_residual_max_Ns", "world_torque_rotation_residual_max_Nms"):
            checks[name+"_"+key] = receipt[key] <= c[key]
        checks[name+"_nonzero_force"] = receipt["peak_force_N"] >= c["minimum_peak_force_N"]
        checks[name+"_finite"] = receipt["finite"]
        checks[name+"_all_codes_zero"] = receipt["all_cut_codes_zero"]
    return {"measurements": measurements, "checks": checks, "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0"])
    args = parser.parse_args()
    output = new_output(args.output_dir)
    protocol_sha = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
    write_json(output/"protocol.json", PROTOCOL)
    write_json(output/"manifest.json", manifest({"protocol_sha256": protocol_sha, "device": args.device}, "task_frame_fixture_actual_kernel"))
    write_json(output/"gpu_before.json", command(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"]))
    implementation = output/"implementation"
    implementation.mkdir()
    for relative in ("scripts/probe_task_frames.py", "configs/cutting_tasks_v1.json", "src/robotic_cutting_sim/live_material.py",
                     "src/robotic_cutting_sim/fixture_contacts.py", "src/robotic_cutting_sim/tlmpm.py", "src/robotic_cutting_sim/tasks.py",
                     "src/robotic_cutting_sim/scene_frames.py", "src/robotic_cutting_sim/coupled_robot.py", "src/robotic_cutting_sim/material_config.py"):
        (implementation/Path(relative).name).write_bytes((ROOT/relative).read_bytes())
    write_json(output/"result.json", {"status": "NOT_RUN", "protocol_sha256": protocol_sha})
    start = time.perf_counter()
    try:
        identity_task = make_task(PROTOCOL["task"], seed=PROTOCOL["seed"])
        yaw_pose = tuple(PROTOCOL["frame_translation_m"])+tuple(Rotation.from_euler("z", PROTOCOL["frame_yaw_degrees"], degrees=True).as_quat())
        yaw_task = replace(identity_task, world_from_workpiece_xyzw=yaw_pose)
        identity, identity_receipt = run_segment(identity_task, args.device, output, "identity")
        print("identity segment recorded", flush=True)
        rotated, rotated_receipt = run_segment(yaw_task, args.device, output, "yaw_translated")
        print("yaw/translated segment recorded", flush=True)
        equivalence = compare(identity, rotated, rotated_receipt, identity_receipt)
        write_json(output/"frame_equivalence.json", equivalence)
        fixture = fixture_controls(identity_task, identity_receipt["fixture"], args.device)
        write_json(output/"fixture_kernel.json", fixture)
        tilt_pose = tuple(PROTOCOL["tilt_translation_m"])+tuple(Rotation.from_euler("xyz", PROTOCOL["tilt_xyz_degrees"], degrees=True).as_quat())
        tilt = tilted_receipt(replace(identity_task, world_from_workpiece_xyzw=tilt_pose), args.device, output)
        passed = equivalence["passed"] and fixture["passed"] and tilt["passed"]
        result = {"status": "PASS_FRAME_AND_FIXTURE_CONTROLS" if passed else "FAIL_FRAME_OR_FIXTURE_CONTROL",
                  "protocol_sha256": protocol_sha, "seed": PROTOCOL["seed"], "frame_equivalence": equivalence,
                  "fixture_kernel_passed": fixture["passed"], "tilted_frame_passed": tilt["passed"],
                  "wall_s": time.perf_counter()-start, "exclusive_timing": False,
                  "robot_rollout": False, "full_task_acceptance": False, "tilted_material_episode": False,
                  "cutting_enabled": False, "scope": "Actual LOCAL material dynamics and physical fixture impulses; no policy observation changes"}
        write_json(output/"result.json", result)
        print(json.dumps(result, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist(), indent=2), flush=True)
        return int(not passed)
    except Exception:
        write_json(output/"result.json", {"status": "ERROR_PRESERVED", "protocol_sha256": protocol_sha,
                   "wall_s": time.perf_counter()-start, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
