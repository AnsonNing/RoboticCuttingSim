#!/usr/bin/env python3
"""Audit recorded scene frames/edge paths and separately score cutting/flattening.

This is offline numeric validation. Runtime acceptance remains separate, even
when the caller identifies a recording as coming from an Isaac runtime.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from robotic_cutting_sim.benchmark import Bump, HeightFieldSpec
from robotic_cutting_sim.geometry import Specimen, sample_particles
from robotic_cutting_sim.recording import ROOT, write_json
from robotic_cutting_sim.scene_frames import (
    WorkpieceTarget, compose_pose, invert_transform, matrix_pose, pose_errors,
    pose_matrix, transform_points,
)
from robotic_cutting_sim.scene_validation import (
    flattening_checks, legacy_heightfield, material_task_metrics,
    mesh_heightfield_metrics, trajectory_metrics,
)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def without_debug(data):
    return {key: value for key, value in data.items() if not key.endswith("_debug")
            and key not in ("local_command_poses_xyzw", "local_actual_poses_xyzw", "position_error_m", "rotation_error_rad", "active_mask")}


def mount_audit(robot, setup, protocol):
    config = setup["robot"]["config"]
    mount = np.r_[config["tool_offset_m"], config["tool_rotation_xyzw"]]
    candidates = []
    # Old receipts omit body names. Infer the matching recorded body only as a
    # numerical diagnostic; named Panda-hand verification stays NOT_VERIFIED.
    for index in range(robot["body_pose"].shape[1]//2):
        try:
            expected = compose_pose(robot["body_pose"][:, index], mount)
            position, angle = pose_errors(expected, robot["tool_pose_xyzw"])
        except ValueError:
            continue
        candidates.append({"body_index": index, "max_translation_error_m": float(position.max()),
                           "max_rotation_error_rad": float(angle.max())})
    candidates.sort(key=lambda row: row["max_translation_error_m"])
    if not candidates:
        return {"status": "NOT_RUN", "reason": "No valid recorded candidate hand body"}, None
    best = candidates[0]
    limit = protocol["tests"]["ROB-01"]
    fits = [row for row in candidates if row["max_translation_error_m"] <= limit["max_fixed_mount_translation_error_m"]
            and row["max_rotation_error_rad"] <= limit["max_fixed_mount_rotation_error_rad"]]
    report = {"status": "PASS" if len(fits) == 1 else "FAIL", "hand_from_edge_pose_xyzw": mount,
              "best_recorded_body": best, "fitting_candidate_count": len(fits),
              "named_hand_identity_status": "NOT_VERIFIED in legacy recording; body names were not recorded",
              "scope": "unique fixed-mount consistency with measured body poses; not a named-link runtime proof"}
    return report, best["body_index"]


def mesh_audit(surface_dir, physics, task, target, heightfield, protocol, output):
    report = json.loads((surface_dir/"result.json").read_text())
    frame = max(report["frames"], key=lambda row: row["physics_timestamp_s"])
    if abs(frame["physics_timestamp_s"]-float(physics["time_s"][-1])) > 1e-6:
        raise ValueError("Final surface and final physical particle timestamp differ")
    path = surface_dir/"surface_debug"/f"{frame['source_frame']:04d}.npz"
    mesh = load_npz(path)
    if not np.array_equal(mesh["current_mechanical_component"], physics["component_ids"]):
        raise ValueError("Surface's current mechanical partition differs from recorded physics")
    selected = []
    for row in frame["component_ranges"]:
        if row["mechanical_component"] in task["reference_substrate_component_ids"]:
            first, count = row["first_triangle"], row["triangles"]
            selected.append(mesh["triangles"][first:first+count])
    if not selected:
        raise ValueError("No mesh triangles have an attributed reference-substrate component")
    metrics = mesh_heightfield_metrics(mesh["vertices_m"], np.vstack(selected), target, heightfield,
                                      protocol["tests"]["TASK-02"]["surface_grid_spacing_m"])
    np.savez_compressed(output/"surface_height_measurements.npz",
        measured_height_local_m=metrics["measured_heightfield_local_m_debug"],
        valid=metrics["valid_heightfield_debug"], target_plane_local_z_m=target.height_m)
    return {"metrics": without_debug(metrics), "threshold_assessment": flattening_checks(metrics, protocol),
            "source_mesh": str(path), "source_mesh_sha256": sha(path),
            "partition_rule": frame["partition_rule"], "future_component_ids_used": frame["future_component_ids_used"],
            "physics_timestamp_s": frame["physics_timestamp_s"],
            "selection_scope": "offline triangle statistics selected by current mechanical substrate provenance; no state or visibility changed"}


def cameras_audit(path, target):
    cameras = json.loads(path.read_text())
    result = {}
    point = target.points_to_world([0, 0, target.height_m])
    for name, camera in cameras.items():
        if not isinstance(camera, dict) or "T_world_camera" not in camera:
            continue
        transform = np.asarray(camera["T_world_camera"])
        toward = point-transform[:3, 3]
        optical = transform_points(invert_transform(transform), point)
        k = np.asarray(camera["K"])
        pixel = k@optical
        uv = pixel[:2]/pixel[2]
        result[name] = {"position_world_m": transform[:3, 3], "optical_forward_world": transform[:3, 2],
                        "target_forward_cosine": float(np.dot(transform[:3, 2], toward)/np.linalg.norm(toward)),
                        "target_optical_z_m": float(optical[2]), "target_pixel_uv": uv,
                        "target_inside_image": bool(optical[2] > 0 and 0 <= uv[0] < camera["width"] and 0 <= uv[1] < camera["height"]),
                        "scope": "recorded calibration geometry only; no native capture gate promoted"}
    return result


def plots(robot, trajectory, target, heightfield, output, surface=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time = robot["time_s"]
    planned, actual = trajectory["local_command_poses_xyzw"], trajectory["local_actual_poses_xyzw"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(planned[:, 0]*1000, planned[:, 1]*1000, label="commanded edge")
    axes[0, 0].plot(actual[:, 0]*1000, actual[:, 1]*1000, label="measured edge")
    for bump in heightfield.bumps:
        circle = plt.Circle((bump.x_m*1000, bump.y_m*1000), bump.radius_x_m*1000, fill=False, color="gray")
        axes[0, 0].add_patch(circle)
    axes[0, 0].set(xlabel="workpiece X [mm]", ylabel="workpiece Y [mm]", aspect="equal")
    axes[0, 1].plot(time, planned[:, 2]*1000, label="commanded edge")
    axes[0, 1].plot(time, actual[:, 2]*1000, label="measured edge")
    axes[0, 1].axhline(target.height_m*1000, color="gray", linestyle="--", label="local target plane")
    axes[0, 1].set(xlabel="simulation time [s]", ylabel="workpiece-local Z [mm]")
    axes[1, 0].plot(time, trajectory["position_error_m"]*1000)
    axes[1, 0].set(xlabel="simulation time [s]", ylabel="3D edge tracking error [mm]")
    axes[1, 1].plot(time, np.rad2deg(trajectory["rotation_error_rad"]))
    axes[1, 1].set(xlabel="simulation time [s]", ylabel="edge orientation error [degrees]")
    for ax in axes.flat:
        if ax in axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Recorded standalone reference — commanded vs measured knife edge, no Isaac runtime claim")
    fig.tight_layout(); fig.savefig(output/"planned_vs_actual_edge.png", dpi=140); plt.close(fig)
    if surface:
        with np.load(output/"surface_height_measurements.npz") as data:
            error = np.where(data["valid"], (data["measured_height_local_m"]-target.height_m)*1000, np.nan)
        fig, ax = plt.subplots(figsize=(7, 4))
        image = ax.imshow(error, origin="lower", extent=[-heightfield.length_m*500, heightfield.length_m*500,
                    -heightfield.width_m*500, heightfield.width_m*500], cmap="coolwarm", vmin=-2, vmax=2)
        ax.set(xlabel="workpiece X [mm]", ylabel="workpiece Y [mm]", title="Actual attributed substrate mesh — signed target error [mm]")
        fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(output/"substrate_surface_error.png", dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--surface-dir", type=Path)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/isaac_acceptance.json")
    parser.add_argument("--workpiece-pose-xyzw", nargs=7, type=float, default=[0, 0, 0, 0, 0, 0, 1])
    parser.add_argument("--runtime-source", choices=["standalone_reference", "live_isaac_lab", "live_isaac_sim_kit", "offline_replay"], default="standalone_reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    target = WorkpieceTarget(tuple(args.workpiece_pose_xyzw), protocol["workpiece"]["target_plane_local_z_m"],
                             protocol["workpiece"]["edge_clearance_above_target_m"])
    setup = json.loads((args.recording_dir/"setup.json").read_text())
    robot = load_npz(args.recording_dir/"robot_state.npz")
    physics = load_npz(args.recording_dir/"physics_state.npz")
    scenario = setup.get("benchmark_spec")
    if scenario:
        heightfield = HeightFieldSpec(**{**scenario["heightfield"], "bumps": tuple(Bump(**row) for row in scenario["heightfield"]["bumps"])})
    else:
        config = setup["config"]
        heightfield = legacy_heightfield(Specimen.generate(42, config["variant"], config["family"]))
    trajectory = trajectory_metrics(robot["time_s"], robot["desired_tool_pose_xyzw"], robot["tool_pose_xyzw"],
                                     target, heightfield, protocol, setup["material"]["blade_half_width"])
    mount, hand_index = mount_audit(robot, setup, protocol)
    if hand_index is not None:
        inverse_mount = matrix_pose(invert_transform(pose_matrix(mount["hand_from_edge_pose_xyzw"])))
        np.savez_compressed(output/"derived_hand_and_edge_trajectories.npz", time_s=robot["time_s"],
            measured_edge_world_xyzw=robot["tool_pose_xyzw"], commanded_edge_world_xyzw=robot["desired_tool_pose_xyzw"],
            measured_candidate_hand_world_xyzw=robot["body_pose"][:, hand_index],
            derived_commanded_hand_world_xyzw=compose_pose(robot["desired_tool_pose_xyzw"], inverse_mount),
            local_measured_edge_xyzw=trajectory["local_actual_poses_xyzw"])
    component_provenance = str(physics["component_provenance"].item()) if "component_provenance" in physics else ""
    task = material_task_metrics(physics["positions"][0], physics["positions"][-1], physics["mass_kg"],
                                physics["component_ids"], component_provenance, target, heightfield)
    surface = mesh_audit(args.surface_dir, physics, task, target, heightfield, protocol, output) if args.surface_dir and task["status"] == "MEASURED" else None
    task1 = (task.get("detached_initial_above_target_mass_fraction") is not None and
             task["detached_initial_above_target_mass_fraction"] >= protocol["tests"]["TASK-01"]["minimum_detached_initial_above_target_mass_fraction"])
    analytic = protocol["workpiece"]
    dims = {"length_m": heightfield.length_m, "width_m": heightfield.width_m,
            "base_height_m": heightfield.base_height_m}
    dim_tolerance = protocol["tests"]["SC-02"]["dimension_config_absolute_tolerance_m"]
    dimension_checks = {key: abs(value-analytic[key]) <= dim_tolerance for key, value in dims.items()}
    dimension_checks["one_central_dome"] = (len(heightfield.bumps) == 1 and heightfield.bumps[0].family == "dome"
        and abs(heightfield.bumps[0].x_m) <= dim_tolerance and abs(heightfield.bumps[0].y_m) <= dim_tolerance)
    if len(heightfield.bumps) == 1:
        bump = heightfield.bumps[0]
        dimension_checks["bump_height"] = abs(bump.height_m-analytic["bump_height_m"]) <= dim_tolerance
        dimension_checks["bump_radii"] = (abs(bump.radius_x_m-analytic["bump_radius_m"]) <= dim_tolerance
                                           and abs(bump.radius_y_m-analytic["bump_radius_m"]) <= dim_tolerance)
    sampler_error = None
    if all(dimension_checks.values()):
        expected_points, _, _ = sample_particles(Specimen(), setup["material"]["dx"]/2, setup["material"]["density"])
        if expected_points.shape == physics["positions"][0].shape:
            sampler_error = float(np.abs(target.points_to_local(physics["positions"][0])-expected_points).max())
    initial_codes_zero = "codes" in physics and not physics["codes"][0].any()
    dimensions = {"status": "PASS" if all(dimension_checks.values()) and sampler_error is not None and sampler_error <= 1e-5 and initial_codes_zero else "FAIL",
                  "checks": dimension_checks, "initial_codes_all_zero": bool(initial_codes_zero),
                  "recorded_initial_particles_vs_D0_sampler_max_abs_error_m": sampler_error,
                  "scope": "analytic D0 configuration and actual initial samples; runtime loading is separate"}
    result = {"scope": "offline_geometry_and_trajectory_audit", "runtime_source_declared": args.runtime_source,
              "protocol_id": protocol["protocol_id"], "protocol_sha256": sha(args.protocol),
              "RT-01": "NOT_RUN by this offline audit", "RT-02": "NOT_RUN by this offline audit",
              "target": {"workpiece_pose_world_xyzw": args.workpiece_pose_xyzw, "target_plane_local_z_m": target.height_m,
                         "commanded_cut_plane_local_z_m": target.edge_height_local_m, "plane_world_abcd": target.plane_world(),
                         "current_TLMPM_nonidentity_pose_supported": bool(np.allclose(target.T_world_workpiece, np.eye(4), atol=1e-9))},
              "analytic_material_spec": asdict(heightfield), "initial_particle_count": len(physics["mass_kg"]),
              "SC-02_dimension_sampling_assessment": dimensions,
              "initial_local_particle_bounds_m": [target.points_to_local(physics["positions"][0]).min(0), target.points_to_local(physics["positions"][0]).max(0)],
              "trajectory": without_debug(trajectory), "fixed_mount_diagnostic": mount,
              "material_task": without_debug(task), "TASK-01_threshold_assessment": "PASS" if task1 else "NOT_RUN" if task["status"] == "NOT_RUN" else "FAIL",
              "TASK-02_threshold_assessment": "NOT_RUN" if surface is None else surface["threshold_assessment"]["status"],
              "surface": surface, "camera_calibration_geometry": {} if not args.calibration_json else cameras_audit(args.calibration_json, target),
              "input_hashes": {name: sha(args.recording_dir/name) for name in ["robot_state.npz", "physics_state.npz", "setup.json"]},
              "validator_source_hashes": {"driver": sha(__file__),
                  "scene_frames": sha(ROOT/"src/robotic_cutting_sim/scene_frames.py"),
                  "scene_validation": sha(ROOT/"src/robotic_cutting_sim/scene_validation.py")},
              "isaac_integration_pass": False, "full_benchmark_ready": False}
    np.savez_compressed(output/"trajectory_errors.npz", time_s=robot["time_s"],
        position_error_m=trajectory["position_error_m"], rotation_error_rad=trajectory["rotation_error_rad"],
        active_mask=trajectory["active_mask"])
    plots(robot, trajectory, target, heightfield, output, surface)
    write_json(output/"result.json", result)
    print(json.dumps({"trajectory": result["trajectory"]["status"], "cut_removal": result["TASK-01_threshold_assessment"],
                      "flattening": result["TASK-02_threshold_assessment"], "isaac_integration_pass": False,
                      "result": str(output/"result.json")}))
    return 0 if result["trajectory"]["status"] == "PASS" and result["TASK-01_threshold_assessment"] == "PASS" and result["TASK-02_threshold_assessment"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
