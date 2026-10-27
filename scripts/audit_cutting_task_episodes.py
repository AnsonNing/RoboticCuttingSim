#!/usr/bin/env python3
"""Independently score saved live CUT/TRIM episodes, without running a solver.

Task scores, tracking screens, contact screens and missing paired controls remain
separate. Oracle graphs/forces are analysis data, never policy observations.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.live_acceptance import (cut_history_metrics, graph_audit,
    load_npz, resolve_recorded_source, sha256)
from robotic_cutting_sim.recording import ROOT, new_output, write_json
from robotic_cutting_sim.scene_frames import invert_transform, pose_errors, transform_poses
from robotic_cutting_sim.tasks import PROTOCOL_PATH, make_task


def tracking_metrics(time, measured, commanded, active, task):
    """Compare measured edge with the actual issued command at every timestamp."""
    time, measured, commanded, active = map(np.asarray, (time, measured, commanded, active))
    if (measured.shape != (len(time), 7) or commanded.shape != measured.shape
            or active.shape != time.shape or active.dtype.kind != "b"
            or len(time) < 2 or np.any(np.diff(time) <= 0)):
        raise ValueError("Tracking requires aligned monotonic timestamps, poses and Boolean active flags")
    position, angle = pose_errors(commanded, measured)
    local_actual = transform_poses(invert_transform(task.T_world_workpiece), measured)
    local_command = transform_poses(invert_transform(task.T_world_workpiece), commanded)
    expected = np.stack([task.reference_edge_pose_world(float(t)) for t in time])
    schedule_position, schedule_angle = pose_errors(expected, commanded)
    rows = {}
    for name, selected in (("all", np.ones(len(time), bool)), ("active", active)):
        rows[name] = None if not selected.any() else {
            "samples": int(selected.sum()),
            "maximum_edge_position_error_m": float(position[selected].max()),
            "rms_edge_position_error_m": float(np.sqrt(np.mean(position[selected]**2))),
            "maximum_edge_local_z_error_m": float(np.abs(local_actual[selected, 2]-local_command[selected, 2]).max()),
            "maximum_edge_rotation_error_deg": float(np.rad2deg(angle[selected]).max())}
    rows.update(maximum_measured_translation_per_control_step_m=float(np.linalg.norm(np.diff(measured[:, :3], axis=0), axis=1).max()),
        commanded_vs_task_schedule_max_position_m=float(schedule_position.max()),
        commanded_vs_task_schedule_max_rotation_deg=float(np.rad2deg(schedule_angle).max()),
        active_source="Recorded actually armed material intervals, aligned to their ending robot sample",
        maximum_measured_local_target_plane_signed_distance_active_m=None if not active.any() else
            float(np.abs((local_actual[active, :3]-task.target_origin_local_m)@np.asarray(task.target_normal_local)).max()))
    return rows, {"time_s": time, "measured_local_xyzw": local_actual, "commanded_local_xyzw": local_command,
                  "position_error_m": position, "rotation_error_deg": np.rad2deg(angle), "active": active}


def component_motion(physics, labels, withdrawal_s):
    """Track reference-mass COMs of final physical components, not render masks."""
    time, positions, mass = [np.asarray(physics[k]) for k in ("time_s", "positions", "mass_kg")]
    post = np.flatnonzero(time >= withdrawal_s-1e-6)
    if len(post) < 2:
        return {"status": "NOT_RUN", "reason": "At least two physics samples after completed withdrawal are required"}, {}
    rows, arrays = [], {"time_s": time[post]}
    for label in np.unique(labels):
        selected = labels == label
        weight = mass[selected]
        com = np.einsum("tnc,n->tc", positions[post][:, selected], weight)/weight.sum()
        offset = com-com[0]
        rows.append({"component_id": int(label), "mass_kg": float(weight.sum()), "particles": int(selected.sum()),
            "com_world_at_withdrawal_m": com[0].tolist(), "com_world_final_m": com[-1].tolist(),
            "final_com_displacement_m": float(np.linalg.norm(offset[-1])),
            "maximum_com_excursion_m": float(np.linalg.norm(offset, axis=1).max())})
        arrays[f"component_{label}_com_world_m"] = com
    return {"status": "MEASURED", "withdrawal_complete_s": float(withdrawal_s),
        "sample_start_s": float(time[post[0]]), "sample_end_s": float(time[post[-1]]),
        "sampling_dt_max_s": float(np.diff(time[post]).max()), "components": rows,
        "scope": "Final graph memberships used only retrospectively for COMs; no minimum-motion task requirement or cross-component recontact claim"}, arrays


def reaction_metrics(diagnostics, robot):
    next_wrench = np.asarray([r["next_material_wrench_world"] for r in diagnostics], float)
    applied = np.asarray([r["applied_previous_material_wrench_world"] for r in diagnostics], float)
    impulses = np.asarray([r["current_material_reaction_impulse_world"] for r in diagnostics], float)
    stats = np.asarray([r["material_contact_statistics"] for r in diagnostics], float)
    time = np.asarray(robot["time_s"])
    if len(diagnostics) != len(time)-1 or stats.shape != (len(diagnostics), 7):
        raise ValueError("Reaction ledger must align to every measured control interval")
    dt = np.diff(time)
    measured = np.asarray(robot["tool_pose_xyzw"])
    delta_rotation = (Rotation.from_quat(measured[1:, 3:])*Rotation.from_quat(measured[:-1, 3:]).inv()).as_rotvec()
    reconstructed_robot_work = float(np.sum(applied[:, :3]*np.diff(measured[:, :3], axis=0))
                                     +np.sum(applied[:, 3:]*delta_rotation))
    actual_work_recorded = all("applied_tool_wrench_work_j" in r for r in diagnostics)
    robot_work = (float(sum(r["applied_tool_wrench_work_j"] for r in diagnostics))
                  if actual_work_recorded else reconstructed_robot_work)
    material_work = float(stats[:, 2].sum())
    return {"all_wrenches_and_impulses_finite": bool(all(np.isfinite(a).all() for a in (next_wrench, applied, impulses, stats))),
        "tool_contact_particle_substep_events": int(stats[:, 1].sum()),
        "control_intervals_with_tool_contacts": int(np.count_nonzero(stats[:, 1])),
        "peak_next_reaction_force_N": float(np.linalg.norm(next_wrench[:, :3], axis=1).max()),
        "peak_next_reaction_torque_Nm": float(np.linalg.norm(next_wrench[:, 3:], axis=1).max()),
        "sum_reaction_impulse_world_Ns_Nms": impulses.sum(0).tolist(),
        "sum_reaction_linear_impulse_norm_Ns": float(np.linalg.norm(impulses[:, :3], axis=1).sum()),
        "reaction_force_times_dt_impulse_max_abs_error": float(np.abs(next_wrench*dt[:, None]-impulses).max()),
        "previous_interval_feedback_carry_max_abs_error": float(np.abs(applied[1:]-next_wrench[:-1]).max()),
        "initial_applied_feedback_max_abs": float(np.abs(applied[0]).max()),
        "material_boundary_work_J": material_work, "robot_applied_material_wrench_work_J": robot_work,
        "robot_work_source": "Recorded native applied-wrench work ledger" if actual_work_recorded else "Recorded actual wrench and measured world edge increments",
        "robot_boundary_reconstruction_work_J": reconstructed_robot_work,
        "interface_work_residual_fraction": None if abs(material_work) < 1e-12 else abs(material_work+robot_work)/abs(material_work),
        "maximum_material_fixture_penetration_m": float(stats[:, 0].max()),
        "maximum_tool_material_penetration_m": float(stats[:, 3].max()),
        "maximum_material_pair_overlap_m": float(stats[:, 6].max()),
        "material_pair_events_unattributed": int(stats[:, 4].sum()),
        "material_pair_accumulated_impulse_norm_Ns_unattributed": float(stats[:, 5].sum()),
        "contact_scope": "Actual retained MPM event counters; pair counters include within-component events and do not prove recontact. Work is a partitioned interface diagnostic, not total energy convergence."}


def interval_state_metrics(records):
    """Retain missing interval instrumentation; never turn it into an infinite pass."""
    determinant = [row.get("min_detF", row.get("minimum_detF")) for row in records]
    speed = [row.get("max_speed_m_s", row.get("maximum_material_speed_m_s")) for row in records]
    complete = lambda values: bool(values) and all(value is not None and np.isfinite(value) for value in values)
    return {"minimum_detF": float(min(determinant)) if complete(determinant) else None,
            "maximum_speed_m_s": float(max(speed)) if complete(speed) else None}


def load_surface(directory, final):
    native = (directory/"frame_manifest.json").is_file()
    if native:
        frames = json.loads((directory/"frame_manifest.json").read_text())
        row = frames[-1 if final else 0]
        path = directory/"surface_debug"/f"surface_{row['frame']:04d}.npz"
        metadata = row["current_component_surface"]
        raw = load_npz(path)
        vertices, triangles, timestamp = raw["vertices_m"], raw["triangles"], raw["physics_timestamp_s"]
        metadata_path = directory/"frame_manifest.json"
    else:
        paths = sorted((directory/"rgbd").glob("surface_*.npz"))
        path = paths[-1 if final else 0]
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        raw = load_npz(path)
        vertices, triangles, timestamp = raw["vertices"], raw["triangles"], raw["timestamp_s"]
    labels = np.full(len(triangles), -1, np.int32)
    for row in metadata["component_ranges"]:
        start = row["first_triangle"]
        labels[start:start+row["triangles"]] = row["mechanical_component"]
    if (labels < 0).any():
        raise ValueError("Current surface component ranges do not cover all actual triangles")
    return {"surface_vertices_world": vertices, "surface_triangles": triangles,
        "surface_triangle_component_ids": labels, "surface_component_provenance": metadata["partition_rule"],
        "surface_timestamp_s": float(timestamp)}, metadata, [path, metadata_path]


def joint_metrics(directory, robot):
    q, qdot = robot["q"], robot["qdot"]
    rows = {"all_joint_positions_and_velocities_finite": bool(np.isfinite(q).all() and np.isfinite(qdot).all()),
        "actual_q_min_rad": q.min(0).tolist(), "actual_q_max_rad": q.max(0).tolist(),
        "maximum_joint_speed_rad_s_by_arm_joint": np.abs(qdot).max(0).tolist(),
        "maximum_joint_excursion_from_initial_rad_by_arm_joint": np.abs(q-q[0]).max(0).tolist(),
        "parked_second_arm_max_drift_from_initial_rad": float(np.abs(q[:, 1]-q[0, 1]).max()),
        "parked_second_arm_max_speed_rad_s": float(np.abs(qdot[:, 1]).max()),
        "parked_scope": "Measured throughout episode; no new acceptance threshold is fitted to these outcomes"}
    scene = next(iter(sorted(directory.glob("initial_workcell_00/workcell.usda"))), None)
    if scene is None:
        scene = directory/"workcell"/"workcell.usda"
    try:
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(scene))
        bounds = {}
        for prim in stage.Traverse():
            if prim.GetName() in [f"panda_joint{i}" for i in range(1, 8)] and prim.IsA(UsdPhysics.RevoluteJoint):
                joint = UsdPhysics.RevoluteJoint(prim)
                bounds[prim.GetName()] = np.deg2rad([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])
        ordered = np.stack([bounds[f"panda_joint{i}"] for i in range(1, 8)])
        maximum_excess = float(max(0, np.max(ordered[:, 0]-q), np.max(q-ordered[:, 1])))
        rows["asset_joint_limit_check"] = {"status": "PASS" if maximum_excess == 0 else "FAIL",
            "minimum_lower_margin_rad": float(np.min(q-ordered[:, 0])),
            "minimum_upper_margin_rad": float(np.min(ordered[:, 1]-q)), "maximum_excess_rad": maximum_excess,
            "bounds_rad": ordered.tolist(), "scene": str(scene), "scene_sha256": sha256(scene),
            "scope": "Composed official USD declared arm bounds; runtime articulation limit tensors were not separately recorded"}
    except Exception as error:
        rows["asset_joint_limit_check"] = {"status": "NOT_RUN", "reason": str(error),
            "scope": "Runtime joint-limit tensors and a readable composed USD are unavailable"}
    return rows


def plots(output, name, task, trace, surfaces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    output.mkdir(exist_ok=True)
    time = trace["time_s"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, axis in zip(axes[:2], [0, 2]):
        ax.plot(time, trace["commanded_local_xyzw"][:, axis]*1000, label="issued command")
        ax.plot(time, trace["measured_local_xyzw"][:, axis]*1000, label="measured native edge", lw=1)
        ax.set_ylabel(f"local {'xz'[axis//2]} (mm)")
    axes[2].plot(time, trace["position_error_m"]*1000)
    axes[2].set_ylabel("position error (mm)")
    axes[3].plot(time, trace["rotation_error_deg"])
    axes[3].set_ylabel("angle error (deg)")
    axes[3].set_xlabel("recorded simulation time (s)")
    for ax in axes:
        ax.axvspan(*task.active_cut_interval_s, alpha=.12, color="orange")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle(name+" — command and actual edge, workpiece-local coordinates")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output/("trajectory."+extension), dpi=150)
    plt.close(fig)
    fig = plt.figure(figsize=(13, 5))
    all_vertices = np.concatenate([task.points_to_local(s["surface_vertices_world"]) for s in surfaces])
    low, high = all_vertices.min(0)*1000, all_vertices.max(0)*1000
    for i, surface in enumerate(surfaces):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        vertices = task.points_to_local(surface["surface_vertices_world"])*1000
        triangles = surface["surface_triangles"]
        labels = surface["surface_triangle_component_ids"]
        # Display stride is explicit; every saved triangle remains in scoring.
        indices = np.arange(0, len(triangles), max(1, len(triangles)//25000))
        colors = plt.cm.tab10(labels[indices] % 10)
        ax.add_collection3d(Poly3DCollection(vertices[triangles[indices]], facecolors=colors, linewidth=0, alpha=.8))
        ax.set(xlim=(low[0]-2, high[0]+2), ylim=(low[1]-2, high[1]+2), zlim=(min(-5, low[2]-2), high[2]+2),
               xlabel="local x (mm)", ylabel="local y (mm)", zlabel="local z (mm)",
               title=f"Actual mesh t={surface['surface_timestamp_s']:g}s; current component colors")
        ax.set_box_aspect(np.maximum(high-low, 1))
        ax.view_init(elev=24, azim=-62)
    fig.suptitle(name+" — recorded current surface; at most 25k faces displayed, full mesh scored")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output/("material_before_after."+extension), dpi=150)
    plt.close(fig)


def audit(directory, output):
    directory = directory.resolve()
    output.mkdir()
    inputs = [directory/f for f in ("manifest.json", "setup.json", "result.json", "robot_state.npz")]
    manifest, setup, result = [json.loads(path.read_text()) for path in inputs[:3]]
    task_json = directory/"task.json"
    recorded_task = json.loads(task_json.read_text()) if task_json.is_file() else setup["task"]
    if task_json.is_file():
        inputs.append(task_json)
    input_pose = recorded_task.get("randomization", {}).get("input_world_from_workpiece_xyzw", recorded_task["world_from_workpiece_xyzw"])
    task = make_task(recorded_task["task_id"], recorded_task["seed"], recorded_task["variant"],
                     world_from_workpiece_xyzw=input_pose,
                     particle_spacing_m=recorded_task["particle_spacing_m"])
    # Refuse to score an episode using a different current recipe or threshold.
    generated = json.loads(json.dumps(asdict(task)))
    recorded = {k: recorded_task[k] for k in generated}
    # Reversing a composed seeded offset introduces only floating-point roundoff;
    # compare the rigid transform, while requiring all other recipe fields exact.
    pose_error = pose_errors(generated.pop("world_from_workpiece_xyzw"), recorded.pop("world_from_workpiece_xyzw"))
    if (generated != recorded or max(pose_error) > 1e-12
            or recorded_task["protocol_sha256"] != sha256(PROTOCOL_PATH)):
        raise ValueError("Recorded task/thresholds do not match this auditor's frozen TaskSpec")
    if task.variant == "randomized":
        from robotic_cutting_sim.task_randomization import RANDOMIZATION_PROTOCOL_PATH
        if recorded_task["randomization"]["protocol_sha256"] != sha256(RANDOMIZATION_PROTOCOL_PATH):
            raise ValueError("Recorded randomization protocol does not match this auditor")
    oracle = directory/"oracle_debug" if (directory/"oracle_debug").is_dir() else directory
    physics_path, diagnostic_path = oracle/"physics_state.npz", oracle/"diagnostics.json"
    inputs += [physics_path, diagnostic_path]
    physics = load_npz(physics_path)
    robot = load_npz(directory/"robot_state.npz")
    diagnostics = json.loads(diagnostic_path.read_text())
    initial_surface, initial_metadata, initial_paths = load_surface(directory, False)
    final_surface, final_metadata, final_paths = load_surface(directory, True)
    inputs += initial_paths+final_paths
    graph, labels = graph_audit(physics, 1)
    if labels is None:
        raise ValueError("Actual reference stencils including gradients are required")
    stencil_hash = graph["stencil_sha256"]
    final_codes_hash = hashlib.sha256(physics["codes"][-1].tobytes()).hexdigest()
    surface_consistent = (final_metadata["stencil_sha256"] == stencil_hash
        and final_metadata["codes_sha256"] == final_codes_hash
        and not final_metadata["future_component_ids_used"]
        and not final_metadata["renderer_partition_changes_physics"])
    if not surface_consistent:
        raise ValueError("Final surface does not bind the actual recorded current mechanical graph")
    recomputed = task.evaluate_oracle({**physics, **final_surface})
    score_path = oracle/"task_evaluation.json"
    original_score = json.loads(score_path.read_text())
    inputs.append(score_path)
    score_checks_agree = recomputed["checks"] == original_score["checks"]
    score_metrics_agree = json.loads(json.dumps(recomputed["metrics"])) == original_score["metrics"]
    active = np.r_[False, [row["active_cut"] for row in diagnostics]].astype(bool)
    tracking, trace = tracking_metrics(robot["time_s"], robot["tool_pose_xyzw"],
        robot["desired_tool_pose_xyzw"], active, task)
    np.savez_compressed(output/"trajectory_errors.npz", **trace)
    motion, motion_arrays = component_motion(physics, labels, task.edge_position_knots_local_txyz[-2][0])
    np.savez_compressed(output/"postwithdraw_component_com.npz", **motion_arrays)
    reaction = reaction_metrics(diagnostics, robot)
    finite = all(np.isfinite(physics[key]).all() for key in ("positions", "velocities", "final_F", "final_Fpi"))
    maximum_speed = max(float(np.linalg.norm(frame, axis=1).max()) for frame in physics["velocities"])
    interval_path = directory/"trace.json"
    interval_records = json.loads(interval_path.read_text()) if interval_path.is_file() else diagnostics
    if interval_path.is_file():
        inputs.append(interval_path)
    interval = interval_state_metrics(interval_records)
    minimum_det, maximum_diagnostic_speed = interval["minimum_detF"], interval["maximum_speed_m_s"]
    native_rigid = [row["native_rigid_contact_audit"] for row in diagnostics if "native_rigid_contact_audit" in row]
    rigid_pen = None if not native_rigid else max(row["max_non_tool_penetration_m"] for row in native_rigid)
    safety = json.loads((ROOT/"configs/isaac_acceptance.json").read_text())["tests"]["ENV-02"]
    numerical_checks = {"finite_saved_material_and_reaction": finite and reaction["all_wrenches_and_impulses_finite"],
        "maximum_saved_material_speed": maximum_speed <= safety["maximum_material_speed_m_s"],
        "tool_material_penetration": reaction["maximum_tool_material_penetration_m"] <= safety["maximum_tool_material_penetration_m"],
        "material_overlap": reaction["maximum_material_pair_overlap_m"] <= safety["maximum_material_overlap_m"]}
    if minimum_det is not None:
        numerical_checks["minimum_detF"] = minimum_det >= safety["minimum_detF"]
    if maximum_diagnostic_speed is not None:
        numerical_checks["maximum_control_interval_material_speed"] = maximum_diagnostic_speed <= safety["maximum_material_speed_m_s"]
    if rigid_pen is not None:
        numerical_checks["native_non_tool_rigid_penetration"] = rigid_pen <= safety["maximum_non_tool_rigid_penetration_m"]
    contacts = {"status": "FAIL_SCREEN" if not all(numerical_checks.values()) else "PASS_AVAILABLE_SCREENS",
        "checks": numerical_checks, "minimum_detF_all_control_intervals": minimum_det,
        "minimum_final_particle_detF": float(np.linalg.det(physics["final_F"]).min()),
        "maximum_saved_material_speed_m_s": maximum_speed,
        "maximum_control_interval_material_speed_m_s": maximum_diagnostic_speed,
        "interval_state_sampling_complete": minimum_det is not None and maximum_diagnostic_speed is not None,
        "native_non_tool_rigid_penetration_m": rigid_pen,
        "native_rigid_contact_coverage": "All recorded native substeps plus boundaries" if native_rigid else "NOT_RUN; native rigid penetration samples absent",
        "threshold_source": "Unchanged broad ENV-02 engineering screens; not promotion of the complete 18-gate protocol",
        "missing_controls": ["Matched settle/elastic material control", "Matched feedback ON/OFF episode", "Full same-source task repetition", "Time/grid convergence and calibrated material fracture"],
        "reaction": reaction}
    sources = {path: resolve_recorded_source(directory, path, value) for path, value in manifest["implementation_sha256"].items()}
    plots(output, directory.name, task, trace, [initial_surface, final_surface])
    evidence = {"recording": str(directory), "task_id": task.task_id, "seed": task.seed,
        "scope": "Independent offline recorded-state audit; no simulation execution or policy observation modification",
        "task_numerical_parameters": {"particle_spacing_m": task.particle_spacing_m, "grid_dx_m": task.grid_dx_m,
            "recorded_numerical_overrides": recorded_task.get("numerical_overrides", {}),
            "frozen_default_particle_spacing_m": json.loads(PROTOCOL_PATH.read_text())[task.task_id]["particle_spacing_m"],
            "scope": "Explicit numerical overrides do not establish acceptance of the unchanged fixed-default resolution"},
        "execution": {"recorded_status": result["status"], "control_steps": len(robot["time_s"])-1,
            "robot_duration_s": float(robot["time_s"][-1]-robot["time_s"][0]),
            "material_duration_s": float(physics["time_s"][-1]-physics["time_s"][0]),
            "material_frames": len(physics["time_s"]), "material_sampling_max_dt_s": float(np.diff(physics["time_s"]).max()),
            "wall_s": result.get("wall_s"), "runtime_backend": manifest["backend"]},
        "task_score": recomputed, "recorded_score_recomputation": {"same_checks": score_checks_agree,
            "same_metrics": score_metrics_agree, "same_status": recomputed["status"] == original_score["status"]},
        "mechanical_graph": graph, "history": cut_history_metrics(physics["codes"], 1),
        "surface_graph_hashes_match_current_recorded_state": surface_consistent,
        "tracking": tracking, "joints": joint_metrics(directory, robot), "postwithdraw_motion": motion,
        "contact_and_numerical_screens": contacts, "source_verification": sources,
        "all_manifest_implementation_bytes_resolved": all(row["verified"] for row in sources.values()),
        "protocol_sha256": sha256(PROTOCOL_PATH), "inputs_sha256": {str(path.relative_to(directory)): sha256(path) for path in set(inputs)},
        "full_18_gate_acceptance": "NOT_SCORED", "full_benchmark_ready": False,
        "limitations": ["Saved-frame topology persistence is measured at the recorder cadence, not every material substep",
            "Component motion is descriptive; preserved topology is not a surface-flatness or recontact proof",
            "Task success is separate from native RGB-D, randomized variants, convergence and repeatability acceptance"]}
    write_json(output/"result.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    analysis_paths = [Path(__file__).resolve(), PROTOCOL_PATH, ROOT/"configs/isaac_acceptance.json",
        ROOT/"configs/task_randomization_v1.json"]
    analysis_paths += [ROOT/"src/robotic_cutting_sim"/name for name in
        ("tasks.py", "task_randomization.py", "live_acceptance.py", "cut_topology.py", "scene_validation.py", "scene_frames.py", "benchmark.py", "recording.py")]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in analysis_paths}
    snapshots = output/"analysis_sources"
    snapshots.mkdir()
    for path in analysis_paths:
        shutil.copyfile(path, snapshots/path.name)
    write_json(output/"analysis_start.json", {"source_sha256": before, "copied_before_analysis": True,
        "recordings": [str(path.resolve()) for path in args.recording_dir]})
    cases = {}
    for directory in args.recording_dir:
        cases[directory.name] = audit(directory, output/directory.name)
        print(directory.name, cases[directory.name]["task_score"]["status"], flush=True)
    unchanged = all(sha256(ROOT/path) == value for path, value in before.items())
    write_json(output/"result.json", {"cases": cases, "analysis_source_sha256": before,
        "analysis_sources_unchanged_during_analysis": unchanged, "full_benchmark_ready": False})
    if not unchanged:
        raise RuntimeError("Analysis implementation changed during execution; receipt retained")


if __name__ == "__main__":
    main()
