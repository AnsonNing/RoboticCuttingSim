#!/usr/bin/env python3
"""Score actual attributed substrate against a previously sealed independent target.

Recompute physical stencil/removal gates and original analytic-plane v1 results.
Never overwrite the source episode, target, legacy scorer or formal observations.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.represented_target import compare_represented_columns, combine_with_physical_receipt
from robotic_cutting_sim.scene_frames import WorkpieceTarget
from robotic_cutting_sim.scene_validation import mesh_heightfield_metrics
from robotic_cutting_sim.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    output = new_output(args.output_dir)
    protocol_path, target_path = args.target/"protocol.json", args.target/"fixed_target.npz"
    protocol = json.loads(protocol_path.read_text())
    sealed = json.loads((args.target/"result.json").read_text())
    if (sealed["status"] != "PASS_FROZEN_GEOMETRY_CONTROLS" or digest(target_path) != sealed["target_file_sha256"]
            or digest(protocol_path) != sealed["protocol_sha256"]):
        raise ValueError("Target controls or immutable target/protocol hash validation failed")
    setup_path = args.episode/"setup.json"
    setup = json.loads(setup_path.read_text())
    task_path = args.episode/"task.json"
    actual_task = json.loads(task_path.read_text()) if task_path.exists() else setup["task"]
    task = replace(make_task("TRIM-01", seed=actual_task["seed"]),
        particle_spacing_m=protocol["target"]["particle_spacing_m"],
        world_from_workpiece_xyzw=tuple(actual_task["world_from_workpiece_xyzw"]))
    normalized = json.loads(json.dumps(asdict(task)))
    if any(normalized[key] != actual_task[key] for key in normalized):
        raise ValueError("Actual episode differs from the frozen coarse TRIM geometry/material/trajectory contract")
    if digest(ROOT/"configs/cutting_tasks_v1.json") != protocol["original_analytic_task_v1_sha256"]:
        raise ValueError("Original analytic Task v1 protocol changed")
    physics_path = args.episode/"physics_state.npz"
    if not physics_path.exists():
        physics_path = args.episode/"oracle_debug/physics_state.npz"
    with np.load(physics_path) as data:
        physics = {key: data[key] for key in data.files}
    count, labels = mechanical_components(physics["reference_stencil_nodes"], physics["reference_stencil_weights"],
        physics["codes"][-1], 1, gradients=physics["reference_stencil_gradients"])
    if "component_ids" in physics and not np.array_equal(labels, physics["component_ids"]):
        raise ValueError("Recorded final mechanical labels differ from the actual saved stencils")
    frame_manifest_path = args.episode/"frame_manifest.json"
    if frame_manifest_path.exists():
        frames = json.loads(frame_manifest_path.read_text())
        last = frames[-1]
        surface_path = args.episode/"surface_debug"/f"surface_{last['frame']:04d}.npz"
        info = last["current_component_surface"]
        with np.load(surface_path) as data:
            vertices, faces, surface_time = data["vertices_m"], data["triangles"], float(data["physics_timestamp_s"])
        if hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest() != last["surface"]["surface_sha256"]:
            raise ValueError("Recorded current mesh hash differs from capture receipt")
    else:
        surface_path = max((args.episode/"rgbd").glob("surface_*.npz"), key=lambda path: int(path.stem.split("_")[-1]))
        info = json.loads(surface_path.with_suffix(".json").read_text())
        with np.load(surface_path) as data:
            vertices, faces, surface_time = data["vertices"], data["triangles"], float(data["timestamp_s"])
    if info["surface_profile"] != "current_volume_v1":
        raise ValueError("Actual substrate does not use the frozen H2 profile")
    reference_local = task.points_to_local(physics["reference_positions"])
    anchored = reference_local[:, 2] <= reference_local[:, 2].min()+1e-7
    substrate_ids = np.unique(labels[anchored])
    face_labels = np.full(len(faces), -1, np.int32)
    for row in info["component_ranges"]:
        face_labels[row["first_triangle"]:row["first_triangle"]+row["triangles"]] = row["mechanical_component"]
    if np.any(face_labels < 0) or not np.isin(face_labels, labels).all():
        raise ValueError("Incomplete current mechanical triangle provenance")
    selected = np.isin(face_labels, substrate_ids)
    geometry_target = WorkpieceTarget(task.world_from_workpiece_xyzw, task.target_origin_local_m[2], task.edge_clearance_m)
    plane = mesh_heightfield_metrics(vertices, faces[selected], geometry_target, task.geometry, protocol["measurement"]["spacing_m"])
    height, valid = plane.pop("measured_heightfield_local_m_debug"), plane.pop("valid_heightfield_debug")
    with np.load(target_path) as target:
        shape, debug = compare_represented_columns(height, valid, target["height_local_m"], target["valid_mask"],
            spacing_xy_m=plane["measurement_spacing_xy_m"], initial_base_volume_m3=float(np.prod(protocol["target"]["dimensions_m"])),
            limits=protocol["shape_limits"])
    if height.size != protocol["measurement"]["columns"]:
        raise ValueError("Comparison did not retain the entire frozen footprint")
    evaluation = task.evaluate_oracle({**physics, "surface_vertices_world": vertices, "surface_triangles": faces,
        "surface_triangle_component_ids": face_labels, "surface_component_provenance": info["partition_rule"],
        "surface_timestamp_s": surface_time})
    combined = combine_with_physical_receipt(shape, evaluation, protocol["required_physical_checks_for_combined_goal"])
    np.savez_compressed(output/"column_comparison.npz", **debug, substrate_vertices_world=vertices,
        all_triangles=faces, actual_triangle_component_ids=face_labels, substrate_triangle_selection=selected)
    paths = [protocol_path, target_path, args.target/"result.json", setup_path, physics_path, surface_path,
             Path(__file__), ROOT/"src/robotic_cutting_sim/represented_target.py", ROOT/"src/robotic_cutting_sim/scene_validation.py",
             ROOT/"src/robotic_cutting_sim/tasks.py", ROOT/"src/robotic_cutting_sim/cut_topology.py"]
    if task_path.exists(): paths.append(task_path)
    if frame_manifest_path.exists(): paths.append(frame_manifest_path)
    else: paths.append(surface_path.with_suffix(".json"))
    hashes = {str(path): digest(path) for path in paths}
    source = output/"implementation"; source.mkdir()
    for path in [Path(__file__), ROOT/"src/robotic_cutting_sim/represented_target.py"]:
        (source/path.name).write_bytes(path.read_bytes())
    result = {"protocol_id": protocol["protocol_id"], "protocol_sha256": sealed["protocol_sha256"],
        "episode": str(args.episode), "fixed_target": str(target_path), "target_sha256": sealed["target_file_sha256"],
        "represented_shape": shape, "combined_goal": combined,
        "original_analytic_plane_v1": evaluation, "original_plane_metrics": plane,
        "actual_mechanical_components": count, "substrate_component_ids": substrate_ids,
        "substrate_triangles": int(selected.sum()), "other_component_triangles": int((~selected).sum()),
        "surface_timestamp_s": surface_time, "physics_timestamp_s": float(physics["time_s"][-1]),
        "input_sha256": hashes, "target_fitted_to_cut_data": False, "input_files_modified": False,
        "scope": "Separate full-column represented-solid task goal; original v1, runtime/cameras/numerical acceptance remain separate"}
    write_json(output/"result.json", result)
    print(json.dumps({"combined": combined["status"], "shape": shape["status"],
                      "original_v1": evaluation["status"], "metrics": shape["metrics"]}, indent=2), flush=True)
    return int(combined["status"] != "PASS_REPRESENTED_GOAL")


if __name__ == "__main__":
    raise SystemExit(main())
