#!/usr/bin/env python3
"""Check every recorded camera box against actual robot visual surfaces.

Intentional own-palm bracket attachment is counted separately. Zero sampled
surface intersections is not closed-volume or native-contact acceptance.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from robotic_cutting_sim.live_acceptance import load_npz, sha256
from robotic_cutting_sim.recording import ROOT, new_output, write_json
from robotic_cutting_sim.scene_frames import pose_matrix
from robotic_cutting_sim.wrist_camera_audit import obb_signed_projection_gap, transform_error
from robotic_cutting_sim.wrist_camera_clearance import (intentional_attachment,
    surface_owner, triangle_obb_intersections, clipped_triangle_vertices_in_box)

PROTOCOL = ROOT/"configs/wrist_camera_clearance_audit_v1.json"


def read(path):
    return json.loads(path.read_text())


def audit(directory, output, protocol):
    from pxr import Usd, UsdGeom
    from robotic_cutting_sim.usd_surface_bridge import stage_triangles
    from robotic_cutting_sim.visual_geometry import non_render_purpose_paths

    output.mkdir()
    setup, rig = read(directory/"setup.json"), read(directory/"camera_rig.json")
    robot = load_npz(directory/"robot_state.npz")
    body_paths = [str(path) for path in robot.get("body_prim_paths", setup.get("body_labels", []))]
    trace = read(directory/"camera_pose_trace.json")
    trace = trace["samples"] if isinstance(trace, dict) else trace
    if len(body_paths) != robot["body_pose"].shape[1]:
        raise ValueError("Actual prim paths are required for every saved robot body")
    stage = Usd.Stage.Open(str(directory/"final_live_scene.usda"))
    excluded = non_render_purpose_paths(stage)
    for spec in rig["specs"].values():
        excluded.extend(box["prim_path"] for box in spec.get("visual_obbs", []))
    for prim in stage.Traverse():
        if prim.GetName() in ("Knife", "LiveKnife"):
            excluded.append(str(prim.GetPath()))
    vertices, faces, _, geometry_metadata = stage_triangles(stage, excluded_prefixes=tuple(excluded))
    models, export_errors = {}, {}
    for index, path in enumerate(body_paths):
        prim = stage.GetPrimAtPath(path)
        world = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        export_errors[path] = transform_error(pose_matrix(robot["body_pose"][-1, index]), world)
        models[path] = {"index": index, "world": world, "triangles": [], "prims": []}
    for row in geometry_metadata["prims"]:
        owner = surface_owner(row["prim_path"], body_paths)
        if owner is None:
            continue
        model = models[owner]
        offset, count = row["first_triangle"], row["triangles"]
        triangles = vertices[faces[offset:offset+count]]
        triangles = (triangles-model["world"][:3, 3])@model["world"][:3, :3]
        model["triangles"].append(triangles)
        model["prims"].append({"path": row["prim_path"], "triangles": count})
    for path, model in models.items():
        if not model["triangles"]:
            raise ValueError("No actual visual triangles for measured body: "+path)
        model["triangles"] = np.concatenate(model["triangles"])
        low, high = model["triangles"].min((0, 1)), model["triangles"].max((0, 1))
        model["bounds"] = np.eye(4);model["bounds"][:3, 3] = (low+high)/2
        model["dimensions"] = np.maximum(high-low, 1e-12)
    pairs, samples, hit_records = {}, [], []
    for sample in trace:
        timestamp = float(sample["timestamp_s"])
        index = int(np.argmin(abs(robot["time_s"]-timestamp)))
        counts = {"unexcepted_surface_intersections": 0, "intentional_attachment_surface_intersections": 0}
        for name, spec in rig["specs"].items():
            camera = sample["cameras"][name]
            parent = spec.get("parent_hand_prim_path")
            for box in spec.get("visual_obbs", []):
                if "T_hand_obb" in box:
                    hand = camera.get("T_world_hand")
                    if hand is None:
                        hand = sample["T_world_hand"][spec["parent_hand_index"]]
                    box_world = np.asarray(hand)@box["T_hand_obb"]
                elif "T_optical_cv_obb" in box:
                    box_world = np.asarray(camera["T_world_camera"])@box["T_optical_cv_obb"]
                else:
                    box_world = np.asarray(box["T_world_obb"])
                for path, model in models.items():
                    body_world = pose_matrix(robot["body_pose"][index, model["index"]])
                    relative = np.linalg.inv(body_world)@box_world
                    gap = obb_signed_projection_gap(relative, box["dimensions_m"], model["bounds"], model["dimensions"])
                    hits = 0
                    if gap <= protocol["triangle_obb_numeric_tolerance_m"]:
                        hit_mask = triangle_obb_intersections(model["triangles"], relative, box["dimensions_m"],
                            protocol["triangle_obb_numeric_tolerance_m"])
                        hits = int(hit_mask.sum())
                    attached = intentional_attachment(box["name"], path, parent)
                    key = name+"|"+box["name"]+"|"+path
                    if key not in pairs:
                        pairs[key] = {"camera": name, "box": box["name"], "body_prim_path": path,
                            "intentional_parent_palm_attachment": attached, "samples": 0,
                            "minimum_bounding_box_signed_projection_gap_m": gap,
                            "maximum_surface_intersections_per_sample": 0, "samples_with_surface_intersections": 0}
                    pair = pairs[key];pair["samples"] += 1
                    pair["minimum_bounding_box_signed_projection_gap_m"] = min(pair["minimum_bounding_box_signed_projection_gap_m"], gap)
                    pair["maximum_surface_intersections_per_sample"] = max(pair["maximum_surface_intersections_per_sample"], hits)
                    pair["samples_with_surface_intersections"] += int(hits > 0)
                    if hits:
                        category = "intentional_attachment_surface_intersections" if attached else "unexcepted_surface_intersections"
                        counts[category] += hits
                        record = {"timestamp_s": timestamp, "camera": name, "box": box["name"], "body_prim_path": path,
                            "triangle_surface_intersections": hits, "intentional_parent_palm_attachment": attached}
                        points = clipped_triangle_vertices_in_box(model["triangles"][hit_mask], relative, box["dimensions_m"])
                        if len(points):
                            bounds = np.stack((points.min(0), points.max(0)))
                            record["clipped_visual_surface_bounds_box_local_m"] = bounds.tolist()
                            if "T_hand_obb" in box:
                                mount = np.asarray(box["T_hand_obb"])
                                hand_points = points@mount[:3, :3].T+mount[:3, 3]
                                hand_bounds = np.stack((hand_points.min(0), hand_points.max(0)))
                                record["clipped_visual_surface_bounds_hand_local_m"] = hand_bounds.tolist()
                                if "first_clipped_bounds_hand_local_m" not in pair:
                                    pair["first_clipped_bounds_hand_local_m"] = hand_bounds.tolist()
                                    pair["all_time_clipped_bounds_hand_local_m"] = hand_bounds.tolist()
                                    pair["maximum_clipped_bounds_change_from_first_m"] = 0.
                                all_time = np.asarray(pair["all_time_clipped_bounds_hand_local_m"])
                                pair["all_time_clipped_bounds_hand_local_m"] = np.stack((np.minimum(all_time[0], hand_bounds[0]),
                                    np.maximum(all_time[1], hand_bounds[1]))).tolist()
                                pair["maximum_clipped_bounds_change_from_first_m"] = max(pair["maximum_clipped_bounds_change_from_first_m"],
                                    float(np.max(abs(hand_bounds-np.asarray(pair["first_clipped_bounds_hand_local_m"])))))
                            if "T_hand_obb" in box and box["name"] == "Bracket" and np.allclose(box["dimensions_m"][:2], [.008, .008], atol=1e-12, rtol=0):
                                # The retained rig's _bracket_obb extends the rod
                                # by 4 mm past each nominal endpoint. These are
                                # descriptive geometry coordinates, not fitted
                                # or newly allowed attachment boundaries.
                                nominal_half_length = (box["dimensions_m"][2]-.008)/2
                                anchor = mount[:3, 3]-nominal_half_length*mount[:3, 2]
                                rear = mount[:3, 3]+nominal_half_length*mount[:3, 2]
                                distances = bounds[:, 2]+nominal_half_length
                                record["surface_axis_distance_from_nominal_anchor_toward_housing_m"] = distances.tolist()
                                pair["nominal_rod_endpoints_hand_m"] = {"anchor": anchor.tolist(), "housing_rear": rear.tolist(),
                                    "source": "Recorded bracket transform and full length; retained _bracket_obb adds 8mm total endpoint padding"}
                                previous = pair.get("all_time_surface_axis_distance_from_nominal_anchor_m", distances.tolist())
                                pair["all_time_surface_axis_distance_from_nominal_anchor_m"] = [min(previous[0], float(distances[0])), max(previous[1], float(distances[1]))]
                        hit_records.append(record)
        samples.append({"timestamp_s": timestamp, "robot_timestamp_alignment_error_s": float(abs(robot["time_s"][index]-timestamp)), **counts})
    maximum_unexcepted = max(row["unexcepted_surface_intersections"] for row in samples)
    checks = {"all_sampled_unexcepted_visual_surface_pairs_clear": maximum_unexcepted <= protocol["maximum_unexcepted_triangle_surface_intersections"],
        "final_export_matches_actual_recorded_bodies": all(error[0] <= protocol["maximum_final_export_body_translation_error_m"]
            and error[1] <= protocol["maximum_final_export_body_rotation_error_rad"] for error in export_errors.values()),
        "every_robot_time_has_a_pose_sample": len(np.unique([row["timestamp_s"] for row in samples])) >= len(robot["time_s"]),
        "samples_match_robot_timestamps": max(row["robot_timestamp_alignment_error_s"] for row in samples) <= 1e-6}
    duration = float(robot["time_s"][-1]-robot["time_s"][0])
    if duration >= 8.-1e-6:
        checks["at_least_401_samples_for_8s"] = len(samples) >= protocol["minimum_expected_robot_samples_for_8s"]
    paths = [directory/name for name in ("setup.json", "camera_rig.json", "robot_state.npz", "camera_pose_trace.json", "final_live_scene.usda")]
    result = {"episode": str(directory), "status": "PASS_SAMPLED_VISUAL_SURFACES" if all(checks.values()) else "FAIL_SAMPLED_VISUAL_SURFACES",
        "checks": checks, "protocol_sha256": sha256(PROTOCOL), "samples": len(samples), "robot_duration_s": duration, "robot_body_count": len(models),
        "maximum_unexcepted_triangle_surface_intersections_per_sample": maximum_unexcepted,
        "maximum_intentional_attachment_intersections_per_sample": max(row["intentional_attachment_surface_intersections"] for row in samples),
        "pair_summary": list(pairs.values()), "body_visual_sources": {path: model["prims"] for path, model in models.items()},
        "final_export_body_errors_translation_m_rotation_rad": export_errors,
        "scope": protocol["scope"], "surface_interpretation": protocol["surface_interpretation"],
        "native_contact_acceptance": "NOT_SCORED", "closed_volume_containment": "NOT_SCORED",
        "continuous_swept_collision": "NOT_SCORED", "exact_minimum_mesh_distance": "NOT_SCORED",
        "input_sha256": {str(path.relative_to(directory)): sha256(path) for path in paths}}
    write_json(output/"sample_records.json", samples)
    write_json(output/"intersection_records.json", hit_records)
    write_json(output/"result.json", result)
    print(directory.name, result["status"], "unexcepted max", maximum_unexcepted, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    paths = [Path(__file__).resolve(), PROTOCOL, ROOT/"src/robotic_cutting_sim/wrist_camera_clearance.py",
        ROOT/"src/robotic_cutting_sim/wrist_camera_audit.py", ROOT/"src/robotic_cutting_sim/scene_frames.py",
        ROOT/"src/robotic_cutting_sim/usd_surface_bridge.py", ROOT/"src/robotic_cutting_sim/visual_geometry.py"]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    snapshot = output/"analysis_sources";snapshot.mkdir()
    for path in paths:shutil.copyfile(path, snapshot/path.name)
    write_json(output/"analysis_start.json", {"source_sha256": hashes, "copied_before_analysis": True})
    cases = {directory.name: audit(directory.resolve(), output/directory.name, read(PROTOCOL)) for directory in args.episode}
    stable = all(sha256(ROOT/path) == digest for path, digest in hashes.items())
    write_json(output/"result.json", {"cases": cases, "analysis_sources_unchanged": stable, "source_sha256": hashes})
    if not stable:raise RuntimeError("Analysis bytes changed during the audit; receipt preserved")


if __name__ == "__main__":
    main()
