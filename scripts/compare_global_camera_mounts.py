#!/usr/bin/env python3
"""Offline full-stage geometry prediction for three fixed Global camera mounts.

No simulator or native camera is advanced. Actual pilot mesh/body states and
runtime K are reused; CPU first hits are predictions, not new RTX/D455 evidence.
Candidates are declared before raycasting; no native noise/holes are synthesized.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from robotic_cutting_sim.camera_rig import pose_matrix
from robotic_cutting_sim.recording import ROOT, new_output, write_json
from robotic_cutting_sim.usd_surface_bridge import stage_triangles


CANDIDATES = {
    "frozen_v1": {"eye_world_m": [1.65, -1.75, 1.50], "target_world_m": [0., 0., .40]},
    "closer_overview_a": {"eye_world_m": [1.10, -1.20, 1.05], "target_world_m": [0., 0., .25]},
    "closer_overview_b": {"eye_world_m": [1.20, -1.30, 1.10], "target_world_m": [0., 0., .30]},
}


def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    transform[:3, 3] = eye
    return transform


def projection_metrics(points, transform, k, width, height):
    local = (points-transform[:3, 3]) @ transform[:3, :3]
    z = local[:, 2]
    uv = (local @ k.T)[:, :2]/z[:, None]
    in_frame = (z > .001) & (z < 20) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    in_depth = in_frame & (z >= .6) & (z <= 6.)
    return {"mesh_vertex_count": len(points), "world_bbox_min_m": points.min(0).tolist(),
            "world_bbox_max_m": points.max(0).tolist(), "projected_uv_min_px": uv.min(0).tolist(),
            "projected_uv_max_px": uv.max(0).tolist(), "optical_z_min_max_m": [float(z.min()), float(z.max())],
            "vertices_inside_rgb_frame_fraction": float(in_frame.mean()),
            "vertices_inside_public_depth_frame_fraction": float(in_depth.mean()),
            "all_vertices_inside_rgb_frame": bool(in_frame.all()), "all_vertices_inside_public_depth_frame": bool(in_depth.all())}


def _object_group(path):
    if "WristCameraRig/GlobalD455" in path:
        return "own_global_camera_housing"
    if path.startswith("/World/Franka_0"):
        return "knife" if path.endswith(("/Knife", "/LiveKnife")) else "franka_0_including_wrist_rig"
    if path.startswith("/World/Franka_1"):
        return "franka_1_including_wrist_rig"
    if "ContinuousSpecimen" in path:
        return "material"
    if path == "/World/Table":
        return "table"
    if path.startswith("/World/TaskFixture"):
        return "fixture"
    return "other_rendered_geometry"


def main():
    from pxr import Gf, Usd, UsdGeom
    import warp as wp
    from PIL import Image
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 2])
    args = p.parse_args()
    episode, output = args.episode.resolve(), new_output(args.output_dir)
    protocol = {"scope": __doc__, "candidates": CANDIDATES, "frames": args.frames,
        "camera_source": "actual recorded global_d455 runtime K/resolution; no intrinsic fitting",
        "geometric_screening_criteria": {"full_table_and_both_robots_inside_rgb": True,
            "minimum_public_optical_depth_m": .6, "minimum_predicted_knife_pixels": 8,
            "minimum_geometric_depth_valid_fraction": .10},
        "native_acceptance_boundary": "Native depth validity/registration/noise and actual knife pixels require a fresh runtime capture; no native PASS follows from this offline screen"}
    write_json(output/"protocol.json", protocol)
    module_path = ROOT/"scripts/audit_wrist_camera_rig.py"
    spec = importlib.util.spec_from_file_location("offline_global_scene_reference", module_path)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module;spec.loader.exec_module(module)
    setup = json.loads((episode/"setup.json").read_text())
    rig = json.loads((episode/"camera_rig.json").read_text())["specs"]
    with np.load(episode/"robot_state.npz") as source:
        robot = {key: source[key] for key in source.files}
    frames = {f["frame"]: f for f in json.loads((episode/"frame_manifest.json").read_text())}
    native_calibration = frames[args.frames[0]]["camera_calibration"]["global_d455"]
    k = np.asarray(native_calibration["K"], float)
    width, height = native_calibration["width"], native_calibration["height"]
    wp.init()
    scene = module.SceneReference(episode, robot, setup, True)
    global_root = scene.stage.GetPrimAtPath(rig["global_d455"]["prim_path"])
    source_root = np.asarray(UsdGeom.Xformable(global_root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    source_camera = np.asarray(rig["global_d455"]["T_world_camera"])
    camera_in_root = np.linalg.inv(source_root) @ source_camera
    input_paths = [Path(__file__), module_path, episode/"setup.json", episode/"camera_rig.json",
                   episode/"robot_state.npz", episode/"frame_manifest.json", episode/"final_live_scene.usda"]
    records = []
    for frame_index in args.frames:
        row = frames[frame_index]
        stamp = row["time_s"]
        state_index = int(np.argmin(abs(robot["time_s"]-stamp)))
        surface_path = episode/"surface_debug"/f"surface_{frame_index:04d}.npz"
        input_paths.append(surface_path)
        with np.load(surface_path) as data:
            vertices, faces = data["vertices_m"], data["triangles"]
        expected_hash = row["surface"]["surface_sha256"]
        if hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest() != expected_hash:
            raise ValueError("Actual pilot mesh differs from its recorded surface receipt")
        for name, candidate in CANDIDATES.items():
            transform = look_at(candidate["eye_world_m"], candidate["target_world_m"])
            root_transform = transform @ np.linalg.inv(camera_in_root)
            xf = UsdGeom.Xformable(global_root);xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(root_transform.T.tolist()))
            unsupported = scene.update(robot["body_pose"][state_index], vertices, faces,
                                       row["current_component_surface"]["component_ranges"])
            points, triangles, _, meta = stage_triangles(scene.stage, excluded_prefixes=tuple(scene.excluded))
            group_indices = {}
            for prim in meta["prims"]:
                group = _object_group(prim["prim_path"])
                offset, count = prim["first_triangle"], prim["triangles"]
                group_indices.setdefault(group, []).append(triangles[offset:offset+count].ravel())
            bounds = {}
            workcell = []
            for group, pieces in group_indices.items():
                xyz = points[np.unique(np.concatenate(pieces))]
                bounds[group] = projection_metrics(xyz, transform, k, width, height)
                if group != "own_global_camera_housing":
                    workcell.append(xyz)
            bounds["entire_workcell_excluding_own_global_housing"] = projection_metrics(np.concatenate(workcell), transform, k, width, height)
            calibration = {**deepcopy(native_calibration), "T_world_camera": transform.tolist()}
            depth, labels = scene.capture(calibration)
            valid = (labels >= 0) & (depth >= .6) & (depth <= 6.)
            material_pixels = int(np.count_nonzero(valid & (labels >= 2)))
            knife_pixels = int(np.count_nonzero(valid & (labels == 1)))
            scene_points = np.concatenate(workcell)
            eye_clearance = float(np.linalg.norm(scene_points-transform[:3, 3], axis=1).min())
            # The enclosing workcell AABB contains every scene triangle. Its
            # distance from the eye is therefore a conservative lower bound;
            # nearest-vertex distance alone would only be an upper bound.
            low, high = scene_points.min(0), scene_points.max(0)
            outside = np.maximum(np.maximum(low-transform[:3, 3], transform[:3, 3]-high), 0.)
            aabb_distance = float(np.linalg.norm(outside))
            own = bounds.get("own_global_camera_housing")
            own_indices = group_indices.get("own_global_camera_housing", [])
            own_vertices = points[np.unique(np.concatenate(own_indices))] if own_indices else np.empty((0, 3))
            radius = float(np.linalg.norm(own_vertices-transform[:3, 3], axis=1).max()) if len(own_vertices) else 0.
            fit_groups = ["table", "franka_0_including_wrist_rig", "franka_1_including_wrist_rig"]
            fit = all(bounds[g]["all_vertices_inside_rgb_frame"] for g in fit_groups)
            result = {"candidate": name, "frame": frame_index, "time_s": stamp,
                "T_world_camera": transform.tolist(), "runtime_K": k.tolist(), "resolution_wh": [width, height],
                "projected_bounds": bounds, "cpu_geometric_depth_valid_fraction": float(valid.mean()),
                "cpu_predicted_visible_knife_pixels": knife_pixels, "cpu_predicted_visible_material_pixels": material_pixels,
                "cpu_predicted_visible_material_component_pixels": {str(label-2): int(np.count_nonzero(valid & (labels == label))) for label in np.unique(labels[labels >= 2])},
                "full_table_and_both_robots_inside_rgb": fit,
                "mesh_vertex_eye_clearance_m": eye_clearance, "global_housing_max_optical_radius_m": radius,
                "eye_to_enclosing_workcell_AABB_lower_bound_m": aabb_distance,
                "conservative_housing_to_workcell_lower_bound_m": aabb_distance-radius,
                "clearance_scope": "Distance to enclosing scene AABB minus maximum actual housing radius; conservative sufficient clearance if positive, not an exact minimum triangle distance",
                "unsupported_geometry": unsupported,
                "offline_geometric_screen_pass": fit and valid.mean() >= .10 and knife_pixels >= 8,
                "native_runtime_validated": False}
            records.append(result)
            np.savez_compressed(output/f"geometry_{name}_{frame_index:04d}.npz", optical_depth_m=depth,
                                first_hit_labels=labels, public_range_valid=valid)
            palette = np.zeros((*labels.shape, 3), np.uint8)
            palette[valid] = [100, 110, 125]
            palette[valid & (labels >= 2)] = [230, 130, 50]
            palette[valid & (labels == 1)] = [70, 220, 210]
            Image.fromarray(palette).save(output/f"offline_geometry_labels_{name}_{frame_index:04d}.png")
            print(name, frame_index, "geometric_valid", result["cpu_geometric_depth_valid_fraction"],
                  "knife", knife_pixels, "material", material_pixels, "fits", fit, flush=True)
    write_json(output/"result.json", {"status": "OFFLINE_GEOMETRY_PREDICTION_ONLY", "records": records,
        "native_runtime_validated": False, "no_source_rig_modified": True})
    write_json(output/"provenance.json", {"source_episode": str(episode), "input_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
