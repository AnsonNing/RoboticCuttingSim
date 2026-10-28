#!/usr/bin/env python3
"""Audit saved measured wrist-camera transforms and actual RGB-D independently.

No solver/camera is advanced. CPU current-scene first hits are an audit reference,
not new native capture or semantic AOV. All valid depth-error tails are retained.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.live_acceptance import load_npz, resolve_recorded_source, sha256
from robotic_cutting_sim.recording import ROOT, new_output, write_json
from robotic_cutting_sim.scene_frames import pose_matrix, transform_points
from robotic_cutting_sim.wrist_camera_audit import (depth_comparison, effective_fov_deg,
    obb_signed_projection_gap, project_points, trace_metrics, transform_error)

PROTOCOL_PATH = ROOT/"configs/wrist_camera_audit_v1.json"


def tool_part_label(path):
    """Keep material component IDs separate from the real rigid tool parts."""
    physical = {"/World/Tools/CuttingTool_0/Blade": 1,
                "/World/Tools/CuttingTool_0/Handle": -3,
                "/World/Tools/CuttingTool_0/Shaft": -4}
    return physical.get(path, 1 if path.endswith(("/Knife", "/LiveKnife")) else None)


def read_json(path):
    return json.loads(Path(path).read_text())


def hands_from_robot(robot, rig, setup):
    for key in ("hand_poses_world_xyzw", "hand_pose_both_xyzw"):
        if key in robot:
            return robot[key]
    if "hand_body_indices" in rig:
        return robot["body_pose"][:, rig["hand_body_indices"]]
    paths = [str(p) for p in robot.get("body_prim_paths", setup.get("body_labels", []))]
    names = [rig["specs"][f"wrist_d405_{i}"]["parent_hand_prim_path"] for i in (0, 1)]
    return robot["body_pose"][:, [paths.index(name) for name in names]]


def frame_inputs(directory):
    """Yield both runtime layouts using actual per-camera dimensions/calibration."""
    manifest = directory/"frame_manifest.json"
    if manifest.is_file():
        for row in read_json(manifest):
            index = int(row["frame"])
            calibration = row["camera_calibration"]
            payloads, paths, raw = {}, [manifest], {}
            for name in calibration:
                candidates = [directory/folder/f"{name}_{index:04d}.npz" for folder in ("observations", "global_debug")]
                path = next((path for path in candidates if path.is_file()), None)
                if path is None:
                    raise FileNotFoundError(f"Actual RGB-D payload missing: {name}, frame {index}")
                payloads[name] = load_npz(path);paths.append(path)
                raw_path = directory/"camera_debug"/f"{name}_{index:04d}.npz"
                if raw_path.is_file():
                    data = load_npz(raw_path)
                    for key in ("rtx_reference_distance_to_image_plane_raw", "rtx_reference_depth_m"):
                        if key in data:
                            raw[name] = data[key];paths.append(raw_path);break
            surface_path = directory/"surface_debug"/f"surface_{index:04d}.npz"
            data = load_npz(surface_path);paths.append(surface_path)
            yield {"index": index, "time_s": float(row["time_s"]), "calibration": calibration,
                "payloads": payloads, "raw_reference": raw, "vertices": data["vertices_m"], "triangles": data["triangles"],
                "surface_time_s": float(data["physics_timestamp_s"]), "surface_metadata": row["current_component_surface"], "paths": paths}
    else:
        for path in sorted((directory/"rgbd").glob("frame_*.npz")):
            suffix = path.stem.split("_")[-1]
            calibration_path = path.with_name("calibration_"+suffix+".json")
            calibration = read_json(calibration_path)
            arrays = load_npz(path)
            payloads = {name: {key: arrays[name+"_"+key] for key in ("rgb", "depth_m", "valid_mask")} for name in calibration}
            surface_path = path.with_name("surface_"+suffix+".npz")
            metadata_path = surface_path.with_suffix(".json")
            data = load_npz(surface_path)
            yield {"index": int(suffix), "time_s": float(arrays["timestamp_s"]), "calibration": calibration,
                "payloads": payloads, "raw_reference": {}, "vertices": data["vertices"], "triangles": data["triangles"],
                "surface_time_s": float(data["timestamp_s"]), "surface_metadata": read_json(metadata_path),
                "paths": [path, calibration_path, surface_path, metadata_path]}


class SceneReference:
    def __init__(self, directory, robot, setup, native):
        from pxr import Usd, UsdGeom
        import warp as wp
        from robotic_cutting_sim.visual_geometry import non_render_purpose_paths

        scene_path = directory/"final_live_scene.usda"
        if not scene_path.is_file():
            raise FileNotFoundError("Full composed actual final_live_scene.usda is required for whole-scene occlusion auditing")
        self.stage = Usd.Stage.Open(str(scene_path))
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        self.ops = []
        paths = [str(p) for p in robot.get("body_prim_paths", setup.get("body_labels", []))]
        if len(paths) != robot["body_pose"].shape[1]:
            raise ValueError("Every actual recorded body pose needs its actual USD prim path")
        for path in paths:
            prim = self.stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("Recorded native body path missing in actual exported stage: "+path)
            xf = UsdGeom.Xformable(prim);xf.ClearXformOpOrder();xf.SetResetXformStack(True)
            self.ops.append(xf.AddTransformOp())
        self.specimen = UsdGeom.Mesh.Get(self.stage, "/World/ContinuousSpecimen")
        if not self.specimen:
            raise ValueError("Actual exported stage lacks the current material mesh")
        self.excluded = non_render_purpose_paths(self.stage) if native else []
        # The Lab ideal renderer includes its actual USD geometry without native
        # single-sided rasterization. Native scene rendering follows USD purpose.
        self.native = native
        path = ROOT/"scripts/audit_live_camera_mesh.py"
        spec = importlib.util.spec_from_file_location("wrist_independent_reference_kernel", path)
        module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module;spec.loader.exec_module(module)
        self.kernel = module.reference_rays
        self.wp = wp

    def update(self, poses, vertices, faces, component_ranges):
        from pxr import Gf, Usd, UsdGeom
        from robotic_cutting_sim.usd_surface_bridge import stage_triangles

        for op, pose in zip(self.ops, poses):
            op.Set(Gf.Matrix4d(pose_matrix(pose).T.tolist()))
        self.specimen.GetPointsAttr().Set(vertices)
        self.specimen.GetFaceVertexCountsAttr().Set(np.full(len(faces), 3, np.int32))
        self.specimen.GetFaceVertexIndicesAttr().Set(faces.ravel())
        self.specimen.GetNormalsAttr().Block()
        points, triangles, _, metadata = stage_triangles(self.stage, excluded_prefixes=tuple(self.excluded))
        ids, sides = np.zeros(len(triangles), np.int32), np.zeros(len(triangles), np.int32)
        for prim in metadata["prims"]:
            path, offset, count = prim["prim_path"], prim["first_triangle"], prim["triangles"]
            part = tool_part_label(path)
            if part is not None:
                ids[offset:offset+count] = part
            if "ContinuousSpecimen" in path:
                for row in component_ranges:
                    start = offset+row["first_triangle"]
                    ids[start:start+row["triangles"]] = 2+int(row["mechanical_component"])
            usd_prim = self.stage.GetPrimAtPath(path)
            if self.native and usd_prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(usd_prim)
                if not mesh.GetDoubleSidedAttr().Get():
                    side = -1 if mesh.GetOrientationAttr().Get() == "leftHanded" else 1
                    if np.linalg.det(np.asarray(UsdGeom.Xformable(usd_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))[:3, :3]) < 0:
                        side = -side
                    sides[offset:offset+count] = side
        wp = self.wp
        self.mesh = wp.Mesh(wp.array(points, dtype=wp.vec3, device="cpu"), wp.array(triangles.ravel(), dtype=int, device="cpu"))
        self.ids = wp.array(ids, dtype=int, device="cpu")
        self.sides = wp.array(sides, dtype=int, device="cpu")
        return metadata["unsupported_geometry"]

    def capture(self, calibration):
        wp = self.wp
        shape = (int(calibration["height"]), int(calibration["width"]))
        depth, labels = wp.empty(shape, dtype=float, device="cpu"), wp.empty(shape, dtype=int, device="cpu")
        wp.launch(self.kernel, dim=shape, inputs=[self.mesh.id, wp.mat44(np.asarray(calibration["T_world_camera"])),
            wp.mat33(np.asarray(calibration["K"])), .001, 20., self.ids, self.sides, depth, labels], device="cpu")
        attribution = labels.numpy()
        self.last_tool_part_labels = attribution
        # Handle/shaft remain real occluders. The existing audit uses IDs>=2
        # exclusively for actual material components, so return them as other.
        ordinary = attribution.copy()
        ordinary[(ordinary == -3) | (ordinary == -4)] = 0
        return depth.numpy(), ordinary


def clearance_metrics(trace, specs, robot, task):
    table = np.eye(4);table[2, 3] = task["table_top_world_z_m"]-.025
    obstacles = [("table", table, [1.8, 1.2, .05])]
    workpiece = pose_matrix(task["world_from_workpiece_xyzw"])
    for row in task["fixture_boxes_local"]:
        box = np.eye(4);box[:3, 3] = row["center_m"]
        obstacles.append((row["name"], workpiece@box, 2*np.asarray(row["half_extents_m"])))
    length, height, width = [task[k] for k in ("blade_length_m", "blade_height_m", "blade_half_width_m")]
    knife_offset = np.eye(4);knife_offset[:3, 3] = [-length/2, 0, (height-.0004)/2]
    rows = []
    for sample in trace:
        stamp = sample["timestamp_s"];index = int(np.argmin(abs(robot["time_s"]-stamp)))
        knife = pose_matrix(robot["tool_pose_xyzw"][index])@knife_offset
        current_obstacles = obstacles+[("knife_envelope", knife, [length, 2*width, height+.0004])]
        for name, spec in specs.items():
            camera = sample["cameras"][name]
            for box in spec.get("visual_obbs", []):
                if "T_hand_obb" in box:
                    hand = camera.get("T_world_hand")
                    if hand is None:
                        hand = sample["T_world_hand"][spec["parent_hand_index"]]
                    world = np.asarray(hand)@box["T_hand_obb"]
                elif "T_optical_cv_obb" in box:
                    world = np.asarray(camera["T_world_camera"])@box["T_optical_cv_obb"]
                else:
                    world = np.asarray(box["T_world_obb"])
                for obstacle, transform, dimensions in current_obstacles:
                    gap = obb_signed_projection_gap(world, box["dimensions_m"], transform, dimensions)
                    rows.append({"time_s": stamp, "camera": name, "box": box["name"], "obstacle": obstacle, "signed_projection_gap_m": gap})
    return rows


def audit(directory, output, protocol, selected_frames=None):
    output.mkdir()
    setup, completion, rig = [read_json(directory/name) for name in ("setup.json", "result.json", "camera_rig.json")]
    trace_path = directory/"camera_pose_trace.json"
    trace = read_json(trace_path)
    trace = trace["samples"] if isinstance(trace, dict) else trace
    robot = load_npz(directory/"robot_state.npz")
    specs = rig["specs"]
    task = setup.get("task") or read_json(directory/"task.json")
    measured_hands = hands_from_robot(robot, rig, setup)
    kinematics = trace_metrics(trace, robot["time_s"], measured_hands, specs, protocol)
    gaps = clearance_metrics(trace, specs, robot, task)
    write_json(output/"visual_clearance_records.json", gaps)
    native = (directory/"frame_manifest.json").is_file()
    scene = SceneReference(directory, robot, setup, native)
    oracle = directory/"oracle_debug" if native else directory
    with np.load(oracle/"physics_state.npz", allow_pickle=False) as source:
        physics = {key: source[key] for key in ("time_s", "codes", "mass_kg", "reference_stencil_nodes", "reference_stencil_weights", "reference_stencil_gradients")}
    nodes, weights, gradients = [physics[k] for k in ("reference_stencil_nodes", "reference_stencil_weights", "reference_stencil_gradients")]
    stencil_hash = hashlib.sha256(nodes.tobytes()+weights.tobytes()+gradients.tobytes()).hexdigest()
    dx = protocol["visibility"]["reference_flat_base_top_grid_spacing_m"]
    geometry = task["geometry"]
    x = (np.arange(round(geometry["length_m"]/dx))+.5)*dx-geometry["length_m"]/2
    y = (np.arange(round(geometry["width_m"]/dx))+.5)*dx-geometry["width_m"]/2
    xx, yy = np.meshgrid(x, y, indexing="ij")
    reference_points = transform_points(pose_matrix(task["world_from_workpiece_xyzw"]),
        np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, geometry["base_height_m"])])
    input_paths = [directory/name for name in ("manifest.json", "setup.json", "result.json", "camera_rig.json", "camera_pose_trace.json", "robot_state.npz", "final_live_scene.usda")]
    input_paths.append(oracle/"physics_state.npz")
    if (directory/"task.json").is_file():input_paths.append(directory/"task.json")
    records, graph_cache = [], {}
    for frame in frame_inputs(directory):
        if selected_frames is not None and frame["index"] not in selected_frames:
            continue
        input_paths.extend(frame["paths"])
        stamp, metadata = frame["time_s"], frame["surface_metadata"]
        robot_index = int(np.argmin(abs(robot["time_s"]-stamp)))
        trace_sample = min(trace, key=lambda r: abs(r["timestamp_s"]-stamp))
        state_index = int(np.argmin(abs(physics["time_s"]-stamp)))
        codes = physics["codes"][state_index]
        code_hash = hashlib.sha256(codes.tobytes()).hexdigest()
        if (metadata["codes_sha256"] != code_hash or metadata["stencil_sha256"] != stencil_hash
                or metadata["future_component_ids_used"] or metadata["renderer_partition_changes_physics"]):
            raise ValueError("Current rendered surface is not bound to current actual MPM support/history")
        if code_hash not in graph_cache:
            graph_cache[code_hash] = mechanical_components(nodes, weights, codes, 1, gradients=gradients)
        count, labels = graph_cache[code_hash]
        ranges = metadata["component_ranges"]
        if count != len(ranges) or any(np.count_nonzero(labels == row["mechanical_component"]) != row["particles"] for row in ranges):
            raise ValueError("Current surface component IDs/counts disagree with actual stress-gradient graph")
        unsupported = scene.update(robot["body_pose"][robot_index], frame["vertices"], frame["triangles"], ranges)
        row = {"frame": frame["index"], "time_s": stamp, "current_mechanical_components": count,
            "current_codes_sha256": code_hash, "unsupported_geometry": unsupported,
            "surface_partition_bound_to_actual_current_support_and_codes": True, "cameras": {},
            "timestamp_alignment_error_s": max(abs(float(robot["time_s"][robot_index])-stamp),
                abs(float(physics["time_s"][state_index])-stamp), abs(float(trace_sample["timestamp_s"])-stamp), abs(frame["surface_time_s"]-stamp))}
        projected_masks = []
        component_pixels = {str(part["mechanical_component"]): 0 for part in ranges}
        for name in protocol["camera_ids"]:
            camera, payload = frame["calibration"][name], frame["payloads"][name]
            depth, valid, rgb = payload["depth_m"], payload["valid_mask"].astype(bool), payload["rgb"]
            if depth.shape != (camera["height"], camera["width"]) or rgb.shape != (*depth.shape, 3):
                raise ValueError("Actual RGB-D shape differs from its per-frame runtime calibration")
            pose_error = transform_error(trace_sample["cameras"][name]["T_world_camera"], camera["T_world_camera"])
            _, _, projected = project_points(reference_points, camera)
            if name.startswith("wrist_"):projected_masks.append(projected)
            raw_geometry, attribution = scene.capture(camera)
            geometry_valid = ((attribution >= 0) & (raw_geometry >= camera["near_m"]) & (raw_geometry <= camera["far_m"]))
            expected = np.where(geometry_valid, raw_geometry, 0.)
            tolerance = protocol["visibility"]["native_global_depth_geometric_match_tolerance_m" if native and name == "global_d455" else "ideal_depth_geometric_match_tolerance_m"]
            consistent = valid & geometry_valid & (abs(depth-expected) <= tolerance)
            per_component = {str(part["mechanical_component"]): int(np.count_nonzero(consistent & (attribution == 2+part["mechanical_component"]))) for part in ranges}
            for key, value in per_component.items():component_pixels[key] = max(component_pixels[key], value)
            fov = effective_fov_deg(camera)
            optics = protocol["optics"]
            sensor = camera.get("sensor_model", specs[name]["sensor_model"])
            view = {"resolution_wh": [int(camera["width"]), int(camera["height"])], "K": camera["K"], "effective_fov_deg": fov,
                "sensor_model": sensor, "valid_depth_fraction": float(valid.mean()),
                "nonfinite_depth_values": int(np.count_nonzero(~np.isfinite(depth))),
                "valid_mask_disagreement_with_positive_depth_pixels": int(np.count_nonzero(valid != (np.isfinite(depth) & (depth > 0)))),
                "valid_depth_outside_declared_working_range_pixels": int(np.count_nonzero(valid & ((depth < camera["near_m"]-optics["working_range_endpoint_tolerance_m"]) | (depth > camera["far_m"]+optics["working_range_endpoint_tolerance_m"])))),
                "calibration_vs_measured_trace_translation_error_m": pose_error[0], "calibration_vs_measured_trace_rotation_error_rad": pose_error[1],
                "reference_footprint_in_frustum_fraction": float(projected.mean()),
                "depth_consistent_material_pixels": int(np.count_nonzero(consistent & (attribution >= 2))),
                "depth_consistent_knife_pixels": int(np.count_nonzero(consistent & (attribution == 1))),
                "current_component_depth_consistent_pixels": per_component,
                "actual_depth_vs_current_whole_scene": depth_comparison(depth, valid, expected, geometry_valid),
                "depth_match_scope": "Counts verified by full current-scene first hit; all valid errors/holes reported without trimming"}
            if setup.get("tool_attachment_mode") == "physical_grasp":
                view["physical_tool_parts"] = {part: {
                    "first_hit_pixels": int(np.count_nonzero(scene.last_tool_part_labels == label)),
                    "working_range_pixels": int(np.count_nonzero((scene.last_tool_part_labels == label) & geometry_valid)),
                    "actual_depth_consistent_pixels": int(np.count_nonzero((scene.last_tool_part_labels == label) & consistent))}
                    for part, label in (("blade", 1), ("handle", -3), ("shaft", -4))}
            if name in frame["raw_reference"]:
                raw = frame["raw_reference"][name]
                raw_valid = np.isfinite(raw) & (raw > 0) & (raw <= 20.)
                view["cpu_geometry_vs_actual_raw_rtx"] = depth_comparison(raw, raw_valid, raw_geometry, attribution >= 0)
            if name.startswith("wrist_"):
                view["nominal_optics_check"] = (abs(fov["horizontal_deg"]-optics["wrist_nominal_horizontal_fov_deg"]) <= optics["wrist_fov_tolerance_deg"]
                    and abs(fov["vertical_deg"]-optics["wrist_nominal_vertical_fov_deg"]) <= optics["wrist_fov_tolerance_deg"])
            row["cameras"][name] = view
        row["reference_footprint_in_union_wrist_frusta_fraction"] = float(np.any(projected_masks, axis=0).mean())
        row["current_component_best_view_pixels"] = component_pixels
        records.append(row)
        write_json(output/"partial.json", records)
        print(directory.name, "frame", frame["index"], "components", count, flush=True)
    if not records:
        raise ValueError("No recorded RGB-D frames were selected")
    visibility = protocol["visibility"]
    checks = {"frame_timestamps": max(row["timestamp_alignment_error_s"] for row in records) <= protocol["transforms"]["maximum_timestamp_alignment_error_s"],
        "reference_footprint_union_coverage": min(row["reference_footprint_in_union_wrist_frusta_fraction"] for row in records) >= visibility["minimum_reference_footprint_fraction_in_union_wrist_frusta"]}
    for name in protocol["camera_ids"]:
        views = [row["cameras"][name] for row in records]
        checks[name+"_nonempty_working_depth"] = min(view["valid_depth_fraction"] for view in views) >= visibility["minimum_valid_depth_fraction"][name]
        checks[name+"_material_visible"] = min(view["depth_consistent_material_pixels"] for view in views) >= visibility["minimum_depth_consistent_material_pixels"][name]
        checks[name+"_knife_visible"] = min(view["depth_consistent_knife_pixels"] for view in views) >= visibility["minimum_depth_consistent_knife_pixels"][name]
        checks[name+"_finite_range_and_masks"] = all(not view["nonfinite_depth_values"] and not view["valid_mask_disagreement_with_positive_depth_pixels"] and not view["valid_depth_outside_declared_working_range_pixels"] for view in views)
        checks[name+"_calibration_measured_pose"] = all(view["calibration_vs_measured_trace_translation_error_m"] <= protocol["transforms"]["maximum_hand_camera_translation_error_m"] and view["calibration_vs_measured_trace_rotation_error_rad"] <= protocol["transforms"]["maximum_hand_camera_rotation_error_rad"] for view in views)
        if name.startswith("wrist_"):checks[name+"_runtime_optics"] = all(view["nominal_optics_check"] for view in views)
    split = [row for row in records if row["current_mechanical_components"] > 1]
    component_visible = bool(split) and all(min(row["current_component_best_view_pixels"].values()) >= visibility["minimum_current_component_depth_consistent_pixels_in_any_postsplit_view"] for row in split)
    gap_min = min((row["signed_projection_gap_m"] for row in gaps), default=None)
    clearance = {"status": "NOT_RUN" if gap_min is None else "PASS_CONSERVATIVE_VISUAL_BOUNDS" if gap_min >= protocol["visual_clearance"]["minimum_certified_obb_separation_from_table_fixture_tool_m"] else "FAIL_CONSERVATIVE_VISUAL_BOUNDS",
        "minimum_signed_projection_gap_m": gap_min, "minimum_record": min(gaps, key=lambda row: row["signed_projection_gap_m"]) if gaps else None,
        "native_rigid_contact_acceptance": "NOT_SCORED", "own_hand_and_link_mesh_clearance": "NOT_SCORED_HERE; separate rig controls apply"}
    complete = (completion.get("steps") == 400 and selected_frames is None
        and abs(float(robot["time_s"][-1]-robot["time_s"][0])-8.) <= 1e-6
        and abs(records[0]["time_s"]) <= 1e-6 and abs(records[-1]["time_s"]-8.) <= 1e-6)
    manifest = read_json(directory/"manifest.json")
    verified_sources = {path: resolve_recorded_source(directory, path, expected)
        for path, expected in manifest.get("implementation_sha256", {}).items()}
    result = {"episode": str(directory.resolve()), "scope": "Independent measured-rig and current-scene RGB-D engineering audit; no new sensor capture or native semantic AOV",
        "complete_8s_episode": complete, "frame_subset": selected_frames, "kinematics": kinematics,
        "visibility_and_optics": {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks},
        "current_component_visibility": "PASS_MINIMAL_PIXEL_EVIDENCE" if component_visible else "FAIL" if split else "NOT_RUN_NO_SPLIT",
        "component_visibility_scope": "Every current component in every recorded post-split partition needs at least one verified pixel in some camera; IDs are never carried across different partitions",
        "visual_clearance": clearance, "frames": len(records), "records": records,
        "sensor_scope": {"wrist_cameras": "ideal_d405_prototype, not validated hardware calibration/noise", "global_camera": "native SingleView D455" if native else "explicit ideal Warp RGB-D from official D455 asset"},
        "full_native_sensor_acceptance": "NOT_SCORED", "task_physics_regression": "SEPARATE_AUDIT_REQUIRED",
        "full_benchmark_ready": False, "protocol_sha256": sha256(PROTOCOL_PATH),
        "runtime_source_verification": verified_sources,
        "all_manifest_implementation_bytes_resolved": bool(verified_sources) and all(row["verified"] for row in verified_sources.values()),
        "input_sha256": {str(path.relative_to(directory)): sha256(path) for path in set(input_paths)}}
    write_json(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    args = parser.parse_args()
    output = new_output(args.output_dir)
    paths = [Path(__file__).resolve(), PROTOCOL_PATH, ROOT/"src/robotic_cutting_sim/wrist_camera_audit.py",
        ROOT/"scripts/audit_live_camera_mesh.py", ROOT/"src/robotic_cutting_sim/usd_surface_bridge.py", ROOT/"src/robotic_cutting_sim/visual_geometry.py",
        ROOT/"src/robotic_cutting_sim/cut_topology.py", ROOT/"src/robotic_cutting_sim/scene_frames.py"]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    snapshots = output/"analysis_sources";snapshots.mkdir()
    for path in paths:shutil.copyfile(path, snapshots/path.name)
    write_json(output/"analysis_start.json", {"source_sha256": hashes, "copied_before_analysis": True})
    protocol = read_json(PROTOCOL_PATH)
    cases = {}
    for directory in args.episode:
        cases[directory.name] = audit(directory.resolve(), output/directory.name, protocol, args.frames)
    stable = all(sha256(ROOT/path) == value for path, value in hashes.items())
    write_json(output/"result.json", {"cases": cases, "analysis_sources_unchanged": stable, "source_sha256": hashes, "full_benchmark_ready": False})
    if not stable:raise RuntimeError("Audit source changed during analysis; receipt retained")


if __name__ == "__main__":
    main()
