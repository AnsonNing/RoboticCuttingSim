#!/usr/bin/env python3
"""Describe native task-object coverage using immutable measured-mesh first hits.

Evaluator-only oracle ROIs; never add masks, repaired depth or oracle data to
formal observations. Native invalid pixels and valid error tails remain counts.
No acceptance threshold is selected or fitted by this descriptive audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from robotic_cutting_sim.recording import new_output, write_json


def error_distribution(error):
    values = np.abs(np.asarray(error, float))
    return {"pixels": int(len(values)),
            "p95_absolute_error_m": float(np.quantile(values, .95)) if len(values) else None,
            "max_absolute_error_m": float(np.max(values)) if len(values) else None,
            "rms_error_m": float(np.sqrt(np.mean(values**2))) if len(values) else None,
            "absolute_error_gt_1mm_pixels": int(np.sum(values > .001)),
            "absolute_error_gt_5mm_pixels": int(np.sum(values > .005)),
            "error_trimming_applied": False}


def object_metrics(mask, geometry, native, native_valid, rtx, rtx_valid):
    mask = np.asarray(mask, bool)
    total = int(mask.sum())
    finite_native = np.isfinite(native) & (native > 0)
    valid_native = mask & native_valid & finite_native
    native_error = native[valid_native].astype(float)-geometry[valid_native].astype(float)
    finite_rtx = np.isfinite(rtx) & (rtx > 0)
    valid_rtx = mask & rtx_valid & finite_rtx
    rtx_error = rtx[valid_rtx].astype(float)-geometry[valid_rtx].astype(float)
    receipt = {"reference_visible_pixels": total,
               "native_valid_pixels": int(valid_native.sum()),
               "native_valid_coverage": float(valid_native.sum()/total) if total else None,
               "native_invalid_pixels": int(np.sum(mask & ~native_valid)),
               "native_valid_but_nonfinite_or_nonpositive_pixels": int(np.sum(mask & native_valid & ~finite_native)),
               "native_depth_vs_mesh": error_distribution(native_error),
               "actual_rtx_valid_pixels": int(valid_rtx.sum()),
               "actual_rtx_invalid_pixels": int(np.sum(mask & ~rtx_valid)),
               "actual_rtx_valid_but_nonfinite_or_nonpositive_pixels": int(np.sum(mask & rtx_valid & ~finite_rtx)),
               "actual_rtx_depth_vs_mesh": error_distribution(rtx_error),
               "native_invalid_or_error_gt_1mm_pixels": total-int(np.sum(abs(native_error) <= .001)),
               "native_invalid_or_error_gt_5mm_pixels": total-int(np.sum(abs(native_error) <= .005)),
               "native_valid_error_pixels_were_dropped": False}
    return receipt, native_error, rtx_error


def summarize(rows, errors):
    count = sum(row["reference_visible_pixels"] for row in rows)
    visible = [row for row in rows if row["reference_visible_pixels"]]
    native_errors = np.concatenate([q[0] for q in errors]) if errors else np.empty(0)
    rtx_errors = np.concatenate([q[1] for q in errors]) if errors else np.empty(0)
    return {"sampled_frames": len(rows), "frames_with_reference_visibility": len(visible),
            "minimum_reference_visible_pixels": min(row["reference_visible_pixels"] for row in rows),
            "maximum_reference_visible_pixels": max(row["reference_visible_pixels"] for row in rows),
            "total_reference_pixel_samples": count,
            "native_valid_coverage_weighted": sum(row["native_valid_pixels"] for row in rows)/count if count else None,
            "minimum_native_valid_coverage_when_visible": min(row["native_valid_coverage"] for row in visible) if visible else None,
            "native_invalid_pixel_samples": sum(row["native_invalid_pixels"] for row in rows),
            "native_valid_but_nonfinite_or_nonpositive_pixel_samples": sum(row["native_valid_but_nonfinite_or_nonpositive_pixels"] for row in rows),
            "native_invalid_or_error_gt_1mm_pixel_samples": sum(row["native_invalid_or_error_gt_1mm_pixels"] for row in rows),
            "native_invalid_or_error_gt_5mm_pixel_samples": sum(row["native_invalid_or_error_gt_5mm_pixels"] for row in rows),
            "native_depth_vs_mesh": error_distribution(native_errors),
            "actual_rtx_invalid_pixel_samples": sum(row["actual_rtx_invalid_pixels"] for row in rows),
            "actual_rtx_depth_vs_mesh": error_distribution(rtx_errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera_audit", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    source = json.loads((args.camera_audit/"input.json").read_text())
    camera_audit = json.loads((args.camera_audit/"result.json").read_text())
    episode = Path(source["episode"])
    frames = {row["frame"]: row for row in json.loads((episode/"frame_manifest.json").read_text())}
    setup = json.loads((episode/"setup.json").read_text())
    if not setup.get("task"):
        raise ValueError("This audit requires an actual recorded task episode")
    implementation = output/"implementation"
    implementation.mkdir()
    (implementation/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (args.camera_audit/"input.json", args.camera_audit/"result.json", episode/"setup.json", Path(__file__))}
    records, groups, group_errors = [], {}, {}
    for frame in camera_audit["records"]:
        index = frame["frame"]
        ranges = frames[index]["current_component_surface"]["component_ranges"]
        component_count = len(ranges)
        for camera_name in frame["cameras"]:
            path = args.camera_audit/camera_name/f"{index:04d}.npz"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as data:
                labels, names = data["geometry_object_labels"], data["object_names"].tolist()
                geometry, native, valid = data["geometry_depth_m"], data["native_depth_m"], data["native_valid_mask"]
                rtx = data["actual_rtx_depth_m"]
                # The exact native-vs-reference mask was captured separately;
                # do not infer reference validity from native depth availability.
                if camera_name == "global":
                    payload_path = episode/"global_debug"/f"global_{index:04d}.npz"
                    key = "valid_mask"
                else:
                    payload_path = episode/"camera_debug"/f"{camera_name}_{index:04d}.npz"
                    key = "rtx_reference_valid_mask"
                with np.load(payload_path) as reference:
                    rtx_valid = reference[key]
                hashes[str(payload_path)] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
                masks = {name: labels == names.index(name) for name in names
                         if name == "knife" or name.startswith("material_component_")}
                masks["all_material"] = np.logical_or.reduce([labels == i for i, name in enumerate(names) if name.startswith("material_component_")])
                objects = {}
                for name, mask in masks.items():
                    metric, native_error, rtx_error = object_metrics(mask, geometry, native, valid, rtx, rtx_valid)
                    objects[name] = metric
                    for split in (["all_frames", "post_separation"] if component_count > 1 else ["all_frames"]):
                        group = (camera_name, name, split)
                        groups.setdefault(group, []).append(metric)
                        group_errors.setdefault(group, []).append((native_error, rtx_error))
                records.append({"frame": index, "time_s": frame["time_s"], "camera": camera_name,
                                "actual_current_mechanical_components": component_count,
                                "current_component_ranges": ranges, "objects": objects,
                                "native": camera_name != "global"})
    summaries = {}
    for (camera, object_name, split), rows in groups.items():
        summaries.setdefault(camera, {}).setdefault(object_name, {})[split] = summarize(rows, group_errors[camera, object_name, split])
    result = {"status": "DESCRIPTIVE_TASK_OBJECT_AUDIT", "command": sys.argv,
              "task_id": setup["task"]["task_id"], "seed": setup["config"]["seed"],
              "episode": str(episode), "camera_audit": str(args.camera_audit),
              "scope": "Recorded measured-mesh first-hit oracle ROI only; no native semantic AOV or formal observation changes",
              "new_acceptance_thresholds": None, "strict_native_CAM04": camera_audit["CAM-04_strict_native"],
              "native_invalid_pixels_and_all_valid_error_tails_retained": True,
              "global_is_ideal_reference_not_native_D455": True,
              "input_sha256": hashes, "summaries": summaries, "records": records}
    write_json(output/"result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_sha256", "summaries", "records")}, indent=2))


if __name__ == "__main__":
    main()
