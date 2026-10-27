#!/usr/bin/env python3
"""Verify RTX sees actual current cut components; labels stay in debug artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd

from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.rtx_outputs import decode_camera_params
from robotic_cutting_sim.usd_surface_bridge import stage_triangles


@wp.kernel
def first_hit(mesh_id: wp.uint64, eye: wp.vec3, rotation: wp.mat33,
              fx: float, fy: float, cx: float, cy: float,
              ids: wp.array2d(dtype=int), depth: wp.array2d(dtype=float)):
    y, x = wp.tid()
    ray = wp.normalize(wp.vec3((float(x) + .5 - cx) / fx, (float(y) + .5 - cy) / fy, 1.0))
    hit = wp.mesh_query_ray(mesh_id, eye, rotation * ray, 20.0)
    ids[y, x] = -1
    depth[y, x] = 0.0
    if hit.result:
        ids[y, x] = hit.face
        depth[y, x] = hit.t * ray[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_run", type=Path)
    parser.add_argument("rtx_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    (output / "oracle_debug").mkdir()
    surfaces = json.loads((args.surface_run / "result.json").read_text())
    rtx = json.loads((args.rtx_run / "result.json").read_text())
    prepared = json.loads((args.rtx_run / "prepared.json").read_text())
    if rtx["mode"] != "ideal_rgbd":
        raise ValueError("Current component geometry QA requires registered ideal RTX depth")
    stage = Usd.Stage.Open(prepared["source_scene"])
    source_rows = {round(r["physics_timestamp_s"], 9): r for r in surfaces["frames"]}
    previous, rows, changed = {}, [], {n: 0 for n in ("d455_0", "d455_1", "global")}
    chip_visible = {n: 0 for n in changed}
    wp.init()
    for index, frame in enumerate(rtx["frames"]):
        timestamp = frame["physics_timestamp_s"]
        source = source_rows[round(timestamp, 9)]
        vertices, triangles, _, geometry = stage_triangles(stage, timestamp * stage.GetTimeCodesPerSecond())
        material = next(p for p in geometry["prims"] if p["prim_path"] == "/World/ContinuousSpecimen")
        with wp.ScopedDevice("cpu"):
            mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device="cpu"),
                           indices=wp.array(triangles.ravel(), dtype=int, device="cpu"))
            row = {"physics_timestamp_s": timestamp, "mechanical_components": source["mechanical_components"], "cameras": {}}
            for name in changed:
                path = args.rtx_run / ("global_debug" if name == "global" else "observations") / f"{name}_{index:04d}.npz"
                observed = np.load(path)
                calib = decode_camera_params(observed["camera_params_raw"])
                k, transform = np.array(calib["K"]), np.array(calib["T_world_camera"])
                ids = wp.empty((calib["height"], calib["width"]), dtype=int, device="cpu")
                depths = wp.empty(ids.shape, dtype=float, device="cpu")
                wp.launch(first_hit, ids.shape, inputs=[mesh.id, wp.vec3(*transform[:3, 3]), wp.mat33(transform[:3, :3]),
                           k[0, 0], k[1, 1], k[0, 2], k[1, 2], ids, depths], device="cpu")
                faces, expected = ids.numpy(), depths.numpy()
                local_face = faces - material["first_triangle"]
                mask = (local_face >= 0) & (local_face < material["triangles"])
                # Current physical components only; these labels never enter observations.
                labels = np.full(mask.shape, -1, np.int32)
                for part in source["component_ranges"]:
                    selected = mask & (local_face >= part["first_triangle"]) & (local_face < part["first_triangle"] + part["triangles"])
                    labels[selected] = part["mechanical_component"]
                valid = mask & (observed["depth_m"] > 0)
                error = np.abs(observed["depth_m"][valid] - expected[valid])
                counts = {str(p["mechanical_component"]): int((labels == p["mechanical_component"]).sum())
                          for p in source["component_ranges"]}
                if source["mechanical_components"] > 1:
                    chip = min(source["component_ranges"], key=lambda p: p["particles"])["mechanical_component"]
                    chip_visible[name] = max(chip_visible[name], counts[str(chip)])
                if name in previous:
                    old_mask, old_depth = previous[name]
                    common = mask & old_mask & (observed["depth_m"] > 0) & (old_depth > 0)
                    changed[name] += int(np.sum(common & (np.abs(observed["depth_m"] - old_depth) > 1e-5)))
                previous[name] = mask, observed["depth_m"].copy()
                row["cameras"][name] = {"visible_material_pixels": int(valid.sum()), "current_component_pixels": counts,
                    "mesh_depth_error_p95_m": float(np.quantile(error, .95)) if len(error) else None,
                    "mesh_depth_error_max_m": float(error.max()) if len(error) else None}
                np.savez_compressed(output / "oracle_debug" / f"{name}_{index:04d}.npz",
                    current_material_component=labels, material_mesh_depth_m=np.where(mask, expected, 0),
                    physics_timestamp_s=timestamp)
            rows.append(row)
        print(f"material visibility QA frame={index} t={timestamp:.3f}s", flush=True)
    depth_errors = [camera["mesh_depth_error_p95_m"] for row in rows for camera in row["cameras"].values()
                    if camera["mesh_depth_error_p95_m"] is not None]
    agreement = bool(depth_errors) and np.isfinite(depth_errors).all() and max(depth_errors) < 1e-4
    passed = (min(changed["d455_0"], changed["d455_1"]) > 0
              and min(chip_visible["d455_0"], chip_visible["d455_1"]) > 0 and agreement)
    result = {"status": "PASS" if passed else "FAIL", "scope": "actual RTX depth observes changing current mechanical component surfaces",
              "source_surface_run": str(args.surface_run.resolve()), "source_rtx_run": str(args.rtx_run.resolve()),
              "changed_depth_material_pixels_10um": changed, "maximum_visible_chip_pixels": chip_visible, "frames": rows,
              "material_mesh_depth_p95_tolerance_m": 1e-4, "material_mesh_depth_agreement": bool(agreement),
              "oracle_labels_in_observation": False, "mask_used_to_create_rendered_cut": False,
              "physics_acceptance": "NOT_ESTABLISHED_BY_RENDERER", "simulated_d455_registration": "NOT_ASSESSED"}
    write_json(output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
