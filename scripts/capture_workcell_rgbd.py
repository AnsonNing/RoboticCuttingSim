#!/usr/bin/env python3
"""Programmable ideal RGB-D with full USD robot/tool/material occlusion.

This uses the existing Warp camera and official USD geometry, not Isaac RTX or
D455 stereo processing. Time samples are measured replay frames, not live Kit.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image
from pxr import Usd

from robotic_cutting_sim.camera import capture
from robotic_cutting_sim.recording import manifest, new_output, write_json
from robotic_cutting_sim.sensor_contract import mode_contract, validate_frame
from robotic_cutting_sim.usd_surface_bridge import stage_triangles, usd_camera_calibration


CAMERAS = {"d455_0": "/World/D455_0/RSD455/Camera_Pseudo_Depth",
           "d455_1": "/World/D455_1/RSD455/Camera_Pseudo_Depth",
           "global": "/World/GlobalCamera"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    if min(args.samples, args.width, args.height) < 1:
        raise ValueError("Samples and resolution must be positive")
    output = new_output(args.output_dir)
    (output / "observations").mkdir(); (output / "global_debug").mkdir()
    stage = Usd.Stage.Open(str(args.scene))
    mesh = stage.GetPrimAtPath("/World/ContinuousSpecimen")
    samples = mesh.GetAttribute("points").GetTimeSamples()
    samples = (np.array(samples)[np.unique(np.linspace(0, len(samples) - 1, min(len(samples), args.samples), dtype=int))]
               if samples else [0.0])
    calib = {name: usd_camera_calibration(stage, path, args.width, args.height,
             near_m=.52 if name != "global" else .01, far_m=6.0 if name != "global" else 20)
             for name, path in CAMERAS.items()}
    write_json(output / "calibration.json", calib)
    write_json(output / "manifest.json", manifest({"scene": str(args.scene.resolve()), "device": args.device,
                "samples": list(map(float, samples)), **mode_contract("ideal_rgbd")}, "full_usd_ideal_rgbd"))
    wp.init()
    rows, previous, changed = [], {}, {name: 0 for name in CAMERAS}
    for index, tc in enumerate(samples):
        tick = time.perf_counter()
        vertices, faces, colors, geometry = stage_triangles(stage, float(tc))
        with wp.ScopedDevice(args.device):
            verts = wp.array(vertices, dtype=wp.vec3, device=args.device)
            tris = wp.array(faces.ravel(), dtype=int, device=args.device)
            face_colors = wp.array(colors, dtype=wp.vec3, device=args.device)
            timestamp = float(tc) / stage.GetTimeCodesPerSecond()
            row = {"frame": index, "physics_timestamp_s": timestamp, "surface_timestamp_s": timestamp,
                   "triangles": len(faces), "cameras": {}}
            for name, calibration in calib.items():
                rgb, depth = capture(verts, tris, calibration, args.device, face_colors)
                frame = validate_frame(rgb, depth, timestamp, timestamp)
                if name in previous:
                    common = (depth > 0) & (previous[name] > 0)
                    changed[name] += int(np.sum(common & (np.abs(depth - previous[name]) > 1e-5)))
                previous[name] = depth.copy()
                target = output / ("global_debug" if name == "global" else "observations")
                np.savez_compressed(target / f"{name}_{index:04d}.npz", **frame)
                Image.fromarray(rgb).save(target / f"{name}_{index:04d}.png")
                row["cameras"][name] = {"valid_pixels": int(frame["valid_mask"].sum()),
                                        "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape)}
            row["wall_s"] = time.perf_counter() - tick
            rows.append(row)
        if index == 0:
            write_json(output / "geometry.json", geometry)
        print(f"frame={index} t={timestamp:.3f}s triangles={len(faces)} wall={row['wall_s']:.3f}s", flush=True)
    result = {"status": "PASS" if all(r["cameras"][n]["valid_pixels"] for r in rows for n in CAMERAS) else "FAIL",
              "scope": "full composed USD to ideal programmable RGB-D; measured replay",
              **mode_contract("ideal_rgbd"), "frames": rows, "changed_depth_pixels_10um": changed,
              "camera_contains": ["dual Panda native meshes", "dual D455 housings", "table", "tray", "blade", "material"],
              "global_in_observation": False, "simulated_d455": "NOT_RUN", "rtx_validated": False,
              "cut_surface_validated": False,
              "limitations": ["Flat display colors; no MDL/texture/stereo noise",
                  "Polygon fans; no subdivision or curved primitive tessellation",
                  "Static USD robot pose unless measured articulation transforms are authored",
                  "Replay timestamps, no live Isaac sensor scheduling",
                  "This renderer does not establish physical material separation"]}
    write_json(output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
