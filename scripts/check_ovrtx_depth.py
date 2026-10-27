#!/usr/bin/env python3
"""Check actual RTX image-plane depth against the independently ray-tested table."""
import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd
from scipy.ndimage import binary_erosion

from robotic_cutting_sim.camera import capture
from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.rtx_outputs import decode_camera_params
from robotic_cutting_sim.usd_surface_bridge import stage_triangles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rtx_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    prepared = json.loads((args.rtx_run / "prepared.json").read_text())
    stage = Usd.Stage.Open(prepared["source_scene"])
    timestamp = float(prepared["time_s"][0])
    vertices, faces, colors, _ = stage_triangles(stage, timestamp * stage.GetTimeCodesPerSecond())
    wp.init()
    vertices = wp.array(vertices, dtype=wp.vec3, device="cpu")
    faces = wp.array(faces.ravel(), dtype=int, device="cpu")
    colors = wp.array(colors, dtype=wp.vec3, device="cpu")
    rows = {}
    for name in ("d455_0", "d455_1", "global"):
        path = args.rtx_run / ("global_debug" if name == "global" else "observations") / f"{name}_0000.npz"
        data = np.load(path)
        calib = decode_camera_params(data["camera_params_raw"])
        if not np.isclose(calib["meters_per_scene_unit"], 1):
            raise ValueError("RTX input did not preserve SI units")
        calib.update(near_m=.01, far_m=20)
        _, independent_depth = capture(vertices, faces, calib, "cpu", colors)
        k, transform = np.asarray(calib["K"]), np.asarray(calib["T_world_camera"])
        y, x = np.indices(independent_depth.shape)
        local = np.stack([(x + .5 - k[0, 2]) / k[0, 0], (y + .5 - k[1, 2]) / k[1, 1], np.ones_like(x)], -1)
        rays = local @ transform[:3, :3].T
        expected = (-.01 - transform[2, 3]) / rays[:, :, 2]
        hits = transform[:3, 3] + rays * expected[:, :, None]
        # The mask depends on the independent ray camera and known table geometry,
        # never on the RTX depth being checked. Erode visible boundaries by3pixels.
        mask = ((np.abs(hits[:, :, 0]) < .85) & (np.abs(hits[:, :, 1]) < .55)
                & (independent_depth > 0) & (np.abs(independent_depth - expected) < 1e-5))
        mask = binary_erosion(mask, iterations=3)
        error = np.abs(data["depth_m"][mask] - expected[mask])
        if not len(error):
            raise RuntimeError(f"No independently visible table pixels in {name}")
        rows[name] = {"status": "PASS" if np.isfinite(error).all() and error.max() < 1e-5 else "FAIL",
                      "pixels": int(mask.sum()), "max_abs_error_m": float(error.max()),
                      "rmse_m": float(np.sqrt(np.mean(error ** 2))), "tolerance_m": 1e-5,
                      "runtime_K": calib["K"]}
        np.savez_compressed(output / f"{name}_plane_qa.npz", mask=mask, expected_depth_m=expected,
                            independent_ray_depth_m=independent_depth)
    result = {"status": "PASS" if all(r["status"] == "PASS" for r in rows.values()) else "FAIL",
              "scope": "actual OVRTX optical-Z SI depth and runtime intrinsics versus known table plane",
              "source_run": str(args.rtx_run.resolve()), "cameras": rows,
              "simulated_d455_noise_validation": "NOT_RUN", "true_cut_surface_validation": "NOT_ASSESSED"}
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
