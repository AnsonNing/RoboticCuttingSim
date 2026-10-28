#!/usr/bin/env python3
"""Check prototype optical-Z clipping against independent fronto-parallel planes.

This tests the actual ideal Warp renderer. It is not stereo, noise, calibration
or a validation of the physical D405's resolution-dependent Min-Z.
"""
import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
import warp as wp

from robotic_cutting_sim.camera import capture
from robotic_cutting_sim.camera_rig import make_rig
from robotic_cutting_sim.recording import ROOT, manifest, new_output, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = new_output(args.output_dir)
    source = [Path(__file__), ROOT/"src/robotic_cutting_sim/camera.py",
              ROOT/"src/robotic_cutting_sim/camera_rig.py", ROOT/"configs/wrist_camera_rig_v1.json"]
    implementation = output/"implementation"
    implementation.mkdir()
    for path in source:
        shutil.copyfile(path, implementation/path.name)
    record = manifest(vars(args)|{"output_dir": str(output)}, "ideal_d405_optical_z_control")
    record["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    write_json(output/"manifest.json", record)
    spec = make_rig("TRIM-01").specs["wrist_d405_0"]
    calibration = {**spec, "T_world_camera": np.eye(4).tolist(), "rgb_far_m": 20.}
    wp.init()
    rows = []
    with wp.ScopedDevice(args.device):
        triangles = wp.array([0, 1, 2, 0, 2, 3], dtype=int)
        for z, should_be_valid in ((.06, False), (.08, True), (.49, True), (.51, False)):
            vertices = wp.array([[-2, -2, z], [2, -2, z], [2, 2, z], [-2, 2, z]], dtype=wp.vec3)
            rgb, depth = capture(vertices, triangles, calibration, args.device)
            valid = depth > 0
            error = float(np.max(np.abs(depth[valid]-z))) if valid.any() else None
            passed = bool(valid.all() and error <= 1e-6) if should_be_valid else bool(not valid.any())
            background = np.array([.12, .15, .19])*255
            rgb_retained = bool(np.any(np.abs(rgb.astype(float)-background) > 1.))
            passed = passed and rgb_retained
            rows.append({"plane_optical_z_m": z, "expected_all_pixels_valid": should_be_valid,
                         "valid_pixels": int(valid.sum()), "pixels": int(valid.size),
                         "max_absolute_depth_error_m": error, "rgb_geometry_retained_outside_depth_range": rgb_retained,
                         "status": "PASS" if passed else "FAIL"})
            np.savez_compressed(output/f"plane_{int(z*1000):03d}mm.npz", rgb=rgb, depth_m=depth, valid_mask=valid)
        vertices = wp.array([[-2, -2, .06], [0, -2, .06], [0, 2, .06], [-2, 2, .06],
                             [-2, -2, .2], [2, -2, .2], [2, 2, .2], [-2, 2, .2]], dtype=wp.vec3)
        triangles = wp.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], dtype=int)
        colors = wp.array([[1., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 0., 1.]], dtype=wp.vec3)
        rgb, depth = capture(vertices, triangles, calibration, args.device, colors)
        midpoint = spec["width"]//2
        left, right = depth[:, :midpoint], depth[:, midpoint:]
        rgb_visible = bool(rgb[:, :midpoint, 0].mean() > rgb[:, :midpoint, 2].mean()
                           and rgb[:, midpoint:, 2].mean() > rgb[:, midpoint:, 0].mean())
        passed = bool(not np.any(left) and np.all(right > 0) and np.max(abs(right-.2)) < 1e-6 and rgb_visible)
        rows.append({"control": "near_6cm_foreground_occludes_20cm_target_in_left_half",
                     "left_valid_pixels": int(np.count_nonzero(left)), "right_valid_pixels": int(np.count_nonzero(right)),
                     "rgb_retains_foreground_and_background": rgb_visible, "status": "PASS" if passed else "FAIL"})
        np.savez_compressed(output/"near_occluder.npz", rgb=rgb, depth_m=depth, valid_mask=depth > 0)
    result = {"status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
              "sensor_model": "ideal_d405_prototype", "near_m": spec["near_m"], "far_m": spec["far_m"],
              "checks": rows, "physical_d405_min_z_validated": False,
              "stereo_noise_physical_calibration": "NOT_RUN"}
    write_json(output/"result.json", result)
    print(result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
