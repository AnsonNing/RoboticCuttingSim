#!/usr/bin/env python3
"""Produce raw-preserving RGB-aligned native depth using the controlled OVRTX profile."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.rtx_outputs import decode_camera_params
from robotic_cutting_sim.single_view_registration import register_ovrtx_single_view_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    (output / "observations").mkdir()
    prepared = json.loads((args.capture_dir / "prepared.json").read_text())
    receipt = json.loads((args.capture_dir / "runtime_audit.json").read_text())
    versions = receipt["packages"]
    rows = []
    for path in sorted((args.capture_dir / "observations").glob("d455_*_*.npz")):
        source = np.load(path)
        raw_path = args.capture_dir / "camera_debug" / path.name
        raw = np.load(raw_path)
        result = register_ovrtx_single_view_depth(raw["pointcloud_position_raw"], raw["depth_sensor_distance_raw"],
            decode_camera_params(source["camera_params_raw"]), ovrtx_version=versions["ovrtx"],
            ovstage_version=versions["ovstage"], native_model_parameters=prepared["depth_model_parameters"])
        np.savez_compressed(output / "observations" / path.name, rgb=source["rgb"], depth_m=result["depth_m"],
            valid_mask=result["valid_mask"], source_index=result["source_index"],
            timestamp_s=source["timestamp_s"], surface_timestamp_s=source["surface_timestamp_s"],
            camera_params_raw=source["camera_params_raw"], rgb_depth_aligned=True,
            source_correspondence_validated=True, simulated_d455_validated=False,
            physical_d455_calibration_validated=False, noise_calibration_validated=False,
            source="native NVIDIA SingleView raw optical Z; empirical version-gated source correspondence")
        rows.append({"frame": path.stem, "diagnostics": result["diagnostics"],
            "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, raw_path)}})
    if not rows:
        raise ValueError("No native depth captures found")
    write_json(output / "result.json", {"status": "PASS_NATIVE_SOURCE_ASSOCIATION",
        "source": str(args.capture_dir.resolve()), "frames": rows,
        "scope": "Standalone native SingleView to original virtual RGB camera; not live Isaac or physical calibration",
        "native_depth_model_parameters": prepared["depth_model_parameters"], "versions": versions,
        "simulated_d455_validated": False, "source_correspondence_validated": True})
    print(json.dumps({"output": str(output), "frames": len(rows), "source_correspondence_validated": True,
                      "simulated_d455_validated": False}))


if __name__ == "__main__":
    main()
