#!/usr/bin/env python3
"""CPU-only audit of preserved native SingleView samples and RGB registration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.rtx_outputs import decode_camera_params
from robotic_cutting_sim.single_view_registration import analytic_plane_depth, register_native_depth


def error_statistics(measured, reference, mask):
    delta = measured[mask].astype(float) - reference[mask].astype(float)
    if not len(delta):
        return {"samples": 0, "mean_error_m": None, "rmse_m": None, "p95_absolute_m": None}
    return {"samples": len(delta), "mean_error_m": float(delta.mean()),
            "mean_absolute_m": float(np.abs(delta).mean()), "rmse_m": float(np.sqrt(np.mean(delta ** 2))),
            "median_absolute_m": float(np.median(np.abs(delta))),
            "p95_absolute_m": float(np.quantile(np.abs(delta), .95)),
            "p99_absolute_m": float(np.quantile(np.abs(delta), .99)),
            "max_absolute_m": float(np.abs(delta).max()),
            "nonzero_above_1um_samples": int(np.sum(np.abs(delta) > 1e-6)),
            "outliers_above_1cm_samples": int(np.sum(np.abs(delta) > .01))}


def analyze_run(label, source, output, plane):
    destination = new_output(output / label)
    observations = new_output(destination / "observations")
    debug = new_output(destination / "registration_debug")
    rows, arrays = [], {}
    prepared = json.loads((source / "prepared.json").read_text())
    for path in sorted((source / "observations").glob("d455_*_*.npz")):
        payload = np.load(path)
        native_path = source / "camera_debug" / path.name
        native = np.load(native_path)
        camera = decode_camera_params(payload["camera_params_raw"])
        registered = register_native_depth(native["pointcloud_position_raw"], native["depth_sensor_distance_raw"], camera)
        native_distance = native["depth_sensor_distance_raw"].reshape(registered["depth_m"].shape)
        ideal = native["ideal_distance_to_image_plane_raw"].reshape(registered["depth_m"].shape)
        native_valid = np.isfinite(native_distance) & (native_distance >= .52) & (native_distance <= 6.)
        ideal_valid = np.isfinite(ideal) & (ideal > 0)
        row = {"frame": path.stem, "registration": registered["diagnostics"],
            "raw_grid_vs_ideal_rgb_grid": error_statistics(native_distance, ideal, native_valid & ideal_valid),
            "registered_vs_ideal_rgb_grid": error_statistics(registered["depth_m"], ideal,
                                                             registered["valid_mask"] & ideal_valid),
            "native_valid_registered_at_ideal_invalid_pixels": int(np.sum(registered["valid_mask"] & ~ideal_valid)),
            "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, native_path)}}
        xyz = native["pointcloud_position_raw"]
        noisy_source = native_valid & (np.abs(native_distance.astype(float) + xyz[..., 2].astype(float)) > 1e-6)
        selected = registered["source_index"][registered["valid_mask"]]
        row["native_scalar_vs_geometric_cloud_differences"] = {
            "input_samples_above_1um": int(noisy_source.sum()),
            "selected_samples_above_1um": int(noisy_source.reshape(-1)[selected].sum()),
            "note": "Not filtered; native outliers and z-buffer/sample collisions remain explicit"}
        if plane:
            equation = plane["plane_world_equation"]
            expected = analytic_plane_depth(camera, equation["normal"], equation["offset"])
            mask = registered["valid_mask"] & (expected > 0)
            row["analytic_plane"] = error_statistics(registered["depth_m"], expected, mask)
            row["analytic_plane"]["p95_tolerance_m"] = .001
            row["analytic_plane"]["status"] = "PASS" if row["analytic_plane"]["p95_absolute_m"] < .001 else "FAIL"
            np.savez_compressed(debug / (path.stem + "_plane.npz"), expected_depth_m=expected,
                                difference_m=np.where(mask, registered["depth_m"] - expected, 0), mask=mask)
        np.savez_compressed(observations / path.name, rgb=payload["rgb"], depth_m=registered["depth_m"],
            valid_mask=registered["valid_mask"], timestamp_s=payload["timestamp_s"],
            surface_timestamp_s=payload["surface_timestamp_s"],
            camera_params_raw=payload["camera_params_raw"], rgb_depth_aligned=False,
            same_virtual_camera_pixel_grid=True, registration_validated=False,
            registration_scope="native XYZ direction associates raw measured Z with original virtual RGB camera",
            simulated_d455_validated=False)
        np.savez_compressed(debug / path.name, source_index=registered["source_index"],
                            native_distance_m=native_distance)
        rows.append(row)
        arrays[path.stem] = {"registered": registered["depth_m"], "native": native_distance.copy()}
    if not rows:
        raise RuntimeError(f"No native D455 captures in {source}")
    return {"source": str(source.resolve()), "depth_model_parameters": prepared.get("depth_model_parameters"),
            "frames": rows}, arrays


def compare_noise_controls(off, on):
    comparison = {}
    for camera in ("d455_0", "d455_1"):
        keys = sorted(set(k for k in off if k.startswith(camera)) & set(on))
        if not keys:
            raise ValueError("Noise OFF/ON controls need matching camera/frame names")
        values = {}
        for kind in ("native", "registered"):
            a = np.array([off[k][kind] for k in keys])
            b = np.array([on[k][kind] for k in keys])
            ma = np.isfinite(a) & (a >= .52) & (a <= 6.)
            mb = np.isfinite(b) & (b >= .52) & (b <= 6.)
            values[kind] = {"off_valid_samples": int(ma.sum()), "on_valid_samples": int(mb.sum()),
                "validity_changed_samples": int(np.sum(ma != mb)),
                "on_minus_off_common_samples": error_statistics(b, a, ma & mb), "temporal": {}}
            for label, sequence, mask in (("off", a, ma), ("on", b, mb)):
                common = mask[1:] & mask[:-1]
                values[kind]["temporal"][label] = {
                    "validity_changed_samples": int(np.sum(mask[1:] != mask[:-1])),
                    "consecutive_common_samples": error_statistics(sequence[1:], sequence[:-1], common)}
        comparison[camera] = values
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=CAPTURE_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plane-fixture", type=Path)
    parser.add_argument("--noise-off-label", default="off")
    parser.add_argument("--noise-on-label", default="on")
    args = parser.parse_args()
    output = new_output(args.output_dir)
    plane = json.loads(args.plane_fixture.read_text()) if args.plane_fixture else None
    runs, arrays = {}, {}
    for item in args.run:
        label, path = item.split("=", 1)
        if not label or label in runs or Path(label).name != label:
            raise ValueError("Use unique simple LABEL=PATH capture labels")
        runs[label], arrays[label] = analyze_run(label, Path(path), output, plane)
    control = (compare_noise_controls(arrays[args.noise_off_label], arrays[args.noise_on_label])
               if args.noise_off_label in arrays and args.noise_on_label in arrays else None)
    preserve = all(frame["registration"]["native_measurement_values_preserved_exactly"]
                   for run in runs.values() for frame in run["frames"])
    plane_pass = (all(frame["analytic_plane"]["status"] == "PASS"
                     for frame in runs[args.noise_off_label]["frames"])
                  if plane is not None and args.noise_off_label in runs else None)
    result = {"status": "BLOCKED" if plane_pass is False else "PASS_NATIVE_SAMPLE_PRESERVATION",
        "scope": "CPU registration audit of actual native NVIDIA SingleView captures",
        "runs": runs, "noise_controls": control, "native_values_exactly_preserved": preserve,
        "noise_off_analytic_plane_registration_pass": plane_pass,
        "simulated_d455_validated": False, "physical_d455_calibration_validated": False,
        "registration_status": "BLOCKED_AT_1MM_P95_PLANE_CRITERION" if plane_pass is False else "PROTOTYPE",
        "limitations": ["Native XYZ is used only for pixel directions, never as measured depth",
            "No consistency rejection, ideal replacement, smoothing, filling or synthetic noise",
            "Noise affects native validity strongly; metric-depth behavior requires explicit control statistics",
            "Target RGB is the original virtual depth camera, not the separate native D455 color imager"]}
    result["implementation_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).resolve().parents[1] / "src/robotic_cutting_sim/single_view_registration.py")}
    write_json(output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "noise_controls")}, indent=2))
    return int(not preserve or plane_pass is False)


if __name__ == "__main__":
    raise SystemExit(main())
