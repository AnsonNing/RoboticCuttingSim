#!/usr/bin/env python3
"""Audit prefrozen H0/H1 native sample association; never fit depth or discard errors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.rtx_outputs import decode_camera_params
from robotic_cutting_sim.single_view_controls import analytic_control, feature_acceptance
from robotic_cutting_sim.single_view_registration import register_native_depth


def stats(values):
    x = np.asarray(values, dtype=float)
    return {"samples": int(x.size), "mean_m": float(x.mean()) if x.size else None,
        "rmse_m": float(np.sqrt(np.mean(x*x))) if x.size else None,
        "p95_absolute_m": float(np.quantile(abs(x), .95)) if x.size else None,
        "max_absolute_m": float(np.max(abs(x))) if x.size else None,
        "above_1um": int(np.sum(abs(x) > 1e-6)), "above_1cm": int(np.sum(abs(x) > .01))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture", type=Path)
    p.add_argument("noise_off", type=Path)
    p.add_argument("noise_on", type=Path)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    output = new_output(args.output_dir)
    implementation = [Path(__file__), Path("src/robotic_cutting_sim/single_view_registration.py"),
                      Path("src/robotic_cutting_sim/single_view_controls.py")]
    snapshots = output / "implementation"
    snapshots.mkdir()
    implementation_hashes = {}
    for path in implementation:
        data = path.read_bytes()
        (snapshots / path.name).write_bytes(data)
        implementation_hashes[str(path)] = hashlib.sha256(data).hexdigest()
    fixture = json.loads((args.fixture / "fixture.json").read_text())
    protocol_bytes = (args.fixture / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    assert hashlib.sha256(protocol_bytes).hexdigest() == fixture["protocol_sha256"]
    criteria = protocol["criteria"]
    rows, arrays, indices, input_hashes = [], {}, {}, {}
    for label, run in (("off", args.noise_off), ("on", args.noise_on)):
        prepared = json.loads((run / "prepared.json").read_text())
        runtime = json.loads((run / "runtime_audit.json").read_text())
        for name in ("ovrtx", "ovstage", "isaacsim"):
            if name in protocol["runtime_profile"]:
                assert runtime["packages"][name] == protocol["runtime_profile"][name]
        if "omni.sensors.nv.camera" in protocol["runtime_profile"]:
            assert runtime["extensions"]["omni.sensors.nv.camera"] == protocol["runtime_profile"]["omni.sensors.nv.camera"]
            assert runtime["kit_version"].startswith(protocol["runtime_profile"]["kit_major_minor"])
        assert len(prepared["time_s"]) == len(fixture["frames"])
        for camera_name in ("d455_0", "d455_1"):
            for frame_index, case in enumerate(fixture["frames"]):
                name = f"{camera_name}_{frame_index:04d}.npz"
                native_path, payload_path = run / "camera_debug" / name, run / "observations" / name
                native, payload = np.load(native_path), np.load(payload_path)
                assert float(payload["surface_timestamp_s"]) == case["time_s"]
                camera = (decode_camera_params(payload["camera_params_raw"]) if "camera_params_raw" in payload else
                          json.loads(str(payload["camera_calibration_json"].item())))
                assert np.allclose(np.asarray(camera["T_world_camera"])[:3, 3], fixture["cameras"][camera_name][1], atol=1e-6, rtol=0)
                plane, expected, box = analytic_control(camera, case, protocol["scene"])
                raw = native["depth_sensor_distance_raw"].squeeze(-1)
                raw_valid = np.isfinite(raw) & (raw >= .52) & (raw <= 6.)
                input_hashes[str(native_path)] = hashlib.sha256(native_path.read_bytes()).hexdigest()
                input_hashes[str(payload_path)] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
                for hypothesis, config in protocol["candidates"].items():
                    reg = register_native_depth(native["pointcloud_position_raw"], raw, camera,
                        pointcloud_pixel_offset=config["pointcloud_to_scalar_rgb_pixel_offset"])
                    valid, depth = reg["valid_mask"], reg["depth_m"]
                    known = np.isfinite(expected) & (expected >= .52) & (expected <= 6.)
                    plane_mask = valid & known & ~box
                    plane_error, full_error = stats(depth[plane_mask]-plane[plane_mask]), stats(depth[valid & known]-expected[valid & known])
                    coverage = float(np.sum(valid & known) / np.sum(known))
                    checks = {"exact_native_values": reg["diagnostics"]["native_measurement_values_preserved_exactly"],
                        "plane_p95": plane_error["p95_absolute_m"] is not None and plane_error["p95_absolute_m"] < criteria["noise_off_plane_p95_absolute_error_m"],
                        "coverage": coverage >= criteria["noise_off_minimum_valid_coverage"]}
                    feature = None
                    if case.get("box"):
                        feature = feature_acceptance(payload["rgb"], depth, valid, plane, box, protocol["feature_classification"])
                        checks.update({"feature_rgb_geometry": feature["rgb_vs_analytic_mask_iou"] >= criteria["feature_rgb_vs_analytic_mask_minimum_iou"],
                            "feature_scanlines": feature["scanlines"] >= criteria["minimum_feature_scanlines"],
                            "feature_registration": feature["boundary_error_p95_px"] is not None and feature["boundary_error_p95_px"] <= criteria["noise_off_feature_p95_error_px"]})
                    row = {"label": label, "camera": camera_name, "frame_index": frame_index,
                        "case_id": case["id"], "split": case["split"], "repeat": case["repeat"], "hypothesis": hypothesis,
                        "plane_error": plane_error, "full_scene_error": full_error, "known_geometry_valid_coverage": coverage,
                        "raw_valid_coverage": float(raw_valid.mean()), "feature": feature,
                        "registration": reg["diagnostics"], "checks": checks,
                        "status": ("PASS" if all(checks.values()) else "FAIL") if label == "off" else "NOISE_DIAGNOSTIC_ONLY"}
                    rows.append(row)
                    arrays[(label, camera_name, frame_index, hypothesis)] = (depth.copy(), valid.copy())
                    if hypothesis == "H1":
                        indices[(label, camera_name, frame_index)] = reg["source_index"].copy()
                        folder = output / label
                        folder.mkdir(exist_ok=True)
                        np.savez_compressed(folder / name, rgb=payload["rgb"], depth_m=depth, valid_mask=valid,
                            source_index=reg["source_index"], analytic_scene_depth_m=expected, analytic_box_mask=box,
                            camera_calibration_json=json.dumps(camera), timestamp_s=payload["timestamp_s"],
                            simulated_d455_validated=False, registration_validated=False)
                arrays[(label, camera_name, frame_index, "raw")] = (raw.copy(), raw_valid.copy())
    noise = []
    for camera_name in ("d455_0", "d455_1"):
        for index, case in enumerate(fixture["frames"]):
            for kind in ("raw", "H0", "H1"):
                off, ma = arrays[("off", camera_name, index, kind)]
                on, mb = arrays[("on", camera_name, index, kind)]
                noise.append({"camera": camera_name, "frame_index": index, "case_id": case["id"], "kind": kind,
                    "off_valid_samples": int(ma.sum()), "on_valid_samples": int(mb.sum()),
                    "validity_changed_samples": int(np.sum(ma != mb)), "common_on_minus_off": stats(on[ma & mb]-off[ma & mb])})
            raw_off, off_mask = arrays[("off", camera_name, index, "raw")]
            raw_on, on_mask = arrays[("on", camera_name, index, "raw")]
            common = off_mask & on_mask
            nonzero = common & (abs(raw_on.astype(float)-raw_off.astype(float)) > 1e-6)
            selected = indices[("on", camera_name, index)]
            chosen = selected[selected >= 0]
            chosen_common = chosen[common.reshape(-1)[chosen]]
            chosen_nonzero = int(np.sum(nonzero.reshape(-1)[chosen]))
            noise.append({"camera": camera_name, "frame_index": index, "case_id": case["id"],
                "kind": "raw_noise_retention_ledger", "input_nonzero_common_native_samples": int(nonzero.sum()),
                "selected_nonzero_native_samples": chosen_nonzero,
                "unselected_nonzero_native_samples": int(nonzero.sum())-chosen_nonzero,
                "selected_source_on_minus_off": stats(raw_on.reshape(-1)[chosen_common]-raw_off.reshape(-1)[chosen_common]),
                "selection": "finite forward native direction, output image bounds, measured-Z z-buffer; no depth-error/consistency rejection",
                "error_based_rejections": 0})
    temporal = []
    for camera_name in ("d455_0", "d455_1"):
        for label in ("off", "on"):
            for kind in ("raw", "H1"):
                for a, b in ((0, 1), (1, 2)):
                    x, mx = arrays[(label, camera_name, a, kind)]
                    y, my = arrays[(label, camera_name, b, kind)]
                    temporal.append({"camera": camera_name, "label": label, "kind": kind, "frames": [a, b],
                        "validity_changed_samples": int(np.sum(mx != my)), "common_temporal_difference": stats(y[mx & my]-x[mx & my])})
    summaries = {}
    for h in protocol["candidates"]:
        summaries[h] = {}
        for split in ("development", "held_out"):
            selected = [r for r in rows if r["label"] == "off" and r["hypothesis"] == h and r["split"] == split]
            summaries[h][split] = {"status": "PASS" if all(r["status"] == "PASS" for r in selected) else "FAIL",
                "frames": len(selected), "failed_frames": [{"case_id": r["case_id"], "camera": r["camera"],
                    "checks": r["checks"]} for r in selected if r["status"] != "PASS"]}
    promotion = all(summaries["H1"][s]["status"] == "PASS" for s in ("development", "held_out"))
    is_kit = "isaacsim" in protocol["runtime_profile"]
    result = {"protocol_id": protocol["protocol_id"], "protocol_sha256": fixture["protocol_sha256"],
        "status": "PASS_SOURCE_CORRESPONDENCE_CONTROLS" if promotion else "BLOCKED",
        "source_correspondence_candidate_promoted": promotion, "summaries": summaries,
        "native_samples_exactly_preserved": all(r["registration"]["native_measurement_values_preserved_exactly"] for r in rows),
        "simulated_d455_validated": False, "physical_d455_calibration_validated": False,
        "scope": ("Actual Isaac Sim6.2/Kit native camera analytic controls; no dynamics episode or hardware calibration" if is_kit else
                  "Standalone native OVRTX analytic controls, not live Isaac or hardware calibration"),
        "rows": rows, "noise": noise, "static_temporal": temporal, "input_sha256": input_hashes,
        "implementation_sha256": implementation_hashes,
        "implementation_snapshot": "Byte snapshots taken before analysis; imported helper code held unchanged during this audit"}
    write_json(output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "noise", "static_temporal", "input_sha256", "implementation_sha256")}, indent=2))
    return int(not promotion)


if __name__ == "__main__":
    raise SystemExit(main())
