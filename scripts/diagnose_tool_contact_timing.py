#!/usr/bin/env python3
"""Read-only timing diagnosis of native contact points versus free-tool poses.

Preserves the original post-step classifier and its failures. PRE/MID results
are separately labelled diagnostics using the SAME1um margin/positive-load
threshold. No particle, geometry, impulse, history or benchmark score is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.scene_frames import validate_pose_xyzw

MARGIN_M = 1e-6
LOADED_MIN_N = 1e-6
REFERENCES = {
    "physx_callback_sequence": "https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Simulation.html#callback-sequence",
    "isaacsim62_raw_contacts": "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/py/source/extensions/isaacsim.sensors.experimental.physics/docs/index.html"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def under(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def point_excess(points_world_m, tool_pose_xyzw, half_extents_m):
    pose = validate_pose_xyzw(tool_pose_xyzw)
    local = (np.asarray(points_world_m) - pose[:3]) @ Rotation.from_quat(pose[3:]).as_matrix()
    return np.maximum(np.abs(local) - np.asarray(half_extents_m), 0.)


def midpoint_pose(before, after):
    before, after = validate_pose_xyzw(before), validate_pose_xyzw(after)
    q = after[3:] * (1 if before[3:] @ after[3:] >= 0 else -1)
    q = before[3:] + q
    return np.r_[(before[:3] + after[:3]) / 2, q / np.linalg.norm(q)]


def quantiles(values):
    return None if not len(values) else {name: float(value) for name, value in zip(
        ("p50_m", "p95_m", "p99_m", "maximum_m"), np.quantile(values, [.5, .95, .99, 1.]))}


def diagnose(rows, spec, hand_path, *, phases=None):
    half = np.asarray(spec["handle_half_extents_m"])
    counts = {key: 0 for key in ("recorded", "post", "pre", "mid", "finger_link")}
    distances = {key: [] for key in ("post", "pre", "mid")}
    outside_axes = {key: np.zeros(3) for key in distances}
    records, other_pairs, selected_count = [], 0, 0
    for previous, row in zip(rows[:-1], rows[1:]):
        if row.get("phase") == "pregrasp_settling" or (phases and row.get("phase") not in phases):
            continue
        selected_count += 1
        sensor, now = row["contacts"], float(row["native_time_s"])
        tolerance = max(1e-7, 4 * np.finfo(np.float32).eps * max(1., abs(now)))
        positions, roles, outside_bodies = [], [], 0
        source_valid = all(sensor.get(k) is True for k in ("sensor_valid", "sensor_fresh", "new_physics_step"))
        for contact in sensor.get("contacts", []):
            a, b, tool = contact["body0"], contact["body1"], "/World/Tools/CuttingTool_0"
            other = b if under(a, tool) and not under(b, tool) else a if under(b, tool) and not under(a, tool) else None
            if other is None:
                continue
            role = 0 if under(other, hand_path + "/panda_leftfinger") else 1 if under(other, hand_path + "/panda_rightfinger") else -1
            dt = float(contact["dt_s"])
            fresh = source_valid and dt > 0 and abs(now - contact["time_s"]) <= tolerance
            if not fresh or np.linalg.norm(contact["raw_impulse_world_ns"]) / dt <= LOADED_MIN_N:
                continue
            if role < 0:
                outside_bodies += 1
            else:
                positions.append(contact["position_world_m"]); roles.append(role)
        other_pairs += outside_bodies
        stamp = {"native_time_s": now, "phase": row["phase"], "sensor_valid_and_fresh": source_valid,
                 "loaded_nonfinger_pairs": outside_bodies,
                 "recorded_bilateral_loaded_handle": bool(sensor.get("bilateral_loaded_handle_contact", False))}
        counts["recorded"] += stamp["recorded_bilateral_loaded_handle"]
        poses = {"post": row["tool_body_pose_xyzw"], "pre": previous["tool_body_pose_xyzw"],
                 "mid": midpoint_pose(previous["tool_body_pose_xyzw"], row["tool_body_pose_xyzw"])}
        roles = np.asarray(roles)
        stamp["bilateral_loaded_finger_links"] = len(np.unique(roles)) == 2
        counts["finger_link"] += stamp["bilateral_loaded_finger_links"]
        for key, pose in poses.items():
            excess = point_excess(positions, pose, half) if positions else np.empty((0, 3))
            maximum = excess.max(axis=1) if positions else np.empty(0)
            inside = maximum <= MARGIN_M
            stamp[key + "_bilateral_loaded_handle"] = len(np.unique(roles[inside])) == 2
            stamp[key + "_maximum_point_excess_m"] = float(maximum.max(initial=0.))
            counts[key] += stamp[key + "_bilateral_loaded_handle"]
            distances[key].extend(maximum.tolist())
            outside_axes[key] = np.maximum(outside_axes[key], excess.max(axis=0, initial=0.))
        stamp["pre_pose_time_s"] = float(previous["native_time_s"])
        stamp["rigid_interval_s"] = now - stamp["pre_pose_time_s"]
        records.append(stamp)
    if not records:
        raise ValueError("No selected actual native rigid samples")
    return {"status": "DIAGNOSTIC_ONLY_NO_RECLASSIFICATION", "samples": selected_count,
        "unchanged_region_margin_m": MARGIN_M, "unchanged_loaded_force_minimum_n": LOADED_MIN_N,
        "bilateral_fraction": {key: value / selected_count for key, value in counts.items()},
        "loaded_nonfinger_pairs": other_pairs,
        "point_outside_handle_distance": {key: quantiles(value) for key, value in distances.items()},
        "maximum_outside_axis_xyz_m": {key: value.tolist() for key, value in outside_axes.items()},
        "records": records, "official_references": REFERENCES,
        "interpretation": "Compare detection-time contact geometry against measured pre-step and integrated post-step body poses; event timing is an empirical pinned-backend finding, not a universal CCD rule",
        "threshold_or_benchmark_score_changed": False, "physical_grasp_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--phase", action="append", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    setup_path = args.episode / "setup.json"
    setup = json.loads(setup_path.read_text())
    trace = args.episode / "physical_grasp_trace_00.jsonl"
    with trace.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = diagnose(rows, setup["physical_tool"], setup["hand_prim"], phases=args.phase)
    records = result.pop("records")
    (args.output_dir / "records.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
    result.update(episode=str(args.episode.resolve()), script_sha256=digest(Path(__file__)),
                  input_sha256={name: digest(args.episode / name) for name in ("setup.json", "manifest.json", "physical_grasp_trace_00.jsonl")})
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "bilateral_fraction": result["bilateral_fraction"]}))


if __name__ == "__main__":
    main()
