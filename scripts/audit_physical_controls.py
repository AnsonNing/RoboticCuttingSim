#!/usr/bin/env python3
"""Independently recompute frozen grasp-control screens from actual raw traces."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_physical_tool import digest, runtime_log_audit, under

TOOL = "/World/Tools/CuttingTool_0"


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def relative(hand, body):
    h, b = np.asarray(hand), np.asarray(body)
    hr = Rotation.from_quat(h[..., 3:])
    return np.c_[hr.inv().apply(b[..., :3] - h[..., :3]), (hr.inv() * Rotation.from_quat(b[..., 3:])).as_quat()]


def pose_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(b[..., :3] - a[..., :3], axis=-1), (Rotation.from_quat(a[..., 3:]).inv() * Rotation.from_quat(b[..., 3:])).magnitude()


def native_sample(row, half, dt, geometry_profile):
    sensor, now = row["contacts"], row["native_time_s"]
    frame = sensor.get("contact_geometry_frame", "LEGACY_POST_STEP_TOOL_POSE")
    pose = np.asarray(row["tool_body_pose_xyzw"])
    errors = []
    if geometry_profile and frame != geometry_profile:
        errors.append("Wrong frozen geometry profile")
    if frame == "PRE_STEP_MEASURED_TOOL_POSE":
        geometry = np.asarray(sensor.get("contact_geometry_tool_pose_xyzw"))
        if (not np.array_equal(geometry, row.get("tool_body_pose_before_step_xyzw")) or
                abs(sensor.get("contact_geometry_time_s", -1) + dt - now) > 1e-9):
            errors.append("Missing actual PRE geometry/time")
    else:
        geometry = pose
    rotation = Rotation.from_quat(geometry[3:])
    loads = np.zeros(2)
    unexpected, outside = 0, 0
    age_tolerance = max(1e-7, 4 * np.finfo(np.float32).eps * max(1., abs(now)))
    for contact in sensor.get("contacts", []):
        a, b = contact["body0"], contact["body1"]
        if under(a, TOOL) == under(b, TOOL):
            continue
        other = b if under(a, TOOL) else a
        valid = (abs(contact["time_s"] - now) <= age_tolerance and
                 abs(contact["dt_s"] - dt) <= max(1e-7, dt * 1e-4))
        if not valid:
            errors.append("Stale raw contact impulse")
            continue
        force = float(np.linalg.norm(contact["raw_impulse_world_ns"]) / contact["dt_s"])
        local = rotation.inv().apply(np.asarray(contact["position_world_m"]) - geometry[:3])
        handle = bool(np.all(np.abs(local) <= half + 1e-6))
        finger = 0 if other.endswith("/panda_leftfinger") else 1 if other.endswith("/panda_rightfinger") else None
        if finger is None:
            unexpected += int(force > .05)
        elif handle:
            loads[finger] += force
        else:
            outside += int(force > .05)
    tensor = sensor.get("tensor_forces", {})
    valid = (all(sensor.get(k) is True for k in ("sensor_valid", "sensor_fresh", "new_physics_step"))
             and abs(sensor.get("sensor_time_s", -1) - now) <= age_tolerance
             and tensor.get("status") == "VALID_NATIVE_NORMAL_AND_FRICTION_DATA"
             and not tensor.get("capacity_may_be_reached", False) and not errors)
    penetration = max([max(0., -x["separation_m"]) for x in tensor.get("records", []) if x.get("separation_m") is not None] or [0.])
    return {"time": now, "hand": row["hand_pose_xyzw"], "tool": pose, "loads": loads,
            "velocity": np.r_[row["tool_linear_velocity_world_m_s"], row["tool_angular_velocity_world_rad_s"]],
            "fingers": np.r_[row["finger_q_m"], row["finger_qdot_m_s"]],
            "valid": bool(valid), "unexpected": unexpected, "outside": outside, "penetration": penetration,
            "free": len(sensor.get("contacts", [])) == 0, "errors": errors,
            "force": np.asarray(tensor.get("total_filtered_force_world_on_tool_n", [np.nan] * 3))}


def load_samples(directory, result, protocol):
    interval = result["evaluation_interval"]
    start, end = interval["raw_trace_start_s"], interval["raw_trace_end_s"]
    path = directory / result["raw_trace"]
    samples = []
    if result["runtime"] == "isaac_sim":
        for row in jsonl(path):
            if start + 1e-7 < row["native_time_s"] <= end + 1e-7:
                samples.append(native_sample(row, np.asarray(result["tool_spec"]["handle_half_extents_m"]),
                    protocol["rigid_dt_s"], protocol.get("native_contact_geometry_profile")))
    else:
        metadata = read(path.with_suffix(".json"))
        with np.load(path, allow_pickle=False) as archive:
            raw = {k: archive[k] for k in archive.files}
        bodies = np.asarray(metadata["shape_body"])
        tool, fingers = metadata["tool_body_index"], [metadata["left_finger_body_index"], metadata["right_finger_body_index"]]
        handles = set(metadata["handle_shape_indices"])
        for i, time in enumerate(raw["rigid_time_s"]):
            if not start + 1e-7 < time <= end + 1e-7:
                continue
            loads, force = np.zeros(2), np.zeros(3)
            unexpected, count, penetration = 0, 0, 0.
            for j in range(raw["contact_offsets"][i], raw["contact_offsets"][i + 1]):
                a, b = int(raw["shape0"][j]), int(raw["shape1"][j])
                if min(a, b) < 0 or (bodies[a] == tool) == (bodies[b] == tool):
                    continue
                on0 = bodies[a] == tool
                own, other = (a, bodies[b]) if on0 else (b, bodies[a])
                f = raw["contact_wrench_on_shape0_world"][j, :3] * (1 if on0 else -1)
                count += 1
                force += f
                if own in handles and other in fingers:
                    loads[fingers.index(other)] += abs(float(f @ raw["normal_world"][j]))
                elif np.linalg.norm(f) > .05:
                    unexpected += 1
                penetration = max(penetration, -float(raw["distance_m"][j]))
            samples.append({"time": float(time), "hand": raw["hand_pose_xyzw"][i], "tool": raw["tool_body_pose_xyzw"][i],
                "velocity": raw["tool_velocity_com_world"][i], "fingers": np.r_[raw["finger_q"][i], raw["finger_qdot"][i]],
                "loads": loads, "force": force, "valid": True, "unexpected": unexpected, "outside": 0,
                "free": count == 0, "penetration": max(0., penetration)})
    return samples, path


def measurements(samples, initial, spec, protocol, duration):
    if len(samples) < 2:
        raise ValueError("Missing actual interval samples")
    limits = protocol["measurement"]
    times = np.array([s["time"] for s in samples])
    hand, body = np.array([s["hand"] for s in samples]), np.array([s["tool"] for s in samples])
    poses = relative(hand, body)
    reference = relative(np.asarray(initial["hand_pose"])[None], np.asarray(initial["tool_body_pose_xyzw"])[None])[0]
    p, r = pose_error(reference, poses)
    last = times[1:] >= times[-1] - limits["stable_window_s"] - 1e-7
    speed = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) / np.diff(times)
    angular = (Rotation.from_quat(poses[:-1, 3:]).inv() * Rotation.from_quat(poses[1:, 3:])).magnitude() / np.diff(times)
    blocks, block = [], []
    for i, sample in enumerate(samples):
        if sample["free"]:
            block.append(i)
        elif block:
            blocks.append(block); block = []
    if block:
        blocks.append(block)
    free = max(blocks, key=len) if blocks else []
    free_time = float(times[free[-1]] - times[free[0]]) if len(free) > 1 else 0.
    com = body[:, :3] + Rotation.from_quat(body[:, 3:]).apply(spec["com_tool_m"])
    acceleration = (float(2 * np.polyfit(times[free] - times[free[0]], com[free, 2], 2)[0])
                    if free_time >= limits["freeflight_minimum_duration_s"] else None)
    return {"samples": len(samples), "complete_2ms_sampling": bool(len(samples) == round(duration / protocol["rigid_dt_s"])
                and np.all(np.diff(times) > 0) and np.max(np.diff(times)) <= protocol["rigid_dt_s"] * 1.01),
            "finite": bool(np.isfinite(np.c_[times, hand, body, np.array([s["loads"] for s in samples]), np.array([s["force"] for s in samples]),
                np.array([s["velocity"] for s in samples]), np.array([s["fingers"] for s in samples])]).all()),
            "all_fresh_valid": all(s["valid"] for s in samples), "max_relative_translation_m": float(p.max()),
            "max_relative_rotation_rad": float(r.max()), "maximum_relative_linear_speed_m_s": float(speed[last].max()),
            "maximum_relative_angular_speed_rad_s": float(angular[last].max()),
            "bilateral_loaded_handle_fraction": float(np.mean([np.all(np.asarray(s["loads"]) > limits["minimum_loaded_finger_force_n"]) for s in samples])),
            "maximum_penetration_m": float(max(s["penetration"] for s in samples)),
            "unintended_loaded_support_steps": sum(s["unexpected"] > 0 for s in samples),
            "finger_points_outside_handle_steps": sum(s["outside"] > 0 for s in samples),
            "mean_contact_force_last_window_world_n": np.mean([s["force"] for s in samples if s["time"] >= times[-1] - limits["stable_window_s"] - 1e-7], axis=0).tolist(),
            "freeflight_duration_s": free_time, "freeflight_acceleration_z_m_s2": acceleration,
            "downward_displacement_m": float(max(0., initial["tool_body_pose_xyzw"][2] - body[:, 2].min()))}


def audit_index(index_path):
    index = read(index_path)
    parent = index_path.parent
    protocol_path = parent / index["protocol"]["path"]
    protocol = read(protocol_path)
    if digest(protocol_path) != index["protocol"]["sha256"]:
        raise ValueError("Frozen protocol hash mismatch")
    limits, cases = protocol["measurement"], {}
    for key, entry in index["cases"].items():
        directory = parent / entry["directory"]
        result = read(directory / "result.json")
        name = key.rsplit("/", 1)[-1]
        source_mismatch = [p for p, h in result["source_sha256"].items()
                           if not (directory / "implementation" / Path(p).name).is_file()
                           or digest(directory / "implementation" / Path(p).name) != h]
        identity = (entry["config"] == protocol["cases"][name] == result["case_config"] and
                    entry["source_sha256"] == result["source_sha256"] and not source_mismatch)
        basic = {"frozen_source_and_config_match": bool(identity),
                 "recorded_material_fingerprints_equal": result.get("material_initial") is not None and result.get("material_initial") == result.get("material_final"),
                 "no_native_tensor_view_invalidations": not runtime_log_audit(directory)["status"].startswith("FAIL")}
        row = {"directory": str(directory.resolve()), "recorded_execution_status": result["status"],
               "source_mismatches": source_mismatch, "checks": basic,
               "input_sha256": {"result.json": digest(directory / "result.json"), "manifest.json": digest(directory / "manifest.json")}}
        if "evaluation_interval" not in result or not result.get("raw_trace"):
            row["status"] = "FAIL_PREGRASP_OR_TRANSFER_NO_PERTURBATION"
            cases[key] = row
            continue
        samples, raw = load_samples(directory, result, protocol)
        metrics = measurements(samples, result["initial_state"], result["tool_spec"], protocol, protocol["cases"][name]["duration_s"])
        controls = [r for r in jsonl(directory / "control_trace.jsonl") if r["phase"] == "perturbation"]
        q = np.asarray([r["state"]["q"][0] for r in controls])
        metrics["complete_20ms_arm_samples"] = len(q) == round(protocol["cases"][name]["duration_s"] / protocol["control_dt_s"])
        metrics["finite"] &= bool(np.isfinite(q).all())
        metrics["final_arm_deflection_rad"] = (q[-1] - np.asarray(result["initial_state"]["q"])[0]).tolist() if len(q) else None
        row["metrics"] = metrics
        checks = {**basic, **{k: metrics[k] for k in ("complete_2ms_sampling", "complete_20ms_arm_samples", "finite", "all_fresh_valid")}}
        if name in ("baseline_hold", "nominal_load"):
            checks.update(relative_translation=metrics["max_relative_translation_m"] <= limits["maximum_relative_translation_m"],
                relative_rotation=metrics["max_relative_rotation_rad"] <= limits["maximum_relative_rotation_rad"],
                relative_linear_speed=metrics["maximum_relative_linear_speed_m_s"] <= limits["maximum_relative_linear_speed_m_s"],
                relative_angular_speed=metrics["maximum_relative_angular_speed_rad_s"] <= limits["maximum_relative_angular_speed_rad_s"],
                bilateral_handle_load=metrics["bilateral_loaded_handle_fraction"] >= limits["minimum_bilateral_loaded_handle_fraction"],
                penetration=metrics["maximum_penetration_m"] <= limits["maximum_contact_penetration_m"],
                no_unintended_support=metrics["unintended_loaded_support_steps"] == 0)
        if name == "release":
            g = metrics["freeflight_acceleration_z_m_s2"]
            checks.update(actual_relative_motion=metrics["max_relative_translation_m"] > limits["release_minimum_relative_translation_m"],
                actual_downward_motion=metrics["downward_displacement_m"] > limits["release_minimum_relative_translation_m"],
                unsupported_gravity=g is not None and abs(g + 9.81) / 9.81 <= limits["freeflight_gravity_relative_error_max"])
        row.update(checks={k: bool(v) for k, v in checks.items()},
                   status="PASS_INDEPENDENT_CASE_SCREENS" if all(checks.values()) else "FAIL_INDEPENDENT_CASE_SCREENS")
        row["input_sha256"].update({raw.name: digest(raw), "control_trace.jsonl": digest(directory / "control_trace.jsonl")})
        cases[key] = row
    comparisons = {}
    for runtime in {x["runtime"] for x in index["cases"].values()}:
        strong_key = runtime + "/nominal_load"
        for name in ("weak_grip", "low_friction"):
            weak_key = runtime + "/" + name
            if strong_key not in cases or weak_key not in cases or any("metrics" not in cases[k] for k in (strong_key, weak_key)):
                comparisons[weak_key] = {"status": "NOT_RUN_UNMATCHED_OR_FAILED_TRANSFER"}
                continue
            a, b = [read(parent / index["cases"][k]["directory"] / "result.json") for k in (strong_key, weak_key)]
            p, r = pose_error(a["initial_state"]["tool_body_pose_xyzw"], b["initial_state"]["tool_body_pose_xyzw"])
            matched = (a["source_sha256"] == b["source_sha256"] and a["case_config"]["edge_wrench_world"] == b["case_config"]["edge_wrench_world"]
                and p <= limits["maximum_matched_start_tool_translation_m"] and r <= limits["maximum_matched_start_tool_rotation_rad"]
                and np.max(np.abs(np.asarray(a["initial_state"]["q"])[0] - np.asarray(b["initial_state"]["q"])[0])) <= limits["maximum_matched_start_arm_joint_error_rad"])
            ma, mb = cases[strong_key]["metrics"], cases[weak_key]["metrics"]
            delta_p = mb["max_relative_translation_m"] - ma["max_relative_translation_m"]
            delta_r = mb["max_relative_rotation_rad"] - ma["max_relative_rotation_rad"]
            larger = delta_p >= limits["minimum_weak_response_translation_increase_m"] or delta_r >= limits["minimum_weak_response_rotation_increase_rad"]
            comparisons[weak_key] = {"status": "PASS_MATCHED_WEAK_RESPONSE" if matched and larger else "FAIL_MATCHED_WEAK_RESPONSE",
                "matched": bool(matched), "translation_increase_m": delta_p, "rotation_increase_rad": delta_r,
                "strong_hold_status": cases[strong_key]["status"]}
        baseline_key = runtime + "/baseline_hold"
        for name in ("nominal_load", "edge_torque"):
            loaded_key = runtime + "/" + name
            key = loaded_key + "_arm_response"
            if any(k not in cases or "metrics" not in cases[k] for k in (baseline_key, loaded_key)):
                comparisons[key] = {"status": "NOT_RUN_MISSING_TRANSFER_OR_BASELINE"}
                continue
            a, b = cases[baseline_key]["metrics"], cases[loaded_key]["metrics"]
            deflection = float(np.max(np.abs(np.asarray(b["final_arm_deflection_rad"]) - a["final_arm_deflection_rad"])))
            force = float(np.linalg.norm(np.asarray(b["mean_contact_force_last_window_world_n"]) - a["mean_contact_force_last_window_world_n"]))
            comparisons[key] = {"status": "PASS_MEASURED_ARM_RESPONSE" if deflection >= limits["minimum_detectable_arm_response_rad"] else "FAIL_MEASURED_ARM_RESPONSE",
                "arm_deflection_change_rad": deflection, "contact_force_change_n": force,
                "contact_force_change_detected": force >= limits["minimum_detectable_contact_force_change_n"]}
    complete = all(runtime + "/" + name in cases for runtime in {x["runtime"] for x in index["cases"].values()} for name in protocol["cases"])
    passed = complete and all(r["status"].startswith("PASS") for r in cases.values()) and all(r["status"].startswith("PASS") for r in comparisons.values())
    return {"status": "PASS_INDEPENDENT_CONTROL_SCREENS" if passed else "FAIL_INDEPENDENT_CONTROL_SCREENS",
            "index": str(index_path.resolve()), "index_sha256": digest(index_path), "protocol_sha256": digest(protocol_path),
            "auditor_sha256": digest(__file__), "cases": cases, "comparisons": comparisons, "complete_seven_case_matrix": complete,
            "scope": "Independent raw contact/state recomputation of existing frozen engineering limits; original results preserved",
            "material_fingerprint_scope": "Recorded before/after particle-state fingerprints agree; no particle snapshots stored by rigid-only controls",
            "full_physics_acceptance": False, "material_reaction_on_off": "NOT_RUN_SEPARATE_CONTROL_REQUIRED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = audit_index(args.index)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.output_dir / "result.json")}))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
