"""Independent recorded-rig mathematics; no runtime camera or physics updates."""
import numpy as np
from scipy.spatial.transform import Rotation

from .scene_frames import pose_matrix, validate_transform


def transform_error(expected, measured):
    a, b = validate_transform(expected), validate_transform(measured)
    return float(np.linalg.norm(a[:3, 3]-b[:3, 3])), float(Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).magnitude())


def trace_metrics(trace, robot_times, measured_hand_poses, specs, protocol):
    """Check every recorded sample against separately saved actual robot hands."""
    times = np.asarray(robot_times, float)
    hands = np.asarray(measured_hand_poses, float)
    if hands.shape != (len(times), 2, 7) or len(trace) < 1 or np.any(np.diff(times) <= 0):
        raise ValueError("Aligned actual robot times and both measured hand poses are required")
    limits = protocol["transforms"]
    camera_ids = protocol["camera_ids"]
    rows = []
    initial_global = None
    for sample in trace:
        timestamp = float(sample["timestamp_s"])
        index = int(np.argmin(abs(times-timestamp)))
        time_error = float(abs(times[index]-timestamp))
        recorded = sample["cameras"]
        if set(recorded) != set(camera_ids):
            raise ValueError("Camera trace must contain exactly the three configured camera IDs")
        row = {"timestamp_s": timestamp, "robot_time_alignment_error_s": time_error, "cameras": {}}
        for name in camera_ids:
            spec, camera = specs[name], recorded[name]
            world = validate_transform(camera["T_world_camera"])
            inverse = validate_transform(camera["T_camera_world"])
            arm = spec.get("parent_hand_index")
            if arm is None:
                initial_global = world.copy() if initial_global is None else initial_global
                expected = initial_global
                hand_error = (0., 0.)
                mount_error = (0., 0.)
            else:
                hand = pose_matrix(hands[index, arm])
                mount = validate_transform(spec["T_hand_optical_cv"])
                expected = hand@mount
                recorded_hand = camera.get("T_world_hand")
                if recorded_hand is None:
                    recorded_hand = sample["T_world_hand"][arm]
                hand_error = transform_error(hand, recorded_hand)
                mount_error = transform_error(mount, camera["T_hand_camera"])
            position, angle = transform_error(expected, world)
            usd = camera.get("T_world_camera_from_current_USD", world)
            usd_position, usd_angle = transform_error(expected, usd)
            row["cameras"][name] = {"translation_error_m": position, "rotation_error_rad": angle,
                "recorded_hand_translation_error_m": hand_error[0], "recorded_hand_rotation_error_rad": hand_error[1],
                "fixed_mount_translation_error_m": mount_error[0], "fixed_mount_rotation_error_rad": mount_error[1],
                "current_usd_translation_error_m": usd_position, "current_usd_rotation_error_rad": usd_angle,
                "inverse_max_abs_identity_error": float(np.max(abs(inverse@world-np.eye(4)))),
                "sensor_model": camera.get("sensor_model", "UNRECORDED")}
        rows.append(row)
    trace_times = np.asarray([row["timestamp_s"] for row in rows])
    unique = np.unique(trace_times)
    full_duration = float(times[-1]-times[0])
    checks = {"trace_time_is_nondecreasing": bool(np.all(np.diff(trace_times) >= -1e-9)),
        "complete_robot_time_span": abs(unique[0]-times[0]) <= limits["maximum_timestamp_alignment_error_s"]
            and abs(unique[-1]-times[-1]) <= limits["maximum_timestamp_alignment_error_s"],
        "every_control_interval_sampled": len(unique) >= len(times) and (len(unique) < 2 or np.diff(unique).max() <= limits["maximum_pose_sample_interval_s"]),
        "timestamps_match_actual_robot": max(row["robot_time_alignment_error_s"] for row in rows) <= limits["maximum_timestamp_alignment_error_s"]}
    if full_duration >= 8.-1e-6:
        checks["full_episode_has_at_least_401_unique_samples"] = len(unique) >= limits["minimum_unique_pose_samples_for_8s_episode"]
    maxima = {}
    for name in camera_ids:
        maxima[name] = {key: max(row["cameras"][name][key] for row in rows)
            for key in rows[0]["cameras"][name] if key != "sensor_model"}
        current = maxima[name]
        position_limit = (limits["maximum_global_translation_drift_m"] if specs[name].get("parent_hand_index") is None
                          else limits["maximum_hand_camera_translation_error_m"])
        angle_limit = (limits["maximum_global_rotation_drift_rad"] if specs[name].get("parent_hand_index") is None
                       else limits["maximum_hand_camera_rotation_error_rad"])
        checks[name+"_actual_fk"] = current["translation_error_m"] <= position_limit and current["rotation_error_rad"] <= angle_limit
        checks[name+"_current_usd"] = current["current_usd_translation_error_m"] <= position_limit and current["current_usd_rotation_error_rad"] <= angle_limit
        checks[name+"_inverse"] = current["inverse_max_abs_identity_error"] <= limits["maximum_inverse_identity_error"]
        checks[name+"_actual_hand_and_immutable_mount"] = (current["recorded_hand_translation_error_m"] <= position_limit
            and current["recorded_hand_rotation_error_rad"] <= angle_limit
            and current["fixed_mount_translation_error_m"] <= position_limit and current["fixed_mount_rotation_error_rad"] <= angle_limit)
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "trace_samples": len(rows), "unique_timestamps": len(unique), "robot_samples": len(times),
        "robot_duration_s": full_duration, "maximum_unique_sample_interval_s": float(np.diff(unique).max()) if len(unique) > 1 else None,
        "maximum_errors_by_camera": maxima, "records": rows}


def project_points(points_world, calibration):
    points = np.asarray(points_world, float)
    transform = validate_transform(calibration["T_world_camera"])
    camera = (points-transform[:3, 3])@transform[:3, :3]
    k = np.asarray(calibration["K"], float)
    safe_z = np.where(abs(camera[:, 2]) > 1e-12, camera[:, 2], 1e-12)
    uv = (camera@k.T)[:, :2]/safe_z[:, None]
    inside = ((camera[:, 2] >= calibration["near_m"]) & (camera[:, 2] <= calibration["far_m"])
        & (uv[:, 0] >= 0) & (uv[:, 0] < calibration["width"])
        & (uv[:, 1] >= 0) & (uv[:, 1] < calibration["height"]))
    return uv, camera[:, 2], inside


def effective_fov_deg(calibration):
    k = np.asarray(calibration["K"], float)
    if k.shape != (3, 3) or min(k[0, 0], k[1, 1], calibration["width"], calibration["height"]) <= 0:
        raise ValueError("A finite positive runtime pinhole calibration is required")
    return {"horizontal_deg": float(np.rad2deg(np.arctan(k[0, 2]/k[0, 0])+np.arctan((calibration["width"]-k[0, 2])/k[0, 0]))),
            "vertical_deg": float(np.rad2deg(np.arctan(k[1, 2]/k[1, 1])+np.arctan((calibration["height"]-k[1, 2])/k[1, 1])))}


def depth_comparison(actual, valid, geometry, geometry_valid):
    actual, valid, geometry, geometry_valid = map(np.asarray, (actual, valid, geometry, geometry_valid))
    if not (actual.shape == valid.shape == geometry.shape == geometry_valid.shape):
        raise ValueError("Depth and masks must have the same recorded resolution")
    common = valid & geometry_valid
    error = actual[common].astype(float)-geometry[common].astype(float)
    union = int(np.count_nonzero(valid | geometry_valid))
    return {"actual_valid_pixels": int(valid.sum()), "geometry_valid_pixels": int(geometry_valid.sum()),
        "common_pixels": int(common.sum()), "valid_mask_iou": float(common.sum()/union) if union else 0.,
        "p95_absolute_error_m": float(np.quantile(abs(error), .95)) if len(error) else None,
        "maximum_absolute_error_m": float(abs(error).max()) if len(error) else None,
        "rms_error_m": float(np.sqrt(np.mean(error**2))) if len(error) else None,
        "absolute_error_gt_1mm_pixels": int(np.count_nonzero(abs(error) > .001)),
        "absolute_error_gt_5mm_pixels": int(np.count_nonzero(abs(error) > .005)),
        "all_valid_error_tails_retained": True}


def obb_signed_projection_gap(first_transform, first_dimensions, second_transform, second_dimensions):
    """SAT gap: positive certifies a separation lower bound; overlap is visual only."""
    a, b = validate_transform(first_transform), validate_transform(second_transform)
    da, db = np.asarray(first_dimensions, float), np.asarray(second_dimensions, float)
    if da.shape != (3,) or db.shape != (3,) or not np.isfinite([da, db]).all() or min(*da, *db) <= 0:
        raise ValueError("OBBs require three finite positive full dimensions")
    axes = [*a[:3, :3].T, *b[:3, :3].T]
    axes += [np.cross(x, y) for x in a[:3, :3].T for y in b[:3, :3].T]
    values = []
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-10:
            continue
        axis = axis/length
        radius = abs(a[:3, :3].T@axis)@(da/2)+abs(b[:3, :3].T@axis)@(db/2)
        values.append(abs((a[:3, 3]-b[:3, 3])@axis)-radius)
    return float(max(values))
