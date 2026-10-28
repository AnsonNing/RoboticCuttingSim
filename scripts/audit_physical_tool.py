#!/usr/bin/env python3
"""Independently inspect recorded free-tool structure, state, contacts and loads.

This is a CPU/offline engineering audit, never a substitute for live hold,
release, grip/friction/load perturbations or matched reaction ON/OFF controls.
Raw episodes are read only. CUT/TRIM scores belong to their separate auditor.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.scene_frames import (compose_pose, invert_transform, matrix_pose,
    pose_errors, pose_matrix, transform_points)

# Representation tolerances, not calibrated physics acceptance thresholds.
LIMITS = {"pose_position_atol_m": 1e-6, "pose_rotation_atol_rad": 1e-5,
          "wrench_atol": 1e-6, "wrench_rtol": 1e-5,
          "handle_point_margin_m": 1e-6, "native_loaded_contact_minimum_n": 1e-6}
HOLD_LIMITS = {"settle_s": 2., "stability_window_s": .5, "translation_tolerance_m": .002,
               "rotation_tolerance_rad": np.pi / 60, "linear_speed_tolerance_m_s": .02,
               "angular_speed_tolerance_rad_s": .2, "min_bilateral_contact_fraction": .95,
               "max_penetration_m": .001}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def under(path, root):
    return str(path) == str(root) or str(path).startswith(str(root).rstrip("/") + "/")


def finite(value, shape=None):
    array = np.asarray(value, float)
    return (shape is None or array.shape == shape) and bool(np.isfinite(array).all())


def pose_check(expected, measured):
    try:
        p, r = pose_errors(expected, measured)
        return {"pass": bool(np.max(p) <= LIMITS["pose_position_atol_m"]
                              and np.max(r) <= LIMITS["pose_rotation_atol_rad"]),
                "maximum_translation_error_m": float(np.max(p)),
                "maximum_rotation_error_rad": float(np.max(r))}
    except (ValueError, IndexError):
        return {"pass": False, "reason": "Missing, invalid or mismatched XYZW poses"}


def source_audit(episode, manifest):
    hashes = manifest.get("implementation_sha256", {})
    mismatches = []
    for original, expected in hashes.items():
        path = episode / "implementation" / Path(original).name
        if not path.is_file() or digest(path) != expected:
            mismatches.append(original)
    integrity = []
    for path in sorted(episode.glob("source_integrity*.json")):
        row = read_json(path)
        integrity.append({"file": path.name, "status": row.get("status"), "sha256": digest(path)})
    return {"pass": bool(hashes) and not mismatches and all(r["status"] == "PASS" for r in integrity),
            "snapshot_count": len(hashes), "mismatches": mismatches,
            "runtime_integrity_receipts": integrity,
            "scope": "Saved implementation hashes and recorded runtime checks; current checkout may differ"}


def native_lifecycle_audit(episode, config):
    source = episode / "implementation/isaac_sim_runner.py"
    rig = episode / "implementation/camera_rig.py"
    if config.get("camera_layout") != "wrist_d405_global_d455" or not source.is_file() or not rig.is_file():
        return {"status": "NOT_RUN_OR_NOT_APPLICABLE"}
    tree = ast.parse(source.read_text())
    reset = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "reset"), None)
    if reset is None:
        return {"status": "NOT_RUN_UNRECOGNIZED_RESET_SOURCE"}
    calls = [(getattr(n.func, "attr", getattr(n.func, "id", "")), n.lineno)
             for n in ast.walk(reset) if isinstance(n, ast.Call)]
    physics = [line for name, line in calls if name == "initialize_physics"]
    cameras = [line for name, line in calls if name == "_create_cameras"]
    author = [line for name, line in calls if name == "author_camera_rig"]
    rig_mutates = any(name in rig.read_text() for name in ("SetActive(False)", "RemovePrim(", "RemoveAPI("))
    early = bool(physics and author and min(author) < min(physics)
                 and "pre_authored_camera_specs" in source.read_text())
    hazard = bool(rig_mutates and physics and cameras and min(cameras) > min(physics) and not early)
    return {"status": "FAIL_UNSAFE_POST_PHYSICS_RIG_MUTATION" if hazard else "PASS_KNOWN_LIFECYCLE_CONTRACT",
            "physics_initialization_lines": physics, "camera_creation_lines": cameras,
            "pre_physics_rig_authoring_lines": author, "preauthored_rig_reused": early,
            "scope": "Recorded source contract: authoring/deactivating rigid camera assets after live physics invalidates global tensor views even when metadata still says fresh"}


def runtime_log_audit(episode):
    files = [episode / "stdout.log", episode / "kit.log", episode.parent / (episode.name + ".log"),
             episode.parent / (episode.name + ".launch.log")]
    pattern = re.compile(r"(?:simulation\s*view|tensor\s*view).*invalidat", re.IGNORECASE)
    rows = []
    for path in files:
        if not path.is_file():
            continue
        count, excerpts = 0, []
        with path.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if pattern.search(line):
                    count += 1
                    if len(excerpts) < 3:
                        excerpts.append({"line": line_number, "text": line.strip()[:500]})
        rows.append({"path": str(path), "sha256": digest(path), "invalidated_view_lines": count, "first_matches": excerpts})
    return {"status": "NOT_RUN_NO_RUNTIME_LOG" if not rows else "FAIL_INVALIDATED_PHYSICS_VIEW" if any(r["invalidated_view_lines"] for r in rows) else "PASS_NO_KNOWN_VIEW_INVALIDATION_LOGS",
            "files": rows, "scope": "Native C++ tensor-view failures can coexist with Python fresh flags and finite cached states"}


def usd_audit(path, spec, hand_paths):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    body_path = "/World/Tools/CuttingTool_0"
    body = stage.GetPrimAtPath(body_path)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        return {"pass": False, "reason": "Independent tool rigid body missing", "body_path": body_path}
    rigid, mass = UsdPhysics.RigidBodyAPI(body), UsdPhysics.MassAPI(body)
    inertia = np.asarray(mass.GetDiagonalInertiaAttr().Get(), float)
    axes = mass.GetPrincipalAxesAttr().Get()
    axes = np.r_[axes.GetImaginary(), axes.GetReal()]
    R = Rotation.from_quat(axes).as_matrix()
    mass_value, com = float(mass.GetMassAttr().Get()), np.asarray(mass.GetCenterOfMassAttr().Get(), float)
    linked = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
            if any(under(str(target), body_path) for target in targets):
                linked.append(str(prim.GetPath()))
    checks = {"outside_hand_hierarchy": not any(under(body_path, hand) for hand in hand_paths),
              "stage_is_si_z_up": UsdGeom.GetStageMetersPerUnit(stage) == 1. and UsdGeom.GetStageUpAxis(stage) == "Z",
              "dynamic_rigid_body_enabled": bool(rigid.GetRigidBodyEnabledAttr().Get()) and not rigid.GetKinematicEnabledAttr().Get(),
              "no_tool_attachment_joint": not linked,
              "nonzero_mass_inertia": finite(inertia, (3,)) and bool(np.all(inertia > 0)) and 0 < mass_value < 1,
              "mass_matches_recorded_tool": bool(np.isclose(mass_value, spec["mass_kg"], rtol=1e-6)),
              "com_matches_recorded_tool": bool(np.allclose(com, spec["com_tool_m"], atol=1e-8, rtol=1e-6)),
              "inertia_matches_recorded_tool": bool(np.allclose(R @ np.diag(inertia) @ R.T,
                                    spec["inertia_com_tool_kg_m2"], atol=1e-10, rtol=1e-6))}
    checks["actual_hand_and_two_finger_bodies_exist"] = all(
        stage.GetPrimAtPath(hand_paths[0] + suffix).HasAPI(UsdPhysics.RigidBodyAPI)
        for suffix in ("", "/panda_leftfinger", "/panda_rightfinger"))
    ancestors = list(body.GetAncestors()) if hasattr(body, "GetAncestors") else []
    # USD versions without GetAncestors use explicit parent traversal.
    parent = body.GetParent()
    while parent and not parent.IsPseudoRoot():
        ancestors.append(parent); parent = parent.GetParent()
    checks["no_rigid_body_parent"] = not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in ancestors)
    cache, shape_rows = UsdGeom.XformCache(), []
    tool_world = np.asarray(cache.GetLocalToWorldTransform(body), float).T
    for shape in spec["shapes"]:
        prim = stage.GetPrimAtPath(body_path + "/" + shape["name"])
        matched = bool(prim and (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)) and prim.HasAPI(UsdPhysics.CollisionAPI)
                       and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        if matched:
            local = np.linalg.inv(tool_world) @ np.asarray(cache.GetLocalToWorldTransform(prim), float).T
            if prim.IsA(UsdGeom.Mesh):
                vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
            else:
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                vertices = np.array([[x, y, z] for x in [-1., 1.] for y in [-1., 1.] for z in [-1., 1.]]) * size / 2
            # A USD primitive collider may carry a scale; inspect the actual
            # affine geometry instead of treating its xform as a rigid pose.
            vertices = vertices @ local[:3, :3].T + local[:3, 3]
            expected = np.asarray(shape["vertices_tool_m"], float)
            vertices = vertices[np.lexsort(vertices.round(10).T[::-1])]
            expected = expected[np.lexsort(expected.round(10).T[::-1])]
            matched = vertices.shape == expected.shape and np.allclose(vertices, expected, atol=1e-8, rtol=1e-6)
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            matched = bool(matched and owner and str(owner.GetPath()) == body_path)
        shape_rows.append({"name": shape["name"], "pass": bool(matched)})
    checks["compound_colliders_match_owned_geometry"] = bool(shape_rows) and all(r["pass"] for r in shape_rows)
    disabled = body.GetAttribute("physxRigidBody:disableGravity").Get()
    checks["tool_gravity_not_disabled"] = disabled is not True
    compensation = body.GetAttribute("mjc:gravcomp").Get()
    if compensation is None:
        compensation = body.GetAttribute("mujoco:gravcomp").Get()
    checks["tool_has_no_authored_gravity_compensation"] = compensation in (None, 0, 0.)
    gravity = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            scene = UsdPhysics.Scene(prim)
            direction, magnitude = scene.GetGravityDirectionAttr().Get(), scene.GetGravityMagnitudeAttr().Get()
            authored = scene.GetGravityMagnitudeAttr().HasAuthoredValueOpinion() and np.isfinite(magnitude)
            gravity.append({"path": str(prim.GetPath()), "explicit_magnitude_m_s2": float(magnitude) if authored else None,
                            "direction": list(direction), "scope": "AUTHORED" if authored else "RUNTIME_DEFAULT_SENTINEL"})
    return {"pass": all(checks.values()), "checks": checks, "body_path": body_path,
            "joint_connections": linked, "mass_kg": mass_value, "com_tool_m": com.tolist(),
            "principal_moments_kg_m2": inertia.tolist(), "shapes": shape_rows,
            "gravity_configuration": gravity, "gravity_behavior": "NOT_RUN_FREE_DROP_CONTROL_REQUIRED"}


def native_contacts(row, tool_path, half, hand_paths, dt):
    """Recompute loaded handle pairs from raw points/impulses, never summary flags."""
    contact = row.get("contacts", {})
    now = float(row["native_time_s"])
    fresh_tolerance = max(1e-7, 4 * np.finfo(np.float32).eps * max(1., abs(now)))
    sensor_valid = contact.get("sensor_valid") is True and abs(now - float(contact.get("sensor_time_s", -1))) <= fresh_tolerance
    loaded, forces, unwanted = set(), {"left_finger": 0., "right_finger": 0.}, 0
    T_tool_world = invert_transform(pose_matrix(row["tool_body_pose_xyzw"]))
    for item in contact.get("contacts", []):
        a, b = item["body0"], item["body1"]
        other = b if under(a, tool_path) and not under(b, tool_path) else a if under(b, tool_path) and not under(a, tool_path) else None
        own_dt, stamp = float(item["dt_s"]), float(item["time_s"])
        fresh = sensor_valid and abs(stamp - now) <= fresh_tolerance and abs(own_dt - dt) <= max(1e-7, dt * 1e-4)
        if other is None or not fresh or own_dt <= 0:
            continue
        hand = hand_paths[0]
        role = "left_finger" if under(other, hand + "/panda_leftfinger") else "right_finger" if under(other, hand + "/panda_rightfinger") else None
        impulse = np.asarray(item["raw_impulse_world_ns"], float)
        point = transform_points(T_tool_world, item["position_world_m"])
        in_handle = bool(np.all(np.abs(point) <= half + LIMITS["handle_point_margin_m"]))
        load = float(np.linalg.norm(impulse) / own_dt)
        if role and in_handle and load > LIMITS["native_loaded_contact_minimum_n"]:
            loaded.add(role); forces[role] += load
        elif not role or not in_handle:
            unwanted += 1
    tensor = contact.get("tensor_forces", {})
    separations = [max(0., -float(item["separation_m"])) for item in tensor.get("records", []) if item.get("kind") == "normal"]
    tensor_valid = tensor.get("status") == "VALID_NATIVE_NORMAL_AND_FRICTION_DATA" and not tensor.get("capacity_may_be_reached", False)
    return {"bilateral_loaded": len(loaded) == 2, "normal_force_n": forces, "unintended_contacts": unwanted,
            "max_penetration_m": max(separations, default=0.), "penetration_measured": tensor_valid,
            "tangential_contact_data_available": tensor_valid,
            "scope": "Loaded raw normal impulses; native tensor normals/friction retained separately"}


def hold_audit(rows, config):
    if not rows or not config:
        return {"status": "NOT_RUN", "reason": "No settle trace/config"}
    times = np.array([r["time_s"] for r in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        return {"status": "FAIL", "reason": "Invalid/nonincreasing settling timestamps"}
    selected = [r for r in rows if r["time_s"] >= times[-1] - config["stability_window_s"] - 1e-9]
    relative = np.array([matrix_pose(invert_transform(pose_matrix(r["hand_pose_xyzw"])) @ pose_matrix(r["tool_body_pose_xyzw"])) for r in selected])
    times = np.array([r["time_s"] for r in selected])
    if len(times) < 3:
        return {"status": "NOT_RUN", "reason": "Too few samples in final stability window"}
    p, rotation = pose_errors(np.tile(relative[0], (len(relative), 1)), relative)
    linear_speed = np.linalg.norm(np.diff(relative[:, :3], axis=0), axis=-1) / np.diff(times)
    angular_speed = (Rotation.from_quat(relative[1:, 3:]) * Rotation.from_quat(relative[:-1, 3:]).inv()).magnitude() / np.diff(times)
    bilateral = float(np.mean([r["contact"]["bilateral_loaded"] for r in selected]))
    max_penetration = max(r["contact"]["max_penetration_m"] for r in selected)
    checks = {"final_window_covered": times[-1] - times[0] >= config["stability_window_s"] - 1e-6,
              "full_settle_duration_covered": rows[-1]["time_s"] - rows[0]["time_s"] + np.median(np.diff([r["time_s"] for r in rows])) >= config["settle_s"] - 1e-6,
              "bilateral_loaded_handle_fraction": bilateral >= config["min_bilateral_contact_fraction"],
              "relative_translation": p.max() <= config["translation_tolerance_m"],
              "relative_rotation": rotation.max() <= config["rotation_tolerance_rad"],
              "relative_linear_speed": linear_speed.max() <= config["linear_speed_tolerance_m_s"],
              "relative_angular_speed": angular_speed.max() <= config["angular_speed_tolerance_rad_s"],
              "no_unintended_support": all(r["contact"]["unintended_contacts"] == 0 for r in selected),
              "penetration_measured": all(r["contact"]["penetration_measured"] for r in selected),
              "penetration_bounded": max_penetration <= config["max_penetration_m"]}
    checks["declared_limits_not_weaker_than_frozen"] = all(
        config[k] <= v if k not in ("settle_s", "stability_window_s", "min_bilateral_contact_fraction") else config[k] >= v
        for k, v in HOLD_LIMITS.items())
    return {"status": "PASS_PREGRASP_ENGINEERING" if all(checks.values()) else "FAIL", "checks": {k: bool(v) for k, v in checks.items()},
            "samples": len(selected), "window_s": float(times[-1] - times[0]),
            "bilateral_loaded_handle_fraction": bilateral, "max_relative_translation_m": float(p.max()),
            "max_relative_rotation_rad": float(rotation.max()), "max_relative_linear_speed_m_s": float(linear_speed.max()),
            "max_relative_angular_speed_rad_s": float(angular_speed.max()), "max_penetration_m": max_penetration,
            "scope": "Only the actual final pregrasp window, not complete cutting or perturbation validation"}


def lab_hold_audit(rows, config, hand_path):
    if not rows or not config:
        return {"status": "NOT_RUN", "reason": "Missing Lab settling contacts/config"}
    normalized = []
    missing_hand = any("hand_pose_xyzw" not in row for row in rows)
    for row in rows:
        pose = row["tool_body_pose_xyzw"]
        # Old rows expose measured relative poses but no separate hand state.
        # That permits descriptive drift, not independent hand/tool verification.
        hand = row.get("hand_pose_xyzw")
        if hand is None:
            hand = matrix_pose(pose_matrix(pose) @ invert_transform(pose_matrix(row["relative_tool_in_hand_xyzw"])))
        loaded, unintended = set(), 0
        for pair in row.get("contact_pairs", []):
            a, b = pair["shape0"], pair["shape1"]
            handle_a, handle_b = a == "/World/Tools/CuttingTool_0/Handle", b == "/World/Tools/CuttingTool_0/Handle"
            other = b if handle_a else a if handle_b else None
            role = "left_finger" if other and under(other, hand_path + "/panda_leftfinger") else "right_finger" if other and under(other, hand_path + "/panda_rightfinger") else None
            if role and np.linalg.norm(pair["force_on_tool_world_n"]) > .05 and row.get(role.replace("_finger", "_handle_normal_n"), 0.) > .05:
                loaded.add(role)
            elif not role:
                unintended += 1
        normalized.append({**row, "time_s": row["settling_time_s"], "hand_pose_xyzw": hand,
            "contact": {"bilateral_loaded": len(loaded) == 2, "unintended_contacts": unintended,
                        "max_penetration_m": row["max_tool_rigid_penetration_m"], "penetration_measured": True}})
    result = hold_audit(normalized, config)
    result["raw_hand_poses_available"] = not missing_hand
    result["contact_sampling_scope"] = "Actual pairs at the last2ms step of each20ms interval; all-substep fraction is an aggregate, not raw pair history"
    window = [row for row in rows if row["settling_time_s"] >= rows[-1]["settling_time_s"] - config["stability_window_s"] - 1e-9]
    result["aggregate_bilateral_substep_fraction_minimum_final_window"] = min(row.get("bilateral_substep_fraction", 0.) for row in window)
    if result["status"] == "PASS_PREGRASP_ENGINEERING":
        result["status"] = "NOT_RUN_INCOMPLETE_HAND_EVIDENCE" if missing_hand else "PASS_PREGRASP_SAMPLED_CONTACTS"
    return result


def trace_state_audit(rows, state, spec, origin):
    if not rows:
        return {"status": "NOT_RUN"}
    times = np.array([row["native_time_s"] for row in rows])
    composed = compose_pose(np.array([r["tool_body_pose_xyzw"] for r in rows]), spec["tool_from_edge_xyz_xyzw"])
    composition = pose_check(composed, np.array([r["measured_edge_pose_xyzw"] for r in rows]))
    finite_substeps = all(finite(row.get(key, [np.nan])) for row in rows for key in (
        "hand_pose_xyzw", "tool_body_pose_xyzw", "finger_q_m", "finger_qdot_m_s",
        "tool_linear_velocity_world_m_s", "tool_angular_velocity_world_rad_s"))
    matches, max_error = 0, 0.
    for index, time in enumerate(state["time_s"]):
        selected = int(np.argmin(np.abs(times - (time + origin))))
        if abs(times[selected] - (time + origin)) > 1e-6:
            continue
        checked = pose_check(state["tool_body_pose_xyzw"][index], rows[selected]["tool_body_pose_xyzw"])
        max_error = max(max_error, checked.get("maximum_translation_error_m", float("inf")))
        if checked["pass"]:
            matches += 1
    return {"status": "PASS_TRACE_STATE_ALIGNMENT" if finite_substeps and composition["pass"] and matches == len(state["time_s"]) and np.all(np.diff(times) > 0) else "FAIL",
            "trace_samples": len(rows), "matched_robot_states": matches, "expected_robot_states": len(state["time_s"]),
            "finite_substep_hand_tool_finger_state": finite_substeps,
            "maximum_tool_body_translation_error_m": max_error, "edge_composition": composition}


def coupled_contact_audit(rows):
    selected = [row for row in rows if row.get("phase") == "coupled_episode"]
    invalid = []
    for index, row in enumerate(selected):
        contact, now = row.get("contacts", {}), float(row["native_time_s"])
        tolerance = max(1e-7, 4 * np.finfo(np.float32).eps * max(1., abs(now)))
        checks = {name: contact.get(name) is True for name in ("sensor_valid", "sensor_fresh", "new_physics_step")}
        checks["sensor_timestamp_current"] = abs(now - float(contact.get("sensor_time_s", -1))) <= tolerance
        tensor = contact.get("tensor_forces", {})
        checks["tensor_contact_view_valid"] = (tensor.get("status") == "VALID_NATIVE_NORMAL_AND_FRICTION_DATA"
                                                and not tensor.get("capacity_may_be_reached", False))
        if not all(checks.values()):
            invalid.append({"formal_substep": index, "native_time_s": now,
                            "failed": [name for name, passed in checks.items() if not passed]})
    return {"status": "NOT_RUN" if not selected else "FAIL_STALE_OR_INVALID_FORMAL_CONTACT_VIEW" if invalid else "PASS_FORMAL_CONTACT_FRESHNESS",
            "formal_substeps": len(selected), "invalid_substeps": len(invalid), "first_invalid": invalid[:5],
            "scope": "Every actual coupled step must retain fresh contact views after camera/render initialization; clock advance alone is insufficient"}


def wrench_application_state_audit(rows, spec):
    checks, samples = [], 0
    for previous, row in zip(rows[:-1], rows[1:]):
        if "force_application_point_world_m" not in row:
            continue
        transform = pose_matrix(previous["tool_body_pose_xyzw"])
        center = transform_points(transform, spec["com_tool_m"])
        edge = compose_pose(previous["tool_body_pose_xyzw"], spec["tool_from_edge_xyz_xyzw"])[:3]
        checks.append(bool(np.allclose(center, row["tool_com_before_step_world_m"], atol=1e-6, rtol=0)
                           and np.allclose(edge, row["force_application_point_world_m"], atol=1e-6, rtol=0)
                           and abs(row["native_time_s"] - previous["native_time_s"] - row["wrench_interval_dt_s"]) <= 1e-6))
        samples += 1
    return {"status": "NOT_RUN" if not samples else "PASS_MEASURED_PRESTEP_LEVER_ARM" if all(checks) else "FAIL",
            "substep_records": samples,
            "scope": "Application point/COM checked against preceding actual free-tool pose, before each rigid step"}


def material_input_audit(diagnostics, state):
    matches = []
    for row in diagnostics:
        if "measured_tool_pose" not in row or "timestamp_s" not in row:
            continue
        index = int(np.argmin(np.abs(state["time_s"] - row["timestamp_s"])))
        checked = pose_check(state["tool_pose_xyzw"][index], row["measured_tool_pose"])
        matches.append(abs(state["time_s"][index] - row["timestamp_s"]) <= 1e-6 and checked["pass"])
    return {"status": "NOT_RUN" if not matches else "PASS_ACTUAL_EDGE_MATERIAL_INPUT" if all(matches) else "FAIL",
            "control_intervals": len(matches), "scope": "Material adapter input versus measured edge state; separation requires independent task audit"}


def wrench_audit(rows):
    evaluated, maximum_error, nonzero, impulse = 0, 0., 0, np.zeros(6)
    checks = {"finite_wrench_records": True, "positive_interval_dt": True, "com_lever_arm": True,
              "no_direct_hand_wrench": True}
    for row in rows:
        if "applied_edge_material_wrench_world" not in row:
            continue
        w = np.asarray(row["applied_edge_material_wrench_world"], float) + np.asarray(row.get("applied_external_edge_wrench_world", np.zeros(6)), float)
        actual = np.asarray(row["applied_total_wrench_about_com_world"], float)
        point, center = np.asarray(row["force_application_point_world_m"], float), np.asarray(row["tool_com_before_step_world_m"], float)
        dt = float(row["wrench_interval_dt_s"])
        valid = bool(finite(w, (6,)) and finite(actual, (6,)) and finite(point, (3,)) and finite(center, (3,)) and np.isfinite(dt))
        checks["finite_wrench_records"] &= valid
        checks["positive_interval_dt"] &= dt > 0
        checks["no_direct_hand_wrench"] &= row.get("direct_hand_wrench_injection") is False
        if not valid:
            continue
        expected = np.r_[w[:3], w[3:] + np.cross(point - center, w[:3])]
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        checks["com_lever_arm"] &= bool(np.allclose(actual, expected, atol=LIMITS["wrench_atol"], rtol=LIMITS["wrench_rtol"]))
        impulse += actual * dt
        nonzero += int(np.linalg.norm(w) > LIMITS["wrench_atol"])
        evaluated += 1
    return {"status": "NOT_RUN" if not evaluated else "PASS_NUMERIC_ROUTING" if all(checks.values()) else "FAIL",
            "checks": checks, "substep_records": evaluated, "nonzero_load_records": nonzero,
            "maximum_force_torque_component_error": maximum_error, "integrated_applied_impulse_com_world": impulse.tolist(),
            "nonzero_load_exercised": nonzero > 0,
            "impulse_momentum_closure": "NOT_SCORED: whole-contact/gravity integration and independent tolerance required",
            "two_way_coupling": "NOT_RUN_MATCHED_REACTION_ON_OFF_REQUIRED"}


def lab_substeps_audit(path, spec, state, setup, diagnostics):
    """Read native MJWarp arrays; retain shape0 force and COM conventions."""
    metadata = read_json(path.with_suffix(".json"))
    if metadata is None:
        return {"status": "NOT_RUN_MISSING_RAW_METADATA"}
    with np.load(path, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    times, offsets = raw["rigid_time_s"], raw["contact_offsets"]
    n, m = len(times), len(raw["shape0"])
    if (offsets.shape != (n + 1,) or offsets.dtype.kind not in "iu" or offsets[0] != 0 or offsets[-1] != m
            or np.any(np.diff(offsets) < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid native raw substep times/contact slices")
    shape_body = np.asarray(metadata["shape_body"], int)
    a, b = raw["shape0"], raw["shape1"]
    if a.min(initial=0) < 0 or b.min(initial=0) < 0 or max(a.max(initial=0), b.max(initial=0)) >= len(shape_body):
        raise ValueError("Contact IDs do not map to recorded actual solver shapes")
    numeric = [array for array in raw.values() if array.dtype.kind in "fiu"]
    checks = {"finite_raw_states_and_contact_data": all(finite(array) for array in numeric),
              "rigid_dt_matches": bool(np.allclose(np.diff(times), metadata["dt_s"], atol=1e-6, rtol=0)),
              "model_mass_matches_spec": bool(np.isclose(metadata["tool_mass_kg"], spec["mass_kg"], rtol=1e-6)),
              "model_com_matches_spec": bool(np.allclose(metadata["tool_com_local_m"], spec["com_tool_m"], atol=1e-8, rtol=1e-6)),
              "model_inertia_matches_spec": bool(np.allclose(metadata["tool_inertia_com_local_kg_m2"], spec["inertia_com_tool_kg_m2"], atol=1e-10, rtol=1e-6)),
              "no_hand_external_wrench": bool(np.max(np.abs(raw["hand_external_wrench_world"]), initial=0.) <= 1e-8)}
    tool = metadata["tool_body_index"]
    fingers = {metadata["left_finger_body_index"]: "left_finger", metadata["right_finger_body_index"]: "right_finger"}
    handles = set(metadata["handle_shape_indices"])
    normalized, settling, wrench_checks, nonzero = [], [], [], 0
    for index in range(n):
        loaded, unintended, penetration = set(), 0, 0.
        for slot in range(offsets[index], offsets[index + 1]):
            first, second = int(a[slot]), int(b[slot])
            tool_first, tool_second = shape_body[first] == tool, shape_body[second] == tool
            if not (tool_first ^ tool_second):
                continue
            tool_shape, other = (first, shape_body[second]) if tool_first else (second, shape_body[first])
            force = raw["contact_wrench_on_shape0_world"][slot, :3] * (1 if tool_first else -1)
            normal_force = abs(float(force @ raw["normal_world"][slot]))
            if tool_shape in handles and other in fingers:
                if normal_force > .05:
                    loaded.add(fingers[other])
            else:
                unintended += 1
            penetration = max(penetration, max(0., -float(raw["distance_m"][slot])))
        pose = raw["tool_body_pose_xyzw"][index]
        edge = compose_pose(pose, spec["tool_from_edge_xyz_xyzw"])
        velocity = raw["tool_velocity_com_world"][index]
        row = {"native_time_s": float(times[index]), "time_s": float(times[index]),
            "hand_pose_xyzw": raw["hand_pose_xyzw"][index], "tool_body_pose_xyzw": pose,
            "measured_edge_pose_xyzw": edge, "finger_q_m": raw["finger_q"][index], "finger_qdot_m_s": raw["finger_qdot"][index],
            "tool_linear_velocity_world_m_s": velocity[:3], "tool_angular_velocity_world_rad_s": velocity[3:],
            "contact": {"bilateral_loaded": len(loaded) == 2, "unintended_contacts": unintended,
                        "max_penetration_m": penetration, "penetration_measured": True}}
        normalized.append(row)
        if raw["phase"][index] == "settling":
            settling.append(row)
        elif raw["phase"][index] == "formal_episode" and diagnostics:
            relative_time = times[index] - setup["settling_runtime_origin_s"]
            step = int(np.ceil(relative_time / float(np.diff(state["time_s"])[0]) - 1e-6)) - 1
            payload = diagnostics[step].get("native_float32_wrench_payload_world") if 0 <= step < len(diagnostics) else None
            if payload is not None:
                payload = np.asarray(payload)
                before = raw["applied_at_tool_pose_xyzw"][index]
                center = transform_points(pose_matrix(before), metadata["tool_com_local_m"])
                point = compose_pose(before, spec["tool_from_edge_xyz_xyzw"])[:3]
                expected = np.r_[payload[:3], payload[3:] + np.cross(point - center, payload[:3])]
                wrench_checks.append(bool(np.allclose(raw["applied_wrench_com_world"][index], expected,
                                      atol=LIMITS["wrench_atol"], rtol=LIMITS["wrench_rtol"])))
                nonzero += int(np.linalg.norm(payload) > LIMITS["wrench_atol"])
        if index:
            checks.setdefault("application_pose_matches_previous_body_state", True)
            checks["application_pose_matches_previous_body_state"] &= pose_check(raw["tool_body_pose_xyzw"][index - 1], raw["applied_at_tool_pose_xyzw"][index])["pass"]
    alignment = trace_state_audit(normalized, state, spec, setup["settling_runtime_origin_s"])
    pregrasp = hold_audit(settling, setup["grasp"])
    checks["measured_robot_raw_state_alignment"] = alignment["status"] == "PASS_TRACE_STATE_ALIGNMENT"
    return {"status": "PASS_RAW_SUBSTEP_ENGINEERING" if all(checks.values()) and all(wrench_checks) else "FAIL",
            "checks": checks, "samples": n, "contact_rows": m,
            "phase_counts": {str(name): int(np.count_nonzero(raw["phase"] == name)) for name in np.unique(raw["phase"])},
            "pregrasp": pregrasp, "trace_state_alignment": alignment,
            "wrench": {"status": "NOT_RUN" if not wrench_checks else "PASS_NUMERIC_ROUTING" if all(wrench_checks) else "FAIL",
                       "substep_records": len(wrench_checks), "nonzero_load_records": nonzero,
                       "nonzero_load_exercised": nonzero > 0, "two_way_coupling": "NOT_RUN_MATCHED_REACTION_ON_OFF_REQUIRED"},
            "contact_scope": "Actual per2ms shape pairs, signed world contact-point wrench on shape0; bilateral loading uses handle IDs and normal force>.05N",
            "input_sha256": {path.name: digest(path), path.with_suffix(".json").name: digest(path.with_suffix(".json"))}}


def audit_episode(episode):
    episode = Path(episode).resolve()
    manifest, setup = read_json(episode / "manifest.json", {}), read_json(episode / "setup.json", {})
    result = {"scope": "Independent offline physical-tool engineering audit; actual runtime evidence remains separate",
              "episode": str(episode), "auditor_limits": LIMITS, "full_physics_acceptance": False,
              "camera_validation": "NOT_RUN; Wrist D405 remains an ideal geometric prototype",
              "required_controls": {name: "NOT_RUN" for name in ("free_drop", "no_cut_hold", "release", "weak_grip", "low_friction", "external_wrench", "matched_reaction_on_off")},
              "TOOL_RIGID_BODY_VALIDATED": False, "PHYSICAL_GRASP_VALIDATED": False,
              "TWO_WAY_COUPLING_VALIDATED": False}
    if setup.get("tool_attachment_mode", manifest.get("config", {}).get("tool_attachment_mode")) != "physical_grasp":
        return {**result, "status": "NOT_RUN_LEGACY_OR_MISSING_PHYSICAL_TOOL"}
    result["runtime"] = "isaac_lab" if "lab" in str(manifest.get("backend", "")).lower() else "isaac_sim"
    spec = setup.get("physical_tool", setup.get("tool_spec"))
    if not spec:
        return {**result, "status": "NOT_RUN_MISSING_TOOL_SPEC"}
    result["sources"] = source_audit(episode, manifest)
    result["runtime_logs"] = runtime_log_audit(episode)
    result["native_view_lifecycle"] = (native_lifecycle_audit(episode, manifest.get("config", {}))
                                      if result["runtime"] == "isaac_sim" else {"status": "NOT_APPLICABLE"})
    stage = episode / "final_live_scene.usda"
    hand_paths = setup.get("hand_prims", [setup.get("hand_prim", setup.get("cutting_hand_body_label", "/World/Franka_0"))])
    result["usd"] = usd_audit(stage, spec, hand_paths) if stage.is_file() else {"pass": False, "reason": "No actual final stage"}
    robot_path, physical_path = episode / "robot_state.npz", episode / "physical_tool_state.npz"
    if not robot_path.is_file():
        return {**result, "status": "NOT_RUN_MISSING_MEASURED_ROBOT_STATE"}
    with np.load(robot_path, allow_pickle=False) as data:
        state = {k: data[k] for k in data.files}
    if physical_path.is_file():
        with np.load(physical_path, allow_pickle=False) as data:
            physical = {k: data[k] for k in data.files}
        if not np.array_equal(physical["time_s"], state["time_s"]):
            raise ValueError("Separate tool and robot clocks differ")
        state.update(physical)
    measured_edge = state.get("tool_edge_pose_xyzw", state.get("tool_pose_xyzw"))
    expected_edge = compose_pose(state["tool_body_pose_xyzw"], spec["tool_from_edge_xyz_xyzw"])
    result["measured_edge"] = pose_check(expected_edge, measured_edge)
    result["robot_tool_file_agreement"] = pose_check(measured_edge, state["tool_pose_xyzw"])
    body_index = setup.get("tool_body_index", setup.get("physical_knife_body_index"))
    result["solver_body_state_agreement"] = (pose_check(state["body_pose"][:, int(body_index)], state["tool_body_pose_xyzw"])
                                             if body_index is not None and "body_pose" in state else {"status": "NOT_RUN"})
    time = np.asarray(state["time_s"], float)
    numeric_keys = ("q", "qdot", "tool_body_pose_xyzw", "tool_linear_velocity_world_m_s", "tool_angular_velocity_world_rad_s")
    fingers = state.get("finger_q_m", state.get("finger_q"))
    finger_v = state.get("finger_qdot_m_s", state.get("finger_qdot"))
    finite_state = all(k in state and finite(state[k]) for k in numeric_keys) and fingers is not None and finger_v is not None and finite(fingers) and finite(finger_v)
    result["state"] = {"pass": bool(finite_state and len(time) > 1 and np.isfinite(time).all() and np.all(np.diff(time) > 0)),
                       "samples": len(time), "episode_s": float(time[-1] - time[0]),
                       "finite_finger_tool_articulation_state": bool(finite_state)}
    hand_tool = [matrix_pose(invert_transform(pose_matrix(h)) @ pose_matrix(t))
                 for h, t in zip(state["hand_pose_xyzw"], state["tool_body_pose_xyzw"])]
    drift_p, drift_r = pose_errors(np.tile(hand_tool[0], (len(time), 1)), hand_tool)
    nominal = compose_pose(compose_pose(state["hand_pose_xyzw"], spec["hand_from_tool_xyz_xyzw"]), spec["tool_from_edge_xyz_xyzw"])
    actual_vs_nominal = pose_errors(nominal, measured_edge)
    desired = state.get("desired_tool_pose_xyzw", state.get("commanded_tool_pose_xyzw"))
    result["slip_and_tracking"] = {"maximum_relative_translation_drift_m": float(np.max(drift_p)),
        "maximum_relative_rotation_drift_rad": float(np.max(drift_r)),
        "maximum_actual_vs_nominal_hand_edge_translation_m": float(np.max(actual_vs_nominal[0])),
        "maximum_actual_vs_nominal_hand_edge_rotation_rad": float(np.max(actual_vs_nominal[1])),
        "command_tracking": pose_check(desired, measured_edge) if desired is not None else {"status": "NOT_RUN"},
        "scope": "Measured tool remains distinct from nominal hand and command; tracking is descriptive, not a grasp gate"}
    trace_path = episode / "physical_grasp_trace_00.jsonl"
    rows, hold_rows = [], []
    if trace_path.is_file():
        with trace_path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        dt = setup["config"]["control_dt"] / setup["config"]["robot_substeps"]
        for row in rows:
            if row.get("phase") == "pregrasp_settling":
                hold_rows.append({**row, "time_s": row["native_time_s"], "contact": native_contacts(row,
                    result["usd"]["body_path"], np.asarray(spec["handle_half_extents_m"]), hand_paths, dt)})
    result["wrench"] = wrench_audit(rows)
    result["wrench_application_state"] = wrench_application_state_audit(rows, spec)
    result["formal_contact_freshness"] = coupled_contact_audit(rows)
    grasp = read_json(episode / "grasp_reset_00.json", {})
    result["pregrasp"] = hold_audit(hold_rows, grasp.get("config", setup.get("grasp")))
    result["trace_state_alignment"] = trace_state_audit(rows, state, spec, setup.get("native_time_origin_s", 0.))
    lab_trace_path = episode / "grasp_settling_trace.json"
    if not rows and lab_trace_path.is_file():
        result["pregrasp"] = lab_hold_audit(read_json(lab_trace_path), setup.get("grasp"), hand_paths[0])
    diagnostics_path = episode / "oracle_debug/diagnostics.json"
    if not diagnostics_path.is_file():
        diagnostics_path = episode / "diagnostics.json"
    diagnostics = read_json(diagnostics_path, [])
    result["material_input"] = material_input_audit(diagnostics, state)
    raw_lab_path = episode / "tool_substeps.npz"
    if raw_lab_path.is_file():
        raw_audit = lab_substeps_audit(raw_lab_path, spec, state, setup, diagnostics)
        result["raw_lab_substeps"] = raw_audit
        if "pregrasp" in raw_audit:
            result["pregrasp"] = raw_audit["pregrasp"]
            result["trace_state_alignment"] = raw_audit["trace_state_alignment"]
            result["wrench"] = raw_audit["wrench"]
    routes = [d for d in diagnostics if "wrench_recipient" in d]
    if not routes:
        routes = [{"wrench_recipient": result["usd"]["body_path"] if d["applied_wrench_receiver_body_index"] == body_index else "OTHER_BODY",
                   "direct_hand_material_wrench": d.get("direct_hand_material_wrench")}
                  for d in diagnostics if "applied_wrench_receiver_body_index" in d]
    result["recipient"] = {"status": "NOT_RUN" if not routes else "PASS_RECORDED_RECIPIENT" if all(
        d["wrench_recipient"] == result["usd"]["body_path"] and d.get("direct_hand_material_wrench") is False for d in routes) else "FAIL",
        "records": len(routes), "scope": "Recorded routing plus per-substep load arithmetic; not ON/OFF causality proof"}
    if not rows and not raw_lab_path.is_file() and "applied_wrench_edge_world" in state:
        payload = np.asarray(state["applied_wrench_edge_world"])
        impulse = np.asarray(state["applied_impulse_to_tool_com_world"])
        valid = payload.shape == impulse.shape == (len(time) - 1, 6) and finite(payload) and finite(impulse)
        force_closes = valid and np.allclose(payload[:, :3] * np.diff(time)[:, None], impulse[:, :3],
                                              atol=LIMITS["wrench_atol"], rtol=LIMITS["wrench_rtol"])
        result["wrench"] = {"status": "NOT_SCORED_INTERVAL_ONLY_WRENCH_EVIDENCE" if force_closes else "FAIL",
            "force_impulse_components_match": bool(force_closes),
            "nonzero_load_exercised": bool(valid and np.any(np.linalg.norm(payload, axis=-1) > LIMITS["wrench_atol"])),
            "com_torque_shift": "NOT_SCORED: need actual per-substep application point/COM or an independent integrated lever-arm receipt",
            "two_way_coupling": "NOT_RUN_MATCHED_REACTION_ON_OFF_REQUIRED"}
    stage_receipt = read_json(episode / "final_live_scene_receipt.json")
    if stage_receipt:
        result["stage_receipt_matches"] = digest(stage) == stage_receipt["sha256"]
    required = [result["sources"]["pass"], not result["native_view_lifecycle"]["status"].startswith("FAIL"),
                not result["runtime_logs"]["status"].startswith("FAIL"),
                result["usd"]["pass"], result["state"]["pass"], result["measured_edge"]["pass"],
                result["robot_tool_file_agreement"]["pass"], result["solver_body_state_agreement"].get("pass", True)]
    if rows:
        required.append(result["pregrasp"]["status"] == "PASS_PREGRASP_ENGINEERING")
        required.append(result["trace_state_alignment"]["status"] != "FAIL")
        required.append(result["formal_contact_freshness"]["status"] == "PASS_FORMAL_CONTACT_FRESHNESS")
    else:
        required.append(result["pregrasp"]["status"] != "FAIL")
        if "raw_lab_substeps" in result:
            required.append(result["raw_lab_substeps"]["status"] == "PASS_RAW_SUBSTEP_ENGINEERING")
    required += [result["wrench"]["status"] != "FAIL", result["wrench_application_state"]["status"] != "FAIL",
                 result["material_input"]["status"] != "FAIL", result["recipient"]["status"] != "FAIL", result.get("stage_receipt_matches", True)]
    result["status"] = "PASS_OFFLINE_ENGINEERING_AUDIT" if all(required) else "FAIL_OFFLINE_ENGINEERING_AUDIT"
    recorded_status = read_json(episode / "result.json", {}).get("status")
    result["recorded_runtime_status"] = recorded_status
    if recorded_status and recorded_status.startswith("PASS") and (result["formal_contact_freshness"]["invalid_substeps"]
            or result["native_view_lifecycle"]["status"].startswith("FAIL") or result["runtime_logs"]["status"].startswith("FAIL")):
        result["rejected_recorded_pass"] = "Unsafe post-physics rigid-asset mutation or stale contact views despite advancing clocks; pregrasp results remain separate"
    paths = [episode / "manifest.json", episode / "setup.json", episode / "result.json", stage, robot_path, physical_path, trace_path,
             episode / "grasp_reset_00.json", lab_trace_path, episode / "grasp_initialization.json",
             diagnostics_path, episode / "final_live_scene_receipt.json", raw_lab_path, raw_lab_path.with_suffix(".json")]
    paths += [Path(row["path"]) for row in result["runtime_logs"]["files"]]
    result["input_sha256"] = {str(p.relative_to(episode)) if p.is_relative_to(episode) else str(p): digest(p) for p in paths if p.is_file()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = audit_episode(args.episode)
        # An invalid/nonfinite evidence payload still needs a durable FAIL
        # receipt instead of crashing while serializing the audit itself.
        json.dumps(result, allow_nan=False)
    except Exception as error:
        result = {"status": "FAIL_INVALID_OR_UNSUPPORTED_EVIDENCE", "error": str(error), "full_physics_acceptance": False}
    result["auditor_sha256"] = digest(Path(__file__))
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.output_dir / "result.json")}))
    return 1 if result["status"].startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
