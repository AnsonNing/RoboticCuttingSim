#!/usr/bin/env python3
"""Read-only Native/Lab material-feedback ON/OFF comparison, with frozen screens."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def archive(path):
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k] for k in value.files}

def pose_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a[..., :3]-b[..., :3], axis=-1), (
        Rotation.from_quat(a[..., 3:]).inv()*Rotation.from_quat(b[..., 3:])).magnitude()

def raw_lab(path):
    data, meta = archive(path/"tool_substeps.npz"), read(path/"tool_substeps.json")
    chosen = np.flatnonzero(data["phase"] == "formal_episode")
    body = np.asarray(meta["shape_body"]); tool = meta["tool_body_index"]
    fingers = [meta["left_finger_body_index"], meta["right_finger_body_index"]]
    forces = []
    for i in chosen:
        lo, hi = data["contact_offsets"][i:i+2]
        a, b = data["shape0"][lo:hi], data["shape1"][lo:hi]
        valid = (a >= 0)&(b >= 0)
        a, b = a[valid], b[valid]
        f = data["contact_wrench_on_shape0_world"][lo:hi, :3][valid]
        sign = ((body[a] == tool)&np.isin(body[b], fingers)).astype(int)-((body[b] == tool)&np.isin(body[a], fingers)).astype(int)
        forces.append((sign[:, None]*f).sum(axis=0))
    return {"time": data["rigid_time_s"][chosen], "contact": np.asarray(forces),
        "applied_com": data["applied_wrench_com_world"][chosen],
        "no_hand_injection": bool(np.max(np.abs(data["hand_external_wrench_world"][chosen])) == 0),
        "hand_injection_scope": "Numeric post-step hand body_f samples; pre-step input recipient/source audit remains separate", "contacts_valid": True}

def raw_native(path):
    rows = [read_row for line in (path/"physical_grasp_trace_00.jsonl").open()
            if (read_row := json.loads(line))["phase"] == "coupled_episode"]
    forces, valid = [], []
    for row in rows:
        contact = row["contacts"]; tensor = contact.get("tensor_forces", {})
        role = tensor.get("force_by_role_world_on_tool_n", {})
        forces.append(np.asarray(role.get("left_finger", [0., 0., 0.]))+np.asarray(role.get("right_finger", [0., 0., 0.])))
        valid.append(contact.get("sensor_valid") and contact.get("sensor_fresh") and contact.get("new_physics_step")
            and tensor.get("status") == "VALID_NATIVE_NORMAL_AND_FRICTION_DATA" and not tensor.get("capacity_may_be_reached", False))
    return {"time": np.asarray([r["native_time_s"] for r in rows]), "contact": np.asarray(forces),
        "applied_com": np.asarray([r["applied_total_wrench_about_com_world"] for r in rows]),
        "no_hand_injection": all(r.get("direct_hand_wrench_injection") is False for r in rows),
        "hand_injection_scope": "Recorded native recipient flag; numeric hand external-force buffer unavailable; source/receiver audit remains separate",
        "contacts_valid": bool(valid) and all(valid)}

def load_episode(path):
    path = Path(path).resolve()
    manifest, result = read(path/"manifest.json"), read(path/"result.json")
    if result.get("source") == "live_isaac_lab":
        native = False
    elif manifest.get("backend") == "LIVE_IsaacSim6.2_cutting":
        native = True
    else:
        raise ValueError("Unrecognized actual recorder runtime provenance")
    diagnostics = read(path/("oracle_debug/diagnostics.json" if native else "diagnostics.json"))
    robot = archive(path/"robot_state.npz")
    if "tool_body_pose_xyzw" not in robot:
        robot["tool_body_pose_xyzw"] = archive(path/"physical_tool_state.npz")["tool_body_pose_xyzw"]
    physics = path/("oracle_debug/physics_state.npz" if native else "physics_state.npz")
    with np.load(physics, allow_pickle=False) as arrays:
        initial_material = {k: arrays[k][0].copy() for k in ("positions", "velocities", "codes")}
        initial_material["mass_kg"] = arrays["mass_kg"].copy()
    saved = manifest["implementation_sha256"]
    snapshots_valid = bool(saved) and all((path/"implementation"/Path(k).name).is_file()
        and sha(path/"implementation"/Path(k).name) == value for k, value in saved.items())
    input_paths = ["manifest.json", "result.json", "robot_state.npz",
        "task.json" if (path/"task.json").is_file() else "setup.json",
        str(physics.relative_to(path)), "oracle_debug/diagnostics.json" if native else "diagnostics.json"]
    input_paths += (["physical_grasp_trace_00.jsonl"] if native else
                    ["physical_tool_state.npz", "tool_substeps.npz", "tool_substeps.json"])
    return {"path": str(path), "runtime": "isaac_sim" if native else "isaac_lab", "manifest": manifest,
        "result": result, "robot": robot, "diagnostics": diagnostics,
        "task": read(path/"task.json") if (path/"task.json").is_file() else read(path/"setup.json")["task"], "initial_material": initial_material,
        "raw": (raw_native if native else raw_lab)(path), "snapshots_valid": snapshots_valid,
        "input_sha256": {p: sha(path/p) for p in input_paths}}

def evaluate(on, off, protocol):
    dt, rigid_dt = protocol["control_dt_s"], protocol["rigid_dt_s"]
    match, ledger, limits = protocol["matching"], protocol["ledger"], protocol["response"]
    a, b = on["robot"], off["robot"]
    q_delta = np.abs(a["q"][:, 0]-b["q"][:, 0])
    tool_p, tool_r = pose_error(a["tool_body_pose_xyzw"], b["tool_body_pose_xyzw"])
    config_a, config_b = dict(on["manifest"]["config"]), dict(off["manifest"]["config"])
    feedback_on, feedback_off = config_a.pop("feedback"), config_b.pop("feedback")
    # Each recorder authors the same static scene in its own output directory.
    for config in (config_a, config_b):
        config.pop("static_scene", None); config.pop("scene", None)
    n = len(on["diagnostics"]); substeps = round(dt/rigid_dt)
    contact_a, contact_b, applied_a, applied_b = [], [], [], []
    raw_complete = True
    for ep, contact, applied in ((on, contact_a, applied_a), (off, contact_b, applied_b)):
        raw = ep["raw"]; time = raw["time"]
        complete = len(time) == n*substeps and bool(len(time)>1) and np.allclose(np.diff(time), rigid_dt, atol=1e-7, rtol=0)
        raw_complete &= complete
        if not complete:raise ValueError("Raw formal substep count/timestamps do not match actual control intervals")
        contact.extend(raw["contact"].reshape(n, substeps, 3).mean(axis=1))
        applied.extend(raw["applied_com"].reshape(n, substeps, 6).mean(axis=1))
    contact_a, contact_b, applied_a, applied_b = map(np.asarray, (contact_a, contact_b, applied_a, applied_b))
    payload_a = np.asarray([r["applied_previous_material_wrench_world"] for r in on["diagnostics"]])
    payload_b = np.asarray([r["applied_previous_material_wrench_world"] for r in off["diagnostics"]])
    reaction = np.asarray([r["next_material_wrench_world"] for r in on["diagnostics"]])
    expected = np.vstack([np.zeros(6), reaction[:-1]])
    external = [r.get("external_edge_wrench_world", r.get("tool_external_wrench_edge_world", np.zeros(6)))
                for ep in (on, off) for r in ep["diagnostics"]]
    initial_p, initial_r = pose_error(a["tool_body_pose_xyzw"][0], b["tool_body_pose_xyzw"][0])
    contact_difference = np.linalg.norm(contact_a-contact_b, axis=1)
    initial_equal = all(np.array_equal(on["initial_material"][k], off["initial_material"][k]) for k in on["initial_material"])
    checks = {
        "same_runtime": on["runtime"] == off["runtime"],
        "both_full_execution": all(ep["result"].get("status") == "PASS_EXECUTION" and ep["result"].get("steps") == 400 for ep in (on, off)),
        "exactly_one_feedback_switch": feedback_on is True and feedback_off is False and config_a == config_b,
        "both_physical_grasp": config_a.get("tool_attachment_mode") == "physical_grasp" and config_b.get("tool_attachment_mode") == "physical_grasp",
        "same_saved_implementation": on["manifest"]["implementation_sha256"] == off["manifest"]["implementation_sha256"],
        "saved_sources_integrity": on["snapshots_valid"] and off["snapshots_valid"],
        "same_task": on["task"] == off["task"], "same_initial_material": initial_equal,
        "same_time_grid": np.array_equal(a["time_s"], b["time_s"]) and len(a["time_s"]) == 401 and np.isclose(a["time_s"][-1], 8.),
        "same_initial_joints": bool(np.max(q_delta[0]) <= match["initial_joint_atol_rad"]),
        "same_initial_tool": bool(initial_p <= match["initial_tool_position_atol_m"] and initial_r <= match["initial_tool_rotation_atol_rad"]),
        "same_desired_pose_sequence": bool(np.allclose(a["desired_tool_pose_xyzw"], b["desired_tool_pose_xyzw"], atol=match["desired_pose_atol"], rtol=0)),
        "actual_contact_samples_complete_and_valid": raw_complete and on["raw"]["contacts_valid"] and off["raw"]["contacts_valid"],
        "off_actual_applied_wrench_zero": bool(np.max(np.abs(applied_b)) <= ledger["off_payload_atol"] and np.max(np.abs(payload_b)) <= ledger["off_payload_atol"]),
        "on_material_force_present": bool(np.max(np.linalg.norm(payload_a[:, :3], axis=1)) >= ledger["minimum_on_material_force_n"]),
        "on_payload_is_previous_actual_reaction": bool(np.allclose(payload_a, expected, atol=ledger["wrench_atol"], rtol=ledger["wrench_rtol"])),
        "on_actual_com_force_matches_payload": bool(np.allclose(applied_a[:, :3], payload_a[:, :3], atol=ledger["wrench_atol"], rtol=ledger["wrench_rtol"])),
        "no_external_tool_load": bool(np.max(np.abs(external)) <= ledger["off_payload_atol"]),
        "no_direct_hand_injection": on["raw"]["no_hand_injection"] and off["raw"]["no_hand_injection"],
        "measured_arm_response_detected": bool(q_delta.max() >= limits["minimum_arm_joint_difference_rad"]),
        "measured_tool_response_detected": bool(tool_p.max() >= limits["minimum_tool_position_difference_m"]),
        "measured_finger_contact_response_detected": bool(contact_difference.max() >= limits["minimum_finger_contact_force_difference_n"]),
    }
    finite = all(np.isfinite(value).all() for value in (q_delta, tool_p, tool_r, contact_a, contact_b, applied_a, applied_b, payload_a, payload_b, reaction))
    checks["finite_states_and_forces"] = finite
    result = {"status": "PASS_MATCHED_FEEDBACK_ENGINEERING_SCREENS" if all(checks.values()) else "FAIL_MATCHED_FEEDBACK_ENGINEERING_SCREENS",
        "checks": {k: bool(v) for k, v in checks.items()}, "full_physics_acceptance": False,
        "scope": "Actual material response comparison only; not task, stable-grasp or full two-way coupling acceptance",
        "feedback_delay_s": dt, "maximum_joint_difference_rad": float(q_delta.max()),
        "maximum_tool_position_difference_m": float(tool_p.max()), "maximum_tool_rotation_difference_rad": float(tool_r.max()),
        "maximum_finger_contact_force_difference_n": float(contact_difference.max()),
        "maximum_on_material_force_n": float(np.linalg.norm(payload_a[:, :3], axis=1).max()),
        "maximum_off_actual_com_payload": float(np.abs(applied_b).max()),
        "initial_tool_position_difference_m": float(initial_p), "initial_tool_rotation_difference_rad": float(initial_r),
        "initial_joint_difference_rad": float(q_delta[0].max()),
        "hand_injection_evidence_scope": {"on": on["raw"]["hand_injection_scope"], "off": off["raw"]["hand_injection_scope"]},
        "interface_work_residual": "NOT_SCORED_BY_THIS_HELPER; preserve separate frozen independent audit and all existing failures",
        "task_status": {name: ep["result"].get("task_success", ep["result"].get("task_goal_status"))
            for name, ep in (("on", on), ("off", off))}}
    series = {"time_s": a["time_s"], "active_joint_difference_rad": a["q"][:, 0]-b["q"][:, 0],
        "tool_position_difference_m": tool_p, "tool_rotation_difference_rad": tool_r,
        "contact_time_s": a["time_s"][1:], "finger_contact_force_on_n": contact_a,
        "finger_contact_force_off_n": contact_b, "applied_com_wrench_on_world": applied_a,
        "applied_com_wrench_off_world": applied_b, "actual_material_payload_on_world": payload_a}
    return result, series

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on", type=Path, required=True);parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/material_feedback_controls_v1.json")
    parser.add_argument("--output-dir", type=Path, required=True);args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = read(args.protocol);(args.output_dir/"protocol.json").write_bytes(args.protocol.read_bytes())
    try:
        on, off = load_episode(args.on), load_episode(args.off)
        result, series = evaluate(on, off, protocol)
        np.savez_compressed(args.output_dir/"comparison.npz", **series)
        result.update(episodes={"on": on["path"], "off": off["path"]},
            input_sha256={"on": on["input_sha256"], "off": off["input_sha256"]})
    except Exception as error:
        result = {"status": "FAIL_UNREADABLE_OR_UNMATCHED_EVIDENCE", "error": str(error), "full_physics_acceptance": False}
    result.update(protocol_sha256=sha(args.protocol), auditor_sha256=sha(Path(__file__)))
    (args.output_dir/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(result["status"], args.output_dir/"result.json")
    return 0 if result["status"] == "PASS_MATCHED_FEEDBACK_ENGINEERING_SCREENS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
