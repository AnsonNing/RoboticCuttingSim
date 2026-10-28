#!/usr/bin/env python3
"""Read-only optional Native place/release/reacquire evidence, not cutting."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_physical_controls import jsonl, native_sample, relative
from audit_physical_tool import digest, pose_check, read_json, runtime_log_audit, source_audit, under, usd_audit


def align_phases(controls, raw, origin, dt, substeps):
    formal = [r for r in raw if r["phase"] != "pregrasp_settling"]
    if len(formal) != len(controls) * substeps:
        raise ValueError("Raw rigid steps do not match actual control intervals")
    end_times = np.array([r["time_s"] + origin for r in controls])
    times = np.array([r["native_time_s"] for r in formal])
    if (np.any(np.diff(end_times) <= 0) or np.any(np.diff(times) <= 0) or
            not np.allclose(np.diff(times), dt, atol=1e-9, rtol=0) or
            not np.allclose(times[substeps - 1::substeps], end_times, atol=1e-9, rtol=0)):
        raise ValueError("Clock mapping has missing, duplicate or shifted physics steps")
    phases = {}
    for i, control in enumerate(controls):
        group = formal[i * substeps:(i + 1) * substeps]
        if not pose_check(group[-1]["tool_body_pose_xyzw"], control["state"]["tool_body_pose_xyzw"])["pass"]:
            raise ValueError("Control state differs from actual ending rigid body state")
        phases.setdefault(control["phase"], []).extend(group)
    return formal, phases


def rack_pairs(row):
    """Positive RAW normal reports identify support; no signed rack wrench."""
    pairs = []
    for pair in row["contacts"].get("contacts", []):
        a, b = pair["body0"], pair["body1"]
        if under(a, "/World/Tools/CuttingTool_0") == under(b, "/World/Tools/CuttingTool_0"):
            continue
        other = b if under(a, "/World/Tools/CuttingTool_0") else a
        force = float(np.linalg.norm(pair["raw_impulse_world_ns"]) / pair["dt_s"])
        if force > 1e-6:
            pairs.append({"other": other, "normal_force_magnitude_n": force,
                          "rack": under(other, "/World/ToolRack"),
                          "finger": other.endswith(("/panda_leftfinger", "/panda_rightfinger"))})
    return pairs


def audit(episode):
    setup, manifest, protocol, recorded = [read_json(episode / p) for p in (
        "setup.json", "manifest.json", "protocol.json", "result.json")]
    controls, raw = jsonl(episode / "control_trace.jsonl"), jsonl(episode / "physical_grasp_trace_00.jsonl")
    substeps = int(setup["config"]["robot_substeps"])
    dt = float(setup["config"]["control_dt"]) / substeps
    formal, phases = align_phases(controls, raw, setup["native_time_origin_s"], dt, substeps)
    spec = setup["physical_tool"]
    half = np.asarray(spec["handle_half_extents_m"])
    states = [native_sample(r, half, dt, "PRE_STEP_MEASURED_TOOL_POSE") for r in formal]
    release, withdraw, close, hold = [phases[k] for k in (
        "release_onto_rack", "withdraw_open_hand", "close_on_handle", "hold_reacquired_tool")]
    end_release = np.asarray(release[-1]["tool_body_pose_xyzw"])
    end_withdraw = np.asarray(withdraw[-1]["tool_body_pose_xyzw"])
    before_lift = np.asarray(close[-1]["tool_body_pose_xyzw"])
    held = np.asarray(hold[-1]["tool_body_pose_xyzw"])
    finger_open = np.asarray(release[-1]["finger_q_m"])[0]
    hand_withdrawal = np.asarray(withdraw[-1]["hand_pose_xyzw"])[:3] - release[-1]["hand_pose_xyzw"][:3]
    withdraw_tool_motion = max(np.linalg.norm(np.asarray(r["tool_body_pose_xyzw"])[:3] - end_release[:3]) for r in withdraw)
    relative_hold = relative([r["hand_pose_xyzw"] for r in hold], [r["tool_body_pose_xyzw"] for r in hold])
    drift = float(np.linalg.norm(relative_hold[:, :3] - relative_hold[0, :3], axis=1).max())
    angle = float((Rotation.from_quat(relative_hold[0, 3:]).inv() * Rotation.from_quat(relative_hold[:, 3:])).magnitude().max())
    hold_samples = [native_sample(r, half, dt, "PRE_STEP_MEASURED_TOOL_POSE") for r in hold]
    bilateral = float(np.mean([np.all(r["loads"] > .05) for r in hold_samples]))
    penetration = max(r["penetration"] for r in hold_samples)
    sources, logs = source_audit(episode, manifest), runtime_log_audit(episode)
    usd = usd_audit(episode / "final_live_scene.usda", spec, setup["hand_prims"])
    checks = {
        "archived_sources_match": sources["pass"], "independent_dynamic_tool_structure": usd["pass"],
        "no_known_tensor_view_invalidations": not logs["status"].startswith("FAIL"),
        "all_actual_contact_steps_fresh": all(r["valid"] for r in states),
        "measured_release_center_within_original_budget": np.linalg.norm(end_release[:3] - protocol["rack_handle_center_world_m"]) <= protocol["maximum_placement_center_error_m"],
        "actual_fingers_open_clear_of_handle": np.min(finger_open) - .00002 > half[1],
        "loaded_rack_support_at_release_and_withdrawal": all(any(p["rack"] for p in rack_pairs(r)) for r in (release[-1], withdraw[-1])),
        "actual_hand_withdrawal": hand_withdrawal[2] >= protocol["open_retract_m"] - protocol["transfer"]["maximum_ik_position_error_m"],
        "tool_remains_within_original_placement_budget_during_withdrawal": withdraw_tool_motion <= protocol["maximum_placement_center_error_m"],
        "actual_tool_liftoff": held[2] - before_lift[2] >= protocol["minimum_lift_m"],
        "loaded_bilateral_reacquired_handle": bilateral >= protocol["minimum_bilateral_loaded_fraction"],
        "stable_reacquired_relative_position": drift <= protocol["maximum_relative_translation_m"],
        "stable_reacquired_relative_rotation": angle <= protocol["maximum_relative_rotation_rad"],
        "bounded_reacquired_penetration": penetration <= protocol["maximum_grasp_penetration_m"],
        "no_loaded_rack_table_palm_support_during_final_hold": all(all(p["finger"] for p in rack_pairs(r)) for r in hold),
    }
    per_phase = {}
    for phase, rows in phases.items():
        pairs = [rack_pairs(r) for r in rows]
        per_phase[phase] = {"actual_rigid_steps": len(rows), "native_start_s": rows[0]["native_time_s"] - dt,
            "native_end_s": rows[-1]["native_time_s"], "loaded_rack_contact_steps": sum(any(p["rack"] for p in x) for x in pairs),
            "loaded_other_nonfinger_steps": sum(any(not p["finger"] and not p["rack"] for p in x) for x in pairs),
            "contact_body_paths": sorted({p["other"] for x in pairs for p in x})}
    return {"status": "PASS_AVAILABLE_RACK_ENGINEERING_SCREENS" if all(checks.values()) else "FAIL_RACK_ENGINEERING_SCREENS",
        "episode": str(episode), "checks": {k: bool(v) for k, v in checks.items()},
        "sources": sources, "runtime_logs": logs, "usd": usd, "phases": per_phase,
        "release_finger_q_m": finger_open.tolist(), "release_center_error_m": float(np.linalg.norm(end_release[:3] - protocol["rack_handle_center_world_m"])),
        "hand_withdrawal_xyz_m": hand_withdrawal.tolist(), "tool_motion_during_withdrawal_m": float(withdraw_tool_motion),
        "tool_lift_m": float(held[2] - before_lift[2]), "bilateral_loaded_handle_fraction": bilateral,
        "hold_translation_drift_m": drift, "hold_rotation_drift_rad": angle, "hold_max_penetration_m": penetration,
        "actual_formal_rigid_steps": len(formal), "settling_steps": len(raw) - len(formal),
        "control_steps": len(controls), "duration_s": controls[-1]["time_s"],
        "rack_force_scope": "RAW positive normal impulses identify support; rack is not in tensor filters, so full signed rack normal+friction wrench is NOT_SCORED",
        "material_scope": "NOT_RUN_RIGID_ONLY; recorded material_unchanged check is source-backed but original before/after hash values were not serialized for independent recomputation",
        "recorded_result_status": recorded["status"], "newton_pickup": "NOT_RUN", "full_physics_acceptance": False,
        "auditor_sha256": digest(__file__), "input_sha256": {p: digest(episode / p) for p in (
            "setup.json", "manifest.json", "protocol.json", "result.json", "control_trace.jsonl", "physical_grasp_trace_00.jsonl", "final_live_scene.usda")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = audit(args.episode.resolve())
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.output_dir / "result.json")}))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
