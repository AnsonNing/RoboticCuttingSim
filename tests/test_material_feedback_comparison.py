"""Synthetic rejection controls for the read-only classifier, not physics proof."""
import copy
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("feedback_comparison", ROOT/"scripts/compare_material_feedback.py")
audit = importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
PROTOCOL = audit.read(ROOT/"configs/material_feedback_controls_v1.json")

def pair():
    n, sub = 400, 10
    pose = np.tile([.1, .2, .3, 0, 0, 0, 1.], (n+1, 1))
    robot = {"time_s": np.arange(n+1)*.02, "q": np.zeros((n+1, 2, 7)),
             "tool_body_pose_xyzw": pose.copy(), "desired_tool_pose_xyzw": pose.copy()}
    reaction = np.tile([.2, 0., 0., 0., 0., 0.], (n, 1))
    payload = np.vstack([np.zeros(6), reaction[:-1]])
    diagnostics = [{"next_material_wrench_world": f.tolist(), "applied_previous_material_wrench_world": p.tolist()}
                   for f, p in zip(reaction, payload)]
    raw = {"time": 2.+(np.arange(n*sub)+1)*.002,
           "contact": np.tile([0., 0., .5], (n*sub, 1)),
           "applied_com": np.repeat(payload, sub, axis=0),
           "no_hand_injection": True, "hand_injection_scope": "synthetic control", "contacts_valid": True}
    on = {"runtime": "isaac_lab", "robot": robot, "diagnostics": diagnostics, "raw": raw,
          "manifest": {"config": {"feedback": True, "tool_attachment_mode": "physical_grasp"},
                       "implementation_sha256": {"a.py": "frozen"}},
          "result": {"status": "PASS_EXECUTION", "steps": 400, "task_success": "NOT_SCORED"},
          "task": {"id": "CUT-01"}, "initial_material": {"positions": np.zeros((3, 3))},
          "snapshots_valid": True}
    off = copy.deepcopy(on)
    off["manifest"]["config"]["feedback"] = False
    off["raw"]["applied_com"][:] = 0
    for row in off["diagnostics"]:row["applied_previous_material_wrench_world"] = [0.]*6
    on["robot"]["q"][:, 0, 0] = np.linspace(0, 2e-5, n+1)
    on["robot"]["tool_body_pose_xyzw"][:, 0] += np.linspace(0, 2e-6, n+1)
    on["raw"]["contact"][:, 0] += .1
    return on, off

def test_synthetic_complete_ledger_does_not_promote_full_physics():
    result, _ = audit.evaluate(*pair(), PROTOCOL)
    assert result["status"] == "PASS_MATCHED_FEEDBACK_ENGINEERING_SCREENS"
    assert result["full_physics_acceptance"] is False
    assert result["task_status"] == {"on": "NOT_SCORED", "off": "NOT_SCORED"}

def test_metadata_feedback_switch_without_measured_response_fails():
    on, off = pair()
    on["robot"] = copy.deepcopy(off["robot"])
    on["raw"]["contact"] = off["raw"]["contact"].copy()
    result, _ = audit.evaluate(on, off, PROTOCOL)
    assert result["status"].startswith("FAIL")
    assert not any(result["checks"][k] for k in ("measured_arm_response_detected", "measured_tool_response_detected", "measured_finger_contact_response_detected"))

def test_actual_off_payload_leak_is_not_accepted_as_disabled_feedback():
    on, off = pair();off["raw"]["applied_com"][100, 2] = .001
    result, _ = audit.evaluate(on, off, PROTOCOL)
    assert not result["checks"]["off_actual_applied_wrench_zero"]
    assert result["status"].startswith("FAIL")

def test_mismatched_sources_and_initial_pose_prevent_a_response_claim():
    on, off = pair();off["manifest"]["implementation_sha256"]["a.py"] = "changed"
    off["robot"]["tool_body_pose_xyzw"][0, 0] += .001
    result, _ = audit.evaluate(on, off, PROTOCOL)
    assert not result["checks"]["same_saved_implementation"]
    assert not result["checks"]["same_initial_tool"]
    assert result["status"].startswith("FAIL")

def test_wrong_reaction_interval_is_rejected():
    on, off = pair()
    on["diagnostics"][0]["applied_previous_material_wrench_world"] = [.2, 0, 0, 0, 0, 0]
    result, _ = audit.evaluate(on, off, PROTOCOL)
    assert not result["checks"]["on_payload_is_previous_actual_reaction"]
    assert result["status"].startswith("FAIL")
