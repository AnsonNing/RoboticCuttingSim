import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("controls_audit", SCRIPTS / "audit_physical_controls.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
POSE = [0., 0., 0., 0., 0., 0., 1.]
HALF = np.array([.008, .011, .020])


def native_row():
    records = [{"body0": audit.TOOL, "body1": "/World/Franka_0/panda_hand/panda_" + side + "finger",
                "raw_impulse_world_ns": [0., sign * .002, 0.], "position_world_m": [0., sign * .011, 0.],
                "time_s": .002, "dt_s": .002} for side, sign in (("left", 1), ("right", -1))]
    return {"native_time_s": .002, "tool_body_pose_xyzw": [.01, 0., 0., 0., 0., 0., 1.],
        "tool_body_pose_before_step_xyzw": POSE, "hand_pose_xyzw": POSE,
        "tool_linear_velocity_world_m_s": [0., 0., 0.], "tool_angular_velocity_world_rad_s": [0., 0., 0.],
        "finger_q_m": [0., 0.], "finger_qdot_m_s": [0., 0.],
        "contacts": {"contact_geometry_frame": "PRE_STEP_MEASURED_TOOL_POSE", "contact_geometry_tool_pose_xyzw": POSE,
          "contact_geometry_time_s": 0., "sensor_valid": True, "sensor_fresh": True, "new_physics_step": True,
          "sensor_time_s": .002, "contacts": records,
          "tensor_forces": {"status": "VALID_NATIVE_NORMAL_AND_FRICTION_DATA", "records": [], "total_filtered_force_world_on_tool_n": [0., 0., 0.]}}}


def test_actual_pre_contact_geometry_distinguishes_contact_and_post_state():
    row = native_row()
    new = audit.native_sample(row, HALF, .002, "PRE_STEP_MEASURED_TOOL_POSE")
    assert new["valid"] and np.all(new["loads"] == 1.)
    row["contacts"]["contact_geometry_frame"] = "LEGACY_POST_STEP_TOOL_POSE"
    old = audit.native_sample(row, HALF, .002, None)
    assert np.all(old["loads"] == 0.)
    assert old["outside"] == 2 and old["unexpected"] == 0


def test_wrong_pre_timestamp_and_nonfinger_support_cannot_certify_grasp():
    row = native_row()
    row["contacts"]["contact_geometry_time_s"] = .002
    sample = audit.native_sample(row, HALF, .002, "PRE_STEP_MEASURED_TOOL_POSE")
    assert not sample["valid"]
    row = native_row()
    row["contacts"]["contacts"][0]["body1"] = "/World/Table"
    sample = audit.native_sample(row, HALF, .002, "PRE_STEP_MEASURED_TOOL_POSE")
    assert sample["unexpected"] == 1 and sample["loads"][0] == 0.


def test_unsupported_gravity_derived_from_actual_com_path():
    protocol = {"measurement": {"stable_window_s": .5, "freeflight_minimum_duration_s": .03,
        "minimum_loaded_finger_force_n": .05}, "rigid_dt_s": .002}
    initial = {"hand_pose": POSE, "tool_body_pose_xyzw": POSE}
    samples = []
    for time in np.arange(1, 251) * .002:
        body = [0., 0., -9.81 * time**2 / 2, 0., 0., 0., 1.]
        samples.append({"time": time, "hand": POSE, "tool": body, "loads": [0., 0.], "force": [0., 0., 0.],
            "velocity": [0., 0., -9.81 * time, 0., 0., 0.], "fingers": [0., 0., 0., 0.],
            "free": True, "valid": True, "unexpected": 0, "outside": 0, "penetration": 0.})
    result = audit.measurements(samples, initial, {"com_tool_m": [0., 0., .01]}, protocol, .5)
    assert abs(result["freeflight_acceleration_z_m_s2"] + 9.81) < 1e-10
    assert result["complete_2ms_sampling"] and result["freeflight_duration_s"] > .49
    assert result["bilateral_loaded_handle_fraction"] == 0.
