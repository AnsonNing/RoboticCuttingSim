import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("rack_audit", SCRIPTS / "audit_tool_rack.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
POSE = [0., 0., 0., 0., 0., 0., 1.]


def test_actual_clock_mapping_rejects_missing_or_shifted_rigid_steps():
    controls = [{"phase": "release", "time_s": .02, "state": {"tool_body_pose_xyzw": POSE}}]
    raw = [{"phase": "rigid_control_probe", "native_time_s": 2. + i * .002, "tool_body_pose_xyzw": POSE} for i in range(1, 11)]
    formal, phases = audit.align_phases(controls, raw, 2., .002, 10)
    assert len(formal) == len(phases["release"]) == 10
    raw[4]["native_time_s"] += .001
    with pytest.raises(ValueError, match="Clock mapping"):
        audit.align_phases(controls, raw, 2., .002, 10)


def test_rack_support_requires_actual_positive_tool_pair_impulse():
    row = {"contacts": {"contacts": [{"body0": "/World/Tools/CuttingTool_0", "body1": "/World/ToolRack/RailLeft",
        "raw_impulse_world_ns": [0., 0., .001], "dt_s": .002}]}}
    assert audit.rack_pairs(row) == [{"other": "/World/ToolRack/RailLeft", "normal_force_magnitude_n": .5, "rack": True, "finger": False}]
    row["contacts"]["contacts"][0]["raw_impulse_world_ns"] = [0., 0., 0.]
    assert audit.rack_pairs(row) == []
