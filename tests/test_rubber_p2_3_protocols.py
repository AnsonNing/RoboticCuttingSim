"""P2-3 protocols are frozen at commit (ADR 0023 §21): declared before their formal / implementation runs.

A failure of a pin means a frozen file was edited. Never update a pin: declare the next version instead.
"""
import hashlib
import json

from robotic_cutting_sim.rubber_material import ROOT

PINS = {
    "configs/rubber/rubber_moving_grip_tests_v1.json": "ff14f39dc750fcf095a1a84ba8c7a48d3f0721a27b08062e6e03680eab7678bc",
}


def _load(name):
    return json.loads((ROOT/name).read_text())


def test_frozen_files_unchanged():
    for name, pin in PINS.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == pin, f"{name} was modified"


def test_moving_grip_protocol_declares_tests_thresholds_and_scenes():
    cfg = _load("configs/rubber/rubber_moving_grip_tests_v1.json")
    assert cfg["schema"] == "rcs_rubber_moving_grip_tests_v1" and cfg["version"] == 1
    assert cfg["calibration_status"].startswith("SIMULATOR_ONLY_UNCALIBRATED")
    assert sorted(cfg["tests"]) == [f"RUBBER-GRIP-{i:02d}" for i in range(1, 5)]
    assert cfg["tests"]["RUBBER-GRIP-01"]["thresholds"] == {"m_max": 0.01, "h_max": 0.002}
    assert cfg["tests"]["RUBBER-GRIP-02"]["thresholds"] == {"ratio_max": 0.65, "floor": 1.0e-4}
    assert cfg["common"]["options"]["constraint_work"] == "kick_average_v1"
    assert cfg["common"]["options"]["dirichlet"] == "consistent_mass_projection_v1"
    assert cfg["common"]["cfl_set"] == [0.5, 0.25] and cfg["common"]["stable_step_guard"] == 1.0
    assert set(cfg["scenes"]) == {"S1", "S2", "S3"}
    assert cfg["provenance"]["seen_before_declaration"] == ["S1"]
    assert sorted(cfg["provenance"]["unseen_before_declaration"]) == ["S2", "S3"]
