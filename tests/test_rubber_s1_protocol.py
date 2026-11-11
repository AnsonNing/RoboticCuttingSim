"""S1 GPU check declaration (ADR 0023 §23.2 decision 2) is frozen at commit: declared before any GPU run of the check.

A failure means the frozen file was edited. Never update the pin: declare the next version instead.
"""
import hashlib
import json

from robotic_cutting_sim.rubber_material import ROOT

PINS = {"configs/rubber/rubber_selfpen_check_v1.json": "cc5cf3339e0c69cff13517686833e2293c101c4242b3702ef9980bb5d08b8bc0"}


def test_frozen_files_unchanged():
    for name, pin in PINS.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == pin, f"{name} was modified"


def test_declaration_content():
    cfg = json.loads((ROOT/"configs"/"rubber"/"rubber_selfpen_check_v1.json").read_text())
    assert cfg["id"] == "rubber_selfpen_check_v1" and cfg["calibration_status"] == "SIMULATOR_ONLY_UNCALIBRATED"
    assert cfg["detector"]["check_substeps"] == 256 and cfg["common"]["min_free_mb_at_start"] == 4096
    assert set(cfg["rows"]) == {"COST", "L2", "TRIP", "NO_FALSE_TRIP"}
