"""R0.4: the pre-declared RUBBER-PHYS tests are frozen at commit (ADR 0023 §14.3).

A failure means a frozen file was edited. Never update the pin: declare rubber_phys_tests_v2 instead.
"""
import hashlib
import json

from robotic_cutting_sim.rubber_material import ROOT

PINS = {
    "configs/rubber/rubber_phys_tests_v1.json": "22e2e5c0e2b3e30e70483b1d9cf64f2990ab5c6e6f81347e5a7c66f3345e8df9",
    "configs/rubber/rubber_record_v1.schema.json": "1a1a941e188a5261708f7b5403a7ddf6d6d519b5cf04ad05a78eba43b273d08a",
}


def test_frozen_files_unchanged():
    for name, pin in PINS.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == pin, f"{name} was modified"


def test_every_adr_test_is_declared_with_thresholds():
    cfg = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v1.json").read_text())
    assert cfg["schema"] == "rcs_rubber_phys_tests_v1"
    assert sorted(cfg["tests"]) == [f"RUBBER-PHYS-{i:02d}" for i in range(1, 12)]
    for test_id, test in cfg["tests"].items():
        assert {"name", "phase", "setup", "reference", "criteria", "thresholds"} <= set(test), test_id
    assert cfg["common"]["pic"] == 0.0 and cfg["common"]["bulk_ratio_beta"] == 20.0
    assert cfg["common"]["artificial_damping_in_dynamics"] == "none"


def test_workload_e_is_separate_from_phase1_workloads():
    e = json.loads((ROOT/"configs"/"rubber"/"workload_e_v1.json").read_text())
    assert [w["id"] for w in e["workloads"]] == ["E"]
    # Phase 1 keeps only its placeholder row for E (unchanged file); the definition lives in configs/rubber/.
    phase1 = json.loads((ROOT/"configs"/"rl_benchmark"/"workloads_v1.json").read_text())
    placeholder = [w for w in phase1["workloads"] if w["id"] == "E"]
    assert len(placeholder) == 1 and placeholder[0]["phase"] == "Phase 2"
    assert e["workloads"][0]["calibration_status"] == "SIMULATOR_ONLY_UNCALIBRATED"
