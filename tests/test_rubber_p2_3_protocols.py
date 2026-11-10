"""P2-3 protocols are frozen at commit (ADR 0023 §21): declared before their formal / implementation runs.

A failure of a pin means a frozen file was edited. Never update a pin: declare the next version instead.
"""
import hashlib
import json

from robotic_cutting_sim.rubber_material import ROOT

PINS = {
    "configs/rubber/rubber_moving_grip_tests_v1.json": "ff14f39dc750fcf095a1a84ba8c7a48d3f0721a27b08062e6e03680eab7678bc",
    "configs/rubber/rubber_visc_tests_v1.json": "95ad1f94a253fe52a388249a7c34a1ecee2b9f040867dd2d65b8de0f7626f080",
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


def test_visc_protocol_declares_every_required_test_with_references_and_thresholds():
    cfg = _load("configs/rubber/rubber_visc_tests_v1.json")
    assert cfg["schema"] == "rcs_rubber_visc_tests_v1" and cfg["version"] == 1
    assert cfg["calibration_status"].startswith("SIMULATOR_ONLY_UNCALIBRATED")
    assert sorted(cfg["tests"]) == [f"RUBBER-VISC-{i:02d}" for i in range(1, 11)]
    grip = cfg["prerequisites"]["rubber_moving_grip_tests_v1"]
    assert grip["sha256"] == PINS["configs/rubber/rubber_moving_grip_tests_v1.json"]
    common = cfg["common"]
    assert common["model"] == "neo_hookean_jbar_prony_v1" and common["pic"] == 0.0
    assert common["options"]["constraint_work"] == "kick_average_v1"
    assert common["options"]["dirichlet"] == "consistent_mass_projection_v1"
    assert "mu0" in common["step_rule"] and "never mu_inf" in common["step_rule"]
    p2 = common["prony_sets"]["P2"]
    assert abs(1-sum(t["g"] for t in p2["terms"])-p2["g_inf"]) < 1e-12
    t = cfg["tests"]
    for tid in ("RUBBER-VISC-01", "RUBBER-VISC-02", "RUBBER-VISC-03", "RUBBER-VISC-04", "RUBBER-VISC-05",
                "RUBBER-VISC-08"):
        assert t[tid]["thresholds"], tid
    for tid in ("RUBBER-VISC-01", "RUBBER-VISC-02", "RUBBER-VISC-03", "RUBBER-VISC-04"):
        assert t[tid]["reference"], tid
    rates = t["RUBBER-VISC-03"]["setup"]
    assert len(rates["fit_rates_hz"]) >= 3 and rates["heldout_rates_hz"]
    assert not set(rates["fit_rates_hz"]) & set(rates["heldout_rates_hz"])
    assert t["RUBBER-VISC-04"]["setup"]["substep_factors"] == [1, 2, 4]
    assert t["RUBBER-VISC-04"]["thresholds"]["tau_dt_rel"] == 0.005
