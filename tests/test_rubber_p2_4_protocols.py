"""P2-4 protocol ``rubber_contact_tests_v1`` is frozen at commit (ADR 0023 §22.6): declared before any formal run.

A failure of the pin means the frozen file was edited. Never update the pin: declare v2 instead.
"""
import hashlib
import json

from robotic_cutting_sim.rubber_material import ROOT

PINS = {
    "configs/rubber/rubber_contact_tests_v1.json": "4d3dc68fc50aeeb247c2b544c65312fd2a651a90a99eb394b11f3fa478b46e4c",
}


def _load(name):
    return json.loads((ROOT/name).read_text())


def test_frozen_files_unchanged():
    for name, pin in PINS.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == pin, f"{name} was modified"


def test_contact_protocol_declares_every_test_with_references_and_thresholds():
    cfg = _load("configs/rubber/rubber_contact_tests_v1.json")
    assert cfg["schema"] == "rcs_rubber_contact_tests_v1" and cfg["version"] == 1
    assert cfg["calibration_status"].startswith("SIMULATOR_ONLY_UNCALIBRATED")
    assert cfg["status"] == "DECLARED_BEFORE_ANY_FORMAL_RUN"
    assert sorted(cfg["tests"]) == [f"RUBBER-CONT-{i:02d}" for i in range(1, 12)]
    common = cfg["common"]
    assert common["contact_law"] == "grid_node_penalty_coulomb_v1"
    assert common["law_parameters"] == {"normal_stiffness_kappa_n": 0.25, "tangential_stiffness_kappa_t": 0.25,
                                        "normal_damping_ratio_zeta_n": 0.5, "tangential_damping_ratio_zeta_t": 0.5}
    assert common["contact_work"].startswith("kick_average_v1")
    assert common["options"]["constraint_work"] == "kick_average_v1"
    assert "never as material or viscoelastic damping" in common["artificial_damping"]
    t = cfg["tests"]
    for tid in ("RUBBER-CONT-01", "RUBBER-CONT-02", "RUBBER-CONT-04", "RUBBER-CONT-05"):
        assert t[tid]["reference"] and t[tid]["thresholds"], tid
    assert t["RUBBER-CONT-03"]["thresholds"] and "2 L / c0" in cfg["scenes"]["IMP"]["reference"]
    assert t["RUBBER-CONT-07"]["thresholds"] and t["RUBBER-CONT-07"]["classification"]
    assert t["RUBBER-CONT-06"]["status"].startswith("NOT_IMPLEMENTED_V1")
    inc = cfg["scenes"]["INC"]
    assert max(inc["stick_ratios_tan_theta_over_mu"]) < 1 < min(inc["slip_ratios_tan_theta_over_mu"])
    assert len(inc["slip_ratios_tan_theta_over_mu"]) >= 3
    assert t["RUBBER-CONT-01"]["thresholds"]["onset_rel"] == 0.02
    assert set(cfg["scenes"]["COUPLE"]["arms"]) == {"A_delayed_20ms", "B_reference_1ms"}
    # the seen probe scenes are not formal scenes
    assert "16 x 8 x 4 mm" in inc["what"] and "40 x 4 x 4 mm" in cfg["scenes"]["IMP"]["what"]


def test_prerequisite_pins_match_the_p2_3_pins():
    cfg = _load("configs/rubber/rubber_contact_tests_v1.json")
    pre = cfg["prerequisites"]
    for name, key in (("configs/rubber/rubber_visc_tests_v2.json", "rubber_visc_tests_v2"),
                      ("configs/rubber/rubber_moving_grip_tests_v1.json", "rubber_moving_grip_tests_v1")):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == pre[key]["sha256"]
