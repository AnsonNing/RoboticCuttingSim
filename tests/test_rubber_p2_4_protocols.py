"""P2-4 protocols ``rubber_contact_tests_v1`` (ADR 0023 §22.6) and ``rubber_contact_tests_v2`` (§22.9, task X1) are
frozen at commit: each declared before any formal run of it.

A failure of a pin means the frozen file was edited. Never update a pin: declare the next version instead.
"""
import hashlib
import json

from robotic_cutting_sim.rubber_material import ROOT

PINS = {
    "configs/rubber/rubber_contact_tests_v1.json": "4d3dc68fc50aeeb247c2b544c65312fd2a651a90a99eb394b11f3fa478b46e4c",
    "configs/rubber/rubber_contact_tests_v2.json": "89bc7680bb9c2ed6322279b198899cef0faf31c4751b6bd849b22ecd24a339d0",
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


def test_contact_protocol_v2_keeps_the_v1_thresholds_and_declares_the_envelope():
    v1 = _load("configs/rubber/rubber_contact_tests_v1.json")
    v2 = _load("configs/rubber/rubber_contact_tests_v2.json")
    assert v2["schema"] == "rcs_rubber_contact_tests_v2" and v2["version"] == 2
    assert v2["status"] == "DECLARED_BEFORE_ANY_FORMAL_RUN"
    assert v2["calibration_status"].startswith("SIMULATOR_ONLY_UNCALIBRATED")
    assert v2["prerequisites"]["rubber_contact_tests_v1"]["sha256"] == PINS["configs/rubber/rubber_contact_tests_v1.json"]
    assert sorted(v2["tests"]) == [f"RUBBER-CONT-{i:02d}" for i in range(1, 12)]
    c = v2["common"]
    assert c["contact_law"] == "grid_node_penalty_coulomb_v2"
    assert c["law_parameters"] == v1["common"]["law_parameters"]                     # not re-tuned
    assert c["options"] == v1["common"]["options"] and c["books_layer_N"] == v1["common"]["books_layer_N"]
    env = c["mu_envelope"]
    assert env["sliding_friction_supported_mu_max"] == 0.6 and env["static_friction_supported_mu_max"] == 0.8
    # every v1 threshold kept (tighter additions allowed); the -05 d metric and the -09 b band replaced with reasons
    for tid in ("RUBBER-CONT-01", "RUBBER-CONT-02", "RUBBER-CONT-03", "RUBBER-CONT-04", "RUBBER-CONT-05"):
        assert v2["tests"][tid]["thresholds"] == v1["tests"][tid]["thresholds"], tid
    assert "f_law_v2" in v2["tests"]["RUBBER-CONT-02"]["criteria"]
    assert "mean_window(F_x(clamp) + T)" in v2["tests"]["RUBBER-CONT-05"]["criteria"]["d_balance"]
    b = v2["tests"]["RUBBER-CONT-09"]
    assert b["thresholds"] == {"k_per_side": 5, "lam": 4.0, "eps_rel": 1e-6}
    assert b["criteria"]["a_zero_rows"] == v1["tests"]["RUBBER-CONT-09"]["criteria"]["a_zero_rows"]
    assert b["criteria"]["c_audits"] == v1["tests"]["RUBBER-CONT-09"]["criteria"]["c_audits"]
    assert b["v1_rule_assessment"].startswith("UNSOUND")
    # the v1 scenes are kept; the envelope scene INC2 is new and unseen
    for name in ("INC", "PEN", "IMP", "PINCH"):
        for k, v in v1["scenes"][name].items():
            assert v2["scenes"][name][k] == v, (name, k)
    assert v2["scenes"]["INC2"]["mu_set"] == [0.6]
    assert any(x.startswith("INC2") for x in v2["provenance"]["unseen_before_declaration"])
