"""P2-3 protocols are frozen at commit (ADR 0023 §21): declared before their formal / implementation runs.

A failure of a pin means a frozen file was edited. Never update a pin: declare the next version instead.
"""
import hashlib
import json

from robotic_cutting_sim.rubber_material import ROOT

PINS = {
    "configs/rubber/rubber_moving_grip_tests_v1.json": "ff14f39dc750fcf095a1a84ba8c7a48d3f0721a27b08062e6e03680eab7678bc",
    "configs/rubber/rubber_visc_tests_v1.json": "95ad1f94a253fe52a388249a7c34a1ecee2b9f040867dd2d65b8de0f7626f080",
    "configs/rubber/rubber_visc_tests_v2.json": "212fe7af3e92dcb854aa95c8192ecb787f52cb6bd186bbc1d48414b8030c4663",
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


def test_visc_v2_rescopes_03_05_without_moving_any_v1_threshold():
    """ADR 0023 §21.7: v2 corrects the -03 estimator / inertia and the -05 quadrature; every v1 threshold is kept."""
    v1 = _load("configs/rubber/rubber_visc_tests_v1.json")
    v2 = _load("configs/rubber/rubber_visc_tests_v2.json")
    assert v2["schema"] == "rcs_rubber_visc_tests_v2" and v2["version"] == 2
    assert v2["calibration_status"].startswith("SIMULATOR_ONLY_UNCALIBRATED")
    assert v2["supersedes"]["sha256"] == PINS["configs/rubber/rubber_visc_tests_v1.json"]
    assert sorted(v2["tests"]) == ["RUBBER-VISC-03", "RUBBER-VISC-05"]
    assert sorted(v2["carried_over_from_v1"]["tests"]+list(v2["tests"])) == sorted(v1["tests"])
    for tid in v2["tests"]:
        old, new = v1["tests"][tid]["thresholds"], v2["tests"][tid]["thresholds"]
        assert {k: new[k] for k in old} == old, tid                    # no v1 threshold moved
        assert set(new)-set(old) <= {"deembed_consistency_rel"}, tid   # only the new method's validity bound
    assert v2["tests"]["RUBBER-VISC-03"]["thresholds"]["deembed_consistency_rel"] == 0.01/5
    s1, s2 = v1["tests"]["RUBBER-VISC-03"]["setup"], v2["tests"]["RUBBER-VISC-03"]["setup"]
    assert s2["fit_rates_hz"] == s1["fit_rates_hz"] and s2["heldout_rates_hz"] == s1["heldout_rates_hz"]
    assert s2["eps0"] == s1["eps0"] and s2["control_dt_s"] == s1["control_dt_s"]
    assert v2["common"]["options"]["visco_energy"] == "branch_work_kick_average_v2"
    assert {k: v for k, v in v2["common"]["options"].items() if k != "visco_energy"} == {
        k: v for k, v in v1["common"]["options"].items() if k != "visco_energy"}
    assert "dynamics_identity_with_v1" in v2["validity"]
