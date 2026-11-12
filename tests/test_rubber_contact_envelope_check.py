"""ENV2 declaration ``configs/rubber/rubber_contact_envelope_v1.json`` (ADR 0023 §23.2 decision 4) is frozen at commit:
declared before any run of it. A failure means the frozen file was edited. Never update the pin: declare v2."""
import hashlib
import json

from robotic_cutting_sim.rubber_material import ROOT

PINS = {"configs/rubber/rubber_contact_envelope_v1.json":
        "d7ecbd3164409cce12d229504864297677baf3307f9763bb7797329506ffe92c",
        "configs/rubber/rubber_contact_envelope_v2.json":
        "1f6163997bdd072e961b2724de951d81f2db29f889d286ce1d1e3c4ba63f1e18"}


def test_frozen_files_unchanged():
    for name, pin in PINS.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == pin, f"{name} was modified"


def test_declaration_content():
    cfg = json.loads((ROOT/"configs"/"rubber"/"rubber_contact_envelope_v1.json").read_text())
    v3 = json.loads((ROOT/"configs"/"rubber"/"rubber_contact_tests_v3.json").read_text())
    assert cfg["id"] == "rubber_contact_envelope_v1" and cfg["common"]["contact_law"] == "grid_node_penalty_coulomb_v3"
    for key in ("law_parameters", "options", "material_elastic", "material_prony_P2"):
        assert cfg["common"][key] == v3["common"][key]
    t1, t4 = v3["tests"]["RUBBER-CONT-01"]["thresholds"], v3["tests"]["RUBBER-CONT-04"]["thresholds"]
    assert cfg["tests"]["ENV-01"]["thresholds"] == {"slip_acc_rel": t1["slip_acc_rel"],
                                                     "dissipation_rel": t4["dissipation_rel"]}
    assert cfg["tests"]["ENV-02"]["thresholds"] == {k: t1[k] for k in ("stick_disp_rel", "stick_acc_rel",
                                                                        "ramp_disp_rel")}
    assert cfg["scenes"]["INC2"]["n_t_set"] == [5, 6] and cfg["scenes"]["STRIP"]["n_t_set"] == [4, 5, 6]
    assert cfg["scenes"]["INC2"]["mu_set"] == cfg["scenes"]["STRIP"]["mu_set"] == [0.3, 0.4, 0.6, 0.8]


def test_declaration_v2_content():
    """rubber_contact_envelope_v2 (after K2-1): law v2 at n_t 4-6, mu 0.3 / 0.4, spin-down scored, drag reported."""
    cfg = json.loads((ROOT/"configs"/"rubber"/"rubber_contact_envelope_v2.json").read_text())
    v2 = json.loads((ROOT/"configs"/"rubber"/"rubber_contact_tests_v2.json").read_text())
    v1 = json.loads((ROOT/"configs"/"rubber"/"rubber_contact_envelope_v1.json").read_text())
    assert cfg["common"]["contact_law"] == "grid_node_penalty_coulomb_v2"
    assert cfg["common"]["law_parameters"] == {k: v for k, v in v1["common"]["law_parameters"].items()
                                               if k != "slip_rate_damping_ratio_zeta_d"}
    assert cfg["common"]["options"] == v2["common"]["options"] if "options" in v2["common"] else True
    assert cfg["tests"]["ENV-01"] == v1["tests"]["ENV-01"] and cfg["tests"]["ENV-02"] == v1["tests"]["ENV-02"]
    assert cfg["tests"]["ENV-04"]["thresholds"] == {"spin_rel": 0.05}
    for s in ("INC2", "STRIP", "SPIN"):
        assert cfg["scenes"][s]["n_t_set"] == [4, 5, 6] and cfg["scenes"][s]["mu_set"] == [0.3, 0.4]
