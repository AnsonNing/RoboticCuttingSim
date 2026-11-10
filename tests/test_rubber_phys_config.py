"""R0.4: the pre-declared RUBBER-PHYS tests are frozen at commit (ADR 0023 §14.3).

A failure means a frozen file was edited. Never update a pin: declare the next version (v6) instead.
"""
import hashlib
import json

from robotic_cutting_sim.rubber_material import ROOT

PINS = {
    "configs/rubber/rubber_phys_tests_v1.json": "22e2e5c0e2b3e30e70483b1d9cf64f2990ab5c6e6f81347e5a7c66f3345e8df9",
    "configs/rubber/rubber_record_v1.schema.json": "1a1a941e188a5261708f7b5403a7ddf6d6d519b5cf04ad05a78eba43b273d08a",
    "configs/rubber/rubber_phys_tests_v2.json": "e4ff8502817458fe723295ead88d9531757b1cff5c771d5d8c352828bab83060",
    "configs/rubber/rubber_phys_tests_v3.json": "c7fdfc95f8bd024cca82a8dd00f6949a0410a0f2c65beae262c32696a848dbd9",
    "configs/rubber/rubber_phys_tests_v4.json": "f1cecee63375fb3cfef7a1319e851df472bd2f7e3e2611a2a172b108b06b38f0",
    "configs/rubber/rubber_phys_tests_v5.json": "7e02f57bd8c7a09c6163fe26d7c7d37f46689314b7e119714fe7285a8ad867f0",
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


def test_v2_differs_from_v1_only_in_the_initializer_stop_rule():
    v1 = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v1.json").read_text())
    v2 = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v2.json").read_text())
    assert v2["supersedes"]["sha256"] == PINS["configs/rubber/rubber_phys_tests_v1.json"]
    assert v1["tests"] == v2["tests"] and v1["p2_1_scope"] == v2["p2_1_scope"]
    c1, c2 = dict(v1["common"]), dict(v2["common"])
    s1, s2 = c1.pop("static_initializer"), c2.pop("static_initializer")
    assert c1 == c2
    assert {k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k)} == {"stop", "stop_v1_status"}


def test_v3_differs_from_v2_only_in_the_declared_items():
    """P2-1 follow-up (ADR 0023 §18): static_stop_v3 + float64 static deflection (-02/-03/-05 notes only), the
    stable-step guard, the redesigned -07 and the new -12; every other test, criterion and threshold is v2's."""
    v2 = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v2.json").read_text())
    v3 = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v3.json").read_text())
    assert v3["supersedes"]["sha256"] == PINS["configs/rubber/rubber_phys_tests_v2.json"]
    assert v3["version"] == 3 and v3["id"] == "rubber_phys_tests_v3"
    assert set(v3["tests"]) == set(v2["tests"]) | {"RUBBER-PHYS-12"}
    for tid, test in v2["tests"].items():
        if tid == "RUBBER-PHYS-07":
            continue
        t3 = dict(v3["tests"][tid])
        if tid in ("RUBBER-PHYS-02", "RUBBER-PHYS-03", "RUBBER-PHYS-05"):
            assert "static_stop_v3" in t3.pop("v3_change")
        assert t3 == test, tid
    t7 = v3["tests"]["RUBBER-PHYS-07"]
    assert t7["thresholds"]["length_rel"] == v2["tests"]["RUBBER-PHYS-07"]["thresholds"]["length_rel"]
    assert t7["thresholds"]["energy_rel_psi"] == v2["tests"]["RUBBER-PHYS-07"]["thresholds"]["energy_rel"]
    assert {v["lambda"] for v in t7["setup"]["variants"].values()} == {1.5, 1.03}
    c2, c3 = dict(v2["common"]), dict(v3["common"])
    assert c3.pop("stable_step_guard")["limit"] == 1.0
    s2, s3 = c2.pop("static_initializer"), c3.pop("static_initializer")
    assert s3["stop_rule_id"] == "static_stop_v3" and {k for k in set(s2) | set(s3) if s2.get(k) != s3.get(k)} == {
        "stop", "stop_rule_id", "stop_v2_status"}
    for key in ("step_rule", "tip"):
        assert c3.pop(key).startswith(c2.pop(key))
    assert c2 == c3


def test_v4_differs_from_v3_only_in_the_declared_items():
    """P2-2 (ADR 0023 §19): solver options (-02/-03/-05 f64 shadow, -07 projection + grid traction work), the L3 v4 rule,
    the specified refinement study and the new throughput / batched sections; thresholds of every v3 test unchanged."""
    v3 = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v3.json").read_text())
    v4 = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v4.json").read_text())
    assert v4["supersedes"]["sha256"] == PINS["configs/rubber/rubber_phys_tests_v3.json"]
    assert v4["version"] == 4 and v4["id"] == "rubber_phys_tests_v4" and set(v4["tests"]) == set(v3["tests"])
    options = {"RUBBER-PHYS-02": {"state_precision": "f64_shadow_v1"}, "RUBBER-PHYS-03": {"state_precision": "f64_shadow_v1"},
               "RUBBER-PHYS-05": {"state_precision": "f64_shadow_v1"},
               "RUBBER-PHYS-07": {"dirichlet": "consistent_mass_projection_v1", "traction_work": "grid_displacement_v1"}}
    for tid, test in v3["tests"].items():
        t4 = json.loads(json.dumps(v4["tests"][tid]))
        if tid == "RUBBER-PHYS-11":
            assert t4["thresholds"] == {"gci": 0.02, "dt_rel": 0.001} == test["thresholds"]
            continue
        t4.pop("v4_change", None)
        if tid in options:
            assert t4["setup"].pop("solver_options") == options[tid]
        if tid == "RUBBER-PHYS-10":
            assert t4["setup"].pop("L3_v4")["rule"].startswith("rubber_l3_band_v4")
            t4["criteria"].pop("L3_v4")
        assert t4 == test, tid
    c3, c4 = dict(v3["common"]), dict(v4["common"])
    c4.pop("solver_options")
    g3, g4 = c3.pop("stable_step_guard"), c4.pop("stable_step_guard")
    assert g3["limit"] == g4["limit"] == 1.0
    assert c3 == c4


def test_v5_differs_from_v4_only_in_the_declared_items():
    """R2.5 review decisions (a) / (b), F7, F9 (ADR 0023 section 21): production grid n_t 4 for the bending-dominated
    tests, guard on for -04 / -09, -08 with the derived deformation reserve, -11 v5; the -11 rule block, thresholds,
    CFL set, guard and verdict CFL byte-identical to v4; every threshold of every test unchanged."""
    v4 = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v4.json").read_text())
    v5 = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v5.json").read_text())
    assert v5["supersedes"]["sha256"] == PINS["configs/rubber/rubber_phys_tests_v4.json"]
    assert v5["version"] == 5 and v5["id"] == "rubber_phys_tests_v5" and set(v5["tests"]) == set(v4["tests"])
    for tid in v4["tests"]:
        assert v5["tests"][tid]["thresholds"] == v4["tests"][tid]["thresholds"], tid
    for tid in ("RUBBER-PHYS-01", "RUBBER-PHYS-06", "RUBBER-PHYS-07", "RUBBER-PHYS-10", "RUBBER-PHYS-12"):
        assert v5["tests"][tid] == v4["tests"][tid], tid
    # -11: the rule block, thresholds, CFL set, guard and verdict CFL stay byte-identical (serialised)
    s4, s5 = v4["tests"]["RUBBER-PHYS-11"]["setup"], v5["tests"]["RUBBER-PHYS-11"]["setup"]
    for key in ("rule", "cfl", "stable_step_guard", "verdict_cfl", "particles_per_cell_axis"):
        assert json.dumps(s5[key]) == json.dumps(s4[key]), key
    assert v5["tests"]["RUBBER-PHYS-11"]["thresholds"] == {"gci": 0.02, "dt_rel": 0.001}
    assert s5["grids_n_t"] == [3, 4, 6, 8] and s5["production_n_t"] == 4 and s5["triplet_finest"] == [4, 6, 8]
    assert set(s5["observables"]) == {"Q1", "Q1w", "Q2", "Q4"}
    for q in ("Q1", "Q2", "Q4"):
        assert s5["observables"][q] == s4["observables"][q], q
    q1w = dict(s5["observables"]["Q1w"])
    assert q1w.pop("W_over_t") == 4 and q1w.pop("kind")
    assert q1w == {k: v for k, v in s4["observables"]["Q1"].items() if k not in ("W_over_t", "kind")}
    assert s5["solver_options"]["Q1w"] == {"state_precision": "f64_shadow_v1"}
    # bending-dominated tests at n_t 4 (-02 extrapolation pair 4 / 6), geometry otherwise unchanged
    t = {tid: json.loads(json.dumps(v5["tests"][tid])) for tid in v5["tests"]}
    assert t["RUBBER-PHYS-02"]["setup"].pop("n_t") == [4, 6]
    assert t["RUBBER-PHYS-03"]["setup"].pop("n_t") == 4 and t["RUBBER-PHYS-05"]["setup"].pop("n_t") == 4
    assert t["RUBBER-PHYS-04"]["setup"].pop("n_t") == 4
    assert t["RUBBER-PHYS-03"]["setup"]["series_t"].pop("L_m") == 0.096
    for tid in ("RUBBER-PHYS-02", "RUBBER-PHYS-05"):
        a, b = dict(t[tid]["setup"]), dict(v4["tests"][tid]["setup"])
        b.pop("n_t")
        assert a == b, tid
    # -08: r from the declared rule, everything else of the measurement unchanged
    s8 = t["RUBBER-PHYS-08"]["setup"]
    assert s8["deformation_reserve"] == 1.5 == s8["deformation_reserve_rule"]["r"]
    assert s8["gamma"] == v4["tests"]["RUBBER-PHYS-08"]["setup"]["gamma"] and s8["gamma_max"] == 2.0
    assert t["RUBBER-PHYS-08"]["criteria"]["shear"] == v4["tests"]["RUBBER-PHYS-08"]["criteria"]["shear"]
    assert t["RUBBER-PHYS-08"]["measurement"] == v4["tests"]["RUBBER-PHYS-08"]["measurement"]
    c4, c5 = dict(v4["common"]), dict(v5["common"])
    c5.pop("production_grid")
    g4, g5 = c4.pop("stable_step_guard"), c5.pop("stable_step_guard")
    assert g4.pop("applies_to") and g5.pop("applies_to") and g4 == g5 and g5["limit"] == 1.0
    assert c4 == c5
    for key in ("throughput_v4", "batched_gpu_v4", "batched_l0_v4", "p2_2_scope"):
        assert v5[key] == v4[key], key


def test_v5_deformation_reserve_follows_the_declared_rule():
    """-08 v5: r = ceil(2 CFL_eff,max(r = 1) / (0.9 limit)) / 2 at gamma_max 2 (R2.5 review decision (b))."""
    import math

    from robotic_cutting_sim import rubber_reference as R
    from robotic_cutting_sim.rubber_constitutive import wave_modulus_bound_np
    v5 = json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v5.json").read_text())
    s8 = v5["tests"]["RUBBER-PHYS-08"]["setup"]
    mu, kappa, rho = s8["mu_pa"], 20.0*s8["mu_pa"], 1150.0
    F = R.simple_shear(s8["gamma_max"], mu)["F"]
    cfl1 = 0.5*math.sqrt(wave_modulus_bound_np(F, mu, kappa, 1.0)/rho)/math.sqrt((kappa+4*mu/3)/rho)
    limit = v5["common"]["stable_step_guard"]["limit"]
    assert math.ceil(2*cfl1/(0.9*limit))/2 == s8["deformation_reserve"] == 1.5
    assert abs(cfl1-1.2966) < 1e-4 and cfl1/1.5 < 0.9*limit
