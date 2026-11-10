"""R0.1: rubber_material_v1 recipe schema, ranges, refusals and derived values (ADR 0023 §14.1)."""
import copy
import json
import math

import pytest

from robotic_cutting_sim import rubber_material as rm

ROOT = rm.ROOT


def _raw():
    return json.loads(rm.DEFAULT_RECIPE.read_text())


def test_default_recipe_loads_and_is_uncalibrated():
    r = rm.load_recipe()
    assert r.id == "rubber_generic_soft_v1"
    assert r.calibration_status == "SIMULATOR_ONLY_UNCALIBRATED"
    assert r.model == "neo_hookean_jbar_v1" and r.prony == ()
    assert r.pic == 0.0 and r.cfl == 0.5 and r.j_domain == (0.2, 5.0)
    assert len(r.source_sha256) == 64


def test_derived_values_match_hand_calculation():
    r = rm.load_recipe()
    d = r.derived(dx=1e-3)
    mu, kappa, rho = 4e5, 8e6, 1150.0
    assert d["mu0_pa"] == mu and d["kappa_pa"] == kappa and d["beta"] == 20.0
    assert d["nu_effective"] == pytest.approx((3*kappa-2*mu)/(2*(3*kappa+mu)), rel=1e-15)
    assert d["nu_effective"] == pytest.approx(0.4754098, abs=1e-7)          # ADR §6: nu_eff 0.4754
    assert d["young_effective_pa"] == pytest.approx(9*kappa*mu/(3*kappa+mu), rel=1e-15)
    assert d["young_effective_pa"]/mu == pytest.approx(2.9508, abs=1e-4)    # ADR §6: E_eff = 2.951 mu
    assert d["c_s_m_s"] == pytest.approx(math.sqrt(mu/rho))
    assert d["c_p_m_s"] == pytest.approx(math.sqrt((kappa+4*mu/3)/rho))
    assert d["dt_max_s"] == pytest.approx(0.5*1e-3/d["c_p_m_s"])
    assert d["lame_lambda_pa"] == pytest.approx(kappa-2*mu/3)


def test_adr_wave_speed_table_beta20():
    # ADR §6 table: mu 0.5 MPa, rho 1150, beta 20 -> c_p 96.3 m/s, dt (CFL 0.5, dx 1 mm) 5.2 us, 3850 substeps
    r = rm.load_recipe()
    rule = rm.step_rule(1e-3, 0.02, mu0_max=5e5, kappa_max=1e7, density_min=1150.0, cfl=0.5)
    assert rule["c_p_max_m_s"] == pytest.approx(96.3, abs=0.05)
    assert rule["dt_max_s"] == pytest.approx(5.2e-6, abs=0.05e-6)
    assert rule["substeps_per_control_step"] == pytest.approx(3850, abs=5)   # ADR: 3,850
    assert rule["dt_s"] <= rule["dt_max_s"] and rule["dt_s"]*rule["substeps_per_control_step"] == pytest.approx(0.02)
    assert rule["cfl_actual"] <= 0.5
    assert r.step_rule(1e-3, 0.02)["mu0_max_pa"] == 2e6   # default: recipe range maximum


def test_step_rule_uses_randomisation_range_maxima():
    r = rm.load_recipe()
    narrow = r.step_rule(1e-3, 0.02, mu_eq_max=1.3e6, density_min=1000.0)
    wide = r.step_rule(1e-3, 0.02)
    assert narrow["substeps_per_control_step"] < wide["substeps_per_control_step"]
    assert narrow["c_p_max_m_s"] == pytest.approx(math.sqrt((20*1.3e6+4*1.3e6/3)/1000))
    with pytest.raises(rm.RubberRecipeError):
        rm.step_rule(1e-3, 0.02, mu0_max=1e6, kappa_max=2e7, density_min=1000., cfl=1.5)


@pytest.mark.parametrize("path,value,message", [
    (("elastic", "poisson_ratio"), 0.499, "refused input"),
    (("poisson_ratio",), 0.49, "refused input"),
    (("numerics", "pic"), 0.05, "pic must be 0"),
    (("bogus",), 1, "unknown keys"),
    (("elastic", "extra"), 1, "unknown keys"),
    (("calibration_status",), "REAL", "calibration_status"),
    (("model",), "neo_hookean_jbar_prony_v1", "does not match"),
    (("numerics", "cfl"), 0.8, "cfl"),
    (("numerics", "particles_per_cell_axis"), 1, "particles_per_cell_axis"),
    (("numerics", "j_domain"), [1.2, 5.0], "j_domain"),
    (("derived",), {"kappa": 1}, "never input"),
])
def test_refusals(path, value, message):
    raw = _raw()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(rm.RubberRecipeError, match=message):
        rm.parse_recipe(raw)


def test_range_and_positivity_refusals():
    raw = _raw()
    raw["elastic"]["shear_modulus_eq_pa"]["value"] = 5e6
    with pytest.raises(rm.RubberRecipeError, match="outside its range"):
        rm.parse_recipe(raw)
    raw = _raw()
    raw["density_kg_m3"]["value"] = float("nan")
    with pytest.raises(rm.RubberRecipeError, match="finite positive"):
        rm.parse_recipe(raw)
    raw = _raw()
    del raw["contact"]
    with pytest.raises(rm.RubberRecipeError, match="missing keys"):
        rm.parse_recipe(raw)


def test_prony_rules_and_coefficients():
    raw = _raw()
    raw["model"] = "neo_hookean_jbar_prony_v1"
    raw["viscoelastic"]["prony"] = [{"g": 0.15, "tau_s": 0.01}, {"g": 0.10, "tau_s": 0.1}]
    r = rm.parse_recipe(raw)
    assert r.g_inf == pytest.approx(0.75)
    assert r.mu0() == pytest.approx(4e5/0.75)
    assert r.kappa() == pytest.approx(20*4e5/0.75)                # kappa = beta mu0 (instantaneous)
    dt = 5e-6
    (a1, b1), (a2, b2) = r.prony_coefficients(dt)
    assert a1 == pytest.approx(math.exp(-dt/0.01), rel=1e-15)
    assert b1 == pytest.approx(0.15*(0.01/dt)*(1-math.exp(-dt/0.01)), rel=1e-12)
    assert b2 == pytest.approx(0.10, rel=1e-4)                    # tau >> dt: b -> g
    bad = copy.deepcopy(raw)
    bad["viscoelastic"]["prony"] = [{"g": 0.6, "tau_s": 0.01}, {"g": 0.4, "tau_s": 0.1}]
    with pytest.raises(rm.RubberRecipeError, match="< 1"):
        rm.parse_recipe(bad)
    bad = copy.deepcopy(raw)
    bad["viscoelastic"]["prony"] = [{"g": 0.1, "tau_s": 0.01}]*5
    with pytest.raises(rm.RubberRecipeError, match="at most"):
        rm.parse_recipe(bad)
    bad = copy.deepcopy(raw)
    bad["viscoelastic"]["prony"][0]["tau_s"] = 0.0
    with pytest.raises(rm.RubberRecipeError):
        rm.parse_recipe(bad)


def test_real_calibrated_requires_frozen_heldout_manifest(tmp_path):
    raw = _raw()
    raw["calibration_status"] = "REAL_CALIBRATED"
    with pytest.raises(rm.RubberRecipeError, match="REAL_CALIBRATED"):
        rm.parse_recipe(raw)
    raw["calibration_manifest"] = "none_v1"
    raw["heldout_verdict"] = "PASS"
    with pytest.raises(rm.RubberRecipeError, match="NO_REAL_DATA"):
        rm.parse_recipe(raw)          # the shipped manifest has no data
    raw["calibration_manifest"] = "missing_v9"
    with pytest.raises(rm.RubberRecipeError, match="not found"):
        rm.parse_recipe(raw, manifest_dir=tmp_path)


def test_values_overrides_stay_in_range():
    r = rm.load_recipe()
    v = r.values(mu_eq=1e6, density=1000.0)
    assert v.mu0 == 1e6 and v.kappa == 2e7 and v.density == 1000.0 and v.beta == 20.0
    with pytest.raises(rm.RubberRecipeError, match="outside the recipe range"):
        r.values(mu_eq=5e6)


def test_resolution_and_lattice_checks():
    r = rm.load_recipe()
    assert rm.resolution_check(r, 0.004, 0.004/3)["label"] == "full"
    with pytest.raises(rm.RubberRecipeError):
        rm.resolution_check(r, 0.004, 0.002)
    assert rm.resolution_check(r, 0.004, 0.002, profile="rubber_coarse_v1")["label"].startswith("reduced_fidelity")
    assert rm.lattice_check(0.06, 0.0005) == 120
    with pytest.raises(rm.RubberRecipeError):
        rm.lattice_check(0.0601, 0.0005)


def test_schema_document_agrees_with_validator_keys():
    schema = json.loads((ROOT/"configs"/"materials"/"rubber_material_v1.schema.json").read_text())
    assert set(schema["properties"]) == rm._TOP_KEYS
    assert set(schema["required"]) == rm._REQUIRED_TOP
    assert set(schema["properties"]["elastic"]["properties"]) == rm._ELASTIC_KEYS
    assert set(schema["properties"]["numerics"]["properties"]) == rm._NUMERICS_KEYS
    assert set(schema["properties"]["contact"]["properties"]) == rm._CONTACT_KEYS
