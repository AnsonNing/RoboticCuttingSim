"""Code checks of the rubber_phys_tests_v3 cases on REDUCED configurations (not formal results): static_stop_v3,
the RUBBER-PHYS-07 v3 variants and the RUBBER-PHYS-12 guard regression run end to end and report what their
evaluators need. The formal configurations run only through scripts/rubber/run_rubber_phys.py --protocol v3."""
import copy
import math

import numpy as np
import pytest

from robotic_cutting_sim import rubber_phys as P
from robotic_cutting_sim import rubber_reference as R
from robotic_cutting_sim.rubber_initializer import relax_to_static


def _reduced_v3(L=0.012, W=0.003, t=0.003, n_t=2):
    cfg = copy.deepcopy(P.load_config(P.CONFIG_V3))
    cfg["tests"]["RUBBER-PHYS-06"]["setup"].update(L_m=L, W_m=W, t_m=t, n_t=n_t)
    cfg["tests"]["RUBBER-PHYS-07"]["setup"]["control_interval_s"] = 2e-4
    cfg["tests"]["RUBBER-PHYS-12"]["setup"]["control_interval_s"] = 2e-4
    return cfg


def test_static_stop_v3_uses_float64_markers_and_minimum_time():
    """24 mm cantilever under 9.81 m/s^2 (sag ~ 0.5 mm). Note: the float64 markers carry a float32-force noise floor
    of ~1e-9 m, so the relative tolerance 1e-4 needs a sag >~ 1e-5 m (every formal -02/-03/-05 sag is >= 1.7e-4 m)."""
    live = P.cantilever_live(0.024, 0.003, 0.003, 2, 4e5, 9.81)
    E = live.material.young_effective()
    w1 = 2*math.pi*R.cantilever_f1(live.material.density, 0.003, 0.003, 0.024, E)
    r = relax_to_static(live, omega1=w1, stop_rule="static_stop_v3")
    assert r["converged"] and r["stop_rule"] == "static_stop_v3"
    assert r["time_s"] >= 3*2*math.pi/w1-1e-12 and len(r["markers_f64_m"]) == 9
    assert r["markers_f32_minus_f64_max_m"] < 1e-6
    with pytest.raises(ValueError, match="stop_rule"):
        relax_to_static(live, omega1=w1, stop_rule="static_stop_v9")
    with pytest.raises(ValueError, match="min_time"):
        relax_to_static(live, rate=10.0, check_interval=1e-3, max_time=1e-2, stop_rule="static_stop_v3")


def test_static_sag_v3_reports_both_deflections():
    r = P.case_static_sag(0.024, 0.003, 0.003, 2, 4e5, 9.81, stop_rule="static_stop_v3", stable_step_guard=1.0)
    assert r["stop_rule"] == "static_stop_v3" and r["invalid"] == 0
    assert r["delta_m"] > 0 and abs(r["delta_m"]/r["delta_x_f32_m"]-1) < 0.05
    assert 0.5 < r["cfl_eff_max_end"] < 0.6


@pytest.mark.parametrize("variant", ["ramped", "instant"])
def test_stretch_release_v3_variant_runs_and_evaluates(variant):
    cfg = _reduced_v3()
    r = P.case_stretch_release_v3(cfg, variant)
    phases = {row["phase"] for row in r["rows"]}
    assert phases == ({"ramp", "hold", "release_ramp", "post"} if variant == "ramped" else {"ramp", "hold", "post"})
    assert r["post_steps"] == len([row for row in r["rows"] if row["phase"] == "post"])
    if variant == "ramped":
        assert r["hold_force_n"][0] > 0
    assert r["at_release"]["elastic"] > 0 and r["energy_drift_rel_psi"] is not None
    other = dict(r, variant="ramped" if variant == "instant" else "instant")
    verdicts, measured = P.evaluate_07_v3([r, other], cfg)
    assert set(verdicts) == {f"{v}_{k}" for v in ("ramped", "instant") for k in ("mean_length", "energy", "valid",
                                                                              "no_fold")}
    assert measured[variant]["lambda"] == cfg["tests"]["RUBBER-PHYS-07"]["setup"]["variants"][variant]["lambda"]


def test_guard_regression_case_reports_flag_steps():
    cfg = _reduced_v3()
    r = P.case_guard_regression(cfg)
    assert r["release_after_step"] > 0 and r["stable_step_guard"] == 1.0
    verdicts, measured = P.evaluate_12(r, cfg)
    assert set(verdicts) == {"flagged", "flag_before_nonfinite"}
    fake = dict(r, first_cfl_flag_step=5, first_j_flag_step=None, first_nonfinite_step=7)
    assert P.evaluate_12(fake, cfg)[0] == {"flagged": "PASS", "flag_before_nonfinite": "PASS"}
    fake.update(first_cfl_flag_step=7)
    assert P.evaluate_12(fake, cfg)[0]["flag_before_nonfinite"] == "FAIL"
    fake.update(first_cfl_flag_step=None, first_nonfinite_step=None)
    assert P.evaluate_12(fake, cfg)[0]["flagged"] == "FAIL"
