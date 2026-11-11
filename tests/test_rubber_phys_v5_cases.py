"""Code checks of the rubber_phys_tests_v5 changes on REDUCED configurations or manufactured rows (not formal
results): the declared deformation reserve of -08, the guard on -04 / -09 and their v5 evaluation, the -02
extrapolation pair taken from the protocol. Formal v5 runs only through scripts/rubber/run_rubber_phys.py --protocol v5
and scripts/rubber/p2_2_v5/*."""
import copy

from robotic_cutting_sim import rubber_phys as P


def _v5():
    return P.load_config(P.CONFIG_V5)


def test_v5_helpers_read_the_declaration():
    v4, v5 = P.load_config(P.CONFIG_V4), _v5()
    assert P.deformation_reserve(v5, "RUBBER-PHYS-08") == 1.5 and P.deformation_reserve(v4, "RUBBER-PHYS-08") == 1.0
    assert P.test_guard(v5, "RUBBER-PHYS-04") == 1.0 == P.test_guard(v5, "RUBBER-PHYS-09")
    assert P.test_guard(v4, "RUBBER-PHYS-04") is None and P.test_guard(v5, "RUBBER-PHYS-02") is None


def test_shear_with_the_reserve_keeps_the_interval_and_adds_substeps():
    """gamma 0.5 (not a formal configuration): r 1.5 gives 300 substeps over the v4 interval, guard on, exact."""
    a = P.case_shear(0.5, stable_step_guard=1.0)
    b = P.case_shear(0.5, stable_step_guard=1.0, deformation_reserve=1.5)
    assert a["substeps"] == 200 and b["substeps"] == 300 and abs(a["control_dt_s"]-b["control_dt_s"]) < 1e-18
    assert b["invalid"] == 0 and abs(b["sigma_12_rel_error"]) < 1e-4 and b["cfl_eff_max_end"] < a["cfl_eff_max_end"]


def test_vibration_and_records_report_guard_flags():
    cfg = copy.deepcopy(_v5())
    cfg["tests"]["RUBBER-PHYS-04"]["setup"].update(L_m=0.024, W_m=0.003, t_m=0.003, n_t=2, record_interval_s=2e-3)
    r = P.case_vibration("dt", cfg, stable_step_guard=1.0)
    assert r["invalid_flags"] == [] and 0.3 < r["cfl_eff_max"] < 1.0 and r["stable_step_guard"] == 1.0
    cfg["tests"]["RUBBER-PHYS-09"]["setup"].update(control_steps=2, control_dt_s=2e-3)
    rec = P.case_records("cantilever_gravity", 0.003, 1, cfg, stable_step_guard=1.0)
    assert rec["valid"] and [x["step"] for x in rec["step_flags"]] == [1, 2] and rec["stable_step_guard"] == 1.0


def _row09(scenario, flags_per_step, nonfinite_step=None):
    steps = [{"step": i, "flags": f, "nonfinite": i == nonfinite_step} for i, f in enumerate(flags_per_step, 1)]
    return {"scenario": scenario, "t_m": 0.003, "W_m": 0.003, "valid": True, "problems": [],
            "invalid_flags": steps[-1]["flags"], "step_flags": steps, "stable_step_guard": 1.0}


def test_09_v5_evaluation():
    cfg = _v5()
    good = [_row09(sc, [[]]*5) for sc in ("cantilever_gravity", "free_vibration") for _ in range(6)]
    folded = [_row09("stretch_release", [[], [], [], ["RUBBER_CFL_EXCEEDED"], ["RUBBER_CFL_EXCEEDED"]])
              for _ in range(6)]
    v, m = P.evaluate_09_v5(good+folded, cfg)
    assert v == {"records": "PASS", "valid": "PASS", "guard_before_nonfinite": "PASS"}
    late = folded[:5]+[_row09("stretch_release", [[], [], [], ["NONFINITE_STATE"], ["NONFINITE_STATE"]], 4)]
    v, m = P.evaluate_09_v5(good+late, cfg)
    assert v["guard_before_nonfinite"] == "FAIL" and v["valid"] == "PASS"
    flagged = good[:11]+[_row09("free_vibration", [[], ["RUBBER_CFL_EXCEEDED"]])]
    v, m = P.evaluate_09_v5(flagged+folded, cfg)
    assert v["valid"] == "FAIL" and len(m["invalid_cantilever_or_vibration"]) == 1


def test_02_extrapolation_pair_follows_the_protocol():
    """v5 pairs n_t 4 / 6 (r 3/2); v4 3 / 4 (r 4/3); exact order-2 data extrapolate to 1."""
    for cfg, (nc, nf) in ((_v5(), (4, 6)), (P.load_config(P.CONFIG_V4), (3, 4))):
        rows = []
        for c in P.configs_02(cfg):
            d = 1.0+0.01/c["n_t"]**2
            rows.append({"label": c["label"], "L_m": c["L"], "t_m": c["t"], "W_m": c["W"], "n_t": c["n_t"],
                         "delta_m": d, "delta_EB_m": d, "ratio_EB": 1.0, "nu_eff": 0.4754, "reaction_residual": 0.0,
                         "initializer": {"converged": True}})
        assert {r["n_t"] for r in rows} == {nc, nf}
        v, m = P.evaluate_02(rows, cfg)
        assert v["extrapolated"] == "PASS"
        assert len(m["rows"]) == 24
        assert max(abs(r["ratio_extrap"]-(1+0.01/int(r["label"].rsplit("nt", 1)[1])**2)) for r in m["rows"]) < 1e-12
