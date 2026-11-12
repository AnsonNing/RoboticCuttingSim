"""R2.2: the declared v4 refinement job list and its evaluation on manufactured values (no physics runs)."""
import json

from robotic_cutting_sim import rubber_refinement as RF
from robotic_cutting_sim.rubber_material import ROOT


def _cfg():
    return json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v4.json").read_text())


def test_job_list_covers_grids_cfl_beta_and_quadrature():
    jobs = RF.jobs(_cfg())
    labels = [label for _, label, _ in jobs]
    assert len(labels) == len(set(labels)) == 4*(4*2+2+1)
    assert "Q1_nt6_cfl0.25_beta20_ppc2" in labels and "Q3_nt4_cfl0.5_beta20_ppc3" in labels
    assert "Q2_nt4_cfl0.5_beta50_ppc2" in labels


def test_evaluation_rule_on_manufactured_values():
    cfg = _cfg()
    rows = []
    for qid, label, run in RF.jobs(cfg):
        q = 1.0+0.05*(1/run["n_t"])**2*(1 if run["cfl"] == 0.5 else 1.0002)
        if qid == "Q3":
            q = 4.1e5*(1+1e-6*run["n_t"])           # grid independent
        rows.append({"status": "OK", "qid": qid, "label": label, "run": run, "Q": q, "invalid": 0, "converged": True})
    verdicts, measured = RF.evaluate(rows, cfg)
    assert verdicts["Q1_gci_production"] == "PASS" and verdicts["Q1_dt"] == "PASS" and verdicts["Q1_valid"] == "PASS"
    assert measured["Q1"]["gci_cfl0.5"]["finest"]["p_obs"] > 1.9
    assert measured["Q3"]["gci_cfl0.5"]["finest"]["status"] == "GRID_INDEPENDENT"
    rows = [r for r in rows if r["label"] != "Q2_nt6_cfl0.5_beta20_ppc2"]
    verdicts, measured = RF.evaluate(rows, cfg)
    assert verdicts["Q2_gci_production"] == "NOT_RUN" and verdicts["Q2_valid"] == "FAIL"


def _cfg_v5():
    return json.loads((ROOT/"configs"/"rubber"/"rubber_phys_tests_v5.json").read_text())


def test_v5_job_list():
    jobs = RF.jobs_v5(_cfg_v5())
    labels = [label for _, label, _ in jobs]
    assert len(labels) == len(set(labels)) == 4*(4+2)
    assert {q for q, _, _ in jobs} == {"Q1", "Q1w", "Q2", "Q4"}
    assert "Q1w_nt8_cfl0.5_beta20_ppc2" in labels and "Q1w_nt4_cfl0.25_beta20_ppc2" in labels
    assert "Q1_nt8_cfl0.25_beta20_ppc2" not in labels and all(run["ppc"] == 2 for _, _, run in jobs)


def test_v5_evaluation_uses_the_finest_triplet_and_reports_the_nt3_bias():
    cfg = _cfg_v5()
    rows = []
    for qid, label, run in RF.jobs_v5(cfg):
        q = 1.0+0.05*(1/run["n_t"])**2*(1 if run["cfl"] == 0.5 else 1.0002)
        rows.append({"status": "OK", "qid": qid, "label": label, "run": run, "Q": q, "invalid": 0, "converged": True})
    verdicts, measured = RF.evaluate_v5(rows, cfg)
    assert set(verdicts.values()) == {"PASS"} and len(verdicts) == 12
    fine = measured["Q1"]["triplet_finest"]
    assert fine["grids_n_t"] == [8, 6, 4] and abs(fine["p_obs"]-2) < 1e-6 and fine["fs"] == 1.25
    assert abs(fine["r21"]-4/3) < 1e-12 and abs(fine["r32"]-1.5) < 1e-12
    bias = measured["Q1"]["rubber_rl_v1_bias_reported_not_scored"]
    assert abs(bias["phi_ext_p_eff"]-1.0) < 1e-9 and abs(bias["nt3_over_ext_p_eff_minus_1"]-0.05/9) < 1e-9
    # the v5 triplet helper equals the v4 evaluation on the same triplet (same rule block)
    v4_like = [dict(r) for r in rows if r["run"]["n_t"] in (3, 4, 6) and r["run"]["cfl"] == 0.5]
    t = RF._triplet({(r["qid"], r["run"]["n_t"], 0.5, 20.0, 2): r for r in v4_like}, "Q1", [3, 4, 6], 0.5,
                    cfg["tests"]["RUBBER-PHYS-11"]["setup"]["rule"])
    assert t["gci_coarse"] == measured["Q1"]["triplet_coarsest"]["gci_coarse"]
    # an order outside the verified window takes Fs 3 (rule unchanged); a missing run is NOT_RUN / invalid
    rows2 = [dict(r, Q=1.0+0.05*(1/r["run"]["n_t"])**3.5) if r["qid"] == "Q2" else r for r in rows]
    verdicts, measured = RF.evaluate_v5(rows2, cfg)
    assert measured["Q2"]["triplet_finest"]["fs"] == 3.0 and measured["Q2"]["triplet_finest"]["p_eff"] == 2.0
    rows3 = [r for r in rows if r["label"] != "Q1w_nt8_cfl0.5_beta20_ppc2"]
    verdicts, _ = RF.evaluate_v5(rows3, cfg)
    assert verdicts["Q1w_gci_production"] == "NOT_RUN" and verdicts["Q1w_valid"] == "FAIL"
