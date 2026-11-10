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
