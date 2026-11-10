"""Compact evidence ``docs/evidence/rl_ready/p2_4_contact_x1_001.json`` of P2-4 task X1 (ADR 0023 §22.9) from the raw
outputs in ``artifacts/rl_ready/p2_4`` (reads only; never re-runs anything; refuses to overwrite).

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/write_contact_x1_evidence.py \
        --formal artifacts/rl_ready/p2_4/contact_v2_001 --v1 artifacts/rl_ready/p2_4/contact_v1_001 \
        --diag artifacts/rl_ready/p2_4/x1_diag_001 --cpu-tests artifacts/rl_ready/p2_4/cpu_tests_x1_001 \
        --out docs/evidence/rl_ready/p2_4_contact_x1_001.json
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _tail(path, n=2):
    p = Path(path)
    return p.read_text().strip().splitlines()[-n:] if p.exists() else ["MISSING"]


def _key(r):
    a = r["args"]
    return " ".join(f"{k}={v}" for k, v in a.items() if k not in ("ts",))


def _row(r):
    return {"slip_err_over_g_cos": r.get("slip_err_gcos"), "dissipation_ratio_run": r.get("D04_ratio_run"),
            "friction_efficiency": r.get("friction_efficiency"), "bottom_contact_fraction": r.get("contact_fraction_bottom"),
            "normal_impulse_fluctuation": r.get("normal_impulse_fluctuation"), "stick_disp_m": r.get("stick_disp_m"),
            "stick_load_share": r.get("stick_load_share"), "slip_misalignment_loss": r.get("slip_misalignment_loss")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--cpu-tests", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT/a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    formal, v1, diag = ROOT/a.formal, ROOT/a.v1, ROOT/a.diag
    meta = json.loads((formal/"meta.json").read_text())
    s2 = json.loads((formal/"summary.json").read_text())
    s1 = json.loads((v1/"summary.json").read_text())
    proto = ROOT/"configs"/"rubber"/"rubber_contact_tests_v2.json"
    A = {_key(r): r for r in _jsonl(diag/"sweepA_v1.jsonl")}
    K = {_key(r): r for r in _jsonl(diag/"sweepK_v2_errormodel.jsonl")}
    C = _jsonl(diag/"sweepC_rest_frictionless.jsonl")[0]["series"]
    G = {_key(r): r for r in _jsonl(diag/"sweepG_dev.jsonl")}
    J = {_key(r): r for r in _jsonl(diag/"sweepJ_dev.jsonl")}
    hs = (ROOT/"artifacts"/"rl_ready"/"p2_4"/"x1_halfspace_stability_001.txt").read_text().splitlines()
    cal = _jsonl(ROOT/"artifacts"/"rl_ready"/"p2_4"/"x1_cont09_band_calibration_001.jsonl")

    def verdicts(s, tid):
        v = s.get(tid, {})
        return {"verdict": v.get("verdict"), "criteria": v.get("criteria"), "reason": v.get("reason")}

    d1, d2 = s1["RUBBER-CONT-01"]["detail"], s2["RUBBER-CONT-01"]["detail"]
    ev = {
        "schema": "rcs_rl_ready_evidence_v1", "id": "p2_4_contact_x1_001",
        "phase": "P2-4 rubber contact, task X1 (decision on the v1 contact FAILs)",
        "adr": "docs/adr/0023-rubber-material-v1.md section 22.9",
        "report": "docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md section 8",
        "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
        "protocol": {"path": "configs/rubber/rubber_contact_tests_v2.json",
                     "sha256": hashlib.sha256(proto.read_bytes()).hexdigest(), "declared_at_commit": "8842573",
                     "law_commit": "0b33246"},
        "decision": json.loads(proto.read_text())["decision_x1"],
        "mu_envelope": json.loads(proto.read_text())["common"]["mu_envelope"],
        "root_cause_evidence_DIAGNOSTIC": {
            "frictionless_rest_v1": {"normal_work_material_J_at_0.3s": C["W_n"][-1],
                                     "nominal_dashpot_dissipation_J_at_0.3s": C["damping"][-1],
                                     "KE_vib_J_at_0.3s": C["KE_vib"][-1], "psi_J_at_0.3s": C["psi"][-1]},
            "frictionless_rest_v1_vs_v2_0.3s": {k.split("law=")[-1]: _row(r) | {"KE_vib_end_over_S": r["KE_vib_end_over_S"]}
                                                for k, r in K.items() if "theta=0" in k},
            "v1_mu_sweep_ratio_1.2": {k: _row(r) for k, r in A.items() if k.endswith("ratio=1.2") and len(k.split()) == 2},
            "v1_refinement_mu_0.8": {k: _row(r) for k, r in A.items() if k.startswith("mu=0.8 ratio=1.2 ")},
            "v2_error_model": {k: _row(r) for k, r in K.items() if "theta" not in k},
            "normal_force_lag_not_adopted": {"sliding (capmode 2, ntau 100 / 200 / 400)": {k: _row(r) for k, r in G.items()
                                                                                         if "capmode=2" in k},
                                             "stick_instant_lag_in_slip (capmode 7)": {k: _row(r) for k, r in J.items()}},
            "halfspace_stability": hs,
        },
        "formal_run_v2": {"dir": "artifacts/rl_ready/p2_4/contact_v2_001", "meta": meta,
                          "verdicts": {t: verdicts(s2, t) for t in sorted(k for k in s2 if k.startswith("RUBBER"))},
                          "errors": s2.get("errors")},
        "v1_vs_v2": {
            "RUBBER-CONT-01_slip": {"v1": [(q["run"], q["err_over_gcos"]) for q in d1["b_slip"]],
                                    "v2": [(q["run"], q["err_over_gcos"]) for q in d2["b_slip"]]},
            "RUBBER-CONT-01_onset": {"v1": [(q["mu"], q["rel"]) for q in d1["c_onset"]],
                                     "v2": [(q["mu"], q["rel"]) for q in d2["c_onset"]]},
            "RUBBER-CONT-01_stick": {"v1": [(q["run"], q["disp_rel_s"]) for q in d1["a_stick"]],
                                     "v2": [(q["run"], q["disp_rel_s"]) for q in d2["a_stick"]]},
            "RUBBER-CONT-01_books": {"v1": [(q["run"], q.get("books_max_rel", q.get("ratio"))) for q in d1["d_books"]],
                                     "v2": [(q["run"], q.get("books_max_rel", q.get("ratio"))) for q in d2["d_books"]]},
            "RUBBER-CONT-04": {"v1": [(q["run"], q["ratio"]) for q in s1["RUBBER-CONT-04"]["detail"]],
                               "v2": [(q["run"], q["ratio"]) for q in s2["RUBBER-CONT-04"]["detail"]]},
            "RUBBER-CONT-02_f": s2["RUBBER-CONT-02"]["detail"].get("f_law_v2"),
            "RUBBER-CONT-03_books": {"v1": [(q["run"], q.get("max_rel", q.get("ratio"))) for q in
                                            s1["RUBBER-CONT-03"]["detail"]["a_books"]],
                                     "v2": [(q["run"], q.get("max_rel", q.get("ratio"))) for q in
                                            s2["RUBBER-CONT-03"]["detail"]["a_books"]]},
            "RUBBER-CONT-05": {"v1_slip_ratio": [q["ratio"] for q in s1["RUBBER-CONT-05"]["detail"]["a_slip_ratio"]],
                               "v2_slip_ratio": [q["ratio"] for q in s2["RUBBER-CONT-05"]["detail"]["a_slip_ratio"]],
                               "v1_per_step_balance": [q["rel"] for q in s1["RUBBER-CONT-05"]["detail"]["d_balance"]],
                               "v2_window_mean_balance": [q["rel"] for q in s2["RUBBER-CONT-05"]["detail"]["d_balance"]],
                               "v2_per_step_reported": s2["RUBBER-CONT-05"]["detail"]["d_balance_v1_per_step_reported"]},
        },
        "cont09_band": {"v1_assessment": json.loads(proto.read_text())["tests"]["RUBBER-CONT-09"]["v1_rule_assessment"],
                        "v2_rule": json.loads(proto.read_text())["tests"]["RUBBER-CONT-09"]["criteria"]["b_perturbation_band"],
                        "calibration_rows": cal, "gpu_rerun": "NOT_RUN (task B2-v2)"},
        "cpu_tests": {"no_cuda": _tail(ROOT/a.cpu_tests/"cpu_novisible.log"),
                      "cuda_visible_0": _tail(ROOT/a.cpu_tests/"cpu_visible0.log")},
        "frozen": "v1 contact kernels byte-identical (source hashes pinned in tests/test_rubber_contact_v2.py; four "
                  "contact_v1_001 jobs rerun byte-identical in x1_v1_identity_002); rubber_contact_tests_v1 and its "
                  "results unchanged; no Phase 1 module, configs/rl_benchmark/*, scripts/rl_ready/* or frozen rubber "
                  "module changed",
    }
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(out)


if __name__ == "__main__":
    main()
