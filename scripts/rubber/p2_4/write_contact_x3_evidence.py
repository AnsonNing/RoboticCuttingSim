"""Compact evidence ``docs/evidence/rl_ready/p2_4_contact_x3_001.json`` of P2-4 task X3 (ADR 0023 §22.11) from the raw
outputs (reads only; never re-runs anything; refuses to overwrite): the formal CPU run of ``rubber_contact_tests_v3``,
the GPU RUBBER-CONT-09 run, the GPU cost rows (batched from the -09 run, single env v2 vs v3) and the CPU suites.

    .venv/bin/python scripts/rubber/p2_4/write_contact_x3_evidence.py --formal artifacts/rl_ready/p2_4/contact_v3_001 \
        --v2 artifacts/rl_ready/p2_4/contact_v2_001 --cont09 artifacts/rl_ready/p2_4/gpu_cont09_v3_001 \
        --cont09-v2 artifacts/rl_ready/p2_4b/gpu_cont09_v2_001 --cost-v2 artifacts/rl_ready/p2_4/gpu_contact_cost_x3_v2_001 \
        --cost-v3 artifacts/rl_ready/p2_4/gpu_contact_cost_x3_v3_001 --cpu-tests artifacts/rl_ready/p2_4/cpu_tests_x3_001 \
        --out docs/evidence/rl_ready/p2_4_contact_x3_001.json
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_tests_v3.json"


def _tail(path, n=2):
    p = Path(path)
    return p.read_text().strip().splitlines()[-n:] if p.exists() else ["MISSING"]


def _r(x, n=5):
    return None if x is None else float(f"{x:.{n}g}")


def _group_table(scored):
    """Per (scene, n_t): worst |slip error|, D04 range, worst hold / ramp displacement (relative to s), books."""
    out = {}
    for m in scored:
        g = out.setdefault(f"{m['scene']}_nt{m['n_t']}", {"slip_runs": 0, "stick_runs": 0, "max_abs_slip_err": 0.0,
                                                          "D04_min": None, "D04_max": None, "max_abs_hold_disp_rel_s": 0.0,
                                                          "max_abs_ramp_disp_rel_s": 0.0, "books_max_rel": 0.0,
                                                          "books_fail_runs": []})
        g["books_max_rel"] = _r(max(g["books_max_rel"], m["books_max_rel"]))
        if not m["books_ok"]:
            g["books_fail_runs"].append(m["run"])
        if m["kind"] == "slip":
            g["slip_runs"] += 1
            g["max_abs_slip_err"] = _r(max(g["max_abs_slip_err"], abs(m["err_over_gcos"])))
            d = m["dissipation_ratio"]
            g["D04_min"] = _r(d if g["D04_min"] is None else min(g["D04_min"], d))
            g["D04_max"] = _r(d if g["D04_max"] is None else max(g["D04_max"], d))
        else:
            g["stick_runs"] += 1
            g["max_abs_hold_disp_rel_s"] = _r(max(g["max_abs_hold_disp_rel_s"], abs(m["disp_rel_s"])))
            g["max_abs_ramp_disp_rel_s"] = _r(max(g["max_abs_ramp_disp_rel_s"], abs(m["ramp_disp_rel_s"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("formal", "v2", "cont09", "cont09-v2", "cost-v2", "cost-v3", "cpu-tests", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = ROOT/a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    formal = ROOT/a.formal
    meta = json.loads((formal/"meta.json").read_text())
    s = json.loads((formal/"summary.json").read_text())
    s2 = json.loads((ROOT/a.v2/"summary.json").read_text())
    c1 = s["RUBBER-CONT-01"]
    scored = c1["detail"]["scored"]
    verdicts = {k: {"verdict": v.get("verdict"), "criteria": v.get("criteria")} for k, v in s.items()
                if k.startswith("RUBBER")}
    c09 = json.loads((ROOT/a.cont09/"result.json").read_text())
    c09v2 = ROOT/a.cont09_v2/"result.json"
    cost = {}
    for law, d in (("v2", a.cost_v2), ("v3", a.cost_v3)):
        p = ROOT/d/"result.json"
        if p.exists():
            r = json.loads(p.read_text())
            cost[law] = {"rows": [{k: row.get(k) for k in ("variant", "particles", "nodes", "substeps",
                                                            "ms_per_substep", "env_steps_per_s",
                                                            "vram_peak_mb_cumulative", "stage_split", "label_gpu",
                                                            "gpu_label")} for row in r.get("rows", [])],
                         "gpu_vs_cpu": {k: v for k, v in r.get("gpu_vs_cpu", {}).items() if k != "values"},
                         "meta": {k: r["meta"].get(k) for k in ("commit", "device_name", "driver", "label_gpu",
                                                                 "shared_gpu", "co_tenant_pids")
                                  if k in r["meta"]}, "x3_wrapper": r.get("x3_wrapper")}
        else:
            cost[law] = "MISSING"
    ev = {
        "schema": "rcs_rl_ready_evidence_v1", "id": "p2_4_contact_x3_001",
        "phase": "P2-4 rubber contact, task X3 (law grid_node_penalty_coulomb_v3, envelope at the acceptance profile)",
        "adr": "docs/adr/0023-rubber-material-v1.md section 22.11",
        "report": "docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md section 11",
        "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
        "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                     "declared_at_commit": "e19f488", "law_commit": "b668bdc"},
        "formal_cpu_run": {"dir": a.formal, "meta": meta, "verdicts": verdicts, "errors": s.get("errors"),
                           "failing_rows": {"RUBBER-CONT-01": c1.get("failing_rows"),
                                            "RUBBER-CONT-04": s.get("RUBBER-CONT-04", {}).get("failing_rows")},
                           "groups": _group_table(scored), "onset": c1.get("onset"), "books_order": c1.get("books_order"),
                           "pinch": s.get("RUBBER-CONT-05", {}).get("detail"),
                           "pen": s.get("RUBBER-CONT-02", {}).get("detail"),
                           "imp": s.get("RUBBER-CONT-03", {}).get("detail"),
                           "reported_mu_1_rows": s.get("reported_mu_1_rows"),
                           "scored_rows": [{k: (_r(v) if isinstance(v, float) else v) for k, v in m.items()
                                            if k != "split"} for m in scored]},
        "v2_formal_for_comparison": {"dir": a.v2, "verdicts": {k: v.get("verdict") for k, v in s2.items()
                                                                if k.startswith("RUBBER")}},
        "cont09_gpu": {"dir": a.cont09, "meta": c09.get("meta"), "verdict": c09.get("cont09", {}).get("verdict"),
                       "cont09": {**{k: c09["cont09"].get(k) for k in ("N", "particles_per_env", "substeps", "k_per_side",
                                                                         "lam", "eps_rel", "episode_order",
                                                                         "contact_law_built", "a_zero_rows", "c_audits",
                                                                         "validity", "verdict")},
                                  "b_perturbation_band": {k: v for k, v in c09["cont09"]["b_perturbation_band"].items()
                                                          if k != "rows_worst_first"},
                                  "b_worst_rows": c09["cont09"]["b_perturbation_band"].get("rows_worst_first", [])[:3]},
                       "batched_cost_rows_v3": c09.get("cost_rows"),
                       "v2_reference": a.cont09_v2 if c09v2.exists() else "MISSING",
                       "batched_cost_rows_v2": json.loads(c09v2.read_text()).get("cost_rows") if c09v2.exists() else None},
        "single_env_cost_rows": cost,
        "cpu_tests": {"no_cuda": _tail(ROOT/a.cpu_tests/"no_cuda.log"),
                      "cuda_visible_0": _tail(ROOT/a.cpu_tests/"cuda0.log")},
        "frozen": ("v1 / v2 contact kernels byte-identical (v1 pins in tests/test_rubber_contact_v2.py, v2 pin "
                   "a186c586994e3cdb in tests/test_rubber_contact_v3.py); rubber_contact_tests_v1 / _v2 unchanged; no Phase "
                   "1 module, configs/rl_benchmark/*, scripts/rl_ready/* or frozen rubber module / protocol changed")}
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(out)


if __name__ == "__main__":
    main()
