"""Compact evidence ``docs/evidence/rl_ready/p2_4_contact_001.json`` of P2-4 (ADR 0023 §22) from the raw outputs in
``artifacts/rl_ready/p2_4`` (reads only; never re-runs anything; refuses to overwrite).

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/write_contact_evidence.py \
        --formal artifacts/rl_ready/p2_4/contact_v1_001 --gpu artifacts/rl_ready/p2_4/gpu_contact_002 \
        --probe artifacts/rl_ready/p2_4/probe_contact_laws_004 --cpu-tests artifacts/rl_ready/p2_4/cpu_tests_001 \
        --out docs/evidence/rl_ready/p2_4_contact_001.json
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _tail(path, n=3):
    p = Path(path)
    return p.read_text().strip().splitlines()[-n:] if p.exists() else ["MISSING"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--cpu-tests", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT/a.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    formal = ROOT/a.formal
    meta = json.loads((formal/"meta.json").read_text())
    summ = json.loads((formal/"summary.json").read_text())
    gpu = json.loads((ROOT/a.gpu/"result.json").read_text())
    probe_meta = json.loads((ROOT/a.probe/"meta.json").read_text())
    probe = json.loads((ROOT/a.probe/"results.json").read_text())
    proto = ROOT/"configs"/"rubber"/"rubber_contact_tests_v1.json"
    inc_probe = {}
    for n, v in probe.items():
        if n.startswith("incline"):
            gc = 9.81*math.cos(math.radians(v["theta_deg"]))
            inc_probe.setdefault(v["law"], {})[str(v["tan_theta_over_mu"])] = {
                "hold_displacement_m": v["displacement_after_ramp_m"],
                "slip_error_over_g_cos": (v["a_fit_m_s2"]-v["a_ref_m_s2"])/gc}
    imp_probe = {n: {"E_end_over_E0": v["E_end_over_E0"], "vcm_end_over_V": v["vcm_end_over_V"],
                     "contact_first_last_s": v["contact_first_last_s"], "max_books_over_E0": v.get("max_books_over_E0")}
                 for n, v in probe.items() if n.startswith("impact")}

    def verdicts(tid):
        v = summ.get(tid, {})
        return {"verdict": v.get("verdict"), "criteria": v.get("criteria"), "reason": v.get("reason")}

    ev = {
        "schema": "rcs_rl_ready_evidence_v1", "id": "p2_4_contact_001", "phase": "P2-4 rubber contact / friction / coupling",
        "adr": "docs/adr/0023-rubber-material-v1.md section 22",
        "report": "docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md",
        "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
        "protocol": {"path": "configs/rubber/rubber_contact_tests_v1.json",
                     "sha256": hashlib.sha256(proto.read_bytes()).hexdigest(), "declared_at_commit": "51023d6",
                     "law_commit": "259f2b6"},
        "decisions": {
            "contact_law": "grid_node_penalty_coulomb_v1 (node penalty + Coulomb spring-dashpot-slider, kappa_n = kappa_t = "
                           "0.25, zeta_n = zeta_t = 0.5), kick-paired contact work; frozen per-particle helper and the "
                           "node velocity projection rejected (creep, dt-dependent loss, chatter)",
            "pads": "moving kinematic box frames (kind pad), per-pad wrench [F, B, 6] single env / [N, G, B, 6] batched; "
                    "Phase 1 delayed handoff reused through the MaterialCoupling surface without changing Phase 1 files",
            "self_contact": "NOT_IMPLEMENTED_V1 (declared limitation; per-control-step detector required before P2-6)",
            "coupling_delay": "kinematic grips have no feedback; with feedback the 20 ms handoff is coupling_delay_significant "
                              "in the whole declared envelope (RUBBER-CONT-07)",
            "gpu_isolation": "env-owned rows by construction; batched tests -08 / -09 NOT_RUN (batched contact path = task B1)"},
        "decision_probe": {"run": a.probe, "commit": probe_meta["commit"], "dirty": probe_meta["dirty"],
                           "label": probe_meta["label"], "incline": inc_probe, "impact": imp_probe,
                           "note": "grid_penalty_undamped at this commit = TANGENTIAL dashpot off only (normal 0.5 kept)"},
        "formal_cpu_run": {"run": a.formal, "commit": meta["commit"], "dirty": meta["dirty"], "label": meta["label"],
                           "start": meta["start"], "end": meta.get("end"), "wall_s": meta.get("wall_s"),
                           "warp": meta["warp"], "python": meta["python"], "protocol_sha256": meta["protocol_sha256"],
                           "results": {tid: verdicts(tid) for tid in sorted(k for k in summ if k.startswith("RUBBER"))},
                           "errors": summ.get("errors")},
        "formal_numbers": {
            "RUBBER-CONT-01": summ["RUBBER-CONT-01"]["detail"], "RUBBER-CONT-02": summ["RUBBER-CONT-02"]["detail"],
            "RUBBER-CONT-03": summ["RUBBER-CONT-03"]["detail"], "RUBBER-CONT-04": summ["RUBBER-CONT-04"]["detail"],
            "RUBBER-CONT-05": summ["RUBBER-CONT-05"]["detail"],
            "RUBBER-CONT-07": {"points": summ["RUBBER-CONT-07"]["points"], "books": summ["RUBBER-CONT-07"]["books"]}},
        "gpu": {"run": a.gpu, "label": gpu["meta"]["label_gpu"], "shared_gpu": gpu["meta"]["shared_gpu"],
                "driver": gpu["meta"]["driver"], "commit": gpu["meta"]["commit"], "dirty_tree": gpu["meta"]["dirty_tree"],
                "protocol_sha256": gpu["meta"]["protocol_sha256"], "co_tenant_before": gpu["meta"]["co_tenant_before"],
                "co_tenant_after": gpu["meta"]["co_tenant_after"], "device": gpu["meta"]["device_name"],
                "rows": [{k: r[k] for k in ("variant", "particles", "nodes", "substeps", "dt_s", "ms_per_substep",
                                            "env_steps_per_s", "vram_peak_mb_cumulative", "l2", "zero_rows_exact")}
                         for r in gpu["rows"]],
                "stage_split_contact_ms_per_substep": next(r for r in gpu["rows"]
                                                           if r["variant"] == "contact")["stage_split_ms_per_substep"],
                "gpu_vs_cpu": {k: v for k, v in gpu["gpu_vs_cpu"].items() if k != "values"},
                "RUBBER-CONT-11": "reported (cost)", "RUBBER-CONT-09": "NOT_RUN (needs the batched contact path)",
                "failed_partial_run_preserved": "artifacts/rl_ready/p2_4/gpu_contact_001 (TypeError after two rows)"},
        "cpu_suites": {"dir": a.cpu_tests, "cuda_visible_devices_empty": _tail(ROOT/a.cpu_tests/"cuda_none.log"),
                       "cuda_visible_devices_0": _tail(ROOT/a.cpu_tests/"cuda_0.log")},
        "not_changed": ["tlmpm_v5.py", "tlmpm_v6.py", "tlmpm_rubber_v1.py", "rubber_constitutive.py",
                        "tlmpm_rubber_visco_v1.py", "rubber_visco_constitutive.py", "rubber_constraint_work.py",
                        "batched/*", "configs/rl_benchmark/*", "scripts/rl_ready/*", "rubber protocols v1-v4, "
                        "moving-grip v1, visc v1-v2"],
    }
    out.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(out)


if __name__ == "__main__":
    main()
