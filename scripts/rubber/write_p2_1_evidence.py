"""Compose docs/evidence/rl_ready/p2_1_rubber_001.json from the raw P2-1 artifacts (compact; no large series).

    PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/write_p2_1_evidence.py --tests-json <pytest counts json>
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT/"artifacts"/"rl_ready"/"p2_1"


def load(path):
    p = ART/path
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-json", help="JSON with pytest commands and counts")
    ap.add_argument("--out", default=str(ROOT/"docs"/"evidence"/"rl_ready"/"p2_1_rubber_001.json"))
    args = ap.parse_args()
    from robotic_cutting_sim import rubber_phys as P
    cfg = P.load_config()
    summary = load("rubber_phys_001/summary.json")
    meta = load("rubber_phys_001/run_meta.json")
    batched = load("batched_04_001/batched_04.json")
    if summary is not None and batched is not None:
        rows = [json.loads(p.read_text()) for p in sorted((ART/"rubber_phys_001"/"raw"/"04").glob("*.json"))]
        verdicts, measured = P.evaluate_04([r for r in rows if r["status"] == "OK"], cfg, batched=batched)
        values = set(verdicts.values())
        summary["RUBBER-PHYS-04"].update({"criteria": verdicts, "measured": measured,
                                          "verdict": "PASS" if values == {"PASS"} else "FAIL" if "FAIL" in values
                                          else "PASS_EXCEPT_NOT_RUN",
                                          "batched_source": "artifacts/rl_ready/p2_1/batched_04_001/batched_04.json"})
    diag07 = {}
    for f in (1, 2):
        d = load(f"diag_07_cfl_factor{f}/diag_07.json")
        if d:
            post = [r for r in d["rows"] if r["phase"] == "post"]
            diag07[f"cfl_{d['cfl']:.3f}"] = {"label": d["label"], "substeps": d["substeps"], "dt_s": d["dt_s"],
                                              "first_flag": d["first_flag"],
                                              "post_J_min": min((r["J_min"] for r in post), default=None),
                                              "post_Jbar_range": [min((r["Jbar_min"] for r in post), default=None),
                                                                  max((r["Jbar_max"] for r in post), default=None)],
                                              "post_energy_drift_rel": (max(abs(r["energy_total_j"]-post[0]["energy_total_j"])
                                                                            for r in post)/abs(post[0]["energy_total_j"])
                                                                        if post and all(r["energy_total_j"] == r["energy_total_j"] for r in post) else None),
                                              "post_mean_length_rel": (sum(r["length_rel"] for r in post)/len(post)
                                                                       if post else None),
                                              "wall_s": d["wall_s"]}
    gpu_meta = load("gpu_smoke_001/meta.json")
    gpu = load("gpu_smoke_001/results.json")
    tests_info = None if not args.tests_json else json.loads(Path(args.tests_json).read_text())
    l01 = (tests_info or {}).get("rubber_batched_l0_l1")
    gpu_rows = gpu or []
    l2 = [r for r in gpu_rows if r.get("part") == "B_batched_L2_R1.7"]
    l3 = [r for r in gpu_rows if r.get("part") == "C_L3_band"]
    crit10 = {"L0": l01 or "NOT_RUN", "L1": l01 or "NOT_RUN",
              "L2": "NOT_RUN" if not l2 else ("PASS" if all(r["pass"] for r in l2) else "FAIL"),
              "L3": "NOT_RUN" if not l3 else ("PASS" if all(r["pass"] for r in l3) else "FAIL")}
    if batched is not None and not batched.get("bitwise_equal"):
        crit10["L0"] = "FAIL"
    values = set(crit10.values())
    if summary is not None:
        summary["RUBBER-PHYS-10"] = {"verdict": "FAIL" if "FAIL" in values else "PASS" if values == {"PASS"}
                                     else "PASS_EXCEPT_NOT_RUN", "criteria": crit10,
                                     "sources": ["tests/test_rubber_batched.py (CPU L0/L1)",
                                                 "artifacts/rl_ready/p2_1/batched_04_001 (CPU L0, RUBBER-PHYS-04 batch)",
                                                 "artifacts/rl_ready/p2_1/gpu_smoke_001 (GPU L2/L3)"]}
        summary["RUBBER-PHYS-11"] = {"verdict": "NOT_RUN", "reason": "P2-2 work package R2.2 (refinement / GCI)"}
    evidence = {
        "schema": "rcs_p2_1_rubber_evidence_v1",
        "date": "2026-11-10",
        "adr": "docs/adr/0023-rubber-material-v1.md",
        "work_packages": ["P2-0 (R0.1-R0.4)", "P2-1 (R1.1-R1.7)"],
        "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED (no real rubber data; numerical verification only)",
        "branch": "claude/rl-rubber",
        "protocol": {"file": "configs/rubber/rubber_phys_tests_v2.json",
                     "sha256": hashlib.sha256((ROOT/"configs"/"rubber"/"rubber_phys_tests_v2.json").read_bytes()).hexdigest(),
                     "v1": "configs/rubber/rubber_phys_tests_v1.json (frozen, superseded only in the initializer stop rule; "
                           "reason and evidence: artifacts/rl_ready/p2_1/protocol_diagnostics_001)"},
        "formal_run": {"dir": "artifacts/rl_ready/p2_1/rubber_phys_001", "meta": meta},
        "rubber_phys": summary,
        "diagnostics_not_formal": {"RUBBER-PHYS-07_release": diag07,
                                   "initializer_stop_v1": "artifacts/rl_ready/p2_1/protocol_diagnostics_001/README.txt"},
        "batched_04": None if batched is None else {k: batched[k] for k in batched if k != "per_env_vibration_metrics"}
                      | {"per_env_vibration_metrics": batched.get("per_env_vibration_metrics")},
        "gpu_smoke": {"status": "NOT_RUN (no GPU slot obtained)"} if gpu is None else {
            "dir": "artifacts/rl_ready/p2_1/gpu_smoke_001", "meta": {k: gpu_meta[k] for k in gpu_meta
                                                                  if k in ("command", "cuda_visible_devices", "device_name",
                                                                           "gpu_sharing", "co_tenant_before",
                                                                           "co_tenant_after", "commit", "warp",
                                                                           "started", "finished")},
            "results": gpu},
        "tests": tests_info,
    }
    Path(args.out).write_text(json.dumps(evidence, indent=1, default=float)+"\n")
    print(json.dumps({k: (v or {}).get("verdict") for k, v in (summary or {}).items()}, indent=1))


if __name__ == "__main__":
    main()
