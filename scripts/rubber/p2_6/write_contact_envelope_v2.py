"""Write ``configs/rubber/contact_envelope_v2.json`` and the ENV2 evidence from the formal run of
``rubber_contact_envelope_v1`` (task ENV2). Read-only on the run; refuses to overwrite either output.

contact_envelope_v2 = contact_envelope_v1 with the v3 rows at n_t 4 / 5 / 6 set by the run's decision rule (scored
values only; the n_t 4 row is min(v1 row, STRIP result)), every other row copied unchanged.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_6/write_contact_envelope_v2.py \
        --run artifacts/rl_ready/p2_6/contact_envelope_v1_001
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_envelope_v1.json"
PROTOCOL_SHA = "d7ecbd3164409cce12d229504864297677baf3307f9763bb7797329506ffe92c"
ENV1 = ROOT/"configs"/"rubber"/"contact_envelope_v1.json"
ENV1_SHA = "08f68f704a8052213be4448f964f1b0fc31da867470d7cc0274f93c71ba1a77a"
OUT_ENV = ROOT/"configs"/"rubber"/"contact_envelope_v2.json"
OUT_EVID = ROOT/"docs"/"evidence"/"rl_ready"/"p2_6_contact_envelope_001.json"
LAW = "grid_node_penalty_coulomb_v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else ROOT/a.run
    for path, sha in ((PROTOCOL, PROTOCOL_SHA), (ENV1, ENV1_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{path} does not match its pin")
    for out in (OUT_ENV, OUT_EVID):
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
    meta = json.loads((run/"meta.json").read_text())
    summary = json.loads((run/"summary.json").read_text())
    if meta["protocol_sha256"] != PROTOCOL_SHA or meta["label"] != "FORMAL rubber_contact_envelope_v1":
        raise SystemExit("not a formal run of rubber_contact_envelope_v1")
    if meta["dirty"]:
        raise SystemExit("the formal run was on a dirty tree")
    env1 = json.loads(ENV1.read_text())
    run_id = run.name
    rows = []
    for r in env1["rows"]:
        r = dict(r)
        if r["law"] == LAW and str(r["n_t"]) in summary["envelope"]:
            res = summary["envelope"][str(r["n_t"])]
            levels = {k: v for k, v in summary["levels"].items() if v["n_t"] == r["n_t"]}
            sliding = sorted({v["mu"] for v in levels.values() if v["sliding_pass"]})
            static = sorted({v["mu"] for v in levels.values() if v["static_pass"]})
            failing = sorted(k for k, v in levels.items() if not (v["sliding_pass"] and v["static_pass"]))
            if r["status"] == "ESTABLISHED":                   # n_t 4: v1 row, narrowed (never widened) by STRIP
                r["sliding_mu_max"] = min(r["sliding_mu_max"], res["sliding_mu_max"] or 0.0) or None
                r["static_mu_max"] = min(r["static_mu_max"], res["static_mu_max"] or 0.0) or None
                if r["sliding_mu_max"] is None or r["static_mu_max"] is None:
                    r.update(status="NOT_ESTABLISHED", sliding_mu_max=None, static_mu_max=None)
                r["env2_check"] = {"run_id": run_id, "protocol": "configs/rubber/rubber_contact_envelope_v1.json",
                                   "protocol_sha256": PROTOCOL_SHA, "scenes": res["scenes"],
                                   "sliding_mu_max": res["sliding_mu_max"], "static_mu_max": res["static_mu_max"],
                                   "failing_levels": failing}
            else:
                r = {"law": LAW, "n_t": r["n_t"], "status": res["status"],
                     "law_status": "EXPERIMENTAL_NOT_ACCEPTED until the K2 review (ADR 0023 §22.11)",
                     "sliding_mu_max": res["sliding_mu_max"], "static_mu_max": res["static_mu_max"],
                     "scored": {"sliding_mu": sliding, "static_mu": static,
                                "sliding_fail_mu": sorted({v["mu"] for v in levels.values() if not v["sliding_pass"]}),
                                "scenes": "INC2 (24 x 12 x 6 mm) and the unseen STRIP (48 x 16 x t mm, dx 1 mm), "
                                          f"n_t {r['n_t']}; P2 hold at 0.95 x 0.8",
                                "tests": "ENV-01 b (slip), d04 (dissipation), e (validity); ENV-02 a (stick), "
                                         "f (load-ramp slip), e"},
                     "run_id": run_id, "run_commit": meta["commit"][:7],
                     "protocol": "configs/rubber/rubber_contact_envelope_v1.json", "protocol_sha256": PROTOCOL_SHA,
                     "evidence": "docs/evidence/rl_ready/p2_6_contact_envelope_001.json",
                     "known_limitations": ["RUBBER-CONT-01 d stick-run books residual (§22.10) reported, not scored",
                                           "mu >= 1.0 refused (contact_coulomb_ill_posed)",
                                           "v3 declared limitations (i)-(v) of ADR 0023 §22.11"],
                     "diagnostic_narrowing": "none"}
                if r["status"] != "ESTABLISHED":
                    r["sliding_mu_max"] = r["static_mu_max"] = None
                    r["reason"] = f"ENV2 {run_id}: failing levels {failing}"
        rows.append(r)
    env2 = {"schema": env1["schema"], "id": "contact_envelope_v2",
            "task": "ENV2 (ADR 0023 §23.2 decision 4; plan §2.4): contact_envelope_v1 + the formal envelope check "
                    "rubber_contact_envelope_v1 (law v3 at n_t 5 / 6, STRIP at n_t 4-6)",
            "declared": f"2026-11-11, written by scripts/rubber/p2_6/write_contact_envelope_v2.py from {run_id} "
                        f"(commit {meta['commit'][:7]}, clean tree); SHA-pinned in tests/test_rubber_contact_envelope.py; "
                        "contact_envelope_v1 unchanged",
            "supersedes": {"contact_envelope_v1": ENV1_SHA},
            "calibration_status": env1["calibration_status"], "rules": env1["rules"], "rows": rows}
    OUT_ENV.write_text(json.dumps(env2, indent=1)+"\n")
    jobs = sorted((run/"jobs").glob("*.json"))
    evidence = {
        "schema": "rcs_rl_ready_evidence_v1", "id": "p2_6_contact_envelope_001",
        "phase": "P2-6 prerequisite ENV2: formal envelope check of law grid_node_penalty_coulomb_v3 at n_t 4-6",
        "adr": "docs/adr/0023-rubber-material-v1.md §23.2 decision 4",
        "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
        "protocol": {"path": "configs/rubber/rubber_contact_envelope_v1.json", "sha256": PROTOCOL_SHA},
        "run": {"dir": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run), "meta": meta,
                "jobs": len(jobs), "errors": summary["errors"]},
        "verdict": summary["verdict"], "ENV-01": summary["ENV-01"], "ENV-02": summary["ENV-02"],
        "envelope": summary["envelope"], "p2_6_friction_range": summary["p2_6_friction_range"],
        "levels": summary["levels"],
        "output": {"path": "configs/rubber/contact_envelope_v2.json",
                   "sha256": hashlib.sha256(OUT_ENV.read_bytes()).hexdigest()},
        "runs": [{k: m.get(k) for k in ("run", "scene", "n_t", "mu", "ratio", "material", "kind", "err_over_gcos",
                                         "dissipation_ratio", "disp_rel_s", "ramp_disp_rel_s", "a_over_gcos",
                                         "books_max_rel", "pen_max_rel_s", "particles", "dt_s", "substeps", "wall_s",
                                         "valid")} for m in summary["runs"]]}
    OUT_EVID.write_text(json.dumps(evidence, indent=1, default=float)+"\n")
    print(json.dumps({"verdict": summary["verdict"], "envelope": summary["envelope"],
                      "range": summary["p2_6_friction_range"], "envelope_v2_sha256": evidence["output"]["sha256"]}))


if __name__ == "__main__":
    main()
