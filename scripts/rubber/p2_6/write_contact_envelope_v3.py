"""Write ``configs/rubber/contact_envelope_v3.json`` and the evidence of the formal run of ``rubber_contact_envelope_v2``
(law v2 at n_t 4 / 5 / 6 after K2-1). Read-only on the run; refuses to overwrite either output.

contact_envelope_v3 = contact_envelope_v2 with (i) the v2 rows at n_t 4 / 5 / 6 from the run's decision rule (scored
values only; NOT_ESTABLISHED if nothing passes) and (ii) every v3 row narrowed in scope to translational sliding only
(K2-1; label contact_slip_rate_equalised): ``scope`` = ``translational_sliding_only``. Other rows unchanged.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_6/write_contact_envelope_v3.py \
        --run artifacts/rl_ready/p2_6/contact_envelope_v2_001
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_envelope_v2.json"
PROTOCOL_SHA = "1f6163997bdd072e961b2724de951d81f2db29f889d286ce1d1e3c4ba63f1e18"
ENV2 = ROOT/"configs"/"rubber"/"contact_envelope_v2.json"
ENV2_SHA = "46f74efa880cf7121e50070a0cf5fb2358e87d7e390f67cd7b608802f9f03b78"
OUT_ENV = ROOT/"configs"/"rubber"/"contact_envelope_v3.json"
OUT_EVID = ROOT/"docs"/"evidence"/"rl_ready"/"p2_6_contact_envelope_002.json"
V2, V3 = "grid_node_penalty_coulomb_v2", "grid_node_penalty_coulomb_v3"
K2_1 = ("K2-1 (docs/reports/rl_ready/P2_K2_COMBINED_REVIEW.md, claude/rl-rubber @60e9119): the patch slip-rate "
        "equalisation exerts an unbounded, mu-independent torque and removes rotation and in-plane bending in slip; the "
        "limits hold for translational sliding only (rows with rotation about the normal, in-plane bending in slip or "
        "several contact regions on one face are outside: label contact_slip_rate_equalised)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--drag-rerun", default=None, help="labelled partial rerun of the reported DRAG row")
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else ROOT/a.run
    for path, sha in ((PROTOCOL, PROTOCOL_SHA), (ENV2, ENV2_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{path} does not match its pin")
    for out in (OUT_ENV, OUT_EVID):
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
    meta = json.loads((run/"meta.json").read_text())
    summary = json.loads((run/"summary.json").read_text())
    if meta["protocol_sha256"] != PROTOCOL_SHA or meta["label"] != "FORMAL rubber_contact_envelope_v2" or meta["dirty"]:
        raise SystemExit("not a clean formal run of rubber_contact_envelope_v2")
    env2 = json.loads(ENV2.read_text())
    rows = []
    for r in env2["rows"]:
        r = dict(r)
        if r["law"] == V3:
            r["scope"] = "translational_sliding_only"
            r["scope_reason"] = K2_1
        elif r["law"] == V2 and str(r["n_t"]) in summary["envelope"]:
            res = summary["envelope"][str(r["n_t"])]
            levels = {k: v for k, v in summary["levels"].items() if v["n_t"] == r["n_t"]}
            spins = {k: v for k, v in summary["ENV-04"]["rows"].items() if v.get("n_t") == r["n_t"]}
            r = {"law": V2, "n_t": r["n_t"], "status": res["status"],
                 "sliding_mu_max": res["sliding_mu_max"] if res["status"] == "ESTABLISHED" else None,
                 "static_mu_max": res["static_mu_max"] if res["status"] == "ESTABLISHED" else None,
                 "scored": {"sliding_mu": sorted({v["mu"] for v in levels.values() if v["sliding_pass_with_spin"]}),
                            "static_mu": sorted({v["mu"] for v in levels.values() if v["static_pass"]}),
                            "spin_down": {k: {q: v.get(q) for q in ("pass", "decel_ratio_5ms", "decel_ratio_10ms",
                                                                     "min_omega")} for k, v in spins.items()},
                            "scenes": f"INC2 (24 x 12 x 6 mm), STRIP (48 x 16 x t mm), SPIN (16 x 16 x 4 mm), n_t {r['n_t']}",
                            "tests": "ENV-01 b, d04, e; ENV-02 a, f, e; ENV-04 a, b, c (K2-1 spin-down)"},
                 "run_id": run.name, "run_commit": meta["commit"][:7],
                 "protocol": "configs/rubber/rubber_contact_envelope_v2.json", "protocol_sha256": PROTOCOL_SHA,
                 "evidence": "docs/evidence/rl_ready/p2_6_contact_envelope_002.json",
                 "known_limitations": ["friction-normal flutter above the row (ADR 0023 §22.9 / §22.10)",
                                       "RUBBER-CONT-01 d stick-run books residual (§22.10), reported only",
                                       "lateral drag (K2-1 b) reported, no independent reference"],
                 "diagnostic_narrowing": "none"}
            if r["status"] != "ESTABLISHED":
                r["reason"] = f"{run.name}: failing levels " + str(sorted(
                    k for k, v in levels.items() if not (v["sliding_pass_with_spin"] and v["static_pass"])))
        rows.append(r)
    env3 = {"schema": env2["schema"], "id": "contact_envelope_v3",
            "task": "ENV2 for the P2-6 fallback law after K2-1: contact_envelope_v2 + the formal check "
                    "rubber_contact_envelope_v2 (law v2 at n_t 4 / 5 / 6, spin-down scored) + the K2-1 scope of the "
                    "v3 rows",
            "declared": f"2026-11-11, written by scripts/rubber/p2_6/write_contact_envelope_v3.py from {run.name} "
                        f"(commit {meta['commit'][:7]}, clean tree); SHA-pinned in tests/test_rubber_contact_envelope.py; "
                        "contact_envelope_v1 / v2 unchanged",
            "supersedes": {"contact_envelope_v2": ENV2_SHA},
            "calibration_status": env2["calibration_status"],
            "rules": dict(env2["rules"], scope=("a row with scope translational_sliding_only supports only scenes "
                                                "whose contact slip is a pure translation; rl.rubber_family refuses it "
                                                "for P2-6 (rotation, in-plane bending in slip)")),
            "rows": rows}
    OUT_ENV.write_text(json.dumps(env3, indent=1)+"\n")
    ev = {"schema": "rcs_rl_ready_evidence_v1", "id": "p2_6_contact_envelope_002",
          "phase": "P2-6 prerequisite ENV2 (law v2 after K2-1): formal envelope check at n_t 4-6 with the K2-1 spin-down",
          "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
          "protocol": {"path": "configs/rubber/rubber_contact_envelope_v2.json", "sha256": PROTOCOL_SHA},
          "run": {"dir": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run), "meta": meta,
                  "errors": summary["errors"]},
          "verdict": summary["verdict"], "ENV-01": summary["ENV-01"], "ENV-02": summary["ENV-02"],
          "ENV-04": summary["ENV-04"], "ENV-05": summary["ENV-05"], "envelope": summary["envelope"],
          "ENV-05_note": ("the DRAG row of the formal run recorded zero displacements: its reference markers were a "
                          "view of the live CPU Warp array (recording bug, runner fixed at 7a36690); the reported "
                          "row is the labelled partial rerun below (DRAG is reported, never scored)"),
          "ENV-05_drag_rerun": (None if a.drag_rerun is None else {
              "dir": a.drag_rerun, "meta": json.loads((ROOT/a.drag_rerun/"meta.json").read_text()),
              "rows": json.loads((ROOT/a.drag_rerun/"partial.json").read_text())["DRAG_nt4_mu0.4"]["rows"]}),
          "p2_6_friction_range": summary["p2_6_friction_range"], "levels": summary["levels"],
          "output": {"path": "configs/rubber/contact_envelope_v3.json",
                     "sha256": hashlib.sha256(OUT_ENV.read_bytes()).hexdigest()},
          "runs": [{k: m.get(k) for k in ("run", "scene", "n_t", "mu", "ratio", "material", "kind", "err_over_gcos",
                                           "dissipation_ratio", "disp_rel_s", "ramp_disp_rel_s", "a_over_gcos",
                                           "books_max_rel", "pen_max_rel_s", "particles", "dt_s", "substeps",
                                           "wall_s", "valid")} for m in summary["runs"]]}
    OUT_EVID.write_text(json.dumps(ev, indent=1, default=float)+"\n")
    print(json.dumps({"verdict": summary["verdict"], "envelope": summary["envelope"],
                      "range": summary["p2_6_friction_range"], "envelope_v3_sha256": ev["output"]["sha256"]}))


if __name__ == "__main__":
    main()
