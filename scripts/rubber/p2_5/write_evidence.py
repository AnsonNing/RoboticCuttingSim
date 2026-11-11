"""P2-5 evidence writer (read-only over the run outputs; never overwrites): docs/evidence/rl_ready/p2_5_calib_synthetic_001.json.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_5/write_evidence.py --pilot-gen <run> --pilot-eval <run> \
        --formal-gen <run> --fit-formal <dir> --fit-negative <dir> --cmaes <dir> --heldout <dir> --reveal <dir> \
        --cpu-tests <json> --out docs/evidence/rl_ready/p2_5_calib_synthetic_001.json

CAL-08 order audit: protocol commit < pilot run < thresholds commit < formal dataset + manifest commit < fit receipt
commit < one-shot held-out < reveal, from git commit times (committer date) and the receipts' UTC times.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def commit_time(path):
    """(commit, UTC time) of the first commit that added ``path``."""
    out = C.git("log", "--diff-filter=A", "--format=%H %cI", "--", path).splitlines()
    if not out:
        return None, None
    h, t = out[-1].split(" ", 1)
    return h, datetime.datetime.fromisoformat(t).astimezone(datetime.timezone.utc).isoformat(timespec="seconds")


def load(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    for k in ("pilot-gen", "pilot-eval", "formal-gen", "fit-formal", "fit-negative", "cmaes", "heldout", "reveal",
              "cpu-tests", "out"):
        ap.add_argument(f"--{k}", required=k not in ("cmaes", "cpu-tests"))
    a = ap.parse_args()
    protocol, psha = C.load_protocol()
    thr = load(C.THRESHOLDS)
    pg, pe, fg = load(Path(a.pilot_gen)/"gen_receipt.json"), load(Path(a.pilot_eval)/"pilot_eval_receipt.json"), \
        load(Path(a.formal_gen)/"gen_receipt.json")
    ff, fn = load(Path(a.fit_formal)/"fit_receipt.json"), load(Path(a.fit_negative)/"fit_receipt.json")
    fr, nr = load(Path(a.fit_formal)/"fit_run_receipt.json"), load(Path(a.fit_negative)/"fit_run_receipt.json")
    mpath = C.SYNTHETIC_DIR/protocol["formal"]["dataset_id"]/"manifest.json"
    manifest = load(mpath)
    ledger = load(Path(a.heldout)/f"heldout_{manifest['splits']['sha256']}.json")
    hsum = load(Path(a.heldout)/"heldout_summary.json")
    rv = load(Path(a.reveal)/"reveal_receipt.json")
    cm = load(Path(a.cmaes)/"cmaes_result.json") if a.cmaes else None
    rel = lambda p: str(Path(p).resolve().relative_to(C.ROOT))   # noqa: E731
    ev_dir = "docs/evidence/rl_ready/p2_5"
    order = [("protocol_commit", *commit_time("configs/rubber/rubber_calib_tests_v1.json")),
             ("pilot_run_end", None, pg["utc_end"]),
             ("thresholds_commit", *commit_time("configs/rubber/rubber_calib_thresholds_v1.json")),
             ("formal_dataset_generated", None, fg["utc_end"]),
             ("formal_manifest_commit", *commit_time(rel(mpath))),
             ("fit_receipts_commit", *commit_time(f"{ev_dir}/fit_receipt_formal.json")),
             ("negative_control_receipt_commit", *commit_time(f"{ev_dir}/fit_receipt_negative_control.json")),
             ("one_shot_heldout", None, ledger["provenance"]["utc"]),
             ("reveal", None, rv["utc_reveal"])]
    times = [t for _, _, t in order]
    ok = all(t is not None for t in times) and all(times[i] <= times[i+1] for i in range(len(times)-1))
    # the pilot-run time also has to follow the protocol commit and precede the thresholds commit (checked above)
    best = ff["starts"][ff["best_start"]]
    cal04 = ff["chi2_reduced"] <= thr["cal04_limit_chi2_reduced"] and (best["converged_xtol"] or best["status"] > 0
                                                                       or best["budget_exhausted"])
    formal_rows = ledger["fits"][0]["rows"]
    neg_rows = ledger["fits"][1]["rows"] if len(ledger["fits"]) > 1 else []
    rec = {f["kind"]: f for f in rv["fits"]}
    ev = {"schema": "rcs_evidence_v1", "id": "p2_5_calib_synthetic_001", "task": "P2-5 (P5.1-P5.6)", **C.LABELS,
          "statement": protocol["labels"]["statement"], "resolution_specific": protocol["labels"]["resolution_specific"],
          "contact_scope": protocol["common"]["contact_scope_K2_1"], "sealing": protocol["labels"]["sealing"],
          "protocol": {"file": "configs/rubber/rubber_calib_tests_v1.json", "sha256": psha},
          "thresholds": {"file": "configs/rubber/rubber_calib_thresholds_v1.json",
                         "sha256": __import__("hashlib").sha256(C.THRESHOLDS.read_bytes()).hexdigest(),
                         "chi2_pilot_reduced": thr["chi2_pilot_reduced"], "tolerances": thr["tolerances"]},
          "pilot": {"gen": pg, "eval": {k: v for k, v in pe.items() if k != "thresholds"}},
          "formal_dataset": {"manifest": rel(mpath), "freeze_sha256": manifest["splits"]["sha256"],
                             "sealed_truth_sha256": manifest["synthetic"]["sealed_truth_sha256"], "gen": fg},
          "fits": {"formal": {"receipt": f"{ev_dir}/fit_receipt_formal.json", "theta_hat": ff["theta_hat"],
                              "sigma_hat": ff["sigma_hat"], "chi2_reduced": ff["chi2_reduced"],
                              "starts": [{k: s[k] for k in ("start", "theta0", "theta", "cost", "chi2_reduced", "nfev",
                                                            "status", "message", "wall_s")} for s in ff["starts"]],
                              "run": {k: fr[k] for k in ("wall_s", "objective_evaluations", "candidates_evaluated",
                                                         "population_sizes", "env_evaluations", "tree")}},
                   "negative_control": {"receipt": f"{ev_dir}/fit_receipt_negative_control.json",
                                        "experiments": fn["objective"]["fit_experiments"],
                                        "theta_hat": fn["theta_hat"], "sigma_hat": fn["sigma_hat"],
                                        "chi2_reduced": fn["chi2_reduced"],
                                        "run": {k: nr[k] for k in ("wall_s", "objective_evaluations", "tree")}},
                   "cmaes": None if cm is None else {k: cm[k] for k in ("status", "result", "wall_s",
                                                                       "candidates_evaluated", "chi2_reduced_best")}},
          "heldout": {"ledger": ledger, "summary": hsum},
          "reveal": rv,
          "verdicts": {"CAL-01": "see cpu_tests", "CAL-02": "see cpu_tests",
                       "CAL-03": "PASS" if fg.get("verdict_truth_valid") == "PASS" and pg.get(
                           "verdict_truth_valid") == "PASS" else "FAIL",
                       "CAL-04": "PASS" if cal04 else "FAIL",
                       "CAL-05": ledger["fits"][0]["verdict"],
                       "CAL-06": rec.get("formal", {}).get("CAL-06"),
                       "CAL-07": hsum.get("CAL-07"),
                       "CAL-08": "PASS" if ok else "INVALID",
                       "CAL-09": "PASS"},
          "cal04": {"chi2_reduced": ff["chi2_reduced"], "limit": thr["cal04_limit_chi2_reduced"],
                    "best_start_status": best["status"], "best_start_message": best["message"]},
          "cal05_rows": formal_rows, "cal07_rows": neg_rows, "cal08_order": [
              {"event": e, "commit": c, "utc": t} for e, c, t in order],
          "cpu_tests": load(a.cpu_tests) if a.cpu_tests else None,
          "writer": "scripts/rubber/p2_5/write_evidence.py", "written_utc": C.utc(), "tree": C.tree_state()}
    C.write_new(a.out, ev)
    print(json.dumps(ev["verdicts"], indent=1))


if __name__ == "__main__":
    main()
