"""P2-5 reveal and parameter recovery (task P5.6; CAL-06). Runs only after the one-shot held-out ledger exists.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_5/reveal.py --gen-run artifacts/rl_ready/p2_5/<formal gen run> \
        --ledger artifacts/rl_ready/p2_5/<run>/heldout --fit-receipt docs/evidence/rl_ready/p2_5/fit_receipt_formal.json \
        --out artifacts/rl_ready/p2_5/<run>/reveal

Checks the sealed truth's SHA against the frozen manifest, then CAL-06: |theta_hat_i - theta*_i| <=
max(3 sigma_hat_i, 0.02 theta*_i) for every parameter with sigma_hat_i / theta_hat_i < 0.5; the others are reported
as non-identifiable. CPU only.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def recovery(theta_hat, sigma_hat, theta_star, params):
    rows, verdict = [], "PASS"
    for k in params:
        th, s, ts = float(theta_hat[k]), float(sigma_hat[k]), float(theta_star[k])
        ident = math.isfinite(s) and s/abs(th) < 0.5
        bound = max(3*s, 0.02*abs(ts)) if math.isfinite(s) else math.inf
        err = abs(th-ts)
        ok = (err <= bound) if ident else None
        if ident and not ok:
            verdict = "FAIL"
        rows.append({"param": k, "theta_hat": th, "theta_star": ts, "sigma_hat": s, "abs_error": err,
                     "rel_error": err/abs(ts), "bound": bound, "identifiable": ident,
                     "verdict": ("PASS" if ok else "FAIL") if ident else "NON_IDENTIFIABLE_REPORTED"})
    return verdict, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-run", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--fit-receipts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    protocol, psha = C.load_protocol()
    from robotic_cutting_sim import rubber_calibration_synthetic as rcs
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    mpath = C.SYNTHETIC_DIR/protocol["formal"]["dataset_id"]/"manifest.json"
    manifest = rc2.load_manifest(mpath)
    ledger = Path(args.ledger)/f"heldout_{manifest['splits']['sha256']}.json"
    if not ledger.is_file():
        raise SystemExit("REVEAL_BEFORE_HELDOUT refused: the one-shot held-out ledger does not exist")
    held = json.loads(ledger.read_text())
    truth = rcs.reveal(Path(args.gen_run)/"sealed"/"truth.json", manifest["synthetic"]["sealed_truth_sha256"])
    out = {"utc_reveal": C.utc(), "tree": C.tree_state(), "protocol_sha256": psha,
           "manifest_freeze_sha256": manifest["splits"]["sha256"], "heldout_ledger": str(ledger),
           "heldout_utc": held.get("provenance", {}).get("utc"), "theta_star": truth["theta_star"],
           "rho_star_kg_m3": truth["rho_star_kg_m3"], "truth_seed": truth["truth_seed"],
           "noise_seed": truth["noise_seed"], "noise_seed_sha256_check":
           rcs.seed_sha256(int(truth["noise_seed"])) == manifest["synthetic"]["noise_seed_sha256"], "fits": [],
           **C.LABELS}
    params = protocol["parameter_space"]["params"]
    for p in args.fit_receipts:
        rec = json.loads(Path(p).read_text())
        verdict, rows = recovery(rec["theta_hat"], rec["sigma_hat"], truth["theta_star"], params)
        out["fits"].append({"fit_receipt": p, "kind": rec["provenance"]["kind"], "CAL-06": verdict, "rows": rows})
    C.write_new(Path(args.out)/"reveal_receipt.json", out)
    for f in out["fits"]:
        print(f["kind"], f["CAL-06"])
        for r in f["rows"]:
            print(f"  {r['param']:9s} hat {r['theta_hat']:.6g} star {r['theta_star']:.6g} sigma {r['sigma_hat']:.3g} "
                  f"rel err {r['rel_error']:.3%} {r['verdict']}")


if __name__ == "__main__":
    main()
