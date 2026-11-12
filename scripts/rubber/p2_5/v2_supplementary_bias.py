"""P2-5 v2 SUPPLEMENTARY DIAGNOSTIC (never scored; run only after the one-shot held-out and the reveal): why theta_hat
differs from theta*. The fit model (n_t 4, CFL 0.5, measured densities) is evaluated at theta_hat and at the revealed
theta* in ONE population: per-experiment chi2 on the formal FIT split and the held-out metrics against the frozen
thresholds. If chi2(theta*) < chi2(theta_hat) the fit did not reach the fit model's optimum (optimisation); if
chi2(theta*) > chi2(theta_hat) the fit model's optimum itself lies away from theta* (resolution bias).
Counts against the formal fit's 3 h budget (GPU lock per evaluation).

    env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python \
        scripts/rubber/p2_5/v2_supplementary_bias.py --reveal artifacts/rl_ready/p2_5/fit_v2_001/reveal \
        --out artifacts/rl_ready/p2_5/fit_v2_001/supplementary_bias --gpu-index 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output exists")
    rv = json.loads((Path(args.reveal)/"reveal_receipt.json").read_text())
    out.mkdir(parents=True)
    C.setup_warp()
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    protocol, _ = C.load_protocol()
    thr = json.loads(C.THRESHOLDS.read_text())["tolerances"]
    fit = json.loads((C.ROOT/"docs/evidence/rl_ready/p2_5/fit_receipt_formal.json").read_text())
    thetas = {"theta_hat": fit["theta_hat"], "theta_star": rv["theta_star"]}
    mpath = C.SYNTHETIC_DIR/protocol["formal"]["dataset_id"]/"manifest.json"
    manifest = rc2.load_manifest(mpath)
    cfg = rce.model_config(protocol["common"], "fit")
    lock = C.GPULockPerEvaluation(args.gpu_index)
    obj = rcf.Objective(mpath, cfg=cfg, population=2, device="cuda:0")
    t0 = time.perf_counter()
    with lock:
        preds = obj.forward([thetas["theta_hat"], thetas["theta_star"]], obj.experiments)
    chi2 = {}
    for name, pred in zip(thetas, preds):
        rows = {}
        for e in obj.experiments:
            r = rce.residuals(e, pred[e["id"]], obj.data[e["id"]], obj.sensors[e["id"]])
            rows[e["id"]] = {"chi2": float(r @ r), "n": int(r.size)}
        tot = sum(v["chi2"] for v in rows.values())
        n = sum(v["n"] for v in rows.values())
        chi2[name] = {"per_experiment": rows, "chi2_reduced_total": tot/(n-6)}
    specimens = {s["specimen_id"]: s for s in manifest["specimens"]}
    m_exps = {e["experiment_id"]: e for e in manifest["experiments"]}
    held = manifest["splits"]["heldout"]
    hexps = [rcf.experiment_definition(m_exps[e], specimens[m_exps[e]["specimen_id"]]) for e in held]
    fwd = rcf.PopulationForward(hexps, {e: rcf.measured_density(specimens[m_exps[e]["specimen_id"]]) for e in held},
                                cfg, 1, device="cuda:0").bind_sensors({e: m_exps[e]["sensors"] for e in held})
    data = {e: {r: rce.read_channels(mpath.parent/f["file"]) for r, f in m_exps[e]["data_files"].items()} for e in held}
    oracle = json.loads((mpath.parent/"oracle_heldout.json").read_text())
    with lock:
        pstar = fwd([thetas["theta_star"]], hexps)[0]
    hrows = []
    for e in hexps:
        m = rce.metrics(e, pstar[e["id"]], data[e["id"]], pstar[e["id"]]["oracle"], oracle[e["id"]],
                        protocol["metrics"]["settings"])
        for k, v in sorted(m.items()):
            hrows.append({"experiment": e["id"], "metric": k, "value_at_theta_star": float(v),
                          "tol": thr[e["id"]][k], "within_tol": bool(v <= thr[e["id"]][k])})
    res = {"label": "SUPPLEMENTARY_DIAGNOSTIC_NOT_SCORED (after the one-shot held-out and the reveal)",
           "thetas": thetas, "fit_split_chi2": chi2, "heldout_at_theta_star": hrows,
           "interpretation": ("optimisation shortfall" if chi2["theta_star"]["chi2_reduced_total"]
                              < chi2["theta_hat"]["chi2_reduced_total"] else "fit-model optimum away from theta*"),
           "gpu_lock": lock.receipt(), "wall_s": time.perf_counter()-t0, "tree": C.tree_state(), "utc": C.utc(),
           **C.LABELS}
    C.write_new(out/"supplementary_bias.json", res)
    for name in thetas:
        print(name, f"chi2_red {chi2[name]['chi2_reduced_total']:.4f}",
              {k: round(v["chi2"]/v["n"], 3) for k, v in chi2[name]["per_experiment"].items()})
    print("held-out rows within tol at theta*:", sum(r["within_tol"] for r in hrows), "/", len(hrows))
    print("interpretation:", res["interpretation"])


if __name__ == "__main__":
    main()
