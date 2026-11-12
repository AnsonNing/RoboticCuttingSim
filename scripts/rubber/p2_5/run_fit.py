"""P2-5 formal fits (task P5.6; protocol rubber_calib_tests_v1 'fit'):

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> \
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_5/run_fit.py \
        --kind formal|negative_control|cmaes --out artifacts/rl_ready/p2_5/<run>/<kind> --gpu-index <N>

formal: TRF (scipy least_squares, method trf) from 3 Latin-hypercube starts (seed 2511, prior-box u-image), the starts
in lockstep (one forward population per round), all fit-split experiments; writes the fit receipt.
negative_control: the same on T2 + C1 only (CAL-07).
cmaes: the in-repo CMA-ES through the ask / tell hook (reported, not formal).
The objective reads the frozen formal manifest only (fit split); sealed paths are refused. Never overwrites.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PROTOCOL_SHA = "1f6816411d141555973537995300ba87927a8162e8eba5f942d51faf40b2ff2b"     # rubber_calib_tests_v1 (c5054b9)


def start_box(protocol, space):
    """u-image of the prior box (protocol fit.formal_method.start_design)."""
    pb, G = protocol["prior_box"], space.G
    lg = lambda p: math.log(p/(1-p))   # noqa: E731
    lo = {"mu_eq_pa": math.log(pb["mu_eq_pa"][0]), "g1": lg(pb["g1"][0]/G), "g2": lg(pb["g2"][0]/(G-pb["g1"][0])),
          "tau1_s": math.log(pb["tau1_s"][0]), "tau2_s": math.log(pb["tau2_s"][0]), "mu_table": pb["mu_table"][0]}
    hi = {"mu_eq_pa": math.log(pb["mu_eq_pa"][1]), "g1": lg(pb["g1"][1]/G), "g2": lg(pb["g2"][1]/(G-pb["g1"][1])),
          "tau1_s": math.log(pb["tau1_s"][1]), "tau2_s": math.log(pb["tau2_s"][1]), "mu_table": pb["mu_table"][1]}
    from robotic_cutting_sim.rubber_calibration_fit import PARAMS
    lo, hi = [lo[k] for k in PARAMS], [hi[k] for k in PARAMS]       # u order = PARAMS order
    if not (space.inside(lo) and space.inside(hi)):
        raise SystemExit("start box outside the fit box")
    return np.array(lo), np.array(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("formal", "negative_control", "cmaes"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny budget against a given (non-formal) manifest")
    args = ap.parse_args()
    protocol, psha = C.load_protocol(None if args.smoke else PROTOCOL_SHA)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists (no overwrite)")
    tree = C.tree_state()
    if not args.smoke and not tree["clean"]:
        raise SystemExit("the tree must be clean")
    mpath = Path(args.manifest) if args.manifest else C.SYNTHETIC_DIR/protocol["formal"]["dataset_id"]/"manifest.json"
    free = C.require_gpu(args.gpu_index)
    out.mkdir(parents=True)
    log = C.Log(out/"log.txt")
    C.setup_warp()
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    manifest = rc2.load_manifest(mpath)
    if not args.smoke and (manifest["status"] != "SYNTHETIC_FROZEN_FOR_HELDOUT"
                           or manifest["synthetic"]["protocol_sha256"] != psha):
        raise SystemExit("the formal fit needs the frozen formal manifest of this protocol")
    cfg = rce.model_config(protocol["common"], "fit")
    space = rcf.space_from_protocol(protocol)
    fm = protocol["fit"]["formal_method"]
    lo, hi = start_box(protocol, space)
    starts = space.latin_hypercube(fm["starts"], seed=2511, box=(lo, hi))
    experiments = None if args.kind != "negative_control" else ["T2", "C1"]
    n_starts = fm["starts"] if args.kind != "cmaes" else 1
    pop = (space.p+1)*n_starts
    obj = rcf.Objective(mpath, cfg=cfg, population=pop, device="cuda:0", experiments=experiments)
    meta = {"kind": args.kind, "smoke": args.smoke, "utc_start": C.utc(), "tree": tree, "protocol_sha256": psha,
            "manifest": str(mpath), "manifest_freeze_sha256": manifest["splits"]["sha256"], "env": C.environment(),
            "module_sha256": C.module_shas(), "gpu_free_mib_at_start": free, "gpu_samples": [C.gpu_sample()],
            "starts_u": starts.tolist(), "starts_theta": [space.from_u(u) for u in starts], **C.LABELS}
    t0 = time.perf_counter()
    if args.kind in ("formal", "negative_control"):
        budget = 2 if args.smoke else fm["max_nfev"]
        res = rcf.fit_trf_lockstep(obj, space, starts, max_nfev=budget, xtol=fm["xtol"], ftol=fm["ftol"],
                                   gtol=fm["gtol"], fd_step=fm["fd_step_u"], log=log)
        meta.update({"wall_s": time.perf_counter()-t0, "objective_evaluations": obj.evaluations,
                     "candidates_evaluated": obj.candidates_evaluated, "population_sizes": res["population_sizes"],
                     "env_evaluations": obj.forward.evaluator.env_evaluations,
                     "forward_receipts": obj.forward.receipts(), "utc_end": C.utc()})
        meta["gpu_samples"].append(C.gpu_sample())
        rec, sha = rcf.freeze_fit(out/"fit_receipt.json", objective=obj, space=space, trf=res,
                                  provenance={"run": meta, "kind": args.kind})
        log(f"fit receipt sha256 {sha}; best start {res['best']} chi2_red "
            f"{res['starts'][res['best']]['chi2_reduced']:.4f} nfev {[s['nfev'] for s in res['starts']]}")
        C.write_new(out/"fit_run_receipt.json", {**meta, "fit_receipt_sha256": sha})
    else:
        cm = protocol["fit"]["comparison_method"]
        es = rcf.CMAES(starts[0], cm["sigma0_u"], cm["popsize"], cm["seed"], lb=space.lb, ub=space.ub)
        gens = 1 if args.smoke else cm["generations"]
        res = rcf.run_ask_tell(es, obj, space, gens, log=log)
        r = obj.residuals_population([res["best_theta"]])[0] if args.smoke else None
        meta.update({"wall_s": time.perf_counter()-t0, "objective_evaluations": obj.evaluations,
                     "candidates_evaluated": obj.candidates_evaluated, "utc_end": C.utc(),
                     "result": res, "status": "REPORTED_NOT_FORMAL", "smoke_check": None if r is None else r.size,
                     "chi2_reduced_best": 2*res["best_loss"]/max(1, (obj.n_residuals or 0)-space.p)})
        C.write_new(out/"cmaes_result.json", meta)
        log(f"CMA-ES best loss {res['best_loss']:.6e}")


if __name__ == "__main__":
    main()
