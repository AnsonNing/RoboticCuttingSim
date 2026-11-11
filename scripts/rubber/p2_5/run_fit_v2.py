"""P2-5 v2 staged fit (protocol rubber_calib_tests_v2 'fit'; supersedes the v1 fit procedure only).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_5/run_fit_v2.py \
        --kind formal|negative_control|pilot_validation --out artifacts/rl_ready/p2_5/<run>/<kind> --gpu-index 0

Stages (``rubber_calibration_fit.fit_staged``): S1 elastic (mu_eq) from T1, S2 visco (g1, tau1, g2, tau2) from R1 + T2
(3 Latin-hypercube starts over the prior box, lockstep), S3 friction (mu_table) from F1, S4 joint polish of all six on
the whole fit split; TRF with bounds = the prior box in u and x_scale "jac"; start = the prior-box centre. Negative
control: S1 and S2 from T2 only, no S3, S4 on T2 + C1. Wall-clock budget per kind (protocol); when it is exhausted the
best iterate of the final stage is the result and is scored normally. ``pilot_validation`` runs the formal method on
the PILOT dataset (method development; the pilot truth is revealed afterwards to check recovery). Never overwrites.
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

PROTOCOL_V2 = C.ROOT/"configs"/"rubber"/"rubber_calib_tests_v2.json"
PROTOCOL_V2_SHA = None      # set by the commit that pins rubber_calib_tests_v2
DEV_SETTINGS = {"deterministic": "default", "fd_step_u": 1e-2, "budget_s": {"pilot_validation": 4800}}


def prior_box_u(prior, space):
    from robotic_cutting_sim.rubber_calibration_fit import PARAMS
    G = space.G
    lg = lambda p: math.log(p/(1-p))   # noqa: E731
    lo = {"mu_eq_pa": math.log(prior["mu_eq_pa"][0]), "g1": lg(prior["g1"][0]/G),
          "g2": lg(prior["g2"][0]/(G-prior["g1"][0])), "tau1_s": math.log(prior["tau1_s"][0]),
          "tau2_s": math.log(prior["tau2_s"][0]), "mu_table": prior["mu_table"][0]}
    hi = {"mu_eq_pa": math.log(prior["mu_eq_pa"][1]), "g1": lg(prior["g1"][1]/G),
          "g2": lg(prior["g2"][1]/(G-prior["g1"][1])), "tau1_s": math.log(prior["tau1_s"][1]),
          "tau2_s": math.log(prior["tau2_s"][1]), "mu_table": prior["mu_table"][1]}
    return np.array([lo[k] for k in PARAMS]), np.array([hi[k] for k in PARAMS])


def centre(prior):
    g = lambda k: math.sqrt(prior[k][0]*prior[k][1])     # noqa: E731
    a = lambda k: 0.5*(prior[k][0]+prior[k][1])           # noqa: E731
    return {"mu_eq_pa": g("mu_eq_pa"), "g1": a("g1"), "tau1_s": g("tau1_s"), "g2": a("g2"), "tau2_s": g("tau2_s"),
            "mu_table": a("mu_table")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("formal", "negative_control", "pilot_validation"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists (no overwrite)")
    tree = C.tree_state()
    if not tree["clean"]:
        raise SystemExit("the tree must be clean")
    protocol, psha = C.load_protocol("1f6816411d141555973537995300ba87927a8162e8eba5f942d51faf40b2ff2b")
    if args.kind == "pilot_validation":
        settings, v2sha = DEV_SETTINGS, None
    else:
        import hashlib
        raw = PROTOCOL_V2.read_bytes()
        v2sha = hashlib.sha256(raw).hexdigest()
        if v2sha != PROTOCOL_V2_SHA:
            raise SystemExit(f"v2 protocol sha {v2sha} != pinned {PROTOCOL_V2_SHA}")
        settings = json.loads(raw)["fit"]
    free = C.require_gpu(args.gpu_index)
    out.mkdir(parents=True)
    log = C.Log(out/"log.txt")
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    if settings["deterministic"] == "run_to_run":
        wp.config.deterministic = wp.config.DeterministicMode.RUN_TO_RUN     # before any module is loaded
    wp.init()
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    from robotic_cutting_sim import rubber_calibration_synthetic as rcs
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    if args.kind == "pilot_validation":
        mpath = C.SYNTHETIC_DIR/"p2_5_pilot_001"/"manifest.json"
    else:
        mpath = C.SYNTHETIC_DIR/protocol["formal"]["dataset_id"]/"manifest.json"
    manifest = rc2.load_manifest(mpath)
    if args.kind != "pilot_validation" and manifest["status"] != "SYNTHETIC_FROZEN_FOR_HELDOUT":
        raise SystemExit("the formal fit needs the frozen formal manifest")
    cfg = rce.model_config(protocol["common"], "fit")
    space = rcf.space_from_protocol(protocol)
    box = prior_box_u(protocol["prior_box"], space)
    if args.kind == "negative_control":
        stages = (
            {"name": "S1_elastic", "params": ["mu_eq_pa"], "experiments": ["T2"], "starts": 1, "max_nfev": 8},
            {"name": "S2_visco", "params": ["g1", "tau1_s", "g2", "tau2_s"], "experiments": ["T2"], "starts": 3,
             "max_nfev": 15},
            {"name": "S4_joint_polish", "params": list(rcf.PARAMS), "experiments": ["T2", "C1"], "starts": 1,
             "max_nfev": 6})
    else:
        stages = rcf.V2_STAGES
    budget = settings["budget_s"][args.kind]
    objectives = []

    used = set()

    def make(exps):
        k = next(i for i, s in enumerate(stages) if s["experiments"] == exps and i not in used)
        used.add(k)
        st = stages[k]
        pop = st["starts"]*(len(st["params"])+1)
        o = rcf.Objective(mpath, cfg=cfg, population=pop, device="cuda:0",
                          experiments=exps if exps is not None else None)
        objectives.append(o)
        return o
    meta = {"kind": args.kind, "utc_start": C.utc(), "tree": tree, "protocol_v1_sha256": psha,
            "protocol_v2_sha256": v2sha, "settings": settings, "manifest": str(mpath),
            "manifest_freeze_sha256": manifest["splits"]["sha256"], "env": C.environment(),
            "module_sha256": C.module_shas(), "gpu_free_mib_at_start": free, "gpu_samples": [C.gpu_sample()],
            "stages": stages, "start_theta": centre(protocol["prior_box"]), "box_u": [box[0].tolist(), box[1].tolist()],
            "warp_deterministic": str(wp.config.deterministic), **C.LABELS}
    t0 = time.perf_counter()
    res = rcf.fit_staged(make, space, box, stages=stages, start_theta=centre(protocol["prior_box"]),
                         budget_s=budget, fd_step=settings["fd_step_u"], log=log)
    meta.update({"wall_s": time.perf_counter()-t0, "utc_end": C.utc(), "budget_hit": res["budget_hit"],
                 "objective_evaluations": sum(o.evaluations for o in objectives),
                 "candidates_evaluated": sum(o.candidates_evaluated for o in objectives),
                 "forward_receipts": [o.forward.receipts() for o in objectives]})
    meta["gpu_samples"].append(C.gpu_sample())
    rec, sha = rcf.freeze_fit_staged(out/"fit_receipt.json", result=res, space=space,
                                     provenance={"run": meta, "kind": args.kind},
                                     protocol_id="rubber_calib_tests_v2")
    log(f"fit receipt sha256 {sha}; chi2_red {res['chi2_reduced']:.4f}; budget_hit {res['budget_hit']}; "
        f"wall {meta['wall_s']:.0f}s")
    summary = {**meta, "fit_receipt_sha256": sha, "chi2_reduced": res["chi2_reduced"]}
    if args.kind == "pilot_validation":
        from reveal import recovery
        truth = rcs.reveal(C.ROOT/"artifacts"/"rl_ready"/"p2_5"/"pilot_gen_001"/"sealed"/"truth.json",
                           manifest["synthetic"]["sealed_truth_sha256"], reason="v2 pilot validation recovery")
        verdict, rows = recovery(rec["theta_hat"], rec["sigma_hat"], truth["theta_star"], rcf.PARAMS)
        summary["pilot_recovery"] = {"verdict": verdict, "rows": rows}
        for r in rows:
            log(f"  {r['param']:9s} hat {r['theta_hat']:.6g} star {r['theta_star']:.6g} sigma {r['sigma_hat']:.3g} "
                f"rel {r['rel_error']:.3%} {r['verdict']}")
        log(f"pilot recovery {verdict}")
    C.write_new(out/"fit_run_receipt.json", summary)


if __name__ == "__main__":
    main()
