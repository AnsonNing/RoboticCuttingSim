"""P2-5 v3 staged fit (protocol rubber_calib_tests_v3 'fit'; the FINAL synthetic attempt of this round).

    env CUDA_VISIBLE_DEVICES=<N> TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python \
        scripts/rubber/p2_5/run_fit_v3.py --kind formal|negative_control|dev_formal001 \
        --out artifacts/rl_ready/p2_5/<run>/<kind> --gpu-index <N>

Stages (``rubber_calibration_fit.fit_staged``, settings from the pinned v3 protocol; ``dev_formal001`` uses the same
settings before the declaration, on the REVEALED formal_001 fit split, labelled development):
- A: mu_eq + g1, tau1, g2, tau2 on R1 + T2 (mu_eq is not frozen from a T1-only fit), 2 lockstep starts (the
  prior-box centre + one seeded LHS point over the prior box);
- B: mu_table on F1;
- C: joint polish of all six on the whole fit split.

Every stage stops on convergence (xtol / ftol) or the GPU-hold budget, not on small caps (max_nfev is a declared
safety bound only). Bounds: the v1 parameter-space box, wider than the prior box the truth is drawn from. The script
takes the blocking GPU lock per forward evaluation and counts only its own hold time. Never overwrites.
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

PROTOCOL_V3 = C.ROOT/"configs"/"rubber"/"rubber_calib_tests_v3.json"
PROTOCOL_V3_SHA = None      # set by the commit that pins rubber_calib_tests_v3
DEV_SETTINGS = {"deterministic": "default", "fd_step_u": 1e-2, "xtol": 1e-4, "ftol": 1e-3, "gtol": 1e-8,
                "budget_s": {"dev_formal001": 3240}, "dataset_id": "p2_5_formal_001", "lhs_seed": 2601,
                "stages_formal": [
                    {"name": "A_elastic_visco", "params": ["mu_eq_pa", "g1", "tau1_s", "g2", "tau2_s"],
                     "experiments": ["R1", "T2"], "starts": 2, "start_design": "current_plus_lhs", "max_nfev": 60},
                    {"name": "B_friction", "params": ["mu_table"], "experiments": ["F1"], "starts": 1,
                     "max_nfev": 15},
                    {"name": "C_joint", "params": ["mu_eq_pa", "g1", "tau1_s", "g2", "tau2_s", "mu_table"],
                     "experiments": None, "starts": 1, "max_nfev": 40}]}


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
    ap.add_argument("--kind", choices=("formal", "negative_control", "dev_formal001"), required=True)
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
    import hashlib
    if args.kind == "dev_formal001":
        settings, v3sha = DEV_SETTINGS, None
    else:
        raw = PROTOCOL_V3.read_bytes()
        v3sha = hashlib.sha256(raw).hexdigest()
        if v3sha != PROTOCOL_V3_SHA:
            raise SystemExit(f"v3 protocol sha {v3sha} != pinned {PROTOCOL_V3_SHA}")
        settings = json.loads(raw)["fit"]
    lock = C.GPULockPerEvaluation(args.gpu_index)
    with lock:
        free = C.require_gpu(args.gpu_index)
    out.mkdir(parents=True)
    log = C.Log(out/"log.txt")
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    mpath = C.SYNTHETIC_DIR/settings["dataset_id"]/"manifest.json"
    manifest = rc2.load_manifest(mpath)
    if manifest["status"] != "SYNTHETIC_FROZEN_FOR_HELDOUT":
        raise SystemExit("the fit needs a frozen formal manifest")
    if args.kind == "dev_formal001" and manifest["dataset_id"] != "p2_5_formal_001":
        raise SystemExit("development runs use the revealed formal_001 only")
    cfg = rce.model_config(protocol["common"], "fit")
    space = rcf.space_from_protocol(protocol)
    prior_u = prior_box_u(protocol["prior_box"], space)
    box = None                      # v3: the v1 parameter-space box, wider than the prior box
    key = "stages_negative_control" if args.kind == "negative_control" else "stages_formal"
    stages = []
    for st in settings[key]:
        st = dict(st)
        if st.get("start_design") == "current_plus_lhs":
            st["lhs_box"] = (prior_u[0], prior_u[1])
        stages.append(st)
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
        return C.LockedObjective(o, lock) if lock is not None else o
    meta = {"kind": args.kind, "utc_start": C.utc(), "tree": tree, "protocol_v1_sha256": psha,
            "protocol_v3_sha256": v3sha, "settings": settings, "manifest": str(mpath),
            "label": "DEVELOPMENT (revealed formal_001; never scored)" if args.kind == "dev_formal001" else "FORMAL",
            "manifest_freeze_sha256": manifest["splits"]["sha256"], "env": C.environment(),
            "module_sha256": C.module_shas(), "gpu_free_mib_at_start": free, "gpu_samples": [C.gpu_sample()],
            "stages": [{k: v for k, v in st.items() if k != "lhs_box"} for st in stages],
            "start_theta": centre(protocol["prior_box"]), "bounds_u": [space.lb.tolist(), space.ub.tolist()],
            "lhs_box_u": [prior_u[0].tolist(), prior_u[1].tolist()],
            "warp_deterministic": str(wp.config.deterministic), **C.LABELS}
    t0 = time.perf_counter()
    res = rcf.fit_staged(make, space, box, stages=stages, start_theta=centre(protocol["prior_box"]),
                         budget_s=budget, fd_step=settings["fd_step_u"], log=log, seed=settings["lhs_seed"],
                         xtol=settings["xtol"], ftol=settings["ftol"], gtol=settings["gtol"], clock=lock.clock)
    meta.update({"wall_s": time.perf_counter()-t0, "utc_end": C.utc(), "budget_hit": res["budget_hit"],
                 "budget_clock": "GPU-lock hold time",
                 "stage_terminations": [[(st.get("termination"), st.get("last_relative_cost_drop"))
                                         for st in x.get("starts", [])] for x in res["stages"]],
                 "budget_clock_used_s": res["budget_clock_used_s"], "budget_s": budget,
                 "gpu_lock": lock.receipt(),
                 "objective_evaluations": sum(o.evaluations for o in objectives),
                 "candidates_evaluated": sum(o.candidates_evaluated for o in objectives),
                 "forward_receipts": [o.forward.receipts() for o in objectives]})
    meta["gpu_samples"].append(C.gpu_sample())
    rec, sha = rcf.freeze_fit_staged(out/"fit_receipt.json", result=res, space=space,
                                     provenance={"run": meta, "kind": args.kind},
                                     protocol_id="rubber_calib_tests_v3")
    log(f"fit receipt sha256 {sha}; chi2_red {res['chi2_reduced']:.4f}; budget_hit {res['budget_hit']}; "
        f"wall {meta['wall_s']:.0f}s")
    summary = {**meta, "fit_receipt_sha256": sha, "chi2_reduced": res["chi2_reduced"]}
    if args.kind == "dev_formal001":
        from reveal import recovery
        rv = json.loads((C.ROOT/"artifacts/rl_ready/p2_5/fit_v2_001/reveal/reveal_receipt.json").read_text())
        verdict, rows = recovery(rec["theta_hat"], rec["sigma_hat"], rv["theta_star"], rcf.PARAMS)
        summary["dev_recovery_against_revealed_formal001"] = {"verdict": verdict, "rows": rows}
        for r in rows:
            log(f"  {r['param']:9s} hat {r['theta_hat']:.6g} star {r['theta_star']:.6g} sigma {r['sigma_hat']:.3g} "
                f"rel {r['rel_error']:.3%} {r['verdict']}")
        log(f"DEVELOPMENT recovery (revealed formal_001, never scored) {verdict}")
    C.write_new(out/"fit_run_receipt.json", summary)


if __name__ == "__main__":
    main()
