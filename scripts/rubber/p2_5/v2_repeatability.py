"""P2-5 v2 method development (PILOT data only): GPU repeatability of the fit-split forward model at theta_pilot.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
        PYTHONPATH=src .venv/bin/python scripts/rubber/p2_5/v2_repeatability.py --mode default|run_to_run \
        --out artifacts/rl_ready/p2_5/v2_dev/repeat_<mode> --gpu-index 0

For every fit-split experiment of the pilot manifest one batch of R identical envs (theta_pilot, the pilot's measured
densities) runs the full record R_runs times; the spread of the weighted residuals across the R x R_runs copies is the
GPU forward noise. Also one forward-difference population (theta_pilot + h e_i, h = 1e-3 in u) gives the FD signal per
parameter. ``--mode run_to_run`` sets wp.config.deterministic = RUN_TO_RUN before any module is loaded.
Never reads the formal dataset or its sealed truth.
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
    ap.add_argument("--mode", choices=("default", "run_to_run"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output exists")
    free = C.require_gpu(args.gpu_index)
    out.mkdir(parents=True)
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    if args.mode == "run_to_run":
        wp.config.deterministic = wp.config.DeterministicMode.RUN_TO_RUN
    wp.init()
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    from robotic_cutting_sim import rubber_calibration_synthetic as rcs
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    protocol, psha = C.load_protocol()
    pilot = C.ROOT/"artifacts"/"rl_ready"/"p2_5"/"pilot_gen_001"
    mpath = C.SYNTHETIC_DIR/"p2_5_pilot_001"/"manifest.json"
    manifest = rc2.load_manifest(mpath)
    theta = rcs.reveal(pilot/"sealed"/"truth.json", manifest["synthetic"]["sealed_truth_sha256"],
                       reason="v2 method development: repeatability at theta_pilot")["theta_star"]
    obj = rcf.Objective(mpath, forward=lambda *a: None)          # data / sensors / densities of the fit split only
    cfg = rce.model_config(protocol["common"], "fit")
    rows, wall = {}, {}
    space = rcf.space_from_protocol(protocol)
    for exp in obj.experiments:
        eid = exp["id"]
        t0 = time.perf_counter()
        b = rce.ExperimentBatch(exp, obj.densities[eid], cfg, args.copies, device="cuda:0")
        res = []
        for _ in range(args.runs):
            recs = b.run([rce.material_from_theta(theta, obj.densities[eid], cfg.beta)]*args.copies)
            for r in recs:
                obs = rce.observables(exp, rce.finalize_record(r), cfg)
                ch = rce.sample_channels(exp, obs, obj.sensors[eid])
                res.append(rce.residuals(exp, ch, obj.data[eid], obj.sensors[eid]))
        R = np.array(res)
        spread = R.std(axis=0)
        u0 = space.to_u(theta)
        fd = {}
        pts = []
        for i, name in enumerate(rcf.PARAMS):
            e = u0.copy()
            e[i] += 1e-3
            pts.append(rce.material_from_theta(space.from_u(e), obj.densities[eid], cfg.beta))
        b2 = rce.ExperimentBatch(exp, obj.densities[eid], cfg, len(pts), device="cuda:0")
        for i, r in enumerate(b2.run(pts)):
            obs = rce.observables(exp, rce.finalize_record(r), cfg)
            ch = rce.sample_channels(exp, obs, obj.sensors[eid])
            d = rce.residuals(exp, ch, obj.data[eid], obj.sensors[eid])-R[0]
            fd[rcf.PARAMS[i]] = {"rms_delta_residual_h1e-3": float(np.sqrt(np.mean(d**2)))}
        wall[eid] = time.perf_counter()-t0
        rows[eid] = {"n_residuals": int(R.shape[1]), "copies_x_runs": int(R.shape[0]),
                     "spread_rms": float(np.sqrt(np.mean(spread**2))), "spread_max": float(spread.max()),
                     "bitwise_identical_copies": bool(all(np.array_equal(R[0], x) for x in R[1:])),
                     "fd_signal_h1e-3": fd, "chi2_mean_square": float(np.mean(R[0]**2)), "wall_s": wall[eid]}
        print(eid, json.dumps({k: v for k, v in rows[eid].items() if k != "fd_signal_h1e-3"}),
              {k: round(v["rms_delta_residual_h1e-3"], 5) for k, v in fd.items()}, flush=True)
        del b, b2
    C.write_new(out/"repeatability.json", {"mode": args.mode, "theta": "theta_pilot (pilot, revealed)",
                                           "rows": rows, "gpu_free_mib_at_start": free, "tree": C.tree_state(),
                                           "utc": C.utc(), "env": C.environment(), "gpu": C.gpu_sample(),
                                           "protocol_sha256": psha, **C.LABELS})


if __name__ == "__main__":
    main()
