"""P2-5 one-shot held-out evaluation (task P5.6; CAL-05 primary, CAL-07 negative control):

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> \
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_5/run_heldout.py \
        --fit-receipts docs/evidence/rl_ready/p2_5/fit_receipt_formal.json \
                       docs/evidence/rl_ready/p2_5/fit_receipt_negative_control.json \
        --ledger artifacts/rl_ready/p2_5/<run>/heldout --gpu-index <N>

Both fit receipts must be committed (tracked and unchanged in the clean tree) before this runs. The fit model
(n_t 4, CFL 0.5, measured densities) predicts every held-out experiment at each theta_hat; metrics against the frozen
held-out data (M8 / M9 against the oracle) are compared with the SHA-pinned thresholds. One-shot per manifest freeze
(rubber_calibration_fit.evaluate_heldout ledger).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PROTOCOL_SHA = "1f6816411d141555973537995300ba87927a8162e8eba5f942d51faf40b2ff2b"     # rubber_calib_tests_v1 (c5054b9)
THRESHOLDS_SHA = "40be2f8bb7053199ef6b29776e0a7f125dff350e224124c10b07e965182caa36"   # rubber_calib_thresholds_v1 (4d7d2c4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-receipts", nargs="+", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--audit-rerun", default=None)
    args = ap.parse_args()
    protocol, psha = C.load_protocol(PROTOCOL_SHA)
    import hashlib
    traw = C.THRESHOLDS.read_bytes()
    tsha = hashlib.sha256(traw).hexdigest()
    if tsha != THRESHOLDS_SHA:
        raise SystemExit(f"thresholds sha {tsha} != pinned {THRESHOLDS_SHA}")
    thresholds = json.loads(traw)
    tree = C.tree_state()
    if not tree["clean"]:
        raise SystemExit("the tree must be clean")
    for p in args.fit_receipts:
        rel = str(Path(p).resolve().relative_to(C.ROOT))
        if not C.git("ls-files", rel):
            raise SystemExit(f"{p} is not committed")
    free = C.require_gpu(args.gpu_index)
    log = C.Log(Path(args.ledger)/"log.txt")
    C.setup_warp()
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    mpath = C.SYNTHETIC_DIR/protocol["formal"]["dataset_id"]/"manifest.json"
    manifest = rc2.load_manifest(mpath)
    specimens = {s["specimen_id"]: s for s in manifest["specimens"]}
    m_exps = {e["experiment_id"]: e for e in manifest["experiments"]}
    held = manifest["splits"]["heldout"]
    dens = {e: rcf.measured_density(specimens[m_exps[e]["specimen_id"]]) for e in held}
    sensors = {e: m_exps[e]["sensors"] for e in held}
    cfg = rce.model_config(protocol["common"], "fit")
    exps_all = [rcf.experiment_definition(m_exps[e], specimens[m_exps[e]["specimen_id"]]) for e in held]
    fwd = rcf.PopulationForward(exps_all, dens, cfg, 1, device="cuda:0").bind_sensors(sensors)
    walls = []

    def predict(theta, exps):
        t0 = time.perf_counter()
        out = fwd([theta], exps)[0]
        walls.append(time.perf_counter()-t0)
        log(f"predicted held-out at a theta_hat in {walls[-1]:.1f}s; invalid "
            f"{ {k: v['invalid_reason'] for k, v in out.items() if v['invalid_reason']} }")
        return out
    t_start = time.perf_counter()
    res = rcf.evaluate_heldout(mpath, args.fit_receipts, args.ledger, predict=predict,
                               thresholds=thresholds["tolerances"], metric_settings=protocol["metrics"]["settings"],
                               audit_rerun=args.audit_rerun,
                               provenance={"tree": tree, "protocol_sha256": psha, "thresholds_sha256": tsha,
                                           "utc": C.utc(), "env": C.environment(), "gpu_free_mib_at_start": free,
                                           "gpu_samples": [C.gpu_sample()], "module_sha256": C.module_shas()})
    viscous = set(protocol["metrics"]["viscous_metrics"])
    summary = {"utc_end": C.utc(), "wall_s": time.perf_counter()-t_start, "predict_wall_s": walls,
               "forward_receipts": fwd.receipts(), "CAL-05": res["fits"][0]["verdict"], **C.LABELS}
    if len(res["fits"]) > 1:
        neg = res["fits"][1]
        failed_viscous = [r for r in neg["rows"] if r["metric"] in viscous and r["verdict"] == "FAIL"]
        summary["CAL-07"] = "PASS" if failed_viscous else "FAIL"
        summary["negative_control_failed_viscous_rows"] = failed_viscous
    C.write_new(Path(args.ledger)/("heldout_summary.json" if not args.audit_rerun
                                   else f"heldout_summary_audit_{args.audit_rerun}.json"), summary)
    log(f"CAL-05 {summary['CAL-05']} CAL-07 {summary.get('CAL-07')}")


if __name__ == "__main__":
    main()
