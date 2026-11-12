"""P2-5 v2 method development (PILOT only, diagnostic): held-out metrics of the pilot-validation theta_hat against the
pilot's own held-out data and oracle, compared with rubber_calib_thresholds_v1 (which were derived from the same pilot,
so this is circular and indicative only; it never touches the formal dataset or its one-shot ledger).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
        PYTHONPATH=src .venv/bin/python scripts/rubber/p2_5/v2_pilot_heldout_check.py \
        --fit-receipt artifacts/rl_ready/p2_5/v2_dev/pilot_validation_001/fit_receipt.json \
        --out artifacts/rl_ready/p2_5/v2_dev/pilot_heldout_check_001 --gpu-index 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-receipt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output exists")
    free = C.require_gpu(args.gpu_index)
    out.mkdir(parents=True)
    C.setup_warp()
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    protocol, _ = C.load_protocol()
    thr = json.loads(C.THRESHOLDS.read_text())["tolerances"]
    rec = json.loads(Path(args.fit_receipt).read_text())
    mpath = C.SYNTHETIC_DIR/"p2_5_pilot_001"/"manifest.json"
    manifest = rc2.load_manifest(mpath)
    if rec["objective"]["manifest_freeze_sha256"] is not None or manifest["dataset_id"] != "p2_5_pilot_001":
        raise SystemExit("pilot only")
    specimens = {s["specimen_id"]: s for s in manifest["specimens"]}
    m_exps = {e["experiment_id"]: e for e in manifest["experiments"]}
    held = manifest["splits"]["heldout"]
    exps = [rcf.experiment_definition(m_exps[e], specimens[m_exps[e]["specimen_id"]]) for e in held]
    dens = {e: rcf.measured_density(specimens[m_exps[e]["specimen_id"]]) for e in held}
    sensors = {e: m_exps[e]["sensors"] for e in held}
    data = {e: {r: rce.read_channels(mpath.parent/f["file"]) for r, f in m_exps[e]["data_files"].items()} for e in held}
    oracle = json.loads((mpath.parent/"oracle_heldout.json").read_text())
    fwd = rcf.PopulationForward(exps, dens, rce.model_config(protocol["common"], "fit"), 1,
                                device="cuda:0").bind_sensors(sensors)
    t0 = time.perf_counter()
    pred = fwd([rec["theta_hat"]], exps)[0]
    rows, verdict = [], "PASS"
    for exp in exps:
        m = rce.metrics(exp, pred[exp["id"]], data[exp["id"]], pred[exp["id"]]["oracle"], oracle[exp["id"]],
                        protocol["metrics"]["settings"])
        for k, v in sorted(m.items()):
            ok = v <= thr[exp["id"]][k]
            verdict = verdict if ok else "FAIL"
            rows.append({"experiment": exp["id"], "metric": k, "value": v, "tol": thr[exp["id"]][k],
                         "verdict": "PASS" if ok else "FAIL"})
            print(exp["id"], k, f"{v:.5g}", f"tol {thr[exp['id']][k]:.5g}", "PASS" if ok else "FAIL", flush=True)
    C.write_new(out/"pilot_heldout_check.json", {"label": "DIAGNOSTIC_PILOT_ONLY (circular: thresholds from this pilot)",
                                                  "theta_hat": rec["theta_hat"], "rows": rows, "verdict": verdict,
                                                  "wall_s": time.perf_counter()-t0, "gpu_free_mib_at_start": free,
                                                  "tree": C.tree_state(), "utc": C.utc(), **C.LABELS})
    print("verdict", verdict)


if __name__ == "__main__":
    main()
