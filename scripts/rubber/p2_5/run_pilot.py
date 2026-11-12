"""P2-5 pilot run (task P5.4): tolerances tol_mk = max(2 (a_mk + b_mk), floor_m) and chi2_pilot (protocol
rubber_calib_tests_v1 tolerance_rule), written to configs/rubber/rubber_calib_thresholds_v1.json (exclusive create).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> \
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_5/run_pilot.py \
        --pilot-run artifacts/rl_ready/p2_5/<pilot gen run> --out-run artifacts/rl_ready/p2_5/<pilot eval run> \
        --gpu-index <N>

The pilot's sealed truth is revealed here (allowed: the pilot is used only for the tolerances). a_mk: the metric
between the pilot's two noise realisations (A as the prediction, B as the observation; 0 for M8 / M9). b_mk: the
metric between the noise-free pilot truth (observation) and the fit model at theta_pilot with the pilot manifest's
measured densities (prediction). chi2_pilot: reduced chi2 of that fit-model prediction against realisation A, fit
split, n - p degrees of freedom.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PROTOCOL_SHA = "1f6816411d141555973537995300ba87927a8162e8eba5f942d51faf40b2ff2b"     # rubber_calib_tests_v1 (c5054b9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-run", required=True)
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--thresholds-out", default=str(C.THRESHOLDS))
    args = ap.parse_args()
    protocol, psha = C.load_protocol(PROTOCOL_SHA)
    out = Path(args.out_run)
    if out.exists() or Path(args.thresholds_out).exists():
        raise SystemExit("output exists (no overwrite)")
    tree = C.tree_state()
    if not tree["clean"]:
        raise SystemExit("the tree must be clean")
    free = C.require_gpu(args.gpu_index)
    out.mkdir(parents=True)
    log = C.Log(out/"log.txt")
    C.setup_warp()
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    from robotic_cutting_sim import rubber_calibration_synthetic as rcs
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    t_start = time.perf_counter()
    pilot = Path(args.pilot_run)
    gen = json.loads((pilot/"gen_receipt.json").read_text())
    mpath = Path(gen["dataset_dir"])/"manifest.json"
    manifest = rc2.load_manifest(mpath)
    if manifest["status"] != "SYNTHETIC_PILOT" or manifest["synthetic"]["protocol_sha256"] != psha:
        raise SystemExit("not a pilot manifest of this protocol")
    truth = rcs.reveal(pilot/"sealed"/"truth.json", manifest["synthetic"]["sealed_truth_sha256"])
    theta_pilot = truth["theta_star"]
    log(f"pilot revealed (sealed sha {manifest['synthetic']['sealed_truth_sha256'][:12]}…)")
    specimens = {s["specimen_id"]: s for s in manifest["specimens"]}
    m_exps = {e["experiment_id"]: e for e in manifest["experiments"]}
    exps = [rcf.experiment_definition(m_exps[e["id"]], specimens[m_exps[e["id"]]["specimen_id"]])
            for e in protocol["experiments"]]
    dens = {e["id"]: rcf.measured_density(specimens[m_exps[e["id"]]["specimen_id"]]) for e in exps}
    sensors = {e["id"]: m_exps[e["id"]]["sensors"] for e in exps}
    cfg = rce.model_config(protocol["common"], "fit")
    fwd = rcf.PopulationForward(exps, dens, cfg, 1, device="cuda:0").bind_sensors(sensors)
    t0 = time.perf_counter()
    pred = fwd([theta_pilot], exps)[0]
    log(f"fit model at theta_pilot: {time.perf_counter()-t0:.1f}s; invalid "
        f"{ {k: v['invalid_reason'] for k, v in pred.items() if v['invalid_reason']} }")
    data_a = {e["id"]: {r: rce.read_channels(mpath.parent/f["file"]) for r, f in m_exps[e["id"]]["data_files"].items()}
              for e in exps}
    data_b = {e["id"]: {r: rce.read_channels(pilot/"pilot_realisation_B"/rcs.file_name(e["id"], r))
                        for r in m_exps[e["id"]]["data_files"]} for e in exps}
    clean = {k: {r: {c: np.asarray(v) for c, v in cols.items()} for r, cols in ch.items()}
             for k, ch in truth["clean_channels"].items()}
    # chi2_pilot (fit split, realisation A)
    res = [rce.residuals(e, pred[e["id"]], data_a[e["id"]], sensors[e["id"]]) for e in exps if e["split"] == "fit"]
    r = np.concatenate(res)
    p = len(protocol["parameter_space"]["params"])
    chi2_pilot = float(r @ r)/(r.size-p)
    per_exp_chi2 = {e["id"]: float(x @ x)/max(1, x.size) for e, x in zip([e for e in exps if e["split"] == "fit"], res)}
    settings = protocol["metrics"]["settings"]
    floors = {m: protocol["metrics"][m]["floor"] for m in protocol["metrics"] if m.startswith("M")}
    tol = {}
    for e in exps:
        if e["split"] != "heldout":
            continue
        eid = e["id"]
        a = rce.metrics(e, data_a[eid], data_b[eid], None, None, settings)
        b = rce.metrics(e, pred[eid], clean[eid], pred[eid]["oracle"], truth["oracle"][eid], settings)
        rows = {}
        for m in protocol["metrics"]["per_type"][e["type"]]:
            am = 0.0 if m in ("M8", "M9") else float(a[m])
            bm = float(b[m])
            value = max(2.0*(am+bm), floors[m])
            rows[m] = {"tol": value, "a_noise_floor": am, "b_discretisation_floor": bm, "floor": floors[m],
                       "binding": "floor" if value == floors[m] else "2(a+b)"}
            if not np.isfinite(value):
                raise SystemExit(f"non-finite tolerance {eid} {m}: a {am} b {bm}")
        tol[eid] = rows
    thresholds = {"schema": "rcs_rubber_calib_thresholds_v1", "id": "rubber_calib_thresholds_v1",
                  "protocol": "rubber_calib_tests_v1", "protocol_sha256": psha, **C.LABELS,
                  "rule": protocol["tolerance_rule"]["formula"],
                  "pilot": {"dataset_id": manifest["dataset_id"], "manifest": os.path.relpath(mpath.absolute(), C.ROOT),
                            "sealed_truth_sha256": manifest["synthetic"]["sealed_truth_sha256"],
                            "generator_commit": gen["tree"]["commit"], "evaluation_commit": tree["commit"],
                            "theta_pilot": theta_pilot, "rho_pilot_kg_m3": truth["rho_star_kg_m3"],
                            "revealed_utc": C.utc()},
                  "chi2_pilot_reduced": chi2_pilot, "chi2_pilot_n_residuals": int(r.size),
                  "chi2_pilot_per_experiment_mean_square": per_exp_chi2,
                  "cal04_limit_chi2_reduced": 1.5*chi2_pilot,
                  "tolerances": {k: {m: v["tol"] for m, v in rows.items()} for k, rows in tol.items()},
                  "components": tol,
                  "frozen": "committed before the formal dataset exists; never changed after it"}
    C.write_new(args.thresholds_out, thresholds)
    C.write_new(out/"pilot_eval_receipt.json",
                {"utc_start": None, "utc_end": C.utc(), "wall_s": time.perf_counter()-t_start, "tree": tree,
                 "protocol_sha256": psha, "env": C.environment(), "gpu_free_mib_at_start": free,
                 "gpu_samples": [C.gpu_sample()], "forward_receipts": fwd.receipts(),
                 "pred_invalid": {k: v["invalid_reason"] for k, v in pred.items()}, "thresholds": thresholds,
                 **C.LABELS})
    log(f"chi2_pilot {chi2_pilot:.4f}; thresholds written to {args.thresholds_out}")


if __name__ == "__main__":
    main()
