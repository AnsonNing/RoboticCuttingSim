"""P2-5 synthetic truth dataset generator (tasks P5.4 pilot, P5.5 formal; protocol rubber_calib_tests_v1).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> \
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_5/gen_synthetic.py \
        --kind pilot|formal --run-dir artifacts/rl_ready/p2_5/<run> --gpu-index <N>

Draws theta* and rho* from the protocol's prior box with a SECRET seed (os.urandom) and the sensor noise with a second
secret seed, simulates every protocol experiment at the truth fidelity (n_t 6, CFL 0.25) on the GPU (one env per
experiment), samples the sensors, adds noise, writes the sealed truth (``<run>/sealed/truth.json``: theta*, rho*,
both seeds, noise-free channels, oracle) and the dataset (CSVs, held-out oracle, v2 manifest) into
``configs/rubber/calibration_synthetic/<dataset_id>/``. Formal: the manifest is frozen (SYNTHETIC_FROZEN_FOR_HELDOUT).
Pilot: a second noise realisation B goes to ``<run>/pilot_realisation_B/``. theta* is never printed. Requires a clean
tree at a commit that contains the pinned protocol; never overwrites anything. Labels: SIMULATOR_ONLY_UNCALIBRATED,
NUMERICAL_VALIDATION_SYNTHETIC.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PROTOCOL_SHA = "1f6816411d141555973537995300ba87927a8162e8eba5f942d51faf40b2ff2b"     # rubber_calib_tests_v1 (c5054b9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("pilot", "formal"), required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpu-index", type=int, required=True, help="physical GPU index (for the free-memory check)")
    ap.add_argument("--smoke", action="store_true", help="2 control steps per experiment, scratch outputs only")
    ap.add_argument("--device", default="cuda:0", help="cpu only with --smoke (pipeline check)")
    args = ap.parse_args()
    protocol, psha = C.load_protocol(None if args.smoke else PROTOCOL_SHA)
    run = Path(args.run_dir)
    if run.exists():
        raise SystemExit(f"{run} exists (no overwrite)")
    tree = C.tree_state()
    if not args.smoke and not tree["clean"]:
        raise SystemExit("the tree must be clean at a committed state")
    dataset_id = protocol[args.kind]["dataset_id"]+("_smoke" if args.smoke else "")
    ds_dir = (run/"smoke_dataset") if args.smoke else (C.SYNTHETIC_DIR/dataset_id)
    if ds_dir.exists():
        raise SystemExit(f"{ds_dir} exists (no overwrite)")
    if args.device == "cpu" and not args.smoke:
        raise SystemExit("cpu only for --smoke")
    free = C.require_gpu(args.gpu_index) if args.device != "cpu" else None
    run.mkdir(parents=True)
    log = C.Log(run/"log.txt")
    wp = C.setup_warp()
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_synthetic as rcs
    from robotic_cutting_sim import rubber_calibration_v2 as rc2
    t_start = time.perf_counter()
    meta = {"kind": args.kind, "smoke": args.smoke, "utc_start": C.utc(), "tree": tree, "protocol_sha256": psha,
            "env": C.environment(), "module_sha256": C.module_shas(), "gpu_free_mib_at_start": free,
            "gpu_samples": [C.gpu_sample()], **C.LABELS}
    truth_seed, noise_seed = rcs.secret_seed(), rcs.secret_seed()
    theta, rho = rcs.draw_truth(protocol["prior_box"], truth_seed)
    cfg = rce.model_config(protocol["common"], "truth")
    exps = rcs.protocol_experiments(protocol)
    clean, oracle_all, receipts = {}, {}, {}
    for e in exps:
        if args.smoke:
            e = dict(e, loading=dict(e["loading"]))
        t0 = time.perf_counter()
        b = rce.ExperimentBatch(e, rho, cfg, 1, device=args.device, parity_mode=args.device == "cpu")
        if args.smoke:
            b.schedule.K = 2
        rec = rce.finalize_record(b.run([rce.material_from_theta(theta, rho, cfg.beta)])[0])
        obs = rce.observables(e, rec, cfg)
        ch = rce.sample_channels(e, obs, rcs.sensors_for(protocol, e["type"]))
        clean[e["id"]] = ch
        oracle_all[e["id"]] = rce.oracle(obs)
        receipts[e["id"]] = {**b.receipt, "invalid_reason": int(rec["invalid_reason"]),
                             **({"translation_check": rce.translation_check(obs)} if e["type"] == "friction_sliding"
                                else {}),
                             "wall_s": time.perf_counter()-t0, "graph": b.batch.capture_receipt,
                             "memory_bytes": int(b.batch.memory_bytes())}
        log(f"truth {e['id']}: particles {b.receipt['particles_per_env']} substeps {b.receipt['substeps']} "
            f"steps {b.schedule.K} invalid {rec['invalid_reason']} wall {receipts[e['id']]['wall_s']:.1f}s")
        del b
        gc.collect()
        meta["gpu_samples"].append(C.gpu_sample())
    invalid = {k: v["invalid_reason"] for k, v in receipts.items() if v["invalid_reason"]}
    rng = np.random.default_rng(noise_seed)
    specimens = rcs.measured_specimens(protocol, rho, rng)
    noisy_a = rcs.noisy_channels(protocol, exps, clean, rng)
    noisy_b = rcs.noisy_channels(protocol, exps, clean, rng) if args.kind == "pilot" else None
    sealed_payload = {"schema": "rcs_rubber_sealed_truth_v1", "dataset_id": dataset_id, "theta_star": theta,
                      "rho_star_kg_m3": rho, "truth_seed": str(truth_seed), "noise_seed": str(noise_seed),
                      "truth_profile": {"n_t": cfg.n_t, "cfl": cfg.cfl}, "clean_channels":
                      {k: rcs.jsonable_channels(v) for k, v in clean.items()}, "oracle": oracle_all, **C.LABELS}
    sealed_path, sealed_sha = rcs.seal(run/"sealed", C.jsonable(sealed_payload))
    held = {e["id"] for e in exps if e["split"] == "heldout"}
    syn = {"generator_commit": tree["commit"], "generator_tree_clean": tree["clean"] or args.smoke,
           "truth_profile": {"n_t": cfg.n_t, "cfl": cfg.cfl, "resolution_profile": cfg.resolution_profile,
                             "options": dict(rce.DEFAULT_OPTIONS), "visco_energy": cfg.visco_energy,
                             "contact_law": cfg.contact_law, "stable_step_guard": cfg.stable_step_guard,
                             "control_dt_s": cfg.control_dt, "step_material": {"mu0_max_pa": cfg.mu0_max,
                                                                               "density_min_kg_m3": cfg.density_min}},
           "sensor_model_id": protocol["sensors"]["model_id"], "sealed_truth_sha256": sealed_sha,
           "prior_box": protocol["prior_box"], "noise_seed_sha256": rcs.seed_sha256(noise_seed)}
    note = ("SYNTHETIC simulator truth (n_t 6, CFL 0.25) with seeded sensor noise; hidden parameters sealed "
            f"(sha256 {sealed_sha}); NUMERICAL_VALIDATION_SYNTHETIC, never a calibration; protocol "
            f"rubber_calib_tests_v1 sha256 {psha}")
    mpath = rcs.write_dataset(ds_dir, protocol=protocol, protocol_sha256=psha, experiments=exps, noisy=noisy_a,
                              specimens=specimens, oracle={k: v for k, v in oracle_all.items() if k in held},
                              dataset_id=dataset_id, status="SYNTHETIC_PILOT", synthetic=syn, note=note)
    if args.kind == "formal":
        frozen = rc2.freeze(json.loads(mpath.read_text()), ds_dir, C.utc())
        mpath.unlink()
        rc2.write_manifest(mpath, frozen)
        rc2.load_manifest(mpath)
    else:
        bdir = run/"pilot_realisation_B"
        bdir.mkdir()
        for e in exps:
            for role, cols in noisy_b[e["id"]].items():
                rce.write_channels(bdir/rcs.file_name(e["id"], role), cols)
    manifest = json.loads(mpath.read_text())
    meta.update({"utc_end": C.utc(), "wall_s": time.perf_counter()-t_start, "dataset_id": dataset_id,
                 "dataset_dir": str(ds_dir), "manifest_status": manifest["status"],
                 "manifest_freeze_sha256": manifest["splits"]["sha256"], "sealed_truth_path": str(sealed_path),
                 "sealed_truth_sha256": sealed_sha, "noise_seed_sha256": rcs.seed_sha256(noise_seed),
                 "secret_seeds": "written only to the sealed file", "truth_receipts": receipts,
                 "truth_invalid": invalid, "verdict_truth_valid": "PASS" if not invalid else "FAIL",
                 "warp_device": str(wp.get_device(args.device))})
    meta["gpu_samples"].append(C.gpu_sample())
    C.write_new(run/"gen_receipt.json", meta)
    log(f"done: dataset {ds_dir} status {manifest['status']} sealed sha {sealed_sha[:12]}… invalid {invalid}")
    if invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
