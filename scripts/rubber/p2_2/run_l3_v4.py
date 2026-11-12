"""RUBBER-PHYS-10 L3 under protocol v4 (rule ``rubber_l3_band_v4``, F3 of ADR 0023 §18.8), GPU.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> PYTHONPATH=src \
        .deps/isaacsim-venv/bin/python scripts/rubber/p2_2/run_l3_v4.py --out artifacts/rl_ready/p2_2/<id>

One process, one device. The setup is read from ``configs/rubber/rubber_phys_tests_v4.json``
(``tests.RUBBER-PHYS-10.setup.L3_v4``): the RUBBER-PHYS-04 heterogeneous envs, the stable-step guard, the number of
single-env GPU reference repeats (each a fresh ``LiveRubberMaterial``), the scored batched run(s) and the held-out
single repeats (classified by the same rule, reported, not scored). Order: references, batched, held-out. Every series
and the evaluation (``rubber_l3.evaluate``) are stored; the output directory is never overwritten.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"scripts"/"rubber"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    import warp as wp
    from gpu_smoke_rubber import gpu_sample, Peak
    from robotic_cutting_sim import rubber_l3, rubber_phys as P
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    wp.init()
    cfg_path = ROOT/"configs"/"rubber"/"rubber_phys_tests_v4.json"
    cfg = json.loads(cfg_path.read_text())
    st = cfg["tests"]["RUBBER-PHYS-10"]["setup"]["L3_v4"]
    envs, steps, dtc = st["envs"], int(st["control_steps"]), float(st["control_dt_s"])
    guard = float(st["stable_step_guard"])
    step_mat = P.material(st["step_material_mu_pa"])
    meta = {"command": " ".join(sys.argv), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device, "device_name": wp.get_device(args.device).name, "pid": os.getpid(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "dirty": bool(subprocess.run(["git", "status", "--porcelain", "src", "configs", "scripts"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip()),
            "config": str(cfg_path.relative_to(ROOT)), "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
            "warp": wp.config.version, "co_tenant_before": gpu_sample(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    peak = Peak()

    def make(kw, dev):
        spec = rs.RubberStripSpec(length=kw["L"], width=kw["W"], thickness=st["t_m"], n_t=st["n_t"], case="A")
        return LiveRubberMaterial(spec, material=P.material(kw["mu"], kw["rho"]), control_dt=dtc,
                                  gravity=(0, 0, -st["gravity_m_s2"]), device=dev, step_material=step_mat,
                                  stable_step_guard=guard, strict=False)

    def single_run():
        series, flags = [], []
        for kw in envs:
            live = make(kw, args.device)
            s = []
            for _ in range(steps):
                live.advance()
                d = live.last_diagnostics
                s.append([float(live.markers()[-1][2]), d["Jbar_min"], d["Jbar_max"], d["elastic"]])
            series.append(s)
            flags.append(int(live.invalid_reason))
            peak.update()
            del live
        return np.array(series), flags                    # [E, steps, 4]

    def batched_run():
        batch = BatchedRubberMaterial([make(kw, "cpu") for kw in envs], device=args.device, strict=False)
        s = []
        for _ in range(steps):
            batch.advance()
            diag = batch.diagnostics_read()
            mk = batch.markers()
            s.append([[float(mk[e][-1][2]), diag[e]["Jbar_min"], diag[e]["Jbar_max"], diag[e]["elastic"]]
                      for e in range(len(envs))])
        peak.update()
        flags = batch.invalid_reason.numpy().tolist()
        receipt = {"dt_s": batch.solver.dt, "substeps": batch.solver.substeps,
                   "particles": batch.packed.particle_counts.tolist(), "guard": batch.solver.cfl_guard_limit}
        del batch
        return np.transpose(np.array(s), (1, 0, 2)), flags, receipt

    t0 = time.perf_counter()
    refs, ref_flags = [], []
    for _ in range(int(st["reference_repeats"])):
        a, f = single_run()
        refs.append(a)
        ref_flags.append(f)
    t_ref = time.perf_counter()-t0
    batched, batched_flags, receipts = [], [], []
    for _ in range(int(st["batched_runs"])):
        a, f, rc = batched_run()
        batched.append(a)
        batched_flags.append(f)
        receipts.append(rc)
    held, held_flags = [], []
    for _ in range(int(st["held_out_single_repeats"])):
        a, f = single_run()
        held.append(a)
        held_flags.append(f)
    wall = time.perf_counter()-t0
    obs = st["observables"]
    names = [f"env{e}_{o}" for e in range(len(envs)) for o in obs]
    R = np.array(refs)                                    # [n, E, steps, 4]
    ref = {f"env{e}_{o}": R[:, e, :, j] for e in range(len(envs)) for j, o in enumerate(obs)}
    cands = {}
    for b, a in enumerate(batched):
        cands[f"batched_{b}"] = {f"env{e}_{o}": a[e, :, j] for e in range(len(envs)) for j, o in enumerate(obs)}
    for h, a in enumerate(held):
        cands[f"heldout_single_{h}"] = {f"env{e}_{o}": a[e, :, j] for e in range(len(envs)) for j, o in enumerate(obs)}
    ev = rubber_l3.evaluate(ref, cands, alpha_family=float(st["alpha_family"]), names=names)
    scored = [k for k in cands if k.startswith("batched_")]
    any_flag = any(any(f) for f in ref_flags+held_flags+batched_flags)
    verdict = "PASS" if all(ev["runs"][k]["verdict"] == "PASS" for k in scored) and not any_flag else "FAIL"
    meta["co_tenant_after"] = gpu_sample()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others = [a for a in meta["co_tenant_before"]["apps"]+meta["co_tenant_after"]["apps"]
              if a.split(",")[0].strip() != str(os.getpid())]
    meta["gpu_sharing"] = "SHARED_GPU" if others else "EXCLUSIVE"
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    res = {"test": "RUBBER-PHYS-10 L3 (protocol v4)", "rule": ev["rule"], "verdict": verdict, "scored_runs": scored,
           "heldout_outside_runs": sum(ev["runs"][k]["verdict"] == "FAIL" for k in cands if k.startswith("heldout")),
           "heldout_runs": len(held), "invalid_flags_any": any_flag, "batched_receipts": receipts,
           "reference_wall_s": t_ref, "wall_s": wall, "vram_peak_mb_process": peak.value, "evaluation": ev,
           "series": {"reference": R.tolist(), "batched": [a.tolist() for a in batched],
                      "heldout": [a.tolist() for a in held]}, "observables": obs, "envs": envs}
    (out/"l3.json").write_text(json.dumps(res, default=float))
    print(json.dumps({"verdict": verdict, "runs": ev["runs"], "sharing": meta["gpu_sharing"], "wall_s": wall,
                      "vram_peak_mb": peak.value}, indent=1, default=float))


if __name__ == "__main__":
    main()
