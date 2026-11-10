"""R2.1 (protocol v4, ``batched_gpu_v4``): RUBBER-PHYS-07 v3 variants and RUBBER-PHYS-12 in the batched material on
a GPU, N copies of the v3 env, each env evaluated by the frozen v3 criteria (rubber_phys.evaluate_07_v3 /
evaluate_12) from its own per-control-step rows (as rubber_phys.case_stretch_release_v3 / case_guard_regression).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> PYTHONPATH=src \
        .deps/isaacsim-venv/bin/python scripts/rubber/p2_2/run_batched_gpu_07_12.py --out artifacts/rl_ready/p2_2/<id>

The batched path has the v1 solver options only (lumped Dirichlet, particle-velocity traction work), i.e. the v3
inputs; strict mode off (flags recorded).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"scripts"/"rubber"))
sys.path.insert(0, str(ROOT/"scripts"/"rubber"/"p2_2"))


def run_case(case, N, device, P, cfg):
    from run_batched_l0_v3 import schedule
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial, decode_reason
    ramped_case = case == "07_ramped"
    dtc0 = float(cfg["tests"]["RUBBER-PHYS-07" if case.startswith("07") else "RUBBER-PHYS-12"]["setup"]
                 ["control_interval_s"])
    lives = [P._strip_07(cfg, "cpu", dtc0, P.guard_limit(cfg), traction_end=ramped_case)[0] for _ in range(N)]
    ref = lives[0]
    m_ref = ref.scene.marker_reference
    L, t = cfg["tests"]["RUBBER-PHYS-06"]["setup"]["L_m"], cfg["tests"]["RUBBER-PHYS-06"]["setup"]["t_m"]
    T_ax = P.axial_period(L, ref.material)
    length_ref = float(m_ref[8][0]-m_ref[0][0])
    batch = BatchedRubberMaterial(lives, device=device, strict=False, parity_mode=device == "cpu")
    G = batch.G
    dtc, ramped, sched = schedule(case, cfg, P, T_ax)
    rows = [[] for _ in range(N)]
    hold_force = np.zeros(N)
    t0 = time.perf_counter()
    for phase, lam, release, frac in sched:
        kw = {}
        if lam is not None:
            tr = np.zeros((N, G, 3))
            tr[:, 1, 0] = (lam-1)*L
            kw["translation"] = tr
        if release:
            act = np.ones((N, G), np.int32)
            act[:, 1] = 0
            kw["active"] = act
        if frac is not None:
            f = hold_force*frac if frac else np.zeros(N)
            kw["traction"] = np.stack([f, np.zeros(N), np.zeros(N)], 1)
        batch.advance(**kw)
        diag = batch.diagnostics_read()
        mk = batch.markers()
        work = batch.solver.work.numpy()
        inv = batch.invalid_reason.numpy()
        if phase == "stretch":
            hold_force = -batch.wrench.numpy()[:, 1, 0]
        for e in range(N):
            d, m = diag[e], mk[e]
            fin = bool(np.isfinite(m).all())
            rows[e].append({"phase": phase, "length_rel": float((m[8][0]-m[0][0])/length_ref) if fin else None,
                            "lateral_m": float(np.abs(m[:, 1:]-m_ref[:, 1:]).max()) if fin else None,
                            "elastic": d["elastic"], "total": d["kinetic_particle"]+d["elastic"]+d["gravity"]-float(work[e]),
                            "J_min": d["J_min"], "J_max": d["J_max"], "Jbar_min": d["Jbar_min"], "Jbar_max": d["Jbar_max"],
                            "cfl_eff_max": d["cfl_eff_max"], "nonfinite": bool(d["nonfinite"]),
                            "flags": decode_reason(int(inv[e]))})
    wall = time.perf_counter()-t0
    results = []
    n_release = next(i for i, s in enumerate(sched) if s[0] != "stretch")
    for e in range(N):
        r = rows[e]
        if case.startswith("07"):
            rel = r[n_release-1]
            after = r[n_release:]
            post = [x for x in after if x["phase"] == "post"]
            finite = all(np.isfinite(x["total"]) and x["length_rel"] is not None for x in after)
            results.append({"variant": case.split("_")[1], "lambda": None, "release": None,
                            "mean_length_rel_error": float(np.mean([x["length_rel"] for x in post])-1) if finite else None,
                            "post_length_std_rel": float(np.std([x["length_rel"] for x in post])) if finite else None,
                            "energy_drift_rel_psi": (float(max(abs(x["total"]-rel["total"]) for x in after)/rel["elastic"])
                                                     if finite else None),
                            "max_lateral_m": max((x["lateral_m"] for x in r if x["lateral_m"] is not None), default=None),
                            "thickness_m": t, "invalid_flags": r[-1]["flags"],
                            "cfl_eff_max": max(x["cfl_eff_max"] for x in r),
                            "J_range": [min(x["J_min"] for x in r), max(x["J_max"] for x in r)],
                            "Jbar_range": [min(x["Jbar_min"] for x in r), max(x["Jbar_max"] for x in r)],
                            "hold_force_n": float(hold_force[e]), "dt_s": batch.solver.dt,
                            "substeps": batch.solver.substeps, "cfl_actual": None})
        else:
            def first(pred):
                return next((i+1 for i, x in enumerate(r) if pred(x)), None)
            results.append({"control_interval_s": dtc, "release_after_step": n_release,
                            "stable_step_guard": P.guard_limit(cfg),
                            "first_cfl_flag_step": first(lambda x: "RUBBER_CFL_EXCEEDED" in x["flags"]),
                            "first_j_flag_step": first(lambda x: "RUBBER_J_DOMAIN" in x["flags"]),
                            "first_nonfinite_step": first(lambda x: "NONFINITE_STATE" in x["flags"])})
    verdicts = []
    for res in results:
        if case.startswith("07"):
            v, meas = P.evaluate_07_v3([res], {**cfg, "tests": {**cfg["tests"], "RUBBER-PHYS-07": {
                **cfg["tests"]["RUBBER-PHYS-07"], "setup": {**cfg["tests"]["RUBBER-PHYS-07"]["setup"],
                                                            "variants": {res["variant"]: None}}}}})
        else:
            v, meas = P.evaluate_12(res, cfg)
        verdicts.append({"criteria": v, "measured": meas,
                         "verdict": "PASS" if set(v.values()) == {"PASS"} else "FAIL"})
    receipt = {"dt_s": batch.solver.dt, "substeps": batch.solver.substeps, "particles_per_env": ref.scene.n,
               "control_steps": len(sched), "control_interval_s": dtc, "graph": batch.capture_receipt}
    del batch
    return {"case": case, "N": N, "wall_s": wall, "receipt": receipt, "per_env": verdicts,
            "verdict": "PASS" if all(v["verdict"] == "PASS" for v in verdicts) else "FAIL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--cases", default="07_ramped,07_instant,12")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    import warp as wp
    from gpu_smoke_rubber import gpu_sample
    from robotic_cutting_sim import rubber_phys as P
    wp.init()
    cfg = P.load_config(P.CONFIG_V3)
    meta = {"command": " ".join(sys.argv), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": wp.get_device(args.device).name, "pid": os.getpid(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "warp": wp.config.version, "co_tenant_before": gpu_sample(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    results = []
    for case in args.cases.split(","):
        try:
            results.append(run_case(case, args.N, args.device, P, cfg))
        except Exception:
            import traceback
            results.append({"case": case, "error": traceback.format_exc(), "verdict": "FAIL"})
        (out/"results.json").write_text(json.dumps(results, indent=1, default=float))
    meta["co_tenant_after"] = gpu_sample()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others = [a for a in meta["co_tenant_before"]["apps"]+meta["co_tenant_after"]["apps"]
              if a.split(",")[0].strip() != str(os.getpid())]
    meta["gpu_sharing"] = "SHARED_GPU" if others else "EXCLUSIVE"
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps([{"case": r["case"], "verdict": r["verdict"]} for r in results]))


if __name__ == "__main__":
    main()
