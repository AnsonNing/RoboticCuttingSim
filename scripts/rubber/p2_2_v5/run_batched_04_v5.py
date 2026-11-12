"""RUBBER-PHYS-04 batched part under protocol v5: heterogeneous CPU batch N = 4 vs four single-env runs, n_t 4,
stable-step guard on, byte-wise L0 (ADR 0023 §20.4, §19A).

    TMPDIR=/mnt/data/ningan/tmp CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .deps/isaacsim-venv/bin/python \
        scripts/rubber/p2_2_v5/run_batched_04_v5.py --out artifacts/rl_ready/p2_2_v5/<id> \
        --formal-dt artifacts/rl_ready/p2_2_v5/<rubber_phys_id>/raw/04/dt.json

Env 0 is the formal "dt" run (L 60 mm, W = t = 6 mm, mu 2 MPa, rho 1150); envs 1-3: L 48 mm / W 12 mm / mu 1 MPa and
rho 1000 (``gpu_smoke_rubber.batched_04_envs``); uniform dx and dt (c_p,max = env 0's), n_t and the guard limit from
``configs/rubber/rubber_phys_tests_v5.json``. After every 1 ms control step every particle / node / group / traction
array, the cell CSR, work, J and CFL violation counters, the diagnostics row, the markers and the clamp wrench of every
env are compared byte-wise (dtype, shape, raw bytes) with the env's own single run; env 0's tip series is also compared
with the formal single dt run's. Writes <out>/batched_04.json (the runner's ``--batched-04``).
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/"configs"/"rubber"/"rubber_phys_tests_v5.json"


def _same_bytes(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--formal-dt")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT/"src"))
    import warp as wp
    from robotic_cutting_sim import rubber_phys as P
    from robotic_cutting_sim import rubber_reference as R
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.batched.rubber_layout import GROUP_ARRAYS, NODE_ARRAYS, PARTICLE_ARRAYS, TRACTION_ARRAYS
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    sys.path.insert(0, str(ROOT/"scripts"/"rubber"))
    from gpu_smoke_rubber import batched_04_envs
    cfg = P.load_config(CONFIG)
    st = cfg["tests"]["RUBBER-PHYS-04"]["setup"]
    guard = P.test_guard(cfg, "RUBBER-PHYS-04")
    assert guard is not None and cfg["version"] == 5
    n_t, t = int(st["n_t"]), float(st["t_m"])
    envs = batched_04_envs()
    step_mat = P.material(st["mu_pa"])
    meta = {"command": " ".join(sys.argv), "python": sys.executable, "pid": os.getpid(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "tree_dirty": bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs", "tests"],
                                              cwd=ROOT, capture_output=True, text=True).stdout.strip()),
            "config": str(CONFIG.relative_to(ROOT)), "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "warp": wp.config.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED"}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))

    def make():
        return [LiveRubberMaterial(rs.RubberStripSpec(length=kw["L"], width=kw["W"], thickness=t, n_t=n_t, case="A"),
                                   material=P.material(kw["mu"], kw["rho"]), control_dt=st["record_interval_s"],
                                   gravity=(0, 0, -st["gravity_m_s2"]), step_material=step_mat, record=True,
                                   stable_step_guard=guard, strict=False) for kw in envs]
    singles = make()
    batch = BatchedRubberMaterial(make(), device="cpu", parity_mode=True, strict=False)
    f_eb = R.cantilever_f1(1150.0, st["W_m"], t, st["L_m"], P.material(st["mu_pa"]).young_effective())
    steps = int(math.ceil(6/f_eb/st["record_interval_s"]))
    mismatches, first, tips = 0, None, []
    K = batch.K
    start = time.perf_counter()
    for k in range(steps):
        batch.advance()
        mk = batch.markers()
        tips.append(float(mk[0][-1][2]-singles[0].scene.marker_reference[-1][2]))
        for e, live in enumerate(singles):
            live.advance()
            snap = batch.solver.snapshot(e)
            s = live.solver
            G, Gt = s.G, s.Gt
            names = [(n, snap[n], getattr(s, n).numpy()) for n in PARTICLE_ARRAYS+NODE_ARRAYS
                     + ("cell_node", "cell_offsets", "cell_particles", "index")]
            names += [(n, snap[n], getattr(s, n).numpy()[:G]) for n in GROUP_ARRAYS]
            names += [(n, snap[n], getattr(s, n).numpy()[:Gt]) for n in TRACTION_ARRAYS]
            names += [("work", snap["work"], s.work.numpy()), ("violations", snap["violations"], s.violations.numpy()),
                      ("cfl_violations", snap["cfl_violations"], s.cfl_violations.numpy()),
                      ("summary", batch.summary.numpy()[e], s.diagnostics.summary.numpy()[0]),
                      ("markers", batch.marker_x.numpy()[e*K:(e+1)*K], s.diagnostics.marker_x.numpy()),
                      ("wrench", batch.wrench.numpy()[e, :G], live.last_wrench)]
            for name, a, b in names:
                if not _same_bytes(a, b):
                    mismatches += 1
                    first = first or {"step": k+1, "env": e, "array": name}
    wall = time.perf_counter()-start
    batch_invalid = batch.invalid_reason.numpy().tolist()
    single_invalid = [int(live.invalid_reason) for live in singles]
    result = {"test": "RUBBER-PHYS-04 batched (protocol v5)", "comparison": "byte-wise (dtype, shape, raw bytes)",
              "N": 4, "envs": envs, "n_t": n_t, "steps": steps, "control_dt_s": st["record_interval_s"],
              "stable_step_guard": guard, "substeps": batch.solver.substeps, "dt_s": batch.solver.dt,
              "particles": batch.packed.particle_counts.tolist(), "array_mismatches": mismatches,
              "first_mismatch": first, "bytewise_equal": mismatches == 0, "wall_s": wall, "device": "cpu",
              "single_dt_s": singles[0].dt, "invalid_reason_batch": batch_invalid,
              "invalid_reason_singles": single_invalid,
              "invalid_reason_all_zero": not any(batch_invalid) and not any(single_invalid),
              "cfl_violations_batch": batch.solver.cfl_violations.numpy().tolist()}
    if args.formal_dt:
        formal = json.loads(Path(args.formal_dt).read_text())
        result["formal_dt_run"] = args.formal_dt
        result["formal_dt_run_tip_equal"] = bool(_same_bytes(np.array(formal["tip_z_m"]), np.array(tips)))
        result["formal_dt_run_dt_equal"] = formal["dt_s"] == batch.solver.dt
    result["bitwise_equal"] = bool(result["bytewise_equal"] and result.get("formal_dt_run_tip_equal", False)
                                   and result.get("formal_dt_run_dt_equal", False))
    metrics = []
    for e, live in enumerate(singles):
        recs = live.records[1:]
        tip = np.array([r["markers_m"][-1][2] for r in recs])-live.scene.marker_reference[-1][2]
        m = P.vibration_metrics(tip, np.array([r["energies_j"]["total"] for r in recs]),
                                np.array([r["energies_j"]["elastic"] for r in recs]), st["record_interval_s"])
        kw = envs[e]
        m["f_EB_hz"] = R.cantilever_f1(kw["rho"], kw["W"], t, kw["L"], P.material(kw["mu"]).young_effective())
        metrics.append({k: v for k, v in m.items() if k != "upward_crossings_s"})
    result["per_env_vibration_metrics"] = metrics
    (out/"batched_04.json").write_text(json.dumps(result, indent=1, default=float))
    meta.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=wall)
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: result.get(k) for k in ("bitwise_equal", "bytewise_equal", "array_mismatches", "first_mismatch",
                                                  "formal_dt_run_tip_equal", "invalid_reason_all_zero", "wall_s")}))


if __name__ == "__main__":
    main()
