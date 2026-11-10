"""RUBBER-PHYS-04 batched part (rubber_phys_tests_v2): heterogeneous CPU batch N = 4 vs four single-env runs.

    PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/run_batched_04.py \
        --out artifacts/rl_ready/p2_1/batched_04_001 [--formal-dt artifacts/rl_ready/p2_1/rubber_phys_001/raw/04/dt.json]

Env 0 is the formal "dt" run (L 60 mm, W = t = 6 mm, mu 2 MPa, rho 1150); envs 1-3: L 48 mm / W 12 mm /
mu 1 MPa and rho 1000; uniform dx and dt (c_p,max = env 0's). After every 1 ms control step every array of every env,
the diagnostics row, the markers and the clamp wrench are compared bitwise with the env's own single run; env 0's tip
series is also compared with the formal single run's.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--formal-dt")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    from robotic_cutting_sim import rubber_phys as P
    from robotic_cutting_sim import rubber_reference as R
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.batched.rubber_layout import GROUP_ARRAYS, NODE_ARRAYS, PARTICLE_ARRAYS
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    sys.path.insert(0, str(ROOT/"scripts"/"rubber"))
    from gpu_smoke_rubber import batched_04_envs
    envs = batched_04_envs()
    step_mat = P.material(2e6)

    def make():
        return [LiveRubberMaterial(rs.RubberStripSpec(length=kw["L"], width=kw["W"], thickness=0.006, n_t=3, case="A"),
                                   material=P.material(kw["mu"], kw["rho"]), control_dt=1e-3, gravity=(0, 0, -9.81),
                                   step_material=step_mat, record=True) for kw in envs]
    singles = make()
    batch = BatchedRubberMaterial(make(), device="cpu", parity_mode=True)
    f_eb = R.cantilever_f1(1150.0, 0.006, 0.006, 0.06, P.material(2e6).young_effective())
    steps = int(math.ceil(6/f_eb/1e-3))
    mismatches, first = 0, None
    tips = []
    start = time.perf_counter()
    for k in range(steps):
        batch.advance()
        mk = batch.markers()
        tips.append(float(mk[0][-1][2]-singles[0].scene.marker_reference[-1][2]))
        for e, live in enumerate(singles):
            live.advance()
            snap = batch.solver.snapshot(e)
            s = live.solver
            names = [(n, snap[n], getattr(s, n).numpy()) for n in PARTICLE_ARRAYS+NODE_ARRAYS]
            names += [(n, snap[n], getattr(s, n).numpy()[:s.G]) for n in GROUP_ARRAYS]
            names += [("summary", batch.summary.numpy()[e], s.diagnostics.summary.numpy()[0]),
                      ("markers", mk[e], s.diagnostics.marker_x.numpy()),
                      ("wrench", batch.wrench.numpy()[e, :s.G], live.last_wrench),
                      ("work", snap["work"], s.work.numpy())]
            for name, a, b in names:
                if not np.array_equal(a, b):
                    mismatches += 1
                    first = first or {"step": k+1, "env": e, "array": name}
    wall = time.perf_counter()-start
    result = {"test": "RUBBER-PHYS-04 batched", "N": 4, "envs": envs, "steps": steps, "control_dt_s": 1e-3,
              "substeps": batch.solver.substeps, "dt_s": batch.solver.dt,
              "particles": batch.packed.particle_counts.tolist(), "array_mismatches": mismatches,
              "first_mismatch": first, "bitwise_equal": mismatches == 0, "wall_s": wall, "device": "cpu",
              "single_dt_s": singles[0].dt}
    if args.formal_dt:
        formal = json.loads(Path(args.formal_dt).read_text())
        result["formal_dt_run_tip_equal"] = bool(np.array_equal(np.array(formal["tip_z_m"]), np.array(tips)))
        result["formal_dt_run_dt_equal"] = formal["dt_s"] == batch.solver.dt
        result["bitwise_equal"] = result["bitwise_equal"] and result["formal_dt_run_tip_equal"]
    metrics = []
    for e, live in enumerate(singles):
        recs = live.records[1:]
        tip = np.array([r["markers_m"][-1][2] for r in recs])-live.scene.marker_reference[-1][2]
        m = P.vibration_metrics(tip, np.array([r["energies_j"]["total"] for r in recs]),
                                np.array([r["energies_j"]["elastic"] for r in recs]), 1e-3)
        kw = envs[e]
        m["f_EB_hz"] = R.cantilever_f1(kw["rho"], kw["W"], 0.006, kw["L"], P.material(kw["mu"]).young_effective())
        metrics.append({k: v for k, v in m.items() if k != "upward_crossings_s"})
    result["per_env_vibration_metrics"] = metrics
    (out/"batched_04.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: result[k] for k in ("bitwise_equal", "array_mismatches", "first_mismatch", "wall_s")}
                     | {"formal": result.get("formal_dt_run_tip_equal")}))


if __name__ == "__main__":
    main()
