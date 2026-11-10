"""DIAGNOSTIC (P2-1 follow-up, item 2): is the static_stop_v2 marker-stationarity test a convergence test?

Re-runs one RUBBER-PHYS-02 configuration exactly as the formal case (rubber_phys.case_static_sag inputs, initializer
relax_mass_proportional_v1 with static_stop_v2), but additionally accumulates the MUSL grid displacement
u_I = sum dt v_I in float64 after every substep (measurement only; the solver is unchanged) and evaluates a
rounding-free tip deflection from x64_p = X_p + sum_c w_pc u_I. After the stop, the damped relaxation continues for
--extra T1 and the x-based (float32 positions, as the formal result) and the float64 deflections, the elastic energy
and the marker change per check are recorded.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_1_followup/diag_stop_rule.py --out <dir> [--label Lt10_t3mm_Wt1_nt3]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))


def main():
    from robotic_cutting_sim import rubber_phys as P
    from robotic_cutting_sim import rubber_reference as R
    from robotic_cutting_sim.rubber_initializer import relax_to_static
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="Lt10_t3mm_Wt1_nt3")
    ap.add_argument("--extra", type=float, default=10.0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    cfg = P.load_config()
    c = next(c for c in P.configs_02(cfg) if c["label"] == args.label)
    live = P.cantilever_live(c["L"], c["W"], c["t"], c["n_t"], c["mu"], c["g"])
    s, sc = live.solver, live.scene
    mat = live.material
    E = mat.young_effective()
    f1 = R.cantilever_f1(mat.density, c["W"], c["t"], c["L"], E)
    w1 = 2*math.pi*f1
    eb = R.euler_bernoulli_sag(mat.density, c["g"], c["W"], c["t"], c["L"], E)
    u = np.zeros((s.node_count, 3))
    orig = s.step

    def step():
        orig()
        u[:] += s.dt*s.new_v.numpy().astype(np.float64)
    s.step = step
    tip = sc.tip_particles
    wts = sc.weights[tip].astype(np.float64)
    nds = sc.nodes[tip]
    X = sc.X.astype(np.float64)

    def measure():
        x = s.x.numpy().astype(np.float64)
        x64 = X[tip]+np.einsum("pc,pcj->pj", wts, u[nds])
        d = s.diagnostics.read()[0]
        return {"delta_x_f32": float(sc.marker_reference[-1][2]-x[tip, 2].mean()),
                "delta_u_f64": float(sc.marker_reference[-1][2]-x64[:, 2].mean()),
                "x_minus_x64_max_m": float(np.abs(x[tip]-x64).max()), "elastic_j": d["elastic"],
                "kinetic_grid_j": d["kinetic_grid"], "max_grid_speed": d["max_grid_speed_m_s"]}
    t0 = time.perf_counter()
    init = relax_to_static(live, omega1=w1)
    at_stop = measure()
    at_stop["t_s"] = init["time_s"]
    # continue the labelled damped relaxation for --extra T1 (same rate), checks every T1/20 as the stop rule
    s.set_initializer_damping(2*w1, initializer=True)
    chunk = max(1, int(round(1/f1/20/s.dt)))
    rows, prev = [], live.markers().copy()
    for k in range(int(round(args.extra*20))):
        for _ in range(chunk):
            s.step()
        mk = live.markers()
        if (k+1) % 20 == 0:
            r = measure()
            r.update(t_s=init["time_s"]+(k+1)*chunk*s.dt, marker_change_m=float(np.abs(mk-prev).max()))
            rows.append(r)
        prev = mk.copy()
    s.set_initializer_damping(0.0)
    final = rows[-1]
    res = {"label": "DIAGNOSTIC (not a RUBBER-PHYS result)", "config": c, "f1_hz": f1, "delta_EB_m": eb,
           "initializer": {k: init[k] for k in ("converged", "time_s", "substeps", "final")},
           "at_stop": at_stop, "after_extra": rows,
           "ratio_EB_at_stop_x_f32": at_stop["delta_x_f32"]/eb, "ratio_EB_at_stop_u_f64": at_stop["delta_u_f64"]/eb,
           "ratio_EB_final_x_f32": final["delta_x_f32"]/eb, "ratio_EB_final_u_f64": final["delta_u_f64"]/eb,
           "elastic_rel_change_after_stop": (final["elastic_j"]-at_stop["elastic_j"])/final["elastic_j"],
           "wall_s": time.perf_counter()-t0}
    (out/"stop_rule.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("ratio", "elastic", "initializer"))}, default=float))


if __name__ == "__main__":
    main()
