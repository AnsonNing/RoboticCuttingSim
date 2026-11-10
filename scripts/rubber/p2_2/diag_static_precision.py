"""DIAGNOSTIC (P2-2 F10; NOT a formal RUBBER-PHYS result): precision of the float32 static state of the -02 cantilever.

Runs one RUBBER-PHYS-02 configuration (protocol v3 inputs: rubber_phys.configs_02, relax_mass_proportional_v1,
static_stop_v3 markers in float64, stable-step guard 1.0) but keeps relaxing (same damping) until --min-periods T1 and
records the float64 tip deflection at every check (T1 / 20). Variables: --precision float32_v1 (v3 state) or
f64_shadow_v1 (F10 option), --cfl (0.5 = v3, 0.25 = dt / 2). Reported: delta / delta_EB at 3 T1 (the v3 stop time)
and at every T1, the drift per T1 over [3, end] T1 (least squares) and the band (max - min) over [3, end] T1.

    PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_2/diag_static_precision.py --out <dir> \
        --label Lt20_t6mm_Wt4_nt3 [--precision f64_shadow_v1] [--cfl 0.25] [--min-periods 10]
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
    ap.add_argument("--label", required=True)
    ap.add_argument("--precision", default="float32_v1")
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--min-periods", type=float, default=10.0)
    ap.add_argument("--dirichlet", default="lumped_v1")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    cfg = P.load_config(P.CONFIG_V3)
    c = next(c for c in P.configs_02(cfg) if c["label"] == args.label)
    live = P.cantilever_live(c["L"], c["W"], c["t"], c["n_t"], c["mu"], c["g"], stable_step_guard=1.0,
                             cfl=args.cfl, state_precision=args.precision, dirichlet=args.dirichlet)
    mat = live.material
    E = mat.young_effective()
    f1 = R.cantilever_f1(mat.density, c["W"], c["t"], c["L"], E)
    T1 = 1/f1
    eb = R.euler_bernoulli_sag(mat.density, c["g"], c["W"], c["t"], c["L"], E)
    t0 = time.perf_counter()
    init = relax_to_static(live, omega1=2*math.pi*f1, stop_rule="static_stop_v3", min_time=args.min_periods*T1,
                           max_time=(args.min_periods+0.5)*T1, keep_history=True)
    z_ref = float(live.scene.marker_reference[-1][2])
    hist = init.pop("history")
    t = np.array([h["t_s"] for h in hist])/T1
    ratio = np.array([(z_ref-h["tip_m"][2])/eb for h in hist])
    psi = np.array([h["elastic_j"] for h in hist])
    late = t >= 3.0-1e-9
    i3 = int(np.argmax(late))
    drift = float(np.polyfit(t[late], ratio[late], 1)[0]) if late.sum() > 2 else None
    res = {"label": "DIAGNOSTIC (P2-2 F10 static precision; not a RUBBER-PHYS result)", "config": c,
           "precision": args.precision, "cfl": args.cfl, "dirichlet": args.dirichlet, "dt_s": live.dt,
           "particles": live.scene.n, "f1_EB_hz": f1, "delta_EB_m": eb,
           "ratio_EB_at_3T1": float(ratio[i3]), "t_at_3T1": float(t[i3]),
           "ratio_EB_per_T1": {f"{k}": float(ratio[int(np.argmin(np.abs(t-k)))]) for k in range(1, int(t[-1])+1)},
           "ratio_EB_end": float(ratio[-1]), "t_end_T1": float(t[-1]),
           "drift_ratio_per_T1_after_3T1": drift, "band_after_3T1": float(ratio[late].max()-ratio[late].min()),
           "psi_rel_change_3T1_to_end": float((psi[-1]-psi[i3])/psi[i3]),
           "reaction_residual_end": hist[-1]["reaction_residual"], "invalid": live.invalid_reason,
           "initializer": {k: init[k] for k in ("converged", "time_s", "substeps", "stop_rule", "cfl")},
           "solver_options": live.solver.solver_receipt()["options"],
           "series": {"t_T1": t[::5].tolist(), "ratio_EB": ratio[::5].tolist()}, "wall_s": time.perf_counter()-t0}
    (out/"diag.json").write_text(json.dumps(res, default=float))
    print(json.dumps({k: res[k] for k in ("precision", "cfl", "ratio_EB_at_3T1", "ratio_EB_end",
                                          "drift_ratio_per_T1_after_3T1", "band_after_3T1", "wall_s")}, default=float))


if __name__ == "__main__":
    main()
