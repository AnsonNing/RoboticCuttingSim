"""DIAGNOSTIC (P2-2 F8; NOT a formal RUBBER-PHYS result): energy loss of a released strip at a Dirichlet clamp.

The RUBBER-PHYS-07 strip (60 x 10 x 4 mm, mu 0.4 MPa, rho 1150, beta 20, case C end-face grips, gravity 0) is
stretched by grip 1 along the cosine path to --lam over 60 T_ax (control interval --prep-interval), held 10 T_ax, then
grip 1 is removed instantaneously (with --free-free grip 0 as well) and the strip runs --post-periods T_ax. Sampled
every --sample-interval: E_total = KE_particle + Psi - W_ext relative to the release, the end-marker length and the
lateral marker excursion. Reported: max |dE| / Psi(release), the mean dE / Psi over the last quarter of the window and
the least-squares slope of dE / Psi per axial period (secular loss).

    PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_2/diag_clamp_dissipation.py --out <dir> \
        [--n-t 4] [--mask x|xyz] [--cfl 0.5] [--dirichlet lumped_v1|consistent_mass_projection_v1] [--free-free]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))


def run(args):
    from robotic_cutting_sim import rubber_phys as P
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    cfg = P.load_config(P.CONFIG_V3)
    st = cfg["tests"]["RUBBER-PHYS-06"]["setup"]
    L, W, t = st["L_m"], st["W_m"], st["t_m"]
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=args.n_t, case="C", end_mask=args.mask)
    live = LiveRubberMaterial(spec, material=P.material(st["mu_pa"]), control_dt=args.prep_interval,
                              gravity=(0, 0, 0), strict=False, cfl=args.cfl, stable_step_guard=1.0,
                              dirichlet=args.dirichlet)
    s, mat = live.solver, live.material
    T_ax = P.axial_period(L, mat)
    dtc = live.control_dt
    n_r, n_h = int(math.ceil(60*T_ax/dtc-1e-9)), int(math.ceil(10*T_ax/dtc-1e-9))
    t0 = time.perf_counter()
    for k in range(n_r+n_h):
        lam = P.cosine_path((k+1)*dtc, n_r*dtc, args.lam)
        live.advance({1: {"translation": np.array([(lam-1)*L, 0.0, 0.0])}})
    prep_wall = time.perf_counter()-t0
    m_ref = live.scene.marker_reference
    length_ref = float(m_ref[8][0]-m_ref[0][0])
    per_sample = max(1, int(round(args.sample_interval/live.dt)))
    samples = int(math.ceil(args.post_periods*T_ax/(per_sample*live.dt)))

    def energy():
        d = s.diagnostics.read()[0]
        return d, d["kinetic_particle"]+d["elastic"]+d["gravity"]-float(s.work.numpy()[0])
    d0, e0 = energy()
    psi0 = d0["elastic"]
    s.begin_control_step()
    s.set_group_motion(1, active=False)
    if args.free_free:
        s.set_group_motion(0, active=False)
    rows = []
    t1 = time.perf_counter()
    for k in range(samples):
        for _ in range(per_sample):
            s.step()
        d, e = energy()
        mk = s.diagnostics.marker_x.numpy()
        rows.append({"t_ms": (k+1)*per_sample*live.dt*1e3, "dE_rel_psi": (e-e0)/psi0,
                     "kinetic_particle": d["kinetic_particle"], "kinetic_grid": d["kinetic_grid"], "elastic": d["elastic"],
                     "length_rel": float((mk[8][0]-mk[0][0])/length_ref),
                     "lateral_m": float(np.abs(mk[:, 1:]-m_ref[:, 1:]).max()), "cfl_eff_max": d["cfl_eff_max"],
                     "nonfinite": bool(d["nonfinite"])})
        if d["nonfinite"]:
            break
    tt = np.array([r["t_ms"] for r in rows])*1e-3/T_ax
    de = np.array([r["dE_rel_psi"] for r in rows])
    q = tt >= 0.75*tt[-1]
    slope = float(np.polyfit(tt, de, 1)[0]) if len(rows) > 2 else None
    res = {"label": "DIAGNOSTIC (P2-2 F8 clamp dissipation; not a RUBBER-PHYS result)", "args": vars(args),
           "n_t": args.n_t, "mask": args.mask, "cfl": args.cfl, "dirichlet": args.dirichlet, "free_free": args.free_free,
           "lambda": args.lam, "dt_s": live.dt, "particles": live.scene.n, "T_ax_s": T_ax, "psi_release_j": psi0,
           "e_total_release_j": e0, "max_abs_dE_rel_psi": float(np.abs(de).max()),
           "mean_dE_rel_psi_last_quarter": float(de[q].mean()), "slope_dE_rel_psi_per_T_ax": slope,
           "mean_length_rel_error": float(np.mean([r["length_rel"] for r in rows])-1),
           "max_lateral_m": float(max(r["lateral_m"] for r in rows)),
           "cfl_eff_max": float(max(r["cfl_eff_max"] for r in rows)), "invalid": live.invalid_reason,
           "guard_violations": int(s.cfl_violations.numpy()[0]), "j_violations": int(s.violations.numpy()[0]),
           "solver_options": s.solver_receipt()["options"], "prep_wall_s": prep_wall,
           "post_wall_s": time.perf_counter()-t1, "rows": rows[::max(1, len(rows)//400)]}
    out = Path(args.out)
    (out/"diag.json").write_text(json.dumps(res, default=float))
    print(json.dumps({k: res[k] for k in ("n_t", "mask", "cfl", "dirichlet", "free_free", "max_abs_dE_rel_psi",
                                          "mean_dE_rel_psi_last_quarter", "slope_dE_rel_psi_per_T_ax",
                                          "mean_length_rel_error", "max_lateral_m", "invalid")}, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-t", type=int, default=4)
    ap.add_argument("--mask", default="x", choices=("x", "xyz"))
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--dirichlet", default="lumped_v1")
    ap.add_argument("--free-free", action="store_true")
    ap.add_argument("--lam", type=float, default=1.03)
    ap.add_argument("--post-periods", type=float, default=20.0)
    ap.add_argument("--prep-interval", type=float, default=0.005)
    ap.add_argument("--sample-interval", type=float, default=1e-4)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    run(args)


if __name__ == "__main__":
    main()
