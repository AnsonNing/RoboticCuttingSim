"""DIAGNOSTIC (not the formal RUBBER-PHYS-07 result): what happens after the grip release of the stretch-release test.

    PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/diag_stretch_release.py --out <dir> --substep-factor 1|2

Same protocol as RUBBER-PHYS-07 (rubber_phys_tests_v2) but strict=False, so the run continues after a flag, and a
record per control step (J, Jbar, speeds, energies, length). ``--substep-factor 2`` halves dt (CFL 0.25) to test whether
the failure is the compressive stiffening of the explicit step (NH axial tangent mu (1 + 2 lambda^-3), 5.7x at
lambda = 0.5). The formal verdict stays the one of the frozen run.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--substep-factor", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    from robotic_cutting_sim import rubber_phys as P
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    cfg = P.load_config()
    st = cfg["tests"]["RUBBER-PHYS-06"]["setup"]
    spec = rs.RubberStripSpec(length=st["L_m"], width=st["W_m"], thickness=st["t_m"], n_t=st["n_t"], case="C",
                              end_mask="x")
    base = LiveRubberMaterial(spec, material=P.material(st["mu_pa"]), control_dt=st["control_interval_s"],
                              gravity=(0, 0, 0))
    live = LiveRubberMaterial(spec, material=P.material(st["mu_pa"]), control_dt=st["control_interval_s"],
                              gravity=(0, 0, 0), substeps=args.substep_factor*base.substeps, strict=False, record=True)
    del base
    L, dtc = st["L_m"], live.control_dt
    T_ax = P.axial_period(L, live.material)
    n_r, n_h = int(math.ceil(60*T_ax/dtc-1e-9)), int(math.ceil(10*T_ax/dtc-1e-9))
    n_post = int(math.ceil(20*T_ax/dtc-1e-9))
    start = time.perf_counter()
    for k in range(n_r+n_h):
        lam = P.cosine_path((k+1)*dtc, n_r*dtc, 1.5)
        live.advance({1: {"translation": np.array([(lam-1)*L, 0.0, 0.0])}})
    for k in range(n_post):
        live.advance({1: {"active": False}} if k == 0 else None)
    recs = live.records[1:]
    m_ref = live.scene.marker_reference
    rows = [{"step": r["step"], "phase": "ramp" if r["step"] <= n_r else "hold" if r["step"] <= n_r+n_h else "post",
             "J_min": r["j_stats"]["J_min"], "J_max": r["j_stats"]["J_max"], "Jbar_min": r["j_stats"]["Jbar_min"],
             "Jbar_max": r["j_stats"]["Jbar_max"], "max_speed_m_s": r["max_speed_m_s"],
             "energy_total_j": r["energies_j"]["total"], "elastic_j": r["energies_j"]["elastic"],
             "kinetic_j": r["energies_j"]["kinetic_particle"],
             "length_rel": (r["markers_m"][8][0]-r["markers_m"][0][0])/(m_ref[8][0]-m_ref[0][0]),
             "lateral_marker_max_m": float(np.abs(np.asarray(r["markers_m"])[:, 1:]-m_ref[:, 1:]).max())
             if np.isfinite(np.asarray(r["markers_m"])).all() else None,
             "flags": r["invalid_flags"], "j_domain_violations": r["j_domain_violations"], "nonfinite": r["nonfinite"]}
            for r in recs]
    first_flag = next((r for r in rows if r["flags"]), None)
    result = {"label": "DIAGNOSTIC (not the formal RUBBER-PHYS-07 result)", "substep_factor": args.substep_factor,
              "substeps": live.substeps, "dt_s": live.dt, "cfl": live.step_rule["cfl_actual"], "T_ax_s": T_ax,
              "ramp_steps": n_r, "hold_steps": n_h, "post_steps": n_post, "first_flag": first_flag, "rows": rows,
              "wall_s": time.perf_counter()-start}
    (out/"diag_07.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({"cfl": result["cfl"], "first_flag": first_flag}, default=float))


if __name__ == "__main__":
    main()
