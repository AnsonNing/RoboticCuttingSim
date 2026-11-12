"""Diagnostic probe (not a protocol test; review F8 recommendation): energy balance of a MOVING clamp, lumped vs the
F8 projection (+ F10 shadow), through the batched path on the Warp CPU device.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3_prep/probe_moving_grip_energy.py \
        --profile smooth --out artifacts/rl_ready/p2_3_prep/<id>.json

Case A cantilever 12 x 3 x 3 mm, n_t 2, mu 0.4 MPa, beta 20, rho 1150, gravity -9.81, control 1e-4 s, guard 1.0; N = 2
envs: the grip lifts 1 mm (+0.5 mm sideways in env 1) and rolls 0.15 rad x (e + 1) over 60 control steps, then holds
for 60. ``--profile step``: linear in time (velocity jumps at start / stop); ``smooth``: C2 smootherstep. Residual
E = Psi + KE_particle + E_grav - W (W = constraint + traction work of the runtime), reported relative to
max(Psi + KE).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_material import MaterialValues

MAT = MaterialValues(mu0=4e5, kappa=8e6, density=1150.0)
CASES = {"lumped": {}, "projection_f64": {"dirichlet": "consistent_mass_projection_v1",
                                          "state_precision": "f64_shadow_v1"}}


def live(opts):
    spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2, case="A")
    return LiveRubberMaterial(spec, material=MAT, control_dt=1e-4, gravity=(0, 0, -9.81), stable_step_guard=1.0,
                              **opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("step", "smooth"), default="smooth")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    result = {"profile": args.profile, "cases": {}}
    for label, opts in CASES.items():
        b = rmb.BatchedRubberMaterial([live(opts) for _ in range(2)], device="cpu", parity_mode=True)
        hist = []
        for k in range(120):
            tau = min(k+1, 60)/60.0
            s = tau if args.profile == "step" else tau**3*(10-15*tau+6*tau*tau)
            tr = np.zeros((2, 1, 3))
            tr[:, 0, 2] = 1e-3*s
            tr[1, 0, 1] = 5e-4*s
            R = np.stack([Rotation.from_rotvec([0.15*s*(e+1), 0, 0]).as_matrix() for e in range(2)])[:, None]
            b.advance(translation=tr, rotation=R)
            d = b.diagnostics_read()
            w = b.solver.work.numpy()
            hist.append([[x["elastic"], x["kinetic_particle"], x["gravity"], w[e]] for e, x in enumerate(d)])
        h = np.array(hist)
        rows = []
        for e in range(2):
            E = h[:, e, 0]+h[:, e, 1]+h[:, e, 2]-h[:, e, 3]
            scale = float(np.max(np.abs(h[:, e, 0])+np.abs(h[:, e, 1])))
            rows.append({"env": e, "max_abs_residual_rel": float(np.max(np.abs(E))/scale),
                         "end_residual_rel": float(E[-1]/scale), "work_end_J": float(h[-1, e, 3]), "scale_J": scale})
        result["cases"][label] = rows
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
