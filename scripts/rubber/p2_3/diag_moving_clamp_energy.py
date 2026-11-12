"""DIAGNOSTIC (P2-3 prerequisite; NOT a protocol result): where the energy books of a MOVING Dirichlet grip open.

Single-env reference ``tlmpm_rubber_v1`` on the Warp CPU device, the scene of ``p2_3_prep/probe_moving_grip_energy.py``
(case A cantilever 12 x 3 x 3 mm, n_t 2, mu 0.4 MPa, beta 20, rho 1150, gravity -9.81, guard 1.0): the grip lifts
1 mm and rolls 0.15 rad about x over ``--move-ms`` (default 6 ms), then holds for the same time. ``--profile step``:
pose linear in time; ``smooth``: C2 smootherstep sampled at the control-step ends (the pose is still linear in s
WITHIN a control step, so the prescribed velocity is piecewise constant per control step).

The solver is subclassed in this script only (host reads after each projection / BC launch; CPU arrays are synchronous)
to evaluate, per substep and per constrained (node, component) row, the constraint impulse Lambda_n (on the material)
and three quadratures of its work:

- ``end``   : Lambda_n . v_presc,n           (what ``tlmpm_rubber_v1`` accumulates in ``work``)
- ``start`` : Lambda_n . g_n,C               (g_n = P2G-pass grid velocity ``old`` = p/m = B pi_n)
- ``kick``  : Lambda_n . (g_n,C + v_presc,n)/2

The discrete scheme (pure FLIP + MUSL, fixed weights) is symplectic Euler in (pi, u) with KE = 1/2 pi^T B pi
(B = M_l^-1 M_c M_l^-1) = the particle KE, g = B pi (ADR 0023 §18.1 F1). Under the F8 projection the kick is
pi_{n+1} = pi_n + dt f(u_n) + Lambda_n and (B pi_{n+1})_C = v_presc,n, so
KE_{n+1} - KE_n = (dt f + Lambda_n) . (g_n + g_{n+1})/2 exactly (KE quadratic): the work conjugate to the discrete KE
is the ``kick`` quadrature. Also recorded: KE of the jaw particles (all 8 stencil nodes constrained), grid KE.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/diag_moving_clamp_energy.py \
        --out artifacts/rl_ready/p2_3/<dir> [--variants all|<name,...>]
"""
import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))

MU, BETA, RHO = 4.0e5, 20.0, 1150.0


def variants():
    """name -> (profile, dirichlet, state_precision, control_dt_s, substep_factor)."""
    out = {}
    for prof in ("step", "smooth"):
        for dname, dmode, prec in (("lumped", "lumped_v1", "float32_v1"),
                                   ("proj", "consistent_mass_projection_v1", "float32_v1"),
                                   ("projf64", "consistent_mass_projection_v1", "f64_shadow_v1")):
            out[f"{prof}_{dname}_c100us_x1"] = (prof, dmode, prec, 1e-4, 1)
        # time-step and control-interval dependence (projection + shadow)
        out[f"{prof}_projf64_c100us_x2"] = (prof, "consistent_mass_projection_v1", "f64_shadow_v1", 1e-4, 2)
        out[f"{prof}_projf64_c25us_x1"] = (prof, "consistent_mass_projection_v1", "f64_shadow_v1", 2.5e-5, 1)
        out[f"{prof}_lumped_c25us_x1"] = (prof, "lumped_v1", "float32_v1", 2.5e-5, 1)
    return out


def make_probe_class():
    from robotic_cutting_sim.tlmpm_rubber_v1 import TLMPMRubber

    class ProbeSolver(TLMPMRubber):
        """Host-side work quadratures; the launches are those of the parent (no state change)."""
        def _init_probe(self):
            self.w_end = self.w_start = self.w_kick = 0.0
            self.n_impulse_rows = 0

        def _launch_projection(self):
            old = self.old.numpy().copy()                      # P2G-pass g_n = p_n / m (before any overwrite)
            super()._launch_projection()
            act = self.group_active.numpy()
            rn, rc_, rg = self.proj_row_node.numpy(), self.proj_row_comp.numpy(), self.proj_row_group.numpy()
            on = act[rg] != 0
            gm = self.gm.numpy().astype(np.float64)
            lam = -gm[rn]*self.proj_mu.numpy()                 # impulse on the material (as _proj_reaction)
            vp = self.proj_vpresc.numpy()
            gn = old[rn, rc_].astype(np.float64)
            self.w_end += float(np.sum((lam*vp)[on]))
            self.w_start += float(np.sum((lam*gn)[on]))
            self.w_kick += float(np.sum((lam*0.5*(gn+vp))[on]))

        def _launch_bc(self, record):
            if record == 0 or self.G == 0:
                return super()._launch_bc(record)
            old = self.old.numpy().copy()
            before = self.new_v.numpy().copy()
            super()._launch_bc(record)
            after = self.new_v.numpy()
            gm = self.gm.numpy().astype(np.float64)
            grp, mask = self.node_group.numpy(), self.node_mask.numpy()
            act = self.group_active.numpy()
            sel = np.flatnonzero(grp >= 0)
            sel = sel[act[grp[sel]] != 0]
            for k in range(3):
                s = sel[((mask[sel] >> k) & 1) == 1]
                vn = after[s, k].astype(np.float64)
                lam = gm[s]*(vn-before[s, k].astype(np.float64))
                gn = old[s, k].astype(np.float64)
                self.w_end += float(np.sum(lam*vn))
                self.w_start += float(np.sum(lam*gn))
                self.w_kick += float(np.sum(lam*0.5*(gn+vn)))
    return ProbeSolver


def run_variant(name, cfg):
    import warp as wp
    from scipy.spatial.transform import Rotation
    from robotic_cutting_sim import live_rubber_material as lrm
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.rubber_material import MaterialValues
    profile, dirichlet, precision, control_dt, factor = cfg
    Probe = make_probe_class()
    original = lrm.TLMPMRubber
    lrm.TLMPMRubber = Probe                     # this script only: the runtime builds the probe subclass
    try:
        spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2, case="A")
        mat = MaterialValues(mu0=MU, kappa=BETA*MU, density=RHO)
        base = lrm.LiveRubberMaterial(spec, material=mat, control_dt=control_dt, gravity=(0, 0, -9.81),
                                      stable_step_guard=1.0, strict=False)
        substeps = base.substeps*factor
        live = lrm.LiveRubberMaterial(spec, material=mat, control_dt=control_dt, gravity=(0, 0, -9.81),
                                      stable_step_guard=1.0, strict=False, substeps=substeps, dirichlet=dirichlet,
                                      state_precision=precision)
    finally:
        lrm.TLMPMRubber = original
    s = live.solver
    s._init_probe()
    sc = live.scene
    jaw_nodes = sc.node_group >= 0
    jaw_particles = jaw_nodes[sc.nodes].all(1)
    n_move = int(round(6e-3/control_dt))
    rows = []
    t0 = time.perf_counter()
    for k in range(2*n_move):
        tau = min(k+1, n_move)/n_move
        sv = tau if profile == "step" else tau**3*(10-15*tau+6*tau*tau)
        R = Rotation.from_rotvec([0.15*sv, 0.0, 0.0]).as_matrix()
        live.advance({0: {"rotation": R, "translation": np.array([0.0, 0.0, 1e-3*sv])}})
        d = live.last_diagnostics
        v = s.v.numpy().astype(np.float64)
        m = s.mass.numpy().astype(np.float64)
        ke_jaw = float(0.5*np.sum(m[jaw_particles]*np.sum(v[jaw_particles]**2, 1)))
        rows.append({"step": k+1, "t_s": (k+1)*control_dt, "elastic": d["elastic"], "kinetic_particle":
                     d["kinetic_particle"], "kinetic_grid": d["kinetic_grid"], "gravity": d["gravity"],
                     "kinetic_jaw_particles": ke_jaw, "work_solver": float(s.work.numpy()[0]),
                     "work_end": s.w_end, "work_start": s.w_start, "work_kick": s.w_kick,
                     "cfl_eff_max": d["cfl_eff_max"], "invalid": live.invalid_reason})
    wall = time.perf_counter()-t0
    h = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    mech = h["elastic"]+h["kinetic_particle"]+h["gravity"]
    scale = float(np.max(np.abs(h["elastic"])+np.abs(h["kinetic_particle"])))
    hold = h["step"] > n_move
    summary = {}
    for wname in ("work_solver", "work_end", "work_start", "work_kick"):
        res = (mech-h[wname])/scale
        summary[wname] = {"max_abs_residual_rel": float(np.max(np.abs(res))), "end_residual_rel": float(res[-1]),
                          "hold_mean_residual_rel": float(np.mean(res[hold])),
                          "hold_range_rel": float(np.ptp(res[hold]))}
    return {"variant": name, "profile": profile, "dirichlet": dirichlet, "state_precision": precision,
            "control_dt_s": control_dt, "substeps": live.substeps, "dt_s": live.dt, "particles": sc.n,
            "jaw_particles": int(jaw_particles.sum()), "constrained_nodes": int(jaw_nodes.sum()),
            "move_steps": n_move, "scale_J": scale, "invalid_reason": live.invalid_reason,
            "max_cfl_eff": float(h["cfl_eff_max"].max()), "wall_s": wall,
            "check_work_solver_minus_end_J": float(np.max(np.abs(h["work_solver"]-h["work_end"]))),
            "residuals": summary, "max_kinetic_jaw_rel": float(np.max(h["kinetic_jaw_particles"])/scale),
            "series": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="all")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    import warp as wp
    wp.config.quiet = True
    wp.init()
    allv = variants()
    names = list(allv) if args.variants == "all" else args.variants.split(",")
    meta = {"label": "DIAGNOSTIC (moving-clamp energy books; not a protocol result)",
            "command": " ".join(sys.argv), "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                                    capture_output=True, text=True).stdout.strip(),
            "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip()),
            "python": platform.python_version(), "warp": wp.config.version, "numpy": np.__version__,
            "device": "cpu", "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED", "variants": names}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    table = []
    for name in names:
        r = run_variant(name, allv[name])
        (out/f"{name}.json").write_text(json.dumps(r, indent=1))
        line = {k: r[k] for k in ("variant", "substeps", "dt_s", "scale_J", "invalid_reason", "max_cfl_eff",
                                  "max_kinetic_jaw_rel", "check_work_solver_minus_end_J")}
        line.update({f"{w}_{q}": r["residuals"][w][q] for w in ("work_end", "work_kick", "work_start")
                     for q in ("end_residual_rel", "max_abs_residual_rel")})
        table.append(line)
        print(json.dumps(line), flush=True)
    (out/"summary.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
