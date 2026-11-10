"""DIAGNOSTIC (P2-1 follow-up; NOT a formal RUBBER-PHYS-07 result): the grip release of RUBBER-PHYS-07 (protocol v2,
lambda 1.5, strip 60 x 10 x 4 mm, n_t 4) at fine time resolution.

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_1_followup/diag_07_release.py prepare --out <dir>
    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_1_followup/diag_07_release.py run --checkpoint <dir>/release.npz \
        --out <dir2> --dt-factor 1|2|4 [--free-free] [--ms 20] [--every 10]

``prepare`` runs the v2 protocol path (CFL 0.5, cosine ramp to --lambda (default 1.5) over 60 T_ax, hold 10 T_ax) and
saves the solver state at the release (x, v, F, work).

Correction (2026-11-10): the ``bc_ke_removed_lumped`` values of the first runs (``followup_07_release_*_001``) are
INVALID: on the Warp CPU device ``array.numpy()`` is a zero-copy view, so the "before" and "after" grid velocities
were the same memory and the measured removal was identically 0. The script now copies them (``*_002`` runs). ``run`` restores it into a fresh solver, deactivates grip 1 (and grip 0 with
--free-free), divides dt by --dt-factor and samples every --every substeps: J / Jbar extremes, minimum principal
stretch, kinetic (particle and grid), elastic and total mechanical energy, the lumped-grid kinetic energy removed by
the Dirichlet projection (both BC passes), the length between the end markers, lateral marker excursion, max speed,
and the deformation-aware CFL bound max_p dt c_bound(F_p) / dx (rubber_constitutive guard formula, float64 numpy).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))


def build(cfg, strict=False):
    from robotic_cutting_sim import rubber_phys as P
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    st = cfg["tests"]["RUBBER-PHYS-06"]["setup"]
    spec = rs.RubberStripSpec(length=st["L_m"], width=st["W_m"], thickness=st["t_m"], n_t=st["n_t"], case="C",
                              end_mask="x")
    return LiveRubberMaterial(spec, material=P.material(st["mu_pa"]), control_dt=st["control_interval_s"],
                              gravity=(0, 0, 0), strict=strict), st


def cfl_bound(F, jbar_p, mu, kappa, rho, dt, dx):
    """max_p dt c_bound / dx with rho c_bound^2 = a + (p'(Jbar) + 5 a I1/(9 J^2)) (J/s_min)^2 + 4a/(3J) s_max J/s_min."""
    F = F.astype(np.float64)
    J = np.linalg.det(F)
    s = np.linalg.svd(F, compute_uv=False)
    Jc = np.maximum(J, 0.05)
    a = mu*Jc**(-2/3)
    I1 = np.einsum("pij,pij->p", F, F)
    jb = np.maximum(jbar_p, 0.05)
    pp = kappa/2*(1+1/jb**2)
    g = np.abs(J)/np.maximum(s[:, 2], 1e-12)
    q = a+(pp+5*a*I1/(9*Jc*Jc))*g*g+4*a/(3*Jc)*s[:, 0]*g
    c = dt*np.sqrt(q/rho)/dx
    return c, s


def prepare(args):
    from robotic_cutting_sim import rubber_phys as P
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    cfg = P.load_config()
    live, st = build(cfg)
    L, dtc = st["L_m"], live.control_dt
    T_ax = P.axial_period(L, live.material)
    n_r, n_h = int(math.ceil(60*T_ax/dtc-1e-9)), int(math.ceil(10*T_ax/dtc-1e-9))
    t0 = time.perf_counter()
    for k in range(n_r+n_h):
        lam = P.cosine_path((k+1)*dtc, n_r*dtc, args.lam)
        live.advance({1: {"translation": np.array([(lam-1)*L, 0.0, 0.0])}})
    s = live.solver
    d = s.diagnostics.read()[0]
    np.savez(out/"release.npz", x=s.x.numpy(), v=s.v.numpy(), F=s.F.numpy(), work=s.work.numpy(),
             b1=s.group_b1.numpy())
    meta = {"label": "DIAGNOSTIC checkpoint at the RUBBER-PHYS-07 (v2 path) release", "lambda": args.lam,
            "ramp_steps": n_r, "hold_steps": n_h,
            "T_ax_s": T_ax, "substeps": live.substeps, "dt_s": live.dt, "particles": live.scene.n,
            "diagnostics_at_release": d, "external_work_j": float(s.work.numpy()[0]),
            "invalid": live.invalid_reason, "wall_s": time.perf_counter()-t0}
    (out/"release_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps(meta, default=float))


def run(args):
    import warp as wp
    from robotic_cutting_sim import rubber_phys as P
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    cfg = P.load_config()
    live, st = build(cfg)
    s = live.solver
    ck = np.load(args.checkpoint)
    s.x.assign(ck["x"])
    s.v.assign(ck["v"])
    s.F.assign(ck["F"])
    s.begin_control_step()
    s.set_group_motion(1, translation=ck["b1"][1], active=False)
    if args.free_free:
        s.set_group_motion(0, active=False)
    s.work.zero_()
    s.dt = live.dt/args.dt_factor
    mat = live.material
    mu, kappa, rho, dx = mat.mu0, mat.kappa, mat.density, s.dx
    m_ref = live.scene.marker_reference
    cell_of = s.scene.nodes[:, 0]
    gnode = s.scene.node_group
    nsub = int(round(args.ms*1e-3/s.dt))
    rows = []
    e_bc = 0.0
    t0 = time.perf_counter()

    def sample(k):
        d = s.diagnostics.read()[0]
        F = s.F.numpy()
        jb = s.jbar.numpy()[cell_of]
        finite = bool(np.isfinite(F).all())
        row = {"substep": k, "t_ms": k*s.dt*1e3, "finite": finite and d["nonfinite"] == 0.0,
               "kinetic_particle": d["kinetic_particle"], "kinetic_grid": d["kinetic_grid"], "elastic": d["elastic"],
               "mechanical": d["kinetic_particle"]+d["elastic"], "bc_ke_removed_lumped": e_bc,
               "J_min": d["J_min"], "J_max": d["J_max"], "Jbar_min": d["Jbar_min"], "Jbar_max": d["Jbar_max"],
               "max_speed": d["max_speed_m_s"], "max_grid_speed": d["max_grid_speed_m_s"],
               "violations": int(s.violations.numpy()[0])}
        if finite:
            c, sv = cfl_bound(F, jb, mu, kappa, rho, s.dt, dx)
            p = int(np.argmax(c))
            mk = s.diagnostics.marker_x.numpy()
            row.update(cfl_bound_max=float(c.max()), cfl_bound_p99=float(np.percentile(c, 99)),
                       cfl_bound_argmax_ref_x_mm=float(s.scene.X[p, 0]*1e3), stretch_min=float(sv[:, 2].min()),
                       stretch_max=float(sv[:, 0].max()), argmin_stretch_ref_x_mm=float(
                           s.scene.X[int(np.argmin(sv[:, 2])), 0]*1e3),
                       length_rel=float((mk[8][0]-mk[0][0])/(m_ref[8][0]-m_ref[0][0])),
                       lateral_max_m=float(np.abs(mk[:, 1:]-m_ref[:, 1:]).max()),
                       momentum_x=float((s.mass.numpy()[:, None]*s.v.numpy()).sum(0)[0]))
        rows.append(row)
        return row["finite"]

    sample(0)
    gm_host = None
    for k in range(1, nsub+1):
        # one substep with the lumped-grid kinetic energy removed by the Dirichlet projection measured around the
        # two BC passes (same launches as TLMPMRubber.step; measurement only)
        n, nn, dt = s.n, s.node_count, s.dt
        with wp.ScopedDevice(s.device):
            from robotic_cutting_sim import tlmpm_v5 as v5
            from robotic_cutting_sim import rubber_constitutive as rc
            from robotic_cutting_sim.tlmpm_v6 import _grid_positions
            from robotic_cutting_sim.tlmpm_rubber_v1 import _stress, _loads
            wp.copy(s.x_start, s.x)
            wp.launch(rc.cell_jbar, s.cells, [s.cell_node, s.cell_offsets, s.cell_particles, s.F, s.jbar])
            wp.launch(_stress, n, [s.F, s.nodes, s.jbar, s.stress, mu, kappa, wp.float64(s.j_domain[0]),
                                   wp.float64(s.j_domain[1]), s.violations])
            wp.launch(_loads, n, [s.x, s.v, s.mass, s.impulse, s.particle_traction, s.traction_force,
                                  s.traction_mass, dt, 0.0, s.radius, s.contact_stiffness, mat.friction_table,
                                  s.use_plane, s.plane, s.boxes, s.box_c, s.box_h, s.work])
            s.gm.zero_(); s.gp.zero_(); s.gj.zero_()
            wp.launch(v5._p2g, (n, 8), [s.v, s.mass, s.slot, s.stress, s.impulse, s.nodes, s.weights, s.gradients,
                                        s.gm, s.gp, s.gj, 1, s.volume, dt, s.gravity])
            wp.launch(v5._grid_update, (1, nn), [s.gm, s.gp, s.gj, s.old, s.new_v, s.slot_code, 1])
            if not args.free_free:
                gm_host = s.gm.numpy().copy()
                before = s.new_v.numpy().copy()
            s._launch_bc(1)
            if not args.free_free:
                after = s.new_v.numpy().copy()
                sel = gnode >= 0
                e_bc += float(0.5*np.sum(gm_host[sel, None].astype(np.float64)*(before[sel].astype(np.float64)**2
                                                                              - after[sel].astype(np.float64)**2)))
            wp.launch(v5._g2p, n, [s.x, s.v, s.slot, s.nodes, s.weights, s.old, s.new_v, 1, dt, 0.0])
            s.gp.zero_(); s.gj.zero_()
            wp.launch(v5._musl, (n, 8), [s.v, s.mass, s.slot, s.nodes, s.weights, s.gp, 1])
            wp.launch(v5._grid_update, (1, nn), [s.gm, s.gp, s.gj, s.old, s.new_v, s.slot_code, 1])
            if not args.free_free:
                before = s.new_v.numpy().copy()
            s._launch_bc(0)
            if not args.free_free:
                after = s.new_v.numpy().copy()
                e_bc += float(0.5*np.sum(gm_host[sel, None].astype(np.float64)*(before[sel].astype(np.float64)**2
                                                                              - after[sel].astype(np.float64)**2)))
            wp.launch(v5._deformation, n, [s.F, s.slot, s.nodes, s.gradients, s.new_v, 1, dt])
            wp.launch(_grid_positions, n, [s.x, s.x_start, s.slot, s.nodes, s.weights, s.new_v, 1, dt])
            wp.launch(v5._advance, 1, [s.index])
        if k % args.every == 0 or k == nsub:
            if not sample(k):
                break
    e0 = rows[0]["mechanical"]
    for r in rows:
        r["mechanical_rel_change"] = (r["mechanical"]-e0)/e0
    first_bad = next((r for r in rows if not r["finite"]), None)
    first_viol = next((r for r in rows if r["violations"] > 0), None)
    res = {"label": "DIAGNOSTIC (not a formal RUBBER-PHYS-07 result)", "checkpoint": args.checkpoint,
           "dt_factor": args.dt_factor, "dt_s": s.dt, "cfl_nominal": s.dt*mat.c_p()/dx, "free_free": args.free_free,
           "substeps_run": rows[-1]["substep"], "every": args.every, "first_nonfinite": first_bad,
           "first_j_domain_violation": first_viol, "rows": rows, "wall_s": time.perf_counter()-t0}
    (out/"diag.json").write_text(json.dumps(res, indent=1, default=float))
    summary = {k: res[k] for k in ("dt_factor", "cfl_nominal", "free_free", "substeps_run")}
    summary["first_nonfinite_t_ms"] = first_bad and first_bad["t_ms"]
    summary["first_violation_t_ms"] = first_viol and first_viol["t_ms"]
    print(json.dumps(summary, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--out", required=True)
    a.add_argument("--lambda", dest="lam", type=float, default=1.5)
    b = sub.add_parser("run")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--dt-factor", type=int, default=1)
    b.add_argument("--free-free", action="store_true")
    b.add_argument("--ms", type=float, default=20.0)
    b.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else run(args)


if __name__ == "__main__":
    main()
