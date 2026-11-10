"""DIAGNOSTIC (P2-4 decision probe, ADR 0023 §22.1; NOT a protocol result): the frozen per-particle helper
(``tlmpm_v5._contact_impulse`` through the ``table_plane`` path of ``tlmpm_rubber_v1``, the ADR 0023 D8 design) vs the
grid-node law ``grid_node_coulomb_v1`` (``rubber_contact``), on the Warp CPU device, single env.

Scenes (strip-local frame, elastic neo_hookean_jbar_v1, mu 0.4 MPa, beta 20, rho 1150, CFL 0.5 unless ``x2``):

Laws: ``particle_frozen`` (frozen helper), ``grid_node`` (``grid_node_coulomb_v1``, velocity-level node projection),
``grid_penalty`` (``grid_node_penalty_coulomb_v1``, node penalty + Coulomb spring-dashpot-slider, the module defaults
kappa_n = kappa_t = 0.25, zeta_n = zeta_t = 0.5), ``grid_penalty_undamped``: at commit 259f2b6 (run
probe_contact_laws_004) this variant set only ``damping_ratio`` = 0, i.e. the TANGENTIAL dashpot off and the normal dashpot
0.5 kept (a normal-dashpot-only variant; its impact rows equal the damped law). Fully undamped data (zeta_n = zeta_t = 0):
probe_contact_laws_003 (single-zeta kernel then) and ``diag_contact_sweep.py``. From the commit after 259f2b6 the variant
sets both dashpots to 0.

- ``incline_<stick|slip>_<law>``: block 12 x 6 x 3 mm (n_t 3, case free) on the plane z = 0, gravity tilted by theta
  about y (g = 9.81 (sin theta, 0, -cos theta)), friction mu_table 0.5, tan theta = 0.8 / 0.95 mu (stick), 1.25 mu
  (slip) plus 0.9 / 0.95 / 1.05 / 1.15; gravity ramped (cosine, updated every substep) over 20 ms (quasi-static: a step load on an undamped elastic block overshoots by 2
  and unloads the contact periodically), 0.12 s, 1 ms control steps; centroid displacement along x, quadratic fit on
  [0.04, 0.12] s;
- ``impact_<law>[_x2]``: rod 30 x 4 x 4 mm (n_t 4, case free), no gravity, uniform initial velocity -0.2 m/s along x
  onto the FRICTIONLESS plane x = 0 (normal +x; the 1-D reference has no friction), 6 ms at 0.02 ms control steps; total energy KE + Psi, centroid velocity, contact
  time; 1-D elastic rod reference: rebound velocity = impact velocity, contact time 2 L / c0, c0 = sqrt(E/rho).

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/probe_contact_laws.py \
        --out artifacts/rl_ready/p2_4/<dir> [--jobs N]
"""
import argparse
import json
import math
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))

MU, BETA, RHO, G = 4.0e5, 20.0, 1150.0, 9.81


def _material(mu_table=0.5):
    from robotic_cutting_sim.rubber_material import MaterialValues
    return MaterialValues(mu0=MU, kappa=BETA*MU, density=RHO, friction_table=mu_table, friction_pad=1.0)


def _solver(law, spec, material, gravity, control_dt, cfl, plane):
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING if hasattr(wp, "LOG_WARNING") else None
    from robotic_cutting_sim import rubber_material as rm
    from robotic_cutting_sim import rubber_scene as rs
    scene = rs.compile_strip(spec)
    rule = rm.step_rule(scene.dx, control_dt, mu0_max=material.mu0, kappa_max=material.kappa,
                        density_min=material.density, cfl=cfl)
    sub = rule["substeps_per_control_step"]
    dt = control_dt/sub
    common = dict(substeps=sub, gravity=gravity, device="cpu", stable_step_guard=1.0)
    if law == "particle_frozen":
        from robotic_cutting_sim.tlmpm_rubber_v1 import TLMPMRubber
        s = TLMPMRubber(scene, material, dt, table_plane=plane, **common)
    else:
        from robotic_cutting_sim.rubber_contact import PENALTY_LAW, PROJECTION_LAW, ContactSpec
        from robotic_cutting_sim.tlmpm_rubber_contact_v1 import TLMPMRubberContact
        lawid = PROJECTION_LAW if law == "grid_node" else PENALTY_LAW
        extra = {"damping_ratio": 0.0, "normal_damping_ratio": 0.0} if law == "grid_penalty_undamped" else {}
        s = TLMPMRubberContact(scene, material, dt, contact=ContactSpec(planes=(tuple(plane),), law=lawid, **extra),
                               **common)
    s.reset()
    return scene, s, rule


def run_incline(name, law, regime):
    import warp as wp
    from robotic_cutting_sim import rubber_scene as rs
    mu = 0.5
    ratio = {"stick": 0.8, "slip": 1.25, "stick95": 0.95, "slip105": 1.05, "stick90": 0.9, "slip115": 1.15}[regime]
    theta = math.atan(ratio*mu)
    gvec = np.array([G*math.sin(theta), 0.0, -G*math.cos(theta)])
    spec = rs.RubberStripSpec(length=0.012, width=0.006, thickness=0.003, n_t=3, case="free")
    control_dt, T, T_ramp = 0.001, 0.12, 0.02
    scene, s, rule = _solver(law, spec, _material(mu), tuple(gvec), control_dt, 0.5, (0.0, 0.0, 1.0, 0.0))
    t0 = time.perf_counter()
    ts, xs = [0.0], [float(s.x.numpy()[:, 0].astype(np.float64).mean())]
    for k in range(int(round(T/control_dt))):
        tm = (k+0.5)*control_dt                 # gravity constant per control step at its midpoint value
        r = 0.5*(1-math.cos(math.pi*tm/T_ramp)) if tm < T_ramp else 1.0
        s.gravity = wp.vec3(*(gvec*r).tolist())
        s.begin_control_step()
        for j in range(s.substeps):
            tj = k*control_dt+(j+0.5)*s.dt          # gravity ramped per substep (smooth, quasi-static)
            rj = 0.5*(1-math.cos(math.pi*tj/T_ramp)) if tj < T_ramp else 1.0
            s.gravity = wp.vec3(*(gvec*rj).tolist())
            s.step()
        ts.append((k+1)*control_dt)
        xs.append(float(s.x.numpy()[:, 0].astype(np.float64).mean()))
    ts, xs = np.array(ts), np.array(xs)-xs[0]
    sel = ts >= 0.04
    c = np.polyfit(ts[sel], xs[sel], 2)
    a_ref = G*(math.sin(theta)-mu*math.cos(theta)) if ratio > 1 else 0.0
    diag = s.diagnostics.read()[0]
    out = {"name": name, "law": law, "regime": regime, "theta_deg": math.degrees(theta), "tan_theta_over_mu": ratio,
           "mu": mu, "gravity_ramp_s": T_ramp, "fit_window_s": [0.04, T], "particles": int(scene.n), "dt_s": s.dt,
           "substeps": s.substeps, "a_fit_m_s2": float(2*c[0]), "v_fit_end_m_s": float(2*c[0]*ts[-1]+c[1]),
           "a_ref_m_s2": a_ref, "displacement_end_m": float(xs[-1]),
           "displacement_after_ramp_m": float(xs[-1]-xs[int(round(T_ramp/control_dt))]),
           "J_min": diag["J_min"], "J_max": diag["J_max"],
           "cfl_violations": int(s.cfl_violations.numpy()[0]), "wall_s": time.perf_counter()-t0,
           "series": {"t": ts.tolist(), "x": xs.tolist()}}
    if law != "particle_frozen":
        out["contact"] = s.contact_diagnostics()
    return out


def run_impact(name, law, cfl):
    from robotic_cutting_sim import rubber_scene as rs
    V = 0.2
    spec = rs.RubberStripSpec(length=0.03, width=0.004, thickness=0.004, n_t=4, case="free")
    control_dt, T = 2.0e-5, 6.0e-3
    m = _material(0.0)                          # frictionless wall: the 1-D rod reference has no friction
    scene, s, rule = _solver(law, spec, m, (0.0, 0.0, 0.0), control_dt, cfl, (1.0, 0.0, 0.0, 0.0))
    s.v.assign(np.tile(np.array([-V, 0.0, 0.0], np.float32), (s.n, 1)))
    M = float(s.mass_value)*scene.n
    E0 = 0.5*M*V*V
    t0 = time.perf_counter()
    rows = []
    for k in range(int(round(T/control_dt))):
        s.begin_control_step()
        for _ in range(s.substeps):
            s.step()
        d = s.diagnostics.read()[0]
        v = s.v.numpy().astype(np.float64)
        row = {"t": (k+1)*control_dt, "E": d["kinetic_particle"]+d["elastic"], "KE": d["kinetic_particle"],
               "psi": d["elastic"], "vcm": float(v[:, 0].mean()), "xmin": float(s.x.numpy()[:, 0].min())}
        if law != "particle_frozen":
            cd = s.contact_diagnostics()
            row.update(events=cd["impulse_events"], pen=cd["max_penetration_m"], Wn=cd["normal_work_material"],
                       Wt=cd["tangential_work_material"], phin=cd["contact_energy_normal"],
                       phit=cd["contact_energy_tangential"])
        else:
            row.update(events=int(row["xmin"] < 0.45*scene.spacing+1e-12))
        rows.append(row)
    E = np.array([r["E"] for r in rows])
    in_contact = [r["t"] for r in rows if r["events"]]
    E_mod = m.young_effective()
    c0 = math.sqrt(E_mod/RHO)
    out = {"name": name, "law": law, "cfl": cfl, "particles": int(scene.n), "dt_s": s.dt, "substeps": s.substeps,
           "V_m_s": V, "E0_J": E0, "E_end_over_E0": float(E[-1]/E0), "E_min_over_E0": float(E.min()/E0),
           "vcm_end_over_V": float(rows[-1]["vcm"]/V), "contact_first_last_s": [in_contact[0], in_contact[-1]]
           if in_contact else None, "contact_time_ref_s": 2*0.03/c0, "c0_m_s": c0,
           "x_min_m": float(min(r["xmin"] for r in rows)), "wall_s": time.perf_counter()-t0, "series": rows}
    if law != "particle_frozen":
        cd = s.contact_diagnostics()
        out["books_end_over_E0"] = float((rows[-1]["E"]-(cd["normal_work_material"]+cd["tangential_work_material"])
                                          - E0)/E0)
        out["max_books_over_E0"] = float(max(abs(r["E"]-r["Wn"]-r["Wt"]-E0) for r in rows)/E0)
        out["max_penetration_m"] = float(max(r["pen"] for r in rows))
        out["normal_work_end_J"] = cd["normal_work_material"]
    return out


JOBS = {}
for _law in ("particle_frozen", "grid_node", "grid_penalty", "grid_penalty_undamped"):
    for _regime in ("stick", "stick90", "stick95", "slip105", "slip115", "slip"):
        JOBS[f"incline_{_regime}_{_law}"] = (run_incline, (_law, _regime))
    JOBS[f"impact_{_law}"] = (run_impact, (_law, 0.5))
    JOBS[f"impact_{_law}_x2"] = (run_impact, (_law, 0.25))


def _job(name):
    fn, args = JOBS[name]
    return fn(name, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    names = [n for n in JOBS if not a.only or n in a.only.split(",")]
    meta = {"label": "DIAGNOSTIC_DECISION_PROBE (ADR 0023 §22.1), not a protocol result",
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            "dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                         cwd=ROOT).stdout.strip()),
            "python": platform.python_version(), "argv": sys.argv, "jobs": names,
            "start": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    with ProcessPoolExecutor(a.jobs) as ex:
        results = dict(zip(names, ex.map(_job, names)))
    meta["end"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    (out/"results.json").write_text(json.dumps(results, indent=1))
    for n, r in results.items():
        r = {k: v for k, v in r.items() if k not in ("series",)}
        print(n, json.dumps(r, default=str)[:900])


if __name__ == "__main__":
    main()
