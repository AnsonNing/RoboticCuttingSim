"""DIAGNOSTIC (task X2, ADR 0023 §22.10; NOT a protocol result): ``diag_contact_x1.py`` with the same instrumentation
plus the X2 development laws (``dev=x2 smode=.. radius=.. tau=.. ds=.. vg=.. switch_t=..``, ``x2_dev_laws.py``) and
custom Prony terms (``prony=g:tau,...`` on the elastic mu0: HF-loss probe). The X1 text follows unchanged.

Root cause of the mu-dependent friction deficit of ``grid_node_penalty_coulomb_v1`` on the formal INC scene (block
16 x 8 x 4 mm, n_t 3, table z = 0, tilted gravity
ramped every substep over 20 ms, 140 ms), Warp CPU, single env.

One run per process; prints one JSON line. Arguments ``key=value``:

    mu=0.8 ratio=1.2 material=elastic|P2 cfl=0.5 nt=3 T=0.14 kn=0.25 kt=0.25 zt=0.5 zn=0.5 law=v1|<law name>
    L=0.016 W=0.008 H=0.004 mu0=4e5

``law=v1`` runs the committed ``grid_node_penalty_coulomb_v1`` kernel; any other value is passed to ``ContactSpec(law=)``
(a law registered in ``rubber_contact.CONTACT_LAWS``). The instrumentation reads the per-node impulses ``ct_jn`` /
``ct_jt`` after every substep of the fit window [0.05 s, T] and decomposes the friction deficit
1 - sum(-J_t,x) / (mu sum J_n,z) into the part carried by STICK nodes (|J_t| < mu |J_n|) and the misalignment of SLIP
nodes (|J_t| = mu |J_n| but not along -x); it also reports the contact fraction of the bottom-face nodes, the
substep-to-substep normal-impulse fluctuation, the vibration energy KE - 1/2 M v_cm^2, the -04 ratio on the window and
the slip acceleration error of -01.

    CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/rubber/p2_4/diag_contact_x1.py mu=0.8 ratio=1.2
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"/"rubber"/"p2_4"))
G = 9.81


def main(argv):
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    import run_contact_v1 as R
    from robotic_cutting_sim.rubber_contact import ContactSpec
    from robotic_cutting_sim.rubber_material import MaterialValues
    a = dict(x.split("=", 1) for x in argv)
    mu, ratio = float(a.get("mu", 0.8)), float(a.get("ratio", 1.2))
    theta_arg = a.get("theta")
    cfl, nt, T = float(a.get("cfl", 0.5)), int(a.get("nt", 3)), float(a.get("T", 0.14))
    L, W, H = float(a.get("L", 0.016)), float(a.get("W", 0.008)), float(a.get("H", 0.004))
    law = a.get("law", "v1")
    kw = dict(normal_stiffness=float(a.get("kn", 0.25)), tangential_stiffness=float(a.get("kt", 0.25)),
              damping_ratio=float(a.get("zt", 0.5)), normal_damping_ratio=float(a.get("zn", 0.5)))
    if law != "v1":
        kw["law"] = law
    mat = R._material(a.get("material", "elastic"), friction_table=mu)
    if "prony" in a:          # X2: custom isochoric Prony terms "g:tau,g:tau" on the elastic mu0 (HF-loss probe)
        terms = tuple(tuple(map(float, q.split(":"))) for q in a["prony"].split(","))
        mat = MaterialValues(mu0=mat.mu0, kappa=mat.kappa, density=mat.density, prony=terms, friction_table=mu)
    if "mu0" in a:
        mu0 = float(a["mu0"])
        mat = MaterialValues(mu0=mu0, kappa=20*mu0, density=mat.density, prony=mat.prony, friction_table=mu)
    theta = math.atan(ratio*mu) if theta_arg is None else float(theta_arg)
    gvec = np.array([G*math.sin(theta), 0.0, -G*math.cos(theta)])
    spec = R._spec(length=L, width=W, thickness=H, n_t=nt, case="free")
    live = R._live(spec, mat, 1e-3, (0, 0, 0), ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),), **kw), cfl=cfl)
    s = live.solver
    if a.get("dev") == "x2":
        import x2_dev_laws
        x2_dev_laws.install(live, smode=int(a.get("smode", 1)), radius=int(a.get("radius", 1)),
                            tau=float(a.get("tau", 1e-3)), ds=float(a.get("ds", 5e-6)),
                            vg=float(a.get("vg", 1e-3)), switch_t=float(a.get("switch_t", 0.0)))
    elif a.get("dev") == "1":
        import x1_dev_laws
        x1_dev_laws.install(live, vmode=int(a.get("vmode", 0)), capmode=int(a.get("capmode", 0)),
                            dirmode=int(a.get("dirmode", 0)), ntau=float(a.get("ntau", 50)))
    M = float(s.mass_value)*s.n
    X = s.node_X.numpy()
    gm = s.gm.numpy()
    st = R.Stepper(live, lambda t: (gvec*R._ramp(t, 0.02), None))
    step_cf = []
    acc = {"S_Jn": 0.0, "S_f": 0.0, "S_slipJn": 0.0, "S_slip_f": 0.0, "S_stickJn": 0.0, "S_stick_f": 0.0,
           "S_absJty": 0.0, "n_sub": 0, "contact_frac": 0.0, "S_fv": 0.0}
    jn_hist = []
    fit0 = 0.05
    base_substeps = st._substeps

    def substeps():
        sv = live.solver
        for _ in range(sv.substeps):
            g, _tr = st.loads(st.t+0.5*sv.dt)
            sv.gravity = wp.vec3(*map(float, g))
            sv.step()
            x = sv.x64.numpy().mean(0)
            dx = x-st.xcm
            st.W_g += st.M*float(np.dot(np.asarray(g, float), 0.5*(st.prev_dx+dx)))
            st.prev_dx, st.xcm = dx, x.copy()
            st.t += sv.dt
            bottom = (np.abs(X[:, 2]) < 1e-12) & (gm > 0)
            step_cf.append(float((np.abs(sv.ct_jn.numpy()[bottom, 2]) > 0).mean()))
            if st.t >= fit0:
                jn = sv.ct_jn.numpy()
                jt = sv.ct_jt.numpy()
                njn = np.abs(jn[:, 2])
                njt = np.linalg.norm(jt, axis=1)
                con = njn > 0
                slip = con & (np.abs(njt-mu*njn) <= 1e-6*mu*njn+1e-300)
                stick = con & ~slip
                f = -jt[:, 0]
                acc["S_Jn"] += njn.sum()
                acc["S_f"] += f.sum()
                acc["S_slipJn"] += njn[slip].sum()
                acc["S_slip_f"] += f[slip].sum()
                acc["S_stickJn"] += njn[stick].sum()
                acc["S_stick_f"] += f[stick].sum()
                acc["S_absJty"] += np.abs(jt[:, 1]).sum()
                acc["n_sub"] += 1
                acc["contact_frac"] += con[bottom].mean()
                vcm = sv.v.numpy().astype(np.float64).mean(0)
                acc["S_fv"] += float(f.sum()*vcm[0])
                if len(jn_hist) < 4000:
                    jn_hist.append(njn[bottom].copy())
    live._run_substeps = substeps
    del base_substeps
    rows = []
    t0 = time.perf_counter()
    for k in range(int(round(T/1e-3))):
        live.advance()
        e = R._energies(live)
        v = s.v.numpy().astype(np.float64)
        vcm = v.mean(0)
        e.update(t=(k+1)*1e-3, x_cm=float(st.xcm[0]), N=float(-live.last_contact["plane_wrench"][0, 2]),
                 W_g=st.W_g, KE_vib=e["KE"]-0.5*M*float(vcm @ vcm), vcm_x=float(vcm[0]))
        e["B"] = e["KE"]+e["psi"]+e["Wb"]-st.W_g-e["W_ext"]-(e["W_n"]+e["W_t"])
        e["contact_frac_step"] = float(np.mean(step_cf)) if step_cf else 0.0
        step_cf.clear()
        rows.append(e)
        if e["invalid"]:
            break
    t = np.array([q["t"] for q in rows])
    x = np.array([q["x_cm"] for q in rows])-float(np.asarray(live.scene.X, np.float64)[:, 0].mean())
    gc = G*math.cos(theta)
    sel = t >= fit0-1e-12
    c = np.polyfit(t[sel], x[sel], 2)
    a_fit = 2*c[0]
    aref = G*(math.sin(theta)-mu*math.cos(theta)) if ratio > 1 else 0.0
    N = np.array([q["N"] for q in rows])
    dxs = np.diff(np.r_[0.0, x])
    ref = mu*float(np.sum(N*dxs))
    last = rows[-1]
    i0 = int(np.argmin(np.abs(t-fit0)))
    refw = mu*float(np.sum(N[i0+1:]*dxs[i0+1:]))
    Dw = -((last["W_n"]+last["W_t"])-(rows[i0]["W_n"]+rows[i0]["W_t"]))
    S = max(q["KE"]+q["psi"] for q in rows)+max(abs(q["W_g"]) for q in rows)
    jh = np.array(jn_hist) if jn_hist else np.zeros((1, 1))
    m_ = jh.mean(0)
    inner = m_ > 0.2*m_.max() if m_.max() > 0 else np.zeros(len(m_), bool)
    fluct = float(np.mean(jh[:, inner].std(0)/m_[inner])) if inner.any() else None
    SJ = acc["S_Jn"] or 1e-300
    out = {"args": a, "law": live.solver.contact.law, "dt": live.dt, "substeps": live.substeps, "nodes": int(len(X)),
           "a_fit": a_fit, "a_ref": aref, "slip_err_gcos": (a_fit-aref)/gc if ratio > 1 else None,
           "a_over_gcos": a_fit/gc, "stick_disp_m": float(x[-1]-x[int(np.argmin(np.abs(t-0.02)))]),
           "D04_ratio_run": (-(last["W_n"]+last["W_t"]))/ref if ratio > 1 else None,
           "D04_ratio_window": Dw/refw if ratio > 1 else None,
           "split_window": {"coulomb": None, "W_n": -(last["W_n"]-rows[i0]["W_n"])/refw if ratio > 1 else None,
                            "W_t": -(last["W_t"]-rows[i0]["W_t"])/refw if ratio > 1 else None},
           "friction_efficiency": acc["S_f"]/(mu*SJ), "stick_load_share": acc["S_stickJn"]/SJ,
           "stick_loss": (mu*acc["S_stickJn"]-acc["S_stick_f"])/(mu*SJ),
           "slip_misalignment_loss": (mu*acc["S_slipJn"]-acc["S_slip_f"])/(mu*SJ),
           "lateral_over_axial": acc["S_absJty"]/max(abs(acc["S_f"]), 1e-300),
           "contact_fraction_bottom": acc["contact_frac"]/max(acc["n_sub"], 1),
           "normal_impulse_fluctuation": fluct,
           "KE_vib_end_over_S": last["KE_vib"]/S, "KE_vib_mean_window_over_S": float(np.mean([q["KE_vib"] for q in rows[i0:]]))/S,
           "books_max_rel": max(abs(q["B"]) for q in rows)/S, "invalid": int(max(q["invalid"] for q in rows)),
           "pen_max_rel_s": max(q["pen"] for q in rows)/live.scene.spacing,
           "wall_s": time.perf_counter()-t0}
    if refw and ratio > 1:
        out["energy_window_over_ref"] = {
            "dKE_vib": (last["KE_vib"]-rows[i0]["KE_vib"])/refw, "dPsi": (last["psi"]-rows[i0]["psi"])/refw,
            "dKE_cm": ((last["KE"]-last["KE_vib"])-(rows[i0]["KE"]-rows[i0]["KE_vib"]))/refw,
            "W_g": (last["W_g"]-rows[i0]["W_g"])/refw}
    if a.get("ts") == "1":
        out["series"] = {k: [q[k] for q in rows] for k in ("t", "x_cm", "KE", "KE_vib", "psi", "W_n", "W_t", "W_g", "P_n", "damping", "coulomb",
                                                            "N", "contact_frac_step", "B")}
    print(json.dumps(out, default=float), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
