"""Formal runner of ``configs/rubber/rubber_contact_tests_v1.json`` (P2-4, ADR 0023 §22.6), single env, Warp CPU.

Runs RUBBER-CONT-01 / -04 (INC), -02 (PEN), -03 (IMP), -05 (PINCH), -07 (COUPLE); -06 is NOT_IMPLEMENTED_V1, -08 / -09
need the batched contact path (NOT_RUN here), -10 is the pytest file ``tests/test_rubber_contact.py``, -11 is the GPU
cost script. Thresholds are read from the pinned protocol; the protocol sha256 is checked at start. Every job writes
its raw series to ``<out>/jobs/<name>.json``; ``summary.json`` holds every criterion with its measured value and verdict
(PASS / FAIL; NOT_RUN where stated). Loads are ramped and updated every substep (protocol ``common.loading``).

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/run_contact_v1.py \
        --out artifacts/rl_ready/p2_4/<dir> [--jobs N] [--only INC,PEN,IMP,PINCH,COUPLE] [--smoke]

``--smoke`` shortens every scene to a few control steps (label SMOKE; never a formal result).
"""
import argparse
import hashlib
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
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_tests_v1.json"
PROTOCOL_SHA = "4d3dc68fc50aeeb247c2b544c65312fd2a651a90a99eb394b11f3fa478b46e4c"
G = 9.81
SMOKE = False


def _cfg():
    return json.loads(PROTOCOL.read_text())


def _ramp(t, T):
    return 0.5*(1-math.cos(math.pi*t/T)) if t < T else 1.0


def _material(kind, **friction):
    from robotic_cutting_sim.rubber_material import MaterialValues
    c = _cfg()["common"]
    m = c["material_elastic"] if kind == "elastic" else c["material_prony_P2"]
    prony = tuple((t["g"], t["tau_s"]) for t in m.get("terms", ()))
    return MaterialValues(mu0=m["mu0_pa"], kappa=m["bulk_ratio_beta"]*m["mu0_pa"], density=m["density_kg_m3"],
                          prony=prony, **friction)


def _live(spec, material, control_dt, gravity, contact, cfl=0.5, substeps=None):
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    o = _cfg()["common"]["options"]
    return LiveRubberMaterial(spec, material=material, control_dt=control_dt, cfl=cfl, substeps=substeps,
                              gravity=gravity, stable_step_guard=o["stable_step_guard"], dirichlet=o["dirichlet"],
                              constraint_work=o["constraint_work"], traction_work=o["traction_work"],
                              state_precision=o["state_precision"],
                              visco_energy=o["visco_energy_with_prony"] if material.prony else "off_v1",
                              contact=contact, strict=False)


class Stepper:
    """One control step of a LiveRubberMaterial with per-substep loads ``loads(t) -> (gravity [3], {traction: F})``
    and the kick-paired gravity work W_g = sum_n M g_n . (dx_cm,n-1 + dx_cm,n) / 2 (float64 shadow positions)."""

    def __init__(self, live, loads):
        self.live, self.loads = live, loads
        s = live.solver
        self.M = float(s.mass_value)*s.n
        self.t = 0.0
        self.W_g = 0.0
        self.prev_dx = np.zeros(3)
        self.xcm = s.x64.numpy().mean(0).copy()
        live._run_substeps = self._substeps

    def _substeps(self):
        import warp as wp
        s = self.live.solver
        for _ in range(s.substeps):
            g, tr = self.loads(self.t+0.5*s.dt)
            s.gravity = wp.vec3(*map(float, g))
            for k, f in (tr or {}).items():
                s.set_traction(k, f)
            s.step()
            x = s.x64.numpy().mean(0)
            dx = x-self.xcm
            self.W_g += self.M*float(np.dot(np.asarray(g, float), 0.5*(self.prev_dx+dx)))
            self.prev_dx, self.xcm = dx, x.copy()
            self.t += s.dt

    def advance(self, **kw):
        w = self.live.advance(**kw)
        return w


def _energies(live):
    d, v, c = live.last_diagnostics, live.last_visco, live.last_contact
    s = live.solver
    psi = v["psi_eq"] if v else d["elastic"]
    Wb = v["branch_work"] if v else 0.0
    psiv = v["psi_visc_stored"] if v else 0.0
    return {"KE": d["kinetic_particle"], "psi": psi, "Wb": Wb, "psi_v": psiv, "W_ext": float(s.work.numpy()[0]),
            "W_n": c["normal_work_material"], "W_t": c["tangential_work_material"], "P_n": c["normal_work_obstacle"],
            "P_t": c["tangential_work_obstacle"], "coulomb": c["coulomb_dissipation"],
            "damping": c["damping_dissipation"], "pen": c["max_penetration_m"], "events": c["impulse_events"],
            "stick": c["stick_events"], "slip": c["slip_events"], "invalid": int(live.invalid_reason),
            "cfl_eff_max": d["cfl_eff_max"]}


def _spec(**kw):
    from robotic_cutting_sim import rubber_scene as rs
    return rs.RubberStripSpec(**kw)


# ----------------------------------------------------------------------------- INC (-01, -04)

def job_inc(name, mu, ratio, material_kind, cfl):
    from robotic_cutting_sim.rubber_contact import ContactSpec
    sc = _cfg()["scenes"]["INC"]
    theta = math.atan(ratio*mu)
    gvec = np.array([G*math.sin(theta), 0.0, -G*math.cos(theta)])
    spec = _spec(length=0.016, width=0.008, thickness=0.004, n_t=3, case="free")
    live = _live(spec, _material(material_kind, friction_table=mu), sc["control_dt_s"], (0, 0, 0),
                 ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),)), cfl=cfl)
    T, Tr, cdt = sc["duration_s"], sc["ramp_s"], sc["control_dt_s"]
    if SMOKE:
        T = 5*cdt
    st = Stepper(live, lambda t: (gvec*_ramp(t, Tr), None))
    rows = []
    t0 = time.perf_counter()
    for k in range(int(round(T/cdt))):
        st.advance()
        e = _energies(live)
        e.update(t=(k+1)*cdt, x_cm=float(st.xcm[0]), N=float(-live.last_contact["plane_wrench"][0, 2]),
                 Ft=float(live.last_contact["plane_wrench"][0, 0]), W_g=st.W_g)
        e["B"] = e["KE"]+e["psi"]+e["Wb"]-st.W_g-e["W_ext"]-(e["W_n"]+e["W_t"])
        rows.append(e)
    return {"name": name, "scene": "INC", "mu": mu, "ratio": ratio, "theta_rad": theta, "material": material_kind,
            "cfl": cfl, "particles": int(live.scene.n), "spacing_m": live.scene.spacing, "dt_s": live.dt,
            "substeps": live.substeps, "x_cm0": float(np.asarray(live.scene.X, np.float64)[:, 0].mean()),
            "wall_s": time.perf_counter()-t0, "rows": rows, "receipt_contact": live.solver.contact.receipt()}


# ----------------------------------------------------------------------------- PEN (-02)

def job_pen(name, n_t, cfl, damped=True):
    from robotic_cutting_sim.rubber_contact import ContactSpec
    sc = _cfg()["scenes"]["PEN"]
    t = 0.004
    dx = t/n_t
    s_ = dx/2
    spec = _spec(length=0.008, width=0.006, thickness=t, n_t=n_t, case="free",
                 traction_regions=(("top", (-1, -1, t-s_), (1, 1, 1)),))
    extra = {} if damped else {"damping_ratio": 0.0, "normal_damping_ratio": 0.0}
    live = _live(spec, _material("P2", friction_table=0.0), sc["control_dt_s"], (0, 0, 0),
                 ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),), **extra), cfl=cfl)
    F = sc["p_top_pa"]*0.008*0.006
    Tr = 0.02
    st = Stepper(live, lambda tt: (np.array([0.0, 0.0, -G])*_ramp(tt, Tr), {0: np.array([0.0, 0.0, -F*_ramp(tt, Tr)])}))
    T, cdt = sc["duration_s"], sc["control_dt_s"]
    if SMOKE:
        T = 5*cdt
    w0, w1 = sc["measure_window_s"]
    s = live.solver
    X = s.node_X.numpy()
    rows, uz, fz, pen = [], [], [], []
    t0 = time.perf_counter()
    for k in range(int(round(T/cdt))):
        st.advance()
        e = _energies(live)
        tk = (k+1)*cdt
        e.update(t=tk, N=float(-live.last_contact["plane_wrench"][0, 2]))
        e["B"] = e["KE"]+e["psi"]+e["Wb"]-st.W_g-e["W_ext"]-(e["W_n"]+e["W_t"])
        rows.append(e)
        if (w0 <= tk <= w1+1e-12) or SMOKE:
            uz.append(s.ct_node_u.numpy()[:, 2].copy())
            fz.append(e["N"])
            pen.append(e["pen"])
    gm = s.gm.numpy().astype(np.float64)
    face = np.flatnonzero((np.abs(X[:, 2]) < 1e-12) & (gm > 0))
    inner = face[(X[face, 0] > dx-1e-9) & (X[face, 0] < 0.008-dx+1e-9) & (X[face, 1] > dx-1e-9)
                 & (X[face, 1] < 0.006-dx+1e-9)]
    uzm = np.mean(uz, 0)
    kn = 0.25*gm/s.dt**2
    A = (np.where((X[:, 0] > 1e-9) & (X[:, 0] < 0.008-1e-9), 1.0, 0.5)
         * np.where((X[:, 1] > 1e-9) & (X[:, 1] < 0.006-1e-9), 1.0, 0.5)*dx*dx)
    N = float(np.mean(fz))
    p = N/(0.008*0.006)
    return {"name": name, "scene": "PEN", "n_t": n_t, "cfl": cfl, "damped": damped, "particles": int(s.n),
            "spacing_m": live.scene.spacing, "dx_m": dx, "dt_s": live.dt, "weight_plus_load_N": s.mass_value*s.n*G+F,
            "reaction_mean_N": N, "pen_max_window_m": float(max(pen)), "pen_mean_inner_m": float(np.mean(-uzm[inner])),
            "inner_nodes": int(len(inner)), "spring_sum_over_reaction": float(np.sum(-uzm[face]*kn[face])/N),
            "inner_ratio_to_undamped_law": float(np.mean(-uzm[inner]/(p*A[inner]/kn[inner]))),
            "wall_s": time.perf_counter()-t0, "rows": rows}


# ----------------------------------------------------------------------------- IMP (-03)

def job_imp(name, material_kind, n_t, cfl):
    from robotic_cutting_sim.rubber_contact import ContactSpec
    sc = _cfg()["scenes"]["IMP"]
    V = sc["V_m_s"]
    spec = _spec(length=0.04, width=0.004, thickness=0.004, n_t=n_t, case="free")
    m = _material(material_kind, friction_table=0.0)
    live = _live(spec, m, sc["control_dt_s"], (0, 0, 0), ContactSpec(planes=((1.0, 0.0, 0.0, 0.0),)), cfl=cfl)
    s = live.solver
    s.set_initial_velocity((-V, 0.0, 0.0))
    E0 = 0.5*float(s.mass_value)*s.n*V*V
    T, cdt = sc["duration_s"], sc["control_dt_s"]
    if SMOKE:
        T = 10*cdt
    rows = []
    t0 = time.perf_counter()
    for k in range(int(round(T/cdt))):
        live.advance()
        e = _energies(live)
        e.update(t=(k+1)*cdt, vcm=float(s.v.numpy()[:, 0].astype(np.float64).mean()))
        e["B"] = e["KE"]+e["psi"]+e["Wb"]-e["W_ext"]-(e["W_n"]+e["W_t"])-E0
        rows.append(e)
    E = m.young_effective()
    return {"name": name, "scene": "IMP", "material": material_kind, "n_t": n_t, "cfl": cfl, "particles": int(s.n),
            "spacing_m": live.scene.spacing, "dt_s": live.dt, "E0_J": E0, "V_m_s": V, "L_m": 0.04,
            "c0_m_s": math.sqrt(E/m.density), "tc_ref_s": 2*0.04/math.sqrt(E/m.density),
            "wall_s": time.perf_counter()-t0, "rows": rows}


# ----------------------------------------------------------------------------- PINCH (-05)

def job_pinch(name, depth):
    from robotic_cutting_sim.rubber_contact import ContactFrame, ContactSpec
    sc = _cfg()["scenes"]["PINCH"]
    L, W, t = 0.03, 0.01, 0.004
    spec = _spec(length=L, width=W, thickness=t, n_t=3, case="A")
    top = ContactFrame(kind="pad", name="top", origin=(L-0.003, W/2, t+0.001),
                       boxes=(((L-0.003, W/2, t+0.001), (0.005, W/2+0.002, 0.001)),))
    bot = ContactFrame(kind="pad", name="bottom", origin=(L-0.003, W/2, -0.001),
                       boxes=(((L-0.003, W/2, -0.001), (0.005, W/2+0.002, 0.001)),))
    live = _live(spec, _material("elastic", friction_pad=sc["mu_pad"]), sc["control_dt_s"], (0, 0, 0),
                 ContactSpec(frames=(top, bot)))
    s = live.solver
    cdt, Tr, Th, vp, Tp = (sc["control_dt_s"], sc["squeeze_ramp_s"], sc["hold_s"], sc["pull_speed_m_s"],
                           sc["pull_duration_s"])
    T = Tr+Th+Tp
    if SMOKE:
        T = 5*cdt
    rows = []
    t0 = time.perf_counter()
    for k in range(int(round(T/cdt))):
        te = (k+1)*cdt
        sq = depth*_ramp(te, Tr)
        px = vp*max(0.0, te-Tr-Th)
        w = live.advance(frames={0: {"translation": np.array([px, 0.0, -sq])},
                                 1: {"translation": np.array([px, 0.0, sq])}})
        bw = live.last_contact["box_wrench"]
        e = _energies(live)
        e.update(t=te, pull_x=px, squeeze=sq, tip_x=float(live.markers()[-1][0]), F_top=bw[0, 0].tolist(),
                 F_bot=bw[1, 0].tolist(), clamp=np.asarray(w)[0].tolist())
        rows.append(e)
    return {"name": name, "scene": "PINCH", "depth_m": depth, "mu_pad": sc["mu_pad"], "particles": int(s.n),
            "spacing_m": live.scene.spacing, "dt_s": live.dt, "wall_s": time.perf_counter()-t0, "rows": rows}


# ----------------------------------------------------------------------------- COUPLE (-07)

def _smoother(x):
    x = min(max(x, 0.0), 1.0)
    return x*x*x*(x*(6*x-15)+10), 30*x*x*(x-1)**2


def job_couple(name, L, hand, Tm, arm):
    sc = _cfg()["scenes"]["COUPLE"]
    h = sc["envelope"]["hand"][hand]
    mh, kh = h["m_h_kg"], h["k_h_n_m"]
    ch = 2*0.7*math.sqrt(kh*mh)
    spec = _spec(length=L, width=0.01, thickness=0.004, n_t=3, case="A")
    m = _material("elastic")
    live = _live(spec, m, 1e-3, (0, 0, 0), None)          # both arms: 1 ms material control steps, the same dt
    sub1 = live.substeps
    s = live.solver
    interval = 20 if arm == "A" else 1
    T = 0.6 if not SMOKE else 0.04
    A0 = 0.005
    hdt = 5e-5
    z, vz = 0.0, 0.0
    F_applied = 0.0
    E_delay = 0.0
    rows = []
    t0 = time.perf_counter()
    nsteps = int(round(T/1e-3))
    k = 0
    while k < nsteps:
        # hand ODE over the next interval with the handed-off force (previous interval's average material force)
        z_start = z
        for j in range(int(round(interval*1e-3/hdt))):
            tt = k*1e-3+(j+0.5)*hdt
            sp, dsp = _smoother(tt/Tm)
            zc, dzc = A0*sp, A0*dsp/Tm
            vz += hdt*(kh*(zc-z)+ch*(dzc-vz)+F_applied)/mh
            z += hdt*vz
        z_end = z
        forces = []
        for j in range(interval):
            zj = z_start+(z_end-z_start)*(j+1)/interval    # the material follows the measured pose linearly
            w = live.advance(motions={0: {"translation": np.array([0.0, 0.0, zj])}})
            forces.append(float(np.asarray(w)[0, 2]))
            d = live.last_diagnostics
            rows.append({"t": (k+j+1)*1e-3, "z_hand": zj, "z_tip": float(live.markers()[-1][2]),
                         "F_material": forces[-1], "F_applied": F_applied, "KE": d["kinetic_particle"],
                         "psi": d["elastic"], "W_ext": float(s.work.numpy()[0]), "invalid": int(live.invalid_reason)})
        F_mat = float(np.mean(forces))
        E_delay += (F_applied-F_mat)*(z_end-z_start)
        F_applied = F_mat
        k += interval
    return {"name": name, "scene": "COUPLE", "L_m": L, "hand": hand, "m_h": mh, "k_h": kh, "T_m": Tm, "arm": arm,
            "interval_ms": interval, "substeps_per_ms": sub1, "dt_s": live.dt, "particles": int(s.n),
            "E_delay_J": E_delay, "wall_s": time.perf_counter()-t0, "rows": rows}


# ----------------------------------------------------------------------------- job table

def jobs(cfg, only):
    out = {}
    inc = cfg["scenes"]["INC"]
    if "INC" in only:
        for mu in inc["mu_set"]:
            for r in inc["stick_ratios_tan_theta_over_mu"]+inc["slip_ratios_tan_theta_over_mu"]:
                out[f"INC_mu{mu}_r{r}"] = (job_inc, (mu, r, "elastic", 0.5))
        v = inc["visco_stick_row"]
        out[f"INC_P2_mu{v['mu']}_r{v['ratio']}"] = (job_inc, (v["mu"], v["ratio"], "P2", 0.5))
        b = inc["books_order_row"]
        out[f"INC_mu{b['mu']}_r{b['ratio']}_cfl0.25"] = (job_inc, (b["mu"], b["ratio"], "elastic", 0.25))
    if "PEN" in only:
        for r in cfg["scenes"]["PEN"]["resolutions"]:
            out[f"PEN_nt{r['n_t']}_cfl{r['cfl']}"] = (job_pen, (r["n_t"], r["cfl"], True))
        lc = cfg["scenes"]["PEN"]["law_check_variant"]
        out[f"PEN_nt{lc['n_t']}_cfl{lc['cfl']}_undamped"] = (job_pen, (lc["n_t"], lc["cfl"], False))
    if "IMP" in only:
        for mk in ("elastic", "P2"):
            for r in cfg["scenes"]["IMP"]["runs"]:
                out[f"IMP_{mk}_nt{r['n_t']}_cfl{r['cfl']}"] = (job_imp, (mk, r["n_t"], r["cfl"]))
    if "PINCH" in only:
        for d in cfg["scenes"]["PINCH"]["squeeze_depths_m"]:
            out[f"PINCH_d{d:.0e}"] = (job_pinch, (d,))
    if "COUPLE" in only:
        env = cfg["scenes"]["COUPLE"]["envelope"]
        for L in env["strip_length_m"]:
            for hand in env["hand"]:
                for Tm in env["T_m_s"]:
                    for arm in ("A", "B"):
                        out[f"COUPLE_L{L}_{hand}_Tm{Tm}_{arm}"] = (job_couple, (L, hand, Tm, arm))
    return out


def _run(item):
    name, (fn, args) = item
    try:
        return name, fn(name, *args)
    except Exception as exc:                                    # recorded, never hidden
        import traceback
        return name, {"name": name, "error": repr(exc), "traceback": traceback.format_exc()}


# ----------------------------------------------------------------------------- evaluation

def _v(ok):
    return "PASS" if ok else "FAIL"


def evaluate(cfg, R):
    th = {k: cfg["tests"][k].get("thresholds", {}) for k in cfg["tests"]}
    out = {}
    errors = {n: r["error"] for n, r in R.items() if "error" in r}
    # -01 / -04
    inc = {n: r for n, r in R.items() if r.get("scene") == "INC" and "error" not in r}
    if inc:
        t1, t4 = th["RUBBER-CONT-01"], th["RUBBER-CONT-04"]
        sc = cfg["scenes"]["INC"]
        crit = {"a_stick": [], "b_slip": [], "c_onset": [], "d_books": [], "e_validity": [], "a_dissipation": []}
        fits = {}
        for n, r in inc.items():
            rows = r["rows"]
            t = np.array([x["t"] for x in rows])
            x = np.array([x["x_cm"] for x in rows])-r["x_cm0"]
            gc = G*math.cos(r["theta_rad"])
            sel = (t >= sc["fit_window_s"][0]-1e-12)
            if sel.sum() < 3:                                       # SMOKE runs only (shortened scenes)
                sel = np.ones_like(sel)
            c = np.polyfit(t[sel], x[sel], 2)
            a = 2*c[0]
            S = max(q["KE"]+q["psi"] for q in rows)+max(abs(q["W_g"]) for q in rows)
            Bmax = max(abs(q["B"]) for q in rows)
            row = {"run": n, "a_fit": a, "a_over_gcos": a/gc, "books_max_rel": Bmax/S, "S_J": S}
            crit["d_books"].append({**row, "ok": Bmax <= t1["books_rel"]*S})
            crit["e_validity"].append({"run": n, "invalid": max(q["invalid"] for q in rows),
                                       "ok": max(q["invalid"] for q in rows) == 0})
            if r["ratio"] < 1:
                i20 = int(np.argmin(np.abs(t-sc["hold_window_s"][0])))
                disp = x[-1]-x[i20]
                crit["a_stick"].append({**row, "disp_m": disp, "disp_rel_s": disp/r["spacing_m"],
                                        "ok": abs(disp) <= t1["stick_disp_rel"]*r["spacing_m"]
                                        and abs(a) <= t1["stick_acc_rel"]*gc})
            else:
                aref = G*(math.sin(r["theta_rad"])-r["mu"]*math.cos(r["theta_rad"]))
                crit["b_slip"].append({**row, "a_ref": aref, "err_over_gcos": (a-aref)/gc,
                                       "ok": abs(a-aref) <= t1["slip_acc_rel"]*gc})
                if r["cfl"] == 0.5 and r["material"] == "elastic":
                    fits.setdefault(r["mu"], []).append((math.tan(r["theta_rad"]), a/gc))
                # -04
                N = np.array([q["N"] for q in rows])
                dxs = np.diff(np.r_[0.0, x])
                ref = r["mu"]*float(np.sum(N*dxs))
                last = rows[-1]
                D = -(last["W_n"]+last["W_t"])
                crit["a_dissipation"].append({"run": n, "D_contact_J": D, "mu_sum_N_dx_J": ref, "ratio": D/ref,
                                              "split": {"coulomb": last["coulomb"]/ref, "dashpots": last["damping"]/ref,
                                                        "normal_work": -last["W_n"]/ref,
                                                        "tangential_work": -last["W_t"]/ref},
                                              "ok": abs(D/ref-1) <= t4["dissipation_rel"]})
        for mu, pts in fits.items():
            tt, aa = np.array(pts).T
            slope, icpt = np.polyfit(tt, aa, 1)
            mu_eff = -icpt/slope
            crit["c_onset"].append({"mu": mu, "slope": slope, "mu_eff": mu_eff, "rel": mu_eff/mu-1,
                                    "ok": abs(mu_eff/mu-1) <= t1["onset_rel"]})
        # books order
        b = sc["books_order_row"]
        n5, n25 = f"INC_mu{b['mu']}_r{b['ratio']}", f"INC_mu{b['mu']}_r{b['ratio']}_cfl0.25"
        if n5 in inc and n25 in inc:
            r5 = next(q for q in crit["d_books"] if q["run"] == n5)
            r25 = next(q for q in crit["d_books"] if q["run"] == n25)
            ok = (r25["books_max_rel"] <= t1["ratio_max"]*r5["books_max_rel"]
                  or max(r5["books_max_rel"], r25["books_max_rel"]) <= t1["floor_rel"])
            crit["d_books"].append({"run": "order", "ratio": r25["books_max_rel"]/max(r5["books_max_rel"], 1e-300),
                                    "ok": ok})
        v01 = {k: _v(all(q["ok"] for q in crit[k])) if crit[k] else "NOT_RUN"
               for k in ("a_stick", "b_slip", "c_onset", "d_books", "e_validity")}
        out["RUBBER-CONT-01"] = {"criteria": v01, "detail": {k: crit[k] for k in v01},
                                 "verdict": _v(all(x == "PASS" for x in v01.values()))}
        out["RUBBER-CONT-04"] = {"criteria": {"a_total": _v(all(q["ok"] for q in crit["a_dissipation"]))},
                                 "detail": crit["a_dissipation"],
                                 "verdict": _v(all(q["ok"] for q in crit["a_dissipation"]))}
    # -02
    pen = {n: r for n, r in R.items() if r.get("scene") == "PEN" and "error" not in r}
    if pen:
        t2 = th["RUBBER-CONT-02"]
        get = lambda n_t, cfl, d=True: next((r for r in pen.values() if r["n_t"] == n_t and r["cfl"] == cfl  # noqa
                                             and r["damped"] == d), None)
        a = [{"run": r["name"], "pen_max_rel_s": r["pen_max_window_m"]/r["spacing_m"],
              "ok": r["pen_max_window_m"] <= t2["pen_max_rel"]*r["spacing_m"]} for r in pen.values()]
        e = [{"run": r["name"], "rel": r["reaction_mean_N"]/r["weight_plus_load_N"]-1,
              "ok": abs(r["reaction_mean_N"]/r["weight_plus_load_N"]-1) <= t2["balance_rel"]} for r in pen.values()]
        r2, r2h, r4, ru = get(2, 0.5), get(2, 0.25), get(4, 0.5), get(2, 0.5, False)
        crit = {"a_bound": a, "e_balance": e}
        if r2 and r2h:
            q = r2h["pen_mean_inner_m"]/r2["pen_mean_inner_m"]
            crit["b_dt"] = [{"ratio": q, "ok": t2["dt_ratio_band"][0] <= q <= t2["dt_ratio_band"][1]}]
        if r2 and r4:
            q = r4["pen_mean_inner_m"]/r2["pen_mean_inner_m"]
            crit["c_dx"] = [{"ratio": q, "ok": t2["dx_ratio_band"][0] <= q <= t2["dx_ratio_band"][1]}]
        if ru:
            crit["d_law"] = [{"spring_sum_over_reaction": ru["spring_sum_over_reaction"],
                              "inner_ratio": ru["inner_ratio_to_undamped_law"],
                              "ok": abs(ru["spring_sum_over_reaction"]-1) <= t2["law_rel"]
                              and abs(ru["inner_ratio_to_undamped_law"]-1) <= t2["law_rel_node"]}]
        v = {k: _v(all(q["ok"] for q in crit[k])) if k in crit else "NOT_RUN"
             for k in ("a_bound", "b_dt", "c_dx", "d_law", "e_balance")}
        out["RUBBER-CONT-02"] = {"criteria": v, "detail": crit, "verdict": _v(all(x == "PASS" for x in v.values()))}
    # -03
    imp = {n: r for n, r in R.items() if r.get("scene") == "IMP" and "error" not in r}
    if imp:
        t3 = th["RUBBER-CONT-03"]
        crit = {"a_books": [], "b_contact_time": [], "c_attribution": [], "d_convergence": [], "e_penetration": []}
        D = {}
        for n, r in imp.items():
            rows, E0 = r["rows"], r["E0_J"]
            Bmax = max(abs(q["B"]) for q in rows)
            Bend = abs(rows[-1]["B"])
            crit["a_books"].append({"run": n, "max_rel": Bmax/E0, "end_rel": Bend/E0,
                                    "ok": Bmax <= t3["books_rel"]*E0 and Bend <= t3["books_end_rel"]*E0})
            ev = [i for i, q in enumerate(rows) if q["events"] > 0]
            last = rows[-1]
            Dc = -(last["W_n"]+last["W_t"])
            Db = last["Wb"]-last["psi_v"]
            separated = rows[-1]["events"] == 0
            D[n] = Dc
            crit["c_attribution"].append({"run": n, "D_contact_rel": Dc/E0, "normal_rel": -last["W_n"]/E0,
                                          "tangential_rel": -last["W_t"]/E0, "D_b_rel": Db/E0,
                                          "E_end_rel": (last["KE"]+last["psi"])/E0,
                                          "vcm_end_over_V": -last["vcm"]/r["V_m_s"], "separated": separated,
                                          "ok": separated and Dc >= 0 and Db >= -1e-12*E0
                                          and (Dc <= t3["contact_loss_max"]*E0 if r["n_t"] == 4 else True)})
            crit["e_penetration"].append({"run": n, "pen_rel_s": max(q["pen"] for q in rows)/r["spacing_m"],
                                          "ok": max(q["pen"] for q in rows) <= t3["pen_max_rel"]*r["spacing_m"]})
            if r["material"] == "elastic" and r["n_t"] == 4 and r["cfl"] == 0.5 and ev:
                tc = (ev[-1]+1-ev[0])*cfg["scenes"]["IMP"]["control_dt_s"]
                crit["b_contact_time"].append({"run": n, "t_c": tc, "ref": r["tc_ref_s"], "rel": tc/r["tc_ref_s"]-1,
                                               "ok": abs(tc/r["tc_ref_s"]-1) <= t3["tc_rel"]})
        for mk in ("elastic", "P2"):
            a, b = f"IMP_{mk}_nt4_cfl0.5", f"IMP_{mk}_nt4_cfl0.25"
            if a in imp and b in imp:
                ra = next(q for q in crit["a_books"] if q["run"] == a)
                rb = next(q for q in crit["a_books"] if q["run"] == b)
                ok = rb["max_rel"] <= t3["ratio_max"]*ra["max_rel"] or max(ra["max_rel"], rb["max_rel"]) <= t3["floor_rel"]
                crit["a_books"].append({"run": f"order_{mk}", "ratio": rb["max_rel"]/ra["max_rel"], "ok": ok})
        a, b = "IMP_elastic_nt4_cfl0.5", "IMP_elastic_nt2_cfl0.5"
        if a in D and b in D:
            crit["d_convergence"].append({"ratio": D[a]/D[b], "ok": D[a] <= t3["conv_ratio_max"]*D[b]})
        v = {k: _v(all(q["ok"] for q in crit[k])) if crit[k] else "NOT_RUN" for k in crit}
        out["RUBBER-CONT-03"] = {"criteria": v, "detail": crit, "verdict": _v(all(x == "PASS" for x in v.values()))}
    # -05
    pin = {n: r for n, r in R.items() if r.get("scene") == "PINCH" and "error" not in r}
    if pin:
        t5 = th["RUBBER-CONT-05"]
        sc = cfg["scenes"]["PINCH"]
        crit = {"a_slip_ratio": [], "b_cone": [], "c_grip_scaling": [], "d_balance": [], "e_penetration": []}
        Ns, ratios = [], []
        for n, r in sorted(pin.items(), key=lambda kv: kv[1]["depth_m"]):
            rows = r["rows"]
            T_end = sc["squeeze_ramp_s"]+sc["hold_s"]+sc["pull_duration_s"]
            win = [q for q in rows if q["t"] > T_end-0.01+1e-12] or rows[-3:]      # (SMOKE: last rows)
            Nt = lambda q: q["F_top"][2]-q["F_bot"][2]                    # noqa: E731  (top pushed +z, bottom -z)
            Tx = lambda q: q["F_top"][0]+q["F_bot"][0]                    # noqa: E731
            mu = r["mu_pad"]
            ratio = float(np.mean([abs(Tx(q))/(mu*Nt(q)) if Nt(q) > 0 else 0.0 for q in win]))
            # slip onset from the tip-marker velocity during the pull
            pull = [q for q in rows if q["t"] > sc["squeeze_ramp_s"]+sc["hold_s"]+1e-12] or rows[-3:]
            vtip = np.diff([q["tip_x"] for q in pull])/sc["control_dt_s"]
            onset = next((pull[i+1]["t"] for i, vv in enumerate(vtip) if vv < 0.5*sc["pull_speed_m_s"]), None)
            crit["a_slip_ratio"].append({"run": n, "ratio": ratio, "N_window_N": float(np.mean([Nt(q) for q in win])),
                                         "slip_onset_s": onset, "tip_speed_window_m_s": float(np.mean(vtip[-20:])),
                                         "ok": t5["slip_ratio_band"][0] <= ratio <= t5["slip_ratio_band"][1]})
            worst = max((abs(Tx(q))/(mu*Nt(q)) for q in rows if Nt(q) > 0), default=0.0)
            crit["b_cone"].append({"run": n, "max_ratio": worst, "ok": worst <= t5["cone_tol"]})
            bal = float(np.mean([abs(q["clamp"][0]+Tx(q)) for q in win])/max(np.mean([abs(Tx(q)) for q in win]), 1e-300))
            crit["d_balance"].append({"run": n, "rel": bal, "ok": bal <= t5["balance_rel"]})
            pmax = max(q["pen"] for q in rows)
            crit["e_penetration"].append({"run": n, "pen_rel_s": pmax/r["spacing_m"],
                                          "ok": pmax <= t5["pen_max_rel"]*r["spacing_m"]})
            Ns.append(crit["a_slip_ratio"][-1]["N_window_N"])
            ratios.append(ratio)
        if len(Ns) >= 2:
            distinct = all(Ns[i+1] >= 1.3*Ns[i] for i in range(len(Ns)-1))
            spread = (max(ratios)-min(ratios))/np.mean(ratios)
            crit["c_grip_scaling"].append({"N": Ns, "ratios": ratios, "spread": spread,
                                           "ok": distinct and spread <= t5["scaling_rel"]})
        v = {k: _v(all(q["ok"] for q in crit[k])) if crit[k] else "NOT_RUN" for k in crit}
        out["RUBBER-CONT-05"] = {"criteria": v, "detail": crit, "verdict": _v(all(x == "PASS" for x in v.values()))}
    # -07
    cpl = {n: r for n, r in R.items() if r.get("scene") == "COUPLE" and "error" not in r}
    if cpl:
        t7 = th["RUBBER-CONT-07"]
        points, books = [], []
        for n, r in cpl.items():
            rows = r["rows"]
            S = max(q["KE"]+q["psi"] for q in rows)+max(abs(q["W_ext"]) for q in rows)
            Bmax = max(abs(q["KE"]+q["psi"]-q["W_ext"]) for q in rows)
            books.append({"run": n, "max_rel": Bmax/S, "ok": Bmax <= t7["books_rel"]*S})
        keys = sorted({n.rsplit("_", 1)[0] for n in cpl})
        for key in keys:
            A, B = cpl.get(key+"_A"), cpl.get(key+"_B")
            if not (A and B):
                points.append({"point": key, "classified": False})
                continue
            zA = np.array([q["z_tip"] for q in A["rows"]])
            zB = np.array([q["z_tip"] for q in B["rows"]])
            hA = np.array([q["z_hand"] for q in A["rows"]])
            hB = np.array([q["z_hand"] for q in B["rows"]])
            e_tip = float(np.sqrt(np.mean((zA-zB)**2))/np.ptp(zB))
            e_hand = float(np.sqrt(np.mean((hA-hB)**2))/np.ptp(hB))
            t = np.array([q["t"] for q in B["rows"]])

            def zeta(z):
                sel = t >= A["T_m"]+0.1
                y = z[sel]-np.mean(z[sel])
                pk = [i for i in range(1, len(y)-1) if y[i] > y[i-1] and y[i] >= y[i+1] and y[i] > 0]
                if len(pk) < 3:
                    return None
                amps = y[pk]
                dec = np.mean(np.log(amps[:-1]/amps[1:]))
                return dec/math.sqrt(4*math.pi**2+dec**2)
            zA_, zB_ = zeta(zA), zeta(zB)
            d_decay = None if (zA_ is None or zB_ is None or zB_ == 0) else abs(zA_/zB_-1)
            Emech = max(q["KE"]+q["psi"] for q in B["rows"])
            e_energy = abs(A["E_delay_J"])/Emech
            ok_decay = d_decay is None or d_decay <= t7["decay_rel"]
            label = ("coupling_delay_acceptable" if (e_tip <= t7["tip_rel"] and ok_decay and e_energy <= t7["energy_rel"])
                     else "coupling_delay_significant")
            points.append({"point": key, "classified": True, "e_tip": e_tip, "e_hand": e_hand,
                           "zeta_A": zA_, "zeta_B": zB_, "d_decay": d_decay if d_decay is not None else "NOT_RESOLVED",
                           "E_delay_A_rel": e_energy, "E_delay_B_rel": abs(B["E_delay_J"])/Emech, "label": label})
        v = {"a_classified": _v(all(p["classified"] for p in points)), "b_material_books": _v(all(b["ok"] for b in books)),
             "c_no_material_attribution": "PASS (by construction: the material parameters are fixed; the delay error is "
                                          "computed on the hand side only)"}
        out["RUBBER-CONT-07"] = {"criteria": v, "points": points, "books": books,
                                 "verdict": _v(v["a_classified"] == "PASS" and v["b_material_books"] == "PASS")}
    for tid in ("RUBBER-CONT-06",):
        out[tid] = {"verdict": "NOT_RUN", "reason": "NOT_IMPLEMENTED_V1 (declared limitation, ADR 0023 §22.3)"}
    for tid in ("RUBBER-CONT-08", "RUBBER-CONT-09"):
        out.setdefault(tid, {"verdict": "NOT_RUN", "reason": "batched contact path not implemented in this package "
                                                             "(ADR 0023 §22.8 task list)"})
    out["errors"] = errors
    return out


def main():
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--only", default="INC,PEN,IMP,PINCH,COUPLE")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    SMOKE = a.smoke
    out = Path(a.out)
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if sha != PROTOCOL_SHA:
        raise SystemExit(f"protocol sha256 {sha} != pinned {PROTOCOL_SHA}")
    cfg = _cfg()
    if a.evaluate_only:
        R = {p.stem: json.loads(p.read_text()) for p in sorted((out/"jobs").glob("*.json"))}
        (out/"summary_reevaluated.json").write_text(json.dumps(evaluate(cfg, R), indent=1, default=float))
        return
    out.mkdir(parents=True, exist_ok=False)
    (out/"jobs").mkdir()
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa
    import warp as wp
    meta = {"label": "SMOKE (not a formal result)" if a.smoke else "FORMAL rubber_contact_tests_v1",
            "protocol_sha256": sha, "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "warp": wp.config.version, "numpy": np.__version__,
            "argv": sys.argv, "start": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "host": platform.node(),
            "cpu_count_used": a.jobs}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    table = jobs(cfg, set(a.only.split(",")))
    R = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, res in ex.map(_run, sorted(table.items(), key=lambda kv: kv[0].startswith("COUPLE"), reverse=True)):
            (out/"jobs"/f"{name}.json").write_text(json.dumps(res, default=float))
            R[name] = res
            print(name, "ERROR "+res["error"] if "error" in res else f"{res.get('wall_s', 0):.1f} s", flush=True)
    meta["end"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    meta["wall_s"] = time.perf_counter()-t0
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    summary = evaluate(cfg, R)
    (out/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for tid, v in summary.items():
        if tid.startswith("RUBBER"):
            print(tid, v.get("verdict"), v.get("criteria", v.get("reason")))


if __name__ == "__main__":
    main()
