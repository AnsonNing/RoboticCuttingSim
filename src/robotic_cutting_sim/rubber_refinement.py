"""RUBBER-PHYS-11 refinement study (ADR 0023 §14.4; protocol ``rubber_phys_tests_v4``; tasks R2.2 / R2.3, F4).

Observables (single-env reference, stable-step guard on, static_stop_v3 for the static ones):

- Q1 static sag (RUBBER-PHYS-02 type): tip deflection delta (m) of a clamped strip under gravity (float64 markers);
- Q2 first bending frequency (RUBBER-PHYS-04 type): f1 (Hz) of the strip released straight under gravity;
- Q3 nominal stress at lambda 1.5 (RUBBER-PHYS-06 type, load half of the path): grip-1 reaction / (W t) (Pa) over the
  control interval whose mid stretch is nearest 1.5; Q5 (mean gauge Jbar error, Jbar extremes) reported with it;
- Q4 large-deflection tip drop (RUBBER-PHYS-05 type, alpha 9): delta (m).

Each ``case_q*`` returns raw measurements (no verdict); ``evaluate`` applies the pre-declared v4 rule (GCI of the
production grid from the three finest grids, dt sensitivity, beta / ppc sensitivity reported) with ``rubber_gci``.
Material status: SIMULATOR_ONLY_UNCALIBRATED.
"""
import math
import time

import numpy as np

from . import rubber_gci as G
from . import rubber_reference as R
from . import rubber_scene as rs
from .live_rubber_material import LiveRubberMaterial
from .rubber_initializer import relax_to_static
from .rubber_material import MaterialValues

RHO = 1150.0
BETA_REF = 20.0


def material(mu, beta=BETA_REF, rho=RHO):
    return MaterialValues(mu0=float(mu), kappa=float(beta)*float(mu), density=float(rho))


def _live(spec, mat, control_dt, gravity, cfl, guard, options, record=False):
    return LiveRubberMaterial(spec, material=mat, control_dt=control_dt, gravity=gravity, cfl=cfl,
                              stable_step_guard=guard, strict=False, record=record, **(options or {}))


def _common(live, run, t0):
    return {"run": run, "particles": live.scene.n, "dt_s": live.dt, "substeps": live.substeps,
            "cfl_actual": live.step_rule["cfl_actual"], "dx_m": live.scene.dx, "invalid": int(live.invalid_reason),
            "guard_violations": int(live.solver.cfl_violations.numpy()[0]),
            "j_violations": int(live.solver.violations.numpy()[0]),
            "solver_options": live.solver.solver_receipt()["options"], "wall_s": time.perf_counter()-t0}


def case_q1(q, run, device="cpu", options=None, guard=1.0):
    """Q1: static sag of the -02 type strip (setup q: L_over_t, t_m, W_over_t, mu_pa, alpha; run: n_t, cfl, beta,
    ppc). Gravity from alpha with E_eff(beta = 20) (the same load for every beta)."""
    t0 = time.perf_counter()
    t, L, W = q["t_m"], q["L_over_t"]*q["t_m"], q["W_over_t"]*q["t_m"]
    E20 = material(q["mu_pa"]).young_effective()
    A, I = R.rect_section(W, t)
    g = q["alpha"]*E20*I/(RHO*A*L**3)
    mat = material(q["mu_pa"], run["beta"])
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=run["n_t"], ppc=run["ppc"], case="A")
    live = _live(spec, mat, 1e-3, (0.0, 0.0, -g), run["cfl"], guard, options)
    E = mat.young_effective()
    f1 = R.cantilever_f1(RHO, W, t, L, E)
    init = relax_to_static(live, omega1=2*math.pi*f1, stop_rule="static_stop_v3")
    delta = float(live.scene.marker_reference[-1][2]-np.asarray(init["markers_f64_m"])[-1][2])
    return dict(_common(live, run, t0), Q=delta, quantity="Q1 static tip deflection (m)", gravity_m_s2=g,
                delta_EB_beta_m=R.euler_bernoulli_sag(RHO, g, W, t, L, E),
                delta_EB_incompressible_m=R.euler_bernoulli_sag(RHO, g, W, t, L, 3*mat.mu0),
                converged=bool(init["converged"]), init_time_s=init["time_s"],
                reaction_residual=init["final"].get("reaction_residual"))


def case_q2(q, run, device="cpu", options=None, guard=1.0):
    """Q2: f1 of the -04 type strip (L_m, W_m, t_m, mu_pa, gravity_m_s2, record_interval_s, periods)."""
    t0 = time.perf_counter()
    mat = material(q["mu_pa"], run["beta"])
    spec = rs.RubberStripSpec(length=q["L_m"], width=q["W_m"], thickness=q["t_m"], n_t=run["n_t"], ppc=run["ppc"],
                              case="A")
    live = _live(spec, mat, q["record_interval_s"], (0, 0, -q["gravity_m_s2"]), run["cfl"], guard, options)
    E = mat.young_effective()
    f_eb = R.cantilever_f1(RHO, q["W_m"], q["t_m"], q["L_m"], E)
    steps = int(math.ceil(q["periods"]/f_eb/live.control_dt))
    z0 = live.scene.marker_reference[-1][2]
    tip, energy, psi = [], [], []
    s = live.solver
    for _ in range(steps):
        live.advance()
        d = live.last_diagnostics
        tip.append(float(live.markers()[-1][2]-z0))
        psi.append(d["elastic"])
        energy.append(d["kinetic_particle"]+d["elastic"]+d["gravity"]-float(s.work.numpy()[0]))
    from .rubber_phys import vibration_metrics
    m = vibration_metrics(np.array(tip), np.array(energy), np.array(psi), live.control_dt)
    return dict(_common(live, run, t0), Q=m.get("f1_hz"), quantity="Q2 first bending frequency (Hz)",
                f_EB_beta_hz=f_eb, f_EB_incompressible_hz=R.cantilever_f1(RHO, q["W_m"], q["t_m"], q["L_m"],
                                                                          3*mat.mu0),
                energy_drift_rel_peak_psi=m["energy_drift_rel_peak_psi"], whole_periods=m.get("whole_periods"),
                amplitude_ratio=(m["amplitude_last_m"]/m["amplitude_first_m"]) if "amplitude_first_m" in m else None)


def case_q3(q, run, device="cpu", options=None, guard=1.0):
    """Q3: nominal stress at lambda 1.5 on the load half of the -06 path (L_m, W_m, t_m, mu_pa, control_interval_s,
    lambda_max, ramp_T_ax); Q5: gauge Jbar error and Jbar extremes over the load half."""
    from .rubber_phys import _gauge_cells, axial_period, cosine_path
    t0 = time.perf_counter()
    mat = material(q["mu_pa"], run["beta"])
    spec = rs.RubberStripSpec(length=q["L_m"], width=q["W_m"], thickness=q["t_m"], n_t=run["n_t"], ppc=run["ppc"],
                              case="C", end_mask="x")
    live = _live(spec, mat, q["control_interval_s"], (0, 0, 0), run["cfl"], guard, options)
    L, dtc = q["L_m"], live.control_dt
    T_ax = axial_period(L, material(q["mu_pa"]))                 # the beta = 20 path for every run (same loading)
    n_r = int(math.ceil(q["ramp_T_ax"]*T_ax/dtc-1e-9))
    gauge = _gauge_cells(live.scene, L)
    K = live.substeps
    lam_prev, rows = 1.0, []
    jb_min, jb_max = 1.0, 1.0
    for k in range(n_r):
        lam = cosine_path((k+1)*dtc, n_r*dtc, q["lambda_max"])
        w = live.advance({1: {"translation": np.array([(lam-1)*L, 0.0, 0.0])}})
        lam_sub = lam_prev+(np.arange(K)+0.5)/K*(lam-lam_prev)
        analytic = float(np.mean([R.uniaxial_stress(x, mat.mu0, mat.kappa)["nominal_stress"]
                                  for x in lam_sub[::max(1, K//64)]]))
        d = live.last_diagnostics
        jb_min, jb_max = min(jb_min, d["Jbar_min"]), max(jb_max, d["Jbar_max"])
        rows.append({"lam_mid": 0.5*(lam_prev+lam), "stress": float(-w[1][0]/(q["W_m"]*q["t_m"])),
                     "analytic": analytic, "gauge_jbar": float(live.solver.jbar_cells()[gauge].mean()),
                     "J_end": R.uniaxial_stress(lam, mat.mu0, mat.kappa)["J"]})
        lam_prev = lam
    r = min(rows, key=lambda r: abs(r["lam_mid"]-1.5))
    mk = live.markers()
    lateral = float(np.abs(mk[:, 1:]-live.scene.marker_reference[:, 1:]).max())
    lam_r = r["lam_mid"]
    return dict(_common(live, run, t0), Q=r["stress"], quantity="Q3 nominal stress at lambda 1.5 (Pa)",
                lam_mid=lam_r, analytic_compressible_pa=r["analytic"],
                analytic_incompressible_pa=float(mat.mu0*(lam_r-lam_r**-2)),
                rel_error_compressible=r["stress"]/r["analytic"]-1,
                Q5_gauge_jbar_max_abs_error=float(max(abs(x["gauge_jbar"]-x["J_end"]) for x in rows)),
                Q5_jbar_range=[jb_min, jb_max], max_lateral_drift_m=lateral, ramp_steps=n_r)


def case_q4(q, run, device="cpu", options=None, guard=1.0):
    """Q4: large-deflection tip drop of the -05 type strip (L_over_t, mu_pa, gravity_m_s2, alpha; geometry from
    alpha with E_eff(beta = 20))."""
    t0 = time.perf_counter()
    E20 = material(q["mu_pa"]).young_effective()
    L = q["alpha"]*E20/(12*RHO*q["gravity_m_s2"]*q["L_over_t"]**2)
    t = L/q["L_over_t"]
    mat = material(q["mu_pa"], run["beta"])
    spec = rs.RubberStripSpec(length=L, width=t, thickness=t, n_t=run["n_t"], ppc=run["ppc"], case="A")
    live = _live(spec, mat, 1e-3, (0.0, 0.0, -q["gravity_m_s2"]), run["cfl"], guard, options)
    E = mat.young_effective()
    f1 = R.cantilever_f1(RHO, t, t, L, E)
    init = relax_to_static(live, omega1=2*math.pi*f1, stop_rule="static_stop_v3")
    delta = float(live.scene.marker_reference[-1][2]-np.asarray(init["markers_f64_m"])[-1][2])
    alpha_beta = R.load_parameter(RHO, q["gravity_m_s2"], t, t, L, E)
    d = live.solver.diagnostics.read()[0]
    return dict(_common(live, run, t0), Q=delta, quantity="Q4 large-deflection tip drop (m)", L_m=L, t_m=t,
                elastica_beta_m=R.elastica(alpha_beta)["tip_drop"]*L,
                elastica_incompressible_m=R.elastica(R.load_parameter(RHO, q["gravity_m_s2"], t, t, L, 3*mat.mu0)
                                                     )["tip_drop"]*L,
                converged=bool(init["converged"]), init_time_s=init["time_s"],
                Jbar_range=[d["Jbar_min"], d["Jbar_max"]])


CASES = {"Q1": case_q1, "Q2": case_q2, "Q3": case_q3, "Q4": case_q4}


def run_label(qid, run):
    return f"{qid}_nt{run['n_t']}_cfl{run['cfl']:g}_beta{run['beta']:g}_ppc{run['ppc']}"


def jobs(cfg):
    """(qid, label, run) for every declared run of RUBBER-PHYS-11 v4."""
    st = cfg["tests"]["RUBBER-PHYS-11"]["setup"]
    out = []
    for qid in st["observables"]:
        runs = []
        for n_t in st["grids_n_t"]:
            for cfl in st["cfl"]:
                runs.append({"n_t": n_t, "cfl": cfl, "beta": 20.0, "ppc": 2})
        for n_t in st["beta_sensitivity"]["n_t"]:
            for beta in st["beta_sensitivity"]["beta"]:
                if beta != 20.0:
                    runs.append({"n_t": n_t, "cfl": 0.5, "beta": beta, "ppc": 2})
        qc = st["quadrature_check"]
        runs.append({"n_t": qc["n_t"].get(qid, qc["n_t"]["default"]), "cfl": 0.5, "beta": 20.0, "ppc": qc["ppc"]})
        for run in runs:
            out.append((qid, run_label(qid, run), run))
    return out


def evaluate(rows, cfg):
    """Apply the declared v4 rule (configs/rubber/rubber_phys_tests_v4.json RUBBER-PHYS-11) to raw rows."""
    st = cfg["tests"]["RUBBER-PHYS-11"]["setup"]
    th = cfg["tests"]["RUBBER-PHYS-11"]["thresholds"]
    rule = st["rule"]
    by = {(r["qid"], r["run"]["n_t"], r["run"]["cfl"], r["run"]["beta"], r["run"]["ppc"]): r for r in rows
          if r.get("status") == "OK"}
    verdicts, measured = {}, {}
    prod = st["production_n_t"]
    for qid in st["observables"]:
        m = {}
        ok_rows = [r for k, r in by.items() if k[0] == qid]
        bad = [r["label"] for r in ok_rows if r.get("invalid") or (r.get("converged") is False) or r["Q"] is None]
        missing = [run_label(qid, run) for q, _, run in jobs(cfg) if q == qid
                   and (qid, run["n_t"], run["cfl"], run["beta"], run["ppc"]) not in by]
        m["invalid_or_unconverged"], m["missing"] = bad, missing
        for cfl in st["cfl"]:
            trip = {}
            for name, grids in (("finest", st["triplet_finest"]), ("coarsest", st["triplet_coarsest"])):
                keys = [(qid, n, cfl, 20.0, 2) for n in sorted(grids, reverse=True)]       # fine -> coarse
                if all(k in by for k in keys):
                    h = [1.0/n for n in sorted(grids, reverse=True)]
                    phi = [by[k]["Q"] for k in keys]
                    res = G.study(h, phi, p_bounds=tuple(rule["p_bounds"]), fs_verified=rule["fs_verified"],
                                  fs_unverified=rule["fs_unverified"], verified_window=tuple(rule["verified_window"]))
                    if res.get("range_rel") is not None and res["range_rel"] <= rule["grid_independent_range_rel"]:
                        res.update(status="GRID_INDEPENDENT", gci_coarse=rule["fs_verified"]*res["range_rel"],
                                   gci_fine=rule["fs_verified"]*res["range_rel"])
                    elif res["status"] == "INCONCLUSIVE" and res.get("e21") is not None:
                        rng = (max(phi)-min(phi))/abs(phi[0])
                        if rng <= rule["grid_independent_range_rel"]:
                            res.update(status="GRID_INDEPENDENT", gci_coarse=rule["fs_verified"]*rng,
                                       gci_fine=rule["fs_verified"]*rng)
                    trip[name] = res
            m[f"gci_cfl{cfl:g}"] = trip
        dt = {}
        for n in st["grids_n_t"]:
            a, b = by.get((qid, n, 0.5, 20.0, 2)), by.get((qid, n, 0.25, 20.0, 2))
            if a and b:
                dt[n] = b["Q"]/a["Q"]-1
        m["dt_change_cfl050_to_cfl025"] = dt
        sens = {}
        for n in st["beta_sensitivity"]["n_t"]:
            base = by.get((qid, n, 0.5, 20.0, 2))
            for beta in st["beta_sensitivity"]["beta"]:
                other = by.get((qid, n, 0.5, beta, 2))
                if base and other and beta != 20.0:
                    sens[f"nt{n}_beta{beta:g}_over_beta20_minus_1"] = other["Q"]/base["Q"]-1
        m["beta_sensitivity"] = sens
        qc = st["quadrature_check"]
        nq = qc["n_t"].get(qid, qc["n_t"]["default"])
        a, b = by.get((qid, nq, 0.5, 20.0, 2)), by.get((qid, nq, 0.5, 20.0, qc["ppc"]))
        m["ppc3_over_ppc2_minus_1"] = (b["Q"]/a["Q"]-1) if a and b else None
        m["values"] = {r["label"]: r["Q"] for r in ok_rows}
        prod_gci = m.get(f"gci_cfl{st['verdict_cfl']:g}", {}).get("finest")
        gci_ok = (prod_gci is not None and prod_gci.get("gci_coarse") is not None
                  and prod_gci["gci_coarse"] <= th["gci"])
        verdicts[f"{qid}_gci_production"] = "PASS" if gci_ok else ("NOT_RUN" if prod_gci is None else "FAIL")
        verdicts[f"{qid}_dt"] = ("PASS" if prod in dt and abs(dt[prod]) <= th["dt_rel"] else
                                 "NOT_RUN" if prod not in dt else "FAIL")
        verdicts[f"{qid}_valid"] = "PASS" if not bad and not missing else "FAIL"
        measured[qid] = m
    return verdicts, measured
