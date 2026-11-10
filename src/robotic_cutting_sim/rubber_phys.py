"""RUBBER-PHYS protocol implementations (ADR 0023 §14.3; frozen protocols
``configs/rubber/rubber_phys_tests_v2.json`` and ``rubber_phys_tests_v3.json``; v3 = P2-1 follow-up:
static_stop_v3 for -02/-03/-05, the redesigned -07, the guard regression -12, see ADR 0023 §18).

Each ``case_*`` function runs one configuration in the single-env reference and returns a JSON-able dict of
measurements (never a verdict); each ``evaluate_*`` function applies the pre-declared criteria of the frozen config to
the case results and returns per-criterion PASS / FAIL. The runner ``scripts/rubber/run_rubber_phys.py`` executes the
cases (one process per case), stores the raw results and writes the evidence. Material status:
``SIMULATOR_ONLY_UNCALIBRATED`` (numerical verification only).
"""
import json
import math
import time

import numpy as np

from . import rubber_material as rm
from . import rubber_reference as R
from . import rubber_scene as rs
from .live_rubber_material import LiveRubberMaterial, decode_reason, validate_records
from .rubber_constitutive import evaluate_pk1
from .rubber_initializer import relax_to_static

CONFIG = rm.ROOT/"configs"/"rubber"/"rubber_phys_tests_v2.json"
CONFIG_V3 = rm.ROOT/"configs"/"rubber"/"rubber_phys_tests_v3.json"
CONFIG_V4 = rm.ROOT/"configs"/"rubber"/"rubber_phys_tests_v4.json"
PROTOCOLS = {"v2": CONFIG, "v3": CONFIG_V3, "v4": CONFIG_V4}
RHO = 1150.0
BETA = 20.0


def load_config(path=CONFIG):
    return json.loads(open(path).read())


def guard_limit(cfg):
    """The protocol's stable-step guard limit (v3: common.stable_step_guard.limit; v2: off)."""
    g = cfg["common"].get("stable_step_guard")
    return None if g is None else float(g["limit"])


def solver_options(cfg, test):
    """The protocol's opt-in solver options of ``test`` (v4: tests.<id>.setup.solver_options; earlier: none = v1)."""
    return dict(cfg["tests"][test]["setup"].get("solver_options", {})) if isinstance(cfg["tests"][test]["setup"],
                                                                                   dict) else {}


def stop_rule_id(cfg):
    return cfg["common"]["static_initializer"].get("stop_rule_id", "static_stop_v2")


def material(mu, rho=RHO, beta=BETA):
    return rm.MaterialValues(mu0=float(mu), kappa=float(beta)*float(mu), density=float(rho))


def _verdict(ok):
    return "PASS" if ok else "FAIL"


# ----------------------------------------------------------------------------- RUBBER-PHYS-01 patch

def patch_target(case, mu=4e5, kappa=8e6):
    kind = case["kind"]
    if kind == "uniaxial_stress":
        return R.uniaxial_stress(case["stretch"], mu, kappa)["F"]
    if kind == "simple_shear":
        return R.simple_shear(case["gamma"], mu)["F"]
    if kind == "rotation_times_uniaxial":
        from scipy.spatial.transform import Rotation
        Rz = Rotation.from_euler("z", case["rotation_deg_z"], degrees=True).as_matrix()
        return Rz@R.uniaxial_stress(case["stretch"], mu, kappa)["F"]
    raise ValueError(kind)


def _block(dx=0.001, cells=6):
    return rs.RubberStripSpec(length=cells*dx, width=cells*dx, thickness=cells*dx, dx=dx, case="block",
                              center=(0.0, 0.0, 0.0))


def case_patch(case, device="cpu", mu=4e5):
    mat = material(mu)
    spec = _block()
    dt_max = 0.5*spec.grid_dx()/mat.c_p()
    live = LiveRubberMaterial(spec, material=mat, control_dt=dt_max*(1-1e-9), gravity=(0, 0, 0), device=device)
    assert live.substeps == 1
    Ft = patch_target(case, mu, mat.kappa)
    live.advance({0: {"affine": Ft, "translation": np.zeros(3)}})
    s, sc = live.solver, live.scene
    f_nodes = s.nodal_internal_force()                 # also leaves solver.stress = P(F_now)
    F = s.F.numpy().astype(np.float64)
    P = s.stress.numpy().astype(np.float64)
    J = np.linalg.det(F)
    jbar_kernel = s.jbar.numpy()[sc.nodes[:, 0]]
    cell_rank = np.empty(sc.node_count, np.int64)
    cell_rank[sc.cell_node] = np.arange(len(sc.cell_node))
    rank = cell_rank[sc.nodes[:, 0]]
    jbar_cells = np.bincount(rank, weights=J)/np.bincount(rank)
    jbar_p = jbar_cells[rank]
    P_ref = np.stack([R.pk1(F[p], mat.mu0, mat.kappa, jbar_p[p]) for p in range(sc.n)])
    err = np.linalg.norm(P-P_ref, axis=(1, 2))
    allowed = 1e-5*np.linalg.norm(P_ref, axis=(1, 2))+1e-3
    interior = sc.interior_nodes()
    boundary = np.setdiff1d(np.flatnonzero(sc.node_has_mass), interior)
    fnorm = np.linalg.norm(f_nodes, axis=1)
    out = {"case": case["id"], "particles": sc.n, "F_target": Ft.tolist(), "substeps": live.substeps,
           "max_F_deviation": float(np.abs(F-Ft).max()),
           "pk1_max_error_pa": float(err.max()), "pk1_max_error_over_allowed": float((err/allowed).max()),
           "pk1_max_relative": float((err/np.linalg.norm(P_ref, axis=(1, 2))).max()),
           "jbar_consistency_max": float(np.abs(jbar_p-J).max()),
           "jbar_kernel_vs_numpy_max": float(np.abs(jbar_kernel-jbar_p).max()),
           "jbar_analytic_max": float(np.abs(jbar_cells-np.linalg.det(Ft)).max()),
           "interior_nodes": int(len(interior)), "boundary_nodes": int(len(boundary)),
           "interior_force_max_n": float(fnorm[interior].max()), "boundary_force_max_n": float(fnorm[boundary].max()),
           "interior_force_ratio": float(fnorm[interior].max()/fnorm[boundary].max())}
    if case["kind"] != "rotation_times_uniaxial":
        from scipy.spatial.transform import Rotation
        worst = 0.0
        F32 = Ft.astype(np.float32)
        P0, _ = evaluate_pk1(F32, np.linalg.det(F32.astype(np.float64)), mat.mu0, mat.kappa)
        for Rm in (Rotation.from_euler("z", 90, degrees=True).as_matrix(),
                   Rotation.from_rotvec(np.deg2rad(37)*np.array([1, 2, 3])/math.sqrt(14)).as_matrix()):
            RF = (Rm@F32.astype(np.float64)).astype(np.float32)
            PR, _ = evaluate_pk1(RF, np.linalg.det(RF.astype(np.float64)), mat.mu0, mat.kappa)
            worst = max(worst, float(np.linalg.norm(PR[0]-Rm@P0[0])/np.linalg.norm(P0[0])))
        out["objectivity_max_relative"] = worst
    return out


def evaluate_01(results, cfg):
    th = cfg["tests"]["RUBBER-PHYS-01"]["thresholds"]
    checks = {
        "pk1_per_particle": all(r["pk1_max_error_over_allowed"] <= 1.0 for r in results),
        "jbar_consistency": all(r["jbar_consistency_max"] <= th["jbar_consistency"] for r in results),
        "jbar_analytic": all(r["jbar_analytic_max"] <= th["jbar_analytic"] for r in results),
        "interior_force": all(r["interior_force_ratio"] <= th["interior_force_ratio"] for r in results),
        "objectivity": all(r.get("objectivity_max_relative", 0.0) <= th["objectivity_rel"] for r in results)}
    measured = {"pk1_max_error_over_allowed": max(r["pk1_max_error_over_allowed"] for r in results),
                "jbar_consistency_max": max(r["jbar_consistency_max"] for r in results),
                "jbar_analytic_max": max(r["jbar_analytic_max"] for r in results),
                "interior_force_ratio_max": max(r["interior_force_ratio"] for r in results),
                "objectivity_max_relative": max(r.get("objectivity_max_relative", 0.0) for r in results)}
    return {k: _verdict(v) for k, v in checks.items()}, measured


# ----------------------------------------------------------------------------- RUBBER-PHYS-08 simple shear

def case_shear(gamma, device="cpu", mu=4e5, substeps=200, stable_step_guard=None):
    mat = material(mu)
    spec = _block()
    dt_max = 0.5*spec.grid_dx()/mat.c_p()
    live = LiveRubberMaterial(spec, material=mat, control_dt=substeps*dt_max*(1-1e-9), gravity=(0, 0, 0),
                              device=device, stable_step_guard=stable_step_guard, strict=stable_step_guard is None)
    assert live.substeps == substeps
    Ft = R.simple_shear(gamma, mu)["F"]
    live.advance({0: {"affine": Ft, "translation": np.zeros(3)}})
    live.solver.nodal_internal_force()
    F = live.solver.F.numpy().astype(np.float64)
    P = live.solver.stress.numpy().astype(np.float64)
    sigma = np.einsum("pij,pkj->pik", P, F)/np.linalg.det(F)[:, None, None]
    m = sigma.mean(0)
    return {"gamma": gamma, "substeps": substeps, "sigma_mean_pa": m.tolist(), "sigma_12_pa": float(m[0, 1]),
            "invalid": int(live.invalid_reason), "stable_step_guard": stable_step_guard,
            "N1_pa": float(m[0, 0]-m[1, 1]), "sigma_12_ref_pa": mu*gamma, "N1_ref_pa": mu*gamma**2,
            "sigma_12_rel_error": float(m[0, 1]/(mu*gamma)-1), "N1_rel_error": float((m[0, 0]-m[1, 1])/(mu*gamma**2)-1),
            "max_F_deviation": float(np.abs(F-Ft).max())}


def evaluate_08(results, cfg):
    th = cfg["tests"]["RUBBER-PHYS-08"]["thresholds"]["rel"]
    checks = {"shear": all(abs(r["sigma_12_rel_error"]) <= th for r in results),
              "normal_difference": all(abs(r["N1_rel_error"]) <= th for r in results)}
    measured = {"max_abs_sigma_12_rel_error": max(abs(r["sigma_12_rel_error"]) for r in results),
                "max_abs_N1_rel_error": max(abs(r["N1_rel_error"]) for r in results)}
    return {k: _verdict(v) for k, v in checks.items()}, measured


# ----------------------------------------------------------------------------- static cantilever (02, 03, 05)

def cantilever_live(L, W, t, n_t, mu, g, device="cpu", control_dt=1e-3, **kw):
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=n_t, case="A")
    return LiveRubberMaterial(spec, material=material(mu), control_dt=control_dt, gravity=(0.0, 0.0, -g),
                              device=device, **kw)


def case_static_sag(L, W, t, n_t, mu, g, *, device="cpu", delta_ref=None, label=None, stop_rule="static_stop_v2",
                    stable_step_guard=None, options=None):
    """Static sag by the relax_mass_proportional_v1 initializer (static_stop_v2, or v3: the deflection is measured
    on the float64 initializer markers, the float32-position value is reported as ``delta_x_f32_m``)."""
    start = time.perf_counter()
    live = cantilever_live(L, W, t, n_t, mu, g, device, stable_step_guard=stable_step_guard, **(options or {}))
    mat = live.material
    E = mat.young_effective()
    f1 = R.cantilever_f1(mat.density, W, t, L, E)
    init = relax_to_static(live, omega1=2*math.pi*f1, stop_rule=stop_rule)
    diag = live.solver.diagnostics.read()[0]
    markers = live.markers()
    delta_x_f32 = float(live.scene.marker_reference[-1][2]-markers[-1][2])
    if stop_rule == "static_stop_v3":
        markers = np.asarray(init["markers_f64_m"])
    delta = float(live.scene.marker_reference[-1][2]-markers[-1][2])
    eb = R.euler_bernoulli_sag(mat.density, g, W, t, L, E)
    return {"label": label, "L_m": L, "W_m": W, "t_m": t, "n_t": n_t, "mu_pa": mu, "gravity_m_s2": g,
            "alpha": R.load_parameter(mat.density, g, W, t, L, E), "particles": live.scene.n,
            "dt_s": live.dt, "cfl": live.step_rule["cfl_actual"], "f1_EB_hz": f1, "delta_m": delta,
            "delta_EB_m": eb, "ratio_EB": delta/eb, "nu_eff": mat.poisson_effective(), "E_eff_pa": E,
            "delta_ref_m": delta_ref, "initializer": init, "Jbar_min": diag["Jbar_min"], "Jbar_max": diag["Jbar_max"],
            "J_min": diag["J_min"], "J_max": diag["J_max"], "tip_x_m": float(markers[-1][0]),
            "reaction_residual": init["final"].get("reaction_residual"), "device": live.device,
            "stop_rule": stop_rule, "delta_x_f32_m": delta_x_f32, "invalid": live.invalid_reason,
            "cfl_eff_max_end": diag["cfl_eff_max"], "solver_options": live.solver.solver_receipt()["options"],
            "wall_s": time.perf_counter()-start}


def configs_02(cfg):
    st = cfg["tests"]["RUBBER-PHYS-02"]["setup"]
    out = []
    for Lt in st["L_over_t"]:
        for t in st["t_m"]:
            for Wt in st["W_over_t"]:
                for n_t in st["n_t"]:
                    L, W = Lt*t, Wt*t
                    E = material(st["mu_pa"]).young_effective()
                    A, I = R.rect_section(W, t)
                    g = st["alpha_target"]*E*I/(RHO*A*L**3)
                    out.append(dict(L=L, W=W, t=t, n_t=n_t, mu=st["mu_pa"], g=g, label=f"Lt{Lt}_t{t*1e3:g}mm_Wt{Wt}_nt{n_t}"))
    return out


def evaluate_02(results, cfg):
    th = cfg["tests"]["RUBBER-PHYS-02"]["thresholds"]
    rows, ok = [], {"narrow_eb": True, "wide_bracket": True, "extrapolated": True, "reaction": True,
                    "initializer_converged": True}
    key = lambda r: (round(r["L_m"]/r["t_m"]), r["t_m"], round(r["W_m"]/r["t_m"]))
    groups = {}
    for r in results:
        groups.setdefault(key(r), {})[r["n_t"]] = r
    for k, by_nt in sorted(groups.items()):
        if 3 not in by_nt or 4 not in by_nt:
            ok["extrapolated"] = False
            continue
        ex = R.richardson_two_grid(by_nt[3]["delta_m"], by_nt[4]["delta_m"], 4/3, 2.0)
        for n_t, r in sorted(by_nt.items()):
            lo, hi = R.plate_bracket(r["delta_EB_m"], r["nu_eff"])
            row = {"label": r["label"], "ratio_EB": r["ratio_EB"], "ratio_extrap": r["delta_m"]/ex,
                   "reaction_residual": r["reaction_residual"], "converged": r["initializer"]["converged"]}
            if k[2] == 1:
                row["narrow_eb_ok"] = abs(r["ratio_EB"]-1) <= th["narrow_eb_rel"]
                ok["narrow_eb"] &= row["narrow_eb_ok"]
            else:
                row["bracket"] = [lo/r["delta_EB_m"], hi/r["delta_EB_m"]]
                row["wide_bracket_ok"] = lo <= r["delta_m"] <= hi
                ok["wide_bracket"] &= row["wide_bracket_ok"]
            row["extrap_ok"] = abs(row["ratio_extrap"]-1) <= th["extrap_rel"]
            ok["extrapolated"] &= row["extrap_ok"]
            ok["reaction"] &= r["reaction_residual"] is not None and r["reaction_residual"] <= th["reaction_rel"]
            ok["initializer_converged"] &= r["initializer"]["converged"]
            rows.append(row)
    if len(results) != cfg["tests"]["RUBBER-PHYS-02"]["setup"]["configurations"]:
        ok = {k: False for k in ok}
    return {k: _verdict(v) for k, v in ok.items()}, {"rows": rows}


def configs_03(cfg):
    st = cfg["tests"]["RUBBER-PHYS-03"]["setup"]
    out = [dict(L=L, W=st["series_L"]["t_m"], t=st["series_L"]["t_m"], n_t=st["n_t"], mu=st["mu_pa"],
                g=st["gravity_m_s2"], label=f"series_L_L{L*1e3:g}mm") for L in st["series_L"]["L_m"]]
    out += [dict(L=st["series_t"]["L_m"], W=t, t=t, n_t=st["n_t"], mu=st["mu_pa"], g=st["gravity_m_s2"],
                 label=f"series_t_t{t*1e3:g}mm") for t in st["series_t"]["t_m"]]
    return out


def evaluate_03(results, cfg):
    th = cfg["tests"]["RUBBER-PHYS-03"]["thresholds"]
    sl = sorted([r for r in results if r["label"].startswith("series_L")], key=lambda r: r["L_m"])
    st = sorted([r for r in results if r["label"].startswith("series_t")], key=lambda r: r["t_m"])
    measured = {}
    ok = {"length_exponent": False, "thickness_exponent": False,
          "initializer_converged": all(r["initializer"]["converged"] for r in results) and len(results) == 6}
    if len(sl) == 3:
        measured["length_exponent"] = R.power_law_exponent([r["L_m"] for r in sl], [r["delta_m"] for r in sl])
        ok["length_exponent"] = th["length_exponent"][0] <= measured["length_exponent"] <= th["length_exponent"][1]
    if len(st) == 3:
        measured["thickness_exponent"] = R.power_law_exponent([r["t_m"] for r in st], [r["delta_m"] for r in st])
        ok["thickness_exponent"] = (th["thickness_exponent"][0] <= measured["thickness_exponent"]
                                    <= th["thickness_exponent"][1])
    measured["ratio_EB"] = {r["label"]: r["ratio_EB"] for r in results}
    return {k: _verdict(v) for k, v in ok.items()}, measured


def configs_05(cfg):
    st = cfg["tests"]["RUBBER-PHYS-05"]["setup"]
    E = material(st["mu_pa"]).young_effective()
    out = []
    for alpha in st["alpha"]:
        L = alpha*E/(12*RHO*st["gravity_m_s2"]*st["L_over_t"]**2)
        t = L/st["L_over_t"]
        drop = R.elastica(alpha)["tip_drop"]*L
        out.append(dict(L=L, W=t, t=t, n_t=st["n_t"], mu=st["mu_pa"], g=st["gravity_m_s2"], delta_ref=drop,
                        label=f"alpha{alpha:g}"))
    return out


def evaluate_05(results, cfg):
    th = cfg["tests"]["RUBBER-PHYS-05"]["thresholds"]
    rows = []
    ok = {"tip_drop": len(results) == 3, "jbar": len(results) == 3, "initializer_converged": len(results) == 3}
    for r in results:
        ratio = r["delta_m"]/r["delta_ref_m"]
        row = {"label": r["label"], "alpha": r["alpha"], "ratio_elastica": ratio, "Jbar": [r["Jbar_min"], r["Jbar_max"]],
               "converged": r["initializer"]["converged"]}
        ok["tip_drop"] &= abs(ratio-1) <= th["tip_rel"]
        ok["jbar"] &= th["jbar_range"][0] <= r["Jbar_min"] and r["Jbar_max"] <= th["jbar_range"][1]
        ok["initializer_converged"] &= r["initializer"]["converged"]
        rows.append(row)
    return {k: _verdict(v) for k, v in ok.items()}, {"rows": rows}


# ----------------------------------------------------------------------------- RUBBER-PHYS-04 free vibration

def vibration_setup(cfg):
    return cfg["tests"]["RUBBER-PHYS-04"]["setup"]


def case_vibration(run_id, cfg, device="cpu", L=None, W=None, mu=None, rho=RHO, substeps=None, record_records=False):
    st = vibration_setup(cfg)
    L = st["L_m"] if L is None else L
    W = st["W_m"] if W is None else W
    t, n_t = st["t_m"], st["n_t"]
    mu = st["mu_pa"] if mu is None else mu
    mat = material(mu, rho)
    E = mat.young_effective()
    f_eb = R.cantilever_f1(rho, W, t, L, E)
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=n_t, case="A")
    live = LiveRubberMaterial(spec, material=mat, control_dt=st["record_interval_s"], substeps=substeps,
                              gravity=(0, 0, -st["gravity_m_s2"]), device=device, record=True)
    steps = int(math.ceil(6/f_eb/live.control_dt))
    start = time.perf_counter()
    for _ in range(steps):
        live.advance()
    wall = time.perf_counter()-start
    recs = live.records[1:]
    tip_z = np.array([r["markers_m"][-1][2] for r in recs])-live.scene.marker_reference[-1][2]
    energy = np.array([r["energies_j"]["total"] for r in recs])
    energy_grid = np.array([r["energies_j"]["kinetic_grid"]+r["energies_j"]["elastic"]+r["energies_j"]["gravity"]
                            for r in recs])
    psi = np.array([r["energies_j"]["elastic"] for r in recs])
    out = {"run": run_id, "L_m": L, "W_m": W, "t_m": t, "n_t": n_t, "mu_pa": mu, "rho": rho, "E_eff_pa": E,
           "f_EB_hz": f_eb, "particles": live.scene.n, "dt_s": live.dt, "substeps_per_record": live.substeps,
           "cfl": live.step_rule["cfl_actual"], "steps": steps, "record_interval_s": live.control_dt,
           "tip_z_m": tip_z.tolist(), "energy_total_j": energy.tolist(), "energy_total_gridke_j": energy_grid.tolist(),
           "elastic_j": psi.tolist(), "wall_s": wall, "device": live.device,
           "invalid": live.invalid_reason, "Jbar_range": [min(r["j_stats"]["Jbar_min"] for r in recs),
                                                          max(r["j_stats"]["Jbar_max"] for r in recs)]}
    out.update(vibration_metrics(tip_z, energy, psi, live.control_dt))
    if record_records:
        out["records"] = live.records
    return out


def vibration_metrics(tip_z, energy, psi, interval):
    t = (np.arange(len(tip_z))+1)*interval
    y = tip_z-tip_z.mean()
    crossings = []
    for i in range(1, len(y)):
        if y[i-1] < 0 <= y[i]:
            crossings.append(t[i-1]+(t[i]-t[i-1])*(-y[i-1])/(y[i]-y[i-1]))
    out = {"upward_crossings_s": crossings}
    if len(crossings) >= 3:
        periods = len(crossings)-1
        out["f1_hz"] = periods/(crossings[-1]-crossings[0])
        def amp(a, b):
            sel = (t >= a) & (t <= b)
            return float((tip_z[sel].max()-tip_z[sel].min())/2)
        out["amplitude_first_m"] = amp(crossings[0], crossings[1])
        out["amplitude_last_m"] = amp(crossings[-2], crossings[-1])
        out["whole_periods"] = periods
    out["energy_drift_j"] = float(np.abs(energy-0.0).max())       # E_total(0) = 0 at the straight rest state
    out["peak_elastic_j"] = float(psi.max())
    out["energy_drift_rel_peak_psi"] = out["energy_drift_j"]/out["peak_elastic_j"]
    return out


def evaluate_04(results, cfg, batched=None):
    th = cfg["tests"]["RUBBER-PHYS-04"]["thresholds"]
    by = {r["run"]: r for r in results}
    ok = {}
    measured = {}
    d, h = by.get("dt"), by.get("dt_half")
    if d is None or "f1_hz" not in d:
        return {k: "FAIL" for k in ("frequency", "energy", "amplitude", "dt_independence", "batched_equal")}, {}
    measured["f1_hz"] = d["f1_hz"]
    measured["f_EB_hz"] = d["f_EB_hz"]
    measured["f1_over_fEB_minus_1"] = d["f1_hz"]/d["f_EB_hz"]-1
    ok["frequency"] = abs(measured["f1_over_fEB_minus_1"]) <= th["frequency_rel"]
    measured["energy_drift_rel_peak_psi"] = max(r["energy_drift_rel_peak_psi"] for r in (d, h) if r)
    ok["energy"] = measured["energy_drift_rel_peak_psi"] <= th["energy_rel_peak_psi"]
    measured["amplitude_ratio_last_first"] = min(r["amplitude_last_m"]/r["amplitude_first_m"] for r in (d, h) if r)
    ok["amplitude"] = measured["amplitude_ratio_last_first"] >= th["amplitude_ratio_min"]
    if h is not None and "f1_hz" in h:
        measured["f1_dt_over_f1_dthalf_minus_1"] = d["f1_hz"]/h["f1_hz"]-1
        ok["dt_independence"] = abs(measured["f1_dt_over_f1_dthalf_minus_1"]) <= th["dt_rel"]
    else:
        ok["dt_independence"] = False
    verdicts = {k: _verdict(v) for k, v in ok.items()}
    if batched is None:
        verdicts["batched_equal"] = "NOT_RUN"
    else:
        verdicts["batched_equal"] = _verdict(bool(batched.get("bitwise_equal")))
        measured["batched"] = {k: v for k, v in batched.items() if k != "records"}
    return verdicts, measured


# ----------------------------------------------------------------------------- RUBBER-PHYS-06 / 07 uniaxial

def uniaxial_live(cfg, device="cpu", record=False):
    """The -06 strip; from protocol v4 on with the protocol's stable-step guard (strict off, flags recorded)."""
    st = cfg["tests"]["RUBBER-PHYS-06"]["setup"]
    spec = rs.RubberStripSpec(length=st["L_m"], width=st["W_m"], thickness=st["t_m"], n_t=st["n_t"], case="C",
                              end_mask="x")
    guard = guard_limit(cfg) if cfg.get("version", 2) >= 4 else None
    return LiveRubberMaterial(spec, material=material(st["mu_pa"]), control_dt=st["control_interval_s"],
                              gravity=(0, 0, 0), device=device, record=record, stable_step_guard=guard,
                              strict=guard is None)


def axial_period(L, mat):
    return 2*L/math.sqrt(mat.young_effective()/mat.density)


def cosine_path(t, T, lam_max):
    return 1+(lam_max-1)*(1-math.cos(math.pi*min(max(t/T, 0.0), 1.0)))/2


def _gauge_cells(scene, L):
    centre = scene.node_X[scene.cell_node, 0]+scene.dx/2
    return np.flatnonzero((centre >= L/4) & (centre <= 3*L/4))


def case_uniaxial(cfg, device="cpu"):
    st = cfg["tests"]["RUBBER-PHYS-06"]["setup"]
    live = uniaxial_live(cfg, device)
    mat, L, W, t = live.material, st["L_m"], st["W_m"], st["t_m"]
    dtc = live.control_dt
    T_ax = axial_period(L, mat)
    n_r = int(math.ceil(60*T_ax/dtc-1e-9))
    n_h = int(math.ceil(10*T_ax/dtc-1e-9))
    T_r = n_r*dtc
    lam_end = [cosine_path((k+1)*dtc, T_r, 2.0) for k in range(n_r)]
    lam_end += [cosine_path(T_r-(k+1)*dtc, T_r, 2.0) for k in range(n_r)]
    lam_end += [1.0]*n_h
    gauge = _gauge_cells(live.scene, L)
    K = live.substeps
    rows, lam_prev = [], 1.0
    start = time.perf_counter()
    m_ref = live.scene.marker_reference
    for k, lam in enumerate(lam_end):
        w = live.advance({1: {"translation": np.array([(lam-1)*L, 0.0, 0.0])}})
        lam_sub = lam_prev+(np.arange(K)+0.5)/K*(lam-lam_prev)
        p_avg = float(np.mean([R.uniaxial_stress(x, mat.mu0, mat.kappa)["nominal_stress"] for x in lam_sub[::max(1, K//64)]]))
        jb = live.solver.jbar_cells()[gauge].mean()
        mk = live.markers()
        rows.append({"step": k+1, "lam_start": lam_prev, "lam_end": lam, "lam_mid": 0.5*(lam_prev+lam),
                     "nominal_stress_pa": float(-w[1][0]/(W*t)), "analytic_avg_pa": p_avg,
                     "gauge_jbar": float(jb), "J_analytic_end": R.uniaxial_stress(lam, mat.mu0, mat.kappa)["J"],
                     "gauge_26_m": float(mk[6][0]-mk[2][0]), "lateral_drift_m": float(np.abs(mk[:, 1:]-m_ref[:, 1:]).max()),
                     "reaction0_x_n": float(w[0][0])})
        lam_prev = lam
    return {"L_m": L, "T_ax_s": T_ax, "ramp_steps": n_r, "hold_steps": n_h, "control_interval_s": dtc,
            "substeps": K, "dt_s": live.dt, "particles": live.scene.n, "rows": rows,
            "gauge_26_reference_m": float(m_ref[6][0]-m_ref[2][0]), "wall_s": time.perf_counter()-start,
            "device": live.device, "invalid": live.invalid_reason}


def evaluate_06(result, cfg):
    th = cfg["tests"]["RUBBER-PHYS-06"]["thresholds"]
    rows, n_r = result["rows"], result["ramp_steps"]
    load, unload = rows[:n_r], rows[n_r:2*n_r]
    points = []
    for lam in (1.25, 1.5, 2.0):
        r = min(load, key=lambda r: abs(r["lam_mid"]-lam))
        points.append(("load", lam, r))
    for lam in (1.25, 1.5):
        r = min(unload, key=lambda r: abs(r["lam_mid"]-lam))
        points.append(("unload", lam, r))
    stress = [{"phase": ph, "lam_target": lam, "lam_mid": r["lam_mid"], "measured_pa": r["nominal_stress_pa"],
               "analytic_pa": r["analytic_avg_pa"], "rel_error": r["nominal_stress_pa"]/r["analytic_avg_pa"]-1}
              for ph, lam, r in points]
    jb_err = max(abs(r["gauge_jbar"]-r["J_analytic_end"]) for r in rows[:2*n_r])
    mirror = []
    for lam in (1.25, 1.5):
        a = min(range(n_r), key=lambda i: abs(load[i]["lam_mid"]-lam))
        b = 2*n_r-1-a                          # mirror interval: same stretch range traversed in reverse
        mirror.append({"lam_mid": load[a]["lam_mid"], "lam_mid_unload": rows[b]["lam_mid"],
                       "rel": rows[b]["nominal_stress_pa"]/load[a]["nominal_stress_pa"]-1})
    hold = rows[2*n_r:]
    residual = abs(np.mean([r["gauge_26_m"] for r in hold])/result["gauge_26_reference_m"]-1)
    ok = {"stress": all(abs(p["rel_error"]) <= th["stress_rel"] for p in stress),
          "gauge_jbar": jb_err <= th["gauge_jbar_abs"],
          "unload_equals_load": all(abs(m["rel"]) <= th["unload_rel"] for m in mirror),
          "residual_length": residual <= th["residual_rel"]}
    return {k: _verdict(v) for k, v in ok.items()}, {"stress_points": stress, "gauge_jbar_max_abs_error": jb_err,
                                                    "unload_vs_load": mirror, "residual_length_rel": residual,
                                                    "max_lateral_drift_m": max(r["lateral_drift_m"] for r in rows)}


def case_stretch_release(cfg, device="cpu"):
    st = cfg["tests"]["RUBBER-PHYS-06"]["setup"]
    live = uniaxial_live(cfg, device, record=True)
    mat, L = live.material, st["L_m"]
    dtc = live.control_dt
    T_ax = axial_period(L, mat)
    n_r = int(math.ceil(60*T_ax/dtc-1e-9))
    n_h = int(math.ceil(10*T_ax/dtc-1e-9))
    n_post = int(math.ceil(20*T_ax/dtc-1e-9))
    T_r = n_r*dtc
    start = time.perf_counter()
    for k in range(n_r+n_h):
        lam = cosine_path((k+1)*dtc, T_r, 1.5)
        live.advance({1: {"translation": np.array([(lam-1)*L, 0.0, 0.0])}})
    rel = live.records[-1]
    for k in range(n_post):
        live.advance({1: {"active": False}} if k == 0 else None)
    recs = live.records[1:]
    post = recs[n_r+n_h:]
    m_ref = live.scene.marker_reference
    length_ref = float(m_ref[8][0]-m_ref[0][0])
    lengths = [r["markers_m"][8][0]-r["markers_m"][0][0] for r in post]
    e_rel = rel["energies_j"]["total"]
    energies = [r["energies_j"]["total"] for r in post]
    return {"T_ax_s": T_ax, "ramp_steps": n_r, "hold_steps": n_h, "post_steps": n_post, "substeps": live.substeps,
            "dt_s": live.dt, "particles": live.scene.n, "length_reference_m": length_ref,
            "length_at_release_m": rel["markers_m"][8][0]-rel["markers_m"][0][0], "post_lengths_m": lengths,
            "mean_length_rel_error": float(np.mean(lengths)/length_ref-1), "energy_at_release_j": e_rel,
            "post_energy_total_j": energies, "energy_drift_rel": float(max(abs(e-e_rel) for e in energies)/abs(e_rel)),
            "post_jbar_range": [min(r["j_stats"]["Jbar_min"] for r in post), max(r["j_stats"]["Jbar_max"] for r in post)],
            "post_J_range": [min(r["j_stats"]["J_min"] for r in post), max(r["j_stats"]["J_max"] for r in post)],
            "invalid_flags": recs[-1]["invalid_flags"], "wall_s": time.perf_counter()-start, "device": live.device}


def evaluate_07(result, cfg):
    th = cfg["tests"]["RUBBER-PHYS-07"]["thresholds"]
    ok = {"mean_length": abs(result["mean_length_rel_error"]) <= th["length_rel"],
          "energy": result["energy_drift_rel"] <= th["energy_rel"]}
    ok["no_permanent_set"] = ok["mean_length"]
    return {k: _verdict(v) for k, v in ok.items()}, {"mean_length_rel_error": result["mean_length_rel_error"],
                                                    "energy_drift_rel": result["energy_drift_rel"],
                                                    "post_J_range": result["post_J_range"],
                                                    "invalid_flags": result["invalid_flags"]}


def evaluate_static_v3(evaluator, results, cfg):
    """v3 wrapper of evaluate_02/03/05: adds the declared validity condition (common.stable_step_guard.applies_to:
    any invalid flag of a configuration FAILs it) to the unchanged v2 criteria."""
    verdicts, measured = evaluator(results, cfg)
    if cfg["version"] >= 3:
        bad = sorted(r["label"] for r in results if r.get("invalid", 0) != 0)
        verdicts["valid"] = _verdict(not bad)
        measured = dict(measured, invalid_configurations=bad,
                        stop_rules=sorted({r.get("stop_rule", "static_stop_v2") for r in results}))
    return verdicts, measured


# ----------------------------------------------------------------------------- RUBBER-PHYS-07 (v3) and -12

def _strip_07(cfg, device, control_dt, guard, traction_end=False, **options):
    st = cfg["tests"]["RUBBER-PHYS-06"]["setup"]
    L = st["L_m"]
    spacing = st["t_m"]/st["n_t"]/2
    regions = (("end_layer", (L-spacing, -1.0, -1.0), (1.0, 1.0, 1.0)),) if traction_end else ()
    spec = rs.RubberStripSpec(length=L, width=st["W_m"], thickness=st["t_m"], n_t=st["n_t"], case="C",
                              end_mask="x", traction_regions=regions)
    return LiveRubberMaterial(spec, material=material(st["mu_pa"]), control_dt=control_dt, gravity=(0, 0, 0),
                              device=device, strict=False, stable_step_guard=guard, **options), st


def case_stretch_release_v3(cfg, variant, device="cpu", options=None):
    """RUBBER-PHYS-07 v3 (``variant`` = "ramped" / "instant", see the v3 protocol): cosine ramp to lambda, hold, then
    either a force-controlled release (grip 1 replaced in the same control step by a traction on the end particle
    layer carrying the hold force, cosine-ramped to zero) or the instantaneous removal of grip 1; then 10 fixed-free
    axial periods. One row per control step (no record files)."""
    t7 = cfg["tests"]["RUBBER-PHYS-07"]
    var = t7["setup"]["variants"][variant]
    dtc = float(t7["setup"]["control_interval_s"])
    ramped = var["release"] == "ramped_traction"
    live, st = _strip_07(cfg, device, dtc, guard_limit(cfg), traction_end=ramped, **(options or {}))
    mat, L, t = live.material, st["L_m"], st["t_m"]
    T_ax = axial_period(L, mat)
    steps = lambda periods: int(math.ceil(periods*T_ax/dtc-1e-9))                       # noqa: E731
    n_r, n_h, n_post = steps(60), steps(10), steps(20)
    n_rel = steps(60) if ramped else 0
    T_r, lam_max = n_r*dtc, float(var["lambda"])
    m_ref = live.scene.marker_reference
    length_ref = float(m_ref[8][0]-m_ref[0][0])
    s = live.solver
    rows = []

    def row(phase):
        d, mk = live.last_diagnostics, live.markers()
        work = float(s.work.numpy()[0])
        rows.append({"step": live.step_count, "phase": phase, "length_rel": float((mk[8][0]-mk[0][0])/length_ref),
                     "lateral_m": float(np.abs(mk[:, 1:]-m_ref[:, 1:]).max()) if np.isfinite(mk).all() else None,
                     "kinetic": d["kinetic_particle"], "elastic": d["elastic"], "external_work": work,
                     "total": d["kinetic_particle"]+d["elastic"]+d["gravity"]-work, "J_min": d["J_min"],
                     "J_max": d["J_max"], "Jbar_min": d["Jbar_min"], "Jbar_max": d["Jbar_max"],
                     "cfl_eff_max": d["cfl_eff_max"], "invalid": int(live.invalid_reason)})
    start = time.perf_counter()
    wrench = None
    for k in range(n_r+n_h):
        lam = cosine_path((k+1)*dtc, T_r, lam_max)
        wrench = live.advance({1: {"translation": np.array([(lam-1)*L, 0.0, 0.0])}})
        row("ramp" if k < n_r else "hold")
    hold_force = (-np.asarray(wrench)[1, :3]).tolist()       # force grip 1 exerts on the material (last hold step)
    rel = dict(rows[-1])
    if ramped:
        T_rel = n_rel*dtc
        for k in range(n_rel):
            f = hold_force[0]*(1+math.cos(math.pi*(k+0.5)*dtc/T_rel))/2
            live.advance({1: {"active": False}} if k == 0 else None, traction={0: [f, 0.0, 0.0]})
            row("release_ramp")
    for k in range(n_post):
        motions = {1: {"active": False}} if (not ramped and k == 0) else None
        live.advance(motions, traction={0: [0.0, 0.0, 0.0]} if ramped else None)
        row("post")
    after = [r for r in rows if r["phase"] in ("release_ramp", "post")]
    post = [r for r in rows if r["phase"] == "post"]
    finite = all(np.isfinite(r["total"]) for r in after)
    return {"variant": variant, "release": var["release"], "lambda": lam_max, "T_ax_s": T_ax, "control_interval_s": dtc,
            "ramp_steps": n_r, "hold_steps": n_h, "release_ramp_steps": n_rel, "post_steps": n_post,
            "substeps": live.substeps, "dt_s": live.dt, "cfl_actual": live.step_rule["cfl_actual"],
            "stable_step_guard": guard_limit(cfg), "particles": live.scene.n, "length_reference_m": length_ref,
            "hold_force_n": hold_force, "at_release": rel,
            "mean_length_rel_error": float(np.mean([r["length_rel"] for r in post])-1) if finite else None,
            "post_length_std_rel": float(np.std([r["length_rel"] for r in post])) if finite else None,
            "energy_drift_rel_psi": (float(max(abs(r["total"]-rel["total"]) for r in after)/rel["elastic"])
                                     if finite else None),
            "max_lateral_m": max((r["lateral_m"] for r in rows if r["lateral_m"] is not None), default=None),
            "thickness_m": t, "invalid_flags": decode_reason(live.invalid_reason),
            "cfl_eff_max": max(r["cfl_eff_max"] for r in rows), "J_range": [min(r["J_min"] for r in rows),
                                                                            max(r["J_max"] for r in rows)],
            "Jbar_range": [min(r["Jbar_min"] for r in rows), max(r["Jbar_max"] for r in rows)], "rows": rows,
            "solver_options": live.solver.solver_receipt()["options"], "wall_s": time.perf_counter()-start,
            "device": live.device}


def evaluate_07_v3(results, cfg):
    th = cfg["tests"]["RUBBER-PHYS-07"]["thresholds"]
    by = {r["variant"]: r for r in results}
    ok, measured = {}, {}
    for variant in cfg["tests"]["RUBBER-PHYS-07"]["setup"]["variants"]:
        r = by.get(variant)
        if r is None:
            ok[f"{variant}_ran"] = False
            continue
        good = r["mean_length_rel_error"] is not None
        ok[f"{variant}_mean_length"] = good and abs(r["mean_length_rel_error"]) <= th["length_rel"]
        ok[f"{variant}_energy"] = good and r["energy_drift_rel_psi"] <= th["energy_rel_psi"]
        ok[f"{variant}_valid"] = not r["invalid_flags"]
        ok[f"{variant}_no_fold"] = r["max_lateral_m"] is not None and r["max_lateral_m"] <= th["lateral_over_t"]*r[
            "thickness_m"]
        measured[variant] = {k: r[k] for k in ("lambda", "release", "mean_length_rel_error", "post_length_std_rel",
                                                "energy_drift_rel_psi", "max_lateral_m", "invalid_flags",
                                                "cfl_eff_max", "J_range", "Jbar_range", "hold_force_n", "dt_s",
                                                "substeps", "cfl_actual")}
    return {k: _verdict(v) for k, v in ok.items()}, measured


def case_guard_regression(cfg, device="cpu"):
    """RUBBER-PHYS-12 (v3): the v2 RUBBER-PHYS-07 path (lambda 1.5, 5 ms control interval, instantaneous release of
    grip 1, CFL 0.5) with the stable-step guard on and strict mode off; flags per control step."""
    t12 = cfg["tests"]["RUBBER-PHYS-12"]["setup"]
    dtc = float(t12["control_interval_s"])
    live, st = _strip_07(cfg, device, dtc, guard_limit(cfg))
    L = st["L_m"]
    T_ax = axial_period(L, live.material)
    steps = lambda periods: int(math.ceil(periods*T_ax/dtc-1e-9))                       # noqa: E731
    n_r, n_h, n_post = steps(60), steps(10), steps(20)
    rows = []
    start = time.perf_counter()
    for k in range(n_r+n_h+n_post):
        if k < n_r+n_h:
            lam = cosine_path((k+1)*dtc, n_r*dtc, float(t12["lambda"]))
            live.advance({1: {"translation": np.array([(lam-1)*L, 0.0, 0.0])}})
        else:
            live.advance({1: {"active": False}} if k == n_r+n_h else None)
        d = live.last_diagnostics
        rows.append({"step": live.step_count, "post_release": k >= n_r+n_h, "flags": decode_reason(live.invalid_reason),
                     "nonfinite": bool(d["nonfinite"]), "cfl_eff_max_end": d["cfl_eff_max"], "J_min": d["J_min"]})

    def first(pred):
        return next((r["step"] for r in rows if pred(r)), None)
    return {"control_interval_s": dtc, "lambda": t12["lambda"], "substeps": live.substeps, "dt_s": live.dt,
            "stable_step_guard": guard_limit(cfg), "release_after_step": n_r+n_h,
            "first_cfl_flag_step": first(lambda r: "RUBBER_CFL_EXCEEDED" in r["flags"]),
            "first_j_flag_step": first(lambda r: "RUBBER_J_DOMAIN" in r["flags"]),
            "first_nonfinite_step": first(lambda r: "NONFINITE_STATE" in r["flags"]),
            "rows": rows, "wall_s": time.perf_counter()-start, "device": live.device}


def evaluate_12(result, cfg):
    flags = [s for s in (result["first_cfl_flag_step"], result["first_j_flag_step"]) if s is not None]
    nonfinite = result["first_nonfinite_step"]
    first_flag = min(flags) if flags else None
    ok = {"flagged": first_flag is not None,
          "flag_before_nonfinite": first_flag is not None and (nonfinite is None or first_flag < nonfinite)}
    return {k: _verdict(v) for k, v in ok.items()}, {k: result[k] for k in (
        "first_cfl_flag_step", "first_j_flag_step", "first_nonfinite_step", "release_after_step", "stable_step_guard")}


# ----------------------------------------------------------------------------- RUBBER-PHYS-09 records

def configs_09(cfg):
    st = cfg["tests"]["RUBBER-PHYS-09"]["setup"]
    return [dict(scenario=sc, t=t, Wt=Wt) for sc in ("cantilever_gravity", "free_vibration", "stretch_release")
            for t in st["t_m"] for Wt in st["W_over_t"]]


def case_records(scenario, t, Wt, cfg, device="cpu"):
    st = cfg["tests"]["RUBBER-PHYS-09"]["setup"]
    L, W, n_t, steps = st["L_over_t"]*t, Wt*t, st["n_t"], st["control_steps"]
    start = time.perf_counter()
    if scenario in ("cantilever_gravity", "free_vibration"):
        spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=n_t, case="A")
        live = LiveRubberMaterial(spec, material=material(2e6), control_dt=st["control_dt_s"], gravity=(0, 0, -9.81),
                                  device=device, record=True, strict=False)
        for _ in range(steps):
            live.advance()
    else:
        spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=n_t, case="C", end_mask="x")
        live = LiveRubberMaterial(spec, material=material(4e5), control_dt=st["control_dt_s"], gravity=(0, 0, 0),
                                  device=device, record=True, strict=False)
        for k in range(steps):
            if k < 2:
                lam = cosine_path((k+1)*live.control_dt, 2*live.control_dt, 1.5)
                live.advance({1: {"translation": np.array([(lam-1)*L, 0, 0])}})
            elif k == 2:
                live.advance()
            else:
                live.advance({1: {"active": False}} if k == 3 else None)
    problems = validate_records(live.records, expected_steps=steps)
    return {"scenario": scenario, "t_m": t, "W_m": W, "L_m": L, "particles": live.scene.n, "steps": steps,
            "substeps": live.substeps, "problems": problems, "valid": not problems,
            "invalid_flags": live.records[-1]["invalid_flags"], "surface_particles": int(live.scene.surface.sum()),
            "wall_s": time.perf_counter()-start, "device": live.device,
            "record_bytes": len(json.dumps(live.records))}


def evaluate_09(results, cfg):
    ok = len(results) == 18 and all(r["valid"] for r in results)
    return {"records": _verdict(ok)}, {"runs": len(results), "valid": sum(r["valid"] for r in results),
                                       "problems": [p for r in results for p in r["problems"]][:20]}
