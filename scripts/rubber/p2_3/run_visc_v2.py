"""Formal runner of ``configs/rubber/rubber_visc_tests_v2.json`` (RUBBER-VISC-03 and -05 re-scored with the v2
estimator, inertia de-embedding and ``visco_energy = branch_work_kick_average_v2``; ADR 0023 §21.7) on the Warp CPU
device.

    CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python \
        scripts/rubber/p2_3/run_visc_v2.py --out artifacts/rl_ready/p2_3/<dir> [--workers 16] [--smoke] \
        [--v1-raw artifacts/rl_ready/p2_3/visc_v1_001/raw]

``--smoke`` truncates every schedule to 3 control steps (runner check, label SMOKE; its verdicts are not results). The
output directory must not exist. Raw per-job series in ``raw/<job>.npz`` (+ ``.json``), verdicts and numbers in
``summary.json``. Every value is SIMULATOR_ONLY_UNCALIBRATED.

Jobs: the six strip rates of -03 (v2 lengths: ``rubber_visco_reference.steady_window_v2``), the 16 Hz strip at substep
factor 2 and with g = 0 (v2 length), the -03e blocks, the -02 creep run and the -04b cantilevers (v1 schedules; booked
by -05). The dynamics columns of every run are compared byte-wise with the ``visc_v1_001`` raw series of the same job
(prefix for the longer 16 / 32 Hz runs): the protocol's validity condition ``dynamics_identity_with_v1``. The v1
helpers that do not depend on the energy option are imported from ``run_visc_v1.py`` (unchanged).
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_visc_v1 as R                     # noqa: E402  (v1 helpers: _setup, _energies, _validity, creep_load, ...)

PROTOCOL = ROOT/"configs"/"rubber"/"rubber_visc_tests_v2.json"
PROTOCOL_V1 = ROOT/"configs"/"rubber"/"rubber_visc_tests_v1.json"
DEFAULT_V1_RAW = ROOT/"artifacts"/"rl_ready"/"p2_3"/"visc_v1_001"/"raw"
STRIP, CANT = R.STRIP, R.CANT
ENERGY_COLUMNS = R.ENERGY_COLUMNS+("branch_work_left_v1",)
TAU_MAX = 0.05
RHO = 1150.0


def load_protocols():
    raw2, raw1 = PROTOCOL.read_bytes(), PROTOCOL_V1.read_bytes()
    return (json.loads(raw2), hashlib.sha256(raw2).hexdigest(), json.loads(raw1), hashlib.sha256(raw1).hexdigest())


# ============================================================================ jobs (worker processes)

def _energy():
    from robotic_cutting_sim import rubber_visco_protocol as vp
    return vp.VISCO_ENERGY_V2


def _energies(live):
    return R._energies(live)+[float(live.solver.visco_branch_work_left.numpy()[0])]


def _strip_live(p, mat, traction=False, substeps=None):
    from robotic_cutting_sim import rubber_visco_protocol as vp
    L = STRIP["L"]
    regions = (("end", (L-0.0005+1e-9, -1, -1), (1, 1, 1)),) if traction else ()
    spec = vp.strip_spec(L=L, W=STRIP["W"], t=STRIP["t"], n_t=STRIP["n_t"], case="C", traction_regions=regions)
    return vp.make_live(spec, mat, control_dt=p["control_dt"], substeps=substeps, energy=_energy())


def job_strip_cycle(p, smoke):
    """-03 strip (+ -05): grip 1 axial u(t) = eps0 L (1 - cos 2 pi f t), grip 0 holds; v2 length."""
    from robotic_cutting_sim import rubber_visco_protocol as vp
    mu0, terms = vp.prony_set("P2")
    if p.get("g_zero"):
        terms = tuple((0.0, tau) for _, tau in terms)
    mat = vp.material(mu0, terms)
    dtc, f = p["control_dt"], p["rate"]
    probe = _strip_live(p, mat)
    sub = probe.substeps*p.get("factor", 1)
    live = _strip_live(p, mat, substeps=sub) if p.get("factor", 1) != 1 else probe
    L, eps0 = STRIP["L"], p["eps0"]
    K = 3 if smoke else int(p["steps"])
    ref_x = live.scene.marker_reference[:, 0]
    rows = []
    for k in range(1, K+1):
        tk = k*dtc
        u = eps0*L*(1-math.cos(2*math.pi*f*tk))
        live.advance({1: {"translation": np.array([u, 0.0, 0.0])}})
        x = live.markers()
        eps = (x[6, 0]-x[2, 0])/(ref_x[6]-ref_x[2])-1.0
        rows.append([tk, u, eps, float(live.last_wrench[1, 0]), float(live.last_wrench[0, 0])]+_energies(live))
    return {"series": {"cols": ["t", "u", "eps", "wrench1_x", "wrench0_x"]+list(ENERGY_COLUMNS), "data": np.array(rows)},
            "info": {"particles": live.scene.n, "substeps": live.substeps, "dt": live.dt, "steps": K,
                     "marker_ref_x": [float(ref_x[2]), float(ref_x[6])], "strip_length_m": L,
                     "density": float(live.material.density), "visco_energy": live.solver.visco_energy,
                     **R._validity(live)}}


def job_block_cycle(p, smoke):
    """-03e: simple shear gamma(t) = gamma0 (1 - cos 2 pi f t) of the block, 3 cycles (v1 definition)."""
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim import rubber_visco_reference as ref
    mu0, terms = vp.prony_set("P2")
    mat = vp.material(mu0, terms)
    dtc, f, g0 = p["control_dt"], p["rate"], p["gamma0"]
    live = vp.make_live(vp.block_spec(), mat, control_dt=dtc, energy=_energy())
    K = 3 if smoke else int(math.ceil(p["cycles"]/(f*dtc)-1e-9))
    t = dtc*np.arange(K+1)

    def F_of(tt):
        F = np.eye(3)
        F[0, 1] = g0*(1-math.cos(2*math.pi*f*tt))
        return F
    blk = ref.HereditaryBlock(t, [F_of(tt) for tt in t], mu0, 20*mu0, terms)
    err, nref, ts_list = [], [], []
    for k in range(1, K+1):
        live.advance({0: {"affine": F_of(t[k])}})
        ts = t[k]-live.dt
        P = live.solver.stress.numpy().astype(np.float64)
        Pr = blk.evaluate(ts)
        err.append(float(np.sqrt(((P-Pr)**2).sum((1, 2))).max()))
        nref.append(float(np.linalg.norm(Pr)))
        ts_list.append(ts)
    return {"series": {"t_stress": ts_list, "abs_err": err, "norm_P_ref": nref, "t": t[1:].tolist()},
            "info": {"particles": live.scene.n, "substeps": live.substeps, "dt": live.dt,
                     "visco_energy": live.solver.visco_energy, **R._validity(live)}}


def job_creep(p, smoke):
    """-02 run (booked by -05): traction on the end layer at x = L, grip 1 inactive, creep then recovery."""
    from robotic_cutting_sim import rubber_visco_protocol as vp
    cfg1 = json.loads(PROTOCOL_V1.read_text())
    mu0, terms = vp.prony_set("P2")
    mat = vp.material(mu0, terms)
    live = _strip_live(p, mat, traction=True)
    dtc = p["control_dt"]
    K = 3 if smoke else int(round(cfg1["tests"]["RUBBER-VISC-02"]["setup"]["t2_s"]/dtc))
    ref_x = live.scene.marker_reference[:, 0]
    area = STRIP["W"]*STRIP["t"]
    rows = []
    for k in range(1, K+1):
        tk = k*dtc
        sig = R.creep_load(tk-dtc/2, cfg1)
        live.advance({1: {"active": False}}, traction={0: np.array([sig*area, 0.0, 0.0])})
        x = live.markers()
        eps = (x[6, 0]-x[2, 0])/(ref_x[6]-ref_x[2])-1.0
        rows.append([tk, sig, eps]+_energies(live))
    return {"series": {"cols": ["t", "sigma_applied", "eps"]+list(ENERGY_COLUMNS), "data": np.array(rows)},
            "info": {"particles": live.scene.n, "substeps": live.substeps, "dt": live.dt,
                     "traction_particles": int(live.scene.traction_groups[0]["particles"]),
                     "visco_energy": live.solver.visco_energy, **R._validity(live)}}


def job_cantilever(p, smoke):
    """-04b run (booked by -05): case A cantilever 40 x 4 x 4 mm n_t 4, grip tap z = A sin^2(pi t / T_p), free decay."""
    from robotic_cutting_sim import rubber_visco_protocol as vp
    cfg1 = json.loads(PROTOCOL_V1.read_text())
    b = cfg1["tests"]["RUBBER-VISC-04"]["setup"]["b_cantilever"]
    mu0, terms = vp.prony_set("P2")
    mat = vp.material(mu0, terms)
    dtc = b["control_dt_s"]
    spec = vp.strip_spec(L=CANT["L"], W=CANT["W"], t=CANT["t"], n_t=CANT["n_t"], case="A")
    base = R._base_substeps(spec, mat, dtc)
    live = vp.make_live(spec, mat, control_dt=dtc, substeps=base*p["factor"], energy=_energy())
    A, Tp = 2.0e-4, 0.043
    K = 3 if smoke else int(round(b["record_s"]/dtc))
    rows = []
    for k in range(1, K+1):
        tk = k*dtc
        z = A*math.sin(math.pi*tk/Tp)**2 if tk < Tp else 0.0
        live.advance({0: {"translation": np.array([0.0, 0.0, z])}})
        x = live.markers()
        rows.append([tk, z, x[8, 2], x[4, 2]]+_energies(live))
    return {"series": {"cols": ["t", "grip_z", "tip_z", "mid_z"]+list(ENERGY_COLUMNS), "data": np.array(rows)},
            "info": {"particles": live.scene.n, "substeps": live.substeps, "dt": live.dt, "base_substeps": base,
                     "tip_reference_z": float(live.scene.marker_reference[8, 2]),
                     "visco_energy": live.solver.visco_energy, **R._validity(live)}}


JOBS = {"strip_cycle": job_strip_cycle, "block_cycle": job_block_cycle, "creep": job_creep, "cantilever": job_cantilever}


def run_job(item):
    job_id, kind, params, raw_dir, smoke = item
    R._setup()
    start = time.time()
    try:
        res, status, err = JOBS[kind](params, smoke), "OK", None
    except Exception:                                   # noqa: BLE001  (preserved as a failed job)
        res, status, err = {"series": {}, "info": {}}, "ERROR", traceback.format_exc()
    wall = time.time()-start
    series = res.get("series", {})
    arrays = {k: np.asarray(v) for k, v in series.items() if k != "cols"}
    if arrays:
        np.savez_compressed(Path(raw_dir)/f"{job_id}.npz", **arrays)
    meta = {"job": job_id, "kind": kind, "params": params, "status": status, "error": err, "wall_s": wall,
            "info": res.get("info", {}), "cols": series.get("cols")}
    (Path(raw_dir)/f"{job_id}.json").write_text(json.dumps(meta, indent=1, default=R._jsonable))
    return meta


def job_list(cfg2, cfg1):
    from robotic_cutting_sim import rubber_visco_reference as ref
    s3 = cfg2["tests"]["RUBBER-VISC-03"]["setup"]
    dtc = s3["control_dt_s"]
    jobs = []
    for f in sorted(s3["fit_rates_hz"]+s3["heldout_rates_hz"]):
        w = ref.steady_window_v2(f, dtc, TAU_MAX)
        jobs.append((f"v03_strip_{f:g}hz", "strip_cycle", dict(rate=f, eps0=s3["eps0"], control_dt=dtc, **w)))
    w16 = ref.steady_window_v2(16.0, dtc, TAU_MAX)
    jobs.append(("v05_strip_16hz_x2", "strip_cycle", dict(rate=16.0, eps0=s3["eps0"], control_dt=dtc, factor=2, **w16)))
    jobs.append(("v05_strip_16hz_g0", "strip_cycle", dict(rate=16.0, eps0=s3["eps0"], control_dt=dtc, g_zero=True,
                                                         **w16)))
    for f in s3["block_finite_amplitude"]["rates_hz"]:
        jobs.append((f"v03e_block_{f:g}hz", "block_cycle", dict(rate=f, gamma0=0.15, cycles=3, control_dt=dtc)))
    jobs.append(("v02_creep", "creep", dict(control_dt=cfg1["tests"]["RUBBER-VISC-02"]["setup"]["control_dt_s"])))
    for f in cfg1["tests"]["RUBBER-VISC-04"]["setup"]["substep_factors"]:
        jobs.append((f"v04b_cantilever_x{f}", "cantilever", dict(factor=f)))
    order = {"cantilever": 0, "strip_cycle": 1, "creep": 2, "block_cycle": 3}
    return sorted(jobs, key=lambda j: (order[j[1]], -j[2].get("factor", 1), j[2].get("rate", 0)))


# ============================================================================ analysis

def _load(raw, job):
    meta = json.loads((raw/f"{job}.json").read_text())
    path = raw/f"{job}.npz"
    return meta, (dict(np.load(path)) if path.exists() else {})


def _table(meta, data):
    return {c: data["data"][:, i] for i, c in enumerate(meta["cols"])}


def _books(s, w_col):
    BN = s["kinetic"]+s["psi_eq"]+s["gravity"]+s[w_col]-s["work_ext"]
    SN = float(np.max(s["kinetic"]+s["psi_eq"])+np.max(np.abs(s[w_col])))
    return BN, SN


def identity_with_v1(raw, v1_raw, metas, require_full_prefix=True):
    """``validity.dynamics_identity_with_v1``: byte-wise comparison of every recorded v1 column except branch_work
    (prefix rows), and of the v2 left-point column with the v1 branch_work column. A smoke run compares its 3 rows
    only (``require_full_prefix`` False; reported, never a validity result)."""
    if v1_raw is None or not Path(v1_raw).exists():
        return {"status": "NOT_RUN", "reason": f"v1 raw artifacts absent: {v1_raw}"}
    v1_raw = Path(v1_raw)
    rows, ok = {}, True
    for m in metas:
        job = m["job"]
        if m["status"] != "OK" or not (v1_raw/f"{job}.json").exists():
            rows[job] = {"compared": False, "reason": "job error" if m["status"] != "OK" else "no v1 job"}
            ok = ok and m["status"] == "OK"
            continue
        m1, d1 = _load(v1_raw, job)
        m2, d2 = _load(raw, job)
        if m1.get("cols"):
            s1, s2 = _table(m1, d1), _table(m2, d2)
            n1, n2 = len(d1["data"]), len(d2["data"])
            n = min(n1, n2)
            bad = [c for c in m1["cols"] if c != "branch_work"
                   and s1[c][:n].tobytes() != s2[c][:n].tobytes()]
            left_equal = s1["branch_work"][:n].tobytes() == s2["branch_work_left_v1"][:n].tobytes()
            first_bad_row = None
            if bad:
                first_bad_row = int(np.flatnonzero(s1[bad[0]][:n] != s2[bad[0]][:n])[0]) \
                    if np.any(s1[bad[0]][:n] != s2[bad[0]][:n]) else None
            r = {"compared": True, "rows_v1": n1, "rows_v2": n2, "rows_compared": n, "full_v1_prefix": n == n1,
                 "mismatching_columns": bad, "first_mismatch_row": first_bad_row,
                 "left_point_equals_v1_branch_work": bool(left_equal),
                 "kick_paired_differs_from_v1_branch_work": bool(s1["branch_work"][:n].tobytes()
                                                                 != s2["branch_work"][:n].tobytes())}
            r["pass"] = bool(not bad and left_equal and (n == n1 or not require_full_prefix))
        else:
            if require_full_prefix:
                bad = [k for k in d1 if k not in d2 or d1[k].tobytes() != d2[k].tobytes()]
            else:
                bad = [k for k in d1 if k not in d2 or d1[k][:len(d2[k])].tobytes() != d2[k].tobytes()]
            r = {"compared": True, "arrays": sorted(d1), "mismatching_arrays": bad, "pass": not bad}
        rows[job] = r
        ok = ok and r["pass"]
    return {"status": "PASS" if ok else "FAIL", "v1_raw": str(v1_raw.relative_to(ROOT)) if v1_raw.is_relative_to(ROOT)
            else str(v1_raw), "jobs": rows}


def analyse(cfg2, cfg1, raw, metas, smoke, v1_raw):
    from scipy.optimize import least_squares
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim import rubber_visco_reference as ref
    res = {}
    status = {m["job"]: m["status"] for m in metas}
    validity = {m["job"]: m["info"].get("invalid_reason", 0) for m in metas}
    cfl_total = sum(int(m["info"].get("cfl_violations", 0)) for m in metas)
    mu0, P2 = vp.prony_set("P2")
    kappa = 20*mu0

    def valid(jobs):
        return all(status.get(j) == "OK" and validity.get(j, 0) == 0 for j in jobs)
    t3 = cfg2["tests"]["RUBBER-VISC-03"]
    s3, thr3 = t3["setup"], t3["thresholds"]
    dtc = s3["control_dt_s"]
    V = STRIP["L"]*STRIP["W"]*STRIP["t"]
    area = STRIP["W"]*STRIP["t"]
    rates = sorted(s3["fit_rates_hz"]+s3["heldout_rates_hz"])
    per = {}
    for job, f in [(f"v03_strip_{f:g}hz", f) for f in rates]+[("v05_strip_16hz_x2", 16.0), ("v05_strip_16hz_g0", 16.0)]:
        meta, d = _load(raw, job)
        if meta["status"] != "OK" or "data" not in d:
            per[job] = None
            continue
        s = _table(meta, d)
        w = 2*math.pi*f
        win_cfg = ref.steady_window_v2(f, dtc, TAU_MAX)
        K, n = win_cfg["periods"], win_cfg["window_samples"]
        tk = s["t"]
        if len(tk) < n+1:                                 # smoke: not a whole window
            n = max(1, len(tk)-1)
        win = slice(len(tk)-n, len(tk))
        first, last = len(tk)-1-n, len(tk)-1
        a, b = meta["info"]["marker_ref_x"]
        Lg, rho = meta["info"]["strip_length_m"], meta["info"]["density"]
        sig1, sig0 = -s["wrench1_x"]/area, s["wrench0_x"]/area
        Eref = complex(ref.uniaxial_complex_modulus(w, mu0, kappa, P2))
        Em1 = ref.fourier_modulus_window_v2(sig1[win], s["eps"][win], tk[win]-dtc/2, tk[win], w, dtc)
        Em0 = ref.fourier_modulus_window_v2(sig0[win], s["eps"][win], tk[win]-dtc/2, tk[win], w, dtc)
        try:
            E1, it1 = ref.deembed_rod(Em1, w, rho, Lg, a, b, 1)
            E0, it0 = ref.deembed_rod(Em0, w, rho, Lg, a, b, 0)
            converged = True
        except RuntimeError:
            E1, E0, it1, it0, converged = Em1, Em0, None, None, False
        lastcyc = tk > tk[-1]-1.0/f+1e-12
        Ev1 = ref.fourier_modulus(sig1[lastcyc], s["eps"][lastcyc], tk[lastcyc]-dtc/2, tk[lastcyc], w, dtc)
        dW = {k: float(s[k][last]-s[k][first]) for k in ("work_ext", "branch_work", "branch_work_left_v1",
                                                         "dissipation", "kinetic", "psi_eq", "psi_visc_stored")}
        loss_ref = math.pi*s3["eps0"]**2*Eref.imag*V
        BN, SN = _books(s, "branch_work")
        BL, SL = _books(s, "branch_work_left_v1")
        T1ref = ref.rod_transfer(w, Eref, rho, Lg, a, b, 1)
        row = {"rate_hz": f, "periods_in_window": K, "samples_in_window": n, "steps": len(tk),
               "window_start_s": float(tk[first]), "marker_ref_x_m": [a, b],
               "E1_pa": E1.real, "E2_pa": E1.imag, "E1_ref_pa": Eref.real, "E2_ref_pa": Eref.imag,
               "E1_rel": E1.real/Eref.real-1, "E2_rel": E1.imag/Eref.imag-1,
               "deembed_iterations_grip1": it1, "deembed_iterations_grip0": it0, "deembed_converged": converged,
               "grip0_E1_rel": E0.real/Eref.real-1, "grip0_E2_rel": E0.imag/Eref.imag-1,
               "deembed_consistency_rel": E0.real/E1.real-1,
               "loss_cycle_j": dW["work_ext"]/K, "loss_ref_j": loss_ref, "loss_rel": dW["work_ext"]/K/loss_ref-1,
               "reported": {"E_meas_not_deembedded_rel": [Em1.real/Eref.real-1, Em1.imag/Eref.imag-1],
                            "transfer_T1_at_Eref": [T1ref.real, T1ref.imag],
                            "v1_last_cycle_estimator_rel": [Ev1.real/Eref.real-1, Ev1.imag/Eref.imag-1],
                            "v1_last_cycle_samples": int(lastcyc.sum())},
               "window": dW,
               "cycle_books_rel": dW["work_ext"]/dW["branch_work"]-1 if dW["branch_work"] else None,
               "cycle_dissipation_rel": dW["dissipation"]/dW["branch_work"]-1 if dW["branch_work"] else None,
               "cycle_books_rel_left_point_v1_reported": (dW["work_ext"]/dW["branch_work_left_v1"]-1
                                                          if dW["branch_work_left_v1"] else None),
               "books_max_abs_rel": float(np.max(np.abs(BN))/SN), "S_N_j": SN,
               "books_trend_rel_per_s": float(np.polyfit(tk, BN/SN, 1)[0]),
               "books_left_point_v1_max_abs_rel_reported": float(np.max(np.abs(BL))/SL),
               "branch_identity_end_rel_reported": float((s["branch_work"][-1]-s["dissipation"][-1]
                                                          - s["psi_visc_stored"][-1])/SN),
               "d_negative": float(np.max(s["dissipation_negative"])),
               "cfl_visco_max": float(np.max(s["cfl_visco_max"]))}
        row["pass_a"] = bool(converged and abs(row["E1_rel"]) <= thr3["e1_rel"] and abs(row["E2_rel"]) <= thr3["e2_rel"]
                             and abs(row["loss_rel"]) <= thr3["loss_rel"]
                             and abs(row["deembed_consistency_rel"]) <= thr3["deembed_consistency_rel"])
        row["pass_a_validity_deembedding"] = bool(converged and abs(row["deembed_consistency_rel"])
                                                  <= thr3["deembed_consistency_rel"])
        per[job] = row
    rows3 = [per.get(f"v03_strip_{f:g}hz") for f in rates]
    ok3a = all(r and r["pass_a"] for r in rows3)
    ident = None
    if all(r for r in rows3):
        sim = {r["rate_hz"]: complex(r["E1_pa"], r["E2_pa"]) for r in rows3}

        def Emodel(x, f, M=2):
            return ref.uniaxial_complex_modulus(2*math.pi*f, mu0, kappa, [(x[2*i], math.exp(x[2*i+1])) for i in range(M)])

        def resid(x, rates_, M=2):
            out = []
            for f in rates_:
                Em, Es = Emodel(x, f, M), sim[f]
                out += [Em.real/Es.real-1, Em.imag/Es.imag-1]
            return out
        fit = least_squares(lambda x: resid(x, s3["fit_rates_hz"]), [0.25, math.log(3e-3), 0.1, math.log(0.1)],
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
        terms_fit = sorted([(fit.x[0], math.exp(fit.x[1])), (fit.x[2], math.exp(fit.x[3]))], key=lambda z: z[1])
        decl = sorted(P2, key=lambda z: z[1])
        b_checks = [{"term": i+1, "g_fit": g, "tau_fit_s": tau, "g_rel": g/decl[i][0]-1, "tau_rel": tau/decl[i][1]-1}
                    for i, (g, tau) in enumerate(terms_fit)]
        ok_b3 = all(abs(c["g_rel"]) <= thr3["g_rel"] and abs(c["tau_rel"]) <= thr3["tau_rel"] for c in b_checks)
        xf = [terms_fit[0][0], math.log(terms_fit[0][1]), terms_fit[1][0], math.log(terms_fit[1][1])]
        held = [{"rate_hz": f, "E1_pred_rel": Emodel(xf, f).real/sim[f].real-1,
                 "E2_pred_rel": Emodel(xf, f).imag/sim[f].imag-1} for f in s3["heldout_rates_hz"]]
        ok_c3 = all(abs(h["E1_pred_rel"]) <= thr3["e1_heldout"] and abs(h["E2_pred_rel"]) <= thr3["e2_heldout"]
                    for h in held)
        one = least_squares(lambda x: resid(x, [4.0], M=1), [0.3, math.log(0.01)], xtol=1e-15, ftol=1e-15)
        single = {"g": one.x[0], "tau_s": math.exp(one.x[1]),
                  "heldout": [{"rate_hz": f, "E1_pred_rel": Emodel(one.x, f, 1).real/sim[f].real-1,
                               "E2_pred_rel": Emodel(one.x, f, 1).imag/sim[f].imag-1} for f in s3["heldout_rates_hz"]]}
        ident = {"b": {"terms": b_checks, "pass": bool(ok_b3), "cost": float(fit.cost)},
                 "c": {"heldout": held, "pass": bool(ok_c3)}, "d_single_rate_control_reported": single}
    blk = []
    for f in s3["block_finite_amplitude"]["rates_hz"]:
        job = f"v03e_block_{f:g}hz"
        meta, d = _load(raw, job)
        if meta["status"] != "OK" or "abs_err" not in d:
            blk.append({"job": job, "status": "ERROR", "pass": False})
            continue
        tk = d["t"]
        win = tk > tk[-1]-1.0/f+1e-12
        rel = float(d["abs_err"][win].max()/d["norm_P_ref"][win].max())
        blk.append({"job": job, "rate_hz": f, "max_rel": rel, "samples": int(win.sum()),
                    "pass": bool(rel <= thr3["block_rel"])})
    ok_e3 = all(b.get("pass") for b in blk)
    jobs3 = [f"v03_strip_{f:g}hz" for f in rates]+[b["job"] for b in blk]
    ok3 = ok3a and ident is not None and ident["b"]["pass"] and ident["c"]["pass"] and ok_e3 and valid(jobs3)
    res["RUBBER-VISC-03"] = {"a_per_rate": rows3, "identification": ident, "e_block": blk, "thresholds": thr3,
                             "pass": {"a": bool(ok3a), "a_validity_deembedding": all(r and r["pass_a_validity_deembedding"]
                                                                                     for r in rows3),
                                      "b": bool(ident and ident["b"]["pass"]), "c": bool(ident and ident["c"]["pass"]),
                                      "e": bool(ok_e3), "valid_jobs": valid(jobs3)},
                             "verdict": "PASS" if ok3 else "FAIL"}
    # ---------------- -05
    thr5 = cfg2["tests"]["RUBBER-VISC-05"]["thresholds"]
    books = {}
    for job in [f"v03_strip_{f:g}hz" for f in rates]+["v05_strip_16hz_x2", "v02_creep"]+[
            f"v04b_cantilever_x{f}" for f in (1, 2, 4)]:
        meta, d = _load(raw, job)
        if meta["status"] != "OK" or "data" not in d:
            books[job] = None
            continue
        s = _table(meta, d)
        BN, SN = _books(s, "branch_work")
        BL, SL = _books(s, "branch_work_left_v1")
        books[job] = {"max_abs_rel": float(np.max(np.abs(BN))/SN), "max_abs_j": float(np.max(np.abs(BN))), "S_N_j": SN,
                      "end_rel": float(BN[-1]/SN), "trend_rel_per_s": float(np.polyfit(s["t"], BN/SN, 1)[0]),
                      "d_negative": float(np.max(s["dissipation_negative"])),
                      "left_point_v1_reported": {"max_abs_rel": float(np.max(np.abs(BL))/SL), "S_N_j": SL,
                                                 "end_rel": float(BL[-1]/SL)},
                      "branch_identity_end_rel_reported": float((s["branch_work"][-1]-s["dissipation"][-1]
                                                                 - s["psi_visc_stored"][-1])/SN),
                      "branch_identity_left_point_end_rel_reported": float((s["branch_work_left_v1"][-1]
                                                                            - s["dissipation"][-1]
                                                                            - s["psi_visc_stored"][-1])/SN)}
    ok5a = all(b and b["max_abs_rel"] <= thr5["books_rel"] for b in books.values())
    b1, b2 = books.get("v03_strip_16hz"), books.get("v05_strip_16hz_x2")
    cons, ok5b = None, False
    if b1 and b2:
        cons = {"ratio": b2["max_abs_j"]/b1["max_abs_j"], "factor1_rel": b1["max_abs_rel"],
                "factor2_rel": b2["max_abs_rel"]}
        ok5b = cons["ratio"] <= thr5["ratio_max"] or b2["max_abs_rel"] <= thr5["floor_rel"]
    cyc = [{"rate_hz": r["rate_hz"], "books_rel": r["cycle_books_rel"], "dissipation_rel": r["cycle_dissipation_rel"],
            "books_rel_left_point_v1_reported": r["cycle_books_rel_left_point_v1_reported"]} for r in rows3 if r]
    ok5c = len(cyc) == len(rates) and all(c["books_rel"] is not None and abs(c["books_rel"]) <= thr5["cycle_books_rel"]
                                          for c in cyc)
    g0 = per.get("v05_strip_16hz_g0")
    neg_all = {m["job"]: None for m in metas}
    for job in neg_all:
        meta, d = _load(raw, job)
        if meta["status"] == "OK" and "data" in d and "dissipation_negative" in (meta["cols"] or []):
            neg_all[job] = float(np.max(_table(meta, d)["dissipation_negative"]))
    ok5d = (len(cyc) == len(rates) and all(c["dissipation_rel"] is not None
                                           and abs(c["dissipation_rel"]) <= thr5["dissipation_rel"] for c in cyc)
            and all(v == 0 for v in neg_all.values() if v is not None))
    e5 = {"books_max_abs_rel": g0["books_max_abs_rel"], "trend_rel_per_s": g0["books_trend_rel_per_s"]} if g0 else None
    jobs5 = list(books)+["v05_strip_16hz_g0"]
    res["RUBBER-VISC-05"] = {"a_books": books, "b_consistency": cons, "c_cycle_books": cyc,
                             "d_dissipation": {"cycles": cyc, "negative_increments": neg_all},
                             "e_elastic_control_reported": e5, "thresholds": thr5,
                             "pass": {"a": bool(ok5a), "b": bool(ok5b), "c": bool(ok5c), "d": bool(ok5d),
                                      "valid_jobs": valid(jobs5)},
                             "verdict": "PASS" if ok5a and ok5b and ok5c and ok5d and valid(jobs5) else "FAIL"}
    res["guard_violations_total"] = cfl_total
    res["_validity"] = {"invalid_jobs": {j: v for j, v in validity.items() if v},
                        "errors": {m["job"]: m["error"] for m in metas if m["status"] != "OK"}}
    ident_v1 = identity_with_v1(raw, v1_raw, metas, require_full_prefix=not smoke)
    res["validity_dynamics_identity_with_v1"] = ident_v1
    if ident_v1["status"] == "FAIL" and not smoke:
        for k in ("RUBBER-VISC-03", "RUBBER-VISC-05"):
            res[k]["verdict_before_validity"] = res[k]["verdict"]
            res[k]["verdict"] = "INVALID (dynamics differ from visc_v1_001)"
    res["consistency_reported"] = consistency_v1_tests(cfg1, raw)
    if smoke:
        for k, v in res.items():
            if isinstance(v, dict) and "verdict" in v:
                v["verdict"] = f"SMOKE ({v['verdict']})"
    return res


def consistency_v1_tests(cfg1, raw):
    """-02 and -04b numbers recomputed on the v2 runs with the v1 analysis (reported; the dynamics identity makes them
    equal to the visc_v1_001 values)."""
    from scipy.optimize import least_squares
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim import rubber_visco_reference as ref
    mu0, P2 = vp.prony_set("P2")
    kappa = 20*mu0
    out = {}
    meta, d = _load(raw, "v02_creep")
    if meta["status"] == "OK" and "data" in d and len(d["data"]) > 10:
        s2 = cfg1["tests"]["RUBBER-VISC-02"]["setup"]
        s = _table(meta, d)
        tk, eps, sig = s["t"], s["eps"], s["sigma_applied"]
        dtc = s2["control_dt_s"]
        eps_ref = ref.boltzmann_staircase(tk, tk-dtc, sig, lambda tt: ref.creep_uniaxial(tt, mu0, kappa, P2))
        E0 = 9*kappa*mu0/(3*kappa+mu0)
        Tax = 4*STRIP["L"]/math.sqrt(E0/1150.0)
        Tr, t1, t2 = s2["T_r_s"], s2["t1_s"], s2["t2_s"]
        w1 = (tk >= Tr+10*Tax-1e-12) & (tk <= t1+1e-12)
        w2 = (tk >= t1+Tr+10*Tax-1e-12) & (tk <= t2+1e-12)
        emax = float(np.max(np.abs(eps_ref)))
        dev = np.abs(eps-eps_ref)
        i_star, i1, i2 = int(np.argmax(tk >= Tr+10*Tax-1e-12)), int(np.argmin(np.abs(tk-t1))), int(np.argmin(np.abs(tk-t2)))
        out["RUBBER-VISC-02"] = {"curve_max_dev_rel_window1": float(dev[w1].max()/emax),
                                 "curve_max_dev_rel_window2": float(dev[w2].max()/emax),
                                 "creep_ratio_rel": float((eps[i1]/eps[i_star])/(eps_ref[i1]/eps_ref[i_star])-1),
                                 "recovery_dev_rel": float(dev[i2]/emax)}
    root = ref.beam_root(CANT["L"], CANT["W"], CANT["t"], 1150.0, mu0, kappa, P2)
    fits = {}
    for f in (1, 2, 4):
        meta, d = _load(raw, f"v04b_cantilever_x{f}")
        if meta["status"] != "OK" or "data" not in d or len(d["data"]) < 50:
            continue
        s = _table(meta, d)
        t0 = 0.043+0.5/root["f_d_hz"]
        sel = s["t"] >= t0
        tt, z = s["t"][sel], s["tip_z"][sel]-meta["info"]["tip_reference_z"]
        amp = np.abs(z-z.mean()).max()

        def damped(x, tt):
            c, B, lam, A, zeta, w, phi = x
            return c+B*np.exp(-lam*(tt-t0))+A*np.exp(-zeta*(tt-t0))*np.cos(w*(tt-t0)+phi)
        best = None
        for phi0 in np.linspace(0, 2*math.pi, 8, endpoint=False):
            x0 = [z.mean(), 0.0, 20.0, amp, root["zeta_1_s"], 2*math.pi*root["f_d_hz"], phi0]
            sol = least_squares(lambda x: (damped(x, tt)-z)/amp, x0, xtol=1e-14, ftol=1e-14, max_nfev=20000)
            if best is None or sol.cost < best.cost:
                best = sol
        fits[f] = {"zeta_1_s": float(best.x[4]), "f_d_hz": float(best.x[5]/(2*math.pi))}
    if len(fits) == 3:
        out["RUBBER-VISC-04b"] = {"fits": fits, "zeta_rel_x2": fits[2]["zeta_1_s"]/fits[1]["zeta_1_s"]-1,
                                  "zeta_rel_x4": fits[4]["zeta_1_s"]/fits[1]["zeta_1_s"]-1,
                                  "f_rel_x2": fits[2]["f_d_hz"]/fits[1]["f_d_hz"]-1,
                                  "f_rel_x4": fits[4]["f_d_hz"]/fits[1]["f_d_hz"]-1,
                                  "f_vs_root": fits[1]["f_d_hz"]/root["f_d_hz"]-1,
                                  "zeta_vs_root": fits[1]["zeta_1_s"]/root["zeta_1_s"]-1}
    out["note"] = "reported only (v1 definitions; the v1 verdicts of -02 / -04 stand)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--v1-raw", default=str(DEFAULT_V1_RAW))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    cfg2, sha2, cfg1, sha1 = load_protocols()
    if cfg2["supersedes"]["sha256"] != sha1:
        sys.exit("protocol v1 hash differs from the one v2 supersedes")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    raw = out/"raw"
    raw.mkdir(parents=True)
    jobs = job_list(cfg2, cfg1)
    start = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    metas = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, (jid, kind, params, str(raw), args.smoke)): jid for jid, kind, params in jobs}
        for fut in as_completed(futs):
            meta = fut.result()
            metas.append(meta)
            print(f"[{time.time()-start:8.1f} s] {meta['job']}: {meta['status']} ({meta['wall_s']:.1f} s)", flush=True)
    results = analyse(cfg2, cfg1, raw, metas, args.smoke, args.v1_raw)
    import scipy
    import warp as wp
    summary = {"schema": "rcs_rubber_visc_run_v2", "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha2,
               "protocol_v1_sha256": sha1, "label": "SMOKE (runner check; not a result)" if args.smoke else "FORMAL",
               "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED (illustrative Prony sets, not fitted to real data)",
               "commit": commit, "dirty_tree": bool(dirty), "command": " ".join(sys.argv), "workers": args.workers,
               "device": "Warp CPU (CUDA_VISIBLE_DEVICES='')", "started": started,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "wall_s": time.time()-start,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "warp": wp.config.version},
               "jobs": sorted([{k: m[k] for k in ("job", "kind", "status", "wall_s")}
                               | {"particles": m["info"].get("particles"), "substeps": m["info"].get("substeps"),
                                  "dt": m["info"].get("dt"), "steps": m["info"].get("steps")} for m in metas],
                              key=lambda r: r["job"]),
               "results": results}
    (out/"summary.json").write_text(json.dumps(summary, indent=1, default=R._jsonable))
    for k, v in results.items():
        if isinstance(v, dict) and "verdict" in v:
            print(k, v["verdict"])
    print("dynamics_identity_with_v1", results["validity_dynamics_identity_with_v1"]["status"])


if __name__ == "__main__":
    main()
