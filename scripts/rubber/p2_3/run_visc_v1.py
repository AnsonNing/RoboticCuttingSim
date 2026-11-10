"""Formal runner of ``configs/rubber/rubber_visc_tests_v1.json`` (RUBBER-VISC-01..09 on the Warp CPU device; -10 is the
GPU cost script ``bench_visc_cost.py``; ADR 0023 §21.3-21.4, task R3.7).

    CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python \
        scripts/rubber/p2_3/run_visc_v1.py --out artifacts/rl_ready/p2_3/<dir> [--workers 16] [--smoke]

``--smoke`` truncates every schedule to 3 control steps (a runner check; its verdicts are not results, label SMOKE).
The output directory must not exist (no overwrite). Raw per-job series in ``raw/<job>.npz`` (+ ``.json``), verdicts and
numbers in ``summary.json``. Every value is SIMULATOR_ONLY_UNCALIBRATED: the Prony sets are illustrative, not fitted to
real data; RUBBER-VISC-03(b/c) is a synthetic dry run of the P2-5 fit / held-out split, not a calibration.

Interpretations recorded in the summary (where the protocol leaves a detail open): the "hold" of -01 is every control
step k >= 2 (stress time t_k - dt >= the ramp end); the "steady cycle" of -03 is the samples with t_k in
(t_end - 1/f, t_end] (16 and 32 Hz have 312.5 / 156.25 samples per period: the window has floor(...) samples); the
"current state" of -06's second schedule is the env's grip position at step 20 (the stretch is applied as an
increment); -07's case A strips are 20 / 16 / 24 mm (W 4, t 4 mm, n_t 4), case C the protocol strip 30 / 24 / 18 mm,
the block 4 mm; 5 control steps of 2e-4 s each.
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
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_visc_tests_v1.json"
STRIP = dict(L=0.030, W=0.006, t=0.003, n_t=3)
CANT = dict(L=0.040, W=0.004, t=0.004, n_t=4)


def load_protocol():
    raw = PROTOCOL.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


# ============================================================================ jobs (run in worker processes)

def _setup():
    import warp as wp
    wp.config.log_level = 30
    wp.init()


def _energies(live):
    d, v, s = live.last_diagnostics, live.last_visco, live.solver
    return [d["kinetic_particle"], v["psi_eq"], d["gravity"], v["branch_work"], float(s.work.numpy()[0]),
            v["dissipation"], v["psi_visc_stored"], float(s.work_end_velocity.numpy()[0]),
            v["dissipation_negative_increments"], v["cfl_visco_max"], d["elastic"]]


ENERGY_COLUMNS = ("kinetic", "psi_eq", "gravity", "branch_work", "work_ext", "dissipation", "psi_visc_stored",
                  "work_end_velocity", "dissipation_negative", "cfl_visco_max", "elastic_mu0")


def _validity(live):
    return {"invalid_reason": int(live.invalid_reason), "invalid_flags": list(_decode(live.invalid_reason)),
            "j_domain_violations": int(live.solver.violations.numpy()[0]),
            "cfl_violations": int(live.solver.cfl_violations.numpy()[0])}


def _decode(v):
    from robotic_cutting_sim.live_rubber_material import decode_reason
    return decode_reason(v)


def _base_substeps(spec, mat, control_dt):
    from robotic_cutting_sim import rubber_visco_protocol as vp
    return vp.make_live(spec, mat, control_dt=control_dt).substeps


def job_block_step(p, smoke):
    """RUBBER-VISC-01 / -04a: one-control-step ramp of the block, hold; per-particle PK1 vs the hereditary integral."""
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim import rubber_visco_reference as ref
    mu0, terms = vp.prony_set(p["set"])
    mat = vp.material(mu0, terms)
    dtc = p["control_dt"]
    spec = vp.block_spec()
    base = _base_substeps(spec, mat, dtc)
    live = vp.make_live(spec, mat, control_dt=dtc, substeps=base*p["factor"])
    if p["path"] == "uniaxial_isochoric":
        lam = 1.5
        A = np.diag([lam, lam**-0.5, lam**-0.5])
    else:
        A = np.eye(3)
        A[0, 1] = 1.0
    steps = 3 if smoke else int(round(p["hold_until"]/dtc))
    blk = ref.HereditaryBlock([0.0, dtc, max(steps, 2)*dtc], [np.eye(3), A, A], mu0, 20*mu0, terms)
    norm = float(np.linalg.norm(blk.evaluate(dtc)))
    t_stress, err, err_step, p11, pref11 = [], [], [], [], []
    for k in range(1, steps+1):
        live.advance({0: {"affine": A}})
        ts = k*dtc-live.dt
        P = live.solver.stress.numpy().astype(np.float64)
        Pr = blk.evaluate(ts)
        t_stress.append(ts)
        err.append(float(np.sqrt(((P-Pr)**2).sum((1, 2))).max()/norm))
        err_step.append(float(np.linalg.norm(Pr-blk.step_closed_form(ts, A, dtc))/norm) if ts >= dtc else np.nan)
        p11.append(float(P[:, 0, 0].mean()))
        pref11.append(float(Pr[0, 0]))
    return {"series": {"t_stress": t_stress, "rel_err": err, "closed_form_step_vs_hereditary": err_step,
                       "P11_mean": p11, "P11_ref": pref11},
            "info": {"particles": live.scene.n, "substeps": live.substeps, "dt": live.dt, "norm_P_ref_ramp": norm,
                     **_validity(live)}}


def job_block_cycle(p, smoke):
    """RUBBER-VISC-03e: simple shear gamma(t) = gamma0 (1 - cos 2 pi f t) of the block, 3 cycles."""
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim import rubber_visco_reference as ref
    mu0, terms = vp.prony_set("P2")
    mat = vp.material(mu0, terms)
    dtc, f, g0 = p["control_dt"], p["rate"], p["gamma0"]
    live = vp.make_live(vp.block_spec(), mat, control_dt=dtc)
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
            "info": {"particles": live.scene.n, "substeps": live.substeps, "dt": live.dt, **_validity(live)}}


def _strip_live(p, mat, traction=False, substeps=None):
    from robotic_cutting_sim import rubber_visco_protocol as vp
    L = STRIP["L"]
    regions = (("end", (L-0.0005+1e-9, -1, -1), (1, 1, 1)),) if traction else ()
    spec = vp.strip_spec(L=L, W=STRIP["W"], t=STRIP["t"], n_t=STRIP["n_t"], case="C", traction_regions=regions)
    return vp.make_live(spec, mat, control_dt=p["control_dt"], substeps=substeps)


def job_strip_cycle(p, smoke):
    """RUBBER-VISC-03 strip (+ -05): grip 1 axial u(t) = eps0 L (1 - cos 2 pi f t), grip 0 holds."""
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
    K = 3 if smoke else int(math.ceil(p["cycles"]/(f*dtc)-1e-9))
    ref_x = live.scene.marker_reference[:, 0]
    rows = []
    for k in range(1, K+1):
        tk = k*dtc
        u = eps0*L*(1-math.cos(2*math.pi*f*tk))
        live.advance({1: {"translation": np.array([u, 0.0, 0.0])}})
        x = live.markers()
        eps = (x[6, 0]-x[2, 0])/(ref_x[6]-ref_x[2])-1.0
        rows.append([tk, u, eps, float(live.last_wrench[1, 0]), float(live.last_wrench[0, 0])]+_energies(live))
    a = np.array(rows)
    return {"series": {"cols": ["t", "u", "eps", "wrench1_x", "wrench0_x"]+list(ENERGY_COLUMNS), "data": a},
            "info": {"particles": live.scene.n, "substeps": live.substeps, "dt": live.dt, "steps": K,
                     "marker_ref_x": [float(ref_x[2]), float(ref_x[6])], **_validity(live)}}


def creep_load(t, cfg):
    s = cfg["tests"]["RUBBER-VISC-02"]["setup"]
    s0, Tr, t1, t2 = s["sigma0_pa"], s["T_r_s"], s["t1_s"], s["t2_s"]
    if t < Tr:
        return s0*(1-math.cos(math.pi*t/Tr))/2
    if t < t1:
        return s0
    if t < t1+Tr:
        return s0*(1+math.cos(math.pi*(t-t1)/Tr))/2
    return 0.0


def job_creep(p, smoke):
    """RUBBER-VISC-02: traction on the end layer at x = L, grip 1 inactive, creep then recovery."""
    from robotic_cutting_sim import rubber_visco_protocol as vp
    cfg, _ = load_protocol()
    mu0, terms = vp.prony_set("P2")
    mat = vp.material(mu0, terms)
    live = _strip_live(p, mat, traction=True)
    dtc = p["control_dt"]
    K = 3 if smoke else int(round(cfg["tests"]["RUBBER-VISC-02"]["setup"]["t2_s"]/dtc))
    ref_x = live.scene.marker_reference[:, 0]
    area = STRIP["W"]*STRIP["t"]
    rows = []
    for k in range(1, K+1):
        tk = k*dtc
        sig = creep_load(tk-dtc/2, cfg)
        live.advance({1: {"active": False}}, traction={0: np.array([sig*area, 0.0, 0.0])})
        x = live.markers()
        eps = (x[6, 0]-x[2, 0])/(ref_x[6]-ref_x[2])-1.0
        rows.append([tk, sig, eps]+_energies(live))
    return {"series": {"cols": ["t", "sigma_applied", "eps"]+list(ENERGY_COLUMNS), "data": np.array(rows)},
            "info": {"particles": live.scene.n, "substeps": live.substeps, "dt": live.dt,
                     "traction_particles": int(live.scene.traction_groups[0]["particles"]), **_validity(live)}}


def job_cantilever(p, smoke):
    """RUBBER-VISC-04b (+ -05): case A cantilever 40 x 4 x 4 mm n_t 4, grip tap z = A sin^2(pi t / T_p), free decay."""
    from robotic_cutting_sim import rubber_visco_protocol as vp
    cfg, _ = load_protocol()
    b = cfg["tests"]["RUBBER-VISC-04"]["setup"]["b_cantilever"]
    mu0, terms = vp.prony_set("P2")
    mat = vp.material(mu0, terms)
    dtc = b["control_dt_s"]
    spec = vp.strip_spec(L=CANT["L"], W=CANT["W"], t=CANT["t"], n_t=CANT["n_t"], case="A")
    base = _base_substeps(spec, mat, dtc)
    live = vp.make_live(spec, mat, control_dt=dtc, substeps=base*p["factor"])
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
                     "tip_reference_z": float(live.scene.marker_reference[8, 2]), **_validity(live)}}


def job_reset_memory(p, smoke):
    """RUBBER-VISC-06 (protocol setup)."""
    from robotic_cutting_sim import rubber_visco_constitutive as rvc
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim.batched import rubber_material_batch as rmb
    from robotic_cutting_sim.rubber_material import MaterialValues
    mu0, P2 = vp.prony_set("P2")
    _, SLS = vp.prony_set("SLS")
    dtc = 2e-4
    step_mat = MaterialValues(mu0=1.4*mu0, kappa=20*1.4*mu0, density=1050.0)
    envs = [dict(L=0.030, mat=vp.material(mu0, P2)), dict(L=0.030, mat=vp.material(mu0, SLS, M=2)),
            dict(L=0.024, mat=vp.material(1.4*mu0, P2, rho=1050.0))]

    def live(e, mat=None):
        spec = vp.strip_spec(L=envs[e]["L"], W=STRIP["W"], t=STRIP["t"], n_t=STRIP["n_t"], case="C")
        return vp.make_live(spec, mat or envs[e]["mat"], control_dt=dtc, step_material=step_mat)
    ranges = {"prony_g_scale": (0.5, 1.2), "prony_tau_scale": (0.5, 2.0)}
    batch = rmb.BatchedRubberMaterial([live(e) for e in range(3)], device="cpu", parity_mode=True, ranges=ranges)
    fresh = rmb.BatchedRubberMaterial([live(e) for e in range(3)], device="cpu", parity_mode=True, ranges=ranges)
    n1 = 3 if smoke else 10
    out = {}

    def tr(base, k):
        return np.array([[[0, 0, 0], [b+0.1*envs[e]["L"]*min(k, n1)/n1, 0, 0]] for e, b in enumerate(base)], float)
    for k in range(1, 2*n1+1):
        batch.advance(translation=tr([0, 0, 0], k))
    out["visco_q_env1_nonzero_before_reset"] = bool(np.any(batch.solver.snapshot(1)["visco_q"]))
    before = {e: batch.solver.snapshot(e) for e in (0, 2)}
    held = [0.1*envs[e]["L"] for e in range(3)]
    batch.reset([1])
    snap, fsnap = batch.solver.snapshot(1), fresh.solver.snapshot(1)
    out["env1_equals_fresh_batch_after_reset"] = [n for n in snap if n != "index" and not vp.same_bytes(snap[n],
                                                                                                         fsnap[n])]
    out["envs_0_2_unchanged_by_reset"] = [(e, n) for e in (0, 2) for n, v in batch.solver.snapshot(e).items()
                                          if n != "index" and not vp.same_bytes(v, before[e][n])]
    single = live(1)
    base = [held[0], 0.0, held[2]]
    mism_fresh, mism_single = [], []
    for k in range(1, 2*n1+1):
        batch.advance(translation=tr(base, k))
        fresh.advance(translation=tr([0, 0, 0], k))
        single.advance({1: {"translation": tr(base, k)[1, 1]}})
        sb, sf = batch.solver.snapshot(1), fresh.solver.snapshot(1)
        mism_fresh += [(k, n) for n in sb if not vp.same_bytes(sb[n], sf[n])]
        mism_fresh += [(k, "visco_diagnostics")] if not vp.same_bytes(batch.visco_summary.numpy()[1],
                                                                    fresh.visco_summary.numpy()[1]) else []
        mism_single += [(k, n) for n in vp.compare_env(batch, 1, single)]
    out["env1_after_reset_equals_fresh_batch_20_steps"] = mism_fresh
    out["env1_after_reset_equals_fresh_single_20_steps"] = mism_single
    # resampled Prony terms (g x 0.8, tau x 1.5)
    m1 = envs[1]["mat"]
    scaled = rmb.scale_prony(m1.prony, 0.8, 1.5)
    m2 = MaterialValues(mu0=m1.mu0, kappa=m1.kappa, density=m1.density, prony=scaled)
    batch.reset([1], materials=[m2])
    row = batch.solver.params64.numpy()[1]
    out["params64_resampled_equals_host_expm1"] = bool(row.tobytes() == rvc.params64_row(scaled,
                                                                                         batch.solver.dt).tobytes())
    single2 = live(1, m2)
    mism_resampled = []
    hold2 = [2*held[0], 0.0, 2*held[2]]                  # envs 0 and 2 hold where the second schedule left them
    for k in range(1, (3 if smoke else 10)+1):
        tr3 = tr([0, 0, 0], k)
        for e in (0, 2):
            tr3[e, 1, 0] = hold2[e]
        batch.advance(translation=tr3)
        single2.advance({1: {"translation": tr3[1, 1]}})
        mism_resampled += [(k, n) for n in vp.compare_env(batch, 1, single2)]
    out["resampled_reset_equals_single_built_with_it"] = mism_resampled
    batch.reset([1])
    out["plain_reset_restores_packed_params64"] = bool(batch.solver.params64.numpy()[1].tobytes()
                                                       == fresh.initial_params64.numpy()[1].tobytes())
    # single env: reset() clears and two episodes repeat
    s = live(1)
    episodes = []
    for _ in range(2):
        s.reset()
        cleared = not (np.any(s.solver.visco_q.numpy()) or np.any(s.solver.visco_s_prev.numpy()))
        snaps = []
        for k in range(1, (3 if smoke else 10)+1):
            s.advance({1: {"translation": tr([0, 0, 0], k)[1, 1]}})
            snaps.append({n: a.numpy().copy() for n, a in s.solver.device_arrays().items()})
        episodes.append((cleared, snaps))
    out["single_reset_clears"] = bool(episodes[0][0] and episodes[1][0])
    out["single_two_episodes_byte_identical"] = [(k, n) for k, (a, b) in enumerate(zip(episodes[0][1],
                                                                                       episodes[1][1]))
                                                 for n in a if not vp.same_bytes(a[n], b[n])]
    out["invalid_reason"] = batch.invalid_reason.numpy().tolist()
    out["cfl_violations"] = batch.solver.cfl_violations.numpy().tolist()
    return {"series": {}, "info": out}


def _l0_case(case, kws, steps, inactive_step=None, smoke=False):
    from scipy.spatial.transform import Rotation
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim.batched import rubber_material_batch as rmb
    from robotic_cutting_sim.rubber_material import MaterialValues
    step_mat = MaterialValues(mu0=8e5, kappa=1.6e7, density=1000.0)

    def live(kw):
        if case == "block":
            spec = vp.block_spec()
        elif case == "A":
            spec = vp.strip_spec(L=kw["L"], W=0.004, t=0.004, n_t=4, case="A",
                                 traction_regions=(("tip", (kw["L"]-0.0005+1e-9, -1, -1), (1, 1, 1)),))
        else:
            spec = vp.strip_spec(L=kw["L"], W=STRIP["W"], t=STRIP["t"], n_t=STRIP["n_t"], case="C",
                                 traction_regions=(("mid", (kw["L"]/2-0.0005, -1, -1), (kw["L"]/2+0.0005, 1, 1)),))
        return vp.make_live(spec, vp.material(kw["mu"], kw["terms"], kw["rho"], M=2), control_dt=2e-4,
                            step_material=step_mat, gravity=(0, 0, -9.81))
    singles = [live(kw) for kw in kws]
    batch = rmb.BatchedRubberMaterial([live(kw) for kw in kws], device="cpu", parity_mode=True)
    N, G = batch.num_envs, batch.G
    mism = []
    for k in range(steps):
        tr = np.zeros((N, G, 3))
        R = np.tile(np.eye(3), (N, G, 1, 1))
        A = np.tile(np.eye(3), (N, G, 1, 1))
        active = np.ones((N, G), np.int32)
        for e in range(N):
            if case == "block":
                A[e, 0] = np.eye(3)+(k+1)*(e+1)*np.array([[0.01, 0.005, 0], [0, -0.005, 0.002], [0, 0, -0.005]])
            else:
                tr[e, 0] = [0.0, 1e-5*(k+1)*(e+1), 3e-5*(k+1)]
                R[e, G-1] = Rotation.from_rotvec([0.01*(k+1)*(e+1), 0.005*(k+1), 0.0]).as_matrix()
                if G > 1:
                    tr[e, 1] = [2e-5*(k+1)*(e+1), 0.0, 0.0]
        if inactive_step == k and N > 1:
            active[1, G-1] = 0
        force = np.zeros((N, 3))
        force[:, 0] = 1e-3*np.arange(1, N+1)
        force[:, 2] = -2e-3
        if case == "block":
            batch.advance(affine=A)
        else:
            batch.advance(translation=tr, rotation=R, active=active, traction=force)
        for e, s in enumerate(singles):
            if case == "block":
                s.advance({0: {"affine": A[e, 0]}})
            else:
                s.advance({g: {"translation": tr[e, g], "rotation": R[e, g], "active": bool(active[e, g])}
                           for g in range(G)}, traction={0: force[e]})
            mism += [(k, e, n) for n in vp.compare_env(batch, e, s)]
    return {"case": case, "N": N, "steps": steps, "particles": [s.scene.n for s in singles], "mismatches": mism,
            "invalid": batch.invalid_reason.numpy().tolist(), "cfl_violations": batch.solver.cfl_violations.numpy()
            .tolist(), "visco_q_nonzero": bool(np.any(batch.solver.visco_q.numpy()))}


def job_l0(p, smoke):
    """RUBBER-VISC-07: L0 byte-wise N = 1 / N = 3 heterogeneous, cases A, C, block; L1; refusals."""
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim.batched import rubber_material_batch as rmb
    from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
    mu0, P2 = vp.prony_set("P2")
    _, SLS = vp.prony_set("SLS")
    steps = 2 if smoke else 5
    het = {"A": [dict(L=0.020, mu=mu0, rho=1150.0, terms=P2), dict(L=0.016, mu=1.4*mu0, rho=1050.0, terms=SLS),
                 dict(L=0.024, mu=0.75*mu0, rho=1250.0, terms=((0.1, 0.002), (0.25, 0.02)))],
           "C": [dict(L=0.030, mu=mu0, rho=1150.0, terms=P2), dict(L=0.024, mu=1.4*mu0, rho=1050.0, terms=SLS),
                 dict(L=0.018, mu=0.75*mu0, rho=1250.0, terms=((0.1, 0.002), (0.25, 0.02)))],
           "block": [dict(L=0.004, mu=mu0, rho=1150.0, terms=P2), dict(L=0.004, mu=1.4*mu0, rho=1050.0, terms=SLS),
                     dict(L=0.004, mu=0.75*mu0, rho=1250.0, terms=((0.1, 0.002), (0.25, 0.02)))]}
    rows = []
    for case in ("A", "C", "block"):
        rows.append(_l0_case(case, het[case][:1], steps, smoke=smoke))
        rows.append(_l0_case(case, het[case], steps, inactive_step=1 if case == "C" else None, smoke=smoke))
    # L1: perturb env 1 only
    from scipy.spatial.transform import Rotation  # noqa: F401
    step_mat = vp.material(8e5, (), 1000.0)

    def live(kw):
        spec = vp.strip_spec(L=kw["L"], W=0.004, t=0.004, n_t=4, case="A")
        return vp.make_live(spec, vp.material(kw["mu"], kw["terms"], kw["rho"], M=2), control_dt=2e-4,
                            step_material=step_mat)
    a = rmb.BatchedRubberMaterial([live(kw) for kw in het["A"]], device="cpu", parity_mode=True)
    b = rmb.BatchedRubberMaterial([live(kw) for kw in het["A"]], device="cpu", parity_mode=True)
    for k in range(steps):
        tr = np.zeros((3, 1, 3))
        tr[:, 0, 2] = 3e-5*(k+1)
        a.advance(translation=tr)
        tr[1, 0, 2] += 2e-4
        b.advance(translation=tr)
    l1 = [(e, n) for e in (0, 2) for n, v in a.solver.snapshot(e).items() if not vp.same_bytes(v, b.solver.snapshot(e)[n])]
    l1 += [(e, "visco_diagnostics") for e in (0, 2) if not vp.same_bytes(a.visco_summary.numpy()[e],
                                                                         b.visco_summary.numpy()[e])]
    perturbed_differs = not vp.same_bytes(a.solver.snapshot(1)["visco_q"], b.solver.snapshot(1)["visco_q"])
    # vars()-complete classification
    names = set(a.solver.device_arrays())
    declared = set(rmb.SOLVER_RESET_ARRAYS+rmb.SOLVER_CONSTANT_ARRAYS+rmb.SOLVER_SHARED_ARRAYS
                   + rmb.SOLVER_RECOMPUTED_ARRAYS)
    for table in (rmb.OPTION_RESET_ARRAYS, rmb.OPTION_CONSTANT_ARRAYS):
        for value in table.values():
            declared |= set(value)
    declared |= set(rmb.MODEL_STATE_RESET_ARRAYS["neo_hookean_jbar_prony_v1"]) | set(rmb.VISCO_ENERGY_RESET_ARRAYS)
    refusals = {}
    for label, lives, kw, code in (
            ("mixed_models", [live(het["A"][0]), vp.make_live(vp.strip_spec(L=0.02, W=0.004, t=0.004, n_t=4, case="A"),
                                                              vp.material(mu0, ()), control_dt=2e-4,
                                                              step_material=step_mat, energy="off_v1")], {},
             "UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS"),
            ("different_M", [live(het["A"][0]), vp.make_live(vp.strip_spec(L=0.02, W=0.004, t=0.004, n_t=4, case="A"),
                                                             vp.material(mu0, SLS), control_dt=2e-4,
                                                             step_material=step_mat)], {},
             "UNSUPPORTED_NONUNIFORM_BATCH"),
            ("different_visco_energy", [live(het["A"][0]), vp.make_live(vp.strip_spec(L=0.02, W=0.004, t=0.004, n_t=4,
                                                                                      case="A"),
                                                                        vp.material(mu0, P2), control_dt=2e-4,
                                                                        step_material=step_mat, energy="off_v1")],
             {}, "UNSUPPORTED_NONUNIFORM_BATCH"),
            ("different_options", [live(het["A"][0]), vp.make_live(vp.strip_spec(L=0.02, W=0.004, t=0.004, n_t=4,
                                                                                 case="A"),
                                                                   vp.material(mu0, P2), control_dt=2e-4,
                                                                   step_material=step_mat,
                                                                   options={**vp.OPTIONS,
                                                                            "state_precision": "float32_v1"})],
             {}, "UNSUPPORTED_NONUNIFORM_BATCH")):
        try:
            rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True, **kw)
            refusals[label] = "NOT_REFUSED"
        except UnsupportedBatchConfiguration as exc:
            refusals[label] = exc.code if hasattr(exc, "code") else str(exc)
            refusals[label+"_ok"] = code in str(exc)
    return {"series": {}, "info": {"cases": rows, "l1_others_changed": l1, "l1_perturbed_env_differs":
                                   perturbed_differs, "vars_complete_symmetric_difference": sorted(names ^ declared),
                                   "refusals": refusals}}


def job_elastic_limit(p, smoke):
    """RUBBER-VISC-09: prony = [] (TLMPMRubberVisco, M = 0) vs tlmpm_rubber_v1 (TLMPMRubberCW, same options); M = 2,
    g = 0 vs elastic; single and batched; 5 control steps of case A, case C, block."""
    from scipy.spatial.transform import Rotation
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim.batched import rubber_material_batch as rmb
    from robotic_cutting_sim.rubber_constraint_work import TLMPMRubberCW
    from robotic_cutting_sim.tlmpm_rubber_visco_v1 import TLMPMRubberVisco
    mu0, P2 = vp.prony_set("P2")
    zero = tuple((0.0, tau) for _, tau in P2)
    steps = 2 if smoke else 5
    out = {}
    for case in ("A", "C", "block"):
        if case == "block":
            spec = vp.block_spec()
        elif case == "A":
            spec = vp.strip_spec(L=0.020, W=0.004, t=0.004, n_t=4, case="A",
                                 traction_regions=(("tip", (0.0195+1e-9, -1, -1), (1, 1, 1)),))
        else:
            spec = vp.strip_spec(L=0.030, W=STRIP["W"], t=STRIP["t"], n_t=STRIP["n_t"], case="C",
                                 traction_regions=(("mid", (0.0145, -1, -1), (0.0155, 1, 1)),))

        def inputs(k, G):
            if case == "block":
                return {0: {"affine": np.eye(3)+(k+1)*np.array([[0.02, 0.01, 0], [0, -0.01, 0.005], [0, 0, -0.01]])}}, None
            m = {0: {"translation": np.array([0.0, 1e-5*(k+1), 3e-5*(k+1)]),
                     "rotation": Rotation.from_rotvec([0.01*(k+1), 0, 0]).as_matrix()}}
            if G > 1:
                m[1] = {"translation": np.array([2e-5*(k+1), 0, 0])}
            return m, {0: np.array([1e-3, 0.0, -2e-3])}
        elastic = vp.make_live(spec, vp.material(mu0, ()), energy="off_v1", gravity=(0, 0, -9.81))
        assert type(elastic.solver) is TLMPMRubberCW
        s = elastic.solver
        empty = TLMPMRubberVisco(s.scene, vp.material(mu0, ()), s.dt, substeps=s.substeps, gravity=s.gravity_tuple,
                                 j_domain=s.j_domain, contact_stiffness=s.contact_stiffness,
                                 contact_radius_scale=elastic.recipe.contact_radius_scale, stable_step_guard=1.0,
                                 **vp.OPTIONS)
        gz = vp.make_live(spec, vp.material(mu0, zero), gravity=(0, 0, -9.81))
        mism_empty, mism_zero = [], []
        for k in range(steps):
            motions, traction = inputs(k, s.G)
            elastic.advance(motions, traction=traction)
            gz.advance(motions, traction=traction)
            empty.begin_control_step()
            for g, m in motions.items():
                empty.set_group_motion(g, **m)
            for t, f in (traction or {}).items():
                empty.set_traction(t, f)
            for _ in range(empty.substeps):
                empty.step()
            empty.diagnostics.launch()
            ea, ee, ez = s.device_arrays(), empty.device_arrays(), gz.solver.device_arrays()
            mism_empty += [(k, n) for n, a in ea.items() if not vp.same_bytes(a.numpy(), ee[n].numpy())]
            mism_empty += [(k, "wrench")] if not vp.same_bytes(s.end_control_step(2e-4), empty.end_control_step(2e-4)) else []
            mism_zero += [(k, n) for n, a in ea.items() if not vp.same_bytes(a.numpy(), ez[n].numpy())]
            mism_zero += [(k, "wrench")] if not vp.same_bytes(elastic.last_wrench, gz.last_wrench) else []
            mism_zero += [(k, "diagnostics")] if not vp.same_bytes(s.diagnostics.summary.numpy(),
                                                                   gz.solver.diagnostics.summary.numpy()) else []
        # batched: g = 0 (M = 2) batch vs elastic batch, N = 2 heterogeneous
        lives = lambda terms, energy: [vp.make_live(spec, vp.material(mu, terms), energy=energy,   # noqa: E731
                                                    gravity=(0, 0, -9.81),
                                                    step_material=vp.material(8e5, ()))
                                       for mu in (mu0, 1.5*mu0)]
        be = rmb.BatchedRubberMaterial(lives((), "off_v1"), device="cpu", parity_mode=True)
        bz = rmb.BatchedRubberMaterial(lives(zero, vp.VISCO_ENERGY), device="cpu", parity_mode=True)
        mism_batched = []
        for k in range(steps):
            motions, traction = inputs(k, be.G)
            kw = {}
            if case == "block":
                kw["affine"] = np.stack([[motions[0]["affine"]]]*2)
            else:
                kw["translation"] = np.stack([[motions[g]["translation"] if g in motions else np.zeros(3)
                                               for g in range(be.G)]]*2)
                kw["traction"] = np.stack([traction[0]]*2)
            be.advance(**kw)
            bz.advance(**kw)
            for e in range(2):
                a, b = be.solver.snapshot(e), bz.solver.snapshot(e)
                mism_batched += [(k, e, n) for n in a if not vp.same_bytes(a[n], b[n])]
            mism_batched += [(k, "summary")] if not vp.same_bytes(be.summary.numpy(), bz.summary.numpy()) else []
            mism_batched += [(k, "wrench")] if not vp.same_bytes(be.wrench.numpy(), bz.wrench.numpy()) else []
        out[case] = {"empty_prony_vs_tlmpm_rubber_v1": mism_empty, "zero_g_vs_elastic_single": mism_zero,
                     "zero_g_vs_elastic_batched": mism_batched, "particles": s.n,
                     "invalid": [int(elastic.invalid_reason), int(gz.invalid_reason)]}
    return {"series": {}, "info": out}


def job_tangent(p, smoke):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import visco_tangent_check as V
    rows, groups, premise, Ns = V.run(per_group=20 if smoke else 400, n_dirs=512, seed=2311)
    ratios = [r["ratio"] for r in rows]
    return {"series": {"ratio": ratios}, "info": {"samples": len(rows), "max_ratio": max(ratios),
                                                  "median_ratio": float(np.median(ratios)), "groups": groups,
                                                  "premise_min_eigenvalue_pa": min(premise),
                                                  "worst": sorted(rows, key=lambda r: -r["ratio"])[:3]}}


JOBS = {"block_step": job_block_step, "block_cycle": job_block_cycle, "strip_cycle": job_strip_cycle,
        "creep": job_creep, "cantilever": job_cantilever, "reset_memory": job_reset_memory, "l0": job_l0,
        "elastic_limit": job_elastic_limit, "tangent": job_tangent}


def run_job(item):
    job_id, kind, params, raw_dir, smoke = item
    _setup()
    start = time.time()
    try:
        res = JOBS[kind](params, smoke)
        status = "OK"
        err = None
    except Exception:                                   # noqa: BLE001  (preserved as a failed job)
        res, status, err = {"series": {}, "info": {}}, "ERROR", traceback.format_exc()
    wall = time.time()-start
    series = res.get("series", {})
    arrays = {k: np.asarray(v) for k, v in series.items() if k != "cols"}
    if arrays:
        np.savez_compressed(Path(raw_dir)/f"{job_id}.npz", **arrays)
    meta = {"job": job_id, "kind": kind, "params": params, "status": status, "error": err, "wall_s": wall,
            "info": res.get("info", {}), "cols": series.get("cols")}
    (Path(raw_dir)/f"{job_id}.json").write_text(json.dumps(meta, indent=1, default=_jsonable))
    return meta


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return str(v)


def job_list(cfg, smoke):
    t = cfg["tests"]
    dtc = t["RUBBER-VISC-01"]["setup"]["control_dt_s"]
    jobs = []
    for s in ("SLS", "P2"):
        hold = 0.06 if s == "SLS" else 0.3
        for path in ("uniaxial_isochoric", "simple_shear"):
            jobs.append((f"v01_{s}_{path}", "block_step", dict(set=s, path=path, factor=1, control_dt=dtc,
                                                               hold_until=hold)))
    for f in t["RUBBER-VISC-04"]["setup"]["substep_factors"][1:]:
        jobs.append((f"v04a_P2_uniaxial_x{f}", "block_step", dict(set="P2", path="uniaxial_isochoric", factor=f,
                                                                    control_dt=dtc, hold_until=0.3)))
    s3 = t["RUBBER-VISC-03"]["setup"]
    for f in s3["block_finite_amplitude"]["rates_hz"]:
        jobs.append((f"v03e_block_{f:g}hz", "block_cycle", dict(rate=f, gamma0=0.15, cycles=3, control_dt=dtc)))
    for f in sorted(s3["fit_rates_hz"]+s3["heldout_rates_hz"]):
        n = max(2, math.ceil(6*0.05*f)+1)
        jobs.append((f"v03_strip_{f:g}hz", "strip_cycle", dict(rate=f, cycles=n, eps0=s3["eps0"], control_dt=dtc)))
    n16 = max(2, math.ceil(6*0.05*16)+1)
    jobs.append(("v05_strip_16hz_x2", "strip_cycle", dict(rate=16.0, cycles=n16, eps0=s3["eps0"], control_dt=dtc,
                                                         factor=2)))
    jobs.append(("v05_strip_16hz_g0", "strip_cycle", dict(rate=16.0, cycles=n16, eps0=s3["eps0"], control_dt=dtc,
                                                         g_zero=True)))
    jobs.append(("v02_creep", "creep", dict(control_dt=t["RUBBER-VISC-02"]["setup"]["control_dt_s"])))
    for f in t["RUBBER-VISC-04"]["setup"]["substep_factors"]:
        jobs.append((f"v04b_cantilever_x{f}", "cantilever", dict(factor=f)))
    jobs += [("v06_reset_memory", "reset_memory", {}), ("v07_l0", "l0", {}), ("v09_elastic_limit", "elastic_limit", {}),
             ("v08b_tangent", "tangent", {})]
    # longest first (better packing)
    order = {"strip_cycle": 0, "cantilever": 1, "creep": 2, "l0": 3, "reset_memory": 3, "block_cycle": 4,
             "block_step": 5, "elastic_limit": 5, "tangent": 6}
    weight = lambda j: (order[j[1]], -j[2].get("factor", 1), j[2].get("rate", 0))   # noqa: E731
    return sorted(jobs, key=weight)


# ============================================================================ analysis

def _load(raw, job):
    meta = json.loads((raw/f"{job}.json").read_text())
    path = raw/f"{job}.npz"
    data = dict(np.load(path)) if path.exists() else {}
    return meta, data


def _table(meta, data):
    return {c: data["data"][:, i] for i, c in enumerate(meta["cols"])}


def _books(s):
    BN = s["kinetic"]+s["psi_eq"]+s["gravity"]+s["branch_work"]-s["work_ext"]
    SN = float(np.max(s["kinetic"]+s["psi_eq"])+np.max(np.abs(s["branch_work"])))
    return BN, SN


def analyse(cfg, raw, metas, smoke):
    t = cfg["tests"]
    res = {}
    status = {m["job"]: m["status"] for m in metas}
    errors = {m["job"]: m["error"] for m in metas if m["status"] != "OK"}
    validity = {}
    cfl_total = 0
    for m in metas:
        info = m["info"]
        if "invalid_reason" in info and isinstance(info["invalid_reason"], int):
            validity[m["job"]] = info["invalid_reason"]
            cfl_total += int(info.get("cfl_violations", 0))
    from robotic_cutting_sim import rubber_visco_reference as ref
    from robotic_cutting_sim import rubber_visco_protocol as vp
    mu0, P2 = vp.prony_set("P2")
    kappa = 20*mu0

    def valid(jobs):
        return all(status.get(j) == "OK" and validity.get(j, 0) == 0 for j in jobs)

    # ---------------- -01
    thr = t["RUBBER-VISC-01"]["thresholds"]["rel_max"]
    rows = []
    for s in ("SLS", "P2"):
        for path in ("uniaxial_isochoric", "simple_shear"):
            job = f"v01_{s}_{path}"
            meta, d = _load(raw, job)
            if meta["status"] != "OK":
                rows.append({"job": job, "status": "ERROR"})
                continue
            err = d["rel_err"]
            hold = err[1:] if len(err) > 1 else err
            rows.append({"job": job, "max_rel_err_hold": float(np.max(hold)), "ramp_step_rel_err": float(err[0]),
                         "closed_form_step_vs_hereditary_max": float(np.nanmax(d["closed_form_step_vs_hereditary"]))
                         if np.isfinite(d["closed_form_step_vs_hereditary"]).any() else None,
                         "steps": len(err), "pass": bool(np.max(hold) <= thr)})
    jobs = [r["job"] for r in rows]
    res["RUBBER-VISC-01"] = {"rows": rows, "threshold": thr,
                             "verdict": "PASS" if all(r.get("pass") for r in rows) and valid(jobs) else "FAIL"}
    # ---------------- -04a
    from scipy.optimize import least_squares
    fits = {}
    for f in t["RUBBER-VISC-04"]["setup"]["substep_factors"]:
        job = "v01_P2_uniaxial_isochoric" if f == 1 else f"v04a_P2_uniaxial_x{f}"
        meta, d = _load(raw, job)
        if meta["status"] != "OK" or len(d.get("t_stress", [])) < 10:
            fits[f] = None
            continue
        ts, y = d["t_stress"][1:], d["P11_mean"][1:]
        tr = cfg["tests"]["RUBBER-VISC-01"]["setup"]["control_dt_s"]
        x0 = [y[-1], (y[0]-y[-1])*0.6, (y[0]-y[-1])*0.4, math.log(0.005), math.log(0.05)]

        def model(x, tt):
            return x[0]+x[1]*np.exp(-(tt-tr)/math.exp(x[3]))+x[2]*np.exp(-(tt-tr)/math.exp(x[4]))
        sol = least_squares(lambda x: (model(x, ts)-y)/np.abs(y).max(), x0, xtol=1e-15, ftol=1e-15, gtol=1e-15,
                            max_nfev=20000)
        taus = sorted([math.exp(sol.x[3]), math.exp(sol.x[4])])
        fits[f] = {"tau1_s": taus[0], "tau2_s": taus[1], "residual_rms_rel": float(np.sqrt(np.mean(sol.fun**2))),
                   "A_inf": float(sol.x[0])}
    thr4 = t["RUBBER-VISC-04"]["thresholds"]
    declared = [tau for _, tau in sorted(P2, key=lambda z: z[1])]
    ok_a = all(fits.get(f) for f in (1, 2, 4))
    checks_a = []
    if ok_a:
        for f in (1, 2, 4):
            for i, key in enumerate(("tau1_s", "tau2_s")):
                rel_decl = fits[f][key]/declared[i]-1
                checks_a.append({"factor": f, "term": i+1, "rel_vs_declared": rel_decl,
                                 "rel_vs_factor1": fits[f][key]/fits[1][key]-1 if f != 1 else 0.0})
        ok_a = all(abs(c["rel_vs_declared"]) <= thr4["tau_dt_rel"] and abs(c["rel_vs_factor1"]) <= thr4["tau_dt_rel"]
                   for c in checks_a)
    # -04b
    root = ref.beam_root(CANT["L"], CANT["W"], CANT["t"], 1150.0, mu0, kappa, P2)
    dec = {}
    for f in t["RUBBER-VISC-04"]["setup"]["substep_factors"]:
        job = f"v04b_cantilever_x{f}"
        meta, d = _load(raw, job)
        if meta["status"] != "OK" or "data" not in d or len(d["data"]) < 50:
            dec[f] = None
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
        dec[f] = {"zeta_1_s": float(best.x[4]), "f_d_hz": float(best.x[5]/(2*math.pi)),
                  "real_exponential_rate_1_s": float(best.x[2]), "amplitude_m": float(best.x[3]),
                  "residual_rms_rel": float(np.sqrt(np.mean(best.fun**2))), "fit_start_s": t0, "samples": int(sel.sum())}
    ok_b = all(dec.get(f) for f in (1, 2, 4))
    checks_b = {}
    if ok_b:
        checks_b = {"zeta_rel_x2": dec[2]["zeta_1_s"]/dec[1]["zeta_1_s"]-1,
                    "zeta_rel_x4": dec[4]["zeta_1_s"]/dec[1]["zeta_1_s"]-1,
                    "f_rel_x2": dec[2]["f_d_hz"]/dec[1]["f_d_hz"]-1, "f_rel_x4": dec[4]["f_d_hz"]/dec[1]["f_d_hz"]-1,
                    "f_vs_root": dec[1]["f_d_hz"]/root["f_d_hz"]-1, "zeta_vs_root": dec[1]["zeta_1_s"]/root["zeta_1_s"]-1}
        ok_b = (abs(checks_b["zeta_rel_x2"]) <= thr4["zeta_dt_rel"] and abs(checks_b["zeta_rel_x4"]) <= thr4["zeta_dt_rel"]
                and abs(checks_b["f_rel_x2"]) <= thr4["f_dt_rel"] and abs(checks_b["f_rel_x4"]) <= thr4["f_dt_rel"]
                and abs(checks_b["f_vs_root"]) <= thr4["f_ref_rel"] and abs(checks_b["zeta_vs_root"]) <= thr4["zeta_ref_rel"])
    jobs4 = ["v01_P2_uniaxial_isochoric", "v04a_P2_uniaxial_x2", "v04a_P2_uniaxial_x4"]+[
        f"v04b_cantilever_x{f}" for f in (1, 2, 4)]
    res["RUBBER-VISC-04"] = {"a": {"fits": fits, "checks": checks_a, "pass": bool(ok_a)},
                             "b": {"fits": dec, "root": root, "checks": checks_b, "pass": bool(ok_b)},
                             "thresholds": thr4,
                             "verdict": "PASS" if ok_a and ok_b and valid(jobs4) else "FAIL"}
    # ---------------- -02
    s2 = t["RUBBER-VISC-02"]["setup"]
    thr2 = t["RUBBER-VISC-02"]["thresholds"]
    meta, d = _load(raw, "v02_creep")
    r2 = {"verdict": "FAIL"}
    if meta["status"] == "OK" and "data" in d and len(d["data"]) > 10:
        s = _table(meta, d)
        tk, eps, sig = s["t"], s["eps"], s["sigma_applied"]
        dtc = s2["control_dt_s"]
        J = lambda tt: ref.creep_uniaxial(tt, mu0, kappa, P2)       # noqa: E731
        eps_ref = ref.boltzmann_staircase(tk, tk-dtc, sig, J)
        E0 = 9*kappa*mu0/(3*kappa+mu0)
        Tax = 4*STRIP["L"]/math.sqrt(E0/1150.0)
        Tr, t1, t2 = s2["T_r_s"], s2["t1_s"], s2["t2_s"]
        w1 = (tk >= Tr+10*Tax-1e-12) & (tk <= t1+1e-12)
        w2 = (tk >= t1+Tr+10*Tax-1e-12) & (tk <= t2+1e-12)
        emax = float(np.max(np.abs(eps_ref)))
        dev = np.abs(eps-eps_ref)
        i_star = int(np.argmax(tk >= Tr+10*Tax-1e-12))
        i1 = int(np.argmin(np.abs(tk-t1)))
        i2 = int(np.argmin(np.abs(tk-t2)))
        ratio_sim, ratio_ref = eps[i1]/eps[i_star], eps_ref[i1]/eps_ref[i_star]
        r2 = {"max_eps_ref": emax, "curve_max_dev_rel_window1": float(dev[w1].max()/emax),
              "curve_max_dev_rel_window2": float(dev[w2].max()/emax), "T_ax_s": Tax, "t_star_s": float(tk[i_star]),
              "creep_ratio_sim": float(ratio_sim), "creep_ratio_ref": float(ratio_ref),
              "creep_ratio_rel": float(ratio_sim/ratio_ref-1), "recovery_dev_rel": float(dev[i2]/emax),
              "eps_t1": float(eps[i1]), "eps_ref_t1": float(eps_ref[i1]), "eps_t2": float(eps[i2]),
              "eps_ref_t2": float(eps_ref[i2])}
        ok = (r2["curve_max_dev_rel_window1"] <= thr2["curve_rel"] and r2["curve_max_dev_rel_window2"] <= thr2["curve_rel"]
              and abs(r2["creep_ratio_rel"]) <= thr2["ratio_rel"] and r2["recovery_dev_rel"] <= thr2["curve_rel"])
        r2["verdict"] = "PASS" if ok and valid(["v02_creep"]) else "FAIL"
    r2["thresholds"] = thr2
    res["RUBBER-VISC-02"] = r2
    # ---------------- -03 strip per rate + identification
    s3 = t["RUBBER-VISC-03"]["setup"]
    thr3 = t["RUBBER-VISC-03"]["thresholds"]
    V = STRIP["L"]*STRIP["W"]*STRIP["t"]
    rates = sorted(s3["fit_rates_hz"]+s3["heldout_rates_hz"])
    per = {}
    strip_jobs = [(f"v03_strip_{f:g}hz", f) for f in rates]+[("v05_strip_16hz_x2", 16.0),
                                                              ("v05_strip_16hz_g0", 16.0)]
    for job, f in strip_jobs:
        if True:
            meta, d = _load(raw, job)
            if meta["status"] != "OK" or "data" not in d:
                per[job] = None
                continue
            s = _table(meta, d)
            dtc = s3["control_dt_s"]
            w = 2*math.pi*f
            T = 1.0/f
            tk = s["t"]
            win = tk > tk[-1]-T+1e-12
            n = int(win.sum())
            sigma = -s["wrench1_x"]/(STRIP["W"]*STRIP["t"])
            E = ref.fourier_modulus(sigma[win], s["eps"][win], tk[win]-dtc/2, tk[win], w, dtc)
            Eref = ref.uniaxial_complex_modulus(w, mu0, kappa, P2)
            last, first = len(tk)-1, len(tk)-1-n
            dW = {k: float(s[k][last]-s[k][first]) for k in ("work_ext", "branch_work", "dissipation", "kinetic",
                                                             "psi_eq", "psi_visc_stored")}
            loss_ref = math.pi*s3["eps0"]**2*Eref.imag*V
            BN, SN = _books(s)
            row = {"rate_hz": f, "samples_in_cycle": n, "samples_per_period": T/dtc, "E1_pa": E.real,
                   "E2_pa": E.imag, "E1_ref_pa": Eref.real, "E2_ref_pa": Eref.imag,
                   "E1_rel": E.real/Eref.real-1, "E2_rel": E.imag/Eref.imag-1,
                   "loss_cycle_j": dW["work_ext"], "loss_ref_j": loss_ref, "loss_rel": dW["work_ext"]/loss_ref-1,
                   "cycle": dW, "cycle_books_rel": dW["work_ext"]/dW["branch_work"]-1 if dW["branch_work"] else None,
                   "cycle_dissipation_rel": dW["dissipation"]/dW["branch_work"]-1 if dW["branch_work"] else None,
                   "books_max_abs_rel": float(np.max(np.abs(BN))/SN), "S_N_j": SN,
                   "books_trend_rel_per_s": float(np.polyfit(tk, BN/SN, 1)[0]),
                   "d_negative": float(np.max(s["dissipation_negative"])),
                   "cfl_visco_max": float(np.max(s["cfl_visco_max"])), "steps": len(tk),
                   "E_alt_sigma_mean_removed": None}
            Ealt = ref.fourier_modulus(sigma[win]-sigma[win].mean(), s["eps"][win], tk[win]-dtc/2, tk[win], w, dtc)
            row["E_alt_sigma_mean_removed"] = {"E1_rel": Ealt.real/Eref.real-1, "E2_rel": Ealt.imag/Eref.imag-1}
            row["pass_a"] = bool(abs(row["E1_rel"]) <= thr3["e1_rel"] and abs(row["E2_rel"]) <= thr3["e2_rel"]
                                 and abs(row["loss_rel"]) <= thr3["loss_rel"])
            per[job] = row
    rows3 = [per.get(f"v03_strip_{f:g}hz") for f in rates]
    ok3a = all(r and r["pass_a"] for r in rows3)
    # identification (b), held-out (c), single-rate control (d)
    ident = None
    if all(r for r in rows3):
        sim = {r["rate_hz"]: complex(r["E1_pa"], r["E2_pa"]) for r in rows3}

        def Emodel(x, f, M=2):
            terms = [(x[2*i], math.exp(x[2*i+1])) for i in range(M)]
            return ref.uniaxial_complex_modulus(2*math.pi*f, mu0, kappa, terms)

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
        held = []
        for f in s3["heldout_rates_hz"]:
            Em, Es = Emodel(xf, f), sim[f]
            held.append({"rate_hz": f, "E1_pred_rel": Em.real/Es.real-1, "E2_pred_rel": Em.imag/Es.imag-1})
        ok_c3 = all(abs(h["E1_pred_rel"]) <= thr3["e1_heldout"] and abs(h["E2_pred_rel"]) <= thr3["e2_heldout"]
                    for h in held)
        one = least_squares(lambda x: resid(x, [4.0], M=1), [0.3, math.log(0.01)], xtol=1e-15, ftol=1e-15)
        single = {"g": one.x[0], "tau_s": math.exp(one.x[1]),
                  "heldout": [{"rate_hz": f, "E1_pred_rel": Emodel(one.x, f, 1).real/sim[f].real-1,
                               "E2_pred_rel": Emodel(one.x, f, 1).imag/sim[f].imag-1} for f in s3["heldout_rates_hz"]]}
        ident = {"b": {"terms": b_checks, "pass": bool(ok_b3), "cost": float(fit.cost)},
                 "c": {"heldout": held, "pass": bool(ok_c3)}, "d_single_rate_control_reported": single}
    # block finite amplitude (e)
    blk = []
    for f in s3["block_finite_amplitude"]["rates_hz"]:
        job = f"v03e_block_{f:g}hz"
        meta, d = _load(raw, job)
        if meta["status"] != "OK" or "abs_err" not in d:
            blk.append({"job": job, "status": "ERROR"})
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
                             "verdict": "PASS" if ok3 else "FAIL"}
    # ---------------- -05
    thr5 = t["RUBBER-VISC-05"]["thresholds"]
    books = {}
    for job in [f"v03_strip_{f:g}hz" for f in rates]+["v05_strip_16hz_x2", "v02_creep"]+[
            f"v04b_cantilever_x{f}" for f in (1, 2, 4)]:
        meta, d = _load(raw, job)
        if meta["status"] != "OK" or "data" not in d:
            books[job] = None
            continue
        s = _table(meta, d)
        BN, SN = _books(s)
        books[job] = {"max_abs_rel": float(np.max(np.abs(BN))/SN), "max_abs_j": float(np.max(np.abs(BN))), "S_N_j": SN,
                      "d_negative": float(np.max(s["dissipation_negative"])),
                      "end_rel": float(BN[-1]/SN)}
    ok5a = all(b and b["max_abs_rel"] <= thr5["books_rel"] for b in books.values())
    b1, b2 = books.get("v03_strip_16hz"), books.get("v05_strip_16hz_x2")
    cons = None
    if b1 and b2:
        cons = {"ratio": b2["max_abs_j"]/b1["max_abs_j"], "factor2_rel": b2["max_abs_rel"]}
        ok5b = cons["ratio"] <= thr5["ratio_max"] or b2["max_abs_rel"] <= thr5["floor_rel"]
    else:
        ok5b = False
    cyc = [{"rate_hz": r["rate_hz"], "books_rel": r["cycle_books_rel"], "dissipation_rel": r["cycle_dissipation_rel"]}
           for r in rows3 if r]
    ok5c = len(cyc) == len(rates) and all(c["books_rel"] is not None and abs(c["books_rel"]) <= thr5["cycle_books_rel"]
                                          for c in cyc)
    ok5d = (len(cyc) == len(rates) and all(c["dissipation_rel"] is not None
                                           and abs(c["dissipation_rel"]) <= thr5["dissipation_rel"] for c in cyc)
            and all(b and b["d_negative"] == 0 for b in books.values()))
    g0 = per.get("v05_strip_16hz_g0")
    e5 = {"books_max_abs_rel": g0["books_max_abs_rel"], "trend_rel_per_s": g0["books_trend_rel_per_s"]} if g0 else None
    jobs5 = [j for j in books]
    res["RUBBER-VISC-05"] = {"a_books": books, "b_consistency": cons, "c_cycle_books": cyc,
                             "d_dissipation": {"cycles": cyc, "negative_increments": {j: b["d_negative"] if b else None
                                                                                       for j, b in books.items()}},
                             "e_elastic_control_reported": e5, "thresholds": thr5,
                             "pass": {"a": bool(ok5a), "b": bool(ok5b), "c": bool(ok5c), "d": bool(ok5d)},
                             "verdict": "PASS" if ok5a and ok5b and ok5c and ok5d and valid(jobs5) else "FAIL"}
    # ---------------- -06
    meta, _ = _load(raw, "v06_reset_memory")
    i6 = meta["info"]
    ok6 = (meta["status"] == "OK" and i6.get("visco_q_env1_nonzero_before_reset")
           and not i6["env1_equals_fresh_batch_after_reset"] and not i6["envs_0_2_unchanged_by_reset"]
           and not i6["env1_after_reset_equals_fresh_batch_20_steps"]
           and not i6["env1_after_reset_equals_fresh_single_20_steps"] and i6["params64_resampled_equals_host_expm1"]
           and not i6["resampled_reset_equals_single_built_with_it"] and i6["plain_reset_restores_packed_params64"]
           and i6["single_reset_clears"] and not i6["single_two_episodes_byte_identical"]
           and not any(i6["invalid_reason"]) and not any(i6["cfl_violations"]))
    cfl_total += int(sum(i6.get("cfl_violations", [0])))
    res["RUBBER-VISC-06"] = {"info": i6, "verdict": "PASS" if ok6 else "FAIL"}
    # ---------------- -07
    meta, _ = _load(raw, "v07_l0")
    i7 = meta["info"]
    ok7 = (meta["status"] == "OK" and all(not c["mismatches"] and c["visco_q_nonzero"] and not any(c["invalid"])
                                          for c in i7["cases"])
           and not i7["l1_others_changed"] and i7["l1_perturbed_env_differs"]
           and not i7["vars_complete_symmetric_difference"]
           and all(i7["refusals"].get(k+"_ok") for k in ("mixed_models", "different_M", "different_visco_energy",
                                                         "different_options")))
    if meta["status"] == "OK":
        cfl_total += int(sum(sum(c["cfl_violations"]) for c in i7["cases"]))
    res["RUBBER-VISC-07"] = {"info": i7, "verdict": "PASS" if ok7 else "FAIL"}
    # ---------------- -09
    meta, _ = _load(raw, "v09_elastic_limit")
    i9 = meta["info"]
    ok9 = meta["status"] == "OK" and all(not v["empty_prony_vs_tlmpm_rubber_v1"] and not v["zero_g_vs_elastic_single"]
                                         and not v["zero_g_vs_elastic_batched"] and not any(v["invalid"])
                                         for v in i9.values())
    res["RUBBER-VISC-09"] = {"info": i9, "verdict": "PASS" if ok9 else "FAIL"}
    # ---------------- -08
    from robotic_cutting_sim import rubber_material as rm
    recipe = rm.load_recipe(rm.ROOT/"configs"/"materials"/"rubber_generic_visco_v1.json")
    rule = recipe.step_rule(1e-3, 2e-4)
    mu0_max = recipe.shear_modulus_eq.hi/recipe.g_inf
    at_mu0 = rm.step_rule(1e-3, 2e-4, mu0_max=mu0_max, kappa_max=recipe.bulk_ratio.value*mu0_max,
                          density_min=recipe.density.lo)
    ok8a = rule["dt_s"] == at_mu0["dt_s"] and rule["mu0_max_pa"] == mu0_max
    meta, d = _load(raw, "v08b_tangent")
    i8 = meta["info"]
    thr8 = t["RUBBER-VISC-08"]["thresholds"]["ratio_max"]
    ok8b = meta["status"] == "OK" and i8["samples"] >= 2000 and i8["max_ratio"] <= thr8
    ok8c = cfl_total == 0
    res["RUBBER-VISC-08"] = {"a_step_rule": {"dt_s": rule["dt_s"], "dt_at_mu0_s": at_mu0["dt_s"], "pass": bool(ok8a)},
                             "b_tangent_check": {**i8, "pass": bool(ok8b)},
                             "c_validity": {"guard_violations_total": cfl_total, "pass": bool(ok8c)},
                             "verdict": "PASS" if ok8a and ok8b and ok8c else "FAIL"}
    res["RUBBER-VISC-10"] = {"verdict": "NOT_RUN_HERE", "note": "GPU cost rows: scripts/rubber/p2_3/bench_visc_cost.py"}
    res["_validity"] = {"invalid_jobs": {j: v for j, v in validity.items() if v}, "errors": errors}
    if smoke:
        for k, v in res.items():
            if isinstance(v, dict) and "verdict" in v:
                v["verdict"] = f"SMOKE ({v['verdict']})"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="job ids (diagnostic subset; never a formal result)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    cfg, sha = load_protocol()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    raw = out/"raw"
    raw.mkdir(parents=True)
    jobs = job_list(cfg, args.smoke)
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
    start = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    metas = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, (jid, kind, params, str(raw), args.smoke)): jid for jid, kind, params in jobs}
        for fut in as_completed(futs):
            meta = fut.result()
            metas.append(meta)
            print(f"[{time.time()-start:8.1f} s] {meta['job']}: {meta['status']} ({meta['wall_s']:.1f} s)", flush=True)
    results = analyse(cfg, raw, metas, args.smoke) if not args.only else {"note": "subset run: no verdicts"}
    import warp as wp
    import scipy
    summary = {"schema": "rcs_rubber_visc_run_v1", "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha,
               "label": "SMOKE (runner check; not a result)" if args.smoke else (
                   "SUBSET (diagnostic)" if args.only else "FORMAL"),
               "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED (illustrative Prony sets, not fitted to real data)",
               "commit": commit, "dirty_tree": bool(dirty), "command": " ".join(sys.argv), "workers": args.workers,
               "device": "Warp CPU (CUDA_VISIBLE_DEVICES='')", "started": started,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "wall_s": time.time()-start,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "warp": wp.config.version},
               "jobs": sorted([{k: m[k] for k in ("job", "kind", "status", "wall_s")}
                               | {"particles": m["info"].get("particles"), "substeps": m["info"].get("substeps"),
                                  "dt": m["info"].get("dt")} for m in metas], key=lambda r: r["job"]),
               "results": results}
    (out/"summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    for k, v in results.items():
        if isinstance(v, dict) and "verdict" in v:
            print(k, v["verdict"])


if __name__ == "__main__":
    main()
