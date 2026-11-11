"""Formal runner of ``configs/rubber/rubber_contact_envelope_v2.json`` (ENV2 for the P2-6 fallback law after K2-1), CPU.

Law ``grid_node_penalty_coulomb_v2`` (every ContactSpec built in a job is a v2 spec with the protocol's parameters;
checked per job). Incline scenes INC2 / STRIP at n_t 4 / 5 / 6, mu 0.3 / 0.4 (the incline job and the per-level
evaluation of ``run_contact_envelope_v1.py``, bound to this protocol); SPIN (K2-1 a, scored): torsional spin-down of a
16 x 16 x 4 mm block against the rigid Coulomb decay; DRAG (K2-1 b, reported): lateral drag of a case-A strip.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_6/run_contact_envelope_v2.py \
        --out artifacts/rl_ready/p2_6/<dir> [--jobs N] [--smoke]
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
sys.path.insert(0, str(ROOT/"scripts"/"rubber"/"p2_4"))
sys.path.insert(0, str(ROOT/"scripts"/"rubber"/"p2_6"))
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_envelope_v2.json"
PROTOCOL_SHA = "1f6163997bdd072e961b2724de951d81f2db29f889d286ce1d1e3c4ba63f1e18"
LAW = "grid_node_penalty_coulomb_v2"
G = 9.81
BUILT = []


def _cfg():
    return json.loads(PROTOCOL.read_text())


def _setup(smoke=False):
    import run_contact_v1 as R
    import run_contact_envelope_v1 as E
    from robotic_cutting_sim import rubber_contact as rct
    R.PROTOCOL = PROTOCOL
    R.SMOKE = smoke
    E.PROTOCOL = PROTOCOL
    E.LAW = LAW
    lp = _cfg()["common"]["law_parameters"]
    base = rct.ContactSpec
    if getattr(base, "_env2v2_runner", False):
        return R, E

    class SpecV2(base):
        _env2v2_runner = True

        def __init__(self, *args, **kw):
            if kw.get("law", LAW) != LAW:
                raise ValueError("run_contact_envelope_v2 builds only the v2 law")
            base.__init__(self, *args, **{**kw, "law": LAW, "normal_stiffness": lp["normal_stiffness_kappa_n"],
                                          "tangential_stiffness": lp["tangential_stiffness_kappa_t"],
                                          "normal_damping_ratio": lp["normal_damping_ratio_zeta_n"],
                                          "damping_ratio": lp["tangential_damping_ratio_zeta_t"]})
            BUILT.append(self.law)
    rct.ContactSpec = SpecV2
    return R, E


def job_spin(name, n_t, mu):
    import run_contact_v1 as R
    from robotic_cutting_sim import rubber_contact as rct
    sc = _cfg()["scenes"]["SPIN"]
    L, W, H = sc["size_m"]
    spec = R._spec(length=L, width=W, thickness=H, n_t=n_t, case="free")
    live = R._live(spec, R._material("elastic", friction_table=mu), sc["control_dt_s"], (0, 0, 0),
                   rct.ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),)), cfl=_cfg()["common"]["cfl"])
    cdt, Tr = sc["control_dt_s"], sc["ramp_s"]
    st = R.Stepper(live, lambda t: (np.array([0.0, 0.0, -G])*R._ramp(t, Tr), None))
    s = live.solver
    settle = int(round(sc["settle_s"]/cdt))
    spin = 2 if R.SMOKE else int(round(sc["spin_s"]/cdt))
    t0 = time.perf_counter()
    for _ in range(2 if R.SMOKE else settle):
        st.advance()
    x = s.x64.numpy()
    c = x.mean(0)
    v = s.v.numpy().astype(np.float64)
    w0 = sc["omega0_rad_s"]
    v[:, 0] += -w0*(x[:, 1]-c[1])
    v[:, 1] += w0*(x[:, 0]-c[0])
    s.v.assign(v.astype(np.float32))

    def omega():
        xx = s.x64.numpy()
        r = xx-xx.mean(0)
        vv = s.v.numpy().astype(np.float64)
        return float(np.sum(r[:, 0]*vv[:, 1]-r[:, 1]*vv[:, 0])/np.sum(r[:, 0]**2+r[:, 1]**2))
    rows = [{"k": 0, "omega": omega()}]
    for k in range(1, spin+1):
        st.advance()
        pw = live.last_contact["plane_wrench"][0]
        rows.append({"k": k, "omega": omega(), "torque_z": float(pw[5]), "N": float(-pw[2]),
                     "invalid": int(live.invalid_reason)})
    M = float(s.mass_value)*s.n
    a = L
    rbar = a*(math.sqrt(2)+math.log(1+math.sqrt(2)))/6
    Iz = M*(a*a+W*W)/12
    Tc = mu*M*G*rbar
    return {"name": name, "scene": "SPIN", "n_t": n_t, "mu": mu, "omega0": w0, "omega_start": rows[0]["omega"],
            "rows": rows, "M_kg": M, "rbar_m": rbar, "Iz": Iz, "T_coulomb": Tc, "decel_ref": Tc/Iz,
            "particles": int(s.n), "dt_s": live.dt, "substeps": live.substeps, "wall_s": time.perf_counter()-t0,
            "receipt_contact": s.contact.receipt()}


def job_drag(name):
    import run_contact_v1 as R
    from robotic_cutting_sim import rubber_contact as rct
    sc = _cfg()["scenes"]["DRAG"]
    spec = R._spec(length=0.06, width=0.008, thickness=0.004, n_t=sc["n_t"], case="A")
    live = R._live(spec, R._material("elastic", friction_table=sc["mu"]), sc["control_dt_s"], (0, 0, 0),
                   rct.ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),)), cfl=_cfg()["common"]["cfl"])
    cdt, Tr = sc["control_dt_s"], sc["ramp_s"]
    st = R.Stepper(live, lambda t: (np.array([0.0, 0.0, -G])*R._ramp(t, Tr), None))
    t0 = time.perf_counter()
    for _ in range(2 if R.SMOKE else int(round(sc["settle_s"]/cdt))):
        st.advance()
    m0 = live.markers()
    rows = []
    for k in range(1, (2 if R.SMOKE else int(round(sc["drag_s"]/cdt)))+1):
        y = sc["speed_m_s"]*k*cdt
        st.advance(motions={0: {"translation": np.array([0.0, y, 0.0])}})
        m = live.markers()
        rows.append({"k": k, "jaw_y": y, "mid_dy": float(m[4][1]-m0[4][1]), "tip_dy": float(m[8][1]-m0[8][1]),
                     "tip_dx": float(m[8][0]-m0[8][0]), "invalid": int(live.invalid_reason)})
    return {"name": name, "scene": "DRAG", "rows": rows, "particles": int(live.scene.n), "dt_s": live.dt,
            "wall_s": time.perf_counter()-t0}


def jobs(cfg):
    import run_contact_envelope_v1 as E
    out = {n: ("inc", args) for n, args in E.jobs(cfg).items()}
    sc = cfg["scenes"]["SPIN"]
    for n_t in sc["n_t_set"]:
        for mu in sc["mu_set"]:
            out[f"SPIN_nt{n_t}_mu{mu}"] = ("spin", (n_t, mu))
    out["DRAG_nt4_mu0.4"] = ("drag", ())
    return out


def _run(item):
    name, (kind, args), smoke = item
    _, E = _setup(smoke)
    BUILT.clear()
    try:
        res = {"inc": E.job, "spin": job_spin, "drag": job_drag}[kind](name, *args)
        res["contact_laws_built"] = sorted(set(BUILT))
        if res["contact_laws_built"] != [LAW]:
            raise AssertionError(f"job built laws {res['contact_laws_built']}")
        return name, res
    except Exception as exc:                                    # recorded, never hidden
        import traceback
        return name, {"name": name, "error": repr(exc), "traceback": traceback.format_exc()}


def evaluate(cfg, R_):
    _, E = _setup(False)
    inc = {n: r for n, r in R_.items() if r.get("scene") == "INC3" or ("error" in r and not n.startswith(("SPIN",
                                                                                                         "DRAG")))}
    base = E.evaluate(cfg, inc)
    th = cfg["tests"]["ENV-04"]["thresholds"]["spin_rel"]
    spins = {}
    for n, r in R_.items():
        if not n.startswith("SPIN"):
            continue
        if "error" in r:
            spins[n] = {"pass": False, "error": r["error"]}
            continue
        rows = r["rows"]
        w0 = r["omega_start"]
        res = {"n_t": r["n_t"], "mu": r["mu"], "omega_start": w0}
        ok = True
        for t_ms in (5, 10):
            if t_ms >= len(rows):
                ok = False
                continue
            sim = w0-rows[t_ms]["omega"]
            ref = r["decel_ref"]*t_ms*1e-3
            res[f"decel_ratio_{t_ms}ms"] = sim/ref
            ok &= abs(sim/ref-1) <= th
        res["min_omega"] = min(q["omega"] for q in rows[1:])
        res["no_reversal"] = res["min_omega"] > 0
        res["valid"] = all(q["invalid"] == 0 for q in rows[1:])
        res["mean_torque_over_Tc"] = float(abs(np.mean([q["torque_z"] for q in rows[1:]]))/r["T_coulomb"])
        res["mean_N_over_Mg"] = float(np.mean([q["N"] for q in rows[1:]])/(r["M_kg"]*G))
        res["pass"] = bool(ok and res["no_reversal"] and res["valid"])
        spins[n] = res
    levels = base["levels"]
    for k, v in levels.items():
        sp = spins.get(f"SPIN_nt{v['n_t']}_mu{v['mu']}")
        v["spin_pass"] = bool(sp and sp["pass"])
        v["sliding_pass_with_spin"] = v["sliding_pass"] and v["spin_pass"]
    nts = sorted({v["n_t"] for v in levels.values()})
    mus = sorted({v["mu"] for v in levels.values()})
    envelope = {}
    for n_t in nts:
        res = {}
        for kind, key in (("sliding", "sliding_pass_with_spin"), ("static", "static_pass")):
            best = None
            for mu in mus:
                lv = [v for v in levels.values() if v["n_t"] == n_t and v["mu"] == mu]
                if lv and all(v[key] for v in lv):
                    best = mu
                else:
                    break
            res[f"{kind}_mu_max"] = best
        res["scenes"] = sorted({v["scene"] for v in levels.values() if v["n_t"] == n_t})
        res["status"] = "ESTABLISHED" if res["sliding_mu_max"] is not None and res["static_mu_max"] is not None \
            else "NOT_ESTABLISHED"
        envelope[str(n_t)] = res
    slide = [envelope[str(n)]["sliding_mu_max"] for n in nts]
    drag = R_.get("DRAG_nt4_mu0.4", {})
    out = {"errors": {n: r["error"] for n, r in R_.items() if "error" in r},
           "ENV-01": base["ENV-01"], "ENV-02": base["ENV-02"],
           "ENV-04": {"verdict": "PASS" if spins and all(s["pass"] for s in spins.values()) else "FAIL",
                      "rows": spins},
           "ENV-05": {"verdict": "REPORTED",
                      "drag": {k: drag.get(k) for k in ("rows", "particles", "dt_s", "wall_s", "error")}},
           "ENV-03": {"verdict": "REPORTED"}, "levels": levels, "envelope": envelope, "runs": base["runs"]}
    out["verdict"] = "PASS" if (all(envelope[str(n)]["sliding_mu_max"] == 0.4 and
                                    envelope[str(n)]["static_mu_max"] == 0.4 for n in nts)
                                and not out["errors"]) else "FAIL"
    out["p2_6_friction_range"] = ([0.3, min(slide)] if all(s is not None for s in slide) else "BLOCKED")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if sha != PROTOCOL_SHA:
        raise SystemExit(f"protocol sha256 {sha} != pinned {PROTOCOL_SHA}")
    cfg = _cfg()
    if cfg["common"]["contact_law"] != LAW:
        raise SystemExit("protocol law mismatch")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/"jobs").mkdir()
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa
    import warp as wp
    meta = {"label": "SMOKE (not a formal result)" if a.smoke else "FORMAL rubber_contact_envelope_v2",
            "protocol_sha256": sha, "law": LAW, "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "warp": wp.config.version, "numpy": np.__version__,
            "argv": sys.argv, "start": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "host": platform.node(),
            "cpu_count_used": a.jobs, "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED"}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    _setup(a.smoke)
    table = jobs(cfg)
    size = lambda kv: (kv[1][0] == "inc" and kv[1][1][0] == "STRIP", kv[1][1][1] if kv[1][0] == "inc" else 0)  # noqa
    order = sorted(table.items(), key=lambda kv: (not size(kv)[0], -size(kv)[1], kv[0]))
    R_ = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, res in ex.map(_run, [(n, j, a.smoke) for n, j in order]):
            (out/"jobs"/f"{name}.json").write_text(json.dumps(res, default=float))
            R_[name] = res
            print(name, "ERROR "+res["error"] if "error" in res else f"{res.get('wall_s', 0):.1f} s", flush=True)
    meta["end"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    meta["wall_s"] = time.perf_counter()-t0
    meta["jobs"] = len(table)
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    summary = evaluate(cfg, R_)
    (out/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("VERDICT", summary["verdict"], json.dumps(summary["envelope"]), "P2-6 range", summary["p2_6_friction_range"])
    for k in ("ENV-01", "ENV-02", "ENV-04"):
        print(k, summary[k]["verdict"], summary[k].get("failing_levels", ""))
    for n, s in sorted(summary["ENV-04"]["rows"].items()):
        print(n, s)


if __name__ == "__main__":
    main()
