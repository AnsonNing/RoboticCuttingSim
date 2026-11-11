"""Formal runner of ``configs/rubber/rubber_contact_tests_v3.json`` (P2-4 task X3, ADR 0023 §22.11), single env, Warp CPU.

Law ``grid_node_penalty_coulomb_v3`` (zeta_d from the protocol): every ``ContactSpec`` built in a job is a v3 spec (the
laws built are recorded per job and checked). Scenes: INC / INC2 at n_t 3 and n_t 4 (scored mu 0.2-0.8: slip, stick,
P2 stick, onset, dissipation, books, load-ramp slip; mu 1.0 rows reported with labels only), PEN / IMP (the v1 job code
of ``run_contact_v1.py``, frictionless: v3 = v2 there), PINCH at n_t 3 and n_t 4 (the v1 pinch scene with the
resolution as a parameter). Evaluation follows the protocol: -01 a-f and -04 per group (scene, n_t); -02 / -03 by
``run_contact_v1.evaluate`` plus -02 f; -05 per resolution with the window-mean balance (v2 rule) and the v1 per-step
metric reported. -06 NOT_IMPLEMENTED_V1, -07 CARRIED_OVER, -08 / -10 pytest, -09 / -11 GPU (separate scripts).

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/run_contact_v3.py \
        --out artifacts/rl_ready/p2_4/<dir> [--jobs N] [--only INC,PEN,IMP,PINCH] [--smoke]
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
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_tests_v3.json"
PROTOCOL_SHA = "36d70beaf63a0ef73195dcc47c62e522596779bb95b423a5544e5f7dc9e6a856"
LAW = "grid_node_penalty_coulomb_v3"
G = 9.81
BUILT = []


def _setup(smoke=False):
    import run_contact_v1 as R
    from robotic_cutting_sim import rubber_contact as rct
    R.PROTOCOL = PROTOCOL
    R.SMOKE = smoke
    zd = R._cfg()["common"]["law_parameters"]["slip_rate_damping_ratio_zeta_d"]
    base = rct.ContactSpec
    if not getattr(base, "_v3_runner", False):
        class SpecV3(base):
            _v3_runner = True

            def __init__(self, *args, **kw):
                if kw.get("law", LAW) != LAW:
                    raise ValueError("run_contact_v3 builds only the v3 law")
                base.__init__(self, *args, **{**kw, "law": LAW, "slip_rate_damping_ratio": zd})
                BUILT.append(self.law)
        rct.ContactSpec = SpecV3
    return R


def job_inc3(name, scene, n_t, mu, ratio, material_kind, cfl, scored):
    import run_contact_v1 as R
    from robotic_cutting_sim import rubber_contact as rct
    sc = R._cfg()["scenes"][scene]
    L, W, H = sc["size_m"]
    theta = math.atan(ratio*mu)
    gvec = np.array([G*math.sin(theta), 0.0, -G*math.cos(theta)])
    spec = R._spec(length=L, width=W, thickness=H, n_t=n_t, case="free")
    live = R._live(spec, R._material(material_kind, friction_table=mu), sc["control_dt_s"], (0, 0, 0),
                   rct.ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),)), cfl=cfl)
    T, Tr, cdt = sc["duration_s"], sc["ramp_s"], sc["control_dt_s"]
    if R.SMOKE:
        T = 5*cdt
    st = R.Stepper(live, lambda t: (gvec*R._ramp(t, Tr), None))
    rows = []
    t0 = time.perf_counter()
    for k in range(int(round(T/cdt))):
        st.advance()
        e = R._energies(live)
        e.update(t=(k+1)*cdt, x_cm=float(st.xcm[0]), N=float(-live.last_contact["plane_wrench"][0, 2]),
                 Ft=float(live.last_contact["plane_wrench"][0, 0]), W_g=st.W_g)
        e["B"] = e["KE"]+e["psi"]+e["Wb"]-st.W_g-e["W_ext"]-(e["W_n"]+e["W_t"])
        rows.append(e)
    return {"name": name, "scene": "INC3", "scene_v3": scene, "n_t": n_t, "mu": mu, "ratio": ratio,
            "theta_rad": theta, "material": material_kind, "cfl": cfl, "scored": scored,
            "particles": int(live.scene.n), "spacing_m": live.scene.spacing, "dt_s": live.dt,
            "substeps": live.substeps, "x_cm0": float(np.asarray(live.scene.X, np.float64)[:, 0].mean()),
            "wall_s": time.perf_counter()-t0, "rows": rows, "receipt_contact": live.solver.contact.receipt()}


def job_pinch3(name, depth, n_t):
    """``run_contact_v1.job_pinch`` with the resolution n_t as a parameter (protocol scenes.PINCH.n_t_set)."""
    import run_contact_v1 as R
    from robotic_cutting_sim.rubber_contact import ContactFrame, ContactSpec
    sc = R._cfg()["scenes"]["PINCH"]
    L, W, t = 0.03, 0.01, 0.004
    spec = R._spec(length=L, width=W, thickness=t, n_t=n_t, case="A")
    top = ContactFrame(kind="pad", name="top", origin=(L-0.003, W/2, t+0.001),
                       boxes=(((L-0.003, W/2, t+0.001), (0.005, W/2+0.002, 0.001)),))
    bot = ContactFrame(kind="pad", name="bottom", origin=(L-0.003, W/2, -0.001),
                       boxes=(((L-0.003, W/2, -0.001), (0.005, W/2+0.002, 0.001)),))
    live = R._live(spec, R._material("elastic", friction_pad=sc["mu_pad"]), sc["control_dt_s"], (0, 0, 0),
                   ContactSpec(frames=(top, bot)))
    s = live.solver
    cdt, Tr, Th, vp, Tp = (sc["control_dt_s"], sc["squeeze_ramp_s"], sc["hold_s"], sc["pull_speed_m_s"],
                           sc["pull_duration_s"])
    T = Tr+Th+Tp
    if R.SMOKE:
        T = 5*cdt
    rows = []
    t0 = time.perf_counter()
    for k in range(int(round(T/cdt))):
        te = (k+1)*cdt
        sq = depth*R._ramp(te, Tr)
        px = vp*max(0.0, te-Tr-Th)
        w = live.advance(frames={0: {"translation": np.array([px, 0.0, -sq])},
                                 1: {"translation": np.array([px, 0.0, sq])}})
        bw = live.last_contact["box_wrench"]
        e = R._energies(live)
        e.update(t=te, pull_x=px, squeeze=sq, tip_x=float(live.markers()[-1][0]), F_top=bw[0, 0].tolist(),
                 F_bot=bw[1, 0].tolist(), clamp=np.asarray(w)[0].tolist())
        rows.append(e)
    return {"name": name, "scene": "PINCH3", "n_t": n_t, "depth_m": depth, "mu_pad": sc["mu_pad"],
            "particles": int(s.n), "spacing_m": live.scene.spacing, "dt_s": live.dt,
            "wall_s": time.perf_counter()-t0, "rows": rows}


def jobs(cfg, only):
    import run_contact_v1 as R
    out = {}
    if "INC" in only:
        for scene in ("INC", "INC2"):
            sc = cfg["scenes"][scene]
            for n_t in sc["n_t_set"]:
                for mu in sc["mu_set"]:
                    for r in sc["stick_ratios_tan_theta_over_mu"]+sc["slip_ratios_tan_theta_over_mu"]:
                        out[f"{scene}_nt{n_t}_mu{mu}_r{r}"] = (job_inc3, (scene, n_t, mu, r, "elastic", 0.5, True))
                for v in sc["visco_stick_rows"]:
                    out[f"{scene}_nt{n_t}_P2_mu{v['mu']}_r{v['ratio']}"] = (
                        job_inc3, (scene, n_t, v["mu"], v["ratio"], "P2", 0.5, True))
                rep = sc["reported_mu_1_rows"]
                for r in rep["ratios"]:
                    out[f"{scene}_nt{n_t}_mu{rep['mu']}_r{r}_reported"] = (
                        job_inc3, (scene, n_t, rep["mu"], r, "elastic", 0.5, False))
        b = cfg["scenes"]["INC"]["books_order_row"]
        out[f"INC_nt{b['n_t']}_mu{b['mu']}_r{b['ratio']}_cfl0.25"] = (
            job_inc3, ("INC", b["n_t"], b["mu"], b["ratio"], "elastic", 0.25, True))
    sub = R.jobs(cfg, only & {"PEN", "IMP"})
    out.update(sub)
    if "PINCH" in only:
        sc = cfg["scenes"]["PINCH"]
        for n_t in sc["n_t_set"]:
            for d in sc["squeeze_depths_m"]:
                out[f"PINCH_nt{n_t}_d{d:.0e}"] = (job_pinch3, (d, n_t))
    return out


def _run(item):
    name, (fn, args), smoke = item
    _setup(smoke)
    BUILT.clear()
    try:
        res = fn(name, *args)
        res["contact_laws_built"] = sorted(set(BUILT))
        if res["contact_laws_built"] != [LAW]:
            raise AssertionError(f"job built laws {res['contact_laws_built']}")
        return name, res
    except Exception as exc:                                    # recorded, never hidden
        import traceback
        return name, {"name": name, "error": repr(exc), "traceback": traceback.format_exc()}


def _v(ok):
    return "PASS" if ok else "FAIL"


def _inc_metrics(r, sc, t1, t4):
    rows = r["rows"]
    t = np.array([x["t"] for x in rows])
    x = np.array([x["x_cm"] for x in rows])-r["x_cm0"]
    gc = G*math.cos(r["theta_rad"])
    sel = t >= sc["fit_window_s"][0]-1e-12
    if sel.sum() < 3:
        sel = np.ones_like(sel)
    a = 2*np.polyfit(t[sel], x[sel], 2)[0]
    S = max(q["KE"]+q["psi"] for q in rows)+max(abs(q["W_g"]) for q in rows)
    Bmax = max(abs(q["B"]) for q in rows)
    out = {"run": r["name"], "scene": r["scene_v3"], "n_t": r["n_t"], "mu": r["mu"], "ratio": r["ratio"],
           "material": r["material"], "cfl": r["cfl"], "a_fit": a, "a_over_gcos": a/gc, "books_max_rel": Bmax/S,
           "S_J": S, "books_max_J": Bmax, "invalid": max(q["invalid"] for q in rows)}
    out["books_ok"] = Bmax <= t1["books_rel"]*S
    s = r["spacing_m"]
    if r["ratio"] < 1:
        i20 = int(np.argmin(np.abs(t-sc["hold_window_s"][0])))
        iramp = int(np.argmin(np.abs(t-sc["ramp_s"])))
        disp = x[-1]-x[i20]
        ramp = x[iramp]                                          # x_cm(T_ramp) - x_cm(0) (x_cm0 = reference)
        out.update(kind="stick", disp_m=disp, disp_rel_s=disp/s, ramp_disp_m=ramp, ramp_disp_rel_s=ramp/s,
                   stick_ok=abs(disp) <= t1["stick_disp_rel"]*s and abs(a) <= t1["stick_acc_rel"]*gc,
                   ramp_ok=abs(ramp) <= t1["ramp_disp_rel"]*s)
    else:
        aref = G*(math.sin(r["theta_rad"])-r["mu"]*math.cos(r["theta_rad"]))
        N = np.array([q["N"] for q in rows])
        dxs = np.diff(np.r_[0.0, x])
        ref = r["mu"]*float(np.sum(N*dxs))
        last = rows[-1]
        D = -(last["W_n"]+last["W_t"])
        out.update(kind="slip", a_ref=aref, err_over_gcos=(a-aref)/gc, slip_ok=abs(a-aref) <= t1["slip_acc_rel"]*gc,
                   D_contact_J=D, mu_sum_N_dx_J=ref, dissipation_ratio=D/ref,
                   split={"coulomb": last["coulomb"]/ref, "dashpots_and_equalisation": last["damping"]/ref,
                          "normal_work": -last["W_n"]/ref, "tangential_work": -last["W_t"]/ref},
                   dissipation_ok=abs(D/ref-1) <= t4["dissipation_rel"])
    return out


def evaluate(cfg, R_):
    import run_contact_v1 as R
    th = {k: cfg["tests"][k].get("thresholds", {}) for k in cfg["tests"]}
    t1, t4, t5 = th["RUBBER-CONT-01"], th["RUBBER-CONT-04"], th["RUBBER-CONT-05"]
    out = {"errors": {n: r["error"] for n, r in R_.items() if "error" in r}}
    inc = [r for r in R_.values() if r.get("scene") == "INC3" and "error" not in r]
    if inc:
        M = [_inc_metrics(r, cfg["scenes"][r["scene_v3"]], t1, t4) for r in inc]
        scored = [m for m, r in zip(M, inc) if r["scored"]]
        reported = [m for m, r in zip(M, inc) if not r["scored"]]
        crit = {"a_stick": [m for m in scored if m["kind"] == "stick"],
                "b_slip": [m for m in scored if m["kind"] == "slip"],
                "d_books": scored, "e_validity": M, "f_ramp_slip": [m for m in scored if m["kind"] == "stick"]}
        onset = []
        groups = {}
        for m in scored:
            if m["kind"] == "slip" and m["cfl"] == 0.5 and m["material"] == "elastic":
                groups.setdefault((m["scene"], m["n_t"], m["mu"]), []).append((math.tan(math.atan(m["ratio"]*m["mu"])),
                                                                               m["a_over_gcos"]))
        for (scene, n_t, mu), pts in sorted(groups.items()):
            tt, aa = np.array(pts).T
            slope, icpt = np.polyfit(tt, aa, 1)
            mu_eff = -icpt/slope
            onset.append({"scene": scene, "n_t": n_t, "mu": mu, "slope": slope, "mu_eff": mu_eff, "rel": mu_eff/mu-1,
                          "ok": abs(mu_eff/mu-1) <= t1["onset_rel"]})
        b = cfg["scenes"]["INC"]["books_order_row"]
        n5 = f"INC_nt{b['n_t']}_mu{b['mu']}_r{b['ratio']}"
        n25 = n5+"_cfl0.25"
        order = []
        m5 = next((m for m in scored if m["run"] == n5), None)
        m25 = next((m for m in scored if m["run"] == n25), None)
        if m5 and m25:
            order.append({"run": "order", "ratio": m25["books_max_rel"]/max(m5["books_max_rel"], 1e-300),
                          "ok": m25["books_max_rel"] <= t1["ratio_max"]*m5["books_max_rel"]
                          or max(m5["books_max_rel"], m25["books_max_rel"]) <= t1["floor_rel"]})
        v01 = {"a_stick": _v(all(m["stick_ok"] for m in crit["a_stick"])),
               "b_slip": _v(all(m["slip_ok"] for m in crit["b_slip"])),
               "c_onset": _v(all(q["ok"] for q in onset)) if onset else "NOT_RUN",
               "d_books": _v(all(m["books_ok"] for m in scored) and all(q["ok"] for q in order)),
               "e_validity": _v(all(m["invalid"] == 0 for m in M)),
               "f_ramp_slip": _v(all(m["ramp_ok"] for m in crit["f_ramp_slip"]))}
        fails = {k: [m["run"] for m in crit[k] if not m.get({"a_stick": "stick_ok", "b_slip": "slip_ok",
                                                             "d_books": "books_ok", "f_ramp_slip": "ramp_ok"}.get(k, "x"),
                                                            m.get("invalid", 0) == 0)]
                 for k in ("a_stick", "b_slip", "d_books", "f_ramp_slip")}
        fails["d_books_order"] = [q for q in order if not q["ok"]]
        fails["c_onset"] = [q for q in onset if not q["ok"]]
        out["RUBBER-CONT-01"] = {"criteria": v01, "verdict": _v(all(x == "PASS" for x in v01.values())),
                                 "failing_rows": fails, "onset": onset, "books_order": order,
                                 "detail": {"scored": scored, "reported_mu_1": reported}}
        slips = [m for m in scored if m["kind"] == "slip"]
        out["RUBBER-CONT-04"] = {"criteria": {"a_total": _v(all(m["dissipation_ok"] for m in slips))},
                                 "verdict": _v(all(m["dissipation_ok"] for m in slips)),
                                 "failing_rows": [m["run"] for m in slips if not m["dissipation_ok"]]}
        out["reported_mu_1_rows"] = {
            "label": ["contact_coulomb_ill_posed", "contact_friction_flutter"], "scored": False,
            "rows": [{k: m.get(k) for k in ("run", "scene", "n_t", "ratio", "err_over_gcos", "dissipation_ratio",
                                            "disp_m", "ramp_disp_m", "books_max_rel", "invalid")} for m in reported]}
    base = {n: r for n, r in R_.items() if r.get("scene") in ("PEN", "IMP") and "error" not in r}
    if base:
        b = R.evaluate(cfg, base)
        for k in ("RUBBER-CONT-02", "RUBBER-CONT-03"):
            if k in b:
                out[k] = b[k]
        pen = [r for r in base.values() if r.get("scene") == "PEN"]
        if pen and "RUBBER-CONT-02" in out:
            t2 = th["RUBBER-CONT-02"]
            f = [{"run": r["name"], "spring_sum_over_reaction": r["spring_sum_over_reaction"],
                  "ok": abs(r["spring_sum_over_reaction"]-1) <= t2["law_rel"]} for r in pen if r["damped"]]
            o = out["RUBBER-CONT-02"]
            o["detail"]["f_law_v2"] = f
            o["criteria"]["f_law_v2"] = _v(all(q["ok"] for q in f)) if f else "NOT_RUN"
            o["verdict"] = _v(all(x == "PASS" for x in o["criteria"].values()))
    pin = [r for r in R_.values() if r.get("scene") == "PINCH3" and "error" not in r]
    if pin:
        sc = cfg["scenes"]["PINCH"]
        T_end = sc["squeeze_ramp_s"]+sc["hold_s"]+sc["pull_duration_s"]
        groups = {}
        for r in sorted(pin, key=lambda r: (r["n_t"], r["depth_m"])):
            rows = r["rows"]
            win = [q for q in rows if q["t"] > T_end-0.01+1e-12] or rows[-3:]
            Nt = lambda q: q["F_top"][2]-q["F_bot"][2]                    # noqa: E731
            Tx = lambda q: q["F_top"][0]+q["F_bot"][0]                    # noqa: E731
            mu = r["mu_pad"]
            ratio = float(np.mean([abs(Tx(q))/(mu*Nt(q)) if Nt(q) > 0 else 0.0 for q in win]))
            worst = max((abs(Tx(q))/(mu*Nt(q)) for q in rows if Nt(q) > 0), default=0.0)
            Txw = np.array([Tx(q) for q in win])
            C = np.array([q["clamp"][0] for q in win])
            mean_bal = float(abs(np.mean(C+Txw))/max(np.mean(np.abs(Txw)), 1e-300))
            per_step = float(np.mean(np.abs(C+Txw))/max(np.mean(np.abs(Txw)), 1e-300))
            pmax = max(q["pen"] for q in rows)
            groups.setdefault(r["n_t"], []).append({
                "run": r["name"], "n_t": r["n_t"], "depth_m": r["depth_m"], "ratio": ratio,
                "N_window_N": float(np.mean([Nt(q) for q in win])), "cone_max_ratio": worst, "balance_rel": mean_bal,
                "per_step_rel_v1_metric": per_step, "pen_rel_s": pmax/r["spacing_m"],
                "a_ok": t5["slip_ratio_band"][0] <= ratio <= t5["slip_ratio_band"][1], "b_ok": worst <= t5["cone_tol"],
                "d_ok": mean_bal <= t5["balance_rel"], "e_ok": pmax <= t5["pen_max_rel"]*r["spacing_m"]})
        scaling = []
        for n_t, g in sorted(groups.items()):
            Ns, ratios = [q["N_window_N"] for q in g], [q["ratio"] for q in g]
            distinct = all(Ns[i+1] >= 1.3*Ns[i] for i in range(len(Ns)-1))
            spread = (max(ratios)-min(ratios))/np.mean(ratios)
            scaling.append({"n_t": n_t, "N": Ns, "ratios": ratios, "spread": spread,
                            "ok": len(Ns) >= 2 and distinct and spread <= t5["scaling_rel"]})
        rows5 = [q for g in groups.values() for q in g]
        v5 = {"a_slip_ratio": _v(all(q["a_ok"] for q in rows5)), "b_cone": _v(all(q["b_ok"] for q in rows5)),
              "c_grip_scaling": _v(all(q["ok"] for q in scaling)), "d_balance": _v(all(q["d_ok"] for q in rows5)),
              "e_penetration": _v(all(q["e_ok"] for q in rows5)), "d_reported_v1_per_step": "REPORTED"}
        out["RUBBER-CONT-05"] = {"criteria": v5, "detail": {"rows": rows5, "scaling": scaling},
                                 "verdict": _v(all(x == "PASS" for k, x in v5.items() if k != "d_reported_v1_per_step"))}
    out["RUBBER-CONT-06"] = {"verdict": "NOT_RUN", "reason": cfg["tests"]["RUBBER-CONT-06"]["status"]}
    out["RUBBER-CONT-07"] = {"verdict": "CARRIED_OVER", "reason": cfg["tests"]["RUBBER-CONT-07"]["status"]}
    out["RUBBER-CONT-08"] = {"verdict": "SEE_PYTEST", "reason": cfg["tests"]["RUBBER-CONT-08"]["under_v3"]}
    out["RUBBER-CONT-09"] = {"verdict": "NOT_RUN_HERE", "reason": cfg["tests"]["RUBBER-CONT-09"]["run"]}
    out["RUBBER-CONT-10"] = {"verdict": "SEE_PYTEST", "reason": cfg["tests"]["RUBBER-CONT-10"]["under_v3"]}
    out["RUBBER-CONT-11"] = {"verdict": "NOT_RUN_HERE", "reason": cfg["tests"]["RUBBER-CONT-11"]["measure"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--only", default="INC,PEN,IMP,PINCH")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if sha != PROTOCOL_SHA:
        raise SystemExit(f"protocol sha256 {sha} != pinned {PROTOCOL_SHA}")
    R = _setup(a.smoke)
    cfg = R._cfg()
    if cfg["common"]["contact_law"] != LAW:
        raise SystemExit("protocol law mismatch")
    out = Path(a.out)
    if a.evaluate_only:
        R_ = {p.stem: json.loads(p.read_text()) for p in sorted((out/"jobs").glob("*.json"))}
        (out/"summary_reevaluated.json").write_text(json.dumps(evaluate(cfg, R_), indent=1, default=float))
        return
    out.mkdir(parents=True, exist_ok=False)
    (out/"jobs").mkdir()
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa
    import warp as wp
    meta = {"label": "SMOKE (not a formal result)" if a.smoke else "FORMAL rubber_contact_tests_v3",
            "protocol_sha256": sha, "law": LAW, "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "warp": wp.config.version, "numpy": np.__version__,
            "argv": sys.argv, "start": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "host": platform.node(),
            "cpu_count_used": a.jobs}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    table = jobs(cfg, set(a.only.split(",")))
    # longest jobs first (n_t 4, INC2, PINCH) so the pool drains evenly
    order = sorted(table.items(), key=lambda kv: (-("nt4" in kv[0]), -("PINCH" in kv[0]), -("INC2" in kv[0]), kv[0]))
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
    for tid, v in summary.items():
        if tid.startswith("RUBBER"):
            print(tid, v.get("verdict"), v.get("criteria", v.get("reason")))


if __name__ == "__main__":
    main()
