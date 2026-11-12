"""Formal runner of ``configs/rubber/rubber_contact_envelope_v1.json`` (task ENV2, ADR 0023 §23.2 decision 4), Warp CPU.

Law ``grid_node_penalty_coulomb_v3`` with the protocol's parameters (every ContactSpec built in a job is a v3 spec;
the laws built are recorded per job and checked). Scenes: INC2 (block 24 x 12 x 6 mm) at n_t 5 / 6 and the unseen
STRIP (48 x 16 x t mm, t = n_t mm, dx 1 mm) at n_t 4 / 5 / 6, mu 0.3 / 0.4 / 0.6 / 0.8: per (scene, n_t, mu) an
elastic hold at 0.95 mu and slip runs at 1.1 / 1.2 / 1.3 mu, plus the P2 hold at mu 0.8. The job code is the
``rubber_contact_tests_v3`` incline job (``run_contact_v3.job_inc3``, its helpers from ``run_contact_v1``) with the
scene geometry and n_t as parameters; the per-run metrics are ``run_contact_v3._inc_metrics`` (thresholds read from
this protocol). The decision rule of the protocol gives sliding_mu_max / static_mu_max per n_t and the verdict.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_6/run_contact_envelope_v1.py \
        --out artifacts/rl_ready/p2_6/<dir> [--jobs N] [--smoke] [--evaluate-only]

``--smoke`` shortens every job to 5 control steps (label SMOKE; never a formal result).
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
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_envelope_v1.json"
PROTOCOL_SHA = "d7ecbd3164409cce12d229504864297677baf3307f9763bb7797329506ffe92c"
ENVELOPE_V1 = ROOT/"configs"/"rubber"/"contact_envelope_v1.json"
LAW = "grid_node_penalty_coulomb_v3"
G = 9.81
BUILT = []


def _cfg():
    return json.loads(PROTOCOL.read_text())


def _setup(smoke=False):
    """run_contact_v1 helpers bound to this protocol (materials, options) and ContactSpec forced to the v3 law with the
    protocol's law parameters."""
    import run_contact_v1 as R
    from robotic_cutting_sim import rubber_contact as rct
    R.PROTOCOL = PROTOCOL
    R.SMOKE = smoke
    lp = _cfg()["common"]["law_parameters"]
    base = rct.ContactSpec
    if getattr(base, "_env2_runner", False):        # patched once per process (modules bind the class on import)
        return R

    class SpecV3(base):
        _env2_runner = True

        def __init__(self, *args, **kw):
            if kw.get("law", LAW) != LAW:
                raise ValueError("run_contact_envelope_v1 builds only the v3 law")
            base.__init__(self, *args, **{**kw, "law": LAW, "normal_stiffness": lp["normal_stiffness_kappa_n"],
                                          "tangential_stiffness": lp["tangential_stiffness_kappa_t"],
                                          "normal_damping_ratio": lp["normal_damping_ratio_zeta_n"],
                                          "damping_ratio": lp["tangential_damping_ratio_zeta_t"],
                                          "slip_rate_damping_ratio": lp["slip_rate_damping_ratio_zeta_d"]})
            BUILT.append(self.law)
    rct.ContactSpec = SpecV3
    return R


def job(name, scene, n_t, mu, ratio, material_kind):
    import run_contact_v1 as R
    from robotic_cutting_sim import rubber_contact as rct
    sc = _cfg()["scenes"][scene]
    if scene == "INC2":
        L, W, H = sc["size_m"]
        spec = R._spec(length=L, width=W, thickness=H, n_t=n_t, case="free")
    else:
        H = sc["thickness_m_by_n_t"][str(n_t)]
        spec = R._spec(length=sc["length_m"], width=sc["width_m"], thickness=H, n_t=n_t, dx=sc["dx_m"], case="free",
                       resolution_profile="rubber_ref_v1")
    theta = math.atan(ratio*mu)
    gvec = np.array([G*math.sin(theta), 0.0, -G*math.cos(theta)])
    live = R._live(spec, R._material(material_kind, friction_table=mu), sc["control_dt_s"], (0, 0, 0),
                   rct.ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),)), cfl=_cfg()["common"]["cfl"])
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
            "theta_rad": theta, "material": material_kind, "cfl": _cfg()["common"]["cfl"], "scored": True,
            "particles": int(live.scene.n), "spacing_m": live.scene.spacing, "dx_m": live.scene.dx, "dt_s": live.dt,
            "substeps": live.substeps, "x_cm0": float(np.asarray(live.scene.X, np.float64)[:, 0].mean()),
            "wall_s": time.perf_counter()-t0, "rows": rows, "receipt_contact": live.solver.contact.receipt()}


def jobs(cfg):
    out = {}
    for scene in ("INC2", "STRIP"):
        sc = cfg["scenes"][scene]
        for n_t in sc["n_t_set"]:
            for mu in sc["mu_set"]:
                for r in sc["stick_ratios_tan_theta_over_mu"]+sc["slip_ratios_tan_theta_over_mu"]:
                    out[f"{scene}_nt{n_t}_mu{mu}_r{r}"] = (scene, n_t, mu, r, "elastic")
            for v in sc["visco_stick_rows"]:
                out[f"{scene}_nt{n_t}_P2_mu{v['mu']}_r{v['ratio']}"] = (scene, n_t, v["mu"], v["ratio"], "P2")
    return out


def _run(item):
    name, args, smoke = item
    _setup(smoke)
    BUILT.clear()
    try:
        res = job(name, *args)
        res["contact_laws_built"] = sorted(set(BUILT))
        if res["contact_laws_built"] != [LAW]:
            raise AssertionError(f"job built laws {res['contact_laws_built']}")
        return name, res
    except Exception as exc:                                    # recorded, never hidden
        import traceback
        return name, {"name": name, "error": repr(exc), "traceback": traceback.format_exc()}


def _v(ok):
    return "PASS" if ok else "FAIL"


def evaluate(cfg, R_):
    import run_contact_v3 as V3
    t1e, t2e = cfg["tests"]["ENV-01"]["thresholds"], cfg["tests"]["ENV-02"]["thresholds"]
    t1 = {**t2e, "slip_acc_rel": t1e["slip_acc_rel"], "books_rel": 0.01}   # books_rel: reported books_ok only
    t4 = {"dissipation_rel": t1e["dissipation_rel"]}
    out = {"errors": {n: r["error"] for n, r in R_.items() if "error" in r}}
    runs = [r for r in R_.values() if "error" not in r]
    M = []
    for r in runs:
        m = V3._inc_metrics(r, cfg["scenes"][r["scene_v3"]], t1, t4)
        m.update(particles=r["particles"], dt_s=r["dt_s"], substeps=r["substeps"], wall_s=r["wall_s"],
                 spacing_m=r["spacing_m"], pen_max_rel_s=max(q["pen"] for q in r["rows"])/r["spacing_m"],
                 valid=m["invalid"] == 0)
        m.pop("books_ok", None)
        M.append(m)
    errored = set(out["errors"])
    levels = {}
    for scene in ("INC2", "STRIP"):
        sc = cfg["scenes"][scene]
        for n_t in sc["n_t_set"]:
            for mu in sc["mu_set"]:
                slips = [m for m in M if m["scene"] == scene and m["n_t"] == n_t and m["mu"] == mu
                         and m["kind"] == "slip"]
                holds = [m for m in M if m["scene"] == scene and m["n_t"] == n_t and m["mu"] == mu
                         and m["kind"] == "stick"]
                expect_s = len(sc["slip_ratios_tan_theta_over_mu"])
                expect_h = len(sc["stick_ratios_tan_theta_over_mu"])+sum(
                    1 for v in sc["visco_stick_rows"] if v["mu"] == mu)
                missing = any(n.startswith(f"{scene}_nt{n_t}_") and f"mu{mu}_" in n for n in errored)
                sl_ok = (len(slips) == expect_s and not missing
                         and all(m["slip_ok"] and m["dissipation_ok"] and m["valid"] for m in slips))
                st_ok = (len(holds) == expect_h and not missing
                         and all(m["stick_ok"] and m["ramp_ok"] and m["valid"] for m in holds))
                onset = None
                if len(slips) >= 2:
                    tt = np.array([m["ratio"]*mu for m in slips])
                    aa = np.array([m["a_over_gcos"] for m in slips])
                    slope, icpt = np.polyfit(tt, aa, 1)
                    onset = {"slope": float(slope), "mu_eff": float(-icpt/slope), "rel": float(-icpt/slope/mu-1)}
                levels[f"{scene}_nt{n_t}_mu{mu}"] = {
                    "scene": scene, "n_t": n_t, "mu": mu, "sliding_pass": bool(sl_ok), "static_pass": bool(st_ok),
                    "slip_runs": len(slips), "hold_runs": len(holds),
                    "max_abs_slip_err_over_gcos": max((abs(m["err_over_gcos"]) for m in slips), default=None),
                    "D04": [m["dissipation_ratio"] for m in slips],
                    "max_abs_hold_disp_rel_s": max((abs(m["disp_rel_s"]) for m in holds), default=None),
                    "max_abs_ramp_disp_rel_s": max((abs(m["ramp_disp_rel_s"]) for m in holds), default=None),
                    "max_hold_a_over_gcos": max((abs(m["a_over_gcos"]) for m in holds), default=None),
                    "onset_reported": onset,
                    "books_max_rel_reported": max((m["books_max_rel"] for m in slips+holds), default=None),
                    "pen_max_rel_s": max((m["pen_max_rel_s"] for m in slips+holds), default=None)}
    env1 = json.loads(ENVELOPE_V1.read_text())
    v1_rows = {(r["law"], r["n_t"]): r for r in env1["rows"]}
    nts = sorted({n for s in ("INC2", "STRIP") for n in cfg["scenes"][s]["n_t_set"]})
    mus = sorted(cfg["scenes"]["STRIP"]["mu_set"])
    envelope = {}
    for n_t in nts:
        res = {}
        for kind in ("sliding", "static"):
            best = None
            for mu in mus:
                lv = [v for v in levels.values() if v["n_t"] == n_t and v["mu"] == mu]
                if lv and all(v[f"{kind}_pass"] for v in lv):
                    best = mu
                else:
                    break
            prior = v1_rows.get((LAW, n_t))
            if best is not None and prior is not None and prior["status"] == "ESTABLISHED":
                best = min(best, float(prior[f"{kind}_mu_max"]))
            res[f"{kind}_mu_max"] = best
        res["scenes"] = sorted({v["scene"] for v in levels.values() if v["n_t"] == n_t})
        res["status"] = "ESTABLISHED" if res["sliding_mu_max"] is not None and res["static_mu_max"] is not None \
            else "NOT_ESTABLISHED"
        envelope[str(n_t)] = res
    target = 0.8
    strip4 = [v for v in levels.values() if v["scene"] == "STRIP" and v["n_t"] == 4]
    verdict = (all(envelope[str(n)]["sliding_mu_max"] == target and envelope[str(n)]["static_mu_max"] == target
                   for n in (5, 6))
               and all(v["sliding_pass"] and v["static_pass"] for v in strip4) and not out["errors"])
    slide = [envelope[str(n)]["sliding_mu_max"] for n in nts]
    out["ENV-01"] = {"verdict": _v(all(v["sliding_pass"] for v in levels.values()) and not out["errors"]),
                     "failing_levels": [k for k, v in levels.items() if not v["sliding_pass"]]}
    out["ENV-02"] = {"verdict": _v(all(v["static_pass"] for v in levels.values()) and not out["errors"]),
                     "failing_levels": [k for k, v in levels.items() if not v["static_pass"]]}
    out["ENV-03"] = {"verdict": "REPORTED"}
    out["levels"] = levels
    out["envelope"] = envelope
    out["verdict"] = _v(verdict)
    out["p2_6_friction_range"] = ([0.3, min(slide)] if all(s is not None for s in slide) and min(slide) >= 0.3
                                  else "BLOCKED")
    out["runs"] = M
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if sha != PROTOCOL_SHA:
        raise SystemExit(f"protocol sha256 {sha} != pinned {PROTOCOL_SHA}")
    cfg = _cfg()
    if cfg["common"]["contact_law"] != LAW:
        raise SystemExit("protocol law mismatch")
    out = Path(a.out)
    if a.evaluate_only:
        R_ = {p.stem: json.loads(p.read_text()) for p in sorted((out/"jobs").glob("*.json"))}
        target = out/"summary_reevaluated.json"
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        target.write_text(json.dumps(evaluate(cfg, R_), indent=1, default=float))
        return
    out.mkdir(parents=True, exist_ok=False)
    (out/"jobs").mkdir()
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa
    import warp as wp
    meta = {"label": "SMOKE (not a formal result)" if a.smoke else "FORMAL rubber_contact_envelope_v1",
            "protocol_sha256": sha, "law": LAW, "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "warp": wp.config.version, "numpy": np.__version__,
            "argv": sys.argv, "start": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "host": platform.node(),
            "cpu_count_used": a.jobs, "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED"}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    table = jobs(cfg)
    order = sorted(table.items(), key=lambda kv: (-kv[1][1], -(kv[1][0] == "STRIP"), kv[0]))   # biggest first
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
    for k in ("ENV-01", "ENV-02"):
        print(k, summary[k]["verdict"], summary[k]["failing_levels"])


if __name__ == "__main__":
    main()
