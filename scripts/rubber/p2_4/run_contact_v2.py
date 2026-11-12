"""Formal runner of ``configs/rubber/rubber_contact_tests_v2.json`` (P2-4 task X1, ADR 0023 §22.9), single env, Warp CPU.

Runs RUBBER-CONT-01 / -04 (INC, INC2), -02 (PEN), -03 (IMP), -05 (PINCH) with the contact law
``grid_node_penalty_coulomb_v2``. The scene code, loads and measurements of INC / PEN / IMP / PINCH are those of
``run_contact_v1.py`` (imported; its protocol path is pointed at v2, whose ``common`` block and v1 scenes are equal):
every ``ContactSpec`` the v1 job functions build is built with ``law='grid_node_penalty_coulomb_v2'`` (the law of every
built spec is recorded per job and checked). INC2 (the unseen envelope scene) and the elastic 0.95 stick row are added.
The evaluation is ``run_contact_v1.evaluate`` (same criteria and thresholds) plus the v2 changes of the protocol:
-02 f_law_v2, -05 d as the window-mean momentum balance (the v1 per-step metric reported). -06 NOT_IMPLEMENTED_V1;
-07 CARRIED_OVER (contact-free; contact_v1_001); -08 is the pytest file ``tests/test_rubber_contact_v2.py``; -09 is the
GPU task B2-v2; -10 the pytest files; -11 not re-measured.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/run_contact_v2.py \
        --out artifacts/rl_ready/p2_4/<dir> [--jobs N] [--only INC,INC2,PEN,IMP,PINCH] [--smoke]
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
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_tests_v2.json"
PROTOCOL_SHA = "89bc7680bb9c2ed6322279b198899cef0faf31c4751b6bd849b22ecd24a339d0"
LAW = "grid_node_penalty_coulomb_v2"
G = 9.81
BUILT = []


def _setup(smoke=False):
    """Point run_contact_v1 at the v2 protocol and make every ContactSpec a v2-law spec (in this process)."""
    import run_contact_v1 as R
    from robotic_cutting_sim import rubber_contact as rct
    R.PROTOCOL = PROTOCOL
    R.SMOKE = smoke
    base = rct.ContactSpec
    if not getattr(base, "_v2_runner", False):
        class SpecV2(base):
            _v2_runner = True

            def __init__(self, *args, **kw):
                if kw.get("law", LAW) != LAW:
                    raise ValueError("run_contact_v2 builds only the v2 law")
                base.__init__(self, *args, **{**kw, "law": LAW})
                BUILT.append(self.law)
        rct.ContactSpec = SpecV2
    return R


def job_inc2(name, mu, ratio, material_kind, cfl):
    """INC2: the unseen envelope block (protocol scenes.INC2), otherwise the INC job of run_contact_v1."""
    import run_contact_v1 as R
    from robotic_cutting_sim import rubber_contact as rct
    sc = R._cfg()["scenes"]["INC2"]
    L, W, H = sc["size_m"]
    theta = math.atan(ratio*mu)
    gvec = np.array([G*math.sin(theta), 0.0, -G*math.cos(theta)])
    spec = R._spec(length=L, width=W, thickness=H, n_t=sc["n_t"], case="free")
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
    return {"name": name, "scene": "INC", "scene_v2": "INC2", "mu": mu, "ratio": ratio, "theta_rad": theta,
            "material": material_kind, "cfl": cfl, "particles": int(live.scene.n), "spacing_m": live.scene.spacing,
            "dt_s": live.dt, "substeps": live.substeps,
            "x_cm0": float(np.asarray(live.scene.X, np.float64)[:, 0].mean()), "wall_s": time.perf_counter()-t0,
            "rows": rows, "receipt_contact": live.solver.contact.receipt()}


def jobs(cfg, only):
    import run_contact_v1 as R
    out = {k: v for k, v in R.jobs(cfg, only & {"INC", "PEN", "IMP", "PINCH"}).items()}
    if "INC" in only:
        r = cfg["scenes"]["INC"]["elastic_stick_row_near_threshold"]
        out[f"INC_mu{r['mu']}_r{r['ratio']}"] = (R.job_inc, (r["mu"], r["ratio"], "elastic", 0.5))
    if "INC2" in only:
        sc = cfg["scenes"]["INC2"]
        for mu in sc["mu_set"]:
            for ratio in sc["stick_ratios_tan_theta_over_mu"]+sc["slip_ratios_tan_theta_over_mu"]:
                out[f"INC2_mu{mu}_r{ratio}"] = (job_inc2, (mu, ratio, "elastic", 0.5))
        v = sc["visco_stick_row"]
        out[f"INC2_P2_mu{v['mu']}_r{v['ratio']}"] = (job_inc2, (v["mu"], v["ratio"], "P2", 0.5))
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


def evaluate(cfg, R_):
    import run_contact_v1 as R
    out = R.evaluate(cfg, R_)
    th = {k: cfg["tests"][k].get("thresholds", {}) for k in cfg["tests"]}
    # -02 f_law_v2: damped runs, sum k_n delta_I / reaction within law_rel
    pen = {n: r for n, r in R_.items() if r.get("scene") == "PEN" and "error" not in r}
    if pen and "RUBBER-CONT-02" in out:
        t2 = th["RUBBER-CONT-02"]
        f = [{"run": r["name"], "spring_sum_over_reaction": r["spring_sum_over_reaction"],
              "inner_ratio": r["inner_ratio_to_undamped_law"],
              "ok": abs(r["spring_sum_over_reaction"]-1) <= t2["law_rel"]} for r in pen.values() if r["damped"]]
        o = out["RUBBER-CONT-02"]
        o["detail"]["f_law_v2"] = f
        o["criteria"]["f_law_v2"] = _v(all(q["ok"] for q in f)) if f else "NOT_RUN"
        o["verdict"] = _v(all(x == "PASS" for x in o["criteria"].values()))
    # -05 d: window-mean momentum balance (the v1 per-step metric reported)
    pin = {n: r for n, r in R_.items() if r.get("scene") == "PINCH" and "error" not in r}
    if pin and "RUBBER-CONT-05" in out:
        t5 = th["RUBBER-CONT-05"]
        sc = cfg["scenes"]["PINCH"]
        T_end = sc["squeeze_ramp_s"]+sc["hold_s"]+sc["pull_duration_s"]
        d, rep = [], []
        for n, r in sorted(pin.items(), key=lambda kv: kv[1]["depth_m"]):
            win = [q for q in r["rows"] if q["t"] > T_end-0.01+1e-12] or r["rows"][-3:]
            Tx = np.array([q["F_top"][0]+q["F_bot"][0] for q in win])
            C = np.array([q["clamp"][0] for q in win])
            mean_bal = float(abs(np.mean(C+Tx))/max(np.mean(np.abs(Tx)), 1e-300))
            per_step = float(np.mean(np.abs(C+Tx))/max(np.mean(np.abs(Tx)), 1e-300))
            d.append({"run": n, "rel": mean_bal, "ok": mean_bal <= t5["balance_rel"]})
            rep.append({"run": n, "per_step_rel_v1_metric": per_step,
                        "rms_over_mean_T": float(np.std(C+Tx)/max(np.mean(np.abs(Tx)), 1e-300))})
        o = out["RUBBER-CONT-05"]
        o["detail"]["d_balance_v1_per_step_reported"] = rep
        o["detail"]["d_balance"] = d
        o["criteria"]["d_balance"] = _v(all(q["ok"] for q in d))
        o["criteria"]["d_reported_v1_per_step"] = "REPORTED"
        o["verdict"] = _v(all(x == "PASS" for k, x in o["criteria"].items() if k != "d_reported_v1_per_step"))
    out["RUBBER-CONT-07"] = {"verdict": "CARRIED_OVER", "reason": cfg["tests"]["RUBBER-CONT-07"]["status"]}
    out["RUBBER-CONT-08"] = {"verdict": "SEE_PYTEST", "reason": cfg["tests"]["RUBBER-CONT-08"]["under_v2"]}
    out["RUBBER-CONT-09"] = {"verdict": "NOT_RUN", "reason": "GPU task B2-v2 (protocol RUBBER-CONT-09.run)"}
    out["RUBBER-CONT-10"] = {"verdict": "SEE_PYTEST", "reason": cfg["tests"]["RUBBER-CONT-10"]["under_v2"]}
    out["RUBBER-CONT-11"] = {"verdict": "NOT_RUN", "reason": cfg["tests"]["RUBBER-CONT-11"]["note_v2"]}
    out["envelope_rows"] = {n: {"mu": r["mu"], "ratio": r["ratio"], "material": r["material"],
                                "label": ([] if r["mu"] <= 0.6 or r["ratio"] < 1 else ["contact_sliding_friction_deficit_v2"])
                                + (["contact_friction_flutter"] if r["mu"] > 0.8 else [])}
                            for n, r in R_.items() if r.get("scene") == "INC" and "error" not in r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--only", default="INC,INC2,PEN,IMP,PINCH")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if sha != PROTOCOL_SHA:
        raise SystemExit(f"protocol sha256 {sha} != pinned {PROTOCOL_SHA}")
    R = _setup(a.smoke)
    cfg = R._cfg()
    out = Path(a.out)
    if a.evaluate_only:
        R_ = {p.stem: json.loads(p.read_text()) for p in sorted((out/"jobs").glob("*.json"))}
        (out/"summary_reevaluated.json").write_text(json.dumps(evaluate(cfg, R_), indent=1, default=float))
        return
    out.mkdir(parents=True, exist_ok=False)
    (out/"jobs").mkdir()
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa
    import warp as wp
    meta = {"label": "SMOKE (not a formal result)" if a.smoke else "FORMAL rubber_contact_tests_v2",
            "protocol_sha256": sha, "law": LAW, "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "warp": wp.config.version, "numpy": np.__version__,
            "argv": sys.argv, "start": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "host": platform.node(),
            "cpu_count_used": a.jobs}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    table = jobs(cfg, set(a.only.split(",")))
    R_ = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, res in ex.map(_run, [(n, j, a.smoke) for n, j in sorted(table.items())]):
            (out/"jobs"/f"{name}.json").write_text(json.dumps(res, default=float))
            R_[name] = res
            print(name, "ERROR "+res["error"] if "error" in res else f"{res.get('wall_s', 0):.1f} s", flush=True)
    meta["end"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    meta["wall_s"] = time.perf_counter()-t0
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    summary = evaluate(cfg, R_)
    (out/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for tid, v in summary.items():
        if tid.startswith("RUBBER"):
            print(tid, v.get("verdict"), v.get("criteria", v.get("reason")))


if __name__ == "__main__":
    main()
