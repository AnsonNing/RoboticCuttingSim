"""Formal runner of ``configs/rubber/rubber_moving_grip_tests_v1.json`` (RUBBER-GRIP-01..04; ADR 0023 §21.1).

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/run_moving_grip_v1.py \
        --out artifacts/rl_ready/p2_3/<dir> [--workers 8] [--smoke]

``--smoke`` truncates every schedule to 3 control steps (a runner check; its verdicts are not results). The output
directory must not exist (no overwrite). Raw per-run series in ``raw/``, verdicts in ``summary.json``.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_moving_grip_tests_v1.json"
OPTIONS = {"dirichlet": "consistent_mass_projection_v1", "constraint_work": "kick_average_v1",
           "state_precision": "f64_shadow_v1"}


def load_protocol():
    raw = PROTOCOL.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def profile_value(name, tau):
    return tau if name == "linear" else tau**3*(10-15*tau+6*tau*tau)


def schedule(scene, profile, smoke=False):
    """Unit motion parameter s at the end of every control step (protocol ``schedule_rule``)."""
    out = []
    for ph in scene["phases"]:
        n = ph["steps"]
        for k in range(n):
            tau = (k+1)/n
            if ph["kind"] == "ramp":
                out.append(profile_value(profile, tau))
            elif ph["kind"] == "ramp_down":
                out.append(1.0-profile_value(profile, tau))
            else:
                out.append(out[-1] if out else 0.0)
    hold_n = scene["phases"][-1]["steps"]
    if smoke:
        out = out[:3]
        hold_n = 1
    return out, hold_n


def make_live(scene, *, L=None, mu=None, rho=None, step_material=None, substeps=None):
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_material import MaterialValues
    extra = {"end_mask": scene["end_mask"]} if scene["case"] == "C" else {}
    spec = rs.RubberStripSpec(length=scene["L_m"] if L is None else L, width=scene["W_m"], thickness=scene["t_m"],
                              n_t=scene["n_t"], case=scene["case"], **extra)
    mu = scene["mu_pa"] if mu is None else mu
    rho = scene["rho_kg_m3"] if rho is None else rho
    mat = MaterialValues(mu0=mu, kappa=20.0*mu, density=rho)
    return LiveRubberMaterial(spec, material=mat, control_dt=scene["control_dt_s"], gravity=tuple(scene["gravity_m_s2"]),
                              stable_step_guard=1.0, strict=False, step_material=step_material, substeps=substeps,
                              **OPTIONS)


def group_motion(scene, s, amp=1.0):
    """{group: motion} at unit parameter s (scaled by amp) for LiveRubberMaterial.advance."""
    from scipy.spatial.transform import Rotation
    if scene["case"] == "A":
        g = scene["grip0"]
        return {0: {"rotation": Rotation.from_rotvec(amp*s*np.asarray(g["rotation_rotvec_rad"])).as_matrix(),
                    "translation": amp*s*np.asarray(g["translation_m"], float)}}
    return {1: {"translation": amp*s*np.asarray(scene["grip1"]["translation_m_per_unit_s"], float)}}


def run_single(job):
    """RUBBER-GRIP-01/02/03: one scene x profile x CFL in the single-env reference."""
    cfg, _ = load_protocol()
    scene = cfg["scenes"][job["scene"]]
    sched, hold_n = schedule(scene, job["profile"], job["smoke"])
    base = make_live(scene)
    substeps = base.substeps*job["substep_factor"]
    live = base if job["substep_factor"] == 1 else make_live(scene, substeps=substeps)
    s = live.solver
    rows = []
    t0 = time.perf_counter()
    for k, sv in enumerate(sched):
        live.advance(group_motion(scene, sv))
        d = live.last_diagnostics
        rows.append({"step": k+1, "s": sv, "elastic": d["elastic"], "kinetic_particle": d["kinetic_particle"],
                     "kinetic_grid": d["kinetic_grid"], "gravity": d["gravity"], "work": float(s.work.numpy()[0]),
                     "work_end_velocity": float(s.work_end_velocity.numpy()[0]), "cfl_eff_max": d["cfl_eff_max"],
                     "J_min": d["J_min"], "J_max": d["J_max"], "invalid_reason": int(live.invalid_reason)})
    wall = time.perf_counter()-t0
    h = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    mech = h["elastic"]+h["kinetic_particle"]+h["gravity"]
    S = float(np.max(h["elastic"]+h["kinetic_particle"]))
    r = (mech-h["work"])/S
    r_v1 = (mech-h["work_end_velocity"])/S
    hold = slice(len(rows)-hold_n, len(rows))
    out = {**job, "particles": live.scene.n, "substeps": live.substeps, "dt_s": live.dt,
           "cfl": live.dt*s.material.c_p()/s.dx, "control_steps": len(rows), "S_J": S,
           "M": float(np.max(np.abs(r))), "H": float(np.mean(r[hold])), "end_residual": float(r[-1]),
           "M_v1": float(np.max(np.abs(r_v1))), "H_v1": float(np.mean(r_v1[hold])), "end_residual_v1": float(r_v1[-1]),
           "invalid_reason_final": int(live.invalid_reason), "invalid_any": int(h["invalid_reason"].max()),
           "cfl_eff_max": float(h["cfl_eff_max"].max()), "J_range": [float(h["J_min"].min()), float(h["J_max"].max())],
           "wall_s": wall, "solver_receipt_options": s.solver_receipt()["options"], "series": rows}
    return out


def _same_bytes(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


OPTION_NAMES = ("work_end_velocity", "work_scratch", "F64", "x64", "proj_row_node", "proj_row_comp", "proj_row_group",
                "proj_row_block", "proj_row_local", "proj_rhs", "proj_vpresc", "proj_mu", "proj_block_offset",
                "proj_block_size", "proj_inv_offset", "proj_inv", "proj_mu_node")


def compare_env(batch, e, live):
    """First byte-level mismatch (name) of env e against its single reference, or None (tests' L0 definition)."""
    from robotic_cutting_sim.batched.rubber_layout import GROUP_ARRAYS, NODE_ARRAYS, PARTICLE_ARRAYS, TRACTION_ARRAYS
    snap = batch.solver.snapshot(e)
    s = live.solver
    G, Gt = s.G, s.Gt
    names = PARTICLE_ARRAYS+NODE_ARRAYS+GROUP_ARRAYS+TRACTION_ARRAYS+("cell_node", "cell_offsets", "cell_particles",
                                                                       "index", "work", "violations", "cfl_violations")
    option_names = [n for n in OPTION_NAMES if hasattr(s, n)]
    if set(option_names) != {n for n in OPTION_NAMES if n in snap}:
        return "option array set"
    for name in names+tuple(option_names):
        a = getattr(s, name).numpy()
        if name in GROUP_ARRAYS:
            a = a[:G]
        elif name in TRACTION_ARRAYS:
            a = a[:Gt]
        elif name.startswith("proj_row") or name in ("proj_rhs", "proj_vpresc", "proj_mu"):
            a = a[:s.proj_rows]
        if not _same_bytes(snap[name], a):
            return name
    if not _same_bytes(batch.summary.numpy()[e], s.diagnostics.summary.numpy()[0]):
        return "diagnostics"
    K = batch.K
    if not _same_bytes(batch.marker_x.numpy()[e*K:(e+1)*K], s.diagnostics.marker_x.numpy()):
        return "markers"
    if not _same_bytes(batch.marker_v.numpy()[e*K:(e+1)*K], s.diagnostics.marker_v.numpy()):
        return "marker velocities"
    if not _same_bytes(batch.wrench.numpy()[e, :G], live.last_wrench):
        return "wrench"
    return None


def run_l0(job):
    """RUBBER-GRIP-04: N = 2 batch vs two single references, byte-wise after every control step."""
    from robotic_cutting_sim.batched import rubber_material_batch as rmb
    from robotic_cutting_sim.rubber_material import MaterialValues
    cfg, _ = load_protocol()
    scene = cfg["scenes"][job["scene"]]
    v = scene["l0_env1"]
    env1 = dict(L=v["L_m"], mu=scene["mu_pa"]*v["mu_scale"], rho=v["rho_kg_m3"])
    step_mat = MaterialValues(mu0=env1["mu"], kappa=20.0*env1["mu"], density=env1["rho"])
    kws = [dict(), env1]
    amps = [1.0, v["amplitude_scale"]]
    singles = [make_live(scene, step_material=step_mat, **kw) for kw in kws]
    batch = rmb.BatchedRubberMaterial([make_live(scene, step_material=step_mat, **kw) for kw in kws], device="cpu",
                                      parity_mode=True, strict=False)
    sched, _ = schedule(scene, "smootherstep", job["smoke"])
    G = batch.G
    first = None
    t0 = time.perf_counter()
    for k, sv in enumerate(sched):
        tr = np.zeros((2, G, 3))
        R = np.tile(np.eye(3), (2, G, 1, 1))
        for e in range(2):
            for g, m in group_motion(scene, sv, amps[e]).items():
                tr[e, g] = m["translation"]
                if "rotation" in m:
                    R[e, g] = m["rotation"]
        batch.advance(translation=tr, rotation=R if scene["case"] == "A" else None)
        for e, live in enumerate(singles):
            live.advance(group_motion(scene, sv, amps[e]))
            bad = compare_env(batch, e, live)
            if bad is not None and first is None:
                first = {"control_step": k+1, "env": e, "array": bad}
        if first is not None:
            break
    return {**job, "N": 2, "particles": batch.packed.particle_counts.tolist(), "substeps": batch.solver.substeps,
            "control_steps_compared": k+1, "control_steps_scheduled": len(sched), "first_mismatch": first,
            "byte_identical_every_step": first is None and k+1 == len(sched),
            "invalid_reason": batch.invalid_reason.numpy().tolist(),
            "options": batch.receipt()["solver"]["options"], "wall_s": time.perf_counter()-t0}


def run_job(job):
    import warp as wp
    wp.init()
    return run_l0(job) if job["kind"] == "l0" else run_single(job)


def evaluate(cfg, singles, l0s):
    T = cfg["tests"]
    t1, t2 = T["RUBBER-GRIP-01"]["thresholds"], T["RUBBER-GRIP-02"]["thresholds"]
    by = {(r["scene"], r["profile"], r["substep_factor"]): r for r in singles}
    rows1, rows2 = [], []
    for sc in cfg["scenes"]:
        for prof in cfg["common"]["profiles"]:
            a, b = by[(sc, prof, 1)], by[(sc, prof, 2)]
            ok1 = a["M"] <= t1["m_max"] and abs(a["H"]) <= t1["h_max"]
            rows1.append({"scene": sc, "profile": prof, "M": a["M"], "H": a["H"], "M_v1": a["M_v1"], "H_v1": a["H_v1"],
                          "end_residual_v1": a["end_residual_v1"], "pass": ok1})
            ratio = b["M"]/a["M"] if a["M"] > 0 else float("inf")
            ok2 = b["M"] <= t2["ratio_max"]*a["M"] or b["M"] <= t2["floor"]
            ratio_v1 = b["M_v1"]/a["M_v1"] if a["M_v1"] > 0 else float("inf")
            rows2.append({"scene": sc, "profile": prof, "M_cfl05": a["M"], "M_cfl025": b["M"], "ratio": ratio,
                          "ratio_v1_reported": ratio_v1, "pass": ok2})
    v3 = [{"scene": r["scene"], "profile": r["profile"], "substep_factor": r["substep_factor"],
           "invalid_any": r["invalid_any"]} for r in singles] + [{"scene": r["scene"], "l0": True,
                                                                   "invalid": r["invalid_reason"]} for r in l0s]
    ok3 = all(r["invalid_any"] == 0 for r in singles) and all(max(r["invalid_reason"]) == 0 for r in l0s)
    rows4 = [{"scene": r["scene"], "byte_identical_every_step": r["byte_identical_every_step"],
              "first_mismatch": r["first_mismatch"], "control_steps": r["control_steps_compared"]} for r in l0s]
    ok4 = all(r["byte_identical_every_step"] for r in l0s)
    verdict = lambda ok: "PASS" if ok else "FAIL"            # noqa: E731
    return {"RUBBER-GRIP-01": {"verdict": verdict(all(r["pass"] for r in rows1)), "rows": rows1},
            "RUBBER-GRIP-02": {"verdict": verdict(all(r["pass"] for r in rows2)), "rows": rows2},
            "RUBBER-GRIP-03": {"verdict": verdict(ok3), "rows": v3},
            "RUBBER-GRIP-04": {"verdict": verdict(ok4), "rows": rows4}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    (out/"raw").mkdir(parents=True)
    cfg, sha = load_protocol()
    sh = lambda *c: subprocess.run(c, cwd=ROOT, capture_output=True, text=True).stdout.strip()     # noqa: E731
    import warp as wp
    meta = {"protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha, "command": " ".join(sys.argv),
            "commit": sh("git", "rev-parse", "HEAD"),
            "dirty_paths": sh("git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests").splitlines(),
            "smoke": args.smoke, "label": "SMOKE (runner check; not a result)" if args.smoke else "FORMAL",
            "python": platform.python_version(), "warp": wp.config.version, "numpy": np.__version__, "device": "cpu",
            "cpu": platform.processor() or platform.machine(), "workers": args.workers,
            "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED", "start_local": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    meta["dirty_tree"] = bool(meta["dirty_paths"])
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    jobs = [{"kind": "single", "scene": sc, "profile": p, "substep_factor": f, "smoke": args.smoke}
            for sc in cfg["scenes"] for p in cfg["common"]["profiles"] for f in (1, 2)]
    jobs += [{"kind": "l0", "scene": sc, "smoke": args.smoke} for sc in cfg["scenes"]]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, jobs))
    singles = [r for r in results if r["kind"] == "single"]
    l0s = [r for r in results if r["kind"] == "l0"]
    for r in singles:
        (out/"raw"/f"{r['scene']}_{r['profile']}_x{r['substep_factor']}.json").write_text(json.dumps(r, indent=1))
    for r in l0s:
        (out/"raw"/f"{r['scene']}_l0.json").write_text(json.dumps(r, indent=1))
    verdicts = evaluate(cfg, singles, l0s)
    summary = {"protocol_sha256": sha, "label": meta["label"], "wall_s": time.perf_counter()-t0,
               "runs": [{k: r[k] for k in r if k != "series"} for r in singles], "l0": l0s, "verdicts": verdicts}
    (out/"summary.json").write_text(json.dumps(summary, indent=1))
    meta["end_local"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v["verdict"] for k, v in verdicts.items()}, indent=1))
    for k, v in verdicts.items():
        for row in v["rows"]:
            print(k, json.dumps(row))


if __name__ == "__main__":
    main()
