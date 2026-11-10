"""F4 / R2.1: batched L0 (CPU bitwise) runs of RUBBER-PHYS-07 v3 (ramped, instant) and RUBBER-PHYS-12 (protocol v3).

    TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_2/run_batched_l0_v3.py \
        --case 07_ramped|07_instant|12 [--copies 2] --out artifacts/rl_ready/p2_2/<id>

The single-env reference (``LiveRubberMaterial``, exactly the v3 case inputs: rubber_phys._strip_07, the cosine path,
the hold-force traction ramp or the instantaneous release, the guard) and a batched ``BatchedRubberMaterial`` of
``--copies`` copies of the same env (Warp CPU, parity_mode) receive the same control-step inputs. After EVERY control
step every particle / node / group / traction array, the work and violation counters, the segmented diagnostics, the
markers, the wrench and the invalid bits of every batched env must equal the single reference bitwise. The first
mismatch (step, env, array) is recorded; the run continues to the end so the batched outcome is also reported.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def schedule(case, cfg, P, T_ax):
    """Per control step: (translation of grip 1 or None, release flag, traction-x or None, phase)."""
    if case.startswith("07"):
        t7 = cfg["tests"]["RUBBER-PHYS-07"]
        var = t7["setup"]["variants"][case.split("_")[1]]
        dtc = float(t7["setup"]["control_interval_s"])
        ramped = var["release"] == "ramped_traction"
        lam_max = float(var["lambda"])
    else:
        t12 = cfg["tests"]["RUBBER-PHYS-12"]["setup"]
        dtc, ramped, lam_max = float(t12["control_interval_s"]), False, float(t12["lambda"])
    steps = lambda periods: int(math.ceil(periods*T_ax/dtc-1e-9))                      # noqa: E731
    n_r, n_h, n_post = steps(60), steps(10), steps(20)
    n_rel = steps(60) if ramped else 0
    out = []
    for k in range(n_r+n_h):
        out.append(("stretch", P.cosine_path((k+1)*dtc, n_r*dtc, lam_max), False, None))
    for k in range(n_rel):
        out.append(("release_ramp", None, k == 0, (1+math.cos(math.pi*(k+0.5)*dtc/(n_rel*dtc)))/2))
    for k in range(n_post):
        out.append(("post", None, (not ramped) and k == 0, 0.0 if ramped else None))
    return dtc, ramped, out


def compare(batch, e, live, names):
    snap = batch.solver.snapshot(e)
    s = live.solver
    G, Gt = s.G, s.Gt
    for name in names["particle"]+names["node"]+("cell_node", "cell_offsets", "cell_particles", "index"):
        if not np.array_equal(snap[name], getattr(s, name).numpy()):
            return name
    for name in names["group"]:
        if not np.array_equal(snap[name], getattr(s, name).numpy()[:G]):
            return name
    for name in names["traction"]:
        if not np.array_equal(snap[name], getattr(s, name).numpy()[:Gt]):
            return name
    for name in ("work", "violations", "cfl_violations"):
        if not np.array_equal(snap[name], getattr(s, name).numpy()):
            return name
    if not np.array_equal(batch.summary.numpy()[e], s.diagnostics.summary.numpy()[0]):
        return "diagnostics"
    K = batch.K
    if not np.array_equal(batch.marker_x.numpy()[e*K:(e+1)*K], s.diagnostics.marker_x.numpy()):
        return "markers"
    if not np.array_equal(batch.wrench.numpy()[e, :G], live.last_wrench):
        return "wrench"
    if int(batch.invalid_reason.numpy()[e]) != int(live.invalid_reason):
        return "invalid_reason"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=("07_ramped", "07_instant", "12"))
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT/"src"))
    import subprocess
    from robotic_cutting_sim import rubber_phys as P
    from robotic_cutting_sim.batched import rubber_layout as RL
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    cfg = P.load_config(P.CONFIG_V3)
    names = {"particle": RL.PARTICLE_ARRAYS, "node": RL.NODE_ARRAYS, "group": RL.GROUP_ARRAYS,
             "traction": RL.TRACTION_ARRAYS}
    ramped_case = args.case == "07_ramped"
    if args.case.startswith("07"):
        dtc0 = float(cfg["tests"]["RUBBER-PHYS-07"]["setup"]["control_interval_s"])
    else:
        dtc0 = float(cfg["tests"]["RUBBER-PHYS-12"]["setup"]["control_interval_s"])
    mk = lambda: P._strip_07(cfg, "cpu", dtc0, P.guard_limit(cfg), traction_end=ramped_case)[0]   # noqa: E731
    live = mk()
    batch = BatchedRubberMaterial([mk() for _ in range(args.copies)], device="cpu", parity_mode=True, strict=False)
    st = cfg["tests"]["RUBBER-PHYS-06"]["setup"]
    L = st["L_m"]
    T_ax = P.axial_period(L, live.material)
    dtc, ramped, sched = schedule(args.case, cfg, P, T_ax)
    N, G = args.copies, live.solver.G
    first_mismatch, hold_force, steps = None, None, 0
    t0 = time.perf_counter()
    for phase, lam, release, frac in sched:
        motions, traction, b_kw = None, None, {}
        if lam is not None:
            tr = np.zeros((N, G, 3))
            tr[:, 1, 0] = (lam-1)*L
            motions = {1: {"translation": tr[0, 1]}}
            b_kw["translation"] = tr
        if release:
            motions = {1: {"active": False}}
            act = np.ones((N, G), np.int32)
            act[:, 1] = 0
            b_kw["active"] = act
        if frac is not None:
            f = hold_force*frac if frac else 0.0
            traction = {0: [f, 0.0, 0.0]}
            b_kw["traction"] = np.tile([[f, 0.0, 0.0]], (N, 1))
        w = live.advance(motions, traction=traction)
        batch.advance(**b_kw)
        steps += 1
        if phase == "stretch":
            hold_force = float(-np.asarray(w)[1, 0])
        if first_mismatch is None:
            for e in range(N):
                bad = compare(batch, e, live, names)
                if bad is not None:
                    first_mismatch = {"step": steps, "env": e, "array": bad, "phase": phase}
                    break
    res = {"label": "F4 batched L0 (CPU bitwise) of a v3 case", "case": args.case, "protocol": "rubber_phys_tests_v3",
           "copies": N, "control_steps": steps, "control_interval_s": dtc, "substeps": live.substeps, "dt_s": live.dt,
           "particles_per_env": live.scene.n, "stable_step_guard": P.guard_limit(cfg),
           "bitwise_equal_every_step": first_mismatch is None, "first_mismatch": first_mismatch,
           "single_invalid": int(live.invalid_reason), "batched_invalid": batch.invalid_reason.numpy().tolist(),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "wall_s": time.perf_counter()-t0}
    (out/"l0.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
