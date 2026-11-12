"""DIAGNOSTIC (not a protocol result): layer-N energy books of the RUBBER-VISC-04b cantilever tap for Prony variants
(elastic g = 0, P2, tau 5 ms only, tau 50 ms only; substep factor 1 / 2), to locate the secular drift of
B_N = E_kin + Psi_eq + W_b - W_ext seen in ``visc_v1_001`` (ADR 0023 §21.6). Also books a DIAGNOSTIC kick-paired branch
work W_kp = sum_n V pb_n : (dF_{n-1} + dF_n) / 2 (the quadrature the kick's KE change implies) next to the declared
left-point W_b of ``branch_work_v1``.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/diag_visc_cantilever_books.py \
        --out artifacts/rl_ready/p2_3/<dir> [--record 0.25]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

VARIANTS = {"g0_x1": (((0.0, 0.005), (0.0, 0.05)), 1), "P2_x1": (((0.2, 0.005), (0.15, 0.05)), 1),
            "tau5_x1": (((0.2, 0.005), (0.0, 0.05)), 1), "tau50_x1": (((0.0, 0.005), (0.15, 0.05)), 1),
            "tau5_x2": (((0.2, 0.005), (0.0, 0.05)), 2)}


@wp.kernel
def _kick_paired_branch_work(pb: wp.array(dtype=wp.mat33d), nodes: wp.array2d(dtype=int),
                             gradients: wp.array2d(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
                             volume: wp.float64, dt: wp.float64, dF_prev: wp.array(dtype=wp.mat33d),
                             work: wp.array(dtype=wp.float64)):
    """DIAGNOSTIC: W_kp += V pb_n : (dF_{n-1} + dF_n) / 2 (the branch force paired with the velocity average the kick
    uses; telescopes against the KE change), dF_n = dt sum_I v_I (x) grad N_I."""
    p = wp.tid()
    grad = wp.mat33d()
    for c in range(8):
        grad += wp.outer(wp.vec3d(velocity[nodes[p, c]]), wp.vec3d(gradients[p, c]))
    dF = dt*grad
    wp.atomic_add(work, 0, wp.float64(0.5)*volume*wp.ddot(pb[p], dF+dF_prev[p]))
    dF_prev[p] = dF


def _attach_kick_paired(solver):
    from robotic_cutting_sim.tlmpm_rubber_visco_v1 import TLMPMRubberVisco

    class Probe(TLMPMRubberVisco):
        def step(self):
            super().step()
            wp.launch(_kick_paired_branch_work, self.n, [self.visco_pb, self.nodes, self.gradients, self.new_v,
                                                         wp.float64(self.volume), wp.float64(self.dt), self.kp_dF,
                                                         self.kp_work], device=self.device)
    solver.kp_dF = wp.zeros(solver.n, dtype=wp.mat33d, device=solver.device)
    solver.kp_work = wp.zeros(1, dtype=wp.float64, device=solver.device)
    solver.__class__ = Probe


def run(args):
    name, record = args
    import math
    import run_visc_v1 as R
    from robotic_cutting_sim import rubber_visco_protocol as vp
    R._setup()
    terms, factor = VARIANTS[name]
    mat = vp.material(4e5, terms)
    spec = vp.strip_spec(L=0.04, W=0.004, t=0.004, n_t=4, case="A")
    base = R._base_substeps(spec, mat, 1e-3)
    live = vp.make_live(spec, mat, control_dt=1e-3, substeps=base*factor)
    _attach_kick_paired(live.solver)
    rows = []
    t0 = time.time()
    for k in range(1, int(round(record/1e-3))+1):
        tk = k*1e-3
        z = 2e-4*math.sin(math.pi*tk/0.043)**2 if tk < 0.043 else 0.0
        live.advance({0: {"translation": np.array([0.0, 0.0, z])}})
        rows.append([tk]+R._energies(live)+[float(live.solver.kp_work.numpy()[0])])
    a = np.array(rows)
    s = {c: a[:, i+1] for i, c in enumerate(R.ENERGY_COLUMNS+("kick_paired_branch_work",))}
    BK = s["kinetic"]+s["psi_eq"]+s["gravity"]+s["kick_paired_branch_work"]-s["work_ext"]
    BN = s["kinetic"]+s["psi_eq"]+s["gravity"]+s["branch_work"]-s["work_ext"]
    SN = float(np.max(s["kinetic"]+s["psi_eq"])+np.max(np.abs(s["branch_work"])))
    i_stop = int(np.argmax(a[:, 0] >= 0.043))
    return {"variant": name, "terms": terms, "factor": factor, "substeps": live.substeps, "S_N": SN,
            "BN_max_abs_rel": float(np.abs(BN).max()/SN), "BN_end_rel": float(BN[-1]/SN),
            "BN_at_grip_stop_rel": float(BN[i_stop]/SN),
            "drift_after_stop_rel_per_s": float(np.polyfit(a[i_stop:, 0], BN[i_stop:]/SN, 1)[0]),
            "D_minus_Wb_end_rel": float((s["dissipation"][-1]-s["branch_work"][-1])/SN),
            "psi_v_end_rel": float(s["psi_visc_stored"][-1]/SN), "invalid": int(live.invalid_reason),
            "kick_paired_BN_max_abs_rel": float(np.abs(BK).max()/SN), "kick_paired_BN_end_rel": float(BK[-1]/SN),
            "kick_paired_drift_after_stop_rel_per_s": float(np.polyfit(a[i_stop:, 0], BK[i_stop:]/SN, 1)[0]),
            "D_minus_kick_paired_Wb_end_rel": float((s["dissipation"][-1]-s["kick_paired_branch_work"][-1])/SN),
            "wall_s": time.time()-t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", type=float, default=0.25)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    with ProcessPoolExecutor(len(VARIANTS)) as pool:
        rows = list(pool.map(run, [(n, args.record) for n in VARIANTS]))
    out.mkdir(parents=True)
    import subprocess
    meta = {"label": "DIAGNOSTIC (not a protocol result)", "command": " ".join(sys.argv),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "rows": rows}
    (out/"summary.json").write_text(json.dumps(meta, indent=1, default=float))
    for r in rows:
        print(json.dumps(r, default=float))


if __name__ == "__main__":
    main()
