"""RUBBER-VISC-08b: Prony tangent check of ``stable_step_guard_visco_v1`` (ADR 0023 §21.2.6; gate before any dynamic
visco run).

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/visco_tangent_check.py \
        --out artifacts/rl_ready/p2_3/<dir> [--per-group 400] [--directions 512] [--seed 2311]

States: F_final = R1 diag(lambda) (I + gamma e_a (x) e_b) R2 with principal stretches lambda_i in [0.6, 1.6], |gamma|
<= 1 on a random off-diagonal, random rotations (det F in [0.216, 4.1] inside the j domain). Q_i come from random
load-unload histories integrated with the exponential update at the batch dt (P2 and SLS sets; dt, dt/2, dt/4 of the
protocol's strip/block step): I -> F_peak (fast ramp of n1 batch-dt steps) -> hold H1 -> F_final (fast ramp of n2
steps) -> hold H2, with F_peak drawn like F_final, n1, n2 in [1, 300], H1 in [0, 3 tau_max], H2 in [0, tau_min]. A hold
at constant S0 is applied in closed form (Q <- exp(-H / tau) Q), identical to H / dt exponential updates with zero
increment. The stress-pass map of the next substep, with Q_n and S0_prev frozen,

    P(F) = P_NH(F, J(F); mu0, kappa) + F sum_i (a_i Q_i^n + b_i (S0(F) - S0_prev) - g_i S0(F))

is differentiated by float64 central differences (full 9 x 9 tangent D = dP/dF, step 1e-6 |F|); the exact
algorithmic acoustic tensor A(N)_jk = sum_LM D_jLkM N_L N_M is evaluated for 512 unit directions (3 axes + 509
random) and its largest symmetric-part eigenvalue compared with rho c^2 of the guard (the Warp float32 function
``rubber_visco_constitutive.cfl_effective_visco`` as implemented, and its float64 twin) at the base state
(Q^{n+1} = a Q^n, S0 = S0(F_final)). Criterion (protocol): ratio <= 1 for every sample (>= 2000 samples). Reported:
max ratio, distribution, worst samples, and the premise (min eigenvalue of the isochoric material part
M0(N)_jk = (F dS0[e_k (x) N] N)_j). The output directory must not exist.
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from robotic_cutting_sim import rubber_visco_constitutive as rvc  # noqa: E402

PROTOCOL = ROOT/"configs"/"rubber"/"rubber_visc_tests_v1.json"
MU0, KAPPA, RHO, DX = 4.0e5, 8.0e6, 1150.0, 1.0e-3
SETS = {"P2": ((0.2, 0.005), (0.15, 0.05)), "SLS": ((0.3, 0.01),)}


def batch_dt():
    """dt of the protocol's strip / block runs: control_dt 2e-4 s, dx 1 mm, CFL 0.5, c_p at mu0 (step rule)."""
    cp = math.sqrt((KAPPA+4*MU0/3)/RHO)
    sub = math.ceil(2e-4/(0.5*DX/cp)*(1-1e-12))
    return 2e-4/sub


def random_F(rng, n):
    out = np.empty((n, 3, 3))
    for k in range(n):
        lam = rng.uniform(0.6, 1.6, 3)
        S = np.eye(3)
        a, b = rng.choice(3, 2, replace=False)
        S[a, b] = rng.uniform(-1.0, 1.0)
        out[k] = (Rotation.random(random_state=rng).as_matrix()@np.diag(lam)@S
                  @ Rotation.random(random_state=rng).as_matrix())
    return out


def history(rng, terms, dt, n):
    """Vectorised load-unload histories; returns F_final [n,3,3], Q_n [n,M,3,3], S0_prev [n,3,3], receipts."""
    coeffs = rvc.prony_coefficients(terms, dt)
    taus = [t for _, t in terms]
    F_peak, F_final = random_F(rng, n), random_F(rng, n)
    n1, n2 = rng.integers(1, 301, n), rng.integers(1, 301, n)
    H1 = rng.uniform(0.0, 3*max(taus), n)
    H2 = rng.uniform(0.0, min(taus), n)
    M = len(terms)
    q = np.zeros((n, M, 3, 3))
    sp = np.zeros((n, 3, 3))
    eye = np.eye(3)

    def ramp(F0, F1, steps):
        nonlocal q, sp
        for k in range(1, int(steps.max())+1):
            s = np.minimum(k/steps, 1.0)[:, None, None]
            s0 = rvc.s0_np(F0+s*(F1-F0), MU0)
            for i, (a, b) in enumerate(coeffs):
                q[:, i] = a*q[:, i]+b*(s0-sp)
            sp = s0

    def hold(H):
        for i, tau in enumerate(taus):
            q[:, i] *= np.exp(-H/tau)[:, None, None]

    ramp(np.broadcast_to(eye, (n, 3, 3)), F_peak, n1)
    hold(H1)
    ramp(F_peak, F_final, n2)
    hold(H2)
    return F_final, q, sp, {"n1": n1, "n2": n2, "H1": H1, "H2": H2}


def stress_map(F, q, sp, terms, coeffs):
    """Discrete stress-pass map with Q_n, S0_prev frozen (float64; Jbar = J(F)); F may carry leading batch dims."""
    s0 = rvc.s0_np(F, MU0)
    over = np.zeros_like(F)
    for i, ((g, _), (a, b)) in enumerate(zip(terms, coeffs)):
        if g != 0:
            over = over+(a*q[i]+b*(s0-sp))-g*s0
    return rvc.nh_pk1_np(F, np.linalg.det(F), MU0, KAPPA)+F@over


def tangent(F, q, sp, terms, coeffs, fn=None):
    """Central-difference dP/dF as D[j, L, k, M] (float64)."""
    h = 1e-6*max(1.0, np.abs(F).max())
    E = np.zeros((9, 3, 3))
    E.reshape(9, 9)[np.arange(9), np.arange(9)] = h
    Fp, Fm = F[None]+E, F[None]-E
    fn = fn or (lambda X: stress_map(X, q, sp, terms, coeffs))
    D = (fn(Fp)-fn(Fm))/(2*h)                     # [kM, j, L]
    return np.transpose(D.reshape(3, 3, 3, 3), (2, 3, 0, 1))


def acoustic_max(D, Ns):
    A = np.einsum("jlkm,nl,nm->njk", D, Ns, Ns)
    A = 0.5*(A+np.swapaxes(A, 1, 2))
    return np.linalg.eigvalsh(A)[:, -1].max(), A


def material_part_min(F, Ns):
    """min over directions of the smallest eigenvalue of M0(N)_jk = (F dS0[e_k (x) N] N)_j (isochoric material part)."""
    D = tangent(F, None, None, None, None, fn=lambda X: F[None]@rvc.s0_np(X, MU0))
    A = np.einsum("jlkm,nl,nm->njk", D, Ns, Ns)
    A = 0.5*(A+np.swapaxes(A, 1, 2))
    return float(np.linalg.eigvalsh(A)[:, 0].min())


def directions(rng, count):
    Ns = rng.normal(size=(count-3, 3))
    return np.vstack([np.eye(3), Ns/np.linalg.norm(Ns, axis=1, keepdims=True)])


def run(per_group=400, n_dirs=512, seed=2311, dts=None):
    rng = np.random.default_rng(seed)
    Ns = directions(rng, n_dirs)
    dt0 = batch_dt()
    dts = dts or {"dt": dt0, "dt/2": dt0/2, "dt/4": dt0/4}
    rows, groups = [], []
    for set_name, terms in SETS.items():
        for dt_name, dt in dts.items():
            coeffs = rvc.prony_coefficients(terms, dt)
            row64 = rvc.params64_row(terms, dt)
            F, q, sp, rec = history(rng, terms, dt, per_group)
            q_base = np.stack([np.stack([a*q[k, i] for i, (a, _) in enumerate(coeffs)]) for k in range(per_group)])
            s0 = rvc.s0_np(F, MU0)
            J = np.linalg.det(F)
            F32 = F.astype(np.float32)
            cfl = rvc.evaluate_visco_cfl(F32, np.linalg.det(F32.astype(np.float64)), MU0, KAPPA, RHO, 1.0,
                                         row64, rvc.sym6(q_base), rvc.sym6(s0))
            bound_warp = RHO*cfl.astype(np.float64)**2           # rho c^2 of the float32 guard (dt / dx = 1)
            ratios = []
            for k in range(per_group):
                D = tangent(F[k], q[k], sp[k], terms, coeffs)
                lam, _ = acoustic_max(D, Ns)
                bound64 = rvc.visco_wave_modulus_bound_np(F[k], J[k], MU0, KAPPA, terms, coeffs, q_base[k], s0[k])
                over = sum(q_base[k, i]-g*s0[k] for i, (g, _) in enumerate(terms))
                r = {"set": set_name, "dt": dt_name, "k": k, "lambda_max_pa": float(lam),
                     "bound_warp_pa": float(bound_warp[k]), "bound_f64_pa": float(bound64),
                     "ratio": float(lam/bound_warp[k]), "ratio_f64": float(lam/bound64), "J": float(J[k]),
                     "overstress_gershgorin_pa": float(rvc.gershgorin_max_np(over)),
                     "overstress_norm_over_mu0": float(np.linalg.norm(over)/MU0),
                     "n1": int(rec["n1"][k]), "n2": int(rec["n2"][k]), "H1_s": float(rec["H1"][k]),
                     "H2_s": float(rec["H2"][k])}
                ratios.append(r["ratio"])
                rows.append(r)
            groups.append({"set": set_name, "dt_name": dt_name, "dt_s": dt, "samples": per_group,
                           "max_ratio": float(max(ratios)), "median_ratio": float(np.median(ratios)),
                           "min_ratio": float(min(ratios))})
    # premise: isochoric material part (sampled on the first 200 states of the first group: it does not depend on Q)
    premise = [material_part_min(F, Ns) for F in random_F(np.random.default_rng(seed+1), 200)]
    return rows, groups, premise, Ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-group", type=int, default=400)
    ap.add_argument("--directions", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2311)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    start = time.time()
    rows, groups, premise, Ns = run(args.per_group, args.directions, args.seed)
    ratios = np.array([r["ratio"] for r in rows])
    worst = sorted(rows, key=lambda r: -r["ratio"])[:10]
    cfg = json.loads(PROTOCOL.read_text())
    thr = cfg["tests"]["RUBBER-VISC-08"]["thresholds"]["ratio_max"]
    verdict = "PASS" if len(rows) >= 2000 and float(ratios.max()) <= thr else "FAIL"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    summary = {"test": "RUBBER-VISC-08b", "label": "FORMAL (protocol rubber_visc_tests_v1)",
               "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), "commit": commit,
               "dirty_tree": dirty, "command": " ".join(sys.argv), "seed": args.seed,
               "samples": len(rows), "directions": len(Ns), "threshold_ratio_max": thr,
               "max_ratio": float(ratios.max()), "max_ratio_f64_twin": float(max(r["ratio_f64"] for r in rows)),
               "median_ratio": float(np.median(ratios)), "p99_ratio": float(np.quantile(ratios, 0.99)),
               "groups": groups, "worst": worst,
               "premise_isochoric_material_part": {"states": len(premise), "min_eigenvalue_pa": float(min(premise)),
                                                   "psd": bool(min(premise) >= -1e-6*MU0),
                                                   "note": "reported; the criterion is the ratio"},
               "batch_dt_s": batch_dt(), "material": {"mu0_pa": MU0, "kappa_pa": KAPPA, "rho": RHO},
               "verdict": verdict, "wall_s": time.time()-start,
               "versions": {"python": platform.python_version(), "numpy": np.__version__}}
    out.mkdir(parents=True)
    (out/"summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out/"rows.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("set", "dt")},
                        set=np.array([r["set"] for r in rows]), dt=np.array([r["dt"] for r in rows]))
    print(json.dumps({k: summary[k] for k in ("samples", "max_ratio", "max_ratio_f64_twin", "median_ratio",
                                              "p99_ratio", "verdict", "wall_s")}))
    print(json.dumps(summary["premise_isochoric_material_part"]))
    for g in groups:
        print(g)


if __name__ == "__main__":
    main()
