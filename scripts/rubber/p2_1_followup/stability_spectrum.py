"""DIAGNOSTIC (P2-1 follow-up, RUBBER-PHYS-07): linear stability of the discrete tlmpm_rubber_v1 operator around
homogeneous deformed states (float64, numpy; no solver code is changed).

Model = exactly the discrete system of tlmpm_rubber_v1 on a PERIODIC block of n^3 reference cells (no free surface,
so every homogeneous F is an equilibrium): 2 particles per cell per axis at fractions 1/4, 3/4, trilinear weights and
the frozen gradient formula, F_p = F0 + sum_c u_c (x) grad N_pc (reference gradients, TLMPM), neo_hookean_jbar_v1
with the cell-mean J-bar (rubber_reference.pk1), nodal force f_I = -sum_p V P_p grad N_pI.

Pure FLIP + MUSL is algebraically symplectic Euler for (u, g) with g = MUSL grid velocity and inverse mass
B = M_l^-1 M_c M_l^-1 (M_l lumped, M_c = W M_p W^T): g+ = g + dt B f(u), u+ = u + dt g+. Its linear stability limit is
dt_crit = 2 / sqrt(max eig(B K)), K = -df/du (central differences, h = 1e-7 dx). The particle kinetic energy equals
1/2 g^T M_l M_c^-1 M_l g. Also reported: dt_crit with the lumped inverse mass M_l^-1 (FEM equivalent, no MUSL), the
smallest non-translational eigenvalue of K (negative = static instability of the discretisation, e.g. J-bar
hourglassing), the symmetry error of K (energy consistency) and the per-particle acoustic bound used by the guard.

Units: dx = 1, rho = 1, mu = 1, kappa = beta mu. CFL = dt c_p / dx, c_p = sqrt((kappa + 4 mu / 3) / rho).

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_1_followup/stability_spectrum.py --out <dir>
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))
from robotic_cutting_sim import rubber_reference as R  # noqa: E402


def periodic_block(n):
    s = 0.5
    cells = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 3)
    sub = np.stack(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij"), -1).reshape(-1, 3)
    cell_of, frac = [], []
    for c, cc in enumerate(cells):
        for q in sub:
            cell_of.append(c)
            frac.append((q+0.5)/2)
    cell_of, frac = np.array(cell_of), np.array(frac)
    pc = cells[cell_of]
    P = len(pc)
    nodes = np.empty((P, 8), int)
    w = np.empty((P, 8))
    gr = np.empty((P, 8, 3))
    for k in range(8):
        bit = np.array([(k >> j) & 1 for j in range(3)])
        xyz = (pc+bit) % n
        nodes[:, k] = (xyz[:, 0]*n+xyz[:, 1])*n+xyz[:, 2]
        f = np.where(bit, frac, 1-frac)
        w[:, k] = f.prod(1)
        for a in range(3):
            gr[:, k, a] = (2*bit[a]-1)*f[:, (a+1) % 3]*f[:, (a+2) % 3]
    return dict(n=n, P=P, nodes=nodes, w=w, gr=gr, cell_of=cell_of, ncell=len(cells), V=s**3, m=s**3, N=n**3)


def pk1_batch(F, Jbar, mu, kappa):
    J = np.linalg.det(F)
    c0 = np.cross(F[:, :, 1], F[:, :, 2])
    c1 = np.cross(F[:, :, 2], F[:, :, 0])
    c2 = np.cross(F[:, :, 0], F[:, :, 1])
    cof = np.stack([c0, c1, c2], axis=2)
    I1 = np.einsum("pij,pij->p", F, F)
    a = mu*J**(-2/3)
    scale = kappa/2*(Jbar-1/Jbar)-a*I1/(3*J)
    return a[:, None, None]*F+scale[:, None, None]*cof


def force(blk, F0, u, mu, kappa):
    F = F0[None]+np.einsum("pci,pcj->pij", u[blk["nodes"]], blk["gr"])
    J = np.linalg.det(F)
    Jbar = np.bincount(blk["cell_of"], J, blk["ncell"])/np.bincount(blk["cell_of"], minlength=blk["ncell"])
    Pk = pk1_batch(F, Jbar[blk["cell_of"]], mu, kappa)
    contrib = -blk["V"]*np.einsum("pij,pcj->pci", Pk, blk["gr"])
    f = np.zeros((blk["N"], 3))
    np.add.at(f, blk["nodes"].reshape(-1), contrib.reshape(-1, 3))
    return f


def stiffness(blk, F0, mu, kappa, h=1e-7):
    N = blk["N"]
    K = np.zeros((3*N, 3*N))
    u = np.zeros((N, 3))
    for d in range(3*N):
        up, um = u.copy(), u.copy()
        up[d//3, d % 3] += h
        um[d//3, d % 3] -= h
        K[:, d] = -(force(blk, F0, up, mu, kappa)-force(blk, F0, um, mu, kappa)).reshape(-1)/(2*h)
    return K


def mass_matrices(blk):
    N = blk["N"]
    Mc = np.zeros((N, N))
    for k in range(8):
        for l in range(8):
            np.add.at(Mc, (blk["nodes"][:, k], blk["nodes"][:, l]), blk["m"]*blk["w"][:, k]*blk["w"][:, l])
    Ml = Mc.sum(1)
    return Ml, Mc


def acoustic_bound(F, mu, kappa, Jbar=None):
    """Per-particle upper bound of rho c_ref^2 (reference-frame P/S wave modulus, max over directions), used by the
    guard: a + (p'(Jbar) + 5 a I1 / (9 J^2)) (J / s_min)^2 + 4 a / (3 J) s_max J / s_min."""
    J = np.linalg.det(F)
    Jb = J if Jbar is None else Jbar
    s = np.linalg.svd(F, compute_uv=False)
    a = mu*J**(-2/3)
    I1 = np.sum(F*F)
    pp = kappa/2*(1+1/Jb**2)
    g = J/s[-1]
    return a+(pp+5*a*I1/(9*J*J))*g*g+4*a/(3*J)*s[0]*g


def acoustic_exact(F, mu, kappa, dirs=4000, seed=0):
    """max over unit N of the largest eigenvalue of the acoustic tensor Q(N) = d(P N)/dw (homogeneous, Jbar = J)."""
    rng = np.random.default_rng(seed)
    Ns = rng.normal(size=(dirs, 3))
    Ns = np.vstack([np.eye(3), Ns/np.linalg.norm(Ns, axis=1, keepdims=True)])
    best = 0.0
    h = 1e-7
    for Nv in Ns:
        Q = np.zeros((3, 3))
        for i in range(3):
            w = np.zeros(3)
            w[i] = h
            Pp = R.pk1(F+np.outer(w, Nv), mu, kappa)
            Pm = R.pk1(F-np.outer(w, Nv), mu, kappa)
            Q[:, i] = (Pp-Pm)@Nv/(2*h)
        best = max(best, np.linalg.eigvalsh(0.5*(Q+Q.T)).max())
    return best


def analyse(blk, F0, mu, kappa, Ml, Mc, cp):
    K = stiffness(blk, F0, mu, kappa)
    asym = np.abs(K-K.T).max()/np.abs(K).max()
    K = 0.5*(K+K.T)
    Ml3 = np.repeat(Ml, 3)
    Mc3 = np.kron(Mc, np.eye(3))
    Meff = np.diag(Ml3)@np.linalg.solve(Mc3, np.diag(Ml3))
    Meff = 0.5*(Meff+Meff.T)
    ev_musl = sla.eigh(K, Meff, eigvals_only=True)
    ev_lump = sla.eigh(K, np.diag(Ml3), eigvals_only=True)
    evK = np.linalg.eigvalsh(K)
    kscale = np.abs(evK).max()
    nontrans = np.sort(evK)[3:]          # 3 rigid translations (periodic)
    return {"cfl_crit_musl": 2/math.sqrt(ev_musl.max())*cp, "cfl_crit_lumped": 2/math.sqrt(ev_lump.max())*cp,
            "lambda_max_musl": float(ev_musl.max()), "K_min_nontranslational_rel": float(nontrans.min()/kscale),
            "K_min_eig_rel": float(evK.min()/kscale), "K_asymmetry_rel": float(asym), "negative_modes": int(
                np.sum(nontrans < -1e-9*kscale))}


def uniaxial_F(lam, mu, kappa):
    s = R.uniaxial_stress(lam, mu, kappa)["lateral"]
    return np.diag([lam, s, s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--beta", type=float, default=20.0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    t0 = time.perf_counter()
    mu, kappa = 1.0, args.beta
    cp = math.sqrt(kappa+4*mu/3)
    blk = periodic_block(args.n)
    Ml, Mc = mass_matrices(blk)
    rows = []
    rng = np.random.default_rng(1)
    cases = [("identity", np.eye(3))]
    for lam in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.25, 1.5, 2.0, 3.0):
        cases.append((f"uniaxial_stress_{lam}", uniaxial_F(lam, mu, kappa)))
    for lam in (0.7, 0.8, 0.9, 1.1):
        cases.append((f"uniaxial_strain_{lam}", np.diag([lam, 1.0, 1.0])))
    for J in (0.5, 0.8, 1.25):
        cases.append((f"isotropic_J_{J}", np.eye(3)*J**(1/3)))
    for g in (0.5, 1.0, 2.0):
        F = np.eye(3)
        F[0, 1] = g
        cases.append((f"simple_shear_{g}", F))
    for k in range(12):
        Q1, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Q2, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        sv = np.exp(rng.uniform(math.log(0.45), math.log(2.2), 3))
        cases.append((f"random_{k}", Q1@np.diag(sv)@Q2))
    base = None
    for name, F0 in cases:
        r = analyse(blk, F0, mu, kappa, Ml, Mc, cp)
        if base is None:
            base = r
        sv = np.linalg.svd(F0, compute_uv=False)
        qb = acoustic_bound(F0, mu, kappa)
        qe = acoustic_exact(F0, mu, kappa, dirs=1500)
        r.update(name=name, F=F0.tolist(), J=float(np.linalg.det(F0)), stretches=sv.tolist(),
                 r_exact=math.sqrt(r["lambda_max_musl"]/base["lambda_max_musl"]),
                 r_acoustic_exact=math.sqrt(qe/(kappa+4*mu/3)), r_bound=math.sqrt(qb/(kappa+4*mu/3)))
        r["bound_over_exact"] = r["r_bound"]/r["r_exact"]
        rows.append(r)
        print(f"{name:24s} J {r['J']:.3f} cfl_crit musl {r['cfl_crit_musl']:.3f} lumped {r['cfl_crit_lumped']:.3f} "
              f"r_exact {r['r_exact']:.3f} r_acoustic {r['r_acoustic_exact']:.3f} r_bound {r['r_bound']:.3f} "
              f"Kmin {r['K_min_nontranslational_rel']:+.2e} neg {r['negative_modes']} asym {r['K_asymmetry_rel']:.1e}",
              flush=True)
    result = {"label": "DIAGNOSTIC (not a RUBBER-PHYS result)", "n_cells": args.n, "beta": args.beta,
              "particles_per_cell_axis": 2, "boundary": "periodic", "rows": rows, "wall_s": time.perf_counter()-t0}
    (out/"stability_spectrum.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
