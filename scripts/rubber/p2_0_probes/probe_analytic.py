"""Throwaway P2-0 probe: analytic checks for the proposed neo_hookean_jbar_v1 energy.

W(F; Jbar) = mu/2 (J^-2/3 I1 - 3) + kappa/4 (Jbar^2 - 1 - 2 ln Jbar),  Jbar = J for one particle per cell.
PK1 P = mu J^-2/3 (F - I1/3 F^-T) + p(Jbar) cof(F),  p(J) = kappa/2 (J - 1/J).
1. finite-difference check of P = dW/dF (Jbar = J);
2. uniaxial-stress stretch curves: lateral stretch, J and nominal stress vs the incompressible limit;
3. wave speeds and explicit dt for the rubber ranges.
"""
import json

import numpy as np
from scipy.optimize import brentq


def W(F, mu, kappa):
    J = np.linalg.det(F)
    I1 = np.sum(F*F)
    return mu/2*(J**(-2/3)*I1-3)+kappa/4*(J*J-1-2*np.log(J))


def P(F, mu, kappa):
    J = np.linalg.det(F)
    I1 = np.sum(F*F)
    cof = J*np.linalg.inv(F).T
    return mu*J**(-2/3)*(F-I1/3*cof/J)+kappa/2*(J-1/J)*cof


def fd_check(seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.
    for _ in range(200):
        F = np.eye(3)+.4*rng.standard_normal((3, 3))
        if np.linalg.det(F) < .2:
            continue
        mu, kappa = 5e5, 1e7
        G = np.zeros((3, 3)); h = 1e-6
        for i in range(3):
            for j in range(3):
                E = np.zeros((3, 3)); E[i, j] = h
                G[i, j] = (W(F+E, mu, kappa)-W(F-E, mu, kappa))/(2*h)
        worst = max(worst, np.abs(G-P(F, mu, kappa)).max()/np.abs(G).max())
    return worst


def uniaxial(lam, mu, kappa):
    """Lateral stretch s with zero lateral Kirchhoff stress; returns s, J, nominal stress P11."""
    def lateral(s):
        J = lam*s*s
        return mu*J**(-2/3)*(s*s-(lam*lam+2*s*s)/3)+kappa/2*(J*J-1)
    s = brentq(lateral, 1e-3, 3.)
    J = lam*s*s
    tau11 = mu*J**(-2/3)*(lam*lam-s*s)
    return s, J, tau11/lam


if __name__ == "__main__":
    out = {"fd_relative_error_max": fd_check()}
    mu = 5e5
    rows = []
    for beta in (2.1667, 5., 10., 20., 50., 3000.):
        for lam in (.7, 1.25, 1.5, 2.0, 3.0):
            s, J, P11 = uniaxial(lam, mu, beta*mu)
            inc = mu*(lam-1/lam**2)
            rows.append(dict(beta=beta, nu=(3*beta-2)/(2*(3*beta+1)), stretch=lam, lateral=s, J=J,
                             nominal_stress_pa=P11, incompressible_pa=inc, stress_ratio=P11/inc))
    out["uniaxial"] = rows
    waves = []
    for G in (2e5, 5e5, 1e6, 2e6):
        for rho in (950., 1150., 1350.):
            for beta in (5., 10., 20., 50., 3000.):
                cs = np.sqrt(G/rho)
                cp = np.sqrt((beta+4/3)*G/rho)
                for dx in (5e-4, 1e-3, 1.5e-3, 2e-3, 4e-3):
                    waves.append(dict(G=G, rho=rho, beta=beta, dx=dx, c_s=cs, c_p=cp, dx_over_cp_s=dx/cp))
    out["waves"] = waves
    json.dump(out, open("analytic.json", "w"), indent=1)
    print("FD max relative error", out["fd_relative_error_max"])
    for r in rows:
        print(f"beta={r['beta']:7.1f} nu={r['nu']:.4f} stretch={r['stretch']:.2f} J={r['J']:.4f} "
              f"P/P_inc={r['stress_ratio']:.4f}")
    for w in waves:
        if w["G"] == 5e5 and w["rho"] == 1150.:
            print(f"G=0.5MPa rho=1150 beta={w['beta']:6.0f} dx={w['dx']*1e3:.1f}mm c_s={w['c_s']:.1f} "
                  f"c_p={w['c_p']:.1f} dx/c_p={w['dx_over_cp_s']*1e6:.2f}us")
