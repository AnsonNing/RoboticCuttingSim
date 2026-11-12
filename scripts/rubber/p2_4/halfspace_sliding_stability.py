"""Analytic reference (task X1, ADR 0023 §22.9): linear stability of steady Coulomb sliding of an isotropic linear-elastic
half-space on a RIGID flat (plane strain), the continuum problem the node penalty law discretises.

Perturbation u ~ exp(i k x - alpha z + s t) of the body z > 0, contact kept (u_z = 0 at z = 0), Coulomb slip
sigma_xz + mu sigma_zz = 0 (sliding direction fixed; mu -> -mu is the other direction). With s = k c_s sigma,
a = sqrt(1 + g2 sigma^2), b = sqrt(1 + sigma^2), g2 = c_s^2 / c_p^2 = (1 - 2 nu) / (2 (1 - nu)), Re a, Re b > 0:

    f(sigma) = i a sigma^2 + mu (sigma^2 + 2 - 2 a b) = 0.

A root with Re sigma > 0 grows at a rate proportional to |k| (no length scale): steady sliding is then unstable at
every wavelength and the problem is ill-posed (Hadamard; the class of results of Renardy 1992 and Martins, Guimaraes &
Faria 1995). The script scans mu and bisects the threshold mu_c(nu) where the growing root appears, and prints the
root at mu_c + 0.05 for the record. The rubber material of the contact protocols has beta = kappa / mu0 = 20, i.e.
nu = (3 beta - 2) / (6 beta + 2) = 0.4754.

    .venv/bin/python scripts/rubber/p2_4/halfspace_sliding_stability.py
"""
import json

import numpy as np
from scipy.optimize import fsolve


def _sqrt(z):
    r = np.sqrt(complex(z))
    return r if r.real >= 0 else -r


def growing_roots(mu, nu, grid=31, span=3.0):
    g2 = (1-2*nu)/(2*(1-nu))

    def f(v):
        s = v[0]+1j*v[1]
        a, b = _sqrt(1+g2*s*s), _sqrt(1+s*s)
        r = 1j*a*s*s+mu*(s*s+2-2*a*b)
        return [r.real, r.imag]
    roots = set()
    for x0 in np.linspace(-span, span, grid):
        for y0 in np.linspace(-span, span, grid):
            v, _info, ier, _msg = fsolve(f, [x0, y0], full_output=True)
            if ier == 1 and np.hypot(*f(v)) < 1e-11 and abs(complex(*v)) > 1e-6:
                s = complex(*v)
                if _sqrt(1+g2*s*s).real > 1e-9 and _sqrt(1+s*s).real > 1e-9 and s.real > 1e-7:
                    roots.add((round(v[0], 7), round(v[1], 7)))
    return sorted(roots)


def threshold(nu, lo=0.2, hi=2.0, tol=1e-3):
    if growing_roots(lo, nu) or not growing_roots(hi, nu):
        return None
    while hi-lo > tol:
        mid = 0.5*(lo+hi)
        if growing_roots(mid, nu):
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)


def main():
    out = {}
    for nu in (0.3, 0.4, 0.45, 0.4754, 0.49):
        mc = threshold(nu)
        out[str(nu)] = {"mu_c": mc, "root_at_mu_c_plus_0.05": growing_roots(mc+0.05, nu) if mc else None,
                        "scan": {str(m): bool(growing_roots(m, nu)) for m in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5)}}
        print(nu, json.dumps(out[str(nu)]), flush=True)


if __name__ == "__main__":
    main()
