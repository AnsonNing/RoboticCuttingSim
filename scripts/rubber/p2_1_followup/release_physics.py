"""Analytic reference numbers for the RUBBER-PHYS-07 release (P2-1 follow-up; no simulation).

Material neo_hookean_jbar_v1 with the RUBBER-PHYS-06/07 strip (mu 0.4 MPa, beta 20, rho 1150, 60 x 10 x 4 mm).
1-D bar under uniaxial stress (lateral faces free, compressible solution of rubber_reference.uniaxial_stress):
nominal stress P(lambda) is concave (P'' < 0) for every lambda, so an instantaneous release (lambda decreasing) is a
compressive SHOCK in the 1-D theory (entropy solution of the Riemann problem, Rankine-Hugoniot jump conditions
[v] = -U [lambda], [P] = -rho U [v]); its dissipation per unit reference volume swept is
D = W(lambda_-) - W(lambda_+) - (P_- + P_+) / 2 (lambda_- - lambda_+) (area between the chord and the curve).
The reflected shock at the fixed end brings the material to rest at a compressive stretch lambda_c.
3-D: lateral inertia makes the bar dispersive (Pochhammer-Chree), so the exact 3-D solution has no shock and no
dissipation (an undular bore); the 1-D numbers bound the strength of the wave, not the 3-D energy budget.
Euler buckling of the clamped-free strip: eps_cr = pi^2 t^2 / (48 L^2) (bending about the thin axis); dynamic
buckling growth rate of a beam under axial compressive strain eps: s_max = (eps / 2) c0 / r_g, r_g = t / sqrt(12),
c0 = sqrt(E_eff / rho), at wavelength 2 pi r_g sqrt(2 / eps).
Mean length of an undamped large-amplitude oscillation: P ~ 3 mu (eps - eps^2) gives <eps> ~ <eps^2> > 0, i.e. the
time-averaged length is NOT the rest length for a non-small amplitude (the v1/v2 mean-length criterion presumes a
linear oscillation).

    PYTHONPATH=src .venv/bin/python scripts/rubber/p2_1_followup/release_physics.py --out <file.json>
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))
from robotic_cutting_sim import rubber_reference as R  # noqa: E402

MU, BETA, RHO = 4e5, 20.0, 1150.0
KAPPA = BETA*MU
L, W, T = 0.06, 0.01, 0.004


def P(lam):
    return R.uniaxial_stress(lam, MU, KAPPA)["nominal_stress"]


def Wd(lam):
    """Strain energy per reference volume along the uniaxial-stress path (= integral of P dlambda from 1)."""
    return quad(P, 1.0, lam, limit=200)[0]


def dP(lam, h=1e-6):
    return (P(lam+h)-P(lam-h))/(2*h)


def shock(lm, lp):
    U2 = (P(lm)-P(lp))/(RHO*(lm-lp))
    D = Wd(lm)-Wd(lp)-0.5*(P(lm)+P(lp))*(lm-lp)
    return math.sqrt(U2), D


def release(lam):
    Ueff = 9*KAPPA*MU/(3*KAPPA+MU)
    c0 = math.sqrt(Ueff/RHO)
    U1, D1 = shock(lam, 1.0)
    v1 = U1*(lam-1.0)
    # reflected shock at the fixed end: from (lambda 1, v1 toward the wall) to (lambda_c, 0)
    f = lambda lc: (P(1.0)-P(lc))/(RHO*(1.0-lc))*(1.0-lc)**2-v1**2   # U_r^2 [lambda]^2 = [v]^2
    lc = brentq(f, 1e-3, 1.0-1e-9)
    U2, D2 = shock(1.0, lc)
    Wst = Wd(lam)
    tension_eps = lam-1.0
    eps_cr = math.pi**2*T**2/(48*L**2)
    rg = T/math.sqrt(12)
    comp = 1.0-lc
    s_max = comp/2*c0/rg
    wavelength = 2*math.pi*rg*math.sqrt(2/comp)
    period_ff = 4*L/c0
    # crude e-fold count over 10 fixed-free periods with compression during about a quarter of each period
    efolds_10 = s_max*10*period_ff/4
    return {"lambda_release": lam, "stored_energy_density_j_m3": Wst, "stored_energy_j": Wst*L*W*T,
            "c0_m_s": c0, "tangent_ratio_P_prime_vs_lambda1": {str(x): dP(x)/dP(1.0) for x in (0.5, 0.63, 1.5, 2.0)},
            "release_shock_speed_m_s": U1, "release_particle_speed_m_s": v1,
            "release_shock_dissipation_fraction_per_pass": D1/Wst,
            "reflected_compressive_stretch_1d": lc, "reflected_shock_speed_m_s": U2,
            "reflected_shock_dissipation_fraction": D2/Wst,
            "euler_buckling_strain_clamped_free": eps_cr, "compression_over_buckling_strain": comp/eps_cr,
            "dynamic_buckling_growth_rate_1_s": s_max, "dynamic_buckling_wavelength_m": wavelength,
            "dynamic_buckling_efold_time_s": 1/s_max, "fixed_free_axial_period_s": period_ff,
            "dynamic_buckling_efolds_over_10_periods_estimate": efolds_10,
            "mean_strain_bias_of_oscillation_estimate": 0.5*tension_eps**2,
            "linear_1d_compressive_stretch": 2.0-lam}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    lam = np.linspace(0.3, 3.0, 271)
    Ppp = np.gradient(np.gradient([P(x) for x in lam], lam), lam)
    res = {"label": "analytic reference (not a simulation)", "material": {"mu_pa": MU, "beta": BETA, "rho": RHO},
           "strip_m": [L, W, T], "P_concave_on_0.3_3.0": bool(np.all(Ppp[2:-2] < 0)),
           "cases": [release(x) for x in (1.02, 1.03, 1.05, 1.1, 1.25, 1.5)]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for c in res["cases"]:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()
               if k not in ("tangent_ratio_P_prime_vs_lambda1",)})
    print("P concave:", res["P_concave_on_0.3_3.0"], res["cases"][-1]["tangent_ratio_P_prime_vs_lambda1"])


if __name__ == "__main__":
    main()
