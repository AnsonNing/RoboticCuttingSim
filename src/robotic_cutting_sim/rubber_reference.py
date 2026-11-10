"""Analytic and semi-analytic references for the rubber tests (ADR 0023 §5, §10, §14.3; task R0.3). Float64.

- ``neo_hookean_jbar_v1``: energy density, exact PK1 (with the cell dilatation J-bar), Cauchy stress;
- homogeneous states: compressible uniaxial stress (lateral stretch, J, nominal stress), incompressible uniaxial,
  simple shear (sigma_12 = mu gamma, sigma_11 - sigma_22 = mu gamma^2);
- beams: Euler-Bernoulli and Timoshenko static cantilever sag under uniform load, first bending frequency,
  the inextensible elastica (BVP) for large gravity sag, the load parameter alpha = q L^3 / (E I);
- Prony series: relaxation function g(t), G'(omega), G''(omega), loss per cycle.

These are the references of RUBBER-PHYS-01..08 and RUBBER-VISC (P2-3); they never read simulator output.
"""
import math

import numpy as np

BETA1_L = 1.8751040687119611   # first root of 1 + cos x cosh x = 0 (clamped-free beam)


# ----------------------------------------------------------------------------- constitutive (float64)

def cofactor(F):
    """cof F = J F^-T from cross products of the columns (finite for any F, including J <= 0)."""
    F = np.asarray(F, float)
    c0 = np.cross(F[:, 1], F[:, 2])
    c1 = np.cross(F[:, 2], F[:, 0])
    c2 = np.cross(F[:, 0], F[:, 1])
    return np.stack([c0, c1, c2], axis=1)


def volumetric_energy(J, kappa):
    """U(J) = kappa/4 (J^2 - 1 - 2 ln J) (per unit reference volume)."""
    return kappa/4*(J*J-1-2*np.log(J))


def pressure(J, kappa):
    """p(J) = U'(J) = kappa/2 (J - 1/J)."""
    return kappa/2*(J-1/J)


def isochoric_energy(F, mu):
    F = np.asarray(F, float)
    J = np.linalg.det(F)
    return mu/2*(J**(-2/3)*np.sum(F*F)-3)


def energy_density(F, mu, kappa, Jbar=None):
    J = np.linalg.det(np.asarray(F, float))
    return isochoric_energy(F, mu)+volumetric_energy(J if Jbar is None else Jbar, kappa)


def pk1(F, mu, kappa, Jbar=None):
    """P = mu J^-2/3 (F - I1/3 F^-T) + p(Jbar) cof F (ADR §5); Jbar = J for a single particle."""
    F = np.asarray(F, float)
    J = np.linalg.det(F)
    Jb = J if Jbar is None else float(Jbar)
    cof = cofactor(F)
    I1 = np.sum(F*F)
    a = mu*J**(-2/3)
    return a*F+(pressure(Jb, kappa)-a*I1/(3*J))*cof


def cauchy(P, F):
    F = np.asarray(F, float)
    return np.asarray(P, float)@F.T/np.linalg.det(F)


# ----------------------------------------------------------------------------- homogeneous states

def uniaxial_stress(stretch, mu, kappa):
    """Compressible NH (J-bar = J) under uniaxial stress: returns dict(lateral, J, nominal_stress, cauchy_stress)."""
    from scipy.optimize import brentq
    lam = float(stretch)

    def lateral_kirchhoff(s):
        J = lam*s*s
        return mu*J**(-2/3)*(s*s-(lam*lam+2*s*s)/3)+kappa/2*(J*J-1)
    s = brentq(lateral_kirchhoff, 1e-3, 3.0, xtol=1e-15, rtol=1e-15)
    J = lam*s*s
    tau11 = mu*J**(-2/3)*(lam*lam-s*s)
    return {"stretch": lam, "lateral": s, "J": J, "nominal_stress": tau11/lam, "cauchy_stress": tau11/J,
            "F": np.diag([lam, s, s])}


def incompressible_uniaxial_nominal(stretch, mu):
    return mu*(stretch-stretch**-2)


def simple_shear(gamma, mu):
    """Simple shear F = I + gamma e1 x e2 (J = 1): sigma_12 = mu gamma, sigma_11 - sigma_22 = mu gamma^2,
    sigma_22 - sigma_33 = 0 (exact for the decoupled NH with J = J-bar = 1)."""
    F = np.eye(3)
    F[0, 1] = gamma
    return {"F": F, "sigma_12": mu*gamma, "normal_difference_12": mu*gamma**2, "normal_difference_23": 0.0}


# ----------------------------------------------------------------------------- beams

def rect_section(width, thickness):
    A = width*thickness
    return A, width*thickness**3/12


def load_parameter(density, gravity, width, thickness, length, young):
    """alpha = q L^3 / (E I), q = rho g A (gravity sag; elastica tip drop / L -> alpha/8 for alpha -> 0)."""
    A, I = rect_section(width, thickness)
    return density*gravity*A*length**3/(young*I)


def euler_bernoulli_sag(density, gravity, width, thickness, length, young):
    """Static tip deflection of a clamped-free beam under its weight: q L^4 / (8 E I)."""
    A, I = rect_section(width, thickness)
    return density*gravity*A*length**4/(8*young*I)


def timoshenko_sag(density, gravity, width, thickness, length, young, poisson):
    """EB plus the shear deflection q L^2 / (2 k G A), k = 10(1+nu)/(12+11nu) (rectangle)."""
    A, _ = rect_section(width, thickness)
    G = young/(2*(1+poisson))
    k = 10*(1+poisson)/(12+11*poisson)
    return (euler_bernoulli_sag(density, gravity, width, thickness, length, young)
            + density*gravity*A*length**2/(2*k*G*A))


def plate_bracket(eb_value, poisson):
    """Wide strip: the sag lies between the plate value EB (1 - nu^2) and the beam value EB."""
    return eb_value*(1-poisson**2), eb_value


def cantilever_f1(density, width, thickness, length, young):
    """First bending frequency (Hz): (1.8751^2 / 2 pi) sqrt(E I / (rho A L^4))."""
    A, I = rect_section(width, thickness)
    return BETA1_L**2/(2*math.pi)*math.sqrt(young*I/(density*A*length**4))


def elastica(alpha, nodes=401):
    """Inextensible elastica of a clamped-free strip under its own weight.

    theta'' = alpha (1 - s) cos(theta), theta(0) = 0, theta'(1) = 0 on s in [0, 1] (theta < 0 is downwards).
    Returns dict(tip_drop, tip_x) in units of L and the BVP success flag. alpha -> 0 gives tip_drop -> alpha/8."""
    from scipy.integrate import solve_bvp
    alpha = float(alpha)
    if alpha <= 0:
        return {"tip_drop": 0.0, "tip_x": 1.0, "success": True}

    def f(s, y):
        return np.vstack([y[1], alpha*(1-s)*np.cos(y[0])])

    def bc(a, b):
        return np.array([a[0], b[1]])
    s = np.linspace(0, 1, nodes)
    guess_theta = -alpha*(s/2-s**2/2+s**3/6)/max(1.0, alpha/2)
    y0 = np.vstack([guess_theta, np.gradient(guess_theta, s)])
    sol = solve_bvp(f, bc, s, y0, tol=1e-10, max_nodes=200000)
    s_fine = np.linspace(0, 1, 20001)
    th = sol.sol(s_fine)[0]
    z = np.concatenate([[0], np.cumsum((np.sin(th[1:])+np.sin(th[:-1]))/2*np.diff(s_fine))])
    x = np.concatenate([[0], np.cumsum((np.cos(th[1:])+np.cos(th[:-1]))/2*np.diff(s_fine))])
    return {"tip_drop": float(-z[-1]), "tip_x": float(x[-1]), "success": bool(sol.success)}


def richardson_two_grid(coarse, fine, ratio, order=2.0):
    """Extrapolated value from two grids with an assumed order (three grids give the observed order, P2-2)."""
    return fine+(fine-coarse)/(ratio**order-1)


def power_law_exponent(xs, ys):
    """Least-squares slope of log y against log x."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    return float(np.polyfit(lx, ly, 1)[0])


# ----------------------------------------------------------------------------- Prony series (P2-3 references)

def prony_relaxation(t, terms):
    """g(t) = g_inf + sum g_i exp(-t/tau_i); ``terms`` = [(g_i, tau_i)]."""
    g_inf = 1-sum(g for g, _ in terms)
    t = np.asarray(t, float)
    return g_inf+sum(g*np.exp(-t/tau) for g, tau in terms)


def prony_moduli(omega, mu0, terms):
    """(G'(omega), G''(omega)) of the isochoric Prony series with instantaneous modulus mu0."""
    g_inf = 1-sum(g for g, _ in terms)
    w = np.asarray(omega, float)
    storage = mu0*(g_inf+sum(g*(w*tau)**2/(1+(w*tau)**2) for g, tau in terms))
    loss = mu0*sum(g*w*tau/(1+(w*tau)**2) for g, tau in terms)
    return storage, loss


def prony_loss_per_cycle(omega, gamma0, mu0, terms):
    """Dissipated energy per cycle and unit volume in small-amplitude shear: pi gamma0^2 G''."""
    return math.pi*gamma0**2*float(prony_moduli(omega, mu0, terms)[1])
