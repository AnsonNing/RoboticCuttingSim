"""Analytic / semi-analytic references of ``neo_hookean_jbar_prony_v1`` (ADR 0023 §21.2.8, protocol
``rubber_visc_tests_v1`` common.analytic_references; task R3.3). Float64, independent of the solver's update.

- small strain: G*(w) = mu0 [g_inf + sum_i g_i i w tau_i / (1 + i w tau_i)], E*(w) = 9 kappa G* / (3 kappa + G*)
  (viscosity on the isochoric part only);
- uniaxial relaxation E(t) and creep compliance J_E(t) by partial fractions of the rational Laplace transforms
  (G(s) = mu0 N(s) / D(s), D = prod(1 + s tau_i)): L[E](s) = E(s) / s with E(s) = 9 kappa G(s) / (3 kappa + G(s)),
  L[J_E](s) = 1 / (s E(s)) = 1 / (3 s G(s)) + 1 / (9 kappa s); simple poles (numpy roots), f(t) = sum_k
  P(p_k) / Q'(p_k) exp(p_k t);
- Boltzmann superposition of J_E on a staircase load (load constant within each control step);
- finite strain hereditary integral for a prescribed F(t) piecewise linear in time:
  Q_i(t) = g_i int_0^t exp(-(t-s)/tau_i) dS0/ds ds, propagated exactly between knots and integrated within an interval
  by parts, int_a^b e^{-(b-s)/tau} dS0 = S0(b) - e^{-(b-a)/tau} S0(a) - (1/tau) int_a^b e^{-(b-s)/tau} S0(s) ds, with
  ``scipy.integrate.quad`` (epsrel 1e-12) on the last integral (no derivative of S0, no use of the solver's
  recurrence); P = F (g_inf S0 + sum_i Q_i) + p(J) cof F (homogeneous block: Jbar = J);
- Euler-Bernoulli cantilever with a viscoelastic modulus: s^2 + (beta1 / L)^4 (I / (rho A)) E(s) = 0, root
  s1 = -zeta + i omega_d by Newton from the instantaneous elastic root.

Terms with g = 0 (padding) are ignored. Small-strain formulas differ from the finite-strain model by O(strain).

Protocol ``rubber_visc_tests_v2`` (ADR 0023 §21.7) adds the declared steady window (K whole periods with an integer
number of control steps, starting after 6 tau_max), the mean-removed estimator over it, and the inertia de-embedding
with the exact steady state of the 1-D linear viscoelastic rod (``rod_transfer``, ``deembed_rod``).
"""
import math

import numpy as np
from scipy import integrate

from .rubber_visco_constitutive import cofactor_np, s0_np as _s0_kernel_exponent

EXACT_EXPONENT = -2.0/3.0


def s0_np(F, mu):
    """S0 with the exact exponent -2/3 (the model; the solver's arithmetic uses -0.66666669, see
    ``rubber_visco_constitutive.EXP_M23``)."""
    return _s0_kernel_exponent(F, mu, EXACT_EXPONENT)

BETA1 = 1.8751040687119611


def _active(terms):
    return [(float(g), float(tau)) for g, tau in terms if float(g) != 0.0]


def g_inf_of(terms):
    return 1.0-math.fsum(float(g) for g, _ in terms)


def relaxation_function(t, terms):
    """g(t) = g_inf + sum_i g_i exp(-t / tau_i) (normalised shear relaxation)."""
    t = np.asarray(t, np.float64)
    return g_inf_of(terms)+sum(g*np.exp(-t/tau) for g, tau in _active(terms))


def complex_shear_modulus(omega, mu0, terms):
    w = np.asarray(omega, np.float64)
    out = g_inf_of(terms)+0j*w
    for g, tau in _active(terms):
        out = out+g*1j*w*tau/(1+1j*w*tau)
    return mu0*out


def uniaxial_complex_modulus(omega, mu0, kappa, terms):
    G = complex_shear_modulus(omega, mu0, terms)
    return 9*kappa*G/(3*kappa+G)


def _shear_polys(mu0, terms):
    """G(s) / mu0 = N(s) / D(s) as numpy.poly1d."""
    act = _active(terms)
    D = np.poly1d([1.0])
    for _, tau in act:
        D = D*np.poly1d([tau, 1.0])
    N = g_inf_of(terms)*D
    for i, (g, tau) in enumerate(act):
        term = np.poly1d([g*tau, 0.0])
        for j, (_, tj) in enumerate(act):
            if j != i:
                term = term*np.poly1d([tj, 1.0])
        N = N+term
    return N, D


def _inverse_laplace_simple(P, Q):
    """(residues, poles) of the proper rational P/Q with simple poles: f(t) = sum r_k exp(p_k t)."""
    poles = np.roots(Q.coeffs)
    dQ = Q.deriv()
    residues = np.array([P(p)/dQ(p) for p in poles])
    return residues, poles


def _evaluate(residues, poles, t):
    t = np.asarray(t, np.float64)
    out = np.zeros_like(t, dtype=complex)
    for r, p in zip(residues, poles):
        out = out+r*np.exp(p*t)
    return out.real


def relaxation_uniaxial(t, mu0, kappa, terms):
    """E(t): L[E] = 9 kappa mu0 N / (s (3 kappa D + mu0 N))."""
    N, D = _shear_polys(mu0, terms)
    P = 9*kappa*mu0*N
    Q = np.poly1d([1.0, 0.0])*(3*kappa*D+mu0*N)
    return _evaluate(*_inverse_laplace_simple(P, Q), t)


def creep_uniaxial(t, mu0, kappa, terms):
    """J_E(t): L[J_E] = D / (3 mu0 s N) + 1 / (9 kappa s) (t >= 0)."""
    N, D = _shear_polys(mu0, terms)
    r1, p1 = _inverse_laplace_simple(D, 3*mu0*np.poly1d([1.0, 0.0])*N)
    t = np.asarray(t, np.float64)
    return _evaluate(r1, p1, t)+1.0/(9*kappa)


def creep_terms(mu0, kappa, terms):
    """Receipt: J_E(t) = sum_k c_k exp(p_k t) (+ the bulk constant); retardation times -1/p_k."""
    N, D = _shear_polys(mu0, terms)
    r, p = _inverse_laplace_simple(D, 3*mu0*np.poly1d([1.0, 0.0])*N)
    return {"coefficients": [float(c.real) for c in r], "poles": [float(q.real) for q in p],
            "retardation_times_s": [float(-1/q.real) for q in p if abs(q) > 0], "bulk_constant": 1.0/(9*kappa)}


def boltzmann_staircase(t, step_times, levels, compliance):
    """eps(t) = sum_k (sigma_k - sigma_{k-1}) J(t - t_k) for a load equal to levels[k] on [step_times[k],
    step_times[k+1]) (sigma_{-1} = 0); ``compliance`` callable J(t) (t >= 0)."""
    t = np.atleast_1d(np.asarray(t, np.float64))
    step_times = np.asarray(step_times, np.float64)
    jumps = np.diff(np.r_[0.0, np.asarray(levels, np.float64)])
    out = np.zeros_like(t)
    for tk, dsig in zip(step_times, jumps):
        if dsig == 0.0:
            continue
        on = t >= tk
        out[on] += dsig*compliance(t[on]-tk)
    return out


# ----------------------------------------------------------------------------- finite-strain hereditary integral

class HereditaryBlock:
    """P_ref(t) of a homogeneous block for F(t) piecewise linear between knots (t_k, F_k), F(0) = I.

    ``evaluate(t)`` returns P (3x3) at any t in [0, knots[-1]] (float64). Q_i is propagated exactly from knot to
    knot; within an interval the convolution is integrated by parts with quad (epsrel 1e-12)."""

    def __init__(self, knot_times, knot_F, mu0, kappa, terms, epsrel=1e-12):
        self.t = np.asarray(knot_times, np.float64)
        self.F = np.asarray(knot_F, np.float64)
        if self.t[0] != 0.0 or not np.allclose(self.F[0], np.eye(3), atol=0):
            raise ValueError("the path must start at t = 0 with F = I")
        self.mu0, self.kappa = float(mu0), float(kappa)
        self.terms = _active(terms)
        self.g_inf = g_inf_of(terms)
        self.epsrel = epsrel
        self._q_knots = [np.zeros((len(self.terms), 3, 3))]
        for k in range(len(self.t)-1):
            self._q_knots.append(self._advance(k, self.t[k+1], self._q_knots[-1]))

    def F_at(self, t):
        k = int(np.clip(np.searchsorted(self.t, t, side="right")-1, 0, len(self.t)-2))
        a, b = self.t[k], self.t[k+1]
        s = (t-a)/(b-a)
        return self.F[k]+s*(self.F[k+1]-self.F[k])

    def _advance(self, k, t, q0):
        """Q_i(t) for t in [t_k, t_{k+1}] from Q_i(t_k)."""
        a = self.t[k]
        if t == a:
            return q0.copy()
        Fa, Fb = self.F[k], self.F[k+1]
        span = self.t[k+1]-a
        s0a = s0_np(Fa, self.mu0)
        Ft = Fa+(t-a)/span*(Fb-Fa)
        s0t = s0_np(Ft, self.mu0)
        moving = not np.array_equal(Fa, Fb)
        out = np.empty_like(q0)
        for i, (g, tau) in enumerate(self.terms):
            decay = math.exp(-(t-a)/tau)
            conv = np.zeros((3, 3))
            if moving:
                for r in range(3):
                    for c in range(r, 3):
                        f = (lambda s, r=r, c=c: math.exp(-(t-s)/tau)
                             * s0_np(Fa+(s-a)/span*(Fb-Fa), self.mu0)[r, c])
                        val, _ = integrate.quad(f, a, t, epsabs=0.0, epsrel=self.epsrel, limit=200)
                        conv[r, c] = conv[c, r] = val
            else:
                conv = s0a*tau*(1.0-decay)          # S0 constant on the interval: the integral in closed form
            out[i] = decay*q0[i]+g*(s0t-decay*s0a-conv/tau)
        return out

    def q_at(self, t):
        k = int(np.clip(np.searchsorted(self.t, t, side="right")-1, 0, len(self.t)-2))
        return self._advance(k, t, self._q_knots[k])

    def evaluate(self, t):
        F = self.F_at(t)
        s0 = s0_np(F, self.mu0)
        S = self.g_inf*s0+self.q_at(t).sum(0)
        J = float(np.linalg.det(F))
        p = 0.5*self.kappa*(J-1.0/J)
        return F@S+p*cofactor_np(F)

    def step_closed_form(self, t, F_hold, t_ramp):
        """Instantaneous-step reference g(t - t_ramp) S0(F_hold) (reported alongside; not the criterion)."""
        s0 = s0_np(F_hold, self.mu0)
        g = self.g_inf+sum(gi*math.exp(-(t-t_ramp)/tau) for gi, tau in self.terms)
        J = float(np.linalg.det(F_hold))
        return F_hold@(g*s0)+0.5*self.kappa*(J-1.0/J)*cofactor_np(F_hold)


# ----------------------------------------------------------------------------- beam free decay

def beam_root(length, width, thickness, density, mu0, kappa, terms, *, tol=1e-13, max_iter=100):
    """Oscillatory root s1 = -zeta + i omega_d of s^2 + (beta1/L)^4 (I / (rho A)) E(s) = 0 (rectangle W x t, bending
    about the width axis: I / A = t^2 / 12). Returns dict(zeta, omega_d, f_d, log_decrement, s, elastic_f)."""
    k4 = (BETA1/length)**4*thickness**2/(12.0*density)

    def E_of(s):
        G = mu0*(g_inf_of(terms)+sum(g*s*tau/(1+s*tau) for g, tau in _active(terms)))
        return 9*kappa*G/(3*kappa+G)

    def f(s):
        return s*s+k4*E_of(s)

    E0 = 9*kappa*mu0/(3*kappa+mu0)
    s = 1j*math.sqrt(k4*E0)
    for _ in range(max_iter):
        h = 1e-7*abs(s)
        d = (f(s+h)-f(s-h))/(2*h)
        step = f(s)/d
        s = s-step
        if abs(step) <= tol*abs(s):
            break
    zeta, wd = -s.real, s.imag
    return {"s": [s.real, s.imag], "zeta_1_s": zeta, "omega_d_rad_s": wd, "f_d_hz": wd/(2*math.pi),
            "log_decrement": 2*math.pi*zeta/wd, "elastic_instantaneous_f_hz": math.sqrt(k4*E0)/(2*math.pi),
            "k4": k4}


def fourier_modulus(sigma, eps, t_sigma, t_eps, omega, control_dt):
    """E* from control-step samples (protocol common.observables.complex_modulus_from_cycle): wrench samples are
    averages over (t_k - dtc, t_k] stamped t_k - dtc/2 and divided by sinc(w dtc / 2)."""
    x = omega*control_dt/2
    sinc = math.sin(x)/x if x != 0 else 1.0
    num = np.sum(np.asarray(sigma)*np.exp(-1j*omega*np.asarray(t_sigma)))/sinc
    e = np.asarray(eps, np.float64)
    den = np.sum((e-e.mean())*np.exp(-1j*omega*np.asarray(t_eps)))
    return complex(num/den)


# ----------------------------------------------------------------------------- rubber_visc_tests_v2 (ADR 0023 §21.7)

MAX_WINDOW_PERIODS = 64


def steady_window_v2(f, control_dt, tau_max):
    """``common.observables.steady_window`` of v2: K = the smallest number of periods with an integer number of control
    steps; n_cycles = K ceil((ceil(6 tau_max f) + K) / K); returns dict(periods K, n_cycles, steps, window_samples)."""
    spp = 1.0/(f*control_dt)
    for K in range(1, MAX_WINDOW_PERIODS+1):
        if abs(K*spp-round(K*spp)) <= 1e-9*K*spp:
            break
    else:
        raise ValueError(f"no whole-period window with <= {MAX_WINDOW_PERIODS} periods at f = {f} Hz")
    n_cycles = K*math.ceil((math.ceil(6.0*tau_max*f-1e-12)+K)/K)
    steps = n_cycles*spp
    if abs(steps-round(steps)) > 1e-9*steps:
        raise ValueError("run length is not an integer number of control steps")
    return {"periods": K, "n_cycles": n_cycles, "steps": int(round(steps)), "window_samples": int(round(K*spp))}


def fourier_modulus_window_v2(sigma, eps, t_sigma, t_eps, omega, control_dt):
    """``common.observables.complex_modulus_from_window`` of v2: the v1 projection with sigma AND eps mean-removed
    over the (whole-period) window; wrench samples stamped t_k - dtc/2 and divided by sinc(w dtc / 2)."""
    x = omega*control_dt/2
    sinc = math.sin(x)/x if x != 0 else 1.0
    s = np.asarray(sigma, np.float64)
    e = np.asarray(eps, np.float64)
    num = np.sum((s-s.mean())*np.exp(-1j*omega*np.asarray(t_sigma, np.float64)))/sinc
    den = np.sum((e-e.mean())*np.exp(-1j*omega*np.asarray(t_eps, np.float64)))
    return complex(num/den)


def rod_transfer(omega, E, rho, length, a, b, end):
    """Exact steady state of the 1-D linear viscoelastic rod rho u_tt = E u_xx, u(0) = 0, u(L) = U e^{iwt}: ratio of the
    end stress to the gauge strain (u(b) - u(a)) / (b - a). ``end`` 1: the driven end x = L,
    k (b - a) cos(kL) / (sin kb - sin ka); ``end`` 0: the fixed end x = 0, k (b - a) / (sin kb - sin ka);
    k = w sqrt(rho / E) (principal branch). The measured apparent modulus is E T."""
    k = omega*np.sqrt(rho/complex(E))
    ratio = k*(b-a)/(np.sin(k*b)-np.sin(k*a))
    return complex(ratio*np.cos(k*length)) if end == 1 else complex(ratio)


def deembed_rod(E_meas, omega, rho, length, a, b, end, *, tol=1e-14, max_iter=100):
    """E_mat with E_meas = E_mat T(w; E_mat) by fixed-point iteration from E_meas (v2
    ``common.observables.inertia_deembedding``); returns (E_mat, iterations). Raises if not converged."""
    E_meas = complex(E_meas)
    E = E_meas
    for it in range(1, max_iter+1):
        new = E_meas/rod_transfer(omega, E, rho, length, a, b, end)
        if abs(new-E) <= tol*abs(E):
            return new, it
        E = new
    raise RuntimeError(f"rod de-embedding did not converge in {max_iter} iterations")


__all__ = ["relaxation_function", "complex_shear_modulus", "uniaxial_complex_modulus", "relaxation_uniaxial",
           "creep_uniaxial", "creep_terms", "boltzmann_staircase", "HereditaryBlock", "beam_root", "fourier_modulus",
           "steady_window_v2", "fourier_modulus_window_v2", "rod_transfer", "deembed_rod"]
