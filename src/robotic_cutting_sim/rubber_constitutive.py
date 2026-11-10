"""``neo_hookean_jbar_v1`` Warp functions shared by the single-env and the batched rubber solvers (ADR 0023 §5; R1.1).

Energy per particle p in reference cell c (V_p uniform, J-bar = mean of det F over the cell's particles):

    Psi = sum_p V_p mu/2 (J_p^-2/3 I1_p - 3) + sum_c V_c U(Jbar_c),   U(J) = kappa/4 (J^2 - 1 - 2 ln J)

and the exact PK1 used by the frozen ``tlmpm_v5._p2g``:

    P_p = mu J^-2/3 (F - I1/3 F^-T) + p(Jbar_c) cof F = a F + (p_bar - a I1 / (3 J)) cof F,   a = mu J^-2/3,
    p(J) = kappa/2 (J - 1/J).

Numerics: F is stored in float32 (as every frozen kernel); determinants, the J-bar cell mean and the PK1 are evaluated
in float64 and the stress is rounded once to float32. ``cof F`` comes from cross products of the columns (no division),
so it stays finite for J <= 0. Valid domain J, Jbar in [lo, hi] (recipe ``j_domain``, default [0.2, 5]); outside it
(including NaN and inversion J <= 0) the caller counts a violation that becomes the sticky ``RUBBER_J_DOMAIN`` flag of
the env. Only to keep the same substep finite, the evaluation uses J_eval = max(J, 0.05) in ``a`` and ``I1/(3J)`` and
Jbar_eval = max(Jbar, 0.05) in ``p_bar``; the state itself is never repaired. There is no SVD and no singular-value
clamp. Every function here is pure: the single and the batched kernels call the same code (structural L0 parity).

Stable-step guard ``stable_step_guard_v1`` (ADR 0023 §7 amendment, P2-1 follow-up): TLMPM integrates with the
REFERENCE gradients, so the explicit stability limit is set by the reference-frame acoustic tensor
Q(N) = d(P N)/dw = a I + (p'(Jbar) + 5 a I1 / (9 J^2)) g g^T - 2 a / (3 J) (f g^T + g f^T), f = F N, g = cof F N
(f . g = J). Its largest eigenvalue over unit N is bounded by
    rho c_bound^2 = a + (p'(Jbar) + 5 a I1 / (9 J^2)) g_max^2 + 2 a / (3 J) (s_max g_max - J),
    g_max = J / s_min = sqrt(lambda_max(cof^T cof)),  s_max = sqrt(lambda_max(F^T F)),  p'(J) = kappa/2 (1 + 1/J^2)
(Weyl; Jbar in place of J, which bounds the cell-averaged volumetric term). At F = I it is kappa + 8 mu / 3 (exact:
kappa + 4 mu / 3, ratio of speeds 1.031 at beta 20); in uniaxial compression it tracks the exact discrete operator
within 1-6 % (scripts/rubber/p2_1_followup/stability_spectrum.py). ``cfl_effective`` = dt c_bound / dx per particle;
the solver counts a violation when it exceeds the declared limit (default 1.0, below the measured critical CFL 1.2-1.5
of ADR E2). Float32 evaluation (a stability bound, not a stress); the same J floors as the stress.
"""
import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

MODEL_ID = "neo_hookean_jbar_v1"
MODEL_VERSION = 1
J_EVAL_FLOOR = 0.05          # finite-evaluation floor only (ADR §5); never a state repair
STABLE_STEP_GUARD_ID = "stable_step_guard_v1"
STABLE_STEP_GUARD_DEFAULT_LIMIT = 1.0


@wp.func
def cofactor_d(f: wp.mat33d):
    """cof F = J F^-T from cross products of the columns of F (finite for any F)."""
    c0 = wp.cross(wp.vec3d(f[0, 1], f[1, 1], f[2, 1]), wp.vec3d(f[0, 2], f[1, 2], f[2, 2]))
    c1 = wp.cross(wp.vec3d(f[0, 2], f[1, 2], f[2, 2]), wp.vec3d(f[0, 0], f[1, 0], f[2, 0]))
    c2 = wp.cross(wp.vec3d(f[0, 0], f[1, 0], f[2, 0]), wp.vec3d(f[0, 1], f[1, 1], f[2, 1]))
    return wp.mat33d(c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2])


@wp.func
def det_d(f: wp.mat33):
    """det F in float64 of the float32 F."""
    return wp.determinant(wp.mat33d(f))


@wp.func
def in_domain(j: wp.float64, lo: wp.float64, hi: wp.float64):
    """True iff lo <= j <= hi (False for NaN)."""
    return j >= lo and j <= hi


@wp.func
def pressure_d(jbar: wp.float64, kappa: wp.float64):
    jb = wp.max(jbar, wp.float64(J_EVAL_FLOOR))
    return wp.float64(0.5)*kappa*(jb-wp.float64(1.0)/jb)


@wp.func
def nh_jbar_pk1(f32: wp.mat33, jbar: wp.float64, mu32: float, kappa32: float):
    """PK1 of neo_hookean_jbar_v1 for one particle (float64 evaluation, float32 result)."""
    f = wp.mat33d(f32)
    mu = wp.float64(mu32)
    kappa = wp.float64(kappa32)
    j = wp.max(wp.determinant(f), wp.float64(J_EVAL_FLOOR))
    cof = cofactor_d(f)
    i1 = wp.ddot(f, f)
    a = mu*wp.pow(j, wp.float64(-2.0/3.0))
    scale = pressure_d(jbar, kappa)-a*i1/(wp.float64(3.0)*j)
    return wp.mat33(a*f+scale*cof)


@wp.func
def iso_energy_density(f32: wp.mat33, mu32: float):
    """mu/2 (J^-2/3 I1 - 3) per unit reference volume (float64; J floored for finiteness only)."""
    f = wp.mat33d(f32)
    j = wp.max(wp.determinant(f), wp.float64(J_EVAL_FLOOR))
    return wp.float64(0.5)*wp.float64(mu32)*(wp.pow(j, wp.float64(-2.0/3.0))*wp.ddot(f, f)-wp.float64(3.0))


@wp.func
def vol_energy_density(jbar: wp.float64, kappa32: float):
    """U(Jbar) = kappa/4 (Jbar^2 - 1 - 2 ln Jbar) per unit reference volume (float64)."""
    jb = wp.max(jbar, wp.float64(J_EVAL_FLOOR))
    return wp.float64(0.25)*wp.float64(kappa32)*(jb*jb-wp.float64(1.0)-wp.float64(2.0)*wp.log(jb))


# ----------------------------------------------------------------------------- stable-step guard (float32)

@wp.func
def cofactor32(f: wp.mat33):
    """cof F = J F^-T from cross products of the columns (float32, finite for any F)."""
    c0 = wp.cross(wp.vec3(f[0, 1], f[1, 1], f[2, 1]), wp.vec3(f[0, 2], f[1, 2], f[2, 2]))
    c1 = wp.cross(wp.vec3(f[0, 2], f[1, 2], f[2, 2]), wp.vec3(f[0, 0], f[1, 0], f[2, 0]))
    c2 = wp.cross(wp.vec3(f[0, 0], f[1, 0], f[2, 0]), wp.vec3(f[0, 1], f[1, 1], f[2, 1]))
    return wp.mat33(c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2])


@wp.func
def sym3_max_eig(s: wp.mat33):
    """Largest eigenvalue of a symmetric 3x3 matrix (trigonometric closed form; never above q + 2p, the mean plus
    twice the root-mean-square deviation, which bounds every eigenvalue)."""
    q = (s[0, 0]+s[1, 1]+s[2, 2])/3.0
    d0 = s[0, 0]-q
    d1 = s[1, 1]-q
    d2 = s[2, 2]-q
    p2 = d0*d0+d1*d1+d2*d2+2.0*(s[0, 1]*s[0, 1]+s[0, 2]*s[0, 2]+s[1, 2]*s[1, 2])
    out = q
    if p2 > 0.0:
        p = wp.sqrt(p2/6.0)
        b = (s-q*wp.identity(n=3, dtype=float))*(1.0/p)
        r = wp.clamp(0.5*wp.determinant(b), -1.0, 1.0)
        out = q+2.0*p*wp.cos(wp.acos(r)/3.0)
    return out


@wp.func
def wave_modulus_bound(f: wp.mat33, jbar: wp.float64, mu: float, kappa: float):
    """Upper bound of rho c^2 of the reference-frame acoustic tensor over all directions (module docstring)."""
    j = wp.max(wp.determinant(f), float(J_EVAL_FLOOR))
    jb = wp.max(float(jbar), float(J_EVAL_FLOOR))
    a = mu*wp.pow(j, -2.0/3.0)
    i1 = wp.ddot(f, f)
    c = cofactor32(f)
    g2 = wp.max(sym3_max_eig(wp.transpose(c)*c), 0.0)
    smax = wp.sqrt(wp.max(sym3_max_eig(wp.transpose(f)*f), 0.0))
    pp = 0.5*kappa*(1.0+1.0/(jb*jb))
    return a+(pp+5.0*a*i1/(9.0*j*j))*g2+2.0*a/(3.0*j)*wp.max(smax*wp.sqrt(g2)-j, 0.0)


@wp.func
def cfl_effective(f: wp.mat33, jbar: wp.float64, mu: float, kappa: float, rho: float, dt_over_dx: float):
    """dt c_bound / dx of one particle (NaN for a non-finite F, so a ``not (c <= limit)`` test counts it)."""
    return dt_over_dx*wp.sqrt(wave_modulus_bound(f, jbar, mu, kappa)/rho)


# ----------------------------------------------------------------------------- kernels shared by single and batch

@wp.kernel
def cell_jbar(cell_node: wp.array(dtype=int), cell_offsets: wp.array(dtype=int),
              cell_particles: wp.array(dtype=int), F: wp.array(dtype=wp.mat33), jbar: wp.array(dtype=wp.float64)):
    """J-bar pass: one thread per occupied reference cell, deterministic gather over the constant cell CSR (no
    atomics). The cell of a particle is its stencil corner-0 node, so Jbar lives in a node-sized array."""
    c = wp.tid()
    start = cell_offsets[c]
    stop = cell_offsets[c+1]
    total = wp.float64(0.0)
    for k in range(start, stop):
        total += det_d(F[cell_particles[k]])
    jbar[cell_node[c]] = total/wp.float64(stop-start)


@wp.kernel
def _pk1_points(F: wp.array(dtype=wp.mat33), jbar: wp.array(dtype=wp.float64), mu: float, kappa: float,
                lo: wp.float64, hi: wp.float64, stress: wp.array(dtype=wp.mat33), valid: wp.array(dtype=int)):
    """Test/diagnostic helper: PK1 and the domain test for independent (F, Jbar) pairs."""
    i = wp.tid()
    stress[i] = nh_jbar_pk1(F[i], jbar[i], mu, kappa)
    valid[i] = int(in_domain(det_d(F[i]), lo, hi) and in_domain(jbar[i], lo, hi))


@wp.kernel
def _cfl_points(F: wp.array(dtype=wp.mat33), jbar: wp.array(dtype=wp.float64), mu: float, kappa: float, rho: float,
                dt_over_dx: float, out: wp.array(dtype=float)):
    """Test/diagnostic helper: cfl_effective for independent (F, Jbar) pairs."""
    i = wp.tid()
    out[i] = cfl_effective(F[i], jbar[i], mu, kappa, rho, dt_over_dx)


def evaluate_cfl_effective(F, jbar, mu, kappa, rho, dt_over_dx, device="cpu"):
    """Host helper: the guard's per-particle dt c_bound / dx through the Warp function (tests, diagnostics)."""
    F = np.ascontiguousarray(np.asarray(F, np.float32).reshape(-1, 3, 3))
    jb = np.ascontiguousarray(np.asarray(jbar, np.float64).reshape(-1))
    with wp.ScopedDevice(device):
        out = wp.zeros(len(F), dtype=float)
        wp.launch(_cfl_points, len(F), [wp.array(F, dtype=wp.mat33), wp.array(jb, dtype=wp.float64), float(mu),
                                        float(kappa), float(rho), float(dt_over_dx), out])
        return out.numpy()


def wave_modulus_bound_np(F, mu, kappa, jbar=None):
    """float64 numpy reference of ``wave_modulus_bound`` (same floors)."""
    F = np.asarray(F, np.float64)
    J = np.linalg.det(F)
    j = max(J, J_EVAL_FLOOR)
    jb = max(J if jbar is None else float(jbar), J_EVAL_FLOOR)
    s = np.linalg.svd(F, compute_uv=False)
    a = mu*j**(-2/3)
    g = abs(J)/s[-1]
    pp = kappa/2*(1+1/jb**2)
    return a+(pp+5*a*np.sum(F*F)/(9*j*j))*g*g+2*a/(3*j)*max(s[0]*g-j, 0.0)


def evaluate_pk1(F, jbar, mu, kappa, j_domain=(0.2, 5.0), device="cpu"):
    """Host helper: PK1 (float32) and validity of each (F, Jbar) through the Warp function (tests, patch checks)."""
    F = np.ascontiguousarray(np.asarray(F, np.float32).reshape(-1, 3, 3))
    jb = np.ascontiguousarray(np.asarray(jbar, np.float64).reshape(-1))
    with wp.ScopedDevice(device):
        f = wp.array(F, dtype=wp.mat33)
        j = wp.array(jb, dtype=wp.float64)
        out = wp.zeros(len(F), dtype=wp.mat33)
        valid = wp.zeros(len(F), dtype=int)
        wp.launch(_pk1_points, len(F), [f, j, float(mu), float(kappa), wp.float64(j_domain[0]),
                                        wp.float64(j_domain[1]), out, valid])
        return out.numpy(), valid.numpy().astype(bool)
