"""``neo_hookean_jbar_prony_v1`` Warp functions shared by the single-env and the batched rubber solvers
(ADR 0023 §10, §21.2; task R3.1).

Model (isochoric Prony series on the J-bar neo-Hookean of ``rubber_constitutive``, Simo 1987 / Holzapfel 1996):

    S0(C) = mu0 J^-2/3 (I - I1/3 C^-1)                    instantaneous isochoric PK2 (zero at F = I)
    S     = g_inf S0 + sum_i Q_i,   g_inf = 1 - sum_i g_i
    Q_i^{n+1} = a_i Q_i^n + b_i (S0_{n+1} - S0_n),   a_i = exp(-dt/tau_i),   b_i = -g_i (tau_i/dt) expm1(-dt/tau_i)
    P     = P_NH(F, Jbar; mu0, kappa) + F . sum_i (Q_i - g_i S0)          (overstress form, §21.2.1)

``P_NH`` is the frozen ``rubber_constitutive.nh_jbar_pk1`` (unchanged; P_NH = F S0 + p(Jbar) cof F, so the overstress
form equals F S + p cof F). Numerics (§21.2.3): F is the float32 state (with ``f64_shadow_v1`` its rounded shadow);
S0, C^-1 = cof^T cof / J^2 (J floored at 0.05 for evaluation only, as §5), the Prony update and the overstress are
float64; Q_i and S0_prev are float64 per particle (``vec6d``: xx, yy, zz, yz, xz, xy); a_i, b_i come from a float64
per-env table ``params64`` = [g_inf, (g_i, tau_i, a_i, b_i) x M] computed on the host. The overstress is multiplied by
F in float64, rounded once to float32 and added to the float32 P_NH. Terms with g_i = 0 are skipped (padding of a
batch with uniform M); if every term has g_i = 0, P is exactly P_NH (no addition: byte-identical to the elastic
stress pass, RUBBER-VISC-09).

Energy densities (§21.2.7, per unit reference volume): stored branch energy |Q_i|^2 / (4 g_i mu0) and the trapezoid
dissipation increment dt (|Q_i^n|^2 + |Q_i^{n+1}|^2) / (4 g_i mu0 tau_i) (Maxwell branches in the strain-like
variable S0 / (2 mu0); |Q|^2 = Q:Q).

Stable-step guard ``stable_step_guard_visco_v1`` (§21.2.6): per particle

    rho c^2 = (1 + sum_i (g_i - b_i)) W(F, Jbar; mu0, kappa) + max(0, lambda_max^G(sum_i (Q_i - g_i S0)))

W = ``rubber_constitutive.wave_modulus_bound`` (float32), lambda_max^G the Gershgorin bound of the symmetric
overstress (float64 inputs; the bound is float32 as the v1 guard). Its premise is checked by the Prony tangent check
(``scripts/rubber/p2_3/visco_tangent_check.py``, RUBBER-VISC-08b) before any dynamic visco run.

The numpy twins at the end (float64) are the references of the tests and of the tangent check.
"""
import math

import numpy as np
import warp as wp

from .rubber_constitutive import J_EVAL_FLOOR, cofactor_d, nh_jbar_pk1, wave_modulus_bound

wp.set_module_options({"enable_backward": False})

MODEL_ID = "neo_hookean_jbar_prony_v1"
MODEL_VERSION = 1
MAX_TERMS = 4
VISCO_GUARD_ID = "stable_step_guard_visco_v1"
P64_G_INF = 0
P64_STRIDE = 4            # per term: g, tau, a, b

vec6d = wp.types.vector(length=6, dtype=wp.float64)


# ----------------------------------------------------------------------------- symmetric 6-vectors (xx yy zz yz xz xy)

@wp.func
def sym6_from_mat(m: wp.mat33d):
    return vec6d(m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1])


@wp.func
def sym6_to_mat(v: vec6d):
    return wp.mat33d(v[0], v[5], v[4], v[5], v[1], v[3], v[4], v[3], v[2])


@wp.func
def sym6_ddot(a: vec6d, b: vec6d):
    """A : B of two symmetric tensors."""
    return (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
            + wp.float64(2.0)*(a[3]*b[3]+a[4]*b[4]+a[5]*b[5]))


@wp.func
def sym6_gershgorin_max(v: vec6d):
    """Gershgorin upper bound of the largest eigenvalue (max over rows of diagonal + |off-diagonals|)."""
    r0 = v[0]+wp.abs(v[5])+wp.abs(v[4])
    r1 = v[1]+wp.abs(v[5])+wp.abs(v[3])
    r2 = v[2]+wp.abs(v[4])+wp.abs(v[3])
    return wp.max(r0, wp.max(r1, r2))


# ----------------------------------------------------------------------------- constitutive functions

@wp.func
def iso_pk2_d(f32: wp.mat33, mu: wp.float64):
    """S0 = mu J^-2/3 (I - I1/3 C^-1), C^-1 = cof^T cof / J^2 (float64 of the float32 F; J floored for evaluation)."""
    f = wp.mat33d(f32)
    j = wp.max(wp.determinant(f), wp.float64(J_EVAL_FLOOR))
    cof = cofactor_d(f)
    cinv = (wp.transpose(cof)*cof)*(wp.float64(1.0)/(j*j))
    i1 = wp.ddot(f, f)
    a = mu*wp.pow(j, wp.float64(-2.0/3.0))
    third = i1/wp.float64(3.0)
    return vec6d(a*(wp.float64(1.0)-third*cinv[0, 0]), a*(wp.float64(1.0)-third*cinv[1, 1]),
                 a*(wp.float64(1.0)-third*cinv[2, 2]), -a*third*cinv[1, 2], -a*third*cinv[0, 2],
                 -a*third*cinv[0, 1])


@wp.func
def prony_update(q: vec6d, a: wp.float64, b: wp.float64, s_new: vec6d, s_old: vec6d):
    """Q^{n+1} = a Q^n + b (S0_{n+1} - S0_n) (exact for S0 linear in time over the step)."""
    return a*q+b*(s_new-s_old)


@wp.func
def overstress_pk1(p_nh: wp.mat33, f32: wp.mat33, over: vec6d):
    """P = P_NH + float32(F . O) (O = sum_i (Q_i - g_i S0), float64 product rounded once)."""
    return p_nh+wp.mat33(wp.mat33d(f32)*sym6_to_mat(over))


@wp.func
def branch_stored_density(q: vec6d, g: wp.float64, mu: wp.float64):
    """|Q|^2 / (4 g mu0): stored energy of one Maxwell branch per unit reference volume."""
    return sym6_ddot(q, q)/(wp.float64(4.0)*g*mu)


@wp.func
def branch_dissipation_increment(q_old: vec6d, q_new: vec6d, g: wp.float64, mu: wp.float64, tau: wp.float64,
                                 dt: wp.float64):
    """dt (|Q^n|^2 + |Q^{n+1}|^2) / (4 g mu0 tau): trapezoid of the branch dissipation rate (>= 0)."""
    return dt*(sym6_ddot(q_old, q_old)+sym6_ddot(q_new, q_new))/(wp.float64(4.0)*g*mu*tau)


@wp.func
def prony_stress_particle(p: int, e: int, f: wp.mat33, jb: wp.float64, mu32: float, kappa32: float,
                          params64: wp.array2d(dtype=wp.float64), M: int, q: wp.array(dtype=vec6d),
                          s_prev: wp.array(dtype=vec6d), dt: wp.float64):
    """Stress pass of one particle (§21.2.2): S0_n, Prony update of every term (in place), S0_prev <- S0_n, overstress
    PK1. Returns (P float32, sum_i Q_i^{n+1}, dissipation density increment, count of negative/non-finite increments)."""
    p_nh = nh_jbar_pk1(f, jb, mu32, kappa32)
    mu = wp.float64(mu32)
    s0 = iso_pk2_d(f, mu)
    ds = s0-s_prev[p]
    over = vec6d()
    qsum = vec6d()
    dens = wp.float64(0.0)
    negative = int(0)
    any_term = int(0)
    for i in range(M):
        col = 1+P64_STRIDE*i
        g = params64[e, col]
        if g != wp.float64(0.0):
            any_term = 1
            tau = params64[e, col+1]
            k = p*M+i
            q_old = q[k]
            q_new = prony_update(q_old, params64[e, col+2], params64[e, col+3], s0, s_prev[p])
            q[k] = q_new
            over += q_new-g*s0
            qsum += q_new
            inc = branch_dissipation_increment(q_old, q_new, g, mu, tau, dt)
            if not (inc >= wp.float64(0.0)):
                negative += 1
            dens += inc
    s_prev[p] = s0
    out = p_nh
    if any_term != 0:
        out = overstress_pk1(p_nh, f, over)
    return out, qsum, dens, negative


@wp.func
def visco_guard_terms(p: int, e: int, params64: wp.array2d(dtype=wp.float64), M: int, q: wp.array(dtype=vec6d),
                      s0: vec6d):
    """(1 + sum_i (g_i - b_i), sum_i (Q_i - g_i S0)) of one particle (terms with g_i = 0 skipped)."""
    factor = wp.float64(1.0)
    over = vec6d()
    for i in range(M):
        col = 1+P64_STRIDE*i
        g = params64[e, col]
        if g != wp.float64(0.0):
            factor += g-params64[e, col+3]
            over += q[p*M+i]-g*s0
    return factor, over


@wp.func
def visco_wave_modulus_bound(f: wp.mat33, jbar: wp.float64, mu: float, kappa: float, factor: wp.float64,
                             over: vec6d):
    """rho c^2 bound of §21.2.6 (float32 result, as the v1 guard)."""
    w = float(factor)*wave_modulus_bound(f, jbar, mu, kappa)
    return w+wp.max(float(sym6_gershgorin_max(over)), 0.0)


@wp.func
def cfl_effective_visco(f: wp.mat33, jbar: wp.float64, mu: float, kappa: float, rho: float, dt_over_dx: float,
                        factor: wp.float64, over: vec6d):
    """dt c_bound,visco / dx of one particle (NaN for non-finite input, so ``not (c <= limit)`` counts it)."""
    return dt_over_dx*wp.sqrt(visco_wave_modulus_bound(f, jbar, mu, kappa, factor, over)/rho)


# ----------------------------------------------------------------------------- host helpers (tests, diagnostics)

@wp.kernel
def _iso_pk2_points(F: wp.array(dtype=wp.mat33), mu: wp.float64, out: wp.array(dtype=vec6d)):
    i = wp.tid()
    out[i] = iso_pk2_d(F[i], mu)


@wp.kernel
def _prony_points(F: wp.array(dtype=wp.mat33), jbar: wp.array(dtype=wp.float64), mu: float, kappa: float,
                  params64: wp.array2d(dtype=wp.float64), M: int, q: wp.array(dtype=vec6d),
                  s_prev: wp.array(dtype=vec6d), dt: wp.float64, stress: wp.array(dtype=wp.mat33),
                  qsum: wp.array(dtype=vec6d), dens: wp.array(dtype=wp.float64), negative: wp.array(dtype=int)):
    i = wp.tid()
    out, qs, d, neg = prony_stress_particle(i, 0, F[i], jbar[i], mu, kappa, params64, M, q, s_prev, dt)
    stress[i] = out
    qsum[i] = qs
    dens[i] = d
    negative[i] = neg


@wp.kernel
def _visco_cfl_points(F: wp.array(dtype=wp.mat33), jbar: wp.array(dtype=wp.float64), mu: float, kappa: float,
                      rho: float, dt_over_dx: float, params64: wp.array2d(dtype=wp.float64), M: int,
                      q: wp.array(dtype=vec6d), s0: wp.array(dtype=vec6d), out: wp.array(dtype=float)):
    i = wp.tid()
    factor, over = visco_guard_terms(i, 0, params64, M, q, s0[i])
    out[i] = cfl_effective_visco(F[i], jbar[i], mu, kappa, rho, dt_over_dx, factor, over)


def evaluate_s0(F, mu, device="cpu"):
    """S0 (vec6d, float64) of each float32 F through the Warp function."""
    F = np.ascontiguousarray(np.asarray(F, np.float32).reshape(-1, 3, 3))
    with wp.ScopedDevice(device):
        out = wp.zeros(len(F), dtype=vec6d)
        wp.launch(_iso_pk2_points, len(F), [wp.array(F, dtype=wp.mat33), wp.float64(mu), out])
        return out.numpy()


def evaluate_prony(F, jbar, mu, kappa, params64_row, q, s_prev, dt, device="cpu"):
    """One stress pass on independent points (each point a 'particle' of env 0): returns P (float32), Q^{n+1}
    [n, M, 6], S0 [n, 6], sum Q, dissipation density increments, negative counts."""
    F = np.ascontiguousarray(np.asarray(F, np.float32).reshape(-1, 3, 3))
    n = len(F)
    row = np.ascontiguousarray(np.asarray(params64_row, np.float64).reshape(1, -1))
    M = (row.shape[1]-1)//P64_STRIDE
    q = np.ascontiguousarray(np.asarray(q, np.float64).reshape(n*max(M, 1), 6)) if M else np.zeros((1, 6))
    with wp.ScopedDevice(device):
        qa = wp.array(q, dtype=vec6d)
        sa = wp.array(np.ascontiguousarray(np.asarray(s_prev, np.float64).reshape(n, 6)), dtype=vec6d)
        stress = wp.zeros(n, dtype=wp.mat33)
        qsum = wp.zeros(n, dtype=vec6d)
        dens = wp.zeros(n, dtype=wp.float64)
        neg = wp.zeros(n, dtype=int)
        wp.launch(_prony_points, n, [wp.array(F, dtype=wp.mat33),
                                     wp.array(np.asarray(jbar, np.float64).reshape(n), dtype=wp.float64), float(mu),
                                     float(kappa), wp.array(row, dtype=wp.float64), M, qa, sa, wp.float64(dt),
                                     stress, qsum, dens, neg])
        return {"P": stress.numpy(), "q": qa.numpy().reshape(n, max(M, 1), 6)[:, :M], "s0": sa.numpy(),
                "qsum": qsum.numpy(), "dissipation_density": dens.numpy(), "negative": neg.numpy()}


def evaluate_visco_cfl(F, jbar, mu, kappa, rho, dt_over_dx, params64_row, q, s0, device="cpu"):
    """Per-point dt c_bound,visco / dx through the Warp guard function."""
    F = np.ascontiguousarray(np.asarray(F, np.float32).reshape(-1, 3, 3))
    n = len(F)
    row = np.ascontiguousarray(np.asarray(params64_row, np.float64).reshape(1, -1))
    M = (row.shape[1]-1)//P64_STRIDE
    q = np.ascontiguousarray(np.asarray(q, np.float64).reshape(n*max(M, 1), 6)) if M else np.zeros((1, 6))
    with wp.ScopedDevice(device):
        out = wp.zeros(n, dtype=float)
        wp.launch(_visco_cfl_points, n, [wp.array(F, dtype=wp.mat33),
                                         wp.array(np.asarray(jbar, np.float64).reshape(n), dtype=wp.float64),
                                         float(mu), float(kappa), float(rho), float(dt_over_dx),
                                         wp.array(row, dtype=wp.float64), M, wp.array(q, dtype=vec6d),
                                         wp.array(np.asarray(s0, np.float64).reshape(n, 6), dtype=vec6d), out])
        return out.numpy()


# ----------------------------------------------------------------------------- coefficients (host, float64)

def prony_coefficients(terms, dt):
    """[(a_i, b_i)] in float64: a_i = exp(-dt/tau_i), b_i = -g_i (tau_i/dt) expm1(-dt/tau_i) (ADR 0023 §10)."""
    dt = float(dt)
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError("dt must be finite and positive")
    return [(math.exp(-dt/float(tau)), -float(g)*(float(tau)/dt)*math.expm1(-dt/float(tau))) for g, tau in terms]


def params64_row(terms, dt):
    """float64 per-env row [g_inf, (g_i, tau_i, a_i, b_i) x M] (``params64``; §21.2.4)."""
    terms = [(float(g), float(tau)) for g, tau in terms]
    row = [1.0-math.fsum(g for g, _ in terms)]
    for (g, tau), (a, b) in zip(terms, prony_coefficients(terms, dt)):
        row += [g, tau, a, b]
    return np.array(row, np.float64)


def param64_names(M):
    names = ["g_inf"]
    for i in range(M):
        names += [f"g{i+1}", f"tau{i+1}_s", f"a{i+1}", f"b{i+1}"]
    return tuple(names)


# ----------------------------------------------------------------------------- numpy twins (float64)

# Warp emits a float literal inside a kernel as float32: ``wp.float64(-2.0/3.0)`` evaluates to -0.66666669 (relative
# 3e-8 from -2/3). The frozen ``rubber_constitutive.nh_jbar_pk1`` / ``iso_energy_density`` and ``iso_pk2_d`` share that
# exponent (so P_NH = F S0 + p cof F holds in the solver's own arithmetic); the twins below use the same value. The
# analytic references (``rubber_visco_reference``) use the exact -2/3 (the difference is ~3e-8 |ln J|).
EXP_M23 = float(np.float32(-2.0/3.0))
J_FLOOR_KERNEL = float(np.float32(J_EVAL_FLOOR))

def sym6(m):
    m = np.asarray(m, np.float64)
    return np.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2], m[..., 1, 2], m[..., 0, 2], m[..., 0, 1]], -1)


def mat_from_sym6(v):
    v = np.asarray(v, np.float64)
    out = np.empty(v.shape[:-1]+(3, 3))
    out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = v[..., 0], v[..., 1], v[..., 2]
    out[..., 1, 2] = out[..., 2, 1] = v[..., 3]
    out[..., 0, 2] = out[..., 2, 0] = v[..., 4]
    out[..., 0, 1] = out[..., 1, 0] = v[..., 5]
    return out


def cofactor_np(F):
    F = np.asarray(F, np.float64)
    c0 = np.cross(F[..., :, 1], F[..., :, 2])
    c1 = np.cross(F[..., :, 2], F[..., :, 0])
    c2 = np.cross(F[..., :, 0], F[..., :, 1])
    return np.stack([c0, c1, c2], -1)


def s0_np(F, mu, exponent=EXP_M23):
    """Matrix S0(F) (float64, same floor and exponent as the Warp function; ``exponent=-2/3`` for the exact model)."""
    F = np.asarray(F, np.float64)
    j = np.maximum(np.linalg.det(F), J_FLOOR_KERNEL)
    cof = cofactor_np(F)
    cinv = np.swapaxes(cof, -1, -2)@cof/(j*j)[..., None, None]
    i1 = np.sum(F*F, axis=(-1, -2))
    a = mu*j**exponent
    return a[..., None, None]*(np.eye(3)-(i1/3.0)[..., None, None]*cinv)


def nh_pk1_np(F, jbar, mu, kappa):
    """float64 twin of ``rubber_constitutive.nh_jbar_pk1`` (no float32 rounding)."""
    F = np.asarray(F, np.float64)
    j = np.maximum(np.linalg.det(F), J_FLOOR_KERNEL)
    jb = np.maximum(np.asarray(jbar, np.float64), J_FLOOR_KERNEL)
    cof = cofactor_np(F)
    i1 = np.sum(F*F, axis=(-1, -2))
    a = mu*j**EXP_M23
    scale = 0.5*kappa*(jb-1.0/jb)-a*i1/(3.0*j)
    return a[..., None, None]*F+scale[..., None, None]*cof


def prony_pk1_np(F, jbar, mu, kappa, terms, coeffs, q_old, s_prev):
    """float64 twin of one stress pass: returns (P, Q_new [M, 3, 3], S0). ``terms`` [(g, tau)], ``coeffs`` [(a, b)],
    ``q_old`` [M, 3, 3], ``s_prev`` [3, 3]."""
    s0 = s0_np(F, mu)
    over = np.zeros((3, 3))
    q_new = []
    for (g, _), (a, b), q in zip(terms, coeffs, q_old):
        qn = a*np.asarray(q)+b*(s0-s_prev)
        q_new.append(qn)
        if g != 0:
            over += qn-g*s0
    return nh_pk1_np(F, jbar, mu, kappa)+np.asarray(F, np.float64)@over, np.array(q_new), s0


def gershgorin_max_np(m):
    m = np.asarray(m, np.float64)
    return float(np.max(np.diag(m)+np.sum(np.abs(m), 1)-np.abs(np.diag(m))))


def visco_wave_modulus_bound_np(F, jbar, mu, kappa, terms, coeffs, q, s0):
    """float64 twin of ``visco_wave_modulus_bound`` (rubber_constitutive.wave_modulus_bound_np for W)."""
    from .rubber_constitutive import wave_modulus_bound_np
    factor = 1.0+sum(g-b for (g, _), (_, b) in zip(terms, coeffs) if g != 0)
    over = sum((qi-g*s0 for (g, _), qi in zip(terms, q) if g != 0), np.zeros((3, 3)))
    return factor*wave_modulus_bound_np(F, mu, kappa, jbar)+max(gershgorin_max_np(over), 0.0)
