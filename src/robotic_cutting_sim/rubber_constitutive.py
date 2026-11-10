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
"""
import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

MODEL_ID = "neo_hookean_jbar_v1"
MODEL_VERSION = 1
J_EVAL_FLOOR = 0.05          # finite-evaluation floor only (ADR §5); never a state repair


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
