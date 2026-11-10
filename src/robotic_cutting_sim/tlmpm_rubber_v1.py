"""Single-env reference rubber solver ``tlmpm_rubber_v1`` (ADR 0023 D1-D6, §5, §7, §9; task R1.2).

Status ``EXPERIMENTAL_NOT_ACCEPTED`` (frozen only at the P2-2 acceptance). Opt-in: nothing in the cutting simulator
imports it. Material ``neo_hookean_jbar_v1`` (``rubber_constitutive``), pure FLIP (pic = 0, anything else refused),
MUSL, grid-consistent positions, no artificial damping in dynamics (the mass-proportional damping impulse exists only
for the labelled static initializer, ``damping_rate`` > 0 refused unless ``initializer=True`` is passed).

One substep (17 launches, ADR §13):

1.  x_start <- x
2.  J-bar pass ``rubber_constitutive.cell_jbar`` (cell CSR gather, float64, deterministic)
3.  stress pass ``_stress``: P = PK1(F, Jbar_cell); J / Jbar outside the domain counts a violation (sticky flag)
3a. (opt-in ``stable_step_guard``, ADR 0023 §7 amendment) ``_cfl_guard``: dt c_bound(F, Jbar) / dx above the declared
    limit counts a violation (sticky ``RUBBER_CFL_EXCEEDED`` in the runtime); off by default (v1 launch sequence)
4.  ``_loads``: impulse = traction + table / fixture contacts (frozen ``_contact_impulse``) [+ initializer damping]
5.  zero grid; frozen ``tlmpm_v5._p2g``; frozen ``tlmpm_v5._grid_update`` (S = 1, slot tables constant 0)
6.  ``_bc`` (reaction pass): nodal Dirichlet groups set the masked velocity components to the position-exact
    prescribed velocity (x(i+1) - x(i)) / dt of the group's interpolated motion; the impulse m (v - v~) and its torque
    about the group origin are accumulated per group in float64 (the measured clamp / grip wrench)
7.  frozen ``_g2p`` (pic = 0); zero gp/gj; frozen ``_musl``; frozen ``_grid_update``; ``_bc`` (no reaction)
8.  frozen ``_deformation`` (F += dt sum v_I (x) grad N_I); frozen ``tlmpm_v6._grid_positions``; frozen ``_advance``

Opt-in solver options (P2-2, ADR 0023 §19; every default is the v1 launch sequence above, bitwise unchanged):

- ``dirichlet="consistent_mass_projection_v1"`` (F8): step 6 is skipped and, after step 7's G2P, the particle
  velocities are projected onto the constraint set in the kinetic-energy (consistent-mass) metric: with
  A = (W M_p W^T) restricted to the constrained (node, component) pairs of each active group (constant in TLMPM,
  inverted once in float64 at construction), mu = A^-1 (p_C - m_C v_presc,C) and v_p -= sum_{I in C} w_pI mu_I;
  the MUSL pass is then repeated, so the MUSL grid velocity meets the constraint exactly (up to float32) and the
  constrained system stays symplectic Euler on the constraint subspace (no projection dissipation). Reaction impulse
  -m_I mu_I per constrained node and component; constraint work = impulse . prescribed velocity;
- ``traction_work="grid_displacement_v1"`` (F9): the traction work is accumulated after step 8 from the
  grid-consistent particle displacement, j_p . (x_p(end) - x_p(start)) / dt, instead of j_p . v_p(start);
- ``state_precision="f64_shadow_v1"`` (F10): F and x are accumulated in float64 shadows (F64 += dt sum v_I (x)
  grad N_I, x64 += dt sum w_pI v_I) and rounded once to the float32 state every substep, instead of the frozen
  float32 ``_deformation`` / ``_grid_positions`` (float32 increments below half an ulp of F ~ 1 are otherwise lost).

Frozen kernels are launched unchanged (``tlmpm_v5.py``, ``tlmpm_v6.py`` byte-identical). Group motions are rigid
(slerp of the rotation and linear translation, x = R(s) X + b(s)) or affine (x = A(s) X + b(s), linear in s; patch /
homogeneous tests), s = substep index / substeps per control step, reference node positions X in float64.
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from . import tlmpm_v5 as v5
from . import rubber_constitutive as rc
from .rubber_constitutive import (cfl_effective, det_d, in_domain, nh_jbar_pk1, iso_energy_density,
                                  vol_energy_density)
from .tlmpm_v6 import _grid_positions

wp.set_module_options({"enable_backward": False})

SOLVER_NAME = "tlmpm_rubber_v1"
STATUS = "EXPERIMENTAL_NOT_ACCEPTED"
ADR = "docs/adr/0023-rubber-material-v1.md"
GROUP_RIGID, GROUP_AFFINE = 0, 1
DIAG_COLUMNS = ("nonfinite", "J_min", "J_max", "volume_ratio", "Jbar_min", "Jbar_max", "frac_cells_jbar_dev_gt_0_05",
                "elastic_iso", "elastic_vol", "elastic", "kinetic_particle", "kinetic_grid", "gravity",
                "max_speed_m_s", "cells", "particles", "max_grid_speed_m_s", "cfl_eff_max")
NDIAG = len(DIAG_COLUMNS)
DIAG_GROUPS = 64
_HERE = Path(__file__).resolve().parent
SOURCES = ("tlmpm_rubber_v1.py", "rubber_constitutive.py")
FROZEN_KERNELS = ("tlmpm_v5._p2g", "tlmpm_v5._grid_update", "tlmpm_v5._g2p", "tlmpm_v5._musl",
                  "tlmpm_v5._deformation", "tlmpm_v5._advance", "tlmpm_v5._contact_impulse", "tlmpm_v5._box_sdf",
                  "tlmpm_v6._grid_positions")


def source_sha256():
    out = {name: hashlib.sha256((_HERE/name).read_bytes()).hexdigest() for name in SOURCES}
    out.update({name: hashlib.sha256((_HERE/name).read_bytes()).hexdigest() for name in ("tlmpm_v5.py", "tlmpm_v6.py")})
    return out


# ----------------------------------------------------------------------------- shared functions (single and batch)

@wp.func
def group_target(kind: int, q0: wp.quatd, q1: wp.quatd, b0: wp.vec3d, b1: wp.vec3d, A0: wp.mat33d, A1: wp.mat33d,
                 s: wp.float64, X: wp.vec3d):
    """Target position of reference point X at interpolation parameter s (rigid: slerp; affine: linear)."""
    b = b0+s*(b1-b0)
    M = A0+s*(A1-A0)
    if kind == GROUP_RIGID:
        M = wp.quat_to_matrix(wp.quat_slerp(q0, q1, s))
    return M*X+b


@wp.func
def apply_mask(v: wp.vec3, vp: wp.vec3d, mask: int):
    out = v
    if (mask & 1) != 0:
        out[0] = float(vp[0])
    if (mask & 2) != 0:
        out[1] = float(vp[1])
    if (mask & 4) != 0:
        out[2] = float(vp[2])
    return out


@wp.func
def bc_reaction(m: float, v_old: wp.vec3, v_new: wp.vec3, x1: wp.vec3d, o1: wp.vec3d):
    """Impulse j = m (v_new - v_old) applied to the material by the group and its moment about the group origin."""
    j = wp.float64(m)*(wp.vec3d(v_new)-wp.vec3d(v_old))
    tau = wp.cross(x1-o1, j)
    return wp.spatial_vectord(j[0], j[1], j[2], tau[0], tau[1], tau[2])


@wp.func
def bc_work(m: float, v_old: wp.vec3, v_new: wp.vec3):
    """Work of the constraint on the material: impulse j = m (v_new - v_old) times the prescribed velocity v_new
    (zero on unmasked components, where j = 0; zero for a fixed group). Numerical constraint dissipation is
    therefore NOT counted as external work and shows up in the energy balance."""
    return wp.float64(m)*wp.dot(wp.vec3d(v_new)-wp.vec3d(v_old), wp.vec3d(v_new))


@wp.func
def contact_load(xp: wp.vec3, vp: wp.vec3, mp: float, dt: float, radius: float, stiffness: float, friction: float,
                 use_plane: int, plane: wp.vec4, box_start: int, box_stop: int, box_c: wp.array(dtype=wp.vec3),
                 box_h: wp.array(dtype=wp.vec3)):
    """Table plane and fixture boxes with the frozen projection/Coulomb helper (as ``fixture_contacts``)."""
    imp = wp.vec3(0.0)
    for b in range(box_start, box_stop):
        hit = v5._box_sdf(xp-box_c[b], box_h[b])
        imp = imp+v5._contact_impulse(hit[3]-radius, wp.vec3(hit[0], hit[1], hit[2]), vp, mp, dt, stiffness, friction)
    if use_plane != 0:
        n = wp.vec3(plane[0], plane[1], plane[2])
        imp = imp+v5._contact_impulse(wp.dot(n, xp)+plane[3]-radius, n, vp, mp, dt, stiffness, friction)
    return imp


# ----------------------------------------------------------------------------- single-env kernels

@wp.kernel
def _stress(F: wp.array(dtype=wp.mat33), nodes: wp.array2d(dtype=int), jbar: wp.array(dtype=wp.float64),
            stress: wp.array(dtype=wp.mat33), mu: float, kappa: float, lo: wp.float64, hi: wp.float64,
            violations: wp.array(dtype=int)):
    p = wp.tid()
    f = F[p]
    jb = jbar[nodes[p, 0]]
    if not (in_domain(det_d(f), lo, hi) and in_domain(jb, lo, hi)):
        wp.atomic_add(violations, 0, 1)
    stress[p] = nh_jbar_pk1(f, jb, mu, kappa)


@wp.kernel
def _cfl_guard(F: wp.array(dtype=wp.mat33), nodes: wp.array2d(dtype=int), jbar: wp.array(dtype=wp.float64),
               mass: wp.array(dtype=float), mu: float, kappa: float, volume: float, dt_over_dx: float, limit: float,
               cfl_violations: wp.array(dtype=int)):
    """Stable-step guard (opt-in): count particles whose dt c_bound / dx exceeds ``limit`` (or is not finite).
    Read-only on the state; the atomic touches the counter only on a violation."""
    p = wp.tid()
    c = cfl_effective(F[p], jbar[nodes[p, 0]], mu, kappa, mass[p]/volume, dt_over_dx)
    if not (c <= limit):
        wp.atomic_add(cfl_violations, 0, 1)


@wp.kernel
def _loads(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
           impulse: wp.array(dtype=wp.vec3), particle_traction: wp.array(dtype=int),
           traction_force: wp.array(dtype=wp.vec3), traction_mass: wp.array(dtype=float), dt: float, damping: float,
           radius: float, stiffness: float, friction: float, use_plane: int, plane: wp.vec4, boxes: int,
           box_c: wp.array(dtype=wp.vec3), box_h: wp.array(dtype=wp.vec3), work: wp.array(dtype=wp.float64)):
    p = wp.tid()
    xp, vp, mp = x[p], v[p], mass[p]
    imp = contact_load(xp, vp, mp, dt, radius, stiffness, friction, use_plane, plane, 0, boxes, box_c, box_h)
    t = particle_traction[p]
    if t >= 0:
        jt = traction_force[t]*(dt*mp/traction_mass[t])
        imp = imp+jt
        wp.atomic_add(work, 0, wp.dot(wp.vec3d(jt), wp.vec3d(vp)))
    if damping > 0.0:
        imp = imp-(damping*dt*mp)*vp
    impulse[p] = imp


@wp.kernel
def _bc(node_group: wp.array(dtype=int), node_mask: wp.array(dtype=int), node_X: wp.array(dtype=wp.vec3d),
        gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), index: wp.array(dtype=int), substeps: int,
        dt: wp.float64, kind: wp.array(dtype=int), active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd),
        q1: wp.array(dtype=wp.quatd), b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d),
        A0: wp.array(dtype=wp.mat33d), A1: wp.array(dtype=wp.mat33d), origin: wp.array(dtype=wp.vec3d),
        reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array(dtype=wp.float64), record: int):
    i = wp.tid()
    g = node_group[i]
    if g >= 0:
        if active[g] != 0:
            k = index[0]
            s0 = wp.float64(k)/wp.float64(substeps)
            s1 = wp.float64(k+1)/wp.float64(substeps)
            X = node_X[i]
            x0 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s0, X)
            x1 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s1, X)
            v = new_v[i]
            vn = apply_mask(v, (x1-x0)/dt, node_mask[i])
            new_v[i] = vn
            if record != 0:
                o1 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s1, origin[g])
                wp.atomic_add(reaction, g, bc_reaction(gm[i], v, vn, x1, o1))
                wp.atomic_add(work, 0, bc_work(gm[i], v, vn))


# ----------------------------------------------------------------------------- opt-in options (P2-2, F8 / F9 / F10)

DIRICHLET_MODES = ("lumped_v1", "consistent_mass_projection_v1")
TRACTION_WORK_MODES = ("particle_velocity_v1", "grid_displacement_v1")
STATE_PRECISION_MODES = ("float32_v1", "f64_shadow_v1")
PROJECTION_MAX_BLOCK = 4096


@wp.kernel
def _proj_rhs(row_node: wp.array(dtype=int), row_comp: wp.array(dtype=int), row_group: wp.array(dtype=int),
              node_X: wp.array(dtype=wp.vec3d), gm: wp.array(dtype=float), gp: wp.array(dtype=wp.vec3),
              index: wp.array(dtype=int), substeps: int, dt: wp.float64, kind: wp.array(dtype=int),
              active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd), q1: wp.array(dtype=wp.quatd),
              b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d), A0: wp.array(dtype=wp.mat33d),
              A1: wp.array(dtype=wp.mat33d), rhs: wp.array(dtype=wp.float64), vpresc: wp.array(dtype=wp.float64)):
    """F8: per constrained (node, component) row: rhs = p_I[k] - m_I v_presc,I[k] (MUSL momentum of the projected
    G2P velocities), v_presc the position-exact prescribed velocity of this substep (as ``_bc``)."""
    r = wp.tid()
    g = row_group[r]
    i = row_node[r]
    k = row_comp[r]
    rhs[r] = wp.float64(0.0)
    vpresc[r] = wp.float64(0.0)
    if active[g] != 0:
        s0 = wp.float64(index[0])/wp.float64(substeps)
        s1 = wp.float64(index[0]+1)/wp.float64(substeps)
        X = node_X[i]
        x0 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s0, X)
        x1 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s1, X)
        vp = (x1-x0)/dt
        vpresc[r] = vp[k]
        rhs[r] = wp.float64(gp[i][k])-wp.float64(gm[i])*vp[k]


@wp.kernel
def _proj_solve(row_block: wp.array(dtype=int), row_local: wp.array(dtype=int), block_offset: wp.array(dtype=int),
                block_size: wp.array(dtype=int), inv_offset: wp.array(dtype=int), inv: wp.array(dtype=wp.float64),
                rhs: wp.array(dtype=wp.float64), row_node: wp.array(dtype=int), row_comp: wp.array(dtype=int),
                row_group: wp.array(dtype=int), active: wp.array(dtype=int), mu_node: wp.array2d(dtype=wp.float64),
                mu: wp.array(dtype=wp.float64)):
    """F8: mu = A^-1 rhs per block (dense float64 inverse of the constant consistent-mass block; fixed-order gather,
    deterministic on any device); scattered to the node's component (each (node, component) is in one block)."""
    r = wp.tid()
    b = row_block[r]
    acc = wp.float64(0.0)
    if active[row_group[r]] != 0:
        n = block_size[b]
        base = inv_offset[b]+row_local[r]*n
        off = block_offset[b]
        for c in range(n):
            acc += inv[base+c]*rhs[off+c]
    mu[r] = acc
    mu_node[row_node[r], row_comp[r]] = acc


@wp.kernel
def _proj_particles(v: wp.array(dtype=wp.vec3), nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
                    mu_node: wp.array2d(dtype=wp.float64)):
    """F8: v_p -= sum_I w_pI mu_I (mu is zero on unconstrained nodes and components)."""
    p = wp.tid()
    d = wp.vec3d()
    for c in range(8):
        i = nodes[p, c]
        w = wp.float64(weights[p, c])
        d += wp.vec3d(w*mu_node[i, 0], w*mu_node[i, 1], w*mu_node[i, 2])
    v[p] = wp.vec3(wp.vec3d(v[p])-d)


@wp.kernel
def _proj_reaction(row_node: wp.array(dtype=int), row_comp: wp.array(dtype=int), row_group: wp.array(dtype=int),
                   node_X: wp.array(dtype=wp.vec3d), gm: wp.array(dtype=float), mu: wp.array(dtype=wp.float64),
                   vpresc: wp.array(dtype=wp.float64), index: wp.array(dtype=int), substeps: int,
                   kind: wp.array(dtype=int), active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd),
                   q1: wp.array(dtype=wp.quatd), b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d),
                   A0: wp.array(dtype=wp.mat33d), A1: wp.array(dtype=wp.mat33d), origin: wp.array(dtype=wp.vec3d),
                   reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array(dtype=wp.float64)):
    """F8: impulse on the material -m_I mu_I e_k, its moment about the group origin (lever: target node position, as
    ``bc_reaction``) and the constraint work impulse . v_presc."""
    r = wp.tid()
    g = row_group[r]
    if active[g] != 0:
        i = row_node[r]
        s1 = wp.float64(index[0]+1)/wp.float64(substeps)
        x1 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s1, node_X[i])
        o1 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s1, origin[g])
        j = wp.vec3d()
        j[row_comp[r]] = -wp.float64(gm[i])*mu[r]
        tau = wp.cross(x1-o1, j)
        wp.atomic_add(reaction, g, wp.spatial_vectord(j[0], j[1], j[2], tau[0], tau[1], tau[2]))
        wp.atomic_add(work, 0, j[row_comp[r]]*vpresc[r])


@wp.kernel
def _traction_work_grid(x: wp.array(dtype=wp.vec3), x_start: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                        particle_traction: wp.array(dtype=int), traction_force: wp.array(dtype=wp.vec3),
                        traction_mass: wp.array(dtype=float), work: wp.array(dtype=wp.float64)):
    """F9: traction work j_p . (x_p(end) - x_p(start)) / dt = f_t (m_p / M_t) . dx_p (grid-consistent displacement)."""
    p = wp.tid()
    t = particle_traction[p]
    if t >= 0:
        f = wp.vec3d(traction_force[t])*(wp.float64(mass[p])/wp.float64(traction_mass[t]))
        wp.atomic_add(work, 0, wp.dot(f, wp.vec3d(x[p])-wp.vec3d(x_start[p])))


@wp.kernel
def _deformation_f64(F: wp.array(dtype=wp.mat33), F64: wp.array(dtype=wp.mat33d), nodes: wp.array2d(dtype=int),
                     gradients: wp.array2d(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3), dt: wp.float64):
    """F10: F64 += dt sum_I v_I (x) grad N_I in float64 (the frozen ``_deformation`` formula), F = float32(F64)."""
    p = wp.tid()
    grad = wp.mat33d()
    for c in range(8):
        grad += wp.outer(wp.vec3d(velocity[nodes[p, c]]), wp.vec3d(gradients[p, c]))
    f = F64[p]+dt*grad
    F64[p] = f
    F[p] = wp.mat33(f)


@wp.kernel
def _positions_f64(x: wp.array(dtype=wp.vec3), x64: wp.array(dtype=wp.vec3d), nodes: wp.array2d(dtype=int),
                   weights: wp.array2d(dtype=float), velocity: wp.array(dtype=wp.vec3), dt: wp.float64):
    """F10: x64 += dt sum_I w_pI v_I (the ``tlmpm_v6._grid_positions`` formula), x = float32(x64)."""
    p = wp.tid()
    vg = wp.vec3d()
    for c in range(8):
        vg += wp.float64(weights[p, c])*wp.vec3d(velocity[nodes[p, c]])
    y = x64[p]+dt*vg
    x64[p] = y
    x[p] = wp.vec3(y)


def projection_blocks(scene, mass_value):
    """F8: constant consistent-mass blocks A = m sum_p w_pI w_pJ over the constrained (node, component) pairs of each
    group, and their float64 inverses (rows ordered by group, component, node id)."""
    W = {}
    nodes, weights = scene.nodes, scene.weights.astype(np.float64)
    rows, blocks = [], []
    for g in range(len(scene.groups)):
        for k in range(3):
            sel = np.flatnonzero((scene.node_group == g) & ((scene.node_mask >> k) & 1 == 1))
            if sel.size == 0:
                continue
            if sel.size > PROJECTION_MAX_BLOCK:
                raise ValueError(f"consistent_mass_projection_v1: block of {sel.size} constrained nodes > "
                                 f"{PROJECTION_MAX_BLOCK} (dense inverse limit)")
            local = {int(n): i for i, n in enumerate(sel)}
            touching = np.flatnonzero(np.isin(nodes, sel).any(1))
            A = np.zeros((sel.size, sel.size))
            for p in touching:
                idx = [(local[int(n)], w) for n, w in zip(nodes[p], weights[p]) if int(n) in local]
                for a, wa in idx:
                    for b, wb in idx:
                        A[a, b] += float(mass_value)*wa*wb
            np.linalg.cholesky(A)          # SPD check (raises if a constrained node has no mass support)
            blocks.append({"group": g, "comp": k, "nodes": sel, "inv": np.linalg.inv(A),
                           "cond": float(np.linalg.cond(A))})
    return blocks


# ----------------------------------------------------------------------------- segmented diagnostics (shared)

@wp.kernel
def diag_particles(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33),
                   mass: wp.array(dtype=float), reference: wp.array(dtype=wp.vec3), gravity: wp.vec3, volume: float,
                   dparams: wp.array2d(dtype=float), particle_offset: wp.array(dtype=int), groups: int,
                   nodes: wp.array2d(dtype=int), jbar: wp.array(dtype=wp.float64), dt_over_dx: float,
                   partials: wp.array2d(dtype=wp.float64)):
    """Thread (e, g) strides over env e's particles: nonfinite, J min/max/sum, isochoric energy, particle KE,
    gravity potential relative to the reference, max speed, max dt c_bound / dx of the stable-step guard (end of the
    control step; float64 accumulation, fixed order: deterministic on any device)."""
    e, grp = wp.tid()
    nonfinite = wp.float64(0.0)
    jmin = wp.float64(1.0e30)
    jmax = wp.float64(-1.0e30)
    jsum = wp.float64(0.0)
    eiso = wp.float64(0.0)
    ke = wp.float64(0.0)
    egrav = wp.float64(0.0)
    vmax = wp.float64(0.0)
    cflmax = wp.float64(0.0)
    mu = dparams[e, 0]
    kappa = dparams[e, 1]
    vol = wp.float64(volume)
    g = wp.vec3d(gravity)
    for p in range(particle_offset[e]+grp, particle_offset[e+1], groups):
        xp, vp, fp = x[p], v[p], F[p]
        ok = True
        for a in range(3):
            if not wp.isfinite(xp[a]) or not wp.isfinite(vp[a]):
                ok = False
            for b in range(3):
                if not wp.isfinite(fp[a, b]):
                    ok = False
        if ok:
            j = det_d(fp)
            jmin = wp.min(jmin, j)
            jmax = wp.max(jmax, j)
            jsum += j
            eiso += vol*iso_energy_density(fp, mu)
            vd = wp.vec3d(vp)
            m = wp.float64(mass[p])
            ke += wp.float64(0.5)*m*wp.dot(vd, vd)
            egrav -= m*wp.dot(g, wp.vec3d(xp)-wp.vec3d(reference[p]))
            vmax = wp.max(vmax, wp.sqrt(wp.dot(vd, vd)))
            cflmax = wp.max(cflmax, wp.float64(cfl_effective(fp, jbar[nodes[p, 0]], mu, kappa, mass[p]/volume,
                                                             dt_over_dx)))
        else:
            nonfinite = wp.float64(1.0)
    row = e*groups+grp
    partials[row, 0] = nonfinite
    partials[row, 1] = jmin
    partials[row, 2] = jmax
    partials[row, 3] = jsum
    partials[row, 4] = eiso
    partials[row, 5] = ke
    partials[row, 6] = egrav
    partials[row, 7] = vmax
    partials[row, 8] = cflmax


@wp.kernel
def diag_cells(jbar: wp.array(dtype=wp.float64), cell_node: wp.array(dtype=int), cell_offsets: wp.array(dtype=int),
               cell_env_offset: wp.array(dtype=int), groups: int, dparams: wp.array2d(dtype=float), volume: float,
               partials: wp.array2d(dtype=wp.float64)):
    """Thread (e, g) strides over env e's cells: Jbar min/max, volumetric energy, count |Jbar - 1| > 0.05."""
    e, grp = wp.tid()
    jmin = wp.float64(1.0e30)
    jmax = wp.float64(-1.0e30)
    evol = wp.float64(0.0)
    dev = wp.float64(0.0)
    kappa = dparams[e, 1]
    for c in range(cell_env_offset[e]+grp, cell_env_offset[e+1], groups):
        jb = jbar[cell_node[c]]
        jmin = wp.min(jmin, jb)
        jmax = wp.max(jmax, jb)
        evol += wp.float64(cell_offsets[c+1]-cell_offsets[c])*wp.float64(volume)*vol_energy_density(jb, kappa)
        if wp.abs(jb-wp.float64(1.0)) > wp.float64(0.05):
            dev += wp.float64(1.0)
    row = e*groups+grp
    partials[row, 0] = jmin
    partials[row, 1] = jmax
    partials[row, 2] = evol
    partials[row, 3] = dev


@wp.kernel
def diag_nodes(gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), node_offset: wp.array(dtype=int),
               groups: int, partials: wp.array2d(dtype=wp.float64)):
    e, grp = wp.tid()
    ke = wp.float64(0.0)
    vmax = wp.float64(0.0)
    for i in range(node_offset[e]+grp, node_offset[e+1], groups):
        vd = wp.vec3d(new_v[i])
        ke += wp.float64(0.5)*wp.float64(gm[i])*wp.dot(vd, vd)
        if gm[i] > 0.0:
            vmax = wp.max(vmax, wp.sqrt(wp.dot(vd, vd)))
    partials[e*groups+grp, 0] = ke
    partials[e*groups+grp, 1] = vmax


@wp.kernel
def diag_combine(pp: wp.array2d(dtype=wp.float64), pc: wp.array2d(dtype=wp.float64),
                 pn: wp.array2d(dtype=wp.float64), gp_: int, gc: int, gn: int, particle_offset: wp.array(dtype=int),
                 cell_env_offset: wp.array(dtype=int), summary: wp.array2d(dtype=wp.float64)):
    e = wp.tid()
    nonfinite = wp.float64(0.0)
    jmin = wp.float64(1.0e30)
    jmax = wp.float64(-1.0e30)
    jsum = wp.float64(0.0)
    eiso = wp.float64(0.0)
    ke = wp.float64(0.0)
    egrav = wp.float64(0.0)
    vmax = wp.float64(0.0)
    cflmax = wp.float64(0.0)
    for k in range(gp_):
        r = e*gp_+k
        nonfinite = wp.max(nonfinite, pp[r, 0])
        jmin = wp.min(jmin, pp[r, 1])
        jmax = wp.max(jmax, pp[r, 2])
        jsum += pp[r, 3]
        eiso += pp[r, 4]
        ke += pp[r, 5]
        egrav += pp[r, 6]
        vmax = wp.max(vmax, pp[r, 7])
        cflmax = wp.max(cflmax, pp[r, 8])
    jbmin = wp.float64(1.0e30)
    jbmax = wp.float64(-1.0e30)
    evol = wp.float64(0.0)
    dev = wp.float64(0.0)
    for k in range(gc):
        r = e*gc+k
        jbmin = wp.min(jbmin, pc[r, 0])
        jbmax = wp.max(jbmax, pc[r, 1])
        evol += pc[r, 2]
        dev += pc[r, 3]
    keg = wp.float64(0.0)
    vgrid = wp.float64(0.0)
    for k in range(gn):
        keg += pn[e*gn+k, 0]
        vgrid = wp.max(vgrid, pn[e*gn+k, 1])
    n = wp.float64(particle_offset[e+1]-particle_offset[e])
    cells = wp.float64(cell_env_offset[e+1]-cell_env_offset[e])
    summary[e, 0] = nonfinite
    summary[e, 1] = jmin
    summary[e, 2] = jmax
    summary[e, 3] = jsum/n
    summary[e, 4] = jbmin
    summary[e, 5] = jbmax
    summary[e, 6] = dev/cells
    summary[e, 7] = eiso
    summary[e, 8] = evol
    summary[e, 9] = eiso+evol
    summary[e, 10] = ke
    summary[e, 11] = keg
    summary[e, 12] = egrav
    summary[e, 13] = vmax
    summary[e, 14] = cells
    summary[e, 15] = n
    summary[e, 16] = vgrid
    summary[e, 17] = cflmax


@wp.kernel
def markers_kernel(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), marker_offsets: wp.array(dtype=int),
                   marker_particles: wp.array(dtype=int), out_x: wp.array(dtype=wp.vec3d),
                   out_v: wp.array(dtype=wp.vec3d)):
    """Cross-section centroids (position, mean particle velocity) in float64, one thread per marker."""
    m = wp.tid()
    sx = wp.vec3d()
    sv = wp.vec3d()
    start = marker_offsets[m]
    stop = marker_offsets[m+1]
    for k in range(start, stop):
        p = marker_particles[k]
        sx += wp.vec3d(x[p])
        sv += wp.vec3d(v[p])
    inv = wp.float64(1.0)/wp.float64(stop-start)
    out_x[m] = sx*inv
    out_v[m] = sv*inv


class SegmentedRubberDiagnostics:
    """Per-env diagnostics over flat arrays (single solver: one env). ``launch()`` allocates nothing."""

    def __init__(self, solver, particle_counts, cell_counts, node_counts):
        self.solver = solver
        d = solver.device
        # fixed reduction width (independent of the env size): the single reference and the batch then sum every
        # env's partials in the same order, so their diagnostics are bitwise equal on CPU (L0)
        self.gp = self.gc = self.gn = DIAG_GROUPS
        N = len(particle_counts)
        with wp.ScopedDevice(d):
            self.pp = wp.zeros((N*self.gp, 9), dtype=wp.float64)
            self.pc = wp.zeros((N*self.gc, 4), dtype=wp.float64)
            self.pn = wp.zeros((N*self.gn, 2), dtype=wp.float64)
            self.summary = wp.zeros((N, NDIAG), dtype=wp.float64)
            self.marker_x = wp.zeros(len(solver.marker_offsets_host)-1, dtype=wp.vec3d)
            self.marker_v = wp.zeros(len(solver.marker_offsets_host)-1, dtype=wp.vec3d)

    def launch(self):
        s = self.solver
        N = s.num_envs
        with wp.ScopedDevice(s.device):
            wp.launch(rc.cell_jbar, s.cells, [s.cell_node, s.cell_offsets, s.cell_particles, s.F, s.jbar])
            wp.launch(diag_particles, (N, self.gp), [s.x, s.v, s.F, s.mass, s.reference, s.gravity, s.volume,
                                                     s.dparams, s.particle_offset, self.gp, s.nodes, s.jbar,
                                                     s.dt/s.dx, self.pp])
            wp.launch(diag_cells, (N, self.gc), [s.jbar, s.cell_node, s.cell_offsets, s.cell_env_offset, self.gc,
                                                 s.dparams, s.volume, self.pc])
            wp.launch(diag_nodes, (N, self.gn), [s.gm, s.new_v, s.node_offset, self.gn, self.pn])
            wp.launch(diag_combine, N, [self.pp, self.pc, self.pn, self.gp, self.gc, self.gn, s.particle_offset,
                                        s.cell_env_offset, self.summary])
            wp.launch(markers_kernel, len(self.marker_x), [s.x, s.v, s.marker_offsets, s.marker_particles,
                                                           self.marker_x, self.marker_v])

    def read(self):
        self.launch()
        rows = self.summary.numpy()
        return [dict(zip(DIAG_COLUMNS, map(float, r))) for r in rows]


# ----------------------------------------------------------------------------- solver

class TLMPMRubber:
    """Single-env reference solver (see the module docstring).

    ``scene``: :class:`rubber_scene.RubberStripScene`; ``material``: :class:`rubber_material.MaterialValues`;
    ``dt`` material time step; ``substeps``: substeps per control step (the group interpolation parameter is
    index / substeps). ``table_plane`` (unit normal, offset; inward normal) and ``fixture_boxes`` ([B, 2, 3]
    centres, half extents) enable contacts with friction ``material.friction_table`` (P2-4 validates them).
    ``stable_step_guard``: None (off, the v1 launch sequence) or the limit of dt c_bound / dx (ADR 0023 §7 amendment;
    ``rubber_constitutive.STABLE_STEP_GUARD_DEFAULT_LIMIT`` = 1.0); violations are counted in ``cfl_violations``."""
    solver_name = SOLVER_NAME
    status = STATUS
    num_envs = 1

    def __init__(self, scene, material, dt, *, substeps=1, gravity=(0.0, 0.0, -9.81), device="cpu", pic=0.0,
                 j_domain=(0.2, 5.0), contact_stiffness=0.1, contact_radius_scale=0.9, table_plane=None,
                 fixture_boxes=None, stable_step_guard=None, dirichlet="lumped_v1",
                 traction_work="particle_velocity_v1", state_precision="float32_v1"):
        from .rubber_material import mass_per_particle
        for name, value, modes in (("dirichlet", dirichlet, DIRICHLET_MODES),
                                   ("traction_work", traction_work, TRACTION_WORK_MODES),
                                   ("state_precision", state_precision, STATE_PRECISION_MODES)):
            if value not in modes:
                raise ValueError(f"{name}={value!r} not in {modes}")
        self.dirichlet, self.traction_work, self.state_precision = dirichlet, traction_work, state_precision
        if pic != 0.0:
            raise ValueError("tlmpm_rubber_v1 refuses pic != 0: rubber runs pure FLIP (ADR 0023 §2 E3, D4)")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not (isinstance(substeps, (int, np.integer)) and substeps >= 1):
            raise ValueError("substeps must be a positive integer")
        lo, hi = map(float, j_domain)
        if not 0 < lo < 1 < hi:
            raise ValueError("j_domain must satisfy 0 < lo < 1 < hi")
        if stable_step_guard is not None:
            stable_step_guard = float(stable_step_guard)
            if not (np.isfinite(stable_step_guard) and stable_step_guard > 0):
                raise ValueError("stable_step_guard must be None (off) or a positive finite CFL limit")
        self.cfl_guard_limit = stable_step_guard
        wp.init()
        self.scene, self.material, self.device = scene, material, str(wp.get_device(device))
        self.dt, self.substeps, self.pic = dt, int(substeps), 0.0
        self.j_domain = (lo, hi)
        self.gravity = wp.vec3(*map(float, gravity))
        self.gravity_tuple = tuple(map(float, gravity))
        self.spacing, self.dx = float(scene.spacing), float(scene.dx)
        self.volume = float(scene.volume)
        self.radius = float(contact_radius_scale)*self.spacing*0.5
        self.contact_stiffness = float(contact_stiffness)
        self.damping_rate = 0.0
        self.n = int(scene.n)
        self.node_count = int(scene.node_count)
        self.cells = len(scene.cell_node)
        self.G = len(scene.groups)
        self.Gt = len(scene.traction_groups)
        self.mass_value = mass_per_particle(material.density, self.spacing)
        plane = np.zeros(4, np.float32) if table_plane is None else np.asarray(table_plane, np.float32)
        if table_plane is not None and (plane.shape != (4,) or not np.isclose(np.linalg.norm(plane[:3]), 1.0)):
            raise ValueError("table_plane needs a unit inward normal and an offset")
        self.use_plane = int(table_plane is not None)
        self.plane = wp.vec4(*plane.tolist())
        boxes = np.zeros((0, 2, 3), np.float32) if fixture_boxes is None else np.asarray(fixture_boxes, np.float32
                                                                                        ).reshape(-1, 2, 3)
        if np.any(boxes[:, 1] <= 0) or not np.isfinite(boxes).all():
            raise ValueError("fixture boxes need finite centres and positive half extents")
        self.boxes = len(boxes)
        self.marker_offsets_host = scene.marker_offsets
        kind = np.array([GROUP_AFFINE if g["kind"] == "affine" else GROUP_RIGID for g in scene.groups] or [0], np.int32)
        G1 = max(self.G, 1)
        tmass = np.array([float(material.density)*0 for _ in range(max(self.Gt, 1))], np.float64)
        for t in range(self.Gt):
            tmass[t] = float(np.count_nonzero(scene.particle_traction == t))*float(self.mass_value)
        with wp.ScopedDevice(self.device):
            self.reference = wp.array(scene.X, dtype=wp.vec3)
            self.x = wp.array(scene.X, dtype=wp.vec3)
            self.x_start = wp.zeros(self.n, dtype=wp.vec3)
            self.v = wp.zeros(self.n, dtype=wp.vec3)
            self.mass = wp.array(np.full(self.n, self.mass_value, np.float32), dtype=float)
            self.F = wp.array(np.tile(np.eye(3, dtype=np.float32), (self.n, 1, 1)), dtype=wp.mat33)
            self.stress = wp.zeros(self.n, dtype=wp.mat33)
            self.impulse = wp.zeros(self.n, dtype=wp.vec3)
            self.nodes = wp.array(scene.nodes, dtype=int)
            self.weights = wp.array(scene.weights, dtype=float)
            self.gradients = wp.array(scene.gradients, dtype=wp.vec3)
            self.slot = wp.zeros((self.n, 8), dtype=int)               # S = 1: constant slot 0
            self.slot_code = wp.zeros(self.node_count, dtype=int)      # constant code 0 (no cutting)
            self.gm = wp.zeros(self.node_count, dtype=float)
            self.gp = wp.zeros(self.node_count, dtype=wp.vec3)
            self.gj = wp.zeros(self.node_count, dtype=wp.vec3)
            self.old = wp.zeros(self.node_count, dtype=wp.vec3)
            self.new_v = wp.zeros(self.node_count, dtype=wp.vec3)
            self.jbar = wp.ones(self.node_count, dtype=wp.float64)
            self.cell_node = wp.array(scene.cell_node, dtype=int)
            self.cell_offsets = wp.array(scene.cell_offsets, dtype=int)
            self.cell_particles = wp.array(scene.cell_particles, dtype=int)
            self.node_group = wp.array(scene.node_group, dtype=int)
            self.node_mask = wp.array(scene.node_mask, dtype=int)
            self.node_X = wp.array(scene.node_X, dtype=wp.vec3d)
            self.group_kind = wp.array(kind, dtype=int)
            self.group_active = wp.array(np.ones(G1, np.int32), dtype=int)
            self.group_origin = wp.array(np.array([g["origin"] for g in scene.groups] or [[0, 0, 0]], np.float64),
                                         dtype=wp.vec3d)
            ident_q = np.tile(np.array([0, 0, 0, 1], np.float64), (G1, 1))
            ident_A = np.tile(np.eye(3), (G1, 1, 1))
            self.group_q0 = wp.array(ident_q, dtype=wp.quatd)
            self.group_q1 = wp.array(ident_q, dtype=wp.quatd)
            self.group_b0 = wp.zeros(G1, dtype=wp.vec3d)
            self.group_b1 = wp.zeros(G1, dtype=wp.vec3d)
            self.group_A0 = wp.array(ident_A, dtype=wp.mat33d)
            self.group_A1 = wp.array(ident_A, dtype=wp.mat33d)
            self.reaction = wp.zeros(G1, dtype=wp.spatial_vectord)
            self.work = wp.zeros(1, dtype=wp.float64)
            self.particle_traction = wp.array(scene.particle_traction, dtype=int)
            self.traction_force = wp.zeros(max(self.Gt, 1), dtype=wp.vec3)
            self.traction_mass = wp.array(tmass.astype(np.float32), dtype=float)
            self.box_c = wp.array(boxes[:, 0].copy() if len(boxes) else np.zeros((1, 3), np.float32), dtype=wp.vec3)
            self.box_h = wp.array(boxes[:, 1].copy() if len(boxes) else np.ones((1, 3), np.float32), dtype=wp.vec3)
            self.index = wp.zeros(1, dtype=int)
            self.violations = wp.zeros(1, dtype=int)
            self.cfl_violations = wp.zeros(1, dtype=int)
            self.dparams = wp.array(np.array([[material.mu0, material.kappa]], np.float64).astype(np.float32),
                                    dtype=float)
            self.particle_offset = wp.array(np.array([0, self.n], np.int32), dtype=int)
            self.cell_env_offset = wp.array(np.array([0, self.cells], np.int32), dtype=int)
            self.node_offset = wp.array(np.array([0, self.node_count], np.int32), dtype=int)
            self.marker_offsets = wp.array(scene.marker_offsets, dtype=int)
            self.marker_particles = wp.array(scene.marker_particles, dtype=int)
        self._init_options()
        self.diagnostics = SegmentedRubberDiagnostics(self, [self.n], [self.cells], [self.node_count])
        for module in (v5.__name__, "robotic_cutting_sim.tlmpm_v6", rc.__name__, __name__):
            wp.load_module(module=module, device=self.device)

    def _init_options(self):
        """Arrays of the opt-in options (none for the defaults: the v1 array set is unchanged)."""
        self.projection_receipt = None
        with wp.ScopedDevice(self.device):
            if self.dirichlet == "consistent_mass_projection_v1":
                blocks = projection_blocks(self.scene, self.mass_value)
                rows = sum(len(b["nodes"]) for b in blocks)
                row_node, row_comp, row_group, row_block, row_local = [], [], [], [], []
                block_offset, block_size, inv_offset, inv = [], [], [], []
                off = ioff = 0
                for bi, b in enumerate(blocks):
                    n = len(b["nodes"])
                    row_node += b["nodes"].tolist()
                    row_comp += [b["comp"]]*n
                    row_group += [b["group"]]*n
                    row_block += [bi]*n
                    row_local += list(range(n))
                    block_offset.append(off)
                    block_size.append(n)
                    inv_offset.append(ioff)
                    inv.append(b["inv"].reshape(-1))
                    off += n
                    ioff += n*n
                rows1 = max(rows, 1)
                as_i = lambda a: wp.array(np.asarray(a or [0], np.int32), dtype=int)       # noqa: E731
                self.proj_rows = rows
                self.proj_row_node, self.proj_row_comp = as_i(row_node), as_i(row_comp)
                self.proj_row_group, self.proj_row_block = as_i(row_group), as_i(row_block)
                self.proj_row_local = as_i(row_local)
                self.proj_block_offset, self.proj_block_size = as_i(block_offset), as_i(block_size)
                self.proj_inv_offset = as_i(inv_offset)
                self.proj_inv = wp.array(np.concatenate(inv) if inv else np.zeros(1), dtype=wp.float64)
                self.proj_rhs = wp.zeros(rows1, dtype=wp.float64)
                self.proj_vpresc = wp.zeros(rows1, dtype=wp.float64)
                self.proj_mu = wp.zeros(rows1, dtype=wp.float64)
                self.proj_mu_node = wp.zeros((self.node_count, 3), dtype=wp.float64)
                self.projection_receipt = {"blocks": [{"group": b["group"], "component": "xyz"[b["comp"]],
                                                       "nodes": int(len(b["nodes"])), "condition": b["cond"]}
                                                      for b in blocks], "rows": rows,
                                           "inverse_bytes": int(self.proj_inv.capacity)}
            if self.traction_work == "grid_displacement_v1":
                self.work_scratch = wp.zeros(1, dtype=wp.float64)    # receives the v1 accumulation (discarded)
            if self.state_precision == "f64_shadow_v1":
                self.F64 = wp.array(np.tile(np.eye(3), (self.n, 1, 1)), dtype=wp.mat33d)
                self.x64 = wp.array(self.scene.X.astype(np.float64), dtype=wp.vec3d)

    def sync_shadow(self):
        """F10: set the float64 shadows from the current float32 F and x (after an external state assignment)."""
        if self.state_precision == "f64_shadow_v1":
            self.F64.assign(self.F.numpy().astype(np.float64))
            self.x64.assign(self.x.numpy().astype(np.float64))

    # ------------------------------------------------------------------ inputs

    def set_initializer_damping(self, rate, *, initializer=False):
        """Mass-proportional damping impulse -rate m v dt: ONLY for the labelled static initializer (ADR §7)."""
        rate = float(rate)
        if rate != 0.0 and not initializer:
            raise ValueError("artificial damping is refused in dynamics; only the static initializer may set it")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("damping rate must be finite and >= 0")
        self.damping_rate = rate

    def set_group_motion(self, g, *, rotation=None, translation=None, affine=None, active=None):
        """Set the END of group ``g``'s motion in the current control step (call after ``begin_control_step``,
        which makes every group hold: start = end = previous end). The map is x = R X + b (rigid) or
        x = A X + b (affine), relative to the reference configuration."""
        if not 0 <= g < self.G:
            raise IndexError(f"group {g} not in [0, {self.G})")
        kind = self.scene.groups[g]["kind"]
        if rotation is not None:
            if kind != "rigid":
                raise ValueError("rotation is for rigid groups")
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(np.asarray(rotation, float)).as_quat()
            prev = self.group_q1.numpy()[g]
            if np.dot(prev, q) < 0:
                q = -q
            self._set_row(self.group_q1, g, q)
        if affine is not None:
            if kind != "affine":
                raise ValueError("affine maps are for affine groups")
            self._set_row(self.group_A1, g, np.asarray(affine, float))
        if translation is not None:
            self._set_row(self.group_b1, g, np.asarray(translation, float))
        if active is not None:
            self._set_row(self.group_active, g, int(bool(active)))

    def _set_row(self, array, row, value):
        host = array.numpy()
        host[row] = value
        array.assign(host)

    def set_traction(self, t, force):
        if not 0 <= t < self.Gt:
            raise IndexError(f"traction group {t} not in [0, {self.Gt})")
        self._set_row(self.traction_force, t, np.asarray(force, np.float32))

    # ------------------------------------------------------------------ substep

    def step(self):
        n, nn, dt = self.n, self.node_count, self.dt
        projection = self.dirichlet == "consistent_mass_projection_v1"
        work_v1 = self.work if self.traction_work == "particle_velocity_v1" else self.work_scratch
        with wp.ScopedDevice(self.device):
            wp.copy(self.x_start, self.x)
            wp.launch(rc.cell_jbar, self.cells, [self.cell_node, self.cell_offsets, self.cell_particles, self.F,
                                                 self.jbar])
            wp.launch(_stress, n, [self.F, self.nodes, self.jbar, self.stress, self.material.mu0,
                                   self.material.kappa, wp.float64(self.j_domain[0]), wp.float64(self.j_domain[1]),
                                   self.violations])
            if self.cfl_guard_limit is not None:
                wp.launch(_cfl_guard, n, [self.F, self.nodes, self.jbar, self.mass, self.material.mu0,
                                          self.material.kappa, self.volume, dt/self.dx, self.cfl_guard_limit,
                                          self.cfl_violations])
            wp.launch(_loads, n, [self.x, self.v, self.mass, self.impulse, self.particle_traction, self.traction_force,
                                  self.traction_mass, dt, self.damping_rate, self.radius, self.contact_stiffness,
                                  self.material.friction_table, self.use_plane, self.plane, self.boxes, self.box_c,
                                  self.box_h, work_v1])
            self.gm.zero_(); self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._p2g, (n, 8), [self.v, self.mass, self.slot, self.stress, self.impulse, self.nodes,
                                        self.weights, self.gradients, self.gm, self.gp, self.gj, 1, self.volume, dt,
                                        self.gravity])
            wp.launch(v5._grid_update, (1, nn), [self.gm, self.gp, self.gj, self.old, self.new_v, self.slot_code, 1])
            if not projection:
                self._launch_bc(1)
            wp.launch(v5._g2p, n, [self.x, self.v, self.slot, self.nodes, self.weights, self.old, self.new_v, 1, dt,
                                   self.pic])
            self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._musl, (n, 8), [self.v, self.mass, self.slot, self.nodes, self.weights, self.gp, 1])
            if projection and self.proj_rows:
                self._launch_projection()
                self.gp.zero_()
                wp.launch(v5._musl, (n, 8), [self.v, self.mass, self.slot, self.nodes, self.weights, self.gp, 1])
            wp.launch(v5._grid_update, (1, nn), [self.gm, self.gp, self.gj, self.old, self.new_v, self.slot_code, 1])
            self._launch_bc(0)
            if self.state_precision == "f64_shadow_v1":
                wp.launch(_deformation_f64, n, [self.F, self.F64, self.nodes, self.gradients, self.new_v,
                                                wp.float64(dt)])
                wp.launch(_positions_f64, n, [self.x, self.x64, self.nodes, self.weights, self.new_v, wp.float64(dt)])
            else:
                wp.launch(v5._deformation, n, [self.F, self.slot, self.nodes, self.gradients, self.new_v, 1, dt])
                wp.launch(_grid_positions, n, [self.x, self.x_start, self.slot, self.nodes, self.weights, self.new_v,
                                               1, dt])
            if self.traction_work == "grid_displacement_v1" and self.Gt:
                wp.launch(_traction_work_grid, n, [self.x, self.x_start, self.mass, self.particle_traction,
                                                   self.traction_force, self.traction_mass, self.work])
            wp.launch(v5._advance, 1, [self.index])

    def _launch_projection(self):
        """F8 (see the module docstring); launched between the MUSL scatter and its repetition."""
        r = self.proj_rows
        groups = [self.group_kind, self.group_active, self.group_q0, self.group_q1, self.group_b0, self.group_b1,
                  self.group_A0, self.group_A1]
        wp.launch(_proj_rhs, r, [self.proj_row_node, self.proj_row_comp, self.proj_row_group, self.node_X, self.gm,
                                 self.gp, self.index, self.substeps, wp.float64(self.dt)]+groups
                  + [self.proj_rhs, self.proj_vpresc])
        self.proj_mu_node.zero_()
        wp.launch(_proj_solve, r, [self.proj_row_block, self.proj_row_local, self.proj_block_offset,
                                   self.proj_block_size, self.proj_inv_offset, self.proj_inv, self.proj_rhs,
                                   self.proj_row_node, self.proj_row_comp, self.proj_row_group, self.group_active,
                                   self.proj_mu_node, self.proj_mu])
        wp.launch(_proj_particles, self.n, [self.v, self.nodes, self.weights, self.proj_mu_node])
        wp.launch(_proj_reaction, r, [self.proj_row_node, self.proj_row_comp, self.proj_row_group, self.node_X,
                                      self.gm, self.proj_mu, self.proj_vpresc, self.index, self.substeps]
                  + groups + [self.group_origin, self.reaction, self.work])

    def _launch_bc(self, record):
        if self.G == 0:
            return
        wp.launch(_bc, self.node_count, [self.node_group, self.node_mask, self.node_X, self.gm, self.new_v,
                                         self.index, self.substeps, wp.float64(self.dt), self.group_kind,
                                         self.group_active, self.group_q0, self.group_q1, self.group_b0,
                                         self.group_b1, self.group_A0, self.group_A1, self.group_origin,
                                         self.reaction, self.work, record], device=self.device)

    # ------------------------------------------------------------------ control-step bookkeeping

    def begin_control_step(self):
        """Zero the substep index and the per-control-step reaction accumulators and make every group hold
        (start = end); new motions are then written with ``set_group_motion`` (no allocation)."""
        self.index.zero_()
        self.reaction.zero_()
        for a0, a1 in ((self.group_q0, self.group_q1), (self.group_b0, self.group_b1), (self.group_A0, self.group_A1)):
            wp.copy(a0, a1)

    def end_control_step(self, control_dt):
        """Group wrenches [G, 6]: force (N) and torque (N m) the material exerts on each group, averaged over the
        control step (= -sum of BC impulses / control_dt)."""
        return -self.reaction.numpy()[:self.G]/float(control_dt)

    def reset(self):
        """Fresh reference state (x = X, v = 0, F = I, zero grid/stress, identity group motions, counters 0)."""
        with wp.ScopedDevice(self.device):
            wp.copy(self.x, self.reference)
            for a in (self.v, self.x_start, self.stress, self.impulse, self.gm, self.gp, self.gj, self.old,
                      self.new_v, self.reaction, self.work, self.index, self.violations, self.cfl_violations,
                      self.traction_force, self.group_b0, self.group_b1):
                a.zero_()
            self.F.assign(np.tile(np.eye(3, dtype=np.float32), (self.n, 1, 1)))
            self.jbar.fill_(1.0)
            G1 = max(self.G, 1)
            ident_q = np.tile(np.array([0, 0, 0, 1], np.float64), (G1, 1))
            self.group_q0.assign(ident_q)
            self.group_q1.assign(ident_q)
            self.group_A0.assign(np.tile(np.eye(3), (G1, 1, 1)))
            self.group_A1.assign(np.tile(np.eye(3), (G1, 1, 1)))
            self.group_active.fill_(1)
            if self.state_precision == "f64_shadow_v1":
                self.F64.assign(np.tile(np.eye(3), (self.n, 1, 1)))
                self.x64.assign(self.scene.X.astype(np.float64))
        self.damping_rate = 0.0

    # ------------------------------------------------------------------ queries

    def nodal_internal_force(self):
        """Diagnostic (does not advance): J-bar + stress + frozen P2G with zero gravity, impulse and velocity
        contributions removed; returns f_I = gj / dt [nodes, 3] (float64) = -sum_p V P_p grad N_I."""
        with wp.ScopedDevice(self.device):
            wp.launch(rc.cell_jbar, self.cells, [self.cell_node, self.cell_offsets, self.cell_particles, self.F,
                                                 self.jbar])
            dummy = wp.zeros(1, dtype=int)
            wp.launch(_stress, self.n, [self.F, self.nodes, self.jbar, self.stress, self.material.mu0,
                                        self.material.kappa, wp.float64(self.j_domain[0]),
                                        wp.float64(self.j_domain[1]), dummy])
            gm = wp.zeros(self.node_count, dtype=float)
            gp = wp.zeros(self.node_count, dtype=wp.vec3)
            gj = wp.zeros(self.node_count, dtype=wp.vec3)
            zero_v = wp.zeros(self.n, dtype=wp.vec3)
            wp.launch(v5._p2g, (self.n, 8), [zero_v, self.mass, self.slot, self.stress, zero_v, self.nodes,
                                             self.weights, self.gradients, gm, gp, gj, 1, self.volume, self.dt,
                                             wp.vec3(0.0)])
            return gj.numpy().astype(np.float64)/self.dt

    def jbar_cells(self):
        """Current J-bar per occupied cell (float64), recomputed from F."""
        wp.launch(rc.cell_jbar, self.cells, [self.cell_node, self.cell_offsets, self.cell_particles, self.F,
                                             self.jbar], device=self.device)
        return self.jbar.numpy()[self.scene.cell_node]

    def memory_bytes(self):
        return int(sum(v.capacity for v in vars(self).values() if isinstance(v, wp.array))
                   + sum(v.capacity for v in vars(self.diagnostics).values() if isinstance(v, wp.array)))

    def device_arrays(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, wp.array)}

    def solver_receipt(self):
        m = self.material
        return {"solver": SOLVER_NAME, "status": STATUS, "adr": ADR, "model": rc.MODEL_ID,
                "model_version": rc.MODEL_VERSION, "pic": 0.0, "transfer": "pure_flip_musl_grid_positions",
                "dt_s": self.dt, "substeps_per_control_step": self.substeps, "dx_m": self.dx,
                "particle_spacing_m": self.spacing, "particles": self.n, "nodes": self.node_count,
                "cells": self.cells, "groups": self.G, "traction_groups": self.Gt, "gravity": list(self.gravity_tuple),
                "j_domain": list(self.j_domain), "material": m.to_dict(),
                "cfl_actual": self.dt*m.c_p()/self.dx, "contact": {"table_plane": bool(self.use_plane),
                "fixture_boxes": self.boxes, "stiffness_fraction": self.contact_stiffness, "radius_m": self.radius,
                "status": "implemented, validated in P2-4 (RUBBER-CONT)"},
                "damping_rate": self.damping_rate, "frozen_kernels": list(FROZEN_KERNELS),
                "stable_step_guard": (None if self.cfl_guard_limit is None else
                                      {"id": rc.STABLE_STEP_GUARD_ID, "limit": self.cfl_guard_limit,
                                       "quantity": "max_p dt c_bound(F_p, Jbar) / dx, every substep"}),
                "options": {"dirichlet": self.dirichlet, "traction_work": self.traction_work,
                            "state_precision": self.state_precision, "projection": self.projection_receipt},
                "source_sha256": source_sha256(), "device": self.device, "warp_version": wp.config.version}
