"""Constraint-work accounting of the F8 projected Dirichlet constraint: option ``constraint_work`` (ADR 0023 §21.1).

Bookkeeping only. No state array changes (tested byte-wise): the particle / grid / F / x / reaction / wrench values are
those of ``tlmpm_rubber_v1`` with ``dirichlet="consistent_mass_projection_v1"``; only the number accumulated in
``work`` (the external work of the energy books) is defined differently.

- ``end_velocity_v1`` (default; the ``tlmpm_rubber_v1`` accounting, unchanged): W += Lambda_n . v_presc,n, the impulse
  of substep n times the prescribed velocity of the drift that follows it.
- ``kick_average_v1``: W += Lambda_n . (g_n,C + v_presc,n) / 2 with g_n the P2G-pass grid velocity ``old`` = p_n / m
  (= B pi_n, the MUSL velocity of the previous substep before its BC overwrite; equal to v_presc,n-1 on an active
  constrained row up to float32 rounding, and the free velocity when the group was inactive).

Why (diagnosis ``scripts/rubber/p2_3/diag_moving_clamp_energy.py``, ADR 0023 §21.1): pure FLIP + MUSL with fixed
weights is symplectic Euler in (pi, u) with KE = 1/2 pi^T B pi = the particle KE (B = M_l^-1 M_c M_l^-1). Under the
F8 projection the kick is pi_{n+1} = pi_n + dt f(u_n) + Lambda_n with Lambda supported on the constrained rows and
(B pi_{n+1})_C = v_presc,n (the standard constrained symplectic Euler for the linear holonomic constraint
u_C = c(t)). KE is quadratic, so KE_{n+1} - KE_n = (dt f + Lambda_n) . (g_n + g_{n+1}) / 2 exactly: the work
conjugate to the discrete kinetic energy is the kick average. The end-velocity quadrature over-counts
1/2 Lambda_n . (v_presc,n - g_n,C) per substep, i.e. 1/2 dv . m_eff dv at every discontinuity dv of the prescribed
velocity; the group pose is linear (slerp) within a control step, so the prescribed velocity jumps at every
control-step boundary even for a smooth control-level profile.

The kick-average work is defined for the projected constraint only: the lumped clamp (``lumped_v1``) is not a
symplectic constraint (its FLIP increment is a lumped-mass correction and the MUSL-pass overwrite does not reach the
particles), so no quadrature of its impulse closes the books; the combination is refused.

Implementation: ``TLMPMRubberCW`` subclasses the frozen single-env reference ``tlmpm_rubber_v1.TLMPMRubber`` (its
source stays byte-identical) and re-launches the parent's F8 kernels in the parent's order, diverting the parent's
end-velocity work to ``work_end_velocity`` (reported) and accumulating the kick-average work into ``work`` with
``proj_work_kick_average`` (shared with the batched path: same kernel, per-env target ``env_of_group[g]``).
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from .tlmpm_rubber_v1 import TLMPMRubber, _proj_particles, _proj_reaction, _proj_rhs, _proj_solve

wp.set_module_options({"enable_backward": False})

CONSTRAINT_WORK_MODES = ("end_velocity_v1", "kick_average_v1")
DEFAULT_CONSTRAINT_WORK = "end_velocity_v1"
REQUIRES_DIRICHLET = "consistent_mass_projection_v1"
ADR = "docs/adr/0023-rubber-material-v1.md section 21.1"
_HERE = Path(__file__).resolve()


def source_sha256():
    return {_HERE.name: hashlib.sha256(_HERE.read_bytes()).hexdigest()}


def check_options(constraint_work, dirichlet):
    if constraint_work not in CONSTRAINT_WORK_MODES:
        raise ValueError(f"constraint_work={constraint_work!r} not in {CONSTRAINT_WORK_MODES}")
    if constraint_work != DEFAULT_CONSTRAINT_WORK and dirichlet != REQUIRES_DIRICHLET:
        raise ValueError(f"constraint_work={constraint_work!r} needs dirichlet={REQUIRES_DIRICHLET!r}: the lumped clamp "
                         "is not a symplectic constraint, no work quadrature closes its energy books (ADR 0023 §21.1)")


@wp.kernel
def proj_work_kick_average(row_node: wp.array(dtype=int), row_comp: wp.array(dtype=int),
                           row_group: wp.array(dtype=int), active: wp.array(dtype=int), gm: wp.array(dtype=float),
                           mu: wp.array(dtype=wp.float64), vpresc: wp.array(dtype=wp.float64),
                           old: wp.array(dtype=wp.vec3), env_of_group: wp.array(dtype=int),
                           work: wp.array(dtype=wp.float64)):
    """W[env] += Lambda . (g_n + v_presc) / 2 per constrained row, Lambda = -m_I mu_I (the impulse on the material, as
    ``_proj_reaction``), g_n = old[I][k] (P2G-pass p / m). Launched after ``_proj_reaction``, before the MUSL
    ``_grid_update`` overwrites ``old``."""
    r = wp.tid()
    g = row_group[r]
    if active[g] != 0:
        i = row_node[r]
        k = row_comp[r]
        j = -wp.float64(gm[i])*mu[r]
        wp.atomic_add(work, env_of_group[g], j*(wp.float64(0.5)*(wp.float64(old[i][k])+vpresc[r])))


class TLMPMRubberCW(TLMPMRubber):
    """``tlmpm_rubber_v1`` with the ``constraint_work`` option (module docstring). Same constructor plus
    ``constraint_work``; with the default the launches and arrays are the parent's (no extra array, no extra launch)."""

    def __init__(self, scene, material, dt, *, constraint_work=DEFAULT_CONSTRAINT_WORK, **kwargs):
        check_options(constraint_work, kwargs.get("dirichlet", "lumped_v1"))
        super().__init__(scene, material, dt, **kwargs)
        self.constraint_work = constraint_work
        if constraint_work != DEFAULT_CONSTRAINT_WORK:
            with wp.ScopedDevice(self.device):
                self.work_end_velocity = wp.zeros(1, dtype=wp.float64)      # the v1 accounting (reported)
                self.env_of_group = wp.zeros(max(self.G, 1), dtype=int)       # one env: work row 0
            wp.load_module(module=__name__, device=self.device)

    def _launch_projection(self):
        if self.constraint_work == DEFAULT_CONSTRAINT_WORK:
            return super()._launch_projection()
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
                  + groups + [self.group_origin, self.reaction, self.work_end_velocity])
        wp.launch(proj_work_kick_average, r, [self.proj_row_node, self.proj_row_comp, self.proj_row_group,
                                              self.group_active, self.gm, self.proj_mu, self.proj_vpresc, self.old,
                                              self.env_of_group, self.work])

    def reset(self):
        super().reset()
        if self.constraint_work != DEFAULT_CONSTRAINT_WORK:
            self.work_end_velocity.zero_()

    def solver_receipt(self):
        out = super().solver_receipt()
        out["options"]["constraint_work"] = self.constraint_work
        out["constraint_work"] = {"id": self.constraint_work, "adr": ADR,
                                  "work": ("W += Lambda . (g_n + v_presc) / 2 (kick average; reported v1 end-velocity "
                                           "work in work_end_velocity)" if self.constraint_work != DEFAULT_CONSTRAINT_WORK
                                           else "W += Lambda . v_presc (tlmpm_rubber_v1)")}
        out["source_sha256"] = {**out["source_sha256"], **source_sha256()}
        return out


def host_kick_average_work(rows, mu, vpresc, gm, old, active):
    """Host reference of one substep's kick-average work (float64; tests): rows = (node, comp, group) arrays."""
    node, comp, group = rows
    on = np.asarray(active)[group] != 0
    lam = -np.asarray(gm, np.float64)[node]*np.asarray(mu, np.float64)
    return float(np.sum((lam*0.5*(np.asarray(old, np.float64)[node, comp]+np.asarray(vpresc, np.float64)))[on]))
