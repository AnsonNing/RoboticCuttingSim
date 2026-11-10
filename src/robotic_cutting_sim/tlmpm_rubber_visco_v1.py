"""Single-env viscoelastic rubber solver ``TLMPMRubberVisco`` (model ``neo_hookean_jbar_prony_v1``; ADR 0023 §10,
§21.2; task R3.4).

Status ``EXPERIMENTAL_NOT_ACCEPTED``; material ``SIMULATOR_ONLY_UNCALIBRATED`` (the Prony parameters of every recipe and
test are illustrative, not fitted to real data). Opt-in: built by ``LiveRubberMaterial`` only for a material with
Prony terms (or the ``visco_energy`` option); the elastic modules ``tlmpm_rubber_v1.py`` and ``rubber_constitutive.py``
are unchanged (their hashes are pinned in ADR 0023 §21).

``step()`` is the ``tlmpm_rubber_v1`` launch sequence (via ``TLMPMRubberCW``, so ``constraint_work`` is available) with
the stress pass replaced, when the material has M >= 1 Prony terms, by the J-bar pass + ``stress_prony``
(``rubber_visco_constitutive.prony_stress_particle``: S0_n, Q_i <- a_i Q_i + b_i (S0_n - S0_prev), S0_prev <- S0_n,
P = P_NH + F sum_i (Q_i - g_i S0)) and the optional stable-step guard by ``cfl_guard_prony``
(``stable_step_guard_visco_v1``, §21.2.6). With M = 0 the frozen ``_stress`` / ``_cfl_guard`` are launched (the elastic
solver, byte-identical; RUBBER-VISC-09). The Prony state after a control step belongs to the last stress pass
(t_k - dt), as the stress array.

State (float64, §21.2.3-21.2.4): ``visco_q[n*M]`` (vec6d, particle-major p*M + i), ``visco_s_prev[n]`` (vec6d),
``params64[1, 1 + 4M]`` = [g_inf, (g_i, tau_i, a_i, b_i) x M] (a_i, b_i from expm1 on the host at the solver dt),
``env_of_particle[n]`` (zeros: the kernels are shared with the batched path, which indexes envs).

Energy option ``visco_energy`` (§21.2.7): ``off_v1`` (default, no array, no launch) or ``branch_work_v1``: the stress
pass stores the branch PK1 ``visco_pb`` = F_n sum_i Q_i^n and adds the explicit dissipation
V sum_i dt (|Q_i^{n-1}|^2 + |Q_i^n|^2) / (4 g_i mu0 tau_i) to ``visco_dissipation`` (negative / non-finite increments
counted in ``visco_d_negative``); after the position update ``visco_branch_work`` adds V visco_pb : (dt sum_I v_I (x)
grad N_I) (the drift's dF, recomputed in float64 from the MUSL grid velocity after the BC) to ``visco_branch_work``.
The layer-N books B_N = E_kin + Psi_eq + E_grav + W_b - W_ext close up to the O(dt) symplectic error.

Energy option ``branch_work_kick_average_v2`` (ADR 0023 §21.7, protocol ``rubber_visc_tests_v2``): the same stress pass,
dissipation and ``visco_pb``; after the position update ``visco_branch_work_kick_average`` adds
V visco_pb_n : (dF_{n-1} + dF_n) / 2 to ``visco_branch_work`` (the branch force paired with the velocity average of the
kick, i.e. the conjugate of the discrete kinetic energy; ``visco_df_prev[n]`` (mat33d) holds dF_{n-1}, zero after a
reset) and the v1 left-point V visco_pb_n : dF_n, with the arithmetic of ``visco_branch_work``, to
``visco_branch_work_left`` (reported). The left-point W_b of ``branch_work_v1`` drifts by about dt w^2 tau / 2 of the
branch dissipation on high-frequency content (§21.7.1). Bookkeeping only: every other array is byte-identical to
``branch_work_v1`` (and the state to ``off_v1``); ``off_v1`` / ``branch_work_v1`` are unchanged.

Visco diagnostics (``ViscoDiagnostics``, fixed-order, per env, shared with the batch): Psi_eq (g_inf mu0 isochoric +
the frozen volumetric column), Psi_v = sum V sum_i |Q_i|^2 / (4 g_i mu0), the accumulators and the max visco CFL of
the current state.
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from . import rubber_constitutive as rc
from . import rubber_visco_constitutive as rvc
from . import tlmpm_rubber_v1 as T
from . import tlmpm_v5 as v5
from .rubber_constitutive import det_d, in_domain, iso_energy_density
from .rubber_constraint_work import DEFAULT_CONSTRAINT_WORK, TLMPMRubberCW
from .rubber_visco_constitutive import (P64_STRIDE, branch_stored_density, cfl_effective_visco,
                                        prony_stress_particle, visco_guard_terms, vec6d)
from .tlmpm_v6 import _grid_positions

wp.set_module_options({"enable_backward": False})

SOLVER_NAME = "tlmpm_rubber_visco_v1"
STATUS = "EXPERIMENTAL_NOT_ACCEPTED"
ADR = "docs/adr/0023-rubber-material-v1.md section 21.2"
VISCO_ENERGY_MODES = ("off_v1", "branch_work_v1", "branch_work_kick_average_v2")
DEFAULT_VISCO_ENERGY = "off_v1"
KICK_AVERAGE_ENERGY = "branch_work_kick_average_v2"         # ADR 0023 §21.7
P_MU, P_KAPPA = 0, 1               # columns of the float32 per-env table (single: dparams; batch: params)
VISCO_DIAG_COLUMNS = ("psi_eq", "psi_eq_iso", "psi_eq_vol", "psi_visc_stored", "branch_work", "dissipation",
                      "dissipation_negative_increments", "cfl_visco_max")
NVDIAG = len(VISCO_DIAG_COLUMNS)
ENERGY_ARRAYS = ("visco_pb", "visco_branch_work", "visco_dissipation", "visco_d_negative")
KICK_AVERAGE_ARRAYS = ("visco_df_prev", "visco_branch_work_left")      # branch_work_kick_average_v2 only
_HERE = Path(__file__).resolve()
SOURCES = ("tlmpm_rubber_visco_v1.py", "rubber_visco_constitutive.py")


def source_sha256():
    out = {name: hashlib.sha256((_HERE.parent/name).read_bytes()).hexdigest() for name in SOURCES}
    return out


# ----------------------------------------------------------------------------- kernels (shared single / batched)

@wp.kernel
def stress_prony(F: wp.array(dtype=wp.mat33), nodes: wp.array2d(dtype=int), jbar: wp.array(dtype=wp.float64),
                 stress: wp.array(dtype=wp.mat33), params: wp.array2d(dtype=float),
                 env_of_particle: wp.array(dtype=int), params64: wp.array2d(dtype=wp.float64), M: int,
                 q: wp.array(dtype=vec6d), s_prev: wp.array(dtype=vec6d), lo: wp.float64, hi: wp.float64,
                 violations: wp.array(dtype=int), dt: wp.float64, volume: wp.float64, energy: int,
                 pb: wp.array(dtype=wp.mat33d), dissipation: wp.array(dtype=wp.float64),
                 d_negative: wp.array(dtype=int)):
    """Prony stress pass (``tlmpm_rubber_v1._stress`` + §21.2.2) with mu0, kappa, params64 of the particle's env."""
    p = wp.tid()
    e = env_of_particle[p]
    f = F[p]
    jb = jbar[nodes[p, 0]]
    if not (in_domain(det_d(f), lo, hi) and in_domain(jb, lo, hi)):
        wp.atomic_add(violations, e, 1)
    out, qsum, dens, negative = prony_stress_particle(p, e, f, jb, params[e, P_MU], params[e, P_KAPPA], params64,
                                                      M, q, s_prev, dt)
    stress[p] = out
    if energy != 0:
        pb[p] = wp.mat33d(f)*rvc.sym6_to_mat(qsum)
        if dens != wp.float64(0.0):
            wp.atomic_add(dissipation, e, volume*dens)
        if negative != 0:
            wp.atomic_add(d_negative, e, negative)


@wp.kernel
def cfl_guard_prony(F: wp.array(dtype=wp.mat33), nodes: wp.array2d(dtype=int), jbar: wp.array(dtype=wp.float64),
                    mass: wp.array(dtype=float), params: wp.array2d(dtype=float),
                    env_of_particle: wp.array(dtype=int), params64: wp.array2d(dtype=wp.float64), M: int,
                    q: wp.array(dtype=vec6d), s_prev: wp.array(dtype=vec6d), volume: float, dt_over_dx: float,
                    limit: float, cfl_violations: wp.array(dtype=int)):
    """``stable_step_guard_visco_v1`` (after the Prony stress pass: Q^n and S0_n = s_prev of this substep)."""
    p = wp.tid()
    e = env_of_particle[p]
    factor, over = visco_guard_terms(p, e, params64, M, q, s_prev[p])
    c = cfl_effective_visco(F[p], jbar[nodes[p, 0]], params[e, P_MU], params[e, P_KAPPA], mass[p]/volume,
                            dt_over_dx, factor, over)
    if not (c <= limit):
        wp.atomic_add(cfl_violations, e, 1)


@wp.kernel
def visco_branch_work(pb: wp.array(dtype=wp.mat33d), nodes: wp.array2d(dtype=int),
                      gradients: wp.array2d(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
                      env_of_particle: wp.array(dtype=int), volume: wp.float64, dt: wp.float64,
                      work: wp.array(dtype=wp.float64)):
    """W_b[env] += V visco_pb : dF, dF = dt sum_I v_I (x) grad N_I (the ``_deformation_f64`` increment, float64)."""
    p = wp.tid()
    grad = wp.mat33d()
    for c in range(8):
        grad += wp.outer(wp.vec3d(velocity[nodes[p, c]]), wp.vec3d(gradients[p, c]))
    wp.atomic_add(work, env_of_particle[p], volume*wp.ddot(pb[p], dt*grad))


@wp.kernel
def visco_branch_work_kick_average(pb: wp.array(dtype=wp.mat33d), nodes: wp.array2d(dtype=int),
                                   gradients: wp.array2d(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
                                   env_of_particle: wp.array(dtype=int), volume: wp.float64, dt: wp.float64,
                                   df_prev: wp.array(dtype=wp.mat33d), work: wp.array(dtype=wp.float64),
                                   work_left: wp.array(dtype=wp.float64)):
    """``branch_work_kick_average_v2`` (ADR 0023 §21.7): W_b[env] += V visco_pb : (dF_{n-1} + dF_n) / 2 and the v1
    left-point W_b,v1[env] += V visco_pb : dF_n (same arithmetic as ``visco_branch_work``); dF_{n-1} <- dF_n."""
    p = wp.tid()
    grad = wp.mat33d()
    for c in range(8):
        grad += wp.outer(wp.vec3d(velocity[nodes[p, c]]), wp.vec3d(gradients[p, c]))
    e = env_of_particle[p]
    df = dt*grad
    wp.atomic_add(work_left, e, volume*wp.ddot(pb[p], df))
    wp.atomic_add(work, e, wp.float64(0.5)*volume*wp.ddot(pb[p], df+df_prev[p]))
    df_prev[p] = df


@wp.kernel
def visco_diag_particles(F: wp.array(dtype=wp.mat33), mass: wp.array(dtype=float), nodes: wp.array2d(dtype=int),
                         jbar: wp.array(dtype=wp.float64), params: wp.array2d(dtype=float),
                         params64: wp.array2d(dtype=wp.float64), M: int, q: wp.array(dtype=vec6d),
                         s_prev: wp.array(dtype=vec6d), particle_offset: wp.array(dtype=int), groups: int,
                         volume: float, dt_over_dx: float, partials: wp.array2d(dtype=wp.float64)):
    """Thread (e, g) strides over env e's particles (fixed order, float64): g_inf-scaled isochoric energy, stored
    branch energy, max visco CFL of the current state (finite particles only)."""
    e, grp = wp.tid()
    eiso = wp.float64(0.0)
    psiv = wp.float64(0.0)
    cflmax = wp.float64(0.0)
    mu32 = params[e, P_MU]
    mu = wp.float64(mu32)
    vol = wp.float64(volume)
    ginf = params64[e, 0]
    for p in range(particle_offset[e]+grp, particle_offset[e+1], groups):
        fp = F[p]
        ok = True
        for a in range(3):
            for b in range(3):
                if not wp.isfinite(fp[a, b]):
                    ok = False
        if ok:
            eiso += vol*ginf*iso_energy_density(fp, mu32)
            for i in range(M):
                g = params64[e, 1+P64_STRIDE*i]
                if g != wp.float64(0.0):
                    psiv += vol*branch_stored_density(q[p*M+i], g, mu)
            factor, over = visco_guard_terms(p, e, params64, M, q, s_prev[p])
            cflmax = wp.max(cflmax, wp.float64(cfl_effective_visco(fp, jbar[nodes[p, 0]], mu32, params[e, P_KAPPA],
                                                                   mass[p]/volume, dt_over_dx, factor, over)))
    row = e*groups+grp
    partials[row, 0] = eiso
    partials[row, 1] = psiv
    partials[row, 2] = cflmax


@wp.kernel
def visco_diag_combine(pp: wp.array2d(dtype=wp.float64), groups: int, summary: wp.array2d(dtype=wp.float64),
                       energy: int, branch: wp.array(dtype=wp.float64), dissipation: wp.array(dtype=wp.float64),
                       d_negative: wp.array(dtype=int), out: wp.array2d(dtype=wp.float64)):
    """Per env: the fixed-order sums of the partials + the frozen diagnostics' volumetric energy (column 8)."""
    e = wp.tid()
    eiso = wp.float64(0.0)
    psiv = wp.float64(0.0)
    cflmax = wp.float64(0.0)
    for k in range(groups):
        r = e*groups+k
        eiso += pp[r, 0]
        psiv += pp[r, 1]
        cflmax = wp.max(cflmax, pp[r, 2])
    evol = summary[e, 8]
    out[e, 0] = eiso+evol
    out[e, 1] = eiso
    out[e, 2] = evol
    out[e, 3] = psiv
    out[e, 4] = wp.float64(0.0)
    out[e, 5] = wp.float64(0.0)
    out[e, 6] = wp.float64(0.0)
    if energy != 0:
        out[e, 4] = branch[e]
        out[e, 5] = dissipation[e]
        out[e, 6] = wp.float64(d_negative[e])
    out[e, 7] = cflmax


class ViscoDiagnostics:
    """Per-env visco diagnostics of a solver (single or batched) after its frozen segmented diagnostics."""

    def __init__(self, solver, num_envs):
        self.solver = solver
        self.groups = T.DIAG_GROUPS
        with wp.ScopedDevice(solver.device):
            self.vpp = wp.zeros((num_envs*self.groups, 3), dtype=wp.float64)
            self.visco_summary = wp.zeros((num_envs, NVDIAG), dtype=wp.float64)

    def launch(self, main_summary, params, particle_offset):
        """``main_summary``: the frozen diagnostics' summary (launched first: it recomputes Jbar from F)."""
        s = self.solver
        N = self.visco_summary.shape[0]
        with wp.ScopedDevice(s.device):
            wp.launch(visco_diag_particles, (N, self.groups), [s.F, s.mass, s.nodes, s.jbar, params, s.params64, s.M,
                                                               s.visco_q, s.visco_s_prev, particle_offset,
                                                               self.groups, s.volume, s.dt/s.dx, self.vpp])
            wp.launch(visco_diag_combine, N, [self.vpp, self.groups, main_summary, int(s.visco_energy_on),
                                              getattr(s, "visco_branch_work", None),
                                              getattr(s, "visco_dissipation", None),
                                              getattr(s, "visco_d_negative", None), self.visco_summary])

    def read(self):
        return [dict(zip(VISCO_DIAG_COLUMNS, map(float, r))) for r in self.visco_summary.numpy()]


# ----------------------------------------------------------------------------- solver

class TLMPMRubberVisco(TLMPMRubberCW):
    """``tlmpm_rubber_v1`` (+ ``constraint_work``) with the Prony stress pass (module docstring). Same constructor
    plus ``visco_energy``; the Prony terms are ``material.prony`` (``rubber_material.MaterialValues``)."""
    solver_name = SOLVER_NAME

    def __init__(self, scene, material, dt, *, visco_energy=DEFAULT_VISCO_ENERGY, **kwargs):
        if visco_energy not in VISCO_ENERGY_MODES:
            raise ValueError(f"visco_energy={visco_energy!r} not in {VISCO_ENERGY_MODES}")
        from .rubber_material import validate_prony
        self.prony = validate_prony(getattr(material, "prony", ()))
        self.M = len(self.prony)
        if self.M > rvc.MAX_TERMS:
            raise ValueError(f"at most {rvc.MAX_TERMS} Prony terms")
        self.visco_energy = visco_energy
        self.visco_energy_on = visco_energy != DEFAULT_VISCO_ENERGY and self.M > 0
        super().__init__(scene, material, dt, **kwargs)
        self.params64_host = rvc.params64_row(self.prony, self.dt).reshape(1, -1)
        with wp.ScopedDevice(self.device):
            self.env_of_particle = wp.zeros(self.n, dtype=int)
            self.params64 = wp.array(self.params64_host, dtype=wp.float64)
            if self.M:
                self.visco_q = wp.zeros(self.n*self.M, dtype=vec6d)
                self.visco_s_prev = wp.zeros(self.n, dtype=vec6d)
            if self.visco_energy_on:
                self.visco_pb = wp.zeros(self.n, dtype=wp.mat33d)
                self.visco_branch_work = wp.zeros(1, dtype=wp.float64)
                self.visco_dissipation = wp.zeros(1, dtype=wp.float64)
                self.visco_d_negative = wp.zeros(1, dtype=int)
                if visco_energy == KICK_AVERAGE_ENERGY:
                    self.visco_df_prev = wp.zeros(self.n, dtype=wp.mat33d)
                    self.visco_branch_work_left = wp.zeros(1, dtype=wp.float64)
        self.visco_diagnostics = ViscoDiagnostics(self, 1) if self.M else None
        for module in (rvc.__name__, __name__):
            wp.load_module(module=module, device=self.device)

    @property
    def model_id(self):
        return rvc.MODEL_ID if self.M else rc.MODEL_ID

    # ------------------------------------------------------------------ substep

    def step(self):
        """``tlmpm_rubber_v1.TLMPMRubber.step`` with the Prony stress pass / guard (M >= 1) and the opt-in branch-work
        launch after the position update."""
        n, nn, dt = self.n, self.node_count, self.dt
        projection = self.dirichlet == "consistent_mass_projection_v1"
        work_v1 = self.work if self.traction_work == "particle_velocity_v1" else self.work_scratch
        with wp.ScopedDevice(self.device):
            wp.copy(self.x_start, self.x)
            wp.launch(rc.cell_jbar, self.cells, [self.cell_node, self.cell_offsets, self.cell_particles, self.F,
                                                 self.jbar])
            if self.M:
                self._launch_stress_prony()
            else:
                wp.launch(T._stress, n, [self.F, self.nodes, self.jbar, self.stress, self.material.mu0,
                                         self.material.kappa, wp.float64(self.j_domain[0]),
                                         wp.float64(self.j_domain[1]), self.violations])
            if self.cfl_guard_limit is not None:
                if self.M:
                    wp.launch(cfl_guard_prony, n, [self.F, self.nodes, self.jbar, self.mass, self.dparams,
                                                   self.env_of_particle, self.params64, self.M, self.visco_q,
                                                   self.visco_s_prev, self.volume, dt/self.dx, self.cfl_guard_limit,
                                                   self.cfl_violations])
                else:
                    wp.launch(T._cfl_guard, n, [self.F, self.nodes, self.jbar, self.mass, self.material.mu0,
                                                self.material.kappa, self.volume, dt/self.dx, self.cfl_guard_limit,
                                                self.cfl_violations])
            wp.launch(T._loads, n, [self.x, self.v, self.mass, self.impulse, self.particle_traction,
                                    self.traction_force, self.traction_mass, dt, self.damping_rate, self.radius,
                                    self.contact_stiffness, self.material.friction_table, self.use_plane, self.plane,
                                    self.boxes, self.box_c, self.box_h, work_v1])
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
                wp.launch(T._deformation_f64, n, [self.F, self.F64, self.nodes, self.gradients, self.new_v,
                                                  wp.float64(dt)])
                wp.launch(T._positions_f64, n, [self.x, self.x64, self.nodes, self.weights, self.new_v,
                                                wp.float64(dt)])
            else:
                wp.launch(v5._deformation, n, [self.F, self.slot, self.nodes, self.gradients, self.new_v, 1, dt])
                wp.launch(_grid_positions, n, [self.x, self.x_start, self.slot, self.nodes, self.weights, self.new_v,
                                               1, dt])
            if self.traction_work == "grid_displacement_v1" and self.Gt:
                wp.launch(T._traction_work_grid, n, [self.x, self.x_start, self.mass, self.particle_traction,
                                                     self.traction_force, self.traction_mass, self.work])
            if self.visco_energy_on:
                if self.visco_energy == KICK_AVERAGE_ENERGY:
                    wp.launch(visco_branch_work_kick_average, n, [self.visco_pb, self.nodes, self.gradients,
                                                                  self.new_v, self.env_of_particle,
                                                                  wp.float64(self.volume), wp.float64(dt),
                                                                  self.visco_df_prev, self.visco_branch_work,
                                                                  self.visco_branch_work_left])
                else:
                    wp.launch(visco_branch_work, n, [self.visco_pb, self.nodes, self.gradients, self.new_v,
                                                     self.env_of_particle, wp.float64(self.volume), wp.float64(dt),
                                                     self.visco_branch_work])
            wp.launch(v5._advance, 1, [self.index])

    def _launch_stress_prony(self):
        on = self.visco_energy_on
        wp.launch(stress_prony, self.n, [self.F, self.nodes, self.jbar, self.stress, self.dparams,
                                         self.env_of_particle, self.params64, self.M, self.visco_q,
                                         self.visco_s_prev, wp.float64(self.j_domain[0]),
                                         wp.float64(self.j_domain[1]), self.violations, wp.float64(self.dt),
                                         wp.float64(self.volume), int(on), self.visco_pb if on else None,
                                         self.visco_dissipation if on else None,
                                         self.visco_d_negative if on else None], device=self.device)

    # ------------------------------------------------------------------ episode / queries

    def reset(self):
        """The parent reset plus the Prony state and accumulators; also the discarded v1 traction-work scratch of
        ``grid_displacement_v1`` (not cleared by the frozen parent; cleared by the batched masked reset), so two
        episodes from reset() are byte-identical (RUBBER-VISC-06)."""
        super().reset()
        for name in ("visco_q", "visco_s_prev", "work_scratch")+ENERGY_ARRAYS+KICK_AVERAGE_ARRAYS:
            if hasattr(self, name):
                getattr(self, name).zero_()

    def read_visco_diagnostics(self):
        """Launch the frozen diagnostics then the visco diagnostics; returns (diag, visco_diag) of env 0."""
        diag = self.diagnostics.read()[0]
        if not self.M:
            return diag, None
        self.visco_diagnostics.launch(self.diagnostics.summary, self.dparams, self.particle_offset)
        return diag, self.visco_diagnostics.read()[0]

    def nodal_internal_force(self):
        if self.M:
            raise NotImplementedError("nodal_internal_force advances the Prony state; it is refused for "
                                      "neo_hookean_jbar_prony_v1 (ADR 0023 §21.2.8)")
        return super().nodal_internal_force()

    def memory_bytes(self):
        extra = 0 if self.visco_diagnostics is None else sum(
            v.capacity for v in vars(self.visco_diagnostics).values() if isinstance(v, wp.array))
        return super().memory_bytes()+int(extra)

    def solver_receipt(self):
        out = super().solver_receipt()
        out["visco_solver"] = SOLVER_NAME
        out["model"] = self.model_id
        out["model_version"] = rvc.MODEL_VERSION if self.M else rc.MODEL_VERSION
        out["options"]["visco_energy"] = self.visco_energy
        out["prony"] = {"terms": [{"g": g, "tau_s": tau} for g, tau in self.prony], "M": self.M,
                        "params64_names": list(rvc.param64_names(self.M)),
                        "params64": self.params64_host[0].tolist(),
                        "coefficients": "a_i = exp(-dt/tau_i), b_i = -g_i (tau_i/dt) expm1(-dt/tau_i), float64 host",
                        "state": {"visco_q": "vec6d float64 [n*M]", "visco_s_prev": "vec6d float64 [n]"},
                        "bytes_per_particle_state": 48*(self.M+1) if self.M else 0,
                        "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED (illustrative Prony parameters, "
                                              "not fitted to real data)"}
        if self.cfl_guard_limit is not None and self.M:
            out["stable_step_guard"] = {"id": rvc.VISCO_GUARD_ID, "limit": self.cfl_guard_limit,
                                        "quantity": "max_p dt c_bound,visco(F_p, Jbar, Q, S0) / dx, every substep",
                                        "adr": "ADR 0023 §21.2.6"}
        out["source_sha256"] = {**out["source_sha256"], **source_sha256()}
        return out


__all__ = ["TLMPMRubberVisco", "ViscoDiagnostics", "stress_prony", "cfl_guard_prony", "visco_branch_work",
           "visco_branch_work_kick_average", "KICK_AVERAGE_ENERGY", "KICK_AVERAGE_ARRAYS",
           "visco_diag_particles", "visco_diag_combine", "VISCO_DIAG_COLUMNS", "DEFAULT_CONSTRAINT_WORK"]
