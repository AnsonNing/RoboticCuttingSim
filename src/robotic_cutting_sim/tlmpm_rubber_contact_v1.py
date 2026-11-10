"""Single-env rubber solver with contact ``TLMPMRubberContact`` (laws ``grid_node_penalty_coulomb_v1`` (default),
``grid_node_penalty_coulomb_v2`` (opt-in, ADR 0023 §22.9) and the rejected projection ``grid_node_coulomb_v1``;
ADR 0023 §22; P2-4).

Status ``EXPERIMENTAL_NOT_ACCEPTED``; material ``SIMULATOR_ONLY_UNCALIBRATED``. Opt-in: ``LiveRubberMaterial`` builds it
only when a ``rubber_contact.ContactSpec`` is passed; without one the elastic / visco solver classes are built
unchanged (``tlmpm_rubber_v1.py``, ``rubber_constitutive.py``, ``tlmpm_rubber_visco_v1.py``,
``rubber_visco_constitutive.py``, ``rubber_constraint_work.py`` are byte-identical).

``step()`` is the ``TLMPMRubberVisco.step`` launch sequence (itself the ``tlmpm_rubber_v1`` sequence; M = 0 launches
the frozen elastic kernels) with three launches added:

- after the P2G ``_grid_update`` (before the P2G-pass BC and G2P): ``rubber_contact.node_contact`` (the contact
  impulse enters the kick; first half of the kick-paired work);
- after the final MUSL ``_grid_update`` (before the MUSL-pass BC): ``rubber_contact.contact_work_end``;
- after the position update: ``rubber_contact.node_displacement`` (u_I += dt v_I).

The frozen per-particle table / fixture path (``table_plane`` / ``fixture_boxes`` of the frozen constructor) is
refused together with a ContactSpec: one contact law per solver. With an empty ContactSpec (no plane, no frame) the
contact kernels change no velocity, so every state array is byte-identical to ``TLMPMRubberVisco`` (tested).

State (registry-like, cleared by ``reset``): ``ct_node_u[nodes]`` (vec3d, node displacement), ``ct_jn`` / ``ct_jt``
(vec3d, this substep's impulses), the frame motion rows, the reactions of the control step
(``ct_plane_reaction[P]``, ``ct_box_reaction[F*B]``, spatial_vectord), the cumulative work ``ct_work[1, 4]``
(``rubber_contact.WORK_COLUMNS``), the control step's max penetration ``ct_pen_max[1]`` and event counts
``ct_counts[1, 3]``. Per-env frictions: ``ct_fric[1, 2]`` = (friction_table, friction_pad) as float32 (the batched
path reads the same values from ``params`` columns 3 and 4).
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from . import rubber_constitutive as rc
from . import rubber_contact as rct
from . import tlmpm_rubber_v1 as T
from . import tlmpm_v5 as v5
from .rubber_contact import ContactSpec
from .tlmpm_rubber_visco_v1 import (DEFAULT_VISCO_ENERGY, KICK_AVERAGE_ENERGY, TLMPMRubberVisco, cfl_guard_prony,
                                    visco_branch_work, visco_branch_work_kick_average)
from .tlmpm_v6 import _grid_positions

wp.set_module_options({"enable_backward": False})

SOLVER_NAME = "tlmpm_rubber_contact_v1"
STATUS = "EXPERIMENTAL_NOT_ACCEPTED"
_HERE = Path(__file__).resolve()
CONTACT_RESET_ARRAYS = ("ct_node_u", "ct_jn", "ct_jt", "ct_plane_reaction", "ct_box_reaction", "ct_work",
                        "ct_pen_max", "ct_counts", "ct_fb0", "ct_fb1", "ct_xi", "ct_vobs", "ct_energy")
CONTACT_CONSTANT_ARRAYS = ("ct_planes", "ct_fkind", "ct_forigin", "ct_box_c", "ct_box_h", "ct_env_of_node",
                           "ct_fric")


def source_sha256():
    return {_HERE.name: hashlib.sha256(_HERE.read_bytes()).hexdigest(), **rct.source_sha256()}


class TLMPMRubberContact(TLMPMRubberVisco):
    """``TLMPMRubberVisco`` (elastic when the material has no Prony terms) plus the ``grid_node_coulomb_v1`` contact
    law. Same constructor plus ``contact`` (a :class:`rubber_contact.ContactSpec`)."""
    solver_name = SOLVER_NAME

    def __init__(self, scene, material, dt, *, contact, **kwargs):
        if not isinstance(contact, ContactSpec):
            raise TypeError("contact must be a rubber_contact.ContactSpec")
        if kwargs.get("table_plane") is not None or kwargs.get("fixture_boxes") is not None:
            raise ValueError("table_plane / fixture_boxes are the frozen per-particle contact path; with a ContactSpec "
                             "pass the obstacles in the spec (one contact law per solver, ADR 0023 §22)")
        super().__init__(scene, material, dt, **kwargs)
        self.contact = contact
        P, F, B = contact.P, contact.F, contact.B
        self.ct_P, self.ct_F, self.ct_B = P, F, B
        self.ct_beta = float(contact.penetration_recovery)
        self.ct_penalty = contact.law in rct.PENALTY_LAWS
        self.ct_law = contact.law                    # scalar: selects the kick kernel (v1 / v2 penalty, projection)
        self.ct_kn, self.ct_kt = float(contact.normal_stiffness), float(contact.tangential_stiffness)
        self.ct_zeta = float(contact.damping_ratio)
        self.ct_zeta_n = float(contact.normal_damping_ratio)
        centres, halves = contact.box_arrays()
        F1, FB1 = max(F, 1), max(F*B, 1)
        ident_q = np.tile(np.array([0, 0, 0, 1], np.float64), (F1, 1))
        with wp.ScopedDevice(self.device):
            self.ct_node_u = wp.zeros(self.node_count, dtype=wp.vec3d)
            self.ct_jn = wp.zeros(self.node_count, dtype=wp.vec3d)
            self.ct_jt = wp.zeros(self.node_count, dtype=wp.vec3d)
            self.ct_env_of_node = wp.zeros(self.node_count, dtype=int)
            self.ct_fric = wp.array(np.array([[material.friction_table, material.friction_pad]], np.float64
                                             ).astype(np.float32), dtype=float)
            self.ct_planes = wp.array(contact.plane_array() if P else np.zeros((1, 4)), dtype=wp.vec4d)
            self.ct_fkind = wp.array(np.array([rct.FRAME_KINDS[f.kind] for f in contact.frames] or [0], np.int32),
                                     dtype=int)
            self.ct_factive = wp.array(np.ones(F1, np.int32), dtype=int)
            self.ct_fq0 = wp.array(ident_q, dtype=wp.quatd)
            self.ct_fq1 = wp.array(ident_q, dtype=wp.quatd)
            self.ct_fb0 = wp.zeros(F1, dtype=wp.vec3d)
            self.ct_fb1 = wp.zeros(F1, dtype=wp.vec3d)
            self.ct_forigin = wp.array(np.array([f.origin for f in contact.frames] or [[0, 0, 0]], np.float64),
                                       dtype=wp.vec3d)
            self.ct_box_c = wp.array(centres if F else np.zeros((1, 3)), dtype=wp.vec3d)
            self.ct_box_h = wp.array(halves if F else np.ones((1, 3)), dtype=wp.vec3d)
            self.ct_plane_reaction = wp.zeros(max(P, 1), dtype=wp.spatial_vectord)
            self.ct_box_reaction = wp.zeros(FB1, dtype=wp.spatial_vectord)
            self.ct_work = wp.zeros((1, rct.NWORK), dtype=wp.float64)
            self.ct_pen_max = wp.zeros(1, dtype=wp.float64)
            self.ct_counts = wp.zeros((1, len(rct.COUNT_COLUMNS)), dtype=int)
            # penalty law: Coulomb spring-slider state per node (elongation, obstacle id, obstacle velocity)
            self.ct_xi = wp.zeros(self.node_count, dtype=wp.vec3d)
            self.ct_obst = wp.full(self.node_count, -1, dtype=int)
            self.ct_vobs = wp.zeros(self.node_count, dtype=wp.vec3d)
            self.ct_energy = wp.zeros((1, 2), dtype=wp.float64)
            self.ct_node_offset = wp.array(np.array([0, self.node_count], np.int32), dtype=int)
        wp.load_module(module=rct.__name__, device=self.device)

    # ------------------------------------------------------------------ inputs

    def set_frame_motion(self, f, *, rotation=None, translation=None, active=None):
        """END pose of contact frame ``f`` in the current control step (after ``begin_control_step``, which makes
        every frame hold), x = R X + b relative to the reference (as ``set_group_motion`` for rigid groups)."""
        if not 0 <= f < self.ct_F:
            raise IndexError(f"contact frame {f} not in [0, {self.ct_F})")
        if rotation is not None:
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(np.asarray(rotation, float)).as_quat()
            prev = self.ct_fq1.numpy()[f]
            if np.dot(prev, q) < 0:
                q = -q
            self._set_row(self.ct_fq1, f, q)
        if translation is not None:
            self._set_row(self.ct_fb1, f, np.asarray(translation, float))
        if active is not None:
            self._set_row(self.ct_factive, f, int(bool(active)))

    def begin_control_step(self):
        super().begin_control_step()
        wp.copy(self.ct_fq0, self.ct_fq1)
        wp.copy(self.ct_fb0, self.ct_fb1)
        for a in (self.ct_plane_reaction, self.ct_box_reaction, self.ct_pen_max, self.ct_counts):
            a.zero_()

    # ------------------------------------------------------------------ substep

    def _launch_contact(self):
        if self.ct_penalty:
            # grid_node_penalty_coulomb_v1 (default) or _v2 (ADR 0023 §22.9: same arguments, dashpots on the realised
            # node velocity); one source for the single and the batched solver
            kernel = rct.node_penalty_contact_v2 if self.ct_law == rct.PENALTY_LAW_V2 else rct.node_penalty_contact
            wp.launch(kernel, self.node_count,
                      [self.gm, self.new_v, self.old, self.node_X, self.ct_node_u, self.node_group, self.group_active,
                       self.ct_env_of_node, self.ct_fric, 0, 1, self.ct_P, self.ct_planes, self.ct_F, self.ct_B,
                       self.ct_fkind, self.ct_factive, self.ct_fq0, self.ct_fq1, self.ct_fb0, self.ct_fb1,
                       self.ct_forigin, self.ct_box_c, self.ct_box_h, self.index, self.substeps, wp.float64(self.dt),
                       wp.float64(self.ct_kn), wp.float64(self.ct_kt), wp.float64(self.ct_zeta), wp.float64(self.ct_zeta_n), self.ct_xi,
                       self.ct_obst, self.ct_vobs,
                       self.ct_jn, self.ct_jt, self.ct_plane_reaction, self.ct_box_reaction, self.ct_work,
                       self.ct_pen_max, self.ct_counts], device=self.device)
            return
        wp.launch(rct.node_contact, self.node_count,
                  [self.gm, self.new_v, self.old, self.node_X, self.ct_node_u, self.node_group, self.group_active,
                   self.ct_env_of_node, self.ct_fric, 0, 1, self.ct_P, self.ct_planes, self.ct_F, self.ct_B,
                   self.ct_fkind, self.ct_factive, self.ct_fq0, self.ct_fq1, self.ct_fb0, self.ct_fb1,
                   self.ct_forigin, self.ct_box_c, self.ct_box_h, self.index, self.substeps, wp.float64(self.dt),
                   wp.float64(self.ct_beta), self.ct_jn, self.ct_jt, self.ct_plane_reaction, self.ct_box_reaction,
                   self.ct_work, self.ct_pen_max, self.ct_counts], device=self.device)

    def step(self):
        """``TLMPMRubberVisco.step`` with the three contact launches (module docstring)."""
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
            self._launch_contact()
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
            wp.launch(rct.contact_work_end, nn, [self.ct_jn, self.ct_jt, self.old, self.ct_env_of_node, self.ct_work])
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
            wp.launch(rct.node_displacement, nn, [self.ct_node_u, self.new_v, wp.float64(dt)])
            if self.ct_penalty:
                wp.launch(rct.slider_update, nn, [self.ct_xi, self.ct_obst, self.ct_vobs, self.new_v, wp.float64(dt)])
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

    # ------------------------------------------------------------------ episode / queries

    def reset(self):
        super().reset()
        for name in CONTACT_RESET_ARRAYS:
            getattr(self, name).zero_()
        self.ct_obst.fill_(-1)
        F1 = max(self.ct_F, 1)
        ident_q = np.tile(np.array([0, 0, 0, 1], np.float64), (F1, 1))
        self.ct_fq0.assign(ident_q)
        self.ct_fq1.assign(ident_q)
        self.ct_factive.fill_(1)

    def set_initial_velocity(self, velocity):
        """Uniform rigid initial velocity of every particle (tests: drop / impact). Grid-consistent (v = W^T a with a
        uniform); call right after ``reset``."""
        self.v.assign(np.tile(np.asarray(velocity, np.float32), (self.n, 1)))

    def contact_wrenches(self, control_dt):
        """Wrenches the material exerts on the obstacles, averaged over the control step: planes [P, 6] (torque about
        the world origin) and boxes [F, B, 6] (torque about the frame's current origin)."""
        cdt = float(control_dt)
        planes = -self.ct_plane_reaction.numpy()[:self.ct_P]/cdt
        boxes = -self.ct_box_reaction.numpy()[:self.ct_F*self.ct_B].reshape(self.ct_F, self.ct_B, 6)/cdt
        return planes, boxes

    def contact_energy(self):
        """Penalty law: stored contact energy (Phi_n, Phi_t) of the current state (zeros for the projection law)."""
        if not self.ct_penalty:
            return 0.0, 0.0
        wp.launch(rct.contact_energy, 1, [self.node_X, self.ct_node_u, self.gm, self.ct_env_of_node, self.ct_fric, 0, 1,
                                          self.ct_P, self.ct_planes, self.ct_F, self.ct_B, self.ct_fkind,
                                          self.ct_factive, self.ct_fq1, self.ct_fb1, self.ct_box_c, self.ct_box_h,
                                          wp.float64(self.dt), wp.float64(self.ct_kn), wp.float64(self.ct_kt),
                                          self.ct_xi, self.ct_obst, self.ct_node_offset, self.ct_energy],
                  device=self.device)
        e = self.ct_energy.numpy()[0]
        return float(e[0]), float(e[1])

    def contact_diagnostics(self):
        work = self.ct_work.numpy()[0]
        counts = self.ct_counts.numpy()[0]
        out = dict(zip(rct.WORK_COLUMNS, map(float, work)))
        out["contact_work_material"] = float(work[0]+work[1])
        out["friction_dissipation"] = float(work[3]-work[1])
        out["normal_dissipation"] = float(work[2]-work[0])
        phin, phit = self.contact_energy()
        out["contact_energy_normal"], out["contact_energy_tangential"] = phin, phit
        out["max_penetration_m"] = float(self.ct_pen_max.numpy()[0])
        out.update({k: int(c) for k, c in zip(rct.COUNT_COLUMNS, counts)})
        return out

    def node_positions(self):
        return self.node_X.numpy()+self.ct_node_u.numpy()

    def solver_receipt(self):
        out = super().solver_receipt()
        out["contact_solver"] = SOLVER_NAME
        out["contact"] = {**self.contact.receipt(), "status": STATUS,
                          "frozen_particle_contact_path": "off (refused together with a ContactSpec)",
                          "friction": {"table_and_fixtures": float(self.material.friction_table),
                                       "pads": float(self.material.friction_pad)},
                          "state": {"ct_node_u": "vec3d float64 [nodes]", "ct_jn/ct_jt": "vec3d float64 [nodes]"},
                          "self_contact": "NOT_IMPLEMENTED_V1 (ADR 0023 §22.3)"}
        out["options"]["contact_law"] = self.contact.law
        out["options"]["contact_work"] = self.contact.work
        out["source_sha256"] = {**out["source_sha256"], **source_sha256()}
        return out


__all__ = ["TLMPMRubberContact", "CONTACT_RESET_ARRAYS", "CONTACT_CONSTANT_ARRAYS", "DEFAULT_VISCO_ENERGY"]
