"""Batched rubber contact ``tlmpm_rubber_contact_v1_batch`` (ADR 0023 §22.2, §22.5; P2-4 task B1).

Opt-in: ``BatchedRubberMaterial`` builds this solver only when its single references were built with a
``rubber_contact.ContactSpec`` (``LiveRubberMaterial(contact=...)`` -> ``TLMPMRubberContact``); without one the batch is
the unchanged ``BatchedTLMPMRubber`` (its three hook methods launch nothing).

The law is NOT re-implemented here. The kick launch is the single solver's own ``TLMPMRubberContact._launch_contact``
(bound to this class), so the kernels of ``rubber_contact`` (``node_penalty_contact`` / ``node_contact``) run with the
batched arrays under the single solver's attribute names; ``contact_work_end``, ``node_displacement`` and
``slider_update`` are launched at the single solver's points through the hooks of ``BatchedTLMPMRubber.step``:

- after the P2G ``_grid_update``: the contact kick (``_launch_contact``);
- after the final MUSL ``_grid_update``: ``rubber_contact.contact_work_end``;
- after the position update: ``rubber_contact.node_displacement`` and, for the penalty law, ``slider_update``.

Batched arrays (``rubber_layout.CONTACT_ARRAY_KINDS``; a single-solver contact array without a declared row kind is
refused, so a new law version must declare its layout): node rows at the env's node range; per-env rows ``ct_work[N, 6]``,
``ct_pen_max[N]``, ``ct_counts[N, 3]``, ``ct_energy[N, 2]``; plane rows ``e*P + k``, frame rows ``e*F + f``, box rows
``(e*F + f)*B + b``. Aliases: ``ct_env_of_node`` is ``env_of_node``, ``ct_node_offset`` is ``node_offset`` and
``ct_fric`` is the strided view ``params[:, 3:5]`` (friction_table, friction_pad: the per-env frictions are the
``params`` columns, so a masked reset with resampled frictions changes them with no extra array). The scalar ``ct_*``
attributes are copied from the single reference (uniform signature). Every write of the contact kernels goes to the
node's own row or to rows of its env (``env_of_node``), so on the Warp CPU device each env's contact state, reactions,
work and counts are byte-identical to its single reference (RUBBER-CONT-08); on a GPU the reaction / work atomics are
order-nondeterministic, the per-node state is not (RUBBER-CONT-09).

Material outputs (``BatchedRubberMaterial`` with contact, after each control step; float64): ``plane_wrench[N, P, 6]``
(torque about the world origin), ``box_wrench[N, F, B, 6]`` (torque about the frame's current origin) - both
``-reaction / control_dt`` as ``TLMPMRubberContact.contact_wrenches`` - and the Phase 1 ``MaterialCoupling`` arrays
``pad_wrench[N, G*B, 6]`` (the box rows of the G frames of kind ``pad`` in frame order: per-pad wrenches [N, G, B, 6]
flattened; ``pad_wrenches()`` returns the [N, G, B, 6] view) and ``pending_wrench[N, 6]`` (zero: rubber v1 has no
edge / tool contact; robot links other than the pads do not touch the strip, ADR 0023 §22.2). These are strip-local
values (= world for an identity strip pose); the strip-to-world transform and the finger recipients are task B3. The
masked reset clears every contact array of the reset envs (``CONTACT_RESET_ARRAYS``).

Status ``EXPERIMENTAL_NOT_ACCEPTED``; material ``SIMULATOR_ONLY_UNCALIBRATED``.
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from .. import rubber_contact as rct
from ..tlmpm_rubber_contact_v1 import TLMPMRubberContact
from .constitutive_rubber import P_FRICTION_PAD, P_FRICTION_TABLE
from .rubber_batch import BatchedTLMPMRubber
from .rubber_layout import CONTACT_ARRAY_KINDS

wp.set_module_options({"enable_backward": False})

SOLVER_NAME = "tlmpm_rubber_contact_v1_batch"
STATUS = "EXPERIMENTAL_NOT_ACCEPTED"
_HERE = Path(__file__).resolve()

# vars()-complete classification of the batched contact arrays (tests/test_rubber_contact_batched.py)
CONTACT_SOLVER_RESET_ARRAYS = ("ct_node_u", "ct_jn", "ct_jt", "ct_xi", "ct_obst", "ct_vobs", "ct_work", "ct_pen_max",
                               "ct_counts", "ct_energy", "ct_plane_reaction", "ct_box_reaction", "ct_factive",
                               "ct_fq0", "ct_fq1", "ct_fb0", "ct_fb1")
CONTACT_SOLVER_CONSTANT_ARRAYS = ("ct_planes", "ct_fkind", "ct_forigin", "ct_box_c", "ct_box_h")
CONTACT_SOLVER_ALIAS_ARRAYS = ("ct_env_of_node", "ct_node_offset", "ct_fric")   # env_of_node, node_offset, params view
CONTACT_MATERIAL_RESET_ARRAYS = ("plane_wrench", "box_wrench", "pad_wrench", "pending_wrench")
CONTACT_MATERIAL_CONSTANT_ARRAYS = ("pad_frames",)


def source_sha256():
    return {_HERE.name: hashlib.sha256(_HERE.read_bytes()).hexdigest(), **rct.source_sha256()}


# ----------------------------------------------------------------------------- kernels (material side)

@wp.kernel
def _contact_wrench_b(plane_reaction: wp.array(dtype=wp.spatial_vectord),
                      box_reaction: wp.array(dtype=wp.spatial_vectord), control_dt: wp.float64, P: int, F: int,
                      B: int, pad_frames: wp.array(dtype=int), G: int, plane_wrench: wp.array3d(dtype=wp.float64),
                      box_wrench: wp.array4d(dtype=wp.float64), pad_wrench: wp.array3d(dtype=wp.float64)):
    """-reaction / control_dt of env e's planes, boxes and pad boxes, written (r * -1) / dt (byte-identical to the
    single reference's NumPy ``-reaction / control_dt`` for every finite value and both zeros, ADR 0023 §20.4)."""
    e = wp.tid()
    for k in range(P):
        r = plane_reaction[e*P+k]
        for j in range(6):
            plane_wrench[e, k, j] = (r[j]*wp.float64(-1.0))/control_dt
    for f in range(F):
        for b in range(B):
            r = box_reaction[(e*F+f)*B+b]
            for j in range(6):
                box_wrench[e, f, b, j] = (r[j]*wp.float64(-1.0))/control_dt
    for g in range(G):
        f = pad_frames[g]
        for b in range(B):
            for j in range(6):
                pad_wrench[e, g*B+b, j] = box_wrench[e, f, b, j]


@wp.kernel
def _contact_validity_b(plane_wrench: wp.array3d(dtype=wp.float64), box_wrench: wp.array4d(dtype=wp.float64),
                        P: int, F: int, B: int, invalid_reason: wp.array(dtype=int)):
    """NONFINITE_STATE (bit 1) for a non-finite contact wrench (launched before the frozen validity kernel, which
    counts the env)."""
    e = wp.tid()
    bad = int(0)
    for k in range(P):
        for j in range(6):
            if not wp.isfinite(plane_wrench[e, k, j]):
                bad = 1
    for f in range(F):
        for b in range(B):
            for j in range(6):
                if not wp.isfinite(box_wrench[e, f, b, j]):
                    bad = 1
    if bad != 0:
        invalid_reason[e] = invalid_reason[e] | 1


@wp.kernel
def _reset_contact_nodes_r(mask: wp.array(dtype=int), env_of_node: wp.array(dtype=int),
                           u: wp.array(dtype=wp.vec3d), jn: wp.array(dtype=wp.vec3d), jt: wp.array(dtype=wp.vec3d),
                           xi: wp.array(dtype=wp.vec3d), vobs: wp.array(dtype=wp.vec3d), obst: wp.array(dtype=int)):
    i = wp.tid()
    if mask[env_of_node[i]] != 0:
        u[i] = wp.vec3d()
        jn[i] = wp.vec3d()
        jt[i] = wp.vec3d()
        xi[i] = wp.vec3d()
        vobs[i] = wp.vec3d()
        obst[i] = -1


@wp.kernel
def _reset_contact_frames_r(mask: wp.array(dtype=int), F: int, active: wp.array(dtype=int),
                            q0: wp.array(dtype=wp.quatd), q1: wp.array(dtype=wp.quatd), b0: wp.array(dtype=wp.vec3d),
                            b1: wp.array(dtype=wp.vec3d)):
    r = wp.tid()
    if mask[r//F] != 0:
        active[r] = 1
        q0[r] = wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        q1[r] = wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        b0[r] = wp.vec3d()
        b1[r] = wp.vec3d()


@wp.kernel
def _reset_contact_rows_r(mask: wp.array(dtype=int), per_env: int, rows: wp.array(dtype=wp.spatial_vectord)):
    r = wp.tid()
    if mask[r//per_env] != 0:
        rows[r] = wp.spatial_vectord()


@wp.kernel
def _reset_contact_envs_r(mask: wp.array(dtype=int), work: wp.array2d(dtype=wp.float64),
                          pen_max: wp.array(dtype=wp.float64), counts: wp.array2d(dtype=int),
                          energy: wp.array2d(dtype=wp.float64), plane_wrench: wp.array3d(dtype=wp.float64),
                          box_wrench: wp.array4d(dtype=wp.float64), pad_wrench: wp.array3d(dtype=wp.float64),
                          pending_wrench: wp.array2d(dtype=wp.float64)):
    e = wp.tid()
    if mask[e] != 0:
        for k in range(work.shape[1]):
            work[e, k] = wp.float64(0.0)
        pen_max[e] = wp.float64(0.0)
        for k in range(counts.shape[1]):
            counts[e, k] = 0
        for k in range(energy.shape[1]):
            energy[e, k] = wp.float64(0.0)
        for k in range(plane_wrench.shape[1]):
            for j in range(6):
                plane_wrench[e, k, j] = wp.float64(0.0)
        for f in range(box_wrench.shape[1]):
            for b in range(box_wrench.shape[2]):
                for j in range(6):
                    box_wrench[e, f, b, j] = wp.float64(0.0)
        for k in range(pad_wrench.shape[1]):
            for j in range(6):
                pad_wrench[e, k, j] = wp.float64(0.0)
        for j in range(6):
            pending_wrench[e, j] = wp.float64(0.0)


# ----------------------------------------------------------------------------- solver

class BatchedTLMPMRubberContact(BatchedTLMPMRubber):
    """``BatchedTLMPMRubber`` plus the contact launches of ``TLMPMRubberContact`` (module docstring)."""
    solver_name = SOLVER_NAME
    # the single solver's contact kick launch, unchanged (same kernels, same argument order; batched arrays under the
    # single solver's names)
    _launch_contact = TLMPMRubberContact._launch_contact

    def __init__(self, packed, device, **kwargs):
        super().__init__(packed, device, **kwargs)
        sig = packed.uniform["contact"]
        for name, value in sig["scalars"]:                    # ct_P, ct_F, ct_B, ct_kn, ..., ct_penalty
            setattr(self, name, value)
        self.contact_signature = sig
        self.frame_kinds = sig["frame_kinds"]
        self.node_count = self.total_nodes
        h, dtypes = packed.host, packed.env["contact_dtypes"]
        with wp.ScopedDevice(self.device):
            for name, kind in CONTACT_ARRAY_KINDS.items():
                if not kind.startswith("alias"):
                    setattr(self, name, wp.array(h[name], dtype=dtypes[name]))
        self.ct_env_of_node = self.env_of_node
        self.ct_node_offset = self.node_offset
        self.ct_fric = self.params[:, int(P_FRICTION_TABLE):int(P_FRICTION_PAD)+1]
        if self.ct_fric.shape != (self.num_envs, 2):
            raise AssertionError("params friction columns")
        self._frame_identity = np.tile(np.array([0, 0, 0, 1], np.float64), (max(self.num_envs*self.ct_F, 1), 1))
        wp.load_module(module=rct.__name__, device=self.device)
        wp.load_module(module=__name__, device=self.device)

    # ------------------------------------------------------------------ substep hooks (BatchedTLMPMRubber.step)

    def _after_p2g_grid_update(self):
        self._launch_contact()

    def _after_final_grid_update(self):
        wp.launch(rct.contact_work_end, self.total_nodes, [self.ct_jn, self.ct_jt, self.old, self.ct_env_of_node,
                                                           self.ct_work], device=self.device)

    def _after_position_update(self):
        wp.launch(rct.node_displacement, self.total_nodes, [self.ct_node_u, self.new_v, wp.float64(self.dt)],
                  device=self.device)
        if self.ct_penalty:
            wp.launch(rct.slider_update, self.total_nodes, [self.ct_xi, self.ct_obst, self.ct_vobs, self.new_v,
                                                            wp.float64(self.dt)], device=self.device)

    # ------------------------------------------------------------------ control step

    def begin_contact_step(self):
        """``TLMPMRubberContact.begin_control_step`` (contact part): every frame holds (start = end), the per-step
        reactions, max penetration and event counts are zeroed; ``ct_work`` is cumulative over the episode."""
        wp.copy(self.ct_fq0, self.ct_fq1)
        wp.copy(self.ct_fb0, self.ct_fb1)
        for a in (self.ct_plane_reaction, self.ct_box_reaction, self.ct_pen_max, self.ct_counts):
            a.zero_()

    def set_frame_motions(self, *, rotation=None, translation=None, active=None):
        """Host [N, F, ...] END poses of the contact frames in this control step (as ``set_frame_motion`` of the
        single solver for every frame; x = R X + b relative to the reference)."""
        N, F = self.num_envs, self.ct_F
        if F == 0:
            if rotation is not None or translation is not None or active is not None:
                raise IndexError("frame motions given for a batch without contact frames")
            return
        if rotation is not None:
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(np.asarray(rotation, float).reshape(N*F, 3, 3)).as_quat()
            prev = self.ct_fq1.numpy()
            q[np.einsum("ij,ij->i", prev, q) < 0] *= -1
            self.ct_fq1.assign(q)
        if translation is not None:
            self.ct_fb1.assign(np.asarray(translation, float).reshape(N*F, 3))
        if active is not None:
            self.ct_factive.assign(np.asarray(active, np.int32).reshape(N*F))

    def launch_contact_energy(self):
        """Penalty law: stored contact energy (Phi_n, Phi_t) of every env into ``ct_energy[N, 2]`` (the single
        solver's ``contact_energy`` kernel, one thread per env)."""
        if not self.ct_penalty:
            return                                # as the single solver: Phi = 0, ct_energy untouched
        wp.launch(rct.contact_energy, self.num_envs,
                  [self.node_X, self.ct_node_u, self.gm, self.ct_env_of_node, self.ct_fric, 0, 1, self.ct_P,
                   self.ct_planes, self.ct_F, self.ct_B, self.ct_fkind, self.ct_factive, self.ct_fq1, self.ct_fb1,
                   self.ct_box_c, self.ct_box_h, wp.float64(self.dt), wp.float64(self.ct_kn), wp.float64(self.ct_kt),
                   self.ct_xi, self.ct_obst, self.ct_node_offset, self.ct_energy], device=self.device)

    def set_initial_velocity(self, env_ids, velocities):
        """Uniform rigid initial velocity per listed env (``TLMPMRubberContact.set_initial_velocity``: float32 rows of
        the env's particles; call right after a reset)."""
        v = self.v.numpy()
        po = self.packed.particle_offset
        for e, vel in zip(np.asarray(env_ids, int).reshape(-1), np.asarray(velocities, float).reshape(-1, 3)):
            v[po[e]:po[e+1]] = np.asarray(vel, np.float32)
        self.v.assign(v)

    # ------------------------------------------------------------------ queries

    def snapshot(self, e):
        out = super().snapshot(e)
        pk = self.packed
        n0, n1 = int(pk.node_offset[e]), int(pk.node_offset[e+1])
        rows = {"plane": self.ct_P, "frame": self.ct_F, "box": self.ct_F*self.ct_B}
        for name, kind in CONTACT_ARRAY_KINDS.items():
            if kind.startswith("alias"):
                continue
            a = getattr(self, name).numpy()
            if kind == "node":
                out[name] = a[n0:n1].copy()
            elif kind == "env":
                out[name] = a[e:e+1].copy()
            else:
                k = rows[kind]
                out[name] = a[e*k:(e+1)*k].copy()
        out["ct_env_of_node"] = self.env_of_node.numpy()[n0:n1]-np.int32(e)
        out["ct_fric"] = self.ct_fric.numpy()[e:e+1].copy()
        return out

    def contact_snapshot_names(self):
        return tuple(k for k, kind in CONTACT_ARRAY_KINDS.items() if kind != "alias_node_offset")

    def options_receipt(self):
        out = super().options_receipt()
        out["contact_law"] = dict(self.contact_signature["parameters"])["law"]
        out["contact_work"] = dict(self.contact_signature["parameters"])["work"]
        return out

    def solver_receipt(self):
        out = super().solver_receipt()
        out["solver"] = SOLVER_NAME
        out["contact_solver"] = "tlmpm_rubber_contact_v1"
        out["contact"] = {"status": STATUS, "P": self.ct_P, "F": self.ct_F, "B": self.ct_B,
                          "frame_kinds": list(self.frame_kinds),
                          "parameters": {k: v for k, v in self.contact_signature["parameters"]},
                          "kick_launch": "TLMPMRubberContact._launch_contact (single solver, unchanged)",
                          "frictions": "params columns friction_table / friction_pad (per env)",
                          "self_contact": "NOT_IMPLEMENTED_V1 (ADR 0023 §22.3)", "adr": "ADR 0023 §22.2, §22.5"}
        out["source_sha256"] = {**out["source_sha256"], **source_sha256()}
        return out


# ----------------------------------------------------------------------------- material helpers

class ContactOutputs:
    """Material-side contact arrays of a ``BatchedRubberMaterial`` (allocated once; module docstring)."""

    def __init__(self, solver, device):
        s = solver
        N, P, F, B = s.num_envs, s.ct_P, s.ct_F, s.ct_B
        pads = [f for f, kind in enumerate(s.frame_kinds) if kind == "pad"]
        self.P, self.F, self.B, self.G = P, F, B, len(pads)
        with wp.ScopedDevice(device):
            self.plane_wrench = wp.zeros((N, max(P, 1), 6), dtype=wp.float64)
            self.box_wrench = wp.zeros((N, max(F, 1), max(B, 1), 6), dtype=wp.float64)
            self.pad_wrench = wp.zeros((N, max(self.G*B, 1), 6), dtype=wp.float64)
            self.pending_wrench = wp.zeros((N, 6), dtype=wp.float64)
            self.pad_frames = wp.array(np.array(pads or [0], np.int32), dtype=int)

    def arrays(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, wp.array)}

    def launch_wrench(self, solver, control_dt, invalid_reason):
        s = solver
        wp.launch(_contact_wrench_b, s.num_envs, [s.ct_plane_reaction, s.ct_box_reaction, wp.float64(control_dt),
                                                  self.P, self.F, self.B, self.pad_frames, self.G, self.plane_wrench,
                                                  self.box_wrench, self.pad_wrench], device=s.device)
        wp.launch(_contact_validity_b, s.num_envs, [self.plane_wrench, self.box_wrench, self.P, self.F, self.B,
                                                    invalid_reason], device=s.device)
        s.launch_contact_energy()      # every control step, as LiveRubberMaterial.advance -> contact_diagnostics

    def launch_reset(self, solver, mask):
        s = solver
        N = s.num_envs
        wp.launch(_reset_contact_nodes_r, s.total_nodes, [mask, s.env_of_node, s.ct_node_u, s.ct_jn, s.ct_jt, s.ct_xi,
                                                          s.ct_vobs, s.ct_obst], device=s.device)
        if s.ct_F:
            wp.launch(_reset_contact_frames_r, N*s.ct_F, [mask, s.ct_F, s.ct_factive, s.ct_fq0, s.ct_fq1, s.ct_fb0,
                                                          s.ct_fb1], device=s.device)
            wp.launch(_reset_contact_rows_r, N*s.ct_F*s.ct_B, [mask, s.ct_F*s.ct_B, s.ct_box_reaction],
                      device=s.device)
        if s.ct_P:
            wp.launch(_reset_contact_rows_r, N*s.ct_P, [mask, s.ct_P, s.ct_plane_reaction], device=s.device)
        wp.launch(_reset_contact_envs_r, N, [mask, s.ct_work, s.ct_pen_max, s.ct_counts, s.ct_energy,
                                             self.plane_wrench, self.box_wrench, self.pad_wrench,
                                             self.pending_wrench], device=s.device)

    def read(self, solver, e=None):
        """Host contact block per env, as ``LiveRubberMaterial.last_contact`` (plane_wrench [P, 6], box_wrench
        [F, B, 6], the ``TLMPMRubberContact.contact_diagnostics`` keys) after the last control step."""
        s = solver
        work, counts = s.ct_work.numpy(), s.ct_counts.numpy()
        energy, pen = s.ct_energy.numpy(), s.ct_pen_max.numpy()
        pw, bw = self.plane_wrench.numpy(), self.box_wrench.numpy()
        out = []
        for k in (range(s.num_envs) if e is None else [e]):
            w = work[k]
            d = {"plane_wrench": pw[k, :self.P], "box_wrench": bw[k, :self.F, :self.B]}
            d.update(dict(zip(rct.WORK_COLUMNS, map(float, w))))
            d["contact_work_material"] = float(w[0]+w[1])
            d["friction_dissipation"] = float(w[3]-w[1])
            d["normal_dissipation"] = float(w[2]-w[0])
            d["contact_energy_normal"], d["contact_energy_tangential"] = (float(energy[k, 0]), float(energy[k, 1]))
            d["max_penetration_m"] = float(pen[k])
            d.update({c: int(v) for c, v in zip(rct.COUNT_COLUMNS, counts[k])})
            out.append(d)
        return out


def contact_audit(solver):
    """Env-ownership audits of the contact rows on host copies (all counts must be 0; RUBBER-CONT-09 c): node env ids
    against the node ranges, slider obstacle ids inside [-1, P + F*B) (env-local), row counts of the plane / frame /
    box tables (N*P, N*F, N*F*B) and the pad-frame table."""
    s = solver
    N, P, F, B = s.num_envs, s.ct_P, s.ct_F, s.ct_B
    no = s.packed.node_offset
    env_n = s.env_of_node.numpy()
    expected = np.repeat(np.arange(N, dtype=np.int32), np.diff(no))
    obst = s.ct_obst.numpy()
    out = {"contact_node_env_mismatch": int(np.count_nonzero(env_n != expected)),
           "contact_obstacle_id_out_of_env": int(np.count_nonzero((obst < -1) | (obst >= P+F*B)))}
    bad_rows = 0
    for name, rows in (("ct_planes", N*P), ("ct_plane_reaction", N*P), ("ct_fq0", N*F), ("ct_fq1", N*F),
                       ("ct_fb0", N*F), ("ct_fb1", N*F), ("ct_factive", N*F), ("ct_fkind", N*F), ("ct_forigin", N*F),
                       ("ct_box_c", N*F*B), ("ct_box_h", N*F*B), ("ct_box_reaction", N*F*B)):
        bad_rows += int(getattr(s, name).shape[0] != max(rows, 1))
    out["contact_row_table_size_mismatch"] = bad_rows
    out["contact_env_table_size_mismatch"] = int(any(getattr(s, n).shape[0] != N for n in
                                                     ("ct_work", "ct_pen_max", "ct_counts", "ct_energy")))
    return out


__all__ = ["BatchedTLMPMRubberContact", "ContactOutputs", "contact_audit", "CONTACT_SOLVER_RESET_ARRAYS",
           "CONTACT_SOLVER_CONSTANT_ARRAYS", "CONTACT_SOLVER_ALIAS_ARRAYS", "CONTACT_MATERIAL_RESET_ARRAYS",
           "CONTACT_MATERIAL_CONSTANT_ARRAYS"]
