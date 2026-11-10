"""Batched execution ``tlmpm_rubber_v1_batch`` of the single-env reference ``tlmpm_rubber_v1`` (ADR 0023 D1, D9; R1.7).

One launch per operation covers all N envs, in the single solver's launch order. UNCHANGED kernels: the frozen
``tlmpm_v5._p2g/_grid_update/_g2p/_musl/_deformation/_advance`` and ``tlmpm_v6._grid_positions`` (global node ids keep
the envs' node ranges disjoint), and ``rubber_constitutive.cell_jbar`` (global cell CSR). ENV kernels (``*_b``): the
single kernels' bodies with per-env loads (``params[e]``, plane / boxes / traction / group rows of env e) and per-env
accumulators (violations, constraint work), calling the same ``@wp.func`` helpers. Every float atomic target belongs to
one env whose particles / nodes are contiguous and in single-env order, so on the Warp CPU device (sequential kernels)
each target receives the same contributions in the same order as in the single solver: CPU parity is bitwise (L0).
On a GPU the frozen P2G/MUSL atomics are the only nondeterminism (J-bar is a gather; diagnostics are fixed-order).
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from .. import tlmpm_v5 as v5
from .. import rubber_constitutive as rc
from ..tlmpm_rubber_v1 import apply_mask, bc_reaction, bc_work, contact_load, group_target
from ..tlmpm_v6 import _grid_positions
from . import constitutive_rubber as cr
from .constitutive_rubber import P_FRICTION_TABLE
from .rubber_layout import GROUP_ARRAYS, NODE_ARRAYS, PARTICLE_ARRAYS, TRACTION_ARRAYS

wp.set_module_options({"enable_backward": False})

SOLVER_NAME = "tlmpm_rubber_v1_batch"
BASE_SOLVER = "tlmpm_rubber_v1"
STATUS = "EXPERIMENTAL_NOT_ACCEPTED"
_HERE = Path(__file__).resolve()


def source_sha256():
    here = _HERE.parent
    out = {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
           for name in ("rubber_batch.py", "rubber_layout.py", "constitutive_rubber.py", "rubber_material_batch.py")}
    pkg = here.parent
    out.update({name: hashlib.sha256((pkg/name).read_bytes()).hexdigest()
                for name in ("tlmpm_rubber_v1.py", "rubber_constitutive.py", "tlmpm_v5.py", "tlmpm_v6.py")})
    return out


@wp.kernel
def _loads_b(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
             impulse: wp.array(dtype=wp.vec3), particle_traction: wp.array(dtype=int),
             traction_force: wp.array(dtype=wp.vec3), traction_mass: wp.array(dtype=float), dt: float,
             radius: float, stiffness: float, params: wp.array2d(dtype=float), use_plane: int,
             plane: wp.array(dtype=wp.vec4), boxes: int, box_c: wp.array(dtype=wp.vec3), box_h: wp.array(dtype=wp.vec3),
             env_of_particle: wp.array(dtype=int), work: wp.array(dtype=wp.float64)):
    """``tlmpm_rubber_v1._loads`` (dynamics: no damping) with friction, plane, boxes and traction rows of env e."""
    p = wp.tid()
    e = env_of_particle[p]
    xp, vp, mp = x[p], v[p], mass[p]
    imp = contact_load(xp, vp, mp, dt, radius, stiffness, params[e, P_FRICTION_TABLE], use_plane, plane[e],
                       e*boxes, e*boxes+boxes, box_c, box_h)
    t = particle_traction[p]
    if t >= 0:
        jt = traction_force[t]*(dt*mp/traction_mass[t])
        imp = imp+jt
        wp.atomic_add(work, e, wp.dot(wp.vec3d(jt), wp.vec3d(vp)))
    impulse[p] = imp


@wp.kernel
def _bc_b(node_group: wp.array(dtype=int), node_mask: wp.array(dtype=int), node_X: wp.array(dtype=wp.vec3d),
          gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), index: wp.array(dtype=int), substeps: int,
          dt: wp.float64, kind: wp.array(dtype=int), active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd),
          q1: wp.array(dtype=wp.quatd), b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d),
          A0: wp.array(dtype=wp.mat33d), A1: wp.array(dtype=wp.mat33d), origin: wp.array(dtype=wp.vec3d),
          reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array(dtype=wp.float64),
          env_of_group: wp.array(dtype=int), record: int):
    """``tlmpm_rubber_v1._bc`` with global group ids and the constraint work of the group's env."""
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
                wp.atomic_add(work, env_of_group[g], bc_work(gm[i], v, vn))


class BatchedTLMPMRubber:
    """N independent rubber envs in flat arrays; ``step()`` is one substep of every env (see module docstring)."""
    solver_name = SOLVER_NAME

    _DTYPES = {
        **dict.fromkeys(("x", "v", "reference", "impulse", "x_start", "gp", "gj", "old", "new_v", "traction_force",
                         "box_c", "box_h", "gradients"), wp.vec3),
        **dict.fromkeys(("F", "stress"), wp.mat33),
        **dict.fromkeys(("mass", "weights", "gm", "traction_mass", "params"), float),
        **dict.fromkeys(("nodes", "slot", "slot_code", "node_group", "node_mask", "particle_traction", "group_kind",
                         "group_active", "cell_node", "cell_offsets", "cell_particles", "marker_offsets",
                         "marker_particles", "env_of_particle", "env_of_node", "env_of_group", "particle_offset",
                         "node_offset", "cell_env_offset"), int),
        "jbar": wp.float64, "node_X": wp.vec3d, "group_origin": wp.vec3d, "group_q0": wp.quatd, "group_q1": wp.quatd,
        "group_b0": wp.vec3d, "group_b1": wp.vec3d, "group_A0": wp.mat33d, "group_A1": wp.mat33d,
        "reaction": wp.spatial_vectord, "plane": wp.vec4,
    }

    def __init__(self, packed, device):
        self.packed = packed
        self.device = str(wp.get_device(device))
        u = packed.uniform
        self.num_envs = packed.num_envs
        self.total_particles, self.total_nodes, self.total_cells = (packed.total_particles, packed.total_nodes,
                                                                    packed.total_cells)
        self.dt, self.dx, self.spacing, self.volume = u["dt"], u["dx"], u["spacing"], u["volume"]
        self.substeps, self.gravity_tuple = u["substeps"], u["gravity"]
        self.gravity = wp.vec3(*u["gravity"])
        self.pic, self.j_domain = 0.0, u["j_domain"]
        self.radius, self.contact_stiffness = u["radius"], u["contact_stiffness"]
        self.G, self.Gt, self.boxes, self.use_plane = u["G"], u["Gt"], u["boxes"], u["use_plane"]
        self.markers_per_env = u["markers"]
        self.model = cr.get_model(packed.constitutive_model)
        h = packed.host
        names = (PARTICLE_ARRAYS + NODE_ARRAYS + GROUP_ARRAYS + TRACTION_ARRAYS
                 + ("cell_node", "cell_offsets", "cell_particles", "marker_offsets", "marker_particles", "params",
                    "plane", "box_c", "box_h", "env_of_particle", "env_of_node", "env_of_group", "particle_offset",
                    "node_offset", "cell_env_offset"))
        with wp.ScopedDevice(self.device):
            for name in names:
                setattr(self, name, wp.array(h[name], dtype=self._DTYPES[name]))
            self.work = wp.zeros(self.num_envs, dtype=wp.float64)
            self.violations = wp.zeros(self.num_envs, dtype=int)
            self.index = wp.zeros(1, dtype=int)
        for module in (v5.__name__, "robotic_cutting_sim.tlmpm_v6", rc.__name__, "robotic_cutting_sim.tlmpm_rubber_v1",
                       cr.__name__, __name__):
            wp.load_module(module=module, device=self.device)

    def step(self):
        """One substep of every env (``tlmpm_rubber_v1.TLMPMRubber.step`` order)."""
        n, nn, dt = self.total_particles, self.total_nodes, self.dt
        with wp.ScopedDevice(self.device):
            wp.copy(self.x_start, self.x)
            self.model.launch_stress(self)
            wp.launch(_loads_b, n, [self.x, self.v, self.mass, self.impulse, self.particle_traction,
                                    self.traction_force, self.traction_mass, dt, self.radius, self.contact_stiffness,
                                    self.params, self.use_plane, self.plane, self.boxes, self.box_c, self.box_h,
                                    self.env_of_particle, self.work])
            self.gm.zero_(); self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._p2g, (n, 8), [self.v, self.mass, self.slot, self.stress, self.impulse, self.nodes,
                                        self.weights, self.gradients, self.gm, self.gp, self.gj, 1, self.volume, dt,
                                        self.gravity])
            wp.launch(v5._grid_update, (1, nn), [self.gm, self.gp, self.gj, self.old, self.new_v, self.slot_code, 1])
            self._launch_bc(1)
            wp.launch(v5._g2p, n, [self.x, self.v, self.slot, self.nodes, self.weights, self.old, self.new_v, 1, dt,
                                   self.pic])
            self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._musl, (n, 8), [self.v, self.mass, self.slot, self.nodes, self.weights, self.gp, 1])
            wp.launch(v5._grid_update, (1, nn), [self.gm, self.gp, self.gj, self.old, self.new_v, self.slot_code, 1])
            self._launch_bc(0)
            wp.launch(v5._deformation, n, [self.F, self.slot, self.nodes, self.gradients, self.new_v, 1, dt])
            wp.launch(_grid_positions, n, [self.x, self.x_start, self.slot, self.nodes, self.weights, self.new_v, 1,
                                           dt])
            wp.launch(v5._advance, 1, [self.index])

    def _launch_bc(self, record):
        if self.G == 0:
            return
        wp.launch(_bc_b, self.total_nodes, [self.node_group, self.node_mask, self.node_X, self.gm, self.new_v,
                                            self.index, self.substeps, wp.float64(self.dt), self.group_kind,
                                            self.group_active, self.group_q0, self.group_q1, self.group_b0,
                                            self.group_b1, self.group_A0, self.group_A1, self.group_origin,
                                            self.reaction, self.work, self.env_of_group, record], device=self.device)

    def device_arrays(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, wp.array)}

    def memory_bytes(self):
        return int(sum(v.capacity for v in self.device_arrays().values()))

    def snapshot(self, e):
        """Env e's arrays in the single solver's names, shapes and LOCAL id spaces (host copies)."""
        pk = self.packed
        p0, p1 = int(pk.particle_offset[e]), int(pk.particle_offset[e+1])
        n0, n1 = int(pk.node_offset[e]), int(pk.node_offset[e+1])
        c0, c1 = int(pk.cell_env_offset[e]), int(pk.cell_env_offset[e+1])
        G, Gt = self.G, self.Gt
        out = {}
        for name in PARTICLE_ARRAYS:
            a = getattr(self, name).numpy()[p0:p1].copy()
            if name == "nodes":
                a = a-np.int32(n0)
            elif name == "particle_traction":
                a = np.where(a >= 0, a-np.int32(e*Gt), a).astype(np.int32)
            out[name] = a
        for name in NODE_ARRAYS:
            a = getattr(self, name).numpy()[n0:n1].copy()
            if name == "node_group":
                a = np.where(a >= 0, a-np.int32(e*G), a).astype(np.int32)
            out[name] = a
        for name in GROUP_ARRAYS:
            out[name] = getattr(self, name).numpy()[e*G:(e+1)*G].copy() if G else getattr(self, name).numpy()[:0]
        for name in TRACTION_ARRAYS:
            out[name] = getattr(self, name).numpy()[e*Gt:(e+1)*Gt].copy() if Gt else getattr(self, name).numpy()[:0]
        out["cell_node"] = self.cell_node.numpy()[c0:c1]-np.int32(n0)
        out["cell_particles"] = self.cell_particles.numpy()[self.cell_offsets.numpy()[c0]:self.cell_offsets.numpy()[c1]]-np.int32(p0)
        out["cell_offsets"] = self.cell_offsets.numpy()[c0:c1+1]-np.int32(p0)
        out["work"] = self.work.numpy()[e:e+1].copy()
        out["violations"] = self.violations.numpy()[e:e+1].copy()
        out["index"] = self.index.numpy().copy()
        out["params"] = self.params.numpy()[e].copy()
        return out

    def solver_receipt(self):
        return {"solver": SOLVER_NAME, "base_solver": BASE_SOLVER, "status": STATUS,
                "execution": "batched_flat_soa_env_kernels_v1", "constitutive_model": self.model.receipt(),
                "dt_s": self.dt, "dx_m": self.dx, "substeps_per_control_step": self.substeps, "pic": 0.0,
                "source_sha256": source_sha256(), "device": self.device, "warp_version": wp.config.version}
