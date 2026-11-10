"""Batched execution ``tlmpm_rubber_v1_batch`` of the single-env reference ``tlmpm_rubber_v1`` (ADR 0023 D1, D9; R1.7).

One launch per operation covers all N envs, in the single solver's launch order. UNCHANGED kernels: the frozen
``tlmpm_v5._p2g/_grid_update/_g2p/_musl/_deformation/_advance`` and ``tlmpm_v6._grid_positions`` (global node ids keep
the envs' node ranges disjoint), and ``rubber_constitutive.cell_jbar`` (global cell CSR). ENV kernels (``*_b``): the
single kernels' bodies with per-env loads (``params[e]``, plane / boxes / traction / group rows of env e) and per-env
accumulators (violations, constraint work), calling the same ``@wp.func`` helpers. Every float atomic target belongs to
one env whose particles / nodes are contiguous and in single-env order, so on the Warp CPU device (sequential kernels)
each target receives the same contributions in the same order as in the single solver: CPU parity is bitwise (L0).
On a GPU the frozen P2G/MUSL atomics are the only nondeterminism (J-bar is a gather; diagnostics are fixed-order).

P2-3 prerequisites (ADR 0023 §20), opt-in through the single references' options (a batch runs the options its lives
were built with; they must agree; the defaults add no array and no launch):

- ``dirichlet="consistent_mass_projection_v1"`` (F8): the single solver's ``_proj_rhs`` / ``_proj_solve`` /
  ``_proj_particles`` launched UNCHANGED over the concatenated rows (global node / group / block ids, the per-env
  float64 block inverses copied bit-for-bit from the single references), ``_proj_reaction_b`` with the env's work;
- ``traction_work="grid_displacement_v1"`` (F9): ``_traction_work_grid_b`` (per-env work); the v1 accumulation goes to
  a discarded per-env ``work_scratch`` as in the single solver;
- ``state_precision="f64_shadow_v1"`` (F10): the single solver's ``_deformation_f64`` / ``_positions_f64`` UNCHANGED
  over flat float64 shadows ``F64`` / ``x64``;
- ``constraint_work="kick_average_v1"`` (ADR 0023 §21.1, bookkeeping of F8): ``_proj_reaction_b``'s end-velocity work
  goes to the per-env ``work_end_velocity`` (reported) and ``rubber_constraint_work.proj_work_kick_average`` (the
  single solver's kernel, UNCHANGED, target ``env_of_group[g]``) accumulates Lambda . (g_n + v_presc) / 2 into
  ``work``. No state array changes.

Viscoelastic model (ADR 0023 §21.2; ``neo_hookean_jbar_prony_v1`` from the lives, registry ``prony_model(M)``): the
stress pass is the J-bar pass + ``tlmpm_rubber_visco_v1.stress_prony`` and the guard ``cfl_guard_prony`` (the single
solver's kernels, per-env ``params``/``params64`` rows and accumulators through ``env_of_particle``); with
``visco_energy = branch_work_v1`` the single solver's ``visco_branch_work`` runs after the position update. Arrays:
``params64[N, 1 + 4M]``, ``visco_q[P*M]``, ``visco_s_prev[P]`` (float64) [, ``visco_pb[P]``, ``visco_branch_work[N]``,
``visco_dissipation[N]``, ``visco_d_negative[N]``]. With ``visco_energy = branch_work_kick_average_v2`` (ADR 0023 §21.7)
the single solver's ``visco_branch_work_kick_average`` runs instead (kick-paired W_b, the v1 left-point sum in
``visco_branch_work_left[N]``, dF_{n-1} in ``visco_df_prev[P]``); every other array and launch is the same.

BC launch (``bc_launch``, ADR 0023 §20.2): ``all_nodes_v1`` is the P2-1/P2-2 launch (``_bc_b`` over every node,
group maps evaluated per node); ``constrained_list_v1`` launches the same body over the packed list of constrained
nodes (increasing global id); ``motion_table_v1`` (default since §20) also evaluates the group maps of every substep
boundary once per control step (``_motion_table_b``, ``prepare_motion_table``) and the BC kernel reads them. On the CPU
device all three visit the constrained nodes in the same order with the same arithmetic (the all-node launch returns
at once on unconstrained nodes; the table holds exactly the M(s), b(s) that ``group_target`` computes before
M X + b), so the results are bitwise equal (tested). The GPU cost of the BC pass was per-thread float64 map evaluation,
not the node loop (``artifacts/rl_ready/p2_3_prep/bc_bench_001/002``).
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from .. import tlmpm_v5 as v5
from .. import rubber_constitutive as rc
from .. import rubber_constraint_work as rcw
from .. import rubber_visco_constitutive as rvc
from ..tlmpm_rubber_v1 import (GROUP_RIGID, _deformation_f64, _positions_f64, _proj_particles, _proj_rhs, _proj_solve, apply_mask,
                               bc_reaction, bc_work, contact_load, group_target)
from ..tlmpm_v6 import _grid_positions
from . import constitutive_rubber as cr
from .constitutive_rubber import P_FRICTION_TABLE
from .errors import UnsupportedBatchConfiguration
from .rubber_layout import (GROUP_ARRAYS, NODE_ARRAYS, PARTICLE_ARRAYS, PROJECTION_BLOCK_ARRAYS, PROJECTION_ROW_ARRAYS,
                            KICK_AVERAGE_ENERGY, SHADOW_ARRAYS, TRACTION_ARRAYS, VISCO_STATE_ARRAYS,
                            visco_energy_arrays)

wp.set_module_options({"enable_backward": False})

SOLVER_NAME = "tlmpm_rubber_v1_batch"
BASE_SOLVER = "tlmpm_rubber_v1"
STATUS = "EXPERIMENTAL_NOT_ACCEPTED"
BC_LAUNCHES = ("constrained_list_v1", "all_nodes_v1", "motion_table_v1")
DEFAULT_BC_LAUNCH = "motion_table_v1"
_HERE = Path(__file__).resolve()


def source_sha256():
    here = _HERE.parent
    out = {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
           for name in ("rubber_batch.py", "rubber_layout.py", "constitutive_rubber.py", "rubber_material_batch.py")}
    pkg = here.parent
    out.update({name: hashlib.sha256((pkg/name).read_bytes()).hexdigest()
                for name in ("tlmpm_rubber_v1.py", "rubber_constitutive.py", "tlmpm_v5.py", "tlmpm_v6.py",
                             "rubber_constraint_work.py", "tlmpm_rubber_visco_v1.py", "rubber_visco_constitutive.py")})
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


@wp.func
def _bc_node(i: int, node_group: wp.array(dtype=int), node_mask: wp.array(dtype=int), node_X: wp.array(dtype=wp.vec3d),
             gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), index: wp.array(dtype=int), substeps: int,
             dt: wp.float64, kind: wp.array(dtype=int), active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd),
             q1: wp.array(dtype=wp.quatd), b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d),
             A0: wp.array(dtype=wp.mat33d), A1: wp.array(dtype=wp.mat33d), origin: wp.array(dtype=wp.vec3d),
             reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array(dtype=wp.float64),
             env_of_group: wp.array(dtype=int), record: int):
    """Body of ``_bc_b`` for node i (same statements, same order)."""
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


@wp.kernel
def _bc_list_b(bc_nodes: wp.array(dtype=int), node_group: wp.array(dtype=int), node_mask: wp.array(dtype=int),
               node_X: wp.array(dtype=wp.vec3d), gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3),
               index: wp.array(dtype=int), substeps: int, dt: wp.float64, kind: wp.array(dtype=int),
               active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd), q1: wp.array(dtype=wp.quatd),
               b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d), A0: wp.array(dtype=wp.mat33d),
               A1: wp.array(dtype=wp.mat33d), origin: wp.array(dtype=wp.vec3d),
               reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array(dtype=wp.float64),
               env_of_group: wp.array(dtype=int), record: int):
    """``_bc_b`` over the packed constrained-node list (``bc_launch="constrained_list_v1"``)."""
    _bc_node(bc_nodes[wp.tid()], node_group, node_mask, node_X, gm, new_v, index, substeps, dt, kind, active, q0, q1,
             b0, b1, A0, A1, origin, reaction, work, env_of_group, record)


@wp.kernel
def _motion_table_b(substeps: int, kind: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd),
                    q1: wp.array(dtype=wp.quatd), b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d),
                    A0: wp.array(dtype=wp.mat33d), A1: wp.array(dtype=wp.mat33d),
                    table_M: wp.array2d(dtype=wp.mat33d), table_b: wp.array2d(dtype=wp.vec3d)):
    """Group maps at s = k / substeps for every substep boundary k of the control step (``group_target`` without the
    final M X + b, same statements): one thread per (k, group), once per control step."""
    k, g = wp.tid()
    s = wp.float64(k)/wp.float64(substeps)
    b = b0[g]+s*(b1[g]-b0[g])
    M = A0[g]+s*(A1[g]-A0[g])
    if kind[g] == GROUP_RIGID:
        M = wp.quat_to_matrix(wp.quat_slerp(q0[g], q1[g], s))
    table_M[k, g] = M
    table_b[k, g] = b


@wp.kernel
def _bc_table_b(bc_nodes: wp.array(dtype=int), node_group: wp.array(dtype=int), node_mask: wp.array(dtype=int),
                node_X: wp.array(dtype=wp.vec3d), gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3),
                index: wp.array(dtype=int), substeps: int, dt: wp.float64, kind: wp.array(dtype=int),
                active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd), q1: wp.array(dtype=wp.quatd),
                b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d), A0: wp.array(dtype=wp.mat33d),
                A1: wp.array(dtype=wp.mat33d), origin: wp.array(dtype=wp.vec3d),
                reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array(dtype=wp.float64),
                env_of_group: wp.array(dtype=int), record: int, table_M: wp.array2d(dtype=wp.mat33d),
                table_b: wp.array2d(dtype=wp.vec3d)):
    """``_bc_b`` over the constrained-node list with the group maps read from the control step's motion table
    (``bc_launch="motion_table_v1"``); beyond the table (index + 1 > substeps: direct ``solver.step()`` calls past
    the control step) the maps are evaluated directly, as in ``_bc_b``."""
    i = bc_nodes[wp.tid()]
    g = node_group[i]
    if g >= 0:
        if active[g] != 0:
            k = index[0]
            X = node_X[i]
            x0 = wp.vec3d()
            x1 = wp.vec3d()
            o1 = wp.vec3d()
            if k+1 <= substeps:
                x0 = table_M[k, g]*X+table_b[k, g]
                x1 = table_M[k+1, g]*X+table_b[k+1, g]
                if record != 0:
                    o1 = table_M[k+1, g]*origin[g]+table_b[k+1, g]
            else:
                s0 = wp.float64(k)/wp.float64(substeps)
                s1 = wp.float64(k+1)/wp.float64(substeps)
                x0 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s0, X)
                x1 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s1, X)
                if record != 0:
                    o1 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s1, origin[g])
            v = new_v[i]
            vn = apply_mask(v, (x1-x0)/dt, node_mask[i])
            new_v[i] = vn
            if record != 0:
                wp.atomic_add(reaction, g, bc_reaction(gm[i], v, vn, x1, o1))
                wp.atomic_add(work, env_of_group[g], bc_work(gm[i], v, vn))


@wp.kernel
def _proj_reaction_b(row_node: wp.array(dtype=int), row_comp: wp.array(dtype=int), row_group: wp.array(dtype=int),
                     node_X: wp.array(dtype=wp.vec3d), gm: wp.array(dtype=float), mu: wp.array(dtype=wp.float64),
                     vpresc: wp.array(dtype=wp.float64), index: wp.array(dtype=int), substeps: int,
                     kind: wp.array(dtype=int), active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd),
                     q1: wp.array(dtype=wp.quatd), b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d),
                     A0: wp.array(dtype=wp.mat33d), A1: wp.array(dtype=wp.mat33d), origin: wp.array(dtype=wp.vec3d),
                     reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array(dtype=wp.float64),
                     env_of_group: wp.array(dtype=int)):
    """``tlmpm_rubber_v1._proj_reaction`` with global group ids and the constraint work of the group's env."""
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
        wp.atomic_add(work, env_of_group[g], j[row_comp[r]]*vpresc[r])


@wp.kernel
def _traction_work_grid_b(x: wp.array(dtype=wp.vec3), x_start: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                          particle_traction: wp.array(dtype=int), traction_force: wp.array(dtype=wp.vec3),
                          traction_mass: wp.array(dtype=float), env_of_particle: wp.array(dtype=int),
                          work: wp.array(dtype=wp.float64)):
    """``tlmpm_rubber_v1._traction_work_grid`` with the work of the particle's env."""
    p = wp.tid()
    t = particle_traction[p]
    if t >= 0:
        f = wp.vec3d(traction_force[t])*(wp.float64(mass[p])/wp.float64(traction_mass[t]))
        wp.atomic_add(work, env_of_particle[p], wp.dot(f, wp.vec3d(x[p])-wp.vec3d(x_start[p])))


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
        "reaction": wp.spatial_vectord, "plane": wp.vec4, "bc_nodes": int,
        # option arrays (ADR 0023 §20)
        "F64": wp.mat33d, "x64": wp.vec3d, "reference64": wp.vec3d, "work_end_velocity": wp.float64,
        **dict.fromkeys(("proj_row_node", "proj_row_comp", "proj_row_group", "proj_row_block", "proj_row_local",
                         "proj_block_offset", "proj_block_size", "proj_inv_offset", "env_of_row"), int),
        **dict.fromkeys(("proj_inv", "proj_rhs", "proj_vpresc", "proj_mu", "proj_mu_node"), wp.float64),
        # neo_hookean_jbar_prony_v1 (ADR 0023 §21.2)
        "visco_q": rvc.vec6d, "visco_s_prev": rvc.vec6d, "params64": wp.float64, "visco_pb": wp.mat33d,
        "visco_branch_work": wp.float64, "visco_dissipation": wp.float64, "visco_d_negative": int,
        "visco_df_prev": wp.mat33d, "visco_branch_work_left": wp.float64,      # branch_work_kick_average_v2
    }
    OPTION_ARRAYS = {"f64_shadow_v1": SHADOW_ARRAYS+("reference64",),
                     "consistent_mass_projection_v1": PROJECTION_ROW_ARRAYS+PROJECTION_BLOCK_ARRAYS
                     + ("proj_inv", "proj_mu_node", "env_of_row")}

    def __init__(self, packed, device, *, bc_launch=DEFAULT_BC_LAUNCH):
        if bc_launch not in BC_LAUNCHES:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER", f"bc_launch={bc_launch!r} not in {BC_LAUNCHES}")
        self.packed = packed
        self.device = str(wp.get_device(device))
        self.bc_launch = bc_launch
        u = packed.uniform
        self.dirichlet, self.traction_work, self.state_precision = (u["dirichlet"], u["traction_work"],
                                                                    u["state_precision"])
        self.constraint_work = u["constraint_work"]
        self.projection = self.dirichlet == "consistent_mass_projection_v1" and u["projection_rows_nonzero"]
        self.kick_average = self.constraint_work != rcw.DEFAULT_CONSTRAINT_WORK
        self.num_envs = packed.num_envs
        self.total_particles, self.total_nodes, self.total_cells = (packed.total_particles, packed.total_nodes,
                                                                    packed.total_cells)
        self.dt, self.dx, self.spacing, self.volume = u["dt"], u["dx"], u["spacing"], u["volume"]
        self.substeps, self.gravity_tuple = u["substeps"], u["gravity"]
        self.gravity = wp.vec3(*u["gravity"])
        self.pic, self.j_domain = 0.0, u["j_domain"]
        self.cfl_guard_limit = u["stable_step_guard"]
        self.radius, self.contact_stiffness = u["radius"], u["contact_stiffness"]
        self.G, self.Gt, self.boxes, self.use_plane = u["G"], u["Gt"], u["boxes"], u["use_plane"]
        self.markers_per_env = u["markers"]
        self.model = cr.get_model(packed.constitutive_model)
        self.M = packed.visco_terms
        if self.M:
            self.model = cr.prony_model(self.M)
        self.visco_energy = u.get("visco_energy", "off_v1")
        self.visco_energy_on = bool(self.M) and self.visco_energy != "off_v1"
        h = packed.host
        names = (PARTICLE_ARRAYS + NODE_ARRAYS + GROUP_ARRAYS + TRACTION_ARRAYS
                 + ("cell_node", "cell_offsets", "cell_particles", "marker_offsets", "marker_particles", "params",
                    "plane", "box_c", "box_h", "env_of_particle", "env_of_node", "env_of_group", "particle_offset",
                    "node_offset", "cell_env_offset", "bc_nodes"))
        if self.state_precision == "f64_shadow_v1":
            names += self.OPTION_ARRAYS["f64_shadow_v1"]
        if self.projection:
            names += self.OPTION_ARRAYS["consistent_mass_projection_v1"]
        if self.M:
            names += VISCO_STATE_ARRAYS+("params64",)
        if self.visco_energy_on:
            names += visco_energy_arrays(self.visco_energy)
        self.bc_node_count = int(np.count_nonzero(h["node_group"] >= 0)) if self.G else 0
        self.proj_rows = int(h["proj_row_node"].shape[0]) if self.projection else 0
        with wp.ScopedDevice(self.device):
            for name in names:
                setattr(self, name, wp.array(h[name], dtype=self._DTYPES[name]))
            if self.traction_work == "grid_displacement_v1":
                self.work_scratch = wp.zeros(self.num_envs, dtype=wp.float64)   # the v1 accumulation (discarded)
            if self.kick_average:
                self.work_end_velocity = wp.zeros(self.num_envs, dtype=wp.float64)   # v1 constraint work (reported)
            self.work = wp.zeros(self.num_envs, dtype=wp.float64)
            if self.bc_launch == "motion_table_v1" and self.G:
                # group maps at every substep boundary of the control step (recomputed by prepare_motion_table)
                self.motion_M = wp.zeros((self.substeps+1, self.num_envs*self.G), dtype=wp.mat33d)
                self.motion_b = wp.zeros((self.substeps+1, self.num_envs*self.G), dtype=wp.vec3d)
            self.violations = wp.zeros(self.num_envs, dtype=int)
            self.cfl_violations = wp.zeros(self.num_envs, dtype=int)
            self.index = wp.zeros(1, dtype=int)
        modules = ((v5.__name__, "robotic_cutting_sim.tlmpm_v6", rc.__name__, "robotic_cutting_sim.tlmpm_rubber_v1",
                    cr.__name__, __name__) + ((rcw.__name__,) if self.kick_average else ())
                   + (("robotic_cutting_sim.rubber_visco_constitutive", "robotic_cutting_sim.tlmpm_rubber_visco_v1")
                      if self.M else ()))
        for module in modules:
            wp.load_module(module=module, device=self.device)

    def step(self):
        """One substep of every env (``tlmpm_rubber_v1.TLMPMRubber.step`` order)."""
        n, nn, dt = self.total_particles, self.total_nodes, self.dt
        projection = self.dirichlet == "consistent_mass_projection_v1"
        work_v1 = self.work if self.traction_work == "particle_velocity_v1" else self.work_scratch
        with wp.ScopedDevice(self.device):
            wp.copy(self.x_start, self.x)
            self.model.launch_stress(self)
            if self.cfl_guard_limit is not None:
                self.model.launch_guard(self)
            wp.launch(_loads_b, n, [self.x, self.v, self.mass, self.impulse, self.particle_traction,
                                    self.traction_force, self.traction_mass, dt, self.radius, self.contact_stiffness,
                                    self.params, self.use_plane, self.plane, self.boxes, self.box_c, self.box_h,
                                    self.env_of_particle, work_v1])
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
                wp.launch(_grid_positions, n, [self.x, self.x_start, self.slot, self.nodes, self.weights, self.new_v, 1,
                                               dt])
            if self.traction_work == "grid_displacement_v1" and self.Gt:
                wp.launch(_traction_work_grid_b, n, [self.x, self.x_start, self.mass, self.particle_traction,
                                                     self.traction_force, self.traction_mass, self.env_of_particle,
                                                     self.work])
            if self.visco_energy_on:
                if self.visco_energy == KICK_AVERAGE_ENERGY:
                    from ..tlmpm_rubber_visco_v1 import visco_branch_work_kick_average
                    wp.launch(visco_branch_work_kick_average, n, [self.visco_pb, self.nodes, self.gradients,
                                                                  self.new_v, self.env_of_particle,
                                                                  wp.float64(self.volume), wp.float64(dt),
                                                                  self.visco_df_prev, self.visco_branch_work,
                                                                  self.visco_branch_work_left])
                else:
                    from ..tlmpm_rubber_visco_v1 import visco_branch_work
                    wp.launch(visco_branch_work, n, [self.visco_pb, self.nodes, self.gradients, self.new_v,
                                                     self.env_of_particle, wp.float64(self.volume), wp.float64(dt),
                                                     self.visco_branch_work])
            wp.launch(v5._advance, 1, [self.index])

    def _launch_projection(self):
        """F8 over all envs' rows (``tlmpm_rubber_v1.TLMPMRubber._launch_projection`` order)."""
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
        wp.launch(_proj_particles, self.total_particles, [self.v, self.nodes, self.weights, self.proj_mu_node])
        wp.launch(_proj_reaction_b, r, [self.proj_row_node, self.proj_row_comp, self.proj_row_group, self.node_X,
                                        self.gm, self.proj_mu, self.proj_vpresc, self.index, self.substeps]
                  + groups + [self.group_origin, self.reaction,
                              self.work_end_velocity if self.kick_average else self.work, self.env_of_group])
        if self.kick_average:
            wp.launch(rcw.proj_work_kick_average, r, [self.proj_row_node, self.proj_row_comp, self.proj_row_group,
                                                      self.group_active, self.gm, self.proj_mu, self.proj_vpresc,
                                                      self.old, self.env_of_group, self.work])

    def prepare_motion_table(self):
        """``motion_table_v1``: evaluate the group maps of the current control step (call after the group motions
        of the step are written; ``BatchedRubberMaterial.set_group_motions`` does)."""
        if self.bc_launch == "motion_table_v1" and self.G:
            wp.launch(_motion_table_b, (self.substeps+1, self.num_envs*self.G),
                      [self.substeps, self.group_kind, self.group_q0, self.group_q1, self.group_b0, self.group_b1,
                       self.group_A0, self.group_A1, self.motion_M, self.motion_b], device=self.device)

    def _launch_bc(self, record):
        if self.G == 0:
            return
        if self.bc_launch == "motion_table_v1":
            if self.bc_node_count:
                wp.launch(_bc_table_b, self.bc_node_count,
                          [self.bc_nodes, self.node_group, self.node_mask, self.node_X, self.gm, self.new_v,
                           self.index, self.substeps, wp.float64(self.dt), self.group_kind, self.group_active,
                           self.group_q0, self.group_q1, self.group_b0, self.group_b1, self.group_A0, self.group_A1,
                           self.group_origin, self.reaction, self.work, self.env_of_group, record, self.motion_M,
                           self.motion_b], device=self.device)
            return
        if self.bc_launch == "constrained_list_v1":
            if self.bc_node_count:
                wp.launch(_bc_list_b, self.bc_node_count,
                          [self.bc_nodes, self.node_group, self.node_mask, self.node_X, self.gm, self.new_v,
                           self.index, self.substeps, wp.float64(self.dt), self.group_kind, self.group_active,
                           self.group_q0, self.group_q1, self.group_b0, self.group_b1, self.group_A0, self.group_A1,
                           self.group_origin, self.reaction, self.work, self.env_of_group, record],
                          device=self.device)
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
        out["cfl_violations"] = self.cfl_violations.numpy()[e:e+1].copy()
        out["index"] = self.index.numpy().copy()
        out["params"] = self.params.numpy()[e].copy()
        if self.traction_work == "grid_displacement_v1":
            out["work_scratch"] = self.work_scratch.numpy()[e:e+1].copy()
        if self.kick_average:
            out["work_end_velocity"] = self.work_end_velocity.numpy()[e:e+1].copy()
        if self.state_precision == "f64_shadow_v1":
            out["F64"] = self.F64.numpy()[p0:p1].copy()
            out["x64"] = self.x64.numpy()[p0:p1].copy()
        if self.projection:
            ro, bo, io = (pk.env[k] for k in ("proj_row_offset", "proj_block_env_offset", "proj_inv_env_offset"))
            r0, r1, b0, b1 = int(ro[e]), int(ro[e+1]), int(bo[e]), int(bo[e+1])
            shift = {"proj_row_node": n0, "proj_row_group": e*G, "proj_row_block": b0}
            for name in PROJECTION_ROW_ARRAYS:
                out[name] = getattr(self, name).numpy()[r0:r1]-shift.get(name, 0)
            for name, base in (("proj_block_offset", r0), ("proj_block_size", 0), ("proj_inv_offset", int(io[e]))):
                out[name] = getattr(self, name).numpy()[b0:b1]-np.int32(base)
            out["proj_inv"] = self.proj_inv.numpy()[int(io[e]):int(io[e+1])].copy()
            out["proj_mu_node"] = self.proj_mu_node.numpy()[n0:n1].copy()
        if self.M:
            M = self.M
            out["visco_q"] = self.visco_q.numpy()[p0*M:p1*M].copy()
            out["visco_s_prev"] = self.visco_s_prev.numpy()[p0:p1].copy()
            out["params64"] = self.params64.numpy()[e:e+1].copy()
        if self.visco_energy_on:
            out["visco_pb"] = self.visco_pb.numpy()[p0:p1].copy()
            for name in ("visco_branch_work", "visco_dissipation", "visco_d_negative"):
                out[name] = getattr(self, name).numpy()[e:e+1].copy()
            if self.visco_energy == KICK_AVERAGE_ENERGY:
                out["visco_df_prev"] = self.visco_df_prev.numpy()[p0:p1].copy()
                out["visco_branch_work_left"] = self.visco_branch_work_left.numpy()[e:e+1].copy()
        return out

    def options_receipt(self):
        return {"dirichlet": self.dirichlet, "traction_work": self.traction_work,
                "state_precision": self.state_precision, "constraint_work": self.constraint_work,
                "projection_rows": self.proj_rows, "visco_terms": self.M, "visco_energy": self.visco_energy,
                "bc_launch": self.bc_launch, "bc_nodes": self.bc_node_count, "adr": "ADR 0023 §20"}

    def solver_receipt(self):
        return {"solver": SOLVER_NAME, "base_solver": BASE_SOLVER, "status": STATUS,
                "execution": "batched_flat_soa_env_kernels_v1", "constitutive_model": self.model.receipt(),
                "dt_s": self.dt, "dx_m": self.dx, "substeps_per_control_step": self.substeps, "pic": 0.0,
                "stable_step_guard": (None if self.cfl_guard_limit is None else
                                      {"id": rc.STABLE_STEP_GUARD_ID, "limit": self.cfl_guard_limit}),
                "options": self.options_receipt(),
                "source_sha256": source_sha256(), "device": self.device, "warp_version": wp.config.version}
