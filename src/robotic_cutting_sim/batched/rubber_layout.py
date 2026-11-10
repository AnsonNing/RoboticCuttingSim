"""Flat per-env layout of N rubber envs (ADR 0023 §12 D9, ADR 0021 §3 conventions; task R1.7).

Every env's initial state is packed from its own fresh single-env reference ``LiveRubberMaterial`` built on the Warp
CPU device (no stepping), so geometry, stencil, cell CSR, groups, masses and markers are identical to the single
solver by construction; this module only concatenates them, in env order with exact per-env counts (no padding):

- per-particle arrays, ``nodes`` / ``cell_node`` as global node ids, ``cell_particles`` / ``marker_particles`` as global
  particle ids, ``cell_offsets`` / ``marker_offsets`` as global CSR offsets;
- per-node arrays, ``node_group`` as global group ids ``e*G + g``; group and traction tables as ``[N*G]`` /
  ``[N*Gt]`` rows; per-env ``params[N, k]`` (``constitutive_rubber.RUBBER_PARAM_NAMES``), table planes and boxes.

Uniform per batch (kernel scalars; a difference raises ``UNSUPPORTED_NONUNIFORM_BATCH``): dt, dx, particle spacing,
substeps and control dt, gravity, pic (= 0), j domain, contact radius / stiffness, G and the group kinds, Gt, box count,
table use. Heterogeneous: geometry (particle counts), material (mu0, kappa, density), frictions, groups' regions.
"""
from dataclasses import dataclass, field

import numpy as np

from . import constitutive_rubber as cr
from .errors import UnsupportedBatchConfiguration

SOLVER = "tlmpm_rubber_v1"
LAYOUT_VERSION = "rubber_batch_layout_v1"
INT32_LIMIT = 2**31

PARTICLE_ARRAYS = ("x", "v", "reference", "mass", "F", "stress", "impulse", "x_start", "nodes", "weights", "gradients",
                   "slot", "particle_traction")
NODE_ARRAYS = ("gm", "gp", "gj", "old", "new_v", "jbar", "slot_code", "node_group", "node_mask", "node_X")
GROUP_ARRAYS = ("group_kind", "group_active", "group_origin", "group_q0", "group_q1", "group_b0", "group_b1",
                "group_A0", "group_A1", "reaction")
TRACTION_ARRAYS = ("traction_force", "traction_mass")
# P2-3 prerequisites (ADR 0023 §20): the P2-2 single-env opt-in options with a batched execution (defaults first; the
# defaults add no array and keep the v1 launch sequence). A batch runs the options of its single references, which
# must agree (uniform signature).
BATCHED_OPTIONS = {"dirichlet": ("lumped_v1", "consistent_mass_projection_v1"),
                   "traction_work": ("particle_velocity_v1", "grid_displacement_v1"),
                   "state_precision": ("float32_v1", "f64_shadow_v1"),
                   # ADR 0023 §21.1 (bookkeeping of the F8 projection; rubber_constraint_work.TLMPMRubberCW)
                   "constraint_work": ("end_velocity_v1", "kick_average_v1")}
SHADOW_ARRAYS = ("F64", "x64")                                      # per particle (f64_shadow_v1)
PROJECTION_ROW_ARRAYS = ("proj_row_node", "proj_row_comp", "proj_row_group", "proj_row_block", "proj_row_local",
                         "proj_rhs", "proj_vpresc", "proj_mu")     # per constrained (node, component) row
PROJECTION_BLOCK_ARRAYS = ("proj_block_offset", "proj_block_size", "proj_inv_offset")   # per block


@dataclass
class PackedRubberBatch:
    num_envs: int
    particle_counts: np.ndarray
    node_counts: np.ndarray
    cell_counts: np.ndarray
    marker_particle_counts: np.ndarray
    particle_offset: np.ndarray
    node_offset: np.ndarray
    cell_env_offset: np.ndarray
    uniform: dict
    host: dict
    env: dict
    receipts: list
    scene_keys: list = field(default_factory=list)
    constitutive_model: str = cr.DEFAULT_MODEL_ID

    @property
    def total_particles(self):
        return int(self.particle_offset[-1])

    @property
    def total_nodes(self):
        return int(self.node_offset[-1])

    @property
    def total_cells(self):
        return int(self.cell_env_offset[-1])

    def layout_receipt(self):
        return {"layout_version": LAYOUT_VERSION, "num_envs": self.num_envs,
                "particle_counts": self.particle_counts.tolist(), "node_counts": self.node_counts.tolist(),
                "cell_counts": self.cell_counts.tolist(), "total_particles": self.total_particles,
                "total_nodes": self.total_nodes, "total_cells": self.total_cells,
                "groups_per_env": self.uniform["G"], "traction_groups_per_env": self.uniform["Gt"],
                "int32_guards": {"particles_x8": self.total_particles*8, "nodes": self.total_nodes,
                                 "limit": INT32_LIMIT},
                "constitutive_model": self.constitutive_model}


def _validate(live):
    from ..live_rubber_material import LiveRubberMaterial
    from ..rubber_constraint_work import TLMPMRubberCW
    from ..tlmpm_rubber_v1 import TLMPMRubber
    if not isinstance(live, LiveRubberMaterial):
        raise TypeError("pack_rubber takes LiveRubberMaterial instances (single-env references)")
    s = live.solver
    if type(s) not in (TLMPMRubber, TLMPMRubberCW):
        raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER", f"{type(s).__name__}; only {SOLVER}")
    if s.device != "cpu":
        raise ValueError("pack_rubber reads single solvers built on the Warp CPU device (ADR 0021 §3.4)")
    options = {"dirichlet": s.dirichlet, "traction_work": s.traction_work, "state_precision": s.state_precision,
               "constraint_work": constraint_work_of(s)}
    for key, value in options.items():
        if value not in BATCHED_OPTIONS[key]:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER", f"{key}={value!r} has no batched execution")
    if (live.step_count != 0 or int(s.index.numpy()[0]) != 0 or not np.array_equal(s.x.numpy(), s.reference.numpy())
            or np.abs(s.v.numpy()).max() != 0.0 or s.damping_rate != 0.0):
        raise UnsupportedBatchConfiguration("UNSUPPORTED_SINGLE_ENV_STATE", "pack takes fresh, unstepped reset states")


def constraint_work_of(solver):
    """``constraint_work`` of a single reference (the frozen ``TLMPMRubber`` has the v1 accounting)."""
    return getattr(solver, "constraint_work", "end_velocity_v1")


def uniform_signature(live):
    s = live.solver
    return {"dt": float(s.dt), "dx": float(s.dx), "spacing": float(s.spacing), "volume": float(s.volume),
            "substeps": int(s.substeps), "control_dt": float(live.control_dt), "gravity": tuple(s.gravity_tuple),
            "pic": float(s.pic), "j_domain": tuple(s.j_domain), "radius": float(s.radius),
            "contact_stiffness": float(s.contact_stiffness), "G": int(s.G), "Gt": int(s.Gt),
            "group_kinds": tuple(g["kind"] for g in live.scene.groups), "boxes": int(s.boxes),
            "use_plane": int(s.use_plane), "markers": len(live.scene.marker_offsets)-1,
            "stable_step_guard": s.cfl_guard_limit, "dirichlet": s.dirichlet, "traction_work": s.traction_work,
            "state_precision": s.state_precision, "projection_rows_nonzero": bool(getattr(s, "proj_rows", 0)),
            "constraint_work": constraint_work_of(s)}


def pack_rubber(lives, *, constitutive_model=cr.DEFAULT_MODEL_ID):
    """Concatenate fresh single-env rubber references (CPU) into one batch."""
    lives = list(lives)
    if not lives:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_EMPTY_BATCH", "num_envs must be >= 1")
    cr.get_model(constitutive_model)
    for live in lives:
        _validate(live)
    sigs = [uniform_signature(live) for live in lives]
    for e, sig in enumerate(sigs[1:], start=1):
        for key, value in sigs[0].items():
            if sig[key] != value:
                raise UnsupportedBatchConfiguration("UNSUPPORTED_NONUNIFORM_BATCH",
                                                    f"{key} differs: env 0 {value!r}, env {e} {sig[key]!r}")
    uniform = dict(sigs[0])
    N, G, Gt, B = len(lives), uniform["G"], uniform["Gt"], uniform["boxes"]
    solvers = [live.solver for live in lives]
    pc = np.array([s.n for s in solvers], np.int64)
    nc = np.array([s.node_count for s in solvers], np.int64)
    cc = np.array([s.cells for s in solvers], np.int64)
    mc = np.array([len(live.scene.marker_particles) for live in lives], np.int64)
    offsets = lambda c: np.r_[0, np.cumsum(c)].astype(np.int64)
    po, no, co, mo = offsets(pc), offsets(nc), offsets(cc), offsets(mc)
    if po[-1]*8 >= INT32_LIMIT or no[-1] >= INT32_LIMIT:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_INT32_OVERFLOW", f"P*8 = {po[-1]*8}, nodes = {no[-1]}")
    host = {}
    for name in PARTICLE_ARRAYS:
        parts = []
        for e, s in enumerate(solvers):
            a = getattr(s, name).numpy()
            if name == "nodes":
                a = a+np.int32(no[e])
            elif name == "particle_traction":
                a = np.where(a >= 0, a+np.int32(e*Gt), a).astype(np.int32)
            parts.append(a)
        host[name] = np.concatenate(parts)
    for name in NODE_ARRAYS:
        parts = []
        for e, s in enumerate(solvers):
            a = getattr(s, name).numpy()
            if name == "node_group":
                a = np.where(a >= 0, a+np.int32(e*G), a).astype(np.int32)
            parts.append(a)
        host[name] = np.concatenate(parts)
    G1 = max(G, 1)
    for name in GROUP_ARRAYS:
        host[name] = np.concatenate([getattr(s, name).numpy()[:G1] for s in solvers]) if G else \
            np.concatenate([getattr(s, name).numpy()[:1] for s in solvers[:1]])
    for name in TRACTION_ARRAYS:
        host[name] = np.concatenate([getattr(s, name).numpy()[:max(Gt, 1)] for s in solvers]) if Gt else \
            getattr(solvers[0], name).numpy()[:1]
    host["cell_node"] = np.concatenate([s.cell_node.numpy()+np.int32(no[e]) for e, s in enumerate(solvers)])
    host["cell_particles"] = np.concatenate([s.cell_particles.numpy()+np.int32(po[e]) for e, s in enumerate(solvers)])
    host["cell_offsets"] = np.concatenate([s.cell_offsets.numpy()[:-1]+np.int32(po[e]) for e, s in enumerate(solvers)]
                                          + [np.array([po[-1]], np.int32)]).astype(np.int32)
    host["marker_particles"] = np.concatenate([lv.scene.marker_particles+np.int32(po[e]) for e, lv in enumerate(lives)])
    host["marker_offsets"] = np.concatenate([lv.scene.marker_offsets[:-1]+np.int32(mo[e]) for e, lv in enumerate(lives)]
                                            + [np.array([mo[-1]], np.int32)]).astype(np.int32)
    host["params"] = np.array([cr.env_parameters(s.material) for s in solvers], np.float64).astype(np.float32)
    host["plane"] = np.stack([np.array(s.plane, np.float32) for s in solvers])
    host["box_c"] = np.concatenate([s.box_c.numpy()[:max(B, 1)] for s in solvers]) if B else solvers[0].box_c.numpy()
    host["box_h"] = np.concatenate([s.box_h.numpy()[:max(B, 1)] for s in solvers]) if B else solvers[0].box_h.numpy()
    host["env_of_particle"] = np.repeat(np.arange(N, dtype=np.int32), pc)
    host["env_of_node"] = np.repeat(np.arange(N, dtype=np.int32), nc)
    host["env_of_group"] = np.repeat(np.arange(N, dtype=np.int32), G1 if G else 0) if G else np.zeros(1, np.int32)
    host["particle_offset"] = po.astype(np.int32)
    host["node_offset"] = no.astype(np.int32)
    host["cell_env_offset"] = co.astype(np.int32)
    # constrained-node list of the BC pass (global node ids in increasing order: the CPU launch visits the same
    # nodes in the same order as the all-node launch, which returns at once on unconstrained nodes)
    bc = np.flatnonzero(host["node_group"] >= 0).astype(np.int32) if G else np.zeros(0, np.int32)
    host["bc_nodes"] = bc if bc.size else np.zeros(1, np.int32)
    option_env = _pack_options(solvers, uniform, host, po, no, N, G)
    env = {"mass_per_particle": np.array([float(s.mass_value) for s in solvers], np.float32),
           "traction_particle_counts": np.array([[np.count_nonzero(lv.scene.particle_traction == t) for t in range(Gt)]
                                                 for lv in lives], np.int64).reshape(N, Gt),
           "marker_reference": np.stack([lv.scene.marker_reference for lv in lives]), **option_env}
    receipts = [{"env": e, "particles": int(s.n), "nodes": int(s.node_count), "cells": int(s.cells),
                 "scene": lv.scene.receipt(), "material": s.material.to_dict(),
                 "params": dict(zip(cr.RUBBER_PARAM_NAMES, host["params"][e].tolist()))}
                for e, (s, lv) in enumerate(zip(solvers, lives))]
    keys = [lv.scene.receipt()["array_sha256"] for lv in lives]
    return PackedRubberBatch(num_envs=N, particle_counts=pc, node_counts=nc, cell_counts=cc,
                             marker_particle_counts=mc, particle_offset=po, node_offset=no, cell_env_offset=co,
                             uniform=uniform, host=host, env=env, receipts=receipts, scene_keys=keys,
                             constitutive_model=constitutive_model)


def projection_cross_group_particles(scene):
    """Number of particles whose stencil holds constrained nodes of two different groups in the same velocity component
    (with nonzero weight). The F8 projection (``tlmpm_rubber_v1.projection_blocks``) builds one block per (group,
    component), so such particles would couple blocks that are solved independently: the batched path refuses them
    (review F8; every current scene has none: case A one group, case C faces L apart, pins inside one group)."""
    nodes, w = scene.nodes, scene.weights
    group, mask = scene.node_group[nodes], scene.node_mask[nodes]
    bad = np.zeros(len(nodes), bool)
    for k in range(3):
        valid = (group >= 0) & (((mask >> k) & 1) == 1) & (w != 0)
        gmin = np.where(valid, group, np.iinfo(group.dtype).max).min(1)
        gmax = np.where(valid, group, -1).max(1)
        bad |= valid.any(1) & (gmin != gmax)
    return int(bad.sum())


def _pack_options(solvers, uniform, host, po, no, N, G):
    """Option arrays (ADR 0023 §20), concatenated from the single references in env order with global ids:
    F64 / x64 shadows; the projection rows (global node and group ids), blocks (global row offsets) and the float64
    block inverses exactly as the single solver computed them (same bits). Returns the per-env offsets."""
    env = {}
    if uniform["state_precision"] == "f64_shadow_v1":
        for name in SHADOW_ARRAYS:
            host[name] = np.concatenate([getattr(s, name).numpy() for s in solvers])
        host["reference64"] = host["x64"].copy()
    if uniform["dirichlet"] == "consistent_mass_projection_v1" and uniform["projection_rows_nonzero"]:
        for e, s in enumerate(solvers):
            coupled = projection_cross_group_particles(s.scene)
            if coupled:
                # code UNSUPPORTED_SOLVER with the fixed tag PROJECTION_CROSS_GROUP_COUPLING (batched/errors.py is a
                # frozen Phase 1 module: no new code is added there)
                raise UnsupportedBatchConfiguration(
                    "UNSUPPORTED_SOLVER",
                    f"PROJECTION_CROSS_GROUP_COUPLING: env {e}: {coupled} particles touch constrained nodes of two groups in the same component; "
                    "consistent_mass_projection_v1 solves per (group, component) blocks independently (ADR 0023 §20.3)")
        rows = np.array([s.proj_rows for s in solvers], np.int64)
        nblocks = np.array([len(s.projection_receipt["blocks"]) for s in solvers], np.int64)
        ninv = np.array([s.proj_inv.shape[0] for s in solvers], np.int64)
        ro, bo, io = (np.r_[0, np.cumsum(c)].astype(np.int64) for c in (rows, nblocks, ninv))
        if ro[-1] >= INT32_LIMIT or io[-1] >= INT32_LIMIT:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_INT32_OVERFLOW", f"rows {ro[-1]}, inverse {io[-1]}")
        cat = lambda name, add: np.concatenate([getattr(s, name).numpy()[:k]+np.int32(a)       # noqa: E731
                                                for s, k, a in zip(solvers, add[1], add[0])]).astype(np.int32)
        host["proj_row_node"] = cat("proj_row_node", (no[:-1], rows))
        host["proj_row_group"] = cat("proj_row_group", (np.arange(N)*G, rows))
        host["proj_row_comp"] = cat("proj_row_comp", (np.zeros(N, np.int64), rows))
        host["proj_row_block"] = cat("proj_row_block", (bo[:-1], rows))
        host["proj_row_local"] = cat("proj_row_local", (np.zeros(N, np.int64), rows))
        host["proj_block_offset"] = cat("proj_block_offset", (ro[:-1], nblocks))
        host["proj_block_size"] = cat("proj_block_size", (np.zeros(N, np.int64), nblocks))
        host["proj_inv_offset"] = cat("proj_inv_offset", (io[:-1], nblocks))
        host["proj_inv"] = np.concatenate([s.proj_inv.numpy() for s in solvers])
        host["env_of_row"] = np.repeat(np.arange(N, dtype=np.int32), rows)
        for name in ("proj_rhs", "proj_vpresc", "proj_mu"):
            host[name] = np.zeros(int(ro[-1]), np.float64)
        host["proj_mu_node"] = np.zeros((int(no[-1]), 3), np.float64)
        env.update(proj_row_offset=ro, proj_block_env_offset=bo, proj_inv_env_offset=io)
    return env


def range_audit(packed, host_arrays):
    """Env-ownership audits on host copies (all counts must be 0)."""
    po, no = packed.particle_offset, packed.node_offset
    env_p = packed.host["env_of_particle"]
    nodes = host_arrays["nodes"]
    out = {"node_out_of_env": int(np.count_nonzero((nodes < no[env_p][:, None]) | (nodes >= no[env_p+1][:, None])))}
    cp, coff, cnode = host_arrays["cell_particles"], host_arrays["cell_offsets"], host_arrays["cell_node"]
    bad_cell = 0
    for e in range(packed.num_envs):
        c0, c1 = packed.cell_env_offset[e], packed.cell_env_offset[e+1]
        ids = cp[coff[c0]:coff[c1]]
        bad_cell += int(np.count_nonzero((ids < po[e]) | (ids >= po[e+1])))
        bad_cell += int(np.count_nonzero((cnode[c0:c1] < no[e]) | (cnode[c0:c1] >= no[e+1])))
    out["cell_csr_out_of_env"] = bad_cell
    mp_, mo = host_arrays["marker_particles"], host_arrays["marker_offsets"]
    K = packed.uniform["markers"]
    bad_marker = 0
    for e in range(packed.num_envs):
        ids = mp_[mo[e*K]:mo[(e+1)*K]]
        bad_marker += int(np.count_nonzero((ids < po[e]) | (ids >= po[e+1])))
    out["marker_out_of_env"] = bad_marker
    G = packed.uniform["G"]
    ng = host_arrays["node_group"]
    env_n = packed.host["env_of_node"]
    held = ng >= 0
    out["group_out_of_env"] = int(np.count_nonzero(held & ((ng < env_n*G) | (ng >= (env_n+1)*G)))) if G else 0
    Gt = packed.uniform["Gt"]
    pt = host_arrays["particle_traction"]
    held = pt >= 0
    out["traction_out_of_env"] = int(np.count_nonzero(held & ((pt < env_p*Gt) | (pt >= (env_p+1)*Gt)))) if Gt else 0
    return out
