"""Flat structure-of-arrays layout of N independent tlmpm_v7_2 envs (ADR 0021 §3.1-3.4, §4).

Every env's initial state is packed from its own frozen single solver, built on
the Warp CPU device through the frozen ``LiveCuttingMaterial.reset`` path with
no stepping (§3.4). Geometry, masses, stencil, link CSR, derived material
config and initial pose are therefore frozen-identical by construction; this
module never re-derives them. It only concatenates:

- per-particle arrays in env order, with exact per-env counts (no padding);
  ``nodes`` become global node ids (``local + node_offset[e]``), the link CSR
  is concatenated with global offsets and global particle ids;
- per-node-slot arrays (``node*S + s``) by global node id;
- small per-env state as ``[N*k]`` rows; ``stats`` and ``contact_totals`` are
  split into separate sum and max arrays (§5 L4), unused entries stay zero;
- per-env float32 parameters ``params[N, k]`` (constitutive.PARAM_NAMES).

Uniform quantities (dt, dx, spacing, substeps, S, K, law, tool profile, B,
contact flags, pic, local gravity, floor) are checked here; any difference
raises ``UnsupportedBatchConfiguration``.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np

from . import constitutive
from .errors import UnsupportedBatchConfiguration

INT32_LIMIT = 2**31
SOLVER = "tlmpm_v7_2"
LAYOUT_VERSION = "rl_batch_layout_v1"

# Per-particle arrays carried over from the single solver (ADR §3.1). Order is the receipt order.
PARTICLE_ARRAYS = ("x", "v", "reference", "mass", "code", "F", "Fpi", "stress", "plastic", "progress",
                   "contact_work", "impulse", "nodes", "weights", "gradients", "slot", "x_start",
                   "front_crossings", "front_reached", "link_velocity_start", "link_codes", "pending_q",
                   "pending_cut")
# Node-slot scratch arrays (node*S + s).
NODE_SLOT_ARRAYS = ("slot_code", "gm", "gp", "gj", "old", "new_v")
# Small per-env state: name -> width key (segments, boxes, trajectory length, or a fixed int).
ENV_ARRAYS = {"edge_flags": "segments", "segment_ok": "segments", "segment_plow": "segments",
              "stroke_engaged": 1, "stroke_any_motion": 1, "stroke": 1, "stroke_travel": 1,
              "box_centers": "boxes", "box_halves": "boxes", "box_reactions": "boxes",
              "slot_overflow": 1, "slot_max_occupied": 1, "previous_cut": 1, "reaction": 1,
              "poses": "trajectory", "twists": "trajectory", "cuts": "trajectory",
              "previous_pose": 1, "link_previous_pose": 1, "link_diagnostics": 2}
# Split families (§5 L4): single-env index -> family.
STATS_WIDTH, TOTALS_WIDTH = 7, 12
STATS_MAX = (0, 3, 6)
STATS_SUM = (1, 2, 4, 5)
TOTALS_MAX = (10, 11)
TOTALS_SUM = tuple(range(10))
# Arrays shared by all envs (uniform), compared once.
SHARED_ARRAYS = ("index", "profile")
LINK_ARRAYS = ("link_offsets", "link_neighbours")

FROZEN_SOURCES = ("tlmpm_v5.py", "tlmpm_v6.py", "tlmpm_v7.py", "tlmpm_v7_1.py", "tlmpm_v7_2.py",
                  "tlmpm_v7_2_contacts.py", "tlmpm_v7_2_frontier.py", "fixture_contacts.py",
                  "material_diagnostics.py", "live_material.py", "coupled_robot.py")
_PACKAGE = Path(__file__).resolve().parents[1]


def frozen_source_sha256():
    return {name: hashlib.sha256((_PACKAGE/name).read_bytes()).hexdigest() for name in FROZEN_SOURCES}


@dataclass
class EnvSpec:
    """One env of a batch: an un-reset ``LiveCuttingMaterial`` on device 'cpu', its initial measured
    edge pose (world, XYZ+XYZW) and the reset seed. ``scene_key`` identifies the geometry (for the
    partial-reset geometry guard); by default the task config hash."""
    material: object
    measured_edge_pose: object
    seed: int = 42
    scene_key: object = None


@dataclass
class PackedBatch:
    """Host side of a packed batch: layout metadata plus concatenated initial arrays."""
    num_envs: int
    particle_counts: np.ndarray
    node_counts: np.ndarray
    link_counts: np.ndarray
    fixture_counts: np.ndarray
    particle_offset: np.ndarray        # [N+1] int64
    node_offset: np.ndarray            # [N+1]
    link_offset: np.ndarray            # [N+1]
    fixture_offset: np.ndarray         # [N+1]
    uniform: dict                      # uniform quantities (kernel scalars)
    host: dict                         # name -> concatenated numpy initial value
    env: dict                          # per-env metadata tables (numpy, [N, ...])
    receipts: list                     # per-env parameter/solver receipts
    constitutive_model: str = constitutive.DEFAULT_MODEL_ID
    scene_keys: list = field(default_factory=list)

    @property
    def total_particles(self):
        return int(self.particle_offset[-1])

    @property
    def total_nodes(self):
        return int(self.node_offset[-1])

    def env_of_particle(self):
        return np.repeat(np.arange(self.num_envs, dtype=np.int32), self.particle_counts)

    def layout_receipt(self):
        u = self.uniform
        return {"layout_version": LAYOUT_VERSION, "num_envs": self.num_envs,
                "particle_counts": self.particle_counts.tolist(), "node_counts": self.node_counts.tolist(),
                "link_counts": self.link_counts.tolist(), "fixture_counts": self.fixture_counts.tolist(),
                "total_particles": self.total_particles, "total_nodes": self.total_nodes,
                "slots_per_node": u["slots"], "trajectory_length": u["trajectory"], "segments": u["segments"],
                "boxes": u["boxes"], "int32_guards": int32_guards(self),
                "constitutive_model": self.constitutive_model,
                "stats_split": {"max": list(STATS_MAX), "sum": list(STATS_SUM)},
                "totals_split": {"max": list(TOTALS_MAX), "sum": list(TOTALS_SUM)}}


# ----------------------------------------------------------------------------- single-env validation

def _validate_material(material):
    from ..live_material import LiveCuttingMaterial
    if not isinstance(material, LiveCuttingMaterial):
        raise TypeError("EnvSpec.material must be a LiveCuttingMaterial")
    if material.solver != SOLVER:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER", f"solver {material.solver!r}; only {SOLVER}")
    if material.tool_profile is None:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER", "legacy wedge path without a tool profile")
    if not material.v7_2_optimizations:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_V7_2_OPTIMIZATIONS_DISABLED",
                                            "v7_2 rollbacks stay single-env features")
    if material.cut_events is not None:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_RECORDER", f"cut_events={material.cut_events!r}")
    if material.config.device != "cpu":
        raise ValueError("Pack builds each frozen single solver on the Warp CPU device (ADR §3.4); "
                         "construct the LiveCuttingMaterial with CouplingConfig(device='cpu')")


def validate_bed(bed):
    """Refuse single solvers the batched kernels do not implement."""
    from .. import tlmpm_v7_2
    if type(bed) is not tlmpm_v7_2.TLMPM:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER", f"{type(bed).__module__}.{type(bed).__name__}")
    if not (bed.optimize_contacts and bed.optimize_frontier):
        raise UnsupportedBatchConfiguration("UNSUPPORTED_V7_2_OPTIMIZATIONS_DISABLED",
                                            "optimize_contacts and optimize_frontier must both be True")
    if bed.config.floor:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_LEGACY_FLOOR", "floor=True tray")
    if bed.cutting_law != 8:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER", f"cutting law {bed.cutting_law}")
    if str(bed.device) != "cpu":
        raise ValueError("Pack reads frozen single solvers built on the Warp CPU device")
    if (int(bed.index.numpy()[0]) != 0 or np.count_nonzero(bed.code.numpy())
            or not np.array_equal(bed.x.numpy(), bed.reference.numpy())):
        raise UnsupportedBatchConfiguration("UNSUPPORTED_SINGLE_ENV_STATE",
                                            "pack takes fresh, unstepped reset states only")


def uniform_signature(bed, control_dt=None):
    """Quantities that must be identical across a batch (kernel scalars of the batched step).

    Returns (signature, tool_keys): tool_keys name the entries that belong to the tool profile."""
    c = bed.config
    fixture = bed.fixture_data is not None
    signature = {
        "dt": float(c.dt), "dx": float(c.dx), "spacing": float(bed.spacing), "radius": float(bed.radius),
        "volume": float(bed.volume), "trajectory": int(bed.trajectory_length), "slots": int(bed.slots_per_node),
        "max_cuts": int(bed.max_cuts), "cutting_law": int(bed.cutting_law), "segments": int(bed.edge_segments),
        "boxes": int(len(bed.box_centers)), "obstacle_count": int(len(bed.obstacle_boxes_edge)),
        "material_contact": bool(c.material_contact), "pic": float(c.pic),
        "gravity": tuple(np.asarray(c.gravity, np.float32).tolist()), "floor": bool(c.floor),
        "fixture_walls": bool(c.fixture_walls), "tool": bool(c.tool),
        "contact_radius_scale": float(c.contact_radius_scale), "fixtures": fixture,
        "law4_parameters": tuple(sorted(bed.law4_parameters.items())),
        "law8_parameters": tuple(sorted(bed.law8_parameters.items())),
        "optimize_contacts": bool(bed.optimize_contacts), "optimize_frontier": bool(bed.optimize_frontier),
        "control_dt": None if control_dt is None else float(control_dt),
        # tool profile
        "blade_profile_xz": np.asarray(bed.blade_profile_xz, np.float32).tobytes(),
        "blade_half_width": float(c.blade_half_width), "blade_length": float(c.blade_length),
        "blade_height": float(c.blade_height), "bevel_depth_m": float(bed.bevel_depth_m),
        "blade_half_thickness_m": float(bed.blade_half_thickness_m),
    }
    tool_keys = ("blade_profile_xz", "blade_half_width", "blade_length", "blade_height", "bevel_depth_m",
                 "blade_half_thickness_m")
    return signature, tool_keys


def check_uniform(signatures, tool_keys):
    first = signatures[0]
    for e, other in enumerate(signatures[1:], start=1):
        for key, value in first.items():
            if other[key] != value:
                code = "UNSUPPORTED_PER_ENV_TOOL_PROFILE" if key in tool_keys else "UNSUPPORTED_NONUNIFORM_BATCH"
                shown = "<bytes>" if isinstance(value, bytes) else value
                raise UnsupportedBatchConfiguration(code, f"{key} differs: env 0 {shown!r}, env {e} "
                                                          f"{'<bytes>' if isinstance(other[key], bytes) else other[key]!r}")


def int32_guards(packed):
    S = packed.uniform["slots"]
    return {"particles_x8": int(packed.total_particles)*8, "nodes_x_slots": int(packed.total_nodes)*S,
            "link_entries": int(packed.link_offset[-1]), "limit": INT32_LIMIT}


def check_int32(total_particles, total_nodes, slots, link_entries):
    if total_particles*8 >= INT32_LIMIT:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_INT32_OVERFLOW", f"P*8 = {total_particles*8}")
    if total_nodes*slots >= INT32_LIMIT:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_INT32_OVERFLOW", f"nodes_total*S = {total_nodes*slots}")
    if link_entries >= INT32_LIMIT:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_INT32_OVERFLOW", f"link entries = {link_entries}")


# ----------------------------------------------------------------------------- packing

def build_single(spec):
    """Frozen reset of one env on the CPU device (no stepping): returns the reset receipt."""
    _validate_material(spec.material)
    return spec.material.reset(np.asarray(spec.measured_edge_pose, float), seed=int(spec.seed))


def pack(specs, *, constitutive_model=constitutive.DEFAULT_MODEL_ID):
    """Pack N env specs (ADR §3.4): frozen CPU reset of each, then concatenation."""
    if not specs:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_EMPTY_BATCH", "num_envs must be >= 1")
    beds, meta = [], []
    for spec in specs:
        reset_receipt = build_single(spec)
        m = spec.material
        meta.append(dict(T_world_local=np.asarray(m.T_world_local, float),
                         measured_edge_pose=np.asarray(m.last_pose, float), seed=int(m.seed),
                         control_dt=float(m.config.control_dt), substeps=int(m.substeps),
                         recipients=m.obstacle_reaction_recipients,
                         scene_key=spec.scene_key if spec.scene_key is not None
                         else getattr(m.task, "config_sha256", None),
                         parameter_receipt=m.parameter_receipt, reset_receipt=reset_receipt))
        beds.append(m.bed)
    packed = pack_from_beds(beds, meta, constitutive_model=constitutive_model)
    return packed


def pack_from_beds(beds, meta=None, *, constitutive_model=constitutive.DEFAULT_MODEL_ID):
    """Concatenate fresh frozen single v7_2 solvers (CPU) into one batch.

    ``meta`` (optional, one dict per env) carries T_world_local, measured_edge_pose, seed, control_dt,
    recipients, scene_key and receipts; tests may pass None for directly constructed solvers."""
    if not beds:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_EMPTY_BATCH", "num_envs must be >= 1")
    constitutive.get_model(constitutive_model)
    n_env = len(beds)
    meta = meta or [dict() for _ in beds]
    for bed in beds:
        validate_bed(bed)
    signatures = []
    for bed, m in zip(beds, meta):
        signature, tool_keys = uniform_signature(bed, m.get("control_dt"))
        signatures.append(signature)
    check_uniform(signatures, tool_keys)
    recipients = [m.get("recipients") for m in meta]
    if any(r != recipients[0] for r in recipients):
        raise UnsupportedBatchConfiguration("UNSUPPORTED_NONUNIFORM_BATCH", "obstacle reaction recipients differ")
    uniform = dict(signatures[0])
    uniform["recipients"] = recipients[0]
    uniform["substeps"] = uniform["trajectory"]-1

    S = uniform["slots"]
    particle_counts = np.array([b.n for b in beds], np.int64)
    node_counts = np.array([b.node_count for b in beds], np.int64)
    link_counts = np.array([len(b.link_neighbours) for b in beds], np.int64)
    fixture_counts = np.array([0 if b.fixture_data is None else len(b.fixture_data[1]) for b in beds], np.int64)
    offsets = lambda counts: np.r_[0, np.cumsum(counts)].astype(np.int64)
    particle_offset, node_offset = offsets(particle_counts), offsets(node_counts)
    link_offset, fixture_offset = offsets(link_counts), offsets(fixture_counts)
    check_int32(int(particle_offset[-1]), int(node_offset[-1]), S, int(link_offset[-1]))

    host = {}
    for name in PARTICLE_ARRAYS:
        parts = []
        for e, bed in enumerate(beds):
            value = getattr(bed, name).numpy()
            if name == "nodes":
                value = value+np.int32(node_offset[e])
            elif name == "pending_q":
                value = np.where(value >= 0, value+np.int32(particle_offset[e]), value).astype(np.int32)
            parts.append(value)
        host[name] = np.concatenate(parts)
    for name in NODE_SLOT_ARRAYS:
        host[name] = np.concatenate([getattr(bed, name).numpy() for bed in beds])
    for name in ENV_ARRAYS:
        host[name] = np.concatenate([getattr(bed, name).numpy() for bed in beds])
    for split, (width, sums, maxima) in (("stats", (STATS_WIDTH, STATS_SUM, STATS_MAX)),
                                         ("contact_totals", (TOTALS_WIDTH, TOTALS_SUM, TOTALS_MAX))):
        values = np.stack([getattr(bed, split).numpy() for bed in beds])
        total, maximum = np.zeros_like(values), np.zeros_like(values)
        total[:, list(sums)] = values[:, list(sums)]
        maximum[:, list(maxima)] = values[:, list(maxima)]
        host[f"{split}_sum"], host[f"{split}_max"] = total.reshape(-1), maximum.reshape(-1)
    host["link_offsets"] = np.concatenate(
        [beds[e].link_offsets.numpy()[:-1]+np.int32(link_offset[e]) for e in range(n_env)]
        + [np.array([link_offset[-1]], np.int32)]).astype(np.int32)
    host["link_neighbours"] = np.concatenate(
        [beds[e].link_neighbours.numpy()+np.int32(particle_offset[e]) for e in range(n_env)]).astype(np.int32)
    host["profile"] = beds[0].profile.numpy()
    host["index"] = beds[0].index.numpy()
    if uniform["fixtures"]:
        host["fixture_centers"] = np.concatenate([b.fixture_data[1].numpy() for b in beds])
        host["fixture_halves"] = np.concatenate([b.fixture_data[2].numpy() for b in beds])
        host["fixture_plane"] = np.stack([np.array(b.fixture_data[3], np.float32) for b in beds])
    else:
        host["fixture_centers"] = np.zeros((0, 3), np.float32)
        host["fixture_halves"] = np.zeros((0, 3), np.float32)
        host["fixture_plane"] = np.zeros((n_env, 4), np.float32)
    host["params"] = np.array([constitutive.env_parameters(b.config, b.slice_efficiency) for b in beds],
                              np.float64).astype(np.float32)
    host["env_of_particle"] = np.repeat(np.arange(n_env, dtype=np.int32), particle_counts)
    host["particle_offset"] = particle_offset.astype(np.int32)
    host["node_offset"] = node_offset.astype(np.int32)

    eye = np.eye(4)
    env = dict(
        T_world_local=np.stack([np.asarray(m.get("T_world_local", eye), float) for m in meta]),
        measured_edge_pose=np.stack([np.asarray(m.get("measured_edge_pose", b.poses.numpy()[0]), float)
                                     for m, b in zip(meta, beds)]),
        initial_local_pose=np.stack([b.poses.numpy()[0] for b in beds]).astype(np.float32),
        initial_previous_pose=np.stack([b.previous_pose.numpy()[0] for b in beds]).astype(np.float32),
        initial_link_previous_pose=np.stack([b.link_previous_pose.numpy()[0] for b in beds]).astype(np.float32),
        initial_box_centers=host["box_centers"].reshape(n_env, -1, 3).copy(),
        initial_box_halves=host["box_halves"].reshape(n_env, -1, 3).copy(),
        seed=np.array([int(m.get("seed", 0)) for m in meta], np.int64),
        grid_origin=np.stack([b.grid_origin for b in beds]), grid_dims=np.stack([b.grid_dims for b in beds]))
    if not all(np.array_equal(env["initial_local_pose"][e], beds[e].poses.numpy()[t])
               for e in range(n_env) for t in (0, -1)):
        raise UnsupportedBatchConfiguration("UNSUPPORTED_SINGLE_ENV_STATE", "initial trajectory is not a tile")
    receipts = []
    for e, (bed, m) in enumerate(zip(beds, meta)):
        receipts.append({"env": e, "particles": int(bed.n), "node_count": int(bed.node_count),
                         "grid_dims": [int(v) for v in bed.grid_dims], "seed": int(m.get("seed", 0)),
                         "scene_key": m.get("scene_key"), "parameter_receipt": m.get("parameter_receipt"),
                         "solver_receipt": bed.solver_receipt(),
                         "params": dict(zip(constitutive.PARAM_NAMES, host["params"][e].tolist()))})
    return PackedBatch(num_envs=n_env, particle_counts=particle_counts, node_counts=node_counts,
                       link_counts=link_counts, fixture_counts=fixture_counts, particle_offset=particle_offset,
                       node_offset=node_offset, link_offset=link_offset, fixture_offset=fixture_offset,
                       uniform=uniform, host=host, env=env, receipts=receipts,
                       constitutive_model=constitutive_model, scene_keys=[m.get("scene_key") for m in meta])


# ----------------------------------------------------------------------------- audits (ADR §4)

def range_audit(packed, nodes, link_offsets, link_neighbours, pending_q, env_of_particle):
    """Isolation audits on host copies: every stencil node, CSR range, neighbour and pending id
    belongs to its own env. Returns counts of violations (all must be 0)."""
    po, no = packed.particle_offset, packed.node_offset
    env = np.asarray(env_of_particle)
    expected_env = packed.env_of_particle()
    lo, hi = no[env][:, None], no[env+1][:, None]
    node_violations = int(np.count_nonzero((nodes < lo) | (nodes >= hi)))
    neighbour_violations = 0
    offset_violations = 0
    for e in range(packed.num_envs):
        start, stop = int(packed.link_offset[e]), int(packed.link_offset[e+1])
        seg = link_offsets[no[e]:no[e+1]+1]
        offset_violations += int(seg[0] != start) + int(seg[-1] != stop) + int(np.count_nonzero(np.diff(seg) < 0))
        ids = link_neighbours[start:stop]
        neighbour_violations += int(np.count_nonzero((ids < po[e]) | (ids >= po[e+1])))
    pending = np.asarray(pending_q)
    active = pending >= 0
    pending_violations = int(np.count_nonzero(active & ((pending < po[env]) | (pending >= po[env+1]))))
    return {"env_of_particle_mismatch": int(np.count_nonzero(env != expected_env)),
            "node_out_of_env": node_violations, "link_offset_violations": offset_violations,
            "neighbour_out_of_env": neighbour_violations, "pending_out_of_env": pending_violations}
