"""``trim_bond_progress_packed_v1``: batched TRIM shaping reward over packed cut codes (ADR 0021 §8).

Status ``PROVISIONAL_SHAPING_NOT_SUCCESS``: a dense training signal only. Success comes exclusively from
the frozen independent evaluator at episode end; this reward never decides it, and its weights are
provisional (materials are uncalibrated).

Pack time (host, NumPy, once per scene): from the compiled task's reference particles and integer
lattice (``task.sample_particles()``, the solver's particle order) and the scoring plane
(``task.target_origin_local_m``, normal +Z; ``plane_local_z_m=None`` -> base height):

* reference-lattice bonds = 6-neighbour pairs of the integer lattice;
* **target bonds**: vertical bonds crossing the plane (lower endpoint ``z <= h``, upper ``z > h``,
  the same "above" test as ``seeded_trim.compile_seed``) inside a protrusion footprint
  (``max_b bump.height(x, y) > 0`` at the column);
* **substrate bonds**: bonds with both endpoints ``z <= h`` (optionally only within a depth band).

Per env these become a flat CSR (global particle ids = local id + ``particle_offset[e]``, offsets
``[N+1]``), concatenated over envs. Per step, a segmented reduction over the device code array
``[P]`` counts severed bonds with the frozen conflict rule ``tlmpm_v5._conflict_packed`` (opposite
sides of some cut). Counts are exact integers (cumulative-sum segmentation, deterministic).

Reward per env and step (``config`` weights)::

    progress   = severed_target / n_target
    off_target = severed_substrate / n_target          # normalized by the target size
    r = w_p * Δprogress - w_o * Δoff_target - time_penalty
    r = -invalid_penalty                                # invalid (terminated) envs: shaping terms zeroed

Codes are permanent (digits only go 0 -> 1|2), so both fractions are nondecreasing. The reward reads
privileged codes; it is never an observation (``rl.observations`` keeps oracle values out of "policy").
"""
from dataclasses import asdict, dataclass, field

import numpy as np

REWARD_VERSION = "trim_bond_progress_packed_v1"
REWARD_STATUS = "PROVISIONAL_SHAPING_NOT_SUCCESS"
LOW_BITS = 0x55555555   # tlmpm_v5.LOW_BITS: bit 2k of every digit k


# ----------------------------------------------------------------------------- frozen conflict rule, ported

def conflict_packed_numpy(a, b):
    """Vectorized ``tlmpm_v5._conflict_packed``: (A1 & B2) | (A2 & B1) != 0 on int32 packed codes."""
    a, b = np.asarray(a, np.int32), np.asarray(b, np.int32)
    la, ha = a & LOW_BITS, (a >> 1) & LOW_BITS
    lb, hb = b & LOW_BITS, (b >> 1) & LOW_BITS
    a1, a2 = la ^ (la & ha), ha ^ (la & ha)
    b1, b2 = lb ^ (lb & hb), hb ^ (lb & hb)
    return ((a1 & b2) | (a2 & b1)) != 0


def conflict_packed_torch(a, b):
    """Torch port of ``tlmpm_v5._conflict_packed`` (same bit operations on int32 tensors)."""
    la, ha = a & LOW_BITS, (a >> 1) & LOW_BITS
    lb, hb = b & LOW_BITS, (b >> 1) & LOW_BITS
    a1, a2 = la ^ (la & ha), ha ^ (la & ha)
    b1, b2 = lb ^ (lb & hb), hb ^ (lb & hb)
    return ((a1 & b2) | (a2 & b1)) != 0


# ----------------------------------------------------------------------------- pack time (host)

def lattice_bonds(lattice):
    """All 6-neighbour pairs ``(a, b, axis)`` of an integer lattice, each once (``lattice[b] = lattice[a] + e_axis``)."""
    lattice = np.asarray(lattice, np.int64)
    if lattice.ndim != 2 or lattice.shape[1] != 3 or len(lattice) == 0 or lattice.min() < 0:
        raise ValueError("lattice must be a nonempty [n,3] array of nonnegative integers")
    shape = lattice.max(0)+2
    index = np.full(tuple(shape), -1, np.int64)
    if len(np.unique(lattice, axis=0)) != len(lattice):
        raise ValueError("lattice has duplicate sites")
    index[tuple(lattice.T)] = np.arange(len(lattice))
    out_a, out_b, out_axis = [], [], []
    for axis in range(3):
        step = np.zeros(3, np.int64)
        step[axis] = 1
        neighbour = index[tuple((lattice+step).T)]
        ok = neighbour >= 0
        out_a.append(np.flatnonzero(ok))
        out_b.append(neighbour[ok])
        out_axis.append(np.full(int(ok.sum()), axis, np.int8))
    return (np.concatenate(out_a).astype(np.int64), np.concatenate(out_b).astype(np.int64),
            np.concatenate(out_axis))


@dataclass
class EnvBonds:
    """One env's bonds in LOCAL particle ids (the solver order of ``task.sample_particles()``)."""

    n_particles: int
    target_a: np.ndarray
    target_b: np.ndarray
    substrate_a: np.ndarray
    substrate_b: np.ndarray
    receipt: dict = field(default_factory=dict)


def build_env_bonds(points, lattice, plane_z, footprint_height, *, substrate_depth_m=None):
    """Target/substrate bond sets from reference points (LOCAL), lattice and the scoring plane.

    ``footprint_height(x, y)`` returns the protrusion height above the base at columns (``> 0`` inside a footprint).
    """
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(lattice):
        raise ValueError("points must be [n,3] and match the lattice")
    a, b, axis = lattice_bonds(lattice)
    above = points[:, 2] > plane_z
    vertical = axis == 2
    crossing = vertical & ~above[a] & above[b]
    columns = points[a[crossing]]
    inside = np.asarray(footprint_height(columns[:, 0], columns[:, 1]), float) > 0
    target = np.flatnonzero(crossing)[inside]
    substrate = ~above[a] & ~above[b]
    if substrate_depth_m is not None:
        if not substrate_depth_m > 0:
            raise ValueError("substrate_depth_m must be positive")
        substrate &= np.minimum(points[a, 2], points[b, 2]) >= plane_z-substrate_depth_m
    substrate = np.flatnonzero(substrate)
    if len(target) == 0:
        raise ValueError("no target bonds cross the scoring plane inside a protrusion footprint")
    receipt = {"particles": int(len(points)), "lattice_bonds": int(len(a)), "plane_local_z_m": float(plane_z),
               "crossing_vertical_bonds": int(crossing.sum()), "target_bonds": int(len(target)),
               "crossing_outside_footprint": int((~inside).sum()), "substrate_bonds": int(len(substrate)),
               "substrate_depth_m": substrate_depth_m}
    i32 = np.int32
    return EnvBonds(len(points), a[target].astype(i32), b[target].astype(i32), a[substrate].astype(i32),
                    b[substrate].astype(i32), receipt)


def bonds_from_task(task, *, substrate_depth_m=None, particles=None):
    """``build_env_bonds`` for a compiled TRIM task (``seeded_trim.compile_seed(...)[0]``)."""
    normal = np.asarray(task.target_normal_local, float)
    if task.goal_kind != "flatten" or not np.allclose(normal, [0., 0., 1.], atol=0, rtol=0):
        raise ValueError("trim_bond_progress_packed_v1 needs a flatten (TRIM) task with a +Z scoring plane")
    points, _, lattice = task.sample_particles() if particles is None else particles
    bumps = tuple(task.geometry.bumps)
    if not bumps:
        raise ValueError("task has no protrusions")

    def footprint(x, y):
        return np.max(np.stack([np.asarray(bump.height(x, y), float) for bump in bumps]), axis=0)
    bonds = build_env_bonds(points, lattice, float(task.target_origin_local_m[2]), footprint,
                            substrate_depth_m=substrate_depth_m)
    bonds.receipt.update(task_id=task.task_id, protrusions=len(bumps))
    return bonds


@dataclass
class PackedBonds:
    """Flat per-env CSR over all envs (global particle ids), host NumPy arrays."""

    particle_offset: np.ndarray     # int64 [N+1]
    target_a: np.ndarray            # int32 [Mt]
    target_b: np.ndarray
    target_offset: np.ndarray       # int64 [N+1]
    substrate_a: np.ndarray         # int32 [Ms]
    substrate_b: np.ndarray
    substrate_offset: np.ndarray    # int64 [N+1]
    receipts: list

    @property
    def num_envs(self):
        return len(self.particle_offset)-1

    @property
    def target_count(self):
        return np.diff(self.target_offset)

    @property
    def substrate_count(self):
        return np.diff(self.substrate_offset)


def pack_bonds(env_bonds, particle_offset=None):
    """Concatenate per-env bonds; ``particle_offset`` (the batched layout's ``[N+1]``) must match the counts."""
    counts = np.array([bonds.n_particles for bonds in env_bonds], np.int64)
    expected = np.r_[0, np.cumsum(counts)]
    if particle_offset is not None and not np.array_equal(np.asarray(particle_offset, np.int64), expected):
        raise ValueError("particle_offset does not match the per-env particle counts")
    if expected[-1] >= 2**31:
        raise ValueError("P >= 2^31 exceeds int32 particle ids")

    def cat(name, offsets):
        return np.concatenate([getattr(b, name).astype(np.int64)+offsets[e] for e, b in enumerate(env_bonds)]
                              ).astype(np.int32)
    t_counts = [len(b.target_a) for b in env_bonds]
    s_counts = [len(b.substrate_a) for b in env_bonds]
    return PackedBonds(expected, cat("target_a", expected), cat("target_b", expected), np.r_[0, np.cumsum(t_counts)],
                       cat("substrate_a", expected), cat("substrate_b", expected), np.r_[0, np.cumsum(s_counts)],
                       [dict(b.receipt) for b in env_bonds])


# ----------------------------------------------------------------------------- reward definition

@dataclass(frozen=True)
class TrimBondRewardConfig:
    progress_weight: float = 1.
    off_target_weight: float = 1.
    time_penalty: float = 1e-4
    invalid_penalty: float = 1.

    def __post_init__(self):
        values = np.array([self.progress_weight, self.off_target_weight, self.time_penalty, self.invalid_penalty])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("reward weights must be finite and nonnegative")

    def to_dict(self):
        return {**asdict(self), "version": REWARD_VERSION, "status": REWARD_STATUS}


def _segment_counts_numpy(flags, offsets):
    cs = np.r_[0, np.cumsum(flags, dtype=np.int64)]
    return cs[offsets[1:]]-cs[offsets[:-1]]


def severed_counts_numpy(codes, packed):
    """NumPy reference: per-env (severed target bonds, severed substrate bonds) as int64 [N]."""
    codes = np.asarray(codes, np.int32)
    if codes.shape != (int(packed.particle_offset[-1]),):
        raise ValueError("codes must be the flat [P] packed code array")
    t = conflict_packed_numpy(codes[packed.target_a], codes[packed.target_b])
    s = conflict_packed_numpy(codes[packed.substrate_a], codes[packed.substrate_b])
    return _segment_counts_numpy(t, packed.target_offset), _segment_counts_numpy(s, packed.substrate_offset)


def reward_numpy(previous_counts, counts, invalid, n_target, config):
    """Reference reward from (target, substrate) severed counts before/after one step."""
    n_target = np.asarray(n_target, np.float64)
    dp = (counts[0]-previous_counts[0])/n_target
    do = (counts[1]-previous_counts[1])/n_target
    shaped = config.progress_weight*dp-config.off_target_weight*do-config.time_penalty
    invalid = np.asarray(invalid, bool)
    return np.where(invalid, -config.invalid_penalty, shaped), dp, do


# ----------------------------------------------------------------------------- device reducer (Torch)

class TrimBondProgressReward:
    """Device reward state for N envs. All per-step work is batched tensor ops over the flat CSR."""

    version, status = REWARD_VERSION, REWARD_STATUS

    def __init__(self, packed, config=None, *, device="cpu"):
        import torch
        self.torch, self.config, self.packed = torch, config or TrimBondRewardConfig(), packed
        self.device = torch.device(device)
        i64 = dict(dtype=torch.int64, device=self.device)
        self.num_envs, self.num_particles = packed.num_envs, int(packed.particle_offset[-1])
        self.target_a = torch.as_tensor(packed.target_a.astype(np.int64), **i64)
        self.target_b = torch.as_tensor(packed.target_b.astype(np.int64), **i64)
        self.substrate_a = torch.as_tensor(packed.substrate_a.astype(np.int64), **i64)
        self.substrate_b = torch.as_tensor(packed.substrate_b.astype(np.int64), **i64)
        self.target_offset = torch.as_tensor(packed.target_offset, **i64)
        self.substrate_offset = torch.as_tensor(packed.substrate_offset, **i64)
        self.n_target = torch.as_tensor(packed.target_count, dtype=torch.float64, device=self.device)
        if (packed.target_count <= 0).any():
            raise ValueError("every env needs at least one target bond")
        self.counts = torch.zeros((2, self.num_envs), **i64)      # last (target, substrate) severed counts

    def metadata(self):
        return {"version": REWARD_VERSION, "status": REWARD_STATUS, "config": self.config.to_dict(),
                "target_bonds": self.packed.target_count.tolist(),
                "substrate_bonds": self.packed.substrate_count.tolist(), "per_env": self.packed.receipts,
                "conflict_rule": "tlmpm_v5._conflict_packed", "reduction": "int64 cumulative-sum segments (exact)"}

    def _segments(self, flags, offsets):
        torch = self.torch
        cs = torch.cat([torch.zeros(1, dtype=torch.int64, device=self.device), torch.cumsum(flags, 0)])
        return cs[offsets[1:]]-cs[offsets[:-1]]

    def severed_counts(self, codes):
        """Device ``codes [P]`` int32 -> int64 ``[2, N]`` (severed target, severed substrate)."""
        torch = self.torch
        if codes.shape != (self.num_particles,) or codes.dtype != torch.int32 or codes.device != self.device:
            raise ValueError(f"codes must be int32 [{self.num_particles}] on {self.device}")
        t = conflict_packed_torch(codes[self.target_a], codes[self.target_b]).to(torch.int64)
        s = conflict_packed_torch(codes[self.substrate_a], codes[self.substrate_b]).to(torch.int64)
        return torch.stack([self._segments(t, self.target_offset), self._segments(s, self.substrate_offset)])

    def reset(self, codes, env_ids=None):
        """Re-baseline the counts of ``env_ids`` (all when None) from the current codes; others untouched."""
        counts = self.severed_counts(codes)
        if env_ids is None:
            self.counts.copy_(counts)
        else:
            ids = self.torch.as_tensor(env_ids, dtype=self.torch.int64, device=self.device)
            self.counts[:, ids] = counts[:, ids]
        return self.fractions(self.counts)

    def fractions(self, counts):
        return {"target_bond_progress_fraction": counts[0].to(self.torch.float64)/self.n_target,
                "off_target_bond_fraction": counts[1].to(self.torch.float64)/self.n_target}

    def step(self, codes, invalid=None):
        """One control step: ``reward [N]`` float32 and a dict of ``[N]`` reward terms."""
        torch, c = self.torch, self.config
        counts = self.severed_counts(codes)
        delta = (counts-self.counts).to(torch.float64)/self.n_target
        shaped = c.progress_weight*delta[0]-c.off_target_weight*delta[1]-c.time_penalty
        if invalid is None:
            invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reward = torch.where(invalid, torch.full_like(shaped, -c.invalid_penalty), shaped)
        self.counts.copy_(counts)
        terms = {"progress_delta": delta[0], "off_target_delta": delta[1], **self.fractions(counts),
                 "severed_target_bonds": counts[0], "severed_substrate_bonds": counts[1],
                 "invalid_penalty_applied": invalid}
        return reward.to(torch.float32), terms
