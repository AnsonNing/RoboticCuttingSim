"""``sever_bond_progress_packed_v1``: batched CUT shaping reward over packed cut codes (P1-5 T5.3).

Status ``PROVISIONAL_SHAPING_NOT_SUCCESS``: a dense training signal only. Success comes exclusively from the
frozen independent evaluator (``goals.evaluate_goal``) at episode end; this reward never decides it, and its
weights are provisional (materials are uncalibrated). It reads privileged codes and is never an observation.

Pack time (host, NumPy, once per scene), from the compiled task's reference particles and integer lattice
(``task.sample_particles()``, the solver's particle order, workpiece-LOCAL) and the goal's declared cut
planes (``goals.goal_planes``: the ``sever`` planes, or the ``divide`` positions), all LOCAL:

* reference-lattice bonds = 6-neighbour pairs of the integer lattice (``reward_packed.lattice_bonds``);
* **target bonds**: bonds whose endpoints lie on opposite sides of some declared plane (signed distance
  ``s = (p - o) . n``; one endpoint ``s <= 0``, the other ``s > 0``, the TRIM "above" test);
* **off-target bonds**: bonds whose midpoint lies farther than the goal's frozen ``plane_error_bound_m``
  (2 mm for ``sever_mid``) from every declared plane. Bonds within the bound that do not cross a plane are
  neutral (the kerf the frozen location statistic tolerates).

Per step the device reducer of ``trim_bond_progress_packed_v1`` (``reward_packed.TrimBondProgressReward``,
reused unchanged: flat per-env CSR, frozen conflict rule ``tlmpm_v5._conflict_packed``, exact int64
cumulative-sum segments) counts severed target and off-target bonds; the off-target set is stored in the
CSR's ``substrate_*`` slots. Reward per env and step::

    progress   = severed_target / n_target
    off_target = severed_off_target / n_target          # normalized by the target size
    r = w_p * d(progress) - w_o * d(off_target) - time_penalty
    r = -invalid_penalty                                 # invalid (terminated) envs: shaping terms zeroed

``bonds_for_task`` dispatches by goal type (``flatten`` -> TRIM bonds, ``sever``/``divide`` -> these), the hook a
configured-scene task family (T5.2) uses; ``reward_class_for_version`` returns the matching device reducer.
"""
from dataclasses import dataclass

import numpy as np

from . import reward_packed as RP

REWARD_VERSION = "sever_bond_progress_packed_v1"
REWARD_STATUS = "PROVISIONAL_SHAPING_NOT_SUCCESS"
SEVER_GOAL_TYPES = ("sever", "divide")


def build_sever_bonds(points, lattice, planes, bound_m):
    """``EnvBonds`` (LOCAL ids) of a sever/divide task: target = plane-crossing bonds, off-target (stored in
    ``substrate_*``) = bonds with midpoint distance > ``bound_m`` from every plane."""
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(lattice):
        raise ValueError("points must be [n,3] and match the lattice")
    if not planes:
        raise ValueError("a sever reward needs at least one declared plane")
    if not (np.isfinite(bound_m) and bound_m > 0):
        raise ValueError("bound_m must be positive")
    a, b, axis = RP.lattice_bonds(lattice)
    origins = np.stack([np.asarray(o, float) for o, _ in planes])
    normals = np.stack([np.asarray(n, float) for _, n in planes])
    if not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-9, rtol=0):
        raise ValueError("plane normals must be unit vectors")
    signed = np.einsum("pkj,kj->pk", points[:, None, :] - origins[None], normals)    # [n, planes]
    above = signed > 0
    crossing = (above[a] != above[b]).any(1)
    midpoint = .5 * (signed[a] + signed[b])
    off = (np.abs(midpoint) > bound_m).all(1)
    target = np.flatnonzero(crossing)
    off_target = np.flatnonzero(off & ~crossing)
    if len(target) == 0:
        raise ValueError("no lattice bond crosses a declared sever plane")
    per_plane = [int(((above[a, k] != above[b, k])).sum()) for k in range(len(planes))]
    receipt = {"particles": int(len(points)), "lattice_bonds": int(len(a)), "planes": len(planes),
               "target_bonds": int(len(target)), "target_bonds_per_plane": per_plane,
               "off_target_bonds": int(len(off_target)), "neutral_bonds": int(len(a) - len(target) - len(off_target)),
               "off_target_bound_m": float(bound_m),
               "target_bonds_by_axis": np.bincount(axis[target], minlength=3).tolist(),
               "off_target_slot": "EnvBonds.substrate_* holds the off-target bonds"}
    i32 = np.int32
    return RP.EnvBonds(len(points), a[target].astype(i32), b[target].astype(i32), a[off_target].astype(i32),
                       b[off_target].astype(i32), receipt)


def sever_bonds_from_task(task, *, particles=None):
    """``build_sever_bonds`` for a compiled sever/divide ``ConfiguredTaskSpec`` (goal planes, frozen bound)."""
    from ..goals import goal_planes
    goal = task.goal
    if goal["type"] not in SEVER_GOAL_TYPES:
        raise ValueError(f"{REWARD_VERSION} needs a sever/divide goal, got {goal['type']!r}")
    planes = goal_planes(goal, task.scene)
    bound = float(goal["plane_error_bound_m"])
    points, _, lattice = task.sample_particles() if particles is None else particles
    bonds = build_sever_bonds(points, lattice, planes, bound)
    bonds.receipt.update(task_id=task.task_id, goal_id=goal["goal_id"], goal_type=goal["type"])
    return bonds


def bonds_for_task(task, *, particles=None, substrate_depth_m=None):
    """Reward bonds by goal kind: ``(reward_version, EnvBonds)``; other goals are refused (never a fallback)."""
    kind = getattr(task, "goal_kind", None)
    goal_type = task.goal["type"] if hasattr(task, "goal_json") and task.goal_json else None
    if goal_type in SEVER_GOAL_TYPES:
        return REWARD_VERSION, sever_bonds_from_task(task, particles=particles)
    if kind == "flatten":
        return RP.REWARD_VERSION, RP.bonds_from_task(task, particles=particles, substrate_depth_m=substrate_depth_m)
    raise ValueError(f"UNSUPPORTED_REWARD: no packed shaping reward for goal kind {kind!r} / type {goal_type!r}")


@dataclass(frozen=True)
class SeverBondRewardConfig(RP.TrimBondRewardConfig):
    """Same provisional weights as the TRIM reward (progress 1, off-target 1, time 1e-4, invalid 1)."""

    def to_dict(self):
        return {**super().to_dict(), "version": REWARD_VERSION, "status": REWARD_STATUS}


def reward_numpy(previous_counts, counts, invalid, n_target, config):
    """NumPy reference reward (identical formula to the TRIM reference)."""
    return RP.reward_numpy(previous_counts, counts, invalid, n_target, config)


def severed_counts_numpy(codes, packed):
    """NumPy reference: per-env (severed target, severed off-target) as int64 [N]."""
    return RP.severed_counts_numpy(codes, packed)


class SeverBondProgressReward(RP.TrimBondProgressReward):
    """Device reward state for N CUT envs: the TRIM reducer on sever bonds (see the module docstring)."""

    version, status = REWARD_VERSION, REWARD_STATUS

    def __init__(self, packed, config=None, *, device="cpu"):
        super().__init__(packed, config or SeverBondRewardConfig(), device=device)

    def metadata(self):
        out = super().metadata()
        out.update(version=REWARD_VERSION, status=REWARD_STATUS, config=self.config.to_dict(),
                   off_target_bonds=out.pop("substrate_bonds"),
                   target_definition="lattice bonds crossing a declared sever plane",
                   off_target_definition="bonds with midpoint farther than plane_error_bound_m from every plane")
        return out

    def step(self, codes, invalid=None):
        reward, terms = super().step(codes, invalid)
        # the shared vector-env contract keeps the CSR slot name ``severed_substrate_bonds`` (here: off-target)
        terms["severed_off_target_bonds"] = terms["severed_substrate_bonds"]
        return reward, terms


def reward_class_for_version(version):
    """Device reducer class and default config of a reward version."""
    if version == REWARD_VERSION:
        return SeverBondProgressReward, SeverBondRewardConfig()
    if version == RP.REWARD_VERSION:
        return RP.TrimBondProgressReward, RP.TrimBondRewardConfig()
    raise ValueError(f"UNSUPPORTED_REWARD: {version!r}")
