"""Provisional shaped reward for learning on the shared CUT/TRIM API.

``CuttingReward(task)(env, info)`` reads privileged simulator state (reference
particles, persistent cut codes, measured free-tool edge pose). It is a
learning signal only: it never enters observations, and task success remains
``env.evaluate()``. Terms and weights are engineering defaults, not calibrated.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = "cutting_reward_v1"  # v0 (label-based progress) was replaced before any recorded use


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 10.        # per unit increase of severed target-plane column fraction
    off_target: float = 2.       # per unit increase of severed-bond fraction away from the target
    gouge_per_m: float = 200.    # armed edge below a flatten target plane
    lateral_per_m: float = 50.   # edge motion across the blade plane while inside the workpiece
    ik_reject: float = .1
    clamp: float = .01
    time: float = .001
    termination: float = 5.

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all() or min(asdict(self).values()) < 0:
            raise ValueError("Reward weights must be finite and nonnegative")


class CuttingReward:
    """Callable reward adapter; keeps per-episode state between steps."""

    def __init__(self, task, weights=RewardWeights(), band_m=None, margin_m=.01):
        self.task, self.weights = task, weights
        # Half a grid cell plus half a particle: at least one particle layer per side.
        self.band_m = float((task.grid_dx_m+task.particle_spacing_m)/2 if band_m is None else band_m)
        self.margin_m = float(margin_m)
        self._episode_key = None

    def to_dict(self):
        return {"version": VERSION, "weights": asdict(self.weights), "band_m": self.band_m,
                "privileged_inputs": ["reference particles and lattice", "cut codes", "measured tool edge pose"],
                "progress": "fraction of target-plane columns with a severed reference bond",
                "status": "PROVISIONAL_SHAPING_NOT_SUCCESS"}

    def _start(self, env):
        """Reference bonds, their target columns and the off-target set (fixed per episode).

        Progress is measured on severed reference bonds (opposite cut sides on
        the two particles), not on labels: a label on one side only separates
        nothing and must not earn reward.
        """
        material = env.backend.material
        reference = np.asarray(material.world_points, float)
        lattice = np.asarray(material.lattice)
        self._cuts = int(getattr(getattr(material, "material_config", None), "cuts", 2) or 2)
        plane = self.task.target_plane_world()
        signed = reference@plane[:3]+plane[3]
        lookup = {tuple(v): i for i, v in enumerate(lattice)}
        pairs = np.array([(i, lookup[tuple(v+step)]) for i, v in enumerate(lattice)
                          for step in np.eye(3, dtype=int) if tuple(v+step) in lookup])
        a, b = pairs.T
        middle = (signed[a]+signed[b])/2
        crossing = (np.abs(middle) <= self.band_m) & (np.abs(signed[a]-signed[b]) > .5*self.task.particle_spacing_m)
        # Columns along the target plane: a column counts once any of its bonds is severed.
        normal = plane[:3]
        tangent = np.cross(normal, [1., 0., 0.] if abs(normal[0]) < .9 else [0., 1., 0.])
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        midpoint = (reference[a]+reference[b])/2
        cells = np.round(np.c_[midpoint@tangent, midpoint@bitangent]/self.task.particle_spacing_m).astype(int)
        if self.task.goal_kind == "flatten":
            above = signed > self.band_m
            above_cells = {tuple(c) for c in np.round(np.c_[reference[above]@tangent, reference[above]@bitangent]
                                                      / self.task.particle_spacing_m).astype(int)}
            crossing &= np.array([tuple(c) in above_cells for c in cells], bool)
            off = middle < -self.band_m-self.task.particle_spacing_m
        else:
            off = np.abs(middle) > 3*self.task.grid_dx_m
        if not crossing.any():
            raise ValueError("Reward target region contains no reference bonds")
        self._a, self._b = a, b
        self._target_bonds = np.flatnonzero(crossing)
        _, self._column = np.unique(cells[crossing], axis=0, return_inverse=True)
        self._columns = int(self._column.max()+1)
        self._off_bonds = np.flatnonzero(off)
        self._bond_count = len(a)
        local = self.task.points_to_local(reference)
        self._box = (local.min(0)-self.margin_m, local.max(0)+self.margin_m)
        self._progress, self._off_target, self._edge = 0., 0., None

    def terms(self, env, info):
        from .cut_topology import conflicting
        if self._episode_key != id(env._history[0]):
            self._episode_key = id(env._history[0])
            self._start(env)
        codes = np.asarray(env._history[-1]["codes"])
        target = self._target_bonds
        severed = conflicting(codes[self._a[target]], codes[self._b[target]], 2)
        progress = float(len(np.unique(self._column[severed]))/self._columns)
        off = self._off_bonds
        off_target = float(conflicting(codes[self._a[off]], codes[self._b[off]], 2).sum()/self._bond_count) if len(off) else 0.
        edge = np.asarray(env._proprio["tool_pose"], float)
        gouge = lateral = 0.
        plane = self.task.target_plane_world()
        if self.task.goal_kind == "flatten":
            gouge = max(0., -(edge[:3]@plane[:3]+plane[3]))
        if self._edge is not None:
            local = self.task.points_to_local(edge[:3])
            if np.all(local >= self._box[0]) and np.all(local <= self._box[1]):
                normal = Rotation.from_quat(edge[3:]).apply([0., 0., 1.])
                lateral = abs(float((edge[:3]-self._edge[:3])@normal))
        report = info.get("action_report") or {}
        clamped = sum(bool(report.get(k)) for k in ("translation_clamped", "rotation_clamped", "workspace_clamped"))
        terms = {"progress": progress-self._progress, "off_target": off_target-self._off_target,
                 "gouge_m": gouge, "lateral_m": lateral, "ik_reject": float(info.get("ik_status") == "rejected"),
                 "clamped": float(clamped), "terminated": float(info.get("termination_reason") is not None),
                 "target_progress": progress}
        self._progress, self._off_target, self._edge = progress, off_target, edge
        return terms

    def __call__(self, env, info):
        t, w = self.terms(env, info), self.weights
        value = (w.progress*t["progress"]-w.off_target*max(t["off_target"], 0.)-w.gouge_per_m*t["gouge_m"]
                 - w.lateral_per_m*t["lateral_m"]-w.ik_reject*t["ik_reject"]-w.clamp*t["clamped"]
                 - w.time-w.termination*t["terminated"])
        info["reward_terms"] = t
        return float(value)
