"""Provisional shaped reward for learning on the shared CUT/TRIM API.

``CuttingReward(task)(env, info)`` reads privileged simulator state (reference
particles, persistent cut codes, measured free-tool edge pose). It is a
learning signal only: it never enters observations, and task success remains
``env.evaluate()``. Terms and weights are engineering defaults, not calibrated.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = "cutting_reward_v0"


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 10.        # per unit increase of target-band separation fraction
    off_target: float = 2.       # per unit increase of labelled fraction outside the target band
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
                "privileged_inputs": ["reference particles", "cut codes", "measured tool edge pose"],
                "status": "PROVISIONAL_SHAPING_NOT_SUCCESS"}

    def _start(self, env):
        reference = np.asarray(env.backend.material.world_points, float)
        plane = self.task.target_plane_world()
        self._signed = reference@plane[:3]+plane[3]
        local = self.task.points_to_local(reference)
        self._box = (local.min(0)-self.margin_m, local.max(0)+self.margin_m)
        if self.task.goal_kind == "flatten":
            self._target = self._signed > self.band_m
        else:
            self._target = np.abs(self._signed) <= self.band_m
        if not self._target.any():
            raise ValueError("Reward target region contains no reference particles")
        self._progress, self._off_target, self._edge = 0., 0., None

    def terms(self, env, info):
        if self._episode_key != id(env._history[0]):
            self._episode_key = id(env._history[0])
            self._start(env)
        codes = np.asarray(env._history[-1]["codes"])
        labelled = codes != 0
        progress = float(labelled[self._target].mean())
        if self.task.goal_kind == "flatten":
            off_target = float((labelled & (self._signed < -self.band_m)).mean())
        else:
            off_target = float((labelled & ~self._target & (np.abs(self._signed) > 3*self.task.grid_dx_m)).mean())
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
