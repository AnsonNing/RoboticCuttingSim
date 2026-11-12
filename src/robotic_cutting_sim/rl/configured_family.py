"""``ConfiguredTaskFamily``: configured-scene (``rcs_scene_v1`` + ``rcs_goal_v1``) task family for the batched vector env
(P1-5 T5.2).

A thin, declarative task family next to the seeded-TRIM ``TaskFamily``: one compiled ``ConfiguredTaskSpec`` per env
slot (``scene_config.compile_task`` of the slot's scene, goal and planner options, compile seed ``sim_seed``), plus
per-slot initial RNG seeds. The vector env packs the tasks into one ``BatchedCuttingMaterial`` and refuses what the
batch does not support, never falling back: a scene solver other than ``tlmpm_v7_2`` raises ``UNSUPPORTED_SOLVER``
(the frozen v5/v7/v7_1 CUT physics stays single-env / Native, P1-5 plan section 2), and slots whose timing, grid,
``max_cuts`` (K), cutting law or action limits differ raise ``UNSUPPORTED_NONUNIFORM_BATCH``. The shaping reward is
dispatched by goal type (``reward_sever.bonds_for_task``: sever/divide -> ``sever_bond_progress_packed_v1``,
TRIM -> ``trim_bond_progress_packed_v1``); success stays the frozen evaluator (``env.evaluate()``).

``ConfiguredTaskFamily.from_cut_workload(N)`` is the pre-declared ``cut_v7_2_v1`` workload (C2) in every slot.
Fidelity is computed against ``native_reference`` (for C2: the T5.8 Native tlmpm_v7_2 CUT episode); ideal tracking,
no robot and no reaction feedback always make a configured batch ``reduced_fidelity``.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

FAMILY_ID = "configured_scene_v1"
SUPPORTED_SOLVER = "tlmpm_v7_2"

# FULL_NATIVE_REFERENCE of C2 (cut_v7_2_v1): the P1-5 T5.8 Native episode (Isaac Sim 6.2, PhysX Franka, physical grasp,
# cameras; scene bar_flat_board_tlmpm_v7_2_v1; frozen evaluator PASS, 2 pieces 0.5 / 0.5). Evidence:
# docs/evidence/rl_ready/p1_5_cut_001.json; raw run artifacts/rl_ready/p1_5/native_cut_v7_2_001 (git 5d3f28d).
NATIVE_CUT_V7_2_REFERENCE = dict(
    id="native_cut_v7_2_reference", solver="tlmpm_v7_2", cutting_law=8, max_cuts=4, slots_per_node=8,
    particle_spacing_m=.001, grid_dx_m=.004, material_dt_s=1e-4, substeps_per_control_step=200, control_dt_s=.02,
    episode_steps=700, obstacle_boxes=3, obstacles="handle + two measured finger pads (updated every control step)",
    robot="Native Isaac Sim 6.2 PhysX Franka, physical grasp, IK tracking, 40 return-home steps after the task",
    reaction_fed_back=True, task_id="CFG-bar_flat_board-sever_mid-e5d82380",
    source="P1-5 T5.8 artifacts/rl_ready/p1_5/native_cut_v7_2_001 (evidence p1_5_cut_001)", status="MEASURED_PASS")
NATIVE_REFERENCE_NOT_RUN = dict(id="native_reference_not_run", status="NOT_RUN")


@dataclass(frozen=True)
class ConfiguredTaskFamily:
    """Per-slot configured scenes (``scenes[e]``, ``goals[e]``, ``planner_options[e]``; paths or dicts) and initial RNG
    seeds (``env_seeds``; default ``sim_seed``). ``particle_spacing_m`` / ``max_episode_steps`` other than the
    declared values are reduced-fidelity overrides (tests). ``initial_edge_poses`` defaults to each task's planner
    HOME (``reference_edge_pose_world(0)``); ``pad_boxes_edge`` with ``pad_recipients`` configures handle + two pads as
    the Native path (replay drive), else the handle box only."""
    scenes: tuple
    goals: tuple
    planner_options: tuple = None
    env_seeds: tuple | None = None
    sim_seed: int = 42
    particle_spacing_m: float | None = None
    max_episode_steps: int | None = None
    initial_edge_poses: tuple | None = None
    pad_boxes_edge: tuple | None = None
    pad_recipients: tuple | None = None
    material_dt_s: float = 1e-4
    family_id: str = FAMILY_ID
    workload_id: str | None = None
    native_reference: dict = field(default_factory=lambda: dict(NATIVE_REFERENCE_NOT_RUN))

    def __post_init__(self):
        n = len(self.scenes)
        if n < 1 or len(self.goals) != n:
            raise ValueError("scenes and goals must hold one entry per env (at least one)")
        object.__setattr__(self, "scenes", tuple(self.scenes))
        object.__setattr__(self, "goals", tuple(self.goals))
        options = (None,) * n if self.planner_options is None else tuple(self.planner_options)
        if len(options) != n:
            raise ValueError("planner_options must hold one entry per env")
        object.__setattr__(self, "planner_options", options)
        if self.env_seeds is not None:
            seeds = tuple(int(s) for s in self.env_seeds)
            if len(seeds) != n or any(s < 0 for s in seeds):
                raise ValueError("env_seeds must hold one nonnegative int per env")
            object.__setattr__(self, "env_seeds", seeds)
        if self.max_episode_steps is not None and (not isinstance(self.max_episode_steps, int)
                                                   or self.max_episode_steps < 1):
            raise ValueError("max_episode_steps must be a positive int or None")
        if (self.pad_boxes_edge is None) != (self.pad_recipients is None):
            raise ValueError("pad_boxes_edge and pad_recipients are given together")
        from ..scene_config import resolve_scene
        for e, scene in enumerate(self.scenes):
            module = resolve_scene(scene)["solver"]["module"]
            if module != SUPPORTED_SOLVER:
                from ..batched.errors import UnsupportedBatchConfiguration
                raise UnsupportedBatchConfiguration(
                    "UNSUPPORTED_SOLVER", f"env {e}: scene solver {module!r}; the batched material runs only "
                    f"{SUPPORTED_SOLVER!r} (frozen {module} stays single_env_headless / Native)")

    # -- the TaskFamily-compatible view used by the vector env
    @property
    def num_envs(self):
        return len(self.scenes)

    @property
    def scene_seeds(self):
        """Configured scenes carry no TRIM scene seed (receipts show None; ``reset(scene_seeds=)`` is refused)."""
        return (None,) * self.num_envs

    family_ids = None

    def family_of(self, e):
        return self.family_id

    def env_seed(self, e):
        return int(self.sim_seed) if self.env_seeds is None else self.env_seeds[e]

    def compile_tasks(self):
        """One ``ConfiguredTaskSpec`` per env (compiled once per distinct slot configuration)."""
        from ..scene_config import compile_task, resolve_scene
        cache, out = {}, []
        for scene, goal, options in zip(self.scenes, self.goals, self.planner_options):
            key = json.dumps([str(scene) if isinstance(scene, (str, Path)) else scene,
                              str(goal) if isinstance(goal, (str, Path)) else goal, options], sort_keys=True,
                             default=str)
            if key not in cache:
                spacing = {} if self.particle_spacing_m is None else {"particle_spacing_m": float(self.particle_spacing_m)}
                cache[key] = compile_task(resolve_scene(scene), goal, planner_options=options, seed=int(self.sim_seed),
                                          **spacing)
            out.append(cache[key])
        return out

    def to_dict(self):
        out = asdict(self)
        out["scenes"] = [str(s) if isinstance(s, (str, Path)) else s for s in self.scenes]
        out["goals"] = [str(g) if isinstance(g, (str, Path)) else g for g in self.goals]
        for key in ("initial_edge_poses", "pad_boxes_edge"):
            if out[key] is not None:
                out[key] = np.asarray(out[key], float).tolist()
        out["kind"] = "ConfiguredTaskFamily"
        return out

    # -- constructors
    @classmethod
    def uniform(cls, scene, goal, num_envs, *, planner_options=None, **kw):
        return cls(scenes=(scene,) * num_envs, goals=(goal,) * num_envs,
                   planner_options=(planner_options,) * num_envs, **kw)

    @classmethod
    def from_cut_workload(cls, num_envs, *, workload=None, env_seeds=None, **kw):
        """The pre-declared ``cut_v7_2_v1`` workload (``rl.cut_workload``, SHA256 pins verified) in every slot."""
        from .cut_workload import load_workload, workload_task_args
        w = workload or load_workload()
        args = workload_task_args(w)
        kw.setdefault("native_reference", dict(NATIVE_CUT_V7_2_REFERENCE))
        return cls.uniform(args["scene"], args["goal"], num_envs, planner_options=args["planner_options"],
                           sim_seed=args["seed"], env_seeds=env_seeds, workload_id=w["id"], **kw)


__all__ = ["ConfiguredTaskFamily", "NATIVE_CUT_V7_2_REFERENCE", "NATIVE_REFERENCE_NOT_RUN", "FAMILY_ID"]
