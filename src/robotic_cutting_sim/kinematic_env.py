"""Ideal-tracking kinematic backend: the knife edge goes exactly where commanded.

No robot, IK, grasp, renderer or Isaac runtime is involved. The measured edge
pose equals the commanded edge pose and drives the same LiveCuttingMaterial as
the live runtimes, so CuttingTaskEnv, the Gym adapter, reward, policies and the
agent API run end to end on CPU or GPU. Use it for pipeline validation, fast RL
pre-training and cutting-law probes. It says nothing about reachability,
grasp, slip, robot dynamics or cameras; results must be labelled
``runtime_source = kinematic_ideal_tracking`` and confirmed on a live runtime.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .coupled_robot import CouplingConfig
from .live_material import LiveCuttingMaterial
from .scene_frames import compose_pose, invert_transform, matrix_pose, pose_matrix
from .tasks import ACTION_MODES, ProfileTaskSpec, make_task


@dataclass(frozen=True)
class KinematicConfig:
    task_id: str = "CUT-01-R"
    seed: int = 42
    particle_spacing_m: float | None = None   # None = protocol spacing (1 mm)
    material_dt: float = .0001                 # explicit, recorded; coarse probes may raise it
    device: str = "cpu"
    cutting_law: int | None = None             # None = task default (v1 tasks 1, v2 tasks 2); 3-8 opt-in
    max_cuts: int = 1
    slice_efficiency: float = 1.
    scenario: str | None = None
    material_profile_id: str | None = None
    solver: str | None = None                  # None = v1-v4 by law; 'tlmpm_v5' opt-in (ADR 0014)
    slots_per_node: int | None = None          # tlmpm_v5 only (None = 8)

    def __post_init__(self):
        if self.solver not in (None, "tlmpm_v5"):
            raise ValueError("solver is None or 'tlmpm_v5'")
        if self.slots_per_node is not None and self.solver != "tlmpm_v5":
            raise ValueError("slots_per_node applies to solver='tlmpm_v5' only (it was ignored without it)")
        allowed = range(1, 16) if self.solver == "tlmpm_v5" else (1, 2)
        if self.max_cuts not in allowed or self.material_dt <= 0 or not np.isfinite(self.material_dt):
            raise ValueError("max_cuts is 1 or 2 (1..15 with solver='tlmpm_v5') and material_dt must be positive")


class KinematicCuttingBackend:
    """Backend contract of CuttingTaskEnv (Lab-style ``measured``/``frames``)."""

    runtime_source = "kinematic_ideal_tracking"

    def __init__(self, config=KinematicConfig(), *, task=None):
        self.config = config
        self.options = config  # CuttingTaskEnv reads tool_attachment_mode from here
        # Optional compiled scene_config.ConfiguredTaskSpec (ADR 0015); None keeps the registered task ids.
        self.configured_task = task
        self.task = self._task(config.seed)
        self.material, self.edge, self.time_s = None, None, 0.
        self.frames, self.calibration = {}, {}
        self.last_diagnostics = {}
        self.stroke_history = []  # law 4 only: per-control-step stroke state (oracle)
        self._surface, self._surface_time = None, None

    @property
    def tool_attachment_mode(self):
        return "kinematic_ideal_tracking"

    def _task(self, seed):
        c = self.config
        from .task_scenarios import configure_task
        if self.configured_task is not None:
            if c.scenario is not None or (c.particle_spacing_m is not None
                                          and c.particle_spacing_m != self.configured_task.particle_spacing_m):
                raise ValueError("A configured task declares its supports and spacing in the scene/compile_task; "
                                 "do not also pass a scenario or another particle spacing")
            from dataclasses import replace
            return configure_task(replace(self.configured_task, seed=int(seed)), material_profile_id=c.material_profile_id)
        task = make_task(c.task_id, seed=int(seed), particle_spacing_m=c.particle_spacing_m)
        return configure_task(task, scenario=c.scenario, material_profile_id=c.material_profile_id)

    def reset(self, seed=42):
        c = self.config
        self.task = self._task(seed)
        coupling = CouplingConfig(control_dt=self.task.control_dt_s, material_dt=c.material_dt,
                                  dx=self.task.grid_dx_m, device=c.device)
        kwargs = {"cutting_law": c.cutting_law, "max_cuts": c.max_cuts, "slice_efficiency": c.slice_efficiency}
        if c.solver is not None:
            kwargs.update(solver=c.solver, slots_per_node=c.slots_per_node)
        if not isinstance(self.task, ProfileTaskSpec):
            if c.max_cuts == 1 and c.cutting_law in (None, 1) and c.device != "cpu" and c.solver is None:
                kwargs = {}  # v1 tasks keep the frozen v1 solver by default
            else:
                # The retained wedge on tlmpm_v2; with law 1 this equals frozen v1
                # bit for bit (tests/test_tool_profile.py) and also compiles for CPU.
                kwargs["tool_profile"] = self.task.tool_profile
        self.material = LiveCuttingMaterial(coupling, task=self.task, **kwargs)
        self.edge = np.asarray(self.task.reference_edge_pose_world(0.), float)
        self.time_s = 0.
        receipt = self.material.reset(self.edge, seed=int(seed))
        self.last_diagnostics = {"material_reset": receipt}
        self.stroke_history = []
        self._surface = None
        return receipt

    def measured(self):
        hand = matrix_pose(pose_matrix(self.edge)@invert_transform(pose_matrix(self.task.hand_from_edge_xyz_xyzw)))
        return {"timestamp_s": self.time_s, "q": np.zeros((2, 7)), "qdot": np.zeros((2, 7)),
                "tool_pose": self.edge.copy(), "hand_pose": hand,
                "tool_attachment_mode": self.tool_attachment_mode}

    def set_gripper(self, command):
        raise ValueError("The kinematic backend has no gripper; use a live runtime for grasp tests")

    # Recorder compatibility (WP-11 Phase G): the only moving "body" is the ideal knife edge.
    rigid_link_paths = ("/kinematic/knife_edge",)

    def body_poses(self):
        """(1, 7) WORLD pose of the ideal knife edge (the recorder's rigid-body track); no robot links exist."""
        return np.asarray(self.edge, float).reshape(1, 7).copy()

    def step(self, action=None):
        if self.material is None:
            raise RuntimeError("Reset the kinematic backend first")
        next_time = round(self.time_s+self.task.control_dt_s, 9)
        if next_time > self.task.duration_s+1e-9:
            raise RuntimeError("Episode ended; call reset")
        if action is None:
            target, cutting, cut_index = self.task.reference_edge_pose_world(next_time), True, None
        else:
            # cut_index is bounded by the retained material history (up to 15 digits with tlmpm_v5).
            command = self.task.validate_action(action, max_cuts=int(self.material.max_cuts))
            if command["mode"] != ACTION_MODES[0]:
                raise ValueError("The kinematic backend accepts Cartesian edge actions only (no arm joints)")
            target, cutting = command["target"], command["cut_enabled"]
            cut_index = command.get("cut_index", 0)
        self.edge = np.asarray(target, float).copy()
        diagnostics = self.material.advance(self.edge, next_time, cutting=cutting, cut_index=cut_index)
        self.time_s = next_time
        if self.material.cutting_law in (4, 5, 6, 7, 8):
            # Oracle-only record (laws 5-8 share the law-4 stroke state; law 6 adds pending counts) of when the
            # edge was engaged (never an observation).
            self.stroke_history.append({"time_s": next_time, "cut_index": diagnostics["cut_index"],
                                        **self.material.bed.cutting_state(count_labels=True)})
        self.last_diagnostics = {"material": diagnostics, "commanded_edge_pose": self.edge.copy(),
                                 "measured_edge_pose": self.edge.copy(), "tracking_error_m": 0.}
        return self.measured(), self.last_diagnostics

    def _extract_surface(self):
        if self._surface_time != self.time_s:
            from .cutting_surface import ComponentSurface
            spacing = self.material.spacing
            extractor = ComponentSurface(spacing, self.config.device, voxel_size=spacing/2, profile="current_volume_v1")
            vertices, triangles, metadata = extractor.extract(self.material.bed)
            vertices = self.task.points_to_world(vertices)
            self._surface, self._surface_time = (vertices, triangles, metadata), self.time_s
        return self._surface

    @property
    def surface(self):
        vertices, triangles, _ = self._extract_surface()
        return vertices, triangles

    @property
    def surface_metadata(self):
        return self._extract_surface()[2]

    def law4_stroke_arrays(self):
        """Per-control-step law-4..8 stroke record as arrays for physics_state.npz ({} otherwise).

        Law 6 adds ``law6_pending_count``, ``law6_pending_unassigned_count`` and ``law6_committed_count``."""
        rows = self.stroke_history
        if not rows:
            return {}
        return {"law4_time_s": np.asarray([r["time_s"] for r in rows], float),
                "law4_stroke": np.asarray([r["stroke"] for r in rows], np.int32),
                "law4_travel_m": np.asarray([r["travel_m"] for r in rows], float),
                "law4_engaged": np.asarray([r["engaged"] for r in rows], np.int32),
                "law4_any_motion": np.asarray([r["any_motion"] for r in rows], np.int32),
                "law4_cut_index": np.asarray([r["cut_index"] for r in rows], np.int32),
                "law4_labelled_count": np.asarray([r["labelled_count"] for r in rows], np.int64),
                **({f"law6_{key}": np.asarray([r[key] for r in rows], np.int64)
                    for key in ("pending_count", "pending_unassigned_count", "committed_count")}
                   if "pending_count" in rows[0] else {})}

    def close(self):
        self.material = None

    def receipt(self):
        return {"runtime_source": self.runtime_source, "config": asdict(self.config),
                "scope": "Ideal edge tracking; no robot, IK, grasp, slip, dynamics or cameras",
                "material": self.material.parameter_receipt if self.material else None}


def make_kinematic_env(task_id="CUT-01-R", *, seed=42, particle_spacing_m=None, material_dt=.0001,
                       device="cpu", cutting_law=None, max_cuts=1, slice_efficiency=1.,
                       observation_profile="state", scenario=None, material_profile_id=None, solver=None,
                       slots_per_node=None, **wrapper_options):
    """CuttingTaskEnv over the kinematic backend (state observations only).

    solver='tlmpm_v5' (opt-in, ADR 0014) passes through to LiveCuttingMaterial."""
    from .live_env import CuttingTaskEnv
    if observation_profile != "state":
        raise ValueError("The kinematic backend renders no cameras; use observation_profile='state'")
    task = wrapper_options.pop("task", None)  # optional compiled ConfiguredTaskSpec (ADR 0015)
    if task is not None:
        task_id = task.task_id
    backend = KinematicCuttingBackend(KinematicConfig(task_id, seed, particle_spacing_m, material_dt, device,
                                                      cutting_law, max_cuts, slice_efficiency, scenario, material_profile_id,
                                                      solver, slots_per_node), task=task)
    env = CuttingTaskEnv(backend, seed=seed, observation_profile="state", **wrapper_options)
    return env
