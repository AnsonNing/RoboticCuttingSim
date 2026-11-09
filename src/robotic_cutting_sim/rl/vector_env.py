"""Batched vector environment ``open_vector_env`` for runtime ``rl_gpu_headless`` (ADR 0021 §8, P1-2).

One process, one ``BatchedCuttingMaterial`` (batched v7_2 kernels, one CUDA graph per batch shape) and
batched Torch action/observation/reward code over N independent seeded TRIM envs (workload A: ideal edge
tracking, no robot). Nothing here is a Python loop over single environments.

```python
with open_vector_env(num_envs=N, task=TaskFamily(scene_seeds=[512]*N), device="cuda") as env:
    obs, info = env.reset(seeds=seeds_int64_N)
    obs, reward, terminated, truncated, info = env.step(actions)      # torch.float32 [N, 8] on the device
    obs, info = env.reset(env_ids=ids, seeds=seeds_for_ids)            # masked partial reset (others untouched)
    results = env.evaluate(env_ids)                                    # frozen seeded_trim evaluator, host
```

Step (all envs, device tensors; no per-env host loop, no per-env host sync):

1. ``edge_delta_v1`` (``rl.actions.EdgeDeltaActionV1``) maps ``actions [N,8]`` and the float64 commanded edge
   pose ``[N,7]`` to the new commanded pose, cut digit and status;
2. ideal tracking: the measured edge pose IS the commanded pose; it and the cut digit are copied into the
   material's device input buffers (``measured_pose``, ``cut_input``);
3. ``BatchedCuttingMaterial.advance(interpolation="device")``: device pose interpolation, one graph launch of
   200 substeps for all envs, wrench handoff, segmented diagnostics, validity flags; in ``strict`` mode one
   int32 validity count is read back (the only host sync of a step) and ``InvalidBatchEnvs`` is raised;
4. ``trim_bond_progress_packed_v1`` reward from the device code array; ``truncated = step_count >= max_steps``,
   ``terminated = invalid_reason != 0`` (training mode, ``strict=False``);
5. ``rl_state_obs_v1`` observation (``"policy"`` deployable, ``"critic"`` only with ``privileged_state``);
6. ``autoreset="same_step"``: done envs are reset by masked device kernels in the same call
   (``info["final_obs"]`` holds the pre-reset observation); the default ``autoreset=False`` never resets.

Privileged replay entry ``step_measured_edge(edge_pose, cut_index, pad_boxes=None, env_mask=None,
actions=None)`` drives selected envs with recorded measured edge poses (``drive: measured_trajectory_replay``,
``info["drive"] == 1``) and the others with actions in the same batched call.

Stream policy (ADR 0021 §7): every Warp launch of a public call runs on Torch's current CUDA stream
(``wp.stream_from_torch``, entered with ``wp.ScopedStream``), so Torch and Warp work of the env is totally
ordered on one stream (``last_warp_stream_handle`` records the stream of the last call; ``debug_sync=True``
synchronizes the device after every public call, a debug mode for suspected stream races). The CUDA graph is captured once at construction on Warp's own stream (a capture cannot
use the legacy default stream) and replayed on the Torch stream.

Success comes only from ``evaluate()`` (the frozen ``goals.evaluate_goal`` of the compiled task) on a host
snapshot at episode end, never from the shaping reward. Fidelity against the Native TRIM reference is computed
into ``receipt()["fidelity"]``; workload A is always ``reduced_fidelity`` (ideal tracking, no robot dynamics,
reaction not fed back, plus pad/resolution/episode differences when present).
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from ..batched.errors import CODES as BATCH_CODES, UnsupportedBatchConfiguration

VECTOR_ENV_VERSION = "rl_vector_env_v1"
RUNTIME = "rl_gpu_headless"
BACKEND = "kinematic_ideal_tracking_batched"
ACTION_SPEC = "edge_delta_v1"
REWARD = "trim_bond_progress_packed_v1"
MATERIAL = "scene_declared"
OBSERVATION_PROFILES = ("state", "privileged_state")
AUTORESET_MODES = (False, "same_step")
TASK_FAMILIES = {"agentic_trim_id": "id", "agentic_trim_ood": "ood"}
DRIVE_ACTION, DRIVE_REPLAY = 0, 1
DRIVE_NAMES = {DRIVE_ACTION: "policy_action_edge_delta_v1", DRIVE_REPLAY: "measured_trajectory_replay"}
STATUS_REPLAY = -1   # action_status of an env driven by a measured trajectory (no action was applied)
DEFAULT_MATERIAL_DT_S = 1e-4

# The reference the fidelity is computed against (ADR 0021 §8): WP-17 Native TRIM configuration.
NATIVE_TRIM_V7_2_REFERENCE = dict(
    id="native_trim_v7_2_reference", solver="tlmpm_v7_2", cutting_law=8, max_cuts=2, slots_per_node=8,
    particle_spacing_m=.001, grid_dx_m=.004, material_dt_s=1e-4, substeps_per_control_step=200,
    control_dt_s=.02, episode_steps=1200, obstacle_boxes=3,
    obstacles="handle + two measured finger pads (updated every control step)",
    robot="Native Isaac Sim 6.2 PhysX Franka, physical grasp, IK tracking", reaction_fed_back=True,
    source="WP-17 native_optimized_001 (seeded_trim rcs_trim_random_sim_v1, seeds 512/15)")

VECTOR_CODES = BATCH_CODES + ("UNSUPPORTED_RUNTIME", "UNSUPPORTED_OPTION", "UNSUPPORTED_TASK",
                              "UNSUPPORTED_DEVICE")


class UnsupportedVectorConfiguration(UnsupportedBatchConfiguration):
    """A vector-env configuration that is not implemented (plan §1); never a silent fallback."""

    def __init__(self, code, detail=""):
        if code not in VECTOR_CODES:
            raise ValueError(f"unknown UNSUPPORTED code {code!r}")
        self.code, self.detail = code, str(detail)
        message = code if not detail else f"{code}: {detail}"
        ValueError.__init__(self, message)


# ----------------------------------------------------------------------------- task family

@dataclass(frozen=True)
class TaskFamily:
    """Seeded TRIM scenes, one scene seed per env slot (fixed for the batch lifetime, ADR §6).

    ``particle_spacing_m`` None keeps the scene-declared 1 mm; any other value is a coarse, reduced-fidelity
    task (tests). ``max_episode_steps`` None is ``duration / control_dt`` (1200 for TRIM); a shorter value is
    reduced fidelity. ``initial_edge_poses`` (per env, WORLD XYZ+XYZW) defaults to
    ``task.reference_edge_pose_world(0)`` as ``kinematic_env``. ``pad_boxes_edge`` (per env ``[3,2,3]``
    edge-frame handle + two finger pads) with ``pad_recipients`` ``(None, left, right)`` configures the
    measured-pad obstacles of the Native path; None keeps the handle box only (ADR workload A)."""
    scene_seeds: tuple
    family_id: str = "agentic_trim_id"
    sim_seed: int = 42
    particle_spacing_m: float | None = None
    max_episode_steps: int | None = None
    initial_edge_poses: tuple | None = None
    pad_boxes_edge: tuple | None = None
    pad_recipients: tuple | None = None
    material_dt_s: float = DEFAULT_MATERIAL_DT_S

    def __post_init__(self):
        seeds = tuple(int(s) for s in self.scene_seeds)
        if not seeds or any(s < 0 for s in seeds):
            raise ValueError("scene_seeds must be a nonempty sequence of nonnegative ints")
        object.__setattr__(self, "scene_seeds", seeds)
        if self.family_id not in TASK_FAMILIES:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_TASK", f"family {self.family_id!r}; "
                                                 f"supported {sorted(TASK_FAMILIES)}")
        if self.max_episode_steps is not None and (not isinstance(self.max_episode_steps, int)
                                                   or self.max_episode_steps < 1):
            raise ValueError("max_episode_steps must be a positive int or None")
        if (self.pad_boxes_edge is None) != (self.pad_recipients is None):
            raise ValueError("pad_boxes_edge and pad_recipients are given together")

    def to_dict(self):
        out = asdict(self)
        for key in ("initial_edge_poses", "pad_boxes_edge"):
            if out[key] is not None:
                out[key] = np.asarray(out[key], float).tolist()
        return out


def _compile(family, seed):
    from ..scene_config import compile_task
    from ..seeded_trim import generate_scene, goal_config
    options = {} if family.particle_spacing_m is None else {"particle_spacing_m": float(family.particle_spacing_m)}
    return compile_task(generate_scene(seed, distribution=TASK_FAMILIES[family.family_id]), goal_config(),
                        seed=int(family.sim_seed), **options)


# ----------------------------------------------------------------------------- helpers

_SM = [c - (1 << 64) if c >= 1 << 63 else c for c in (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB)]


def splitmix64(x):
    """SplitMix64 finalizer on int64 tensors (two's-complement wrap; logical shifts emulated by masks).

    Counter-based and per-element, so it is identical on CPU and CUDA and independent of the batch size."""
    def lsr(v, s):
        return (v >> s) & ((1 << (64 - s)) - 1)
    z = x + _SM[0]
    z = (z ^ lsr(z, 30)) * _SM[1]
    z = (z ^ lsr(z, 27)) * _SM[2]
    return z ^ lsr(z, 31)


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _box_space(low, high, shape):
    try:
        from gymnasium import spaces
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    except ImportError:   # documented minimal stand-in (no sampling/contains semantics beyond shape/bounds)
        return MinimalBox(low, high, shape)


@dataclass(frozen=True)
class MinimalBox:
    """Used only when gymnasium is not importable: shape, dtype and bounds of a float32 box."""
    low: float
    high: float
    shape: tuple
    dtype: type = np.float32


# ----------------------------------------------------------------------------- environment

class VectorCuttingEnv:
    """N seeded TRIM envs in one batched material (see the module docstring). Use ``open_vector_env``."""

    metadata = {"render_modes": [], "runtime": RUNTIME, "backend": BACKEND, "version": VECTOR_ENV_VERSION}

    def __init__(self, num_envs, *, runtime=RUNTIME, task=None, material=MATERIAL, observation_profile="state",
                 device="cuda", backend=BACKEND, action_spec=ACTION_SPEC, reward=REWARD, autoreset=False,
                 strict=True, output_dir=None, parity_mode=False, reward_config=None, cameras=None, sensors=None,
                 recorders=None, debug_sync=False):
        t0 = time.perf_counter()
        import torch
        import warp as wp
        self.torch, self.wp = torch, wp
        if runtime != RUNTIME:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_RUNTIME", f"{runtime!r}; only {RUNTIME!r}")
        if backend != BACKEND:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_BACKEND", f"{backend!r}; only {BACKEND!r} "
                                                 "(isaac_lab_newton is P1-1b)")
        if material != MATERIAL:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"material={material!r}; only {MATERIAL!r}")
        if observation_profile not in OBSERVATION_PROFILES:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_SENSOR" if "vision" in str(observation_profile)
                                                 or "camera" in str(observation_profile) else "UNSUPPORTED_OPTION",
                                                 f"observation_profile={observation_profile!r}; "
                                                 f"supported {OBSERVATION_PROFILES}")
        if action_spec != ACTION_SPEC:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"action_spec={action_spec!r}")
        if reward != REWARD:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"reward={reward!r}")
        if not any(autoreset is mode or autoreset == mode for mode in AUTORESET_MODES) or autoreset is True:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"autoreset={autoreset!r}; "
                                                 f"supported {AUTORESET_MODES}")
        if cameras or sensors:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_SENSOR", f"cameras={cameras!r} sensors={sensors!r}")
        if recorders:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_RECORDER", f"recorders={recorders!r}")
        if not isinstance(task, TaskFamily):
            raise UnsupportedVectorConfiguration("UNSUPPORTED_TASK", "task must be a TaskFamily of seeded TRIM scenes")
        if not isinstance(num_envs, int) or num_envs < 1 or len(task.scene_seeds) != num_envs:
            raise ValueError("num_envs must be a positive int equal to len(task.scene_seeds)")
        self.torch_device = torch.device(device)
        if self.torch_device.type == "cuda" and self.torch_device.index is None:
            self.torch_device = torch.device("cuda", torch.cuda.current_device())
        if self.torch_device.type == "cpu" and not parity_mode:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_CPU_WITHOUT_PARITY_MODE",
                                                 "device='cpu' is for tests (parity_mode=True)")
        if self.torch_device.type not in ("cpu", "cuda"):
            raise UnsupportedVectorConfiguration("UNSUPPORTED_DEVICE", str(device))
        self.wp_device = "cpu" if self.torch_device.type == "cpu" else f"cuda:{self.torch_device.index}"
        self._cuda = self.torch_device.type == "cuda"
        self.num_envs, self.task_family = num_envs, task
        self.observation_profile, self.autoreset, self.strict = observation_profile, autoreset, bool(strict)
        self.parity_mode = bool(parity_mode)
        self.debug_sync = bool(debug_sync)   # ADR §14 risk 4: synchronize the device after every public call
        self.last_warp_stream_handle = None  # Warp stream used by the last public call (stream-policy test hook)
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None:
            if self.output_dir.exists() and any(self.output_dir.iterdir()):
                raise FileExistsError(f"output_dir {self.output_dir} exists and is not empty (never overwritten)")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._streams = {}
        self._closed = False
        self._build(reward_config)
        self.build_wall_s = time.perf_counter() - t0
        if self.output_dir is not None:
            (self.output_dir / "vector_env_receipt_open.json").write_text(
                json.dumps(self.receipt(), indent=1, sort_keys=True, default=_json_default) + "\n")

    # ------------------------------------------------------------------ construction

    def _build(self, reward_config):
        torch, wp = self.torch, self.wp
        from ..batched import layout
        from ..batched.material_batch import BatchedCuttingMaterial
        from ..coupled_robot import CouplingConfig
        from ..live_material import LiveCuttingMaterial
        from .actions import EdgeDeltaActionV1
        from .observations import StateObservation
        from .reward_packed import TrimBondProgressReward, TrimBondRewardConfig, bonds_from_task, pack_bonds

        fam, N = self.task_family, self.num_envs
        tasks = {}
        for seed in sorted(set(fam.scene_seeds)):
            tasks[seed] = _compile(fam, seed)
        self.tasks = [tasks[s] for s in fam.scene_seeds]
        first = self.tasks[0]
        for task in self.tasks:
            if (task.control_dt_s, task.duration_s, task.grid_dx_m, task.max_cuts, task.cutting_law,
                    task.solver_module) != (first.control_dt_s, first.duration_s, first.grid_dx_m, first.max_cuts,
                                            first.cutting_law, first.solver_module):
                raise UnsupportedVectorConfiguration("UNSUPPORTED_NONUNIFORM_BATCH", "task timing/solver differs")
        self.control_dt = float(first.control_dt_s)
        declared_steps = int(round(first.duration_s / self.control_dt))
        self.declared_episode_steps = declared_steps
        self.max_episode_steps = declared_steps if fam.max_episode_steps is None else int(fam.max_episode_steps)
        if fam.initial_edge_poses is None:
            initial = np.stack([np.asarray(t.reference_edge_pose_world(0.), float) for t in self.tasks])
        else:
            initial = np.asarray(fam.initial_edge_poses, float)
            if initial.shape != (N, 7):
                raise ValueError(f"initial_edge_poses must be [{N},7]")
        pads = None if fam.pad_boxes_edge is None else np.asarray(fam.pad_boxes_edge, np.float32)
        if pads is not None and pads.shape != (N, 3, 2, 3):
            raise ValueError(f"pad_boxes_edge must be [{N},3,2,3]")
        limits = [t.action_limits() for t in self.tasks]
        if any(l != limits[0] for l in limits):
            raise UnsupportedVectorConfiguration("UNSUPPORTED_NONUNIFORM_BATCH", "action limits differ")
        self.action_limits = limits[0]

        specs, self._env_host = [], []
        particles_cache = {}
        bonds = []
        for e, task in enumerate(self.tasks):
            config = CouplingConfig(control_dt=self.control_dt, material_dt=float(fam.material_dt_s),
                                    dx=float(task.grid_dx_m), device="cpu", feedback=True)
            m = LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                                    solver=task.solver_module, v7_2_optimizations=True)
            if pads is not None:
                m.configure_pad_obstacles(pads[e], list(fam.pad_recipients))
            specs.append(layout.EnvSpec(m, initial[e], int(fam.sim_seed), task.config_sha256))
        self.material = BatchedCuttingMaterial(specs, device=self.wp_device, parity_mode=self.parity_mode,
                                               strict=self.strict)
        if self._cuda:
            wp.synchronize_device(self.wp_device)
        pk = self.material.packed
        for e, (spec, task) in enumerate(zip(specs, self.tasks)):
            m = spec.material
            key = task.config_sha256
            if key not in particles_cache:
                particles_cache[key] = (np.asarray(m.points), np.asarray(m.masses), np.asarray(m.lattice))
            points, masses, lattice = particles_cache[key]
            self._env_host.append(dict(task=task, scene_seed=fam.scene_seeds[e], scene_key=key,
                                       T_world_local=np.asarray(m.T_world_local, float), points=points,
                                       masses=masses, lattice=lattice))
            bonds.append(bonds_from_task(task, particles=(points, masses, lattice)))
        del specs   # frozen CPU single solvers are no longer needed (their arrays are packed)
        T_world = np.stack([np.asarray(t.T_world_workpiece, float) for t in self.tasks])
        if not np.allclose(T_world, pk.env["T_world_local"], atol=1e-12, rtol=0):
            raise ValueError("task.T_world_workpiece differs from the material T_world_local")

        dev = self.torch_device
        with self._stream():
            self.actions = EdgeDeltaActionV1(self.action_limits, T_world, first.max_cuts, device=dev)
            self.reward = TrimBondProgressReward(pack_bonds(bonds, pk.particle_offset),
                                                 reward_config or TrimBondRewardConfig(), device=dev)
            self.observation = StateObservation(N, max_steps=self.max_episode_steps, control_dt=self.control_dt,
                                                profile=self.observation_profile, device=dev)
            mt, s = self.material, self.material.solver
            view = wp.to_torch
            # zero-copy Torch views of the material state the env reads or writes
            self._measured = view(mt.measured_pose)              # [N,7] float64
            self._initial_pose = view(mt.initial_measured_pose)  # [N,7] float64 (constant)
            self._cut_input = view(mt.cut_input)                 # [N] int32
            self._step_count = view(mt.step_count)               # [N] int64
            self._invalid = view(mt.invalid_reason)              # [N] int32
            self._seed = view(mt.seed)                           # [N] int64
            self._reset_mask = view(mt.reset_mask)               # [N] int32
            self._reset_seed = view(mt.reset_seed)               # [N] int64
            self._pending_wrench = view(mt.pending_wrench)       # [N,6] float64
            self._codes = view(s.code)                           # [P] int32
            self._box_centers = view(s.box_centers).view(N, s.boxes, 3)
            self._box_halves = view(s.box_halves).view(N, s.boxes, 3)
            self.commanded_pose = self._initial_pose.clone()
            self.previous_action = torch.zeros((N, 8), dtype=torch.float32, device=dev)
            self.armed = torch.zeros(N, dtype=torch.bool, device=dev)
            self.episode = torch.zeros(N, dtype=torch.int64, device=dev)
            self.drive = torch.zeros(N, dtype=torch.int8, device=dev)
            self.action_steps = torch.zeros(N, dtype=torch.int64, device=dev)
            self.replay_steps = torch.zeros(N, dtype=torch.int64, device=dev)
            self.replay_pad_steps = torch.zeros(N, dtype=torch.int64, device=dev)
            self.reward.reset(self._codes)
        self.boxes = int(s.boxes)
        self.pads_configured = pads is not None
        self.single_action_space = _box_space(-1., 1., (8,))
        self.action_space = _box_space(-1., 1., (N, 8))
        self.single_observation_space = self._obs_space(None)
        self.observation_space = self._obs_space(N)
        self.metadata = {**type(self).metadata, "autoreset_mode": "same_step" if self.autoreset else "disabled",
                         "observation_profile": self.observation_profile, "max_episode_steps": self.max_episode_steps}

    def _obs_space(self, n):
        shape = (lambda d: (d,)) if n is None else (lambda d: (n, d))
        policy = _box_space(-np.inf, np.inf, shape(self.observation.policy_dim))
        if self.observation_profile != "privileged_state":
            try:
                from gymnasium import spaces
                return spaces.Dict({"policy": policy})
            except ImportError:
                return {"policy": policy}
        critic = _box_space(-np.inf, np.inf, shape(self.observation.critic_dim))
        try:
            from gymnasium import spaces
            return spaces.Dict({"policy": policy, "critic": critic})
        except ImportError:
            return {"policy": policy, "critic": critic}

    @contextmanager
    def _stream(self):
        """Run Warp work on Torch's current stream (ADR §7 single-stream policy)."""
        if not self._cuda:
            yield
            return
        ts = self.torch.cuda.current_stream(self.torch_device)
        ws = self._streams.get(ts.cuda_stream)
        if ws is None:
            ws = self._streams[ts.cuda_stream] = self.wp.stream_from_torch(ts)
        with self.wp.ScopedStream(ws):
            self.last_warp_stream_handle = self.wp.get_stream(self.wp_device).cuda_stream
            yield
            if self.debug_sync:
                self.wp.synchronize_device(self.wp_device)
                self.torch.cuda.synchronize(self.torch_device)

    # ------------------------------------------------------------------ observations

    def _privileged(self):
        fractions = self.reward.fractions(self.reward.counts)
        return {"target_bond_progress_fraction": fractions["target_bond_progress_fraction"],
                "off_target_bond_fraction": fractions["off_target_bond_fraction"],
                "material_wrench_world": self._pending_wrench,
                "invalid_flag": (self._invalid != 0).to(self.torch.float32)}

    def _observe(self):
        privileged = self._privileged() if self.observation_profile == "privileged_state" else None
        return self.observation.compute(commanded_pose=self.commanded_pose, step_count=self._step_count,
                                        previous_action=self.previous_action, armed=self.armed,
                                        privileged=privileged)

    def _reset_info(self):
        return {"step_count": self._step_count.clone(), "seed": self._seed.clone(), "episode": self.episode.clone(),
                "drive": self.drive.clone()}

    # ------------------------------------------------------------------ reset

    def reset(self, *, seeds=None, env_ids=None, scene_seeds=None, options=None, seed=None):
        """Full (``env_ids=None``) or masked partial reset; returns ``(obs, info)`` for all envs.

        ``seeds`` (int64, one per reset env) replace the env RNG seeds (None keeps them); ``seed`` (Gymnasium
        style, full reset only) sets ``seeds = seed + arange(N)``. ``scene_seeds`` must equal the slots' scene
        seeds: a geometry change raises ``UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET`` (use a new env).
        Non-reset envs are untouched (bitwise on CPU)."""
        self._check_open()
        torch, N = self.torch, self.num_envs
        if options:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"reset options {sorted(options)}")
        if seed is not None:
            if env_ids is not None or seeds is not None:
                raise ValueError("seed= is the full-reset Gymnasium form; use seeds= with env_ids")
            seeds = int(seed) + np.arange(N, dtype=np.int64)
        if env_ids is None:
            ids = np.arange(N)
        else:
            ids = np.asarray(env_ids.detach().cpu() if hasattr(env_ids, "detach") else env_ids,
                             np.int64).reshape(-1)
            if len(np.unique(ids)) != len(ids):
                raise ValueError("env_ids must be unique")
        if ids.size and (ids.min() < 0 or ids.max() >= N):
            raise ValueError(f"env_ids must be in [0, {N})")
        if seeds is not None:
            seeds = np.asarray(seeds.detach().cpu() if hasattr(seeds, "detach") else seeds, np.int64).reshape(-1)
            if len(seeds) != len(ids):
                raise ValueError("one seed per reset env")
        if scene_seeds is not None:
            scene_seeds = [int(s) for s in np.asarray(scene_seeds).reshape(-1)]
            if len(scene_seeds) != len(ids):
                raise ValueError("one scene seed per reset env")
            for e, s in zip(ids, scene_seeds):
                if s != self.task_family.scene_seeds[e]:
                    raise UnsupportedVectorConfiguration(
                        "UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET",
                        f"env {int(e)} holds scene seed {self.task_family.scene_seeds[e]}, requested {s}; the batch "
                        "keeps each slot's geometry (open a new env for other scenes)")
        if ids.size == 0:
            with self._stream():
                return self._observe(), self._reset_info()
        from ..batched import material_batch as MB
        if env_ids is None and self.material.device_hashgrid_guard == MB.DEVICE_HASHGRID_GUARD:
            check_single_device_hashgrid = MB.check_single_device_hashgrid
            check_single_device_hashgrid(own=self.material.solver.hash)   # a grid built after construction
        order = np.argsort(ids, kind="stable")
        ids_sorted = ids[order]
        seeds_sorted = None if seeds is None else seeds[order]
        with self._stream():
            self.material.reset(None if env_ids is None else ids_sorted, seeds_sorted)
            index = torch.as_tensor(ids_sorted, dtype=torch.int64, device=self.torch_device)
            self._after_reset(index=index)
            return self._observe(), self._reset_info()

    def _after_reset(self, index=None, mask=None):
        """Env-side state of reset envs (material already reset): commanded pose, previous action, counters."""
        torch = self.torch
        if mask is None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.torch_device)
            mask[index] = True
        self.commanded_pose.copy_(torch.where(mask[:, None], self._initial_pose, self.commanded_pose))
        self.previous_action.masked_fill_(mask[:, None], 0.)
        self.armed.masked_fill_(mask, False)
        self.episode.add_(mask.to(torch.int64))
        self.drive.masked_fill_(mask, DRIVE_ACTION)
        counts = self.reward.severed_counts(self._codes)
        self.reward.counts.copy_(torch.where(mask[None], counts, self.reward.counts))

    def _reset_device(self, mask):
        """Masked device reset (autoreset): no host sync; next seed = splitmix64(seed + episode + 1)."""
        torch = self.torch
        self._reset_mask.copy_(mask.to(torch.int32))
        nxt = splitmix64(self._seed + self.episode + 1)
        self._reset_seed.copy_(torch.where(mask, nxt, self._seed))
        self.material.reset_masked(full=False)
        self._after_reset(mask=mask)

    # ------------------------------------------------------------------ step

    def step(self, actions):
        """Apply ``edge_delta_v1`` actions ``[N,8]`` to all envs; returns obs, reward, terminated, truncated, info."""
        return self._step(actions, None, None, None, None)

    def step_measured_edge(self, edge_pose, cut_index, pad_boxes=None, env_mask=None, actions=None):
        """PRIVILEGED replay drive (``drive: measured_trajectory_replay``).

        ``edge_pose [N,7]`` float64 WORLD measured edge poses and ``cut_index [N]`` int32 (-1 disarmed) drive the
        envs in ``env_mask`` (``[N]`` bool, default all); ``pad_boxes [N,B,2,3]`` float32 edge-frame obstacle
        boxes (as ``update_pad_obstacles``) are written for those envs first. Envs outside the mask apply
        ``actions [N,8]`` (required then) exactly as ``step``. All in one batched control step."""
        return self._step(actions, edge_pose, cut_index, pad_boxes, env_mask)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("vector env is closed")

    def _check_tensor(self, value, shape, dtype, name):
        torch = self.torch
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.device != self.torch_device:
            raise ValueError(f"{name} must be a torch tensor {shape} on {self.torch_device} (no implicit copies)")
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}")

    def _step(self, actions, edge_pose, cut_index, pad_boxes, env_mask):
        self._check_open()
        torch, N = self.torch, self.num_envs
        replay = edge_pose is not None
        if replay:
            self._check_tensor(edge_pose, (N, 7), torch.float64, "edge_pose")
            self._check_tensor(cut_index, (N,), torch.int32, "cut_index")
            if env_mask is not None:
                self._check_tensor(env_mask, (N,), torch.bool, "env_mask")
                if actions is None:
                    raise ValueError("actions are required for envs outside env_mask")
            elif actions is not None:
                raise ValueError("actions are ignored when every env replays; pass env_mask to mix")
            if pad_boxes is not None:
                if not self.pads_configured:
                    raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", "pad_boxes need a TaskFamily with "
                                                         "pad_boxes_edge (handle + two pads)")
                self._check_tensor(pad_boxes, (N, self.boxes, 2, 3), torch.float32, "pad_boxes")
        if actions is not None:
            self._check_tensor(actions, (N, 8), None, "actions")
            if not actions.dtype.is_floating_point:
                raise ValueError("actions must be a floating tensor")
        with self._stream():
            if actions is not None:
                result = self.actions.apply(actions, self.commanded_pose)
                pose, cut, status = result.commanded_pose, result.cut_digit, result.status
                executed, armed, clamped = result.executed_action, result.armed, (
                    result.translation_clamped | result.rotation_clamped | result.workspace_clamped)
                out_of_box = result.out_of_box
            if replay:
                mask = torch.ones(N, dtype=torch.bool, device=self.torch_device) if env_mask is None else env_mask
                replay_executed = torch.zeros((N, 8), dtype=torch.float32, device=self.torch_device)
                replay_executed[:, 6] = torch.where(cut_index >= 0, 1., -1.)
                if actions is None:
                    pose, cut = edge_pose, cut_index
                    status = torch.full((N,), STATUS_REPLAY, dtype=torch.int32, device=self.torch_device)
                    executed, armed = replay_executed, cut_index >= 0
                    clamped = torch.zeros(N, dtype=torch.bool, device=self.torch_device)
                    out_of_box = clamped
                else:
                    m = mask[:, None]
                    pose = torch.where(m, edge_pose, pose)
                    cut = torch.where(mask, cut_index, cut)
                    status = torch.where(mask, torch.full_like(status, STATUS_REPLAY), status)
                    executed = torch.where(m, replay_executed, executed)
                    armed = torch.where(mask, cut_index >= 0, armed)
                    clamped = clamped & ~mask
                    out_of_box = out_of_box & ~mask
                if pad_boxes is not None:
                    m3 = mask[:, None, None]
                    self._box_centers.copy_(torch.where(m3, pad_boxes[:, :, 0], self._box_centers))
                    self._box_halves.copy_(torch.where(m3, pad_boxes[:, :, 1], self._box_halves))
                    self.replay_pad_steps.add_(mask.to(torch.int64))
                self.drive.copy_(mask.to(torch.int8))
                self.replay_steps.add_(mask.to(torch.int64))
                self.action_steps.add_((~mask).to(torch.int64))
            else:
                self.drive.zero_()
                self.action_steps.add_(1)
            # ideal tracking: measured = commanded
            self.commanded_pose.copy_(pose)
            self._measured.copy_(pose)
            self._cut_input.copy_(cut.to(torch.int32))
            self.material.advance(interpolation="device")      # strict: one int32 readback, raises on invalid
            invalid = self._invalid != 0
            reward, terms = self.reward.step(self._codes, invalid)
            self.previous_action.copy_(executed)
            self.armed.copy_(armed)
            step_count = self._step_count.clone()
            truncated = step_count >= self.max_episode_steps
            terminated = invalid.clone()
            obs = self._observe()
            info = {"step_count": step_count, "action_status": status.to(torch.int32),
                    "invalid_reason": self._invalid.clone(), "cut_digit": cut.to(torch.int32),
                    "action_clamped": clamped, "action_out_of_box": out_of_box, "drive": self.drive.clone(),
                    "seed": self._seed.clone(), "episode": self.episode.clone(), **terms}
            if self.autoreset == "same_step":
                done = terminated | truncated
                info["final_obs"] = obs
                info["autoreset"] = done
                self._reset_device(done)
                obs = self._observe()
            return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ evaluation (frozen evaluator, host)

    def snapshot_world(self, e):
        """Host WORLD state of env ``e`` (positions, velocities, codes) from device slices; oracle, never obs."""
        from ..scene_frames import transform_points
        s, pk = self.material.solver, self.material.packed
        p0, p1 = int(pk.particle_offset[e]), int(pk.particle_offset[e + 1])
        with self._stream():
            x, v, code = s.x[p0:p1].numpy(), s.v[p0:p1].numpy(), s.code[p0:p1].numpy()
            step = int(self.material.step_count[e:e + 1].numpy()[0])
        T = self._env_host[e]["T_world_local"]
        return dict(positions=transform_points(T, x), velocities=v @ T[:3, :3].T, codes=np.asarray(code),
                    step_count=step, time_s=step * self.control_dt)

    def evaluation_reference(self, e):
        """Reference arrays of env ``e`` in the frozen evaluator's keys (as ``run_baseline.evaluate_states``)."""
        from ..scene_frames import transform_points
        pk = self.material.packed
        h = self._env_host[e]
        p0, p1 = int(pk.particle_offset[e]), int(pk.particle_offset[e + 1])
        return dict(mass_kg=np.asarray(pk.host["mass"][p0:p1], float).copy(),
                    reference_positions=transform_points(h["T_world_local"], h["points"]),
                    reference_positions_local=np.asarray(h["points"]).copy(),
                    reference_indices=np.asarray(h["lattice"]).copy(),
                    reference_stencil_nodes=pk.host["nodes"][p0:p1] - np.int32(pk.node_offset[e]),
                    reference_stencil_weights=pk.host["weights"][p0:p1].copy(),
                    reference_stencil_gradients=pk.host["gradients"][p0:p1].copy())

    def evaluate(self, env_ids=None, *, frames=None):
        """Frozen seeded_trim goal evaluation of the CURRENT state of ``env_ids`` (host; call at episode end).

        The recording handed to ``task.evaluate_goal`` (= ``goals.evaluate_goal``, the same call as
        ``seeded_trim.evaluate_recording`` without ``surface_state``) has two frames: the reset state (t = 0,
        reference positions, zero velocities and codes) and the current state at ``t = step_count * dt``.
        ``frames`` (optional, per env: list of ``snapshot_world`` dicts in time order, the first after t = 0)
        replaces the final frame by that series plus the current state. Returns one dict per env with the
        full evaluator result and a summary."""
        self._check_open()
        ids = range(self.num_envs) if env_ids is None else [int(e) for e in np.asarray(env_ids).reshape(-1)]
        out = []
        for e in ids:
            start = time.perf_counter()
            task, ref = self._env_host[e]["task"], self.evaluation_reference(e)
            final = self.snapshot_world(e)
            series = list((frames or {}).get(e, [])) + [final]
            series = [s for s in series if s["step_count"] > 0]
            n = len(ref["mass_kg"])
            data = dict(ref, time_s=np.asarray([0.] + [s["time_s"] for s in series], float),
                        positions=np.stack([ref["reference_positions"]] + [s["positions"] for s in series]),
                        velocities=np.stack([np.zeros((n, 3))] + [s["velocities"] for s in series]),
                        codes=np.stack([np.zeros(n, np.int32)] + [s["codes"].astype(np.int32) for s in series]),
                        code_encoding="packed2", max_cuts=int(task.max_cuts),
                        particle_spacing_m=task.particle_spacing_m)
            result = task.evaluate_goal(data)
            out.append(dict(env=e, scene_seed=self._env_host[e]["scene_seed"], task_id=task.task_id,
                            step_count=final["step_count"], time_s=final["time_s"], frames=len(data["time_s"]),
                            final_codes_sha256=hashlib.sha256(np.ascontiguousarray(final["codes"])).hexdigest(),
                            labelled=int(np.count_nonzero(final["codes"])),
                            summary=summarize_evaluation(result), result=result,
                            evaluator="goals.evaluate_goal via ConfiguredTaskSpec.evaluate_goal (frozen goal "
                                      f"{task.goal.get('goal_id')}); host snapshot at episode end",
                            wall_s=time.perf_counter() - start))
        return out

    # ------------------------------------------------------------------ receipt

    def fidelity(self):
        """Reduced-fidelity reasons computed against ``NATIVE_TRIM_V7_2_REFERENCE`` (batch-wide and per env)."""
        ref, pk = NATIVE_TRIM_V7_2_REFERENCE, self.material.packed
        u = pk.uniform
        reasons = ["ideal_edge_tracking_no_robot_dynamics", "reaction_not_fed_back"]
        actual = dict(solver="tlmpm_v7_2", cutting_law=int(u["cutting_law"]), max_cuts=int(u["max_cuts"]),
                      slots_per_node=int(u["slots"]), particle_spacing_m=float(u["spacing"]),
                      grid_dx_m=float(u["dx"]), material_dt_s=float(u["dt"]),
                      substeps_per_control_step=int(u["substeps"]), control_dt_s=self.control_dt,
                      episode_steps=self.max_episode_steps, obstacle_boxes=self.boxes)
        for key in ("cutting_law", "max_cuts", "slots_per_node", "particle_spacing_m", "grid_dx_m", "material_dt_s",
                    "substeps_per_control_step", "control_dt_s", "episode_steps"):
            if not np.isclose(actual[key], ref[key], rtol=1e-9, atol=0):
                reasons.append(f"{key}={actual[key]} (reference {ref[key]})")
        if self.boxes == 1:
            reasons.append("no_finger_pad_obstacles")
        per_env = []
        with self._stream():
            action_steps = self.action_steps.cpu().numpy()
            replay_steps = self.replay_steps.cpu().numpy()
            replay_pad_steps = self.replay_pad_steps.cpu().numpy()
        for e in range(self.num_envs):
            env_reasons = []
            if self.pads_configured and (action_steps[e] > 0 or replay_pad_steps[e] < replay_steps[e]):
                env_reasons.append("finger_pad_boxes_static_in_edge_frame_on_non_measured_steps")
            per_env.append(dict(env=e, scene_seed=self.task_family.scene_seeds[e],
                                drive_steps={DRIVE_NAMES[DRIVE_ACTION]: int(action_steps[e]),
                                             DRIVE_NAMES[DRIVE_REPLAY]: int(replay_steps[e])},
                                replay_steps_with_measured_pads=int(replay_pad_steps[e]), reasons=env_reasons))
        return dict(reference=ref, actual=actual, reduced_fidelity=True, reasons=reasons, per_env=per_env,
                    computed_not_hand_set=True)

    def receipt(self):
        import torch
        import warp as wp
        from ..batched.layout import frozen_source_sha256
        root = Path(__file__).resolve().parents[1]
        evaluator_sources = {name: _sha256_file(root / name) for name in ("goals.py", "seeded_trim.py",
                                                                           "scene_config.py", "control.py")}
        rl_sources = {name: _sha256_file(Path(__file__).with_name(name))
                      for name in ("vector_env.py", "actions.py", "observations.py", "reward_packed.py")}
        try:
            import gymnasium
            gym_version = gymnasium.__version__
        except ImportError:
            gym_version = None
        return dict(
            version=VECTOR_ENV_VERSION, runtime=RUNTIME, backend=BACKEND, num_envs=self.num_envs,
            device=str(self.torch_device), warp_device=self.wp_device, parity_mode=self.parity_mode,
            strict=self.strict, autoreset=self.autoreset, observation_profile=self.observation_profile,
            observation=self.observation.metadata(), action_spec=self.actions.spec(), reward=self.reward.metadata(),
            task_family=self.task_family.to_dict(),
            tasks=[dict(env=e, scene_seed=h["scene_seed"], task_id=h["task"].task_id, config_sha256=h["scene_key"])
                   for e, h in enumerate(self._env_host)],
            episode=dict(control_dt_s=self.control_dt, declared_steps=self.declared_episode_steps,
                         max_episode_steps=self.max_episode_steps),
            stream_policy="Warp launches on torch.cuda.current_stream via wp.stream_from_torch + wp.ScopedStream; "
                          "graph captured on Warp's own stream at construction" if self._cuda else "cpu (no streams)",
            debug_sync=self.debug_sync,
            fidelity=self.fidelity(), material=self.material.receipt(),
            versions=dict(python=platform.python_version(), torch=torch.__version__, warp=wp.config.version,
                          numpy=np.__version__, gymnasium=gym_version),
            frozen_source_sha256=frozen_source_sha256(), evaluator_source_sha256=evaluator_sources,
            rl_source_sha256=rl_sources, build_wall_s=getattr(self, "build_wall_s", None),
            calibration="SIMULATOR_ONLY_UNCALIBRATED", success_source="evaluate() only (frozen evaluator)")

    # ------------------------------------------------------------------ lifetime

    def close(self):
        if self._closed:
            return
        if self.output_dir is not None:
            try:
                (self.output_dir / "vector_env_receipt_close.json").write_text(
                    json.dumps(self.receipt(), indent=1, sort_keys=True, default=_json_default) + "\n")
            except Exception as error:   # never hide a close failure; record and continue releasing
                (self.output_dir / "vector_env_close_error.txt").write_text(repr(error) + "\n")
        if self._cuda:
            self.wp.synchronize_device(self.wp_device)
        self.material.release()      # frees the device HashGrid: another env may be opened in this process
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return "<bytes>"
    return str(value)


def summarize_evaluation(result):
    """Compact view of a ``goals.evaluate_goal`` result (same fields as the P1-0 baseline summary)."""
    outcome = result.get("outcome", {})
    metrics = outcome.get("metrics", {})
    items = metrics.get("items", [])
    return dict(verdict=result.get("verdict"), integrity=result.get("integrity", {}).get("status"),
                integrity_checks=result.get("integrity", {}).get("checks"), outcome=outcome.get("status"),
                outcome_checks=outcome.get("checks"), mechanical_components=metrics.get("mechanical_components"),
                largest_component_below_plane_mass_share=metrics.get("largest_component_below_plane_mass_share"),
                detached_below_plane_mass_fraction=metrics.get("detached_below_plane_mass_fraction"),
                removed_mass_fraction={i.get("id"): i.get("removed_mass_fraction") for i in items},
                residual_substrate_height_m={i.get("id"): i.get("residual_substrate_height_m") for i in items})


@contextmanager
def open_vector_env(num_envs, runtime=RUNTIME, task=None, material=MATERIAL, observation_profile="state",
                    device="cuda", backend=BACKEND, action_spec=ACTION_SPEC, reward=REWARD, autoreset=False,
                    strict=True, output_dir=None, **options):
    """Open a ``VectorCuttingEnv`` (ADR 0021 §8) and close it on exit."""
    env = VectorCuttingEnv(num_envs, runtime=runtime, task=task, material=material,
                           observation_profile=observation_profile, device=device, backend=backend,
                           action_spec=action_spec, reward=reward, autoreset=autoreset, strict=strict,
                           output_dir=output_dir, **options)
    try:
        yield env
    finally:
        env.close()


__all__ = ["open_vector_env", "VectorCuttingEnv", "TaskFamily", "UnsupportedVectorConfiguration",
           "NATIVE_TRIM_V7_2_REFERENCE", "splitmix64", "summarize_evaluation"]
