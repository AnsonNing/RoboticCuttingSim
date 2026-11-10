"""Batched vector environment ``open_vector_env`` for runtime ``rl_gpu_headless`` (ADR 0021 §8, P1-2).

P1-5 T5.2: ``task`` may also be a ``rl.configured_family.ConfiguredTaskFamily`` (configured ``rcs_scene_v1`` /
``rcs_goal_v1`` scenes on ``tlmpm_v7_2``, e.g. the CUT workload ``cut_v7_2_v1``); the shaping reward is dispatched by
goal type (``reward=None``: ``trim_bond_progress_packed_v1`` for TRIM, ``sever_bond_progress_packed_v1`` for
sever/divide goals) and fidelity is computed against the family's Native reference.

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
snapshot at episode end, never from the shaping reward. ``evaluation_stream=True`` (opt-in, P1-2b) records the
Native WP-17 sample cadence (reset, every 10 steps, done) on the device and ``evaluate()`` passes those frames
(``rl.evaluation_stream``); the default keeps the two-frame evaluation and every result names its mode.

P1-2b additions (both backends): ``tasks=`` per-env task selection (``[512, 15]`` or dicts with ``scene_seed``,
``family_id``, ``seed``); ``info["termination_reason"]`` (``rl.termination`` bitfield, decoded by
``decode_termination``); the material input guard (``rl.material_guard``, default on, ``input_guard=False`` rolls
it back) that holds an env whose edge/pad input leaves the valid domain; infrastructure failures raised as
``VectorEnvInfrastructureError``. ``backend="isaac_lab_newton"`` (workload B, robot-batched) dispatches to
``rl.lab_vector_env.LabVectorCuttingEnv`` with the same public contract (``rl.vector_contract``). Fidelity against the Native TRIM reference is computed
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
VECTOR_API_REVISION = "p1_2b"   # tasks=, termination_reason, input guard, evaluation stream, isaac_lab_newton dispatch
RUNTIME = "rl_gpu_headless"
BACKEND = "kinematic_ideal_tracking_batched"
LAB_BACKEND = "isaac_lab_newton"
BACKENDS = (BACKEND, LAB_BACKEND)
ACTION_SPEC = "edge_delta_v1"
REWARD = "trim_bond_progress_packed_v1"
SEVER_REWARD = "sever_bond_progress_packed_v1"         # P1-5 T5.3 (rl.reward_sever), configured CUT scenes
REWARD_BY_GOAL = "by_goal_type_v1"                      # reward=None: dispatch by the tasks' goal type (P1-5 T5.2)
REWARDS = (REWARD, SEVER_REWARD, REWARD_BY_GOAL)
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
                              "UNSUPPORTED_DEVICE", "UNSUPPORTED_AUTORESET_FINAL_OBS")


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
    family_ids: tuple | None = None     # per-env family (P1-2b ``tasks=``); None: ``family_id`` everywhere
    env_seeds: tuple | None = None      # per-env initial RNG seed (P1-2b ``tasks=``); None: ``sim_seed`` everywhere

    def __post_init__(self):
        seeds = tuple(int(s) for s in self.scene_seeds)
        if not seeds or any(s < 0 for s in seeds):
            raise ValueError("scene_seeds must be a nonempty sequence of nonnegative ints")
        object.__setattr__(self, "scene_seeds", seeds)
        for family in (self.family_id, *(self.family_ids or ())):
            if family not in TASK_FAMILIES:
                raise UnsupportedVectorConfiguration("UNSUPPORTED_TASK", f"family {family!r}; "
                                                     f"supported {sorted(TASK_FAMILIES)}")
        if self.family_ids is not None:
            object.__setattr__(self, "family_ids", tuple(self.family_ids))
            if len(self.family_ids) != len(seeds):
                raise ValueError("family_ids must hold one family per env")
        if self.env_seeds is not None:
            env_seeds = tuple(int(s) for s in self.env_seeds)
            if len(env_seeds) != len(seeds) or any(s < 0 for s in env_seeds):
                raise ValueError("env_seeds must hold one nonnegative int per env")
            object.__setattr__(self, "env_seeds", env_seeds)
        if self.max_episode_steps is not None and (not isinstance(self.max_episode_steps, int)
                                                   or self.max_episode_steps < 1):
            raise ValueError("max_episode_steps must be a positive int or None")
        if (self.pad_boxes_edge is None) != (self.pad_recipients is None):
            raise ValueError("pad_boxes_edge and pad_recipients are given together")

    def family_of(self, e):
        return self.family_id if self.family_ids is None else self.family_ids[e]

    def env_seed(self, e):
        return int(self.sim_seed) if self.env_seeds is None else self.env_seeds[e]

    def to_dict(self):
        out = asdict(self)
        for key in ("initial_edge_poses", "pad_boxes_edge"):
            if out[key] is not None:
                out[key] = np.asarray(out[key], float).tolist()
        return out


def _compile(family, seed, family_id=None):
    from ..scene_config import compile_task
    from ..seeded_trim import generate_scene, goal_config
    options = {} if family.particle_spacing_m is None else {"particle_spacing_m": float(family.particle_spacing_m)}
    family_id = family.family_id if family_id is None else family_id
    return compile_task(generate_scene(seed, distribution=TASK_FAMILIES[family_id]), goal_config(),
                        seed=int(family.sim_seed), **options)


def compile_family(family):
    """One compiled task per env (compiled once per distinct (family, scene seed))."""
    cache = {}
    for e, seed in enumerate(family.scene_seeds):
        key = (family.family_of(e), seed)
        if key not in cache:
            cache[key] = _compile(family, seed, key[0])
    return [cache[(family.family_of(e), seed)] for e, seed in enumerate(family.scene_seeds)]


TASK_SELECTION_KEYS = ("scene_seed", "family_id", "seed")


def resolve_tasks(num_envs, task=None, tasks=None):
    """``open_vector_env(tasks=...)``: per-env task/seed selection of a heterogeneous batch.

    ``tasks``: one item per env, an int scene seed (``[512, 15]``) or a dict with ``scene_seed`` and optional
    ``family_id`` (``agentic_trim_id`` | ``agentic_trim_ood``) and ``seed`` (the env's initial RNG seed). With a
    ``TaskFamily`` as ``task`` its other options (spacing, episode length, poses, pads, dt) are kept and the
    per-env selection is replaced; without one, a default ``TaskFamily`` is built. Anything else (compiled task
    objects, CUT scenes) raises ``UNSUPPORTED_TASK``."""
    from dataclasses import replace
    if tasks is None:
        return task
    items = list(tasks)
    if not isinstance(num_envs, int) or len(items) != num_envs:
        raise ValueError(f"tasks must hold one entry per env ({num_envs}), got {len(items)}")
    seeds, families, env_seeds = [], [], []
    for e, item in enumerate(items):
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            item = {"scene_seed": int(item)}
        if not isinstance(item, dict) or "scene_seed" not in item or set(item) - set(TASK_SELECTION_KEYS):
            raise UnsupportedVectorConfiguration("UNSUPPORTED_TASK", f"tasks[{e}]={item!r}: an int scene seed or a "
                                                 f"dict with keys {TASK_SELECTION_KEYS} (scene_seed required)")
        seeds.append(int(item["scene_seed"]))
        families.append(item.get("family_id"))
        env_seeds.append(item.get("seed"))
    if task is not None and not isinstance(task, TaskFamily):
        raise UnsupportedVectorConfiguration("UNSUPPORTED_TASK", "task must be a TaskFamily (or None) with tasks=")
    base = task if task is not None else TaskFamily(scene_seeds=tuple(seeds))
    default_family = base.family_id
    family_ids = tuple(f if f is not None else default_family for f in families)
    uniform_family = len(set(family_ids)) == 1
    default_seed = int(base.sim_seed)
    resolved_seeds = tuple(int(s) if s is not None else default_seed for s in env_seeds)
    return replace(base, scene_seeds=tuple(seeds), family_id=family_ids[0] if uniform_family else default_family,
                   family_ids=None if uniform_family else family_ids,
                   env_seeds=None if all(s is None for s in env_seeds) else resolved_seeds)


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


def _guard_config(value, config_type):
    """``input_guard``: None/True (default config), False (disabled: the P1-2 behaviour), a dict or a config."""
    if value is False:
        return None
    if value is None or value is True:
        return config_type().validate()
    if isinstance(value, config_type):
        return value.validate()
    if isinstance(value, dict):
        return config_type(**value).validate()
    raise ValueError("input_guard must be None, a bool, a dict or a MaterialInputGuardConfig")


# ----------------------------------------------------------------------------- environment

class VectorCuttingEnv:
    """N seeded TRIM envs in one batched material (see the module docstring). Use ``open_vector_env``."""

    metadata = {"render_modes": [], "runtime": RUNTIME, "backend": BACKEND, "version": VECTOR_ENV_VERSION}

    def __init__(self, num_envs, *, runtime=RUNTIME, task=None, material=MATERIAL, observation_profile="state",
                 device="cuda", backend=BACKEND, action_spec=ACTION_SPEC, reward=None, autoreset=False,
                 strict=True, output_dir=None, parity_mode=False, reward_config=None, cameras=None, sensors=None,
                 recorders=None, debug_sync=False, particle_contact_mode="frozen_v1", particle_contact_options=None,
                 evaluation_stream=None, input_guard=None, terminal_capture=False,
                 allow_multiple_device_hashgrids=False):
        t0 = time.perf_counter()
        import torch
        import warp as wp
        self.torch, self.wp = torch, wp
        if runtime != RUNTIME:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_RUNTIME", f"{runtime!r}; only {RUNTIME!r}")
        if backend != BACKEND:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_BACKEND", f"{backend!r}; VectorCuttingEnv is {BACKEND!r} "
                                                 f"(open_vector_env dispatches {LAB_BACKEND!r} to LabVectorCuttingEnv)")
        if material != MATERIAL:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"material={material!r}; only {MATERIAL!r}")
        if observation_profile not in OBSERVATION_PROFILES:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_SENSOR" if "vision" in str(observation_profile)
                                                 or "camera" in str(observation_profile) else "UNSUPPORTED_OPTION",
                                                 f"observation_profile={observation_profile!r}; "
                                                 f"supported {OBSERVATION_PROFILES}")
        if action_spec != ACTION_SPEC:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"action_spec={action_spec!r}")
        if reward is not None and reward not in REWARDS:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"reward={reward!r}; supported {REWARDS} or None "
                                                 "(by goal type)")
        self.requested_reward = reward
        if not any(autoreset is mode or autoreset == mode for mode in AUTORESET_MODES) or autoreset is True:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"autoreset={autoreset!r}; "
                                                 f"supported {AUTORESET_MODES}")
        if cameras or sensors:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_SENSOR", f"cameras={cameras!r} sensors={sensors!r}")
        if recorders:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_RECORDER", f"recorders={recorders!r}")
        from .configured_family import ConfiguredTaskFamily
        if not isinstance(task, (TaskFamily, ConfiguredTaskFamily)):
            raise UnsupportedVectorConfiguration("UNSUPPORTED_TASK", "task must be a TaskFamily of seeded TRIM scenes or a "
                                                 "ConfiguredTaskFamily of tlmpm_v7_2 configured scenes")
        self.configured = isinstance(task, ConfiguredTaskFamily)
        if not isinstance(num_envs, int) or num_envs < 1 or len(task.scene_seeds) != num_envs:
            raise ValueError("num_envs must be a positive int equal to len(task.scene_seeds)")
        from .evaluation_stream import stream_config
        from .material_guard import MaterialInputGuardConfig
        self.evaluation_stream_config = stream_config(evaluation_stream)
        self.terminal_capture_enabled = bool(terminal_capture)
        if self.terminal_capture_enabled and autoreset != "same_step":
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", "terminal_capture=True captures envs that finish "
                                                 "under autoreset='same_step'; without autoreset evaluate() at the "
                                                 "episode end instead")
        if self.evaluation_stream_config is not None and autoreset == "same_step" and not self.terminal_capture_enabled:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", "evaluation_stream with autoreset='same_step' "
                                                 "needs terminal_capture=True (else a finished episode's samples are "
                                                 "overwritten in the same call)")
        self.input_guard_config = _guard_config(input_guard, MaterialInputGuardConfig)
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
        # P1-3a opt-in material execution (batched/particle_contacts.py); "frozen_v1" is the default and rollback.
        self.particle_contact_mode = particle_contact_mode
        self.particle_contact_options = particle_contact_options
        # DIAGNOSTICS ONLY (review F7): forwarded to BatchedCuttingMaterial; lets several batches with device grids
        # live in one process (P1-3a samestate/ab/stagebench). Refused by default; recorded in the receipt.
        self.allow_multiple_device_hashgrids = bool(allow_multiple_device_hashgrids)
        self.debug_sync = bool(debug_sync)   # ADR §14 risk 4: synchronize the device after every public call
        self.last_warp_stream_handle = None  # Warp stream used by the last public call (stream-policy test hook)
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None:
            if self.output_dir.exists() and any(self.output_dir.iterdir()):
                raise FileExistsError(f"output_dir {self.output_dir} exists and is not empty (never overwritten)")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._streams = {}
        self._closed = False
        self.infrastructure_error = None
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
        from .reward_packed import pack_bonds
        from .reward_sever import bonds_for_task, reward_class_for_version

        fam, N = self.task_family, self.num_envs
        self.tasks = fam.compile_tasks() if self.configured else compile_family(fam)
        first = self.tasks[0]
        for e, task in enumerate(self.tasks):
            if task.solver_module != "tlmpm_v7_2":
                raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER", f"env {e}: solver {task.solver_module!r}; "
                                                    "only tlmpm_v7_2")
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
        bonds, reward_versions = [], []
        for e, task in enumerate(self.tasks):
            config = CouplingConfig(control_dt=self.control_dt, material_dt=float(fam.material_dt_s),
                                    dx=float(task.grid_dx_m), device="cpu", feedback=True)
            m = LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                                    solver=task.solver_module, v7_2_optimizations=True)
            if pads is not None:
                m.configure_pad_obstacles(pads[e], list(fam.pad_recipients))
            specs.append(layout.EnvSpec(m, initial[e], fam.env_seed(e), task.config_sha256))
        self.material = BatchedCuttingMaterial(specs, device=self.wp_device, parity_mode=self.parity_mode,
                                               strict=self.strict, particle_contact_mode=self.particle_contact_mode,
                                               particle_contact_options=self.particle_contact_options,
                                               allow_multiple_device_hashgrids=self.allow_multiple_device_hashgrids)
        if self._cuda:
            wp.synchronize_device(self.wp_device)
        pk = self.material.packed
        for e, (spec, task) in enumerate(zip(specs, self.tasks)):
            m = spec.material
            key = task.config_sha256
            if key not in particles_cache:
                particles_cache[key] = (np.asarray(m.points), np.asarray(m.masses), np.asarray(m.lattice))
            points, masses, lattice = particles_cache[key]
            self._env_host.append(dict(task=task, scene_seed=fam.scene_seeds[e], family_id=fam.family_of(e), scene_key=key,
                                       T_world_local=np.asarray(m.T_world_local, float), points=points,
                                       masses=masses, lattice=lattice))
            version, env_bonds = bonds_for_task(task, particles=(points, masses, lattice))
            reward_versions.append(version)
            bonds.append(env_bonds)
        del specs   # frozen CPU single solvers are no longer needed (their arrays are packed)
        if len(set(reward_versions)) != 1:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_NONUNIFORM_BATCH", f"reward kinds differ {reward_versions}")
        self.reward_version = reward_versions[0]
        if self.requested_reward not in (None, REWARD_BY_GOAL, self.reward_version):
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"reward={self.requested_reward!r} does not match "
                                                 f"the tasks' goal type ({self.reward_version})")
        reward_cls, default_reward_config = reward_class_for_version(self.reward_version)
        T_world = np.stack([np.asarray(t.T_world_workpiece, float) for t in self.tasks])
        if not np.allclose(T_world, pk.env["T_world_local"], atol=1e-12, rtol=0):
            raise ValueError("task.T_world_workpiece differs from the material T_world_local")

        dev = self.torch_device
        with self._stream():
            self.actions = EdgeDeltaActionV1(self.action_limits, T_world, first.max_cuts, device=dev)
            self.reward = reward_cls(pack_bonds(bonds, pk.particle_offset), reward_config or default_reward_config,
                                     device=dev)
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
            self._last_pose = view(mt.last_pose)                # [N,7] float64 (last accepted measured pose)
            self.reward.reset(self._codes)
            from .material_guard import MaterialInputGuard
            from .evaluation_stream import EvaluationStream
            self.input_guard = (None if self.input_guard_config is None else
                                MaterialInputGuard(mt, N, dev, self.input_guard_config))
            from .evaluation_stream import TerminalCapture
            self.terminal = TerminalCapture(mt, dev) if self.terminal_capture_enabled else None
            self.evaluation_stream = (None if self.evaluation_stream_config is None else
                                      EvaluationStream(mt, self.max_episode_steps, dev, self.evaluation_stream_config,
                                                       banks=2 if self.terminal is not None else 1))
            self.termination_reason = torch.zeros(N, dtype=torch.int32, device=dev)
            self.autoreset_count = torch.zeros(N, dtype=torch.int64, device=dev)
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
        return self._infrastructure_guarded(self._reset_impl, seeds=seeds, env_ids=env_ids, scene_seeds=scene_seeds,
                                            options=options, seed=seed)

    def _reset_impl(self, *, seeds=None, env_ids=None, scene_seeds=None, options=None, seed=None):
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
        self.termination_reason.masked_fill_(mask, 0)
        if self.input_guard is not None:
            self.input_guard.reset(mask)
        if self.evaluation_stream is not None:
            self.evaluation_stream.reset(mask)
            self.evaluation_stream.capture(mask)        # the reset state is the step-0 sample (Native capture())

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
        if self.infrastructure_error is not None:
            raise RuntimeError(f"vector env had an infrastructure error ({self.infrastructure_error['kind']}); "
                               "close it and open a new env (or process)")

    def _check_tensor(self, value, shape, dtype, name):
        torch = self.torch
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.device != self.torch_device:
            raise ValueError(f"{name} must be a torch tensor {shape} on {self.torch_device} (no implicit copies)")
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}")

    def _step(self, actions, edge_pose, cut_index, pad_boxes, env_mask):
        self._check_open()
        return self._infrastructure_guarded(self._step_impl, actions, edge_pose, cut_index, pad_boxes, env_mask)

    def _infrastructure_guarded(self, fn, *args, **kwargs):
        """Run a public call; an infrastructure failure (OOM, CUDA/Warp error) becomes
        ``VectorEnvInfrastructureError`` with every env marked ``TERM_INFRASTRUCTURE`` and the env refuses further
        calls. Task failures (strict-mode invalid envs) and caller errors propagate unchanged."""
        from .termination import TERM_INFRASTRUCTURE, VectorEnvInfrastructureError, classify_exception
        try:
            return fn(*args, **kwargs)
        except VectorEnvInfrastructureError:
            raise
        except Exception as error:
            category, kind = classify_exception(error)
            if category != "infrastructure":
                raise
            reason = np.full(self.num_envs, TERM_INFRASTRUCTURE, np.int32)
            self.infrastructure_error = dict(kind=kind, error=f"{type(error).__name__}: {error}"[:2000],
                                             call=getattr(fn, "__name__", str(fn)))
            raise VectorEnvInfrastructureError(kind, self.infrastructure_error["error"], reason) from error

    def _step_impl(self, actions, edge_pose, cut_index, pad_boxes, env_mask):
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
            mask = None
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
            cut = cut.to(torch.int32)
            box_write = mask
            if self.input_guard is not None:
                # P1-2b material input guard: out-of-domain envs are held (no host sync), others untouched
                pose, cut, box_write = self.input_guard.apply(pose, cut, self._last_pose, step_count=self._step_count + 1,
                                                              boxes=pad_boxes, box_mask=mask)
                armed = armed & ~self.input_guard.held
            if replay:
                if pad_boxes is not None:
                    m3 = box_write[:, None, None]
                    self._box_centers.copy_(torch.where(m3, pad_boxes[:, :, 0], self._box_centers))
                    self._box_halves.copy_(torch.where(m3, pad_boxes[:, :, 1], self._box_halves))
                    self.replay_pad_steps.add_(box_write.to(torch.int64))
                self.drive.copy_(mask.to(torch.int8))
                self.replay_steps.add_(mask.to(torch.int64))
                self.action_steps.add_((~mask).to(torch.int64))
            else:
                self.drive.zero_()
                self.action_steps.add_(1)
            # ideal tracking: measured = commanded (held envs: the last accepted pose)
            self.commanded_pose.copy_(pose)
            self._measured.copy_(pose)
            self._cut_input.copy_(cut)
            self.material.advance(interpolation="device")      # strict: one int32 readback, raises on invalid
            from .termination import INVALID_STATE_BITS, TERM_MATERIAL_INVALID, TERM_TRUNCATED
            reason = torch.where(self._invalid != 0, TERM_MATERIAL_INVALID, 0).to(torch.int32)
            if self.input_guard is not None:
                reason = reason | self.input_guard.termination_bits()
            invalid = (reason & INVALID_STATE_BITS) != 0
            reward, terms = self.reward.step(self._codes, invalid)
            self.previous_action.copy_(executed)
            self.armed.copy_(armed)
            step_count = self._step_count.clone()
            truncated = step_count >= self.max_episode_steps
            terminated = reason != 0
            self.termination_reason.copy_(reason | torch.where(truncated, TERM_TRUNCATED, 0).to(torch.int32))
            if self.evaluation_stream is not None:
                self.evaluation_stream.capture(self.evaluation_stream.due(step_count, terminated | truncated))
            obs = self._observe()
            info = {"step_count": step_count, "action_status": status.to(torch.int32),
                    "invalid_reason": self._invalid.clone(), "cut_digit": cut,
                    "action_clamped": clamped, "action_out_of_box": out_of_box, "drive": self.drive.clone(),
                    "seed": self._seed.clone(), "episode": self.episode.clone(),
                    "termination_reason": self.termination_reason.clone(), **terms}
            if self.input_guard is not None:
                info["input_guard_reason"] = self.input_guard.reason.clone()
            if self.autoreset == "same_step":
                done = terminated | truncated
                info["final_obs"] = obs
                info["autoreset"] = done
                if self.terminal is not None:      # review F5: keep the finished episode for evaluate()
                    self.terminal.capture(done, step_count, self.episode)
                    if self.evaluation_stream is not None:
                        self.evaluation_stream.flip(done)
                    info["terminal_captured"] = done.clone()
                self.autoreset_count.add_(done.to(torch.int64))
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
        # copies: on the Warp CPU device .numpy() aliases the live solver arrays
        return dict(positions=transform_points(T, x), velocities=v @ T[:3, :3].T, codes=np.array(code, copy=True),
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

    def evaluate(self, env_ids=None, *, frames=None, episode="current"):
        """Frozen seeded_trim goal evaluation of ``env_ids`` (host; call at episode end).

        ``episode="current"`` scores the CURRENT state. Default (``evaluation_mode =
        two_frame_reset_and_current_v1``): the recording handed to ``task.evaluate_goal`` (= ``goals.evaluate_goal``,
        the same call as ``seeded_trim.evaluate_recording`` without ``surface_state``) has two frames, the reset
        state (t = 0, reference positions, zero velocities and codes) and the current state at
        ``t = step_count * dt``; ``frames`` (optional, per env: ``snapshot_world`` dicts in time order, the first
        after t = 0) are inserted between them. With ``evaluation_stream`` enabled (``evaluation_mode =
        evaluation_stream_v1``) the recorded samples (reset, every ``every`` steps, done) plus the current state are
        passed instead (``frames`` is then refused). An env that was autoreset inside ``step()`` and holds its fresh
        state (step 0) is refused (``NOT_EVALUATED``), never scored silently (review F5).

        ``episode="last_completed"`` (needs ``terminal_capture=True`` with ``autoreset="same_step"``) scores the
        state captured when the env's last episode ended inside ``step()`` (and, with the stream, that episode's
        samples). Returns one dict per env with the full evaluator result, a summary, the mode and
        ``episode_scored``."""
        from .evaluation_stream import DEFAULT_MODE, STREAM_VERSION, evaluate_frozen
        self._check_open()
        if episode not in ("current", "last_completed"):
            raise ValueError("episode is 'current' or 'last_completed'")
        stream = self.evaluation_stream
        if stream is not None and frames:
            raise ValueError("frames= and the evaluation stream are exclusive (the stream already holds the samples)")
        if episode == "last_completed" and self.terminal is None:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", "episode='last_completed' needs "
                                                 "terminal_capture=True with autoreset='same_step'")
        ids = range(self.num_envs) if env_ids is None else [int(e) for e in np.asarray(env_ids).reshape(-1)]
        with self._stream():
            steps = self._step_count.cpu().clone().numpy()
            autoreset = self.autoreset_count.cpu().clone().numpy()
        out = []
        for e in ids:
            h = self._env_host[e]
            extra = dict(scene_seed=h["scene_seed"], family_id=h["family_id"], episode_scored=episode)
            if episode == "current":
                if autoreset[e] > 0 and steps[e] == 0:
                    raise RuntimeError(f"NOT_EVALUATED: env {e} was autoreset inside step() and holds the fresh "
                                       "state of its next episode; use terminal_capture=True and "
                                       "evaluate(episode='last_completed')")
                final = self.snapshot_world(e)
                completed = False
            else:
                with self._stream():
                    final = self.terminal.host_snapshot(e, h["T_world_local"], self.control_dt)
                if final is None:
                    raise RuntimeError(f"NOT_EVALUATED: env {e} has not finished an episode inside step()")
                extra["completed_episode_index"] = final.pop("episode")
                completed = True
            if stream is None:
                series, mode = list((frames or {}).get(e, [])), DEFAULT_MODE
            else:
                with self._stream():
                    series, dropped = stream.host_frames(e, h["T_world_local"], self.control_dt, completed=completed)
                mode = STREAM_VERSION
                extra.update(stream_samples=len(series), stream_every=stream.every, stream_dropped_samples=dropped)
            out.append(evaluate_frozen(h["task"], self.evaluation_reference(e), final, series, mode=mode, env=e,
                                       extra=extra))
        return out

    def _terminal_receipt(self):
        if self.terminal is None:
            return {"enabled": False, "statement": "evaluate() scores the current state; an env autoreset in step() "
                    "and still at step 0 is refused (NOT_EVALUATED)"}
        return self.terminal.receipt()

    # ------------------------------------------------------------------ receipt

    def fidelity(self):
        """Reduced-fidelity reasons computed against the family's Native reference (batch-wide and per env):
        ``NATIVE_TRIM_V7_2_REFERENCE`` for seeded TRIM, ``task_family.native_reference`` for configured scenes."""
        ref = self.task_family.native_reference if self.configured else NATIVE_TRIM_V7_2_REFERENCE
        pk = self.material.packed
        u = pk.uniform
        reasons = ["ideal_edge_tracking_no_robot_dynamics", "reaction_not_fed_back"]
        if ref.get("status") == "NOT_RUN":
            reasons.append("native_reference_not_run")
        actual = dict(solver="tlmpm_v7_2", cutting_law=int(u["cutting_law"]), max_cuts=int(u["max_cuts"]),
                      slots_per_node=int(u["slots"]), particle_spacing_m=float(u["spacing"]),
                      grid_dx_m=float(u["dx"]), material_dt_s=float(u["dt"]),
                      substeps_per_control_step=int(u["substeps"]), control_dt_s=self.control_dt,
                      episode_steps=self.max_episode_steps, obstacle_boxes=self.boxes)
        for key in ("cutting_law", "max_cuts", "slots_per_node", "particle_spacing_m", "grid_dx_m", "material_dt_s",
                    "substeps_per_control_step", "control_dt_s", "episode_steps"):
            if key in ref and not np.isclose(actual[key], ref[key], rtol=1e-9, atol=0):
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
                      for name in ("vector_env.py", "actions.py", "observations.py", "reward_packed.py",
                                   "termination.py", "material_guard.py", "evaluation_stream.py")}
        try:
            import gymnasium
            gym_version = gymnasium.__version__
        except ImportError:
            gym_version = None
        from .termination import termination_table
        return dict(
            version=VECTOR_ENV_VERSION, api_revision=VECTOR_API_REVISION, runtime=RUNTIME, backend=BACKEND,
            num_envs=self.num_envs, termination=termination_table(),
            input_guard={"enabled": False, "statement": "input_guard=False (P1-2 behaviour)"}
            if self.input_guard is None else self.input_guard.receipt(),
            evaluation_stream=None if self.evaluation_stream is None else self.evaluation_stream.receipt(),
            evaluation_mode="evaluation_stream_v1" if self.evaluation_stream is not None
            else "two_frame_reset_and_current_v1", infrastructure_error=self.infrastructure_error,
            terminal_capture=self._terminal_receipt(),
            allow_multiple_device_hashgrids=self.allow_multiple_device_hashgrids,
            device_hashgrid_guard=self.material.device_hashgrid_guard,
            device=str(self.torch_device), warp_device=self.wp_device, parity_mode=self.parity_mode,
            strict=self.strict, autoreset=self.autoreset, observation_profile=self.observation_profile,
            observation=self.observation.metadata(), action_spec=self.actions.spec(), reward=self.reward.metadata(),
            task_family=self.task_family.to_dict(),
            tasks=[dict(env=e, scene_seed=h["scene_seed"], family_id=h["family_id"], task_id=h["task"].task_id,
                        config_sha256=h["scene_key"], initial_seed=self.task_family.env_seed(e))
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
                residual_substrate_height_m={i.get("id"): i.get("residual_substrate_height_m") for i in items},
                piece_mass_fractions=metrics.get("piece_mass_fractions"),
                substantial_pieces=metrics.get("substantial_pieces"),
                main_pieces_mass_fraction=metrics.get("main_pieces_mass_fraction"))


@contextmanager
def open_vector_env(num_envs, runtime=RUNTIME, task=None, material=MATERIAL, observation_profile="state",
                    device="cuda", backend=BACKEND, action_spec=ACTION_SPEC, reward=None, autoreset=False,
                    strict=True, output_dir=None, tasks=None, **options):
    """Open a vector env and close it on exit.

    ``backend="kinematic_ideal_tracking_batched"`` (default, workload A): ``VectorCuttingEnv`` (ADR 0021 §8).
    ``backend="isaac_lab_newton"`` (workload B, robot-batched, ADR 0022): ``rl.lab_vector_env.LabVectorCuttingEnv``
    (Isaac Lab venv; it keeps ``launch_simulation`` open for its lifetime). Both have the same public contract
    (``rl.vector_contract``). ``tasks``: per-env task selection (``resolve_tasks``), e.g. ``tasks=[512, 15]``."""
    task = resolve_tasks(num_envs, task, tasks)
    if backend == LAB_BACKEND:
        from .lab_vector_env import LabVectorCuttingEnv
        env = LabVectorCuttingEnv(num_envs, runtime=runtime, task=task, material=material,
                                  observation_profile=observation_profile, device=device, action_spec=action_spec,
                                  reward=REWARD if reward is None else reward, autoreset=autoreset, strict=strict,
                                  output_dir=output_dir, **options)
    else:
        env = VectorCuttingEnv(num_envs, runtime=runtime, task=task, material=material,
                               observation_profile=observation_profile, device=device, backend=backend,
                               action_spec=action_spec, reward=reward, autoreset=autoreset, strict=strict,
                               output_dir=output_dir, **options)
    try:
        yield env
    finally:
        env.close()


__all__ = ["open_vector_env", "VectorCuttingEnv", "TaskFamily", "UnsupportedVectorConfiguration", "REWARDS",
           "NATIVE_TRIM_V7_2_REFERENCE", "splitmix64", "summarize_evaluation", "resolve_tasks", "compile_family",
           "BACKENDS", "LAB_BACKEND"]
