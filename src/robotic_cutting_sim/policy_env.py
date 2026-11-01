"""Policy-ready research environment (WP-11 Phase G, preset ``cutting_research_v1``).

``PolicyReadyEnv`` wraps one ``live_env.CuttingTaskEnv`` and keeps its
``reset/step/observe/render/evaluate/close`` contract. It adds, all opt-in
through ``open_cutting_env(..., preset="cutting_research_v1")``:

* a research preset: physical grasp, pregrasped tool, gripper held, HOME reset
  whenever the scene declares one, the compiled task's own action limits;
* validated actions with an explicit status in every ``info``
  (``ACCEPTED`` / ``CLAMPED`` / ``REJECTED_<reason>``), execution logging and
  never a silent substitution: a rejected action is not executed, and the
  explicit HOLD that keeps the 50 Hz clock running is reported as such;
* deployable observations (cameras, joints, end-effector, gripper, timestamp,
  task information, episode state) kept apart from a privileged debug dict
  that exists only with ``observation_profile="privileged_state"``;
* optional per-episode recording through ``scripts/run_wp10_live.EpisodeRecorder``.

No physics, IK, renderer, cutting law or threshold lives here; the wrapped
runtime keeps all of them. ``reset()`` is the direct simulator reset (RL);
``return_home()`` is the separate physical return used for demonstrations.
"""
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field, replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .control import DELTA_FRAMES, DELTA_MODE, ActionLimits, IKFailure
from .tasks import ACTION_MODES

ROOT = Path(__file__).resolve().parents[2]
PRESET_ID = "cutting_research_v1"
POLICY_ENV_VERSION = "policy_ready_env_v1"
TASK_SWITCH_VERSION = "task_switch_v1"   # PolicyReadyEnv.switch_task (WP-12 Phase 6, opt-in)
PRESETS = (PRESET_ID,)
ACTION_TYPES = {"edge_pose": ACTION_MODES[0], "edge_delta": DELTA_MODE, "joint_position": ACTION_MODES[1]}
ACTION_KEYS = frozenset({"type", "mode", "target", "frame", "cut_enabled", "cut_index", "gripper"})
GRIPPER_COMMANDS = ("hold", "release")
GRIPPER_CONTROL_MODES = ("hold", "policy")
REJECTION_POLICIES = ("hold", "no_step")
OBSERVATION_PROFILES = ("deployable", "privileged_state")
REJECTION_REASONS = ("INVALID_SCHEMA", "INVALID_TARGET", "CUT_INDEX_OUT_OF_RANGE", "JOINT_LIMIT",
                     "GRIPPER_LOCKED", "GRIPPER_UNSUPPORTED", "UNSUPPORTED_BY_RUNTIME",
                     "COLLISION_FIXTURE", "COLLISION_TABLE", "IK_UNREACHABLE")
ACTION_STATUSES = ("ACCEPTED", "CLAMPED", *("REJECTED_"+reason for reason in REJECTION_REASONS))
# Franka Panda joint limits (rad); a live runtime's own IK model limits take precedence.
PANDA_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -.0175, -2.8973])
PANDA_UPPER = np.array([2.8973, 1.7628, 2.8973, -.0698, 2.8973, 3.7525, 2.8973])
# Names that must never appear anywhere inside a deployable observation (tests walk the whole tree).
PRIVILEGED_KEYS = frozenset({
    "positions", "velocities", "codes", "cut_codes", "component_labels", "components", "material_state",
    "reference_positions", "reference_positions_world", "reference_indices", "mass_kg", "oracle_geometry",
    "contact_wrench_world", "next_material_wrench_world", "force", "wrench", "tool_truth",
    "tool_body_pose_xyzw", "tool_linear_velocity_world_m_s", "tool_angular_velocity_world_rad_s",
    "grasp_status", "privileged", "material", "oracle",
    # WP-12 Layer 6: render-surface provenance and the Hermite cut-event record are oracle/debug data
    "surface_provenance", "hermite", "cut_events"})
# Reset-check fields that are bookkeeping of the env itself (clocks, counters, logs, lifecycle) or deployable
# measurements (joint encoders, camera timestamps). Every other reset_checks field (material state, cut codes,
# simulator tool/edge pose and speeds, the grasp re-stabilization receipt) is privileged and appears in reset
# info only under info["privileged"] with observation_profile="privileged_state" (WP-11 review finding 9).
DEPLOYABLE_RESET_CHECKS = ("time_zero", "oracle_history_single_sample", "lifecycle_ready_after_reset",
                           "ik_failures_zero", "not_terminated", "control_steps_zero", "action_log_empty",
                           "observation_timestamp_matches", "q_active", "camera_timestamp_matches")
# Info-only names that are privileged in addition to PRIVILEGED_KEYS (walked by find_privileged_info_keys).
PRIVILEGED_INFO_KEYS = PRIVILEGED_KEYS | frozenset({
    "reset_checks_privileged", "labels_clean", "material_at_reference_max_m", "material_velocity_max_m_s",
    "tool_pose_world_xyzw", "edge_pose_world_xyzw", "tool_linear_speed_m_s", "tool_angular_speed_rad_s",
    "grasp_reset_status", "material_codes_sha256", "particles", "measured_tool_pose"})
# Inner (CuttingTaskEnv) step-info keys that are privileged: removed from the default step info.
PRIVILEGED_STEP_INFO = ("grasp_status",)
# Controller inputs that are simulator truth, not deployable sensing (documented in POLICY_READY_API.md).
PRIVILEGED_CONTROLLER_INPUTS = {
    "edge_feedback=measured (path_v2/path_v3 measured-edge feedback)":
        "simulator tool-edge pose of the physically held knife (tool truth); not in the deployable observation",
    "reward.CuttingReward": "reference particles, cut codes and the measured tool edge (shaping only)",
    "env.evaluate()": "oracle particle history, codes and components (scoring only, never an input)",
}


class ActionRejected(ValueError):
    """An action that fails validation; carries a REJECTION_REASONS entry and a human-readable detail."""

    def __init__(self, reason, detail):
        if reason not in REJECTION_REASONS:
            raise ValueError(f"Unknown rejection reason {reason!r}")
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, str(detail)


@dataclass(frozen=True)
class ResearchPreset:
    """The ``cutting_research_v1`` defaults. Every field is recorded in reset info and manifests."""

    preset_id: str = PRESET_ID
    tool_attachment_mode: str = "physical_grasp"
    tool_init_mode: str = "pregrasped"
    gripper_control_mode: str = "hold"       # "hold": only `hold`; "policy": hold/release accepted (physical grasp)
    home_reset: str = "when_declared"       # Native resets to the scene HOME whenever the scene declares one
    delta_frame: str = "world"
    rejection_policy: str = "hold"          # "hold": rejected -> explicit cut-disabled HOLD step; "no_step": nothing runs
    collision_check: bool = True            # sampled blade/handle vs declared fixture boxes and table (geometry only)
    collision_tolerance_m: float = .0005
    include_global_camera: bool = True
    camera_every: int = 1                   # control steps between rendered camera observations (live only)
    default_cut_enabled: bool = False       # an action without cut_enabled is NOT armed

    def __post_init__(self):
        if (self.preset_id not in PRESETS or self.gripper_control_mode not in GRIPPER_CONTROL_MODES
                or self.rejection_policy not in REJECTION_POLICIES or self.delta_frame not in DELTA_FRAMES
                or self.tool_attachment_mode != "physical_grasp" or self.tool_init_mode != "pregrasped"
                or self.home_reset != "when_declared" or not isinstance(self.camera_every, int) or self.camera_every < 1
                or not np.isfinite(self.collision_tolerance_m) or self.collision_tolerance_m < 0):
            raise ValueError("Invalid cutting_research_v1 preset field")

    def to_dict(self):
        return {**asdict(self), "version": POLICY_ENV_VERSION}


# ----------------------------------------------------------------------------- geometry checks

def tool_sample_points_edge(profile, spacing_m=.002):
    """Edge-frame sample points on the blade faces/edge line and the handle box faces (geometry only)."""
    section = np.asarray(profile.cross_section_xz(), float)
    half = float(profile.half_width_m)
    ys = np.linspace(-half, half, max(2, int(np.ceil(2*half/spacing_m))+1))
    points = []
    closed = np.vstack([section, section[:1]])
    for a, b in zip(closed[:-1], closed[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b-a)/spacing_m))+1)
        for u in np.linspace(0., 1., n):
            x, z = a+(b-a)*u
            points.append(np.c_[np.full_like(ys, x), ys, np.full_like(ys, z)])
    try:
        from .tool_profile import handle_box_edge
        center, half_extents = handle_box_edge(profile)
        grids = [np.linspace(-h, h, max(2, int(np.ceil(2*h/(2*spacing_m)))+1)) for h in half_extents]
        for axis in range(3):
            for sign in (-1., 1.):
                others = [grids[k] for k in range(3) if k != axis]
                u, v = np.meshgrid(*others, indexing="ij")
                face = np.zeros((u.size, 3))
                face[:, axis] = sign*half_extents[axis]
                face[:, [k for k in range(3) if k != axis]] = np.c_[u.ravel(), v.ravel()]
                points.append(face+center)
    except (ValueError, AttributeError):
        pass  # legacy wedge: no profile handle box
    return np.vstack(points)


def check_tool_collision(task, edge_pose_world, *, tolerance_m=.0005, points_edge=None):
    """Sampled knife (blade + handle) vs declared rigid fixture boxes and the table top.

    Geometry only (declared boxes, nominal tool at the commanded edge pose); the
    workpiece is not an obstacle (cutting it is the task). Not the full C2
    MuJoCo checker: robot links and cameras are not included.
    Returns ``{status: PASS|COLLISION_FIXTURE|COLLISION_TABLE, ...}``.
    """
    pose = np.asarray(edge_pose_world, float)
    if points_edge is None:
        points_edge = tool_sample_points_edge(task.tool_profile)
    world = Rotation.from_quat(pose[3:]).apply(points_edge)+pose[:3]
    table_clearance = float(world[:, 2].min()-task.table_top_world_z_m)
    result = {"status": "PASS", "table_clearance_m": table_clearance, "fixture_penetration_m": 0.,
              "fixture": None, "samples": int(len(points_edge)), "tolerance_m": float(tolerance_m),
              "scope": "Sampled blade/handle vs declared fixture boxes and table top; robot links/cameras not checked"}
    T = np.asarray(task.T_world_workpiece, float)
    local = (world-T[:3, 3])@T[:3, :3]
    for box in task.fixture_boxes_local:
        offset = np.abs(local-np.asarray(box.center_m))-np.asarray(box.half_extents_m)
        inside = np.all(offset < 0, axis=1)
        if inside.any():
            depth = float((-offset[inside]).min(axis=1).max())
            if depth > result["fixture_penetration_m"]:
                result.update(fixture_penetration_m=depth, fixture=box.name)
    if result["fixture_penetration_m"] > tolerance_m:
        result["status"] = "COLLISION_FIXTURE"
    elif table_clearance < -tolerance_m:
        result["status"] = "COLLISION_TABLE"
    return result


# ----------------------------------------------------------------------------- process resources

def process_memory(pid=None):
    """Host RSS (MiB, /proc) and this process's GPU memory per device (MiB, nvidia-smi; None if unavailable)."""
    pid = os.getpid() if pid is None else int(pid)
    rss = None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = float(line.split()[1])/1024.
    except OSError:
        pass
    gpu = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            gpu = {}
            for row in out.stdout.strip().splitlines():
                parts = [p.strip() for p in row.split(",")]
                if len(parts) == 3 and parts[0] == str(pid):
                    gpu[parts[1]] = gpu.get(parts[1], 0.)+float(parts[2])
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu = None
    return {"pid": pid, "host_rss_mib": rss, "gpu_used_mib": gpu,
            "gpu_used_mib_total": None if gpu is None else float(sum(gpu.values())), "wall_time": time.time()}


def load_episode_recorder():
    """``EpisodeRecorder`` from ``scripts/run_wp10_live.py`` (read-only reuse of the recorded storage format)."""
    spec = importlib.util.spec_from_file_location("rcs_run_wp10_live_recorder", ROOT/"scripts/run_wp10_live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EpisodeRecorder, module.jsonable


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def find_privileged_keys(tree, path="", keys=PRIVILEGED_KEYS):
    """Paths of every PRIVILEGED_KEYS name (or of ``keys``) anywhere inside a nested observation."""
    found = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            here = f"{path}/{key}"
            if key in keys:
                found.append(here)
            found += find_privileged_keys(value, here, keys)
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            found += find_privileged_keys(value, f"{path}[{i}]", keys)
    return found


def find_privileged_info_keys(info, path=""):
    """Paths of every PRIVILEGED_INFO_KEYS name inside a reset/step info outside ``info["privileged"]``."""
    tree = {key: value for key, value in info.items() if key != "privileged"} if isinstance(info, dict) else info
    return find_privileged_keys(tree, path, PRIVILEGED_INFO_KEYS)


# ----------------------------------------------------------------------------- the wrapper

class PolicyReadyEnv:
    """``cutting_research_v1`` over one CuttingTaskEnv (kinematic or Isaac Sim Native).

    Action (one mapping per 20 ms control step)::

        {"type": "edge_pose",      "target": [x, y, z, qx, qy, qz, qw]}       # absolute WORLD knife edge
        {"type": "edge_delta",     "target": [dx, dy, dz, rx, ry, rz], "frame": "world" | "edge"}
        {"type": "joint_position", "target": q7}                              # live runtimes only
        # optional on every action: "cut_enabled": bool (default False), "cut_index": int in [0, max_cuts),
        #                           "gripper": "hold" | "release" (release needs gripper_control_mode="policy")

    The legacy ``mode`` names are accepted in place of ``type``. ``info`` of every
    step carries ``action_status``, ``action_rejection``, ``action_requested``,
    ``action_executed`` (kind REQUESTED / HOLD_AFTER_REJECTION / NOT_EXECUTED),
    ``action_report``, ``collision_check``, ``action_timestamp_s`` and the episode state.
    """

    def __init__(self, inner, preset=None, *, observation_profile="deployable", recording=None, runtime_profile=None):
        preset = preset or ResearchPreset()
        if observation_profile not in OBSERVATION_PROFILES:
            raise ValueError(f"observation_profile must be one of {OBSERVATION_PROFILES}")
        if inner.observation_profile != "state":
            raise ValueError("PolicyReadyEnv manages cameras itself; open the inner env with observation_profile='state'")
        self.inner, self.preset, self.observation_profile = inner, preset, observation_profile
        self.recording = dict(recording) if recording else None
        if self.recording is not None and "root" not in self.recording:
            raise ValueError("recording needs a 'root' directory")
        # recording["episode_index_start"] (opt-in, default 0): the first episode index of this process, used by the
        # collector's process recycling so episode ids stay continuous across fresh processes
        start = int(self.recording.get("episode_index_start", 0)) if self.recording is not None else 0
        if start < 0:
            raise ValueError("recording episode_index_start must be >= 0")
        self._recorder, self._episode_index, self._recorder_finished = None, start-1, True
        self._camera, self._camera_time = {}, None
        self.action_log, self.episode_receipts = [], []
        self._step_index, self._ready = 0, False
        self._points_edge = None
        self._recorder_class = None
        self.reset_info = None
        self.last_reset_checks = None
        self.runtime_profile = runtime_profile   # WP-11 Phase H4 receipt (None: no profile requested)
        self._pending_switch, self.task_switches = None, []   # WP-12 Phase 6 switch_task receipts (opt-in)

    # -- delegated views
    @property
    def task(self):
        return self.inner.task

    @property
    def backend(self):
        return self.inner.backend

    @property
    def time_s(self):
        return self.inner.time_s

    @property
    def runtime_source(self):
        return self.inner.runtime_source

    @property
    def lifecycle(self):
        return self.inner.lifecycle

    @property
    def episode_state(self):
        return self.inner.episode_state

    @property
    def action_limits(self):
        return self.inner.action_limits or ActionLimits()

    @property
    def max_cuts(self):
        material = self.backend.material
        return int(getattr(material, "max_cuts", 1)) if material is not None else int(getattr(self.task, "max_cuts", 1))

    @property
    def cameras_available(self):
        return self.runtime_source != "kinematic_ideal_tracking" and bool(
            getattr(getattr(self.backend, "config", None), "camera", False))

    @property
    def physical_grasp(self):
        options = getattr(self.backend, "options", getattr(self.backend, "config", None))
        return getattr(options, "tool_attachment_mode", None) == "physical_grasp"

    @property
    def recorder(self):
        return self._recorder

    @property
    def surface_profile(self):
        """Rendered surface profile: ``particle_surface_v1`` (default) or ``cut_aware_surface_v1`` (opt-in)."""
        selection = getattr(self.backend, "surface_selection", None)
        selection = selection() if callable(selection) else None
        return "particle_surface_v1" if selection is None else selection["surface_profile"]

    def surface_provenance(self):
        """Privileged per-extraction surface provenance (cut-aware only; None by default). Debug/oracle data:
        needs observation_profile='privileged_state' and never enters observe() or step info."""
        if self.observation_profile != "privileged_state":
            raise PermissionError("Surface provenance is privileged; use observation_profile='privileged_state'")
        provenance = getattr(self.backend, "surface_provenance", None)
        return provenance() if callable(provenance) else None

    # -- lifecycle
    def reset(self, seed=None, *, controller=None):
        """Direct simulator reset (robot, knife, grasp, material, labels, clocks); never a physical motion."""
        self._close_recorder({"status": "ABANDONED_BY_RESET"})
        start = time.perf_counter()
        observation, info = self.inner.reset(seed=seed)
        self._episode_index += 1
        self._step_index = 0
        self.action_log = []
        self._camera, self._camera_time = {}, None
        self._ready = True
        self._task_info = self._make_task_info()
        self._refresh_cameras(force=True)
        checks = self.reset_checks()
        # The full checks (privileged fields included) stay available to the harness as ``last_reset_checks``
        # (the collector stores them in episode.json); reset info carries only the deployable/bookkeeping part.
        self.last_reset_checks = checks
        self.reset_info = {**info, "preset": self.preset.to_dict(), "observation_profile": self.observation_profile,
                           "episode_index": self._episode_index,
                           "reset_checks": {key: checks[key] for key in DEPLOYABLE_RESET_CHECKS if key in checks},
                           "home_reset": getattr(self.backend, "home", None) is not None,
                           "reset_wall_s": time.perf_counter()-start, "max_cuts": self.max_cuts,
                           "cameras_available": self.cameras_available}
        if self.runtime_profile is not None:
            self.reset_info.update(runtime_profile=copy.deepcopy(self.runtime_profile),
                                   acceptance_valid=bool(self.runtime_profile["acceptance_valid"]))
        if self.surface_profile != "particle_surface_v1":
            # WP-12 Layer 6 (opt-in): the profile NAME is configuration, not oracle state; the default info is unchanged.
            self.reset_info["surface_profile"] = self.surface_profile
        if self._pending_switch is not None:   # WP-12 Phase 6 task_switch_v1 (opt-in): absent without a switch
            self.reset_info["task_switch"] = {**self._pending_switch, "applied_at_episode_index": self._episode_index}
            self._pending_switch = None
        if self.recording is not None:
            self._open_recorder(controller or self.recording.get("controller", "external"))
        info = copy.deepcopy(self.reset_info)
        if self.observation_profile == "privileged_state":
            info["privileged"] = {**self.privileged(), "reset_checks": copy.deepcopy(checks)}
        return self.observe(), info

    def switch_task(self, task=None, *, scene=None, goal=None, planner_options=None, tool_profile=None,
                    particle_spacing_m=None, seed=None):
        """Opt-in (WP-12 Phase 6, ``task_switch_v1``): replace the compiled task between episodes in one process.

        Pass a compiled ``ConfiguredTaskSpec`` or ``scene=`` + ``goal=`` (compiled exactly as ``open_policy_env``
        does, with the env's runtime profile applied and checked). The switch only rewrites the backend's task
        configuration (kinematic: the configured task and the solver fields it declares; Isaac Sim: ``task_id``,
        ``configured_task`` and ``max_cuts``); the runtime builds material, labels, surface, contacts and the
        grasp afresh at the next ``reset()``, which is required before any step. Compatibility rules: same
        control dt; Isaac Sim also the same grid dx and tool profile (the held knife is not exchanged).
        The next reset info carries the receipt under ``"task_switch"``; without a switch nothing changes.
        """
        from .scene_config import ConfiguredTaskSpec
        if (scene is None) != (goal is None) or (task is None) == (scene is None):
            raise ValueError("switch_task needs a compiled task or scene+goal (one of them)")
        self._close_recorder({"status": "ABANDONED_BY_TASK_SWITCH"})
        profile = None
        if self.runtime_profile is not None:
            from . import runtime_profiles as rp
            profile = rp.runtime_profile(self.runtime_profile["profile_id"])
            if particle_spacing_m is not None:
                rp.check_caller_overrides(profile, particle_spacing_m=particle_spacing_m)
            if scene is not None:
                scene = rp.profile_scene(scene, profile)
        if task is None:
            task = compile_policy_task(scene, goal, tool_profile=tool_profile, planner_options=planner_options,
                                       particle_spacing_m=particle_spacing_m,
                                       seed=int(getattr(self.task, "seed", 42) if seed is None else seed))
        if not isinstance(task, ConfiguredTaskSpec):
            raise TypeError("switch_task expects a compiled scene_config.ConfiguredTaskSpec")
        old, backend = self.task, self.backend
        if abs(float(task.control_dt_s)-float(old.control_dt_s)) > 1e-12:
            raise ValueError("switch_task keeps the control dt of the open env")
        receipt = {"version": TASK_SWITCH_VERSION, "from_task_id": old.task_id, "to_task_id": task.task_id,
                   "to_config_sha256": getattr(task, "config_sha256", None), "runtime_source": self.runtime_source,
                   "requested_at_episode_index": self._episode_index}
        if profile is not None:
            from . import runtime_profiles as rp
            material_dt = float(getattr(backend.config, "material_dt"))
            self.runtime_profile = rp.profile_receipt(profile, rp.check_task(profile, task, material_dt))
            receipt["runtime_profile"] = self.runtime_profile["profile_id"]
        if self.runtime_source == "kinematic_ideal_tracking":
            config = backend.config
            changes = dict(task_id=task.task_id, max_cuts=int(task.max_cuts), cutting_law=task.cutting_law,
                           solver=task.solver_module if task.solver_module in ("tlmpm_v5", "tlmpm_v7", "tlmpm_v7_1") else None,
                           slots_per_node=task.scene["solver"].get("slots_per_node"))
            if changes["solver"] is None:
                changes["cut_events"] = None
                if config.surface_profile is not None and config.surface_profile != "particle_surface_v1":
                    raise ValueError("The env's cut-aware surface needs a tlmpm_v5 task")
            from .materials_wp12 import admit, is_profile
            if is_profile(task.scene["workpiece"]["material_profile"]):
                admit(task.scene["workpiece"]["material_profile"], task.grid_dx_m, config.material_dt,
                      task.particle_spacing_m)
            new = replace(config, **changes)
            backend.config = backend.options = new
            backend.configured_task = task
            backend.task = backend._task(int(task.seed))
            backend.material = None   # the next reset builds the material of the new task
        else:
            config = backend.config
            if abs(float(task.grid_dx_m)-float(config.dx)) > 1e-12:
                raise ValueError(f"Isaac Sim switch_task keeps the material grid dx {config.dx}")
            if task.tool_profile_id != old.tool_profile_id:
                raise ValueError("Isaac Sim switch_task keeps the held tool profile")
            from .scene_config import configured_task_payload
            backend.config = replace(config, task_id=task.task_id, configured_task=configured_task_payload(task),
                                     max_cuts=int(task.max_cuts))
            backend.task = task
        inner = self.inner
        inner.task = backend.task
        if hasattr(task, "action_limits"):
            inner.action_limits = task.action_limits()
        inner._ready, self._ready = False, False
        self._points_edge = None
        self._pending_switch = receipt
        self.task_switches.append(receipt)
        return copy.deepcopy(receipt)

    def _open_recorder(self, controller):
        if self._recorder_class is None:
            self._recorder_class, _ = load_episode_recorder()
        root = Path(self.recording["root"])
        root.mkdir(parents=True, exist_ok=True)
        self._recorder = self._recorder_class(root/f"episode_{self._episode_index:03d}", self.inner, controller,
                                              int(self.recording.get("every", 10)))
        self._recorder_finished = False

    def finish_recording(self, result):
        """Close the current episode recording with ``result`` (merged into its result.json); returns its metrics."""
        if self._recorder is None or self._recorder_finished:
            return None
        metrics = self._recorder.finish(jsonable(result))
        self._recorder_finished = True
        return metrics

    def _close_recorder(self, result):
        if self._recorder is not None and not self._recorder_finished:
            try:
                self.finish_recording(result)
            except Exception as error:  # an unusable partial recording is still kept on disk
                (Path(self._recorder.output)/"recorder_close_error.txt").write_text(repr(error))
                self._recorder_finished = True

    def close(self):
        self._close_recorder({"status": "ABANDONED_BY_CLOSE"})
        self.inner.close()
        self._ready = False

    def _require_ready(self):
        self.inner._require_ready()
        if not self._ready:
            raise RuntimeError("Call reset on the environment first")

    # -- task information (a-priori, never oracle state)
    def _make_task_info(self):
        task = self.task
        goal = getattr(task, "goal", None)
        if callable(goal) or not isinstance(goal, dict):
            goal = {"type": getattr(task, "goal_kind", None)}
        scene = getattr(task, "scene", None)
        return {"task_id": task.task_id, "goal": copy.deepcopy(goal), "goal_text": goal.get("description", ""),
                "scene_id": scene.get("scene_id") if isinstance(scene, dict) else None,
                "tool_profile_id": getattr(task, "tool_profile_id", None), "duration_s": float(task.duration_s),
                "control_dt_s": float(task.control_dt_s), "max_cuts": self.max_cuts,
                "home_declared": getattr(task, "home", None) is not None}

    def task_spec(self):
        """Full a-priori task description: scene config, goal config and action interface (JSON-serializable)."""
        task = self.task
        spec = {**self._task_info, "scene": getattr(task, "scene", None), "action_types": list(ACTION_TYPES),
                "action_limits": self.action_limits.to_dict(), "preset": self.preset.to_dict(),
                "world_from_workpiece_xyzw": list(task.world_from_workpiece_xyzw)}
        if self.runtime_profile is not None:
            spec.update(runtime_profile=self.runtime_profile["profile_id"],
                        acceptance_valid=bool(self.runtime_profile["acceptance_valid"]))
        return jsonable(spec)

    # -- observations
    def _refresh_cameras(self, force=False):
        if not self.cameras_available:
            return
        if not force and self._step_index % self.preset.camera_every:
            return
        from .research_env import vision_observation
        backend = self.backend
        frames = backend.frames if hasattr(backend, "measured") else backend.capture()[0]
        block = vision_observation(frames, backend.calibration, self.time_s, "vision_only",
                                   include_global_camera=self.preset.include_global_camera)
        self._camera = {key: block[key] for key in ("rgb", "depth_m", "valid_mask", "calibration")}
        self._camera_time = float(block["timestamp_s"])
        self.inner._frames = frames   # fresh frames of this step: the recorder reuses them instead of re-rendering

    def observe(self):
        """Deployable observation. Never contains material, labels, components, forces or tool truth."""
        self._require_ready()
        inner = self.inner.observe()
        proprio = inner["proprio"]
        if "hand_pose" in proprio:
            ee = np.asarray(proprio["hand_pose"], float).copy()
        else:
            from .scene_frames import invert_transform, matrix_pose, pose_matrix
            ee = matrix_pose(pose_matrix(proprio["tool_pose"])@invert_transform(pose_matrix(self.task.hand_from_edge_xyz_xyzw)))
        finger = proprio.get("finger_q")
        gripper = {"available": finger is not None, "control_mode": self.preset.gripper_control_mode,
                   "command": getattr(self.backend, "gripper_command", None) if finger is not None else None,
                   "width_m": float(np.asarray(finger, float)[0].sum()) if finger is not None else None}
        observation = {"timestamp_s": float(inner["timestamp_s"]), "proprio": proprio,
                       "ee_pose_world_xyzw": ee, "edge_command_world_xyzw": np.asarray(self.inner._commanded, float).copy(),
                       "gripper": gripper,
                       "cameras": copy.deepcopy(self._camera), "camera_timestamp_s": self._camera_time,
                       "camera_age_s": None if self._camera_time is None else float(self.time_s-self._camera_time),
                       "task": {**copy.deepcopy(self._task_info),
                                "time_remaining_s": float(max(0., self.task.duration_s-self.time_s))},
                       "episode": {**self.lifecycle.snapshot(), "control_step": self._step_index,
                                   "episode_index": self._episode_index}}
        return observation

    def privileged(self, *, components=False):
        """Simulator-only debug state; available only with observation_profile='privileged_state'."""
        if self.observation_profile != "privileged_state":
            raise PermissionError("Privileged state needs observation_profile='privileged_state' (explicit opt-in)")
        self._require_ready()
        backend, material = self.backend, self.backend.material
        state = material.history_state() if hasattr(material, "history_state") else material.world_state()
        measured = backend.measured() if hasattr(backend, "measured") else backend.proprioception()
        diagnostics = getattr(backend, "last_diagnostics", {}) or {}
        out = {"timestamp_s": self.time_s, "privileged": True,
               "material_state": {k: np.asarray(state[k]).copy() for k in ("positions", "velocities", "codes")},
               "cut_codes": np.asarray(state["codes"]).copy(),
               "code_encoding": getattr(material, "code_encoding", "ternary"), "max_cuts": self.max_cuts,
               "oracle_geometry": {"reference_positions_world": np.asarray(material.world_points).copy(),
                                   "reference_indices": np.asarray(material.lattice).copy(),
                                   "mass_kg": np.asarray(material.masses).copy(),
                                   "T_world_workpiece": np.asarray(self.task.T_world_workpiece).copy()},
               "contact_wrench_world": np.asarray(getattr(material, "pending_wrench", np.zeros(6)), float).copy(),
               "tool_truth": {k: np.asarray(measured[k]).copy() for k in
                              ("tool_pose", "tool_body_pose_xyzw", "tool_linear_velocity_world_m_s",
                               "tool_angular_velocity_world_rad_s") if k in measured},
               "grasp_status": copy.deepcopy(diagnostics.get("grasp_status"))}
        if components:
            out["component_labels"] = self.component_labels()
        return out

    def component_labels(self):
        """(count, per-particle labels) of the actual MPM stencil graph (oracle; privileged)."""
        from .cut_topology import mechanical_components, mechanical_components_packed
        material = self.backend.material
        bed = material.bed
        codes = np.asarray(material.history_state()["codes"])
        if getattr(material, "code_encoding", "ternary") == "packed2":
            count, labels = mechanical_components_packed(bed.nodes.numpy(), bed.weights.numpy(), codes)
        else:
            count, labels = mechanical_components(bed.nodes.numpy(), bed.weights.numpy(), codes,
                                                  cuts=max(1, int(material.max_cuts)))
        return {"count": int(count), "labels": np.asarray(labels)}

    def render(self, camera="global", mode="rgb_array"):
        return self.inner.render(camera, mode)

    # -- actions
    def _normalize(self, action):
        if not isinstance(action, dict):
            raise ActionRejected("INVALID_SCHEMA", "An action is a mapping")
        unknown = set(action)-ACTION_KEYS
        if unknown:
            raise ActionRejected("INVALID_SCHEMA", f"Unknown action keys {sorted(unknown)}")
        if ("type" in action) == ("mode" in action) and not (set(action) <= {"gripper"}):
            raise ActionRejected("INVALID_SCHEMA", "Give exactly one of 'type' or the legacy 'mode'")
        gripper = action.get("gripper")
        if gripper is not None and gripper not in GRIPPER_COMMANDS:
            raise ActionRejected("INVALID_SCHEMA", f"gripper must be one of {GRIPPER_COMMANDS}")
        if set(action) <= {"gripper"}:
            if gripper is None:
                raise ActionRejected("INVALID_SCHEMA", "Empty action")
            return None, gripper
        mode = ACTION_TYPES.get(action["type"]) if "type" in action else action["mode"]
        if mode not in ACTION_TYPES.values():
            raise ActionRejected("INVALID_SCHEMA", f"Unknown action type/mode {action.get('type', action.get('mode'))!r}")
        if "frame" in action and (mode != DELTA_MODE or action["frame"] not in DELTA_FRAMES):
            raise ActionRejected("INVALID_SCHEMA", "frame is 'world' or 'edge' and only for edge_delta")
        try:
            target = np.asarray(action.get("target"), float)
        except (TypeError, ValueError):
            raise ActionRejected("INVALID_TARGET", "target is not numeric") from None
        shape = {ACTION_MODES[0]: (7,), DELTA_MODE: (6,), ACTION_MODES[1]: (7,)}[mode]
        if target.shape != shape or not np.isfinite(target).all():
            raise ActionRejected("INVALID_TARGET", f"{mode} target needs a finite {shape} vector")
        if mode == ACTION_MODES[0] and abs(np.linalg.norm(target[3:])-1.) > 1e-6:
            raise ActionRejected("INVALID_TARGET", "edge_pose quaternion must be unit XYZW")
        enabled = action.get("cut_enabled", self.preset.default_cut_enabled)
        if not isinstance(enabled, (bool, np.bool_)):
            raise ActionRejected("INVALID_SCHEMA", "cut_enabled must be a Boolean")
        command = {"mode": mode, "target": target.copy(), "cut_enabled": bool(enabled)}
        if "cut_index" in action:
            index = action["cut_index"]
            if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                    or not 0 <= index < self.max_cuts):
                raise ActionRejected("CUT_INDEX_OUT_OF_RANGE", f"cut_index must be an integer in [0, {self.max_cuts})")
            command["cut_index"] = int(index)
        if "frame" in action:
            command["frame"] = action["frame"]
        elif mode == DELTA_MODE:
            command["frame"] = self.preset.delta_frame
        return command, gripper

    def _check_gripper(self, gripper):
        if gripper is None:
            return None
        if not self.physical_grasp:
            if gripper == "hold":
                return None   # nothing to command: the kinematic edge has no gripper
            raise ActionRejected("GRIPPER_UNSUPPORTED", "This runtime has no physical gripper")
        if gripper == "release" and self.preset.gripper_control_mode != "policy":
            raise ActionRejected("GRIPPER_LOCKED", "gripper_control_mode='hold' keeps the tool grasped")
        return gripper

    def _resolve(self, command):
        if command is None:
            return None, None
        if command["mode"] == ACTION_MODES[1]:
            if self.runtime_source == "kinematic_ideal_tracking":
                raise ActionRejected("UNSUPPORTED_BY_RUNTIME", "The kinematic backend has no arm joints")
            ik = getattr(self.backend, "ik", None)
            lower = np.asarray(getattr(ik, "lower", PANDA_LOWER), float)[:7]
            upper = np.asarray(getattr(ik, "upper", PANDA_UPPER), float)[:7]
            q = command["target"]
            if np.any(q < lower) or np.any(q > upper):
                raise ActionRejected("JOINT_LIMIT", f"joints {np.flatnonzero((q < lower) | (q > upper)).tolist()} outside limits")
            return command, None
        resolved, report = self.inner._resolve(dict(command))
        resolved.pop("frame", None)
        if report is not None:
            report["resolved_target_world_xyz_xyzw"] = np.asarray(resolved["target"]).tolist()
        return resolved, report

    def _collision(self, command):
        if command is None or command["mode"] != ACTION_MODES[0] or not self.preset.collision_check:
            return {"status": "NOT_RUN", "reason": "joint target or check disabled" if command else "gripper-only action"}
        if self._points_edge is None:
            self._points_edge = tool_sample_points_edge(self.task.tool_profile)
        result = check_tool_collision(self.task, command["target"], tolerance_m=self.preset.collision_tolerance_m,
                                      points_edge=self._points_edge)
        if result["status"] != "PASS":
            raise ActionRejected(result["status"], f"{result['status']} (fixture {result['fixture']}, "
                                 f"penetration {result['fixture_penetration_m']*1e3:.2f} mm, "
                                 f"table clearance {result['table_clearance_m']*1e3:.2f} mm)")
        return result

    def _hold_command(self):
        return {"mode": ACTION_MODES[0], "target": np.asarray(self.inner._commanded, float).copy(), "cut_enabled": False}

    def step(self, action, *, annotation=None):
        """One 20 ms control step with an explicit action status; returns (obs, reward, terminated, truncated, info)."""
        self._require_ready()
        inner = self.inner
        if inner.time_s >= self.task.duration_s-1e-9 or inner._terminated:
            raise RuntimeError("Episode ended; call reset")
        self.lifecycle.require(("READY", "RUNNING"), "step")
        t_in, wall = self.time_s, time.perf_counter()
        before = self.observe() if self._recorder is not None else None
        requested = jsonable(action) if isinstance(action, dict) else {"invalid": repr(action)}
        rejection, report, collision, command, gripper = None, None, {"status": "NOT_RUN"}, None, None
        try:
            command, gripper = self._normalize(action)
            gripper = self._check_gripper(gripper)
            command, report = self._resolve(command)
            collision = self._collision(command)
        except ActionRejected as error:
            rejection = error
            if error.reason.startswith("COLLISION"):
                collision = {"status": error.reason, "detail": error.detail}
        result, kind, executed, ik_info = None, None, None, None
        if rejection is None:
            payload = dict(command) if command is not None else {}
            if gripper is not None:
                payload["gripper"] = gripper
            if not payload:   # gripper "hold" on a gripperless runtime: an explicit hold of the commanded edge
                payload, kind = self._hold_command(), "HOLD_GRIPPER_NOOP"
            gripper_before = getattr(self.backend, "gripper_command", None)
            try:
                result = inner.step(payload)
                executed, kind = payload, kind or "REQUESTED"
            except IKFailure as failure:
                rejection, ik_info = ActionRejected("IK_UNREACHABLE", str(failure).split(":")[0]), failure.info
                # The inner env sets the gripper before the runtime IK fails (no physics step ran): undo it, so no
                # part of a rejected action executes (WP-11 review finding 10).
                if gripper is not None and gripper_before is not None and \
                        getattr(self.backend, "gripper_command", None) != gripper_before:
                    self.backend.set_gripper(gripper_before)
        if rejection is not None:
            if self.preset.rejection_policy == "no_step":
                kind = "NOT_EXECUTED"
                observation = self.observe()
                info = {"task_id": self.task.task_id, "timestamp_s": self.time_s, "runtime_source": self.runtime_source,
                        "termination_reason": None, "ik_status": "rejected" if ik_info else "not_run",
                        "episode_state": self.lifecycle.snapshot()}
                result = (observation, 0., False, False, info)
            else:
                hold = self._hold_command()
                try:
                    result = inner.step(hold)
                    executed, kind = hold, "HOLD_AFTER_REJECTION"
                except IKFailure:
                    q = np.asarray(inner._proprio["q"], float)[0]
                    hold = {"mode": ACTION_MODES[1], "target": q, "cut_enabled": False}
                    result = inner.step(hold)
                    executed, kind = hold, "JOINT_HOLD_AFTER_REJECTION"
        if kind != "NOT_EXECUTED":
            self._step_index += 1
            self._refresh_cameras()
        _, reward, terminated, truncated, info = result
        clamped = bool(report and any(report.get(k, False) for k in ("translation_clamped", "rotation_clamped",
                                                                      "workspace_clamped")))
        status = ("REJECTED_"+rejection.reason) if rejection is not None else ("CLAMPED" if clamped else "ACCEPTED")
        info = {**info, "action_status": status,
                "action_rejection": None if rejection is None else {"reason": rejection.reason, "detail": rejection.detail},
                "action_requested": requested,
                "action_executed": None if executed is None else {"kind": kind, **jsonable(executed)},
                "action_report": report, "collision_check": collision, "control_step": self._step_index,
                "action_timestamp_s": t_in, "timestamp_s": self.time_s, "preset": PRESET_ID}
        if kind == "NOT_EXECUTED":
            info["action_executed"] = {"kind": kind}
        if ik_info is not None:
            info.update(ik_status="rejected", ik_info=jsonable(ik_info))
        observation = self.observe()
        self.action_log.append({"control_step": self._step_index, "action_timestamp_s": t_in,
                                "timestamp_s": self.time_s, "status": status, "executed_kind": kind,
                                "mode": None if executed is None else executed.get("mode"),
                                "cut_enabled": None if executed is None else bool(executed.get("cut_enabled", False)),
                                "cut_index": None if executed is None else executed.get("cut_index"),
                                "terminated": bool(terminated), "truncated": bool(truncated),
                                "wall_s": time.perf_counter()-wall})
        if self._recorder is not None and kind != "NOT_EXECUTED":
            phase = (annotation or {}).get("policy_phase")
            self._recorder.step({k: v for k, v in executed.items()}, (observation, reward, terminated, truncated,
                                jsonable({k: v for k, v in info.items() if k != "privileged"})),
                                before, time.perf_counter()-wall, policy_phase=phase,
                                policy_tracking=(annotation or {}).get("policy_tracking"))
        # ``grasp_status`` (contact-based grasp monitor, simulator truth) is privileged (WP-11 review finding 9):
        # it leaves the returned info and is under info["privileged"]["grasp_status"] with
        # observation_profile="privileged_state". The recording above keeps the full inner info (actions.jsonl is
        # privileged storage) and the termination reason stays in info["termination_reason"]/"episode_state".
        info = {key: value for key, value in info.items() if key not in PRIVILEGED_STEP_INFO}
        if self.observation_profile == "privileged_state":
            info["privileged"] = self.privileged()
        return observation, reward, terminated, truncated, info

    # -- episode end
    def end_task(self, reason="CONTROLLER_DONE"):
        return self.inner.end_task(reason)

    def evaluate(self, **options):
        """The task's own goal/oracle evaluator (read-only); never an observation.

        With a runtime profile the result also carries ``runtime_profile`` and ``acceptance_valid``."""
        result = self.inner.evaluate(**options)
        if self.runtime_profile is not None and isinstance(result, dict):
            result = {**result, "runtime_profile": self.runtime_profile["profile_id"],
                      "acceptance_valid": bool(self.runtime_profile["acceptance_valid"])}
        return result

    def return_home(self, **options):
        """Optional physical return-to-home (demonstrations); recorded steps when recording is on."""
        on_step = options.pop("on_step", None)
        recorder = self._recorder

        def record(action, result, before, label):
            if recorder is not None and not self._recorder_finished:
                observation, reward, terminated, truncated, info = result
                recorder.step(action, (observation, reward, terminated, truncated, jsonable(info)), before, None,
                              policy_phase=f"RETURN_HOME:{label}")
            if on_step is not None:
                on_step(action, result, before, label)
        return self.inner.return_home(on_step=record, **options)

    # -- reliability evidence
    def reset_checks(self):
        """Post-reset state isolation summary (material, labels, clocks, history, tool, grasp); no arrays."""
        inner, backend, material = self.inner, self.backend, self.backend.material
        state = material.history_state() if hasattr(material, "history_state") else material.world_state()
        codes, positions, velocities = (np.asarray(state[k]) for k in ("codes", "positions", "velocities"))
        reference = np.asarray(material.world_points, float)
        measured = backend.measured() if hasattr(backend, "measured") else backend.proprioception()
        grasp = getattr(backend, "grasp_reset_receipt", None)
        checks = {
            "time_zero": bool(abs(self.time_s) < 1e-12),
            "labels_clean": bool(np.count_nonzero(codes) == 0),
            "material_at_reference_max_m": float(np.abs(positions-reference).max()),
            "material_velocity_max_m_s": float(np.abs(velocities).max()),
            "oracle_history_single_sample": bool(len(inner._history) == 1),
            "lifecycle_ready_after_reset": bool(self.lifecycle.state == "READY" and len(self.lifecycle.history) == 2),
            "ik_failures_zero": bool(inner._ik_failures == 0), "not_terminated": bool(not inner._terminated),
            "control_steps_zero": bool(inner._control_steps_count == 0 and self._step_index == 0),
            "action_log_empty": bool(not self.action_log),
            "observation_timestamp_matches": bool(abs(float(measured["timestamp_s"])-self.time_s) < 1e-9),
            "tool_pose_world_xyzw": np.asarray(measured.get("tool_body_pose_xyzw", measured["tool_pose"]), float).tolist(),
            "edge_pose_world_xyzw": np.asarray(measured["tool_pose"], float).tolist(),
            "tool_linear_speed_m_s": (float(np.linalg.norm(measured["tool_linear_velocity_world_m_s"]))
                                      if "tool_linear_velocity_world_m_s" in measured else 0.),
            "tool_angular_speed_rad_s": (float(np.linalg.norm(measured["tool_angular_velocity_world_rad_s"]))
                                         if "tool_angular_velocity_world_rad_s" in measured else 0.),
            "q_active": np.asarray(measured["q"], float)[0].tolist(),
            "grasp_reset_status": None if grasp is None else grasp.get("status"),
            "camera_timestamp_matches": (None if not self.cameras_available else
                                         bool(self._camera_time is not None and abs(self._camera_time-self.time_s) < 1e-9)),
            "material_codes_sha256": __import__("hashlib").sha256(np.ascontiguousarray(codes).tobytes()).hexdigest(),
            "particles": int(len(codes)),
        }
        return checks


# ----------------------------------------------------------------------------- open_cutting_env(preset=...)

def research_preset(**overrides):
    return replace(ResearchPreset(), **overrides) if overrides else ResearchPreset()


def compile_policy_task(scene, goal, *, tool_profile=None, planner_options=None, particle_spacing_m=None, seed=42,
                        solver_profile=None):
    """compile_task(scene, goal) with an optional tool profile override (written into the scene config)."""
    from .scene_config import compile_task, resolve_scene
    scene = resolve_scene(scene)
    if tool_profile is not None:
        scene = copy.deepcopy(scene)
        scene["tool"]["profile_id"] = str(tool_profile)
    kwargs = {} if particle_spacing_m is None else {"particle_spacing_m": particle_spacing_m}
    return compile_task(scene, goal, planner_options=planner_options, seed=seed, solver_profile=solver_profile, **kwargs)


@contextmanager
def open_policy_env(task_id=None, *, runtime, output_dir, seed=42, preset=PRESET_ID, task=None, scene=None, goal=None,
                    tool_profile=None, planner_options=None, particle_spacing_m=None, recording=None, solver_profile=None,
                    observation_profile="deployable", preset_overrides=None, runtime_profile=None, **backend_options):
    """The ``open_cutting_env(..., preset=...)`` path: compile, open the inner env, wrap it, close everything.

    ``runtime_profile`` (WP-11 Phase H4, opt-in): a declared profile id of
    ``configs/wp11/runtime_profiles_v1.json`` (``full_physics_v1`` | ``research_fast_v1``).
    It sets the solver resolution, material dt, Native stage reset, camera and recording
    defaults, refuses caller overrides of the numerics it fixes, and labels every reset
    info / evaluation with ``runtime_profile`` and ``acceptance_valid``. None changes nothing."""
    from .live_env import open_cutting_env
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose one of {PRESETS}")
    if runtime not in ("kinematic", "isaac_sim"):
        raise NotImplementedError("cutting_research_v1 is implemented for runtime='kinematic' and 'isaac_sim' "
                                  "(Isaac Lab: NOT_RUN)")
    if (scene is None) != (goal is None):
        raise ValueError("scene and goal go together")
    profile = None
    if runtime_profile is not None:
        from . import runtime_profiles as rp
        profile = rp.runtime_profile(runtime_profile)
        rp.check_caller_overrides(profile, particle_spacing_m=particle_spacing_m, backend_options=backend_options)
        if scene is not None:
            scene = rp.profile_scene(scene, profile)
        elif profile.get("solver_override"):
            raise ValueError(f"runtime_profile {runtime_profile!r} overrides the scene solver; pass scene= and goal=")
        backend_options["material_dt"] = float(profile["material_dt_s"])
        if profile.get("history_stride") is not None:
            backend_options.setdefault("history_stride", int(profile["history_stride"]))
        if runtime == "isaac_sim":
            for key, value in (profile.get("native") or {}).items():
                backend_options.setdefault(key, value)
        preset_overrides = {**(profile.get("preset_overrides") or {}), **(preset_overrides or {})}
        if recording is not None:
            recording = {**(profile.get("recording") or {}), **dict(recording)}
    if scene is not None:
        if task is not None:
            raise ValueError("Pass a compiled task or scene+goal, not both")
        task = compile_policy_task(scene, goal, tool_profile=tool_profile, planner_options=planner_options,
                                   particle_spacing_m=particle_spacing_m, seed=seed, solver_profile=solver_profile)
    elif tool_profile is not None or planner_options is not None or solver_profile is not None:
        raise ValueError("tool_profile/planner_options apply to a scene+goal compile")
    settings = research_preset(**(preset_overrides or {}))
    limits = backend_options.pop("action_limits", None)
    if limits is None and task is not None and hasattr(task, "action_limits"):
        limits = task.action_limits()
    options = dict(action_limits=limits or ActionLimits(), delta_frame=settings.delta_frame, ik_failure="raise")
    if runtime == "isaac_sim":
        options.update(tool_attachment_mode=settings.tool_attachment_mode, tool_init_mode=settings.tool_init_mode,
                       camera=backend_options.pop("camera", True))
    receipt = None
    if profile is not None:
        from . import runtime_profiles as rp
        receipt = rp.profile_receipt(profile, rp.check_task(profile, task, backend_options["material_dt"]))
        if runtime == "isaac_sim":
            backend_options.setdefault("dx", float(task.grid_dx_m))   # Native material grid = the task grid
    options.update(backend_options)
    with open_cutting_env(task_id or "CUT-01", runtime=runtime, output_dir=output_dir, seed=seed, task=task,
                          observation_profile="state", **options) as inner:
        env = PolicyReadyEnv(inner, settings, observation_profile=observation_profile, recording=recording,
                             runtime_profile=receipt)
        try:
            yield env
        finally:
            env._close_recorder({"status": "ABANDONED_BY_CLOSE"})
