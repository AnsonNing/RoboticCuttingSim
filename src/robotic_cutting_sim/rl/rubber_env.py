"""``RubberVectorEnv``: N rubber strip envs in one ``BatchedRubberMaterial`` (P2-6, ADR 0023 §23.2 decision 7).

Opened by ``open_vector_env(num_envs, task=RubberStripTaskFamily(...), backend="kinematic_grip_batched")``
(runtime ``rl_gpu_headless``). Task ``rubber_strip_drag_v1`` (``rl.rubber_family``): a case-A strip on the table,
dragged by a kinematic grip toward a tip goal. One process, one batch, one CUDA graph set per batch shape; nothing
here is a Python loop over single environments.

```python
with open_vector_env(num_envs=8, task=RubberStripTaskFamily(num_envs=8, slot_seed=0),
                     backend="kinematic_grip_batched", device="cuda") as env:
    obs, info = env.reset(seeds=seeds)                       # obs["policy"]: float32 [N, 78] on the device
    obs, reward, terminated, truncated, info = env.step(a)   # a: float32 [N, 6]
    obs, info = env.reset(env_ids=ids, seeds=s)              # masked partial reset with resampling
    results = env.evaluate(env_ids)                          # rubber_tracking_eval_v1 (host)
```

Step (device tensors, no per-env host loop, no host read except strict mode's validity count):

1. ``grip_delta_v1``: ``a [N, 6]`` clipped to [-1, 1] -> a translation (<= 5 mm) and an axis-angle rotation
   (<= 0.05 rad, world axes, about the grip origin) of the float64 commanded grip pose; a pose that leaves the
   workspace box, lowers the jaw below the clearance or tilts the grip too far is held (``action_status`` 2). During
   the ``preload`` steps the seeded smootherstep path replaces the action (``drive`` PRELOAD, ``action_status`` 3).
2. Ideal tracking: the END pose of the control step is the commanded pose. Its quaternion and translation
   ``b = p - R(q) o`` are written into Warp staging arrays (Torch views) and staged by
   ``BatchedRubberMaterial.advance_device`` (``set_group_motions_device`` + ``set_traction_device``; a kernel does the
   host path's hemisphere alignment), which runs the substeps (CUDA graph chunks), wrenches, diagnostics, the S1
   self-penetration detector and the validity kernel.
3. ``termination_reason`` (``rl.termination``): any rubber invalid bit (NONFINITE 1, J_DOMAIN 2, CFL 4,
   SELF_CONTACT_UNRESOLVED 8) sets ``TERM_MATERIAL_INVALID`` (sticky); ``TERM_TRUNCATED`` follows the horizon.
4. ``rubber_tip_tracking_v1`` reward (float64 on the device, PROVISIONAL_SHAPING_NOT_SUCCESS), observations
   (``policy`` 78 dims deployable; ``critic`` only with ``observation_profile="privileged_state"``).

Resets are host-staged (``BatchedRubberMaterial.reset`` with the sampled ``MaterialValues``; a density change
rebuilds the F8 block inverses on the host) and their wall time is recorded (``reset_walls_s``). Success comes only
from ``evaluate()`` (``rubber_tracking_eval_v1``). Stream policy as ADR 0021 §7: Warp launches run on Torch's current
stream. Refused: cameras / sensors (``UNSUPPORTED_SENSOR``), recorders, ``autoreset="same_step"``
(``UNSUPPORTED_OPTION``), a slot change on reset (``UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET``; ``rebuild``),
friction outside the envelope (``UNSUPPORTED_FRICTION_OUT_OF_ENVELOPE``). Material status:
``SIMULATOR_ONLY_UNCALIBRATED``; coupling ``kinematic_grip_ideal_tracking``; perception ``ideal_marker_tracking``.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from .rubber_family import RubberStripTaskFamily, UnsupportedRubberConfiguration

ENV_VERSION = "rl_rubber_vector_env_v1"
API_REVISION = "p2_6"
RUNTIME = "rl_gpu_headless"
BACKEND = "kinematic_grip_batched"
ACTION_SPEC = "grip_delta_v1"
REWARD = "rubber_tip_tracking_v1"
EVALUATOR = "rubber_tracking_eval_v1"
MATERIAL = "task_declared"
OBSERVATION_PROFILES = ("state", "privileged_state")
# F2 (P6.7 review): step-info keys derived from oracle material state (validity bits and the termination / penalty
# flags computed from them); a deployable policy must not consume them. Listed in env.metadata["privileged_info_keys"]
# as on the Phase 1 backends (rl.vector_contract.PRIVILEGED_INFO_KEYS names the same three keys).
PRIVILEGED_INFO_KEYS = ("invalid_reason", "termination_reason", "invalid_penalty_applied")
DRIVE_ACTION, DRIVE_PRELOAD = 0, 2
DRIVE_NAMES = {DRIVE_ACTION: "policy_action_grip_delta_v1", DRIVE_PRELOAD: "PRELOAD"}
STATUS_APPLIED, STATUS_CLIPPED, STATUS_HELD, STATUS_PRELOAD = 0, 1, 2, 3
TIP_MARKER = 8
POLICY_TERMS = (("grip_position_m", 3), ("grip_quaternion_xyzw", 4), ("grip_twist_commanded", 6),
                ("time_remaining_fraction", 1), ("previous_action", 6), ("marker_positions_m", 27),
                ("marker_velocities_m_s", 27), ("goal_m", 3), ("preload_flag", 1))
CRITIC_TERMS = (("mu0_mpa", 1), ("density_t_m3", 1), ("friction_table", 1), ("prony_g_scale", 1),
                ("prony_tau_scale", 1), ("tip_traction_n", 3), ("grip_wrench", 6), ("plane_wrench", 6),
                ("psi_eq_j", 1), ("kinetic_j", 1), ("gravity_energy_j", 1), ("jbar_min", 1), ("jbar_max", 1),
                ("cfl_eff_max", 1), ("contact_impulse_events", 1), ("max_penetration_m", 1))
POLICY_DIM = sum(n for _, n in POLICY_TERMS)          # 78
CRITIC_DIM = POLICY_DIM+sum(n for _, n in CRITIC_TERMS)   # 106
SOURCE_FILES = ("rl/rubber_env.py", "rl/rubber_family.py", "rl/rubber_contract.py", "rubber_contact_envelope.py",
                "batched/rubber_material_batch.py", "batched/rubber_batch.py", "batched/rubber_contact_batch.py",
                "batched/rubber_layout.py", "live_rubber_material.py", "rubber_contact.py",
                "tlmpm_rubber_contact_v1.py", "tlmpm_rubber_v1.py", "tlmpm_rubber_visco_v1.py",
                "rubber_self_penetration.py", "rubber_material.py", "rubber_scene.py", "rl/termination.py")
_PKG = Path(__file__).resolve().parents[1]


def source_sha256():
    return {name: hashlib.sha256((_PKG/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


# ----------------------------------------------------------------------------- quaternion helpers (XYZW, float64)

def qmul(a, b):
    import torch
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw,
                        aw*bw-ax*bx-ay*by-az*bz), -1)


def qconj(q):
    import torch
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def qexp(v):
    """Unit quaternion of the rotation vector v [..., 3]."""
    import torch
    th = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    half = 0.5*th
    k = torch.where(th > 1e-12, torch.sin(half)/torch.where(th > 1e-12, th, torch.ones_like(th)), 0.5-th*th/48.0)
    return torch.cat((v*k, torch.cos(half)), -1)


def qlog(q):
    """Rotation vector of the unit quaternion q (shortest arc)."""
    import torch
    q = torch.where(q[..., 3:] < 0, -q, q)
    s = torch.linalg.vector_norm(q[..., :3], dim=-1, keepdim=True)
    ang = 2.0*torch.atan2(s, q[..., 3:])
    k = torch.where(s > 1e-12, ang/torch.where(s > 1e-12, s, torch.ones_like(s)), 2.0/q[..., 3:].clamp(min=1e-300))
    return q[..., :3]*k


def qrot(q, v):
    """Rotate v [..., 3] by the unit quaternion q [..., 4] (broadcast)."""
    import torch
    u, w = q[..., :3], q[..., 3:]
    t = 2.0*torch.linalg.cross(u.expand_as(v), v, dim=-1)
    return v+w*t+torch.linalg.cross(u.expand_as(t), t, dim=-1)


def smootherstep(tau):
    return tau*tau*tau*(tau*(6.0*tau-15.0)+10.0)


# ----------------------------------------------------------------------------- single-env reference driver

def drive_single_reference(live, q, b, traction):
    """Advance a single ``LiveRubberMaterial`` reference (RL-02 a / c) with the grip END quaternion ``q`` [4]
    (float64, XYZW), translation ``b`` [3] and tip traction [3] (float32): the quaternion is staged with the batch's
    hemisphere rule (``q *= -1`` if dot(prev, q) < 0) right after ``begin_control_step`` (which makes every group
    hold), then ``LiveRubberMaterial.advance`` runs unchanged."""
    s = live.solver
    orig = s.begin_control_step
    q = np.array(q, np.float64)
    b = np.array(b, np.float64)

    def begin():
        orig()
        prev = s.group_q1.numpy()[0]
        qq = q.copy()
        if np.dot(prev, qq) < 0:
            qq *= -1
        s._set_row(s.group_q1, 0, qq)
        s._set_row(s.group_b1, 0, b)
    s.begin_control_step = begin
    try:
        return live.advance(traction={0: np.asarray(traction, np.float32)})
    finally:
        del s.begin_control_step


def _box_space(low, high, shape):
    try:
        from gymnasium import spaces
        return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    except ImportError:
        class _Box:
            def __init__(self, shape):
                self.shape, self.dtype = shape, np.float32
        return _Box(shape)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class RubberVectorEnv:
    """See the module docstring. Use ``open_vector_env(..., backend="kinematic_grip_batched")``."""

    metadata = {"render_modes": [], "runtime": RUNTIME, "backend": BACKEND, "version": ENV_VERSION,
                "privileged_info_keys": PRIVILEGED_INFO_KEYS}

    def __init__(self, num_envs, *, task, runtime=RUNTIME, observation_profile="state", device="cuda",
                 backend=BACKEND, action_spec=ACTION_SPEC, reward=REWARD, material=MATERIAL, autoreset=False,
                 strict=True, output_dir=None, parity_mode=False, cameras=None, sensors=None, recorders=None,
                 debug_sync=False, graph_chunk=None):
        t0 = time.perf_counter()
        import torch
        import warp as wp
        self.torch, self.wp = torch, wp
        if backend != BACKEND:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_BACKEND", f"{backend!r}; RubberVectorEnv is {BACKEND!r}")
        if runtime != RUNTIME:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_RUNTIME", f"{runtime!r}; only {RUNTIME!r}")
        if not isinstance(task, RubberStripTaskFamily):
            raise UnsupportedRubberConfiguration("UNSUPPORTED_TASK", "task must be a RubberStripTaskFamily")
        if action_spec != ACTION_SPEC or reward != REWARD or material != MATERIAL:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_OPTION", f"action_spec={action_spec!r} reward={reward!r} "
                                                 f"material={material!r}; rubber: {ACTION_SPEC}, {REWARD}, {MATERIAL}")
        if cameras or sensors or (observation_profile not in OBSERVATION_PROFILES and (
                "vision" in str(observation_profile) or "camera" in str(observation_profile))):
            raise UnsupportedRubberConfiguration("UNSUPPORTED_SENSOR", f"cameras={cameras!r} sensors={sensors!r} "
                                                 f"observation_profile={observation_profile!r} (state profiles only)")
        if observation_profile not in OBSERVATION_PROFILES:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_OPTION", f"observation_profile={observation_profile!r}")
        if recorders:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_RECORDER", f"recorders={recorders!r}")
        if autoreset is not False:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_OPTION", f"autoreset={autoreset!r}; v1 supports False")
        if not isinstance(num_envs, int) or num_envs < 1 or num_envs != task.num_envs:
            raise ValueError("num_envs must be a positive int equal to the task family's env count")
        self.torch_device = torch.device(device)
        if self.torch_device.type == "cuda" and self.torch_device.index is None:
            self.torch_device = torch.device("cuda", torch.cuda.current_device())
        if self.torch_device.type not in ("cpu", "cuda"):
            raise UnsupportedRubberConfiguration("UNSUPPORTED_DEVICE", str(device))
        if self.torch_device.type == "cpu" and not parity_mode:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_CPU_WITHOUT_PARITY_MODE",
                                                 "device='cpu' is for tests (parity_mode=True)")
        self.wp_device = "cpu" if self.torch_device.type == "cpu" else f"cuda:{self.torch_device.index}"
        self._cuda = self.torch_device.type == "cuda"
        self.num_envs, self.family = num_envs, task
        self.observation_profile, self.strict, self.parity_mode = observation_profile, bool(strict), bool(parity_mode)
        self.autoreset, self.debug_sync = False, bool(debug_sync)
        self.max_episode_steps = task.max_episode_steps
        self.metadata = dict(type(self).metadata, autoreset_mode=False, observation_profile=observation_profile,
                             max_episode_steps=self.max_episode_steps)
        self.last_warp_stream_handle = None
        self._streams = {}
        self._closed = False
        self.infrastructure_error = None
        self._has_reset = False
        self.reset_walls_s = []
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None:
            if self.output_dir.exists() and any(self.output_dir.iterdir()):
                raise FileExistsError(f"output_dir {self.output_dir} exists and is not empty (never overwritten)")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._build(graph_chunk)
        self.build_wall_s = time.perf_counter()-t0
        if self.output_dir is not None:
            (self.output_dir/"rubber_vector_env_receipt_open.json").write_text(
                json.dumps(self.receipt(), indent=1, sort_keys=True, default=_json_default)+"\n")

    # ------------------------------------------------------------------ construction

    def _build(self, graph_chunk):
        torch, wp, N, fam = self.torch, self.wp, self.num_envs, self.family
        from ..batched.rubber_material_batch import BatchedRubberMaterial
        t0 = time.perf_counter()
        lives = [fam.build_life(slot) for slot in fam.env_slots]
        self.lives_build_wall_s = time.perf_counter()-t0
        kw = {} if graph_chunk is None else {"graph_chunk": int(graph_chunk)}
        self.material = BatchedRubberMaterial(lives, device=self.wp_device, parity_mode=self.parity_mode,
                                              strict=self.strict, ranges=fam.batch_ranges(), **kw)
        del lives
        m = self.material
        if m.G != 1 or m.Gt != 1 or m.K != 9 or m.contact is None or m.self_penetration is None:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_TASK", "rubber_strip_drag_v1 needs one grip group, one "
                                                 "traction group, 9 markers, the table contact and the S1 detector")
        dev, f64 = self.torch_device, torch.float64
        geo = [fam.geometry(s) for s in fam.env_slots]
        self.origin = torch.tensor(np.stack([g["origin"] for g in geo]), dtype=f64, device=dev)
        self.jaw_corners = torch.tensor(np.stack([g["jaw_corners"] for g in geo]), dtype=f64, device=dev)
        self.length = torch.tensor([g["length"] for g in geo], dtype=f64, device=dev)
        self.n_t = [g["n_t"] for g in geo]
        self.box_lo = torch.tensor(fam.workspace_box[0], dtype=f64, device=dev)
        self.box_hi = torch.tensor(fam.workspace_box[1], dtype=f64, device=dev)
        with wp.ScopedDevice(self.wp_device):
            self.q1_w = wp.zeros(N, dtype=wp.quatd)
            self.b1_w = wp.zeros(N, dtype=wp.vec3d)
            self.traction_w = wp.zeros(N, dtype=wp.vec3)
        self.q1_t, self.b1_t = wp.to_torch(self.q1_w), wp.to_torch(self.b1_w)
        self.traction_t = wp.to_torch(self.traction_w)
        self.marker_x_t = wp.to_torch(m.marker_x).view(N, m.K, 3)
        self.marker_v_t = wp.to_torch(m.marker_v).view(N, m.K, 3)
        self.invalid_t = wp.to_torch(m.invalid_reason)
        self.wrench_t = wp.to_torch(m.wrench)
        self.summary_t = wp.to_torch(m.summary)
        self.visco_t = wp.to_torch(m.visco_summary)
        self.plane_wrench_t = wp.to_torch(m.contact.plane_wrench)
        self.ct_counts_t = wp.to_torch(m.solver.ct_counts)
        self.ct_pen_t = wp.to_torch(m.solver.ct_pen_max)
        z = lambda *s, dt=f64: torch.zeros(*s, dtype=dt, device=dev)   # noqa: E731
        self.grip_p, self.grip_q, self.twist = z(N, 3), z(N, 4), z(N, 6)
        self.grip_q[:, 3] = 1.0
        self.previous_action = z(N, 6, dt=torch.float32)
        self.goal, self.preload, self.params5 = z(N, 3), z(N, 2), z(N, 5)
        self.traction64 = z(N, 3)
        self.step_count = z(N, dt=torch.int64)
        self.seed = torch.arange(N, dtype=torch.int64, device=dev)
        self.episode = z(N, dt=torch.int64)
        self.drive = z(N, dt=torch.int8)
        self.termination_reason = z(N, dt=torch.int32)
        self.within_streak = z(N, dt=torch.int64)
        self.tip_distance = z(N)
        self.samples = [None]*N

    # ------------------------------------------------------------------ spaces

    @property
    def single_action_space(self):
        return _box_space(-1.0, 1.0, (6,))

    @property
    def action_space(self):
        return _box_space(-1.0, 1.0, (self.num_envs, 6))

    def _obs_space(self, n):
        shape = (lambda d: (d,)) if n is None else (lambda d: (n, d))
        out = {"policy": _box_space(-np.inf, np.inf, shape(POLICY_DIM))}
        if self.observation_profile == "privileged_state":
            out["critic"] = _box_space(-np.inf, np.inf, shape(CRITIC_DIM))
        try:
            from gymnasium import spaces
            return spaces.Dict(out)
        except ImportError:
            return out

    @property
    def single_observation_space(self):
        return self._obs_space(None)

    @property
    def observation_space(self):
        return self._obs_space(self.num_envs)

    @property
    def env_metadata(self):
        return {"autoreset_mode": self.autoreset, "observation_profile": self.observation_profile,
                "max_episode_steps": self.max_episode_steps, "privileged_info_keys": PRIVILEGED_INFO_KEYS}

    def observation_metadata(self):
        out = {"policy": {"dim": POLICY_DIM, "dtype": "float32", "deployable": True,
                          "terms": [{"name": n, "dim": d} for n, d in POLICY_TERMS],
                          "perception": "ideal_marker_tracking"}}
        if self.observation_profile == "privileged_state":
            out["critic"] = {"dim": CRITIC_DIM, "dtype": "float32", "deployable": False,
                             "terms": [{"name": "policy", "dim": POLICY_DIM}]+[{"name": n, "dim": d}
                                                                             for n, d in CRITIC_TERMS]}
        out["info_privileged"] = ["invalid_reason"]
        return out

    # ------------------------------------------------------------------ stream / guards

    @contextmanager
    def _stream(self):
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

    def _check_open(self):
        if self._closed:
            raise RuntimeError("vector env is closed")
        if self.infrastructure_error is not None:
            raise RuntimeError(f"vector env had an infrastructure error ({self.infrastructure_error['kind']}); "
                               "close it and open a new env (or process)")

    def _guarded(self, fn, *args, **kwargs):
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

    # ------------------------------------------------------------------ observations

    def _observe(self):
        torch = self.torch
        N = self.num_envs
        remaining = (self.max_episode_steps-self.step_count).to(torch.float64)/self.max_episode_steps
        preload = (self.step_count < self.family.preload_steps).to(torch.float64)
        policy = torch.cat((self.grip_p, self.grip_q, self.twist, remaining[:, None],
                            self.previous_action.to(torch.float64), self.marker_x_t.reshape(N, -1),
                            self.marker_v_t.reshape(N, -1), self.goal, preload[:, None]), 1).to(torch.float32)
        policy = torch.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)     # only after NONFINITE_STATE
        obs = {"policy": policy}
        if self.observation_profile == "privileged_state":
            s = self.summary_t
            from ..tlmpm_rubber_v1 import DIAG_COLUMNS as C
            col = C.index
            crit = torch.cat((self.params5, self.traction64, self.wrench_t[:, 0, :], self.plane_wrench_t[:, 0, :],
                              self.visco_t[:, 0:1], s[:, col("kinetic_particle"):col("kinetic_particle")+1],
                              s[:, col("gravity"):col("gravity")+1], s[:, col("Jbar_min"):col("Jbar_min")+1],
                              s[:, col("Jbar_max"):col("Jbar_max")+1], s[:, col("cfl_eff_max"):col("cfl_eff_max")+1],
                              self.ct_counts_t[:, 0:1].to(torch.float64), self.ct_pen_t[:, None]), 1)
            obs["critic"] = torch.cat((policy, torch.nan_to_num(crit.to(torch.float32), nan=0.0, posinf=0.0,
                                                                neginf=0.0)), 1)
        return obs

    def _reset_info(self):
        return {"step_count": self.step_count.clone(), "seed": self.seed.clone(), "episode": self.episode.clone(),
                "drive": self.drive.clone()}

    # ------------------------------------------------------------------ reset

    def reset(self, *, seeds=None, env_ids=None, options=None, seed=None):
        """Full (``env_ids=None``) or masked partial reset with per-env resampling (``seeds``: one int64 per reset
        env; None keeps each env's seed; ``seed``: Gymnasium full-reset form, seeds = seed + arange(N)).
        ``options={"slots": [...]}`` must name the envs' current slots (a change is refused: ``rebuild``)."""
        self._check_open()
        return self._guarded(self._reset_impl, seeds=seeds, env_ids=env_ids, options=options, seed=seed)

    def _reset_impl(self, *, seeds=None, env_ids=None, options=None, seed=None):
        torch, N = self.torch, self.num_envs
        options = dict(options or {})
        slots = options.pop("slots", None)
        if options:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_OPTION", f"reset options {sorted(options)}")
        if seed is not None:
            if env_ids is not None or seeds is not None:
                raise ValueError("seed= is the full-reset Gymnasium form; use seeds= with env_ids")
            seeds = int(seed)+np.arange(N, dtype=np.int64)
        ids = np.arange(N) if env_ids is None else np.asarray(
            env_ids.detach().cpu() if hasattr(env_ids, "detach") else env_ids, np.int64).reshape(-1)
        if len(np.unique(ids)) != len(ids) or (ids.size and (ids.min() < 0 or ids.max() >= N)):
            raise ValueError(f"env_ids must be unique and in [0, {N})")
        if seeds is not None:
            seeds = np.asarray(seeds.detach().cpu() if hasattr(seeds, "detach") else seeds, np.int64).reshape(-1)
            if len(seeds) != len(ids):
                raise ValueError("one seed per reset env")
        if slots is not None:
            slots = list(slots)
            if len(slots) != len(ids):
                raise ValueError("one slot per reset env")
            for e, s in zip(ids, slots):
                if s != self.family.env_slots[e]:
                    raise UnsupportedRubberConfiguration(
                        "UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET",
                        f"env {int(e)} holds slot {self.family.env_slots[e]}, requested {s}; use rebuild()")
        if ids.size == 0:
            with self._stream():
                return self._observe(), self._reset_info()
        t0 = time.perf_counter()
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        current = self.seed.cpu().numpy()
        new_seeds = current[ids] if seeds is None else seeds[order]
        samples = [self.family.sample(int(s), self.family.env_slots[e]) for e, s in zip(ids, new_seeds)]
        for e, smp in zip(ids, samples):
            self.family.check_material(smp["material"], self.n_t[e])
        with self._stream():
            self.material.reset(env_ids=ids, seeds=new_seeds, materials=[s["material"] for s in samples])
            self.material.launch_diagnostics()   # markers / summary of the reset state (others recomputed unchanged)
            dev, f64 = self.torch_device, torch.float64
            idx = torch.as_tensor(ids, dtype=torch.int64, device=dev)
            host = lambda key: torch.as_tensor(np.stack([s[key] for s in samples]), dtype=f64, device=dev)  # noqa
            self.goal[idx] = host("goal_m")
            self.preload[idx] = host("preload")
            self.traction64[idx] = host("traction_n")
            self.traction_t[idx] = self.traction64[idx].to(torch.float32)
            self.params5[idx] = torch.as_tensor(np.array([[s["draws"]["mu0_pa"]*1e-6, s["draws"]["density_kg_m3"]*1e-3,
                                                           s["draws"]["friction_table"], s["draws"]["prony_g_scale"],
                                                           s["draws"]["prony_tau_scale"]] for s in samples]),
                                                dtype=f64, device=dev)
            self.seed[idx] = torch.as_tensor(new_seeds, dtype=torch.int64, device=dev)
            self.grip_p[idx] = self.origin[idx]
            self.grip_q[idx] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=f64, device=dev)
            self.twist[idx] = 0.0
            self.previous_action[idx] = 0.0
            self.step_count[idx] = 0
            self.episode[idx] += 1
            self.drive[idx] = DRIVE_PRELOAD if self.family.preload_steps > 0 else DRIVE_ACTION
            self.termination_reason[idx] = 0
            self.within_streak[idx] = 0
            tip = self.marker_x_t[idx, TIP_MARKER]
            self.tip_distance[idx] = torch.linalg.vector_norm(tip-self.goal[idx], dim=-1)
            for e, s in zip(ids, samples):
                self.samples[int(e)] = s
            out = self._observe(), self._reset_info()
        if self._cuda:
            self.torch.cuda.synchronize(self.torch_device)
        self.reset_walls_s.append({"envs": len(ids), "wall_s": time.perf_counter()-t0})
        self._has_reset = True
        return out

    def rebuild(self, task):
        """Rebuild with a new task family (new slots; full rebuild, new graphs); call ``reset`` afterwards."""
        self._check_open()
        if not isinstance(task, RubberStripTaskFamily) or task.num_envs != self.num_envs:
            raise UnsupportedRubberConfiguration("UNSUPPORTED_TASK", "rebuild needs a RubberStripTaskFamily with the "
                                                 "same env count")
        self.family = task
        self.max_episode_steps = task.max_episode_steps
        self.metadata["max_episode_steps"] = self.max_episode_steps
        self._has_reset = False
        self._build(None)

    # ------------------------------------------------------------------ step

    def _workspace_ok(self, p, q):
        torch, fam = self.torch, self.family
        off = p-self.origin
        inside = ((off >= self.box_lo) & (off <= self.box_hi)).all(1)
        corners = qrot(q[:, None, :], self.jaw_corners-self.origin[:, None, :])+p[:, None, :]
        clear = corners[..., 2].amin(1) >= fam.jaw_min_clearance
        ez = qrot(q, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=self.torch_device).expand_as(p))
        tilt = ez[:, 2] >= math.cos(fam.tilt_max)
        return inside & clear & tilt

    def step(self, actions):
        """Apply ``grip_delta_v1`` actions ``[N, 6]`` (float32, on the env's device) to all envs."""
        self._check_open()
        if not self._has_reset:
            raise RuntimeError("call reset() before step()")
        return self._guarded(self._step_impl, actions)

    def _step_impl(self, actions):
        torch, N, fam = self.torch, self.num_envs, self.family
        if (not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (N, 6)
                or actions.device != self.torch_device or actions.dtype != torch.float32):
            raise ValueError(f"actions must be a float32 torch tensor ({N}, 6) on {self.torch_device}")
        from .termination import INVALID_STATE_BITS, TERM_MATERIAL_INVALID, TERM_TRUNCATED, TERMINAL_BITS
        f64 = torch.float64
        with self._stream():
            a = actions.to(f64)
            out_of_box = (a.abs() > 1.0).any(1)
            ac = a.clamp(-1.0, 1.0)
            preload = self.step_count < fam.preload_steps
            dth = ac[:, 3:]*fam.max_rotation
            q_c = qmul(qexp(dth), self.grip_q)
            q_c = q_c/torch.linalg.vector_norm(q_c, dim=1, keepdim=True)
            p_c = self.grip_p+ac[:, :3]*fam.max_translation
            ok = self._workspace_ok(p_c, q_c)
            p_pol = torch.where(ok[:, None], p_c, self.grip_p)
            q_pol = torch.where(ok[:, None], q_c, self.grip_q)
            tau = ((self.step_count+1).to(f64)/max(fam.preload_steps, 1)).clamp(max=1.0)
            s = smootherstep(tau)
            zero = torch.zeros_like(s)
            p_pre = self.origin+torch.stack((zero, self.preload[:, 0]*s, fam.preload_lift*s), 1)
            half = 0.5*self.preload[:, 1]*s
            q_pre = torch.stack((zero, zero, torch.sin(half), torch.cos(half)), 1)
            p_new = torch.where(preload[:, None], p_pre, p_pol)
            q_new = torch.where(preload[:, None], q_pre, q_pol)
            dq = qmul(q_new, qconj(self.grip_q))
            self.twist.copy_(torch.cat(((p_new-self.grip_p)/fam.control_dt, qlog(dq)/fam.control_dt), 1))
            status = torch.where(preload, STATUS_PRELOAD, torch.where(~ok, STATUS_HELD, torch.where(
                out_of_box, STATUS_CLIPPED, STATUS_APPLIED))).to(torch.int32)
            clamped = ~preload & ~ok
            self.grip_p.copy_(p_new)
            self.grip_q.copy_(q_new)
            self.q1_t.copy_(q_new)
            self.b1_t.copy_(p_new-qrot(q_new, self.origin))
            self.material.advance_device(self.q1_w, self.b1_w, self.traction_w)
            self.step_count += 1
            self.drive.copy_(torch.where(self.step_count < fam.preload_steps, DRIVE_PRELOAD, DRIVE_ACTION).to(
                torch.int8))
            inv = self.invalid_t.to(torch.int32)
            reason = self.termination_reason | torch.where(inv != 0, TERM_MATERIAL_INVALID, 0).to(torch.int32)
            reason = (reason & ~TERM_TRUNCATED) | torch.where(self.step_count >= self.max_episode_steps,
                                                              TERM_TRUNCATED, 0).to(torch.int32)
            self.termination_reason.copy_(reason)
            tip = self.marker_x_t[:, TIP_MARKER]
            dist = torch.linalg.vector_norm(tip-self.goal, dim=1)
            self.tip_distance.copy_(dist)
            invalid_state = ((reason & INVALID_STATE_BITS) != 0).to(f64)
            d_term = torch.where(torch.isfinite(dist), dist/self.length, torch.ones_like(dist))
            r64 = -d_term-0.01*(ac*ac).sum(1)-1.0*invalid_state
            r64 = torch.where(preload, torch.zeros_like(r64), r64)
            within = (~preload) & (dist <= fam.eval_tolerance)
            self.within_streak.copy_(torch.where(within, self.within_streak+1, torch.zeros_like(self.within_streak)))
            self.previous_action.copy_(torch.where(preload[:, None], torch.zeros_like(ac), ac).to(torch.float32))
            self.last_reward64 = r64
            terminated = (reason & TERMINAL_BITS) != 0
            truncated = (reason & TERM_TRUNCATED) != 0
            info = {"step_count": self.step_count.clone(), "action_status": status, "invalid_reason": inv.clone(),
                    "action_clamped": clamped, "action_out_of_box": out_of_box & ~preload,
                    "drive": torch.where(preload, DRIVE_PRELOAD, DRIVE_ACTION).to(torch.int8),
                    "seed": self.seed.clone(), "episode": self.episode.clone(), "termination_reason": reason.clone(),
                    "tip_distance_m": dist.clone(), "within_streak": self.within_streak.clone(),
                    "invalid_penalty_applied": (invalid_state > 0) & ~preload}
            obs = self._observe()
            return obs, r64.to(torch.float32), terminated, truncated, info

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, env_ids=None):
        """``rubber_tracking_eval_v1`` on host copies of the device counters (success never from the reward)."""
        self._check_open()
        from .termination import TERMINAL_BITS, decode_termination
        from ..batched.rubber_material_batch import decode_reason
        ids = list(range(self.num_envs)) if env_ids is None else [int(e) for e in env_ids]
        with self._stream():
            host = {k: getattr(self, k).cpu().numpy() for k in ("step_count", "termination_reason", "within_streak",
                                                                "tip_distance", "seed", "episode", "goal")}
            inv = self.invalid_t.cpu().numpy()
        fam = self.family
        out = []
        for e in ids:
            steps, reason = int(host["step_count"][e]), int(host["termination_reason"][e])
            window_ok = steps >= fam.preload_steps+fam.eval_window and int(host["within_streak"][e]) >= fam.eval_window
            success = bool(window_ok and not (reason & TERMINAL_BITS))
            out.append({"env": e, "slot": fam.env_slots[e], "seed": int(host["seed"][e]),
                        "episode": int(host["episode"][e]), "step_count": steps, "evaluator": EVALUATOR,
                        "success": success, "verdict": "SUCCESS" if success else "NOT_SUCCESS",
                        "tip_distance_m": float(host["tip_distance"][e]), "goal_m": host["goal"][e].tolist(),
                        "within_streak": int(host["within_streak"][e]), "window_steps": fam.eval_window,
                        "tolerance_m": fam.eval_tolerance, "termination_reason": decode_termination(reason),
                        "invalid_reason": decode_reason(int(inv[e])),
                        "labels": {"calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
                                   "coupling": "kinematic_grip_ideal_tracking",
                                   "perception": "ideal_marker_tracking",
                                   "test_configuration": fam.variant}})
        return out

    # ------------------------------------------------------------------ receipt / close

    def receipt(self):
        from .termination import termination_table
        import torch
        import warp as wp
        fam = self.family
        return {"version": ENV_VERSION, "api_revision": API_REVISION, "runtime": RUNTIME, "backend": BACKEND,
                "num_envs": self.num_envs, "device": str(self.torch_device), "parity_mode": self.parity_mode,
                "strict": self.strict, "termination": termination_table(),
                "termination_mapping": "rubber invalid bits 1 NONFINITE / 2 J_DOMAIN / 4 CFL / 8 SELF_CONTACT_UNRESOLVED "
                                       "-> TERM_MATERIAL_INVALID (detail: info['invalid_reason'], privileged)",
                "evaluation_mode": EVALUATOR, "evaluator": {"id": EVALUATOR, "window_steps": fam.eval_window,
                                                            "tolerance_m": fam.eval_tolerance},
                "success_source": f"evaluate() ({EVALUATOR}); never the reward",
                "fidelity": {"reduced_fidelity": (fam.variant is not None or fam.resolution is not None),
                             "test_configuration": fam.variant, "resolution_variant": fam.resolution,
                             "reduced_fidelity_label": ("thickness_resolution_n_t3" if fam.resolution == "rubber_rl_v1"
                                                        else None),
                             "coupling": "kinematic_grip_ideal_tracking (no wrench feedback)",
                             "robot_feedback_coupling": "NOT_IMPLEMENTED_PHASE2",
                             "perception": "ideal_marker_tracking",
                             "self_contact": "NOT_IMPLEMENTED_V1 (detector surface_grid_v1)",
                             "resolution": ("rubber_rl_v1 (n_t 3, dx 4/3 mm)" if fam.resolution == "rubber_rl_v1"
                                            else "rubber_ref_v1 (n_t 4-6 at dx 1 mm)")},
                "observation": self.observation_metadata(), "observation_profile": self.observation_profile,
                "action_spec": {"id": ACTION_SPEC, "dim": 6, "max_translation_m": fam.max_translation,
                                "max_rotation_rad": fam.max_rotation,
                                "workspace_box_m": fam.workspace_box.tolist(),
                                "jaw_min_clearance_m": fam.jaw_min_clearance, "tilt_max_rad": fam.tilt_max},
                "reward": {"id": REWARD, "status": "PROVISIONAL_SHAPING_NOT_SUCCESS"},
                "tasks": fam.receipt(), "episode": {"max_episode_steps": self.max_episode_steps,
                                                    "preload_steps": fam.preload_steps, "autoreset": False},
                "calibration": "SIMULATOR_ONLY_UNCALIBRATED", "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
                "profile": fam.profile, "contact_law": fam.law,
                "envelope": {str(n): fam.envelope.receipt_row(fam.law, n) for n in fam.n_ts},
                "labels": dict(fam.config["labels"]), "resolution": fam.resolution_label(),
                "material": {k: v for k, v in self.material.receipt().items() if k != "envs"},
                "step_staging": "device (advance_device); resets host-staged",
                "stream_policy": "Warp launches on torch.cuda.current_stream via wp.stream_from_torch + "
                                 "wp.ScopedStream",
                "versions": {"env": ENV_VERSION, "python": platform.python_version(), "warp": wp.config.version,
                             "torch": torch.__version__, "sources_sha256": source_sha256()},
                "build_wall_s": getattr(self, "build_wall_s", None),
                "lives_build_wall_s": getattr(self, "lives_build_wall_s", None),
                "reset_walls_s": list(self.reset_walls_s)}

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.output_dir is not None:
            try:
                (self.output_dir/"rubber_vector_env_receipt_close.json").write_text(
                    json.dumps(self.receipt(), indent=1, sort_keys=True, default=_json_default)+"\n")
            except Exception:                                    # the close receipt never masks the caller's error
                pass
        self.material = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_rubber_vector_env(num_envs, *, task, backend=BACKEND, action_spec=None, reward=None, material=None,
                           tasks=None, **kwargs):
    """Dispatch target of ``rl.vector_env.open_vector_env`` for the rubber backend: maps the cutting API's default
    values (``edge_delta_v1``, ``trim_bond_progress_packed_v1``, ``scene_declared``) to the rubber defaults and refuses
    every other non-rubber value; ``tasks=`` (cutting per-env selection) is refused."""
    if backend != BACKEND:
        raise UnsupportedRubberConfiguration("UNSUPPORTED_BACKEND", f"a RubberStripTaskFamily runs on {BACKEND!r}, "
                                             f"got backend={backend!r}")
    if not isinstance(task, RubberStripTaskFamily):
        raise UnsupportedRubberConfiguration("UNSUPPORTED_TASK", f"backend {BACKEND!r} needs a RubberStripTaskFamily "
                                             f"(got {type(task).__name__})")
    if tasks is not None:
        raise UnsupportedRubberConfiguration("UNSUPPORTED_OPTION", "tasks= selects cutting scenes; pass the rubber "
                                             "slots to RubberStripTaskFamily")
    unknown = sorted(set(kwargs) - _ENV_KWARGS)
    if unknown:   # F4 (P6.7 review): Phase 1 options (evaluation_stream, terminal_capture, input_guard, tool_mount, ...)
        raise UnsupportedRubberConfiguration("UNSUPPORTED_OPTION", f"options {unknown} are not supported by the rubber "
                                             f"backend {BACKEND!r}")
    defaults = {"action_spec": ("edge_delta_v1", ACTION_SPEC), "reward": ("trim_bond_progress_packed_v1", REWARD),
                "material": ("scene_declared", MATERIAL)}
    given = {"action_spec": action_spec, "reward": reward, "material": material}
    mapped = {}
    for key, (cutting_default, rubber) in defaults.items():
        value = given[key]
        mapped[key] = rubber if value in (None, cutting_default, rubber) else value
    return RubberVectorEnv(num_envs, task=task, backend=backend, **mapped, **kwargs)


_ENV_KWARGS = frozenset(n for n, p in __import__("inspect").signature(RubberVectorEnv.__init__).parameters.items()
                        if p.kind == p.KEYWORD_ONLY)


__all__ = ["RubberVectorEnv", "open_rubber_vector_env", "drive_single_reference", "ENV_VERSION", "BACKEND",
           "POLICY_DIM", "CRITIC_DIM", "PRIVILEGED_INFO_KEYS", "POLICY_TERMS", "CRITIC_TERMS", "TIP_MARKER", "source_sha256",
           "qmul", "qexp", "qlog", "qrot", "qconj", "smootherstep"]
