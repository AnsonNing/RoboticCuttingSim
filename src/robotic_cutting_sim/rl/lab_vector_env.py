"""``open_vector_env(backend="isaac_lab_newton")``: workload B (robot-batched) behind the vector-env contract.

``LabVectorCuttingEnv`` wraps ``rl.lab_batched_env.IsaacLabBatchedCuttingEnv`` (N Franka worlds on Isaac Lab Newton /
MuJoCo-Warp in one process, ``BatchedCuttingMaterial(backend="isaac_lab_newton")`` coupled with the delayed
reaction wrench; ADR 0022) and exposes the public contract of workload A's ``VectorCuttingEnv`` (P1-2b,
checked by ``rl.vector_contract`` for both backends): ``reset(seeds=, env_ids=)`` -> ``(obs, info)``,
``step(actions [N,8])`` -> ``(obs, reward, terminated, truncated, info)``, the deployable ``"policy"`` state
observation (``rl_state_obs_v1``) and, only with ``observation_profile="privileged_state"``, the ``"critic"``
block; ``info`` tensors with the same keys, shapes and dtypes; ``termination_reason`` (``rl.termination``);
``evaluate(env_ids)`` with the frozen evaluator (two-frame default or the opt-in evaluation stream);
``receipt()``.

Mapping (P1-1b report, "Dispatch ... needs"): ``launch_simulation(cfg, {"headless": True, "visualizer": []})`` is
entered at construction and exited in ``close()``; ``reset()`` -> ``env.reset()`` (B1 HOME / B2 settled
snapshot, material attached on the first call); ``reset(env_ids)`` -> ``env.reset_envs(device bool mask)``;
``step`` -> ``env.step``; the reward is the env's ``trim_bond_progress_packed_v1`` over ``env.coupling`` (the same
class as workload A). Refused with explicit codes: cameras/sensors (``UNSUPPORTED_SENSOR``), recorders,
``autoreset="same_step"`` (``UNSUPPORTED_AUTORESET_FINAL_OBS``: the Lab env's masked reset needs the next
``forward()`` and it returns no final observation), ``device="cpu"`` without ``cpu_test_mode=True``
(``UNSUPPORTED_RIGID_CPU``), robot-only stepping (``material`` must be the scene-declared material), per-env
initial edge poses and pad boxes (the robot's HOME and measured finger pads define them), unknown options.

Workload B with ``tool_mount="rigid_mount_v1"`` (B1) is ``reduced_fidelity`` (rigid knife mount, no grasp);
B2 (``physical_grasp_v1``) is not reduced. With the default point grip profile it lost its grasp in every S3 env
(P1-1b report); ``grip_contact_profile="mjwarp_grip_v1"`` (opt-in, ``batched.lab_scene.CONTACT_SOLVER_PROFILES``:
MuJoCo ``box_box`` collider + ``impratio`` 10, B2 only) held it and passed S3 (P1-1b "B2 grasp follow-up").

``grip_contact_profile`` (P1 consolidation) is validated here before Isaac Lab is imported: a profile outside
``lab_scene.GRIP_PROFILES`` raises ``UNSUPPORTED_GRIP_CONTACT_PROFILE``, a non-default profile without
``tool_mount="physical_grasp_v1"`` raises ``UNSUPPORTED_TOOL_MOUNT``; the profile and its MuJoCo-Warp solver options
are recorded in ``receipt()`` and ``fidelity()`` (a listed difference, never a silent change).
"""
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import time

import numpy as np

from .vector_env import (ACTION_SPEC, AUTORESET_MODES, LAB_BACKEND, MATERIAL, OBSERVATION_PROFILES, REWARD, RUNTIME,
                         VECTOR_API_REVISION, VECTOR_ENV_VERSION, TaskFamily, UnsupportedVectorConfiguration,
                         _box_space, _guard_config, _json_default, _sha256_file, compile_family)

LAB_MATERIAL = "tlmpm_v7_2"
LAB_OPTIONS = ("tool_mount", "grip_contact_profile", "deterministic_mode", "use_cuda_graph",
               "max_consecutive_ik_failures", "feedback", "ik_iterations", "velocity_feedforward", "cpu_test_mode",
               "evaluation_stream", "input_guard")


def check_grip_contact_profile(profile, tool_mount="rigid_mount_v1"):
    """``(profile, tool_mount)`` of an ``isaac_lab_newton`` request, refused before Isaac Lab is imported.

    ``profile`` None is the default point profile. Unknown profiles raise ``UNSUPPORTED_GRIP_CONTACT_PROFILE``; a
    non-default profile needs the B2 physical grasp (``UNSUPPORTED_TOOL_MOUNT``, as ``BatchedLabOptions.validate``)."""
    from ..batched import lab_scene
    from ..pad_contact import POINT_PROFILE
    profile = POINT_PROFILE if profile is None else profile
    if not isinstance(profile, str) or profile not in lab_scene.GRIP_PROFILES:
        raise UnsupportedVectorConfiguration("UNSUPPORTED_GRIP_CONTACT_PROFILE",
                                             f"grip_contact_profile={profile!r}; supported {lab_scene.GRIP_PROFILES}")
    if profile != POINT_PROFILE and tool_mount != "physical_grasp_v1":
        raise UnsupportedVectorConfiguration("UNSUPPORTED_TOOL_MOUNT", f"grip_contact_profile={profile!r} needs "
                                             f"tool_mount='physical_grasp_v1' (B2), got {tool_mount!r}")
    return profile, tool_mount


def grip_profile_record(profile, tool_mount):
    """Receipt/fidelity record of the grip contact profile (``lab_scene.CONTACT_SOLVER_PROFILES`` when it has one)."""
    from ..batched import lab_scene
    solver = lab_scene.contact_solver_profile(profile)
    return {"grip_contact_profile": profile, "tool_mount": tool_mount,
            "applies": tool_mount == "physical_grasp_v1",
            "contact_solver_profile": None if solver is None else dict(solver),
            "statement": "opt-in MuJoCo-Warp contact-solver options of the B2 model (listed difference, not a "
                         "calibrated grip)" if solver is not None else
                         "authored grip profile of the single Lab env (no solver options changed)"}


class _LabGuardHook:
    """``IsaacLabBatchedCuttingEnv.material_input_guard``: called in ``_get_dones`` after the edge and pad-box kernels
    and before ``coupling.set_measured_poses`` (device only)."""

    def __init__(self, owner):
        import torch
        import warp as wp
        self.owner, self.torch, self.wp = owner, torch, wp
        env = owner.env
        N = owner.num_envs
        self.pose = torch.zeros((N, 7), dtype=torch.float64, device=owner.torch_device)
        self.cut = torch.zeros(N, dtype=torch.int32, device=owner.torch_device)
        self.pose_wp = wp.from_torch(self.pose, dtype=wp.float64)
        self.cut_wp = wp.from_torch(self.cut)
        c = env.coupling
        B = int(c.solver.boxes)
        self.centers = wp.to_torch(c.box_centers).view(N, B, 3)
        self.halves = wp.to_torch(c.box_halves).view(N, B, 3)
        self.last_pose = wp.to_torch(c.last_pose)
        owner.input_guard.init_boxes(self.centers, self.halves)

    def __call__(self, edge, cut):
        wp = self.wp
        pose, digit = self.owner.input_guard.apply_in_place_boxes(
            wp.to_torch(edge), wp.to_torch(cut), self.last_pose, self.centers, self.halves,
            step_count=self.owner.env.episode_length_buf)
        self.pose.copy_(pose)
        self.cut.copy_(digit.to(self.torch.int32))
        return self.pose_wp, self.cut_wp

    def termination_bits(self):
        return self.owner.input_guard.termination_bits()


class LabVectorCuttingEnv:
    """Workload B vector env (module docstring). Use ``open_vector_env(..., backend="isaac_lab_newton")``."""

    metadata = {"render_modes": [], "runtime": RUNTIME, "backend": LAB_BACKEND, "version": VECTOR_ENV_VERSION}

    def __init__(self, num_envs, *, runtime=RUNTIME, task=None, material=MATERIAL, observation_profile="state",
                 device="cuda", action_spec=ACTION_SPEC, reward=REWARD, autoreset=False, strict=True,
                 output_dir=None, cameras=None, sensors=None, recorders=None, **options):
        t0 = time.perf_counter()
        # ---- refusals that need no Isaac Lab import
        if runtime != RUNTIME:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_RUNTIME", f"{runtime!r}; only {RUNTIME!r}")
        if cameras or sensors or "vision" in str(observation_profile) or "camera" in str(observation_profile):
            raise UnsupportedVectorConfiguration("UNSUPPORTED_SENSOR", f"isaac_lab_newton batched path has no cameras "
                                                 f"or sensors (cameras={cameras!r}, sensors={sensors!r}, "
                                                 f"observation_profile={observation_profile!r})")
        if observation_profile not in OBSERVATION_PROFILES:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"observation_profile={observation_profile!r}")
        if recorders:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_RECORDER", f"recorders={recorders!r}")
        if autoreset == "same_step":
            raise UnsupportedVectorConfiguration(
                "UNSUPPORTED_AUTORESET_FINAL_OBS", "isaac_lab_newton: IsaacLabBatchedCuttingEnv has no same-step "
                "autoreset with final_obs (its masked reset needs the next forward(); ADR 0022 §6); call "
                "reset(env_ids=...) on done envs")
        if autoreset is not False or not any(autoreset is m or autoreset == m for m in AUTORESET_MODES):
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"autoreset={autoreset!r}; isaac_lab_newton "
                                                 "supports autoreset=False")
        if material != MATERIAL:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"material={material!r}; only {MATERIAL!r} "
                                                 f"(the scene-declared {LAB_MATERIAL}; robot-only stepping is not a "
                                                 "vector-env task)")
        if action_spec != ACTION_SPEC:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"action_spec={action_spec!r}")
        if reward != REWARD:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"reward={reward!r}")
        unknown = sorted(set(options) - set(LAB_OPTIONS))
        if unknown:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"isaac_lab_newton options {unknown}; "
                                                 f"supported {LAB_OPTIONS}")
        self.grip_contact_profile, self.tool_mount = check_grip_contact_profile(
            options.get("grip_contact_profile"), options.get("tool_mount", "rigid_mount_v1"))
        cpu_test_mode = bool(options.pop("cpu_test_mode", False))
        if str(device).startswith("cpu") and not cpu_test_mode:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_RIGID_CPU", "isaac_lab_newton on device='cpu' is the "
                                                 "Warp-CPU MJWarp test runtime (cpu_test_mode=True)")
        if not isinstance(task, TaskFamily):
            raise UnsupportedVectorConfiguration("UNSUPPORTED_TASK", "task must be a TaskFamily of seeded TRIM scenes "
                                                 "(or pass tasks=[...])")
        if not isinstance(num_envs, int) or num_envs < 1 or len(task.scene_seeds) != num_envs:
            raise ValueError("num_envs must be a positive int equal to len(task.scene_seeds)")
        if task.initial_edge_poses is not None or task.pad_boxes_edge is not None:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", "isaac_lab_newton: the robot HOME defines the "
                                                 "initial edge and the measured finger pads the pad boxes "
                                                 "(initial_edge_poses / pad_boxes_edge are workload A options)")
        from .evaluation_stream import stream_config
        from .material_guard import MaterialInputGuardConfig
        self.evaluation_stream_config = stream_config(options.pop("evaluation_stream", None))
        self.input_guard_config = _guard_config(options.pop("input_guard", None), MaterialInputGuardConfig)
        # ---- Isaac Lab runtime
        try:
            import torch
            import warp as wp
            from isaaclab.app.sim_launcher import launch_simulation
        except ImportError as error:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_BACKEND", "isaac_lab_newton needs the Isaac Lab runtime "
                                                 f"(.deps/isaaclab-venv): {error}") from error
        self.torch, self.wp = torch, wp
        self.torch_device = torch.device(device)
        if self.torch_device.type == "cuda" and self.torch_device.index is None:
            self.torch_device = torch.device("cuda", torch.cuda.current_device())
        lab_device = "cpu" if self.torch_device.type == "cpu" else f"cuda:{self.torch_device.index}"
        self.num_envs, self.task_family, self.observation_profile = num_envs, task, observation_profile
        self.autoreset, self.strict = False, bool(strict)
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None:
            if self.output_dir.exists() and any(self.output_dir.iterdir()):
                raise FileExistsError(f"output_dir {self.output_dir} exists and is not empty (never overwritten)")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self.infrastructure_error = None
        self.tasks = compile_family(task)
        self._stack = ExitStack()
        try:
            from .lab_batched_env import open_lab_batched_env
            lab_options = dict(options, grip_contact_profile=self.grip_contact_profile)   # None -> the default
            tool_mount = lab_options.pop("tool_mount", "rigid_mount_v1")
            self.cfg, factory = open_lab_batched_env(
                num_envs, task=self.tasks[0], tasks=self.tasks, tool_mount=tool_mount, device=lab_device,
                material=LAB_MATERIAL, strict=self.strict, seed=int(task.sim_seed),
                material_dt_s=float(task.material_dt_s), cpu_test_mode=cpu_test_mode,
                max_episode_steps=task.max_episode_steps, **lab_options)
            self._stack.enter_context(launch_simulation(self.cfg, {"headless": True, "visualizer": [],
                                                                   "device": lab_device}))
            self.env = factory()
            self.lab_options = self.env.options
            t_build = time.perf_counter()
            self.env.reset()       # HOME (B1) or settle + snapshot (B2); attaches the batched material
            self.initial_reset_wall_s = time.perf_counter() - t_build
            self._bind()
        except BaseException:
            self._stack.close()
            raise
        self.build_wall_s = time.perf_counter() - t0
        if self.output_dir is not None:
            (self.output_dir / "vector_env_receipt_open.json").write_text(
                json.dumps(self.receipt(), indent=1, sort_keys=True, default=_json_default) + "\n")

    # ------------------------------------------------------------------ construction

    def _bind(self):
        torch, wp, env, N = self.torch, self.wp, self.env, self.num_envs
        from .evaluation_stream import EvaluationStream
        from .material_guard import MaterialInputGuard
        from .observations import StateObservation
        if not env.coupling.material_coupled:
            raise RuntimeError("the batched material was not attached by the first reset")
        self.material = env.coupling
        dev = self.torch_device
        self.max_episode_steps = int(env.max_steps)
        self.declared_episode_steps = int(env.declared_steps)
        self.control_dt = float(env.task.control_dt_s)
        self.observation = StateObservation(N, max_steps=self.max_episode_steps, control_dt=self.control_dt,
                                            profile=self.observation_profile, device=dev)
        self.reward = env.reward
        self._seed = wp.to_torch(self.material.seed)
        self._invalid = wp.to_torch(self.material.invalid_reason)
        self._pending_wrench = wp.to_torch(self.material.pending_wrench)
        self._cut_digit = wp.to_torch(env.cut_digit)
        if self.task_family.env_seeds is not None:
            self._seed.copy_(torch.as_tensor(self.task_family.env_seeds, dtype=torch.int64, device=dev))
        self.episode = torch.zeros(N, dtype=torch.int64, device=dev)
        self.drive = torch.zeros(N, dtype=torch.int8, device=dev)      # always policy_action_edge_delta_v1
        self.input_guard = (None if self.input_guard_config is None else
                            MaterialInputGuard(self.material, N, dev, self.input_guard_config))
        if self.input_guard is not None:
            env.material_input_guard = _LabGuardHook(self)
        self.evaluation_stream = (None if self.evaluation_stream_config is None else
                                  EvaluationStream(self.material, self.max_episode_steps, dev,
                                                   self.evaluation_stream_config))
        if self.evaluation_stream is not None:
            self.evaluation_stream.capture(torch.ones(N, dtype=torch.bool, device=dev))
        self.single_action_space = _box_space(-1., 1., (8,))
        self.action_space = _box_space(-1., 1., (N, 8))
        self.single_observation_space = self._obs_space(None)
        self.observation_space = self._obs_space(N)
        self.metadata = {**type(self).metadata, "autoreset_mode": "disabled",
                         "observation_profile": self.observation_profile, "max_episode_steps": self.max_episode_steps,
                         "tool_mount": self.lab_options.tool_mount}

    def _obs_space(self, n):
        shape = (lambda d: (d,)) if n is None else (lambda d: (n, d))
        spaces = {"policy": _box_space(-np.inf, np.inf, shape(self.observation.policy_dim))}
        if self.observation_profile == "privileged_state":
            spaces["critic"] = _box_space(-np.inf, np.inf, shape(self.observation.critic_dim))
        try:
            from gymnasium import spaces as gym_spaces
            return gym_spaces.Dict(spaces)
        except ImportError:
            return spaces

    # ------------------------------------------------------------------ observation

    def _privileged(self):
        from .termination import INVALID_STATE_BITS
        fractions = self.reward.fractions(self.reward.counts)
        return {"target_bond_progress_fraction": fractions["target_bond_progress_fraction"],
                "off_target_bond_fraction": fractions["off_target_bond_fraction"],
                "material_wrench_world": self._pending_wrench,
                "invalid_flag": ((self.env.termination_reason & INVALID_STATE_BITS) != 0).to(self.torch.float32)}

    def _observe(self):
        env = self.env
        privileged = self._privileged() if self.observation_profile == "privileged_state" else None
        return self.observation.compute(commanded_pose=env.commanded, step_count=env.episode_length_buf,
                                        previous_action=env.previous_action, armed=env.armed, privileged=privileged)

    def _reset_info(self):
        return {"step_count": self.env.episode_length_buf.to(self.torch.int64).clone(), "seed": self._seed.clone(),
                "episode": self.episode.clone(), "drive": self.drive.clone()}

    # ------------------------------------------------------------------ guarded calls

    def _check_open(self):
        if self._closed:
            raise RuntimeError("vector env is closed")
        if self.infrastructure_error is not None:
            raise RuntimeError(f"vector env had an infrastructure error ({self.infrastructure_error['kind']}); "
                               "close it and open a new env (or process)")

    def _infrastructure_guarded(self, fn, *args, **kwargs):
        from .vector_env import VectorCuttingEnv
        return VectorCuttingEnv._infrastructure_guarded(self, fn, *args, **kwargs)

    # ------------------------------------------------------------------ reset

    def reset(self, *, seeds=None, env_ids=None, scene_seeds=None, options=None, seed=None):
        """Full (``env.reset()``) or masked partial (``env.reset_envs(mask)``) reset; ``(obs, info)`` of all envs.

        ``seeds`` replace the material RNG seeds of the reset envs (the rigid HOME is deterministic; B2 restores
        its settled snapshot). ``scene_seeds`` must equal the slots' scene seeds."""
        self._check_open()
        return self._infrastructure_guarded(self._reset_impl, seeds=seeds, env_ids=env_ids, scene_seeds=scene_seeds,
                                            options=options, seed=seed)

    def _reset_impl(self, *, seeds=None, env_ids=None, scene_seeds=None, options=None, seed=None):
        torch, N, env = self.torch, self.num_envs, self.env
        if options:
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"reset options {sorted(options)}")
        if seed is not None:
            if env_ids is not None or seeds is not None:
                raise ValueError("seed= is the full-reset Gymnasium form; use seeds= with env_ids")
            seeds = int(seed) + np.arange(N, dtype=np.int64)
        if env_ids is None:
            ids = np.arange(N)
        else:
            ids = np.asarray(env_ids.detach().cpu() if hasattr(env_ids, "detach") else env_ids, np.int64).reshape(-1)
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
                        f"env {int(e)} holds scene seed {self.task_family.scene_seeds[e]}, requested {s}")
        if ids.size == 0:
            return self._observe(), self._reset_info()
        index = torch.as_tensor(ids, dtype=torch.int64, device=self.torch_device)
        mask = torch.zeros(N, dtype=torch.bool, device=self.torch_device)
        mask[index] = True
        if env_ids is None:
            env.reset()
        else:
            env.reset_envs(mask)
        if seeds is not None:
            values = torch.as_tensor(seeds, dtype=torch.int64, device=self.torch_device)
            new = self._seed.clone()
            new[index] = values
            self._seed.copy_(new)
        self.episode.add_(mask.to(torch.int64))
        if self.input_guard is not None:
            hook = env.material_input_guard
            self.input_guard.reset(mask, hook.centers, hook.halves)
        if self.evaluation_stream is not None:
            self.evaluation_stream.reset(mask)
            self.evaluation_stream.capture(mask)
        return self._observe(), self._reset_info()

    # ------------------------------------------------------------------ step

    def step(self, actions):
        """``edge_delta_v1`` actions ``[N,8]`` for every world; returns obs, reward, terminated, truncated, info."""
        self._check_open()
        return self._infrastructure_guarded(self._step_impl, actions)

    def _step_impl(self, actions):
        torch, N, env = self.torch, self.num_envs, self.env
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (N, 8) \
                or actions.device != self.torch_device:
            raise ValueError(f"actions must be a torch tensor {(N, 8)} on {self.torch_device} (no implicit copies)")
        if not actions.dtype.is_floating_point:
            raise ValueError("actions must be a floating tensor")
        _, reward, terminated, truncated, extras = env.step(actions)
        terminated, truncated = terminated.clone(), truncated.clone()     # Lab reuses its buffers
        step_count = env.episode_length_buf.to(torch.int64).clone()
        if self.evaluation_stream is not None:
            self.evaluation_stream.capture(self.evaluation_stream.due(step_count, terminated | truncated))
        obs = self._observe()
        terms = dict(extras.get("reward_terms", {}))
        info = {"step_count": step_count, "action_status": env.action_status.to(torch.int32).clone(),
                "invalid_reason": self._invalid.to(torch.int32).clone(), "cut_digit": self._cut_digit.clone(),
                "action_clamped": env.action_clamped.clone(), "action_out_of_box": env.action_out_of_box.clone(),
                "drive": self.drive.clone(), "seed": self._seed.clone(), "episode": self.episode.clone(),
                "termination_reason": env.termination_reason.to(torch.int32).clone(),
                "rigid_reason": self.wp.to_torch(env.rigid_reason).clone(),
                "ik_streak": self.wp.to_torch(env.ik_streak).clone(), **terms}
        if self.input_guard is not None:
            info["input_guard_reason"] = self.input_guard.reason.clone()
        return obs, reward.to(torch.float32), terminated, truncated, info

    # ------------------------------------------------------------------ evaluation

    def snapshot_world(self, e):
        return self.env.snapshot_world(e)

    def evaluation_reference(self, e):
        return self.env.evaluation_reference(e)

    def evaluate(self, env_ids=None, *, frames=None, episode="current"):
        """Frozen goal evaluation of ``env_ids`` (host, episode end); modes as ``VectorCuttingEnv.evaluate``. No
        autoreset on this backend, so only ``episode="current"``. A non-finite env gets the same structured
        ``FAIL_INTEGRITY`` entry as workload A (``evaluation_stream.integrity_failure_result``)."""
        from .evaluation_stream import (DEFAULT_MODE, NONFINITE_MATERIAL_BITS, STREAM_VERSION, NonFiniteSnapshot,
                                        evaluate_frozen, integrity_failure_result)
        from .termination import decode_material_invalid, decode_termination
        self._check_open()
        if episode != "current":
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", "isaac_lab_newton has no same-step autoreset; "
                                                 "evaluate(episode='current') only")
        stream = self.evaluation_stream
        if stream is not None and frames:
            raise ValueError("frames= and the evaluation stream are exclusive")
        ids = range(self.num_envs) if env_ids is None else [int(e) for e in np.asarray(env_ids).reshape(-1)]
        invalid = self._invalid.to(self.torch.int32).cpu().clone().numpy()
        termination = self.env.termination_reason.to(self.torch.int32).cpu().clone().numpy()
        steps = self.env.episode_length_buf.to(self.torch.int64).cpu().clone().numpy()
        mode = DEFAULT_MODE if stream is None else STREAM_VERSION
        out = []
        for e in ids:
            h = self.env._env_host[e]
            extra = dict(scene_seed=self.task_family.scene_seeds[e], family_id=self.task_family.family_of(e),
                         episode_scored="current", invalid_reason=int(invalid[e]),
                         invalid_reasons=decode_material_invalid(invalid[e]), termination_reason=int(termination[e]),
                         termination_reasons=decode_termination(termination[e]))
            failure = dict(env=e, control_dt=self.control_dt, mode=mode, extra=extra, invalid_reason=invalid[e],
                           termination_reason=termination[e])
            if int(invalid[e]) & NONFINITE_MATERIAL_BITS:
                out.append(integrity_failure_result(h["task"], step_count=steps[e], cause="nonfinite_material_bit",
                                                    **failure))
                continue
            try:
                final = self.snapshot_world(e)
                if stream is None:
                    series = list((frames or {}).get(e, []))
                else:
                    series, dropped = stream.host_frames(e, h["T_world_local"], self.control_dt)
                    extra.update(stream_samples=len(series), stream_every=stream.every,
                                 stream_dropped_samples=dropped)
            except NonFiniteSnapshot as error:
                out.append(integrity_failure_result(h["task"], step_count=steps[e], cause=f"nonfinite_snapshot: {error}",
                                                    **failure))
                continue
            out.append(evaluate_frozen(h["task"], self.evaluation_reference(e), final, series, mode=mode, env=e,
                                       extra=extra))
        return out

    # ------------------------------------------------------------------ receipt

    def fidelity(self):
        fid = self.lab_options.fidelity()
        reasons = list(fid["reasons"])
        if self.max_episode_steps != self.declared_episode_steps:
            reasons.append(f"episode_steps={self.max_episode_steps} (reference {self.declared_episode_steps})")
        if self.task_family.particle_spacing_m is not None and self.task_family.particle_spacing_m != .001:
            reasons.append(f"particle_spacing_m={self.task_family.particle_spacing_m} (reference 0.001)")
        grip = grip_profile_record(self.grip_contact_profile, self.tool_mount)
        return {**fid, "reasons": reasons, "reduced_fidelity": bool(reasons), "computed_not_hand_set": True,
                "grip_contact_profile": grip["grip_contact_profile"],
                "contact_solver_profile": None if grip["contact_solver_profile"] is None
                else grip["contact_solver_profile"]["version"]}

    def receipt(self):
        import torch
        import warp as wp
        from .termination import termination_table
        root = Path(__file__).resolve().parent
        try:
            import gymnasium
            gym_version = gymnasium.__version__
        except ImportError:
            gym_version = None
        return dict(
            version=VECTOR_ENV_VERSION, api_revision=VECTOR_API_REVISION, runtime=RUNTIME, backend=LAB_BACKEND,
            num_envs=self.num_envs, device=str(self.torch_device), strict=self.strict, autoreset=False,
            observation_profile=self.observation_profile, observation=self.observation.metadata(),
            action_spec=self.env.action_spec.spec(), reward=self.reward.metadata(),
            task_family=self.task_family.to_dict(),
            tasks=[dict(env=e, scene_seed=self.task_family.scene_seeds[e], family_id=self.task_family.family_of(e),
                        task_id=t.task_id, config_sha256=t.config_sha256, initial_seed=self.task_family.env_seed(e))
                   for e, t in enumerate(self.tasks)],
            episode=dict(control_dt_s=self.control_dt, declared_steps=self.declared_episode_steps,
                         max_episode_steps=self.max_episode_steps),
            termination=termination_table(),
            input_guard={"enabled": False} if self.input_guard is None else self.input_guard.receipt(),
            evaluation_stream=None if self.evaluation_stream is None else self.evaluation_stream.receipt(),
            evaluation_mode="evaluation_stream_v1" if self.evaluation_stream is not None
            else "two_frame_reset_and_current_v1", infrastructure_error=self.infrastructure_error,
            terminal_capture={"enabled": False, "statement": "no same-step autoreset on isaac_lab_newton"},
            fidelity=self.fidelity(), lab=self.env.receipt(), material=self.material.receipt(),
            grip=grip_profile_record(self.grip_contact_profile, self.tool_mount),
            launch_simulation={"headless": True, "visualizer": [], "lifetime": "entered at open, exited in close()"},
            versions=dict(python=platform.python_version(), torch=torch.__version__, warp=wp.config.version,
                          numpy=np.__version__, gymnasium=gym_version),
            rl_source_sha256={name: _sha256_file(root / name) for name in (
                "lab_vector_env.py", "lab_batched_env.py", "vector_env.py", "termination.py", "material_guard.py",
                "evaluation_stream.py")},
            build_wall_s=getattr(self, "build_wall_s", None), initial_reset_wall_s=self.initial_reset_wall_s,
            calibration="SIMULATOR_ONLY_UNCALIBRATED", success_source="evaluate() only (frozen evaluator)")

    # ------------------------------------------------------------------ lifetime

    def close(self):
        if self._closed:
            return
        errors = []
        if self.output_dir is not None:
            try:
                (self.output_dir / "vector_env_receipt_close.json").write_text(
                    json.dumps(self.receipt(), indent=1, sort_keys=True, default=_json_default) + "\n")
            except Exception as error:
                (self.output_dir / "vector_env_close_error.txt").write_text(repr(error) + "\n")
        try:
            if self.torch_device.type == "cuda":
                self.wp.synchronize_device(f"cuda:{self.torch_device.index}")
        except Exception as error:      # an unusable device after an infrastructure error: keep releasing
            errors.append(error)
        try:
            self.material.release()      # the device HashGrid
        except Exception as error:
            errors.append(error)
        try:
            self.env.close()
        except Exception as error:
            errors.append(error)
        finally:
            self._stack.close()          # exits launch_simulation
            self._closed = True
        self.close_errors = [repr(e) for e in errors]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["LabVectorCuttingEnv", "LAB_OPTIONS", "check_grip_contact_profile", "grip_profile_record"]
