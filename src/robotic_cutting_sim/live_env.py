"""Shared research API over actual Isaac Lab and Isaac Sim cutting runtimes.

The wrapper has no physics, IK or renderer of its own. Native backends retain
ownership of those operations; oracle history is used only by evaluate().
"""
from contextlib import contextmanager
import copy
from dataclasses import asdict

import numpy as np

from .control import DELTA_FRAMES, DELTA_MODE, ActionLimits, IKFailure, bound_target, nominal_edge_pose, resolve_delta, validate_cut_index
from .research_env import proprio_observation, vision_observation
from .tasks import ACTION_MODES, make_task

OBSERVATION_PROFILES = ("vision_proprio", "vision_only", "state")
IK_FAILURE_MODES = ("raise", "reject")
WRAPPER_OPTIONS = ("action_limits", "delta_frame", "ik_failure", "max_consecutive_ik_failures", "reward", "history_stride")


class CuttingTaskEnv:
    """reset/step/observe/render/close/evaluate for both shared cutting tasks.

    A mapping action contains mode, target and optional cut_enabled. Cartesian
    targets are absolute WORLD knife-edge XYZ+XYZW; joint targets are seven
    active-arm positions in radians. None runs the scripted reference baseline.
    Explicit actions arm cutting independently of the reference-path schedule.
    The second Franka holds its normal park posture. No force or oracle state
    enters observe(); Global RGB-D is an explicit opt-in.

    Action API v2 (all opt-in; defaults keep the historical behavior):
    ``cartesian_edge_delta`` moves the commanded edge by [dx,dy,dz,rx,ry,rz]
    in the edge or world frame, bounded by ``action_limits``; absolute Cartesian
    targets are bounded too when ``action_limits.limit_absolute``. With
    ``ik_failure="reject"`` an unreachable target is not executed: the step
    holds the last accepted target and ``info["ik_status"]`` says so. The
    ``state`` profile returns measured proprioception without images. ``reward``
    is an optional privileged callable ``reward(env, info) -> float``.
    """

    def __init__(self, backend, task_id=None, *, seed=42, action_mode=ACTION_MODES[0],
                 observation_profile="vision_proprio", include_global_camera=False,
                 action_limits=None, delta_frame="edge", ik_failure="raise",
                 max_consecutive_ik_failures=None, reward=None, history_stride=1):
        if (action_mode not in ACTION_MODES+(DELTA_MODE,) or observation_profile not in OBSERVATION_PROFILES
                or delta_frame not in DELTA_FRAMES or ik_failure not in IK_FAILURE_MODES):
            raise ValueError("Unsupported action mode, observation profile, delta frame or IK failure mode")
        if max_consecutive_ik_failures is not None and int(max_consecutive_ik_failures) < 1:
            raise ValueError("max_consecutive_ik_failures must be positive or None")
        if not isinstance(history_stride, (int, np.integer)) or history_stride < 1:
            raise ValueError('History stride must be a positive integer')
        self.history_stride, self._control_steps_count = int(history_stride), 0
        task = getattr(backend, "task", None)
        self.task = task or make_task(task_id, seed=seed)
        self.backend, self.action_mode, self.observation_profile = backend, action_mode, observation_profile
        self.include_global_camera = bool(include_global_camera)
        self.action_limits = action_limits
        self.delta_frame, self.ik_failure, self.reward = delta_frame, ik_failure, reward
        self.max_consecutive_ik_failures = max_consecutive_ik_failures
        self._closed, self._ready = False, False
        self._history, self._observation = [], None
        self._proprio, self._commanded, self._ik_failures = None, None, 0

    def reset(self, seed=None):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if seed is None:
            seed = getattr(self.task, "seed", 42)
        self.backend.reset(seed=int(seed))
        self.task = self.backend.task
        self._history, self._ready = [], True
        self._terminated = False
        self._ik_failures = 0
        self._control_steps_count = 0
        self._refresh()
        self._commanded = self._nominal_edge()
        self._record_oracle()
        return self.observe(), {"task_id": self.task.task_id, "seed": int(seed),
            "runtime_source": self.runtime_source, "task_protocol_sha256": self.task.to_dict()["protocol_sha256"]}

    @property
    def runtime_source(self):
        declared = getattr(self.backend, "runtime_source", None)
        if isinstance(declared, str):
            return declared
        return ("live_isaac_lab" if hasattr(self.backend, "measured") else "live_isaac_sim")

    @property
    def time_s(self):
        return float(self.backend.time_s)

    def _require_ready(self):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._ready:
            raise RuntimeError("Call reset on the environment first")

    def _refresh(self):
        if hasattr(self.backend, "measured"):
            self._frames, proprio = self.backend.frames, self.backend.measured()
        else:
            if self.observation_profile != 'state':
                self._frames, _ = self.backend.capture()
            else:
                self._frames = {}
            proprio = self.backend.proprioception()
        self._proprio = proprio
        if self.observation_profile == "state":
            # Same measured allowlist as vision_proprio, without any image.
            self._observation = {"timestamp_s": self.time_s, "proprio": proprio_observation(proprio, self.time_s)}
            return
        self._observation = vision_observation(self._frames, self.backend.calibration,
            self.time_s, self.observation_profile, proprio,
            include_global_camera=self.include_global_camera)

    def _nominal_edge(self):
        hand_from_edge = getattr(self.task, "hand_from_edge_xyz_xyzw", None)
        if hand_from_edge is None or "hand_pose" not in self._proprio:
            return np.asarray(self._proprio["tool_pose"], float).copy()
        return nominal_edge_pose(self._proprio, hand_from_edge)

    def _limits(self):
        return self.action_limits or ActionLimits()

    def _resolve(self, action):
        """Delta/bounded actions -> one validated absolute action and a report."""
        report = None
        if action.get("mode") == DELTA_MODE:
            # A delta may name its own frame ("edge" | "world"); otherwise the env default.
            frame = action.get("frame", self.delta_frame)
            target, report = resolve_delta(self._commanded, action.get("target"), self._limits(),
                frame=frame, T_world_workpiece=self.task.T_world_workpiece)
            action = {key: value for key, value in action.items() if key != "frame"}
            action.update({"mode": ACTION_MODES[0], "target": target})
        elif action.get("mode") == ACTION_MODES[0] and self.action_limits and self.action_limits.limit_absolute:
            target, report = bound_target(self._commanded, action.get("target"), self.action_limits,
                                          self.task.T_world_workpiece)
            action = {**action, "target": target}
        return action, report

    def _record_oracle(self, force=False):
        if self._history and not force and self._control_steps_count % self.history_stride:
            return
        material = self.backend.material
        state = (material.history_state() if hasattr(material, 'history_state') else material.world_state())
        self._history.append({"time_s": self.time_s,
            **{name: state[name].copy() for name in ("positions", "velocities", "codes")}})

    def step(self, action=None):
        self._require_ready()
        if self.time_s >= self.task.duration_s-1e-9:
            raise RuntimeError("Episode ended; call reset")
        if self._terminated:
            raise RuntimeError("Grasp lost or sensor invalid; call reset")
        report = None
        requested_reference = None
        if action is None:
            reference = getattr(self.task, 'reference_edge_pose_world', None)
            if reference is not None:
                requested_reference = reference(self.time_s + self.task.control_dt_s)
        if action is not None:
            validate_cut_index(action, self.backend.material)
            gripper = None
            if not isinstance(action, dict):
                action = {"mode": self.action_mode, "target": action, "cut_enabled": True}
            if "gripper" in action:
                action = dict(action)
                gripper = action.pop("gripper")
                options=getattr(self.backend,"options",getattr(self.backend,"config",None))
                if getattr(options,"tool_attachment_mode",None) != "physical_grasp":
                    raise ValueError("Gripper release/hold requires physical_grasp")
                if gripper not in ("hold","close","open","release"):
                    raise ValueError("Unknown physical gripper command")
                if not action:
                    measured=(self.backend.measured() if hasattr(self.backend,"measured")
                              else self.backend.proprioception())
                    action={"mode":"joint_position_rad","target":measured["q"][0],"cut_enabled":False}
            if action:
                action, report = self._resolve(action)
                if report is not None:
                    report['resolved_target_world_xyz_xyzw'] = np.asarray(action['target']).tolist()
            action = self.task.validate_action(action)
            if gripper is not None:
                self.backend.set_gripper(gripper)
        ik_status, ik_info = "ok", None
        try:
            self.backend.step(action)
        except IKFailure as failure:
            if self.ik_failure == "raise":
                raise
            # Both runtimes fail before writing targets: hold instead of moving.
            ik_status, ik_info = "rejected", failure.info
            self._ik_failures += 1
            hold = {"mode": ACTION_MODES[0], "target": self._commanded,
                    "cut_enabled": bool(action["cut_enabled"]) if action else True,
                    **({"cut_index": action["cut_index"]} if action and "cut_index" in action else {})}
            try:
                self.backend.step(self.task.validate_action(hold))
            except IKFailure:
                hold = {**hold, "mode": ACTION_MODES[1], "target": np.asarray(self._proprio["q"])[0]}
                self.backend.step(self.task.validate_action(hold))
        else:
            self._ik_failures = 0
            if action is not None and action["mode"] == ACTION_MODES[0]:
                self._commanded = np.asarray(action["target"], float).copy()
        self._refresh()
        if action is None or action["mode"] != ACTION_MODES[0]:
            self._commanded = self._nominal_edge()
        self._control_steps_count += 1
        # Goal success is deliberately evaluated separately from elapsed time.
        diagnostics=getattr(self.backend,"last_diagnostics",{})
        grasp_payload = diagnostics.get("grasp_status", {})
        grasp_status = grasp_payload.get('status') if isinstance(grasp_payload, dict) else grasp_payload
        self._terminated = bool(diagnostics.get("grasp_lost",False) or grasp_status == "INVALID_CONTACT_DATA")
        reason = grasp_status if self._terminated else None
        if (not self._terminated and self.max_consecutive_ik_failures is not None
                and self._ik_failures >= self.max_consecutive_ik_failures):
            self._terminated, reason = True, "IK_FAILURES"
        self._record_oracle(force=self._terminated or self.time_s >= self.task.duration_s-1e-9)
        info = {"task_id": self.task.task_id, "timestamp_s": self.time_s, "runtime_source": self.runtime_source,
                "grasp_status": grasp_status, "termination_reason": reason,
                "ik_status": ik_status, "ik_info": ik_info, "action_report": report,
                "consecutive_ik_failures": self._ik_failures}
        if requested_reference is not None:
            info['requested_reference_target_world_xyz_xyzw'] = np.asarray(requested_reference).tolist()
        reward = 0. if self.reward is None else float(self.reward(self, info))
        return self.observe(), reward, self._terminated, self.time_s >= self.task.duration_s-1e-9, info

    def observe(self):
        self._require_ready()
        return copy.deepcopy(self._observation)

    def render(self, camera="global", mode="rgb_array"):
        self._require_ready()
        if camera == "global" and "global_d455" in self._frames:
            camera = "global_d455"
        if camera not in self._frames or mode not in ("rgb_array", "depth_array"):
            raise ValueError("Choose an available camera and rgb_array/depth_array")
        return self._frames[camera]["rgb" if mode == "rgb_array" else "depth_m"].copy()

    def evaluate(self, profile=None, *, goal=None, represented_target=None, push_interval_s=None):
        """Independent oracle score; a completed episode alone never means success.

        A v3 demo task (tasks.DemoTaskSpec) with neither ``profile`` nor
        ``goal`` returns its demo report (``task.evaluate_demo``); the frozen
        two-part oracle is NOT_APPLICABLE there. ``push_interval_s`` is the
        recorded actual push phase (else the task reference schedule is used,
        and the report says so).
        """
        self._require_ready()
        if len(self._history) < 2:
            return {"task_id": self.task.task_id, "status": "NOT_RUN", "success": False,
                    "episode_complete": False, "reason": "No physical step recorded since reset",
                    "runtime_source": self.runtime_source}
        material = self.backend.material
        bed = material.bed
        if getattr(self.task, "evaluate_goal", None) is not None and profile is None and goal is None:
            return self._evaluate_configured_goal(material, bed)
        refresh_surface = getattr(self.backend, 'refresh_surface_for_evaluation', None)
        if refresh_surface is not None:
            refresh_surface()
        if hasattr(self.backend, "surface_arrays"):
            vertices, triangles = self.backend.surface_arrays
            metadata = self.backend.surface_info
        else:
            vertices, triangles = self.backend.surface
            metadata = self.backend.surface_metadata
        face_components = np.full(len(triangles), -1, np.int32)
        for row in metadata["component_ranges"]:
            first = row["first_triangle"]
            face_components[first:first+row["triangles"]] = row["mechanical_component"]
        if np.any(face_components < 0):
            raise ValueError("Every scored triangle needs actual current mechanical provenance")
        data = {
            "time_s": [s["time_s"] for s in self._history],
            **{name: np.asarray([s[name] for s in self._history]) for name in ("positions", "velocities", "codes")},
            "mass_kg": material.masses, "reference_positions": material.world_points,
            "reference_stencil_nodes": bed.nodes.numpy(), "reference_stencil_weights": bed.weights.numpy(),
            "reference_stencil_gradients": bed.gradients.numpy(),
            "surface_vertices_world": vertices, "surface_triangles": triangles,
            "surface_triangle_component_ids": face_components,
            "surface_component_provenance": metadata["partition_rule"], "surface_timestamp_s": self.time_s}
        from .tasks import DemoTaskSpec
        packed_cuts = (int(material.max_cuts) if getattr(material, 'code_encoding', 'ternary') == 'packed2'
                       and getattr(material, 'max_cuts', 1) > 1 else None)
        if packed_cuts is not None:
            # tlmpm_v5 packed history (ADR 0014; K=1 packed equals ternary). Multi-cut goals use the
            # packed evaluator; the ternary demo/oracle evaluators get the exact ternary re-encoding
            # only while it has at most two digits.
            from .multi_cut_evaluation import MultiCutGoal, MultiCutGoalK, evaluate_multi_cut_packed
            if isinstance(goal, (MultiCutGoal, MultiCutGoalK)):
                data.update(reference_indices=material.lattice, code_encoding='packed2', max_cuts=packed_cuts)
                if isinstance(goal, MultiCutGoal):
                    goal = MultiCutGoalK(goal.planes, 3, goal.minimum_part_mass_fraction,
                                         goal.minimum_main_parts_mass_fraction, goal.plane_error_bound_m)
                return {**evaluate_multi_cut_packed(data, goal), 'runtime_source': self.runtime_source}
            if packed_cuts > 2:
                return {'status': 'NOT_RUN_EXPLICIT_MULTICUT_GOAL_REQUIRED', 'success': False,
                        'runtime_source': self.runtime_source,
                        'reason': 'Declare MultiCutGoalK to score more than two packed cut digits'}
            from .cut_topology import packed_to_ternary
            data['codes'] = packed_to_ternary(data['codes'], packed_cuts).astype(data['codes'].dtype)
            data['code_encoding_scored'] = 'ternary_from_packed2'
        if isinstance(self.task, DemoTaskSpec) and profile is None and goal is None:
            data['reference_indices'] = material.lattice
            if push_interval_s is not None:
                data['push_interval_s'] = tuple(float(value) for value in push_interval_s)
            return {**self.task.evaluate_demo(data), 'runtime_source': self.runtime_source}
        if getattr(material, 'max_cuts', 1) > 1:
            from .multi_cut_evaluation import MultiCutGoal, evaluate_multi_cut
            if not isinstance(goal, MultiCutGoal):
                return {'status':'NOT_RUN_EXPLICIT_MULTICUT_GOAL_REQUIRED', 'success':False,
                    'runtime_source':self.runtime_source,
                    'reason':'Declare MultiCutGoal before scoring two-digit history; the v1 oracle is single-cut'}
            data['reference_indices'] = material.lattice
            return {**evaluate_multi_cut(data, goal), 'runtime_source':self.runtime_source}
        if profile is not None:
            from .evaluation import evaluate_recording
            import hashlib
            nodes, weights, gradients = [data[k] for k in ("reference_stencil_nodes", "reference_stencil_weights", "reference_stencil_gradients")]
            data["surface_graph_verified"] = (metadata.get("stencil_sha256") == hashlib.sha256(nodes.tobytes()+weights.tobytes()+gradients.tobytes()).hexdigest()
                and metadata.get("codes_sha256") == hashlib.sha256(data["codes"][-1].tobytes()).hexdigest()
                and metadata.get("future_component_ids_used") is False)
            data["geometry_frame_id"] = "workpiece_local"
            data["source_evidence"] = {"implementation_verified": False,
                "physics_state_from_recording": True, "contact_deformation_traceable": False,
                "scope": "Live in-memory measurements; archived source/contact validation requires the offline evaluator"}
            result, _ = evaluate_recording(data, task=self.task, goal=goal, profile=profile,
                represented_target=represented_target)
            return {**result, "runtime_source": self.runtime_source}
        result = self.task.evaluate_oracle(data)
        return {**result, "success": result["status"] == "PASS_TASK_METRICS",
                "episode_complete": self.time_s >= self.task.duration_s-1e-9, "runtime_source": self.runtime_source}

    def _evaluate_configured_goal(self, material, bed):
        # Configured task (scene_config.ConfiguredTaskSpec, ADR 0015): its declared goal's evaluator.
        # The extracted surface is optional quality evidence here; a failed
        # extraction is recorded and never blocks the outcome.
        data = {"time_s": [s["time_s"] for s in self._history],
                **{name: np.asarray([s[name] for s in self._history]) for name in ("positions", "velocities", "codes")},
                "mass_kg": material.masses, "reference_positions": material.world_points,
                "reference_indices": material.lattice, "reference_stencil_nodes": bed.nodes.numpy(),
                "reference_stencil_weights": bed.weights.numpy(), "reference_stencil_gradients": bed.gradients.numpy(),
                "max_cuts": int(getattr(material, "max_cuts", 1)),
                "code_encoding": getattr(material, "code_encoding", "ternary"),
                "particle_spacing_m": float(self.task.particle_spacing_m)}
        surface_status = "MEASURED"
        try:
            refresh_surface = getattr(self.backend, "refresh_surface_for_evaluation", None)
            if refresh_surface is not None:
                refresh_surface()
            if hasattr(self.backend, "surface_arrays"):
                (vertices, triangles), metadata = self.backend.surface_arrays, self.backend.surface_info
            else:
                (vertices, triangles), metadata = self.backend.surface, self.backend.surface_metadata
            face_components = np.full(len(triangles), -1, np.int32)
            for row in metadata["component_ranges"]:
                face_components[row["first_triangle"]:row["first_triangle"]+row["triangles"]] = row["mechanical_component"]
            data.update(surface_vertices_world=vertices, surface_triangles=triangles,
                        surface_triangle_component_ids=face_components)
        except Exception as error:  # quality evidence only; the outcome never depends on it
            surface_status = f"NOT_RUN: surface extraction failed ({error!r})"
        result = self.task.evaluate_goal(data)
        return {**result, "surface_extraction": surface_status, "runtime_source": self.runtime_source,
                "episode_complete": self.time_s >= self.task.duration_s-1e-9}

    def close(self):
        if not self._closed:
            self.backend.close()
            self._closed = True
            self._history.clear()
            self._observation = None


@contextmanager
def open_cutting_env(task_id="CUT-01", *, runtime="isaac_sim", output_dir, seed=42,
                     action_mode=ACTION_MODES[0], observation_profile="vision_proprio",
                     include_global_camera=False, camera_layout="wrist_d405_global_d455", task=None, **backend_options):
    """Start a pinned live runtime in its dedicated venv, then close it on exit.

    Use a fresh output directory. Isaac Sim defaults to real native SingleView
    RGB-D; the separate Lab kit-less route explicitly uses ideal Warp RGB-D.
    A process should own only one Isaac runtime at a time.

    ``runtime="kinematic"`` (opt-in) opens the ideal-tracking CPU/GPU backend
    with the same API and ``observation_profile="state"``. ``task`` is a compiled
    ``scene_config.ConfiguredTaskSpec`` (ADR 0015); it currently runs on the
    kinematic runtime only (the Isaac runtimes build tasks from registered ids
    and raise NotImplementedError rather than silently using another task).
    """
    if task is not None and runtime != "kinematic":
        raise NotImplementedError("Isaac Sim/Lab build tasks from registered task ids; a compiled ConfiguredTaskSpec "
                                  "runs on runtime='kinematic' until a live passthrough is implemented")
    if runtime == "kinematic":
        yield from _kinematic_session(task, task_id, output_dir, seed, action_mode, observation_profile, backend_options)
        return
    from .recording import manifest, new_output, write_json
    if runtime not in ("isaac_sim", "isaac_lab") or action_mode not in ACTION_MODES+(DELTA_MODE,):
        raise ValueError("Choose a supported live runtime and action mode")
    make_task(task_id, seed=seed)
    wrapper_options = {key: backend_options.pop(key) for key in WRAPPER_OPTIONS if key in backend_options}
    output = new_output(output_dir)
    wrapper = None
    if runtime == "isaac_sim":
        from .isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app
        mode = "joint" if action_mode == ACTION_MODES[1] else "cartesian"
        config = IsaacSimCuttingConfig(task_id=task_id, seed=seed, action_mode=mode,
            **{"camera_mode": "native_single_view", "camera_layout": camera_layout, **backend_options})
        write_json(output/"manifest.json", manifest({**asdict(config),
            "formal_observation": {"profile": observation_profile, "include_global_camera": bool(include_global_camera)}}, "live_isaac_sim_shared_api"))
        app = launch_app(config, output)
        try:
            wrapper = CuttingTaskEnv(LiveIsaacSimCutting(app, config, output), task_id, seed=seed,
                action_mode=action_mode, observation_profile=observation_profile,
                include_global_camera=include_global_camera, **wrapper_options)
            yield wrapper
        finally:
            if wrapper:
                wrapper.close()
            else:
                app.close(wait_for_replicator=False)
    elif runtime == "isaac_lab":
        from argparse import Namespace
        from isaaclab.app import launch_simulation
        from .isaac_lab_env import IsaacLabCuttingEnv, LabCuttingConfig, make_lab_cfg
        from .workcell import export
        static = output/"workcell"
        static.mkdir()
        export(Namespace(variant="D0", family="dome", seed=seed, spacing=.002, asset_root=None), static)
        config = LabCuttingConfig(task_id=task_id, seed=seed,
            action_mode=ACTION_MODES[0] if action_mode == DELTA_MODE else action_mode,
            static_scene=str((static/"workcell.usda").resolve()), camera_layout=camera_layout,
            include_global_camera=include_global_camera, **backend_options)
        write_json(output/"manifest.json", manifest({**asdict(config),
            "formal_observation": {"profile": observation_profile, "include_global_camera": bool(include_global_camera)}}, "live_isaac_lab_shared_api"))
        cfg = make_lab_cfg(config)
        with launch_simulation(cfg, {"headless": True, "visualizer": [], "device": config.device}):
            wrapper = CuttingTaskEnv(IsaacLabCuttingEnv(cfg, config), task_id, seed=seed,
                action_mode=action_mode, observation_profile=observation_profile,
                include_global_camera=include_global_camera, **wrapper_options)
            try:
                yield wrapper
            finally:
                wrapper.close()
    else:
        raise ValueError("runtime must be isaac_sim or isaac_lab")


def _kinematic_session(task, task_id, output_dir, seed, action_mode, observation_profile, options):
    """open_cutting_env(runtime="kinematic"): the ideal-tracking backend behind the same API."""
    from .recording import manifest, new_output, write_json
    if observation_profile != "state":
        raise ValueError("The kinematic runtime renders no cameras; use observation_profile='state'")
    output = new_output(output_dir)
    options = dict(options, seed=seed, action_mode=action_mode)
    if task is not None:
        from .scene_config import make_configured_kinematic_env
        env = make_configured_kinematic_env(task, **options)
    else:
        from .kinematic_env import make_kinematic_env
        env = make_kinematic_env(task_id, **options)
    write_json(output/"manifest.json", manifest({"runtime": "kinematic", "task_id": env.task.task_id, "seed": seed,
        "configured_task_sha256": getattr(env.task, "config_sha256", None), "action_mode": action_mode,
        "formal_observation": {"profile": observation_profile}}, "kinematic_shared_api"))
    try:
        yield env
    finally:
        env.close()
