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
WRAPPER_OPTIONS = ("action_limits", "delta_frame", "ik_failure", "max_consecutive_ik_failures", "reward", "history_stride",
                   "expose_episode_state")


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

    Episode lifecycle (WP-11 Phase D, ``episode_lifecycle``): every info carries
    ``episode_state`` (state, termination reason, home outcome; never material or
    force data); ``expose_episode_state=True`` (opt-in) also puts it in the
    observation under ``"episode"``. ``end_task(reason)`` ends the task phase;
    ``return_home()`` is the separate physical return-to-home motion with its own
    ``home_outcome``. ``reset()`` stays the simulator state reset.
    """

    def __init__(self, backend, task_id=None, *, seed=42, action_mode=ACTION_MODES[0],
                 observation_profile="vision_proprio", include_global_camera=False,
                 action_limits=None, delta_frame="edge", ik_failure="raise",
                 max_consecutive_ik_failures=None, reward=None, history_stride=1, task=None,
                 expose_episode_state=False):
        if (action_mode not in ACTION_MODES+(DELTA_MODE,) or observation_profile not in OBSERVATION_PROFILES
                or delta_frame not in DELTA_FRAMES or ik_failure not in IK_FAILURE_MODES):
            raise ValueError("Unsupported action mode, observation profile, delta frame or IK failure mode")
        if max_consecutive_ik_failures is not None and int(max_consecutive_ik_failures) < 1:
            raise ValueError("max_consecutive_ik_failures must be positive or None")
        if not isinstance(history_stride, (int, np.integer)) or history_stride < 1:
            raise ValueError('History stride must be a positive integer')
        self.history_stride, self._control_steps_count = int(history_stride), 0
        # ``task``: an explicit task object (e.g. a compiled ConfiguredTaskSpec) for runtimes
        # that build their task at reset; otherwise the backend's task or the registered id.
        self.task = task or getattr(backend, "task", None) or make_task(task_id, seed=seed)
        from .episode_lifecycle import EpisodeStateMachine
        self.lifecycle = EpisodeStateMachine()
        self.expose_episode_state = bool(expose_episode_state)
        self.last_return_home = None
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
        self.lifecycle.transition("RESETTING", time_s=getattr(self.backend, "time_s", None), reason="reset")
        self._ready = False
        self.backend.reset(seed=int(seed))
        self.task = self.backend.task
        self._history, self._ready = [], True
        self._terminated = False
        self._ik_failures = 0
        self._control_steps_count = 0
        self.last_return_home = None
        self._refresh()
        self._reset_proprio = copy.deepcopy(self._proprio)
        self._commanded = self._nominal_edge()
        self._record_oracle()
        self.lifecycle.transition("READY", time_s=self.time_s, reason="reset_complete")
        return self.observe(), {"task_id": self.task.task_id, "seed": int(seed),
            "runtime_source": self.runtime_source, "task_protocol_sha256": self.task.to_dict()["protocol_sha256"],
            "episode_state": self.lifecycle.snapshot()}

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
        self.lifecycle.require(("READY", "RUNNING"), "step")
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
        if self.lifecycle.state == "READY":  # only once a control step was actually executed
            self.lifecycle.transition("RUNNING", time_s=self.time_s, reason="first_step")
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
        if self._terminated:
            self.lifecycle.transition("TASK_FAILED", time_s=self.time_s, reason=reason)
        elif self.time_s >= self.task.duration_s-1e-9:
            self.lifecycle.transition("TASK_COMPLETE", time_s=self.time_s, reason="TIME_LIMIT")
        info = {"task_id": self.task.task_id, "timestamp_s": self.time_s, "runtime_source": self.runtime_source,
                "grasp_status": grasp_status, "termination_reason": reason,
                "ik_status": ik_status, "ik_info": ik_info, "action_report": report,
                "consecutive_ik_failures": self._ik_failures, "episode_state": self.lifecycle.snapshot()}
        if requested_reference is not None:
            info['requested_reference_target_world_xyz_xyzw'] = np.asarray(requested_reference).tolist()
        reward = 0. if self.reward is None else float(self.reward(self, info))
        return self.observe(), reward, self._terminated, self.time_s >= self.task.duration_s-1e-9, info

    def observe(self):
        self._require_ready()
        observation = copy.deepcopy(self._observation)
        if self.expose_episode_state:
            observation["episode"] = self.lifecycle.snapshot()
        return observation

    @property
    def episode_state(self):
        return self.lifecycle.state

    def end_task(self, reason="CONTROLLER_DONE"):
        """End the task phase normally (TASK_COMPLETE) before the time limit; the task is not scored here."""
        self._require_ready()
        self.lifecycle.require(("RUNNING",), "end_task")
        if not self._history or self._history[-1]["time_s"] != self.time_s:  # never a duplicate timestamp
            self._record_oracle(force=True)
        self.lifecycle.transition("TASK_COMPLETE", time_s=self.time_s, reason=str(reason))
        return self.lifecycle.snapshot()

    def return_home(self, *, planner=None, path_checker=None, on_step=None, config=None):
        """Physical return-to-home after the task phase (episode_lifecycle); never a teleport.

        WITHDRAW/lift -> hand-down -> q_home as preflighted joint targets executed by the
        runtime with cutting disabled and the grasp monitored. Returns
        ``{"home_outcome": {status, reason, checks}, "measurement": ..., "plan": ...}``;
        the task outcome (``evaluate()``) is unaffected. ``planner(q_start)`` and
        ``path_checker(qs, modes)`` default to the backend's ``plan_return_home`` /
        ``check_measured_path``. ``on_step(action, result, before_observation, label)``
        is called after every executed control step (e.g. a recorder).
        """
        from .episode_lifecycle import TASK_ENDED, evaluate_home_outcome, load_return_home_config
        self._require_ready()
        config = config or load_return_home_config()
        machine = self.lifecycle
        if machine.state == "RUNNING":
            self.end_task("RETURN_HOME_REQUESTED")
        machine.require(TASK_ENDED, "return_home")
        home = getattr(self.backend, "home", None)
        planner = planner or getattr(self.backend, "plan_return_home", None)
        path_checker = path_checker or getattr(self.backend, "check_measured_path", None)
        if home is None or planner is None:
            outcome = {"status": "NOT_APPLICABLE", "reason": "NO_DECLARED_HOME" if home is None else "RUNTIME_CANNOT_PLAN",
                       "checks": {}, "scope": "No physical return-to-home exists for this runtime/scene"}
            machine.home_outcome = outcome["status"]
            machine.transition("FINISHED", time_s=self.time_s, reason=f"home_{outcome['reason']}")
            self.last_return_home = {"home_outcome": outcome, "measurement": None, "plan": None}
            return self.last_return_home
        machine.transition("RETRACTING", time_s=self.time_s, reason="return_home")
        try:
            result = self._execute_return_home(home, planner, path_checker, on_step, config)
        except BaseException as error:
            outcome = {"status": "FAIL", "reason": f"EXCEPTION:{type(error).__name__}", "checks": {}, "error": repr(error)}
            machine.home_outcome = outcome["status"]
            if machine.state in ("RETRACTING", "RETURNING_HOME"):
                machine.transition("FINISHED", time_s=self.time_s, reason=outcome["reason"])
            self.last_return_home = {"home_outcome": outcome, "measurement": None, "plan": None}
            raise
        outcome = evaluate_home_outcome(result["measurement"], config)
        result["home_outcome"] = outcome
        machine.home_outcome = outcome["status"]
        machine.transition("FINISHED", time_s=self.time_s,
                           reason="home_PASS" if outcome["status"] == "PASS" else f"home_FAIL:{outcome['reason']}")
        self.last_return_home = result
        return result

    def _execute_return_home(self, home, planner, path_checker, on_step, config):
        from .episode_lifecycle import pose_error
        machine = self.lifecycle
        q_home = np.asarray(home["joint_positions"], float)
        start_time = self.time_s
        q_start = np.asarray(self._proprio["q"], float)[0]
        plan = planner(q_start)
        targets = np.asarray(plan.get("q", []), float).reshape(-1, 7)
        labels = list(plan.get("labels", []))
        summary = {key: plan.get(key) for key in ("status", "reason", "variant", "attempts", "retract", "preflight",
                                                  "home_edge_pose_world_xyzw", "start_edge_pose_world_xyzw", "limits")}
        measurement = {"plan_status": plan.get("status"), "plan_reason": plan.get("reason"), "executed": False,
                       "aborted_reason": None, "grasp_lost": bool(self._terminated), "steps": 0,
                       "start_time_s": start_time, "q_start": q_start.tolist(), "q_home": q_home.tolist()}
        if plan.get("status") != "PLANNED" or not len(targets):
            return {"measurement": measurement, "plan": summary}
        codes_before = np.asarray(self.backend.material.history_state()["codes"]).copy()
        measured_q, slip, rotation, contacts, penetration, reaction = [q_start], [], [], [], [], []
        aborted = None
        substeps = int(getattr(getattr(self.backend, "config", None), "robot_substeps", 1) or 1)
        for target, label in zip(targets, labels):
            if label != "RETRACTING" and machine.state == "RETRACTING":
                machine.transition("RETURNING_HOME", time_s=self.time_s, reason=label)
            action = self.task.validate_action({"mode": ACTION_MODES[1], "target": target, "cut_enabled": False})
            before = self.observe() if on_step is not None else None
            self.backend.step(action)
            self._refresh()
            self._commanded = self._nominal_edge()
            measurement["steps"] += 1
            diagnostics = getattr(self.backend, "last_diagnostics", {}) or {}
            grasp = diagnostics.get("grasp_status", {})
            status = grasp.get("status") if isinstance(grasp, dict) else grasp
            if isinstance(grasp, dict):
                slip.append(float(grasp.get("slip_since_settling_m", 0.)))
                rotation.append(float(grasp.get("rotation_since_settling_rad", 0.)))
            for row in (getattr(self.backend, "grasp_trace", None) or [])[-substeps:]:
                contact = row.get("contacts", {})
                contacts.append(int(contact.get("unintended_contact_count", 0)))
                for record in (contact.get("tensor_forces") or {}).get("records", []):
                    if record.get("separation_m") is not None:
                        penetration.append(max(0., -float(record["separation_m"])))
            wrench = diagnostics.get("next_material_wrench_world")
            if wrench is not None:
                reaction.append(float(np.linalg.norm(np.asarray(wrench, float)[:3])))
            measured_q.append(np.asarray(self._proprio["q"], float)[0])
            lost = bool(diagnostics.get("grasp_lost", False) or status == "INVALID_CONTACT_DATA")
            info = {"task_id": self.task.task_id, "timestamp_s": self.time_s, "runtime_source": self.runtime_source,
                    "grasp_status": status, "termination_reason": "GRASP_LOST" if lost else None,
                    "ik_status": "not_used_joint_space_target", "ik_info": None, "action_report": None,
                    "consecutive_ik_failures": 0, "return_home_phase": label, "episode_state": machine.snapshot()}
            if on_step is not None:
                on_step(action, (self.observe(), 0., lost, False, info), before, label)
            if lost:
                aborted = "GRASP_LOST"
                break
        if machine.state == "RETRACTING":
            machine.transition("RETURNING_HOME", time_s=self.time_s, reason="aborted" if aborted else "no_joint_segment")
        codes_after = np.asarray(self.backend.material.history_state()["codes"])
        final = self._proprio
        reset_tool = np.asarray(self._reset_proprio.get("tool_body_pose_xyzw", self._reset_proprio["tool_pose"]), float)
        tool_now = np.asarray(final.get("tool_body_pose_xyzw", final["tool_pose"]), float)
        tool_t, tool_r = pose_error(reset_tool, tool_now)
        edge_t, edge_r = pose_error(np.asarray(self._reset_proprio["tool_pose"], float), np.asarray(final["tool_pose"], float))
        measured_q = np.asarray(measured_q)
        clearance = {"status": "NOT_RUN"}
        if path_checker is not None:
            modes = ["contact_free" if label == "RETRACTING" else "free"
                     for label in ["RETRACTING"]+labels[:len(measured_q)-1]]
            clearance = path_checker(measured_q, modes)
        measurement.update(
            executed=aborted is None and measurement["steps"] == len(targets), aborted_reason=aborted,
            grasp_lost=bool(aborted == "GRASP_LOST" or self._terminated), end_time_s=self.time_s,
            duration_s=self.time_s-start_time, planned_steps=int(len(targets)),
            home_joint_error_max_rad=float(np.abs(measured_q[-1]-q_home).max()),
            home_joint_error_rad=(measured_q[-1]-q_home).tolist(),
            tool_translation_error_m=tool_t, tool_rotation_error_rad=tool_r,
            edge_translation_error_m=edge_t, edge_rotation_error_rad=edge_r,
            tool_pose_reference="tool body pose measured right after the reset pregrasp settle",
            grasp_slip_max_m=max(slip, default=None), grasp_rotation_max_rad=max(rotation, default=None),
            grasp_slip_final_m=slip[-1] if slip else None, grasp_rotation_final_rad=rotation[-1] if rotation else None,
            unintended_tool_contacts_max=max(contacts, default=0 if not hasattr(self.backend, "grasp_trace") else None),
            max_tool_contact_penetration_m=max(penetration, default=0.),
            max_material_reaction_force_n=max(reaction, default=None),
            label_changes=int(np.count_nonzero(codes_after != codes_before)),
            measured_path_clearance_status=clearance.get("status"), measured_path_clearance=clearance,
            measured_q_final=measured_q[-1].tolist(),
            secondary_arm_q_final=np.asarray(final["q"], float)[1].tolist() if np.asarray(final["q"]).shape[0] > 1 else None)
        return {"measurement": measurement, "plan": summary, "measured_q": measured_q, "labels": labels}

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
    ``scene_config.ConfiguredTaskSpec`` (ADR 0015). It runs on the kinematic
    runtime and (WP-11 Phase D) on Isaac Sim, whose runner recompiles it from its
    recorded scene/goal config and requires the same id; solver, law and K come
    from the scene. Isaac Lab still builds tasks from registered ids and raises
    NotImplementedError rather than silently using another task.
    """
    if task is not None and runtime == "isaac_lab":
        raise NotImplementedError("Isaac Lab builds tasks from registered task ids; a compiled ConfiguredTaskSpec "
                                  "runs on runtime='kinematic' or 'isaac_sim'")
    if runtime == "kinematic":
        yield from _kinematic_session(task, task_id, output_dir, seed, action_mode, observation_profile, backend_options)
        return
    from .recording import manifest, new_output, write_json
    if runtime not in ("isaac_sim", "isaac_lab") or action_mode not in ACTION_MODES+(DELTA_MODE,):
        raise ValueError("Choose a supported live runtime and action mode")
    if task is None:
        make_task(task_id, seed=seed)
    wrapper_options = {key: backend_options.pop(key) for key in WRAPPER_OPTIONS if key in backend_options}
    output = new_output(output_dir)
    wrapper = None
    if runtime == "isaac_sim":
        from .isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app
        mode = "joint" if action_mode == ACTION_MODES[1] else "cartesian"
        if task is not None:
            # Compiled configured task: the config carries its recompilable payload (not the object).
            from .scene_config import ConfiguredTaskSpec, configured_task_payload
            if not isinstance(task, ConfiguredTaskSpec):
                raise TypeError("open_cutting_env(task=...) expects a compiled scene_config.ConfiguredTaskSpec")
            task_id = task.task_id
            backend_options = {"tool_attachment_mode": "physical_grasp", "max_cuts": task.max_cuts, **backend_options,
                               "configured_task": configured_task_payload(task)}
        config = IsaacSimCuttingConfig(task_id=task_id, seed=seed, action_mode=mode,
            **{"camera_mode": "native_single_view", "camera_layout": camera_layout, **backend_options})
        write_json(output/"manifest.json", manifest({**asdict(config),
            "formal_observation": {"profile": observation_profile, "include_global_camera": bool(include_global_camera)}}, "live_isaac_sim_shared_api"))
        app = launch_app(config, output)
        try:
            wrapper = CuttingTaskEnv(LiveIsaacSimCutting(app, config, output), task_id, seed=seed,
                action_mode=action_mode, observation_profile=observation_profile,
                include_global_camera=include_global_camera, task=task, **wrapper_options)
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
