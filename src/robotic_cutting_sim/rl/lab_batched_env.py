"""Workload B runtime: ``IsaacLabBatchedCuttingEnv`` -- N Franka worlds on Isaac Lab Newton / MuJoCo-Warp in one
process (ADR 0022 §2, §4, §6, §7), driven by the ``edge_delta_v1`` action contract of workload A.

Per control step (all device work, no host sync in training mode)::

    _pre_physics_step(a[N,8])  EdgeDeltaActionV1.apply(a, commanded)  -> batched LM IK (n_problems=N)
                               -> hold on IK failure (REJECTED_IK_UNREACHABLE) -> joint targets (+ velocity ff)
                               -> actuators.target_command mask writes
    10 x substep               state-force callback: apply_world_wrenches (dim N) before every 2 ms substep
    _get_dones                 edge kernel -> pad boxes in place -> material.set_measured_poses + advance
                               -> rigid validity (strict: one int32 read) -> dones
    _get_observations          rl_state_obs_v1 policy block

The material is reached only through :class:`MaterialCoupling` (ADR 0021 §7 semantics: material step k reads
the edge measured at the end of rigid interval k; its reaction ``pending_wrench`` is applied in every substep of
interval k+1, from the same buffer). Stage S0 has no batched material on this backend:
``BatchedCuttingMaterial`` refuses ``backend="isaac_lab_newton"`` until the material work package adds it, so
the only built-in coupling is :class:`RobotOnlyCoupling` (no material, zero reaction, explicitly labelled
``robot_only_no_material_coupling``). Anything that needs material state (reward, cut outcome, evaluator)
raises ``UNSUPPORTED_MATERIAL_COUPLING``. Cameras/sensors are UNSUPPORTED (ADR 0022 §10).

Partial reset (ADR 0022 §6) is masked: ``reset_envs(mask)`` writes joint state, targets (and the B2 tool) with
``*_mask`` APIs only, which flag ``_world_reset_mask`` / ``_fk_reset_mask`` for exactly those worlds; the next
``forward()`` runs masked FK and the per-world MJWarp solver reset. No all-world refresh, settle or
``sim.reset()`` runs on a partial reset. ``last_reset_mask_audit`` records the two masks right after the writes
(test T1).
"""
from dataclasses import asdict

import numpy as np
import torch
import warp as wp

from isaaclab.envs import DirectRLEnv

from ..batched import lab_scene
from ..batched.lab_scene import BatchedLabOptions, unsupported
from ..batched.robot_kernels import (RIGID_REASONS, RobotFrames, decode_rigid_reason, resolve_index_tables,
                                     rigid_validity_kernel)
from ..batched.robot_ik import BatchedEdgeIK, ik_rejection_status_code
from .actions import EdgeDeltaActionV1
from .observations import StateObservation

RUNTIME = "rl_gpu_headless"
BACKEND = "isaac_lab_newton"
ENV_VERSION = "isaac_lab_batched_cutting_env_v1"
IK_STREAK_LIMIT = 0  # 0: an IK failure streak never invalidates an env by itself (held, reported per step)


# ----------------------------------------------------------------------------- material coupling interface

class MaterialCoupling:
    """The coupling surface the batched env drives (ADR 0021 §7); ``BatchedCuttingMaterial`` will implement it.

    Required members (device arrays on the env's device, float64 unless stated):

    * ``pending_wrench [N,6]`` -- world wrench at the edge, output of the previous ``advance`` (read in place by
      the wrench kernel during every substep of the next interval; never copied);
    * ``pad_wrench [N,B,6]`` -- per-box world wrenches (rows 1, 2 = left, right finger pads);
    * ``box_centers / box_halves [N*B] wp.vec3`` (float32) -- written in place by the pad-box kernel before
      ``advance`` (``None`` when the coupling has no boxes);
    * ``set_measured_poses(edge [N,7] wp.float64, cut [N] wp.int32)``, ``advance()``, ``reset_masked(mask)``.
    """

    material_coupled = False
    name = "abstract"

    def receipt(self):
        return {"coupling": self.name, "material_coupled": self.material_coupled}


class RobotOnlyCoupling(MaterialCoupling):
    """No material: zero reaction wrench, no boxes. Robot-only stepping for S0; never a material result."""

    name = "robot_only_no_material_coupling"

    def __init__(self, num_envs, device, boxes=3):
        with wp.ScopedDevice(device):
            self.pending_wrench = wp.zeros((num_envs, 6), dtype=wp.float64)
            self.pad_wrench = wp.zeros((num_envs, boxes, 6), dtype=wp.float64)
            self.box_centers = None
            self.box_halves = None
            self.measured_pose = wp.zeros((num_envs, 7), dtype=wp.float64)
            self.cut_input = wp.zeros(num_envs, dtype=wp.int32)
        self.num_envs, self.device, self.boxes = num_envs, device, boxes

    def set_measured_poses(self, measured_pose, cut):
        wp.copy(self.measured_pose, measured_pose)
        wp.copy(self.cut_input, cut)

    def advance(self):
        return None

    def reset_masked(self, mask):
        return None

    def receipt(self):
        return {"coupling": self.name, "material_coupled": False,
                "statement": "no material is simulated; reaction wrench identically zero; no cut outcome"}


# ----------------------------------------------------------------------------- kernels local to the env

@wp.kernel
def _cut_from_torch(digit: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    out[e] = digit[e]


class IsaacLabBatchedCuttingEnv(DirectRLEnv):
    """N worlds of workload B. Construct with :func:`make_batched_env` (inside ``launch_simulation``)."""

    runtime_source = "isaac_lab_newton_batched"

    def __init__(self, cfg, options, task, *, coupling=None):
        self.options = options.validate()
        self.task = task
        self._bound = False
        self._coupling_factory = coupling
        super().__init__(cfg)
        self._bind()
        self._bound = True

    # ------------------------------------------------------------------ binding

    def _setup_scene(self):
        from isaaclab_newton.physics import NewtonManager
        if self.options.cameras or self.options.sensors:
            raise unsupported("UNSUPPORTED_SENSOR", "no cameras in the kit-less batched path")
        NewtonManager.register_state_force_callback(self._state_force)

    def _bind(self):
        from isaaclab_newton.physics import NewtonManager

        from ..finger_geometry import discover_pads
        from ..tool_profile import handle_box_edge, tool_spec_for_task
        o, N = self.options, self.num_envs
        self.model = NewtonManager.get_model()
        self.solver = self.sim.physics_manager._solver
        self.robots = [self.scene[f"franka_{i}"] for i in range(2)]
        for robot in self.robots:
            for name, actuator in robot.cfg.actuators.items():
                if type(actuator).__name__ != "ImplicitActuatorCfg":
                    raise unsupported("UNSUPPORTED_ACTUATOR_OVERRIDE", f"{name}: {type(actuator).__name__}")
        self.spec = tool_spec_for_task(self.task)
        self.wp_device = self.model.device
        self.tables = resolve_index_tables(self.model.body_label, self.model.body_world.numpy(), N, o.tool_mount)
        hand_path = self.model.body_label[int(self.tables["hand"][0])]
        self.pad_geometry = discover_pads(self.sim.stage, hand_path)
        pad_corners = np.asarray([pad["corners_body_m"] for pad in self.pad_geometry], float)
        self.frames = RobotFrames(self.tables, self.spec, self.task.hand_from_edge_xyz_xyzw, o.tool_mount,
                                  pad_corners=pad_corners, handle_box=handle_box_edge(self.task.tool_profile),
                                  device=self.wp_device)
        factory = self._coupling_factory
        self.coupling = factory(N, self.wp_device) if callable(factory) else RobotOnlyCoupling(N, self.wp_device)
        # IK on the Franka 0 prototype of the cloner (as Isaac Lab's Newton IK action)
        source = lab_scene.ENV_TEMPLATE.format(0)+"/Franka_0"
        prototype = NewtonManager._cl_protos.get(source)
        if prototype is None:
            raise RuntimeError(f"No cloner prototype builder for {source}; got {sorted(NewtonManager._cl_protos)}")
        self.ik = BatchedEdgeIK(prototype, N, self.task.hand_from_edge_xyz_xyzw, iterations=o.ik_iterations,
                                position_tolerance_m=o.ik_position_tolerance_m,
                                rotation_tolerance_rad=o.ik_rotation_tolerance_rad, device=self.wp_device,
                                joint_names=list(self.robots[0].joint_names))
        # action, observation, per-env state (Torch views on the same device)
        T = np.repeat(np.asarray(self.task.T_world_workpiece, float)[None], N, 0)
        self.action_spec = EdgeDeltaActionV1(self.task.action_limits(), T, int(self.task.max_cuts), device=self.device)
        self.max_steps = int(round(float(self.task.duration_s)/float(self.task.control_dt_s)))
        self.observation = StateObservation(N, max_steps=self.max_steps, control_dt=float(self.task.control_dt_s),
                                            device=self.device)
        f64 = dict(dtype=torch.float64, device=self.device)
        self.commanded = torch.zeros((N, 7), **f64)
        self.home_commanded = torch.zeros((N, 7), **f64)
        self.previous_action = torch.zeros((N, 8), dtype=torch.float32, device=self.device)
        self.armed = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.action_status = torch.zeros(N, dtype=torch.int32, device=self.device)
        self.ik_streak = wp.zeros(N, dtype=wp.int32, device=self.wp_device)
        self.cut_digit = wp.zeros(N, dtype=wp.int32, device=self.wp_device)
        self.rigid_reason = wp.zeros(N, dtype=wp.int32, device=self.wp_device)
        self.rigid_count = wp.zeros(1, dtype=wp.int32, device=self.wp_device)
        q0, q1 = lab_scene.home_joint_positions(self.task, o.tool_mount, o.grasp)
        self._joint_index = [self._lab_order(robot) for robot in self.robots]
        self.home_targets = [torch.as_tensor(np.tile(q[idx], (N, 1)), dtype=torch.float32, device=self.device)
                             for q, idx in zip((q0, q1), self._joint_index)]
        self.finger_hold = lab_scene.finger_hold_target(self.task, o.tool_mount, o.grasp)
        names = list(self.robots[0].joint_names)
        self._finger_cols = torch.tensor([names.index("panda_finger_joint1"), names.index("panda_finger_joint2")],
                                         device=self.device)
        self.last_targets = self.home_targets[0].clone()
        self.reject_ik = ik_rejection_status_code()
        starts = self.model.joint_q_start.numpy()
        world_of_joint = self.model.joint_world.numpy() if hasattr(self.model, "joint_world") else None
        coord_start = np.zeros(N+1, np.int32)
        if world_of_joint is not None:
            for e in range(N):
                joints = np.flatnonzero(world_of_joint == e)
                coord_start[e] = starts[joints.min()]
            coord_start[N] = self.model.joint_coord_count
        else:
            per = self.model.joint_coord_count//N
            coord_start[:] = np.arange(N+1)*per
        self.coord_start = wp.array(coord_start, dtype=wp.int32, device=self.wp_device)
        self.last_reset_mask_audit = None
        self.step_index = 0

    def _lab_order(self, robot):
        names = list(robot.joint_names)
        canonical = [f"panda_joint{i}" for i in range(1, 8)]+["panda_finger_joint1", "panda_finger_joint2"]
        if sorted(names) != sorted(canonical):
            raise RuntimeError(f"Unexpected Franka joints {names}")
        return np.asarray([canonical.index(n) for n in names], int)

    # ------------------------------------------------------------------ physics callbacks

    def _state_force(self, state):
        if self._bound:
            self.frames.launch_wrench(state.body_q, self.model.body_com, state.body_f, self.physics_dt,
                                      feedback=self.options.feedback, pending_wrench=self.coupling.pending_wrench,
                                      pad_wrench=self.coupling.pad_wrench)

    # ------------------------------------------------------------------ DirectRLEnv hooks

    def _pre_physics_step(self, actions):
        result = self.action_spec.apply(actions.to(self.device), self.commanded)
        joint_pos = wp.from_torch(self.robots[0].data.joint_pos.torch.contiguous())
        self.ik.solve_from(joint_pos, wp.from_torch(result.commanded_pose.contiguous()))
        ik_status = wp.to_torch(self.ik.status)
        rejected_action = result.status >= 2
        ik_failed = (ik_status != 0) & ~rejected_action
        accept = ~rejected_action & ~ik_failed
        self.action_status = torch.where(ik_failed, torch.full_like(result.status, self.reject_ik), result.status)
        self.commanded = torch.where(accept[:, None], result.commanded_pose, self.commanded)
        hold_row = self.action_spec._hold_row.expand_as(result.executed_action)
        self.previous_action = torch.where(ik_failed[:, None], hold_row, result.executed_action)
        self.armed = result.armed & accept
        digit = torch.where(accept, result.cut_digit, torch.full_like(result.cut_digit, -1))
        wp.copy(self.cut_digit, wp.from_torch(digit.to(torch.int32).contiguous()))
        streak = wp.to_torch(self.ik_streak)
        streak.copy_(torch.where(ik_failed, streak+1, torch.zeros_like(streak)))
        solved = wp.to_torch(self.ik.joint_q_out)
        targets = torch.where(accept[:, None], solved, self.last_targets)
        targets[:, self._finger_cols] = self.finger_hold
        if self.options.velocity_feedforward:
            velocity = (targets-self.last_targets)/self.step_dt
            velocity[:, self._finger_cols] = 0.
        else:
            velocity = torch.zeros_like(targets)
        self.last_targets = targets
        franka0, franka1 = self.robots
        franka0.actuators.target_command.set_position_mask(value=targets.contiguous())
        franka0.actuators.target_command.set_velocity_mask(value=velocity.contiguous())
        franka1.actuators.target_command.set_position_mask(value=self.home_targets[1])
        franka1.actuators.target_command.set_velocity_mask(value=torch.zeros_like(self.home_targets[1]))
        self.frames.applied_impulse.zero_()

    def _apply_action(self):
        pass  # Lab's scene writes the persistent actuator targets before every substep.

    def _get_dones(self):
        from isaaclab_newton.physics import NewtonManager
        state = NewtonManager.get_state_0()
        self.frames.launch_edge(state.body_q)
        if self.coupling.box_centers is not None:
            self.frames.launch_pad_boxes(state.body_q, self.coupling.box_centers, self.coupling.box_halves)
        else:
            self.frames.launch_pad_boxes(state.body_q)
        self.coupling.set_measured_poses(self.frames.measured_edge, self.cut_digit)
        self.coupling.advance()
        self.rigid_count.zero_()
        data = self.solver.mjw_data
        wp.launch(rigid_validity_kernel, self.num_envs, device=self.wp_device,
                  inputs=[state.joint_q, self.coord_start, state.body_q, self.frames.tool, data.nacon,
                          int(data.naconmax), self.ik_streak, IK_STREAK_LIMIT, self.frames.edge_status,
                          self.rigid_reason, self.rigid_count])
        if self.options.strict:
            self.check_valid()
        self.step_index += 1
        invalid = wp.to_torch(self.rigid_reason) != 0
        truncated = self.episode_length_buf >= self.max_steps
        return invalid, truncated

    def check_valid(self):
        """Strict mode: one int32 read; raises with env ids and reasons."""
        count = int(self.rigid_count.numpy()[0])
        if count:
            reasons = self.rigid_reason.numpy()
            ids = np.flatnonzero(reasons)
            raise FloatingPointError(f"Invalid rigid worlds {ids.tolist()}: "
                                     f"{ {int(e): decode_rigid_reason(reasons[e]) for e in ids} }")
        return 0

    def _get_rewards(self):
        # Rewards need the material (trim_bond_progress_packed_v1); robot-only stepping returns no reward.
        self.extras["reward_available"] = self.coupling.material_coupled
        return torch.zeros(self.num_envs, device=self.device)

    def compute_task_reward(self):
        raise unsupported("UNSUPPORTED_MATERIAL_COUPLING", "task reward needs the batched material coupling")

    def evaluate(self):
        raise unsupported("UNSUPPORTED_MATERIAL_COUPLING", "cut outcome/evaluator needs the batched material")

    def _get_observations(self):
        return self.observation.compute(commanded_pose=self.commanded, step_count=self.episode_length_buf,
                                        previous_action=self.previous_action, armed=self.armed)

    def _reset_envs_from_buffer(self):
        return None  # explicit reset_envs(mask) only; no nonzero() host sync, no same-step autoreset

    # ------------------------------------------------------------------ reset

    def _reset_idx(self, env_ids):
        """Full reset path of ``DirectRLEnv.reset()`` (build time): masked writes for the given ids."""
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[torch.as_tensor(env_ids, device=self.device, dtype=torch.long)] = True
        if not self._bound:
            return
        self.reset_envs(mask, _initial=True)

    def reset(self, seed=None, options=None):
        obs, extras = super().reset(seed=seed, options=options)
        from isaaclab_newton.physics import NewtonManager
        self.frames.launch_edge(NewtonManager.get_state_0().body_q)
        self.home_commanded.copy_(wp.to_torch(self.frames.nominal_edge))
        self.commanded.copy_(self.home_commanded)
        return self._get_observations(), extras

    def reset_envs(self, mask, *, _initial=False):
        """Masked per-world reset (ADR 0022 §6.1). ``mask``: ``torch.bool [N]`` on the env device."""
        from isaaclab_newton.physics import NewtonManager
        if self.options.tool_mount == "physical_grasp_v1":
            raise unsupported("UNSUPPORTED_GRASP_RESETTLE_ON_PARTIAL_RESET",
                              "B2 snapshot restore is stage S2 (grasp_batch); not wired in S0")
        mask = mask.to(device=self.device, dtype=torch.bool).contiguous()
        wmask = wp.from_torch(mask, dtype=wp.bool)
        for robot, home in zip(self.robots, self.home_targets):
            robot.write_joint_state_to_sim_mask(position=home, velocity=torch.zeros_like(home), env_mask=wmask)
            robot.actuators.target_command.set_position_mask(value=home, env_mask=wmask)
            robot.actuators.target_command.set_velocity_mask(value=torch.zeros_like(home), env_mask=wmask)
        world = NewtonManager._world_reset_mask
        fk = NewtonManager._fk_reset_mask
        self.last_reset_mask_audit = {"world_reset_mask": None if world is None else wp.clone(world),
                                      "fk_reset_mask": None if fk is None else wp.clone(fk), "mask": mask.clone()}
        self.frames.reset_masked(wmask)
        self.coupling.reset_masked(wmask)
        self.last_targets = torch.where(mask[:, None], self.home_targets[0], self.last_targets)
        self.commanded = torch.where(mask[:, None], self.home_commanded, self.commanded)
        self.previous_action = torch.where(mask[:, None], torch.zeros_like(self.previous_action), self.previous_action)
        self.armed = self.armed & ~mask
        self.episode_length_buf = torch.where(mask, torch.zeros_like(self.episode_length_buf), self.episode_length_buf)
        for values in (self.ik_streak, self.rigid_reason, self.cut_digit):
            t = wp.to_torch(values)
            t.copy_(torch.where(mask, torch.zeros_like(t), t))

    # ------------------------------------------------------------------ receipts

    def receipt(self):
        model = self.model
        world = model.body_world.numpy()
        return {"version": ENV_VERSION, "runtime": RUNTIME, "backend": BACKEND, "num_envs": self.num_envs,
                "device": str(self.device), "options": {k: v for k, v in asdict(self.options).items() if k != "grasp"},
                "grasp": self.options.grasp.to_dict(), "scene": lab_scene.scene_receipt(self.options, self.task),
                "coupling": self.coupling.receipt(), "ik": self.ik.receipt(), "action": self.action_spec.spec(),
                "observation": self.observation.metadata(),
                "bodies_per_world": int(np.sum(world == 0)), "global_bodies": int(np.sum(world < 0)),
                "dofs_per_world": int(model.joint_dof_count//self.num_envs),
                "index_tables": {k: v.tolist() for k, v in self.tables.items()},
                "rigid_reasons": {str(k): v for k, v in RIGID_REASONS.items()},
                "material_results": "NONE (robot-only)" if not self.coupling.material_coupled
                else "SIMULATOR_ONLY_UNCALIBRATED"}


def make_batched_env(options, task, *, coupling=None):
    """Build the cfg and the env; call inside ``isaaclab.app.sim_launcher.launch_simulation(cfg, ...)``."""
    cfg = lab_scene.make_batched_lab_cfg(options, task)
    return IsaacLabBatchedCuttingEnv(cfg, options, task, coupling=coupling)


def open_vector_env(num_envs, *, backend="isaac_lab_newton", task=None, tool_mount="rigid_mount_v1", device="cuda:0",
                    material=None, observation_profile="state", **kwargs):
    """``open_vector_env(backend="isaac_lab_newton", tool_mount=...)`` wiring (ADR 0022 §9).

    Returns ``(cfg, factory)``: the caller enters ``launch_simulation(cfg, {"headless": True, "visualizer": []})``
    and calls ``factory()`` inside it (Isaac Lab owns the process-wide simulation context)."""
    if backend != BACKEND:
        raise unsupported("UNSUPPORTED_BACKEND", f"{backend!r}; workload B is {BACKEND!r}")
    if observation_profile != "state":
        raise unsupported("UNSUPPORTED_SENSOR", f"observation_profile={observation_profile!r}; state only (S0)")
    if task is None:
        raise ValueError("a compiled tlmpm_v7_2 task (e.g. seeded TRIM) is required")
    options = BatchedLabOptions(num_envs=int(num_envs), tool_mount=tool_mount, device=device, material=material,
                                **kwargs).validate()
    cfg = lab_scene.make_batched_lab_cfg(options, task)
    return cfg, (lambda coupling=None: IsaacLabBatchedCuttingEnv(cfg, options, task, coupling=coupling))


__all__ = ["IsaacLabBatchedCuttingEnv", "MaterialCoupling", "RobotOnlyCoupling", "make_batched_env",
           "open_vector_env", "ENV_VERSION", "RUNTIME", "BACKEND"]
