"""Live Isaac Lab 3.0 DirectRLEnv / InteractiveScene cutting task.

The name of Lab's base class does not imply policy training. Actions are fixed
reference blade-edge poses. Lab owns both articulations, dynamics and clock;
the external retained TLMPM owns the material. Cameras in this kit-less adapter
are explicitly ideal Warp RGB-D of the CURRENT full USD scene, not native D455.
Full Kit and native sensors have separate runners and acceptance results.
"""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg, NewtonManager, MJWarpSolverCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg, MujocoJointCfg, NewtonArticulationCfg

from .coupled_robot import CouplingConfig, reference_pose
from .live_material import LiveCuttingMaterial
from .recording import ROOT
from .robot import compose_tool_pose, wedge_mesh
from .scene_frames import WorkpieceTarget, compose_pose, invert_transform, matrix_pose, pose_matrix
from .workcell import ASSETS, PARK_JOINTS


@wp.kernel
def _edge_wrench(body_q: wp.array(dtype=wp.transform), body_com: wp.array(dtype=wp.vec3),
                 wrench: wp.array(dtype=wp.spatial_vector), hand: int,
                 body_f: wp.array(dtype=wp.spatial_vector)):
    pose = body_q[hand]
    edge = wp.transform_point(pose, wp.vec3(0., 0., .18))
    center = wp.transform_point(pose, body_com[hand])
    w = wrench[0]
    force = wp.vec3(w[0], w[1], w[2])
    torque = wp.vec3(w[3], w[4], w[5])+wp.cross(edge-center, force)
    body_f[hand] = body_f[hand]+wp.spatial_vector(force, torque)


@wp.kernel
def _native_contact_audit(count: wp.array(dtype=int), distance: wp.array(dtype=float),
                          geom: wp.array(dtype=wp.vec2i), world: wp.array(dtype=int), shape_map: wp.array2d(dtype=int),
                          knife: int, audit: wp.array(dtype=float)):
    i = wp.tid()
    if i < count[0]:
        pair = geom[i]
        if pair[0] >= 0 and pair[1] >= 0:
            a, b = shape_map[world[i], pair[0]], shape_map[world[i], pair[1]]
            penetration = wp.max(0., -distance[i])
            wp.atomic_max(audit, 3, float(count[0]))
            if a != knife and b != knife:
                wp.atomic_max(audit, 0, penetration)
                wp.atomic_add(audit, 2, 1.)
            else:
                wp.atomic_max(audit, 1, penetration)


def _spawn_franka(prim_path, cfg, translation=None, orientation=None, *, task=None, **kwargs):
    """Spawn through Lab, attaching the physical wedge before Newton imports USD."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    if prim_path.endswith("Franka_0"):
        stage = prim.GetStage()
        hand_prim = next(p for p in Usd.PrimRange(prim) if p.GetName() == "panda_hand"
                         and p.HasAPI(UsdPhysics.RigidBodyAPI))
        mesh = UsdGeom.Mesh.Define(stage, str(hand_prim.GetPath())+"/Knife")
        dimensions = (task.blade_length_m, task.blade_height_m, task.blade_half_width_m) if task else (.012, .004, .025)
        mount = task.hand_from_edge_xyz_xyzw if task else (0, 0, .18, 1, 0, 0, 0)
        vertices, faces = wedge_mesh(*dimensions)
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(.38, .65, .85)])
        xf = UsdGeom.Xformable(mesh)
        xf.AddTranslateOp().Set(Gf.Vec3d(*mount[:3]))
        xf.AddOrientOp().Set(Gf.Quatf(float(mount[6]), Gf.Vec3f(*mount[3:6])))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return prim


@dataclass(frozen=True)
class LabCuttingConfig:
    device: str = "cuda:0"
    material_device: str = "cuda:0"
    seed: int = 42
    variant: str = "D0"
    task_id: str | None = None
    task_variant: str = "fixed"
    particle_spacing_m: float | None = None
    action_mode: str = "cartesian_edge_pose_world"
    surface_profile: str | None = None
    dx: float = .004
    feedback: bool = True
    width: int = 640
    height: int = 480
    global_width: int = 1280
    global_height: int = 960
    velocity_feedforward: bool | None = None
    experiment: str = "cut"
    camera: bool = True
    camera_layout: str = "legacy_fixed_d455"
    include_global_camera: bool = False
    static_scene: str = str(ROOT/"artifacts/research_handoff_001/workcell/workcell.usda")

    def __post_init__(self):
        if self.camera_layout not in ("legacy_fixed_d455", "wrist_d405_global_d455"):
            raise ValueError("Unsupported camera layout")
        if self.task_id not in (None, "CUT-01", "TRIM-01"):
            raise ValueError("Unsupported live cutting task")
        if self.particle_spacing_m is not None and self.task_id is None:
            raise ValueError("Particle-spacing overrides require a CUT/TRIM task; legacy recipes remain fixed")
        if self.action_mode not in ("cartesian_edge_pose_world", "joint_position_rad"):
            raise ValueError("Unsupported action mode")
        if self.velocity_feedforward is None:
            object.__setattr__(self, "velocity_feedforward", self.task_id is not None)
        if self.surface_profile is None:
            object.__setattr__(self, "surface_profile", "current_volume_v1" if self.task_id else "legacy")


def make_lab_cfg(options):
    """Use the actual GA workspace APIs and explicit solver settings."""
    asset = ROOT/"artifacts/workcell_asset_cache"/ASSETS["franka"]
    if not asset.is_file():
        raise FileNotFoundError(asset)
    from .tasks import make_task
    task = make_task(options.task_id, seed=options.seed, variant=options.task_variant,
                     particle_spacing_m=options.particle_spacing_m) if options.task_id else None

    def robot(index):
        q = dict(zip([f"panda_joint{i}" for i in range(1, 8)], PARK_JOINTS))
        q["panda_finger_joint.*"] = .04
        if index == 1:
            q["panda_joint1"] = -np.pi/2
            if options.camera_layout == "wrist_d405_global_d455":
                from .camera_rig import SECONDARY_OBSERVATION_JOINTS
                q.update(dict(zip([f"panda_joint{i}" for i in range(1, 8)], SECONDARY_OBSERVATION_JOINTS)))
        return ArticulationCfg(prim_path=f"/World/Franka_{index}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(asset), func=partial(_spawn_franka, task=task) if task else _spawn_franka,
                variants={"Physics": "mujoco", "Colliders": "convex_hulls"},
                articulation_props=[NewtonArticulationCfg(self_collision_enabled=True)],
                rigid_props={"/.*": [MujocoRigidBodyCfg(gravcomp=1.)]},
                joint_drive_props={"/.*": [MujocoJointCfg(actuatorgravcomp=True)]}),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-.46, .12, -.01) if index == 0 else (.46, .12, -.01),
                rot=(0., 0., 0., 1.) if index == 0 else (0., 0., 1., 0.),
                joint_pos=q),
            actuators={
                "shoulder": ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-4]"],
                    stiffness=1000., damping=80., joint_effort_limit=87., armature=.2, viscous_friction=0.),
                "forearm": ImplicitActuatorCfg(joint_names_expr=["panda_joint[5-7]"],
                    stiffness=600., damping=40., joint_effort_limit=12., armature=.07, viscous_friction=0.),
                "fingers": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"],
                    stiffness=350., damping=10., joint_effort_limit=20., armature=.1, viscous_friction=0.)})

    def fixed_box(name, center, size, color, rotation=(0., 0., 0., 1.)):
        return AssetBaseCfg(prim_path="/World/"+name, init_state=AssetBaseCfg.InitialStateCfg(pos=center, rot=rotation),
            spawn=sim_utils.CuboidCfg(size=size,
                collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)))

    @configclass
    class CuttingSceneCfg(InteractiveSceneCfg):
        table = fixed_box("Table", (0., 0., -.035), (1.8, 1.2, .05), (.28, .31, .35))
        tray_floor = fixed_box("TrayFloor", (0., 0., -.005), (.136, .096, .01), (.35, .38, .42))
        tray_left = fixed_box("TrayLeft", (-.064, 0., .0075), (.008, .088, .015), (.35, .38, .42))
        tray_right = fixed_box("TrayRight", (.064, 0., .0075), (.008, .088, .015), (.35, .38, .42))
        tray_front = fixed_box("TrayFront", (0., -.044, .0075), (.12, .008, .015), (.35, .38, .42))
        tray_back = fixed_box("TrayBack", (0., .044, .0075), (.12, .008, .015), (.35, .38, .42))
        franka_0 = robot(0)
        franka_1 = robot(1)

    scene_cfg = CuttingSceneCfg(num_envs=1, env_spacing=2.)
    if task:
        for name in ("tray_floor", "tray_left", "tray_right", "tray_front", "tray_back"):
            setattr(scene_cfg, name, None)
        scene_cfg.table = fixed_box("Table", (0., 0., task.table_top_world_z_m-.025),
                                    (1.8, 1.2, .05), (.28, .31, .35))
        for index, box in enumerate(task.fixture_boxes_local):
            center = tuple(task.points_to_world(box.center_m))
            setattr(scene_cfg, f"task_fixture_{index}", fixed_box(f"TaskFixture_{index}", center,
                tuple(2*np.asarray(box.half_extents_m)), (.22, .27, .31), tuple(task.world_from_workpiece_xyzw[3:])))

    observation = {"q": [2, 7], "qdot": [2, 7], "tool_pose": 7}
    if options.camera:
        camera_ids = ("wrist_d405_0", "wrist_d405_1") if options.camera_layout == "wrist_d405_global_d455" else ("d455_0", "d455_1")
        if options.include_global_camera:
            camera_ids += ("global_d455" if options.camera_layout == "wrist_d405_global_d455" else "global",)
        if options.camera_layout == "wrist_d405_global_d455":
            from .camera_rig import make_rig
            camera_specs = make_rig(options.task_id or "TRIM-01").specs
        for name in camera_ids:
            height, width = ((camera_specs[name]["height"], camera_specs[name]["width"])
                if options.camera_layout == "wrist_d405_global_d455" else (options.height, options.width))
            observation[name+"_rgb"] = [height, width, 3]
            observation[name+"_depth_m"] = [height, width]
    cfg = DirectRLEnvCfg(seed=options.seed, decimation=10, episode_length_s=8.,
        action_space=7, observation_space=observation, ui_window_class_type=None,
        scene=scene_cfg,
        sim=sim_utils.SimulationCfg(device=options.device, dt=.002, render_interval=10,
            physics=NewtonCfg(use_cuda_graph=False, load_visual_shapes=False,
                default_shape_cfg=NewtonShapeCfg(gap=.001),
                solver_cfg=MJWarpSolverCfg(use_mujoco_cpu=options.device == "cpu", iterations=50,
                    nconmax=256, njmax=1024, integrator="implicitfast", cone="elliptic"))))
    return cfg


class IsaacLabCuttingEnv(DirectRLEnv):
    """A real Lab task; no standalone articulated simulator is constructed."""
    runtime_source = "live_isaac_lab"

    def __init__(self, cfg, options):
        self.options = options
        self.target = WorkpieceTarget()
        from .tasks import make_task
        self.task = make_task(options.task_id, seed=options.seed, variant=options.task_variant,
            particle_spacing_m=options.particle_spacing_m) if options.task_id else None
        self.mount = np.array(self.task.hand_from_edge_xyz_xyzw if self.task else (0, 0, .18, 1, 0, 0, 0), float)
        self.material = LiveCuttingMaterial(CouplingConfig(dx=options.dx,
            device=options.material_device, feedback=options.feedback, variant=options.variant), self.target, task=self.task)
        self._bound = False
        self._capture_time = None
        self.time_s = 0.
        self.last_diagnostics = {}
        super().__init__(cfg)
        self.model = NewtonManager.get_model()
        self.robots = [self.scene[f"franka_{i}"] for i in range(2)]
        self.hand = next(i for i, label in enumerate(self.model.body_label)
                         if "Franka_0/" in label and label.endswith("/panda_hand"))
        self.hands = [next(i for i, label in enumerate(self.model.body_label)
                          if f"Franka_{arm}/" in label and label.endswith("/panda_hand")) for arm in range(2)]
        self.knife_shape = next(i for i, label in enumerate(self.model.shape_label) if label.endswith("/Knife"))
        if int(self.model.shape_body.numpy()[self.knife_shape]) != self.hand:
            raise RuntimeError("Physical wedge collider is not attached to the measured Panda hand")
        self._bind_ik()
        self._force_buffer = wp.zeros(1, dtype=wp.spatial_vector, device=self.model.device)
        self._contact_audit = wp.zeros(4, dtype=float, device=self.model.device)
        # Read-only debug access to the pinned native solver's signed contact
        # distances; stepping still belongs entirely to Lab's PhysicsManager.
        self._native_solver = self.sim.physics_manager._solver
        self._bind_visuals()
        self._bound = True

    def _setup_scene(self):
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        stage = self.sim.stage
        source = Usd.Stage.Open(self.options.static_scene)
        if source is None:
            raise FileNotFoundError(self.options.static_scene)
        # Reuse only static workcell objects. Neither robots nor material replay
        # are imported from the preview; both are owned by this live task.
        selected = ["D455_0", "D455_1", "GlobalCamera", "EnvironmentLight"]
        if self.options.camera_layout == "wrist_d405_global_d455":
            selected = ["EnvironmentLight"]
        for name in selected:
            path = "/World/"+name
            if not Sdf.CopySpec(source.GetRootLayer(), path, stage.GetRootLayer(), path):
                raise RuntimeError(f"Static object copy failed: {path}")
            if name.startswith("Tray") or name == "Table":
                UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        NewtonManager.register_state_force_callback(self._native_force)

    def _native_force(self, state):
        if self._bound:
            self._sample_native_contacts()
            wp.launch(_edge_wrench, dim=1, inputs=[state.body_q, self.model.body_com,
                self._force_buffer, self.hand, state.body_f], device=self.model.device)

    def _sample_native_contacts(self):
        data = self._native_solver.mjw_data
        wp.launch(_native_contact_audit, dim=data.naconmax,
            inputs=[data.nacon, data.contact.dist, data.contact.geom, data.contact.worldid,
                self._native_solver.mjc_geom_to_newton_shape, self.knife_shape, self._contact_audit],
            device=self.model.device)

    def _bind_ik(self):
        import newton.ik as ik
        self._ik_state = self.model.state()
        self._ik_q = wp.clone(self.model.joint_q).reshape((1, self.model.joint_coord_count))
        self._position = ik.IKObjectivePosition(link_index=self.hand, link_offset=wp.vec3(*self.mount[:3]),
            target_positions=wp.array([wp.vec3(0, 0, .1)], dtype=wp.vec3, device=self.model.device))
        self._rotation = ik.IKObjectiveRotation(link_index=self.hand,
            link_offset_rotation=wp.quat(*self.mount[3:]),
            target_rotations=wp.array([wp.vec4(0, 0, 0, 1)], dtype=wp.vec4, device=self.model.device))
        limits = ik.IKObjectiveJointLimit(self.model.joint_limit_lower, self.model.joint_limit_upper, weight=10.)
        self._ik = ik.IKSolver(model=self.model, n_problems=1,
            objectives=[self._position, self._rotation, limits], lambda_initial=.01,
            jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self._joint_coords = []
        starts = self.model.joint_q_start.numpy()
        for arm, robot in enumerate(self.robots):
            coords = []
            for name in robot.joint_names:
                joint = next(i for i, label in enumerate(self.model.joint_label)
                    if f"Franka_{arm}/" in label and label.endswith("/"+name))
                coords.append(starts[joint])
            self._joint_coords.append(np.array(coords, dtype=int))
        self._park_q = self.model.joint_q.numpy().copy()
        for arm, robot in enumerate(self.robots):
            self._park_q[self._joint_coords[arm]] = robot.data.default_joint_pos.torch.detach().cpu().numpy()[0]

    def _solve_ik(self, desired):
        import newton
        state = NewtonManager.get_state_0()
        self._ik_q.assign(state.joint_q.numpy().reshape(1, -1))
        self._position.set_target_position(0, wp.vec3(*desired[:3]))
        self._rotation.set_target_rotation(0, wp.vec4(*desired[3:]))
        self._ik.step(self._ik_q, self._ik_q, iterations=48)
        q = np.clip(self._ik_q.numpy()[0], self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy())
        self._ik_state.joint_q.assign(q)
        newton.eval_fk(self.model, self._ik_state.joint_q, self.model.joint_qd, self._ik_state)
        edge = compose_tool_pose(self._ik_state.body_q.numpy()[self.hand], self.mount[:3], self.mount[3:])
        error = float(np.linalg.norm(edge[:3]-desired[:3]))
        if error > .001:
            raise RuntimeError(f"Active Lab model IK position error {error}m")
        return q, error

    def measured(self):
        import newton
        state = NewtonManager.get_state_0()
        # Publish final-integration FK, not MuJoCo's pre-integration cached xpos.
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        body = state.body_q.numpy().copy()
        q, v = state.joint_q.numpy(), state.joint_qd.numpy()
        hand = body[self.hand].astype(float)
        return {"q": np.array([q[indices][:7] for indices in self._joint_coords]),
            "qdot": np.array([v[indices][:7] for indices in self._joint_coords]),
            "hand_pose": hand, "hand_poses": body[self.hands].astype(float), "body_pose": body,
            "tool_pose": compose_tool_pose(hand, self.mount[:3], self.mount[3:]), "timestamp_s": self.time_s}

    def reference_pose(self, time_s):
        return (self.task.reference_edge_pose_world(time_s) if self.task else
            self.target.poses_to_world(reference_pose(time_s, self.target.edge_height_local_m)))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not self._bound:
            return
        self.time_s = 0.
        self._runtime_time_origin = self.sim.physics_manager.get_simulation_time()
        self._force_buffer.zero_()
        self._contact_audit.zero_()
        # Lab initializes reset defaults, but a DirectRLEnv subclass must write
        # them. Seed IK from the nonsingular Panda park configuration.
        for arm, robot in enumerate(self.robots):
            values = self._park_q[self._joint_coords[arm]]
            tensor = torch.tensor(values[None], device=self.device, dtype=torch.float32)
            robot.write_joint_position_to_sim_index(position=tensor)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(tensor))
        self.scene.write_data_to_sim()
        self.sim.forward()
        desired = self.reference_pose(0.)
        q, _ = self._solve_ik(desired)
        for arm, robot in enumerate(self.robots):
            values = q[self._joint_coords[arm]] if arm == 0 else self._park_q[self._joint_coords[arm]]
            tensor = torch.tensor(values[None], device=self.device, dtype=torch.float32)
            robot.write_joint_position_to_sim_index(position=tensor)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(tensor))
            robot.actuators.target_command.set_position_index(value=tensor)
            robot.actuators.target_command.set_velocity_index(value=torch.zeros_like(tensor))
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(0.)
        # MuJoCo reset clears warm starts but deliberately defers native qpos
        # synchronization until the next step. Refresh position/contact caches
        # without integrating so contact audits cannot read the old park pose.
        import mujoco_warp
        self._native_solver._update_mjc_data(self._native_solver.mjw_data, self.model, NewtonManager.get_state_0())
        mujoco_warp.fwd_position(self._native_solver.mjw_model, self._native_solver.mjw_data)
        self.setup_receipt = self.material.reset(self.measured()["tool_pose"], self.cfg.seed)
        self.setup_receipt["native_contact_reset_refresh"] = "native coordinate sync + fwd_position; zero integration"
        self._last_joint_targets = [q[self._joint_coords[0]].copy(), self._park_q[self._joint_coords[1]].copy()]
        self.setup_receipt["velocity_feedforward"] = self.options.velocity_feedforward
        self.setup_receipt["experiment"] = self.options.experiment
        self.setup_receipt["velocity_target_source"] = "commanded IK joint finite difference" if self.options.velocity_feedforward else "zero"
        from .usd_surface_bridge import DynamicSurfaceBridge
        self._bridge = DynamicSurfaceBridge(self.sim.stage)
        self._capture_time = None
        self.camera_pose_trace = []
        self.last_diagnostics = {}
        self._step_cut_enabled, self._step_cut_index = True, None

    def _pre_physics_step(self, actions):
        if self.time_s >= 8.-1e-9:
            raise RuntimeError("Episode ended; call reset")
        self._previous = self.measured()
        values = actions[0].detach().cpu().numpy().astype(float)
        if self._step_action_mode == "joint_position_rad":
            import newton
            q = NewtonManager.get_state_0().joint_q.numpy().copy()
            coordinates = self._joint_coords[0][:7]
            if (not np.isfinite(values).all() or np.any(values < self.model.joint_limit_lower.numpy()[coordinates])
                    or np.any(values > self.model.joint_limit_upper.numpy()[coordinates])):
                raise ValueError("Joint command exceeds the actual articulation limits")
            q[coordinates] = values
            self._ik_state.joint_q.assign(q)
            newton.eval_fk(self.model, self._ik_state.joint_q, self.model.joint_qd, self._ik_state)
            self._desired = compose_tool_pose(self._ik_state.body_q.numpy()[self.hand], self.mount[:3], self.mount[3:])
            self._ik_error = 0.
        else:
            self._desired = values
            q, self._ik_error = self._solve_ik(self._desired)
        for arm, robot in enumerate(self.robots):
            values = q[self._joint_coords[arm]] if arm == 0 else self._park_q[self._joint_coords[arm]]
            robot.actuators.target_command.set_position_index(
                value=torch.tensor(values[None], device=self.device, dtype=torch.float32))
            velocity = (values-self._last_joint_targets[arm])/self.step_dt if self.options.velocity_feedforward else np.zeros_like(values)
            robot.actuators.target_command.set_velocity_index(
                value=torch.tensor(velocity[None], device=self.device, dtype=torch.float32))
            self._last_joint_targets[arm] = values.copy()
        applied = self.material.pending_wrench if self.options.feedback else np.zeros(6)
        self._force_buffer.assign(np.asarray(applied, np.float32).reshape(1, 6))
        self._contact_audit.zero_()

    def _apply_action(self):
        pass  # Lab's scene writes the persistent actuator targets before its own dynamics step.

    def _get_rewards(self):
        self.time_s = float(self.episode_length_buf[0])*self.step_dt
        measured = self.measured()
        diagnostics = self.material.advance(measured["tool_pose"], self.time_s,
            cutting=self.options.experiment == "cut" and self._step_cut_enabled, cut_index=self._step_cut_index)
        self._sample_native_contacts()
        contact = self._contact_audit.numpy()
        delta_rotation = (Rotation.from_quat(measured["tool_pose"][3:])
            * Rotation.from_quat(self._previous["tool_pose"][3:]).inv()).as_rotvec()
        applied = diagnostics["applied_previous_material_wrench_world"]
        self.last_diagnostics = {**diagnostics, "desired_tool_pose": self._desired.copy(),
            "ik_position_error_m": self._ik_error,
            "runtime_timestamp_s": self.sim.physics_manager.get_simulation_time()-self._runtime_time_origin,
            "native_runtime_timestamp_s": self.sim.physics_manager.get_simulation_time(),
            "native_rigid_contact_audit": {"source": "actual MJWarp nacon/contact.dist/geom; pre-substeps plus final",
                "max_non_tool_penetration_m": float(contact[0]), "max_tool_rigid_penetration_m": float(contact[1]),
                "observed_non_tool_contact_slots": int(contact[2]), "max_simultaneous_contacts": int(contact[3])},
            "native_float32_wrench_payload_world": self._force_buffer.numpy()[0].astype(float),
            "applied_tool_wrench_work_j": float(applied[:3]@(measured["tool_pose"][:3]-self._previous["tool_pose"][:3])
                + applied[3:]@delta_rotation)}
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        # Preserve terminal material for inspection; explicit reset, never implicit same-step reset.
        return torch.zeros(1, dtype=torch.bool, device=self.device), torch.zeros(1, dtype=torch.bool, device=self.device)

    def _bind_visuals(self):
        from pxr import Gf, UsdGeom
        from .cutting_surface import ComponentSurface
        from .usd_surface_bridge import usd_camera_calibration
        self._body_ops = []
        for i, path in enumerate(self.model.body_label):
            prim = self.sim.stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Live Lab body has no USD prim: {path}")
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.SetResetXformStack(True)
            self._body_ops.append((i, xf.AddTransformOp()))
        self._surface = ComponentSurface(self.material.spacing, self.options.material_device,
            voxel_size=self.material.spacing/2 if self.options.surface_profile == "current_volume_v1" else min(self.material.spacing, .0015),
            profile=self.options.surface_profile)
        if self.options.camera_layout == "wrist_d405_global_d455":
            self.camera_rig = None
            self.calibration = {}
            return
        paths = {"d455_0": "/World/D455_0/RSD455/Camera_Pseudo_Depth",
                 "d455_1": "/World/D455_1/RSD455/Camera_Pseudo_Depth", "global": "/World/GlobalCamera"}
        self.calibration = {name: usd_camera_calibration(self.sim.stage, path,
            self.options.width if name != "global" else self.options.global_width,
            self.options.height if name != "global" else self.options.global_height,
            near_m=.52 if name != "global" else .01,
            far_m=6. if name != "global" else 20.) for name, path in paths.items()}

    def _get_observations(self):
        from pxr import Gf
        from .usd_surface_bridge import stage_triangles
        from .camera import capture
        measured = self.measured()
        policy = {key: torch.as_tensor(measured[key][None], device=self.device) for key in ("q", "qdot", "tool_pose")}
        if self._capture_time != self.time_s:
            vertices, faces, self.surface_metadata = self._surface.extract(self.material.bed)
            if self.task:
                vertices = self.task.points_to_world(vertices).astype(np.float32)
            self.surface = (vertices, faces)
            self._bridge.update(vertices, faces, self.time_s)
            self._bridge.mesh.CreateDisplayColorAttr([Gf.Vec3f(.72, .32, .13)])
            for i, op in self._body_ops:
                op.Set(Gf.Matrix4d(pose_matrix(measured["body_pose"][i]).T.tolist()))
            if self.options.camera and self.options.camera_layout == "wrist_d405_global_d455":
                self._update_wrist_cameras(measured)
            self.frames = {}
            if self.options.camera:
                points, triangles, colors, _ = stage_triangles(self.sim.stage)
                device = self.options.material_device
                with wp.ScopedDevice(device):
                    points = wp.array(points, dtype=wp.vec3)
                    indices = wp.array(triangles.ravel(), dtype=int)
                    colors = wp.array(colors, dtype=wp.vec3)
                    for name, calibration in self.calibration.items():
                        rgb, depth = capture(points, indices, calibration, device, colors)
                        self.frames[name] = {"rgb": rgb, "depth_m": depth, "valid_mask": depth > 0,
                            "timestamp_s": self.time_s, "surface_timestamp_s": self.time_s,
                            "source": "live_isaac_lab_ideal_warp_rgbd",
                            "sensor_model": calibration["sensor_model"]}
            self._capture_time = self.time_s
        if self.options.camera:
            names = ("wrist_d405_0", "wrist_d405_1") if self.options.camera_layout == "wrist_d405_global_d455" else ("d455_0", "d455_1")
            if self.options.include_global_camera:
                names += ("global_d455" if self.options.camera_layout == "wrist_d405_global_d455" else "global",)
            for name in names:
                for key in ("rgb", "depth_m"):
                    policy[name+"_"+key] = torch.as_tensor(self.frames[name][key][None], device=self.device)
        return {"policy": policy}

    def _update_wrist_cameras(self, measured):
        from .camera_rig import make_rig
        from .usd_surface_bridge import usd_camera_calibration
        if self.camera_rig is None:
            self.camera_rig = make_rig(self.task.task_id if self.task else "TRIM-01",
                initial_hand_poses_world_xyzw=measured["hand_poses"])
            self.camera_rig_paths = self.camera_rig.author_usd(self.sim.stage,
                [self.model.body_label[index] for index in self.hands],
                global_d455_asset_path=str(ROOT/"artifacts/workcell_asset_cache"/ASSETS["d455"]))
        calibration = self.camera_rig.calibrations(measured["hand_poses"])
        row = {"timestamp_s": self.time_s, "T_world_hand": [pose_matrix(p).tolist() for p in measured["hand_poses"]],
               "cameras": {}}
        for name, spec in calibration.items():
            actual = usd_camera_calibration(self.sim.stage, self.camera_rig_paths[name]["depth_camera_path"],
                spec["width"], spec["height"], near_m=spec["near_m"], far_m=spec["far_m"])
            expected = np.asarray(spec["T_world_camera"])
            observed = np.asarray(actual["T_world_camera"])
            error = float(np.max(np.abs(expected-observed)))
            if error > 3e-6:
                raise RuntimeError(f"Wrist camera does not follow its measured hand: {name}, error={error}")
            model = "ideal_d405_prototype" if name.startswith("wrist_") else "ideal_rgbd_from_official_d455_asset"
            self.calibration[name] = {**spec, **actual, "sensor_model": model, "timestamp_s": self.time_s,
                                      "rgb_far_m": 20.}
            row["cameras"][name] = {"T_world_camera": observed.tolist(),
                "T_camera_world": np.linalg.inv(observed).tolist(),
                "T_hand_camera": self.camera_rig.specs[name].get("T_hand_optical_cv"),
                "composition_max_abs_error": error, "sensor_model": model}
        self.camera_pose_trace.append(row)

    def reference_action(self):
        trajectory_time = self.time_s+self.step_dt if self.options.experiment == "cut" else 0.
        desired = self.reference_pose(trajectory_time)
        return torch.tensor(desired[None], device=self.device, dtype=torch.float32)

    def reset(self, seed=None, options=None):
        if (seed is not None and self.task is not None and self.task.variant == "randomized"
                and int(seed) != self.task.seed):
            raise ValueError("Lab randomized geometry/fixtures are fixed at construction; open a new environment with the new seed")
        if seed is not None:
            self.cfg.seed = int(seed)
        return super().reset(seed=seed, options=options)

    def step(self, action=None):
        self._step_action_mode = self.options.action_mode
        self._step_cut_enabled, self._step_cut_index = True, None
        if action is None:
            self._step_action_mode = "cartesian_edge_pose_world"
            action = self.reference_action()
        elif isinstance(action, dict):
            if not self.task:
                raise ValueError("Task action mappings require a CUT/TRIM task")
            command = self.task.validate_action(action)
            self._step_action_mode = command["mode"]
            self._step_cut_enabled, self._step_cut_index = command["cut_enabled"], 0
            values = command["target"]
            if values.shape == (2, 7):
                parked = self._park_q[self._joint_coords[1]][:7]
                if not np.allclose(values[1], parked, atol=1e-8, rtol=0):
                    raise ValueError("Franka 2 is parked in this task version")
                values = values[0]
            action = torch.tensor(values[None], device=self.device, dtype=torch.float32)
        obs, reward, terminated, _, info = super().step(action)
        truncated = torch.tensor([self.time_s >= 8.-1e-9], device=self.device)
        return obs, reward, terminated, truncated, info

    def debug_state(self):
        return {"robot": self.measured(), "material": self.material.debug_state(),
                "coupling": self.last_diagnostics.copy()}
