"""Live Isaac Lab 3.0 DirectRLEnv / InteractiveScene cutting task.

The name of Lab's base class does not imply policy training. Actions are fixed
reference blade-edge poses. Lab owns both articulations, dynamics and clock;
the external retained TLMPM owns the material. Cameras in this kit-less adapter
are explicitly ideal Warp RGB-D of the CURRENT full USD scene, not native D455.
Full Kit and native sensors have separate runners and acceptance results.
"""
from dataclasses import dataclass
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


def _spawn_franka(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn through Lab, attaching the physical wedge before Newton imports USD."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    if prim_path.endswith("Franka_0"):
        stage = prim.GetStage()
        hand_prim = next(p for p in Usd.PrimRange(prim) if p.GetName() == "panda_hand"
                         and p.HasAPI(UsdPhysics.RigidBodyAPI))
        mesh = UsdGeom.Mesh.Define(stage, str(hand_prim.GetPath())+"/Knife")
        vertices, faces = wedge_mesh()
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(.38, .65, .85)])
        xf = UsdGeom.Xformable(mesh)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, .18))
        xf.AddOrientOp().Set(Gf.Quatf(0, Gf.Vec3f(1, 0, 0)))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return prim


@dataclass(frozen=True)
class LabCuttingConfig:
    device: str = "cuda:0"
    material_device: str = "cuda:0"
    seed: int = 42
    variant: str = "D0"
    dx: float = .004
    feedback: bool = True
    width: int = 640
    height: int = 480
    global_width: int = 1280
    global_height: int = 960
    velocity_feedforward: bool = False
    experiment: str = "cut"
    camera: bool = True
    static_scene: str = str(ROOT/"artifacts/research_handoff_001/workcell/workcell.usda")


def make_lab_cfg(options):
    """Use the actual GA workspace APIs and explicit solver settings."""
    asset = ROOT/"artifacts/workcell_asset_cache"/ASSETS["franka"]
    if not asset.is_file():
        raise FileNotFoundError(asset)

    def robot(index):
        q = dict(zip([f"panda_joint{i}" for i in range(1, 8)], PARK_JOINTS))
        q["panda_finger_joint.*"] = .04
        if index == 1:
            q["panda_joint1"] = -np.pi/2
        return ArticulationCfg(prim_path=f"/World/Franka_{index}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(asset), func=_spawn_franka,
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

    def fixed_box(name, center, size, color):
        return AssetBaseCfg(prim_path="/World/"+name, init_state=AssetBaseCfg.InitialStateCfg(pos=center),
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

    observation = {"q": [2, 7], "qdot": [2, 7], "tool_pose": 7}
    if options.camera:
        for name in ("d455_0", "d455_1"):
            observation[name+"_rgb"] = [options.height, options.width, 3]
            observation[name+"_depth_m"] = [options.height, options.width]
    cfg = DirectRLEnvCfg(seed=options.seed, decimation=10, episode_length_s=8.,
        action_space=7, observation_space=observation, ui_window_class_type=None,
        scene=CuttingSceneCfg(num_envs=1, env_spacing=2.),
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
        self.material = LiveCuttingMaterial(CouplingConfig(dx=options.dx,
            device=options.material_device, feedback=options.feedback, variant=options.variant), self.target)
        self._bound = False
        self._capture_time = None
        self.time_s = 0.
        self.last_diagnostics = {}
        super().__init__(cfg)
        self.model = NewtonManager.get_model()
        self.robots = [self.scene[f"franka_{i}"] for i in range(2)]
        self.hand = next(i for i, label in enumerate(self.model.body_label)
                         if "Franka_0/" in label and label.endswith("/panda_hand"))
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
        self._position = ik.IKObjectivePosition(link_index=self.hand, link_offset=wp.vec3(0, 0, .18),
            target_positions=wp.array([wp.vec3(0, 0, .1)], dtype=wp.vec3, device=self.model.device))
        self._rotation = ik.IKObjectiveRotation(link_index=self.hand,
            link_offset_rotation=wp.quat(1, 0, 0, 0),
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
        edge = compose_tool_pose(self._ik_state.body_q.numpy()[self.hand], (0, 0, .18), (1, 0, 0, 0))
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
            "hand_pose": hand, "body_pose": body,
            "tool_pose": compose_tool_pose(hand, (0, 0, .18), (1, 0, 0, 0)), "timestamp_s": self.time_s}

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
        desired = self.target.poses_to_world(reference_pose(0, self.target.height_m+self.target.edge_clearance_m))
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
        self.last_diagnostics = {}

    def _pre_physics_step(self, actions):
        if self.time_s >= 8.-1e-9:
            raise RuntimeError("Episode ended; call reset")
        self._desired = actions[0].detach().cpu().numpy().astype(float)
        self._previous = self.measured()
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
        diagnostics = self.material.advance(measured["tool_pose"], self.time_s, cutting=self.options.experiment == "cut")
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
        self._surface = ComponentSurface(self.options.dx/2, self.options.material_device,
            voxel_size=min(self.options.dx/2, .0015))
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
            self.surface = (vertices, faces)
            self._bridge.update(vertices, faces, self.time_s)
            self._bridge.mesh.CreateDisplayColorAttr([Gf.Vec3f(.72, .32, .13)])
            for i, op in self._body_ops:
                op.Set(Gf.Matrix4d(pose_matrix(measured["body_pose"][i]).T.tolist()))
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
                            "source": "live_isaac_lab_ideal_warp_rgbd"}
            self._capture_time = self.time_s
        if self.options.camera:
            for name in ("d455_0", "d455_1"):
                for key in ("rgb", "depth_m"):
                    policy[name+"_"+key] = torch.as_tensor(self.frames[name][key][None], device=self.device)
        return {"policy": policy}

    def reference_action(self):
        trajectory_time = self.time_s+self.step_dt if self.options.experiment == "cut" else 0.
        desired = self.target.poses_to_world(reference_pose(trajectory_time,
            self.target.height_m+self.target.edge_clearance_m))
        return torch.tensor(desired[None], device=self.device, dtype=torch.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.cfg.seed = int(seed)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, _, info = super().step(action)
        truncated = torch.tensor([self.time_s >= 8.-1e-9], device=self.device)
        return obs, reward, terminated, truncated, info

    def debug_state(self):
        return {"robot": self.measured(), "material": self.material.debug_state(),
                "coupling": self.last_diagnostics.copy()}
