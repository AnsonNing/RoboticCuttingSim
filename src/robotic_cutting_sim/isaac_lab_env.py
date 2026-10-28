"""Live Isaac Lab 3.0 DirectRLEnv / InteractiveScene cutting task.

The name of Lab's base class does not imply policy training. Actions are fixed
reference blade-edge poses. Lab owns both articulations, dynamics and clock;
the external retained TLMPM owns the material. Cameras in this kit-less adapter
are explicitly ideal Warp RGB-D of the CURRENT full USD scene, not native D455.
Full Kit and native sensors have separate runners and acceptance results.
"""
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from .workcell import ASSETS, PARK_JOINTS, SECONDARY_PARK_JOINTS
from .physical_tool import ToolGraspConfig
from .tool_profile import tool_spec_for_task
from .pad_contact import POINT_PROFILE, PATCH_PROFILE, patch_receipt


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


def _spawn_franka(prim_path, cfg, translation=None, orientation=None, *, task=None,
                  tool_attachment_mode="legacy_fixed_tool", **kwargs):
    """Spawn through Lab, attaching the physical wedge before Newton imports USD."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    if prim_path.endswith("Franka_0") and tool_attachment_mode == "legacy_fixed_tool":
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


def _spawn_free_tool(prim_path, cfg, translation=None, orientation=None, *, spec, grasp,
                     contact_profile=POINT_PROFILE, **kwargs):
    from .newton_tool import TOOL_PATH, author_newton_tool
    if prim_path != TOOL_PATH:
        raise ValueError("The single live tool must be outside the robot hierarchy")
    orientation = (1., 0., 0., 0.) if orientation is None else orientation
    translation = (0., 0., .4) if translation is None else translation
    pose = np.array([*translation, *orientation[1:], orientation[0]], float)
    return author_newton_tool(sim_utils.SimulationContext.instance().stage, spec, pose, grasp,
        contact_profile=contact_profile)


@dataclass(frozen=True)
class LabCuttingConfig:
    device: str = "cuda:0"
    material_device: str = "cuda:0"
    seed: int = 42
    variant: str = "D0"
    task_id: str | None = None
    task_variant: str = "fixed"
    workpiece_kind: str = "rectangle"
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
    tool_attachment_mode: str = "legacy_fixed_tool"
    tool_init_mode: str = "pregrasped"
    grasp: ToolGraspConfig = field(default_factory=ToolGraspConfig)
    grip_contact_profile: str = POINT_PROFILE
    static_scene: str = str(ROOT/"artifacts/research_handoff_001/workcell/workcell.usda")
    # None keeps the historical position-only Lab IK gate; policy use sets
    # the Native rotation gate (0.01 rad) so orientation errors are not silent.
    ik_rotation_tolerance_rad: float | None = None

    def __post_init__(self):
        if self.grip_contact_profile not in (POINT_PROFILE, PATCH_PROFILE):
            raise ValueError("Unsupported Lab grip contact profile")
        if self.grip_contact_profile != POINT_PROFILE and self.tool_attachment_mode != "physical_grasp":
            raise ValueError("Finite pad contacts require an independent physical-grasp tool")
        if self.tool_attachment_mode not in ("legacy_fixed_tool", "physical_grasp"):
            raise ValueError("Unsupported tool attachment mode")
        if self.tool_init_mode not in ("pregrasped", "free", "pickup"):
            raise ValueError("Unsupported tool initialization mode")
        if self.tool_init_mode == "pickup":
            raise NotImplementedError("Lab pickup is not implemented; no fixed-tool fallback is allowed")
        if self.tool_attachment_mode == "legacy_fixed_tool" and self.tool_init_mode != "pregrasped":
            raise ValueError("Legacy fixed tools cannot be initialized free")
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
                     particle_spacing_m=options.particle_spacing_m, workpiece_kind=options.workpiece_kind) if options.task_id else None
    physical = options.tool_attachment_mode == "physical_grasp"
    tool_spec = tool_spec_for_task(task) if physical else None

    def robot(index):
        q = dict(zip([f"panda_joint{i}" for i in range(1, 8)], PARK_JOINTS))
        q["panda_finger_joint.*"] = .04
        if physical and index == 0:
            q["panda_finger_joint.*"] = options.grasp.initial_finger_position_m
        if index == 1:
            q.update(dict(zip([f"panda_joint{i}" for i in range(1, 8)], SECONDARY_PARK_JOINTS)))
        return ArticulationCfg(prim_path=f"/World/Franka_{index}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(asset),
                func=partial(_spawn_franka, task=task, tool_attachment_mode=options.tool_attachment_mode),
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
                    stiffness=options.grasp.stiffness_n_m if physical and index == 0 else 350.,
                    damping=options.grasp.damping_n_s_m if physical and index == 0 else 10.,
                    joint_effort_limit=options.grasp.max_finger_force_n if physical and index == 0 else 20.,
                    armature=.1, viscous_friction=0.)})

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
    if physical:
        from .newton_tool import TOOL_PATH
        scene_cfg.cutting_tool = RigidObjectCfg(prim_path=TOOL_PATH,
            spawn=sim_utils.SpawnerCfg(func=partial(_spawn_free_tool, spec=tool_spec, grasp=options.grasp,
                contact_profile=options.grip_contact_profile)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., .4)))
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
    if physical:
        observation.pop("tool_pose")
        observation.update(finger_q=[2, 2], finger_qdot=[2, 2], hand_pose=7)
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
        self.physical_tool = options.tool_attachment_mode == "physical_grasp"
        self.target = WorkpieceTarget()
        from .tasks import make_task
        self.task = make_task(options.task_id, seed=options.seed, variant=options.task_variant,
            particle_spacing_m=options.particle_spacing_m, workpiece_kind=options.workpiece_kind) if options.task_id else None
        self.mount = np.array(self.task.hand_from_edge_xyz_xyzw if self.task else (0, 0, .18, 1, 0, 0, 0), float)
        self.tool_spec = tool_spec_for_task(self.task) if self.physical_tool else None
        self.gripper_mode = "hold" if options.tool_init_mode == "pregrasped" else "release"
        self.gripper_target_m = options.grasp.closed_target_m
        self.external_tool_wrench_world = np.zeros(6)
        self.grasp_settling_trace = []
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
        self._native_solver = self.sim.physics_manager._solver
        if self.physical_tool:
            from .newton_tool import NewtonTool
            self.dynamic_tool = NewtonTool(self.model, self._native_solver, self.tool_spec,
                contact_profile=options.grip_contact_profile)
            self.knife_shape = next(i for i in self.dynamic_tool.shapes if any(
                s.role == "blade" and self.model.shape_label[i].endswith("/"+s.name) for s in self.tool_spec.shapes))
        else:
            self.knife_shape = next(i for i, label in enumerate(self.model.shape_label) if label.endswith("/Knife"))
            if int(self.model.shape_body.numpy()[self.knife_shape]) != self.hand:
                raise RuntimeError("Legacy wedge collider is not attached to the measured Panda hand")
        self._bind_ik()
        self._force_buffer = wp.zeros(1, dtype=wp.spatial_vector, device=self.model.device)
        self._contact_audit = wp.zeros(4, dtype=float, device=self.model.device)
        # Read-only debug access to the pinned native solver's signed contact
        # distances; stepping still belongs entirely to Lab's PhysicsManager.
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
        if self.physical_tool:
            NewtonManager.register_post_step_callback(self._native_post_step)

    def _native_force(self, state):
        if self._bound:
            if self.physical_tool:
                from .newton_tool import apply_tool_wrench
                wp.launch(apply_tool_wrench, dim=1, inputs=[state.body_q, self.model.body_com,
                    self._force_buffer, self.dynamic_tool.body, wp.vec3(*self.tool_spec.tool_from_edge_xyz_xyzw[:3]),
                    self.physics_dt, state.body_f, self.dynamic_tool.applied_impulse,
                    self.dynamic_tool.current_applied_com, self.dynamic_tool.current_applied_pose,
                    self.hand, self.dynamic_tool.current_pre_hand_wrench], device=self.model.device)
            else:
                self._sample_native_contacts()
                wp.launch(_edge_wrench, dim=1, inputs=[state.body_q, self.model.body_com,
                    self._force_buffer, self.hand, state.body_f], device=self.model.device)

    def _native_post_step(self):
        if self._bound and self.physical_tool:
            self.dynamic_tool.post_substep(NewtonManager.get_state_0(), self.physics_dt)

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
        self._joint_coords, self._joint_dofs = [], []
        starts = self.model.joint_q_start.numpy()
        dof_starts = self.model.joint_qd_start.numpy()
        for arm, robot in enumerate(self.robots):
            coords, dofs = [], []
            for name in robot.joint_names:
                joint = next(i for i, label in enumerate(self.model.joint_label)
                    if f"Franka_{arm}/" in label and label.endswith("/"+name))
                coords.append(starts[joint])
                dofs.append(dof_starts[joint])
            self._joint_coords.append(np.array(coords, dtype=int))
            self._joint_dofs.append(np.array(dofs, dtype=int))
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
        q = self._ik_q.numpy()[0]
        # A free tool adds seven coordinates but six DOFs. Clip only actual
        # robot scalar joints, never quaternion/free-body coordinates.
        for coords, dofs in zip(self._joint_coords, self._joint_dofs):
            q[coords] = np.clip(q[coords], self.model.joint_limit_lower.numpy()[dofs],
                               self.model.joint_limit_upper.numpy()[dofs])
        self._ik_state.joint_q.assign(q)
        newton.eval_fk(self.model, self._ik_state.joint_q, self.model.joint_qd, self._ik_state)
        edge = compose_tool_pose(self._ik_state.body_q.numpy()[self.hand], self.mount[:3], self.mount[3:])
        error = float(np.linalg.norm(edge[:3]-desired[:3]))
        rotation_error = float((Rotation.from_quat(edge[3:]).inv()*Rotation.from_quat(desired[3:])).magnitude())
        self._ik_rotation_error = rotation_error
        tolerance = self.options.ik_rotation_tolerance_rad
        if error > .001 or (tolerance is not None and rotation_error > tolerance):
            from .control import IKFailure
            raise IKFailure(f"Active Lab model IK error {error}m, {rotation_error}rad",
                            {"position_error_m": error, "rotation_error_rad": rotation_error, "converged": False})
        return q, error

    def measured(self):
        import newton
        state = NewtonManager.get_state_0()
        # Publish final-integration FK, not MuJoCo's pre-integration cached xpos.
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        body = state.body_q.numpy().copy()
        q, v = state.joint_q.numpy(), state.joint_qd.numpy()
        hand = body[self.hand].astype(float)
        measured = {"q": np.array([q[indices][:7] for indices in self._joint_coords]),
            "qdot": np.array([v[indices][:7] for indices in self._joint_dofs]),
            "finger_q": np.array([q[indices][7:9] for indices in self._joint_coords]),
            "finger_qdot": np.array([v[indices][7:9] for indices in self._joint_dofs]),
            "hand_pose": hand, "hand_poses": body[self.hands].astype(float), "body_pose": body,
            "tool_pose": compose_tool_pose(hand, self.mount[:3], self.mount[3:]), "timestamp_s": self.time_s}
        if self.physical_tool:
            measured.update(self.dynamic_tool.measured(state))
        return measured

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
        self.gripper_mode = "hold" if self.options.tool_init_mode == "pregrasped" else "release"
        self.external_tool_wrench_world = np.zeros(6)
        if self.physical_tool:
            self.set_grip_force_limit(self.options.grasp.max_finger_force_n)
            self.set_grip_friction(self.options.grasp.friction)
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
        if self.physical_tool:
            q[self._joint_coords[0][7:9]] = self.options.grasp.initial_finger_position_m
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
        if self.physical_tool:
            from .newton_tool import relative_tool_pose
            nominal_pose = matrix_pose(pose_matrix(desired) @ np.linalg.inv(self.tool_spec.T_tool_edge))
            self.dynamic_tool.reset_pose(NewtonManager.get_state_0(), nominal_pose)
            import newton
            state = NewtonManager.get_state_0()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            initial = self.measured()
            self._initial_relative_tool = relative_tool_pose(initial["hand_pose"], initial["tool_body_pose_xyzw"])
        # MuJoCo reset clears warm starts but deliberately defers native qpos
        # synchronization until the next step. Refresh position/contact caches
        # without integrating so contact audits cannot read the old park pose.
        import mujoco_warp
        self._native_solver._update_mjc_data(self._native_solver.mjw_data, self.model, NewtonManager.get_state_0())
        mujoco_warp.fwd_position(self._native_solver.mjw_model, self._native_solver.mjw_data)
        if self.physical_tool:
            if self.options.tool_init_mode == "pregrasped":
                self._settle_physical_grasp()
            else:
                self.grasp_settling_trace = []
                self.grasp_initialization = {"status": "FREE_TOOL_NOT_GRASPED", "hidden_attachment": False}
            # Settling is real rigid dynamics, outside the formal8s budget.
            self.time_s = 0.
            self.episode_length_buf[env_ids] = 0
            self._runtime_time_origin = self.sim.physics_manager.get_simulation_time()
            self.dynamic_tool.begin_interval()
            self.dynamic_tool.phase = "formal_episode"
            from .grasp_monitor import GraspMonitor
            settled = self.measured()
            self.grasp_monitor = GraspMonitor(settled["hand_pose"], settled["tool_body_pose_xyzw"], self.tool_spec)
        self.setup_receipt = self.material.reset(self.measured()["tool_pose"], self.cfg.seed)
        self.setup_receipt["native_contact_reset_refresh"] = "native coordinate sync + fwd_position; zero integration"
        self._last_joint_targets = [q[self._joint_coords[0]].copy(), self._park_q[self._joint_coords[1]].copy()]
        self.setup_receipt["velocity_feedforward"] = self.options.velocity_feedforward
        self.setup_receipt["experiment"] = self.options.experiment
        self.setup_receipt["velocity_target_source"] = "commanded IK joint finite difference" if self.options.velocity_feedforward else "zero"
        self.setup_receipt["tool_attachment_mode"] = self.options.tool_attachment_mode
        if self.physical_tool:
            from dataclasses import asdict
            self.setup_receipt.update(tool_spec=self.tool_spec.to_dict(), grasp=asdict(self.options.grasp),
                tool_initialization=self.grasp_initialization, tool_body_index=self.dynamic_tool.body,
                tool_joint_index=self.dynamic_tool.joint,
                actual_tool_mass_kg=float(self.model.body_mass.numpy()[self.dynamic_tool.body]),
                actual_tool_com_local_m=self.model.body_com.numpy()[self.dynamic_tool.body].tolist(),
                actual_tool_inertia_local_kg_m2=self.model.body_inertia.numpy()[self.dynamic_tool.body].tolist(),
                rigid_model_gravity_world_m_s2=self.model.gravity.numpy().tolist(),
                tool_body_world_index=int(self.model.body_world.numpy()[self.dynamic_tool.body]),
                native_mujoco_gravity_world_m_s2=self._native_solver.mjw_model.opt.gravity.numpy().tolist(),
                grip_contact_profile=self.options.grip_contact_profile,
                finite_pad_patch=patch_receipt(self.tool_spec.handle_half_extents_m, self.options.grasp.friction)
                    if self.options.grip_contact_profile == PATCH_PROFILE else None,
                grasp_monitor_reference=self.grasp_monitor.last,
                tool_force_receiver="independent tool body only; never panda_hand",
                measured_tool_pose_source="Newton free joint/body_q; no episode pose correction",
                settling_runtime_origin_s=self._runtime_time_origin)
        from .usd_surface_bridge import DynamicSurfaceBridge
        self._bridge = DynamicSurfaceBridge(self.sim.stage)
        self._capture_time = None
        self.camera_pose_trace = []
        self.last_diagnostics = {}
        self._step_cut_enabled, self._step_cut_index = True, None
        self._rigid_probe_active = False
        self._rigid_probe_time_s = 0.
        self._rigid_probe_target = self.measured()["q"][0].copy()

    def _settle_physical_grasp(self):
        grasp = self.options.grasp
        self.grasp_settling_trace = []
        self.dynamic_tool.begin_interval()
        for arm, robot in enumerate(self.robots):
            target = self.measured()["q"][arm]
            fingers = [grasp.closed_target_m]*2 if arm == 0 else [.04]*2
            values = np.r_[target, fingers]
            robot.actuators.target_command.set_position_index(
                value=torch.tensor(values[None], device=self.device, dtype=torch.float32))
            robot.actuators.target_command.set_velocity_index(
                value=torch.zeros((1, 9), device=self.device))
        count = int(round(grasp.settle_s/self.physics_dt))
        for index in range(count):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            if (index+1) % self.cfg.decimation == 0:
                state = self.measured()
                row = self.dynamic_tool.diagnostics(NewtonManager.get_state_0(), state["hand_pose"], self._initial_relative_tool)
                row.update(settling_time_s=(index+1)*self.physics_dt, hand_pose_xyzw=state["hand_pose"],
                    finger_q=state["finger_q"], finger_qdot=state["finger_qdot"])
                self.grasp_settling_trace.append(row)
                self.dynamic_tool.begin_interval()
        window = [r for r in self.grasp_settling_trace if r["settling_time_s"] >= grasp.settle_s-grasp.stability_window_s-1e-9]
        checks = {"actual_free_body": True, "no_hidden_attachment": True, "nonempty_stability_window": bool(window),
            "bilateral_load_bearing_contacts": bool(window) and np.mean([r["bilateral_substep_fraction"] for r in window]) >= grasp.min_bilateral_contact_fraction,
            "translation_stable": bool(window) and max(r["relative_translation_drift_m"] for r in window) <= grasp.translation_tolerance_m,
            "rotation_stable": bool(window) and max(r["relative_rotation_drift_rad"] for r in window) <= grasp.rotation_tolerance_rad,
            "linear_speed_stable": bool(window) and max(np.linalg.norm(r["tool_linear_velocity_world_m_s"]) for r in window) <= grasp.linear_speed_tolerance_m_s,
            "angular_speed_stable": bool(window) and max(np.linalg.norm(r["tool_angular_velocity_world_rad_s"]) for r in window) <= grasp.angular_speed_tolerance_rad_s,
            "penetration_bounded": bool(window) and max(r["max_tool_rigid_penetration_m"] for r in window) <= grasp.max_penetration_m,
            "no_unintended_support": bool(window) and all(r["unintended_tool_contact_slots"] == 0 for r in window)}
        self.grasp_initialization = {"status": "PASS_PHYSICAL_PREGRASP" if all(checks.values()) else "FAIL_PHYSICAL_PREGRASP",
            "checks": checks, "rigid_settle_s": count*self.physics_dt, "material_advanced_s": 0.,
            "formal_episode_time_s": 0., "window_s": grasp.stability_window_s,
            "min_bilateral_fraction": grasp.min_bilateral_contact_fraction, "hidden_attachment": False}
        if not all(checks.values()):
            raise RuntimeError(f"Physical pregrasp failed: {self.grasp_initialization}")

    def set_gripper(self, mode, target_m=None):
        if not self.physical_tool:
            raise ValueError("Gripper/tool skills require physical_grasp; no welded-tool release is simulated")
        if mode not in ("hold", "position", "release"):
            raise ValueError("Gripper mode must be hold, position or release")
        if target_m is not None and (not np.isfinite(target_m) or not 0. <= target_m <= .04):
            raise ValueError("Finger target must be finite and within0..40mm")
        self.gripper_mode = mode
        if target_m is not None:self.gripper_target_m = float(target_m)
        return {"mode": mode, "target_m": self._finger_target()}

    def _finger_target(self):
        return .04 if self.gripper_mode == "release" else self.gripper_target_m

    def solve_edge_ik(self, edge_pose):
        desired = np.asarray(edge_pose, float)
        q, error = self._solve_ik(desired)
        edge = compose_tool_pose(self._ik_state.body_q.numpy()[self.hand], self.mount[:3], self.mount[3:])
        rotation_error = float((Rotation.from_quat(desired[3:]).inv()*Rotation.from_quat(edge[3:])).magnitude())
        return q[self._joint_coords[0]][:7].copy(), {
            "position_error_m": error, "rotation_error_rad": rotation_error,
            "residual_scope": "planned hand-mount FK; independent free tool not substituted"}

    def set_grip_force_limit(self, limit_n):
        if not self.physical_tool or not np.isfinite(limit_n) or not 0. < limit_n <= 40.:
            raise ValueError("Physical per-finger effort limit must be finite within0..40N")
        self.robots[0].write_joint_effort_limit_to_sim_index(limits=float(limit_n), joint_ids=[7, 8])
        return {"per_finger_limit_n": float(limit_n), "physics": "Lab joint_effort_limit, native joint DOF properties"}

    def set_grip_friction(self, coefficient):
        if not self.physical_tool or not np.isfinite(coefficient) or not 0. <= coefficient <= 2.:
            raise ValueError("Finite grip friction must be within0..2")
        import newton
        bodies = self.model.shape_body.numpy()
        mask = np.isin(bodies, [self.dynamic_tool.body, self.dynamic_tool.left, self.dynamic_tool.right])
        mu = self.model.shape_material_mu.numpy();mu[mask] = coefficient
        self.model.shape_material_mu.assign(mu)
        torsional = None
        if self.options.grip_contact_profile == PATCH_PROFILE:
            torsional = patch_receipt(self.tool_spec.handle_half_extents_m, coefficient)["torsional_friction_length_m"]
            values = self.model.shape_material_mu_torsional.numpy();values[mask] = torsional
            self.model.shape_material_mu_torsional.assign(values)
        NewtonManager.add_model_change(newton.ModelFlags.SHAPE_PROPERTIES)
        return {"coefficient": float(coefficient), "applied_to": "actual tool AND both finger collision shapes",
                "torsional_coefficient_m": torsional, "shape_indices": np.flatnonzero(mask).tolist()}

    def set_external_tool_wrench(self, wrench_world):
        wrench = np.asarray(wrench_world, float)
        if not self.physical_tool or wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("Expected finite world force/torque6 at actual edge in physical mode")
        self.external_tool_wrench_world = wrench.copy()

    def rigid_probe_step(self, joint_targets=None):
        """Actual rigid-only diagnostic step; reset is required before cutting."""
        if not self.physical_tool or self.time_s != 0.:
            raise RuntimeError("Rigid-only probes require physical mode and a fresh formal reset")
        self._rigid_probe_active = True
        if joint_targets is not None:
            values = np.asarray(joint_targets, float)
            dofs = self._joint_dofs[0][:7]
            if values.shape != (7,) or not np.isfinite(values).all() or np.any(values < self.model.joint_limit_lower.numpy()[dofs]) or np.any(values > self.model.joint_limit_upper.numpy()[dofs]):
                raise ValueError("Expected seven finite joint positions within actual arm limits")
            self._rigid_probe_target = values.copy()
        self.dynamic_tool.phase = "rigid_probe"
        self.dynamic_tool.begin_interval()
        self._force_buffer.assign(self.external_tool_wrench_world.astype(np.float32).reshape(1, 6))
        for arm, robot in enumerate(self.robots):
            q = np.r_[self._rigid_probe_target, [self._finger_target()]*2] if arm == 0 else self._park_q[self._joint_coords[arm]]
            robot.actuators.target_command.set_position_index(value=torch.tensor(q[None], device=self.device, dtype=torch.float32))
            robot.actuators.target_command.set_velocity_index(value=torch.zeros((1, 9), device=self.device))
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.physics_dt)
        self._rigid_probe_time_s += self.step_dt
        state = self.measured();state["timestamp_s"] = self._rigid_probe_time_s
        debug = self.dynamic_tool.diagnostics(NewtonManager.get_state_0(), state["hand_pose"], self._initial_relative_tool)
        return state, {"timestamp_s": self._rigid_probe_time_s, "formal_episode_time_s": 0.,
            "material_advanced_s": 0., "physical_tool": debug, "external_tool_wrench_edge_world": self.external_tool_wrench_world.copy(),
            "gripper_mode": self.gripper_mode, "gripper_target_m": self._finger_target()}

    def _pre_physics_step(self, actions):
        if self.time_s >= 8.-1e-9:
            raise RuntimeError("Episode ended; call reset")
        self._previous = self.measured()
        values = actions[0].detach().cpu().numpy().astype(float)
        if self._step_action_mode == "joint_position_rad":
            import newton
            q = NewtonManager.get_state_0().joint_q.numpy().copy()
            coordinates = self._joint_coords[0][:7]
            dofs = self._joint_dofs[0][:7]
            if (not np.isfinite(values).all() or np.any(values < self.model.joint_limit_lower.numpy()[dofs])
                    or np.any(values > self.model.joint_limit_upper.numpy()[dofs])):
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
            if self.physical_tool and arm == 0:
                values[7:9] = self._finger_target()
            robot.actuators.target_command.set_position_index(
                value=torch.tensor(values[None], device=self.device, dtype=torch.float32))
            velocity = (values-self._last_joint_targets[arm])/self.step_dt if self.options.velocity_feedforward else np.zeros_like(values)
            if self.physical_tool:velocity[7:9] = 0.
            robot.actuators.target_command.set_velocity_index(
                value=torch.tensor(velocity[None], device=self.device, dtype=torch.float32))
            self._last_joint_targets[arm] = values.copy()
        applied = self.material.pending_wrench if self.options.feedback else np.zeros(6)
        if self.physical_tool:
            applied = np.asarray(applied)+self.external_tool_wrench_world
            self.dynamic_tool.begin_interval()
        self._force_buffer.assign(np.asarray(applied, np.float32).reshape(1, 6))
        self._contact_audit.zero_()

    def _apply_action(self):
        pass  # Lab's scene writes the persistent actuator targets before its own dynamics step.

    def _get_rewards(self):
        self.time_s = float(self.episode_length_buf[0])*self.step_dt
        measured = self.measured()
        diagnostics = self.material.advance(measured["tool_pose"], self.time_s,
            cutting=self.options.experiment == "cut" and self._step_cut_enabled, cut_index=self._step_cut_index)
        tool_debug = None
        if self.physical_tool:
            tool_debug = self.dynamic_tool.diagnostics(NewtonManager.get_state_0(), measured["hand_pose"], self._initial_relative_tool)
            contact = [tool_debug["max_non_tool_penetration_m"], tool_debug["max_tool_rigid_penetration_m"],
                       0., float(self._native_solver.mjw_data.nacon.numpy()[0])]
        else:
            self._sample_native_contacts()
            contact = self._contact_audit.numpy()
        delta_rotation = (Rotation.from_quat(measured["tool_pose"][3:])
            * Rotation.from_quat(self._previous["tool_pose"][3:]).inv()).as_rotvec()
        applied = diagnostics["applied_previous_material_wrench_world"]
        self.last_diagnostics = {**diagnostics, "desired_tool_pose": self._desired.copy(),
            "tool_attachment_mode": self.options.tool_attachment_mode,
            "ik_position_error_m": self._ik_error,
            "runtime_timestamp_s": self.sim.physics_manager.get_simulation_time()-self._runtime_time_origin,
            "native_runtime_timestamp_s": self.sim.physics_manager.get_simulation_time(),
            "native_rigid_contact_audit": {"source": "actual MJWarp nacon/contact.dist/geom; pre-substeps plus final",
                "max_non_tool_penetration_m": float(contact[0]), "max_tool_rigid_penetration_m": float(contact[1]),
                "observed_non_tool_contact_slots": int(contact[2]), "max_simultaneous_contacts": int(contact[3])},
            "native_float32_wrench_payload_world": self._force_buffer.numpy()[0].astype(float),
            "applied_tool_wrench_work_j": float(applied[:3]@(measured["tool_pose"][:3]-self._previous["tool_pose"][:3])
                + applied[3:]@delta_rotation)}
        if self.physical_tool:
            grasp_status = self.grasp_monitor.update(measured["hand_pose"], measured["tool_body_pose_xyzw"],
                time_s=self.time_s, bilateral_loaded_tool_fraction=tool_debug["bilateral_substep_fraction"],
                contacts_valid=tool_debug["observed_substeps"] == self.cfg.decimation,
                gripper_command=self.gripper_mode)
            self.last_diagnostics.update(physical_tool=tool_debug, gripper_control_mode=self.gripper_mode,
                gripper_target_m=self._finger_target(), finger_q=measured["finger_q"], finger_qdot=measured["finger_qdot"],
                tool_external_wrench_edge_world=self.external_tool_wrench_world.copy(),
                applied_wrench_receiver_body_index=self.dynamic_tool.body,
                direct_hand_material_wrench=False,
                grasp_monitor=grasp_status, grasp_status=grasp_status["status"], grasp_lost=grasp_status["grasp_lost"],
                grasp_contact_valid=tool_debug["bilateral_substep_fraction"] >= self.options.grasp.min_bilateral_contact_fraction)
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
        keys = ("q", "qdot", "finger_q", "finger_qdot", "hand_pose") if self.physical_tool else ("q", "qdot", "tool_pose")
        policy = {key: torch.as_tensor(measured[key][None], device=self.device) for key in keys}
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
        if self.physical_tool and self._rigid_probe_active:
            raise RuntimeError("Rigid-only probes cannot resume a formal cutting episode; reset first")
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
            self._step_cut_enabled, self._step_cut_index = command["cut_enabled"], command.get("cut_index", 0)
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
