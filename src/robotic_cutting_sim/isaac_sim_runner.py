"""Live Isaac Sim 6.2 PhysX / retained TLMPM / RTX integration.

Isaac modules are imported only after SimulationApp starts. The Newton robot
object supplies IK targets only: its dynamics step methods are never called.
All published robot poses come from native PhysX tensors. No replay input exists.
"""
from dataclasses import asdict, dataclass, replace
import importlib.metadata
from pathlib import Path
import time

import numpy as np

from .recording import ROOT, write_json


@dataclass(frozen=True)
class IsaacSimCuttingConfig:
    seed: int = 42
    steps: int = 400
    control_dt: float = .02
    robot_substeps: int = 10
    material_dt: float = .0001
    dx: float = .004
    device: str | None = None
    gpu: int = 0
    width: int = 640
    height: int = 480
    global_resolution_scale: int = 2
    record_every: int = 10
    render_warmup: int = 5
    camera_mode: str = "rtx_rgbd"
    camera_layout: str = "legacy_fixed_d455"
    observation_motion_probe: bool = False
    feedback: bool = True
    velocity_feedforward: bool | None = None
    task_id: str | None = None
    task_variant: str = "fixed"
    action_mode: str = "cartesian"
    native_noise: str | None = None
    surface_profile: str | None = None
    particle_spacing_m: float | None = None
    scene: str | None = None

    def __post_init__(self):
        if self.gpu < 0:
            raise ValueError("gpu must be a nonnegative physical GPU index")
        if self.device is None:
            object.__setattr__(self, "device", f"cuda:{self.gpu}")
        if min(self.steps, self.robot_substeps, self.width, self.height, self.global_resolution_scale, self.record_every, self.render_warmup) < 1:
            raise ValueError("Step counts, resolution and recording intervals must be positive")
        values = np.asarray([self.control_dt, self.material_dt, self.dx])
        if not np.isfinite(values).all() or np.min(values) <= 0:
            raise ValueError("Spatial and time steps must be finite and positive")
        if abs(round(self.control_dt/self.material_dt)*self.material_dt-self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of material_dt")
        if self.camera_mode not in ("rtx_rgbd", "native_single_view"):
            raise ValueError("Choose rtx_rgbd or native_single_view; no ideal fallback is available")
        if self.camera_layout not in ("legacy_fixed_d455", "wrist_d405_global_d455"):
            raise ValueError("Unknown native camera_layout")
        if self.camera_layout == "wrist_d405_global_d455" and self.camera_mode != "native_single_view":
            raise ValueError("The wrist D405 / global D455 layout requires camera_mode='native_single_view'")
        if self.camera_layout == "wrist_d405_global_d455" and (self.width, self.height) != (640,480):
            raise ValueError("The validated native global D455 profile requires640x480")
        if self.observation_motion_probe and self.camera_layout != "wrist_d405_global_d455":
            raise ValueError("The observation motion probe requires the wrist-camera layout")
        if self.task_id not in (None, "CUT-01", "TRIM-01"):
            raise ValueError("task_id must be CUT-01, TRIM-01 or None for the legacy D0 recipe")
        if self.action_mode not in ("cartesian", "joint"):
            raise ValueError("action_mode must be cartesian or joint")
        if self.velocity_feedforward is None:
            object.__setattr__(self,"velocity_feedforward",self.task_id is not None)
        if self.native_noise is None:
            object.__setattr__(self,"native_noise","off" if self.task_id else "on")
        if self.native_noise not in ("off", "on"):
            raise ValueError("native_noise must be off or on, matching independent sensor controls")
        if self.surface_profile is None:
            object.__setattr__(self,"surface_profile","current_volume_v1" if self.task_id else "legacy")
        if self.surface_profile not in ("legacy","current_volume_v1"):
            raise ValueError("surface_profile must be legacy or current_volume_v1")
        if self.particle_spacing_m is not None:
            if self.task_id is None:
                raise ValueError("Explicit particle spacing requires CUT/TRIM task_id; legacy D0 is unchanged")
            if not np.isfinite(self.particle_spacing_m) or self.particle_spacing_m<=0:
                raise ValueError("Explicit task particle spacing must be finite and positive")


def runtime_versions():
    packages = {}
    for name in ("isaacsim", "isaacsim-kernel", "isaacsim-core", "isaacsim-sensor", "torch", "warp-lang", "newton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def require_sim_62(packages):
    for name in ("isaacsim", "isaacsim-kernel", "isaacsim-core", "isaacsim-sensor"):
        if packages.get(name) != "6.2.0.0":
            raise RuntimeError(f"Requires actual {name}==6.2.0.0; found {packages.get(name)!r}")


def numpy_data(data):
    return data.numpy().copy() if hasattr(data, "numpy") else np.asarray(data).copy()


def pose_from_wxyz(position, orientation):
    position, orientation = np.asarray(position, float), np.asarray(orientation, float)
    if position.shape != (3,) or orientation.shape != (4,) or not np.isfinite(np.r_[position, orientation]).all():
        raise ValueError("Expected a finite measured position and WXYZ quaternion")
    norm = np.linalg.norm(orientation)
    if abs(norm-1) > 1e-3:
        raise ValueError("Measured orientation is not a unit quaternion")
    return np.r_[position, orientation[[1, 2, 3, 0]]/norm]


def validate_native_action(action, mode):
    """Validate an explicit arm command without importing the native runtime."""
    value=np.asarray(action,float)
    if mode=="cartesian":
        if value.shape!=(7,) or not np.isfinite(value).all():
            raise ValueError("Native Cartesian action must contain seven finite values")
        from .robot import validate_pose
        return validate_pose(value)
    if mode!="joint":
        raise ValueError("Native arm action mode must be cartesian or joint")
    if value.shape not in ((7,),(2,7)) or not np.isfinite(value).all():
        raise ValueError("Native joint action must contain seven finite values or two such arm targets")
    return value.copy()


def calibration_from_replicator(parameters):
    """Use runtime projection, including render-product aspect fitting."""
    width, height = map(int, parameters["renderProductResolution"])
    projection = np.asarray(parameters["cameraProjection"], float).reshape(4,4)
    view = np.asarray(parameters["cameraViewTransform"], float).reshape(4,4)
    scale = float(parameters["metersPerSceneUnit"])
    if min(width,height) <= 0 or not np.isfinite(scale) or scale <= 0 or not np.isfinite(projection).all():
        raise ValueError("Malformed native camera-parameter annotator output")
    world = np.linalg.inv(view.T)
    world[:3,3] *= scale
    world[:3,:3] = world[:3,:3] @ np.diag([1,-1,-1])
    k = [[projection[0,0]*width/2,0,width*(1-projection[2,0])/2],
         [0,projection[1,1]*height/2,height*(1+projection[2,1])/2],[0,0,1]]
    return {"width":width,"height":height,"K":np.asarray(k).tolist(),"T_world_camera":world.tolist(),
            "meters_per_scene_unit":scale,"source":"native Replicator camera_params projection/view",
            "convention":"OpenCV +X right,+Y down,+Z forward; optical Z meters"}


class LiveIsaacSimCutting:
    """One controlled live loop; material state and forces remain debug-only."""

    def __init__(self, app, config, output):
        self.app, self.config, self.output = app, config, Path(output)
        self.time_s = 0.
        self.step_index = 0
        self.reset_index = -1
        self._closed = False

    def reset(self,seed=None):
        from argparse import Namespace
        import omni.timeline
        import omni.usd
        import warp as wp
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from .coupled_robot import CouplingConfig, reference_pose
        from .cutting_surface import ComponentSurface
        from .live_material import LiveCuttingMaterial
        from .robot import DualFrankaController, RobotConfig, compose_tool_pose, wedge_mesh
        from .usd_surface_bridge import DynamicSurfaceBridge
        from .workcell import export

        if seed is not None:
            self.config=replace(self.config,seed=int(seed))
        c = self.config
        self.reset_index += 1
        self.SimulationManager = SimulationManager
        self.compose_tool_pose = compose_tool_pose
        self.reference_pose = reference_pose
        self.task=None
        self.mount=np.array([0.,0.,.18,1.,0.,0.,0.])
        if c.task_id:
            from .tasks import make_task
            self.task=make_task(c.task_id,seed=c.seed,variant=c.task_variant,particle_spacing_m=c.particle_spacing_m)
            if abs(self.task.control_dt_s-c.control_dt)>1e-12 or abs(self.task.grid_dx_m-c.dx)>1e-12:
                raise ValueError("Native Kit control_dt/dx must match the shared task specification")
            self.reference_pose=self.task.reference_edge_pose_world
            self.mount=np.asarray(self.task.hand_from_edge_xyz_xyzw,float)
        self.wp = wp
        omni.timeline.get_timeline_interface().stop()
        context = omni.usd.get_context()
        if c.scene:
            source = Path(c.scene).resolve()
        else:
            scene_dir = self.output/f"initial_workcell_{self.reset_index:02d}"
            scene_dir.mkdir()
            export(Namespace(variant="D0", family="dome", seed=c.seed, spacing=c.dx/2, asset_root=None), scene_dir)
            source = scene_dir/"workcell.usda"
        if not context.open_stage(str(source)):
            raise RuntimeError("Isaac could not open the initial workcell")
        for _ in range(300):
            self.app.update()
            if context.get_stage_loading_status()[2] == 0:
                break
        else:
            raise TimeoutError("Workcell loading exceeded300 Kit updates")
        stage = self.stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        if UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != "Z":
            raise ValueError("Live fixture currently requires a meter/Z-up source workcell")
        hand_paths = []
        camera_rigid_paths=[]
        for prim in stage.Traverse():
            if any(str(prim.GetPath()).startswith(f"/World/D455_{i}/") for i in (0,1)) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                camera_rigid_paths.append(str(prim.GetPath()))
        for index in (0, 1):
            root = stage.GetPrimAtPath(f"/World/Franka_{index}")
            if not root:
                raise RuntimeError(f"Missing Franka_{index}")
            root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
            root.GetVariantSets().GetVariantSet("Colliders").SetVariantSelection("convex_hulls")
            PhysxSchema.PhysxArticulationAPI.Apply(root)
            hand = [p for p in Usd.PrimRange(root) if p.GetName() == "panda_hand" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(hand) != 1:
                raise RuntimeError(f"Expected one native rigid hand on Franka_{index}; found{len(hand)}")
            hand_paths.append(str(hand[0].GetPath()))
        old = stage.GetPrimAtPath("/World/KinematicBlade")
        if old:
            old.SetActive(False)
        # A visual child shares the actual hand transform; no commanded world
        # pose is written to this mesh during any episode step.
        knife_path = hand_paths[0]+"/LiveKnife"
        self.knife_path=knife_path
        knife = UsdGeom.Mesh.Define(stage, knife_path)
        blade_dimensions=([self.task.blade_length_m,self.task.blade_height_m,self.task.blade_half_width_m]
                          if self.task else [.012,.004,.025])
        vertices, faces = wedge_mesh(*blade_dimensions)
        knife.CreatePointsAttr(vertices)
        knife.CreateFaceVertexCountsAttr([3]*len(faces))
        knife.CreateFaceVertexIndicesAttr(faces.ravel())
        knife.CreateSubdivisionSchemeAttr("none")
        knife.CreateDisplayColorAttr([Gf.Vec3f(.38, .65, .85)])
        xf = UsdGeom.Xformable(knife)
        xf.AddTranslateOp().Set(Gf.Vec3d(*self.mount[:3]))
        xf.AddOrientOp().Set(Gf.Quatf(float(self.mount[6]),Gf.Vec3f(*self.mount[3:6])))
        if self.task:
            self._author_task_fixture()
        for name in ("Table", "TrayFloor", "TrayLeft", "TrayRight", "TrayFront", "TrayBack"):
            prim = stage.GetPrimAtPath("/World/"+name)
            if prim:
                UsdPhysics.CollisionAPI.Apply(prim)
        # PreviewSurface keeps the bounded native geometry cache independent of
        # unresolved remote MDL libraries. It does not alter geometry or physics.
        for name, color in {"Franka_0":(.8,.82,.85),"Franka_1":(.8,.82,.85),
                            "ContinuousSpecimen":(.72,.32,.13), "D455_0":(.2,.25,.3),"D455_1":(.2,.25,.3)}.items():
            prim = stage.GetPrimAtPath("/World/"+name)
            material = UsdShade.Material.Define(stage, "/World/LiveMaterials/"+name)
            shader = UsdShade.Shader.Define(stage, str(material.GetPath())+"/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        SimulationManager.setup_simulation(dt=c.control_dt/c.robot_substeps, device="cpu")
        if not SimulationManager.switch_physics_engine("physx"):
            raise RuntimeError("Native Isaac PhysX engine is unavailable")
        SimulationManager.enable_gpu_dynamics(False)
        SimulationManager.enable_fabric(False)
        self.arms = Articulation(["/World/Franka_0", "/World/Franka_1"])
        self.hand = RigidPrim(hand_paths[0])
        self.hands = RigidPrim(hand_paths)
        self.hand_paths = hand_paths
        SimulationManager.initialize_physics()
        if not self.arms.is_physics_tensor_entity_valid() or not self.hand.is_physics_tensor_entity_valid():
            raise RuntimeError("Actual PhysX articulation/hand tensor views did not initialize")
        expected = [f"panda_joint{i}" for i in range(1, 8)]+["panda_finger_joint1", "panda_finger_joint2"]
        if self.arms.dof_names != expected:
            raise RuntimeError(f"Unexpected PhysX DOF order: {self.arms.dof_names}")
        self.arms.set_dof_gains([1000.]*4+[600.]*3+[350.,350.], [80.]*4+[40.]*3+[10.,10.])
        self.arms.set_dof_max_efforts([87.]*4+[12.]*3+[20.,20.])
        self.arms.set_solver_iteration_counts(position_counts=16, velocity_counts=4)
        self.ik = DualFrankaController(RobotConfig(control_dt=c.control_dt, tool_shape="cut_wedge",
                   tool_offset_m=tuple(self.mount[:3]),tool_rotation_xyzw=tuple(self.mount[3:]),
                   wedge_length_m=blade_dimensions[0],wedge_height_m=blade_dimensions[1],wedge_half_width_m=blade_dimensions[2]))
        self.park_q=self.ik.initial_q.reshape(2,9).copy()
        if c.camera_layout == "wrist_d405_global_d455":
            from .camera_rig import SECONDARY_OBSERVATION_JOINTS
            self.park_q[1,:7]=SECONDARY_OBSERVATION_JOINTS
        paths=[]
        for index,label in enumerate(self.ik.model.body_label):
            source_path=label.split("/",1)[1]
            paths.append(f"/World/Franka_{index//self.ik.bodies_per_arm}"+source_path[len("/panda"):])
        self.rigid_links=RigidPrim(paths)
        self.rigid_link_paths=paths
        joints, ik_info = self.ik.inverse_kinematics(self.reference_pose(0.))
        if not ik_info["converged"]:
            raise RuntimeError(f"Initial IK failed: {ik_info}")
        q = self.park_q.copy()
        q[0, :7] = joints
        self.arms.set_dof_positions(q)  # reset only; never called by step()
        self.arms.set_dof_velocities(np.zeros_like(q))
        self.arms.set_dof_position_targets(q)
        self.arms.set_dof_velocity_targets(np.zeros_like(q))
        self.previous_command_q=q.copy()
        self.arms.set_dof_efforts(self.arms.get_dof_gravity_compensation_forces())
        SimulationManager.step()
        self.native_time_origin = SimulationManager.get_simulation_time()
        self.native_step_origin = SimulationManager.get_num_physics_steps()
        import omni.physx
        self.physx=omni.physx.get_physx_interface()
        self.physx.update_transformations(False,True,True)
        self.time_s, self.step_index = 0., 0
        initial = self.proprioception()
        attachment=self._check_attachment(initial["tool_pose"])
        material_config=CouplingConfig(control_dt=c.control_dt, material_dt=c.material_dt,
                       dx=c.dx, device=c.device, feedback=c.feedback)
        self.material = (LiveCuttingMaterial(material_config,task=self.task) if self.task else LiveCuttingMaterial(material_config))
        material_setup = self.material.reset(initial["tool_pose"], c.seed)
        spacing=self.task.particle_spacing_m if self.task else c.dx/2
        voxel=spacing/2 if c.surface_profile=="current_volume_v1" else min(spacing,.0015)
        self.surface = ComponentSurface(spacing, c.device, voxel_size=voxel,profile=c.surface_profile)
        self.bridge = DynamicSurfaceBridge(stage)
        self._update_surface()
        self._create_cameras()
        setup = {"config":asdict(c), "source_scene":str(source), "initial_ik":ik_info,
            "material":material_setup["material"],"material_setup":material_setup,
            "initial_mass_kg":material_setup["initial_mass_kg"],
            "world_from_workpiece_xyzw":material_setup["world_from_workpiece_xyzw"],
            "T_hand_edge_xyz_xyzw":self.mount.tolist(),
            "blade_dimensions_m":dict(zip(("length","height","half_width"),blade_dimensions)),
            "task":self.task.to_dict() if self.task else None,
            "cutting_hand_body_label":self.ik.model.body_label[self.ik.hand_local],
            "cutting_hand_body_index":self.ik.hand_local,"physical_knife_body_index":self.ik.hand_local,
            "joint_names":[self.arms.dof_names[:7],self.arms.dof_names[:7]],
            "robot_physics":"Isaac Sim6.2 native PhysX CPU articulation",
            "ik":"Newton analytic IK only; zero standalone robot dynamics steps",
            "native_time_origin_s":self.native_time_origin, "native_step_origin":self.native_step_origin,
            "initial_rendered_attachment":attachment,
            "stationary_camera_kinematic_rigid_prims":camera_rigid_paths,
            "knife_prim":knife_path, "hand_prim":hand_paths[0], "hand_prims":hand_paths,
            "camera_layout":c.camera_layout,"camera_rig":getattr(self,"camera_rig_receipt",None),
            "observation_motion_probe":c.observation_motion_probe,
            "dof_names":self.arms.dof_names,
            "feedback_delay_s":c.control_dt, "proprioception_source":"actual PhysX tensors"}
        write_json(self.output/("setup.json" if self.reset_index==0 else f"reset_setup_{self.reset_index:02d}.json"), setup)
        return initial

    def _author_task_fixture(self):
        from pxr import Gf, UsdGeom, UsdPhysics
        task,stage=self.task,self.stage
        for prim in list(stage.Traverse()):
            if str(prim.GetPath()).startswith("/World/Tray"):
                prim.SetActive(False)
        root=UsdGeom.Xform.Define(stage,"/World/TaskFixture")
        pose=np.asarray(task.world_from_workpiece_xyzw,float)
        root.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
        root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(pose[6]),Gf.Vec3d(*pose[3:6])))
        for index,box in enumerate(task.fixture_boxes_local):
            cube=UsdGeom.Cube.Define(stage,f"/World/TaskFixture/Box_{index}")
            cube.CreateSizeAttr(2.)
            cube.GetPrim().SetCustomDataByKey("research:fixture_name",box.name)
            cube.AddTranslateOp().Set(Gf.Vec3d(*box.center_m))
            cube.AddScaleOp().Set(Gf.Vec3f(*box.half_extents_m))
            cube.CreateDisplayColorAttr([Gf.Vec3f(.22,.27,.31)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        table=UsdGeom.Xformable(stage.GetPrimAtPath("/World/Table"))
        for op in table.GetOrderedXformOps():
            if op.GetOpType()==UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(0,0,task.table_top_world_z_m-.025))

    def body_poses(self):
        positions,orientations=self.rigid_links.get_world_poses()
        p,r=numpy_data(positions),numpy_data(orientations)
        return np.asarray([pose_from_wxyz(position,orientation) for position,orientation in zip(p,r)])

    def proprioception(self):
        q = numpy_data(self.arms.get_dof_positions())
        qdot = numpy_data(self.arms.get_dof_velocities())
        p, r = self.hands.get_world_poses()
        hand_both = np.asarray([pose_from_wxyz(position,orientation)
                               for position,orientation in zip(numpy_data(p),numpy_data(r))])
        if hand_both.shape != (2,7):
            raise ValueError("Expected both actual native PhysX hand poses")
        hand = hand_both[0]
        tool = self.compose_tool_pose(hand, self.mount[:3], self.mount[3:])
        if not np.isfinite(np.r_[q.ravel(),qdot.ravel(),tool]).all():
            raise FloatingPointError("Nonfinite native PhysX state")
        return {"q":q[:,:7].copy(), "qdot":qdot[:,:7].copy(), "hand_pose":hand,
                "hand_pose_both_xyzw":hand_both,
                "tool_pose":tool, "timestamp_s":self.time_s}

    def _update_surface(self):
        vertices, faces, self.surface_info = self.surface.extract(self.material.bed)
        if self.task:
            vertices=self.task.points_to_world(vertices).astype(np.float32)
        self.surface_receipt = self.bridge.update(vertices, faces, self.time_s)
        self.surface_arrays = (vertices, faces)

    def _check_attachment(self, measured_edge):
        from pxr import Usd, UsdGeom
        from scipy.spatial.transform import Rotation
        transform=np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(self.knife_path)).ComputeLocalToWorldTransform(
                             Usd.TimeCode.Default())).T
        translation=float(np.linalg.norm(transform[:3,3]-measured_edge[:3]))
        rotation=float((Rotation.from_matrix(transform[:3,:3]).inv()*Rotation.from_quat(measured_edge[3:])).magnitude())
        if translation>1e-5 or rotation>1e-4:
            raise RuntimeError(f"Native rendered knife differs from measured physical edge: {translation}m/{rotation}rad")
        return {"translation_error_m":translation,"rotation_error_rad":rotation}

    def _create_cameras(self):
        import omni.replicator.core as rep
        import omni.kit.app
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera, SingleViewDepthCameraSensor
        from .usd_surface_bridge import usd_camera_calibration
        self.cameras, self.calibration, self.camera_params, self.native_rgb, self.native_xyz = {}, {}, {}, {}, {}
        self.native_parameters={}
        self.native_registration={}
        kit=omni.kit.app.get_app()
        manager=kit.get_extension_manager()
        extension=manager.get_extension_dict(manager.get_enabled_extension_id("omni.sensors.nv.camera"))
        self.sensor_runtime={"isaacsim_version":runtime_versions()["isaacsim"],"kit_version":kit.get_kit_version(),
             "camera_extension_version":extension["package"]["version"]}
        c = self.config
        paths = {"d455_0":"/World/D455_0/RSD455/Camera_Pseudo_Depth",
                 "d455_1":"/World/D455_1/RSD455/Camera_Pseudo_Depth", "global":"/World/GlobalCamera"}
        self.camera_specs={}
        self.camera_rig_receipt=None
        self.camera_extrinsics={}
        self.camera_pose_trace=[]
        if c.camera_layout == "wrist_d405_global_d455":
            from .camera_rig import author_camera_rig
            from .workcell import ASSETS
            asset=ROOT/"artifacts/workcell_asset_cache"/ASSETS["d455"]
            if not asset.is_file():
                raise FileNotFoundError(f"The official Isaac6.2 D455 asset is required: {asset}")
            self.camera_specs=author_camera_rig(self.stage,self.hand_paths,asset,
                task_id=c.task_id or "TRIM-01",camera_layout=c.camera_layout,
                initial_hand_poses_world_xyzw=self.proprioception()["hand_pose_both_xyzw"])
            paths={name:spec["optical_prim_path"] for name,spec in self.camera_specs.items()}
            if set(paths)!={"wrist_d405_0","wrist_d405_1","global_d455"}:
                raise ValueError("The new native rig must expose both wrist D405 IDs and global_d455")
            from pxr import Gf,UsdGeom
            for path in paths.values():
                # Capture the actual nearest occluder before applying the model
                # range. Clipping at7cm would falsely reveal material behind a
                # closer finger or bracket.
                UsdGeom.Camera.Get(self.stage,path).GetClippingRangeAttr().Set(Gf.Vec2f(.001,20.))
            from .native_camera_geometry import camera_physics_receipt
            physics=camera_physics_receipt(self.stage,self.camera_specs,self.hand_paths)
            self.camera_rig_receipt={"camera_layout":c.camera_layout,"specs":self.camera_specs,
                "camera_physics":physics,
                "official_D455_asset":str(asset),"secondary_joint_hold_rad":self.park_q[1,:7].tolist(),
                "D405_model":"ideal_d405_prototype; actual RTX geometry, unvalidated physical sensor model",
                "native_D455_model":"Controlled SingleView profile; physical calibration unvalidated"}
            write_json(self.output/"camera_rig.json",self.camera_rig_receipt)
        native_ids=({"global_d455"} if c.camera_layout=="wrist_d405_global_d455" else
                    {"d455_0","d455_1"} if c.camera_mode=="native_single_view" else set())
        # Native assets contain same-named pre-authored2048x1080 products. In
        #6.2 their Hydra texture adoption can alias the two camera views. Create
        #fresh uniquely named live products at the explicitly requested sizes.
        for prim in list(self.stage.Traverse()):
            if prim.GetTypeName()=="RenderProduct" and (c.camera_layout=="wrist_d405_global_d455" or
                    any(str(prim.GetPath()).startswith(f"/World/D455_{i}/") for i in (0,1))):
                prim.SetActive(False)
        for name, path in paths.items():
            scale=c.global_resolution_scale if name == "global" else 1
            height,width=c.height*scale,c.width*scale
            if self.camera_specs:
                width,height=int(self.camera_specs[name]["width"]),int(self.camera_specs[name]["height"])
            if name in native_ids:
                sensor = SingleViewDepthCameraSensor(RtxCamera(path,tick_rate=0.), resolution=(height,width),
                         annotators=["distance_to_image_plane","depth_sensor_distance"])
                sensor.set_sensor_baseline(95.)
                sensor.set_sensor_focal_length(317.0431137)
                sensor.set_sensor_size(640.)
                sensor.set_sensor_maximum_disparity(150.)
                sensor.set_sensor_disparity_confidence(1.)
                sensor.set_sensor_distance_cutoffs(.52,6.)
                noise=0. if c.native_noise=="off" else .25
                sensor.set_sensor_noise_parameters(noise,noise)
                sensor.set_sensor_disparity_noise_downscale(1.)
                sensor.set_enabled_outlier_removal(True)
                sensor.set_sensor_output_mode(0)
                sensor.set_enabled_post_processing(True)
                from pxr import Sdf
                sensor.render_product.GetPrim().CreateAttribute("omni:rtx:post:depthSensor:effectiveDistance",Sdf.ValueTypeNames.Float).Set(4.)
                prefix="omni:rtx:post:depthSensor:"
                self.native_parameters[name]={str(a.GetName())[len(prefix):]:a.Get()
                    for a in sensor.render_product.GetPrim().GetAttributes() if str(a.GetName()).startswith(prefix)}
                self.cameras[name] = sensor
                self.native_rgb[name]=rep.AnnotatorRegistry.get_annotator("rgb",device="cpu")
                self.native_xyz[name]=rep.AnnotatorRegistry.get_annotator("DepthSensorPointCloudPosition",device="cpu")
                for annotator in (self.native_rgb[name],self.native_xyz[name]):
                    annotator.attach(str(sensor.render_product.GetPath()))
            else:
                self.cameras[name] = CameraSensor(RtxCamera(path,tick_rate=0.), resolution=(height,width),
                            annotators=["rgb","distance_to_image_plane"])
                if self.camera_specs:
                    intrinsic=np.asarray(self.camera_specs[name]["K"],float)
                    self.cameras[name].render_product.CreatePixelAspectRatioAttr(float(intrinsic[1,1]/intrinsic[0,0]))
            self.calibration[name] = usd_camera_calibration(self.stage,path,width,height)
            if self.camera_specs:
                spec=self.camera_specs[name]
                self.calibration[name].update(sensor_model=spec["sensor_model"],near_m=spec["near_m"],far_m=spec["far_m"],
                    camera_id=name,camera_layout=c.camera_layout,capability=spec.get("capability"))
            self.camera_params[name] = rep.AnnotatorRegistry.get_annotator("camera_params")
            self.camera_params[name].attach(str(self.cameras[name].render_product.GetPath()))
        write_json(self.output/"calibration_authored.json",self.calibration)
        write_json(self.output/"native_depth_model_receipt.json",{"runtime":self.sensor_runtime,"parameters":self.native_parameters})
        self.authored_calibration={name:dict(value) for name,value in self.calibration.items()}
        if self.camera_specs:
            self.fixed_world_optical={name:value["T_world_camera"] for name,value in self.calibration.items()
                                      if self.camera_specs[name].get("parent_hand_index") is None}
            self._update_camera_extrinsics(self.proprioception()["hand_pose_both_xyzw"])
        write_json(self.output/"camera_products.json",{name:{"product":str(sensor.render_product.GetPath()),
            "camera_targets":[str(p) for p in sensor.render_product.GetCameraRel().GetTargets()],
            "pixel_aspect_ratio":float(sensor.render_product.GetPixelAspectRatioAttr().Get()),
            "resolution":[int(x) for x in sensor.render_product.GetResolutionAttr().Get()]}
            for name,sensor in self.cameras.items()})

    def _update_camera_extrinsics(self, measured_hand_poses):
        """Refresh optical extrinsics after every actual20ms articulation step."""
        import json
        from .native_camera_geometry import measured_camera_extrinsics
        from .usd_surface_bridge import usd_camera_calibration
        current={name:usd_camera_calibration(self.stage,spec["optical_prim_path"],spec["width"],spec["height"])["T_world_camera"]
                 for name,spec in self.camera_specs.items()}
        sample=measured_camera_extrinsics(self.camera_specs,measured_hand_poses,current,self.fixed_world_optical,
                                         self.time_s,self.SimulationManager.get_simulation_time())
        for name,pose in sample["cameras"].items():
            self.calibration[name]={**self.calibration[name],**pose}
        self.camera_extrinsics=sample
        self.camera_pose_trace.append(sample)
        suffix="" if self.reset_index==0 else f"_reset_{self.reset_index:02d}"
        with (self.output/f"camera_extrinsics{suffix}.jsonl").open("a") as stream:
            stream.write(json.dumps(sample,allow_nan=False)+"\n")

    def step(self,action=None,*,cutting=True):
        c, manager = self.config, self.SimulationManager
        if c.observation_motion_probe and action is not None:
            raise ValueError("The separate observation motion probe uses scripted actions only")
        mode=c.action_mode
        if isinstance(action,dict):
            if self.task is None:
                raise ValueError("Task action mappings require a shared CUT/TRIM specification")
            command=self.task.validate_action(action)
            mode="cartesian" if command["mode"]=="cartesian_edge_pose_world" else "joint"
            action=command["target"]
            cutting=bool(cutting and command["cut_enabled"])
        desired = self.reference_pose(self.time_s+c.control_dt)
        # Seed IK from current PHYSX joints, without advancing Newton dynamics.
        q = numpy_data(self.arms.get_dof_positions())
        self.ik.state.joint_q.assign(q.ravel().astype(np.float32))
        if action is None or mode=="cartesian":
            if action is not None:
                desired=validate_native_action(action,"cartesian")
            targets, ik_info = self.ik.inverse_kinematics(desired)
            if not ik_info["converged"]:
                raise RuntimeError(f"IK failed before native articulation step: {ik_info}")
        else:
            target=validate_native_action(action,"joint")
            if self.task and target.ndim==2 and not np.allclose(target[1],self.park_q[1,:7],atol=1e-6,rtol=0):
                raise ValueError("CUT/TRIM keep the secondary Franka at its declared park target")
            targets=(np.stack([self.ik._validate_joints(row) for row in target]) if target.ndim==2
                     else self.ik._validate_joints(target))
            desired=None
            ik_info={"used":False,"reason":"explicit native joint target action"}
        target_q = self.park_q.copy()
        if np.asarray(targets).ndim==2:
            target_q[:,:7]=targets
        else:
            target_q[0,:7]=targets
        if c.observation_motion_probe:
            # Actual articulation motion for a separate camera calibration probe;
            # disabled in all main physical task baselines.
            target_q[1,0] += .08*np.sin(2*np.pi*(self.time_s+c.control_dt)/2.)
        self.arms.set_dof_position_targets(target_q)
        velocity_target=(target_q-self.previous_command_q)/c.control_dt if c.velocity_feedforward else np.zeros_like(target_q)
        self.arms.set_dof_velocity_targets(velocity_target)
        self.previous_command_q=target_q.copy()
        applied = self.material.pending_wrench.copy() if c.feedback else np.zeros(6)
        start = time.perf_counter()
        for _ in range(c.robot_substeps):
            current = self.proprioception()
            self.hand.apply_forces_and_torques_at_pos(applied[None,:3],applied[None,3:],
                         positions=current["tool_pose"][None,:3],local_frame=False)
            self.arms.set_dof_efforts(self.arms.get_dof_gravity_compensation_forces())
            manager.step()
        self.step_index += 1
        self.time_s = self.step_index*c.control_dt
        native_relative = manager.get_simulation_time()-self.native_time_origin
        if abs(native_relative-self.time_s) > 1e-6:
            raise RuntimeError(f"Native physics clock drift: {native_relative} versus{self.time_s}")
        self.physx.update_transformations(False,True,True)
        actual = self.proprioception()
        attachment=self._check_attachment(actual["tool_pose"])
        diagnostics = self.material.advance(actual["tool_pose"],self.time_s,cutting=cutting,
                           cut_index=0 if action is not None else None)
        if not np.array_equal(diagnostics["applied_previous_material_wrench_world"],applied):
            raise AssertionError("The material receipt does not match the wrench actually applied to PhysX")
        state=self.material.debug_state()
        if not all(np.isfinite(state[name]).all() for name in ("positions","velocities","F","plastic_strain")):
            raise FloatingPointError("Nonfinite live TLMPM state")
        determinant=np.linalg.det(state["F"])
        diagnostics.update(minimum_detF=float(determinant.min()),
                           maximum_material_speed_m_s=float(np.linalg.norm(state["velocities"],axis=1).max()),
                           maximum_plastic_strain=float(state["plastic_strain"].max()))
        self._update_surface()
        if c.camera_layout == "wrist_d405_global_d455":
            self._update_camera_extrinsics(actual["hand_pose_both_xyzw"])
        return actual, {**diagnostics,"commanded_tool_pose":desired,"desired_tool_pose":desired,
                 "commanded_joint_positions_rad":target_q[:,:7].copy(),"action_mode":mode,
                 "commanded_joint_velocity_rad_s":velocity_target,"velocity_feedforward":c.velocity_feedforward,
                 "ik":ik_info,"rendered_attachment":attachment,
                 "robot_physics_wall_s":time.perf_counter()-start,
                 "native_physics_timestamp_s":manager.get_simulation_time(),
                 "native_relative_timestamp_s":native_relative,
                 "relative_native_physics_steps":manager.get_num_physics_steps()-self.native_step_origin}

    def capture(self):
        import omni.replicator.core as rep
        manager = self.SimulationManager
        before = (manager.get_simulation_time(),manager.get_num_physics_steps())
        robot_before=self.proprioception()
        latest = {}
        def data(name,sensor,annotator):
            if name in self.native_rgb and annotator in ("rgb","depth_sensor_point_cloud_position"):
                source=self.native_rgb[name] if annotator=="rgb" else self.native_xyz[name]
                value=source.get_data()
                return (value.get("data"),value.get("info",{})) if isinstance(value,dict) else (value,{})
            return sensor.get_data(annotator)
        for _ in range(self.config.render_warmup+30):
            rep.orchestrator.step(rt_subframes=1,delta_time=0.)
            latest = {name:{a:data(name,sensor,a) for a in
                       (["rgb","distance_to_image_plane","depth_sensor_distance","depth_sensor_point_cloud_position"]
                       if name in self.native_parameters else ["rgb","distance_to_image_plane"])}
                      for name,sensor in self.cameras.items()}
            if _ >= self.config.render_warmup-1 and all(value[0] is not None for data in latest.values() for value in data.values()):
                break
        else:
            raise TimeoutError("Live native RTX cameras did not produce arrays")
        after = (manager.get_simulation_time(),manager.get_num_physics_steps())
        if before != after:
            raise RuntimeError(f"Rendering advanced uncontrolled physics: {before}->{after}")
        robot_after=self.proprioception()
        if not all(np.array_equal(robot_before[key],robot_after[key]) for key in ("q","qdot","tool_pose","hand_pose_both_xyzw")):
            raise RuntimeError("Rendering changed native measured robot state while physics was held")
        frames = {}
        parameters_by_name={name:annotator.get_data() for name,annotator in self.camera_params.items()}
        write_json(self.output/"camera_params_latest.json",parameters_by_name)
        self.camera_runtime_extrinsic_checks={}
        for name, payload in latest.items():
            parameters = parameters_by_name[name]
            runtime_calibration=calibration_from_replicator(parameters)
            self.calibration[name] = {**self.calibration[name],**runtime_calibration}
            expected_pose=np.asarray((self.camera_extrinsics["cameras"][name] if self.camera_specs
                                      else self.authored_calibration[name])["T_world_camera"])
            runtime_pose=np.asarray(self.calibration[name]["T_world_camera"])
            if self.camera_specs:
                self.calibration[name]["T_camera_world"]=np.linalg.inv(runtime_pose).tolist()
            if not np.allclose(runtime_pose,expected_pose,atol=1e-5,rtol=0):
                raise RuntimeError(f"Runtime camera pose does not match its own optical prim: {name}; "
                                   f"runtime={runtime_pose[:3,3]} expected={expected_pose[:3,3]}")
            if self.camera_specs:
                from .native_camera_geometry import transform_error
                from .usd_surface_bridge import usd_camera_calibration
                spec=self.camera_specs[name]
                current=usd_camera_calibration(self.stage,spec["optical_prim_path"],spec["width"],spec["height"])
                check=transform_error(current["T_world_camera"],runtime_pose)
                if check["translation_error_m"]>1e-5 or check["rotation_error_rad"]>1e-4:
                    raise RuntimeError(f"Runtime optical camera differs from current moving USD prim: {name}: {check}")
                self.camera_runtime_extrinsic_checks[name]={"runtime_vs_current_USD":check,
                    "runtime_vs_measured_hand_composition":transform_error(expected_pose,runtime_pose),
                    "physics_timestamp_s":self.time_s,"rendering_advanced_physics":False}
            rgb = numpy_data(payload["rgb"][0])
            depth = numpy_data(payload["distance_to_image_plane"][0]).squeeze()
            expected=(self.calibration[name]["height"],self.calibration[name]["width"])
            if rgb.shape[:2] != expected or depth.shape != expected:
                raise ValueError(f"Malformed native RTX arrays: {name}")
            rgb = rgb[:,:,:3].astype(np.uint8,copy=False)
            near,far=((self.camera_specs[name]["near_m"],self.camera_specs[name]["far_m"])
                      if self.camera_specs else (0.,20.))
            valid = (np.isfinite(depth)&(depth>=near)&(depth<=far)&(depth>0) if self.camera_specs
                     else np.isfinite(depth)&(depth>0)&(depth<20.))
            frame = {"rgb":rgb,"depth_m":np.where(valid,depth,0).astype(np.float32),"valid_mask":valid,
                "physics_timestamp_s":self.time_s,"surface_timestamp_s":self.time_s,
                "native_physics_timestamp_s":before[0],"live_source":"Isaac Sim6.2 Kit/PhysX/RTX"}
            if self.camera_specs:
                raw_valid=np.isfinite(depth)&(depth>0)&(depth<20.)
                frame["rtx_reference_distance_to_image_plane_raw"]=depth.copy()
                frame["rtx_reference_depth_m"]=np.where(raw_valid,depth,0).astype(np.float32)
                frame["rtx_reference_valid_mask"]=raw_valid
                frame["rtx_reference_range_rejected_mask"]=raw_valid&~valid
            if name in self.native_parameters:
                from .single_view_registration import register_kit_single_view_depth
                frame["depth_sensor_distance_raw"] = numpy_data(payload["depth_sensor_distance"][0])
                frame["depth_sensor_point_cloud_position_raw"] = numpy_data(payload["depth_sensor_point_cloud_position"][0]).reshape(*expected,-1)
                if not self.camera_specs:
                    frame["rtx_reference_depth_m"] = frame.pop("depth_m")
                    frame["rtx_reference_valid_mask"] = frame.pop("valid_mask")
                raw = frame["depth_sensor_distance_raw"]
                frame["native_raw_valid_mask"] = np.isfinite(raw)&(raw>=.52)&(raw<=6)
                registered=register_kit_single_view_depth(frame["depth_sensor_point_cloud_position_raw"],raw,
                            self.calibration[name],**self.sensor_runtime,native_model_parameters=self.native_parameters[name])
                frame["depth_m"]=registered["depth_m"]
                frame["valid_mask"]=registered["valid_mask"]
                if self.camera_specs:
                    frame["depth_sensor_registered_depth_before_range_raw"]=frame["depth_m"].copy()
                    frame["depth_sensor_registered_valid_mask_before_range_raw"]=frame["valid_mask"].copy()
                    accepted=frame["valid_mask"]&(frame["depth_m"]>=near)&(frame["depth_m"]<=far)
                    frame["depth_m"]=np.where(accepted,frame["depth_m"],0).astype(np.float32)
                    frame["valid_mask"]=accepted
                frame["native_depth_rgb_registered"]=True
                frame["native_source_index_debug"]=registered["source_index"]
                self.native_registration[name]=registered["diagnostics"]
            if self.camera_specs:
                calibration=self.calibration[name]
                frame.update(camera_id=name,sensor_model=self.camera_specs[name]["sensor_model"],
                    K=np.asarray(calibration["K"]),T_world_camera=np.asarray(calibration["T_world_camera"]),
                    T_camera_world=np.linalg.inv(np.asarray(calibration["T_world_camera"])),
                    camera_extrinsics_timestamp_s=self.camera_extrinsics["timestamp_s"],
                    prototype_sensor_model=name.startswith("wrist_d405"),
                    physical_sensor_calibration_validated=False)
                pose=self.camera_extrinsics["cameras"][name]
                if pose["T_world_hand"] is not None:
                    frame.update(T_world_hand=np.asarray(pose["T_world_hand"]),T_hand_camera=np.asarray(pose["T_hand_camera"]))
            frames[name] = frame
        write_json(self.output/"calibration_runtime.json",self.calibration)
        write_json(self.output/"native_registration_latest.json",self.native_registration)
        return frames,{name:{a:str(value[1]) for a,value in payload.items()} for name,payload in latest.items()}

    def close(self):
        if not self._closed:
            self.app.close(wait_for_replicator=False)
            self._closed=True


def launch_app(config, output):
    import sys
    require_sim_62(runtime_versions())
    from isaacsim import SimulationApp
    argv=sys.argv
    try:
        sys.argv=argv[:1]  # Project arguments are already parsed; Kit receives explicit options only.
        return SimulationApp({"headless":True,"active_gpu":config.gpu,"physics_gpu":config.gpu,
            "multi_gpu":False,"width":config.width,"height":config.height,"renderer":"RaytracedLighting",
            "enable_crashreporter":False,"extra_args":[f"--/log/file={Path(output).resolve()/'kit.log'}",
                         "--ext-folder",str(ROOT/".deps/isaacsim-extension-cache")]},
            experience=str(ROOT/"configs/isaac_sim_cutting.kit"))
    finally:
        sys.argv=argv
