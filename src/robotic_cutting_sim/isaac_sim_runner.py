"""Live Isaac Sim 6.2 PhysX / retained TLMPM / RTX integration.

Isaac modules are imported only after SimulationApp starts. The Newton robot
object supplies IK targets only: its dynamics step methods are never called.
All published robot poses come from native PhysX tensors. No replay input exists.
"""
from dataclasses import asdict, dataclass
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
    device: str = "cuda:1"
    gpu: int = 1
    width: int = 640
    height: int = 480
    global_resolution_scale: int = 2
    record_every: int = 10
    render_warmup: int = 5
    camera_mode: str = "rtx_rgbd"
    feedback: bool = True
    scene: str | None = None

    def __post_init__(self):
        if min(self.steps, self.robot_substeps, self.width, self.height, self.global_resolution_scale, self.record_every, self.render_warmup) < 1:
            raise ValueError("Step counts, resolution and recording intervals must be positive")
        values = np.asarray([self.control_dt, self.material_dt, self.dx])
        if not np.isfinite(values).all() or np.min(values) <= 0:
            raise ValueError("Spatial and time steps must be finite and positive")
        if abs(round(self.control_dt/self.material_dt)*self.material_dt-self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of material_dt")
        if self.camera_mode not in ("rtx_rgbd", "native_single_view"):
            raise ValueError("Choose rtx_rgbd or native_single_view; no ideal fallback is available")


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

    def reset(self):
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

        c = self.config
        self.reset_index += 1
        self.SimulationManager = SimulationManager
        self.compose_tool_pose = compose_tool_pose
        self.reference_pose = reference_pose
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
        vertices, faces = wedge_mesh()
        knife.CreatePointsAttr(vertices)
        knife.CreateFaceVertexCountsAttr([3]*len(faces))
        knife.CreateFaceVertexIndicesAttr(faces.ravel())
        knife.CreateSubdivisionSchemeAttr("none")
        knife.CreateDisplayColorAttr([Gf.Vec3f(.38, .65, .85)])
        xf = UsdGeom.Xformable(knife)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, .18))
        xf.AddOrientOp().Set(Gf.Quatf(0, Gf.Vec3f(1, 0, 0)))
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
        SimulationManager.initialize_physics()
        if not self.arms.is_physics_tensor_entity_valid() or not self.hand.is_physics_tensor_entity_valid():
            raise RuntimeError("Actual PhysX articulation/hand tensor views did not initialize")
        expected = [f"panda_joint{i}" for i in range(1, 8)]+["panda_finger_joint1", "panda_finger_joint2"]
        if self.arms.dof_names != expected:
            raise RuntimeError(f"Unexpected PhysX DOF order: {self.arms.dof_names}")
        self.arms.set_dof_gains([1000.]*4+[600.]*3+[350.,350.], [80.]*4+[40.]*3+[10.,10.])
        self.arms.set_dof_max_efforts([87.]*4+[12.]*3+[20.,20.])
        self.arms.set_solver_iteration_counts(position_counts=16, velocity_counts=4)
        self.ik = DualFrankaController(RobotConfig(control_dt=c.control_dt, tool_shape="cut_wedge", tool_rotation_xyzw=(1.,0.,0.,0.)))
        paths=[]
        for index,label in enumerate(self.ik.model.body_label):
            source_path=label.split("/",1)[1]
            paths.append(f"/World/Franka_{index//self.ik.bodies_per_arm}"+source_path[len("/panda"):])
        self.rigid_links=RigidPrim(paths)
        self.rigid_link_paths=paths
        joints, ik_info = self.ik.inverse_kinematics(reference_pose(0.))
        if not ik_info["converged"]:
            raise RuntimeError(f"Initial IK failed: {ik_info}")
        q = self.ik.initial_q.reshape(2, 9).copy()
        q[0, :7] = joints
        self.arms.set_dof_positions(q)  # reset only; never called by step()
        self.arms.set_dof_velocities(np.zeros_like(q))
        self.arms.set_dof_position_targets(q)
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
        self.material = LiveCuttingMaterial(CouplingConfig(control_dt=c.control_dt, material_dt=c.material_dt,
                       dx=c.dx, device=c.device, feedback=c.feedback))
        material_setup = self.material.reset(initial["tool_pose"], c.seed)
        self.surface = ComponentSurface(c.dx/2, c.device, voxel_size=min(c.dx/2, .0015))
        self.bridge = DynamicSurfaceBridge(stage)
        self._update_surface()
        self._create_cameras()
        setup = {"config":asdict(c), "source_scene":str(source), "initial_ik":ik_info,
            "material":material_setup["material"],"material_setup":material_setup,
            "initial_mass_kg":material_setup["initial_mass_kg"],
            "world_from_workpiece_xyzw":material_setup["world_from_workpiece_xyzw"],
            "T_hand_edge_xyz_xyzw":[0.,0.,.18,1.,0.,0.,0.],
            "cutting_hand_body_label":self.ik.model.body_label[self.ik.hand_local],
            "cutting_hand_body_index":self.ik.hand_local,"physical_knife_body_index":self.ik.hand_local,
            "joint_names":[self.arms.dof_names[:7],self.arms.dof_names[:7]],
            "robot_physics":"Isaac Sim6.2 native PhysX CPU articulation",
            "ik":"Newton analytic IK only; zero standalone robot dynamics steps",
            "native_time_origin_s":self.native_time_origin, "native_step_origin":self.native_step_origin,
            "initial_rendered_attachment":attachment,
            "stationary_camera_kinematic_rigid_prims":camera_rigid_paths,
            "knife_prim":knife_path, "hand_prim":hand_paths[0], "dof_names":self.arms.dof_names,
            "feedback_delay_s":c.control_dt, "proprioception_source":"actual PhysX tensors"}
        write_json(self.output/("setup.json" if self.reset_index==0 else f"reset_setup_{self.reset_index:02d}.json"), setup)
        return initial

    def body_poses(self):
        positions,orientations=self.rigid_links.get_world_poses()
        p,r=numpy_data(positions),numpy_data(orientations)
        return np.asarray([pose_from_wxyz(position,orientation) for position,orientation in zip(p,r)])

    def proprioception(self):
        q = numpy_data(self.arms.get_dof_positions())
        qdot = numpy_data(self.arms.get_dof_velocities())
        p, r = self.hand.get_world_poses()
        hand = pose_from_wxyz(numpy_data(p)[0], numpy_data(r)[0])
        tool = self.compose_tool_pose(hand, (0.,0.,.18), (1.,0.,0.,0.))
        if not np.isfinite(np.r_[q.ravel(),qdot.ravel(),tool]).all():
            raise FloatingPointError("Nonfinite native PhysX state")
        return {"q":q[:,:7].copy(), "qdot":qdot[:,:7].copy(), "hand_pose":hand,
                "tool_pose":tool, "timestamp_s":self.time_s}

    def _update_surface(self):
        vertices, faces, self.surface_info = self.surface.extract(self.material.bed)
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
        # Native assets contain same-named pre-authored2048x1080 products. In
        #6.2 their Hydra texture adoption can alias the two camera views. Create
        #fresh uniquely named live products at the explicitly requested sizes.
        for prim in list(self.stage.Traverse()):
            if prim.GetTypeName()=="RenderProduct" and any(str(prim.GetPath()).startswith(f"/World/D455_{i}/") for i in (0,1)):
                prim.SetActive(False)
        for name, path in paths.items():
            scale=c.global_resolution_scale if name == "global" else 1
            height,width=c.height*scale,c.width*scale
            if name != "global" and c.camera_mode == "native_single_view":
                sensor = SingleViewDepthCameraSensor(RtxCamera(path,tick_rate=0.), resolution=(height,width),
                         annotators=["distance_to_image_plane","depth_sensor_distance"])
                sensor.set_sensor_baseline(95.)
                sensor.set_sensor_focal_length(317.0431137)
                sensor.set_sensor_size(640.)
                sensor.set_sensor_maximum_disparity(150.)
                sensor.set_sensor_disparity_confidence(1.)
                sensor.set_sensor_distance_cutoffs(.52,6.)
                sensor.set_sensor_noise_parameters(.25,.25)
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
            self.calibration[name] = usd_camera_calibration(self.stage,path,width,height)
            self.camera_params[name] = rep.AnnotatorRegistry.get_annotator("camera_params")
            self.camera_params[name].attach(str(self.cameras[name].render_product.GetPath()))
        write_json(self.output/"calibration_authored.json",self.calibration)
        write_json(self.output/"native_depth_model_receipt.json",{"runtime":self.sensor_runtime,"parameters":self.native_parameters})
        self.authored_calibration={name:dict(value) for name,value in self.calibration.items()}
        write_json(self.output/"camera_products.json",{name:{"product":str(sensor.render_product.GetPath()),
            "camera_targets":[str(p) for p in sensor.render_product.GetCameraRel().GetTargets()],
            "resolution":[int(x) for x in sensor.render_product.GetResolutionAttr().Get()]}
            for name,sensor in self.cameras.items()})

    def step(self):
        c, manager = self.config, self.SimulationManager
        desired = self.reference_pose(self.time_s+c.control_dt)
        # Seed IK from current PHYSX joints, without advancing Newton dynamics.
        q = numpy_data(self.arms.get_dof_positions())
        self.ik.state.joint_q.assign(q.ravel().astype(np.float32))
        targets, ik_info = self.ik.inverse_kinematics(desired)
        if not ik_info["converged"]:
            raise RuntimeError(f"IK failed before native articulation step: {ik_info}")
        target_q = self.ik.initial_q.reshape(2,9).copy()
        target_q[0,:7] = targets
        self.arms.set_dof_position_targets(target_q)
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
        diagnostics = self.material.advance(actual["tool_pose"],self.time_s)
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
        return actual, {**diagnostics,"commanded_tool_pose":desired,"desired_tool_pose":desired,"ik":ik_info,"rendered_attachment":attachment,
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
                       if name != "global" and self.config.camera_mode == "native_single_view" else ["rgb","distance_to_image_plane"])}
                      for name,sensor in self.cameras.items()}
            if _ >= self.config.render_warmup-1 and all(value[0] is not None for data in latest.values() for value in data.values()):
                break
        else:
            raise TimeoutError("Live native RTX cameras did not produce arrays")
        after = (manager.get_simulation_time(),manager.get_num_physics_steps())
        if before != after:
            raise RuntimeError(f"Rendering advanced uncontrolled physics: {before}->{after}")
        robot_after=self.proprioception()
        if not all(np.array_equal(robot_before[key],robot_after[key]) for key in ("q","qdot","tool_pose")):
            raise RuntimeError("Rendering changed native measured robot state while physics was held")
        frames = {}
        parameters_by_name={name:annotator.get_data() for name,annotator in self.camera_params.items()}
        write_json(self.output/"camera_params_latest.json",parameters_by_name)
        for name, payload in latest.items():
            parameters = parameters_by_name[name]
            self.calibration[name] = calibration_from_replicator(parameters)
            expected_pose=np.asarray(self.authored_calibration[name]["T_world_camera"])
            runtime_pose=np.asarray(self.calibration[name]["T_world_camera"])
            if not np.allclose(runtime_pose,expected_pose,atol=1e-5,rtol=0):
                raise RuntimeError(f"Runtime camera pose does not match its own optical prim: {name}; "
                                   f"runtime={runtime_pose[:3,3]} expected={expected_pose[:3,3]}")
            rgb = numpy_data(payload["rgb"][0])
            depth = numpy_data(payload["distance_to_image_plane"][0]).squeeze()
            scale=self.config.global_resolution_scale if name=="global" else 1
            expected=(self.config.height*scale,self.config.width*scale)
            if rgb.shape[:2] != expected or depth.shape != expected:
                raise ValueError(f"Malformed native RTX arrays: {name}")
            rgb = rgb[:,:,:3].astype(np.uint8,copy=False)
            valid = np.isfinite(depth)&(depth>0)&(depth<20)
            frame = {"rgb":rgb,"depth_m":np.where(valid,depth,0).astype(np.float32),"valid_mask":valid,
                "physics_timestamp_s":self.time_s,"surface_timestamp_s":self.time_s,
                "native_physics_timestamp_s":before[0],"live_source":"Isaac Sim6.2 Kit/PhysX/RTX"}
            if self.config.camera_mode == "native_single_view" and name != "global":
                from .single_view_registration import register_kit_single_view_depth
                frame["depth_sensor_distance_raw"] = numpy_data(payload["depth_sensor_distance"][0])
                frame["depth_sensor_point_cloud_position_raw"] = numpy_data(payload["depth_sensor_point_cloud_position"][0]).reshape(*expected,-1)
                frame["rtx_reference_depth_m"] = frame.pop("depth_m")
                frame["rtx_reference_valid_mask"] = frame.pop("valid_mask")
                raw = frame["depth_sensor_distance_raw"]
                frame["native_raw_valid_mask"] = np.isfinite(raw)&(raw>=.52)&(raw<=6)
                registered=register_kit_single_view_depth(frame["depth_sensor_point_cloud_position_raw"],raw,
                            self.calibration[name],**self.sensor_runtime,native_model_parameters=self.native_parameters[name])
                frame["depth_m"]=registered["depth_m"]
                frame["valid_mask"]=registered["valid_mask"]
                frame["native_depth_rgb_registered"]=True
                frame["native_source_index_debug"]=registered["source_index"]
                self.native_registration[name]=registered["diagnostics"]
            frames[name] = frame
        write_json(self.output/"calibration_runtime.json",self.calibration)
        write_json(self.output/"native_registration_latest.json",self.native_registration)
        return frames,{name:{a:str(value[1]) for a,value in payload.items()} for name,payload in latest.items()}


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
