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


def wait_for_workcell_stage(app, context, receipt_path, timeout_s=300.):
    """Allow cold shader/asset work time without a host-speed-dependent tick cap."""
    start, updates = time.monotonic(), 0
    while True:
        app.update()
        updates += 1
        status = list(context.get_stage_loading_status())
        elapsed = time.monotonic()-start
        loaded = status[2] == 0
        if loaded or elapsed >= timeout_s:
            write_json(receipt_path, {"status": "PASS_STAGE_LOADED" if loaded else "BLOCKED_STAGE_LOAD_TIMEOUT",
                "wall_s": elapsed, "kit_updates": updates, "loading_status": status,
                "timeout_s": timeout_s, "physics_or_camera_smoke": "NOT_RUN_BY_THIS_CHECK"})
            if not loaded:
                raise TimeoutError(f"Workcell loading exceeded {timeout_s}s; loading status={status}")
            return
        time.sleep(.01)


STAGE_RESET_MODES = ("reopen", "reload_in_place")


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
    tool_attachment_mode: str = "legacy_fixed_tool"
    tool_init_mode: str = "pregrasped"
    grasp: object | None = None
    tool_spec: object | None = None
    workpiece_kind: str = "rectangle"
    camera: bool = True
    overview_camera: bool = False  # presentation-only RGB /World/PresentationCamera/{Overview,Closeup}; never an observation
    scenario: str | None = None
    material_profile_id: str | None = None
    max_cuts: int = 1
    ik_iterations: int = 48
    initial_joint_positions: tuple | None = None
    cutting_law: int | None = None  # None = task default (v2 tasks: law 2); 3-8 are opt-in tlmpm_v3/tlmpm_v4
    # Opt-in (WP-11B): carry another registered profile tool on the task's hand
    # trajectory, translated by a world offset (tool_profile.retarget_task_tool).
    tool_profile_id: str | None = None
    tool_profile_hand_offset_world_m: tuple = (0., 0., 0.)
    # Opt-in (WP-11 Phase D): a compiled scene_config.ConfiguredTaskSpec as its
    # configured_task_payload JSON; reset() recompiles it and requires the same id.
    # task_id must then be that CFG-... id; solver/law/K come from the scene.
    configured_task: str | None = None
    # Opt-in (WP-11 Phase H1): how reset() renews the USD stage after the first reset.
    # "reopen" (default, unchanged): export/open the workcell again; Kit retains ~55-65 MiB host
    # memory per open_stage of this workcell (docs/reports/WP11_PHASE_H_PROFILING.md).
    # "reload_in_place": keep the open stage, release cameras/physics, clear the session layer and
    # reload the unchanged root layer from disk, then author the reset exactly as after an open.
    stage_reset: str = "reopen"
    # Opt-in (WP-11 Phase H3): when the visual ContinuousSpecimen surface (no physics API) is
    # re-extracted. "every_step" (default, unchanged) | "on_capture": only before a camera render
    # and for evaluation, so every rendered frame still shows the surface of its own time step.
    surface_update: str = "every_step"

    def __post_init__(self):
        if self.tool_attachment_mode not in ("legacy_fixed_tool", "physical_grasp"):
            raise ValueError("Unknown tool attachment mode; no implicit fixed-tool fallback")
        if self.tool_init_mode not in ("pregrasped", "free", "pickup"):
            raise ValueError("Unknown tool initialization mode")
        if self.tool_attachment_mode == "physical_grasp" and self.tool_init_mode == "pickup":
            raise NotImplementedError("Use the explicit rack pickup demonstration; cutting reset starts pregrasped")
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
        if self.overview_camera and (self.task_id is None or not (self.camera and self.camera_layout == "wrist_d405_global_d455")):
            raise ValueError("The presentation overview camera requires a task, live cameras and the wrist D405 / global D455 layout")
        from .tasks import PROFILE_TASK_IDS, TASK_IDS, DemoTaskSpec
        if self.configured_task is not None:
            _validate_configured_config(self)
        else:
            if self.task_id not in (None, *TASK_IDS):
                raise ValueError(f'task_id must be one of {TASK_IDS} or None for legacy D0')
            if self.task_id in PROFILE_TASK_IDS and self.tool_attachment_mode != 'physical_grasp':
                raise ValueError('Ruler tasks require physical_grasp; no fixed-tool substitute is implemented')
            if self.max_cuts not in (1, 2) or self.ik_iterations < 1:
                raise ValueError('One or two cut digits and positive IK iterations are required')
            if self.task_id in DemoTaskSpec.TASK_IDS and self.max_cuts != 2:
                raise ValueError(f'{self.task_id} declares two cut digits; set max_cuts=2 explicitly')
            if (self.scenario is not None or self.material_profile_id is not None or self.max_cuts == 2
                    or self.cutting_law is not None) and self.task_id not in PROFILE_TASK_IDS:
                raise ValueError('Scenarios, food profiles, cutting-law selection and two-cut history require opt-in v2 tasks')
        if self.tool_profile_id is not None:
            from .tasks import ProfileTaskSpec
            from .tool_profile import PROFILES
            if (self.tool_profile_id not in PROFILES or self.task_id not in ProfileTaskSpec.TASK_IDS
                    or self.tool_attachment_mode != 'physical_grasp'):
                raise ValueError('tool_profile_id needs a registered profile, a v2 ProfileTaskSpec task and physical_grasp')
        elif any(self.tool_profile_hand_offset_world_m):
            raise ValueError('A tool hand offset requires an explicit tool_profile_id')
        if self.surface_update not in ("every_step", "on_capture"):
            raise ValueError("surface_update must be every_step or on_capture")
        if self.stage_reset not in STAGE_RESET_MODES:
            raise ValueError(f"stage_reset must be one of {STAGE_RESET_MODES}")
        if self.stage_reset == "reload_in_place" and not (self.camera and self.tool_attachment_mode == "physical_grasp"):
            # Validated only with live cameras and the physical grasp (WP-11 Phase H1): with camera=False the
            # second reset segfaults inside PhysX start_simulation in Isaac Sim 6.2 (3 of 3 probes).
            raise ValueError("stage_reset='reload_in_place' requires camera=True and physical_grasp")
        if self.cutting_law not in (None, 1, 2, 3, 4, 5, 6, 7, 8):
            raise ValueError('cutting_law is None or 1-8')
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


def _validate_configured_config(config):
    """Config rules for a compiled configured task (WP-11 Phase D); registered tasks never get here."""
    import json
    from .scene_config import SOLVERS
    payload = json.loads(config.configured_task)
    scene = payload.get('scene') or {}
    solver = scene.get('solver') or {}
    if config.task_id != payload.get('task_id') or not str(config.task_id).startswith('CFG-'):
        raise ValueError('A configured task needs task_id equal to its compiled CFG-... id')
    if config.tool_attachment_mode != 'physical_grasp':
        raise ValueError('Configured ruler-tool tasks require physical_grasp; no fixed-tool substitute is implemented')
    if any(value is not None for value in (config.scenario, config.material_profile_id, config.tool_profile_id,
                                          config.particle_spacing_m, config.initial_joint_positions)):
        raise ValueError('The scene declares workcell, material, tool, spacing and HOME; no runtime override is accepted')
    if config.cutting_law not in (None, solver.get('cutting_law')):
        raise ValueError('cutting_law must be None or the scene solver law')
    limit = SOLVERS.get(solver.get('module'), {}).get('max_cuts', 0)
    if config.max_cuts != solver.get('max_cuts') or not 1 <= config.max_cuts <= limit or config.ik_iterations < 1:
        raise ValueError(f"max_cuts must equal the scene solver max_cuts {solver.get('max_cuts')} (limit {limit}) "
                         "and IK iterations must be positive")


def renew_stage_layers(stage):
    """Return an open USD stage to its on-disk composition (stage_reset='reload_in_place', WP-11 Phase H1).

    Clears the session layer (every reset edit is authored there) and reloads the root layer and any
    other dirty file layer from disk. Raises if any non-session layer stays edited. Pure USD; no Kit."""
    session = stage.GetSessionLayer()
    dirty_before = sorted(layer.identifier for layer in stage.GetUsedLayers() if layer.dirty and layer != session)
    session.Clear()
    stage.GetRootLayer().Reload(force=True)
    reloaded = [stage.GetRootLayer().identifier]
    for layer in stage.GetUsedLayers():
        if layer != session and layer.dirty and not layer.anonymous:
            layer.Reload(force=True)
            reloaded.append(layer.identifier)
    still_dirty = sorted(layer.identifier for layer in stage.GetUsedLayers() if layer != session and layer.dirty)
    if still_dirty or not session.empty:
        raise RuntimeError(f"reload_in_place left edited layers: {still_dirty}")
    return {"dirty_layers_before": dirty_before, "reloaded_layers": reloaded, "session_layer_empty": bool(session.empty)}


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
        self.home = None  # declared HOME (WP-11 Phase D), set at reset

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
        self.physical_grasp = c.tool_attachment_mode == "physical_grasp"
        from .task_scenarios import initial_joints
        self.initial_joints = c.initial_joint_positions or initial_joints(c.scenario, c.task_id)
        self.pad_geometry = []
        if hasattr(self, "tool_contacts"):
            self.tool_contacts.close()
        self.external_tool_wrench = np.zeros(6)
        self.gripper_command = "hold"
        self._rigid_probe_only = False
        self.reset_index += 1
        self.SimulationManager = SimulationManager
        self.compose_tool_pose = compose_tool_pose
        self.reference_pose = reference_pose
        self.task=None
        self.mount=np.array([0.,0.,.18,1.,0.,0.,0.])
        if c.configured_task is not None:
            # WP-11 Phase D: the compiled scene/goal task, rebuilt from its recorded config
            # (same id and sha256 required); fixtures, tool, material and HOME follow the scene.
            from .scene_config import task_from_payload
            self.task = task_from_payload(c.configured_task)
            if abs(self.task.control_dt_s-c.control_dt)>1e-12 or abs(self.task.grid_dx_m-c.dx)>1e-12:
                raise ValueError("Native Kit control_dt/dx must match the shared task specification")
            self.reference_pose=self.task.reference_edge_pose_world
            self.mount=np.asarray(self.task.hand_from_edge_xyz_xyzw,float)
        elif c.task_id:
            from .tasks import make_task
            self.task=make_task(c.task_id,seed=c.seed,variant=c.task_variant,particle_spacing_m=c.particle_spacing_m,
                                workpiece_kind=c.workpiece_kind)
            from .task_scenarios import configure_task
            self.task = configure_task(self.task, scenario=c.scenario, material_profile_id=c.material_profile_id)
            if c.tool_profile_id is not None:
                from .tool_profile import retarget_task_tool
                self.task = retarget_task_tool(self.task, c.tool_profile_id, c.tool_profile_hand_offset_world_m)
            if abs(self.task.control_dt_s-c.control_dt)>1e-12 or abs(self.task.grid_dx_m-c.dx)>1e-12:
                raise ValueError("Native Kit control_dt/dx must match the shared task specification")
            self.reference_pose=self.task.reference_edge_pose_world
            self.mount=np.asarray(self.task.hand_from_edge_xyz_xyzw,float)
        self.home = self._declared_home()
        self.home_edge_pose = None
        if self.home is not None:
            if c.initial_joint_positions is not None or not self.physical_grasp:
                raise ValueError("A declared HOME needs physical_grasp and no separate initial_joint_positions")
            self.initial_joints = tuple(self.home["joint_positions"])
            home_fk = DualFrankaController(RobotConfig(control_dt=c.control_dt, tool_shape="cut_wedge",
                tool_offset_m=tuple(self.mount[:3]), tool_rotation_xyzw=tuple(self.mount[3:])))
            self.home_edge_pose = np.asarray(home_fk.forward_kinematics(self.initial_joints), float)
        self.wp = wp
        omni.timeline.get_timeline_interface().stop()
        context = omni.usd.get_context()
        stage_key = (str(Path(c.scene).resolve()) if c.scene else None, int(c.seed), float(c.dx))
        previous_stage = getattr(self, "_stage_source", None)
        if (c.stage_reset == "reload_in_place" and previous_stage is not None
                and previous_stage["key"] == stage_key and context.get_stage() is not None):
            # Same workcell file (same export arguments): renew the open stage in place instead of
            # open_stage (whose Kit-side host memory is not returned). The composed stage equals a
            # fresh open of the same file: empty session layer, root layer re-read from disk.
            source = previous_stage["source"]
            self.stage_reset_receipt = self._reload_stage_in_place(context, source)
        else:
            if c.scene:
                source = Path(c.scene).resolve()
            else:
                scene_dir = self.output/f"initial_workcell_{self.reset_index:02d}"
                scene_dir.mkdir()
                export(Namespace(variant="D0", family="dome", seed=c.seed, spacing=c.dx/2, asset_root=None), scene_dir)
                source = scene_dir/"workcell.usda"
            if not context.open_stage(str(source)):
                raise RuntimeError("Isaac could not open the initial workcell")
            self._stage_source = {"key": stage_key, "source": source}
            self.stage_reset_receipt = {"mode": "open_stage", "source": str(source), "reset_index": self.reset_index}
        wait_for_workcell_stage(self.app, context, self.output/f"stage_load_{self.reset_index:02d}.json")
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
        blade_dimensions=([self.task.blade_length_m,self.task.blade_height_m,self.task.blade_half_width_m]
                          if self.task else [.012,.004,.025])
        if self.physical_grasp:
            from .physical_tool import ToolGraspConfig, make_tool_spec
            from .native_tool import author_tool, nominal_tool_pose
            from .physical_contacts import NativeToolContacts
            self.grasp = (ToolGraspConfig(**c.grasp) if isinstance(c.grasp, dict)
                          else c.grasp or ToolGraspConfig())
            self.grip_force_limit_n=self.grasp.max_finger_force_n
            self.grip_friction=self.grasp.friction
            from .tasks import ProfileTaskSpec
            if isinstance(self.task, ProfileTaskSpec):
                # v2 task: the profile tool; v1 tasks keep the original path below.
                from .tool_profile import require_task_tool, tool_spec_for_task
                self.tool_spec = c.tool_spec or tool_spec_for_task(self.task)
                require_task_tool(self.tool_spec, self.task)
            else:
                self.tool_spec = c.tool_spec or make_tool_spec(self.task,
                    hand_from_edge_xyz_xyzw=self.mount, blade_length_m=blade_dimensions[0],
                    blade_height_m=blade_dimensions[1], blade_half_width_m=blade_dimensions[2])
                from .tool_compatibility import require_matching_blade
                require_matching_blade(self.tool_spec,self.mount,blade_dimensions)
            self.finger_paths = [next(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(hand_paths[0]))
                                     if p.GetName() == name)
                                 for name in ("panda_leftfinger", "panda_rightfinger")]
            # HOME reset (WP-11 Phase D): the knife at FK(q_home)*hand_from_tool, blade below the fingers.
            authored_edge = self.reference_pose(0.) if self.home_edge_pose is None else self.home_edge_pose
            self.tool_path = author_tool(stage, self.tool_spec,
                nominal_tool_pose(authored_edge, self.tool_spec), self.grasp, self.finger_paths)
            knife_path = self.knife_path = self.tool_path+"/CuttingEdge"
            half = np.asarray(self.tool_spec.handle_half_extents_m)
            self.tool_contacts = NativeToolContacts(stage, self.tool_path, self.finger_paths,
                hand_paths[0], c.control_dt/c.robot_substeps, handle_bounds_tool_m=(-half, half))
        else:
            knife = UsdGeom.Mesh.Define(stage, knife_path)
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
        self.pre_authored_camera_specs=None
        if self.physical_grasp and c.camera and c.camera_layout == "wrist_d405_global_d455":
            # Contact sensors create a view over every rigid in the stage.
            # Finish camera deletions/API removal before any view is created.
            # Nominal FK only chooses a fixed optical mount at reset; every
            # runtime camera extrinsic still comes from the actual articulation.
            from .camera_rig import author_camera_rig
            from .scene_frames import matrix_pose
            from .workcell import ASSETS
            self.ik = DualFrankaController(RobotConfig(control_dt=c.control_dt, tool_shape="cut_wedge",
                tool_offset_m=tuple(self.mount[:3]),tool_rotation_xyzw=tuple(self.mount[3:]),
                wedge_length_m=blade_dimensions[0],wedge_height_m=blade_dimensions[1],wedge_half_width_m=blade_dimensions[2]))
            reset_q=self.ik.initial_q.reshape(2,9).copy()
            if self.initial_joints is not None:
                reset_q[0, :7] = self.ik._validate_joints(self.initial_joints)
                self.ik.state.joint_q.assign(reset_q.ravel())
            if self.home_edge_pose is None:
                reset_q[0,:7],_=self.ik.inverse_kinematics(self.reference_pose(0.), iterations=c.ik_iterations)
            self.ik.state.joint_q.assign(reset_q.ravel())
            self.ik.newton.eval_fk(self.ik.model,self.ik.state.joint_q,self.ik.state.joint_qd,self.ik.state)
            body_q=self.ik.state.body_q.numpy()
            nominal_hands=[body_q[self.ik.hand_local+i*self.ik.bodies_per_arm] for i in range(2)]
            self.pre_authored_camera_specs=author_camera_rig(stage,hand_paths,
                ROOT/"artifacts/workcell_asset_cache"/ASSETS["d455"],task_id=c.task_id or "TRIM-01",
                camera_layout=c.camera_layout,initial_hand_poses_world_xyzw=nominal_hands)
        SimulationManager.setup_simulation(dt=c.control_dt/c.robot_substeps, device="cpu")
        if not SimulationManager.switch_physics_engine("physx"):
            raise RuntimeError("Native Isaac PhysX engine is unavailable")
        SimulationManager.enable_gpu_dynamics(False)
        SimulationManager.enable_fabric(False)
        if self.physical_grasp:
            scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
            if len(scenes) != 1:
                raise RuntimeError("Physical tool requires one explicit gravity scene")
            physics_scene=UsdPhysics.Scene(scenes[0])
            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
            physics_scene.CreateGravityMagnitudeAttr(9.81)
        self.arms = Articulation(["/World/Franka_0", "/World/Franka_1"])
        self.hand = RigidPrim(hand_paths[0])
        self.hands = RigidPrim(hand_paths)
        self.hand_paths = hand_paths
        if self.physical_grasp:
            self.tool_body = RigidPrim(self.tool_path)
            if getattr(self.task, 'tool_profile_id', None):
                from .finger_geometry import discover_pads
                self.pad_geometry = discover_pads(stage, hand_paths[0])
                self.pad_bodies = RigidPrim([pad['body_path'] for pad in self.pad_geometry])
        SimulationManager.initialize_physics()
        if not self.arms.is_physics_tensor_entity_valid() or not self.hand.is_physics_tensor_entity_valid():
            raise RuntimeError("Actual PhysX articulation/hand tensor views did not initialize")
        expected = [f"panda_joint{i}" for i in range(1, 8)]+["panda_finger_joint1", "panda_finger_joint2"]
        if self.arms.dof_names != expected:
            raise RuntimeError(f"Unexpected PhysX DOF order: {self.arms.dof_names}")
        self.arms.set_dof_gains([1000.]*4+[600.]*3+[350.,350.], [80.]*4+[40.]*3+[10.,10.])
        self.arms.set_dof_max_efforts([87.]*4+[12.]*3+[20.,20.])
        if self.physical_grasp:
            self.arms.set_dof_gains([1000.]*4+[600.]*3+[self.grasp.stiffness_n_m]*2,
                                    [80.]*4+[40.]*3+[self.grasp.damping_n_s_m]*2)
            self.arms.set_dof_max_efforts([87.]*4+[12.]*3+[self.grasp.max_finger_force_n]*2)
        self.arms.set_solver_iteration_counts(position_counts=16, velocity_counts=4)
        if not (self.physical_grasp and self.pre_authored_camera_specs is not None):
            self.ik = DualFrankaController(RobotConfig(control_dt=c.control_dt, tool_shape="cut_wedge",
                   tool_offset_m=tuple(self.mount[:3]),tool_rotation_xyzw=tuple(self.mount[3:]),
                   wedge_length_m=blade_dimensions[0],wedge_height_m=blade_dimensions[1],wedge_half_width_m=blade_dimensions[2]))
        self.park_q=self.ik.initial_q.reshape(2,9).copy()
        paths=[]
        for index,label in enumerate(self.ik.model.body_label):
            source_path=label.split("/",1)[1]
            paths.append(f"/World/Franka_{index//self.ik.bodies_per_arm}"+source_path[len("/panda"):])
        self.rigid_link_paths=paths+([self.tool_path] if self.physical_grasp else [])
        self.rigid_links=RigidPrim(self.rigid_link_paths)
        if self.initial_joints is not None:
            seed_q = self.ik.initial_q.copy()
            seed_q[:7] = self.ik._validate_joints(self.initial_joints)
            self.ik.state.joint_q.assign(seed_q)
        if self.home_edge_pose is not None:
            joints = self.ik._validate_joints(self.initial_joints)
            ik_info = self._home_receipt()
        else:
            joints, ik_info = self.ik.inverse_kinematics(self.reference_pose(0.), iterations=c.ik_iterations)
        if not ik_info["converged"]:
            raise RuntimeError(f"Initial IK failed: {ik_info}")
        q = self.park_q.copy()
        q[0, :7] = joints
        if self.physical_grasp:
            q[0, 7:] = self.grasp.initial_finger_position_m
        self.arms.set_dof_positions(q)  # reset only; never called by step()
        self.arms.set_dof_velocities(np.zeros_like(q))
        self.arms.set_dof_position_targets(q)
        self.arms.set_dof_velocity_targets(np.zeros_like(q))
        self.previous_command_q=q.copy()
        self.arms.set_dof_efforts(self.arms.get_dof_gravity_compensation_forces())
        SimulationManager.step()
        if self.physical_grasp:
            com_position,com_orientation=self.tool_body.get_coms()
            self.dynamic_tool_properties={"path":self.tool_path,
                "mass_kg":numpy_data(self.tool_body.get_masses()).tolist(),
                "com_pose_native_position":numpy_data(com_position).tolist(),
                "com_pose_native_orientation_wxyz":numpy_data(com_orientation).tolist(),
                "inertia_native_flat_kg_m2":numpy_data(self.tool_body.get_inertias()).tolist(),
                "gravity_world_m_s2":[0.,0.,-9.81],"readback_source":"actual PhysX tensor view"}
            self._settle_grasp(q)
        self.native_time_origin = SimulationManager.get_simulation_time()
        self.native_step_origin = SimulationManager.get_num_physics_steps()
        import omni.physx
        self.physx=omni.physx.get_physx_interface()
        self.physx.update_transformations(False,True,True)
        self.time_s, self.step_index = 0., 0
        initial = self.proprioception()
        self.last_diagnostics = {}
        self.grasp_monitor = None
        if self.physical_grasp and c.tool_init_mode == "pregrasped":
            from .grasp_monitor import GraspMonitor
            self.grasp_monitor = GraspMonitor(initial["hand_pose"], initial["tool_body_pose_xyzw"], self.tool_spec)
        attachment=self._check_attachment(initial["tool_pose"])
        material_config=CouplingConfig(control_dt=c.control_dt, material_dt=c.material_dt,
                       dx=c.dx, device=c.device, feedback=c.feedback)
        if c.configured_task is not None:
            # Solver, law and K from the scene (tlmpm_v5 packed history allowed up to its limit).
            self.material = LiveCuttingMaterial(material_config, task=self.task, max_cuts=self.task.max_cuts,
                cutting_law=self.task.cutting_law, solver="tlmpm_v5" if self.task.solver_module == "tlmpm_v5" else None)
        else:
            self.material = (LiveCuttingMaterial(material_config,task=self.task,max_cuts=c.max_cuts,cutting_law=c.cutting_law) if self.task else LiveCuttingMaterial(material_config))
        if self.pad_geometry:
            self.material.configure_pad_obstacles(self._pad_boxes(initial['tool_pose']),
                [None] + [pad['body_path'] for pad in self.pad_geometry])
        material_setup = self.material.reset(initial["tool_pose"], c.seed)
        if self.pad_geometry:
            write_json(self.output / f'finger_geometry_{self.reset_index:02d}.json',
                       {'pads': self.pad_geometry, 'boxes_edge': self._pad_boxes(initial['tool_pose']),
                        'scope': 'Actual USD pads; conservative edge-aligned bounds at each coupling boundary; separate finger reactions'})
        spacing=self.task.particle_spacing_m if self.task else c.dx/2
        voxel=spacing/2 if c.surface_profile=="current_volume_v1" else min(spacing,.0015)
        self.surface = ComponentSurface(spacing, c.device, voxel_size=voxel,profile=c.surface_profile)
        self.bridge = DynamicSurfaceBridge(stage)
        self._update_surface()
        self.overview_spec,self.overview_sensor,self.overview_frame=None,None,None
        self.closeup_spec,self.closeup_sensor,self.closeup_frame=None,None,None
        if c.camera:
            self._create_cameras()
        else:
            self.cameras,self.calibration,self.camera_specs,self.camera_pose_trace = {},{},{},[]
        setup = {"config":asdict(c), "source_scene":str(source), "initial_ik":ik_info,
            "material":material_setup["material"],"material_setup":material_setup,
            "initial_mass_kg":material_setup["initial_mass_kg"],
            "world_from_workpiece_xyzw":material_setup["world_from_workpiece_xyzw"],
            "T_hand_edge_xyz_xyzw":self.mount.tolist(),
            "blade_dimensions_m":dict(zip(("length","height","half_width"),blade_dimensions)),
            "task":self.task.to_dict() if self.task else None,
            "cutting_hand_body_label":self.ik.model.body_label[self.ik.hand_local],
            "cutting_hand_body_index":self.ik.hand_local,
            "physical_knife_body_index":len(paths) if self.physical_grasp else self.ik.hand_local,
            "tool_attachment_mode":c.tool_attachment_mode,
            "tool_init_mode":c.tool_init_mode,
            "physical_tool":self.tool_spec.to_dict() if self.physical_grasp else None,
            "dynamic_tool_properties":getattr(self,"dynamic_tool_properties",None),
            "grasp_stabilization":getattr(self, "grasp_reset_receipt", None),
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
            "feedback_delay_s":c.control_dt, "proprioception_source":"actual PhysX tensors",
            "stage_reset":self.stage_reset_receipt}
        if self.task is not None and (getattr(self,"fixture_plan",None) or {}).get("any_material_declared"):
            from .native_fixture import authored_material_receipt
            setup["task_fixture_materials"]=authored_material_receipt(stage,self.fixture_plan)
        if self.home is not None:
            setup["home_reset"]={**ik_info, "measured_after_pregrasp":{"q":initial["q"][0].tolist(),
                "hand_pose_xyzw":np.asarray(initial["hand_pose"]).tolist(),
                "tool_body_pose_xyzw":np.asarray(initial["tool_body_pose_xyzw"]).tolist(),
                "tool_edge_pose_xyzw":np.asarray(initial["tool_pose"]).tolist(),
                "joint_error_vs_home_max_rad":float(np.abs(initial["q"][0]-np.asarray(self.initial_joints)).max())},
                "secondary_arm_q":initial["q"][1].tolist(), "secondary_arm_park_q":self.park_q[1,:7].tolist()}
        if self.overview_spec is not None:
            setup["presentation_overview_camera"]=self.overview_spec
        if self.closeup_spec is not None:
            setup["presentation_closeup_camera"]=self.closeup_spec
        write_json(self.output/("setup.json" if self.reset_index==0 else f"reset_setup_{self.reset_index:02d}.json"), setup)
        return initial

    def _release_cameras(self):
        """Detach every replicator annotator and destroy the render products of the previous reset."""
        released = []
        for group in ("camera_params", "native_rgb", "native_xyz"):
            for name, annotator in (getattr(self, group, None) or {}).items():
                annotator.detach()
                released.append(f"{group}:{name}")
        sensors = dict(getattr(self, "cameras", None) or {})
        for kind in ("overview", "closeup"):
            if getattr(self, f"{kind}_sensor", None) is not None:
                sensors[kind] = getattr(self, f"{kind}_sensor")
        for name, sensor in sensors.items():
            # Same teardown the sensor runs when it is garbage collected (pinned Isaac Sim 6.2 API).
            sensor._invalidate_sensor()
            released.append(f"sensor:{name}")
        self.cameras, self.camera_params, self.native_rgb, self.native_xyz = {}, {}, {}, {}
        self.overview_sensor = self.closeup_sensor = None
        return released

    def _reload_stage_in_place(self, context, source):
        """stage_reset='reload_in_place': renew the already open stage without open_stage (WP-11 Phase H1).

        Releases the previous reset's cameras, contact sensor, tensor views and PhysX objects, clears the
        session layer (every reset edit is authored there), reloads the root layer and any other dirty
        file layer from disk, and replays the SimulationManager stage-opened reset. reset() then authors
        the episode exactly as after open_stage."""
        from isaacsim.core.simulation_manager import SimulationManager
        stage = context.get_stage()
        if Path(stage.GetRootLayer().realPath).resolve() != Path(source).resolve():
            raise RuntimeError("reload_in_place: the open stage is not the reset workcell")
        released = self._release_cameras()
        for name in ("arms", "hand", "hands", "tool_body", "rigid_links", "pad_bodies",
                     "bridge", "surface", "material", "grasp_monitor"):   # tool_contacts: closed at the top of reset
            if getattr(self, name, None) is not None:
                setattr(self, name, None)
        import gc
        gc.collect()   # finalize the old tensor views / prim wrappers while their physics objects still exist
        SimulationManager.invalidate_physics()
        # The interface SimulationManager started the simulation with (force_load + start_simulation):
        # reset (releases every physics object), then detach/attach as a stage close/open does.
        stage_update = SimulationManager._physics_stage_update_interface
        stage_update.reset_simulation()
        stage_update.on_detach()
        layers = renew_stage_layers(stage)
        stage_update.on_attach(context.get_stage_id())
        SimulationManager._on_stage_opened(None)  # the same simulation-manager reset an OPENED event runs
        import omni.kit.undo
        omni.kit.undo.clear_stack()   # open_stage starts a new undo history; so does this reset
        omni.kit.undo.clear_history()
        return {"mode": "reload_in_place", "source": str(source), "reset_index": self.reset_index,
                "released": released, **layers,
                "note": "PhysX reset/detach/attach restarts the simulation clock (native_time_origin_s repeats); "
                        "the native physics step counter keeps counting (native_step_origin grows)"}

    def plan_return_home(self, q_start):
        """Preflighted return-to-home joint plan from measured joints (separate .venv process; nothing moves)."""
        if self.home is None:
            raise ValueError("No HOME is declared for this scene/scenario")
        if self.config.configured_task is None:
            raise NotImplementedError("Return-to-home planning is implemented for configured scene tasks only")
        from .motion_preflight import run_cli
        receipt = run_cli("return-home", self.config.configured_task, q_start=np.asarray(q_start, float).tolist(),
                          log_dir=self.output/"motion_preflight")
        if receipt.get("status") == "ERROR":
            return {**receipt, "status": "REFUSED", "reason": receipt["reason"], "q": [], "labels": []}
        return receipt

    def check_measured_path(self, qs, modes):
        """C2 geometric clearance of an executed joint path (separate .venv process)."""
        if self.config.configured_task is None:
            return {"status": "NOT_RUN", "reason": "configured scene tasks only"}
        from .motion_preflight import run_cli
        return run_cli("check-path", self.config.configured_task, q=np.asarray(qs, float).tolist(), modes=list(modes),
                       log_dir=self.output/"motion_preflight")

    def _declared_home(self):
        """HOME from the configured scene workcell or the scenario (wp11_scenarios_v1), else None."""
        c=self.config
        if c.configured_task is not None:
            return self.task.home
        if c.scenario is None:
            return None
        from .task_scenarios import scenario_home
        return scenario_home(c.scenario)

    def _home_receipt(self):
        """Reset-at-HOME receipt: joints are set directly (no IK); FK vs the task's t = 0 reference."""
        from .scene_frames import pose_errors
        reference=np.asarray(self.reference_pose(0.),float)
        translation,rotation=pose_errors(reference[None],self.home_edge_pose[None])
        return {"used":False,"converged":True,"reason":"declared home_joint_positions (no initial IK)",
                "home_scenario":self.home["scenario"],"home_joint_positions":list(self.home["joint_positions"]),
                "home_source":self.home.get("source"),"home_edge_pose_fk_world_xyzw":self.home_edge_pose.tolist(),
                "home_declared_hand_pose_world_xyzw":list(self.home["hand_pose_world_xyzw"]),
                "reference_edge_pose_t0_world_xyzw":reference.tolist(),
                "fk_vs_reference_t0_translation_m":float(translation[0]),"fk_vs_reference_t0_rotation_rad":float(rotation[0]),
                "secondary_arm":"shared normal safe park (unchanged)"}

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
        # Declared (v3) fixture materials: display colour + PhysX material; undeclared
        # boxes keep the historical colour and no physics material binding.
        from .native_fixture import bind_physics_material, fixture_authoring_plan
        self.fixture_plan=plan=fixture_authoring_plan(task,getattr(self.config,"scenario",None))
        for index,(box,row) in enumerate(zip(task.fixture_boxes_local,plan["boxes"])):
            cube=UsdGeom.Cube.Define(stage,f"/World/TaskFixture/Box_{index}")
            cube.CreateSizeAttr(2.)
            cube.GetPrim().SetCustomDataByKey("research:fixture_name",box.name)
            cube.AddTranslateOp().Set(Gf.Vec3d(*box.center_m))
            cube.AddScaleOp().Set(Gf.Vec3f(*box.half_extents_m))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*row["display_color_rgb"])])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            if row["material_declared"]:
                cube.GetPrim().SetCustomDataByKey("research:fixture_role",row["role"])
                bind_physics_material(stage,cube.GetPrim(),row["physics_material"])
        if plan["table"] is not None:
            bind_physics_material(stage,stage.GetPrimAtPath("/World/Table"),plan["table"]["physics_material"])
        table=UsdGeom.Xformable(stage.GetPrimAtPath("/World/Table"))
        for op in table.GetOrderedXformOps():
            if op.GetOpType()==UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(0,0,task.table_top_world_z_m-.025))

    def body_poses(self):
        positions,orientations=self.rigid_links.get_world_poses()
        p,r=numpy_data(positions),numpy_data(orientations)
        return np.asarray([pose_from_wxyz(position,orientation) for position,orientation in zip(p,r)])

    def set_gripper(self, command):
        if not self.physical_grasp:
            raise ValueError("Gripper tool skills require physical_grasp; legacy attachment cannot release")
        if command not in ("hold", "close", "open", "release"):
            raise ValueError("Choose hold, close, open or release")
        self.gripper_command = command

    def set_external_tool_wrench(self, wrench_world):
        value = np.asarray(wrench_world, float)
        if not self.physical_grasp or value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError("A physical tool and finite world-frame edge force/torque are required")
        self.external_tool_wrench = value.copy()

    def set_grip_force_limit(self, limit_n):
        if not self.physical_grasp or not np.isfinite(limit_n) or not 0 <= limit_n <= 70:
            raise ValueError("Physical finger effort limit must be finite and between0 and70N")
        self.grip_force_limit_n = float(limit_n)
        self.arms.set_dof_max_efforts([87.]*4+[12.]*3+[float(limit_n)]*2)

    def set_grip_friction(self, coefficient):
        from pxr import UsdPhysics
        if not self.physical_grasp or not np.isfinite(coefficient) or not 0 <= coefficient <= 2:
            raise ValueError("Finite contact friction between0 and2 is required")
        material=UsdPhysics.MaterialAPI(self.stage.GetPrimAtPath("/World/Tools/GripMaterial"))
        material.CreateStaticFrictionAttr(float(coefficient))
        material.CreateDynamicFrictionAttr(float(coefficient))
        self.grip_friction = float(coefficient)

    def solve_edge_ik(self, edge_pose):
        measured=numpy_data(self.arms.get_dof_positions())
        self.ik.state.joint_q.assign(measured.ravel().astype(np.float32))
        return self.ik.inverse_kinematics(edge_pose)

    def rigid_probe_step(self, joint_targets=None):
        """Same rigid bodies/contact controller, no material solve or Task claim.

        A subsequent cutting step requires reset, because its material clock
        has deliberately not advanced during these independent grasp controls.
        """
        if not self.physical_grasp:
            raise ValueError("Rigid tool controls require physical_grasp")
        self._rigid_probe_only=True
        target=self.previous_command_q.copy()
        if joint_targets is not None:
            target[0,:7]=self.ik._validate_joints(joint_targets)
        target[0,7:] = .04 if self.gripper_command in ("open","release") else self.grasp.closed_target_m
        self.arms.set_dof_position_targets(target)
        self.arms.set_dof_velocity_targets(np.zeros_like(target))
        self.previous_command_q=target.copy()
        for _ in range(self.config.robot_substeps):
            before=self.proprioception()
            wrench=self.external_tool_wrench
            self.tool_body.apply_forces_and_torques_at_pos(wrench[None,:3],wrench[None,3:],
                positions=before["tool_pose"][None,:3],local_frame=False)
            self.arms.set_dof_efforts(self.arms.get_dof_gravity_compensation_forces())
            self.SimulationManager.step()
            self._grasp_sample("rigid_control_probe",np.zeros(6),before)
        self.step_index += 1
        self.time_s=self.step_index*self.config.control_dt
        self.physx.update_transformations(False,True,True)
        return self.proprioception(),self.grasp_trace[-1]

    def _grasp_sample(self, phase, applied_edge_wrench=None, application_state=None):
        from .scene_frames import matrix_pose, pose_matrix
        state = self.proprioception()
        relative = matrix_pose(np.linalg.inv(pose_matrix(state["hand_pose"])) @
                               pose_matrix(state["tool_body_pose_xyzw"]))
        contacts = self.tool_contacts.sample(tool_pose_xyzw=state["tool_body_pose_xyzw"],
            contact_geometry_pose_xyzw=None if application_state is None else application_state["tool_body_pose_xyzw"])
        if phase != "pregrasp_settling" and not all(contacts.get(key) for key in
                ("sensor_valid","sensor_fresh","new_physics_step")):
            raise RuntimeError("INVALID_OR_STALE_NATIVE_CONTACT_VIEW: reset instead of using stale rigid state")
        row = {"phase":phase, "native_time_s":self.SimulationManager.get_simulation_time(),
               "hand_pose_xyzw":state["hand_pose"], "tool_body_pose_xyzw":state["tool_body_pose_xyzw"],
               "measured_edge_pose_xyzw":state["tool_pose"], "relative_tool_pose_xyzw":relative,
               "tool_linear_velocity_world_m_s":state["tool_linear_velocity_world_m_s"],
               "tool_angular_velocity_world_rad_s":state["tool_angular_velocity_world_rad_s"],
               "finger_q_m":state["finger_q"], "finger_qdot_m_s":state["finger_qdot"],
               "gripper_command":self.gripper_command, "contacts":contacts}
        row.update(commanded_finger_force_limit_n=getattr(self,"grip_force_limit_n",self.grasp.max_finger_force_n),
                   commanded_contact_friction=getattr(self,"grip_friction",self.grasp.friction))
        if application_state is not None:
            row["tool_body_pose_before_step_xyzw"] = application_state["tool_body_pose_xyzw"].copy()
        if applied_edge_wrench is not None:
            from .physical_tool import edge_wrench_to_com, world_com_position
            row.update(applied_edge_material_wrench_world=np.asarray(applied_edge_wrench).copy(),
                applied_external_edge_wrench_world=self.external_tool_wrench.copy(),
                force_application_point_world_m=application_state["tool_pose"][:3].copy(),
                tool_com_before_step_world_m=world_com_position(application_state["tool_body_pose_xyzw"],self.tool_spec),
                tool_body_pose_before_step_xyzw=application_state["tool_body_pose_xyzw"].copy(),
                tool_linear_velocity_before_step_world_m_s=application_state["tool_linear_velocity_world_m_s"].copy(),
                tool_angular_velocity_before_step_world_rad_s=application_state["tool_angular_velocity_world_rad_s"].copy(),
                applied_total_wrench_about_com_world=edge_wrench_to_com(
                    applied_edge_wrench+self.external_tool_wrench, application_state["tool_body_pose_xyzw"],self.tool_spec),
                wrench_interval_dt_s=self.config.control_dt/self.config.robot_substeps,
                direct_hand_wrench_injection=False)
        self.grasp_trace.append(row)
        import json
        def json_default(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        with (self.output/f"physical_grasp_trace_{self.reset_index:02d}.jsonl").open("a") as stream:
            stream.write(json.dumps(row,default=json_default,allow_nan=False)+"\n")
        return row

    def _settle_grasp(self, q):
        """Actual finger actuation before material exists or the episode starts."""
        from scipy.spatial.transform import Rotation
        from .scene_frames import pose_errors
        c, g, manager = self.config, self.grasp, self.SimulationManager
        self.grasp_trace = []
        hold = q.copy()
        hold[0, 7:] = g.closed_target_m if c.tool_init_mode == "pregrasped" else .04
        self.arms.set_dof_position_targets(hold)
        if c.tool_init_mode == "free":
            self.gripper_command = "release"
            self.grasp_reset_receipt = {"status":"NOT_REQUESTED_FREE_TOOL", "material_steps":0,
                                        "task_time_consumed_s":0.}
            return
        steps = int(np.ceil(g.settle_s/(c.control_dt/c.robot_substeps)))
        for _ in range(steps):
            before = self.proprioception()
            self.arms.set_dof_efforts(self.arms.get_dof_gravity_compensation_forces())
            manager.step()
            self._grasp_sample("pregrasp_settling", application_state=before)
        final = self.grasp_trace[-1]["native_time_s"]
        rows = [r for r in self.grasp_trace if r["native_time_s"] >= final-g.stability_window_s-1e-9]
        relative = np.asarray([r["relative_tool_pose_xyzw"] for r in rows])
        translation, rotation = pose_errors(np.tile(relative[0], (len(rows), 1)), relative)
        dt = np.diff([r["native_time_s"] for r in rows])
        speed = np.linalg.norm(np.diff(relative[:, :3], axis=0), axis=1)/dt
        angular = (Rotation.from_quat(relative[1:, 3:]) * Rotation.from_quat(relative[:-1, 3:]).inv()).magnitude()/dt
        bilateral = np.mean([r["contacts"]["bilateral_loaded_handle_contact"] for r in rows])
        penetration = max([max(0.,-point["separation_m"]) for r in rows
            for point in r["contacts"]["tensor_forces"].get("records",[])
            if point.get("separation_m") is not None] or [0.])
        checks = {"sampled_final_window":len(rows)>2,
                  "both_fingers_contact_handle":bool(bilateral >= g.min_bilateral_contact_fraction),
                  "relative_translation":bool(translation.max() <= g.translation_tolerance_m),
                  "relative_rotation":bool(rotation.max() <= g.rotation_tolerance_rad),
                  "relative_linear_speed":bool(speed.max() <= g.linear_speed_tolerance_m_s),
                  "relative_angular_speed":bool(angular.max() <= g.angular_speed_tolerance_rad_s),
                  "recorded_native_contact_separations":all(r["contacts"]["tensor_forces"].get("status") ==
                      "VALID_NATIVE_NORMAL_AND_FRICTION_DATA" for r in rows),
                  "rigid_contact_penetration":bool(penetration <= g.max_penetration_m),
                  "no_unintended_support":all(r["contacts"]["unintended_contact_count"] == 0 for r in rows)}
        self.grasp_reset_receipt = {"status":"PASS_STABLE_GRASP" if all(checks.values()) else "FAIL_GRASP_STABILIZATION",
            "checks":checks, "rigid_steps":steps, "settling_s":steps*c.control_dt/c.robot_substeps,
            "task_time_consumed_s":0., "material_steps":0, "hidden_attachment":False,
            "max_relative_translation_m":float(translation.max()),
            "max_relative_rotation_rad":float(rotation.max()), "max_relative_linear_speed_m_s":float(speed.max()),
            "max_relative_angular_speed_rad_s":float(angular.max()), "bilateral_handle_contact_fraction":float(bilateral),
            "maximum_native_rigid_contact_penetration_m":float(penetration),
            "config":asdict(g)}
        write_json(self.output/f"grasp_reset_{self.reset_index:02d}.json",self.grasp_reset_receipt)
        if not all(checks.values()):
            raise RuntimeError("FAIL_GRASP_STABILIZATION: independent tool did not establish a stable bilateral handle grasp")

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
        extra = {}
        if getattr(self,"physical_grasp",False):
            from .native_tool import tool_state
            extra = tool_state(self.tool_body,self.tool_spec)
            tool = extra["tool_pose"]
            extra.update(finger_q=q[:,7:].copy(),finger_qdot=qdot[:,7:].copy(),
                         tool_attachment_mode="physical_grasp")
        if not np.isfinite(np.r_[q.ravel(),qdot.ravel(),tool]).all():
            raise FloatingPointError("Nonfinite native PhysX state")
        return {"q":q[:,:7].copy(), "qdot":qdot[:,:7].copy(), "hand_pose":hand,
                "hand_pose_both_xyzw":hand_both,
                "tool_pose":tool, "timestamp_s":self.time_s, **extra}

    def _pad_poses(self):
        positions, orientations = self.pad_bodies.get_world_poses()
        return {pad['body_path']: pose_from_wxyz(p, q) for pad, p, q in zip(
            self.pad_geometry, numpy_data(positions), numpy_data(orientations))}

    def _pad_boxes(self, edge):
        from .finger_geometry import pad_boxes_edge
        from .tool_profile import handle_box_edge
        return np.asarray([handle_box_edge(self.task.tool_profile),
                           *pad_boxes_edge(self.pad_geometry, self._pad_poses(), edge)], np.float32)

    def _update_surface(self):
        vertices, faces, self.surface_info = self.surface.extract(self.material.bed)
        if self.task:
            vertices=self.task.points_to_world(vertices).astype(np.float32)
        self.surface_receipt = self.bridge.update(vertices, faces, self.time_s)
        self.surface_arrays = (vertices, faces)
        self._surface_time = self.time_s

    def refresh_surface_for_evaluation(self):
        if getattr(self, '_surface_time', None) != self.time_s:
            self._update_surface()

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
            self.camera_specs=(self.pre_authored_camera_specs if self.pre_authored_camera_specs is not None else
                author_camera_rig(self.stage,self.hand_paths,asset,
                task_id=c.task_id or "TRIM-01",camera_layout=c.camera_layout,
                initial_hand_poses_world_xyzw=self.proprioception()["hand_pose_both_xyzw"]))
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
        if c.overview_camera:
            # Presentation-only RGB: kept out of self.cameras/calibration, so it is
            # never an observation, calibration or camera-contract stream.
            # The close-up is authored with the overview (same flag, same rules).
            from .presentation_camera import author_overview_camera, closeup_camera_spec, overview_camera_spec
            from .native_camera_geometry import transform_error
            for kind,make_spec in (("overview",overview_camera_spec),("closeup",closeup_camera_spec)):
                spec=make_spec(self.task)
                author_overview_camera(self.stage,spec)
                sensor=CameraSensor(RtxCamera(spec["prim_path"],tick_rate=0.),
                    resolution=(spec["height"],spec["width"]),annotators=["rgb"])
                authored=usd_camera_calibration(self.stage,spec["prim_path"],spec["width"],spec["height"])
                spec["authored_usd_readback"]={"T_world_camera":authored["T_world_camera"],"K":authored["K"],
                    "pose_error_vs_spec":transform_error(authored["T_world_camera"],spec["T_world_camera"]),
                    "render_product":str(sensor.render_product.GetPath())}
                setattr(self,f"{kind}_sensor",sensor)
                setattr(self,f"{kind}_spec",spec)
                write_json(self.output/f"presentation_{kind}_camera.json",spec)

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
        if self._rigid_probe_only:
            raise RuntimeError("Reset after rigid-only controls before starting coupled cutting")
        if c.observation_motion_probe and action is not None:
            raise ValueError("The separate observation motion probe uses scripted actions only")
        mode=c.action_mode
        self._command_cut_index=0
        if isinstance(action,dict):
            from .control import validate_cut_index
            validate_cut_index(action, self.material)
            if self.task is None:
                raise ValueError("Task action mappings require a shared CUT/TRIM specification")
            command=self.task.validate_action(action)
            mode="cartesian" if command["mode"]=="cartesian_edge_pose_world" else "joint"
            action=command["target"]
            cutting=bool(cutting and command["cut_enabled"])
            self._command_cut_index=command.get("cut_index",0)
        # The scripted reference is needed only without an explicit action (an explicit
        # action may run past the task episode, e.g. the physical return-to-home).
        desired = self.reference_pose(self.time_s+c.control_dt) if action is None else None
        # Seed IK from current PHYSX joints, without advancing Newton dynamics.
        q = numpy_data(self.arms.get_dof_positions())
        self.ik.state.joint_q.assign(q.ravel().astype(np.float32))
        if action is None or mode=="cartesian":
            if action is not None:
                desired=validate_native_action(action,"cartesian")
            targets, ik_info = self.ik.inverse_kinematics(desired, iterations=c.ik_iterations)
            if not ik_info["converged"]:
                from .control import IKFailure
                raise IKFailure(f"IK failed before native articulation step: {ik_info}", ik_info)
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
        if self.physical_grasp:
            target_q[0,7:] = (.04 if self.gripper_command in ("open","release")
                              else self.grasp.closed_target_m)
        if c.observation_motion_probe:
            # Actual articulation motion for a separate camera calibration probe;
            # disabled in all main physical task baselines.
            target_q[1,0] += .08*np.sin(2*np.pi*(self.time_s+c.control_dt)/2.)
        self.arms.set_dof_position_targets(target_q)
        velocity_target=(target_q-self.previous_command_q)/c.control_dt if c.velocity_feedforward else np.zeros_like(target_q)
        if self.physical_grasp:
            velocity_target[:,7:] = 0.
        self.arms.set_dof_velocity_targets(velocity_target)
        self.previous_command_q=target_q.copy()
        applied = self.material.pending_wrench.copy() if c.feedback else np.zeros(6)
        start = time.perf_counter()
        material_work, external_work = 0., 0.
        for _ in range(c.robot_substeps):
            current = self.proprioception()
            wrench = applied+(self.external_tool_wrench if self.physical_grasp else 0.)
            receiver = self.tool_body if self.physical_grasp else self.hand
            receiver.apply_forces_and_torques_at_pos(wrench[None,:3],wrench[None,3:],
                         positions=current["tool_pose"][None,:3],local_frame=False)
            if self.pad_geometry and c.feedback:
                finger_wrenches = np.asarray([self.material.pending_obstacle_wrenches.get(
                    pad['body_path'], np.zeros(6)) for pad in self.pad_geometry])
                self.pad_bodies.apply_forces_and_torques_at_pos(finger_wrenches[:, :3], finger_wrenches[:, 3:],
                    positions=np.tile(current['tool_pose'][:3], (len(self.pad_geometry), 1)), local_frame=False)
            self.arms.set_dof_efforts(self.arms.get_dof_gravity_compensation_forces())
            manager.step()
            if self.physical_grasp:
                from scipy.spatial.transform import Rotation
                row = self._grasp_sample("coupled_episode", applied, current)
                after = row["measured_edge_pose_xyzw"]
                displacement = np.r_[after[:3]-current["tool_pose"][:3],
                    (Rotation.from_quat(after[3:])*Rotation.from_quat(current["tool_pose"][3:]).inv()).as_rotvec()]
                material_work += float(applied@displacement)
                external_work += float(self.external_tool_wrench@displacement)
        self.step_index += 1
        self.time_s = self.step_index*c.control_dt
        native_relative = manager.get_simulation_time()-self.native_time_origin
        if abs(native_relative-self.time_s) > 1e-6:
            raise RuntimeError(f"Native physics clock drift: {native_relative} versus{self.time_s}")
        self.physx.update_transformations(False,True,True)
        actual = self.proprioception()
        attachment=self._check_attachment(actual["tool_pose"])
        if self.pad_geometry:
            self.material.update_pad_obstacles(self._pad_boxes(actual['tool_pose']))
        diagnostics = self.material.advance(actual["tool_pose"],self.time_s,cutting=cutting,
                           cut_index=self._command_cut_index if action is not None else None)
        if not np.array_equal(diagnostics["applied_previous_material_wrench_world"],applied):
            raise AssertionError("The material receipt does not match the wrench actually applied to PhysX")
        state=self.material.debug_state()
        if not all(np.isfinite(state[name]).all() for name in ("positions","velocities","F","plastic_strain")):
            raise FloatingPointError("Nonfinite live TLMPM state")
        determinant=np.linalg.det(state["F"])
        diagnostics.update(minimum_detF=float(determinant.min()),
                           maximum_material_speed_m_s=float(np.linalg.norm(state["velocities"],axis=1).max()),
                           maximum_plastic_strain=float(state["plastic_strain"].max()))
        if c.camera and c.surface_update == "every_step":
            self._update_surface()
        if c.camera and c.camera_layout == "wrist_d405_global_d455":
            self._update_camera_extrinsics(actual["hand_pose_both_xyzw"])
        physical = {}
        if self.physical_grasp:
            from .physical_tool import edge_wrench_to_com
            interval_rows = self.grasp_trace[-c.robot_substeps:]
            operational = self.grasp_monitor.update(actual["hand_pose"], actual["tool_body_pose_xyzw"],
                time_s=self.time_s,
                bilateral_loaded_tool_fraction=float(np.mean([r["contacts"]["bilateral_loaded_tool_contact"] for r in interval_rows])),
                contacts_valid=all(r["contacts"]["sensor_valid"] and r["contacts"]["sensor_fresh"] for r in interval_rows),
                gripper_command=self.gripper_command) if self.grasp_monitor is not None else {
                    "status":"FREE_TOOL", "grasp_lost":False}
            physical = {"tool_attachment_mode":"physical_grasp", "wrench_recipient":self.tool_path,
                "grasp_lost": operational["grasp_lost"], "grasp_status": operational,
                "direct_hand_material_wrench":False, "external_edge_wrench_world":self.external_tool_wrench.copy(),
                "applied_total_tool_wrench_about_com_world":edge_wrench_to_com(
                    applied+self.external_tool_wrench, actual["tool_body_pose_xyzw"], self.tool_spec),
                "tool_body_pose_xyzw":actual["tool_body_pose_xyzw"],
                "tool_linear_velocity_world_m_s":actual["tool_linear_velocity_world_m_s"],
                "tool_angular_velocity_world_rad_s":actual["tool_angular_velocity_world_rad_s"],
                "grasp":self.grasp_trace[-1], "gripper_command":self.gripper_command,
                "applied_tool_wrench_work_j":material_work,
                "external_tool_wrench_work_j":external_work}
        self.last_diagnostics = {**diagnostics,"commanded_tool_pose":desired,"desired_tool_pose":desired,
                 "commanded_joint_positions_rad":target_q[:,:7].copy(),"action_mode":mode,
                 "commanded_joint_velocity_rad_s":velocity_target,"velocity_feedforward":c.velocity_feedforward,
                 "ik":ik_info,"rendered_attachment":attachment,
                 "robot_physics_wall_s":time.perf_counter()-start,
                 "native_physics_timestamp_s":manager.get_simulation_time(),
                 "native_relative_timestamp_s":native_relative,
                 "relative_native_physics_steps":manager.get_num_physics_steps()-self.native_step_origin,
                 **physical}
        return actual, self.last_diagnostics

    def capture(self):
        import omni.replicator.core as rep
        if self.config.surface_update == "on_capture":
            self.refresh_surface_for_evaluation()   # the surface of this time step, before rendering it
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
        overview=getattr(self,"overview_sensor",None)
        overview_rgb=None
        closeup=getattr(self,"closeup_sensor",None)
        closeup_rgb=None
        for _ in range(self.config.render_warmup+30):
            rep.orchestrator.step(rt_subframes=1,delta_time=0.)
            latest = {name:{a:data(name,sensor,a) for a in
                       (["rgb","distance_to_image_plane","depth_sensor_distance","depth_sensor_point_cloud_position"]
                       if name in self.native_parameters else ["rgb","distance_to_image_plane"])}
                      for name,sensor in self.cameras.items()}
            if overview is not None:
                overview_rgb=overview.get_data("rgb")[0]
            if closeup is not None:
                closeup_rgb=closeup.get_data("rgb")[0]
            if (_ >= self.config.render_warmup-1 and all(value[0] is not None for data in latest.values() for value in data.values())
                    and (overview is None or overview_rgb is not None) and (closeup is None or closeup_rgb is not None)):
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
        if overview is not None:
            rgb=numpy_data(overview_rgb)
            if rgb.shape[:2] != (self.overview_spec["height"],self.overview_spec["width"]):
                raise ValueError("Malformed presentation overview RGB array")
            # Not returned in frames: presentation-only, recorded separately.
            self.overview_frame={"rgb":rgb[:,:,:3].astype(np.uint8,copy=False),"physics_timestamp_s":self.time_s,
                                 "native_physics_timestamp_s":before[0],"camera_id":"overview_rgb",
                                 "presentation_only":True,"live_source":"Isaac Sim6.2 Kit/PhysX/RTX"}
        if closeup is not None:
            rgb=numpy_data(closeup_rgb)
            if rgb.shape[:2] != (self.closeup_spec["height"],self.closeup_spec["width"]):
                raise ValueError("Malformed presentation close-up RGB array")
            # Not returned in frames: presentation-only, recorded separately.
            self.closeup_frame={"rgb":rgb[:,:,:3].astype(np.uint8,copy=False),"physics_timestamp_s":self.time_s,
                                "native_physics_timestamp_s":before[0],"camera_id":"closeup_rgb",
                                "presentation_only":True,"live_source":"Isaac Sim6.2 Kit/PhysX/RTX"}
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
