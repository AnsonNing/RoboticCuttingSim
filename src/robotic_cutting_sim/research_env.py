"""Live articulated cutting environment with explicit vision-only boundaries.

Uses the standalone coupled Panda/TLMPM backend and the full official USD
workcell for ideal RGB-D. Isaac/RTX capture remains an independently tested
renderer adapter; this class never silently substitutes it for simulated D455.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
import time

import numpy as np

from .recording import ROOT


@dataclass(frozen=True)
class ResearchEnvConfig:
    variant: str = "D0"
    family: str = "dome"
    device: str = "cuda:0"
    dx: float = .004
    material_dt: float = .0001
    control_dt: float = .02
    max_steps: int = 400
    camera_mode: str = "ideal_rgbd"
    observation_profile: str = "vision_proprio"
    width: int = 640
    height: int = 480
    scene_path: str | None = None
    feedback: bool = True

    def __post_init__(self):
        if self.camera_mode not in ("ideal_rgbd", "simulated_d455"):
            raise ValueError("camera_mode must be ideal_rgbd or simulated_d455")
        if self.observation_profile not in ("vision_only", "vision_proprio"):
            raise ValueError("Unknown observation profile")
        if min(self.width, self.height, self.max_steps) < 1:
            raise ValueError("Resolution and step count must be positive")
        if not np.isfinite([self.dx, self.material_dt, self.control_dt]).all() or min(self.dx, self.material_dt, self.control_dt) <= 0:
            raise ValueError("Spatial and time steps must be finite and positive")
        if abs(round(self.control_dt/self.material_dt)*self.material_dt-self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of material_dt")


def vision_observation(frames, calibration, timestamp_s, profile, proprio=None, *,
                       include_global_camera=False):
    """Allowlist named RGB-D sensors; global inclusion is an explicit option."""
    from .sensor_contract import validate_frame
    wrist = "wrist_d405_0" in frames or "wrist_d405_1" in frames
    names = ("wrist_d405_0", "wrist_d405_1") if wrist else ("d455_0", "d455_1")
    if include_global_camera:
        names += ("global_d455" if wrist else "global",)
    for name in names:
        for source, description in ((frames[name], "Camera and physics"),
                                    (calibration[name], "Calibration and camera")):
            for field in ("timestamp_s", "physics_timestamp_s"):
                recorded_time = float(source.get(field, timestamp_s))
                if not np.isfinite(recorded_time) or abs(recorded_time-timestamp_s) > 1e-9:
                    raise ValueError(description+" timestamps must match")
    validated = {name: validate_frame(frames[name]["rgb"], frames[name]["depth_m"],
                    timestamp_s, frames[name]["surface_timestamp_s"]) for name in names}
    observation = {key: {name: validated[name][key] for name in names}
                   for key in ("rgb", "depth_m", "valid_mask")}
    observation["calibration"] = {name: calibration[name] for name in names}
    observation["timestamp_s"] = float(timestamp_s)
    if profile == "vision_proprio":
        if proprio is None or abs(float(proprio["timestamp_s"])-timestamp_s) > 1e-8:
            raise ValueError("Measured robot and camera timestamps must match")
        observation["proprio"] = {key: np.asarray(proprio[key]).copy() for key in ("q", "qdot", "tool_pose")}
    elif profile != "vision_only":
        raise ValueError("Unknown observation profile")
    return observation


class ResearchCuttingEnv:
    """reset/step/observe/render/close for the actually articulated testbed.

    None actions follow the slow reference path; 7D actions are desired WORLD
    WEDGE-EDGE poses (XYZ + XYZW). The low-level robot controller performs IK and
    advances articulated dynamics. Measured poses drive MPM contact, with a
    one-control-interval reaction-wrench feedback delay. No policy is trained.
    """
    def __init__(self, config=None):
        self.config = config or ResearchEnvConfig()
        if self.config.camera_mode == "simulated_d455":
            raise NotImplementedError("Live simulated_d455 is not validated: native SingleView AOV capture exists, but registration/calibration remains open; use the explicit RTX capture tools")
        self.core = None
        self._closed = False
        self._temp = None
        self._steps = 0
        self._last_obs = None
        self._surface_time = None

    def _create_stage(self):
        from pxr import Gf, Usd, UsdGeom
        from .usd_surface_bridge import DynamicSurfaceBridge, usd_camera_calibration
        from .robot import wedge_mesh
        c = self.config
        source = Path(c.scene_path) if c.scene_path else None
        if source is None:
            from argparse import Namespace
            from .workcell import export
            ROOT.joinpath("artifacts").mkdir(exist_ok=True)
            self._temp = tempfile.TemporaryDirectory(prefix="research_env_", dir=ROOT/"artifacts")
            destination = Path(self._temp.name)
            export(Namespace(variant="D0", family="dome", seed=self.seed, spacing=c.dx/2,
                             asset_root=None), destination)
            source = destination/"workcell.usda"
        if not source.is_file():
            raise FileNotFoundError(source)
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths = [str(source.resolve())]
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        stage.SetTimeCodesPerSecond(50)
        # The dynamic wedge replaces the preview's independent kinematic box.
        preview = stage.GetPrimAtPath("/World/KinematicBlade")
        if preview:
            preview.SetActive(False)
        blade = UsdGeom.Mesh.Define(stage, "/World/Knife")
        vertices, faces = wedge_mesh(self.core.material_config.blade_length,
                                     self.core.material_config.blade_height,
                                     self.core.material_config.blade_half_width)
        blade.CreatePointsAttr(vertices)
        blade.CreateFaceVertexCountsAttr([3]*len(faces))
        blade.CreateFaceVertexIndicesAttr(faces.ravel())
        blade.CreateDisplayColorAttr([Gf.Vec3f(.38, .65, .85)])
        blade.CreateSubdivisionSchemeAttr("none")
        blade.CreateDoubleSidedAttr(True)
        self._knife_op = UsdGeom.Xformable(blade).AddTransformOp()
        self._body_ops = []
        robot = self.core.robot
        for body, label in enumerate(robot.model.body_label):
            source_path = label.split("/", 1)[1]
            arm = body//robot.bodies_per_arm
            path = f"/World/Franka_{arm}"+source_path[len("/panda"):]
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Robot geometry/link mapping is missing: {path}")
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            self._body_ops.append((body, xf.AddTransformOp()))
            xf.SetResetXformStack(True)
        self.stage = stage
        self._bridge = DynamicSurfaceBridge(stage)
        paths = {"d455_0": "/World/D455_0/RSD455/Camera_Pseudo_Depth",
                 "d455_1": "/World/D455_1/RSD455/Camera_Pseudo_Depth", "global": "/World/GlobalCamera"}
        self.calibration = {name: usd_camera_calibration(stage, path, c.width, c.height,
            near_m=.52 if name != "global" else .01, far_m=6.0 if name != "global" else 20.)
            for name, path in paths.items()}
        self.scene_source = str(source.resolve())

    def reset(self, seed=42):
        if self._closed:
            raise RuntimeError("Environment is closed")
        from .coupled_robot import CoupledCutting, CouplingConfig
        from .cutting_surface import ComponentSurface
        self.seed = int(seed)
        c = self.config
        self.core = CoupledCutting(CouplingConfig(control_dt=c.control_dt, material_dt=c.material_dt,
                 dx=c.dx, device=c.device, feedback=c.feedback, variant=c.variant, family=c.family))
        _, setup = self.core.reset(self.seed)
        if self._temp is not None:
            self._temp.cleanup(); self._temp = None
        self._create_stage()
        self._surface = ComponentSurface(c.dx/2, c.device, voxel_size=min(c.dx/2, .0015))
        self._steps = 0; self._surface_time = None; self._last_obs = None
        observation = self.observe()
        return observation, {"seed": self.seed, "config": asdict(c), "backend": "standalone_panda_tlmpm",
                             "camera_backend": "full_usd_ideal_warp_rgbd", "setup": setup,
                             "capabilities": self.capabilities()}

    @staticmethod
    def capabilities():
        return {"articulated_robots": True, "persistent_cutting": "experimental_online_TLMPM",
                "material_reaction_feedback": "one_interval_explicit_delay",
                "ideal_rgbd": True, "simulated_d455_validated": False,
                "physics_calibrated": False, "policy_training": False}

    def step(self, action=None, *, cut_index=None):
        if self.core is None or self._closed:
            raise RuntimeError("Call reset before step")
        if self._steps >= self.config.max_steps:
            raise RuntimeError("Episode truncated; call reset")
        _, diagnostics = self.core.step(action, cut_index=cut_index)
        q, v = self.core.bed.x.numpy(), self.core.bed.v.numpy()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise FloatingPointError("Nonfinite material state")
        self._steps += 1
        tick = time.perf_counter()
        observation = self.observe()
        info = {"physics_wall_s": diagnostics["material_wall_s"]+diagnostics["physics_wall_s"],
                "ik_wall_s": diagnostics["ik_wall_s"], "camera_wall_s": time.perf_counter()-tick,
                "capabilities": self.capabilities()}
        return observation, 0., False, self._steps >= self.config.max_steps, info

    def observe(self):
        if self.core is None or self._closed:
            raise RuntimeError("Call reset before observe")
        if self._surface_time == self.core.time_s:
            return self._last_obs
        import warp as wp
        from pxr import Gf
        from scipy.spatial.transform import Rotation
        from .camera import capture
        from .sensor_contract import validate_frame
        from .usd_surface_bridge import stage_triangles
        timestamp = self.core.time_s
        vertices, faces, self._surface_metadata = self._surface.extract(self.core.bed)
        self._bridge.update(vertices, faces, timestamp)
        def matrix(pose):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
            transform[:3, 3] = pose[:3]
            return Gf.Matrix4d(transform.T.tolist())
        for body, op in self._body_ops:
            op.Set(matrix(self.core.robot.state.body_q.numpy()[body]))
        proprio = self.core.robot.proprioception()
        self._knife_op.Set(matrix(proprio["tool_pose"]))
        vertices, faces, colors, _ = stage_triangles(self.stage)
        frames = {}
        with wp.ScopedDevice(self.config.device):
            verts = wp.array(vertices, dtype=wp.vec3)
            tris = wp.array(faces.ravel(), dtype=int)
            face_colors = wp.array(colors, dtype=wp.vec3)
            for name, calibration in self.calibration.items():
                rgb, depth = capture(verts, tris, calibration, self.config.device, face_colors)
                frames[name] = validate_frame(rgb, depth, timestamp, timestamp)
        self._frames = frames
        self._surface_time = timestamp
        self._last_obs = vision_observation(frames, self.calibration, timestamp,
                          self.config.observation_profile, proprio)
        return self._last_obs

    def render(self):
        if self._last_obs is None or self._closed:
            raise RuntimeError("Call reset before render")
        return self._frames["global"]["rgb"].copy()

    def debug_material_state(self):
        if self.core is None or self._closed:
            raise RuntimeError("Call reset first")
        return {**self.core.debug_material_state(), "mass_kg": self.core.masses.copy(),
                "timestamp_s": self.core.time_s}

    def debug_diagnostics(self):
        if self.core is None or self._closed:
            raise RuntimeError("Call reset first")
        return {"coupling": self.core.last_diagnostics, "surface": self._surface_metadata}

    def close(self):
        self.core = None; self.stage = None; self._last_obs = None
        self._surface = None; self._bridge = None; self._frames = None; self._body_ops = None
        if self._temp is not None:
            self._temp.cleanup(); self._temp = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
