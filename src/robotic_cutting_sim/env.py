"""Small research environment: real MPM, scripted/direct blade, live ideal RGB-D.

The separate USD workcell includes robot/sensor assets. This process advances a
kinematic blade and material; it does not claim articulated robot or D455 RTX physics.
"""
from dataclasses import asdict,dataclass
import time

import numpy as np
import yaml

from .geometry import Specimen,sample_particles
from .recording import ROOT
from .trajectory import Trajectory,reference_trajectory


def build_observation(rgbd,calibration,time_s,profile="vision_only",proprio=None):
    if profile not in ("vision_only","vision_proprio"):
        raise ValueError("Profile must be vision_only or vision_proprio")
    names={"d455_0":"ideal_0","d455_1":"ideal_1"}
    result={key:{out:rgbd[source][key] for out,source in names.items()}
            for key in ["rgb","depth_m","valid_mask"]}
    result["calibration"]={out:calibration[source] for out,source in names.items()}
    result["timestamp_s"]=float(time_s)
    if profile=="vision_proprio":
        if proprio is None:
            raise ValueError("vision_proprio requires an explicit real proprioception provider")
        result["proprio"]={key:proprio[key] for key in ["q","qdot","tool_pose"] if key in proprio}
    return result


@dataclass
class EnvConfig:
    case: str = "indent_unload"
    variant: str = "D0"
    voxel_size: float = .004
    control_dt: float = 1/60
    substeps: int = 4
    max_steps: int = 120
    device: str = "cuda:0"
    observation_profile: str = "vision_only"
    material_path: str = str(ROOT/"configs/materials/plasticine_provisional.yaml")

    def __post_init__(self):
        if self.substeps<1 or self.max_steps<1 or min(self.voxel_size,self.control_dt)<=0:
            raise ValueError("Positive timestep, voxel size and step counts required")


class CuttingEnv:
    """Facade for future researchers; no policy/training dependency.

    Action is None for the reference script, or target tool-center [xyz,qxyzw]
    in world/SI units reached over the next control step. Reward is always0;
    research scoring is not implemented. Global rendering is debug-only.
    """
    def __init__(self,config=None,proprio_provider=None):
        self.config=config or EnvConfig()
        if self.config.observation_profile=="vision_proprio" and proprio_provider is None:
            raise ValueError("vision_proprio requires a proprioception provider; robot states are not fabricated")
        self.proprio_provider=proprio_provider
        self.bed=None; self.cameras=None; self._closed=False; self._steps=0; self._last_rgbd=None

    def reset(self,seed=42):
        if self._closed:
            raise RuntimeError("Environment is closed")
        from .newton_probe import NewtonTestbed
        from .live_camera import LiveCameras
        spec=Specimen.generate(seed,self.config.variant)
        material=yaml.safe_load(__import__("pathlib").Path(self.config.material_path).read_text())
        spacing=self.config.voxel_size/2
        points,mass,_=sample_particles(spec,spacing,material["density_kg_m3"])
        self.bed=NewtonTestbed(points,mass,spacing,self.config.voxel_size,self.config.case,material,self.config.device)
        self.trajectory=reference_trajectory(self.config.case)
        self.time_s=0.0; self._steps=0; self.seed=seed
        self.tool_pose=self.trajectory.at(0)[0]
        for state in (self.bed.state,self.bed.next_state):
            state.body_q.assign(np.array([self.tool_pose],dtype=np.float32))
            state.body_qd.zero_()
        self.cameras=LiveCameras(spacing,self.config.device,self.bed.fixture_boxes)
        obs=self.observe()
        return obs,{"seed":seed,"particle_count":len(points),"initial_mass_kg":float(mass.sum()),
                    "backend":"newton_implicit_mpm","camera_backend":"ideal_warp_ray_camera",
                    "robot_dynamics":False,"persistent_cutting":False,"config":asdict(self.config)}

    def step(self,action=None):
        if self.bed is None or self._closed:
            raise RuntimeError("Call reset before step")
        if self._steps>=self.config.max_steps:
            raise RuntimeError("Episode truncated; call reset")
        dt=self.config.control_dt; subdt=dt/self.config.substeps
        if action is not None:
            action=np.asarray(action,dtype=float)
            if action.shape!=(7,) or not np.isfinite(action).all():
                raise ValueError("Action must be finite xyz+unit quaternion XYZW, shape(7,)")
            player=Trajectory(np.array([[self.time_s,*self.tool_pose],[self.time_s+dt,*action]]))
        else:
            player=self.trajectory
        tick=time.perf_counter()
        max_correction=0.0
        for sub in range(self.config.substeps):
            pose,twist=player.at(self.time_s+sub*subdt)
            _,diagnostics=self.bed.step(subdt,pose,twist)
            max_correction=max(max_correction,diagnostics["max_projection_correction_m"])
        self.time_s+=dt; self._steps+=1; self.tool_pose=player.at(self.time_s)[0]
        q,v=self.bed.snapshot()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise FloatingPointError("Nonfinite MPM state")
        physics_wall=time.perf_counter()-tick
        tick=time.perf_counter(); obs=self.observe(); render_wall=time.perf_counter()-tick
        return obs,0.0,False,self._steps>=self.config.max_steps,{
            "physics_wall_s":physics_wall,"render_wall_s":render_wall,
            "max_projection_correction_m":max_correction,
            "capabilities":{"material_deformation":True,"persistent_cutting":False,
                            "ideal_rgbd":True,"d455_rtx":False,"articulated_robots":False}}

    def observe(self):
        if self.bed is None or self._closed:
            raise RuntimeError("Call reset before observe")
        self._last_rgbd=self.cameras.capture(self.bed.state.particle_q,self.bed.model.particle_radius,
                                            self.tool_pose,self.bed.half_extents)
        proprio=self.proprio_provider() if self.proprio_provider else None
        return build_observation(self._last_rgbd,self.cameras.calibration,self.time_s,
                                 self.config.observation_profile,proprio)

    def render(self):
        if self._last_rgbd is None:
            raise RuntimeError("Call reset before render")
        return self._last_rgbd["global"]["rgb"].copy()

    def debug_material_state(self):
        """Explicit oracle API, kept outside policy observation."""
        if self.bed is None or self._closed:
            raise RuntimeError("Call reset first")
        q,v=self.bed.snapshot()
        return {"positions_m":q,"velocities_m_s":v,"mass_kg":self.bed.model.particle_mass.numpy(),"timestamp_s":self.time_s}

    def close(self):
        self.bed=None; self.cameras=None; self._last_rgbd=None; self._closed=True

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
