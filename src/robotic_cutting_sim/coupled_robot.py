"""Experimental staggered MuJoCo Panda / persistent-cut TLMPM coupling.

Measured robot knife poses drive material contact. Equal-opposite TLMPM impulses
become a world wrench on the articulated hand in the next control interval.
That explicit one-interval delay is recorded, not hidden as monolithic coupling.
All force and material state remain separate from robot proprioception.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .geometry import Specimen, sample_particles
from .robot import DualFrankaController, RobotConfig, validate_pose
from .tlmpm import TLMPM, TLMPMConfig


@dataclass(frozen=True)
class CouplingConfig:
    control_dt: float = .02
    material_dt: float = .0001
    dx: float = .004
    device: str = "cuda:0"
    feedback: bool = True
    variant: str = "D0"
    family: str = "dome"
    cut_height_m: float = .0155

    def __post_init__(self):
        if not np.isfinite(self.control_dt) or not np.isfinite(self.material_dt) or min(self.control_dt, self.material_dt) <= 0:
            raise ValueError("Positive finite timesteps are required")
        substeps = round(self.control_dt / self.material_dt)
        if substeps < 1 or abs(substeps * self.material_dt - self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of material_dt")


def reference_pose(time_s, cut_height=.0155):
    """Slow 8 s edge path, with a settling hold before the active cut."""
    knots = np.array([[0, -.030, .060], [1., -.030, cut_height],
                      [1.6, -.030, cut_height], [5.6, .045, cut_height],
                      [6.4, .080, .060], [8., .080, .060]])
    return np.array([np.interp(time_s, knots[:, 0], knots[:, 1]), 0.,
                     np.interp(time_s, knots[:, 0], knots[:, 2]), 0., 0., 0., 1.])


def interpolate_measured_poses(previous, current, substeps, control_dt):
    previous, current = validate_pose(previous), validate_pose(current)
    fraction = np.arange(substeps + 1) / substeps
    rotations = Rotation.from_quat(np.array([previous[3:], current[3:]]))
    quaternions = Slerp([0., 1.], rotations)(fraction).as_quat()
    xyz = previous[:3] + fraction[:, None] * (current[:3] - previous[:3])
    linear = (current[:3] - previous[:3]) / control_dt
    angular = (rotations[1] * rotations[0].inv()).as_rotvec() / control_dt
    return np.c_[xyz, quaternions].astype(np.float32), np.tile(np.r_[linear, angular], (substeps + 1, 1)).astype(np.float32)


class CoupledCutting:
    """Minimal independently executable coupling testbed; cameras are external."""

    def __init__(self, config=None):
        self.config = config or CouplingConfig()
        self.robot = None
        self.bed = None

    def reset(self, seed=42):
        import warp as wp
        c = self.config
        start = time.perf_counter()
        self.substeps = round(c.control_dt / c.material_dt)
        self.material_config = TLMPMConfig(dx=c.dx, dt=c.material_dt, cuts=1)
        robot_config = RobotConfig(control_dt=c.control_dt, substeps=10,
            tool_shape="cut_wedge", tool_rotation_xyzw=(1., 0., 0., 0.),
            wedge_length_m=self.material_config.blade_length,
            wedge_height_m=self.material_config.blade_height,
            wedge_half_width_m=self.material_config.blade_half_width)
        self.robot = DualFrankaController(robot_config)
        first = reference_pose(0, c.cut_height_m)
        joints, info = self.robot.inverse_kinematics(first)
        if not info["converged"]:
            raise RuntimeError(f"Initial edge pose is unreachable: {info}")
        proprio = self.robot.reset(joints)
        self.last_tool_pose = proprio["tool_pose"].copy()
        self.specimen = Specimen.generate(seed, c.variant, c.family)
        self.points, self.masses, self.lattice = sample_particles(self.specimen, c.dx / 2,
                                                                self.material_config.density)
        self.bed = TLMPM(self.points, self.masses, c.dx / 2, self.material_config, c.device)
        poses = np.tile(self.last_tool_pose, (self.substeps + 1, 1)).astype(np.float32)
        self.bed.set_trajectory(poses, np.zeros((self.substeps + 1, 6), np.float32),
                                np.full(self.substeps + 1, -1, np.int32))
        wp.load_module(module="robotic_cutting_sim.tlmpm", device=c.device)
        with wp.ScopedCapture(device=c.device) as capture:
            for _ in range(self.substeps):
                self.bed.step()
        self.graph = capture.graph
        self.time_s = 0.
        self.pending_wrench_world = np.zeros(6)
        self.last_diagnostics = {}
        self.seed = seed
        return proprio, {"setup_wall_s": time.perf_counter() - start,
                         "config": asdict(c), "material": asdict(self.material_config),
                         "robot": self.robot.description(),
                         "feedback_delay_s": c.control_dt, "initial_ik": info}

    def step(self, target_tool_pose=None, *, cutting=True):
        import warp as wp
        if self.bed is None:
            raise RuntimeError("Call reset before step")
        c = self.config
        target_time = self.time_s + c.control_dt
        desired = reference_pose(target_time, c.cut_height_m) if target_tool_pose is None else validate_pose(target_tool_pose)
        applied = self.pending_wrench_world.copy() if c.feedback else np.zeros(6)
        proprio, robot_info = self.robot.step_cartesian(desired, tool_wrench_world=applied)
        actual = proprio["tool_pose"]
        poses, twists = interpolate_measured_poses(self.last_tool_pose, actual, self.substeps, c.control_dt)
        active = cutting and self.time_s >= 1.6 - 1e-8 and self.time_s < 5.6 - 1e-8
        self.bed.poses.assign(poses)
        self.bed.twists.assign(twists)
        self.bed.cuts.assign(np.full(self.substeps + 1, 0 if active else -1, np.int32))
        self.bed.index.zero_()
        self.bed.reaction.zero_()
        self.bed.stats.zero_()
        start = time.perf_counter()
        wp.capture_launch(self.graph)
        wp.synchronize_device(c.device)
        material_wall = time.perf_counter() - start
        reaction_impulse = self.bed.reaction.numpy()[0].astype(float)
        self.pending_wrench_world = reaction_impulse / c.control_dt
        self.last_tool_pose = actual.copy()
        self.time_s = target_time
        self.last_diagnostics = {
            "timestamp_s": self.time_s, "robot_timestamp_s": proprio["timestamp_s"],
            "desired_tool_pose": desired, "measured_tool_pose": actual,
            "applied_previous_material_wrench_world": applied,
            "current_material_reaction_impulse_world": reaction_impulse,
            "next_material_wrench_world": self.pending_wrench_world.copy(),
            "material_contact_statistics": self.bed.stats.numpy().copy(),
            "feedback_enabled": c.feedback, "feedback_delay_s": c.control_dt,
            "material_wall_s": material_wall, **robot_info,
            "active_cut": active,
        }
        return proprio, self.last_diagnostics.copy()

    def debug_material_state(self):
        """Oracle diagnostic state, kept out of returned proprioception."""
        return {key: value.copy() for key, value in self.bed.snapshot().items()}
