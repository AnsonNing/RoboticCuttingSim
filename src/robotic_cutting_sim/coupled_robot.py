"""Experimental staggered MuJoCo Panda / persistent-cut TLMPM coupling.

Measured robot knife poses drive material contact. Equal-opposite TLMPM impulses
become a world wrench on the articulated hand in the next control interval.
That explicit one-interval delay is recorded, not hidden as monolithic coupling.
All force and material state remain separate from robot proprioception.
"""
from dataclasses import asdict, dataclass, replace
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .benchmark import BenchmarkSpec, sample_specimen
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
    benchmark_spec: BenchmarkSpec | None = None

    def __post_init__(self):
        if not np.isfinite(self.control_dt) or not np.isfinite(self.material_dt) or min(self.control_dt, self.material_dt) <= 0:
            raise ValueError("Positive finite timesteps are required")
        substeps = round(self.control_dt / self.material_dt)
        if substeps < 1 or abs(substeps * self.material_dt - self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of material_dt")
        if self.variant not in ("D0", "D1", "D2", "D3"):
            raise ValueError("variant must be D0, D1, D2 or D3")


def material_recipe(config, benchmark_spec):
    """Map an optional explicit recipe and record every unsupported request.

    No recipe retains the original TLMPM material, including friction=.3.
    Supplying BenchmarkSpec deliberately applies its supported values; its default
    friction=.4 is therefore an explicit new experiment rather than a hidden
    change to the frozen legacy D0 test.
    """
    material = TLMPMConfig(dx=config.dx, dt=config.material_dt, cuts=1)
    receipt = {"source": "legacy_TLMPM_defaults" if benchmark_spec is None else "BenchmarkSpec",
               "applied": {}, "unapplied": {}, "deferred_to_sensor_adapter": {},
               "calibration": "provisional_uncalibrated"}
    if benchmark_spec is not None:
        recipe = benchmark_spec.material
        mapping = {"density_kg_m3": "density", "young_modulus_pa": "young",
                   "poisson_ratio": "poisson", "yield_stress_pa": "yield_stress",
                   "friction": "contact_friction"}
        values = {target: getattr(recipe, source) for source, target in mapping.items()}
        if recipe.fracture_energy_j_m2 is not None:
            mapping["fracture_energy_j_m2"] = "cutting_energy"
            values["cutting_energy"] = recipe.fracture_energy_j_m2
        material = replace(material, **values)
        receipt["applied"] = {source: {"backend_field": target, "value": getattr(material, target)}
                              for source, target in mapping.items()}
        receipt["unapplied"] = {
            "hardening_ratio": {"requested": recipe.hardening_ratio, "status": "UNSUPPORTED",
                "reason": "TLMPM uses perfect Hencky J2 plasticity without hardening"},
            "critical_stress_pa": {"requested": recipe.critical_stress_pa, "status": "UNSUPPORTED",
                "reason": "The retained cut law is edge work, not critical-stress failure"},
            "failure_model": {"requested": recipe.failure_model, "status": "NOT_APPLIED",
                "reason": "Actual backend is experimental edge-work activation plus persistent logical-grid cuts"},
            "blade_half_thickness_m": {"requested": benchmark_spec.blade_half_thickness_m,
                "status": "UNSUPPORTED", "reason": "A rectangular blade half-thickness has no exact triangular-wedge counterpart"},
        }
        receipt["deferred_to_sensor_adapter"] = {
            "camera_translation_m": benchmark_spec.camera_translation_m,
            "camera_rotation_deg": benchmark_spec.camera_rotation_deg,
            "status": "NOT_APPLIED_BY_PHYSICS_ADAPTER"}
        receipt["calibration"] = recipe.calibration
    receipt["actual_backend_material"] = asdict(material)
    return material, receipt


def active_cut_index(time_s, cutting=True, cut_index=None):
    """A caller may explicitly enable the retained cut digit outside the script."""
    if cut_index is not None and cut_index not in (-1, 0):
        raise ValueError("This coupling adapter retains cut digit0; cut_index must be -1,0 or None")
    if not cutting:
        return -1
    if cut_index is not None:
        return cut_index
    return 0 if 1.6 - 1e-8 <= time_s < 5.6 - 1e-8 else -1


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
        self.benchmark_spec = c.benchmark_spec
        if self.benchmark_spec is None and c.variant in ("D2", "D3"):
            self.benchmark_spec = BenchmarkSpec.generate(seed, c.variant)
        if self.benchmark_spec is not None and self.benchmark_spec.seed != seed:
            raise ValueError("reset seed must match an explicitly supplied BenchmarkSpec.seed")
        self.material_config, self.parameter_receipt = material_recipe(c, self.benchmark_spec)
        self.parameter_receipt["scenario_selection"] = ("provided_BenchmarkSpec" if c.benchmark_spec is not None
            else "auto_generated_BenchmarkSpec" if self.benchmark_spec is not None else "legacy_Specimen")
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
        if self.benchmark_spec is None:
            self.specimen = Specimen.generate(seed, c.variant, c.family)
            self.points, self.masses, self.lattice = sample_particles(self.specimen, c.dx / 2,
                                                                    self.material_config.density)
            self.sampling_receipt = {"source": "legacy_single_bump_Specimen", "variant": c.variant,
                                     "seed": seed, "family": c.family, "preassigned_components": False}
        else:
            self.specimen = self.benchmark_spec.heightfield
            self.points, self.masses, self.lattice, self.sampling_receipt = sample_specimen(self.benchmark_spec, c.dx / 2)
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
                         "parameter_receipt": self.parameter_receipt,
                         "sampling_receipt": self.sampling_receipt,
                         "benchmark_spec": None if self.benchmark_spec is None else self.benchmark_spec.to_dict(),
                         "feedback_delay_s": c.control_dt, "initial_ik": info}

    def step(self, target_tool_pose=None, *, cutting=True, cut_index=None):
        import warp as wp
        if self.bed is None:
            raise RuntimeError("Call reset before step")
        c = self.config
        current_cut = active_cut_index(self.time_s, cutting, cut_index)
        target_time = self.time_s + c.control_dt
        desired = reference_pose(target_time, c.cut_height_m) if target_tool_pose is None else validate_pose(target_tool_pose)
        applied = self.pending_wrench_world.copy() if c.feedback else np.zeros(6)
        proprio, robot_info = self.robot.step_cartesian(desired, tool_wrench_world=applied)
        actual = proprio["tool_pose"]
        poses, twists = interpolate_measured_poses(self.last_tool_pose, actual, self.substeps, c.control_dt)
        self.bed.poses.assign(poses)
        self.bed.twists.assign(twists)
        self.bed.cuts.assign(np.full(self.substeps + 1, current_cut, np.int32))
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
            "active_cut": current_cut >= 0, "cut_index": current_cut,
        }
        return proprio, self.last_diagnostics.copy()

    def debug_material_state(self):
        """Oracle diagnostic state, kept out of returned proprioception."""
        return {key: value.copy() for key, value in self.bed.snapshot().items()}
