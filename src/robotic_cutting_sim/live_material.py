"""Retained TLMPM material in a live external articulation loop.

This adapter owns no robot. Its input is the measured fixed blade-edge pose at
each control boundary; its output is the next-interval reaction wrench. The
caller advances its actual Isaac articulation and applies that wrench. The
20 ms explicit coupling delay is identical to the standalone reference.
"""
from dataclasses import asdict
import time

import numpy as np

from .coupled_robot import CouplingConfig, active_cut_index, interpolate_measured_poses, material_recipe
from .geometry import Specimen, sample_particles
from .robot import validate_pose
from .scene_frames import WorkpieceTarget


class LiveCuttingMaterial:
    def __init__(self, config=None, target=None):
        self.config = config or CouplingConfig()
        self.target = target or WorkpieceTarget()
        # The retained floor/tray implementation is in the world identity frame.
        # Reject other workpiece poses until the fixture and gravity are also transformed.
        if not np.allclose(self.target.world_from_workpiece_xyzw, [0, 0, 0, 0, 0, 0, 1], atol=1e-12):
            raise NotImplementedError("Live TLMPM currently requires an identity workpiece pose including its tray")
        if self.config.variant not in ("D0", "D1") or self.config.benchmark_spec is not None:
            raise ValueError("Live Isaac integration validates the retained D0/D1 recipe first")
        self.bed = None

    def reset(self, measured_edge_pose, seed=42):
        import warp as wp
        from .tlmpm import TLMPM

        start = time.perf_counter()
        c = self.config
        self.seed = int(seed)
        self.last_pose = validate_pose(measured_edge_pose)
        self.substeps = round(c.control_dt / c.material_dt)
        self.material_config, self.parameter_receipt = material_recipe(c, None)
        self.specimen = Specimen.generate(self.seed, c.variant, c.family)
        self.points, self.masses, self.lattice = sample_particles(
            self.specimen, c.dx/2, self.material_config.density)
        self.bed = TLMPM(self.points, self.masses, c.dx/2, self.material_config, c.device)
        self.bed.set_trajectory(np.tile(self.last_pose, (self.substeps+1, 1)).astype(np.float32),
            np.zeros((self.substeps+1, 6), np.float32), np.full(self.substeps+1, -1, np.int32))
        wp.load_module(module="robotic_cutting_sim.tlmpm", device=c.device)
        with wp.ScopedCapture(device=c.device) as capture:
            for _ in range(self.substeps):
                self.bed.step()
        self.graph = capture.graph
        self.pending_wrench = np.zeros(6)
        self.time_s = 0.
        self.last_diagnostics = {}
        return {"seed": self.seed, "material": asdict(self.material_config),
            "parameter_receipt": self.parameter_receipt, "particles": len(self.points),
            "initial_mass_kg": float(self.masses.sum()), "initial_cut_codes": [0],
            "feedback_delay_s": c.control_dt, "reset_wall_s": time.perf_counter()-start,
            "workpiece_local_target_height_m": self.target.height_m,
            "world_from_workpiece_xyzw": list(self.target.world_from_workpiece_xyzw)}

    def advance(self, measured_edge_pose, time_s, *, cutting=True):
        import warp as wp

        if self.bed is None:
            raise RuntimeError("Reset material from the measured initial edge before advancing")
        c = self.config
        if not np.isclose(time_s-self.time_s, c.control_dt, atol=1e-8, rtol=0):
            raise ValueError("Robot/material timestamps must differ by exactly one control interval")
        actual = validate_pose(measured_edge_pose)
        cut = active_cut_index(self.time_s, cutting)
        poses, twists = interpolate_measured_poses(self.last_pose, actual, self.substeps, c.control_dt)
        self.bed.poses.assign(poses)
        self.bed.twists.assign(twists)
        self.bed.cuts.assign(np.full(self.substeps+1, cut, np.int32))
        self.bed.index.zero_()
        self.bed.reaction.zero_()
        self.bed.stats.zero_()
        start = time.perf_counter()
        wp.capture_launch(self.graph)
        wp.synchronize_device(c.device)
        impulse = self.bed.reaction.numpy()[0].astype(float)
        previous = self.pending_wrench.copy()
        self.pending_wrench = impulse/c.control_dt
        self.time_s = float(time_s)
        self.last_pose = actual.copy()
        self.last_diagnostics = {
            "timestamp_s": self.time_s, "measured_tool_pose": actual,
            "applied_previous_material_wrench_world": previous if c.feedback else np.zeros(6),
            "current_material_reaction_impulse_world": impulse,
            "next_material_wrench_world": self.pending_wrench.copy(),
            "material_contact_statistics": self.bed.stats.numpy().copy(),
            "feedback_enabled": c.feedback, "feedback_delay_s": c.control_dt,
            "material_wall_s": time.perf_counter()-start, "active_cut": cut >= 0, "cut_index": cut}
        return self.last_diagnostics.copy()

    def debug_state(self):
        """Oracle diagnostics only; never include this dictionary in observations."""
        return self.bed.snapshot()
