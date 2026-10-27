"""Retained TLMPM material in a live external articulation loop.

This adapter owns no robot. Its input is the measured fixed blade-edge pose at
each control boundary; its output is the next-interval reaction wrench. The
caller advances its actual Isaac articulation and applies that wrench. The
20 ms explicit coupling delay is identical to the standalone reference.
"""
from dataclasses import asdict, replace
import time

import numpy as np

from .coupled_robot import CouplingConfig, active_cut_index, interpolate_measured_poses, material_recipe
from .geometry import Specimen, sample_particles
from .robot import validate_pose
from .scene_frames import (WorkpieceTarget, pose_matrix, invert_transform, transform_points,
    transform_poses, rotate_wrench_at_same_point)


class LiveCuttingMaterial:
    def __init__(self, config=None, target=None, *, task=None):
        self.config = config or CouplingConfig()
        self.target = target or WorkpieceTarget()
        self.task = task
        # The retained floor/tray implementation is in the world identity frame.
        # Reject other workpiece poses until the fixture and gravity are also transformed.
        if task is None and not np.allclose(self.target.world_from_workpiece_xyzw, [0, 0, 0, 0, 0, 0, 1], atol=1e-12):
            raise NotImplementedError("Live TLMPM currently requires an identity workpiece pose including its tray")
        if self.config.variant not in ("D0", "D1") or self.config.benchmark_spec is not None:
            raise ValueError("Live Isaac integration validates the retained D0/D1 recipe first")
        if task is not None and (not np.isclose(task.control_dt_s, self.config.control_dt)
                                 or not np.isclose(task.grid_dx_m, self.config.dx)):
            raise ValueError("Task and material must use the same control interval and reference grid")
        self.T_world_local = pose_matrix(task.world_from_workpiece_xyzw) if task else np.eye(4)
        self.T_local_world = invert_transform(self.T_world_local)
        self.spacing = task.particle_spacing_m if task else self.config.dx/2
        self.bed = None

    @property
    def world_points(self):
        """Reference particle locations in WORLD; solver points remain LOCAL."""
        return transform_points(self.T_world_local, self.points) if self.task else self.points.copy()

    def _pose_local(self, pose):
        return transform_poses(self.T_local_world, pose) if self.task else pose

    def reset(self, measured_edge_pose, seed=42):
        import warp as wp
        from .tlmpm import TLMPM

        start = time.perf_counter()
        c = self.config
        self.seed = int(seed)
        self.last_pose = validate_pose(measured_edge_pose)
        self.substeps = round(c.control_dt / c.material_dt)
        self.material_config, self.parameter_receipt = material_recipe(c, None)
        fixture_kwargs = {}
        if self.task:
            recipe = self.task.material
            if recipe.hardening_ratio != 0 or recipe.critical_stress_pa is not None:
                raise NotImplementedError("Retained cutting law supports perfect plasticity and edge work only")
            self.material_config = replace(self.material_config, density=recipe.density_kg_m3,
                young=recipe.young_modulus_pa, poisson=recipe.poisson_ratio,
                yield_stress=recipe.yield_stress_pa, contact_friction=recipe.friction,
                cutting_energy=recipe.fracture_energy_j_m2 or self.material_config.cutting_energy,
                blade_length=self.task.blade_length_m, blade_height=self.task.blade_height_m,
                blade_half_width=self.task.blade_half_width_m, floor=False, fixture_walls=False,
                gravity=tuple(self.T_local_world[:3, :3]@np.array([0., 0., -9.81])))
            self.parameter_receipt = {"source": "CuttingTaskSpec", "requested": asdict(recipe),
                "actual_backend_material": asdict(self.material_config),
                "calibration": recipe.calibration, "solver_coordinates": "static_workpiece_local",
                "moving_workpiece_frame_supported": False,
                "cutting_law": "retained experimental edge-work persistent logical-grid separation"}
            self.specimen = self.task.geometry
            self.points, self.masses, self.lattice = self.task.sample_particles()
            # World table z=top: n_local.x + (origin_world.z-top) >= 0.
            normal = self.T_world_local[2, :3]
            plane = np.r_[normal, self.T_world_local[2, 3]-self.task.table_top_world_z_m]
            boxes = [(box.center_m, box.half_extents_m) for box in self.task.fixture_boxes_local]
            fixture_kwargs = {"fixture_boxes": boxes, "table_plane": plane}
        else:
            self.specimen = Specimen.generate(self.seed, c.variant, c.family)
            self.points, self.masses, self.lattice = sample_particles(
                self.specimen, self.spacing, self.material_config.density)
        self.bed = TLMPM(self.points, self.masses, self.spacing, self.material_config, c.device, **fixture_kwargs)
        self.bed.set_trajectory(np.tile(self._pose_local(self.last_pose), (self.substeps+1, 1)).astype(np.float32),
            np.zeros((self.substeps+1, 6), np.float32), np.full(self.substeps+1, -1, np.int32))
        wp.load_module(module="robotic_cutting_sim.tlmpm", device=c.device)
        if self.task:
            wp.load_module(module="robotic_cutting_sim.fixture_contacts", device=c.device)
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
            "workpiece_local_target_height_m": (self.task.target_origin_local_m[2]
                if self.task and self.task.goal_kind == "flatten" else None if self.task else self.target.height_m),
            "world_from_workpiece_xyzw": list(self.task.world_from_workpiece_xyzw if self.task else self.target.world_from_workpiece_xyzw),
            "task_id": self.task.task_id if self.task else None, "particle_spacing_m": self.spacing}

    def advance(self, measured_edge_pose, time_s, *, cutting=True, cut_index=None):
        import warp as wp

        if self.bed is None:
            raise RuntimeError("Reset material from the measured initial edge before advancing")
        c = self.config
        if not np.isclose(time_s-self.time_s, c.control_dt, atol=1e-8, rtol=0):
            raise ValueError("Robot/material timestamps must differ by exactly one control interval")
        actual = validate_pose(measured_edge_pose)
        if cut_index is not None:
            if not isinstance(cut_index, (int, np.integer)) or not 0 <= cut_index < self.material_config.cuts:
                raise ValueError("Explicit cut index exceeds retained material history")
            cut = int(cut_index) if cutting else -1
        else:
            cut = self.task.active_cut_index(self.time_s) if self.task and cutting else active_cut_index(self.time_s, cutting)
        poses, twists = interpolate_measured_poses(self._pose_local(self.last_pose), self._pose_local(actual),
            self.substeps, c.control_dt)
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
        if self.task:
            impulse = rotate_wrench_at_same_point(self.T_world_local, impulse)
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
        """Solver-LOCAL oracle diagnostics; never include in observations."""
        return self.bed.snapshot()

    def world_state(self):
        """WORLD oracle diagnostics for recording/evaluation, separate from policy."""
        state = self.debug_state()
        if self.task:
            state["positions"] = transform_points(self.T_world_local, state["positions"])
            state["velocities"] = state["velocities"]@self.T_world_local[:3, :3].T
        return state
