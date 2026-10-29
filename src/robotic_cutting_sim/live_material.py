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
    def __init__(self, config=None, target=None, *, task=None, tool_profile=None, cutting_law=None,
                 slice_efficiency=1., obstacle_boxes_edge=None, max_cuts=1):
        """``tool_profile`` (tool_profile.ToolProfile) selects the tlmpm_v2 solver
        with that blade polygon; None keeps the frozen v1 solver and task wedge.

        v2 tasks (tasks.ProfileTaskSpec) default to their profile, cutting law 2
        and the handle as a blunt obstacle. ``max_cuts`` (1 or 2) sets how many
        cut digits the material keeps; it requires tlmpm_v2 above 1.
        """
        from .tasks import ProfileTaskSpec
        from .tool_profile import handle_box_edge
        self.config = config or CouplingConfig()
        if isinstance(task, ProfileTaskSpec):
            tool_profile = tool_profile or task.tool_profile
            cutting_law = 2 if cutting_law is None else cutting_law
            if obstacle_boxes_edge is None:
                obstacle_boxes_edge = [handle_box_edge(tool_profile)]
        if max_cuts not in (1, 2) or (max_cuts == 2 and tool_profile is None):
            raise ValueError("max_cuts is 1 or 2; two cut digits require a tool profile (tlmpm_v2)")
        if tool_profile is None and (cutting_law not in (None, 1) or obstacle_boxes_edge is not None):
            raise ValueError("Cutting law 2 and tool obstacles require a tool profile (tlmpm_v2)")
        self.tool_profile = tool_profile
        self.cutting_law = 1 if cutting_law is None else int(cutting_law)
        self.slice_efficiency, self.max_cuts = float(slice_efficiency), int(max_cuts)
        self.obstacle_boxes_edge = obstacle_boxes_edge
        self.obstacle_reaction_recipients = None
        self.pending_obstacle_wrenches = {}
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

    def configure_pad_obstacles(self, boxes, recipients):
        """Configure before reset; handle reaction stays on tool, pads on fingers."""
        if self.tool_profile is None:
            raise ValueError('Finger obstacles are opt-in for profile tasks only')
        boxes = np.asarray(boxes, np.float32)
        if boxes.shape != (3, 2, 3) or len(recipients) != 3 or recipients[0] is not None:
            raise ValueError('Expected one handle and two measured pad boxes')
        if not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= 0):
            raise ValueError('Pad bounds must be finite and nonempty')
        self.obstacle_boxes_edge = boxes
        self.obstacle_reaction_recipients = tuple(recipients)

    def update_pad_obstacles(self, boxes):
        boxes = np.asarray(boxes, np.float32)
        if self.obstacle_reaction_recipients is None or boxes.shape != (3, 2, 3):
            raise ValueError('Configure the actual pad recipients before updating geometry')
        if not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= 0):
            raise ValueError('Pad bounds must be finite and nonempty')
        self.bed.box_centers.assign(boxes[:, 0].copy())
        self.bed.box_halves.assign(boxes[:, 1].copy())

    def reset(self, measured_edge_pose, seed=42):
        import warp as wp
        if self.tool_profile is None:
            from .tlmpm import TLMPM
            solver_module, solver_kwargs = "robotic_cutting_sim.tlmpm", {}
        else:
            if self.cutting_law in (4, 5, 6, 7, 8):
                from .tlmpm_v4 import TLMPM
            elif self.cutting_law == 3:
                from .tlmpm_v3 import TLMPM
            else:
                from .tlmpm_v2 import TLMPM
            polygon, _ = self.tool_profile.tlmpm_profile()
            solver_module, solver_kwargs = TLMPM.__module__, {"blade_profile_xz": polygon,
                "cutting_law": self.cutting_law, "slice_efficiency": self.slice_efficiency,
                "obstacle_boxes_edge": self.obstacle_boxes_edge}

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
        if self.tool_profile is not None:
            profile = self.tool_profile
            self.material_config = replace(self.material_config, blade_length=profile.blade_depth_m,
                blade_height=profile.thickness_m, blade_half_width=profile.half_width_m, cuts=self.max_cuts)
            self.parameter_receipt = {**self.parameter_receipt, "solver": solver_module.rsplit('.',1)[-1], "tool_profile": profile.to_dict(),
                "cutting_law": self.cutting_law, "slice_efficiency": self.slice_efficiency, "max_cuts": self.max_cuts,
                "obstacle_boxes_edge": None if self.obstacle_boxes_edge is None
                    else np.asarray(self.obstacle_boxes_edge, float).tolist(),
                'obstacle_reaction_recipients': self.obstacle_reaction_recipients}
        self.bed = TLMPM(self.points, self.masses, self.spacing, self.material_config, c.device,
                         **fixture_kwargs, **solver_kwargs)
        if self.cutting_law in (4, 5, 6, 7, 8):
            self.parameter_receipt = {**self.parameter_receipt, "solver_module": solver_module,
                                      "law4_parameters": dict(self.bed.law4_parameters),
                                      "law4_reference": "docs/adr/0006-swept-band-cutting-law-4.md"}
        if self.cutting_law == 5:
            self.parameter_receipt = {**self.parameter_receipt, "law5_parameters": dict(self.bed.law5_parameters),
                                      "law5_reference": "docs/adr/0008-cutting-law-5-lead-preassignment.md"}
        if self.cutting_law == 6:
            self.parameter_receipt = {**self.parameter_receipt, "law6_parameters": dict(self.bed.law6_parameters),
                                      "law6_reference": "docs/adr/0010-cutting-law-6-deferred-lead-commit.md"}
        if self.cutting_law == 7:
            self.parameter_receipt = {**self.parameter_receipt, "law7_parameters": dict(self.bed.law7_parameters),
                                      "law7_reference": "docs/adr/0011-cutting-law-7-eq8-coupled-flip.md"}
        if self.cutting_law == 8:
            self.parameter_receipt = {**self.parameter_receipt, "law8_parameters": dict(self.bed.law8_parameters),
                                      "law8_reference": "docs/adr/0012-cutting-law-8-plow-gated-lead.md"}
        self.bed.set_trajectory(np.tile(self._pose_local(self.last_pose), (self.substeps+1, 1)).astype(np.float32),
            np.zeros((self.substeps+1, 6), np.float32), np.full(self.substeps+1, -1, np.int32))
        wp.load_module(module=solver_module, device=c.device)
        if self.task:
            wp.load_module(module="robotic_cutting_sim.fixture_contacts", device=c.device)
        if wp.get_device(c.device).is_cuda:
            with wp.ScopedCapture(device=c.device) as capture:
                for _ in range(self.substeps):
                    self.bed.step()
            self.graph = capture.graph
        else:
            self.graph = None  # CPU: no graph capture; advance() replays the substeps.
        self.pending_wrench = np.zeros(6)
        self.pending_obstacle_wrenches = {}
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
        if self.tool_profile is not None:
            self.bed.box_reactions.zero_()
        self.bed.stats.zero_()
        start = time.perf_counter()
        if self.graph is None:
            for _ in range(self.substeps):
                self.bed.step()
        else:
            wp.capture_launch(self.graph)
        wp.synchronize_device(c.device)
        impulse = self.bed.reaction.numpy()[0].astype(float)
        if self.task:
            impulse = rotate_wrench_at_same_point(self.T_world_local, impulse)
        total_impulse = impulse.copy()
        obstacle_wrenches = {}
        if self.obstacle_reaction_recipients is not None:
            for recipient, box_impulse in zip(self.obstacle_reaction_recipients, self.bed.box_reactions.numpy()):
                if recipient is not None:
                    value = box_impulse.astype(float)
                    if self.task:
                        value = rotate_wrench_at_same_point(self.T_world_local, value)
                    impulse -= value
                    obstacle_wrenches[recipient] = value / c.control_dt
        previous = self.pending_wrench.copy()
        self.pending_wrench = impulse/c.control_dt
        self.pending_obstacle_wrenches = obstacle_wrenches
        self.time_s = float(time_s)
        self.last_pose = actual.copy()
        self.last_diagnostics = {
            "timestamp_s": self.time_s, "measured_tool_pose": actual,
            "applied_previous_material_wrench_world": previous if c.feedback else np.zeros(6),
            "current_material_reaction_impulse_world": impulse,
            'current_total_material_reaction_impulse_world': total_impulse,
            'next_finger_wrenches_about_edge_world': obstacle_wrenches,
            "next_material_wrench_world": self.pending_wrench.copy(),
            "material_contact_statistics": self.bed.stats.numpy().copy(),
            "feedback_enabled": c.feedback, "feedback_delay_s": c.control_dt,
            "material_wall_s": time.perf_counter()-start, "active_cut": cut >= 0, "cut_index": cut}
        return self.last_diagnostics.copy()

    def cutting_state(self):
        """Law-4 stroke state after the last substep (oracle only, never observed)."""
        return self.bed.cutting_state() if hasattr(self.bed, 'cutting_state') else {'cutting_law': self.cutting_law}

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

    def history_state(self):
        """Same recorded positions/velocities/codes without copying unused F tensors."""
        state = dict(positions=self.bed.x.numpy(), velocities=self.bed.v.numpy(), codes=self.bed.code.numpy())
        if self.task:
            state['positions'] = transform_points(self.T_world_local, state['positions'])
            state['velocities'] = state['velocities'] @ self.T_world_local[:3, :3].T
        return state
