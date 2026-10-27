"""Small shared task definitions for live CUT/TRIM adapters and offline scoring.

Sampling is LOCAL and initially unlabelled. Recorded evaluator coordinates are
WORLD. Oracle topology/surface metrics never become policy observations and do
not establish robot, camera, runtime or calibrated-fracture acceptance.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .benchmark import Bump, BenchmarkSpec, HeightFieldSpec, MaterialParameters, sample_specimen
from .cut_topology import mechanical_components
from .scene_frames import (WorkpieceTarget, compose_pose, invert_transform, matrix_pose,
                           pose_matrix, transform_points, transform_poses, validate_pose_xyzw)
from .scene_validation import material_task_metrics, mesh_heightfield_metrics

PROTOCOL_PATH = Path(__file__).resolve().parents[2]/"configs/cutting_tasks_v1.json"
ACTION_MODES = ("cartesian_edge_pose_world", "joint_position_rad")


@dataclass(frozen=True)
class FixtureBox:
    name: str
    center_m: tuple
    half_extents_m: tuple

    def __post_init__(self):
        if (not self.name or np.asarray(self.center_m).shape != (3,)
                or np.asarray(self.half_extents_m).shape != (3,)
                or not np.isfinite([self.center_m, self.half_extents_m]).all()
                or min(self.half_extents_m) <= 0):
            raise ValueError("Fixture boxes require a name and finite local centre/positive half extents")

    def corners_local(self):
        signs = np.stack(np.meshgrid(*[[-1, 1]]*3, indexing="ij"), -1).reshape(-1, 3)
        return np.asarray(self.center_m)+signs*np.asarray(self.half_extents_m)


@dataclass(frozen=True)
class CuttingTaskSpec:
    task_id: str
    seed: int
    variant: str
    geometry: HeightFieldSpec
    particle_spacing_m: float
    grid_dx_m: float
    fixture_boxes_local: tuple
    world_from_workpiece_xyzw: tuple
    table_top_world_z_m: float
    hand_from_edge_xyz_xyzw: tuple
    duration_s: float
    control_dt_s: float
    material: MaterialParameters
    goal_kind: str
    target_origin_local_m: tuple
    target_normal_local: tuple
    edge_clearance_m: float
    blade_length_m: float
    blade_height_m: float
    blade_half_width_m: float
    edge_orientation_local_xyzw: tuple
    edge_position_knots_local_txyz: tuple
    active_cut_interval_s: tuple = (1.6, 5.6)

    def __post_init__(self):
        if self.task_id not in ("CUT-01", "TRIM-01") or self.goal_kind not in ("sever", "flatten"):
            raise ValueError("Unsupported task/goal")
        if self.variant != "fixed":
            raise ValueError("Only the explicit fixed variant is implemented")
        for pose in (self.world_from_workpiece_xyzw, self.hand_from_edge_xyz_xyzw,
                     (0, 0, 0, *self.edge_orientation_local_xyzw)):
            validate_pose_xyzw(pose)
        values = [self.particle_spacing_m, self.grid_dx_m, self.duration_s, self.control_dt_s,
                  self.blade_length_m, self.blade_height_m, self.blade_half_width_m]
        if not np.isfinite(values).all() or min(values) <= 0 or not np.isfinite(self.table_top_world_z_m):
            raise ValueError("Task dimensions/timesteps must be finite and positive")
        if not np.isfinite(self.edge_clearance_m) or self.edge_clearance_m < 0:
            raise ValueError("Edge clearance must be finite and nonnegative")
        if not np.isclose(self.duration_s/self.control_dt_s, round(self.duration_s/self.control_dt_s), atol=1e-9):
            raise ValueError("Episode duration must contain an integral number of control steps")
        normal = np.asarray(self.target_normal_local, float)
        if (normal.shape != (3,) or np.asarray(self.target_origin_local_m).shape != (3,)
                or not np.isfinite([normal, self.target_origin_local_m]).all()
                or not np.isclose(np.linalg.norm(normal), 1., atol=1e-9, rtol=0)):
            raise ValueError("Task target needs a finite local origin and unit normal")
        knots = np.asarray(self.edge_position_knots_local_txyz, float)
        if (knots.ndim != 2 or knots.shape[1] != 4 or not np.isfinite(knots).all()
                or np.any(np.diff(knots[:, 0]) <= 0) or knots[0, 0] != 0 or knots[-1, 0] != self.duration_s):
            raise ValueError("Scripted edge knots must span the episode with increasing finite times")

    @property
    def control_dt(self):
        return self.control_dt_s

    @property
    def T_world_workpiece(self):
        return pose_matrix(self.world_from_workpiece_xyzw)

    def points_to_world(self, local):
        return transform_points(self.T_world_workpiece, local)

    def points_to_local(self, world):
        return transform_points(invert_transform(self.T_world_workpiece), world)

    def target_plane_world(self):
        normal = self.T_world_workpiece[:3, :3]@self.target_normal_local
        origin = self.points_to_world(self.target_origin_local_m)
        return np.r_[normal, -normal@origin]

    def sample_particles(self):
        """Return LOCAL points, immutable reference masses, lattice; never cut labels."""
        points, masses, lattice, _ = sample_specimen(
            BenchmarkSpec(self.task_id, self.seed, self.geometry, self.material), self.particle_spacing_m)
        return points, masses, lattice

    def reference_edge_pose_local(self, time_s):
        """Scripted baseline only; future action callers may explicitly arm cutting."""
        if not np.isfinite(time_s) or not 0 <= time_s <= self.duration_s+1e-9:
            raise ValueError("Reference time must be within the task episode")
        knots = np.asarray(self.edge_position_knots_local_txyz)
        xyz = [np.interp(time_s, knots[:, 0], knots[:, axis]) for axis in (1, 2, 3)]
        return np.r_[xyz, self.edge_orientation_local_xyzw]

    def reference_edge_pose_world(self, time_s):
        return transform_poses(self.T_world_workpiece, self.reference_edge_pose_local(time_s))

    def reference_hand_pose_world(self, time_s):
        inverse_mount = matrix_pose(invert_transform(pose_matrix(self.hand_from_edge_xyz_xyzw)))
        return compose_pose(self.reference_edge_pose_world(time_s), inverse_mount)

    def active_cut_index(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError("Time must be finite")
        start, stop = self.active_cut_interval_s
        return 0 if start-1e-9 <= time_s < stop-1e-9 else -1

    def reference_action(self, time_s):
        return {"mode": "cartesian_edge_pose_world", "target": self.reference_edge_pose_world(time_s),
                "cut_enabled": self.active_cut_index(time_s) >= 0}

    def validate_action(self, action):
        """Validate edge XYZ+XYZW or active-arm7/both-arm2x7 joint positions.

        A 7-vector joint action explicitly leaves the second arm parked; no
        dynamics, IK, observations, forces or hidden cut schedule are added here.
        """
        if not isinstance(action, dict) or set(action)-{"mode", "target", "cut_enabled"}:
            raise ValueError("Action is a mapping with mode, target and optional cut_enabled only")
        mode = action.get("mode")
        target = np.asarray(action.get("target"), float)
        if mode == ACTION_MODES[0]:
            validate_pose_xyzw(target)
            if target.shape != (7,):
                raise ValueError("Cartesian action needs one world knife-edge pose")
        elif mode == ACTION_MODES[1]:
            if target.shape not in ((7,), (2, 7)) or not np.isfinite(target).all():
                raise ValueError("Joint action needs finite active-arm7 or both-arm2x7 positions in radians")
        else:
            raise ValueError("Unsupported action mode")
        enabled = action.get("cut_enabled", True)
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("cut_enabled must be an explicit Boolean arming flag")
        return {"mode": mode, "target": target.copy(), "cut_enabled": bool(enabled)}

    def to_dict(self):
        return {**asdict(self), "protocol_id": _protocol()["protocol_id"],
                "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
                "sample_coordinate_frame": "workpiece_local", "evaluator_coordinate_frame": "world",
                "action_modes": ACTION_MODES, "runtime_validated": False,
                "numerical_overrides": {"particle_spacing_m": {"protocol": _protocol()[self.task_id]["particle_spacing_m"],
                    "actual": self.particle_spacing_m}} if self.particle_spacing_m != _protocol()[self.task_id]["particle_spacing_m"] else {}}

    def evaluate_oracle(self, recording):
        return evaluate_task_oracle(self, recording)


def _protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def make_task(task_id="CUT-01", seed=42, variant="fixed", *, world_from_workpiece_xyzw=(0, 0, 0, 0, 0, 0, 1),
              particle_spacing_m=None):
    protocol = _protocol()
    if task_id not in ("CUT-01", "TRIM-01"):
        raise ValueError("task_id must be CUT-01 or TRIM-01; legacy D0 is unchanged")
    config = protocol[task_id]
    geometry = config["geometry"]
    return CuttingTaskSpec(task_id, int(seed), variant,
        HeightFieldSpec(geometry["length_m"], geometry["width_m"], geometry["base_height_m"],
            tuple(Bump(**row) for row in geometry["bumps"])),
        config["particle_spacing_m"] if particle_spacing_m is None else float(particle_spacing_m), config["grid_dx_m"],
        tuple(FixtureBox(row["name"], tuple(row["center_m"]), tuple(row["half_extents_m"])) for row in config["fixture_boxes_local"]),
        tuple(world_from_workpiece_xyzw), protocol["common"]["table_top_world_z_m"], tuple(config["hand_from_edge_xyz_xyzw"]),
        protocol["episode"]["duration_s"], protocol["episode"]["control_dt_s"], MaterialParameters(**protocol["material"]),
        config["goal_kind"], tuple(config["target_origin_local_m"]), tuple(config["target_normal_local"]),
        config.get("edge_clearance_m", 0.), config["blade_length_m"], config["blade_height_m"], config["blade_half_width_m"],
        tuple(config["edge_orientation_local_xyzw"]), tuple(tuple(row) for row in config["edge_position_knots_local_txyz"]),
        tuple(protocol["episode"]["active_cut_interval_s"]))


def _interface_location(nodes, weights, gradients, labels, reference_local, task):
    """Exact extrema of cross-component particle-pair/node midpoint distances.

    This uses actual MPM support, including zero-weight stress-gradient entries;
    it is a cut-location statistic, never a visual or six-neighbour cut proof.
    """
    support = ((weights > 0) | (np.linalg.norm(gradients, axis=-1) > 0)).ravel()
    particle = np.repeat(np.arange(len(labels)), nodes.shape[1])[support]
    node = nodes.ravel()[support]
    order = np.argsort(node, kind="stable")
    groups = np.split(particle[order], np.flatnonzero(np.diff(node[order]))+1)
    signed = (reference_local-task.target_origin_local_m)@np.asarray(task.target_normal_local)
    maximum, incidences, interfaces = 0., 0, 0
    for group in groups:
        occupied = np.unique(labels[group])
        if len(occupied) < 2:
            continue
        interfaces += 1
        for i, first in enumerate(occupied):
            a = signed[group[labels[group] == first]]
            for second in occupied[i+1:]:
                b = signed[group[labels[group] == second]]
                maximum = max(maximum, abs((a.min()+b.min())/2), abs((a.max()+b.max())/2))
                incidences += len(a)*len(b)
    return {"cross_component_reference_node_count": interfaces,
            "severed_particle_pair_node_incidences": incidences,
            "maximum_reference_midpoint_plane_distance_m": None if not interfaces else float(maximum)}


def evaluate_task_oracle(task, data):
    """Score recorded WORLD state. Missing physical stencils/surface is NOT_RUN.

    Required: time_s(T), positions(T,N,3), reference_positions(N,3), mass_kg(N),
    codes(T,N), reference_stencil_nodes/weights(N,8), gradients(N,8,3).
    Optional TRIM: surface_vertices_world, surface_triangles,
    surface_triangle_component_ids, surface_component_provenance,
    surface_timestamp_s. Component IDs must refer to the recomputed final graph.
    """
    required = ("time_s", "positions", "reference_positions", "mass_kg", "codes",
                "reference_stencil_nodes", "reference_stencil_weights", "reference_stencil_gradients")
    missing = [name for name in required if name not in data]
    base = {"task_id": task.task_id, "scope": "Offline oracle task metrics; no runtime, robot, camera or calibrated-fracture acceptance",
            "protocol_id": _protocol()["protocol_id"], "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            "full_simulation_acceptance": False, "policy_observation": False}
    if missing:
        return {**base, "status": "NOT_RUN", "missing_evidence": missing, "checks": {}}
    time, positions, reference, masses, codes, nodes, weights, gradients = [np.asarray(data[name]) for name in required]
    n = len(reference)
    if (reference.shape != (n, 3) or positions.shape != (len(time), n, 3) or codes.shape != (len(time), n)
            or masses.shape != (n,) or nodes.shape != (n, 8) or weights.shape != nodes.shape or gradients.shape != (n, 8, 3)
            or len(time) < 2 or np.any(np.diff(time) <= 0) or codes.dtype.kind not in "iu" or np.any((codes < 0) | (codes > 2))
            or nodes.dtype.kind not in "iu" or np.any(masses <= 0)
            or not all(np.isfinite(value).all() for value in (time, positions, reference, masses, weights, gradients))):
        raise ValueError("Oracle recording shapes, one-cut history, mass or numeric state are invalid")
    protocol = _protocol()
    common, limits = protocol["common"], protocol[task.task_id]
    initial_count, _ = mechanical_components(nodes, weights, codes[0], 1, gradients=gradients)
    final_count, labels = mechanical_components(nodes, weights, codes[-1], 1, gradients=gradients)
    expected, expected_mass, _ = task.sample_particles()
    actual_reference = task.points_to_local(reference)
    sample_match = expected.shape == actual_reference.shape and np.allclose(expected, actual_reference, atol=1e-7, rtol=0)
    total_mass = float(masses.sum())
    expected_total = float(expected_mass.sum())
    mass_error = abs(total_mass-expected_total)
    rewrites = int(np.count_nonzero((codes[:-1] != 0) & (codes[1:] != codes[:-1])))
    post = np.flatnonzero(time >= task.active_cut_interval_s[1]-1e-9)
    partitions = {codes[-1].tobytes(): final_count}
    post_counts = []
    for index in post:
        key = codes[index].tobytes()
        if key not in partitions:
            partitions[key] = mechanical_components(nodes, weights, codes[index], 1, gradients=gradients)[0]
        post_counts.append(partitions[key])
    persistence_s = 0. if len(post) < 2 else float(time[post[-1]]-time[post[0]])
    checks = {"complete_episode": time[-1]-time[0] >= task.duration_s-1e-9,
              "initial_continuous_graph": initial_count == 1, "initial_codes_zero": not codes[0].any(),
              "reference_matches_single_continuous_task_sampler": bool(sample_match),
              "reference_mass_budget_and_unchanged_count": mass_error <= max(common["mass_absolute_tolerance_kg"], common["mass_relative_tolerance"]*expected_total),
              "persistent_history_no_digit_rewrites": rewrites == 0,
              "postcut_disconnected_topology_persists": bool(post_counts) and min(post_counts) >= 2 and persistence_s >= common["minimum_postcut_topology_persistence_s"]}
    metrics = {"initial_components": initial_count, "final_components": final_count,
               "component_masses_kg": [float(masses[labels == label].sum()) for label in np.unique(labels)],
               "recorded_mass_kg": total_mass, "expected_task_sampler_mass_kg": expected_total,
               "reference_mass_absolute_error_kg": mass_error, "particle_count_all_frames": n,
               "mass_scope": "Recorded immutable reference-mass vector and aligned particle counts, not framewise mass measurements",
               "nonzero_digit_rewrites": rewrites, "postcut_graph_count_minimum": min(post_counts, default=None),
               "postcut_persistence_time_s": persistence_s,
               "component_provenance": "Recomputed actual eight-node compatibility graph including positive weights, stress gradients and code-zero bridges"}
    missing = []
    if task.goal_kind == "sever":
        fractions = np.sort(np.asarray(metrics["component_masses_kg"])/total_mass)[::-1]
        substantial = int(np.count_nonzero(fractions >= limits["minimum_substantial_part_mass_fraction_each"]))
        location = _interface_location(nodes, weights, gradients, labels, actual_reference, task)
        metrics.update(location, substantial_parts=substantial, two_main_parts_mass_fraction=float(fractions[:2].sum()),
                       minor_parts_mass_fraction=float(fractions[2:].sum()))
        checks.update({"two_substantial_physical_parts": substantial == limits["required_substantial_parts"],
            "two_main_parts_retain_mass": fractions[:2].sum() >= limits["minimum_two_main_parts_total_mass_fraction"],
            "cut_at_workpiece_local_target_plane": location["maximum_reference_midpoint_plane_distance_m"] is not None
                and location["maximum_reference_midpoint_plane_distance_m"] <= limits["maximum_severed_reference_stencil_midpoint_plane_distance_m"]})
    else:
        target = WorkpieceTarget(task.world_from_workpiece_xyzw, task.target_origin_local_m[2], task.edge_clearance_m)
        trim = material_task_metrics(reference, positions[-1], masses, labels, metrics["component_provenance"], target, task.geometry)
        metrics["removed_reference_above_plane_mass_fraction"] = trim["detached_initial_above_target_mass_fraction"]
        metrics["reference_above_plane_region"] = "Particle centres initially above target; no fractional-cell or analytic-volume correction"
        checks["reference_above_plane_material_removed"] = (trim["detached_initial_above_target_mass_fraction"] is not None
            and trim["detached_initial_above_target_mass_fraction"] >= limits["minimum_detached_reference_above_plane_mass_fraction"])
        surface_fields = ("surface_vertices_world", "surface_triangles", "surface_triangle_component_ids",
                          "surface_component_provenance", "surface_timestamp_s")
        missing = [name for name in surface_fields if name not in data]
        if not missing:
            triangles = np.asarray(data["surface_triangles"])
            face_component = np.asarray(data["surface_triangle_component_ids"])
            if face_component.shape != (len(triangles),) or face_component.dtype.kind not in "iu":
                raise ValueError("Surface needs one current actual mechanical component ID per triangle")
            selected = np.isin(face_component, trim["reference_substrate_component_ids"])
            checks["surface_has_explicit_mechanical_provenance"] = bool(str(data["surface_component_provenance"]).strip())
            checks["surface_timestamp_matches_final_physics"] = abs(float(data["surface_timestamp_s"])-float(time[-1])) <= 1e-6
            if selected.any():
                surface = mesh_heightfield_metrics(data["surface_vertices_world"], triangles[selected], target, task.geometry,
                    limits["surface_grid_spacing_m"])
                metrics["surface"] = {name: value for name, value in surface.items() if not name.endswith("_debug")}
                names = {"maximum_substrate_surface_residual_height_m": "maximum_substrate_surface_residual_height_m",
                    "surface_p95_absolute_plane_error_m": "surface_p95_absolute_plane_error_m",
                    "surface_rms_plane_error_m": "surface_rms_plane_error_m", "maximum_gouge_depth_m": "maximum_gouge_depth_m",
                    "overcut_volume_fraction_of_initial_base": "maximum_overcut_volume_fraction_of_initial_base",
                    "missing_surface_fraction": "maximum_missing_surface_fraction"}
                checks.update({name: surface[name] is not None and surface[name] <= limits[limit] for name, limit in names.items()})
            else:
                checks["actual_substrate_surface_present"] = False
    checks = {name: bool(value) for name, value in checks.items()}
    if not checks["complete_episode"]:
        missing.append("Complete scripted-duration episode; partial smoke cannot score final task completion")
        status = "NOT_RUN"
    else:
        status = "FAIL_TASK_METRICS" if not all(checks.values()) else "NOT_RUN" if missing else "PASS_TASK_METRICS"
    return {**base, "status": status, "checks": checks, "metrics": metrics, "missing_evidence": missing}
