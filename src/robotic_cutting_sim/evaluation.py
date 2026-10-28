"""Evaluation v2: recorded physics, operation outcome and geometric quality.

Consumes recorded arrays only. Never controls physics, stops an episode, changes
material or supplies privileged metrics to policy observations. The original
task evaluator remains the exact, separately named precision benchmark.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .cut_topology import mechanical_components
from .scene_frames import pose_matrix, invert_transform, transform_points
from .surface_evaluation import TargetGeometry, RegionSpec, evaluate_height_surface, evaluate_3d_surface

PROTOCOL_PATH = Path(__file__).resolve().parents[2]/"configs/evaluation_v2.json"


@dataclass(frozen=True)
class GoalSpec:
    goal_type: str
    target: TargetGeometry | None = None
    bounds_xy_m: tuple | None = None
    regions: tuple = ()
    thresholds: dict = field(default_factory=dict)
    applicable_metrics: tuple = ()
    frame_id: str = "workpiece_local"
    world_from_goal_xyzw: tuple = (0., 0., 0., 0., 0., 0., 1.)
    version: str = "goal_v2"
    source: str = "explicit_caller_goal"
    spacing_m: float = .001
    duration_s: float | None = None
    domain: RegionSpec | None = None

    def __post_init__(self):
        if self.goal_type not in ("sever", "trim_surface", "remove_region", "shape_to_target", "diagnostic_only"):
            raise ValueError("Unsupported goal type")
        pose_matrix(self.world_from_goal_xyzw)
        if not self.version or not self.source or not self.frame_id or not np.isfinite(self.spacing_m) or self.spacing_m <= 0:
            raise ValueError("Goal requires version, source, frame and positive spacing")
        if self.target is not None and self.target.frame_id != self.frame_id:
            raise ValueError("Goal and target must explicitly share one coordinate frame")
        if any(not isinstance(r, RegionSpec) for r in self.regions):
            raise ValueError("Goal regions must be RegionSpec instances")
        if self.duration_s is not None and (not np.isfinite(self.duration_s) or self.duration_s <= 0):
            raise ValueError("Expected episode duration must be positive")
        for name, rule in self.thresholds.items():
            if not name or not isinstance(rule, dict) or not rule or set(rule)-{"min", "max"} or not np.isfinite(list(rule.values())).all():
                raise ValueError("Thresholds are named metric paths with finite min/max limits")


def goal_for_task(task):
    """Task-derived ROIs/targets are fixed before looking at final outcomes."""
    if task.goal_kind == "sever":
        return GoalSpec("sever", world_from_goal_xyzw=task.world_from_workpiece_xyzw,
            applicable_metrics=("components", "mass", "cut_location", "persistence"),
            source="cutting_tasks_v1.json/"+task.task_id)
    length, width = task.geometry.length_m, task.geometry.width_m
    bounds = ((-length/2, -width/2), (length/2, width/2))
    band = json.loads(PROTOCOL_PATH.read_text())["roi"]["boundary_band_m"]
    regions = [RegionSpec("boundary", "boundary", {"bounds_xy_m": bounds, "width_m": band}, "task_geometry_before_evaluation"),
        RegionSpec("interior", "box", {"bounds_xy_m": [[-length/2+band, -width/2+band], [length/2-band, width/2-band]]}, "task_geometry_before_evaluation")]
    for index, bump in enumerate(task.geometry.bumps):
        regions.append(RegionSpec(f"bump_{index}", "ellipse", {"center_xy_m": [bump.x_m, bump.y_m],
            "radii_xy_m": [bump.radius_x_m, bump.radius_y_m]}, "task_initial_bump_"+str(index)))
    domain = RegionSpec("object_footprint", "box", {"bounds_xy_m":bounds}, "initial_task_footprint")
    if task.workpiece_kind == "capsule":
        parameters={"length_m":length,"width_m":width}
        domain=RegionSpec("object_footprint","capsule",parameters,"initial_task_capsule")
        regions[0]=RegionSpec("boundary","capsule",{**parameters,"boundary_band_m":band},"initial_task_capsule_boundary")
    return GoalSpec("trim_surface", TargetGeometry("plane", "cutting_tasks_v1.json/"+task.task_id,
        "analytic_task_v1", parameters={"origin_m": task.target_origin_local_m, "normal": task.target_normal_local}),
        bounds, tuple(regions), world_from_goal_xyzw=task.world_from_workpiece_xyzw,
        applicable_metrics=("detached_reference_mass", "per_bump_removal", "surface", "protected_material_deficit"),
        source="cutting_tasks_v1.json/"+task.task_id,domain=domain)


def _integrity(data, duration_s):
    required = ("time_s", "positions", "reference_positions", "mass_kg", "codes",
        "reference_stencil_nodes", "reference_stencil_weights", "reference_stencil_gradients")
    missing = [k for k in required if k not in data]
    if missing:
        return {"status": "NOT_MEASURABLE", "missing_evidence": missing, "checks": {},
            "full_physics_acceptance": False}, None, None
    t, q, ref, m, codes, nodes, weights, gradients = [np.asarray(data[k]) for k in required]
    n = len(ref)
    shapes = (ref.shape == (n, 3) and q.shape == (len(t), n, 3) and m.shape == (n,)
        and codes.shape == (len(t), n) and nodes.shape == (n, 8) and weights.shape == nodes.shape
        and gradients.shape == (n, 8, 3) and len(t) >= 2 and n > 0)
    if not shapes:
        return {"status": "FAIL", "reason": "Invalid aligned recorded-state shapes", "full_physics_acceptance": False}, None, None
    checks = {"finite_particle_state": all(np.isfinite(a).all() for a in (t, q, ref, m, weights, gradients)),
        "positive_reference_masses": bool(np.all(m > 0)), "increasing_time": bool(np.all(np.diff(t) > 0)),
        "valid_stencil_node_indices": bool(nodes.dtype.kind in "iu" and np.all(nodes >= 0)),
        "valid_cut_history": bool(codes.dtype.kind in "iu" and np.all((codes >= 0) & (codes <= 2)))}
    if "velocities" in data:
        checks["finite_velocities"] = np.asarray(data["velocities"]).shape == q.shape and bool(np.isfinite(data["velocities"]).all())
    if not all(checks.values()):
        return {"status": "FAIL", "checks": checks, "full_physics_acceptance": False}, None, None
    rewrites = int(np.count_nonzero((codes[:-1] != 0) & (codes[1:] != codes[:-1])))
    checks["persistent_history_no_rewrites"] = rewrites == 0
    source = data.get("source_evidence", {})
    checks["recorded_source_bytes_verified"] = source.get("implementation_verified") is True
    checks["physics_recording_provenance"] = source.get("physics_state_from_recording") is True
    checks["contact_deformation_history_traceable"] = source.get("contact_deformation_traceable") is True
    reference_mass = float(m.sum())
    mass_status = "REFERENCE_MASS_AND_COUNT_ONLY"
    mass_error = None
    if "frame_mass_kg" in data:
        fm = np.asarray(data["frame_mass_kg"], float)
        checks["finite_aligned_frame_mass"] = fm.shape == (len(t), n) and bool(np.isfinite(fm).all() and np.all(fm > 0))
        if checks["finite_aligned_frame_mass"]:
            mass_error = float(np.max(abs(fm.sum(1)-reference_mass)))
            checks["mass_conserved"] = mass_error <= max(1e-9, reference_mass*1e-6)
            mass_status = "MEASURED_PER_FRAME"
    elif "expected_reference_mass_kg" in data:
        mass_error = abs(reference_mass-float(data["expected_reference_mass_kg"]))
        checks["reference_mass_matches_initial_recipe"] = mass_error <= max(1e-9, reference_mass*1e-6)
    if "deformation_jacobian" in data:
        j = np.asarray(data["deformation_jacobian"], float)
        checks["positive_finite_deformation_jacobian"] = bool(j.size and np.isfinite(j).all() and np.all(j > 0))
    topology_cache = {}
    for code in codes:
        key = code.tobytes()
        if key not in topology_cache:
            topology_cache[key] = mechanical_components(nodes, weights, code, 1, gradients=gradients)
    counts = [int(topology_cache[c.tobytes()][0]) for c in codes]
    labels = topology_cache[codes[-1].tobytes()][1]
    tail = len(t)-1
    while tail > 0 and counts[tail-1] >= 2:
        tail -= 1
    persistent = 0. if counts[-1] < 2 else float(t[-1]-t[tail])
    completed = None if duration_s is None else bool(t[-1]-t[0] >= duration_s-1e-9)
    numeric_failure = any(not value for key, value in checks.items() if key not in (
        "recorded_source_bytes_verified", "physics_recording_provenance", "contact_deformation_history_traceable"))
    status = "FAIL" if numeric_failure else "PASS_RECORDED_STATE_INTEGRITY" if all(checks.values()) else "NOT_MEASURABLE_INCOMPLETE_PROVENANCE"
    metrics = {"initial_components": counts[0], "final_components": counts[-1],
        "component_masses_kg": [float(m[labels == i].sum()) for i in np.unique(labels)],
        "reference_mass_kg": reference_mass, "mass_error_kg": mass_error, "mass_measurement_scope": mass_status,
        "particle_count_all_frames": n, "nonzero_history_rewrites": rewrites,
        "sampled_separation_persistence_s": persistent, "component_counts_by_sample": counts,
        "separation_scope": "Current actual MPM support graph independently recomputed at each saved code state, including stress gradients; no future component labels"}
    return {"status": status, "checks": checks, "metrics": metrics, "episode_completed": completed,
        "full_physics_acceptance": False, "limitations": ["No calibrated material or converged coupling-energy claim",
            "Constant reference masses and particle counts are not independently sampled dynamic masses"]}, labels, np.asarray(counts)


def _removal(data, labels, goal, task):
    m = np.asarray(data["mass_kg"], float)
    ref = transform_points(invert_transform(pose_matrix(goal.world_from_goal_xyzw)), data["reference_positions"])
    final = transform_points(invert_transform(pose_matrix(goal.world_from_goal_xyzw)), np.asarray(data["positions"])[-1])
    anchors = ref[:, 2] <= ref[:, 2].min()+1e-7
    substrate = np.isin(labels, np.unique(labels[anchors]))
    if goal.target is None or goal.target.kind not in ("plane", "heightfield"):
        return {"status": "NOT_MEASURABLE", "reason": "Removal requires an explicit initial target region or target height geometry"}, substrate
    target_height = goal.target.heights(ref[:, :2])
    known = np.isfinite(target_height)
    target_region = np.ones(len(ref),bool) if goal.domain is None else goal.domain.mask(ref[:,:2])
    above = known & target_region & (ref[:, 2] > target_height+1e-7)
    below = known & ~above
    detached = ~substrate
    above_mass = float(m[above].sum())
    removed = float(m[above & detached].sum())
    per_bump = []
    if task is not None:
        # Overlaps have a fixed initial-height argmax owner, never counted twice.
        heights = np.asarray([b.height(ref[:, 0], ref[:, 1]) for b in task.geometry.bumps])
        if len(heights):
            owner = np.argmax(heights, axis=0)
            for i in range(len(heights)):
                region = above & (owner == i) & (heights[i] > 0)
                mass = float(m[region].sum())
                per_bump.append({"bump_index": i, "initial_target_mass_kg": mass,
                    "detached_target_mass_kg": float(m[region & detached].sum()),
                    "detached_target_mass_fraction": None if mass == 0 else float(m[region & detached].sum()/mass),
                    "ownership": "Initial analytic height argmax, ties by initial bump index"})
    density = None if task is None else task.material.density_kg_m3
    return {"status": "MEASURED" if known[target_region].all() and above_mass else "NOT_MEASURABLE" if not above_mass else "MEASURED_WITH_UNKNOWN_TARGET_REGION",
        "initial_target_mass_kg": above_mass, "detached_target_mass_kg": removed,
        "removed_reference_above_target_mass_fraction": None if not above_mass else removed/above_mass,
        "remaining_target_mass_kg": above_mass-removed,
        "protected_reference_material_detached_mass_kg": float(m[below & detached].sum()),
        "protected_reference_material_detached_volume_m3": None if density is None else float(m[below & detached].sum()/density),
        "protected_deficit_scope": "Actual detached reference mass below target, distinct from mesh deficit and plastic deformation; volume uses provisional reference density",
        "remaining_particle_center_positive_residual_m": None if not substrate.any() else float(np.maximum(final[substrate & known, 2]-target_height[substrate & known], 0).max()) if (substrate & known).any() else None,
        "per_bump": per_bump, "unknown_target_particles": int((target_region & ~known).sum()),
        "target_region_particles": int(target_region.sum()),
        "reference_region_scope": "Initial particle centres, no fractional-cell correction or renderer inference"}, substrate


def _assess_thresholds(result, thresholds):
    checks = {}
    for path, limits in thresholds.items():
        value = result
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            checks[path] = "NOT_MEASURABLE"
        else:
            checks[path] = "PASS" if all(value >= limit if op == "min" else value <= limit for op, limit in limits.items()) else "FAIL"
    status = ("NOT_CONFIGURED" if not checks else "FAIL" if "FAIL" in checks.values()
        else "NOT_MEASURABLE" if "NOT_MEASURABLE" in checks.values() else "PASS")
    return {"status": status, "checks": checks, "thresholds": thresholds}


def _inherited_outcome_checks(task, goal, strict):
    if (strict is None or task is None or goal.goal_type != ("sever" if task.goal_kind == "sever" else "trim_surface")
            or goal.source != "cutting_tasks_v1.json/"+task.task_id):
        return None
    expected = goal_for_task(task)
    if (goal.frame_id != expected.frame_id or (goal.target is None) != (expected.target is None)
            or (goal.target is not None and goal.target.kind != expected.target.kind)):
        return None
    def canonical(value):
        return json.dumps(value,sort_keys=True,default=lambda x:np.asarray(x).tolist())
    if (not np.allclose(pose_matrix(goal.world_from_goal_xyzw),pose_matrix(expected.world_from_goal_xyzw),atol=1e-12,rtol=0)
            or canonical(None if goal.target is None else asdict(goal.target)) != canonical(None if expected.target is None else asdict(expected.target))
            or canonical(None if goal.domain is None else asdict(goal.domain)) != canonical(None if expected.domain is None else asdict(expected.domain))):
        return None
    names = ("complete_episode", "initial_continuous_graph", "initial_codes_zero", "reference_matches_single_continuous_task_sampler",
        "reference_mass_budget_and_unchanged_count", "persistent_history_no_digit_rewrites", "postcut_disconnected_topology_persists")
    names += (("two_substantial_physical_parts", "two_main_parts_retain_mass", "cut_at_workpiece_local_target_plane")
        if goal.goal_type == "sever" else ("reference_above_plane_material_removed",))
    return {name: strict.get("checks",{}).get(name) for name in names}


def _check_status(checks):
    return "PASS" if checks and all(v is True for v in checks.values()) else "FAIL" if checks and any(v is False for v in checks.values()) else "NOT_MEASURABLE"


def evaluate_recording(data, *, task=None, goal=None, profile="diagnostic", represented_target=None):
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if profile not in protocol["profiles"]:
        raise ValueError("Choose diagnostic, research or precision")
    if goal is None:
        if task is None:
            raise ValueError("Provide a task or explicit GoalSpec")
        goal = goal_for_task(task)
    integrity, labels, counts = _integrity(data, goal.duration_s if task is None else task.duration_s)
    frame_mismatch = data.get("geometry_frame_id",goal.frame_id) != goal.frame_id
    outcome = {"status": "NOT_MEASURABLE", "goal_type": goal.goal_type}
    surface = {"status": "NOT_APPLICABLE" if goal.target is None else "NOT_MEASURABLE"}
    strict = None
    raw = {}
    if task is not None:
        try:
            strict = task.evaluate_oracle(data)
        except (ValueError, KeyError, IndexError) as error:
            strict = {"status": "NOT_MEASURABLE", "reason": str(error)}
    substrate = None
    if labels is not None:
        outcome.update(status="MEASURED", metrics=integrity["metrics"])
        if goal.goal_type in ("trim_surface", "remove_region", "shape_to_target") and not frame_mismatch:
            removal, substrate = _removal(data, labels, goal, task)
            outcome["removal"] = removal
        if strict is not None and goal.goal_type == "sever":
            outcome["severing"] = strict.get("metrics", {})
    mesh_keys = ("surface_vertices_world", "surface_triangles")
    if goal.target is not None and all(k in data for k in mesh_keys):
        vertices, faces = [np.asarray(data[k]) for k in mesh_keys]
        surface_provenance_valid = bool(data.get("surface_component_provenance")) and bool(
            data.get("surface_graph_verified", False)) and abs(float(data.get("surface_timestamp_s", -1))-float(np.asarray(data.get("time_s", [-2]))[-1])) <= 1e-6
        if substrate is not None:
            face_labels = np.asarray(data.get("surface_triangle_component_ids", []))
            if face_labels.shape != (len(faces),):
                surface_provenance_valid = False
            else:
                faces = faces[np.isin(face_labels, np.unique(labels[substrate]))]
        if not surface_provenance_valid:
            surface = {"status": "NOT_MEASURABLE", "reason": "Current mesh/physics graph provenance or timestamp is missing/mismatched"}
        elif goal.target.kind in ("plane", "heightfield") and goal.bounds_xy_m is not None:
            surface, raw = evaluate_height_surface(vertices, faces, goal.target,
                world_from_target_xyzw=goal.world_from_goal_xyzw, actual_frame_id=data.get("geometry_frame_id", ""),
                bounds_xy_m=goal.bounds_xy_m, spacing_m=goal.spacing_m, regions=goal.regions,
                represented_target=represented_target,domain=goal.domain)
        elif goal.target.kind in ("mesh", "implicit"):
            local = transform_points(invert_transform(pose_matrix(goal.world_from_goal_xyzw)), vertices)
            surface, raw = evaluate_3d_surface(local, faces, goal.target, actual_frame_id=data.get("geometry_frame_id", ""))
    result = {"protocol_id": protocol["protocol_id"], "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "profile": profile, "goal": asdict(goal), "units": protocol["units"],
        "episode_completed": integrity.get("episode_completed"), "physical_integrity": integrity,
        "cutting_outcome": outcome, "surface_quality": surface,
        "legacy_precision_benchmark": strict, "evaluation_never_controls_execution": True,
        "full_physics_acceptance": False, "success": None}
    # Callable SDFs are runtime extension points; save their explicit provenance,
    # never pretend that a Python function is a serializable target artifact.
    if goal.target is not None and goal.target.kind == "implicit":
        result["goal"]["target"]["parameters"] = {k:v for k,v in result["goal"]["target"]["parameters"].items() if k != "signed_distance"}
    thresholds = _assess_thresholds(result, goal.thresholds)
    inherited = _inherited_outcome_checks(task, goal, strict)
    if profile == "diagnostic":
        assessment = {"goal_status": "DIAGNOSTIC_ONLY", "surface_quality_status": "DIAGNOSTIC_ONLY", "thresholds": thresholds}
    elif profile == "precision" and strict is not None:
        surface_checks = {name:strict.get("checks",{}).get(name) for name in (
            "maximum_substrate_surface_residual_height_m", "surface_p95_absolute_plane_error_m", "surface_rms_plane_error_m",
            "maximum_gouge_depth_m", "overcut_volume_fraction_of_initial_base", "missing_surface_fraction")}
        assessment = {"goal_status":_check_status(inherited),
            "inherited_outcome_checks":inherited,
            "surface_quality_status":_check_status(surface_checks) if goal.goal_type == "trim_surface" else "NOT_APPLICABLE",
            "surface_quality_checks":surface_checks if goal.goal_type == "trim_surface" else {},
            "combined_legacy_status":strict["status"],"thresholds":thresholds,"original_benchmark_unchanged":True}
        result["success"] = (strict.get("status") == "PASS_TASK_METRICS"
            and inherited is not None and integrity["status"] == "PASS_RECORDED_STATE_INTEGRITY")
    else:
        goal_status = ("PASS" if inherited and all(v is True for v in inherited.values()) else "FAIL" if inherited and any(v is False for v in inherited.values())
            else thresholds["status"] if goal.thresholds else "NOT_MEASURABLE_NO_GOAL_THRESHOLDS")
        if goal.thresholds and thresholds["status"] != "PASS":
            goal_status = thresholds["status"]
        if any(path.startswith("surface_quality.") for path in goal.thresholds) and surface["status"] in (
                "NOT_MEASURABLE", "MEASURED_WITH_GAPS", "MEASURED_DIRECTED_ONLY"):
            goal_status = "NOT_MEASURABLE_INCOMPLETE_GEOMETRY"
        if goal.goal_type == "diagnostic_only":
            goal_status = "DIAGNOSTIC_ONLY"
        assessment = {"goal_status": goal_status, "inherited_outcome_checks": inherited,
            "surface_quality_status": "MEASURED_SEPARATELY_NO_UNIVERSAL_PRECISION_GATE", "thresholds": thresholds}
        result["success"] = (goal_status == "PASS" and integrity["status"] == "PASS_RECORDED_STATE_INTEGRITY"
            and integrity.get("episode_completed") is True)
    result["profile_assessment"] = assessment
    if goal.thresholds and profile != "diagnostic":
        if thresholds["status"] != "PASS":
            result["success"] = False
        if any(path.startswith("surface_quality.") for path in goal.thresholds) and surface["status"] in (
                "NOT_MEASURABLE", "MEASURED_WITH_GAPS", "MEASURED_DIRECTED_ONLY"):
            assessment["surface_quality_status"] = "NOT_MEASURABLE_INCOMPLETE_GEOMETRY"
            result["success"] = False
    result["coordinate_registration"] = {"status":"NOT_MEASURABLE_FRAME_MISMATCH" if frame_mismatch else "DECLARED_FRAME_MATCH" if "geometry_frame_id" in data else "TASK_OR_CALLER_FRAME_CONTRACT",
        "actual_frame_id":data.get("geometry_frame_id"),"goal_frame_id":goal.frame_id,"fitted_registration":False}
    if frame_mismatch:
        assessment["goal_status"] = "NOT_MEASURABLE_FRAME_MISMATCH"
        result["success"] = None if profile == "diagnostic" else False
    result["status"] = "EVALUATED" if labels is not None else integrity["status"]
    return result, raw
