"""Workpiece-local geometry, measured edge tracking and separate task diagnostics.

These routines analyze actual numeric state. A successful offline audit never
establishes Isaac runtime execution or native camera validation.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .benchmark import Bump, HeightFieldSpec
from .scene_frames import WorkpieceTarget, pose_errors, validate_pose_xyzw


def legacy_heightfield(specimen):
    return HeightFieldSpec(specimen.length, specimen.width, specimen.base_height,
        (Bump(specimen.center_x, specimen.center_y, specimen.bump_radius,
              specimen.bump_radius, specimen.bump_height, specimen.family),))


def _points_in_triangle(points, triangle):
    a, b, c = np.asarray(triangle)
    u, v = b-a, c-a
    determinant = u[0]*v[1]-u[1]*v[0]
    if abs(determinant) < 1e-16:
        return np.zeros(len(points), bool)
    delta = points-a
    s = (delta[:, 0]*v[1]-delta[:, 1]*v[0])/determinant
    t = (u[0]*delta[:, 1]-u[1]*delta[:, 0])/determinant
    return (s >= -1e-10) & (t >= -1e-10) & (s+t <= 1+1e-10)


def edge_sweep_metrics(edge_poses_world, target, heightfield, *, edge_half_width_m=.025,
                       sample_spacing_m=.0005, height_tolerance_m=.0005):
    """Projected swept finite edge across the analytic bump cross-section.

    This geometric reach/intersection check cannot prove material contact or cut.
    Only adjacent edges within the fixed local height tolerance cover a section.
    """
    poses = target.poses_to_local(validate_pose_xyzw(edge_poses_world))
    if poses.ndim != 2 or len(poses) < 2 or min(edge_half_width_m, sample_spacing_m) <= 0:
        raise ValueError("Need at least two edge poses and positive geometric resolution")
    nx, ny = np.ceil([heightfield.length_m/sample_spacing_m, heightfield.width_m/sample_spacing_m]).astype(int)
    x = (np.arange(nx)+.5)*heightfield.length_m/nx-heightfield.length_m/2
    y = (np.arange(ny)+.5)*heightfield.width_m/ny-heightfield.width_m/2
    xy = np.stack(np.meshgrid(x, y, indexing="xy"), -1).reshape(-1, 2)
    section = heightfield.height(xy[:, 0], xy[:, 1]) > target.edge_height_local_m+1e-9
    sample = xy[section]
    direction = Rotation.from_quat(poses[:, 3:]).apply(np.tile([0, 1, 0], (len(poses), 1)))
    ends = np.stack([poses[:, :3]-edge_half_width_m*direction,
                     poses[:, :3]+edge_half_width_m*direction], axis=1)
    height_valid = np.max(np.abs(ends[:, :, 2]-target.edge_height_local_m), axis=1) <= height_tolerance_m
    covered = np.zeros(len(sample), bool)
    for i in range(len(poses)-1):
        if not (height_valid[i] and height_valid[i+1]):
            continue
        quad = np.array([ends[i, 0, :2], ends[i, 1, :2], ends[i+1, 1, :2], ends[i+1, 0, :2]])
        covered |= _points_in_triangle(sample, quad[[0, 1, 2]]) | _points_in_triangle(sample, quad[[0, 2, 3]])
    along = np.linspace(-edge_half_width_m, edge_half_width_m, 41)
    edge_samples = poses[:, None, :3]+direction[:, None, :]*along[None, :, None]
    point_height = heightfield.height(edge_samples[:, :, 0], edge_samples[:, :, 1])
    in_footprint = ((np.abs(edge_samples[:, :, 0]) <= heightfield.length_m/2)
                    & (np.abs(edge_samples[:, :, 1]) <= heightfield.width_m/2))
    actual_intersection = bool(np.any(in_footprint & (point_height > edge_samples[:, :, 2])))
    return {"scope": "analytic workpiece-local finite-edge geometry only",
            "cross_section_samples": len(sample), "cross_section_area_m2": float(section.sum()*heightfield.length_m*heightfield.width_m/(nx*ny)),
            "cross_section_sweep_coverage": None if not len(sample) else float(covered.mean()),
            "actual_edge_intersects_original_material": actual_intersection,
            "edge_endpoint_height_valid_fraction": float(height_valid.mean()),
            "local_edge_center_bounds_m": [poses[:, :3].min(0).tolist(), poses[:, :3].max(0).tolist()],
            "cut_plane_local_z_m": target.edge_height_local_m,
            "target_plane_local_z_m": target.height_m,
            "sampling_spacing_m": sample_spacing_m,
            "height_tolerance_m": height_tolerance_m}


def trajectory_metrics(times_s, commanded_edge_world, measured_edge_world, target,
                       heightfield, protocol, edge_half_width_m=.025):
    time = np.asarray(times_s, dtype=float)
    command, actual = validate_pose_xyzw(commanded_edge_world), validate_pose_xyzw(measured_edge_world)
    if (command.shape != actual.shape or command.shape != (len(time), 7)
            or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)):
        raise ValueError("Commanded/measured poses and increasing timestamps must align")
    start, stop = protocol["episode"]["active_cut_interval_s"]
    active = (time >= start-1e-9) & (time < stop-1e-9)
    if active.sum() < 2:
        return {"status": "NOT_RUN", "reason": "No complete active cutting interval in the recording"}
    position_error, rotation_error = pose_errors(command, actual)
    local_command, local_actual = target.poses_to_local(command), target.poses_to_local(actual)
    height_error = np.abs(local_actual[:, 2]-local_command[:, 2])
    normal = Rotation.from_quat(local_actual[:, 3:]).apply(np.tile([0., 0., 1.], (len(actual), 1)))
    normal_tilt = np.rad2deg(np.arccos(np.clip(normal[:, 2], -1., 1.)))
    sweep = edge_sweep_metrics(actual[active], target, heightfield, edge_half_width_m=edge_half_width_m,
                              height_tolerance_m=protocol["tests"]["ROB-02"]["active_max_edge_height_error_from_command_m"])
    gates = protocol["tests"]["ROB-02"]
    metrics = {"active_max_edge_position_error_m": float(position_error[active].max()),
               "active_max_edge_height_error_from_command_m": float(height_error[active].max()),
               "active_max_edge_rotation_error_deg": float(np.rad2deg(rotation_error[active]).max()),
               "all_phase_max_edge_position_error_m": float(position_error.max()),
               "max_measured_edge_translation_per_control_step_m": float(np.linalg.norm(np.diff(actual[:, :3], axis=0), axis=1).max())}
    passed = {name: value <= gates[name] for name, value in metrics.items()}
    passed["minimum_analytic_bump_cross_section_sweep_coverage"] = (sweep["cross_section_sweep_coverage"] is not None
        and sweep["cross_section_sweep_coverage"] >= gates["minimum_analytic_bump_cross_section_sweep_coverage"])
    passed["actual_edge_intersects_original_material"] = sweep["actual_edge_intersects_original_material"]
    return {"status": "PASS" if all(passed.values()) else "FAIL", "scope": "numeric edge tracking and analytic reach; runtime provenance is separate",
            "metrics": metrics, "checks": passed, "active_interval_s": [start, stop],
            "active_sample_count": int(active.sum()), "sweep": sweep,
            "active_local_edge_height_range_m": [float(local_actual[active, 2].min()), float(local_actual[active, 2].max())],
            "active_plane_normal_tilt_max_deg": float(normal_tilt[active].max()),
            "local_command_poses_xyzw": local_command, "local_actual_poses_xyzw": local_actual,
            "position_error_m": position_error, "rotation_error_rad": rotation_error,
            "active_mask": active}


def material_task_metrics(initial_points_world, final_points_world, masses, component_ids,
                          component_provenance, target, heightfield, *, reference_plane_tolerance_m=1e-7):
    initial, final = target.points_to_local(initial_points_world), target.points_to_local(final_points_world)
    mass, labels = np.asarray(masses, float), np.asarray(component_ids)
    if (initial.shape != final.shape or initial.ndim != 2 or initial.shape[1] != 3
            or mass.shape != (len(initial),) or labels.shape != mass.shape
            or not np.isfinite(mass).all() or np.any(mass <= 0) or labels.dtype.kind not in "iu"):
        raise ValueError("Need aligned finite initial/final particles, positive masses and integer component IDs")
    if not component_provenance or np.any(labels < 0):
        return {"status": "NOT_RUN", "reason": "Complete mechanical component IDs and explicit provenance are required"}
    anchors = initial[:, 2] <= initial[:, 2].min()+reference_plane_tolerance_m
    substrate_ids = np.unique(labels[anchors])
    substrate = np.isin(labels, substrate_ids)
    above = initial[:, 2] > target.height_m+reference_plane_tolerance_m
    above_mass, detached_above = mass[above].sum(), mass[above & ~substrate].sum()
    return {"status": "MEASURED", "scope": "mechanical component mass plus workpiece-local reference region; not an observation",
            "component_provenance": component_provenance, "component_count": int(len(np.unique(labels))),
            "reference_substrate_component_ids": substrate_ids.tolist(), "detached_mass_kg": float(mass[~substrate].sum()),
            "reference_above_target_mass_kg": float(above_mass), "detached_initial_above_target_mass_kg": float(detached_above),
            "detached_initial_above_target_mass_fraction": None if above_mass <= 0 else float(detached_above/above_mass),
            "remaining_initial_above_target_mass_kg": float(mass[above & substrate].sum()),
            "reference_plane_tolerance_m": reference_plane_tolerance_m,
            "current_substrate_particle_center_max_residual_m": float(np.maximum(final[substrate, 2]-target.height_m, 0).max()),
            "surface_flattening_status": "NOT_RUN; needs measured substrate surface/depth, not centers or a particle radius proxy",
            "substrate_mask_debug": substrate}


def mesh_heightfield_metrics(vertices_world, triangles, target, heightfield, spacing_m=.001):
    """Actual substrate triangle vertical intersections in the workpiece frame.

    Caller must select substrate triangles by mechanical-component provenance,
    not by screen masks or visual appearance. Missing rays remain missing and
    contribute the full target height to the conservative overcut integral.
    """
    vertices = target.points_to_local(vertices_world)
    faces = np.asarray(triangles)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in "iu" or not faces.size or faces.min() < 0 or faces.max() >= len(vertices)
            or not np.isfinite(spacing_m) or spacing_m <= 0):
        raise ValueError("Need a finite substrate triangle mesh and positive measurement spacing")
    nx, ny = np.ceil([heightfield.length_m/spacing_m, heightfield.width_m/spacing_m]).astype(int)
    sx, sy = heightfield.length_m/nx, heightfield.width_m/ny
    x = (np.arange(nx)+.5)*sx-heightfield.length_m/2
    y = (np.arange(ny)+.5)*sy-heightfield.width_m/2
    height = np.full((ny, nx), -np.inf)
    for face in faces:
        triangle = vertices[face]
        lo, hi = triangle[:, :2].min(0), triangle[:, :2].max(0)
        ix = np.flatnonzero((x >= lo[0]-1e-12) & (x <= hi[0]+1e-12))
        iy = np.flatnonzero((y >= lo[1]-1e-12) & (y <= hi[1]+1e-12))
        if not len(ix) or not len(iy):
            continue
        u, v = triangle[1]-triangle[0], triangle[2]-triangle[0]
        determinant = u[0]*v[1]-u[1]*v[0]
        if abs(determinant) < 1e-16:
            continue
        gx, gy = np.meshgrid(x[ix], y[iy])
        a = ((gx-triangle[0, 0])*v[1]-(gy-triangle[0, 1])*v[0])/determinant
        b = (u[0]*(gy-triangle[0, 1])-u[1]*(gx-triangle[0, 0]))/determinant
        inside = (a >= -1e-10) & (b >= -1e-10) & (a+b <= 1+1e-10)
        z = triangle[0, 2]+a*u[2]+b*v[2]
        sub = height[np.ix_(iy, ix)]
        sub[inside] = np.maximum(sub[inside], z[inside])
        height[np.ix_(iy, ix)] = sub
    valid = np.isfinite(height)
    error = height[valid]-target.height_m
    conservative = np.where(valid, height, 0.)
    deficit = np.maximum(target.height_m-conservative, 0.)
    return {"scope": "vertical intersections of provided mechanical-substrate mesh in workpiece coordinates",
            "measurement_spacing_xy_m": [sx, sy], "grid_shape_xy": [int(nx), int(ny)],
            "valid_surface_fraction": float(valid.mean()), "missing_surface_fraction": float(1-valid.mean()),
            "maximum_substrate_surface_residual_height_m": None if not valid.any() else float(np.maximum(error, 0).max()),
            "surface_p95_absolute_plane_error_m": None if not valid.any() else float(np.percentile(np.abs(error), 95)),
            "surface_rms_plane_error_m": None if not valid.any() else float(np.sqrt(np.mean(error*error))),
            "maximum_gouge_depth_m": float(deficit.max()),
            "overcut_volume_m3": float(deficit.sum()*sx*sy),
            "overcut_volume_fraction_of_initial_base": float(deficit.sum()*sx*sy/(heightfield.length_m*heightfield.width_m*heightfield.base_height_m)),
            "measured_heightfield_local_m_debug": np.where(valid, height, np.nan),
            "valid_heightfield_debug": valid}


def flattening_checks(surface_metrics, protocol):
    gates = protocol["tests"]["TASK-02"]
    limits = {"maximum_substrate_surface_residual_height_m": gates["maximum_substrate_surface_residual_height_m"],
              "surface_p95_absolute_plane_error_m": gates["surface_p95_absolute_plane_error_m"],
              "surface_rms_plane_error_m": gates["surface_rms_plane_error_m"],
              "maximum_gouge_depth_m": gates["maximum_gouge_depth_m"],
              "overcut_volume_fraction_of_initial_base": gates["maximum_overcut_volume_fraction_of_initial_base"],
              "missing_surface_fraction": gates["maximum_unobserved_or_empty_surface_bin_fraction"]}
    checks = {name: surface_metrics.get(name) is not None and surface_metrics[name] <= limit for name, limit in limits.items()}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "scope": "surface metric threshold assessment; runtime provenance and mesh/physics consistency assessed separately"}
