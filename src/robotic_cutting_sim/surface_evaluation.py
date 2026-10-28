"""Read-only geometric measurements in an explicit, unfitted target frame.

Missing intersections remain NaN in raw arrays. Conditional measured distances,
unknown coverage and conservative missing-column budgets are distinct quantities.
None of these measurements establishes physical material loss or knife gouging.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .scene_frames import pose_matrix, invert_transform, transform_points


@dataclass(frozen=True)
class TargetGeometry:
    kind: str
    source: str
    version: str
    frame_id: str = "workpiece_local"
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("plane", "heightfield", "mesh", "implicit"):
            raise ValueError("Supported targets: plane, heightfield, mesh, implicit")
        if not self.source or not self.version or not self.frame_id:
            raise ValueError("Target source, version and coordinate frame are required")

    def heights(self, xy):
        if self.kind == "plane":
            normal = np.asarray(self.parameters.get("normal", (0., 0., 1.)), float)
            origin = np.asarray(self.parameters["origin_m"], float)
            if (normal.shape != (3,) or origin.shape != (3,) or not np.isfinite([normal, origin]).all()
                    or not np.isclose(np.linalg.norm(normal), 1.) or abs(normal[2]) < 1e-9):
                raise ValueError("Height comparison requires a finite unit plane normal with nonzero Z")
            return origin[2]-((xy-origin[:2])@normal[:2])/normal[2]
        if self.kind == "heightfield":
            x, y, height = [np.asarray(self.parameters[k], float) for k in ("x_m", "y_m", "height_m")]
            if (x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2
                    or height.shape != (len(y), len(x)) or np.any(np.diff(x) <= 0)
                    or np.any(np.diff(y) <= 0) or not all(np.isfinite(a).all() for a in (x, y, height))):
                raise ValueError("Heightfield needs increasing finite axes and a complete Y by X height array")
            return RegularGridInterpolator((y, x), height, bounds_error=False, fill_value=np.nan)(xy[:, ::-1])
        raise ValueError("This target is not a Z heightfield; use 3D surface metrics")


@dataclass(frozen=True)
class RegionSpec:
    name: str
    kind: str = "all"
    parameters: dict = field(default_factory=dict)
    source: str = "predeclared_goal"

    def mask(self, xy):
        if not self.name or not self.source:
            raise ValueError("Every region requires a name and predeclared source")
        if self.kind == "all":
            return np.ones(len(xy), bool)
        if self.kind == "box":
            low, high = np.asarray(self.parameters["bounds_xy_m"], float)
            if low.shape != (2,) or high.shape != (2,) or not np.isfinite([low, high]).all() or np.any(high <= low):
                raise ValueError("Region bounds must be finite and increasing")
            return ((xy >= low) & (xy <= high)).all(1)
        if self.kind == "ellipse":
            center = np.asarray(self.parameters["center_xy_m"], float)
            radii = np.asarray(self.parameters["radii_xy_m"], float)
            if center.shape != (2,) or radii.shape != (2,) or not np.isfinite([center, radii]).all() or np.any(radii <= 0):
                raise ValueError("Ellipse needs a finite centre and positive radii")
            return np.sum(((xy-center)/radii)**2, axis=1) <= 1.
        if self.kind == "capsule":
            length, width = self.parameters["length_m"], self.parameters["width_m"]
            if not np.isfinite([length,width]).all() or width <= 0 or length < width:
                raise ValueError("Capsule region requires length >= positive width")
            distance = np.sqrt(np.maximum(abs(xy[:,0])-(length-width)/2,0)**2+xy[:,1]**2)
            inside = distance <= width/2
            if "boundary_band_m" in self.parameters:
                band = float(self.parameters["boundary_band_m"])
                if not np.isfinite(band) or band <= 0:
                    raise ValueError("Boundary band must be finite and positive")
                inside &= width/2-distance <= band
            return inside
        if self.kind == "boundary":
            low, high = np.asarray(self.parameters["bounds_xy_m"], float)
            width = float(self.parameters["width_m"])
            if low.shape != (2,) or high.shape != (2,) or not np.isfinite([*low, *high, width]).all() or width <= 0 or np.any(high <= low):
                raise ValueError("Boundary needs finite increasing bounds and positive width")
            inside = ((xy >= low) & (xy <= high)).all(1)
            return inside & (np.minimum(xy-low, high-xy).min(1) <= width)
        raise ValueError("Unknown evaluation region")


def validate_mesh(vertices, faces):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))):
        raise ValueError("Need finite vertices and valid integer triangle indices")
    return vertices, faces


def sample_columns(vertices, faces, xy):
    vertices, faces = validate_mesh(vertices, faces)
    height = np.full(len(xy), np.nan)
    for triangle in vertices[faces]:
        lo, hi = triangle[:, :2].min(0), triangle[:, :2].max(0)
        selected = np.flatnonzero(((xy >= lo-1e-12) & (xy <= hi+1e-12)).all(1))
        if not len(selected):
            continue
        u, v = triangle[1]-triangle[0], triangle[2]-triangle[0]
        det = u[0]*v[1]-u[1]*v[0]
        if abs(det) < 1e-16:
            continue
        p = xy[selected]-triangle[0, :2]
        a = (p[:, 0]*v[1]-p[:, 1]*v[0])/det
        b = (u[0]*p[:, 1]-u[1]*p[:, 0])/det
        inside = (a >= -1e-10) & (b >= -1e-10) & (a+b <= 1+1e-10)
        indices = selected[inside]
        height[indices] = np.fmax(height[indices], triangle[0, 2]+a[inside]*u[2]+b[inside]*v[2])
    return height


def _column_statistics(actual, target, selected, cell_area, bottom):
    observed, target_known = np.isfinite(actual), np.isfinite(target)
    paired = selected & observed & target_known
    error = actual[paired]-target[paired]
    count = int(selected.sum())
    missing = selected & ~observed
    unknown_target = selected & ~target_known
    measured_deficit = np.maximum(-error, 0.)
    # A bound for missing rays, not an imputed measured height or actual loss.
    unknown_budget = np.maximum(target[missing & target_known]-bottom, 0.).sum()*cell_area
    return {"status": "NOT_MEASURABLE" if not len(error) else "MEASURED_WITH_GAPS" if missing.any() or unknown_target.any() else "MEASURED",
        "columns": count, "area_m2": count*cell_area, "paired_columns": int(paired.sum()),
        "paired_coverage_fraction": None if not count else float(paired.sum()/count),
        "missing_actual_columns": int(missing.sum()), "unknown_target_columns": int(unknown_target.sum()),
        "missing_actual_fraction": None if not count else float(missing.sum()/count),
        "unknown_target_fraction": None if not count else float(unknown_target.sum()/count),
        "measured_maximum_surface_deficit_m": None if not len(error) else float(measured_deficit.max()),
        "measured_maximum_positive_residual_m": None if not len(error) else float(np.maximum(error, 0.).max()),
        "paired_p95_absolute_error_m": None if not len(error) else float(np.percentile(abs(error), 95)),
        "paired_rms_error_m": None if not len(error) else float(np.sqrt(np.mean(error**2))),
        "measured_deficit_volume_m3": float(measured_deficit.sum()*cell_area) if len(error) else None,
        "unknown_missing_column_volume_budget_m3": float(unknown_budget),
        "whole_region_deficit_volume_m3": None if missing.any() or unknown_target.any() or not count else float(measured_deficit.sum()*cell_area),
        "true_material_gouge": "NOT_MEASURABLE_FROM_RECONSTRUCTED_SURFACE_ALONE",
        "error_scope": "Conditional on paired intersections; no missing ray was filled, trimmed or counted as zero error"}


def evaluate_height_surface(vertices_world, faces, target, *, world_from_target_xyzw,
                            actual_frame_id, bounds_xy_m, spacing_m=.001, regions=(),
                            represented_target=None, bottom_m=0., domain=None):
    """All raw rays, validity, XY and ROI masks are returned for artifact export.

    Represented target must be independently fixed in the SAME target frame.
    It is never aligned to or fitted against the observed result.
    """
    if actual_frame_id != target.frame_id:
        return {"status": "NOT_MEASURABLE", "reason": "Target/actual coordinate frame mismatch"}, {}
    low, high = np.asarray(bounds_xy_m, float)
    if (low.shape != (2,) or high.shape != (2,) or np.any(high <= low)
            or not np.isfinite([*low, *high, spacing_m, bottom_m]).all() or spacing_m <= 0):
        raise ValueError("Finite increasing XY bounds and positive spacing are required")
    shape = np.ceil((high-low)/spacing_m).astype(int)
    step = (high-low)/shape
    x, y = [low[i]+(np.arange(shape[i])+.5)*step[i] for i in (0, 1)]
    xy = np.stack(np.meshgrid(x, y, indexing="xy"), -1).reshape(-1, 2)
    vertices = transform_points(invert_transform(pose_matrix(world_from_target_xyzw)), vertices_world)
    height = sample_columns(vertices, faces, xy)
    target_height = target.heights(xy)
    selected_regions = (RegionSpec("entire_region"), *regions)
    if len({r.name for r in selected_regions}) != len(selected_regions):
        raise ValueError("Evaluation region names must be unique")
    domain_mask = np.ones(len(xy),bool) if domain is None else domain.mask(xy)
    masks = {r.name: r.mask(xy) & domain_mask for r in selected_regions}
    area = float(np.prod(step))
    measured = {name: _column_statistics(height, target_height, mask, area, bottom_m) for name, mask in masks.items()}
    boundary = np.zeros(len(xy), bool)
    for region in regions:
        if region.kind == "boundary":
            boundary |= masks[region.name]
    result = {"status": measured["entire_region"]["status"], "target_kind": target.kind,
        "target_source": target.source, "target_version": target.version, "target_frame_id": target.frame_id,
        "measurement_spacing_xy_m": step.tolist(), "grid_shape_xy": shape.tolist(),
        "analytic_target": measured, "represented_target": {"status": "NOT_RUN_NO_INDEPENDENT_TARGET"},
        "missing_classification": {"boundary_missing_columns": int(np.sum(boundary & ~np.isfinite(height))),
            "interior_unknown_columns": int(np.sum(domain_mask & ~boundary & ~np.isfinite(height))),
            "outside_predeclared_object_domain_columns": int((~domain_mask).sum()),
            "outside_domain_measured_surface_columns": int(np.sum(~domain_mask & np.isfinite(height))),
            "reason": "No vertical mesh intersection. Boundary location is predeclared; cause is not inferred as physical gouging."},
        "unresolved_error_sources": ["particle quadrature", "surface reconstruction", "physical deformation"],
        "no_registration_fitted": True}
    debug = {"xy_local_m": xy, "actual_height_m": height, "analytic_target_height_m": target_height,
        "object_domain_mask": domain_mask,
        "actual_valid": np.isfinite(height), "analytic_target_valid": np.isfinite(target_height),
        **{"roi_"+name: mask for name, mask in masks.items()}}
    if represented_target is not None:
        if (represented_target.kind != "mesh" or represented_target.frame_id != target.frame_id
                or not represented_target.parameters.get("independent_uncut_target", False)):
            raise ValueError("Represented mesh requires independent uncut provenance in the same frame")
        p = represented_target.parameters
        rh = sample_columns(p["vertices_m"], p["triangles"], xy)
        result["represented_target"] = {"source": represented_target.source, "version": represented_target.version,
            "regions": {name: _column_statistics(height, rh, mask, area, bottom_m) for name, mask in masks.items()},
            "reconstruction_floor_vs_analytic": _column_statistics(rh, target_height, masks["entire_region"], area, bottom_m)}
        debug.update(represented_target_height_m=rh, represented_target_valid=np.isfinite(rh))
    return result, debug


def evaluate_3d_surface(vertices, faces, target, *, actual_frame_id):
    """Descriptive area quadrature, in already matched target-local coordinates."""
    if actual_frame_id != target.frame_id:
        return {"status": "NOT_MEASURABLE", "reason": "Target/actual coordinate frame mismatch"}, {}
    vertices, faces = validate_mesh(vertices, faces)
    if not len(faces):
        return {"status": "NOT_MEASURABLE", "reason": "Missing actual surface mesh"}, {}
    if target.kind == "mesh":
        from .surface_distance import describe_meshes
        tv, tf = validate_mesh(target.parameters["vertices_m"], target.parameters["triangles"])
        if not len(tf):
            return {"status": "NOT_MEASURABLE", "reason": "Missing target surface mesh"}, {}
        return describe_meshes(vertices, faces, tv, tf)
    if target.kind == "implicit":
        sdf = target.parameters["signed_distance"]
        triangle = vertices[faces]
        centroids = triangle.mean(1)
        distance = np.asarray(sdf(centroids), float)
        area = np.linalg.norm(np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]), axis=1)/2
        if distance.shape != (len(faces),) or not np.isfinite(distance).all() or area.sum() <= 0:
            raise ValueError("Implicit target must return finite signed Euclidean distance per triangle centroid")
        from .surface_controls import weighted_quantile
        return {"status": "MEASURED_DIRECTED_ONLY", "area_m2": float(area.sum()),
            "area_weighted_p95_distance_m": weighted_quantile(abs(distance), area, .95),
            "area_weighted_rms_distance_m": float(np.sqrt(np.average(distance**2, weights=area))),
            "target_to_actual_coverage": "NOT_MEASURABLE_WITHOUT_TARGET_SURFACE_SAMPLES",
            "target_source": target.source, "target_version": target.version}, {"signed_distance_m": distance, "triangle_area_m2": area}
    raise ValueError("3D comparison expects a mesh or signed-distance target")
