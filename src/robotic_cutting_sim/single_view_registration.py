"""Register native NVIDIA SingleView depth to its original virtual RGB camera.

The point-cloud AOV supplies directions in the left camera's USD axes. The
distance AOV supplies measured optical-Z samples on the stereo-reprojected grid.
Projection associates each sample with the original RGB pixel; its measured
distance is retained verbatim. Point-cloud Z is NEVER substituted for measured
depth, and disagreement with point-cloud Z is NEVER a rejection criterion.

This does not calibrate a physical D455 or align its separate color imager.
"""
import numpy as np


def register_native_depth(pointcloud_usd, native_distance_m, camera, *, min_depth_m=.52, max_depth_m=6.,
                          pointcloud_pixel_offset=(0., 0.)):
    """Return optical-Z depth/mask and exact native sample provenance.

    Uses nearest measured-Z z-buffer selection when native samples collide in
    one output pixel. No averaging, filling, random noise, geometric correction,
    ideal-depth access or depth/point-cloud consistency filtering is performed.
    ``source_index`` is a flattened source AOV index, or -1 for output holes.
    """
    points = np.asarray(pointcloud_usd)
    distance = np.asarray(native_distance_m)
    if distance.ndim == 3 and distance.shape[-1] == 1:
        distance = distance[..., 0]
    if points.ndim != 3 or points.shape[-1] not in (3, 4) or points.shape[:2] != distance.shape:
        raise ValueError("Pointcloud must be HxWx3/4 and native distance HxW/1 with matching samples")
    if distance.dtype.kind != "f" or not 0 <= min_depth_m < max_depth_m or not np.isfinite(max_depth_m):
        raise ValueError("Floating native distances and a finite positive sensor range are required")
    k = np.asarray(camera["K"], dtype=float)
    width, height = int(camera["width"]), int(camera["height"])
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or min(width, height) < 1):
        raise ValueError("A finite perspective K and positive output dimensions are required")
    if camera.get("model", "pinhole").lower() not in ("pinhole", "perspective"):
        raise ValueError("This registration helper validates perspective cameras only")
    offset = np.asarray(pointcloud_pixel_offset, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("A finite 2D source-correspondence offset is required")
    raw_valid = np.isfinite(distance) & (distance >= min_depth_m) & (distance <= max_depth_m) & (distance > 0)
    xyz = points[..., :3].astype(np.float64)
    # Native XYZ has USD axes: X right, Y up, Z backward. Its depth/range
    # magnitude is not used to decide agreement with the noisy measurement.
    xyz *= [1., -1., -1.]
    point_valid = np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
    if points.shape[-1] == 4:
        point_valid &= np.isfinite(points[..., 3]) & (points[..., 3] > 0)
    candidates = np.flatnonzero(raw_valid & point_valid)
    directions = xyz.reshape(-1, 3)[candidates]
    projection = directions @ k.T
    uv = projection[:, :2] / projection[:, 2:3] + offset
    inside = (np.isfinite(uv).all(-1) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    candidates, uv = candidates[inside], uv[inside]
    pixel = np.floor(uv[:, 0]).astype(np.int64) + width * np.floor(uv[:, 1]).astype(np.int64)
    measured = distance.reshape(-1)[candidates]
    # Tie-breaking is deterministic and preserves an actual source sample.
    order = np.lexsort((candidates, measured, pixel))
    ordered_pixel = pixel[order]
    keep = np.r_[True, ordered_pixel[1:] != ordered_pixel[:-1]] if len(order) else np.zeros(0, bool)
    chosen = order[keep]
    output = np.zeros(height * width, dtype=distance.dtype)
    source_index = np.full(height * width, -1, dtype=np.int64)
    output[pixel[chosen]] = measured[chosen]
    source_index[pixel[chosen]] = candidates[chosen]
    output = output.reshape(height, width)
    source_index = source_index.reshape(height, width)
    valid = source_index >= 0
    disagreement = np.abs(distance[raw_valid & point_valid].astype(float) - xyz[..., 2][raw_valid & point_valid])
    diagnostics = {
        "raw_valid_samples": int(raw_valid.sum()),
        "point_direction_invalid_samples": int(np.sum(raw_valid & ~point_valid)),
        "outside_rgb_image_samples": int(np.sum(raw_valid & point_valid)) - len(candidates),
        "projected_samples": len(candidates), "duplicate_or_occluded_samples": len(candidates) - len(chosen),
        "registered_pixels": int(valid.sum()),
        "pointcloud_geometry_depth_outside_native_range": int(np.sum(raw_valid & point_valid &
            ((xyz[..., 2] < min_depth_m) | (xyz[..., 2] > max_depth_m)))),
        "pointcloud_measurement_disagreement_gt_1um": int(np.sum(disagreement > 1e-6)),
        "pointcloud_measurement_disagreement_gt_0_1mm": int(np.sum(disagreement > 1e-4)),
        "consistency_filter_applied": False,
        "pointcloud_pixel_offset": offset.tolist(),
        "native_measurement_values_preserved_exactly": bool(np.array_equal(output[valid], distance.reshape(-1)[source_index[valid]])),
        "projection": "native left-camera USD XYZ to CV axes, runtime K, floor to pixel, measured-Z z-buffer",
        "depth": "raw native DepthSensorDistance optical Z in meters; invalid=0",
        "rgb_target": "original virtual depth camera LdrColor; not the separate physical D455 color imager",
    }
    return {"depth_m": output, "valid_mask": valid, "source_index": source_index, "diagnostics": diagnostics}


def analytic_plane_depth(camera, normal_world, offset):
    """Independent ray/plane optical-Z depth for renderer calibration controls."""
    k = np.asarray(camera["K"], dtype=float)
    transform = np.asarray(camera["T_world_camera"], dtype=float)
    normal = np.asarray(normal_world, dtype=float)
    if k.shape != (3, 3) or transform.shape != (4, 4) or normal.shape != (3,):
        raise ValueError("Expected K[3,3], world-from-CV-camera[4,4] and plane normal[3]")
    y, x = np.indices((camera["height"], camera["width"]))
    pixel = np.stack([x + .5, y + .5, np.ones_like(x)], -1)
    local = pixel @ np.linalg.inv(k).T
    rays = local @ transform[:3, :3].T
    denominator = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -(normal @ transform[:3, 3] + offset) / denominator
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0)
