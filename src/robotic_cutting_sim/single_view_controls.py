"""Independent analytic geometry and fixed RGB/depth feature acceptance."""
import numpy as np

from .single_view_registration import analytic_plane_depth


def analytic_control(camera, case, scene):
    """Intersect camera rays with a known plane and optional world-axis box."""
    plane = analytic_plane_depth(camera, [-case["slope_x"], -case["slope_y"], 1.], -case["distance_m"])
    if not case.get("box"):
        return plane, plane.copy(), np.zeros_like(plane, dtype=bool)
    k = np.asarray(camera["K"])
    pose = np.asarray(camera["T_world_camera"])
    y, x = np.indices(plane.shape)
    rays = np.stack([x + .5, y + .5, np.ones_like(x)], -1) @ np.linalg.inv(k).T @ pose[:3, :3].T
    center = np.array([*scene["box_center_xy_m"], case["distance_m"] - scene["box_center_z_below_plane_origin_m"]])
    half = np.asarray(scene["box_half_extents_m"])
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = (center - half - pose[:3, 3]) / rays, (center + half - pose[:3, 3]) / rays
    near, far = np.max(np.minimum(a, b), -1), np.min(np.maximum(a, b), -1)
    hit = (far >= near) & (near > 0) & (near < plane)
    return plane, np.where(hit, near, plane), hit


def feature_acceptance(rgb, depth, valid, plane, analytic_box, rules):
    """All analytic feature scanlines count, including missing native data."""
    color = np.asarray(rgb).astype(float)
    red = ((color[..., 0] > rules["rgb_red_minimum_8bit"]) &
           (color[..., 0] > rules["rgb_red_to_green_ratio"] * color[..., 1]) &
           (color[..., 0] > rules["rgb_red_to_blue_ratio"] * color[..., 2]))
    measured_box = valid & (depth < plane - rules["depth_below_analytic_plane_minimum_m"])
    union = np.sum(red | analytic_box)
    errors = []
    missing = 0
    for axis in (0, 1):
        expected = np.moveaxis(analytic_box, axis, 0)
        rgb_lines = np.moveaxis(red, axis, 0)
        depth_lines = np.moveaxis(measured_box, axis, 0)
        for i in np.flatnonzero(expected.any(1)):
            rgb_pixels, depth_pixels = np.flatnonzero(rgb_lines[i]), np.flatnonzero(depth_lines[i])
            if not len(rgb_pixels) or not len(depth_pixels):
                errors.extend([expected.shape[1], expected.shape[1]])
                missing += 1
            else:
                errors.extend([abs(int(rgb_pixels[0])-int(depth_pixels[0])),
                               abs(int(rgb_pixels[-1])-int(depth_pixels[-1]))])
    return {"analytic_feature_pixels": int(analytic_box.sum()), "rgb_feature_pixels": int(red.sum()),
        "depth_feature_pixels": int(measured_box.sum()), "scanlines": len(errors)//2,
        "missing_feature_scanlines": missing,
        "rgb_vs_analytic_mask_iou": float(np.sum(red & analytic_box) / union) if union else 0.,
        "depth_vs_rgb_mask_iou": float(np.sum(red & measured_box) / np.sum(red | measured_box)) if np.any(red | measured_box) else 0.,
        "boundary_error_p95_px": float(np.quantile(errors, .95)) if errors else None,
        "boundary_error_max_px": float(max(errors)) if errors else None}
