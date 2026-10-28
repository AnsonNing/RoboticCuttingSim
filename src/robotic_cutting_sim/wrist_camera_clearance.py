"""Sampled visual-surface checks; never a substitute for native contacts."""
import numpy as np

from .scene_frames import validate_transform


def surface_owner(prim_path, body_paths, excluded_prefixes=()):
    """Assign nested USD visuals to their closest recorded rigid-body ancestor."""
    if any(prim_path == prefix or prim_path.startswith(prefix+"/") for prefix in excluded_prefixes):
        return None
    matches = [path for path in body_paths if prim_path == path or prim_path.startswith(path+"/")]
    return max(matches, key=len) if matches else None


def intentional_attachment(box_name, body_path, parent_hand_path):
    """A bracket may meet its parent palm; the exception never extends to fingers."""
    return box_name == "Bracket" and body_path == parent_hand_path


def triangle_obb_intersections(triangles, transform, dimensions, tolerance_m=1e-8):
    """Triangle/OBB SAT with a distance tolerance, retaining boundary contact.

    This is a triangle-surface test, not a closed-volume containment test.
    """
    transform = validate_transform(transform)
    triangles = (np.asarray(triangles, float)-transform[:3, 3])@transform[:3, :3]
    dimensions = np.asarray(dimensions, float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("Expected N triangles, a rigid box transform and positive full dimensions")
    if not np.isfinite(triangles).all() or not np.isfinite(dimensions).all() or tolerance_m < 0:
        raise ValueError("Finite triangles and nonnegative distance tolerance are required")
    half = dimensions/2
    edges = (triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 1], triangles[:, 0]-triangles[:, 2])
    axes = [np.broadcast_to(a, (len(triangles), 3)) for a in np.eye(3)]
    axes += [np.cross(edges[0], edges[1])]
    axes += [np.cross(edge, axis) for edge in edges for axis in np.eye(3)]
    overlap = np.ones(len(triangles), bool)
    for axis in axes:
        projection = np.sum(triangles*axis[:, None, :], axis=2)
        radius = np.abs(axis)@half
        epsilon = tolerance_m*np.linalg.norm(axis, axis=1)
        overlap &= (projection.min(1) <= radius+epsilon) & (projection.max(1) >= -radius-epsilon)
    return overlap


def clipped_triangle_vertices_in_box(triangles, transform, dimensions):
    """Clip triangle surfaces to six box planes; return box-local vertices.

    These are intersections of visual surfaces, not an overlap volume. The
    zero-tolerance clipping keeps SAT's separate 10 nm boundary tolerance from
    manufacturing an apparent finite intersection extent.
    """
    transform = validate_transform(transform)
    local = (np.asarray(triangles, float)-transform[:3, 3])@transform[:3, :3]
    half = np.asarray(dimensions, float)/2
    clipped = []
    for triangle in local:
        polygon = list(triangle)
        for axis in range(3):
            for sign in (-1, 1):
                result = []
                for start, end in zip(polygon, polygon[1:]+polygon[:1]):
                    d0, d1 = sign*start[axis]-half[axis], sign*end[axis]-half[axis]
                    if d0 <= 0:
                        result.append(start)
                    if (d0 <= 0) != (d1 <= 0):
                        result.append(start+(end-start)*(d0/(d0-d1)))
                polygon = result
                if not polygon:
                    break
            if not polygon:
                break
        clipped.extend(polygon)
    return np.asarray(clipped, float).reshape(-1, 3)
