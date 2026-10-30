"""ADR 0013 skim orientation metrics (tau, psi, rho, mu) of recorded edge poses.

Read-only helper for TRIM receipts. Edge frame convention (tool_profile): +x
the advance (away from the blade body), +y the edge line, +z the blade normal.
With target normal n_t and in-plane cutting direction c:

- edge tilt   tau = arcsin |e . n_t|                      (edge parallel to the surface: 0)
- edge skew   psi = angle between the lines e and n_t x c (edge square to the cut: 0)
- blade rake  rho = signed angle from n_t to n_b about e; positive when the blade
                    body rises behind the edge (advance tilted toward -n_t)
- motion elevation mu = arcsin(m . n_t) of the edge-centre velocity direction m.

These are geometry/kinematics descriptors, never material state or observations.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _unit(vector, what):
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm <= 0:
        raise ValueError(f"{what} must be a finite nonzero 3-vector")
    return vector/norm


def pose_orientation_metrics(quaternions_xyzw, target_normal, cutting_direction):
    """Per-pose (tau, psi, rho) in degrees for (T, 4) or (4,) edge orientations."""
    n_t = _unit(target_normal, "Target normal")
    c = _unit(cutting_direction, "Cutting direction")
    c = c-(c@n_t)*n_t
    if np.linalg.norm(c) < 1e-9:
        raise ValueError("Cutting direction must not be parallel to the target normal")
    c /= np.linalg.norm(c)
    square = np.cross(n_t, c)
    rotations = Rotation.from_quat(np.atleast_2d(np.asarray(quaternions_xyzw, float)))
    d, e, n_b = (rotations.apply(axis) for axis in np.eye(3))
    tau = np.degrees(np.arcsin(np.clip(np.abs(e@n_t), 0, 1)))
    psi = np.degrees(np.arccos(np.clip(np.abs(e@square), 0, 1)))
    # n_t projected onto the plane normal to e, expressed in the (d, n_b) basis of that plane.
    rho = np.degrees(np.arctan2(-(d@n_t), n_b@n_t))
    return tau, psi, rho


def motion_elevation_deg(positions, target_normal, *, min_step_m=1e-7):
    """mu per recorded step (deg) from consecutive edge-centre positions; NaN where the edge did not move."""
    n_t = _unit(target_normal, "Target normal")
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("Need at least two edge-centre positions")
    step = np.diff(positions, axis=0)
    length = np.linalg.norm(step, axis=1)
    mu = np.full(len(step), np.nan)
    moving = length > min_step_m
    mu[moving] = np.degrees(np.arcsin(np.clip(step[moving]@n_t/length[moving], -1, 1)))
    return mu


def _stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
            "max_abs": float(np.abs(values).max()), "samples": int(len(values))}


def skim_orientation_report(edge_poses_world, target_normal, cutting_direction, *, source="commanded"):
    """tau/psi/rho/mu summary (deg) over a sequence of WORLD edge poses (T, 7)."""
    poses = np.asarray(edge_poses_world, float)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("Need at least two finite world edge poses (x y z qx qy qz qw)")
    tau, psi, rho = pose_orientation_metrics(poses[:, 3:], target_normal, cutting_direction)
    mu = motion_elevation_deg(poses[:, :3], target_normal)
    return {"source": source, "poses": int(len(poses)), "tau_edge_tilt_deg": _stats(tau),
            "psi_edge_skew_deg": _stats(psi), "rho_blade_rake_deg": _stats(rho), "mu_motion_elevation_deg": _stats(mu),
            "target_normal": _unit(target_normal, "Target normal").tolist(),
            "cutting_direction": _unit(cutting_direction, "Cutting direction").tolist(),
            "convention": "docs/adr/0013 (rho > 0: blade body rises behind the edge)"}
