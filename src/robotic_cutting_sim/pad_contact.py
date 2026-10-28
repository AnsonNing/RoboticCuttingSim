"""Finite soft-pad torsional reduction, derived from the nominal overlap area.

This is an explicit uniform-pressure contact approximation, not calibrated
Franka pad compliance. It never changes geometry, mass, tool pose or adhesion.
"""
from functools import lru_cache
import numpy as np

POINT_PROFILE = "legacy_point_contact"
PATCH_PROFILE = "finite_pad_patch_v3"
# Actual composed official Panda USD Cube scales (size=2), in hand X/Z.
PAD_HALF_XZ_M = (.00875, .00825)

@lru_cache(maxsize=16)
def rectangular_mean_radius(half_x_m, half_z_m):
    """Area average of sqrt(x²+z²), using independent tensor Gauss quadrature."""
    a, b = float(half_x_m), float(half_z_m)
    if not np.isfinite([a, b]).all() or min(a, b) <= 0:
        raise ValueError("Contact patch half dimensions must be finite and positive")
    x, w = np.polynomial.legendre.leggauss(256)
    radius = np.sqrt((a*x[:, None])**2+(b*x[None, :])**2)
    return float(np.sum(radius*w[:, None]*w[None, :])/4)

def patch_receipt(handle_half_extents_m, friction):
    half = np.asarray(handle_half_extents_m, float)
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError("Expected finite positive tool handle half dimensions")
    if not np.isfinite(friction) or not 0 <= friction <= 2:
        raise ValueError("Finite sliding friction must be within0..2")
    overlap = (min(PAD_HALF_XZ_M[0], half[0]), min(PAD_HALF_XZ_M[1], half[2]))
    mean_radius = rectangular_mean_radius(*overlap)
    return {"profile": PATCH_PROFILE, "pad_half_xz_m": list(PAD_HALF_XZ_M),
        "nominal_overlap_half_xz_m": list(overlap), "mean_overlap_radius_m": mean_radius,
        "sliding_friction": float(friction), "torsional_friction_length_m": float(friction)*mean_radius,
        "pad_contact_dimension": 4, "rolling_resistance_enabled": False,
        "pressure_model": "uniform nominal overlap rectangle; finite Coulomb torsional reduction μ*mean_radius",
        "physical_calibration": "NOT_VALIDATED", "adhesion": 0.,
        "reference": "https://mujoco.readthedocs.io/en/latest/computation/#condim"}
