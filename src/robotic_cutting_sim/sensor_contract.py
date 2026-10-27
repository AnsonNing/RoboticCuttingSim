"""Explicit camera modes and payload checks, without importing a renderer."""
from enum import StrEnum

import numpy as np


class CameraMode(StrEnum):
    IDEAL_RGBD = "ideal_rgbd"
    SIMULATED_D455 = "simulated_d455"


def mode_contract(mode):
    mode = CameraMode(mode)
    return {
        "mode": mode.value,
        "renderer": "Warp ray camera" if mode == CameraMode.IDEAL_RGBD else "Isaac Sim 6.2 RTX",
        "depth_model": ("ideal pinhole optical Z" if mode == CameraMode.IDEAL_RGBD
                        else "official D455 SingleViewDepthCameraSensor post-processing"),
        "depth_convention": "optical Z in meters; invalid=0",
        "camera_axes": "OpenCV +X right, +Y down, +Z forward",
        "calibration_status": "provisional; not validated against a physical D455",
        "force_observation": False,
        "oracle_observation": False,
    }


def validate_frame(rgb, depth_m, physics_timestamp_s, surface_timestamp_s):
    """Reject stale geometry, nonfinite depth and malformed programmable frames."""
    rgb, depth = np.asarray(rgb), np.asarray(depth_m)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB HxWx3 and depth HxW")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Depth must be finite and nonnegative; use zero for invalid")
    times = np.array([physics_timestamp_s, surface_timestamp_s], float)
    if not np.isfinite(times).all() or np.min(times) < 0 or abs(times[0] - times[1]) > 1e-9:
        raise ValueError("Physics and surface timestamps must match and be nonnegative")
    return {"rgb": rgb, "depth_m": depth.astype(np.float32, copy=False),
            "valid_mask": depth > 0, "timestamp_s": float(times[0]),
            "surface_timestamp_s": float(times[1])}
