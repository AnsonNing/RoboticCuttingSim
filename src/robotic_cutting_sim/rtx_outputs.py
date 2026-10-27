"""Decode the documented OVRTX 260-byte CameraParams output for unit/pose QA."""
import numpy as np


def decode_camera_params(raw):
    buffer = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1)
    if buffer.size != 260:
        raise ValueError("OVRTX CameraParams must contain exactly 260 bytes")
    world_to_camera_row = np.frombuffer(buffer, dtype="<f4", count=16).reshape(4, 4).astype(float)
    projection_row = np.frombuffer(buffer, dtype="<f4", count=16, offset=64).reshape(4, 4).astype(float)
    scale = float(np.frombuffer(buffer, dtype="<f4", count=1, offset=128)[0])
    width, height = map(int, np.frombuffer(buffer, dtype="<u4", count=2, offset=132))
    if scale <= 0 or not np.isfinite(scale) or min(width, height) <= 0:
        raise ValueError("Invalid runtime camera scale/resolution")
    usd_pose = np.linalg.inv(world_to_camera_row.T)
    usd_pose[:3, 3] *= scale
    cv_pose = usd_pose.copy()
    cv_pose[:3, :3] = usd_pose[:3, :3] @ np.diag([1, -1, -1])
    k = [[float(projection_row[0, 0]) * width / 2, 0, width * (1 - float(projection_row[2, 0])) / 2],
         [0, float(projection_row[1, 1]) * height / 2, height * (1 + float(projection_row[2, 1])) / 2],
         [0, 0, 1]]
    return {"meters_per_scene_unit": scale, "width": width, "height": height,
            "T_world_camera": cv_pose.tolist(), "K": k, "projection_row": projection_row.tolist(),
            "model": buffer[148:173].tobytes().split(b"\0")[0].decode("utf-8", errors="replace")}
