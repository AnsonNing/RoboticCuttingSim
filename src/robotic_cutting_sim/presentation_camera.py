"""Presentation-only overview camera for demo videos (ADR 0007 section 1).

The overview camera is RGB only, world fixed and authored only on request.
It is not part of any observation, camera contract, calibration file or
policy input. Its frames are recorded separately as ``overview_rgb_*`` for
the demo renderer. The pose here is pure geometry and needs no Isaac.
Matrices act on column vectors; ``T_world_camera`` uses OpenCV axes (+Z
forward, +Y down).
"""
import numpy as np

PRIM_PATH = "/World/PresentationCamera/Overview"
CAMERA_ID = "overview_rgb"
# 3/4 view from the front-right (world -y is the front, +x the right) of the workpiece.
EYE_OFFSET_M = (.38, -.68, .24)
WIDTH, HEIGHT, HFOV_DEG = 960, 540, 42.  # presentation only; 42 deg frames the board, knife and hand
FRANKA_BASES_WORLD_M = {"franka_0": (-.46, .12, -.01), "franka_1": (.46, .12, -.01)}
FRANKA_0_SHOULDER_WORLD_M = (-.46, .12, .323)  # approximate joint-2 height above the base


def _look_at_cv(eye, target, up=(0., 0., 1.)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    result[:3, 3] = eye
    return result


def intrinsics(width=WIDTH, height=HEIGHT, hfov_deg=HFOV_DEG):
    focal = width/(2*np.tan(np.deg2rad(hfov_deg)/2))
    return np.array([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])


def project(T_world_camera, K, points_world):
    """Pixel (u, v) and optical depth of WORLD points."""
    points = np.atleast_2d(np.asarray(points_world, float))
    camera = (np.linalg.inv(T_world_camera)@np.c_[points, np.ones(len(points))].T)[:3].T
    pixels = (np.asarray(K)@camera.T).T
    return pixels[:, :2]/pixels[:, 2:3], camera[:, 2]


def overview_camera_spec(task, *, eye_offset_m=EYE_OFFSET_M, width=WIDTH, height=HEIGHT, hfov_deg=HFOV_DEG):
    """World-fixed overview pose aimed at the workpiece top centre.

    Includes a projection check of the workpiece corners, both Franka bases
    and the task's reference edge at t = 0 (inside the image and in front of
    the camera), so the framing is recorded rather than assumed.
    """
    T = task.T_world_workpiece
    height_m = task.geometry.base_height_m
    target = T[:3, :3]@np.array([0., 0., height_m])+T[:3, 3]
    eye = target+np.asarray(eye_offset_m, float)
    T_world_camera = _look_at_cv(eye, target)
    K = intrinsics(width, height, hfov_deg)
    half = np.array([task.geometry.length_m/2, task.geometry.width_m/2, 0.])
    corners = [T[:3, :3]@(half*np.array([sx, sy, 0.])+np.array([0., 0., z]))+T[:3, 3]
               for sx in (-1, 1) for sy in (-1, 1) for z in (0., height_m)]
    named = {"workpiece_corners": np.asarray(corners), **{name: np.asarray([pose]) for name, pose in FRANKA_BASES_WORLD_M.items()},
             "franka_0_shoulder_approx": np.asarray([FRANKA_0_SHOULDER_WORLD_M]),
             "reference_edge_t0": np.asarray([task.reference_edge_pose_world(0.)[:3]])}
    framing = {}
    for name, points in named.items():
        pixels, depth = project(T_world_camera, K, points)
        inside = (depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        framing[name] = {"pixels_uv": pixels.tolist(), "depth_m": depth.tolist(), "all_inside": bool(inside.all())}
    return {"camera_id": CAMERA_ID, "prim_path": PRIM_PATH, "role": "presentation_only",
            "observation": False, "camera_contract": False, "modalities": ["rgb"],
            "eye_world_m": eye.tolist(), "look_at_target_world_m": target.tolist(),
            "look_at_target": "workpiece top centre (task.T_world_workpiece, local z = base height)",
            "eye_offset_from_target_m": list(map(float, eye_offset_m)),
            "working_distance_m": float(np.linalg.norm(eye-target)),
            "width": int(width), "height": int(height), "hfov_deg": float(hfov_deg), "K": K.tolist(),
            "T_world_camera": T_world_camera.tolist(), "convention": "OpenCV +Z forward, +Y down",
            "framing_check": framing,
            "scope": "World-fixed presentation RGB for demo videos; never an observation, calibration or policy input"}


def author_overview_camera(stage, spec):
    """Author the USD camera prim for ``spec`` (square pixels, near 1 mm)."""
    from pxr import Gf, UsdGeom
    from .camera_rig import _CV_TO_USD, _set_transform
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError("The overview camera requires an SI (1 m per unit) USD stage")
    UsdGeom.Xform.Define(stage, PRIM_PATH.rsplit("/", 1)[0])
    camera = UsdGeom.Camera.Define(stage, PRIM_PATH)
    _set_transform(camera.GetPrim(), np.asarray(spec["T_world_camera"])@_CV_TO_USD)
    camera.CreateProjectionAttr("perspective")
    aperture = float(20*np.tan(np.deg2rad(spec["hfov_deg"])/2))
    camera.CreateFocalLengthAttr(10.)
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture*spec["height"]/spec["width"])
    camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 20.))
    camera.GetPrim().SetCustomDataByKey("research:role", "presentation_only")
    return camera
