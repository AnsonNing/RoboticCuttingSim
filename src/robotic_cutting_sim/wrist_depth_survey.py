"""Wrist D405 depth survey of the SLICE-PUSH-01 bar (ADR 0009, controller profile ``path_v2_vision``).

Pure functions on one rendered depth frame: back-projection with the frame's
own K and ``T_world_camera`` (OpenCV optical convention, depth = optical Z in
metres, invalid = 0), selection of bar-top points, and sub-pixel face
positions along the bar axis. ``estimate_shift`` turns a calibration frame
(before cut 0) and a survey frame (after cut 0) into the rigid +x shift
estimate of the bar part that cut 1 will sever.

Inputs are only the depth image, its valid mask, the camera intrinsics and
extrinsics (robot kinematics) and the task's nominal geometry. No particle
state, labels, component ids or oracle positions are read here.

Face positions are workpiece-local x (``task.world_from_workpiece``). Along
each image line that runs along local x, the face feature of a material run
is where its top profile crosses a level 0.2 mm below the local top height:
interpolated between two depth samples, or (if the profile is not seen
falling) the midpoint between the last hit and the next pixel's ray at that
level. A level crossing instead of the silhouette makes the feature nearly
independent of the viewing angle on the rendered surface's rounded edges
(offline: per-frame spread 0.04 mm across camera poses, against 0.6 mm for
the raw silhouette). The frame value is the median over lines inside the
bar's nominal |y| band minus a margin.

Methods (``estimate_shift``): ``kerf_face`` is ADR 0009's formula (calibration
end face -> r_s; survey kerf face). ``end_face_plus_x`` (the path_v2_vision
default) applies the same r_s construction to the +x end face of the part
right of cut 0 in both frames: with the wrist camera 45 mm +x of the hand,
the -x end (calibration) and the cut-0 kerf (from the cut-1 start) are out
of view, while the +x end is in view from both settle poses. See
docs/evidence/phase3/demo3_wrist_survey_offline_001.json.
"""
from dataclasses import asdict, dataclass

import numpy as np

# Outermost particle centres of the 1 mm SLICE-PUSH-01 lattice (workpiece-local x, m).
OUTER_PARTICLE_CENTRE_LOCAL_X = {-1: -.0795, 1: .0795}
# First particle centre on the +x side of the cut-0 plane x = -30 mm (ADR 0009).
KERF_RIGHT_PARTICLE_CENTRE_LOCAL_X = -.0295
MAX_ABS_SHIFT_M = .006

MATERIAL, BACKGROUND, EXCLUDED = 1, 0, -1


@dataclass(frozen=True)
class BarGeometry:
    """Nominal bar geometry (task constants known before the episode, never material state)."""
    T_world_workpiece: tuple      # 4x4 nested tuple
    board_top_local_z: float
    bar_top_local_z: float
    half_width_m: float
    half_length_m: float

    @classmethod
    def from_task(cls, task):
        boards = [box for box in task.fixture_boxes_local if getattr(box, "role", "board") == "board"]
        board_top = max(box.center_m[2]+box.half_extents_m[2] for box in (boards or task.fixture_boxes_local))
        return cls(tuple(map(tuple, np.asarray(task.T_world_workpiece, float))), float(board_top),
                   float(board_top+task.geometry.base_height_m), float(task.geometry.width_m/2),
                   float(task.geometry.length_m/2))

    @property
    def T(self):
        return np.asarray(self.T_world_workpiece, float)


@dataclass(frozen=True)
class SurveyConfig:
    height_margin_m: float = .003          # bar-top band: board top + margin .. bar top + margin
    y_margin_m: float = .003               # lines with |y| <= half width - margin
    blade_clearance_m: float = .002        # exclude points this close to the blade slab (if an edge pose is given)
    blade_half_thickness_m: float = .00075
    min_material_px: int = 3               # material samples in a run and in its reference window
    reference_inner_m: tuple = (.002, .006)  # top-height reference window inside a run end
    level_drop_m: float = .0002            # feature level below the reference top height
    min_lines: int = 15                    # lines with a bracketed face
    max_line_spread_m: float = .0006       # 1.4826*MAD over lines
    max_bracket_m: float = .002            # a hit/miss bracket wider than this is not a face
    kerf_band_m: tuple = (-.006, .006)     # kerf right face search band around the expected plane (+-max shift)
    max_abs_shift_m: float = MAX_ABS_SHIFT_M


def back_project(depth_m, K, T_world_camera, valid_mask=None):
    """World points, pixel rays and validity of one depth frame (OpenCV optical, depth = optical Z)."""
    depth = np.asarray(depth_m, float)
    K, T = np.asarray(K, float), np.asarray(T_world_camera, float)
    if depth.ndim != 2 or K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError("Need an HxW depth image, a 3x3 K and a 4x4 T_world_camera")
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, bool)
    v, u = np.mgrid[:depth.shape[0], :depth.shape[1]]
    rays_camera = np.stack([(u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones(depth.shape)], -1)
    rays = rays_camera@T[:3, :3].T
    points = T[:3, 3]+rays*np.where(valid, depth, 0.)[..., None]
    return dict(points_world=points, rays_world=rays, origin_world=T[:3, 3].copy(), valid=valid)


def _to_local(cloud, geometry):
    T = geometry.T
    R, t = T[:3, :3], T[:3, 3]
    points = (cloud["points_world"]-t)@R
    rays = cloud["rays_world"]@R
    origin = (cloud["origin_world"]-t)@R
    return points, rays, origin


def select_bar_top(cloud, geometry, config=SurveyConfig(), *, edge_pose_world=None):
    """Per-pixel class (MATERIAL / BACKGROUND / EXCLUDED) and local geometry of one back-projected frame.

    MATERIAL: valid, local z in (board top + margin, bar top + margin], not near the blade.
    BACKGROUND: valid and at or below board top + margin (board, gaps, table).
    EXCLUDED: invalid, above bar top + margin (knife, fingers, hand), or in the blade slab.
    """
    points, rays, origin = _to_local(cloud, geometry)
    z = points[..., 2]
    labels = np.full(z.shape, EXCLUDED, np.int8)
    valid = cloud["valid"]
    low = geometry.board_top_local_z+config.height_margin_m
    high = geometry.bar_top_local_z+config.height_margin_m
    labels[valid & (z <= low)] = BACKGROUND
    labels[valid & (z > low) & (z <= high)] = MATERIAL
    knife = None
    if edge_pose_world is not None:
        from scipy.spatial.transform import Rotation
        edge = np.asarray(edge_pose_world, float)
        R = geometry.T[:3, :3]
        edge_local = (edge[:3]-geometry.T[:3, 3])@R
        normal_local = R.T@Rotation.from_quat(edge[3:]).apply([0., 0., 1.])  # blade normal
        distance = np.abs((points-edge_local)@normal_local)
        near_blade = valid & (distance <= config.blade_half_thickness_m+config.blade_clearance_m) & \
            (z >= edge_local[2]-config.blade_clearance_m)
        labels[near_blade] = EXCLUDED
        knife = dict(edge_local_m=edge_local.tolist(), blade_normal_local=normal_local.tolist(),
                     excluded_px=int(near_blade.sum()))
    # Image axis along local x: the one whose pixel step has the larger local-x component.
    axis_dirs = np.asarray(cloud["rays_world"][1, 1]-cloud["rays_world"][1, 0]), \
        np.asarray(cloud["rays_world"][1, 1]-cloud["rays_world"][0, 1])
    R = geometry.T[:3, :3]
    along_u, along_v = (R.T@axis_dirs[0])[0], (R.T@axis_dirs[1])[0]
    axis = 1 if abs(along_u) >= abs(along_v) else 0       # array axis of the scan (1 = columns u, 0 = rows v)
    sign = np.sign(along_u if axis == 1 else along_v)
    return dict(labels=labels, points_local=points, rays_local=rays, origin_local=origin, scan_axis=axis,
                scan_sign=float(sign), counts=dict(valid_px=int(valid.sum()), material_px=int((labels == MATERIAL).sum()),
                                                   background_px=int((labels == BACKGROUND).sum())),
                knife=knife)


def _lines(selection, geometry, config):
    """(labels, x, z, ray, y) per scan line ordered along +local x, for lines inside the |y| band."""
    labels, points, rays = selection["labels"], selection["points_local"], selection["rays_local"]
    if selection["scan_axis"] == 0:  # scan along rows v: transpose so each line is a row of the array
        labels, points, rays = labels.T, points.transpose(1, 0, 2), rays.transpose(1, 0, 2)
    if selection["scan_sign"] < 0:
        labels, points, rays = labels[:, ::-1], points[:, ::-1], rays[:, ::-1]
    limit = geometry.half_width_m-config.y_margin_m
    for line in range(labels.shape[0]):
        material = labels[line] == MATERIAL
        if material.sum() < config.min_material_px:
            continue
        y = float(np.median(points[line, material, 1]))
        if abs(y) > limit:
            continue
        yield labels[line], points[line], rays[line], y


def _ray_x_at(origin, ray, z):
    if abs(ray[2]) < 1e-12:
        return np.nan
    return float(origin[0]+ray[0]*(z-origin[2])/ray[2])


def _summarise(values, brackets, lines, config, name):
    values, brackets = np.asarray(values, float), np.asarray(brackets, float)
    record = dict(face=name, lines_scanned=int(lines), lines_used=int(len(values)))
    if len(values) < config.min_lines:
        return dict(record, valid=False, reason=f"{name}: {len(values)} bracketed lines < {config.min_lines}", x_m=None)
    x = float(np.median(values))
    spread = float(1.4826*np.median(np.abs(values-x)))
    record.update(x_m=x, line_spread_m=spread, bracket_median_m=float(np.median(brackets)),
                  x_p05_m=float(np.percentile(values, 5)), x_p95_m=float(np.percentile(values, 95)))
    if spread > config.max_line_spread_m:
        return dict(record, valid=False, reason=f"{name}: line spread {spread:.6f} m > {config.max_line_spread_m}")
    return dict(record, valid=True, reason=None)


def _face_feature(labels, points, rays, origin, outer, side, config):
    """(feature x, bracket width) of the material run ending at ``outer`` whose outward direction is ``side``.

    The feature is the outermost crossing of the top profile with the level
    ``z_ref - level_drop`` (z_ref: median height 2-6 mm inside the run end),
    searched from the run end inward (noise inside the flat top cannot end it
    early). It is linearly interpolated between two surface samples when the
    visible profile crosses the level, else bracketed between the outermost
    hit and the next pixel's ray at the level (midpoint). None if the run is
    too short or no sample of its last 2 mm is above the level.
    """
    x, z = points[:, 0], points[:, 2]
    run = [outer]
    while 0 <= run[-1]-side < len(labels) and labels[run[-1]-side] == MATERIAL:
        run.append(run[-1]-side)
    run = np.asarray(run)                                  # outer -> inner
    depth = side*(x[outer]-x[run])                        # distance inside the run end (>= 0)
    reference = run[(depth >= config.reference_inner_m[0]) & (depth <= config.reference_inner_m[1])]
    if len(run) < config.min_material_px or len(reference) < config.min_material_px:
        return None
    level = float(np.median(z[reference]))-config.level_drop_m
    near = run[depth <= config.reference_inner_m[0]]       # outer -> inner, the last 2 mm of the run
    below = 0
    while below < len(near) and z[near[below]] <= level:    # outermost samples at or below the level
        below += 1
    if below == len(near):
        return None
    if below:
        inner, outer_sample = near[below], near[below-1]
        weight = (z[inner]-level)/(z[inner]-z[outer_sample])
        return float(x[inner]+weight*(x[outer_sample]-x[inner])), abs(float(x[outer_sample]-x[inner]))
    miss = _ray_x_at(origin, rays[outer+side], level)
    bracket = side*(miss-x[outer])
    if not np.isfinite(miss) or not 0 <= bracket <= config.max_bracket_m:
        return None
    return float((x[outer]+miss)/2), float(bracket)


def end_face_x(selection, geometry, side=-1, config=SurveyConfig()):
    """Local x of the bar's ``side`` end face (-1: -x end, +1: +x end) from one selection.

    Per line: the outermost material run on ``side`` must be followed by a
    background pixel (else the end is occluded or outside the view on that line).
    """
    if side not in (-1, 1):
        raise ValueError("side is -1 or +1")
    origin = selection["origin_local"]
    values, brackets, lines = [], [], 0
    for labels, points, rays, _ in _lines(selection, geometry, config):
        lines += 1
        material = np.nonzero(labels == MATERIAL)[0]
        outer = int(material.max() if side > 0 else material.min())
        if not 0 <= outer+side < len(labels) or labels[outer+side] != BACKGROUND:
            continue
        feature = _face_feature(labels, points, rays, origin, outer, side, config)
        if feature is not None:
            values.append(feature[0])
            brackets.append(feature[1])
    return _summarise(values, brackets, lines, config, f"end_face_{'+' if side > 0 else '-'}x")


def kerf_right_face_x(selection, geometry, expected_x=-.030, band=None, config=SurveyConfig()):
    """Local x of the first material face at x > ``expected_x`` scanning along +x inside ``band``.

    The face is a background -> material transition (the -x-facing left face of the
    part right of the kerf) whose material pixel lies in [expected + band[0], expected + band[1]].
    """
    low, high = (config.kerf_band_m if band is None else band)
    origin = selection["origin_local"]
    values, brackets, lines = [], [], 0
    for labels, points, rays, _ in _lines(selection, geometry, config):
        lines += 1
        x = points[:, 0]
        for i in range(1, len(labels)):
            if labels[i] != MATERIAL or labels[i-1] != BACKGROUND or not expected_x+low <= x[i] <= expected_x+high:
                continue
            feature = _face_feature(labels, points, rays, origin, i, -1, config)
            if feature is not None:
                values.append(feature[0])
                brackets.append(feature[1])
            break
    return _summarise(values, brackets, lines, config, "kerf_right_face")


def survey_frame(frame, geometry, config=SurveyConfig(), *, edge_pose_world=None, faces=("end_-x", "end_+x", "kerf")):
    """Measurements of one observation dict (depth_m, valid_mask, K, T_world_camera, physics_timestamp_s)."""
    cloud = back_project(frame["depth_m"], frame["K"], frame["T_world_camera"], frame.get("valid_mask"))
    selection = select_bar_top(cloud, geometry, config, edge_pose_world=edge_pose_world)
    record = dict(physics_timestamp_s=float(frame["physics_timestamp_s"]), camera_id=str(frame.get("camera_id", "")),
                  counts=selection["counts"], knife=selection["knife"],
                  camera_origin_local_m=selection["origin_local"].tolist())
    if "end_-x" in faces:
        record["end_minus_x"] = end_face_x(selection, geometry, -1, config)
    if "end_+x" in faces:
        record["end_plus_x"] = end_face_x(selection, geometry, 1, config)
    if "kerf" in faces:
        record["kerf_right"] = kerf_right_face_x(selection, geometry, config=config)
    return record


METHODS = ("kerf_face", "end_face_plus_x")


def estimate_shift(calibration, survey, *, method="end_face_plus_x", config=SurveyConfig()):
    """(shift estimate in m or None, validity record) from two ``survey_frame`` records.

    ``kerf_face`` (ADR 0009 formulas): r_s from an end face in the calibration
    frame (the -x end if measured, else the +x end, both against the outermost
    particle centres -/+79.5 mm), then shift = x_kf + 29.5 mm + r_s.
    ``end_face_plus_x``: the same r_s from the +x end in the calibration frame;
    the survey frame measures the same +x end face of the part right of cut 0,
    so shift = (x_end,+ - r_s) - 79.5 mm = x_end,+ - x_end,+,0.
    Invalid (shift None) if a face is missing or invalid, or |shift| > max.
    """
    if method not in METHODS:
        raise ValueError(f"method is one of {METHODS}")
    reasons = []
    side, face = None, None
    preferred = ("end_plus_x",) if method == "end_face_plus_x" else ("end_minus_x", "end_plus_x")
    for key in preferred:
        if (calibration.get(key) or {}).get("valid"):
            side, face = (1 if key == "end_plus_x" else -1), calibration[key]
            break
    if face is None:
        reasons.append("calibration: no valid end face (" + "; ".join(
            str((calibration.get(key) or {}).get("reason")) for key in preferred) + ")")
    r_s = None if face is None else side*face["x_m"]-OUTER_PARTICLE_CENTRE_LOCAL_X[1]
    measured = survey.get("kerf_right" if method == "kerf_face" else "end_plus_x") or {}
    if not measured.get("valid"):
        reasons.append(f"survey: {measured.get('reason', 'face not measured')}")
    shift = None
    if not reasons:
        if method == "kerf_face":
            shift = measured["x_m"]-KERF_RIGHT_PARTICLE_CENTRE_LOCAL_X+r_s
        else:
            shift = measured["x_m"]-r_s-OUTER_PARTICLE_CENTRE_LOCAL_X[1]
        if not abs(shift) <= config.max_abs_shift_m:
            reasons.append(f"|shift| {abs(shift):.6f} m > {config.max_abs_shift_m}")
    valid = not reasons
    record = dict(method=method, valid=valid, reasons=reasons,
                  shift_m=float(shift) if shift is not None else None,
                  applied_shift_m=float(shift) if valid else 0.,
                  r_s_m=None if r_s is None else float(r_s), calibration_face=None if face is None else
                  ("end_plus_x" if side > 0 else "end_minus_x"),
                  calibration_face_x_m=None if face is None else face["x_m"],
                  survey_face=("kerf_right" if method == "kerf_face" else "end_plus_x"),
                  survey_face_x_m=measured.get("x_m"),
                  calibration_timestamp_s=calibration.get("physics_timestamp_s"),
                  survey_timestamp_s=survey.get("physics_timestamp_s"),
                  config=asdict(config))
    return (float(shift) if valid else None), record
