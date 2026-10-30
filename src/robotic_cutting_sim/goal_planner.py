"""Scripted goal planner: scene + goal config -> knife-edge Waypoints (ADR 0015 §4).

``plan(scene, goal, tool_profile, options)`` builds a ``policies.Waypoint`` list
for ``WaypointPolicy`` from the SCENE CONFIG ONLY (declared workpiece geometry,
pose, protrusions and supports) and the goal. It never reads particle, label,
component or any other oracle material state; there is no task or environment
argument. Poses are WORLD knife-edge XYZ+XYZW (the scene pose, including yaw,
maps the local plan to world). Waypoint labels are ``PHASE`` or ``PHASE:op``.

Operations:

* sever / divide plane -> generalized through-cut. The blade normal is the plane
  normal, the advance ``a`` is the declared ``approach_local`` (default: local -z
  projected into the plane) and the edge line is ``n x a``:
  APPROACH (above the start) -> PRE_CUT (start, ``clearance_m`` before the
  material along ``a``) -> CUT (armed, its own cut digit, to ``overtravel_m``
  beyond the material, or onto the support top when the plane has no relief
  slot under it) -> HOLD -> WITHDRAW (back along -a, then up).
* remove_protrusions -> one ``surface_trim`` pass per selected protrusion
  (ascending along the skim direction): REORIENT (free space above the pass
  start) -> PRE_CUT (edge at plane + edge clearance) -> SKIM (armed, own digit,
  declared rake/speed) -> WITHDRAW (up); then the next protrusion. trim_plane
  is one pass over its region.
* move_piece -> blade-face push with cutting disabled: APPROACH_PUSH -> LOWER
  (into the kerf) -> PUSH -> PUSH_HOLD -> RELEASE -> RETRACT.
* sequence -> the steps in order; cut digits are numbered across the episode.

Every plan starts at HOME and ends with RETURN_HOME (the declared home edge pose).
A scene whose workcell scenario declares ``home_joint_positions`` (wp11_workcell_v1)
gets HOME = FK(q_home) composed with the tool's hand_from_edge (blade hanging below
the fingers) unless the planner options declare a home explicitly.

Orientation metrics (ADR 0013): ``orientation_metrics`` returns edge tilt tau,
edge skew psi, blade rake rho and motion elevation mu; ``skim_rotation``
builds a skim edge frame with tau = 0 and declared psi/rho.
"""
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from .control import ActionLimits
from .policies import CONTROLLER_PROFILES, Waypoint, controller_profile, shape_approaches

PLANNER_VERSION = "goal_planner_v1"
DEFAULT_OPTIONS = {
    "controller_profile": "v1",     # v1 | path_v2 (policies.CONTROLLER_PROFILES)
    "home_position_local_m": None,  # None: (0, 0, top + home_height_m)
    "home_orientation_local_xyzw": None,  # None: vertical cut frame (advance -z, edge +y, normal +x)
    "home_height_m": .04,
    "safe_height_m": .03,           # free-space travel height above the highest material point
    "cut_speed_m_s": .01,
    "cut_clearance_m": .008,
    "overtravel_m": .004,
    "hold_s": .3,
    "kerf_speed_m_s": .02,
    "saw_amplitude_m": 0.,
    "saw_period_s": .4,
    "rake_deg": 35.,                # ADR 0013 rho for surface_trim / trim_plane passes
    "skew_deg": 0.,
    "skim_speed_m_s": .025,
    "skim_margin_m": .01,
    "skim_direction_local": [1., 0., 0.],
    "protrusion_order": "along_skim",   # along_skim | goal
    "push_release_m": .002,
    "max_translation_m": .002,      # ActionLimits used for the reference schedule
    "max_rotation_deg": 2.,
}
# Opt-in options (WP-11 Phase E): accepted when declared, absent otherwise, so plans, controller
# configs and compiled ids of goals that do not declare them are unchanged.
OPT_IN_OPTIONS = {
    # "bbox" (the v1 rule, used when absent): a through-cut spans the whole workpiece bounding box along
    # the advance. "plane_section": it spans only the cut plane's section of that box, so an inclined
    # advance (oblique plane) ends overtravel_m past the section instead of past the far box corner.
    "cut_extent": ("bbox", "plane_section"),
    # ADR 0016: wrist-depth survey + relocation of the later cuts of a >= 2-plane sever goal
    # (perception_relocate). Absent: no survey waypoints, plans and reports unchanged.
    "perception": ("wrist_depth_relocate_v1",),
}


def planner_options(goal, overrides=None):
    """Defaults <- goal["planner"] <- overrides, validated (unknown keys are errors)."""
    merged = dict(DEFAULT_OPTIONS)
    for source in (goal.get("planner") or {}, overrides or {}):
        unknown = set(source)-set(DEFAULT_OPTIONS)-set(OPT_IN_OPTIONS)
        if unknown:
            from .scene_config import SceneConfigError
            raise SceneConfigError(f"Unknown planner options {sorted(unknown)}")
        merged.update(source)
    if merged["controller_profile"] not in CONTROLLER_PROFILES or merged["controller_profile"] == "path_v2_vision":
        from .scene_config import SceneConfigError
        raise SceneConfigError("planner controller_profile must be v1, path_v2, path_v3 or path_v3a")
    if merged["protrusion_order"] not in ("along_skim", "goal"):
        from .scene_config import SceneConfigError
        raise SceneConfigError("planner protrusion_order must be along_skim or goal")
    for key, allowed in OPT_IN_OPTIONS.items():
        if key in merged and merged[key] not in allowed:
            from .scene_config import SceneConfigError
            raise SceneConfigError(f"planner {key} must be one of {allowed}")
    if merged.get("perception") is not None:
        from .perception_relocate import check_options
        check_options(merged)
    return merged


def action_limits(options, scene=None):
    """Step limits from the planner options; the workspace box from the scene workcell when it declares one."""
    extra = {}
    if scene is not None:
        from .scene_config import workcell_action_workspace
        box = workcell_action_workspace(scene)
        if box is not None:
            extra = dict(workspace_min_local_m=box[0], workspace_max_local_m=box[1])
    return ActionLimits(max_translation_m=float(options["max_translation_m"]),
                        max_rotation_rad=float(np.deg2rad(options["max_rotation_deg"])), **extra)


def workcell_home_edge_pose(scene, tool_profile):
    """WORLD edge pose of the workcell HOME (hand FK(q_home) composed with hand_from_edge), or None."""
    from .scene_config import workcell_home
    home = workcell_home(scene)
    if home is None:
        return None
    from .scene_frames import matrix_pose, pose_matrix
    return np.asarray(matrix_pose(pose_matrix(home["hand_pose_world_xyzw"])@pose_matrix(tool_profile.hand_from_edge())), float)


def policy_options(options):
    return controller_profile(options["controller_profile"])[0]


def ideal_duration_s(task, waypoints, limits, options, max_s=600.):
    """Ideal-integrator time the plan needs (inf beyond ``max_s``); for error messages only."""
    from .policies import ideal_plan_end_s
    value = ideal_plan_end_s(task, waypoints, max_s=max_s, limits=limits, **policy_options(options))
    return float("inf") if value is None else value


# ---------------------------------------------------------------- frames and ADR 0013 metrics

def _normalize(vector, what="vector"):
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"{what} is zero or not finite")
    return vector/norm


def edge_rotation(advance, edge_line, blade_normal):
    """Edge-frame rotation with columns (advance x, edge line y, blade normal z); right-handed."""
    R = np.column_stack([_normalize(advance), _normalize(edge_line), _normalize(blade_normal)])
    if not np.allclose(R.T@R, np.eye(3), atol=1e-9) or np.linalg.det(R) < 0:
        raise ValueError("Edge frame axes must be orthonormal and right-handed")
    return Rotation.from_matrix(R)


def through_cut_rotation(plane_normal, approach=None):
    """Blade normal = plane normal, advance = approach projected into the plane (default -z), edge = n x a."""
    n = _normalize(plane_normal, "plane normal")
    a = np.array([0., 0., -1.]) if approach is None else np.asarray(approach, float)
    a = a-(a@n)*n
    if np.linalg.norm(a) < 1e-6:
        raise ValueError("The approach lies along the plane normal; declare approach_local inside the plane")
    a = _normalize(a)
    return edge_rotation(a, np.cross(n, a), n)


def skim_rotation(cut_direction, target_normal=(0., 0., 1.), rake_deg=0., skew_deg=0.):
    """Skim edge frame with tau = 0, psi = skew and rho = rake (ADR 0013).

    The edge line is ``n_t x c`` turned by psi about n_t; the advance is the
    in-surface direction square to it, tilted by rho below the surface so the
    blade face rises behind the edge (rho > 0).
    """
    n_t = _normalize(target_normal, "target normal")
    c = np.asarray(cut_direction, float)
    c = _normalize(c-(c@n_t)*n_t, "cut direction in the target plane")
    e = Rotation.from_rotvec(np.deg2rad(skew_deg)*n_t).apply(np.cross(n_t, c))
    x0 = np.cross(e, n_t)
    rho = np.deg2rad(rake_deg)
    return edge_rotation(np.cos(rho)*x0-np.sin(rho)*n_t, e, np.cos(rho)*n_t+np.sin(rho)*x0)


def orientation_metrics(edge_orientation_xyzw, motion=None, target_normal=(0., 0., 1.), cut_direction=None):
    """ADR 0013 tau, psi, rho, mu (degrees) of an edge orientation and motion, in one frame.

    tau = asin|e.n_t|; psi = angle between e and n_t x c (sign-free, c the cut
    direction, default the motion projected into the target plane); rho =
    signed angle from n_t (projected normal to e) to the blade normal n_b about
    e, positive when the blade face rises behind the edge; mu = asin(m.n_t).
    """
    R = Rotation.from_quat(edge_orientation_xyzw).as_matrix()
    e, n_b = R[:, 1], R[:, 2]
    n_t = _normalize(target_normal, "target normal")
    tau = np.degrees(np.arcsin(np.clip(abs(e@n_t), 0, 1)))
    projected = n_t-(n_t@e)*e
    rho = None if np.linalg.norm(projected) < 1e-9 else float(np.degrees(np.arctan2(e@np.cross(projected, n_b), projected@n_b)))
    m = None if motion is None or np.linalg.norm(motion) < 1e-12 else _normalize(motion)
    mu = None if m is None else float(np.degrees(np.arcsin(np.clip(m@n_t, -1, 1))))
    if cut_direction is None and m is not None and np.linalg.norm(m-(m@n_t)*n_t) > 1e-9:
        cut_direction = m-(m@n_t)*n_t
    psi = None
    if cut_direction is not None:
        u = _normalize(np.cross(n_t, _normalize(cut_direction)))
        psi = float(np.degrees(np.arccos(np.clip(abs(e@u), 0, 1))))
    return {"tau_deg": float(tau), "psi_deg": psi, "rho_deg": rho, "mu_deg": mu,
            "definition": "docs/adr/0013 (tau edge tilt, psi edge skew, rho blade rake, mu motion elevation)"}


# ---------------------------------------------------------------- scene geometry (config only)

class _Frame:
    def __init__(self, scene):
        from .scene_config import scene_world_pose
        pose = scene_world_pose(scene)
        self.R = Rotation.from_quat(pose[3:])
        self.t = np.asarray(pose[:3], float)

    def pose(self, position_local, rotation_local):
        return tuple(map(float, np.r_[self.R.apply(position_local)+self.t, (self.R*rotation_local).as_quat()]))


def _bbox_corners(scene):
    from .scene_config import scene_top_m
    base = scene["workpiece"]["base_box_m"]
    hi = np.array([base["length_m"]/2, base["width_m"]/2, scene_top_m(scene)])
    lo = np.array([-hi[0], -hi[1], 0.])
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _support_boxes(scene):
    from .scene_config import raw_support_boxes
    return [(np.asarray(b.center_m, float), np.asarray(b.half_extents_m, float)) for b in raw_support_boxes(scene)]


def _relief_end(scene, start, a, e, t_end, profile):
    """(t_end, end_on_support): stop on the support top when the edge line below the bar hits a support."""
    base = scene["workpiece"]["base_box_m"]
    reach = min(profile.half_width_m, np.hypot(base["length_m"], base["width_m"])/2+.005)
    margin = profile.thickness_m/2+5e-4
    end = start+a*t_end
    samples = end+np.linspace(-reach, reach, 49)[:, None]*e
    footprint = (np.abs(samples[:, 0]) <= base["length_m"]/2+.005) & (np.abs(samples[:, 1]) <= base["width_m"]/2+.005)
    samples = samples[footprint & (samples[:, 2] < 0)]
    hits = [c for c, h in _support_boxes(scene) if len(samples) and np.any(np.all(np.abs(samples-c) <= h+margin, axis=1))]
    if not hits or a[2] >= -1e-9:
        return t_end, False
    top = max(c[2]+h[2] for c, h in _support_boxes(scene))
    return (top-start[2])/a[2], True


def _home(scene, options, tool_profile=None):
    from .scene_config import scene_top_m
    edge = None if tool_profile is None else workcell_home_edge_pose(scene, tool_profile)
    if edge is not None:
        # Workcell HOME (WP-11 Phase D): the knife edge at FK(q_home), expressed in local
        # coordinates; an explicitly declared home position/orientation still wins.
        frame = _Frame(scene)
        position = (frame.R.inv().apply(edge[:3]-frame.t) if options["home_position_local_m"] is None
                    else np.asarray(options["home_position_local_m"], float))
        rotation = (frame.R.inv()*Rotation.from_quat(edge[3:]) if options["home_orientation_local_xyzw"] is None
                    else Rotation.from_quat(options["home_orientation_local_xyzw"]))
        return position, rotation
    position = options["home_position_local_m"]
    position = np.array([0., 0., scene_top_m(scene)+options["home_height_m"]]) if position is None else np.asarray(position, float)
    rotation = (through_cut_rotation([1., 0., 0.]) if options["home_orientation_local_xyzw"] is None
                else Rotation.from_quat(options["home_orientation_local_xyzw"]))
    return position, rotation


def _plane_section(corners, q, n):
    """Points where the plane (q, n) crosses the 12 edges of the box given by its 8 corners (None if it misses)."""
    signed = (corners-q)@n
    points = []
    for i in range(8):
        for bit in (1, 2, 4):
            j = i | bit
            if j == i:
                continue
            a, b = signed[i], signed[j]
            if a == b:
                if a == 0:
                    points += [corners[i], corners[j]]
                continue
            if min(a, b) <= 0 <= max(a, b):
                points.append(corners[i]+(corners[j]-corners[i])*a/(a-b))
    return np.asarray(points) if points else None


def _through_cut(scene, origin, normal, approach, digit, profile, options, frame):
    rotation = through_cut_rotation(normal, approach)
    R = rotation.as_matrix()
    a, e, n = R[:, 0], R[:, 1], R[:, 2]
    base = scene["workpiece"]["base_box_m"]
    centre = np.array([0., 0., base["height_m"]/2])
    q = centre-((centre-origin)@n)*n
    corners = _bbox_corners(scene)
    section = _plane_section(corners, q, n) if options.get("cut_extent", "bbox") == "plane_section" else None
    t = ((corners if section is None else section)-q)@a
    start = q+a*(t.min()-options["cut_clearance_m"])
    t_end, on_support = _relief_end(scene, start, a, e, (t.max()-t.min())+options["cut_clearance_m"]+options["overtravel_m"], profile)
    end = start+a*t_end
    up = np.array([0., 0., options["safe_height_m"]])
    op = f"cut{digit}"
    plan = [Waypoint(frame.pose(start+up, rotation), label=f"APPROACH:{op}"),
            Waypoint(frame.pose(start, rotation), label=f"PRE_CUT:{op}"),
            Waypoint(frame.pose(end, rotation), cut=True, cut_index=digit, speed_m_s=options["cut_speed_m_s"],
                     saw_amplitude_m=options["saw_amplitude_m"], saw_period_s=options["saw_period_s"], label=f"CUT:{op}"),
            Waypoint(frame.pose(end, rotation), hold_s=options["hold_s"], label=f"HOLD:{op}"),
            Waypoint(frame.pose(start, rotation), speed_m_s=options["kerf_speed_m_s"], label=f"WITHDRAW:{op}"),
            Waypoint(frame.pose(start+up, rotation), label=f"WITHDRAW:{op}")]
    report = {"op": op, "kind": "through_cut", "digit": digit, "plane_origin_local_m": list(map(float, origin)),
              "plane_normal_local": list(map(float, n)), "advance_local": a.tolist(), "edge_line_local": e.tolist(),
              "start_local_m": start.tolist(), "end_local_m": end.tolist(), "end_on_support": bool(on_support)}
    if "cut_extent" in options:
        report["cut_extent"] = options["cut_extent"] if section is not None or options["cut_extent"] == "bbox" else "bbox_fallback"
    return plan, report


def _support_along(bump, direction):
    """Half extent of a protrusion footprint along a unit in-plane direction (rectangle bound, conservative)."""
    return abs(direction[0])*bump["radius_x_m"]+abs(direction[1])*bump["radius_y_m"]


def _skim_pass(scene, centre_xy, low, high, plane_z, clearance, digit, op, profile, options, frame, others=()):
    c = np.r_[_normalize(np.r_[options["skim_direction_local"][:2], 0.], "skim direction")[:2], 0.]
    rotation = skim_rotation(c, rake_deg=options["rake_deg"], skew_deg=options["skew_deg"])
    from .scene_config import scene_top_m
    centre = np.r_[centre_xy, 0.]
    z = plane_z+clearance
    start, end = centre+c*low, centre+c*high
    start[2] = end[2] = z
    safe = scene_top_m(scene)+options["safe_height_m"]
    plan = [Waypoint(frame.pose(np.r_[start[:2], safe], rotation), label=f"REORIENT:{op}"),
            Waypoint(frame.pose(start, rotation), label=f"PRE_CUT:{op}"),
            Waypoint(frame.pose(end, rotation), cut=True, cut_index=digit, speed_m_s=options["skim_speed_m_s"],
                     saw_amplitude_m=options["saw_amplitude_m"], saw_period_s=options["saw_period_s"], label=f"SKIM:{op}"),
            Waypoint(frame.pose(np.r_[end[:2], safe], rotation), label=f"WITHDRAW:{op}")]
    metrics = orientation_metrics(rotation.as_quat(), end-start)
    lateral = np.cross([0., 0., 1.], c)
    warnings = []
    for name, bump in others:
        offset = np.asarray([bump["x_m"], bump["y_m"], 0.])-centre
        along = offset@c
        if low-_support_along(bump, c) < along < high+_support_along(bump, c):
            warnings.append(f"pass {op} crosses the footprint of protrusion {name}")
    report = {"op": op, "kind": "surface_trim", "digit": digit, "plane_local_z_m": float(plane_z),
              "edge_clearance_m": float(clearance), "start_local_m": start.tolist(), "end_local_m": end.tolist(),
              "direction_local": c.tolist(), "edge_line_local": lateral.tolist(), "declared_rake_deg": options["rake_deg"],
              "declared_skew_deg": options["skew_deg"], "orientation_metrics": metrics, "warnings": warnings}
    return plan, report


def _protrusion_passes(scene, goal, digit, profile, options, frame):
    rows = scene["workpiece"]["protrusions"]
    ids = [row["id"] for row in rows]
    select = goal["select"]
    if select == "all":
        chosen = list(range(len(rows)))
    elif isinstance(select, list):
        unknown = [s for s in select if s not in ids]
        if unknown:
            raise ValueError(f"Goal selects unknown protrusions {unknown}")
        chosen = [ids.index(s) for s in select]
    else:
        region = select["region_local_m"]
        chosen = [i for i, r in enumerate(rows) if ("x" not in region or region["x"][0] <= r["x_m"] <= region["x"][1])
                  and ("y" not in region or region["y"][0] <= r["y_m"] <= region["y"][1])]
    if not chosen:
        raise ValueError("remove_protrusions selects no protrusion of this scene")
    c = _normalize(np.r_[options["skim_direction_local"][:2], 0.])
    along = {i: np.array([rows[i]["x_m"], rows[i]["y_m"], 0.])@c for i in range(len(rows))}
    if options["protrusion_order"] == "along_skim":
        chosen = sorted(chosen, key=lambda i: along[i])
    plane_z = scene["workpiece"]["base_box_m"]["height_m"] if goal["plane_local_z_m"] is None else goal["plane_local_z_m"]
    plan, reports = [], []
    for i in chosen:
        bump, reach = rows[i], _support_along(rows[i], c)
        low, high = -reach-options["skim_margin_m"], reach+options["skim_margin_m"]
        for j, other in enumerate(rows):  # never run a pass into a neighbour: clip at the mid gap
            if j == i:
                continue
            gap_centre = along[j]-along[i]
            other_reach = _support_along(other, c)
            if gap_centre > 0:
                high = min(high, (reach+gap_centre-other_reach)/2)
            elif gap_centre < 0:
                low = max(low, (-reach+gap_centre+other_reach)/2)
        others = [(rows[j]["id"], rows[j]) for j in range(len(rows)) if j != i]
        steps, report = _skim_pass(scene, [bump["x_m"], bump["y_m"]], low, high, plane_z, goal["edge_clearance_m"],
                                   digit, bump["id"], profile, options, frame, others)
        plan += steps
        reports.append(report)
        digit += 1
    return plan, reports, digit


def _trim_plane_pass(scene, goal, digit, profile, options, frame):
    normal = np.asarray(goal["plane"]["normal_local"], float)
    if not np.allclose(normal, [0., 0., 1.], atol=1e-9):
        raise NotImplementedError("goal_planner trim_plane supports horizontal (+z) planes only")
    base = scene["workpiece"]["base_box_m"]
    region = goal["region_local_m"] or {}
    x = region.get("x", [-base["length_m"]/2, base["length_m"]/2])
    y = region.get("y", [-base["width_m"]/2, base["width_m"]/2])
    c = _normalize(np.r_[options["skim_direction_local"][:2], 0.])
    centre = np.array([(x[0]+x[1])/2, (y[0]+y[1])/2, 0.])
    corners = np.array([[a, b, 0.] for a in x for b in y])-centre
    along = corners@c
    steps, report = _skim_pass(scene, centre[:2], along.min()-options["skim_margin_m"], along.max()+options["skim_margin_m"],
                               goal["plane"]["origin_local_m"][2], goal["edge_clearance_m"], digit, f"trim{digit}",
                               profile, options, frame)
    return steps, [report], digit+1


def _surface_sweep(scene, goal, step_index, profile, options, frame):
    """move_piece push_mode surface_sweep (WP-11 C5b): a disarmed near-horizontal sweep of the base top.

    From SCENE CONFIG ONLY (never where the piece actually is): the edge, in the
    skim frame (declared rake/skew, edge leading along ``direction_local``),
    descends ``skim_margin_m`` behind the selector box at ``sweep_height_m``
    above the base top and sweeps to ``sweep_overtravel_m`` past the base end,
    so every position the piece can have on the base top ahead of the box is
    swept. APPROACH_PUSH -> LOWER -> PUSH -> PUSH_HOLD -> RELEASE -> RETRACT.
    """
    from .scene_config import scene_top_m
    d = _normalize(goal["direction_local"], "push direction")
    if abs(d[2]) > 1e-9:
        raise ValueError("surface_sweep pushes along the base top: direction_local must be horizontal")
    if "box_local_m" not in goal["selector"]:
        raise ValueError("surface_sweep needs a box_local_m selector (where the piece starts)")
    box = goal["selector"]["box_local_m"]
    low, high = np.asarray(box["min"], float), np.asarray(box["max"], float)
    corners = np.array([[x, y, 0.] for x in (low[0], high[0]) for y in (low[1], high[1])])
    base = scene["workpiece"]["base_box_m"]
    base_corners = np.array([[x, y, 0.] for x in (-base["length_m"]/2, base["length_m"]/2)
                             for y in (-base["width_m"]/2, base["width_m"]/2)])
    centre = np.r_[(low[:2]+high[:2])/2, 0.]
    lateral = centre-(centre@d)*d
    start_along = float((corners@d).min())-options["skim_margin_m"]
    end_along = float((base_corners@d).max())+goal.get("sweep_overtravel_m", .005)
    z = base["height_m"]+goal.get("sweep_height_m", .002)
    start, end = lateral+d*start_along, lateral+d*end_along
    start[2] = end[2] = z
    rotation = skim_rotation(d, rake_deg=options["rake_deg"], skew_deg=options["skew_deg"])
    safe = scene_top_m(scene)+options["safe_height_m"]
    released = end-options["push_release_m"]*d
    op = f"move{step_index}"
    speed = goal["push_speed_m_s"]
    plan = [Waypoint(frame.pose(np.r_[start[:2], safe], rotation), label=f"APPROACH_PUSH:{op}"),
            Waypoint(frame.pose(start, rotation), speed_m_s=options["kerf_speed_m_s"], label=f"LOWER:{op}"),
            Waypoint(frame.pose(end, rotation), speed_m_s=speed, label=f"PUSH:{op}"),
            Waypoint(frame.pose(end, rotation), hold_s=options["hold_s"], label=f"PUSH_HOLD:{op}"),
            Waypoint(frame.pose(released, rotation), speed_m_s=speed, label=f"RELEASE:{op}"),
            Waypoint(frame.pose(np.r_[released[:2], safe], rotation), label=f"RETRACT:{op}")]
    reach = profile.half_width_m
    warnings = [f"protrusion {row['id']} lies in the sweep corridor" for row in scene["workpiece"]["protrusions"]
                if not (low[0] <= row["x_m"] <= high[0] and low[1] <= row["y_m"] <= high[1])
                and start_along-_support_along(row, d) <= np.array([row["x_m"], row["y_m"], 0.])@d
                <= end_along+_support_along(row, d)
                and abs((np.array([row["x_m"], row["y_m"], 0.])-lateral)@np.cross([0., 0., 1.], d)) <= reach]
    report = {"op": op, "kind": "surface_sweep_push", "digit": None, "push_mode": "surface_sweep",
              "start_local_m": start.tolist(), "end_local_m": end.tolist(), "direction_local": d.tolist(),
              "sweep_height_above_base_top_m": float(z-base["height_m"]), "edge_half_length_m": float(reach),
              "orientation_metrics": orientation_metrics(rotation.as_quat(), end-start), "cut_enabled": False,
              "base_length_m": base["length_m"], "distance_m_used": False, "warnings": warnings,
              "inputs": "selector box, base box, protrusions and tool profile from the configs; not the piece's state"}
    return plan, [report]


def _push(scene, goal, step_index, profile, options, frame):
    from .scene_config import scene_top_m
    if goal.get("push_mode") == "surface_sweep":
        return _surface_sweep(scene, goal, step_index, profile, options, frame)
    d = _normalize(goal["direction_local"], "push direction")
    rotation = through_cut_rotation(d)
    base = scene["workpiece"]["base_box_m"]
    selector = goal["selector"]
    if "side_of_plane" in selector:
        row = selector["side_of_plane"]
        origin, normal = np.asarray(row["origin_local_m"], float), np.asarray(row["normal_local"], float)
        sign = 1. if row["side"] == "positive" else -1.
        if sign*(normal@d) <= 0:
            raise ValueError("The push direction must point away from the cut plane into the selected side")
        centre = np.zeros(3)
        face = centre-((centre-origin)@normal)*normal
    else:
        box = selector["box_local_m"]
        low, high = np.asarray(box["min"], float), np.asarray(box["max"], float)
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        centre = (low+high)/2
        face = centre+(corners@d).min()*d-(centre@d)*d-profile.thickness_m*d
    support_top = max(c[2]+h[2] for c, h in _support_boxes(scene))
    kerf = np.r_[face[:2], support_top+goal["push_edge_height_m"]]
    safe = np.r_[face[:2], scene_top_m(scene)+options["safe_height_m"]]
    pushed = kerf+goal["distance_m"]*d
    released = pushed-options["push_release_m"]*d
    op = f"move{step_index}"
    speed = goal["push_speed_m_s"]
    plan = [Waypoint(frame.pose(safe, rotation), label=f"APPROACH_PUSH:{op}"),
            Waypoint(frame.pose(kerf, rotation), speed_m_s=options["kerf_speed_m_s"], label=f"LOWER:{op}"),
            Waypoint(frame.pose(pushed, rotation), speed_m_s=speed, label=f"PUSH:{op}"),
            Waypoint(frame.pose(pushed, rotation), hold_s=options["hold_s"], label=f"PUSH_HOLD:{op}"),
            Waypoint(frame.pose(released, rotation), speed_m_s=speed, label=f"RELEASE:{op}"),
            Waypoint(frame.pose(np.r_[released[:2], safe[2]], rotation), label=f"RETRACT:{op}")]
    report = {"op": op, "kind": "blade_face_push", "digit": None, "kerf_local_m": kerf.tolist(),
              "pushed_to_local_m": pushed.tolist(), "direction_local": d.tolist(), "cut_enabled": False,
              "base_length_m": base["length_m"]}
    return plan, [report]


def _operations(scene, goal, profile, options, frame, digit=0, step_index=0):
    from .goals import goal_planes
    kind = goal["type"]
    if kind in ("sever", "divide"):
        plan, reports = [], []
        approaches = [row.get("approach_local") for row in goal["planes"]] if kind == "sever" else None
        planes = goal_planes(goal, scene)
        survey = None
        if options.get("perception") is not None and kind == "sever" and len(planes) >= 2:
            from . import perception_relocate as hook
            side, normal = hook.eligible_side(goal, planes)
            survey = (hook, side, through_cut_rotation(normal))
            steps, op = hook.survey_waypoints(scene, side, survey[2], 0, options, frame)
            plan += steps
            reports.append({"op": op, "kind": "perception_survey", "role": "calibration", "digit": None, "side": side,
                            "survey_pose_local_m": hook.survey_position_local(scene, side, options).tolist()})
        for k, (origin, normal) in enumerate(planes):
            if survey is not None and k > 0:
                steps, op = survey[0].survey_waypoints(scene, survey[1], survey[2], k, options, frame)
                plan += steps
                reports.append({"op": op, "kind": "perception_survey", "role": "survey", "digit": None,
                                "side": survey[1], "next_cut_op": f"cut{digit}",
                                "survey_pose_local_m": survey[0].survey_position_local(scene, survey[1], options).tolist()})
            steps, report = _through_cut(scene, origin, normal, None if approaches is None else approaches[k],
                                         digit, profile, options, frame)
            plan += steps
            reports.append(report)
            digit += 1
        return plan, reports, digit
    if kind == "remove_protrusions":
        return _protrusion_passes(scene, goal, digit, profile, options, frame)
    if kind == "trim_plane":
        return _trim_plane_pass(scene, goal, digit, profile, options, frame)
    if kind == "move_piece":
        plan, reports = _push(scene, goal, step_index, profile, options, frame)
        return plan, reports, digit
    plan, reports = [], []
    for index, step in enumerate(goal["steps"]):
        steps, step_reports, digit = _operations(scene, step, profile, options, frame, digit, index)
        plan += steps
        reports += [{**row, "step": index, "step_goal_id": step["goal_id"]} for row in step_reports]
    return plan, reports, digit


def plan_with_report(scene, goal, tool_profile, options=None):
    """(waypoints, report). Inputs: scene/goal configs, a ToolProfile and planner options only."""
    from .goals import validate_goal
    from .scene_config import resolve_scene
    scene, goal = resolve_scene(scene), validate_goal(goal)
    options = planner_options(goal, options) if options is None or set(options) != set(DEFAULT_OPTIONS) else dict(options)
    frame = _Frame(scene)
    home_position, home_rotation = _home(scene, options, tool_profile)
    home = Waypoint(frame.pose(home_position, home_rotation), label="HOME")
    body, reports, digits = _operations(scene, goal, tool_profile, options, frame)
    plan = [home, *body, Waypoint(home.pose, label="RETURN_HOME")]
    _, shaping = controller_profile(options["controller_profile"])
    plan = shape_approaches(plan, **shaping)
    report = {"planner": PLANNER_VERSION, "scene_id": scene["scene_id"], "goal_id": goal["goal_id"],
              "tool_profile_id": tool_profile.profile_id, "options": options, "operations": reports,
              "digits_used": digits, "phases": [w.label for w in plan],
              "inputs": "scene config (declared geometry, pose, protrusions, supports), goal config, tool profile, "
                        "planner options; no particle, label, component or oracle state"}
    if options.get("perception") is not None:  # ADR 0016 (absent otherwise: reports of other plans are unchanged)
        from .perception_relocate import perception_block
        report["perception"] = perception_block(reports, options)
    from .scene_config import workcell_home
    workcell = workcell_home(scene)
    if workcell is not None:  # only scenes whose workcell declares a HOME (reports of other scenes are unchanged)
        report["home"] = {"source": "workcell", "scenario": workcell["scenario"],
                          "joint_positions": list(workcell["joint_positions"]),
                          "hand_pose_world_xyzw": list(workcell["hand_pose_world_xyzw"]),
                          "edge_pose_world_xyzw": list(map(float, home.pose)),
                          "explicit_option_override": [key for key in ("home_position_local_m", "home_orientation_local_xyzw")
                                                       if options[key] is not None]}
    return plan, report


def plan(scene, goal, tool_profile, options=None):
    """WaypointPolicy waypoints (WORLD edge poses) for a goal on a scene; see the module docstring."""
    return plan_with_report(scene, goal, tool_profile, options)[0]
