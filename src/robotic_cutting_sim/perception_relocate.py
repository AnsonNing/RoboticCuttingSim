"""Wrist-depth relocation as an opt-in planner perception hook (ADR 0016, ``wrist_depth_relocate_v1``).

For a configured ``sever`` goal with >= 2 vertical planes (one shared normal along
local +-x, strictly monotonic offsets) the goal planner, when its options declare
``perception = "wrist_depth_relocate_v1"``, inserts planned survey waypoints:
``survey0`` (calibration) after HOME and ``survey{k+1}`` after cut k's withdrawal.
The survey pose is derived from the scene config only (no look-at aiming, no
observation-pose optimization, no visibility gate): the knife edge in the cuts'
own orientation at local (s (L/2 + 10 mm), 0, top + cut_clearance_m), s = the
side the cut sequence progresses toward.

At runtime ``RelocatePolicy`` takes the current wrist D405_0 depth frame at the
end of each SURVEY hold, measures the bar end face on side s with the unchanged
``wrist_depth_survey.end_face_x`` and shifts every later non-survey waypoint up
to RETURN_HOME by the end face displacement since calibration (local x).

Inputs: depth / valid mask / K / T_world_camera of the frame, the nominal edge
from joint-state proprioception (blade exclusion), the scene config. Never
particle, label, component, oracle or tool-truth state. The tracking feedback of
the underlying WaypointPolicy (``edge_estimator``) is a separate input of the
runner and is not used here.
"""
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .policies import Waypoint, WaypointPolicy, edge_from_observation

HOOK = "wrist_depth_relocate_v1"
HOOKS = (HOOK,)
ADR = "docs/adr/0016-configured-perception-relocation.md"
CAMERA_ID = "wrist_d405_0"
SURVEY_OFFSET_M = .010           # edge beyond the measured end face, along the progression side
SURVEY_SLOW_DISTANCE_M = .010    # last part of the descent at the profile approach speed
MAX_ABS_SHIFT_M = .006
REQUIRED_PROFILE = "path_v2"
SURVEY_PHASES = ("SURVEY_APPROACH", "SURVEY", "SURVEY_LEAVE")
INPUTS = {
    "deployable": ["wrist_d405_0 depth_m, valid_mask, K (rendered sensor)",
                   "T_world_camera of the frame (robot kinematics + the fixed hand-eye mount)",
                   "nominal edge pose from joint-state proprioception (policies.edge_from_observation; blade exclusion)",
                   "scene config (base box, world pose, supports) and goal planes (survey pose, bands, side)"],
    "privileged_not_used_by_the_hook": ["measured tool-edge pose (--edge-feedback measured): WaypointPolicy tracking "
                                        "feedback only, simulator tool truth, as live_001"],
    "never_used": ["particle positions", "cut labels / codes", "mechanical components", "oracle part positions"],
}


def _error(text):
    from .scene_config import SceneConfigError
    return SceneConfigError(text)


def check_options(options):
    """Validate the hook option set (called by goal_planner.planner_options when ``perception`` is declared)."""
    hook = options.get("perception")
    if hook is None:
        return
    if hook not in HOOKS:
        raise _error(f"planner perception must be one of {HOOKS}")
    if options["controller_profile"] != REQUIRED_PROFILE:
        raise _error(f"planner perception {hook} is declared for controller_profile {REQUIRED_PROFILE} only")


def eligible_side(goal, planes):
    """(side s, unit normal) for an eligible sever goal; SceneConfigError otherwise."""
    if goal["type"] != "sever" or len(planes) < 2:
        raise _error(f"perception {HOOK} needs a sever goal with >= 2 planes")
    if any(row.get("approach_local") is not None for row in goal["planes"]):
        raise _error(f"perception {HOOK} supports the default vertical approach only (no approach_local)")
    normal = np.asarray(planes[0][1], float)
    if not all(np.allclose(np.asarray(n, float), normal, atol=1e-9) for _, n in planes) or \
            not np.isclose(abs(normal[0]), 1., atol=1e-9):
        raise _error(f"perception {HOOK} needs planes sharing one normal along local +-x")
    offsets = np.array([np.asarray(o, float)@normal for o, _ in planes])
    steps = np.diff(offsets)
    if not (np.all(steps > 1e-9) or np.all(steps < -1e-9)):
        raise _error(f"perception {HOOK} needs strictly monotonic plane offsets (each next cut on the outer piece)")
    direction = np.sign(steps[0])*normal
    return int(np.sign(direction[0])), normal


def survey_position_local(scene, side, options):
    from .scene_config import scene_top_m
    base = scene["workpiece"]["base_box_m"]
    return np.array([side*(base["length_m"]/2+SURVEY_OFFSET_M), 0., scene_top_m(scene)+options["cut_clearance_m"]])


def survey_waypoints(scene, side, rotation, index, options, frame):
    """SURVEY_APPROACH (safe, then slow start) -> SURVEY (approach speed, settle hold) -> SURVEY_LEAVE."""
    from .policies import controller_profile
    _, shaping = controller_profile(options["controller_profile"])
    here = survey_position_local(scene, side, options)
    up = np.array([0., 0., options["safe_height_m"]])
    slow = np.array([0., 0., SURVEY_SLOW_DISTANCE_M])
    op = f"survey{index}"
    plan = [Waypoint(frame.pose(here+up, rotation), label=f"SURVEY_APPROACH:{op}"),
            Waypoint(frame.pose(here+slow, rotation), label=f"SURVEY_APPROACH:{op}"),
            Waypoint(frame.pose(here, rotation), speed_m_s=shaping["approach_speed_m_s"], hold_s=shaping["settle_s"],
                     label=f"SURVEY:{op}"),
            Waypoint(frame.pose(here+up, rotation), label=f"SURVEY_LEAVE:{op}")]
    return plan, op


def perception_block(reports, options):
    """Plan-report block of the hook (exactly one eligible sever goal), read by ``RelocatePolicy``."""
    rows = [r for r in reports if r.get("kind") == "perception_survey"]
    calibrations = [r for r in rows if r["role"] == "calibration"]
    if len(calibrations) != 1:
        raise _error(f"perception {HOOK} needs exactly one sever goal with >= 2 planes; found {len(calibrations)}")
    return {"hook": HOOK, "adr": ADR, "camera_id": CAMERA_ID, "side": calibrations[0]["side"],
            "survey_pose_local_m": calibrations[0]["survey_pose_local_m"], "survey_offset_m": SURVEY_OFFSET_M,
            "survey_height": "scene top + cut_clearance_m", "settle_and_approach": options["controller_profile"],
            "calibration_op": calibrations[0]["op"],
            "surveys": [{"op": r["op"], "next_cut_op": r["next_cut_op"]} for r in rows if r["role"] == "survey"],
            "max_abs_shift_m": MAX_ABS_SHIFT_M, "method": "end_face_relative (wrist_depth_survey.end_face_x)",
            "shift_rule": "each valid survey sets the offset R_world_local (shift, 0, 0) of every non-SURVEY waypoint from "
                          "the next cut's first waypoint up to RETURN_HOME (exclusive)",
            "inputs": INPUTS}


def scene_bar_geometry(scene):
    """``wrist_depth_survey.BarGeometry`` from the scene config (declared pose, base box, supports; never material state)."""
    from scipy.spatial.transform import Rotation
    from .scene_config import raw_support_boxes, resolve_scene, scene_top_m, scene_world_pose
    from .wrist_depth_survey import BarGeometry
    scene = resolve_scene(scene)
    pose = scene_world_pose(scene)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    boxes = raw_support_boxes(scene)
    board_top = max(b.center_m[2]+b.half_extents_m[2] for b in boxes) if boxes else 0.
    base = scene["workpiece"]["base_box_m"]
    return BarGeometry(tuple(map(tuple, T)), float(board_top), float(scene_top_m(scene)),
                       float(base["width_m"]/2), float(base["length_m"]/2))


def measure_end_face(frame, geometry, side, config=None, *, edge_pose_world=None):
    """End-face record of one depth frame (``wrist_depth_survey.end_face_x`` on ``side``)."""
    from .wrist_depth_survey import SurveyConfig, back_project, end_face_x, select_bar_top
    config = SurveyConfig() if config is None else config
    cloud = back_project(frame["depth_m"], frame["K"], frame["T_world_camera"], frame.get("valid_mask"))
    selection = select_bar_top(cloud, geometry, config, edge_pose_world=edge_pose_world)
    return dict(physics_timestamp_s=float(frame["physics_timestamp_s"]), camera_id=str(frame.get("camera_id", "")),
                counts=selection["counts"], knife=selection["knife"],
                camera_origin_local_m=selection["origin_local"].tolist(), side=int(side),
                end_face=end_face_x(selection, geometry, side, config))


def relative_shift(calibration, survey, max_abs_shift_m=MAX_ABS_SHIFT_M):
    """(shift along local +x or None, record) = x_face(survey) - x_face(calibration)."""
    reasons = []
    for name, record in (("calibration", calibration), ("survey", survey)):
        face = (record or {}).get("end_face") or {}
        if not face.get("valid"):
            reasons.append(f"{name}: {face.get('reason', 'not measured')}")
    shift = None
    if not reasons:
        shift = float(survey["end_face"]["x_m"]-calibration["end_face"]["x_m"])
        if not abs(shift) <= max_abs_shift_m:
            reasons.append(f"|shift| {abs(shift):.6f} m > {max_abs_shift_m}")
    valid = not reasons
    return (shift if valid else None), dict(
        method="end_face_relative", valid=valid, reasons=reasons, shift_m=shift, applied_shift_m=shift if valid else 0.,
        calibration_face_x_m=((calibration or {}).get("end_face") or {}).get("x_m"),
        survey_face_x_m=((survey or {}).get("end_face") or {}).get("x_m"),
        calibration_timestamp_s=(calibration or {}).get("physics_timestamp_s"),
        survey_timestamp_s=(survey or {}).get("physics_timestamp_s"), max_abs_shift_m=max_abs_shift_m)


def events_from_plan(waypoints, perception):
    """Calibration / survey waypoint indices and the waypoints each survey shifts, from the final plan labels."""
    labels = [w.label for w in waypoints]

    def index_of(label):
        hits = [i for i, value in enumerate(labels) if value == label]
        if len(hits) != 1:
            raise ValueError(f"plan has {len(hits)} waypoints labelled {label!r}; expected 1")
        return hits[0]

    end = len(labels)-1 if labels and labels[-1] == "RETURN_HOME" else len(labels)
    events = [dict(kind="calibration", op=perception["calibration_op"],
                   index=index_of(f"SURVEY:{perception['calibration_op']}"))]
    for row in perception["surveys"]:
        start = next(i for i, value in enumerate(labels) if value.split(":", 1)[-1] == row["next_cut_op"]
                     and ":" in value)
        shifted = [i for i in range(start, end) if labels[i].split(":", 1)[0] not in SURVEY_PHASES]
        events.append(dict(kind="survey", op=row["op"], index=index_of(f"SURVEY:{row['op']}"),
                           next_cut_op=row["next_cut_op"], shift_from=start, shifted=shifted))
    order = [e["index"] for e in events]
    if order != sorted(order) or any(e["kind"] == "survey" and not e["index"] < e["shift_from"] for e in events):
        raise ValueError("survey waypoints must precede the cuts they relocate")
    return events


@dataclass
class RelocatePolicy(WaypointPolicy):
    """WaypointPolicy (delta law) plus the ADR 0016 wrist-depth survey and relocation."""
    depth_source: Callable | None = None
    perception: dict = field(default_factory=dict)
    geometry: object = None
    survey_config: object = None
    camera_id: str = CAMERA_ID
    events: list = field(default_factory=list)
    vision: dict = field(default_factory=dict)
    last_vision: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.command_mode != "delta":
            raise ValueError(f"{HOOK} is declared for the delta law (path_v2) only")
        if self.perception.get("hook") != HOOK:
            raise ValueError(f"RelocatePolicy needs a plan report perception block of {HOOK}")
        from .wrist_depth_survey import SurveyConfig
        if self.survey_config is None:
            self.survey_config = SurveyConfig()
        self.waypoints = list(self.waypoints)
        self._planned = [w.pose for w in self.waypoints]
        self.events = events_from_plan(self.waypoints, self.perception)
        self.side = int(self.perception["side"])
        R = np.asarray(self.geometry.T, float)[:3, :3]
        self.axis_world = R@np.array([1., 0., 0.])
        self.vision = dict(adr=ADR, hook=HOOK, camera_id=self.camera_id, side=self.side,
                           survey_pose_local_m=self.perception["survey_pose_local_m"],
                           events=[{k: v for k, v in e.items() if k != "shifted"} for e in self.events],
                           calibration=None, surveys=[], applied_shift_m=0., applied_shift_world_m=[0., 0., 0.],
                           shifted_labels=[], inputs=INPUTS)
        self._done = set()

    def _measure(self, observation, event):
        stamp = float(observation["timestamp_s"])
        record = dict(kind=event["kind"], op=event["op"], waypoint=event["index"],
                      label=self.waypoints[event["index"]].label, observation_timestamp_s=stamp,
                      frame_timestamp_s=None, valid=False, reason=None)
        if self.depth_source is None:
            return dict(record, reason="no depth source")
        try:
            frame = self.depth_source(self.camera_id)
        except Exception as error:  # an unavailable camera invalidates the survey, never the episode
            return dict(record, reason=f"depth source failed: {error!r}")
        record["frame_timestamp_s"] = float(frame["physics_timestamp_s"])
        if not abs(record["frame_timestamp_s"]-stamp) <= 1e-9:
            return dict(record, reason="stale depth frame (timestamp differs from the observation)")
        edge = edge_from_observation(observation, self.task)  # nominal, from joint-state proprioception
        measurement = measure_end_face(frame, self.geometry, self.side, self.survey_config, edge_pose_world=edge)
        face = measurement["end_face"]
        return dict(record, valid=bool(face.get("valid")), reason=face.get("reason"), measurement=measurement)

    def _apply(self, event, shift):
        vector = shift*self.axis_world
        for k in event["shifted"]:
            w = self.waypoints[k]
            self.waypoints[k] = replace(w, pose=tuple(np.r_[np.asarray(self._planned[k][:3], float)+vector,
                                                            self._planned[k][3:]]))
        self.vision.update(applied_shift_m=float(shift), applied_shift_world_m=vector.tolist(),
                           shifted_labels=[self.waypoints[k].label for k in event["shifted"]])

    def __call__(self, observation):
        self.last_vision = None
        before = self.index
        action = super().__call__(observation)
        for n, event in enumerate(self.events):
            k = event["index"]
            if n in self._done or not before <= k < self.index:
                continue
            self._done.add(n)
            record = self._measure(observation, event)
            out = dict(kind=event["kind"], op=event["op"], record=record)
            if event["kind"] == "calibration":
                self.vision["calibration"] = record
            else:
                calibration = self.vision["calibration"]
                reasons = [f"{name}: {'not measured' if rec is None else rec['reason']}"
                           for name, rec in (("calibration", calibration), ("survey", record))
                           if rec is None or not rec["valid"]]
                shift, estimate = relative_shift((calibration or {}).get("measurement"), record.get("measurement"))
                if reasons:
                    shift = None
                    estimate.update(valid=False, reasons=reasons, applied_shift_m=0.)
                if shift is not None:
                    self._apply(event, shift)
                self.vision["surveys"].append(dict(op=event["op"], next_cut_op=event["next_cut_op"], record=record,
                                                   estimate=estimate))
                out.update(estimate=estimate, applied_shift_m=self.vision["applied_shift_m"],
                           applied_shift_world_m=self.vision["applied_shift_world_m"])
                # Restart from the waypoint after the survey with the (possibly shifted) targets.
                self.index, self._start_time, self._hold_start = k+1, None, None
                action = super().__call__(observation)
            self.last_vision = out
        return action


def relocate_policy(task, waypoints, perception, *, depth_source=None, limits=None, **policy_options):
    """RelocatePolicy over a goal-planner plan whose report carries a ``perception`` block."""
    from .control import ActionLimits
    return RelocatePolicy(task, list(waypoints), limits or ActionLimits(), **policy_options, depth_source=depth_source,
                          perception=dict(perception), geometry=scene_bar_geometry(task.scene))
