"""``blade_master_flat_board_v1`` workcell: one wooden board lying on the table (WP-12 Phase 2, ADR 0017).

One record (``configs/wp12_workcells_v1.json``, optionally deep-merged with
per-scene ``overrides``) is the single source for

* the table box (visual + PhysX collider; top = Franka mounting plane),
* the cutting board box(es) (visual mesh = PhysX collider = TLMPM fixture
  contact geometry, all built here from the same numbers),
* the workpiece support plane (board top = table top + board thickness =
  workpiece-local z = 0),
* the HOME and the action workspace (z minimum derived from the board),
* the collision preflight (the C2 checker reads ``table_box_world`` and the
  task fixture boxes).

There are no risers or pedestals: the board bottom rests on the table top and
the workpiece bottom on the board top; ``check_scene`` refuses any scene whose
declared workpiece pose would float above or sink into the board. Everything
here is pure (numpy only) and additive; ``wp11_workcell_v1`` and the v1-v3
scenarios are untouched.

Scene usage (``rcs_scene_v1``)::

    "supports": {"kind": "flat_board",
                 "workcell": {"profile": "blade_master_flat_board_v1",
                              "overrides": {"cutting_board": {"thickness_m": 0.03}}}}

``overrides`` (optional) may change ``table`` and ``cutting_board`` fields
only; the workpiece ``world_pose.translation_m[2]`` must then equal the new
board top (``board_top_world_z``), which ``place_on_board`` fills in.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT/"configs/wp12_workcells_v1.json"
FIXTURE_MODE = "flat_board_v1"
BOARD_KINDS = ("solid", "slotted")
_TABLE_KEYS = ("top_world_z_m", "center_xy_world_m", "half_extents_xy_m", "thickness_m", "friction", "restitution",
               "display_color_rgb", "scope")
_BOARD_KEYS = ("kind", "center_xy_world_m", "yaw_deg", "length_x_m", "width_y_m", "thickness_m", "slots", "slot_depth_m",
               "friction", "restitution", "display_color_rgb", "appearance", "min_workpiece_margin_m", "scope")
_TOP_KEYS = ("fixture_mode", "decision", "description", "table", "cutting_board", "friction_scope", "home_joint_positions",
             "home_finger_position_m", "home_hand_pose_world_xyzw", "home_source", "preflight", "action_workspace_local_m")


class FlatBoardError(ValueError):
    """A flat-board workcell record or override is malformed."""


def profiles():
    """{name: record} of the versioned flat-board workcells."""
    return json.loads(PATH.read_text())["workcells"]


def _merge(base, override, where):
    out = deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise FlatBoardError(f"{where}: unknown override key {key!r}")
        out[key] = _merge(base[key], value, f"{where}.{key}") if isinstance(base[key], dict) and isinstance(value, dict) else deepcopy(value)
    return out


def _num(value, where, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FlatBoardError(f"{where} must be a finite number")
    if positive and not value > 0:
        raise FlatBoardError(f"{where} must be positive")
    if nonnegative and value < 0:
        raise FlatBoardError(f"{where} must be nonnegative")
    return float(value)


def _vec(value, where, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise FlatBoardError(f"{where} must be a list of {size} numbers")
    return [_num(v, f"{where}[{i}]") for i, v in enumerate(value)]


def _friction(value, where):
    if value == "task_material":
        return value
    return _num(value, where, nonnegative=True)


def _color(value, where):
    rgb = _vec(value, where, 3)
    if min(rgb) < 0 or max(rgb) > 1:
        raise FlatBoardError(f"{where} must be an RGB colour in [0, 1]")
    return rgb


def validate_spec(spec):
    """Normalize ``supports.workcell`` of a flat-board scene: {"profile", "overrides"} (overrides may be {})."""
    if not isinstance(spec, dict) or set(spec)-{"profile", "overrides"} or "profile" not in spec:
        raise FlatBoardError("A flat-board workcell is {'profile': <name>, 'overrides': {...} (optional)}")
    if spec["profile"] not in profiles():
        raise FlatBoardError(f"Unknown flat-board workcell profile {spec['profile']!r}; known {sorted(profiles())}")
    overrides = spec.get("overrides", {}) or {}
    if not isinstance(overrides, dict) or set(overrides)-{"table", "cutting_board"}:
        raise FlatBoardError("workcell.overrides may only change 'table' and 'cutting_board'")
    out = {"profile": spec["profile"], "overrides": deepcopy(overrides)}
    resolve(out)   # validates the merged record
    return out


def resolve(spec):
    """The merged and validated record of a flat-board workcell spec ({"profile", "overrides"})."""
    record = profiles()[spec["profile"]]
    record = _merge(record, spec.get("overrides") or {}, "workcell.overrides")
    if set(record)-set(_TOP_KEYS) or record.get("fixture_mode") != FIXTURE_MODE:
        raise FlatBoardError(f"Workcell record must declare fixture_mode {FIXTURE_MODE!r} and known keys only")
    table, board = record["table"], record["cutting_board"]
    if set(table)-set(_TABLE_KEYS) or set(board)-set(_BOARD_KEYS):
        raise FlatBoardError("Unknown table/cutting_board keys")
    t = {"top_world_z_m": _num(table["top_world_z_m"], "table.top_world_z_m"),
         "center_xy_world_m": _vec(table["center_xy_world_m"], "table.center_xy_world_m", 2),
         "half_extents_xy_m": _vec(table["half_extents_xy_m"], "table.half_extents_xy_m", 2),
         "thickness_m": _num(table["thickness_m"], "table.thickness_m", positive=True),
         "friction": _friction(table["friction"], "table.friction"),
         "restitution": _num(table["restitution"], "table.restitution", nonnegative=True),
         "display_color_rgb": _color(table["display_color_rgb"], "table.display_color_rgb")}
    if min(t["half_extents_xy_m"]) <= 0:
        raise FlatBoardError("table.half_extents_xy_m must be positive")
    if board["kind"] not in BOARD_KINDS:
        raise FlatBoardError(f"cutting_board.kind must be one of {BOARD_KINDS}")
    b = {"kind": board["kind"], "center_xy_world_m": _vec(board["center_xy_world_m"], "cutting_board.center_xy_world_m", 2),
         "yaw_deg": _num(board["yaw_deg"], "cutting_board.yaw_deg"),
         **{key: _num(board[key], f"cutting_board.{key}", positive=True) for key in ("length_x_m", "width_y_m", "thickness_m")},
         "slot_depth_m": _num(board["slot_depth_m"], "cutting_board.slot_depth_m", nonnegative=True),
         "friction": _friction(board["friction"], "cutting_board.friction"),
         "restitution": _num(board["restitution"], "cutting_board.restitution", nonnegative=True),
         "display_color_rgb": _color(board["display_color_rgb"], "cutting_board.display_color_rgb"),
         "appearance": str(board.get("appearance", "")),
         "min_workpiece_margin_m": _num(board["min_workpiece_margin_m"], "cutting_board.min_workpiece_margin_m", nonnegative=True)}
    if not isinstance(board["slots"], list):
        raise FlatBoardError("cutting_board.slots must be a list")
    slots = []
    for i, row in enumerate(board["slots"]):
        if not isinstance(row, dict) or set(row) != {"x_board_m", "width_m"}:
            raise FlatBoardError(f"cutting_board.slots[{i}] must be {{'x_board_m', 'width_m'}}")
        slots.append({"x_board_m": _num(row["x_board_m"], f"cutting_board.slots[{i}].x_board_m"),
                      "width_m": _num(row["width_m"], f"cutting_board.slots[{i}].width_m", positive=True)})
    slots.sort(key=lambda row: row["x_board_m"])
    b["slots"] = slots
    if b["kind"] == "solid" and (slots or b["slot_depth_m"] != 0):
        raise FlatBoardError("A solid board has no slots and slot_depth_m 0; use kind 'slotted' explicitly")
    if b["kind"] == "slotted":
        if not slots:
            raise FlatBoardError("A slotted board declares at least one slot")
        if not 0 < b["slot_depth_m"] < b["thickness_m"]:
            raise FlatBoardError("slot_depth_m must be in (0, thickness_m): the board stays one continuous body")
        edges = [-b["length_x_m"]/2]
        for row in slots:
            lo, hi = row["x_board_m"]-row["width_m"]/2, row["x_board_m"]+row["width_m"]/2
            if not edges[-1] < lo or not hi < b["length_x_m"]/2:
                raise FlatBoardError("Slots must be ordered, disjoint and strictly inside the board length")
            edges += [lo, hi]
    preflight = record.get("preflight")
    if not isinstance(preflight, dict) or "blade_board_contact_min_distance_m" not in preflight:
        raise FlatBoardError("The workcell record must declare preflight.blade_board_contact_min_distance_m")
    if _num(preflight["blade_board_contact_min_distance_m"], "preflight.blade_board_contact_min_distance_m") < 0:
        raise FlatBoardError("preflight.blade_board_contact_min_distance_m must be >= 0: the knife never enters the rigid board")
    out = deepcopy(record)
    out["table"], out["cutting_board"] = {**table, **t}, {**board, **b}
    out["profile"] = spec["profile"]
    # The board must rest fully on the table top (no overhang that would need hidden support).
    corners = board_corners_xy_world(out)
    tc, th = np.asarray(t["center_xy_world_m"]), np.asarray(t["half_extents_xy_m"])
    if np.any(np.abs(corners-tc) > th+1e-12):
        raise FlatBoardError("The cutting board footprint must lie inside the table top")
    return out


# ---------------------------------------------------------------- derived geometry (one source)

def board_top_world_z(record):
    """World z of the board top = table top + board thickness (the workpiece support plane)."""
    return float(record["table"]["top_world_z_m"]+record["cutting_board"]["thickness_m"])


def _board_rotation(record):
    yaw = math.radians(record["cutting_board"]["yaw_deg"])
    return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


def board_corners_xy_world(record):
    board = record["cutting_board"]
    half = np.array([board["length_x_m"]/2, board["width_y_m"]/2])
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    return np.asarray(board["center_xy_world_m"])+(signs*half)@_board_rotation(record).T


def board_boxes_board_frame(record):
    """[(name, center, half)] of the board in the BOARD frame (origin board centre, z = 0 at the table top).

    Solid: one box. Slotted: a continuous base slab under the grooves plus one
    top segment per groove-free interval (all role 'board').
    """
    board = record["cutting_board"]
    L, W, T = board["length_x_m"], board["width_y_m"], board["thickness_m"]
    if board["kind"] == "solid":
        return [("flat_board", (0., 0., T/2), (L/2, W/2, T/2))]
    depth = board["slot_depth_m"]
    boxes = [("flat_board_base", (0., 0., (T-depth)/2), (L/2, W/2, (T-depth)/2))]
    edges = [-L/2]
    for row in board["slots"]:
        edges += [row["x_board_m"]-row["width_m"]/2, row["x_board_m"]+row["width_m"]/2]
    edges.append(L/2)
    for i in range(0, len(edges), 2):
        lo, hi = round(edges[i], 12), round(edges[i+1], 12)
        boxes.append((f"flat_board_segment_{i//2}", (round((lo+hi)/2, 12), 0., T-depth/2), (round((hi-lo)/2, 12), W/2, depth/2)))
    return boxes


def board_boxes_world(record):
    """[(name, center_world, half, yaw_deg)] board boxes in WORLD (z from the table top)."""
    board = record["cutting_board"]
    R = _board_rotation(record)
    base = np.r_[board["center_xy_world_m"], record["table"]["top_world_z_m"]]
    out = []
    for name, center, half in board_boxes_board_frame(record):
        xy = R@np.asarray(center[:2])
        out.append((name, tuple(map(float, base+np.r_[xy, center[2]])), tuple(map(float, half)), float(board["yaw_deg"])))
    return out


def table_box_world(record):
    """(center_world, half_extents) of the table box; its top is ``table.top_world_z_m``."""
    table = record["table"]
    th = table["thickness_m"]
    return (tuple(map(float, (*table["center_xy_world_m"], table["top_world_z_m"]-th/2))),
            tuple(map(float, (*table["half_extents_xy_m"], th/2))))


def _local_from_world(world_pose_xyz_yaw):
    x, y, z, yaw_deg = world_pose_xyz_yaw
    yaw = math.radians(yaw_deg)
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0.], [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
    return R, np.array([x, y, z], float)


def fixture_boxes_local(record, world_pose_xyz_yaw, material_friction):
    """The board as workpiece-local ``MaterialFixtureBox`` es (TLMPM fixture = PhysX collider = visual box).

    ``world_pose_xyz_yaw`` is the workpiece (x, y, z, yaw_deg); the board yaw
    must equal the workpiece yaw so the local boxes stay axis aligned (an
    oriented TLMPM fixture box does not exist; the case raises).
    """
    from .tasks import MaterialFixtureBox
    board = record["cutting_board"]
    if abs(((world_pose_xyz_yaw[3]-board["yaw_deg"]+180.) % 360.)-180.) > 1e-9:
        raise FlatBoardError("flat_board_v1 needs the workpiece yaw equal to the board yaw (no oriented TLMPM fixture boxes)")
    R, t = _local_from_world(world_pose_xyz_yaw)
    friction = float(material_friction) if board["friction"] == "task_material" else float(board["friction"])
    boxes = []
    for name, center, half, _ in board_boxes_world(record):
        local = R.T@(np.asarray(center)-t)
        boxes.append(MaterialFixtureBox(name, tuple(round(float(v), 12) for v in local), tuple(map(float, half)), role="board",
                                        support_name="cutting_board", display_color_rgb=tuple(board["display_color_rgb"]),
                                        static_friction=friction, dynamic_friction=friction,
                                        restitution=float(board["restitution"])))
    return boxes


def table_contact_material(record, material_friction):
    table = record["table"]
    friction = float(material_friction) if table["friction"] == "task_material" else float(table["friction"])
    return dict(static_friction=friction, dynamic_friction=friction, restitution=float(table["restitution"]))


def check_placement(record, world_pose_xyz_yaw, footprint_local_xy):
    """List of (reason, detail) why a workpiece cannot rest on this board (empty = feasible).

    ``footprint_local_xy``: (N, 2) local outline points of the material
    (base box corners and protrusion bounding rectangles).
    """
    problems = []
    top = board_top_world_z(record)
    if abs(world_pose_xyz_yaw[2]-top) > 1e-9:
        problems.append(("workpiece_not_on_board", f"workpiece bottom z {world_pose_xyz_yaw[2]:.6f} m != board top "
                                                   f"{top:.6f} m (table top + board thickness)"))
    board = record["cutting_board"]
    if abs(((world_pose_xyz_yaw[3]-board["yaw_deg"]+180.) % 360.)-180.) > 1e-9:
        problems.append(("workpiece_yaw_differs_from_board", f"workpiece yaw {world_pose_xyz_yaw[3]} deg, board {board['yaw_deg']} deg"))
    R, t = _local_from_world(world_pose_xyz_yaw)
    world_xy = np.asarray(footprint_local_xy, float)@R[:2, :2].T+t[:2]
    in_board = (world_xy-np.asarray(board["center_xy_world_m"]))@_board_rotation(record)
    margin = board["min_workpiece_margin_m"]
    limit = np.array([board["length_x_m"]/2, board["width_y_m"]/2])-margin
    if np.any(np.abs(in_board) > limit+1e-12):
        problems.append(("workpiece_off_board", f"material outline leaves the board top minus {margin*1e3:.1f} mm margin"))
    return problems


def place_on_board(scene):
    """A copy of an rcs_scene_v1 dict with the workpiece z set to the board top of its flat-board workcell."""
    scene = deepcopy(scene)
    spec = scene["supports"]["workcell"]
    record = resolve(validate_spec(spec))
    pose = scene["workpiece"].setdefault("world_pose", {})
    translation = list(pose.get("translation_m", [0., 0., 0.]))
    translation[2] = board_top_world_z(record)
    pose["translation_m"] = translation
    return scene


def home(record):
    """HOME of the workcell (as task_scenarios.scenario_home), or None when the record declares none."""
    if record.get("home_joint_positions") is None:
        return None
    joints = np.asarray(record["home_joint_positions"], float)
    hand = np.asarray(record["home_hand_pose_world_xyzw"], float)
    if joints.shape != (7,) or hand.shape != (7,) or not np.isfinite(np.r_[joints, hand]).all():
        raise FlatBoardError(f"Workcell {record['profile']} declares a malformed HOME")
    return dict(scenario=record["profile"], workcell_profile=record["profile"], joint_positions=tuple(map(float, joints)),
                finger_position_m=float(record.get("home_finger_position_m", float("nan"))),
                hand_pose_world_xyzw=tuple(map(float, hand)), source=record.get("home_source"))


def action_workspace(record):
    """(min, max) local edge-centre workspace; z min = board top (0) or the groove floor of a slotted board."""
    box = record["action_workspace_local_m"]
    board = record["cutting_board"]
    zmin = -board["slot_depth_m"] if board["kind"] == "slotted" else 0.
    return (float(box["min_xy"][0]), float(box["min_xy"][1]), float(zmin)), tuple(map(float, box["max"]))


def geometry_receipt(record, world_pose_xyz_yaw, material_friction):
    """Compact JSON-able single-source receipt (table, board world/local boxes, planes, workspace)."""
    return {"profile": record["profile"], "fixture_mode": FIXTURE_MODE,
            "table_box_world": {"center_m": list(table_box_world(record)[0]), "half_extents_m": list(table_box_world(record)[1])},
            "table_top_world_z_m": float(record["table"]["top_world_z_m"]),
            "board_top_world_z_m": board_top_world_z(record), "board_kind": record["cutting_board"]["kind"],
            "board_boxes_world": [{"name": n, "center_m": list(c), "half_extents_m": list(h), "yaw_deg": y}
                                  for n, c, h, y in board_boxes_world(record)],
            "board_boxes_local": [{"name": b.name, "center_m": list(b.center_m), "half_extents_m": list(b.half_extents_m)}
                                  for b in fixture_boxes_local(record, world_pose_xyz_yaw, material_friction)],
            "action_workspace_local_m": [list(v) for v in action_workspace(record)],
            "risers": 0, "pedestals": 0}
