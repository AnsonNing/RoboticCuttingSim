"""Configurable scenes (``rcs_scene_v1``) and compiled configured tasks (ADR 0015, WP-11 Phase S).

A scene says WHAT EXISTS; a goal (``goals.py``, ``rcs_goal_v1``) says what should
be achieved; ``compile_task(scene, goal)`` joins them into a ``ConfiguredTaskSpec``
(a ``tasks.ProfileTaskSpec`` subclass), so the existing runtimes, recorders and
receipts consume it unchanged. Everything here is additive and opt-in: v1-v4
tasks, scenarios, plans and thresholds keep their exact behaviour.

Schema ``rcs_scene_v1`` (JSON under ``configs/wp11/scenes/``; SI units, world Z up,
quaternions XYZW; unknown keys are errors)::

    {
      "schema": "rcs_scene_v1",
      "scene_id": "bar_three_bumps",            # [A-Za-z0-9_.-]+
      "description": "...",                     # optional
      "workpiece": {
        "base_box_m": {"length_m": .16, "width_m": .03, "height_m": .02},
        "protrusions": [                        # optional; ONE continuous heightfield,
          {"id": "p0", "family": "dome",        # never separate bodies (max-union over the base)
           "x_m": -.05, "y_m": 0., "radius_x_m": .012, "radius_y_m": .012, "height_m": .008}],
        "world_pose": {"translation_m": [0, 0, 0], "yaw_deg": 0.},   # yaw about world z only
        "material_profile": "cutting_tasks_v1"  # v1 provisional material, or a food_profiles_v1 id
      },
      "supports": {                             # local support footprint, top at workpiece z = 0
        "kind": "solid_board" | "slotted_board" | "boxes" | "flat_board",   # flat_board: see below
        "x_range_m": [-.09, .09], "half_width_y_m": .021, "thickness_m": .01,   # solid/slotted
        "slots": [{"x_m": 0., "width_m": .04}],                               # slotted only
        "names": ["left_pedestal", "right_pedestal"],                         # optional, per segment
        "boxes": [{"name": ..., "center_m": [3], "half_extents_m": [3]}],     # kind "boxes" only
        "workcell": null | {"from_scenario": "blademaster_workcell_v1"}
                         | {"cutting_board": {...}, "riser": {...}, "table": {...}}
      },
      "tool": {"profile_id": "ruler_knife_v1", "grasp_preset": "default"},
      "solver": {"module": "tlmpm_v4" | "tlmpm_v5", "cutting_law": 8, "max_cuts": 2,
                 "particle_spacing_m": .001, "grid_dx_m": .004,
                 "slots_per_node": 16},                 # optional, tlmpm_v5 only (absent: solver default 8)
      "table_top_world_z_m": -.01,              # optional (v1 common default)
      "randomization": null | {"seed": 7, "ranges": [
          {"path": "workpiece.world_pose.yaw_deg", "low": -15, "high": 15},
          {"path": "workpiece.protrusions.*.x_m", "low": -.003, "high": .003, "mode": "offset"}]}
    }

``workcell`` compiles every support segment into a wood board plus a grey riser to
the table with exactly the ``wp10_scenarios_v3`` ``cutting_board_v1`` code
(``task_scenarios._cutting_board_boxes``) and materials (``MaterialFixtureBox``).
Without a workcell the raw support boxes are used (v2-task semantics). Slotted
segment coordinates are rounded to 1e-12 m so that equal configs give bitwise
equal boxes.

WP-12 (ADR 0017): ``"supports": {"kind": "flat_board", "workcell": {"profile":
"blade_master_flat_board_v1", "overrides": {...}}}`` takes the table and ONE
continuous cutting board lying on it from ``configs/wp12_workcells_v1.json``
(``flat_board_workcell``): no riser, the board box(es) are the TLMPM fixture,
PhysX collider and visual box, the workpiece z must equal the board top and the
table top comes from the record. HOME and action workspace come from the record.

Randomization draws ``uniform(low, high)`` with ``numpy.random.default_rng(seed)``
in declared order (a ``*`` path expands over list items in order); ``mode``
``"absolute"`` (default) replaces the value and ``"offset"`` adds to it. The
resolved scene records every sampled value under ``randomization.sampled`` and is
validated again; an infeasible draw raises ``SceneInfeasible`` with a reason.

Workpiece yaw: the configured task carries ``world_from_workpiece_xyzw`` with the
yaw. Particles, fixture boxes, TLMPM gravity and the table plane are all handled
in the workpiece-local solver frame by ``LiveCuttingMaterial`` (yaw keeps local z
= world z), the scripted planner works in local coordinates and transforms to
world, and evaluators read world recordings through the task pose. Paths that
still assume identity orientation raise explicitly: ``task_scenarios.apply_scenario``
(any configured task is rejected there) and the Isaac Sim / Isaac Lab runtimes,
which build tasks from registered ids only (``open_cutting_env(task=...)`` raises
NotImplementedError for them).

WP-11 Phase D: Isaac Sim accepts a compiled task (``open_cutting_env(runtime=
"isaac_sim", task=compiled)``); the runner recompiles it from its recorded
scene/goal/planner config and requires the same id. Isaac Lab still raises.
A workcell ``from_scenario`` that declares ``home_joint_positions``
(``wp11_workcell_v1``) also sets the planner HOME (FK(q_home) composed with the
tool's hand_from_edge) and the action workspace; scenes without one are unchanged.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import ClassVar

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .benchmark import FAMILIES, Bump, HeightFieldSpec, MaterialParameters
from .tasks import ACTION_MODES, FixtureBox, ProfileTaskSpec, _protocol

SCENE_SCHEMA = "rcs_scene_v1"
CONFIGURED_TASK_PROTOCOL = "rcs_configured_task_v1"
ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT/"configs/wp11/scenes"
GOAL_DIR = ROOT/"configs/wp11/goals"
V1_MATERIAL_PROFILE = "cutting_tasks_v1"
SUPPORT_KINDS = ("solid_board", "slotted_board", "boxes", "flat_board")   # flat_board: WP-12 (ADR 0017)
SOLVERS = {"tlmpm_v4": {"laws": (4, 5, 6, 7, 8), "max_cuts": 2},
           "tlmpm_v5": {"laws": (4, 5, 8), "max_cuts": 15}}
_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class SceneConfigError(ValueError):
    """A scene or goal config does not satisfy its versioned schema."""


class SceneInfeasible(SceneConfigError):
    """A (randomized) scene is geometrically infeasible; ``reason`` names why."""

    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# ---------------------------------------------------------------- small validators

def _keys(mapping, where, required=(), optional=()):
    if not isinstance(mapping, dict):
        raise SceneConfigError(f"{where} must be a JSON object")
    unknown = set(mapping)-set(required)-set(optional)
    missing = [key for key in required if key not in mapping]
    if unknown:
        raise SceneConfigError(f"{where}: unknown keys {sorted(unknown)}")
    if missing:
        raise SceneConfigError(f"{where}: missing keys {missing}")


def _number(value, where, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise SceneConfigError(f"{where} must be a finite number")
    if positive and not value > 0:
        raise SceneConfigError(f"{where} must be positive")
    if nonnegative and value < 0:
        raise SceneConfigError(f"{where} must be nonnegative")
    return float(value)


def _vector(value, where, size=3):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise SceneConfigError(f"{where} must be a list of {size} numbers")
    return [_number(v, f"{where}[{i}]") for i, v in enumerate(value)]


def _integer(value, where, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SceneConfigError(f"{where} must be an integer")
    if (low is not None and value < low) or (high is not None and value > high):
        raise SceneConfigError(f"{where} must be in [{low}, {high}]")
    return int(value)


def _identifier(value, where):
    if not isinstance(value, str) or not _ID.match(value):
        raise SceneConfigError(f"{where} must match [A-Za-z0-9_.-]+")
    return value


def load_json(source, directory=SCENE_DIR):
    """A dict (deep-copied), or a JSON file path (absolute, relative, or a bare name in ``directory``)."""
    if isinstance(source, dict):
        return deepcopy(source)
    path = Path(source)
    for candidate in (path, directory/path, directory/f"{path}.json"):
        if candidate.exists():
            path = candidate
            break
    if not path.exists():
        raise SceneConfigError(f"Config file not found: {source}")
    return json.loads(path.read_text())


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


# ---------------------------------------------------------------- scene schema

def validate_scene(source):
    """Validate an ``rcs_scene_v1`` document; returns a normalized copy with defaults filled."""
    doc = load_json(source, SCENE_DIR)
    _keys(doc, "scene", ("schema", "scene_id", "workpiece", "supports", "tool", "solver"),
          ("description", "table_top_world_z_m", "randomization"))
    if doc["schema"] != SCENE_SCHEMA:
        raise SceneConfigError(f"scene.schema must be {SCENE_SCHEMA!r}")
    out = {"schema": SCENE_SCHEMA, "scene_id": _identifier(doc["scene_id"], "scene.scene_id"),
           "description": str(doc.get("description", ""))}
    workpiece = doc["workpiece"]
    _keys(workpiece, "workpiece", ("base_box_m",), ("protrusions", "world_pose", "material_profile", "geometry_provider"))
    base = workpiece["base_box_m"]
    _keys(base, "workpiece.base_box_m", ("length_m", "width_m", "height_m"))
    base = {key: _number(base[key], f"workpiece.base_box_m.{key}", positive=True) for key in ("length_m", "width_m", "height_m")}
    # WP-12 Phase 4c (opt-in): workpiece.geometry_provider; absent (= heightfield_v1) keeps every scene unchanged.
    from . import geometry_providers as gp
    try:
        provider = gp.validate_provider(workpiece.get("geometry_provider"), base)
    except gp.GeometryProviderError as error:
        raise SceneConfigError(str(error)) from None
    provider_id = gp.DEFAULT_PROVIDER if provider is None else provider["id"]
    families = gp.protrusion_families(provider_id)
    protrusions, ids = [], set()
    for i, row in enumerate(workpiece.get("protrusions", [])):
        where = f"workpiece.protrusions[{i}]"
        _keys(row, where, ("id", "family", "x_m", "y_m", "radius_x_m", "radius_y_m", "height_m"),
              gp.protrusion_extra_keys(provider_id))
        pid = _identifier(row["id"], f"{where}.id")
        if pid in ids or pid in ("all", "base"):
            raise SceneConfigError(f"{where}.id {pid!r} is duplicated or reserved")
        ids.add(pid)
        if row["family"] not in families:
            raise SceneConfigError(f"{where}.family must be one of {families}")
        try:
            extras = gp.normalize_protrusion_extras(provider_id, row, where)
        except gp.GeometryProviderError as error:
            raise SceneConfigError(str(error)) from None
        protrusions.append({"id": pid, "family": row["family"],
                            **{key: _number(row[key], f"{where}.{key}") for key in ("x_m", "y_m")},
                            **{key: _number(row[key], f"{where}.{key}", positive=True)
                               for key in ("radius_x_m", "radius_y_m", "height_m")}, **extras})
    if provider_id == gp.SDF_CSG and protrusions:
        raise SceneConfigError("workpiece.geometry_provider sdf_csg_experimental_v1 takes no protrusions")
    pose = workpiece.get("world_pose", {})
    _keys(pose, "workpiece.world_pose", (), ("translation_m", "yaw_deg"))
    world_pose = {"translation_m": _vector(pose.get("translation_m", [0., 0., 0.]), "workpiece.world_pose.translation_m"),
                  "yaw_deg": _number(pose.get("yaw_deg", 0.), "workpiece.world_pose.yaw_deg")}
    material = workpiece.get("material_profile", V1_MATERIAL_PROFILE)
    if material != V1_MATERIAL_PROFILE:
        from .food_materials import profiles
        from .materials_wp12 import is_profile
        if material not in profiles() and not is_profile(material):
            raise SceneConfigError(f"workpiece.material_profile must be {V1_MATERIAL_PROFILE!r}, a food_profiles_v1 id "
                                   "or a materials_wp12_v1 id")
    out["workpiece"] = {"base_box_m": base, "protrusions": protrusions, "world_pose": world_pose,
                        "material_profile": material}
    if provider is not None:   # only opt-in scenes carry the key (default scene_json and task ids unchanged)
        out["workpiece"]["geometry_provider"] = provider
    out["supports"] = _validate_supports(doc["supports"])
    tool = doc["tool"]
    _keys(tool, "tool", ("profile_id",), ("grasp_preset",))
    from .tool_profile import PROFILES
    if tool["profile_id"] not in PROFILES:
        raise SceneConfigError(f"tool.profile_id must be one of {sorted(PROFILES)}")
    out["tool"] = {"profile_id": tool["profile_id"], "grasp_preset": str(tool.get("grasp_preset", "default"))}
    solver = doc["solver"]
    _keys(solver, "solver", ("module", "cutting_law", "max_cuts"), ("particle_spacing_m", "grid_dx_m", "slots_per_node"))
    if solver["module"] not in SOLVERS:
        raise SceneConfigError(f"solver.module must be one of {sorted(SOLVERS)}")
    limits = SOLVERS[solver["module"]]
    law = _integer(solver["cutting_law"], "solver.cutting_law")
    if law not in limits["laws"]:
        raise SceneConfigError(f"solver.cutting_law {law} is not available in {solver['module']} (laws {limits['laws']})")
    out["solver"] = {"module": solver["module"], "cutting_law": law,
                     "max_cuts": _integer(solver["max_cuts"], "solver.max_cuts", 1, limits["max_cuts"]),
                     "particle_spacing_m": _number(solver.get("particle_spacing_m", .001), "solver.particle_spacing_m", positive=True),
                     "grid_dx_m": _number(solver.get("grid_dx_m", .004), "solver.grid_dx_m", positive=True)}
    if "slots_per_node" in solver:  # opt-in (WP-11 Phase E round 2): kept only when declared, so other ids are unchanged
        if solver["module"] != "tlmpm_v5":
            raise SceneConfigError("solver.slots_per_node applies to tlmpm_v5 only")
        out["solver"]["slots_per_node"] = _integer(solver["slots_per_node"], "solver.slots_per_node", 1, 64)
    out["table_top_world_z_m"] = _number(doc.get("table_top_world_z_m", _protocol()["common"]["table_top_world_z_m"]),
                                         "table_top_world_z_m")
    if out["supports"]["kind"] == "flat_board":   # the workcell record is the single source of the table top
        top = flat_board_record(out)["table"]["top_world_z_m"]
        if "table_top_world_z_m" in doc and abs(out["table_top_world_z_m"]-top) > 1e-12:
            raise SceneConfigError(f"table_top_world_z_m {out['table_top_world_z_m']} differs from the flat-board "
                                   f"workcell table top {top}; omit it (the workcell record is the single source)")
        out["table_top_world_z_m"] = float(top)
    out["randomization"] = _validate_randomization(doc.get("randomization"))
    _check_geometry(out)
    return out


def _validate_supports(supports):
    _keys(supports, "supports", ("kind",), ("x_range_m", "half_width_y_m", "thickness_m", "slots", "names", "boxes", "workcell"))
    kind = supports["kind"]
    if kind not in SUPPORT_KINDS:
        raise SceneConfigError(f"supports.kind must be one of {SUPPORT_KINDS}")
    out = {"kind": kind}
    if kind == "flat_board":
        # WP-12 blade_master_flat_board_v1 (ADR 0017): the board comes only from the workcell record.
        _keys(supports, "supports", ("kind", "workcell"))
        from .flat_board_workcell import FlatBoardError, validate_spec
        try:
            out["workcell"] = validate_spec(supports["workcell"])
        except FlatBoardError as error:
            raise SceneConfigError(f"supports.workcell: {error}") from None
        return out
    if kind == "boxes":
        if any(key in supports for key in ("x_range_m", "half_width_y_m", "thickness_m", "slots", "names")):
            raise SceneConfigError("supports.kind 'boxes' takes explicit boxes only")
        rows = supports.get("boxes")
        if not isinstance(rows, list) or not rows:
            raise SceneConfigError("supports.boxes must be a nonempty list")
        out["boxes"] = []
        for i, row in enumerate(rows):
            _keys(row, f"supports.boxes[{i}]", ("name", "center_m", "half_extents_m"))
            half = _vector(row["half_extents_m"], f"supports.boxes[{i}].half_extents_m")
            if min(half) <= 0:
                raise SceneConfigError(f"supports.boxes[{i}].half_extents_m must be positive")
            out["boxes"].append({"name": _identifier(row["name"], f"supports.boxes[{i}].name"),
                                 "center_m": _vector(row["center_m"], f"supports.boxes[{i}].center_m"), "half_extents_m": half})
    else:
        if "boxes" in supports:
            raise SceneConfigError("supports.boxes is only for kind 'boxes'")
        x_range = _vector(supports.get("x_range_m"), "supports.x_range_m", 2)
        if not x_range[0] < x_range[1]:
            raise SceneConfigError("supports.x_range_m must be increasing")
        out.update(x_range_m=x_range,
                   half_width_y_m=_number(supports.get("half_width_y_m"), "supports.half_width_y_m", positive=True),
                   thickness_m=_number(supports.get("thickness_m", .01), "supports.thickness_m", positive=True))
        slots = supports.get("slots", [])
        if kind == "solid_board" and slots:
            raise SceneConfigError("A solid board has no slots; use kind 'slotted_board'")
        if kind == "slotted_board" and not slots:
            raise SceneConfigError("A slotted board declares at least one slot")
        out["slots"] = []
        for i, row in enumerate(slots):
            _keys(row, f"supports.slots[{i}]", ("x_m", "width_m"))
            out["slots"].append({"x_m": _number(row["x_m"], f"supports.slots[{i}].x_m"),
                                 "width_m": _number(row["width_m"], f"supports.slots[{i}].width_m", positive=True)})
        out["slots"].sort(key=lambda row: row["x_m"])
        segments = support_segments(out)
        names = supports.get("names")
        if names is not None:
            if not isinstance(names, list) or len(names) != len(segments) or len(set(names)) != len(names):
                raise SceneConfigError(f"supports.names needs {len(segments)} distinct names")
            names = [_identifier(name, "supports.names[]") for name in names]
        out["names"] = names
    workcell = supports.get("workcell")
    if workcell is not None:
        if "from_scenario" in workcell:
            _keys(workcell, "supports.workcell", ("from_scenario",))
            from .task_scenarios import scenarios
            if workcell["from_scenario"] not in scenarios() or scenarios()[workcell["from_scenario"]].get("fixture_mode") != "cutting_board_v1":
                raise SceneConfigError("supports.workcell.from_scenario must name a cutting_board_v1 scenario")
        else:
            _keys(workcell, "supports.workcell", ("cutting_board", "riser"), ("table",))
        workcell = deepcopy(workcell)
    out["workcell"] = workcell
    return out


def support_segments(supports):
    """[(x_lo, x_hi)] board segments of a solid/slotted support (rounded to 1e-12 m)."""
    lo, hi = supports["x_range_m"]
    edges = [lo]
    for slot in supports["slots"]:
        a, b = slot["x_m"]-slot["width_m"]/2, slot["x_m"]+slot["width_m"]/2
        if not edges[-1] < a or not b < hi:
            raise SceneConfigError("Slots must be ordered, disjoint and strictly inside supports.x_range_m")
        edges += [a, b]
    edges.append(hi)
    return [(round(edges[i], 12), round(edges[i+1], 12)) for i in range(0, len(edges), 2)]


def _validate_randomization(randomization):
    if randomization is None:
        return None
    _keys(randomization, "randomization", ("seed", "ranges"), ("sampled", "rng"))
    seed = _integer(randomization["seed"], "randomization.seed", 0)
    if not isinstance(randomization["ranges"], list) or not randomization["ranges"]:
        raise SceneConfigError("randomization.ranges must be a nonempty list")
    ranges = []
    for i, row in enumerate(randomization["ranges"]):
        _keys(row, f"randomization.ranges[{i}]", ("path", "low", "high"), ("mode",))
        low, high = _number(row["low"], f"randomization.ranges[{i}].low"), _number(row["high"], f"randomization.ranges[{i}].high")
        if not low <= high or row.get("mode", "absolute") not in ("absolute", "offset"):
            raise SceneConfigError(f"randomization.ranges[{i}] needs low <= high and mode absolute|offset")
        if not isinstance(row["path"], str) or not row["path"].startswith(("workpiece.", "supports.")):
            raise SceneConfigError(f"randomization.ranges[{i}].path must start with workpiece. or supports.")
        ranges.append({"path": row["path"], "low": low, "high": high, "mode": row.get("mode", "absolute")})
    out = {"seed": seed, "ranges": ranges}
    if "sampled" in randomization:
        out["sampled"] = deepcopy(randomization["sampled"])
        out["rng"] = str(randomization.get("rng", ""))
    return out


def _check_geometry(scene):
    """Feasibility of a normalized scene; raises SceneInfeasible with a reason."""
    try:
        scene_heightfield(scene)
    except ValueError as error:
        raise SceneInfeasible("protrusion_outside_base", str(error)) from None
    base = scene["workpiece"]["base_box_m"]
    if not scene["solver"]["particle_spacing_m"] <= base["height_m"]/2:
        raise SceneInfeasible("spacing_too_coarse", "particle spacing must give at least two base layers")
    if "geometry_provider" in scene["workpiece"]:   # WP-12 Phase 4c opt-in providers: lattice-level validation
        from .geometry_providers import lattice_report, sample_lattice
        geometry = scene_heightfield(scene)
        if not scene["solver"]["particle_spacing_m"] <= geometry.min_height_m/2:
            raise SceneInfeasible("spacing_too_coarse", "particle spacing must give at least two layers under the "
                                                        "lowest point of the surface noise")
        report = lattice_report(sample_lattice(geometry, scene["solver"]["particle_spacing_m"], 1.)[2])
        if not report["ok"]:
            raise SceneInfeasible(report["problems"][0], f"lattice check {report}")
    supports = scene["supports"]
    if supports["kind"] == "flat_board":
        from .flat_board_workcell import check_placement
        problems = check_placement(flat_board_record(scene), _pose_xyz_yaw(scene), _footprint_local_xy(scene))
        if problems:
            raise SceneInfeasible(*problems[0])
    elif supports["kind"] != "boxes":
        lo, hi = supports["x_range_m"]
        if lo > -base["length_m"]/2 or hi < base["length_m"]/2:
            raise SceneInfeasible("supports_shorter_than_workpiece", f"x range {supports['x_range_m']}")
    if scene["workpiece"]["world_pose"]["translation_m"][2] <= scene["table_top_world_z_m"]:
        raise SceneInfeasible("workpiece_below_table", "workpiece bottom must lie above the table top")


def _path_tokens(path):
    return [int(token) if token.isdigit() else token for token in path.split(".")]


def _expand(doc, tokens, prefix=()):
    if not tokens:
        return [prefix]
    head, rest = tokens[0], tokens[1:]
    node = doc
    for token in prefix:
        node = node[token]
    if head == "*":
        if not isinstance(node, list):
            raise SceneConfigError(f"'*' needs a list at {'.'.join(map(str, prefix))}")
        return [path for i in range(len(node)) for path in _expand(doc, rest, prefix+(i,))]
    if (isinstance(node, list) and not (isinstance(head, int) and head < len(node))) or \
            (isinstance(node, dict) and head not in node) or not isinstance(node, (list, dict)):
        raise SceneConfigError(f"Randomization path {'.'.join(map(str, prefix+(head,)))} does not exist")
    return _expand(doc, rest, prefix+(head,))


def resolve_scene(source):
    """Validated scene with randomization applied and recorded (idempotent on a resolved scene)."""
    scene = validate_scene(source)
    randomization = scene["randomization"]
    if randomization is None or "sampled" in randomization:
        return scene
    rng = np.random.default_rng(randomization["seed"])
    resolved, sampled = deepcopy(scene), []
    for row in randomization["ranges"]:
        for path in _expand(resolved, _path_tokens(row["path"])):
            parent = resolved
            for token in path[:-1]:
                parent = parent[token]
            nominal = parent[path[-1]]
            if isinstance(nominal, bool) or not isinstance(nominal, (int, float)):
                raise SceneConfigError(f"Randomization path {'.'.join(map(str, path))} is not numeric")
            draw = float(rng.uniform(row["low"], row["high"]))
            value = nominal+draw if row["mode"] == "offset" else draw
            parent[path[-1]] = float(value)
            sampled.append({"path": ".".join(map(str, path)), "nominal": float(nominal), "draw": draw,
                            "mode": row["mode"], "value": float(value)})
    resolved["randomization"] = {**randomization, "sampled": sampled,
                                 "rng": "numpy.random.default_rng(seed).uniform in declared order"}
    return validate_scene(resolved)


# ---------------------------------------------------------------- scene geometry (config only, no oracle)

def scene_heightfield(scene):
    if "geometry_provider" in scene["workpiece"]:   # WP-12 Phase 4c opt-in (geometry_providers.py)
        from .geometry_providers import GeometryProviderError, geometry_from_scene
        try:
            return geometry_from_scene(scene["workpiece"])
        except GeometryProviderError as error:
            raise ValueError(str(error)) from None
    base = scene["workpiece"]["base_box_m"]
    bumps = tuple(Bump(row["x_m"], row["y_m"], row["radius_x_m"], row["radius_y_m"], row["height_m"], row["family"])
                  for row in scene["workpiece"]["protrusions"])
    return HeightFieldSpec(base["length_m"], base["width_m"], base["height_m"], bumps)


def scene_world_pose(scene):
    """World-from-workpiece XYZ+XYZW: translation and yaw about world z (exact identity at yaw 0)."""
    pose = scene["workpiece"]["world_pose"]
    yaw = pose["yaw_deg"]
    quaternion = (0., 0., 0., 1.) if yaw == 0 else tuple(map(float, Rotation.from_euler("z", yaw, degrees=True).as_quat()))
    return (*map(float, pose["translation_m"]), *quaternion)


def scene_top_m(scene):
    """Highest local material z (base plus the tallest protrusion)."""
    rows = scene["workpiece"]["protrusions"]
    noise = (scene["workpiece"].get("geometry_provider") or {}).get("surface_noise") or {}
    return (scene["workpiece"]["base_box_m"]["height_m"]+max((row["height_m"] for row in rows), default=0.)
            + noise.get("amplitude_m", 0.))


def flat_board_record(scene):
    """Resolved ``blade_master_flat_board_v1``-style workcell record of a flat-board scene, else None."""
    if scene["supports"]["kind"] != "flat_board":
        return None
    from .flat_board_workcell import resolve
    return resolve(scene["supports"]["workcell"])


def _pose_xyz_yaw(scene):
    pose = scene["workpiece"]["world_pose"]
    return (*map(float, pose["translation_m"]), float(pose["yaw_deg"]))


def _footprint_local_xy(scene):
    """Local xy outline of the material: base box corners and protrusion bounding rectangles."""
    base = scene["workpiece"]["base_box_m"]
    hx, hy = base["length_m"]/2, base["width_m"]/2
    points = [(sx*hx, sy*hy) for sx in (-1, 1) for sy in (-1, 1)]
    from .geometry_providers import protrusion_half_extents
    for row in scene["workpiece"]["protrusions"]:
        rx, ry = protrusion_half_extents(row)   # radius_x/y unless the row declares a yaw (heightfield_v2)
        points += [(row["x_m"]+sx*rx, row["y_m"]+sy*ry) for sx in (-1, 1) for sy in (-1, 1)]
    return np.asarray(points, float)


def raw_support_boxes(scene):
    """Local support FixtureBoxes before any workcell (top at workpiece z = 0).

    A ``flat_board`` scene returns the board's ``MaterialFixtureBox`` es from its
    workcell record (the same boxes the TLMPM fixture, the PhysX collider and the
    visual use; no riser).
    """
    supports = scene["supports"]
    if supports["kind"] == "flat_board":
        from .flat_board_workcell import fixture_boxes_local
        return tuple(fixture_boxes_local(flat_board_record(scene), _pose_xyz_yaw(scene), scene_material(scene).friction))
    if supports["kind"] == "boxes":
        return tuple(FixtureBox(row["name"], tuple(row["center_m"]), tuple(row["half_extents_m"])) for row in supports["boxes"])
    segments = support_segments(supports)
    names = supports["names"] or ([f"board_{i}" for i in range(len(segments))] if len(segments) > 1 else ["support_floor"])
    thickness = supports["thickness_m"]
    return tuple(FixtureBox(name, (round((lo+hi)/2, 12), 0., -thickness/2), (round((hi-lo)/2, 12), supports["half_width_y_m"], thickness/2))
                 for name, (lo, hi) in zip(names, segments))


def workcell_config(scene):
    """The cutting_board_v1 config dict of the scene workcell (None without one)."""
    workcell = scene["supports"]["workcell"]
    if workcell is None or scene["supports"]["kind"] == "flat_board":
        return None
    if "from_scenario" in workcell:
        from .task_scenarios import scenarios
        source = scenarios()[workcell["from_scenario"]]
        config = {key: deepcopy(source[key]) for key in ("cutting_board", "riser") }
        if "table" in source:
            config["table"] = deepcopy(source["table"])
    else:
        config = deepcopy(workcell)
    return {"extend_fixture_bases_to_table": True, "fixture_mode": "cutting_board_v1", **config}


def workcell_scenario(scene):
    """Name of the scenario a scene's workcell comes from (``supports.workcell.from_scenario``), else None."""
    workcell = scene["supports"]["workcell"]
    if scene["supports"]["kind"] == "flat_board":
        return None
    return workcell.get("from_scenario") if isinstance(workcell, dict) else None


def workcell_home(scene):
    """HOME declared by the scene's workcell scenario (``task_scenarios.scenario_home``), else None.

    Only ``from_scenario`` workcells can declare a HOME (wp11_scenarios_v1); an
    inline workcell and every v1-v3 scenario keep the planner default home.
    """
    if scene["supports"]["kind"] == "flat_board":
        from .flat_board_workcell import home
        return home(flat_board_record(scene))
    from .task_scenarios import scenario_home
    return scenario_home(workcell_scenario(scene))


def workcell_action_workspace(scene):
    """(min, max) local edge workspace declared by the scene's workcell scenario, else None."""
    if scene["supports"]["kind"] == "flat_board":
        from .flat_board_workcell import action_workspace
        return action_workspace(flat_board_record(scene))
    from .task_scenarios import action_workspace
    return action_workspace(workcell_scenario(scene))


def scene_material(scene):
    material = MaterialParameters(**_protocol()["material"])
    name = scene["workpiece"]["material_profile"]
    if name == V1_MATERIAL_PROFILE:
        return material
    from .materials_wp12 import is_profile, recipe
    if is_profile(name):   # WP-12 Phase 4c materials_wp12_v1 (opt-in)
        return recipe(name, material)
    from .food_materials import profiles
    profile = profiles()[name]
    return replace(material, **{key: value for key, value in profile.items() if key != "material_dt_s"},
                   calibration="provisional_uncalibrated")


# ---------------------------------------------------------------- configured task

class _ConfiguredIds(tuple):
    """TASK_IDS stand-in: any compiled ``CFG-...`` id is accepted by the CuttingTaskSpec guard."""

    def __contains__(self, task_id):
        return isinstance(task_id, str) and task_id.startswith("CFG-")


@dataclass(frozen=True)
class ConfiguredTaskSpec(ProfileTaskSpec):
    """A task compiled from an ``rcs_scene_v1`` scene and an ``rcs_goal_v1`` goal.

    The base-class fields carry the compiled workpiece, fixtures, tool mount,
    target and a dense reference schedule (ideal-tracking run of the goal
    planner): ``edge_position_knots_local_txyz`` at every control step, plus
    per-knot orientations, cut digits and phase labels, so ``reference_action``
    reproduces the scripted plan including REORIENT and multi-digit cuts. The
    frozen two-part v1 oracle is NOT_APPLICABLE; ``evaluate_goal`` is the
    goal-driven evaluator (``goals.evaluate_goal``).
    """

    max_cuts: int = 1
    solver_module: str = "tlmpm_v4"
    cutting_law: int = 8
    scene_json: str = ""
    goal_json: str = ""
    reference_orientations_local_xyzw: tuple = ()
    reference_cut_indices: tuple = ()     # digit armed on the motion from knot i to i+1 (-1 disarmed)
    reference_phase_labels: tuple = ()    # waypoint label pursued from knot i
    controller_json: str = ""
    TASK_IDS: ClassVar[tuple] = _ConfiguredIds()

    def __post_init__(self):
        super().__post_init__()
        knots = len(self.edge_position_knots_local_txyz)
        if (self.max_cuts < 1 or self.max_cuts > SOLVERS.get(self.solver_module, {"max_cuts": 0})["max_cuts"]
                or len(self.reference_orientations_local_xyzw) != knots or len(self.reference_cut_indices) != knots
                or len(self.reference_phase_labels) != knots):
            raise ValueError("A configured task needs a known solver, max_cuts within its limit and per-knot schedule rows")
        if any(not -1 <= index < self.max_cuts for index in self.reference_cut_indices):
            raise ValueError("Reference cut digits must be below the scene solver max_cuts")

    # -- configuration views
    @property
    def scene(self):
        return json.loads(self.scene_json)

    @property
    def goal(self):
        return json.loads(self.goal_json)

    @property
    def controller(self):
        return json.loads(self.controller_json) if self.controller_json else {}

    @property
    def protrusion_ids(self):
        return tuple(row["id"] for row in self.scene["workpiece"]["protrusions"])

    @property
    def fixture_materials(self):
        from .task_scenarios import fixture_materials
        return fixture_materials(self)

    @property
    def flat_board_record(self):
        """Resolved flat-board workcell record (WP-12, ADR 0017) or None."""
        return flat_board_record(self.scene)

    @property
    def table_box_world(self):
        """(center, half extents) WORLD table box declared by a flat-board workcell, else None (runtime default)."""
        record = self.flat_board_record
        if record is None:
            return None
        from .flat_board_workcell import table_box_world
        return table_box_world(record)

    def workcell_geometry(self):
        """Single-source geometry receipt of a flat-board workcell, else None."""
        record = self.flat_board_record
        if record is None:
            return None
        from .flat_board_workcell import geometry_receipt
        return geometry_receipt(record, _pose_xyz_yaw(self.scene), self.material.friction)

    @property
    def table_contact_material(self):
        record = self.flat_board_record
        if record is not None:
            from .flat_board_workcell import table_contact_material
            return table_contact_material(record, self.material.friction)
        config = workcell_config(self.scene)
        if not config or "table" not in config:
            return None
        from .task_scenarios import _resolve_friction
        friction = _resolve_friction(self, config["table"]["friction"])
        return dict(static_friction=friction, dynamic_friction=friction, restitution=float(config["table"]["restitution"]))

    def merged_config(self):
        return {"protocol_id": CONFIGURED_TASK_PROTOCOL, "scene": self.scene, "goal": self.goal,
                "controller": self.controller, "particle_spacing_m": self.particle_spacing_m}

    @property
    def config_sha256(self):
        return sha256_json(self.merged_config())

    @property
    def protocol_path(self):
        return None

    def protocol_document(self):
        return {"protocol_id": CONFIGURED_TASK_PROTOCOL, "common": _protocol()["common"],
                self.task_id: {**self.merged_config(), "particle_spacing_m": self.particle_spacing_m,
                               "config_sha256": self.config_sha256}}

    # -- reference schedule
    def _times(self):
        return np.asarray(self.edge_position_knots_local_txyz, float)[:, 0]

    def reference_edge_pose_local(self, time_s):
        if not np.isfinite(time_s) or not 0 <= time_s <= self.duration_s+1e-9:
            raise ValueError("Reference time must be within the task episode")
        knots = np.asarray(self.edge_position_knots_local_txyz, float)
        xyz = [np.interp(time_s, knots[:, 0], knots[:, axis]) for axis in (1, 2, 3)]
        times = knots[:, 0]
        i = int(np.clip(np.searchsorted(times, time_s, side="right")-1, 0, len(times)-2))
        q = np.asarray(self.reference_orientations_local_xyzw[i:i+2], float)
        u = float(np.clip((time_s-times[i])/(times[i+1]-times[i]), 0., 1.))
        orientation = Slerp([0., 1.], Rotation.from_quat(q))(u).as_quat()
        return np.r_[xyz, orientation]

    def active_cut_index(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError("Time must be finite")
        i = int(np.searchsorted(self._times(), time_s+1e-9, side="right")-1)
        return int(self.reference_cut_indices[i]) if 0 <= i < len(self.reference_cut_indices) else -1

    def reference_action(self, time_s):
        index = self.active_cut_index(time_s)
        return {"mode": "cartesian_edge_pose_world", "target": self.reference_edge_pose_world(time_s),
                "cut_enabled": index >= 0, **({"cut_index": index} if index >= 0 else {})}

    def phase_intervals_s(self):
        """{label: [first start, last end]} of the reference schedule (planned, not measured)."""
        times, intervals = self._times(), {}
        for i, label in enumerate(self.reference_phase_labels[:-1]):
            start, end = float(times[i]), float(times[i+1])
            if label in intervals:
                intervals[label][1] = end
            else:
                intervals[label] = [start, end]
        return intervals

    def validate_action(self, action, *, max_cuts=None):
        """As CuttingTaskSpec, but ``cut_index`` may address any digit below the scene max_cuts.

        An explicit ``max_cuts`` (the runtime material history) can only narrow that range."""
        index = action.get("cut_index") if isinstance(action, dict) else None
        if index is None:
            return super().validate_action(action)
        limit = self.max_cuts if max_cuts is None else min(int(self.max_cuts), int(max_cuts))
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < limit:
            raise ValueError(f"cut_index must be an integer in [0, {limit})")
        command = super().validate_action({key: value for key, value in action.items() if key != "cut_index"})
        command["cut_index"] = int(index)
        return command

    # -- receipts and evaluation
    def to_dict(self):
        protocol = self.protocol_document()
        return {**asdict(self), "protocol_id": protocol["protocol_id"], "protocol_sha256": self.config_sha256,
                "sample_coordinate_frame": "workpiece_local", "evaluator_coordinate_frame": "world",
                "action_modes": ACTION_MODES, "runtime_validated": False, "tool_profile": self.tool_profile.to_dict(),
                "configured_task": {"scene": self.scene, "goal": self.goal, "controller": self.controller,
                                    "config_sha256": self.config_sha256},
                "fixture_materials": list(self.fixture_materials), "frozen_two_part_oracle": "NOT_APPLICABLE",
                "numerical_overrides": ({"particle_spacing_m": {"scene": self.scene["solver"]["particle_spacing_m"],
                                                                 "actual": self.particle_spacing_m}}
                                        if self.particle_spacing_m != self.scene["solver"]["particle_spacing_m"] else {}),
                # WP-12 flat-board workcell only (absent for every other scene, so their receipts are unchanged)
                **({"workcell_geometry": self.workcell_geometry()} if self.flat_board_record is not None else {})}

    # -- runtime support (WP-11 Phase D)
    @property
    def home(self):
        """HOME declared by the scene workcell (joints, hand pose, source), or None."""
        return workcell_home(self.scene)

    def action_limits(self):
        """The ActionLimits the reference schedule was compiled with (planner step limits, workcell workspace)."""
        from . import goal_planner
        return goal_planner.action_limits(self.controller["planner"], self.scene)

    def evaluate_oracle(self, recording):
        return {"task_id": self.task_id, "status": "NOT_APPLICABLE", "checks": {}, "metrics": {}, "missing_evidence": [],
                "protocol_id": CONFIGURED_TASK_PROTOCOL, "protocol_sha256": self.config_sha256,
                "reason": "Configured tasks are scored by their declared goal, not the frozen two-part v1 oracle",
                "use_instead": "task.evaluate_goal(recording) (robotic_cutting_sim.goals.evaluate_goal)",
                "full_simulation_acceptance": False, "policy_observation": False}

    def evaluate_goal(self, recording, **options):
        from .goals import evaluate_goal
        return evaluate_goal(self.goal, recording, task=self, **options)

    def sample_particles(self):
        """As CuttingTaskSpec; an opt-in geometry provider (WP-12 Phase 4c) samples its own solid on the same lattice."""
        from .geometry_providers import ProviderGeometry, sample_lattice
        if not isinstance(self.geometry, ProviderGeometry):
            return super().sample_particles()
        return sample_lattice(self.geometry, self.particle_spacing_m, self.material.density_kg_m3)


def _goal_target(goal, scene):
    """(goal_kind, base task id, target origin, target normal, edge clearance) of a validated goal."""
    first = goal["steps"][0] if goal["type"] == "sequence" else goal
    height = scene["workpiece"]["base_box_m"]["height_m"]
    if first["type"] in ("remove_protrusions", "trim_plane"):
        plane_z = (first["plane_local_z_m"] if first["type"] == "remove_protrusions" else first["plane"]["origin_local_m"][2])
        plane_z = height if plane_z is None else plane_z
        return "flatten", "TRIM-01", (0., 0., float(plane_z)), (0., 0., 1.), float(first["edge_clearance_m"])
    from .goals import goal_planes
    planes = goal_planes(first, scene)
    if planes:
        origin, normal = planes[0]
        return "sever", "CUT-01", tuple(map(float, origin)), tuple(map(float, normal)), 0.
    return "sever", "CUT-01", (0., 0., 0.), (1., 0., 0.), 0.


def compile_task(scene, goal, *, particle_spacing_m=None, planner_options=None, seed=42):
    """Compile ``rcs_scene_v1`` + ``rcs_goal_v1`` into a ConfiguredTaskSpec.

    ``scene``/``goal`` are dicts or JSON paths (bare names resolve under
    ``configs/wp11/scenes`` / ``configs/wp11/goals``). The id is
    ``CFG-<scene_id>-<goal_id>-<hash8>`` over the merged config; duration and
    control dt come from the goal; the reference schedule is the goal planner's
    plan run on an ideal integrator with the controller's action limits.
    ``particle_spacing_m`` is an explicit numerical override (recorded).
    """
    from . import goal_planner
    from .goals import validate_goal
    from .policies import WaypointPolicy
    from .tool_profile import PROFILES
    scene = resolve_scene(scene)
    goal = validate_goal(goal)
    profile = PROFILES[scene["tool"]["profile_id"]]
    spacing = scene["solver"]["particle_spacing_m"] if particle_spacing_m is None else float(particle_spacing_m)
    if not spacing <= scene["workpiece"]["base_box_m"]["height_m"]/2:
        raise SceneInfeasible("spacing_too_coarse", "particle spacing must give at least two base layers")
    episode = goal["episode"]
    options = goal_planner.planner_options(goal, planner_options)
    waypoints, report = goal_planner.plan_with_report(scene, goal, profile, options)
    digits = report["digits_used"]
    if digits > scene["solver"]["max_cuts"]:
        raise SceneConfigError(f"Goal {goal['goal_id']!r} needs {digits} cut digits; scene solver "
                               f"{scene['solver']['module']} declares max_cuts {scene['solver']['max_cuts']}")
    goal_kind, base_id, origin, normal, clearance = _goal_target(goal, scene)
    world = scene_world_pose(scene)
    controller = {"planner": options, "plan_report_sha256": sha256_json(_jsonable(report)),
                  "reference": "goal_planner plan on an ideal integrator (policies.WaypointPolicy + control.resolve_delta)"}
    merged = {"protocol_id": CONFIGURED_TASK_PROTOCOL, "scene": scene, "goal": goal, "controller": controller,
              "particle_spacing_m": spacing}
    task_id = f"CFG-{scene['scene_id']}-{goal['goal_id']}-{sha256_json(merged)[:8]}"
    home = np.asarray(waypoints[0].pose, float)
    duration, dt = episode["duration_s"], episode["control_dt_s"]
    common = dict(geometry=scene_heightfield(scene), particle_spacing_m=spacing, grid_dx_m=scene["solver"]["grid_dx_m"],
                  fixture_boxes_local=raw_support_boxes(scene), world_from_workpiece_xyzw=world,
                  table_top_world_z_m=scene["table_top_world_z_m"], hand_from_edge_xyz_xyzw=tuple(profile.hand_from_edge()),
                  duration_s=duration, control_dt_s=dt, material=scene_material(scene), goal_kind=goal_kind,
                  target_origin_local_m=origin, target_normal_local=normal, edge_clearance_m=clearance,
                  blade_length_m=profile.blade_depth_m, blade_height_m=profile.thickness_m, blade_half_width_m=profile.half_width_m,
                  tool_profile_id=profile.profile_id, base_task_id=base_id, max_cuts=scene["solver"]["max_cuts"],
                  solver_module=scene["solver"]["module"], cutting_law=scene["solver"]["cutting_law"],
                  scene_json=canonical_json(scene), goal_json=canonical_json(goal), controller_json=canonical_json(controller))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_quat(world[3:]).as_matrix(), world[:3]
    T_inv = np.linalg.inv(T)
    home_local = np.r_[T_inv[:3, :3]@home[:3]+T_inv[:3, 3], (Rotation.from_matrix(T_inv[:3, :3])*Rotation.from_quat(home[3:])).as_quat()]
    provisional = ConfiguredTaskSpec(task_id, int(seed), "fixed", edge_orientation_local_xyzw=tuple(home_local[3:]),
                                     edge_position_knots_local_txyz=((0., *home_local[:3]), (duration, *home_local[:3])),
                                     reference_orientations_local_xyzw=(tuple(home_local[3:]),)*2,
                                     reference_cut_indices=(-1, -1), reference_phase_labels=("HOME", "HOME"), **common)
    if scene["supports"]["workcell"] is not None and scene["supports"]["kind"] != "flat_board":
        from .task_scenarios import _cutting_board_boxes
        boxes = _cutting_board_boxes(provisional, workcell_config(scene), world[2])
        provisional = replace(provisional, fixture_boxes_local=tuple(boxes))
    limits = goal_planner.action_limits(options, scene)
    schedule = ideal_schedule(provisional, waypoints, limits, goal_planner.policy_options(options))
    if schedule is None:
        raise SceneConfigError(f"The planned motion does not finish within the goal duration {duration} s; "
                               f"needs {goal_planner.ideal_duration_s(provisional, waypoints, limits, options):.2f} s")
    times, poses, cuts, labels = schedule
    local = [np.r_[T_inv[:3, :3]@p[:3]+T_inv[:3, 3], (Rotation.from_matrix(T_inv[:3, :3])*Rotation.from_quat(p[3:])).as_quat()]
             for p in poses]
    armed = [i for i, c in enumerate(cuts) if c >= 0]
    nominal = local[armed[0]][3:] if armed else local[0][3:]
    interval = (float(times[armed[0]]), float(times[armed[-1]+1])) if armed else (0., 0.)
    return replace(provisional, edge_orientation_local_xyzw=tuple(map(float, nominal)),
                   edge_position_knots_local_txyz=tuple((float(t), *map(float, p[:3])) for t, p in zip(times, local)),
                   reference_orientations_local_xyzw=tuple(tuple(map(float, p[3:])) for p in local),
                   reference_cut_indices=tuple(int(c) for c in cuts), reference_phase_labels=tuple(labels),
                   active_cut_interval_s=interval)


def configured_task_payload(task):
    """Canonical JSON that recompiles ``task`` exactly (runtime configs carry this, not the object).

    Recompiling is deterministic: ``task_from_payload`` re-runs ``compile_task`` on
    the recorded scene, goal, planner options, spacing and seed and requires the
    same id and config sha256, so a runtime can never silently build another task.
    """
    if not isinstance(task, ConfiguredTaskSpec):
        raise TypeError("Expected a ConfiguredTaskSpec from compile_task")
    spacing = None if task.particle_spacing_m == task.scene["solver"]["particle_spacing_m"] else float(task.particle_spacing_m)
    return canonical_json({"protocol_id": CONFIGURED_TASK_PROTOCOL, "task_id": task.task_id, "seed": int(task.seed),
                           "config_sha256": task.config_sha256, "scene": task.scene, "goal": task.goal,
                           "planner_options": task.controller["planner"], "particle_spacing_m": spacing})


def task_from_payload(payload):
    """The ConfiguredTaskSpec of a ``configured_task_payload`` string (verified id and sha256)."""
    data = json.loads(payload) if isinstance(payload, str) else deepcopy(payload)
    _keys(data, "configured task payload", ("protocol_id", "task_id", "seed", "config_sha256", "scene", "goal",
                                            "planner_options", "particle_spacing_m"))
    if data["protocol_id"] != CONFIGURED_TASK_PROTOCOL:
        raise SceneConfigError(f"Configured task payload must be {CONFIGURED_TASK_PROTOCOL}")
    task = compile_task(data["scene"], data["goal"], particle_spacing_m=data["particle_spacing_m"],
                        planner_options=data["planner_options"], seed=data["seed"])
    if task.task_id != data["task_id"] or task.config_sha256 != data["config_sha256"]:
        raise SceneConfigError(f"Recompiled task {task.task_id} ({task.config_sha256[:12]}) differs from the payload "
                               f"{data['task_id']} ({data['config_sha256'][:12]})")
    return task


def ideal_schedule(task, waypoints, limits, policy_options):
    """Dense control-rate (times, WORLD poses, armed digit per motion, label per knot); None if it overruns."""
    from .control import resolve_delta
    from .policies import WaypointPolicy
    policy = WaypointPolicy(task, list(waypoints), limits, **policy_options)
    dt, duration = task.control_dt_s, task.duration_s
    steps = int(round(duration/dt))
    edge = np.asarray(waypoints[0].pose, float)
    poses, cuts, labels = [edge.copy()], [], []
    for step in range(steps+1):
        action = policy({"proprio": {"tool_pose": edge.copy()}, "timestamp_s": step*dt})
        if policy.done:
            break
        if step == steps:
            return None
        labels.append(policy.phase)
        cuts.append(int(action.get("cut_index", 0)) if action["cut_enabled"] else -1)
        if action["mode"] == "cartesian_edge_pose_world":  # absolute targets (controller profile path_v3)
            from .control import bound_target
            edge, _ = bound_target(edge, action["target"], limits, task.T_world_workpiece)
        else:
            edge, _ = resolve_delta(edge, action["target"], limits, frame=action.get("frame", "world"),
                                    T_world_workpiece=task.T_world_workpiece)
        poses.append(edge.copy())
    while len(poses) < steps+1:
        labels.append("DONE")
        cuts.append(-1)
        poses.append(poses[-1].copy())
    labels.append("DONE")
    cuts.append(-1)
    times = [round(i*dt, 9) for i in range(steps+1)]
    times[-1] = duration
    return times, poses, cuts, labels


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_configured_kinematic_env(task, **options):
    """``make_kinematic_env(task=task)`` with the scene solver's cutting law and max_cuts.

    ``tlmpm_v5`` scenes raise NotImplementedError until the kinematic backend
    exposes that solver; nothing falls back silently to another solver.
    """
    from .kinematic_env import KinematicConfig, make_kinematic_env
    if not isinstance(task, ConfiguredTaskSpec):
        raise TypeError("Expected a ConfiguredTaskSpec from compile_task")
    if task.solver_module == "tlmpm_v5" and "solver" not in KinematicConfig.__dataclass_fields__:
        raise NotImplementedError("tlmpm_v5 is not available in the kinematic backend yet (ADR 0014 pending)")
    if task.solver_module == "tlmpm_v5":
        options.setdefault("solver", "tlmpm_v5")
    declared = task.scene["solver"].get("slots_per_node")
    if declared is not None:  # opt-in scene slot capacity (LiveCuttingMaterial also reads it from the task)
        if options.get("slots_per_node") not in (None, declared):
            raise ValueError(f"slots_per_node {options['slots_per_node']} differs from the scene's {declared}")
        options["slots_per_node"] = declared
    options.setdefault("cutting_law", task.cutting_law)
    options.setdefault("max_cuts", task.max_cuts)
    from .materials_wp12 import admit, is_profile
    if is_profile(task.scene["workpiece"]["material_profile"]):   # refuse an unstable (dt, dx) before building a solver
        admit(task.scene["workpiece"]["material_profile"], task.grid_dx_m, options.get("material_dt", .0001),
              task.particle_spacing_m)
    return make_kinematic_env(task=task, **options)
