"""Task goals (``rcs_goal_v1``) and their read-only evaluators (ADR 0015, WP-11 Phase S).

A goal says WHAT SHOULD BE ACHIEVED, independently of the scene and of the
controller. Each goal type has its own evaluator; success follows the goal, not
a universal flat-surface rule. Evaluators only read a recorded WORLD episode
(the arrays ``CuttingTaskEnv.evaluate`` builds); oracle material state is
evaluation-only and never an observation.

Schema ``rcs_goal_v1`` (JSON under ``configs/wp11/goals/``; positions and
directions in WORKPIECE LOCAL coordinates, so they follow the scene pose/yaw;
unknown keys are errors)::

    {"schema": "rcs_goal_v1", "goal_id": "sever_mid", "type": <type>, "description": "...",
     "episode": {"duration_s": 8.0, "control_dt_s": 0.02},     # control dt default 0.02 s
     "planner": {...},                                          # optional goal_planner options
     ...type fields...}

Types and fields (defaults in brackets):

* ``sever``: ``planes`` [{origin_local_m, normal_local, approach_local?}],
  ``expected_pieces`` [len(planes)+1], ``minimum_part_mass_fraction`` [0.4 for one
  plane, else 0.5/pieces], ``minimum_main_parts_mass_fraction`` [0.99],
  ``plane_error_bound_m`` [0.002]. Location: the frozen 2 mm midpoint statistic
  (``tasks._interface_location``) for grid-aligned planes; otherwise the
  ADR 0006 Amendment 1 swept-plane rule (actual statistic <= ideal-geometric-label
  statistic + one particle spacing, and no particle farther than two spacings
  from every plane in a substantial piece of another region).
* ``divide``: ``pieces`` N, ``axis_local`` [[1,0,0]], ``positions_m`` [equal length],
  ``minimum_part_mass_fraction`` [0.5/N], ``minimum_main_parts_mass_fraction`` [0.99].
  Verdict: N substantial pieces holding the mass; plane location is quality.
* ``remove_protrusions``: ``select`` "all" | [ids] | {"region_local_m": {"x": [lo, hi], "y": [lo, hi]}},
  ``plane_local_z_m`` [base height], ``edge_clearance_m`` [0.0002],
  ``minimum_removed_fraction`` [0.9] per selected protrusion (on the particles
  initially above the plane that belong to it: the bump with the largest height
  contribution at the particle), ``minimum_substrate_share`` [0.99],
  ``maximum_detached_below_fraction`` [0.01],
  ``minimum_unselected_retained_fraction`` [0.9, null disables]. Precision
  (residual heights, optional surface metrics) is reported as quality only.
* ``trim_plane``: ``plane`` {origin_local_m, normal_local}, ``region_local_m`` [whole],
  ``edge_clearance_m``, ``minimum_removed_fraction``, ``minimum_substrate_share``,
  ``maximum_detached_below_fraction``. Separation sets the verdict; precision is quality.
* ``move_piece``: ``selector`` {"side_of_plane": {origin_local_m, normal_local, side}} |
  {"box_local_m": {"min": [3], "max": [3]}}, ``direction_local``, ``distance_m`` [0.02]
  (planner), ``minimum_displacement_m`` [0.01], ``maximum_other_displacement_m`` [0.002],
  ``minimum_piece_mass_fraction`` [0.05], ``minimum_selector_share`` [0.5],
  ``push_edge_height_m`` [0.003], ``push_speed_m_s`` [0.01]. No label change while
  disarmed during the move window. Opt-in planner fields (absent unless declared, so
  goals without them are unchanged): ``push_mode`` "kerf_face" (the default
  vertical blade-face push in a kerf) | "surface_sweep" (WP-11 C5b: a disarmed,
  near-horizontal sweep of the base top along ``direction_local`` at
  ``sweep_height_m`` [0.002] above it, from behind the selector box to
  ``sweep_overtravel_m`` [0.005] past the base end; ``distance_m`` is not used).
* ``sequence``: ``steps`` [inline step goals without schema/episode, or {"ref": "file.json"}];
  every step must hold at the end; per-step results are reported.

Evaluator version (opt-in, top-level ``evaluator_version``; absent = ``goal_eval_v1``,
so the normalized form, compiled task ids and verdicts of goals that do not declare it
are unchanged): ``goal_eval_v2`` (ADR 0015 Amendment 1) scopes the steps of a
``sequence`` to what each step does. Cut digits are allocated to steps by the declared
convention (steps in order; sever = one digit per plane, divide = pieces - 1,
remove_protrusions = one per selected protrusion, trim_plane = one, move_piece = none),
cross-checked against the goal planner's plan report when a compiled task is given.
A sever/divide step is scored on the components of its OWN digits only (piece checks
and plane location). A remove_protrusions/trim_plane step is scored on the components
of the earlier sever/divide steps' digits plus its own: substrate integrity is judged
per substrate piece that existed before the step (each must keep one substrate
component holding its below-plane mass), and removal/retention use that scoped
substrate. move_piece steps and non-sequence goals are scored exactly as v1. Every
threshold is the v1 threshold.

Evaluation output (``evaluate_goal``) keeps three layers separate:
``integrity`` (recording/history validity: initial state, digit persistence,
sampler/mass match, completeness), ``outcome`` (the goal verdict and per-item
results) and ``quality`` (precision metrics, never part of the verdict). Codes
may be ternary (v1-v4, ``c = sum d_k 3^k``) or packed (``tlmpm_v5``, 2 bits per
digit, ``recording["code_encoding"] = "packed2"``, as ``tlmpm_v5`` materials report).
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from .cut_topology import mechanical_components

GOAL_SCHEMA = "rcs_goal_v1"
EVALUATION_SCHEMA = "rcs_goal_evaluation_v1"
GOAL_TYPES = ("sever", "divide", "remove_protrusions", "trim_plane", "move_piece", "sequence")
ENCODINGS = ("ternary", "packed2")
EVALUATOR_VERSIONS = ("goal_eval_v1", "goal_eval_v2")
_ENCODING_ALIASES = {"packed_2bit": "packed2"}
PROVENANCE = "Recomputed actual eight-node compatibility graph including positive weights, stress gradients and code-zero bridges"
_STEP_TYPES = GOAL_TYPES[:-1]
_TYPE_FIELDS = {
    "sever": {"planes": None, "expected_pieces": None, "minimum_part_mass_fraction": None,
              "minimum_main_parts_mass_fraction": .99, "plane_error_bound_m": .002},
    "divide": {"pieces": None, "axis_local": [1., 0., 0.], "positions_m": None, "minimum_part_mass_fraction": None,
               "minimum_main_parts_mass_fraction": .99, "plane_error_bound_m": .002},
    "remove_protrusions": {"select": "all", "plane_local_z_m": None, "edge_clearance_m": .0002,
                           "minimum_removed_fraction": .9, "minimum_substrate_share": .99,
                           "maximum_detached_below_fraction": .01, "minimum_unselected_retained_fraction": .9},
    "trim_plane": {"plane": None, "region_local_m": None, "edge_clearance_m": .0002, "minimum_removed_fraction": .9,
                   "minimum_substrate_share": .99, "maximum_detached_below_fraction": .01},
    "move_piece": {"selector": None, "direction_local": None, "distance_m": .02, "minimum_displacement_m": .01,
                   "maximum_other_displacement_m": .002, "minimum_piece_mass_fraction": .05,
                   "minimum_selector_share": .5, "push_edge_height_m": .003, "push_speed_m_s": .01},
    "sequence": {"steps": None},
}
# Opt-in fields: validated and kept only when declared (no defaults are filled in), so the normalized
# form and compiled task ids of goals that do not declare them are unchanged.
_OPT_IN_FIELDS = {"move_piece": ("push_mode", "sweep_height_m", "sweep_overtravel_m")}
PUSH_MODES = ("kerf_face", "surface_sweep")
_REQUIRED = {"sever": ("planes",), "divide": ("pieces",), "trim_plane": ("plane",),
             "move_piece": ("selector", "direction_local"), "sequence": ("steps",)}


def _error(message):
    from .scene_config import SceneConfigError
    return SceneConfigError(message)


def _unit(value, where):
    from .scene_config import _vector
    vector = np.asarray(_vector(value, where), float)
    norm = np.linalg.norm(vector)
    if not norm > 0:
        raise _error(f"{where} must be a nonzero vector")
    if not np.isclose(norm, 1., atol=1e-9, rtol=0):
        raise _error(f"{where} must be a unit vector")
    return vector.tolist()


def _plane(row, where, approach=False):
    from .scene_config import _keys, _vector
    _keys(row, where, ("origin_local_m", "normal_local"), ("approach_local",) if approach else ())
    out = {"origin_local_m": _vector(row["origin_local_m"], f"{where}.origin_local_m"),
           "normal_local": _unit(row["normal_local"], f"{where}.normal_local")}
    if approach and "approach_local" in row:
        out["approach_local"] = _unit(row["approach_local"], f"{where}.approach_local")
    return out


def _fraction(value, where, *, allow_none=False):
    if value is None and allow_none:
        return None
    from .scene_config import _number
    value = _number(value, where)
    if not 0 <= value <= 1:
        raise _error(f"{where} must be within [0, 1]")
    return value


def _validate_fields(goal, kind, where, directory):
    from .scene_config import _integer, _keys, _number, _vector
    fields = _TYPE_FIELDS[kind]
    out = {key: deepcopy(goal.get(key, default)) for key, default in fields.items()}
    for key in _REQUIRED.get(kind, ()):
        if goal.get(key) is None:
            raise _error(f"{where}.{key} is required for a {kind} goal")
    if kind == "sever":
        if not isinstance(out["planes"], list) or not out["planes"]:
            raise _error(f"{where}.planes must be a nonempty list")
        out["planes"] = [_plane(row, f"{where}.planes[{i}]", approach=True) for i, row in enumerate(out["planes"])]
        pieces = len(out["planes"])+1 if out["expected_pieces"] is None else _integer(out["expected_pieces"], f"{where}.expected_pieces", 2)
        out["expected_pieces"] = pieces
        if out["minimum_part_mass_fraction"] is None:
            out["minimum_part_mass_fraction"] = .4 if pieces == 2 else .5/pieces
    elif kind == "divide":
        pieces = out["pieces"] = _integer(out["pieces"], f"{where}.pieces", 2, 16)
        out["axis_local"] = _unit(out["axis_local"], f"{where}.axis_local")
        if out["positions_m"] is not None:
            positions = _vector(out["positions_m"], f"{where}.positions_m", pieces-1)
            if np.any(np.diff(positions) <= 0):
                raise _error(f"{where}.positions_m must be increasing")
            out["positions_m"] = positions
        if out["minimum_part_mass_fraction"] is None:
            out["minimum_part_mass_fraction"] = .5/pieces
    elif kind == "remove_protrusions":
        select = out["select"]
        if select != "all":
            if isinstance(select, list):
                if not select or len(set(select)) != len(select) or not all(isinstance(s, str) for s in select):
                    raise _error(f"{where}.select must be 'all', a list of distinct protrusion ids, or a region")
            else:
                _keys(select, f"{where}.select", ("region_local_m",))
                out["select"] = {"region_local_m": _region(select["region_local_m"], f"{where}.select.region_local_m")}
        if out["plane_local_z_m"] is not None:
            out["plane_local_z_m"] = _number(out["plane_local_z_m"], f"{where}.plane_local_z_m")
        out["edge_clearance_m"] = _number(out["edge_clearance_m"], f"{where}.edge_clearance_m", nonnegative=True)
        out["minimum_unselected_retained_fraction"] = _fraction(out["minimum_unselected_retained_fraction"],
                                                                f"{where}.minimum_unselected_retained_fraction", allow_none=True)
    elif kind == "trim_plane":
        out["plane"] = _plane(out["plane"], f"{where}.plane")
        if out["region_local_m"] is not None:
            out["region_local_m"] = _region(out["region_local_m"], f"{where}.region_local_m")
        out["edge_clearance_m"] = _number(out["edge_clearance_m"], f"{where}.edge_clearance_m", nonnegative=True)
    elif kind == "move_piece":
        selector = out["selector"]
        if not isinstance(selector, dict) or len(selector) != 1:
            raise _error(f"{where}.selector needs exactly one of side_of_plane / box_local_m")
        if "side_of_plane" in selector:
            row = selector["side_of_plane"]
            _keys(row, f"{where}.selector.side_of_plane", ("origin_local_m", "normal_local", "side"))
            if row["side"] not in ("positive", "negative"):
                raise _error(f"{where}.selector.side_of_plane.side must be positive or negative")
            out["selector"] = {"side_of_plane": {**_plane({k: row[k] for k in ("origin_local_m", "normal_local")},
                                                          f"{where}.selector.side_of_plane"), "side": row["side"]}}
        elif "box_local_m" in selector:
            row = selector["box_local_m"]
            _keys(row, f"{where}.selector.box_local_m", ("min", "max"))
            low, high = _vector(row["min"], f"{where}.selector.box_local_m.min"), _vector(row["max"], f"{where}.selector.box_local_m.max")
            if np.any(np.asarray(high) <= np.asarray(low)):
                raise _error(f"{where}.selector.box_local_m must be nonempty")
            out["selector"] = {"box_local_m": {"min": low, "max": high}}
        else:
            raise _error(f"{where}.selector needs side_of_plane or box_local_m")
        out["direction_local"] = _unit(out["direction_local"], f"{where}.direction_local")
        for key in ("distance_m", "minimum_displacement_m", "maximum_other_displacement_m", "push_edge_height_m", "push_speed_m_s"):
            out[key] = _number(out[key], f"{where}.{key}", positive=True)
        if "push_mode" in goal:
            if goal["push_mode"] not in PUSH_MODES:
                raise _error(f"{where}.push_mode must be one of {PUSH_MODES}")
            out["push_mode"] = goal["push_mode"]
        for key in ("sweep_height_m", "sweep_overtravel_m"):
            if key in goal:
                if goal.get("push_mode") != "surface_sweep":
                    raise _error(f"{where}.{key} applies to push_mode surface_sweep only")
                out[key] = _number(goal[key], f"{where}.{key}", positive=True)
    else:  # sequence
        if not isinstance(out["steps"], list) or not out["steps"]:
            raise _error(f"{where}.steps must be a nonempty list")
        steps = []
        for i, step in enumerate(out["steps"]):
            if isinstance(step, dict) and set(step) == {"ref"}:
                from .scene_config import load_json
                step = load_json(step["ref"], directory)
                step = {key: value for key, value in step.items() if key not in ("schema", "episode", "planner")}
            steps.append(_validate_step(step, f"{where}.steps[{i}]", directory))
        out["steps"] = steps
    for key, value in out.items():
        if key.startswith(("minimum_", "maximum_")) and key.endswith("fraction") or key.endswith("_share"):
            out[key] = _fraction(value, f"{where}.{key}", allow_none=key == "minimum_unselected_retained_fraction")
    if kind in ("sever", "divide"):
        out["plane_error_bound_m"] = _number(out["plane_error_bound_m"], f"{where}.plane_error_bound_m", positive=True)
    return out


def _region(row, where):
    from .scene_config import _keys, _vector
    _keys(row, where, (), ("x", "y"))
    out = {}
    for axis in ("x", "y"):
        if axis in row:
            lo, hi = _vector(row[axis], f"{where}.{axis}", 2)
            if not lo < hi:
                raise _error(f"{where}.{axis} must be increasing")
            out[axis] = [lo, hi]
    return out


def _validate_step(step, where, directory):
    from .scene_config import _identifier, _keys
    if not isinstance(step, dict) or step.get("type") not in _STEP_TYPES:
        raise _error(f"{where}.type must be one of {_STEP_TYPES} (sequences do not nest)")
    kind = step["type"]
    _keys(step, where, ("type",), ("goal_id", "description", *_TYPE_FIELDS[kind], *_OPT_IN_FIELDS.get(kind, ())))
    out = {"type": kind, "goal_id": _identifier(step.get("goal_id", f"step_{kind}"), f"{where}.goal_id"),
           "description": str(step.get("description", ""))}
    out.update(_validate_fields(step, kind, where, directory))
    return out


def validate_goal(source, directory=None):
    """Validate an ``rcs_goal_v1`` document; returns a normalized copy with defaults filled."""
    from .scene_config import GOAL_DIR, _identifier, _keys, _number, load_json
    directory = GOAL_DIR if directory is None else Path(directory)
    if not isinstance(source, dict) and Path(source).exists():
        directory = Path(source).resolve().parent
    goal = load_json(source, directory)
    if not isinstance(goal, dict) or goal.get("type") not in GOAL_TYPES:
        raise _error(f"goal.type must be one of {GOAL_TYPES}")
    kind = goal["type"]
    _keys(goal, "goal", ("schema", "goal_id", "type", "episode"),
          ("description", "planner", "evaluator_version", *_TYPE_FIELDS[kind], *_OPT_IN_FIELDS.get(kind, ())))
    if goal["schema"] != GOAL_SCHEMA:
        raise _error(f"goal.schema must be {GOAL_SCHEMA!r}")
    episode = goal["episode"]
    _keys(episode, "goal.episode", ("duration_s",), ("control_dt_s",))
    duration = _number(episode["duration_s"], "goal.episode.duration_s", positive=True)
    dt = _number(episode.get("control_dt_s", .02), "goal.episode.control_dt_s", positive=True)
    if not np.isclose(duration/dt, round(duration/dt), atol=1e-9):
        raise _error("goal.episode.duration_s must be an integral number of control steps")
    planner = goal.get("planner", {})
    if not isinstance(planner, dict):
        raise _error("goal.planner must be an object")
    out = {"schema": GOAL_SCHEMA, "goal_id": _identifier(goal["goal_id"], "goal.goal_id"), "type": kind,
           "description": str(goal.get("description", "")), "episode": {"duration_s": duration, "control_dt_s": dt},
           "planner": deepcopy(planner)}
    out.update(_validate_fields(goal, kind, "goal", directory))
    if "evaluator_version" in goal:  # opt-in: kept only when declared (ADR 0015 Amendment 1)
        if goal["evaluator_version"] not in EVALUATOR_VERSIONS:
            raise _error(f"goal.evaluator_version must be one of {EVALUATOR_VERSIONS}")
        out["evaluator_version"] = goal["evaluator_version"]
    if planner:
        from .goal_planner import planner_options
        planner_options(out)  # unknown planner options are schema errors too
    return out


def goal_planes(goal, scene):
    """[(origin_local, unit normal_local)] cut planes of a sever/divide goal (empty for other types)."""
    if goal["type"] == "sever":
        return [(np.asarray(row["origin_local_m"], float), np.asarray(row["normal_local"], float)) for row in goal["planes"]]
    if goal["type"] == "divide":
        axis = np.asarray(goal["axis_local"], float)
        if goal["positions_m"] is not None:
            positions = goal["positions_m"]
        else:
            base = scene["workpiece"]["base_box_m"]
            extent = abs(axis[0])*base["length_m"]+abs(axis[1])*base["width_m"]+abs(axis[2])*base["height_m"]
            positions = [-extent/2+extent*k/goal["pieces"] for k in range(1, goal["pieces"])]
        offset = np.array([0., 0., 0.])
        return [(offset+axis*p, axis) for p in positions]
    return []


# ---------------------------------------------------------------- codes and components

def decode_digit(codes, digit, encoding="ternary"):
    codes = np.asarray(codes, np.int64)
    return (codes//3**int(digit)) % 3 if encoding == "ternary" else (codes >> (2*int(digit))) & 3


def packed_conflicting(a, b):
    """ADR 0014 bitmask test: some digit is 1 in one code and 2 in the other."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    low = 0x55555555
    a1, a2 = a & low & ~(a >> 1), (a >> 1) & low & ~a
    b1, b2 = b & low & ~(b >> 1), (b >> 1) & low & ~b
    return ((a1 & b2) | (a2 & b1)) != 0


def packed_components(nodes, weights, codes, *, gradients=None):
    """``cut_topology.mechanical_components`` with the packed conflict test (same grouping and union-find)."""
    from . import cut_topology
    official = getattr(cut_topology, "mechanical_components_packed", None)
    if official is not None:
        return official(nodes, weights, codes, gradients=gradients)
    codes = np.asarray(codes, np.int64)
    parent = np.arange(len(codes))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for group in _node_groups(nodes, weights, gradients):
        representatives = {}
        for p in group:
            c = int(codes[p])
            if c in representatives:
                a, b = find(p), find(representatives[c])
                if a != b:
                    parent[b] = a
            else:
                representatives[c] = p
        occupied = list(representatives)
        for i, a in enumerate(occupied):
            for b in occupied[i+1:]:
                if not packed_conflicting(a, b):
                    x, y = find(representatives[a]), find(representatives[b])
                    if x != y:
                        parent[y] = x
    roots = np.array([find(i) for i in range(len(codes))])
    unique, labels = np.unique(roots, return_inverse=True)
    return len(unique), labels


def _node_groups(nodes, weights, gradients):
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    support = (weights > 0).ravel()
    if gradients is not None:
        support |= (np.linalg.norm(np.asarray(gradients), axis=-1) > 0).ravel()
    particle = np.repeat(np.arange(nodes.shape[0]), nodes.shape[1])[support]
    node = nodes.ravel()[support]
    order = np.argsort(node, kind="stable")
    return np.split(particle[order], np.flatnonzero(np.diff(node[order]))+1)


class _Recording:
    """Validated view of a recorded WORLD episode plus a component cache."""

    def __init__(self, data, task=None, encoding=None, cuts=None):
        required = ("time_s", "positions", "codes", "mass_kg", "reference_positions",
                    "reference_stencil_nodes", "reference_stencil_weights", "reference_stencil_gradients")
        self.missing = [key for key in required if key not in data]
        self.data, self.task = data, task
        if self.missing:
            return
        self.time = np.asarray(data["time_s"], float)
        self.positions = np.asarray(data["positions"], float)
        self.codes = np.asarray(data["codes"])
        self.masses = np.asarray(data["mass_kg"], float)
        self.reference = np.asarray(data["reference_positions"], float)
        self.nodes = np.asarray(data["reference_stencil_nodes"])
        self.weights = np.asarray(data["reference_stencil_weights"])
        self.gradients = np.asarray(data["reference_stencil_gradients"])
        n = len(self.masses)
        if self.codes.ndim == 1:
            self.codes = self.codes[None]
        if (self.reference.shape != (n, 3) or self.positions.ndim != 3 or self.positions.shape[1:] != (n, 3)
                or self.codes.shape != (len(self.time), n) or len(self.positions) != len(self.time)
                or self.codes.dtype.kind not in "iu" or self.nodes.shape[0] != n or self.weights.shape != self.nodes.shape
                or self.gradients.shape != (*self.nodes.shape, 3) or self.nodes.dtype.kind not in "iu"
                or np.any(np.diff(self.time) <= 0) or np.any(self.masses <= 0) or np.any(self.codes < 0)
                or not all(np.isfinite(v).all() for v in (self.time, self.positions, self.masses, self.reference,
                                                          self.weights, self.gradients))):
            raise ValueError("Goal recording shapes, times, masses, codes or stencils are invalid")
        self.encoding = encoding or data.get("code_encoding", "ternary")
        self.encoding = _ENCODING_ALIASES.get(self.encoding, self.encoding)
        if self.encoding not in ENCODINGS:
            raise ValueError(f"code_encoding must be one of {ENCODINGS}")
        declared = cuts or data.get("max_cuts") or getattr(task, "max_cuts", None)
        top = int(self.codes.max(initial=0))
        needed = 1
        while (3**needed if self.encoding == "ternary" else 4**needed) <= top:
            needed += 1
        self.cuts = max(int(declared or 1), needed)
        self.spacing = data.get("particle_spacing_m", getattr(task, "particle_spacing_m", None))
        self._cache = {}

    def components(self, frame=-1):
        return self.components_of(self.codes[frame])

    def components_of(self, codes):
        codes = np.asarray(codes)
        key = codes.tobytes()
        if key not in self._cache:
            if self.encoding == "ternary":
                self._cache[key] = mechanical_components(self.nodes, self.weights, codes, self.cuts, gradients=self.gradients)
            else:
                self._cache[key] = packed_components(self.nodes, self.weights, codes, gradients=self.gradients)
        return self._cache[key]

    def local(self, points):
        return points if self.task is None else self.task.points_to_local(points)


# ---------------------------------------------------------------- integrity layer

def integrity_report(rec):
    """Recording/history validity; never part of the goal outcome."""
    checks, metrics = {}, {}
    initial_count, _ = rec.components(0)
    checks["initial_codes_zero"] = not rec.codes[0].any()
    checks["initial_continuous_graph"] = initial_count == 1
    digits = [decode_digit(rec.codes, k, rec.encoding) for k in range(rec.cuts)]
    rewrites = sum(int(np.count_nonzero((d[:-1] != 0) & (d[1:] != d[:-1]))) for d in digits)
    checks["persistent_history_no_digit_rewrites"] = rewrites == 0
    checks["valid_digit_values"] = all(int(d.max(initial=0)) <= 2 for d in digits)
    metrics.update(initial_components=int(initial_count), nonzero_digit_rewrites=rewrites, code_encoding=rec.encoding,
                   cuts=rec.cuts, frames=int(len(rec.time)), particles=int(len(rec.masses)),
                   recorded_mass_kg=float(rec.masses.sum()))
    task = rec.task
    if task is not None:
        expected, expected_mass, _ = task.sample_particles()
        actual = task.points_to_local(rec.reference)
        checks["reference_matches_scene_sampler"] = bool(expected.shape == actual.shape
                                                         and np.allclose(expected, actual, atol=1e-7, rtol=0))
        common = task.protocol_document()["common"]
        error = abs(float(rec.masses.sum())-float(expected_mass.sum()))
        checks["reference_mass_budget"] = error <= max(common["mass_absolute_tolerance_kg"],
                                                       common["mass_relative_tolerance"]*float(expected_mass.sum()))
        checks["complete_episode"] = bool(rec.time[-1]-rec.time[0] >= task.duration_s-1e-9)
        metrics["reference_mass_absolute_error_kg"] = error
    checks = {key: bool(value) for key, value in checks.items()}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "metrics": metrics,
            "scope": "Recording and persistent-history validity only; not the goal outcome"}


# ---------------------------------------------------------------- location statistics

def interface_by_plane(nodes, weights, gradients, labels, reference_local, planes):
    """Frozen ``_interface_location`` midpoint statistic, per declared plane.

    Every cross-component reference node is assigned to the plane nearest to
    the mean reference position of its particles; for one plane this is
    exactly the frozen statistic (``tasks._interface_location``).
    """
    labels, reference = np.asarray(labels), np.asarray(reference_local, float)
    signed = np.stack([(reference-o)@n for o, n in planes], 1)
    out = [{"cross_component_reference_node_count": 0, "severed_particle_pair_node_incidences": 0, "maximum": 0.}
           for _ in planes]
    for group in _node_groups(nodes, weights, gradients):
        occupied = np.unique(labels[group])
        if len(occupied) < 2:
            continue
        k = int(np.argmin(np.abs(signed[group].mean(0))))
        row, s = out[k], signed[group, k]
        row["cross_component_reference_node_count"] += 1
        for i, first in enumerate(occupied):
            a = s[labels[group] == first]
            for second in occupied[i+1:]:
                b = s[labels[group] == second]
                row["maximum"] = max(row["maximum"], abs((a.min()+b.min())/2), abs((a.max()+b.max())/2))
                row["severed_particle_pair_node_incidences"] += len(a)*len(b)
    return [{"cross_component_reference_node_count": row["cross_component_reference_node_count"],
             "severed_particle_pair_node_incidences": row["severed_particle_pair_node_incidences"],
             "maximum_reference_midpoint_plane_distance_m": (float(row["maximum"])
                 if row["cross_component_reference_node_count"] else None)} for row in out]


def _grid_aligned(normal):
    return bool(np.isclose(np.abs(np.asarray(normal, float)).max(), 1., atol=1e-9, rtol=0))


def _sever_location(rec, labels, fractions_order, planes, bound):
    reference = rec.local(rec.reference)
    actual = interface_by_plane(rec.nodes, rec.weights, rec.gradients, labels, reference, planes)
    signed = np.stack([(reference-o)@n for o, n in planes], 1)
    region = ((signed > 0).astype(np.int64)*(2**np.arange(len(planes)))).sum(1)
    ideal = interface_by_plane(rec.nodes, rec.weights, rec.gradients, region, reference, planes)
    rows, checks = [], {}
    for k, ((origin, normal), a, i) in enumerate(zip(planes, actual, ideal)):
        statistic = a["maximum_reference_midpoint_plane_distance_m"]
        aligned = _grid_aligned(normal)
        row = {"plane": k, "origin_local_m": list(map(float, origin)), "normal_local": list(map(float, normal)),
               "grid_aligned": aligned, "actual": a, "ideal_geometric_label": i,
               "frozen_2mm_statistic_m": statistic, "frozen_bound_m": bound}
        if aligned:
            row["rule"] = "frozen midpoint statistic <= plane_error_bound_m"
            row["pass"] = statistic is not None and statistic <= bound
        else:
            ideal_m = i["maximum_reference_midpoint_plane_distance_m"]
            row["rule"] = "ADR 0006 Amendment 1: actual <= ideal-geometric-label statistic + one particle spacing"
            row["particle_spacing_m"] = rec.spacing
            row["pass"] = (rec.spacing is not None and statistic is not None and ideal_m is not None
                           and statistic <= ideal_m+rec.spacing+1e-12)
        rows.append(row)
    checks["every_plane_has_an_interface"] = all(a["cross_component_reference_node_count"] > 0 for a in actual)
    checks["cut_at_declared_planes"] = all(row["pass"] for row in rows)
    if not all(row["grid_aligned"] for row in rows):
        far = (np.abs(signed) > 2*rec.spacing).all(1) if rec.spacing else np.zeros(len(labels), bool)
        wrong = 0
        for component in fractions_order:
            member = labels == component
            if not member.any():
                continue
            majority = np.bincount(region[member], weights=rec.masses[member]).argmax()
            wrong += int(np.count_nonzero(member & far & (region != majority)))
        checks["no_far_particle_in_another_region_piece"] = rec.spacing is not None and wrong == 0
        rows.append({"far_particles_in_another_region_piece": wrong})
    return checks, rows


def _piece_fractions(rec, labels, count):
    mass = np.bincount(labels, weights=rec.masses, minlength=count)
    order = np.argsort(mass)[::-1]
    return mass[order]/rec.masses.sum(), order


# ---------------------------------------------------------------- outcome evaluators

def _sever_outcome(goal, rec, scene, components=None):
    count, labels = rec.components(-1) if components is None else components
    fractions, order = _piece_fractions(rec, labels, count)
    pieces = goal["expected_pieces"] if goal["type"] == "sever" else goal["pieces"]
    substantial = int(np.count_nonzero(fractions >= goal["minimum_part_mass_fraction"]))
    checks = {"expected_substantial_pieces": substantial == pieces,
              "main_pieces_retain_mass": float(fractions[:pieces].sum()) >= goal["minimum_main_parts_mass_fraction"]}
    planes = goal_planes(goal, scene)
    location_checks, location = _sever_location(rec, labels, order[:pieces], planes, goal["plane_error_bound_m"])
    metrics = {"mechanical_components": int(count), "piece_mass_fractions": fractions[:max(pieces+2, 4)].tolist(),
               "substantial_pieces": substantial, "main_pieces_mass_fraction": float(fractions[:pieces].sum()),
               "minor_pieces_mass_fraction": float(fractions[pieces:].sum())}
    quality = {"plane_location": location}
    if goal["type"] == "sever":
        checks.update(location_checks)
    else:
        quality["location_checks"] = location_checks
        if pieces > 1:
            quality["piece_mass_fraction_spread"] = float(fractions[:pieces].max()-fractions[:pieces].min())
    return checks, metrics, quality, []


def protrusion_ownership(task, reference_local, plane_z):
    """Per particle: index of the protrusion it initially belongs to (-1 none), from the scene heightfield.

    A particle belongs to protrusion i when its centre lies above the plane and
    bump i has the largest height contribution at its (x, y) (max-union rule).
    """
    bumps = task.geometry.bumps
    reference_local = np.asarray(reference_local, float)
    owner = np.full(len(reference_local), -1, np.int64)
    if not bumps:
        return owner
    heights = np.stack([bump.height(reference_local[:, 0], reference_local[:, 1]) for bump in bumps], 1)
    above = reference_local[:, 2] > plane_z+1e-7
    best = heights.argmax(1)
    inside = heights.max(1) > 0
    owner[above & inside] = best[above & inside]
    return owner


def _substrate(rec, labels, plane_z, region_mask=None):
    reference = rec.local(rec.reference)
    anchors = reference[:, 2] <= reference[:, 2].min()+1e-7
    substrate_ids = np.unique(labels[anchors])
    substrate = np.isin(labels, substrate_ids)
    below = reference[:, 2] <= plane_z+1e-7
    below_mass = float(rec.masses[below].sum())
    per_component = np.bincount(labels[below], weights=rec.masses[below], minlength=labels.max()+1)
    share = None if below_mass <= 0 else float(per_component.max()/below_mass)
    detached_below = None if below_mass <= 0 else float(rec.masses[below & ~substrate].sum()/below_mass)
    return reference, substrate, substrate_ids, share, detached_below


def _residual(rec, substrate, mask, plane_z):
    selected = substrate & mask
    if not selected.any():
        return None
    final = rec.local(rec.positions[-1])
    return float(np.maximum(final[selected, 2]-plane_z, 0).max())


def _surface_quality(rec, task, plane_z, substrate_ids):
    fields = ("surface_vertices_world", "surface_triangles", "surface_triangle_component_ids")
    if task is None or any(name not in rec.data for name in fields):
        return {"status": "NOT_RUN", "missing": [name for name in fields if name not in rec.data] or ["task"]}
    from .scene_frames import WorkpieceTarget
    from .scene_validation import mesh_heightfield_metrics
    triangles = np.asarray(rec.data["surface_triangles"])
    faces = np.asarray(rec.data["surface_triangle_component_ids"])
    selected = np.isin(faces, substrate_ids)
    if not selected.any():
        return {"status": "NO_SUBSTRATE_SURFACE"}
    target = WorkpieceTarget(task.world_from_workpiece_xyzw, plane_z, task.edge_clearance_m)
    metrics = mesh_heightfield_metrics(rec.data["surface_vertices_world"], triangles[selected], target, task.geometry, .001)
    return {"status": "MEASURED", **{k: v for k, v in metrics.items() if not k.endswith("_debug")},
            "scope": "Frozen TRIM surface metrics, reported as quality only"}


def selected_protrusions(goal, task):
    ids = task.protrusion_ids
    select = goal["select"]
    if select == "all":
        return list(range(len(ids)))
    if isinstance(select, list):
        unknown = [s for s in select if s not in ids]
        if unknown:
            raise _error(f"remove_protrusions selects unknown protrusions {unknown}; scene has {list(ids)}")
        return [ids.index(s) for s in select]
    region = select["region_local_m"]
    chosen = []
    for i, bump in enumerate(task.geometry.bumps):
        x_ok = "x" not in region or region["x"][0] <= bump.x_m <= region["x"][1]
        y_ok = "y" not in region or region["y"][0] <= bump.y_m <= region["y"][1]
        if x_ok and y_ok:
            chosen.append(i)
    return chosen


def _remove_outcome(goal, rec, scene):
    task = rec.task
    if task is None or not hasattr(task, "protrusion_ids"):
        return {}, {}, {}, ["A configured task (scene protrusions and pose) is required for remove_protrusions"]
    plane_z = scene["workpiece"]["base_box_m"]["height_m"] if goal["plane_local_z_m"] is None else goal["plane_local_z_m"]
    count, labels = rec.components(-1)
    reference, substrate, substrate_ids, share, detached_below = _substrate(rec, labels, plane_z)
    owner = protrusion_ownership(task, reference, plane_z)
    selected = selected_protrusions(goal, task)
    items, checks = [], {}
    for i, pid in enumerate(task.protrusion_ids):
        member = owner == i
        mass = float(rec.masses[member].sum())
        removed = None if mass <= 0 else float(rec.masses[member & ~substrate].sum()/mass)
        chosen = i in selected
        if chosen:
            passed = removed is not None and removed >= goal["minimum_removed_fraction"]
        elif goal["minimum_unselected_retained_fraction"] is None:
            passed = True
        else:
            passed = removed is not None and 1-removed >= goal["minimum_unselected_retained_fraction"]
        items.append({"id": pid, "selected": chosen, "reference_particles_above_plane": int(member.sum()),
                      "reference_mass_above_plane_kg": mass, "removed_mass_fraction": removed, "pass": bool(passed),
                      "residual_substrate_height_m": _residual(rec, substrate, member | _footprint(task, i, reference), plane_z)})
    checks["selection_nonempty"] = bool(selected)
    checks["selected_protrusions_removed"] = bool(selected) and all(row["pass"] for row in items if row["selected"])
    if goal["minimum_unselected_retained_fraction"] is not None:
        checks["unselected_protrusions_retained"] = all(row["pass"] for row in items if not row["selected"])
    checks["substrate_holds_below_plane_mass"] = share is not None and share >= goal["minimum_substrate_share"]
    checks["detached_below_plane_mass_bounded"] = detached_below is not None and detached_below <= goal["maximum_detached_below_fraction"]
    checks["single_substrate_component"] = len(substrate_ids) == 1
    metrics = {"mechanical_components": int(count), "plane_local_z_m": float(plane_z),
               "largest_component_below_plane_mass_share": share, "detached_below_plane_mass_fraction": detached_below,
               "reference_substrate_component_ids": substrate_ids.tolist(), "items": items,
               "ownership_rule": "particle centre above the plane; bump with the largest height contribution at (x, y)"}
    quality = {"residual_substrate_height_m": {row["id"]: row["residual_substrate_height_m"] for row in items},
               "surface": _surface_quality(rec, task, plane_z, substrate_ids)}
    return checks, metrics, quality, []


def _footprint(task, index, reference):
    bump = task.geometry.bumps[index]
    return bump.height(reference[:, 0], reference[:, 1]) > 0


def _region_mask(points_local, region):
    mask = np.ones(len(points_local), bool)
    for axis, column in (("x", 0), ("y", 1)):
        if region and axis in region:
            mask &= (points_local[:, column] >= region[axis][0]) & (points_local[:, column] <= region[axis][1])
    return mask


def _trim_outcome(goal, rec, scene):
    origin, normal = np.asarray(goal["plane"]["origin_local_m"]), np.asarray(goal["plane"]["normal_local"])
    count, labels = rec.components(-1)
    reference = rec.local(rec.reference)
    signed = (reference-origin)@normal
    region = _region_mask(reference, goal["region_local_m"])
    anchors = reference[:, 2] <= reference[:, 2].min()+1e-7
    substrate_ids = np.unique(labels[anchors])
    substrate = np.isin(labels, substrate_ids)
    above, below = (signed > 1e-7) & region, signed <= 1e-7
    above_mass, below_mass = float(rec.masses[above].sum()), float(rec.masses[below].sum())
    removed = None if above_mass <= 0 else float(rec.masses[above & ~substrate].sum()/above_mass)
    per_component = np.bincount(labels[below], weights=rec.masses[below], minlength=count)
    share = None if below_mass <= 0 else float(per_component.max()/below_mass)
    detached_below = None if below_mass <= 0 else float(rec.masses[below & ~substrate].sum()/below_mass)
    checks = {"material_above_plane_present": above_mass > 0,
              "material_above_plane_removed": removed is not None and removed >= goal["minimum_removed_fraction"],
              "substrate_holds_below_plane_mass": share is not None and share >= goal["minimum_substrate_share"],
              "detached_below_plane_mass_bounded": detached_below is not None and detached_below <= goal["maximum_detached_below_fraction"],
              "single_substrate_component": len(substrate_ids) == 1}
    final = rec.local(rec.positions[-1])
    residual = (float(np.maximum((final[substrate & region]-origin)@normal, 0).max()) if (substrate & region).any() else None)
    metrics = {"mechanical_components": int(count), "removed_above_plane_mass_fraction": removed,
               "reference_above_plane_mass_kg": above_mass, "largest_component_below_plane_mass_share": share,
               "detached_below_plane_mass_fraction": detached_below, "reference_substrate_component_ids": substrate_ids.tolist()}
    plane_z = float(origin[2]) if _grid_aligned(normal) and abs(normal[2]) > .5 else None
    quality = {"current_substrate_particle_center_max_residual_m": residual,
               "surface": (_surface_quality(rec, rec.task, plane_z, substrate_ids) if plane_z is not None
                           else {"status": "NOT_RUN", "reason": "surface metrics are defined for horizontal planes"})}
    return checks, metrics, quality, []


def _window(goal, rec, windows, step_index):
    """(start, end, source) of a move window."""
    key = goal["goal_id"]
    for source in (windows, rec.data.get("phase_intervals_s")):
        if source and key in source:
            return float(source[key][0]), float(source[key][1]), "declared_window"
    intervals = rec.task.phase_intervals_s() if rec.task is not None and hasattr(rec.task, "phase_intervals_s") else {}
    labels = [label for label in intervals if label.split(":", 1)[-1] == f"move{step_index}" and label.startswith(("PUSH",))]
    if labels:
        start = min(intervals[label][0] for label in labels)
        end = max(intervals[label][1] for label in labels)
        return start, end, "task_reference_schedule"
    return float(rec.time[0]), float(rec.time[-1]), "whole_recording"


def selector_mask(goal, reference_local):
    selector = goal["selector"]
    if "side_of_plane" in selector:
        row = selector["side_of_plane"]
        signed = (reference_local-np.asarray(row["origin_local_m"]))@np.asarray(row["normal_local"])
        return signed > 0 if row["side"] == "positive" else signed < 0
    box = selector["box_local_m"]
    return np.all((reference_local >= np.asarray(box["min"])) & (reference_local <= np.asarray(box["max"])), axis=1)


def _move_outcome(goal, rec, scene, windows=None, step_index=0):
    start_s, end_s, source = _window(goal, rec, windows, step_index)
    before = np.flatnonzero(rec.time <= start_s+1e-9)
    after = np.flatnonzero(rec.time >= end_s-1e-9)
    if not len(before) or not len(after) or after[0] <= before[-1]:
        return {}, {}, {}, ["Recorded frames at or before the move start and at or after its end"]
    start, end = int(before[-1]), int(after[0])
    count, labels = rec.components(start)
    reference = rec.local(rec.reference)
    selector = selector_mask(goal, reference)
    total = rec.masses.sum()
    component_mass = np.bincount(labels, weights=rec.masses, minlength=count)

    def centroid(frame):
        return np.stack([np.bincount(labels, weights=rec.masses*rec.positions[frame, :, axis], minlength=count)
                         for axis in range(3)], 1)/component_mass[:, None]

    displacement = centroid(end)-centroid(start)
    direction = (rec.task.T_world_workpiece[:3, :3] if rec.task is not None else np.eye(3))@np.asarray(goal["direction_local"])
    overlap = np.bincount(labels[selector], weights=rec.masses[selector], minlength=count)
    moved = int(np.argmax(overlap))
    share = float(overlap[moved]/rec.masses[selector].sum()) if selector.any() else 0.
    pieces = np.flatnonzero(component_mass/total >= goal["minimum_piece_mass_fraction"])
    others = [int(c) for c in pieces if c != moved]
    other_motion = [float(np.linalg.norm(displacement[c])) for c in others]
    window = slice(start, end+1)
    changed = np.any(rec.codes[window] != rec.codes[start], axis=1)
    if "cut_enabled" in rec.data:
        disarmed = ~np.asarray(rec.data["cut_enabled"], bool)[window]
        changes_disarmed = int(np.count_nonzero(changed & disarmed))
    else:
        changes_disarmed = int(np.count_nonzero(changed))
    end_count, end_labels = rec.components(end)
    pairs = np.unique(np.stack([end_labels, labels], 1), axis=0)
    merged = len(np.unique(pairs[:, 0])) < len(pairs)
    along = float(displacement[moved]@direction)
    checks = {"piece_exists_at_move_start": count >= 2,
              "selected_piece_identified": share >= goal["minimum_selector_share"]
                  and component_mass[moved]/total >= goal["minimum_piece_mass_fraction"],
              "selected_piece_displaced": along >= goal["minimum_displacement_m"],
              "other_pieces_stationary": all(d <= goal["maximum_other_displacement_m"] for d in other_motion),
              "no_label_change_while_disarmed": changes_disarmed == 0,
              "components_never_merge": not merged}
    metrics = {"window_s": [float(rec.time[start]), float(rec.time[end])], "window_source": source,
               "components_at_start": int(count), "components_at_end": int(end_count),
               "moved_component": moved, "selector_mass_share": share,
               "displacement_along_direction_m": along, "displacement_m": displacement[moved].tolist(),
               "other_piece_displacement_norm_m": other_motion,
               "label_changes_while_disarmed_frames": changes_disarmed,
               "label_rule": "cut_enabled per frame when recorded, else any code change in the window"}
    return checks, metrics, {"component_mass_fractions": (component_mass/total).tolist()}, []


_OUTCOMES = {"sever": _sever_outcome, "divide": _sever_outcome, "remove_protrusions": _remove_outcome,
             "trim_plane": _trim_outcome, "move_piece": _move_outcome}


def _outcome(goal, rec, scene, windows=None, step_index=0):
    function = _OUTCOMES[goal["type"]]
    if goal["type"] == "move_piece":
        checks, metrics, quality, missing = function(goal, rec, scene, windows, step_index)
    else:
        checks, metrics, quality, missing = function(goal, rec, scene)
    checks = {key: bool(value) for key, value in checks.items()}
    status = "NOT_RUN" if missing else "PASS" if checks and all(checks.values()) else "FAIL"
    return {"status": status, "checks": checks, "metrics": metrics, "missing_evidence": missing}, quality


# ---------------------------------------------------------------- goal_eval_v2 (ADR 0015 Amendment 1)

V2_SCOPE = ("goal_eval_v2: each sequence step is scored on the mechanical components of the cut digits it is "
            "responsible for; sever/divide on its own digits; remove_protrusions/trim_plane on the earlier sever/divide "
            "digits plus its own, with substrate integrity per substrate piece that existed before the step; "
            "move_piece as v1; every threshold is the v1 threshold")


def mask_digits(codes, digits, encoding="ternary"):
    """Codes with every digit outside ``digits`` set to 0 (unassigned): the history of those digits alone."""
    codes = np.asarray(codes, np.int64)
    out = np.zeros_like(codes)
    for k in sorted(set(int(d) for d in digits)):
        d = decode_digit(codes, k, encoding)
        out += d*3**k if encoding == "ternary" else d << (2*k)
    return out


def _scene_selected(goal, scene):
    """Indices of the scene protrusions a remove_protrusions goal selects (scene config only)."""
    rows = scene["workpiece"]["protrusions"]
    ids = [row["id"] for row in rows]
    select = goal["select"]
    if select == "all":
        return list(range(len(rows)))
    if isinstance(select, list):
        unknown = [s for s in select if s not in ids]
        if unknown:
            raise _error(f"remove_protrusions selects unknown protrusions {unknown}; scene has {ids}")
        return [ids.index(s) for s in select]
    region = select["region_local_m"]
    return [i for i, row in enumerate(rows)
            if ("x" not in region or region["x"][0] <= row["x_m"] <= region["x"][1])
            and ("y" not in region or region["y"][0] <= row["y_m"] <= region["y"][1])]


def step_digit_allocation(goal, scene):
    """[[digits] per step] of a sequence by the declared convention (steps in order, digits numbered across the
    episode; sever: one per plane, divide: pieces - 1, remove_protrusions: one per selected protrusion,
    trim_plane: one, move_piece: none). This is the goal_planner numbering (``goal_planner._operations``)."""
    rows, digit = [], 0
    for step in goal["steps"]:
        kind = step["type"]
        count = (len(step["planes"]) if kind == "sever" else step["pieces"]-1 if kind == "divide"
                 else len(_scene_selected(step, scene)) if kind == "remove_protrusions" else 1 if kind == "trim_plane" else 0)
        rows.append(list(range(digit, digit+count)))
        digit += count
    return rows


def _plan_digit_allocation(goal, scene, task):
    """[[digits] per step] from the goal planner's report for the compiled task, or None when unavailable."""
    controller = getattr(task, "controller", None) if task is not None else None
    if not controller or "planner" not in controller:
        return None
    from .goal_planner import plan_with_report
    from .tool_profile import PROFILES
    _, report = plan_with_report(scene, goal, PROFILES[scene["tool"]["profile_id"]], controller["planner"])
    rows = [[] for _ in goal["steps"]]
    for op in report["operations"]:
        if op.get("digit") is not None:
            rows[int(op["step"])].append(int(op["digit"]))
    return [sorted(row) for row in rows]


def _scoped_substrate(rec, pre, post, plane_z, goal):
    """Per prior piece (anchored component of ``pre``): below-plane share/detachment/anchored count under ``post``."""
    reference = rec.local(rec.reference)
    anchors = reference[:, 2] <= reference[:, 2].min()+1e-7
    (_, pre_labels), (post_count, post_labels) = pre, post
    substrate_ids = np.unique(post_labels[anchors])
    substrate = np.isin(post_labels, substrate_ids)
    below = reference[:, 2] <= plane_z+1e-7
    pieces = []
    for piece in np.unique(pre_labels[anchors]):
        member = pre_labels == piece
        part = member & below
        mass = float(rec.masses[part].sum())
        per = np.bincount(post_labels[part], weights=rec.masses[part], minlength=post_count)
        share = None if mass <= 0 else float(per.max()/mass)
        detached = None if mass <= 0 else float(rec.masses[part & ~substrate].sum()/mass)
        anchored = np.unique(post_labels[anchors & member])
        pieces.append({"prior_piece": int(piece), "mass_fraction": float(rec.masses[member].sum()/rec.masses.sum()),
                       "below_plane_mass_kg": mass, "largest_component_below_plane_mass_share": share,
                       "detached_below_plane_mass_fraction": detached, "substrate_components": int(len(anchored)),
                       "pass": bool(share is not None and share >= goal["minimum_substrate_share"]
                                    and detached is not None and detached <= goal["maximum_detached_below_fraction"]
                                    and len(anchored) == 1)})
    checks = {"prior_pieces_present": bool(pieces),
              "substrate_holds_below_plane_mass_per_prior_piece":
                  bool(pieces) and all(r["largest_component_below_plane_mass_share"] is not None
                                       and r["largest_component_below_plane_mass_share"] >= goal["minimum_substrate_share"]
                                       for r in pieces),
              "detached_below_plane_mass_bounded_per_prior_piece":
                  bool(pieces) and all(r["detached_below_plane_mass_fraction"] is not None
                                       and r["detached_below_plane_mass_fraction"] <= goal["maximum_detached_below_fraction"]
                                       for r in pieces),
              "single_substrate_component_per_prior_piece": bool(pieces) and all(r["substrate_components"] == 1 for r in pieces)}
    return reference, substrate, substrate_ids, checks, pieces


def _remove_outcome_v2(goal, rec, scene, pre, post):
    task = rec.task
    if task is None or not hasattr(task, "protrusion_ids"):
        return {}, {}, {}, ["A configured task (scene protrusions and pose) is required for remove_protrusions"]
    plane_z = scene["workpiece"]["base_box_m"]["height_m"] if goal["plane_local_z_m"] is None else goal["plane_local_z_m"]
    reference, substrate, substrate_ids, substrate_checks, pieces = _scoped_substrate(rec, pre, post, plane_z, goal)
    owner = protrusion_ownership(task, reference, plane_z)
    selected = selected_protrusions(goal, task)
    items, checks = [], {}
    for i, pid in enumerate(task.protrusion_ids):
        member = owner == i
        mass = float(rec.masses[member].sum())
        removed = None if mass <= 0 else float(rec.masses[member & ~substrate].sum()/mass)
        chosen = i in selected
        if chosen:
            passed = removed is not None and removed >= goal["minimum_removed_fraction"]
        elif goal["minimum_unselected_retained_fraction"] is None:
            passed = True
        else:
            passed = removed is not None and 1-removed >= goal["minimum_unselected_retained_fraction"]
        items.append({"id": pid, "selected": chosen, "reference_particles_above_plane": int(member.sum()),
                      "reference_mass_above_plane_kg": mass, "removed_mass_fraction": removed, "pass": bool(passed),
                      "residual_substrate_height_m": _residual(rec, substrate, member | _footprint(task, i, reference), plane_z)})
    checks["selection_nonempty"] = bool(selected)
    checks["selected_protrusions_removed"] = bool(selected) and all(row["pass"] for row in items if row["selected"])
    if goal["minimum_unselected_retained_fraction"] is not None:
        checks["unselected_protrusions_retained"] = all(row["pass"] for row in items if not row["selected"])
    checks.update(substrate_checks)
    metrics = {"mechanical_components": int(post[0]), "plane_local_z_m": float(plane_z), "prior_pieces": pieces,
               "reference_substrate_component_ids": substrate_ids.tolist(), "items": items,
               "ownership_rule": "particle centre above the plane; bump with the largest height contribution at (x, y)"}
    quality = {"residual_substrate_height_m": {row["id"]: row["residual_substrate_height_m"] for row in items},
               "surface": _surface_quality(rec, task, plane_z, substrate_ids)}
    return checks, metrics, quality, []


def _trim_outcome_v2(goal, rec, scene, pre, post):
    origin, normal = np.asarray(goal["plane"]["origin_local_m"]), np.asarray(goal["plane"]["normal_local"])
    if not (_grid_aligned(normal) and abs(normal[2]) > .5):
        return {}, {}, {}, ["goal_eval_v2 scopes trim_plane steps with a horizontal plane only"]
    plane_z = float(origin[2])
    reference, substrate, substrate_ids, substrate_checks, pieces = _scoped_substrate(rec, pre, post, plane_z, goal)
    signed = (reference-origin)@normal
    region = _region_mask(reference, goal["region_local_m"])
    above = (signed > 1e-7) & region
    above_mass = float(rec.masses[above].sum())
    removed = None if above_mass <= 0 else float(rec.masses[above & ~substrate].sum()/above_mass)
    checks = {"material_above_plane_present": above_mass > 0,
              "material_above_plane_removed": removed is not None and removed >= goal["minimum_removed_fraction"],
              **substrate_checks}
    final = rec.local(rec.positions[-1])
    residual = (float(np.maximum((final[substrate & region]-origin)@normal, 0).max()) if (substrate & region).any() else None)
    metrics = {"mechanical_components": int(post[0]), "removed_above_plane_mass_fraction": removed,
               "reference_above_plane_mass_kg": above_mass, "prior_pieces": pieces,
               "reference_substrate_component_ids": substrate_ids.tolist()}
    quality = {"current_substrate_particle_center_max_residual_m": residual,
               "surface": _surface_quality(rec, rec.task, plane_z, substrate_ids)}
    return checks, metrics, quality, []


def _sequence_v2(goal, rec, scene, windows, task):
    declared = step_digit_allocation(goal, scene)
    planned = _plan_digit_allocation(goal, scene, task)
    allocation = {"rule": "declared convention (steps in order; sever one per plane, divide pieces - 1, "
                          "remove_protrusions one per selected protrusion, trim_plane one, move_piece none)",
                  "declared": declared, "plan_report": planned,
                  "plan_cross_check": "NOT_AVAILABLE" if planned is None else "MATCH" if planned == declared else "MISMATCH"}
    needed = max([d for row in declared for d in row], default=-1)+1
    problem = None
    if allocation["plan_cross_check"] == "MISMATCH":
        problem = "The declared step digit allocation differs from the planner report"
    elif needed > rec.cuts:
        problem = f"The sequence needs {needed} cut digits; the recording keeps {rec.cuts}"
    final = rec.codes[-1]
    steps, quality, prior = [], {}, []
    for index, step in enumerate(goal["steps"]):
        own = declared[index]
        scope = {"own_digits": own, "prior_sever_digits": list(prior)}
        if problem is not None:
            outcome, step_quality = {"status": "NOT_RUN", "checks": {}, "metrics": {}, "missing_evidence": [problem]}, {}
        elif step["type"] in ("sever", "divide"):
            components = rec.components_of(mask_digits(final, own, rec.encoding))
            outcome, step_quality = _finish(*_sever_outcome(step, rec, scene, components))
            scope["components_from"] = "own digits"
        elif step["type"] in ("remove_protrusions", "trim_plane"):
            pre = rec.components_of(mask_digits(final, prior, rec.encoding))
            post = rec.components_of(mask_digits(final, prior+own, rec.encoding))
            function = _remove_outcome_v2 if step["type"] == "remove_protrusions" else _trim_outcome_v2
            outcome, step_quality = _finish(*function(step, rec, scene, pre, post))
            scope["components_from"] = "prior sever digits (pieces before the step) and prior sever + own digits (after)"
        else:
            outcome, step_quality = _outcome(step, rec, scene, windows, index)
            scope["components_from"] = "goal_eval_v1 (move window state; unchanged)"
        outcome["digit_scope"] = scope
        steps.append({"index": index, "goal_id": step["goal_id"], "type": step["type"], "outcome": outcome})
        quality[f"{index}:{step['goal_id']}"] = step_quality
        if step["type"] in ("sever", "divide"):
            prior += own
    statuses = [row["outcome"]["status"] for row in steps]
    status = "NOT_RUN" if "NOT_RUN" in statuses else "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
    outcome = {"status": status, "checks": {f"step_{row['index']}_{row['goal_id']}": row["outcome"]["status"] == "PASS"
                                           for row in steps},
               "steps": steps, "rule": "every step must hold at the end of the episode", "evaluator_version": "goal_eval_v2",
               "scope": V2_SCOPE, "digit_allocation": allocation}
    return outcome, quality


def _finish(checks, metrics, quality, missing):
    checks = {key: bool(value) for key, value in checks.items()}
    status = "NOT_RUN" if missing else "PASS" if checks and all(checks.values()) else "FAIL"
    return {"status": status, "checks": checks, "metrics": metrics, "missing_evidence": missing}, quality


def evaluate_goal(goal, recording, *, task=None, scene=None, encoding=None, cuts=None, windows=None):
    """Score a recorded WORLD episode against a goal (read-only; three layers).

    ``task`` (a ConfiguredTaskSpec) supplies the scene, frames, protrusion ids
    and reference schedule; ``scene`` may be given instead for frame-free plane
    goals with an identity pose. ``windows`` maps move-step goal ids to
    [start_s, end_s] (else ``recording["phase_intervals_s"]``, else the task's
    planned schedule, else the whole recording; the source is reported).
    """
    goal = validate_goal(goal)
    if scene is None:
        if task is None or not hasattr(task, "scene"):
            raise ValueError("evaluate_goal needs a ConfiguredTaskSpec (task=) or a scene")
        scene = task.scene
    else:
        from .scene_config import resolve_scene
        scene = resolve_scene(scene)
    base = {"schema": EVALUATION_SCHEMA, "goal_id": goal["goal_id"], "goal_type": goal["type"],
            "task_id": getattr(task, "task_id", None), "scene_id": scene["scene_id"],
            "policy_observation": False, "full_simulation_acceptance": False,
            "scope": "Read-only goal evaluation of recorded oracle material state; separate from the frozen v1 oracle"}
    rec = _Recording(recording, task, encoding, cuts)
    if rec.missing:
        return {**base, "verdict": "NOT_RUN", "missing_evidence": rec.missing, "integrity": {"status": "NOT_RUN"},
                "outcome": {"status": "NOT_RUN"}, "quality": {}}
    integrity = integrity_report(rec)
    version = goal.get("evaluator_version", "goal_eval_v1")
    if goal["type"] == "sequence" and version == "goal_eval_v2":
        outcome, quality = _sequence_v2(goal, rec, scene, windows, task)
    elif goal["type"] == "sequence":
        steps, quality = [], {}
        for index, step in enumerate(goal["steps"]):
            outcome, step_quality = _outcome(step, rec, scene, windows, index)
            steps.append({"index": index, "goal_id": step["goal_id"], "type": step["type"], "outcome": outcome})
            quality[f"{index}:{step['goal_id']}"] = step_quality
        statuses = [row["outcome"]["status"] for row in steps]
        status = "NOT_RUN" if "NOT_RUN" in statuses else "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
        outcome = {"status": status, "checks": {f"step_{row['index']}_{row['goal_id']}": row["outcome"]["status"] == "PASS"
                                               for row in steps},
                   "steps": steps, "rule": "every step must hold at the end of the episode"}
    else:
        outcome, quality = _outcome(goal, rec, scene, windows)
    if integrity["status"] != "PASS":
        verdict = "FAIL_INTEGRITY"
    else:
        verdict = outcome["status"]
    result = {**base, "verdict": verdict, "success": verdict == "PASS", "integrity": integrity, "outcome": outcome,
              "quality": {"status": "REPORTED_NOT_SCORED", **quality},
              "code_encoding": rec.encoding, "cuts": rec.cuts, "provenance": PROVENANCE}
    if "evaluator_version" in goal:  # absent for goals that do not declare it (v1 output unchanged)
        result["evaluator_version"] = version
    return result
