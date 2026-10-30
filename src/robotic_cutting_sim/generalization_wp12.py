"""WP-12 Phase 5 generalization benchmark: config-driven Scene + Goal generators, feasibility, summaries.

Extends the WP-11 Phase E machinery (``generalization.py``: build/compile/staged preflight/episode metrics) to the
``blade_master_flat_board_v1`` workcell; nothing in WP-11 changes. A benchmark file (schema
``rcs_wp12_benchmark_v1``, e.g. ``configs/wp12/benchmark_v1_DRAFT.json``) declares, as DATA:

* ``dimensions``: every randomization dimension with its group (scene / cutting / physics), kind
  (uniform / choice / integer) and optional vector ``size``;
* ``range_sets``: an ``ID`` and an ``OOD`` range per dimension (``uniform``, ``choice``, ``integer``, ``fixed``, and for
  OOD the shells ``uniform_outside`` / ``integer_outside`` around the ID interval); families may override them;
* ``families`` A-H (spec 5.1): builder, variants, randomized dimensions, OOD dimensions, seed base and salt;
* ``seeds``: split offsets (benchmark / held_out / dev), range-set offsets and counts, expanded per family;
* the feasibility precedence, the Native sub-sampling rule, metrics, targets and the execution budget.

Randomization (spec 5.2): every dimension draws from its OWN stream
``numpy.random.default_rng([benchmark salt, family salt, seed, sha256(dimension name)[:8]])``, so dimensions are
independent and composable (adding or removing one never changes the others' values) and every sample is
reproducible from (family, range set, seed). An OOD sample takes all ID draws and replaces exactly one OOD dimension
(``ood_mode = one_at_a_time``: the family's ``ood_dimensions`` cycled by seed index, so the failure boundary can be
attributed to one dimension) or all of them (``all``) by its OOD range. OOD ranges must be disjoint from ID.

Feasibility (spec 5.3, ``classify_feasibility``) runs BEFORE execution and never drops a sample: scene geometry
(generator checks + ``scene_config`` validation and compile) -> supported envelope
(``configs/wp12/supported_envelope_v1.json``; null bounds are NOT_EVALUATED and mark the sample envelope_pending) ->
motion preflight (IK -> ``ik_unreachable``; collision/clearance -> ``collision_constrained``). The primary class is the
first failing class in the declared precedence; every check result is kept. After execution a valid_feasible sample
becomes task_success / execution_failure / not_run (``classify_execution``). Summaries (``summarize``) report the
feasibility rate, success among feasible, overall success and the class distribution with Wilson 95 % intervals and
explicit denominators, per stratum, family, range set and overall, and assert that no sample is lost.

The generators read the scene CONFIG only (never particles or oracle state); the solver has no task-specific branch:
every family is a scene + an ``rcs_goal_v1`` goal of the existing types (sever, divide, remove_protrusions, trim_plane,
move_piece, sequence). Patterns that need a missing goal feature raise ``NotExpressible`` with the proposal
(``MIXED_OPERATIONS``, ``MISSING_FEATURES``).
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import generalization as G

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "rcs_wp12_benchmark_v1"
ENVELOPE_SCHEMA = "rcs_wp12_supported_envelope_v1"
DRAFT = ROOT/"configs/wp12/benchmark_v1_DRAFT.json"
ENVELOPE = ROOT/"configs/wp12/supported_envelope_v1.json"
GROUPS = ("scene", "cutting", "physics")
KINDS = ("uniform", "choice", "integer")
SPEC_KEYS = ("uniform", "choice", "integer", "fixed", "uniform_outside", "integer_outside")
SPLITS = ("benchmark", "held_out", "dev")
RANGE_SETS = ("ID", "OOD")
FEASIBILITY_CLASSES = ("valid_feasible", "scene_geometrically_infeasible", "ik_unreachable", "collision_constrained",
                       "physically_unsupported")
INFEASIBLE_CLASSES = FEASIBILITY_CLASSES[1:]
EXECUTION_CLASSES = ("task_success", "execution_failure", "not_run")
REPORT_CLASSES = INFEASIBLE_CLASSES+EXECUTION_CLASSES
DEFAULT_PRECEDENCE = ("scene_geometrically_infeasible", "physically_unsupported", "ik_unreachable", "collision_constrained")
EXECUTION_FAILURE_KINDS = ("solver_invalid", "plan_not_finished", "integrity", "outcome", "crash", "grasp", "other")
WILSON_Z = 1.959963984540054

# Spec 5.8 mixed-operation tasks -> family / variant and the existing goal types that express them.
MIXED_OPERATIONS = {
    "cut_cut": {"family": "D", "variant": "cut_cut", "goal": "sequence[sever, sever] (goal_eval_v2a step scopes)"},
    "cut_push": {"family": "G", "variant": "cut_push", "goal": "sequence[sever, move_piece kerf_face]"},
    "trim_trim": {"family": "E", "variant": "trim_trim", "goal": "sequence[trim_plane p0 region, trim_plane p1 region]"},
    "trim_clear_chip": {"family": "G", "variant": "trim_clear_chip",
                        "goal": "sequence[remove_protrusions [p0], move_piece surface_sweep box over the chip]"},
    "sever_trim": {"family": "F", "variant": "sever_then_trim", "goal": "sequence[sever, remove_protrusions all]"},
    "trim_sever": {"family": "F", "variant": "trim_then_sever", "goal": "sequence[remove_protrusions all, sever]"},
    "bumps_sequential": {"family": "E", "variant": "bumps_sequential", "goal": "sequence[remove_protrusions [p_i] ...]"},
    "cut_reposition_cut": {"family": "G", "variant": "cut_reposition_cut", "goal": None, "status": "NOT_EXPRESSIBLE",
                           "missing": "planner_offset_local_m"},
}
MISSING_FEATURES = {
    "planner_offset_local_m": (
        "cut -> reposition -> cut needs the second cut planned where the moved piece IS (declared push displacement) while "
        "the evaluator keeps the plane in reference coordinates. Today a sever plane is both the planner target and the "
        "evaluator plane. Minimal versioned addition: an opt-in sever/divide STEP field 'planner_offset_local_m' "
        "(goals._OPT_IN_FIELDS, ignored by every evaluator) that goal_planner adds to the plane origin when planning, "
        "or an opt-in planner option 'follow_declared_moves' that offsets later steps by the earlier move_piece "
        "distance_m * direction_local. Open loop; the measured relocation is ADR 0016 wrist_depth_relocate_v1 (live only)."),
    "per_step_planner_options": (
        "goal_planner options (rake_deg, skew_deg, skim_direction_local, speeds) are goal-wide, so a sequence cannot use "
        "two rakes or two skim directions (e.g. trim -> trim at different rho). Minimal addition: an opt-in step field "
        "'planner' merged over the goal's planner block for that step's operations only."),
    "continuous_physics_overrides": (
        "rcs_scene_v1 selects a material only by profile id, so friction, density, contact stiffness and cutting "
        "resistance can be varied only as the discrete materials_wp12_v1 profiles. Minimal addition: an opt-in "
        "workpiece.material_overrides {friction, density_kg_m3, fracture_energy_j_m2} validated by materials_wp12.admit "
        "(scene_config.py change; owned by the Phase 4c agent). TLMPM contact parameters stay solver config."),
    "edge_lateral_offset": (
        "The 'approach position' along the edge line (which part of the 120 mm edge meets the material) is fixed by the "
        "planner (edge centre over the plane section). Only the armed start distance (cut_clearance_m) is variable."),
}


class BenchmarkConfigError(ValueError):
    """The benchmark file is malformed (never a sample class)."""


class NotExpressible(ValueError):
    """The requested operation pattern needs a goal/planner feature that does not exist (see MISSING_FEATURES)."""


# ----------------------------------------------------------------------------- config

def load_benchmark(path=DRAFT):
    doc = json.loads(Path(path).read_text())
    validate_benchmark(doc)
    return doc


def load_envelope(path=ENVELOPE):
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != ENVELOPE_SCHEMA:
        raise BenchmarkConfigError(f"{path} is not a {ENVELOPE_SCHEMA} file")
    for rule in doc.get("rules", []):
        if rule.get("kind") not in ("max", "min", "range", "in_set", "subset") or "parameter" not in rule or "id" not in rule:
            raise BenchmarkConfigError(f"Malformed envelope rule {rule}")
    return doc


def _support(spec):
    """(kind, payload) of a range spec: ('interval', [lo, hi]) / ('intervals', [[lo, hi], ...]) / ('set', [values])."""
    key = _spec_key(spec)
    value = spec[key]
    if key == "uniform":
        return "intervals", [[float(value[0]), float(value[1])]]
    if key == "uniform_outside":
        (a, b), (c, d) = value["outer"], value["inner"]
        return "intervals", [iv for iv in ([a, c], [d, b]) if iv[1] > iv[0]]
    if key == "integer":
        return "set", list(range(int(value[0]), int(value[1])+1))
    if key == "integer_outside":
        (a, b), (c, d) = value["outer"], value["inner"]
        return "set", [i for i in range(int(a), int(b)+1) if not c <= i <= d]
    if key == "choice":
        return "set", list(value)
    return "set", [value]


def _spec_key(spec):
    keys = [k for k in SPEC_KEYS if k in spec]
    if len(keys) != 1 or set(spec)-set(SPEC_KEYS)-{"note"}:
        raise BenchmarkConfigError(f"A range spec has exactly one of {SPEC_KEYS} (and an optional note): {spec}")
    return keys[0]


def _disjoint(id_spec, ood_spec):
    a_kind, a = _support(id_spec)
    b_kind, b = _support(ood_spec)
    if a_kind == "set" or b_kind == "set":
        if a_kind != b_kind:
            return False
        return not (set(map(json.dumps, a)) & set(map(json.dumps, b)))
    return all(hi <= lo2 or hi2 <= lo for lo, hi in a for lo2, hi2 in b)


def family_spec(config, family, dimension, range_set):
    """Effective range spec of a dimension for a family and range set (family override, else global)."""
    override = config["families"][family].get("ranges", {}).get(range_set, {})
    if dimension in override:
        return override[dimension]
    return config["range_sets"][range_set].get(dimension)


def seed_list(config, family, split, range_set):
    rule = config["seeds"]
    fam = config["families"][family]
    count = int(rule["counts"][split][range_set])
    start = int(fam["seed_base"])+int(rule["split_offsets"][split])+int(rule["range_set_offsets"][range_set])
    return [start+i for i in range(count)]


def strata(config):
    """[{id, family, range_set}] in declared order (families x range sets unless declared)."""
    if "strata" in config:
        return deepcopy(config["strata"])
    return [{"id": f"{f}_{rs}", "family": f, "range_set": rs} for f in config["families"] for rs in RANGE_SETS
            if rs in config["families"][f].get("range_sets", RANGE_SETS)]


def sample_ids(config, split="benchmark"):
    """[(family, range_set, seed, index)] of a split in declared order (strata, then seed order)."""
    out = []
    for stratum in strata(config):
        for i, seed in enumerate(seed_list(config, stratum["family"], split, stratum["range_set"])):
            out.append((stratum["family"], stratum["range_set"], seed, i))
    return out


def make_sample_id(config, family, range_set, split, seed):
    return f"{config['sample_prefix']}_{family}_{range_set}_{split}_s{int(seed)}"


def validate_benchmark(doc):
    if doc.get("schema") != SCHEMA:
        raise BenchmarkConfigError(f"Not a {SCHEMA} benchmark file")
    if doc.get("status") not in ("DRAFT", "FROZEN"):
        raise BenchmarkConfigError("status must be DRAFT or FROZEN")
    dims = doc["dimensions"]
    for name, dim in dims.items():
        if dim.get("group") not in GROUPS or dim.get("kind") not in KINDS:
            raise BenchmarkConfigError(f"Dimension {name}: group in {GROUPS}, kind in {KINDS}")
    for rs in RANGE_SETS:
        for name, spec in doc["range_sets"][rs].items():
            if name not in dims:
                raise BenchmarkConfigError(f"range_sets.{rs}.{name} is not a declared dimension")
            key = _spec_key(spec)
            if rs == "ID" and key.endswith("_outside"):
                raise BenchmarkConfigError(f"ID range of {name} cannot be an outside shell")
    for name, fam in doc["families"].items():
        if fam.get("builder") not in BUILDERS:
            raise BenchmarkConfigError(f"Family {name}: builder must be one of {sorted(BUILDERS)}")
        allowed = BUILDERS[fam["builder"]][1]
        if not fam.get("variants") or set(fam["variants"])-set(allowed):
            raise BenchmarkConfigError(f"Family {name}: variants must be a nonempty subset of {allowed}")
        for dim in fam["randomize"]:
            if dim not in dims:
                raise BenchmarkConfigError(f"Family {name} randomizes unknown dimension {dim}")
            if family_spec(doc, name, dim, "ID") is None:
                raise BenchmarkConfigError(f"Family {name}: no ID range for {dim}")
        for rs, rows in fam.get("ranges", {}).items():
            for dim, spec in rows.items():
                _spec_key(spec)
                if dim not in fam["randomize"]:
                    raise BenchmarkConfigError(f"Family {name}: range override for non-randomized {dim}")
        if set(fam.get("ood_dimensions", []))-set(fam["randomize"]):
            raise BenchmarkConfigError(f"Family {name}: ood_dimensions must be randomized dimensions")
        if "OOD" in fam.get("range_sets", RANGE_SETS):
            if not fam.get("ood_dimensions"):
                raise BenchmarkConfigError(f"Family {name}: an OOD stratum needs ood_dimensions")
            for variant in fam["variants"]:
                if not ood_candidates(doc, name, variant):
                    raise BenchmarkConfigError(f"Family {name}: variant {variant} reads none of the ood_dimensions")
            for dim in fam["ood_dimensions"]:
                ood = family_spec(doc, name, dim, "OOD")
                if ood is None:
                    raise BenchmarkConfigError(f"Family {name}: no OOD range for {dim}")
                if not _disjoint(family_spec(doc, name, dim, "ID"), ood):
                    raise BenchmarkConfigError(f"Family {name}: the OOD range of {dim} overlaps its ID range")
    missing = set(doc["common"]["nominal"])-set(dims)
    if missing:
        raise BenchmarkConfigError(f"common.nominal has undeclared dimensions {sorted(missing)}")
    # seeds: every (family, seed) is unique over splits and range sets; stratum sizes
    seen = {}
    for stratum in strata(doc):
        for split in SPLITS:
            for seed in seed_list(doc, stratum["family"], split, stratum["range_set"]):
                key = (stratum["family"], seed)
                if key in seen:
                    raise BenchmarkConfigError(f"Seed {seed} of family {stratum['family']} is in {seen[key]} and "
                                               f"{(split, stratum['range_set'])}")
                seen[key] = (split, stratum["range_set"])
    all_seeds = [s for (_, s) in seen]
    if len(all_seeds) != len(set(all_seeds)):
        raise BenchmarkConfigError("Seeds must be unique across families (seed_base spacing)")
    minimum = int(doc["sampling"]["min_per_stratum"])
    for stratum in strata(doc):
        if len(seed_list(doc, stratum["family"], "benchmark", stratum["range_set"])) < minimum:
            raise BenchmarkConfigError(f"Stratum {stratum['id']} has fewer than {minimum} benchmark configurations")
    if int(doc["native"]["selection"]["k"]) < int(doc["native"]["selection"].get("min_k", 3)):
        raise BenchmarkConfigError("Native sub-sampling needs k >= 3 per stratum")
    for value in _choice_values(doc, "material_profile"):
        if not _material_known(value):
            raise BenchmarkConfigError(f"material_profile {value!r} is not a known profile")
    return doc


def _choice_values(doc, name):
    out = []
    for rs in RANGE_SETS:
        specs = [doc["range_sets"][rs].get(name)]+[f.get("ranges", {}).get(rs, {}).get(name) for f in doc["families"].values()]
        for spec in specs:
            if spec is not None:
                out += _support(spec)[1]
    if name in doc["common"]["nominal"]:
        out.append(doc["common"]["nominal"][name])
    return out


def _material_known(name):
    from .scene_config import V1_MATERIAL_PROFILE
    if name == V1_MATERIAL_PROFILE:
        return True
    from .food_materials import profiles
    if name in profiles():
        return True
    try:
        from .materials_wp12 import is_profile
    except ImportError:
        return False
    return is_profile(name)


# ----------------------------------------------------------------------------- draws

def _dimension_key(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def stream(config, family, seed, name):
    """The dimension's own RNG stream (independent of every other dimension)."""
    return np.random.default_rng([int(config["salt"]), int(config["families"][family]["salt"]), int(seed),
                                  _dimension_key(name)])


def draw_spec(spec, rng, size=None):
    """One value (or ``size`` values) of a range spec from ``rng``."""
    key = _spec_key(spec)
    n = 1 if size is None else int(size)
    if key == "fixed":
        values = [deepcopy(spec["fixed"]) for _ in range(n)]
    elif key == "uniform":
        lo, hi = spec["uniform"]
        values = [float(v) for v in rng.uniform(lo, hi, n)]
    elif key == "uniform_outside":
        intervals = _support(spec)[1]
        lengths = np.array([hi-lo for lo, hi in intervals])
        values = []
        for u in rng.uniform(0., lengths.sum(), n):
            i = int(np.searchsorted(np.cumsum(lengths), u, side="right"))
            i = min(i, len(intervals)-1)
            values.append(float(intervals[i][0]+(u-(np.cumsum(lengths)[i]-lengths[i]))))
    else:
        options = _support(spec)[1]
        values = [deepcopy(options[int(j)]) for j in rng.integers(0, len(options), n)]
        if key in ("integer", "integer_outside"):
            values = [int(v) for v in values]
    return values[0] if size is None else values


_PROTRUSION_DIMS = ("protrusion_family", "protrusion_count", "protrusion_spread_fraction", "protrusion_x_fraction",
                    "protrusion_x_jitter_m", "protrusion_y_m", "protrusion_radius_x_m", "protrusion_radius_y_m",
                    "protrusion_height_m")
_SKIM_DIMS = ("rake_deg", "skew_deg", "skim_direction", "skim_speed_m_s")
# Dimensions a builder variant does NOT read (they are still drawn, so the other streams never shift, and recorded with
# applied = False). An OOD sample's OOD dimension is chosen only among the dimensions its variant reads.
UNUSED_DIMENSIONS = {
    ("oblique_cut", "tilt"): ("cut_plane_yaw_deg", "advance_tilt_deg"),
    ("oblique_cut", "plane_yaw"): ("cut_tilt_deg", "advance_tilt_deg"),
    ("oblique_cut", "advance_tilt"): ("cut_tilt_deg", "cut_plane_yaw_deg"),
    ("trim", "skim_protrusion"): ("trim_depth_m",),
    ("trim", "trim_plane_protrusion"): ("trim_depth_m",),
    ("trim", "trim_layer"): _PROTRUSION_DIMS,
    ("multi_slice", "cut_cut"): ("cut_count",),
    ("multi_protrusion", "remove_all"): ("selected_protrusion_key",),
    ("multi_protrusion", "bumps_sequential"): ("selected_protrusion_key",),
    ("multi_protrusion", "trim_trim"): ("protrusion_count", "selected_protrusion_key"),
    ("cut_push", "cut_push"): _PROTRUSION_DIMS+_SKIM_DIMS+("clear_direction",),
    ("cut_push", "trim_clear_chip"): ("cut_fraction", "push_distance_m", "cut_speed_m_s", "cut_clearance_m"),
    ("shape_pose_support", "sever"): _PROTRUSION_DIMS+_SKIM_DIMS,
    ("shape_pose_support", "trim"): ("cut_speed_m_s", "cut_clearance_m"),
}


def variant_of(config, family, index):
    fam = config["families"][family]
    return fam["variants"][index % len(fam["variants"])]


def ood_candidates(config, family, variant):
    unused = set(UNUSED_DIMENSIONS.get((config["families"][family]["builder"], variant), ()))
    return [d for d in config["families"][family].get("ood_dimensions", []) if d not in unused]


def draw_values(config, family, range_set, seed, index):
    """(values, rows, ood_dimensions_applied): nominal values overlaid with the family's draws."""
    fam = config["families"][family]
    dims = config["dimensions"]
    variant = variant_of(config, family, index)
    unused = set(UNUSED_DIMENSIONS.get((fam["builder"], variant), ()))
    values = deepcopy(config["common"]["nominal"])
    rows = []
    ood = []
    if range_set == "OOD":
        candidates = ood_candidates(config, family, variant)
        if config["sampling"].get("ood_mode", "one_at_a_time") == "all":
            ood = candidates
        else:
            ood = [candidates[(index//len(fam["variants"])) % len(candidates)]]
    for name in fam["randomize"]:
        used = "OOD" if name in ood else "ID"
        spec = family_spec(config, family, name, used)
        value = draw_spec(spec, stream(config, family, seed, name), dims[name].get("size"))
        values[name] = value
        rows.append({"name": name, "group": dims[name]["group"], "range_set": used, "spec": spec, "value": value,
                     "applied": name not in unused})
    return values, rows, ood


# ----------------------------------------------------------------------------- scene / goal helpers

def _round_to(value, step):
    return float(round(round(value/step)*step, 12))


def _dims(common, v):
    rule = common["dimension_rule"]
    return (_round_to(v["length_m"], rule["length_step_m"]), _round_to(v["width_m"], rule["width_step_m"]),
            _round_to(v["height_m"], rule["height_step_m"]))


def _node_x(common, x):
    return _round_to(x, common["solver"]["grid_dx_m"])


def _record(common):
    from .flat_board_workcell import profiles
    return profiles()[common["workcell_profile"]]


def _board_frame_x(common, pose, x_local):
    """Board-frame x of a workpiece-local x on the local x axis (the board is co-rotated with the workpiece)."""
    board = _record(common)["cutting_board"]
    yaw = math.radians(pose[2])
    world = np.array([pose[0], pose[1]])+x_local*np.array([math.cos(yaw), math.sin(yaw)])
    rel = world-np.asarray(board["center_xy_world_m"], float)
    return float(rel@np.array([math.cos(yaw), math.sin(yaw)]))


def _board_overrides(common, v, pose, slot_x_local=()):
    """cutting_board overrides: yaw co-rotated with the workpiece (no oriented TLMPM fixture), and the drawn board
    dimensions/friction/kind; a slotted board gets one groove under each declared local x."""
    out = {"yaw_deg": float(pose[2])}
    for key, name in (("length_x_m", "board_length_m"), ("width_y_m", "board_width_m"), ("thickness_m", "board_thickness_m"),
                      ("friction", "board_friction")):
        if v.get(name) is not None:
            out[key] = v[name] if name != "board_friction" or isinstance(v[name], str) else float(v[name])
    if v.get("board_kind", "solid") == "slotted":
        rule = common["slotted_rule"]
        out.update(kind="slotted", slot_depth_m=float(rule["slot_depth_m"]),
                   slots=[{"x_board_m": round(_board_frame_x(common, pose, x), 6), "width_m": float(rule["slot_width_m"])}
                          for x in sorted(slot_x_local)])
    return out


def _scene(common, sample_id, dims, pose, protrusions, board, material, description):
    from .flat_board_workcell import place_on_board
    L, W, H = dims
    scene = {"schema": "rcs_scene_v1", "scene_id": sample_id, "description": description,
             "workpiece": {"base_box_m": {"length_m": L, "width_m": W, "height_m": H}, "protrusions": protrusions,
                           "world_pose": {"translation_m": [float(pose[0]), float(pose[1]), 0.], "yaw_deg": float(pose[2])},
                           "material_profile": material},
             "supports": {"kind": "flat_board", "workcell": {"profile": common["workcell_profile"], "overrides": {"cutting_board": board}}},
             "tool": deepcopy(common["tool"]),
             "solver": {"module": common["solver"]["module"], "cutting_law": common["solver"]["cutting_law"],
                        "max_cuts": 1, "particle_spacing_m": common["solver"]["particle_spacing_m"],
                        "grid_dx_m": common["solver"]["grid_dx_m"]}}
    if "slots_per_node" in common["solver"]:
        scene["solver"]["slots_per_node"] = int(common["solver"]["slots_per_node"])
    try:
        return place_on_board(scene)   # workpiece z = board top of the (overridden) board
    except ValueError:
        return scene   # an invalid board override is reported by scene validation (scene_geometrically_infeasible)


def _goal(common, goal_id, kind, description, fields, planner=None):
    return {"schema": "rcs_goal_v1", "goal_id": goal_id, "type": kind, "description": description,
            "episode": {"duration_s": common["duration_rule"]["probe_duration_s"], "control_dt_s": common["control_dt_s"]},
            "planner": {**common["planner"], **(planner or {})}, "evaluator_version": common["evaluator_version"], **fields}


def _min_part(common, fractions):
    return round(float(common["sever_rule"]["minimum_part_fraction_of_smallest_nominal"])*min(fractions), 6)


def _sever_fields(common, planes, length):
    """Sever fields: expected pieces and the WP-11 Phase E minimum-part rule; other thresholds are goals.py defaults."""
    mids = sorted(_plane_x_at(p, p["origin_local_m"][2]) for p in planes)
    edges = [-length/2, *mids, length/2]
    fractions = [(b-a)/length for a, b in zip(edges[:-1], edges[1:])]
    return {"planes": planes, "expected_pieces": len(planes)+1,
            "minimum_part_mass_fraction": _min_part(common, fractions)}, fractions


def _plane_x_at(plane, z):
    (x0, y0, z0), (nx, ny, nz) = plane["origin_local_m"], plane["normal_local"]
    return x0-nz*(z-z0)/nx if abs(nx) > 1e-12 else x0


def _plane(x, height, tilt_deg=0., yaw_deg=0., advance_deg=0.):
    """Plane through (x, 0, H/2) (or (x, 0, 0) when vertical and unyawed): normal +x turned about local +y by the tilt
    (C2 OBLIQUE convention, as generalization._oblique_plane) and then about local z by the yaw; ``advance_deg`` tilts
    the advance inside the plane toward -edge line (WP-11 Test A s3 slanted chop)."""
    t, y = math.radians(tilt_deg), math.radians(yaw_deg)
    n = np.array([math.cos(t)*math.cos(y), math.cos(t)*math.sin(y), -math.sin(t)])
    if abs(tilt_deg) < 1e-12 and abs(yaw_deg) < 1e-12:
        row = G._vertical_plane(x)
    else:
        row = {"origin_local_m": [float(x), 0., height/2], "normal_local": [float(c) for c in n]}
    if abs(advance_deg) > 1e-12:
        down = np.array([0., 0., -1.])
        a0 = down-(down@n)*n
        a0 /= np.linalg.norm(a0)
        e0 = np.cross(n, a0)
        b = math.radians(advance_deg)
        row["approach_local"] = [float(c) for c in (math.cos(b)*a0-math.sin(b)*e0)]
    return row


def _bump(pid, family, x, y, rx, ry, h):
    return {"id": pid, "family": family, "x_m": float(x), "y_m": float(y), "radius_x_m": float(rx),
            "radius_y_m": float(ry), "height_m": float(h)}


def _protrusions(v, centres):
    out = []
    for i, x in enumerate(centres):
        fam = v["protrusion_family"][i] if isinstance(v["protrusion_family"], list) else v["protrusion_family"]
        pick = (lambda name: v[name][i] if isinstance(v[name], list) else v[name])
        out.append(_bump(f"p{i}", fam, x, pick("protrusion_y_m"), pick("protrusion_radius_x_m"),
                         pick("protrusion_radius_y_m"), pick("protrusion_height_m")))
    return out


def _skim_planner(v):
    direction = {"+x": [1., 0., 0.], "-x": [-1., 0., 0.]}[v.get("skim_direction", "+x")]
    return {"rake_deg": float(v["rake_deg"]), "skew_deg": float(v.get("skew_deg", 0.)),
            "skim_direction_local": direction, "skim_speed_m_s": float(v["skim_speed_m_s"])}


def _cut_planner(v):
    return {"cut_speed_m_s": float(v["cut_speed_m_s"]), "cut_clearance_m": float(v["cut_clearance_m"])}


def _remove_step(select, goal_id):
    return {"type": "remove_protrusions", "goal_id": goal_id, "select": select}


def _trim_region_step(bump, H, margin, goal_id):
    return {"type": "trim_plane", "goal_id": goal_id, "plane": {"origin_local_m": [0., 0., H], "normal_local": [0., 0., 1.]},
            "region_local_m": {"x": [bump["x_m"]-bump["radius_x_m"]-margin, bump["x_m"]+bump["radius_x_m"]+margin]}}


def _spread_centres(v, L, n):
    a = float(v["protrusion_spread_fraction"])*L
    centres = np.linspace(-a, a, n) if n > 1 else np.array([a])
    jitter = v["protrusion_x_jitter_m"]
    return [float(c+(jitter[i] if isinstance(jitter, list) else jitter)) for i, c in enumerate(centres)]


# ----------------------------------------------------------------------------- family builders
# Each builder: (config, family, values, variant, sample_id) -> (scene, goal, derived, envelope_params, geometry)
# ``geometry`` is the generator-level description used by ``geometry_checks`` (planes, protrusions, base).

def _common_scene(config, v, sample_id, protrusions, slot_x=(), description=""):
    common = config["common"]
    dims = _dims(common, v)
    pose = (float(v["workpiece_x_m"]), float(v["workpiece_y_m"]), float(v["workpiece_yaw_deg"]))
    board = _board_overrides(common, v, pose, slot_x)
    scene = _scene(common, sample_id, dims, pose, protrusions, board, v["material_profile"], description)
    return scene, dims, pose


def _base_params(v, dims, protrusions):
    out = {"board_kind": v.get("board_kind", "solid"), "material_profile": v["material_profile"],
           "workpiece_height_m": dims[2], "workpiece_yaw_abs_deg": abs(float(v["workpiece_yaw_deg"]))}
    if protrusions:
        out["protrusion_families"] = sorted({p["family"] for p in protrusions})
    return out


def _sever_params(planes, v):
    tilts, yaws, adv = [], [], []
    for p in planes:
        n = np.asarray(p["normal_local"], float)
        tilts.append(math.degrees(math.asin(min(1., abs(n[2])))))
        yaws.append(math.degrees(math.atan2(abs(n[1]), abs(n[0]))))
        a = p.get("approach_local")
        adv.append(0. if a is None else math.degrees(math.acos(min(1., abs(float(np.asarray(a)@[0., 0., -1.]))/
                                                                    max(1e-12, math.hypot(math.hypot(a[0], a[1]), a[2]))))))
    return {"sever_tilt_abs_deg_max": max(tilts), "sever_plane_yaw_abs_deg_max": max(yaws),
            "advance_tilt_abs_deg_max": max(adv), "cut_speed_m_s": float(v["cut_speed_m_s"])}


def _skim_params(v):
    return {"rake_deg_max": float(v["rake_deg"]), "rake_deg_min": float(v["rake_deg"]),
            "skew_abs_deg_max": abs(float(v.get("skew_deg", 0.))), "skim_speed_m_s": float(v["skim_speed_m_s"])}


def build_single_cut(config, family, v, variant, sample_id):
    common = config["common"]
    L, W, H = _dims(common, v)
    x = _node_x(common, float(v["cut_fraction"])*L)
    plane = _plane(x, H)
    scene, dims, pose = _common_scene(config, v, sample_id, [], [x], f"WP-12 Phase 5 {family} single straight cut")
    fields, fractions = _sever_fields(common, [plane], L)
    goal = _goal(common, f"{sample_id}_sever", "sever", f"Sever at local x = {x*1e3:.2f} mm (vertical, node plane).",
                 fields, _cut_planner(v))
    params = {**_base_params(v, dims, []), **_sever_params([plane], v), "cut_digits": 1, "operation_pattern": "sever"}
    return scene, goal, {"cut_x_local_m": x, "nominal_piece_fractions": fractions}, params, {"planes": [plane]}


def build_oblique_cut(config, family, v, variant, sample_id):
    common = config["common"]
    L, W, H = _dims(common, v)
    tilt = float(v["cut_tilt_deg"]) if variant == "tilt" else 0.
    yaw = float(v["cut_plane_yaw_deg"]) if variant == "plane_yaw" else 0.
    adv = float(v["advance_tilt_deg"]) if variant == "advance_tilt" else 0.
    x = float(v["cut_fraction"])*L
    if tilt == 0. and yaw == 0.:
        x = _node_x(common, x)   # vertical, unyawed planes on node planes (WP-11 cut_plane_rule)
    plane = _plane(x, H, tilt, yaw, adv)
    slot_x = [_plane_x_at(plane, 0.)] if v.get("board_kind") == "slotted" else [x]
    scene, dims, pose = _common_scene(config, v, sample_id, [], slot_x, f"WP-12 Phase 5 {family} {variant} cut")
    fields, fractions = _sever_fields(common, [plane], L)
    goal = _goal(common, f"{sample_id}_sever_{variant}", "sever",
                 f"Sever at local x = {x*1e3:.2f} mm: tilt {tilt:.2f} deg about local y, plane yaw {yaw:.2f} deg, "
                 f"advance tilt {adv:.2f} deg.", fields, _cut_planner(v))
    params = {**_base_params(v, dims, []), **_sever_params([plane], v), "cut_digits": 1, "operation_pattern": "sever"}
    derived = {"cut_x_local_m": x, "applied_tilt_deg": tilt, "applied_plane_yaw_deg": yaw, "applied_advance_tilt_deg": adv,
               "nominal_piece_fractions": fractions}
    return scene, goal, derived, params, {"planes": [plane]}


def build_trim(config, family, v, variant, sample_id):
    common = config["common"]
    L, W, H = _dims(common, v)
    protrusions = [] if variant == "trim_layer" else _protrusions(v, [float(v["protrusion_x_fraction"])*L])
    scene, dims, pose = _common_scene(config, v, sample_id, protrusions, [], f"WP-12 Phase 5 {family} {variant}")
    planner = _skim_planner(v)
    params = {**_base_params(v, dims, protrusions), **_skim_params(v), "cut_digits": 1}
    if variant == "skim_protrusion":
        goal = _goal(common, f"{sample_id}_skim", "remove_protrusions",
                     f"Remove the {protrusions[0]['family']} protrusion p0 with one surface_trim pass (rake "
                     f"{planner['rake_deg']:.1f} deg).", {"select": "all"}, planner)
        params["operation_pattern"] = "remove_protrusions"
    elif variant == "trim_plane_protrusion":
        step = _trim_region_step(protrusions[0], H, common["trim_region_margin_m"], "trim_p0")
        goal = _goal(common, f"{sample_id}_trim_plane", "trim_plane",
                     "Trim the plane z = H over the protrusion region (one pass).",
                     {k: step[k] for k in ("plane", "region_local_m")}, planner)
        params["operation_pattern"] = "trim_plane"
    else:   # trim_layer: one full-length near-horizontal pass at depth d below the base top
        depth = _round_to(float(v["trim_depth_m"]), common["dimension_rule"]["height_step_m"])
        goal = _goal(common, f"{sample_id}_trim_layer", "trim_plane",
                     f"Trim a full-length top layer {depth*1e3:.1f} mm deep (plane z = H - depth, one pass).",
                     {"plane": {"origin_local_m": [0., 0., H-depth], "normal_local": [0., 0., 1.]}}, planner)
        params.update(operation_pattern="trim_plane", trim_depth_m=depth)
    return scene, goal, {"variant": variant}, params, {"protrusions": protrusions}


def _ordered(v, planes):
    order = v.get("cut_order", "ascending")
    idx = list(range(len(planes)))
    if order == "descending":
        idx = idx[::-1]
    elif order == "shuffled":
        keys = v["cut_order_key"][:len(planes)]
        idx = [int(i) for i in np.argsort(keys, kind="stable")]
    return [planes[i] for i in idx], order, idx


def build_multi_slice(config, family, v, variant, sample_id):
    common = config["common"]
    L, W, H = _dims(common, v)
    k = 2 if variant == "cut_cut" else int(v["cut_count"])
    jitter = v["slice_jitter_m"]
    xs = sorted(_node_x(common, -L/2+L*(i+1)/(k+1)+jitter[i]) for i in range(k))
    planes, order, idx = _ordered(v, [_plane(x, H) for x in xs])
    scene, dims, pose = _common_scene(config, v, sample_id, [], xs, f"WP-12 Phase 5 {family} {variant}")
    params = {**_base_params(v, dims, []), **_sever_params(planes, v), "cut_digits": k}
    if variant == "multi_plane_sever":
        fields, fractions = _sever_fields(common, planes, L)
        goal = _goal(common, f"{sample_id}_sever{k}", "sever",
                     f"{k} through-cuts at local x = {', '.join('%.1f' % (p['origin_local_m'][0]*1e3) for p in planes)} mm "
                     f"in that order ({order}).", fields, _cut_planner(v))
        params["operation_pattern"] = "sever"
    else:
        steps = []
        for i, p in enumerate(planes):
            fields, _ = _sever_fields(common, [p], L)
            steps.append({"type": "sever", "goal_id": f"cut{i}", **fields})
        fractions = None
        goal = _goal(common, f"{sample_id}_cut_cut", "sequence",
                     f"Cut -> cut: sever at x = {planes[0]['origin_local_m'][0]*1e3:.1f} mm, then at "
                     f"{planes[1]['origin_local_m'][0]*1e3:.1f} mm.", {"steps": steps}, _cut_planner(v))
        params["operation_pattern"] = "sequence:sever>sever"
    return scene, goal, {"cut_x_local_m": xs, "cut_order": order, "order_index": idx, "nominal_piece_fractions": fractions}, \
        params, {"planes": planes}


def build_multi_protrusion(config, family, v, variant, sample_id):
    common = config["common"]
    L, W, H = _dims(common, v)
    n = 2 if variant == "trim_trim" else int(v["protrusion_count"])
    protrusions = _protrusions(v, _spread_centres(v, L, n))
    scene, dims, pose = _common_scene(config, v, sample_id, protrusions, [], f"WP-12 Phase 5 {family} {variant}")
    planner = _skim_planner(v)
    params = {**_base_params(v, dims, protrusions), **_skim_params(v)}
    ids = [p["id"] for p in protrusions]
    derived = {"protrusions": n, "families": [p["family"] for p in protrusions]}
    if variant == "remove_all":
        goal = _goal(common, f"{sample_id}_remove_all", "remove_protrusions", f"Remove all {n} protrusions (one pass each).",
                     {"select": "all"}, planner)
        params.update(cut_digits=n, operation_pattern="remove_protrusions")
    elif variant == "remove_selected":
        k = min(n-1, int(float(v["selected_protrusion_key"])*n))
        goal = _goal(common, f"{sample_id}_remove_p{k}", "remove_protrusions",
                     f"Remove protrusion p{k} only and keep the others.", {"select": [f"p{k}"]}, planner)
        params.update(cut_digits=1, operation_pattern="remove_protrusions")
        derived["selected"] = f"p{k}"
    elif variant == "bumps_sequential":
        goal = _goal(common, f"{sample_id}_bumps_seq", "sequence",
                     "Remove " + " then ".join(ids) + " (one remove step each).",
                     {"steps": [_remove_step([pid], f"remove_{pid}") for pid in ids]}, planner)
        params.update(cut_digits=n, operation_pattern="sequence:" + ">".join(["remove_protrusions"]*n))
    else:   # trim_trim
        steps = [_trim_region_step(p, H, common["trim_region_margin_m"], f"trim_{p['id']}") for p in protrusions]
        goal = _goal(common, f"{sample_id}_trim_trim", "sequence", "Trim -> trim: trim_plane over p0's region, then p1's.",
                     {"steps": steps}, planner)
        params.update(cut_digits=2, operation_pattern="sequence:trim_plane>trim_plane")
    return scene, goal, derived, params, {"protrusions": protrusions, "separate_removal": n > 1}


def build_sever_trim(config, family, v, variant, sample_id):
    common = config["common"]
    L, W, H = _dims(common, v)
    n = int(v["protrusion_count"])
    centres = _spread_centres(v, L, n)
    protrusions = _protrusions(v, centres)
    x = _node_x(common, float(v["cut_fraction"])*L)
    plane = _plane(x, H)
    scene, dims, pose = _common_scene(config, v, sample_id, protrusions, [x], f"WP-12 Phase 5 {family} {variant}")
    fields, fractions = _sever_fields(common, [plane], L)
    sever = {"type": "sever", "goal_id": "sever", **fields}
    remove = _remove_step("all", "remove_all")
    steps = [sever, remove] if variant == "sever_then_trim" else [remove, sever]
    goal = _goal(common, f"{sample_id}_{variant}", "sequence",
                 f"{'Sever then trim' if variant == 'sever_then_trim' else 'Trim then sever'}: sever at x = {x*1e3:.1f} mm, "
                 f"remove all {n} protrusions.", {"steps": steps}, {**_cut_planner(v), **_skim_planner(v)})
    params = {**_base_params(v, dims, protrusions), **_sever_params([plane], v), **_skim_params(v), "cut_digits": 1+n,
              "operation_pattern": "sequence:" + ">".join(s["type"] for s in steps)}
    return scene, goal, {"cut_x_local_m": x, "nominal_piece_fractions": fractions}, params, \
        {"planes": [plane], "protrusions": protrusions, "separate_removal": n > 1, "plane_must_miss_protrusions": True}


def build_cut_push(config, family, v, variant, sample_id):
    common = config["common"]
    if variant == "cut_reposition_cut":
        raise NotExpressible("cut -> reposition -> cut: " + MISSING_FEATURES["planner_offset_local_m"])
    L, W, H = _dims(common, v)
    if variant == "cut_push":
        x = _node_x(common, float(v["cut_fraction"])*L)
        plane = _plane(x, H)
        scene, dims, pose = _common_scene(config, v, sample_id, [], [x], f"WP-12 Phase 5 {family} cut -> push")
        fields, fractions = _sever_fields(common, [plane], L)
        push = {"type": "move_piece", "goal_id": "push", "direction_local": [1., 0., 0.],
                "distance_m": float(v["push_distance_m"]),
                "selector": {"side_of_plane": {"origin_local_m": [x, 0., 0.], "normal_local": [1., 0., 0.], "side": "positive"}}}
        goal = _goal(common, f"{sample_id}_cut_push", "sequence",
                     f"Sever at x = {x*1e3:.1f} mm, then a cut-disabled blade-face push of the +x piece by "
                     f"{float(v['push_distance_m'])*1e3:.1f} mm.", {"steps": [{"type": "sever", "goal_id": "sever", **fields}, push]},
                     _cut_planner(v))
        params = {**_base_params(v, dims, []), **_sever_params([plane], v), "cut_digits": 1,
                  "operation_pattern": "sequence:sever>move_piece"}
        return scene, goal, {"cut_x_local_m": x, "nominal_piece_fractions": fractions}, params, {"planes": [plane]}
    # trim_clear_chip
    protrusions = _protrusions(v, [float(v["protrusion_x_fraction"])*L])
    p = protrusions[0]
    scene, dims, pose = _common_scene(config, v, sample_id, protrusions, [], f"WP-12 Phase 5 {family} trim -> clear chip")
    direction = {"+x": [1., 0., 0.], "-x": [-1., 0., 0.]}[v.get("clear_direction", "+x")]
    clearance = float(common["chip_selector_clearance_m"])
    move = {"type": "move_piece", "goal_id": "clear_chip", "direction_local": direction, "push_mode": "surface_sweep",
            "selector": {"box_local_m": {"min": [p["x_m"]-p["radius_x_m"], p["y_m"]-p["radius_y_m"], H+clearance],
                                         "max": [p["x_m"]+p["radius_x_m"], p["y_m"]+p["radius_y_m"], H+p["height_m"]+.005]}}}
    goal = _goal(common, f"{sample_id}_trim_clear", "sequence",
                 f"Remove p0, then sweep its chip along {v.get('clear_direction', '+x')} off the workpiece.",
                 {"steps": [_remove_step(["p0"], "remove_p0"), move]}, _skim_planner(v))
    params = {**_base_params(v, dims, protrusions), **_skim_params(v), "cut_digits": 1,
              "operation_pattern": "sequence:remove_protrusions>move_piece"}
    return scene, goal, {"clear_direction": direction}, params, {"protrusions": protrusions}


def build_shape_pose_support(config, family, v, variant, sample_id):
    common = config["common"]
    L, W, H = _dims(common, v)
    if variant == "sever":
        x = _node_x(common, float(v["cut_fraction"])*L)
        plane = _plane(x, H)
        scene, dims, pose = _common_scene(config, v, sample_id, [], [x], f"WP-12 Phase 5 {family} shape/pose/support sever")
        fields, fractions = _sever_fields(common, [plane], L)
        goal = _goal(common, f"{sample_id}_sever", "sever", f"Sever at local x = {x*1e3:.2f} mm on a "
                     f"{v.get('board_kind', 'solid')} board.", fields, _cut_planner(v))
        params = {**_base_params(v, dims, []), **_sever_params([plane], v), "cut_digits": 1, "operation_pattern": "sever"}
        return scene, goal, {"cut_x_local_m": x, "nominal_piece_fractions": fractions}, params, {"planes": [plane]}
    x = float(v["protrusion_x_fraction"])*L
    protrusions = _protrusions(v, [x])
    scene, dims, pose = _common_scene(config, v, sample_id, protrusions, [_node_x(common, float(v["cut_fraction"])*L)],
                                      f"WP-12 Phase 5 {family} shape/pose/support trim")
    planner = _skim_planner(v)
    goal = _goal(common, f"{sample_id}_skim", "remove_protrusions", "Remove p0 with one surface_trim pass.",
                 {"select": "all"}, planner)
    params = {**_base_params(v, dims, protrusions), **_skim_params(v), "cut_digits": 1,
              "operation_pattern": "remove_protrusions"}
    return scene, goal, {"variant": variant}, params, {"protrusions": protrusions}


BUILDERS = {
    "single_cut": (build_single_cut, ("vertical",)),
    "oblique_cut": (build_oblique_cut, ("tilt", "plane_yaw", "advance_tilt")),
    "trim": (build_trim, ("skim_protrusion", "trim_plane_protrusion", "trim_layer")),
    "multi_slice": (build_multi_slice, ("multi_plane_sever", "cut_cut")),
    "multi_protrusion": (build_multi_protrusion, ("remove_all", "remove_selected", "bumps_sequential", "trim_trim")),
    "sever_trim": (build_sever_trim, ("sever_then_trim", "trim_then_sever")),
    "cut_push": (build_cut_push, ("cut_push", "trim_clear_chip", "cut_reposition_cut")),
    "shape_pose_support": (build_shape_pose_support, ("sever", "trim")),
}


def build_sample(config, family, range_set, seed, *, split="benchmark", index=None):
    """One sample: {"sample_id", "family", "range_set", "split", "stratum", "seed", "index", "variant", "scene", "goal",
    "draws", "ood_dimensions", "derived", "envelope_parameters", "geometry"} (no compile). Declared seeds keep their
    declared index; ``index`` is required only for seeds outside every declared list (development probes)."""
    fam = config["families"][family]
    seeds = seed_list(config, family, split, range_set)
    if int(seed) in seeds:
        if index is not None and index != seeds.index(int(seed)):
            raise ValueError("A declared seed keeps its declared index")
        index = seeds.index(int(seed))
    elif index is None:
        raise ValueError(f"Seed {seed} is not declared for {family}/{range_set}/{split}; pass index=")
    values, rows, ood = draw_values(config, family, range_set, seed, index)
    variant = variant_of(config, family, index)
    sample_id = make_sample_id(config, family, range_set, split, seed)
    scene, goal, derived, params, geometry = BUILDERS[fam["builder"]][0](config, family, values, variant, sample_id)
    return {"sample_id": sample_id, "family": family, "range_set": range_set, "split": split,
            "stratum": f"{family}_{range_set}", "seed": int(seed), "index": int(index), "variant": variant,
            "declared": int(seed) in seeds, "scene": scene, "goal": goal, "draws": rows, "ood_dimensions": ood,
            "derived": derived, "envelope_parameters": params, "geometry": geometry,
            "rng": "numpy.random.default_rng([benchmark salt, family salt, seed, sha256(dimension)[:8]]) per dimension"}


def sample_by_id(config, sample_id):
    """Rebuild a declared sample from its id."""
    prefix = config["sample_prefix"]
    parts = sample_id[len(prefix)+1:].split("_")
    if not sample_id.startswith(prefix+"_") or len(parts) < 4 or not parts[-1].startswith("s"):
        raise ValueError(f"Malformed sample id {sample_id}")
    family, range_set, split, seed = parts[0], parts[1], "_".join(parts[2:-1]), parts[-1]
    return build_sample(config, family, range_set, int(seed[1:]), split=split)


# ----------------------------------------------------------------------------- feasibility

def geometry_checks(sample, common=None):
    """Generator-level scene-geometry problems [(reason, detail)] (scene validation/compile adds its own)."""
    problems = []
    base = sample["scene"]["workpiece"]["base_box_m"]
    L, H = base["length_m"], base["height_m"]
    geo = sample.get("geometry", {})
    planes = geo.get("planes", [])
    min_gap = 2*(common or {}).get("solver", {}).get("grid_dx_m", .004)
    xs = []
    for p in planes:
        for z in (0., H):
            x = _plane_x_at(p, z)
            if not -L/2+min_gap <= x <= L/2-min_gap:
                problems.append(("cut_plane_outside_workpiece", f"plane x {x*1e3:.2f} mm at z {z*1e3:.1f} mm vs half "
                                                                f"length {L/2*1e3:.1f} mm minus {min_gap*1e3:.0f} mm"))
                break
        xs.append(_plane_x_at(p, H/2))
    xs.sort()
    if any(b-a < min_gap for a, b in zip(xs[:-1], xs[1:])):
        problems.append(("cut_planes_too_close", f"plane mid-height x {[round(x*1e3, 2) for x in xs]} mm (< {min_gap*1e3:.0f} mm)"))
    bumps = geo.get("protrusions", [])
    if geo.get("separate_removal") and len(bumps) > 1:
        ordered = sorted(bumps, key=lambda r: r["x_m"])
        for a, b in zip(ordered[:-1], ordered[1:]):
            if a["x_m"]+a["radius_x_m"] > b["x_m"]-b["radius_x_m"]:
                problems.append(("protrusions_overlap", f"{a['id']} and {b['id']} footprints overlap along x"))
    if geo.get("plane_must_miss_protrusions"):
        for p in planes:
            for b in bumps:
                x = _plane_x_at(p, H)
                if b["x_m"]-b["radius_x_m"] <= x <= b["x_m"]+b["radius_x_m"]:
                    problems.append(("cut_plane_intersects_protrusion", f"plane at x {x*1e3:.1f} mm crosses {b['id']}"))
    return problems


def _rule_applies(rule, params):
    return all(params.get(k) == value for k, value in (rule.get("when") or {}).items())


def evaluate_envelope(envelope, params):
    """{"status": SUPPORTED | UNSUPPORTED | PENDING, "rules": [...]} (null bounds -> NOT_EVALUATED -> PENDING)."""
    rows, unsupported, pending = [], [], []
    for rule in envelope.get("rules", []):
        name, kind = rule["parameter"], rule["kind"]
        if name not in params or not _rule_applies(rule, params):
            rows.append({"id": rule["id"], "status": "NOT_APPLICABLE"})
            continue
        value = params[name]
        bounds = {k: rule.get(k) for k in ("min", "max", "values") if k in rule}
        needed = {"max": ("max",), "min": ("min",), "range": ("min", "max"), "in_set": ("values",), "subset": ("values",)}[kind]
        if any(rule.get(k) is None for k in needed):
            rows.append({"id": rule["id"], "status": "NOT_EVALUATED", "value": value, "reason": "bound undeclared"})
            pending.append(rule["id"])
            continue
        if kind == "max":
            ok = value <= rule["max"]+1e-12
        elif kind == "min":
            ok = value >= rule["min"]-1e-12
        elif kind == "range":
            ok = rule["min"]-1e-12 <= value <= rule["max"]+1e-12
        elif kind == "in_set":
            ok = value in rule["values"]
        else:
            ok = set(value) <= set(rule["values"])
        rows.append({"id": rule["id"], "status": "PASS" if ok else "FAIL", "value": value, **bounds})
        if not ok:
            unsupported.append(rule["id"])
    status = "UNSUPPORTED" if unsupported else ("PENDING" if pending or envelope.get("status") != "DECLARED" else "SUPPORTED")
    return {"status": status, "unsupported_rules": unsupported, "pending_rules": pending,
            "envelope_status": envelope.get("status"), "envelope_id": envelope.get("envelope_id"), "rules": rows}


def classify_preflight(receipt):
    """[classes] a motion-preflight receipt fails: ik_unreachable (IK pose error or joint-path checks) and/or
    collision_constrained (any collision / clearance check). Empty list = PASS."""
    if receipt.get("status") == "PASS":
        return [], []
    if receipt.get("reason") == "IK_FAILURE":
        f = receipt.get("ik_failure", {})
        return ["ik_unreachable"], [f"ik pose error at sample {f.get('sample')} ({f.get('label')})"]
    failing = list(receipt.get("path", {}).get("failing", []))
    worst = receipt.get("path", {}).get("worst", {})
    ik = [n for n in failing if n in G.IK_PATH_CHECKS]
    col = [n for n in failing if n not in G.IK_PATH_CHECKS]
    classes, reasons = [], []
    if ik:
        classes.append("ik_unreachable")
        reasons.append("joint path: " + ", ".join(ik))
    if col:
        classes.append("collision_constrained")
        reasons.append("collision/clearance: " + ", ".join(
            f"{n} (slack {worst[n]['slack']*1e3:.2f} mm at {worst[n].get('label')}, pair {worst[n].get('pair')})"
            if n in worst else n for n in col))
    if not classes:   # a FAIL receipt with no named check (e.g. an ERROR status) is a classification error
        return None, [f"preflight status {receipt.get('status')} without failing checks: {receipt.get('reason')}"]
    return classes, reasons


def _scene_error_is_geometric(error):
    from .scene_config import SceneConfigError, SceneInfeasible
    if isinstance(error, SceneInfeasible):
        return True
    text = str(error)
    if isinstance(error, SceneConfigError):   # flat-board record problems (board off the table, slots) are geometry
        return text.startswith("supports.workcell:") or "protrusion" in text
    return isinstance(error, (ValueError, NotImplementedError, RuntimeError))   # planner/compile geometry (as WP-11)


def classify_feasibility(sample, envelope, *, compile_fn=None, preflight_fn=None, precedence=DEFAULT_PRECEDENCE,
                         run_preflight_when_unsupported=True, common=None):
    """Pre-execution class of one sample with every check kept. ``compile_fn(sample) -> (task, info)`` and
    ``preflight_fn(task) -> receipt`` are injectable (CPU tests use synthetic ones); ``preflight_fn=None`` records the
    preflight as NOT_RUN (the sample cannot be valid_feasible then: class None, ``classification_error``)."""
    checks, failing, reasons = {}, [], {}
    geo = geometry_checks(sample, common)
    task, info = None, None
    if geo:
        checks["scene_geometry"] = {"status": "FAIL", "stage": "generator", "problems": [list(p) for p in geo]}
    elif compile_fn is not None:
        try:
            task, info = compile_fn(sample)
            checks["scene_geometry"] = {"status": "PASS", "stage": "compile"}
        except Exception as error:   # classified, never raised past the sample
            if not _scene_error_is_geometric(error):
                checks["scene_geometry"] = {"status": "ERROR", "stage": "compile", "error": f"{type(error).__name__}: {error}"}
                return _result(sample, None, checks, [], {}, f"config error: {type(error).__name__}: {error}", info)
            stage = "scene" if type(error).__name__ == "SceneInfeasible" else "compile"
            checks["scene_geometry"] = {"status": "FAIL", "stage": stage, "problems": [[type(error).__name__, str(error)]]}
    else:
        checks["scene_geometry"] = {"status": "PASS", "stage": "generator_only"}
    if checks["scene_geometry"]["status"] == "FAIL":
        failing.append("scene_geometrically_infeasible")
        reasons["scene_geometrically_infeasible"] = "; ".join(f"{a}: {b}" for a, b in checks["scene_geometry"]["problems"])
    params = dict(sample["envelope_parameters"])
    if info is not None and "digits_used" in info:
        params["cut_digits"] = int(info["digits_used"])
    env = evaluate_envelope(envelope, params)
    material = params.get("material_profile")
    if task is not None and material is not None and common is not None:
        try:
            from .materials_wp12 import is_profile, stability
            if is_profile(material):
                receipt = stability(material, task.grid_dx_m, common["material_dt_s"], task.particle_spacing_m)
                env["material_admission"] = receipt
                if not receipt["admitted"]:
                    env["status"] = "UNSUPPORTED"
                    env["unsupported_rules"].append("material_cfl_admission")
        except ImportError:
            pass
    checks["envelope"] = env
    if env["status"] == "UNSUPPORTED":
        failing.append("physically_unsupported")
        reasons["physically_unsupported"] = "outside the declared envelope: " + ", ".join(env["unsupported_rules"])
    if task is None or (env["status"] == "UNSUPPORTED" and not run_preflight_when_unsupported):
        checks["preflight"] = {"status": "NOT_RUN", "reason": "no compiled task" if task is None else "physically unsupported"}
    elif preflight_fn is None:
        checks["preflight"] = {"status": "NOT_RUN", "reason": "no preflight function"}
    else:
        try:
            receipt = preflight_fn(task)
        except Exception as error:
            receipt = {"status": "ERROR", "reason": f"{type(error).__name__}: {error}"}
        classes, why = classify_preflight(receipt)
        checks["preflight"] = {"status": receipt.get("status"), "classes": classes, "reasons": why, "receipt": receipt}
        if classes is None:
            if not failing:
                return _result(sample, None, checks, failing, reasons, "preflight error: " + "; ".join(why), info)
        else:
            for cls, text in zip(classes, why):
                failing.append(cls)
                reasons[cls] = text
    ordered = [c for c in precedence if c in failing]
    if ordered:
        return _result(sample, ordered[0], checks, ordered, reasons, reasons[ordered[0]], info)
    if checks["preflight"]["status"] != "PASS":
        return _result(sample, None, checks, [], {}, f"preflight {checks['preflight']['status']}: "
                                                    f"{checks['preflight'].get('reason')}", info)
    return _result(sample, "valid_feasible", checks, [], {}, "every pre-execution check passed", info)


def _result(sample, cls, checks, failing, reasons, reason, info):
    env = checks.get("envelope", {})
    return {"sample_id": sample["sample_id"], "family": sample["family"], "range_set": sample["range_set"],
            "split": sample["split"], "stratum": sample["stratum"], "seed": sample["seed"], "variant": sample["variant"],
            "ood_dimensions": sample["ood_dimensions"], "feasibility_class": cls, "classification_error": cls is None,
            "failing_classes": failing, "class_reasons": reasons, "reason": reason, "checks": checks,
            "envelope_pending": env.get("status") == "PENDING",
            "compile": None if info is None else {k: v for k, v in info.items() if k not in ("scene", "goal")}}


# ----------------------------------------------------------------------------- compile / preflight glue (reuse WP-11)

def _shim(config, sample):
    common = config["common"]
    return {"common": {"material_seed": common["material_seed"], "duration_rule": common["duration_rule"],
                       "history": common["history"]},
            "families": {sample["family"]: {"solver": {"module": common["solver"]["module"],
                                                        **({"slots_per_node": common["solver"]["slots_per_node"]}
                                                           if "slots_per_node" in common["solver"] else {})}}}}


def compile_sample(config, sample):
    """(ConfiguredTaskSpec, info) by WP-11 ``generalization.compile_sample`` (solver = the benchmark solver for every
    digit count, max_cuts = digits; duration = plan end + settle, rounded up)."""
    return G.compile_sample(_shim(config, sample), sample)


def preflight(task, geometry="staged_v2"):
    """WP-11 staged C2 motion preflight (Newton IK + MuJoCo distances, flat-board blade/board rule)."""
    from . import motion_preflight as mp
    checker = mp.MotionChecker(task)
    receipt, _ = G.staged_preflight(task, checker, geometry=geometry)
    receipt["checker_build_wall_s"] = checker.build_wall_s
    return receipt


def history_stride(config, steps):
    return G.history_stride(_shim(config, {"family": "_"}), steps)


# ----------------------------------------------------------------------------- execution classes

def classify_execution(result):
    """(class, failure_kind, reason) of an executed episode result (kinematic or Native)."""
    if result is None or result.get("status") in (None, "NOT_RUN"):
        return "not_run", None, (result or {}).get("reason", "not executed")
    if result.get("status") == "ABORTED" and not result.get("solver_invalid"):
        return "execution_failure", "crash", result.get("reason") or result.get("error") or "aborted"
    if result.get("grasp_failure"):
        return "execution_failure", "grasp", str(result["grasp_failure"])
    cls, reason = G.classify_material(result)
    if cls == "TASK_SUCCESS":
        if result.get("partial"):
            return "not_run", None, "partial (smoke) run: never a task result"
        return "task_success", None, reason
    if cls == "NOT_RUN":
        return "not_run", None, reason
    kind = ("solver_invalid" if reason.startswith("solver state invalid") else
            "plan_not_finished" if reason.startswith("the scripted plan") else
            "integrity" if reason.startswith("integrity") else "outcome" if reason.startswith("outcome") else "other")
    if result.get("partial"):
        return "not_run", None, f"partial (smoke) run: {reason}"
    return "execution_failure", kind, reason


def final_row(feasibility, execution=None):
    """Report row: the feasibility class for an infeasible sample, else the execution class (never dropped)."""
    row = {k: feasibility.get(k) for k in ("sample_id", "family", "range_set", "split", "stratum", "seed", "variant",
                                            "ood_dimensions", "feasibility_class", "envelope_pending")}
    if feasibility.get("feasibility_class") in INFEASIBLE_CLASSES:
        row.update(**{"class": feasibility["feasibility_class"], "failure_kind": None, "reason": feasibility["reason"]})
    elif feasibility.get("feasibility_class") == "valid_feasible":
        cls, kind, reason = classify_execution(execution)
        row.update(**{"class": cls, "failure_kind": kind, "reason": reason})
    else:
        row.update(**{"class": "not_run", "failure_kind": None,
                      "reason": f"feasibility not determined: {feasibility.get('reason')}"})
    return row


# ----------------------------------------------------------------------------- summary

def wilson(k, n, z=WILSON_Z):
    """Wilson score interval {"k", "n", "p", "lo", "hi"} (None fields when n = 0)."""
    if n <= 0:
        return {"k": int(k), "n": int(n), "p": None, "lo": None, "hi": None}
    p = k/n
    den = 1+z*z/n
    centre = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    lo = 0. if k == 0 else max(0., centre-half)   # exact endpoints at k = 0 and k = n
    hi = 1. if k == n else min(1., centre+half)
    return {"k": int(k), "n": int(n), "p": p, "lo": lo, "hi": hi}


def _group(rows):
    n = len(rows)
    counts = {c: sum(r["class"] == c for r in rows) for c in REPORT_CLASSES}
    if sum(counts.values()) != n:
        raise AssertionError(f"Class counts {counts} do not add up to {n} samples (unknown class)")
    feasible = sum(r.get("feasibility_class") == "valid_feasible" for r in rows)
    undetermined = sum(r.get("feasibility_class") is None for r in rows)
    success = counts["task_success"]
    executed = sum(r.get("feasibility_class") == "valid_feasible" and r["class"] != "not_run" for r in rows)
    kinds = {k: sum(r.get("failure_kind") == k for r in rows) for k in EXECUTION_FAILURE_KINDS}
    return {"samples": n, "counts": counts,
            "class_distribution": {c: (counts[c]/n if n else None) for c in REPORT_CLASSES},
            "execution_failure_kinds": kinds, "valid_feasible": feasible, "feasibility_undetermined": undetermined,
            "executed_among_feasible": executed, "envelope_pending": sum(bool(r.get("envelope_pending")) for r in rows),
            "feasibility_rate": wilson(feasible, n), "success_among_feasible": wilson(success, feasible),
            "success_among_executed": wilson(success, executed), "overall_success": wilson(success, n)}


def summarize(rows, declared_ids=None, targets=None):
    """Tables per stratum, family, range set and overall. ``declared_ids``: every declared sample id; an id without a
    row is added as not_run ('no record'), so no sample can disappear from a denominator."""
    rows = list(rows)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate sample rows")
    missing = []
    if declared_ids is not None:
        known = set(ids)
        for sid, meta in declared_ids:
            if sid not in known:
                missing.append(sid)
                rows.append({"sample_id": sid, **meta, "feasibility_class": None, "class": "not_run",
                             "failure_kind": None, "reason": "no record (missing output)", "envelope_pending": False})
        extra = set(ids)-{sid for sid, _ in declared_ids}
        if extra:
            raise AssertionError(f"Rows for undeclared samples: {sorted(extra)[:5]}")
    out = {"rule": "feasibility rate = valid_feasible / samples; success among feasible = task_success / valid_feasible "
                   "(not_run feasible samples stay in the denominator); overall = task_success / samples; Wilson 95 % "
                   "score intervals; every declared sample is counted exactly once (missing outputs are not_run)",
           "missing_records": missing,
           "overall": _group(rows),
           "by_range_set": {rs: _group([r for r in rows if r["range_set"] == rs]) for rs in RANGE_SETS
                            if any(r["range_set"] == rs for r in rows)},
           "by_family": {f: _group([r for r in rows if r["family"] == f]) for f in dict.fromkeys(r["family"] for r in rows)},
           "by_family_range_set": {s: _group([r for r in rows if r["stratum"] == s])
                                   for s in dict.fromkeys(r["stratum"] for r in rows)}}
    if sum(t["samples"] for t in out["by_family_range_set"].values()) != len(rows):
        raise AssertionError("Strata do not partition the samples")
    ood = [r for r in rows if r["range_set"] == "OOD" and r.get("ood_dimensions")]
    if ood:
        dims = sorted({d for r in ood for d in r["ood_dimensions"]})
        out["ood_boundary_by_dimension"] = {d: _group([r for r in ood if d in r["ood_dimensions"]]) for d in dims}
    if targets:
        out["targets"] = {}
        for name, target in targets.items():
            group = out["by_range_set"].get(target["range_set"], {})
            stat = group.get(target["metric"]) if group else None
            p = None if stat is None else stat["p"]
            out["targets"][name] = {**target, "estimate": stat,
                                    "met_point": None if p is None else p >= target["value"],
                                    "met_lower_bound": None if p is None else stat["lo"] >= target["value"],
                                    "note": "engineering target (spec 5.7); not met is reported as not met, never redefined"}
    return out


# ----------------------------------------------------------------------------- Native stratified sub-sampler

def native_selection(config, feasibility_rows, split="benchmark"):
    """Native sample per stratum by the declared rule (pre-execution: uses feasibility classes only, never outcomes).

    Rules: ``first_k_valid_feasible`` (declared seed order) or ``seeded_shuffle_valid_feasible`` (a permutation of the
    stratum's valid_feasible samples from ``default_rng([native salt, sha256(stratum)[:8]])``, then the first k).
    Returns {stratum: {"selected", "shortfall", "not_selected": {sample_id: reason}}}."""
    rule = config["native"]["selection"]
    k = int(rule["k"])
    by_id = {r["sample_id"]: r for r in feasibility_rows}
    out = {}
    for stratum in strata(config):
        if rule.get("strata", "all") != "all" and stratum["id"] not in rule["strata"]:
            continue
        ids = [make_sample_id(config, stratum["family"], stratum["range_set"], split, s)
               for s in seed_list(config, stratum["family"], split, stratum["range_set"])]
        candidates = [i for i in ids if by_id.get(i, {}).get("feasibility_class") == "valid_feasible"]
        if rule["rule"] == "seeded_shuffle_valid_feasible":
            rng = np.random.default_rng([int(rule["salt"]), _dimension_key(stratum["id"])])
            candidates = [candidates[int(j)] for j in rng.permutation(len(candidates))]
        elif rule["rule"] != "first_k_valid_feasible":
            raise BenchmarkConfigError(f"Unknown Native selection rule {rule['rule']!r}")
        selected = candidates[:k]
        reasons = {}
        for i in ids:
            if i in selected:
                continue
            cls = by_id.get(i, {}).get("feasibility_class")
            reasons[i] = (f"not valid_feasible ({cls})" if cls != "valid_feasible" else
                          f"valid_feasible, not among the first {k} by rule {rule['rule']}")
        out[stratum["id"]] = {"selected": selected, "shortfall": max(0, k-len(selected)),
                              "shortfall_reason": None if len(selected) >= k else
                              f"only {len(selected)} valid_feasible samples in the stratum",
                              "not_selected": reasons}
    return out


# ----------------------------------------------------------------------------- freeze readiness

def coverage(config, split="benchmark"):
    """Per family: variants and OOD dimensions reached by the declared seeds of a split, and the declared ones that
    are never reached (a warning to resolve before the freeze: more seeds or fewer OOD dimensions)."""
    out = {}
    for family, fam in config["families"].items():
        variants, ood = {}, {}
        for rs in RANGE_SETS:
            for i, _ in enumerate(seed_list(config, family, split, rs)):
                variant = variant_of(config, family, i)
                variants[variant] = variants.get(variant, 0)+1
                if rs == "OOD":
                    candidates = ood_candidates(config, family, variant)
                    dim = candidates[(i//len(fam["variants"])) % len(candidates)]
                    ood[dim] = ood.get(dim, 0)+1
        out[family] = {"variants": variants, "ood_dimensions": ood,
                       "variants_never_reached": [v for v in fam["variants"] if v not in variants],
                       "ood_dimensions_never_reached": [d for d in fam.get("ood_dimensions", []) if d not in ood]}
    return out


ENVELOPE_DIMENSIONS = {"rake_deg": ("rake_deg_max", "rake_deg_min"), "cut_tilt_deg": ("sever_tilt_abs_deg_max",),
                       "cut_plane_yaw_deg": ("sever_plane_yaw_abs_deg_max",), "advance_tilt_deg": ("advance_tilt_abs_deg_max",),
                       "skew_deg": ("skew_abs_deg_max",), "cut_speed_m_s": ("cut_speed_m_s",),
                       "skim_speed_m_s": ("skim_speed_m_s",), "trim_depth_m": ("trim_depth_m",),
                       "height_m": ("workpiece_height_m",), "material_profile": ("material_profile",)}


def freeze_problems(config, envelope):
    """Why the benchmark cannot be frozen yet (empty list = ready): envelope declared and complete, every family's ID
    range inside the envelope where a rule maps a dimension, DRAFT marker present."""
    problems = []
    if envelope.get("status") != "DECLARED":
        problems.append(f"envelope status {envelope.get('status')} (needs DECLARED after the Phase 4 xhigh decision)")
    for rule in envelope.get("rules", []):
        needed = {"max": ("max",), "min": ("min",), "range": ("min", "max"), "in_set": ("values",), "subset": ("values",)}[rule["kind"]]
        if any(rule.get(k) is None for k in needed):
            problems.append(f"envelope rule {rule['id']} has an undeclared bound")
    rules = {}
    for rule in envelope.get("rules", []):
        if not rule.get("when"):
            rules.setdefault(rule["parameter"], []).append(rule)
    for family, fam in config["families"].items():
        for dim in fam["randomize"]:
            for param in ENVELOPE_DIMENSIONS.get(dim, ()):
                for rule in rules.get(param, []):
                    kind, support = _support(family_spec(config, family, dim, "ID"))
                    values = [abs(v) for iv in support for v in iv] if kind == "intervals" and param.endswith("abs_deg_max") \
                        else ([v for iv in support for v in iv] if kind == "intervals" else support)
                    bad = []
                    for value in values:
                        if rule["kind"] == "in_set" and rule.get("values") is not None and value not in rule["values"]:
                            bad.append(value)
                        if rule["kind"] in ("max", "range") and rule.get("max") is not None and isinstance(value, (int, float)) \
                                and value > rule["max"]+1e-12:
                            bad.append(value)
                        if rule["kind"] in ("min", "range") and rule.get("min") is not None and isinstance(value, (int, float)) \
                                and value < rule["min"]-1e-12:
                            bad.append(value)
                    if bad:
                        problems.append(f"family {family}: ID range of {dim} reaches {bad} outside envelope rule {rule['id']}")
    for family, row in coverage(config).items():
        if row["variants_never_reached"] or row["ood_dimensions_never_reached"]:
            problems.append(f"family {family}: benchmark seeds never reach variants {row['variants_never_reached']} / OOD "
                            f"dimensions {row['ood_dimensions_never_reached']}")
    if config.get("status") == "FROZEN" and "DRAFT" in str(config.get("benchmark_id", "")):
        problems.append("a FROZEN file must not carry a DRAFT id")
    return problems


# ----------------------------------------------------------------------------- quality (report-only)

def physical_quality(task, material, evaluation, metrics, stability, actions=None):
    """Spec 5.6 physical-quality block (read-only, never part of a verdict)."""
    out = {"substrate_retention": None, "material_loss": None, "contact_stability": stability,
           "knife_grasp_stability": "NOT_APPLICABLE (kinematic ideal tracking: no grasp)",
           "physical_cutting_validity": None, "penetration": None, "remaining_unwanted_bridges": None}
    try:
        steps = metrics.get("steps") or [metrics]
        shares = [s.get("substrate_share") for s in steps if s.get("substrate_share") is not None]
        out["substrate_retention"] = min(shares) if shares else None
        losses = [s.get(k) for s in steps for k in ("overcut_debris_mass_fraction", "overcut_detached_below_plane_fraction")
                  if s.get(k) is not None]
        out["material_loss"] = max(losses) if losses else None
        comps = metrics.get("mechanical_components", next((s.get("mechanical_components") for s in steps
                                                           if s.get("mechanical_components") is not None), None))
        out["mechanical_components"] = comps
        integrity = evaluation.get("integrity", {})
        out["physical_cutting_validity"] = {"integrity": integrity.get("status"),
                                           "failed_integrity_checks": [k for k, ok in integrity.get("checks", {}).items() if not ok]}
    except Exception as error:   # report-only
        out["error"] = repr(error)
    if material is not None:
        try:
            z = material.bed.x.numpy()[:, 2]
            spacing = float(material.spacing)
            out["penetration"] = {"min_particle_z_local_m": float(z.min()),
                                  "particles_below_board_top": int((z < -1e-6).sum()),
                                  "rule": "workpiece-local z < 0 is below the rigid board top (TLMPM fixture contact)",
                                  "particle_spacing_m": spacing}
        except Exception as error:
            out["penetration"] = {"error": repr(error)}
    if actions is not None:
        out["workspace_clamped_steps"] = int(sum(bool(a.get("workspace_clamped")) for a in actions))
    return out


def geometry_quality(env, task, level="basic"):
    """Spec 5.6 geometry-quality block of the final state on the env's selected surface profile (report-only):
    ``basic``: mesh validity, mesh shells vs mechanical components, volume consistency; ``full`` adds
    ``surface_metrics_v1.surface_report`` (cut-face error). Errors are recorded, never raised."""
    import time
    from . import surface_metrics_v1 as SM
    from .goals import packed_components
    out = {"level": level, "surface_profile": getattr(env.backend.config, "surface_profile", None) or "particle_surface_v1"}
    try:
        if hasattr(env.backend, "_surface_time"):
            env.backend._surface_time = None   # drop the render cache (not physics) so the timing is a fresh extraction
        start = time.perf_counter()
        surface = env.backend.surface   # KinematicCuttingBackend.surface is a property (WORLD vertices, triangles)
        vertices, faces = surface() if callable(surface) else surface
        out["extract_wall_s"] = time.perf_counter()-start
        material = env.backend.material
        bed = material.bed
        local = np.asarray(task.points_to_local(np.asarray(vertices, float)), float)
        faces = np.asarray(faces, np.int64)
        positions, codes = bed.x.numpy().astype(float), bed.code.numpy()
        F = bed.F.numpy().astype(float)
        encoding = getattr(material, "code_encoding", "ternary")
        count, labels = packed_components(bed.nodes.numpy(), bed.weights.numpy(), codes, gradients=bed.gradients.numpy()) \
            if encoding == "packed2" else (None, None)
        out.update(vertices=int(len(local)), triangles=int(len(faces)), mechanical_components=count)
        start = time.perf_counter()
        out["mesh_validity"] = SM.mesh_validity_v1(local, faces)
        if labels is not None:
            out["topology"] = SM.topology_consistency(local, faces, positions, labels, material.masses)
            density = float(task.material.density_kg_m3) if hasattr(task.material, "density_kg_m3") else None
            if density:
                out["volume"] = SM.volume_report(local, faces, positions, labels, np.asarray(material.masses)/density, F)
        if level == "full" and labels is not None:
            out["surface_report"] = SM.surface_report(bed.reference.numpy().astype(float), material.lattice, positions, codes,
                                                      local, faces, spacing=float(material.spacing), labels=labels,
                                                      masses=material.masses, encoding=encoding,
                                                      cuts=int(getattr(material, "max_cuts", 1)), F=F, reference_space=False)
        out["metrics_wall_s"] = time.perf_counter()-start
        out["status"] = "MEASURED"
    except Exception as error:
        out.update(status="ERROR", error=repr(error))
    return out
