"""WP-11 Phase E generalization rounds: seeded scene + goal samples and outcome classes.

The round file (``configs/wp11/generalization_round1.json``, schema
``rcs_generalization_round_v1``) predeclares every family, seed, randomization
range, goal and outcome-class definition before any sample runs. This module
only turns a (family, seed) into an ``rcs_scene_v1`` scene and an
``rcs_goal_v1`` goal by the declared rules, classifies the stage results and
summarizes a round. It never changes physics, cutting laws, evaluators or
thresholds, and it never reads oracle material state to build a scene or plan.

Randomization: ``numpy.random.default_rng([salt, seed])`` draws ``uniform(low,
high)`` for each entry of the family's ``draws`` list, in declared order (every
draw is made even if a rule does not use it, so the stream never depends on
earlier values). Discrete choices that must cover every declared value (cut
angle, protrusion family, rake, slice count) follow declared seed-index rules,
not the RNG. Every sampled and derived value is recorded per sample, and the
scene-level values are also written into the scene's ``randomization.sampled``
block (ADR 0015), so a compiled task carries them.

Round 2 (``configs/wp11/generalization_round2.json``) adds, all opt-in so round 1 builds,
compiles and ids are unchanged: ``common.sample_prefix`` / ``common.round_label`` (default
"E1" / "round 1"), a per-family ``solver`` override (tlmpm_v5 with ``slots_per_node``), a
per-family ``evaluator_version`` for its goals, the builders ``fixed_workpiece_posed``
(Test A variants on a drawn pose) and ``sequence_mixed`` (Test D2 declared step patterns),
and ``stages.stage2_subset`` (a predeclared rule-based stage-2 subset).
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROUND1 = ROOT/"configs/wp11/generalization_round1.json"
SCHEMA = "rcs_generalization_round_v1"
CLASSES = ("SCENE_INFEASIBLE", "IK_FAILURE", "GRASP_FAILURE", "CUTTING_FAILURE", "TASK_SUCCESS", "NOT_RUN")
IK_PATH_CHECKS = ("joint_limit_margin", "joint_continuity")


def load_round(path=ROUND1):
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} round file")
    return doc


def sample_ids(config):
    """[(family, seed)] in declared order."""
    return [(name, int(seed)) for name, family in config["families"].items() for seed in family["seeds"]]


def make_sample_id(config, family, seed):
    return f"{config['common'].get('sample_prefix', 'E1')}_{family}_s{int(seed)}"


def stage2_selection(config, stage1_rows):
    """Stage-1 rows (declared order) that stage 2 runs: every feasible sample, or the declared subset rule.

    ``stages.stage2_subset`` = {"all_feasible": [families], "first_feasible_per_family": {"count": n,
    "families": [...]}} selects every feasible sample of the first list and the first n feasible samples
    (declared seed order) of each family of the second. Families in neither list run no stage 2.
    """
    rule = config.get("stages", {}).get("stage2_subset")
    feasible = [row for row in stage1_rows if row.get("feasible")]
    if rule is None:
        return feasible
    order = {key: i for i, key in enumerate(sample_ids(config))}
    feasible.sort(key=lambda row: order[(row["family"], int(row["seed"]))])
    every = set(rule.get("all_feasible", []))
    first = rule.get("first_feasible_per_family", {"count": 0, "families": []})
    taken, out = {}, []
    for row in feasible:
        family = row["family"]
        if family in every:
            out.append(row)
        elif family in first["families"] and taken.get(family, 0) < int(first["count"]):
            taken[family] = taken.get(family, 0)+1
            out.append(row)
    return out


# ----------------------------------------------------------------------------- draws

def _draws(config, family, seed):
    spec = config["families"][family]
    rng = np.random.default_rng([int(spec["salt"]), int(seed)])
    values, rows = {}, []
    for name, low, high in spec.get("draws", []):
        value = float(rng.uniform(low, high))
        values[name] = value
        rows.append({"name": name, "low": float(low), "high": float(high), "value": value})
    return values, rows, rng


def _index(config, family, seed, index=None):
    """Seed index in the declared list; ``index`` is only for development seeds outside the round."""
    seeds = list(config["families"][family]["seeds"])
    if int(seed) in seeds:
        if index is not None and index != seeds.index(int(seed)):
            raise ValueError("A declared seed keeps its declared index")
        return seeds.index(int(seed))
    if index is None:
        raise ValueError(f"Seed {seed} is not declared for {family}; pass index= for a development sample")
    return int(index)


def _supports(common, length, width, slots):
    rule = common["support_rule"]
    out = {"kind": "slotted_board" if slots else "solid_board",
           "x_range_m": [-(length/2+rule["x_margin_m"]), length/2+rule["x_margin_m"]],
           "half_width_y_m": width/2+rule["y_margin_m"], "thickness_m": rule["thickness_m"],
           "workcell": deepcopy(common["workcell"])}
    if slots:
        out["slots"] = [{"x_m": float(x), "width_m": float(w)} for x, w in sorted(slots)]
    return out


def _scene(config, sample_id, dims, pose, protrusions, slots, scene_rows, description):
    common = config["common"]
    length, width, height = dims
    scene = {"schema": "rcs_scene_v1", "scene_id": sample_id, "description": description,
             "workpiece": {"base_box_m": {"length_m": length, "width_m": width, "height_m": height},
                           "protrusions": protrusions,
                           "world_pose": {"translation_m": [pose[0], pose[1], common["world_z_m"]], "yaw_deg": pose[2]},
                           "material_profile": common["material_profile"]},
             "supports": _supports(common, length, width, slots),
             "tool": deepcopy(common["tool"]),
             "solver": {"module": "tlmpm_v5", "cutting_law": common["solver"]["cutting_law"], "max_cuts": 15,
                        "particle_spacing_m": common["solver"]["particle_spacing_m"],
                        "grid_dx_m": common["solver"]["grid_dx_m"]}}
    if scene_rows:
        # ADR 0015 randomization block, already resolved: resolve_scene keeps it (idempotent on "sampled").
        scene["randomization"] = {"seed": int(scene_rows[0]["seed"]),
                                  "ranges": [{"path": r["path"], "low": r["low"], "high": r["high"], "mode": "absolute"}
                                             for r in scene_rows],
                                  "sampled": [{"path": r["path"], "nominal": r["draw"], "draw": r["draw"],
                                               "mode": "absolute", "value": r["value"]} for r in scene_rows],
                                  "rng": config["common"]["rng"]}
    return scene


def _goal(config, goal_id, kind, description, fields, planner=None, evaluator_version=None):
    goal = {"schema": "rcs_goal_v1", "goal_id": goal_id, "type": kind, "description": description,
            "episode": {"duration_s": config["common"]["duration_rule"]["probe_duration_s"],
                        "control_dt_s": config["common"]["control_dt_s"]},
            "planner": {**config["common"]["planner"], **(planner or {})}, **fields}
    if evaluator_version is not None:  # round 2 opt-in (ADR 0015 Amendment 1); absent in round 1
        goal["evaluator_version"] = evaluator_version
    return goal


def _min_part(config, fractions):
    rule = config["common"]["sever_rule"]
    return round(float(rule["minimum_part_fraction_of_smallest_nominal"])*min(fractions), 6)


def _vertical_plane(x):
    return {"origin_local_m": [float(x), 0., 0.], "normal_local": [1., 0., 0.]}


def _oblique_plane(x, height, angle_deg):
    """Plane through (x, 0, H/2); normal +x turned about local +y by angle (C2 OBLIQUE frame; edge line +y)."""
    a = math.radians(angle_deg)
    return {"origin_local_m": [float(x), 0., height/2], "normal_local": [math.cos(a), 0., -math.sin(a)]}


def _plane_x_at(plane, z):
    (x0, _, z0), (nx, _, nz) = plane["origin_local_m"], plane["normal_local"]
    return x0-nz*(z-z0)/nx


def _sever_fields(config, planes, length, expected=None):
    # Nominal piece fractions from the plane x positions (vertical planes: origin x; oblique: x at mid-height).
    mids = sorted(p["origin_local_m"][0] for p in planes)
    edges = [-length/2, *mids, length/2]
    fractions = [(b-a)/length for a, b in zip(edges[:-1], edges[1:])]
    return {"planes": planes, "expected_pieces": expected or len(planes)+1,
            "minimum_part_mass_fraction": _min_part(config, fractions),
            "minimum_main_parts_mass_fraction": config["common"]["sever_rule"]["minimum_main_parts_mass_fraction"],
            "plane_error_bound_m": config["common"]["sever_rule"]["plane_error_bound_m"]}, fractions


def _remove_fields(config, select="all"):
    rule = config["common"]["remove_rule"]
    return {"select": select, "plane_local_z_m": None, "edge_clearance_m": rule["edge_clearance_m"],
            "minimum_removed_fraction": rule["minimum_removed_fraction"],
            "minimum_substrate_share": rule["minimum_substrate_share"],
            "maximum_detached_below_fraction": rule["maximum_detached_below_fraction"]}


def _relief_slot(config, plane, height, width_m):
    """Relief slot under a cut plane: centred where the plane meets local z = -slot_centre_depth_m."""
    depth = config["common"]["support_rule"]["slot_centre_depth_m"]
    return (_plane_x_at(plane, -depth), width_m)


def _scene_rows(seed, values, mapping, specs, applied=None):
    rows = []
    for name, path in mapping:
        low, high = specs[name]
        rows.append({"seed": int(seed), "path": path, "low": low, "high": high, "draw": values[name],
                     "value": (applied or {}).get(name, values[name])})
    return rows


def _round_to(value, step):
    return float(round(round(value/step)*step, 12))


def _dims(config, values):
    """Declared discretization of the drawn base box (common.dimension_rule)."""
    rule = config["common"]["dimension_rule"]
    return (_round_to(values["length_m"], rule["length_step_m"]), _round_to(values["width_m"], rule["width_step_m"]),
            _round_to(values["height_m"], rule["height_step_m"]))


def _node_x(config, x):
    """Vertical cut plane x snapped to the nearest solver node plane (common.cut_plane_rule)."""
    return _round_to(x, config["common"]["solver"]["grid_dx_m"])


_POSE_PATHS = (("length_m", "workpiece.base_box_m.length_m"), ("width_m", "workpiece.base_box_m.width_m"),
               ("height_m", "workpiece.base_box_m.height_m"), ("x_m", "workpiece.world_pose.translation_m.0"),
               ("y_m", "workpiece.world_pose.translation_m.1"), ("yaw_deg", "workpiece.world_pose.yaw_deg"))


def _bump(config, pid, family, x, y, rx, ry, h):
    return {"id": pid, "family": family, "x_m": float(x), "y_m": float(y), "radius_x_m": float(rx),
            "radius_y_m": float(ry), "height_m": float(h)}


def build_sample(config, family, seed, *, index=None):
    """{"family", "seed", "sample_id", "scene", "goal", "draws", "derived"} by the declared rules (no compile).

    ``index`` is accepted only for development seeds that are not part of the round.
    """
    spec = config["families"][family]
    values, rows, rng = _draws(config, family, seed)
    index = _index(config, family, seed, index)
    specs = {name: (low, high) for name, low, high in spec.get("draws", [])}
    sample_id = make_sample_id(config, family, seed)
    rules = spec.get("index_rules", {})
    derived = {"seed_index": index}
    kind = spec["builder"]
    version = spec.get("evaluator_version")
    v = values
    if kind == "fixed_workpiece":
        wp = spec["workpiece"]
        dims, pose = (wp["length_m"], wp["width_m"], wp["height_m"]), (0., 0., 0.)
    elif kind == "fixed_workpiece_posed":
        wp = spec["workpiece"]
        dims, pose = (wp["length_m"], wp["width_m"], wp["height_m"]), (v["x_m"], v["y_m"], v["yaw_deg"])
    else:
        dims, pose = _dims(config, v), (v["x_m"], v["y_m"], v["yaw_deg"])
        derived["applied_dims_m"] = list(dims)
    L, W, H = dims
    applied = {"length_m": L, "width_m": W, "height_m": H}
    if kind == "fixed_workpiece":
        scene_rows = []
    elif kind == "fixed_workpiece_posed":
        scene_rows = _scene_rows(seed, v, _POSE_PATHS[3:], specs)
    else:
        scene_rows = _scene_rows(seed, v, _POSE_PATHS, specs, applied)
    protrusions, slots, planner = [], [], {}

    if kind == "cut":
        angle = rules["cut_angle_deg"][index % len(rules["cut_angle_deg"])]
        x_c = _node_x(config, v["cut_fraction"]*L) if angle == 0 else v["cut_fraction"]*L
        plane = _vertical_plane(x_c) if angle == 0 else _oblique_plane(x_c, H, angle)
        fields, fractions = _sever_fields(config, [plane], L)
        slots = [_relief_slot(config, plane, H, spec["slot_width_m"])]
        goal = _goal(config, f"{sample_id}_sever", "sever",
                     f"Sever at local x = {x_c*1e3:.2f} mm (mid-height), cut angle {angle} deg about local y.", fields)
        derived.update(cut_angle_deg=angle, cut_x_local_m=x_c, nominal_piece_fractions=fractions)
    elif kind == "trim":
        fam = rules["protrusion_family"][index % len(rules["protrusion_family"])]
        rake = rules["rake_deg"][(index//len(rules["protrusion_family"])) % len(rules["rake_deg"])]
        protrusions = [_bump(config, "p0", fam, v["bump_x_fraction"]*L, v["bump_y_m"], v["bump_radius_x_m"],
                             v["bump_radius_y_m"], v["bump_height_m"])]
        scene_rows += _scene_rows(seed, v, (("bump_y_m", "workpiece.protrusions.0.y_m"),
                                            ("bump_radius_x_m", "workpiece.protrusions.0.radius_x_m"),
                                            ("bump_radius_y_m", "workpiece.protrusions.0.radius_y_m"),
                                            ("bump_height_m", "workpiece.protrusions.0.height_m")), specs)
        planner = {"rake_deg": rake}
        goal = _goal(config, f"{sample_id}_trim", "remove_protrusions",
                     f"Remove the {fam} protrusion p0 with one surface_trim pass at rake {rake} deg.", _remove_fields(config),
                     planner)
        derived.update(protrusion_family=fam, rake_deg=rake, bump_x_local_m=protrusions[0]["x_m"])
    elif kind == "slice_push":
        x1, x2 = _node_x(config, -v["slice_left_offset_m"]), _node_x(config, v["slice_right_offset_m"])
        planes = [_vertical_plane(x1), _vertical_plane(x2)]
        fields, fractions = _sever_fields(config, planes, L)
        slots = [_relief_slot(config, p, H, spec["slot_width_m"]) for p in planes]
        push = deepcopy(spec["push"])
        push.update(type="move_piece", goal_id="push_right",
                    selector={"side_of_plane": {"origin_local_m": [x2, 0., 0.], "normal_local": [1., 0., 0.], "side": "positive"}},
                    direction_local=[1., 0., 0.])
        goal = _goal(config, f"{sample_id}_slice_push", "sequence",
                     f"Two through-cuts at local x = {x1*1e3:.2f} / {x2*1e3:.2f} mm, then a cut-disabled blade-face push "
                     f"of the +x piece by {push['distance_m']*1e3:.0f} mm.",
                     {"steps": [{"type": "sever", "goal_id": "slice_two", **fields}, push]})
        derived.update(cut_x_local_m=[x1, x2], nominal_piece_fractions=fractions,
                       wrist_depth_relocation="NOT_EXERCISED (ADR 0009 survey is a live-only feature)")
    elif kind in ("fixed_workpiece", "fixed_workpiece_posed"):
        wp = spec["workpiece"]
        variant = spec["variants"][index]
        derived.update(variant=variant)
        protrusions = [deepcopy(row) for row in wp["protrusions"]]
        x_c = wp["cut_x_m"]
        slots = [(wp["relief_slot"]["x_m"], wp["relief_slot"]["width_m"])]
        if variant["op"] == "sever":
            angle = variant.get("cut_angle_deg", 0.)
            plane = _vertical_plane(x_c) if angle == 0 else _oblique_plane(x_c, H, angle)
            if kind == "fixed_workpiece_posed" and wp["relief_slot"].get("rule") == "under_plane":
                slots = [_relief_slot(config, plane, H, wp["relief_slot"]["width_m"])]
                derived.update(relief_slot_x_local_m=slots[0][0])
            if "approach_local" in variant:
                plane["approach_local"] = variant["approach_local"]
            fields, fractions = _sever_fields(config, [plane], L)
            goal = _goal(config, f"{sample_id}_sever", "sever", variant["text"], fields)
            derived.update(nominal_piece_fractions=fractions)
        else:
            planner = {key: variant[key] for key in ("rake_deg", "skew_deg", "skim_direction_local") if key in variant}
            goal = _goal(config, f"{sample_id}_trim", "remove_protrusions", variant["text"], _remove_fields(config), planner)
    elif kind == "multi_slice":
        k = rules["slices"][index % len(rules["slices"])]
        jitter = [v[f"position_jitter_{i}_m"] for i in range(max(rules["slices"]))][:k]
        positions = [-L/2+L*(i+1)/(k+1)+jitter[i] for i in range(k)]
        slots = [(x, spec["slot_width_m"]) for x in positions]
        goal = _goal(config, f"{sample_id}_divide{k+1}", "divide",
                     f"Divide into {k+1} pieces with {k} through-cuts at local x = "
                     + ", ".join(f"{x*1e3:.2f}" for x in positions)+" mm (one digit each).",
                     {"pieces": k+1, "axis_local": [1., 0., 0.], "positions_m": positions,
                      "minimum_main_parts_mass_fraction": config["common"]["sever_rule"]["minimum_main_parts_mass_fraction"],
                      "plane_error_bound_m": config["common"]["sever_rule"]["plane_error_bound_m"]})
        derived.update(slices=k, cut_x_local_m=positions)
    elif kind == "three_protrusions":
        fams = list(rules["protrusion_family_cycle"])
        offset = index % len(fams)
        centres = spec["protrusion_x_fractions"]
        for i, centre in enumerate(centres):
            fam = fams[(offset+i) % len(fams)]
            protrusions.append(_bump(config, f"p{i}", fam, centre*L+v[f"p{i}_x_jitter_m"], v[f"p{i}_y_m"],
                                     v[f"p{i}_radius_x_m"], v[f"p{i}_radius_y_m"], v[f"p{i}_height_m"]))
            scene_rows += _scene_rows(seed, v, ((f"p{i}_y_m", f"workpiece.protrusions.{i}.y_m"),
                                                (f"p{i}_radius_x_m", f"workpiece.protrusions.{i}.radius_x_m"),
                                                (f"p{i}_radius_y_m", f"workpiece.protrusions.{i}.radius_y_m"),
                                                (f"p{i}_height_m", f"workpiece.protrusions.{i}.height_m")), specs)
        planner = {"rake_deg": spec["rake_deg"]}
        goal = _goal(config, f"{sample_id}_remove3", "remove_protrusions",
                     f"Remove protrusions p0, p1, p2 one at a time (one surface_trim pass and digit each, rake "
                     f"{spec['rake_deg']} deg).", _remove_fields(config), planner)
        derived.update(protrusion_families=[p["family"] for p in protrusions])
    elif kind == "random_mixed":
        choice = spec["goal_kinds"][int(rng.integers(len(spec["goal_kinds"])))]
        n_bumps = int(rng.integers(1, spec["max_protrusions"]+1))
        families = [spec["protrusion_families"][int(rng.integers(len(spec["protrusion_families"])))] for _ in range(n_bumps)]
        angle = spec["cut_angles_deg"][int(rng.integers(len(spec["cut_angles_deg"])))]
        rake = spec["rakes_deg"][int(rng.integers(len(spec["rakes_deg"])))]
        selected = int(rng.integers(n_bumps))
        derived.update(goal_kind=choice, protrusions=n_bumps, protrusion_families=families, cut_angle_deg=angle,
                       rake_deg=rake, selected_index=selected,
                       discrete_rng="rng.integers after the uniform draws, in the order goal kind, protrusion count, "
                                    "families, cut angle, rake, selected protrusion")
        bump_x = [v["p0_x_fraction"]*L, v["p1_x_fraction"]*L]
        for i in range(n_bumps):
            protrusions.append(_bump(config, f"p{i}", families[i], bump_x[i], v[f"p{i}_y_m"], v[f"p{i}_radius_m"],
                                     v[f"p{i}_radius_m"], v[f"p{i}_height_m"]))
        x_c = _node_x(config, v["cut_fraction"]*L) if angle == 0 else v["cut_fraction"]*L
        plane = _vertical_plane(x_c) if angle == 0 else _oblique_plane(x_c, H, angle)
        slots = [_relief_slot(config, plane, H, spec["slot_width_m"])]
        sever, fractions = _sever_fields(config, [plane], L)
        remove_all = _remove_fields(config)
        if choice == "sever":
            goal = _goal(config, f"{sample_id}_sever", "sever", f"Sever at local x = {x_c*1e3:.2f} mm, angle {angle} deg.", sever)
        elif choice == "remove_all":
            goal = _goal(config, f"{sample_id}_remove_all", "remove_protrusions",
                         f"Remove all {n_bumps} protrusions (rake {rake} deg).", remove_all, {"rake_deg": rake})
        elif choice == "remove_selected":
            goal = _goal(config, f"{sample_id}_remove_p{selected}", "remove_protrusions",
                         f"Remove protrusion p{selected} only and keep the others (rake {rake} deg).",
                         _remove_fields(config, [f"p{selected}"]), {"rake_deg": rake})
        elif choice == "sever_then_remove_all":
            goal = _goal(config, f"{sample_id}_sever_remove", "sequence",
                         f"Sever at local x = {x_c*1e3:.2f} mm (angle {angle} deg), then remove all {n_bumps} protrusions "
                         f"(rake {rake} deg).",
                         {"steps": [{"type": "sever", "goal_id": "sever", **sever},
                                    {"type": "remove_protrusions", "goal_id": "remove_all", **remove_all}]},
                         {"rake_deg": rake})
        else:
            raise ValueError(f"Unknown Test D goal kind {choice!r}")
        derived.update(cut_x_local_m=x_c, nominal_piece_fractions=fractions)
    elif kind == "sequence_mixed":
        pattern = spec["step_patterns"][index % len(spec["step_patterns"])]
        families = [spec["protrusion_families"][int(rng.integers(len(spec["protrusion_families"])))] for _ in range(2)]
        rake = spec["rakes_deg"][index % len(spec["rakes_deg"])]
        derived.update(step_pattern=pattern, protrusion_families=families, rake_deg=rake,
                       discrete_rng="rng.integers after the uniform draws: the two protrusion families, in order")
        for i in range(2):
            protrusions.append(_bump(config, f"p{i}", families[i], v[f"p{i}_x_fraction"]*L, v[f"p{i}_y_m"],
                                     v[f"p{i}_radius_m"], v[f"p{i}_radius_m"], v[f"p{i}_height_m"]))
            scene_rows += _scene_rows(seed, v, ((f"p{i}_y_m", f"workpiece.protrusions.{i}.y_m"),
                                                (f"p{i}_radius_m", f"workpiece.protrusions.{i}.radius_x_m"),
                                                (f"p{i}_radius_m", f"workpiece.protrusions.{i}.radius_y_m"),
                                                (f"p{i}_height_m", f"workpiece.protrusions.{i}.height_m")), specs)
        steps, cuts, texts = [], [], []
        for token in pattern:
            op, _, arg = token.partition(":")
            if op == "sever":
                angle = float(arg)
                x_c = _node_x(config, v["cut_fraction"]*L) if angle == 0 else v["cut_fraction"]*L
                plane = _vertical_plane(x_c) if angle == 0 else _oblique_plane(x_c, H, angle)
                fields, fractions = _sever_fields(config, [plane], L)
                slots.append(_relief_slot(config, plane, H, spec["slot_width_m"]))
                steps.append({"type": "sever", "goal_id": "sever", **fields})
                cuts.append({"cut_angle_deg": angle, "cut_x_local_m": x_c, "nominal_piece_fractions": fractions})
                texts.append(f"sever at local x = {x_c*1e3:.2f} mm (angle {angle:g} deg)")
            elif op == "remove":
                select = "all" if arg == "all" else [arg]
                steps.append({"type": "remove_protrusions", "goal_id": f"remove_{arg}", **_remove_fields(config, select)})
                texts.append(f"remove {'all protrusions' if arg == 'all' else arg + ' (keep the other)'}")
            elif op == "trim":
                bump = protrusions[int(arg[1:])]
                margin = spec["trim_region_margin_m"]
                region = [bump["x_m"]-bump["radius_x_m"]-margin, bump["x_m"]+bump["radius_x_m"]+margin]
                rule = config["common"]["remove_rule"]
                steps.append({"type": "trim_plane", "goal_id": f"trim_{arg}",
                              "plane": {"origin_local_m": [0., 0., H], "normal_local": [0., 0., 1.]},
                              "region_local_m": {"x": region}, "edge_clearance_m": rule["edge_clearance_m"],
                              "minimum_removed_fraction": rule["minimum_removed_fraction"],
                              "minimum_substrate_share": rule["minimum_substrate_share"],
                              "maximum_detached_below_fraction": rule["maximum_detached_below_fraction"]})
                texts.append(f"trim the plane z = H over local x [{region[0]*1e3:.1f}, {region[1]*1e3:.1f}] mm ({arg})")
            else:
                raise ValueError(f"Unknown Test D2 step {token!r}")
        derived.update(sever=cuts)
        goal = _goal(config, f"{sample_id}_{'_'.join(t.replace(':', '') for t in pattern)}", "sequence",
                     "; then ".join(texts)+f" (rake {rake:g} deg).", {"steps": steps}, {"rake_deg": rake}, version)
    else:
        raise ValueError(f"Unknown builder {kind!r}")
    if version is not None and goal.get("evaluator_version") != version:
        goal["evaluator_version"] = version

    scene = _scene(config, sample_id, dims, pose, protrusions, slots, scene_rows,
                   f"Phase E {config['common'].get('round_label', 'round 1')} sample {family} seed {seed} "
                   f"({spec['description']})")
    return {"family": family, "seed": int(seed), "sample_id": sample_id, "scene": scene, "goal": goal,
            "declared": int(seed) in spec["seeds"], "draws": rows, "derived": derived, "rng": config["common"]["rng"].replace("salt", str(spec["salt"]))}


# ----------------------------------------------------------------------------- compile

def compile_sample(config, sample):
    """(ConfiguredTaskSpec, info) with the declared solver and duration rules applied.

    Solver: ``tlmpm_v4`` with max_cuts = digits when the plan uses <= 2 cut digits,
    else ``tlmpm_v5`` with max_cuts = digits. Duration: the ideal plan end (first
    DONE knot of a probe compile) + settle, rounded up. Raises the scene_config /
    planner errors unchanged (the caller classifies them).
    """
    from . import goal_planner
    from .scene_config import compile_task, resolve_scene
    from .tool_profile import PROFILES
    common = config["common"]
    scene, goal = deepcopy(sample["scene"]), deepcopy(sample["goal"])
    resolved = resolve_scene(scene)
    _, report = goal_planner.plan_with_report(resolved, goal, PROFILES[resolved["tool"]["profile_id"]])
    digits = int(report["digits_used"])
    override = config["families"].get(sample.get("family"), {}).get("solver")
    if override is not None:  # round 2 opt-in: e.g. tlmpm_v5 with slots_per_node 16 for every digit count
        scene["solver"].update(module=override["module"], max_cuts=max(1, digits))
        if "slots_per_node" in override:
            scene["solver"]["slots_per_node"] = int(override["slots_per_node"])
    elif digits <= 2:
        scene["solver"].update(module="tlmpm_v4", max_cuts=max(1, digits))
    else:
        scene["solver"].update(module="tlmpm_v5", max_cuts=digits)
    probe = compile_task(scene, goal, seed=common["material_seed"])
    labels = list(probe.reference_phase_labels)
    times = probe._times()
    done = next((float(t) for t, label in zip(times, labels) if label == "DONE"), None)
    if done is None:
        raise RuntimeError("The probe compile has no DONE knot")
    rule = common["duration_rule"]
    step = float(rule["round_up_to_s"])
    duration = math.ceil((done+float(rule["settle_after_plan_s"]))/step-1e-9)*step
    goal["episode"]["duration_s"] = float(duration)
    task = compile_task(scene, goal, seed=common["material_seed"])
    extra = {"slots_per_node": scene["solver"]["slots_per_node"]} if "slots_per_node" in scene["solver"] else {}
    return task, {**extra, "digits_used": digits, "solver_module": scene["solver"]["module"], "max_cuts": scene["solver"]["max_cuts"],
                  "plan_end_s": done, "duration_s": float(duration), "plan_report_operations": report["operations"],
                  "phases": report["phases"], "scene": scene, "goal": goal}


def history_stride(config, steps):
    rule = config["common"]["history"]
    return max(int(rule["minimum_stride"]), int(math.ceil(steps/float(rule["target_frames"]))))


# ----------------------------------------------------------------------------- stage-1 preflight

def _removed_sets(labels, protrusion_ids, aliases=None):
    """Per plan sample: protrusion ids removed by earlier remove operations of the same plan (ordered).

    ``aliases`` (staged_v2 only) maps other operation names (trim_plane passes) to the protrusion ids they
    remove; without it only operations named after a protrusion remove one (staged_v1, round 1).
    """
    ops = [label.split(":", 1)[1] if ":" in label else None for label in labels]
    first = {}
    for i, op in enumerate(ops):
        if op is not None:
            first.setdefault(op, i)
    removes = {op: ((op,) if op in protrusion_ids else tuple((aliases or {}).get(op, ()))) for op in first}
    order = sorted((op for op in first if removes[op]), key=first.get)

    def ids(ops_done):
        out = []
        for op in ops_done:
            out += [p for p in removes[op] if p not in out]
        return tuple(out)
    out = []
    for i, (label, op) in enumerate(zip(labels, ops)):
        if label.split(":", 1)[0] in ("RETURN_HOME", "DONE"):
            out.append(ids(order))
        elif op is None:
            out.append(ids([o for o in order if first[o] < i]))
        else:
            out.append(ids([o for o in order if first[o] < first[op]]))
    return out


def trim_aliases(task):
    """{trim op name: protrusion ids whose centre lies in a trim_plane step's region at or below the base top}."""
    from .goals import step_digit_allocation
    goal, scene = task.goal, task.scene
    steps = goal["steps"] if goal["type"] == "sequence" else [goal]
    digits = step_digit_allocation(goal, scene) if goal["type"] == "sequence" else [[0]]
    base = scene["workpiece"]["base_box_m"]["height_m"]
    out = {}
    for step, row in zip(steps, digits):
        if step["type"] != "trim_plane" or step["plane"]["origin_local_m"][2] > base+1e-9:
            continue
        region = step["region_local_m"] or {}
        out[f"trim{row[0]}"] = tuple(p["id"] for p in scene["workpiece"]["protrusions"]
                                     if all(region[a][0] <= p[f"{a}_m"] <= region[a][1] for a in region))
    return out


def _variant_task(task, removed):
    from dataclasses import replace
    keep = tuple(b for pid, b in zip(task.protrusion_ids, task.geometry.bumps) if pid not in removed)
    return task if len(keep) == len(task.geometry.bumps) else replace(task, geometry=replace(task.geometry, bumps=keep))


def staged_preflight(task, checker=None, *, until_phase=None, geometry="staged_v1"):
    """Stage-1 motion preflight (round file ``stages.stage1_geometry``).

    Scenes with at most one protrusion, or plans without remove operations, use
    ``motion_preflight.preflight_plan`` unchanged on the declared geometry. Otherwise the
    same dense IK chain is checked in contiguous chunks; each chunk uses the declared
    workpiece without the protrusions removed by earlier remove operations of the plan
    (their chips are loose material, which only the physics stage models). Joint
    continuity across chunk boundaries is checked on the whole chain.

    ``until_phase`` (default None: the full plan, as in round 1) checks only the plan prefix
    through that phase, exactly as ``motion_preflight.preflight_plan(until_phase=...)`` does.

    ``geometry="staged_v2"`` (round 2 opt-in; round 1 is ``staged_v1``): every scene with two or
    more protrusions is staged (also before any removal), trim_plane passes remove the protrusions
    whose centre lies in their region (``trim_aliases``), and a chunk with two or more remaining
    protrusions is checked against each remaining protrusion's own C2 hull in turn (one checker per
    single-protrusion variant; the chunk passes only if every variant passes and the worst slack is
    the minimum over them). That is blade/robot vs the union of per-protrusion hulls instead of the
    C2 model's one hull over all protrusions, which bridges the gaps between them.
    """
    if geometry not in ("staged_v1", "staged_v2"):
        raise ValueError("geometry must be staged_v1 or staged_v2")
    from . import motion_preflight as mp
    checker = checker or mp.MotionChecker(task)
    q_home = task.home["joint_positions"]
    poses, labels, armed = mp.plan_samples(task, until_phase=until_phase)
    v2 = geometry == "staged_v2"
    removed = _removed_sets(labels, set(task.protrusion_ids), trim_aliases(task) if v2 else None)
    if len(task.protrusion_ids) <= 1 or (not v2 and not any(removed)):
        receipt = (mp.preflight_plan(task, q_home, checker=checker) if until_phase is None else
                   mp.preflight_plan(task, q_home, until_phase=until_phase, checker=checker))
        return {**receipt, "geometry": "declared (motion_preflight.preflight_plan unchanged)"}, {(): checker}
    import time
    start = time.perf_counter()
    th = checker.thresholds
    qs, seed, failure = [], np.asarray(q_home, float), None
    ik_worst = {"position_error_m": 0., "rotation_error_rad": 0.}
    for i, (pose, flag) in enumerate(zip(poses, armed)):
        q, info = checker.solve_ik(pose, seed, flag)
        tol_p = th["ik_active_position_m"] if flag else th["ik_position_m"]
        tol_r = th["ik_active_rotation_rad"] if flag else th["ik_rotation_rad"]
        ik_worst = {key: max(ik_worst[key], float(info[key])) for key in ik_worst}
        if info["position_error_m"] > tol_p or info["rotation_error_rad"] > tol_r:
            failure = {"sample": i, "label": labels[i], "ik_info": info, "position_tolerance_m": tol_p,
                       "rotation_tolerance_rad": tol_r}
            break
        qs.append(q)
        seed = q
    receipt = {"samples": int(len(poses)), **({} if until_phase is None else {"until_phase": until_phase}),
               "ik_worst": ik_worst, "q_home": list(map(float, q_home)),
               "phases": sorted(set(label.split(":", 1)[0] for label in labels)),
               "geometry": ("staged: declared workpiece minus protrusions removed by earlier remove operations" if not v2 else
                            "staged_v2: declared workpiece minus protrusions removed by earlier remove/trim operations; "
                            "union of per-protrusion C2 hulls")}
    if failure is not None:
        return {**receipt, "status": "FAIL", "reason": "IK_FAILURE", "ik_failure": failure,
                "wall_s": time.perf_counter()-start}, {(): checker}
    qs = np.asarray(qs)
    modes = [mp.phase_mode(label, flag) for label, flag in zip(labels, armed)]
    checkers, chunks, worst, failing, single = {(): checker}, [], {}, set(), {}
    i = 0
    while i < len(qs):
        j = i
        while j < len(qs) and removed[j] == removed[i]:
            j += 1
        key = removed[i]
        if key not in checkers:
            checkers[key] = mp.MotionChecker(_variant_task(task, set(key)))
        remaining = [p for p in task.protrusion_ids if p not in key]
        if v2 and len(remaining) >= 2:
            variants = []
            for keep in remaining:
                if keep not in single:
                    single[keep] = mp.MotionChecker(_variant_task(task, set(task.protrusion_ids)-{keep}))
                variants.append((keep, single[keep].check_joint_path(qs[i:j], modes[i:j], labels[i:j])))
        else:
            variants = [(None, checkers[key].check_joint_path(qs[i:j], modes[i:j], labels[i:j]))]
        chunk_failing = sorted(set().union(*(set(part["failing"]) for _, part in variants)))
        chunks.append({"removed_protrusions": list(key), "samples": [i, j], "status": "PASS" if not chunk_failing else "FAIL",
                       "failing": chunk_failing, **({"per_protrusion_hull": [keep for keep, _ in variants]} if v2 and
                                                    variants[0][0] is not None else {})})
        for keep, part in variants:
            for name, row in part["worst"].items():
                row = {**row, "sample": row["sample"]+i, "removed_protrusions": list(key),
                       **({"hull_of": keep} if keep is not None else {})}
                if name not in worst or row["slack"] < worst[name]["slack"]:
                    worst[name] = row
            failing |= set(part["failing"])
        i = j
    step = np.abs(np.diff(qs, axis=0)).max(1) if len(qs) > 1 else np.zeros(0)
    if len(step):
        k = int(np.argmax(step))
        slack = th["joint_step_rad"]-float(step[k])
        if "joint_continuity" not in worst or slack < worst["joint_continuity"]["slack"]:
            worst["joint_continuity"] = {"slack": slack, "value": -float(step[k]), "threshold": -th["joint_step_rad"],
                                         "sample": k+1, "label": labels[k+1], "whole_chain": True}
        if slack < 0:
            failing.add("joint_continuity")
    failing = sorted(failing)
    binding = failing[0] if failing else (min(worst, key=lambda n: worst[n]["slack"]) if worst else None)
    path = {"status": "PASS" if not failing else "FAIL", "failing": failing, "binding": binding, "worst": worst,
            "samples": int(len(qs)), "chunks": chunks, "thresholds": th,
            "checker": "C2 GeometryModel per chunk (scripts/preflight_wp11_orientations.py via motion_preflight)"}
    return {**receipt, "status": path["status"], "reason": None if path["status"] == "PASS" else "SCENE_INFEASIBLE",
            "path": path, "q_until_end": qs[-1].tolist(), "wall_s": time.perf_counter()-start}, checkers


def final_geometry_checker(task, checkers):
    """The checker of the geometry after the last remove operation (for the informational return-home plan)."""
    return checkers[max(checkers, key=len)]


# ----------------------------------------------------------------------------- classification

def classify_preflight(receipt):
    """(class, reason) of a stage-1 preflight receipt (``motion_preflight.preflight_plan``)."""
    if receipt["status"] == "PASS":
        return None, None
    if receipt.get("reason") == "IK_FAILURE":
        failure = receipt.get("ik_failure", {})
        return "IK_FAILURE", f"ik_pose_error at sample {failure.get('sample')} ({failure.get('label')})"
    failing = list(receipt.get("path", {}).get("failing", []))
    if failing and set(failing) <= set(IK_PATH_CHECKS):
        return "IK_FAILURE", "joint path: " + ", ".join(failing)
    worst = receipt.get("path", {}).get("worst", {})
    detail = ", ".join(f"{name} (slack {worst[name]['slack']*1e3:.2f} mm/mrad at {worst[name].get('label')}, "
                       f"pair {worst[name].get('pair')})" for name in failing if name in worst)
    return "SCENE_INFEASIBLE", "collision/clearance: "+detail


def classify_material(run):
    """(class, reason) of a stage-2 kinematic run result (``run_generalization.material`` result.json)."""
    if run is None or run.get("status") == "NOT_RUN":
        return "NOT_RUN", (run or {}).get("reason", "no stage-2 result")
    if run.get("solver_invalid"):
        return "CUTTING_FAILURE", f"solver state invalid: {run['solver_invalid']}"
    if not run.get("policy_done", False):
        return "CUTTING_FAILURE", "the scripted plan did not finish within the episode"
    verdict = run.get("evaluation", {}).get("verdict")
    if verdict == "PASS":
        return "TASK_SUCCESS", "goal evaluator PASS (kinematic ideal tracking)"
    if verdict == "FAIL_INTEGRITY":
        failed = [k for k, ok in run["evaluation"]["integrity"]["checks"].items() if not ok]
        return "CUTTING_FAILURE", "integrity: "+", ".join(failed)
    if verdict == "FAIL":
        return "CUTTING_FAILURE", "outcome: "+", ".join(failed_checks(run["evaluation"]))
    return "NOT_RUN", f"goal evaluator verdict {verdict!r}"


def failed_checks(evaluation):
    outcome = evaluation.get("outcome", {})
    if "steps" in outcome:
        return [f"step{row['index']}:{row['goal_id']}:{name}" for row in outcome["steps"]
                for name, ok in row["outcome"].get("checks", {}).items() if not ok]
    return [name for name, ok in outcome.get("checks", {}).items() if not ok]


# ----------------------------------------------------------------------------- metrics

def _sever_metrics(outcome, quality, goal):
    m = outcome.get("metrics", {})
    expected = goal.get("expected_pieces", goal.get("pieces"))
    errors = []
    for row in quality.get("plane_location", []):
        if "frozen_2mm_statistic_m" not in row:
            continue
        stat = row["frozen_2mm_statistic_m"]
        if stat is None:
            errors.append(None)
        elif row["grid_aligned"]:
            errors.append(stat)
        else:
            ideal = row["ideal_geometric_label"]["maximum_reference_midpoint_plane_distance_m"]
            errors.append(None if ideal is None else stat-ideal)
    return {"cut_completion": min(1., m.get("substantial_pieces", 0)/expected) if expected else None,
            "separation": bool(outcome.get("checks", {}).get("expected_substantial_pieces")),
            "mechanical_components": m.get("mechanical_components"), "expected_pieces": expected,
            "cut_position_error_m": None if not errors or None in errors else max(errors),
            "overcut_debris_mass_fraction": m.get("minor_pieces_mass_fraction"),
            "definitions": {"cut_completion": "substantial pieces / expected pieces (capped at 1)",
                            "cut_position_error_m": "max over planes: frozen 2 mm midpoint statistic (grid-aligned) or "
                                                    "actual - ideal-geometric-label statistic (oblique, ADR 0006 Am. 1)",
                            "overcut_debris_mass_fraction": "mass outside the expected main pieces"}}


def _remove_metrics(outcome, quality):
    m = outcome.get("metrics", {})
    items = [row for row in m.get("items", []) if row["selected"]]
    removed = [row["removed_mass_fraction"] for row in items]
    residual = [row["residual_substrate_height_m"] for row in items if row["residual_substrate_height_m"] is not None]
    return {"cut_completion": None if not removed or None in removed else min(removed),
            "per_protrusion_removed": {row["id"]: row["removed_mass_fraction"] for row in m.get("items", [])},
            "separation": bool(outcome.get("checks", {}).get("selected_protrusions_removed"))
                          and bool(outcome.get("checks", {}).get("single_substrate_component",
                                   outcome.get("checks", {}).get("single_substrate_component_per_prior_piece"))),
            "mechanical_components": m.get("mechanical_components"),
            "cut_position_error_m": max(residual) if residual else None,
            "overcut_detached_below_plane_fraction": m.get("detached_below_plane_mass_fraction"),
            "substrate_share": m.get("largest_component_below_plane_mass_share"),
            "definitions": {"cut_completion": "min removed mass fraction over selected protrusions",
                            "cut_position_error_m": "max residual substrate particle-centre height above the plane in "
                                                    "the selected footprints (reported quality)",
                            "overcut_detached_below_plane_fraction": "below-plane mass detached from the substrate"}}


def _trim_metrics(outcome, quality):
    m, checks = outcome.get("metrics", {}), outcome.get("checks", {})
    pieces = m.get("prior_pieces") or []
    detached = (m.get("detached_below_plane_mass_fraction") if not pieces else
                max((r["detached_below_plane_mass_fraction"] or 0.) for r in pieces))
    return {"cut_completion": m.get("removed_above_plane_mass_fraction"),
            "separation": bool(checks.get("material_above_plane_removed"))
                          and bool(checks.get("single_substrate_component", checks.get("single_substrate_component_per_prior_piece"))),
            "mechanical_components": m.get("mechanical_components"),
            "cut_position_error_m": quality.get("current_substrate_particle_center_max_residual_m"),
            "overcut_detached_below_plane_fraction": detached,
            "definitions": {"cut_completion": "removed above-plane mass fraction in the region",
                            "cut_position_error_m": "max residual substrate particle-centre height above the plane in "
                                                    "the region (reported quality)"}}


def _move_metrics(outcome):
    m = outcome.get("metrics", {})
    return {"push_displacement_m": m.get("displacement_along_direction_m"),
            "other_piece_displacement_max_m": max(m.get("other_piece_displacement_norm_m") or [0.]),
            "label_changes_while_disarmed_frames": m.get("label_changes_while_disarmed_frames"),
            "separation": bool(outcome.get("checks", {}).get("selected_piece_displaced"))}


def episode_metrics(goal, evaluation):
    """Uniform report metrics from a goal evaluation (read-only; never part of a verdict)."""
    outcome, quality = evaluation.get("outcome", {}), evaluation.get("quality", {})
    kind = goal["type"]
    if kind in ("sever", "divide"):
        return _sever_metrics(outcome, quality, goal)
    if kind == "remove_protrusions":
        return _remove_metrics(outcome, quality)
    if kind == "sequence":
        steps = []
        for row, step in zip(outcome.get("steps", []), goal["steps"]):
            q = quality.get(f"{row['index']}:{row['goal_id']}", {})
            if step["type"] in ("sever", "divide"):
                steps.append(_sever_metrics(row["outcome"], q, step))
            elif step["type"] == "remove_protrusions":
                steps.append(_remove_metrics(row["outcome"], q))
            elif step["type"] == "trim_plane":
                steps.append(_trim_metrics(row["outcome"], q))
            elif step["type"] == "move_piece":
                steps.append(_move_metrics(row["outcome"]))
        completions = [s["cut_completion"] for s in steps if "cut_completion" in s]
        errors = [s["cut_position_error_m"] for s in steps if s.get("cut_position_error_m") is not None]
        return {"steps": steps, "cut_completion": min(completions) if completions and None not in completions else None,
                "separation": all(s.get("separation") for s in steps),
                "cut_position_error_m": max(errors) if errors else None}
    return {}


# ----------------------------------------------------------------------------- summary

def summarize(rows, families):
    """Per family/test: total, class counts, feasible fraction, success among feasible, overall (no exclusion)."""
    out = {}
    for name in families:
        group = [r for r in rows if r["family"] == name]
        counts = {c: sum(r["class"] == c for r in group) for c in CLASSES}
        total = len(group)
        feasible = total-counts["SCENE_INFEASIBLE"]-counts["IK_FAILURE"]
        success = counts["TASK_SUCCESS"]
        extra = {}
        if any(r.get("stage2_selected") is False for r in rows):  # round files with a stage-2 subset rule
            selected = sum(bool(r.get("stage2_selected")) for r in group)
            extra = {"stage2_selected": selected, "kinematic_task_success_among_stage2": success/selected if selected else None,
                     "not_run_not_selected": sum(r.get("stage2_selected") is False and r["class"] == "NOT_RUN" for r in group)}
        out[name] = {**extra, "samples": total, "counts": counts, "preflight_feasible": feasible,
                     "feasible_fraction": feasible/total if total else None,
                     "kinematic_task_success_among_feasible": success/feasible if feasible else None,
                     "overall_kinematic_success": success/total if total else None,
                     "not_run_among_feasible": counts["NOT_RUN"],
                     "rule": "feasible = samples - SCENE_INFEASIBLE - IK_FAILURE; NOT_RUN feasible samples stay in the "
                             "feasible denominator as non-successes; nothing is excluded"}
    return out
