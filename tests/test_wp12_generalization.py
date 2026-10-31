"""WP-12 Phase 5 generalization infrastructure (generalization_wp12): generators, feasibility, summaries. CPU only."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from robotic_cutting_sim import generalization_wp12 as W
from robotic_cutting_sim.goals import GOAL_TYPES, _TYPE_FIELDS, validate_goal
from robotic_cutting_sim.scene_config import SceneInfeasible, resolve_scene

PRE_D5_DRAFT = W.ROOT/"tests/fixtures/wp12_benchmark_v1_DRAFT_pre_d5.json"   # the DRAFT as committed in 73c6e43


@pytest.fixture(scope="module")
def config():
    return W.load_benchmark()


@pytest.fixture(scope="module")
def envelope():
    """The DECLARED envelope (xhigh decision D2)."""
    return W.load_envelope()


def _template_of(envelope):
    """A synthetic TEMPLATE built from the declared rules: same rules, every bound null (the pre-declaration state)."""
    env = deepcopy(envelope)
    env["status"] = "TEMPLATE"
    for rule in env["rules"]:
        for key in ("min", "max", "values"):
            if key in rule:
                rule[key] = None
    return env


@pytest.fixture(scope="module")
def template(envelope):
    return _template_of(envelope)


def _declared_envelope(**bounds):
    env = {"schema": W.ENVELOPE_SCHEMA, "envelope_id": "test", "status": "DECLARED", "rules": []}
    for rid, rule in bounds.items():
        env["rules"].append({"id": rid, **rule})
    return env


# ----------------------------------------------------------------------------- config

def test_draft_is_marked_and_thresholds_are_the_goal_defaults(config):
    assert config["status"] == "DRAFT" and "DRAFT" in config["benchmark_id"]
    assert "NOT FROZEN" in config["draft_notice"]
    snapshot = config["thresholds"]["snapshot"]
    current = {k: {kk: vv for kk, vv in v.items() if vv is not None and not isinstance(vv, (list, dict))}
               for k, v in _TYPE_FIELDS.items()}
    assert snapshot == current


def test_declared_envelope_invariants(envelope):
    """The file is the xhigh D2 declaration: DECLARED, every bound set, every rule with evidence and a note."""
    assert envelope["status"] == "DECLARED" and envelope["schema"] == W.ENVELOPE_SCHEMA
    assert "WP12_PHASE4_XHIGH_DECISION" in envelope["decision_ref"]
    ids = [r["id"] for r in envelope["rules"]]
    assert len(ids) == len(set(ids))
    for rule in envelope["rules"]:
        for key in W._NEEDED[rule["kind"]]:
            assert rule.get(key) is not None, (rule["id"], key)
        if rule["kind"] == "range":
            assert rule["min"] <= rule["max"], rule["id"]
        assert rule.get("evidence") and rule.get("note"), rule["id"]
    rules = {r["id"]: r for r in envelope["rules"]}
    kinds = set(rules["board_kinds"]["values"])
    assert kinds == {"solid", "slotted"}
    for rule in envelope["rules"]:
        for key, value in (rule.get("when") or {}).items():
            assert key == "board_kind" and value in kinds, rule["id"]
    # the decision's bounds (D2 table and the paragraph under it)
    assert rules["trim_rake_deg"]["max"] == 44.0 and rules["trim_rake_deg_min"]["min"] == 9.2
    assert rules["sever_tilt_solid_board"]["max"] == 9.9 and rules["sever_tilt_solid_board"]["when"] == {"board_kind": "solid"}
    assert rules["sever_tilt_slotted_board"]["max"] == 34.9 and rules["sever_tilt_slotted_board"]["when"] == {"board_kind": "slotted"}
    assert rules["trim_skew_abs_deg"]["max"] == 30.0 and rules["sever_plane_yaw_abs_deg"]["max"] == 35.0
    assert rules["sever_advance_tilt_abs_deg"]["max"] == 30.0 and rules["cut_digits_max"]["max"] == 5
    assert (rules["workpiece_height_m"]["min"], rules["workpiece_height_m"]["max"]) == (0.004, 0.035)
    assert (rules["trim_depth_m"]["min"], rules["trim_depth_m"]["max"]) == (0.0, 0.0)   # trim_layer: known failure
    assert "wp12_stiff_v1" not in rules["material_profiles"]["values"]
    assert "cutting_tasks_v1" in rules["material_profiles"]["values"]


def test_synthetic_template_has_fields_only(template, envelope):
    """The pre-declaration TEMPLATE state is still representable (built in-test from the declared rules)."""
    assert template["status"] == "TEMPLATE"
    assert [r["id"] for r in template["rules"]] == [r["id"] for r in envelope["rules"]]
    for rule in template["rules"]:
        for key in ("min", "max", "values"):
            if key in rule:
                assert rule[key] is None, rule["id"]


def test_every_declared_sample_builds_and_validates(config):
    n = 0
    for split in W.SPLITS:
        for family, rs, seed, _ in W.sample_ids(config, split):
            sample = W.build_sample(config, family, rs, seed, split=split)
            validate_goal(sample["goal"])
            assert sample["goal"]["type"] in GOAL_TYPES
            assert sample["goal"]["evaluator_version"] == "goal_eval_v2a"
            assert sample["scene"]["supports"]["kind"] == "flat_board"
            n += 1
    assert n == sum(len(W.seed_list(config, s["family"], split, s["range_set"]))
                    for s in W.strata(config) for split in W.SPLITS)


def test_strata_have_at_least_ten_benchmark_configurations(config):
    strata = W.strata(config)
    assert len(strata) >= 6
    for stratum in strata:
        assert len(W.seed_list(config, stratum["family"], "benchmark", stratum["range_set"])) >= 10
    bad = deepcopy(config)
    bad["seeds"]["counts"]["benchmark"]["OOD"] = 9
    with pytest.raises(W.BenchmarkConfigError, match="fewer than 10"):
        W.validate_benchmark(bad)


def test_native_subsampling_rule_needs_three_per_stratum(config):
    bad = deepcopy(config)
    bad["native"]["selection"]["k"] = 2
    with pytest.raises(W.BenchmarkConfigError, match="k >= 3"):
        W.validate_benchmark(bad)


# ----------------------------------------------------------------------------- reproducibility / ranges

def test_seed_reproducibility(config):
    a = W.build_sample(config, "E", "ID", 55000)
    b = W.build_sample(config, "E", "ID", 55000)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    c = W.build_sample(config, "E", "ID", 55001)
    assert a["draws"] != c["draws"]
    assert W.sample_by_id(config, a["sample_id"])["scene"] == a["scene"]
    held = W.build_sample(config, "A", "OOD", W.seed_list(config, "A", "held_out", "OOD")[0], split="held_out")
    assert W.sample_by_id(config, held["sample_id"])["goal"] == held["goal"]


def test_dimensions_are_independent_streams(config):
    seed = W.seed_list(config, "C", "benchmark", "ID")[3]
    base = {r["name"]: r["value"] for r in W.build_sample(config, "C", "ID", seed)["draws"]}
    reduced = deepcopy(config)
    reduced["families"]["C"]["randomize"].remove("skew_deg")
    reduced["families"]["C"]["randomize"].insert(0, "cut_speed_m_s")   # add one, remove one, reorder
    reduced["families"]["C"]["ranges"] = {"ID": {"cut_speed_m_s": {"uniform": [0.008, 0.012]}}}
    other = {r["name"]: r["value"] for r in W.build_sample(reduced, "C", "ID", seed)["draws"]}
    for name, value in base.items():
        if name != "skew_deg":
            assert other[name] == value, name


def _in_support(spec, value):
    kind, support = W._support(spec)
    values = value if isinstance(value, list) else [value]
    if kind == "set":
        return all(v in support for v in values)
    return all(any(lo-1e-12 <= v <= hi+1e-12 for lo, hi in support) for v in values)


def test_draws_respect_declared_ranges_and_ood_is_outside_id(config):
    for split in ("benchmark", "held_out", "dev"):
        for family, rs, seed, _ in W.sample_ids(config, split):
            sample = W.build_sample(config, family, rs, seed, split=split)
            ood = sample["ood_dimensions"]
            assert (len(ood) == 1) == (rs == "OOD")
            for row in sample["draws"]:
                assert _in_support(row["spec"], row["value"]), (sample["sample_id"], row["name"])
                expected = "OOD" if row["name"] in ood else "ID"
                assert row["range_set"] == expected
                if row["range_set"] == "OOD":
                    assert not _in_support(W.family_spec(config, family, row["name"], "ID"), row["value"])
                    assert row["applied"], "an OOD dimension must be one the variant reads"


def test_overlapping_ood_range_is_rejected(config):
    bad = deepcopy(config)
    bad["range_sets"]["OOD"]["rake_deg"] = {"uniform": [30.0, 50.0]}
    with pytest.raises(W.BenchmarkConfigError, match="overlaps"):
        W.validate_benchmark(bad)
    bad = deepcopy(config)
    bad["range_sets"]["OOD"]["material_profile"] = {"choice": ["cutting_tasks_v1", "wp12_firm_v1"]}
    with pytest.raises(W.BenchmarkConfigError, match="overlaps"):
        W.validate_benchmark(bad)


def test_held_out_dev_and_benchmark_seeds_are_disjoint(config):
    seen = {}
    for family in config["families"]:
        for split in W.SPLITS:
            for rs in W.RANGE_SETS:
                for seed in W.seed_list(config, family, split, rs):
                    assert seed not in seen, (seed, family, split, rs, seen.get(seed))
                    seen[seed] = (family, split, rs)
    bad = deepcopy(config)
    bad["seeds"]["split_offsets"]["held_out"] = 5   # overlaps benchmark seeds 5..9
    with pytest.raises(W.BenchmarkConfigError, match="Seed"):
        W.validate_benchmark(bad)


def test_coverage_reaches_every_variant_and_ood_dimension(config):
    for family, row in W.coverage(config).items():
        assert not row["variants_never_reached"], family
        assert not row["ood_dimensions_never_reached"], family


# ----------------------------------------------------------------------------- scene semantics

def test_board_is_co_rotated_and_workpiece_on_board(config):
    from robotic_cutting_sim.flat_board_workcell import board_top_world_z, resolve, validate_spec
    for family, rs, seed, _ in W.sample_ids(config, "dev"):
        scene = W.build_sample(config, family, rs, seed, split="dev")["scene"]
        board = scene["supports"]["workcell"]["overrides"]["cutting_board"]
        assert board["yaw_deg"] == scene["workpiece"]["world_pose"]["yaw_deg"]
        try:
            record = resolve(validate_spec(scene["supports"]["workcell"]))
        except ValueError:
            continue   # reported as scene_geometrically_infeasible by the classifier
        assert abs(scene["workpiece"]["world_pose"]["translation_m"][2]-board_top_world_z(record)) < 1e-12


def test_slotted_variant_only_where_declared_and_groove_under_the_plane(config):
    for family, rs, seed, _ in W.sample_ids(config, "benchmark"):
        sample = W.build_sample(config, family, rs, seed)
        board = sample["scene"]["supports"]["workcell"]["overrides"]["cutting_board"]
        if board.get("kind") == "slotted":
            fixed = W.VARIANT_BOARD_KIND.get((config["families"][family]["builder"], sample["variant"]))
            assert fixed == "slotted" or "board_kind" in config["families"][family]["randomize"], family
            if sample["variant"] == "sever":
                x = sample["derived"]["cut_x_local_m"]
                pose = sample["scene"]["workpiece"]["world_pose"]
                expected = W._board_frame_x(config["common"], (*pose["translation_m"][:2], pose["yaw_deg"]), x)
                assert abs(board["slots"][0]["x_board_m"]-expected) < 1e-6


def test_vertical_planes_are_on_node_planes(config):
    dx = config["common"]["solver"]["grid_dx_m"]
    for family in ("A", "D", "F", "G"):
        for seed in W.seed_list(config, family, "benchmark", "ID"):
            sample = W.build_sample(config, family, "ID", seed)
            goal = sample["goal"]
            steps = goal["steps"] if goal["type"] == "sequence" else [goal]
            for step in steps:
                for plane in step.get("planes", []):
                    if plane["normal_local"] == [1., 0., 0.]:
                        assert abs(plane["origin_local_m"][0]/dx-round(plane["origin_local_m"][0]/dx)) < 1e-9


def test_mixed_operations_map_to_existing_goal_types(config):
    for name, row in W.MIXED_OPERATIONS.items():
        family = row["family"]
        variants = config["families"][family]["variants"]
        if row.get("status") == "NOT_EXPRESSIBLE":
            assert row["missing"] in W.MISSING_FEATURES
            assert row["variant"] not in variants
            index = 0
            cfg = deepcopy(config)
            cfg["families"][family]["variants"] = [row["variant"]]
            with pytest.raises(W.NotExpressible):
                W.build_sample(cfg, family, "ID", 999999, split="dev", index=index)
            continue
        assert row["variant"] in variants, name
        index = variants.index(row["variant"])
        sample = W.build_sample(config, family, "ID", 990000+index, split="dev", index=index)
        assert sample["variant"] == row["variant"]
        goal = validate_goal(sample["goal"])
        assert goal["type"] == "sequence" or name in ("bumps_sequential",)
        for step in goal.get("steps", []):
            assert step["type"] in GOAL_TYPES[:-1]


def test_cut_reposition_cut_is_expressible_with_the_step_offset(config):
    """WP-12 Phase 6: sever x1 -> kerf_face push by d -> sever reference x2 with planner_offset_local_m [d, 0, 0]."""
    assert "planner_offset_local_m" in W.IMPLEMENTED_FEATURES and "planner_offset_local_m" not in W.MISSING_FEATURES
    assert "per_step_planner_options" in W.IMPLEMENTED_FEATURES and "per_step_planner_options" not in W.MISSING_FEATURES
    assert set(W.MISSING_FEATURES) == {"continuous_physics_overrides", "edge_lateral_offset"}
    assert W.MIXED_OPERATIONS["cut_reposition_cut"].get("status") != "NOT_EXPRESSIBLE"
    assert ("cut_push", "cut_reposition_cut") in W.UNUSED_DIMENSIONS
    variants = config["families"]["G"]["variants"]
    index = variants.index("cut_reposition_cut")
    dx = config["common"]["solver"]["grid_dx_m"]
    seeds = [s for i, s in enumerate(W.seed_list(config, "G", "benchmark", "ID")) if i % len(variants) == index]
    assert seeds
    for seed in seeds:
        sample = W.build_sample(config, "G", "ID", seed)
        assert sample["variant"] == "cut_reposition_cut" and not W.geometry_checks(sample, config["common"])
        goal = validate_goal(sample["goal"])
        assert [s["type"] for s in goal["steps"]] == ["sever", "move_piece", "sever"]
        first, push, second = goal["steps"]
        x1, x2 = first["planes"][0]["origin_local_m"][0], second["planes"][0]["origin_local_m"][0]
        L = sample["scene"]["workpiece"]["base_box_m"]["length_m"]
        assert x1 < x2 < L/2 and abs(x2-W._node_x(config["common"], (x1+L/2)/2)) < 1e-12
        assert abs(x2/dx-round(x2/dx)) < 1e-9 and abs(x1/dx-round(x1/dx)) < 1e-9
        assert second["planner_offset_local_m"] == [push["distance_m"], 0., 0.]
        assert push.get("push_mode", "kerf_face") == "kerf_face" and push["direction_local"] == [1., 0., 0.]
        assert "planner_offset_local_m" not in first and "planner_offset_local_m" not in push
        fractions = sample["derived"]["second_cut_nominal_piece_fractions"]
        assert second["minimum_part_mass_fraction"] == W._min_part(config["common"], fractions)
        params = sample["envelope_parameters"]
        assert params["cut_digits"] == 2 and params["operation_pattern"] == "sequence:sever>move_piece>sever"
        unused = {r["name"] for r in sample["draws"] if not r["applied"]}
        assert "push_distance_m" not in unused and "cut_fraction" not in unused and "rake_deg" in unused
    task, info = W.compile_sample(config, W.build_sample(config, "G", "ID", seeds[0]))
    assert info["digits_used"] == 2
    ops = info["plan_report_operations"]
    assert [op.get("planner_offset_local_m") is not None for op in ops] == [False, False, True]


def test_one_dev_sample_per_family_compiles(config):
    for family in config["families"]:
        seed = W.seed_list(config, family, "dev", "ID")[0]
        sample = W.build_sample(config, family, "ID", seed, split="dev")
        task, info = W.compile_sample(config, sample)
        assert task.solver_module == "tlmpm_v5" and info["max_cuts"] >= 1
        assert info["duration_s"] > info["plan_end_s"]
        resolve_scene(sample["scene"])


# ----------------------------------------------------------------------------- feasibility classifier

class _Task:
    grid_dx_m, particle_spacing_m = .004, .001


def _sample(config, family="C", rs="ID"):
    return W.build_sample(config, family, rs, W.seed_list(config, family, "dev", rs)[0], split="dev")


PASS = {"status": "PASS"}


def test_classifier_valid_feasible_and_pending_envelope(config, envelope, template):
    row = W.classify_feasibility(_sample(config), envelope, compile_fn=lambda s: (_Task(), {"digits_used": 1}),
                                 preflight_fn=lambda t: PASS, common=config["common"])
    assert row["feasibility_class"] == "valid_feasible"
    assert row["envelope_pending"] is False and row["checks"]["envelope"]["status"] == "SUPPORTED"
    statuses = {r["id"]: r["status"] for r in row["checks"]["envelope"]["rules"]}
    assert statuses["trim_rake_deg"] == "PASS" and statuses["board_kinds"] == "PASS"
    assert statuses["sever_tilt_solid_board"] == "NOT_APPLICABLE"   # a skim carries no sever tilt
    # the synthetic TEMPLATE (null bounds): nothing is evaluated and the sample is pending, never silently supported
    row = W.classify_feasibility(_sample(config), template, compile_fn=lambda s: (_Task(), {"digits_used": 1}),
                                 preflight_fn=lambda t: PASS, common=config["common"])
    assert row["feasibility_class"] == "valid_feasible"
    assert row["envelope_pending"] is True
    assert all(r["status"] in ("NOT_EVALUATED", "NOT_APPLICABLE") for r in row["checks"]["envelope"]["rules"])


def test_classifier_declared_envelope_board_dependent_tilt(config, envelope):
    """sever_tilt_* rules apply by board kind: 9.9 deg on the solid board, 34.9 deg on the explicit slotted variant."""
    sample = deepcopy(W.build_sample(config, "B", "OOD", W.seed_list(config, "B", "dev", "OOD")[0], split="dev"))
    for board, tilt, expected in (("solid", 5., "SUPPORTED"), ("solid", 12., "UNSUPPORTED"), ("slotted", 25., "SUPPORTED"),
                                  ("slotted", 35., "UNSUPPORTED")):
        params = {**sample["envelope_parameters"], "board_kind": board, "sever_tilt_abs_deg_max": tilt}
        env = W.evaluate_envelope(envelope, params)
        assert env["status"] == expected, (board, tilt, env["unsupported_rules"])
        rows = {r["id"]: r["status"] for r in env["rules"]}
        other = "sever_tilt_slotted_board" if board == "solid" else "sever_tilt_solid_board"
        assert rows[other] == "NOT_APPLICABLE"


def test_classifier_scene_geometrically_infeasible(config, template):
    sample = _sample(config, "A")
    sample["geometry"]["planes"][0]["origin_local_m"][0] = 1.0   # plane outside the workpiece (generator check)
    row = W.classify_feasibility(sample, template, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: PASS)
    assert row["feasibility_class"] == "scene_geometrically_infeasible"
    assert row["checks"]["scene_geometry"]["problems"][0][0] == "cut_plane_outside_workpiece"
    assert row["checks"]["preflight"]["status"] == "NOT_RUN"

    def infeasible(_):
        raise SceneInfeasible("workpiece_off_board", "outline leaves the board")
    row = W.classify_feasibility(_sample(config, "A"), template, compile_fn=infeasible, preflight_fn=lambda t: PASS)
    assert row["feasibility_class"] == "scene_geometrically_infeasible"
    assert row["checks"]["scene_geometry"]["stage"] == "scene"


def test_classifier_board_off_table_from_real_validation(config, template):
    sample = deepcopy(_sample(config, "A"))
    sample["scene"]["workpiece"]["world_pose"]["yaw_deg"] = 7.0
    sample["scene"]["supports"]["workcell"]["overrides"]["cutting_board"]["yaw_deg"] = 7.0
    row = W.classify_feasibility(sample, template, compile_fn=lambda s: W.compile_sample(config, s), preflight_fn=lambda t: PASS)
    assert row["feasibility_class"] == "scene_geometrically_infeasible"
    assert "table" in row["reason"]


def test_classifier_ik_and_collision(config, template):
    ik = {"status": "FAIL", "reason": "IK_FAILURE", "ik_failure": {"sample": 7, "label": "SKIM:p0"}}
    row = W.classify_feasibility(_sample(config), template, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: ik)
    assert row["feasibility_class"] == "ik_unreachable"
    joint = {"status": "FAIL", "reason": "SCENE_INFEASIBLE", "path": {"failing": ["joint_limit_margin"], "worst": {}}}
    row = W.classify_feasibility(_sample(config), template, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: joint)
    assert row["feasibility_class"] == "ik_unreachable"
    col = {"status": "FAIL", "reason": "SCENE_INFEASIBLE",
           "path": {"failing": ["blade_collision"], "worst": {"blade_collision": {"slack": -.001, "label": "PRE_CUT", "pair": ["a", "b"]}}}}
    row = W.classify_feasibility(_sample(config), template, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: col)
    assert row["feasibility_class"] == "collision_constrained"
    both = {"status": "FAIL", "path": {"failing": ["joint_continuity", "hand_table"], "worst": {}}}
    row = W.classify_feasibility(_sample(config), template, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: both)
    assert row["feasibility_class"] == "ik_unreachable"
    assert row["failing_classes"] == ["ik_unreachable", "collision_constrained"]


def test_classifier_physically_unsupported_and_precedence(config):
    env = _declared_envelope(rake={"parameter": "rake_deg_max", "kind": "max", "max": 44.0})
    sample = deepcopy(_sample(config))
    sample["envelope_parameters"]["rake_deg_max"] = 50.0
    ik = {"status": "FAIL", "reason": "IK_FAILURE", "ik_failure": {}}
    row = W.classify_feasibility(sample, env, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: ik)
    assert row["feasibility_class"] == "physically_unsupported"
    assert row["failing_classes"] == ["physically_unsupported", "ik_unreachable"]
    assert row["checks"]["preflight"]["status"] == "FAIL"   # reachability is still recorded
    row = W.classify_feasibility(sample, env, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: ik,
                                 precedence=("ik_unreachable", "physically_unsupported"))
    assert row["feasibility_class"] == "ik_unreachable"
    row = W.classify_feasibility(sample, env, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: PASS,
                                 run_preflight_when_unsupported=False)
    assert row["feasibility_class"] == "physically_unsupported" and row["checks"]["preflight"]["status"] == "NOT_RUN"
    sample["envelope_parameters"]["rake_deg_max"] = 30.0
    row = W.classify_feasibility(sample, env, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: PASS)
    assert row["feasibility_class"] == "valid_feasible" and not row["envelope_pending"]


def test_classifier_material_cfl_admission(config, template):
    sample = deepcopy(_sample(config, "A"))
    sample["envelope_parameters"]["material_profile"] = "wp12_stiff_v1"
    row = W.classify_feasibility(sample, template, compile_fn=lambda s: (_Task(), {}), preflight_fn=lambda t: PASS,
                                 common=config["common"])
    assert row["feasibility_class"] == "physically_unsupported"
    assert "material_cfl_admission" in row["reason"]


def test_classifier_errors_are_kept_not_classified(config, template):
    from robotic_cutting_sim.scene_config import SceneConfigError

    def bad(_):
        raise SceneConfigError("Unknown planner options ['typo']")
    row = W.classify_feasibility(_sample(config), template, compile_fn=bad, preflight_fn=lambda t: PASS)
    assert row["feasibility_class"] is None and row["classification_error"]

    def crash(_):
        raise RuntimeError("mujoco died")
    row = W.classify_feasibility(_sample(config), template, compile_fn=lambda s: (_Task(), {}), preflight_fn=crash)
    assert row["feasibility_class"] is None and "preflight" in row["reason"]
    assert W.final_row(row)["class"] == "not_run"


def test_envelope_rule_kinds():
    env = _declared_envelope(
        a={"parameter": "x", "kind": "max", "max": 1.0}, b={"parameter": "x", "kind": "min", "min": 0.0},
        c={"parameter": "y", "kind": "range", "min": 2.0, "max": 3.0}, d={"parameter": "k", "kind": "in_set", "values": ["solid"]},
        e={"parameter": "f", "kind": "subset", "values": ["dome", "gaussian"]},
        g={"parameter": "x", "kind": "max", "max": 0.1, "when": {"k": "slotted"}})
    ok = W.evaluate_envelope(env, {"x": .5, "y": 2.5, "k": "solid", "f": ["dome"]})
    assert ok["status"] == "SUPPORTED"
    assert {r["id"]: r["status"] for r in ok["rules"]}["g"] == "NOT_APPLICABLE"
    bad = W.evaluate_envelope(env, {"x": .5, "y": 2.5, "k": "slotted", "f": ["dome", "ridge"]})
    assert bad["status"] == "UNSUPPORTED" and set(bad["unsupported_rules"]) == {"d", "e", "g"}
    pending = W.evaluate_envelope({**env, "rules": env["rules"]+[{"id": "h", "parameter": "x", "kind": "max", "max": None}]},
                                  {"x": .5, "y": 2.5, "k": "solid", "f": ["dome"]})
    assert pending["status"] == "PENDING" and pending["pending_rules"] == ["h"]


def test_freeze_problems(config, envelope, template):
    problems = W.freeze_problems(config, template)
    assert any("TEMPLATE" in p for p in problems)
    assert sum("undeclared bound" in p for p in problems) == len(template["rules"])
    # the re-declared DRAFT (decision D5) is inside the declared envelope: no range or sample problem
    assert W.freeze_problems(config, envelope) == []
    tight = deepcopy(envelope)
    next(r for r in tight["rules"] if r["id"] == "trim_rake_deg")["max"] = 30.0
    problems = W.freeze_problems(config, tight)
    assert any("rake_deg" in p and "trim_rake_deg" in p for p in problems)
    assert any(p.startswith("ID sample P5_C_ID_") and "trim_rake_deg" in p for p in problems)


def test_freeze_problems_checks_when_rules(config, envelope):
    """Rules with a 'when' condition were skipped before (finding 2 of the xhigh decision); they are checked now."""
    cfg = deepcopy(config)
    cfg["families"]["H"]["ranges"] = {"ID": {"board_kind": {"choice": ["slotted"]}}}
    assert W.freeze_problems(cfg, envelope) == []
    tight = deepcopy(envelope)
    next(r for r in tight["rules"] if r["id"] == "sever_tilt_slotted_board")["max"] = 15.0
    problems = W.freeze_problems(config, tight)   # B tilt_slotted ID base reaches 20 deg on the slotted board
    assert any("family B: ID range of cut_tilt_deg" in p and "sever_tilt_slotted_board" in p for p in problems)
    assert not any("sever_tilt_solid_board" in p for p in problems)   # tilt_slotted never runs on the solid board
    loose = deepcopy(config)
    loose["range_sets"]["ID"]["cut_tilt_solid_deg"] = {"uniform": [-12.0, 12.0]}
    loose["range_sets"]["OOD"]["cut_tilt_solid_deg"] = {"uniform_outside": {"outer": [-20.0, 20.0], "inner": [-12.0, 12.0]}}
    problems = W.freeze_problems(loose, envelope)
    assert any("cut_tilt_solid_deg" in p and "sever_tilt_solid_board" in p for p in problems)


def test_freeze_problems_flag_pre_d5_draft(envelope):
    """The DRAFT before the D5 re-declaration: B 'tilt' ran on the solid board up to 20 deg and C had trim_layer. With
    'when' rules evaluated, the B range and exactly the B ID samples with |tilt| > 9.9 deg are flagged."""
    old = W.load_benchmark(PRE_D5_DRAFT)
    problems = W.freeze_problems(old, envelope)
    assert any("family B: ID range of cut_tilt_deg" in p and "sever_tilt_solid_board" in p for p in problems)
    assert any("family C: ID range of trim_depth_m" in p for p in problems)
    flagged_b = sorted(p.split()[2] for p in problems if p.startswith("ID sample P5_B_ID_benchmark"))
    expected = []
    for seed in W.seed_list(old, "B", "benchmark", "ID"):
        sample = W.build_sample(old, "B", "ID", seed)
        if sample["variant"] == "tilt" and abs(sample["derived"]["applied_tilt_deg"]) > 9.9:
            assert sample["envelope_parameters"]["board_kind"] == "solid"
            expected.append(sample["sample_id"])
    assert flagged_b == sorted(expected) and len(expected) == 2   # 2 of 10 B ID benchmark samples (decision finding 2)
    assert all("sever_tilt_solid_board" in p for p in problems if p.startswith("ID sample P5_B_"))
    assert sum(p.startswith("ID sample P5_C_ID_benchmark") for p in problems) == 3   # the trim_layer samples
    # without the sample check only the range problems remain
    assert not any(p.startswith("ID sample") for p in W.freeze_problems(old, envelope, check_samples=False))


def test_d5_redeclaration(config, envelope):
    """Decision D5: B has no ID stratum; tilt goes to the slotted variant; trim_layer is gone; speeds are fixed in ID;
    F and G (and the E2-dependent variants) are ID-unconfirmed and outside the 5.7 targets."""
    assert config["status"] == "DRAFT" and "redeclaration_d5" in config
    assert config["families"]["B"]["range_sets"] == ["OOD"]
    assert "B_ID" not in {s["id"] for s in W.strata(config)}
    assert all(f != "B" or rs == "OOD" for f, rs, _, _ in W.sample_ids(config, "benchmark"))
    assert "trim_layer" not in config["families"]["C"]["variants"]
    assert "trim_depth_m" not in config["families"]["C"]["randomize"]
    ident = config["range_sets"]["ID"]
    assert ident["cut_speed_m_s"]["uniform"] == [0.01, 0.01] and ident["skim_speed_m_s"]["uniform"] == [0.025, 0.025]
    for split in W.SPLITS:
        for family, rs, seed, _ in W.sample_ids(config, split):
            sample = W.build_sample(config, family, rs, seed, split=split)
            params = sample["envelope_parameters"]
            board = sample["scene"]["supports"]["workcell"]["overrides"]["cutting_board"]
            if sample["variant"] == "tilt_slotted":
                assert board["kind"] == "slotted" and params["board_kind"] == "slotted"
                x0 = W._plane_x_at(sample["geometry"]["planes"][0], 0.)
                pose = sample["scene"]["workpiece"]["world_pose"]
                expected = W._board_frame_x(config["common"], (*pose["translation_m"][:2], pose["yaw_deg"]), x0)
                assert abs(board["slots"][0]["x_board_m"]-expected) < 1e-6
            if sample["variant"] in ("tilt_solid", "plane_yaw", "advance_tilt"):
                assert board.get("kind", "solid") == "solid" and params["board_kind"] == "solid"
            if sample["variant"] == "tilt_solid":
                assert 0. < params["sever_tilt_abs_deg_max"] <= 9.9+1e-9
            if sample["variant"] == "tilt_slotted":
                assert 20. < params["sever_tilt_abs_deg_max"] <= 34.9+1e-9
            if rs == "ID":
                if "cut_speed_m_s" in params:
                    assert params["cut_speed_m_s"] == 0.01
                if "skim_speed_m_s" in params:
                    assert params["skim_speed_m_s"] == 0.025
                assert W.evaluate_envelope(envelope, params)["status"] == "SUPPORTED", sample["sample_id"]
    population = {(f, v): W.id_population(config, f, v) for f, fam in config["families"].items() for v in fam["variants"]}
    assert population[("F", "sever_then_trim")] == population[("F", "trim_then_sever")] == "ID_unconfirmed"
    assert all(population[("G", v)] == "ID_unconfirmed" for v in config["families"]["G"]["variants"])
    assert population[("C", "trim_plane_protrusion")] == population[("E", "trim_trim")] == "ID_unconfirmed"
    validated = {k for k, v in population.items() if v == "ID_validated"}
    assert validated == {("A", "vertical"), ("C", "skim_protrusion"), ("D", "multi_plane_sever"), ("D", "cut_cut"),
                         ("E", "remove_all"), ("E", "remove_selected"), ("E", "bumps_sequential"), ("H", "sever"),
                         ("H", "trim"), ("B", "tilt_slotted"), ("B", "tilt_solid"), ("B", "plane_yaw"),
                         ("B", "advance_tilt")}   # B has no ID samples at all
    for name in ("kinematic_ID", "native_ID"):
        assert config["targets"][name]["population"] == "ID_validated"
    bad = deepcopy(config)
    bad["target_scope"]["id_unconfirmed"]["G"] = ["no_such_variant"]
    with pytest.raises(W.BenchmarkConfigError, match="target_scope"):
        W.validate_benchmark(bad)


# ----------------------------------------------------------------------------- execution / summary

def _run(verdict="PASS", **extra):
    return {"status": "COMPLETE", "policy_done": True, "solver_invalid": None,
            "evaluation": {"verdict": verdict, "integrity": {"checks": {"a": verdict != "FAIL_INTEGRITY"}},
                           "outcome": {"checks": {"pieces": verdict == "PASS"}}}, **extra}


def test_execution_classes():
    assert W.classify_execution(_run())[0] == "task_success"
    assert W.classify_execution(_run("FAIL"))[:2] == ("execution_failure", "outcome")
    assert W.classify_execution(_run("FAIL_INTEGRITY"))[:2] == ("execution_failure", "integrity")
    assert W.classify_execution(_run(solver_invalid="slot overflow"))[:2] == ("execution_failure", "solver_invalid")
    assert W.classify_execution(_run(policy_done=False))[:2] == ("execution_failure", "plan_not_finished")
    assert W.classify_execution(_run(partial=True))[0] == "not_run"
    assert W.classify_execution(None)[0] == "not_run"
    assert W.classify_execution({"status": "ABORTED", "reason": "segfault"})[:2] == ("execution_failure", "crash")
    assert W.classify_execution({"status": "COMPLETE", "grasp_failure": "GRASP_SLIP"})[:2] == ("execution_failure", "grasp")


def test_wilson_interval():
    w = W.wilson(8, 10)
    assert w["p"] == .8 and math.isclose(w["lo"], 0.4901625, rel_tol=1e-5) and math.isclose(w["hi"], 0.9433178, rel_tol=1e-5)
    assert W.wilson(0, 0)["p"] is None
    z = W.wilson(0, 10)
    assert z["lo"] == 0. and math.isclose(z["hi"], 0.2775328, rel_tol=1e-5)
    assert W.wilson(10, 10)["hi"] == 1.


def _row(sid, family, rs, feas, cls, ood=None):
    return {"sample_id": sid, "family": family, "range_set": rs, "stratum": f"{family}_{rs}", "feasibility_class": feas,
            "class": cls, "failure_kind": "outcome" if cls == "execution_failure" else None, "ood_dimensions": ood or [],
            "envelope_pending": True}


def test_summary_denominators_and_no_sample_dropped():
    rows = [_row("s1", "A", "ID", "valid_feasible", "task_success"), _row("s2", "A", "ID", "valid_feasible", "execution_failure"),
            _row("s3", "A", "ID", "valid_feasible", "not_run"), _row("s4", "A", "ID", "ik_unreachable", "ik_unreachable"),
            _row("s5", "A", "OOD", "physically_unsupported", "physically_unsupported", ["rake_deg"]),
            _row("s6", "A", "OOD", "valid_feasible", "task_success", ["rake_deg"])]
    declared = [(r["sample_id"], {k: r[k] for k in ("family", "range_set", "stratum")}) for r in rows]
    declared.append(("s7", {"family": "A", "range_set": "OOD", "stratum": "A_OOD", "ood_dimensions": ["cut_fraction"]}))
    out = W.summarize(rows, declared, {"k": {"range_set": "ID", "metric": "success_among_feasible", "value": .9}})
    t = out["overall"]
    assert t["samples"] == 7 and sum(t["counts"].values()) == 7
    assert out["missing_records"] == ["s7"] and t["counts"]["not_run"] == 2
    ident = out["by_range_set"]["ID"]
    assert ident["feasibility_rate"]["k"] == 3 and ident["feasibility_rate"]["n"] == 4
    assert ident["success_among_feasible"]["k"] == 1 and ident["success_among_feasible"]["n"] == 3   # not_run stays in
    assert ident["success_among_executed"]["n"] == 2
    assert ident["overall_success"]["n"] == 4
    assert out["targets"]["k"]["met_point"] is False
    assert out["ood_boundary_by_dimension"]["rake_deg"]["samples"] == 2
    assert sum(g["samples"] for g in out["by_family_range_set"].values()) == 7
    with pytest.raises(AssertionError, match="Duplicate"):
        W.summarize(rows+[rows[0]])
    with pytest.raises(AssertionError, match="add up"):
        W.summarize([{**rows[0], "class": "mystery"}])
    with pytest.raises(AssertionError, match="undeclared"):
        W.summarize(rows, declared[:2])


def test_summary_target_population():
    rows = [_row("v1", "A", "ID", "valid_feasible", "task_success"), _row("v2", "A", "ID", "valid_feasible", "task_success"),
            _row("u1", "F", "ID", "valid_feasible", "execution_failure"), _row("o1", "A", "OOD", "valid_feasible", "task_success")]
    pop = (lambda r: "ID_unconfirmed" if r["family"] == "F" else "ID_validated")
    target = {"k": {"range_set": "ID", "population": "ID_validated", "metric": "success_among_feasible", "value": .9}}
    out = W.summarize(rows, targets=target, population=pop)
    assert out["by_id_population"]["ID_validated"]["samples"] == 2 and out["by_id_population"]["ID_unconfirmed"]["samples"] == 1
    assert out["targets"]["k"]["estimate"]["k"] == 2 and out["targets"]["k"]["estimate"]["n"] == 2
    assert out["targets"]["k"]["met_point"] is True
    assert out["by_range_set"]["ID"]["success_among_feasible"]["n"] == 3   # the unconfirmed row is still reported
    with pytest.raises(AssertionError, match="population"):
        W.summarize(rows, targets=target)


def test_final_row_keeps_infeasible_class():
    feas = {"sample_id": "x", "family": "B", "range_set": "OOD", "stratum": "B_OOD", "feasibility_class": "collision_constrained",
            "reason": "hand_table", "envelope_pending": True}
    assert W.final_row(feas, _run())["class"] == "collision_constrained"
    feas["feasibility_class"] = "valid_feasible"
    assert W.final_row(feas, _run())["class"] == "task_success"
    assert W.final_row(feas, None)["class"] == "not_run"


def test_native_selection_rule(config):
    rows = []
    for stratum in W.strata(config):
        for i, seed in enumerate(W.seed_list(config, stratum["family"], "benchmark", stratum["range_set"])):
            cls = "valid_feasible" if (i % 3 != 1 and not (stratum["id"] == "B_OOD" and i > 2)) else "ik_unreachable"
            rows.append({"sample_id": W.make_sample_id(config, stratum["family"], stratum["range_set"], "benchmark", seed),
                         "feasibility_class": cls, "outcome_never_read": "task_success"})
    sel = W.native_selection(config, rows)
    assert set(sel) == {s["id"] for s in W.strata(config)}
    a = sel["A_ID"]
    assert len(a["selected"]) == 3 and a["shortfall"] == 0
    assert [int(s.split("_s")[-1]) % 1000 for s in a["selected"]] == [0, 2, 3]
    assert sel["B_OOD"]["shortfall"] == 1 and sel["B_OOD"]["shortfall_reason"]
    assert all(s not in sel["A_ID"]["not_selected"] for s in sel["A_ID"]["selected"])
    assert len(sel["A_ID"]["selected"])+len(sel["A_ID"]["not_selected"]) == 10
    shuffled = deepcopy(config)
    shuffled["native"]["selection"]["rule"] = "seeded_shuffle_valid_feasible"
    s1, s2 = W.native_selection(shuffled, rows), W.native_selection(shuffled, rows)
    assert s1 == s2 and all(len(v["selected"]) == 3 for k, v in s1.items() if k != "B_OOD")


def test_preflight_mapping():
    assert W.classify_preflight({"status": "PASS"}) == ([], [])
    assert W.classify_preflight({"status": "ERROR", "reason": "x"})[0] is None
