"""WP-12 Phase 5 generalization infrastructure (generalization_wp12): generators, feasibility, summaries. CPU only."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from robotic_cutting_sim import generalization_wp12 as W
from robotic_cutting_sim.goals import GOAL_TYPES, _TYPE_FIELDS, validate_goal
from robotic_cutting_sim.scene_config import SceneInfeasible, resolve_scene


@pytest.fixture(scope="module")
def config():
    return W.load_benchmark()


@pytest.fixture(scope="module")
def template():
    return W.load_envelope()


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


def test_envelope_template_has_fields_only(template):
    assert template["status"] == "TEMPLATE"
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
            assert "board_kind" in config["families"][family]["randomize"], family
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
            with pytest.raises(W.NotExpressible, match="reposition"):
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


def test_classifier_valid_feasible_and_pending_envelope(config, template):
    row = W.classify_feasibility(_sample(config), template, compile_fn=lambda s: (_Task(), {"digits_used": 1}),
                                 preflight_fn=lambda t: PASS, common=config["common"])
    assert row["feasibility_class"] == "valid_feasible"
    assert row["envelope_pending"] is True
    assert all(r["status"] in ("NOT_EVALUATED", "NOT_APPLICABLE") for r in row["checks"]["envelope"]["rules"])


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


def test_freeze_problems(config, template):
    assert any("TEMPLATE" in p for p in W.freeze_problems(config, template))
    env = deepcopy(template)
    env["status"] = "DECLARED"
    for rule in env["rules"]:
        for key in ("min", "max"):
            if key in rule:
                rule[key] = {"max": 1e9, "min": -1e9}[key]
        if "values" in rule:
            rule["values"] = ["cutting_tasks_v1", "solid", "slotted", "dome", "gaussian", "ellipsoid_cap", "ridge"]
    env["rules"] = [r for r in env["rules"] if r["parameter"] != "operation_pattern"]
    assert W.freeze_problems(config, env) == []
    tight = deepcopy(env)
    next(r for r in tight["rules"] if r["id"] == "trim_rake_deg")["max"] = 30.0
    problems = W.freeze_problems(config, tight)
    assert any("rake_deg" in p and "trim_rake_deg" in p for p in problems)


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
