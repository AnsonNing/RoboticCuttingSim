"""WP-11 Phase E round 1: predeclared round file, seeded samples, outcome classes (CPU, no physics)."""
from copy import deepcopy

import numpy as np
import pytest

from robotic_cutting_sim import generalization as G
from robotic_cutting_sim import goal_planner
from robotic_cutting_sim.goals import validate_goal
from robotic_cutting_sim.scene_config import SceneConfigError, resolve_scene
from robotic_cutting_sim.tool_profile import PROFILES

CONFIG = G.load_round()
FAMILIES = ("CUT", "TRIM", "SLICE_PUSH", "TEST_A", "TEST_B", "TEST_C", "TEST_D")


def test_round_file_declares_every_family_seed_and_class():
    assert CONFIG["status"] == "PREDECLARED" and tuple(CONFIG["families"]) == FAMILIES
    for name, family in CONFIG["families"].items():
        assert len(family["seeds"]) >= 5 and len(set(family["seeds"])) == len(family["seeds"]), name
        for row in family.get("draws", []):
            assert len(row) == 3 and row[1] <= row[2], (name, row)
    assert set(CONFIG["outcome_classes"]) == set(G.CLASSES)
    assert len(G.sample_ids(CONFIG)) == sum(len(f["seeds"]) for f in CONFIG["families"].values())
    assert {f["salt"] for f in CONFIG["families"].values()}.__len__() == len(FAMILIES)


@pytest.mark.parametrize("family", FAMILIES)
def test_declared_samples_are_deterministic_valid_and_record_their_draws(family):
    for seed in CONFIG["families"][family]["seeds"]:
        a, b = G.build_sample(CONFIG, family, seed), G.build_sample(CONFIG, family, seed)
        assert a == b and a["declared"]
        for row in a["draws"]:
            assert row["low"] <= row["value"] <= row["high"]
        scene = resolve_scene(a["scene"])
        goal = validate_goal(a["goal"])
        assert scene["supports"]["workcell"] == {"from_scenario": "wp11_workcell_v1"}
        assert scene["tool"]["profile_id"] == "ruler_knife_v2_y"
        assert scene["workpiece"]["world_pose"]["translation_m"][2] == CONFIG["common"]["world_z_m"]
        assert goal["planner"]["controller_profile"] == "path_v2" and goal["planner"]["cut_extent"] == "plane_section"
        if family != "TEST_A":
            sampled = {row["path"]: row["value"] for row in scene["randomization"]["sampled"]}
            assert sampled["workpiece.base_box_m.length_m"] == scene["workpiece"]["base_box_m"]["length_m"]
            assert sampled["workpiece.world_pose.yaw_deg"] == scene["workpiece"]["world_pose"]["yaw_deg"]


def test_index_rules_cover_every_declared_discrete_value():
    angles = [G.build_sample(CONFIG, "CUT", s)["derived"]["cut_angle_deg"] for s in CONFIG["families"]["CUT"]["seeds"]]
    assert angles == [0., 20.]*3
    trims = {(G.build_sample(CONFIG, "TRIM", s)["derived"]["protrusion_family"], G.build_sample(CONFIG, "TRIM", s)["derived"]["rake_deg"])
             for s in CONFIG["families"]["TRIM"]["seeds"]}
    assert trims == {(f, r) for f in ("dome", "gaussian", "ellipsoid_cap") for r in (10., 25.)}
    slices = [G.build_sample(CONFIG, "TEST_B", s)["derived"]["slices"] for s in CONFIG["families"]["TEST_B"]["seeds"]]
    assert slices == [3, 4, 3, 4, 3]
    variants = [G.build_sample(CONFIG, "TEST_A", s)["derived"]["variant"]["op"] for s in CONFIG["families"]["TEST_A"]["seeds"]]
    assert variants.count("sever") == 4 and variants.count("trim") == 4


def test_declared_seed_keeps_its_index_and_dev_seeds_need_one():
    seed = CONFIG["families"]["CUT"]["seeds"][0]
    with pytest.raises(ValueError):
        G.build_sample(CONFIG, "CUT", seed, index=1)
    with pytest.raises(ValueError):
        G.build_sample(CONFIG, "CUT", 90001)
    assert not G.build_sample(CONFIG, "CUT", 90001, index=1)["declared"]


@pytest.mark.parametrize("family,index", [("CUT", 0), ("CUT", 1), ("TRIM", 0), ("SLICE_PUSH", 0), ("TEST_B", 0),
                                          ("TEST_C", 0)])
def test_dev_sample_compiles_with_the_declared_solver_and_duration_rules(family, index):
    sample = G.build_sample(CONFIG, family, 90001+index, index=index)
    task, info = G.compile_sample(CONFIG, sample)
    assert info["solver_module"] == ("tlmpm_v4" if info["digits_used"] <= 2 else "tlmpm_v5")
    assert task.max_cuts == max(1, info["digits_used"]) and task.cutting_law == 8
    assert task.duration_s % .5 == 0 and info["plan_end_s"]+1. <= task.duration_s < info["plan_end_s"]+1.5+1e-9
    assert task.particle_spacing_m == .001 and task.grid_dx_m == .004 and task.home is not None
    labels = task.reference_phase_labels
    assert "RETURN_HOME" in labels and labels[-1] == "DONE"  # knot labels name the waypoint pursued next


def _oblique(cut_extent):
    sample = G.build_sample(CONFIG, "CUT", 90002, index=1)
    scene, goal = resolve_scene(sample["scene"]), deepcopy(sample["goal"])
    if cut_extent is None:
        goal["planner"].pop("cut_extent")
    else:
        goal["planner"]["cut_extent"] = cut_extent
    return goal_planner.plan_with_report(scene, goal, PROFILES["ruler_knife_v2_y"])[1]["operations"][0], scene


def test_cut_extent_is_opt_in_and_plane_section_ends_an_oblique_cut_just_below_the_bar():
    bbox, scene = _oblique(None)
    section, _ = _oblique("plane_section")
    assert "cut_extent" not in bbox and section["cut_extent"] == "plane_section"
    assert bbox["end_local_m"][2] < -.012                       # beyond the wp11 action workspace
    overtravel = goal_planner.DEFAULT_OPTIONS["overtravel_m"]
    a = np.asarray(section["advance_local"])
    assert section["end_local_m"][2] == pytest.approx(overtravel*a[2], abs=1e-9)
    assert bbox["start_local_m"] != section["start_local_m"]
    with pytest.raises(SceneConfigError):
        _oblique("other")


def test_cut_extent_does_not_change_a_vertical_cut():
    sample = G.build_sample(CONFIG, "CUT", 90001, index=0)
    scene, goal = resolve_scene(sample["scene"]), deepcopy(sample["goal"])
    with_option = goal_planner.plan_with_report(scene, goal, PROFILES["ruler_knife_v2_y"])[1]["operations"][0]
    goal["planner"].pop("cut_extent")
    without = goal_planner.plan_with_report(scene, goal, PROFILES["ruler_knife_v2_y"])[1]["operations"][0]
    np.testing.assert_allclose(with_option["start_local_m"], without["start_local_m"], atol=1e-12)
    np.testing.assert_allclose(with_option["end_local_m"], without["end_local_m"], atol=1e-12)


def test_removed_sets_follow_the_plan_order():
    labels = ["HOME", "REORIENT:p0", "SKIM:p0", "WITHDRAW:p0", "REORIENT:p1", "SKIM:p1", "APPROACH:cut2", "CUT:cut2",
              "RETURN_HOME", "DONE"]
    removed = G._removed_sets(labels, {"p0", "p1", "p2"})
    assert removed[:4] == [()]*4 and removed[4:6] == [("p0",)]*2
    assert removed[6:8] == [("p0", "p1")]*2 and removed[8:] == [("p0", "p1")]*2


def test_preflight_classes():
    assert G.classify_preflight({"status": "PASS"}) == (None, None)
    assert G.classify_preflight({"status": "FAIL", "reason": "IK_FAILURE", "ik_failure": {"sample": 3, "label": "CUT:cut0"}})[0] == "IK_FAILURE"
    joint = {"status": "FAIL", "reason": "SCENE_INFEASIBLE", "path": {"failing": ["joint_limit_margin"], "worst": {}}}
    assert G.classify_preflight(joint)[0] == "IK_FAILURE"
    both = {"status": "FAIL", "reason": "SCENE_INFEASIBLE",
            "path": {"failing": ["blade_collision", "joint_continuity"],
                     "worst": {"blade_collision": {"slack": -.001, "label": "PRE_CUT:p0", "pair": ["a", "b"]}}}}
    assert G.classify_preflight(both)[0] == "SCENE_INFEASIBLE"


def test_material_classes_and_summary_never_exclude():
    ok = {"status": "COMPLETE", "policy_done": True, "evaluation": {"verdict": "PASS"}}
    assert G.classify_material(ok)[0] == "TASK_SUCCESS"
    assert G.classify_material({**ok, "evaluation": {"verdict": "FAIL", "outcome": {"checks": {"a": False}}}})[0] == "CUTTING_FAILURE"
    assert G.classify_material({**ok, "solver_invalid": "slot overflow"})[0] == "CUTTING_FAILURE"
    assert G.classify_material({**ok, "policy_done": False})[0] == "CUTTING_FAILURE"
    assert G.classify_material(None)[0] == "NOT_RUN"
    rows = [{"family": "CUT", "class": c} for c in ("TASK_SUCCESS", "TASK_SUCCESS", "CUTTING_FAILURE", "NOT_RUN",
                                                     "IK_FAILURE", "SCENE_INFEASIBLE")]
    table = G.summarize(rows, ["CUT"])["CUT"]
    assert table["samples"] == 6 and table["preflight_feasible"] == 4
    assert table["kinematic_task_success_among_feasible"] == .5 and table["overall_kinematic_success"] == 2/6
    assert table["counts"]["GRASP_FAILURE"] == 0 and table["not_run_among_feasible"] == 1


def test_history_stride_rule():
    assert G.history_stride(CONFIG, 100) == 5 and G.history_stride(CONFIG, 1200) == 30


def test_episode_metrics_on_goal_evaluations_of_synthetic_recordings():
    from robotic_cutting_sim.goals import evaluate_goal
    from robotic_cutting_sim.scene_config import compile_task
    from test_scene_goal_config import plane_codes, protrusion_codes, recording
    cut = G.build_sample(CONFIG, "CUT", 90001, index=0)
    _, info = G.compile_sample(CONFIG, cut)
    task = compile_task(info["scene"], info["goal"])  # formal 1 mm, solver-phase 4 mm stencil below
    local = task.sample_particles()[0]
    plane = info["goal"]["planes"][0]
    codes = plane_codes(local, [(plane["origin_local_m"], plane["normal_local"])])
    evaluation = evaluate_goal(task.goal, recording(task, codes, dx=task.grid_dx_m), task=task)
    metrics = G.episode_metrics(task.goal, evaluation)
    assert evaluation["verdict"] == "PASS" and metrics["cut_completion"] == 1. and metrics["separation"]
    assert metrics["cut_position_error_m"] == pytest.approx(.0015) and metrics["overcut_debris_mass_fraction"] < .01
    uncut = G.episode_metrics(task.goal, evaluate_goal(task.goal, recording(task, np.zeros(len(local)), dx=task.grid_dx_m),
                                                       task=task))
    assert uncut["cut_completion"] == .5 and not uncut["separation"]
    trim = G.build_sample(CONFIG, "TRIM", 90001, index=0)
    _, info = G.compile_sample(CONFIG, trim)
    task = compile_task(info["scene"], info["goal"], particle_spacing_m=.004)
    height = info["scene"]["workpiece"]["base_box_m"]["height_m"]
    evaluation = evaluate_goal(task.goal, recording(task, protrusion_codes(task, [0], plane_z=height)), task=task)
    metrics = G.episode_metrics(task.goal, evaluation)
    assert metrics["cut_completion"] > .9 and metrics["per_protrusion_removed"]["p0"] == metrics["cut_completion"]


def test_dimension_and_cut_plane_rules_make_the_frozen_location_statistic_meaningful():
    """Ideal plane labels on node-aligned planes of even-mm bars score 1.5 mm (frozen bound 2 mm) at 1 mm / 4 mm grid."""
    from robotic_cutting_sim.goals import interface_by_plane
    from robotic_cutting_sim.scene_config import compile_task
    from test_scene_goal_config import stencils
    for seed in (90001, 90003):
        sample = G.build_sample(CONFIG, "CUT", seed, index=0)
        L, W, H = sample["derived"]["applied_dims_m"]
        assert round(L*1e3) % 2 == 0 and round(W*1e3) % 2 == 0 and abs(H*1e3-round(H*1e3)) < 1e-9
        x = sample["goal"]["planes"][0]["origin_local_m"][0]
        assert abs(x/.004-round(x/.004)) < 1e-9
        _, info = G.compile_sample(CONFIG, sample)
        task = compile_task(info["scene"], info["goal"])
        local = task.sample_particles()[0]
        nodes, weights, gradients = stencils(task, task.grid_dx_m)  # nodes at multiples of dx, as tlmpm_v4/v5
        labels = (local[:, 0] > x).astype(np.int64)
        row = interface_by_plane(nodes, weights, gradients, labels, local, [(np.array([x, 0., 0.]), np.array([1., 0., 0.]))])[0]
        assert row["maximum_reference_midpoint_plane_distance_m"] == pytest.approx(.0015, abs=1e-6)
    slice_push = G.build_sample(CONFIG, "SLICE_PUSH", 90001, index=0)
    for plane in slice_push["goal"]["steps"][0]["planes"]:
        assert abs(plane["origin_local_m"][0]/.004-round(plane["origin_local_m"][0]/.004)) < 1e-9
