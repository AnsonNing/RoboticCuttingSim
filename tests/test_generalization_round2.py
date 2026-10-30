"""WP-11 Phase E round 2: predeclared round file, builders, stage-2 subset rule, staged_v2 geometry (CPU, no physics).

Only non-declared development seeds (90001+) are built here; declared samples are built by the round's stages.
"""
import json

import numpy as np
import pytest

from robotic_cutting_sim import generalization as G
from robotic_cutting_sim.goals import step_digit_allocation, validate_goal
from robotic_cutting_sim.scene_config import resolve_scene

CONFIG = G.load_round(G.ROOT/"configs/wp11/generalization_round2.json")
ROUND1 = G.load_round()
FAMILIES = ("TEST_D2", "CUT_W", "OBLIQUE_W", "TRIM_W", "TEST_A_W")


def test_round2_declares_families_seeds_subset_and_classes():
    assert CONFIG["status"] == "PREDECLARED" and tuple(CONFIG["families"]) == FAMILIES
    assert CONFIG["common"]["sample_prefix"] == "E2" and CONFIG["outcome_classes"] == ROUND1["outcome_classes"]
    seeds = [s for f in CONFIG["families"].values() for s in f["seeds"]]
    assert len(seeds) == len(set(seeds)) == 46 and len(CONFIG["families"]["TEST_D2"]["seeds"]) == 6
    assert sum(len(CONFIG["families"][f]["seeds"]) for f in FAMILIES[1:]) == 40
    salts = [f["salt"] for f in CONFIG["families"].values()] + [f["salt"] for f in ROUND1["families"].values()]
    assert len(salts) == len(set(salts))
    for family in CONFIG["families"].values():
        assert family["purpose"] and all(len(row) == 3 and row[1] <= row[2] for row in family.get("draws", []))
    d2 = CONFIG["families"]["TEST_D2"]
    assert d2["solver"] == {"module": "tlmpm_v5", "slots_per_node": 16} and d2["evaluator_version"] == "goal_eval_v2"
    assert len(d2["step_patterns"]) == 6 and all(len(p) >= 2 for p in d2["step_patterns"])
    assert CONFIG["stages"]["stage1_geometry_version"] == "staged_v2"
    assert CONFIG["families"]["OBLIQUE_W"]["index_rules"]["cut_angle_deg"][:2] == [35.0, -35.0]
    assert G.make_sample_id(CONFIG, "CUT_W", 2101) == "E2_CUT_W_s2101"
    assert G.make_sample_id(ROUND1, "CUT", 1101) == "E1_CUT_s1101"


def test_stage2_subset_rule_is_declared_order_first_three_feasible_per_sweep_family():
    rows = [{"family": f, "seed": s, "feasible": (s % 3 != 0)} for f, s in G.sample_ids(CONFIG)]
    chosen = G.stage2_selection(CONFIG, list(reversed(rows)))
    by_family = {}
    for row in chosen:
        by_family.setdefault(row["family"], []).append(row["seed"])
    assert by_family["TEST_D2"] == [s for s in CONFIG["families"]["TEST_D2"]["seeds"] if s % 3]
    for family in FAMILIES[1:]:
        assert by_family[family] == [s for s in CONFIG["families"][family]["seeds"] if s % 3][:3]
    assert len(chosen) - len(by_family["TEST_D2"]) <= 12
    # Round 1 has no subset rule: every feasible sample, in stage-1 order.
    r1 = [{"family": f, "seed": s, "feasible": s % 2 == 0} for f, s in G.sample_ids(ROUND1)]
    assert G.stage2_selection(ROUND1, r1) == [row for row in r1 if row["feasible"]]


@pytest.mark.parametrize("index", range(6))
def test_d2_dev_samples_compile_on_tlmpm_v5_s16_with_goal_eval_v2(index):
    sample = G.build_sample(CONFIG, "TEST_D2", 90001+index, index=index)
    assert not sample["declared"] and sample["sample_id"] == f"E2_TEST_D2_s{90001+index}"
    goal = validate_goal(sample["goal"])
    assert goal["evaluator_version"] == "goal_eval_v2" and goal["type"] == "sequence"
    pattern = CONFIG["families"]["TEST_D2"]["step_patterns"][index]
    assert [s["type"] for s in goal["steps"]] == [{"sever": "sever", "remove": "remove_protrusions", "trim": "trim_plane"}[
        t.split(":")[0]] for t in pattern]
    task, info = G.compile_sample(CONFIG, sample)
    assert info["solver_module"] == "tlmpm_v5" and info["slots_per_node"] == 16 and task.max_cuts == info["digits_used"]
    assert task.scene["solver"]["slots_per_node"] == 16
    from robotic_cutting_sim.goals import _plan_digit_allocation
    assert _plan_digit_allocation(task.goal, task.scene, task) == step_digit_allocation(task.goal, task.scene)
    scene = resolve_scene(sample["scene"])
    bumps = sorted(scene["workpiece"]["protrusions"], key=lambda p: p["x_m"])
    assert bumps[1]["x_m"]-bumps[1]["radius_x_m"]-(bumps[0]["x_m"]+bumps[0]["radius_x_m"]) >= .014


@pytest.mark.parametrize("family,index", [("CUT_W", 0), ("OBLIQUE_W", 1), ("TEST_A_W", 2), ("TEST_A_W", 7)])
def test_sweep_dev_samples_build_by_the_round1_builders(family, index):
    sample = G.build_sample(CONFIG, family, 90001+index, index=index)
    task, info = G.compile_sample(CONFIG, sample)
    assert "slots_per_node" not in info and "evaluator_version" not in sample["goal"]
    assert info["solver_module"] == "tlmpm_v4"
    if family == "TEST_A_W":
        pose = sample["scene"]["workpiece"]["world_pose"]
        values = {d["name"]: d["value"] for d in sample["draws"]}
        assert pose["translation_m"][:2] == [values["x_m"], values["y_m"]] and pose["yaw_deg"] == values["yaw_deg"]
        assert sample["derived"]["variant"] == CONFIG["families"]["TEST_A_W"]["variants"][index]
    if family == "OBLIQUE_W":
        assert sample["derived"]["cut_angle_deg"] == -35.0


def test_round1_builds_are_unchanged_by_round2_options():
    sample = G.build_sample(ROUND1, "TEST_D", 1702)
    assert sample["sample_id"] == "E1_TEST_D_s1702" and "evaluator_version" not in sample["goal"]
    assert sample["scene"]["description"].startswith("Phase E round 1 sample TEST_D seed 1702")
    task, info = G.compile_sample(ROUND1, sample)
    assert "slots_per_node" not in info and info["solver_module"] == "tlmpm_v5"
    evidence = {row["sample_id"]: row["task_id"]
                for row in json.loads((G.ROOT/"docs/evidence/wp11/phaseE_round1_001.json").read_text())["samples"]}
    assert task.task_id == evidence["E1_TEST_D_s1702"]


def test_removed_sets_with_trim_aliases_and_staged_v1_default():
    labels = ["HOME", "PRE_CUT:trim0", "SKIM:trim0", "WITHDRAW:trim0", "CUT:cut1", "PRE_CUT:p1", "SKIM:p1", "RETURN_HOME"]
    assert G._removed_sets(labels, {"p0", "p1"}) == [(), (), (), (), (), (), (), ("p1",)]
    aliased = G._removed_sets(labels, {"p0", "p1"}, {"trim0": ("p0",)})
    assert aliased == [(), (), (), (), ("p0",), ("p0",), ("p0",), ("p0", "p1")]
    sample = G.build_sample(CONFIG, "TEST_D2", 90006, index=5)  # trim:p0, sever:0, remove:p1
    task, _ = G.compile_sample(CONFIG, sample)
    assert G.trim_aliases(task) == {"trim0": ("p0",)}
    with pytest.raises(ValueError, match="staged_v1 or staged_v2"):
        G.staged_preflight(task, checker=object(), geometry="staged_v3")


def test_episode_metrics_for_v2_sequences_with_trim_steps():
    goal = {"type": "sequence", "steps": [{"type": "sever", "expected_pieces": 2}, {"type": "trim_plane"}]}
    evaluation = {"outcome": {"steps": [
        {"index": 0, "goal_id": "sever", "outcome": {"checks": {"expected_substantial_pieces": True},
                                                     "metrics": {"substantial_pieces": 2, "minor_pieces_mass_fraction": 0.}}},
        {"index": 1, "goal_id": "trim_p1", "outcome": {
            "checks": {"material_above_plane_removed": True, "single_substrate_component_per_prior_piece": True},
            "metrics": {"removed_above_plane_mass_fraction": .97, "mechanical_components": 3,
                        "prior_pieces": [{"detached_below_plane_mass_fraction": 1e-4},
                                         {"detached_below_plane_mass_fraction": 0.}]}}}]},
        "quality": {"0:sever": {"plane_location": []}, "1:trim_p1": {"current_substrate_particle_center_max_residual_m": .001}}}
    metrics = G.episode_metrics(goal, evaluation)
    assert metrics["separation"] and metrics["cut_completion"] == pytest.approx(.97)
    assert metrics["steps"][1]["overcut_detached_below_plane_fraction"] == pytest.approx(1e-4)
