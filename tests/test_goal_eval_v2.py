"""goal_eval_v2 (ADR 0015 Amendment 1): step-scoped sequence evaluation on synthetic packed-code fixtures (CPU).

v1 (no ``evaluator_version``) stays byte-identical: its evaluations of the fixtures below are pinned by the
SHA-256 of their canonical JSON, computed with the goals.py of HEAD c29f02a before goal_eval_v2 existed.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from robotic_cutting_sim import goals
from robotic_cutting_sim.goals import evaluate_goal, mask_digits, step_digit_allocation, validate_goal
from robotic_cutting_sim.scene_config import SceneConfigError, compile_task, load_json

from test_scene_goal_config import goal, recording, scene, synthetic_dx, to_packed, with_solver

SEVER_X = -.024  # between p0 (-50 +- 12 mm) and p1 (0 +- 12 mm), on a stencil node plane (dx = spacing = 4 mm)


def sever_step(x=SEVER_X):
    return {"type": "sever", "goal_id": "sever", "planes": [{"origin_local_m": [x, 0., 0.], "normal_local": [1., 0., 0.]}],
            "expected_pieces": 2, "minimum_part_mass_fraction": .3}


def remove_step(select="all", goal_id="remove_all"):
    doc = {key: value for key, value in goal("remove_all_protrusions").items() if key not in ("schema", "episode")}
    doc.update(type="remove_protrusions", goal_id=goal_id, select=select)
    doc.pop("description")
    return doc


def trim_step(x, goal_id="trim"):
    return {"type": "trim_plane", "goal_id": goal_id, "plane": {"origin_local_m": [0., 0., .02], "normal_local": [0., 0., 1.]},
            "region_local_m": {"x": list(x)}}


def sequence(steps, goal_id="sever_then_remove", version=None, duration=60.):
    doc = {"schema": "rcs_goal_v1", "goal_id": goal_id, "type": "sequence", "episode": {"duration_s": duration},
           "steps": deepcopy(steps)}
    if version is not None:
        doc["evaluator_version"] = version
    return doc


def task_for(doc, cuts):
    return compile_task(with_solver(scene("bar_three_bumps"), "tlmpm_v5", cuts), doc, particle_spacing_m=.004)


@pytest.fixture(scope="module")
def sever_remove_task():
    return task_for(sequence([sever_step(), remove_step()]), 4)


def digit_codes(task, rows):
    """Ternary codes from [(digit, values)] rows (values 0/1/2 per particle)."""
    codes = np.zeros(len(task.sample_particles()[0]), np.int64)
    for digit, values in rows:
        codes += np.asarray(values, np.int64)*3**digit
    return codes


def sever_values(task, x=SEVER_X):
    local = task.sample_particles()[0]
    return np.where(local[:, 0] > x, 2, 1)


def chip_values(task, index, plane_z=.02):
    """Protrusion ``index``'s above-plane particles 2, a 2-cell band below the plane 1 (as test_scene_goal_config)."""
    local = task.sample_particles()[0]
    owner = goals.protrusion_ownership(task, local, plane_z)
    band = (local[:, 2] <= plane_z) & (local[:, 2] >= plane_z-2*synthetic_dx(task))
    return np.where(owner == index, 2, np.where(band, 1, 0))


def packed_recording(task, codes, cuts):
    return {**recording(task, to_packed(codes, cuts), dx=task.particle_spacing_m), "code_encoding": "packed2"}


def sever_then_remove_codes(task, sever_digit=0, first_chip=1):
    return digit_codes(task, [(sever_digit, sever_values(task))] + [(first_chip+i, chip_values(task, i)) for i in range(3)])


def digest(result):
    return hashlib.sha256(json.dumps(result, sort_keys=True, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                     else v.item() if isinstance(v, np.generic) else str(v)).encode()).hexdigest()


# ---------------------------------------------------------------- schema and ids

def test_evaluator_version_is_opt_in_and_keeps_v1_normal_form_and_ids(sever_remove_task):
    v1 = sequence([sever_step(), remove_step()])
    assert "evaluator_version" not in validate_goal(v1)
    assert validate_goal(sequence([sever_step(), remove_step()], version="goal_eval_v2"))["evaluator_version"] == "goal_eval_v2"
    with pytest.raises(SceneConfigError, match="evaluator_version"):
        validate_goal(sequence([sever_step(), remove_step()], version="goal_eval_v9"))
    with pytest.raises(SceneConfigError, match="unknown keys"):  # steps never carry their own version
        validate_goal(sequence([{**sever_step(), "evaluator_version": "goal_eval_v2"}, remove_step()]))
    v2_task = task_for(sequence([sever_step(), remove_step()], version="goal_eval_v2"), 4)
    assert v2_task.task_id != sever_remove_task.task_id  # a declared version is part of the goal
    assert task_for(v1, 4).task_id == sever_remove_task.task_id


def test_pinned_compiled_ids_do_not_change():
    from test_wp11_phase_c5b import SCENE
    assert compile_task(SCENE, "trim_bump_rho10").task_id == "CFG-trim01_wp11_workcell-trim_bump_rho10-a33f6613"
    # Round-1 Phase E sequence samples (stage 1 of 4b2fe08, docs/evidence/wp11/phaseE_round1_001.json).
    from robotic_cutting_sim import generalization as G
    config = G.load_round()
    evidence = {row["sample_id"]: row["task_id"]
                for row in json.loads(open("docs/evidence/wp11/phaseE_round1_001.json").read())["samples"]}
    for family, seed in (("TEST_D", 1702), ("TEST_D", 1704), ("SLICE_PUSH", 1301), ("TEST_C", 1601), ("CUT", 1102)):
        sample = G.build_sample(config, family, seed)
        task, _ = G.compile_sample(config, sample)
        assert task.task_id == evidence[sample["sample_id"]], sample["sample_id"]


def test_digit_allocation_follows_the_declared_convention_and_the_plan(sever_remove_task):
    doc = validate_goal(sequence([trim_step([-.065, -.035]), sever_step(), remove_step(["p2", "p1"], "remove_two")]))
    assert step_digit_allocation(doc, sever_remove_task.scene) == [[0], [1], [2, 3]]
    assert goals._plan_digit_allocation(doc, sever_remove_task.scene, task_for(doc, 4)) == [[0], [1], [2, 3]]
    assert goals._plan_digit_allocation(sever_remove_task.goal, sever_remove_task.scene, sever_remove_task) == [[0], [1, 2, 3]]


def test_mask_digits_keeps_only_the_named_digits():
    ternary = np.array([0, 1+2*3+1*9, 2*27])
    assert mask_digits(ternary, [1], "ternary").tolist() == [0, 6, 0]
    packed = to_packed(ternary, 4)
    assert mask_digits(packed, [0, 3], "packed2").tolist() == to_packed(np.array([0, 1, 54]), 4).tolist()


# ---------------------------------------------------------------- v1 unchanged

V1_PINS = {  # goals.py at c29f02a (before goal_eval_v2), these fixtures, sha256 of the canonical evaluation JSON
    "sequence_sever_then_remove": "c95956045e5371006f41dc82f82aa60505b6346312b4f6137cda8cb894dc5a06",
    "remove_all": "25cef1c94559a9a367c5f964b0aec1118f18355808fff4f5e9af23c056043a07",
    "sever_mid_on_bumps": "ee8b375ba2c9e8498eafee796ee12e3b041f50d4c63dc098f842b100b17a64f3",
}


def v1_fixture_results(task):
    data = packed_recording(task, sever_then_remove_codes(task), 4)
    bumps = compile_task(scene("bar_three_bumps"), goal("remove_all_protrusions"), particle_spacing_m=.004)
    chips = packed_recording(bumps, digit_codes(bumps, [(i, chip_values(bumps, i)) for i in range(3)]), 3)
    return {"sequence_sever_then_remove": evaluate_goal(sequence([sever_step(), remove_step()]), data, task=task),
            "remove_all": evaluate_goal(goal("remove_all_protrusions"), chips, task=bumps),
            "sever_mid_on_bumps": evaluate_goal(goal("sever_mid"), chips, task=bumps)}


def test_v1_evaluations_are_byte_identical_to_the_pre_v2_evaluator(sever_remove_task):
    results = v1_fixture_results(sever_remove_task)
    for name, result in results.items():
        assert "evaluator_version" not in result and "evaluator_version" not in result["outcome"]
        assert digest(result) == V1_PINS[name], name
    # The round-1 composition failure is reproduced by v1 on the synthetic sever-then-remove history.
    failed = goals_failed(results["sequence_sever_then_remove"])
    assert "step0:sever:cut_at_declared_planes" in failed
    assert {"step1:remove_all:single_substrate_component", "step1:remove_all:substrate_holds_below_plane_mass"} <= set(failed)


def goals_failed(result):
    from robotic_cutting_sim.generalization import failed_checks
    return failed_checks(result)


# ---------------------------------------------------------------- v2 scoped verdicts

def test_v2_scopes_sever_location_and_remove_integrity(sever_remove_task):
    doc = sequence([sever_step(), remove_step()], version="goal_eval_v2")
    task = task_for(doc, 4)
    result = evaluate_goal(doc, packed_recording(task, sever_then_remove_codes(task), 4), task=task)
    assert result["evaluator_version"] == "goal_eval_v2" and result["integrity"]["status"] == "PASS"
    assert result["verdict"] == "PASS", goals_failed(result)
    sever, remove = (row["outcome"] for row in result["outcome"]["steps"])
    assert sever["digit_scope"] == {"own_digits": [0], "prior_sever_digits": [], "components_from": "own digits"}
    location = result["quality"]["0:sever"]["plane_location"][0]
    assert location["frozen_2mm_statistic_m"] <= .002  # only the sever digit's own interface
    assert sever["metrics"]["mechanical_components"] == 2
    assert remove["digit_scope"]["own_digits"] == [1, 2, 3] and remove["digit_scope"]["prior_sever_digits"] == [0]
    pieces = remove["metrics"]["prior_pieces"]
    assert len(pieces) == 2 and all(row["pass"] and row["substrate_components"] == 1 for row in pieces)
    assert [row["removed_mass_fraction"] for row in remove["metrics"]["items"]] == [1., 1., 1.]
    assert result["outcome"]["digit_allocation"]["plan_cross_check"] == "MATCH"
    # Ternary codes give the same verdicts and items.
    ternary = evaluate_goal(doc, recording(task, sever_then_remove_codes(task), dx=task.particle_spacing_m), task=task)
    assert ternary["verdict"] == "PASS"
    assert ternary["outcome"]["steps"][1]["outcome"]["metrics"]["items"] == remove["metrics"]["items"]


def test_v2_negative_remove_step_that_splits_the_substrate_fails():
    doc = sequence([sever_step(), remove_step()], version="goal_eval_v2")
    task = task_for(doc, 4)
    local = task.sample_particles()[0]
    codes = sever_then_remove_codes(task)
    # The p1 skim digit (2) also cuts straight down through the +x piece at x = +24 mm (a real split below the plane).
    split = np.where(local[:, 0] > .024, 2, 1)
    chip = chip_values(task, 1)
    codes += (np.where(chip == 2, 2, split)-chip)*3**2
    result = evaluate_goal(doc, packed_recording(task, codes, 4), task=task)
    assert result["integrity"]["status"] == "PASS" and result["verdict"] == "FAIL"
    remove = result["outcome"]["steps"][1]["outcome"]
    assert not remove["checks"]["single_substrate_component_per_prior_piece"]
    assert not remove["checks"]["substrate_holds_below_plane_mass_per_prior_piece"]
    assert result["outcome"]["steps"][0]["outcome"]["status"] == "PASS"  # the sever itself is still right
    split_piece = [row for row in remove["metrics"]["prior_pieces"] if row["substrate_components"] > 1]
    assert len(split_piece) == 1 and split_piece[0]["largest_component_below_plane_mass_share"] < .99


def test_v2_negative_misplaced_sever_still_fails_location():
    doc = sequence([sever_step(), remove_step()], version="goal_eval_v2")
    task = task_for(doc, 4)
    local = task.sample_particles()[0]
    codes = sever_then_remove_codes(task)
    codes += (np.where(local[:, 0] > SEVER_X+.008, 2, 1)-sever_values(task))  # cut 8 mm off
    result = evaluate_goal(doc, packed_recording(task, codes, 4), task=task)
    sever = result["outcome"]["steps"][0]["outcome"]
    assert result["verdict"] == "FAIL" and not sever["checks"]["cut_at_declared_planes"]
    assert sever["checks"]["expected_substantial_pieces"]


def test_v2_remove_first_then_sever_and_scoped_retention():
    # remove p1 (keep p0, p2), then sever, then trim p0's region: v1 fails retention (p0 trimmed later) and the
    # single-substrate rule (the later sever); v2 scores each step on what it did.
    steps = [remove_step(["p1"], "remove_p1"), sever_step(), trim_step([-.065, -.035], "trim_p0")]
    doc = sequence(steps, goal_id="remove_sever_trim", version="goal_eval_v2")
    task = task_for(doc, 3)
    codes = digit_codes(task, [(0, chip_values(task, 1)), (1, sever_values(task)), (2, chip_values(task, 0))])
    data = packed_recording(task, codes, 3)
    v2 = evaluate_goal(doc, data, task=task)
    assert v2["verdict"] == "PASS", goals_failed(v2)
    assert [row["outcome"]["digit_scope"]["own_digits"] for row in v2["outcome"]["steps"]] == [[0], [1], [2]]
    assert v2["outcome"]["steps"][2]["outcome"]["digit_scope"]["prior_sever_digits"] == [1]
    v1 = evaluate_goal({key: value for key, value in doc.items() if key != "evaluator_version"}, data, task=task)
    failed = goals_failed(v1)
    assert v1["verdict"] == "FAIL" and "step0:remove_p1:unselected_protrusions_retained" in failed
    assert "step0:remove_p1:single_substrate_component" in failed and "step2:trim_p0:single_substrate_component" in failed


def test_v2_on_a_single_goal_equals_v1(sever_remove_task):
    bumps = compile_task(scene("bar_three_bumps"), goal("remove_all_protrusions"), particle_spacing_m=.004)
    chips = packed_recording(bumps, digit_codes(bumps, [(i, chip_values(bumps, i)) for i in range(3)]), 3)
    v1 = evaluate_goal(goal("remove_all_protrusions"), chips, task=bumps)
    v2 = evaluate_goal({**goal("remove_all_protrusions"), "evaluator_version": "goal_eval_v2"}, chips, task=bumps)
    assert v2.pop("evaluator_version") == "goal_eval_v2" and digest(v2) == digest(v1)
