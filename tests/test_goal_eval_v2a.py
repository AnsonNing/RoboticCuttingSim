"""goal_eval_v2a (ADR 0015 Amendment 2) on synthetic packed-code fixtures (CPU).

v1 and v2 are unchanged: their evaluations of these fixtures are pinned by the SHA-256 of the canonical JSON,
computed with the goals.py of b624df6 (before goal_eval_v2a existed).
"""
import numpy as np
import pytest

from robotic_cutting_sim import goals
from robotic_cutting_sim.goals import evaluate_goal, validate_goal

from test_goal_eval_v2 import (V1_PINS, chip_values, digest, digit_codes, packed_recording, remove_step,
                               sever_step, sever_then_remove_codes, sequence, sever_values, synthetic_dx,
                               task_for, trim_step, v1_fixture_results)


def without_version(doc):
    return {key: value for key, value in doc.items() if key != "evaluator_version"}


def versions(doc, version):
    return {**without_version(doc), "evaluator_version": version}


# ---------------------------------------------------------------- fixtures

def sever_remove(version):
    doc = sequence([sever_step(), remove_step()], version=version)
    return doc, task_for(doc, 4), 4


def unowned_extra_cut(version):
    """sever + remove all (digits 0-3) and an extra through-cut at x = +40 mm on digit 4, which no step owns."""
    doc = sequence([sever_step(), remove_step()], version=version)
    task = task_for(doc, 5)
    local = task.sample_particles()[0]
    return doc, task, sever_then_remove_codes(task)+np.where(local[:, 0] > .040, 2, 1)*3**4


def remove_then_sever(version, chip_labelled):
    """remove p1 (digit 0), then sever at x = 0 through where p1 was (digit 1). Physically the later sever never meets
    the removed chip (chip_labelled False); a chip left in place may also be labelled (True)."""
    doc = sequence([remove_step(["p1"], "remove_p1"), sever_step(0.)], goal_id="remove_then_sever", version=version)
    task = task_for(doc, 2)
    local = task.sample_particles()[0]
    chip = chip_values(task, 1)
    sever = np.where(local[:, 0] > 0., 2, 1)
    return doc, task, digit_codes(task, [(0, chip), (1, sever if chip_labelled else np.where(chip == 2, 0, sever))])


def local_chip(task, index, plane_z=.02):
    """Protrusion ``index`` above the plane 2; a two-cell band below the plane only under its footprint 1; else 0."""
    local = task.sample_particles()[0]
    owner = goals.protrusion_ownership(task, local, plane_z)
    bump = task.geometry.bumps[index]
    dx = synthetic_dx(task)
    band = ((local[:, 2] <= plane_z) & (local[:, 2] >= plane_z-2*dx)
            & (np.abs(local[:, 0]-bump.x_m) <= bump.radius_x_m+2*dx))
    return np.where(owner == index, 2, np.where(band, 1, 0))


def joint_detachment(version):
    """remove p0 (digit 0) then remove p1 (digit 1). Each digit also cuts one side of a below-plane block at the +x bar
    end (Q: x > 67 mm, z > 8 mm): digit 0 its -x side, digit 1 its bottom. Each digit alone leaves Q attached (through
    the other side); together they detach Q (substrate below the plane)."""
    doc = sequence([remove_step(["p0"], "remove_p0"), remove_step(["p1"], "remove_p1")], goal_id="remove_remove",
                   version=version)
    task = task_for(doc, 2)
    local = task.sample_particles()[0]
    x, z = local[:, 0], local[:, 2]
    block = (x > .067) & (z > .008)
    a = local_chip(task, 0)
    a = np.where(block, 2, np.where((x > .055) & (x <= .067), 1, np.where(x > .067, 0, a)))
    b = local_chip(task, 1)
    b = np.where(block, 2, np.where((x > .067) & (z <= .008), 1, b))
    return doc, task, digit_codes(task, [(0, a), (1, b)]), block


V2_PINS = {  # goals.py at b624df6 (before goal_eval_v2a), the fixtures above under goal_eval_v2
    "sever_remove": "87134ef741e87009fdfde7195e9dfc7000c77170fc723315e1e0a7195bc21c61",
    "unowned_extra_cut": "f4749313b46014c25ab9317642a9d8f05246325159110c8a30217e5cf8db3521",
    "remove_then_sever_skipping_chip": "f96833b3f56122eb04f4dafd59d83f9e61abec8fca8c8897a450b42aad0ed1b5",
}


def run(doc, task, codes, cuts):
    return evaluate_goal(doc, packed_recording(task, codes, cuts), task=task)


# ---------------------------------------------------------------- schema; v1 and v2 unchanged

def test_v2a_is_opt_in_and_versioned():
    doc, task, _ = sever_remove("goal_eval_v2a")
    assert validate_goal(doc)["evaluator_version"] == "goal_eval_v2a"
    assert "evaluator_version" not in validate_goal(without_version(doc))
    assert task.task_id not in (task_for(without_version(doc), 4).task_id, task_for(versions(doc, "goal_eval_v2"), 4).task_id)


def test_v1_and_v2_evaluations_are_unchanged():
    results = v1_fixture_results(sever_remove(None)[1])
    assert {name: digest(result) for name, result in results.items()} == V1_PINS
    doc, task, cuts = sever_remove("goal_eval_v2")
    assert digest(run(doc, task, sever_then_remove_codes(task), cuts)) == V2_PINS["sever_remove"]
    doc, task, codes = unowned_extra_cut("goal_eval_v2")
    assert digest(run(doc, task, codes, 5)) == V2_PINS["unowned_extra_cut"]
    doc, task, codes = remove_then_sever("goal_eval_v2", False)
    assert digest(run(doc, task, codes, 2)) == V2_PINS["remove_then_sever_skipping_chip"]


def test_v2a_equals_v2_step_results_when_nothing_is_detached_earlier():
    doc, task, cuts = sever_remove("goal_eval_v2a")
    codes = sever_then_remove_codes(task)
    v2a = run(doc, task, codes, cuts)
    v2 = run(versions(doc, "goal_eval_v2"), task_for(versions(doc, "goal_eval_v2"), 4), codes, cuts)
    assert v2a["verdict"] == v2["verdict"] == "PASS"
    assert v2a["outcome"]["final_all_digit_substrate"]["status"] == "PASS"
    for a, b in zip(v2a["outcome"]["steps"], v2["outcome"]["steps"]):
        assert a["outcome"]["checks"] == b["outcome"]["checks"]
        assert a["outcome"]["digit_scope"]["detached_by_earlier_steps"]["particles"] == 0


# ---------------------------------------------------------------- the three v2a rules

def test_unowned_digit_fails_with_an_explicit_reason():
    doc, task, codes = unowned_extra_cut("goal_eval_v2a")
    v2a = run(doc, task, codes, 5)
    assert v2a["integrity"]["status"] == "PASS" and v2a["verdict"] == "FAIL"
    assert not v2a["outcome"]["checks"]["no_unowned_digit_labels"]
    assert all(row["outcome"]["status"] == "PASS" for row in v2a["outcome"]["steps"])   # steps alone pass
    assert "Digits [4] carry labels but no step owns them" in v2a["outcome"]["fail_reasons"][0]
    assert v2a["outcome"]["digit_allocation"]["unowned_digits_labelled_particles"] == {"4": int(np.count_nonzero(codes >= 81))}
    assert run(versions(doc, "goal_eval_v2"), task_for(versions(doc, "goal_eval_v2"), 5), codes, 5)["verdict"] == "PASS"
    assert run(without_version(doc), task_for(without_version(doc), 5), codes, 5)["verdict"] == "FAIL"


def test_allocation_that_cannot_be_determined_is_not_run(monkeypatch):
    doc, task, cuts = sever_remove("goal_eval_v2a")
    # A recording that keeps one digit cannot hold the four the sequence needs.
    short = {**packed_recording(task, sever_values(task), 1), "max_cuts": 1}
    result = evaluate_goal(doc, short, task=task, cuts=1)
    assert result["outcome"]["status"] == "NOT_RUN" and "no_unowned_digit_labels" not in result["outcome"]["checks"]
    assert "needs 4 cut digits" in result["outcome"]["steps"][0]["outcome"]["missing_evidence"][0]
    # A planner report that numbers the digits differently.
    monkeypatch.setattr(goals, "_plan_digit_allocation", lambda goal, scene, task: [[1], [0, 2, 3]])
    mismatch = run(doc, task, sever_then_remove_codes(task), cuts)
    assert mismatch["outcome"]["status"] == "NOT_RUN"
    assert mismatch["outcome"]["digit_allocation"]["plan_cross_check"] == "MISMATCH"
    assert "no_unowned_digit_labels" not in mismatch["outcome"]["checks"]


def test_joint_detachment_by_two_remove_steps_fails_the_final_check():
    doc, task, codes, block = joint_detachment("goal_eval_v2a")
    local = task.sample_particles()[0]
    below = local[:, 2] <= .02
    masses = task.sample_particles()[1]
    assert masses[block & below].sum()/masses[below].sum() > .01         # more than the 1% detachment allowance
    v2a = run(doc, task, codes, 2)
    assert all(row["outcome"]["status"] == "PASS" for row in v2a["outcome"]["steps"]), \
        [row["outcome"]["checks"] for row in v2a["outcome"]["steps"]]
    final = v2a["outcome"]["final_all_digit_substrate"]
    assert v2a["verdict"] == "FAIL" and final["status"] == "FAIL"
    assert not final["checks"]["final_detached_below_plane_mass_bounded_per_piece"]
    assert final["detached_below_plane_mass_fraction_total"] == pytest.approx(
        masses[block & below].sum()/masses[below].sum())
    assert run(versions(doc, "goal_eval_v2"), task_for(versions(doc, "goal_eval_v2"), 2), codes, 2)["verdict"] == "PASS"
    assert run(without_version(doc), task_for(without_version(doc), 2), codes, 2)["verdict"] == "FAIL"


@pytest.mark.parametrize("chip_labelled", [False, True])
def test_remove_before_sever_passes(chip_labelled):
    doc, task, codes = remove_then_sever("goal_eval_v2a", chip_labelled)
    v2a = run(doc, task, codes, 2)
    assert v2a["verdict"] == "PASS", [(r["goal_id"], r["outcome"]["checks"]) for r in v2a["outcome"]["steps"]]
    sever = v2a["outcome"]["steps"][1]["outcome"]
    assert sever["metrics"]["mechanical_components"] == 2
    assert sever["digit_scope"]["detached_by_earlier_steps"]["particles"] == int(np.count_nonzero(chip_values(task, 1) == 2))
    assert v2a["outcome"]["final_all_digit_substrate"]["status"] == "PASS"
    v2 = run(versions(doc, "goal_eval_v2"), task_for(versions(doc, "goal_eval_v2"), 2), codes, 2)
    assert v2["verdict"] == ("PASS" if chip_labelled else "FAIL")       # the v2 false FAIL is the chip bridge


def test_v2a_negatives_still_fail():
    doc, task, cuts = sever_remove("goal_eval_v2a")
    local = task.sample_particles()[0]
    codes = sever_then_remove_codes(task)
    split = codes + (np.where(chip_values(task, 1) == 2, 2, np.where(local[:, 0] > .024, 2, 1))-chip_values(task, 1))*3**2
    result = run(doc, task, split, cuts)
    assert result["verdict"] == "FAIL"
    assert not result["outcome"]["steps"][1]["outcome"]["checks"]["single_substrate_component_per_prior_piece"]
    misplaced = codes + (np.where(local[:, 0] > -.016, 2, 1)-sever_values(task))
    result = run(doc, task, misplaced, cuts)
    assert result["verdict"] == "FAIL" and not result["outcome"]["steps"][0]["outcome"]["checks"]["cut_at_declared_planes"]


def test_retention_is_not_scored_for_a_protrusion_an_earlier_step_removed():
    doc = sequence([remove_step(["p0"], "remove_p0"), remove_step(["p1"], "remove_p1"), trim_step([-.065, -.035], "trim_p0")],
                   goal_id="remove_remove_trim", version="goal_eval_v2a")
    task = task_for(doc, 3)
    codes = digit_codes(task, [(0, local_chip(task, 0)), (1, local_chip(task, 1)), (2, np.zeros(len(local_chip(task, 0)), int))])
    result = run(doc, task, codes, 3)
    second = result["outcome"]["steps"][1]["outcome"]
    p0 = next(row for row in second["metrics"]["items"] if row["id"] == "p0")
    assert p0["retention_not_applicable_removed_earlier"] and p0["pass"] and p0["removed_by_earlier_steps_mass_fraction"] == 1.
    trim = result["outcome"]["steps"][2]["outcome"]
    assert trim["checks"]["material_above_plane_removed"]                       # removed earlier counts as removed
    assert trim["metrics"]["removed_by_earlier_steps_above_plane_mass_fraction"] == 1.
    assert result["verdict"] == "PASS", goals_checks(result)


def goals_checks(result):
    return {row["goal_id"]: {k: v for k, v in row["outcome"]["checks"].items() if not v} for row in result["outcome"]["steps"]}
