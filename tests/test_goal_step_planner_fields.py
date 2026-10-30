"""WP-12 Phase 6 part A: opt-in sequence-step fields ``planner_offset_local_m`` and ``planner`` (per-step options).

Both are read by goal_planner only; every evaluator ignores them. Goals that do not declare them keep their
normalized form, compiled task ids, plan reports and waypoints byte-identical (pinned against the tree before the
change in ``tests/fixtures/wp12_goal_step_fields_pin.json``; regenerate ONLY from an unmodified checkout with
``python tests/test_goal_step_planner_fields.py --write-pin``).
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from robotic_cutting_sim import goal_planner as gp  # noqa: E402
from robotic_cutting_sim import goals as goals_mod  # noqa: E402
from robotic_cutting_sim import scene_config as sc  # noqa: E402
from robotic_cutting_sim.goals import step_digit_allocation, validate_goal  # noqa: E402
from robotic_cutting_sim.scene_config import SceneConfigError  # noqa: E402
from robotic_cutting_sim.tool_profile import PROFILES  # noqa: E402

PIN = ROOT/"tests/fixtures/wp12_goal_step_fields_pin.json"
# The phase5 pin sections were computed from the DRAFT benchmark as committed in 73c6e43; the DRAFT itself is
# re-declared later (xhigh decision D5), so the pin reads that exact snapshot, not the live DRAFT.
PIN_DRAFT = ROOT/"tests/fixtures/wp12_benchmark_v1_DRAFT_pre_d5.json"
WP11_SCENES, WP11_GOALS = ROOT/"configs/wp11/scenes", ROOT/"configs/wp11/goals"
WP12_SCENES, WP12_GOALS = ROOT/"configs/wp12/scenes", ROOT/"configs/wp12/goals"
PIN_PAIRS = [
    (WP11_SCENES/"bar_160x30x20.json", WP11_GOALS/"sever_mid.json"),
    (WP11_SCENES/"bar_160x30x20.json", WP11_GOALS/"divide_4.json"),
    (WP11_SCENES/"bar_160x30x20.json", WP11_GOALS/"slice_then_push.json"),
    (WP11_SCENES/"bar_160x30x20.json", WP11_GOALS/"trim_plane_z20.json"),
    (WP11_SCENES/"bar_slotted_board_4slots.json", WP11_GOALS/"divide_4.json"),
    (WP11_SCENES/"bar_slotted_board_4slots.json", WP11_GOALS/"slice_then_push.json"),
    (WP11_SCENES/"bar_three_bumps.json", WP11_GOALS/"remove_all_protrusions.json"),
    (WP11_SCENES/"trim01_wp11_workcell.json", WP11_GOALS/"trim_bump_rho10.json"),
    (WP11_SCENES/"trim01_wp11_workcell.json", WP11_GOALS/"trim_bump_rho10_clear.json"),
    (WP11_SCENES/"trim01_wp11_workcell.json", WP11_GOALS/"trim_plane_z20.json"),
    (WP12_SCENES/"bar_flat_board.json", WP11_GOALS/"sever_mid.json"),
    (WP12_SCENES/"bar_flat_board.json", WP11_GOALS/"slice_then_push.json"),
    (WP12_SCENES/"bar_flat_board.json", WP12_GOALS/"divide_5_slices.json"),
    (WP12_SCENES/"bar_flat_board.json", WP12_GOALS/"partial_cut_mid_depth.json"),
    (WP12_SCENES/"bar_flat_board_slotted.json", WP12_GOALS/"divide_5_slices.json"),
    (WP12_SCENES/"bar_three_bumps_flat_board.json", WP12_GOALS/"remove_all_protrusions_rho10.json"),
    (WP12_SCENES/"trim01_flat_board.json", WP11_GOALS/"trim_bump_rho10.json"),
    (WP12_SCENES/"trim01_flat_board.json", WP11_GOALS/"trim_plane_z20.json"),
]


def _key(scene, goal):
    return f"{Path(scene).parent.parent.name}/{Path(scene).stem}|{Path(goal).parent.parent.name}/{Path(goal).stem}"


def _waypoint_rows(waypoints):
    return [{**asdict(w), "pose": [float(v) for v in w.pose]} for w in waypoints]


def pin_row(scene, goal):
    """Normalized goal, plan report, waypoints and compiled task id of one (scene, goal) pair (or the compile error)."""
    scene_doc = sc.resolve_scene(scene)
    goal_doc = validate_goal(goal)
    profile = PROFILES[scene_doc["tool"]["profile_id"]]
    row = {"goal_sha256": sc.sha256_json(goal_doc)}
    try:
        waypoints, report = gp.plan_with_report(scene_doc, goal_doc, profile)
        row["waypoints_sha256"] = sc.sha256_json(_waypoint_rows(waypoints))
        row["plan_report_sha256"] = sc.sha256_json(sc._jsonable(report))
    except Exception as error:  # pinned as is
        row["plan_error"] = f"{type(error).__name__}: {error}"
    try:
        row["task_id"] = sc.compile_task(scene, goal).task_id
    except Exception as error:
        row["compile_error"] = f"{type(error).__name__}: {error}"
    return row


def pin_phase5_rows():
    """Compiled ids of the first dev sample of every Phase 5 family/variant reached by dev seeds 0, 1."""
    from robotic_cutting_sim import generalization_wp12 as W
    config = W.load_benchmark(PIN_DRAFT)
    out = {}
    for family in sorted(config["families"]):
        for seed in W.seed_list(config, family, "dev", "ID")[:2]:
            sample = W.build_sample(config, family, "ID", seed, split="dev")
            row = {"variant": sample["variant"], "goal_sha256": sc.sha256_json(validate_goal(sample["goal"]))}
            try:
                row["task_id"] = W.compile_sample(config, sample)[0].task_id
            except Exception as error:
                row["compile_error"] = f"{type(error).__name__}: {error}"
            out[sample["sample_id"]] = row
    return out


def pin_phase5_goals(families=("A", "B", "C", "D", "E", "F", "H")):
    """Normalized-goal sha256 of every declared sample of the families whose variants did not change (no compile)."""
    from robotic_cutting_sim import generalization_wp12 as W
    config = W.load_benchmark(PIN_DRAFT)
    out = {}
    for family in families:
        for split in W.SPLITS:
            for range_set in W.RANGE_SETS:
                for seed in W.seed_list(config, family, split, range_set):
                    sample = W.build_sample(config, family, range_set, seed, split=split)
                    out[sample["sample_id"]] = sc.sha256_json(validate_goal(sample["goal"]))
    return out


def compute_pin():
    return {"pairs": {_key(s, g): pin_row(s, g) for s, g in PIN_PAIRS}, "phase5_dev": pin_phase5_rows(),
            "phase5_goals": pin_phase5_goals()}


# ----------------------------------------------------------------------------- goals

BAR = WP12_SCENES/"bar_flat_board.json"
X1, X2, PUSH = 0., .04, .02   # node planes of the synthetic 8 mm grid below and of the 4 mm solver grid


def _sever(x, goal_id, **extra):
    return {"type": "sever", "goal_id": goal_id, "planes": [{"origin_local_m": [x, 0., 0.], "normal_local": [1., 0., 0.]}],
            "expected_pieces": 2, "minimum_part_mass_fraction": .1, **extra}


def _push(x=X1, d=PUSH):
    return {"type": "move_piece", "goal_id": "push", "direction_local": [1., 0., 0.], "distance_m": d,
            "selector": {"side_of_plane": {"origin_local_m": [x, 0., 0.], "normal_local": [1., 0., 0.], "side": "positive"}}}


def reposition_goal(offset=(PUSH, 0., 0.), version=None, planner=None, goal_id="cut_repos_cut", second_extra=None):
    second = _sever(X2, "sever2", **({} if offset is None else {"planner_offset_local_m": list(offset)}), **(second_extra or {}))
    doc = {"schema": "rcs_goal_v1", "goal_id": goal_id, "type": "sequence", "episode": {"duration_s": 60.},
           "steps": [_sever(X1, "sever1"), _push(), second]}
    if version is not None:
        doc["evaluator_version"] = version
    if planner is not None:
        doc["planner"] = planner
    return doc


def _plan(scene, goal):
    scene_doc = sc.resolve_scene(scene)
    return gp.plan_with_report(scene_doc, goal, PROFILES[scene_doc["tool"]["profile_id"]])


def test_pin_existing_goals_unchanged():
    """Normalized goals, plan reports, waypoints and compiled ids of goals without the new fields equal the pre-change tree."""
    pin = json.loads(PIN.read_text())
    now = compute_pin()
    for section in ("pairs", "phase5_dev", "phase5_goals"):
        assert set(now[section]) == set(pin[section]), section
        for key, row in pin[section].items():
            assert now[section][key] == row, (section, key)


def test_fields_are_opt_in_and_kept_only_when_declared():
    plain = validate_goal(reposition_goal(offset=None))
    for step in plain["steps"]:
        assert "planner_offset_local_m" not in step and "planner" not in step
    shifted = validate_goal(reposition_goal())
    assert shifted["steps"][2]["planner_offset_local_m"] == [PUSH, 0., 0.]
    assert "planner_offset_local_m" not in shifted["steps"][0]
    without = {**shifted, "steps": [{k: v for k, v in step.items() if k != "planner_offset_local_m"} for step in shifted["steps"]]}
    assert without == plain   # the only difference is the declared key
    staged = validate_goal(reposition_goal(offset=None, second_extra={"planner": {"cut_speed_m_s": .005, "hold_s": 0}}))
    assert staged["steps"][2]["planner"] == {"cut_speed_m_s": .005, "hold_s": 0.}


@pytest.mark.parametrize("offset, message", [
    ([.01, 0.], "list of 3"), ([.01, 0., 0., 0.], "list of 3"), ([float("nan"), 0., 0.], "finite"),
    ([float("inf"), 0., 0.], "finite"), ([.21, 0., 0.], "norm"), ([.15, .15, 0.], "norm"), ("0.01", "list of 3"),
    ([True, 0., 0.], "finite"),
])
def test_offset_validation_refuses(offset, message):
    with pytest.raises(SceneConfigError, match=message):
        validate_goal(reposition_goal(offset=offset))


def test_offset_accepted_values_and_any_step_type():
    assert validate_goal(reposition_goal(offset=(.2, 0., 0.)))["steps"][2]["planner_offset_local_m"] == [.2, 0., 0.]
    assert validate_goal(reposition_goal(offset=(0, -.01, .005)))["steps"][2]["planner_offset_local_m"] == [0., -.01, .005]
    doc = reposition_goal(offset=None)
    for step in doc["steps"]:
        step["planner_offset_local_m"] = [0., .001, 0.]
    doc["steps"].append({"type": "trim_plane", "goal_id": "trim", "planner_offset_local_m": [0., 0., .001],
                         "planner": {"rake_deg": 12.}, "plane": {"origin_local_m": [0., 0., .015], "normal_local": [0., 0., 1.]}})
    doc["steps"].append({"type": "remove_protrusions", "goal_id": "rp", "planner": {"skim_direction_local": [-1., 0., 0.]},
                         "planner_offset_local_m": [0., 0., 0.]})
    doc["steps"].append({"type": "divide", "goal_id": "dv", "pieces": 2, "planner_offset_local_m": [0., 0., 0.]})
    steps = validate_goal(doc)["steps"]
    assert all("planner_offset_local_m" in step for step in steps)


def test_fields_are_refused_at_the_top_level_and_ref_steps_keep_dropping_planner(tmp_path):
    top = {**_sever(X1, "sever_top"), "schema": "rcs_goal_v1", "episode": {"duration_s": 8.},
           "planner_offset_local_m": [.01, 0., 0.]}
    with pytest.raises(SceneConfigError, match="unknown keys"):
        validate_goal(top)
    ref = {**_sever(X1, "sever_ref"), "schema": "rcs_goal_v1", "episode": {"duration_s": 8.}, "planner": {"rake_deg": 5.}}
    (tmp_path/"ref.json").write_text(json.dumps(ref))
    doc = {"schema": "rcs_goal_v1", "goal_id": "seq_ref", "type": "sequence", "episode": {"duration_s": 20.},
           "steps": [{"ref": "ref.json"}]}
    (tmp_path/"seq.json").write_text(json.dumps(doc))
    step = validate_goal(tmp_path/"seq.json")["steps"][0]
    assert "planner" not in step and "planner_offset_local_m" not in step


@pytest.mark.parametrize("planner, message", [
    ({"controller_profile": "path_v2"}, "goal-wide"), ({"home_height_m": .05}, "goal-wide"),
    ({"home_position_local_m": [0., 0., .1]}, "goal-wide"), ({"home_orientation_local_xyzw": [0, 0, 0, 1]}, "goal-wide"),
    ({"max_translation_m": .001}, "goal-wide"), ({"max_rotation_deg": 1.}, "goal-wide"),
    ({"perception": "wrist_depth_relocate_v1"}, "goal-wide"), ({"blade_colour": "red"}, "unknown"),
    ({}, "nonempty"), ("rake_deg", "nonempty object"), ({"rake_deg": "ten"}, "finite number"),
    ({"rake_deg": float("nan")}, "finite"), ({"cut_speed_m_s": 0.}, "positive"), ({"hold_s": -1.}, "nonnegative"),
    ({"protrusion_order": "random"}, "protrusion_order"), ({"cut_extent": "everything"}, "cut_extent"),
    ({"skim_direction_local": [0., 0., 1.]}, "horizontal"), ({"skim_direction_local": [1., 0.]}, "list of 3"),
])
def test_step_planner_validation_refuses(planner, message):
    with pytest.raises(SceneConfigError, match=message):
        validate_goal(reposition_goal(offset=None, second_extra={"planner": planner}))


def test_step_planner_accepts_every_per_operation_option():
    values = {"rake_deg": 20., "skew_deg": 5., "skim_speed_m_s": .02, "skim_margin_m": .005, "skim_direction_local": [-1., 0., 0.],
              "protrusion_order": "goal", "cut_speed_m_s": .008, "cut_clearance_m": .006, "overtravel_m": -.002, "hold_s": .2,
              "kerf_speed_m_s": .01, "saw_amplitude_m": .001, "saw_period_s": .5, "safe_height_m": .025,
              "push_release_m": .001, "cut_extent": "plane_section"}
    assert set(values) == set(gp.STEP_PLANNER_OPTIONS)
    step = validate_goal(reposition_goal(offset=None, second_extra={"planner": values}))["steps"][2]
    assert step["planner"] == values


def test_digit_allocation_and_goal_planes_ignore_the_fields():
    scene = sc.resolve_scene(BAR)
    plain = validate_goal(reposition_goal(offset=None))
    both = validate_goal(reposition_goal(second_extra={"planner": {"cut_speed_m_s": .005}}))
    assert step_digit_allocation(both, scene) == step_digit_allocation(plain, scene) == [[0], [], [1]]
    for a, b in zip(plain["steps"], both["steps"]):
        assert [(o.tolist(), n.tolist()) for o, n in goals_mod.goal_planes(a, scene)] == \
            [(o.tolist(), n.tolist()) for o, n in goals_mod.goal_planes(b, scene)]


# ----------------------------------------------------------------------------- planner

def _by_op(waypoints):
    out = {}
    for w in waypoints:
        out.setdefault(w.label.split(":")[-1] if ":" in w.label else w.label, []).append(w)
    return out


@pytest.mark.parametrize("profile", ["v1", "path_v2"])
@pytest.mark.parametrize("yaw", [0., 1.5])
def test_offset_translates_exactly_the_steps_waypoints(profile, yaw):
    scene = json.loads(BAR.read_text())
    scene["workpiece"]["world_pose"]["yaw_deg"] = yaw
    scene["supports"]["workcell"]["overrides"] = {"cutting_board": {"yaw_deg": yaw}}
    from robotic_cutting_sim.flat_board_workcell import place_on_board
    scene = place_on_board(scene)
    offset = np.array([PUSH, .003, -.001])
    plain_wp, plain_report = _plan(scene, reposition_goal(offset=None, planner={"controller_profile": profile}))
    shift_wp, shift_report = _plan(scene, reposition_goal(offset=offset, planner={"controller_profile": profile}))
    assert [w.label for w in plain_wp] == [w.label for w in shift_wp]
    R = gp._Frame(sc.resolve_scene(scene)).R
    world_offset = R.apply(offset)
    a, b = _by_op(plain_wp), _by_op(shift_wp)
    assert set(a) == {"HOME", "cut0", "move1", "cut1", "RETURN_HOME"}
    for op in a:
        for u, v in zip(a[op], b[op]):
            pu, pv = np.asarray(u.pose), np.asarray(v.pose)
            if op == "cut1":
                np.testing.assert_allclose(pv[:3]-pu[:3], world_offset, atol=1e-12)
                np.testing.assert_allclose(pv[3:], pu[3:], atol=0)
            else:
                assert u.pose == v.pose
            assert {**asdict(u), "pose": None} == {**asdict(v), "pose": None}
    assert plain_report["digits_used"] == shift_report["digits_used"] == 2
    for row_a, row_b in zip(plain_report["operations"], shift_report["operations"]):
        extra = {k: v for k, v in row_b.items() if k not in row_a}
        assert extra == ({"planner_offset_local_m": offset.tolist()} if row_b["step"] == 2 else {})
        assert {k: v for k, v in row_b.items() if k in row_a} == row_a   # report geometry stays the declared one


def test_per_step_rake_gives_the_declared_rho_per_trim_step():
    scene = WP12_SCENES/"bar_three_bumps_flat_board.json"
    rows = sc.resolve_scene(scene)["workpiece"]["protrusions"][:2]
    H = sc.resolve_scene(scene)["workpiece"]["base_box_m"]["height_m"]
    steps = [{"type": "trim_plane", "goal_id": f"trim_{r['id']}", "planner": planner,
              "plane": {"origin_local_m": [0., 0., H], "normal_local": [0., 0., 1.]},
              "region_local_m": {"x": [r["x_m"]-r["radius_x_m"]-.003, r["x_m"]+r["radius_x_m"]+.003]}}
             for r, planner in zip(rows, ({"rake_deg": 10.}, {"rake_deg": 30., "skim_direction_local": [-1., 0., 0.]}))]
    doc = {"schema": "rcs_goal_v1", "goal_id": "trim_trim_two_rakes", "type": "sequence", "episode": {"duration_s": 60.},
           "planner": {"rake_deg": 20.}, "steps": steps}
    waypoints, report = _plan(scene, doc)
    trims = [row for row in report["operations"] if row["kind"] == "surface_trim"]
    assert [row["declared_rake_deg"] for row in trims] == [10., 30.]
    for row, rho in zip(trims, (10., 30.)):
        assert row["orientation_metrics"]["rho_deg"] == pytest.approx(rho, abs=1e-9)
    assert trims[0]["direction_local"][0] > 0 > trims[1]["direction_local"][0]
    assert trims[0]["step_planner_options"] == {"rake_deg": 10.}
    assert report["options"]["rake_deg"] == 20.   # the goal-wide options are unchanged
    # Each step equals the same trim planned alone with those options goal-wide.
    for k, step in enumerate(steps):
        alone = {**doc, "goal_id": f"alone{k}", "planner": {**doc["planner"], **step["planner"]},
                 "steps": [{key: v for key, v in step.items() if key != "planner"}]}
        _, single = _plan(scene, alone)
        row = single["operations"][0]
        for key in ("start_local_m", "end_local_m", "orientation_metrics", "direction_local", "declared_rake_deg"):
            assert row[key] == trims[k][key], key
    # Without per-step options both trims use the goal rake.
    plain = {**doc, "steps": [{key: v for key, v in step.items() if key != "planner"} for step in steps]}
    _, base = _plan(scene, plain)
    assert [row["declared_rake_deg"] for row in base["operations"]] == [20., 20.]
    assert all("step_planner_options" not in row for row in base["operations"])


def test_step_planner_only_changes_its_own_step():
    plain_wp, plain_report = _plan(BAR, reposition_goal(offset=None))
    slow_wp, slow_report = _plan(BAR, reposition_goal(offset=None, second_extra={"planner": {"cut_speed_m_s": .004, "hold_s": .1}}))
    a, b = _by_op(plain_wp), _by_op(slow_wp)
    for op in ("HOME", "cut0", "move1", "RETURN_HOME"):
        assert [asdict(w) for w in a[op]] == [asdict(w) for w in b[op]]
    speeds = {w.speed_m_s for w in b["cut1"] if w.cut}
    holds = {w.hold_s for w in b["cut1"] if w.label.startswith("HOLD")}
    assert speeds == {.004} and holds == {.1}
    assert slow_report["operations"][-1]["step_planner_options"] == {"cut_speed_m_s": .004, "hold_s": .1}


def test_compiled_task_id_changes_only_when_a_field_is_declared():
    plain = sc.compile_task(BAR, reposition_goal(offset=None), particle_spacing_m=.004)
    again = sc.compile_task(BAR, reposition_goal(offset=None), particle_spacing_m=.004)
    shifted = sc.compile_task(BAR, reposition_goal(), particle_spacing_m=.004)
    assert plain.task_id == again.task_id != shifted.task_id
    assert shifted.task_id == sc.task_from_payload(sc.configured_task_payload(shifted)).task_id


# ----------------------------------------------------------------------------- evaluators ignore both fields

def _synthetic(task):
    """Packed two-digit synthetic recording: digit 0 at X1, digit 1 at X2, the +x part of X1 moved by PUSH."""
    sys.path.insert(0, str(ROOT/"tests"))
    from test_scene_goal_config import recording, to_packed
    local = task.sample_particles()[0]
    codes = np.where(local[:, 0] > X1, 2, 1)+3*np.where(local[:, 0] > X2, 2, 1)
    moved = local[:, 0] > X1
    data = recording(task, to_packed(codes, 2), frames=[0., task.duration_s/2, task.duration_s],
                     moves=[(1, moved, [PUSH, 0., 0.]), (2, moved, [PUSH, 0., 0.])], dx=2*task.particle_spacing_m)
    return {**data, "code_encoding": "packed2"}


def _scrub(result):
    text = json.dumps(result, sort_keys=True, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                      else v.item() if isinstance(v, np.generic) else str(v))
    return text


@pytest.mark.parametrize("version", [None, "goal_eval_v2", "goal_eval_v2a"])
def test_evaluate_goal_is_identical_with_and_without_the_fields(version):
    scene = json.loads(BAR.read_text())
    scene["solver"]["max_cuts"] = 2
    plain_goal = reposition_goal(offset=None, version=version)
    field_goal = reposition_goal(version=version, second_extra={"planner": {"cut_speed_m_s": .006}})
    plain = sc.compile_task(scene, plain_goal, particle_spacing_m=.004)
    fields = sc.compile_task(scene, field_goal, particle_spacing_m=.004)
    data = _synthetic(plain)
    a = plain.evaluate_goal(deepcopy(data))
    b = fields.evaluate_goal(deepcopy(data))
    assert a["outcome"]["status"] in ("PASS", "FAIL")
    assert _scrub(a).replace(plain.task_id, "<id>") == _scrub(b).replace(fields.task_id, "<id>")
    assert _scrub(goals_mod.evaluate_goal(validate_goal(plain_goal), deepcopy(data), scene=sc.resolve_scene(scene))) == \
        _scrub(goals_mod.evaluate_goal(validate_goal(field_goal), deepcopy(data), scene=sc.resolve_scene(scene)))


if __name__ == "__main__" and "--write-pin" in sys.argv:
    import subprocess
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    doc = {"source": f"Computed at {head} before the opt-in step fields existed (goals.py / goal_planner.py / "
                     "generalization_wp12.py unmodified); tests/test_goal_step_planner_fields.py compute_pin()",
           **compute_pin()}
    PIN.write_text(json.dumps(doc, indent=1, sort_keys=True)+"\n")
    print(f"wrote {PIN}: {len(doc['pairs'])} pairs, {len(doc['phase5_dev'])} phase5 dev samples")
    sys.exit(0)
