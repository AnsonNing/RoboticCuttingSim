"""WP-12 Phase 4 E2 envelope-confirmation protocol and runner (scripts/run_wp12_phase4_e2.py): CPU only, no physics."""
from pathlib import Path
import math
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))

import run_wp12_phase4_e2 as E2  # noqa: E402
from robotic_cutting_sim import generalization_wp12 as W  # noqa: E402
from robotic_cutting_sim.goals import validate_goal  # noqa: E402


@pytest.fixture(scope="module")
def protocol():
    return E2.load_protocol()


DECISION_ROWS = {  # decision 'Next steps' item 3, in order
    "SLOT_p20", "SLOT_m20", "SLOT_p25", "SOLID_p02", "SOLID_p05", "YAW_p15", "YAW_m15", "ADV_p15", "ADV_m15",
    "CUT_v08", "CUT_v12", "SKIM_v20", "SKIM_v30", "SKIM_r35", "SKIM_psi_p10", "SKIM_psi_m10", "TRIM_PLANE",
    "TRIM_TRIM", "CUT_PUSH", "CUT_PUSH_TRIM"}


def test_protocol_rows_are_the_decision_list(protocol):
    ids = [r["id"] for r in protocol["rows"]]
    assert set(ids) == DECISION_ROWS and len(ids) == 20
    assert protocol["budget"]["gpu_hours_cap"] <= 3.0 and protocol["budget"]["max_concurrent_jobs"] <= 2
    assert protocol["common"]["planner"]["controller_profile"] == "path_v2"
    assert protocol["common"]["solver"]["cutting_law"] == 8 and protocol["common"]["material_dt_s"] == 1e-4
    assert protocol["common"]["solver"]["particle_spacing_m"] == .001 and protocol["common"]["material_seed"] == 42
    assert protocol["declared_predictions"]["SOLID_p05"] == "FAIL"
    for row in protocol["rows"]:
        assert row["prediction"]["verdict"] in ("CONFIRMED", "FAIL", "NO_DIRECTIONAL_PREDICTION",
                                                "NOT_EXECUTABLE_BOARD_COLLISION", "PASS_KINEMATIC_PREFLIGHT_FAIL")


def test_every_row_builds_a_valid_scene_and_goal(protocol):
    for row in protocol["rows"]:
        sample = E2.build_row_sample(protocol, row)
        goal = validate_goal(sample["goal"])
        assert goal["evaluator_version"] == "goal_eval_v2a" and goal["planner"]["controller_profile"] == "path_v2"
        assert sample["scene"]["supports"]["kind"] == "flat_board"
        assert not W.geometry_checks(sample, protocol["common"]), row["id"]
        board = sample["scene"]["supports"]["workcell"]["overrides"]["cutting_board"]
        assert (board.get("kind") == "slotted") == row["id"].startswith("SLOT"), row["id"]
        params = sample["envelope_parameters"]
        for key, value in row["values"].items():
            if key in ("cut_speed_m_s", "skim_speed_m_s"):
                assert params[key] == value


def test_tilted_planes_pass_through_x0_at_the_board_top(protocol):
    for row in protocol["rows"]:
        if row["variant"] not in ("tilt_slotted", "tilt_solid"):
            continue
        sample = E2.build_row_sample(protocol, row)
        plane = sample["geometry"]["planes"][0]
        assert abs(W._plane_x_at(plane, 0.)) < 1e-9, row["id"]
        tilt = row["values"].get("cut_tilt_deg", row["values"].get("cut_tilt_solid_deg"))
        assert math.isclose(sample["envelope_parameters"]["sever_tilt_abs_deg_max"], abs(tilt), abs_tol=1e-9)
        if row["variant"] == "tilt_slotted":
            slot = sample["scene"]["supports"]["workcell"]["overrides"]["cutting_board"]["slots"][0]
            pose = sample["scene"]["workpiece"]["world_pose"]
            assert abs(slot["x_board_m"]-W._board_frame_x(protocol["common"], (*pose["translation_m"][:2], pose["yaw_deg"]), 0.)) < 1e-9


def test_cut_push_trim_trims_the_stationary_piece(protocol):
    row = next(r for r in protocol["rows"] if r["id"] == "CUT_PUSH_TRIM")
    sample = E2.build_row_sample(protocol, row)
    steps = validate_goal(sample["goal"])["steps"]
    assert [s["type"] for s in steps] == ["sever", "move_piece", "remove_protrusions"]
    assert all("planner_offset_local_m" not in s for s in steps)
    p0 = sample["scene"]["workpiece"]["protrusions"][0]
    assert p0["x_m"]+p0["radius_x_m"] < steps[0]["planes"][0]["origin_local_m"][0]
    assert steps[1]["selector"]["side_of_plane"]["side"] == "positive"


def test_board_collision_rule():
    def feas(failing, worst):
        return {"checks": {"preflight": {"receipt": {"path": {"failing": failing, "worst": worst}}}}}
    hit = feas(["blade_collision"], {"blade_collision": {"slack": -.0155, "label": "HOLD:cut0",
                                                         "pair": ["tool_blade", "static_flat_board"]}})
    assert E2.board_collision(hit)[0]["check"] == "blade_collision"
    wp = feas(["blade_collision"], {"blade_collision": {"slack": -.005, "pair": ["tool_blade", "static_workpiece_bump"]}})
    assert E2.board_collision(wp) == []
    assert E2.board_collision(feas([], {})) == [] and E2.board_collision(None) == []
    verdict, why = E2.verdict({"kind": "sever", "expected_pieces": 2}, hit, {"status": "COMPLETE"})
    assert verdict == "NOT_EXECUTABLE_BOARD_COLLISION"


def _run(**kw):
    base = {"status": "COMPLETE", "label": "E2_FORMAL", "partial": False, "policy_done": True, "solver_invalid": None,
            "verdict": "PASS", "mechanical_components": 2, "direct_code0_bridges_by_digit": {"0": 0},
            "direct_code0_bridges_total": 0}
    return {**base, **kw}


def test_verdict_rules():
    decl = {"kind": "sever", "expected_pieces": 2, "bridge_digits": [0]}
    ok = {"checks": {"preflight": {"status": "PASS"}}}
    bad = {"checks": {"preflight": {"status": "FAIL", "receipt": {"path": {"failing": ["hand_table"], "worst": {}}}}}}
    assert E2.verdict(decl, ok, _run())[0] == "CONFIRMED"
    assert E2.verdict(decl, bad, _run())[0] == "PASS_KINEMATIC_PREFLIGHT_FAIL"
    assert E2.verdict(decl, ok, _run(mechanical_components=1))[0] == "FAIL"
    assert E2.verdict(decl, ok, _run(direct_code0_bridges_by_digit={"0": 3}))[0] == "FAIL"
    assert E2.verdict(decl, ok, _run(direct_code0_bridges_by_digit={"0": 0, "1": 7}))[0] == "CONFIRMED"   # trim digit
    assert E2.verdict(decl, ok, _run(verdict="FAIL"))[0] == "FAIL"
    assert E2.verdict(decl, ok, _run(label="DEV"))[0] == "NOT_RUN"
    assert E2.verdict(decl, ok, _run(partial=True))[0] == "NOT_RUN"
    assert E2.verdict(decl, ok, {"status": "NOT_RUN"})[0] == "NOT_RUN"
    assert E2.verdict(decl, ok, {"status": "NO_RESULT"})[0] == "ERROR"
    assert E2.verdict(decl, ok, _run(status="ABORTED"))[0] == "ERROR"
    trim = {"kind": "trim"}
    assert E2.verdict(trim, ok, _run(mechanical_components=1, direct_code0_bridges_by_digit={"0": 9}))[0] == "CONFIRMED"


def test_armed_ends_from_local_poses():
    q = Rotation.identity().as_quat()
    tilted = Rotation.from_euler("x", 15, degrees=True).as_quat()   # edge line (column 1) tilted out of the horizontal
    poses = np.array([[0, 0, .03, *q], [0, 0, .01, *q], [0, 0, .0002, *q], [0, 0, .0002, *q], [0, 0, .005, *tilted],
                      [0, 0, .004, *tilted], [0, 0, .01, *q]])
    cut = np.array([-1, 0, 0, -1, 1, 1, -1])
    ends = E2._armed_ends({"step_cut": cut, "step_pose_end_local": poses, "blade_half_width_m": .06})
    assert [(e["cut_index"], e["first_step"], e["last_step"]) for e in ends] == [(0, 2, 3), (1, 5, 6)]
    assert math.isclose(ends[0]["lowest_edge_point_local_z_mm"], .2, abs_tol=1e-9)
    assert math.isclose(ends[1]["lowest_edge_point_local_z_mm"], 4.-60*math.sin(math.radians(15)), abs_tol=1e-6)


def test_run_refuses_undeclared_rows(tmp_path):
    with pytest.raises(SystemExit, match="not a declared E2 row"):
        E2.main(["run", "--row", "NOPE", "--out", str(tmp_path/"x")])
