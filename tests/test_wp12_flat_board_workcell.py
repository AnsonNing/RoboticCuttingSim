"""WP-12 Phase 2: blade_master_flat_board_v1 workcell (ADR 0017), CPU only.

One record is the single source of the table, the board (TLMPM fixture =
PhysX collider = visual box), the support plane, HOME, the action workspace and
the preflight; wp11_workcell_v1 compiles exactly as before.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from robotic_cutting_sim import flat_board_workcell as fb
from robotic_cutting_sim import scene_config as sc
from robotic_cutting_sim.control import ActionLimits, bound_target
from robotic_cutting_sim.native_fixture import fixture_authoring_plan

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT/"configs/wp12/scenes"
GOALS = ROOT/"configs/wp12/goals"
WP11_GOALS = ROOT/"configs/wp11/goals"
PIN = ROOT/"tests/fixtures/wp12_wp11_compile_pin.json"


def scene(name="bar_flat_board"):
    return sc.load_json(SCENES/f"{name}.json")


def with_overrides(doc, **overrides):
    doc = deepcopy(doc)
    merged = deepcopy(doc["supports"]["workcell"].get("overrides", {}))
    for key, value in overrides.items():
        merged.setdefault(key, {}).update(value)
    doc["supports"]["workcell"]["overrides"] = merged
    return doc


@pytest.fixture(scope="module")
def cut_task():
    return sc.compile_task(scene(), WP11_GOALS/"sever_mid.json")


def world_box(task, box):
    return task.points_to_world(np.asarray(box.center_m))[None][0], np.asarray(box.half_extents_m)


def test_board_bottom_on_table_and_workpiece_on_board_top(cut_task):
    record = cut_task.flat_board_record
    geometry = cut_task.workcell_geometry()
    table_center, table_half = cut_task.table_box_world
    table_top = table_center[2]+table_half[2]
    assert table_top == pytest.approx(record["table"]["top_world_z_m"], abs=1e-12) == pytest.approx(-0.01)
    assert cut_task.table_top_world_z_m == pytest.approx(table_top, abs=1e-12)
    (board,) = cut_task.fixture_boxes_local
    center, half = world_box(cut_task, board)
    assert center[2]-half[2] == pytest.approx(table_top, abs=1e-12)              # board bottom = table top
    board_top = center[2]+half[2]
    assert board_top == pytest.approx(geometry["board_top_world_z_m"], abs=1e-12)
    assert cut_task.world_from_workpiece_xyzw[2] == pytest.approx(board_top, abs=1e-12)  # workpiece bottom = board top
    assert board.center_m[2]+board.half_extents_m[2] == pytest.approx(0., abs=1e-12)    # local support plane z = 0
    particles, _, _ = cut_task.sample_particles()
    assert particles[:, 2].min() >= 0.                                              # no initial penetration
    assert particles[:, 2].min() <= cut_task.particle_spacing_m                     # no floating gap


def test_no_riser_or_pedestal_anywhere(cut_task):
    boxes = cut_task.fixture_boxes_local
    assert len(boxes) == 1 and boxes[0].name == "flat_board"
    assert {box.role for box in boxes} == {"board"}
    assert not any("riser" in box.name or "pedestal" in box.name for box in boxes)
    assert {row["role"] for row in cut_task.fixture_materials} == {"board"}
    geometry = cut_task.workcell_geometry()
    assert geometry["risers"] == 0 and geometry["pedestals"] == 0
    # The planner's support geometry is the same board (no hidden support either).
    assert sc.raw_support_boxes(cut_task.scene) == tuple(boxes)


def test_fixture_boxes_equal_the_board_record_in_world(cut_task):
    record = cut_task.flat_board_record
    rows = fb.board_boxes_world(record)
    assert len(rows) == len(cut_task.fixture_boxes_local)
    for (name, center, half, _), box in zip(rows, cut_task.fixture_boxes_local):
        world_center, world_half = world_box(cut_task, box)
        assert box.name == name
        np.testing.assert_allclose(world_center, center, atol=1e-12)
        np.testing.assert_allclose(world_half, half, atol=0)
    board = record["cutting_board"]
    assert rows[0][2] == pytest.approx((board["length_x_m"]/2, board["width_y_m"]/2, board["thickness_m"]/2))


def test_visual_collider_and_tlmpm_from_the_same_numbers(cut_task):
    plan = fixture_authoring_plan(cut_task)
    for row, box in zip(plan["boxes"], cut_task.fixture_boxes_local):
        # Native authors one Cube per row: its scale (visual) and CollisionAPI (collider) use exactly these numbers,
        # and LiveCuttingMaterial passes the same (center, half) to the TLMPM fixture contact.
        assert row["center_m"] == list(box.center_m) and row["half_extents_m"] == list(box.half_extents_m)
        assert row["material_declared"] and row["role"] == "board"
        assert row["physics_material"]["static_friction"] == pytest.approx(cut_task.material.friction)
    assert plan["table"] is not None
    assert plan["table"]["physics_material"]["static_friction"] == pytest.approx(cut_task.material.friction)
    # PhysX board friction == TLMPM material-board friction (task material) with "task_material".
    assert cut_task.fixture_boxes_local[0].static_friction == pytest.approx(cut_task.material.friction)
    receipt = cut_task.to_dict()["workcell_geometry"]
    assert receipt["board_boxes_local"][0]["center_m"] == list(cut_task.fixture_boxes_local[0].center_m)
    assert receipt["table_box_world"]["center_m"] == list(cut_task.table_box_world[0])


def test_runner_and_preflight_read_the_declared_table():
    source = (ROOT/"src/robotic_cutting_sim/isaac_sim_runner.py").read_text()
    assert 'getattr(task,"table_box_world",None)' in source
    c2 = (ROOT/"scripts/preflight_wp11_orientations.py").read_text()
    assert "getattr(task, 'table_box_world', None)" in c2
    preflight = (ROOT/"src/robotic_cutting_sim/motion_preflight.py").read_text()
    assert 'record["preflight"]["blade_board_contact_min_distance_m"]' in preflight


def test_thickness_and_position_move_contact_plane_ik_target_and_preflight_together(cut_task):
    base = cut_task
    moved_doc = fb.place_on_board(with_overrides(scene(), cutting_board={"thickness_m": 0.03, "center_xy_world_m": [0.01, 0.03]},
                                                 table={}))
    moved_doc["scene_id"] = "bar_flat_board_thick30"
    moved = sc.compile_task(moved_doc, WP11_GOALS/"sever_mid.json")
    assert moved.world_from_workpiece_xyzw[2] == pytest.approx(base.world_from_workpiece_xyzw[2]+0.01, abs=1e-12)
    (board,) = moved.fixture_boxes_local
    assert board.center_m[2]+board.half_extents_m[2] == pytest.approx(0., abs=1e-12)   # still the local support plane
    assert board.center_m[0] == pytest.approx(0.01) and board.half_extents_m[2] == pytest.approx(0.015)
    center, half = world_box(moved, board)
    assert center[2]-half[2] == pytest.approx(-0.01, abs=1e-12) and center[2]+half[2] == pytest.approx(0.02, abs=1e-12)
    # IK targets: the armed cut ends on the (moved) board top in world; HOME is a robot pose and stays.
    def cut_end_world(task):   # deepest reference edge pose (the end of the armed through-cut)
        times = np.asarray(task.edge_position_knots_local_txyz)[:, 0]
        return min((np.asarray(task.reference_edge_pose_world(float(t))) for t in times), key=lambda pose: pose[2])
    assert cut_end_world(base)[2] == pytest.approx(base.workcell_geometry()["board_top_world_z_m"], abs=1e-9)
    assert cut_end_world(moved)[2] == pytest.approx(moved.workcell_geometry()["board_top_world_z_m"], abs=1e-9)
    np.testing.assert_allclose(base.reference_edge_pose_world(0.), moved.reference_edge_pose_world(0.), atol=1e-12)
    # Preflight geometry: the C2 checker builds its table/board boxes from these task properties.
    assert moved.table_box_world == base.table_box_world
    assert moved.flat_board_record["preflight"] == base.flat_board_record["preflight"]
    # Changing the thickness without moving the workpiece is refused (no float, no initial penetration).
    stale = with_overrides(scene(), cutting_board={"thickness_m": 0.03})
    with pytest.raises(sc.SceneInfeasible, match="workpiece_not_on_board"):
        sc.validate_scene(stale)


def test_infeasible_or_inconsistent_placements_are_refused():
    off = deepcopy(scene())
    off["workpiece"]["world_pose"]["translation_m"][1] = -0.05
    with pytest.raises(sc.SceneInfeasible, match="workpiece_off_board"):
        sc.validate_scene(off)
    yawed = deepcopy(scene())
    yawed["workpiece"]["world_pose"]["yaw_deg"] = 15.
    with pytest.raises(sc.SceneInfeasible, match="workpiece_yaw_differs_from_board"):
        sc.validate_scene(yawed)
    table = deepcopy(scene())
    table["table_top_world_z_m"] = 0.0
    with pytest.raises(sc.SceneConfigError, match="single source"):
        sc.validate_scene(table)
    hanging = with_overrides(scene(), cutting_board={"center_xy_world_m": [0., -0.02]})
    with pytest.raises(sc.SceneConfigError, match="inside the table top"):
        sc.validate_scene(hanging)
    implicit_slots = with_overrides(scene(), cutting_board={"slots": [{"x_board_m": 0., "width_m": .008}]})
    with pytest.raises(sc.SceneConfigError, match="kind 'slotted' explicitly"):
        sc.validate_scene(implicit_slots)
    through = with_overrides(scene(), cutting_board={"kind": "slotted", "slot_depth_m": 0.02,
                                                     "slots": [{"x_board_m": 0., "width_m": .008}]})
    with pytest.raises(sc.SceneConfigError, match="continuous body"):
        sc.validate_scene(through)
    legacy = deepcopy(scene())
    legacy["supports"]["x_range_m"] = [-0.09, 0.09]
    with pytest.raises(sc.SceneConfigError, match="unknown keys"):
        sc.validate_scene(legacy)
    record = deepcopy(fb.profiles()["blade_master_flat_board_v1"])
    record["preflight"]["blade_board_contact_min_distance_m"] = -1e-4
    with pytest.raises(fb.FlatBoardError, match="never enters"):
        original = fb.profiles
        try:
            fb.profiles = lambda: {"blade_master_flat_board_v1": record}
            fb.resolve({"profile": "blade_master_flat_board_v1"})
        finally:
            fb.profiles = original


def test_default_board_is_solid_and_the_slotted_variant_is_explicit():
    solid = sc.compile_task(scene(), WP11_GOALS/"sever_mid.json")
    assert solid.flat_board_record["cutting_board"]["kind"] == "solid"
    slotted = sc.compile_task(scene("bar_flat_board_slotted"), GOALS/"divide_5_slices.json")
    record = slotted.flat_board_record
    assert record["cutting_board"]["kind"] == "slotted"
    names = [box.name for box in slotted.fixture_boxes_local]
    assert names == ["flat_board_base"]+[f"flat_board_segment_{i}" for i in range(5)]
    base = slotted.fixture_boxes_local[0]
    depth = record["cutting_board"]["slot_depth_m"]
    assert base.center_m[2]+base.half_extents_m[2] == pytest.approx(-depth)          # groove floor, one continuous slab
    for box in slotted.fixture_boxes_local[1:]:
        assert box.center_m[2]+box.half_extents_m[2] == pytest.approx(0., abs=1e-12)
        assert box.center_m[2]-box.half_extents_m[2] == pytest.approx(-depth, abs=1e-12)
    # Groove x positions (board frame) match the four slice planes of the goal.
    gaps = [(a.center_m[0]+a.half_extents_m[0], b.center_m[0]-b.half_extents_m[0])
            for a, b in zip(slotted.fixture_boxes_local[1:], slotted.fixture_boxes_local[2:])]
    assert [round((lo+hi)/2, 6) for lo, hi in gaps] == [-0.048, -0.016, 0.016, 0.048]


def test_action_workspace_floor_is_the_board_top(cut_task):
    limits = cut_task.action_limits()
    assert limits.workspace_min_local_m[2] == 0.
    assert tuple(limits.workspace_max_local_m) == (0.14, 0.12, 0.22)
    T = cut_task.T_world_workpiece
    reference = np.asarray(cut_task.reference_edge_pose_world(0.))
    below = reference.copy()
    below[:3] = T[:3, :3]@np.array([0.05, 0., -0.004])+T[:3, 3]
    loose = ActionLimits(max_translation_m=1., max_rotation_rad=np.pi, workspace_min_local_m=limits.workspace_min_local_m,
                         workspace_max_local_m=limits.workspace_max_local_m)
    pose, report = bound_target(reference, below, loose, T)
    assert report["workspace_clamped"]
    assert (np.linalg.inv(T)@np.r_[pose[:3], 1.])[2] == pytest.approx(0., abs=1e-12)
    slotted = sc.compile_task(scene("bar_flat_board_slotted"), GOALS/"divide_5_slices.json")
    assert slotted.action_limits().workspace_min_local_m[2] == pytest.approx(-0.008)


def test_home_is_the_record_home_with_the_blade_hanging(cut_task):
    home = cut_task.home
    record = fb.profiles()["blade_master_flat_board_v1"]
    assert home["joint_positions"] == tuple(record["home_joint_positions"])
    assert home["workcell_profile"] == "blade_master_flat_board_v1"
    edge = np.asarray(cut_task.reference_edge_pose_world(0.))
    local = np.linalg.inv(cut_task.T_world_workpiece)@np.r_[edge[:3], 1.]
    np.testing.assert_allclose(local[:3], [0., 0., 0.188], atol=2e-6)   # same workpiece-relative HOME as wp11_workcell_v1
    from scipy.spatial.transform import Rotation
    hand = np.asarray(home["hand_pose_world_xyzw"])
    np.testing.assert_allclose(Rotation.from_quat(hand[3:]).apply([0, 0, 1]), [0, 0, -1], atol=5e-6)   # IK residual ~1 urad


@pytest.mark.parametrize("scene_name,goal,digits,min_edge_z", [
    ("bar_flat_board", WP11_GOALS/"sever_mid.json", 1, 0.0),
    ("trim01_flat_board", WP11_GOALS/"trim_bump_rho10.json", 1, 0.0202),
    ("bar_flat_board", GOALS/"divide_5_slices.json", 4, 0.0),
    ("bar_flat_board_slotted", GOALS/"divide_5_slices.json", 4, -0.004),
    ("bar_three_bumps_flat_board", GOALS/"remove_all_protrusions_rho10.json", 3, 0.0202),
])
def test_scene_goal_compile_on_the_flat_board(scene_name, goal, digits, min_edge_z):
    task = sc.compile_task(scene(scene_name), goal)
    assert task.task_id.startswith(f"CFG-{scene_name}-")
    used = sorted({c for c in task.reference_cut_indices if c >= 0})
    assert used == list(range(digits))
    knots = np.asarray(task.edge_position_knots_local_txyz, float)
    assert knots[:, 3].min() == pytest.approx(min_edge_z, abs=1e-9)
    # A solid board: no reference edge centre below the rigid board top.
    if task.flat_board_record["cutting_board"]["kind"] == "solid":
        assert knots[:, 3].min() >= -1e-12
    assert sc.task_from_payload(sc.configured_task_payload(task)).task_id == task.task_id
    if "trim" in scene_name or "bumps" in scene_name:
        assert task.controller["planner"]["rake_deg"] == 10.0


def test_wp11_workcell_compiled_output_is_unchanged():
    pins = json.loads(PIN.read_text())
    for key, pin in pins.items():
        if key == "source":
            continue
        scene_name, goal = key.split("|")
        task = sc.compile_task(ROOT/f"configs/wp11/scenes/{scene_name}.json", WP11_GOALS/f"{goal}.json")
        assert task.task_id == pin["task_id"] and task.config_sha256 == pin["config_sha256"]
        assert [[b.name, getattr(b, "role", None), list(b.center_m), list(b.half_extents_m)]
                for b in task.fixture_boxes_local] == pin["fixture_boxes"]
        assert list(task.world_from_workpiece_xyzw) == pin["world"] and task.table_top_world_z_m == pin["table_top"]
        assert (None if task.home is None else list(task.home["joint_positions"])) == pin["home_joints"]
        limits = task.action_limits()
        assert [list(limits.workspace_min_local_m), list(limits.workspace_max_local_m)] == pin["workspace"]
        knots = np.asarray(task.edge_position_knots_local_txyz, float)
        assert len(knots) == pin["knots"][0]
        assert float(knots[:, 1:].sum()) == pytest.approx(pin["knots"][1], rel=1e-12, abs=1e-12)
        assert float(knots[:, 3].min()) == pytest.approx(pin["knots"][2], abs=1e-12)
        assert sum(c >= 0 for c in task.reference_cut_indices) == pin["cuts"]
        assert task.flat_board_record is None and task.table_box_world is None
        assert "workcell_geometry" not in task.to_dict()


def test_flat_board_profile_is_not_a_scenario_and_wp11_is_untouched():
    from robotic_cutting_sim.task_scenarios import scenarios
    assert "blade_master_flat_board_v1" not in scenarios()
    assert scenarios()["wp11_workcell_v1"]["fixture_mode"] == "cutting_board_v1"
    doc = deepcopy(scene())
    doc["supports"] = {"kind": "solid_board", "x_range_m": [-0.09, 0.09], "half_width_y_m": 0.025,
                       "workcell": {"from_scenario": "blade_master_flat_board_v1"}}
    with pytest.raises(sc.SceneConfigError):
        sc.validate_scene(doc)


def _franka_asset_present():
    from robotic_cutting_sim.workcell import ASSETS
    return (ROOT/"artifacts/workcell_asset_cache"/ASSETS["franka"]).exists()


@pytest.mark.skipif(not _franka_asset_present(), reason="Franka USD asset cache (artifacts/workcell_asset_cache) not present")
def test_c2_preflight_model_uses_the_declared_table_and_board(cut_task):
    pytest.importorskip("mujoco")
    pytest.importorskip("newton")
    from robotic_cutting_sim.motion_preflight import MotionChecker
    checker = MotionChecker(cut_task)
    geo = checker.geo
    g = geo.by_name["static_table"]
    center, half = cut_task.table_box_world
    np.testing.assert_allclose(geo.m.geom_size[g], half, atol=1e-7)
    geo.set_q(np.asarray(cut_task.home["joint_positions"]))
    np.testing.assert_allclose(geo.d.geom_xpos[g], center, atol=1e-7)
    b = geo.by_name["static_flat_board"]
    (board,) = cut_task.fixture_boxes_local
    np.testing.assert_allclose(geo.m.geom_size[b], board.half_extents_m, atol=1e-7)
    np.testing.assert_allclose(geo.d.geom_xpos[b], world_box(cut_task, board)[0], atol=1e-7)
    assert geo.geoms[b]["role"] == "board"
