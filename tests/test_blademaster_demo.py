"""BladeMaster Phase 2A (ADR 0007): scenario v3, task v3, plan builders, demo evaluators (CPU)."""
import copy
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.demo_evaluation import (fit_swept_plane, label_distance_distribution, post_cut_manipulation,
                                                 slice_push_report, swept_plane_interface, trim_separation)
from robotic_cutting_sim.multi_cut_evaluation import MultiCutGoal
from robotic_cutting_sim.policies import Waypoint, WaypointPolicy, skim, slice_push, through_cut
from robotic_cutting_sim.scene_frames import WorkpieceTarget
from robotic_cutting_sim.task_scenarios import (PATH, PATH_V2, PATH_V3, apply_scenario, board_particle_clearance,
                                                fixture_materials, initial_joints, scenarios, table_material)
from robotic_cutting_sim.tasks import (PROTOCOL_PATH, PROTOCOL_V2_PATH, PROTOCOL_V3_PATH, TASK_IDS, DemoTaskSpec,
                                       FixtureBox, MaterialFixtureBox, ProfileTaskSpec, _interface_location, make_task)

SCENARIO = "blademaster_workcell_v1"
V3_TASKS = ("CUT-01-R", "CUT-02", "TRIM-01-R", "SLICE-PUSH-01")


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=lambda x: np.asarray(x).tolist()).encode()).hexdigest()


# Pinned before Phase 2A (HEAD 59e4efb tree): v1/v2 protocols, receipts, scenarios and plans must not change.
PROTOCOL_SHA = {PROTOCOL_PATH: "feb3a8ddb99504ccabee122615b841c844fd8a0527fc1eec713cd70c9d7013a1",
                PROTOCOL_V2_PATH: "47af3668c5328946919cefc6c51fc21d96fdea87dbae6ced2f9b906006640fb7",
                PATH: "7bcef84042fb310eb48e64080240cba7c7006cd7dbd71c123412fccf0ba17732",
                PATH_V2: "32144d87334e7e0dfa7f5b2f0084f88b019507d461a65c619fbef0e58b4f1c88"}
RECEIPT_SHA = {"CUT-01": "e797c96a5fdec027bf04d5e38874645d06516d8142d87a3c67af08395e9bd87a",
               "TRIM-01": "4fde02e1f0ffb7d4b6926e372ead0c4a730862f3c0d4e9b400cc6ee1a9e6df6d",
               "CUT-01-R": "cac75b5823defbd1f055daf7cb26c085de8dbd5c51b115f69ce70f713567b921",
               "TRIM-01-R": "e7c382436fd673e39f1461c31d0f93a5e2ff13fa7f83366e1515b22483a6499a",
               "CUT-02": "6de41f6eb616578eab43ea928e77aad40e056b434e4f4472f52da1fe0275d565"}
SCENARIO_SHA = {("raised_workpiece_50mm_v1", "CUT-01-R"): "39a74b240d84e1ab21a6eb9224d78721c0c39a34d3d93505aa557dbc048c0334",
                ("raised_workpiece_50mm_v1", "TRIM-01-R"): "501b9b3a1ebc47f7545318594ca57dc8d3ffa18fb2bd2ebe4d16a3be4f62e313",
                ("raised_workpiece_50mm_v1", "CUT-02"): "8d9528421d4c5a2bc17ba1defa59ee42047aec77b8d05d5d7070b90d2abdd03c",
                ("ruler_workcell_v2", "CUT-01-R"): "39a74b240d84e1ab21a6eb9224d78721c0c39a34d3d93505aa557dbc048c0334",
                ("ruler_workcell_v2", "TRIM-01-R"): "a32dc69dd4b1df7a3e21b995ea50589f1fac05c6b65f6a7dcff1fb387fbb8194",
                ("ruler_workcell_v2", "CUT-02"): "8d9528421d4c5a2bc17ba1defa59ee42047aec77b8d05d5d7070b90d2abdd03c"}
WAYPOINT_FIELDS = ("pose", "cut", "saw_amplitude_m", "saw_period_s", "speed_m_s", "cut_index")
PLAN_SHA = {("CUT-01-R", False): "4eb88ce14bc88aedb49bf412cc28de1c6e5b3d5225a32d51050091b0a8744b40",
            ("CUT-01-R", True): "9da3d7de3dc7a9a9f0a6c789ba0c9eafed96f7620d66662152215f0bd6e8986f",
            ("CUT-02", False): "a239658f8f566668e0d52fd5f8ffcbf3301e2532788b43b33ee21c026c7afed2",
            ("CUT-02", True): "3ea87dbf442f541cb1469fcfb6dc11242abc69bbcf365dd9db09dfd4c0408cf8",
            ("CUT-01", False): "e014899c6ba145d63b5b4c12f56986921345da4ddea418a13e23b8c6e1b9cb62",
            ("CUT-01", True): "dcaf4436359108b61aa3d772da0d36bfc1463654b1646ce22a7a7d313d2da25e"}
SKIM_SHA = "46b8d1e97773d66ea4195e349d9cafbb4958004785187e297dfa4be5aea459e0"
ROLLOUT_SHA = ("be1ed8f9da6b3c73d417000514ea797d8c77a09f3718ad2876552ae156c1e0af",
               "6a4c5e2a900213a16fcd03d895e4e6b1076d79e965a15d22f06294e70aba7c03")


def _plan_sha(plan):
    return hashlib.sha256(json.dumps([[np.asarray(getattr(w, f)).tolist() for f in WAYPOINT_FIELDS]
                                      for w in plan]).encode()).hexdigest()


def _rollout(policy, task, steps=600, start_offset=(.01, 0., .02)):
    """Ideal integrator: the observed edge moves by each commanded delta."""
    edge = np.asarray(task.reference_edge_pose_world(0.), float)
    edge = np.r_[edge[:3]+start_offset, edge[3:]]
    actions, phases = [], []
    for k in range(steps):
        action = policy({"proprio": {"tool_pose": edge.copy()}, "timestamp_s": k*task.control_dt_s})
        actions.append(np.r_[action["target"], float(action["cut_enabled"]), action.get("cut_index", 0)])
        phases.append((getattr(policy, "phase", ""), edge.copy()))
        edge = np.r_[edge[:3]+action["target"][:3],
                     (Rotation.from_rotvec(action["target"][3:])*Rotation.from_quat(edge[3:])).as_quat()]
    return np.asarray(actions), phases


# --- v1/v2 behaviour unchanged ------------------------------------------------------------------------------

def test_v1_v2_protocols_receipts_scenarios_and_plans_are_unchanged():
    for path, digest in PROTOCOL_SHA.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path.name
    assert TASK_IDS[:5] == ("CUT-01", "TRIM-01", "CUT-01-R", "TRIM-01-R", "CUT-02")
    for task_id, digest in RECEIPT_SHA.items():
        assert _sha(make_task(task_id).to_dict()) == digest, task_id
        assert not isinstance(make_task(task_id), DemoTaskSpec)
    for (scenario, task_id), digest in SCENARIO_SHA.items():
        task = apply_scenario(make_task(task_id), scenario)
        assert _sha(task.to_dict()) == digest, (scenario, task_id)
        assert type(task.fixture_boxes_local[0]) is FixtureBox
    for (task_id, tilted), digest in PLAN_SHA.items():
        kwargs = dict(tilt_deg=20, cut_index=1) if tilted else {}
        assert _plan_sha(through_cut(make_task(task_id), **kwargs)) == digest, (task_id, tilted)
    assert _plan_sha(skim(make_task("TRIM-01-R"))) == SKIM_SHA


def test_waypoint_policy_default_is_bit_identical_and_estimator_is_used():
    task = make_task("CUT-01-R")
    for kwargs, digest in zip(({}, {"saw_amplitude_m": .003, "tilt_deg": 10.}), ROLLOUT_SHA):
        actions, _ = _rollout(WaypointPolicy(task, through_cut(task, **kwargs)), task)
        assert hashlib.sha256(actions.tobytes()).hexdigest() == digest
    # An explicit estimator replaces the nominal proprio edge (e.g. a measured edge 1 mm lower).
    plan = through_cut(task)
    observation = {"proprio": {"tool_pose": np.asarray(plan[0].pose, float)}, "timestamp_s": 0.}
    nominal = WaypointPolicy(task, plan)(observation)
    lowered = np.asarray(plan[0].pose, float)-[0., 0., .001, 0., 0., 0., 0.]
    corrected = WaypointPolicy(task, plan, edge_estimator=lambda obs: lowered)(observation)
    assert nominal["target"][2] == pytest.approx(-.002)              # nominal: at waypoint 0 -> descends to 1
    assert corrected["target"][2] == pytest.approx(.001)           # measured: correct the 1 mm error first


def test_hold_waypoint_keeps_pose_for_its_duration():
    task = make_task("CUT-01-R")
    pose = tuple(task.reference_edge_pose_world(0.))
    later = tuple(np.r_[np.asarray(pose[:3])+[0., 0., .01], pose[3:]])
    policy = WaypointPolicy(task, [Waypoint(pose, hold_s=.5, label="hold"), Waypoint(later, label="up")])
    actions, phases = _rollout(policy, task, steps=40, start_offset=(0., 0., 0.))
    held = [k for k, (phase, _) in enumerate(phases) if phase == "hold"]
    assert len(held) == round(.5/task.control_dt_s) and np.allclose(actions[held, :6], 0.)
    assert phases[-1][0] == "done"


# --- SLICE-PUSH-01 task (cutting_tasks_v3) --------------------------------------------------------------------

def test_slice_push_task_loads_as_versioned_v3_task():
    task = make_task("SLICE-PUSH-01")
    assert isinstance(task, DemoTaskSpec) and isinstance(task, ProfileTaskSpec) and "SLICE-PUSH-01" in TASK_IDS
    assert task.tool_profile.profile_id == "ruler_knife_v1" and task.base_task_id == "CUT-01"
    assert (task.duration_s, task.control_dt_s, task.max_cuts) == (16., .02, 2)
    assert round(task.duration_s/task.control_dt_s) == 800
    assert task.goal_kind == "sever" and task.goal_profile == "multi_cut_push_v1"
    cut01r = make_task("CUT-01-R")
    assert task.geometry == cut01r.geometry and task.material == cut01r.material
    assert task.edge_orientation_local_xyzw == cut01r.edge_orientation_local_xyzw
    assert task.hand_from_edge_xyz_xyzw == cut01r.hand_from_edge_xyz_xyzw
    assert (task.particle_spacing_m, task.grid_dx_m) == (.001, .004)
    points, masses, lattice = task.sample_particles()
    expected, _, _ = cut01r.sample_particles()
    assert np.array_equal(points, expected)
    fractions = [masses[points[:, 0] < -.03].sum(), masses[np.abs(points[:, 0]) < .03].sum(), masses[points[:, 0] > .03].sum()]
    assert np.allclose(np.asarray(fractions)/masses.sum(), [.3125, .375, .3125])
    # Board segments: top at local z = 0, 10 mm thick, 8 mm relief slots centred on both planes.
    boxes = sorted(task.fixture_boxes_local, key=lambda b: b.center_m[0])
    spans = [(b.center_m[0]-b.half_extents_m[0], b.center_m[0]+b.half_extents_m[0]) for b in boxes]
    assert np.allclose(spans, [(-.09, -.034), (-.026, .026), (.034, .09)])
    assert all(np.isclose(b.center_m[2]+b.half_extents_m[2], 0.) and np.isclose(2*b.half_extents_m[2], .01)
               and np.isclose(b.half_extents_m[1], .021) for b in boxes)
    for (_, right), (left, _), (origin, _) in zip(spans[:-1], spans[1:], task.multi_cut_planes_local):
        assert np.isclose(left-right, .008) and np.isclose((left+right)/2, origin[0])
    assert task.multi_cut_planes_local == (((-.03, 0., 0.), (1., 0., 0.)), ((.03, 0., 0.), (1., 0., 0.)))
    # Reference schedule: cut 0 then cut 1, then the cut-disabled push; knots start above cut 0.
    assert [task.active_cut_index(t) for t in (0., 2., 5., 9., 12.5, 15.)] == [-1, 0, -1, 1, -1, -1]
    assert task.reference_action(9.)["cut_index"] == 1 and task.reference_action(13.)["cut_enabled"] is False
    start = task.reference_edge_pose_local(0.)
    assert np.isclose(start[0], -.03) and start[2] > task.geometry.base_height_m
    assert np.isclose(task.reference_edge_pose_local(task.cut_intervals_s[0][1])[2], -.004)
    assert np.isclose(task.reference_edge_pose_local(task.cut_intervals_s[1][1])[0], .03)
    push = [task.reference_edge_pose_local(t) for t in task.push_interval_s]
    assert np.allclose(push[1][:3]-push[0][:3], [.02, 0., 0.]) and np.isclose(push[0][2], .003)
    receipt = task.to_dict()
    json.dumps(receipt, default=lambda x: np.asarray(x).tolist())
    assert receipt["protocol_sha256"] == hashlib.sha256(PROTOCOL_V3_PATH.read_bytes()).hexdigest()
    assert receipt["protocol_id"] == "cutting_tasks_v3_20261029" and receipt["frozen_two_part_oracle"] == "NOT_APPLICABLE"
    assert task.evaluate_oracle({})["status"] == "NOT_APPLICABLE"
    protocol = json.loads(PROTOCOL_V3_PATH.read_text())
    assert "before any SLICE-PUSH-01 episode" in protocol["freeze_boundary"]
    entry = task.protocol_document()["SLICE-PUSH-01"]
    assert not any(k.startswith(("minimum_", "maximum_", "required_")) for k in entry)  # v1 two-part limits dropped
    assert task.thresholds["multi_cut"] == {"minimum_part_mass_fraction": .25, "minimum_main_parts_mass_fraction": .99,
                                            "plane_error_bound_m": .002}
    goal = task.multi_cut_goal()
    assert isinstance(goal, MultiCutGoal) and goal.plane_error_bound_m == .002
    with pytest.raises(NotImplementedError):
        make_task("SLICE-PUSH-01", variant="randomized")
    with pytest.raises(NotImplementedError):
        make_task("SLICE-PUSH-01", workpiece_kind="capsule")


# --- Scenario blademaster_workcell_v1 -------------------------------------------------------------------------

def test_scenario_v3_inherits_ruler_workcell_v2_placement_rake_and_seed():
    v2, v3 = scenarios()["ruler_workcell_v2"], scenarios()[SCENARIO]
    assert v3["world_translation_m"] == v2["world_translation_m"] and v3["trim_rake_deg"] == v2["trim_rake_deg"]
    assert initial_joints(SCENARIO, "TRIM-01-R") == initial_joints("ruler_workcell_v2", "TRIM-01-R")
    assert sorted(v3["allowed_tasks"]) == sorted(V3_TASKS)
    for task_id in ("CUT-01-R", "TRIM-01-R", "CUT-02"):
        old, new = apply_scenario(make_task(task_id), "ruler_workcell_v2"), apply_scenario(make_task(task_id), SCENARIO)
        assert new.world_from_workpiece_xyzw == old.world_from_workpiece_xyzw
        assert new.edge_orientation_local_xyzw == old.edge_orientation_local_xyzw
        assert new.geometry == old.geometry and new.material == old.material
    with pytest.raises(ValueError):
        apply_scenario(make_task("CUT-01"), SCENARIO)
    with pytest.raises(ValueError):
        apply_scenario(make_task("SLICE-PUSH-01"), "ruler_workcell_v2")


@pytest.mark.parametrize("task_id", V3_TASKS)
def test_scenario_v3_board_and_riser_boxes_with_materials(task_id):
    original = make_task(task_id)
    task = apply_scenario(original, SCENARIO)
    table = task.table_top_world_z_m-task.world_from_workpiece_xyzw[2]
    boxes, materials = task.fixture_boxes_local, fixture_materials(task)
    assert len(boxes) == 2*len(original.fixture_boxes_local) == len(materials)
    assert all(isinstance(b, MaterialFixtureBox) and isinstance(b, FixtureBox) for b in boxes)
    for support, board, riser in zip(original.fixture_boxes_local, boxes[0::2], boxes[1::2]):
        lo = lambda b, axis: b.center_m[axis]-b.half_extents_m[axis]
        hi = lambda b, axis: b.center_m[axis]+b.half_extents_m[axis]
        assert (board.role, riser.role) == ("board", "riser")
        assert board.support_name == riser.support_name == support.name
        assert np.isclose(hi(board, 2), hi(support, 2)) and np.isclose(hi(board, 2)-lo(board, 2), .01)
        assert np.isclose(board.half_extents_m[1], .05) and np.isclose(board.half_extents_m[0], support.half_extents_m[0])
        assert np.isclose(hi(riser, 2), lo(board, 2)) and np.isclose(lo(riser, 2), table)
        assert np.allclose(riser.half_extents_m[:2], support.half_extents_m[:2])
        assert np.allclose([board.center_m[0], riser.center_m[0]], support.center_m[0])
        # Board + riser contain the ruler_workcell_v2 extended support box exactly within the support y extent.
        assert np.isclose(lo(riser, 2), table) and hi(board, 2) == pytest.approx(hi(support, 2))
        for material, box in ((board, "board"), (riser, "riser")):
            assert material.static_friction == material.dynamic_friction == task.material.friction == .3
            assert material.restitution == 0.
    assert {tuple(m["display_color_rgb"]) for m in materials if m["role"] == "board"} == {(.72, .53, .34)}
    assert {tuple(m["display_color_rgb"]) for m in materials if m["role"] == "riser"} == {(.55, .55, .57)}
    assert all(m["material_declared"] for m in materials)
    assert table_material(task, SCENARIO) == {"static_friction": .3, "dynamic_friction": .3, "restitution": 0.}
    json.dumps(task.to_dict(), default=lambda x: np.asarray(x).tolist())
    # The widened board never reaches material: every actual 1 mm particle stays inside |y| < 50 - 5 mm.
    points, _, _ = task.sample_particles()
    margin = scenarios()[SCENARIO]["cutting_board"]["particle_y_margin_m"]
    reach = float(np.abs(points[:, 1]).max())+task.particle_spacing_m/2
    assert reach < .05-margin, reach
    assert board_particle_clearance(task, scenarios()[SCENARIO])["clear"]
    if task_id == "CUT-01-R":
        print(f"{task_id}: max |y| particle extent {reach*1e3:.1f} mm < board half-width 50 mm - margin {margin*1e3:.0f} mm")


def test_v2_scenario_boxes_report_undeclared_materials():
    task = apply_scenario(make_task("CUT-02"), "ruler_workcell_v2")
    assert [m["material_declared"] for m in fixture_materials(task)] == [False, False]
    assert table_material(task, "ruler_workcell_v2") is None and table_material(task, None) is None
    with pytest.raises(ValueError):
        MaterialFixtureBox("bad", (0., 0., 0.), (1., 1., 1.), role="lid")


# --- Plan builders ----------------------------------------------------------------------------------------------

def test_through_cut_offset_shifts_along_the_local_target_normal():
    task = make_task("CUT-01-R")
    base, shifted = through_cut(task), through_cut(task, offset_m=.03)
    assert _plan_sha(through_cut(task, offset_m=0.)) == _plan_sha(base)
    for a, b in zip(base, shifted):
        assert np.allclose(np.subtract(b.pose[:3], a.pose[:3]), [.03, 0., 0.]) and np.allclose(a.pose[3:], b.pose[3:])
    yawed = make_task("CUT-01-R", world_from_workpiece_xyzw=(.1, .2, .0, *Rotation.from_euler("z", 90, degrees=True).as_quat()))
    delta = np.subtract(through_cut(yawed, offset_m=-.02)[2].pose[:3], through_cut(yawed)[2].pose[:3])
    assert np.allclose(delta, [0., -.02, 0.], atol=1e-12)


def test_slice_push_plan_geometry_flags_and_timing():
    task = apply_scenario(make_task("SLICE-PUSH-01"), SCENARIO)
    plan = slice_push(task)
    local = np.asarray([task.points_to_local(np.asarray(w.pose[:3])) for w in plan])
    labels = [w.label for w in plan]
    cuts = [(i, w) for i, w in enumerate(plan) if w.cut]
    assert [w.cut_index for _, w in cuts] == [0, 1] and [labels[i] for i, _ in cuts] == ["cut_0", "cut_1"]
    assert all(w.speed_m_s == .01 for _, w in cuts)
    for (index, _), plane in zip(cuts, (-.03, .03)):
        assert np.allclose(local[index-1:index+1, 0], plane)          # descent stays on the declared plane
        assert np.isclose(local[index, 2], -.004)                       # 4 mm below the bar bottom
        assert local[index-1, 2] > task.geometry.base_height_m          # armed only from above the bar
    assert np.allclose(local[labels.index("move_1")], [.03, 0., .028])    # moves between cuts above the bar
    kerf, pushed = local[labels.index("raise_in_kerf")], local[labels.index("push")]
    assert np.allclose(kerf, [.03, 0., .003]) and np.allclose(pushed-kerf, [.02, 0., 0.])
    assert plan[labels.index("push")].speed_m_s == .01 and plan[labels.index("push_hold")].hold_s == .5
    assert plan[labels.index("hold_0")].hold_s == plan[labels.index("hold_1")].hold_s == .5
    assert all(not w.cut for w in plan[labels.index("hold_1"):])
    assert local[-1, 2] > task.geometry.base_height_m+.02
    orientation = {tuple(np.round(w.pose[3:], 12)) for w in plan}
    assert len(orientation) == 1
    actions, phases = _rollout(WaypointPolicy(task, plan), task, steps=800, start_offset=(0., 0., 0.))
    done = [k for k, (phase, _) in enumerate(phases) if phase == "done"]
    assert done and done[0]*task.control_dt_s < task.duration_s-1.   # ideal tracking fits the 16 s episode
    for (phase, edge), action in zip(phases, actions):
        if action[6]:                                                    # cut armed only on a declared plane
            x = task.points_to_local(edge[:3])[0]
            assert phase in ("cut_0", "cut_1") and min(abs(x+.03), abs(x-.03)) < 1e-4
            assert action[7] == (1 if phase == "cut_1" else 0)
    with pytest.raises(ValueError):
        slice_push(make_task("CUT-01-R"))


# --- Evaluators ---------------------------------------------------------------------------------------------------

def _chain(x, dx=.004, origin=-.1):
    """Stencils linking particles through 1D linear nodes along x (8 slots, 2 active)."""
    cell = np.floor((np.asarray(x)-origin)/dx).astype(int)
    nodes = np.zeros((len(x), 8), np.int32)
    nodes[:, 0], nodes[:, 1] = cell, cell+1
    weights = np.zeros((len(x), 8))
    weights[:, 0], weights[:, 1] = .5, .5
    return nodes, weights, np.zeros((len(x), 8, 3))


def test_swept_plane_interface_matches_frozen_statistic_and_follows_the_plane():
    task = make_task("CUT-01-R")
    x = np.arange(-.0195, .02, .001)
    reference = np.stack([x, np.zeros_like(x), np.full_like(x, .01)], 1)
    nodes, weights, gradients = _chain(x)
    labels = (x > 0).astype(int)
    frozen = _interface_location(nodes, weights, gradients, labels, reference, task)
    ours = swept_plane_interface(nodes, weights, gradients, labels, reference, task.target_origin_local_m,
                                 task.target_normal_local)
    for key in frozen:
        assert ours[key] == frozen[key]
    shifted = swept_plane_interface(nodes, weights, gradients, labels, reference, (.003, 0., 0.), (2., 0., 0.))
    assert shifted["maximum_reference_midpoint_plane_distance_m"] > frozen["maximum_reference_midpoint_plane_distance_m"]
    none = swept_plane_interface(nodes, weights, gradients, np.zeros_like(labels), reference, (0, 0, 0), (1, 0, 0))
    assert none["maximum_reference_midpoint_plane_distance_m"] is None


def test_fit_swept_plane_recovers_vertical_and_oblique_cuts():
    z = np.linspace(.03, -.004, 30)
    vertical = fit_swept_plane(np.stack([np.full_like(z, .01), np.zeros_like(z), z], 1), np.tile([0., 1., 0.], (30, 1)),
                               reference_normal=(1, 0, 0))
    assert np.allclose(vertical["normal"], [1., 0., 0.]) and vertical["max_residual_m"] < 1e-12
    assert np.isclose(np.asarray(vertical["origin"])@vertical["normal"], .01)
    tilt = np.deg2rad(20.)
    advance = np.array([np.sin(tilt), 0., -np.cos(tilt)])
    rng = np.random.default_rng(42)
    points = advance[None]*np.linspace(0, .035, 40)[:, None]+rng.normal(0, 5e-5, (40, 3))
    oblique = fit_swept_plane(points, np.tile([0., 1., 0.], (40, 1)), reference_normal=(1, 0, 0))
    assert np.allclose(oblique["normal"], [np.cos(tilt), 0., np.sin(tilt)], atol=5e-3)
    assert oblique["rms_residual_m"] < 1e-4
    with pytest.raises(ValueError):
        fit_swept_plane(np.zeros((5, 3)), np.tile([0., 1., 0.], (5, 1)))


def test_label_distance_distribution_counts_only_labelled_particles():
    x = np.arange(-.0195, .02, .001)
    reference = np.stack([x, np.zeros_like(x), np.zeros_like(x)], 1)
    codes = np.zeros((2, len(x)), np.int32)
    codes[1, np.abs(x) < .002] = np.where(x[np.abs(x) < .002] < 0, 1, 2)
    codes[1, np.abs(x-.015) < .001] += 3                                  # digit 1 elsewhere
    result = label_distance_distribution(reference, codes, (0, 0, 0), (1, 0, 0), digit=0)
    assert result["labelled_particles"] == 4 and result["maximum_m"] == pytest.approx(.0015)
    assert result["side_agreement_fraction"] == 1. and result["side_convention_digit1_negative"]
    assert label_distance_distribution(reference, codes, (0, 0, 0), (1, 0, 0))["labelled_particles"] == 6
    empty = label_distance_distribution(reference, np.zeros(len(x), np.int32), (0, 0, 0), (1, 0, 0))
    assert empty["labelled_particles"] == 0 and empty["maximum_m"] is None


def _column():
    z = np.arange(.0005, .028, .001)
    reference = np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
    nodes, weights, gradients = _chain(z, origin=-.002)
    return z, reference, nodes, weights, gradients


def test_trim_separation_positive_and_negative_cases():
    z, reference, nodes, weights, gradients = _column()
    target = WorkpieceTarget((0, 0, 0, 0, 0, 0, 1), .02)
    masses = np.ones(len(z))
    clean = np.where(z > .02, 2, 1).astype(np.int32)
    before = [a.copy() for a in (reference, nodes, weights, gradients, clean)]
    result = trim_separation(reference, reference, masses, nodes, weights, gradients, clean, target)
    assert result["status"] == "PASS_TRIM_SEPARATION", result
    assert result["metrics"]["removed_reference_above_plane_mass_fraction"] == 1.
    assert all(np.array_equal(a, b) for a, b in zip(before, (reference, nodes, weights, gradients, clean)))
    uncut = trim_separation(reference, reference, masses, nodes, weights, gradients, np.zeros(len(z), np.int32), target)
    assert uncut["status"] == "FAIL_TRIM_SEPARATION" and not uncut["checks"]["reference_above_plane_material_removed"]
    gouged = np.where(z > .02, 1, np.where(z > .012, 2, 1)).astype(np.int32)   # a below-plane slab detached too
    result = trim_separation(reference, reference, masses, nodes, weights, gradients, gouged, target)
    assert result["checks"]["reference_above_plane_material_removed"]
    assert not result["checks"]["detached_below_plane_mass_bounded"] and result["status"] == "FAIL_TRIM_SEPARATION"
    assert result["metrics"]["largest_component_below_plane_mass_share"] < .99


def _three_piece_recording(push_m=.015, neighbour_m=0., code_change=False, merge_after=False, one_cut=False):
    x = np.arange(-.0445, .045, .001)
    reference = np.stack([x, np.zeros_like(x), np.zeros_like(x)], 1)
    nodes, weights, gradients = _chain(x)
    digit0, digit1 = np.where(x < -.015, 1, 2), np.where(x < .015, 1, 2)
    final = (digit0 if one_cut else digit0+3*digit1).astype(np.int32)
    time = np.round(np.arange(0., 2.01, .1), 10)
    ramp = np.clip((time-.5)/1., 0., 1.)
    positions = np.repeat(reference[None], len(time), 0)
    positions[:, x > .015, 0] += (push_m*ramp)[:, None]
    positions[:, np.abs(x) < .015, 0] += (neighbour_m*ramp)[:, None]
    codes = np.repeat(final[None], len(time), 0)
    if code_change:
        codes[10, 40] = 0
    if merge_after:
        codes[-2:] = 0
    return dict(time_s=time, positions=positions, codes=codes, mass_kg=np.ones(len(x)), reference_stencil_nodes=nodes,
                reference_stencil_weights=weights, reference_stencil_gradients=gradients), x > .015


def _manipulate(data, selector, **kwargs):
    return post_cut_manipulation(data["time_s"], data["positions"], data["codes"], data["mass_kg"],
                                 data["reference_stencil_nodes"], data["reference_stencil_weights"],
                                 data["reference_stencil_gradients"], .5, 1.5, (1., 0., 0.), selector, **kwargs)


def test_post_cut_manipulation_pass_and_read_only():
    data, selector = _three_piece_recording()
    before = copy.deepcopy(data)
    result = _manipulate(data, selector)
    assert result["status"] == "PASS_POST_CUT_MANIPULATION", result
    assert result["metrics"]["components_at_push_start"] == 3
    assert result["metrics"]["pushed_displacement_along_push_m"] == pytest.approx(.015)
    for key in data:
        assert np.array_equal(data[key], before[key])


@pytest.mark.parametrize("case,failed", [
    (dict(neighbour_m=.005), "other_pieces_stationary"),
    (dict(push_m=.005), "pushed_piece_displaced"),
    (dict(code_change=True), "no_label_changes_during_push"),
    (dict(merge_after=True), "components_never_merge"),
    (dict(one_cut=True), "components_after_cuts")])
def test_post_cut_manipulation_negative_cases(case, failed):
    data, selector = _three_piece_recording(**case)
    result = _manipulate(data, selector)
    assert result["status"] == "FAIL_POST_CUT_MANIPULATION" and result["checks"][failed] is False, result["checks"]


def test_post_cut_manipulation_needs_frames_around_the_push():
    data, selector = _three_piece_recording()
    result = post_cut_manipulation(data["time_s"], data["positions"], data["codes"], data["mass_kg"],
                                   data["reference_stencil_nodes"], data["reference_stencil_weights"],
                                   data["reference_stencil_gradients"], .5, 5., (1., 0., 0.), selector)
    assert result["status"] == "NOT_RUN"


def _slice_push_recording(task, plane_shift_m=0., push_m=.015):
    points, masses, lattice = task.sample_particles()
    world = task.points_to_world(points)
    nodes, weights, gradients = _chain(points[:, 0], dx=.008, origin=-.1)  # 5 mm particles
    d0 = np.where(points[:, 0] < -.03+plane_shift_m, 1, 2)
    d1 = np.where(points[:, 0] < .03, 1, 2)
    time = np.round(np.arange(0., 16.01, .5), 10)
    codes = np.zeros((len(time), len(points)), np.int32)
    codes[time >= task.cut_intervals_s[0][1]] += d0.astype(np.int32)
    codes[time >= task.cut_intervals_s[1][1]] += (3*d1).astype(np.int32)
    start, stop = task.push_interval_s
    ramp = np.clip((time-start)/(stop-start), 0., 1.)
    positions = np.repeat(world[None], len(time), 0)
    positions[:, points[:, 0] > .03, :] += (push_m*ramp)[:, None, None]*task.push_direction_world()
    return dict(time_s=time, positions=positions, codes=codes, mass_kg=masses, reference_positions=world,
                reference_indices=lattice, reference_stencil_nodes=nodes, reference_stencil_weights=weights,
                reference_stencil_gradients=gradients)


def test_slice_push_report_pass_and_failures():
    task = apply_scenario(make_task("SLICE-PUSH-01", particle_spacing_m=.005), SCENARIO)
    data = _slice_push_recording(task)
    before = copy.deepcopy(data)
    report = slice_push_report(task, data)
    assert report["status"] == "PASS_SLICE_PUSH", report
    assert report["multi_cut"]["status"] == "PASS_MULTI_CUT_OUTCOME"
    assert report["post_cut_manipulation"]["metrics"]["components_at_push_start"] == 3
    assert report["push_interval_source"] == "task_reference_schedule" and report["frozen_two_part_oracle"] == "NOT_APPLICABLE"
    # Synthetic labels cover whole sides (not a swept band): every particle is labelled on the correct side.
    assert all(d["labelled_particles"] == len(data["mass_kg"]) and d["side_agreement_fraction"] == 1.
               for d in report["label_distance_to_declared_planes"])
    for key in data:
        assert np.array_equal(data[key], before[key])
    assert task.evaluate_demo(data)["status"] == "PASS_SLICE_PUSH"
    misplaced = slice_push_report(task, _slice_push_recording(task, plane_shift_m=.01))
    assert misplaced["status"] == "FAIL_SLICE_PUSH" and not misplaced["multi_cut"]["checks"]["declared_cut_planes"]
    unpushed = slice_push_report(task, _slice_push_recording(task, push_m=0.))
    assert unpushed["status"] == "FAIL_SLICE_PUSH"
    assert not unpushed["post_cut_manipulation"]["checks"]["pushed_piece_displaced"]
    recorded = slice_push_report(task, {**data, "push_interval_s": (12., 15.)})
    assert recorded["push_interval_source"] == "recorded_push_phase" and recorded["status"] == "PASS_SLICE_PUSH"
    partial = {k: v for k, v in data.items() if k != "reference_indices"}
    assert slice_push_report(task, partial)["status"] == "NOT_RUN"
