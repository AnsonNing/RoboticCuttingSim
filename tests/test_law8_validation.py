"""CPU tests of the WP-11 Phase F helpers (robotic_cutting_sim.law8_validation) and the row plans.

No material simulation runs here: the helpers are checked on synthetic inputs and
every Phase F plan/schedule is driven on an ideal integrator to show it stays within
the action limits and the episode, before any GPU row.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import law8_validation as L
from robotic_cutting_sim.blade_probe import severed_bonds as probe_severed_bonds
from robotic_cutting_sim.control import ActionLimits
from robotic_cutting_sim.tasks import PROFILE_TASK_IDS, TASK_IDS, V4_TASK_IDS, make_task

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location("run_phase_f_material", ROOT/"scripts"/"run_phase_f_material.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- tasks

def test_phase_f_variants_are_versioned_and_inherit_thresholds():
    for task_id in L.ProfileTaskSpecV4F.TASK_IDS:
        assert task_id not in TASK_IDS and task_id not in PROFILE_TASK_IDS and task_id not in V4_TASK_IDS
    for rake in (14., 16.):
        task = L.make_phase_f_task(f"TRIM-02-K2-R{int(rake)}")
        reference = make_task("TRIM-02-K2-R15")
        expected = Rotation.from_euler("y", rake, degrees=True)
        assert (Rotation.from_quat(task.edge_orientation_local_xyzw)*expected.inv()).magnitude() < 1e-12
        assert task.blade_rake_deg == rake and task.tool_profile_id == "ruler_knife_v2_y"
        # Everything but the rake is the TRIM-02-K2 family entry.
        for field in ("geometry", "fixture_boxes_local", "edge_position_knots_local_txyz", "edge_clearance_m",
                      "target_origin_local_m", "particle_spacing_m", "grid_dx_m", "duration_s"):
            assert getattr(task, field) == getattr(reference, field), field
        assert task.to_dict()["declared_orientation_metrics"]["blade_rake_deg"] == rake


def test_wide_bar_variant_is_cut02_geometry_with_the_v2_knife():
    task = L.make_phase_f_task("CUT-04-K2-W60")
    cut02, cut03 = make_task("CUT-02"), make_task("CUT-03-K2")
    assert task.geometry == cut02.geometry and task.fixture_boxes_local == cut02.fixture_boxes_local
    assert task.edge_position_knots_local_txyz == cut02.edge_position_knots_local_txyz
    assert task.edge_orientation_local_xyzw == cut03.edge_orientation_local_xyzw
    assert task.tool_profile_id == cut03.tool_profile_id == "ruler_knife_v2_y"
    assert task.blade_half_width_m == cut03.blade_half_width_m == .06
    assert task.protocol_document()["CUT-04-K2-W60"]["maximum_severed_reference_stencil_midpoint_plane_distance_m"] == .002
    assert L.make_phase_f_task("CUT-03-K2", grid_dx_m=.008).grid_dx_m == .008


def test_configured_f211_scene_overrides_only_tool_and_solver():
    from robotic_cutting_sim.scene_config import load_json
    scene, goal = L.configured_phase_f_scene_goal("F2.11_slotted_bar_three_cuts_push")
    base = load_json("bar_slotted_board_4slots.json")
    assert scene["tool"]["profile_id"] == "ruler_knife_v2_y" and scene["solver"]["module"] == "tlmpm_v5"
    assert scene["solver"]["max_cuts"] == 3 and scene["supports"] == base["supports"]
    assert scene["workpiece"] == base["workpiece"]
    planes = goal["steps"][0]["planes"]
    assert [p["origin_local_m"][0] for p in planes] == [-.048, -.016, .016]
    assert goal["steps"][1]["type"] == "move_piece"


# ----------------------------------------------------------------------------- scheduled motion

def _integrate(policy, task, start_pose, steps):
    edge, poses, phases = np.asarray(start_pose, float), [], []
    for k in range(steps):
        action = policy({"proprio": {"tool_pose": edge.copy()}, "timestamp_s": round(k*task.control_dt_s, 9)})
        delta = action["target"]
        edge = np.r_[edge[:3]+delta[:3], (Rotation.from_rotvec(delta[3:])*Rotation.from_quat(edge[3:])).as_quat()]
        poses.append(edge)
        phases.append((policy.phase, action["cut_enabled"]))
    return np.asarray(poses), phases


def test_scheduled_policy_tracks_exactly_and_labels_contiguous_phases():
    task = make_task("CUT-03-K2")
    R = Rotation.from_quat(task.reference_edge_pose_world(0.)[3:])
    start = np.array([0., 0., .05])
    segments = [L.line("a", start, start+[0., 0., -.01], R, .01, .02, cut=True),
                L.hold("b", start+[0., 0., -.01], R, .1)]
    policy = L.ScheduledPolicy(task, segments, ActionLimits())
    assert np.isclose(policy.duration_s, 1.1)
    poses, phases = _integrate(policy, task, np.r_[start, R.as_quat()], 60)
    assert np.allclose(poses[49, :3], start+[0., 0., -.01], atol=1e-12)
    assert np.allclose(poses[24, :3], start+[0., 0., -.005], atol=1e-12)
    labels = [p[0] for p in phases]
    assert labels[:50] == ["a"]*50 and labels[50:55] == ["b"]*5 and labels[55:] == ["done"]*5
    assert all(c for _, c in phases[:50]) and not any(c for _, c in phases[50:])


def test_scheduled_policy_rejects_a_schedule_beyond_the_limits():
    task = make_task("CUT-03-K2")
    R = Rotation.from_quat(task.reference_edge_pose_world(0.)[3:])
    fast = L.Segment("fast", .1, lambda s: np.r_[[0., 0., .05-.5*s], R.as_quat()], True)  # 10 mm per step
    policy = L.ScheduledPolicy(task, [fast], ActionLimits())
    with pytest.raises(ValueError, match="exceeds the action limits"):
        policy({"proprio": {"tool_pose": np.r_[0., 0., .05, R.as_quat()]}, "timestamp_s": 0.})


def test_triangle_wave_profile():
    values = [L.triangle_wave(s, 8., .5) for s in (0., .0625, .125, .25, .375, .4375, .5)]
    assert np.allclose(values, [0., 4., 8., 0., -8., -4., 0.])


def test_every_phase_f_plan_fits_the_episode_and_the_limits():
    from robotic_cutting_sim.policies import ideal_plan_end_s
    runner = _runner()
    for row, spec in runner.ROWS.items():
        if "configured" in spec:
            continue
        task = L.make_phase_f_task(spec["task"])
        if spec["kind"] in runner.SCHEDULES:
            policy = L.ScheduledPolicy(task, runner.SCHEDULES[spec["kind"]](task), ActionLimits())
            assert policy.duration_s <= task.duration_s-task.control_dt_s, row
            _integrate(policy, task, task.reference_edge_pose_world(0.),
                       int(round(policy.duration_s/task.control_dt_s))+1)
            assert policy.max_translation_m <= .002+1e-12 and policy.max_rotation_rad <= np.deg2rad(2.)+1e-12
        else:
            builder, tolerance = runner.WAYPOINT_PLANS[spec["kind"]]
            options = {} if tolerance is None else {"position_tolerance_m": tolerance}
            end = ideal_plan_end_s(task, builder(task), **options)
            assert end is not None and end <= task.duration_s-.1, (row, end)


def test_rotation_and_draw_schedules_keep_the_centre_and_plane():
    runner = _runner()
    task = make_task("CUT-03-K2")
    R0 = Rotation.from_quat(task.reference_edge_pose_world(0.)[3:])
    normal = R0.apply([0., 0., 1.])
    for kind, phase in (("pure_rotation", "rotate"), ("pure_draw", "draw")):
        policy = L.ScheduledPolicy(task, runner.SCHEDULES[kind](task), ActionLimits())
        start = policy.starts[[s.label for s in policy.segments].index(phase)]
        for s in np.linspace(0., .5 if kind == "pure_rotation" else .8, 11):
            pose, segment = policy.target(start+s+1e-12)
            R = Rotation.from_quat(pose[3:])
            assert np.allclose(R.apply([0., 0., 1.]), normal, atol=1e-12)   # rotation only about the blade normal
            if kind == "pure_rotation":
                assert np.allclose(pose[:3], [0., 0., .01], atol=1e-12)
            else:
                assert np.allclose(pose[[0, 2]], [0., .01], atol=1e-12) and abs(pose[1]) <= .006+1e-12


# ----------------------------------------------------------------------------- swept region

def test_swept_region_of_a_vertical_descent_and_distances():
    R = Rotation.from_quat([0., np.sqrt(.5), 0., np.sqrt(.5)])   # advance -z, edge line +y, normal +x
    poses = np.array([np.r_[[0., 0., z], R.as_quat()] for z in np.linspace(.03, .01, 11)])
    region = L.swept_region([poses], .06, origin=[0., 0., .02], edge_line=[0., 1., 0.], advance=[0., 0., -1.])
    extent = region.extent()
    assert np.allclose(extent["u_m"], [-.06, .06], atol=3e-4) and np.allclose(extent["w_m"], [-.01, .01], atol=3e-4)
    points = np.array([[.005, 0., .02], [0., 0., .005], [0., .07, .02], [0., 0., .0099]])
    distance = region.distance_m(points)
    assert distance[0] == 0.                               # off-plane offset is not part of the in-plane distance
    assert abs(distance[1]-.005) < 4e-4 and abs(distance[2]-.01) < 4e-4 and distance[3] < 4e-4
    report = L.label_region_report(region, points, tolerance_m=.004, what="test")
    assert report["outside_tolerance"] == 2


def test_swept_region_of_a_pure_rotation_is_a_bowtie():
    R0 = Rotation.from_quat([0., np.sqrt(.5), 0., np.sqrt(.5)])
    normal = R0.apply([0., 0., 1.])
    poses = np.array([np.r_[[0., 0., .01], (Rotation.from_rotvec(np.deg2rad(a)*normal)*R0).as_quat()]
                      for a in np.linspace(-8., 8., 9)])
    region = L.swept_region([poses], .06, origin=[0., 0., .01], edge_line=[0., 1., 0.], advance=[0., 0., -1.])
    tip_dip = .06*np.sin(np.deg2rad(8.))
    # At the tip (u = 0.059) the region spans the whole +/-8 deg fan; at the centre only the edge line itself.
    assert region.distance_m(np.array([[0., .059, .01-tip_dip*.95]]))[0] == 0.
    assert abs(region.distance_m(np.array([[0., 0., .005]]))[0]-.005) < 4e-4


def test_edge_sweep_points_cover_large_steps():
    R = Rotation.identity()
    poses = np.array([np.r_[[0., 0., 0.], R.as_quat()], np.r_[[.002, 0., 0.], R.as_quat()]])
    points = L.edge_sweep_points(poses, .01, cell_m=2.5e-4)
    xs = np.unique(np.round(points[:, 0], 9))
    assert xs.max() == pytest.approx(.002) and np.diff(xs).max() <= 1.25e-4+1e-12


# ----------------------------------------------------------------------------- bonds, bridges, mass

def test_lattice_bonds_and_severed_bonds_match_the_probe_reference():
    lattice = np.stack(np.meshgrid(np.arange(4), np.arange(3), np.arange(2), indexing="ij"), -1).reshape(-1, 3)
    a, b = L.lattice_bonds(lattice)
    assert len(a) == 3*3*2+4*2*2+4*3*1
    assert np.all(np.abs(lattice[a]-lattice[b]).sum(1) == 1)
    codes = np.where(lattice[:, 0] < 2, 1, 2)
    codes[(lattice[:, 1] == 2)] = 0                      # a code-0 strip keeps those bonds intact
    sa, sb = L.severed_bonds(lattice, codes, 1)
    assert len(sa) == probe_severed_bonds(lattice, codes, 1) == 4
    assert np.all(lattice[sa, 0]+lattice[sb, 0] == 3)


def test_bridge_report_counts_direct_code0_bridges():
    nodes = np.array([[0]*8, [0]*8, [0]*8, [1]*8])
    weights = np.full((4, 8), 1/8)
    gradients = np.zeros((4, 8, 3))
    codes = np.array([1, 2, 0, 0])
    report = L.bridge_report(nodes, weights, gradients, codes)
    assert report["direct_code_zero_bridges"] == 1 and report["nodes_with_digits_1_2"] == 1
    packed = np.array([1, 2, 0, 0])                         # K = 1 packed codes equal ternary codes
    assert L.bridge_report(nodes, weights, gradients, packed, packed=True)["direct_code_zero_bridges"] == 1


def test_mass_check_against_the_sampler():
    task = make_task("CUT-03-K2", particle_spacing_m=.004)
    _, masses, _ = task.sample_particles()
    assert L.mass_check(task, masses, len(masses))["pass"]
    assert not L.mass_check(task, masses*(1+1e-5), len(masses))["pass"]
    assert not L.mass_check(task, masses[:-1], len(masses)-1)["pass"]


# ----------------------------------------------------------------------------- per-step summaries

def _step(phase, blade=0., grip=None, contacts=0, cut=True, plow=0, ok=0, stroke=0):
    return {"phase": phase, "time_s": 0., "cut_enabled": cut, "blade_force_world_N": [blade, 0., 0.],
            "total_force_world_N": [blade+(grip or 0.), 0., 0.], "handle_force_local_N": None if grip is None else [grip, 0., 0.],
            "contact_statistics": [0., contacts, 0., 0., 0., 0., 0.], "plowing_segments_last_substep": plow,
            "cutting_segments_last_substep": ok, "edge_segments": 60, "stroke": stroke, "engaged": 0, "any_motion": 0,
            "travel_m": 0., "labelled_count": 0}


def test_contact_window_force_uses_contact_steps_only():
    steps = [_step("press")]*5+[_step("press", 2., contacts=3)]*5+[_step("done", 9., contacts=3)]
    window = L.contact_window_force(steps, ("press",))
    assert window["contact_steps"] == 5 and window["mean_N"] == pytest.approx(2.)
    assert window["all_phase_steps_mean_N"] == pytest.approx(1.)
    grip = L.contact_window_force([_step("h", grip=.2), _step("h", grip=0.)], ("h",), use="grip", contact="grip")
    assert grip["contact_steps"] == 1 and grip["mean_N"] == pytest.approx(.2)


def test_plowing_summary_fractions():
    steps = [_step("skim", ok=60, plow=60)]*2+[_step("skim", ok=60, plow=0)]*2+[_step("x", cut=False)]
    summary = L.plowing_summary(steps)
    assert summary["plowing_step_fraction_of_cutting_steps"] == .5
    assert summary["plowing_segment_fraction_of_cutting_segments"] == .5


def test_swept_normal_report_flags_the_draw_fallback():
    R = Rotation.from_quat([0., np.sqrt(.5), 0., np.sqrt(.5)])
    q = R.as_quat().tolist()
    draw = [{"phase": "draw", "time_s": .02*k, "edge_pose_world": [0., .001*k, .01]+q} for k in range(5)]
    report = L.swept_normal_report(draw, ("draw",))
    assert all(s["fallback_steps"] == 4 for s in report["segments"].values())
    descend = [{"phase": "d", "time_s": .02*k, "edge_pose_world": [0., 0., .02-.0002*k]+q} for k in range(5)]
    report = L.swept_normal_report(descend, ("d",))
    assert all(s["fallback_steps"] == 0 and s["max_normal_angle_deg"] < 1e-9 for s in report["segments"].values())
    normal = R.apply([0., 0., 1.])
    rotate = [{"phase": "r", "time_s": .02*k,
               "edge_pose_world": [0., 0., .01]+(Rotation.from_rotvec(np.deg2rad(k)*normal)*R).as_quat().tolist()}
              for k in range(5)]
    report = L.swept_normal_report(rotate, ("r",))
    assert report["segments"]["+0.0mm"]["fallback_steps"] == 4          # the centre does not move
    assert report["segments"]["+15.0mm"]["fallback_steps"] == 0 and report["segments"]["+15.0mm"]["max_normal_angle_deg"] < 1e-6


def test_digit_persistence_and_proximity():
    from robotic_cutting_sim.cut_topology import pack_digits
    old = pack_digits(np.array([[0, 0], [1, 0], [0, 0], [2, 0]]))
    new = pack_digits(np.array([[1, 0], [1, 2], [0, 1], [2, 1]]))
    out = L.digit_persistence([1, 5, 9, 12], old, new, 2, {0: 3, 1: 20})
    assert out["0"]["changes_total"] == 1 and out["0"]["changes_after_end"] == 0
    assert out["1"]["changes_total"] == 3 and out["1"]["changes_after_end"] == 0
    late = L.digit_persistence([1, 5, 9, 12], old, new, 2, {0: 0, 1: 6})
    assert late["0"]["changes_after_end"] == 1 and late["1"]["changes_after_end"] == 2
    positions = np.array([[0., 0., 0.], [.0005, 0., 0.], [.01, 0., 0.]])
    near = L.cross_component_proximity(positions, np.array([0, 1, 1]), 0, 1, .001)
    assert near["pairs_within"] == 1 and near["minimum_distance_m"] == pytest.approx(.0005)


def test_phase_f_protocol_rows_are_all_implemented():
    runner = _runner()
    protocol = json.loads(L.PHASE_F_PROTOCOL_PATH.read_text())
    assert set(runner.PROTOCOL_ROW.values()) == set(protocol["rows"])
    assert set(runner.ROW_INTERPRETATION) == {spec["kind"] for spec in runner.ROWS.values()}
