"""CPU tests of the WP-11 Phase F v2 row F2.9_blunt_handle_v2 (scripts/run_phase_f_v2_material.py).

No material simulation runs here: the declared protocol, the plan (ideal integrator from the
task start pose), the blade/grip clearance, the force estimate arithmetic and the workpiece
motion metrics are checked before any GPU row. The phaseF_v1 protocol and runner must stay
byte-identical to the files the v1 records were produced with.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import law8_validation as L
from robotic_cutting_sim.control import ActionLimits
from robotic_cutting_sim.policies import WaypointPolicy, ideal_plan_end_s

ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL_SHA256 = '0efcf88552c2ee8a5ccfa202c98821c786e745b6452ac0cc38bf93a250e8c68b'
V1_RUNNER_SHA256 = '5aea75d789bc0c46a954b0763fc9a5ed996da0f9b9c85749c7ab37e80e90b989'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v2():
    return _load('run_phase_f_v2_material', ROOT/'scripts'/'run_phase_f_v2_material.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _trajectory(runner, task):
    plan = runner.blunt_handle_v2_plan(task)
    policy = WaypointPolicy(task, plan, ActionLimits(), position_tolerance_m=runner.PARAMS['position_tolerance_m'])
    edge, poses, phases, armed = np.asarray(task.reference_edge_pose_world(0.), float), [], [], []
    for k in range(int(round(task.duration_s/task.control_dt_s))):
        action = policy({"proprio": {"tool_pose": edge.copy()}, "timestamp_s": round(k*task.control_dt_s, 9)})
        delta = action["target"]
        edge = np.r_[edge[:3]+delta[:3], (Rotation.from_rotvec(delta[3:])*Rotation.from_quat(edge[3:])).as_quat()]
        poses.append(edge)
        phases.append(policy.phase)
        armed.append(bool(action["cut_enabled"]))
        if policy.done:
            break
    return plan, policy, np.asarray(poses), phases, armed


def test_v1_protocol_and_runner_are_unchanged():
    assert _sha(L.PHASE_F_PROTOCOL_PATH) == V1_PROTOCOL_SHA256
    assert _sha(ROOT/'scripts'/'run_phase_f_material.py') == V1_RUNNER_SHA256
    v1 = json.loads(L.PHASE_F_PROTOCOL_PATH.read_text())
    assert v1["protocol_id"] == "wp11_phaseF_v1" and "F2.9_blunt_handle" in v1["rows"]


def test_v2_protocol_declares_one_row_with_the_v1_criteria():
    runner = _v2()
    protocol = json.loads((ROOT/'configs'/'wp11'/'phaseF_protocol_v2.json').read_text())
    assert protocol["protocol_id"] == "wp11_phaseF_v2" and list(protocol["rows"]) == ["F2.9_blunt_handle_v2"]
    row = protocol["rows"]["F2.9_blunt_handle_v2"]
    for phrase in ("zero new labels", "one component", "> 0.1 N", "mass conserved"):
        assert phrase in row["pass"]
    assert runner.HANDLE_FORCE_N == runner.F.HANDLE_FORCE_N == .1
    assert runner.PRESS_PHASES == ('handle_press', 'handle_hold')
    # The registration is local to the v2 runner's private module instance.
    fresh = _load('run_phase_f_material_fresh', ROOT/'scripts'/'run_phase_f_material.py')
    assert runner.KIND not in fresh.WAYPOINT_PLANS and set(fresh.ROWS) == set(runner.F.ROWS)


def test_force_estimate_matches_the_material_and_the_grip_face():
    runner = _v2()
    task = L.make_phase_f_task('CUT-03-K2')
    material = task.to_dict()["material"]
    estimate = runner.PROTOCOL_V2["rows"][runner.ROW]["force_estimate"]
    E, sigma_y = material["young_modulus_pa"], material["yield_stress_pa"]
    assert (E, sigma_y) == (50e3, 1e3)
    footprint = runner.footprint_report(task, runner.blunt_handle_v2_plan(task)[4].pose)
    A = footprint["footprint_area_m2"]
    assert np.isclose(A, estimate["contact_area_m2"], rtol=1e-9)
    H, d = task.geometry.base_height_m, runner.PARAMS["press_depth_m"]
    assert np.isclose(estimate["elastic_unconfined_N"], E*d/H*A, rtol=1e-3)
    assert np.isclose(estimate["yield_uniaxial_N"], sigma_y*A, rtol=1e-3)
    assert np.isclose(estimate["yield_confined_punch_N"], (2+np.pi)/np.sqrt(3)*sigma_y*A, rtol=5e-3)
    assert np.isclose(estimate["yield_onset_depth_m"], H*sigma_y/E)
    weight = material["density_kg_m3"]*task.geometry.length_m*task.geometry.width_m*H*9.81
    assert np.isclose(estimate["bar_weight_N"], weight, rtol=1e-3)
    assert min(estimate["yield_uniaxial_N"], estimate["elastic_unconfined_N"]) > 4*runner.HANDLE_FORCE_N
    assert runner.PROTOCOL_V2["rows"][runner.ROW]["prediction"]["grip_contact_window_mean_N"][0] > runner.HANDLE_FORCE_N


def test_grip_footprint_is_over_the_right_pedestal_and_the_bar_top():
    runner = _v2()
    task = L.make_phase_f_task('CUT-03-K2')
    plan = runner.blunt_handle_v2_plan(task)
    report = runner.footprint_report(task, plan[4].pose)
    assert report["inside_pedestal_span"] and report["inside_bar_top"]
    assert np.allclose(report["footprint_local_x_m"], [.043, .067]) and np.allclose(report["footprint_local_y_m"], [-.011, .011])
    assert np.isclose(report["grip_bottom_local_z_m"], task.geometry.base_height_m-runner.PARAMS["press_depth_m"])
    start = runner.grip_corners_local(task, plan[2].pose)
    assert np.isclose(start[:, 2].min(), task.geometry.base_height_m+runner.PARAMS["start_grip_gap_m"])
    # Blade normal horizontal, edge line vertical: the grip's lowest face is the handle's far (-y edge) face.
    R = Rotation.from_quat(plan[4].pose[3:])
    assert np.allclose(R.apply([0., 1., 0.]), [0., 0., 1.]) and np.allclose(R.apply([0., 0., 1.]), [0., -1., 0.])


def test_plan_fits_the_episode_limits_and_keeps_the_blade_off_the_bar():
    runner = _v2()
    task = L.make_phase_f_task('CUT-03-K2')
    plan, policy, poses, phases, armed = _trajectory(runner, task)
    assert policy.done and len(poses)*task.control_dt_s <= task.duration_s-.1
    end = ideal_plan_end_s(task, plan, position_tolerance_m=runner.PARAMS["position_tolerance_m"])
    assert end is not None and end <= task.duration_s-.1
    limits = ActionLimits()
    local = task.points_to_local(poses[:, :3])
    assert np.all(local >= np.asarray(limits.workspace_min_local_m)) and np.all(local <= np.asarray(limits.workspace_max_local_m))
    assert {p for p, a in zip(phases, armed) if a} == {'handle_press', 'handle_hold'}
    clearance = runner.tool_clearance(task, poses, phases)
    assert min(c["blade_min_m"] for c in clearance.values()) > .005
    assert min(clearance[p]["blade_min_m"] for p in runner.PRESS_PHASES) >= .05
    for phase in ('lift', 'reorient', 'lower_outside'):
        assert clearance[phase]["grip_min_m"] > .002, phase
    assert np.isclose(clearance['handle_hold']["grip_min_m"], -runner.PARAMS["press_depth_m"], atol=1e-6)
    press = [i for i, p in enumerate(phases) if p == 'handle_press']
    speed = np.linalg.norm(np.diff(poses[press, :3], axis=0), axis=1)/task.control_dt_s
    assert np.allclose(speed[:-1], runner.PARAMS["press_speed_m_s"], atol=1e-9)


def test_blade_samples_cover_the_prism_and_box_distance():
    runner = _v2()
    task = L.make_phase_f_task('CUT-03-K2')
    blade = runner.blade_samples_edge(task.tool_profile)
    assert np.isclose(blade[:, 1].min(), -task.tool_profile.half_width_m) and np.isclose(blade[:, 1].max(), task.tool_profile.half_width_m)
    assert np.isclose(blade[:, 0].max(), 0.) and np.isclose(blade[:, 0].min(), -task.tool_profile.blade_depth_m)
    low, high = np.zeros(3), np.ones(3)
    d = runner._box_distance(np.array([[.5, .5, .5], [2., .5, .5], [2., 2., .5]]), low, high)
    assert np.allclose(d, [-.5, 1., np.sqrt(2.)])


def test_workpiece_motion_separates_rigid_motion_from_compression():
    runner = _v2()
    task = L.make_phase_f_task('CUT-03-K2')
    points, masses, _ = task.sample_particles()
    points, masses = np.asarray(points, float), np.asarray(masses, float)
    world = points @ task.T_world_workpiece[:3, :3].T + task.T_world_workpiece[:3, 3]
    footprint = runner.footprint_report(task, runner.blunt_handle_v2_plan(task)[4].pose)
    spacing = task.particle_spacing_m
    moved = world+[0., -.003, 0.]
    rigid = runner.workpiece_motion(task, moved, world, masses, footprint, spacing)
    assert np.isclose(rigid["rigid_fit_all_particles"]["com_displacement_norm_m"], .003)
    assert abs(rigid["under_footprint"]["compression_m"]) < 1e-12 and rigid["rigid_fit_all_particles"]["rotation_deg"] < 1e-6
    yawed = Rotation.from_euler('z', 1.4, degrees=True).apply(world)
    assert np.isclose(runner.workpiece_motion(task, yawed, world, masses, footprint, spacing)["rigid_fit_all_particles"]["yaw_deg"], 1.4)
    local = task.points_to_local(world)
    squeezed = local.copy()
    squeezed[:, 2] -= .002*local[:, 2]/task.geometry.base_height_m * (np.abs(local[:, 0]-.055) < .012)
    compressed = runner.workpiece_motion(task, squeezed @ task.T_world_workpiece[:3, :3].T + task.T_world_workpiece[:3, 3],
                                         world, masses, footprint, spacing)
    under = compressed["under_footprint"]
    assert under["top_layer"] > 0 and under["bottom_layer"] > 0
    assert 1.5e-3 < under["compression_m"] < 2e-3 and under["top_mean_dz_m"] < 0
    assert compressed["rigid_fit_all_particles"]["com_displacement_norm_m"] < under["compression_m"]


def test_verdict_requires_control_validity():
    runner = _v2()
    ok = {'a': True, 'b': True}
    assert runner.verdict_of(ok, {'blade': True}, True) == ('PASS', 'PASS')
    assert runner.verdict_of(ok, {'blade': False}, True) == ('INVALID_CONTROL', 'INVALID_CONTROL')
    assert runner.verdict_of({'a': True, 'b': False}, {'blade': True}, True) == ('FAIL', 'FAIL')
    assert runner.verdict_of(ok, {'blade': True}, False) == ('PASS', 'PASS_NONFORMAL')


def test_evaluate_v2_on_a_synthetic_episode(monkeypatch):
    """evaluate_v2 end to end on the ideal trajectory with synthetic forces (common metrics stubbed)."""
    runner = _v2()
    task = L.make_phase_f_task('CUT-03-K2')
    _, _, poses, phases, armed = _trajectory(runner, task)
    steps = []
    for k, (pose, phase, cut) in enumerate(zip(poses, phases, armed)):
        grip = [0., 0., .5] if phase in runner.PRESS_PHASES and k % 2 else [0., 0., 0.]
        steps.append(dict(step=k+1, time_s=round((k+1)*task.control_dt_s, 9), phase=phase, cut_enabled=cut,
                          edge_pose_world=pose.tolist(), blade_force_world_N=[0., 0., 0.], handle_force_local_N=grip,
                          contact_statistics=[0.]*6, stroke=0, travel_m=0., engaged=False, any_motion=False,
                          labelled_count=0, cutting_segments_last_substep=0, edge_segments=8))
    points, masses, _ = task.sample_particles()
    world = np.asarray(points, float) @ task.T_world_workpiece[:3, :3].T + task.T_world_workpiece[:3, 3]
    times = np.asarray([s["time_s"] for s in steps])[4::5]
    data = {"time_s": times, "positions": np.repeat(world[None], len(times), 0), "mass_kg": np.asarray(masses, float),
            "reference_positions": world}
    base = {'final_mechanical_components': 1, 'final_labelled_count': 0, 'max_labelled_count': 0,
            'mass_check': {'pass': True}}
    monkeypatch.setattr(runner.F, 'common_metrics', lambda *a, **k: (dict(base), None))
    checks, validity, metrics = runner.evaluate_v2(task, steps, data, {}, {}, {'particle_spacing_m': task.particle_spacing_m,
                                                                               'stability': {'stable': True}})
    assert all(checks.values()) and all(validity.values())
    assert np.isclose(metrics['contact_window']['mean_N'], .5) and metrics['blade_contact_steps']['episode'] == 0
    assert metrics['footprint']['pose_source'] == 'recorded'
    assert 'error' not in metrics['tool_clearance_recorded_poses'] and 'error' not in metrics['workpiece_motion']
    assert metrics['workpiece_motion']['hold_end_vs_pre_press']['under_footprint']['compression_m'] == 0.
    steps[-1]['contact_statistics'] = [0., 3., 0., 0., 0., 0.]
    _, validity, _ = runner.evaluate_v2(task, steps, data, {}, {}, {'particle_spacing_m': task.particle_spacing_m,
                                                                    'stability': {'stable': True}})
    assert not validity['zero_blade_contact_steps_episode']
