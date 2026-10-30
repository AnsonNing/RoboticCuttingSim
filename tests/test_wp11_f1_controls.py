"""CPU checks of the opt-in WP-11 Phase F1 path of run_physical_tool_controls (protocol v8)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_physical_tool_controls as controls  # noqa: E402

V7 = ROOT/"configs/physical_tool_controls_v7_ruler_v2_native.json"
V8 = ROOT/"configs/physical_tool_controls_v8_ruler_v2_wp11.json"


@pytest.fixture(scope="module")
def protocol():
    return json.loads(V8.read_text())


def test_v8_inherits_v7_blocks_unchanged(protocol):
    v7 = json.loads(V7.read_text())
    for key in ("normal_grip", "transfer", "measurement", "rigid_dt_s", "control_dt_s", "native_contact_geometry_profile"):
        assert protocol[key] == v7[key]
    assert protocol["parent_protocol"]["sha256"] == controls.sha256(V7)
    assert protocol["start_profile"] == controls.WP11_START_PROFILE
    assert "start_profile" not in v7  # earlier protocols never reach the F1 worker


def test_v8_cases_are_predeclared_with_known_gates(protocol):
    groups = set(protocol["gate_groups"])
    for name, case in protocol["cases"].items():
        assert case["gate"] in groups, name
        assert case["start"] in ("home", "trim")
        assert case["force_limit_n"] <= protocol["normal_grip"]["max_finger_force_n"]
        if "comparison" in case:
            assert case["comparison"] in protocol["cases"]
            assert case["gate"] in ("negative_control", "diagnostic")
    stress = [n for n, c in protocol["cases"].items() if n.startswith("force_") and max(map(abs, c["force_edge_n"])) >= 2]
    assert stress and all(protocol["cases"][n]["gate"] == "reported" for n in stress)


def test_reorient_schedule_is_the_planner_rate(protocol):
    task = controls.wp11_task(protocol)
    start, targets = controls.wp11_reorient_schedule(task)
    poses = np.vstack([start[None], targets])
    rotation = Rotation.from_quat(poses[:, 3:])
    per_step = (rotation[1:]*rotation[:-1].inv()).magnitude()
    assert per_step.max() <= np.deg2rad(task.controller["planner"]["max_rotation_deg"])+1e-9
    assert np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).max() <= task.controller["planner"]["max_translation_m"]+1e-9
    assert np.allclose(start, task.reference_edge_pose_world(0.))
    double = controls.wp11_rate_targets(targets, 2)
    assert np.array_equal(double[-1], targets[-1]) and len(double) == int(np.ceil(len(targets)/2))
    with pytest.raises(ValueError):
        controls.wp11_rate_targets(targets, 0)


def test_edge_wrench_uses_measured_edge_axes():
    pose = np.r_[0.1, 0.2, 0.3, Rotation.from_euler("y", 10, degrees=True).as_quat()]
    wrench = controls.wp11_edge_wrench({"force_edge_n": [-1., 0., 0.], "torque_edge_n_m": [0., 0., .05]}, pose)
    matrix = Rotation.from_quat(pose[3:]).as_matrix()
    assert np.allclose(wrench[:3], -matrix[:, 0]) and np.allclose(wrench[3:], .05*matrix[:, 2])


def test_replay_is_reexpressed_in_the_edge_frame(tmp_path, monkeypatch):
    source_rotation = Rotation.from_euler("z", 30, degrees=True)
    rows = [{"edge_pose_world": [0., 0., 0., *source_rotation.as_quat()], "blade_force_world_N": [0.01*k, 0., -0.02]} for k in range(3)]
    path = tmp_path/"actions.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    monkeypatch.setattr(controls, "ROOT", tmp_path)
    spec = {"path": "actions.jsonl", "sha256": controls.sha256(path), "steps": 3,
            "force_field": "blade_force_world_N", "edge_pose_field": "edge_pose_world"}
    target = Rotation.from_euler("x", 90, degrees=True)
    wrenches = controls.wp11_replay_wrenches(spec, 10., np.r_[1., 2., 3., target.as_quat()])
    expected = 10*target.apply(source_rotation.inv().apply(np.asarray([r["blade_force_world_N"] for r in rows])))
    assert np.allclose(wrenches[:, :3], expected) and np.all(wrenches[:, 3:] == 0)
    with pytest.raises(ValueError):
        controls.wp11_replay_wrenches({**spec, "sha256": "0"*64}, 1., np.r_[0., 0., 0., 0., 0., 0., 1.])


def _metrics(**overrides):
    metrics = {"evidence_available": True, "finite": True, "raw_sampling_complete": True, "all_contact_samples_valid": True,
               "arm_samples_complete": True, "max_relative_translation_m": 1e-6, "max_relative_rotation_rad": 1e-5,
               "maximum_relative_linear_speed_last_window_m_s": 1e-4, "maximum_relative_angular_speed_last_window_rad_s": 1e-3,
               "bilateral_loaded_handle_fraction": 1., "maximum_contact_penetration_m": 1e-5, "unintended_loaded_contact_steps": 0,
               "maximum_downward_tool_displacement_m": 0., "freeflight_vertical_acceleration_m_s2": None}
    metrics.update(overrides)
    return metrics


def test_hold_gate_rejects_slip_and_monitor_loss(protocol):
    extra = {"grasp_monitor": {"ever_lost": False}}
    hold = {"gate": "hold"}
    assert controls.classify_wp11(hold, _metrics(), extra, protocol, True, True)["status"] == "PASS_CASE_GATES"
    assert controls.classify_wp11(hold, _metrics(max_relative_translation_m=.0021), extra, protocol, True, True)["status"] == "FAIL_CASE_GATES"
    assert controls.classify_wp11(hold, _metrics(), {"grasp_monitor": {"ever_lost": True}}, protocol, True, True)["status"] == "FAIL_CASE_GATES"
    assert controls.classify_wp11(hold, _metrics(), extra, protocol, False, True)["status"] == "FAIL_CASE_GATES"
    reported = controls.classify_wp11({"gate": "reported"}, _metrics(max_relative_translation_m=.03), extra, protocol, True, True)
    assert reported["status"] == "PASS_CASE_GATES" and not reported["hold_gates_all_pass"] and not reported["gated"]
    with pytest.raises(ValueError):
        controls.classify_wp11({"gate": "unknown"}, _metrics(), extra, protocol, True, True)


def test_negative_comparison_needs_matched_start_and_larger_response(protocol):
    state = {"tool_body_pose_xyzw": [0., 0., 0., 0., 0., 0., 1.], "q": [[0.]*7, [0.]*7]}
    config = {"start": "trim", "force_edge_n": None}
    reference = {"case": "trim_hold_30s", "runtime": "isaac_sim", "source_sha256": {"a": "1"}, "initial_state": state,
                 "case_config": config, "metrics": _metrics(), "classification": {"hold_gates_all_pass": True}}
    weak = {**reference, "case": "weak_grip_trim", "metrics": _metrics(max_relative_translation_m=.03, bilateral_loaded_handle_fraction=.1),
            "extra_metrics": {"grasp_monitor": {"ever_lost": True}}}
    row = controls.compare_wp11(reference, weak, protocol)
    assert row["matched"] and row["larger_response"] and row["grip_loss_event"]
    moved = {**weak, "initial_state": {**state, "tool_body_pose_xyzw": [.01, 0., 0., 0., 0., 0., 1.]}}
    assert not controls.compare_wp11(reference, moved, protocol)["larger_response"]
    same = {**weak, "metrics": _metrics(max_relative_translation_m=.004)}
    assert not controls.compare_wp11(reference, same, protocol)["larger_response"]


def test_extra_metrics_handles_slip_and_loss_reports(protocol):
    """Regression (run f1_grasp_001): monitor reports have no time key; SLIPPING/LOST must not raise."""
    pose = [0., 0., 0., 0., 0., 0., 1.]
    samples = [{"time_s": .002*(i+1), "left_load_n": 18., "right_load_n": 17., "contact_force": [0., 0., .9],
                "tool": pose, "hand": pose} for i in range(400)]
    monitor = [{"status": status, "grasp_lost": status == "LOST", "slip_since_settling_m": .001*i, "rotation_since_settling_rad": .01*i}
               for i, status in enumerate(["HOLDING", "SLIPPING", "SLIPPING", "LOST"])]
    extra = controls.wp11_extra_metrics(samples, {"hand_pose": pose, "tool_body_pose_xyzw": pose}, monitor, protocol)
    events = extra["grasp_monitor"]
    assert events["first_slipping_time_s"] == pytest.approx(.04) and events["first_lost_time_s"] == pytest.approx(.08)
    assert events["ever_lost"] and events["final_status"] == "LOST"
    json.dumps(extra, allow_nan=False)
