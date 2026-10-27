import copy
import json

import numpy as np
import pytest

from robotic_cutting_sim.live_acceptance import (
    cut_history_metrics, gate, graph_audit, matched_recordings,
    read_reset_metrics, repeated_state_metrics, same_partition,
    score_live_camera_records,
)


def test_missing_measurements_never_promote_a_gate_and_measured_failures_remain_visible():
    assert gate({"available_metric": True}, missing=["required control"])["status"] == "NOT_RUN"
    assert gate({"available_metric": False}, missing=["required control"])["status"] == "FAIL"
    assert gate()["status"] == "NOT_RUN"


def test_partition_labels_may_be_renamed_but_a_bridge_cannot_be_ignored():
    assert same_partition([0, 0, 1, 1], [7, 7, 2, 2])
    assert not same_partition([0, 0, 1, 1], [7, 7, 7, 2])


def test_persistent_digits_detect_reset_and_reassignment_independently():
    history = np.array([[0, 0], [1, 2], [7, 5], [6, 4]], dtype=int)
    metrics = cut_history_metrics(history, 2)
    assert metrics["nonzero_digit_rewrites_by_cut"] == [2, 0]
    assert metrics["initial_nonzero_codes"] == 0
    # Assigning another digit never rewrites the first digit.
    assert cut_history_metrics(history[:3], 2)["nonzero_digit_rewrites"] == 0


def test_invalid_cut_code_cannot_be_interpreted_as_valid_history():
    with pytest.raises(ValueError):
        cut_history_metrics(np.array([[0], [9]]), 2)


def test_graph_includes_unassigned_gradient_only_bridge():
    # The middle code-zero particle bridges opposite side codes via stress
    # gradients at node1; dropping zero weights would falsely report a cut.
    physics = {"codes": np.array([[0, 0, 0], [1, 0, 2]]),
        "reference_stencil_nodes": np.array([[0, 1], [1, 2], [2, 3]]),
        "reference_stencil_weights": np.array([[1., 0], [1, 0], [1, 0]]),
        "reference_stencil_gradients": np.tile([[[1., 0, 0], [1., 0, 0]]], (3, 1, 1))}
    result, _ = graph_audit(physics, 1)
    assert result["initial_components"] == 1
    assert result["final_components"] == 1
    physics["codes"][-1, 1] = 1
    result, _ = graph_audit(physics, 1)
    assert result["final_components"] == 2


def test_recorded_false_component_partition_fails_recomputation():
    physics = {"codes": np.array([[0, 0], [1, 2]]),
        "reference_stencil_nodes": np.ones((2, 1), int),
        "reference_stencil_weights": np.ones((2, 1)),
        "reference_stencil_gradients": np.ones((2, 1, 3)), "component_ids": np.array([0, 0])}
    report, _ = graph_audit(physics, 1)
    assert report["status"] == "FAIL"
    assert report["final_components"] == 2


def test_repeat_errors_measure_euclidean_motion_and_separate_bitwise_claims():
    first = {"time_s": np.array([0, 1]), "positions": np.zeros((2, 2, 3)),
        "velocities": np.zeros((2, 2, 3)), "mass_kg": np.array([1., 3.]),
        "codes": np.array([[0, 0], [1, 2]]), "component_ids": np.array([0, 1])}
    second = copy.deepcopy(first)
    second["positions"][1, 0] = [3e-5, 4e-5, 0]
    second["codes"][0, 0] = 1
    result = repeated_state_metrics(first, second)
    assert result["all_frame_max_position_error_m"] == pytest.approx(5e-5)
    assert result["final_component_com_error_m"] == pytest.approx(5e-5)
    assert result["velocities_bitwise_equal"]
    assert not result["positions_bitwise_equal"]
    assert result["differing_code_particle_frames"] == 1


def paired_control():
    first = {"runtime": {"source": "live_isaac_lab"},
        "config": {"seed": 42, "feedback": True, "camera": True, "static_scene": "one.usda"},
        "local_code_sha256": {"src/robotic_cutting_sim/isaac_lab_env.py": "actual_same",
                              "src/robotic_cutting_sim/isaac_sim_runner.py": "irrelevant_one"}}
    second = copy.deepcopy(first)
    second["config"].update(feedback=False, camera=False, static_scene="two.usda")
    second["local_code_sha256"]["src/robotic_cutting_sim/isaac_sim_runner.py"] = "irrelevant_two"
    setup = {"material": {"density": 1500., "contact_friction": .3}}
    robot = {"time_s": np.array([0., 1.]), "desired_tool_pose_xyzw": np.zeros((2, 7))}
    return first, second, setup, copy.deepcopy(setup), robot, copy.deepcopy(robot)


def test_feedback_control_is_matched_despite_independent_observation_only_code():
    assert all(matched_recordings(*paired_control(), feedback_control=True).values())


def test_changed_actual_recipe_or_command_invalidates_feedback_control():
    args = list(paired_control())
    args[3]["material"]["contact_friction"] = .4
    args[5]["desired_tool_pose_xyzw"][1, 0] = .001
    result = matched_recordings(*args, feedback_control=True)
    assert not result["same_material_recipe"]
    assert not result["same_commanded_edge_path"]


def test_changed_actual_controller_source_invalidates_feedback_control():
    args = list(paired_control())
    args[1]["local_code_sha256"]["src/robotic_cutting_sim/isaac_lab_env.py"] = "changed"
    result = matched_recordings(*args, feedback_control=True)
    assert not result["same_source_isaac_lab_env.py"]


def test_reset_sensor_holes_are_separate_from_common_valid_depth_error(tmp_path):
    (tmp_path/"observations").mkdir()
    (tmp_path/"reset_observations").mkdir()
    initial = np.array([[1., 1.], [2., 0]], dtype=np.float32)
    reset = np.array([[1.003, 0], [2., 0]], dtype=np.float32)
    np.savez(tmp_path/"observations/d455_0_0000.npz", depth_m=initial, valid_mask=initial > 0)
    np.savez(tmp_path/"reset_observations/d455_0.npz", depth_m=reset, valid_mask=reset > 0)
    (tmp_path/"reset_receipt.json").write_text(json.dumps({"q_max_error_rad": 0., "material_max_error_m": 0.,
        "nonzero_cut_codes": 0, "dynamics_reset_measured": True,
        "camera_reset_repeatability": {"d455_0": {"max_depth_error_m": 1., "valid_mask_changed_pixels": 1}}}))
    report = read_reset_metrics(tmp_path)
    measured = report["raw_camera_rechecks"]["d455_0"]
    assert measured["matches_instrumented_receipt"]
    assert measured["max_all_pixel_depth_error_m"] == 1.
    assert measured["common_valid_max_error_m"] == pytest.approx(.003, abs=1e-7)
    assert not report["depth_valid_mask_exact"]["d455_0"]


def camera_records():
    row = {"geometry_vs_actual_rtx": {"p95_absolute_error_m": 1e-7, "valid_mask_iou": 1.},
        "actual_native_vs_geometry": {"p95_absolute_error_m": 2e-7, "valid_mask_iou": .4, "max_absolute_error_m": .9},
        "timestamp_span_s": 0., "native_valid_chip_pixels": 30,
        "rtx_depth_consistent_attributed_pixels": {name: 30 for name in ("franka_0", "franka_1", "knife", "material", "table")}}
    return [{"cameras": {name: copy.deepcopy(row) for name in ("d455_0", "d455_1", "global")}}]


def test_accurate_visual_geometry_cannot_hide_actual_native_depth_holes():
    from robotic_cutting_sim.recording import ROOT
    protocol = json.loads((ROOT/"configs/isaac_acceptance.json").read_text())
    result = score_live_camera_records(camera_records(), protocol)
    assert result["CAM-03"]["status"] == "PASS"
    assert result["CAM-04"]["status"] == "FAIL"
    assert result["CAM-04"]["checks"]["visual_mesh_RTX_bridge_masks"]
    assert not result["CAM-04"]["checks"]["actual_native_valid_masks"]
    assert result["CAM-04"]["measurements"]["actual_native"]["d455_0"]["maximum_absolute_error_m"] == .9


def test_camera_presence_is_not_enough_when_knife_has_too_few_visible_pixels():
    from robotic_cutting_sim.recording import ROOT
    protocol = json.loads((ROOT/"configs/isaac_acceptance.json").read_text())
    records = camera_records()
    records[0]["cameras"]["global"]["rtx_depth_consistent_attributed_pixels"]["knife"] = 24
    assert score_live_camera_records(records, protocol)["CAM-03"]["status"] == "FAIL"
