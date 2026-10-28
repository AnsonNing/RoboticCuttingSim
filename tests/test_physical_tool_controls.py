"""Analytic negative controls for control classification; no simulator claims."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_physical_tool_controls import (
    classify_case, compare_cases, measure_case, native_samples, smooth_joint_targets, summarize_matrix,
)

PROTOCOL=json.loads((ROOT/"configs/physical_tool_controls_v1.json").read_text())


def analytic_trace(duration=2., release=False, hand_moves=False):
    times=np.arange(1,round(duration/.002)+1)*.002
    initial={"hand_pose":[0.,0.,.3,0.,0.,0.,1.],"tool_body_pose_xyzw":[0.,0.,.25,0.,0.,0.,1.],"q":np.zeros((2,7))}
    samples=[]
    for t in times:
        hand=np.array(initial["hand_pose"]);tool=np.array(initial["tool_body_pose_xyzw"])
        if hand_moves:hand[0]=t;tool[0]=t
        if release:tool[2]-=.5*9.81*t*t
        samples.append(dict(time_s=t,hand=hand,tool=tool,linear=[0,0,-9.81*t if release else 0],angular=[0,0,0],
            left_load_n=0. if release else 10.,right_load_n=0. if release else 10.,valid=True,unexpected=0,
            contact_count=0 if release else 4,penetration_m=0.,contact_force=[0.,0.,.3]))
    control=[{"phase":"perturbation","state":{"q":np.zeros((2,7))}} for _ in range(round(duration/.02))]
    return samples,control,initial,{"com_tool_m":[0.,0.,0.]}


def test_quintic_targets_respect_speed_and_step_bounds_at_large_joint_motion():
    start=np.zeros(7);target=np.array([2.,-.8,.4,-1.,.2,.6,-.1])
    trajectory=smooth_joint_targets(start,target,PROTOCOL["transfer"],.02)
    np.testing.assert_allclose(trajectory[-1],target)
    delta=np.diff(np.vstack([start,trajectory]),axis=0)
    assert np.max(np.abs(delta)) <= .005+1e-10
    assert np.max(np.abs(delta))/.02 <= .25+1e-10
    assert len(trajectory)*.02 >= 1.875*2/.25
    assert np.max(np.abs(delta[0])) < .00001 and np.max(np.abs(delta[-1])) < .00001


def test_arm_motion_cannot_masquerade_as_independent_release():
    samples,control,initial,spec=analytic_trace(.5,hand_moves=True)
    m=measure_case(samples,control,initial,spec,PROTOCOL,.5)
    assert m["max_relative_translation_m"] == 0.
    assert classify_case("release",m,PROTOCOL)["status"]=="FAIL_CASE_ENGINEERING_SCREENS"


def test_unsupported_parabolic_com_motion_passes_release_but_not_stable_hold():
    samples,control,initial,spec=analytic_trace(.5,release=True)
    m=measure_case(samples,control,initial,spec,PROTOCOL,.5)
    assert abs(m["freeflight_vertical_acceleration_m_s2"]+9.81)<1e-10
    assert classify_case("release",m,PROTOCOL)["status"]=="PASS_CASE_ENGINEERING_SCREENS"
    assert classify_case("baseline_hold",m,PROTOCOL)["status"]=="FAIL_CASE_ENGINEERING_SCREENS"


def test_missing_substeps_or_invalid_sensor_cannot_pass_stable_grasp():
    samples,control,initial,spec=analytic_trace()
    good=measure_case(samples,control,initial,spec,PROTOCOL,2.)
    assert classify_case("baseline_hold",good,PROTOCOL)["status"]=="PASS_CASE_ENGINEERING_SCREENS"
    missing=measure_case(samples[::10],control,initial,spec,PROTOCOL,2.)
    assert classify_case("baseline_hold",missing,PROTOCOL)["status"]=="FAIL_CASE_ENGINEERING_SCREENS"
    samples[0]["valid"]=False
    invalid=measure_case(samples,control,initial,spec,PROTOCOL,2.)
    assert not classify_case("baseline_hold",invalid,PROTOCOL)["checks"]["all_contact_samples_valid"]


def test_weak_response_requires_same_load_source_and_matched_actual_start():
    initial=analytic_trace()[2]
    strong={"runtime":"isaac_sim","initial_state":initial,"source_sha256":{"solver":"a"},
        "case_config":{"edge_wrench_world":[0,0,-1,0,0,0]},
        "metrics":{"max_relative_translation_m":.0001,"max_relative_rotation_rad":.001}}
    weak=deepcopy(strong);weak["metrics"]["max_relative_translation_m"] = .03
    assert compare_cases(strong,weak,PROTOCOL)["larger_weak_response"]
    weak["initial_state"]["tool_body_pose_xyzw"][0] += .02
    assert not compare_cases(strong,weak,PROTOCOL)["matched"]
    weak=deepcopy(strong);weak["metrics"]["max_relative_translation_m"] = .03
    weak["source_sha256"]["solver"]="different"
    assert not compare_cases(strong,weak,PROTOCOL)["larger_weak_response"]
    weak=deepcopy(strong);weak["metrics"]["max_relative_translation_m"] = .03
    weak["case_config"]["edge_wrench_world"][2]=-10
    assert not compare_cases(strong,weak,PROTOCOL)["matched"]


def test_stress_can_hold_without_being_falsely_required_to_drop():
    samples,control,initial,spec=analytic_trace()
    m=measure_case(samples,control,initial,spec,PROTOCOL,2.)
    assert classify_case("stress_load",m,PROTOCOL)["status"]=="PASS_CASE_ENGINEERING_SCREENS"
    assert not classify_case("stress_load",m,PROTOCOL)["full_physics_acceptance"]


def test_exit_zero_or_empty_matrix_does_not_prove_control_execution(tmp_path):
    result=summarize_matrix({"cases":{"isaac_sim/baseline_hold":{"directory":"missing","runtime":"isaac_sim","returncode":0}}},tmp_path,PROTOCOL)
    assert result["status"]=="FAIL_OR_INCOMPLETE_RIGID_CONTROLS"
    assert not result["all_requested_cases_executed"]
    assert not result["full_physics_acceptance"]


def test_invalidated_native_view_cannot_be_hidden_by_a_success_receipt(tmp_path):
    case=tmp_path/"case";case.mkdir()
    (case/"result.json").write_text(json.dumps({"status":"EXECUTED_RIGID_CONTROL","metrics":{"evidence_available":True}}))
    index={"cases":{"isaac_sim/baseline_hold":{"directory":"case","runtime":"isaac_sim","returncode":0,
        "status":"FAIL_NATIVE_CONTACT_VIEW_INVALIDATION","native_contact_view_invalidations":1}}}
    result=summarize_matrix(index,tmp_path,PROTOCOL)
    assert result["status"]=="FAIL_OR_INCOMPLETE_RIGID_CONTROLS"
    assert not result["all_requested_cases_executed"]


def test_outside_handle_finger_point_is_excluded_from_handle_load_without_inventing_other_body(tmp_path):
    def contact(other, point):
        return {"body0":"/World/Tools/CuttingTool_0","body1":other,
            "raw_impulse_world_ns":[0.,.002,0.],"dt_s":.002,"time_s":.002,"position_world_m":point}
    contacts=[contact("/World/Franka_0/panda_leftfinger",[.008002,0.,0.]),
        contact("/World/Franka_0/panda_rightfinger",[0.,.011,0.]),
        contact("/World/Table",[0.,0.,-.02])]
    row={"native_time_s":.002,"hand_pose_xyzw":[0.,0.,0.,0.,0.,0.,1.],
        "tool_body_pose_xyzw":[0.,0.,0.,0.,0.,0.,1.],"tool_linear_velocity_world_m_s":[0.,0.,0.],
        "tool_angular_velocity_world_rad_s":[0.,0.,0.],
        "contacts":{"contacts":contacts,"raw_contact_count":3,"sensor_valid":True,"sensor_fresh":True,
            "new_physics_step":True,"tensor_forces":{"status":"VALID_NATIVE_NORMAL_AND_FRICTION_DATA"}}}
    path=tmp_path/"raw.jsonl";path.write_text(json.dumps(row)+"\n")
    sample=native_samples(path,0.,.002,{"handle_half_extents_m":[.008,.011,.02]},.05)[0]
    assert sample["unexpected"] == 1  # Actual table pair, independent of the finger-region test.
    assert sample["left_load_n"] == 0. and sample["right_load_n"] == 1.
    assert sample["loaded_finger_points_outside_handle"] == 1
    assert abs(sample["maximum_loaded_finger_point_outside_handle_m"]-2e-6) < 1e-12


def test_moving_rigid_contact_uses_actual_detection_pose_without_expanding_handle_region(tmp_path):
    # A handle moving 1 mm after detection makes the old world contact point
    # leave its POST-step box. The actual detection pose still contains it.
    before=[0.,0.,0.,0.,0.,0.,1.];after=[-.001,0.,0.,0.,0.,0.,1.]
    contact={"body0":"/World/Tools/CuttingTool_0","body1":"/World/Franka_0/panda_leftfinger",
        "raw_impulse_world_ns":[0.,.002,0.],"dt_s":.002,"time_s":.002,"position_world_m":[.008,0.,0.]}
    row={"native_time_s":.002,"hand_pose_xyzw":after,"tool_body_pose_xyzw":after,
        "tool_body_pose_before_step_xyzw":before,"tool_linear_velocity_world_m_s":[-.5,0.,0.],
        "tool_angular_velocity_world_rad_s":[0.,0.,0.],"contacts":{"contacts":[contact],"raw_contact_count":1,
            "sensor_valid":True,"sensor_fresh":True,"new_physics_step":True,
            "tensor_forces":{"status":"VALID_NATIVE_NORMAL_AND_FRICTION_DATA"},
            "contact_geometry_frame":"PRE_STEP_MEASURED_TOOL_POSE","contact_geometry_time_s":0.,
            "contact_geometry_tool_pose_xyzw":before}}
    path=tmp_path/"raw.jsonl";path.write_text(json.dumps(row)+"\n");spec={"handle_half_extents_m":[.008,.011,.02]}
    sample=native_samples(path,0.,.002,spec,.05,geometry_profile="PRE_STEP_MEASURED_TOOL_POSE")[0]
    assert sample["left_load_n"] == 1. and sample["loaded_finger_points_outside_handle"] == 0
    row["contacts"].pop("contact_geometry_frame");row["contacts"].pop("contact_geometry_tool_pose_xyzw")
    path.write_text(json.dumps(row)+"\n")
    legacy=native_samples(path,0.,.002,spec,.05)[0]
    assert legacy["left_load_n"] == 0. and legacy["loaded_finger_points_outside_handle"] == 1
    with pytest.raises(ValueError,match="frozen profile"):
        native_samples(path,0.,.002,spec,.05,geometry_profile="PRE_STEP_MEASURED_TOOL_POSE")


def test_pre_step_profile_rejects_unmatched_pose_or_timestamp(tmp_path):
    before=[0.,0.,0.,0.,0.,0.,1.]
    row={"native_time_s":.002,"tool_body_pose_xyzw":before,"tool_body_pose_before_step_xyzw":before,
        "contacts":{"contact_geometry_frame":"PRE_STEP_MEASURED_TOOL_POSE",
            "contact_geometry_tool_pose_xyzw":before,"contact_geometry_time_s":.002}}
    path=tmp_path/"raw.jsonl";path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="pose/time"):
        native_samples(path,0.,.002,{"handle_half_extents_m":[.008,.011,.02]},.05)
    row["contacts"]["contact_geometry_time_s"]=0.
    row["tool_body_pose_before_step_xyzw"]=[.001,0.,0.,0.,0.,0.,1.]
    path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="pose/time"):
        native_samples(path,0.,.002,{"handle_half_extents_m":[.008,.011,.02]},.05)


def test_another_robot_finger_is_unintended_support_not_an_active_grasp(tmp_path):
    contact={"body0":"/World/Tools/CuttingTool_0","body1":"/World/Franka_1/panda_leftfinger",
        "raw_impulse_world_ns":[0.,.002,0.],"dt_s":.002,"time_s":.002,"position_world_m":[0.,0.,0.]}
    row={"native_time_s":.002,"hand_pose_xyzw":[0.,0.,0.,0.,0.,0.,1.],
        "tool_body_pose_xyzw":[0.,0.,0.,0.,0.,0.,1.],"tool_linear_velocity_world_m_s":[0.,0.,0.],
        "tool_angular_velocity_world_rad_s":[0.,0.,0.],"contacts":{"contacts":[contact],"raw_contact_count":1,
            "sensor_valid":True,"sensor_fresh":True,"new_physics_step":True,
            "tensor_forces":{"status":"VALID_NATIVE_NORMAL_AND_FRICTION_DATA"}}}
    path=tmp_path/"raw.jsonl";path.write_text(json.dumps(row)+"\n")
    sample=native_samples(path,0.,.002,{"handle_half_extents_m":[.008,.011,.02]},.05)[0]
    assert sample["left_load_n"] == 0. and sample["unexpected"] == 1


def test_lab_only_contact_protocol_cannot_silently_run_on_native_backend(tmp_path):
    protocol=deepcopy(PROTOCOL);protocol["allowed_runtimes"]=["isaac_lab"]
    frozen=tmp_path/"protocol.json";frozen.write_text(json.dumps(protocol))
    output=tmp_path/"forbidden_run"
    result=subprocess.run([sys.executable,str(ROOT/"scripts/run_physical_tool_controls.py"),
        "--protocol",str(frozen),"--runtime","isaac_sim","--output-dir",str(output),"--dry-run"],
        capture_output=True,text=True,check=False)
    assert result.returncode == 2 and "allowed_runtimes" in result.stderr
    assert not output.exists()
