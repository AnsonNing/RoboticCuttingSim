"""Known geometry and adversarial evidence; no runtime/physical-cutting claim."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from robotic_cutting_sim.evaluation import GoalSpec, evaluate_recording, goal_for_task
from robotic_cutting_sim.surface_evaluation import TargetGeometry, RegionSpec, evaluate_height_surface, evaluate_3d_surface
from robotic_cutting_sim.tasks import PROTOCOL_PATH
from test_cutting_tasks import small_recording


def flat_mesh(z=.02, low=-.01, high=.01):
    return np.array([[low, low, z], [high, low, z], [high, high, z], [low, high, z]]), np.array([[0, 1, 2], [0, 2, 3]])


def columns(v, f, target=None, **kwargs):
    return evaluate_height_surface(v, f, target or TargetGeometry("plane", "analytic fixture", "v1", parameters={"origin_m": [0, 0, .02]}),
        world_from_target_xyzw=kwargs.pop("pose", (0,0,0,0,0,0,1)),
        actual_frame_id=kwargs.pop("frame", "workpiece_local"), bounds_xy_m=[[-.01,-.01],[.01,.01]], spacing_m=.001, **kwargs)


def with_provenance(data):
    return {**data, "source_evidence": {"implementation_verified":True,"physics_state_from_recording":True,"contact_deformation_traceable":True}}


def test_missing_mesh_is_unknown_and_never_a_measured_twenty_mm_gouge():
    v, f = flat_mesh()
    result, raw = columns(v, f[:0])
    metric = result["analytic_target"]["entire_region"]
    assert metric["status"] == "NOT_MEASURABLE"
    assert metric["missing_actual_fraction"] == 1
    assert metric["measured_maximum_surface_deficit_m"] is None
    assert metric["whole_region_deficit_volume_m3"] is None
    assert np.isnan(raw["actual_height_m"]).all()
    assert metric["unknown_missing_column_volume_budget_m3"] == pytest.approx(.02*.02*.02)
    assert metric["true_material_gouge"].startswith("NOT_MEASURABLE")


def test_flat_partial_missing_boundary_and_real_mesh_deficit_remain_distinct():
    v,f = flat_mesh()
    region = RegionSpec("edge", "boundary", {"bounds_xy_m":[[-.01,-.01],[.01,.01]],"width_m":.003})
    exact,raw = columns(v,f,regions=(region,))
    assert exact["analytic_target"]["entire_region"]["paired_p95_absolute_error_m"] == 0
    missing,_ = columns(v,f[:1],regions=(region,))
    m = missing["analytic_target"]["entire_region"]
    assert m["missing_actual_fraction"] > .4 and m["paired_p95_absolute_error_m"] == 0
    assert m["whole_region_deficit_volume_m3"] is None
    assert missing["missing_classification"]["boundary_missing_columns"] > 0
    deficit,_ = columns(v-[0,0,.004],f)
    assert deficit["analytic_target"]["entire_region"]["measured_maximum_surface_deficit_m"] == pytest.approx(.004)
    assert deficit["analytic_target"]["entire_region"]["missing_actual_fraction"] == 0


def test_independent_represented_reference_exposes_reconstruction_floor_without_fitting():
    v,f = flat_mesh(.0185)
    target = TargetGeometry("mesh", "independent uncut fixture", "frozen_before_cut", parameters={
        "vertices_m":v, "triangles":f,"independent_uncut_target":True})
    r,raw = columns(v,f,represented_target=target)
    assert r["analytic_target"]["entire_region"]["paired_p95_absolute_error_m"] == pytest.approx(.0015)
    assert r["represented_target"]["regions"]["entire_region"]["paired_p95_absolute_error_m"] == 0
    assert r["represented_target"]["reconstruction_floor_vs_analytic"]["paired_p95_absolute_error_m"] == pytest.approx(.0015)
    with pytest.raises(ValueError,match="independent"):
        columns(v,f,represented_target=replace(target,parameters={"vertices_m":v,"triangles":f}))


def test_target_frame_mismatch_cannot_pass_and_rigid_world_transform_is_invariant():
    v,f=flat_mesh()
    result,_=columns(v,f,frame="unregistered_sensor")
    assert result["status"] == "NOT_MEASURABLE"
    from robotic_cutting_sim.scene_frames import pose_matrix,transform_points
    from scipy.spatial.transform import Rotation
    pose=[.2,-.1,.3,*Rotation.from_euler("xyz",[.2,.3,.4]).as_quat()]
    r,_=columns(transform_points(pose_matrix(pose),v),f,pose=pose)
    assert r["analytic_target"]["entire_region"]["paired_p95_absolute_error_m"] < 1e-15


def test_nonflat_heightfield_and_curved_implicit_targets():
    v,f=flat_mesh()
    v[:,2]=v[:,0]*.2+v[:,1]*.1+.02
    target=TargetGeometry("heightfield","independent sloped surface","v1",parameters={
        "x_m":[-.01,.01],"y_m":[-.01,.01],"height_m":[[.017,.021],[.019,.023]]})
    r,_=columns(v,f,target)
    assert r["analytic_target"]["entire_region"]["paired_p95_absolute_error_m"] < 1e-16
    sphere=TargetGeometry("implicit","analytic sphere radius 1m","v1",parameters={
        "signed_distance":lambda q:np.linalg.norm(q,axis=1)-1})
    sv=np.array([[1,0,0],[0,1,0],[0,0,1]],float)
    sf=np.array([[0,1,2]])
    r,raw=evaluate_3d_surface(sv,sf,sphere,actual_frame_id="workpiece_local")
    assert r["status"] == "MEASURED_DIRECTED_ONLY"
    assert r["area_weighted_p95_distance_m"] == pytest.approx(1-1/np.sqrt(3))
    assert r["target_to_actual_coverage"].startswith("NOT_MEASURABLE")


def test_mesh_target_keeps_bidirectional_samples_and_does_not_invent_open_volume():
    v,f=flat_mesh()
    target=TargetGeometry("mesh","independent nonhorizontal mesh","v1",parameters={"vertices_m":v,"triangles":f})
    r,raw=evaluate_3d_surface(v+[0,0,.002],f,target,actual_frame_id="workpiece_local")
    assert r["actual_to_target"]["centroid_area_weighted_p95_distance_m"] == pytest.approx(.002,abs=1e-7)
    assert r["target_to_actual"]["all_triangle_centroids"] == 2
    assert r["signed_absolute_volume_difference_m3"] is None


def test_integrity_does_not_require_separation_and_missing_sources_never_pass():
    task,data=small_recording()
    data["codes"][:]=0
    goal=GoalSpec("diagnostic_only",duration_s=8.)
    r,_=evaluate_recording(with_provenance(data),goal=goal)
    assert r["physical_integrity"]["status"] == "PASS_RECORDED_STATE_INTEGRITY"
    assert r["cutting_outcome"]["metrics"]["final_components"] == 1
    assert r["success"] is None
    r,_=evaluate_recording(data,task=task,profile="research")
    assert not r["success"] and r["physical_integrity"]["status"].startswith("NOT_MEASURABLE")
    for key in ["mass_kg","reference_stencil_gradients"]:
        bad=dict(data);bad.pop(key)
        r,_=evaluate_recording(bad,task=task,profile="research")
        assert not r["success"] and r["physical_integrity"]["status"] == "NOT_MEASURABLE"


def test_cut_goal_ignores_planarity_and_all_legacy_numbers_stay_exact():
    task,data=small_recording()
    frozen=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    legacy=task.evaluate_oracle(data)
    before={k:np.asarray(v).copy() for k,v in data.items()}
    r,_=evaluate_recording(with_provenance(data),task=task,profile="research")
    assert r["surface_quality"]["status"] == "NOT_APPLICABLE"
    assert r["legacy_precision_benchmark"] == legacy
    assert r["success"] and r["profile_assessment"]["goal_status"] == "PASS"
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == frozen
    for key in before:
        np.testing.assert_array_equal(data[key],before[key])
    json.dumps(r,allow_nan=False)


def test_uncut_partial_and_overcut_removal_are_continuous_and_physical():
    task,data=small_recording("TRIM-01")
    ref=data["reference_positions"]
    above=ref[:,2]>.008+1e-7
    r,_=evaluate_recording(with_provenance(data),task=task)
    full=r["cutting_outcome"]["removal"]
    assert full["removed_reference_above_target_mass_fraction"] == pytest.approx(1)
    assert len(full["per_bump"]) == 1
    partial={**data,"codes":data["codes"].copy()}
    partial["codes"][1:,above & (ref[:,0]>0)]=0
    rp,_=evaluate_recording(with_provenance(partial),task=task)
    assert rp["cutting_outcome"]["removal"]["removed_reference_above_target_mass_fraction"] < 1
    uncut={**data,"codes":np.zeros_like(data["codes"])}
    ru,_=evaluate_recording(with_provenance(uncut),task=task)
    assert ru["cutting_outcome"]["removal"]["removed_reference_above_target_mass_fraction"] == 0
    overcut={**data,"codes":data["codes"].copy()}
    overcut["codes"][1:]=np.where(ref[:,2] < .004,1,2)
    ro,_=evaluate_recording(with_provenance(overcut),task=task)
    assert ro["cutting_outcome"]["removal"]["protected_reference_material_detached_mass_kg"] > 0


def test_nan_mass_loss_rewritten_history_and_unknown_thresholds_cannot_pass():
    task,data=small_recording()
    for name in ("nan","mass_loss","rewrite"):
        bad={k:np.asarray(v).copy() for k,v in data.items()}
        if name=="nan":bad["positions"][-1,0,0]=np.nan
        if name=="mass_loss":
            bad["frame_mass_kg"]=np.tile(bad["mass_kg"],(3,1));bad["frame_mass_kg"][-1]*=.9
        if name=="rewrite":bad["codes"][-1]=0
        r,_=evaluate_recording(with_provenance(bad),task=task,profile="research")
        assert r["physical_integrity"]["status"] == "FAIL" and not r["success"]
    goal=replace(goal_for_task(task),thresholds={"surface_quality.missing_metric":{"max":.001}})
    r,_=evaluate_recording(with_provenance(data),task=task,goal=goal,profile="research")
    assert r["profile_assessment"]["goal_status"] == "NOT_MEASURABLE" and not r["success"]


def test_empty_region_is_not_a_pass_or_a_zero_error():
    v,f=flat_mesh()
    r,_=columns(v,f,regions=(RegionSpec("out_of_bounds","box",{"bounds_xy_m":[[1,1],[2,2]]}),))
    region=r["analytic_target"]["out_of_bounds"]
    assert region["status"] == "NOT_MEASURABLE" and region["paired_p95_absolute_error_m"] is None


def test_capsule_domain_keeps_outside_rays_and_does_not_call_them_missing_material():
    v,f=flat_mesh()
    region=RegionSpec("initial_capsule","capsule",{"length_m":.02,"width_m":.01})
    r,raw=columns(v,f,domain=region)
    assert r["missing_classification"]["outside_predeclared_object_domain_columns"] > 0
    assert r["analytic_target"]["entire_region"]["area_m2"] < .02*.02
    assert len(raw["actual_height_m"]) == 400
    assert r["analytic_target"]["entire_region"]["missing_actual_columns"] == 0


def test_unsupported_surface_threshold_and_custom_region_never_borrow_legacy_success():
    task,data=small_recording()
    goal=GoalSpec("remove_region",duration_s=8.)
    r,_=evaluate_recording(with_provenance(data),task=task,goal=goal,profile="research")
    assert not r["success"] and r["profile_assessment"]["inherited_outcome_checks"] is None


def test_task_api_exposes_profiles_while_default_oracle_contract_is_preserved():
    task,data=small_recording()
    r=task.evaluate(with_provenance(data),profile="research")
    assert r["success"] and r["profile"] == "research"
    assert r["legacy_precision_benchmark"] == task.evaluate_oracle(data)


def test_top_level_frame_mismatch_and_modified_target_cannot_inherit_benchmark_pass():
    task,data=small_recording()
    r,_=evaluate_recording({**with_provenance(data),'geometry_frame_id':'unregistered_camera_frame'},task=task,profile='research')
    assert not r['success'] and r['profile_assessment']['goal_status']=='NOT_MEASURABLE_FRAME_MISMATCH'
    goal=replace(goal_for_task(task),world_from_goal_xyzw=(.1,0,0,0,0,0,1))
    r,_=evaluate_recording(with_provenance(data),task=task,goal=goal,profile='research')
    assert not r['success'] and r['profile_assessment']['inherited_outcome_checks'] is None
    r,_=evaluate_recording(with_provenance(data),task=task,goal=goal,profile='precision')
    assert not r['success'] and r['legacy_precision_benchmark']['status']=='PASS_TASK_METRICS'


def test_precision_additional_unmeasurable_limits_cannot_be_ignored_by_legacy_pass():
    task,data=small_recording()
    goal=replace(goal_for_task(task),thresholds={'surface_quality.absent_metric':{'max':.001}})
    r,_=evaluate_recording(with_provenance(data),task=task,goal=goal,profile='precision')
    assert r['legacy_precision_benchmark']['status']=='PASS_TASK_METRICS'
    assert not r['success'] and r['profile_assessment']['thresholds']['status']=='NOT_MEASURABLE'
