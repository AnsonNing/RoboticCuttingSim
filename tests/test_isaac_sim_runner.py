import numpy as np
import pytest
import json

from robotic_cutting_sim.isaac_sim_runner import (
    IsaacSimCuttingConfig, LiveIsaacSimCutting, calibration_from_replicator, pose_from_wxyz, require_sim_62, validate_native_action,
)


def test_cold_workcell_wait_uses_elapsed_time_not_a_300_tick_cap(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from robotic_cutting_sim import isaac_sim_runner as module
    calls = []
    app = SimpleNamespace(update=lambda: calls.append(1))
    context = SimpleNamespace(get_stage_loading_status=lambda: (0, 0, int(len(calls) <= 350)))
    monkeypatch.setattr(module.time, "monotonic", lambda: len(calls)*.01)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    receipt = tmp_path/"load.json"
    module.wait_for_workcell_stage(app, context, receipt, timeout_s=10.)
    result = json.loads(receipt.read_text())
    assert result["status"] == "PASS_STAGE_LOADED" and result["kit_updates"] == 351


def test_workcell_wait_preserves_a_bounded_timeout_diagnostic(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from robotic_cutting_sim import isaac_sim_runner as module
    calls = []
    app = SimpleNamespace(update=lambda: calls.append(1))
    context = SimpleNamespace(get_stage_loading_status=lambda: (2, 1, 1))
    monkeypatch.setattr(module.time, "monotonic", lambda: len(calls))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    receipt = tmp_path/"timeout.json"
    with pytest.raises(TimeoutError, match="loading status"):
        module.wait_for_workcell_stage(app, context, receipt, timeout_s=2.)
    assert json.loads(receipt.read_text())["status"] == "BLOCKED_STAGE_LOAD_TIMEOUT"


def test_native_gpu_selection_supports_single_gpu_and_explicit_devices():
    assert IsaacSimCuttingConfig().gpu == 0
    assert IsaacSimCuttingConfig().device == "cuda:0"
    assert IsaacSimCuttingConfig(gpu=3).device == "cuda:3"
    assert IsaacSimCuttingConfig(gpu=1, device="cpu").device == "cpu"
    with pytest.raises(ValueError, match="nonnegative"):
        IsaacSimCuttingConfig(gpu=-1)


def test_live_runner_rejects_older_or_missing_runtime_and_no_ideal_fallback():
    with pytest.raises(RuntimeError,match="6.2.0.0"):
        require_sim_62({"isaacsim":"6.1.0.0"})
    with pytest.raises(ValueError,match="no ideal fallback"):
        IsaacSimCuttingConfig(camera_mode="ideal_rgbd")
    with pytest.raises(ValueError,match="positive"):
        IsaacSimCuttingConfig(record_every=0)


def test_native_wxyz_pose_becomes_project_xyzw_without_changing_translation():
    np.testing.assert_allclose(pose_from_wxyz([1,2,3],[.5,.5,.5,.5]),[1,2,3,.5,.5,.5,.5])
    with pytest.raises(ValueError,match="unit quaternion"):
        pose_from_wxyz([1,2,3],[2,0,0,0])


def test_runtime_projection_calibration_preserves_aspect_and_offsets():
    projection=np.eye(4)
    projection[0,0],projection[1,1]=2.,3.
    projection[2,0],projection[2,1]=.1,-.2
    parameters={"renderProductResolution":[640,480],"cameraProjection":projection.ravel(),
                "cameraViewTransform":np.eye(4).ravel(),"metersPerSceneUnit":1.}
    calibration=calibration_from_replicator(parameters)
    np.testing.assert_allclose(calibration["K"],[[640,0,288],[0,720,192],[0,0,1]])
    np.testing.assert_allclose(calibration["T_world_camera"],np.diag([1,-1,-1,1]))


def test_task_defaults_preserve_legacy_controller_and_declare_new_recipe():
    legacy=IsaacSimCuttingConfig()
    task=IsaacSimCuttingConfig(task_id="CUT-01")
    assert legacy.velocity_feedforward is False and legacy.native_noise=="on"
    assert task.velocity_feedforward is True and task.native_noise=="off"
    assert legacy.surface_profile=="legacy" and task.surface_profile=="current_volume_v1"
    assert IsaacSimCuttingConfig(task_id="TRIM-01",velocity_feedforward=False,native_noise="on").velocity_feedforward is False
    with pytest.raises(ValueError,match="task_id"):
        IsaacSimCuttingConfig(task_id="UNKNOWN")
    with pytest.raises(ValueError,match="legacy D0"):
        IsaacSimCuttingConfig(particle_spacing_m=.002)
    assert IsaacSimCuttingConfig(task_id="TRIM-01",particle_spacing_m=.002).particle_spacing_m==.002


def test_native_explicit_actions_have_separate_joint_and_world_pose_semantics():
    joints=np.arange(7.)/10
    np.testing.assert_array_equal(validate_native_action(joints,"joint"),joints)
    np.testing.assert_allclose(validate_native_action([.1,.2,.3,0,0,0,2],"cartesian"),[.1,.2,.3,0,0,0,1])
    with pytest.raises(ValueError,match="Quaternion"):
        validate_native_action(np.zeros(7),"cartesian")
    with pytest.raises(ValueError,match="seven finite"):
        validate_native_action(np.zeros(13),"joint")


def test_task_fixture_usd_uses_same_local_boxes_and_workpiece_frame():
    from types import SimpleNamespace
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    stage=Usd.Stage.CreateInMemory()
    table=UsdGeom.Cube.Define(stage,"/World/Table")
    table.AddTranslateOp().Set(Gf.Vec3d(0,0,-.035))
    UsdGeom.Cube.Define(stage,"/World/TrayFloor")
    q=Rotation.from_euler("z",.4).as_quat()
    box=SimpleNamespace(name="left_pedestal",center_m=(-.055,0,-.005),half_extents_m=(.035,.025,.005))
    env=LiveIsaacSimCutting(None,None,"unused")
    env.stage=stage
    env.task=SimpleNamespace(world_from_workpiece_xyzw=(.02,.03,0,*q),fixture_boxes_local=(box,),table_top_world_z_m=-.01)
    env._author_task_fixture()
    assert not stage.GetPrimAtPath("/World/TrayFloor").IsActive()
    cube=stage.GetPrimAtPath("/World/TaskFixture/Box_0")
    assert cube.HasAPI(UsdPhysics.CollisionAPI)
    actual=np.asarray(UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    np.testing.assert_allclose(actual[:3,3],Rotation.from_quat(q).apply(box.center_m)+[.02,.03,0],atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(actual[:3,:3],axis=0),box.half_extents_m,atol=1e-9)
