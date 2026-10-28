import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig
from robotic_cutting_sim.native_camera_geometry import camera_physics_receipt, measured_camera_extrinsics
from robotic_cutting_sim.scene_frames import pose_matrix


def rig_inputs():
    hands=np.array([[.1,.2,.3,0.,0.,0.,1.],[-.2,.4,.6,0.,0.,0.,1.]])
    mount=np.eye(4);mount[:3,3]=[.025,-.02,.01]
    mount[:3,:3]=Rotation.from_euler("y",.3).as_matrix()
    specs={f"wrist_d405_{arm}":{"parent_hand_index":arm,"T_hand_optical_cv":mount,
            "sensor_model":"ideal_d405_prototype"} for arm in (0,1)}
    specs["global_d455"]={"parent_hand_index":None,"sensor_model":"native_single_view_D455"}
    fixed={"global_d455":np.eye(4)}
    usd={name:pose_matrix(hands[s["parent_hand_index"]])@mount for name,s in specs.items() if s["parent_hand_index"] is not None}
    usd.update(fixed)
    return hands,mount,specs,fixed,usd


def test_two_actual_hands_move_both_optical_frames_and_keep_fixed_mounts():
    hands,mount,specs,fixed,usd=rig_inputs()
    first=measured_camera_extrinsics(specs,hands,usd,fixed,0.,.006)
    hands[0,:3]+=[.01,-.02,.03]
    hands[1,3:]=Rotation.from_euler("z",.2).as_quat()
    for arm in (0,1):
        usd[f"wrist_d405_{arm}"]=pose_matrix(hands[arm])@mount
    second=measured_camera_extrinsics(specs,hands,usd,fixed,.02,.026)
    for name in specs:
        row=second["cameras"][name]
        np.testing.assert_allclose(np.asarray(row["T_camera_world"])@row["T_world_camera"],np.eye(4),atol=1e-12)
        if name.startswith("wrist"):
            assert not np.allclose(first["cameras"][name]["T_world_camera"],row["T_world_camera"])
            np.testing.assert_allclose(row["T_hand_camera"],mount)
            np.testing.assert_allclose(np.asarray(row["T_world_hand"])@row["T_hand_camera"],row["T_world_camera"])
    np.testing.assert_array_equal(first["cameras"]["global_d455"]["T_world_camera"],second["cameras"]["global_d455"]["T_world_camera"])
    assert second["timestamp_s"]==.02


def test_frozen_camera_world_transform_is_rejected_after_actual_hand_moves():
    hands,mount,specs,fixed,usd=rig_inputs()
    hands[1,0]+=.01
    with pytest.raises(RuntimeError,match="wrist_d405_1"):
        measured_camera_extrinsics(specs,hands,usd,fixed,.02,.026)


def test_fixed_global_drift_and_improper_optical_transform_are_rejected():
    hands,mount,specs,fixed,usd=rig_inputs()
    usd["global_d455"]=np.eye(4);usd["global_d455"][0,3]=.001
    with pytest.raises(RuntimeError,match="global_d455"):
        measured_camera_extrinsics(specs,hands,usd,fixed,0.,.006)
    usd["global_d455"]=np.diag([1.,-1.,1.,1.])
    with pytest.raises(ValueError,match="proper rigid"):
        measured_camera_extrinsics(specs,hands,usd,fixed,0.,.006)


def test_new_layout_requires_explicit_native_global_profile_and_keeps_legacy_default():
    assert IsaacSimCuttingConfig().camera_layout=="legacy_fixed_d455"
    with pytest.raises(ValueError,match="native_single_view"):
        IsaacSimCuttingConfig(camera_layout="wrist_d405_global_d455")
    with pytest.raises(ValueError,match="640x480"):
        IsaacSimCuttingConfig(camera_layout="wrist_d405_global_d455",camera_mode="native_single_view",width=1280)
    config=IsaacSimCuttingConfig(camera_layout="wrist_d405_global_d455",camera_mode="native_single_view")
    assert not config.observation_motion_probe


def test_camera_hierarchy_excludes_real_hand_body_but_rejects_camera_contacts():
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.CreateInMemory()
    hand_paths=["/World/Hand0","/World/Hand1"]
    specs={}
    for arm,path in enumerate(hand_paths):
        hand=UsdGeom.Xform.Define(stage,path)
        UsdPhysics.RigidBodyAPI.Apply(hand.GetPrim())
        camera=UsdGeom.Camera.Define(stage,path+"/D405/Optical")
        specs[f"wrist_d405_{arm}"]={"parent_hand_index":arm,"optical_prim_path":str(camera.GetPath())}
    global_body=UsdGeom.Xform.Define(stage,"/World/GlobalD455")
    body=UsdPhysics.RigidBodyAPI.Apply(global_body.GetPrim())
    body.CreateRigidBodyEnabledAttr(False)
    UsdGeom.Camera.Define(stage,"/World/GlobalD455/Optical")
    specs["global_d455"]={"parent_hand_index":None,"optical_prim_path":"/World/GlobalD455/Optical"}
    assert camera_physics_receipt(stage,specs,hand_paths)["enabled_camera_rigid_or_contact_prims"]==[]
    collider=UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/Hand1/D405"))
    with pytest.raises(RuntimeError,match="CollisionAPI"):
        camera_physics_receipt(stage,specs,hand_paths)
    collider.CreateCollisionEnabledAttr(False)
    body.GetRigidBodyEnabledAttr().Set(True)
    with pytest.raises(RuntimeError,match="RigidBodyAPI"):
        camera_physics_receipt(stage,specs,hand_paths)
