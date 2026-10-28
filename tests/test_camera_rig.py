"""Coordinate, lens and physics-neutral composition contracts for wrist rigs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.camera_rig import (
    CONFIG_PATH, LEGACY_LAYOUT, SECONDARY_OBSERVATION_JOINTS, author_camera_rig, make_rig, pose_matrix,
)


def test_rigid_wrist_mounts_follow_measured_hands_and_global_is_fixed():
    initial = np.array([[-.045, 0., .24, 1., 0., 0., 0.], [.20, .18, .34, 1., 0., 0., 0.]])
    rig = make_rig(initial_hand_poses_world_xyzw=initial)
    reference = rig.calibrations(initial)
    for t in (0., .2, .7, 1.):
        poses = initial.copy()
        poses[0, :3] += [.03*t, -.02*t, .01*t]
        poses[1, :3] += [-.01*t, .02*t, .015*t]
        poses[0, 3:] = (Rotation.from_rotvec([.02*t, -.03*t, .05*t])*Rotation.from_quat(initial[0, 3:])).as_quat()
        poses[1, 3:] = (Rotation.from_rotvec([-.04*t, .01*t, .02*t])*Rotation.from_quat(initial[1, 3:])).as_quat()
        calibs = rig.calibrations(poses)
        for index in (0, 1):
            spec = calibs[f"wrist_d405_{index}"]
            np.testing.assert_allclose(np.linalg.inv(pose_matrix(poses[index])) @ spec["T_world_camera"],
                                       spec["T_hand_optical_cv"], atol=1e-14)
        np.testing.assert_array_equal(calibs["global_d455"]["T_world_camera"], reference["global_d455"]["T_world_camera"])
    assert len(SECONDARY_OBSERVATION_JOINTS) == 7


def test_prototype_fov_and_logical_depth_range_are_explicit():
    rig = make_rig()
    for index in (0, 1):
        spec = rig.specs[f"wrist_d405_{index}"]
        k = np.asarray(spec["K"])
        fov = np.rad2deg(2*np.arctan(np.array([spec["width"], spec["height"]])/(2*k.diagonal()[:2])))
        np.testing.assert_allclose(fov, spec["effective_pinhole_fov_deg"], atol=1e-12)
        assert spec["width"] == 848 and spec["height"] == 480
        assert k[0, 0] == k[1, 1]
        assert spec["sensor_model"] == "ideal_d405_prototype"
        assert not spec["hardware_calibration_validated"] and not spec["d405_sensor_model_validated"]
        assert (spec["near_m"], spec["far_m"]) == (.07, .50)
        assert spec["render_near_m"] < .01 and spec["render_far_m"] >= 20
        housing = spec["visual_obbs"][0]
        assert housing["dimensions_m"] == [.042, .042, .023]
        assert spec["physical_dimensions_m"] == {"width": .042, "height": .042, "depth": .023}
        center_z = np.asarray(housing["T_optical_cv_obb"])[2, 3]
        assert center_z + housing["dimensions_m"][2]/2 < 0
    assert CONFIG_PATH.name == "wrist_camera_rig_v2.json"
    np.testing.assert_allclose(np.asarray(rig.specs["global_d455"]["T_world_camera"])[:3, 3], [1.1, -1.2, 1.05])


@pytest.fixture
def authored_rig(tmp_path):
    pxr = pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdRender
    asset_path = tmp_path/"official_asset_shape_fixture.usda"
    source = Usd.Stage.CreateNew(str(asset_path))
    UsdGeom.SetStageMetersPerUnit(source, 1.)
    UsdGeom.Xform.Define(source, "/Root")
    body = UsdGeom.Xform.Define(source, "/Root/RSD455").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    UsdGeom.Cube.Define(source, "/Root/RSD455/HousingGeometry").CreateSizeAttr(.1)
    for name in ("Camera_Pseudo_Depth", "Camera_OmniVision_OV9782_Color"):
        camera = UsdGeom.Camera.Define(source, "/Root/RSD455/"+name)
        camera.CreateHorizontalApertureAttr(3.896)
        camera.CreateVerticalApertureAttr(2.453)
        camera.CreateFocalLengthAttr(1.93)
    UsdRender.Product.Define(source, "/Root/RSD455/UnsafeSourceProduct")
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, "/World")
    initial = np.array([[-.045, 0., .24, 1., 0., 0., 0.], [.20, .18, .34, 1., 0., 0., 0.]])
    for index in (0, 1):
        hand = UsdGeom.Xform.Define(stage, f"/World/Hand_{index}")
        hand.AddTransformOp().Set(Gf.Matrix4d(pose_matrix(initial[index]).T.tolist()))
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Hand_0"))
    for path in ("/World/D455_0", "/World/D455_1", "/World/GlobalCamera"):
        UsdGeom.Xform.Define(stage, path)
    specs = author_camera_rig(stage, ["/World/Hand_0", "/World/Hand_1"], asset_path,
                              initial_hand_poses_world_xyzw=initial)
    return stage, specs, initial, asset_path


def test_usd_optical_frames_match_measured_composition_and_no_new_physics(authored_rig):
    from pxr import Usd, UsdGeom
    from robotic_cutting_sim.usd_surface_bridge import usd_camera_calibration
    stage, specs, initial, _ = authored_rig
    expected = make_rig(initial_hand_poses_world_xyzw=initial).calibrations(initial)
    for name, spec in specs.items():
        actual = usd_camera_calibration(stage, spec["depth_camera_path"], spec["width"], spec["height"])
        np.testing.assert_allclose(actual["T_world_camera"], expected[name]["T_world_camera"], atol=1e-12)
        np.testing.assert_allclose(actual["K"], spec["K"], rtol=2e-7)
        assert actual["near_m"] == pytest.approx(.001)
        assert actual["far_m"] == 20
        color = usd_camera_calibration(stage, spec["color_camera_path"], spec["width"], spec["height"])
        if name.startswith("wrist"):
            np.testing.assert_allclose(color["T_world_camera"], actual["T_world_camera"], atol=1e-12)
        root = stage.GetPrimAtPath(spec["prim_path"])
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            assert not any("Physics" in a or "Physx" in a for a in prim.GetAppliedSchemas())
    assert "PhysicsRigidBodyAPI" in stage.GetPrimAtPath("/World/Hand_0").GetAppliedSchemas()
    assert not stage.GetPrimAtPath("/World/WristCameraRig/GlobalD455/RSD455/UnsafeSourceProduct").IsActive()
    for path in ("/World/D455_0", "/World/D455_1", "/World/GlobalCamera"):
        assert not stage.GetPrimAtPath(path).IsActive()


def test_legacy_layout_does_not_author_or_deactivate_existing_cameras(authored_rig):
    stage, _, initial, asset = authored_rig
    prim = stage.GetPrimAtPath("/World/D455_0")
    prim.SetActive(True)
    before = stage.GetRootLayer().ExportToString()
    result = author_camera_rig(stage, ["/World/Hand_0", "/World/Hand_1"], asset,
                              camera_layout=LEGACY_LAYOUT, initial_hand_poses_world_xyzw=initial)
    assert result == {}
    assert stage.GetRootLayer().ExportToString() == before


def test_mounts_are_identical_across_reset_hand_poses_and_stereo_is_reference_only(authored_rig):
    from pxr import UsdGeom
    stage, specs, initial, _ = authored_rig
    moved = initial.copy()
    moved[:, :3] += [.1, -.04, .05]
    moved[:, 3:] = Rotation.from_euler("xyz", [[.2, -.1, .1], [.3, .2, -.2]]).as_quat()
    before = make_rig(initial_hand_poses_world_xyzw=initial)
    after = make_rig(initial_hand_poses_world_xyzw=moved)
    for index in (0, 1):
        name = f"wrist_d405_{index}"
        np.testing.assert_array_equal(before.specs[name]["T_hand_optical_cv"], after.specs[name]["T_hand_optical_cv"])
        spec = specs[name]
        right = stage.GetPrimAtPath(spec["right_optical_reference_path"])
        assert right.IsA(UsdGeom.Xform) and not right.IsA(UsdGeom.Camera)
        local = np.asarray(UsdGeom.Xformable(right).GetLocalTransformation()).T
        np.testing.assert_allclose(local[:3, 3], [.018, 0, 0], atol=1e-15)


def test_triangle_obb_separating_axes_control():
    path = Path(__file__).resolve().parents[1]/"scripts/audit_camera_mount_geometry.py"
    spec = importlib.util.spec_from_file_location("camera_mount_geometry_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    triangles = np.array([[[-2., 0., 0.], [2., 0., 0.], [0., 2., 0.]],
                          [[2., 2., 2.], [3., 2., 2.], [2., 3., 2.]],
                          [[.75, .75, 0.], [2., 0., 0.], [0., 2., 0.]]])
    # Third triangle's AABB overlaps, while its diagonal separating axis does not.
    np.testing.assert_array_equal(module.triangle_box_overlap(triangles, np.eye(4), [1., 1., 1.]),
                                  [True, False, False])


def test_invalid_measured_pose_rejected():
    with pytest.raises(ValueError):
        make_rig(initial_hand_poses_world_xyzw=np.zeros((2, 7)))
    with pytest.raises(ValueError):
        make_rig().calibrations(np.zeros((1, 7)))
