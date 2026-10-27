import numpy as np
import pytest

from robotic_cutting_sim.sensor_contract import CameraMode, mode_contract, validate_frame
from robotic_cutting_sim.usd_surface_bridge import DynamicSurfaceBridge, stage_triangles, usd_camera_calibration
from robotic_cutting_sim.rtx_outputs import decode_camera_params


def test_modes_do_not_label_ideal_frames_as_d455():
    assert mode_contract(CameraMode.IDEAL_RGBD)["depth_model"] == "ideal pinhole optical Z"
    assert "SingleView" in mode_contract("simulated_d455")["depth_model"]
    with pytest.raises(ValueError):
        mode_contract("d455_like")


def test_frame_rejects_stale_surface_and_invalid_depth():
    rgb = np.zeros((2, 3, 3), np.uint8)
    depth = np.ones((2, 3), np.float32)
    with pytest.raises(ValueError, match="timestamps"):
        validate_frame(rgb, depth, 2.0, 1.9)
    depth[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_frame(rgb, depth, 2.0, 2.0)


def test_topology_change_replaces_indices_extent_and_normals():
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    bridge = DynamicSurfaceBridge(stage)
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    b = np.vstack([a, [[2, 0, 0], [3, 0, 0], [2, 1, 0]]]).astype(np.float32)
    first = bridge.update(a, np.array([[0, 1, 2]]), 0.0)
    bridge.mesh.CreateNormalsAttr().Set([(0, 0, 1)] * 3)
    second = bridge.update(b, np.array([[0, 1, 2], [3, 4, 5]]), .1)
    assert first["topology_sha256"] != second["topology_sha256"]
    assert list(bridge.mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 3, 4, 5]
    assert np.array_equal(np.asarray(bridge.mesh.GetExtentAttr().Get()), [[0, 0, 0], [3, 1, 0]])
    assert bridge.mesh.GetNormalsAttr().Get() is None
    with pytest.raises(ValueError, match="strictly"):
        bridge.update(b, np.array([[0, 1, 2]]), .1)
    with pytest.raises(ValueError, match="stale"):
        bridge.update(b, np.array([[0, 1, 2]]), .2, surface_timestamp_s=.1)
    with pytest.raises(ValueError, match="range"):
        bridge.update(a, np.array([[0, 1, 3]]), .2)
    assert len(bridge.mesh.GetPointsAttr().Get()) == 6  # Failed writes preserve last frame.


def test_usd_camera_projection_and_transforms():
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateHorizontalApertureAttr(20)
    camera.CreateVerticalApertureAttr(10)
    camera.CreateFocalLengthAttr(10)
    camera.CreateHorizontalApertureOffsetAttr(1)
    camera.CreateVerticalApertureOffsetAttr(1)
    UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(0, 0, 100))
    c = usd_camera_calibration(stage, "/World/Camera", 200, 100)
    assert np.allclose(c["K"], [[100, 0, 90], [0, 100, 60], [0, 0, 1]])
    assert np.allclose(c["T_world_camera"], [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]])


def test_stage_geometry_obeys_visibility_and_si_units():
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0, 0, 100))
    hidden = UsdGeom.Cube.Define(stage, "/World/Hidden")
    hidden.CreateVisibilityAttr("invisible")
    vertices, faces, colors, receipt = stage_triangles(stage)
    assert vertices.shape == (8, 3) and faces.shape == (12, 3)
    assert np.allclose(vertices.min(0), [-.01, -.01, .99])
    assert [r["prim_path"] for r in receipt["prims"]] == ["/World/Cube"]


def test_runtime_camera_buffer_checks_units_and_usd_cv_pose():
    buffer = bytearray(260)
    matrix = np.eye(4, dtype="<f4")
    matrix[3, 2] = -100  # Row-vector world-to-camera: camera world Z=100 scene units.
    buffer[:64] = matrix.tobytes()
    buffer[64:128] = np.eye(4, dtype="<f4").tobytes()
    buffer[128:132] = np.array([.01], "<f4").tobytes()
    buffer[132:140] = np.array([640, 480], "<u4").tobytes()
    result = decode_camera_params(np.frombuffer(buffer, dtype=np.uint8))
    assert np.allclose(result["T_world_camera"], [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]])
    assert result["width"] == 640
    with pytest.raises(ValueError, match="260"):
        decode_camera_params(np.zeros(256, np.uint8))
