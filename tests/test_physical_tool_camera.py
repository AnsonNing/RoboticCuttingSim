"""Whole-scene CPU visual attribution controls, not native sensor acceptance."""
import importlib.util
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom


def test_physical_tool_parts_are_occluders_and_never_become_material_labels(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/audit_wrist_camera_rig.py"
    spec = importlib.util.spec_from_file_location("physical_camera_audit_test", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    stage = Usd.Stage.CreateNew(str(tmp_path / "final_live_scene.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = "/World/Tools/CuttingTool_0"
    UsdGeom.Xform.Define(stage, root)
    # Three separate boxes exercise visual IDs; this is not a physical tool or
    # a dynamic-grasp fixture and supplies no runtime/grasp acceptance evidence.
    for name, x in [("Handle", -.05), ("Shaft", 0.), ("Blade", .05)]:
        cube = UsdGeom.Cube.Define(stage, root + "/" + name)
        cube.CreateSizeAttr(.02)
        cube.AddTranslateOp().Set(Gf.Vec3d(x, 0., .3))
    mesh = UsdGeom.Mesh.Define(stage, "/World/ContinuousSpecimen")
    vertices = np.array([[-.5, -.5, .6], [.5, -.5, .6], [.5, .5, .6], [-.5, .5, .6]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    mesh.CreatePointsAttr(vertices.tolist()); mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist()); mesh.CreateDoubleSidedAttr(True)
    stage.GetRootLayer().Save()
    robot = {"body_prim_paths": np.array([root]), "body_pose": np.array([[[0, 0, 0, 0, 0, 0, 1]]])}
    reference = module.SceneReference(tmp_path, robot, {}, native=True)
    reference.update(robot["body_pose"][0], vertices, faces,
                     [{"first_triangle": 0, "triangles": 2, "mechanical_component": 0}])
    depth, labels = reference.capture({"width": 80, "height": 60, "K": [[50, 0, 40], [0, 50, 30], [0, 0, 1]],
                                      "T_world_camera": np.eye(4).tolist()})
    for label in (1, -3, -4):
        mask = reference.last_tool_part_labels == label
        assert mask.any()
        # Edge rays hit side faces as well as the front face of each real box.
        assert np.all((depth[mask] >= .29 - 1e-6) & (depth[mask] <= .31 + 1e-6))
        np.testing.assert_allclose(depth[mask].min(), .29, atol=1e-6)
        assert np.all(labels[mask] == (1 if label == 1 else 0))
    assert np.any(labels == 2)
    assert module.tool_part_label("/World/Legacy/Knife") == 1
    assert module.tool_part_label("/World/Unrelated/Blade") is None
