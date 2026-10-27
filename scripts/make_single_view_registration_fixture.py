#!/usr/bin/env python3
"""Author a static analytic slanted-plane control for native SingleView AOVs."""
import argparse
from pathlib import Path

import numpy as np

from robotic_cutting_sim.recording import new_output, write_json


def main():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    path = output / "slanted_plane.usda"
    stage = Usd.Stage.CreateNew(str(path.resolve()))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(1.)
    stage.SetFramesPerSecond(1.)
    stage.SetStartTimeCode(0.)
    stage.SetEndTimeCode(2.)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    materials = []
    for index, color in enumerate([(.18, .35, .60), (.78, .72, .42)]):
        material = UsdShade.Material.Define(stage, f"/World/PlaneMaterial{index}")
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.8)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials.append(material)
    x_edges = np.linspace(-2., 2., 33)
    y_edges = np.linspace(-1.5, 1.5, 25)
    for ix in range(32):
        for iy in range(24):
            xy = np.array([[x_edges[ix], y_edges[iy]], [x_edges[ix+1], y_edges[iy]],
                           [x_edges[ix+1], y_edges[iy+1]], [x_edges[ix], y_edges[iy+1]]])
            vertices = np.c_[xy, 1. + .25 * xy[:, 0] + .10 * xy[:, 1]]
            mesh = UsdGeom.Mesh.Define(stage, f"/World/SlantedPlane/tile_{ix}_{iy}")
            mesh.CreatePointsAttr(vertices.astype(np.float32))
            mesh.CreateFaceVertexCountsAttr([3, 3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[(ix + iy) % 2])
    cameras = {"d455_0": ("/World/D455_0/RSD455/Camera_Pseudo_Depth", [0., 0., 0.]),
               "d455_1": ("/World/D455_1/RSD455/Camera_Pseudo_Depth", [.10, 0., 0.]),
               "global": ("/World/GlobalCamera", [0., -.15, 0.])}
    for name, (prim_path, position) in cameras.items():
        camera = UsdGeom.Camera.Define(stage, prim_path)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955 * .75)
        camera.CreateFocalLengthAttr(20.955 * 317.04312 / 640.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
        xf = UsdGeom.Xformable(camera)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddOrientOp().Set(Gf.Quatf(0., Gf.Vec3f(1., 0., 0.)))
    stage.GetRootLayer().Save()
    write_json(output / "fixture.json", {"status": "AUTHORED_CONTROL_FIXTURE",
        "scene": str(path.resolve()), "plane_world_equation": {"normal": [-.25, -.10, 1.], "offset": -1.},
        "depth_equation": "n dot (camera_origin + optical_Z * world_ray) + offset = 0",
        "cameras": cameras, "source_times_s": [0., 1., 2.],
        "scope": "analytic renderer control, no physics or physical D455 calibration"})
    print(path)


if __name__ == "__main__":
    main()
