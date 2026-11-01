#!/usr/bin/env python3
"""Author the preregistered native-depth plane/distance/feature matrix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robotic_cutting_sim.recording import new_output, write_json


def main():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("configs/single_view_correspondence_v1.json"))
    args = parser.parse_args()
    output = new_output(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    (output / "protocol.json").write_bytes(args.protocol.read_bytes())
    frames = []
    for case in protocol["cases"]:
        for repeat in range(case.get("repeats", 1)):
            frames.append({**case, "repeat": repeat, "time_s": len(frames)})
    path = output / "holdout_controls.usda"
    stage = Usd.Stage.CreateNew(str(path.resolve()))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(1.); stage.SetFramesPerSecond(1.)
    stage.SetStartTimeCode(0.); stage.SetEndTimeCode(len(frames) - 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    materials = []
    for index, color in enumerate([(.18, .35, .60), (.78, .72, .42), (.8, .10, .06)]):
        material = UsdShade.Material.Define(stage, f"/World/FixtureMaterial{index}")
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.8)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials.append(material)
    scene = protocol["scene"]
    nx, ny = scene["checker_tiles_xy"]
    xs = np.linspace(-scene["plane_half_width_m"], scene["plane_half_width_m"], nx + 1)
    ys = np.linspace(-scene["plane_half_height_m"], scene["plane_half_height_m"], ny + 1)
    for ix in range(nx):
        for iy in range(ny):
            xy = np.array([[xs[ix], ys[iy]], [xs[ix+1], ys[iy]], [xs[ix+1], ys[iy+1]], [xs[ix], ys[iy+1]]])
            mesh = UsdGeom.Mesh.Define(stage, f"/World/ControlPlane/tile_{ix}_{iy}")
            mesh.CreateFaceVertexCountsAttr([3, 3]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
            mesh.CreateSubdivisionSchemeAttr("none"); mesh.CreateDoubleSidedAttr(True)
            points = mesh.CreatePointsAttr()
            for frame in frames:
                z = frame["distance_m"] + frame["slope_x"] * xy[:, 0] + frame["slope_y"] * xy[:, 1]
                points.Set(np.c_[xy, z].astype(np.float32), Usd.TimeCode(frame["time_s"]))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[(ix + iy) % 2])
    box = UsdGeom.Cube.Define(stage, "/World/RedDepthStep")
    box.CreateSizeAttr(2.)
    xf = UsdGeom.Xformable(box)
    position = xf.AddTranslateOp()
    xf.AddScaleOp().Set(Gf.Vec3f(*scene["box_half_extents_m"]))
    visibility = box.CreateVisibilityAttr()
    for frame in frames:
        tc = Usd.TimeCode(frame["time_s"])
        position.Set(Gf.Vec3d(*scene["box_center_xy_m"], frame["distance_m"] - scene["box_center_z_below_plane_origin_m"]), tc)
        visibility.Set(UsdGeom.Tokens.inherited if frame.get("box") else UsdGeom.Tokens.invisible, tc)
    UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(materials[2])
    cameras = {"d455_0": ("/World/D455_0/RSD455/Camera_Pseudo_Depth", [0., 0., 0.]),
               "d455_1": ("/World/D455_1/RSD455/Camera_Pseudo_Depth", [.1, 0., 0.]),
               "global": ("/World/GlobalCamera", [0., -.15, 0.])}
    for name, (prim_path, eye) in cameras.items():
        optics = protocol['runtime_profile']
        width, height = optics.get('width',640), optics.get('height',480)
        focal = optics.get('focal_length_px',317.04312)
        camera = UsdGeom.Camera.Define(stage, prim_path)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955 * height/width)
        camera.CreateFocalLengthAttr(20.955 * focal / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
        pose = UsdGeom.Xformable(camera)
        pose.AddTranslateOp().Set(Gf.Vec3d(*eye))
        pose.AddOrientOp().Set(Gf.Quatf(0., Gf.Vec3f(1., 0., 0.)))
    stage.GetRootLayer().Save()
    write_json(output / "fixture.json", {"protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "scene": str(path.resolve()), "frames": frames, "cameras": cameras,
        "scope": "preregistered renderer-only analytic plane/foreground-box controls; not physical hardware"})
    print(json.dumps({"scene": str(path), "frames": len(frames), "protocol_sha256": protocol_hash}))


if __name__ == "__main__":
    main()
