#!/usr/bin/env python3
"""Export recorded ParticleSurface snapshots and blade poses into one USD replay.

This file replays measured solver state; it does not itself run cutting physics.
Usage: export_replay.py SCENE.usda PHYSICS_RUN CAMERA_RUN OUTPUT.usda
"""
import sys
from pathlib import Path

import numpy as np
from pxr import Gf,Usd,UsdGeom,Vt

scene,physics,camera,output=map(Path,sys.argv[1:])
if output.exists():
    raise SystemExit("Refusing to overwrite replay")
stage=Usd.Stage.Open(str(scene)); stage.GetRootLayer().Export(str(output))
stage=Usd.Stage.Open(str(output)); stage.SetTimeCodesPerSecond(60); stage.SetFramesPerSecond(60)
mesh=UsdGeom.Mesh.Get(stage,"/World/ContinuousSpecimen")
mesh.GetPointsAttr().Clear(); mesh.GetFaceVertexCountsAttr().Clear(); mesh.GetFaceVertexIndicesAttr().Clear()
state=np.load(physics/"physics_state.npz")
times=state["time_s"]; tool=state["tool_pose_xyzw"]
if len(tool)!=len(times):
    raise SystemExit("Pre-fix tool timeline not eligible for replay export")
blade=UsdGeom.Xformable(stage.GetPrimAtPath("/World/KinematicBlade"))
translate=next(op for op in blade.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate)
for t,pose in zip(times,tool):
    translate.Set(Gf.Vec3d(*pose[:3]),float(t)*60)
for path in sorted((camera/"surface").glob("*.npz")):
    data=np.load(path); t=float(data["physics_timestamp_s"])*60
    vertices=data["vertices_m"]; faces=data["triangles"]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices),t)
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.ravel()),t)
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)),t)
    mesh.CreateExtentAttr().Set([Gf.Vec3f(*vertices.min(0)),Gf.Vec3f(*vertices.max(0))],t)
stage.SetStartTimeCode(0); stage.SetEndTimeCode(float(times[-1])*60)
metadata=stage.GetRootLayer().customLayerData
metadata.update(status="RECORDED_MPM_SURFACE_REPLAY",physics_run=str(physics),camera_run=str(camera),
                warning="Recorded deformation, no cutting or live simulation acceptance")
stage.GetRootLayer().customLayerData=metadata
stage.GetRootLayer().Save()
print(output)
