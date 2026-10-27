#!/usr/bin/env python3
"""CPU nearest-visual-mesh visibility/FOV diagnostic at actual camera calibrations.

This is an independent geometry diagnostic, not RTX semantic segmentation or a
substitute for live camera acceptance. Guide-purpose collision geometry is
excluded, as it is from the ordinary renderer.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.usd_surface_bridge import stage_triangles


@wp.kernel
def visibility_rays(mesh_id: wp.uint64, camera_pose: wp.mat44, k: wp.mat33,
                    ids: wp.array(dtype=int), depth: wp.array2d(dtype=float), labels: wp.array2d(dtype=int)):
    y, x = wp.tid()
    local = wp.vec3((float(x)+.5-k[0, 2])/k[0, 0], (float(y)+.5-k[1, 2])/k[1, 1], 1.)
    rotation = wp.mat33(camera_pose[0,0],camera_pose[0,1],camera_pose[0,2],
        camera_pose[1,0],camera_pose[1,1],camera_pose[1,2],camera_pose[2,0],camera_pose[2,1],camera_pose[2,2])
    eye = wp.vec3(camera_pose[0,3],camera_pose[1,3],camera_pose[2,3])
    query = wp.mesh_query_ray(mesh_id, eye, rotation * wp.normalize(local), 20.)
    labels[y,x] = -1
    depth[y,x] = 0.
    if query.result:
        labels[y,x] = ids[query.face]
        depth[y,x] = query.t / wp.length(local)


def main():
    from pxr import Usd, UsdGeom
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("runtime_calibration", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--times", type=float, nargs="+", default=[0., 4., 8.])
    args = parser.parse_args()
    output = new_output(args.output_dir)
    stage = Usd.Stage.Open(str(args.scene))
    cameras = json.loads(args.runtime_calibration.read_text())
    # Session-only edits never modify source USD assets.
    stage.SetEditTarget(stage.GetSessionLayer())
    excluded = []
    for prim in stage.Traverse():
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputePurpose() not in ("default", "render"):
            imageable.MakeInvisible()
            excluded.append(str(prim.GetPath()))
    names = ["other", "franka_0", "franka_1", "knife", "material", "table", "fixture"]
    records = []
    wp.init()
    for time_s in args.times:
        v, f, _, meta = stage_triangles(stage, time_s * stage.GetTimeCodesPerSecond())
        ids = np.zeros(len(f), np.int32)
        for row in meta["prims"]:
            path = row["prim_path"]
            name = ("material" if "ContinuousSpecimen" in path else "knife" if "Blade" in path or "Knife" in path
                else "franka_0" if path.startswith("/World/Franka_0/") else "franka_1" if path.startswith("/World/Franka_1/")
                else "table" if path == "/World/Table" else "fixture" if "Fixture" in path else "other")
            ids[row["first_triangle"]:row["first_triangle"]+row["triangles"]] = names.index(name)
        mesh = wp.Mesh(wp.array(v,dtype=wp.vec3,device="cpu"),wp.array(f.ravel(),dtype=int,device="cpu"))
        for camera_name, camera in cameras.items():
            shape = (camera["height"], camera["width"])
            labels = wp.empty(shape,dtype=int,device="cpu")
            depth = wp.empty(shape,dtype=float,device="cpu")
            wp.launch(visibility_rays,dim=shape,inputs=[mesh.id,wp.mat44(np.asarray(camera["T_world_camera"])),
                wp.mat33(np.asarray(camera["K"])),wp.array(ids,dtype=int,device="cpu"),depth,labels],device="cpu")
            image, z = labels.numpy(), depth.numpy()
            k = np.asarray(camera["K"])
            row = {"time_s":time_s,"camera":camera_name,"visible_pixels": {name:int(np.sum(image == i)) for i,name in enumerate(names)},
                "hfov_deg":float(np.rad2deg(2*np.arctan(shape[1]/(2*k[0,0])))),
                "vfov_deg":float(np.rad2deg(2*np.arctan(shape[0]/(2*k[1,1])))),
                "material_optical_z_range_m": [float(z[image == names.index("material")].min()),float(z[image == names.index("material")].max())]
                    if np.any(image == names.index("material")) else None}
            records.append(row)
            np.savez_compressed(output / f"{camera_name}_{time_s:.3f}.npz",label=image,depth_m=z,object_names=names)
    result = {"status":"DIAGNOSTIC", "scope":"CPU closest visual triangle hit, ordinary render purposes; not actual RTX segmentation",
        "source_scene":str(args.scene.resolve()), "source_sha256":hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "runtime_calibration":str(args.runtime_calibration.resolve()),"excluded_non_render_purpose_prims":excluded,
        "unsupported_geometry":meta["unsupported_geometry"],"records":records}
    write_json(output/"result.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
