"""Ideal, programmable RGB-D camera for ParticleSurface feasibility.

This Warp ray camera is not Isaac Sim RTX, a RealSense sensor, or modeled stereo.
Camera frame is OpenCV: +X right, +Y down, +Z forward; depth is optical Z in m.
"""
import time

import numpy as np
import warp as wp

from .recording import manifest, write_json


def calibration(eye, target=(0,0,.015), width=640, height=480, hfov_deg=87):
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target)-eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0,0,1]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack([right,down,forward])
    transform = np.eye(4); transform[:3,:3] = rotation; transform[:3,3] = eye
    focal = width / (2*np.tan(np.deg2rad(hfov_deg)/2))
    return {"width": width, "height": height, "K": [[focal,0,width/2],[0,focal,height/2],[0,0,1]],
            "T_world_camera": transform.tolist(), "convention": "OpenCV +Z forward, +Y down",
            "depth": "optical Z, meters; invalid=0", "near_m": .52, "far_m": 6.0,
            "sensor_model": "ideal pinhole, provisional D455-like HFOV only"}


@wp.kernel
def ray_camera(mesh_id: wp.uint64, eye: wp.vec3, rotation: wp.mat33,
               fx: float, fy: float, cx: float, cy: float, near: float, far: float,
               depth: wp.array2d(dtype=float), rgb: wp.array2d(dtype=wp.vec3)):
    y, x = wp.tid()
    local = wp.vec3((float(x)+.5-cx)/fx, (float(y)+.5-cy)/fy, 1.0)
    local = wp.normalize(local)
    direction = rotation * local
    query = wp.mesh_query_ray(mesh_id, eye, direction, far)
    depth[y,x] = 0.0
    rgb[y,x] = wp.vec3(.12,.15,.19)
    if query.result:
        z = query.t * local[2]
        if z >= near and z <= far:
            depth[y,x] = z
            shade = .3 + .7*wp.abs(wp.dot(query.normal, wp.normalize(wp.vec3(-.4,-.3,1.0))))
            rgb[y,x] = wp.vec3(.72,.32,.13)*shade


def capture(vertices, indices, calib, device):
    mesh = wp.Mesh(points=vertices, indices=indices)
    depth = wp.empty((calib["height"],calib["width"]), dtype=float, device=device)
    rgb = wp.empty(depth.shape, dtype=wp.vec3, device=device)
    transform = np.array(calib["T_world_camera"])
    k = np.array(calib["K"])
    wp.launch(ray_camera, dim=depth.shape,
              inputs=[mesh.id,wp.vec3(*transform[:3,3]),wp.mat33(transform[:3,:3]),
                      k[0,0],k[1,1],k[0,2],k[1,2],calib["near_m"],calib["far_m"],depth,rgb],device=device)
    return np.clip(rgb.numpy()*255,0,255).astype(np.uint8), depth.numpy()


def plane_check(device):
    # Exact plane at world Z=0, camera oblique, analytic forward-depth per pixel.
    c = calibration([0,-.5,.55])
    vertices = wp.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]],dtype=wp.vec3,device=device)
    indices = wp.array([0,1,2,0,2,3],dtype=int,device=device)
    _, depth = capture(vertices,indices,c,device)
    t = np.array(c["T_world_camera"]); k = np.array(c["K"])
    y,x = np.indices(depth.shape)
    local = np.stack([(x+.5-k[0,2])/k[0,0],(y+.5-k[1,2])/k[1,1],np.ones_like(x)],-1)
    rays = local @ t[:3,:3].T
    expected = -t[2,3]/rays[:,:,2]
    valid = (depth>0)&(expected>=c["near_m"])&(expected<=c["far_m"])
    err = np.abs(depth[valid]-expected[valid])
    if len(err) == 0 or err.max() > 1e-5:
        raise AssertionError("Ideal camera plane depth failed 10 micrometer numerical tolerance")
    return {"status":"PASS","valid_pixels":int(valid.sum()),"max_abs_error_m":float(err.max()),
            "rmse_m":float(np.sqrt(np.mean(err**2))),"tolerance_m":1e-5}


def run(args, output):
    from newton.geometry import ParticleSurface
    from PIL import Image

    wp.init()
    device = args.device
    state = np.load(args.input / "physics_state.npz")
    episode = manifest({"input":str(args.input.resolve()),"surface_voxel_size_m":args.surface_voxel_size,
                        "renderer":"custom_warp_ray_camera", "device":device}, "particle_surface_camera_probe")
    write_json(output/"manifest.json",episode)
    cameras = {"ideal_0":calibration([0,-.5,.55]), "ideal_1":calibration([.5,-.4,.25]),
               "global":calibration([.8,-.8,.8],width=640,height=480,hfov_deg=65)}
    write_json(output/"calibration.json",cameras)
    qa = plane_check(device)
    with wp.ScopedDevice(device):
        surface = ParticleSurface(voxel_size=args.surface_voxel_size, smooth_lambda=0.0,
                                  mesh_smooth_iterations=0, device=device)
        spacing = float(state["mass_kg"][0]/1500)**(1/3)
        radii = wp.full(state["positions"].shape[1], value=spacing/2, dtype=float,device=device)
        frame_ids = np.unique(np.linspace(0,len(state["positions"])-1,args.samples,dtype=int))
        rows = []; previous = {}; dynamic_pixels = {name:0 for name in cameras}
        mesh_path = output/"surface"; mesh_path.mkdir()
        rgbd_path = output/"rgbd"; rgbd_path.mkdir()
        for i in frame_ids:
            tick = time.perf_counter()
            positions = wp.array(state["positions"][i],dtype=wp.vec3,device=device)
            vertices, indices, normals = surface.extract(positions,radii)
            if vertices is None or indices is None or len(indices)==0:
                raise RuntimeError(f"Empty ParticleSurface at frame {i}")
            verts, tris, norms = vertices.numpy(), indices.numpy(), normals.numpy()
            if not np.isfinite(verts).all() or len(tris)%3 or tris.min()<0 or tris.max()>=len(verts):
                raise RuntimeError("Invalid extracted topology")
            np.savez_compressed(mesh_path/f"{i:04d}.npz", vertices_m=verts, triangles=tris.reshape(-1,3), normals=norms,
                                physics_timestamp_s=state["time_s"][i])
            extraction_s = time.perf_counter()-tick
            tick = time.perf_counter()
            for name, calib in cameras.items():
                rgb,depth = capture(vertices,indices,calib,device)
                valid = depth>0
                if name in previous:
                    common = valid & (previous[name]>0)
                    dynamic_pixels[name] += int(np.sum(common & (np.abs(depth-previous[name])>1e-5)))
                previous[name] = depth.copy()
                np.savez_compressed(rgbd_path/f"{name}_{i:04d}.npz", rgb=rgb,depth_m=depth,valid_mask=valid,
                                    physics_timestamp_s=state["time_s"][i], surface_timestamp_s=state["time_s"][i])
                Image.fromarray(rgb).save(rgbd_path/f"{name}_{i:04d}.png")
            rows.append({"frame":int(i),"time_s":float(state["time_s"][i]),"vertices":len(verts),
                         "triangles":len(tris)//3,"extraction_wall_s":extraction_s,"three_camera_wall_s":time.perf_counter()-tick})
    result = {"status":"PASS" if min(dynamic_pixels.values())>0 else "FAIL",
              "scope":"ParticleSurface mesh to programmable ideal RGB-D, recorded MPM states",
              "renderer":"custom Warp mesh ray tracing", "plane_qa":qa,"frames":rows,
              "changed_depth_pixels_10um":dynamic_pixels,"isaac_rtx_d455":"BLOCKED",
              "modeled_stereo_depth":"NOT_RUN", "true_cut_depth":"NOT_RUN", "vision_proven":False,
              "limitations":["Offline replay feasibility, not live Kit synchronization",
                              "Ideal shading and pinhole rays, not RealSense noise or stereo",
                              "Surface smoothing may bridge a physical gap; no cut validated"]}
    write_json(output/"result.json",result)
    return result
