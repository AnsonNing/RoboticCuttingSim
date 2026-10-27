#!/usr/bin/env python3
"""Compare actual live Kit camera arrays with recorded measured-state geometry.

The CPU first-hit scene is an independent reference, not new RTX imagery or a
live episode. Native noisy output and deterministic RTX reference comparisons
remain separate. No valid error tail is trimmed from either comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from robotic_cutting_sim.cutting_surface import wedge_mesh
from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.usd_surface_bridge import stage_triangles
from robotic_cutting_sim.visual_geometry import non_render_purpose_paths


def task_session_geometry(stage, setup):
    """Recompose recorded task session geometry over its legacy weak USD layer."""
    from pxr import Gf, UsdGeom

    task = setup.get("task")
    if not task:
        return None
    disabled = []
    for prim in list(stage.Traverse()):
        if str(prim.GetPath()).startswith("/World/Tray"):
            disabled.append(str(prim.GetPath()))
            prim.SetActive(False)
    root = UsdGeom.Xform.Define(stage, "/World/TaskFixture")
    pose = task["world_from_workpiece_xyzw"]
    root.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
    root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
    for index, box in enumerate(task["fixture_boxes_local"]):
        cube = UsdGeom.Cube.Define(stage, f"/World/TaskFixture/Box_{index}")
        cube.CreateSizeAttr(2.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*box["center_m"]))
        cube.AddScaleOp().Set(Gf.Vec3f(*box["half_extents_m"]))
    table = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Table"))
    for op in table.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(0., 0., task["table_top_world_z_m"]-.025))
    return {"source": "recorded setup.json task; not current mutable task defaults",
            "task": task["task_id"], "disabled_legacy_tray_prims": disabled,
            "fixture_boxes_local": task["fixture_boxes_local"],
            "world_from_workpiece_xyzw": pose,
            "table_top_world_z_m": task["table_top_world_z_m"]}


@wp.kernel
def reference_rays(mesh_id: wp.uint64, camera_pose: wp.mat44, k: wp.mat33, near: float, far: float,
                   ids: wp.array(dtype=int), sides: wp.array(dtype=int), depth: wp.array2d(dtype=float), labels: wp.array2d(dtype=int)):
    y,x=wp.tid()
    local=wp.vec3((float(x)+.5-k[0,2])/k[0,0],(float(y)+.5-k[1,2])/k[1,1],1.)
    rotation=wp.mat33(camera_pose[0,0],camera_pose[0,1],camera_pose[0,2],
        camera_pose[1,0],camera_pose[1,1],camera_pose[1,2],camera_pose[2,0],camera_pose[2,1],camera_pose[2,2])
    eye=wp.vec3(camera_pose[0,3],camera_pose[1,3],camera_pose[2,3])
    norm=wp.length(local)
    start=eye+rotation*local*near
    direction=rotation*local/norm
    offset=float(0.)
    labels[y,x]=-1
    depth[y,x]=0.
    for _ in range(64):
        query=wp.mesh_query_ray(mesh_id,start+direction*offset,direction,(far-near)*norm-offset)
        if not query.result:
            break
        if sides[query.face]==0 or query.sign*float(sides[query.face])>0.:
            labels[y,x]=ids[query.face]
            depth[y,x]=near+(offset+query.t)/norm
            break
        offset+=query.t+1e-6


def comparison(depth,valid,reference,reference_valid):
    common=valid&reference_valid
    delta=depth[common].astype(float)-reference[common].astype(float)
    union=int(np.sum(valid|reference_valid))
    return {"valid_pixels":int(valid.sum()),"reference_valid_pixels":int(reference_valid.sum()),
        "common_pixels":int(common.sum()),"valid_mask_iou":float(common.sum()/union) if union else 0.,
        "mean_error_m":float(delta.mean()) if len(delta) else None,
        "rms_error_m":float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
        "p95_absolute_error_m":float(np.quantile(abs(delta),.95)) if len(delta) else None,
        "max_absolute_error_m":float(max(abs(delta))) if len(delta) else None,
        "absolute_error_gt_1mm_pixels":int(np.sum(abs(delta)>.001)),
        "error_trimming_applied":False}


def main():
    from pxr import Gf,Sdf,Usd,UsdGeom
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode",type=Path)
    p.add_argument("output_dir",type=Path)
    p.add_argument("--frames",type=int,nargs="+")
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    output=new_output(args.output_dir)
    setup=json.loads((args.episode/"setup.json").read_text())
    frames=json.loads((args.episode/"frame_manifest.json").read_text())
    cameras=json.loads((args.episode/"calibration_runtime.json").read_text())
    criteria=json.loads(Path("configs/isaac_acceptance.json").read_text())["tests"]
    camera_gate=criteria["CAM-04"]
    state=np.load(args.episode/"robot_state.npz")
    stage=Usd.Stage.Open(setup["source_scene"])
    stage.SetEditTarget(stage.GetSessionLayer())
    for arm in (0,1):
        root=stage.GetPrimAtPath(f"/World/Franka_{arm}")
        root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        root.GetVariantSets().GetVariantSet("Colliders").SetVariantSelection("convex_hulls")
    stage.GetPrimAtPath("/World/KinematicBlade").SetActive(False)
    task_geometry_receipt = task_session_geometry(stage, setup)
    knife=UsdGeom.Mesh.Define(stage,setup["knife_prim"])
    kv,kf=wedge_mesh(**setup.get("blade_dimensions_m", {}))
    knife.CreatePointsAttr(kv);knife.CreateFaceVertexCountsAttr([3]*len(kf));knife.CreateFaceVertexIndicesAttr(kf.ravel())
    xf=UsdGeom.Xformable(knife)
    xf.AddTranslateOp().Set(Gf.Vec3d(*setup["T_hand_edge_xyz_xyzw"][:3]))
    q=setup["T_hand_edge_xyz_xyzw"][3:]
    xf.AddOrientOp().Set(Gf.Quatf(q[3],Gf.Vec3f(*q[:3])))
    operations=[]
    for path in state["body_prim_paths"]:
        xf=UsdGeom.Xformable(stage.GetPrimAtPath(str(path)))
        xf.ClearXformOpOrder()
        op=xf.AddTransformOp();xf.SetResetXformStack(True)
        operations.append(op)
    excluded=non_render_purpose_paths(stage)
    specimen=UsdGeom.Mesh.Define(stage,"/World/ContinuousSpecimen")
    specimen.CreateDoubleSidedAttr(True)
    specimen.GetNormalsAttr().Block()
    base_names=["other","franka_0","franka_1","knife","substrate_or_continuous_material","table","fixture","chip"]
    hashes={}
    for path in [args.episode/"setup.json",args.episode/"robot_state.npz",args.episode/"frame_manifest.json",args.episode/"calibration_runtime.json",
                 Path(setup["source_scene"]),Path("configs/isaac_acceptance.json")]:
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    snapshots=output/"implementation";snapshots.mkdir()
    for path in (Path(__file__),Path("src/robotic_cutting_sim/visual_geometry.py"),
                 Path("src/robotic_cutting_sim/usd_surface_bridge.py"),Path("src/robotic_cutting_sim/cutting_surface.py")):
        data=path.read_bytes();(snapshots/path.name).write_bytes(data)
        hashes[str(path)]=hashlib.sha256(data).hexdigest()
    write_json(output/"input.json",{"episode":str(args.episode.resolve()),"input_sha256":hashes,"excluded_guide_prims":excluded,
        "task_session_geometry":task_geometry_receipt,"blade_dimensions_m":setup.get("blade_dimensions_m", {"length":.012,"height":.004,"half_width":.025}),
        "reference":"Actual per-link PhysX poses + actual current material mesh + measured fixed knife mount; CPU visual triangle first hits",
        "scope":"Independent recorded-live-data audit, not native semantic segmentation or new live camera capture"})
    wp.init()
    records=[]
    for row in frames:
        index,time_s=row["frame"],row["time_s"]
        if args.frames is not None and index not in args.frames:
            continue
        tick=time.perf_counter()
        state_index=int(np.argmin(abs(state["time_s"]-time_s)))
        state_delta=float(abs(state["time_s"][state_index]-time_s))
        for op,pose in zip(operations,state["body_pose"][state_index]):
            matrix=np.eye(4);matrix[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();matrix[:3,3]=pose[:3]
            op.Set(Gf.Matrix4d(matrix.T.tolist()))
        mesh_path=args.episode/"surface_debug"/f"surface_{index:04d}.npz"
        data=np.load(mesh_path)
        source_vertices,source_faces=data["vertices_m"],data["triangles"]
        surface_hash=hashlib.sha256(source_vertices.tobytes()+source_faces.tobytes()).hexdigest()
        if surface_hash!=row["surface"]["surface_sha256"]:
            raise ValueError(f"Recorded material mesh differs from captured surface receipt at{index}")
        specimen.GetPointsAttr().Set(source_vertices)
        specimen.GetFaceVertexCountsAttr().Set([3]*len(source_faces))
        specimen.GetFaceVertexIndicesAttr().Set(source_faces.ravel())
        vertices,faces,_,metadata=stage_triangles(stage,excluded_prefixes=tuple(excluded))
        ranges=row["current_component_surface"]["component_ranges"]
        names=base_names+([f"material_component_{r['mechanical_component']}" for r in ranges] if setup.get("task") else [])
        face_ids=np.zeros(len(faces),np.int32)
        face_sides=np.zeros(len(faces),np.int32)
        for prim in metadata["prims"]:
            path=prim["prim_path"]
            label=("knife" if path==setup["knife_prim"] else "franka_0" if path.startswith("/World/Franka_0/")
                else "franka_1" if path.startswith("/World/Franka_1/") else "substrate_or_continuous_material" if "ContinuousSpecimen" in path
                else "table" if path=="/World/Table" else "fixture" if "Tray" in path or path.startswith("/World/TaskFixture/") else "other")
            offset=prim["first_triangle"]
            usd_prim=stage.GetPrimAtPath(path)
            if usd_prim.IsA(UsdGeom.Mesh):
                usd_mesh=UsdGeom.Mesh(usd_prim)
                if not usd_mesh.GetDoubleSidedAttr().Get():
                    side=-1 if usd_mesh.GetOrientationAttr().Get()=="leftHanded" else 1
                    if np.linalg.det(np.array(UsdGeom.Xformable(usd_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))[:3,:3])<0:
                        side=-side
                    face_sides[offset:offset+prim["triangles"]]=side
            face_ids[offset:offset+prim["triangles"]]=names.index(label)
            if label=="substrate_or_continuous_material":
                if setup.get("task"):
                    for r in ranges:
                        start=offset+r["first_triangle"]
                        face_ids[start:start+r["triangles"]]=names.index(f"material_component_{r['mechanical_component']}")
                elif len(ranges)>1:
                    chip=min(ranges,key=lambda r:r["particles"])
                    start=offset+chip["first_triangle"]
                    face_ids[start:start+chip["triangles"]]=names.index("chip")
        mesh=wp.Mesh(wp.array(vertices,dtype=wp.vec3,device=args.device),wp.array(faces.ravel(),dtype=int,device=args.device))
        ids=wp.array(face_ids,dtype=int,device=args.device)
        sides=wp.array(face_sides,dtype=int,device=args.device)
        frame_record={"frame":index,"time_s":time_s,"robot_state_timestamp_error_s":state_delta,"cameras":{},
            "source_surface_sha256":surface_hash,"surface_file_sha256":hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            "unsupported_geometry":metadata["unsupported_geometry"]}
        for camera_name,camera in cameras.items():
            folder="global_debug" if camera_name=="global" else "observations"
            payload_path=args.episode/folder/f"{camera_name}_{index:04d}.npz"
            payload=np.load(payload_path)
            actual_depth,actual_valid=payload["depth_m"],payload["valid_mask"]
            if camera_name=="global":
                reference,reference_valid=actual_depth,actual_valid
            else:
                raw_path=args.episode/"camera_debug"/f"{camera_name}_{index:04d}.npz"
                raw=np.load(raw_path)
                reference,reference_valid=raw["rtx_reference_depth_m"],raw["rtx_reference_valid_mask"]
            shape=(camera["height"],camera["width"])
            depth=wp.empty(shape,dtype=float,device=args.device);labels=wp.empty(shape,dtype=int,device=args.device)
            clipping=UsdGeom.Camera.Get(stage,{"d455_0":"/World/D455_0/RSD455/Camera_Pseudo_Depth",
                "d455_1":"/World/D455_1/RSD455/Camera_Pseudo_Depth","global":"/World/GlobalCamera"}[camera_name]).GetClippingRangeAttr().Get()
            wp.launch(reference_rays,dim=shape,inputs=[mesh.id,wp.mat44(np.asarray(camera["T_world_camera"])),
                wp.mat33(np.asarray(camera["K"])),float(clipping[0]),min(float(clipping[1]),20.),ids,sides,depth,labels],device=args.device)
            geometry,attribution=depth.numpy(),labels.numpy()
            geometry_valid=attribution>=0
            geometry_comparison=comparison(geometry,geometry_valid,reference,reference_valid)
            native_comparison=comparison(actual_depth,actual_valid,geometry,geometry_valid)
            native_vs_rtx=comparison(actual_depth,actual_valid,reference,reference_valid)
            timestamps=[time_s,float(payload["physics_timestamp_s"]),float(payload["surface_timestamp_s"]),
                float(data["physics_timestamp_s"]),float(payload["native_physics_timestamp_s"])-setup["native_time_origin_s"]]
            consistent=reference_valid&geometry_valid&(abs(geometry-reference)<=camera_gate["mesh_depth_p95_absolute_error_m"])
            counts={name:int(np.sum(attribution==i)) for i,name in enumerate(names)}
            verified={name:int(np.sum((attribution==i)&consistent)) for i,name in enumerate(names)}
            counts["material"]=counts["substrate_or_continuous_material"]+counts["chip"]
            verified["material"]=verified["substrate_or_continuous_material"]+verified["chip"]
            component_visibility={}
            if setup.get("task"):
                for r in ranges:
                    component_name=f"material_component_{r['mechanical_component']}"
                    counts["material"]+=counts[component_name]
                    verified["material"]+=verified[component_name]
                    component_visibility[str(r["mechanical_component"])]=dict(particles=r["particles"],
                        geometric_pixels=counts[component_name],rtx_depth_consistent_pixels=verified[component_name],
                        native_valid_pixels=int(np.sum((attribution==names.index(component_name))&actual_valid)))
            camera_record={"geometry_vs_actual_rtx":geometry_comparison,"actual_native_vs_geometry":native_comparison,
                "actual_native_vs_rtx":native_vs_rtx,"timestamp_span_s":float(max(timestamps)-min(timestamps)),
                "geometric_visible_pixels":counts,"rtx_depth_consistent_attributed_pixels":verified,
                "native_valid_chip_pixels":int(np.sum((attribution==names.index("chip"))&actual_valid)),
                "input_observation_sha256":hashlib.sha256(payload_path.read_bytes()).hexdigest()}
            if setup.get("task"):
                camera_record["current_mechanical_component_visibility"]=component_visibility
                # For CUT these are two substantial pieces, not a small chip.
                # The smaller component is a visible separated component only;
                # count is zero until there is an actual current split.
                secondary=min(ranges,key=lambda r:(r["particles"],-r["mechanical_component"]))
                camera_record["native_valid_separated_component_pixels"]=(component_visibility[str(secondary["mechanical_component"])]["native_valid_pixels"] if len(ranges)>1 else 0)
            if camera_name!="global":
                camera_record["input_native_and_reference_aovs_sha256"]=hashlib.sha256(raw_path.read_bytes()).hexdigest()
            frame_record["cameras"][camera_name]=camera_record
            target=output/camera_name;target.mkdir(exist_ok=True)
            np.savez_compressed(target/f"{index:04d}.npz",geometry_depth_m=geometry,geometry_valid_mask=geometry_valid,
                geometry_object_labels=attribution,object_names=names,actual_rtx_depth_m=reference,
                native_depth_m=actual_depth,native_valid_mask=actual_valid)
        frame_record["audit_wall_s"]=time.perf_counter()-tick
        records.append(frame_record)
        write_json(output/"partial.json",records)
        print(f"Camera mesh audit{index}/{len(frames)-1}: {frame_record['audit_wall_s']:.2f}s",flush=True)
    threshold=criteria["CAM-03"]["minimum_visible_pixels_each_required_object"]
    global_pass=all(all(r["cameras"]["global"]["rtx_depth_consistent_attributed_pixels"][name]>=threshold
        for name in criteria["CAM-03"]["required_objects"]) for r in records)
    def depth_pass(measure):
        return (measure["p95_absolute_error_m"] is not None and measure["p95_absolute_error_m"]<=camera_gate["mesh_depth_p95_absolute_error_m"]
                and measure["valid_mask_iou"]>=camera_gate["minimum_reference_depth_valid_mask_iou"])
    sync_pass=all(r["robot_state_timestamp_error_s"]<=1e-6 and all(c["timestamp_span_s"]<=camera_gate["max_physics_surface_camera_timestamp_difference_s"]
        for c in r["cameras"].values()) for r in records)
    geometry_pass=all(depth_pass(c["geometry_vs_actual_rtx"]) for r in records for c in r["cameras"].values())
    native_pass=all(depth_pass(c["actual_native_vs_geometry"]) for r in records for name,c in r["cameras"].items() if name!="global")
    if setup.get("task"):
        chip=max(c["native_valid_separated_component_pixels"] for r in records for name,c in r["cameras"].items() if name!="global")
    else:
        chip=max(c["native_valid_chip_pixels"] for r in records for c in r["cameras"].values())
    chip_pass=chip>=camera_gate["minimum_visible_chip_pixels_in_any_recorded_frame"]
    result={"scope":"Recorded actual live Kit arrays vs measured-state CPU geometry; no new sensor rendering or semantic AOV",
        "frames":len(records),"partial_frame_subset":args.frames is not None,"CAM-03":"PASS_GEOMETRIC_ATTRIBUTION" if global_pass else "FAIL",
        "CAM-04_strict_native":"PASS" if sync_pass and geometry_pass and native_pass and chip_pass else "FAIL",
        "geometry_bridge":"PASS" if geometry_pass and sync_pass else "FAIL","time_synchronization":"PASS" if sync_pass else "FAIL",
        "native_depth_and_mask":"PASS" if native_pass else "FAIL","maximum_native_valid_chip_pixels":chip if not setup.get("task") else None,
        "maximum_native_valid_separated_component_pixels":chip if setup.get("task") else None,
        "separated_component_visibility":"PASS" if chip_pass else "FAIL",
        "task_id":(setup.get("task") or {}).get("task_id"),
        "native_semantic_aov_captured":False,"ideal_fallback_for_native_gate_used":False,
        "all_valid_error_tails_retained":True,"records":records}
    write_json(output/"result.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
    return int(not(global_pass and geometry_pass and native_pass and sync_pass and chip_pass))


if __name__=="__main__":
    raise SystemExit(main())
