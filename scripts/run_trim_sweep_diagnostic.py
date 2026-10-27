#!/usr/bin/env python3
"""Frozen physical TRIM trajectory controls, without an Isaac/runtime claim."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from robotic_cutting_sim.coupled_robot import CouplingConfig
from robotic_cutting_sim.cutting_surface import ComponentSurface,CurrentMechanicalPartition
from robotic_cutting_sim.live_material import LiveCuttingMaterial
from robotic_cutting_sim.recording import manifest,new_output,write_json
from robotic_cutting_sim.scene_frames import WorkpieceTarget
from robotic_cutting_sim.scene_validation import material_task_metrics
from robotic_cutting_sim.tasks import make_task


def reference_pose(task,case,t):
    knots=np.asarray(case["knots_local_txyz"])
    xyz=[np.interp(t,knots[:,0],knots[:,j]) for j in (1,2,3)]
    from robotic_cutting_sim.scene_frames import transform_poses
    return transform_poses(task.T_world_workpiece,np.r_[xyz,task.edge_orientation_local_xyzw])


def run(args):
    protocol_path=ROOT/"configs/trim_sweep_diagnostic_v1.json"
    protocol=json.loads(protocol_path.read_text());case=protocol[args.case]
    out=new_output(args.output_dir)
    (out/"implementation").mkdir()
    files=[Path(__file__).resolve(),protocol_path,ROOT/"configs/cutting_tasks_v1.json",ROOT/"configs/particle_surface_affine_v1.json"]
    files += [ROOT/"src/robotic_cutting_sim"/name for name in ("tasks.py","live_material.py","tlmpm.py","fixture_contacts.py",
        "cut_topology.py","cutting_surface.py","coupled_robot.py","scene_frames.py","scene_validation.py","affine_surface_controls.py","benchmark.py")]
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    for path in files:
        shutil.copyfile(path,out/"implementation"/path.name)
    write_json(out/"manifest.json",{**manifest({"case":args.case,"device":args.device,"protocol":protocol},"kinematic_material_diagnostic"),
        "implementation_sha256":hashes,"live_isaac":False,"robot_dynamics":False,"kinematic_tool":True})
    task=make_task(protocol["task_id"],protocol["seed"])
    dt=protocol["control_dt_s"]
    material=LiveCuttingMaterial(CouplingConfig(control_dt=dt,material_dt=protocol["material_dt_s"],dx=task.grid_dx_m,
        device=args.device,feedback=False),task=task)
    setup=material.reset(reference_pose(task,case,0),task.seed)
    write_json(out/"setup.json",{"task":task.to_dict(),"material":setup,"diagnostic_case":case,
        "robot_feedback_applied":False,"tool_pose_source":"ideal scripted kinematic diagnostic, no articulation"})
    partition=CurrentMechanicalPartition(material.bed.nodes.numpy(),material.bed.weights.numpy(),material.bed.gradients.numpy(),1)
    states=[];rows=[];diags=[]
    def record():
        state=material.world_state()
        if not all(np.isfinite(state[key]).all() for key in ("positions","velocities","F","plastic_strain")):
            raise FloatingPointError("Nonfinite sampled material diagnostic state")
        count,labels,provenance=partition.partition(state["codes"])
        states.append({key:state[key].copy() for key in ("positions","velocities","codes")})
        rows.append({"time_s":material.time_s,"mechanical_components":count,"provenance":provenance,
            "tool_pose_xyzw":reference_pose(task,case,material.time_s),"minimum_detF":float(np.linalg.det(state["F"]).min()),
            "maximum_material_speed_m_s":float(np.linalg.norm(state["velocities"],axis=1).max())})
        return state,labels
    record();start=time.perf_counter()
    steps=round(case["duration_s"]/dt);every=round(protocol["record_interval_s"]/dt)
    for step in range(steps):
        previous_time=step*dt
        enabled=any(a-1e-9<=previous_time<b-1e-9 for a,b in case["active_cut_intervals_s"])
        diagnostic=material.advance(reference_pose(task,case,(step+1)*dt),(step+1)*dt,cutting=enabled,cut_index=0)
        diags.append(diagnostic)
        if (step+1)%every==0 or step==steps-1:
            final,labels=record()
            if (step+1)%50==0:
                print(f"KINEMATIC diagnostic {args.case} step={step+1}/{steps} t={material.time_s:.2f}s components={rows[-1]['mechanical_components']}",flush=True)
    codes=np.asarray([state["codes"] for state in states])
    payload={"time_s":np.asarray([row["time_s"] for row in rows]),"positions":np.asarray([state["positions"] for state in states]),
        "velocities":np.asarray([state["velocities"] for state in states]),"codes":codes,"mass_kg":material.masses,
        "reference_positions":material.world_points,"reference_positions_local":material.points,"reference_indices":material.lattice,
        "reference_stencil_nodes":partition.nodes,"reference_stencil_weights":partition.weights,"reference_stencil_gradients":partition.gradients,
        "tool_pose_xyzw":np.asarray([row["tool_pose_xyzw"] for row in rows]),"component_ids":labels,
        "final_F":final["F"],"final_Fpi":final["Fpi"],"final_cut_progress":final["cut_progress"],"final_plastic_strain":final["plastic_strain"]}
    np.savez_compressed(out/"physics_state.npz",**payload)
    write_json(out/"diagnostics.json",diags);write_json(out/"frame_manifest.json",rows)
    surface=ComponentSurface(task.particle_spacing_m,args.device,voxel_size=task.particle_spacing_m/2,profile="current_volume_v1")
    vertices,triangles,surface_info=surface.extract(material.bed)
    vertices=task.points_to_world(vertices)
    face_components=np.full(len(triangles),-1,np.int32)
    for row in surface_info["component_ranges"]:
        first=row["first_triangle"];face_components[first:first+row["triangles"]]=row["mechanical_component"]
    np.savez_compressed(out/"final_surface.npz",vertices_world=vertices,triangles=triangles,triangle_component_ids=face_components)
    write_json(out/"surface_provenance.json",surface_info)
    payload.update(surface_vertices_world=vertices,surface_triangles=triangles,surface_triangle_component_ids=face_components,
        surface_component_provenance=json.dumps(surface_info),surface_timestamp_s=material.time_s)
    shared=task.evaluate_oracle(payload)
    write_json(out/"shared_task_v1_evaluation.json",shared)
    target=WorkpieceTarget(task.world_from_workpiece_xyzw,task.target_origin_local_m[2],task.edge_clearance_m)
    metrics=material_task_metrics(material.world_points,payload["positions"][-1],material.masses,labels,
        "actual conservative reference-stencil graph, no label edits",target,task.geometry)
    tail=[row for row in rows if row["time_s"]>=case["postwithdraw_start_s"]-1e-9]
    rewrites=int(np.count_nonzero((codes[:-1]!=0)&(codes[1:]!=codes[:-1])))
    detached=metrics["detached_initial_above_target_mass_fraction"]
    limits=protocol["candidate_checks"]
    checks={"initial_continuous":rows[0]["mechanical_components"]==1,"final_disconnected":rows[-1]["mechanical_components"]>=2,
        "removed_reference_above_plane_fraction":detached is not None and detached>=limits["minimum_detached_reference_above_plane_mass_fraction"],
        "postwithdraw_topology_persistent":min(row["mechanical_components"] for row in tail)>=2
            and tail[-1]["time_s"]-tail[0]["time_s"]>=limits["minimum_postwithdraw_disconnected_duration_s"]-1e-9,
        "no_nonzero_digit_rewrites":rewrites==0,"no_particle_removal":all(len(state["positions"])==len(material.points) for state in states)}
    code,counts=np.unique(codes[-1],return_counts=True)
    cap=material.points[:,2]>.02
    cap_code,cap_counts=np.unique(codes[-1,cap],return_counts=True)
    result={"status":"PASS_KINEMATIC_SEVERING_CANDIDATE" if all(checks.values()) else "FAIL_KINEMATIC_SEVERING_CANDIDATE",
        "checks":checks,"case":args.case,"duration_s":case["duration_s"],"steps":steps,"wall_s":time.perf_counter()-start,
        "final_components":rows[-1]["mechanical_components"],"component_masses_kg":[float(material.masses[labels==label].sum()) for label in np.unique(labels)],
        "detached_initial_above_target_mass_fraction":detached,"nonzero_digit_rewrites":rewrites,
        "code_histogram":dict(zip(code.tolist(),counts.tolist())),"above_plane_code_histogram":dict(zip(cap_code.tolist(),cap_counts.tolist())),
        "peak_tool_reaction_N":float(np.linalg.norm([d["next_material_wrench_world"][:3] for d in diags],axis=1).max()),
        "live_isaac":False,"articulated_robot":False,"robot_reaction_feedback":False,"full_simulation_acceptance":False,
        "original_schedule_task_v1_status":shared["status"],"schedule_boundary":"Second-pass16s has a separately declared postwithdraw phase; original v1 schedule evaluation is retained unchanged"}
    write_json(out/"result.json",result);print(json.dumps(result,indent=2))
    return 0 if all(checks.values()) else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=["fast_sweep","second_pass"],required=True)
    parser.add_argument("--device",default="cuda:1")
    parser.add_argument("--output-dir",type=Path,required=True)
    arguments=parser.parse_args()
    try:
        raise SystemExit(run(arguments))
    except Exception as error:
        if arguments.output_dir.is_dir():
            import traceback
            (arguments.output_dir/"failure.log").write_text(traceback.format_exc())
            write_json(arguments.output_dir/"result.json",{"status":"FAIL_INFRASTRUCTURE","error":str(error),"live_isaac":False})
        raise
