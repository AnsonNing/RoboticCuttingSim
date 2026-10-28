#!/usr/bin/env python3
"""Newton FK/IK and actual USD-pad/table preflight; no dynamics or image rendering."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.task_scenarios import apply_scenario, initial_joints
from robotic_cutting_sim.robot import RobotConfig,DualFrankaController
from robotic_cutting_sim.workcell import SECONDARY_PARK_JOINTS, ASSETS
from robotic_cutting_sim.finger_geometry import discover_pads, pad_plane_clearance
from robotic_cutting_sim.scene_frames import pose_matrix,transform_points
from robotic_cutting_sim.tool_profile import tool_spec_for_task
from robotic_cutting_sim.camera_rig import make_rig
from robotic_cutting_sim.recording import new_output,write_json,manifest


def main():
    import mujoco
    from pxr import Usd
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',default='ruler_workcell_v2')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=new_output(a.out)
    write_json(out/'manifest.json',manifest(dict(scenario=a.scenario,seed=42,penetration_limit_m=.0001,
        minimum_pad_handle_camera_plane_clearance_m=.006),'newton_geometry_preflight'))
    task=apply_scenario(make_task('TRIM-01-R'),a.scenario)
    mount=task.hand_from_edge_xyz_xyzw
    c=DualFrankaController(RobotConfig(tool_shape='cut_wedge',tool_offset_m=tuple(mount[:3]),
        tool_rotation_xyzw=tuple(mount[3:]),wedge_length_m=task.blade_length_m,
        wedge_height_m=task.blade_height_m,wedge_half_width_m=task.blade_half_width_m))
    q=c.initial_q.copy();q[:7]=initial_joints('raised_workpiece_50mm_v1',task.task_id);c.state.joint_q.assign(q)
    q[:7],info=c.inverse_kinematics(task.reference_edge_pose_world(0.),iterations=128)
    initial=q[:7].copy();c.state.joint_q.assign(q)
    asset=ROOT/'artifacts/workcell_asset_cache'/ASSETS['franka'];stage=Usd.Stage.Open(str(asset))
    hand=next(prim for prim in stage.Traverse() if prim.GetName()=='panda_hand')
    pads=discover_pads(stage,str(hand.GetPath()))
    pad_indices=[next(i for i,label in enumerate(c.model.body_label[:c.bodies_per_arm])
                      if label.endswith(pad['body_path'])) for pad in pads]
    spec=tool_spec_for_task(task);rig=make_rig(task.task_id)
    origin=task.points_to_world(np.array(task.target_origin_local_m));normal=np.asarray(task.T_world_workpiece[:3,:3])@task.target_normal_local
    signs=np.array(list(__import__('itertools').product((-1.,1.),repeat=3)))
    rows=[];contacts=[];clearances=[];previous=initial
    for time_s in np.arange(401)*.02:
        targets,info=c.inverse_kinematics(task.reference_edge_pose_world(float(time_s)),iterations=128)
        step=float(np.max(np.abs(targets-previous)));previous=targets.copy()
        q[:7]=targets;q[7:9]=.0112;q[9:16]=SECONDARY_PARK_JOINTS;c.state.joint_q.assign(q)
        c.newton.eval_fk(c.model,c.state.joint_q,c.state.joint_qd,c.state)
        body=c.state.body_q.numpy();hand_pose=body[c.hand_local]
        c.solver._update_mjc_data(c.solver.mj_data,c.model,c.state);mujoco.mj_fwdPosition(c.solver.mj_model,c.solver.mj_data)
        for contact in c.solver.mj_data.contact[:c.solver.mj_data.ncon]:
            if contact.dist<-.0001:
                contacts.append(dict(time_s=float(time_s),distance_m=float(contact.dist),geoms=[
                    mujoco.mj_id2name(c.solver.mj_model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in contact.geom]))
        pad_poses={pad['body_path']:body[index] for pad,index in zip(pads,pad_indices)}
        clearance=pad_plane_clearance(pads,pad_poses,origin,normal)
        T_hand=pose_matrix(hand_pose);T_tool=T_hand@spec.T_hand_tool
        for shape in spec.shapes:
            if shape.role=='handle':
                clearance['handle']=float(np.min((transform_points(T_tool,shape.vertices_tool_m)-origin)@normal))
        for box in rig.specs['wrist_d405_0']['visual_obbs']:
            corners=signs*np.asarray(box['dimensions_m'])/2
            clearance['camera0_'+box['name']]=float(np.min((transform_points(T_hand@box['T_hand_obb'],corners)-origin)@normal))
        clearances.append(clearance)
        rows.append(dict(time_s=float(time_s),q=targets,ik=info,joint_margin_rad=float(np.minimum(targets-c.lower[:7],c.upper[:7]-targets).min()),joint_step_rad=step))
    minima={key:min(row[key] for row in clearances) for key in clearances[0]}
    checks=dict(all_ik_converged=all(row['ik']['converged'] for row in rows),table_and_self_collision_clear=not contacts,
                pads_handle_camera_clear_target_plane=all(value>=.006 for value in minima.values()))
    write_json(out/'result.json',dict(status='PASS_GEOMETRY_PREFLIGHT' if all(checks.values()) else 'FAIL_GEOMETRY_PREFLIGHT',
        checks=checks,scenario=a.scenario,initial_joint_positions=initial,minimum_plane_clearance_m=minima,
        minimum_joint_margin_rad=min(row['joint_margin_rad'] for row in rows),max_joint_step_rad=max(row['joint_step_rad'] for row in rows),
        contacts=contacts,rows=rows,scope='Actual Newton/MuJoCo robot/table and actual composed USD pad corners; no dynamics. Blade is wedge proxy; fixtures and camera/robot intersection remain separate live audits.'))
    print(checks,minima,initial.tolist(),flush=True)


if __name__=='__main__':main()
