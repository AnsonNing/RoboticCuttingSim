#!/usr/bin/env python3
"""Static actual official-collider park check; no observation-pose optimization."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from robotic_cutting_sim.robot import DualFrankaController,RobotConfig
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.workcell import SECONDARY_PARK_JOINTS
from robotic_cutting_sim.recording import new_output,write_json,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir",type=Path)
    args=parser.parse_args()
    out=new_output(args.output_dir)
    import mujoco
    rows={}
    for task_id in ("CUT-01","TRIM-01"):
        task=make_task(task_id)
        # Existing task IK moves only the active cutting arm. Secondary q is
        # identical to the original safe park, independent of camera layout.
        robot=DualFrankaController(RobotConfig(tool_offset_m=tuple(task.hand_from_edge_xyz_xyzw[:3]),
            tool_rotation_xyzw=tuple(task.hand_from_edge_xyz_xyzw[3:])))
        q=robot.initial_q.copy()
        q[:7],ik=robot.inverse_kinematics(task.reference_edge_pose_world(0.))
        robot.solver.mj_data.qpos[:]=q
        robot.solver.mj_data.qvel[:]=0
        mujoco.mj_forward(robot.solver.mj_model,robot.solver.mj_data)
        contacts=[]
        for i in range(robot.solver.mj_data.ncon):
            c=robot.solver.mj_data.contact[i]
            contacts.append({"geometry":[mujoco.mj_id2name(robot.solver.mj_model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in c.geom],
                "signed_distance_m":float(c.dist)})
        arm_penetrations=[c for c in contacts if c['signed_distance_m'] < -1e-5
            and not any('knife' in str(name) for name in c['geometry'])]
        rows[task_id]={"ik":ik,"secondary_joints_rad":q[9:16],"normal_park":bool(np.array_equal(q[9:16],np.asarray(SECONDARY_PARK_JOINTS,dtype=q.dtype))),
            "contacts":contacts,"robot_penetrating_contacts":arm_penetrations}
        np.savez_compressed(out/(task_id+"_static.npz"),q=q)
    passed=all(r['normal_park'] and r['ik']['converged'] and not r['robot_penetrating_contacts'] for r in rows.values())
    write_json(out/'result.json',{'status':'PASS_STATIC_OFFICIAL_COLLIDERS' if passed else 'FAIL_STATIC_OFFICIAL_COLLIDERS',
        'tasks':rows,'scope':'Actual official convex colliders at task-start arm poses. Camera visual geometry is audited separately; not full trajectory clearance or sensor physics.',
        'camera_aiming_or_secondary_IK':False,'penetration_tolerance_m':1e-5})
    write_json(out/'manifest.json',manifest({},'static_parked_arm_collider_check'))
    print('PASS' if passed else 'FAIL')
    return int(not passed)

if __name__=='__main__':
    raise SystemExit(main())
