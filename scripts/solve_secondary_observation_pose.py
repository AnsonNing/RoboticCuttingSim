#!/usr/bin/env python3
"""Historical v1-v3 observation-pose experiment; never used by the v4 runtime.

Official asset/FK/IK plus a static MuJoCo contact diagnostic; no cutting episode.
Retained solely for reproducing old observation-arm experiments. Current
Franka 1 stays at its normal park with a fixed hand-local wrist camera.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.recording import manifest, new_output, write_json
from robotic_cutting_sim.robot import DualFrankaController, RobotConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    output = new_output(args.output_dir)
    desired = np.array([.20, .18, .34, 1., 0., 0., 0.])
    robot = DualFrankaController(RobotConfig(device="cpu", backend="mujoco_cpu",
        tool_rotation_xyzw=(1, 0, 0, 0), tool_shape="cut_wedge"))
    wp, newton = robot.wp, robot.newton
    import newton.ik as ik
    import mujoco
    hand = robot.hand_local+robot.bodies_per_arm
    position = ik.IKObjectivePosition(link_index=hand, link_offset=wp.vec3(0.),
        target_positions=wp.array([wp.vec3(*desired[:3])], dtype=wp.vec3, device="cpu"))
    orientation = ik.IKObjectiveRotation(link_index=hand, link_offset_rotation=wp.quat(0, 0, 0, 1),
        target_rotations=wp.array([wp.vec4(*desired[3:])], dtype=wp.vec4, device="cpu"))
    limit = ik.IKObjectiveJointLimit(robot.model.joint_limit_lower, robot.model.joint_limit_upper, weight=10.)
    solver = ik.IKSolver(model=robot.model, n_problems=1, objectives=[position, orientation, limit],
        lambda_initial=.01, jacobian_mode=ik.IKJacobianType.ANALYTIC)
    q = robot.initial_q.copy()
    arm0, arm0_receipt = robot.inverse_kinematics([-.045, 0, .06, 0, 0, 0, 1], iterations=96)
    q[:7] = arm0
    # Seed toward the workspace instead of the old outward parking direction.
    q[9:16] = [-.3, -.5, 0., -2.5, 0., 2., .8]
    q_gpu = wp.array(q.reshape(1, -1), dtype=float, device="cpu")
    solver.step(q_gpu, q_gpu, iterations=160)
    solved = q_gpu.numpy()[0]
    state = robot.model.state()
    # Select an interior redundant branch with the actual imported model's FK.
    # The direct Newton objective can leave a redundant wrist joint on a limit.
    from scipy.optimize import least_squares
    lower, upper = robot.lower[9:16]+.05, robot.upper[9:16]-.05
    midpoint = (lower+upper)/2
    def residual(arm_q):
        candidate = solved.copy()
        candidate[9:16] = arm_q
        state.joint_q.assign(candidate)
        newton.eval_fk(robot.model, state.joint_q, robot.model.joint_qd, state)
        pose = state.body_q.numpy()[hand]
        rot = (Rotation.from_quat(desired[3:]).inv()*Rotation.from_quat(pose[3:])).as_rotvec()
        return np.r_[1000*(pose[:3]-desired[:3]), 1000*rot, .001*(arm_q-midpoint)]
    refined = least_squares(residual, np.clip(solved[9:16], lower+1e-5, upper-1e-5),
        bounds=(lower, upper), diff_step=.002, max_nfev=200, xtol=1e-11, ftol=1e-11, gtol=1e-9)
    solved[9:16] = refined.x
    state.joint_q.assign(solved)
    newton.eval_fk(robot.model, state.joint_q, robot.model.joint_qd, state)
    poses = state.body_q.numpy()
    actual = poses[hand].astype(float)
    pe = float(np.linalg.norm(actual[:3]-desired[:3]))
    re = float((Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(desired[3:])).magnitude())
    margins = np.minimum(solved-robot.lower, robot.upper-solved)
    robot.solver.mj_data.qpos[:] = solved
    robot.solver.mj_data.qvel[:] = 0
    mujoco.mj_forward(robot.solver.mj_model, robot.solver.mj_data)
    contacts = []
    for index in range(robot.solver.mj_data.ncon):
        contact = robot.solver.mj_data.contact[index]
        names = [mujoco.mj_id2name(robot.solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in contact.geom]
        contacts.append({"geometry": names, "signed_distance_m": float(contact.dist)})
    arm_margin = np.minimum(solved[9:16]-robot.lower[9:16], robot.upper[9:16]-solved[9:16])
    penetration = [c for c in contacts if c["signed_distance_m"] < -1e-5]
    passed = pe < .001 and re < .01 and bool(np.all(arm_margin >= .0499)) and not penetration
    result = {"status": "PASS_KINEMATIC_LAYOUT" if passed else "FAIL_KINEMATIC_LAYOUT",
        "secondary_observation_joints_rad": solved[9:16], "desired_secondary_hand_world_xyzw": desired,
        "actual_secondary_hand_world_xyzw": actual, "position_error_m": pe, "rotation_error_rad": re,
        "minimum_joint_limit_margin_rad_or_m": float(margins.min()),
        "secondary_arm_minimum_joint_limit_margin_rad": float(arm_margin.min()),
        "static_penetrating_contact_count": len(penetration),
        "arm0_initial_ik": arm0_receipt, "arm0_joint_positions_rad": solved[:7],
        "secondary_hand_index": hand, "all_body_pose_world_xyzw": poses,
        "static_mujoco_contacts": contacts, "camera_housing_contact_validation": "NOT_RUN",
        "scope": "Reachability and static official convex-collider diagnostic; no runtime/camera or complete trajectory acceptance"}
    np.savez_compressed(output/"state.npz", q=solved, body_pose=poses, lower=robot.lower, upper=robot.upper)
    write_json(output/"result.json", result)
    write_json(output/"manifest.json", manifest({"hand_goal": desired.tolist()}, "secondary_observation_pose_official_asset"))
    print({key: value for key, value in result.items() if key != "all_body_pose_world_xyzw"}, flush=True)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
