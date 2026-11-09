"""ADR 0022 §8 C2: batched LM edge IK (``n_problems = N``) on the Franka 0 prototype, Warp CPU, torch-free.

* N problems in one solver equal N single-problem solves of the same prototype bitwise (lambda is per row);
* against the frozen ``DualFrankaController.inverse_kinematics`` (two-arm model, knife shape on the hand) on 50
  TRIM reference targets: both pass the 1 mm / 0.01 rad gate; the joint difference is reported, not claimed equal;
* an unreachable target gets ``REJECTED_IK_UNREACHABLE`` only in its own row.
"""
from functools import lru_cache

import numpy as np
import pytest
import warp as wp

from robotic_cutting_sim.batched import lab_scene
from robotic_cutting_sim.batched.robot_ik import IK_OK, IK_UNREACHABLE, BatchedEdgeIK, ik_rejection_status_code

wp.config.quiet = True
DEVICE = "cpu"


@lru_cache(maxsize=None)
def task():
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    return compile_task(generate_scene(512), goal_config(), seed=42, particle_spacing_m=.0025)


@lru_cache(maxsize=None)
def prototype(mount):
    return lab_scene.build_newton_rigid_scene(task(), 1, mount, device=DEVICE).prototype_builder


def targets(count=50):
    t = task()
    return np.asarray([t.reference_edge_pose_world(x) for x in np.linspace(0., float(t.duration_s), count)])


def home_seed(n):
    q0, _ = lab_scene.home_joint_positions(task(), "rigid_mount_v1", lab_scene._grasp_default())
    return np.tile(q0.astype(np.float32), (n, 1))


def solve(builder, seeds, goals):
    ik = BatchedEdgeIK(builder, len(goals), task().hand_from_edge_xyz_xyzw, device=DEVICE)
    ik.solve_from(wp.array(seeds, dtype=float, device=DEVICE), wp.array(goals, dtype=wp.float64, device=DEVICE))
    return ik


@pytest.mark.parametrize("mount", lab_scene.TOOL_MOUNTS)
def test_batched_ik_equals_single_problem_solves_bitwise(mount):
    goals = targets(12)
    batched = solve(prototype(mount), home_seed(len(goals)), goals)
    assert np.all(batched.status.numpy() == IK_OK)
    single = BatchedEdgeIK(prototype(mount), 1, task().hand_from_edge_xyz_xyzw, device=DEVICE)
    for e in range(len(goals)):
        single.solve_from(wp.array(home_seed(1), dtype=float, device=DEVICE),
                          wp.array(goals[e:e+1], dtype=wp.float64, device=DEVICE))
        for name in ("joint_q_out", "edge", "position_error", "rotation_error", "status"):
            assert np.array_equal(getattr(single, name).numpy()[0], getattr(batched, name).numpy()[e]), (e, name)


def test_batched_ik_against_dual_franka_controller_on_trim_targets():
    from robotic_cutting_sim.robot import DualFrankaController, RobotConfig
    goals = targets(50)
    mount = np.asarray(task().hand_from_edge_xyz_xyzw, float)
    batched = solve(prototype("rigid_mount_v1"), home_seed(len(goals)), goals)
    q_batched = batched.joint_q_out.numpy()
    controller = DualFrankaController(RobotConfig(backend="mujoco_warp", device=DEVICE,
                                                  tool_offset_m=tuple(mount[:3]), tool_rotation_xyzw=tuple(mount[3:])))
    home = home_seed(1)[0, :7].astype(float)
    reference, info = [], []
    for goal in goals:
        controller.reset(home)
        q, report = controller.inverse_kinematics(goal)
        reference.append(q)
        info.append(report)
    reference = np.asarray(reference)
    assert all(r["converged"] for r in info), "frozen controller IK must pass its own gate on every TRIM target"
    assert np.all(batched.status.numpy() == IK_OK)
    assert batched.position_error.numpy().max() <= 1e-3 and batched.rotation_error.numpy().max() <= .01
    delta = np.abs(q_batched[:, :7]-reference)
    print(f"C2 joint delta vs DualFrankaController: max {delta.max():.3e} rad, mean {delta.mean():.3e} rad; "
          f"batched max pos err {batched.position_error.numpy().max():.3e} m, "
          f"max rot err {batched.rotation_error.numpy().max():.3e} rad")


def test_unreachable_target_rejects_only_its_row():
    goals = targets(4)
    goals[2, :3] = [1.5, 0., .2]   # far outside the Panda workspace
    ik = solve(prototype("rigid_mount_v1"), home_seed(4), goals)
    status = ik.status.numpy()
    assert status.tolist() == [IK_OK, IK_OK, IK_UNREACHABLE, IK_OK]
    from robotic_cutting_sim.policy_env import ACTION_STATUSES
    assert ACTION_STATUSES[ik_rejection_status_code()] == "REJECTED_IK_UNREACHABLE"
    ok = solve(prototype("rigid_mount_v1"), home_seed(3), goals[[0, 1, 3]])
    assert np.array_equal(ok.joint_q_out.numpy(), ik.joint_q_out.numpy()[[0, 1, 3]])
