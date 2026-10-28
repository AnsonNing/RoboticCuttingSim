"""CPU contracts for action API v2, IK rejection, state profile, reward and Gym.

Fake backends only; actual runtime behavior is validated separately on GPU.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.control import (ActionLimits, DELTA_MODE, IKFailure, bound_target,
                                         compose_delta, normalized_delta, resolve_delta)
from robotic_cutting_sim.live_env import CuttingTaskEnv
from robotic_cutting_sim.reward import CuttingReward
from robotic_cutting_sim.tasks import make_task
from test_live_env import backend

EDGE = np.array([0., 0., .05, *Rotation.from_euler("y", 90, degrees=True).as_quat()])


def tracking_backend(reachable=lambda pose: True, task_id="CUT-01"):
    """Fake Lab backend whose measured edge follows reachable Cartesian targets."""
    b = backend("lab")
    task = make_task(task_id)
    b.task.hand_from_edge_xyz_xyzw = task.hand_from_edge_xyz_xyzw
    b.task.T_world_workpiece = task.T_world_workpiece
    b.task.duration_s = 1.
    b.edge = EDGE.copy()
    b.attempts = []
    base_reset, base_step = b.reset, b.step

    def reset(seed):
        b.edge = EDGE.copy()
        base_reset(seed)

    def step(action):
        b.attempts.append(action)
        if action["mode"] == "cartesian_edge_pose_world":
            if not reachable(action["target"]):
                raise IKFailure("unreachable", {"converged": False})
            b.edge = np.asarray(action["target"], float).copy()
        base_step(action)

    def measured():
        return {"timestamp_s": b.time_s, "q": np.zeros((2, 7)), "qdot": np.zeros((2, 7)),
                "tool_pose": b.edge.copy()}

    b.reset, b.step, b.measured = reset, step, measured
    return b


def test_delta_in_edge_and_world_frames():
    turned = compose_delta(EDGE, [.001, 0, 0, 0, 0, 0], "edge")
    # Edge-local +x (advance) points down in the vertical CUT orientation.
    assert np.allclose(turned[:3], [0, 0, .049], atol=1e-12)
    world = compose_delta(EDGE, [.001, 0, 0, 0, 0, np.pi/2], "world")
    assert np.allclose(world[:3], [.001, 0, .05])
    expected = Rotation.from_rotvec([0, 0, np.pi/2])*Rotation.from_quat(EDGE[3:])
    assert (Rotation.from_quat(world[3:])*expected.inv()).magnitude() < 1e-12


def test_step_limits_preserve_direction_and_report():
    limits = ActionLimits()
    pose, report = resolve_delta(EDGE, [0, .03, .04, 0, 0, .5], limits, frame="world")
    step = pose[:3]-EDGE[:3]
    assert np.isclose(np.linalg.norm(step), limits.max_translation_m)
    assert np.allclose(step/np.linalg.norm(step), [0, .6, .8])
    turned = (Rotation.from_quat(pose[3:])*Rotation.from_quat(EDGE[3:]).inv()).as_rotvec()
    assert np.isclose(np.linalg.norm(turned), limits.max_rotation_rad)
    assert report["translation_clamped"] and report["rotation_clamped"] and not report["workspace_clamped"]
    assert np.isclose(report["requested_translation_m"], .05)
    assert np.allclose(normalized_delta([2, -1, 0, 0, 0, 1], limits),
                       [limits.max_translation_m, -limits.max_translation_m, 0, 0, 0, limits.max_rotation_rad])


def test_absolute_bound_and_workspace_box():
    limits = ActionLimits(max_translation_m=1., max_rotation_rad=np.pi)
    far = np.r_[[.5, 0, .05], EDGE[3:]]
    pose, report = bound_target(EDGE, far, limits)
    assert report["workspace_clamped"] and np.isclose(pose[0], limits.workspace_max_local_m[0])
    with pytest.raises(ValueError):
        ActionLimits(workspace_min_local_m=(0, 0, 0), workspace_max_local_m=(0, 1, 1))


def test_wrapper_delta_accumulates_on_commanded_edge():
    b = tracking_backend()
    env = CuttingTaskEnv(b, action_mode=DELTA_MODE, observation_profile="state")
    obs, _ = env.reset()
    assert set(obs) == {"timestamp_s", "proprio"}
    for _ in range(3):
        _, reward, terminated, _, info = env.step(np.array([.01, 0, 0, 0, 0, 0]))
    assert reward == 0 and not terminated and info["ik_status"] == "ok"
    assert info["action_report"]["translation_clamped"]
    assert np.allclose(b.edge[:3], [0, 0, .05-3*.002])


def test_bounded_absolute_is_opt_in():
    b = tracking_backend()
    far = np.r_[[0, 0, .10], EDGE[3:]]
    env = CuttingTaskEnv(b)
    env.reset()
    env.step({"mode": "cartesian_edge_pose_world", "target": far})
    assert np.allclose(b.edge, far)  # historical unbounded behavior
    env = CuttingTaskEnv(tracking_backend(), action_limits=ActionLimits())
    env.reset()
    _, _, _, _, info = env.step({"mode": "cartesian_edge_pose_world", "target": far})
    assert np.isclose(env.backend.edge[2], .052) and info["action_report"]["translation_clamped"]


def test_ik_failure_raises_by_default_and_rejects_with_hold():
    below_start_only = lambda pose: pose[2] < .045
    env = CuttingTaskEnv(tracking_backend(below_start_only), action_mode=DELTA_MODE)
    env.reset()
    with pytest.raises(IKFailure):
        env.step(np.zeros(6))
    b = tracking_backend(lambda pose: pose[1] < .003)
    env = CuttingTaskEnv(b, action_mode=DELTA_MODE, delta_frame="world", ik_failure="reject",
                         max_consecutive_ik_failures=3)
    env.reset()
    env.step(np.array([0, .002, 0, 0, 0, 0]))
    for expected in (1, 2):
        _, _, terminated, _, info = env.step(np.array([0, .002, 0, 0, 0, 0]))
        assert info["ik_status"] == "rejected" and info["consecutive_ik_failures"] == expected and not terminated
        # The rejected target is replaced by a hold at the last accepted target.
        assert np.allclose(b.attempts[-1]["target"][:3], [0, .002, .05])
    _, _, terminated, _, info = env.step(np.array([0, .002, 0, 0, 0, 0]))
    assert terminated and info["termination_reason"] == "IK_FAILURES"
    env.reset()
    _, _, terminated, _, info = env.step(np.array([0, -.002, 0, 0, 0, 0]))
    assert not terminated and info["ik_status"] == "ok" and info["consecutive_ik_failures"] == 0


class _Material:
    def __init__(self, task):
        self.task = task
        self.world_points = task.points_to_world(task.sample_particles()[0])
        self.codes = np.zeros(len(self.world_points), np.int32)
        self.masses = np.ones(len(self.world_points))

    def world_state(self):
        n = len(self.world_points)
        return {"positions": self.world_points, "velocities": np.zeros((n, 3)), "codes": self.codes.copy()}


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
def test_reward_tracks_target_progress_and_penalties(task_id):
    b = tracking_backend(task_id=task_id)
    task = make_task(task_id, particle_spacing_m=.004)
    b.task.goal_kind, b.task.grid_dx_m = task.goal_kind, task.grid_dx_m
    b.material = _Material(task)
    env = CuttingTaskEnv(b, action_mode=DELTA_MODE, delta_frame="world", reward=CuttingReward(task))
    env.reset()
    _, idle, _, _, info = env.step(np.zeros(6))
    assert info["reward_terms"]["target_progress"] == 0 and idle < 0
    plane = task.target_plane_world()
    signed = b.material.world_points@plane[:3]+plane[3]
    band = (task.grid_dx_m+task.particle_spacing_m)/2
    target = signed > band if task_id == "TRIM-01" else np.abs(signed) <= band
    b.material.codes[target] = 1
    _, reward, _, _, info = env.step(np.zeros(6))
    assert info["reward_terms"]["target_progress"] == 1 and reward > 9
    b.material.codes[signed < -3*task.grid_dx_m] = 2
    _, reward, _, _, info = env.step(np.zeros(6))
    assert info["reward_terms"]["off_target"] > 0 and reward < 0


def test_gym_adapter_spaces_and_step():
    gym = pytest.importorskip("gymnasium")
    from robotic_cutting_sim.gym_env import CuttingGymEnv
    env = CuttingGymEnv(CuttingTaskEnv(tracking_backend(), action_mode=DELTA_MODE, observation_profile="state"))
    assert env.observation_space.shape == (2*7*2+7,)
    obs, reward, terminated, truncated, info = env.step(np.array([1, 0, 0, 0, 0, 0], np.float32))
    assert env.observation_space.contains(obs) and info["ik_status"] == "ok"
    assert np.allclose(env.env.backend.edge[:3], [0, 0, .048])
    vision = CuttingGymEnv(CuttingTaskEnv(tracking_backend(), action_mode=DELTA_MODE))
    obs, _ = vision.reset()
    assert vision.observation_space.contains(obs) and set(obs) >= {"state", "d455_0_rgb", "d455_0_depth_m"}
    from gymnasium.utils.env_checker import check_env
    check_env(CuttingGymEnv(CuttingTaskEnv(tracking_backend(), action_mode=DELTA_MODE,
                                           observation_profile="state")), skip_render_check=True)
