"""CPU wrapper contracts with explicit fake backends, not runtime validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim.live_env import CuttingTaskEnv
from robotic_cutting_sim.tasks import make_task


def backend(kind):
    actual_task = make_task("CUT-01")
    task = SimpleNamespace(task_id="CUT-01", duration_s=.04,
        validate_action=actual_task.validate_action, to_dict=actual_task.to_dict,
        evaluate_oracle=lambda data: {"status": "FAIL_TASK_METRICS", "reason": "Synthetic uncut fixture"})
    array = lambda data: SimpleNamespace(numpy=lambda: np.asarray(data))
    material = SimpleNamespace(masses=np.ones(2), world_points=np.zeros((2, 3)),
        world_state=lambda: {"positions": np.zeros((2, 3)), "velocities": np.zeros((2, 3)), "codes": np.zeros(2, np.int32)},
        bed=SimpleNamespace(nodes=array(np.zeros((2, 8), int)), weights=array(np.zeros((2, 8))),
                            gradients=array(np.zeros((2, 8, 3)))))
    b = SimpleNamespace(task=task, time_s=0., material=material, closed=False, commands=[])

    def frames():
        return {name: {"rgb": np.zeros((2, 3, 3), np.uint8), "depth_m": np.ones((2, 3), np.float32),
            "surface_timestamp_s": b.time_s, "force": np.ones(6), "oracle": "hidden"}
            for name in ("d455_0", "d455_1", "global")}

    def proprio():
        return {"timestamp_s": b.time_s, "q": np.zeros((2, 7)), "qdot": np.zeros((2, 7)),
                "tool_pose": np.array([0, 0, .05, 0, 0, 0, 1]), "force": np.ones(6)}

    def reset(seed):
        b.time_s = 0.
        b.frames = frames()

    def step(action):
        b.commands.append(action)
        b.time_s = round(b.time_s+.02, 6)
        b.frames = frames()

    b.reset, b.step = reset, step
    b.close = lambda: setattr(b, "closed", True)
    b.calibration = {name: {"K": np.eye(3).tolist()} for name in ("d455_0", "d455_1", "global")}
    b.surface = (np.zeros((3, 3)), np.array([[0, 1, 2]]))
    b.surface_metadata = {"component_ranges": [{"first_triangle": 0, "triangles": 1, "mechanical_component": 0}],
                          "partition_rule": "Synthetic CPU unit fixture"}
    if kind == "lab":
        b.measured = proprio
    else:
        b.capture = lambda: (frames(), {})
        b.proprioception = proprio
        b.surface_arrays, b.surface_info = b.surface, b.surface_metadata
    return b


@pytest.mark.parametrize("kind", ["lab", "kit"])
def test_same_api_keeps_privileged_payload_out_and_separates_time_from_success(kind):
    native = backend(kind)
    env = CuttingTaskEnv(native)
    obs, info = env.reset(seed=7)
    assert set(obs) == {"rgb", "depth_m", "valid_mask", "calibration", "timestamp_s", "proprio"}
    assert set(obs["proprio"]) == {"q", "qdot", "tool_pose"}
    assert set(obs["rgb"]) == {"d455_0", "d455_1"}
    assert env.evaluate()["status"] == "NOT_RUN"
    pose = obs["proprio"]["tool_pose"]
    _, reward, terminated, truncated, info = env.step({"mode": "cartesian_edge_pose_world", "target": pose, "cut_enabled": False})
    assert not truncated and not terminated and reward == 0
    assert native.commands[0]["cut_enabled"] is False
    env.step({"mode": "joint_position_rad", "target": np.zeros(7)})
    assert env.evaluate()["episode_complete"] and not env.evaluate()["success"]
    assert env.render("global").shape == (2, 3, 3)
    with pytest.raises(RuntimeError, match="Episode ended"):
        env.step()
    env.reset(seed=7)
    assert len(env._history) == 1 and env.observe()["timestamp_s"] == 0
    env.close()
    env.close()
    assert native.closed
    with pytest.raises(RuntimeError, match="closed"):
        env.observe()


def test_wrapper_rejects_stale_camera_and_preserves_vision_only_contract():
    native = backend("kit")
    env = CuttingTaskEnv(native, observation_profile="vision_only")
    obs, _ = env.reset()
    assert "proprio" not in obs
    old, _ = native.capture()
    native.capture = lambda: (old, {})
    with pytest.raises(ValueError, match="timestamps must match"):
        env.step()


@pytest.mark.parametrize("kind", ["lab", "kit"])
def test_omitted_reset_seed_preserves_constructed_task_seed(kind):
    native = backend(kind)
    native.task.seed = 73
    original = native.reset
    received = []

    def reset(seed):
        received.append(seed)
        original(seed)

    native.reset = reset
    env = CuttingTaskEnv(native)
    _, info = env.reset()
    assert received == [73] and info["seed"] == 73


@pytest.mark.parametrize('kind',['lab','kit'])
def test_gripper_only_control_holds_arm_and_rejects_invalid_arm_transaction(kind):
    b=backend(kind)
    b.options=SimpleNamespace(tool_attachment_mode='physical_grasp')
    b.gripper_commands=[]
    b.set_gripper=b.gripper_commands.append
    env=CuttingTaskEnv(b)
    env.reset()
    with pytest.raises(ValueError):
        env.step({'gripper':'release','mode':'joint_position_rad','target':[1,2]})
    assert not b.gripper_commands and not b.commands
    env.step({'gripper':'release'})
    assert b.gripper_commands==['release']
    assert b.commands[0]['mode']=='joint_position_rad'
    assert not b.commands[0]['cut_enabled']


def test_gripper_cannot_release_a_legacy_attachment():
    b=backend('kit');b.options=SimpleNamespace(tool_attachment_mode='legacy_fixed_tool')
    env=CuttingTaskEnv(b);env.reset()
    with pytest.raises(ValueError,match='physical_grasp'):
        env.step({'gripper':'release'})
    assert not b.commands
