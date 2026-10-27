"""CPU checks for the vision observation boundary, independent of GPU physics."""

from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim.env import CuttingEnv, EnvConfig, build_observation
from robotic_cutting_sim.trajectory import reference_trajectory


@pytest.fixture
def camera_inputs():
    rgbd, calibration = {}, {}
    for index, name in enumerate(("ideal_0", "ideal_1", "global")):
        depth = np.full((3, 4), 0.6 + index * 0.1, dtype=np.float32)
        depth[0, 0] = 0.0
        rgbd[name] = {
            "rgb": np.full((3, 4, 3), index * 60, dtype=np.uint8),
            "depth_m": depth,
            "valid_mask": depth > 0.0,
            "force_N": np.array([1.0, 2.0, 3.0]),
            "particle_positions_m": np.zeros((2, 3)),
            "cut_labels": np.array([0, 1]),
        }
        calibration[name] = {
            "K": [[2.0, 0.0, 2.0], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]],
            "T_world_camera": np.eye(4).tolist(),
            "width": 4,
            "height": 3,
            "sensor_model": "ideal pinhole",
            "source_camera": name,
        }
    return rgbd, calibration


def test_vision_observation_exports_only_the_two_calibrated_rgbd_views(camera_inputs):
    rgbd, calibration = camera_inputs
    observation = build_observation(rgbd, calibration, 0.125)

    assert set(observation) == {
        "rgb", "depth_m", "valid_mask", "calibration", "timestamp_s"
    }
    assert type(observation["timestamp_s"]) is float
    assert observation["timestamp_s"] == 0.125
    for field in ("rgb", "depth_m", "valid_mask", "calibration"):
        assert set(observation[field]) == {"d455_0", "d455_1"}
    for output_name, input_name in (("d455_0", "ideal_0"), ("d455_1", "ideal_1")):
        for field in ("rgb", "depth_m", "valid_mask"):
            np.testing.assert_array_equal(
                observation[field][output_name], rgbd[input_name][field]
            )
            assert observation[field][output_name].dtype == rgbd[input_name][field].dtype
        assert observation["calibration"][output_name] == calibration[input_name]
        assert observation["calibration"][output_name]["sensor_model"] == "ideal pinhole"


def test_vision_only_does_not_expose_proprio_force_or_oracle(camera_inputs):
    rgbd, calibration = camera_inputs
    observation = build_observation(
        rgbd,
        calibration,
        0.0,
        profile="vision_only",
        proprio={
            "q": np.zeros(7),
            "force_N": np.ones(3),
            "particle_positions_m": np.ones((2, 3)),
        },
    )

    assert "proprio" not in observation
    assert set(observation) == {
        "rgb", "depth_m", "valid_mask", "calibration", "timestamp_s"
    }


def test_vision_proprio_filters_force_and_exports_only_available_proprio(camera_inputs):
    rgbd, calibration = camera_inputs
    supplied = {
        "q": np.arange(7, dtype=np.float32),
        "tool_pose": np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0]),
        "force_N": np.ones(3),
        "torque_Nm": np.ones(3),
        "particle_positions_m": np.ones((2, 3)),
        "cut_labels": np.array([0, 1]),
    }
    observation = build_observation(
        rgbd, calibration, 0.25, profile="vision_proprio", proprio=supplied
    )

    assert set(observation) == {
        "rgb", "depth_m", "valid_mask", "calibration", "timestamp_s", "proprio"
    }
    assert set(observation["proprio"]) == {"q", "tool_pose"}
    for name in ("q", "tool_pose"):
        np.testing.assert_array_equal(observation["proprio"][name], supplied[name])


def test_vision_proprio_includes_supplied_joint_velocities(camera_inputs):
    rgbd, calibration = camera_inputs
    qdot = np.linspace(-0.1, 0.1, 7, dtype=np.float32)
    observation = build_observation(
        rgbd, calibration, 0.0, profile="vision_proprio", proprio={"qdot": qdot}
    )

    assert set(observation["proprio"]) == {"qdot"}
    np.testing.assert_array_equal(observation["proprio"]["qdot"], qdot)


@pytest.mark.parametrize("profile", ["force", "oracle", "unknown"])
def test_unknown_observation_profile_is_rejected(camera_inputs, profile):
    rgbd, calibration = camera_inputs
    with pytest.raises(ValueError):
        build_observation(rgbd, calibration, 0.0, profile=profile)


def test_vision_proprio_requires_an_explicit_proprio_provider(camera_inputs):
    rgbd, calibration = camera_inputs
    with pytest.raises(ValueError):
        build_observation(rgbd, calibration, 0.0, profile="vision_proprio")


def test_user_supplied_scene_mode_false_claim(camera_inputs):
    """Keep the research facade's unsupported physics capabilities explicit."""
    rgbd, calibration = camera_inputs
    env = CuttingEnv(EnvConfig(case="settle", substeps=1, max_steps=1))
    # A CPU stand-in exercises the real facade and observation path. It does
    # not validate Newton physics, cameras, or articulated USD execution.
    env.bed = SimpleNamespace(
        state=SimpleNamespace(particle_q=np.zeros((2, 3))),
        model=SimpleNamespace(particle_radius=np.ones(2)),
        half_extents=np.array([0.001, 0.025, 0.010]),
        snapshot=lambda: (np.zeros((2, 3)), np.zeros((2, 3))),
        step=lambda *args: (
            np.array([100.0, 200.0, 300.0]),
            {"max_projection_correction_m": 0.001},
        ),
    )
    env.cameras = SimpleNamespace(
        calibration=calibration, capture=lambda *args: rgbd
    )
    env.time_s = 0.0
    env.trajectory = reference_trajectory("settle")
    env.tool_pose = env.trajectory.at(0.0)[0]

    observation, reward, terminated, truncated, info = env.step()

    assert info["capabilities"]["persistent_cutting"] is False
    assert info["capabilities"]["d455_rtx"] is False
    assert info["capabilities"]["articulated_robots"] is False
    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert set(observation) == {
        "rgb", "depth_m", "valid_mask", "calibration", "timestamp_s"
    }
    assert observation["timestamp_s"] == env.config.control_dt
    np.testing.assert_array_equal(env.render(), rgbd["global"]["rgb"])
    env.close()
