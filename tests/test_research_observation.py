import numpy as np
import pytest

from robotic_cutting_sim.research_env import ResearchEnvConfig, ResearchCuttingEnv, vision_observation


def payload():
    frame = {"rgb": np.zeros((3, 4, 3), np.uint8), "depth_m": np.ones((3, 4), np.float32),
             "surface_timestamp_s": .02, "force": np.ones(6), "particle_oracle": np.ones((3, 3))}
    frames = {name: frame for name in ("d455_0", "d455_1", "global")}
    proprio = {"q": np.zeros((2, 7)), "qdot": np.zeros((2, 7)), "tool_pose": np.r_[np.zeros(6), 1.],
               "timestamp_s": .02, "wrench": np.ones(6)}
    return frames, {n: {} for n in frames}, proprio


def test_real_vision_observation_excludes_force_oracle_global():
    frames, calibration, proprio = payload()
    obs = vision_observation(frames, calibration, .02, "vision_proprio", proprio)
    assert set(obs) == {"rgb", "depth_m", "valid_mask", "calibration", "timestamp_s", "proprio"}
    assert set(obs["proprio"]) == {"q", "qdot", "tool_pose"}
    assert set(obs["rgb"]) == {"d455_0", "d455_1"}
    assert obs["proprio"]["q"].shape == (2, 7)


def test_stale_surface_and_stale_robot_are_rejected():
    frames, calibration, proprio = payload()
    with pytest.raises(ValueError, match="timestamps"):
        vision_observation(frames, calibration, .03, "vision_proprio", proprio)
    proprio["timestamp_s"] = 0.
    with pytest.raises(ValueError, match="timestamps"):
        vision_observation(frames, calibration, .02, "vision_proprio", proprio)


def test_d455_mode_cannot_silently_fall_back_to_ideal():
    with pytest.raises(NotImplementedError, match="registration"):
        ResearchCuttingEnv(ResearchEnvConfig(camera_mode="simulated_d455"))


def test_vision_only_has_no_proprioception():
    frames, calibration, proprio = payload()
    assert "proprio" not in vision_observation(frames, calibration, .02, "vision_only", proprio)


def test_wrist_names_and_optional_global_never_expose_debug_state():
    frames, calibration, proprio = payload()
    renamed = dict(zip(("wrist_d405_0", "wrist_d405_1", "global_d455"), frames.values()))
    calibration = {name: {"timestamp_s": .02, "sensor_model": "ideal_d405_prototype"} for name in renamed}
    observation = vision_observation(renamed, calibration, .02, "vision_proprio", proprio)
    assert set(observation["rgb"]) == {"wrist_d405_0", "wrist_d405_1"}
    observation = vision_observation(renamed, calibration, .02, "vision_proprio", proprio, include_global_camera=True)
    assert set(observation["rgb"]) == set(renamed)
    assert "force" not in observation and "particle_oracle" not in observation
    calibration["wrist_d405_0"]["timestamp_s"] = .01
    with pytest.raises(ValueError, match="Calibration"):
        vision_observation(renamed, calibration, .02, "vision_proprio", proprio)


@pytest.mark.parametrize("recorded_time", [.01, float("nan")])
@pytest.mark.parametrize("source", ["frame", "calibration"])
def test_native_physics_timestamp_must_match_current_surface_and_robot(recorded_time, source):
    frames, calibration, proprio = payload()
    container = frames if source == "frame" else calibration
    container["d455_0"]["physics_timestamp_s"] = recorded_time
    with pytest.raises(ValueError, match="timestamps must match"):
        vision_observation(frames, calibration, .02, "vision_proprio", proprio)
