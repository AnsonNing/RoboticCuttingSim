import numpy as np
import pytest

from robotic_cutting_sim.robot import RobotConfig, compose_tool_pose, validate_pose


def test_downward_hand_puts_blade_below_wrist():
    pose = compose_tool_pose([.2, -.1, .3, 1, 0, 0, 0], [0, 0, .18])
    np.testing.assert_allclose(pose[:3], [.2, -.1, .12], atol=1e-12)
    assert pose.shape == (7,)


def test_quaternion_sign_does_not_change_attached_blade_pose():
    a = compose_tool_pose([0, 0, 0, 0, 0, 2, 2], [.1, 0, .2])
    b = compose_tool_pose([0, 0, 0, 0, 0, -2, -2], [.1, 0, .2])
    np.testing.assert_allclose(a[:3], [0, .1, .2], atol=1e-12)
    np.testing.assert_allclose(a[:3], b[:3], atol=1e-12)


@pytest.mark.parametrize("pose", [[0] * 7, [0] * 6, [0, 0, np.nan, 0, 0, 0, 1]])
def test_robot_rejects_invalid_tool_pose(pose):
    with pytest.raises(ValueError):
        validate_pose(pose)


@pytest.mark.parametrize("kwargs", [{"backend": "fake"}, {"substeps": 0}, {"substeps": 2.5},
                                    {"control_dt": np.inf}, {"tool_half_extents_m": [1, 1, np.inf]}])
def test_robot_rejects_invalid_physics_configuration(kwargs):
    with pytest.raises(ValueError):
        RobotConfig(**kwargs)
