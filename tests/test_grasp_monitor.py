"""Episode loss and seating-offset contracts, without a runtime or simulated forces."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.grasp_monitor import GraspMonitor
from robotic_cutting_sim.physical_tool import make_tool_spec
from robotic_cutting_sim.scene_frames import matrix_pose, pose_matrix


def setup_monitor():
    spec = make_tool_spec()
    hand = np.array([.3, -.2, .4, 0., 0., 0., 1.])
    tool = matrix_pose(pose_matrix(hand) @ spec.T_hand_tool)
    tool[2] -= .0008  # settled seating offset is not ongoing slip
    return GraspMonitor(hand, tool, spec), hand, tool


def test_common_hand_and_tool_motion_is_not_slip():
    monitor, hand, tool = setup_monitor()
    motion = np.eye(4)
    motion[:3, :3] = Rotation.from_euler("xyz", [.2, -.1, .8]).as_matrix()
    motion[:3, 3] = [.02, .04, .06]
    state = monitor.update(matrix_pose(motion @ pose_matrix(hand)),
        matrix_pose(motion @ pose_matrix(tool)), time_s=.02, bilateral_loaded_tool_fraction=1.)
    assert state["status"] == "HOLDING"
    assert state["slip_since_settling_m"] < 1e-12
    assert state["nominal_grasp_offset_m"] == pytest.approx(.0008)


def test_release_and_drop_latch_until_reset():
    monitor, hand, tool = setup_monitor()
    state = monitor.update(hand, tool, time_s=.02, bilateral_loaded_tool_fraction=0., gripper_command="release")
    assert state["status"] == "RELEASING" and not state["grasp_lost"]
    dropped = tool.copy()
    dropped[2] -= .03
    state = monitor.update(hand, dropped, time_s=.1, bilateral_loaded_tool_fraction=0.)
    assert state["status"] == "LOST" and state["grasp_lost"]
    state = monitor.update(hand, tool, time_s=.12, bilateral_loaded_tool_fraction=1.)
    assert state["grasp_lost"]  # no accidental auto recovery or hidden reset
    assert GraspMonitor(hand, tool, make_tool_spec()).last["grasp_lost"] is False


def test_contact_failure_is_explicit_and_pose_alone_cannot_validate_hold():
    monitor, hand, tool = setup_monitor()
    state = monitor.update(hand, tool, time_s=.02, bilateral_loaded_tool_fraction=0., contacts_valid=False)
    assert state["status"] == "INVALID_CONTACT_DATA"
    state = monitor.update(hand, tool, time_s=.04, bilateral_loaded_tool_fraction=.9)
    assert state["status"] == "SLIPPING"


def test_sustained_loss_of_support_with_slip_terminates():
    monitor, hand, tool = setup_monitor()
    tool[2] -= .004
    for i in range(1, 7):
        state = monitor.update(hand, tool, time_s=i*.02, bilateral_loaded_tool_fraction=0.)
    assert state["grasp_lost"]


@pytest.mark.parametrize("time_s,fraction", [(0., 1.), (.02, -1.), (.02, np.nan)])
def test_reject_bad_timestamps_or_contact_fractions(time_s, fraction):
    monitor, hand, tool = setup_monitor()
    with pytest.raises(ValueError):
        monitor.update(hand, tool, time_s=time_s, bilateral_loaded_tool_fraction=fraction)
