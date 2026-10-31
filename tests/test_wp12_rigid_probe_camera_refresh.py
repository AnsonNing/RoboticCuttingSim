"""WP-12 Phase 2b: opt-in wrist-camera extrinsic refresh in ``rigid_probe_step`` (CPU, no Isaac).

Phase 2a found that ``capture()`` after rigid-only controls fails the runner's own
wrist-camera extrinsic check, because extrinsics are recomposed only in coupled
``step()`` calls. ``rigid_probe_step(..., refresh_camera_extrinsics=True)`` recomposes
them after the rigid step; the default (False) keeps the historical behaviour.
"""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim.isaac_sim_runner import LiveIsaacSimCutting


class _Arms:
    def __init__(self):
        self.targets = []

    def set_dof_position_targets(self, target):
        self.targets.append(np.array(target))

    def set_dof_velocity_targets(self, target):
        pass

    def set_dof_efforts(self, efforts):
        pass

    def get_dof_gravity_compensation_forces(self):
        return np.zeros((2, 9))


def _fake(camera=True, layout="wrist_d405_global_d455", specs=True):
    calls = {"extrinsics": [], "sim_steps": 0}
    hands = np.tile(np.r_[0., 0., .5, 0., 0., 0., 1.], (2, 1))
    env = SimpleNamespace()
    env.physical_grasp = True
    env._rigid_probe_only = False
    env.previous_command_q = np.zeros((2, 9))
    env.ik = SimpleNamespace(_validate_joints=lambda q: np.asarray(q, float))
    env.gripper_command = "hold"
    env.grasp = SimpleNamespace(closed_target_m=.01)
    env.arms = _Arms()
    env.config = SimpleNamespace(robot_substeps=2, control_dt=.02, camera=camera, camera_layout=layout)
    env.external_tool_wrench = np.zeros(6)
    env.tool_body = SimpleNamespace(apply_forces_and_torques_at_pos=lambda *a, **k: None)
    env.SimulationManager = SimpleNamespace(step=lambda: calls.__setitem__("sim_steps", calls["sim_steps"]+1))
    env.grasp_trace = []
    env._grasp_sample = lambda phase, wrench, state: env.grasp_trace.append({"phase": phase})
    env.step_index = 0
    env.time_s = 0.
    env.physx = SimpleNamespace(update_transformations=lambda *a: None)
    env.proprioception = lambda: {"tool_pose": np.r_[0., 0., .3, 0., 0., 0., 1.], "hand_pose_both_xyzw": hands}
    env.camera_specs = {"wrist_d405_0": {}} if specs else {}
    env._update_camera_extrinsics = lambda poses: calls["extrinsics"].append((env.time_s, np.array(poses)))
    return env, calls


def _step(env, **kwargs):
    return LiveIsaacSimCutting.rigid_probe_step(env, np.zeros(7), **kwargs)


def test_default_is_unchanged_and_does_not_refresh_extrinsics():
    signature = inspect.signature(LiveIsaacSimCutting.rigid_probe_step)
    assert signature.parameters["refresh_camera_extrinsics"].default is False
    assert signature.parameters["refresh_camera_extrinsics"].kind is inspect.Parameter.KEYWORD_ONLY
    env, calls = _fake()
    _step(env)
    _step(env)
    assert calls["extrinsics"] == []
    assert calls["sim_steps"] == 4 and env.step_index == 2 and env._rigid_probe_only


def test_opt_in_refreshes_after_each_rigid_step_with_the_measured_hands():
    env, calls = _fake()
    _step(env, refresh_camera_extrinsics=True)
    _step(env, refresh_camera_extrinsics=True)
    assert [t for t, _ in calls["extrinsics"]] == pytest.approx([.02, .04])
    assert calls["extrinsics"][-1][1].shape == (2, 7)
    # Same rigid controls as the default path: same targets, same physics steps.
    reference, reference_calls = _fake()
    _step(reference)
    _step(reference)
    assert reference_calls["sim_steps"] == calls["sim_steps"]
    assert all(np.array_equal(a, b) for a, b in zip(reference.arms.targets, env.arms.targets))


@pytest.mark.parametrize("camera, layout, specs", [(False, "wrist_d405_global_d455", True),
                                                    (True, "fixed", True),
                                                    (True, "wrist_d405_global_d455", False)])
def test_opt_in_is_a_no_op_without_the_wrist_rig(camera, layout, specs):
    env, calls = _fake(camera, layout, specs)
    _step(env, refresh_camera_extrinsics=True)
    assert calls["extrinsics"] == []
