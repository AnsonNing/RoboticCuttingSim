"""Command routing contracts only; actual WebRTC is verified separately."""
from types import SimpleNamespace

import numpy as np

from robotic_cutting_sim.remote_control import RemoteControlLoop
from robotic_cutting_sim.tasks import make_task


def controller():
    env = SimpleNamespace(task=make_task("CUT-01"), time_s=0., actions=[])
    def step(action):
        env.actions.append(action)
        env.time_s += env.task.control_dt_s
    def reset(task_id):
        env.task = make_task(task_id)
        env.actions.clear()
        env.time_s = 0.
    env.step = step
    env.evaluate = lambda: {"status": "NOT_RUN"}
    cameras, events = [], []
    loop = RemoteControlLoop(env, reset, set_camera=cameras.append, on_event=events.append)
    return loop, cameras, events


def test_pause_never_advances_and_step_is_one_coupled_interval():
    loop, _, _ = controller()
    for _ in range(20):
        loop.tick()
    assert loop.env.time_s == 0.
    loop.request("step"); loop.tick()
    assert np.isclose(loop.env.time_s, .02) and not loop.playing
    loop.tick()
    assert len(loop.env.actions) == 1
    loop.request("play"); loop.tick(); loop.tick()
    loop.request("pause"); loop.tick(); loop.tick()
    assert len(loop.env.actions) == 3


def test_bad_action_is_rejected_before_physics_and_reset_switches_shared_task():
    loop, _, events = controller()
    loop.request("action", mode="joint_position_rad", target="[0, 0]", cut_enabled=False)
    loop.tick()
    assert not loop.env.actions and events[-1]["status"] == "REJECTED"
    loop.request("reset", task_id="TRIM-01"); loop.tick()
    assert loop.env.task.task_id == "TRIM-01" and loop.env.time_s == 0.
    loop.request("action", mode="cartesian_edge_pose_world", target="[0,0,0.1,0,0,0,1]", cut_enabled=False)
    loop.tick()
    assert len(loop.env.actions) == 1 and not loop.env.actions[0]["cut_enabled"]


def test_camera_selection_and_evaluation_do_not_advance_physics():
    loop, cameras, _ = controller()
    loop.request("camera", camera="wrist_d405_1"); loop.tick()
    loop.request("evaluate"); loop.tick()
    assert cameras == ["wrist_d405_1"] and loop.env.time_s == 0.
    assert loop.snapshot()["evaluation"]["status"] == "NOT_RUN"


def test_physical_gripper_command_is_same_loop_and_legacy_release_is_rejected():
    loop,_,events=controller()
    loop.env.backend=SimpleNamespace(config=SimpleNamespace(tool_attachment_mode='legacy_fixed_tool'))
    loop.request('gripper',mode='release');loop.tick()
    assert not loop.env.actions and events[-1]['status']=='REJECTED'
    loop.env.backend.config.tool_attachment_mode='physical_grasp'
    loop.request('gripper',mode='release');loop.tick()
    assert loop.env.actions==[{'gripper':'release'}]
    assert loop.env.time_s==.02
