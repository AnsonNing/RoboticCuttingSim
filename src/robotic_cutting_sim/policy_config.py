"""Opt-in live policy preset; historical CLI/config defaults remain unchanged."""
from .control import ActionLimits, DELTA_MODE
from .physical_tool import ToolGraspConfig


def policy_preset(runtime):
    if runtime not in ('isaac_sim', 'isaac_lab'):
        raise ValueError('A policy preset requires an actual live runtime')
    options = dict(action_mode=DELTA_MODE, action_limits=ActionLimits(),
                   ik_failure='reject', delta_frame='world',
                   tool_attachment_mode='physical_grasp', tool_init_mode='pregrasped',
                   grasp=ToolGraspConfig(initial_finger_position_m=.0112))
    if runtime == 'isaac_lab':
        options.update(grip_contact_profile='finite_pad_patch_v3', ik_rotation_tolerance_rad=.01)
    return options
