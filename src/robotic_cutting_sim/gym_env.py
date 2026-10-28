"""Gymnasium adapter over CuttingTaskEnv for RL (state and/or image observations).

Actions are normalized relative edge motions in [-1, 1]^6, scaled by the
wrapped environment's ActionLimits (default 2 mm / 2 deg per 20 ms step), with
cutting armed. Observations are either a flat measured-proprioception vector
("state") or a Dict of that vector plus wrist RGB-D ("vision_proprio"). The
constructor performs one reset to discover array shapes. gymnasium is an
optional dependency of this module only.
"""
import numpy as np

from .control import DELTA_MODE, ActionLimits, normalized_delta

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only without gymnasium
    gym, spaces = None, None

STATE_KEYS = ("q", "qdot", "hand_pose", "finger_q", "finger_qdot", "tool_pose")


def flatten_proprio(proprio):
    """Fixed-order float32 vector from the deployable proprio allowlist."""
    return np.concatenate([np.asarray(proprio[k], np.float32).ravel() for k in STATE_KEYS if k in proprio])


if gym is not None:
    class CuttingGymEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self, env, *, cut_enabled=True, seed=None):
            if env.observation_profile not in ("state", "vision_proprio"):
                raise ValueError("RL adapter supports the state and vision_proprio profiles")
            if env.action_limits is None:
                env.action_limits = ActionLimits()
            self.env, self.cut_enabled = env, bool(cut_enabled)
            self.action_space = spaces.Box(-1., 1., (6,), np.float32)
            observation, _ = self.env.reset(seed=seed)
            self.observation_space = self._space(observation)

        def _space(self, observation):
            state = flatten_proprio(observation["proprio"])
            state_space = spaces.Box(-np.inf, np.inf, state.shape, np.float32)
            if self.env.observation_profile == "state":
                return state_space
            items = {"state": state_space}
            for name, rgb in observation["rgb"].items():
                items[name+"_rgb"] = spaces.Box(0, 255, rgb.shape, np.uint8)
                depth = observation["depth_m"][name]
                items[name+"_depth_m"] = spaces.Box(0., np.inf, depth.shape, np.float32)
            return spaces.Dict(items)

        def _convert(self, observation):
            state = flatten_proprio(observation["proprio"])
            if self.env.observation_profile == "state":
                return state
            converted = {"state": state}
            for name, rgb in observation["rgb"].items():
                converted[name+"_rgb"] = np.asarray(rgb, np.uint8)
                depth = np.asarray(observation["depth_m"][name], np.float32)
                converted[name+"_depth_m"] = np.where(np.isfinite(depth), depth, 0.).astype(np.float32)
            return converted

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            observation, info = self.env.reset(seed=seed)
            return self._convert(observation), info

        def step(self, action):
            delta = normalized_delta(np.asarray(action, float), self.env.action_limits)
            observation, reward, terminated, truncated, info = self.env.step(
                {"mode": DELTA_MODE, "target": delta, "cut_enabled": self.cut_enabled})
            return self._convert(observation), float(reward), bool(terminated), bool(truncated), info

        def render(self):
            return self.env.render()

        def close(self):
            self.env.close()
