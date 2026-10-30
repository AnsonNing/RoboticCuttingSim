"""Example external policies for the policy-ready API and the dataset collector (WP-11 Phase G).

A policy factory has the signature ``factory(task_spec: dict, seed: int) -> policy`` where
``policy(observation) -> action`` uses only the deployable observation and the
a-priori ``task_spec`` (``PolicyReadyEnv.task_spec()``). These are interface
examples, not trained or useful cutting policies.
"""
import numpy as np


def random_delta_policy(task_spec, seed, *, invalid_every=37, scale=1.):
    """Uniform bounded world deltas (cut disarmed) plus a deliberately invalid action every ``invalid_every`` steps.

    The invalid actions (unit-less quaternion, out-of-range cut index, unknown key) exercise the explicit
    REJECTED_* statuses; everything else is ACCEPTED or CLAMPED by the env's action limits.
    """
    rng = np.random.default_rng(int(seed))
    limits = task_spec["action_limits"]
    step = {"n": 0}
    invalid = ({"type": "edge_pose", "target": [0., 0., 0., 0., 0., 0., 2.]},
               {"type": "edge_delta", "target": [0.]*6, "cut_index": int(task_spec["max_cuts"])},
               {"type": "edge_delta", "target": [0.]*6, "speed": 1.})

    def policy(observation):
        step["n"] += 1
        if invalid_every and step["n"] % invalid_every == 0:
            return dict(invalid[(step["n"]//invalid_every) % len(invalid)])
        delta = np.r_[rng.uniform(-1, 1, 3)*limits["max_translation_m"]*scale*1.5,
                      rng.uniform(-1, 1, 3)*limits["max_rotation_rad"]*scale*.5]
        return {"type": "edge_delta", "target": delta.tolist(), "frame": "world", "cut_enabled": False}
    return policy


def hold_policy(task_spec, seed):
    """Zero world deltas with cutting disarmed (a do-nothing baseline that must FAIL any cutting goal)."""
    def policy(observation):
        return {"type": "edge_delta", "target": [0.]*6, "frame": "world", "cut_enabled": False}
    return policy
