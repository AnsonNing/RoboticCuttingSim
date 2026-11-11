"""RL-05 scripted controllers (rl.rubber_scripted): deployable observation only, batched, declared parameters."""
import math

import pytest

torch = pytest.importorskip("torch")

from robotic_cutting_sim.rl import rubber_scripted as RS  # noqa: E402


def _obs(grip, yaw, tip, base, goal, n=1):
    o = torch.zeros((n, 78), dtype=torch.float32)
    o[:, 0:3] = torch.tensor(grip)
    o[:, 3:7] = torch.tensor([0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2)])
    o[:, 20:23] = torch.tensor(base)
    o[:, 44:47] = torch.tensor(tip)
    o[:, 74:77] = torch.tensor(goal)
    return o


def test_declared_pivot_parameters():
    c = RS.scripted_pivot_v1(1, "cpu", 0.005, 0.05)
    assert c.params()["mode"] == "pivot_grip" and RS.PIVOT_V1 == {"mode": "pivot_grip", "k_radial": 0.5,
                                                                 "k_yaw": 0.5, "k_fine": 0.5, "r_fine": 0.015,
                                                                 "eps": 0.004}


def test_pivot_yaws_toward_a_lateral_goal_and_pulls_toward_a_near_one():
    c = RS.scripted_pivot_v1(1, "cpu", 0.005, 0.05)
    a = c(_obs((0, 0.008, 0.002), 0.0, (0.1, 0.008, 0.002), (0, 0.008, 0.002), (0.1, 0.06, 0.002)))
    assert a[0, 5] > 0.9                                    # goal to the +y side: yaw +, saturated
    a = c(_obs((0, 0.008, 0.002), 0.0, (0.1, 0.008, 0.002), (0, 0.008, 0.002), (0.07, 0.008, 0.002)))
    assert a[0, 0] < -0.9 and abs(float(a[0, 5])) < 1e-6    # goal nearer than the strip length: pull back
    a = c(_obs((0, 0.008, 0.002), 0.0, (0.1, 0.008, 0.002), (0, 0.008, 0.002), (0.105, 0.010, 0.002)))
    assert abs(float(a[0, 5])) == 0.0 and a[0, 0] > 0      # fine zone: translate by the tip error, keep the yaw
    a = c(_obs((0, 0.008, 0.002), 0.0, (0.1, 0.008, 0.002), (0, 0.008, 0.002), (0.101, 0.009, 0.002)))
    assert bool((a == 0).all())                             # inside eps: hold
    assert a.dtype == torch.float32 and bool((a.abs() <= 1).all())


def test_controllers_never_read_privileged_columns():
    o = _obs((0, 0.008, 0.002), 0.3, (0.1, 0.04, 0.002), (0, 0.008, 0.002), (0.12, 0.07, 0.002), n=2)
    for f in (RS.scripted_pivot_v1(2, "cpu", 0.005, 0.05), lambda x: RS.scripted_drag_v1(x, 0.005)):
        a = f(o)
        b = f(torch.cat((o, torch.full((2, 28), float("nan"))), 1)[:, :78])     # only the 78 policy columns exist
        assert torch.equal(a, b) and tuple(a.shape) == (2, 6)
