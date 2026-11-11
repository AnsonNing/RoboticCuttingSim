"""Scripted controllers of the P2-6 rubber task (RL-05 sanity; deployable observation only, never privileged state).

Every controller reads only the 78-dim ``policy`` observation of ``rubber_strip_drag_v*`` (grip pose, the 9 marker
positions / velocities, goal, preload flag) and returns ``grip_delta_v1`` actions ``[N, 6]`` (float32). They are batched
torch code (one call for all envs) and keep a small per-env state that ``reset(env_ids)`` clears.

- ``rubber_scripted_drag_v1`` (``scripted_drag_v1``): a[0:2] = clip((goal - tip) / max_translation, -1, 1), the rest 0.
  The RL-05 controller of ``rubber_rl_tests_v1`` / ``_v2`` (v2 RL-05 FAIL: it ignores the strip's rotation).
- ``rubber_scripted_tow_v1`` (``ScriptedTowV1``): see its docstring (declared by ``rubber_rl_tests_v3``).

Observation layout (``rl.rubber_env.POLICY_TERMS``): grip position 0:3, grip quaternion 3:7, twist 7:13, time 13,
previous action 14:20, markers 20:47 (marker k at 20 + 3k), marker velocities 47:74, goal 74:77, preload flag 77.
"""
import math

GRIP_P = slice(0, 3)
GRIP_Q = slice(3, 7)
MARKERS = slice(20, 47)
GOAL = slice(74, 77)
PRELOAD = 77


def _markers(obs):
    return obs[:, MARKERS].double().reshape(-1, 9, 3)


def scripted_drag_v1(obs, max_translation):
    """rubber_scripted_drag_v1 (v1 / v2 protocols)."""
    import torch
    tip = obs[:, 20+24:20+27].double()
    a = torch.zeros((obs.shape[0], 6), dtype=torch.float32, device=obs.device)
    a[:, :2] = ((obs[:, GOAL][:, :2].double()-tip[:, :2])/max_translation).clamp(-1, 1).float()
    return a


def _wrap(a):
    import torch
    return torch.remainder(a+math.pi, 2*math.pi)-math.pi


class ScriptedTowV1:
    """``rubber_scripted_tow_v1``: tow the strip so that its tip trails the grip into the goal.

    The strip is stiff in its own plane (W = 16 mm >> t), so on the table it behaves like a trailer hitched at the
    grip: pulled, it follows the grip; pushed along its axis, it ruckles and swings. Per policy step, with the tip
    x_tip = marker 8, the strip axis u = (marker 8 - marker 0) / |...| in the table plane (heading psi), the strip length
    l = |marker 8 - marker 0| and the goal g (all from the observation):

    1. aim point: p* = g + l * d, d = unit(g - x_tip) if |g - x_tip| > ``eps`` (the grip passes the goal on the tip's
       approach line and stops one strip length beyond it, so the tip trails into g), else p* = grip (hold);
    2. towing check: c = u . d (cos of the angle between the strip axis and the approach direction). If c > -``c_pull``
       the strip is not yet trailing (pushing / sideways): the grip yaws toward psi* = atan2(-d) (the trailing
       heading) at rate ``k_yaw`` and moves sideways only (the component of p* - grip normal to u, gain ``k_side``), which
       swings the strip about its tip instead of pushing it;
    3. trailing (c <= -``c_pull``): the grip moves toward p* with gain ``k_tow`` (fraction of the remaining distance per
       step), yaw toward psi* at ``k_yaw``;
    4. actions clipped to [-1, 1]; z and roll / pitch 0 (the jaw height and tilt of the preload are kept).

    Parameters are fixed at construction (declared values in ``rubber_rl_tests_v3``)."""

    def __init__(self, num_envs, device, max_translation, max_rotation, *, mode="hybrid", k_tow=0.5, k_side=0.5,
                 k_push=0.5, k_yaw=0.5, c_pull=0.5, c_push=0.5, eps=0.004):
        if mode not in ("tow", "push", "hybrid"):
            raise ValueError(f"mode {mode!r}")
        self.N, self.device, self.mode = num_envs, device, mode
        self.max_translation, self.max_rotation = float(max_translation), float(max_rotation)
        self.k_tow, self.k_side, self.k_push, self.k_yaw = k_tow, k_side, k_push, k_yaw
        self.c_pull, self.c_push, self.eps = c_pull, c_push, eps

    def reset(self, env_ids=None):
        """Stateless (kept for the controller interface)."""

    def params(self):
        return {"id": "rubber_scripted_tow_v1", "mode": self.mode, "k_tow": self.k_tow, "k_side": self.k_side,
                "k_push": self.k_push, "k_yaw": self.k_yaw, "c_pull": self.c_pull, "c_push": self.c_push,
                "eps_m": self.eps}

    def __call__(self, obs):
        import torch
        f64 = torch.float64
        m = _markers(obs)
        tip = m[:, 8, :2]
        axis = m[:, 8, :2]-m[:, 0, :2]
        ell = torch.linalg.vector_norm(axis, dim=1, keepdim=True).clamp(min=1e-6)
        u = axis/ell
        g = obs[:, GOAL][:, :2].double()
        grip = obs[:, GRIP_P][:, :2].double()
        e = g-tip
        dist = torch.linalg.vector_norm(e, dim=1, keepdim=True)
        d = e/dist.clamp(min=1e-9)
        c = (u*d).sum(1, keepdim=True)
        normal = torch.stack((-u[:, 1], u[:, 0]), 1)
        psi = torch.atan2(u[:, 1], u[:, 0])
        psi_tow = torch.atan2(-d[:, 1], -d[:, 0])
        psi_push = torch.atan2(d[:, 1], d[:, 0])
        # towing: the grip leads one strip length beyond the goal on the tip's approach line
        to_target = g+ell*d-grip
        tow_move = torch.where(c <= -self.c_pull, self.k_tow*to_target,
                               self.k_side*(to_target*normal).sum(1, keepdim=True)*normal)
        # pushing: the grip moves by the tip error (the strip roughly along the approach line)
        push_move = self.k_push*e
        if self.mode == "tow":
            move, psi_star = tow_move, psi_tow
        elif self.mode == "push":
            move, psi_star = push_move, psi_push
        else:                                  # hybrid: push when the strip already points at the goal, else tow
            pushing = c >= self.c_push
            move = torch.where(pushing, push_move, tow_move)
            psi_star = torch.where(pushing[:, 0], psi_push, psi_tow)
        active = dist > self.eps
        move = torch.where(active, move, torch.zeros_like(move))
        dpsi = torch.where(active[:, 0], _wrap(psi_star-psi), torch.zeros_like(psi))
        a = torch.zeros((obs.shape[0], 6), dtype=f64, device=obs.device)
        a[:, :2] = move/self.max_translation
        a[:, 5] = self.k_yaw*dpsi/self.max_rotation
        return a.clamp(-1.0, 1.0).to(torch.float32)


__all__ = ["scripted_drag_v1", "ScriptedTowV1"]
