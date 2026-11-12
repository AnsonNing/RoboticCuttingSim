"""Scripted controllers of the P2-6 rubber task (RL-05 sanity; deployable observation only, never privileged state).

Every controller reads only the 78-dim ``policy`` observation of ``rubber_strip_drag_v*`` (grip pose, the 9 marker
positions / velocities, goal, preload flag) and returns ``grip_delta_v1`` actions ``[N, 6]`` (float32). They are batched
torch code (one call for all envs) and keep a small per-env state that ``reset(env_ids)`` clears.

- ``rubber_scripted_drag_v1`` (``scripted_drag_v1``): a[0:2] = clip((goal - tip) / max_translation, -1, 1), the rest 0.
  The RL-05 controller of ``rubber_rl_tests_v1`` / ``_v2`` (v2 RL-05 FAIL: it ignores the strip's rotation).
- ``rubber_scripted_pivot_v1`` (``scripted_pivot_v1`` -> ``ScriptedTowV1(mode="pivot_grip", k_radial=0.5, k_yaw=0.5,
  k_fine=0.5, r_fine=0.015, eps=0.004)``): the RL-05 controller declared by ``rubber_rl_tests_v3``. The strip is stiff
  in its own plane and lies on the table with friction: dragged sideways it rotates about its tip, pushed it ruckles,
  but rotated about the grip it sweeps its tip. Per step: the grip yaw psi_g (from the observed grip quaternion) is
  turned toward the bearing of the goal from the grip at k_yaw x the heading error (0.05 rad per step at most), and the
  grip moves along its own x axis by k_radial x (rho - l) (rho = |goal - grip|, l = |marker 8 - marker 0|, the strip's
  current length); within r_fine of the goal the grip only translates by k_fine x (goal - tip) and keeps its yaw;
  within eps nothing moves. The heading loop never closes through the strip's own (whipping) heading. Development:
  ``scripts/rubber/p2_6/diag_rl05_controller.py`` (DIAGNOSTIC, coarse CPU stand-in, seeds 5000-5011).
- ``ScriptedTowV1`` modes ``tow`` / ``push`` / ``hybrid`` / ``pivot``: development candidates (not declared).

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
                 k_push=0.5, k_yaw=0.5, c_pull=0.5, c_push=0.5, eps=0.004, workspace_xy=None, margin=0.005,
                 k_radial=0.5, k_fine=0.5, r_fine=0.015):
        if mode not in ("tow", "push", "hybrid", "pivot", "pivot_grip"):
            raise ValueError(f"mode {mode!r}")
        self.N, self.device, self.mode = num_envs, device, mode
        self.max_translation, self.max_rotation = float(max_translation), float(max_rotation)
        self.k_tow, self.k_side, self.k_push, self.k_yaw = k_tow, k_side, k_push, k_yaw
        self.c_pull, self.c_push, self.eps = c_pull, c_push, eps
        # workspace_xy: ((x_lo, y_lo), (x_hi, y_hi)) of the grip origin in world coordinates (the task config's
        # declared box plus the grip origin's reference); the commanded grip target is kept ``margin`` inside it
        self.workspace_xy, self.margin = workspace_xy, float(margin)
        self.k_radial, self.k_fine, self.r_fine = k_radial, k_fine, r_fine

    def reset(self, env_ids=None):
        """Stateless (kept for the controller interface)."""

    def params(self):
        return {"id": "rubber_scripted_pivot_v1" if self.mode == "pivot_grip" else f"scripted_dev_{self.mode}",
                "mode": self.mode, "k_tow": self.k_tow, "k_side": self.k_side,
                "k_push": self.k_push, "k_yaw": self.k_yaw, "c_pull": self.c_pull, "c_push": self.c_push,
                "eps_m": self.eps, "workspace_xy": self.workspace_xy, "margin_m": self.margin,
                "k_radial": self.k_radial, "k_fine": self.k_fine, "r_fine_m": self.r_fine}

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
        if self.mode == "pivot_grip":
            # pivot on the GRIP heading: yaw the grip so that its x axis points at the goal, move the grip along
            # its own x axis by the radial error (rho - l); near the goal small translations by the tip error. The
            # loop never closes through the (whipping) strip heading
            q = obs[:, GRIP_Q].double()
            psi_g = torch.atan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2))
            ug = torch.stack((torch.cos(psi_g), torch.sin(psi_g)), 1)
            bearing = g-grip
            rho = torch.linalg.vector_norm(bearing, dim=1, keepdim=True)
            psi_bear = torch.atan2(bearing[:, 1], bearing[:, 0])
            fine = dist <= self.r_fine
            move = torch.where(fine, self.k_fine*e, self.k_radial*(rho-ell)*ug)
            psi = psi_g
            psi_star = torch.where(fine[:, 0], psi_g, psi_bear)
        elif self.mode == "pivot":
            # pivot: point the strip at the goal from the grip (yaw about the grip sweeps the tip), fix the radial
            # error by moving the grip along the strip axis; near the goal small translations by the tip error
            bearing = g-grip
            rho = torch.linalg.vector_norm(bearing, dim=1, keepdim=True)
            psi_bear = torch.atan2(bearing[:, 1], bearing[:, 0])
            radial = self.k_radial*(rho-ell)*u
            fine = dist <= self.r_fine
            move = torch.where(fine, self.k_fine*e, radial)
            psi_star = torch.where(fine[:, 0], psi, psi_bear)
        elif self.mode == "tow":
            move, psi_star = tow_move, psi_tow
        elif self.mode == "push":
            move, psi_star = push_move, psi_push
        else:                                  # hybrid: push when the strip already points at the goal, else tow
            pushing = c >= self.c_push
            move = torch.where(pushing, push_move, tow_move)
            psi_star = torch.where(pushing[:, 0], psi_push, psi_tow)
        active = dist > self.eps
        move = torch.where(active, move, torch.zeros_like(move))
        if self.workspace_xy is not None:                 # keep the commanded grip inside the declared box
            lo = torch.tensor(self.workspace_xy[0], dtype=f64, device=obs.device)+self.margin
            hi = torch.tensor(self.workspace_xy[1], dtype=f64, device=obs.device)-self.margin
            move = torch.minimum(torch.maximum(grip+move, lo), hi)-grip
        dpsi = torch.where(active[:, 0], _wrap(psi_star-psi), torch.zeros_like(psi))
        a = torch.zeros((obs.shape[0], 6), dtype=f64, device=obs.device)
        a[:, :2] = move/self.max_translation
        a[:, 5] = self.k_yaw*dpsi/self.max_rotation
        return a.clamp(-1.0, 1.0).to(torch.float32)


class ScriptedPivotHoldV1:
    """``rubber_scripted_pivot_hold_v1``: the v3 pivot approach plus a latched hold phase (rubber_rl_tests_v4).

    Deployable observation only (grip pose, markers, marker velocities, goal). State per env: ``hold`` (latched).

    - Approach (not holding): exactly ``rubber_scripted_pivot_v1`` without its fine zone: grip yaw toward the goal
      bearing (k_yaw), grip translation along its own axis by k_radial (rho - l).
    - Latch: ``hold`` becomes True when the tip distance falls to ``r_in`` and stays True until it exceeds ``r_out``
      (hysteresis; the v3 controller switched its yaw loop on and off at one radius, which re-excited the swing).
    - Hold: no yaw; the grip translates by k_hold (e - k_v v_tip dt) where e = goal - tip and v_tip is the observed tip
      marker velocity (damping: the tip's own motion over the next control step is subtracted from the correction), and
      not at all while |e - k_v v_tip dt| <= ``deadband``.

    Every parameter may be a float or a per-env list (development batches); the declared controller uses floats."""

    NAMES = ("k_radial", "k_yaw", "r_in", "r_out", "k_hold", "k_v", "deadband")

    def __init__(self, num_envs, device, max_translation, max_rotation, control_dt=0.02, *, k_radial=0.5,
                 k_yaw=0.5, r_in=0.008, r_out=0.03, k_hold=0.3, k_v=1.0, deadband=0.003):
        import torch
        self.N, self.device = num_envs, device
        self.max_translation, self.max_rotation, self.dt = float(max_translation), float(max_rotation), float(control_dt)
        self.raw = dict(k_radial=k_radial, k_yaw=k_yaw, r_in=r_in, r_out=r_out, k_hold=k_hold, k_v=k_v,
                        deadband=deadband)
        for k, v in self.raw.items():
            t = torch.as_tensor(v, dtype=torch.float64, device=device).reshape(-1, 1)
            setattr(self, k, t.expand(num_envs, 1) if t.shape[0] == 1 else t)
        self.hold = torch.zeros((num_envs, 1), dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.hold.zero_()
        else:
            self.hold[list(env_ids)] = False

    def params(self):
        return {"id": "rubber_scripted_pivot_hold_v1", **{k: (v if not hasattr(v, "__len__") else list(v))
                                                          for k, v in self.raw.items()}}

    def __call__(self, obs):
        import torch
        f64 = torch.float64
        m = _markers(obs)
        tip = m[:, 8, :2]
        v_tip = obs[:, 47+24:47+26].double()
        ell = torch.linalg.vector_norm(m[:, 8, :2]-m[:, 0, :2], dim=1, keepdim=True)
        g = obs[:, GOAL][:, :2].double()
        grip = obs[:, GRIP_P][:, :2].double()
        e = g-tip
        dist = torch.linalg.vector_norm(e, dim=1, keepdim=True)
        preload = obs[:, PRELOAD:PRELOAD+1] > 0.5
        self.hold = torch.where(preload, torch.zeros_like(self.hold),
                                torch.where(dist <= self.r_in, torch.ones_like(self.hold),
                                            torch.where(dist > self.r_out, torch.zeros_like(self.hold), self.hold)))
        q = obs[:, GRIP_Q].double()
        psi_g = torch.atan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2))
        ug = torch.stack((torch.cos(psi_g), torch.sin(psi_g)), 1)
        bearing = g-grip
        rho = torch.linalg.vector_norm(bearing, dim=1, keepdim=True)
        approach = self.k_radial*(rho-ell)*ug
        dpsi = _wrap(torch.atan2(bearing[:, 1], bearing[:, 0])-psi_g)[:, None]*self.k_yaw
        corr = e-self.k_v*v_tip*self.dt
        hold_move = torch.where(torch.linalg.vector_norm(corr, dim=1, keepdim=True) > self.deadband,
                                self.k_hold*corr, torch.zeros_like(corr))
        move = torch.where(self.hold, hold_move, approach)
        yaw = torch.where(self.hold, torch.zeros_like(dpsi), dpsi)
        a = torch.zeros((obs.shape[0], 6), dtype=f64, device=obs.device)
        a[:, :2] = move/self.max_translation
        a[:, 5:6] = yaw/self.max_rotation
        return a.clamp(-1.0, 1.0).to(torch.float32)


PIVOT_V1 = {"mode": "pivot_grip", "k_radial": 0.5, "k_yaw": 0.5, "k_fine": 0.5, "r_fine": 0.015, "eps": 0.004}


def scripted_pivot_v1(num_envs, device, max_translation, max_rotation):
    """``rubber_scripted_pivot_v1`` (rubber_rl_tests_v3 RL-05): ``ScriptedTowV1`` with the declared ``PIVOT_V1``."""
    return ScriptedTowV1(num_envs, device, max_translation, max_rotation, **PIVOT_V1)


__all__ = ["scripted_drag_v1", "ScriptedTowV1", "scripted_pivot_v1", "PIVOT_V1", "ScriptedPivotHoldV1"]
