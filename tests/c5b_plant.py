"""Laggy knife-edge plant fitted to the WP-11 C5 live Native trace (CPU test helper, WP-11 Phase C5b).

Fit (scratch analysis of artifacts/wp11/phaseC5/live_trim_rho10_001/episode_000/actions.jsonl,
PRE_CUT .. HOLD_AFTER_STOP, 2.84-14.0 s, 559 steps): per WORLD axis an ARX(2, 3) model of the
measured edge m_k (after step k) on the accepted absolute command c_k (the target executed in
step k) plus a constant,

    m_k = a1 m_{k-1} + a2 m_{k-2} + b0 c_k + b1 c_{k-1} + b2 c_{k-2} + const,

one-step residual rms x 20.8, y 18.6, z 9.0 um (used here as white measurement noise on the
measured edge, an upper bound: the residual also holds model error). The b's are rescaled here to unit DC gain (fitted DC gains
1.0005 / 0.971 / 1.0003); the static command-minus-measured offset in the WITHDRAW hold was
(+0.062, -0.697, +0.332) mm. Rotation: first-order lag (coefficient 0.6 per step) with a constant
0.8 mrad offset; the nominal (hand FK) edge differs from the measured edge by the grasp offset.
Variants: an extra transport delay (steps) and a pose-dependent sag (offset z changes linearly with
the commanded height) probe robustness beyond the fitted model; ``floor_z`` (WP-11 review finding 8)
blocks the measured edge below a WORLD height (a blade stopped by an obstacle).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.control import DELTA_MODE, bound_target, resolve_delta

ARX = {  # axis: (a1, a2), (b0, b1, b2), residual rms (m)
    0: ((1.528, -.748), (.721, -.884, .383), 20.8e-6),
    1: ((1.610, -.710), (.213, .132, -.248), 18.6e-6),
    2: ((1.513, -.635), (.406, -.274, -.009), 9.0e-6),
}
STATIC_OFFSET_M = np.array([.062e-3, -.697e-3, .332e-3])   # command - measured at rest
GRASP_OFFSET_M = np.array([0., .7e-3, 0.])                  # nominal - measured
ROTATION_OFFSET = Rotation.from_rotvec([0., 8e-4, 0.])


class LaggyEdgePlant:
    """Mini env: commanded-pose bookkeeping as CuttingTaskEnv (delta composes on, absolute is bounded
    from, the previous command), then the fitted plant produces the measured edge."""

    def __init__(self, start_pose, limits, T_world_workpiece, *, seed=0, extra_delay_steps=0, sag_per_m=0.,
                 noise_scale=1., floor_z=None):
        self.limits, self.T = limits, T_world_workpiece
        self.rng = np.random.default_rng(seed)
        self.delay, self.sag_per_m, self.noise_scale = int(extra_delay_steps), float(sag_per_m), float(noise_scale)
        self.floor_z = None if floor_z is None else float(floor_z)
        start = np.asarray(start_pose, float)
        self.commanded = start.copy()
        self.z0 = float(start[2])
        self.inputs = [start[:3].copy() for _ in range(3+self.delay)]
        self.outputs = [start[:3]-self._offset(start[:3]) for _ in range(2)]
        self.rotation = Rotation.from_quat(start[3:])
        self.time_s, self.dt = 0., .02
        self.measured = self._measure()

    def _offset(self, command):
        return STATIC_OFFSET_M+np.array([0., 0., self.sag_per_m*(self.z0-float(command[2]))])

    def _measure(self):
        return np.r_[self.outputs[-1]+getattr(self, "noise", 0.), (ROTATION_OFFSET*self.rotation).as_quat()]

    def observation(self):
        nominal = self.measured.copy()
        nominal[:3] += GRASP_OFFSET_M
        return {"proprio": {"tool_pose": nominal}, "timestamp_s": self.time_s}

    def estimator(self, observation):
        return self.measured.copy()

    def step(self, action):
        if action["mode"] == DELTA_MODE:
            target, _ = resolve_delta(self.commanded, action["target"], self.limits, frame=action.get("frame", "world"),
                                      T_world_workpiece=self.T)
        else:
            target, _ = bound_target(self.commanded, action["target"], self.limits, self.T)
        self.commanded = target
        self.inputs.append(target[:3].copy())
        u = self.inputs[-1-self.delay], self.inputs[-2-self.delay], self.inputs[-3-self.delay]
        y = np.empty(3)
        for axis, ((a1, a2), b, rms) in ARX.items():
            b = np.asarray(b)*(1-a1-a2)/sum(b)  # unit DC gain
            shifted = [ui[axis]-self._offset(ui)[axis] for ui in u]
            y[axis] = a1*self.outputs[-1][axis]+a2*self.outputs[-2][axis]+b@shifted
        if self.floor_z is not None:
            y[2] = max(y[2], self.floor_z)
        self.outputs.append(y)
        self.noise = self.noise_scale*np.array([ARX[a][2] for a in range(3)])*self.rng.standard_normal(3)
        relative = (Rotation.from_quat(target[3:])*self.rotation.inv()).as_rotvec()
        self.rotation = Rotation.from_rotvec(.6*relative)*self.rotation
        self.time_s = round(self.time_s+self.dt, 9)
        self.measured = self._measure()
        return self.observation()


def rollout(task, plan, policy_options, *, max_s=10., stop_after=None, **plant_options):
    """(rows, policy): rows of (time, phase, command, measured) until the policy leaves ``stop_after``."""
    from robotic_cutting_sim.policies import WaypointPolicy
    limits = task.action_limits()
    plant = LaggyEdgePlant(plan[0].pose, limits, task.T_world_workpiece, **plant_options)
    policy = WaypointPolicy(task, list(plan), limits, edge_estimator=plant.estimator, **policy_options)
    observation, rows, seen = plant.observation(), [], False
    for _ in range(int(round(max_s/plant.dt))):
        action = policy(observation)
        phase = policy.phase
        seen |= stop_after is not None and phase.split(":")[0] == stop_after
        if policy.done or (seen and phase.split(":")[0] != stop_after):
            break
        observation = plant.step(action)
        rows.append(dict(time_s=plant.time_s, phase=phase, command=plant.commanded.copy(), measured=plant.measured.copy(),
                         cut=bool(action["cut_enabled"]),
                         tracking=None if policy.last_tracking is None else dict(policy.last_tracking)))
    return rows, policy
