"""Batched policies and the measured-trajectory driver for workload A (outside the env, ADR 0021 §8).

* ``recorded_command_policy_v1`` (:class:`RecordedCommandPolicyV1`): replays the COMMANDED edge targets that a
  recorded WP-17 Native episode submitted (``actions.jsonl`` ``resolved_target_world_xyz_xyzw``, all 1200
  ``ACCEPTED``) as real ``edge_delta_v1`` actions through ``env.step``. The actions are computed open loop on
  the host (:func:`recorded_command_actions`): at each step the normalized edge-frame delta from the
  reconstructed commanded pose to the recorded target, cast to float32 (the action dtype), then pushed through
  the frozen NumPy path (``control.normalized_delta`` + ``control.resolve_delta(frame="edge")``) to obtain the
  next reconstructed commanded pose. Because every action is computed against the reconstructed pose, the
  float32 rounding does not accumulate. The reconstruction residual against the recorded targets is reported.
  Under ideal tracking the material sees these commanded poses, NOT the measured Native poses (the robot's
  tracking error is absent): a different material input than the Native episode.
* :class:`RandomPolicyV1`: uniform ``[-1,1]^6`` motion, arm with probability ``arm_probability``, uniform
  digit selector; counter-based (SplitMix64 of env seed, episode, step, component), so every env has its own
  reproducible stream independent of N and of the other envs; no per-env loop.
* :class:`MeasuredTrajectoryDriver` (privileged, not a policy): device tables of recorded measured edge poses,
  resolved cut digits and pad boxes, gathered per env by its own step count for ``env.step_measured_edge``.

Policies take ``(obs, info)`` with ``info["step_count"]`` (``[N]`` int64, steps already taken in the episode),
``info["seed"]`` and ``info["episode"]`` as returned by ``reset``/``step``; they never read material state.
"""
import json
from pathlib import Path

import numpy as np

POLICY_RECORDED = "recorded_command_policy_v1"
POLICY_RANDOM = "random_policy_v1"


# ----------------------------------------------------------------------------- recorded commands (host)

def load_recorded_commands(attempt_dir, native_dir=None):
    """Commanded edge targets, cut flags/digits and the reset edge of one WP-17 Native attempt (CPU only)."""
    from .replay import _sha256, default_native_dir
    attempt = Path(attempt_dir)
    native = Path(native_dir) if native_dir is not None else default_native_dir(attempt)
    setup = json.loads((native / "setup.json").read_text())
    targets, enabled, index, statuses = [], [], [], []
    with (attempt / "actions.jsonl").open() as stream:
        for step, line in enumerate(stream, start=1):
            row = json.loads(line)
            if int(row["step"]) != step:
                raise ValueError("actions.jsonl must hold consecutive steps 1..T")
            targets.append(row["resolved_target_world_xyz_xyzw"])
            enabled.append(bool(row["submitted"]["cut_enabled"]))
            index.append(int(row["submitted"].get("cut_index", 0)))
            statuses.append(row["info"]["action_status"])
    payload = json.loads((attempt / "compiled_task.json").read_text())
    return dict(targets=np.asarray(targets, float), cut_enabled=np.asarray(enabled, bool),
                cut_index=np.asarray(index, np.int32), statuses=statuses,
                reset_edge=np.asarray(setup["home_reset"]["measured_after_pregrasp"]["tool_edge_pose_xyzw"], float),
                task_id=payload.get("task_id"), config_sha256=payload.get("config_sha256"),
                actions_sha256=_sha256(attempt / "actions.jsonl"))


def selector_for_digit(digit, max_cuts):
    """A selector value in [-1,1] that ``edge_delta_v1`` maps back to ``digit`` (bin centre)."""
    return float(2. * (int(digit) + .5) / int(max_cuts) - 1.)


def encode_edge_delta(reference, target, limits):
    """Normalized edge-frame ``a[0:6]`` (float64, unclipped) that moves ``reference`` to ``target``."""
    from scipy.spatial.transform import Rotation
    q_ref = Rotation.from_quat(reference[3:])
    translation = q_ref.inv().apply(np.asarray(target[:3], float) - np.asarray(reference[:3], float))
    rotation = (q_ref.inv() * Rotation.from_quat(target[3:])).as_rotvec()
    return np.r_[translation / limits.max_translation_m, rotation / limits.max_rotation_rad]


def recorded_command_actions(targets, cut_enabled, cut_index, initial_pose, limits, T_world_workpiece, max_cuts):
    """Open-loop ``edge_delta_v1`` actions [T,8] float32 reproducing ``targets`` from ``initial_pose``.

    Returns ``(actions, reconstructed [T,7] float64, report)``; the reconstruction is the frozen NumPy control
    path applied to the float32 actions."""
    from ..control import normalized_delta, resolve_delta
    from ..scene_frames import validate_pose_xyzw
    targets = np.asarray(targets, float)
    T = len(targets)
    actions = np.zeros((T, 8), np.float32)
    reconstructed = np.zeros((T, 7))
    commanded = validate_pose_xyzw(np.asarray(initial_pose, float)).copy()
    clamped = 0
    max_abs = 0.
    for t in range(T):
        a6 = encode_edge_delta(commanded, targets[t], limits)
        max_abs = max(max_abs, float(np.abs(a6).max()))
        a = np.r_[a6, 1. if cut_enabled[t] else -1., selector_for_digit(cut_index[t], max_cuts)].astype(np.float32)
        actions[t] = a
        commanded, report = resolve_delta(commanded, normalized_delta(a[:6].astype(float), limits), limits,
                                          frame="edge", T_world_workpiece=T_world_workpiece)
        clamped += int(report["translation_clamped"] or report["rotation_clamped"] or report["workspace_clamped"])
        reconstructed[t] = commanded
    report = residual_report(reconstructed, targets)
    report.update(steps=T, clamped_steps=clamped, max_abs_normalized_component=max_abs,
                  path="encode_edge_delta (SciPy) -> float32 -> control.normalized_delta + resolve_delta(edge)")
    return actions, reconstructed, report


def residual_report(poses, targets):
    """Max position (m) and rotation (rad) difference between pose rows and recorded targets."""
    poses, targets = np.asarray(poses, float), np.asarray(targets, float)
    dot = np.abs(np.sum(poses[:, 3:] * targets[:, 3:], axis=1) /
                 (np.linalg.norm(poses[:, 3:], axis=1) * np.linalg.norm(targets[:, 3:], axis=1)))
    return dict(max_position_diff_m=float(np.abs(poses[:, :3] - targets[:, :3]).max()),
                max_rotation_diff_rad=float((2 * np.arccos(np.clip(dot, -1., 1.))).max()),
                max_quaternion_component_diff=float(np.abs(np.where(
                    (np.sum(poses[:, 3:] * targets[:, 3:], 1) < 0)[:, None], -poses[:, 3:], poses[:, 3:])
                    - targets[:, 3:]).max()))


# ----------------------------------------------------------------------------- device policies

class RecordedCommandPolicyV1:
    """Device table ``[E, T, 8]`` of recorded-command actions; env ``n`` follows table ``table_index[n]``.

    Envs with ``table_index < 0`` (or past the table end) get the HOLD row (no motion, disarmed)."""

    version = POLICY_RECORDED

    def __init__(self, action_tables, table_index, *, device):
        import torch
        self.torch = torch
        tables = np.stack([np.asarray(t, np.float32) for t in action_tables])
        if tables.ndim != 3 or tables.shape[2] != 8:
            raise ValueError("action tables must be [E, T, 8]")
        self.device = torch.device(device)
        self.tables = torch.as_tensor(tables, device=self.device)
        self.table_index = torch.as_tensor(np.asarray(table_index, np.int64), device=self.device)
        self.steps = tables.shape[1]
        self.hold = torch.tensor([0., 0., 0., 0., 0., 0., -1., 0.], dtype=torch.float32, device=self.device)

    def __call__(self, obs, info):
        torch = self.torch
        step = info["step_count"].to(torch.int64)
        valid = (self.table_index >= 0) & (step < self.steps)
        rows = self.tables[self.table_index.clamp(min=0), step.clamp(0, self.steps - 1)]
        return torch.where(valid[:, None], rows, self.hold)


class RandomPolicyV1:
    """Counter-based uniform random ``edge_delta_v1`` actions (see the module docstring)."""

    version = POLICY_RANDOM

    def __init__(self, *, arm_probability=.5, motion_scale=1., salt=0x5EED):
        if not 0. <= arm_probability <= 1. or not 0. < motion_scale <= 1.:
            raise ValueError("arm_probability in [0,1], motion_scale in (0,1]")
        self.arm_probability, self.motion_scale, self.salt = float(arm_probability), float(motion_scale), int(salt)

    def uniform(self, seed, episode, step):
        """``[N, 8]`` float64 uniforms in [0,1) from (seed, episode, step, component)."""
        import torch
        from .vector_env import splitmix64
        component = torch.arange(8, dtype=torch.int64, device=seed.device)
        key = splitmix64(splitmix64(splitmix64(seed[:, None] ^ self.salt) + episode[:, None]) + step[:, None])
        bits = splitmix64(key + component[None])
        return ((bits >> 11) & ((1 << 53) - 1)).to(torch.float64) * (2. ** -53)

    def __call__(self, obs, info):
        import torch
        u = self.uniform(info["seed"].to(torch.int64), info["episode"].to(torch.int64),
                         info["step_count"].to(torch.int64))
        motion = (2. * u[:, :6] - 1.) * self.motion_scale
        arm = torch.where(u[:, 6] < self.arm_probability, 1., -1.)
        selector = 2. * u[:, 7] - 1.
        return torch.cat([motion, arm[:, None].to(motion.dtype), selector[:, None]], 1).to(torch.float32)


class MeasuredTrajectoryDriver:
    """Device tables of recorded measured edge poses ``[E,T,7]``, resolved digits ``[E,T]`` and pad boxes
    ``[E,T,3,2,3]`` (``rl.replay.MeasuredTrajectory``); env ``n`` replays table ``table_index[n]``.

    ``inputs(step_count)`` gathers the inputs of control step ``step_count + 1`` (no host sync). PRIVILEGED:
    for ``env.step_measured_edge`` only."""

    drive = "measured_trajectory_replay"

    def __init__(self, trajectories, table_index, *, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        lengths = {t.steps for t in trajectories}
        if len(lengths) != 1:
            raise ValueError("trajectories must have equal lengths")
        self.steps = lengths.pop()
        self.pose = torch.as_tensor(np.stack([t.edge_pose_world_xyzw for t in trajectories]), dtype=torch.float64,
                                    device=self.device)
        self.cut = torch.as_tensor(np.stack([t.resolved_cut for t in trajectories]), dtype=torch.int32,
                                   device=self.device)
        self.pads = torch.as_tensor(np.stack([t.pad_boxes_edge for t in trajectories]), dtype=torch.float32,
                                    device=self.device)
        self.table_index = torch.as_tensor(np.asarray(table_index, np.int64), device=self.device)
        self.mask = self.table_index >= 0

    def inputs(self, step_count):
        idx = self.table_index.clamp(min=0)
        step = step_count.to(self.torch.int64).clamp(0, self.steps - 1)
        return self.pose[idx, step], self.cut[idx, step], self.pads[idx, step]
