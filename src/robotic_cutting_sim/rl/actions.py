"""Batched device action spec ``edge_delta_v1`` (ADR 0021 §8, A = 8).

``a[0:6]``  normalized edge-frame delta in ``[-1, 1]^6`` (translation, rotation vector),
``a[6]``    arm flag (``> 0`` arms the cut),
``a[7]``    cut-digit selector in ``[-1, 1]``, mapped to ``{0..K-1}``.

A Torch port, batched over N envs on one device, of the frozen host path a single env
runs for one ``edge_delta`` action with ``frame="edge"``::

    delta = control.normalized_delta(a[0:6], limits)                       # clip to [-1,1], scale
    target, report = control.resolve_delta(commanded, delta, limits,       # clamp, compose, bound_target
                                           frame="edge", T_world_workpiece=T)
    status = "CLAMPED" if any(report[*_clamped]) else "ACCEPTED"          # policy_env.PolicyReadyEnv.step

computed in float64 with SciPy's quaternion conventions (``from_quat``/compose normalize,
``as_rotvec`` canonical hemisphere with the 1e-3 Taylor branch, ``from_rotvec``). The
commanded edge pose state stays float64 (it accumulates across steps like the frozen
``inner._commanded``); observations cast it.

Invalid input is handled as ``PolicyReadyEnv`` handles it (validation order: target ->
``cut_enabled`` -> ``cut_index``); a rejected action is not executed and the explicit
HOLD keeps the clock running (``rejection_policy="hold"``): the commanded pose is
unchanged and the cut is disarmed (digit ``-1``).

* non-finite ``a[0:6]`` -> ``REJECTED_INVALID_TARGET`` (frozen: "target needs a finite vector");
* non-finite ``a[6]``   -> ``REJECTED_INVALID_SCHEMA`` (frozen: "cut_enabled must be a Boolean");
* non-finite ``a[7]``   -> ``REJECTED_CUT_INDEX_OUT_OF_RANGE`` (frozen: cut_index not an integer in range).

Finite out-of-range values are clipped to ``[-1, 1]`` exactly as ``normalized_delta``
clips (silently; the frozen status then comes only from the ``resolve_delta`` report).
``out_of_box`` reports that clipping separately; it never changes the status. The
selector is clipped likewise, so every finite selector maps to a valid digit.

Not ported (``NOT_IMPLEMENTED`` in this spec, recorded in ``spec()``): the
``PolicyReadyEnv`` sampled tool-vs-fixture/table collision check
(``REJECTED_COLLISION_*``) and IK rejection; workload A has no arm.
"""
from dataclasses import dataclass

import numpy as np
import torch

from ..control import ActionLimits
from ..policy_env import ACTION_STATUSES

ACTION_SPEC_VERSION = "edge_delta_v1"
ACTION_DIM = 8
ACTION_NAMES = ("delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "arm", "cut_digit_selector")
STATUS_CODE = {name: code for code, name in enumerate(ACTION_STATUSES)}
ACCEPTED, CLAMPED = STATUS_CODE["ACCEPTED"], STATUS_CODE["CLAMPED"]
REJECTED_INVALID_TARGET = STATUS_CODE["REJECTED_INVALID_TARGET"]
REJECTED_INVALID_SCHEMA = STATUS_CODE["REJECTED_INVALID_SCHEMA"]
REJECTED_CUT_INDEX = STATUS_CODE["REJECTED_CUT_INDEX_OUT_OF_RANGE"]
UNIT_TOLERANCE = 1e-6   # scene_frames.validate_pose_xyzw
TAYLOR_ANGLE = 1e-3     # SciPy as_rotvec / from_rotvec branch


def status_names(codes):
    """Device int codes -> ``policy_env.ACTION_STATUSES`` names (host; for logs and tests only)."""
    return [ACTION_STATUSES[int(c)] for c in torch.as_tensor(codes).cpu().tolist()]


# ----------------------------------------------------------------------------- SciPy-convention quaternion ops (XYZW)

def _normalize(q):
    return q/torch.sqrt(q[..., 0]*q[..., 0]+q[..., 1]*q[..., 1]+q[..., 2]*q[..., 2]+q[..., 3]*q[..., 3])[..., None]


def _compose(p, q):
    """``Rotation(p) * Rotation(q)``: Hamilton product, then normalized (``Rotation(..., normalize=True)``)."""
    c0 = p[..., 1]*q[..., 2]-p[..., 2]*q[..., 1]
    c1 = p[..., 2]*q[..., 0]-p[..., 0]*q[..., 2]
    c2 = p[..., 0]*q[..., 1]-p[..., 1]*q[..., 0]
    x = p[..., 3]*q[..., 0]+q[..., 3]*p[..., 0]+c0
    y = p[..., 3]*q[..., 1]+q[..., 3]*p[..., 1]+c1
    z = p[..., 3]*q[..., 2]+q[..., 3]*p[..., 2]+c2
    w = p[..., 3]*q[..., 3]-p[..., 0]*q[..., 0]-p[..., 1]*q[..., 1]-p[..., 2]*q[..., 2]
    return _normalize(torch.stack([x, y, z, w], -1))


def _inv(q):
    return torch.cat([-q[..., :3], q[..., 3:]], -1)


def _norm3(v):
    return torch.sqrt(v[..., 0]*v[..., 0]+v[..., 1]*v[..., 1]+v[..., 2]*v[..., 2])


def _as_rotvec(q):
    w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
    flip = (w < 0) | ((w == 0) & ((x < 0) | ((x == 0) & ((y < 0) | ((y == 0) & (z < 0))))))
    q = torch.where(flip[..., None], -q, q)
    angle = 2*torch.atan2(_norm3(q[..., :3]), q[..., 3])
    a2 = angle*angle
    small = 2+a2/12+7*a2*a2/2880
    large = angle/torch.where(angle <= TAYLOR_ANGLE, torch.ones_like(angle), torch.sin(angle/2))
    return torch.where(angle <= TAYLOR_ANGLE, small, large)[..., None]*q[..., :3]


def _from_rotvec(r):
    angle = _norm3(r)
    a2 = angle*angle
    small = .5-a2/48+a2*a2/3840
    large = torch.sin(angle/2)/torch.where(angle <= TAYLOR_ANGLE, torch.ones_like(angle), angle)
    scale = torch.where(angle <= TAYLOR_ANGLE, small, large)
    return torch.cat([scale[..., None]*r, torch.cos(angle/2)[..., None]], -1)


def _as_matrix(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2, w2 = x*x, y*y, z*z, w*w
    xy, zw, xz, yw, yz, xw = x*y, z*w, x*z, y*w, y*z, x*w
    rows = [x2-y2-z2+w2, 2*(xy-zw), 2*(xz+yw), 2*(xy+zw), -x2+y2-z2+w2, 2*(yz-xw),
            2*(xz-yw), 2*(yz+xw), -x2-y2+z2+w2]
    return torch.stack(rows, -1).reshape(*q.shape[:-1], 3, 3)


def _clamp_norm(v, limit):
    """control._clamp_norm, batched: (vector, norm, clamped)."""
    norm = _norm3(v)
    clamped = norm > limit
    scale = torch.where(clamped, limit/torch.where(clamped, norm, torch.ones_like(norm)), torch.ones_like(norm))
    return torch.where(clamped[..., None], v*scale[..., None], v), norm, clamped


def _transform_points(T, p):
    """scene_frames.transform_points for per-env [N,4,4] transforms and [N,3] points."""
    return torch.einsum("nij,nj->ni", T[:, :3, :3], p)+T[:, :3, 3]


# ----------------------------------------------------------------------------- spec

@dataclass
class ActionResult:
    """Per-env outputs of one batched action application (all device tensors, leading dim N)."""

    commanded_pose: torch.Tensor          # [N,7] float64, new commanded WORLD edge pose (unchanged on rejection)
    cut_digit: torch.Tensor               # [N] int32, -1 disarmed (also on rejection), else 0..K-1
    status: torch.Tensor                  # [N] int32 index into policy_env.ACTION_STATUSES
    executed_action: torch.Tensor         # [N,8] float32 executed normalized action (HOLD row on rejection)
    armed: torch.Tensor                   # [N] bool
    translation_clamped: torch.Tensor     # [N] bool  (resolve_delta report)
    rotation_clamped: torch.Tensor        # [N] bool
    workspace_clamped: torch.Tensor       # [N] bool
    out_of_box: torch.Tensor              # [N] bool, a finite component was outside [-1,1] (clipped; not a status)
    reference_invalid: torch.Tensor       # [N] bool, commanded pose quaternion not unit (frozen path would raise)
    requested_translation_m: torch.Tensor  # [N] float64
    requested_rotation_rad: torch.Tensor  # [N] float64
    applied_translation_m: torch.Tensor   # [N] float64
    applied_rotation_rad: torch.Tensor    # [N] float64


class EdgeDeltaActionV1:
    """``edge_delta_v1`` over N envs: uniform ``ActionLimits`` and K; per-env ``T_world_workpiece``."""

    version = ACTION_SPEC_VERSION
    dim = ACTION_DIM

    def __init__(self, limits, T_world_workpiece, max_cuts, *, device="cpu"):
        if not isinstance(limits, ActionLimits):
            raise TypeError("limits must be a control.ActionLimits (validated by its __post_init__)")
        if isinstance(max_cuts, bool) or not isinstance(max_cuts, (int, np.integer)) or not 1 <= max_cuts <= 15:
            raise ValueError("max_cuts must be an integer in [1, 15] (PACKED_MAX_CUTS)")
        from ..scene_frames import invert_transform, validate_transform
        T = np.asarray(T_world_workpiece, np.float64)
        if T.ndim != 3 or T.shape[1:] != (4, 4) or len(T) < 1:
            raise ValueError("T_world_workpiece must be [N,4,4]")
        T = np.stack([validate_transform(m) for m in T])
        self.limits, self.max_cuts, self.num_envs = limits, int(max_cuts), len(T)
        self.device = torch.device(device)
        f64 = dict(dtype=torch.float64, device=self.device)
        self.T = torch.as_tensor(T, **f64)
        self.T_inv = torch.as_tensor(np.stack([invert_transform(m) for m in T]), **f64)
        self.max_translation = torch.tensor(float(limits.max_translation_m), **f64)
        self.max_rotation = torch.tensor(float(limits.max_rotation_rad), **f64)
        self.workspace_min = torch.as_tensor(np.asarray(limits.workspace_min_local_m, float), **f64)
        self.workspace_max = torch.as_tensor(np.asarray(limits.workspace_max_local_m, float), **f64)
        self._hold_row = torch.tensor([0., 0., 0., 0., 0., 0., -1., 0.], dtype=torch.float32, device=self.device)

    def spec(self):
        return {"version": ACTION_SPEC_VERSION, "dim": ACTION_DIM, "names": list(ACTION_NAMES), "frame": "edge",
                "limits": self.limits.to_dict(), "max_cuts": self.max_cuts,
                "cut_digit_mapping": "digit = min(floor((clip(a7,-1,1)+1)/2*K), K-1); a6 > 0 arms, else -1",
                "statuses": list(ACTION_STATUSES), "rejection_policy": "hold",
                "out_of_range": "clipped to [-1,1] like control.normalized_delta (reported as out_of_box, not a status)",
                "not_implemented": {"collision_check": "PolicyReadyEnv sampled tool-vs-fixture/table check not ported",
                                    "ik_rejection": "no arm in workload A"},
                "port_of": ["control.normalized_delta", "control.resolve_delta(frame='edge')", "control.bound_target",
                            "control.ActionLimits", "policy_env.PolicyReadyEnv.step status rule"]}

    def cut_digit_from_selector(self, selector):
        s = torch.clamp(torch.nan_to_num(selector.to(torch.float64), nan=0.), -1., 1.)
        return torch.clamp(torch.floor((s+1.)/2.*self.max_cuts), max=self.max_cuts-1).to(torch.int32)

    @torch.no_grad()
    def apply(self, actions, commanded_pose):
        """``actions [N,8]`` (any float dtype), ``commanded_pose [N,7]`` float64 WORLD -> :class:`ActionResult`."""
        N = self.num_envs
        if actions.shape != (N, ACTION_DIM) or commanded_pose.shape != (N, 7):
            raise ValueError(f"expected actions [{N},{ACTION_DIM}] and commanded_pose [{N},7], got "
                             f"{tuple(actions.shape)} and {tuple(commanded_pose.shape)}")
        if actions.device != self.device or commanded_pose.device != self.device:
            raise ValueError("actions and commanded_pose must be on the action spec's device (no implicit copies)")
        if commanded_pose.dtype != torch.float64:
            raise ValueError("commanded_pose is float64 state (frozen inner._commanded is float64)")
        a = actions.to(torch.float64)
        finite = torch.isfinite(a)
        bad_target = ~finite[:, :6].all(1)
        bad_arm = ~bad_target & ~finite[:, 6]
        bad_cut = ~bad_target & ~bad_arm & ~finite[:, 7]
        rejected = bad_target | bad_arm | bad_cut
        clean = torch.where(finite, a, torch.zeros_like(a))
        out_of_box = (clean.abs() > 1.).any(1) & ~rejected
        clipped = torch.clamp(clean, -1., 1.)
        # control.normalized_delta
        delta_t = clipped[:, :3]*self.max_translation
        delta_r = clipped[:, 3:6]*self.max_rotation
        ref = commanded_pose
        ref_norm = torch.sqrt((ref[:, 3:]*ref[:, 3:]).sum(1))
        reference_invalid = ~torch.isfinite(ref).all(1) | ((ref_norm-1.).abs() > UNIT_TOLERANCE)
        # control.resolve_delta: clamp the commanded components, then compose_delta(frame="edge")
        trans, req_t, t_clamped = _clamp_norm(delta_t, self.max_translation)
        rot, req_r, r_clamped = _clamp_norm(delta_r, self.max_rotation)
        q_ref = _normalize(ref[:, 3:])
        target_p = ref[:, :3]+torch.einsum("nij,nj->ni", _as_matrix(q_ref), trans)
        target_q = _compose(q_ref, _from_rotvec(rot))
        # control.bound_target(reference, target, limits, T)
        translation, _, _ = _clamp_norm(target_p-ref[:, :3], self.max_translation)
        relative = _as_rotvec(_compose(_normalize(target_q), _inv(q_ref)))
        rotvec, _, _ = _clamp_norm(relative, self.max_rotation)
        position = ref[:, :3]+translation
        local = _transform_points(self.T_inv, position)
        bounded = torch.minimum(torch.maximum(local, self.workspace_min), self.workspace_max)
        workspace_clamped = (bounded != local).any(1)
        position = torch.where(workspace_clamped[:, None], _transform_points(self.T, bounded), position)
        quaternion = _compose(_from_rotvec(rotvec), q_ref)
        quaternion = quaternion/torch.sqrt((quaternion*quaternion).sum(1, keepdim=True))
        pose = torch.cat([position, quaternion], 1)
        clamped = t_clamped | r_clamped | workspace_clamped
        status = torch.where(clamped, torch.full_like(req_t, CLAMPED, dtype=torch.int32),
                             torch.full_like(req_t, ACCEPTED, dtype=torch.int32))
        status = torch.where(bad_cut, torch.full_like(status, REJECTED_CUT_INDEX), status)
        status = torch.where(bad_arm, torch.full_like(status, REJECTED_INVALID_SCHEMA), status)
        status = torch.where(bad_target, torch.full_like(status, REJECTED_INVALID_TARGET), status)
        armed = (clipped[:, 6] > 0) & ~rejected
        digit = self.cut_digit_from_selector(clipped[:, 7])
        cut = torch.where(armed, digit, torch.full_like(digit, -1))
        executed = torch.where(rejected[:, None], self._hold_row.expand(N, ACTION_DIM), clipped.to(torch.float32))
        zero = torch.zeros_like(req_t)
        keep = ~rejected
        return ActionResult(
            commanded_pose=torch.where(rejected[:, None], ref, pose), cut_digit=cut, status=status,
            executed_action=executed, armed=armed, translation_clamped=t_clamped & keep,
            rotation_clamped=r_clamped & keep, workspace_clamped=workspace_clamped & keep, out_of_box=out_of_box,
            reference_invalid=reference_invalid, requested_translation_m=torch.where(keep, req_t, zero),
            requested_rotation_rad=torch.where(keep, req_r, zero),
            applied_translation_m=torch.where(keep, _norm3(position-ref[:, :3]), zero),
            applied_rotation_rad=torch.where(keep, _norm3(rotvec), zero))
