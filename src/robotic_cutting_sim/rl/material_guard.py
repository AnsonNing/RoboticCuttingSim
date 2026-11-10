"""Material input-domain guard of the vector envs (P1-2b task 5): device-side, no host sync.

Before the measured edge pose, the cut digit and the pad obstacle boxes of a control step reach
``BatchedCuttingMaterial`` (its pose interpolation and the 200-substep CUDA graph), every env's input is checked
against a per-env valid domain with Torch device ops only. An env that fails any check is latched (sticky until
its reset): from that step on its material input is HELD -- measured pose = the material's ``last_pose`` (the last
accepted pose, zero twist), cut digit -1 (disarmed), pad boxes = the last accepted boxes -- and the env reports
``termination.TERM_INPUT_DOMAIN`` (terminated). Other envs are unaffected (pure per-env ``torch.where``).

Checks (``MaterialInputGuardConfig``; each a bit of ``input_guard_reason``):

* ``NONFINITE_POSE``: a non-finite component of the measured pose;
* ``QUATERNION_NORM``: ``| |q| - 1 | > quaternion_norm_tolerance`` (the frozen chain normalizes any nonzero
  norm; the guard only rejects garbage such as zero or huge quaternions);
* ``OUT_OF_DOMAIN``: the edge position, in the env's workpiece LOCAL frame, outside the axis-aligned box spanned by
  the env's reference grid (``grid_origin .. grid_origin + grid_dims * dx``) and its initial measured edge position,
  expanded by ``margin_m`` on every side;
* ``TRANSLATION_JUMP`` / ``ROTATION_JUMP``: the step from the last accepted pose exceeds
  ``max_translation_per_step_m`` / ``max_rotation_per_step_rad`` (a 20 ms control step);
* ``PAD_BOX``: a non-finite pad box, a half extent outside ``(0, max_pad_half_extent_m]`` or an edge-frame box
  centre farther than ``max_pad_center_m`` from the edge.

The defaults are far outside every legitimate input recorded so far (P1-1b S3 B1/B2 inputs: max step 12 mm /
0.17 rad, pad centres <= 0.12 m, half extents <= 0.03 m; workload A actions are limited to 1 mm / step by
``edge_delta_v1``). The guard is not a physics fix: it keeps out-of-domain INPUT away from the material; it does
not repair a material state that became invalid inside the domain (that is ``MATERIAL_INVALID``).
"""
from dataclasses import asdict, dataclass

import numpy as np

GUARD_VERSION = "material_input_guard_v1"
NONFINITE_POSE, QUATERNION_NORM, OUT_OF_DOMAIN, TRANSLATION_JUMP, ROTATION_JUMP, PAD_BOX = 1, 2, 4, 8, 16, 32
GUARD_REASONS = {NONFINITE_POSE: "NONFINITE_POSE", QUATERNION_NORM: "QUATERNION_NORM", OUT_OF_DOMAIN: "OUT_OF_DOMAIN",
                 TRANSLATION_JUMP: "TRANSLATION_JUMP", ROTATION_JUMP: "ROTATION_JUMP", PAD_BOX: "PAD_BOX"}


def decode_guard_reason(value):
    return [name for bit, name in sorted(GUARD_REASONS.items()) if int(value) & bit]


@dataclass(frozen=True)
class MaterialInputGuardConfig:
    enabled: bool = True
    margin_m: float = .25
    max_translation_per_step_m: float = .05
    max_rotation_per_step_rad: float = .5
    quaternion_norm_tolerance: float = .5
    max_pad_center_m: float = .5
    max_pad_half_extent_m: float = .1

    def validate(self):
        for name in ("margin_m", "max_translation_per_step_m", "max_rotation_per_step_rad",
                     "quaternion_norm_tolerance", "max_pad_center_m", "max_pad_half_extent_m"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive finite number")
        return self


def domain_boxes(material, margin_m):
    """Per-env LOCAL-frame domain ``(lo [N,3], hi [N,3])`` from the packed grid and the initial measured edge."""
    pk = material.packed
    env = pk.env
    dx = float(pk.uniform["dx"])
    origin = np.asarray(env["grid_origin"], float)
    upper = origin + np.asarray(env["grid_dims"], float) * dx
    T_world_local = np.asarray(env["T_world_local"], float)
    initial = np.asarray(material.initial_measured_pose.numpy(), float)       # build time (not a step)
    R, t = T_world_local[:, :3, :3], T_world_local[:, :3, 3]
    local = np.einsum("nji,nj->ni", R, initial[:, :3] - t)
    lo = np.minimum(origin, local) - float(margin_m)
    hi = np.maximum(upper, local) + float(margin_m)
    return lo, hi


class MaterialInputGuard:
    """Per-env latched input guard (module docstring). All tensors live on ``device``; no host sync per step."""

    def __init__(self, material, num_envs, device, config=None):
        import torch
        self.torch = torch
        self.config = (config or MaterialInputGuardConfig()).validate()
        self.num_envs, self.device = int(num_envs), device
        lo, hi = domain_boxes(material, self.config.margin_m)
        T = np.asarray(material.packed.env["T_world_local"], float)
        f64 = dict(dtype=torch.float64, device=device)
        self.lo, self.hi = torch.as_tensor(lo, **f64), torch.as_tensor(hi, **f64)
        self.R_world_local = torch.as_tensor(T[:, :3, :3], **f64)
        self.t_world_local = torch.as_tensor(T[:, :3, 3], **f64)
        self.reason = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self.trip_step = torch.full((self.num_envs,), -1, dtype=torch.int64, device=device)
        self.trips = torch.zeros(self.num_envs, dtype=torch.int64, device=device)   # latch events since construction
        self._saved_boxes = None
        self._domain_host = dict(lo=lo.tolist(), hi=hi.tolist())

    # ------------------------------------------------------------------ checks (device)

    def _bit(self, condition, bit):
        torch = self.torch
        return torch.where(condition, torch.full_like(self.reason, bit), torch.zeros_like(self.reason))

    def check_pose(self, pose, last_pose):
        """``int32 [N]`` guard bits of WORLD poses ``[N,7]`` against the last accepted poses ``[N,7]``."""
        torch, c = self.torch, self.config
        pose = pose.to(torch.float64)
        last = last_pose.to(torch.float64)
        finite = torch.isfinite(pose).all(1)
        safe = torch.where(finite[:, None], pose, last)
        norm = torch.linalg.vector_norm(safe[:, 3:], dim=1)
        bad_norm = finite & ((norm - 1.).abs() > c.quaternion_norm_tolerance)
        local = torch.einsum("nji,nj->ni", self.R_world_local, safe[:, :3] - self.t_world_local)
        outside = finite & ((local < self.lo) | (local > self.hi)).any(1)
        jump = torch.linalg.vector_norm(safe[:, :3] - last[:, :3], dim=1)
        q1 = safe[:, 3:] / norm.clamp_min(1e-12)[:, None]
        q0 = last[:, 3:] / torch.linalg.vector_norm(last[:, 3:], dim=1).clamp_min(1e-12)[:, None]
        angle = 2. * torch.arccos((q0 * q1).sum(1).abs().clamp(0., 1.))
        bits = self._bit(~finite, NONFINITE_POSE) | self._bit(bad_norm, QUATERNION_NORM)
        bits = bits | self._bit(outside, OUT_OF_DOMAIN)
        bits = bits | self._bit(finite & (jump > c.max_translation_per_step_m), TRANSLATION_JUMP)
        bits = bits | self._bit(finite & ~bad_norm & (angle > c.max_rotation_per_step_rad), ROTATION_JUMP)
        return bits

    def check_boxes(self, centers, halves, mask=None):
        """``int32 [N]`` ``PAD_BOX`` bits of edge-frame boxes ``centers/halves [N,B,3]`` (rows of ``mask`` only)."""
        torch, c = self.torch, self.config
        centers, halves = centers.to(torch.float64), halves.to(torch.float64)
        finite = torch.isfinite(centers).all(2).all(1) & torch.isfinite(halves).all(2).all(1)
        h = torch.where(torch.isfinite(halves), halves, torch.zeros_like(halves))
        ctr = torch.where(torch.isfinite(centers), centers, torch.zeros_like(centers))
        bad = ~finite | (h <= 0).any(2).any(1) | (h > c.max_pad_half_extent_m).any(2).any(1)
        bad = bad | (torch.linalg.vector_norm(ctr, dim=2) > c.max_pad_center_m).any(1)
        if mask is not None:
            bad = bad & mask
        return self._bit(bad, PAD_BOX)

    def latch(self, bits, step_count=None):
        """OR ``bits`` into the sticky per-env reason; returns the ``bool [N]`` held mask."""
        torch = self.torch
        if not self.config.enabled:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        newly = (self.reason == 0) & (bits != 0)
        self.reason.bitwise_or_(bits.to(torch.int32))
        self.trips.add_(newly.to(torch.int64))
        if step_count is not None:
            self.trip_step.copy_(torch.where(newly, step_count.to(torch.int64), self.trip_step))
        return self.reason != 0

    @property
    def held(self):
        return self.reason != 0

    def hold(self, pose, cut, last_pose):
        """Held input of latched envs: ``pose`` -> ``last_pose``, ``cut`` -> -1; others unchanged."""
        torch = self.torch
        held = self.held
        pose_out = torch.where(held[:, None], last_pose.to(torch.float64), pose.to(torch.float64))
        cut_out = torch.where(held, torch.full_like(cut, -1), cut)
        return pose_out, cut_out

    # ------------------------------------------------------------------ backends

    def apply(self, pose, cut, last_pose, *, step_count=None, boxes=None, box_mask=None):
        """Workload A: check, latch and hold. ``boxes [N,B,2,3]`` (rows ``box_mask``) are the replay pad boxes about
        to be written. Returns ``(pose, cut, box_write_mask)``; ``box_write_mask`` excludes held envs."""
        bits = self.check_pose(pose, last_pose)
        if boxes is not None:
            bits = bits | self.check_boxes(boxes[:, :, 0], boxes[:, :, 1], box_mask)
        held = self.latch(bits, step_count)
        pose_out, cut_out = self.hold(pose, cut, last_pose)
        write = None if box_mask is None else box_mask & ~held
        return pose_out, cut_out, write

    def init_boxes(self, centers, halves):
        """Workload B: the accepted boxes before the first step (the material's packed/reset boxes)."""
        self._saved_boxes = (centers.detach().clone(), halves.detach().clone())

    def apply_in_place_boxes(self, pose, cut, last_pose, centers, halves, *, step_count=None):
        """Workload B: the pad kernel already wrote ``centers/halves`` ``[N,B,3]`` (views of the material's box
        arrays) in place. Held envs get their last accepted boxes back; accepted boxes are saved."""
        torch = self.torch
        bits = self.check_pose(pose, last_pose) | self.check_boxes(centers, halves)
        held = self.latch(bits, step_count)
        if self._saved_boxes is None:
            self._saved_boxes = (centers.clone(), halves.clone())
        saved_c, saved_h = self._saved_boxes
        m = held[:, None, None]
        centers.copy_(torch.where(m, saved_c, centers))
        halves.copy_(torch.where(m, saved_h, halves))
        saved_c.copy_(centers)
        saved_h.copy_(halves)
        return self.hold(pose, cut, last_pose)

    def reset(self, mask, centers=None, halves=None):
        """Clear the latch of the envs in ``mask`` (bool ``[N]``). Workload B passes the material's (reset) box
        views so that the saved boxes of those envs become their reset boxes."""
        torch = self.torch
        self.reason.copy_(torch.where(mask, torch.zeros_like(self.reason), self.reason))
        self.trip_step.copy_(torch.where(mask, torch.full_like(self.trip_step, -1), self.trip_step))
        if self._saved_boxes is not None and centers is not None:
            m = mask[:, None, None]
            saved_c, saved_h = self._saved_boxes
            saved_c.copy_(torch.where(m, centers.to(saved_c.dtype), saved_c))
            saved_h.copy_(torch.where(m, halves.to(saved_h.dtype), saved_h))

    def termination_bits(self):
        from .termination import TERM_INPUT_DOMAIN
        return self._bit(self.held, TERM_INPUT_DOMAIN)

    def receipt(self):
        return {"version": GUARD_VERSION, "config": asdict(self.config), "reasons": {str(k): v for k, v in
                GUARD_REASONS.items()}, "domain_local_frame": self._domain_host,
                "domain": "reference grid AABB U initial measured edge, expanded by margin_m (workpiece LOCAL frame)",
                "held_input": "measured pose = material last_pose, cut -1, pad boxes = last accepted",
                "host_sync": "none (device torch ops)", "raises": False}


__all__ = ["MaterialInputGuard", "MaterialInputGuardConfig", "GUARD_REASONS", "GUARD_VERSION", "decode_guard_reason",
           "domain_boxes"]
