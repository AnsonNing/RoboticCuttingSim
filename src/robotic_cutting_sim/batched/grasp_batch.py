"""B2 physical grasp of workload B (ADR 0022 §5.1, §6.3): per-world settle gates, settled-grasp snapshot, restore.

At build (and ``rebuild``) all worlds start at the declared HOME and settle together for ``grasp.settle_s``
(1000 substeps of 2 ms), the procedure of the single Lab env (``isaac_lab_env._settle_physical_grasp``). Over the
final ``grasp.stability_window_s`` the per-world gates of that procedure are evaluated on the device:

* bilateral load-bearing contact fraction (mean over the window) >= ``min_bilateral_contact_fraction``;
* hand-relative tool translation / rotation drift <= ``translation_tolerance_m`` / ``rotation_tolerance_rad``;
* tool linear / angular speed <= ``linear_speed_tolerance_m_s`` / ``angular_speed_tolerance_rad_s``;
* tool rigid penetration <= ``max_penetration_m``; no unintended tool contact slots; non-empty window.

Strict mode raises with the failing env ids; training mode marks them invalid. A per-slot snapshot (Franka q/qd,
tool pose/twist, actuator targets, initial relative tool pose, measured edge, boxes) is then kept on the device and
``reset(env_ids)`` restores it with masked writes. The only difference from a fresh settle is the zeroed MuJoCo
warm start (UNVERIFIED, measured by G3 in stage S2). Re-settling one world would step every world, so a different
initial pose, grasp randomization or tool re-pickup on partial reset is ``UNSUPPORTED_GRASP_RESETTLE_ON_PARTIAL_RESET``.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp

from .robot_kernels import _relative

GRASP_BATCH_VERSION = "lab_batched_grasp_settle_v1"
# Settle accumulator columns (per world).
ACC_ROWS, ACC_BILATERAL_SUM, ACC_TRANSLATION, ACC_ROTATION, ACC_LINEAR, ACC_ANGULAR, ACC_PENETRATION, ACC_UNINTENDED = range(8)
ACC_WIDTH = 8
CHECKS = ("nonempty_stability_window", "bilateral_load_bearing_contacts", "translation_stable", "rotation_stable",
          "linear_speed_stable", "angular_speed_stable", "penetration_bounded", "no_unintended_support")


@wp.kernel
def initial_relative_kernel(body_q: wp.array(dtype=wp.transform), hand: wp.array(dtype=wp.int32),
                            tool: wp.array(dtype=wp.int32), R_out: wp.array(dtype=wp.mat33d),
                            p_out: wp.array(dtype=wp.vec3d)):
    """dim = N. ``relative_tool_pose(hand, tool)`` at settle start (the drift reference)."""
    e = wp.tid()
    R, p = _relative(body_q[hand[e]], body_q[tool[e]])
    R_out[e] = R
    p_out[e] = p


@wp.kernel
def settle_window_kernel(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                         hand: wp.array(dtype=wp.int32), tool: wp.array(dtype=wp.int32),
                         reference_R: wp.array(dtype=wp.mat33d), reference_p: wp.array(dtype=wp.vec3d),
                         integral: wp.array2d(dtype=float), acc: wp.array2d(dtype=float)):
    """dim = N. One decimation-boundary row inside the stability window (``_settle_physical_grasp`` row)."""
    e = wp.tid()
    R, p = _relative(body_q[hand[e]], body_q[tool[e]])
    Rr = reference_R[e]
    dR = wp.transpose(Rr)*R
    dp = wp.transpose(Rr)*(p-reference_p[e])
    translation = float(wp.length(dp))
    c = (dR[0, 0]+dR[1, 1]+dR[2, 2]-wp.float64(1.0))/wp.float64(2.0)
    rotation = float(wp.acos(wp.clamp(c, wp.float64(-1.0), wp.float64(1.0))))
    v = body_qd[tool[e]]
    linear = wp.length(wp.vec3(v[0], v[1], v[2]))
    angular = wp.length(wp.vec3(v[3], v[4], v[5]))
    fraction = float(0.)
    if integral[e, 9] > 0.:
        fraction = integral[e, 8]/integral[e, 9]
    acc[e, 0] = acc[e, 0]+1.
    acc[e, 1] = acc[e, 1]+fraction
    acc[e, 2] = wp.max(acc[e, 2], translation)
    acc[e, 3] = wp.max(acc[e, 3], rotation)
    acc[e, 4] = wp.max(acc[e, 4], linear)
    acc[e, 5] = wp.max(acc[e, 5], angular)
    acc[e, 6] = wp.max(acc[e, 6], integral[e, 10])
    acc[e, 7] = acc[e, 7]+integral[e, 11]


@wp.kernel
def settle_gates_kernel(acc: wp.array2d(dtype=float), min_bilateral: float, translation_tolerance: float,
                        rotation_tolerance: float, linear_tolerance: float, angular_tolerance: float,
                        max_penetration: float, failed_checks: wp.array(dtype=wp.int32),
                        count: wp.array(dtype=wp.int32)):
    """dim = N. ``failed_checks[e]`` = bit i set when ``CHECKS[i]`` fails; ``count[0]`` = failing worlds."""
    e = wp.tid()
    rows = acc[e, 0]
    bits = wp.int32(0)
    if rows <= 0.:
        bits = bits | 1
    elif acc[e, 1]/rows < min_bilateral:
        bits = bits | 2
    if acc[e, 2] > translation_tolerance:
        bits = bits | 4
    if acc[e, 3] > rotation_tolerance:
        bits = bits | 8
    if acc[e, 4] > linear_tolerance:
        bits = bits | 16
    if acc[e, 5] > angular_tolerance:
        bits = bits | 32
    if acc[e, 6] > max_penetration:
        bits = bits | 64
    if acc[e, 7] != 0.:
        bits = bits | 128
    failed_checks[e] = bits
    if bits != 0:
        wp.atomic_add(count, 0, 1)


def decode_checks(bits):
    return [name for i, name in enumerate(CHECKS) if int(bits) & (1 << i)]


def launch_gates(acc, grasp, failed_checks, count, device):
    count.zero_()
    wp.launch(settle_gates_kernel, acc.shape[0], device=device,
              inputs=[acc, float(grasp.min_bilateral_contact_fraction), float(grasp.translation_tolerance_m),
                      float(grasp.rotation_tolerance_rad), float(grasp.linear_speed_tolerance_m_s),
                      float(grasp.angular_speed_tolerance_rad_s), float(grasp.max_penetration_m), failed_checks, count])


def host_gate_reference(rows, grasp):
    """The frozen ``_settle_physical_grasp`` check dictionary for one world's window rows (host; tests)."""
    window = rows
    return {"nonempty_stability_window": bool(window),
            "bilateral_load_bearing_contacts": bool(window) and np.mean([r["bilateral_substep_fraction"] for r in window]) >= grasp.min_bilateral_contact_fraction,
            "translation_stable": bool(window) and max(r["relative_translation_drift_m"] for r in window) <= grasp.translation_tolerance_m,
            "rotation_stable": bool(window) and max(r["relative_rotation_drift_rad"] for r in window) <= grasp.rotation_tolerance_rad,
            "linear_speed_stable": bool(window) and max(np.linalg.norm(r["tool_linear_velocity_world_m_s"]) for r in window) <= grasp.linear_speed_tolerance_m_s,
            "angular_speed_stable": bool(window) and max(np.linalg.norm(r["tool_angular_velocity_world_rad_s"]) for r in window) <= grasp.angular_speed_tolerance_rad_s,
            "penetration_bounded": bool(window) and max(r["max_tool_rigid_penetration_m"] for r in window) <= grasp.max_penetration_m,
            "no_unintended_support": bool(window) and all(r["unintended_tool_contact_slots"] == 0 for r in window)}


@dataclass
class GraspSnapshot:
    """Per-slot settled state on the device (ADR 0022 §6.3). Torch tensors ``[N, ...]`` plus Warp references."""

    franka_q: list            # 2 x [N, 9] float32 (articulation joint order)
    franka_targets: list      # 2 x [N, 9] float32 position targets
    tool_pose: object         # [N, 7] float32 XYZ + XYZW (root link pose)
    tool_velocity: object     # [N, 6] float32
    relative_R: object        # wp.array mat33d [N]
    relative_p: object        # wp.array vec3d [N]
    measured_edge: object     # [N, 7] float64
    nominal_edge: object      # [N, 7] float64
    box_centers: object       # wp.array vec3 [N*B]
    box_halves: object        # wp.array vec3 [N*B]
    passed: object            # [N] bool (strict mode: all True)
    failed_checks: object     # wp.array int32 [N]

    def receipt(self):
        bits = self.failed_checks.numpy()
        return {"version": GRASP_BATCH_VERSION, "slots": len(bits),
                "failed": {int(e): decode_checks(b) for e, b in enumerate(bits) if b},
                "restore": "masked joint/root writes of the settled slot state; MuJoCo warm start zeroed (UNVERIFIED, G3)"}


__all__ = ["GRASP_BATCH_VERSION", "CHECKS", "ACC_WIDTH", "initial_relative_kernel", "settle_window_kernel",
           "settle_gates_kernel", "launch_gates", "decode_checks", "host_gate_reference", "GraspSnapshot"]
