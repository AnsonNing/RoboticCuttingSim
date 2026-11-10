"""Batched LM edge IK for workload B (ADR 0022 §4.2): ``n_problems = N`` on the Franka 0 prototype model.

The objectives are those of ``robot.DualFrankaController`` / ``isaac_lab_env._bind_ik``: position and rotation of
``panda_hand`` with ``link_offset = hand_from_edge``, joint-limit weight 10, ``lambda_initial = 0.01``, analytic
Jacobian, ``ik_iterations`` (48) LM iterations, one seed (the live joint positions). Each env is its own LM
problem (lambda is per row), so N problems never couple.

After the solve, coordinates are clipped to the joint limits (``isaac_lab_env._solve_ik``), FK runs on an
N-replicated copy of the prototype, and a device kernel writes the edge residuals and ``status[N]``:
``ACCEPTED`` or the frozen ``REJECTED_IK_UNREACHABLE`` (``policy_env.ACTION_STATUSES``) when the composed edge
misses the target by more than 1 mm or ``rotation_tolerance_rad`` (``isaac_lab_env.py:462-469``), or when the
solution is not finite. The caller holds the previous targets of rejected envs (``live_env.py:225-248``).

Everything is Warp launches on one device (no host sync); the prototype comes from a Newton ``ModelBuilder``
(Isaac Lab: ``NewtonManager._cl_protos[source]``; C tests: ``lab_scene.build_newton_rigid_scene``).
"""
import numpy as np
import warp as wp

from .pose_interp import _compose, _inv, _mul, _add, _fma, _norm4_blas, _normalize4
from .robot_kernels import _apply, _blas_normalize, _matrix, _pose_p, _pose_q, _vadd

ROBOT_IK_VERSION = "lab_batched_lm_ik_v1"
IK_OK = 0
IK_UNREACHABLE = 1
IK_NONFINITE = 2
IK_STATUS_NAMES = ("OK", "REJECTED_IK_UNREACHABLE", "REJECTED_IK_NONFINITE")


def ik_rejection_status_code():
    """Index of the frozen ``REJECTED_IK_UNREACHABLE`` in ``policy_env.ACTION_STATUSES``."""
    from ..policy_env import ACTION_STATUSES
    return ACTION_STATUSES.index("REJECTED_IK_UNREACHABLE")


@wp.kernel
def _write_targets(target: wp.array2d(dtype=wp.float64), positions: wp.array(dtype=wp.vec3),
                   rotations: wp.array(dtype=wp.vec4)):
    """``wp.vec3(*desired[:3])`` / ``wp.vec4(*desired[3:])`` of the frozen path (float64 -> float32)."""
    e = wp.tid()
    positions[e] = wp.vec3(wp.float32(target[e, 0]), wp.float32(target[e, 1]), wp.float32(target[e, 2]))
    rotations[e] = wp.vec4(wp.float32(target[e, 3]), wp.float32(target[e, 4]), wp.float32(target[e, 5]),
                           wp.float32(target[e, 6]))


@wp.kernel
def _scatter_seed(joint_pos: wp.array2d(dtype=float), coord_ids: wp.array(dtype=wp.int32),
                  default_q: wp.array(dtype=float), q: wp.array2d(dtype=float)):
    """dim = (N, C): prototype default, then the live articulation joints at their prototype coordinates."""
    e, c = wp.tid()
    q[e, c] = default_q[c]
    for j in range(coord_ids.shape[0]):
        if coord_ids[j] == c:
            q[e, c] = joint_pos[e, j]


@wp.kernel
def _clip_and_flatten(q: wp.array2d(dtype=float), lower: wp.array(dtype=float), upper: wp.array(dtype=float),
                      clip_coord: wp.array(dtype=wp.int32), clip_dof: wp.array(dtype=wp.int32),
                      flat: wp.array(dtype=float), finite: wp.array(dtype=wp.int32)):
    """dim = N. ``np.clip`` of the robot joint coordinates only, then the FK input row; non-finite flag."""
    e = wp.tid()
    C = q.shape[1]
    ok = wp.int32(1)
    for c in range(C):
        if not wp.isfinite(q[e, c]):
            ok = wp.int32(0)
    for j in range(clip_coord.shape[0]):
        c = clip_coord[j]
        d = clip_dof[j]
        q[e, c] = wp.min(wp.max(q[e, c], lower[d]), upper[d])
    for c in range(C):
        flat[e*C+c] = q[e, c]
    finite[e] = ok


@wp.kernel
def _residual(body_q: wp.array(dtype=wp.transform), hand: int, bodies_per_world: int,
              hand_edge_p: wp.vec3d, hand_edge_q: wp.vec4d, target: wp.array2d(dtype=wp.float64),
              finite: wp.array(dtype=wp.int32), position_tolerance: wp.float64, rotation_tolerance: wp.float64,
              edge: wp.array2d(dtype=wp.float64), position_error: wp.array(dtype=wp.float64),
              rotation_error: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    """dim = N. Edge of the solution (``compose_tool_pose(hand FK, hand_from_edge)``), residuals, gate."""
    e = wp.tid()
    hb = body_q[e*bodies_per_world+hand]
    hp = _pose_p(hb)
    hq = _blas_normalize(_pose_q(hb))
    p = _vadd(hp, _apply(_matrix(_normalize4(hq)), hand_edge_p))
    q = _compose(_normalize4(hq), _normalize4(hand_edge_q))
    for i in range(3):
        edge[e, i] = p[i]
    for i in range(4):
        edge[e, 3+i] = q[i]
    dx = p[0]-target[e, 0]
    dy = p[1]-target[e, 1]
    dz = p[2]-target[e, 2]
    err = wp.sqrt(_fma(dz, dz, _fma(dy, dy, _mul(dx, dx))))   # np.linalg.norm (BLAS ddot chain)
    desired = _normalize4(wp.vec4d(target[e, 3], target[e, 4], target[e, 5], target[e, 6]))
    r = _compose(_inv(q), desired)                             # Rotation(edge).inv() * Rotation(desired)
    rot = wp.float64(2.0)*wp.atan2(wp.sqrt(_add(_add(_mul(r[0], r[0]), _mul(r[1], r[1])), _mul(r[2], r[2]))),
                                   wp.abs(r[3]))
    position_error[e] = err
    rotation_error[e] = rot
    code = wp.int32(0)
    if finite[e] == 0 or not (wp.isfinite(err) and wp.isfinite(rot)):
        code = wp.int32(2)
    elif err > position_tolerance or rot > rotation_tolerance:
        code = wp.int32(1)
    status[e] = code


class BatchedEdgeIK:
    """N independent LM problems on the Franka 0 prototype; FK residual and status on the device."""

    def __init__(self, prototype_builder, num_envs, hand_from_edge, *, iterations=48, position_tolerance_m=1e-3,
                 rotation_tolerance_rad=.01, device="cpu", hand_suffix="/panda_hand",
                 joint_names=None):
        import newton
        import newton.ik as ik
        self.num_envs, self.iterations, self.device = int(num_envs), int(iterations), device
        mount = np.asarray(hand_from_edge, float)
        if mount.shape != (7,) or not np.isfinite(mount).all():
            raise ValueError("hand_from_edge must be a finite XYZ+XYZW pose")
        self.mount = mount
        with wp.ScopedDevice(device):
            self.model = prototype_builder.finalize(device=device)
            labels = list(self.model.body_label)
            hands = [i for i, label in enumerate(labels) if label.endswith(hand_suffix)]
            if len(hands) != 1:
                raise RuntimeError(f"Prototype must hold exactly one {hand_suffix}, got {len(hands)}")
            self.hand = hands[0]
            C = self.model.joint_coord_count
            self.n_coords = C
            N = self.num_envs
            self.q = wp.zeros((N, C), dtype=float)
            self.target = wp.zeros((N, 7), dtype=wp.float64)
            self.position = ik.IKObjectivePosition(link_index=self.hand, link_offset=wp.vec3(*mount[:3]),
                                                   target_positions=wp.zeros(N, dtype=wp.vec3))
            self.rotation = ik.IKObjectiveRotation(link_index=self.hand, link_offset_rotation=wp.quat(*mount[3:]),
                                                   target_rotations=wp.zeros(N, dtype=wp.vec4))
            limits = ik.IKObjectiveJointLimit(self.model.joint_limit_lower, self.model.joint_limit_upper, weight=10.)
            self.solver = ik.IKSolver(model=self.model, n_problems=N, objectives=[self.position, self.rotation, limits],
                                      lambda_initial=.01, jacobian_mode=ik.IKJacobianType.ANALYTIC)
            replicated = newton.ModelBuilder()
            replicated.replicate(prototype_builder, N)
            self.fk_model = replicated.finalize(device=device)
            self.fk_state = self.fk_model.state()
            self.bodies_per_world = self.model.body_count
            if self.fk_model.body_count != N*self.bodies_per_world:
                raise RuntimeError("Replicated FK model does not hold N prototype copies")
            # Robot scalar joints that the frozen path clips (never free-body coordinates).
            jtype = self.model.joint_type.numpy()
            q_start, qd_start = self.model.joint_q_start.numpy(), self.model.joint_qd_start.numpy()
            scalar = [j for j, t in enumerate(jtype) if t in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC))]
            self.clip_coord = wp.array(np.asarray([q_start[j] for j in scalar], np.int32), dtype=wp.int32)
            self.clip_dof = wp.array(np.asarray([qd_start[j] for j in scalar], np.int32), dtype=wp.int32)
            names = joint_names or [f"panda_joint{i}" for i in range(1, 8)]+["panda_finger_joint1", "panda_finger_joint2"]
            jl = list(self.model.joint_label)
            coord_ids = []
            for name in names:
                hits = [j for j, label in enumerate(jl) if label.endswith("/"+name)]
                if len(hits) != 1:
                    raise RuntimeError(f"Prototype joint {name!r} not unique: {hits}")
                coord_ids.append(int(q_start[hits[0]]))
            self.joint_names = list(names)
            self.coord_ids = wp.array(np.asarray(coord_ids, np.int32), dtype=wp.int32)
            self.default_q = wp.clone(self.model.joint_q)
            self.flat = wp.zeros(N*C, dtype=float)
            self.finite = wp.zeros(N, dtype=wp.int32)
            self.edge = wp.zeros((N, 7), dtype=wp.float64)
            self.position_error = wp.zeros(N, dtype=wp.float64)
            self.rotation_error = wp.zeros(N, dtype=wp.float64)
            self.status = wp.zeros(N, dtype=wp.int32)
            self.joint_q_out = wp.zeros((N, len(names)), dtype=float)
        self.position_tolerance = float(position_tolerance_m)
        self.rotation_tolerance = float(rotation_tolerance_rad)

    def receipt(self):
        return {"version": ROBOT_IK_VERSION, "n_problems": self.num_envs, "iterations": self.iterations,
                "objectives": ["position(panda_hand, link_offset=hand_from_edge)",
                               "rotation(panda_hand, link_offset_rotation=hand_from_edge)", "joint_limit(weight=10)"],
                "lambda_initial": .01, "jacobian": "analytic", "seeds": 1, "prototype_coords": self.n_coords,
                "position_tolerance_m": self.position_tolerance, "rotation_tolerance_rad": self.rotation_tolerance,
                "fk": "N-replicated prototype, newton.eval_fk", "host_syncs": 0}

    def seed_from_joint_pos(self, joint_pos):
        """``joint_pos [N, 9]`` (Warp float32, articulation joint order) -> prototype seed rows."""
        wp.launch(_scatter_seed, (self.num_envs, self.n_coords), device=self.device,
                  inputs=[joint_pos, self.coord_ids, self.default_q, self.q])

    def set_targets(self, target_pose):
        """``target_pose [N, 7]`` float64 world edge targets (device copy, no host sync)."""
        wp.copy(self.target, target_pose)
        wp.launch(_write_targets, self.num_envs, device=self.device,
                  inputs=[self.target, self.position.target_positions, self.rotation.target_rotations])

    def solve(self):
        """LM solve, clip, FK, residual and status. Results stay in ``q``, ``edge``, ``*_error``, ``status``."""
        import newton
        self.solver.step(self.q, self.q, iterations=self.iterations)
        wp.launch(_clip_and_flatten, self.num_envs, device=self.device,
                  inputs=[self.q, self.model.joint_limit_lower, self.model.joint_limit_upper, self.clip_coord,
                          self.clip_dof, self.flat, self.finite])
        wp.copy(self.fk_state.joint_q, self.flat)
        newton.eval_fk(self.fk_model, self.fk_state.joint_q, self.fk_model.joint_qd, self.fk_state)
        wp.launch(_residual, self.num_envs, device=self.device,
                  inputs=[self.fk_state.body_q, self.hand, self.bodies_per_world, wp.vec3d(*self.mount[:3]),
                          wp.vec4d(*self.mount[3:]), self.target, self.finite, wp.float64(self.position_tolerance),
                          wp.float64(self.rotation_tolerance), self.edge, self.position_error, self.rotation_error,
                          self.status])
        wp.launch(_gather, (self.num_envs, len(self.joint_names)), device=self.device,
                  inputs=[self.q, self.coord_ids, self.joint_q_out])

    def solve_from(self, joint_pos, target_pose):
        self.seed_from_joint_pos(joint_pos)
        self.set_targets(target_pose)
        self.solve()


@wp.kernel
def _gather(q: wp.array2d(dtype=float), coord_ids: wp.array(dtype=wp.int32), out: wp.array2d(dtype=float)):
    e, j = wp.tid()
    out[e, j] = q[e, coord_ids[j]]


__all__ = ["ROBOT_IK_VERSION", "BatchedEdgeIK", "IK_OK", "IK_UNREACHABLE", "IK_NONFINITE", "IK_STATUS_NAMES",
           "ik_rejection_status_code"]
