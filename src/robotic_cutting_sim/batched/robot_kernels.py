"""Per-world robot kernels of workload B (ADR 0022 §5.2-§5.3, §6.3, §7): index tables, edge and pad frames,
graph-safe wrench application, contact sampling, grasp monitor and rigid validity.

Every kernel is launched with ``dim = N`` (or over the shared contact pool) and touches only the bodies of
its own world through per-world global index tables. The tables are resolved once from labels
(``/World/envs/env_{e}/...``) and asserted with ``body_world[idx[e]] == e``; arithmetic layouts are never
trusted (the welded B1 tool is interleaved in the link order).

Frames (ADR 0022 §5.2): the edge pose is composed in float64 from the float32 ``body_q`` with SciPy's
operation order (the exact ``_mul``/``_add``/``_fma`` helpers of :mod:`.pose_interp`), so it matches the frozen
NumPy ``compose_pose`` / ``compose_tool_pose`` to <= 1 float64 ulp (C1 measures it):

* B1 ``rigid_mount_v1``  measured edge = ``compose_pose(compose_pose(hand, hand_from_tool), tool_from_edge)``;
* B2 ``physical_grasp_v1`` measured edge = ``compose_pose(tool, tool_from_edge)`` (``physical_tool.measured_edge_pose``);
* nominal edge (both) = ``compose_tool_pose(hand, hand_from_edge)`` (``robot.compose_tool_pose``, deployable).

Wrenches (replaces the frozen 0.18 m constant of ``isaac_lab_env._edge_wrench``): the world material wrench
``(f, tau_edge)`` acts at the CURRENT edge point ``p_e = body_q[tool] * tool_from_edge.p`` on the tool body:
``tau_com = tau_edge + (p_e - c_com) x f`` (``newton_tool.apply_tool_wrench``); pad wrenches go to the finger
bodies at the same ``p_e`` (``isaac_lab_env._pad_wrenches``). B1 needs no special lever: the tool body exists in
both modes. The kernel runs in the state-force callback before every 2 ms substep, inside the captured graph.
"""
import numpy as np
import warp as wp

from .pose_interp import _add, _compose, _fma, _mul, _norm4_blas, _normalize4, _sub

ROBOT_KERNELS_VERSION = "lab_robot_kernels_v1"
EDGE_STATUS_OK = 0
EDGE_STATUS_NONFINITE = 1
EDGE_STATUS_NON_UNIT_QUATERNION = 2
# Rigid validity bits (sticky until reset): ADR 0022 §7 "Host syncs per step".
RIGID_NONFINITE_STATE = 1
RIGID_CONTACT_POOL_OVERFLOW = 2
RIGID_IK_STREAK = 4
RIGID_EDGE_POSE = 8
RIGID_REASONS = {RIGID_NONFINITE_STATE: "RIGID_NONFINITE_STATE", RIGID_CONTACT_POOL_OVERFLOW: "CONTACT_POOL_OVERFLOW",
                 RIGID_IK_STREAK: "IK_FAILURE_STREAK", RIGID_EDGE_POSE: "INVALID_EDGE_POSE"}
# Contact sample columns (newton_tool.collect_tool_contacts / integrate_contact_sample, per world).
SAMPLE_WIDTH = 16
GRASP_HOLDING, GRASP_SLIPPING, GRASP_LOST, GRASP_INVALID_CONTACT = 0, 1, 2, 3
GRASP_STATUS_NAMES = ("HOLDING", "SLIPPING", "LOST", "INVALID_CONTACT_DATA")


def decode_rigid_reason(value):
    return [name for bit, name in RIGID_REASONS.items() if int(value) & bit]


# ----------------------------------------------------------------------------- index tables

FRANKA_BODIES = {"hand": "panda_hand", "left": "panda_leftfinger", "right": "panda_rightfinger"}


def _find(labels, world_prefix, scope, suffix):
    hits = [i for i, label in enumerate(labels) if label.startswith(world_prefix+"/") and f"/{scope}/" in label+"/"
            and label.endswith("/"+suffix)]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one body '{scope}/.../{suffix}' in {world_prefix}, got {len(hits)}")
    return hits[0]


def resolve_index_tables(body_label, body_world, num_envs, tool_mount, *, env_template="/World/envs/env_{}",
                         rigid_tool_name="CuttingTool", free_tool_name="Tool"):
    """Per-world global body indices: ``hand[N]``, ``tool[N]``, ``finger[N,2]`` (left, right), ``park_hand[N]``.

    Resolved from labels and asserted to belong to their world (``body_world[idx[e]] == e``)."""
    labels = list(body_label)
    world = np.asarray(body_world)
    hand, tool, finger, park = [], [], [], []
    for e in range(int(num_envs)):
        prefix = env_template.format(e)
        hand.append(_find(labels, prefix, "Franka_0", FRANKA_BODIES["hand"]))
        finger.append([_find(labels, prefix, "Franka_0", FRANKA_BODIES["left"]),
                       _find(labels, prefix, "Franka_0", FRANKA_BODIES["right"])])
        park.append(_find(labels, prefix, "Franka_1", FRANKA_BODIES["hand"]))
        if tool_mount == "rigid_mount_v1":
            tool.append(_find(labels, prefix, "Franka_0", rigid_tool_name))
        else:
            hits = [i for i, label in enumerate(labels) if label == f"{prefix}/{free_tool_name}"]
            if len(hits) != 1:
                raise RuntimeError(f"Expected one free tool body {prefix}/{free_tool_name}, got {len(hits)}")
            tool.append(hits[0])
    tables = {"hand": np.asarray(hand, np.int32), "tool": np.asarray(tool, np.int32),
              "finger": np.asarray(finger, np.int32).reshape(-1, 2), "park_hand": np.asarray(park, np.int32)}
    for name, values in tables.items():
        owners = world[values.reshape(len(values), -1)]
        expected = np.arange(len(values))[:, None]
        if not np.array_equal(owners, np.broadcast_to(expected, owners.shape)):
            raise RuntimeError(f"Index table {name!r} crosses worlds: body_world={owners.tolist()}")
    if len(set(tables["tool"].tolist())) != len(tables["tool"]):
        raise RuntimeError("Tool bodies are not unique per world")
    return tables


def resolve_joint_tables(joint_label, joint_q_start, joint_qd_start, num_envs, *, env_template="/World/envs/env_{}"):
    """Per-world coordinate/DOF indices of both Frankas' 9 joints: ``coords[N,2,9]``, ``dofs[N,2,9]``."""
    labels = list(joint_label)
    q_start, qd_start = np.asarray(joint_q_start), np.asarray(joint_qd_start)
    names = [f"panda_joint{i}" for i in range(1, 8)]+["panda_finger_joint1", "panda_finger_joint2"]
    coords = np.zeros((num_envs, 2, 9), np.int32)
    dofs = np.zeros((num_envs, 2, 9), np.int32)
    for e in range(int(num_envs)):
        for arm in (0, 1):
            for k, name in enumerate(names):
                j = _find(labels, env_template.format(e), f"Franka_{arm}", name)
                coords[e, arm, k], dofs[e, arm, k] = q_start[j], qd_start[j]
    return coords, dofs


# ----------------------------------------------------------------------------- float64 frame helpers

@wp.func
def _matrix(q: wp.vec4d):
    """SciPy ``as_matrix`` (XYZW), one IEEE operation per term."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    x2 = _mul(x, x)
    y2 = _mul(y, y)
    z2 = _mul(z, z)
    w2 = _mul(w, w)
    xy = _mul(x, y)
    zw = _mul(z, w)
    xz = _mul(x, z)
    yw = _mul(y, w)
    yz = _mul(y, z)
    xw = _mul(x, w)
    two = wp.float64(2.0)
    return wp.mat33d(_add(_sub(_sub(x2, y2), z2), w2), _mul(two, _sub(xy, zw)), _mul(two, _add(xz, yw)),
                     _mul(two, _add(xy, zw)), _add(_sub(_sub(y2, x2), z2), w2), _mul(two, _sub(yz, xw)),
                     _mul(two, _sub(xz, yw)), _mul(two, _add(yz, xw)), _add(_sub(z2, _add(x2, y2)), w2))


@wp.func
def _apply(R: wp.mat33d, v: wp.vec3d):
    """``Rotation.apply`` of one rotation (single or a length-1 batch; the host matmul/dgemv FMA chain
    ``fma(R2,v2, fma(R0,v0, R1*v1))``, the order ``pose_interp`` also pins). A multi-row batch would use
    NumPy einsum's ``(R0 v0 + R2 v2) + R1 v1`` instead; the frozen paths compose one pose at a time."""
    return wp.vec3d(_fma(R[0, 2], v[2], _fma(R[0, 0], v[0], _mul(R[0, 1], v[1]))),
                    _fma(R[1, 2], v[2], _fma(R[1, 0], v[0], _mul(R[1, 1], v[1]))),
                    _fma(R[2, 2], v[2], _fma(R[2, 0], v[0], _mul(R[2, 1], v[1]))))


@wp.func
def _vadd(a: wp.vec3d, b: wp.vec3d):
    return wp.vec3d(_add(a[0], b[0]), _add(a[1], b[1]), _add(a[2], b[2]))


@wp.func
def _vsub(a: wp.vec3d, b: wp.vec3d):
    return wp.vec3d(_sub(a[0], b[0]), _sub(a[1], b[1]), _sub(a[2], b[2]))


@wp.func
def _pose_p(t: wp.transform):
    p = wp.transform_get_translation(t)
    return wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))


@wp.func
def _pose_q(t: wp.transform):
    q = wp.transform_get_rotation(t)
    return wp.vec4d(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]), wp.float64(q[3]))


@wp.func
def _compose_p(p: wp.vec3d, q: wp.vec4d, lp: wp.vec3d):
    """Position of ``scene_frames.compose_pose`` (a length-1 Rotation batch): ``p + R.apply(lp)``."""
    return _vadd(p, _apply(_matrix(_normalize4(q)), lp))


@wp.func
def _compose_q(q: wp.vec4d, lq: wp.vec4d):
    """Orientation of ``compose_pose``: ``(Rotation.from_quat(q) * Rotation.from_quat(lq)).as_quat()``."""
    return _compose(_normalize4(q), _normalize4(lq))


@wp.func
def _blas_normalize(q: wp.vec4d):
    """``robot.validate_pose``: ``q / np.linalg.norm(q)`` (OpenBLAS ddot FMA chain)."""
    n = _norm4_blas(q)
    return wp.vec4d(q[0]/n, q[1]/n, q[2]/n, q[3]/n)


@wp.func
def _store7(out: wp.array2d(dtype=wp.float64), e: int, p: wp.vec3d, q: wp.vec4d):
    for i in range(3):
        out[e, i] = p[i]
    for i in range(4):
        out[e, 3+i] = q[i]


@wp.func
def _transform_finite(t: wp.transform):
    p = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)
    return (wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]) and wp.isfinite(q[0])
            and wp.isfinite(q[1]) and wp.isfinite(q[2]) and wp.isfinite(q[3]))


# ----------------------------------------------------------------------------- edge pose and pad boxes

@wp.kernel
def edge_pose_kernel(body_q: wp.array(dtype=wp.transform), source: wp.array(dtype=wp.int32),
                     hand: wp.array(dtype=wp.int32), chain_p: wp.array(dtype=wp.vec3d),
                     chain_q: wp.array(dtype=wp.vec4d), chain_length: int,
                     hand_edge_p: wp.vec3d, hand_edge_q: wp.vec4d,
                     measured_edge: wp.array2d(dtype=wp.float64), nominal_edge: wp.array2d(dtype=wp.float64),
                     status: wp.array(dtype=wp.int32)):
    """dim = N. ``measured_edge[e] = body_q[source[e]] o chain[0] o ... o chain[L-1]`` (each a ``compose_pose``)
    and ``nominal_edge[e] = compose_tool_pose(body_q[hand[e]], hand_from_edge)``; float64 out."""
    e = wp.tid()
    src = body_q[source[e]]
    hb = body_q[hand[e]]
    code = wp.int32(0)
    if not (_transform_finite(src) and _transform_finite(hb)):
        code = wp.int32(1)
    else:
        nq = _norm4_blas(_pose_q(src))
        nh = _norm4_blas(_pose_q(hb))
        if wp.abs(nq - wp.float64(1.0)) > wp.float64(1.0e-6) or wp.abs(nh - wp.float64(1.0)) > wp.float64(1.0e-6):
            code = wp.int32(2)
    status[e] = code
    p = _pose_p(src)
    q = _pose_q(src)
    for k in range(chain_length):
        p = _compose_p(p, q, chain_p[k])
        q = _compose_q(q, chain_q[k])
    _store7(measured_edge, e, p, q)
    # robot.compose_tool_pose: validate_pose (BLAS norm), single Rotation (matmul apply)
    hp = _pose_p(hb)
    hq = _blas_normalize(_pose_q(hb))
    _store7(nominal_edge, e, _vadd(hp, _apply(_matrix(_normalize4(hq)), hand_edge_p)),
            _compose_q(hq, hand_edge_q))


@wp.func
def _pose_from_row(pose: wp.array2d(dtype=wp.float64), e: int):
    return wp.vec3d(pose[e, 0], pose[e, 1], pose[e, 2]), wp.vec4d(pose[e, 3], pose[e, 4], pose[e, 5], pose[e, 6])


@wp.kernel
def pad_boxes_kernel(body_q: wp.array(dtype=wp.transform), finger: wp.array2d(dtype=wp.int32),
                     edge_pose: wp.array2d(dtype=wp.float64), corners: wp.array2d(dtype=wp.vec3d),
                     handle_center: wp.vec3, handle_half: wp.vec3, boxes: int,
                     box_centers: wp.array(dtype=wp.vec3), box_halves: wp.array(dtype=wp.vec3)):
    """dim = N. Device port of ``finger_geometry.pad_boxes_edge`` + ``isaac_lab_env._pad_boxes``: box 0 is the
    constant handle box, boxes 1-2 the edge-aligned bounds of the measured pad corners (float32 out, in place)."""
    e = wp.tid()
    ep, eq = _pose_from_row(edge_pose, e)
    R = _matrix(_normalize4(eq))
    box_centers[e*boxes] = handle_center
    box_halves[e*boxes] = handle_half
    for j in range(2):
        b = body_q[finger[e, j]]
        bp = _pose_p(b)
        Rb = _matrix(_normalize4(_pose_q(b)))
        lo = wp.vec3d(wp.float64(1.0e300), wp.float64(1.0e300), wp.float64(1.0e300))
        hi = wp.vec3d(wp.float64(-1.0e300), wp.float64(-1.0e300), wp.float64(-1.0e300))
        for c in range(8):
            local = corners[j, c]
            world = _vadd(_apply(Rb, local), bp)
            d = _vsub(world, ep)
            # R^T d
            x = wp.vec3d(_add(_add(_mul(R[0, 0], d[0]), _mul(R[1, 0], d[1])), _mul(R[2, 0], d[2])),
                         _add(_add(_mul(R[0, 1], d[0]), _mul(R[1, 1], d[1])), _mul(R[2, 1], d[2])),
                         _add(_add(_mul(R[0, 2], d[0]), _mul(R[1, 2], d[1])), _mul(R[2, 2], d[2])))
            for i in range(3):
                lo[i] = wp.min(lo[i], x[i])
                hi[i] = wp.max(hi[i], x[i])
        half_c = wp.vec3(wp.float32((lo[0]+hi[0])/wp.float64(2.0)), wp.float32((lo[1]+hi[1])/wp.float64(2.0)),
                         wp.float32((lo[2]+hi[2])/wp.float64(2.0)))
        half_h = wp.vec3(wp.float32((hi[0]-lo[0])/wp.float64(2.0)), wp.float32((hi[1]-lo[1])/wp.float64(2.0)),
                         wp.float32((hi[2]-lo[2])/wp.float64(2.0)))
        box_centers[e*boxes+1+j] = half_c
        box_halves[e*boxes+1+j] = half_h


# ----------------------------------------------------------------------------- wrench application

@wp.kernel
def apply_world_wrenches(body_q: wp.array(dtype=wp.transform), body_com: wp.array(dtype=wp.vec3),
                         tool: wp.array(dtype=wp.int32), hand: wp.array(dtype=wp.int32),
                         finger: wp.array2d(dtype=wp.int32), pad_box: wp.array(dtype=wp.int32),
                         edge_offset: wp.vec3, dt: float, feedback: int,
                         pending_wrench: wp.array2d(dtype=wp.float64), pad_wrench: wp.array3d(dtype=wp.float64),
                         external_wrench: wp.array2d(dtype=wp.float64),
                         body_f: wp.array(dtype=wp.spatial_vector), applied_impulse: wp.array2d(dtype=float),
                         applied_com: wp.array(dtype=wp.spatial_vector), applied_pose: wp.array(dtype=wp.transform),
                         pre_hand_wrench: wp.array(dtype=wp.spatial_vector),
                         applied_material: wp.array2d(dtype=wp.float64)):
    """dim = N. World e: material + external wrench at the current edge -> tool body; pad wrenches -> fingers.

    ``pending_wrench`` is the material output of the previous control step (one buffer, never copied), so the
    interval-(k+1) wrench equals the step-k output bitwise; ``applied_material`` records the float64 value read.
    Frozen float32 payloads: ``(pending + external)`` is summed in float64 and cast once, as the single env
    assigns ``np.asarray(applied, np.float32)``."""
    e = wp.tid()
    t = tool[e]
    pose = body_q[t]
    edge = wp.transform_point(pose, edge_offset)
    center = wp.transform_point(pose, body_com[t])
    w = wp.spatial_vector()
    for k in range(6):
        material = wp.float64(0.0)
        if feedback != 0:
            material = pending_wrench[e, k]
        applied_material[e, k] = material
        w[k] = wp.float32(material+external_wrench[e, k])
    force = wp.vec3(w[0], w[1], w[2])
    torque = wp.vec3(w[3], w[4], w[5])+wp.cross(edge-center, force)
    body_f[t] = body_f[t]+wp.spatial_vector(force, torque)
    applied_com[e] = wp.spatial_vector(force, torque)
    applied_pose[e] = pose
    pre_hand_wrench[e] = body_f[hand[e]]
    for k in range(3):
        applied_impulse[e, k] = applied_impulse[e, k]+dt*force[k]
        applied_impulse[e, k+3] = applied_impulse[e, k+3]+dt*torque[k]
    if feedback != 0:
        for j in range(2):
            body = finger[e, j]
            b = pad_box[j]
            pf = wp.vec3(wp.float32(pad_wrench[e, b, 0]), wp.float32(pad_wrench[e, b, 1]), wp.float32(pad_wrench[e, b, 2]))
            pt = wp.vec3(wp.float32(pad_wrench[e, b, 3]), wp.float32(pad_wrench[e, b, 4]), wp.float32(pad_wrench[e, b, 5]))
            c = wp.transform_point(body_q[body], body_com[body])
            body_f[body] = body_f[body]+wp.spatial_vector(pf, pt+wp.cross(edge-c, pf))


# ----------------------------------------------------------------------------- contacts, grasp monitor, validity

@wp.kernel
def collect_world_contacts(count: wp.array(dtype=int), shape0: wp.array(dtype=int), shape1: wp.array(dtype=int),
                           forces: wp.array(dtype=wp.spatial_vector), normal: wp.array(dtype=wp.vec3),
                           world_point: wp.array(dtype=wp.vec3), distance: wp.array(dtype=float),
                           shape_body: wp.array(dtype=int), shape_world: wp.array(dtype=wp.int32),
                           is_handle: wp.array(dtype=int), body_q: wp.array(dtype=wp.transform),
                           body_com: wp.array(dtype=wp.vec3), tool: wp.array(dtype=wp.int32),
                           finger: wp.array2d(dtype=wp.int32), num_envs: int, sample: wp.array2d(dtype=float)):
    """dim = contact pool. ``newton_tool.collect_tool_contacts`` with the contact's world as the row index."""
    i = wp.tid()
    if i < count[0]:
        a, b = shape0[i], shape1[i]
        if a >= 0 and b >= 0:
            e = wp.max(shape_world[a], shape_world[b])
            if e >= 0 and e < num_envs:
                ba, bb = shape_body[a], shape_body[b]
                penetration = wp.max(0., -distance[i])
                wp.atomic_max(sample, e, 13, float(count[0]))
                t = tool[e]
                if ba == t or bb == t:
                    sign = float(1.)
                    other = bb+0
                    tool_shape = a+0
                    if bb == t:
                        sign = -1.
                        other = ba+0
                        tool_shape = b+0
                    w = forces[i]*sign
                    f = wp.vec3(w[0], w[1], w[2])
                    center = wp.transform_point(body_q[t], body_com[t])
                    torque = wp.vec3(w[3], w[4], w[5])+wp.cross(world_point[i]-center, f)
                    for k in range(3):
                        wp.atomic_add(sample, e, k, f[k])
                        wp.atomic_add(sample, e, k+3, torque[k])
                    fn = wp.abs(wp.dot(f, normal[i]))
                    if is_handle[tool_shape] != 0 and other == finger[e, 0]:
                        wp.atomic_add(sample, e, 6, fn)
                        wp.atomic_add(sample, e, 14, 1.)
                    elif is_handle[tool_shape] != 0 and other == finger[e, 1]:
                        wp.atomic_add(sample, e, 7, fn)
                        wp.atomic_add(sample, e, 15, 1.)
                    else:
                        wp.atomic_add(sample, e, 11, 1.)
                    wp.atomic_max(sample, e, 10, penetration)
                else:
                    wp.atomic_max(sample, e, 12, penetration)


@wp.kernel
def integrate_world_contacts(sample: wp.array2d(dtype=float), dt: float, integral: wp.array2d(dtype=float)):
    """dim = N. ``newton_tool.integrate_contact_sample`` per world."""
    e = wp.tid()
    for k in range(8):
        integral[e, k] = integral[e, k]+sample[e, k]*dt
    if sample[e, 6] > 0.05 and sample[e, 7] > 0.05:
        integral[e, 8] = integral[e, 8]+1.
    integral[e, 9] = integral[e, 9]+1.
    integral[e, 10] = wp.max(integral[e, 10], sample[e, 10])
    integral[e, 11] = integral[e, 11]+sample[e, 11]
    integral[e, 12] = wp.max(integral[e, 12], sample[e, 12])
    integral[e, 13] = wp.max(integral[e, 13], sample[e, 13])
    integral[e, 14] = integral[e, 14]+sample[e, 14]
    integral[e, 15] = integral[e, 15]+sample[e, 15]


@wp.func
def _relative(hand: wp.transform, tool: wp.transform):
    """``inv(T_world_hand) @ T_world_tool`` in float64 (rotation, translation)."""
    Rh = _matrix(_normalize4(_pose_q(hand)))
    Rt = _matrix(_normalize4(_pose_q(tool)))
    d = _vsub(_pose_p(tool), _pose_p(hand))
    return wp.transpose(Rh)*Rt, wp.transpose(Rh)*d


@wp.kernel
def grasp_monitor_kernel(body_q: wp.array(dtype=wp.transform), hand: wp.array(dtype=wp.int32),
                         tool: wp.array(dtype=wp.int32), reference_R: wp.array(dtype=wp.mat33d),
                         reference_p: wp.array(dtype=wp.vec3d), integral: wp.array2d(dtype=float), dt: float,
                         translation_warning: float, rotation_warning: float, translation_loss: float,
                         rotation_loss: float, unsupported_loss_s: float, decimation: int,
                         unsupported_s: wp.array(dtype=float), lost: wp.array(dtype=wp.int32),
                         slip: wp.array(dtype=float), angle: wp.array(dtype=float), status: wp.array(dtype=wp.int32)):
    """dim = N. ``grasp_monitor.GraspMonitor.update`` per world (latched loss; operational, not scoring)."""
    e = wp.tid()
    R, p = _relative(body_q[hand[e]], body_q[tool[e]])
    Rr = reference_R[e]
    dR = wp.transpose(Rr)*R
    dp = wp.transpose(Rr)*(p-reference_p[e])
    translation = float(wp.length(dp))
    c = (float(dR[0, 0]+dR[1, 1]+dR[2, 2])-1.)/2.
    rotation = wp.acos(wp.clamp(c, -1., 1.))
    observed = integral[e, 9]
    fraction = float(0.)
    if observed > 0.:
        fraction = integral[e, 8]/observed
    if fraction == 0.:
        unsupported_s[e] = unsupported_s[e]+dt
    else:
        unsupported_s[e] = 0.
    warning = translation > translation_warning or rotation > rotation_warning
    if translation > translation_loss or rotation > rotation_loss or (warning and unsupported_s[e] >= unsupported_loss_s):
        lost[e] = 1
    slip[e] = translation
    angle[e] = rotation
    code = wp.int32(0)
    if int(observed) != decimation:
        code = wp.int32(3)
    elif lost[e] != 0:
        code = wp.int32(2)
    elif warning or fraction < .95:
        code = wp.int32(1)
    status[e] = code


@wp.kernel
def rigid_validity_kernel(joint_q: wp.array(dtype=float), coord_start: wp.array(dtype=wp.int32),
                          body_q: wp.array(dtype=wp.transform), tool: wp.array(dtype=wp.int32),
                          nacon: wp.array(dtype=int), naconmax: int, ik_streak: wp.array(dtype=wp.int32),
                          ik_streak_limit: int, edge_status: wp.array(dtype=wp.int32),
                          reason: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    """dim = N. Sticky per-world rigid flags; ``count[0]`` = number of invalid worlds (the strict-mode read)."""
    e = wp.tid()
    r = reason[e]
    for i in range(coord_start[e], coord_start[e+1]):
        if not wp.isfinite(joint_q[i]):
            r = r | 1
    if not _transform_finite(body_q[tool[e]]):
        r = r | 1
    if nacon[0] > naconmax:
        r = r | 2
    if ik_streak_limit > 0 and ik_streak[e] >= ik_streak_limit:
        r = r | 4
    if edge_status[e] != 0:
        r = r | 8
    reason[e] = r
    if r != 0:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def masked_zero_rows(mask: wp.array(dtype=wp.bool), values: wp.array2d(dtype=wp.float64)):
    """dim = (N, width). Zero the rows of the masked worlds (reset of pending/external wrench buffers)."""
    e, k = wp.tid()
    if mask[e]:
        values[e, k] = wp.float64(0.0)


@wp.kernel
def masked_zero_rows_f32(mask: wp.array(dtype=wp.bool), values: wp.array2d(dtype=float)):
    e, k = wp.tid()
    if mask[e]:
        values[e, k] = 0.


# ----------------------------------------------------------------------------- host helpers

class RobotFrames:
    """Device frame/wrench state of N worlds plus the constant chains (one per batch)."""

    def __init__(self, tables, spec, hand_from_edge, tool_mount, *, pad_corners=None, handle_box=None,
                 device="cpu", boxes=3):
        self.device = device
        self.num_envs = len(tables["hand"])
        self.tool_mount = tool_mount
        self.boxes = int(boxes)
        N = self.num_envs
        with wp.ScopedDevice(device):
            self.hand = wp.array(tables["hand"], dtype=wp.int32)
            self.tool = wp.array(tables["tool"], dtype=wp.int32)
            self.finger = wp.array(tables["finger"], dtype=wp.int32)
            if tool_mount == "rigid_mount_v1":
                chain = [np.asarray(spec.hand_from_tool_xyz_xyzw, float), np.asarray(spec.tool_from_edge_xyz_xyzw, float)]
                self.source = self.hand
            else:
                chain = [np.asarray(spec.tool_from_edge_xyz_xyzw, float)]
                self.source = self.tool
            self.chain = chain
            self.chain_p = wp.array(np.stack([c[:3] for c in chain]), dtype=wp.vec3d)
            self.chain_q = wp.array(np.stack([c[3:] for c in chain]), dtype=wp.vec4d)
            mount = np.asarray(hand_from_edge, float)
            self.hand_edge_p = wp.vec3d(*mount[:3])
            self.hand_edge_q = wp.vec4d(*mount[3:])
            self.edge_offset = wp.vec3(*np.asarray(spec.tool_from_edge_xyz_xyzw, float)[:3])
            self.measured_edge = wp.zeros((N, 7), dtype=wp.float64)
            self.nominal_edge = wp.zeros((N, 7), dtype=wp.float64)
            self.edge_status = wp.zeros(N, dtype=wp.int32)
            self.pending_wrench = wp.zeros((N, 6), dtype=wp.float64)
            self.pad_wrench = wp.zeros((N, self.boxes, 6), dtype=wp.float64)
            self.external_wrench = wp.zeros((N, 6), dtype=wp.float64)
            self.applied_impulse = wp.zeros((N, 6), dtype=float)
            self.applied_com = wp.zeros(N, dtype=wp.spatial_vector)
            self.applied_pose = wp.zeros(N, dtype=wp.transform)
            self.pre_hand_wrench = wp.zeros(N, dtype=wp.spatial_vector)
            self.applied_material = wp.zeros((N, 6), dtype=wp.float64)
            self.pad_box = wp.array(np.array([1, 2], np.int32), dtype=wp.int32)
            if pad_corners is not None:
                corners = np.asarray(pad_corners, float)
                if corners.shape != (2, 8, 3):
                    raise ValueError("pad corners must be [2 pads, 8, 3] in finger-body coordinates")
                self.pad_corners = wp.array(corners, dtype=wp.vec3d)
                center, half = (np.asarray(v, float) for v in handle_box)
                self.handle_center = wp.vec3(*center.astype(np.float32))
                self.handle_half = wp.vec3(*half.astype(np.float32))
                self.box_centers = wp.zeros(N*self.boxes, dtype=wp.vec3)
                self.box_halves = wp.zeros(N*self.boxes, dtype=wp.vec3)

    def launch_edge(self, body_q):
        wp.launch(edge_pose_kernel, self.num_envs, device=self.device,
                  inputs=[body_q, self.source, self.hand, self.chain_p, self.chain_q, len(self.chain),
                          self.hand_edge_p, self.hand_edge_q, self.measured_edge, self.nominal_edge, self.edge_status])

    def launch_pad_boxes(self, body_q, box_centers=None, box_halves=None):
        """Write the 3 boxes per env in place (default: own buffers; pass the material's arrays to write there)."""
        wp.launch(pad_boxes_kernel, self.num_envs, device=self.device,
                  inputs=[body_q, self.finger, self.measured_edge, self.pad_corners, self.handle_center,
                          self.handle_half, self.boxes,
                          self.box_centers if box_centers is None else box_centers,
                          self.box_halves if box_halves is None else box_halves])

    def launch_wrench(self, body_q, body_com, body_f, dt, feedback=True, pending_wrench=None, pad_wrench=None):
        wp.launch(apply_world_wrenches, self.num_envs, device=self.device,
                  inputs=[body_q, body_com, self.tool, self.hand, self.finger, self.pad_box, self.edge_offset,
                          float(dt), int(bool(feedback)),
                          self.pending_wrench if pending_wrench is None else pending_wrench,
                          self.pad_wrench if pad_wrench is None else pad_wrench, self.external_wrench, body_f,
                          self.applied_impulse, self.applied_com, self.applied_pose, self.pre_hand_wrench,
                          self.applied_material])

    def reset_masked(self, mask):
        """Zero pending/pad/external wrench and impulse rows of the masked worlds (ADR 0022 §6.4)."""
        N = self.num_envs
        for values in (self.pending_wrench, self.external_wrench, self.applied_material):
            wp.launch(masked_zero_rows, (N, 6), device=self.device, inputs=[mask, values])
        wp.launch(masked_zero_rows, (N, self.boxes*6), device=self.device,
                  inputs=[mask, self.pad_wrench.reshape((N, self.boxes*6))])
        wp.launch(masked_zero_rows_f32, (N, 6), device=self.device, inputs=[mask, self.applied_impulse])


def numpy_edge_reference(body_q, source, hand, chain, hand_from_edge):
    """The frozen NumPy path the edge kernel reproduces (``scene_frames.compose_pose``,
    ``robot.compose_tool_pose``) for host float32 ``body_q`` rows."""
    from ..robot import compose_tool_pose
    from ..scene_frames import compose_pose
    measured, nominal = [], []
    body_q = np.asarray(body_q)
    for s, h in zip(source, hand):
        pose = body_q[s].astype(float)
        for link in chain:
            pose = compose_pose(pose, link)
        measured.append(pose)
        mount = np.asarray(hand_from_edge, float)
        nominal.append(compose_tool_pose(body_q[h].astype(float), mount[:3], mount[3:]))
    return np.asarray(measured), np.asarray(nominal)


def numpy_wrench_reference(body_q, body_com, tool, hand, finger, edge_offset, pending, pad, external, feedback=True):
    """Host lever-arm reference of :func:`apply_world_wrenches` (float32 payloads as the frozen buffers)."""
    from ..scene_frames import pose_matrix
    body_q, body_com = np.asarray(body_q), np.asarray(body_com)
    out = {}
    for e in range(len(tool)):
        T = pose_matrix(body_q[tool[e]].astype(float))
        edge = T[:3, :3] @ np.asarray(edge_offset, float)+T[:3, 3]
        center = T[:3, :3] @ body_com[tool[e]].astype(float)+T[:3, 3]
        w = (np.asarray(pending[e], float)*(1. if feedback else 0.)+np.asarray(external[e], float)).astype(np.float32)
        w = w.astype(float)
        out[int(tool[e])] = np.r_[w[:3], w[3:]+np.cross(edge-center, w[:3])]
        if feedback:
            for j in range(2):
                body = int(finger[e][j])
                F = pose_matrix(body_q[body].astype(float))
                c = F[:3, :3] @ body_com[body].astype(float)+F[:3, 3]
                p = np.asarray(pad[e][1+j], float).astype(np.float32).astype(float)
                out[body] = np.r_[p[:3], p[3:]+np.cross(edge-c, p[:3])]
    return out


__all__ = ["ROBOT_KERNELS_VERSION", "resolve_index_tables", "resolve_joint_tables", "edge_pose_kernel",
           "pad_boxes_kernel", "apply_world_wrenches", "collect_world_contacts", "integrate_world_contacts",
           "grasp_monitor_kernel", "rigid_validity_kernel", "RobotFrames", "numpy_edge_reference",
           "numpy_wrench_reference", "decode_rigid_reason", "RIGID_REASONS", "SAMPLE_WIDTH"]
