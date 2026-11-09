"""Device pose interpolation for the batched material coupling (ADR 0021 §7.1, ``device_f64_slerp_v1``).

One Warp kernel reproduces, per env and per substep sample, the frozen host path of a
control step::

    live_material.advance -> coupled_robot.interpolate_measured_poses(
        scene_frames.transform_poses(T_local_world, last_pose),
        scene_frames.transform_poses(T_local_world, measured_pose), substeps, control_dt)

It computes in float64 with the same operation sequence as SciPy's numpy (Cython)
``Rotation`` backend (``from_quat`` / ``__mul__`` normalize; ``inv`` negates xyz;
``as_rotvec`` takes the canonical w >= 0 hemisphere; ``Slerp`` evaluates
``q0 * exp(f * log(q0^-1 q1))``) and casts to float32 last, exactly where the frozen
path calls ``astype(np.float32)``.

Outputs are written in place into preallocated arrays with the layout of the frozen
``TLMPM.set_trajectory`` (``tlmpm_v5.py:803-814``), batched over envs:

* ``poses[N, T]``  ``wp.transform``      = the frozen ``(T, 7)`` float32 XYZ+XYZW rows;
* ``twists[N, T]`` ``wp.spatial_vector`` = the frozen ``(T, 6)`` float32 ``[linear, angular]`` rows;

with ``T = substeps + 1``. A flat ``[N*T]`` solver array can be passed as
``array.reshape((N, T))`` (a view). The launch allocates nothing and is graph-capturable.

The per-env workpiece rotation quaternion ``q_local_world[e]`` is computed once on the
host at pack time with ``Rotation.from_matrix`` -- the call ``transform_poses`` repeats
on every control step -- so it is bit-identical by construction.

Validity: the frozen path raises on a non-finite pose (``validate_pose``) or a quaternion
whose norm is not within 1e-6 of 1 (``validate_pose_xyzw``). The kernel cannot raise; it
writes ``status[e]`` (``STATUS_OK``, ``STATUS_NONFINITE``, ``STATUS_NON_UNIT_QUATERNION``)
and the outputs of a non-OK env are unspecified. The caller must treat such an env as
invalid (strict mode: raise; training mode: terminate), never continue silently.
"""
import numpy as np
import warp as wp

POSE_INTERPOLATION_VERSION = "device_f64_slerp_v1"
STATUS_OK = 0
STATUS_NONFINITE = 1
STATUS_NON_UNIT_QUATERNION = 2
STATUS_NAMES = ("OK", "NONFINITE_POSE", "NON_UNIT_QUATERNION")
UNIT_TOLERANCE = 1e-6   # scene_frames.validate_pose_xyzw


# ----------------------------------------------------------------------------- exact float64 arithmetic
# Every product/sum below is one IEEE round-to-nearest operation, never contracted into an FMA by the
# compiler (CUDA: the __d*_rn intrinsics are never fused; NVRTC would otherwise fuse a*b+c). Where the
# frozen host path itself uses an FMA chain (OpenBLAS ddot behind np.linalg.norm, the small dgemv behind
# transform_points' matmul), _fma reproduces it. tests/test_batched_pose_interp.py asserts these host
# orders on the machine that runs the test, so a different BLAS kernel fails loudly, not silently.

_MUL = """
#if defined(__CUDA_ARCH__)
return __dmul_rn(a, b);
#else
return a * b;
#endif
"""
_ADD = """
#if defined(__CUDA_ARCH__)
return __dadd_rn(a, b);
#else
return a + b;
#endif
"""
_SUB = """
#if defined(__CUDA_ARCH__)
return __dsub_rn(a, b);
#else
return a - b;
#endif
"""


@wp.func_native(_MUL)
def _mul(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_ADD)
def _add(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_SUB)
def _sub(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native("return fma(a, b, c);")
def _fma(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64: ...


@wp.func
def _norm4(q: wp.vec4d):
    """SciPy Cython quaternion norm: plain sequential sum of squares."""
    s = _add(_add(_add(_mul(q[0], q[0]), _mul(q[1], q[1])), _mul(q[2], q[2])), _mul(q[3], q[3]))
    return wp.sqrt(s)


@wp.func
def _norm3(v: wp.vec3d):
    return wp.sqrt(_add(_add(_mul(v[0], v[0]), _mul(v[1], v[1])), _mul(v[2], v[2])))


@wp.func
def _norm4_blas(q: wp.vec4d):
    """np.linalg.norm (OpenBLAS ddot): fma(x3,x3, fma(x2,x2, fma(x1,x1, x0*x0)))."""
    return wp.sqrt(_fma(q[3], q[3], _fma(q[2], q[2], _fma(q[1], q[1], _mul(q[0], q[0])))))


@wp.func
def _normalize4(q: wp.vec4d):
    n = _norm4(q)
    return wp.vec4d(q[0]/n, q[1]/n, q[2]/n, q[3]/n)


@wp.func
def _compose(p: wp.vec4d, q: wp.vec4d):
    """SciPy ``compose_quat`` (Hamilton product p*q, XYZW), followed by the normalization of ``Rotation(...)``."""
    c0 = _sub(_mul(p[1], q[2]), _mul(p[2], q[1]))
    c1 = _sub(_mul(p[2], q[0]), _mul(p[0], q[2]))
    c2 = _sub(_mul(p[0], q[1]), _mul(p[1], q[0]))
    x = _add(_add(_mul(p[3], q[0]), _mul(q[3], p[0])), c0)
    y = _add(_add(_mul(p[3], q[1]), _mul(q[3], p[1])), c1)
    z = _add(_add(_mul(p[3], q[2]), _mul(q[3], p[2])), c2)
    w = _sub(_sub(_sub(_mul(p[3], q[3]), _mul(p[0], q[0])), _mul(p[1], q[1])), _mul(p[2], q[2]))
    return _normalize4(wp.vec4d(x, y, z, w))


@wp.func
def _inv(q: wp.vec4d):
    return wp.vec4d(-q[0], -q[1], -q[2], q[3])


@wp.func
def _as_rotvec(q_in: wp.vec4d):
    """SciPy ``as_rotvec``: canonical hemisphere, 2*atan2, Taylor branch at angle <= 1e-3."""
    zero = wp.float64(0.0)
    q = q_in
    flip = q[3] < zero
    if q[3] == zero:
        if q[0] < zero:
            flip = True
        elif q[0] == zero:
            if q[1] < zero:
                flip = True
            elif q[1] == zero:
                if q[2] < zero:
                    flip = True
    if flip:
        q = -q
    ax = _norm3(wp.vec3d(q[0], q[1], q[2]))
    angle = wp.float64(2.0)*wp.atan2(ax, q[3])
    scale = wp.float64(0.0)
    if angle <= wp.float64(1.0e-3):
        a2 = _mul(angle, angle)
        scale = _add(_add(wp.float64(2.0), a2/wp.float64(12.0)), _mul(_mul(wp.float64(7.0), a2), a2)/wp.float64(2880.0))
    else:
        scale = angle/wp.sin(angle/wp.float64(2.0))
    return wp.vec3d(_mul(scale, q[0]), _mul(scale, q[1]), _mul(scale, q[2]))


@wp.func
def _from_rotvec(r: wp.vec3d):
    """SciPy ``from_rotvec``: Taylor branch at angle <= 1e-3 (not renormalized, as in SciPy)."""
    angle = _norm3(r)
    scale = wp.float64(0.0)
    if angle <= wp.float64(1.0e-3):
        a2 = _mul(angle, angle)
        scale = _add(_sub(wp.float64(0.5), a2/wp.float64(48.0)), _mul(a2, a2)/wp.float64(3840.0))
    else:
        scale = wp.sin(angle/wp.float64(2.0))/angle
    return wp.vec4d(_mul(scale, r[0]), _mul(scale, r[1]), _mul(scale, r[2]), wp.cos(angle/wp.float64(2.0)))


@wp.func
def _finite7(pose: wp.array2d(dtype=wp.float64), e: int):
    ok = True
    for i in range(7):
        if not wp.isfinite(pose[e, i]):
            ok = False
    return ok


@wp.func
def _local_position(T: wp.mat44d, pose: wp.array2d(dtype=wp.float64), e: int):
    """``transform_points``: ``p @ R.T + t`` (host dgemv order: fma(p2,R2, fma(p0,R0, p1*R1)), then + t)."""
    p0 = pose[e, 0]
    p1 = pose[e, 1]
    p2 = pose[e, 2]
    return wp.vec3d(_add(_fma(p2, T[0, 2], _fma(p0, T[0, 0], _mul(p1, T[0, 1]))), T[0, 3]),
                    _add(_fma(p2, T[1, 2], _fma(p0, T[1, 0], _mul(p1, T[1, 1]))), T[1, 3]),
                    _add(_fma(p2, T[2, 2], _fma(p0, T[2, 0], _mul(p1, T[2, 1]))), T[2, 3]))


@wp.func
def _local_quaternion(qT: wp.vec4d, pose: wp.array2d(dtype=wp.float64), e: int):
    """``transform_poses`` (from_quat normalizes, compose normalizes), then ``validate_pose`` and the
    interpolation's ``Rotation.from_quat`` (two further normalizations), in that order."""
    q = _normalize4(wp.vec4d(pose[e, 3], pose[e, 4], pose[e, 5], pose[e, 6]))
    q = _compose(qT, q)
    n = _norm4_blas(q)                       # robot.validate_pose: q / np.linalg.norm(q)
    q = wp.vec4d(q[0]/n, q[1]/n, q[2]/n, q[3]/n)
    return _normalize4(q)                    # Rotation.from_quat


@wp.kernel
def pose_interp(last_pose: wp.array2d(dtype=wp.float64), measured_pose: wp.array2d(dtype=wp.float64),
                T_local_world: wp.array(dtype=wp.mat44d), q_local_world: wp.array(dtype=wp.vec4d),
                substeps: int, control_dt: wp.float64,
                poses: wp.array2d(dtype=wp.transform), twists: wp.array2d(dtype=wp.spatial_vector),
                status: wp.array(dtype=wp.int32)):
    """dim = (N, substeps+1). World measured poses -> per-substep LOCAL tool poses and twists (float32)."""
    e, k = wp.tid()
    T = T_local_world[e]
    qT = q_local_world[e]
    if k == 0:
        code = wp.int32(0)
        if not (_finite7(last_pose, e) and _finite7(measured_pose, e)):
            code = wp.int32(1)
        else:
            na = _norm4_blas(wp.vec4d(last_pose[e, 3], last_pose[e, 4], last_pose[e, 5], last_pose[e, 6]))
            nb = _norm4_blas(wp.vec4d(measured_pose[e, 3], measured_pose[e, 4], measured_pose[e, 5], measured_pose[e, 6]))
            if wp.abs(na - wp.float64(1.0)) > wp.float64(1.0e-6) or wp.abs(nb - wp.float64(1.0)) > wp.float64(1.0e-6):
                code = wp.int32(2)
        status[e] = code
    pa = _local_position(T, last_pose, e)
    pb = _local_position(T, measured_pose, e)
    q0 = _local_quaternion(qT, last_pose, e)
    q1 = _local_quaternion(qT, measured_pose, e)
    f = wp.float64(k)/wp.float64(substeps)
    # Slerp([0, 1], [q0, q1])(f) = q0 * from_rotvec(as_rotvec(q0^-1 q1) * f)
    r = _as_rotvec(_compose(_inv(q0), q1))
    qk = _compose(q0, _from_rotvec(wp.vec3d(_mul(r[0], f), _mul(r[1], f), _mul(r[2], f))))
    d = wp.vec3d(_sub(pb[0], pa[0]), _sub(pb[1], pa[1]), _sub(pb[2], pa[2]))
    x = wp.vec3d(_add(pa[0], _mul(f, d[0])), _add(pa[1], _mul(f, d[1])), _add(pa[2], _mul(f, d[2])))
    w = _as_rotvec(_compose(q1, _inv(q0)))
    poses[e, k] = wp.transform(wp.vec3(wp.float32(x[0]), wp.float32(x[1]), wp.float32(x[2])),
                               wp.quat(wp.float32(qk[0]), wp.float32(qk[1]), wp.float32(qk[2]), wp.float32(qk[3])))
    twists[e, k] = wp.spatial_vector(wp.float32(d[0]/control_dt), wp.float32(d[1]/control_dt),
                                     wp.float32(d[2]/control_dt), wp.float32(w[0]/control_dt),
                                     wp.float32(w[1]/control_dt), wp.float32(w[2]/control_dt))


@wp.kernel
def fill_cut_rows(cut: wp.array(dtype=wp.int32), cuts: wp.array2d(dtype=wp.int32)):
    """dim = (N, T). ``cuts[e, :] = cut[e]`` as the frozen ``np.full(substeps+1, cut, np.int32)``."""
    e, k = wp.tid()
    cuts[e, k] = cut[e]


def local_frames(T_local_world, device=None):
    """Pack-time host step: per-env ``[N,4,4]`` T_local_world -> (``wp.mat44d[N]``, ``wp.vec4d[N]`` quaternion).

    The quaternion is ``Rotation.from_matrix(T[:3,:3]).as_quat()``, exactly what
    ``scene_frames.transform_poses`` computes for this transform. Each matrix is checked with
    ``scene_frames.validate_transform`` (proper rigid transform).
    """
    from scipy.spatial.transform import Rotation

    from ..scene_frames import validate_transform
    T = np.asarray(T_local_world, dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (4, 4) or len(T) < 1:
        raise ValueError("T_local_world must be [N,4,4]")
    T = np.stack([validate_transform(m) for m in T])
    q = np.stack([Rotation.from_matrix(m[:3, :3]).as_quat() for m in T]).astype(np.float64)
    return (wp.array(T, dtype=wp.mat44d, device=device), wp.array(q, dtype=wp.vec4d, device=device))


def launch_pose_interp(last_pose, measured_pose, T_local_world, q_local_world, poses, twists, status,
                       substeps, control_dt, device=None, stream=None):
    """Launch :func:`pose_interp` over ``(N, substeps+1)``; validates shapes only (no host sync, no allocation)."""
    N = last_pose.shape[0]
    T = int(substeps)+1
    if (substeps < 1 or last_pose.shape != (N, 7) or measured_pose.shape != (N, 7) or poses.shape != (N, T)
            or twists.shape != (N, T) or T_local_world.shape != (N,) or q_local_world.shape != (N,)
            or status.shape != (N,)):
        raise ValueError(f"pose_interp shapes: N={N}, T={T}; got last {last_pose.shape}, measured "
                         f"{measured_pose.shape}, poses {poses.shape}, twists {twists.shape}, "
                         f"T_local_world {T_local_world.shape}, q {q_local_world.shape}, status {status.shape}")
    if last_pose.dtype != wp.float64 or measured_pose.dtype != wp.float64:
        raise ValueError("pose_interp inputs are float64 (the frozen host path is float64 until the final cast)")
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be positive and finite")
    wp.launch(pose_interp, dim=(N, T), device=device if device is not None else poses.device, stream=stream,
              inputs=[last_pose, measured_pose, T_local_world, q_local_world, int(substeps),
                      wp.float64(control_dt), poses, twists, status])


def launch_fill_cuts(cut, cuts, device=None, stream=None):
    N, T = cuts.shape
    if cut.shape != (N,):
        raise ValueError("cut must be [N] for cuts [N,T]")
    wp.launch(fill_cut_rows, dim=(N, T), device=device if device is not None else cuts.device, stream=stream,
              inputs=[cut, cuts])
