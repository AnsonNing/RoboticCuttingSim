"""Device pose interpolation for the batched material coupling (ADR 0021 §7.1, ``device_f64_slerp_v2``).

Two Warp kernels reproduce, per env and per substep sample, the live host chain of one
``LiveCuttingMaterial.advance`` control step (``live_material.py:321,334,376``)::

    actual = robot.validate_pose(measured_pose)                 # finite, nonzero; q / np.linalg.norm(q)
    poses, twists = coupled_robot.interpolate_measured_poses(
        scene_frames.transform_poses(T_local_world, last_pose),  # last_pose = previous step's ``actual``
        scene_frames.transform_poses(T_local_world, actual), substeps, control_dt)
    last_pose = actual

* :func:`validate_measured` (dim N) is ``validate_pose`` plus the checks of ``transform_poses``: it writes
  ``actual[N,7]`` (the measured pose with its quaternion divided by the OpenBLAS ``np.linalg.norm``) and
  ``status[N]``. The caller stores ``actual`` as the next ``last_pose``, exactly as the live material does.
* :func:`pose_interp` (dim ``(N, substeps+1)``) is ``transform_poses`` + ``interpolate_measured_poses`` from
  ``last_pose`` to ``actual``.

They compute in float64 with the same operation sequence as SciPy's numpy (Cython)
``Rotation`` backend (``from_quat`` / ``__mul__`` normalize; ``inv`` negates xyz;
``as_rotvec`` takes the canonical w >= 0 hemisphere; ``Slerp`` evaluates
``q0 * exp(f * log(q0^-1 q1))``) and cast to float32 last, exactly where the frozen
path calls ``astype(np.float32)``.

Outputs are written in place into preallocated arrays with the layout of the frozen
``TLMPM.set_trajectory`` (``tlmpm_v5.py:803-814``), batched over envs:

* ``poses[N, T]``  ``wp.transform``      = the frozen ``(T, 7)`` float32 XYZ+XYZW rows;
* ``twists[N, T]`` ``wp.spatial_vector`` = the frozen ``(T, 6)`` float32 ``[linear, angular]`` rows;

with ``T = substeps + 1``. A flat ``[N*T]`` solver array can be passed as
``array.reshape((N, T))`` (a view). With a preallocated ``actual`` the launches allocate nothing
and are graph-capturable.

The per-env workpiece rotation quaternion ``q_local_world[e]`` is computed once on the
host at pack time with ``Rotation.from_matrix`` -- the call ``transform_poses`` repeats
on every control step -- so it is bit-identical by construction.

Validity (``status[e]``), mirroring what the live chain raises on:

* ``STATUS_NONFINITE``: a non-finite ``last_pose`` or ``measured_pose`` (``validate_pose`` /
  ``validate_pose_xyzw``);
* ``STATUS_ZERO_QUATERNION``: measured quaternion norm < 1e-8 (``validate_pose``: "Quaternion must be
  nonzero");
* ``STATUS_NON_UNIT_QUATERNION``: ``last_pose`` quaternion norm not within 1e-6 of 1, or a measured norm
  that overflows (``transform_poses`` -> ``validate_pose_xyzw``).

A measured quaternion of any other finite, nonzero norm is accepted and normalized, as the live material
accepts it (``device_f64_slerp_v1`` refused norms outside 1 +- 1e-6 and normalized in a different order,
which differed from the live chain by up to ~1e3 float32 ulp in the twists of 1e-9..1e-6 rad rotations
of float32-rounded quaternions). The kernels cannot raise; the outputs of a non-OK env are unspecified and
the caller must treat that env as invalid (strict mode: raise; training mode: terminate), never continue
silently.
"""
import numpy as np
import warp as wp

POSE_INTERPOLATION_VERSION = "device_f64_slerp_v2"
STATUS_OK = 0
STATUS_NONFINITE = 1
STATUS_NON_UNIT_QUATERNION = 2
STATUS_ZERO_QUATERNION = 3
STATUS_NAMES = ("OK", "NONFINITE_POSE", "NON_UNIT_QUATERNION", "ZERO_QUATERNION")
UNIT_TOLERANCE = 1e-6   # scene_frames.validate_pose_xyzw
NONZERO_TOLERANCE = 1e-8   # robot.validate_pose


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
    """For an already validated pose (``last_pose`` or ``actual``): ``transform_poses`` (from_quat
    normalizes, compose normalizes), then ``interpolate_measured_poses``' own ``validate_pose`` and
    ``Rotation.from_quat`` (two further normalizations), in that order."""
    q = _normalize4(wp.vec4d(pose[e, 3], pose[e, 4], pose[e, 5], pose[e, 6]))
    q = _compose(qT, q)
    n = _norm4_blas(q)                       # robot.validate_pose: q / np.linalg.norm(q)
    q = wp.vec4d(q[0]/n, q[1]/n, q[2]/n, q[3]/n)
    return _normalize4(q)                    # Rotation.from_quat


@wp.kernel
def validate_measured(last_pose: wp.array2d(dtype=wp.float64), measured_pose: wp.array2d(dtype=wp.float64),
                      actual: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    """dim = N. ``actual = validate_pose(measured)`` and the live chain's checks (module docstring)."""
    e = wp.tid()
    code = wp.int32(0)
    nb = wp.float64(1.0)
    if not (_finite7(last_pose, e) and _finite7(measured_pose, e)):
        code = wp.int32(1)
    else:
        nb = _norm4_blas(wp.vec4d(measured_pose[e, 3], measured_pose[e, 4], measured_pose[e, 5], measured_pose[e, 6]))
        na = _norm4_blas(wp.vec4d(last_pose[e, 3], last_pose[e, 4], last_pose[e, 5], last_pose[e, 6]))
        if nb < wp.float64(1.0e-8):
            code = wp.int32(3)
        elif not wp.isfinite(nb) or wp.abs(na - wp.float64(1.0)) > wp.float64(1.0e-6):
            code = wp.int32(2)
    for i in range(3):
        actual[e, i] = measured_pose[e, i]
    for i in range(3, 7):
        if code == wp.int32(0):
            actual[e, i] = measured_pose[e, i]/nb
        else:
            actual[e, i] = measured_pose[e, i]
    status[e] = code


@wp.kernel
def pose_interp(last_pose: wp.array2d(dtype=wp.float64), actual: wp.array2d(dtype=wp.float64),
                T_local_world: wp.array(dtype=wp.mat44d), q_local_world: wp.array(dtype=wp.vec4d),
                substeps: int, control_dt: wp.float64,
                poses: wp.array2d(dtype=wp.transform), twists: wp.array2d(dtype=wp.spatial_vector)):
    """dim = (N, substeps+1). Validated WORLD poses -> per-substep LOCAL tool poses and twists (float32)."""
    e, k = wp.tid()
    T = T_local_world[e]
    qT = q_local_world[e]
    pa = _local_position(T, last_pose, e)
    pb = _local_position(T, actual, e)
    q0 = _local_quaternion(qT, last_pose, e)
    q1 = _local_quaternion(qT, actual, e)
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
    """dim = (N, T). ``cuts[e, :] = cut[e]`` as the frozen ``np.full(substeps+1, cut, np.int32)`` (unchecked)."""
    e, k = wp.tid()
    cuts[e, k] = cut[e]


@wp.kernel
def fill_cut_rows_checked(cut: wp.array(dtype=wp.int32), max_cuts: int, flag: int,
                          cuts: wp.array2d(dtype=wp.int32), invalid_reason: wp.array(dtype=int)):
    """dim = (N, T). As :func:`fill_cut_rows`, but a digit outside ``-1..max_cuts-1`` (the frozen
    ``LiveCuttingMaterial.advance`` refusal "Explicit cut index exceeds retained material history") is
    written as ``-1`` (disarmed) and ``flag`` is OR-ed into ``invalid_reason[e]`` before any substep runs,
    so an invalid digit never labels particles (a digit >= 16 would overflow the 2-bit packed code)."""
    e, k = wp.tid()
    digit = cut[e]
    if digit < -1 or digit >= max_cuts:
        cuts[e, k] = -1
        if k == 0:
            invalid_reason[e] = invalid_reason[e] | flag
    else:
        cuts[e, k] = digit


@wp.kernel
def disarm_invalid_cut_rows(cuts: wp.array2d(dtype=wp.int32), max_cuts: int, flag: int,
                            invalid_reason: wp.array(dtype=int)):
    """dim = N. Caller-written ``cuts[N, T]`` (``advance(interpolation="preset")``): every digit outside
    ``-1..max_cuts-1`` becomes ``-1`` and the env is flagged, before any substep runs."""
    e = wp.tid()
    bad = int(0)
    for k in range(cuts.shape[1]):
        digit = cuts[e, k]
        if digit < -1 or digit >= max_cuts:
            cuts[e, k] = -1
            bad = 1
    if bad != 0:
        invalid_reason[e] = invalid_reason[e] | flag


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


def launch_validate_measured(last_pose, measured_pose, actual, status, device=None, stream=None):
    """Launch :func:`validate_measured` over N (``actual`` and ``status`` are outputs)."""
    N = last_pose.shape[0]
    if last_pose.shape != (N, 7) or measured_pose.shape != (N, 7) or actual.shape != (N, 7) or status.shape != (N,):
        raise ValueError(f"validate_measured shapes: last {last_pose.shape}, measured {measured_pose.shape}, "
                         f"actual {actual.shape}, status {status.shape}")
    if last_pose.dtype != wp.float64 or measured_pose.dtype != wp.float64 or actual.dtype != wp.float64:
        raise ValueError("pose_interp inputs are float64 (the frozen host path is float64 until the final cast)")
    wp.launch(validate_measured, dim=N, device=device if device is not None else actual.device, stream=stream,
              inputs=[last_pose, measured_pose, actual, status])


def launch_pose_interp(last_pose, measured_pose, T_local_world, q_local_world, poses, twists, status,
                       substeps, control_dt, device=None, stream=None, *, actual=None):
    """Launch :func:`validate_measured` (N) then :func:`pose_interp` (``(N, substeps+1)``).

    ``last_pose`` is the previous step's validated pose (``LiveCuttingMaterial.last_pose``); ``actual``
    (``[N,7]`` float64, output) receives ``validate_pose(measured_pose)`` -- the caller's next ``last_pose``.
    Production passes a preallocated ``actual`` (then nothing is allocated and nothing syncs); without it a
    scratch array is allocated (component tests). Validates shapes only."""
    N = last_pose.shape[0]
    T = int(substeps)+1
    if (substeps < 1 or last_pose.shape != (N, 7) or measured_pose.shape != (N, 7) or poses.shape != (N, T)
            or twists.shape != (N, T) or T_local_world.shape != (N,) or q_local_world.shape != (N,)
            or status.shape != (N,)):
        raise ValueError(f"pose_interp shapes: N={N}, T={T}; got last {last_pose.shape}, measured "
                         f"{measured_pose.shape}, poses {poses.shape}, twists {twists.shape}, "
                         f"T_local_world {T_local_world.shape}, q {q_local_world.shape}, status {status.shape}")
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be positive and finite")
    device = device if device is not None else poses.device
    if actual is None:
        actual = wp.empty((N, 7), dtype=wp.float64, device=device)
    launch_validate_measured(last_pose, measured_pose, actual, status, device=device, stream=stream)
    wp.launch(pose_interp, dim=(N, T), device=device, stream=stream,
              inputs=[last_pose, actual, T_local_world, q_local_world, int(substeps), wp.float64(control_dt),
                      poses, twists])
    return actual


def launch_fill_cuts(cut, cuts, device=None, stream=None, *, max_cuts=None, invalid_reason=None, flag=8):
    """Fill ``cuts[N,T]`` from per-env digits. With ``max_cuts`` (production), out-of-range digits are
    disarmed (-1) and ``flag`` is OR-ed into ``invalid_reason[N]`` (:func:`fill_cut_rows_checked`)."""
    N, T = cuts.shape
    if cut.shape != (N,):
        raise ValueError("cut must be [N] for cuts [N,T]")
    device = device if device is not None else cuts.device
    if max_cuts is None:
        wp.launch(fill_cut_rows, dim=(N, T), device=device, stream=stream, inputs=[cut, cuts])
        return
    if invalid_reason is None or invalid_reason.shape != (N,):
        raise ValueError("checked cut fill needs invalid_reason [N]")
    wp.launch(fill_cut_rows_checked, dim=(N, T), device=device, stream=stream,
              inputs=[cut, int(max_cuts), int(flag), cuts, invalid_reason])


def launch_disarm_invalid_cuts(cuts, max_cuts, invalid_reason, flag=8, device=None, stream=None):
    N = cuts.shape[0]
    if invalid_reason.shape != (N,):
        raise ValueError("invalid_reason must be [N] for cuts [N,T]")
    wp.launch(disarm_invalid_cut_rows, dim=N, device=device if device is not None else cuts.device, stream=stream,
              inputs=[cuts, int(max_cuts), int(flag), invalid_reason])
