"""ADR 0021 §7.1: the device pose interpolation equals the frozen SciPy path to <= 1 float32 ulp.

Reference per env: ``interpolate_measured_poses(transform_poses(T_e, last), transform_poses(T_e, measured),
substeps, control_dt)`` -- the call ``live_material.advance`` makes. Runs on the Warp CPU device.
"""
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from scipy.spatial.transform import Rotation  # noqa: E402

from robotic_cutting_sim.batched import pose_interp as PI  # noqa: E402
from robotic_cutting_sim.coupled_robot import interpolate_measured_poses  # noqa: E402
from robotic_cutting_sim.scene_frames import invert_transform, pose_matrix, transform_poses  # noqa: E402

DEVICE = "cpu"
SUBSTEPS = 200
DT = .02


def _ordered(a):
    i = np.ascontiguousarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulp_distance(a, b):
    return np.abs(_ordered(a)-_ordered(b))


def _random_transforms(rng, n):
    out = []
    for _ in range(n):
        T = pose_matrix(np.r_[rng.uniform(-.5, .5, 3), Rotation.random(random_state=rng).as_quat()])
        out.append(invert_transform(T))
    return np.stack(out)


def _reference(last, measured, T_local_world):
    poses, twists = [], []
    for a, b, T in zip(last, measured, T_local_world):
        p, t = interpolate_measured_poses(transform_poses(T, a), transform_poses(T, b), SUBSTEPS, DT)
        poses.append(p)
        twists.append(t)
    return np.stack(poses), np.stack(twists)


def _device(last, measured, T_local_world):
    N, T = len(last), SUBSTEPS+1
    Tw, qw = PI.local_frames(T_local_world, device=DEVICE)
    poses = wp.zeros((N, T), dtype=wp.transform, device=DEVICE)
    twists = wp.zeros((N, T), dtype=wp.spatial_vector, device=DEVICE)
    status = wp.full(N, -1, dtype=wp.int32, device=DEVICE)
    PI.launch_pose_interp(wp.array(np.asarray(last, np.float64), dtype=wp.float64, device=DEVICE),
                          wp.array(np.asarray(measured, np.float64), dtype=wp.float64, device=DEVICE),
                          Tw, qw, poses, twists, status, SUBSTEPS, DT)
    wp.synchronize_device(DEVICE)
    return poses.numpy().reshape(N, T, 7), twists.numpy().reshape(N, T, 6), status.numpy()


def _check(last, measured, T_local_world, label):
    ref_p, ref_t = _reference(last, measured, T_local_world)
    dev_p, dev_t, status = _device(last, measured, T_local_world)
    assert (status == PI.STATUS_OK).all(), label
    up, ut = ulp_distance(dev_p, ref_p), ulp_distance(dev_t, ref_t)
    worst = {"pose_max_ulp": int(up.max()), "twist_max_ulp": int(ut.max()),
             "pose_components_off_by_1ulp": int((up == 1).sum()), "twist_components_off_by_1ulp": int((ut == 1).sum()),
             "components": int(up.size+ut.size)}
    assert up.max() <= 1 and ut.max() <= 1, (label, worst, np.argwhere(up > 1)[:5], np.argwhere(ut > 1)[:5])
    return worst


def _random_unit_quats(rng, n):
    return Rotation.random(n, random_state=rng).as_quat()


def test_random_poses_heterogeneous_transforms():
    rng = np.random.default_rng(20261110)
    N = 32
    last = np.c_[rng.uniform(-.2, .2, (N, 3)), _random_unit_quats(rng, N)]
    # Realistic per-step motion (<= 2 mm, <= 2 deg) and arbitrary jumps.
    small = Rotation.from_rotvec(rng.normal(size=(N, 3))*np.deg2rad(1.))
    measured = np.c_[last[:, :3]+rng.normal(scale=1e-3, size=(N, 3)),
                     (small*Rotation.from_quat(last[:, 3:])).as_quat()]
    measured[N//2:, 3:] = _random_unit_quats(rng, N-N//2)
    _check(last, measured, _random_transforms(rng, N), "random")


def test_sign_flipped_quaternions():
    rng = np.random.default_rng(7)
    N = 16
    q = _random_unit_quats(rng, N)
    last = np.c_[rng.uniform(-.1, .1, (N, 3)), q]
    measured = last.copy()
    measured[:, 3:] = -(Rotation.from_rotvec(rng.normal(size=(N, 3))*1e-2)*Rotation.from_quat(q)).as_quat()
    measured[:4, 3:] = -q[:4]                     # identical rotation, opposite sign
    last[8:12, 3:] *= -1                          # negative-w previous poses
    _check(last, measured, _random_transforms(rng, N), "sign flip")


@pytest.mark.parametrize("angle", [0., 1e-12, 1e-9, 1e-6, 5e-4, 1e-3*(1-1e-12), 1e-3, 1e-3*(1+1e-12), 2e-3, 1e-2])
def test_near_zero_rotations_and_taylor_branch_boundary(angle):
    rng = np.random.default_rng(11)
    N = 8
    q = _random_unit_quats(rng, N)
    axes = rng.normal(size=(N, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    last = np.c_[rng.uniform(-.1, .1, (N, 3)), q]
    measured = np.c_[last[:, :3]+rng.normal(scale=1e-4, size=(N, 3)),
                     (Rotation.from_rotvec(axes*angle)*Rotation.from_quat(q)).as_quat()]
    _check(last, measured, _random_transforms(rng, N), f"angle {angle}")


@pytest.mark.parametrize("angle", [np.pi/2, 3., np.pi-1e-6, np.pi])
def test_large_rotations(angle):
    rng = np.random.default_rng(13)
    N = 8
    q = _random_unit_quats(rng, N)
    axes = rng.normal(size=(N, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    last = np.c_[rng.uniform(-.1, .1, (N, 3)), q]
    measured = np.c_[rng.uniform(-.1, .1, (N, 3)), (Rotation.from_rotvec(axes*angle)*Rotation.from_quat(q)).as_quat()]
    _check(last, measured, _random_transforms(rng, N), f"angle {angle}")


def test_identity_and_trim_workpiece_transform_and_simple_axes():
    # TRIM: world_from_workpiece translation (0, 0, 0.01), yaw 0 (seeded_trim.generate_scene).
    T_trim = invert_transform(pose_matrix([0., 0., .01, 0., 0., 0., 1.]))
    last = np.array([[-.03, 0., .06, 1., 0., 0., 0.], [0., 0., .03, 0., 0., 0., 1.],
                     [.01, .02, .03, 0., 0., np.sqrt(.5), np.sqrt(.5)], [0., 0., 0., 0., 1., 0., 0.]])
    measured = last.copy()
    measured[:, 0] += .001
    measured[1, 3:] = Rotation.from_rotvec([0., 0., np.deg2rad(2.)]).as_quat()
    measured[3, 3:] = [0., 0., 1., 0.]
    for T in (np.eye(4), T_trim):
        _check(last, measured, np.repeat(T[None], len(last), 0), "axis-aligned")
    # A held tool (identical poses) gives a constant tile and zero twist, like live_material's initial tile.
    _, twists, _ = _device(last, last, np.repeat(T_trim[None], len(last), 0))
    assert not np.any(twists)


def test_layout_matches_frozen_set_trajectory_arrays():
    """The [N,T] transform/spatial_vector rows are the same bytes the frozen solver builds from (T,7)/(T,6)."""
    rng = np.random.default_rng(3)
    last = np.c_[rng.uniform(-.1, .1, (2, 3)), _random_unit_quats(rng, 2)]
    measured = np.c_[rng.uniform(-.1, .1, (2, 3)), _random_unit_quats(rng, 2)]
    T = _random_transforms(rng, 2)
    ref_p, ref_t = _reference(last, measured, T)
    dev_p, dev_t, _ = _device(last, measured, T)
    for e in range(2):
        frozen_p = wp.array(ref_p[e], dtype=wp.transform, device=DEVICE).numpy()
        frozen_t = wp.array(ref_t[e], dtype=wp.spatial_vector, device=DEVICE).numpy()
        assert frozen_p.shape == (SUBSTEPS+1, 7) and frozen_t.shape == (SUBSTEPS+1, 6)
        assert ulp_distance(frozen_p, dev_p[e]).max() <= 1 and ulp_distance(frozen_t, dev_t[e]).max() <= 1


def test_status_codes_for_inputs_the_frozen_path_rejects():
    last = np.tile([0., 0., .03, 0., 0., 0., 1.], (4, 1))
    measured = last.copy()
    measured[1, 0] = np.nan
    measured[2, 3:] = [0., 0., 0., 1.+1e-5]
    measured[3, 3:] = [0., 0., 0., 1.+5e-7]   # inside validate_pose_xyzw's 1e-6 tolerance
    T = np.repeat(np.eye(4)[None], 4, 0)
    _, _, status = _device(last, measured, T)
    assert status.tolist() == [PI.STATUS_OK, PI.STATUS_NONFINITE, PI.STATUS_NON_UNIT_QUATERNION, PI.STATUS_OK]
    with pytest.raises(ValueError):
        transform_poses(np.eye(4), measured[2])
    _check(last[[3]], measured[[3]], T[:1], "near-unit input")


def test_fill_cuts_and_shape_guards():
    cut = wp.array(np.array([-1, 0, 1], np.int32), dtype=wp.int32, device=DEVICE)
    cuts = wp.zeros((3, SUBSTEPS+1), dtype=wp.int32, device=DEVICE)
    PI.launch_fill_cuts(cut, cuts)
    np.testing.assert_array_equal(cuts.numpy(), np.repeat(np.array([-1, 0, 1], np.int32)[:, None], SUBSTEPS+1, 1))
    Tw, qw = PI.local_frames(np.repeat(np.eye(4)[None], 2, 0), device=DEVICE)
    bad = wp.zeros((2, SUBSTEPS), dtype=wp.transform, device=DEVICE)
    with pytest.raises(ValueError):
        PI.launch_pose_interp(wp.zeros((2, 7), dtype=wp.float64, device=DEVICE),
                              wp.zeros((2, 7), dtype=wp.float64, device=DEVICE), Tw, qw, bad,
                              wp.zeros((2, SUBSTEPS+1), dtype=wp.spatial_vector, device=DEVICE),
                              wp.zeros(2, dtype=wp.int32, device=DEVICE), SUBSTEPS, DT)
    with pytest.raises(ValueError):
        PI.local_frames(np.diag([1., 1., -1., 1.])[None])   # reflection is not a rigid transform


def test_host_float64_operation_orders_the_kernel_reproduces():
    """The kernel mirrors this host's NumPy/OpenBLAS orders (np.linalg.norm = sequential FMA chain; the
    (1,3)@(3,3).T matmul in transform_points = fma(p2,R2, fma(p0,R0, p1*R1))). If a different BLAS kernel
    is selected on another machine this test fails first, explaining any float32 ulp difference above."""
    from fractions import Fraction

    def fma(a, b, c):
        return float(Fraction(a)*Fraction(b)+Fraction(c))
    rng = np.random.default_rng(5)
    for _ in range(300):
        x = rng.normal(size=4)
        assert np.linalg.norm(x) == np.sqrt(fma(x[3], x[3], fma(x[2], x[2], fma(x[1], x[1], x[0]*x[0]))))
        p, m = rng.normal(size=(1, 3)), rng.normal(size=(3, 3))
        expected = [fma(p[0, 2], m[j, 2], fma(p[0, 0], m[j, 0], p[0, 1]*m[j, 1])) for j in range(3)]
        assert (p@m.T)[0].tolist() == expected
