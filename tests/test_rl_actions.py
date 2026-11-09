"""edge_delta_v1 Torch port vs the frozen NumPy path (control.normalized_delta + resolve_delta(frame="edge")
+ bound_target, PolicyReadyEnv status rule). Torch test: run in .deps/isaacsim-venv (plan D-09); in .venv it is
skipped and therefore NOT_RUN, never a pass."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.spatial.transform import Rotation  # noqa: E402

from robotic_cutting_sim.control import ActionLimits, normalized_delta, resolve_delta  # noqa: E402
from robotic_cutting_sim.policy_env import ACTION_STATUSES  # noqa: E402
from robotic_cutting_sim.rl import actions as A  # noqa: E402
from robotic_cutting_sim.scene_frames import pose_matrix  # noqa: E402

F32_TOL = 2.**-23   # float32 relative epsilon; gate is |diff| <= F32_TOL*max(1,|x|)
TRIM_LIMITS = ActionLimits(max_translation_m=.001, max_rotation_rad=float(np.deg2rad(1.)))


def _transforms(rng, n):
    out = [pose_matrix([0., 0., .01, 0., 0., 0., 1.])]   # seeded_trim world_from_workpiece
    for _ in range(n-1):
        yaw = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_quat()
        out.append(pose_matrix(np.r_[rng.uniform(-.3, .3, 2), rng.uniform(0., .05), yaw]))
    return np.stack(out)


def _frozen(actions, ref, limits, T):
    """Single-env frozen path per env; returns pose, status name, report (None when rejected)."""
    a = np.asarray(actions, float)
    if not np.isfinite(a[:6]).all():
        return ref.copy(), "REJECTED_INVALID_TARGET", None
    if not np.isfinite(a[6]):
        return ref.copy(), "REJECTED_INVALID_SCHEMA", None
    if not np.isfinite(a[7]):
        return ref.copy(), "REJECTED_CUT_INDEX_OUT_OF_RANGE", None
    pose, report = resolve_delta(ref, normalized_delta(a[:6], limits), limits, frame="edge", T_world_workpiece=T)
    clamped = any(report[k] for k in ("translation_clamped", "rotation_clamped", "workspace_clamped"))
    return pose, "CLAMPED" if clamped else "ACCEPTED", report


def _random_refs(rng, T, n):
    local = np.c_[rng.uniform(-.15, .15, n), rng.uniform(-.13, .13, n), rng.uniform(-.015, .165, n)]
    world = np.einsum("nij,nj->ni", T[:, :3, :3], local)+T[:, :3, 3]
    return np.c_[world, Rotation.random(n, random_state=rng).as_quat()]


def _assert_close(x, ref, label):
    diff = np.abs(np.asarray(x)-np.asarray(ref))
    bound = F32_TOL*np.maximum(1., np.abs(ref))
    assert (diff <= bound).all(), (label, float(diff.max()))
    return float(diff.max())


@pytest.mark.parametrize("limits", [ActionLimits(), TRIM_LIMITS], ids=["default", "trim"])
def test_random_actions_match_frozen_numpy(limits):
    rng = np.random.default_rng(20261110)
    N, rounds = 64, 40
    T = _transforms(rng, N)
    spec = A.EdgeDeltaActionV1(limits, T, 2)
    worst, statuses, flags = 0., set(), {"translation": 0, "rotation": 0, "workspace": 0, "out_of_box": 0}
    for _ in range(rounds):
        ref = _random_refs(rng, T, N)
        act = rng.uniform(-1.6, 1.6, (N, 8))
        act[rng.random(N) < .3, :6] *= .4          # many inside the box
        res = spec.apply(torch.as_tensor(act, dtype=torch.float32), torch.as_tensor(ref))
        act32 = act.astype(np.float32).astype(float)  # the frozen path sees the same float32 action values
        names = A.status_names(res.status)
        for e in range(N):
            pose, status, report = _frozen(act32[e], ref[e], limits, T[e])
            assert names[e] == status, (e, names[e], status)
            statuses.add(status)
            worst = max(worst, _assert_close(res.commanded_pose[e].numpy(), pose, "pose"))
            assert bool(res.translation_clamped[e]) == report["translation_clamped"]
            assert bool(res.rotation_clamped[e]) == report["rotation_clamped"]
            assert bool(res.workspace_clamped[e]) == report["workspace_clamped"]
            for key in ("requested_translation_m", "requested_rotation_rad", "applied_translation_m",
                        "applied_rotation_rad"):
                _assert_close(getattr(res, key)[e].item(), report[key], key)
        flags["translation"] += int(res.translation_clamped.sum())
        flags["rotation"] += int(res.rotation_clamped.sum())
        flags["workspace"] += int(res.workspace_clamped.sum())
        flags["out_of_box"] += int(res.out_of_box.sum())
    assert statuses == {"ACCEPTED", "CLAMPED"}
    assert all(flags.values()), flags
    assert worst < 1e-12, worst   # observed float64-level agreement (gate above is the float32 tolerance)


def test_rollout_accumulation_matches_frozen():
    """300 consecutive steps: each path accumulates its own commanded pose (like inner._commanded)."""
    rng = np.random.default_rng(3)
    N = 8
    T = _transforms(rng, N)
    spec = A.EdgeDeltaActionV1(TRIM_LIMITS, T, 2)
    ref = _random_refs(rng, T, N)
    dev = torch.as_tensor(ref)
    worst = 0.
    for step in range(300):
        act = rng.uniform(-1.2, 1.2, (N, 8)).astype(np.float32)
        dev = spec.apply(torch.as_tensor(act), dev).commanded_pose
        ref = np.stack([_frozen(act[e].astype(float), ref[e], TRIM_LIMITS, T[e])[0] for e in range(N)])
        worst = max(worst, _assert_close(dev.numpy(), ref, f"step {step}"))
    assert worst < 1e-12, worst


def test_nonfinite_actions_are_rejected_with_frozen_precedence_and_hold():
    rng = np.random.default_rng(1)
    N = 7
    T = _transforms(rng, N)
    spec = A.EdgeDeltaActionV1(ActionLimits(), T, 2)
    ref = _random_refs(rng, T, N)
    act = np.full((N, 8), .5, np.float32)
    act[0, 2] = np.nan                       # target
    act[1, 5] = np.inf; act[1, 6] = np.nan   # target wins over cut_enabled
    act[2, 6] = np.nan                       # cut_enabled
    act[3, 6] = -np.inf; act[3, 7] = np.nan  # cut_enabled wins over cut_index
    act[4, 7] = np.inf                       # cut_index
    act[5, 0] = 7.                           # finite, out of box: clipped, not rejected
    res = spec.apply(torch.as_tensor(act), torch.as_tensor(ref))
    assert A.status_names(res.status) == ["REJECTED_INVALID_TARGET", "REJECTED_INVALID_TARGET",
                                          "REJECTED_INVALID_SCHEMA", "REJECTED_INVALID_SCHEMA",
                                          "REJECTED_CUT_INDEX_OUT_OF_RANGE", "CLAMPED", "ACCEPTED"]
    for e in range(5):
        np.testing.assert_array_equal(res.commanded_pose[e].numpy(), ref[e])   # HOLD: unchanged, bitwise
        assert int(res.cut_digit[e]) == -1 and not bool(res.armed[e])
        np.testing.assert_array_equal(res.executed_action[e].numpy(), [0, 0, 0, 0, 0, 0, -1, 0])
        assert torch.isfinite(res.executed_action[e]).all()
    assert bool(res.out_of_box[5]) and not bool(res.out_of_box[6])
    # the frozen path itself refuses these inputs
    with pytest.raises(ValueError):
        normalized_delta(act[0, :6], ActionLimits())
    _assert_close(res.commanded_pose[5].numpy(), _frozen(act[5].astype(float), ref[5], ActionLimits(), T[5])[0], "clip")


@pytest.mark.parametrize("K,expected", [(1, [0, 0, 0, 0, 0]), (2, [0, 0, 1, 1, 1]), (3, [0, 1, 1, 2, 2])])
def test_cut_digit_selector_and_arm_flag(K, expected):
    T = np.repeat(pose_matrix([0., 0., .01, 0., 0., 0., 1.])[None], 5, 0)
    spec = A.EdgeDeltaActionV1(ActionLimits(), T, K)
    sel = torch.tensor([-1., -.01, 0., .5, 3.], dtype=torch.float64)
    assert spec.cut_digit_from_selector(sel).tolist() == expected
    ref = torch.as_tensor(np.tile([0., 0., .04, 0., 0., 0., 1.], (5, 1)))
    act = torch.zeros(5, 8)
    act[:, 7] = sel.float()
    act[:, 6] = torch.tensor([1., .1, 0., -0., -1.])
    res = spec.apply(act, ref)
    assert res.cut_digit.tolist() == [expected[0], expected[1], -1, -1, -1]
    assert res.cut_digit.dtype == torch.int32 and res.status.dtype == torch.int32


def test_status_codes_are_policy_env_statuses_and_guards():
    assert A.ACTION_STATUSES is ACTION_STATUSES
    assert A.status_names(torch.arange(len(ACTION_STATUSES))) == list(ACTION_STATUSES)
    T = np.repeat(np.eye(4)[None], 2, 0)
    spec = A.EdgeDeltaActionV1(ActionLimits(), T, 2)
    assert spec.spec()["dim"] == 8 and spec.spec()["version"] == "edge_delta_v1"
    ref = torch.as_tensor(np.tile([0., 0., .04, 0., 0., 0., 1.], (2, 1)))
    with pytest.raises(ValueError):
        spec.apply(torch.zeros(2, 6), ref)
    with pytest.raises(ValueError):
        spec.apply(torch.zeros(2, 8), ref.float())
    with pytest.raises(ValueError):
        A.EdgeDeltaActionV1(ActionLimits(), T, 16)
    with pytest.raises(TypeError):
        A.EdgeDeltaActionV1({"max_translation_m": .002}, T, 2)
    bad = ref.clone()
    bad[0, 6] = 1.1
    assert spec.apply(torch.zeros(2, 8), bad).reference_invalid.tolist() == [True, False]
