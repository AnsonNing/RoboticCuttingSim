"""ADR 0021 §5 L1 isolation and §7 coupling on the Warp CPU device (bitwise).

Perturbing one env's inputs (or poisoning its state with NaN) leaves every other env bitwise unchanged, and
the batched coupling (wrench handoff, segmented diagnostics, step counts) equals each env's frozen
``LiveCuttingMaterial`` exactly.
"""
import numpy as np
import pytest

from batched_fixtures import (advance_single, batch_material, compare_env, compare_snapshots, env_state,
                              start_pose, step_material, trim_material)
from robotic_cutting_sim.batched.material_batch import INVALID_NONFINITE_STATE, InvalidBatchEnvs

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"), pytest.mark.cpu_sim]

CONFIGS = [(512, None, 42), (15, {"young_modulus_pa": 8e4}, 42), (7, None, 42)]
BASE = ((.001, 0), (.001, 0), (.001, 1))


def _poses(step_dx, starts, step):
    """Pose of every env after ``step`` control steps (cumulative x offsets)."""
    out = []
    for e, start in enumerate(starts):
        pose = start.copy()
        pose[0] += sum(step_dx[e][: step+1])
        out.append(pose)
    return np.stack(out)


def test_coupling_and_state_equal_frozen_singles_and_perturbation_is_isolated():
    seeds = [c[0] for c in CONFIGS]
    starts = [start_pose(s) for s in seeds]
    singles = []
    for seed, material, reset_seed in CONFIGS:
        single = trim_material(seed, material=material)
        single.reset(start_pose(seed), seed=reset_seed)
        singles.append(single)
    baseline = batch_material(CONFIGS)
    perturbed = batch_material(CONFIGS)
    dx = [[d for d, _ in BASE] for _ in CONFIGS]
    cuts = [c for _, c in BASE]
    dx_perturbed = [row[:] for row in dx]
    dx_perturbed[1] = [.0015, -.0005, .002]              # env 1 only: other motion ...
    for step in range(len(BASE)):
        poses = _poses(dx, starts, step)
        poses_perturbed = _poses(dx_perturbed, starts, step)
        cut = np.array([cuts[step]]*3, np.int32)
        cut_perturbed = cut.copy()
        cut_perturbed[1] = 1-cuts[step]                    # ... and the other digit
        diagnostics = [advance_single(s, p, step, int(c)) for s, p, c in zip(singles, poses, cut)]
        step_material(baseline, poses, cut)
        step_material(perturbed, poses_perturbed, cut_perturbed)
        reductions = baseline.diagnostics_read()
        for e, single in enumerate(singles):
            assert compare_env(baseline.solver, e, single.bed) == [], f"step {step} env {e}"
            # 20 ms delayed wrench handoff (identity workpiece rotation: exact float64 copies)
            np.testing.assert_array_equal(baseline.pending_wrench.numpy()[e], single.pending_wrench)
            np.testing.assert_array_equal(baseline.applied_previous.numpy()[e],
                                          diagnostics[e]["applied_previous_material_wrench_world"])
            np.testing.assert_array_equal(baseline.last_pose.numpy()[e], single.last_pose)
            assert baseline.step_count.numpy()[e] == step+1
            frozen = single.numerical_diagnostics("device_reduction_v1")
            assert reductions[e]["nonfinite"] is False
            for key, value in frozen.items():
                assert reductions[e][key] == value, (step, e, key)
        for e in (0, 2):
            assert compare_snapshots(env_state(baseline, e), env_state(perturbed, e)) == [], f"step {step} env {e}"
        assert compare_snapshots(env_state(baseline, 1), env_state(perturbed, 1)) != []
    assert np.count_nonzero(baseline.pending_wrench.numpy()) > 0
    assert baseline.invalid_reason.numpy().tolist() == [0, 0, 0]


def test_nonfinite_env_is_flagged_and_isolated():
    configs = [(512, None, 42), (15, None, 42)]
    clean = batch_material(configs, strict=False)
    poisoned = batch_material(configs, strict=False)
    v = poisoned.solver.v.numpy()
    first = int(poisoned.packed.particle_offset[1])
    v[first+10] = np.nan                                  # env 1 only
    poisoned.solver.v.assign(v)
    starts = [start_pose(512), start_pose(15)]
    poses = np.stack([p+np.r_[.001, 0, 0, 0, 0, 0, 0] for p in starts])
    step_material(clean, poses, [0, 0])
    step_material(poisoned, poses, [0, 0])
    reasons = poisoned.invalid_reason.numpy()
    assert reasons[0] == 0 and reasons[1] & INVALID_NONFINITE_STATE   # NaN also reaches env 1's wrench
    assert int(poisoned.invalid_count.numpy()[0]) == 1
    assert compare_snapshots(env_state(clean, 0), env_state(poisoned, 0)) == []
    strict = batch_material(configs)
    v = strict.solver.v.numpy()
    v[first+10] = np.inf
    strict.solver.v.assign(v)
    with pytest.raises(InvalidBatchEnvs) as info:
        step_material(strict, poses, [0, 0])
    assert info.value.env_ids == [1] and "NONFINITE_STATE" in info.value.reasons[1]
