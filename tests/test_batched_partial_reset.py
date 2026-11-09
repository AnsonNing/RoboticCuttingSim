"""ADR 0021 §5 L1 / §6: masked partial reset on the Warp CPU device (bitwise).

Resetting env k at control step s makes it bitwise equal to a fresh run from that point; the other envs stay
bitwise equal to the run without the reset. A full reset equals a freshly packed batch over every array.
The coverage test enumerates vars() of the material, its solver and its diagnostics, so an array that reset
neither restores nor lists as constant / recomputed fails it.
"""
import numpy as np
import pytest
import warp as wp

from batched_fixtures import batch_material, compare_snapshots, env_state, start_pose, step_material
from robotic_cutting_sim.batched import material_batch as MB

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]

CONFIGS = [(512, None, 42), (15, {"young_modulus_pa": 8e4}, 43), (7, None, 44)]
DX, CUTS = (.001, .001, .001, .001, .001), (0, 0, 1, 0, 1)
PRE = 3   # control steps before the reset (seed 15 is first labelled in its third step)


def _poses(starts, offsets):
    out = []
    for start, offset in zip(starts, offsets):
        pose = start.copy()
        pose[0] += offset
        out.append(pose)
    return np.stack(out)


def _all_arrays(material):
    out = {}
    for prefix, owner in (("", material), ("solver.", material.solver), ("diagnostics.", material.diagnostics)):
        for name, value in vars(owner).items():
            if isinstance(value, wp.array):
                out[prefix+name] = value
    return out


def _classified():
    solver = set(MB.SOLVER_RESET_ARRAYS) | set(MB.SOLVER_CONSTANT_ARRAYS) | set(MB.SOLVER_SHARED_ARRAYS)
    material = set(MB.MATERIAL_RESET_ARRAYS) | set(MB.MATERIAL_CONSTANT_ARRAYS) | set(MB.RECOMPUTED_EVERY_CONTROL_STEP)
    return ({f"solver.{n}" for n in solver} | material | {f"diagnostics.{n}" for n in MB.DIAGNOSTICS_SCRATCH})


def _compare_all(a, b, skip=()):
    """Names of arrays (all of vars()) that differ, except recomputed scratch and ``skip``."""
    excluded = set(MB.RECOMPUTED_EVERY_CONTROL_STEP) | {f"diagnostics.{n}" for n in MB.DIAGNOSTICS_SCRATCH}
    arrays_a, arrays_b = _all_arrays(a), _all_arrays(b)
    assert set(arrays_a) == set(arrays_b)
    return [n for n in arrays_a if n not in excluded and n not in skip
            and not np.array_equal(arrays_a[n].numpy(), arrays_b[n].numpy())]


def test_reset_coverage_is_vars_complete():
    material = batch_material(CONFIGS[:1])
    names = set(_all_arrays(material))
    unclassified = names - _classified()
    assert not unclassified, f"arrays neither reset nor declared constant/recomputed: {sorted(unclassified)}"
    stale = _classified() - names
    assert not stale, f"declared arrays that do not exist: {sorted(stale)}"
    model_state = {v.name for v in material.solver.model.per_particle_state}
    assert model_state <= set(MB.SOLVER_RESET_ARRAYS)       # the constitutive registry drives the reset


@pytest.mark.cpu_sim
def test_partial_reset_equals_fresh_and_leaves_others_unchanged():
    starts = [start_pose(seed) for seed, _, _ in CONFIGS]
    run = batch_material(CONFIGS)
    reference = batch_material(CONFIGS)        # never reset
    offsets = np.zeros(3)
    for step in range(PRE):
        offsets += DX[step]
        poses = _poses(starts, offsets)
        step_material(run, poses, [CUTS[step]]*3)
        step_material(reference, poses, [CUTS[step]]*3)
    assert np.count_nonzero(run.snapshot(1)["code"]) > 0, "env 1 was cut before its reset"
    run.reset([1], seeds=[99])
    fresh = batch_material([CONFIGS[0], (15, {"young_modulus_pa": 8e4}, 99), CONFIGS[2]])
    after_reset = env_state(run, 1)
    bad = compare_snapshots(after_reset, env_state(fresh, 1))
    assert bad == ["index"], bad                 # only the shared substep counter (zeroed before every step)
    for e in (0, 2):
        assert compare_snapshots(env_state(run, e), env_state(reference, e)) == []
    # Continue: env 1 restarts from its initial pose; envs 0 and 2 continue unchanged.
    restart = np.zeros(3)
    for step in range(PRE, PRE+2):
        offsets += DX[step]
        restart += DX[step-PRE]
        poses = _poses(starts, offsets)
        restarted = poses.copy()
        restarted[1] = _poses(starts, restart)[1]
        step_material(run, restarted, [CUTS[step], CUTS[step-PRE], CUTS[step]])
        step_material(reference, poses, [CUTS[step]]*3)
        fresh_poses = _poses(starts, restart)
        step_material(fresh, fresh_poses, [CUTS[step-PRE]]*3)
        for e in (0, 2):
            assert compare_snapshots(env_state(run, e), env_state(reference, e)) == [], f"step {step} env {e}"
        assert compare_snapshots(env_state(run, 1), env_state(fresh, 1)) == [], f"step {step} env 1"
    assert int(run.seed.numpy()[1]) == 99 and int(run.step_count.numpy()[1]) == 2
    assert int(run.step_count.numpy()[0]) == PRE+2


@pytest.mark.cpu_sim
def test_full_reset_equals_freshly_packed():
    starts = [start_pose(seed) for seed, _, _ in CONFIGS[:2]]
    run = batch_material(CONFIGS[:2])
    offsets = np.zeros(2)
    for step in range(2):
        offsets += DX[step]
        step_material(run, _poses(starts, offsets), [CUTS[step]]*2)
    assert np.count_nonzero(run.solver.code.numpy()) > 0
    fresh = batch_material(CONFIGS[:2])
    assert _compare_all(run, fresh) != []
    run.reset()
    assert _compare_all(run, fresh) == []
    # A batch that steps after the full reset tracks a fresh batch exactly.
    offsets = np.zeros(2)
    offsets += DX[0]
    step_material(run, _poses(starts, offsets), [0, 0])
    step_material(fresh, _poses(starts, offsets), [0, 0])
    assert _compare_all(run, fresh) == []


def test_reset_argument_checks():
    material = batch_material(CONFIGS[:1])
    with pytest.raises(ValueError):
        material.reset([1])
    with pytest.raises(ValueError):
        material.reset([0], seeds=[1, 2])
    key = material.packed.scene_keys[0]
    material.reset([0], scene_keys=[key])             # same geometry: allowed
    material.reset([])                                # no-op
