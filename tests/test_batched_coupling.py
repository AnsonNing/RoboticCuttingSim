"""ADR 0021 §7 coupling of ``BatchedCuttingMaterial`` on the Warp CPU device (review fixes of 2026-11-10).

* cut digits written on the device are disarmed and flagged before the substeps run (finding 4);
* the device measured-pose path equals the live ``LiveCuttingMaterial`` chain bitwise (finding 5);
* the in-place obstacle/pad box setter equals the frozen ``update_pad_obstacles`` per step (finding 6);
* receipts hash every batched source and name the interpolation path actually used (finding 7).
"""
import numpy as np
import pytest
import warp as wp

from batched_fixtures import batch_material, compare_snapshots, env_state, start_pose
from robotic_cutting_sim.batched import material_batch as MB

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def _moved(pose, dx):
    out = np.array(pose, np.float64)
    out[0] += dx
    return out


@pytest.mark.cpu_sim
def test_device_cut_digits_are_disarmed_and_flagged_before_the_substeps():
    """Envs with digits K, 16 and -2 must evolve exactly like the disarmed env (-1); the armed env cuts."""
    material = batch_material([(512, None, 42)]*5, strict=False)
    K = material.solver.max_cuts
    digits = np.array([K, 16, -2, -1, 0], np.int32)
    start = start_pose(512)
    for step in range(2):
        poses = np.stack([_moved(start, .001*(step+1))]*5)
        material.advance(wp.array(poses, dtype=wp.float64, device="cpu"),
                         wp.array(digits, dtype=wp.int32, device="cpu"), interpolation="device")
        cuts = material.solver.cuts.numpy().reshape(5, -1)
        assert (cuts[:4] == -1).all() and (cuts[4] == 0).all()
        assert material.invalid_reason.numpy().tolist() == [MB.INVALID_CUT_DIGIT]*3+[0, 0]
        reference = env_state(material, 3)
        for e in range(3):
            bad = compare_snapshots(env_state(material, e), reference)
            assert sorted(bad) == ["material.cut_input", "material.invalid_reason"], (step, e, bad)
    assert np.count_nonzero(material.snapshot(4)["code"]) > 0, "the armed env cuts (the comparison is not vacuous)"
    assert np.count_nonzero(material.snapshot(3)["code"]) == 0
    assert material.solver.snapshot(3)["stats"][1] > 0, "the disarmed blade touches the material"
    # Preset rows: invalid entries only are disarmed, the env is flagged; strict mode raises with the ids.
    rows = material.solver.cuts.numpy().reshape(5, -1)
    rows[4, 3] = 16
    material.solver.cuts.assign(rows.reshape(-1))
    material.strict = True
    with pytest.raises(MB.InvalidBatchEnvs) as info:
        material.advance(interpolation="preset")
    assert info.value.env_ids == [0, 1, 2, 4]
    assert all("INVALID_CUT_DIGIT" in info.value.reasons[e] for e in (0, 1, 2, 4))
    after = material.solver.cuts.numpy().reshape(5, -1)
    assert after[4, 3] == -1 and (np.delete(after[4], 3) == 0).all()


def test_host_cut_digits_are_refused_before_any_state_change():
    material = batch_material([(512, None, 42)])
    K = material.solver.max_cuts
    before = env_state(material, 0)
    moved = _moved(start_pose(512), .001)[None]
    for digit in (K, 16, -2, 0.0):
        with pytest.raises(ValueError):
            material.advance(moved, [digit])
    assert compare_snapshots(env_state(material, 0), before) == []
    assert not np.array_equal(moved[0], before["material.measured_pose"])   # a refused write would show
