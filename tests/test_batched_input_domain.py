"""P1-1b B2 follow-up: the material's rigid-tool input domain (``material_input_domain_v1``), Warp CPU device.

An out-of-domain edge pose or pad box of one env (a lost grasp swinging or ejecting the tool, a teleport, an
oversized box) sets that env's ``INPUT_OUT_OF_DOMAIN`` bit and HOLDS its substep inputs on the device before the
substeps run (blade at the last in-domain substep pose, zero twist, cut disarmed, pads at the last in-domain
boxes) until its reset; no host sync, no recapture. The other envs stay bitwise equal to a batch that never saw
the bad input. Without the guard the same 1e7 m edge jump flung env 1's particles to ~1e7 m and the step time
grew without bound (``scripts/rl_ready/probe_material_ood.py --scenario huge``; P1-1b B2 report).
"""
import numpy as np
import pytest

from batched_fixtures import compare_snapshots, env_state, step_material, yawed_start_pose, yawed_trim_material
from robotic_cutting_sim.batched import layout
from robotic_cutting_sim.batched import material_batch as MB

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"), pytest.mark.cpu_sim]

BOXES = [[[.004, 0., .004], [.003, .003, .003]], [[0., .01, .004], [.004, .003, .004]],
         [[0., -.01, .004], [.004, .003, .004]]]          # edge frame: handle, left pad, right pad
SEEDS = (512, 15)


def _batch(strict=False):
    specs = []
    for seed in SEEDS:
        m = yawed_trim_material(seed, 0., BOXES)
        specs.append(layout.EnvSpec(m, yawed_start_pose(m, seed), 42))
    return MB.BatchedCuttingMaterial(specs, device="cpu", parity_mode=True, strict=strict)


def _poses(material, step):
    poses = material.initial_measured_pose.numpy().copy()
    poses[:, 0] += .001*(step+1)                           # 1 mm per control step: 0.05 m/s, in domain
    return poses


def _held(material, e):
    s, T = material.solver, material.solver.trajectory_length
    poses = s.poses.numpy().reshape(material.num_envs, T, 7)[e]
    twists = s.twists.numpy().reshape(material.num_envs, T, 6)[e]
    cuts = s.cuts.numpy().reshape(material.num_envs, T)[e]
    hold = material.hold_pose.numpy()[e]
    return bool(np.all(poses == hold) and np.all(twists == 0) and np.all(cuts == -1))


def _particles(material, e):
    off = material.packed.particle_offset
    return material.solver.x.numpy()[int(off[e]):int(off[e+1])]


@pytest.mark.parametrize("scenario", ["teleport_1e7_m", "swing_5_m_s"])
def test_out_of_domain_edge_is_flagged_held_and_isolated(scenario):
    clean, bad = _batch(), _batch()
    for step in range(3):
        poses, cut = _poses(clean, step), [0, 0]
        step_material(clean, poses, cut)
        poses_bad = poses.copy()
        if step == 1:
            if scenario == "teleport_1e7_m":
                poses_bad[1, :3] = 1e7
            else:
                poses_bad[1, 0] += .1                          # 0.1 m in 20 ms = 5 m/s > 3.4 m/s
        step_material(bad, poses_bad, cut)                     # step 2: back in range, still held (latched)
        assert compare_snapshots(env_state(clean, 0), env_state(bad, 0)) == [], f"step {step} env 0"
        reasons = bad.invalid_reason.numpy()
        if step == 0:
            assert reasons.tolist() == [0, 0]
            assert compare_snapshots(env_state(clean, 1), env_state(bad, 1)) == []
        else:
            assert reasons[0] == 0 and reasons[1] & MB.INVALID_INPUT_DOMAIN, (step, reasons)
            assert _held(bad, 1)
            x = _particles(bad, 1)
            assert np.isfinite(x).all() and np.abs(x).max() < .2   # nothing flung (pre-guard: ~1e7 m)
    assert clean.invalid_reason.numpy().tolist() == [0, 0]
    assert MB.decode_reason(bad.invalid_reason.numpy()[1]) == ["INPUT_OUT_OF_DOMAIN"]
    # masked reset clears the bit and the hold; the env then follows its inputs again
    bad.reset_masked(np.array([0, 1]))
    assert bad.invalid_reason.numpy().tolist() == [0, 0]
    assert np.array_equal(bad.hold_pose.numpy()[1], bad.initial_local_pose.numpy()[1])
    step_material(bad, _poses(bad, 0), [0, 0])
    assert bad.invalid_reason.numpy().tolist() == [0, 0] and not _held(bad, 1)


def test_out_of_domain_pad_box_is_held_and_strict_raises():
    clean, bad = _batch(), _batch()
    good = np.stack([np.asarray(BOXES, np.float32)]*2)
    oversized = good.copy()
    oversized[1, 1, 1] = [.5, .5, .5]                       # finite, positive: accepted by the setter, out of domain
    for step in range(2):
        poses = _poses(clean, step)
        for m, boxes in ((clean, good), (bad, oversized if step == 1 else good)):
            m.set_obstacle_boxes(boxes)
            step_material(m, poses, [0, 0])
        assert compare_snapshots(env_state(clean, 0), env_state(bad, 0)) == []
    assert bad.invalid_reason.numpy().tolist() == [0, MB.INVALID_INPUT_DOMAIN]
    halves = bad.solver.box_halves.numpy().reshape(2, 3, 3)
    assert np.array_equal(halves[1], good[1, :, 1])         # the last in-domain boxes, not the oversized ones
    assert _held(bad, 1)
    strict = _batch(strict=True)
    step_material(strict, _poses(strict, 0), [0, 0])
    poses = _poses(strict, 1)
    poses[1, :3] = 1e7
    with pytest.raises(MB.InvalidBatchEnvs) as info:
        step_material(strict, poses, [0, 0])
    assert info.value.env_ids == [1] and info.value.reasons[1] == ["INPUT_OUT_OF_DOMAIN"]


def test_receipt_records_the_input_domain():
    receipt = _batch().receipt()["input_domain"]
    assert receipt["version"] == MB.INPUT_DOMAIN_VERSION == "material_input_domain_v1"
    assert {k: receipt[k] for k in MB.INPUT_DOMAIN} == MB.INPUT_DOMAIN
