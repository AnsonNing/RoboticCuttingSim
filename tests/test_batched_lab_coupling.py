"""``BatchedCuttingMaterial(backend="isaac_lab_newton")`` through the ``MaterialCoupling`` surface that
``rl/lab_batched_env.py`` drives (ADR 0022 §2.5, §5.2, §7): in-place box writes, device ``set_measured_poses``,
``advance()``, in-place ``pending_wrench``/``pad_wrench`` reads and ``reset_masked(mask)``. The backend changes
validation and labels only: driven with the same synthetic measured-edge and pad sequence, it equals the
kinematic backend bitwise over every array (Warp CPU device)."""
import numpy as np
import pytest
import warp as wp

from batched_fixtures import compare_snapshots, env_state, yawed_start_pose, yawed_trim_material
from robotic_cutting_sim.batched import layout
from robotic_cutting_sim.batched import material_batch as MB
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]

BOXES = [[[-.004, 0., .006], [.003, .003, .003]], [[.002, .008, 0.], [.004, .003, .004]],
         [[.002, -.008, 0.], [.004, .003, .004]]]
ROWS = ((512, 1.5), (512, -2.))


@wp.kernel
def _write_boxes(boxes: wp.array4d(dtype=wp.float32), nboxes: int, centers: wp.array(dtype=wp.vec3),
                 halves: wp.array(dtype=wp.vec3)):
    """Stand-in for the robot path's pad-box kernel: writes the material's box arrays in place."""
    e = wp.tid()
    for b in range(nboxes):
        centers[e*nboxes+b] = wp.vec3(boxes[e, b, 0, 0], boxes[e, b, 0, 1], boxes[e, b, 0, 2])
        halves[e*nboxes+b] = wp.vec3(boxes[e, b, 1, 0], boxes[e, b, 1, 1], boxes[e, b, 1, 2])


def _specs(boxes=BOXES):
    specs = []
    for seed, yaw in ROWS:
        material = yawed_trim_material(seed, yaw, boxes)
        specs.append(layout.EnvSpec(material, yawed_start_pose(material, seed), 42))
    return specs


def _sequence(starts, steps):
    """Synthetic measured edges (float32-rounded, slightly non-unit quaternions) and pad boxes per step."""
    rng = np.random.default_rng(5)
    out = []
    for step in range(steps):
        poses = np.stack([p.copy() for p in starts])
        poses[:, 0] += .001*(step+1)
        poses[:, 3:] *= 1+rng.uniform(-5e-7, 5e-7, (len(starts), 1))
        poses = poses.astype(np.float32).astype(np.float64)
        boxes = np.repeat(np.asarray(BOXES, np.float32)[None], len(starts), 0)
        boxes[:, 1:, 0, 1] += np.float32(.0004*(step+1))*np.array([1, -1], np.float32)
        out.append((poses, np.array([0, 0 if step < 2 else 1], np.int32), boxes))
    return out


@pytest.mark.cpu_sim
def test_lab_backend_coupling_surface_equals_kinematic_backend_bitwise():
    lab = MB.BatchedCuttingMaterial(_specs(), device="cpu", parity_mode=True, backend="isaac_lab_newton")
    kinematic = MB.BatchedCuttingMaterial(_specs(), device="cpu", parity_mode=True)
    assert lab.material_coupled and lab.name == "isaac_lab_newton_measured_edge_and_pads_v1"
    assert lab.box_centers is lab.solver.box_centers and lab.pending_wrench.shape == (2, 6)
    assert lab.pad_wrench.shape == (2, 3, 6)
    pending, pad = lab.pending_wrench, lab.pad_wrench           # read in place: the same arrays every step
    starts = [lab.initial_measured_pose.numpy()[e] for e in range(2)]
    sequence = _sequence(starts, 4)
    for step, (poses, cuts, boxes) in enumerate(sequence):
        if step == 3:                                       # masked reset of env 0 through the surface
            lab.reset_masked(wp.array(np.array([True, False]), dtype=wp.bool, device="cpu"))
            kinematic.reset([0])
            for e in range(2):
                assert compare_snapshots(env_state(lab, e), env_state(kinematic, e)) == [], ("reset", e)
            poses = poses.copy()
            poses[0] = sequence[0][0][0]
        wp.launch(_write_boxes, 2, [wp.array(boxes, dtype=wp.float32, device="cpu"), 3, lab.box_centers,
                                    lab.box_halves], device="cpu")
        lab.set_measured_poses(wp.array(poses, dtype=wp.float64, device="cpu"),
                               wp.array(cuts, dtype=wp.int32, device="cpu"))
        lab.advance()
        kinematic.set_obstacle_boxes(boxes)
        kinematic.advance(poses, cuts, interpolation="device")
        assert lab.pending_wrench is pending and lab.pad_wrench is pad
        for e in range(2):
            assert compare_snapshots(env_state(lab, e), env_state(kinematic, e)) == [], (step, e)
        assert np.array_equal(lab.reset_mask.numpy(), kinematic.reset_mask.numpy()) or step < 3
    assert np.all(np.abs(pad.numpy()[:, 1:]).sum(-1) > 0) and np.all(np.abs(pending.numpy()).sum(1) > 0)
    assert np.count_nonzero(lab.solver.code.numpy()) > 0
    assert lab.seed.numpy().tolist() == [42, 42] and lab.step_count.numpy().tolist() == [1, 4]
    receipt = lab.receipt()
    assert receipt["backend"] == "isaac_lab_newton" and receipt["material_coupled"] is True
    assert "pad_boxes_written_in_place_by_backend_each_control_step" in receipt["material_fidelity_reasons"]
    assert kinematic.receipt()["backend"] == MB.BACKEND


def test_lab_backend_requires_pad_recipients_and_accepts_mask_forms():
    from batched_fixtures import batch_material
    with pytest.raises(UnsupportedBatchConfiguration) as info:
        batch_material([(512, None, 42)], backend="isaac_lab_newton")
    assert info.value.code == "UNSUPPORTED_MATERIAL_COUPLING"
    material = MB.BatchedCuttingMaterial(_specs(), device="cpu", parity_mode=True, backend="isaac_lab_newton")
    material.seed.assign(np.array([5, 6], np.int64))
    for mask in (wp.array(np.array([0, 1], np.int32), dtype=wp.int32, device="cpu"), np.array([False, True])):
        material.reset_masked(mask)
        assert material.reset_mask.numpy().tolist() == [0, 1] and material.seed.numpy().tolist() == [5, 6]
    with pytest.raises(ValueError):
        material.reset_masked(wp.zeros(2, dtype=wp.float32, device="cpu"))
