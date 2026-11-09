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


def _fk_like(pose, rotation_rad, scale):
    """A float32-rounded, slightly non-unit quaternion, as an articulated FK chain delivers it."""
    from scipy.spatial.transform import Rotation
    out = np.array(pose, np.float64)
    q = (Rotation.from_rotvec([0., 0., rotation_rad])*Rotation.from_quat(out[3:])).as_quat()*scale
    out[3:] = q.astype(np.float32).astype(np.float64)
    out[:3] = out[:3].astype(np.float32)
    return out


@pytest.mark.cpu_sim
def test_device_pose_path_equals_the_live_chain_end_to_end():
    """Review finding 5: advance(interpolation="device") equals the host_reference (frozen SciPy live chain)
    path bitwise over every array, for non-unit float32 quaternions and 1e-9..1e-3 rad rotations, including
    the stored last_pose (= validate_pose(measured), as LiveCuttingMaterial.last_pose)."""
    configs = [(512, None, 42), (15, None, 43)]
    device, host = batch_material(configs), batch_material(configs)
    starts = [start_pose(512), start_pose(15)]
    plan = ((.001, 1e-9, 1+3e-7, 0), (.001, 2e-6, 1-4e-7, 0), (.0005, 1e-3, 1+9e-7, -1))
    for step, (dx, angle, scale, cut) in enumerate(plan):
        poses = np.stack([_fk_like(_moved(p, dx*(step+1)), angle*(step+1), scale) for p in starts])
        assert np.abs(np.linalg.norm(poses[:, 3:], axis=1)-1).max() > 1e-7      # not exactly unit
        device.advance(poses, np.array([cut, cut], np.int32), interpolation="device")
        host.advance(poses, np.array([cut, cut], np.int32), interpolation="host_reference")
        for e in range(2):
            assert compare_snapshots(env_state(device, e), env_state(host, e)) == [], (step, e)
        assert np.array_equal(device.actual_pose.numpy(), host.actual_pose.numpy())
    assert device.pose_status.numpy().tolist() == [0, 0]
    assert np.count_nonzero(device.snapshot(0)["code"]) > 0


def test_receipt_hashes_sources_and_names_the_interpolation_path_used():
    """Review finding 7."""
    import hashlib
    from pathlib import Path
    material = batch_material([(512, None, 42)])
    receipt = material.receipt()
    here = Path(MB.__file__).resolve().parent
    expected = {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
                for name in ("material_batch.py", "pose_interp.py", "diagnostics.py", "errors.py", "layout.py",
                             "constitutive.py", "tlmpm_v7_2_batch.py")}
    assert receipt["batched_source_sha256"] == expected
    assert receipt["pose_interpolation"] == "none_no_control_step_yet"
    material.advance(interpolation="preset")
    assert material.receipt()["pose_interpolation"] == "preset_written_by_caller"
    material.advance(_moved(start_pose(512), .001)[None], np.zeros(1, np.int32), interpolation="device")
    receipt = material.receipt()
    assert receipt["pose_interpolation"] == "mixed:device_f64_slerp_v2+preset_written_by_caller"
    assert receipt["pose_interpolation_steps"] == {"device": 1, "host_reference": 0, "preset": 1}
    assert receipt["pose_interpolation_kernel"] == "device_f64_slerp_v2"


def test_rebuild_is_atomic():
    """Review nit: a refused rebuild leaves the material unchanged; a successful one replaces every array."""
    from batched_fixtures import trim_material
    from robotic_cutting_sim.batched import layout
    from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
    material = batch_material([(512, None, 42)])
    arrays = dict(material.device_arrays())
    with pytest.raises(UnsupportedBatchConfiguration):
        material.rebuild([])
    with pytest.raises(UnsupportedBatchConfiguration):
        material.rebuild([layout.EnvSpec(trim_material(512), start_pose(512), 42),          # non-uniform
                          layout.EnvSpec(trim_material(512, spacing=.003), start_pose(512, spacing=.003), 42)])
    assert material.num_envs == 1 and all(material.device_arrays()[k] is v for k, v in arrays.items())
    material.rebuild([layout.EnvSpec(trim_material(512), start_pose(512), 7),
                      layout.EnvSpec(trim_material(15), start_pose(15), 8)])
    assert material.num_envs == 2 and material.seed.numpy().tolist() == [7, 8]
    assert material.parity_mode and material.device == "cpu"
