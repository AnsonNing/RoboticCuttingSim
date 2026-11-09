"""ADR 0021 §5 L0 for the physics paths the TRIM sequences leave at zero (review finding 2, 2026-11-10).

Bitwise (``np.array_equal`` over every Warp array of the frozen single solver, after every substep) on the
Warp CPU device, and every test asserts that the quantity it exercises is non-zero:

* ``test_block_scenes_contacts_boxes_and_per_env_digits_bitwise`` -- three heterogeneous lattice blocks (sizes,
  Young's modulus, friction, particle-contact stiffness, yield stress, fracture energy) driven by a bare
  ``BatchedTLMPMv7_2``: compressed lattices with conflicting cut codes (particle-particle contacts), a handle
  box and two pad boxes overlapping the material (box reactions in every row), a different cut digit per env
  in the same substep (0, 1 and -1) and a rotated tool;
* ``test_pad_recipients_and_rotated_workpiece_wrench_bitwise`` -- ``BatchedCuttingMaterial`` against frozen
  ``LiveCuttingMaterial`` singles with handle + two pads configured as reaction recipients and a yawed workpiece
  (non-identity ``T_world_local``): material wrench, pad wrenches and the delayed handoff are bitwise equal.

Not reached by any full-solver scene tried: the frontier link label (``link_diagnostics[1]``, a pending link
labelled in a later substep). That path is covered bitwise at kernel level by the WP17 fixtures whose expected
diagnostics are ``(., 1)`` (``test_batched_cpu_parity.py::test_wp17_frontier_fixture_batched_bitwise``).
"""
import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from batched_fixtures import block_bed, compare_env
from robotic_cutting_sim.batched import layout
from robotic_cutting_sim.batched.tlmpm_v7_2_batch import BatchedTLMPMv7_2

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]

# Edge-frame boxes: a "handle" behind and above the edge, two pads beside the blade (all overlap the blocks).
OBSTACLES = [[[-.010, 0., .004], [.004, .004, .004]], [[0., .010, 0.], [.003, .003, .003]],
             [[0., -.010, 0.], [.003, .003, .003]]]
SUBSTEPS = 40
BLOCKS = (
    # size, frozen TLMPMConfig overrides, lattice compression, cut digit, edge x path, tool yaw [rad]
    ((.030, .020, .012), {}, 1., 0, (-.016, -.008), 0.),
    ((.026, .024, .010), dict(young=8e4, contact_friction=.45, particle_contact_stiffness=.08, yield_stress=700.),
     .72, 1, (-.002, .002), .1),
    ((.034, .018, .012), dict(cutting_energy=.09, young=6e4), .8, -1, (-.004, .003), -.05),
)


def _trajectory(path, yaw, T):
    q = Rotation.from_rotvec([0., 0., yaw]).as_quat()
    poses = np.tile(np.r_[0., 0., .006, q], (T, 1)).astype(np.float32)
    poses[:, 0] = np.linspace(path[0], path[1], T)
    twist = np.array([(path[1]-path[0])/((T-1)*1e-4), 0., 0., 0., 0., 0.], np.float32)
    return poses, np.tile(twist, (T, 1))


@pytest.mark.cpu_sim
def test_block_scenes_contacts_boxes_and_per_env_digits_bitwise():
    T = SUBSTEPS+1
    singles, beds = [], []
    for size, config, *_ in BLOCKS:
        for out in (singles, beds):
            bed = block_bed(size, OBSTACLES, **config)
            bed.set_trajectory(*_trajectory((0., 0.), 0., T), np.full(T, -1, np.int32))
            out.append(bed)
    packed = layout.pack_from_beds(beds)
    batch = BatchedTLMPMv7_2(packed, "cpu")
    assert len(set(packed.particle_counts.tolist())) == 3
    params = batch.params.numpy()
    assert all(not np.array_equal(params[a], params[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    offsets = packed.particle_offset
    x, v, code = batch.x.numpy(), batch.v.numpy(), batch.code.numpy()
    rng = np.random.default_rng(0)
    initial_codes = []
    poses, twists, cuts = [], [], []
    for e, (single, (_, _, compression, digit, path, yaw)) in enumerate(zip(singles, BLOCKS)):
        reference = single.reference.numpy()
        center = reference.mean(0)
        xe = (center+compression*(reference-center)).astype(np.float32)   # 2r = 1.8 mm: compressed pairs overlap
        ve = rng.normal(scale=.01, size=reference.shape).astype(np.float32)
        ce = (np.where(reference[:, 0] < center[0], 1, 2) if compression < 1 else np.zeros(len(reference))).astype(np.int32)
        single.x.assign(xe)
        single.v.assign(ve)
        single.code.assign(ce)
        rows = slice(int(offsets[e]), int(offsets[e+1]))
        x[rows], v[rows], code[rows] = xe, ve, ce
        initial_codes.append(ce)
        p, w = _trajectory(path, yaw, T)
        single.poses.assign(p)
        single.twists.assign(w)
        single.cuts.assign(np.full(T, digit, np.int32))
        for array in (single.index, single.reaction, single.box_reactions, single.stats):
            array.zero_()
        poses.append(p)
        twists.append(w)
        cuts.append(np.full(T, digit, np.int32))
    batch.x.assign(x)
    batch.v.assign(v)
    batch.code.assign(code)
    batch.poses.assign(np.concatenate(poses))
    batch.twists.assign(np.concatenate(twists))
    batch.cuts.assign(np.concatenate(cuts))
    batch.zero_control_step_accumulators()
    for substep in range(SUBSTEPS):
        for single in singles:
            single.step()
        batch.step()
        for e, single in enumerate(singles):
            assert compare_env(batch, e, single) == [], f"substep {substep} env {e}"
    # Non-vacuous: every path named in the docstring was exercised.
    stats = np.stack([s.stats.numpy() for s in singles])
    boxes = np.stack([np.abs(s.box_reactions.numpy()).sum(1) > 0 for s in singles])
    assert (stats[:, 1] > 0).all(), ("blade contacts", stats[:, 1])
    assert (stats[1:, 4] > 0).all(), ("particle-particle contacts", stats[:, 4])
    assert boxes.any(axis=1).all() and boxes.any(axis=0).all(), boxes     # every env and every box reacts
    assert singles[0].link_diagnostics.numpy()[0] > 0, "frontier link candidates"
    codes = [s.code.numpy() for s in singles]
    assert np.any((codes[0] & 3) != 0), "digit 0 labels in env 0"
    assert np.any(((codes[1] >> 2) & 3) != 0) and not np.any((initial_codes[1] >> 2) & 3), "digit 1 labels in env 1"
    assert np.array_equal(codes[2], initial_codes[2]), "env 2 is disarmed in the same substeps"


def test_host_wrench_rotation_order_the_kernel_reproduces():
    """``rotate_wrench_at_same_point`` (w[:3] @ R.T, a (3,)@(3,3) NumPy product) uses the BLAS order
    fma(c,R2, fma(a,R0, b*R1)) on this host; ``material_batch._rotate`` reproduces it. On a host whose BLAS
    selects another kernel this test fails first and the wrench test below may then differ by ~1 ulp."""
    from fractions import Fraction

    from robotic_cutting_sim.scene_frames import rotate_wrench_at_same_point

    def fma(a, b, c):
        return float(Fraction(a)*Fraction(b)+Fraction(c))
    rng = np.random.default_rng(9)
    for _ in range(300):
        T = np.eye(4)
        T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        w = rng.normal(size=6).astype(np.float32).astype(float)
        out = rotate_wrench_at_same_point(T, w)
        R = T[:3, :3]
        expected = [fma(v[2], R[j, 2], fma(v[0], R[j, 0], v[1]*R[j, 1])) for v in (w[:3], w[3:]) for j in range(3)]
        assert out.tolist() == expected


def test_wrench_kernel_non_identity_rotation_bitwise():
    """``_wrench_b`` against ``live_material.advance``'s host arithmetic (rotate, subtract recipients' box
    impulses, divide by control_dt, delayed handoff) for 64 random rotations and recipients (None, a, b)."""
    from robotic_cutting_sim.batched import material_batch as MB
    from robotic_cutting_sim.scene_frames import rotate_wrench_at_same_point
    rng = np.random.default_rng(1)
    N, B, dt = 64, 3, .02
    Tw = np.repeat(np.eye(4)[None], N, 0)
    Tw[:, :3, :3] = Rotation.random(N, random_state=rng).as_matrix()
    reaction = rng.normal(size=(N, 6)).astype(np.float32)
    boxes = rng.normal(size=(N*B, 6)).astype(np.float32)
    pending0 = rng.normal(size=(N, 6))
    args = [wp.array(reaction, dtype=wp.spatial_vector, device="cpu"),
            wp.array(boxes, dtype=wp.spatial_vector, device="cpu"),
            wp.array(np.ascontiguousarray(Tw[:, :3, :3]), dtype=wp.mat33d, device="cpu"),
            wp.array(np.array([0, 1, 1], np.int32), dtype=int, device="cpu"), B, wp.float64(dt), 1,
            wp.array(pending0, dtype=wp.float64, device="cpu"), wp.zeros((N, 6), dtype=wp.float64, device="cpu"),
            wp.full((N, B, 6), 7., dtype=wp.float64, device="cpu")]
    wp.launch(MB._wrench_b, N, args, device="cpu")
    pending, previous, pad = args[7].numpy(), args[8].numpy(), args[9].numpy()
    for e in range(N):
        impulse = rotate_wrench_at_same_point(Tw[e], reaction[e].astype(float))
        for b in (1, 2):
            value = rotate_wrench_at_same_point(Tw[e], boxes[e*B+b].astype(float))
            impulse -= value
            assert np.array_equal(pad[e, b], value/dt)
        assert np.array_equal(pending[e], impulse/dt)
        assert np.array_equal(previous[e], pending0[e])
        assert not np.any(pad[e, 0])
    assert not np.allclose(Tw[:, :3, :3], np.eye(3))


PAD_BOXES = {   # edge frame: [handle (non-recipient), left pad, right pad]
    "handle_touches": [[[.004, 0., .004], [.003, .003, .003]], [[0., .01, .004], [.004, .003, .004]],
                       [[0., -.01, .004], [.004, .003, .004]]],
    "handle_clear": [[[-.004, 0., .006], [.003, .003, .003]], [[.002, .008, 0.], [.004, .003, .004]],
                     [[.002, -.008, 0.], [.004, .003, .004]]],
}


def _pad_scene(update_boxes=None):
    """(batch material, frozen singles, start poses): seeds 512 (yaw 1.5 deg, handle touches) and 512 (yaw -2 deg)."""
    from batched_fixtures import yawed_start_pose, yawed_trim_material
    from robotic_cutting_sim.batched.material_batch import BatchedCuttingMaterial
    rows = ((512, 1.5, "handle_touches"), (512, -2., "handle_clear"))
    singles, specs, starts = [], [], []
    for seed, yaw, boxes in rows:
        single = yawed_trim_material(seed, yaw, PAD_BOXES[boxes])
        start = yawed_start_pose(single, seed)
        single.reset(start, seed=42)
        singles.append(single)
        starts.append(start)
        specs.append(layout.EnvSpec(yawed_trim_material(seed, yaw, PAD_BOXES[boxes]), start, 42))
    return BatchedCuttingMaterial(specs, device="cpu", parity_mode=True), singles, starts


@pytest.mark.cpu_sim
def test_pad_recipients_and_rotated_workpiece_wrench_bitwise():
    material, singles, starts = _pad_scene()
    assert material.recipients.numpy().tolist() == [0, 1, 1]
    for e in range(2):
        assert not np.allclose(material.packed.env["T_world_local"][e][:3, :3], np.eye(3))
    touched = np.zeros((2, 3), bool)
    current = [s.copy() for s in starts]
    for step in range(3):
        for e in range(2):
            current[e] = current[e].copy()
            current[e][0] += .001
            singles[e].advance(current[e], (step+1)*.02, cutting=True, cut_index=0)
        material.advance(np.stack(current), np.zeros(2, np.int32), interpolation="device")
        pending, previous, pad = (material.pending_wrench.numpy(), material.applied_previous.numpy(),
                                  material.pad_wrench.numpy())
        for e, single in enumerate(singles):
            assert compare_env(material.solver, e, single.bed) == [], (step, e)
            assert np.array_equal(pending[e], single.pending_wrench), (step, e)
            assert np.array_equal(previous[e], single.last_diagnostics["applied_previous_material_wrench_world"])
            assert np.array_equal(pad[e, 1], single.pending_obstacle_wrenches["left"]), (step, e)
            assert np.array_equal(pad[e, 2], single.pending_obstacle_wrenches["right"]), (step, e)
            assert not np.any(pad[e, 0]), "the handle is not a recipient"
            touched[e] |= np.abs(single.bed.box_reactions.numpy()).sum(1) > 0
    assert touched.all(axis=0).any() and touched[0].all(), touched     # handle, left and right all react
    assert np.all(np.abs(material.pad_wrench.numpy()[:, 1:]).sum(-1) > 0)
    assert np.all(np.abs(material.pending_wrench.numpy()).sum(1) > 0)
    assert np.count_nonzero(singles[1].bed.code.numpy()) > 0, "env 1 cuts"
