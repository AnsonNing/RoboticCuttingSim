"""P1-5 T5.3: sever_bond_progress_packed_v1 (pack-time sever bonds, NumPy reference vs the Torch device reducer,
dispatch by goal type). NumPy parts run in .venv; Torch parts need .deps/isaacsim-venv (plan D-09) and are NOT_RUN
where torch is missing. CPU only."""
import numpy as np
import pytest

from robotic_cutting_sim.rl import reward_packed as RP
from robotic_cutting_sim.rl import reward_sever as RS

SPACING = .001


def _bar(nx=8, ny=3, nz=2):
    """Lattice of a small bar centred on x = 0 (sites at +-0.5, +-1.5 ... mm), like the CUT-01-R sampler."""
    lattice = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
    points = (lattice + .5) * SPACING - np.array([nx, ny, 0]) * SPACING / 2
    return points, lattice


def _valid_codes(rng, n, K):
    digits = rng.integers(0, 3, size=(n, K)).astype(np.int64)
    return (digits * (4 ** np.arange(K))).sum(1).astype(np.int32)


def test_target_bonds_cross_the_plane_and_off_target_respects_the_bound():
    points, lattice = _bar()
    plane = [(np.zeros(3), np.array([1., 0., 0.]))]
    bonds = RS.build_sever_bonds(points, lattice, plane, .002)
    xa, xb = points[bonds.target_a, 0], points[bonds.target_b, 0]
    assert len(bonds.target_a) == 3 * 2 and np.all(np.sign(xa) != np.sign(xb))
    mid = .5 * (points[bonds.substrate_a, 0] + points[bonds.substrate_b, 0])
    assert np.all(np.abs(mid) > .002)
    a, b, _ = RP.lattice_bonds(lattice)
    all_mid = .5 * (points[a, 0] + points[b, 0])
    assert len(bonds.substrate_a) == int((np.abs(all_mid) > .002).sum())
    r = bonds.receipt
    assert r["target_bonds"] + r["off_target_bonds"] + r["neutral_bonds"] == r["lattice_bonds"]
    assert r["target_bonds_by_axis"] == [6, 0, 0]


def test_divide_planes_union_and_refusals():
    points, lattice = _bar(nx=12)
    planes = [(np.array([-.002, 0, 0]), np.array([1., 0, 0])), (np.array([.002, 0, 0]), np.array([1., 0, 0]))]
    bonds = RS.build_sever_bonds(points, lattice, planes, .001)
    assert bonds.receipt["target_bonds_per_plane"] == [6, 6] and len(bonds.target_a) == 12
    with pytest.raises(ValueError):
        RS.build_sever_bonds(points, lattice, [], .002)
    with pytest.raises(ValueError):
        RS.build_sever_bonds(points, lattice, [(np.zeros(3), np.array([2., 0, 0]))], .002)
    with pytest.raises(ValueError):
        RS.build_sever_bonds(points, lattice, [(np.array([1., 0, 0]), np.array([1., 0, 0]))], .002)


def test_perfect_cut_gives_full_progress_and_no_off_target():
    points, lattice = _bar()
    bonds = RS.build_sever_bonds(points, lattice, [(np.zeros(3), np.array([1., 0., 0.]))], .002)
    packed = RP.pack_bonds([bonds])
    codes = np.where(points[:, 0] > 0, 2, 1).astype(np.int32)             # digit 0: negative / positive side
    target, off = RS.severed_counts_numpy(codes, packed)
    assert target.tolist() == [6] and off.tolist() == [0]
    wrong = np.where(points[:, 0] > .003, 2, 1).astype(np.int32)           # a cut 3.5 mm from the plane
    target, off = RS.severed_counts_numpy(wrong, packed)
    assert target.tolist() == [0] and off.tolist() == [6]
    reward, dp, do = RS.reward_numpy((np.zeros(1), np.zeros(1)), (np.array([6]), np.array([0])), [False],
                                     packed.target_count, RS.SeverBondRewardConfig())
    assert dp.tolist() == [1.] and do.tolist() == [0.] and np.isclose(reward[0], 1. - 1e-4)


def test_cut_workload_bonds_and_dispatch():
    from robotic_cutting_sim.rl.cut_workload import compile_workload_task
    task = compile_workload_task()
    version, bonds = RS.bonds_for_task(task)
    assert version == RS.REWARD_VERSION
    r = bonds.receipt
    assert r["target_bonds"] == 600 and r["target_bonds_by_axis"] == [600, 0, 0]      # 30 x 20 crossing columns
    assert r["off_target_bound_m"] == .002 and r["task_id"] == task.task_id
    points, _, lattice = task.sample_particles()
    a, b = bonds.target_a, bonds.target_b
    assert np.all(points[a, 0] * points[b, 0] < 0) and np.allclose(np.abs(points[a, 0]), 5e-4)


def test_dispatch_trim_and_refusal():
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    trim = compile_task(generate_scene(512, distribution="id"), goal_config(), seed=42, particle_spacing_m=.004)
    version, bonds = RS.bonds_for_task(trim)
    assert version == RP.REWARD_VERSION and bonds.receipt["target_bonds"] > 0

    class Other:
        goal_kind, goal_json = "move", '{"type": "move_piece"}'
        goal = {"type": "move_piece"}
    with pytest.raises(ValueError, match="UNSUPPORTED_REWARD"):
        RS.bonds_for_task(Other())
    with pytest.raises(ValueError, match="UNSUPPORTED_REWARD"):
        RS.reward_class_for_version("nope")
    cls, config = RS.reward_class_for_version(RS.REWARD_VERSION)
    assert cls is RS.SeverBondProgressReward and config.to_dict()["status"] == "PROVISIONAL_SHAPING_NOT_SUCCESS"
    assert config.to_dict()["version"] == RS.REWARD_VERSION


def test_torch_reducer_matches_the_numpy_reference_over_two_envs():
    torch = pytest.importorskip("torch")
    p1, l1 = _bar()
    p2, l2 = _bar(nx=10, ny=2)
    plane = [(np.zeros(3), np.array([1., 0., 0.]))]
    env_bonds = [RS.build_sever_bonds(p1, l1, plane, .002), RS.build_sever_bonds(p2, l2, plane, .002)]
    packed = RP.pack_bonds(env_bonds)
    reward = RS.SeverBondProgressReward(packed, device="cpu")
    meta = reward.metadata()
    assert meta["version"] == RS.REWARD_VERSION and meta["status"] == RS.REWARD_STATUS
    assert meta["off_target_bonds"] == packed.substrate_count.tolist() and "substrate_bonds" not in meta
    rng = np.random.default_rng(3)
    P = int(packed.particle_offset[-1])
    codes = np.zeros(P, np.int32)
    reward.reset(torch.as_tensor(codes))
    previous = RS.severed_counts_numpy(codes, packed)
    for step in range(5):
        fresh = _valid_codes(rng, P, 4)
        codes = np.where(codes != 0, codes, np.where(rng.random(P) < .3, fresh, 0)).astype(np.int32)
        invalid = np.array([False, step == 3])
        r, terms = reward.step(torch.as_tensor(codes), torch.as_tensor(invalid))
        counts = RS.severed_counts_numpy(codes, packed)
        expected, _, _ = RS.reward_numpy(previous, counts, invalid, packed.target_count, RS.SeverBondRewardConfig())
        np.testing.assert_allclose(r.numpy(), expected.astype(np.float32), rtol=0, atol=0)
        assert terms["severed_target_bonds"].tolist() == counts[0].tolist()
        assert terms["severed_off_target_bonds"].tolist() == counts[1].tolist()
        assert terms["severed_substrate_bonds"].tolist() == counts[1].tolist()      # contract slot name (alias)
        previous = counts
