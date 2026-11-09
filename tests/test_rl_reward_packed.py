"""trim_bond_progress_packed_v1: pack-time bond CSR, frozen conflict rule port, NumPy reference vs Torch device path.

NumPy/Warp parts run in .venv; Torch parts need .deps/isaacsim-venv (plan D-09) and are NOT_RUN where torch is
missing. CPU only."""
import numpy as np
import pytest

from robotic_cutting_sim.rl import reward_packed as RP

SPACING = .001


def _valid_codes(rng, n, K):
    digits = rng.integers(0, 3, size=(n, K)).astype(np.int64)
    return (digits*(4**np.arange(K))).sum(1).astype(np.int32)


def _frozen_conflicts(K):
    wp = pytest.importorskip("warp")
    from robotic_cutting_sim import tlmpm_v5 as V5

    @wp.kernel
    def frozen(a: wp.array(dtype=int), b: wp.array(dtype=int), out: wp.array(dtype=int)):
        i = wp.tid()
        out[i] = wp.where(V5._conflict_packed(a[i], b[i]), 1, 0)
    rng = np.random.default_rng(K)
    n = 20000
    a, b = _valid_codes(rng, n, K), _valid_codes(rng, n, K)
    b[:500] = a[:500]                                   # identical codes never conflict
    out = wp.zeros(n, dtype=int, device="cpu")
    wp.launch(frozen, n, [wp.array(a, dtype=int, device="cpu"), wp.array(b, dtype=int, device="cpu"), out],
              device="cpu")
    expected = out.numpy().astype(bool)
    assert expected.any() and not expected[:500].any()
    return a, b, expected


@pytest.mark.parametrize("K", [1, 2, 15])
def test_numpy_conflict_port_matches_the_frozen_warp_function(K):
    a, b, expected = _frozen_conflicts(K)
    np.testing.assert_array_equal(RP.conflict_packed_numpy(a, b), expected)


@pytest.mark.parametrize("K", [1, 2, 15])
def test_torch_conflict_port_matches_the_frozen_warp_function(K):
    torch = pytest.importorskip("torch")
    a, b, expected = _frozen_conflicts(K)
    np.testing.assert_array_equal(RP.conflict_packed_torch(torch.as_tensor(a), torch.as_tensor(b)).numpy(), expected)


def test_lattice_bonds_are_the_six_neighbour_pairs_independent_of_order():
    lattice = np.stack(np.meshgrid(*(np.arange(3),)*3, indexing="ij"), -1).reshape(-1, 3)
    perm = np.random.default_rng(0).permutation(len(lattice))
    a, b, axis = RP.lattice_bonds(lattice[perm])
    assert len(a) == 54 and np.bincount(axis).tolist() == [18, 18, 18]
    diff = lattice[perm][b]-lattice[perm][a]
    assert (np.abs(diff).sum(1) == 1).all() and (diff.sum(1) == 1).all()
    with pytest.raises(ValueError):
        RP.lattice_bonds(np.r_[lattice, lattice[:1]])


def _synthetic_env(nx=6, ny=4, protrusion=((2, 4), (1, 3))):
    """Substrate 3 layers (z <= 3 mm), protrusion 2 layers over a column block; plane at 3 mm."""
    sites = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(3)]
    sites += [(i, j, k) for i in range(*protrusion[0]) for j in range(*protrusion[1]) for k in (3, 4)]
    lattice = np.array(sites, np.int64)
    points = (lattice+.5)*SPACING
    (x0, x1), (y0, y1) = protrusion

    def footprint(x, y):
        inside = (x > x0*SPACING) & (x < x1*SPACING) & (y > y0*SPACING) & (y < y1*SPACING)
        return np.where(inside, .002, 0.)
    return points, lattice, footprint


def test_synthetic_bond_sets_and_hand_counted_severing():
    points, lattice, footprint = _synthetic_env()
    bonds = RP.build_env_bonds(points, lattice, .003, footprint)
    assert bonds.receipt["target_bonds"] == 4 and bonds.receipt["crossing_outside_footprint"] == 0
    assert bonds.receipt["substrate_bonds"] == 5*4*3+6*3*3+6*4*2
    assert (points[bonds.target_a, 2] <= .003).all() and (points[bonds.target_b, 2] > .003).all()
    packed = RP.pack_bonds([bonds])
    codes = np.where(points[:, 2] > .003, 1, 2).astype(np.int32)        # ideal planar cut, digit 0
    t, s = RP.severed_counts_numpy(codes, packed)
    assert t.tolist() == [4] and s.tolist() == [0]
    codes[(lattice[:, 0] == 2) & (lattice[:, 2] >= 3)] = 0              # one protrusion row not reached
    t, s = RP.severed_counts_numpy(codes, packed)
    assert t.tolist() == [2] and s.tolist() == [0]
    inner = np.flatnonzero((lattice == [3, 2, 1]).all(1))[0]           # interior substrate particle, 6 neighbours
    codes[inner] = 1
    t, s = RP.severed_counts_numpy(codes, packed)
    assert s.tolist() == [6]
    codes[inner] = 1+4                                                  # a second digit does not change digit 0
    assert RP.severed_counts_numpy(codes, packed)[1].tolist() == [6]
    depth = RP.build_env_bonds(points, lattice, .003, footprint, substrate_depth_m=.0021)
    assert depth.receipt["substrate_bonds"] == 5*4*2+6*3*2+6*4*1


def _three_envs():
    envs = [_synthetic_env(), _synthetic_env(8, 5, ((1, 4), (1, 4))), _synthetic_env(5, 4, ((0, 2), (0, 4)))]
    bonds = [RP.build_env_bonds(p, l, .003, f) for p, l, f in envs]
    return envs, bonds, RP.pack_bonds(bonds)


def test_packing_offsets_and_guards():
    envs, bonds, packed = _three_envs()
    assert packed.num_envs == 3 and packed.particle_offset[-1] == sum(len(p) for p, _, _ in envs)
    for e, b in enumerate(bonds):
        lo, hi = packed.target_offset[e], packed.target_offset[e+1]
        np.testing.assert_array_equal(packed.target_a[lo:hi], b.target_a+packed.particle_offset[e])
    with pytest.raises(ValueError):
        RP.pack_bonds(bonds, particle_offset=[0, 1, 2, 3])
    p, l, _ = envs[0]
    with pytest.raises(ValueError):
        RP.build_env_bonds(p, l, .003, lambda x, y: np.zeros_like(x))
    with pytest.raises(ValueError):
        RP.TrimBondRewardConfig(time_penalty=-1.)


def _cutting_sequence(rng, envs, steps=6):
    """Monotone code histories (digits only go 0 -> 1|2), concatenated over envs."""
    histories = []
    for points, lattice, _ in envs:
        final = np.where(points[:, 2] > .003, 1, 2).astype(np.int32)+4*rng.integers(0, 3, len(points)).astype(np.int32)
        order = rng.permutation(len(points))
        seq = []
        for s in range(steps+1):
            codes = np.zeros(len(points), np.int32)
            done = order[:len(points)*s//steps]
            codes[done] = final[done]
            seq.append(codes)
        histories.append(seq)
    return [np.concatenate([h[s] for h in histories]) for s in range(steps+1)]


def test_numpy_reference_reward_sequence():
    rng = np.random.default_rng(4)
    envs, _, packed = _three_envs()
    seq = _cutting_sequence(rng, envs)
    config = RP.TrimBondRewardConfig()
    prev = RP.severed_counts_numpy(seq[0], packed)
    total = np.zeros(3)
    for codes in seq[1:]:
        counts = RP.severed_counts_numpy(codes, packed)
        reward, dp, do = RP.reward_numpy(prev, counts, np.zeros(3, bool), packed.target_count, config)
        assert (dp >= 0).all() and (do >= 0).all()
        total += dp
        prev = counts
    np.testing.assert_allclose(total, 1.)            # every target bond severed by the end
    reward, _, _ = RP.reward_numpy(prev, prev, np.array([False, True, False]), packed.target_count, config)
    np.testing.assert_allclose(reward, [-config.time_penalty, -config.invalid_penalty, -config.time_penalty])


def test_torch_device_path_matches_numpy_reference_with_partial_reset():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    envs, _, packed = _three_envs()
    seq = _cutting_sequence(rng, envs)
    config = RP.TrimBondRewardConfig(progress_weight=2., off_target_weight=.5, time_penalty=1e-3, invalid_penalty=3.)
    reward_fn = RP.TrimBondProgressReward(packed, config)
    reward_fn.reset(torch.as_tensor(seq[0]))
    prev = RP.severed_counts_numpy(seq[0], packed)
    for s, codes in enumerate(seq[1:]):
        invalid = np.array([False, s == 3, False])
        reward, terms = reward_fn.step(torch.as_tensor(codes), torch.as_tensor(invalid))
        counts = RP.severed_counts_numpy(codes, packed)
        expected, dp, do = RP.reward_numpy(prev, counts, invalid, packed.target_count, config)
        assert reward.dtype == torch.float32 and reward.shape == (3,)
        np.testing.assert_array_equal(reward.numpy(), expected.astype(np.float32))
        np.testing.assert_array_equal(terms["severed_target_bonds"].numpy(), counts[0])
        np.testing.assert_array_equal(terms["severed_substrate_bonds"].numpy(), counts[1])
        np.testing.assert_array_equal(terms["progress_delta"].numpy(), dp)
        prev = counts
    # partial reset of env 1 to fresh codes: env 0 and 2 baselines untouched
    fresh = seq[-1].copy()
    lo, hi = packed.particle_offset[1], packed.particle_offset[2]
    fresh[lo:hi] = 0
    before = reward_fn.counts.clone()
    reward_fn.reset(torch.as_tensor(fresh), env_ids=[1])
    assert reward_fn.counts[:, 0].tolist() == before[:, 0].tolist() and reward_fn.counts[:, 2].tolist() == before[:, 2].tolist()
    assert reward_fn.counts[:, 1].tolist() == [0, 0]
    reward, _ = reward_fn.step(torch.as_tensor(fresh))
    np.testing.assert_allclose(reward.numpy(), -config.time_penalty, rtol=1e-6)
    with pytest.raises(ValueError):
        reward_fn.step(torch.as_tensor(fresh.astype(np.int64)))


@pytest.fixture(scope="module")
def trim_seeds():
    from robotic_cutting_sim.seeded_trim import compile_seed
    out = []
    for seed in (512, 15):
        task, _ = compile_seed(seed)
        particles = task.sample_particles()
        out.append((seed, task, particles, RP.bonds_from_task(task, particles=particles)))
    return out


def test_real_trim_seed_reference_lattice_bonds(trim_seeds):
    for seed, task, (points, _, lattice), bonds in trim_seeds:
        h = float(task.target_origin_local_m[2])
        assert h == task.geometry.base_height_m
        r = bonds.receipt
        assert r["particles"] == len(points) and r["target_bonds"] > 50, (seed, r)
        a, b = bonds.target_a, bonds.target_b
        assert (lattice[b]-lattice[a] == [0, 0, 1]).all()
        assert (points[a, 2] <= h).all() and (points[b, 2] > h).all()
        # one crossing bond per footprint column that carries material above the plane
        cols = {tuple(c) for c in lattice[points[:, 2] > h][:, :2]}
        assert r["target_bonds"]+r["crossing_outside_footprint"] == len(cols)
        assert (points[bonds.substrate_a, 2] <= h).all() and (points[bonds.substrate_b, 2] <= h).all()


def _ideal_cut_codes(points, h):
    return np.where(points[:, 2] > h, 1, 2).astype(np.int32)


def test_real_trim_seed_semantics_numpy(trim_seeds):
    bonds = [b for *_, b in trim_seeds]
    packed = RP.pack_bonds(bonds)
    points = [p for _, _, (p, _, _), _ in trim_seeds]
    h = [float(t.target_origin_local_m[2]) for _, t, _, _ in trim_seeds]
    zero = np.zeros(packed.particle_offset[-1], np.int32)
    assert [c.tolist() for c in RP.severed_counts_numpy(zero, packed)] == [[0, 0], [0, 0]]
    ideal = np.concatenate([_ideal_cut_codes(p, z) for p, z in zip(points, h)])
    t, s = RP.severed_counts_numpy(ideal, packed)
    assert t.tolist() == packed.target_count.tolist() and s.tolist() == [0, 0]
    deep = np.concatenate([_ideal_cut_codes(p, z-SPACING) for p, z in zip(points, h)])   # 1 mm too deep
    t, s = RP.severed_counts_numpy(deep, packed)
    assert t.tolist() == [0, 0] and (s > 0).all()


def test_real_trim_seed_torch_matches_numpy(trim_seeds):
    torch = pytest.importorskip("torch")
    bonds = [b for *_, b in trim_seeds]+[trim_seeds[0][-1]]          # N=3: seeds 512, 15, 512
    packed = RP.pack_bonds(bonds)
    rng = np.random.default_rng(15)
    reward_fn = RP.TrimBondProgressReward(packed)
    P = int(packed.particle_offset[-1])
    reward_fn.reset(torch.zeros(P, dtype=torch.int32))
    codes = np.zeros(P, np.int32)
    prev = RP.severed_counts_numpy(codes, packed)
    for _ in range(4):
        grow = rng.random(P) < .25
        codes = np.where(grow & (codes == 0), _valid_codes(rng, P, 2), codes)
        reward, terms = reward_fn.step(torch.as_tensor(codes))
        counts = RP.severed_counts_numpy(codes, packed)
        expected, _, _ = RP.reward_numpy(prev, counts, np.zeros(3, bool), packed.target_count, reward_fn.config)
        np.testing.assert_array_equal(terms["severed_target_bonds"].numpy(), counts[0])
        np.testing.assert_array_equal(terms["severed_substrate_bonds"].numpy(), counts[1])
        np.testing.assert_array_equal(reward.numpy(), expected.astype(np.float32))
        prev = counts
    assert reward_fn.metadata()["status"] == "PROVISIONAL_SHAPING_NOT_SUCCESS"
