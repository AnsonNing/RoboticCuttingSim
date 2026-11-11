"""P1-6 (cycle p16c1): opt-in ``alias_free_v2`` particle contacts of the batched v7_2 solver (Warp CPU, bitwise).

``alias_free_v2`` = ``alias_free_v1`` (C1 + C2 + fallback guard) with the contact query walking env-major buckets
(``EnvMajorGrid``: key ``env*C + cell``, no per-cell group binary search) and, by default, a per-env query shrink
to ``2r + 2 dt V`` while every particle of the env is slower than ``V = shrink_speed_m_s``. Every scene compares
with the frozen single solver after every substep (``np.array_equal`` on every array, as the v1 tests do) and
asserts non-vacuity: contacts are non-zero, some particle receives >= 2 non-zero contributions, the shrink is
active, and its speed check is necessary (disabling the check while keeping the shrunken reach changes the
result of the fast env).
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

import test_batched_particle_contacts as T1
from batched_fixtures import (SEQUENCE, advance_single, batch_material, compare_env, compare_snapshots, drive_solver,
                              env_state, host_inputs, start_pose, step_material, trim_material)
from robotic_cutting_sim.batched import layout
from robotic_cutting_sim.batched import material_batch as MB
from robotic_cutting_sim.batched import particle_contacts as PC
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.batched.tlmpm_v7_2_batch import BatchedTLMPMv7_2

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]

V2 = PC.ALIAS_FREE_V2_MODE
VARIANTS = {"v2_shrink": {}, "v2_no_shrink": {"shrink_speed_m_s": None}}


def _config(options, reference_dims):
    return PC.resolve(V2, dict(options, reference_hash_dims=reference_dims))


# ----------------------------------------------------------------------------- candidate sequence (kernel level)

@wp.kernel
def _env_major_sequence(x: wp.array(dtype=wp.vec3), env_of_particle: wp.array(dtype=int), reach: float, inv: float,
                        dims: wp.vec3i, cells: int, n: int, keys: wp.array(dtype=int), starts: wp.array(dtype=int),
                        ends: wp.array(dtype=int), ids: wp.array(dtype=int), seq: wp.array2d(dtype=int),
                        count: wp.array(dtype=int)):
    """The cell walk of ``particle_contacts._particle_contacts_env_major_b`` (copied), recording candidates."""
    p = wp.tid()
    e = env_of_particle[p]
    xp = x[p]
    x0 = int(wp.floor((xp[0]-reach)*inv))
    y0 = int(wp.floor((xp[1]-reach)*inv))
    z0 = int(wp.floor((xp[2]-reach)*inv))
    x1 = wp.min(int(wp.floor((xp[0]+reach)*inv)), x0+dims[0]-1)
    y1 = wp.min(int(wp.floor((xp[1]+reach)*inv)), y0+dims[1]-1)
    z1 = wp.min(int(wp.floor((xp[2]+reach)*inv)), z0+dims[2]-1)
    j = int(0)
    for cz in range(z0, z1+1):
        for cy in range(y0, y1+1):
            for cx in range(x0, x1+1):
                k = e*cells+PC._cell_index(cx, cy, cz, dims)
                first = starts[k]
                if first >= 0 and first < n:
                    if keys[first] == k:
                        for i in range(first, ends[k]):
                            if j < T1.K_SEQ:
                                seq[p, j] = ids[i]
                            j += 1
    count[p] = j


def _sequences(kernel, args, n):
    seq = wp.full((n, T1.K_SEQ), -1, dtype=int, device="cpu")
    count = wp.zeros(n, dtype=int, device="cpu")
    wp.launch(kernel, n, args+[seq, count], device="cpu")
    return seq.numpy(), count.numpy()


@pytest.mark.parametrize("dims", [(5, 5, 5), (64, 64, 32), (7, 3, 4)])
def test_env_major_grid_reproduces_warp_grouped_query_sequence(dims):
    """``EnvMajorGrid`` (key-validated buckets, no memset) walked in Warp's cell order gives exactly the candidate
    sequence of a grouped ``wp.HashGrid(*dims)`` query, also after a rebuild on moved points (stale bucket bounds
    of the previous build must never be used); the shrunken reach gives an order-preserving subsequence."""
    rng = np.random.default_rng(5)
    radius = .00045
    h = 2.4*radius
    parts = [rng.uniform(-.03, .05, (1500, 3)), (np.indices((8, 8, 8)).reshape(3, -1).T*h-.004),
             rng.uniform(-.01, .01, (700, 3))]
    env = np.repeat(np.arange(3, dtype=np.int32), [len(p) for p in parts])
    n = len(env)
    env_of_particle = wp.array(env, dtype=int, device="cpu")
    x_first = np.concatenate(parts).astype(np.float32)
    x = wp.array(x_first, dtype=wp.vec3, device="cpu")
    grid_env = PC.EnvMajorGrid(SimpleNamespace(num_envs=3, total_particles=n, device="cpu"), dims, h)
    for trial, x_host in enumerate((x_first, (x_first+rng.normal(scale=.004, size=x_first.shape)).astype(np.float32))):
        x.assign(x_host)
        grid = wp.HashGrid(*dims, device="cpu")
        grid.build(x, h, groups=env_of_particle)
        grid_env.build(SimpleNamespace(total_particles=n, x=x, env_of_particle=env_of_particle))
        warp_seq, warp_count = _sequences(T1._warp_sequence, [grid.id, x, env_of_particle, radius], n)
        common = [grid_env.inv, grid_env.dims, grid_env.cells, n, grid_env.keys, grid_env.starts, grid_env.ends,
                  grid_env.ids]
        frozen_reach = float(np.float32(np.float32(2.4)*np.float32(radius)))
        env_seq, env_count = _sequences(_env_major_sequence, [x, env_of_particle, frozen_reach]+common, n)
        assert np.array_equal(warp_count, env_count), trial
        assert np.array_equal(warp_seq, env_seq), trial
        assert warp_count.max() <= T1.K_SEQ and warp_count.min() >= 1
        slow = PC.shrunken_reach(radius, 1e-4, PC.DEFAULT_SHRINK_SPEED_M_S)
        assert slow < frozen_reach
        sub_seq, sub_count = _sequences(_env_major_sequence, [x, env_of_particle, slow]+common, n)
        assert (sub_count <= env_count).all() and sub_count.sum() < env_count.sum()
        for p in range(0, n, 7):
            full = list(env_seq[p, :env_count[p]])
            it = iter(full)
            assert all(q in it for q in sub_seq[p, :sub_count[p]]), (trial, p)   # ordered subsequence
            xp = x_host[p]
            near = [q for q in full if env[q] == env[p] and np.all(np.abs(x_host[q]-xp) <= .9*slow)]
            assert set(near) <= set(sub_seq[p, :sub_count[p]]), (trial, p)


# ----------------------------------------------------------------------------- block scenes (bare solver)

@pytest.mark.cpu_sim
@pytest.mark.parametrize("reference_dims", [PC.FROZEN_HASH_DIMS, T1.HEAVY], ids=["frozen_dims", "heavy_alias_5x5x5"])
def test_block_scenes_v2_bitwise(reference_dims):
    """Compressed heterogeneous blocks with conflicting codes: v2 (shrink on/off) bitwise equal to the frozen
    single after every substep; the shrink is active (all envs slow) and contacts are non-zero."""
    configs = [_config(o, reference_dims) for o in VARIANTS.values()]
    singles, batches = T1._block_scene(configs, reference_dims=None if reference_dims == PC.FROZEN_HASH_DIMS
                                       else reference_dims)
    mismatches, seen = T1._run_blocks(singles, batches)
    for substep, rows in enumerate(mismatches):
        for name, row in zip(VARIANTS, rows):
            assert row == [[], [], []], (substep, name, row)
    stats = np.stack([s.stats.numpy() for s in singles])
    assert (stats[1:, 4] > 0).all(), ("particle-particle contacts", stats[:, 4])
    assert seen["max_contributions"] >= 2, seen
    shrink, plain = batches
    r = shrink.particle_contact_receipt()
    assert r["particle_contact_mode"] == V2 and r["query_reach_slow_m"] < r["query_reach_frozen_m"]
    assert r["guard_violation_substeps"] == [0, 0, 0]
    assert plain.particle_contact_receipt()["query_reach_slow_m"] is None
    assert shrink.particle_contact.fast_flag.numpy().tolist() == [0, 0, 0]


@pytest.mark.cpu_sim
def test_shrink_speed_check_is_necessary_and_exact():
    """Env 1 gets particles at 20 m/s (above the shrink speed, below the certified alias bound): v2 uses the
    frozen reach there and stays bitwise for every env; with the speed check disabled (shrunken reach kept) env 1
    differs, so the check is what keeps the result exact, while the slow envs 0 and 2 stay bitwise."""
    base = _config({}, PC.FROZEN_HASH_DIMS)
    singles, batches = T1._block_scene([base, base], boost=(1, 40, 20.))
    unchecked = batches[1].particle_contact
    assert unchecked.speed_bound_m_s > 20.
    unchecked.config = replace(unchecked.config, shrink_speed_m_s=1.0e3)   # check never fires, reach stays slow
    mismatches, _ = T1._run_blocks(singles, batches, substeps=12, probe_every=False)
    assert all(row[0] == [[], [], []] for row in mismatches), [row[0] for row in mismatches]
    assert all(row[1][0] == [] and row[1][2] == [] for row in mismatches)
    assert any(row[1][1] != [] for row in mismatches), "the shrink without its speed check must differ"
    for batch in batches:
        assert batch.contact_guard_violations.numpy().tolist() == [0, 0, 0]
        assert batch.particle_contact.fast_flag.numpy().tolist() == [0, 0, 0]


# ----------------------------------------------------------------------------- TRIM scenes

@pytest.mark.cpu_sim
def test_full_resolution_trim_substep_bitwise_v2():
    """One substep of a full-resolution TRIM env (1 mm, ~1e5 particles; the frozen 64x64x32 grid aliases)."""
    pose = start_pose(512, spacing=.001)
    single = trim_material(512, spacing=.001)
    single.reset(pose, seed=42)
    packed = layout.pack([layout.EnvSpec(trim_material(512, spacing=.001), pose, 42)])
    solver = BatchedTLMPMv7_2(packed, "cpu", particle_contact=_config({}, PC.FROZEN_HASH_DIMS))
    moved = pose.copy()
    moved[0] += .001
    poses, twists = host_inputs([single], [pose], [moved], solver.substeps)
    bed = single.bed
    bed.poses.assign(poses)
    bed.twists.assign(twists)
    bed.cuts.assign(np.zeros(solver.trajectory_length, np.int32))
    for array in (bed.index, bed.reaction, bed.box_reactions, bed.stats):
        array.zero_()
    bed.step()
    solver.poses.assign(poses)
    solver.twists.assign(twists)
    solver.cuts.assign(np.zeros(solver.trajectory_length, np.int32))
    solver.zero_control_step_accumulators()
    solver.step()
    assert compare_env(solver, 0, bed) == []
    assert bed.stats.numpy()[1] > 0, "the substep has blade contacts"


@pytest.mark.cpu_sim
def test_n3_heterogeneous_trim_sequence_bitwise_v2():
    """ADR 0021 L0 heterogeneous N=3 cutting sequence (digits 0, 1) in alias_free_v2, every array after every
    control step against the frozen singles (grids forced to alias at 8x8x8, as in the v1 test)."""
    configs = [(512, None), (15, {"young_modulus_pa": 8e4, "friction": .4}),
               (7, {"yield_stress_pa": 1500., "fracture_energy_j_m2": .08})]
    solver, singles, starts = T1._trim_batch_and_singles(configs, _config({}, (8, 8, 8)))
    for single in singles:
        single.bed.hash = wp.HashGrid(8, 8, 8, device="cpu")
    current = [p.copy() for p in starts]
    per_env_dx = (.001, .0012, .0008)
    for step, (_, cut) in enumerate(SEQUENCE[:4]):
        last = [p.copy() for p in current]
        for e in range(3):
            current[e] = current[e].copy()
            current[e][0] += per_env_dx[e]
            advance_single(singles[e], current[e], step, cut)
        poses, twists = host_inputs(singles, last, current, solver.substeps)
        drive_solver(solver, poses, twists, [cut]*3)
        for e, single in enumerate(singles):
            assert compare_env(solver, e, single.bed) == [], (step, e)
    assert all(np.count_nonzero(s.bed.code.numpy()) > 0 for s in singles)
    assert solver.contact_guard_violations.numpy().tolist() == [0, 0, 0]


# ----------------------------------------------------------------------------- material

@pytest.mark.cpu_sim
def test_partial_reset_v2_equals_default_mode():
    """Partial reset in alias_free_v2: every env state equals frozen_v1 after every control step, before and
    after reset([1])."""
    starts = [start_pose(seed) for seed, _, _ in T1.CONFIGS]
    v2 = batch_material(T1.CONFIGS, particle_contact_mode=V2)
    default = batch_material(T1.CONFIGS)
    offsets, restart = np.zeros(3), np.zeros(3)
    dx, cuts = (.001, .001, .001, .001), (0, 0, 1, 0)
    for step in range(4):
        offsets += dx[step]
        poses = T1._poses(starts, offsets)
        if step >= 2:
            restart += dx[step-2]
            poses[1] = T1._poses(starts, restart)[1]
        for material in (v2, default):
            step_material(material, poses, [cuts[step]]*3)
        for e in range(3):
            assert compare_snapshots(env_state(v2, e), env_state(default, e)) == [], (step, e)
        if step == 1:
            for material in (v2, default):
                material.reset([1], seeds=[99])
    assert v2.invalid_reason.numpy().tolist() == [0, 0, 0]


def test_reset_coverage_v2_is_vars_complete():
    material = batch_material(T1.CONFIGS[:1], particle_contact_mode=V2)
    names = set()
    for prefix, owner in (("", material), ("solver.", material.solver), ("diagnostics.", material.diagnostics)):
        names |= {prefix+n for n, v in vars(owner).items() if isinstance(v, wp.array)}
    solver = (set(MB.SOLVER_RESET_ARRAYS) | set(MB.SOLVER_CONSTANT_ARRAYS) | set(MB.SOLVER_SHARED_ARRAYS)
              | set(MB.CONTACT_GUARD_RESET_ARRAYS) | set(MB.CONTACT_GUARD_SCRATCH))
    material_names = (set(MB.MATERIAL_RESET_ARRAYS) | set(MB.MATERIAL_CONSTANT_ARRAYS)
                      | set(MB.RECOMPUTED_EVERY_CONTROL_STEP) | set(MB.RESET_INPUTS))
    classified = ({f"solver.{n}" for n in solver} | material_names
                  | {f"diagnostics.{n}" for n in MB.DIAGNOSTICS_SCRATCH})
    assert names == classified, names ^ classified


def test_v2_receipt_resolve_and_v1_unchanged():
    v1 = PC.resolve(PC.ALIAS_FREE_MODE)
    assert v1.env_major_grid is False and v1.shrink_speed_m_s is None
    m1 = batch_material(T1.CONFIGS[:1], particle_contact_mode=PC.ALIAS_FREE_MODE)
    r1 = m1.receipt()["particle_contact"]
    assert not {"env_major_grid", "shrink_speed_m_s", "query_reach_slow_m"} & set(r1)
    assert m1.solver.particle_contact.env_grid is None
    full = PC.resolve(V2)
    assert (full.skip_zero_impulse_atomics, full.alias_free_hash_dims, full.speed_guard) == (True, True, "fallback")
    assert full.env_major_grid and full.shrink_speed_m_s == PC.DEFAULT_SHRINK_SPEED_M_S
    assert PC.resolve(V2, {"shrink_speed_m_s": 0}).shrink_speed_m_s is None
    for options in ({"env_major_grid": False}, {"speed_guard": "off"}, {"speed_guard": "flag"},
                    {"skip_zero_impulse_atomics": False}, {"alias_free_hash_dims": False}, {"typo": 1}):
        with pytest.raises(UnsupportedBatchConfiguration) as info:
            PC.resolve(V2, options)
        assert info.value.code == "UNSUPPORTED_PARTICLE_CONTACT_MODE"
    with pytest.raises(ValueError):
        PC.resolve(V2, {"shrink_speed_m_s": -1.})
    m2 = batch_material(T1.CONFIGS, particle_contact_mode=V2)
    r2 = m2.receipt()["particle_contact"]
    assert r2["particle_contact_mode"] == V2 and r2["env_major_grid_bytes"] > 0
    s = m2.solver
    assert r2["query_reach_slow_m"] == PC.shrunken_reach(s.radius, s.dt, PC.DEFAULT_SHRINK_SPEED_M_S)
    assert 2*s.radius < r2["query_reach_slow_m"] < r2["query_reach_frozen_m"]
    assert s.hash is s.particle_contact.grid
    # The shrunken reach covers every pair that can have a non-zero impulse while both speeds are <= V.
    V = PC.DEFAULT_SHRINK_SPEED_M_S
    assert r2["query_reach_slow_m"] >= 2*s.radius+2*s.dt*V
