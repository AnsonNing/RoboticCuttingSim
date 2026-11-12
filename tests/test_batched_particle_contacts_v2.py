"""P1-6 (cycle p16c1): opt-in ``alias_free_v2`` particle contacts of the batched v7_2 solver (Warp CPU, bitwise).

``alias_free_v2`` = ``alias_free_v1`` (C1 + C2 + fallback guard) with the contact query walking env-major buckets
(``EnvMajorGrid``: key ``env*C + cell``, no per-cell group binary search) and, by default, a per-env query shrink
to ``2r + 2 dt V`` while every particle of the env is slower than ``V = shrink_speed_m_s``. Every scene compares
with the frozen single solver after every substep (``np.array_equal`` on every array, as the v1 tests do) and
asserts non-vacuity: contacts are non-zero, some particle receives >= 2 non-zero contributions, and the speed
check is necessary (disabling the check while keeping the shrunken reach changes the result of the fast env).

Where the shrink is engaged is read from the solver's own per-env counter (``shrink_engaged_substeps``, written by
the contact kernel when it selects the shrunken reach). In the compressed block scenes it is engaged only in the
first substep (later substeps are faster than V) and the TRIM scenes have no particle-particle contacts (PR #14
review finding 1); ``test_shrink_engaged_under_slow_contacts_with_fast_slow_transitions`` is the scene where
the shrink is engaged under real particle-particle contacts for many substeps, including a fast -> slow
transition of one env.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

import test_batched_particle_contacts as T1
import test_batched_paths_parity as PP
from batched_fixtures import (SEQUENCE, advance_single, batch_material, block_bed, compare_env, compare_snapshots,
                              drive_solver, env_state, host_inputs, start_pose, step_material, trim_material)
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
    single after every substep, with non-zero contacts. These scenes test the env-major regrouping; the shrink is
    engaged only while every particle of an env is slower than V, here only in the first substep (the compressed
    lattices and the table plane accelerate particles to metres per second), so the shrink under contact is
    covered by ``test_shrink_engaged_under_slow_contacts_with_fast_slow_transitions``."""
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
    engaged = r["shrink_engaged_substeps"]
    assert 1 <= sum(engaged) < 3*PP.SUBSTEPS, engaged     # engaged, but not in every substep (see the docstring)
    assert plain.particle_contact_receipt()["shrink_engaged_substeps"] == [0, 0, 0]


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
    # The checked batch never queries env 1 with the shrunken reach; the unchecked one always does.
    checked, unchecked_engaged = (b.particle_contact.shrink_engaged.numpy().tolist() for b in batches)
    assert checked[1] == 0 and unchecked_engaged[1] == 12, (checked, unchecked_engaged)


# ----------------------------------------------------------------------------- shrink under slow contacts

# Small, very soft lattice blocks (Pa), lifted 6 mm above the table plane and the fixture, the tool parked 0.3 m
# away: nothing but the blocks' own stress, gravity and the particle contacts moves them, so every particle stays
# far below V = 0.25 m/s. The lattice is compressed to 0.88 (spacing 1.76 mm < 2r = 1.8 mm) and the codes
# alternate per x layer (1, 2, 1, ...), so every x-neighbour pair is a conflicting contact pair; an inward
# velocity field keeps the layers pressed together for the whole scene.
SLOW_BLOCKS = (((.016, .012, .010), dict(young=200.)),
               ((.014, .014, .008), dict(young=300., contact_friction=.45, particle_contact_stiffness=.08)),
               ((.018, .010, .010), dict(young=140.)))
SLOW_SUBSTEPS = 24
BOOST_SUBSTEPS = (6, 9)     # env 1 is boosted before substep 6 and restored to its pre-boost x, v before substep 9


def _slow_contact_scene(contact_configs):
    T = SLOW_SUBSTEPS+1
    singles, beds = [], [[] for _ in contact_configs]
    for size, config in SLOW_BLOCKS:
        for out in [singles]+beds:
            bed = block_bed(size, PP.OBSTACLES, **config)
            bed.set_trajectory(*PP._trajectory((.3, .3), 0., T), np.full(T, -1, np.int32))
            out.append(bed)
    batches = [BatchedTLMPMv7_2(layout.pack_from_beds(b), "cpu", particle_contact=cfg)
               for b, cfg in zip(beds, contact_configs)]
    rng = np.random.default_rng(1)
    states = []
    for single in singles:
        reference = single.reference.numpy()
        center = reference.mean(0)
        xe = (center+.88*(reference-center)+np.array([0., 0., .006])).astype(np.float32)
        ve = (rng.normal(scale=.005, size=reference.shape)-3.*(reference-center)).astype(np.float32)
        layer = np.round((reference[:, 0]-reference[:, 0].min())/.002).astype(int)
        ce = (1+layer % 2).astype(np.int32)
        states.append((xe, ve, ce))
        single.x.assign(xe)
        single.v.assign(ve)
        single.code.assign(ce)
        for array in (single.index, single.reaction, single.box_reactions, single.stats):
            array.zero_()
    for batch in batches:
        for name, k in (("x", 0), ("v", 1), ("code", 2)):
            getattr(batch, name).assign(np.concatenate([s[k] for s in states]))
        for name in ("poses", "twists", "cuts"):
            getattr(batch, name).assign(np.concatenate([getattr(s, name).numpy() for s in singles]))
        batch.zero_control_step_accumulators()
    return singles, batches


def _env_rows(batch, e):
    offsets = batch.packed.particle_offset
    return slice(int(offsets[e]), int(offsets[e+1]))


def _edit_env(obj, rows, x=None, v=None):
    """Write x / v of one env (``rows`` of a batch, None for a single) identically on every side."""
    for name, value in (("x", x), ("v", v)):
        if value is not None:
            array = getattr(obj, name)
            host = array.numpy()
            host[slice(None) if rows is None else rows] = value
            array.assign(host)


@pytest.mark.cpu_sim
def test_shrink_engaged_under_slow_contacts_with_fast_slow_transitions():
    """The query shrink is engaged under real particle-particle contacts for many substeps and stays bitwise.

    Envs 0 and 2 are slow (|v| < V) with non-zero contacts in every substep: the contact kernel must select the
    shrunken reach for them in every substep (its own counter ``shrink_engaged``, read after every substep).
    Env 1 is slow, then 30 of its particles are boosted to 20 m/s for three substeps (fast: the frozen reach),
    then its x and v are restored to the pre-boost values (a teleport; the same edit on the single and on every
    batch), after which it is slow again and must be queried with the shrunken reach again. Every env stays
    bitwise equal to its frozen single after every substep. A control batch whose speed check never fires (the
    shrunken reach kept for env 1 while it is fast) must differ in env 1, so the scene detects a fast-flag
    lookup that reads the wrong entry, a shrink that is skipped after the first substep, or a flag that is not
    cleared after a fast substep."""
    base = _config({}, PC.FROZEN_HASH_DIMS)
    singles, batches = _slow_contact_scene([base, base])
    checked, unchecked = batches
    unchecked.particle_contact.config = replace(unchecked.particle_contact.config, shrink_speed_m_s=1.0e3)
    rows = _env_rows(checked, 1)
    rng = np.random.default_rng(2)
    boosted = rng.choice(rows.stop-rows.start, 30, replace=False)
    direction = rng.normal(size=(30, 3))
    boost = (20.*direction/np.linalg.norm(direction, axis=1, keepdims=True)).astype(np.float32)
    saved = None
    engaged, contacts, speeds, mismatches = [], [], [], []
    previous_engaged = np.zeros((2, 3), np.int64)
    previous_contacts = np.zeros(3)
    for substep in range(SLOW_SUBSTEPS):
        if substep == BOOST_SUBSTEPS[0]:
            saved = singles[1].x.numpy().copy(), singles[1].v.numpy().copy()
            v_boost = saved[1].copy()
            v_boost[boosted] = boost
            for obj, r in ((singles[1], None), (checked, rows), (unchecked, rows)):
                _edit_env(obj, r, v=v_boost)
        elif substep == BOOST_SUBSTEPS[1]:
            for obj, r in ((singles[1], None), (checked, rows), (unchecked, rows)):
                _edit_env(obj, r, x=saved[0], v=saved[1])
        speeds.append([float(np.linalg.norm(s.v.numpy(), axis=1).max()) for s in singles])
        for single in singles:
            single.step()
        for batch in batches:
            batch.step()
        mismatches.append([[compare_env(batch, e, single) for e, single in enumerate(singles)] for batch in batches])
        now = np.array([b.particle_contact.shrink_engaged.numpy() for b in batches], np.int64)
        engaged.append((now-previous_engaged).tolist())
        previous_engaged = now
        total = checked.stats_sum.numpy().reshape(3, -1)[:, 4]
        contacts.append((total-previous_contacts).tolist())
        previous_contacts = total.copy()
    engaged, contacts, speeds = np.array(engaged), np.array(contacts), np.array(speeds)
    fast = np.zeros(SLOW_SUBSTEPS, bool)
    fast[BOOST_SUBSTEPS[0]:BOOST_SUBSTEPS[1]] = True
    # Bitwise: every env of the checked batch after every substep.
    assert all(row[0] == [[], [], []] for row in mismatches), [(i, row[0]) for i, row in enumerate(mismatches)]
    # The scene is what it claims: envs 0, 2 always slower than V, env 1 slower than V outside the boost.
    V = PC.DEFAULT_SHRINK_SPEED_M_S
    assert speeds[:, [0, 2]].max() < V and speeds[~fast, 1].max() < V and speeds[fast, 1].min() > V, speeds
    # Envs 0 and 2: shrunken reach in every substep, with non-zero contacts in every substep.
    assert (engaged[:, 0, [0, 2]] == 1).all(), engaged[:, 0]
    assert (contacts[:, [0, 2]] > 0).all(), contacts
    # Env 1: shrunken reach before and after the boost (fast -> slow transition), frozen reach while fast;
    # contacts while it is queried with the shrunken reach after the transition.
    assert engaged[:, 0, 1].tolist() == (~fast).astype(int).tolist(), engaged[:, 0, 1]
    assert (contacts[BOOST_SUBSTEPS[1]:, 1] > 0).sum() >= 5, contacts[:, 1]
    assert (contacts[:BOOST_SUBSTEPS[0], 1] > 0).all(), contacts[:, 1]
    # Control: without the speed check env 1 keeps the shrunken reach while fast, and the result differs there
    # (the boosted substeps miss contacts), while the slow envs 0 and 2 stay bitwise.
    assert (engaged[:, 1, 1] == 1).all()
    assert all(row[1][0] == [] and row[1][2] == [] for row in mismatches)
    assert not any(row[1][1] for row in mismatches[:BOOST_SUBSTEPS[0]])
    assert any(row[1][1] for row in mismatches[BOOST_SUBSTEPS[0]:BOOST_SUBSTEPS[1]]), "control must differ"
    receipt = checked.particle_contact_receipt()
    assert receipt["shrink_engaged_substeps"] == engaged[:, 0].sum(0).tolist()
    assert receipt["guard_violation_substeps"] == [0, 0, 0]


def test_v2_launch_refuses_radius_or_dt_other_than_construction():
    """``reach_slow`` is derived from the construction radius and dt; a launch with other values (a caller that
    sub-cycles with a larger dt, or a different radius) is refused before anything runs, not silently unsafe."""
    singles, batches = _slow_contact_scene([_config({}, PC.FROZEN_HASH_DIMS)])
    solver = batches[0]
    contacts = solver.particle_contact
    assert (contacts.construction_radius, contacts.construction_dt) == (solver.radius, solver.dt)
    impulse = solver.impulse.numpy().copy()
    for r, dt in ((solver.radius, 2*solver.dt), (1.5*solver.radius, solver.dt), (solver.radius, .5*solver.dt)):
        with pytest.raises(UnsupportedBatchConfiguration) as info:
            contacts.launch(solver, r, dt)
        assert info.value.code == "UNSUPPORTED_PARTICLE_CONTACT_MODE"
        assert "differs from the construction" in str(info.value)
    assert np.array_equal(solver.impulse.numpy(), impulse)
    assert contacts.shrink_engaged.numpy().tolist() == [0, 0, 0]
    contacts.launch(solver, solver.radius, solver.dt)     # the construction values (every current caller)
    assert contacts.shrink_engaged.numpy().tolist() == [1, 1, 1]


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
