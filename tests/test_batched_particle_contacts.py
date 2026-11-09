"""P1-3a: opt-in ``alias_free_v1`` particle contacts of the batched v7_2 solver (Warp CPU device, bitwise).

Every scene compares the opt-in mode with the frozen single solver (ADR 0021 §5 L0: every Warp array, after
every substep or control step, ``np.array_equal``) and, where cheap, with the default ``frozen_v1`` batch. The
sub-flags are exercised independently: C1 (skip exactly-zero impulse atomics), C2 (alias-free hash dims) with
the speed guard in ``fallback`` / ``flag`` / ``off``.

Forced heavy aliasing: the frozen single's own ``hash`` attribute is replaced by a ``HashGrid(5, 5, 5)`` (an
instance attribute of the test object; the frozen kernel and module are untouched), so its query sees many
aliased candidates; the alias-free batch is configured with ``reference_hash_dims=(5, 5, 5)`` so its guard
certifies (and its fallback reproduces) that reference. Non-vacuity is asserted: aliased candidates exist,
contacts are non-zero, some particle receives >= 2 non-zero contributions in one substep, the guard fallback
runs and is necessary (the unguarded variant differs once the certificate is violated).
"""
import hashlib
import inspect

import numpy as np
import pytest
import warp as wp

import test_batched_paths_parity as PP
from batched_fixtures import (SEQUENCE, advance_single, batch_material, block_bed, compare_env, compare_snapshots,
                              drive_solver, env_state, host_inputs, start_pose, step_material, trim_material)
from robotic_cutting_sim import tlmpm_v5 as v5
from robotic_cutting_sim.batched import layout
from robotic_cutting_sim.batched import material_batch as MB
from robotic_cutting_sim.batched import particle_contacts as PC
from robotic_cutting_sim.batched import tlmpm_v7_2_batch as TB
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.batched.tlmpm_v7_2_batch import BatchedTLMPMv7_2

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]

AF = PC.ALIAS_FREE_MODE
HEAVY = (5, 5, 5)
VARIANTS = {   # name -> alias_free_v1 options (reference_hash_dims is added per scene)
    "c1": {"alias_free_hash_dims": False},
    "c2_fallback": {"skip_zero_impulse_atomics": False},
    "c1_c2_fallback": {},
    "c1_c2_flag": {"speed_guard": "flag"},
    "c1_c2_unguarded": {"speed_guard": "off"},
}
# Source of the default kernel at the P1-3a base commit 4fc431e (the rollback path must stay byte-identical).
FROZEN_BATCH_KERNEL_SHA256 = "10c6c55043827b8b264ee9e855874bb5cfd295a3282bf390dac5ae37ca23a4aa"


def _config(options, reference_dims):
    return PC.resolve(AF, dict(options, reference_hash_dims=reference_dims))


# ----------------------------------------------------------------------------- non-vacuity probe (test only)

@wp.kernel
def _probe(grid: wp.uint64, x: wp.array(dtype=wp.vec3), reference: wp.array(dtype=wp.vec3),
           v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), code: wp.array(dtype=int), radius: float,
           exclusion: float, dt: float, stiffness: float, friction: float, inv: float,
           contributions: wp.array(dtype=int), aliased: wp.array(dtype=int), aliased_nonzero: wp.array(dtype=int)):
    """The frozen pair loop of tlmpm_v5._particle_contacts, counting instead of applying: non-zero contributions
    per particle, aliased candidates (virtual cell outside the visited box, Warp's float32 cell arithmetic) and
    aliased pairs with a non-zero impulse."""
    p = wp.tid()
    xp = x[p]
    reach = 2.4*radius
    query = wp.hash_grid_query(grid, xp, reach)
    for q in query:
        xq = x[q]
        alias = False
        for k in range(3):
            cell = int(wp.floor(xq[k]*inv))
            if cell < int(wp.floor((xp[k]-reach)*inv)) or cell > int(wp.floor((xp[k]+reach)*inv)):
                alias = True
        if alias:
            wp.atomic_add(aliased, 0, 1)
        if q > p:
            if v5._conflict_packed(code[p], code[q]) or wp.length_sq(reference[p]-reference[q]) > exclusion**2.0:
                delta = xp-xq
                distance = wp.length(delta)
                if distance > 1.0e-10:
                    effective = mass[p]*mass[q]/(mass[p]+mass[q])
                    ji = v5._contact_impulse(distance-2.0*radius, delta/distance, v[p]-v[q], effective, dt,
                                             stiffness, friction)
                    if ji[0] != 0.0 or ji[1] != 0.0 or ji[2] != 0.0:
                        wp.atomic_add(contributions, p, 1)
                        wp.atomic_add(contributions, q, 1)
                        if alias:
                            wp.atomic_add(aliased_nonzero, 0, 1)


def probe(bed):
    """(max non-zero contributions of one particle, aliased candidates, aliased non-zero pairs) of the frozen
    single's next contact stage, on its own grid (dims as configured on the instance)."""
    c, n = bed.config, bed.n
    h = 2.4*bed.radius
    bed.hash.build(bed.x, h)
    counts = wp.zeros(n, dtype=int, device="cpu")
    aliased, nonzero = wp.zeros(1, dtype=int, device="cpu"), wp.zeros(1, dtype=int, device="cpu")
    wp.launch(_probe, n, [bed.hash.id, bed.x, bed.reference, bed.v, bed.mass, bed.code, bed.radius, 3*c.dx, c.dt,
                          c.particle_contact_stiffness, c.contact_friction, float(np.float32(1.0)/np.float32(h)),
                          counts, aliased, nonzero], device="cpu")
    return int(counts.numpy().max()), int(aliased.numpy()[0]), int(nonzero.numpy()[0])


# ----------------------------------------------------------------------------- block scenes (bare solver)

def _block_scene(contact_configs, reference_dims=None, boost=None):
    """PP.BLOCKS (three heterogeneous lattice blocks, compressed lattices with conflicting codes, handle and pads,
    per-env digits) as frozen singles plus one bare batch per contact config (None = frozen_v1).
    ``reference_dims`` replaces the singles' grid (forced aliasing); ``boost`` = (env, count, speed) gives
    ``count`` particles of that env a large velocity (speed-guard scenes)."""
    T = PP.SUBSTEPS+1
    singles, beds = [], [[] for _ in contact_configs]
    for size, config, *_ in PP.BLOCKS:
        bed = block_bed(size, PP.OBSTACLES, **config)
        bed.set_trajectory(*PP._trajectory((0., 0.), 0., T), np.full(T, -1, np.int32))
        if reference_dims is not None:
            bed.hash = wp.HashGrid(*reference_dims, device="cpu")
        singles.append(bed)
        for out in beds:
            b = block_bed(size, PP.OBSTACLES, **config)
            b.set_trajectory(*PP._trajectory((0., 0.), 0., T), np.full(T, -1, np.int32))
            out.append(b)
    batches = [BatchedTLMPMv7_2(layout.pack_from_beds(b), "cpu", particle_contact=cfg)
               for b, cfg in zip(beds, contact_configs)]
    rng = np.random.default_rng(0)
    states = []
    for e, (single, (_, _, compression, digit, path, yaw)) in enumerate(zip(singles, PP.BLOCKS)):
        reference = single.reference.numpy()
        center = reference.mean(0)
        xe = (center+compression*(reference-center)).astype(np.float32)
        ve = rng.normal(scale=.01, size=reference.shape).astype(np.float32)
        if boost is not None and boost[0] == e:
            chosen = rng.choice(len(ve), boost[1], replace=False)
            direction = rng.normal(size=(boost[1], 3))
            ve[chosen] = (boost[2]*direction/np.linalg.norm(direction, axis=1, keepdims=True)).astype(np.float32)
        ce = (np.where(reference[:, 0] < center[0], 1, 2) if compression < 1
              else np.zeros(len(reference))).astype(np.int32)
        p, w = PP._trajectory(path, yaw, T)
        states.append((xe, ve, ce, p, w, np.full(T, digit, np.int32)))
        single.x.assign(xe)
        single.v.assign(ve)
        single.code.assign(ce)
        single.poses.assign(p)
        single.twists.assign(w)
        single.cuts.assign(np.full(T, digit, np.int32))
        for array in (single.index, single.reaction, single.box_reactions, single.stats):
            array.zero_()
    for batch in batches:
        for name, k in (("x", 0), ("v", 1), ("code", 2)):
            getattr(batch, name).assign(np.concatenate([s[k] for s in states]))
        for name, k in (("poses", 3), ("twists", 4), ("cuts", 5)):
            getattr(batch, name).assign(np.concatenate([s[k] for s in states]))
        batch.zero_control_step_accumulators()
    return singles, batches


def _run_blocks(singles, batches, substeps=PP.SUBSTEPS, probe_every=True):
    """Step all; return per-substep (single env -> differing arrays per batch), probe maxima."""
    worst, aliased, aliased_nonzero = 0, 0, 0
    mismatches = []
    for substep in range(substeps):
        if probe_every:
            for single in singles:
                m, a, z = probe(single)
                worst, aliased, aliased_nonzero = max(worst, m), aliased+a, aliased_nonzero+z
        for single in singles:
            single.step()
        for batch in batches:
            batch.step()
        mismatches.append([[compare_env(batch, e, single) for e, single in enumerate(singles)] for batch in batches])
    return mismatches, dict(max_contributions=worst, aliased_candidates=aliased, aliased_nonzero=aliased_nonzero)


@pytest.mark.cpu_sim
@pytest.mark.parametrize("reference_dims", [PC.FROZEN_HASH_DIMS, HEAVY], ids=["frozen_dims", "heavy_alias_5x5x5"])
def test_block_scenes_every_variant_bitwise(reference_dims):
    """Compressed heterogeneous blocks with conflicting codes: >= 2 non-zero contributions per particle, contacts
    in envs 1-2, every variant bitwise equal to the frozen single after every substep (and to frozen_v1)."""
    configs = [None]+[_config(o, reference_dims) for o in VARIANTS.values()]
    singles, batches = _block_scene(configs, reference_dims=None if reference_dims == PC.FROZEN_HASH_DIMS
                                    else reference_dims)
    if reference_dims != PC.FROZEN_HASH_DIMS:
        # The frozen_v1 batch keeps 64x64x32: it is compared with the frozen singles only at frozen dims.
        batches[0].hash = wp.HashGrid(*reference_dims, device="cpu")
        batches[0].hash.reserve(batches[0].total_particles, with_groups=True)
    mismatches, seen = _run_blocks(singles, batches)
    for substep, rows in enumerate(mismatches):
        for name, row in zip(["frozen_v1"]+list(VARIANTS), rows):
            assert row == [[], [], []], (substep, name, row)
    stats = np.stack([s.stats.numpy() for s in singles])
    assert (stats[1:, 4] > 0).all(), ("particle-particle contacts", stats[:, 4])
    assert seen["max_contributions"] >= 2, seen
    if reference_dims == HEAVY:
        assert seen["aliased_candidates"] > 100_000, seen          # heavy aliasing in the reference grid
    else:
        assert seen["aliased_candidates"] == 0, seen               # small blocks never alias at 64x64x32
    for name, batch in zip(VARIANTS, batches[1:]):
        receipt = batch.particle_contact_receipt()
        assert receipt["particle_contact_mode"] == AF
        if receipt["speed_guard"] != "off":
            assert receipt["guard_violation_substeps"] == [0, 0, 0], (name, receipt)
        if receipt["alias_free_hash_dims"]:
            assert all(d >= r for d, r in zip(receipt["hash_dims"], reference_dims))


@pytest.mark.cpu_sim
def test_speed_guard_fallback_reproduces_aliased_reference():
    """A few env-1 particles at 80 m/s: closing speeds make aliased (5x5x5) candidates contribute (the
    certificate is violated, aliased non-zero pairs > 0). ``fallback`` stays bitwise equal for every env and
    counts the substeps only for env 1; ``flag`` (no fallback) counts them too but env 1 differs (so the
    fallback is necessary), while envs 0 and 2 (certified) stay bitwise in both."""
    configs = [_config({}, HEAVY), _config({"speed_guard": "flag"}, HEAVY)]
    singles, batches = _block_scene(configs, reference_dims=HEAVY, boost=(1, 6, 80.))
    bound = batches[0].particle_contact.speed_bound_m_s
    assert 5. < bound < 80., bound
    mismatches, seen = _run_blocks(singles, batches, substeps=12)
    assert seen["aliased_nonzero"] > 0, seen
    fallback = [row[0] for row in mismatches]
    flagged = [row[1] for row in mismatches]
    assert all(r == [[], [], []] for r in fallback), fallback
    assert all(r[0] == [] and r[2] == [] for r in flagged), flagged
    assert any(r[1] != [] for r in flagged), "the unguarded alias-free result must differ once uncertified"
    for batch in batches:
        violations = batch.contact_guard_violations.numpy()
        assert violations[1] > 0 and violations[0] == 0 and violations[2] == 0, violations
        assert batch.contact_guard_flag.numpy().tolist() == [0, 0, 0]
        assert batch.contact_guard_any.numpy().tolist() == [0]


# ----------------------------------------------------------------------------- TRIM scenes

@pytest.mark.cpu_sim
@pytest.mark.parametrize("variant", ["c1_c2_fallback", "c1"])
def test_full_resolution_trim_substep_bitwise(variant):
    """One substep of a full-resolution TRIM env (1 mm, ~1e5 particles; 64x64x32 aliases ~2.3x in x)."""
    pose = start_pose(512, spacing=.001)
    single = trim_material(512, spacing=.001)
    single.reset(pose, seed=42)
    packed = layout.pack([layout.EnvSpec(trim_material(512, spacing=.001), pose, 42)])
    assert packed.total_particles > 100_000
    solver = BatchedTLMPMv7_2(packed, "cpu", particle_contact=_config(VARIANTS[variant], PC.FROZEN_HASH_DIMS))
    receipt = solver.particle_contact_receipt()
    span = receipt["env_cell_spans"]
    if variant == "c1":
        assert receipt["hash_dims"] == [64, 64, 32] and span is None
    else:
        assert receipt["hash_dims"][0] > 64 and receipt["hash_dims"][0] >= span[0][0]+3
    moved = pose.copy()
    moved[0] += .001
    poses, twists = host_inputs([single], [pose], [moved], solver.substeps)
    bed = single.bed
    bed.poses.assign(poses)
    bed.twists.assign(twists)
    bed.cuts.assign(np.zeros(solver.trajectory_length, np.int32))
    for array in (bed.index, bed.reaction, bed.box_reactions, bed.stats):
        array.zero_()
    _, aliased, _ = probe(bed)
    assert aliased > 1_000_000, aliased
    bed.step()
    solver.poses.assign(poses)
    solver.twists.assign(twists)
    solver.cuts.assign(np.zeros(solver.trajectory_length, np.int32))
    solver.zero_control_step_accumulators()
    solver.step()
    assert compare_env(solver, 0, bed) == []
    assert bed.stats.numpy()[1] > 0, "the substep has blade contacts"


def _trim_batch_and_singles(configs, particle_contact):
    singles, specs, starts = [], [], []
    for seed, material in configs:
        pose = start_pose(seed)
        single = trim_material(seed, material=material)
        single.reset(pose, seed=42)
        singles.append(single)
        specs.append(layout.EnvSpec(trim_material(seed, material=material), pose, 42))
        starts.append(pose)
    return BatchedTLMPMv7_2(layout.pack(specs), "cpu", particle_contact=particle_contact), singles, starts


@pytest.mark.cpu_sim
def test_n3_heterogeneous_trim_sequence_bitwise():
    """ADR 0021 L0 heterogeneous N=3 cutting sequence (digits 0, 1) in alias_free_v1, every array after every
    control step against the frozen singles, whose grids are forced to alias (8x8x8: 21.6 mm period)."""
    configs = [(512, None), (15, {"young_modulus_pa": 8e4, "friction": .4}),
               (7, {"yield_stress_pa": 1500., "fracture_energy_j_m2": .08})]
    solver, singles, starts = _trim_batch_and_singles(configs, _config({}, (8, 8, 8)))
    for single in singles:
        single.bed.hash = wp.HashGrid(8, 8, 8, device="cpu")     # forced aliasing in the frozen reference
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
    assert solver.particle_contact_receipt()["hash_dims"] != [64, 64, 32]


# ----------------------------------------------------------------------------- material: guard bit, partial reset

CONFIGS = [(512, None, 42), (15, {"young_modulus_pa": 8e4}, 43), (7, None, 44)]


def _poses(starts, offsets):
    out = []
    for start, offset in zip(starts, offsets):
        pose = start.copy()
        pose[0] += offset
        out.append(pose)
    return np.stack(out)


@pytest.mark.cpu_sim
def test_partial_reset_alias_free_equals_default_mode():
    """Partial reset in alias_free_v1: every env state (solver + material rows) equals frozen_v1 after every
    control step, before and after reset([1]); the guard arrays of the reset env are cleared."""
    starts = [start_pose(seed) for seed, _, _ in CONFIGS]
    af = batch_material(CONFIGS, particle_contact_mode=AF)
    default = batch_material(CONFIGS)
    offsets, restart = np.zeros(3), np.zeros(3)
    dx, cuts = (.001, .001, .001, .001), (0, 0, 1, 0)
    for step in range(4):
        offsets += dx[step]
        poses = _poses(starts, offsets)
        if step >= 2:
            restart += dx[step-2]
            poses[1] = _poses(starts, restart)[1]
        for material in (af, default):
            step_material(material, poses, [cuts[step]]*3)
        for e in range(3):
            assert compare_snapshots(env_state(af, e), env_state(default, e)) == [], (step, e)
        if step == 1:
            af.solver.contact_guard_violations.assign(np.array([0, 5, 0], np.int32))   # stale count on env 1
            for material in (af, default):
                material.reset([1], seeds=[99])
            assert af.solver.contact_guard_violations.numpy().tolist() == [0, 0, 0]
            for e in range(3):
                assert compare_snapshots(env_state(af, e), env_state(default, e)) == [], ("reset", e)
    assert af.invalid_reason.numpy().tolist() == [0, 0, 0]


def test_reset_coverage_alias_free_is_vars_complete():
    material = batch_material(CONFIGS[:1], particle_contact_mode=AF)
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


@pytest.mark.cpu_sim
def test_material_guard_sets_sticky_invalid_bit_and_reset_clears_it():
    """A lowered guard bound (0.3 m/s) with an injected 1 m/s particle in env 1: env 1 gets the sticky
    CONTACT_SPEED_GUARD bit (fallback ran; the state still equals frozen_v1), env 0 stays valid; strict mode
    raises with the reason; reset([1]) clears count and bit."""
    starts = [start_pose(seed) for seed, _, _ in CONFIGS[:2]]
    options = {"speed_bound_m_s": .3}
    af = batch_material(CONFIGS[:2], particle_contact_mode=AF, particle_contact_options=options, strict=False)
    default = batch_material(CONFIGS[:2])
    p0 = int(af.packed.particle_offset[1])
    for material in (af, default):
        v = material.solver.v.numpy()
        v[p0+5] = (1., 0., 0.)
        material.solver.v.assign(v)
    poses = _poses(starts, np.full(2, .001))
    for material in (af, default):
        step_material(material, poses, [0, 0])
    assert compare_snapshots(env_state(af, 0), env_state(default, 0)) == []
    assert compare_snapshots(env_state(af, 1), env_state(default, 1)) == ["material.invalid_reason"]
    reasons = af.invalid_reason.numpy()
    assert reasons[0] == 0 and reasons[1] == MB.INVALID_CONTACT_SPEED_GUARD, reasons
    assert MB.decode_reason(reasons[1]) == ["CONTACT_SPEED_GUARD"]
    violations = af.solver.contact_guard_violations.numpy()
    assert violations[0] == 0 and violations[1] >= 1, violations
    assert af.receipt()["particle_contact"]["guard_violation_substeps"] == violations.tolist()
    af.strict = True
    with pytest.raises(MB.InvalidBatchEnvs, match="CONTACT_SPEED_GUARD"):
        af.check_valid()
    af.reset([1])
    assert af.invalid_reason.numpy().tolist() == [0, 0]
    assert af.solver.contact_guard_violations.numpy().tolist() == [0, 0]


# ----------------------------------------------------------------------------- configuration, layout, receipts

def test_default_mode_is_unchanged_and_receipts_name_the_mode():
    material = batch_material(CONFIGS[:1])
    s = material.solver
    assert s.particle_contact is None and material.particle_contact_mode == "frozen_v1"
    assert not any(n.startswith("contact_guard") for n in vars(s))
    receipt = material.receipt()
    assert receipt["particle_contact_mode"] == "frozen_v1"
    assert receipt["particle_contact"] == {"particle_contact_mode": "frozen_v1", "hash_dims": [64, 64, 32],
                                           "kernel": "_particle_contacts_b", "parity_level": "reference"}
    assert receipt["solver"]["particle_contact"] == receipt["particle_contact"]
    assert "particle_contacts.py" in receipt["batched_source_sha256"]
    assert "particle_contacts.py" in receipt["solver"]["source_sha256"]
    source = inspect.getsource(TB._particle_contacts_b.func)
    assert hashlib.sha256(source.encode()).hexdigest() == FROZEN_BATCH_KERNEL_SHA256


def test_alias_free_receipt_layout_and_dims():
    material = batch_material(CONFIGS, particle_contact_mode=AF)
    s = material.solver
    r = material.receipt()["particle_contact"]
    assert r["particle_contact_mode"] == AF and r["skip_zero_impulse_atomics"] and r["alias_free_hash_dims"]
    assert r["speed_guard"] == "fallback" and r["certificate"] == "checked_every_substep_frozen_fallback"
    assert r["reference_hash_dims"] == [64, 64, 32] and r["fallback_hash_dims"] == [64, 64, 32]
    assert r["margin_m"] == PC.DEFAULT_MARGIN_M and r["guard_speed_bound_m_s"] == r["certified_speed_bound_m_s"]
    h = 2.4*s.radius
    assert r["cell_width_m"] == h and r["alias_distance_m"] == pytest.approx(29*h)
    assert r["certified_speed_bound_m_s"] == pytest.approx(.5*(29*h*.99-2*s.radius)/s.dt)
    # No aliasing inside any env (reference positions, float32 cell arithmetic as Warp's hash_grid_index).
    x = s.reference.numpy()
    inv = np.float32(1.)/np.float32(h)
    cells = np.floor(x*inv).astype(np.int64)
    for e in range(3):
        c = cells[int(material.packed.particle_offset[e]):int(material.packed.particle_offset[e+1])]
        assert np.all(c.max(0)-c.min(0)+4 <= np.array(r["hash_dims"])), (e, r["hash_dims"])
        assert np.all(np.array(r["env_cell_spans"][e]) >= c.max(0)-c.min(0)+1)
    # Warp static-descriptor guard: one device HashGrid per alias-free material, rebuilt before graph launches.
    assert s.hash is s.particle_contact.grid
    assert r["hashgrid_descriptor_guard"] == "single_grid_and_host_rebuild_before_graph_launch_v1"
    s.particle_contact.restore_descriptor(s)
    assert s.contact_guard_flag.shape == (3,) and s.contact_guard_violations.shape == (3,)
    assert s.contact_guard_any.shape == (1,)
    assert material.memory_bytes() > batch_material(CONFIGS).memory_bytes()


def test_resolve_refusals_and_options():
    assert PC.resolve() == PC.ParticleContactConfig()
    full = PC.resolve(AF)
    assert (full.skip_zero_impulse_atomics, full.alias_free_hash_dims, full.speed_guard) == (True, True, "fallback")
    assert PC.resolve(AF, {"alias_free_hash_dims": False}).speed_guard == "off"
    bad = [("frozen_v2", None), ("frozen_v1", {"skip_zero_impulse_atomics": True}),
           (AF, {"alias_free_hash_dims": False, "speed_guard": "fallback"}),
           (AF, {"alias_free_hash_dims": False, "skip_zero_impulse_atomics": False}),
           (AF, {"speed_guard": "sometimes"}), (AF, {"typo": 1})]
    for mode, options in bad:
        with pytest.raises(UnsupportedBatchConfiguration) as info:
            PC.resolve(mode, options)
        assert info.value.code == "UNSUPPORTED_PARTICLE_CONTACT_MODE"
    with pytest.raises(ValueError):
        PC.resolve(AF, {"reference_hash_dims": (2, 64, 32)})
    with pytest.raises(UnsupportedBatchConfiguration):
        batch_material(CONFIGS[:1], particle_contact_mode=AF, particle_contact_options={"speed_bound_m_s": 1e6})
    m = batch_material(CONFIGS[:1], particle_contact_mode=AF, particle_contact_options={"speed_bound_m_s": 1.})
    assert m.solver.particle_contact.speed_bound_m_s == 1.
    m.rebuild([layout.EnvSpec(trim_material(512), start_pose(512), 42)])
    assert m.particle_contact_mode == AF and m.solver.particle_contact.speed_bound_m_s == 1.


# ----------------------------------------------------------------------------- fallback grid vs Warp HashGrid

K_SEQ = 1024


@wp.kernel
def _warp_sequence(grid: wp.uint64, x: wp.array(dtype=wp.vec3), env_of_particle: wp.array(dtype=int),
                   radius: float, seq: wp.array2d(dtype=int), count: wp.array(dtype=int)):
    p = wp.tid()
    j = int(0)
    for q in wp.hash_grid_query(grid, x[p], 2.4*radius, env_of_particle[p]):
        if j < K_SEQ:
            seq[p, j] = q
        j += 1
    count[p] = j


@wp.kernel
def _fallback_sequence(x: wp.array(dtype=wp.vec3), env_of_particle: wp.array(dtype=int), radius: float, inv: float,
                       dims: wp.vec3i, cells: int, starts: wp.array(dtype=int), ends: wp.array(dtype=int),
                       ids: wp.array(dtype=int), seq: wp.array2d(dtype=int), count: wp.array(dtype=int)):
    """The cell walk of ``particle_contacts._particle_contacts_fallback_b`` (copied), recording candidates."""
    p = wp.tid()
    e = env_of_particle[p]
    xp = x[p]
    reach = 2.4*radius
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
                for i in range(starts[k], ends[k]):
                    if j < K_SEQ:
                        seq[p, j] = ids[i]
                    j += 1
    count[p] = j


@pytest.mark.parametrize("dims", [(5, 5, 5), (64, 64, 32), (7, 3, 4)])
def test_fallback_grid_reproduces_warp_grouped_query_sequence(dims):
    """``FallbackGrid`` buckets walked in Warp's cell order give exactly the candidate sequence of a grouped
    ``wp.HashGrid(*dims)`` query (ids and order), for 3 envs with overlapping coordinates, negative coordinates,
    lattice points on cell boundaries and forced aliasing."""
    from types import SimpleNamespace
    rng = np.random.default_rng(3)
    radius = .00045
    h = 2.4*radius
    parts = [rng.uniform(-.03, .05, (1500, 3)), (np.indices((8, 8, 8)).reshape(3, -1).T*h-.004),
             rng.uniform(-.01, .01, (700, 3))]
    x_host = np.concatenate(parts).astype(np.float32)
    env = np.repeat(np.arange(3, dtype=np.int32), [len(p) for p in parts])
    n = len(x_host)
    x = wp.array(x_host, dtype=wp.vec3, device="cpu")
    env_of_particle = wp.array(env, dtype=int, device="cpu")
    grid = wp.HashGrid(*dims, device="cpu")
    grid.build(x, h, groups=env_of_particle)
    fallback = PC.FallbackGrid(SimpleNamespace(num_envs=3, total_particles=n, device="cpu", x=x,
                                               env_of_particle=env_of_particle), dims, h)
    fallback.build(SimpleNamespace(total_particles=n, x=x, env_of_particle=env_of_particle))
    out = []
    for kernel, args in ((_warp_sequence, [grid.id, x, env_of_particle, radius]),
                         (_fallback_sequence, [x, env_of_particle, radius, fallback.inv, fallback.dims, fallback.cells,
                                               fallback.starts, fallback.ends, fallback.ids])):
        seq = wp.full((n, K_SEQ), -1, dtype=int, device="cpu")
        count = wp.zeros(n, dtype=int, device="cpu")
        wp.launch(kernel, n, args+[seq, count], device="cpu")
        out.append((seq.numpy(), count.numpy()))
    assert np.array_equal(out[0][1], out[1][1])
    assert np.array_equal(out[0][0], out[1][0])
    assert out[0][1].max() <= K_SEQ and out[0][1].min() >= 1    # untruncated; every query sees its own particle
