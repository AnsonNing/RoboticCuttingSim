"""ADR 0021 §5 L0: the batched v7_2 solver equals the frozen single solver bit for bit on the Warp CPU device.

Every env is compared with its own frozen ``tlmpm_v7_2.TLMPM`` (built and stepped through the frozen
``LiveCuttingMaterial`` path) after every control step, over every Warp array of the single solver
(vars()-complete, mapped back by ``snapshot``) with ``np.array_equal``. Both sides consume identical
per-substep pose/twist arrays (frozen SciPy interpolation), cut digits and obstacle boxes. A mismatch is a
defect to fix, never a reason for a tolerance.
"""
import numpy as np
import pytest
import warp as wp

import test_wp17_frontier as W17
from batched_fixtures import (SEQUENCE, advance_single, compare_env, compare_snapshots, drive_solver,
                              host_inputs, start_pose, trim_material)
from robotic_cutting_sim import tlmpm_v7_2_frontier as Fr
from robotic_cutting_sim.batched import layout
from robotic_cutting_sim.batched import tlmpm_v7_2_batch as TB
from robotic_cutting_sim.batched.tlmpm_v7_2_batch import BatchedTLMPMv7_2

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _batch_and_singles(configs):
    """configs: [(seed, material overrides)] -> (solver, singles, start poses)."""
    singles, specs, starts = [], [], []
    for seed, material in configs:
        pose = start_pose(seed)
        single = trim_material(seed, material=material)
        single.reset(pose, seed=42)
        singles.append(single)
        specs.append(layout.EnvSpec(trim_material(seed, material=material), pose, 42))
        starts.append(pose)
    packed = layout.pack(specs)
    return BatchedTLMPMv7_2(packed, "cpu"), singles, starts


def _run_sequence(solver, singles, starts, sequence, *, per_env_dx=None):
    current = [p.copy() for p in starts]
    labelled = []
    for e, single in enumerate(singles):
        assert compare_env(solver, e, single.bed) == [], f"env {e} differs at pack"
    for step, (dx, cut) in enumerate(sequence):
        last = [p.copy() for p in current]
        for e in range(len(current)):
            current[e] = current[e].copy()
            current[e][0] += dx if per_env_dx is None else per_env_dx[e]
            advance_single(singles[e], current[e], step, cut)
        poses, twists = host_inputs(singles, last, current, solver.substeps)
        drive_solver(solver, poses, twists, [cut]*len(singles))
        for e, single in enumerate(singles):
            bad = compare_env(solver, e, single.bed)
            assert bad == [], f"control step {step} env {e}: arrays differ {bad}"
        labelled.append([int(np.count_nonzero(s.bed.code.numpy())) for s in singles])
    return labelled


@pytest.mark.cpu_sim
def test_n1_trim_cutting_sequence_bitwise():
    solver, singles, starts = _batch_and_singles([(512, None)])
    labelled = _run_sequence(solver, singles, starts, SEQUENCE)
    # The sequence actually cuts (codes appear and grow under digit 0, then digit 1) and exercises the
    # frontier: link candidate events are found.
    assert labelled[0][0] == 0 and labelled[2][0] > labelled[1][0] > 0 and labelled[4][0] >= labelled[3][0]
    assert singles[0].bed.link_diagnostics.numpy()[0] > 0
    digits = singles[0].bed.code.numpy()
    assert np.any(digits & 3) and np.any((digits >> 2) & 3), "both digits 0 and 1 were written"


@pytest.mark.cpu_sim
def test_n3_heterogeneous_bitwise():
    configs = [(512, None), (15, {"young_modulus_pa": 8e4, "friction": .4}),
               (7, {"yield_stress_pa": 1500., "fracture_energy_j_m2": .08})]
    solver, singles, starts = _batch_and_singles(configs)
    counts = solver.packed.particle_counts
    assert len(set(counts.tolist())) == 3 and len(set(solver.packed.node_counts.tolist())) >= 2
    params = solver.params.numpy()
    assert not np.array_equal(params[0], params[1]) and not np.array_equal(params[0], params[2])
    labelled = _run_sequence(solver, singles, starts, SEQUENCE[:4], per_env_dx=(.001, .0012, .0008))
    assert all(v > 0 for v in labelled[-1])
    audit = layout.range_audit(solver.packed, solver.nodes.numpy(), solver.link_offsets.numpy(),
                               solver.link_neighbours.numpy(), solver.pending_q.numpy(),
                               solver.env_of_particle.numpy())
    assert set(audit.values()) == {0}, audit


@pytest.mark.cpu_sim
def test_identical_copies_are_mutually_identical():
    specs = [layout.EnvSpec(trim_material(512), start_pose(512), 42) for _ in range(3)]
    solver = BatchedTLMPMv7_2(layout.pack(specs), "cpu")
    materials = [s.material for s in specs]
    current = [start_pose(512)]*3
    for step, (dx, cut) in enumerate(SEQUENCE[:2]):
        last = current
        current = [p.copy() for p in current]
        for p in current:
            p[0] += dx
        poses, twists = host_inputs(materials, last, current, solver.substeps)
        drive_solver(solver, poses, twists, [cut]*3)
        first = solver.snapshot(0)
        for e in (1, 2):
            assert compare_snapshots(first, solver.snapshot(e)) == [], f"copy {e} differs at step {step}"
    assert np.count_nonzero(solver.snapshot(0)["code"]) > 0


@pytest.mark.cpu_sim
def test_full_size_trim_substep_smoke():
    """One substep of a full-resolution TRIM env (1 mm particles, ~1e5) with blade contact."""
    pose = start_pose(512, spacing=.001)
    single = trim_material(512, spacing=.001)
    single.reset(pose, seed=42)
    packed = layout.pack([layout.EnvSpec(trim_material(512, spacing=.001), pose, 42)])
    assert packed.total_particles > 100_000
    solver = BatchedTLMPMv7_2(packed, "cpu")
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


# ----------------------------------------------------------------------------- WP17 frontier fixtures (kernel level)

_PER_PARTICLE = ("start", "x", "velocity", "mass", "old_codes", "codes", "reached", "nodes", "weights",
                 "gradients", "pending_q", "pending_cut")
_UNIFORM = ("segments", "width", "band", "bevel", "half_thickness", "profile", "profile_count", "radius", "dt")


def _batched_complete_links(datas):
    """Launch the batched frontier kernel over envs built from WP17 fixture dicts (uniform scalars of env 0)."""
    u = datas[0]
    counts = [len(d["x"]) for d in datas]
    node_counts = [len(d["offsets"])-1 for d in datas]
    po = np.r_[0, np.cumsum(counts)]
    no = np.r_[0, np.cumsum(node_counts)]
    lo = np.r_[0, np.cumsum([len(d["neighbours"]) for d in datas])]
    cat = {}
    for name in _PER_PARTICLE:
        parts = []
        for e, d in enumerate(datas):
            value = np.asarray(d[name]).copy()
            if name == "nodes":
                value = value+no[e]
            if name == "pending_q":
                value = np.where(value >= 0, value+po[e], value)
            parts.append(value)
        cat[name] = np.concatenate(parts)
    offsets = np.concatenate([np.asarray(d["offsets"][:-1])+lo[e] for e, d in enumerate(datas)] + [[lo[-1]]])
    neighbours = np.concatenate([np.asarray(d["neighbours"])+po[e] for e, d in enumerate(datas)])
    env = np.repeat(np.arange(len(datas)), counts).astype(np.int32)
    params = np.zeros((len(datas), len(TB.constitutive.PARAM_NAMES)), np.float32)
    params[:, TB.constitutive.PARAM_NAMES.index("contact_friction")] = [d["friction"] for d in datas]
    params[:, TB.constitutive.PARAM_NAMES.index("slice_efficiency")] = [d["slice_efficiency"] for d in datas]
    A = lambda value, dtype: wp.array(np.asarray(value), dtype=dtype, device="cpu")
    vec, i32 = wp.vec3, int
    arrays = dict(start=A(cat["start"], vec), x=A(cat["x"], vec), velocity=A(cat["velocity"], vec),
                  mass=A(cat["mass"], float), old_codes=A(cat["old_codes"], i32), codes=A(cat["codes"], i32),
                  reached=A(cat["reached"], i32), nodes=A(cat["nodes"].astype(np.int32), i32),
                  weights=A(cat["weights"], float), gradients=A(cat["gradients"], vec),
                  offsets=A(offsets.astype(np.int32), i32), neighbours=A(neighbours.astype(np.int32), i32),
                  poses=A(np.concatenate([d["poses"] for d in datas]), wp.transform),
                  previous=A(np.concatenate([d["previous"] for d in datas]), wp.transform),
                  twists=A(np.concatenate([d["twists"] for d in datas]), wp.spatial_vector),
                  cuts=A(np.concatenate([d["cuts"] for d in datas]), i32), index=A(np.zeros(1, np.int32), i32),
                  stroke=A(np.concatenate([d["stroke"] for d in datas]), i32),
                  segment_ok=A(np.concatenate([d["segment_ok"] for d in datas]), i32),
                  profile=A(u["profile"], wp.vec2), params=wp.array(params, dtype=float, device="cpu"),
                  pending_q=A(cat["pending_q"].astype(np.int32), i32),
                  pending_cut=A(cat["pending_cut"].astype(np.int32), i32),
                  diagnostics=A(np.concatenate([d["diagnostics"] for d in datas]), i32),
                  env=A(env, i32))
    readonly = {k: v.numpy().copy() for k, v in arrays.items()
                if k not in ("codes", "pending_q", "pending_cut", "diagnostics")}
    a = arrays
    wp.launch(TB._complete_links_b, len(env), [a["start"], a["x"], a["velocity"], a["mass"], a["old_codes"],
        a["codes"], a["reached"], a["nodes"], a["weights"], a["gradients"], a["offsets"], a["neighbours"],
        a["poses"], a["previous"], a["twists"], a["cuts"], a["index"], a["stroke"], a["segment_ok"],
        u["segments"], u["width"], u["band"], u["bevel"], u["half_thickness"], a["profile"], u["profile_count"],
        u["radius"], u["dt"], a["params"], a["pending_q"], a["pending_cut"], a["diagnostics"], a["env"], 1],
        device="cpu")
    for name, before in readonly.items():
        np.testing.assert_array_equal(arrays[name].numpy(), before, err_msg=name)
    out = []
    for e in range(len(datas)):
        q = arrays["pending_q"].numpy()[po[e]:po[e+1]]
        out.append({"codes": arrays["codes"].numpy()[po[e]:po[e+1]], "pending_q": np.where(q >= 0, q-po[e], q),
                    "pending_cut": arrays["pending_cut"].numpy()[po[e]:po[e+1]],
                    "diagnostics": arrays["diagnostics"].numpy()[2*e:2*e+2]})
    return out


def _with_uniform(data, uniform_source):
    data = dict(data)
    for name in _UNIFORM:
        data[name] = uniform_source[name]
    return data


@pytest.mark.parametrize("case", [row[0] for row in W17._CASES])
def test_wp17_frontier_fixture_batched_bitwise(case):
    """Each WP17 fixture as env 1 of a heterogeneous 3-env batch equals the frozen hoisted kernel."""
    target = W17._fixture(case)
    datas = [_with_uniform(W17._fixture(companion), target) for companion in ("first_worked", "pending_reuse")]
    datas.insert(1, target)
    batched = _batched_complete_links(datas)
    for e, data in enumerate(datas):
        reference = W17._launch(Fr._complete_links, W17._arguments(data))
        for name in W17._OUTPUTS:
            np.testing.assert_array_equal(batched[e][name], reference[name], err_msg=f"{case} env {e} {name}")


@pytest.mark.parametrize("case", ["no_contact", "pending_disarmed", "pending_large_rotation",
                                  "pending_outside_edge", "pending_band_exit", "pending_side_exit",
                                  "pending_bevel_exit", "pending_lost_interface"])
def test_wp17_pending_lifecycle_batched_bitwise(case):
    """Two launches with retained pending evidence, as test_wp17_frontier._pending_lifecycle, batched."""
    first = W17._fixture(case)
    datas = [_with_uniform(W17._fixture("contact"), first), first, _with_uniform(W17._fixture("no_contact"), first)]
    phase1 = _batched_complete_links(datas)
    restored = W17._fixture("contact")
    restored["previous"] = restored["poses"].copy()
    second = []
    for e, data in enumerate(datas):
        data = dict(data)
        for name in ("start", "x", "previous", "poses", "twists", "cuts", "reached", "segment_ok"):
            data[name] = restored[name]
        for name in W17._OUTPUTS:
            data[name] = phase1[e][name].copy()
        second.append(data)
    for data in second:
        data["band"] = restored["band"]
    phase2 = _batched_complete_links(second)
    # Reference: the frozen kernel with the same two launches per env.
    for e, (data1, data2) in enumerate(zip(datas, second)):
        arguments = W17._arguments(data1)
        ref1 = W17._launch(Fr._complete_links, arguments)
        for name in ("start", "x", "previous", "poses", "twists", "cuts", "reached", "segment_ok"):
            arguments[W17._NAMES.index(name)].assign(data2[name])
        arguments[W17._NAMES.index("band")] = data2["band"]
        ref2 = W17._launch(Fr._complete_links, arguments)
        for name in W17._OUTPUTS:
            np.testing.assert_array_equal(phase1[e][name], ref1[name], err_msg=f"{case} env {e} launch 1 {name}")
            np.testing.assert_array_equal(phase2[e][name], ref2[name], err_msg=f"{case} env {e} launch 2 {name}")
    retained = case == "no_contact"
    assert phase2[1]["codes"].tolist() == ([1, 0] if retained else [0, 0])
