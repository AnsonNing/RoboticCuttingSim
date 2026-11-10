"""R1.7 / RUBBER-PHYS-10 (CPU): batched rubber vs the single-env reference (ADR 0021 L0, L1; ADR 0023 §12).

L0: CPU bitwise equality of every array of every env after every control step (N = 1, N = 3 heterogeneous in
geometry, mu, rho, friction and motion, N = 3 copies), including diagnostics, markers and wrenches.
L1: perturbing one env leaves the others bitwise unchanged; masked partial reset (vars()-complete) equals a fresh env,
also with resampled material parameters; env-ownership audits; refusals.
"""
import numpy as np
import pytest

from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.batched.rubber_layout import GROUP_ARRAYS, NODE_ARRAYS, PARTICLE_ARRAYS, TRACTION_ARRAYS
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_material import MaterialValues

STEP_MATERIAL = MaterialValues(mu0=1e6, kappa=2e7, density=1000.0)


def _live(L=0.012, W=0.003, t=0.003, n_t=2, mu=4e5, rho=1150.0, ft=0.8, traction=True):
    regions = (("tip", (L-0.0015, -1, -1), (1, 1, 1)),) if traction else ()
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=n_t, case="A", traction_regions=regions)
    mat = MaterialValues(mu0=mu, kappa=20*mu, density=rho, friction_table=ft)
    return LiveRubberMaterial(spec, material=mat, control_dt=2e-4, gravity=(0, 0, -9.81),
                              step_material=STEP_MATERIAL, table_plane=(0.0, 0.0, 1.0, 0.0))


HETERO = [dict(), dict(L=0.009, W=0.006, mu=7e5, rho=1050.0, ft=0.4), dict(L=0.015, mu=3e5, rho=1200.0, ft=1.2)]


def _inputs(k, N):
    tr = np.zeros((N, 1, 3))
    for e in range(N):
        tr[e, 0] = [0.0, 0.0, 2e-5*(k+1)*(e+1)]
    force = np.zeros((N, 3))
    force[:, 0] = 1e-3*np.arange(1, N+1)
    return tr, force


def _compare(batch, e, live):
    snap = batch.solver.snapshot(e)
    s = live.solver
    G, Gt = s.G, s.Gt
    for name in PARTICLE_ARRAYS+NODE_ARRAYS+("cell_node", "cell_offsets", "cell_particles", "index"):
        assert np.array_equal(snap[name], getattr(s, name).numpy()), name
    for name in GROUP_ARRAYS:
        assert np.array_equal(snap[name], getattr(s, name).numpy()[:G]), name
    for name in TRACTION_ARRAYS:
        assert np.array_equal(snap[name], getattr(s, name).numpy()[:Gt]), name
    assert np.array_equal(snap["work"], s.work.numpy()) and np.array_equal(snap["violations"], s.violations.numpy())
    assert np.array_equal(batch.summary.numpy()[e], s.diagnostics.summary.numpy()[0]), "diagnostics"
    K = batch.K
    assert np.array_equal(batch.marker_x.numpy()[e*K:(e+1)*K], s.diagnostics.marker_x.numpy()), "markers"
    assert np.array_equal(batch.wrench.numpy()[e, :G], live.last_wrench), "wrench"


def _run_l0(kwargs_list, steps=3):
    singles = [_live(**kw) for kw in kwargs_list]
    batch = rmb.BatchedRubberMaterial([_live(**kw) for kw in kwargs_list], device="cpu", parity_mode=True)
    N = len(kwargs_list)
    for k in range(steps):
        tr, force = _inputs(k, N)
        batch.advance(translation=tr, traction=force)
        for e, live in enumerate(singles):
            live.advance({0: {"translation": tr[e, 0]}}, traction={0: force[e]})
            _compare(batch, e, live)
    return batch, singles


def test_l0_single_env_batch():
    _run_l0([dict()])


def test_l0_heterogeneous_batch():
    batch, singles = _run_l0(HETERO)
    assert len(set(batch.packed.particle_counts.tolist())) == 3
    assert any(np.abs(s.solver.impulse.numpy()).max() > 0 for s in singles)     # table contact active


def test_l0_copies():
    _run_l0([dict(), dict(), dict()])


def test_l1_perturbation_isolation():
    a = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True)
    b = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True)
    for k in range(3):
        tr, force = _inputs(k, 3)
        a.advance(translation=tr, traction=force)
        tr2, force2 = tr.copy(), force.copy()
        tr2[1, 0, 2] += 3e-4
        force2[1, 0] = 0.05
        b.advance(translation=tr2, traction=force2)
    for e in (0, 2):
        sa, sb = a.solver.snapshot(e), b.solver.snapshot(e)
        for name in sa:
            assert np.array_equal(sa[name], sb[name]), (e, name)
        assert np.array_equal(a.summary.numpy()[e], b.summary.numpy()[e])
    assert not np.array_equal(a.solver.snapshot(1)["x"], b.solver.snapshot(1)["x"])


def _classified():
    return set(rmb.SOLVER_RESET_ARRAYS+rmb.SOLVER_CONSTANT_ARRAYS+rmb.SOLVER_SHARED_ARRAYS)


def test_reset_classification_is_vars_complete():
    batch = rmb.BatchedRubberMaterial([_live()], device="cpu", parity_mode=True)
    solver_names = set(batch.solver.device_arrays())
    assert solver_names == _classified(), solver_names ^ _classified()
    material = {k for k, v in vars(batch).items() if hasattr(v, "numpy") and k != "solver"}
    declared = set(rmb.MATERIAL_RESET_ARRAYS+rmb.MATERIAL_CONSTANT_ARRAYS+rmb.RESET_INPUTS
                   + rmb.RECOMPUTED_EVERY_CONTROL_STEP+rmb.DIAGNOSTICS_SCRATCH)
    assert material == declared, material ^ declared


def test_partial_reset_equals_fresh_env_and_keeps_others():
    batch = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True)
    fresh = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True)
    for k in range(2):
        tr, force = _inputs(k, 3)
        batch.advance(translation=tr, traction=force)
    before = {e: batch.solver.snapshot(e) for e in (0, 2)}
    batch.reset([1])
    snap, ref = batch.solver.snapshot(1), fresh.solver.snapshot(1)
    reset_names = set(rmb.SOLVER_RESET_ARRAYS) | set(rmb.SOLVER_CONSTANT_ARRAYS)
    for name in snap:
        if name == "index":
            continue
        assert np.array_equal(snap[name], ref[name]), name
    covered = set(snap) | {"reference", "plane", "box_c", "box_h", "env_of_particle", "env_of_node", "env_of_group",
                           "particle_offset", "node_offset", "cell_env_offset", "marker_offsets", "marker_particles",
                           "group_kind", "group_origin", "slot", "slot_code", "node_X", "node_mask", "bc_nodes"}
    assert reset_names <= covered | {"params"}
    for name in rmb.MATERIAL_RESET_ARRAYS:
        assert np.array_equal(getattr(batch, name).numpy()[1], getattr(fresh, name).numpy()[1]), name
    for e in (0, 2):
        after = batch.solver.snapshot(e)
        for name in after:
            if name != "index":
                assert np.array_equal(after[name], before[e][name]), (e, name)
    # the reset env then evolves exactly like a fresh env (index is the shared lockstep counter)
    tr, force = _inputs(0, 3)
    batch.advance(translation=tr, traction=force)
    fresh.advance(translation=tr, traction=force)
    for name, value in batch.solver.snapshot(1).items():
        assert np.array_equal(value, fresh.solver.snapshot(1)[name]), name


def test_reset_with_resampled_material_equals_single_reference():
    ranges = {"mu0": (3e5, 1e6), "density": (1000.0, 1200.0), "friction_table": (0.3, 1.5)}
    batch = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True, ranges=ranges)
    tr, force = _inputs(0, 3)
    batch.advance(translation=tr, traction=force)
    new = MaterialValues(mu0=8e5, kappa=1.6e7, density=1010.0, friction_table=1.1)
    batch.reset([1], materials=[new])
    single = _live(**{**HETERO[1], "mu": 8e5, "rho": 1010.0, "ft": 1.1})
    assert np.array_equal(batch.solver.params.numpy()[1][:4], np.float32([8e5, 1.6e7, 1010.0, 1.1]))
    for k in range(2):
        tr, force = _inputs(k, 3)
        batch.advance(translation=tr, traction=force)
        single.advance({0: {"translation": tr[1, 0]}}, traction={0: force[1]})
        _compare(batch, 1, single)
    with pytest.raises(ValueError, match="outside the declared range"):
        batch.reset([0], materials=[MaterialValues(mu0=2e6, kappa=4e7, density=1100.0)])
    batch.reset([0, 2], seeds=[11, 12], rng_ranges=True)
    p = batch.solver.params.numpy()
    assert 3e5 <= p[0, 0] <= 1e6 and 1000 <= p[2, 2] <= 1200 and int(batch.seed.numpy()[2]) == 12


def test_audits_and_receipt():
    batch = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True)
    assert all(v == 0 for v in batch.audits().values())
    rc = batch.receipt()
    assert rc["calibration"] == "SIMULATOR_ONLY_UNCALIBRATED" and rc["solver"]["pic"] == 0.0
    assert rc["layout"]["num_envs"] == 3 and rc["device_hashgrid_guard"].startswith("not_applicable")


def test_refusals():
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_EMPTY_BATCH"):
        rmb.BatchedRubberMaterial([], device="cpu", parity_mode=True)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_CPU_WITHOUT_PARITY_MODE"):
        rmb.BatchedRubberMaterial([_live()], device="cpu")
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial([_live(), _live(n_t=3)], device="cpu", parity_mode=True)
    stepped = _live()
    stepped.advance()
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_SINGLE_ENV_STATE"):
        rmb.BatchedRubberMaterial([stepped], device="cpu", parity_mode=True)
    batch = rmb.BatchedRubberMaterial([_live(), _live()], device="cpu", parity_mode=True)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET"):
        batch.reset([1], scene_keys=["other"])
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS"):
        rmb.BatchedRubberMaterial([_live()], device="cpu", parity_mode=True,
                                  constitutive_model="neo_hookean_jbar_prony_v1")


def test_strict_mode_raises_with_env_ids():
    spec = rs.RubberStripSpec(length=0.004, width=0.004, thickness=0.004, dx=0.001, case="block")
    lives = [LiveRubberMaterial(spec, material=MaterialValues(mu0=4e5, kappa=8e6, density=1150.0), control_dt=1e-4,
                                gravity=(0, 0, 0)) for _ in range(2)]
    batch = rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True)
    A = np.tile(np.eye(3), (2, 1, 1, 1))
    A[1, 0] = np.diag([0.5, 0.5, 0.5])
    with pytest.raises(rmb.InvalidRubberEnvs) as info:
        batch.advance(affine=A)
    assert info.value.env_ids == [1] and info.value.reasons[1] == ["RUBBER_J_DOMAIN"]
    batch.reset([1])
    assert int(batch.invalid_reason.numpy()[1]) == 0
