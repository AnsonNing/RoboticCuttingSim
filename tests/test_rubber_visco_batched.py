"""ADR 0023 §21.2.4-21.2.5 / R3.6: batched ``neo_hookean_jbar_prony_v1`` (small CPU configurations of RUBBER-VISC-06,
-07, -09; the formal runs use the protocol's sizes, ``scripts/rubber/p2_3/run_visc_v1.py``).

- L0: N = 1 and N = 3 heterogeneous (geometry, mu0, rho, g, tau; M = 2 uniform, SLS padded) for case A (moving grip
  with rotation), case C (two grips, one inactive for one step) and the block (affine), every array (incl. visco_q,
  visco_s_prev, params64, the visco energy arrays), diagnostics, visco diagnostics, markers and wrenches byte-wise after
  every control step;
- L1: perturbing one env leaves the others byte-unchanged; vars()-complete reset classification with the model state;
- masked reset clears the material memory (fresh batch / fresh single after the reset), resampled Prony terms,
  params64 = host expm1 values, the next plain reset restores the packed values;
- M = 2 with g = 0 batched is byte-identical to the elastic batch (VISC-09 batched);
- refusals: mixed models, different M, different visco_energy, a material with another M on reset.
"""
import numpy as np
import pytest

from robotic_cutting_sim import rubber_visco_constitutive as rvc
from robotic_cutting_sim import rubber_visco_protocol as vp
from robotic_cutting_sim.batched import constitutive_rubber as cr
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.rubber_material import MaterialValues

P2 = ((0.2, 0.005), (0.15, 0.05))
SLS = ((0.3, 0.01),)
STEP_MATERIAL = MaterialValues(mu0=1e6, kappa=2e7, density=1000.0)
HETERO = [dict(), dict(L=0.009, mu=5.6e5, rho=1050.0, terms=SLS), dict(L=0.015, mu=3e5, rho=1200.0,
                                                                        terms=((0.1, 0.002), (0.25, 0.02)))]


def _spec(case, L=0.012):
    if case == "block":
        return vp.block_spec()
    region = (("tip", (L-0.0015, -1, -1), (1, 1, 1)),) if case == "A" else \
        (("mid", (L/2-0.00075, -1, -1), (L/2+0.00075, 1, 1)),)
    return vp.strip_spec(L=L, W=0.003, t=0.003, n_t=2, case=case, traction_regions=region)


def _live(case="A", L=0.012, mu=4e5, rho=1150.0, terms=P2, energy=vp.VISCO_ENERGY, M=2):
    return vp.make_live(_spec(case, L), vp.material(mu, terms, rho, M=M if terms else None), control_dt=2e-4,
                        gravity=(0, 0, -9.81), energy=energy, step_material=STEP_MATERIAL)


def _inputs(k, N, G, case):
    from scipy.spatial.transform import Rotation
    tr = np.zeros((N, G, 3))
    R = np.tile(np.eye(3), (N, G, 1, 1))
    A = np.tile(np.eye(3), (N, G, 1, 1))
    for e in range(N):
        if case == "block":
            A[e, 0] = np.eye(3)+(k+1)*(e+1)*np.array([[0.01, 0.005, 0.0], [0.0, -0.005, 0.002], [0.0, 0.0, -0.005]])
        else:
            tr[e, 0] = [0.0, 1e-5*(k+1)*(e+1), 3e-5*(k+1)]
            R[e, G-1] = Rotation.from_rotvec([0.01*(k+1)*(e+1), 0.005*(k+1), 0.0]).as_matrix()
            if G > 1:
                tr[e, 1] = [2e-5*(k+1)*(e+1), 0.0, 0.0]
    force = np.zeros((N, 3))
    force[:, 0] = 1e-3*np.arange(1, N+1)
    force[:, 2] = -2e-3
    return tr, R, A, force


def _advance(batch, singles, k, case, inactive_step=None):
    N, G = batch.num_envs, batch.G
    tr, R, A, force = _inputs(k, N, G, case)
    active = np.ones((N, G), np.int32)
    if inactive_step == k and N > 1:
        active[1, G-1] = 0
    if case == "block":
        batch.advance(affine=A)
    else:
        batch.advance(translation=tr, rotation=R, active=active, traction=force)
    for e, live in enumerate(singles):
        if case == "block":
            live.advance({0: {"affine": A[e, 0]}})
        else:
            motions = {g: {"translation": tr[e, g], "rotation": R[e, g], "active": bool(active[e, g])} for g in range(G)}
            live.advance(motions, traction={0: force[e]})


def _run_l0(case, kws, steps=5, inactive_step=None):
    singles = [_live(case, **kw) for kw in kws]
    batch = rmb.BatchedRubberMaterial([_live(case, **kw) for kw in kws], device="cpu", parity_mode=True)
    for k in range(steps):
        _advance(batch, singles, k, case, inactive_step)
        for e, live in enumerate(singles):
            assert vp.compare_env(batch, e, live) == [], (case, k, e)
    return batch, singles


@pytest.mark.parametrize("case", ["A", "C", "block"])
def test_l0_single_env_batch(case):
    batch, singles = _run_l0(case, [dict()])
    assert batch.solver.M == 2 and batch.solver.model.model_id == "neo_hookean_jbar_prony_v1"
    assert np.any(batch.solver.visco_q.numpy())
    assert batch.receipt()["solver"]["constitutive_model"]["terms"] == 2


@pytest.mark.parametrize("case", ["A", "C", "block"])
def test_l0_heterogeneous_batch(case):
    kws = HETERO if case != "block" else [dict(), dict(mu=5.6e5, rho=1050.0, terms=SLS),
                                          dict(mu=3e5, terms=((0.1, 0.002), (0.25, 0.02)))]
    batch, singles = _run_l0(case, kws, inactive_step=1 if case == "C" else None)
    assert len({s.solver.n for s in singles}) == (3 if case != "block" else 1)
    assert all(float(s.solver.visco_dissipation.numpy()[0]) > 0 for s in singles)


def test_l1_perturbation_isolation():
    a = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True)
    b = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True)
    for k in range(3):
        tr, R, _, force = _inputs(k, 3, 1, "A")
        a.advance(translation=tr, rotation=R, traction=force)
        tr[1, 0, 2] += 3e-4
        force[1, 0] = 0.05
        b.advance(translation=tr, rotation=R, traction=force)
    for e in (0, 2):
        sa, sb = a.solver.snapshot(e), b.solver.snapshot(e)
        for name in sa:
            assert vp.same_bytes(sa[name], sb[name]), (e, name)
        assert vp.same_bytes(a.visco_summary.numpy()[e], b.visco_summary.numpy()[e])
    assert not vp.same_bytes(a.solver.snapshot(1)["visco_q"], b.solver.snapshot(1)["visco_q"])


@pytest.mark.parametrize("energy", ["off_v1", vp.VISCO_ENERGY])
def test_reset_classification_is_vars_complete_with_the_model_state(energy):
    batch = rmb.BatchedRubberMaterial([_live(energy=energy)], device="cpu", parity_mode=True)
    names = set(batch.solver.device_arrays())
    declared = set(rmb.SOLVER_RESET_ARRAYS+rmb.SOLVER_CONSTANT_ARRAYS+rmb.SOLVER_SHARED_ARRAYS
                   + rmb.SOLVER_RECOMPUTED_ARRAYS)
    for table in (rmb.OPTION_RESET_ARRAYS, rmb.OPTION_CONSTANT_ARRAYS):
        for value in table.values():
            declared |= set(value)
    declared |= set(rmb.MODEL_STATE_RESET_ARRAYS["neo_hookean_jbar_prony_v1"])
    if energy != "off_v1":
        declared |= set(rmb.VISCO_ENERGY_RESET_ARRAYS)
    assert names == declared, names ^ declared
    state = {v.name for v in batch.solver.model.per_particle_state}
    assert state == {"visco_q", "visco_s_prev"} and state <= set(rmb.MODEL_STATE_RESET_ARRAYS[
        "neo_hookean_jbar_prony_v1"])
    material = {k for k, v in vars(batch).items() if hasattr(v, "numpy") and k != "solver"}
    declared = set(rmb.MATERIAL_RESET_ARRAYS+rmb.MATERIAL_CONSTANT_ARRAYS+rmb.RESET_INPUTS
                   + rmb.RECOMPUTED_EVERY_CONTROL_STEP+rmb.DIAGNOSTICS_SCRATCH+rmb.VISCO_MATERIAL_CONSTANT_ARRAYS
                   + rmb.VISCO_RESET_INPUTS)
    assert material == declared, material ^ declared
    diag = {k for k, v in vars(batch.visco_diagnostics).items() if hasattr(v, "numpy")}
    assert diag == set(rmb.VISCO_DIAGNOSTICS_SCRATCH)


def test_masked_reset_clears_material_memory_and_resamples_prony():
    ranges = {"prony_g_scale": (0.5, 1.2), "prony_tau_scale": (0.5, 2.0)}
    batch = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True, ranges=ranges)
    fresh = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True, ranges=ranges)
    for k in range(3):
        tr, R, _, force = _inputs(k, 3, 1, "A")
        batch.advance(translation=tr, rotation=R, traction=force)
    assert np.any(batch.solver.snapshot(1)["visco_q"])
    before = {e: batch.solver.snapshot(e) for e in (0, 2)}
    batch.reset([1])
    snap, ref = batch.solver.snapshot(1), fresh.solver.snapshot(1)
    for name in snap:
        if name != "index":
            assert vp.same_bytes(snap[name], ref[name]), name
    for e in (0, 2):
        for name, value in batch.solver.snapshot(e).items():
            if name != "index":
                assert vp.same_bytes(value, before[e][name]), (e, name)
    # resampled Prony terms (g x 0.8, tau x 1.5): params64 = host expm1 row; equals a single built with them
    m = batch.packed.receipts and _live(**HETERO[1]).material
    scaled = rmb.scale_prony(m.prony, 0.8, 1.5)
    m2 = MaterialValues(mu0=m.mu0, kappa=m.kappa, density=m.density, prony=scaled)
    batch.reset([1], materials=[m2])
    row = batch.solver.params64.numpy()[1]
    assert row.tobytes() == rvc.params64_row(scaled, batch.solver.dt).tobytes()
    single = vp.make_live(_spec("A", 0.009), m2, control_dt=2e-4, gravity=(0, 0, -9.81),
                          step_material=STEP_MATERIAL)
    for k in range(3):
        tr, R, _, force = _inputs(k, 3, 1, "A")
        batch.advance(translation=tr, rotation=R, traction=force)
        single.advance({0: {"translation": tr[1, 0], "rotation": R[1, 0]}}, traction={0: force[1]})
        assert vp.compare_env(batch, 1, single) == [], k
    batch.reset([1])
    assert batch.solver.params64.numpy()[1].tobytes() == fresh.solver.params64.numpy()[1].tobytes()
    with pytest.raises(ValueError, match="prony_tau_scale"):
        batch.reset([1], materials=[MaterialValues(mu0=m.mu0, kappa=m.kappa, density=m.density,
                                                   prony=rmb.scale_prony(m.prony, 1.0, 3.0))])
    batch.reset([0, 2], rng_ranges=True)
    assert batch.solver.params64.numpy()[0].tobytes() != fresh.solver.params64.numpy()[0].tobytes()


@pytest.mark.parametrize("case", ["A", "block"])
def test_visc09_batched_zero_g_equals_the_elastic_batch(case):
    zero = ((0.0, 0.005), (0.0, 0.05))
    el = rmb.BatchedRubberMaterial([_live(case, terms=(), energy="off_v1"), _live(case, mu=6e5, terms=(),
                                                                                  energy="off_v1")],
                                   device="cpu", parity_mode=True)
    vz = rmb.BatchedRubberMaterial([_live(case, terms=zero), _live(case, mu=6e5, terms=zero)], device="cpu",
                                   parity_mode=True)
    assert el.solver.M == 0 and vz.solver.M == 2
    for k in range(5):
        tr, R, A, force = _inputs(k, 2, 1, case)
        for b in (el, vz):
            if case == "block":
                b.advance(affine=A)
            else:
                b.advance(translation=tr, rotation=R, traction=force)
        for e in range(2):
            se, sz = el.solver.snapshot(e), vz.solver.snapshot(e)
            for name in se:
                assert vp.same_bytes(se[name], sz[name]), (k, e, name)
        assert vp.same_bytes(el.summary.numpy(), vz.summary.numpy())
        assert vp.same_bytes(el.wrench.numpy(), vz.wrench.numpy())


def test_refusals():
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS"):
        rmb.BatchedRubberMaterial([_live(), _live(terms=(), energy="off_v1")], device="cpu", parity_mode=True)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial([_live(), _live(terms=SLS, M=None)], device="cpu", parity_mode=True)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial([_live(), _live(energy="off_v1")], device="cpu", parity_mode=True)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_MIXED_CONSTITUTIVE_MODELS"):
        rmb.BatchedRubberMaterial([_live(terms=(), energy="off_v1")], device="cpu", parity_mode=True,
                                  constitutive_model="neo_hookean_jbar_prony_v1")
    batch = rmb.BatchedRubberMaterial([_live(), _live()], device="cpu", parity_mode=True)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        batch.reset([1], materials=[MaterialValues(mu0=4e5, kappa=8e6, density=1150.0, prony=SLS)])
    with pytest.raises(ValueError, match="randomisation range"):
        rmb.BatchedRubberMaterial([_live(terms=(), energy="off_v1")], device="cpu", parity_mode=True,
                                  ranges={"prony_g_scale": (0.5, 1.0)})
    with pytest.raises(ValueError, match="sum g"):
        rmb.BatchedRubberMaterial([_live()], device="cpu", parity_mode=True, ranges={"prony_g_scale": (0.5, 3.0)})
    assert cr.DECLARED_LATER == {}
    with pytest.raises(UnsupportedBatchConfiguration):
        cr.prony_model(5)
