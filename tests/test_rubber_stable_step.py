"""ADR 0023 §7 amendment (P2-1 follow-up): deformation-aware stable step of tlmpm_rubber_v1.

- ``rubber_constitutive.wave_modulus_bound`` (Warp, float32) equals its float64 numpy reference and bounds the exact
  reference-frame acoustic tensor from above for every deformation tested (finite differences of the PK1);
- the opt-in ``stable_step_guard`` is read-only (bitwise identical state with the guard on), counts violations,
  sets the sticky ``RUBBER_CFL_EXCEEDED`` bit (strict mode raises) and is cleared by reset;
- the step rule's ``deformation_reserve`` divides dt and is refused below 1 (r = 1 is the v1 rule);
- batched L0 with the guard on: per-env violation counters and every array bitwise equal to the single references
  (non-vacuous: one env trips, the others do not);
- regression for the P2-1 "test-order pollution" report: the rubber runtimes leave Warp's process-global state
  (default device, launch array access mode, config flags) unchanged.
"""
import math

import numpy as np
import pytest
import warp as wp

from robotic_cutting_sim import rubber_constitutive as rc
from robotic_cutting_sim import rubber_material as rm
from robotic_cutting_sim import rubber_reference as R
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial, RubberInvalidState
from robotic_cutting_sim.rubber_material import MaterialValues

MU, KAPPA, RHO = 4e5, 8e6, 1150.0


def _acoustic_max(F, mu, kappa, dirs=400, seed=0, h=1e-7):
    rng = np.random.default_rng(seed)
    Ns = rng.normal(size=(dirs, 3))
    Ns = np.vstack([np.eye(3), Ns/np.linalg.norm(Ns, axis=1, keepdims=True)])
    best = 0.0
    for N in Ns:
        Q = np.zeros((3, 3))
        for i in range(3):
            w = np.zeros(3)
            w[i] = h
            Q[:, i] = (R.pk1(F+np.outer(w, N), mu, kappa)-R.pk1(F-np.outer(w, N), mu, kappa))@N/(2*h)
        best = max(best, np.linalg.eigvalsh(0.5*(Q+Q.T)).max())
    return best


def _random_F(rng, lo=0.4, hi=2.5):
    Q1, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q2, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    F = Q1@np.diag(np.exp(rng.uniform(math.log(lo), math.log(hi), 3)))@Q2
    return F if np.linalg.det(F) > 0 else -F


def test_bound_matches_numpy_and_bounds_the_exact_acoustic_tensor():
    rng = np.random.default_rng(7)
    Fs = [np.eye(3), np.diag([0.5, 1.39, 1.39]), np.diag([1.5, 0.82, 0.82])]+[_random_F(rng) for _ in range(40)]
    Fs = np.array(Fs)
    J = np.linalg.det(Fs)
    warp_c = rc.evaluate_cfl_effective(Fs, J, MU, KAPPA, RHO, 1.0)
    ref = np.array([math.sqrt(rc.wave_modulus_bound_np(F, MU, KAPPA)/RHO) for F in Fs])
    assert np.max(np.abs(warp_c-ref)/ref) < 1e-5
    assert rc.wave_modulus_bound_np(np.eye(3), MU, KAPPA) == pytest.approx(KAPPA+8*MU/3, rel=1e-12)
    for k, F in enumerate(Fs[:20]):
        exact = _acoustic_max(F, MU, KAPPA, seed=k)
        assert rc.wave_modulus_bound_np(F, MU, KAPPA) >= exact*(1-1e-6), k
    # uniaxial states: the bound is tight (the exact discrete-operator ratios are in the P2-1 follow-up spectrum)
    for lam in (0.4, 0.6, 1.5, 2.0):
        s = R.uniaxial_stress(lam, MU, KAPPA)["lateral"]
        F = np.diag([lam, s, s])
        assert rc.wave_modulus_bound_np(F, MU, KAPPA)/_acoustic_max(F, MU, KAPPA) < 1.15


def test_nonfinite_deformation_counts_as_violation():
    c = rc.evaluate_cfl_effective(np.full((1, 3, 3), np.nan), [1.0], MU, KAPPA, RHO, 1.0)
    assert not np.isfinite(c[0])


def test_step_rule_deformation_reserve():
    base = rm.step_rule(1e-3, 0.02, mu0_max=MU, kappa_max=KAPPA, density_min=RHO)
    same = rm.step_rule(1e-3, 0.02, mu0_max=MU, kappa_max=KAPPA, density_min=RHO, deformation_reserve=1.0)
    double = rm.step_rule(1e-3, 0.02, mu0_max=MU, kappa_max=KAPPA, density_min=RHO, deformation_reserve=2.0)
    assert same["dt_s"] == base["dt_s"] and same["substeps_per_control_step"] == base["substeps_per_control_step"]
    assert double["dt_max_s"] == pytest.approx(base["dt_max_s"]/2) and double["deformation_reserve"] == 2.0
    for bad in (0.5, float("nan"), float("inf")):
        with pytest.raises(rm.RubberRecipeError):
            rm.step_rule(1e-3, 0.02, mu0_max=MU, kappa_max=KAPPA, density_min=RHO, deformation_reserve=bad)


def _strip(**kw):
    spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2, case="A")
    return LiveRubberMaterial(spec, material=MaterialValues(mu0=MU, kappa=KAPPA, density=RHO), control_dt=2e-4,
                              gravity=(0, 0, -9.81), **kw)


def test_guard_is_read_only_and_quiet_at_small_strain():
    off, on = _strip(), _strip(stable_step_guard=rc.STABLE_STEP_GUARD_DEFAULT_LIMIT)
    with pytest.raises(ValueError):
        _strip(stable_step_guard=-1.0)
    for k in range(3):
        motion = {0: {"translation": np.array([0.0, 0.0, 2e-5*(k+1)])}}
        off.advance(motion)
        on.advance(motion)
    for name, a in off.solver.device_arrays().items():
        assert np.array_equal(a.numpy(), getattr(on.solver, name).numpy()), name
    assert int(on.solver.cfl_violations.numpy()[0]) == 0 and on.invalid_reason == 0
    c = on.last_diagnostics["cfl_eff_max"]
    assert 0.5 < c < 0.6        # CFL 0.5 x 1.031 (bound at F = I) plus the small deformation
    assert on.receipt()["stable_step_guard"]["id"] == rc.STABLE_STEP_GUARD_ID
    assert off.receipt()["stable_step_guard"] is None


def _block(**kw):
    spec = rs.RubberStripSpec(length=0.004, width=0.004, thickness=0.004, dx=0.001, case="block")
    return LiveRubberMaterial(spec, material=MaterialValues(mu0=MU, kappa=KAPPA, density=RHO), control_dt=1e-4,
                              gravity=(0, 0, 0), **kw)


def test_guard_flags_strong_compression_and_reset_clears_it():
    """Imposed homogeneous F = diag(0.3, 1.6, 1.6) (J = 0.77, inside the J domain): the reference-frame wave speed
    is about 3x its reference value, so dt c_bound / dx is about 1.5 at CFL 0.5 -> RUBBER_CFL_EXCEEDED."""
    A = np.diag([0.3, 1.6, 1.6])
    quiet = _block(strict=False)
    quiet.advance({0: {"affine": A}})
    assert quiet.invalid_reason == 0                     # guard off: nothing flags (J stays in the domain)
    live = _block(strict=False, stable_step_guard=1.0, record=True)
    live.advance({0: {"affine": A}})
    assert live.invalid_reason == 4 and live.records[-1]["invalid_flags"] == ["RUBBER_CFL_EXCEEDED"]
    assert live.records[-1]["stable_step"]["guard_violations_step"] > 0
    assert live.last_diagnostics["cfl_eff_max"] > 1.0
    live.reset()
    assert live.invalid_reason == 0 and int(live.solver.cfl_violations.numpy()[0]) == 0
    strict = _block(stable_step_guard=1.0)
    with pytest.raises(RubberInvalidState, match="RUBBER_CFL_EXCEEDED"):
        strict.advance({0: {"affine": A}})


STEP_MATERIAL = MaterialValues(mu0=1e6, kappa=2e7, density=1000.0)
HETERO = [dict(mu=4e5, rho=1150.0), dict(L=0.009, W=0.006, mu=7e5, rho=1050.0), dict(L=0.015, mu=3e5, rho=1200.0)]


def _live(L=0.012, W=0.003, mu=4e5, rho=1150.0, guard=0.35):
    spec = rs.RubberStripSpec(length=L, width=W, thickness=0.003, n_t=2, case="A",
                              traction_regions=(("tip", (L-0.0015, -1, -1), (1, 1, 1)),))
    return LiveRubberMaterial(spec, material=MaterialValues(mu0=mu, kappa=20*mu, density=rho), control_dt=2e-4,
                              gravity=(0, 0, -9.81), step_material=STEP_MATERIAL, table_plane=(0.0, 0.0, 1.0, 0.0),
                              stable_step_guard=guard, strict=False)


def test_batched_l0_with_guard_on():
    """Limit 0.35: at the batch dt the mu 0.7 MPa env sits at dt c_bound / dx ~ 0.43 (trips), the others at
    ~ 0.26-0.31 (quiet). Every array, the per-env counters, diagnostics and validity bits equal the singles."""
    singles = [_live(**kw) for kw in HETERO]
    batch = rmb.BatchedRubberMaterial([_live(**kw) for kw in HETERO], device="cpu", parity_mode=True, strict=False)
    assert batch.receipt()["solver"]["stable_step_guard"]["limit"] == 0.35
    for k in range(2):
        tr = np.zeros((3, 1, 3))
        tr[:, 0, 2] = 2e-5*(k+1)
        force = np.zeros((3, 3))
        batch.advance(translation=tr, traction=force)
        for e, live in enumerate(singles):
            live.advance({0: {"translation": tr[e, 0]}}, traction={0: force[e]})
            snap = batch.solver.snapshot(e)
            for name in ("x", "v", "F", "stress", "jbar", "new_v"):
                ref = getattr(live.solver, name).numpy()
                assert np.array_equal(snap[name], ref), (e, name)
            assert np.array_equal(snap["cfl_violations"], live.solver.cfl_violations.numpy()), e
            assert np.array_equal(batch.summary.numpy()[e], live.solver.diagnostics.summary.numpy()[0]), e
            assert int(batch.invalid_reason.numpy()[e]) == live.invalid_reason, e
    counts = batch.solver.cfl_violations.numpy()
    assert counts[1] > 0 and counts[0] == 0 and counts[2] == 0
    assert rmb.decode_reason(batch.invalid_reason.numpy()[1]) == ["RUBBER_CFL_EXCEEDED"]
    batch.reset([1])
    assert int(batch.solver.cfl_violations.numpy()[1]) == 0 and int(batch.invalid_reason.numpy()[1]) == 0


def _warp_global_state():
    cfg = wp.config
    return {"default_device": str(wp.get_device()), "launch_array_access_mode": str(cfg.launch_array_access_mode),
            "mode": cfg.mode, "verify_fp": cfg.verify_fp, "verify_cuda": cfg.verify_cuda,
            "max_unroll": cfg.max_unroll, "enable_backward": cfg.enable_backward}


def test_rubber_runtimes_leave_warp_global_state_unchanged():
    """P2-1 follow-up item 3. The reported order dependence of
    tests/test_batched_particle_contacts.py::test_fallback_grid_reproduces_warp_grouped_query_sequence is not rubber
    state: that test fails alone whenever a GPU is visible (its ``FallbackGrid.build`` launches on Warp's default
    device, cuda:0, on CPU arrays, racing the host radix sort) and passes with CUDA hidden or after
    ``wp.set_device('cpu')`` (artifacts/rl_ready/p2_1/pollution_diag_001). This test pins that the rubber runtimes,
    single and batched, change none of Warp's process-global settings."""
    before = _warp_global_state()
    live = _strip(stable_step_guard=1.0)
    live.advance()
    batch = rmb.BatchedRubberMaterial([_live(guard=None)], device="cpu", parity_mode=True)
    batch.advance(translation=np.zeros((1, 1, 3)))
    batch.reset()
    assert _warp_global_state() == before
