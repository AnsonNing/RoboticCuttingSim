"""ADR 0023 §21.7 / protocol ``rubber_visc_tests_v2``: energy option ``branch_work_kick_average_v2`` and the v2
estimator / inertia de-embedding (``rubber_visco_reference``).

- bitwise-off: ``off_v1`` and ``branch_work_v1`` build no v2 array; the v2 option adds exactly ``visco_df_prev`` and
  ``visco_branch_work_left``;
- bookkeeping only: with a moving grip (case A with rotation, case C two grips), every state array, the wrench, the
  frozen diagnostics and the visco diagnostics (except the branch-work column) are byte-identical between ``off_v1``,
  ``branch_work_v1`` and ``branch_work_kick_average_v2``; ``visco_pb`` / D / d_negative byte-identical to v1; the v2
  left-point accumulator is byte-identical to the v1 ``visco_branch_work``;
- the kick-paired accumulator equals a host float64 recomputation every substep (dF_{n-1} carried across control
  steps);
- reset() clears the v2 arrays, episodes repeat; batched L0 byte-wise (N = 1, N = 3 heterogeneous), masked reset,
  vars()-complete classification with the v2 arrays, mixed v1 / v2 batch refused;
- property (small tap): the kick-paired W_b closes the books better than the left-point sum and equals D + Psi_v;
- v2 window, estimator (exact on whole-period synthetic sinusoids, the v1 window is not), rod transfer (PDE solution,
  small-kL expansion at the actual marker layers), de-embedding round trip.

No artifacts, torch or CUDA needed (Warp CPU device).
"""
import math

import numpy as np
import pytest

from robotic_cutting_sim import rubber_visco_protocol as vp
from robotic_cutting_sim import rubber_visco_reference as ref
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.rubber_material import MaterialValues
from robotic_cutting_sim.tlmpm_rubber_visco_v1 import KICK_AVERAGE_ARRAYS, KICK_AVERAGE_ENERGY, TLMPMRubberVisco

P2 = ((0.2, 0.005), (0.15, 0.05))
SLS = ((0.3, 0.01),)
STEP_MATERIAL = MaterialValues(mu0=1e6, kappa=2e7, density=1000.0)
V2 = KICK_AVERAGE_ENERGY
BRANCH_COLUMN = 4                     # tlmpm_rubber_visco_v1.VISCO_DIAG_COLUMNS index of "branch_work"


def _spec(case, L=0.012):
    region = (("tip", (L-0.0015, -1, -1), (1, 1, 1)),) if case == "A" else \
        (("mid", (L/2-0.00075, -1, -1), (L/2+0.00075, 1, 1)),)
    return vp.strip_spec(L=L, W=0.003, t=0.003, n_t=2, case=case, traction_regions=region)


def _live(case="A", energy=V2, L=0.012, mu=4e5, rho=1150.0, terms=P2, M=2):
    return vp.make_live(_spec(case, L), vp.material(mu, terms, rho, M=M), control_dt=2e-4, gravity=(0, 0, -9.81),
                        energy=energy, step_material=STEP_MATERIAL)


def _motions(case, k, scale=1.0):
    from scipy.spatial.transform import Rotation
    m = {0: {"translation": scale*np.array([0.0, 1e-5*(k+1), 3e-5*(k+1)]),
             "rotation": Rotation.from_rotvec([scale*0.01*(k+1), 0.0, 0.0]).as_matrix()}}
    if case == "C":
        m[1] = {"translation": scale*np.array([2e-5*(k+1), 0.0, 0.0])}
    return m


def _same(a, b):
    return vp.same_bytes(a, b)


def test_default_and_v1_build_no_v2_array():
    for energy in ("off_v1", "branch_work_v1"):
        s = _live(energy=energy).solver
        assert not any(hasattr(s, n) for n in KICK_AVERAGE_ARRAYS), energy
    s1, s2 = _live(energy="branch_work_v1").solver, _live(energy=V2).solver
    assert set(s2.device_arrays())-set(s1.device_arrays()) == set(KICK_AVERAGE_ARRAYS)
    assert set(s1.device_arrays()) <= set(s2.device_arrays())
    assert s2.visco_df_prev.dtype.__name__ == "mat33d" and s2.visco_df_prev.shape == (s2.n,)
    assert s2.solver_receipt()["options"]["visco_energy"] == V2
    with pytest.raises(ValueError):
        _live(energy="branch_work_kick_average_v3")


@pytest.mark.parametrize("case", ["A", "C"])
def test_v2_is_bookkeeping_only_and_its_left_point_sum_is_the_v1_branch_work(case):
    off, v1, v2 = _live(case, "off_v1"), _live(case, "branch_work_v1"), _live(case, V2)
    for k in range(5):
        for live in (off, v1, v2):
            live.advance(_motions(case, k), traction={0: np.array([1e-3, 0.0, -2e-3])})
        a0, a1, a2 = (x.solver.device_arrays() for x in (off, v1, v2))
        for name, arr in a0.items():                         # every state / option / work array of off_v1
            assert _same(arr.numpy(), a1[name].numpy()), (case, k, name, "v1")
            assert _same(arr.numpy(), a2[name].numpy()), (case, k, name, "v2")
        for name in ("visco_pb", "visco_dissipation", "visco_d_negative"):
            assert _same(a1[name].numpy(), a2[name].numpy()), (case, k, name)
        assert _same(a1["visco_branch_work"].numpy(), a2["visco_branch_work_left"].numpy()), (case, k)
        for x in (v1, v2):
            assert _same(off.last_wrench, x.last_wrench)
            assert _same(off.solver.diagnostics.summary.numpy(), x.solver.diagnostics.summary.numpy())
        d1, d2 = v1.solver.visco_diagnostics.visco_summary.numpy(), v2.solver.visco_diagnostics.visco_summary.numpy()
        keep = [c for c in range(d1.shape[1]) if c != BRANCH_COLUMN]
        assert _same(d1[:, keep], d2[:, keep])
        assert v2.last_visco["branch_work"] == float(v2.solver.visco_branch_work.numpy()[0])
    w1, wkp = float(v1.solver.visco_branch_work.numpy()[0]), float(v2.solver.visco_branch_work.numpy()[0])
    assert w1 != 0.0 and wkp != 0.0 and wkp != w1


def test_kick_paired_accumulator_equals_a_host_recompute_every_substep():
    class Probe(TLMPMRubberVisco):
        def step(self):
            w0 = float(self.visco_branch_work.numpy()[0])
            l0 = float(self.visco_branch_work_left.numpy()[0])
            super().step()
            pb = self.visco_pb.numpy()
            v = self.new_v.numpy().astype(np.float64)
            grad = np.einsum("pci,pcj->pij", v[self.scene.nodes], self.scene.gradients.astype(np.float64))
            dF = self.dt*grad
            V = self.volume
            host_w = float(np.sum(V*0.5*np.einsum("pij,pij->p", pb, dF+self.host_prev)))
            host_l = float(np.sum(V*np.einsum("pij,pij->p", pb, dF)))
            self.host_prev = dF
            w1 = float(self.visco_branch_work.numpy()[0])
            l1 = float(self.visco_branch_work_left.numpy()[0])
            self.checks.append((w1-w0, host_w, w1, l1-l0, host_l, l1))
            assert np.allclose(self.visco_df_prev.numpy(), dF, rtol=1e-12, atol=1e-14*max(np.abs(dF).max(), 1e-30))

    live = _live("A")
    s = live.solver
    probe = Probe(s.scene, live.material, s.dt, substeps=s.substeps, gravity=s.gravity_tuple, stable_step_guard=1.0,
                  visco_energy=V2, **vp.OPTIONS)
    probe.checks, probe.host_prev = [], np.zeros((probe.n, 3, 3))
    for k in range(3):                                       # dF_{n-1} crosses two control-step boundaries
        probe.begin_control_step()
        for g, m in _motions("A", k).items():
            probe.set_group_motion(g, **m)
        probe.set_traction(0, np.array([1e-3, 0.0, -2e-3]))
        for _ in range(probe.substeps):
            probe.step()
        probe.diagnostics.launch()
    assert len(probe.checks) == 3*probe.substeps
    for dw, hw, w1, dl, hl, l1 in probe.checks:
        assert abs(dw-hw) <= 1e-10*abs(hw)+4*np.spacing(abs(w1))+1e-30
        assert abs(dl-hl) <= 1e-10*abs(hl)+4*np.spacing(abs(l1))+1e-30
    assert any(abs(c[1]) > 0 for c in probe.checks)


def test_single_reset_clears_the_v2_arrays_and_episodes_repeat():
    live = _live("C")
    episodes = []
    for _ in range(2):
        live.reset()
        s = live.solver
        assert not np.any(s.visco_df_prev.numpy()) and float(s.visco_branch_work_left.numpy()[0]) == 0.0
        assert float(s.visco_branch_work.numpy()[0]) == 0.0
        snaps = []
        for k in range(3):
            live.advance(_motions("C", k))
            snaps.append({n: a.numpy().copy() for n, a in s.device_arrays().items()})
        assert np.any(s.visco_df_prev.numpy())
        episodes.append(snaps)
    for a, b in zip(*episodes):
        for name in a:
            assert _same(a[name], b[name]), name


def test_kick_paired_work_closes_the_books_and_equals_the_branch_energy():
    """Small cantilever tap (320 particles, tau 50 ms term only, 150 control steps): the kick-paired W_b closes the
    layer-N books far better than the v1 left-point sum and equals the branch's own energy D + Psi_v (from Q alone),
    which the left-point sum misses by about its book drift (ADR 0023 §21.7.1)."""
    live = vp.make_live(vp.strip_spec(L=0.012, W=0.003, t=0.003, n_t=2, case="A"),
                        vp.material(4e5, ((0.0, 0.005), (0.15, 0.05)), M=2), control_dt=2e-4, energy=V2)
    rows = []
    for k in range(1, 151):
        tk = k*2e-4
        z = 1e-4*math.sin(math.pi*tk/0.004)**2 if tk < 0.004 else 0.0
        live.advance({0: {"translation": np.array([0.0, 0.0, z])}})
        d, v, s = live.last_diagnostics, live.last_visco, live.solver
        rows.append([d["kinetic_particle"], v["psi_eq"], d["gravity"], v["branch_work"],
                     float(s.visco_branch_work_left.numpy()[0]), float(s.work.numpy()[0]), v["dissipation"],
                     v["psi_visc_stored"]])
    KE, PE, G, WK, WL, WX, D, PV = np.array(rows).T
    S = np.max(KE+PE)+np.max(np.abs(WK))
    books_kick, books_left = np.abs(KE+PE+G+WK-WX).max()/S, np.abs(KE+PE+G+WL-WX).max()/S
    assert books_kick < books_left/3
    assert abs(WK[-1]-D[-1]-PV[-1]) < 1e-3*S < 0.1*abs(WL[-1]-D[-1]-PV[-1])
    assert int(live.invalid_reason) == 0


def _batched_inputs(k, N, G):
    tr = np.zeros((N, G, 3))
    for e in range(N):
        tr[e, 0] = [0.0, 1e-5*(k+1)*(e+1), 3e-5*(k+1)]
        if G > 1:
            tr[e, 1] = [2e-5*(k+1)*(e+1), 0.0, 0.0]
    force = np.zeros((N, 3))
    force[:, 0] = 1e-3*np.arange(1, N+1)
    return tr, force


@pytest.mark.parametrize("case,kws", [("A", [dict()]),
                                      ("C", [dict(), dict(L=0.009, mu=5.6e5, rho=1050.0, terms=SLS),
                                             dict(L=0.015, mu=3e5, rho=1200.0, terms=((0.1, 0.002), (0.25, 0.02)))])])
def test_batched_l0_with_the_v2_option(case, kws):
    singles = [_live(case, **kw) for kw in kws]
    batch = rmb.BatchedRubberMaterial([_live(case, **kw) for kw in kws], device="cpu", parity_mode=True)
    assert batch.solver.visco_energy == V2
    for k in range(4):
        tr, force = _batched_inputs(k, batch.num_envs, batch.G)
        batch.advance(translation=tr, traction=force)
        for e, live in enumerate(singles):
            live.advance({g: {"translation": tr[e, g]} for g in range(batch.G)}, traction={0: force[e]})
            assert vp.compare_env(batch, e, live) == [], (case, k, e)
            snap = batch.solver.snapshot(e)
            assert set(KICK_AVERAGE_ARRAYS) <= set(snap)
    assert np.any(batch.solver.visco_df_prev.numpy())


def test_batched_masked_reset_clears_the_v2_arrays_and_classification_is_vars_complete():
    kws = [dict(), dict(L=0.009, mu=5.6e5, rho=1050.0, terms=SLS)]
    batch = rmb.BatchedRubberMaterial([_live("A", **kw) for kw in kws], device="cpu", parity_mode=True)
    fresh = rmb.BatchedRubberMaterial([_live("A", **kw) for kw in kws], device="cpu", parity_mode=True)
    for k in range(3):
        tr, force = _batched_inputs(k, 2, batch.G)
        batch.advance(translation=tr, traction=force)
    before = batch.solver.snapshot(0)
    assert np.any(batch.solver.snapshot(1)["visco_df_prev"])
    batch.reset([1])
    after, ref1 = batch.solver.snapshot(1), fresh.solver.snapshot(1)
    for name in after:
        if name != "index":
            assert _same(after[name], ref1[name]), name
    for name, arr in batch.solver.snapshot(0).items():
        assert _same(arr, before[name]), name
    names = set(batch.solver.device_arrays())
    declared = set(rmb.SOLVER_RESET_ARRAYS+rmb.SOLVER_CONSTANT_ARRAYS+rmb.SOLVER_SHARED_ARRAYS
                   + rmb.SOLVER_RECOMPUTED_ARRAYS)
    for table in (rmb.OPTION_RESET_ARRAYS, rmb.OPTION_CONSTANT_ARRAYS):
        for value in table.values():
            declared |= set(value)
    declared |= set(rmb.MODEL_STATE_RESET_ARRAYS["neo_hookean_jbar_prony_v1"]) | set(rmb.VISCO_ENERGY_RESET_ARRAYS)
    declared |= set(rmb.VISCO_ENERGY_V2_RESET_ARRAYS)
    assert names == declared, names ^ declared


def test_mixed_v1_v2_batch_is_refused():
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial([_live("A", "branch_work_v1"), _live("A", V2)], device="cpu", parity_mode=True)


# ----------------------------------------------------------------------------- v2 estimator and de-embedding

MU0, KAPPA, RHO, L, DTC = 4e5, 8e6, 1150.0, 0.030, 2e-4
A_MARK, B_MARK = 0.00725, 0.02225                         # the strip's marker layers k = 2, 6 (n_t 3, dx 1 mm)


def test_steady_window_v2_matches_the_protocol_table():
    expected = {1: (1, 10000, 5000), 2: (1, 5000, 2500), 4: (1, 3750, 1250), 8: (1, 2500, 625), 16: (2, 2500, 625),
                32: (4, 2500, 625)}
    for f, (K, steps, n) in expected.items():
        w = ref.steady_window_v2(float(f), DTC, 0.05)
        assert (w["periods"], w["steps"], w["window_samples"]) == (K, steps, n), f
        assert (w["steps"]-w["window_samples"])*DTC >= 6*0.05-1e-12          # the window starts after 6 tau_max


@pytest.mark.parametrize("f", [8.0, 16.0, 32.0])
def test_whole_period_estimator_is_exact_on_steady_sinusoids_and_the_v1_window_is_not(f):
    w = 2*math.pi*f
    E = complex(ref.uniaxial_complex_modulus(w, MU0, KAPPA, P2))
    Einf = float(ref.uniaxial_complex_modulus(0.0, MU0, KAPPA, P2).real)
    win = ref.steady_window_v2(f, DTC, 0.05)
    tk = DTC*np.arange(1, win["steps"]+1)
    x = w*DTC/2
    sig = Einf*0.002-(E*0.002*np.exp(1j*w*(tk-DTC/2))*math.sin(x)/x).real      # exact control-step averages
    eps = 0.002*(1-np.cos(w*tk))
    n = win["window_samples"]
    Ew = ref.fourier_modulus_window_v2(sig[-n:], eps[-n:], tk[-n:]-DTC/2, tk[-n:], w, DTC)
    assert abs(Ew/E-1) < 1e-12
    last = tk > tk[-1]-1/f+1e-12                                               # the v1 "last full cycle"
    E1 = ref.fourier_modulus(sig[last], eps[last], tk[last]-DTC/2, tk[last], w, DTC)
    leak = abs(E1.real/E.real-1)
    assert (leak < 1e-12) if f == 8.0 else (leak > 1e-3)                      # 312.5 / 156.25 samples per period


def test_rod_transfer_is_the_pde_solution_and_matches_the_quasistatic_expansion():
    E = complex(ref.uniaxial_complex_modulus(2*math.pi*32, MU0, KAPPA, P2))
    w = 2*math.pi*32
    k = w*np.sqrt(RHO/E)
    u = lambda xx: np.sin(k*xx)/np.sin(k*L)                                    # noqa: E731  (u(0) = 0, u(L) = 1)
    h2 = 1e-4                                                                  # second difference: u'' = -k^2 u
    assert abs((u(0.011+h2)-2*u(0.011)+u(0.011-h2))/h2**2+k*k*u(0.011)) < 1e-6*abs(k*k*u(0.011))
    h = 1e-6
    gauge = (u(B_MARK)-u(A_MARK))/(B_MARK-A_MARK)
    for end, xe in ((1, L), (0, 0.0)):
        strain_end = (u(xe+h)-u(xe-h))/(2*h)
        assert abs(E*strain_end/gauge/(E*ref.rod_transfer(w, E, RHO, L, A_MARK, B_MARK, end))-1) < 1e-8
    for f in (0.5, 1.0, 2.0):                                                   # small kL: O(w^2) term
        wf = 2*math.pi*f
        E0 = 1.0e6
        c = RHO*wf*wf/E0
        T1 = ref.rod_transfer(wf, E0, RHO, L, A_MARK, B_MARK, 1)
        T0 = ref.rod_transfer(wf, E0, RHO, L, A_MARK, B_MARK, 0)
        m2 = (A_MARK**2+A_MARK*B_MARK+B_MARK**2)/6
        assert abs(T1-(1-c*(L*L/2-m2))) < 3*(c*L*L)**2
        assert abs(T0-(1+c*m2)) < 3*(c*L*L)**2


@pytest.mark.parametrize("end", [0, 1])
def test_deembedding_round_trip(end):
    for f in (1.0, 16.0, 32.0):
        w = 2*math.pi*f
        E = complex(ref.uniaxial_complex_modulus(w, MU0, KAPPA, P2))
        meas = E*ref.rod_transfer(w, E, RHO, L, A_MARK, B_MARK, end)
        back, it = ref.deembed_rod(meas, w, RHO, L, A_MARK, B_MARK, end)
        assert abs(back/E-1) < 1e-13 and it < 100
