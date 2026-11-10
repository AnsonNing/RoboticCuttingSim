"""ADR 0023 §21.1: ``constraint_work`` (bookkeeping of the F8 projected Dirichlet constraint), single env and batched.

- the default builds the frozen ``tlmpm_rubber_v1.TLMPMRubber`` (no new array, no new launch);
- ``kick_average_v1`` changes no state array (byte-wise) and only the number in ``work``; the v1 end-velocity
  constraint work is reported in ``work_end_velocity`` (byte-equal to the default run's ``work`` without traction);
- the kernel equals a host float64 evaluation of Lambda . (g_n + v_presc) / 2 every substep;
- a moving grip closes its energy books with the kick average (step profile: the v1 accounting is off by ~19 %);
- lumped + kick average and unknown ids are refused; mixed values in one batch are refused;
- batched L0 byte-wise (N = 1, N = 3 heterogeneous, case C two grips with rotation and an inactive grip), every array
  every control step; masked partial reset of the new array.
"""
import numpy as np
import pytest

from robotic_cutting_sim import rubber_constraint_work as rcw
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.batched.rubber_layout import GROUP_ARRAYS, NODE_ARRAYS, PARTICLE_ARRAYS, TRACTION_ARRAYS
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_material import MaterialValues
from robotic_cutting_sim.tlmpm_rubber_v1 import TLMPMRubber

F8 = {"dirichlet": "consistent_mass_projection_v1"}
KICK = {**F8, "constraint_work": "kick_average_v1"}
KICK_ALL = {**KICK, "traction_work": "grid_displacement_v1", "state_precision": "f64_shadow_v1"}
STEP_MATERIAL = MaterialValues(mu0=1e6, kappa=2e7, density=1000.0)
HETERO = [dict(), dict(L=0.009, W=0.006, mu=7e5, rho=1050.0), dict(L=0.015, mu=3e5, rho=1200.0)]
OPTION_NAMES = ("work_end_velocity", "work_scratch", "F64", "x64", "proj_row_node", "proj_row_comp", "proj_row_group",
                "proj_row_block", "proj_row_local", "proj_rhs", "proj_vpresc", "proj_mu", "proj_block_offset",
                "proj_block_size", "proj_inv_offset", "proj_inv", "proj_mu_node")


def _same_bytes(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _live(options, L=0.012, W=0.003, t=0.003, n_t=2, mu=4e5, rho=1150.0, case="A", traction=True, guard=1.0,
          control_dt=2e-4):
    kw = {}
    if traction:
        region = (("tip", (L-0.0015, -1, -1), (1, 1, 1)),) if case == "A" else \
            (("mid", (L/2-0.00075, -1, -1), (L/2+0.00075, 1, 1)),)
        kw["traction_regions"] = region
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=n_t, case=case,
                              **({"end_mask": "x"} if case == "C" else {}), **kw)
    return LiveRubberMaterial(spec, material=MaterialValues(mu0=mu, kappa=20*mu, density=rho), control_dt=control_dt,
                              gravity=(0, 0, -9.81), step_material=STEP_MATERIAL, stable_step_guard=guard, **options)


def _motion(k, e, G):
    from scipy.spatial.transform import Rotation
    tr = np.zeros((G, 3))
    R = np.tile(np.eye(3), (G, 1, 1))
    tr[0] = [0.0, 1e-5*(k+1)*(e+1), 3e-5*(k+1)]
    R[G-1] = Rotation.from_rotvec([0.01*(k+1)*(e+1), 0.005*(k+1), 0.0]).as_matrix()
    if G > 1:
        tr[1] = [2e-5*(k+1)*(e+1), 0.0, 0.0]
    return tr, R


def _advance_single(live, k, e=0, active=None):
    G = live.solver.G
    tr, R = _motion(k, e, G)
    motions = {g: {"translation": tr[g], "rotation": R[g]} for g in range(G)}
    if active is not None:
        for g in range(G):
            motions[g]["active"] = bool(active[g])
    traction = {0: np.array([1e-3*(e+1), 0.0, -2e-3])} if live.solver.Gt else None
    live.advance(motions, traction=traction)


def test_default_builds_the_frozen_solver_class():
    assert type(_live(F8).solver) is TLMPMRubber
    assert type(_live({}).solver) is TLMPMRubber
    s = _live(KICK).solver
    assert type(s) is rcw.TLMPMRubberCW and s.constraint_work == "kick_average_v1"
    receipt = s.solver_receipt()
    assert receipt["options"]["constraint_work"] == "kick_average_v1"
    assert "rubber_constraint_work.py" in receipt["source_sha256"] and receipt["solver"] == "tlmpm_rubber_v1"


def test_refusals():
    with pytest.raises(ValueError, match="needs dirichlet"):
        _live({"constraint_work": "kick_average_v1"})
    with pytest.raises(ValueError, match="not in"):
        _live({**F8, "constraint_work": "midpoint"})
    live = _live(F8)
    with pytest.raises(ValueError, match="needs dirichlet"):
        rcw.TLMPMRubberCW(live.scene, live.material, live.dt, constraint_work="kick_average_v1")


@pytest.mark.parametrize("extra", [{}, {"state_precision": "f64_shadow_v1"}], ids=["f32", "f64_shadow"])
def test_kick_average_changes_only_the_work(extra):
    a = _live({**F8, **extra}, traction=False)
    b = _live({**KICK, **extra}, traction=False)
    for k in range(4):
        _advance_single(a, k)
        _advance_single(b, k)
        sa, sb = a.solver.device_arrays(), b.solver.device_arrays()
        assert set(sb) - set(sa) == {"work_end_velocity", "env_of_group"} and set(sa) <= set(sb)
        for name, arr in sa.items():
            if name != "work":
                assert _same_bytes(arr.numpy(), sb[name].numpy()), (k, name)
        assert _same_bytes(sa["work"].numpy(), sb["work_end_velocity"].numpy()), k
        assert _same_bytes(a.last_wrench, b.last_wrench)
        assert _same_bytes(a.solver.diagnostics.summary.numpy(), b.solver.diagnostics.summary.numpy())
    assert sa["work"].numpy()[0] != sb["work"].numpy()[0]


def test_kernel_equals_host_reference_every_substep():
    class Probe(rcw.TLMPMRubberCW):
        def _launch_projection(self):
            old = self.old.numpy().copy()
            before = float(self.work.numpy()[0])
            super()._launch_projection()
            rows = (self.proj_row_node.numpy(), self.proj_row_comp.numpy(), self.proj_row_group.numpy())
            host = rcw.host_kick_average_work(rows, self.proj_mu.numpy(), self.proj_vpresc.numpy(), self.gm.numpy(),
                                              old, self.group_active.numpy())
            after = float(self.work.numpy()[0])
            self.checks.append((after-before, host, after))

    live = _live(KICK, traction=False)
    probe = Probe(live.scene, live.material, live.dt, substeps=live.substeps, gravity=(0, 0, -9.81),
                  stable_step_guard=1.0, constraint_work="kick_average_v1", **F8)
    probe.checks = []
    for k in range(3):
        tr, R = _motion(k, 0, 1)
        probe.begin_control_step()
        probe.set_group_motion(0, translation=tr[0], rotation=R[0])
        for _ in range(probe.substeps):
            probe.step()
    assert len(probe.checks) == 3*probe.substeps
    for got, host, after in probe.checks:      # increment of a running float64 sum: tolerance from its ulp
        assert abs(got-host) <= 1e-12*abs(host)+4*np.spacing(abs(after))
    assert any(abs(h) > 0 for _, h, _ in probe.checks)


def _moving_grip_residuals(options, profile, n_move=60):
    """The diagnostic scene of ``scripts/rubber/p2_3/diag_moving_clamp_energy.py`` (step / smooth profile)."""
    from scipy.spatial.transform import Rotation
    spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2, case="A")
    live = LiveRubberMaterial(spec, material=MaterialValues(mu0=4e5, kappa=8e6, density=1150.0), control_dt=1e-4,
                              gravity=(0, 0, -9.81), stable_step_guard=1.0, **options)
    mech, work_v1, work = [], [], []
    for k in range(2*n_move):
        tau = min(k+1, n_move)/n_move
        sv = tau if profile == "step" else tau**3*(10-15*tau+6*tau*tau)
        live.advance({0: {"rotation": Rotation.from_rotvec([0.15*sv, 0, 0]).as_matrix(),
                          "translation": np.array([0.0, 0.0, 1e-3*sv])}})
        d = live.last_diagnostics
        mech.append([d["elastic"], d["kinetic_particle"], d["gravity"]])
        work.append(float(live.solver.work.numpy()[0]))
        work_v1.append(float(getattr(live.solver, "work_end_velocity", live.solver.work).numpy()[0]))
    mech = np.array(mech)
    scale = np.max(np.abs(mech[:, 0])+np.abs(mech[:, 1]))
    total = mech.sum(1)
    return (total-np.array(work))/scale, (total-np.array(work_v1))/scale, live


@pytest.mark.parametrize("profile", ["step", "smooth"])
def test_moving_grip_books_close_with_the_kick_average(profile):
    res, res_v1, live = _moving_grip_residuals(KICK, profile)
    assert live.invalid_reason == 0
    assert np.abs(res).max() <= 0.01                          # O(dt) symplectic fluctuation (0.38 % / 0.23 %)
    v1_end = res_v1[-1]
    assert (v1_end < -0.15) if profile == "step" else (-0.03 < v1_end < -0.01)    # the v1 quadrature bias


def _compare(batch, e, live):
    snap = batch.solver.snapshot(e)
    s = live.solver
    G, Gt = s.G, s.Gt
    names = PARTICLE_ARRAYS+NODE_ARRAYS+GROUP_ARRAYS+TRACTION_ARRAYS+("cell_node", "cell_offsets", "cell_particles",
                                                                       "index", "work", "violations", "cfl_violations")
    option_names = [n for n in OPTION_NAMES if hasattr(s, n)]
    assert set(option_names) == {n for n in OPTION_NAMES if n in snap}, "option array set"
    for name in names+tuple(option_names):
        a = getattr(s, name).numpy()
        if name in GROUP_ARRAYS:
            a = a[:G]
        elif name in TRACTION_ARRAYS:
            a = a[:Gt]
        elif name.startswith("proj_row") or name in ("proj_rhs", "proj_vpresc", "proj_mu"):
            a = a[:s.proj_rows]
        assert _same_bytes(snap[name], a), name
    assert _same_bytes(batch.summary.numpy()[e], s.diagnostics.summary.numpy()[0]), "diagnostics"
    K = batch.K
    assert _same_bytes(batch.marker_x.numpy()[e*K:(e+1)*K], s.diagnostics.marker_x.numpy()), "markers"
    assert _same_bytes(batch.wrench.numpy()[e, :G], live.last_wrench), "wrench"


def _run_l0(options, kws, steps=3, case="A", inactive_step=None):
    singles = [_live(options, case=case, **kw) for kw in kws]
    batch = rmb.BatchedRubberMaterial([_live(options, case=case, **kw) for kw in kws], device="cpu",
                                      parity_mode=True)
    N, G = batch.num_envs, batch.G
    for k in range(steps):
        tr = np.zeros((N, G, 3))
        R = np.zeros((N, G, 3, 3))
        for e in range(N):
            tr[e], R[e] = _motion(k, e, G)
        active = np.ones((N, G), np.int32)
        if inactive_step == k and N > 1:
            active[1, G-1] = 0
        force = np.zeros((N, 3))
        force[:, 0] = 1e-3*np.arange(1, N+1)
        force[:, 2] = -2e-3
        batch.advance(translation=tr, rotation=R, active=active, traction=force)
        for e, live in enumerate(singles):
            _advance_single(live, k, e, active=active[e])
            _compare(batch, e, live)
    return batch, singles


@pytest.mark.parametrize("options", [KICK, KICK_ALL], ids=["F8_kick", "F8_F9_F10_kick"])
def test_l0_batched_single_and_heterogeneous(options):
    batch, singles = _run_l0(options, [dict()])
    assert batch.receipt()["solver"]["options"]["constraint_work"] == "kick_average_v1"
    batch, singles = _run_l0(options, HETERO)
    assert len(set(s.solver.proj_rows for s in singles)) > 1
    assert all(abs(float(s.solver.work_end_velocity.numpy()[0])) > 0 for s in singles)


def test_l0_two_grips_rotation_inactive_grip():
    kws = [dict(), dict(L=0.009, mu=6e5, rho=1050.0), dict(L=0.015, mu=3e5)]
    batch, _ = _run_l0(KICK_ALL, kws, steps=3, case="C", inactive_step=1)
    assert batch.G == 2


def test_partial_reset_clears_the_reported_v1_work():
    batch = rmb.BatchedRubberMaterial([_live(KICK_ALL, **kw) for kw in HETERO], device="cpu", parity_mode=True)
    fresh = rmb.BatchedRubberMaterial([_live(KICK_ALL, **kw) for kw in HETERO], device="cpu", parity_mode=True)
    for k in range(2):
        tr = np.stack([_motion(k, e, 1)[0] for e in range(3)])
        batch.advance(translation=tr)
    before = {e: batch.solver.snapshot(e) for e in (0, 2)}
    assert all(abs(before[e]["work_end_velocity"][0]) > 0 for e in before)
    batch.reset([1])
    for name, value in batch.solver.snapshot(1).items():
        if name != "index":
            assert _same_bytes(value, fresh.solver.snapshot(1)[name]), name
    for e in (0, 2):
        for name, value in batch.solver.snapshot(e).items():
            if name != "index":
                assert _same_bytes(value, before[e][name]), (e, name)


def test_vars_complete_reset_classification_with_kick_average():
    batch = rmb.BatchedRubberMaterial([_live(KICK_ALL)], device="cpu", parity_mode=True)
    names = set(batch.solver.device_arrays())
    declared = set(rmb.SOLVER_RESET_ARRAYS+rmb.SOLVER_CONSTANT_ARRAYS+rmb.SOLVER_SHARED_ARRAYS
                   + rmb.SOLVER_RECOMPUTED_ARRAYS)
    for table in (rmb.OPTION_RESET_ARRAYS, rmb.OPTION_CONSTANT_ARRAYS):
        for value in table.values():
            declared |= set(value)
    assert names == declared, names ^ declared


def test_mixed_constraint_work_is_refused():
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial([_live(F8), _live(KICK)], device="cpu", parity_mode=True)
