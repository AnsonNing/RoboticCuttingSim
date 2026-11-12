"""ADR 0023 §21.2 / R3.4: single-env ``TLMPMRubberVisco`` (``neo_hookean_jbar_prony_v1``).

- RUBBER-VISC-09 (single): prony = [] runs the frozen elastic pass, byte-identical to ``tlmpm_rubber_v1`` (with the
  same options, via ``TLMPMRubberCW``) for 5 control steps of case A, case C and the block; M = 2 with every g = 0 is
  byte-identical to the elastic run;
- the energy accumulators equal a host float64 recomputation every substep (dissipation from Q before / after, branch
  work from visco_pb and the MUSL grid velocity);
- reset() clears visco_q / visco_s_prev / accumulators; two episodes from reset() are byte-identical;
- the visco guard counts a forced overstress the elastic guard does not see;
- model selection in ``LiveRubberMaterial``; receipts; ``nodal_internal_force`` refused.
"""
import numpy as np
import pytest

from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim import rubber_visco_constitutive as rvc
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_constraint_work import TLMPMRubberCW
from robotic_cutting_sim.rubber_material import MaterialValues
from robotic_cutting_sim.tlmpm_rubber_v1 import TLMPMRubber
from robotic_cutting_sim.tlmpm_rubber_visco_v1 import TLMPMRubberVisco

OPT = {"dirichlet": "consistent_mass_projection_v1", "constraint_work": "kick_average_v1",
       "traction_work": "grid_displacement_v1", "state_precision": "f64_shadow_v1"}
P2 = ((0.2, 0.005), (0.15, 0.05))
ZERO = ((0.0, 0.005), (0.0, 0.05))
STEP_MATERIAL = MaterialValues(mu0=1e6, kappa=2e7, density=1000.0)


def _same_bytes(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _spec(case):
    if case == "block":
        return rs.RubberStripSpec(length=0.004, width=0.004, thickness=0.004, dx=0.001, case="block")
    kw = {"end_mask": "x"} if case == "C" else {}
    region = (("tip", (0.0105, -1, -1), (1, 1, 1)),) if case == "A" else (("mid", (0.005, -1, -1), (0.007, 1, 1)),)
    return rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2, case=case, traction_regions=region,
                              **kw)


def _live(case, prony=(), options=OPT, energy="off_v1", mu=4e5, **kw):
    m = MaterialValues(mu0=mu, kappa=20*mu, density=1150.0, prony=prony)
    return LiveRubberMaterial(_spec(case), material=m, control_dt=2e-4, gravity=(0, 0, -9.81), stable_step_guard=1.0,
                              step_material=STEP_MATERIAL, visco_energy=energy, **options, **kw)


def _inputs(live, k):
    from scipy.spatial.transform import Rotation
    case = live.spec.case
    if case == "block":
        A = np.eye(3)+(k+1)*np.array([[0.02, 0.01, 0.0], [0.0, -0.01, 0.005], [0.0, 0.0, -0.01]])
        return {0: {"affine": A}}, None
    motions = {0: {"translation": np.array([0.0, 1e-5*(k+1), 3e-5*(k+1)]),
                   "rotation": Rotation.from_rotvec([0.01*(k+1), 0.0, 0.0]).as_matrix()}}
    if case == "C":
        motions[1] = {"translation": np.array([2e-5*(k+1), 0.0, 0.0])}
    return motions, {0: np.array([1e-3, 0.0, -2e-3])}


def _visco_solver_like(live, prony, energy="off_v1"):
    """A TLMPMRubberVisco on the same scene / dt / options as ``live``'s solver."""
    s = live.solver
    m = MaterialValues(mu0=live.material.mu0, kappa=live.material.kappa, density=live.material.density, prony=prony)
    return TLMPMRubberVisco(s.scene, m, s.dt, substeps=s.substeps, gravity=s.gravity_tuple, j_domain=s.j_domain,
                            contact_stiffness=s.contact_stiffness, contact_radius_scale=live.recipe.contact_radius_scale,
                            stable_step_guard=1.0, visco_energy=energy, **OPT)


def _drive(solver, motions, traction):
    solver.begin_control_step()
    for g, m in motions.items():
        solver.set_group_motion(g, **m)
    for t, f in (traction or {}).items():
        solver.set_traction(t, f)
    for _ in range(solver.substeps):
        solver.step()
    solver.diagnostics.launch()                # as LiveRubberMaterial.advance (recomputes jbar from F)


@pytest.mark.parametrize("case", ["A", "C", "block"])
def test_visc09_empty_prony_is_the_frozen_elastic_solver(case):
    ref = _live(case)
    assert type(ref.solver) is TLMPMRubberCW
    vis = _visco_solver_like(ref, ())
    assert vis.M == 0 and vis.model_id == "neo_hookean_jbar_v1"
    for k in range(5):
        motions, traction = _inputs(ref, k)
        ref.advance(motions, traction=traction)
        _drive(vis, motions, traction)
        sa, sb = ref.solver.device_arrays(), vis.device_arrays()
        assert set(sb)-set(sa) == {"env_of_particle", "params64"}
        for name, arr in sa.items():
            assert _same_bytes(arr.numpy(), sb[name].numpy()), (case, k, name)
        assert _same_bytes(ref.solver.end_control_step(2e-4), vis.end_control_step(2e-4))


@pytest.mark.parametrize("case", ["A", "C", "block"])
def test_visc09_zero_g_is_byte_identical_to_the_elastic_run(case):
    ref = _live(case)
    zero = _live(case, prony=ZERO, energy="branch_work_v1")
    assert type(zero.solver) is TLMPMRubberVisco and zero.solver.M == 2
    for k in range(5):
        motions, traction = _inputs(ref, k)
        ref.advance(motions, traction=traction)
        zero.advance(motions, traction=traction)
        sa, sb = ref.solver.device_arrays(), zero.solver.device_arrays()
        for name, arr in sa.items():
            assert _same_bytes(arr.numpy(), sb[name].numpy()), (case, k, name)
        assert _same_bytes(ref.last_wrench, zero.last_wrench)
        assert _same_bytes(ref.solver.diagnostics.summary.numpy(), zero.solver.diagnostics.summary.numpy())
        assert not np.any(zero.solver.visco_q.numpy()) and zero.last_visco["dissipation"] == 0.0
        assert zero.last_visco["branch_work"] == 0.0
        assert zero.last_visco["psi_eq"] == ref.last_diagnostics["elastic"]          # g_inf = 1


def test_energy_accumulators_equal_a_host_recompute_every_substep():
    class Probe(TLMPMRubberVisco):
        def step(self):
            q0 = self.visco_q.numpy().copy()
            F0 = self.F.numpy().astype(np.float64)
            d0 = float(self.visco_dissipation.numpy()[0])
            w0 = float(self.visco_branch_work.numpy()[0])
            super().step()
            q1 = self.visco_q.numpy()
            terms = self.prony
            V, mu = self.volume, float(np.float32(self.material.mu0))
            host_d = 0.0
            for i, (g, tau) in enumerate(terms):
                a, b = q0[i::self.M], q1[i::self.M]
                sq = lambda x: np.sum(x[:, :3]**2, 1)+2*np.sum(x[:, 3:]**2, 1)      # noqa: E731
                host_d += np.sum(V*self.dt*(sq(a)+sq(b))/(4*g*mu*tau))
            pb = self.visco_pb.numpy()
            v = self.new_v.numpy().astype(np.float64)
            grad = np.einsum("pci,pcj->pij", v[self.scene.nodes], self.scene.gradients.astype(np.float64))
            host_w = float(np.sum(V*np.einsum("pij,pij->p", pb, self.dt*grad)))
            d1 = float(self.visco_dissipation.numpy()[0])
            w1 = float(self.visco_branch_work.numpy()[0])
            self.checks.append((d1-d0, host_d, d1, w1-w0, host_w, w1))
            pbq = np.einsum("pij,pjk->pik", F0, rvc.mat_from_sym6(
                q1.reshape(self.n, self.M, 6).sum(1)))
            assert np.allclose(pb, pbq, rtol=1e-12, atol=1e-12*max(np.abs(pb).max(), 1))

    live = _live("A", prony=P2, energy="branch_work_v1")
    s = live.solver
    probe = Probe(s.scene, live.material, s.dt, substeps=s.substeps, gravity=s.gravity_tuple, stable_step_guard=1.0,
                  visco_energy="branch_work_v1", **OPT)
    probe.checks = []
    for k in range(2):
        motions, traction = _inputs(live, k)
        _drive(probe, motions, traction)
    assert len(probe.checks) == 2*probe.substeps
    for dd, hd, d1, dw, hw, w1 in probe.checks:
        assert dd >= 0 and abs(dd-hd) <= 1e-10*abs(hd)+4*np.spacing(abs(d1))
        assert abs(dw-hw) <= 1e-10*abs(hw)+4*np.spacing(abs(w1))+1e-30
    assert any(c[1] > 0 for c in probe.checks) and any(abs(c[4]) > 0 for c in probe.checks)
    assert int(probe.visco_d_negative.numpy()[0]) == 0


def test_reset_clears_the_memory_and_episodes_repeat():
    live = _live("A", prony=P2, energy="branch_work_v1")
    names = ("visco_q", "visco_s_prev", "visco_pb", "visco_branch_work", "visco_dissipation", "visco_d_negative")

    def episode():
        live.reset()
        for name in names:
            assert not np.any(getattr(live.solver, name).numpy()), name
        snaps = []
        for k in range(3):
            motions, traction = _inputs(live, k)
            live.advance(motions, traction=traction)
            snaps.append({n: a.numpy().copy() for n, a in live.solver.device_arrays().items()})
        return snaps

    first, second = episode(), episode()
    assert np.any(first[-1]["visco_q"]) and first[-1]["visco_dissipation"][0] > 0
    for a, b in zip(first, second):
        for name in a:
            assert _same_bytes(a[name], b[name]), name


def test_visco_guard_counts_a_forced_overstress():
    live = _live("block", prony=P2, strict=False)
    s = live.solver
    q = s.visco_q.numpy()
    q[::2, 0] = 2e9                    # a huge positive overstress in xx of term 1
    s.visco_q.assign(q)
    s.step()
    assert int(s.cfl_violations.numpy()[0]) > 0
    elastic = _live("block", strict=False)
    elastic.solver.step()
    assert int(elastic.solver.cfl_violations.numpy()[0]) == 0


def test_model_selection_and_receipts():
    assert type(_live("A").solver) is TLMPMRubberCW
    assert type(_live("A", options={}).solver) is TLMPMRubber
    vis = _live("A", prony=P2, energy="branch_work_v1")
    s = vis.solver
    assert type(s) is TLMPMRubberVisco and s.M == 2 and s.params64.numpy().dtype == np.float64
    assert s.params64.numpy()[0].tobytes() == rvc.params64_row(P2, s.dt).tobytes()
    assert s.visco_q.numpy().dtype == np.float64 and s.visco_q.shape == (2*s.n,)
    r = s.solver_receipt()
    assert r["model"] == "neo_hookean_jbar_prony_v1" and r["options"]["visco_energy"] == "branch_work_v1"
    assert r["stable_step_guard"]["id"] == "stable_step_guard_visco_v1"
    assert r["prony"]["bytes_per_particle_state"] == 144
    assert "tlmpm_rubber_visco_v1.py" in r["source_sha256"] and "SIMULATOR_ONLY" in r["prony"]["calibration_status"]
    assert vis.step_rule["mu0_max_pa"] == STEP_MATERIAL.mu0
    with pytest.raises(NotImplementedError):
        s.nodal_internal_force()
    with pytest.raises(ValueError, match="visco_energy"):
        _live("A", prony=P2, energy="sometimes")
    with pytest.raises(ValueError, match="needs dirichlet"):
        _live("A", prony=P2, options={"constraint_work": "kick_average_v1"})
    vis.advance(*_inputs(vis, 0))
    assert set(vis.last_visco) >= {"psi_eq", "psi_visc_stored", "branch_work", "dissipation", "cfl_visco_max"}


def test_block_relaxation_follows_the_hereditary_integral():
    """One-control-step shear ramp of the block, then hold: every particle's PK1 equals the finite-strain hereditary
    integral at t_k - dt (the time of the last stress pass) to float32 accuracy."""
    from robotic_cutting_sim import rubber_visco_reference as ref
    prony = ((0.3, 0.002),)
    live = _live("block", prony=prony, options={"dirichlet": "consistent_mass_projection_v1",
                                                "state_precision": "f64_shadow_v1"})
    A = np.eye(3)
    A[0, 1] = 0.02
    block = ref.HereditaryBlock([0.0, 2e-4, 0.02], [np.eye(3), A, A], 4e5, 8e6, prony)
    for k in range(30):
        live.advance({0: {"affine": A}})
        P = live.solver.stress.numpy().astype(np.float64)
        Pr = block.evaluate((k+1)*2e-4-live.dt)
        assert np.abs(P-Pr).max() <= 1e-6*np.abs(Pr).max(), k
    assert P[:, 0, 1].mean() < 0.75*block.evaluate(2e-4)[0, 1]
