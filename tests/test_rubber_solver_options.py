"""P2-2 opt-in options of ``tlmpm_rubber_v1`` (ADR 0023 §19; F8 / F9 / F10). Every default is the v1 launch
sequence (the existing L0 / v1 tests cover it); these tests check the options themselves:

- defaults allocate no option arrays and are reported as the v1 ids; an unknown option value is refused;
- ``dirichlet="consistent_mass_projection_v1"``: after a substep the MUSL momentum of the projected particle velocities
  vanishes on every clamped (node, component) to float32 round-off (the lumped v1 projection does not achieve it);
  the static clamp reaction of a relaxed cantilever balances the weight; the block inverses are exact;
- ``traction_work="grid_displacement_v1"``: the accumulated work equals sum_p f_t (m_p / M_t) . (x_end - x_start);
- ``state_precision="f64_shadow_v1"``: x and F are the float32 roundings of float64 shadows that reproduce the
  float32 path to float32 precision; reset restores the shadows;
- the batched path refuses lives built with a non-default option (no batched execution yet).
"""
import math

import numpy as np
import pytest

from robotic_cutting_sim import rubber_reference as R
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim import tlmpm_rubber_v1 as T
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.batched.rubber_layout import pack_rubber
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_initializer import relax_to_static
from robotic_cutting_sim.rubber_material import MaterialValues

MAT = MaterialValues(mu0=4e5, kappa=8e6, density=1150.0)


def _cantilever(**kw):
    spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2, case="A")
    return LiveRubberMaterial(spec, material=MAT, control_dt=1e-4, gravity=(0, 0, -200.0), **kw)


def _musl_momentum(live):
    s, sc = live.solver, live.scene
    m = s.mass.numpy().astype(np.float64)
    v = s.v.numpy().astype(np.float64)
    p = np.zeros((sc.node_count, 3))
    for c in range(8):
        np.add.at(p, sc.nodes[:, c], (sc.weights[:, c].astype(np.float64)*m)[:, None]*v)
    return p


def test_defaults_are_v1_and_unknown_values_are_refused():
    live = _cantilever()
    s = live.solver
    assert s.solver_receipt()["options"] == {"dirichlet": "lumped_v1", "traction_work": "particle_velocity_v1",
                                             "state_precision": "float32_v1", "projection": None}
    assert not any(name.startswith(("proj_", "F64", "x64", "work_scratch")) for name in vars(s))
    for key in ("dirichlet", "traction_work", "state_precision"):
        with pytest.raises(ValueError):
            _cantilever(**{key: "nope"})


def test_projection_meets_the_constraint_in_the_musl_metric():
    proj, lump = _cantilever(dirichlet="consistent_mass_projection_v1"), _cantilever()
    for b in proj.solver.projection_receipt["blocks"]:
        assert b["nodes"] > 0 and np.isfinite(b["condition"])
    blocks = T.projection_blocks(proj.scene, proj.solver.mass_value)
    for b in blocks:                      # the stored inverse is the inverse of the consistent-mass block
        sel, W = b["nodes"], proj.scene.weights.astype(np.float64)
        A = np.zeros((len(sel), len(sel)))
        idx = {int(n): i for i, n in enumerate(sel)}
        for p in range(proj.scene.n):
            hit = [(idx[int(n)], w) for n, w in zip(proj.scene.nodes[p], W[p]) if int(n) in idx]
            for a, wa in hit:
                for c, wc in hit:
                    A[a, c] += proj.solver.mass_value*wa*wc
        assert np.allclose(A@b["inv"], np.eye(len(sel)), atol=1e-9)
    clamp = proj.scene.node_group == 0
    worst = {}
    for name, live in (("proj", proj), ("lump", lump)):
        live.advance()
        p = _musl_momentum(live)
        worst[name] = np.abs(p[clamp]).max()/np.abs(p).max()
    assert worst["proj"] < 1e-5
    assert worst["lump"] > 100*worst["proj"]


def test_projection_static_reaction_balances_the_weight():
    live = _cantilever(dirichlet="consistent_mass_projection_v1", stable_step_guard=1.0)
    E = live.material.young_effective()
    f1 = R.cantilever_f1(live.material.density, 0.003, 0.003, 0.012, E)
    out = relax_to_static(live, omega1=2*math.pi*f1, stop_rule="static_stop_v3")
    assert out["converged"]
    assert out["final"]["reaction_residual"] < 1e-3
    assert live.invalid_reason == 0


def test_grid_displacement_traction_work():
    spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2, case="A",
                              traction_regions=(("end", (0.012-0.00075, -1.0, -1.0), (1.0, 1.0, 1.0)),))
    live = LiveRubberMaterial(spec, material=MAT, control_dt=2e-5, gravity=(0, 0, 0),
                              traction_work="grid_displacement_v1")
    s = live.solver
    f = np.array([0.0, 0.0, -0.05])
    sel = s.scene.particle_traction == 0
    m = s.mass.numpy().astype(np.float64)
    expected = 0.0
    for _ in range(3):
        s.begin_control_step()
        s.set_traction(0, f)
        for _ in range(s.substeps):
            s.step()
            dx = s.x.numpy().astype(np.float64)-s.x_start.numpy().astype(np.float64)
            expected += float(np.sum((m[sel]/m[sel].sum())[:, None]*f[None, :]*dx[sel]))
    assert float(s.work.numpy()[0]) == pytest.approx(expected, rel=1e-6)
    assert float(s.work.numpy()[0]) > 0


def test_f64_shadow_tracks_the_float32_path_and_resets():
    a, b = _cantilever(state_precision="f64_shadow_v1"), _cantilever()
    for _ in range(3):
        a.advance()
        b.advance()
    sa, sb = a.solver, b.solver
    assert np.array_equal(sa.F.numpy(), sa.F64.numpy().astype(np.float32))
    assert np.array_equal(sa.x.numpy(), sa.x64.numpy().astype(np.float32))
    disp = np.abs(sb.x.numpy()-sb.reference.numpy()).max()
    assert disp > 0
    ulp = np.spacing(np.float32(np.abs(sb.x.numpy()).max()))
    assert np.abs(sa.x.numpy()-sb.x.numpy()).max() <= 4*ulp            # same path up to float32 rounding
    assert np.abs(sa.F.numpy()-sb.F.numpy()).max() < 1e-5
    a.reset()
    assert np.array_equal(sa.F64.numpy(), np.tile(np.eye(3), (sa.n, 1, 1)))
    assert np.array_equal(sa.x64.numpy(), a.scene.X.astype(np.float64))


def test_batched_path_refuses_the_opt_in_options():
    for kw in ({"dirichlet": "consistent_mass_projection_v1"}, {"state_precision": "f64_shadow_v1"},
               {"traction_work": "grid_displacement_v1"}):
        with pytest.raises(UnsupportedBatchConfiguration):
            pack_rubber([_cantilever(**kw)])
