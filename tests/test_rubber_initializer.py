"""R1.5: static-equilibrium initializer (ADR 0023 §7): convergence, reaction balance, CFL refusal."""
import math

import pytest

from robotic_cutting_sim import rubber_phys as P
from robotic_cutting_sim import rubber_reference as R
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_initializer import relax_to_static


def _live(cfl=None, substeps=None):
    spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2)
    return LiveRubberMaterial(spec, material=P.material(4e5), control_dt=2e-4, gravity=(0, 0, -9.81), cfl=cfl,
                              substeps=substeps)


def _omega1(live):
    m = live.material
    return 2*math.pi*R.cantilever_f1(m.density, 0.003, 0.003, 0.012, m.young_effective())


def test_mass_proportional_converges_and_balances_weight():
    live = _live()
    r = relax_to_static(live, omega1=_omega1(live))
    assert r["converged"] and r["initializer"] == "relax_mass_proportional_v1" and r["stop_rule"] == "static_stop_v2"
    assert r["final"]["reaction_residual"] < 1e-3
    assert live.solver.damping_rate == 0.0 and abs(live.solver.v.numpy()).max() == 0.0
    tip = live.scene.marker_reference[-1][2]-live.markers()[-1][2]
    assert tip > 0


def test_kinetic_damping_refused_above_cfl_025():
    live = _live()
    assert live.step_rule["cfl_actual"] > 0.25
    with pytest.raises(ValueError, match="refused at CFL"):
        relax_to_static(live, method="kinetic", omega1=_omega1(live))


def test_kinetic_damping_at_cfl_025_runs():
    live = _live(cfl=0.25)
    assert live.step_rule["cfl_actual"] <= 0.25
    r = relax_to_static(live, method="kinetic", omega1=_omega1(live), max_time=3*2*math.pi/_omega1(live))
    assert r["initializer"] == "relax_kinetic_v1" and not r["nonfinite"]
