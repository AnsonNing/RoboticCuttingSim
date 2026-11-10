"""R1.2: single-env reference solver tlmpm_rubber_v1 (ADR 0023 §5, §7, §9)."""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import rubber_phys as P
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.tlmpm_rubber_v1 import TLMPMRubber


def _mat(mu=4e5):
    return P.material(mu)


def test_refusals():
    sc = rs.compile_strip(rs.RubberStripSpec(length=0.006, width=0.003, thickness=0.003, n_t=2))
    with pytest.raises(ValueError, match="pic"):
        TLMPMRubber(sc, _mat(), 1e-6, pic=0.05)
    with pytest.raises(ValueError, match="dt"):
        TLMPMRubber(sc, _mat(), float("nan"))
    s = TLMPMRubber(sc, _mat(), 1e-6)
    with pytest.raises(ValueError, match="refused in dynamics"):
        s.set_initializer_damping(10.0)
    s.set_initializer_damping(10.0, initializer=True)
    assert s.damping_rate == 10.0


@pytest.mark.parametrize("case", [{"id": "uniaxial_1.25", "kind": "uniaxial_stress", "stretch": 1.25},
                                  {"id": "shear_1", "kind": "simple_shear", "gamma": 1.0}])
def test_patch_rubber_phys_01_case(case):
    r = P.case_patch(case)
    assert r["pk1_max_error_over_allowed"] <= 1.0
    assert r["jbar_consistency_max"] <= 1e-6 and r["jbar_analytic_max"] <= 1e-5
    assert r["interior_force_ratio"] <= 1e-5
    assert r["objectivity_max_relative"] <= 1e-5


def test_fully_held_block_reaction_equals_weight():
    spec = rs.RubberStripSpec(length=0.004, width=0.004, thickness=0.004, dx=0.001, case="block")
    live = LiveRubberMaterial(spec, material=_mat(), control_dt=1e-4, gravity=(0, 0, -9.81))
    w = live.advance()
    weight = live.total_mass*9.81
    assert w.shape == (1, 6)
    assert w[0, 2] == pytest.approx(-weight, rel=1e-6)
    assert abs(w[0, 0]) < 1e-9*weight and abs(w[0, 1]) < 1e-9*weight
    assert np.abs(live.solver.x.numpy()-live.scene.X).max() == 0.0


def test_energy_conservation_2k_substeps():
    """Clamped strip released straight under gravity, pure FLIP, no damping: E_total = KE + Psi + E_grav."""
    spec = rs.RubberStripSpec(length=0.03, width=0.003, thickness=0.003, n_t=3)
    live = LiveRubberMaterial(spec, material=_mat(2e6), control_dt=2.6e-4, gravity=(0, 0, -9.81), record=True)
    for _ in range(20):
        live.advance()
    assert live.substeps*20 >= 2000
    rec = live.records[1:]
    psi = max(r["energies_j"]["elastic"] for r in rec)
    drift = max(abs(r["energies_j"]["total"]) for r in rec)
    assert psi > 0 and drift <= 1e-3*psi
    assert all(r["energies_j"]["external_work"] == 0.0 for r in rec)   # a fixed clamp does no work


def test_rigid_jaw_rotation_is_followed_exactly():
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.004, n_t=4)
    live = LiveRubberMaterial(spec, material=_mat(), control_dt=2e-4, gravity=(0, 0, 0))
    sc = live.scene
    o = np.array([0.0, 0.002, 0.002])
    for k in range(1, 6):
        Rm = Rotation.from_rotvec([0, 0, np.deg2rad(2*k)]).as_matrix()
        live.advance({0: {"rotation": Rm, "translation": o-Rm@o}})
    Rm = Rotation.from_rotvec([0, 0, np.deg2rad(10)]).as_matrix()
    inner = np.flatnonzero(sc.X[:, 0] < -sc.dx*1.2)       # particles whose whole stencil is held
    x = live.solver.x.numpy()[inner].astype(np.float64)
    expected = (sc.X[inner].astype(np.float64)-o)@Rm.T+o
    assert np.abs(x-expected).max() < 2e-7
    F = live.solver.F.numpy()[inner].astype(np.float64)
    assert np.abs(F-Rm).max() < 2e-5
    assert int(live.solver.index.numpy()[0]) == live.substeps


def test_traction_group_force_balance():
    spec = rs.RubberStripSpec(length=0.004, width=0.004, thickness=0.004, dx=0.001, case="free",
                              traction_regions=(("all", (-1, -1, -1), (1, 1, 1)),))
    live = LiveRubberMaterial(spec, material=_mat(), control_dt=1e-4, gravity=(0, 0, 0))
    f = np.array([0.01, 0.0, 0.0])
    live.advance(traction={0: f})
    v = live.solver.v.numpy().astype(np.float64)
    momentum = (v*live.solver.mass_value).sum(0)
    assert momentum == pytest.approx(f*live.control_dt, rel=1e-4)
