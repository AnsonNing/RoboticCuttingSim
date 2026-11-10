"""R0.3: analytic references (ADR 0023 §5, §10, E5 table)."""
import math

import numpy as np
import pytest

from robotic_cutting_sim import rubber_reference as R


def test_pk1_is_the_energy_derivative():
    rng = np.random.default_rng(0)
    mu, kappa = 5e5, 1e7
    worst = 0.0
    for _ in range(60):
        F = np.eye(3)+0.3*rng.standard_normal((3, 3))
        if np.linalg.det(F) < 0.3:
            continue
        G = np.zeros((3, 3))
        h = 1e-6
        for i in range(3):
            for j in range(3):
                E = np.zeros((3, 3))
                E[i, j] = h
                G[i, j] = (R.energy_density(F+E, mu, kappa)-R.energy_density(F-E, mu, kappa))/(2*h)
        worst = max(worst, np.abs(G-R.pk1(F, mu, kappa)).max()/np.abs(G).max())
    assert worst < 1e-7


def test_cofactor_finite_for_inverted_and_singular_F():
    F = np.diag([1.0, 1.0, -0.5])
    assert np.allclose(R.cofactor(F), np.linalg.det(F)*np.linalg.inv(F).T)
    S = np.diag([1.0, 1.0, 0.0])
    assert np.all(np.isfinite(R.cofactor(S)))


@pytest.mark.parametrize("beta,J,ratio", [
    (10, (1.024, 1.049, 1.103, 1.225), (0.949, 0.923)),
    (20, (1.012, 1.025, 1.054, 1.125), (0.973, 0.958)),
    (50, (1.005, 1.010, 1.023, 1.054), (0.989, 0.982)),
    (3000, (1.000, 1.000, 1.000, 1.001), (1.000, 1.000)),
])
def test_uniaxial_matches_adr_e5_table(beta, J, ratio):
    mu = 5e5
    for lam, expected in zip((1.25, 1.5, 2.0, 3.0), J):
        assert R.uniaxial_stress(lam, mu, beta*mu)["J"] == pytest.approx(expected, abs=6e-4)
    for lam, expected in zip((1.5, 2.0), ratio):
        got = R.uniaxial_stress(lam, mu, beta*mu)["nominal_stress"]/R.incompressible_uniaxial_nominal(lam, mu)
        assert got == pytest.approx(expected, abs=6e-4)


def test_uniaxial_state_has_zero_lateral_stress():
    mu, kappa = 4e5, 8e6
    for lam in (0.7, 1.5, 3.0):
        u = R.uniaxial_stress(lam, mu, kappa)
        sigma = R.cauchy(R.pk1(u["F"], mu, kappa), u["F"])
        assert abs(sigma[1, 1]) < 1e-6*mu and abs(sigma[2, 2]) < 1e-6*mu
        assert sigma[0, 0] == pytest.approx(u["cauchy_stress"], rel=1e-9)
        assert R.pk1(u["F"], mu, kappa)[0, 0] == pytest.approx(u["nominal_stress"], rel=1e-9)


def test_simple_shear_closed_form():
    mu, kappa = 4e5, 8e6
    for gamma in (0.5, 1.0, 2.0):
        s = R.simple_shear(gamma, mu)
        sigma = R.cauchy(R.pk1(s["F"], mu, kappa), s["F"])
        assert sigma[0, 1] == pytest.approx(s["sigma_12"], rel=1e-12)
        assert sigma[0, 0]-sigma[1, 1] == pytest.approx(s["normal_difference_12"], rel=1e-12)
        assert abs(sigma[1, 1]-sigma[2, 2]) < 1e-9*mu


def test_beam_closed_forms():
    # ADR E3 beam: L 100 mm, W = t = 10 mm, mu 2 MPa, beta 20, rho 1150 -> EB 2.867 mm, f1 11.57 Hz
    E = 9*40e6*2e6/(3*40e6+2e6)
    assert R.euler_bernoulli_sag(1150, 9.81, .01, .01, .1, E) == pytest.approx(2.8674e-3, rel=2e-4)
    assert R.cantilever_f1(1150, .01, .01, .1, E) == pytest.approx(11.572, rel=2e-4)
    assert R.BETA1_L == pytest.approx(1.8751, abs=1e-4)
    t = R.timoshenko_sag(1150, 9.81, .01, .01, .1, E, 0.4754)
    assert t > R.euler_bernoulli_sag(1150, 9.81, .01, .01, .1, E)
    lo, hi = R.plate_bracket(1.0, 0.4754)
    assert lo == pytest.approx(1-0.4754**2) and hi == 1.0


def test_elastica_limits_and_adr_e4_value():
    for alpha in (1e-3, 1e-2):
        assert R.elastica(alpha)["tip_drop"] == pytest.approx(alpha/8, rel=1e-3)
    e = R.elastica(9.17562)
    assert e["success"] and e["tip_drop"] == pytest.approx(0.6782, abs=2e-4)   # ADR E4: 67.82 mm at L 100 mm
    assert R.load_parameter(1150, 9.81, .005, .005, .1, 9*4e6*2e5/(3*4e6+2e5)) == pytest.approx(9.1756, rel=1e-4)


def test_prony_limits():
    terms = [(0.15, 0.01), (0.10, 0.1)]
    assert R.prony_relaxation(0.0, terms) == pytest.approx(1.0)
    assert R.prony_relaxation(1e3, terms) == pytest.approx(0.75)
    Gs0, Gl0 = R.prony_moduli(1e-9, 1.0, terms)
    Gsi, Gli = R.prony_moduli(1e13, 1.0, terms)
    assert Gs0 == pytest.approx(0.75) and Gl0 == pytest.approx(0.0, abs=1e-8)
    assert Gsi == pytest.approx(1.0) and Gli == pytest.approx(0.0, abs=1e-8)
    w = 1/0.01
    assert R.prony_moduli(w, 1.0, [(0.2, 0.01)])[1] == pytest.approx(0.1)
    assert R.prony_loss_per_cycle(w, 0.01, 2.0, [(0.2, 0.01)]) == pytest.approx(math.pi*1e-4*0.2)


def test_richardson_and_exponent_helpers():
    assert R.richardson_two_grid(0.9, 0.95, 2.0, 2.0) == pytest.approx(0.95+0.05/3)
    xs = np.array([1., 2., 4.])
    assert R.power_law_exponent(xs, 3*xs**4) == pytest.approx(4.0)
