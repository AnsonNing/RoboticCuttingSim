"""R2.2: grid-convergence arithmetic (rubber_gci; Celik et al. 2008 / Roache GCI) on manufactured sequences."""
import math

import pytest

from robotic_cutting_sim import rubber_gci as G


@pytest.mark.parametrize("p", [1.0, 1.7, 2.0, 3.0])
def test_observed_order_and_extrapolation_are_exact_for_a_power_law(p):
    h = (1/6, 1/4, 1/3)                      # n_t 6, 4, 3: non-uniform ratios 1.5, 1.333
    exact, C = 0.98, 0.37
    phi = [exact+C*hh**p for hh in h]
    o = G.observed_order(h, phi)
    assert o["converged"] and o["s"] == 1 and o["p"] == pytest.approx(p, rel=1e-9)
    assert G.extrapolated(phi[0], phi[1], o["r21"], o["p"]) == pytest.approx(exact, rel=1e-12)
    res = G.study(h, phi, p_bounds=(0.0, 10.0), verified_window=(0.0, 10.0))
    # with the true p, the GCI is Fs times the exact relative error of the fine / coarse solution
    assert res["gci_fine"] == pytest.approx(1.25*abs(phi[0]-exact)/abs(phi[0]), rel=1e-9)
    assert res["gci_coarse"] == pytest.approx(1.25*abs(phi[2]-exact)/abs(phi[2]), rel=1e-9)


def test_verdict_rule_clips_the_order_and_raises_the_safety_factor():
    h = (1/6, 1/4, 1/3)
    phi = [1.0+0.5*hh**4.0 for hh in h]       # p_obs = 4: outside the verified window [1, 3]
    res = G.study(h, phi)
    assert res["p_obs"] == pytest.approx(4.0, rel=1e-9) and res["p_eff"] == 2.0
    assert res["fs"] == 3.0 and not res["order_verified"]
    r21 = 1.5
    assert res["gci_fine"] == pytest.approx(3.0*abs((phi[1]-phi[0])/phi[0])/(r21**2-1), rel=1e-12)


def test_oscillatory_and_degenerate_sequences_are_flagged():
    h = (1/6, 1/4, 1/3)
    osc = G.study(h, [1.0, 1.01, 0.995])
    assert osc["status"] == "OSCILLATORY" and osc["fs"] == 3.0
    flat = G.study(h, [1.0, 1.0, 1.02])
    assert flat["status"] == "INCONCLUSIVE" and flat["gci_coarse"] is None
    with pytest.raises(ValueError):
        G.observed_order((1/3, 1/4, 1/6), [1, 1, 1])


def test_uniform_ratio_closed_form():
    # constant r: p = ln(e32 / e21) / ln r without iteration
    h = (1.0, 2.0, 4.0)
    phi = (1.0, 1.1, 1.5)
    o = G.observed_order(h, phi)
    assert o["p"] == pytest.approx(math.log(0.4/0.1)/math.log(2.0), rel=1e-12)
