"""ADR 0023 §21.2 / R3.1-R3.3: ``neo_hookean_jbar_prony_v1`` functions, recipe values and analytic references.

- S0 = 2 dPsi_iso/dC (float64 finite differences); the Warp S0 equals its numpy twin; objectivity (Q lives in the
  reference frame: S0(RF) = S0(F), P(RF) = R P(F));
- the exponential update is exact for S0 linear in time; the Warp update equals the numpy recurrence;
- a_i, b_i in params64 are the host float64 expm1 values (byte-equal);
- every g_i = 0: P byte-equal to ``rubber_constitutive.nh_jbar_pk1`` (random F), Q untouched;
- overstress form = F (g_inf S0 + sum Q) + p cof F; visco guard (g = 0: byte-equal to the elastic guard);
- recipe ``rubber_generic_visco_v1`` (illustrative, SIMULATOR_ONLY_UNCALIBRATED), refusals, the mu0 step rule
  (RUBBER-VISC-08a), padding;
- references: SLS / P2 limits, E * J_E interconversion, declared design values, hereditary integral vs closed forms,
  beam root (elastic limit at g = 0).
"""
import math

import numpy as np
import pytest
from scipy import integrate
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import rubber_constitutive as rc
from robotic_cutting_sim import rubber_material as rm
from robotic_cutting_sim import rubber_visco_constitutive as rvc
from robotic_cutting_sim import rubber_visco_reference as ref

MU, KAPPA = 4.0e5, 8.0e6
P2 = ((0.2, 0.005), (0.15, 0.05))
SLS = ((0.3, 0.01),)
DT = 5.7142857142857145e-06
VISCO_RECIPE = rm.ROOT/"configs"/"materials"/"rubber_generic_visco_v1.json"


def _random_F(rng, n, stretch=(0.6, 1.6), shear=0.8):
    out = []
    for _ in range(n):
        lam = rng.uniform(*stretch, 3)
        U = np.diag(lam)
        S = np.eye(3)
        S[0, 1], S[0, 2], S[1, 2] = rng.uniform(-shear, shear, 3)
        out.append(Rotation.random(random_state=rng).as_matrix()@U@S@Rotation.random(random_state=rng).as_matrix())
    return np.array(out)


def _psi_iso_C(C, mu):
    J = math.sqrt(np.linalg.det(C))
    return 0.5*mu*(J**(-2/3)*np.trace(C)-3)


def test_s0_is_twice_the_derivative_of_the_isochoric_energy():
    rng = np.random.default_rng(1)
    for F in _random_F(rng, 20):
        C = F.T@F
        S_fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                h = 1e-6*np.abs(C).max()
                E = np.zeros((3, 3))
                E[i, j] += h/2
                E[j, i] += h/2
                S_fd[i, j] = 2*(_psi_iso_C(C+E, MU)-_psi_iso_C(C-E, MU))/(2*h)
        assert np.allclose(rvc.s0_np(F, MU), S_fd, rtol=1e-6, atol=1e-6*MU)
    F32 = _random_F(rng, 50).astype(np.float32)
    warp = rvc.evaluate_s0(F32, MU)
    twin = rvc.sym6(rvc.s0_np(F32.astype(np.float64), MU))
    scale = np.abs(twin).max(1, keepdims=True)          # float64 evaluation-order differences (cancellation)
    assert np.all(np.abs(warp-twin) <= 1e-9*scale)
    assert np.abs(rvc.evaluate_s0(np.eye(3)[None], MU)).max() == 0.0          # stress-free reference


def test_objectivity_reference_frame_state():
    rng = np.random.default_rng(2)
    terms = list(P2)
    coeffs = rvc.prony_coefficients(terms, 1e-3)
    for F in _random_F(rng, 10):
        R = Rotation.random(random_state=rng).as_matrix()
        assert np.allclose(rvc.s0_np(R@F, MU), rvc.s0_np(F, MU), rtol=1e-12, atol=1e-9)
        q0 = rng.normal(0, 1e4, (2, 3, 3))
        q0 = q0+np.swapaxes(q0, 1, 2)
        sp = rvc.s0_np(np.eye(3)+0.1*rng.normal(size=(3, 3)), MU)
        P, q, _ = rvc.prony_pk1_np(F, np.linalg.det(F), MU, KAPPA, terms, coeffs, q0, sp)
        Pr, qr, _ = rvc.prony_pk1_np(R@F, np.linalg.det(F), MU, KAPPA, terms, coeffs, q0, sp)
        assert np.allclose(qr, q, rtol=1e-10, atol=1e-6)
        assert np.allclose(Pr, R@P, rtol=1e-10, atol=1e-5)


def test_update_is_exact_for_linear_s0():
    g, tau, dt, rate = 0.3, 0.01, 1e-3, 7.0e5
    (a, b), = rvc.prony_coefficients([(g, tau)], dt)
    q, n = 0.0, 50
    for k in range(n):
        q = a*q+b*rate*dt                     # S0(t) = rate t on every component
    t = n*dt
    assert q == pytest.approx(g*rate*tau*(1-math.exp(-t/tau)), rel=1e-13)


def test_warp_update_equals_the_numpy_recurrence():
    rng = np.random.default_rng(3)
    row = rvc.params64_row(P2, DT)
    coeffs = rvc.prony_coefficients(P2, DT)
    path = [np.eye(3)+s*np.array([[0.3, 0.2, 0], [0, -0.1, 0.05], [0.02, 0, -0.12]]) for s in np.linspace(0, 1, 30)]
    F32 = np.array(path, np.float32)
    q = np.zeros((1, 2, 6))
    sp = np.zeros((1, 6))
    qn = np.zeros((2, 3, 3))
    spn = np.zeros((3, 3))
    for F in F32:
        out = rvc.evaluate_prony(F[None], [np.linalg.det(F.astype(np.float64))], MU, KAPPA, row, q, sp, DT)
        q, sp = out["q"], out["s0"]
        s0 = rvc.s0_np(F.astype(np.float64), MU)
        qn = np.array([a*qi+b*(s0-spn) for (a, b), qi in zip(coeffs, qn)])
        spn = s0
        assert np.all(np.abs(q[0]-rvc.sym6(qn)) <= 1e-9*max(np.abs(s0).max(), 1.0))
        assert out["dissipation_density"][0] >= 0 and out["negative"][0] == 0
    assert rng is not None


def test_params64_coefficients_are_the_host_expm1_values():
    for terms, dt in ((P2, DT), (SLS, 1e-6), (((0.1, 10.0),), 5e-6)):
        row = rvc.params64_row(terms, dt)
        assert row.dtype == np.float64 and row.shape == (1+4*len(terms),)
        assert row[0] == 1.0-math.fsum(g for g, _ in terms)
        for i, (g, tau) in enumerate(terms):
            a, b = math.exp(-dt/tau), -g*(tau/dt)*math.expm1(-dt/tau)
            assert row[1+4*i:5+4*i].tobytes() == np.array([g, tau, a, b], np.float64).tobytes()
    recipe = rm.load_recipe(VISCO_RECIPE)
    assert recipe.prony_coefficients(DT) == rvc.prony_coefficients(recipe.prony_terms(), DT)
    m = recipe.nominal()
    assert m.params64(DT).tobytes() == rvc.params64_row(recipe.prony_terms(), DT).tobytes()


def test_zero_g_is_byte_equal_to_the_elastic_pk1():
    rng = np.random.default_rng(4)
    F = _random_F(rng, 200).astype(np.float32)
    jbar = np.linalg.det(F.astype(np.float64))*rng.uniform(0.95, 1.05, len(F))
    row = rvc.params64_row(((0.0, 0.005), (0.0, 0.05)), DT)
    q = rng.normal(0, 1e4, (len(F), 2, 6))
    sp = rng.normal(0, 1e4, (len(F), 6))
    out = rvc.evaluate_prony(F, jbar, MU, KAPPA, row, q, sp, DT)
    elastic, _ = rc.evaluate_pk1(F, jbar, MU, KAPPA)
    assert out["P"].dtype == elastic.dtype and out["P"].tobytes() == elastic.tobytes()
    assert out["q"].tobytes() == q.tobytes()                       # g = 0 terms are skipped
    assert np.all(out["dissipation_density"] == 0)


def test_overstress_form_and_warp_twin():
    rng = np.random.default_rng(5)
    terms, coeffs = list(P2), rvc.prony_coefficients(P2, DT)
    F = _random_F(rng, 40, stretch=(0.8, 1.3), shear=0.3)
    F32 = F.astype(np.float32)
    J = np.linalg.det(F32.astype(np.float64))
    q = rng.normal(0, 3e4, (len(F), 2, 3, 3))
    q = q+np.swapaxes(q, -1, -2)
    sp = rvc.s0_np(_random_F(rng, len(F), stretch=(0.9, 1.1), shear=0.1), MU)
    out = rvc.evaluate_prony(F32, J, MU, KAPPA, rvc.params64_row(P2, DT), rvc.sym6(q), rvc.sym6(sp), DT)
    for k in range(len(F)):
        Fk = F32[k].astype(np.float64)
        P, qn, s0 = rvc.prony_pk1_np(Fk, J[k], MU, KAPPA, terms, coeffs, q[k], sp[k])
        S = (1-0.35)*s0+qn.sum(0)
        p = 0.5*KAPPA*(J[k]-1/J[k])
        assert np.allclose(P, Fk@S+p*rvc.cofactor_np(Fk), rtol=1e-10, atol=1e-6)
        assert np.allclose(out["P"][k], P, rtol=2e-6, atol=2e-6*np.abs(P).max())
        tol = 1e-9*max(np.abs(s0).max(), np.abs(qn).max())     # float64 evaluation-order differences of S0
        assert np.all(np.abs(out["q"][k]-rvc.sym6(qn)) <= tol)
        assert np.all(np.abs(out["qsum"][k]-rvc.sym6(qn.sum(0))) <= 2*tol)
        dens = sum(DT*(np.sum(a*a)+np.sum(b*b))/(4*g*MU*tau) for (g, tau), a, b in zip(terms, q[k], qn))
        assert out["dissipation_density"][k] == pytest.approx(dens, rel=1e-8)


def test_visco_guard():
    rng = np.random.default_rng(6)
    F = _random_F(rng, 100, stretch=(0.7, 1.4), shear=0.5).astype(np.float32)
    J = np.linalg.det(F.astype(np.float64))
    dtdx, rho = 5e-6/1e-3, 1150.0
    row0 = rvc.params64_row(((0.0, 0.005), (0.0, 0.05)), DT)
    q = rng.normal(0, 5e4, (len(F), 2, 6))
    s0 = rng.normal(0, 5e4, (len(F), 6))
    zero = rvc.evaluate_visco_cfl(F, J, MU, KAPPA, rho, dtdx, row0, q, s0)
    elastic = rc.evaluate_cfl_effective(F, J, MU, KAPPA, rho, dtdx)
    assert zero.tobytes() == elastic.tobytes()
    row = rvc.params64_row(P2, DT)
    visco = rvc.evaluate_visco_cfl(F, J, MU, KAPPA, rho, dtdx, row, q, s0)
    assert np.all(visco >= elastic)
    terms, coeffs = list(P2), rvc.prony_coefficients(P2, DT)
    for k in range(10):
        qm = rvc.mat_from_sym6(q[k])
        w = rvc.visco_wave_modulus_bound_np(F[k], J[k], MU, KAPPA, terms, coeffs, qm, rvc.mat_from_sym6(s0[k]))
        assert visco[k] == pytest.approx(dtdx*math.sqrt(w/rho), rel=2e-5)
    big = np.zeros_like(q)
    big[:, 0, 0] = 1e9                    # a forced overstress raises the bound
    assert np.all(rvc.evaluate_visco_cfl(F, J, MU, KAPPA, rho, dtdx, row, big, np.zeros_like(s0)) > 1.0)


# ----------------------------------------------------------------------------- R3.2 recipe / material values

def test_visco_recipe_values_and_status():
    r = rm.load_recipe(VISCO_RECIPE)
    assert r.model == rm.PRONY_MODEL and r.calibration_status == "SIMULATOR_ONLY_UNCALIBRATED"
    assert "NOT fitted" in r.provenance["values"]
    m = r.nominal()
    assert m.prony == ((0.15, 0.01), (0.1, 0.1)) and m.g_inf == pytest.approx(0.75)
    assert m.mu0 == pytest.approx(r.shear_modulus_eq.value/0.75) and m.kappa == pytest.approx(20*m.mu0)
    assert m.to_dict()["prony"] == [{"g": 0.15, "tau_s": 0.01}, {"g": 0.1, "tau_s": 0.1}]
    assert "prony" not in rm.load_recipe().nominal().to_dict()        # elastic receipts unchanged
    assert r.values(mu_eq=6e5).prony == m.prony


def test_step_rule_uses_mu0_not_mu_inf():
    """RUBBER-VISC-08a: the dt of a Prony recipe equals the elastic rule at mu0 = mu_eq / g_inf."""
    r = rm.load_recipe(VISCO_RECIPE)
    rule = r.step_rule(1e-3, 2e-4)
    mu0_max = r.shear_modulus_eq.hi/r.g_inf
    at_mu0 = rm.step_rule(1e-3, 2e-4, mu0_max=mu0_max, kappa_max=20*mu0_max, density_min=r.density.lo)
    at_mueq = rm.step_rule(1e-3, 2e-4, mu0_max=r.shear_modulus_eq.hi, kappa_max=20*r.shear_modulus_eq.hi,
                           density_min=r.density.lo)
    assert rule["dt_s"] == at_mu0["dt_s"] and rule["dt_s"] < at_mueq["dt_s"]
    assert rule["mu0_max_pa"] == pytest.approx(mu0_max)


@pytest.mark.parametrize("prony, match", [
    (((0.1, 0.01),)*5, "at most 4"),
    (((0.6, 0.01), (0.4, 0.1)), "g_inf > 0"),
    (((-0.1, 0.01),), ">= 0"),
    (((0.1, 0.0),), "finite positive"),
    (((0.1, float("nan")),), "finite positive"),
    (((float("inf"), 0.1),), ">= 0"),
])
def test_prony_refusals(prony, match):
    with pytest.raises(rm.RubberRecipeError, match=match):
        rm.MaterialValues(mu0=MU, kappa=KAPPA, density=1150.0, prony=prony)


def test_padding():
    padded = rm.pad_prony(SLS, 2)
    assert padded == ((0.3, 0.01), (0.0, 1.0))
    m = rm.MaterialValues(mu0=MU, kappa=KAPPA, density=1150.0, prony=padded)
    row = m.params64(DT)
    assert row[5] == 0.0 and row[8] == 0.0 and m.g_inf == pytest.approx(0.7)
    with pytest.raises(rm.RubberRecipeError):
        rm.pad_prony(P2, 1)


# ----------------------------------------------------------------------------- R3.3 references

@pytest.mark.parametrize("terms", [SLS, P2])
def test_relaxation_and_creep_limits(terms):
    g_inf = 1-sum(g for g, _ in terms)
    E0, Einf = 9*KAPPA*MU/(3*KAPPA+MU), 9*KAPPA*MU*g_inf/(3*KAPPA+MU*g_inf)
    assert ref.relaxation_uniaxial(0.0, MU, KAPPA, terms) == pytest.approx(E0, rel=1e-12)
    assert ref.relaxation_uniaxial(100.0, MU, KAPPA, terms) == pytest.approx(Einf, rel=1e-12)
    assert ref.creep_uniaxial(0.0, MU, KAPPA, terms) == pytest.approx(1/E0, rel=1e-12)
    assert ref.creep_uniaxial(100.0, MU, KAPPA, terms) == pytest.approx(1/Einf, rel=1e-12)
    assert np.all(np.diff(ref.relaxation_uniaxial(np.linspace(0, 0.5, 200), MU, KAPPA, terms)) <= 1e-9*E0)
    assert np.all(np.diff(ref.creep_uniaxial(np.linspace(0, 0.5, 200), MU, KAPPA, terms)) >= -1e-9/E0)


@pytest.mark.parametrize("terms", [SLS, P2])
def test_interconversion_E_star_J_equals_t(terms):
    for t in (0.003, 0.02, 0.2):
        val, _ = integrate.quad(lambda s: ref.relaxation_uniaxial(t-s, MU, KAPPA, terms)
                                * ref.creep_uniaxial(s, MU, KAPPA, terms), 0, t, epsabs=0, epsrel=1e-12, limit=200)
        assert val == pytest.approx(t, rel=1e-9)


def test_declared_design_values():
    """Values quoted in ADR 0023 §21.3 / the protocol (P2 set; three decimals: E'' at 8 Hz is 0.11550 MPa, quoted
    as 0.116)."""
    declared = {1: (0.788, 0.058), 4: (0.883, 0.115), 8: (0.937, 0.116), 16: (0.988, 0.127), 32: (1.063, 0.134)}
    for f, (e1, e2) in declared.items():
        E = ref.uniaxial_complex_modulus(2*math.pi*f, MU, KAPPA, P2)/1e6
        assert abs(E.real-e1) <= 6e-4 and abs(E.imag-e2) <= 6e-4, f
    ret = sorted(ref.creep_terms(MU, KAPPA, P2)["retardation_times_s"])
    assert round(ret[0]*1e3, 1) == 6.2 and round(ret[1]*1e3, 1) == 61.9
    root = ref.beam_root(0.04, 0.004, 0.004, 1150.0, MU, KAPPA, P2)
    assert round(root["zeta_1_s"], 3) == 4.743 and round(root["f_d_hz"], 3) == 11.706
    assert root["log_decrement"] == pytest.approx(0.405, abs=5e-4)


def test_beam_root_elastic_limit():
    elastic = ref.beam_root(0.04, 0.004, 0.004, 1150.0, MU, KAPPA, ((0.0, 0.01),))
    assert abs(elastic["zeta_1_s"]) < 1e-9
    assert elastic["f_d_hz"] == pytest.approx(elastic["elastic_instantaneous_f_hz"], rel=1e-12)


def test_hereditary_integral_step_limit_and_linear_ramp():
    lam = 1.5
    A = np.diag([lam, lam**-0.5, lam**-0.5])
    block = ref.HereditaryBlock([0.0, 1e-7, 0.05], [np.eye(3), A, A], MU, KAPPA, P2)
    for t in (0.001, 0.01, 0.04):
        assert np.allclose(block.evaluate(t), block.step_closed_form(t, A, 1e-7), rtol=1e-4, atol=0)
    # small strain: S0 is linear in gamma to O(gamma), so Q = g mu0 gamma_dot tau (1 - e^{-t/tau}) on the shear entry
    gam, T = 1e-6, 0.02
    G = np.eye(3)
    G[0, 1] = gam
    block = ref.HereditaryBlock([0.0, T, 0.05], [np.eye(3), G, G], MU, KAPPA, SLS)
    (g, tau), = SLS
    for t in (0.005, 0.02):
        q = block.q_at(t)[0][0, 1]
        assert q == pytest.approx(g*MU*(gam/T)*tau*(1-math.exp(-t/tau)), rel=1e-5)


def test_hereditary_integral_equals_a_fine_exponential_recurrence():
    """The exponential update with many substeps converges (O(h^2)) to the quadrature (independent methods)."""
    gam = 1.0
    G = np.eye(3)
    G[0, 1] = gam
    T = 2e-4
    block = ref.HereditaryBlock([0.0, T, 0.02], [np.eye(3), G, G], MU, KAPPA, P2)
    n = 4000
    h = T/n
    coeffs = rvc.prony_coefficients(P2, h)
    q = np.zeros((2, 3, 3))
    sp = np.zeros((3, 3))
    for k in range(1, n+1):
        s0 = rvc.s0_np(np.eye(3)+(k/n)*(G-np.eye(3)), MU)
        q = np.array([a*qi+b*(s0-sp) for (a, b), qi in zip(coeffs, q)])
        sp = s0
    assert np.allclose(q, block.q_at(T), rtol=1e-6, atol=1e-6*np.abs(q).max())
