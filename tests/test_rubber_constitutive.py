"""R1.1: neo_hookean_jbar_v1 Warp functions vs float64 references (ADR 0023 §5)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import rubber_constitutive as rcon
from robotic_cutting_sim import rubber_reference as R

MU, KAPPA = 4e5, 8e6


def _random_F(n, seed=0, scale=0.35, jmin=0.3):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        F = np.eye(3)+scale*rng.standard_normal((3, 3))
        if np.linalg.det(F) > jmin:
            out.append(F)
    return np.asarray(out)


def test_pk1_matches_float64_reference():
    F = _random_F(200).astype(np.float32)
    Jb = np.linalg.det(F.astype(np.float64))*(1+0.01*np.random.default_rng(1).standard_normal(len(F)))
    P, valid = rcon.evaluate_pk1(F, Jb, MU, KAPPA)
    worst = max(np.linalg.norm(P[i]-R.pk1(F[i].astype(np.float64), MU, KAPPA, Jb[i]))/np.linalg.norm(
        R.pk1(F[i].astype(np.float64), MU, KAPPA, Jb[i])) for i in range(len(F)))
    assert worst < 3e-7     # one float32 rounding of a float64 evaluation
    assert valid.all()


def test_pk1_is_derivative_of_the_discrete_energy_fd():
    """Central differences of Psi(F) = W_iso(F) + U(J(F)) (one particle per cell) in float64 vs the kernel."""
    F = _random_F(30, seed=3)
    worst = 0.0
    for Fi in F:
        G = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                E = np.zeros((3, 3))
                E[i, j] = 1e-6
                G[i, j] = (R.energy_density(Fi+E, MU, KAPPA)-R.energy_density(Fi-E, MU, KAPPA))/2e-6
        P, _ = rcon.evaluate_pk1(Fi, np.linalg.det(Fi.astype(np.float32).astype(np.float64)), MU, KAPPA)
        worst = max(worst, np.abs(P[0]-G).max()/np.abs(G).max())
    assert worst < 1e-5     # float32 storage of F and P


def test_jbar_derivative_term_matches_cell_energy():
    """With Jbar = mean det F over the cell, dPsi_cell/dF_p = V(W_iso' + U'(Jbar) cof F_p): the kernel's p(Jbar)."""
    rng = np.random.default_rng(5)
    Fs = _random_F(8, seed=7, scale=0.1)
    def psi(Fs):
        J = np.linalg.det(Fs)
        return sum(R.isochoric_energy(F, MU) for F in Fs)+len(Fs)*R.volumetric_energy(J.mean(), KAPPA)
    p = 3
    G = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            plus, minus = Fs.copy(), Fs.copy()
            plus[p, i, j] += 1e-6
            minus[p, i, j] -= 1e-6
            G[i, j] = (psi(plus)-psi(minus))/2e-6
    jbar = np.linalg.det(Fs.astype(np.float32).astype(np.float64)).mean()
    P, _ = rcon.evaluate_pk1(Fs[p], jbar, MU, KAPPA)
    assert np.abs(P[0]-G).max()/np.abs(G).max() < 1e-5
    del rng


def test_objectivity():
    F = _random_F(50, seed=11).astype(np.float32)
    Jb = np.linalg.det(F.astype(np.float64))
    for Rm in (Rotation.from_euler("z", 90, degrees=True).as_matrix(),
               Rotation.from_rotvec(np.deg2rad(37)*np.array([1, 2, 3])/np.sqrt(14)).as_matrix()):
        RF = np.einsum("ij,njk->nik", Rm, F.astype(np.float64)).astype(np.float32)
        P, _ = rcon.evaluate_pk1(F, Jb, MU, KAPPA)
        PR, _ = rcon.evaluate_pk1(RF, np.linalg.det(RF.astype(np.float64)), MU, KAPPA)
        rel = np.linalg.norm(PR-np.einsum("ij,njk->nik", Rm, P), axis=(1, 2))/np.linalg.norm(P, axis=(1, 2))
        assert rel.max() < 1e-5


def test_reference_state_is_stress_free():
    P, valid = rcon.evaluate_pk1(np.eye(3), 1.0, MU, KAPPA)
    assert np.abs(P).max() == 0.0 and valid.all()


def test_j_domain_flags_and_finite_evaluation():
    cases = [np.diag([1, 1, 0.1]),      # J 0.1 < 0.2
             np.diag([1, 1, -0.5]),     # inverted
             np.diag([2, 2, 2]),        # J 8 > 5
             np.zeros((3, 3)),          # J 0
             np.diag([1, 1, 1.0])]
    F = np.asarray(cases, np.float32)
    Jb = np.array([1.0, 1.0, 1.0, 1.0, 6.0])   # last: particle fine, cell Jbar outside the domain
    P, valid = rcon.evaluate_pk1(F, Jb, MU, KAPPA)
    assert valid.tolist() == [False, False, False, False, False]
    assert np.isfinite(P).all()
    P, valid = rcon.evaluate_pk1(np.full((3, 3), np.nan, np.float32), 1.0, MU, KAPPA)
    assert not valid[0]
    P, valid = rcon.evaluate_pk1(np.eye(3), np.nan, MU, KAPPA)
    assert not valid[0]


def test_cofactor_finite_for_nonpositive_J():
    F = np.diag([1, 1, -0.5]).astype(np.float32)
    P, _ = rcon.evaluate_pk1(F, 1.0, MU, KAPPA)       # Jbar = 1: pressure 0, only the isochoric part
    Je = rcon.J_EVAL_FLOOR
    Fd = F.astype(np.float64)
    a = MU*Je**(-2/3)
    expected = a*Fd-a*np.sum(Fd*Fd)/(3*Je)*R.cofactor(Fd)
    assert np.allclose(P[0], expected, rtol=1e-6)
