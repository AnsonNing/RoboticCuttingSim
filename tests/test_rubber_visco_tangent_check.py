"""RUBBER-VISC-08b regression (small sample of ``scripts/rubber/p2_3/visco_tangent_check.py``; the formal check uses
>= 2000 states): the visco guard bounds the exact algorithmic acoustic tensor of history-generated states; with every
g = 0 the tangent equals the elastic one."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"/"rubber"/"p2_3"))
import visco_tangent_check as V  # noqa: E402


def test_small_tangent_check_bounds_every_sample():
    rows, groups, premise, Ns = V.run(per_group=8, n_dirs=128, seed=11)
    assert len(rows) == 48 and len(Ns) == 128
    assert max(r["ratio"] for r in rows) <= 1.0
    assert all(abs(r["ratio"]-r["ratio_f64"]) < 1e-5 for r in rows)


def test_tangent_of_zero_g_equals_the_elastic_tangent():
    rng = np.random.default_rng(3)
    F = V.random_F(rng, 1)[0]
    terms = ((0.0, 0.005), (0.0, 0.05))
    coeffs = V.rvc.prony_coefficients(terms, 5e-6)
    q = rng.normal(0, 1e4, (2, 3, 3))
    D = V.tangent(F, q, np.zeros((3, 3)), terms, coeffs)
    De = V.tangent(F, None, None, None, None, fn=lambda X: V.rvc.nh_pk1_np(X, np.linalg.det(X), V.MU0, V.KAPPA))
    assert np.allclose(D, De, rtol=0, atol=1e-6*np.abs(De).max())
