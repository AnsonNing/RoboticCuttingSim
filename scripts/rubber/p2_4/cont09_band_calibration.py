"""Calibration of the RUBBER-CONT-09 (b) perturbation-band rule (task X1 item, ADR 0023 §22.9; analysis, no simulation).

Null hypothesis H0 of (b): env e's observable does not depend on env 1's input, so on a nondeterministic GPU the
episodes A_1..A_k (reference input) and B_1..B_k (perturbed env 1) are EXCHANGEABLE samples of the same run-to-run
noise. For one scalar (one component of one observable of one env at one control step):

    d = max_{i,j} |A_i - B_j|                (cross spread)
    s = max(max_i A_i - min_i A_i, max_j B_j - min_j B_j)   (within spread, both sides)
    v2 rule: d <= lam * s + eps_rel * max_i |A_i|

The v1 rule was k = 1 on the A side (one repeat A, A'), lam = 1, s = |A - A'|, and one B: under H0 with a continuous
symmetric noise P(|A - B| > |A - A'|) = 1/2 by exchangeability (swap A' and B), so every scalar whose noise exceeds the
eps_rel floor is a coin flip and a family of ~10^2..10^3 scalars rejects almost surely.

This script prints the per-scalar false-reject probability P(d > lam s | H0) (eps_rel = 0, i.e. the worst case) by
Monte Carlo for Gaussian, Laplace, Student-t(3) and uniform noise, k in {1, 3, 4, 5}, and the family-wise bound
min(1, n_scalars * p) (union bound; the scalars are positively correlated, so it is conservative).

    .venv/bin/python scripts/rubber/p2_4/cont09_band_calibration.py [--n 2000000] [--seed 20261111]
"""
import argparse
import json

import numpy as np


def false_reject(k, lam, dist, n, rng):
    shape = (n, 2*k)
    if dist == "gauss":
        x = rng.standard_normal(shape)
    elif dist == "laplace":
        x = rng.laplace(size=shape)
    elif dist == "t3":
        x = rng.standard_t(3, size=shape)
    elif dist == "uniform":
        x = rng.uniform(-1, 1, size=shape)
    else:
        raise ValueError(dist)
    a, b = x[:, :k], x[:, k:]
    d = np.maximum(a.max(1)-b.min(1), b.max(1)-a.min(1))
    if k == 1:
        return None
    s = np.maximum(np.ptp(a, axis=1), np.ptp(b, axis=1))
    return float(np.mean(d > lam*s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {"n_samples": a.n, "seed": a.seed, "v1_rule_per_scalar_false_reject": 0.5, "rows": []}
    for k in (3, 4, 5):
        for lam in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
            row = {"k": k, "lam": lam}
            for dist in ("gauss", "laplace", "t3", "uniform"):
                row[dist] = false_reject(k, lam, dist, a.n, rng)
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
    # the v1 rule (k = 1, one B), by simulation for the record
    x = rng.standard_normal((a.n, 3))
    out["v1_rule_simulated_gauss"] = float(np.mean(np.abs(x[:, 0]-x[:, 2]) > np.abs(x[:, 0]-x[:, 1])))
    print(json.dumps({"v1_rule_simulated_gauss": out["v1_rule_simulated_gauss"]}))


if __name__ == "__main__":
    main()
