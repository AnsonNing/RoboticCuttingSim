"""RUBBER-PHYS-10 L3 rule v4 (``rubber_l3_band_v4``; protocol ``rubber_phys_tests_v4``, F3 of ADR 0023 §18.8).

The v2 band ([min, max] of 3 single-env GPU repeats widened by 1e-5) is statistically weak: a fourth exchangeable run
leaves the range of three with probability 1/2. v4 scores one scalar statistic per (env e, observable o), so the time
correlation of a series does not enter the false-reject rate:

    M(run) = max_k |y_run(k) - ybar_ref(k)|        (k = control steps, ybar_ref = mean of the n reference repeats)

The reference distribution of M comes from the n single-env GPU repeats themselves by leave-one-out
(M_i = max_k |y_i(k) - ybar_{-i}(k)|). Upper bound, one-sided prediction bound for one new run at level alpha_stat:

    lognormal:  U = exp(mean(ln M_i) + t_{1 - alpha_stat, n - 1} sqrt(1 + 1/n) sd(ln M_i))   (sd ddof 1)
    fallback (Shapiro-Wilk p < 0.01 on ln M_i, or some M_i = 0): distribution-free one-sided Cantelli bound
                U = mean(M_i) + sqrt(1 / alpha_stat - 1) sqrt(1 + 1/n) sd(M_i)
    all M_i = 0 (bitwise-deterministic reference): U = 0

alpha_stat = alpha_family / (number of scored statistics) (Bonferroni), so the nominal probability that an exchangeable
batched run is rejected anywhere is <= alpha_family whatever the correlation between the statistics. A batched run
PASSes L3 when M <= U for every statistic. The LOO statistics use means of n - 1 repeats (slightly wider than the
batched run's n-repeat mean): conservative towards acceptance by at most sqrt((1 + 1/(n-1)) / (1 + 1/n)) - 1 (< 1 % at
n >= 10). Pure numpy / scipy; no Warp.
"""
import math

import numpy as np

RULE_ID = "rubber_l3_band_v4"


def loo_statistics(ref):
    """ref [n, K]: n repeats of one observable over K control steps -> leave-one-out M_i [n]."""
    ref = np.asarray(ref, np.float64)
    n = ref.shape[0]
    if n < 3:
        raise ValueError("at least 3 reference repeats")
    total = ref.sum(0)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.max(np.abs(ref[i]-(total-ref[i])/(n-1)))
    return out


def upper_bound(m, alpha_stat):
    """One-sided prediction bound of the next M from the reference statistics m (see the module docstring)."""
    from scipy import stats
    m = np.asarray(m, np.float64)
    n = len(m)
    inflate = math.sqrt(1+1/n)
    if np.all(m == 0.0):
        return {"upper": 0.0, "method": "deterministic_reference", "shapiro_p": None}
    if np.all(m > 0.0):
        y = np.log(m)
        shapiro_p = float(stats.shapiro(y).pvalue) if n >= 3 else None
        if shapiro_p is not None and shapiro_p >= 0.01:
            k = float(stats.t.ppf(1-alpha_stat, n-1))*inflate
            return {"upper": float(math.exp(y.mean()+k*y.std(ddof=1))), "method": "lognormal_prediction",
                    "shapiro_p": shapiro_p, "k": k}
    else:
        shapiro_p = None
    k = math.sqrt(1/alpha_stat-1)*inflate
    return {"upper": float(m.mean()+k*m.std(ddof=1)), "method": "cantelli_fallback", "shapiro_p": shapiro_p, "k": k}


def evaluate(ref, candidates, *, alpha_family=0.05, names=None):
    """ref: {stat_name: [n, K]}; candidates: {run_name: {stat_name: [K]}} -> classification per statistic and run."""
    names = sorted(ref) if names is None else list(names)
    alpha_stat = alpha_family/len(names)
    bounds, rows = {}, []
    for name in names:
        r = np.asarray(ref[name], np.float64)
        m = loo_statistics(r)
        b = upper_bound(m, alpha_stat)
        mean = r.mean(0)
        bounds[name] = dict(b, loo_statistics=m.tolist(), n=int(r.shape[0]),
                            ref_spread_rel_max=float(np.max((r.max(0)-r.min(0))/np.maximum(np.abs(mean), 1e-300))))
        for run, series in candidates.items():
            M = float(np.max(np.abs(np.asarray(series[name], np.float64)-mean)))
            rows.append({"run": run, "statistic": name, "M": M, "upper": b["upper"], "method": b["method"],
                         "class": "WITHIN" if M <= b["upper"] else "OUTSIDE",
                         "M_over_upper": (M/b["upper"]) if b["upper"] > 0 else (0.0 if M == 0 else math.inf)})
    runs = {}
    for run in candidates:
        mine = [r for r in rows if r["run"] == run]
        runs[run] = {"outside": sum(r["class"] == "OUTSIDE" for r in mine), "statistics": len(mine),
                     "verdict": "PASS" if all(r["class"] == "WITHIN" for r in mine) else "FAIL",
                     "max_M_over_upper": max(r["M_over_upper"] for r in mine)}
    return {"rule": RULE_ID, "alpha_family": alpha_family, "alpha_stat": alpha_stat, "statistics": names,
            "bounds": bounds, "rows": rows, "runs": runs}
