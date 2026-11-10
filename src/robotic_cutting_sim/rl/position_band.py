"""Position band v2 (review fix F2): derive and apply the L3 position tolerance against Native.

The method is pre-declared in ``configs/rl_benchmark/position_band_v2.json`` (schema
``rcs_rl_position_band_method_v2``). The derived per-seed values live in
``configs/rl_benchmark/position_band_v2_reference.json`` (schema ``rcs_rl_position_band_reference_v2``), written by
``scripts/rl_ready/make_position_band_v2.py`` from the reference runs. Consumers (for example an acceptance-harness
judge) call ``load_reference()``, ``classify(seed, metric_m)`` and ``aggregate(classes)``; nothing here imports Warp
or Torch.

Statistics (both scored, classified separately): ``max_position_diff_m_over_time`` (primary, max over the 121
Native samples) and ``final_max_position_diff_m`` (amendment 1: the statistic of the P1-6 acceptance judge, at
control step 1200) of a full 1200-step WP-17 replay. The method file also carries, after derivation,
``position_band_m = {"<seed>": {"high_m": U}}`` for ``final_max_position_diff_m`` in the judge's format
(``rl.perf_acceptance.load_position_band``). Tolerance: one-sided log-normal
prediction bound ``exp(mean(ln m) + k * sd(ln m))`` with ``k = t(1 - alpha, n - 1) * sqrt(1 + 1/n)`` (nominal
false-reject rate ``alpha`` per run), or the reference maximum when the Shapiro-Wilk check on ``ln m`` rejects
normality at p < 0.01 (nominal false-reject rate ``1/(n + 1)``).
"""
import json
import math
from pathlib import Path

METHOD_SCHEMA = "rcs_rl_position_band_method_v2"
REFERENCE_SCHEMA = "rcs_rl_position_band_reference_v2"
BAND_ID = "position_band_v2"
METRIC = "max_position_diff_m_over_time"
FINAL_METRIC = "final_max_position_diff_m"   # the P1-6 judge's statistic (amendment 1)
METRICS = (METRIC, FINAL_METRIC)
JUDGE_METRIC = FINAL_METRIC
# written after derivation (judge view, status); not part of the method digest
DERIVED_KEYS = ("position_band_m", "position_band_m_statistic", "position_band_m_by_statistic", "derived", "version",
                "status")
ROOT = Path(__file__).resolve().parents[3]
METHOD_PATH = ROOT / "configs" / "rl_benchmark" / "position_band_v2.json"
REFERENCE_PATH = ROOT / "configs" / "rl_benchmark" / "position_band_v2_reference.json"
MIN_RUNS = 10
NORMALITY_P_MIN = 0.01

WITHIN, BELOW, OUTSIDE = "WITHIN", "BELOW_REFERENCE", "OUTSIDE"
INVALID, NO_DATA, NOT_APPLICABLE = "INVALID_PROCESS", "NO_POSITION_DATA", "NOT_APPLICABLE"
SCORED = (WITHIN, BELOW, OUTSIDE)
CLASSES = (WITHIN, BELOW, OUTSIDE, INVALID, NO_DATA, NOT_APPLICABLE)


def load_method(path=METHOD_PATH):
    method = json.loads(Path(path).read_text())
    if method.get("schema") != METHOD_SCHEMA:
        raise ValueError(f"{path}: schema {method.get('schema')!r} is not {METHOD_SCHEMA}")
    return method


def method_digest(method=None):
    """SHA256 of the declared method (the method file without the derived keys written after derivation)."""
    import hashlib
    method = load_method() if method is None else method
    core = {k: v for k, v in method.items() if k not in DERIVED_KEYS}
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def judge_table(reference):
    """``position_band_m`` in the judge's format: ``{"<seed>": {"high_m": U}}`` of ``JUDGE_METRIC``."""
    return {seed: {"high_m": band["upper_m"], "low_descriptive_m": band["lower_m"]}
            for seed, band in reference["statistics"][JUDGE_METRIC]["seeds"].items() if band["status"] == "DERIVED"}


def derive(values, alpha=0.01, min_runs=MIN_RUNS):
    """Band of one seed from its reference metric values (m). Returns the reference-file entry for that seed."""
    import numpy as np
    from scipy import stats
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("reference values must be finite and positive")
    n = int(values.size)
    entry = dict(n=n, values_m=values.tolist(), alpha=float(alpha), min_m=float(values.min()),
                 max_m=float(values.max()))
    if n < min_runs:
        entry.update(status="INCONCLUSIVE_UNDERSAMPLED", upper_m=None, lower_m=None, kind=None,
                     nominal_false_reject_rate=None)
        return entry
    y = np.log(values)
    mean, sd = float(y.mean()), float(y.std(ddof=1))
    k = float(stats.t.ppf(1. - alpha, n - 1) * math.sqrt(1. + 1. / n))
    shapiro_p = float(stats.shapiro(y).pvalue)
    entry.update(log_mean=mean, log_sd=sd, k=k, shapiro_wilk_p=shapiro_p,
                 geometric_mean_m=float(math.exp(mean)), lower_m=float(math.exp(mean - k * sd)))
    if shapiro_p < NORMALITY_P_MIN:
        entry.update(status="DERIVED", kind="fallback_empirical_max", upper_m=float(values.max()),
                     nominal_false_reject_rate=1. / (n + 1))
    else:
        entry.update(status="DERIVED", kind="one_sided_lognormal_prediction_bound",
                     upper_m=float(math.exp(mean + k * sd)), nominal_false_reject_rate=float(alpha))
    return entry


def load_reference(path=REFERENCE_PATH):
    reference = json.loads(Path(path).read_text())
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(f"{path}: schema {reference.get('schema')!r} is not {REFERENCE_SCHEMA}")
    return reference


def classify(seed, metric_m, reference=None, *, invalid=False, applicable=True, metric=METRIC):
    """Class of one run: ``WITHIN`` / ``BELOW_REFERENCE`` (a sub-label of within) / ``OUTSIDE`` for a scored run,
    else ``INVALID_PROCESS`` / ``NO_POSITION_DATA`` / ``NOT_APPLICABLE``. Unknown or under-sampled seeds raise."""
    if not applicable:
        return NOT_APPLICABLE
    if invalid:
        return INVALID
    if metric_m is None:
        return NO_DATA
    reference = load_reference() if reference is None else reference
    seeds = reference["statistics"][metric]["seeds"] if "statistics" in reference else reference["seeds"]
    band = seeds.get(str(int(seed)))
    if band is None or band.get("status") != "DERIVED":
        raise ValueError(f"no derived position band v2 for seed {seed}")
    metric_m = float(metric_m)
    if not math.isfinite(metric_m) or metric_m > band["upper_m"]:
        return OUTSIDE
    return BELOW if metric_m < band["lower_m"] else WITHIN


def critical_count(n, alpha=0.01, level=0.05):
    """Smallest c with P[Binomial(n, alpha) > c] <= level (the number of OUTSIDE runs a MET set may contain)."""
    if n <= 0:
        return 0
    cdf, c = 0., 0
    while True:
        cdf += math.comb(n, c) * alpha ** c * (1. - alpha) ** (n - c)
        if 1. - cdf <= level + 1e-15:
            return c
        c += 1


def aggregate(classes, alpha=0.01, level=0.05):
    """Verdict of a run set: ``MET`` / ``NOT_MET`` over the scored runs, ``NO_SCORED_RUNS`` if there are none."""
    classes = list(classes)
    unknown = set(classes) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown classes {sorted(unknown)}")
    scored = [c for c in classes if c in SCORED]
    outside = sum(c == OUTSIDE for c in scored)
    n = len(scored)
    c = critical_count(n, alpha, level)
    verdict = "NO_SCORED_RUNS" if n == 0 else "MET" if outside <= c else "NOT_MET"
    return dict(verdict=verdict, scored=n, outside=outside, critical_count=c,
                not_scored={k: classes.count(k) for k in (INVALID, NO_DATA, NOT_APPLICABLE) if k in classes})
