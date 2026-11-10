"""P1-4 memory-stability statistics: robust slopes and the pre-declared bounds B1-B5 (NumPy only).

Bounds and their numbers come from ``configs/rl_benchmark/long_rollout_v1.json`` (``bounds``), declared before
any run; this module never changes them. Inputs are the child ledger records (``sample``, ``episode``,
``reset``, ``heartbeat``, ``census``) and the supervisor telemetry (per-PID VRAM with timestamps).

Slopes: the gate statistic is the Theil-Sen slope (median of pairwise slopes; robust to single spikes such as
an ``nvidia-smi`` sample taken during a transient allocation) with Sen's 95 % interval; the ordinary
least-squares slope with its t-based 95 % interval is always reported next to it.
"""
import math

import numpy as np

MIB = 1024. * 1024.


def _t_quantile(conf, dof):
    try:
        from scipy.stats import t
        return float(t.ppf(.5 + conf / 2., dof))
    except ImportError:   # normal approximation (documented in the result)
        return 1.959963984540054


def _z_quantile(conf):
    try:
        from scipy.stats import norm
        return float(norm.ppf(.5 + conf / 2.))
    except ImportError:
        return 1.959963984540054


def ols_slope(x, y, conf=.95):
    """OLS slope, intercept and the t-based ``conf`` interval of the slope (None when n < 3 or x constant)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return dict(method="ols", n=n, slope=None, intercept=None, ci_low=None, ci_high=None)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    se = math.sqrt(max((resid ** 2).sum() / (n - 2), 0.) / sxx)
    half = _t_quantile(conf, n - 2) * se
    return dict(method="ols", n=n, slope=float(slope), intercept=float(intercept), stderr=float(se),
                ci_low=float(slope - half), ci_high=float(slope + half), conf=conf)


def theil_sen(x, y, conf=.95, max_points=2000):
    """Theil-Sen slope with Sen's (1968) ``conf`` interval; inputs above ``max_points`` are thinned evenly
    (deterministic, every k-th point, reported as ``n_used``)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    out = dict(method="theil_sen", n=n, slope=None, intercept=None, ci_low=None, ci_high=None, conf=conf)
    if n < 3 or np.ptp(x) == 0:
        return out
    if n > max_points:
        index = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
        x, y = x[index], y[index]
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = dx != 0
    slopes = np.sort((y[j] - y[i])[ok] / dx[ok])
    m = len(slopes)
    slope = float(np.median(slopes))
    nn = len(x)
    var_s = nn * (nn - 1) * (2 * nn + 5) / 18.
    c = _z_quantile(conf) * math.sqrt(var_s)
    lo = int(max(0, math.floor((m - c) / 2.)))
    hi = int(min(m - 1, math.ceil((m + c) / 2.)))
    out.update(n_used=nn, slope=slope, intercept=float(np.median(y - slope * x)), ci_low=float(slopes[lo]),
               ci_high=float(slopes[hi]))
    return out


def slopes(x, y, *, per=1000., scale=1., conf=.95):
    """Both slopes of ``y * scale`` per ``per`` units of ``x``."""
    ts, ols = theil_sen(x, np.asarray(y, float) * scale, conf), ols_slope(x, np.asarray(y, float) * scale, conf)
    for d in (ts, ols):
        for key in ("slope", "ci_low", "ci_high"):
            if d.get(key) is not None:
                d[key] = d[key] * per
        d["per"] = per
    return dict(theil_sen=ts, ols=ols)


def quintile_medians(values):
    values = np.asarray(values, float)
    if len(values) < 5:
        return None
    parts = np.array_split(values, 5)
    return [float(np.median(p)) for p in parts]


def warmup_end_step(episode_records, reset_records, num_envs, quota_classes):
    """Fixed rule: warm-up ends when every env has completed its first (quota) episode and its first reset after
    it. Returns the batch step, or None if some env never got there."""
    first_end = {}
    for r in episode_records:
        if r.get("episode_index", -1) >= 0 and r.get("end_class") in quota_classes and r["env"] not in first_end:
            first_end[r["env"]] = r["global_step_end"]
    if len(first_end) < num_envs:
        return None
    end = 0
    for e, step in first_end.items():
        later = [r["global_step"] for r in reset_records if e in r.get("ids", []) and r["global_step"] >= step]
        if not later:
            return None
        end = max(end, min(later))
    return int(end)


def map_time_to_step(times, anchor_t, anchor_step):
    """Batch step at wall times ``times`` by linear interpolation of the child's (t, global_step) anchors;
    times outside the anchors map to None."""
    at, st = np.asarray(anchor_t, float), np.asarray(anchor_step, float)
    order = np.argsort(at)
    at, st = at[order], st[order]
    times = np.asarray(times, float)
    if len(at) < 2:
        return np.full(len(times), np.nan)
    out = np.interp(times, at, st)
    out[(times < at[0]) | (times > at[-1])] = np.nan
    return out


def _range_check(values, limit):
    values = [v for v in values if v is not None]
    if not values:
        return dict(status="NOT_APPLICABLE", reason="no values (device counters absent)")
    spread = float(max(values) - min(values))
    return dict(status="PASS" if spread <= limit else "FAIL", max_minus_min=spread, min=float(min(values)),
                max=float(max(values)), n=len(values), limit=limit)


GATED = ("B0", "B1", "B2", "B3", "B4")


def evaluate_bounds(records, telemetry, bounds, *, num_envs, quota_classes, gate="theil_sen", required=GATED):
    """B0 (VRAM budget), B1-B5 for ONE process (attempt). ``records``: child ledger; ``telemetry``: supervisor
    samples of that attempt. Returns per-bound dicts plus ``verdict``: FAIL if any gated bound fails,
    NOT_EVALUABLE if a ``required`` bound is not PASS (missing samples or counters), else PASS."""
    samples = [r for r in records if r.get("type") == "sample"]
    episodes = [r for r in records if r.get("type") == "episode"]
    resets = [r for r in records if r.get("type") == "reset"]
    anchors = [(r["t"], r["global_step"]) for r in records if r.get("type") in ("heartbeat", "sample")
               and r.get("global_step") is not None]
    warm = warmup_end_step(episodes, resets, num_envs, quota_classes)
    out = dict(warmup_end_batch_step=warm, gate_statistic=gate)
    if warm is None:
        out.update(verdict="NOT_EVALUABLE", reason="warm-up not reached (an env never completed episode + reset)")
        return out
    post = [s for s in samples if s["global_step"] >= warm]
    b = bounds

    # B1: device allocators flat
    lim1 = b["B1"]["max_minus_min_mib"] * MIB
    out["B1"] = dict(mempool_current=_range_check([s.get("mempool_current_bytes") for s in post], lim1),
                     torch_allocated=_range_check([s.get("torch_allocated_bytes") for s in post], lim1))
    parts = [v["status"] for v in out["B1"].values()]
    out["B1"]["status"] = "FAIL" if "FAIL" in parts else ("PASS" if "PASS" in parts else "NOT_APPLICABLE")

    # B2: own-PID VRAM (supervisor nvidia-smi samples mapped to batch steps)
    own = [(s["t"], s["gpu"].get("own_memory_mib")) for s in telemetry
           if isinstance(s.get("gpu"), dict) and s["gpu"].get("ok") and s["gpu"].get("own_memory_mib") is not None]
    b2 = dict(samples_total=len(own))
    if own and anchors:
        t = np.array([o[0] for o in own])
        mem = np.array([o[1] for o in own], float)
        step = map_time_to_step(t, [a[0] for a in anchors], [a[1] for a in anchors])
        keep = np.isfinite(step) & (step >= warm) & (mem > 0)
        b2["samples_post_warmup"] = int(keep.sum())
        b2["peak_own_mib_all"] = float(mem.max())
        if keep.sum() >= b["B2"].get("min_samples", 10):
            st, mm = step[keep], mem[keep]
            b2["slope_mib_per_1000_batch_steps"] = slopes(st, mm, per=1000.)
            diffs = np.diff(mm[np.argsort(st)])
            b2["max_single_increase_mib"] = float(diffs.max()) if len(diffs) else 0.
            sl = b2["slope_mib_per_1000_batch_steps"][gate]["slope"]
            ok = sl is not None and sl <= b["B2"]["slope_mib_per_1000_batch_steps"] and \
                b2["max_single_increase_mib"] <= b["B2"]["max_single_increase_mib"]
            b2["status"] = "PASS" if ok else "FAIL"
            b2["ols_ci_high_exceeds_bound"] = bool(
                (b2["slope_mib_per_1000_batch_steps"]["ols"]["ci_high"] or 0.) > b["B2"]["slope_mib_per_1000_batch_steps"])
        else:
            b2["status"] = "NOT_EVALUABLE"
    else:
        b2["status"] = "NOT_APPLICABLE" if not own else "NOT_EVALUABLE"
    out["B2"] = b2
    budget = b.get("B0", {}).get("own_vram_peak_mib")
    if budget is not None:
        peak = b2.get("peak_own_mib_all")
        out["B0"] = dict(status="NOT_APPLICABLE" if peak is None else ("PASS" if peak <= budget else "FAIL"),
                         peak_own_mib=peak, limit_mib=budget)

    # B3: RSS at post-gc episode-end samples, per 1000 env-steps
    gc_samples = [s for s in post if s.get("kind") == "episode_end_post_gc" and s.get("rss_mib") is not None]
    b3 = dict(samples=len(gc_samples))
    if len(gc_samples) >= b["B3"].get("min_samples", 5):
        x = np.array([s["global_step"] for s in gc_samples], float) * num_envs
        y = np.array([s["rss_mib"] for s in gc_samples], float)
        b3["slope_mib_per_1000_env_steps"] = slopes(x, y, per=1000.)
        q = quintile_medians(y)
        b3["quintile_medians_mib"] = q
        drift = q[4] - q[1]
        allowed = max(b["B3"]["quintile_drift_mib"], b["B3"]["quintile_drift_fraction"] * q[1])
        b3.update(last_minus_second_quintile_mib=drift, allowed_drift_mib=allowed)
        sl = b3["slope_mib_per_1000_env_steps"][gate]["slope"]
        b3["status"] = "PASS" if (sl is not None and sl <= b["B3"]["slope_mib_per_1000_env_steps"]
                                  and drift <= allowed) else "FAIL"
        b3["ols_ci_high_exceeds_bound"] = bool(
            (b3["slope_mib_per_1000_env_steps"]["ols"]["ci_high"] or 0.) > b["B3"]["slope_mib_per_1000_env_steps"])
    else:
        b3["status"] = "NOT_EVALUABLE"
    out["B3"] = b3

    # B4: graph and grid state
    captures = [s.get("graph_captures") for s in samples if s.get("graph_captures") is not None]
    census = [s.get("live_device_hashgrids") for s in samples if s.get("live_device_hashgrids") is not None]
    census += [r.get("live_device_hashgrids") for r in records if r.get("type") == "census"
               and r.get("live_device_hashgrids") is not None]
    inits = [s.get("hashgrid_device_inits") for s in samples if s.get("hashgrid_device_inits") is not None]
    modules = [s.get("warp_modules") for s in post if s.get("warp_modules") is not None]
    b4 = dict(graph_captures_max=max(captures) if captures else None,
              expected_graph_captures=b["B4"]["graph_captures_per_process"],
              live_device_hashgrids_values=sorted(set(census)), hashgrid_device_inits_max=max(inits) if inits else None,
              warp_modules_post_warmup=sorted(set(modules)))
    if not captures and not census:
        b4["status"] = "NOT_APPLICABLE"
    else:
        checks = dict(
            graph_recaptures_zero=bool(captures) and max(captures) == b["B4"]["graph_captures_per_process"],
            hashgrid_census_one=bool(census) and set(census) == {b["B4"]["live_device_hashgrids"]},
            hashgrid_device_inits_one=bool(inits) and max(inits) == b["B4"]["live_device_hashgrids"],
            warp_modules_constant=len(set(modules)) <= 1)
        b4["checks"] = checks
        b4["invalid_multi_grid_process"] = bool(census) and max(census) > 1
        b4["status"] = "PASS" if all(checks.values()) else "FAIL"
    out["B4"] = b4

    # B5: reset latency (reported, not a gate)
    kinds = {}
    for r in resets:
        if r.get("wall_s") is not None:
            kinds.setdefault(r.get("kind", "unknown"), []).append(r["wall_s"])
    out["B5"] = dict(status="REPORTED", gate=False, per_kind={
        k: dict(n=len(v), p50_s=float(np.percentile(v, 50)), p95_s=float(np.percentile(v, 95)),
                max_s=float(max(v))) for k, v in sorted(kinds.items())})

    statuses = {k: out[k]["status"] if k in out else "MISSING" for k in GATED}
    if "FAIL" in statuses.values():
        out["verdict"] = "FAIL"
    elif any(statuses[k] != "PASS" for k in required) or "NOT_EVALUABLE" in statuses.values():
        out["verdict"] = "NOT_EVALUABLE"
    else:
        out["verdict"] = "PASS"
    out["gated_statuses"] = statuses
    out["required"] = list(required)
    return out


__all__ = ["GATED", "ols_slope", "theil_sen", "slopes", "quintile_medians", "warmup_end_step", "map_time_to_step",
           "evaluate_bounds"]
