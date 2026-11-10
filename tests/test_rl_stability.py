"""P1-4: robust slopes, warm-up rule and memory bounds B0-B5 (``rl.stability``; NumPy only)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robotic_cutting_sim.rl import stability as T  # noqa: E402
from robotic_cutting_sim.rl.supervision import QUOTA_CLASSES  # noqa: E402

BOUNDS = json.loads((ROOT / "configs" / "rl_benchmark" / "long_rollout_v1.json").read_text())["bounds"]
BOUNDS = dict(BOUNDS, B0=dict(BOUNDS["B0"], own_vram_peak_mib=1536))


def test_theil_sen_exact_line_and_robust_to_a_spike():
    x = np.arange(100.)
    y = 3. + .5 * x
    ts = T.theil_sen(x, y)
    assert ts["slope"] == pytest.approx(.5) and ts["intercept"] == pytest.approx(3.)
    y2 = y.copy()
    y2[95] += 1000.                                   # one transient sample (high leverage)
    assert T.theil_sen(x, y2)["slope"] == pytest.approx(.5)
    assert abs(T.ols_slope(x, y2)["slope"] - .5) > .01


def test_slope_intervals_cover_the_true_slope_on_noisy_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10000, 300)
    y = 1000. + 2e-3 * x + rng.normal(0, 5., len(x))
    s = T.slopes(x, y, per=1000.)
    for method in ("theil_sen", "ols"):
        d = s[method]
        assert d["ci_low"] < 2. < d["ci_high"] and d["slope"] == pytest.approx(2., abs=.5) and d["per"] == 1000.
    assert T.ols_slope([1, 2], [1, 2])["slope"] is None and T.theil_sen([1, 1, 1], [1, 2, 3])["slope"] is None


def test_theil_sen_thins_large_inputs_deterministically():
    x = np.arange(5000.)
    out = T.theil_sen(x, 2. * x, max_points=500)
    assert out["n"] == 5000 and out["n_used"] == 500 and out["slope"] == pytest.approx(2.)


def test_quintile_medians_and_time_to_step_map():
    assert T.quintile_medians([1, 2, 3, 4]) is None
    assert T.quintile_medians(list(range(10))) == [.5, 2.5, 4.5, 6.5, 8.5]
    steps = T.map_time_to_step([0.5, 1.5, 5.], [1., 2., 3.], [10, 20, 30])
    assert np.isnan(steps[0]) and steps[1] == pytest.approx(15.) and np.isnan(steps[2])


def _episode(env, index, end, cls="TASK_SUCCESS"):
    return dict(type="episode", env=env, episode_index=index, end_class=cls, global_step_end=end)


def test_warmup_rule_needs_first_quota_episode_and_reset_of_every_env():
    eps = [_episode(0, -1, 3, "TRUNCATED_BY_FORCED_RESET"), _episode(0, 0, 10), _episode(1, 0, 14)]
    resets = [dict(type="reset", global_step=10, ids=[0]), dict(type="reset", global_step=14, ids=[1])]
    assert T.warmup_end_step(eps, resets, 2, QUOTA_CLASSES) == 14
    assert T.warmup_end_step(eps, resets[:1], 2, QUOTA_CLASSES) is None
    assert T.warmup_end_step(eps[:2], resets, 2, QUOTA_CLASSES) is None


def _synthetic(*, steps=4000, n=8, vram_leak=0., rss_leak=0., mempool_jump=0, census=1, captures=1, modules=7):
    """Child ledger + supervisor telemetry of a synthetic run (1 s per batch step)."""
    records, telemetry = [], []
    for s in range(0, steps + 1, 50):
        records.append(dict(type="heartbeat", t=float(s), global_step=s))
        records.append(dict(type="sample", kind="periodic", t=float(s), global_step=s,
                            mempool_current_bytes=437_089_923 + (mempool_jump if s > steps // 2 else 0),
                            torch_allocated_bytes=48_477_696, graph_captures=captures, hashgrid_device_inits=1,
                            warp_modules=modules, rss_mib=1900. + rss_leak * s * n / 1000.))
    for k, end in enumerate(range(300, steps, 150)):
        e = k % n
        records.append(_episode(e, k // n, end))
        records.append(dict(type="reset", global_step=end, ids=[e], wall_s=.005, kind="episode_end"))
        records.append(dict(type="sample", kind="episode_end_post_gc", t=float(end), global_step=end,
                            rss_mib=1900. + rss_leak * end * n / 1000., live_device_hashgrids=census,
                            graph_captures=captures, hashgrid_device_inits=1, warp_modules=modules,
                            mempool_current_bytes=437_089_923, torch_allocated_bytes=48_477_696))
    for t in np.arange(0., steps, 5.):
        telemetry.append(dict(type="sample", t=float(t), gpu=dict(ok=True, own_memory_mib=1378. + vram_leak * t / 1000.)))
    return records, telemetry


def test_bounds_pass_on_flat_memory():
    records, telemetry = _synthetic()
    out = T.evaluate_bounds(records, telemetry, BOUNDS, num_envs=8, quota_classes=QUOTA_CLASSES)
    assert out["verdict"] == "PASS", out["gated_statuses"]
    assert out["warmup_end_batch_step"] == 300 + 150 * 7
    assert out["B1"]["mempool_current"]["max_minus_min"] == 0
    assert out["B2"]["slope_mib_per_1000_batch_steps"]["theil_sen"]["slope"] == pytest.approx(0.)
    assert out["B5"]["per_kind"]["episode_end"]["p95_s"] == pytest.approx(.005) and out["B5"]["gate"] is False


@pytest.mark.parametrize("kwargs,bound", [
    (dict(vram_leak=3.), "B2"),               # 3 MiB per 1000 batch steps > 2
    (dict(rss_leak=1.), "B3"),                # 1 MiB per 1000 env-steps > 0.5
    (dict(mempool_jump=2 * 1024 * 1024), "B1"),
    (dict(captures=2), "B4"),
    (dict(census=2), "B4"),
])
def test_bounds_fail_on_leaks_and_state_changes(kwargs, bound):
    records, telemetry = _synthetic(**kwargs)
    out = T.evaluate_bounds(records, telemetry, BOUNDS, num_envs=8, quota_classes=QUOTA_CLASSES)
    assert out[bound]["status"] == "FAIL" and out["verdict"] == "FAIL"
    if kwargs.get("census"):
        assert out["B4"]["invalid_multi_grid_process"]


def test_bounds_single_vram_jump_fails_even_with_flat_slope():
    records, telemetry = _synthetic()
    telemetry[len(telemetry) // 2]["gpu"]["own_memory_mib"] += 100.
    out = T.evaluate_bounds(records, telemetry, BOUNDS, num_envs=8, quota_classes=QUOTA_CLASSES)
    assert out["B2"]["max_single_increase_mib"] == pytest.approx(100.) and out["B2"]["status"] == "FAIL"


def test_bounds_vram_budget_b0():
    records, telemetry = _synthetic()
    telemetry[3]["gpu"]["own_memory_mib"] = 1600.                  # before warm-up still counts for the budget
    out = T.evaluate_bounds(records, telemetry, BOUNDS, num_envs=8, quota_classes=QUOTA_CLASSES)
    assert out["B0"]["status"] == "FAIL" and out["verdict"] == "FAIL"


def test_missing_required_counters_are_not_evaluable_not_pass():
    records, telemetry = _synthetic()
    for r in records:
        if r.get("type") == "sample":
            r["mempool_current_bytes"] = r["torch_allocated_bytes"] = None
    out = T.evaluate_bounds(records, telemetry, BOUNDS, num_envs=8, quota_classes=QUOTA_CLASSES)
    assert out["B1"]["status"] == "NOT_APPLICABLE" and out["verdict"] == "NOT_EVALUABLE"
    out = T.evaluate_bounds(records, [], BOUNDS, num_envs=8, quota_classes=QUOTA_CLASSES, required=("B3", "B4"))
    assert out["verdict"] == "PASS"
    out = T.evaluate_bounds(records[:3], telemetry, BOUNDS, num_envs=8, quota_classes=QUOTA_CLASSES)
    assert out["verdict"] == "NOT_EVALUABLE" and out["warmup_end_batch_step"] is None
