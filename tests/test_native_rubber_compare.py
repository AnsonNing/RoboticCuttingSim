"""P6.4: CPU tests of the RL-09 comparison logic (``rubber_native_compare``; no Warp, no Kit), plus an
artifact-backed recomputation of the P6.5 evidence that is NOT_RUN (skipped) when the artifacts are absent."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from robotic_cutting_sim import rubber_native_compare as C

ROOT = Path(__file__).resolve().parents[1]
K, S = 30, 46
NAMES = tuple(f"s{j}" for j in range(S))


def _receipt(**over):
    r = {"material": {"solver": {"device": "cuda:0", "dt_s": 3.8e-6, "x": [1, 2]}, "build_wall_s": 1.2,
                      "graph": {"capture_wall_s": 0.3, "chunk": 500}}, "family": {"id": "t", "mu": 0.35}}
    for path, value in over.items():
        node = r
        keys = path.split("/")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return r


def test_receipts_exempt_only_device_and_timing():
    a = _receipt()
    b = _receipt(**{"material/solver/device": "cuda:1", "material/build_wall_s": 9.9,
                    "material/graph/capture_wall_s": 7.0})
    res = C.compare_receipts(a, b)
    assert res["identical"] and len(res["exempt"]) == 3
    for over in ({"family/mu": 0.3500000000000001}, {"material/solver/x": [1, 3]}, {"material/solver/dt_s": -0.0},
                 {"family/id": "u"}):
        assert not C.compare_receipts(a, _receipt(**over))["identical"], over
    missing = _receipt()
    del missing["family"]["mu"]
    assert not C.compare_receipts(a, missing)["identical"]
    assert C.compare_receipts({"x": 0.0}, {"x": -0.0})["identical"] is False
    assert C.compare_receipts({"x": math.nan}, {"x": math.nan})["identical"] is True
    assert C.compare_receipts({"x": 1}, {"x": 1.0})["identical"] is False      # type is part of the receipt


def _inputs(rng, t):
    measured = np.c_[rng.normal(size=(K, 3)), np.tile([0, 0, 0, 1.0], (K, 1))]
    return {"measured_world": measured, "material_p": measured[:, :3]-t, "material_q": measured[:, 3:7].copy(),
            "motion_rotation": rng.normal(size=(K, 3, 3)), "motion_translation": rng.normal(size=(K, 3)),
            "staged_q1": rng.normal(size=(K, 4)), "staged_b1": rng.normal(size=(K, 3)),
            "invalid_reason": np.zeros(K, np.int64), "completed": np.int64(K)}


def test_inputs_bitwise():
    t = np.array([-0.0615, -0.13, -0.01])
    nat = _inputs(np.random.default_rng(0), t)
    rep = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in nat.items()}
    assert C.check_inputs(nat, {"r0": rep, "r1": rep}, t)["verdict"] == "PASS"
    for key in C.INPUT_KEYS:
        bad = dict(rep)
        bad[key] = rep[key].copy()
        bad[key].reshape(-1)[5] = np.nextafter(bad[key].reshape(-1)[5], np.inf)
        res = C.check_inputs(nat, {"r0": rep, "bad": bad}, t)
        assert res["verdict"] == "FAIL" and res["replays_bitwise_equal"] == 1, key
    # the Native material pose must be the recorded measured pose in the strip frame
    off = dict(nat, material_p=nat["material_p"]+1e-12)
    assert C.check_inputs(off, {"r0": dict(rep, material_p=off["material_p"])}, t)["verdict"] == "FAIL"
    assert C.check_inputs(nat, {}, t)["verdict"] == "FAIL"


def _series(rng, n, scale=1e-6):
    base = np.linspace(0, 1, K)[:, None]*np.arange(1, S+1)[None, :]
    return {f"replay_{i:02d}": base+scale*rng.normal(size=(K, S)) for i in range(n)}, base


def test_physics_band_accepts_an_exchangeable_run_and_rejects_a_shift():
    rng = np.random.default_rng(1)
    band, base = _series(rng, 20)
    held, _ = _series(rng, 10)
    native = base+1e-6*rng.normal(size=(K, S))
    res = C.check_physics(native, band, held, NAMES)
    assert res["verdict"] == "PASS" and res["statistics"] == S and res["band_replays"] == 20
    assert res["alpha_stat"] == pytest.approx(0.05/S)
    assert res["heldout"]["runs"] == 10 and res["heldout"]["scored"] is False
    shifted = native.copy()
    shifted[:, 7] += 1e-3
    res = C.check_physics(shifted, band, held, NAMES)
    assert res["verdict"] == "FAIL" and res["native_outside"] == 1 and res["native_worst"][0]["statistic"] == "s7"
    nan = native.copy()
    nan[3, 2] = np.nan
    assert C.check_physics(nan, band, held, NAMES)["verdict"] == "FAIL"


def test_physics_deterministic_reference_needs_bitwise_native():
    base = np.tile(np.linspace(0, 1, K)[:, None], (1, S))
    band = {f"r{i}": base.copy() for i in range(20)}
    assert C.check_physics(base.copy(), band, {}, NAMES)["verdict"] == "PASS"
    assert C.check_physics(np.nextafter(base, 2.0), band, {}, NAMES)["verdict"] == "FAIL"


def test_invalid_and_bridge_clauses():
    assert C.check_invalid(np.zeros(100, np.int64))["verdict"] == "PASS"
    assert C.check_invalid(np.r_[np.zeros(99, np.int64), 8])["verdict"] == "FAIL"
    assert C.check_invalid(np.zeros(0, np.int64))["verdict"] == "FAIL"
    good = [{"max_abs_error_m": 5e-8, "vertices": 10, "expected_vertices": 10, "topology_equal": True}]*101
    assert C.check_bridge(good, 101)["verdict"] == "PASS"
    assert C.check_bridge(good[:100], 101)["verdict"] == "FAIL"
    assert C.check_bridge(good[:-1]+[dict(good[0], max_abs_error_m=1.1e-6)], 101)["verdict"] == "FAIL"
    assert C.check_bridge(good[:-1]+[dict(good[0], topology_equal=False)], 101)["verdict"] == "FAIL"
    assert C.check_bridge(good[:-1]+[dict(good[0], vertices=9)], 101)["verdict"] == "FAIL"


def test_evaluate_rl09_end_to_end():
    rng = np.random.default_rng(2)
    t = np.array([-0.0615, -0.13, -0.01])
    nat = _inputs(rng, t)
    band_s, base = _series(rng, 20)
    held_s, _ = _series(rng, 10)
    nat["statistics"] = base+1e-6*rng.normal(size=(K, S))
    copy = {k: v for k, v in nat.items() if k != "statistics"}
    band = {k: dict(copy, statistics=v) for k, v in band_s.items()}
    held = {f"replay_{20+i:02d}": dict(copy, statistics=v) for i, v in enumerate(held_s.values())}
    receipts = {f"replay_{i:02d}": _receipt(**{"material/build_wall_s": float(i)}) for i in range(30)}
    rows = [{"max_abs_error_m": 5e-8, "vertices": 10, "expected_vertices": 10, "topology_equal": True}]*(K+1)
    kw = dict(native_receipt=_receipt(), replay_receipts=receipts, native_arrays=nat, band_arrays=band,
              heldout_arrays=held, stat_names=NAMES, placement_translation=t, bridge_rows=rows, expected_steps=K)
    res = C.evaluate_rl09(**kw)
    assert res["verdict"] == "PASS", {k: v["verdict"] for k, v in res["clauses"].items()}
    assert res["camera_consistency"]["scored"] is False
    bad_receipts = dict(receipts, replay_05=_receipt(**{"family/mu": 0.4}))
    assert C.evaluate_rl09(**dict(kw, replay_receipts=bad_receipts))["clauses"]["a_receipts"]["verdict"] == "FAIL"
    inv = dict(nat, invalid_reason=np.r_[np.zeros(K-1, np.int64), 1])
    assert C.evaluate_rl09(**dict(kw, native_arrays=inv))["verdict"] == "FAIL"
    short = dict(nat, completed=np.int64(K-1))
    assert C.evaluate_rl09(**dict(kw, native_arrays=short))["clauses"]["completeness"]["verdict"] == "FAIL"
    few = {k: band[k] for k in list(band)[:19]}
    assert C.evaluate_rl09(**dict(kw, band_arrays=few))["verdict"] == "FAIL"


EVIDENCE = ROOT/"docs"/"evidence"/"rl_ready"/"p2_6_native_rubber_001.json"


def test_evidence_recomputes_from_artifacts():
    """NOT_RUN unless the P6.5 evidence and its artifacts exist on this machine."""
    if not EVIDENCE.is_file():
        pytest.skip("NOT_RUN: P6.5 evidence absent")
    ev = json.loads(EVIDENCE.read_text())
    native_dir = ROOT/ev["native_row"]["run"]
    replay_dir = ROOT/ev["replay_meta"]["out"] if "out" in ev["replay_meta"] else None
    if not (native_dir/"native_arrays.npz").is_file() or replay_dir is None or not replay_dir.is_dir():
        pytest.skip("NOT_RUN: P6.5 artifacts absent (artifacts/ is machine-local)")
    import sys
    sys.path.insert(0, str(ROOT/"scripts"/"rubber"/"p2_6"))
    import compare_native_rubber as CMP
    from robotic_cutting_sim.native_rubber_host import EPISODE, STAT_NAMES
    native = CMP.load_native(native_dir)
    band, held, receipts, _, _ = CMP.load_replays(replay_dir)
    res = C.evaluate_rl09(native_receipt=native["receipt"], replay_receipts=receipts, native_arrays=native["arrays"],
                          band_arrays=band, heldout_arrays=held, stat_names=STAT_NAMES,
                          placement_translation=native["scene"]["placement"]["world_from_strip_translation_m"],
                          bridge_rows=native["bridge_rows"], expected_steps=EPISODE["steps"])
    assert res["verdict"] == ev["verdict"]
    assert {k: v["verdict"] for k, v in res["clauses"].items()} == \
        {k: v["verdict"] for k, v in ev["evaluation"]["clauses"].items()}
