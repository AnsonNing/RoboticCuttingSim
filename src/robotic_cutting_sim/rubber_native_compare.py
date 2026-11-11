"""RL-09 comparison of the Full Native rubber run with its headless GPU replays (task P6.4 / P6.5; protocol
``rubber_rl_tests_v1`` RL-09 as declared in ``docs/reports/rl_ready/P2_REMAINING_PLAN.md`` §4.8, §4.6).

Clauses, all scored, verdict PASS only if every clause holds:

(a) receipts identical to the headless reference except device / timing: every leaf of the Native receipt equals the
    corresponding leaf of EVERY replay receipt; the only exempt leaves are those named in ``RECEIPT_EXEMPT_KEYS``
    (device and wall times; reported with their values);
(b) material inputs = recorded measured clamp poses, bitwise float64: the Native strip-frame material pose equals the
    strip-frame pose recomputed from the recorded measured world pose, and every replay's material pose, motion
    (rotation, translation) and solver-staged group rows (q1, b1) equal the Native ones bit for bit;
(c) physics: every one of the 46 statistics (markers 27, clamp wrench 6, plane wrench 6, psi_eq, KE, E_grav, contact
    work, Jbar min / max, CFL_eff max) of the Native run inside ``rubber_l3_band_v4`` built from the 20 band replays,
    alpha_family 0.05 (``rubber_l3.evaluate``); the 10 held-out replays are classified by the same bounds and
    reported, not scored;
(d) invalid bits 0 at every Native step;
(e) bridge integrity: at every surface update the USD mesh vertices equal the surface particles under the strip pose
    within 1e-6 m (and the vertex count and topology match); camera consistency is reported, never scored.

Pure numpy / scipy (no Warp, no Kit): runs on CPU in CI.
"""
import math

import numpy as np

from . import rubber_l3

RULE_ID = "rubber_rl_tests_v1/RL-09"
RECEIPT_EXEMPT_KEYS = ("device", "build_wall_s", "capture_wall_s")
INPUT_KEYS = ("material_p", "material_q", "motion_rotation", "motion_translation", "staged_q1", "staged_b1")
BRIDGE_TOLERANCE_M = 1e-6
ALPHA_FAMILY = 0.05
BAND_REPLAYS = 20
HELDOUT_REPLAYS = 10


def flatten(obj, prefix=""):
    """{path: leaf} of nested dicts / lists (lists indexed)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(obj, (list, tuple)):
        if not obj:
            out[prefix] = []
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _leaf_key(path):
    return path.rsplit("/", 1)[-1].split("[", 1)[0]


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))
    return type(a) is type(b) and a == b


def compare_receipts(native, replay, exempt=RECEIPT_EXEMPT_KEYS):
    """Leaf-by-leaf equality except the exempt keys -> {identical, differences, exempt}."""
    fa, fb = flatten(native), flatten(replay)
    diffs, exempted = [], []
    for path in sorted(set(fa) | set(fb)):
        if _leaf_key(path) in exempt:
            exempted.append({"path": path, "native": fa.get(path), "replay": fb.get(path)})
            continue
        if path not in fa or path not in fb:
            diffs.append({"path": path, "native": fa.get(path, "<missing>"), "replay": fb.get(path, "<missing>")})
        elif not _same(fa[path], fb[path]):
            diffs.append({"path": path, "native": fa[path], "replay": fb[path]})
    return {"identical": not diffs, "differences": diffs, "exempt": exempted}


def bitwise_equal(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def check_inputs(native, replays, placement_translation):
    """Clause (b). ``native`` / ``replays``: EpisodeArrays dicts (``EpisodeArrays.load``)."""
    t = np.asarray(placement_translation, np.float64)
    measured = np.asarray(native["measured_world"], np.float64)
    recomputed_p = measured[:, :3]-t
    recomputed_q = measured[:, 3:7].copy()
    from_measured = bitwise_equal(native["material_p"], recomputed_p) and bitwise_equal(native["material_q"],
                                                                                         recomputed_q)
    rows = []
    for name, r in replays.items():
        eq = {k: bitwise_equal(native[k], r[k]) for k in INPUT_KEYS}
        rows.append({"replay": name, "bitwise": eq, "all": all(eq.values())})
    ok = bool(from_measured and rows and all(r["all"] for r in rows))
    return {"verdict": "PASS" if ok else "FAIL", "native_material_pose_equals_recorded_measured_pose": from_measured,
            "replays": len(rows), "replays_bitwise_equal": sum(r["all"] for r in rows),
            "rows": [r for r in rows if not r["all"]][:10], "keys": list(INPUT_KEYS)}


def check_physics(native, band, heldout, stat_names, alpha_family=ALPHA_FAMILY):
    """Clause (c): ``native`` [K, S], ``band`` {name: [K, S]} (20), ``heldout`` {name: [K, S]} (reported)."""
    names = list(stat_names)
    band_names = list(band)
    ref = {s: np.stack([np.asarray(band[b], np.float64)[:, j] for b in band_names]) for j, s in enumerate(names)}
    cands = {"native": {s: np.asarray(native, np.float64)[:, j] for j, s in enumerate(names)}}
    for h, arr in heldout.items():
        cands[f"heldout:{h}"] = {s: np.asarray(arr, np.float64)[:, j] for j, s in enumerate(names)}
    ev = rubber_l3.evaluate(ref, cands, alpha_family=alpha_family, names=names)
    nat = ev["runs"]["native"]
    native_rows = [r for r in ev["rows"] if r["run"] == "native"]
    held = {k: v for k, v in ev["runs"].items() if k.startswith("heldout:")}
    worst = sorted(native_rows, key=lambda r: -r["M_over_upper"])[:8]
    return {"verdict": nat["verdict"], "rule": ev["rule"], "alpha_family": ev["alpha_family"],
            "alpha_stat": ev["alpha_stat"], "band_replays": len(band_names), "statistics": len(names),
            "native_outside": nat["outside"], "native_max_M_over_upper": nat["max_M_over_upper"],
            "native_worst": worst, "native_rows": native_rows,
            "heldout": {"runs": len(held), "pass": sum(v["verdict"] == "PASS" for v in held.values()),
                        "per_run": held, "scored": False},
            "bounds": {s: {k: v for k, v in b.items() if k != "loo_statistics"} for s, b in ev["bounds"].items()}}


def check_invalid(native_invalid):
    inv = np.asarray(native_invalid, np.int64)
    ok = bool(len(inv) and (inv == 0).all())
    return {"verdict": "PASS" if ok else "FAIL", "steps": int(len(inv)), "nonzero_steps": int((inv != 0).sum()),
            "max_bits": int(inv.max()) if len(inv) else None}


def check_bridge(rows, expected_updates, tol=BRIDGE_TOLERANCE_M):
    errs = [r.get("max_abs_error_m", math.inf) for r in rows]
    counts = all(r.get("vertices") == r.get("expected_vertices") for r in rows)
    topo = all(r.get("topology_equal", False) for r in rows)
    ok = bool(len(rows) == expected_updates and counts and topo and max(errs, default=math.inf) <= tol)
    return {"verdict": "PASS" if ok else "FAIL", "updates": len(rows), "expected_updates": int(expected_updates),
            "max_abs_error_m": max(errs, default=None), "tolerance_m": tol, "vertex_counts_equal": counts,
            "topology_equal": topo}


def evaluate_rl09(*, native_receipt, replay_receipts, native_arrays, band_arrays, heldout_arrays, stat_names,
                  placement_translation, bridge_rows, expected_steps, camera_consistency=None):
    """Full RL-09 verdict from loaded artifacts (see the module docstring)."""
    completed = int(native_arrays.get("completed", len(native_arrays["statistics"])))
    rec = {name: compare_receipts(native_receipt, r) for name, r in replay_receipts.items()}
    a_ok = bool(rec) and all(v["identical"] for v in rec.values())
    first_diff = next((v["differences"][:10] for v in rec.values() if v["differences"]), [])
    a = {"verdict": "PASS" if a_ok else "FAIL", "replays": len(rec),
         "identical": sum(v["identical"] for v in rec.values()), "first_differences": first_diff,
         "exempt_keys": list(RECEIPT_EXEMPT_KEYS),
         "exempt_example": next(iter(rec.values()))["exempt"] if rec else []}
    all_replays = {**band_arrays, **heldout_arrays}
    b = check_inputs(native_arrays, all_replays, placement_translation)
    c = check_physics(native_arrays["statistics"], {k: v["statistics"] for k, v in band_arrays.items()},
                      {k: v["statistics"] for k, v in heldout_arrays.items()}, stat_names)
    d = check_invalid(native_arrays["invalid_reason"][:completed])
    e = check_bridge(bridge_rows, expected_steps+1)
    complete = {"verdict": "PASS" if completed == expected_steps and len(band_arrays) == BAND_REPLAYS else "FAIL",
                "native_completed_steps": completed, "expected_steps": int(expected_steps),
                "band_replays": len(band_arrays), "heldout_replays": len(heldout_arrays),
                "declared_band_replays": BAND_REPLAYS, "declared_heldout_replays": HELDOUT_REPLAYS}
    clauses = {"completeness": complete, "a_receipts": a, "b_inputs": b, "c_physics": c, "d_invalid_bits": d,
               "e_bridge_integrity": e}
    verdict = "PASS" if all(v["verdict"] == "PASS" for v in clauses.values()) else "FAIL"
    return {"rule": RULE_ID, "verdict": verdict, "clauses": clauses,
            "camera_consistency": {"scored": False, "summary": camera_consistency}}


__all__ = ["evaluate_rl09", "compare_receipts", "check_inputs", "check_physics", "check_invalid", "check_bridge",
           "flatten", "bitwise_equal", "RECEIPT_EXEMPT_KEYS", "INPUT_KEYS", "RULE_ID", "BAND_REPLAYS",
           "HELDOUT_REPLAYS"]
