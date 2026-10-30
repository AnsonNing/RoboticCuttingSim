"""WP-12 Phase 3 driver helpers (CPU, no Warp): threshold rows and the B-vs-A comparison decision rule."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wp12_phase3_surface", ROOT/"scripts/run_wp12_phase3_surface.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def test_check_and_verdict():
    assert D.check(.04, "<=", .05)["verdict"] == "PASS"
    assert D.check(.06, "<=", .05)["verdict"] == "FAIL"
    assert D.check(None, "<=", .05)["verdict"] == "NOT_EVALUABLE"
    assert D.check(float("nan"), ">=", .99)["verdict"] == "NOT_EVALUABLE"
    assert D.verdict_of({"a": D.check(1, "==", 1), "b": D.not_applicable("x")}) == "PASS"
    assert D.verdict_of({"a": D.check(1, "==", 1), "b": D.check(None, "==", 1)}) == "NOT_EVALUABLE"
    assert D.verdict_of({"a": D.check(2, "==", 1), "b": D.check(None, "==", 1)}) == "FAIL"


def test_comparison_bands_and_repeat_rule():
    a = [{"current_p95_s": .50, "coverage": .90, "normal_p95_deg": 10., "closed": True}]*2
    b = [{"current_p95_s": .40, "coverage": 1., "normal_p95_deg": 10.5, "closed": True},
         {"current_p95_s": .43, "coverage": 1., "normal_p95_deg": 11.5, "closed": True}]
    out = D.compare("X", a, b)
    assert out["current_p95_s"]["decision"] == "B_BETTER"          # 0.07-0.10 s better in every repeat (> 0.06)
    assert out["normal_p95_deg"]["decision"] == "B_WORSE"          # 1.5 deg worse in one repeat (> 1 deg)
    assert out["coverage"]["decision"] == "B_BETTER"               # beyond the (zero) repeat spread
    assert out["closed"]["decision"] == "NOT_DIFFERENT"
    assert out["rim_corner_p95_s"]["decision"] == "NOT_COMPARABLE"
    close = D.compare("X", [{"current_p95_s": .50}]*2, [{"current_p95_s": .45}, {"current_p95_s": .55}])
    assert close["current_p95_s"]["decision"] == "NOT_DIFFERENT"   # within the 0.06 s band


def test_protocol_thresholds_are_read_from_the_frozen_file():
    T = D.protocol_thresholds()
    assert T["S1_flat_cut"]["reference_p95_s"] == .1 and T["G2_static_unit_test"]["hermite"]["cut_face_max_s"] == .25
    assert T["S2_deformed_cut"]["current_max_s"] == 2.0 and T["all"]["volume_current_rel"] == .02
    sc = {"reference_p95_s": .05, "reference_max_s": .3, "current_p95_s": .4, "current_max_s": 1.8,
          "abs_bias_toward_side2_s": .01, "abs_thickness_bias_s": .02, "normal_p95_deg": 5., "coverage": .995,
          "rim_corner_p95_s": .3, "closed": True, "shells_equal_components": True}
    assert D.verdict_of(D.s1_checks(sc, T)) == "FAIL"                       # max 1.8 > 1.5
    assert D.verdict_of(D.s1_checks(sc, T, current_max=2.0)) == "PASS"     # S2 override
    assert set(D.s1_checks(sc, T, rows=D.s1_variant_rows("S3"))) == {"reference_p95_s", "reference_max_s"}


def test_topology_hash_depends_on_faces_and_reference_vertices():
    f = np.array([[0, 1, 2]])
    x = np.eye(3)
    assert D.topo_hash(f, x) == D.topo_hash(f.copy(), x.copy())
    assert D.topo_hash(f, x) != D.topo_hash(f[:, ::-1], x)
    assert D.topo_hash(f, x) != D.topo_hash(f, x+1e-12)


def test_unevaluated_rows_read_not_run_never_pass():
    """Decision review F11: the S8 rows are not evaluated for B_mid (recontact frames cover A and B only), so its S8
    verdict must read NOT_RUN, not PASS; FAIL and NOT_EVALUABLE still take precedence."""
    assert D.verdict_of({"a": D.check(1, "==", 1), "b": D.not_run("x")}) == "NOT_RUN"
    assert D.verdict_of({"a": D.check(2, "==", 1), "b": D.not_run("x")}) == "FAIL"
    assert D.verdict_of({"a": D.check(None, "==", 1), "b": D.not_run("x")}) == "NOT_EVALUABLE"
    assert D.verdict_of({"a": D.check(1, "==", 1), "b": D.not_applicable("x")}) == "PASS"
    T = D.protocol_thresholds()
    rows = D.s8_rows(D.B_MID, {"recontact_frames": []}, T)
    assert set(rows) == {"vertices_shared_across_conflicting_codes", "shell_set_unchanged_since_separation",
                         "all_temporal_slow_motion_residual_p95_s"}
    assert {r["verdict"] for r in rows.values()} == {"NOT_RUN"}
    rep = {"scalars": {}, "temporal_fixed_state": {"topology_stable": True, "residual_max_s": 0.},
           "volume_checks": {"reference_abs_error_m3": 0., "reference_bound_m3": 1., "reference_source": "test",
                             "current_rel_error": .001}}
    result = {"frames": [{"reports": {D.B_MID: rep}}], "recontact_frames": []}
    out = D.case_checks("S8", result, T)[D.B_MID]
    assert all(out["checks"][k]["verdict"] == "PASS" for k in out["checks"] if k.startswith("all_volume"))
    assert out["verdict"] == "NOT_RUN"
