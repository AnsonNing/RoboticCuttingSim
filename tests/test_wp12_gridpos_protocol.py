"""WP-12 Phase 4 tlmpm_v6 candidate: the gridpos protocol, runner and summarizer (CPU, no simulation).

- the protocol declares the decision's stages: S1 = the kill set, S2/S3 rows, S4 Native not scheduled; ids unique;
  budget cap 18 GPU-h; every row kind has a runner driver; E1 rows copy the frozen E1 protocol rows; RES rows name an
  existing resolution_002 V5 stand-in;
- queue order: per row V6 then V5 (no V5 for stand-in rows), repeats last; S4 never queued;
- the summarizer's row gate implements the kill rule (frozen verdict, integrity, components vs V5, TRIM thresholds,
  not worse than V5) and never gates report / characterization rows on their verdict;
- the runner refuses undeclared rows, DEV/formal output paths of the wrong form and formal rows without a committed
  protocol; frozen runners and the Phase F v1 hash pin are untouched.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"configs/wp12/phase4_gridpos_protocol_v1.json"


def load(name):
    spec = importlib.util.spec_from_file_location(f"test_gridpos_{name}", ROOT/f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def protocol():
    return json.loads(PROTOCOL.read_text())


def test_protocol_stages_match_the_decision(protocol):
    rows = protocol["rows"]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    s1 = [r["id"] for r in rows if r["stage"] == "S1"]
    assert s1 == ["M1a", "M3b", "M4", "M5a", "M5b", "M8b", "F2.7_withdrawal_v2", "F2.9_blunt_handle_v2", "FB_CUT", "FB_TRIM_r10"]
    assert all(r["gate"] == "kill" for r in rows if r["stage"] == "S1")
    assert {r["id"] for r in rows if r.get("repeat", 1) == 2} == {"M1a", "M3b", "FB_CUT"}
    s2 = {r["id"] for r in rows if r["stage"] == "S2"}
    for need in ("M1b", "M2", "M3a", "M3c", "M5c", "M6", "M7", "M8a", "M9a", "M9b", "M9c", "M9d", "F2.1_gate_14", "F2.11_multicut_history",
                 "F3_cut_dt_half", "POSTCUT_press", "POSTCUT_push", "POSTCUT_stop_withdraw"):
        assert need in s2
    assert sum(r["id"].startswith("PhaseR_Rb_") for r in rows) == 7
    s3 = {r["id"] for r in rows if r["stage"] == "S3"}
    for need in ("FB_partial", "FB_TestB", "FB_TestC", "4c_G01", "4c_G07", "E1_A_p05", "TEST_D2_s2501", "TRIM_W_s2307",
                 "EFF_CUT", "EFF_TRIM", "EFF_OBL20", "EFF_MULTI2", "EFF_S7", "EFF_G3b", "RES_CUT_s050_dt0050", "DT_CUT_dt0050"):
        assert need in s3
    assert all(r.get("scheduled") is False for r in rows if r["stage"] == "S4")
    assert protocol["budget"]["cap_gpu_h"] == 18.0
    assert protocol["candidate"]["solver"] == "tlmpm_v6" and protocol["candidate"]["position_update"] == "grid_musl_v1"
    assert protocol["reopen_measure"]["rows"] == ["EFF_CUT", "EFF_TRIM", "EFF_MULTI2", "EFF_S7", "EFF_G3b"]


def test_rows_are_runnable_and_copy_frozen_specs(protocol):
    runner = load("run_wp12_gridpos")
    e1 = {r["id"]: r for r in json.loads((ROOT/"configs/wp12/phase4_e1_protocol_v1.json").read_text())["rows"]}
    for row in protocol["rows"]:
        assert row["kind"] in runner.DRIVERS or row["kind"] == "native", row["id"]
        if row["kind"] == "verify":
            spec = row["args"]["row"]
            assert spec["kind"] in ("res", "e1", "probe")
            if spec["kind"] == "e1":
                assert spec["e1_spec"] == e1[spec["e1_spec"]["id"]]
        if row.get("v5_standin"):
            assert (ROOT/row["v5_standin"]/"result.json").exists(), row["v5_standin"]
            assert row["gate"] == "characterization"


def test_queue_order(protocol):
    runner = load("run_wp12_gridpos")
    jobs = [(arm, rid) for _, arm, rid in runner.queue_jobs(protocol, "S1")]
    assert jobs[:2] == [("V6", "M1a"), ("V5", "M1a")]
    assert jobs[-6:] == [("V6", "M1a_r2"), ("V5", "M1a_r2"), ("V6", "M3b_r2"), ("V5", "M3b_r2"), ("V6", "FB_CUT_r2"),
                         ("V5", "FB_CUT_r2")]
    s3 = [(arm, rid) for _, arm, rid in runner.queue_jobs(protocol, "S3")]
    assert ("V6", "RES_CUT_s050_dt0050") in s3 and ("V5", "RES_CUT_s050_dt0050") not in s3
    assert not any(rid.startswith("NAT_") for _, rid in s3)


def _result(verdict="PASS", components=2, integrity=True, status="COMPLETED", trim=None):
    return {"status": status, "verdict": verdict, "integrity": {"pass": integrity, "checks": {}},
            "final_topology": {"mechanical_components": components}, "trim_numbers": trim}


def test_row_gate_kill_rule():
    S = load("summarize_wp12_gridpos")
    rows = S.protocol_rows()
    m1a, m3b, m4, m2, f29 = rows["M1a"], rows["M3b"], rows["M4"], rows["M2"], rows["F2.9_blunt_handle"]
    assert not S.row_gate(m1a, _result(), _result())["fail"]
    assert S.row_gate(m1a, _result("FAIL"), _result())["fail"]                                  # frozen criterion
    assert S.row_gate(m1a, _result(components=3), _result())["fail"]                            # components vs V5
    assert S.row_gate(m1a, _result(integrity=False), _result())["fail"]                         # integrity
    assert S.row_gate(m1a, _result(status="ERROR"), _result())["fail"]
    assert S.row_gate(m1a, _result(), None)["details"].get("components_v5") is None             # V5 NOT_RUN: no comparison
    assert S.row_gate(m3b, _result(trim={"removal": .89, "substrate": 1.}), _result())["fail"]
    assert S.row_gate(m3b, _result(trim={"removal": .95, "substrate": .985}), _result())["fail"]
    assert not S.row_gate(m3b, _result(trim={"removal": .95, "substrate": .999}), _result())["fail"]
    assert not S.row_gate(m4, _result(components=1), _result(components=1))["fail"]
    assert S.row_gate(m2, _result("FAIL"), _result("PASS"))["fail"]                             # worse than V5
    report = S.row_gate(f29, _result("FAIL", components=1), _result("FAIL", components=1))
    assert not report["fail"]                                                                    # report row: not gated
    assert S.row_gate(rows["M1a"], None, _result())["pending"]
    # S1: every failure kills, even when V5 fails too; S2/S3 acceptance: reported, not attributable to v6
    assert S.row_gate(m1a, _result("FAIL"), _result("FAIL"))["fail"]
    both = S.row_gate(rows["FB_TestB"], _result("FAIL", components=5), _result("FAIL", components=5))
    assert not both["fail"] and both["v5_also_fails"] and both["reasons"]
    assert S.row_gate(rows["FB_TestB"], _result("FAIL", components=4), _result("FAIL", components=5))["fail"]
    assert rows["4c_G07"]["gate"] == "known_failure"


def test_runner_refuses_bad_rows_and_paths(tmp_path):
    runner = load("run_wp12_gridpos")
    with pytest.raises(SystemExit):
        runner.main(["run", "--row", "NOPE", "--arm", "V6", "--out", str(tmp_path/"gridpos_dev_001/V6/NOPE"), "--dev"])
    with pytest.raises(SystemExit):     # DEV output outside gridpos_dev_NNN/<arm>/
        runner.main(["run", "--row", "M4", "--arm", "V6", "--out", str(tmp_path/"elsewhere/M4"), "--dev"])
    with pytest.raises(SystemExit):     # arm directory must match
        runner.main(["run", "--row", "M4", "--arm", "V6", "--out", str(tmp_path/"gridpos_dev_001/V5/M4"), "--dev"])
    with pytest.raises(SystemExit):     # repeat beyond the declared count
        runner.main(["run", "--row", "M4_r2", "--arm", "V6", "--out", str(tmp_path/"gridpos_dev_001/V6/M4_r2"), "--dev"])
    with pytest.raises(SystemExit):     # S4 Native is not scheduled
        runner.main(["run", "--row", "NAT_FB_CUT", "--arm", "V6", "--out", str(tmp_path/"gridpos_dev_001/V6/NAT_FB_CUT"), "--dev"])
    with pytest.raises(SystemExit):
        runner.main(["run", "--row", "M4", "--arm", "V6", "--out", str(tmp_path/"gridpos_dev_001/V6/M4"), "--dev-max-steps", "3"])
    assert not (tmp_path/"gridpos_dev_001").exists()


def test_frozen_runners_untouched():
    # Phase F v1 runner CPU pin (tests/test_phase_f_v2.py) and the other wrapped frozen drivers.
    assert hashlib.sha256((ROOT/"scripts/run_phase_f_material.py").read_bytes()).hexdigest() == \
        "5aea75d789bc0c46a954b0763fc9a5ed996da0f9b9c85749c7ab37e80e90b989"
    assert hashlib.sha256((ROOT/"scripts/run_wp12_resolution.py").read_bytes()).hexdigest() == \
        "a7e34259bef2470ac57d4f43d88cdb98d8a92a3f5eabdaad2b595cf9393f6eca"
    assert hashlib.sha256((ROOT/"scripts/run_wp12_phase4_e1.py").read_bytes()).hexdigest() == \
        "8fec36362b9f4eac9beda94abc9c3b62aac163bc5888c03b9fd8e5310f976e60"
