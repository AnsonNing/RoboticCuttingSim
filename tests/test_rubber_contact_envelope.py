"""ENV1: the rubber contact friction envelope ``configs/rubber/contact_envelope_v1.json`` (ADR 0023 §23.2 decision 4).

The file is SHA-pinned (a failure means it was edited: declare contact_envelope_v2 instead, never update the pin). The
ESTABLISHED rows are re-derived here from the formal evidence files (contact_v2_001, contact_v3_001) and the loader's
refusals are checked. CPU only, no Warp.
"""
import hashlib
import json

import pytest

from robotic_cutting_sim.rubber_contact_envelope import (FrictionOutOfEnvelope, REFUSAL_CODE, load_envelope,
                                                         parse_envelope)
from robotic_cutting_sim.rubber_material import ROOT

PINS = {"configs/rubber/contact_envelope_v1.json": "08f68f704a8052213be4448f964f1b0fc31da867470d7cc0274f93c71ba1a77a",
        "configs/rubber/contact_envelope_v2.json": "46f74efa880cf7121e50070a0cf5fb2358e87d7e390f67cd7b608802f9f03b78"}
V2, V3 = "grid_node_penalty_coulomb_v2", "grid_node_penalty_coulomb_v3"
EVID = ROOT/"docs"/"evidence"/"rl_ready"


def test_frozen_files_unchanged():
    for name, pin in PINS.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == pin, f"{name} was modified"


def test_rows_and_protocol_pins():
    env = load_envelope(expected_sha256=PINS["configs/rubber/contact_envelope_v1.json"])
    assert env.id == "contact_envelope_v1"
    assert {(r["law"], r["n_t"]) for r in env.rows} == {(V2, 3), (V2, 4), (V2, 5), (V2, 6),
                                                         (V3, 3), (V3, 4), (V3, 5), (V3, 6)}
    est = {(r["law"], r["n_t"]): r for r in env.rows if r["status"] == "ESTABLISHED"}
    assert set(est) == {(V2, 3), (V3, 3), (V3, 4)}
    assert (est[(V2, 3)]["sliding_mu_max"], est[(V2, 3)]["static_mu_max"]) == (0.6, 0.8)
    for n_t in (3, 4):
        assert (est[(V3, n_t)]["sliding_mu_max"], est[(V3, n_t)]["static_mu_max"]) == (0.8, 0.8)
    for r in est.values():                                  # the cited protocol files are the pinned ones
        assert hashlib.sha256((ROOT/r["protocol"]).read_bytes()).hexdigest() == r["protocol_sha256"]


def test_v3_rows_follow_from_contact_v3_001():
    run = json.loads((EVID/"p2_4_contact_x3_001.json").read_text())["formal_cpu_run"]
    assert run["meta"]["protocol_sha256"].startswith("36d70bea") and run["meta"]["dirty"] is False
    crit = run["verdicts"]["RUBBER-CONT-01"]["criteria"]
    assert all(crit[k] == "PASS" for k in ("a_stick", "b_slip", "c_onset", "e_validity", "f_ramp_slip"))
    assert crit["d_books"] == "FAIL"                        # the declared known limitation of the row
    assert run["verdicts"]["RUBBER-CONT-04"]["verdict"] == "PASS"
    for n_t in (3, 4):
        for scene in ("INC", "INC2"):
            g = run["groups"][f"{scene}_nt{n_t}"]
            assert g["max_abs_slip_err"] <= 0.01 and abs(g["D04_min"]-1) <= 0.05 and abs(g["D04_max"]-1) <= 0.05
            assert g["max_abs_hold_disp_rel_s"] <= 0.01
    mus = {(o["scene"], o["n_t"], o["mu"]) for o in run["onset"] if o["ok"]}
    assert {(s, n, m) for s in ("INC", "INC2") for n in (3, 4) for m in (0.2, 0.4, 0.6, 0.8)} <= mus


def test_v2_row_follows_from_contact_v2_001():
    d = json.loads((EVID/"p2_4_contact_x1_001.json").read_text())
    assert d["formal_run_v2"]["meta"]["protocol_sha256"].startswith("89bc7680")
    slip = dict(d["v1_vs_v2"]["RUBBER-CONT-01_slip"]["v2"])
    diss = dict(d["v1_vs_v2"]["RUBBER-CONT-04"]["v2"])
    ok = {n: abs(slip[n]) <= 0.01 and abs(diss[n]-1) <= 0.05 for n in slip}
    assert all(ok[n] for n in ok if "mu0.4" in n or "mu0.6" in n)              # sliding 0.4 / 0.6 PASS
    assert not all(ok[n] for n in ok if "mu0.8" in n)                          # mu 0.8 outside the row
    stick = dict(d["v1_vs_v2"]["RUBBER-CONT-01_stick"]["v2"])
    assert any("mu0.8_r0.95" in n for n in stick) and d["mu_envelope"]["static_friction_supported_mu_max"] == 0.8
    assert d["mu_envelope"]["sliding_friction_supported_mu_max"] == 0.6


def test_refusals_and_range():
    env = load_envelope()
    assert env.friction_range(V3, [3, 4]) == (0.3, 0.8)
    assert env.friction_range(V2, [3]) == (0.3, 0.6)
    assert env.check_friction(V3, 4, 0.8) == 0.8
    for law, n_t, mu in ((V3, 4, 0.81), (V3, 4, 1.0), (V3, 3, 1.2), (V2, 3, 0.7), (V3, 5, 0.3), (V2, 4, 0.3),
                         ("grid_node_penalty_coulomb_v1", 3, 0.3), (V3, 7, 0.3), (V3, 4, float("nan")),
                         (V3, 4, -0.1)):
        with pytest.raises(FrictionOutOfEnvelope) as err:
            env.check_friction(law, n_t, mu)
        assert err.value.code == REFUSAL_CODE and str(err.value).startswith(REFUSAL_CODE)
    assert env.check_friction(V2, 3, 0.8, kind="static") == 0.8
    with pytest.raises(FrictionOutOfEnvelope):
        env.friction_range(V3, [4, 5, 6])                 # n_t 5 / 6 not established before ENV2
    with pytest.raises(FrictionOutOfEnvelope):
        env.friction_range(V3, [4], lower=0.9)            # empty


def test_parser_rejects_malformed():
    base = json.loads((ROOT/"configs"/"rubber"/"contact_envelope_v1.json").read_text())
    for mutate in (lambda d: d.update(schema="x"),
                   lambda d: d["rows"][4].update(sliding_mu_max=1.0),
                   lambda d: d["rows"][1].update(sliding_mu_max=0.4),
                   lambda d: d["rows"].append(dict(d["rows"][0])),
                   lambda d: d["rows"][0].pop("protocol_sha256"),
                   lambda d: d["rows"][0].update(status="DIAGNOSTIC")):
        d = json.loads(json.dumps(base))
        mutate(d)
        with pytest.raises(ValueError):
            parse_envelope(d)


def test_v2_adds_only_the_env2_rows():
    """contact_envelope_v2 (ENV2, run contact_envelope_v1_001): v3 at n_t 5 / 6 established at 0.8 / 0.8 from the
    formal check; the n_t 4 row keeps 0.8 (the STRIP rows did not narrow it); every other row unchanged."""
    v1 = load_envelope()
    v2 = load_envelope(ROOT/"configs"/"rubber"/"contact_envelope_v2.json",
                       expected_sha256=PINS["configs/rubber/contact_envelope_v2.json"])
    ev = json.loads((EVID/"p2_6_contact_envelope_001.json").read_text())
    assert ev["verdict"] == "PASS" and ev["run"]["meta"]["dirty"] is False and not ev["run"]["errors"]
    assert ev["protocol"]["sha256"] == "d7ecbd3164409cce12d229504864297677baf3307f9763bb7797329506ffe92c"
    assert ev["output"]["sha256"] == PINS["configs/rubber/contact_envelope_v2.json"]
    assert ev["run"]["jobs"] == 85 and all(v["sliding_pass"] and v["static_pass"] for v in ev["levels"].values())
    for n_t in (5, 6):
        r = v2.established(V3, n_t)
        assert (r["sliding_mu_max"], r["static_mu_max"]) == (0.8, 0.8) and r["run_id"] == "contact_envelope_v1_001"
        assert ev["envelope"][str(n_t)]["sliding_mu_max"] == 0.8
    assert v2.established(V3, 4)["env2_check"]["sliding_mu_max"] == 0.8
    for r in v1.rows:
        if not (r["law"] == V3 and r["n_t"] in (4, 5, 6)):
            assert v2.row(r["law"], r["n_t"]) == r
    assert v2.friction_range(V3, [4, 5, 6]) == (0.3, 0.8)
    for mu in (0.81, 1.0):
        with pytest.raises(FrictionOutOfEnvelope):
            v2.check_friction(V3, 6, mu)
