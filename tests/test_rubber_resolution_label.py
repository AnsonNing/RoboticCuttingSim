"""K2 review K2-2 (Q1w decision option (c), conditions 1 and 2): every resolution profile carries ``measured_bias`` and
``known_limitation`` with width qualifiers, the scene compiler validates the profile against n_t, and the label reaches
the scene, single-env (LiveRubberMaterial, record header) and batched receipts. Values from protocol v5 evidence
(docs/evidence/rl_ready/p2_2_v5_001.json)."""
import json
from pathlib import Path

import pytest

from robotic_cutting_sim import rubber_material as rm
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_material import RESOLUTION_PROFILES, MaterialValues

ROOT = Path(__file__).resolve().parents[1]
STEP_MATERIAL = MaterialValues(mu0=1e6, kappa=2e7, density=1000.0)


def test_every_profile_carries_bias_and_limitation():
    for name, prof in RESOLUTION_PROFILES.items():
        assert prof["measured_bias"]["schema"] == "rcs_resolution_bias_v1", name
        lim = prof["known_limitation"]
        assert lim["id"] and lim["summary"] and lim["decision"] and lim["cite_when"], name
        lo, hi = prof["cells_through_thickness_band"]
        assert lo == prof["min_cells_through_thickness"], name
    ref = RESOLUTION_PROFILES["rubber_ref_v1"]
    assert ref["known_limitation"]["id"] == "rubber_phys_11_v5_q1w_gci_2.16pct"
    assert ref["label"].endswith(ref["known_limitation"]["id"])
    q = ref["measured_bias"]["observables"]
    assert (q["Q1w"]["bias_pct"], q["Q1w"]["bias_range_pct"]) == (0.62, [0.50, 0.78])
    assert (q["Q1w"]["gci_production_pct"], q["Q1w"]["fs"], q["Q1w"]["p_obs"]) == (2.16, 3.0, 3.12)
    assert (q["Q1"]["bias_pct"], q["Q2"]["bias_pct"]) == (0.39, -0.39)
    # width qualifiers: the geometry of every signed number names W and L/t; the -02 rows cover W = t and W = 4t
    for key in ("Q1", "Q1w"):
        assert "W = " in q[key]["geometry"] and "L/t 20" in q[key]["geometry"]
    assert set(ref["measured_bias"]["width_and_slenderness_RUBBER_PHYS_02"]) >= {
        "L_over_t_10_W_t", "L_over_t_10_W_4t", "L_over_t_20_W_t", "L_over_t_20_W_4t"}
    assert "W = 4t" in ref["known_limitation"]["summary"] and "W = t" in ref["known_limitation"]["summary"]
    rl = RESOLUTION_PROFILES["rubber_rl_v1"]["measured_bias"]["observables"]
    assert (rl["Q1"]["bias_pct"], rl["Q1w"]["bias_pct"], rl["Q2"]["bias_pct"]) == (0.97, 1.39, -0.83)
    assert RESOLUTION_PROFILES["rubber_coarse_v1"]["measured_bias"]["status"] == "NOT_MEASURED"


def test_q4_is_oscillatory_not_signed():
    for name in ("rubber_ref_v1", "rubber_rl_v1"):
        q4 = RESOLUTION_PROFILES[name]["measured_bias"]["observables"]["Q4"]
        assert q4["bias_pct"] is None and q4["sign"].startswith("OSCILLATORY") and q4["abs_bias_pct_max"] > 0


def test_numbers_match_the_v5_evidence():
    """Point values (p_eff 2 extrapolation of n_t 4 / 6 / 8) and ranges (n_t 8 value .. p = 1 extrapolation of
    n_t 6 / 8) recomputed from the committed v5 evidence table, to the quoted 0.01 %."""
    table = json.loads((ROOT/"docs"/"evidence"/"rl_ready"/"p2_2_v5_001.json").read_text())["refinement_v5"]["table"]
    for name, n_t in (("rubber_ref_v1", 4), ("rubber_rl_v1", 3)):
        obs = RESOLUTION_PROFILES[name]["measured_bias"]["observables"]
        for key in ("Q1", "Q1w", "Q2"):
            v = {int(k): x for k, x in table[key]["values_cfl0.5"].items()}
            point = 100*table[key]["rubber_rl_v1_bias_reported_not_scored"][f"nt{n_t}_over_ext_p_eff_minus_1"]
            ext1 = v[8]-(v[6]-v[8])/(8/6-1)
            ends = sorted((100*(v[n_t]/v[8]-1), 100*(v[n_t]/ext1-1)))
            assert obs[key]["bias_pct"] == pytest.approx(point, abs=0.006), (name, key)
            assert obs[key]["bias_range_pct"] == pytest.approx(ends, abs=0.006), (name, key)
    assert table["Q4"]["triplet_4_6_8"]["status"] == "OSCILLATORY"
    assert table["Q1w"]["production_gci_nt4"] == pytest.approx(0.0216, abs=5e-5)


def test_profile_band_and_refusals():
    assert [rm.profile_for_n_t(n) for n in (2, 3, 4, 6, 8)] == ["rubber_coarse_v1", "rubber_rl_v1"] + \
        ["rubber_ref_v1"]*3
    assert rm.profile_for_n_t(0.004/(0.004/3)) == "rubber_rl_v1"
    with pytest.raises(rm.RubberRecipeError, match="no resolution profile"):
        rm.profile_for_n_t(1)
    for n_t, name in ((3, "rubber_ref_v1"), (4, "rubber_rl_v1"), (2, "rubber_rl_v1"), (6, "rubber_rl_v1")):
        with pytest.raises(ValueError, match="covers n_t"):
            rs.compile_strip(rs.RubberStripSpec(length=0.012, width=0.004, thickness=0.004, n_t=n_t,
                                                resolution_profile=name))
    with pytest.raises(ValueError, match="unknown resolution profile"):
        rs.compile_strip(rs.RubberStripSpec(length=0.012, width=0.004, thickness=0.004, n_t=4,
                                            resolution_profile="rubber_full_v1"))


@pytest.mark.parametrize("n_t,profile,expected", [(3, None, "rubber_rl_v1"), (4, None, "rubber_ref_v1"),
                                                  (4, "rubber_ref_v1", "rubber_ref_v1"), (2, None, "rubber_coarse_v1"),
                                                  (6, None, "rubber_ref_v1")])
def test_scene_receipt_carries_the_label(n_t, profile, expected):
    sc = rs.compile_strip(rs.RubberStripSpec(length=0.012, width=0.004, thickness=0.004, n_t=n_t,
                                             resolution_profile=profile))
    r = sc.receipt()
    rs.validate_scene_receipt(r)
    res = r["resolution"]
    assert r["resolution_profile"] == res["resolution_profile"] == expected
    assert res["known_limitation"] == RESOLUTION_PROFILES[expected]["known_limitation"]
    assert res["label"] == RESOLUTION_PROFILES[expected]["label"]
    assert res["bias_measured_at_this_n_t"] == (n_t in (3, 4))
    if expected != "rubber_coarse_v1":
        assert res["measured_bias"]["observables"]["Q1w"].startswith("+")
    bad = dict(r, resolution=dict(res, resolution_profile="rubber_coarse_v1" if expected != "rubber_coarse_v1"
                                  else "rubber_ref_v1"))
    with pytest.raises(ValueError, match="resolution label"):
        rs.validate_scene_receipt(bad)


def _live(t=0.003, n_t=3, profile=None):
    spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=t, n_t=n_t, case="A", resolution_profile=profile)
    return LiveRubberMaterial(spec, material=MaterialValues(mu0=4e5, kappa=8e6, density=1150.0), control_dt=2e-4,
                              gravity=(0, 0, -9.81), step_material=STEP_MATERIAL)


def test_live_receipt_and_record_header_carry_the_label():
    live = _live(t=0.004, n_t=4, profile="rubber_ref_v1")
    rc = live.receipt()
    assert rc["resolution"]["resolution_profile"] == "rubber_ref_v1"
    assert rc["resolution"]["measured_bias"] == RESOLUTION_PROFILES["rubber_ref_v1"]["measured_bias"]
    assert rc["resolution"]["known_limitation"]["id"] == "rubber_phys_11_v5_q1w_gci_2.16pct"
    assert rc["scene"]["resolution"]["known_limitation"]["id"] == "rubber_phys_11_v5_q1w_gci_2.16pct"
    assert live.header()["scene"]["resolution"]["resolution_profile"] == "rubber_ref_v1"


def test_batched_receipt_carries_every_env_label():
    lives = [_live(t=0.003, n_t=3), _live(t=0.004, n_t=4), _live(t=0.003, n_t=3)]
    batch = rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True)
    rc = batch.receipt()
    res = rc["resolution"]
    assert res["per_env"] == ["rubber_rl_v1", "rubber_ref_v1", "rubber_rl_v1"]
    assert set(res["profiles"]) == {"rubber_rl_v1", "rubber_ref_v1"}
    for name, block in res["profiles"].items():
        assert block["measured_bias"] == RESOLUTION_PROFILES[name]["measured_bias"]
        assert block["known_limitation"] == RESOLUTION_PROFILES[name]["known_limitation"]
    for e, env in enumerate(rc["envs"]):
        assert env["scene"]["resolution"]["resolution_profile"] == res["per_env"][e]
