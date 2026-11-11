"""P2-5 declarations: rubber_calib_tests_v1 is SHA-pinned (declared alone before any pilot or formal run), its
experiments are consistent with the forward models (lattice, splits, metrics), and the run scripts carry the pins."""
import hashlib
import json
import re

import pytest

from robotic_cutting_sim.rubber_material import ROOT

PROTOCOL = ROOT/"configs"/"rubber"/"rubber_calib_tests_v1.json"
PROTOCOL_SHA = "1f6816411d141555973537995300ba87927a8162e8eba5f942d51faf40b2ff2b"
SCRIPTS = ROOT/"scripts"/"rubber"/"p2_5"


def test_protocol_is_pinned():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA


@pytest.mark.parametrize("name", ["gen_synthetic.py", "run_pilot.py", "run_fit.py", "run_heldout.py"])
def test_scripts_carry_the_protocol_pin(name):
    text = (SCRIPTS/name).read_text()
    assert re.search(r'^PROTOCOL_SHA = "%s"' % PROTOCOL_SHA, text, re.M)


def test_protocol_content():
    p = json.loads(PROTOCOL.read_text())
    assert p["labels"]["calibration_status"] == "SIMULATOR_ONLY_UNCALIBRATED"
    assert p["labels"]["validation_kind"] == "NUMERICAL_VALIDATION_SYNTHETIC"
    assert "RESOLUTION_SPECIFIC" in p["labels"]["resolution_specific"]
    assert "TRANSLATIONAL" in p["common"]["contact_scope_K2_1"].upper()
    ids = [e["id"] for e in p["experiments"]]
    assert len(ids) == len(set(ids)) == 13
    assert {e["id"] for e in p["experiments"] if e["split"] == "fit"} == {"T1", "T2", "R1", "C1", "G1", "F1"}
    assert set(p["tests"]) == {f"CAL-0{i}" for i in range(1, 10)}
    # every specimen on the particle lattice of both fidelities
    from robotic_cutting_sim import rubber_calibration_experiments as rce
    from robotic_cutting_sim import rubber_calibration_synthetic as rcs
    from robotic_cutting_sim import rubber_scene as rs
    for fid in ("fit", "truth"):
        cfg = rce.model_config(p["common"], fid)
        for e in rcs.protocol_experiments(p):
            spec = rce.build_spec(e, cfg)
            dx = spec.grid_dx()
            for v in (spec.length, spec.width, spec.thickness):
                k = v/(dx/2)
                assert abs(k-round(k)) < 1e-6, (fid, e["id"], v)
            assert rce.steps(e, cfg) >= 1
    # the parameter box respects the step material
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    sp = rcf.space_from_protocol(p)
    assert abs(sp.mu0_max()-p["common"]["step_material"]["mu0_max_pa"]) < 1e-6
    assert rs.MARKERS == 9
