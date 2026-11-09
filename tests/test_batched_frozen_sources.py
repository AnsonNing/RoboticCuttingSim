"""ADR 0021 status: the batched runtime changes no frozen solver, coupling path or evaluator.

SHA256 pins of the frozen modules (as committed at the start of P1-1). The v7_2 solver and frontier pins equal
the hashes recorded in the WP-17 receipts (docs/evidence/wp17/performance_001.json). A failure here means a
frozen file was edited; that is never fixed by updating the pin.
"""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/"src"/"robotic_cutting_sim"
PINS = {
    "tlmpm_v5.py": "a7a05b2fe709fbff2e915c0df0855079d5bccbd8e766c787d2f628e7cc3bf34b",
    "tlmpm_v6.py": "13a3adff0fa6b8a5d57fd894fcb6df17ca19fd78578b7c51d4541d583cb7ce44",
    "tlmpm_v7.py": "4c7a75f489f110b884fa36350608b501adc5ed2f77252f0777f725bf88fd7266",
    "tlmpm_v7_1.py": "7549044028308a32b7265257bbeaae2cd5ae3fab38fc79aec1ef11e4c63549f3",
    "tlmpm_v7_2.py": "052e2c448f62c9f0eedf3c405e2c30133ecd44112c90a0f148cdb31f4948f30e",
    "tlmpm_v7_2_contacts.py": "1e74476bd39180e86ad86bec40dd9728a0f3f7e2317f14cb9db10268705446d4",
    "tlmpm_v7_2_frontier.py": "d2af0151c1f34f37bc80fdf7096d7f96d2d14127e11bc887408d8cd086d87fb6",
    "fixture_contacts.py": "651cf168808b7e0bd5c1a75cc7e405f54470c056ee050a62f3d824212ba4b4e8",
    "material_diagnostics.py": "57e48aa3c2ecec092d7079dd5c81be41d2d602b8d188097a7e6b3bc8695165db",
    "live_material.py": "58466e5a1f974c3c433aace2895ad7577e93fb59e376ad4f508969bad86dffb7",
    "coupled_robot.py": "040b7595cd329a523400ffdf86f16f057d6d2cc7ea242893dd4c21a640851316",
    "evaluation.py": "ba7b93eff2e39e835dce053de5558bee58025e5dc9885440a6035feefa4d8ca1",
    "surface_evaluation.py": "018b035d3976b593eff4fc57323e1223da38d31391bc95ef6202ebf9b979fb59",
    "multi_cut_evaluation.py": "260c68ba5d503bae70eef77598fa9f723ce745ad2b25701387e276f588b3d5b0",
    "demo_evaluation.py": "e1a52457c476cb0b48ad6235f70f02f106be720de7a119d59498577e8d045767",
    "seeded_trim.py": "ed1a89228259ac558b8ebe70b995180299ac8534296a01660cd5bd8adf0046c1",
}


@pytest.mark.parametrize("name", sorted(PINS))
def test_frozen_source_unchanged(name):
    assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() == PINS[name], f"{name} was modified"


def test_batch_receipts_report_the_pinned_sources():
    from robotic_cutting_sim.batched.layout import FROZEN_SOURCES, frozen_source_sha256
    hashes = frozen_source_sha256()
    assert set(FROZEN_SOURCES) <= set(PINS)
    assert all(hashes[name] == PINS[name] for name in FROZEN_SOURCES)


def test_v7_2_pins_match_wp17_receipts():
    text = (ROOT/"docs"/"evidence"/"wp17"/"performance_001.json").read_text()
    json.loads(text)
    for name in ("tlmpm_v7_2.py", "tlmpm_v7_2_frontier.py"):
        assert PINS[name] in text, name
