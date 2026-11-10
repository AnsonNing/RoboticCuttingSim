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
    # Review note N5 (2480dc2 review): the evaluator path (``evaluate()`` runs ``goals.evaluate_goal`` via
    # ``scene_config``), the policy/control surface and the pre-v5 solvers. Pinned at their f242881^ content (the
    # RL-ready start; ``git diff f242881^ 2480dc2`` is empty for these files).
    "goals.py": "c2cb544a0dee129d71054c0df3266601fbc6e75b41e94c6a00a735812bba8602",
    "policy_env.py": "70c13aab61a1c23b2d68d8096da932f45416b9bb7eae067da36ac2ef1dd8f039",
    "control.py": "040c567b29c4d912db6da2faa33ccd46777eadeaa78b5791115c10995ee51b71",
    "scene_config.py": "ff32062a24a6b972ec1918bba41cc691a7bcc964b79be5b9c99f30ef8bce79ee",
    "tlmpm.py": "a39c151cb8122a739b18991699527d2557219b5945d7e44be5ca50fd6e7ff7fe",
    "tlmpm_v2.py": "b2946e95dcb70492d1e06f210681788ced6e5f54fdf343aa31cd69ca85ff66b0",
    "tlmpm_v3.py": "9073036b0a86076ae44ad268a91427a4ce20c32133f8759d67595bd4836a6b1a",
    "tlmpm_v4.py": "cb8985ebbfe530be741025f009400622f6b72cdceacf239f952612c91c2eaea9",
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
