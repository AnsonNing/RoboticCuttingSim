"""P2-6 frozen-source pins (P6.6): the accepted P2-6 modules of record for rubber_rl_tests_v2 / v3 / v4 and RL-09.

The original digests are audited against the frozen commit. The three additive high-speed API
edits have their own declared digests and separate Reference behavior checks. CPU only.
"""
import hashlib
import subprocess

import pytest

from robotic_cutting_sim.rubber_material import ROOT

PACKAGE = ROOT/"src"/"robotic_cutting_sim"
# Pin updates (declared reasons):
#   P6.7 review Z-a / Z-b / F9: rl/rubber_env.py (F2 privileged_info_keys, F4 UNSUPPORTED_OPTION for Phase 1 options),
#   rl/rubber_contract.py (F2 additive clause), native_rubber_host.py (F9 stale docstring / EPISODE default text);
#   batched/rubber_material_batch.py added (F7) after the F1 bounded LRU and the F3 InvalidBatchEnvs base.
#   Physics and the policy observation are byte-identical (CPU L0 / parity tests).
PINS = {
    "batched/rubber_material_batch.py": "b7e4fc7af6b1849d2460b55f768d03808e94753ec4a952204dd77f7b8b7afcf7",
    "rl/rubber_env.py": "b8766171ae27f50dc9eb2dc847f35e64de5b47fa51363c5d31097b396e5a83ff",
    "rl/rubber_family.py": "e265fc7ac27a1a3c558d5cc16b9a7384623ef7b38c08eca5315a7402c3e08b06",
    "rl/rubber_contract.py": "7b5ef01ae7f67686a83f1f69cb4cd5b9479c336c0f7874cb0b0bfa24c2ecc24e",
    "rl/rubber_scripted.py": "4ddf48e95509b04e5e12ba39519e421d3bb41ca1261d80a75b49e5d62060caf4",
    "rubber_contact_envelope.py": "b696dfc006e3e70b4dfe2dbd66ead49cf13c29687702aa98059ed02e83a3df9f",
    "native_rubber_host.py": "16153a6d4d662d7b094dbd54dad1fe394f965f65d8382913affeba89a162ff5b",
    "rubber_native_compare.py": "ae4428ed680b0664e05d439f244cdd1ba8231e445e3ba71f684378cd71c6023c",
}


# High-speed opt-in profiles, effective receipts and eager benchmarking. The original
# task declaration, ordinary Reference timestep, observation and evaluator stay pinned by behavior tests.
REFERENCE_COMMIT = "e13919ded49cf6de9ee8bcb5d62138c4991bda2e"
HIGH_SPEED_PINS = {
    "batched/rubber_material_batch.py": "66b5b2fa995951478490b8c9b8f5f8f6242750ade2b4156186331f82de7143a1",
    "rl/rubber_env.py": "64b715bed7a6c66c5d3d0bf65a6f1dc110f553f8d9d54eceeee7d9c2be925c76",
    "rl/rubber_family.py": "fccff1febfa9ee746ff0776443f1dabdf5db938c0b71b7a6c22bacdfff4229c4",
}


def test_historical_p2_6_source_digests():
    try:
        present = subprocess.run(["git", "cat-file", "-e", REFERENCE_COMMIT], cwd=ROOT,
                                 capture_output=True, check=False)
    except FileNotFoundError:
        pytest.skip("Git is needed to inspect the P2-6 source commit")
    if present.returncode != 0:
        pytest.skip("P2-6 source commit unavailable in this checkout")
    for name, pin in PINS.items():
        original = subprocess.check_output(
            ["git", "show", f"{REFERENCE_COMMIT}:src/robotic_cutting_sim/{name}"], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == pin, f"{name} differs from frozen P2-6 evidence"


def test_current_rubber_source_digests():
    for name, pin in (PINS | HIGH_SPEED_PINS).items():
        assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() == pin, f"{name} was modified"


def test_vector_env_dispatch_branch_present():
    text = (PACKAGE/"rl"/"vector_env.py").read_text()
    assert text.count('backend == "kinematic_grip_batched"') == 1 and "from .rubber_env import open_rubber_vector_env" in text
