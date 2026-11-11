"""P2-6 frozen-source pins (P6.6): the accepted P2-6 modules of record for rubber_rl_tests_v2 / v3 / v4 and RL-09.

A failure means a pinned module changed after P6.6: a behaviour change needs a new versioned module or protocol, never
a silent edit (update a pin only with a declared reason in the commit). CPU only.
"""
import hashlib

from robotic_cutting_sim.rubber_material import ROOT

PACKAGE = ROOT/"src"/"robotic_cutting_sim"
PINS = {
    "rl/rubber_env.py": "7f70e81758612de19520f768d85138a6bed4b6e1006c4fe51b4c104dda022d48",
    "rl/rubber_family.py": "e265fc7ac27a1a3c558d5cc16b9a7384623ef7b38c08eca5315a7402c3e08b06",
    "rl/rubber_contract.py": "32857340274bf59931c5ab71da7eb68ce1647c94d7966787588f66c890a76f6d",
    "rl/rubber_scripted.py": "4ddf48e95509b04e5e12ba39519e421d3bb41ca1261d80a75b49e5d62060caf4",
    "rubber_contact_envelope.py": "b696dfc006e3e70b4dfe2dbd66ead49cf13c29687702aa98059ed02e83a3df9f",
    "native_rubber_host.py": "72356f0ddac8575b408a5dae2279f76907e31d531dbad27ce0010d2253b24596",
    "rubber_native_compare.py": "ae4428ed680b0664e05d439f244cdd1ba8231e445e3ba71f684378cd71c6023c",
}


def test_p2_6_sources_unchanged():
    for name, pin in PINS.items():
        assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() == pin, f"{name} was modified"


def test_vector_env_dispatch_branch_present():
    text = (PACKAGE/"rl"/"vector_env.py").read_text()
    assert text.count('backend == "kinematic_grip_batched"') == 1 and "from .rubber_env import open_rubber_vector_env" in text
