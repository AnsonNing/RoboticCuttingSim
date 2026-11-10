"""Pins of the P1-6 acceptance contracts (Phase 1 xhigh acceptance review, finding 2; decision D-14).

``acceptance_v1.json`` was amended in place four times before any baseline while it was declared frozen; it is frozen
now at the bytes below (the Phase 1 merge). ``acceptance_v2.json`` is the contract of every real acceptance; it
freezes at the commit that merges it into ``main`` (its pin changed once before that, when the G9 window was
re-declared to 229-400 after the follow-up review, before any candidate was judged: ``a008dd38…`` -> ``b66ace29…``). Any change to either file is a new version
(``acceptance_v3.json``), never an edit: these tests fail on any byte change. The band files they point at are pinned
too, because ``physics.position_band.path`` makes them part of the contract.
"""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "configs" / "rl_benchmark"
PINS = {
    "acceptance_v1.json": "3592d9408f314c3d2825171361b8ec640534c60577bb6216a10c4f2b22897bf4",
    "acceptance_v2.json": "b66ace2967b4aad8382ca4297032c0e513a4cf3ebe6564dad03494b8e7dd612a",
    "position_band_v2.json": "c60a9d58950f02051f42c58ef28d00b3602ce38c38b0aa7918de8ae3d5331fac",
    "position_band_p1_0.json": "6e8aaffaf6111dd90d7727939f8e90229791b32de0adc94f814b854e666564d0",
}


@pytest.mark.parametrize("name", sorted(PINS))
def test_acceptance_file_is_byte_identical_to_its_pin(name):
    assert hashlib.sha256((BENCH / name).read_bytes()).hexdigest() == PINS[name], \
        f"{name} changed: a frozen acceptance file is never edited; declare the next version instead"


@pytest.mark.parametrize("name, version", [("acceptance_v1.json", "acceptance_v1"),
                                           ("acceptance_v2.json", "acceptance_v2")])
def test_version_field_matches_the_file_name(name, version):
    """The external caller refuses a config whose ``version`` disagrees with its file name."""
    cfg = json.loads((BENCH / name).read_text())
    assert cfg["version"] == version and cfg["schema"] == "rcs_perf_acceptance_v1"


def test_every_acceptance_config_is_pinned():
    assert {p.name for p in BENCH.glob("acceptance_v*.json")} <= set(PINS)


def test_acceptance_configs_point_at_pinned_band_files():
    for name in ("acceptance_v1.json", "acceptance_v2.json"):
        band = json.loads((BENCH / name).read_text())["physics"]["position_band"]["path"]
        assert Path(band).name in PINS and (ROOT / band).is_file()
