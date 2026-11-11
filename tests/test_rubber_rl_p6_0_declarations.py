"""P6.0 declarations (ADR 0023 §23.2 decision 7, plan §4.8) are frozen at commit: the task config
``configs/rubber/rl/rubber_strip_drag_v1.json`` and the protocol ``configs/rubber/rubber_rl_tests_v1.json`` were
declared together, alone, before any P2-6 protocol run. A failure means a frozen file was edited: never update a pin,
declare v2 instead. CPU only, no Warp / Torch."""
import hashlib
import json

from robotic_cutting_sim.rubber_contact_envelope import load_envelope
from robotic_cutting_sim.rubber_material import ROOT

PINS = {"configs/rubber/rl/rubber_strip_drag_v1.json":
        "fba43424372e17fe926fad697ff8336ea7c5ed0c2d8ecdbdefb9978a994f4397",
        "configs/rubber/rubber_rl_tests_v1.json": "8e4b2593e968532e10808d4828f093df964747c18c8b98d260d1eae68535fd4c"}


def _sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def test_frozen_files_unchanged():
    for name, pin in PINS.items():
        assert _sha(name) == pin, f"{name} was modified"


def test_task_config_is_consistent_with_its_inputs():
    cfg = json.loads((ROOT/"configs"/"rubber"/"rl"/"rubber_strip_drag_v1.json").read_text())
    assert cfg["calibration_status"] == "SIMULATOR_ONLY_UNCALIBRATED" and cfg["backend"] == "kinematic_grip_batched"
    assert _sha(cfg["recipe"]["path"]) == cfg["recipe"]["sha256"]
    env = load_envelope(ROOT/cfg["contact"]["envelope"]["path"], expected_sha256=cfg["contact"]["envelope"]["sha256"])
    n_ts = sorted({s["n_t"] for s in cfg["slots"]})
    assert n_ts == [4, 5, 6] and len(cfg["slots"]) == 9
    assert tuple(cfg["randomisation"]["ranges"]["friction_table"]) == env.friction_range(cfg["contact"]["law"], n_ts)
    assert cfg["observation"]["policy"]["dim"] == sum(d for _, d in cfg["observation"]["policy"]["terms"]) == 78
    assert cfg["observation"]["critic"]["dim"] == sum(d for _, d in cfg["observation"]["critic"]["terms"]) == 106
    assert cfg["action"]["dim"] == 6 and cfg["done"]["max_episode_steps"] == 250 and cfg["preload"]["steps"] == 10
    v3 = json.loads((ROOT/"configs"/"rubber"/"rubber_contact_tests_v3.json").read_text())
    assert cfg["contact"]["law_parameters"] == v3["common"]["law_parameters"]


def test_protocol_declares_rl01_to_rl11_and_pins_its_inputs():
    p = json.loads((ROOT/"configs"/"rubber"/"rubber_rl_tests_v1.json").read_text())
    assert set(p["tests"]) == {f"RL-{i:02d}" for i in range(1, 12)}
    assert p["prerequisites"]["task_config"]["sha256"] == PINS["configs/rubber/rl/rubber_strip_drag_v1.json"]
    assert p["prerequisites"]["contact_envelope"]["sha256"] == _sha(p["prerequisites"]["contact_envelope"]["path"])
    assert p["tests"]["RL-04"]["thresholds"] == {"k_per_side": 5, "lam": 4.0, "eps_rel": 1e-06}
