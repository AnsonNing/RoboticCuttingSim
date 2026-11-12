"""P6.0 declarations (ADR 0023 §23.2 decision 7, plan §4.8) are frozen at commit: the task config
``configs/rubber/rl/rubber_strip_drag_v1.json`` and the protocol ``configs/rubber/rubber_rl_tests_v1.json`` were
declared together, alone, before any P2-6 protocol run. v1 (130ccd4) is superseded by v2 after the K2 review
finding K2-1 (the v3 contact law is wrong for spin / sideways drag) and was never run formally; both stay pinned. A failure means a frozen file was edited: never update a pin,
declare v2 instead. CPU only, no Warp / Torch."""
import hashlib
import json

from robotic_cutting_sim.rubber_contact_envelope import load_envelope
from robotic_cutting_sim.rubber_material import ROOT

PINS = {"configs/rubber/rl/rubber_strip_drag_v1.json":
        "fba43424372e17fe926fad697ff8336ea7c5ed0c2d8ecdbdefb9978a994f4397",
        "configs/rubber/rubber_rl_tests_v1.json": "8e4b2593e968532e10808d4828f093df964747c18c8b98d260d1eae68535fd4c",
        "configs/rubber/rl/rubber_strip_drag_v2.json":
        "7395e6104cab16a247dc03afb133f8bd65c96d68a0d01d058bc59e6fbbf2addb",
        "configs/rubber/rubber_rl_tests_v2.json": "6db5468172e0cf7c507c51b97757550442715ef80221cf818f6fb69cdd189548",
        "configs/rubber/rubber_rl_tests_v3.json": "7dc077639ab8651e4ba6a92eac037486e4268834ec9f47f32b635cb4a93c6e2f",
        "configs/rubber/rubber_rl_tests_v4.json": "ce8dcc1a5bfab15d96e02e2f94433f273229584b6295d8929b5ba5c9d93acc0b",
        "configs/rubber/rubber_rl_tests_v5.json": "5200ed1be70e5eadae3e90a65f98f3a724c159516fb7e5da11522091c631f0eb"}


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


def test_v2_declarations_after_k2_1():
    from robotic_cutting_sim.rubber_contact_envelope import FrictionOutOfEnvelope
    import pytest
    cfg = json.loads((ROOT/"configs"/"rubber"/"rl"/"rubber_strip_drag_v2.json").read_text())
    v1 = json.loads((ROOT/"configs"/"rubber"/"rl"/"rubber_strip_drag_v1.json").read_text())
    p = json.loads((ROOT/"configs"/"rubber"/"rubber_rl_tests_v2.json").read_text())
    assert cfg["supersedes"]["rubber_strip_drag_v1"]["sha256"] == PINS["configs/rubber/rl/rubber_strip_drag_v1.json"]
    assert "K2-1" in cfg["supersedes"]["rubber_strip_drag_v1"]["reason"] and "K2-1" in p["supersedes"]["rubber_rl_tests_v1"]["reason"]
    assert cfg["profile"]["id"] == "rubber_rl_env_v2" and cfg["contact"]["law"] == "grid_node_penalty_coulomb_v2"
    v2 = json.loads((ROOT/"configs"/"rubber"/"rubber_contact_tests_v2.json").read_text())
    assert cfg["contact"]["law_parameters"] == {k: v for k, v in v1["contact"]["law_parameters"].items()
                                                if k != "slip_rate_damping_ratio_zeta_d"}
    for key in ("profile", "scene", "slots", "preload", "goal", "action", "observation", "reward", "done", "evaluator"):
        a, b = dict(cfg[key]) if isinstance(cfg[key], dict) else cfg[key], v1[key]
        if key == "profile":
            a = {k: v for k, v in a.items() if k not in ("id", "difference_to_v1")}
            b = {k: v for k, v in b.items() if k != "id"}
        assert a == b, key                                       # identical to v1 except the contact law / friction
    env = load_envelope(ROOT/cfg["contact"]["envelope"]["path"], expected_sha256=cfg["contact"]["envelope"]["sha256"])
    assert tuple(cfg["randomisation"]["ranges"]["friction_table"]) == env.friction_range(
        cfg["contact"]["law"], [4, 5, 6], general_slip=True) == (0.3, 0.3)
    with pytest.raises(FrictionOutOfEnvelope):
        env.friction_range("grid_node_penalty_coulomb_v3", [4, 5, 6], general_slip=True)
    assert p["prerequisites"]["task_config"]["sha256"] == PINS["configs/rubber/rl/rubber_strip_drag_v2.json"]
    assert "fallback_trigger" in p["tests"]["RL-08"] and "1.0 %" in p["tests"]["RL-08"]["fallback_trigger"]
    assert set(p["tests"]) == {f"RL-{i:02d}" for i in range(1, 12)}
    assert v2["id"] == "rubber_contact_tests_v2"


def test_v3_supersedes_rl05_only():
    v2 = json.loads((ROOT/"configs"/"rubber"/"rubber_rl_tests_v2.json").read_text())
    v3 = json.loads((ROOT/"configs"/"rubber"/"rubber_rl_tests_v3.json").read_text())
    assert v3["supersedes"]["rubber_rl_tests_v2"]["sha256"] == PINS["configs/rubber/rubber_rl_tests_v2.json"]
    assert v3["prerequisites"] == v2["prerequisites"]
    for k in v2["tests"]:
        if k != "RL-05":
            assert v3["tests"][k] == v2["tests"][k], k
    r5 = v3["tests"]["RL-05"]
    assert r5["pass"] == v2["tests"]["RL-05"]["pass"]               # the criterion is unchanged
    assert "7000 + e" in r5["procedure"] and "2000 + e" not in r5["procedure"]
    assert r5["controller_v3"]["mode"] == "pivot_grip" and r5["controller_v3"]["k_radial"] == 0.5
    assert "5000-5011" in r5["development"]["development_seeds"]


def test_v4_supersedes_rl05_only_and_is_final():
    v2 = json.loads((ROOT/"configs"/"rubber"/"rubber_rl_tests_v2.json").read_text())
    v4 = json.loads((ROOT/"configs"/"rubber"/"rubber_rl_tests_v4.json").read_text())
    assert v4["supersedes"]["rubber_rl_tests_v3"]["sha256"] == PINS["configs/rubber/rubber_rl_tests_v3.json"]
    assert "no v5" in v4["final_attempt"] and "success not demonstrated by a scripted controller" in v4["final_attempt"]
    for k in v2["tests"]:
        if k != "RL-05":
            assert v4["tests"][k] == v2["tests"][k], k
    r5 = v4["tests"]["RL-05"]
    assert r5["pass"] == v2["tests"]["RL-05"]["pass"] and r5["formal_seed_base"] == 8000
    from robotic_cutting_sim.rl.rubber_scripted import ScriptedPivotHoldV1
    assert set(ScriptedPivotHoldV1.NAMES) <= set(r5["controller_v4"])


def test_v5_adds_rl12_only():
    v4 = json.loads((ROOT/"configs"/"rubber"/"rubber_rl_tests_v4.json").read_text())
    v5 = json.loads((ROOT/"configs"/"rubber"/"rubber_rl_tests_v5.json").read_text())
    assert v5["supersedes"]["rubber_rl_tests_v4"]["sha256"] == PINS["configs/rubber/rubber_rl_tests_v4.json"]
    assert set(v5["tests"]) == set(v4["tests"]) | {"RL-12"}
    for k in v4["tests"]:
        assert v5["tests"][k] == v4["tests"][k], k
    for k, v in v4["common"]["seeds"].items():
        assert v5["common"]["seeds"][k] == v, k
    assert "32 MiB" in v5["tests"]["RL-12"]["pass"] and "12100 + i" in v5["tests"]["RL-12"]["procedure"]
