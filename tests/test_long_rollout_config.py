"""P1-4 T4.3: the pre-declared long-rollout configuration (``configs/rl_benchmark/long_rollout_v1.json``).

The file is pinned by SHA-256: it was declared before any run and must not change afterwards (a later change
needs a new versioned file, never an edit)."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robotic_cutting_sim.rl import supervision as S  # noqa: E402
from robotic_cutting_sim.rl.long_rollout import RolloutSchedule  # noqa: E402

PATH = ROOT / "configs" / "rl_benchmark" / "long_rollout_v1.json"
CONFIG = json.loads(PATH.read_text())
PINNED_SHA256 = "4813f96e8d183e483173fb7f27dcf77cc5a29783c9eb0d79304e8a305d8de3ef"


def test_config_is_pinned():
    assert hashlib.sha256(PATH.read_bytes()).hexdigest() == PINNED_SHA256


def test_schema_and_declaration():
    assert CONFIG["schema"] == "rcs_rl_long_rollout_v1" and CONFIG["status"] == "PRE_DECLARED"
    assert CONFIG["declared_before_any_run"] is True
    assert CONFIG["process_rules"]["live_device_hashgrids_required"] == 1
    assert CONFIG["common"]["strict"] is False and CONFIG["common"]["workload"] == "A"
    assert CONFIG["common"]["free_vram_margin_mib"] == 4096


def test_lr_a_matches_the_plan():
    run = CONFIG["runs"]["LR-A"]
    assert run["num_envs"] == 8 and run["episodes_per_env"] >= 8 and run["episode_steps"] == 1200
    assert run["particle_contact_mode"] == "frozen_v1" and run["own_vram_bound_mib"] <= 1536
    assert [e["env"] for e in run["envs"]] == list(range(8))
    drives = [e["drive"] for e in run["envs"]]
    assert drives == ["replay_canary"] * 2 + ["recorded_command_policy_v1"] * 4 + ["random_policy_v1"] * 2
    assert {e["scene_seed"] for e in run["envs"]} == {512, 15}
    schedule = RolloutSchedule(8, episode_steps=1200, episodes_per_env=run["episodes_per_env"],
                               offset_steps=run["offset_steps_per_env"], base_seed=run["base_seed"])
    assert schedule.expected_batch_steps() == run["expected_batch_steps_if_all_full_length"] == 10650
    assert run["max_wall_s"] >= 3600 * run["estimate_wall_h"][1] * 1.5


def test_canary_rule_covers_every_canary_seed():
    native = CONFIG["canary_rule"]["native_labelled"]
    assert native == {"512": 37958, "15": 85786}
    for name in ("LR-A",):
        for e in CONFIG["runs"][name]["envs"]:
            if e["drive"] == "replay_canary":
                assert str(e["scene_seed"]) in native


def test_bounds_are_the_plan_numbers():
    b = CONFIG["bounds"]
    assert b["slope_gate_statistic"] == "theil_sen"
    assert b["B1"]["max_minus_min_mib"] == 1.0
    assert b["B2"]["slope_mib_per_1000_batch_steps"] == 2.0 and b["B2"]["max_single_increase_mib"] == 64.0
    assert b["B3"]["slope_mib_per_1000_env_steps"] == 0.5 and b["B3"]["quintile_drift_mib"] == 64.0
    assert b["B3"]["quintile_drift_fraction"] == 0.03
    assert b["B4"]["graph_captures_per_process"] == 1 and b["B4"]["live_device_hashgrids"] == 1
    assert b["B5"]["gate"] is False


def test_classification_and_recovery_use_known_classes():
    cls = CONFIG["classification"]
    assert cls.get("split", "p1_4_v1") == "p1_4_v1"
    # v1 was declared before INFRA_CONTACT_POOL_OVERFLOW / ENV_INFRASTRUCTURE existed (added for termination_reason_v1)
    assert set(cls["process_classes"]) == {"COMPLETE", *S.INFRA_CLASSES} - {"INFRA_CONTACT_POOL_OVERFLOW"}
    assert set(cls["episode_classes"]) == set(S.EPISODE_CLASSES) - {S.ENV_INFRASTRUCTURE}
    assert set(cls["numerical_invalid_reasons"]) <= set(S.INVALID_REASONS.values())
    rec = CONFIG["recovery"]
    assert rec["max_restarts"] == 1
    assert set(rec["restart_on"]) | set(rec["no_restart_on"]) == set(S.INFRA_CLASSES) - {"INFRA_CONTACT_POOL_OVERFLOW"}
    assert not set(rec["restart_on"]) & set(rec["no_restart_on"])
    policy = S.SupervisorPolicy.from_config(CONFIG, CONFIG["runs"]["LR-A"])
    assert policy.hang_min_s == 300 and policy.hang_factor_median_step == 20 and policy.max_restarts == 1
    assert policy.max_wall_s == CONFIG["runs"]["LR-A"]["max_wall_s"] and policy.sample_interval_s == 5


def test_native_limitations_are_retained():
    text = " ".join(CONFIG["native_limitations_retained"])
    assert "reload_in_place" in text and "camera-off" in text and "NOT POLICY_READY" in text


# ----------------------------------------------------------------------------- long_rollout_v2 (T4.5 / T4.6)

PATH_V2 = ROOT / "configs" / "rl_benchmark" / "long_rollout_v2.json"
CONFIG_V2 = json.loads(PATH_V2.read_text())
PINNED_SHA256_V2 = "524c84d6ca11c886179cf3883f9d9df312fd8445f325ea4b13cae9585be2658d"


def test_v2_is_pinned_and_uses_the_env_split():
    assert hashlib.sha256(PATH_V2.read_bytes()).hexdigest() == PINNED_SHA256_V2
    assert CONFIG_V2["id"] == "long_rollout_v2" and CONFIG_V2["status"] == "PRE_DECLARED"
    cls = CONFIG_V2["classification"]
    assert cls["split"] == "termination_reason_v1" and cls["split"] in S.SPLITS
    assert set(cls["process_classes"]) == {"COMPLETE", *S.INFRA_CLASSES}
    assert set(cls["task_process_classes"]) == {S.TASK_STRICT_INVALID_ENV}
    assert set(cls["episode_classes"]) == set(S.EPISODE_CLASSES)
    rec = CONFIG_V2["recovery"]
    assert set(rec["restart_on"]) | set(rec["no_restart_on"]) == set(S.INFRA_CLASSES)
    assert "LR-A" not in CONFIG_V2["runs"]                    # LR-A stays declared by v1 only


def test_v2_runs():
    runs = CONFIG_V2["runs"]
    a2 = runs["LR-A2"]
    assert a2["particle_contact_mode"] == "alias_free_v1" and a2["num_envs"] == 8 and a2["episodes_per_env"] == 4
    assert RolloutSchedule(8, episode_steps=1200, episodes_per_env=4, offset_steps=150,
                           base_seed=a2["base_seed"]).expected_batch_steps() == a2["expected_batch_steps_if_all_full_length"]
    b = runs["LR-B"]
    assert b["mode"] == "reset_stress" and b["num_envs"] == 16 and len(b["envs"]) == 16
    assert [e["drive"] for e in b["envs"]][:2] == ["replay_canary"] * 2
    assert b["expected_forced_resets"] == round(14 * b["batch_steps"] * b["forced_reset_probability"])
    c = runs["LR-C"]
    assert c["mode"] == "rebuild_cycles" and c["cycles"] == 10 and c["steps_per_cycle"] == 50
    assert c["bounds"]["C2"]["max_minus_min_mib"] == 1.0 and c["bounds"]["C3"]["slope_mib_per_cycle"] == 8.0
    d = runs["LR-D"]
    assert set(d["probes"]) == {"nan_env:1@20", "bad_pose:0@20", "sigsegv@20", "hang@20", "device_oom@20"}
    assert d["recovery"]["max_restarts"] == 0 and d["timeouts"]["hang_min_s"] == 60
    policy = S.SupervisorPolicy.from_config(CONFIG_V2, d)
    assert policy.hang_min_s == 60 and policy.max_restarts == 0 and policy.split == "termination_reason_v1"
    for name in ("LR-A2", "LR-B", "LR-C", "LR-D"):
        assert runs[name]["own_vram_bound_mib"] <= 2253
