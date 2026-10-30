"""WP-11 delta review E2: the collector also walks the policy-facing reset/step info for privileged names.

Kinematic CPU backend at coarse spacing, collector run in-process; a leak is injected into the step info of one
episode and the reset info of another. Each leak is recorded in episode.json and fails that episode's
``no_privileged_keys_in_info`` reliability check; the clean episode passes it.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _collector():
    spec = importlib.util.spec_from_file_location("collect_dataset_e2", ROOT/"scripts/collect_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_sim
def test_privileged_info_keys_are_recorded_as_per_episode_leaks(tmp_path, monkeypatch):
    pytest.importorskip("warp")
    from robotic_cutting_sim import dataset as ds
    from robotic_cutting_sim.policy_env import PolicyReadyEnv
    collector = _collector()
    hold = {"controller": "policy", "policy": "robotic_cutting_sim.policy_examples:hold_policy", "max_steps": 4}
    plan = {"schema": "rcs_episode_plan_v1", "episodes": [dict(hold, label="step_leak", seed=1),
                                                         dict(hold, label="reset_leak", seed=2),
                                                         dict(hold, label="clean", seed=3)]}
    (tmp_path/"plan.json").write_text(json.dumps(plan))
    resets = []
    reset, step = PolicyReadyEnv.reset, PolicyReadyEnv.step

    def leaky_reset(self, *args, **kwargs):
        observation, info = reset(self, *args, **kwargs)
        resets.append(len(resets))
        if resets[-1] == 1:
            info = dict(info, reset_checks=dict(info["reset_checks"], labels_clean=True))
        return observation, info

    def leaky_step(self, *args, **kwargs):
        observation, reward, terminated, truncated, info = step(self, *args, **kwargs)
        if resets[-1] == 0:
            info = dict(info, grasp_status="HOLDING", privileged={"edge_pose_world_xyzw": [0.]*7})
        return observation, reward, terminated, truncated, info
    monkeypatch.setattr(PolicyReadyEnv, "reset", leaky_reset)
    monkeypatch.setattr(PolicyReadyEnv, "step", leaky_step)
    out = tmp_path/"dataset"
    status = collector.main(["--runtime", "kinematic", "--scene", "bar_160x30x20", "--goal", "sever_mid",
                             "--spacing", "0.004", "--material-dt", "0.0005", "--device", "cpu",
                             "--episode-plan", str(tmp_path/"plan.json"), "--record-every", "10", "--out", str(out)])
    assert status == 3                                   # scored, not POLICY_READY (the injected leaks fail it)
    assert resets == [0, 1, 2]
    metas = [ds.load_episode(out, i)["meta"] for i in range(3)]
    steps = [m["task_steps"] for m in metas]
    assert steps[0] >= 1
    # one count per leaking info dict; info["privileged"] itself is never walked
    assert metas[0]["info_privileged_leaks"] == steps[0]
    assert metas[0]["info_privileged_leak_paths"] == ["step:/grasp_status"]
    assert metas[1]["info_privileged_leaks"] == 1
    assert metas[1]["info_privileged_leak_paths"] == ["reset:/reset_checks/labels_clean"]
    assert metas[2]["info_privileged_leaks"] == 0 and metas[2]["info_privileged_leak_paths"] == []
    assert all(m["observation_privileged_leaks"] == 0 for m in metas)
    report = json.loads((out/"reliability.json").read_text())
    checks = [row["checks"]["no_privileged_keys_in_info"] for row in report["episodes"]]
    assert checks == [False, False, True]
    assert {"episode_000:no_privileged_keys_in_info", "episode_001:no_privileged_keys_in_info"} <= set(report["failing"])
    assert "episode_002:no_privileged_keys_in_info" not in report["failing"]
