"""P1-5 T5.12: RL pipeline smoke S-CPU (``configs/rl_benchmark/rl_smoke_v1.json``) and the rollout consumer check.

Torch test: run in ``.deps/isaacsim-venv`` with the D-09 overlay; in ``.venv`` (no torch) it is skipped = NOT_RUN.
S-CPU is a COARSE, SHORTENED seeded TRIM batch (2.5 mm particles, 10-step episodes; ``reduced_fidelity``) on the
Warp/Torch CPU device: N=3 (one reference_command_policy_v1 env, two RandomPolicyV1 envs with SplitMix64 forced
resets), autoreset same_step, terminal capture, 40 batch steps. No training."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "rl_ready" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rollout(tmp_path_factory):
    smoke = _load("run_rl_smoke")
    declaration = json.loads((ROOT / "configs/rl_benchmark/rl_smoke_v1.json").read_text())
    spec = next(r for r in declaration["runs"] if r["id"] == "S-CPU")
    out = tmp_path_factory.mktemp("smoke") / "s_cpu"
    manifest = smoke.run(spec, out, device="cpu", cpu=True)
    return out, manifest, spec


def test_s_cpu_rollout_consumer_check_passes(rollout):
    out, manifest, spec = rollout
    report = _load("check_rl_rollout").check(out)
    failed = {k: v for k, v in report["items"].items() if v["status"] == "FAIL"}
    assert report["status"] == "PASS" and not failed, failed
    assert report["steps"] == spec["batch_steps"] == manifest["steps"]
    assert manifest["schema"] == "rcs_rl_rollout_v1" and manifest["obs_critic"] is None


def test_s_cpu_episode_ends_and_formats(rollout):
    out, manifest, spec = rollout
    from robotic_cutting_sim.rl.rollout import RESET_AUTO, RESET_FORCED, load_rollout
    _, a, episodes = load_rollout(out)
    counts = manifest["end_counts"]
    assert counts["TRUNCATED_TIME_LIMIT"] >= 3 and counts["TRUNCATED_BY_FORCED_RESET"] >= 1
    assert (a["reset_kind"] == RESET_AUTO).sum() >= 3 and (a["reset_kind"] == RESET_FORCED).sum() >= 1
    assert all(len(c["sha256"]) == 64 for c in manifest["chunks"]) and len(manifest["chunks"]) == 3   # 16+16+8
    evaluated = [e for e in episodes if e["evaluator"] is not None]
    assert evaluated and all(e["evaluator"]["scored"] == "last_completed" for e in evaluated)
    assert all(e["end"] in ("TRUNCATED_TIME_LIMIT", "TERMINATED_INVALID", "TRUNCATED_BY_FORCED_RESET") for e in episodes)
    assert a["obs_policy"].shape == (40, 3, 18) and a["actions"].dtype == np.float32
    assert manifest["env_receipt"]["reward"]["version"] == "trim_bond_progress_packed_v1"


def test_check_detects_a_broken_chain(rollout, tmp_path):
    import shutil
    out, _, _ = rollout
    broken = tmp_path / "broken"
    shutil.copytree(out, broken)
    with np.load(broken / "chunk_00001.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays["obs_policy"][3, 0, 0] += 1.
    np.savez_compressed(broken / "chunk_00001.npz", **arrays)
    report = _load("check_rl_rollout").check(broken)
    assert report["status"] == "FAIL" and report["items"]["chunks_sha256"]["status"] == "FAIL"


# ----------------------------------------------------------------------------- rcs_rl_rollout_v2 (review finding 12)

def _rewrite_as_v2(out, dest, critic=False):
    """The S-CPU transitions written again through the v2 writer (same arrays; v2 manifest roles)."""
    from robotic_cutting_sim.rl.rollout import SCHEMA_V2, RolloutWriter, load_rollout
    m, a, episodes = load_rollout(out)
    T, N, D = a["obs_policy"].shape
    base = {k: m[k] for k in ("env_receipt", "single_observation_space", "single_action_space")}
    writer = RolloutWriter(dest, N, obs_dim=D, chunk_size=16, critic_dim=D if critic else None, manifest=base,
                           schema=SCHEMA_V2)
    t = {k: torch.as_tensor(v) for k, v in a.items()}
    for i in range(T):
        obs, nxt = {"policy": t["obs_policy"][i]}, {"policy": t["next_obs_policy"][i]}
        if critic:
            obs["critic"], nxt["critic"] = t["obs_policy"][i], t["next_obs_policy"][i]
        writer.add(obs=obs, actions=t["actions"][i], reward=t["rewards"][i], terminated=t["terminated"][i],
                   truncated=t["truncated"][i], next_obs=nxt, reset_kind=t["reset_kind"][i],
                   info={k: t[k][i] for k in ("episode", "step_count", "seed", "drive", "action_status",
                                              "invalid_reason", "termination_reason")})
    for row in episodes:
        writer.episode_record(**row)
    return writer.close()


def test_rollout_v2_labels_privileged_fields_and_passes_the_consumer_check(rollout, tmp_path):
    out, _, _ = rollout
    check = _load("check_rl_rollout").check
    v1 = check(out)
    assert v1["schema"] == "rcs_rl_rollout_v1" and v1["field_roles"] == "UNLABELLED_V1"
    assert "non_deployable_fields_labelled" not in v1["items"]              # v1 result unchanged
    for critic in (False, True):
        dest = tmp_path / f"v2_critic{int(critic)}"
        manifest = _rewrite_as_v2(out, dest, critic=critic)
        assert manifest["schema"] == "rcs_rl_rollout_v2" and manifest["version"] == 2
        assert manifest["deployable_policy_inputs"] == ["next_obs_policy", "obs_policy"]
        privileged = {"invalid_reason", "termination_reason"} | ({"next_obs_critic", "obs_critic"} if critic else set())
        assert set(manifest["privileged_non_deployable"]) == privileged
        assert {"rewards", "terminated"} <= set(manifest["oracle_derived"])
        report = check(dest)
        failed = {k: v for k, v in report["items"].items() if v["status"] == "FAIL"}
        assert report["status"] == "PASS" and not failed, failed
        assert report["items"]["non_deployable_fields_labelled"]["status"] == "PASS"
        assert report["field_roles"] == "rollout_field_roles_v1"


def test_rollout_v2_check_fails_an_unlabelled_privileged_field(rollout, tmp_path):
    out, _, _ = rollout
    dest = tmp_path / "v2"
    _rewrite_as_v2(out, dest)
    m = json.loads((dest / "manifest.json").read_text())
    m["privileged_non_deployable"].remove("termination_reason")
    m["field_roles"]["termination_reason"] = "bookkeeping"
    m["deployable_policy_inputs"].append("invalid_reason")
    (dest / "manifest.json").write_text(json.dumps(m))
    report = _load("check_rl_rollout").check(dest)
    item = report["items"]["non_deployable_fields_labelled"]
    assert report["status"] == "FAIL" and item["status"] == "FAIL"
    assert any("termination_reason" in x for x in item["detail"]) and any("deployable" in x for x in item["detail"])
