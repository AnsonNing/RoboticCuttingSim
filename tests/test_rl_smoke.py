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
