"""P1-2b: both vector-env backends through the same contract (``rl.vector_contract.check_vector_env_contract``).

Each backend runs in its own process through ``scripts/rl_ready/run_p1_2b_smoke.py --device cpu`` (Warp CPU, coarse
2.5 mm TRIM seeds 512 and 15 via ``tasks=[512, 15]``, 3-step episodes, partial reset of env 1 after step 1,
evaluation stream every 2 steps), which calls the one checker and writes ``contract.json``:

* workload A (``kinematic_ideal_tracking_batched``) needs Torch (``.deps/isaacsim-venv`` or ``.deps/isaaclab-venv``);
* workload B (``isaac_lab_newton``, B1 rigid mount, Warp-CPU MJWarp ``cpu_test_mode``) needs Isaac Lab
  (``.deps/isaaclab-venv`` with the D-09 overlay). Skipped elsewhere = NOT_RUN, never a pass.
The CUDA form of the same worker is the P1-2b GPU smoke (report ``P1_1_P1_2_MILESTONES.md`` section P1-2b)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch", reason="contract worker needs Torch (NOT_RUN in .venv)")

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/rl_ready/run_p1_2b_smoke.py"
BACKENDS = ("kinematic_ideal_tracking_batched", "isaac_lab_newton")
_REPORTS = {}


def _run(backend, tmp_path_factory):
    if backend == "isaac_lab_newton":
        pytest.importorskip("isaaclab", reason="workload B needs the Isaac Lab venv (NOT_RUN here)")
    if backend in _REPORTS:
        return _REPORTS[backend]
    out = tmp_path_factory.mktemp(backend) / "run"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run([sys.executable, str(WORKER), "--backend", backend, "--device", "cpu", "--tasks", "512",
                           "15", "--steps", "3", "--reset-step", "1", "--reset-envs", "1", "--spacing", "0.0025",
                           "--max-episode-steps", "3", "--evaluation-stream", "--stream-every", "2", "--out", str(out)],
                          env=env, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    row = json.loads((out / "row.json").read_text()) if (out / "row.json").exists() else {}
    if proc.returncode != 0:
        raise AssertionError(f"{backend} worker failed ({proc.returncode}): {row.get('error')}\n{proc.stderr[-3000:]}")
    _REPORTS[backend] = row
    return row


@pytest.mark.parametrize("backend", BACKENDS)
def test_contract_on_cpu(backend, tmp_path_factory):
    row = _run(backend, tmp_path_factory)
    contract = row["contract"]
    assert row["status"] == "PASS" and contract["status"] == "PASS" and contract["backend"] == backend
    assert contract["clauses"] == ["receipt/spaces/metadata", "full reset", "partial reset", "steps", "evaluate"]
    assert contract["final_step_count"] == [3, 1]
    assert contract["termination_reason"] == [["TRUNCATED_HORIZON"], []]
    assert [e["scene_seed"] for e in contract["evaluation"]] == [512, 15]
    assert [e["frames"] for e in contract["evaluation"]] == [3, 2]     # samples 0, 2 + done 3 / reset 0 + step 1
    assert all(e["mode"] == "evaluation_stream_v1" for e in contract["evaluation"])
    assert row["hashgrids"]["cuda_count"] == 0 and row["fidelity"]["reduced_fidelity"] is True


def test_both_backends_report_the_same_contract(tmp_path_factory):
    a = _run(BACKENDS[0], tmp_path_factory)["contract"]
    b = _run(BACKENDS[1], tmp_path_factory)["contract"]
    for key in ("contract", "num_envs", "steps", "clauses", "final_step_count", "termination_reason",
                "evaluation_mode", "observation_profile", "api_revision", "runtime"):
        assert a[key] == b[key], key
    assert [e["frames"] for e in a["evaluation"]] == [e["frames"] for e in b["evaluation"]]
