"""WP-12 Phase 2b: the declared live workcell protocol and the partial-cut goal (CPU, no Isaac)."""
import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from robotic_cutting_sim import goal_planner
from robotic_cutting_sim.scene_config import compile_task

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT/"configs/wp12/workcell_live_v1.json").read_text())
OVERRIDES = PROTOCOL["controller"]["planner_overrides"]


def _ops(scene, goal, overrides=OVERRIDES):
    goal = json.loads((ROOT/goal).read_text())
    goal["episode"]["duration_s"] = 120.0   # the live duration rule's probe compile
    task = compile_task(str(ROOT/scene), goal, planner_options=overrides, seed=42)
    _, report = goal_planner.plan_with_report(task.scene, task.goal, task.tool_profile, task.controller["planner"])
    return task, report["operations"]


def test_protocol_episodes_reference_committed_configs():
    ids = [e["id"] for e in PROTOCOL["episodes"]]
    assert ids == ["CUT", "PARTIAL", "TRIM", "TESTB", "TESTC", "WP11_CUT_REPRO", "TRIM_PATH_V3"]
    for e in PROTOCOL["episodes"]:
        if e["scene"].startswith("artifacts/"):
            continue   # the WP-11 reproduction reads a git-ignored live record
        assert (ROOT/e["scene"]).is_file() and (ROOT/e["goal"]).is_file()
    assert OVERRIDES == {"max_rotation_deg": 1.0, "max_translation_m": 0.001, "controller_profile": "path_v2"}


def test_full_cut_ends_on_the_board_top_and_partial_cut_stops_at_mid_depth():
    _, cut = _ops("configs/wp12/scenes/bar_flat_board.json", "configs/wp11/goals/sever_mid.json")
    assert cut[0]["end_local_m"][2] == pytest.approx(0.0, abs=1e-12) and cut[0]["end_on_support"]
    task, partial = _ops("configs/wp12/scenes/bar_flat_board.json", "configs/wp12/goals/partial_cut_mid_depth.json")
    assert task.controller["planner"]["overtravel_m"] == pytest.approx(-0.010)
    assert partial[0]["end_local_m"][2] == pytest.approx(0.010, abs=1e-9) and not partial[0]["end_on_support"]
    assert partial[0]["start_local_m"] == pytest.approx(cut[0]["start_local_m"])


def test_partial_goal_differs_from_sever_mid_only_by_id_description_and_planner_depth():
    a = json.loads((ROOT/"configs/wp11/goals/sever_mid.json").read_text())
    b = json.loads((ROOT/"configs/wp12/goals/partial_cut_mid_depth.json").read_text())
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert differing == ["description", "goal_id", "planner"]
    assert b["planner"] == {"overtravel_m": -0.010}


@pytest.mark.parametrize('caller_tmpdir', [False, True])
def test_first_launch_creates_logs_and_uses_portable_tmpdir(tmp_path, monkeypatch, caller_tmpdir):
    """Exercise a real CPU child in a fresh output tree; no Isaac/GPU is launched."""
    spec = importlib.util.spec_from_file_location('first_workcell_launch', ROOT/'scripts/run_wp12_workcell_live.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    protocol = {'runtime': {'tmpdir': str(tmp_path/'absent_original_host_tmp')}}
    monkeypatch.setattr(runner, 'load_protocol', lambda: protocol)
    monkeypatch.setattr(runner, 'gpu_snapshot', lambda: {})
    monkeypatch.setattr(runner, 'git_state', lambda: {})

    def prepare(protocol, episode, out):
        out.mkdir()
        return {}

    def command(protocol, prep, inputs, run):
        script = ("import json,os,pathlib; p=pathlib.Path("+repr(str(run))+"); p.mkdir(); "
                  "(p/'result.json').write_text(json.dumps({'status':'TEST_CHILD','tmpdir':os.environ['TMPDIR']}))")
        return [sys.executable, '-c', script]

    monkeypatch.setattr(runner, 'prepare', prepare)
    monkeypatch.setattr(runner, 'runner_command', command)
    if caller_tmpdir:
        monkeypatch.setenv('TMPDIR', str(tmp_path))
    else:
        monkeypatch.delenv('TMPDIR', raising=False)
    assert runner.cmd_launch(SimpleNamespace(out='artifacts/native_first_run', episode='CUT', disk_floor_gb=0)) == 0
    assert (tmp_path/'artifacts/wp12/logs/native_first_run_CUT.log').is_file()
    attempt = json.loads((tmp_path/'artifacts/native_first_run/attempt.json').read_text())
    used = Path(attempt['env']['TMPDIR'])
    assert used.is_dir() and used != Path(protocol['runtime']['tmpdir'])
    if caller_tmpdir:
        assert used == tmp_path
    assert attempt['result_status'] == 'TEST_CHILD'
