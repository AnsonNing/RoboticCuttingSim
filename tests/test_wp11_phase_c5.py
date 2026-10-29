"""WP-11 Phase C5: live-runner options for the configured near-horizontal TRIM episode."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_c5", ROOT/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hold_until_time_limit_is_configured_only():
    module = _script("run_wp10_live")
    with pytest.raises(SystemExit):
        module.main(["--runtime", "isaac_sim", "--out", "unused", "--task", "CUT-01-R", "--hold-until-time-limit"])


def test_hold_until_time_limit_parses_for_a_configured_run(monkeypatch):
    module = _script("run_wp10_live")
    captured = {}

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(module, "new_output", lambda path: Path(path))
    monkeypatch.setattr(module, "write_json", lambda *a, **k: None)
    monkeypatch.setattr(module, "manifest", lambda config, kind: captured.setdefault("config", config))
    monkeypatch.setattr(module, "run_configured", stop)
    with pytest.raises(Stop):
        module.main(["--runtime", "isaac_sim", "--out", "unused", "--scene", "trim01_wp11_workcell",
                     "--goal", "trim_bump_rho10", "--controller", "scripted", "--edge-feedback", "measured",
                     "--stop-after-phase", "WITHDRAW", "--hold-until-time-limit", "--return-home", "--steps", "700",
                     "--planner-options", '{"max_rotation_deg": 1.0, "max_translation_m": 0.001}'])
    assert captured["config"]["hold_until_time_limit"] is True


def test_preflight_runner_refuses_an_existing_output(tmp_path):
    module = _script("run_phase_c5_trim_preflight")
    (tmp_path/"exists").mkdir()
    import sys
    argv, sys.argv = sys.argv, ["x", "--out", str(tmp_path/"exists")]
    try:
        with pytest.raises(Exception):
            module.main()
    finally:
        sys.argv = argv
