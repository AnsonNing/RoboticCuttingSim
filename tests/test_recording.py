import json
import subprocess

import numpy as np
import pytest

from robotic_cutting_sim.recording import write_json


def test_numeric_audit_receipt_preserves_boolean_types_and_rejects_nan(tmp_path):
    output = tmp_path/"receipt.json"
    write_json(output, {"pass": np.bool_(True), "n": np.int64(2),
                        "error": np.float32(.5), "values": np.array([1, 2])})
    assert json.loads(output.read_text()) == {"pass": True, "n": 2, "error": .5, "values": [1, 2]}
    with pytest.raises(ValueError, match="Out of range"):
        write_json(output, {"error": np.float64(np.nan)})


@pytest.mark.parametrize("failure", ["missing_command", "timeout"])
def test_manifest_preserves_unknown_provenance_after_command_failure(tmp_path, monkeypatch, failure):
    from robotic_cutting_sim import recording

    checkout = tmp_path/".deps"/"newton"
    checkout.mkdir(parents=True)
    (tmp_path/"src").mkdir()
    monkeypatch.setattr(recording, "ROOT", tmp_path)
    monkeypatch.setattr(recording.importlib.metadata, "version", lambda name: "fixture")

    def run(args, **kwargs):
        if args == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(args, 128, stdout="incomplete status", stderr="git failed")
        if kwargs["cwd"] == checkout and args == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="1"*40+"\n", stderr="")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 30, output=b"partial output", stderr=b"still running")
        raise FileNotFoundError("git executable unavailable")

    monkeypatch.setattr(recording.subprocess, "run", run)
    result = recording.manifest({}, "fixture")
    source = result["source_checkouts"]["newton"]
    assert source["sha"] == "1"*40
    assert source["dirty"] is None and source["tracked_patch_sha256"] is None
    assert result["git_sha"] is None and result["git_dirty"] is None
    error = source["command_errors"]["diff"]
    assert error["returncode"] is None and error["error"]
    assert error["stdout"] == ("partial output" if failure == "timeout" else "")
    assert error["stderr"] == ("still running" if failure == "timeout" else "")
    assert result["git_command_errors"]["status"]["returncode"] == 128
