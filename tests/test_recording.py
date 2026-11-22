import json
import subprocess

import numpy as np
import pytest

from robotic_cutting_sim.recording import CameraFrameWriter, savez_fast, write_json


def test_numeric_audit_receipt_preserves_boolean_types_and_rejects_nan(tmp_path):
    output = tmp_path/"receipt.json"
    write_json(output, {"pass": np.bool_(True), "n": np.int64(2),
                        "error": np.float32(.5), "values": np.array([1, 2])})
    assert json.loads(output.read_text()) == {"pass": True, "n": 2, "error": .5, "values": [1, 2]}
    with pytest.raises(ValueError, match="Out of range"):
        write_json(output, {"error": np.float64(np.nan)})


def test_camera_writer_preserves_order_and_lossless_frames(tmp_path):
    frames = tmp_path / "observations"
    frames.mkdir()
    writer = CameraFrameWriter(tmp_path, savez_fast, max_bytes=16, max_frames=1, chunk_frames=2)
    for frame in range(5):
        rgb = np.full((2, 2, 3), frame, dtype=np.uint8)
        rgb.setflags(write=False)
        writer.submit({"frame": frame, "time_s": frame / 50},
                      [(frames / f"wrist_{frame:04d}.npz", {"rgb": rgb})])
    timing = writer.finish()
    rows = [json.loads(row) for row in (tmp_path / "frame_manifest.jsonl").read_text().splitlines()]
    assert [row["frame"] for row in rows] == list(range(5))
    assert timing["peak_pending_bytes"] == 12 <= timing["max_pending_bytes"]
    assert json.loads((tmp_path / "frame_manifest_checkpoint.json").read_text()) == {
        "committed_frames": 5, "complete": True,
    }
    for frame in range(5):
        with np.load(frames / f"wrist_{frame:04d}.npz", allow_pickle=False) as saved:
            assert np.all(saved["rgb"] == frame)


def test_camera_writer_failure_retains_only_completed_prefix(tmp_path):
    frames = tmp_path / "observations"
    frames.mkdir()

    def fail_second(path, **payload):
        if path.stem.endswith("0001"):
            raise OSError("write failed")
        np.savez_compressed(path, **payload)

    writer = CameraFrameWriter(tmp_path, fail_second, chunk_frames=1)
    for frame in range(2):
        writer.submit({"frame": frame}, [(frames / f"wrist_{frame:04d}.npz", {"rgb": np.zeros((2, 2, 3))})])
    with pytest.raises(RuntimeError, match="Camera writer failed"):
        writer.finish()
    assert json.loads((tmp_path / "frame_manifest_checkpoint.json").read_text()) == {
        "committed_frames": 1, "complete": False,
    }
    assert not (tmp_path / "frame_manifest.json").exists()


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
