import gzip
import hashlib
import json
import builtins

import pytest

from robotic_cutting_sim import native_sensor_runtime as runtime
from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig


def _index(tmp_path, monkeypatch):
    source = tmp_path/"cache/example"
    source.mkdir(parents=True)
    raw = b'{"official-package": "unchanged metadata"}\n'
    (source/"summaries.json").write_bytes(raw)
    registry = json.dumps({"files": {"summaries.json": hashlib.sha1(raw).hexdigest()[:16]}}).encode()
    (source/"registry.json").write_bytes(registry)
    monkeypatch.setattr(runtime, "INDICES", (
        ("example", "kit/prod/default", "kit-default", hashlib.sha256(registry).hexdigest()),))
    return source, raw


def test_offline_metadata_preserves_exact_bytes_and_never_clobbers(tmp_path, monkeypatch):
    _, raw = _index(tmp_path, monkeypatch)
    output = tmp_path/"episode"
    registries, receipt = runtime.prepare_offline_registry(output, cache_root=tmp_path/"cache")
    mirror = output/"registry/kit-default/v2/summaries.gz"
    assert gzip.decompress(mirror.read_bytes()) == raw
    assert registries[0]["url"] == (output/"registry/kit-default").as_uri()
    assert receipt["native_sensor_updates"] == "NOT_RUN"
    assert len(receipt["files"]) == 2
    mirror.write_bytes(b"preserved failure artifact")
    with pytest.raises(ValueError, match="already differs"):
        runtime.prepare_offline_registry(output, cache_root=tmp_path/"cache")
    assert mirror.read_bytes() == b"preserved failure artifact"


def test_corrupt_official_metadata_is_rejected_before_activation(tmp_path, monkeypatch):
    source, _ = _index(tmp_path, monkeypatch)
    (source/"summaries.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        runtime.prepare_offline_registry(tmp_path/"episode", cache_root=tmp_path/"cache")


def test_sensor_repair_is_explicit_and_versioned():
    assert IsaacSimCuttingConfig().native_sensor_runtime is None
    assert IsaacSimCuttingConfig(native_sensor_runtime=runtime.PROFILE).native_sensor_runtime == runtime.PROFILE
    with pytest.raises(ValueError, match="native_sensor_runtime"):
        IsaacSimCuttingConfig(native_sensor_runtime="unversioned")


def test_kit_import_failure_is_recorded_before_context_close(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "carb.eventdispatcher":
            raise ModuleNotFoundError("deliberate unavailable Kit binding")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ModuleNotFoundError, match="deliberate unavailable"):
        runtime.activate_offline_contact_extension(object(), tmp_path)
    receipt = json.loads((tmp_path/"native_sensor_runtime_v1/activation.json").read_text())
    assert receipt["status"] == "FAIL_EXTENSION_ACTIVATION"
    assert receipt["activation_stage"] == "IMPORT_KIT_BINDINGS"
    assert "deliberate unavailable Kit binding" in receipt["error"]
    assert "ModuleNotFoundError" in receipt["traceback"]
