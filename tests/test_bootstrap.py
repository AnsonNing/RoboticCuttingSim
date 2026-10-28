"""Installation safety and relocation checks, without network/GPU dependencies."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from bootstrap_runtime import expected_pins, installer_environment, main, probe_environment, select_python
from fetch_runtime_assets import fetch_assets


def test_runtime_locks_keep_usd_providers_separate():
    base, sim, lab = [expected_pins(name) for name in ("base", "isaac_sim", "isaac_lab")]
    assert base["usd-core"] == "26.8" and "usd-exchange" not in base
    assert "usd-core" not in sim and "usd-exchange" not in sim
    assert lab["usd-exchange"] == "3.0.1" and "usd-core" not in lab
    for pins in (sim, lab):
        assert pins["torch"] == "2.11.0+cu128"
        assert pins["warp-lang"] == "1.17.0" and pins["newton"] == "1.6.1"
    assert "torchaudio" in sim and "torchaudio" not in lab


def test_explicit_python_command_resolves_path_before_creating_environment(tmp_path, monkeypatch):
    alias = tmp_path/"python3.12"
    alias.symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(tmp_path))
    selected = select_python("python3.12", None, {}, False, tmp_path/"unused.log")
    assert selected == str(alias)
    assert Path(selected).is_file()


def test_large_download_defaults_respect_explicit_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("UV_HTTP_TIMEOUT", raising=False)
    monkeypatch.delenv("UV_CONCURRENT_DOWNLOADS", raising=False)
    initial = installer_environment(tmp_path, False)
    assert initial["UV_HTTP_TIMEOUT"] == "900" and initial["UV_CONCURRENT_DOWNLOADS"] == "4"
    monkeypatch.setenv("UV_HTTP_TIMEOUT", "1800")
    monkeypatch.setenv("UV_CONCURRENT_DOWNLOADS", "2")
    actual = installer_environment(tmp_path, True)
    assert actual["UV_HTTP_TIMEOUT"] == "1800" and actual["UV_CONCURRENT_DOWNLOADS"] == "2"
    assert actual["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_dry_run_relocates_with_checkout_and_writes_nothing(tmp_path):
    checkout = tmp_path/"fresh clone with spaces"
    (checkout/"scripts").mkdir(parents=True)
    for name in ("bootstrap_runtime.py", "fetch_runtime_assets.py"):
        (checkout/"scripts"/name).write_bytes((ROOT/"scripts"/name).read_bytes())
    result = subprocess.run([sys.executable, str(checkout/"scripts/bootstrap_runtime.py"),
                             "--runtime", "all", "--dry-run"], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["root"] == str(checkout)
    assert plan["environments"]["isaac_sim"] == str(checkout/".deps/isaacsim-venv")
    assert not (checkout/".deps").exists() and not (checkout/"artifacts").exists()


def test_reject_shared_runtime_environment_without_mutation(tmp_path):
    with pytest.raises(ValueError, match="three different"):
        main(["--sim-env", str(tmp_path), "--lab-env", str(tmp_path), "--dry-run"])
    assert not list(tmp_path.iterdir())


def test_existing_conflicting_usd_provider_is_rejected_and_preserved(tmp_path, monkeypatch):
    (tmp_path/"bin").mkdir()
    python = tmp_path/"bin/python"
    python.write_text("preserve this")
    data = {"version": [3, 12, 12], "packages": {**expected_pins("isaac_sim"), "usd-core": "26.8"},
            "project_origin": None}
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: json.dumps(data))
    with pytest.raises(RuntimeError, match="Forbidden USD/runtime provider"):
        probe_environment(tmp_path, "isaac_sim")
    assert python.read_text() == "preserve this"


def test_existing_environment_without_project_installation_is_rejected(tmp_path, monkeypatch):
    (tmp_path/"bin").mkdir()
    (tmp_path/"bin/python").write_text("preserved incomplete environment")
    data = {"version": [3, 12, 12], "packages": expected_pins("base"), "project_origin": None}
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: json.dumps(data))
    with pytest.raises(RuntimeError, match="Missing editable project installation"):
        probe_environment(tmp_path, "base")


def test_asset_manifest_rejects_escaping_paths(tmp_path):
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"source": "file:///unused", "files": [{"path": "../outside"}]}))
    with pytest.raises(ValueError, match="Unsafe pinned asset path"):
        fetch_assets(tmp_path/"cache", manifest)
    assert not (tmp_path/"cache").exists()


def test_asset_cache_mismatch_is_preserved(tmp_path):
    cache = tmp_path/"cache"
    cache.mkdir()
    cached = cache/"robot.usda"
    cached.write_text("local asset retained")
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"source": "file:///unused", "files":
           [{"path": "robot.usda", "bytes": 1, "sha256": "0"*64}]}))
    with pytest.raises(RuntimeError, match="preserved"):
        fetch_assets(cache, manifest)
    assert cached.read_text() == "local asset retained"
