#!/usr/bin/env python3
"""Install pinned project-local base, Sim6.2 and Lab3 environments.

The bootstrap itself uses only Python3.10+ standard libraries. It never changes
the system driver or packages, shares site-packages, or overwrites an existing
environment. GPU/task verification is a separate deployment-validation command.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shlex
import shutil
import subprocess
import sys
import zipfile

from fetch_runtime_assets import fetch_assets, fetch_file, sha256

ROOT = Path(__file__).resolve().parents[1]
LAB_PACKAGES = ("isaaclab", "isaaclab_newton", "isaaclab_ov", "isaaclab_physx",
                "isaaclab_visualizers", "isaaclab_assets", "isaaclab_ppisp", "isaaclab_contrib")


def installer_environment(cache, accept_eula):
    """Large CUDA wheels need bounded concurrency and a longer HTTP timeout."""
    return {**os.environ, "UV_CACHE_DIR": str(cache), "UV_PYTHON_INSTALL_DIR": str(ROOT/".deps/python"),
            "UV_PYTHON_BIN_DIR": str(ROOT/".deps/python/bin"),
            "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "900"),
            "UV_HTTP_RETRIES": os.environ.get("UV_HTTP_RETRIES", "8"),
            "UV_CONCURRENT_DOWNLOADS": os.environ.get("UV_CONCURRENT_DOWNLOADS", "4"),
            "OMNI_KIT_ACCEPT_EULA": "YES" if accept_eula else "NO"}


def read_pins(path):
    pins = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            pins.update(read_pins(Path(path).parent/line[3:].strip()))
        else:
            name, version = line.split("==", 1)
            pins[name.lower().replace("_", "-")] = version
    return pins


def expected_pins(runtime):
    if runtime == "base":
        return read_pins(ROOT/"requirements-robot.lock")
    pins = read_pins(ROOT/f"requirements-{runtime.replace('_', '-')}.lock")
    pins.update(read_pins(ROOT/"requirements-torch-cu128.lock"))
    if runtime == "isaac_lab":
        # The native Sim wheel requires torchaudio; kit-less Lab did not use it.
        pins.pop("torchaudio", None)
    return pins


def probe_environment(path, runtime):
    python = Path(path)/"bin/python"
    if not python.is_file():
        raise RuntimeError(f"Existing environment has no Python; preserved: {path}")
    code = """import importlib.metadata as m, importlib.util, json, sys
s=importlib.util.find_spec('robotic_cutting_sim')
print(json.dumps({'version':list(sys.version_info[:3]),'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()},'project_origin':s.origin if s else None}))"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    result = json.loads(subprocess.check_output([str(python), "-c", code], text=True, env=environment))
    problems = []
    if result["version"][:2] != [3, 12]:
        problems.append(f"Python must be3.12; found{result['version']}")
    actual = result["packages"]
    for name, value in expected_pins(runtime).items():
        if actual.get(name) != value:
            problems.append(f"{name}: expected{value}, found{actual.get(name)}")
    forbidden = ({"usd-exchange"} if runtime == "base" else
                 {"usd-core", "isaacsim"} if runtime == "isaac_lab" else {"usd-core", "usd-exchange"})
    problems.extend(f"Forbidden USD/runtime provider: {name}" for name in sorted(forbidden & actual.keys()))
    if result["project_origin"] is not None and Path(result["project_origin"]).resolve() != ROOT/"src/robotic_cutting_sim/__init__.py":
        problems.append("Editable project must point to this checkout, not another repository")
    result["project_install"] = ("this_checkout_editable" if result["project_origin"] else
                                 "not_installed; existing repository scripts explicitly add src")
    if runtime == "isaac_lab":
        for name in LAB_PACKAGES:
            if name.replace("_", "-") not in actual:
                problems.append(f"Missing pinned-source editable{ name}")
    result.update(status="PASS_INSTALL_METADATA" if not problems else "BLOCKED_EXISTING_ENVIRONMENT",
                  runtime=runtime, environment=str(Path(path).resolve()), problems=problems,
                  gpu_validation="NOT_RUN")
    if problems:
        raise RuntimeError("Environment validation failed; no packages changed:\n"+"\n".join(problems))
    return result


def run(command, log, environment=None):
    command = [str(value) for value in command]
    print("$ "+shlex.join(command), flush=True)
    with Path(log).open("a") as stream:
        stream.write("\n$ "+shlex.join(command)+"\n"); stream.flush()
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"Command exited{process.returncode}; preserved log: {log}")


def ensure_uv(cache, refs):
    target = ROOT/".deps/uv/uv"
    ref = refs["uv"]
    if target.is_file():
        version = subprocess.check_output([str(target), "--version"], text=True).strip()
        if version != "uv "+ref["version"]:
            raise RuntimeError(f"Existing project-local uv differs from pin; preserved: {version}")
        return target
    archive = cache/"bootstrap"/Path(ref["url"]).name
    fetch_file(ref["url"], archive, ref["sha256"], ref["bytes"])
    with zipfile.ZipFile(archive) as wheel:
        data = wheel.read(ref["member"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data); target.chmod(0o755)
    return target


def select_python(requested, uv, environment, managed, log):
    candidates = [requested] if requested else [shutil.which("python3.12"), sys.executable]
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.expanduser(str(candidate))
        candidate = shutil.which(candidate) or str(Path(candidate).absolute())
        try:
            version = subprocess.check_output([str(candidate), "-c", "import sys; print('%d.%d'%sys.version_info[:2])"], text=True).strip()
            if version == "3.12":
                return str(candidate)
        except (OSError, subprocess.CalledProcessError):
            pass
    if requested:
        raise RuntimeError("--python must name a working Python3.12 interpreter")
    if not managed:
        raise RuntimeError("Python3.12 was not found. Pass --python PATH or --managed-python for a project-local download")
    run([uv, "python", "install", "3.12.12"], log, environment)
    return subprocess.check_output([str(uv), "python", "find", "3.12.12"], env=environment, text=True).strip()


def ensure_source(name, target, refs, log):
    ref = refs[name]
    if target.exists():
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != ref["sha"]:
            raise RuntimeError(f"Existing{name} checkout differs from pin; preserved: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "init", target], log)
        run(["git", "-C", target, "remote", "add", "origin", ref["url"]], log)
        run(["git", "-C", target, "fetch", "--depth", "1", "origin", ref["sha"]], log)
        run(["git", "-C", target, "checkout", "--detach", ref["sha"]], log)
    dirty = subprocess.check_output(["git", "-C", str(target), "diff", "HEAD", "--name-only"], text=True).strip()
    if name == "IsaacLab" and dirty:
        raise RuntimeError(f"Pinned IsaacLab source has tracked edits; preserved: {dirty}")
    return {**ref, "path": str(target), "tracked_modified_files": dirty.splitlines()}


def install_environment(runtime, target, python, uv, environment, log, lab_source):
    run([uv, "venv", "--python", python, target], log, environment)
    executable = target/"bin/python"
    install = [uv, "pip", "install", "--python", executable, "--link-mode", "hardlink", "--no-deps"]
    run(install+["pip==25.0.1", "setuptools==78.1.0", "wheel==0.45.1"], log, environment)
    if runtime == "base":
        run(install+["-r", ROOT/"requirements-robot.lock"], log, environment)
    else:
        torch_lock = ROOT/"requirements-torch-cu128.lock"
        if runtime == "isaac_lab":
            # Match the original kit-less environment, without an unused audio wheel.
            torch_lock = Path(log).parent/"lab-torch.lock"
            torch_lock.write_text("\n".join(line for line in (ROOT/"requirements-torch-cu128.lock").read_text().splitlines()
                                             if not line.startswith("torchaudio=="))+"\n")
        run(install+["-r", torch_lock, "--index-url", "https://download.pytorch.org/whl/cu128",
                     "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"], log, environment)
        run(install+["-r", ROOT/f"requirements-{runtime.replace('_', '-')}.lock",
                     "--index-url", "https://pypi.org/simple", "--extra-index-url", "https://pypi.nvidia.com",
                     "--index-strategy", "unsafe-best-match"], log, environment)
    if runtime == "isaac_lab":
        for name in LAB_PACKAGES:
            run(install+["--no-build-isolation", "-e", lab_source/"source"/name], log, environment)
    run(install+["--no-build-isolation", "-e", ROOT], log, environment)
    return probe_environment(target, runtime)


def install_shadercache(cache, refs):
    ref = refs["shadercache"]
    target = ROOT/".deps/isaacsim-extension-cache"/ref["extension_name"]
    config = target/"config/extension.toml"
    if target.exists():
        if not config.is_file() or sha256(config) != ref["config_sha256"]:
            raise RuntimeError(f"Existing shadercache differs from the pinned Kit build; preserved: {target}")
        return {"status": "EXISTING_CONFIG_VERIFIED", "path": str(target), **ref}
    archive = cache/"bootstrap"/Path(ref["url"]).name
    source = fetch_file(ref["url"], archive, ref["sha256"], ref["bytes"])
    with zipfile.ZipFile(archive) as compressed:
        for entry in compressed.infolist():
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe vendor shadercache member: {relative}")
        target.mkdir(parents=True)
        compressed.extractall(target)
    if sha256(config) != ref["config_sha256"]:
        raise RuntimeError("Unpacked vendor shadercache config differs from the validated build")
    return {"status": "PINNED_OFFICIAL_ARCHIVE_INSTALLED", "source": source, "path": str(target), **ref}


def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=
        "Partial failed environments are preserved. Retry in a new checkout or use new explicit environment paths. "
        "UV_HTTP_TIMEOUT (default900s), UV_CONCURRENT_DOWNLOADS (default4) and UV_HTTP_RETRIES are configurable.")
    p.add_argument("--runtime", choices=("base", "isaac_sim", "isaac_lab", "all"), default="base")
    p.add_argument("--python", help="Explicit Python3.12 interpreter; never a shared site-packages environment")
    p.add_argument("--managed-python", action="store_true", help="Download project-local Python3.12.12 if absent")
    p.add_argument("--base-env", type=Path, default=ROOT/".venv")
    p.add_argument("--sim-env", type=Path, default=ROOT/".deps/isaacsim-venv")
    p.add_argument("--lab-env", type=Path, default=ROOT/".deps/isaaclab-venv")
    p.add_argument("--lab-source", type=Path, default=ROOT/".deps/IsaacLab")
    p.add_argument("--cache-dir", type=Path, default=ROOT/".deps/uv-cache")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--accept-eula", action="store_true", help="Accept NVIDIA Omniverse terms for the requested Sim installation")
    p.add_argument("--fetch-sources", action="store_true", help="Also fetch all pinned comparison source repositories")
    p.add_argument("--skip-assets", action="store_true")
    p.add_argument("--skip-shadercache", action="store_true", help="Let Kit fetch its own Vulkan shadercache on first launch")
    p.add_argument("--check-only", action="store_true", help="Validate existing env metadata; do not install/download")
    p.add_argument("--dry-run", action="store_true", help="Print paths/actions without creating or installing anything")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    runtimes = ("base", "isaac_sim", "isaac_lab") if args.runtime == "all" else (args.runtime,)
    paths = {"base": args.base_env.resolve(), "isaac_sim": args.sim_env.resolve(), "isaac_lab": args.lab_env.resolve()}
    if len(set(paths.values())) != 3:
        raise ValueError("Base, Sim and Lab must use three different environments")
    plan = {"root": str(ROOT), "runtimes": list(runtimes), "environments": {name: str(paths[name]) for name in runtimes},
            "cache_dir": str(args.cache_dir.resolve()), "existing_environments": "validate_and_skip_without_mutation",
            "download_settings": {key: installer_environment(args.cache_dir, args.accept_eula)[key] for key in
                                  ("UV_HTTP_TIMEOUT", "UV_HTTP_RETRIES", "UV_CONCURRENT_DOWNLOADS")},
            "usd_providers": {"base": "usd-core26.8", "isaac_sim": "Isaac6.2 bundled USD25.11", "isaac_lab": "usd-exchange3.0.1"},
            "gpu_validation": "NOT_RUN", "driver_changes": False}
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", **plan}, indent=2)); return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This pinned installation currently targets Linux x86_64 only")
    new_sim = "isaac_sim" in runtimes and not paths["isaac_sim"].exists()
    if new_sim and not args.accept_eula and not args.check_only:
        raise RuntimeError("Isaac installation requires explicit --accept-eula; see NVIDIA Omniverse License Agreement")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_dir or ROOT/"artifacts/bootstrap"/stamp
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "RUNNING", **plan, "environments_checked": [], "eula_accepted": bool(args.accept_eula),
               "eula_url": "https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html",
               "lock_sha256": {path.name: sha256(path) for path in
                  (ROOT/"requirements-robot.lock", ROOT/"requirements.lock", ROOT/"requirements-isaac-sim.lock",
                   ROOT/"requirements-isaac-lab.lock", ROOT/"requirements-torch-cu128.lock", ROOT/"configs/bootstrap_refs.json",
                   ROOT/"configs/source_refs.json", ROOT/"configs/runtime_assets.lock.json")}}
    try:
        if new_sim and args.accept_eula:
            print("NVIDIA Omniverse license accepted by explicit --accept-eula: "+receipt["eula_url"], flush=True)
        # Inspect all requested existing envs before allowing any installation.
        for runtime in runtimes:
            if paths[runtime].exists() or args.check_only:
                receipt["environments_checked"].append({**probe_environment(paths[runtime], runtime), "action": "existing_verified_skipped"})
        if "isaac_lab" in runtimes and paths["isaac_lab"].exists():
            source_refs = json.loads((ROOT/"configs/source_refs.json").read_text())
            if not args.lab_source.is_dir():
                raise RuntimeError("Existing Lab environment requires its pinned --lab-source checkout")
            receipt["lab_source"] = ensure_source("IsaacLab", args.lab_source.resolve(), source_refs, output/"sources.log")
        if not args.check_only:
            refs = json.loads((ROOT/"configs/bootstrap_refs.json").read_text())
            source_refs = json.loads((ROOT/"configs/source_refs.json").read_text())
            needed = [name for name in runtimes if not paths[name].exists()]
            if needed:
                cache = args.cache_dir.resolve(); cache.mkdir(parents=True, exist_ok=True)
                uv = ensure_uv(cache, refs)
                environment = installer_environment(cache, args.accept_eula)
                python = select_python(args.python, uv, environment, args.managed_python, output/"python.log")
                receipt["python"] = python
                if "isaac_lab" in needed:
                    receipt["lab_source"] = ensure_source("IsaacLab", args.lab_source.resolve(), source_refs, output/"sources.log")
                if args.fetch_sources:
                    receipt["source_checkouts"] = {name: ensure_source(name, ROOT/".deps"/name, source_refs, output/"sources.log") for name in source_refs}
                for runtime in needed:
                    result = install_environment(runtime, paths[runtime], python, uv, environment,
                                                 output/f"{runtime}_install.log", args.lab_source.resolve())
                    receipt["environments_checked"].append({**result, "action": "new_official_wheel_install"})
            if not args.skip_assets:
                receipt["assets"] = fetch_assets()
            if "isaac_sim" in runtimes and not args.skip_shadercache:
                receipt["shadercache"] = install_shadercache(args.cache_dir.resolve(), refs)
        receipt["status"] = "PASS_INSTALL_METADATA"
    except Exception as error:
        receipt.update(status="BLOCKED_INSTALL", error=str(error))
        raise
    finally:
        (output/"bootstrap.json").write_text(json.dumps(receipt, indent=2)+"\n")
        print(f"Installation receipt: {output/'bootstrap.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
