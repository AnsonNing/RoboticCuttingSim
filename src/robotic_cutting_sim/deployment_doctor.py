"""Portable read-only deployment preflight. Metadata inspection never boots Kit."""
import csv
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATHS = {"base": ".venv/bin/python", "isaac_sim": ".deps/isaacsim-venv/bin/python",
                 "isaac_lab": ".deps/isaaclab-venv/bin/python"}
COMMON = {"newton": "1.6.1", "warp-lang": "1.17.0", "mujoco": "3.12.0", "mujoco-warp": "3.12.0",
          "numpy": None, "scipy": None, "Pillow": None, "matplotlib": None}
REQUIRED = {
    "base": {**COMMON, "usd-core": "26.8", "pytest": None, "newton-usd-schemas": "0.6.0"},
    "isaac_sim": {**COMMON, "isaacsim": "6.2.0.0", "isaacsim-kernel": "6.2.0.0",
                  "isaacsim-core": "6.2.0.0", "torch": "2.11.0+cu128", "imageio": None, "newton-usd-schemas": "0.5.0"},
    "isaac_lab": {**COMMON, "isaaclab": "28.0.0", "isaaclab-newton": "9.1.4",
                  "torch": "2.11.0+cu128", "imageio": None, "usd-exchange": "3.0.1", "newton-usd-schemas": "0.6.0"},
}
# These dependencies are declared by unused installed extras. Installing them
# to silence pip-check would contaminate the single USD-provider contract.
OPTIONAL_METADATA = {
    "isaac_sim": {("mujoco-usd-converter", "usd-exchange"), ("urdf-usd-converter", "usd-exchange"),
                  ("osqp", "joblib")},
    "isaac_lab": {("isaaclab-ov", "ovstage")}, "base": set(),
}


def run_command(argv, *, cwd=ROOT, timeout=30):
    """Read-only subprocess receipt, including failures rather than hiding them."""
    try:
        result = subprocess.run([str(v) for v in argv], cwd=cwd, text=True, capture_output=True, timeout=timeout)
        return {"command": list(map(str, argv)), "returncode": result.returncode,
                "stdout": result.stdout[-65536:], "stderr": result.stderr[-65536:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(map(str, argv)), "returncode": None, "stdout": "", "stderr": str(error)}


def parse_gpu_csv(text):
    result = []
    for row in csv.reader(text.splitlines()):
        if len(row) != 8:
            continue
        try:
            index, uuid, name, total, free, driver, capability, utilization = [v.strip() for v in row]
            gpu = {"physical_index": int(index), "uuid": uuid, "name": name,
                   "memory_total_mib": int(total), "memory_free_mib": int(free), "driver_version": driver,
                   "compute_capability": capability, "utilization_percent": float(utilization)}
        except ValueError:
            continue
        gpu["rtx_family_candidate"] = bool(re.search(r"RTX|\b(?:L4|L40S?|A40|A10)\b", name, re.I))
        result.append(gpu)
    return result


def select_gpu(gpus, requested="auto"):
    """Select a physical index; never rewrite CUDA_VISIBLE_DEVICES."""
    if not gpus:
        return {"status": "BLOCKED_NO_NVIDIA_GPU", "physical_index": None, "cuda_device": None}
    if str(requested) != "auto":
        try:
            index = int(requested)
        except (TypeError, ValueError):
            return {"status": "BLOCKED_INVALID_GPU", "physical_index": None, "cuda_device": None}
        selected = next((g for g in gpus if g["physical_index"] == index), None)
        if selected is None:
            return {"status": "BLOCKED_REQUESTED_GPU_ABSENT", "physical_index": None, "cuda_device": None}
        basis = "Explicit physical GPU index"
    else:
        candidates = [g for g in gpus if g["rtx_family_candidate"]]
        selected = (sorted(candidates, key=lambda g: (-g["memory_free_mib"], g["utilization_percent"], g["physical_index"]))[0]
                    if candidates else min(gpus, key=lambda g: g["physical_index"]))
        basis = ("Most free VRAM among RTX-family candidates; native smoke must verify actual compatibility"
                 if candidates else "No recognized RTX family: fallback to first physical GPU; native compatibility unverified")
    return {"status": "SELECTED_CANDIDATE" if selected["rtx_family_candidate"] else "WARN_RTX_COMPATIBILITY_UNVERIFIED",
            "physical_index": selected["physical_index"], "cuda_device": f"cuda:{selected['physical_index']}",
            "uuid": selected["uuid"], "name": selected["name"], "memory_free_mib": selected["memory_free_mib"], "basis": basis,
            "native_compatibility_validated": False, "cuda_visible_devices_modified": False}


def classify_pip_check(runtime, receipt):
    lines = [s.strip() for s in (receipt.get("stdout", "")+"\n"+receipt.get("stderr", "")).splitlines() if s.strip()]
    warnings = [s for s in lines if s != "No broken requirements found."]
    expected, unexpected = [], []
    for line in warnings:
        match = re.fullmatch(r"(\S+) \S+ requires ([\w-]+), which is not installed\.", line)
        (expected if match and (match[1].lower(), match[2].lower()) in OPTIONAL_METADATA[runtime] else unexpected).append(line)
    status = ("PASS" if receipt.get("returncode") == 0 and not warnings else
              "WARN_EXPECTED_OPTIONAL_METADATA" if receipt.get("returncode") == 1 and expected and not unexpected else
              "WARN_UNEXPECTED_METADATA" if receipt.get("returncode") in (0, 1) else "NOT_RUN_PIP_CHECK")
    return {"status": status, "returncode": receipt.get("returncode"), "expected_optional_lines": expected,
            "unexpected_lines": unexpected, "raw": receipt, "dependencies_changed": False}


def assess_runtime(runtime, metadata, pip_check, *, project_root=None):
    packages = metadata.get("packages", {})
    missing = [k for k in REQUIRED[runtime] if not packages.get(k)]
    mismatched = {k: {"expected": v, "actual": packages.get(k)} for k, v in REQUIRED[runtime].items()
                  if v is not None and packages.get(k) and packages[k] != v}
    providers = {k: packages.get(k) for k in ("usd-core", "usd-exchange")}
    conflicts = ((runtime == "isaac_sim" and any(providers.values())) or
                 (runtime == "isaac_lab" and bool(providers["usd-core"])) or
                 (runtime == "base" and bool(providers["usd-exchange"])))
    python_ok = metadata.get("python_version_info", [])[:2] == [3, 12]
    origin = metadata.get("project_import_origin")
    same_project = (bool(origin) and Path(origin).resolve().is_relative_to(Path(project_root).resolve()/"src/robotic_cutting_sim")) if project_root else None
    blocked = bool(missing or mismatched or conflicts or not python_ok or metadata.get("probe_error") or same_project is False)
    return {"status": "BLOCKED_REQUIRED_RUNTIME_METADATA" if blocked else "PASS_METADATA_PRECHECK",
            "python_3_12": python_ok, "missing_required_distributions": missing, "version_mismatches": mismatched,
            "usd_provider_conflict": bool(conflicts), "external_usd_providers": providers,
            "pip_check": pip_check, "metadata": metadata, "runtime_imports_executed": False,
            "project_origin_matches_checkout": same_project,
            "kit_or_lab_boot_validated": False}


def probe_runtime(runtime, executable, *, root=ROOT):
    executable = Path(executable).expanduser().absolute()
    names = sorted(set(REQUIRED[runtime]) | {"usd-core", "usd-exchange", "torch", "torchvision", "isaacsim-sensor"})
    code = """import importlib.metadata as m,importlib.util as u,json,sys
names=json.loads(sys.argv[1]);packages={};locations={}
for name in names:
 try:
  d=m.distribution(name);packages[name]=d.version;locations[name]=str(d.locate_file(''))
 except m.PackageNotFoundError:packages[name]=None
project=u.find_spec('robotic_cutting_sim');lab=u.find_spec('isaaclab')
print(json.dumps({'python_version':sys.version,'python_version_info':list(sys.version_info[:3]),'executable':sys.executable,'prefix':sys.prefix,'packages':packages,'distribution_locations':locations,'project_import_origin':project.origin if project else None,'isaaclab_import_origin':lab.origin if lab else None,'project_package_executed':False}))
"""
    # Isolated mode ignores caller PYTHONPATH/PYTHONHOME and the working directory,
    # while retaining the selected venv's installed editable-package machinery.
    raw = run_command([executable, "-I", "-c", code, json.dumps(names)], cwd=root)
    try:
        metadata = json.loads(raw["stdout"]) if raw["returncode"] == 0 else {"probe_error": raw["stderr"]}
    except (json.JSONDecodeError, TypeError):
        metadata = {"probe_error": "Metadata subprocess did not return its JSON contract", "raw": raw}
    pip = classify_pip_check(runtime, run_command([executable, "-I", "-m", "pip", "check"], cwd=root))
    result = assess_runtime(runtime, metadata, pip, project_root=root)
    result["requested_python"] = str(executable)
    return result


def _directory_inventory(paths, patterns):
    found = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for pattern in patterns:
                found.extend(str(p) for p in path.glob(pattern))
    return sorted(set(found))[:100]


def collect_report(*, root=ROOT, gpu="auto", runtime="both", pythons=None, asset_cache=None):
    root = Path(root).resolve()
    selected = ["base"]+(["isaac_sim", "isaac_lab"] if runtime == "both" else [runtime])
    pythons = pythons or {}
    gpu_probe = run_command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version,compute_cap,utilization.gpu",
                             "--format=csv,noheader,nounits"], cwd=root)
    gpus = parse_gpu_csv(gpu_probe["stdout"])
    selection = select_gpu(gpus, gpu)
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    mask_compatible = mask is None or (mask.strip() == "0" and selection["physical_index"] == 0)
    selection.update(cuda_visible_devices=mask, physical_cuda_numbering_unambiguous=mask_compatible)
    if not mask_compatible:
        selection["status"] = "BLOCKED_EXISTING_CUDA_VISIBLE_DEVICES_MAPPING"
    runtimes = {name: probe_runtime(name, pythons.get(name) or root/RUNTIME_PATHS[name], root=root) for name in selected}
    sim_meta = runtimes.get("isaac_sim", {}).get("metadata", {})
    site_dirs = sorted(set(sim_meta.get("distribution_locations", {}).values()))
    extension_roots = [root/".deps/isaacsim-extension-cache", Path.home()/".local/share/ov/data/exts/v2"]
    extension_roots += [Path(p)/"isaacsim/extscache" for p in site_dirs]
    extension_roots += [Path(p)/"isaacsim/exts" for p in site_dirs]
    extensions = _directory_inventory(extension_roots, ["omni.sensors.nv.camera*", "omni.physx*", "omni.syntheticdata*", "omni.kit.livestream*",
        "isaacsim.sensors.experimental.physics*", "isaacsim.robot.schema*"])
    cache = Path(asset_cache or root/"artifacts/workcell_asset_cache").expanduser().resolve()
    required_assets = ["Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda", "Isaac/Sensors/RealSense/D455/rsd455.usd"]
    asset_status = {p: {"present": (cache/p).is_file(), "bytes": (cache/p).stat().st_size if (cache/p).is_file() else 0} for p in required_assets}
    asset_lock = root/"configs/runtime_assets.lock.json"
    asset_hashes = []
    if asset_lock.is_file():
        lock = json.loads(asset_lock.read_text())
        for item in lock.get("files", []):
            relative = Path(item.get("relative_path", item.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts or not (cache/relative).resolve().is_relative_to(cache):
                asset_hashes.append({"path": str(relative), "status": "FAIL_UNSAFE_MANIFEST_PATH"})
                continue
            path = cache/relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            asset_hashes.append({"path": str(path), "expected_sha256": item.get("sha256"), "actual_sha256": actual,
                                 "status": "PASS" if actual and actual == item.get("sha256") else "MISSING" if actual is None else "FAIL_HASH"})
    source_pins = json.loads((root/"configs/source_refs.json").read_text()) if (root/"configs/source_refs.json").is_file() else {}
    source_revisions = {name: {"expected": pin["sha"], "actual": run_command(["git", "-C", root/".deps"/name, "rev-parse", "HEAD"], cwd=root)}
                        for name, pin in source_pins.items() if (root/".deps"/name).is_dir()}
    if "isaac_lab" in runtimes:
        # A custom --lab-source editable is valid only at the pinned revision.
        # Locate the installed package without executing its Kit/Lab imports.
        origin = runtimes["isaac_lab"]["metadata"].get("isaaclab_import_origin")
        source_root = run_command(["git", "-C", Path(origin).parent, "rev-parse", "--show-toplevel"], cwd=root) if origin else None
        source_revisions["IsaacLab"] = {
            "expected": source_pins.get("IsaacLab", {}).get("sha"), "installed_package_origin": origin,
            "source_root": source_root,
            "actual": run_command(["git", "-C", source_root["stdout"].strip(), "rev-parse", "HEAD"], cwd=root)
            if source_root and source_root["returncode"] == 0 else {"returncode": None, "stdout": "", "stderr": "Installed Lab package origin is absent or not a Git source"}}
    for value in source_revisions.values():
        value["status"] = "PASS_SOURCE_PIN" if value["actual"]["returncode"] == 0 and value["actual"]["stdout"].strip() == value["expected"] else "WARN_SOURCE_PIN_MISMATCH"
    usage = shutil.disk_usage(root)
    ram = {}
    if Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                key, value = line.split(":", 1);ram[key+"_bytes"] = int(value.split()[0])*1024
    smi_header = run_command(["nvidia-smi"], cwd=root)
    advertised = re.search(r"CUDA Version:\s*([\d.]+)", smi_header["stdout"])
    nvcc = shutil.which("nvcc") or next((str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/nvcc")) if p.is_file()), None)
    blocked = selection["status"].startswith("BLOCKED") or any(r["status"].startswith("BLOCKED") for r in runtimes.values())
    warnings = any(r["pip_check"]["status"] != "PASS" for r in runtimes.values())
    distribution = platform.freedesktop_os_release() if Path("/etc/os-release").is_file() else {}
    supported_platform = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")
    blocked |= not supported_platform
    measured_driver = (570, 211, 1)
    selected_driver = next((g["driver_version"] for g in gpus if g["physical_index"] == selection["physical_index"]), "")
    actual_driver = tuple(int(v) for v in selected_driver.split(".") if v.isdigit())
    installation_reference_gib = sum({"base": 1.2, "isaac_sim": 11., "isaac_lab": 7.8}[k]
        for k, r in runtimes.items() if r["status"].startswith("BLOCKED")) + (1.6 if not extensions else 0.)
    advisories = [
        {"check": "platform", "status": "PASS" if supported_platform else "BLOCKED", "value": [platform.system(), platform.machine()]},
        {"check": "tested_distribution", "status": "PASS" if distribution.get("ID") == "ubuntu" and distribution.get("VERSION_ID") in ("22.04", "24.04") else "WARN_UNTESTED_DISTRIBUTION", "value": distribution},
        {"check": "driver_reference", "status": "WARN_BELOW_MEASURED_REFERENCE" if actual_driver and actual_driver < measured_driver else "PASS_RECORDED_REFERENCE_ONLY",
         "value": selected_driver, "scope": "570.211.01 was measured here, not a universal minimum; only an actual smoke verifies this machine"},
        {"check": "free_vram", "status": "WARN_SHARED_DEVICE_HEADROOM" if selection.get("memory_free_mib", 0) < 4096 else "PASS_ADVISORY_ONLY",
         "value_mib": selection.get("memory_free_mib"), "scope": "4GiB is an advisory compact-scene headroom reference, never an arbitrary compatibility block"},
        {"check": "disk_installation_headroom", "status": "WARN_REFERENCE_HEADROOM" if usage.free/1024**3 < installation_reference_gib else "PASS_ADVISORY_ONLY",
         "free_gib": usage.free/1024**3, "missing_installation_reference_gib": installation_reference_gib, "scope": "Reference sizes only; existing envs are not duplicated, no deletion is performed"},
        {"check": "extensions", "status": "PRESENT_NOT_BOOT_VALIDATED" if extensions else "WARN_FIRST_BOOT_DOWNLOAD", "action": "Existing Kit launch fetches missing pinned extensions; doctor never boots it"},
        {"check": "asset_cache", "status": "WARN_MISSING_OR_HASH_MISMATCH" if any(a["status"] != "PASS" for a in asset_hashes) or not all(a["present"] for a in asset_status.values()) else "PASS_PINNED_HASHES" if asset_hashes else "PRESENT_COMPOSITION_NOT_VALIDATED",
         "action": "Use scripts/fetch_runtime_assets.py for pinned official cache; static validation checks composition"},
    ]
    warnings |= any(a["status"].startswith("WARN") for a in advisories)
    if "isaac_lab" in runtimes and source_revisions["IsaacLab"]["status"] != "PASS_SOURCE_PIN":
        runtimes["isaac_lab"]["status"] = "BLOCKED_INSTALLED_LAB_SOURCE_PIN"
        blocked = True
    return {"schema": "portable_deployment_doctor_v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "BLOCKED_PRECHECK" if blocked else "PASS_METADATA_PRECHECK_WITH_WARNINGS" if warnings else "PASS_METADATA_PRECHECK",
            "scope": "Read-only OS/hardware/distribution metadata; never Kit boot, physics, rendering or installation proof",
            "root": str(root), "host": {"os": platform.platform(), "distribution": distribution,
                "machine": platform.machine(), "cpu_count": os.cpu_count(),
                "doctor_python": sys.version, "ram": ram}, "gpus": gpus, "gpu_probe": gpu_probe, "gpu_selection": selection,
            "driver_advertised_max_cuda": advertised[1] if advertised else None,
            "local_cuda_toolkit": run_command([nvcc, "--version"], cwd=root) if nvcc else {"status": "NOT_INSTALLED_NOT_REQUIRED_FOR_PINNED_WHEELS"},
            "runtimes": runtimes, "disk": {"path": str(root), "total_bytes": usage.total, "free_bytes": usage.free,
                "advisory_reference_installed_sizes_gib": {"base": 1.2, "isaac_sim": 11., "isaac_lab": 7.8, "extension_cache": 1.6},
                "scope": "Observed reference sizes, not minimum/free-space gates; downloads and outputs require additional space. No files deleted."},
            "extensions": {"searched_roots": list(map(str, extension_roots)), "found": extensions, "runtime_loading_validated": False},
            "asset_cache": {"path": str(cache), "roots": asset_status, "composition_validated": False,
                "pinned_file_checks": asset_hashes,
                "scope": "Presence only; missing cache may be fetched by the bounded static exporter, composition is tested by validation"},
            "source_revisions": source_revisions, "advisories": advisories, "runtime_imports_executed": False, "driver_changed": False,
            "project_or_dependencies_changed": False, "full_benchmark_ready": False}


def write_new_report(output, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return output
