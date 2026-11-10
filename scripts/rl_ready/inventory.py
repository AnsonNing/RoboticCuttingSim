#!/usr/bin/env python3
"""P1-0 environment inventory: git state, the three isolated runtimes, GPUs, RAM and disk.

Writes one JSON file (never overwrites). Package versions come from each interpreter in a subprocess
(``importlib.metadata`` plus an import attempt with a timeout; failures are recorded, not hidden). No Kit
application is started. Paths are reported relative to the repository; process names and environment
variables are not recorded.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.rl.telemetry import APP_FIELDS, GPU_FIELDS, parse_compute_apps, parse_query_gpu  # noqa: E402

RUNTIMES = {".venv": ".venv/bin/python", ".deps/isaacsim-venv": ".deps/isaacsim-venv/bin/python",
            ".deps/isaaclab-venv": ".deps/isaaclab-venv/bin/python"}
DISTRIBUTIONS = ("warp-lang", "numpy", "torch", "newton", "mujoco-warp", "mujoco", "isaaclab", "isaacsim", "scipy")
IMPORTS = {"warp": "__version__", "numpy": "__version__", "torch": "__version__", "newton": "__version__",
           "mujoco_warp": "__version__"}
# Importing isaacsim / isaaclab outside Kit is not a supported way to probe them; only metadata is read.
PROBE = r"""
import importlib, importlib.metadata as md, json, sys
out = {"python": sys.version.split()[0], "distributions": {}, "imports": {}}
for name in %(dists)r:
    try:
        out["distributions"][name] = md.version(name)
    except Exception as error:
        out["distributions"][name] = None
for module, attribute in %(imports)r.items():
    try:
        value = importlib.import_module(module)
        out["imports"][module] = {"ok": True, "version": str(getattr(value, attribute, None))}
        if module == "torch":
            out["imports"][module]["cuda"] = value.version.cuda
            out["imports"][module]["cuda_available"] = bool(value.cuda.is_available())
    except Exception as error:
        out["imports"][module] = {"ok": False, "error": type(error).__name__ + ": " + str(error)[:200]}
print("INVENTORY_JSON " + json.dumps(out))
"""


def run(args, timeout=60, **kw):
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kw)
        return out.returncode, out.stdout, out.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, "", type(error).__name__


def git():
    def g(*args):
        code, out, _ = run(["git", *args], cwd=ROOT)
        return out.strip() if code == 0 else None
    status = g("status", "--porcelain")
    return dict(head=g("rev-parse", "HEAD"), branch=g("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(status), dirty_paths=status.splitlines() if status else [])


def runtime(label, relative):
    python = ROOT / relative
    if not python.exists():
        return dict(label=label, interpreter=relative, status="MISSING")
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH",)}
    env["CUDA_VISIBLE_DEVICES"] = "1"   # torch.cuda.is_available() only; no kernels or allocations are issued
    start = time.perf_counter()
    code, out, err = run([str(python), "-c", PROBE % dict(dists=DISTRIBUTIONS, imports=IMPORTS)], timeout=180,
                         env=env, cwd="/")
    line = next((l for l in out.splitlines() if l.startswith("INVENTORY_JSON ")), None)
    result = dict(label=label, interpreter=relative, probe_wall_s=time.perf_counter() - start, exit_code=code)
    if line is None:
        result.update(status="PROBE_FAILED", stderr_tail=err[-400:])
    else:
        result.update(status="OK", **json.loads(line[len("INVENTORY_JSON "):]))
    return result


def gpus():
    code, out, _ = run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"])
    rows = parse_query_gpu(out) if code == 0 else []
    code_apps, apps_text, _ = run(["nvidia-smi", f"--query-compute-apps={','.join(APP_FIELDS)}",
                                   "--format=csv,noheader,nounits"])
    apps = parse_compute_apps(apps_text) if code_apps == 0 else []
    own = os.getpid()
    for row in rows:
        mine = [a for a in apps if a["gpu_uuid"] == row["uuid"]]
        row["foreign_compute_processes"] = sum(1 for a in mine if a["pid"] != own)
        row["foreign_compute_memory_mib"] = sum(a["memory_mib"] or 0 for a in mine if a["pid"] != own)
        row.pop("uuid")
    code_v, version, _ = run(["nvidia-smi", "--version"])
    return dict(status="OK" if rows else "UNAVAILABLE", gpus=rows,
                nvidia_smi_version=version.strip().splitlines() if code_v == 0 else None,
                note="Snapshot at inventory time; GPU sharing per benchmark row comes from the run's own telemetry")


def memory():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            info[key + "_gib"] = round(int(value.split()[0]) / 2 ** 20, 2)
    return info


def disks():
    out = {}
    for label, path in (("/", "/"), ("/mnt/data", "/mnt/data")):
        try:
            usage = shutil.disk_usage(path)
            out[label] = dict(total_gib=round(usage.total / 2 ** 30, 1), free_gib=round(usage.free / 2 ** 30, 1),
                              used_percent=round(100 * usage.used / usage.total, 1))
        except OSError as error:
            out[label] = dict(error=type(error).__name__)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    started = time.time()
    data = dict(schema="rcs_rl_inventory_v1", created_unix_s=started,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                command=" ".join([Path(sys.executable).name, *[str(a) for a in sys.argv]]),
                host=dict(platform=platform.platform(), cpu_count=os.cpu_count(), machine=platform.machine()),
                git=git(), runtimes=[runtime(label, rel) for label, rel in RUNTIMES.items()], gpu=gpus(),
                ram=memory(), disk=disks(),
                kit_started=False, decision_d08_runtime=".deps/isaacsim-venv/bin/python with PYTHONPATH=src and no "
                "Kit application for headless GPU material runs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(data, stream, indent=1, sort_keys=True)
    print(json.dumps(dict(out=str(args.out), git=data["git"]["head"], dirty=data["git"]["dirty"],
                          runtimes={r["label"]: r.get("status") for r in data["runtimes"]})))


if __name__ == "__main__":
    main()
