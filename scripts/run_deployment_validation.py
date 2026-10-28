#!/usr/bin/env python3
"""Validate portable static/smoke/full scopes with existing recorders and audits.

Known TRIM surface failure stays FAIL_TASK_METRICS. Allowing it never asserts
complete benchmark success, calibrated sensor fidelity or converged physics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from robotic_cutting_sim.deployment_doctor import ROOT, collect_report, write_new_report
from robotic_cutting_sim.deployment_validation import CAMERA_IDS, classify_episode, make_plan, summarize_validation


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def run_logged(command, path, timeout):
    """Own child process group; timeout cannot leave an untracked Kit worker."""
    start = time.monotonic()
    record = {"command": command, "log": str(path), "returncode": None, "timed_out": False}
    try:
        with Path(path).open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record["returncode"] = child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                record["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
                record["interrupted"] = isinstance(error, KeyboardInterrupt)
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    record["returncode"] = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    record["returncode"] = child.wait()
    except OSError as error:
        record["error"] = str(error)
    record["wall_s"] = time.monotonic()-start
    record["log_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tier", choices=("static", "smoke", "full"), default="smoke")
    parser.add_argument("--runtime", choices=("isaac_sim", "isaac_lab", "both"), default="both")
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--base-python", type=Path)
    parser.add_argument("--sim-python", type=Path)
    parser.add_argument("--lab-python", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800., help="Seconds per recorder or audit subprocess")
    parser.add_argument("--dry-run", action="store_true", help="Write plans only; status remains NOT_RUN")
    parser.add_argument("--skip-cpu-tests", action="store_true", help="Explicitly omit CPU regression; retain SKIPPED_BY_REQUEST")
    parser.add_argument("--allow-expected-trim-geometry-fail", action="store_true",
                        help="Allow only frozen surface-geometry failures after full TRIM physical invariants pass")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory must be new; failed evidence is never overwritten")
    output.mkdir(parents=True)
    pythons = {k: v for k, v in {"base": args.base_python, "isaac_sim": args.sim_python, "isaac_lab": args.lab_python}.items() if v}
    doctor = collect_report(root=ROOT, gpu=args.gpu, runtime=args.runtime, pythons=pythons)
    write_new_report(output/"doctor", doctor)
    selected = doctor["gpu_selection"]["physical_index"]
    plan = make_plan(ROOT, output, tier=args.tier, runtime=args.runtime, gpu=selected if selected is not None else 0,
                     pythons=pythons, seed=args.seed)
    sources = [Path(__file__), ROOT/"src/robotic_cutting_sim/deployment_doctor.py", ROOT/"src/robotic_cutting_sim/deployment_validation.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    snapshot = output/"deployment_sources";snapshot.mkdir()
    for path in sources:(snapshot/path.name).write_bytes(path.read_bytes())
    protocol = {"schema": "portable_deployment_validation_v1", "tier": args.tier, "runtime": args.runtime,
        "seed": args.seed, "gpu_selection": doctor["gpu_selection"], "commands": plan, "tool_source_sha256": hashes,
        "allow_expected_trim_geometry_fail": args.allow_expected_trim_geometry_fail,
        "scope": "Static composition, actual runtime/named RGB-D contracts and optional full sampled separation; no complete scientific/hardware acceptance",
        "full_case_recipe": "400x20ms; CUT1mm particles, TRIM explicit2mm; existing4mm material grid and current v3 wrist/Global rig",
        "doctor_does_not_boot_kit": True, "driver_or_dependencies_changed": False}
    write(output/"protocol.json", protocol)
    if args.dry_run:
        write(output/"result.json", {"status": "PLAN_ONLY_NOT_RUN", "requested_scope_pass": False,
                                    "stages": [{"name": p["name"], "status": "NOT_RUN"} for p in plan],
                                    "full_benchmark_ready": False, "protocol": "protocol.json"})
        print("PLAN_ONLY_NOT_RUN", output/"result.json")
        return 0
    stages, failed_smokes = [], set()
    for stage in plan:
        row = {k: v for k, v in stage.items() if not k.endswith("command")}
        if stage["kind"] == "cpu" and args.skip_cpu_tests:
            row["status"] = "SKIPPED_BY_REQUEST"
        elif doctor["runtimes"][stage["runtime"]]["status"].startswith("BLOCKED") or (
                stage["kind"] == "episode" and (doctor["gpu_selection"]["status"].startswith("BLOCKED") or stage["runtime"] in failed_smokes)):
            row.update(status="BLOCKED", reason="Required runtime metadata/GPU mapping or preceding smoke contract is blocked; see doctor and prior stages")
        else:
            print("RUN", stage["name"], flush=True)
            process = run_logged(stage["command"], output/(stage["name"]+".log"), args.timeout)
            row["process"] = process
            directory = Path(stage["output_dir"]) if stage["output_dir"] else None
            if stage["kind"] == "cpu":
                text = Path(process["log"]).read_text(errors="replace") if Path(process["log"]).is_file() else ""
                passed = re.search(r"(\d+) passed", text)
                row.update(status="PASS" if process["returncode"] == 0 and passed else "FAIL",
                           tests_passed=int(passed[1]) if passed else 0)
            elif stage["kind"] == "static":
                result = read(directory/"result.json") or {}
                cameras = read(directory/"cameras.json") or {}
                topology = read(directory/"static_validation.json") or {}
                row["checks"] = {"process": process["returncode"] == 0, "result_status": result.get("status") == "PASS",
                    "official_assets_composed": result.get("official_assets_composed") is True,
                    "static_topology": topology.get("status") == "PASS", "named_camera_prims": set(cameras) == CAMERA_IDS,
                    "usd_exists": (directory/"workcell.usda").is_file()}
                row["status"] = "PASS" if all(row["checks"].values()) else "FAIL"
                row["scope"] = "Static USD only; no physics/camera runtime proof"
            else:
                result = read(directory/"result.json")
                camera, task = None, None
                if process["returncode"] == 0 and result and result.get("status") in ("PASS_EXECUTION", "PASS_SMOKE"):
                    row["camera_audit_process"] = run_logged(stage["camera_audit_command"], output/(stage["name"]+"_camera_audit.log"), args.timeout)
                    audit = read(Path(stage["camera_audit_dir"])/"result.json") or {}
                    camera = audit.get("cases", {}).get(directory.name)
                    if audit.get("analysis_sources_unchanged") is not True or row["camera_audit_process"]["returncode"] != 0:
                        camera = None
                    if stage["task_audit_command"]:
                        row["task_audit_process"] = run_logged(stage["task_audit_command"], output/(stage["name"]+"_task_audit.log"), args.timeout)
                        audit = read(Path(stage["task_audit_dir"])/"result.json") or {}
                        task = audit.get("cases", {}).get(directory.name)
                        if audit.get("analysis_sources_unchanged_during_analysis") is not True or row["task_audit_process"]["returncode"] != 0:
                            task = None
                row["classification"] = classify_episode(result, camera, task, runtime=stage["runtime"],
                    expected_steps=stage["steps"], returncode=process["returncode"], task_id=stage["task_id"],
                    reset=read(directory/"reset_receipt.json") if stage["runtime"] == "isaac_sim" else (result or {}).get("reset_check"))
                row["status"] = "PASS" if row["classification"]["status"] == "PASS_REQUESTED_RUNTIME_CONTRACT" else "FAIL"
                if stage["steps"] == 2 and row["status"] != "PASS":failed_smokes.add(stage["runtime"])
        stages.append(row)
        write(output/"progress.json", {"completed": stages, "next_stage": plan[len(stages)]["name"] if len(stages) < len(plan) else None})
        print("DONE", stage["name"], row["status"], flush=True)
    result = summarize_validation(stages, tier=args.tier, allow_expected_trim_geometry_fail=args.allow_expected_trim_geometry_fail)
    stable = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    result.update(stages=stages, tool_sources_unchanged=stable, doctor="doctor/result.json", protocol="protocol.json")
    if not stable:
        result.update(status="FAIL_TOOL_SOURCE_CHANGED", requested_scope_pass=False)
    write(output/"result.json", result)
    print(result["status"], output/"result.json", flush=True)
    return 0 if result["requested_scope_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
