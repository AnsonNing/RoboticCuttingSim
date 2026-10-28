#!/usr/bin/env python3
"""Run a small reproducible matrix of actual Isaac cutting episodes, no training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from robotic_cutting_sim.recording import ROOT, manifest, new_output, write_json


def summarize_case(output, task_id, runtime, returncode):
    output = Path(output)
    result = json.loads((output/"result.json").read_text()) if (output/"result.json").exists() else {}
    goal_file = output/("oracle_debug/task_evaluation.json" if runtime == "isaac_sim" else "task_evaluation.json")
    goal = json.loads(goal_file.read_text()) if goal_file.exists() else {}
    checks = goal.get("checks", {})
    common = ["complete_episode", "initial_continuous_graph", "initial_codes_zero",
        "reference_matches_single_continuous_task_sampler", "reference_mass_budget_and_unchanged_count",
        "persistent_history_no_digit_rewrites", "postcut_disconnected_topology_persists"]
    physical = common+(["reference_above_plane_material_removed"] if task_id == "TRIM-01" else
        ["two_substantial_physical_parts", "two_main_parts_retain_mass", "cut_at_workpiece_local_target_plane"])
    full = result.get("status") == "PASS_EXECUTION" and returncode == 0
    return {"task_id": task_id, "artifact": str(output), "runtime": runtime, "returncode": returncode,
        "episode_execution": "PASS" if full else ("FAIL" if returncode else result.get("status", "FAIL")),
        "physical_cutting": ("PASS" if all(checks.get(name, False) for name in physical) else "FAIL") if full else "NOT_RUN",
        "physical_cutting_scope": "Actual stencil topology, reference-region removal, mass and history; not calibrated fracture or energy convergence",
        "task_goal_v1": goal.get("status", "NOT_RUN"), "task_checks": checks, "task_metrics": goal.get("metrics", {}),
        "camera_acceptance": "SEPARATE_AUDIT_REQUIRED", "full_simulation_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime", choices=("isaac_lab", "isaac_sim"), default="isaac_sim")
    parser.add_argument("--tasks", nargs="+", choices=("CUT-01", "TRIM-01"), default=["CUT-01", "TRIM-01"])
    parser.add_argument("--variants", nargs="+", choices=("fixed", "randomized"), default=["fixed"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--camera-layout", choices=("legacy_fixed_d455", "wrist_d405_global_d455"), default="wrist_d405_global_d455")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--particle-spacing", type=float, help="Explicit override for all cases; otherwise CUT1mm/TRIM2mm")
    args = parser.parse_args()
    if not 1 <= args.steps <= 400:
        parser.error("steps must be1..400")
    matrix = [(task, variant, seed) for task in args.tasks for variant in args.variants for seed in args.seeds]
    if len(set(matrix)) != len(matrix):
        parser.error("Duplicate matrix cases are not allowed")
    from robotic_cutting_sim.tasks import make_task
    for task, variant, seed in matrix:
        make_task(task, seed=seed, variant=variant,
            particle_spacing_m=args.particle_spacing if args.particle_spacing is not None else (.002 if task == "TRIM-01" else .001))
    output = new_output(args.output_dir)
    source = Path(__file__).resolve()
    shutil.copyfile(source, output/source.name)
    write_json(output/"manifest.json", manifest(vars(args)|{"output_dir": str(output),
        "launcher_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}, "actual_cutting_task_matrix"))
    executable = ROOT/".deps"/("isaacsim-venv" if args.runtime == "isaac_sim" else "isaaclab-venv")/"bin/python"
    runner = ROOT/"scripts"/("run_isaac_sim_cutting.py" if args.runtime == "isaac_sim" else "run_isaac_lab_cutting.py")
    rows = []
    for task, variant, seed in matrix:
        name = f"{task.lower()}_{variant}_seed{seed}"
        case = output/name
        spacing = args.particle_spacing if args.particle_spacing is not None else (.002 if task == "TRIM-01" else .001)
        command = [str(executable), str(runner), "--task-id", task, "--task-variant", variant,
            "--seed", str(seed), "--steps", str(args.steps), "--output-dir", str(case),
            "--device", f"cuda:{args.gpu}", "--particle-spacing", str(spacing), "--camera-layout", args.camera_layout]
        command += (["--gpu", str(args.gpu), "--camera-mode", "native_single_view"] if args.runtime == "isaac_sim" else
                    ["--material-device", f"cuda:{args.gpu}", "--reset-check"])
        print(f"RUN {name}: actual {args.runtime}, spacing={spacing}m", flush=True)
        write_json(output/"progress.json", {"running": name, "completed": rows, "total_cases": len(matrix)})
        with (output/(name+".log")).open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = summarize_case(case, task, args.runtime, process.returncode)
        row["command"] = command
        rows.append(row)
        print(f"DONE {name}: execution={row['episode_execution']}, physics={row['physical_cutting']}, goal={row['task_goal_v1']}", flush=True)
    result = {"status": "COMPLETE" if all(row["episode_execution"] == "PASS" for row in rows) else "INCOMPLETE_OR_FAILED",
        "scope": "Actual episode matrix; individual task/physics/camera criteria remain separate",
        "cases": rows, "full_simulation_acceptance": False, "policy_training": False}
    write_json(output/"result.json", result)
    write_json(output/"progress.json", {"running": None, "completed": rows, "total_cases": len(matrix)})
    return 0 if all(row["episode_execution"] in ("PASS", "PASS_SMOKE") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
