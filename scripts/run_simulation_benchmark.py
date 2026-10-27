#!/usr/bin/env python3
"""Execute seeded D0–D3 robot/cutting episodes and programmable vision metrics.

This is an executable engineering benchmark, not a calibrated task-success claim.
Every case uses one connected initial specimen and the same scripted controller.
Unsupported recipe parameters, numerical failures and sensor limits stay visible.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from robotic_cutting_sim.benchmark import BenchmarkSpec, MetricConfig, analyze_recording
from robotic_cutting_sim.coupled_robot import CouplingConfig
from robotic_cutting_sim.recording import ROOT, manifest, new_output, write_json
from run_coupled_cutting import run as simulate


def execute(argv, log):
    tick = time.perf_counter()
    with log.open("w") as handle:
        result = subprocess.run(list(map(str, argv)), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    receipt = {"command": list(map(str, argv)), "returncode": result.returncode,
               "wall_s": time.perf_counter()-tick}
    write_json(log.with_suffix(".receipt.json"), receipt)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", choices=["D0", "D1", "D2", "D3"], default=["D0", "D1", "D2", "D3"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--duration", type=float, default=8.)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workcell", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if not np.isfinite(args.duration) or args.duration < .1:
        p.error("duration must be finite and at least .1s")
    output = new_output(args.output_dir)
    config = {"variants": args.variants, "seeds": args.seeds, "duration_s": args.duration,
              "device": args.device, "controller": "fixed 8s approach/hold/sweep/withdraw, no policy",
              "camera_mode": "ideal_rgbd", "rgbd_samples_per_case": 5,
              "success_thresholds": "not calibrated; report numerical and geometric diagnostics separately"}
    write_json(output/"manifest.json", manifest(config, "standalone_complete_simulation_benchmark"))
    scene = args.workcell
    if scene is None:
        from argparse import Namespace
        from robotic_cutting_sim.workcell import export
        workcell = new_output(output/"workcell")
        export(Namespace(variant="D0", family="dome", seed=42, spacing=.002, asset_root=None), workcell)
        scene = workcell/"workcell.usda"
    scene = scene.resolve()
    write_json(output/"protocol.json", {**config,
        "scope": "actual articulated physics plus measured surface replay to ideal full-scene RGB-D",
        "observation_contract": "rgb/depth/validity/calibration/time and measured q/qdot/tool_pose; no force/oracle/global",
        "oracle_metrics": "component and particle-surface proxies stored separately",
        "material_recipe": "explicit BenchmarkSpec for all cases; unapplied fields disclosed",
        "two_way_coupling": "20ms explicit staggered wrench feedback",
        "simulation_performance_scope": "includes setup, diagnostics and disk IO; concurrent jobs may affect timings"})
    tick = time.perf_counter()
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            case = new_output(output/f"{variant}_seed{seed}")
            spec = BenchmarkSpec.generate(seed, variant)
            write_json(case/"scenario.json", spec.to_dict())
            row = {"variant": variant, "seed": seed, "status": "FAIL"}
            print(f"Starting {variant} seed={seed}", flush=True)
            try:
                physics = new_output(case/"physics")
                result = simulate(physics, CouplingConfig(device=args.device, variant=variant, benchmark_spec=spec), seed, args.duration)
                if result["status"] == "FAIL":
                    raise RuntimeError(f"Numerical physics guard failed: {result['guard_failure']}")
                receipt = json.loads((physics/"setup.json").read_text())["parameter_receipt"]
                metrics = analyze_recording(physics/"physics_state.npz", MetricConfig(
                    density_kg_m3=spec.material.density_kg_m3, particle_support_radius_m=.001),
                    run_metadata={"wall_s": result["wall_s"]})
                write_json(case/"oracle_metrics.json", metrics)
                surface = case/"surface"
                execute([sys.executable, ROOT/"scripts/export_cutting_replay.py", physics, scene, surface,
                    "--robot-replay", physics/"dual_franka_replay.usda", "--samples", 5,
                    "--voxel-size", .0015, "--device", args.device], case/"surface.log")
                cameras = case/"ideal_rgbd"
                execute([sys.executable, ROOT/"scripts/capture_workcell_rgbd.py",
                    surface/"cutting_workcell_replay.usda", cameras, "--samples", 5,
                    "--device", args.device], case/"camera.log")
                vision = json.loads((cameras/"result.json").read_text())
                if vision["status"] != "PASS":
                    raise RuntimeError("Recorded full-scene ideal RGB-D validation failed")
                row.update({"status": "PASS_EXECUTION", "physics_status": result["status"],
                    "initial_mechanical_components": result["initial_mechanical_components"],
                    "final_mechanical_components": result["final_mechanical_components"],
                    "particle_count": result["particle_count"], "max_speed_m_s": result["maximum_speed_m_s"],
                    "minimum_detF": result["minimum_detF"], "peak_material_force_N": result["maximum_material_force_N"],
                    "mass_budget": metrics["components"], "surface_metrics": metrics["final_surface"],
                    "parameter_application": receipt, "camera_status": vision["status"],
                    "changed_depth_pixels_10um": vision["changed_depth_pixels_10um"],
                    "physics_wall_s": result["wall_s"], "task_success": None,
                    "task_success_scope": "not inferred from a mechanical split or numerical execution"})
            except Exception as error:
                (case/"failure.log").write_text(traceback.format_exc())
                row["error"] = str(error)
                print(f"Case failed: {variant} seed={seed}: {error}", flush=True)
            write_json(case/"result.json", row)
            rows.append(row)
            print(f"Finished {variant} seed={seed}: {row['status']}", flush=True)
    good = all(r["status"] == "PASS_EXECUTION" for r in rows)
    summary = {"status": "PASS_EXECUTION" if good else "FAIL", "scope": "standalone articulated cutting + vision benchmark execution",
        "cases": rows, "wall_s": time.perf_counter()-tick,
        "benchmark_executable": True, "benchmark_ready": False, "formal_task_success_rate": None,
        "full_isaac_lab_runtime": "BLOCKED", "simulated_d455_validated": False,
        "limitations": ["Provisional material and cutting parameters, no physical calibration",
            "Partial recipe application is reported per case; D3 hardening and camera perturbations require adapters",
            "Fixed sweep is a baseline, not an optimal or learned cutting controller",
            "Surface proxies and source mechanical components are oracle metrics, excluded from observations",
            "Ideal full-scene RGB-D is validated separately from native D455 sensor calibration"]}
    write_json(output/"result.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2), flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
