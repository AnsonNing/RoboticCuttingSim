#!/usr/bin/env python3
"""Prepare D0–D3 initial states or compare explicitly supplied recorded runs.

This standalone scaffold neither launches a solver nor marks a formal cutting
benchmark complete. Outputs refuse to overwrite prior evidence.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from robotic_cutting_sim.benchmark import (
    BenchmarkSpec, MetricConfig, compare_recordings, sample_specimen,
)
from robotic_cutting_sim.recording import manifest, write_json


def labeled_paths(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Use LABEL=PATH for every recorded run and metadata file")
        label, path = value.split("=", 1)
        if not label or label in result:
            raise ValueError("Run labels must be nonempty and unique")
        result[label] = Path(path)
    return result


def prepare(args, output):
    entries = []
    for variant in args.variants:
        for seed in args.seeds:
            spec = BenchmarkSpec.generate(seed, variant)
            q, mass, lattice, receipt = sample_specimen(spec, args.spacing_m)
            case = output/f"{variant}_seed{seed}"
            case.mkdir()
            np.savez_compressed(case/"initial_particles.npz", positions_m=q, mass_kg=mass,
                                lattice_indices=lattice)
            write_json(case/"scenario.json", spec.to_dict())
            write_json(case/"sampling_receipt.json", receipt)
            entries.append({"case": case.name, **receipt})
    result = {"status": "PASS", "scope": "initial_state_generation_only",
              "simulation_status": "NOT_RUN", "benchmark_ready": False,
              "parameter_application_status": "RECIPE_ONLY; a backend must declare supported/applied parameters",
              "entries": entries}
    write_json(output/"result.json", result)
    return result


def compare(args, output):
    paths = labeled_paths(args.run)
    for label, path in paths.items():
        if path.is_dir():
            paths[label] = path/"physics_state.npz"
    metadata_paths = labeled_paths(args.metadata)
    if set(metadata_paths)-set(paths):
        raise ValueError("Metadata labels must match supplied recorded runs")
    metadata = {label: json.loads(path.read_text()) for label, path in metadata_paths.items()}
    config = MetricConfig(args.length_m, args.width_m, args.target_height_m,
                          args.density_kg_m3, args.surface_cell_size_m,
                          args.particle_support_radius_m)
    result = compare_recordings(paths, config, metadata)
    write_json(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scene = commands.add_parser("prepare", help="Generate continuous initial material states only")
    scene.add_argument("--variants", nargs="+", choices=["D0", "D1", "D2", "D3"],
                       default=["D0", "D1", "D2", "D3"])
    scene.add_argument("--seeds", nargs="+", type=int, default=[42])
    scene.add_argument("--spacing-m", type=float, default=.002)
    scene.add_argument("--output-dir", type=Path, required=True)
    report = commands.add_parser("compare", help="Offline diagnostics from explicit LABEL=NPZ inputs")
    report.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    report.add_argument("--metadata", action="append", default=[], metavar="LABEL=JSON")
    report.add_argument("--length-m", type=float, default=.120)
    report.add_argument("--width-m", type=float, default=.080)
    report.add_argument("--target-height-m", type=float, default=.015)
    report.add_argument("--density-kg-m3", type=float, default=1500.)
    report.add_argument("--surface-cell-size-m", type=float, default=.004)
    report.add_argument("--particle-support-radius-m", type=float, default=0.)
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {key: value for key, value in vars(args).items() if key != "output_dir"}
    write_json(output/"manifest.json", manifest(config, "benchmark_foundation_offline"))
    try:
        result = prepare(args, output) if args.command == "prepare" else compare(args, output)
    except (ValueError, KeyError, OSError) as error:
        write_json(output/"result.json", {"status": "FAIL", "scope": "benchmark_foundation",
                                         "benchmark_ready": False, "error": str(error)})
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "benchmark_ready": False, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
