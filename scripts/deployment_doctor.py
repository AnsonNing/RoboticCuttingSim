#!/usr/bin/env python3
"""Inspect a portable checkout without importing/booting Isaac or changing it."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from robotic_cutting_sim.deployment_doctor import ROOT, collect_report, write_new_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="auto", help="Physical nvidia-smi index or auto; no CUDA masking")
    parser.add_argument("--runtime", choices=("isaac_sim", "isaac_lab", "both"), default="both")
    parser.add_argument("--base-python", type=Path)
    parser.add_argument("--sim-python", type=Path)
    parser.add_argument("--lab-python", type=Path)
    parser.add_argument("--asset-cache", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must be new; existing evidence is preserved")
    report = collect_report(root=ROOT, gpu=args.gpu, runtime=args.runtime,
        pythons={k: v for k, v in {"base": args.base_python, "isaac_sim": args.sim_python, "isaac_lab": args.lab_python}.items() if v},
        asset_cache=args.asset_cache)
    write_new_report(args.output_dir, report)
    print(report["status"], "GPU", report["gpu_selection"].get("physical_index"), args.output_dir/"result.json")
    return 2 if report["status"].startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
