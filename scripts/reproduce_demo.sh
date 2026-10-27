#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_root="${1:-artifacts/demo_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$run_root" ]]; then
  echo "Choose a fresh output path: $run_root exists" >&2
  exit 1
fi
mkdir -p "$run_root"
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
# Choose a free GPU explicitly before launch: CUDA_VISIBLE_DEVICES=1 bash ...
.venv/bin/python -m robotic_cutting_sim.cli scene --output-dir "$run_root/scene"
.venv/bin/python scripts/run_logged.py "$run_root/indent.log" -- \
  .venv/bin/python -m robotic_cutting_sim.cli physics-probe \
  --case indent_unload --output-dir "$run_root/indent"
.venv/bin/python scripts/run_logged.py "$run_root/camera.log" -- \
  .venv/bin/python -m robotic_cutting_sim.cli camera-check \
  --input "$run_root/indent" --samples 9 --output-dir "$run_root/camera"
.venv/bin/python scripts/export_replay.py "$run_root/scene/physics_testbed.usda" \
  "$run_root/indent" "$run_root/camera" "$run_root/recorded_deformation.usda"
echo "Recorded deformation scene: $run_root/recorded_deformation.usda"
