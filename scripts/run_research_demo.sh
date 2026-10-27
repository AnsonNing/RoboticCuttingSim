#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_root="${1:-artifacts/research_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$run_root" ]]; then
  echo "Choose a fresh output path: $run_root exists" >&2
  exit 1
fi
mkdir -p "$run_root"
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
.venv/bin/python -m robotic_cutting_sim.cli scene --preset workcell --output-dir "$run_root/workcell"
.venv/bin/python scripts/run_logged.py "$run_root/live_env.log" -- \
  .venv/bin/python -m robotic_cutting_sim.cli env-demo --steps 120 --output-dir "$run_root/live_env"
.venv/bin/python scripts/export_replay.py "$run_root/workcell/workcell.usda" \
  "$run_root/live_env" "$run_root/live_env" "$run_root/research_workcell_replay.usda"
echo "Research workcell replay: $run_root/research_workcell_replay.usda"
