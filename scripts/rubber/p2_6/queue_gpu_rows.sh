#!/usr/bin/env bash
# Queue one P6.3 GPU invocation under the blocking per-GPU lock (one RCS job per GPU; other agents interleave between
# invocations). Usage: scripts/rubber/p2_6/queue_gpu_rows.sh <gpu index> <rows comma list> <out dir>
set -u
GPU="$1"; ROWS="$2"; OUT="$3"
cd "$(dirname "$0")/../../.."
exec flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu${GPU}.lock \
  env CUDA_VISIBLE_DEVICES=${GPU} TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src \
  .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/run_rubber_rl_gpu.py --rows "${ROWS}" --out "${OUT}" \
  --device cuda:0 --gpu-index "${GPU}"
