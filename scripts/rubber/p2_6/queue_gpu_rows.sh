#!/usr/bin/env bash
# Queue one P6.3 GPU invocation under the blocking per-GPU lock (one RCS job per GPU; other agents interleave between
# invocations). OPENBLAS_NUM_THREADS defaults to 1: the host-staged F8 inverse of a reset (np.linalg.inv of
# 3 blocks of ~360 nodes) took up to ~6 s with the default all-core OpenBLAS pool on a loaded host (consistent with the RL-06 reset p95 of 5.0 s, not verified on that run)
# and 0.06 s with one thread; its bits depend on the thread count but single and batched solvers of one process agree.
# Usage: scripts/rubber/p2_6/queue_gpu_rows.sh <gpu index> <rows comma list> <out dir>
set -u
GPU="$1"; ROWS="$2"; OUT="$3"
cd "$(dirname "$0")/../../.."
exec flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu${GPU}.lock \
  env CUDA_VISIBLE_DEVICES=${GPU} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src \
  .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/run_rubber_rl_gpu.py --rows "${ROWS}" --out "${OUT}" \
  --device cuda:0 --gpu-index "${GPU}"
