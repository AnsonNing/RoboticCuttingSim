#!/bin/bash
# Formal workload A throughput rows in an EXCLUSIVE_GPU window on GPU1 (held under rcs_rl_runner/gpu1.lock).
# One process per (N, variant); alternating variants; before every row the foreign compute on GPU1 is logged.
# A row counts as EXCLUSIVE_GPU only if no foreign compute process is on GPU1 during it (the harness samples
# co-tenants itself; the classification is done afterwards from those samples).
set -u
R=/home/ningan/RoboticCuttingSim-rl-review
D=/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/exclusive_001
cd $R
HEAD=$(git rev-parse HEAD)
echo '{}' > $D/variant_default.json
echo '{"particle_contact_mode": "alias_free_v1"}' > $D/variant_alias_free_v1.json
log(){ echo "$(date -Is) $*" | tee -a $D/chain_a.log; }
log "chain start head=$HEAD"
for n in 1 4 8 16; do
  if [ $n -eq 1 ]; then steps=300; else steps=100; fi
  for v in default alias_free_v1; do
    name=a_n${n}_${v}
    fg=$(nvidia-smi --id=1 --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
    log "start $name foreign_before=[$fg]"
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/data/ningan/tmp \
      timeout 3600 .deps/isaacsim-venv/bin/python scripts/rl_ready/run_perf_acceptance.py \
      --variant-json $D/variant_${v}.json --mode throughput --num-envs $n --seeds 512 \
      --warmup-steps 20 --steps $steps --device cuda:0 --gpu-index 1 --out $D/$name > $D/$name.log 2>&1
    log "end $name rc=$?"
  done
done
log "chain done"
