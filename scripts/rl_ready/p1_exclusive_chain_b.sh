#!/bin/bash
# P1 performance table, exclusive window on GPU1, chain B (the rows chain A does not cover).
#
# Waits for chain A ("chain done" in chain_a.log), then holds rcs_rl_runner/gpu1.lock for the whole chain
# (one process per row). Benchmark code runs from the frozen review worktree (integration commit 691b571);
# nothing is edited there. Before and after every row: GPU1 compute apps (with names), three utilisation
# samples and one pmon sample go to chain_b.log; a chain-level sampler (exclusive_gpu_sampler.py, own = this
# shell's process tree) records GPU1 every 2 s for the whole chain. Rows are ordered by priority
# (single-env, workload A s15, CUT N=8, B1 N=4, B2 N=4, B1 N=8 first); the optional rows after them start only
# while the chain is inside its time budget (elapsed + estimate <= BUDGET_MIN), otherwise they are logged NOT_RUN.
#
#   nohup scripts/rl_ready/p1_exclusive_chain_b.sh > <D>/chain_b.nohup 2>&1 &
set -u
R=/home/ningan/RoboticCuttingSim-rl-review
X=/home/ningan/RoboticCuttingSim-rl-excl
D=/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/exclusive_001
LOCK=/mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock
LOG=$D/chain_b.log
BUDGET_MIN=${BUDGET_MIN:-180}
log(){ echo "$(date -Is) $*" | tee -a $LOG; }

if [ "${1:-}" != "--locked" ]; then
  log "chain_b waiting for chain A"
  until grep -q "chain done" $D/chain_a.log; do sleep 2; done
  log "chain A done; taking $LOCK (blocking)"
  exec flock $LOCK "$0" --locked
fi

T0=$(date +%s)
cd $R
HEAD=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | wc -l)
log "chain_b start (lock held) head=$HEAD tracked_changes=$DIRTY budget_min=$BUDGET_MIN pid=$$"
PYTHONPATH=$R/src $X/.venv/bin/python $X/scripts/rl_ready/exclusive_gpu_sampler.py --gpu-index 1 --root-pid $$ \
  --stop-file $D/chain_b.stop --out $D/sampler_chain_b.jsonl > $D/sampler_chain_b.stderr 2>&1 &
SAMPLER=$!

snap(){  # $1 = label
  local apps util pm
  apps=$(nvidia-smi --id=1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')
  util=""
  for i in 1 2 3; do
    util="$util$(nvidia-smi --id=1 --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits);"
    sleep 1
  done
  pm=$(nvidia-smi pmon -i 1 -c 1 -s u | grep -v '^#' | awk '{print $2":"$4}' | tr '\n' ' ')
  log "$1 compute_apps=[$apps] util_mem=[$util] pmon_pid_sm=[$pm]"
}

free_ok(){  # free MiB >= 2560 (row estimate) + 4096 (left for others)
  local used total
  read used total < <(nvidia-smi --id=1 --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr ',' ' ')
  [ $((total - used)) -ge 6656 ]
}

optional_ok(){  # $1 = estimate in minutes
  local el=$(( ($(date +%s) - T0) / 60 ))
  [ $((el + $1)) -le $BUDGET_MIN ]
}


row(){  # $1 = name, $2 = priority|optional:<min>, rest = command (run in $R)
  local name=$1 kind=$2; shift 2
  if [[ $kind == optional:* ]] && ! optional_ok ${kind#optional:}; then
    log "skip $name NOT_RUN (budget: elapsed $(( ($(date +%s) - T0) / 60 )) min + estimate ${kind#optional:} > $BUDGET_MIN)"
    return
  fi
  local waited=0
  until free_ok; do
    if [ $waited -ge 600 ]; then log "skip $name BLOCKED (GPU1 free memory below 6656 MiB for 10 min)"; return; fi
    sleep 10; waited=$((waited + 10))
  done
  snap "before $name"
  log "start $name cmd=[$*]"
  local t=$(date +%s)
  env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/data/ningan/tmp \
    timeout 7200 "$@" > $D/$name.log 2>&1
  local rc=$?
  log "end $name rc=$rc wall_s=$(( $(date +%s) - t ))"
  snap "after $name"
}

moverun(){  # $1 = source dir under artifacts/rl_ready (frozen runners pick <name>_NNN there), $2 = row name
  local src
  src=$(ls -d $R/artifacts/rl_ready/$1/excl001_$2_[0-9][0-9][0-9] 2>/dev/null | tail -1)
  if [ -n "$src" ] && [ ! -e $D/$(basename $src) ]; then
    mv -n "$src" $D/ && log "moved $src -> $D/$(basename $src)"
  else
    log "move $2: no run dir found or target exists ($src)"
  fi
}

SIM=.deps/isaacsim-venv/bin/python
LAB=.deps/isaaclab-venv/bin/python

# 1. single_env_headless: frozen single LiveCuttingMaterial replay of the WP-17 recordings (P1-0 command)
for s in 15 512; do
  row single_s$s priority $SIM scripts/rl_ready/run_baseline.py --mode replay --seed $s --gpu-index 1 --warm-resets 5 \
    --run-name excl001_single_s$s
  moverun p1_0 single_s$s
done

# 2. batched workload A, seed 15, N=8, one process per variant (P1-6 R0 harness, as chain A)
for v in default alias_free_v1; do
  row a_n8_s15_$v priority $SIM scripts/rl_ready/run_perf_acceptance.py --variant-json $D/variant_$v.json \
    --mode throughput --num-envs 8 --seeds 15 --warmup-steps 20 --steps 100 --device cuda:0 --gpu-index 1 \
    --out $D/a_n8_s15_$v
done

# 3. batched CUT N=8 (P1-5 m2_cut_n8 command: 2 episodes)
row cut_n8 priority $SIM scripts/rl_ready/run_cut_batched.py --num-envs 8 --episodes 2 --gpu-index 1 --run-name excl001_cut_n8
moverun p1_5 cut_n8

# 4. workload B (Isaac Lab Newton robot + material, S3 recorded commands, frozen evaluator per env)
B="scripts/rl_ready/probe_lab_batched_gpu.py --probe g2 --material tlmpm_v7_2 --evaluate --driver recorded --scene-seeds 512 15 --steps 1200 --gpu-index 1"
row b1_n4 priority $LAB $B --num-envs 4 --out $D/b1_n4
row b2_n4 priority $LAB $B --num-envs 4 --tool-mount physical_grasp_v1 --grip-profile mjwarp_grip_v1 --out $D/b2_n4
row b1_n8 priority $LAB $B --num-envs 8 --out $D/b1_n8

# 5. optional rows (budget gated)
row cut_n1 optional:8 $SIM scripts/rl_ready/run_cut_batched.py --num-envs 1 --gpu-index 1 --run-name excl001_cut_n1
moverun p1_5 cut_n1
row b1_n1 optional:10 $LAB $B --num-envs 1 --out $D/b1_n1
row cut_n16 optional:35 $SIM scripts/rl_ready/run_cut_batched.py --num-envs 16 --gpu-index 1 --run-name excl001_cut_n16
moverun p1_5 cut_n16
row b1_n16 optional:70 $LAB $B --num-envs 16 --out $D/b1_n16

touch $D/chain_b.stop
wait $SAMPLER
log "chain_b done elapsed_min=$(( ($(date +%s) - T0) / 60 ))"
