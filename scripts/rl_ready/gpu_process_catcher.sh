#!/bin/bash
# Identify short-lived processes on one GPU (P1 exclusive window): every ~1 s list the GPU's processes
# (`nvidia-smi -q -d PIDS`, compute and graphics, with names) and log every PID other than the display server
# together with its `ps` line (owner, parent, command) while it is still alive. Read-only; stops at the stop file.
#   gpu_process_catcher.sh <gpu index> <out log> <stop file>
GPU=$1; OUT=$2; STOP=$3
while [ ! -e "$STOP" ]; do
  nvidia-smi -q -d PIDS -i $GPU | awk '/Process ID/{pid=$NF} /Type/{t=$NF} /Name/{$1=$1; n=$0} /Used GPU Memory/{print pid, t, $(NF-1), n}' |
  while read pid t mem name; do
    [ "$pid" = "4060" ] && continue
    echo "$(date +%s.%N) pid=$pid type=$t mem_mib=$mem $name | ps: $(ps -o user=,ppid=,etimes=,args= -p $pid 2>/dev/null | cut -c1-200)" >> "$OUT"
  done
  sleep 1
done
