#!/usr/bin/env python3
"""Chain-level GPU co-tenant sampler for the exclusive-window performance rows (P1 performance table).

Every ``--interval`` seconds one ``robotic_cutting_sim.rl.telemetry.take_sample`` of GPU ``--gpu-index`` is
appended to ``--out`` (JSON lines). "Own" processes are the process trees of the ``--root-pid`` processes (the
benchmark chain shell), so every PID outside the chain is foreign. Each line also carries the GPU's compute-app
list with process names (``nvidia-smi --query-compute-apps=pid,process_name,used_memory``) so a foreign process can
be identified afterwards. Stops when every root PID has exited or ``--stop-file`` exists.

    PYTHONPATH=<frozen checkout>/src .venv/bin/python scripts/rl_ready/exclusive_gpu_sampler.py \
        --gpu-index 1 --root-pid <chain pid> --out <dir>/sampler.jsonl
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from robotic_cutting_sim.rl.telemetry import process_tree, take_sample


def compute_apps_named(gpu_uuid):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[1] == gpu_uuid:
            rows.append(dict(pid=int(parts[0]), name=os.path.basename(parts[2]), memory_mib=parts[3]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--root-pid", type=int, nargs="+", required=True)
    p.add_argument("--interval", type=float, default=2.)
    p.add_argument("--stop-file", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    with out.open("x") as stream:
        while True:
            alive = [pid for pid in args.root_pid if Path(f"/proc/{pid}").exists()]
            if not alive or (args.stop_file and Path(args.stop_file).exists()):
                break
            t0 = time.monotonic()
            own = set()
            for pid in alive:
                own |= set(process_tree(pid))
            sample = take_sample(args.gpu_index, own)
            if sample.get("ok"):
                sample["compute_apps_named"] = compute_apps_named(sample["gpu"]["uuid"])
            sample["own_pids"] = sorted(own)
            stream.write(json.dumps(sample, default=str) + "\n")
            stream.flush()
            time.sleep(max(0., args.interval - (time.monotonic() - t0)))


if __name__ == "__main__":
    main()
