#!/usr/bin/env python3
"""Run the declared E1 rows (configs/wp12/phase4_e1_protocol_v1.json) with at most N concurrent jobs.

Each row is one ``scripts/run_wp12_phase4_e1.py`` process under GNU time, logged to
``artifacts/wp12/logs/<out name>_<row>.log``. Rows whose output directory exists are skipped, so nothing is
overwritten. Lanes are the given devices. Before each launch the lane's free memory is read with nvidia-smi; a
lane with less than ``--min-free-mib`` waits. Other users' processes are never touched. ``queue.jsonl`` in the
output directory records each job's command, device, start/end time and the subprocess exit code.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4_e1_queue.py \\
      --out artifacts/wp12/phase4_e1_001 --devices cuda:0 cuda:1
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_wp12_phase4_e1 import PROTOCOL, protocol_rows  # noqa: E402


def free_mib(device):
    index = device.split(':')[1]
    out = subprocess.run(['nvidia-smi', f'--id={index}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    p.add_argument('--rows', nargs='*')
    p.add_argument('--max-jobs', type=int, default=2)
    p.add_argument('--min-free-mib', type=int, default=3000)
    a = p.parse_args()
    if a.max_jobs > 2:
        p.error('the protocol allows at most 2 concurrent E1 jobs')
    rows = protocol_rows(PROTOCOL)
    names = a.rows or list(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    logs = ROOT/'artifacts/wp12/logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'TMPDIR': os.environ.get('TMPDIR', '/mnt/data/ningan/tmp')}
    pending = [n for n in names if not (a.out/n).exists()]
    for n in names:
        if n not in pending:
            print(f'{n}: exists, skipped (never overwritten)', flush=True)
    running = {}   # device -> (name, Popen, record)
    lanes = list(a.devices)[:a.max_jobs] if len(a.devices) >= a.max_jobs else list(a.devices)
    with (a.out/'queue.jsonl').open('a') as journal:
        while pending or running:
            for device, (name, proc, record) in list(running.items()):
                if proc.poll() is not None:
                    record.update(end=time.strftime('%Y-%m-%d %H:%M:%S'), returncode=proc.returncode)
                    journal.write(json.dumps(record)+'\n')
                    journal.flush()
                    print(f'{name}: exit {proc.returncode} on {device}', flush=True)
                    del running[device]
            slots = [d for d in lanes if d not in running]
            if len(running) >= a.max_jobs:
                slots = []
            for device in slots:
                if not pending:
                    break
                if free_mib(device) < a.min_free_mib:
                    continue
                name = pending.pop(0)
                log = logs/f'{a.out.name}_{name}.log'
                command = ['/usr/bin/time', '-v', str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run_wp12_phase4_e1.py'),
                           '--row', name, '--device', device, '--out', str(a.out/name)]
                with log.open('w') as stream:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                running[device] = (name, proc, {'row': name, 'device': device, 'command': command, 'log': str(log),
                                                'start': time.strftime('%Y-%m-%d %H:%M:%S')})
                print(f'{name}: started on {device}', flush=True)
                time.sleep(5)
            time.sleep(10)


if __name__ == '__main__':
    main()
