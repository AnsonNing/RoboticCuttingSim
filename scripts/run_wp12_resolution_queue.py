#!/usr/bin/env python3
"""Queue for the WP-12 resolution study (configs/wp12/resolution_protocol_v1.json): rows in the declared order.

At most ``--jobs`` (default 2) rows of this study at once on one device; rows marked ``exclusive`` run alone. Every
row writes ``<out>/<row_id>/`` (the runner refuses an existing directory) and ``artifacts/wp12/logs/<out name>_<row_id>.log``
with GNU time. Rows whose directory exists are skipped (never overwritten). Declared failure rule: a row that ends with
SlotOverflowError is queued once more as ``<row_id>_slots16``. Budget: no new row starts once the summed row wall
time exceeds ``--gpu-hours`` or the queue wall time exceeds ``--wall-hours``; such rows are recorded NOT_RUN (budget).
The queue log ``<out>/queue.jsonl`` records start/end, GNU time exit status and wall, the rows running alongside and an
nvidia-smi snapshot. It never stops a process it did not start.

  TMPDIR=/mnt/data/ningan/tmp nohup .venv/bin/python scripts/run_wp12_resolution_queue.py \\
      --out artifacts/wp12/resolution_002 --device cuda:1 > artifacts/wp12/logs/resolution_002_queue.log 2>&1 &
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs/wp12/resolution_protocol_v1.json'


def smi():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except Exception as error:
        return [repr(error)]


def gnu_time(log):
    text = Path(log).read_text(errors='replace') if Path(log).exists() else ''
    status = re.findall(r'Exit status: (\d+)', text)
    wall = re.findall(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', text)
    rss = re.findall(r'Maximum resident set size \(kbytes\): (\d+)', text)
    seconds = None
    if wall:
        parts = [float(v) for v in wall[-1].split(':')]
        seconds = sum(v*60**i for i, v in enumerate(reversed(parts)))
    return {'exit_status': int(status[-1]) if status else None, 'wall_s': seconds,
            'max_rss_kb': int(rss[-1]) if rss else None, 'slot_overflow': 'SlotOverflowError' in text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--gpu-hours', type=float, default=8.)
    ap.add_argument('--wall-hours', type=float, default=6.)
    ap.add_argument('--rows', nargs='*', help='subset of declared row ids (default: all, declared order)')
    a = ap.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    order = list(protocol['rows']) if not a.rows else a.rows
    a.out.mkdir(parents=True, exist_ok=True)
    logs = ROOT/'artifacts/wp12/logs'
    logs.mkdir(parents=True, exist_ok=True)
    qlog = a.out/'queue.jsonl'
    env = {**os.environ, 'TMPDIR': os.environ.get('TMPDIR', '/mnt/data/ningan/tmp')}
    pending = [r for r in order]
    running = {}
    used_s, start = 0., time.monotonic()

    def write(rec):
        with qlog.open('a') as f:
            f.write(json.dumps(rec)+'\n')

    while pending or running:
        for rid, (proc, t0, log, alongside) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[rid]
            info = gnu_time(log)
            used_s += time.monotonic()-t0
            write({'event': 'end', 'row': rid, 'time': time.time(), 'returncode': proc.returncode, **info,
                   'queue_wall_s': time.monotonic()-t0, 'alongside': alongside})
            print(f'{rid}: end rc={proc.returncode} exit={info["exit_status"]} wall={info["wall_s"]}', flush=True)
            if info['slot_overflow'] and not rid.endswith('_slots16'):
                pending.insert(0, rid+'_slots16')
        if pending:
            rid = pending[0]
            spec = protocol['rows'].get(rid.removesuffix('_slots16'), {})
            exclusive = bool(spec.get('exclusive'))
            if (a.out/rid).exists():
                pending.pop(0)
                write({'event': 'skip_exists', 'row': rid, 'time': time.time()})
                continue
            over = used_s > a.gpu_hours*3600 or time.monotonic()-start > a.wall_hours*3600
            if over:
                pending.pop(0)
                write({'event': 'NOT_RUN', 'row': rid, 'reason': 'budget', 'time': time.time(), 'used_row_s': used_s})
                continue
            can = (not running) if exclusive else (len(running) < a.jobs and not any(
                protocol['rows'].get(r.removesuffix('_slots16'), {}).get('exclusive') for r in running))
            if can:
                pending.pop(0)
                log = logs/f'{a.out.name}_{rid}.log'
                cmd = ['/usr/bin/time', '-v', str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run_wp12_resolution.py'),
                       '--row', rid, '--device', a.device, '--out', str(a.out/rid)]
                with open(log, 'w') as f:
                    f.write('$ '+' '.join(cmd)+'\n')
                    f.flush()
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
                alongside = list(running)
                running[rid] = (proc, time.monotonic(), log, alongside)
                write({'event': 'start', 'row': rid, 'time': time.time(), 'command': cmd, 'alongside': alongside,
                       'nvidia_smi': smi(), 'used_row_s': used_s})
                print(f'{rid}: start (alongside {alongside})', flush=True)
                continue
        time.sleep(2)
    write({'event': 'queue_done', 'time': time.time(), 'used_row_s': used_s, 'queue_wall_s': time.monotonic()-start})


if __name__ == '__main__':
    sys.exit(main())
