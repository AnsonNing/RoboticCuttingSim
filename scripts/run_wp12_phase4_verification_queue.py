#!/usr/bin/env python3
"""Queue for configs/wp12/phase4_verification_protocol_v1.json: every declared row in order, one process per row.

Lanes are device slots (default cuda:0, cuda:0, cuda:1, cuda:1: at most two concurrent rows per GPU). A row whose
output directory exists is never rerun (reported SKIPPED_EXISTS). Once the summed wall of finished rows plus the
estimates of running rows reaches the protocol's ``budget.cap_gpu_h``, no new row starts and the rest are NOT_RUN.
Each row runs under GNU time; logs go to artifacts/wp12/logs/<run>_<row>.log, one JSON line per row to
<out>/queue.jsonl. Other users' processes are never touched.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4_verification_queue.py \\
      --out artifacts/wp12/phase4_verify_001
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs/wp12/phase4_verification_protocol_v1.json'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lanes', nargs='+', default=['cuda:0', 'cuda:0', 'cuda:1', 'cuda:1'])
    p.add_argument('--rows', nargs='*', help='subset (declared ids, protocol order kept)')
    a = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    rows = [r for r in protocol['rows'] if not a.rows or r['id'] in a.rows]
    cap_s = float(protocol['budget']['cap_gpu_h'])*3600
    out = a.out if a.out.is_absolute() else ROOT/a.out
    out.mkdir(parents=True, exist_ok=True)
    logs = ROOT/'artifacts/wp12/logs'
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    state = {'next': 0, 'done_wall': 0., 'running': {}, 'stopped': False}
    env = {**os.environ, 'TMPDIR': os.environ.get('TMPDIR', '/mnt/data/ningan/tmp')}

    def record(entry):
        with lock, (out/'queue.jsonl').open('a') as f:
            f.write(json.dumps(entry)+'\n')

    def worker(device):
        while True:
            with lock:
                if state['next'] >= len(rows):
                    return
                row = rows[state['next']]
                state['next'] += 1
                committed = state['done_wall']+sum(state['running'].values())
                if state['stopped'] or committed >= cap_s:
                    state['stopped'] = True
                    skip = 'NOT_RUN_BUDGET_CAP'
                elif (out/row['id']).exists():
                    skip = 'SKIPPED_EXISTS'
                else:
                    skip = None
                    state['running'][row['id']] = float(row.get('estimated_wall_s') or 0)
            if skip:
                record({'row': row['id'], 'status': skip, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
                continue
            log = logs/f'{out.name}_{row["id"]}.log'
            cmd = ['/usr/bin/time', '-v', str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run_wp12_phase4_verification.py'),
                   '--row', row['id'], '--device', device, '--out', str(out/row['id'])]
            start, t0 = time.strftime('%Y-%m-%d %H:%M:%S'), time.monotonic()
            with log.open('w') as f:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
            wall = time.monotonic()-t0
            with lock:
                state['running'].pop(row['id'], None)
                state['done_wall'] += wall
            record({'row': row['id'], 'device': device, 'command': cmd, 'log': str(log), 'start': start,
                    'end': time.strftime('%Y-%m-%d %H:%M:%S'), 'wall_s': wall, 'returncode': rc})

    threads = [threading.Thread(target=worker, args=(d,)) for d in a.lanes]
    for i, t in enumerate(threads):
        t.start()
        time.sleep(5)   # stagger CUDA context creation
    for t in threads:
        t.join()
    print('done; summed wall', round(state['done_wall']/3600, 2), 'h; stopped by cap' if state['stopped'] else '',
          file=sys.stderr)


if __name__ == '__main__':
    main()
