#!/usr/bin/env python3
"""Durable, sequential WP-13 cohort; never reruns an existing experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=a.resume)
    protocol = ROOT/'configs/wp13/focused_v1.json'
    doc = json.loads(protocol.read_text())
    # Protocol plus physical source must be committed before formal outcomes.
    protected = [str(protocol.relative_to(ROOT)), *doc['rows'], 'src/robotic_cutting_sim/tlmpm_v7.py']
    subprocess.run(['git', 'ls-files', '--error-unmatch', *protected], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *protected], cwd=ROOT, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # The decisive basic CUT/PARTIAL/TRIM first; only one final cohort.
    first = ['C00_cut', 'P00_partial', 'T00_trim']
    rows = sorted(doc['rows'], key=lambda r: (first.index(Path(r).stem) if Path(r).stem in first else 3,
                                             doc['rows'].index(r)))
    with (a.root/'ledger.jsonl').open('a') as ledger:
        for case in rows:
            for solver in doc['arms']:
                name = Path(case).stem
                out = a.root/solver/name
                if out.exists():
                    print(f'PRESERVED {out}', flush=True)
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                log = a.root/f'{solver}_{name}.log'
                command = [sys.executable, 'scripts/run_wp13.py', '--case', case, '--solver', solver,
                           '--device', a.device, '--out', str(out), '--label', 'WP13_FOCUSED_FORMAL']
                record = {'case': name, 'solver': solver, 'command': command, 'cwd': str(ROOT),
                          'commit': commit, 'started': now(), 'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest()}
                start = time.monotonic()
                with log.open('x') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT/'src'),
                                             'PYTHONUNBUFFERED': '1'}, stdout=stream, stderr=subprocess.STDOUT)
                    record['pid'] = child.pid
                    ledger.write(json.dumps({**record, 'state': 'STARTED'})+'\n'); ledger.flush(); os.fsync(ledger.fileno())
                    code = child.wait()
                record.update(state='FINISHED', returncode=code, ended=now(), wall_s=time.monotonic()-start)
                summary = out/'summary.json'
                if summary.exists():
                    result = json.loads(summary.read_text())
                    record.update(outcome=result['class'], components=result['metrics']['mechanical_components'])
                ledger.write(json.dumps(record)+'\n'); ledger.flush(); os.fsync(ledger.fileno())
                print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
