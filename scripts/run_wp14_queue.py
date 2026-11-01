#!/usr/bin/env python3
"""Sequential, append-only compiled-planner paired experiments, with frozen inputs."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text())
    protected = [str(a.protocol), *protocol['rows'], 'scripts/run_wp13.py',
                 'scripts/run_wp14_queue.py', 'src/robotic_cutting_sim/tlmpm_v7.py']
    subprocess.run(['git', 'ls-files', '--error-unmatch', *protected], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *protected], check=True)
    a.root.mkdir(parents=True, exist_ok=a.resume)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    with (a.root/'ledger.jsonl').open('a') as ledger:
        def record(value):
            value['timestamp'] = datetime.now(timezone.utc).isoformat()
            ledger.write(json.dumps(value)+'\n'); ledger.flush(); os.fsync(ledger.fileno())
            print(json.dumps(value), flush=True)
        for case in protocol['rows']:
            for solver in protocol['arms']:
                name = Path(case).stem
                out = a.root/solver/name
                if out.exists():
                    print(f'PRESERVED {out}', flush=True)
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, 'scripts/run_wp13.py', '--case', case, '--solver', solver,
                           '--planner-source', 'compiled', '--device', protocol['device'],
                           '--out', str(out), '--label', protocol['id']]
                base = dict(case=case, solver=solver, command=command, commit=commit,
                            protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest())
                start = time.monotonic()
                with (a.root/f'{solver}_{name}.log').open('x') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env={**os.environ,
                        'PYTHONPATH': str(ROOT/'src'), 'PYTHONUNBUFFERED': '1'}, stdout=log, stderr=subprocess.STDOUT)
                    record({**base, 'state': 'STARTED', 'pid': child.pid})
                    code = child.wait()
                result = {'returncode': code, 'wall_s': time.monotonic()-start}
                if (out/'summary.json').exists():
                    summary = json.loads((out/'summary.json').read_text())
                    result.update(outcome=summary['class'], components=summary['metrics']['mechanical_components'])
                record({**base, **result, 'state': 'FINISHED'})


if __name__ == '__main__':
    main()
