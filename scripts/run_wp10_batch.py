#!/usr/bin/env python3
"""Execute a declared WP-10 job list; retain every failure and continue the list."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = json.loads(args.plan.read_text())
    (args.out / 'plan.json').write_text(json.dumps(jobs, indent=2) + '\n')
    rows = []
    for job in jobs:
        start = time.monotonic()
        command = [sys.executable, 'scripts/wp10_record.py', '--out',
                   str(args.out / job['id']), '--', *job['command']]
        process = subprocess.run(command, stdout=subprocess.DEVNULL, check=False)
        result_path = Path(job['result'])
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        rows.append(dict(id=job['id'], command=job['command'], returncode=process.returncode,
                         artifact=job['result'], status=result.get('status', 'FAIL_NO_RESULT'),
                         wall_s=time.monotonic() - start))
        (args.out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(rows[-1]), flush=True)
    return int(any(row['returncode'] != 0 or row['status'].startswith('FAIL') for row in rows))


if __name__ == '__main__':
    sys.exit(main())
