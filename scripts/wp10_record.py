#!/usr/bin/env python3
"""Record one command in a fresh directory, including failures and source identity."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply a command after --')
    args.out.mkdir(parents=True, exist_ok=False)
    patch = subprocess.check_output(['git', 'diff', 'HEAD', '--binary'])
    (args.out / 'source.patch').write_bytes(patch)
    source_hashes = {}
    for directory in ('src', 'configs', 'scripts'):
        for source in sorted(Path(directory).rglob('*')):
            if source.is_file() and source.suffix in ('.py', '.json', '.yaml', '.yml', '.kit'):
                destination = args.out / 'source' / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt = dict(command=command, cwd=str(Path.cwd()), seed=args.seed,
                   commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   source_patch_sha256=hashlib.sha256(patch).hexdigest(),
                   source_snapshot_sha256=source_hashes,
                   untracked=subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], text=True).splitlines(),
                   started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH')},
                   status='RUNNING')
    path = args.out / 'command.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    start = time.monotonic()
    try:
        with (args.out / 'output.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        receipt.update(returncode=result.returncode, status='PASS' if result.returncode == 0 else 'FAIL')
    except BaseException as error:
        receipt.update(returncode=1, status='FAIL', exception=repr(error))
        raise
    finally:
        receipt.update(wall_s=time.monotonic() - start,
                       finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return receipt['returncode']


if __name__ == '__main__':
    sys.exit(main())
