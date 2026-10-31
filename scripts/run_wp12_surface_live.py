#!/usr/bin/env python3
"""WP-12 Phase 3 Layer 6: live Native episodes with the opt-in surface_profile (configs/wp12/surface_live_v1.json).

Inputs are prepared exactly as Phase 2b (``scripts/run_wp12_workcell_live.py prepare`` from
``configs/wp12/workcell_live_v1.json``: same scene, goal, planner overrides, duration and history stride), then the
unchanged runner ``scripts/run_wp10_live.py`` is launched with the Phase 2b flags plus ``--surface-profile`` and
``--surface-timing-log``. Outputs are new directories; nothing is ever overwritten. Nothing here changes physics,
evaluators or thresholds.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_surface_live.py launch --episode B_CUT \\
      --out artifacts/wp12/surface_live_001
"""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs/wp12/surface_live_v1.json'


def workcell_module():
    spec = importlib.util.spec_from_file_location('rcs_run_wp12_workcell_live', ROOT/'scripts/run_wp12_workcell_live.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode_spec(protocol, episode):
    for spec in protocol['episodes']:
        if spec['id'] == episode:
            return spec
    raise KeyError(f'{episode} is not a declared episode of {PROTOCOL.name}')


def cmd_launch(args):
    wc = workcell_module()
    protocol = json.loads(PROTOCOL.read_text())
    spec = episode_spec(protocol, args.episode)
    base = wc.load_protocol()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT/out
    out.mkdir(parents=True, exist_ok=False)
    prep = wc.prepare(base, spec['phase2b_episode'], out/'inputs')
    record = ROOT/spec['phase2b_record']/'inputs'/'prepare.json'
    recorded_id = json.loads(record.read_text())['live_task_id'] if record.exists() else None
    if recorded_id is not None and recorded_id != prep['live_task_id']:
        wc.write_new(out/'attempt.json', {'status': 'NOT_RUN', 'reason': f'task id {prep["live_task_id"]} differs from '
                                          f'the Phase 2b record {recorded_id}'})
        return 3
    free = shutil.disk_usage(out).free/1e9
    if free < args.disk_floor_gb:
        wc.write_new(out/'attempt.json', {'status': 'NOT_RUN', 'reason': f'disk {free:.1f} GB < floor {args.disk_floor_gb} GB'})
        return 3
    run_dir = out/'run'
    command = wc.runner_command(base, prep, (out/'inputs').relative_to(ROOT), run_dir.relative_to(ROOT))
    command += ['--surface-profile', spec['surface_profile'], '--surface-timing-log']
    env = {**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'PYTHONUNBUFFERED': '1', 'TMPDIR': base['runtime']['tmpdir']}
    log = ROOT/'artifacts/wp12/logs'/f'{out.name}_{args.episode}.log'
    if log.exists():
        raise FileExistsError(log)
    attempt = {'episode': args.episode, 'spec': spec, 'protocol': str(PROTOCOL.relative_to(ROOT)),
               'protocol_sha256': wc.sha256(PROTOCOL), 'phase2b_task_id': recorded_id, 'live_task_id': prep['live_task_id'],
               'command': command, 'cwd': str(ROOT),
               'env': {k: env[k] for k in ('OMNI_KIT_ACCEPT_EULA', 'PYTHONUNBUFFERED', 'TMPDIR')}, 'log': str(log),
               'started': datetime.now().astimezone().isoformat(), 'gpu_before': wc.gpu_snapshot(),
               'disk_free_gb_before': free, **wc.git_state()}
    print(json.dumps({k: attempt[k] for k in ('episode', 'live_task_id', 'log', 'started')}), flush=True)
    start = time.monotonic()
    with log.open('w') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        attempt['pid'] = proc.pid
        (out/'attempt_running.json').write_text(json.dumps(wc.jsonable(attempt), indent=1))
        returncode = proc.wait()
    result = json.loads((run_dir/'result.json').read_text()) if (run_dir/'result.json').exists() else None
    attempt.update(returncode=returncode, wall_s=time.monotonic()-start, ended=datetime.now().astimezone().isoformat(),
                   run_dir_bytes=wc.du_bytes(run_dir), disk_free_gb_after=shutil.disk_usage(out).free/1e9,
                   result_status=(result or {}).get('status'),
                   note='A zero return code is not task success; see the analysis.')
    wc.write_new(out/'attempt.json', attempt)
    (out/'attempt_running.json').unlink()
    print(json.dumps({k: attempt[k] for k in ('episode', 'returncode', 'wall_s', 'result_status')}), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('launch')
    s.add_argument('--episode', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--disk-floor-gb', type=float, default=15.)
    s.set_defaults(func=cmd_launch)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
