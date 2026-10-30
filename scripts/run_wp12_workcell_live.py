#!/usr/bin/env python3
"""WP-12 Phase 2b: live Native episodes on blade_master_flat_board_v1 (declared in configs/wp12/workcell_live_v1.json).

Every episode runs through the unchanged WP-11 live path ``scripts/run_wp10_live.py --runtime isaac_sim
--scene/--goal`` (physical grasp of ruler_knife_v2_y, HOME reset, goal planner, measured edge feedback, hold to
the time limit, physical return home). Nothing here changes physics, evaluators or thresholds.

Subcommands (outputs are new directories; nothing is ever overwritten):

* ``prepare --episode ID --out DIR``: compile with the declared live overrides and duration rule; write
  ``DIR/scene.json``, ``DIR/goal.json`` and ``DIR/prepare.json``. CPU only.
* ``launch --episode ID --out artifacts/wp12/workcell_live_NNN``: prepare into ``OUT/inputs`` and run the Native
  episode into ``OUT/run`` (one process; the attempt receipt records command, GPU, disk, wall time).
* ``analyze --root OUT [--out OUT/analysis_NNN]``: read-only analysis (``run_wp12_workcell_live_analysis.py``).

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_workcell_live.py launch --episode CUT \\
      --out artifacts/wp12/workcell_live_001
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
PROTOCOL = ROOT/'configs/wp12/workcell_live_v1.json'


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                else v.item() if isinstance(v, np.generic) else str(v)))


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; outputs are never overwritten')
    path.write_text(json.dumps(jsonable(payload), indent=1, allow_nan=True))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    run = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {'git_sha': run('rev-parse', 'HEAD'), 'git_dirty': bool(run('status', '--porcelain')),
            'git_status_porcelain': run('status', '--porcelain').splitlines(), 'root': str(ROOT)}


def load_protocol():
    return json.loads(PROTOCOL.read_text())


def episode_spec(protocol, episode):
    for spec in protocol['episodes']:
        if spec['id'] == episode:
            return spec
    raise KeyError(f'{episode} is not a declared episode of {PROTOCOL.name}')


def _flat(value, prefix=''):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flat(item, f'{prefix}.{key}' if prefix else key))
        return out
    return {prefix: value}


def compile_live(protocol, spec):
    from robotic_cutting_sim.scene_config import compile_task
    scene = json.loads((ROOT/spec['scene']).read_text())
    source_goal = json.loads((ROOT/spec['goal']).read_text())
    options = spec.get('planner_overrides', protocol['controller']['planner_overrides'])
    seed = protocol['runtime']['seed']
    goal = deepcopy(source_goal)
    if 'duration_rule' in spec:   # recorded live goal reused as-is (WP-11 reproduction)
        task = compile_task(deepcopy(scene), deepcopy(goal), planner_options=options, seed=seed)
        done = next((float(t) for t, label in zip(task._times(), task.reference_phase_labels) if label == 'DONE'), None)
        duration = float(goal['episode']['duration_s'])
    else:
        rule = protocol['controller']['duration_rule']
        goal['episode']['duration_s'] = float(rule['probe_duration_s'])
        probe = compile_task(deepcopy(scene), deepcopy(goal), planner_options=options, seed=seed)
        done = next((float(t) for t, label in zip(probe._times(), probe.reference_phase_labels) if label == 'DONE'), None)
        if done is None:
            raise RuntimeError('The live probe compile has no DONE knot')
        step = float(rule['round_up_to_s'])
        duration = math.ceil((float(rule['plan_end_factor'])*done+float(rule['settle_after_plan_s']))/step-1e-9)*step
        goal['episode']['duration_s'] = float(duration)
        task = compile_task(deepcopy(scene), deepcopy(goal), planner_options=options, seed=seed)
    steps = int(round(task.duration_s/task.control_dt_s))
    stride = 19 if spec['id'] == 'WP11_CUT_REPRO' else max(5, math.ceil(steps/40))
    a, b = _flat(source_goal), _flat(goal)
    return {'task': task, 'scene': scene, 'goal': goal, 'options': options, 'plan_end_s': done, 'duration_s': duration,
            'steps': steps, 'history_stride': stride,
            'goal_diff_vs_source': sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))}


def prepare(protocol, episode, out):
    from robotic_cutting_sim import goal_planner
    spec = episode_spec(protocol, episode)
    built = compile_live(protocol, spec)
    if built['goal_diff_vs_source'] not in ([], ['episode.duration_s']):
        raise SystemExit(f'Live goal differs from the source goal beyond the duration: {built["goal_diff_vs_source"]}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'scene.json', built['scene'])
    write_new(out/'goal.json', built['goal'])
    task = built['task']
    _, report = goal_planner.plan_with_report(task.scene, task.goal, task.tool_profile, task.controller['planner'])
    receipt = {'episode': episode, 'spec': spec, 'protocol': str(PROTOCOL.relative_to(ROOT)), 'protocol_sha256': sha256(PROTOCOL),
               'source_scene_sha256': sha256(ROOT/spec['scene']), 'source_goal_sha256': sha256(ROOT/spec['goal']),
               'live_task_id': task.task_id, 'live_config_sha256': task.config_sha256,
               'solver_module': task.solver_module, 'max_cuts': task.max_cuts,
               'particles': int(task.sample_particles()[0].shape[0]), 'live_plan_end_s': built['plan_end_s'],
               'duration_s': built['duration_s'], 'steps': built['steps'], 'history_stride': built['history_stride'],
               'record_every': protocol['runtime']['record_every'], 'goal_diff_vs_source': built['goal_diff_vs_source'],
               'planner_overrides': built['options'], 'planner_options_resolved': task.controller['planner'],
               'plan_report_operations': report.get('operations'), 'phases_live': list(dict.fromkeys(task.reference_phase_labels)),
               'workcell_geometry': task.workcell_geometry(), 'compiled_with': sys.executable, **git_state(),
               'prepared': datetime.now().astimezone().isoformat()}
    write_new(out/'prepare.json', receipt)
    return receipt


def runner_command(protocol, prep, inputs, run_dir):
    runtime = protocol['runtime']
    return [runtime['python'], 'scripts/run_wp10_live.py', '--runtime', 'isaac_sim',
            '--scene', str(Path(inputs)/'scene.json'), '--goal', str(Path(inputs)/'goal.json'),
            '--planner-options', json.dumps(prep['planner_overrides']), *runtime['flags'],
            '--record-every', str(prep['record_every']), '--oracle-stride', str(prep['history_stride']),
            '--steps', str(prep['steps']), '--gpu', str(runtime['gpu']), '--seed', str(runtime['seed']), '--out', str(run_dir)]


def gpu_snapshot():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory,process_name', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout
    return {'gpus': [line.strip() for line in out.strip().splitlines()], 'compute_apps': [line.strip() for line in apps.strip().splitlines()]}


def du_bytes(path):
    out = subprocess.run(['du', '-sb', str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def cmd_prepare(args):
    receipt = prepare(load_protocol(), args.episode, args.out)
    print(json.dumps({k: receipt[k] for k in ('episode', 'live_task_id', 'live_plan_end_s', 'duration_s', 'steps', 'history_stride')}))
    return 0


def cmd_launch(args):
    protocol = load_protocol()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT/out
    out.mkdir(parents=True, exist_ok=False)
    prep = prepare(protocol, args.episode, out/'inputs')
    free = shutil.disk_usage(out).free/1e9
    if free < args.disk_floor_gb:
        write_new(out/'attempt.json', {'status': 'NOT_RUN', 'reason': f'disk {free:.1f} GB < floor {args.disk_floor_gb} GB'})
        return 3
    run_dir = out/'run'
    command = runner_command(protocol, prep, (out/'inputs').relative_to(ROOT), run_dir.relative_to(ROOT))
    env = {**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'PYTHONUNBUFFERED': '1', 'TMPDIR': protocol['runtime']['tmpdir']}
    log = ROOT/'artifacts/wp12/logs'/f'{out.name}_{args.episode}.log'
    if log.exists():
        raise FileExistsError(log)
    attempt = {'episode': args.episode, 'command': command, 'cwd': str(ROOT),
               'env': {k: env[k] for k in ('OMNI_KIT_ACCEPT_EULA', 'PYTHONUNBUFFERED', 'TMPDIR')}, 'log': str(log),
               'started': datetime.now().astimezone().isoformat(), 'gpu_before': gpu_snapshot(),
               'disk_free_gb_before': free, **git_state()}
    print(json.dumps({k: attempt[k] for k in ('episode', 'log', 'started')}), flush=True)
    start = time.monotonic()
    with log.open('w') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        attempt['pid'] = proc.pid
        (out/'attempt_running.json').write_text(json.dumps(jsonable(attempt), indent=1))
        returncode = proc.wait()
    result = json.loads((run_dir/'result.json').read_text()) if (run_dir/'result.json').exists() else None
    attempt.update(returncode=returncode, wall_s=time.monotonic()-start, ended=datetime.now().astimezone().isoformat(),
                   run_dir_bytes=du_bytes(run_dir), disk_free_gb_after=shutil.disk_usage(out).free/1e9,
                   result_status=(result or {}).get('status'),
                   note='A zero return code is not task success; see the analysis.')
    write_new(out/'attempt.json', attempt)
    (out/'attempt_running.json').unlink()
    print(json.dumps({k: attempt[k] for k in ('episode', 'returncode', 'wall_s', 'result_status')}), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('prepare')
    s.add_argument('--episode', required=True)
    s.add_argument('--out', required=True)
    s.set_defaults(func=cmd_prepare)
    s = sub.add_parser('launch')
    s.add_argument('--episode', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--disk-floor-gb', type=float, default=15.)
    s.set_defaults(func=cmd_launch)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
