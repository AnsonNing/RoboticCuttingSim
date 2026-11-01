#!/usr/bin/env python3
"""Small frozen WP-14 Native cohort through the maintained physical-grasp runner."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_wp13 import ROOT, build


def write_new(path, data):
    path.open('x').write(json.dumps(data, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=a.resume)
    protocol = json.loads(a.protocol.read_text())
    protected = [str(a.protocol), *protocol['native_rows'], 'src/robotic_cutting_sim/tlmpm_v7.py']
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *protected], check=True)
    subprocess.run(['git', 'ls-files', '--error-unmatch', *protected], check=True, stdout=subprocess.DEVNULL)
    for case_path in protocol['native_rows']:
        row = json.loads(Path(case_path).read_text())
        name = row['id']
        for solver in protocol['native_arms']:
            out = a.root/solver/name
            if out.exists():
                print(f'PRESERVED {out}', flush=True)
                continue
            out.mkdir(parents=True)
            inputs = out/'inputs'; inputs.mkdir()
            task, info = build(row, solver)
            if solver != 'tlmpm_v5' and protocol.get('candidate_profile'):
                from robotic_cutting_sim.research_profiles import select_profile
                from robotic_cutting_sim.scene_config import compile_task
                info['scene'] = select_profile(info['scene'], protocol['candidate_profile'])
                task = compile_task(info['scene'], info['goal'], planner_options=task.controller['planner'], seed=row.get('seed',42))
            write_new(inputs/'scene.json', info['scene'])
            write_new(inputs/'goal.json', info['goal'])
            options = task.controller['planner']
            from robotic_cutting_sim.goal_planner import plan_with_report
            _,plan_report=plan_with_report(task.scene,task.goal,task.tool_profile,options)
            steps = round(task.duration_s/task.control_dt_s)
            write_new(inputs/'prepare.json', {'episode': row['episode'], 'sample_id': name,
                'live_task_id': task.task_id, 'planner_overrides': options, 'duration_s': task.duration_s,
                'steps': steps, 'history_stride': max(5, math.ceil(steps/40)), 'record_every': 10,
                'plan_report_operations':plan_report['operations'],'live_config_sha256':task.config_sha256,
                'spec': {'id': row['episode']}, 'wp14_case': row, 'solver_module': solver})
            command = [str(ROOT/'.deps/isaacsim-venv/bin/python'), 'scripts/run_wp10_live.py',
                       '--runtime', 'isaac_sim', '--scene', str(inputs/'scene.json'), '--goal', str(inputs/'goal.json'),
                       '--planner-options', json.dumps(options), '--controller', 'scripted', '--edge-feedback', 'measured',
                       '--hold-until-time-limit', '--return-home', '--record-cameras', '--overview-camera',
                       '--record-every', '10', '--oracle-stride', str(max(5, math.ceil(steps/40))),
                       '--steps', str(steps), '--gpu', str(a.gpu), '--seed', '42', '--out', str(out/'run'),
                       '--surface-profile', protocol['surface_profile'], '--surface-timing-log']
            before = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.used,utilization.gpu',
                                              '--format=csv,noheader,nounits'], text=True).splitlines()
            free = int(before[a.gpu].split(',')[1])
            if free < 12000:
                write_new(out/'attempt.json', {'status': 'NOT_RUN', 'reason': 'less than 12 GB GPU headroom', 'gpu': before})
                raise SystemExit(3)
            attempt = {'status': 'STARTED', 'case': row, 'solver': solver, 'command': command, 'cwd': str(ROOT),
                       'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       'started': datetime.now(timezone.utc).isoformat(), 'gpu_before': before,
                       'material_seed': 42, 'scene_seed': row['seed'], 'task_id_prepared': task.task_id,
                       'surface_profile': protocol['surface_profile']}
            write_new(out/'launch.json', attempt)
            started = time.monotonic()
            with (out/'native.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env={**os.environ,
                    'PYTHONPATH': str(ROOT/'src'), 'PYTHONUNBUFFERED': '1', 'OMNI_KIT_ACCEPT_EULA': 'YES'},
                    stdout=log, stderr=subprocess.STDOUT)
                write_new(out/'pid.json', {'pid': child.pid})
                code = child.wait()
            attempt.update(status='FINISHED', returncode=code, wall_s=time.monotonic()-started,
                           ended=datetime.now(timezone.utc).isoformat())
            result = out/'run/result.json'
            if result.exists():
                attempt['runner_result'] = json.loads(result.read_text())
            write_new(out/'attempt.json', attempt)
            print(json.dumps({k: attempt[k] for k in ('status', 'case', 'solver', 'returncode', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
