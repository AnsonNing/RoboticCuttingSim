#!/usr/bin/env python3
"""Read-only Native analysis views: reconstruct metadata, never replace run outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_wp13 import ROOT, G


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--view', default='analysis_view_001')
    a = p.parse_args(); root = a.run_root.resolve()
    launch = json.loads((root/'launch.json').read_text())
    command = launch['command']
    options = json.loads(command[command.index('--planner-options')+1])
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.goal_planner import plan_with_report
    task = compile_task(root/'inputs/scene.json', root/'inputs/goal.json', planner_options=options, seed=42)
    _, report = plan_with_report(task.scene, task.goal, task.tool_profile, task.controller['planner'])
    view = root/a.view; view.mkdir(exist_ok=False)
    inputs = view/'inputs'; inputs.mkdir()
    for name in ('scene.json', 'goal.json'):
        (inputs/name).symlink_to(root/'inputs'/name)
    (view/'run').symlink_to(root/'run', target_is_directory=True)
    (view/'attempt.json').symlink_to(root/'attempt.json')
    prepare = {'episode': launch['case']['episode'], 'planner_overrides': options,
               'live_task_id': task.task_id, 'plan_report_operations': report['operations'],
               'reconstructed_from': str(root/'launch.json'), 'wp13_source_root': str(root),
               'input_sha256': {n: G.sha(root/'inputs'/n) for n in ('scene.json', 'goal.json')},
               'scope': 'Analysis-only metadata derived from actual launched inputs and unchanged planner; '
                        'original simulation files and incomplete analysis metadata remain untouched.'}
    G.write_new(inputs/'prepare.json', prepare)
    args = [sys.executable, 'scripts/run_wp12_workcell_live_analysis.py', '--root', str(view),
            '--out', str(view/'analysis'), '--skip-path-check']
    code = subprocess.run(args, cwd=ROOT).returncode
    G.write_new(view/'analysis_command.json', {'command': args, 'returncode': code})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
