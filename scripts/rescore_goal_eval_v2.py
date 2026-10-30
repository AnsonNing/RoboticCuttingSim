#!/usr/bin/env python3
"""POST_HOC_DIAGNOSTIC: rescore recorded Phase E kinematic episodes under goal_eval_v2 (read-only).

For each sample: the round's stage-1 record gives the compiled scene and goal; ``compile_task`` must
reproduce the recorded task id; the preserved ``physics_state.npz`` is the recording. The script scores it
twice with ``goals.evaluate_goal``: the goal as recorded (v1, a reproduction check against the recorded
``result.json`` verdict and failed checks) and the same goal with ``evaluator_version = goal_eval_v2``.
Nothing under the round's directories is written or changed; the recorded verdicts and evidence stand.
Positions are stored as float32 (the in-process evaluation used float64), so position-derived quality
numbers can differ in the last digits; outcome checks use reference positions and codes only.

  .venv/bin/python scripts/rescore_goal_eval_v2.py --stage1 artifacts/wp11/phaseE/round1_stage1_001 \
      --material-root artifacts/wp11/phaseE/round1_kinematic_001 --samples E1_TEST_D_s1702 E1_TEST_D_s1704 \
      --out artifacts/wp11/phaseE/round2_posthoc_v2_001
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import generalization as G  # noqa: E402
from robotic_cutting_sim.goals import evaluate_goal  # noqa: E402
from robotic_cutting_sim.scene_config import compile_task  # noqa: E402

LABEL = 'POST_HOC_DIAGNOSTIC'


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                 else v.item() if isinstance(v, np.generic) else str(v)))


def recording(path, task):
    data = np.load(path)
    return {'time_s': data['time_s'], 'positions': data['positions'].astype(np.float64),
            'velocities': data['velocities'].astype(np.float64), 'codes': data['codes'], 'mass_kg': data['mass_kg'],
            'reference_positions': data['reference_positions'], 'reference_indices': data['reference_indices'],
            'reference_stencil_nodes': data['reference_stencil_nodes'],
            'reference_stencil_weights': data['reference_stencil_weights'],
            'reference_stencil_gradients': data['reference_stencil_gradients'], 'max_cuts': int(data['max_cuts']),
            'code_encoding': str(data['code_encoding']), 'particle_spacing_m': float(task.particle_spacing_m)}


def step_summary(evaluation):
    outcome = evaluation['outcome']
    rows = []
    for row in outcome.get('steps', []):
        o = row['outcome']
        m = o.get('metrics', {})
        rows.append({'index': row['index'], 'goal_id': row['goal_id'], 'type': row['type'], 'status': o['status'],
                     'failed_checks': [k for k, ok in o.get('checks', {}).items() if not ok],
                     'checks': o.get('checks'), 'digit_scope': o.get('digit_scope'),
                     'mechanical_components': m.get('mechanical_components'),
                     'piece_mass_fractions': m.get('piece_mass_fractions'),
                     'removed': {i['id']: i['removed_mass_fraction'] for i in m.get('items', [])} or None,
                     'prior_pieces': m.get('prior_pieces'),
                     'largest_component_below_plane_mass_share': m.get('largest_component_below_plane_mass_share')})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage1', required=True)
    p.add_argument('--material-root', required=True)
    p.add_argument('--samples', nargs='+', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {'label': LABEL, 'command': sys.argv, 'git_sha': git('rev-parse', 'HEAD'),
              'git_dirty': bool(git('status', '--porcelain')), 'root': str(ROOT),
              'started': datetime.now().astimezone().isoformat(),
              'scope': 'Read-only rescoring of preserved round-1 kinematic episodes under goal_eval_v2; the round-1 '
                       'verdicts, classes and evidence are unchanged and this is not a round result.', 'samples': []}
    for sample_id in args.samples:
        s1 = json.loads((Path(args.stage1)/f'{sample_id}.json').read_text())
        result = json.loads((Path(args.material_root)/sample_id/'result.json').read_text())
        scene, goal = s1['scene_compiled'], s1['goal_compiled']
        task = compile_task(scene, goal, seed=42)
        if task.task_id != result['task_id'] or task.task_id != s1['compile']['task_id']:
            raise SystemExit(f'{sample_id}: compiled {task.task_id}, recorded {result["task_id"]}')
        data = recording(Path(args.material_root)/sample_id/'physics_state.npz', task)
        v1 = evaluate_goal(goal, data, task=task)
        recorded = result['evaluation']
        reproduction = {'recorded_verdict': recorded['verdict'], 'rescored_v1_verdict': v1['verdict'],
                        'recorded_failed_checks': G.failed_checks(recorded), 'rescored_v1_failed_checks': G.failed_checks(v1),
                        'match': recorded['verdict'] == v1['verdict'] and G.failed_checks(recorded) == G.failed_checks(v1)}
        v2 = evaluate_goal({**goal, 'evaluator_version': 'goal_eval_v2'}, data, task=task)
        row = {'sample_id': sample_id, 'task_id': task.task_id, 'label': LABEL, 'round1_class': result.get('class'),
               'v1_reproduction': reproduction, 'v2_verdict': v2['verdict'], 'v2_integrity': v2['integrity']['status'],
               'v2_failed_checks': G.failed_checks(v2), 'v2_steps': step_summary(v2),
               'v2_digit_allocation': v2['outcome'].get('digit_allocation'),
               'v2_sever_plane_location': {k: q.get('plane_location') for k, q in v2['quality'].items()
                                           if isinstance(q, dict) and 'plane_location' in q},
               'v1_sever_plane_location': {k: q.get('plane_location') for k, q in v1['quality'].items()
                                           if isinstance(q, dict) and 'plane_location' in q}}
        (out/f'{sample_id}_v1.json').write_text(json.dumps(jsonable(v1), indent=1))
        (out/f'{sample_id}_v2.json').write_text(json.dumps(jsonable(v2), indent=1))
        report['samples'].append(row)
        print(f'[{LABEL}] {sample_id}: recorded {recorded["verdict"]} | v1 rescored {v1["verdict"]} '
              f'(match {reproduction["match"]}) | v2 {v2["verdict"]} {G.failed_checks(v2)}', flush=True)
    report['finished'] = datetime.now().astimezone().isoformat()
    (out/'posthoc_summary.json').write_text(json.dumps(jsonable(report), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
