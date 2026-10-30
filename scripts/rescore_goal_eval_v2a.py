#!/usr/bin/env python3
"""POST_HOC_DIAGNOSTIC: rescore recorded Phase E kinematic episodes under goal_eval_v2a (read-only).

For each sample the round's stage-1 record gives the compiled scene and goal; ``compile_task`` must reproduce
the recorded task id, and the preserved ``physics_state.npz`` is the recording. The goal is scored as recorded
(a reproduction check against the recorded ``result.json`` verdict and failed checks), then explicitly with
``goal_eval_v1`` (no version), ``goal_eval_v2`` and ``goal_eval_v2a``. The question answered is whether any
v2a verdict differs from the v2 verdict. Nothing under the round directories is written or changed; the recorded
verdicts and evidence stand. Positions are stored as float32 (the in-process evaluation used float64), so
position-derived quality numbers can differ in the last digits; outcome checks use reference positions and codes.

  .venv/bin/python scripts/rescore_goal_eval_v2a.py \\
      --round round2 artifacts/wp11/phaseE/round2_stage1_001 artifacts/wp11/phaseE/round2_kinematic_001 \\
          E2_TEST_D2_s2501 E2_TEST_D2_s2502 E2_TEST_D2_s2503 E2_TEST_D2_s2504 E2_TEST_D2_s2505 E2_TEST_D2_s2506 \\
      --round round1 artifacts/wp11/phaseE/round1_stage1_001 artifacts/wp11/phaseE/round1_kinematic_001 \\
          E1_TEST_D_s1702 E1_TEST_D_s1704 \\
      --out artifacts/wp11/phaseE/review_posthoc_v2a_001
"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import generalization as G  # noqa: E402
from robotic_cutting_sim.goals import evaluate_goal  # noqa: E402
from robotic_cutting_sim.scene_config import compile_task  # noqa: E402

LABEL = 'POST_HOC_DIAGNOSTIC'
_spec = importlib.util.spec_from_file_location('rescore_goal_eval_v2_for_v2a', ROOT/'scripts'/'rescore_goal_eval_v2.py')
V2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V2)   # read-only reuse: recording loader, jsonable, step_summary


def with_version(goal, version):
    doc = {k: v for k, v in goal.items() if k != 'evaluator_version'}
    return doc if version == 'goal_eval_v1' else {**doc, 'evaluator_version': version}


def v2a_extras(evaluation):
    outcome = evaluation['outcome']
    final = outcome.get('final_all_digit_substrate') or {}
    return {'sequence_checks': outcome.get('checks'), 'fail_reasons': outcome.get('fail_reasons'),
            'unowned_digits_labelled_particles': (outcome.get('digit_allocation') or {}).get('unowned_digits_labelled_particles'),
            'final_all_digit_substrate': {k: final.get(k) for k in ('status', 'checks', 'plane_local_z_m', 'pieces',
                                                                     'detached_below_plane_mass_fraction_total', 'reason')},
            'detached_by_earlier_steps': {row['goal_id']: row['outcome'].get('digit_scope', {}).get('detached_by_earlier_steps')
                                          for row in outcome.get('steps', [])}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--round', nargs='+', action='append', required=True, metavar='NAME STAGE1 MATERIAL SAMPLE',
                   help='round name, stage-1 directory, material directory, then sample ids')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)   # never overwrite
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    report = {'label': LABEL, 'command': sys.argv, 'git_sha': git('rev-parse', 'HEAD'),
              'git_dirty': bool(git('status', '--porcelain')), 'root': str(ROOT),
              'started': datetime.now().astimezone().isoformat(),
              'scope': 'Read-only rescoring of preserved Phase E kinematic episodes under goal_eval_v2a; recorded '
                       'verdicts, classes and evidence are unchanged and this is not a round result.', 'samples': []}
    for row in args.round:
        name, stage1, material, *samples = row
        for sample_id in samples:
            start = time.monotonic()
            s1 = json.loads((Path(stage1)/f'{sample_id}.json').read_text())
            result = json.loads((Path(material)/sample_id/'result.json').read_text())
            scene, goal = s1['scene_compiled'], s1['goal_compiled']
            task = compile_task(scene, goal, seed=42)
            if task.task_id != result['task_id'] or task.task_id != s1['compile']['task_id']:
                raise SystemExit(f'{sample_id}: compiled {task.task_id}, recorded {result["task_id"]}')
            data = V2.recording(Path(material)/sample_id/'physics_state.npz', task)
            recorded = result['evaluation']
            as_recorded = evaluate_goal(goal, data, task=task)
            scored = {v: evaluate_goal(with_version(goal, v), data, task=task)
                      for v in ('goal_eval_v1', 'goal_eval_v2', 'goal_eval_v2a')}
            entry = {'round': name, 'sample_id': sample_id, 'task_id': task.task_id, 'label': LABEL,
                     'recorded_evaluator_version': goal.get('evaluator_version', 'goal_eval_v1'),
                     'recorded_class': result.get('class'),
                     'reproduction': {'recorded_verdict': recorded['verdict'], 'rescored_verdict': as_recorded['verdict'],
                                      'recorded_failed_checks': G.failed_checks(recorded),
                                      'rescored_failed_checks': G.failed_checks(as_recorded),
                                      'match': recorded['verdict'] == as_recorded['verdict']
                                      and G.failed_checks(recorded) == G.failed_checks(as_recorded)},
                     'verdicts': {v: e['verdict'] for v, e in scored.items()},
                     'failed_checks': {v: G.failed_checks(e) for v, e in scored.items()},
                     'v2a_differs_from_v2': scored['goal_eval_v2a']['verdict'] != scored['goal_eval_v2']['verdict'],
                     'v2a_steps': V2.step_summary(scored['goal_eval_v2a']), 'v2a': v2a_extras(scored['goal_eval_v2a']),
                     'wall_s': time.monotonic()-start}
            for version, evaluation in scored.items():
                (out/f'{sample_id}_{version}.json').write_text(json.dumps(V2.jsonable(evaluation), indent=1))
            report['samples'].append(entry)
            print(f'[{LABEL}] {sample_id}: recorded {recorded["verdict"]} (reproduced {entry["reproduction"]["match"]}) | '
                  f'v1 {entry["verdicts"]["goal_eval_v1"]} | v2 {entry["verdicts"]["goal_eval_v2"]} | '
                  f'v2a {entry["verdicts"]["goal_eval_v2a"]} {entry["v2a"]["fail_reasons"]} '
                  f'final {entry["v2a"]["final_all_digit_substrate"]["status"]} | differs {entry["v2a_differs_from_v2"]}',
                  flush=True)
    report['any_v2a_verdict_differs_from_v2'] = any(s['v2a_differs_from_v2'] for s in report['samples'])
    report['finished'] = datetime.now().astimezone().isoformat()
    (out/'posthoc_summary.json').write_text(json.dumps(V2.jsonable(report), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
