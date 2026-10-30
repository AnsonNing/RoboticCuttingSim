"""WP-12 Phase 5: the declared Native success definition (benchmark_v1.json native.success_definition; freeze decision
finding 1) in run_wp12_phase5.native_execution. CPU only: synthetic attempt directories for every class, and the
stored Phase 2b records workcell_live_001-007 (artifacts; the test skips that part if they are absent)."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from robotic_cutting_sim import generalization_wp12 as W

ROOT = W.ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(f'test_p5ns_{name}', ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def P5():
    return _load('run_wp12_phase5')


GOOD_EPISODE = {'task_end_reason': 'CONTROLLER_DONE', 'policy_final_phase': 'DONE', 'runtime_source': 'live_isaac_sim',
                'task_outcome': {'verdict': 'PASS', 'integrity': {'checks': {'complete_episode': True}}}}


def _row(phase='CUT:cut0', ik='ok', clamp=False, slip=0., rot=0., lost=False):
    return {'policy_phase': phase, 'info': {'ik_status': ik, 'action_report': {'workspace_clamped': clamp}},
            'grasp_status': {'slip_since_settling_m': slip, 'rotation_since_settling_rad': rot, 'grasp_lost': lost}}


def _attempt(root, sid='S', *, top=None, episode=None, rows=None, finite=True, failure=None, launch=True, attempt=None,
             run=True):
    d = Path(root)/sid
    (d/'inputs').mkdir(parents=True)
    if attempt is not None:
        (d/'attempt.json').write_text(json.dumps(attempt))
    if launch:
        (d/'launch.json').write_text(json.dumps({'status': 'LAUNCHING'}))
    if run:
        (d/'run').mkdir()
        if top is not None:
            (d/'run'/'result.json').write_text(json.dumps(top))
        if failure is not None:
            (d/'run'/'failure.json').write_text(json.dumps(failure))
        if episode is not None:
            (d/'run'/'episode_000').mkdir()
            (d/'run'/'episode_000'/'result.json').write_text(json.dumps(episode))
            with (d/'run'/'episode_000'/'actions.jsonl').open('w') as f:
                for r in (rows if rows is not None else [_row(), _row('DONE')]):
                    f.write(json.dumps(r)+'\n')
            x = np.zeros((2, 5, 3), np.float32)
            if not finite:
                x[1, 2, 0] = np.nan
            np.savez_compressed(d/'run'/'episode_000'/'physics_state.npz', positions=x, velocities=np.zeros_like(x))
    return sid


def _cls(P5, root, sid):
    execution = P5.native_execution(root, sid)
    row = W.final_row({'sample_id': sid, 'family': 'A', 'range_set': 'ID', 'split': 'benchmark', 'stratum': 'A_ID',
                       'seed': 0, 'variant': 'vertical', 'ood_dimensions': None, 'feasibility_class': 'valid_feasible',
                       'envelope_pending': False}, execution)
    return row['class'], row['failure_kind'], row['reason']


TOP = {'status': 'PASS_EXECUTION'}


@pytest.mark.parametrize('case, expected', [
    (dict(top=TOP, episode=GOOD_EPISODE), ('task_success', None)),
    (dict(top=TOP, episode=GOOD_EPISODE, rows=[_row(slip=0.0021)]), ('execution_failure', 'grasp')),
    (dict(top=TOP, episode=GOOD_EPISODE, rows=[_row(rot=math.pi/60+1e-4)]), ('execution_failure', 'grasp')),
    (dict(top=TOP, episode=GOOD_EPISODE, rows=[_row(lost=True)]), ('execution_failure', 'grasp')),
    (dict(top=TOP, episode=GOOD_EPISODE, rows=[_row(ik='fail')]), ('execution_failure', 'other')),
    (dict(top=TOP, episode=GOOD_EPISODE, rows=[_row(clamp=True)]), ('execution_failure', 'other')),
    # the physical return home is outside the task phase: its IK / grasp rows never decide task success
    (dict(top=TOP, episode=GOOD_EPISODE, rows=[_row(), _row('RETURN_HOME:lift', ik='fail', slip=0.01)]), ('task_success', None)),
    (dict(top=TOP, episode=GOOD_EPISODE, finite=False), ('execution_failure', 'solver_invalid')),
    (dict(top=TOP, episode={**GOOD_EPISODE, 'task_end_reason': 'TIME_LIMIT'}), ('execution_failure', 'plan_not_finished')),
    (dict(top=TOP, episode={**GOOD_EPISODE, 'task_outcome': {'verdict': 'FAIL_INTEGRITY', 'integrity': {'checks': {'complete_episode': False}}}}),
     ('execution_failure', 'integrity')),
    (dict(top=TOP, episode={**GOOD_EPISODE, 'task_outcome': {'verdict': 'FAIL', 'outcome': {'checks': {'expected_pieces': False}}}}),
     ('execution_failure', 'outcome')),
    # plan-preflight refusals are execution failures, never not_run (the provisional mapping called them not_run)
    (dict(top={'status': 'REFUSED_PLAN_PREFLIGHT_IK_FAILURE', 'refused_before_launch': True}), ('execution_failure', 'other')),
    (dict(top={'status': 'REFUSED_PLAN_PREFLIGHT', 'refused_before_launch': True}), ('execution_failure', 'other')),
    # launched without an episode result: crash
    (dict(top={'status': 'FAIL_EXECUTION'}, failure={'error': 'RuntimeError: kit died'}), ('execution_failure', 'crash')),
    (dict(top={'status': 'FAIL_EXECUTION'}, failure={'error': 'SlotOverflowError: node 7'}), ('execution_failure', 'solver_invalid')),
    (dict(), ('execution_failure', 'crash')),                                  # run dir, no result.json at all
    (dict(run=False), ('execution_failure', 'crash')),                         # launch receipt, process never wrote run/
    (dict(top=TOP, episode={**GOOD_EPISODE, 'task_outcome': {}}), ('execution_failure', 'crash')),   # no verdict
    # never launched: not_run (stays in the feasible denominator)
    (dict(run=False, launch=False, attempt={'status': 'NOT_RUN', 'reason': '--dry-run (command recorded only)'}), ('not_run', None)),
    (dict(run=False, launch=False, attempt={'status': 'NOT_RUN', 'reason': 'disk 3.0 GB < floor 15 GB'}), ('not_run', None)),
    (dict(run=False, launch=False), ('not_run', None)),
])
def test_native_classes(P5, tmp_path, case, expected):
    sid = _attempt(tmp_path, **case)
    cls, kind, reason = _cls(P5, tmp_path, sid)
    assert (cls, kind) == expected, reason


def test_absent_attempt_is_not_run(P5, tmp_path):
    assert P5.native_execution(tmp_path, 'missing') is None
    assert W.final_row({'sample_id': 'x', 'feasibility_class': 'valid_feasible'}, None)['class'] == 'not_run'


def test_grasp_gates_are_the_declared_warning_gates():
    WA = _load('run_wp12_workcell_live_analysis')
    assert WA.GRASP_GATES['max_slip_m'] == 0.002 and WA.GRASP_GATES['max_rotation_rad'] == math.pi/60
    config = json.loads(W.BENCHMARK_V1.read_text())
    text = ' '.join(config['native']['success_definition']['task_success_requires'])
    assert 'slip > 0.002 m' in text and 'pi/60' in text and 'workspace-clamped' in text


def test_map_live_class_table():
    assert W.map_live_class('TASK_SUCCESS', 'ok')[:2] == ('task_success', None)
    assert W.map_live_class('GRASP_FAILURE', 'g')[:2] == ('execution_failure', 'grasp')
    assert W.map_live_class('IK_FAILURE', 'i')[:2] == ('execution_failure', 'other')
    assert W.map_live_class('SCENE_INFEASIBLE', 's')[:2] == ('execution_failure', 'other')
    assert W.map_live_class('NOT_RUN', 'no result')[:2] == ('execution_failure', 'crash')
    assert W.map_live_class(None, 'dry run', launched=False)[:2] == ('not_run', None)
    for reason, kind in (('solver state invalid: x', 'solver_invalid'), ('the scripted plan did not finish', 'plan_not_finished'),
                         ('integrity: a', 'integrity'), ('outcome: b', 'outcome')):
        assert W.map_live_class('CUTTING_FAILURE', reason)[:2] == ('execution_failure', kind)
    with pytest.raises(ValueError):
        W.map_live_class('SOMETHING', 'x')


def test_kinematic_results_are_unaffected():
    """A kinematic result (no native_classification) keeps the kinematic classification."""
    assert W.classify_execution({'status': 'ABORTED', 'reason': 'reset failed'})[:2] == ('execution_failure', 'crash')
    assert W.classify_execution(None)[0] == 'not_run'


PHASE2B = ROOT/'docs/evidence/wp12/phase2b_native_001.json'


@pytest.mark.skipif(not (ROOT/'artifacts/wp12/workcell_live_007/run/result.json').exists(),
                    reason='Phase 2b records artifacts/wp12/workcell_live_001-007 not present')
def test_reproduces_phase2b_records(P5):
    """native_execution on the Phase 2b attempt directories reproduces their recorded live class (analysis.json and
    the committed evidence) and maps it by the declared rule: 001, 003-007 task_success; 002 (the PARTIAL episode,
    whose sever verdict is expected FAIL) execution_failure / outcome."""
    evidence = {e['root']: e for e in json.loads(PHASE2B.read_text())['live_episodes']}
    expected = {1: 'task_success', 2: 'execution_failure', 3: 'task_success', 4: 'task_success', 5: 'task_success',
                6: 'task_success', 7: 'task_success'}
    for i, cls in expected.items():
        sid = f'workcell_live_{i:03d}'
        execution = P5.native_execution(ROOT/'artifacts/wp12', sid)
        live = execution['live']
        analysis = json.loads((ROOT/f'artifacts/wp12/{sid}/analysis_001/analysis.json').read_text())
        assert live['class'] == analysis['class'] == evidence[f'artifacts/wp12/{sid}']['class'], sid
        assert live['reason'] == analysis['class_reason'], sid
        assert live['ik'] == analysis['ik'], sid
        got = W.classify_execution(execution)
        assert got[0] == cls, (sid, got)
        if i == 2:
            assert got[1] == 'outcome'
