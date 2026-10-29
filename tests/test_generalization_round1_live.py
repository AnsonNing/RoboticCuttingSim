"""WP-11 Phase E round 1 live stage declaration (configs/wp11/generalization_round1_live.json)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _live_module():
    spec = importlib.util.spec_from_file_location('run_generalization_live', ROOT/'scripts'/'run_generalization_live.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _live_module()


@pytest.fixture(scope='module')
def declared():
    return L.load_live()


def test_declaration_pins_the_round_file_and_nominated_seeds(declared):
    live, config = declared
    assert live['status'] == 'PREDECLARED' and live['round_id'] == config['round_id']
    ids = [s['sample_id'] for s in sorted(live['samples'], key=lambda s: s['priority'])]
    assert ids == ['E1_TRIM_s1203', 'E1_CUT_s1101', 'E1_TEST_A_s6', 'E1_TEST_C_s1602', 'E1_TEST_B_s1502',
                   'E1_CUT_s1102', 'E1_TRIM_s1201', 'E1_SLICE_PUSH_s1301', 'E1_TEST_A_s2']
    for spec in live['samples']:
        assert spec['seed'] in config['families'][spec['family']]['seeds']
    assert not any('TEST_D' in s for s in ids)
    assert {'E1_TEST_D_s1701', 'E1_TEST_D_s1702', 'E1_TEST_D_s1704'} <= set(live['held'])
    assert set(live['outcome_classes']) == {'SCENE_INFEASIBLE', 'IK_FAILURE', 'GRASP_FAILURE', 'CUTTING_FAILURE',
                                            'TASK_SUCCESS', 'NOT_RUN'}


def test_live_recompile_changes_only_the_duration(declared):
    live, config = declared
    spec = L.sample_spec(live, 'E1_TRIM_s1203')
    built = L.compile_live(live, config, spec['family'], spec['seed'])
    stage1 = json.loads((ROOT/live['kinematic']['stage1_dir']/'E1_TRIM_s1203.json').read_text()) \
        if (ROOT/live['kinematic']['stage1_dir']/'E1_TRIM_s1203.json').exists() else None
    if stage1 is not None:
        assert built['kinematic_task'].task_id == stage1['compile']['task_id']
    assert built['goal_diff_vs_kinematic'] == ['episode.duration_s']
    for key in ('duration_s', 'steps', 'history_stride'):
        assert built[key] == spec['declared_live_task'][key]
    assert built['task'].task_id == spec['declared_live_task']['live_task_id_at_declaration_cpu']


def _episode(verdict='PASS', end='CONTROLLER_DONE'):
    return {'task_end_reason': end, 'task_outcome': {'verdict': verdict, 'integrity': {'checks': {'complete_episode': True}},
                                                     'outcome': {'checks': {'x': verdict == 'PASS'}}}}


def test_class_precedence(declared):
    live, _ = declared
    ok_grasp = {'grasp_lost': False, 'max_slip_m': 1e-6, 'max_rotation_rad': 1e-4}
    ok_ik = {'task_steps_ik_not_ok': 0, 'task_steps_workspace_clamped': 0}
    top = {'status': 'PASS_EXECUTION'}
    assert L.classify_live(live, top, _episode(), [], None, True, ok_grasp, ok_ik)[0] == 'TASK_SUCCESS'
    assert L.classify_live(live, top, _episode('FAIL'), [], None, True, ok_grasp, ok_ik)[0] == 'CUTTING_FAILURE'
    assert L.classify_live(live, top, _episode(end='TIME_LIMIT'), [], None, True, ok_grasp, ok_ik)[0] == 'CUTTING_FAILURE'
    slip = {**ok_grasp, 'max_slip_m': .0021}
    assert L.classify_live(live, top, _episode(), [], None, True, slip, ok_ik)[0] == 'GRASP_FAILURE'
    assert L.classify_live(live, top, _episode(), [], None, True, {**ok_grasp, 'grasp_lost': True}, ok_ik)[0] == 'GRASP_FAILURE'
    clamp = {**ok_ik, 'task_steps_workspace_clamped': 1}
    assert L.classify_live(live, top, _episode(), [], None, True, slip, clamp)[0] == 'IK_FAILURE'
    assert L.classify_live(live, top, _episode(), [], None, False, ok_grasp, ok_ik)[0] == 'CUTTING_FAILURE'
    refused = {'status': 'SCENE_INFEASIBLE', 'refused_before_launch': True}
    assert L.classify_live(live, refused, None, [], None, True, ok_grasp, ok_ik)[0] == 'SCENE_INFEASIBLE'
    crash = {'status': 'FAIL_EXECUTION'}
    assert L.classify_live(live, crash, None, [], {'error': "SlotOverflowError('x')"}, True, ok_grasp, ok_ik)[0] == 'CUTTING_FAILURE'
    assert L.classify_live(live, crash, None, [], {'error': 'RuntimeError()'}, True, ok_grasp, ok_ik)[0] == 'NOT_RUN'
    assert L.classify_live(live, None, None, [], None, True, ok_grasp, ok_ik)[0] == 'NOT_RUN'


def test_task_phase_keeps_the_planner_return_home_and_hold():
    rows = [{'policy_phase': p} for p in ('HOME', 'SKIM:p0', 'RETURN_HOME', 'HOLD_AFTER_STOP', 'RETURN_HOME:RETRACTING',
                                          'RETURN_HOME:MOVE_HOME')]
    assert [L.is_lifecycle_home(r) for r in rows] == [False, False, False, False, True, True]
    assert L._op_of('SKIM:p0') == 'p0' and L._op_of('RETURN_HOME:RETRACTING') is None and L._op_of('RETURN_HOME') is None
