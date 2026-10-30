"""WP-11 Phase E live_003: the runner's default plan preflight (scripts/wp11_motion_preflight.py ``plan``).

Scenes with two or more protrusions use the round-1 staged geometry (generalization.staged_preflight);
zero/one-protrusion (and legacy configured) tasks keep exactly the Phase D/C5 check.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _script('wp11_motion_preflight')


class _Checker:
    build_wall_s = 0.

    def __init__(self, task):
        self.task = task


def _stub(monkeypatch, protrusions):
    from robotic_cutting_sim import generalization, motion_preflight, scene_config
    task = SimpleNamespace(home={'joint_positions': [0.]*7}, protrusion_ids=tuple(f'p{i}' for i in range(protrusions)),
                           task_id='T', config_sha256='sha')
    calls = []

    def plan(task_, q_home, *, until_phase=None, checker=None):
        calls.append(('preflight_plan', q_home, until_phase))
        return {'status': 'PASS', 'reason': None, 'until_phase': until_phase}

    def staged(task_, checker=None, *, until_phase=None):
        calls.append(('staged_preflight', until_phase))
        return {'status': 'PASS', 'reason': None, 'geometry': 'staged'}, {(): checker}

    monkeypatch.setattr(scene_config, 'task_from_payload', lambda payload: task)
    monkeypatch.setattr(motion_preflight, 'MotionChecker', _Checker)
    monkeypatch.setattr(motion_preflight, 'preflight_plan', plan)
    monkeypatch.setattr(generalization, 'staged_preflight', staged)
    return calls


@pytest.mark.parametrize('protrusions', [0, 1])
@pytest.mark.parametrize('until', [None, 'REORIENT'])
def test_zero_or_one_protrusion_keeps_exactly_the_current_check(monkeypatch, protrusions, until):
    calls = _stub(monkeypatch, protrusions)
    out = CLI.run({'mode': 'plan', 'payload': {}, 'until_phase': until})
    expected = [('preflight_plan', [0.]*7, None)] + ([] if until is None else [('preflight_plan', [0.]*7, until)])
    assert calls == expected
    assert 'preflight_geometry' not in out
    assert set(out) == {'status', 'reason', 'required', 'full_plan', 'executed_prefix', 'mode', 'task_id',
                        'config_sha256', 'checker_build_wall_s', 'wall_s'}
    assert out['required'] == ('prefix' if until else 'full')


@pytest.mark.parametrize('protrusions', [2, 3])
@pytest.mark.parametrize('until', [None, 'SKIM'])
def test_two_or_more_protrusions_use_the_round1_staged_geometry(monkeypatch, protrusions, until):
    calls = _stub(monkeypatch, protrusions)
    out = CLI.run({'mode': 'plan', 'payload': {}, 'until_phase': until})
    assert calls == [('staged_preflight', None)] + ([] if until is None else [('staged_preflight', until)])
    assert out['preflight_geometry'] == {'rule': 'staged_multi_protrusion_v1', 'helper': 'generalization.staged_preflight',
                                         'protrusions': protrusions}
    assert out['status'] == 'PASS' and out['full_plan']['geometry'] == 'staged'


def test_staged_helper_default_is_unchanged_full_plan(monkeypatch):
    """until_phase is additive: without it the round-1 helper samples the full plan as before."""
    from robotic_cutting_sim import generalization, motion_preflight
    seen = []

    def samples(task, *, until_phase=None, **kw):
        seen.append(until_phase)
        return [], [], []

    monkeypatch.setattr(motion_preflight, 'plan_samples', samples)
    monkeypatch.setattr(motion_preflight, 'preflight_plan', lambda task, q, **kw: seen.append(kw) or {'status': 'PASS'})
    task = SimpleNamespace(home={'joint_positions': [0.]*7}, protrusion_ids=('p0',))
    receipt, _ = generalization.staged_preflight(task, checker=object())
    assert seen == [None, {'checker': seen[1]['checker']}] and receipt['status'] == 'PASS'


def _real_checker_available():
    """The real C2 checker needs Newton (FK/IK), MuJoCo and the official Franka asset cache; CI has none."""
    if importlib.util.find_spec('newton') is None or importlib.util.find_spec('mujoco') is None:
        return False
    from robotic_cutting_sim.workcell import ASSETS
    return (ROOT/'artifacts'/'workcell_asset_cache'/ASSETS['franka']).is_file()


@pytest.mark.skipif(not _real_checker_available(),
                    reason="NOT_RUN: real motion preflight needs newton, mujoco and the Franka asset cache")
def test_test_c_live_task_passes_the_default_preflight_with_staged_geometry():
    """The live_001 Test C task (refused by the single-hull default) passes the new default check."""
    from robotic_cutting_sim.scene_config import configured_task_payload
    live_mod = _script('run_generalization_live')
    live, config = live_mod.load_live()
    spec = live_mod.sample_spec(live, 'E1_TEST_C_s1602')
    task = live_mod.compile_live(live, config, spec['family'], spec['seed'])['task']
    assert task.task_id == spec['declared_live_task']['live_task_id_at_declaration_cpu']
    out = CLI.run({'mode': 'plan', 'payload': configured_task_payload(task)})
    assert out['preflight_geometry']['protrusions'] == 3
    assert out['status'] == 'PASS' and out['required'] == 'full'
    chunks = out['full_plan']['path']['chunks']
    assert [c['removed_protrusions'] for c in chunks] == [[], ['p0'], ['p0', 'p1'], ['p0', 'p1', 'p2']]


def test_measured_path_staged_chunks_by_removed_protrusions(monkeypatch):
    import numpy as np
    from robotic_cutting_sim import generalization, motion_preflight
    live_mod = _script('run_generalization_live')
    seen = []

    class Checker:
        thresholds = {'joint_step_rad': 1.}

        def __init__(self, task):
            self.task = task

        def check_joint_path(self, qs, modes, labels):
            seen.append((tuple(b for b in self.task.bumps), list(labels)))
            return {'status': 'PASS', 'failing': [], 'worst': {'blade_collision': {'slack': .001, 'sample': 0}}}

    monkeypatch.setattr(motion_preflight, 'MotionChecker', Checker)
    monkeypatch.setattr(generalization, '_variant_task', lambda task, removed: SimpleNamespace(
        bumps=tuple(p for p in task.protrusion_ids if p not in removed)))
    task = SimpleNamespace(protrusion_ids=('p0', 'p1'))
    labels = ['HOME', 'SKIM:p0', 'WITHDRAW:p0', 'SKIM:p1', 'RETURN_HOME', 'HOLD_AFTER_STOP']
    rows = [{'policy_phase': label, 'measured': {'q': [np.zeros(7).tolist()]}, 'submitted': {'cut_enabled': False}}
            for label in labels]
    out = live_mod.measured_path_clearance_staged(task, rows)
    assert out['status'] == 'PASS'
    assert [c['removed_protrusions'] for c in out['chunks']] == [[], ['p0'], ['p0', 'p1']]
    assert seen == [(('p0', 'p1'), labels[:3]), (('p1',), labels[3:4]), ((), labels[4:])]

