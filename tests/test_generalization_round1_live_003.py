"""WP-11 Phase E live_003 declaration (configs/wp11/generalization_round1_live_003.json)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_003_declaration_inherits_live_001_and_declares_only_test_c():
    live_mod = _script('run_generalization_live')
    live, config = live_mod.load_live(ROOT/'configs/wp11/generalization_round1_live_003.json')
    base, _ = live_mod.load_live()
    assert live['stage_id'] == 'wp11_phaseE_round1_live_003' and live['status'] == 'PREDECLARED'
    assert [s['sample_id'] for s in live['samples']] == ['E1_TEST_C_s1602']
    own = live['samples'][0]
    assert own == {**live_mod.sample_spec(base, 'E1_TEST_C_s1602'), 'purpose': own['purpose'], 'priority': 1}
    for key in ('task', 'controller', 'schedule', 'runtime', 'success_criteria', 'outcome_classes', 'precedence',
                'home_outcome', 'failure_policy', 'round_file_sha256'):
        assert live[key] == base[key]
    assert live['preflight']['mode'] == 'runner_default_staged'   # not round1_staged: no --skip-plan-preflight
