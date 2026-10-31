"""The actual legacy driver must record/execute the compiled planner overrides."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_wp13 as driver


@pytest.mark.parametrize('case_name,depth', [('P00_partial', -.010), ('P01_shallow', -.012)])
def test_compiled_planner_reaches_real_driver_manifest_without_physics(tmp_path, monkeypatch, case_name, depth):
    from robotic_cutting_sim import goal_planner, scene_config
    case = json.loads((driver.ROOT/f'configs/wp13/cases/{case_name}.json').read_text())
    task, info = driver.build(case, 'tlmpm_v7')
    config = driver.E2.pseudo_config(driver.E2.load_protocol())
    args = SimpleNamespace(out=tmp_path/'run', device='cpu', steps=None, expect_task_id=None,
        feasibility_class=None, surface_profile='particle_surface_v1', geometry_level=None, no_geometry=True)
    sample = dict(sample_id=case_name, family='PARTIAL', range_set='wp13', split='wp13',
                  seed=case['seed'], variant='PARTIAL', draws=[], derived={}, envelope_parameters={}, ood_dimensions=[])
    calls = []
    def before_physics(compiled, **kwargs):
        calls.append(kwargs['action_limits'])
        assert compiled is task
        raise RuntimeError('Intentional test stop before constructing physics')
    original_options = goal_planner.planner_options
    original_compile = driver.P5.W.compile_sample
    monkeypatch.setattr(scene_config, 'make_configured_kinematic_env', before_physics)
    assert driver.execute_episode(task, info, config, sample, args, extra={}) == 1
    record = json.loads((args.out/'manifest.json').read_text())['config']
    options = record['planner_options']
    assert options['overtravel_m'] == depth
    assert options['controller_profile'] == 'path_v2'
    assert options['max_translation_m'] == .001
    assert options['max_rotation_deg'] == 1.
    assert calls == [goal_planner.action_limits(task.controller['planner'], task.scene)]
    assert goal_planner.planner_options is original_options
    assert driver.P5.W.compile_sample is original_compile
    # Persisted input and complete compiled options agree; no frozen task mutation.
    assert options == task.controller['planner']
    assert json.loads((args.out/'result.json').read_text())['status'] == 'ABORTED'
