"""The direct scene entry must retain the validated material and compiled path."""
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_wp14_scene as S


@pytest.mark.parametrize('name',['standard_cut','firm_offset_cut'])
def test_direct_scene_launch_preserves_compiled_inputs_and_no_fallback(name,tmp_path):
    from robotic_cutting_sim.scene_config import compile_task
    record=S.prepare(name,tmp_path/'new')
    command=record['command']
    value=lambda flag:command[command.index(flag)+1]
    task=compile_task(value('--scene'),value('--goal'),
                      planner_options=json.loads(value('--planner-options')),seed=int(value('--seed')))
    assert task.solver_module=='tlmpm_v7'
    assert record['profile']['id']=='v7_cut_research_v1'
    assert task.controller['planner']['controller_profile']=='path_v2'
    assert json.loads(value('--planner-options'))==task.controller['planner']
    assert int(value('--steps'))*task.control_dt_s==task.duration_s
    assert '--skip-preflight' not in command
    assert '--return-home' in command and '--record-cameras' in command
    with pytest.raises(FileExistsError):S.prepare(name,tmp_path/'new')
