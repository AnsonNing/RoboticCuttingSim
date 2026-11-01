"""A scoped profile must refuse unsupported tasks and runtime substitutions."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_wp13 import build
from robotic_cutting_sim import research_profiles as P


def proposed(name='C00_cut'):
    row = json.loads((ROOT/f'configs/wp14/cases/{name}.json').read_text())
    task,_ = build(row,'tlmpm_v7')
    scene=task.scene;scene['solver']['research_profile']='v7_cut_research_v1'
    return replace(task,scene_json=json.dumps(scene))


@pytest.mark.parametrize('name',['C00_cut','C01_aspect_firm','H00_cut','H01_cut'])
def test_proposed_configuration_envelope_is_independent_of_outcomes(name):
    task=proposed(name)
    receipt=P.validate_runtime(task,'tlmpm_v7',8,.0001,4,8,1.)
    assert receipt['front']=='finite_relative_sweep_v1'
    assert receipt['contact']=='velocity_projection_tau_v1'
    assert len(receipt['source_sha256']['tlmpm_v7.py'])==64


@pytest.mark.parametrize('change',['solver','law','geometry','material','size','placement','normal','location','goal','controller','spacing','threshold'])
def test_profile_refuses_unsupported_configuration(change):
    task=proposed();scene=task.scene;goal=task.goal;options=task.controller['planner'];spacing=.001
    if change=='solver':scene['solver']['module']='tlmpm_v5'
    elif change=='law':scene['solver']['cutting_law']=5
    elif change=='geometry':scene['workpiece']['protrusions']=[{'id':'unexpected'}]
    elif change=='material':scene['workpiece']['material_profile']='wp12_soft_elastic_v1'
    elif change=='size':scene['workpiece']['base_box_m']['width_m']=.06
    elif change=='placement':scene['workpiece']['world_pose']['yaw_deg']=20.
    elif change=='normal':goal['planes'][0]['normal_local']=[.99,0.,.1]
    elif change=='location':goal['planes'][0]['origin_local_m'][0]=.02
    elif change=='goal':goal['type']='trim_plane'
    elif change=='controller':options['controller_profile']='v1'
    elif change=='spacing':spacing=.002
    elif change=='threshold':goal['plane_error_bound_m']=.01
    with pytest.raises(ValueError,match='v7_cut_research_v1:'):
        P.validate_task(scene,goal,options,spacing)


@pytest.mark.parametrize('solver,law,dt,cuts,slots,efficiency',[
    ('tlmpm_v5',8,.0001,4,8,1.),('tlmpm_v7',5,.0001,4,8,1.),
    ('tlmpm_v7',8,.00005,4,8,1.),('tlmpm_v7',8,.0001,2,8,1.),
    ('tlmpm_v7',8,.0001,4,16,1.),('tlmpm_v7',8,.0001,4,8,2.)])
def test_profile_rejects_runtime_overrides(solver,law,dt,cuts,slots,efficiency):
    with pytest.raises(ValueError,match='runtime'):
        P.validate_runtime(proposed(),solver,law,dt,cuts,slots,efficiency)


@pytest.mark.parametrize('name',['H02_cut','H03_cut'])
def test_failed_predeclared_boundaries_remain_outside_adopted_configuration(name):
    task=proposed(name)
    with pytest.raises(ValueError,match='outside profile'):
        P.validate_task(task.scene,task.goal,task.controller['planner'],.001)


def test_unmeasured_half_grid_location_is_refused_without_changing_evaluator():
    task=proposed();goal=task.goal;goal['planes'][0]['origin_local_m'][0]=.002
    assert goal['plane_error_bound_m']==.002
    with pytest.raises(ValueError,match='lattice phase'):
        P.validate_task(task.scene,goal,task.controller['planner'],.001)


def test_profile_refuses_changed_solver_source_binding(monkeypatch):
    original=P.document('v7_cut_research_v1');changed=deepcopy(original)
    changed['source_sha256']['tlmpm_v7.py']='0'*64
    monkeypatch.setattr(P,'document',lambda _:changed)
    task=proposed()
    with pytest.raises(ValueError,match='validated solver source differs'):
        P.validate_task(task.scene,task.goal,task.controller['planner'],.001)


def test_no_profile_retains_existing_selection():
    task,_=build({'episode':'CUT','seed':42},'tlmpm_v5')
    assert P.validate_runtime(task,'tlmpm_v5',8,.0001,4,8,1.) is None


def test_profile_rejects_effective_material_override():
    from robotic_cutting_sim.task_scenarios import configure_task
    task=configure_task(proposed(),material_profile_id='cheddar_provisional_v1')
    with pytest.raises(ValueError,match='effective material'):
        P.validate_runtime(task,'tlmpm_v7',8,.0001,4,8,1.)


def test_profile_rejects_direct_material_parameter_override():
    task=proposed()
    task=replace(task,material=replace(task.material,friction=.9))
    with pytest.raises(ValueError,match='effective material'):
        P.validate_runtime(task,'tlmpm_v7',8,.0001,4,8,1.)


def test_public_profile_factory_and_runtime_payload_roundtrip():
    from robotic_cutting_sim.scene_config import configured_task_payload,task_from_payload
    task=P.compile_research_task('v7_cut_research_v1','configs/wp12/scenes/bar_flat_board.json','sever_mid')
    assert task.solver_module=='tlmpm_v7'
    assert task.duration_s==14.
    assert task.scene['solver']['research_profile']=='v7_cut_research_v1'
    assert task_from_payload(configured_task_payload(task)).task_id==task.task_id


def test_runtime_profile_selection_refuses_silent_solver_override(tmp_path):
    from robotic_cutting_sim.scene_config import make_configured_kinematic_env
    task=P.compile_research_task('v7_cut_research_v1','configs/wp12/scenes/bar_flat_board.json','sever_mid')
    env=make_configured_kinematic_env(task,device='cpu',solver='tlmpm_v5')
    with pytest.raises(ValueError,match='runtime solver'):
        env.reset(seed=42)
    env.close()
