"""Normal profile dispatch and Global-only rig selection preserve frozen evidence."""
from copy import deepcopy
from contextlib import contextmanager
import numpy as np
import pytest

from robotic_cutting_sim.scene_config import compile_task, configured_task_payload, task_from_payload
from robotic_cutting_sim.camera_rig import make_rig

SCENE='configs/wp12/scenes/bar_flat_board.json'
PROFILE='v7_cut_research_v1'


def test_normal_profile_compile_matches_frozen_native_task_and_payload():
    from robotic_cutting_sim.policy_env import compile_policy_task
    task=compile_task(SCENE,'sever_mid',solver_profile=PROFILE)
    assert task.task_id=='CFG-bar_flat_board-sever_mid-5da99059'
    assert task.duration_s==14.
    assert compile_policy_task(SCENE,'sever_mid',solver_profile=PROFILE).config_sha256==task.config_sha256
    assert task_from_payload(configured_task_payload(task)).config_sha256==task.config_sha256
    assert compile_task(task.scene,'sever_mid').task_id==task.task_id
    assert compile_task(SCENE,'sever_mid').solver_module=='tlmpm_v5'


def test_profile_tag_cannot_silently_replace_contradictory_solver():
    task=compile_task(SCENE,'sever_mid',solver_profile=PROFILE)
    scene=deepcopy(task.scene);scene['solver']['module']='tlmpm_v5'
    with pytest.raises(ValueError,match='solver/law'):
        compile_task(scene,'sever_mid')
    with pytest.raises(ValueError,match='controller'):
        compile_task(SCENE,'sever_mid',solver_profile=PROFILE,planner_options={'controller_profile':'v1'})
    with pytest.raises(ValueError,match='discretization'):
        compile_task(SCENE,'sever_mid',solver_profile=PROFILE,particle_spacing_m=.002)


def test_normal_environment_scene_dispatch_uses_the_same_compiled_task(tmp_path,monkeypatch):
    from robotic_cutting_sim import live_env
    seen={}
    def session(task,*args):
        seen['task']=task
        yield 'opened'
    monkeypatch.setattr(live_env,'_kinematic_session',session)
    with live_env.open_cutting_env(scene=SCENE,goal='sever_mid',solver_profile=PROFILE,
                                  runtime='kinematic',observation_profile='state',output_dir=tmp_path) as value:
        assert value=='opened'
    assert seen['task'].task_id=='CFG-bar_flat_board-sever_mid-5da99059'
    with pytest.raises(ValueError,match='go together'):
        with live_env.open_cutting_env(scene=SCENE,output_dir=tmp_path):pass


def test_new_rig_changes_only_global_and_preserves_real_hand_local_mounts():
    hands=np.array([[.2,.1,.3,0,0,0,1],[-.3,.2,.5,0,0,1,0]])
    old=make_rig(initial_hand_poses_world_xyzw=hands)
    new=make_rig(initial_hand_poses_world_xyzw=hands,rig_profile='wrist_camera_rig_v6')
    for name in ('wrist_d405_0','wrist_d405_1'):
        assert old.specs[name]==new.specs[name]
        np.testing.assert_array_equal(np.array(new.specs[name]['T_hand_camera'])[:3,:3],np.eye(3))
    g=new.calibrations(hands)['global_d455'];g2=new.calibrations(hands+np.array([.1,0,0,0,0,0,0]))['global_d455']
    assert g['T_world_camera']==g2['T_world_camera']
    assert g['width']==960 and g['height']==720
    assert g['K'][0][0]==g['K'][1][1]
    assert g['K'][0][0]*.095/g['native_near_m']<g['native_maximum_disparity']
    assert .6<g['working_distance_to_target_m']<6.
    assert old.specs['global_d455']['width']==640
    with pytest.raises(ValueError):make_rig(rig_profile='unknown')


def test_fixed_pose_corrections_retain_controlled_v5_global_optics():
    old=make_rig()
    for profile in ('wrist_camera_rig_v7','wrist_camera_rig_v8'):
        rig=make_rig(rig_profile=profile)
        for name in ('wrist_d405_0','wrist_d405_1'):assert rig.specs[name]==old.specs[name]
        for key in ('width','height','K','near_m','far_m','native_near_m','native_far_m'):
            assert rig.specs['global_d455'][key]==old.specs['global_d455'][key]
        assert rig.specs['global_d455']['T_world_camera']!=old.specs['global_d455']['T_world_camera']
