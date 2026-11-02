"""Explicit camera version and measured-state matching for the media refresh."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from robotic_cutting_sim.camera_rig import make_rig


def test_v9_physical_distance_and_wrist_identity():
    old, new = make_rig(), make_rig(rig_profile='wrist_camera_rig_v9')
    for name in ('wrist_d405_0', 'wrist_d405_1'):
        assert new.specs[name] == old.specs[name]
    global_view = new.specs['global_d455']
    eye = np.asarray(global_view['T_world_camera'])[:3, 3]
    target = np.asarray(global_view['look_at_target_world_m'])
    assert np.linalg.norm(eye-target) == pytest.approx(.60, abs=1e-9)
    forward = np.asarray(global_view['T_world_camera'])[:3, 2]
    assert np.dot(forward, (target-eye)/.60) == pytest.approx(1.)
    for key in ('K', 'width', 'height', 'native_near_m', 'native_far_m'):
        assert global_view[key] == old.specs['global_d455'][key]
    assert global_view['near_m'] == global_view['native_near_m'] == .52
    assert make_rig().specs == old.specs


def test_v9_target_translation_is_frozen_at_scene_creation():
    target=np.array([.02,-.01,.12])
    rig=make_rig(rig_profile='wrist_camera_rig_v9',global_target_world_m=target)
    view=rig.specs['global_d455']
    eye=np.asarray(view['T_world_camera'])[:3,3]
    np.testing.assert_allclose(view['look_at_target_world_m'],target)
    assert np.linalg.norm(eye-target)==pytest.approx(.6,abs=1e-9)


def test_normal_scene_launcher_only_changes_explicit_camera_selection(tmp_path):
    spec=importlib.util.spec_from_file_location('wp14_scene_global60',Path('scripts/run_wp14_scene.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=module.prepare('standard_cut',tmp_path/'old')
    new=module.prepare('standard_cut',tmp_path/'new',camera_rig_profile='wrist_camera_rig_v9')
    assert old['task_id']==new['task_id']
    assert old['profile']==new['profile']
    assert old['source_inputs']==new['source_inputs']
    assert old['camera_rig']=='wrist_camera_rig_v5'
    assert new['camera_rig']=='wrist_camera_rig_v9'
    assert new['command'][-2:]==['--camera-rig-profile','wrist_camera_rig_v9']


def test_replay_refuses_interpolation_and_missing_state():
    spec = importlib.util.spec_from_file_location('replay_global', Path('scripts/rerender_native_global.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.exact_index([0., .2, .4], .2) == 1
    with pytest.raises(ValueError, match='interpolation refused'):
        module.exact_index([0., .2, .4], .3)


def test_replay_cannot_substitute_a_surface_when_only_positions_exist(tmp_path):
    spec=importlib.util.spec_from_file_location('replay_global',Path('scripts/rerender_native_global.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    np.savez(tmp_path/'physics_state.npz',positions=np.zeros((1,1,3)))
    with pytest.raises(ValueError,match='no substitute surface'):
        module.require_saved_surfaces(tmp_path,[{'frame':0,'time_s':0.}])


def test_exact_staged_surface_snapshot_rejects_stale_or_modified_geometry(tmp_path):
    import hashlib
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('wp10_surface_save',Path('scripts/run_wp10_live.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    faces=np.array([[0,1,2]],np.int32)
    backend=SimpleNamespace(_surface_time=.2,surface_arrays=(vertices,faces),
        surface_receipt={'surface_sha256':hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()})
    module.save_surface_snapshot(backend,tmp_path,0,.2)
    with np.load(tmp_path/'surface_debug/surface_0000.npz') as data:
        np.testing.assert_array_equal(data['vertices_m'],vertices)
        np.testing.assert_array_equal(data['triangles'],faces)
    with pytest.raises(ValueError,match='stale'):
        module.save_surface_snapshot(backend,tmp_path,1,.4)
    vertices[0,0]=.1
    with pytest.raises(ValueError,match='differ'):
        module.save_surface_snapshot(backend,tmp_path,1,.2)


def test_v10_reference_low_angle_is_actual_narrow_lens_with_fixed_wrists():
    target=np.array([.02,-.01,.12])
    old=make_rig()
    rig=make_rig(rig_profile='wrist_camera_rig_v10',global_target_world_m=target)
    g=rig.specs['global_d455']
    eye=np.asarray(g['T_world_camera'])[:3,3]
    direction=(eye-target)/np.linalg.norm(eye-target)
    reference=np.array([.38,-.68,.24]);reference/=np.linalg.norm(reference)
    np.testing.assert_allclose(direction,reference,atol=1e-12)
    assert np.linalg.norm(eye-target)==pytest.approx(.60,abs=1e-12)
    assert 16 < np.degrees(np.arcsin(direction[2])) < 18
    hfov=np.degrees(2*np.arctan(g['width']/(2*g['K'][0][0])))
    assert hfov==pytest.approx(24.)
    assert g['K'][0][0]==g['K'][1][1]
    assert g['K'][0][0]*.095/g['native_near_m']<g['native_maximum_disparity']
    for name in ('wrist_d405_0','wrist_d405_1'):
        assert rig.specs[name]==old.specs[name]
    assert make_rig().specs==old.specs
    with pytest.raises(ValueError,match='finite world XYZ'):
        make_rig(rig_profile='wrist_camera_rig_v10',global_target_world_m=[0,float('nan'),0])


def test_frontal_view_uses_measured_base_and_workspace_and_rotates_with_them():
    from scipy.spatial.transform import Rotation
    from robotic_cutting_sim.camera_rig import arm0_front_view,pose_matrix
    base=np.array([-.46,.12,-.01,0,0,0,1.])
    target=np.array([0.,0.,.03])
    T,receipt=arm0_front_view(base,target,.6,17.)
    base_T=pose_matrix(base)
    camera_local=(np.linalg.inv(base_T)@T)[:3,3]
    assert camera_local[0]>0
    bearing=target[:2]-base[:2]
    # Same bearing from base through workarea to camera, facing back.
    offset=T[:2,3]-target[:2]
    assert bearing[0]*offset[1]-bearing[1]*offset[0]==pytest.approx(0.,abs=1e-12)
    assert np.dot(T[:2,2],bearing)<0
    assert np.linalg.norm(T[:3,3]-target)==pytest.approx(.6)
    assert receipt['workarea_azimuth_from_base_positive_X_deg']==pytest.approx(-14.6208739886)
    rotation=Rotation.from_euler('z',73,degrees=True)
    moved_base=np.r_[rotation.apply(base[:3]),rotation.as_quat()]
    moved,_=arm0_front_view(moved_base,rotation.apply(target),.6,17.)
    np.testing.assert_allclose(moved[:3,3],rotation.apply(T[:3,3]),atol=1e-12)
    np.testing.assert_allclose(moved[:3,2],rotation.apply(T[:3,2]),atol=1e-12)
    with pytest.raises(ValueError,match='behind'):
        arm0_front_view(base,[-1.,.12,.03],.6,17.)
    old=make_rig();new=make_rig(rig_profile='wrist_camera_rig_v12')
    for name in ('wrist_d405_0','wrist_d405_1'):
        assert old.specs[name]==new.specs[name]
    g=new.specs['global_d455']
    assert g['render_near_m']==.001
    assert g['native_near_m']==.52
