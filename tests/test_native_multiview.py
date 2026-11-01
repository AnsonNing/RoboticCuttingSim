"""Camera provenance and VFR packet timing must survive layout changes."""
from copy import deepcopy
from pathlib import Path
import sys
import json
import subprocess
import numpy as np
import pytest
from robotic_cutting_sim.native_multiview import check_set,CAMERAS


def frames():
    out={}
    for name in CAMERAS:
        mount=np.eye(4);mount[:3,3]=[-.085,.045,.020]
        out[name]=dict(camera_id=np.array(name),live_source=np.array('Isaac Sim6.2 Kit/PhysX/RTX'),
            physics_timestamp_s=np.array(.2),surface_timestamp_s=np.array(.2),
            camera_extrinsics_timestamp_s=np.array(.2),native_physics_timestamp_s=np.array(2.206),
            rgb=np.zeros((8,12,3),np.uint8),depth_m=np.full((8,12),.2),valid_mask=np.ones((8,12),bool),
            K=np.eye(3),T_world_camera=mount if name.startswith('wrist') else np.eye(4))
        if name.startswith('wrist'):out[name].update(T_world_hand=np.eye(4),T_hand_camera=mount)
    return out


@pytest.mark.parametrize('change',['timestamp','native_clock','identity','extrinsics','moving_global','mount'])
def test_mismatched_camera_or_clock_refuses_remaster(change):
    first=frames();current=deepcopy(first)
    if change=='timestamp':current['wrist_d405_1']['surface_timestamp_s']=.19
    elif change=='native_clock':current['wrist_d405_0']['native_physics_timestamp_s']=2.20
    elif change=='identity':current['wrist_d405_1']['camera_id']=np.array('overview_rgb')
    elif change=='extrinsics':current['wrist_d405_0']['T_world_camera']=np.eye(4)
    elif change=='moving_global':current['global_d455']['T_world_camera'][0,3]=.1
    elif change=='mount':current['wrist_d405_1']['T_hand_camera'][0,3]=0.
    with pytest.raises(ValueError):check_set(current,first)


def test_truthful_parked_wrist_may_have_no_valid_depth():
    f=frames();f['wrist_d405_1']['valid_mask'][:]=False;f['wrist_d405_1']['depth_m'][:]=0
    assert check_set(f,f)['valid_fraction']['wrist_d405_1']==0.


def test_archived_capture_gap_is_explicit_and_current_recording_still_refuses_it():
    initial=frames();f=deepcopy(initial);f['wrist_d405_0']=None
    with pytest.raises(ValueError,match='missing current'):check_set(f,initial)
    assert check_set(f,initial,archived=True)['missing_cameras']==['wrist_d405_0']


def test_optional_actual_closeup_requires_the_same_capture_timestamp(tmp_path):
    from robotic_cutting_sim.native_multiview import read_closeup
    rgb=np.full((16,24,3),127,np.uint8)
    np.savez(tmp_path/'closeup_rgb_0003.npz',rgb=rgb,physics_timestamp_s=.6)
    np.testing.assert_array_equal(read_closeup(tmp_path,3,.6)['rgb'],rgb)
    with pytest.raises(ValueError,match='timestamp'):read_closeup(tmp_path,3,.8)


def test_irregular_capture_all_frames_and_exact_physics_pts(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from encode_wp13_media import encode_timed_frames
    files=[];times=[0.,.2,.3,.44]
    for i,t in enumerate(times):
        p=tmp_path/f'global_d455_{i:04d}.npz'
        np.savez(p,rgb=np.full((16,24,3),i*60,np.uint8),physics_timestamp_s=t);files.append(p)
    def transform(path):
        with np.load(path) as data:return data['rgb']
    receipt=encode_timed_frames(files,tmp_path/'clip.mp4',transform,times)
    assert receipt['pts_check']=='PASS'
    assert receipt['decoded_pts_relative_s']==times
    assert receipt['frames']==4
