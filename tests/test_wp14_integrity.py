"""Compiled command replay must detect altered physical execution receipts."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from summarize_wp14 import replay
from run_wp13 import build


@pytest.mark.parametrize('corruption', [None, 'position', 'angle', 'cut', 'phase'])
def test_replay_audits_actual_recorded_command_prefix(corruption):
    task, _ = build({'id': 'C00_cut', 'episode': 'CUT', 'seed': 42}, 'tlmpm_v7')
    actions = deepcopy(json.loads((ROOT/'tests/fixtures/wp14/c00_recorded_prefix.json').read_text())['rows'])
    row = actions[-1]
    if corruption == 'position':
        row['edge_pose_world'][0] += .001
    elif corruption == 'angle':
        row['edge_pose_world'][3:] = (Rotation.from_rotvec([.01, 0., 0.])*
                                     Rotation.from_quat(row['edge_pose_world'][3:])).as_quat().tolist()
    elif corruption == 'cut':
        row['cut_enabled'] = True
    elif corruption == 'phase':
        row['phase'] = 'WRONG'
    audit = replay(task, actions)
    assert audit['status'] == ('PASS' if corruption is None else 'FAIL')
    assert audit['controls'] == 3


def test_selected_trace_observes_finite_link_sweep_without_changing_solver_inputs():
    import warp as wp
    import diagnose_wp14_front as D
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V2_Y
    device = 'cpu'
    array = lambda value, dtype: wp.array(value, dtype=dtype, device=device)
    ids = array([0, 1], int); partner = array([1, 0], int)
    x = array([[.0001, 0., -.0005], [.0001, 0., .0005]], wp.vec3)
    v = wp.zeros(2, dtype=wp.vec3, device=device)
    old_pose = array([[0.,0.,0.,0.,0.,0.,1.]], wp.transform)
    poses = array([[.0002,0.,0.,0.,0.,0.,1.]], wp.transform)
    saved_pose = wp.zeros(1, dtype=wp.transform, device=device)
    pre = wp.zeros((2,4), dtype=float, device=device)
    work = array([0.,0.], float)
    wp.launch(D.before, 2, [ids,work,x,v,poses,old_pose,saved_pose,pre],device=device)
    work.assign(np.array([1.e-7,0.],np.float32))
    code = array([0,0],int); reached = array([0,0],int)
    progress = array([1.,0.],float)
    cuts = array([0],int); index = array([0],int); stroke = array([1],int)
    segment = array([1,1],int); engaged = array([1],int)
    twists = array([[.01,0.,0.,0.,0.,0.]],wp.spatial_vector)
    polygon,_ = RULER_KNIFE_V2_Y.tlmpm_profile()
    profile = array(polygon,wp.vec2)
    empty = wp.zeros(0,dtype=wp.vec3,device=device)
    step = array([4],int)
    counts = wp.zeros((2,len(D.COUNTERS)),dtype=int,device=device)
    keys = wp.full((2,len(D.SAMPLES)),1.e20,dtype=float,device=device)
    samples = wp.zeros((2,len(D.SAMPLES),len(D.COLUMNS)),dtype=float,device=device)
    inputs = [x,v,code,work,progress,poses,old_pose,twists,cuts,index,stroke,segment,engaged,reached]
    original = [a.numpy().copy() for a in inputs]
    wp.launch(D.observe,2,[ids,partner,x,x,code,work,progress,v,poses,saved_pose,twists,
        cuts,index,stroke,segment,engaged,reached,profile,len(polygon),.06,.024,.0034,.00045,2,
        empty,empty,wp.vec4(0.,0.,1.,0.),step,pre,counts,keys,samples],device=device)
    for a,expected in zip(inputs,original):
        np.testing.assert_array_equal(a.numpy(),expected)
    assert counts.numpy()[:,1].tolist() == [1,1]
    assert counts.numpy()[:,2].tolist() == [1,1]
    assert counts.numpy()[:,9].tolist() == [1,1]
    assert counts.numpy()[:,5].tolist() == [1,0]
    np.testing.assert_allclose(samples.numpy()[:,0,2:4],[[.0001,-.0001]]*2,atol=1e-10)
