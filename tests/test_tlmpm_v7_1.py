"""Finite interface evidence and shared v7 kinematics for the opt-in candidate."""
import numpy as np
import pytest
import warp as wp

from robotic_cutting_sim import tlmpm_v7 as V, tlmpm_v7_1 as C


@pytest.mark.parametrize('mode', ['contact', 'no_contact', 'no_sweep', 'disarmed', 'disarmed_pending', 'stop', 'outside_edge', 'rotation', 'expired', 'pending_rotation', 'pending_outside', 'pending_band', 'pending_side'])
def test_completion_requires_finite_sweep_and_actual_endpoint_work(mode):
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V2_Y
    a = lambda value,dtype: wp.array(value,dtype=dtype,device='cpu')
    height = .01 if mode == 'no_contact' else .0002
    y = .1 if mode in ('outside_edge','pending_outside') else 0.
    x = a([[.0001,y,-height],[.0001,y,height]],wp.vec3)
    velocity = wp.zeros(2,dtype=wp.vec3,device='cpu')
    mass = a([1.5e-6]*2,float)
    old_codes = a([0,0],int); codes = a([0,0],int); reached = a([1,0],int)
    nodes = a([[0]*8,[0]*8],int); weights = a([[1.]*8]*2,float)
    gradients = wp.zeros((2,8),dtype=wp.vec3,device='cpu')
    offsets = a([0,2],int); neighbours = a([0,1],int)
    previous = a([[0.,0.,0.,0.,0.,0.,1.]],wp.transform)
    shift = .004 if mode == 'expired' else (0. if mode=='no_sweep' else .0002)
    poses = a([[shift,0.,0.,0.,0.,0.,1.]],wp.transform)
    twists = a([[0. if mode=='stop' else .01,0.,0.,0.,0.,1. if mode in ('rotation','pending_rotation') else 0.]],wp.spatial_vector)
    cuts = a([-1 if mode in ('disarmed','disarmed_pending') else 0],int); index = a([0],int); stroke = a([1],int)
    segment = a([1,1],int)
    polygon,_ = RULER_KNIFE_V2_Y.tlmpm_profile(); profile = a(polygon,wp.vec2)
    seeded = mode in ('expired','disarmed_pending') or mode.startswith('pending_')
    pending_q = a([1 if seeded else -1,-1],int)
    pending_cut = a([0 if seeded else -1,-1],int)
    diagnostics = wp.zeros(2,dtype=int,device='cpu')
    original_positions = x.numpy().copy()
    if mode=='pending_side':reached.assign(np.array([2,0],np.int32))
    args=[x,x,velocity,mass,old_codes,codes,reached,nodes,weights,gradients,
        offsets,neighbours,poses,previous,twists,cuts,index,stroke,segment,2,.06,
        .00005 if mode=='pending_band' else .024,.003383,.00075,
        profile,len(polygon),.00045,.0001,.3,1.,pending_q,pending_cut,diagnostics]
    wp.launch(C._complete_links,2,args,device='cpu')
    assert codes.numpy().tolist() == ([1,0] if mode=='contact' else [0,0])
    np.testing.assert_array_equal(x.numpy(),original_positions)
    if seeded:assert pending_q.numpy()[0] == -1
    if mode.startswith('pending_'):
        # Return to the bevel without a new crossing: the expired event must
        # not complete a digit, even though contact work is now available.
        x.assign(np.array([[.0001,0.,-height],[.0001,0.,height]],np.float32))
        twists.assign(np.array([[.01,0.,0.,0.,0.,0.]],np.float32))
        reached.assign(np.array([1,0],np.int32))
        wp.copy(previous,poses)
        args[21]=.024
        wp.launch(C._complete_links,2,args,device='cpu')
        assert codes.numpy().tolist()==[0,0]


def test_candidate_retains_exact_v7_uncut_cpu_motion_and_deformation():
    from robotic_cutting_sim.blade_probe import block
    from robotic_cutting_sim.material_config import TLMPMConfig
    points,masses=block(size_m=(.024,.012,.008),spacing=.002)
    config=TLMPMConfig(cuts=1,floor=False,fixture_walls=False,tool=False,material_contact=False,gravity=(0.,0.,0.))
    beds=[cls(points,masses,.002,config,'cpu') for cls in (V.TLMPM,C.TLMPM)]
    for bed in beds:
        bed.v.assign(np.tile([.01,.02,0.],(len(points),1)).astype(np.float32))
        for _ in range(10):bed.step(advance=False)
    for name in ('positions','velocities','F','Fpi','codes','contact_work_j'):
        np.testing.assert_array_equal(beds[0].snapshot()[name],beds[1].snapshot()[name])
    receipt=beds[1].solver_receipt()
    assert receipt['solver']=='tlmpm_v7_1'
    assert receipt['contact']=='velocity_projection_tau_v1'
    assert receipt['position_update']=='grid_musl_v1'
    assert receipt['link_completion_counts']['completed_side_digits']==0


@pytest.mark.cpu_sim
@pytest.mark.parametrize('control',['no_contact','side_push','flat_press','blunt','stop','withdraw'])
def test_candidate_physical_negatives_use_actual_candidate_class(control,monkeypatch):
    from robotic_cutting_sim.blade_probe import drive_blade,Phase,VERTICAL_CUT,FLAT_SKIM
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    rotation,start,phases,extra={
        'no_contact':(VERTICAL_CUT,[.001,0.,.042],[Phase((.05,0.,0.),.1),Phase((0.,0.,-.05),.1)],{}),
        'side_push':(VERTICAL_CUT,[-.034,0.,.002],[Phase((.05,0.,0.),.3)],{}),
        'flat_press':(FLAT_SKIM,[.010,0.,.014],[Phase((0.,0.,-.02),.3)],{}),
        'blunt':(VERTICAL_CUT,[-.05,0.,.022],[Phase((0.,0.,-.05),.2)],
                 {'obstacle_boxes_edge':[[[0.,0.,.05],[.006,.006,.006]]]}),
        'stop':(VERTICAL_CUT,[.001,0.,.016],[Phase((0.,0.,-.05),.2),Phase((0.,0.,0.),.2)],{}),
        'withdraw':(VERTICAL_CUT,[.001,0.,.016],[Phase((0.,0.,-.05),.2),Phase((0.,0.,.05),.2)],{}),
    }[control]
    # Test-local constructor injection; production solver selection is never
    # swapped. Verify the actual class receipt rather than trusting the driver id.
    monkeypatch.setattr(V,'TLMPM',C.TLMPM)
    result=drive_blade(RULER_KNIFE_V1,rotation,start,phases,cutting_law=8,solver_version=7,
                       device='cpu',record_history=True,**extra)
    assert result['solver_module']=='robotic_cutting_sim.tlmpm_v7_1'
    assert result['solver_receipt']['solver']=='tlmpm_v7_1'
    count,_=mechanical_components_packed(result['reference_stencil_nodes'],result['reference_stencil_weights'],
        result['state']['codes'],1,gradients=result['reference_stencil_gradients'])
    assert count==1
    if control in ('stop','withdraw'):
        first,second=result['labelled_after_phase']
        assert first>0 and second==first
        assert not result['state']['codes'][result['points'][:,2]<.0059].any()
    else:
        assert not result['state']['codes'].any()
        assert result['solver_receipt']['link_completion_counts']['completed_side_digits']==0
    assert np.isfinite(result['state']['positions']).all()
    assert np.linalg.det(result['state']['F']).min()>0.


def test_public_switch_keeps_the_explicit_candidate_class(tmp_path):
    import json
    from pathlib import Path
    from robotic_cutting_sim.live_env import open_cutting_env
    from robotic_cutting_sim.scene_config import compile_task
    scene=json.loads((Path(__file__).resolve().parents[1]/'configs/wp12/scenes/bar_flat_board.json').read_text())
    scene['solver']['particle_spacing_m']=.004
    with open_cutting_env(runtime='kinematic',scene=scene,goal='sever_mid',
                          preset='cutting_research_v1',device='cpu',output_dir=tmp_path/'run') as env:
        env.reset(seed=42)
        scene['solver']['module']='tlmpm_v7_1'
        env.switch_task(task=compile_task(scene,'sever_mid',seed=42))
        env.reset(seed=42)
        assert type(env.backend.material.bed) is C.TLMPM
        assert env.backend.material.parameter_receipt['tlmpm_v7_1']['solver']=='tlmpm_v7_1'
