"""Measured small rotations keep actual finite/contact evidence; large sweeps refuse."""
import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation
from robotic_cutting_sim import tlmpm_v7 as V, tlmpm_v7_1 as I, tlmpm_v7_2 as C


def completion(module, omega, *, contact=True, sweep=True, armed=True):
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V2_Y
    a=lambda value,dtype:wp.array(value,dtype=dtype,device='cpu')
    h=.0002 if contact else .01
    x=a([[.0001,0.,-h],[.0001,0.,h]],wp.vec3)
    zero=wp.zeros(2,dtype=wp.vec3,device='cpu');mass=a([1.5e-6]*2,float)
    old=a([0,0],int);codes=a([0,0],int);reached=a([1,0],int)
    nodes=a([[0]*8]*2,int);weights=a([[1.]*8]*2,float)
    gradients=wp.zeros((2,8),dtype=wp.vec3,device='cpu')
    previous=a([[0,0,0,0,0,0,1]],wp.transform)
    quat=Rotation.from_rotvec([0,0,omega*.0001]).as_quat()
    poses=a([[.0002 if sweep else 0,0,0,*quat]],wp.transform)
    twists=a([[.01 if sweep else 0,0,0,0,0,omega]],wp.spatial_vector)
    pending=a([-1,-1],int);pending_cut=a([-1,-1],int);diag=wp.zeros(2,dtype=int,device='cpu')
    polygon,_=RULER_KNIFE_V2_Y.tlmpm_profile()
    wp.launch(module._complete_links,2,[x,x,zero,mass,old,codes,reached,nodes,weights,gradients,
        a([0,2],int),a([0,1],int),poses,previous,twists,a([0 if armed else -1],int),a([0],int),
        a([1],int),a([1,1],int),2,.06,.024,.003383,.00075,a(polygon,wp.vec2),len(polygon),
        .00045,.0001,.3,1.,pending,pending_cut,diag],device='cpu')
    np.testing.assert_array_equal(x.numpy(),np.array([[.0001,0.,-h],[.0001,0.,h]],np.float32))
    return codes.numpy()


def test_real_small_rigid_rotation_has_a_finite_worked_interface():
    assert completion(I,.01).tolist()==[0,0]  # immutable translation-only parent
    assert completion(C,.01).tolist()==[1,0]
    assert completion(C,0).tolist()==completion(I,0).tolist()


@pytest.mark.parametrize('omega,options',[(2.,{}),(.01,{'contact':False}),(.01,{'sweep':False}),(.01,{'armed':False})])
def test_no_completion_without_bounded_sweep_contact_and_arming(omega,options):
    assert completion(C,omega,**options).tolist()==[0,0]


@pytest.mark.cpu_sim
@pytest.mark.parametrize('control',['no_contact','side_push','flat_press','blunt','stop','withdraw'])
def test_actual_candidate_physical_negatives(control,monkeypatch):
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
    assert result['solver_module']=='robotic_cutting_sim.tlmpm_v7_2'
    assert result['solver_receipt']['solver']=='tlmpm_v7_2'
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

