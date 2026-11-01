"""Independent graph checks of the small public actual-Native evidence bundle.

This audits one saved episode; it is not another physics trial or success rate.
"""
from pathlib import Path
import numpy as np
from robotic_cutting_sim.cut_topology import mechanical_components_packed
from robotic_cutting_sim.scene_config import compile_task

ROOT=Path(__file__).resolve().parents[1]


def test_recorded_native_two_bumps_are_continuous_then_two_separate_chips():
    z=np.load(ROOT/'docs/evidence/wp14_followup/native_two_bumps_states_v1.npz')
    task=compile_task(ROOT/'configs/wp14_followup/two_bumps_scene_rigid_v2.json',
                      ROOT/'configs/wp14_followup/two_bumps_goal_v1.json')
    reference=z['reference_positions_local'];mass=z['mass_kg'];codes=z['codes']
    np.testing.assert_array_equal(reference,task.sample_particles()[0])
    assert codes.shape==(4,len(reference)) and len(reference)==103933
    assert not codes[0].any() and np.isfinite(z['positions_world_m']).all()
    assert bool(z['oracle_evidence_not_observation']) and not bool(z['F_recorded'])
    base=reference[:,2]<=.02
    targets=[(reference[:,2]>.02)&(b.height(reference[:,0],reference[:,1])>0) for b in task.geometry.bumps]
    counts=[];removed=[]
    for code in codes:
        count,labels=mechanical_components_packed(z['reference_stencil_nodes'],z['reference_stencil_weights'],
            code,2,gradients=z['reference_stencil_gradients'])
        counts.append(count);main=np.bincount(labels[base],weights=mass[base]).argmax()
        assert mass[base&(labels==main)].sum()/mass[base].sum()>=.99
        removed.append([mass[t&(labels!=main)].sum()/mass[t].sum() for t in targets])
    assert counts==[1,2,3,3]
    np.testing.assert_array_equal(removed,[[0,0],[1,0],[1,1],[1,1]])
    for cut in (0,1):
        digits=(codes>>(2*cut))&3
        assert not (digits==3).any()
        assert not np.any((digits[:-1]!=0)&(digits[1:]!=digits[:-1]))
