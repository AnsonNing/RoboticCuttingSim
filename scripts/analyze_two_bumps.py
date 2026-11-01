#!/usr/bin/env python3
"""Read-only per-protrusion actual mechanical detachment, never a height score."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from robotic_cutting_sim.cut_topology import mechanical_components_packed
from robotic_cutting_sim.demo_media import sha256_file, load_jsonl
from robotic_cutting_sim.scene_config import compile_task


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--goal',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();start=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
    task=compile_task(a.scene,a.goal)
    d=np.load(a.episode/'physics_state.npz');times=d['time_s'];codes=d['codes'];mass=d['mass_kg']
    if 'reference_positions_local' in d:
        X=d['reference_positions_local']
        reference_source='recorded material-local reference_positions_local'
    elif (a.episode/'final_state.npz').exists():
        final=np.load(a.episode/'final_state.npz');X=final['reference']
        world=X@task.T_world_workpiece[:3,:3].T+task.T_world_workpiece[:3,3]
        if not np.allclose(world,d['reference_positions'],atol=1e-7,rtol=0):
            raise ValueError('Saved material-local reference does not match the world history')
        reference_source='exact recorded material-local final_state.reference, checked against world history'
    else:raise ValueError('Exact material-local reference required; world points cannot rebuild the solver stencil')
    if 'reference_stencil_nodes' in d:
        nodes,weights,gradients=[d[k] for k in ('reference_stencil_nodes','reference_stencil_weights','reference_stencil_gradients')]
    else:
        from robotic_cutting_sim.cut_aware_surface import reference_stencil
        nodes,weights,gradients=reference_stencil(X,task.grid_dx_m)
    rows=load_jsonl(a.episode/'actions.jsonl');plane=task.scene['workpiece']['base_box_m']['height_m']
    base=X[:,2]<=plane
    bumps={r['id']:(X[:,2]>plane)&(b.height(X[:,0],X[:,1])>0) for r,b in zip(task.scene['workpiece']['protrusions'],task.geometry.bumps)}
    phases={}
    for row in rows:
        phase=row.get('policy_phase',row.get('phase'))
        if phase:phases.setdefault(phase,float(row['time_s']))
    checkpoints={'initial':0,'final':len(times)-1}
    for label,phase in [('after_A','REORIENT:B_irregular'),('after_B','RETURN_HOME')]:
        if phase in phases:checkpoints[label]=min(int(np.searchsorted(times,phases[phase])),len(times)-1)
    measurements={};labels_out={}
    for name,index in checkpoints.items():
        count,labels=mechanical_components_packed(nodes,weights,codes[index],task.max_cuts,gradients=gradients)
        base_mass=np.bincount(labels[base],weights=mass[base],minlength=count)
        main=int(np.argmax(base_mass));detached=labels!=main
        by_bump={}
        for key,mask in bumps.items():
            by_bump[key]={'target_particles':int(mask.sum()),'target_mass_kg':float(mass[mask].sum()),
                'detached_fraction':float(mass[mask&detached].sum()/mass[mask].sum()),
                'component_ids':np.unique(labels[mask]).tolist(),
                'detached_component_ids':np.unique(labels[mask&detached]).tolist()}
        measurements[name]={'frame_index':index,'time_s':float(times[index]),'mechanical_components':int(count),
            'component_masses_kg':np.bincount(labels,weights=mass,minlength=count).tolist(),
            'substrate_retained_fraction':float(mass[base&~detached].sum()/mass[base].sum()),
            'detached_below_plane_fraction':float(mass[base&detached].sum()/mass[base].sum()),'bumps':by_bump}
        labels_out[name]=labels
    final=measurements['final'];a_key,b_key=bumps
    if (a.episode/'summary.json').exists():
        expected=json.loads((a.episode/'summary.json').read_text())['metrics']['mechanical_components']
        if final['mechanical_components']!=expected:raise ValueError('Rebuilt final stencil graph differs from live solver measurement')
    checks={'initial_single_body':measurements['initial']['mechanical_components']==1,
        'unchanged_particle_mass_count':len(mass)==len(task.sample_particles()[0]),
        'both_bumps_detached':all(v['detached_fraction']>=.9 for v in final['bumps'].values()),
        'base_retained':final['substrate_retained_fraction']>=.99,
        'limited_detached_substrate':final['detached_below_plane_fraction']<=.01,
        'two_separate_chips':bool(set(final['bumps'][a_key]['detached_component_ids']) and set(final['bumps'][b_key]['detached_component_ids']) and not set(final['bumps'][a_key]['detached_component_ids'])&set(final['bumps'][b_key]['detached_component_ids'])),
        'after_A_recorded':'after_A' in measurements,'after_B_recorded':'after_B' in measurements}
    if 'after_A' in measurements:
        checks['A_detached_before_B']=measurements['after_A']['bumps'][a_key]['detached_fraction']>=.9
        checks['B_still_connected_after_A']=measurements['after_A']['bumps'][b_key]['detached_fraction']<.1
    np.savez_compressed(a.out/'component_labels.npz',**labels_out)
    receipt={'schema':'sequential_two_bump_detachment_v1','status':'PASS' if all(checks.values()) else 'FAIL',
        'checks':checks,'measurements':measurements,'first_phase_times_s':phases,
        'reference_source':reference_source,
        'scope':'Actual positive weight/gradient packed-code reference stencil graph, including code-zero bridges. No height drop substituted for detachment. Native grasp/HOME/contact/quality evaluated independently.',
        'source_sha256':{str(a.episode/'physics_state.npz'):sha256_file(a.episode/'physics_state.npz'),str(a.scene):sha256_file(a.scene),str(a.goal):sha256_file(a.goal)},'wall_s':time.monotonic()-start}
    (a.out/'detachment.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
