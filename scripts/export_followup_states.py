#!/usr/bin/env python3
"""Export four measured Native material states for independent topology checks.

Oracle evidence only; never an observation or initialization geometry.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from robotic_cutting_sim.demo_media import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--detachment',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    rec=json.loads(a.detachment.read_text());z=np.load(a.episode/'physics_state.npz')
    names=['initial','after_A','after_B','final'];indices=[rec['measurements'][n]['frame_index'] for n in names]
    payload={key:z[key] for key in ('reference_positions_local','reference_indices','reference_stencil_nodes',
        'reference_stencil_weights','reference_stencil_gradients','mass_kg')}
    payload.update(checkpoint_names=np.array(names),time_s=z['time_s'][indices],codes=z['codes'][indices],
        positions_world_m=z['positions'][indices],source_frame_indices=np.array(indices),
        source_physics_sha256=np.array(sha256_file(a.episode/'physics_state.npz')),
        oracle_evidence_not_observation=np.array(True),F_recorded=np.array(False))
    np.savez_compressed(a.out,**payload)
    with a.out.with_suffix('.json').open('x') as f:json.dump({'schema':'native_two_bump_measured_states_v1',
        'sha256':sha256_file(a.out),'source_episode':str(a.episode),'source_detachment':str(a.detachment),
        'checkpoints':names,'source_physics_sha256':str(payload['source_physics_sha256']),
        'scope':'Exact recorded Native arrays at four measured times. Complete unchanged particle budget and actual reference stencil. '
                'No inferred F, particle deletion, relabelling or mesh repair; oracle evidence, never a policy observation.'},f,indent=2)


if __name__=='__main__':main()
