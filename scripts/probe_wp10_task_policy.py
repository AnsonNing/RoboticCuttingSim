#!/usr/bin/env python3
"""Full task/controller probe with ideal edge tracking; never a robot episode."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim.control import ActionLimits
from robotic_cutting_sim.kinematic_env import make_kinematic_env
from robotic_cutting_sim.policies import WaypointPolicy, through_cut, skim
from robotic_cutting_sim.recording import new_output, manifest, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',default='TRIM-01-R')
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--spacing',type=float,default=.001)
    p.add_argument('--saw',type=float,default=0.)
    p.add_argument('--scenario')
    p.add_argument('--cutting-law',type=int,choices=(1,2,3,4,5),help='None keeps the task default (v2 tasks: law 2)')
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=new_output(a.out);start=time.monotonic()
    write_json(out/'manifest.json',manifest({**vars(a),'out':str(a.out),'seed':42},'kinematic_ideal_tracking'))
    env=make_kinematic_env(a.task,device=a.device,particle_spacing_m=a.spacing,scenario=a.scenario,cutting_law=a.cutting_law,
        action_limits=ActionLimits(),ik_failure='reject',delta_frame='world',history_stride=10)
    observation,_=env.reset()
    plan=skim(env.task,saw_amplitude_m=a.saw) if env.task.goal_kind=='flatten' else through_cut(env.task,saw_amplitude_m=a.saw)
    policy=WaypointPolicy(env.task,plan,env.action_limits)
    with (out/'actions.jsonl').open('w') as stream:
        for step in range(a.steps):
            action=policy(observation);observation,reward,terminated,truncated,info=env.step(action)
            write=dict(step=step+1,time_s=env.time_s,action={**action,'target':np.asarray(action['target']).tolist()},info=info)
            import json
            stream.write(json.dumps(write)+'\n')
            if terminated or truncated:break
    strict=env.evaluate();research=env.evaluate(profile='research')
    np.savez_compressed(out/'physics_state.npz',time_s=[s['time_s'] for s in env._history],
        **{key:np.asarray([s[key] for s in env._history]) for key in ('positions','velocities','codes')},
        mass_kg=env.backend.material.masses,reference_positions=env.backend.material.world_points,
        reference_stencil_nodes=env.backend.material.bed.nodes.numpy(),reference_stencil_weights=env.backend.material.bed.weights.numpy(),
        reference_stencil_gradients=env.backend.material.bed.gradients.numpy(),**env.backend.law4_stroke_arrays())
    write_json(out/'result.json',dict(status='PASS_EXECUTION',runtime_source=env.runtime_source,
        cutting_law=env.backend.material.cutting_law,material_receipt=env.backend.material.parameter_receipt,
        task_status=strict['status'],strict=strict,research=research,wall_s=time.monotonic()-start))
    print(strict['status'],strict.get('metrics',{}).get('removed_reference_above_plane_mass_fraction'),flush=True)


if __name__=='__main__':main()
