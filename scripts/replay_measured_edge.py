#!/usr/bin/env python3
"""Replay a recorded live episode's MEASURED edge poses through the material only.

The input is an episode directory written by run_wp10_live.py (actions.jsonl).
Each control step drives a fresh LiveCuttingMaterial with the measured tool pose
the live material received and that step's recorded cut digit, then scores the
result with the frozen task oracle. This isolates the cutting law from the robot:
no IK, grasp, robot dynamics or cameras run here, finger-pad obstacles are not
replayed (handle only) and there is no reaction feedback into the trajectory.
Results are labelled ``measured_trajectory_material_replay``; they never count as
a live episode.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.kinematic_env import make_kinematic_env
from robotic_cutting_sim.recording import manifest, new_output, write_json


def recorded_steps(episode):
    for line in (Path(episode) / 'actions.jsonl').open():
        row = json.loads(line)
        material = row['privileged_diagnostics']
        yield row['time_s'], np.asarray(material['measured_tool_pose'], float), bool(material['active_cut']), int(material['cut_index'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--scenario')
    p.add_argument('--cutting-law', type=int, choices=(1, 2, 3, 4, 5, 6, 7))
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out, start = new_output(a.out), time.monotonic()
    write_json(out / 'manifest.json', manifest({**vars(a), 'episode': str(a.episode), 'out': str(a.out)},
                                               'measured_trajectory_material_replay'))
    env = make_kinematic_env(a.task, device=a.device, scenario=a.scenario, cutting_law=a.cutting_law, history_stride=10)
    env.reset()
    steps = 0
    for time_s, pose, active, digit in recorded_steps(a.episode):
        action = dict(mode='cartesian_edge_pose_world', target=pose, cut_enabled=active, cut_index=max(digit, 0))
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if not np.isclose(env.time_s, time_s, atol=1e-9):
            raise ValueError('Replay clock diverged from the recorded episode')
        if terminated or truncated:
            break
    strict, research = env.evaluate(), env.evaluate(profile='research')
    np.savez_compressed(out / 'physics_state.npz', time_s=[s['time_s'] for s in env._history],
        **{key: np.asarray([s[key] for s in env._history]) for key in ('positions', 'velocities', 'codes')},
        mass_kg=env.backend.material.masses, reference_positions=env.backend.material.world_points,
        reference_stencil_nodes=env.backend.material.bed.nodes.numpy(),
        reference_stencil_weights=env.backend.material.bed.weights.numpy(),
        reference_stencil_gradients=env.backend.material.bed.gradients.numpy(), **env.backend.law4_stroke_arrays())
    write_json(out / 'result.json', dict(status='PASS_EXECUTION', runtime_source='measured_trajectory_material_replay',
        source_episode=str(a.episode), steps=steps, cutting_law=env.backend.material.cutting_law,
        task_status=strict['status'], strict=strict, research=research, wall_s=time.monotonic() - start))
    print(strict['status'], strict.get('metrics', {}).get('final_components'), flush=True)


if __name__ == '__main__':
    main()
