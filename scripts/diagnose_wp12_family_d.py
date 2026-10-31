#!/usr/bin/env python3
"""Read-only Family D diagnosis of saved benchmark runs; never executes or rescores a task.

Uses the first ID and OOD sample in declared seed order. Reference-neighbour
interfaces below are diagnostics, not the evaluator's eight-node graph.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def outcome_failures(evaluation):
    outcome = evaluation.get('outcome') or {}
    return {'failed_checks': [k for k, v in (outcome.get('checks') or {}).items() if v is False],
            'steps': [outcome_failures(s) for s in outcome.get('steps', [])]}


def diagnose(run):
    result, manifest = read(run/'result.json'), read(run/'manifest.json')
    scene, goal = manifest['config']['scene'], manifest['config']['goal']
    pose = scene['workpiece']['world_pose']
    rotation = Rotation.from_euler('z', pose['yaw_deg'], degrees=True).as_matrix()
    origin = np.asarray(pose['translation_m'])
    actions = [json.loads(line) for line in (run/'actions.jsonl').read_text().splitlines()]
    with np.load(run/'physics_state.npz') as state:
        times, codes = state['time_s'], state['codes']
        reference = (state['reference_positions']-origin)@rotation
        positions = (state['positions']-origin)@rotation
        lattice = state['reference_indices']
    # Dense lookup only over this regular reference lattice; no state is modified.
    indices = lattice-lattice.min(axis=0)
    shape = indices.max(axis=0)+1
    lookup = np.full(tuple(shape), -1, dtype=np.int64)
    lookup[tuple(indices.T)] = np.arange(len(indices))
    left, right = lookup[:-1].ravel(), lookup[1:].ravel()
    valid = (left >= 0) & (right >= 0)
    left, right = left[valid], right[valid]
    rows = []
    for digit, plane in enumerate(goal['planes']):
        armed = [a for a in actions if a['cut_enabled'] and a['cut_index'] == digit]
        start, end = armed[0]['time_s'], armed[-1]['time_s']
        before = max(0, int(np.searchsorted(times, start, side='right')-1))
        after = min(len(times)-1, int(np.searchsorted(times, end)))
        x = plane['origin_local_m'][0]
        region = np.abs(reference[:, 0]-x) <= .010
        displacement = positions[:, region, 0]-reference[region, 0]
        final_digit = (codes[-1] >> (2*digit)) & 3
        a, b = final_digit[left], final_digit[right]
        conflict = ((a == 1) & (b == 2)) | ((a == 2) & (b == 1))
        midpoint = (reference[left[conflict], 0]+reference[right[conflict], 0])/2
        commanded = (np.asarray([a['edge_pose_world'][:3] for a in armed])-origin)@rotation
        rows.append({'digit': digit, 'declared_x_local_m': x, 'armed_time_s': [start, end],
                     'max_commanded_plane_error_m': float(np.max(np.abs(commanded[:, 0]-x))),
                     'near_plane_reference_region_m': [x-.010, x+.010],
                     'before_armed_sample_time_s': float(times[before]),
                     'before_armed_displacement_x_p05_p50_p95_m': np.percentile(displacement[before], [5, 50, 95]).tolist(),
                     'after_armed_sample_time_s': float(times[after]),
                     'after_armed_displacement_x_p05_p50_p95_m': np.percentile(displacement[after], [5, 50, 95]).tolist(),
                     'final_opposing_digit_x_neighbour_links': int(conflict.sum()),
                     'interface_reference_midpoint_x_range_m': [float(midpoint.min()), float(midpoint.max())] if len(midpoint) else None,
                     'interface_reference_midpoint_offset_median_m': float(np.median(midpoint)-x) if len(midpoint) else None})
    files = ['result.json', 'manifest.json', 'actions.jsonl', 'physics_state.npz']
    return {'sample_id': result['sample_id'], 'seed': result['seed'], 'variant': result['variant'],
            'source_dir': str(run.relative_to(ROOT)), 'sources_sha256': {n: sha(run/n) for n in files},
            'runtime_source': result['runtime_source'], 'material_seed': manifest['config']['material_seed'],
            'original_class': result['class'], 'original_evaluation': result['evaluation'],
            'physical_quality': result['physical_quality'], 'metrics': result['metrics'],
            'policy_done': result['policy_done'], 'original_episode_wall_s': result['runtime']['episode_wall_s'],
            'diagnostic_per_cut': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/wp12/phase5_001/kinematic_benchmark')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    started = time.perf_counter()
    paths = sorted(args.root.glob('P5_D_*_benchmark_s*/result.json'))
    census = []
    for path in paths:
        r = read(path)
        census.append({'sample_id': r['sample_id'], 'class': r['class'], 'metrics': r['metrics'],
                       'integrity': r['evaluation']['integrity']['status'],
                       'outcome': outcome_failures(r['evaluation'])})
    chosen = [next(p.parent for p in paths if f'_D_{rs}_' in p.parent.name) for rs in ('ID', 'OOD')]
    result = {'schema': 'rcs_wp12_family_d_diagnostic_v1', 'created': datetime.now().astimezone().isoformat(),
              'command': sys.argv, 'scope': 'READ_ONLY_DIAGNOSTIC; original benchmark verdicts unchanged',
              'selection': 'first ID + first OOD saved sample in declared seed order; census over all 20',
              'census': census, 'representatives': [diagnose(p) for p in chosen]}
    result['diagnostic_wall_s'] = time.perf_counter()-started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'out': str(args.out), 'samples': len(census), 'wall_s': result['diagnostic_wall_s']}))


if __name__ == '__main__':
    main()
