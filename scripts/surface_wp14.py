#!/usr/bin/env python3
"""Unchanged v1 reconstruction plus a bounded self-intersection measurement."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from run_wp13 import G


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--state', default='final_state.npz')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    command = [sys.executable, 'scripts/surface_wp13.py', '--run', str(a.run),
               '--out', str(a.out), '--state', a.state, '--device', a.device]
    subprocess.run(command, check=True)
    from robotic_cutting_sim import surface_metrics_v1 as S
    data = deepcopy(json.loads((a.out/'summary.json').read_text()))
    for row in data['profiles']:
        mesh = a.out/(row['profile']+'.npz')
        z = np.load(mesh)
        v, f = z['vertices'], z['faces']
        _, shell = S.shells(v, f)
        row['self_intersections'] = {'status': 'MEASURED_BOUNDED_PROXY',
            **S.self_intersections(v, f, max_triangles=512, seed=1401, shell=shell),
            'seed': 1401, 'scope': '512 sampled triangles, vertex-disjoint segment/triangle crossing; '
            'coplanar overlaps and unsampled defects are not ruled out. No mesh repair follows this measurement.'}
        row['mesh_sha256'] = G.sha(mesh)
    data['command'] = command
    G.write_new(a.out/'wp14_summary.json', data)


if __name__ == '__main__':
    main()
