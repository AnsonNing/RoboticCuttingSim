#!/usr/bin/env python3
"""Explain why a recorded cut is (not) separated: actual MPM stencil bridges.

Reads a physics_state.npz (live, kinematic or replay) and reports the final
compatibility-graph component count, nodes shared by conflicting/unassigned
fields, and the code-zero particles that touch both cut sides. Oracle-only
diagnostic; it never changes a recording or a score.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.task_scenarios import configure_task
from robotic_cutting_sim.tasks import make_task


def stencil_nodes(points, dx):
    """Same reference-grid stencil construction as tlmpm/tlmpm_v2/tlmpm_v3."""
    origin = np.floor(points.min(0) / dx) * dx - 2 * dx
    dims = np.ceil((points.max(0) - origin) / dx).astype(int) + 3
    cells = np.floor((points - origin) / dx).astype(int)
    nodes = np.empty((len(points), 8), np.int64)
    for corner in range(8):
        bit = np.array([(corner >> j) & 1 for j in range(3)])
        xyz = cells + bit
        nodes[:, corner] = (xyz[:, 0] * dims[1] + xyz[:, 1]) * dims[2] + xyz[:, 2]
    return nodes


def analyze(path, task_id, scenario=None, frame=-1, limit=20):
    data = np.load(path)
    task = configure_task(make_task(task_id), scenario=scenario)
    points, _, _ = task.sample_particles()
    nodes = data['reference_stencil_nodes'] if 'reference_stencil_nodes' in data.files else stencil_nodes(points, task.grid_dx_m)
    codes = data['codes'][frame]
    count, _ = mechanical_components(nodes, np.ones(nodes.shape), codes, 1)
    has = {k: np.zeros(nodes.max() + 1, bool) for k in (0, 1, 2)}
    for k in has:
        has[k][nodes[codes == k].ravel()] = True
    zero = np.flatnonzero(codes == 0)
    direct = zero[has[2][nodes[zero]].any(1) & has[1][nodes[zero]].any(1)]
    report = dict(source=str(path), task_id=task_id, scenario=scenario, frame=frame, components=int(count),
                  code_counts={int(k): int(v) for k, v in zip(*np.unique(codes, return_counts=True))},
                  nodes_with_codes_0_2=int((has[0] & has[2]).sum()), nodes_with_codes_0_1=int((has[0] & has[1]).sum()),
                  nodes_with_codes_1_2=int((has[1] & has[2]).sum()), direct_code_zero_bridges=int(len(direct)),
                  bridge_local_xyz_mm=np.round(points[direct][:limit] * 1000, 1).tolist())
    if task.goal_kind == 'flatten':
        above = points[:, 2] > task.target_origin_local_m[2]
        report.update(above_plane_code_counts={int(k): int(v) for k, v in zip(*np.unique(codes[above], return_counts=True))},
                      above_plane_code_zero_local_xyz_mm=np.round(points[above & (codes == 0)][:limit] * 1000, 1).tolist())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('physics_state', type=Path)
    p.add_argument('--task', required=True)
    p.add_argument('--scenario')
    p.add_argument('--frame', type=int, default=-1)
    a = p.parse_args()
    print(json.dumps(analyze(a.physics_state, a.task, a.scenario, a.frame), indent=1))


if __name__ == '__main__':
    main()
