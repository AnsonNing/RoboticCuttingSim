#!/usr/bin/env python3
"""Explain why a recorded cut is (not) separated: actual MPM stencil bridges.

Reads a physics_state.npz (live, kinematic or replay) and reports the
compatibility-graph component count (actual 8-node stencil: recorded weights
and gradients when present, otherwise rebuilt exactly as the solver does), nodes shared by conflicting/unassigned
fields, and the code-zero particles that touch both cut sides. Oracle-only
diagnostic; it never changes a recording or a score. A tlmpm_v5 recording
(``code_encoding`` = 'packed2' and ``max_cuts`` in the npz, ADR 0014) is read with
the packed-code graph; ``--digit`` selects the cut digit for the bridge counts.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.cut_topology import decode_digits, mechanical_components, mechanical_components_packed
from robotic_cutting_sim.task_scenarios import configure_task
from robotic_cutting_sim.tasks import make_task


def stencil(points, dx):
    """Same reference-grid stencil construction as tlmpm/tlmpm_v2/tlmpm_v3/tlmpm_v4 ``__init__``."""
    origin = np.floor(points.min(0) / dx) * dx - 2 * dx
    dims = np.ceil((points.max(0) - origin) / dx).astype(int) + 3
    cells = np.floor((points - origin) / dx).astype(int)
    fraction = (points - origin) / dx - cells
    nodes = np.empty((len(points), 8), np.int64)
    weights = np.empty((len(points), 8), np.float32)
    gradients = np.empty((len(points), 8, 3), np.float32)
    for corner in range(8):
        bit = np.array([(corner >> j) & 1 for j in range(3)])
        xyz = cells + bit
        nodes[:, corner] = (xyz[:, 0] * dims[1] + xyz[:, 1]) * dims[2] + xyz[:, 2]
        f = np.where(bit, fraction, 1 - fraction)
        weights[:, corner] = f.prod(1)
        for axis in range(3):
            gradients[:, corner, axis] = (2 * bit[axis] - 1) * f[:, (axis + 1) % 3] * f[:, (axis + 2) % 3] / dx
    return nodes, weights, gradients


def stencil_nodes(points, dx):
    return stencil(points, dx)[0]


def analyze(path, task_id, scenario=None, frame=-1, limit=20, digit_index=0):
    data = np.load(path)
    task = configure_task(make_task(task_id), scenario=scenario)
    points, _, _ = task.sample_particles()
    recorded = [key in data.files for key in
                ('reference_stencil_nodes', 'reference_stencil_weights', 'reference_stencil_gradients')]
    if all(recorded):
        nodes, weights, gradients = (data[key] for key in
            ('reference_stencil_nodes', 'reference_stencil_weights', 'reference_stencil_gradients'))
        support = 'recorded_nodes_weights_gradients'
    else:
        nodes, weights, gradients = stencil(points, task.grid_dx_m)
        if recorded[0]:
            if not np.array_equal(np.asarray(data['reference_stencil_nodes'], np.int64), nodes):
                raise ValueError('Recorded stencil nodes differ from the reconstructed reference stencil')
            support = 'recorded_nodes_reconstructed_weights_gradients'
        else:
            support = 'reconstructed_nodes_weights_gradients'
    nodes = np.asarray(nodes, np.int64)
    if len(nodes) != data['codes'].shape[1] or len(points) != len(nodes):
        raise ValueError('Task particle sampling does not match the recording; pass the recorded task/scenario')
    # Same validity rule as mechanical_components: weight>0 or |gradient|>0.
    valid = (np.asarray(weights) > 0) | (np.linalg.norm(np.asarray(gradients), axis=-1) > 0)
    all_codes = data['codes']
    packed = 'code_encoding' in data.files and str(data['code_encoding']) == 'packed2'
    if packed:
        cuts = int(data['max_cuts'])
        components = mechanical_components_packed
    else:
        cuts = 2 if int(np.max(all_codes)) > 2 else 1
        components = mechanical_components
    if not 0 <= digit_index < cuts:
        raise ValueError(f'--digit must be in 0..{cuts - 1}')
    codes = all_codes[frame]
    count, labels = components(nodes, weights, codes, cuts, gradients=gradients)
    final_count, final_labels = (count, labels) if frame in (-1, len(all_codes) - 1) else \
        components(nodes, weights, all_codes[-1], cuts, gradients=gradients)
    masses = data['mass_kg'] if 'mass_kg' in data.files else np.ones(len(codes))
    fractions = np.sort(np.bincount(final_labels, weights=masses))[::-1] / masses.sum()
    if packed:
        digit = decode_digits(codes, cuts)[:, digit_index].astype(np.int64)
    else:
        digit = (codes // 3 ** digit_index) % 3  # first cut digit by default; same as codes for one retained cut
    has = {k: np.zeros(nodes.max() + 1, bool) for k in (0, 1, 2)}
    for k in has:
        has[k][nodes[digit == k][valid[digit == k]]] = True
    zero = np.flatnonzero(digit == 0)
    touches = lambda k: (has[k][nodes[zero]] & valid[zero]).any(1)
    direct = zero[touches(2) & touches(1)]
    report = dict(source=str(path), task_id=task_id, scenario=scenario, frame=frame, components=int(count),
                  final_components=int(final_count), final_component_mass_fractions=np.round(fractions[:6], 6).tolist(),
                  cuts_in_graph=cuts, code_encoding='packed2' if packed else 'ternary', bridge_digit=digit_index,
                  stencil_support=dict(source=support, valid_entry_rule='weight>0 or |gradient|>0',
                      valid_entries=int(valid.sum()), total_entries=int(valid.size)),
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
    p.add_argument('--digit', type=int, default=0, help='Cut digit for the bridge counts (default 0)')
    a = p.parse_args()
    print(json.dumps(analyze(a.physics_state, a.task, a.scenario, a.frame, digit_index=a.digit), indent=1))


if __name__ == '__main__':
    main()
