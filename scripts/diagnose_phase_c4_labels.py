#!/usr/bin/env python3
"""Read-only C4 label diagnosis: why reference above-plane material stays with the substrate.

For one run directory of ``scripts/run_phase_c4_trim_material.py`` this reports,
for reference particles initially above the target plane, how many ended with
each side digit, the first history frame where each got its digit, its height
relative to the edge at the frame before labelling (the band assigns the side
from the CURRENT position against the swept surface through the edge), and
where the attached (not removed) above-plane mass is. It also counts below-plane
particles labelled to the upper side. Oracle-only; nothing is rescored.

  .venv/bin/python scripts/diagnose_phase_c4_labels.py artifacts/wp11/phaseC4/k2_r00_skim_law8_001
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.cut_topology import mechanical_components  # noqa: E402
from robotic_cutting_sim.tasks import make_task  # noqa: E402


def _pct(values, q=(0, 10, 50, 90, 100)):
    values = np.asarray(values, float)
    return np.percentile(values, q).round(3).tolist() if len(values) else None


def _counts(values, decimals=1):
    keys, counts = np.unique(np.round(np.asarray(values, float), decimals), return_counts=True)
    return {f'{k:.{decimals}f}': int(c) for k, c in zip(keys, counts)}


def diagnose(run):
    run = Path(run)
    result = json.loads((run / 'result.json').read_text())
    task = make_task(result['task_id'])
    data = np.load(run / 'physics_state.npz')
    positions, codes, times = data['positions'], data['codes'], data['time_s']   # loaded once each
    masses = data['mass_kg']
    steps = [json.loads(line) for line in (run / 'actions.jsonl').open()]
    step_times = np.asarray([s['time_s'] for s in steps])
    edges = np.asarray([s['edge_pose_world'] for s in steps])
    reference = task.points_to_local(data['reference_positions'])
    target = task.target_origin_local_m[2]
    above = reference[:, 2] > target + 1e-9
    final = codes[-1]
    count, labels = mechanical_components(data['reference_stencil_nodes'], data['reference_stencil_weights'], final, 1,
                                          gradients=data['reference_stencil_gradients'])
    substrate_ids = result['metrics']['trim_separation']['metrics']['reference_substrate_component_ids']
    substrate = np.isin(labels, substrate_ids)
    first = np.where((codes != 0).any(0), np.argmax(codes != 0, axis=0), -1)

    def at_label(mask):
        """Height (mm) above the edge and local x relative to the edge at the frame before each particle's label."""
        idx = np.flatnonzero(mask & (first >= 0))
        if not len(idx):
            return None
        frame = np.maximum(first[idx] - 1, 0)
        p = task.points_to_local(positions[frame, idx])
        e = task.points_to_local(edges[np.clip(np.searchsorted(step_times, times[frame] - 1e-9), 0, len(edges) - 1), :3])
        return {'particles': int(len(idx)), 'height_above_edge_mm_pct': _pct((p[:, 2] - e[:, 2]) * 1e3),
                'x_relative_to_edge_mm_pct': _pct((p[:, 0] - e[:, 0]) * 1e3),
                'z_displacement_before_label_mm_pct': _pct((p[:, 2] - reference[idx, 2]) * 1e3),
                'label_time_s_pct': _pct(times[first[idx]])}

    groups = {}
    for name, mask in (('above_digit2_upper_side', above & (final == 2)), ('above_digit1_lower_side', above & (final == 1)),
                       ('above_code0', above & (final == 0))):
        groups[name] = {'particles': int(mask.sum()), 'mass_fraction_of_above_plane': float(masses[mask].sum() / masses[above].sum()),
                        'initial_local_z_mm': _counts(reference[mask, 2] * 1e3),
                        'initial_local_x_mm_hist': np.histogram(reference[mask, 0] * 1e3, bins=np.arange(-14, 15, 4))[0].tolist(),
                        'attached_to_substrate_fraction': float(substrate[mask].mean()) if mask.any() else None,
                        'at_label': at_label(mask)}
    attached = above & substrate
    below_upper = ~above & (final == 2)
    # Direct substrate depression: final height change of the top below-plane layer under the armed path.
    armed = task.points_to_local(edges[[s['cut_enabled'] for s in steps], :3])
    top = reference[~above, 2].max()
    layer = ~above & np.isclose(reference[:, 2], top) & (reference[:, 0] >= armed[:, 0].min()) \
        & (reference[:, 0] <= armed[:, 0].max()) & substrate
    sink = (task.points_to_local(positions[-1][layer])[:, 2] - reference[layer, 2]) * 1e3
    depression = {'top_layer_local_z_mm': float(top * 1e3), 'particles': int(layer.sum()),
                  'final_dz_mm_pct': _pct(sink), 'fraction_sunk_more_than_0p5mm': float(np.mean(sink < -.5)),
                  'scope': 'Final minus reference height of the substrate top particle layer under the armed edge '
                           'path (substrate component only); particle centres, not the frozen surface mesh'}
    return {'run': run.name, 'task_id': result['task_id'], 'verdict': result['verdict'],
            'substrate_top_layer_depression': depression,
            'x_bins_mm': np.arange(-14, 15, 4).tolist(), 'above_plane_groups': groups,
            'attached_above_plane': {'particles': int(attached.sum()),
                                     'mass_fraction_of_above_plane': float(masses[attached].sum() / masses[above].sum()),
                                     'digit_counts': _counts(final[attached], 0),
                                     'initial_local_z_mm': _counts(reference[attached, 2] * 1e3)},
            'below_plane_digit2_upper_side': {'particles': int(below_upper.sum()),
                                              'initial_local_z_mm': _counts(reference[below_upper, 2] * 1e3),
                                              'at_label': at_label(below_upper)},
            'edge_local_z_during_armed_mm': _pct(task.points_to_local(edges[[s['cut_enabled'] for s in steps], :3])[:, 2] * 1e3),
            'history_frame_spacing_s': float(np.median(np.diff(times))),
            'scope': 'Frame-sampled (history_stride) oracle diagnosis of recorded labels; never an observation'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', type=Path)
    a = p.parse_args()
    print(json.dumps([diagnose(run) for run in a.runs], indent=1))


if __name__ == '__main__':
    main()
