#!/usr/bin/env python3
"""Compare a tlmpm_v5 kinematic row with the tlmpm_v4 row of the same law/plan (ADR 0014 validation 1).

Reads two physics_state.npz files (v5: packed codes with ``code_encoding``; v4:
ternary) of the same task/plan and reports, at their common final time, the
number of particles whose decoded final codes differ, the final position
difference, and both final stencil-graph component partitions. GPU runs are not
bitwise reproducible (atomic accumulation order), so this is a within-GPU-
nondeterminism comparison, not an identity check. Read-only.

  .venv/bin/python scripts/compare_v5_v4_rows.py --v5 <dir> --v4 <dir> [--v4-repeat <dir>] --out <file.json>
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.cut_topology import mechanical_components, packed_to_ternary  # noqa: E402


def load(directory):
    path = Path(directory)
    data = np.load(path / 'physics_state.npz' if path.is_dir() else path)
    codes = data['codes'][-1]
    packed = 'code_encoding' in data.files and str(data['code_encoding']) == 'packed2'
    cuts = int(data['max_cuts']) if packed else (2 if int(data['codes'].max()) > 2 else 1)
    ternary = packed_to_ternary(codes, cuts) if packed else codes.astype(np.int64)
    count, labels = mechanical_components(data['reference_stencil_nodes'], data['reference_stencil_weights'], ternary,
                                          cuts, gradients=data['reference_stencil_gradients'])
    masses = data['mass_kg']
    fractions = np.sort(np.bincount(labels, weights=masses) / masses.sum())[::-1]
    return {'path': str(directory), 'packed': packed, 'cuts': cuts, 'time_s': float(np.asarray(data['time_s'])[-1]),
            'codes': ternary, 'positions': data['positions'][-1], 'components': int(count), 'labels': labels,
            'fractions': fractions, 'verdict': _verdict(path)}


def _verdict(path):
    result = path / 'result.json' if path.is_dir() else None
    if result is None or not result.exists():
        return None
    value = json.loads(result.read_text())
    return value.get('verdict', value.get('task_status'))


def compare(a, b):
    if a['time_s'] != b['time_s'] or a['codes'].shape != b['codes'].shape:
        raise ValueError(f"Rows differ in final time or particle count: {a['path']} vs {b['path']}")
    differ = a['codes'] != b['codes']
    dx = np.linalg.norm(a['positions'] - b['positions'], axis=1)
    pairs = set(zip(a['labels'].tolist(), b['labels'].tolist()))
    return {'final_time_s': a['time_s'], 'particles': int(len(dx)), 'final_codes_differing': int(differ.sum()),
            'final_labelled': [int(np.count_nonzero(a['codes'])), int(np.count_nonzero(b['codes']))],
            'final_position_difference_max_m': float(dx.max()), 'final_position_difference_p99_m': float(np.percentile(dx, 99)),
            'final_position_difference_mean_m': float(dx.mean()),
            'components': [a['components'], b['components']],
            'identical_component_partition': a['components'] == b['components'] == len(pairs),
            'component_mass_fractions': [a['fractions'][:4].tolist(), b['fractions'][:4].tolist()],
            'verdicts': [a['verdict'], b['verdict']]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v5', required=True)
    p.add_argument('--v4', required=True)
    p.add_argument('--v4-repeat', help='Optional second v4 run: GPU nondeterminism baseline')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; never overwrite')
    v5, v4 = load(a.v5), load(a.v4)
    report = {'command': sys.argv, 'v5': a.v5, 'v4': a.v4, 'v5_vs_v4': compare(v5, v4)}
    if a.v4_repeat:
        repeat = load(a.v4_repeat)
        report.update(v4_repeat=a.v4_repeat, v4_repeat_vs_v4=compare(repeat, v4), v5_vs_v4_repeat=compare(v5, repeat))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
