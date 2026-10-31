#!/usr/bin/env python3
"""WP-12 Phase 4a: read-only chip-region analysis of skim digits in a ``run_wp12_phase4a_diagnose.py`` run.

For each skim digit k (horizontal armed motion) the chip region is the reference material above the trim plane
(edge height minus the declared 0.2 mm clearance) whose x lies inside the armed skim span. Reported: unassigned
chip particles with their decisive ``cut_diagnostics`` class, closest approach to the edge (min f with the stroke
on), carry displacement, final f, s; direct code-0 bridges of the digit inside / outside the chip region; and the
diagnostic-only relabel test (unassigned chip particles get the chip digit 2) with the resulting component count.
Writes ``<run>/phase4a_chips_NNN.json``.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import cut_diagnostics as cd  # noqa: E402
from robotic_cutting_sim.cut_topology import mechanical_components, mechanical_components_packed  # noqa: E402


def components(codes, data, encoding, K):
    args = (data['stencil_nodes'], data['stencil_weights'], codes)
    if encoding == 'packed2':
        return mechanical_components_packed(*args, K, gradients=data['stencil_gradients'])
    return mechanical_components(*args, cuts=K, gradients=data['stencil_gradients'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('--clearance', type=float, default=2e-4)
    args = parser.parse_args()
    data = np.load(args.run/'wp12_phase4a_diag.npz')
    meta = json.loads((args.run/'wp12_phase4a_diag.json').read_text())['meta']
    encoding, K = meta['code_encoding'], int(meta['max_cuts'])
    X, Xf, codes = data['reference_local'].astype(float), data['final_positions'].astype(float), data['final_codes']
    masses = data['mass_kg']
    table = {k[len('diag_particle_'):]: data[k] for k in data.files if k.startswith('diag_particle_')}
    step_cut = data['step_cut']
    count, labels = components(codes, data, encoding, K)
    out = {'run': str(args.run), 'components': int(count),
           'fractions': (np.sort(np.bincount(labels, weights=masses))[::-1]/masses.sum())[:5].round(4).tolist(), 'digits': {}}
    for k in range(K):
        steps = np.flatnonzero(step_cut == k)
        if not len(steps):
            continue
        p0, p1 = data['step_pose_start_local'][steps[0], :3], data['step_pose_end_local'][steps[-1], :3]
        if abs(p1[2]-p0[2]) > 2e-3:      # not a (near-)horizontal skim
            continue
        R = Rotation.from_quat(data['step_pose_end_local'][steps[-1], 3:]).as_matrix()
        d = R[:, 0]
        plane = p1[2]-args.clearance
        lo, hi = sorted((p0[0], p1[0]))
        chip = (X[:, 2] > plane) & (X[:, 0] >= lo) & (X[:, 0] <= hi)
        digits = cd.digits_of(codes, k, encoding)
        un = chip & (digits == 0)
        info = {'plane_z_mm': round(plane*1e3, 3), 'span_x_mm': [round(lo*1e3, 2), round(hi*1e3, 2)],
                'rake_deg': float(np.degrees(np.arcsin(-d[2]))), 'chip_particles': int(chip.sum()),
                'chip_digit_counts': {int(a): int(b) for a, b in zip(*np.unique(digits[chip], return_counts=True))},
                'unassigned_chip': int(un.sum())}
        bridges = cd.direct_bridges(data['stencil_nodes'], data['stencil_weights'], data['stencil_gradients'], digits)
        info['direct_bridges'] = {'total': int(len(bridges)), 'in_chip_region': int(chip[bridges].sum())}
        if un.any():
            idx, classes = cd.classify_unassigned(table, k, un)
            o = data['step_pose_end_local'][steps[-1], :3].astype(float)
            info['unassigned_by_class'] = {c: int(np.count_nonzero(classes == j)) for j, c in enumerate(cd.UNASSIGNED_CLASSES)
                                           if np.count_nonzero(classes == j)}
            disp = Xf[un]-X[un]
            info['unassigned_detail'] = [{'reference_mm': (X[i]*1e3).round(2).tolist(),
                                          'min_f_stroke_mm': float(table['min_f_stroke_m'][i, k]*1e3),
                                          'min_f_armed_mm': float(table['min_f_armed_m'][i, k]*1e3),
                                          'carry_mm': float(np.linalg.norm(Xf[i]-X[i])*1e3),
                                          'final_f_mm': float((Xf[i]-o)@d*1e3), 'final_z_mm': float(Xf[i, 2]*1e3),
                                          'lead_rule_blocked_by_plow_substeps': int(table['lead_plow_blocked'][i, k]),
                                          'is_direct_bridge': bool(i in set(bridges.tolist())),
                                          'first_fail': dict(zip(cd.REASONS, table['first_fail'][i, k].tolist()))}
                                         for i in np.flatnonzero(un)[:12]]
            info['unassigned_carry_mm_median'] = float(np.median(np.linalg.norm(disp, axis=1))*1e3)
            new = codes.astype(np.int64).copy()
            if encoding == 'packed2':
                new[un] |= 2 << (2*k)
            else:
                new[un] += 2*3**k
            c, _ = components(new.astype(codes.dtype), data, encoding, K)
            info['relabel_unassigned_chip_as_chip_diagnostic_only'] = {'components': int(c)}
        # Chip particles that took the substrate digit (digit 1): where were they at assignment?
        sub = chip & (digits == 1)
        if sub.any():
            a_x = table['assign_x_m'][sub, k]
            info['chip_with_substrate_digit'] = {'count': int(sub.sum()),
                'median_assign_z_minus_reference_mm': float(np.median(a_x[:, 2]-X[sub, 2])*1e3),
                'median_assign_s_mm': float(np.median(table['assign_s_m'][sub, k])*1e3),
                'reference_layers_mm': {str(round(float(z)*1e3, 1)): int(c) for z, c in zip(*np.unique(X[sub, 2], return_counts=True))}}
        out['digits'][str(k)] = info
    n = 1
    while (args.run/f'phase4a_chips_{n:03d}.json').exists():
        n += 1
    (args.run/f'phase4a_chips_{n:03d}.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
